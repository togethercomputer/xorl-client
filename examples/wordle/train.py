"""Run the unified Wordle GRPO program on XoRL, River, or Tinker."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import secrets
from pathlib import Path
from typing import Any

import httpx

from .artifacts import (
    ArtifactStore,
    WandbSink,
    git_source_info,
    redact_url,
)
from .backends.river import create_river_backend
from .backends.tinker import create_tinker_backend
from .backends.xorl import create_xorl_backend
from .backends.base import (
    BackendStep,
    CheckpointResult,
    ForwardBackwardResult,
    OptimizerResult,
    PublishResult,
)
from .config import BackendName, ExperimentConfig, load_config
from .metrics import number, numeric_metrics
from .rollout import _LegacyXorlSamplerBackend, build_group_datums
from .runner import ExperimentRunner as _UnifiedExperimentRunner
from .task import WordleTask, file_sha256


async def _server_info(url: str, timeout: float) -> dict[str, Any]:
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ("/get_server_info", "/server_info"):
            try:
                response = await client.get(f"{url.rstrip('/')}{path}")
                response.raise_for_status()
                value = response.json()
                return value if isinstance(value, dict) else {"response": value}
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
    assert last_error is not None
    return {"unavailable": f"{type(last_error).__name__}: capability probe failed"}


def _validate_capabilities(
    config: ExperimentConfig, info: dict[str, dict], *, backend: BackendName = "xorl"
) -> None:
    """Fail only on explicitly reported incompatibilities."""

    available = [value for value in info.values() if "unavailable" not in value]
    if config.model.mode == "lora":
        for value in available:
            if value.get("enable_lora") is False:
                raise ValueError(
                    "requested LoRA mode but an endpoint reports enable_lora=false"
                )
    if config.preset == "zero_k3":
        for value in available:
            if value.get("batch_invariant") is False or value.get("zero_k3") is False:
                raise ValueError(
                    "zero_k3 requested but endpoint capabilities reject it"
                )
    if config.router_replay.enabled:
        if backend == "tinker":
            raise ValueError("Tinker does not support Router Replay")
        for value in available:
            if value.get("r3") is False or value.get("routing_replay") is False:
                raise ValueError(
                    "Router Replay requested but endpoint capabilities reject it"
                )


def _normalize_forward_result(result: Any) -> dict[str, float]:
    metrics = numeric_metrics(getattr(result, "metrics", {}))
    outputs = list(getattr(result, "loss_fn_outputs", []) or [])
    losses = [
        float(value)
        for output in outputs
        if (value := number(getattr(output, "loss", None))) is not None
    ]
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    return metrics


# Compatibility wrapper for callers of the original XoRL-only public example.
class XorlTrainerBackend:
    def __init__(self, training_client: Any, config: ExperimentConfig):
        self.client = training_client
        self.config = config

    async def forward_backward(self, datums: list[Any]) -> dict[str, float]:
        result = await self.client.forward_backward(
            datums,
            self.config.trainer.xorl_loss_name(),
            loss_fn_params=self.config.trainer.effective_loss_fn_params(backend="xorl"),
        )
        return _normalize_forward_result(result)


class _LegacyXorlBackend(_LegacyXorlSamplerBackend):
    """Adapter for callers of the original XoRL-only runner constructor."""

    def __init__(
        self, tokenizer: Any, sampler: Any, trainer: Any, config: ExperimentConfig
    ):
        super().__init__(tokenizer, sampler, config)
        self.trainer = trainer

    async def begin_step(self, step_number: int) -> BackendStep:
        return BackendStep(number=step_number, state=[])

    async def add_group(self, step: BackendStep, group: Any) -> None:
        r3_enabled = bool(self.config.r3 and self.config.r3.enabled)
        datums, _ = build_group_datums(group.trajectories, r3_enabled=r3_enabled)
        step.state.extend(datums)
        step.submitted_datums += len(datums)

    async def finish_forward_backward(self, step: BackendStep) -> ForwardBackwardResult:
        result = await self.trainer.forward_backward(step.state)
        metrics = (
            numeric_metrics(result)
            if isinstance(result, dict)
            else _normalize_forward_result(result)
        )
        step.forward_backward_calls = 1
        step.forward_complete = True
        count = step.submitted_datums
        alignment = {
            "submitted_datums": float(count),
            "returned_trainer_rows": float(count),
            "aligned_datums": float(count),
            "missing_datums": 0.0,
            "mismatched_datums": 0.0,
        }
        return ForwardBackwardResult(metrics, alignment, 1, count, count)

    async def optimizer_step(
        self, step: BackendStep, *, learning_rate: float
    ) -> OptimizerResult:
        del learning_rate
        step.optimizer_requested = True
        metrics = numeric_metrics(await self.trainer.optimizer_step(step.number))
        step.optimizer_complete = True
        return OptimizerResult(
            metrics, gradient_metrics_reported=any("grad" in key for key in metrics)
        )

    async def publish_policy(self, step: BackendStep) -> PublishResult:
        result = await self.trainer.sync()
        success = bool(result.get("success"))
        step.policy_published = success
        return PublishResult(success=success, metrics=numeric_metrics(result))

    async def checkpoint(self, step: BackendStep, *, name: str) -> CheckpointResult:
        return CheckpointResult(
            path=str(await self.trainer.checkpoint(name)), optimizer=True
        )

    async def close(self) -> None:
        return None


class ExperimentRunner(_UnifiedExperimentRunner):
    """Unified runner with a source-compatible legacy constructor."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        task: WordleTask,
        store: ArtifactStore,
        session_id: str,
        backend: Any = None,
        wandb: WandbSink | None = None,
        tokenizer: Any = None,
        sampler: Any = None,
        trainer: Any = None,
    ) -> None:
        if backend is None:
            if tokenizer is None or sampler is None or trainer is None:
                raise TypeError(
                    "legacy runner requires tokenizer, sampler, and trainer"
                )
            backend = _LegacyXorlBackend(tokenizer, sampler, trainer, config)
        super().__init__(
            config=config,
            task=task,
            backend=backend,
            store=store,
            session_id=session_id,
            wandb=wandb,
        )


def _metric_leaf(key: str) -> str:
    return key.lower().replace(":", "/").rsplit("/", 1)[-1]


def _k3_max(metrics: list[dict[str, float]]) -> float | None:
    values: list[float] = []
    for record in metrics:
        selected = [
            abs(float(value))
            for key, value in record.items()
            if _metric_leaf(key).endswith("kl_k3_debug_max")
            or _metric_leaf(key) in {"kl_sample_train_k3", "k3", "k3_max"}
        ]
        if not selected or not all(math.isfinite(value) for value in selected):
            return None
        values.append(max(selected))
    return max(values) if values else None


def _ratio_error_max(metrics: list[dict[str, float]]) -> float | None:
    values: list[float] = []
    for record in metrics:
        errors = []
        for key, value in record.items():
            parsed = float(value)
            if not math.isfinite(parsed):
                return None
            leaf = _metric_leaf(key)
            # ``abs_logratio_*`` is centered at zero and must not be treated
            # as an importance-ratio metric centered at one.
            if "logratio" in leaf:
                continue
            if leaf.endswith("ratio_error"):
                errors.append(abs(parsed))
            elif leaf.endswith(("ratio_mean", "ratio_min", "ratio_max")):
                errors.append(abs(parsed - 1.0))
        if not errors:
            return None
        values.append(max(errors))
    return max(values) if values else None


async def _create_backend(
    backend: BackendName,
    config: ExperimentConfig,
    *,
    checkpoint: str | None,
    resume_step: int,
):
    if backend == "xorl":
        return await create_xorl_backend(config, resume_checkpoint=checkpoint)
    if backend == "river":
        return await create_river_backend(
            config, resume_checkpoint=checkpoint, resume_step=resume_step
        )
    return await create_tinker_backend(config, resume_checkpoint=checkpoint)


def _apply_cli_overrides(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    artifact_updates: dict[str, Any] = {}
    if args.output_dir:
        artifact_updates["output_dir"] = args.output_dir
    if args.resume_from:
        artifact_updates.update(
            resume_from=args.resume_from,
            output_dir=args.resume_from,
        )
    if artifact_updates:
        config.artifacts = config.artifacts.model_copy(update=artifact_updates)
    if args.backend == "xorl":
        current = config.backends.xorl
        if current is None:
            raise ValueError("configuration has no backends.xorl section")
        updates: dict[str, Any] = {}
        if args.trainer_url:
            updates["trainer_url"] = args.trainer_url
        if args.generation_url:
            updates["generation_url"] = args.generation_url
        if args.sync_url:
            updates["sync_urls"] = args.sync_url
        if updates:
            config.backends = config.backends.model_copy(
                update={"xorl": current.model_copy(update=updates)}
            )
    return config


async def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    backend_name: BackendName = args.backend
    config = _apply_cli_overrides(load_config(args.config), args)
    config.validate_backend(backend_name)
    if backend_name == "xorl":
        xorl_config = config.backends.xorl
        assert xorl_config is not None
        if (
            not xorl_config.trainer_url
            or not xorl_config.generation_url
            or not xorl_config.sync_urls
        ):
            raise ValueError(
                "XoRL requires trainer_url, generation_url, and at least one sync_url"
            )
    task = WordleTask(
        train_targets_path=config.wordle.train_targets_path,
        eval_targets_path=config.wordle.eval_targets_path,
        legal_guesses_path=config.wordle.legal_guesses_path,
        train_targets=config.wordle.train_targets,
        eval_targets=config.wordle.eval_targets,
        max_turns=config.wordle.max_turns,
    )
    store = ArtifactStore(config.artifacts.output_dir)
    resume = bool(config.artifacts.resume_from)
    session_id = secrets.token_hex(12)
    public_config = config.model_dump(mode="json")
    public_config["selected_backend"] = backend_name
    public_config["run_metadata"] = {
        "session_id": session_id,
        "parent_run": config.artifacts.resume_from,
    }
    source = git_source_info(config.artifacts.source_root)
    endpoint_capabilities: dict[str, dict[str, Any]] = {}
    if backend_name == "xorl":
        xorl_config = config.backends.xorl
        assert xorl_config is not None
        capability_urls = [
            xorl_config.generation_url,
            *xorl_config.sync_urls,
        ]
        capability_values = await asyncio.gather(
            *(
                _server_info(url, min(config.generation.timeout, 30.0))
                for url in capability_urls
            )
        )
        endpoint_capabilities = {
            redact_url(url): value
            for url, value in zip(capability_urls, capability_values, strict=True)
        }
        _validate_capabilities(config, endpoint_capabilities, backend=backend_name)
    source.update(
        {
            "selected_backend": backend_name,
            "endpoint_capabilities": endpoint_capabilities,
            "dataset_hashes": dict(task.dataset_hashes),
            "seeds": {
                "target": config.wordle.target_seed,
                "sampling": config.generation.sampling_seed,
            },
            "implementation_hashes": {
                str(path.relative_to(Path(__file__).parent)): file_sha256(path)
                for path in sorted(
                    [
                        Path(__file__),
                        Path(__file__).with_name("artifacts.py"),
                        Path(__file__).with_name("config.py"),
                        Path(__file__).with_name("metrics.py"),
                        Path(__file__).with_name("reward.py"),
                        Path(__file__).with_name("rollout.py"),
                        Path(__file__).with_name("runner.py"),
                        Path(__file__).with_name("task.py"),
                        Path(__file__).with_name("training.py"),
                        *Path(__file__).with_name("backends").glob("*.py"),
                    ]
                )
            },
        }
    )
    xorl = public_config.get("backends", {}).get("xorl")
    if isinstance(xorl, dict):
        xorl["trainer_url"] = redact_url(xorl.get("trainer_url", ""))
        xorl["generation_url"] = redact_url(xorl.get("generation_url", ""))
        xorl["sync_urls"] = [redact_url(value) for value in xorl.get("sync_urls", [])]
    legacy_endpoints = public_config.get("endpoints")
    if isinstance(legacy_endpoints, dict):
        legacy_endpoints["generation_url"] = redact_url(
            legacy_endpoints.get("generation_url", "")
        )
        legacy_endpoints["sync_urls"] = [
            redact_url(value) for value in legacy_endpoints.get("sync_urls", [])
        ]
    river = public_config.get("backends", {}).get("river")
    if isinstance(river, dict):
        river["endpoint"] = redact_url(river.get("endpoint", ""))
    store.initialize(
        run_config=public_config,
        source_info=source,
        resume=resume,
    )
    checkpoint = None
    resume_step = 0
    if resume:
        state = store.load_resume(
            model=config.model.model, model_id=config.model.model_id
        )
        if state.backend and state.backend != backend_name:
            raise ValueError(
                f"resume backend mismatch: {state.backend!r} != {backend_name!r}"
            )
        checkpoint = state.checkpoint_path
        resume_step = state.step
        session_id = state.session_id
        public_config["run_metadata"]["session_id"] = session_id
    backend = await _create_backend(
        backend_name,
        config,
        checkpoint=checkpoint,
        resume_step=resume_step,
    )
    try:
        sink = WandbSink(config.wandb, run_config=public_config)
        runner = ExperimentRunner(
            config=config,
            task=task,
            backend=backend,
            store=store,
            session_id=session_id,
            wandb=sink,
        )
        return await runner.run(start_step=resume_step + 1)
    finally:
        await backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("xorl", "river", "tinker"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--trainer-url", help="override backends.xorl.trainer_url")
    parser.add_argument(
        "--generation-url", help="override backends.xorl.generation_url"
    )
    parser.add_argument(
        "--sync-url",
        action="append",
        default=[],
        help="replace backends.xorl.sync_urls; repeat for each direct sampler",
    )
    return parser


def main() -> None:
    result = asyncio.run(_run_cli(build_parser().parse_args()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
