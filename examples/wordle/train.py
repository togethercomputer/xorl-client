"""Run endpoint-driven Wordle GRPO without managing endpoint deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import secrets
from typing import Any
from urllib.parse import urlsplit

import httpx

from xorl_client import SamplingClient, ServiceClient, types
from xorl_client.rl import learning_rate_at_step

from .artifacts import ArtifactStore, git_source_info, redact_url, terminal_audit
from .config import ExperimentConfig, load_config
from .rollout import GroupCoalescer, build_group_datums, rollout_complete_groups
from .task import WordleTask, file_sha256


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return None


def _normalize_forward_result(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in dict(getattr(result, "metrics", {}) or {}).items():
        number = _number(value)
        if number is not None:
            metrics[str(key)] = number
    outputs = getattr(result, "loss_fn_outputs", []) or []
    losses = [
        float(output.loss)
        for output in outputs
        if getattr(output, "loss", None) is not None
    ]
    k3 = [
        float(output.k3)
        for output in outputs
        if getattr(output, "k3", None) is not None
    ]
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    if k3:
        metrics["k3"] = sum(k3) / len(k3)
    return metrics


class XorlTrainerBackend:
    """Narrow typed adapter over public XoRL client operations."""

    def __init__(self, training_client, config: ExperimentConfig):
        self.client = training_client
        self.config = config

    async def forward_backward(self, datums: list[types.Datum]) -> dict[str, float]:
        result = await self.client.forward_backward(
            datums,
            self.config.trainer.loss_fn,
            loss_fn_params=self.config.trainer.effective_loss_fn_params(),
        )
        return _normalize_forward_result(result)

    async def optimizer_step(self, step: int) -> dict[str, float]:
        learning_rate = learning_rate_at_step(
            base_learning_rate=self.config.trainer.learning_rate,
            step=step,
            total_steps=self.config.trainer.steps,
            schedule=self.config.trainer.learning_rate_schedule,
            warmup_steps=self.config.trainer.learning_rate_warmup_steps,
            min_learning_rate=self.config.trainer.min_learning_rate,
        )
        result = await self.client.optim_step(
            types.AdamParams(
                learning_rate=learning_rate,
                beta1=self.config.trainer.beta1,
                beta2=self.config.trainer.beta2,
                eps=self.config.trainer.eps,
                weight_decay=self.config.trainer.weight_decay,
                grad_clip_norm=self.config.trainer.grad_clip_norm,
            )
        )
        return {str(key): float(value) for key, value in result.metrics.items()}

    async def sync(self) -> dict[str, Any]:
        result = await self.client.sync_weights_to_inference(
            sync_method=self.config.endpoints.sync_method,
        )
        return {
            "success": bool(result.success),
            "message": result.message,
            "transfer_time": float(result.transfer_time),
            "total_bytes": int(result.total_bytes),
        }

    async def checkpoint(self, name: str) -> str:
        result = await self.client.save_state(name)
        return str(result.path)

    async def restore(self, path: str) -> str:
        result = await self.client.load_state_with_optimizer(path)
        return str(result.path)


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
    return {"unavailable": f"{type(last_error).__name__}: {last_error}"}


def _validate_capabilities(config: ExperimentConfig, info: dict[str, dict]) -> None:
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
    if config.r3.required:
        for value in available:
            if value.get("r3") is False or value.get("routing_replay") is False:
                raise ValueError(
                    "R3 requested but endpoint capabilities reject routing replay"
                )


def _finite_metric(metrics: list[dict[str, float]], names: tuple[str, ...]) -> bool:
    values = [
        float(value)
        for record in metrics
        for key, value in record.items()
        if any(name in key.lower() for name in names)
    ]
    return bool(values) and all(math.isfinite(value) for value in values)


def _metric_leaf(key: str) -> str:
    return key.lower().replace(":", "/").rsplit("/", 1)[-1]


def _k3_max(metrics: list[dict[str, float]]) -> float | None:
    """Return one K3 value per forward result, preferring the tokenwise maximum."""

    values: list[float] = []
    for record in metrics:
        debug = [
            float(value)
            for key, value in record.items()
            if _metric_leaf(key).endswith("kl_k3_debug_max")
            and math.isfinite(float(value))
        ]
        summary = [
            float(value)
            for key, value in record.items()
            if _metric_leaf(key) in {"kl_sample_train_k3", "k3"}
            and math.isfinite(float(value))
        ]
        selected = debug or summary
        if not selected:
            return None
        values.append(max(abs(value) for value in selected))
    return max(values) if values else None


def _ratio_error_max(metrics: list[dict[str, float]]) -> float | None:
    """Return max distance from the identity ratio across every forward result."""

    values: list[float] = []
    for record in metrics:
        record_errors: list[float] = []
        for key, value in record.items():
            number = float(value)
            if not math.isfinite(number):
                continue
            leaf = _metric_leaf(key)
            if leaf.endswith("ratio_error"):
                record_errors.append(abs(number))
            elif leaf.endswith(("ratio_mean", "ratio_min", "ratio_max")):
                record_errors.append(abs(number - 1.0))
        if not record_errors:
            return None
        values.append(max(record_errors))
    return max(values) if values else None


class ExperimentRunner:
    def __init__(self, *, config, task, tokenizer, sampler, trainer, store, session_id):
        self.config: ExperimentConfig = config
        self.task: WordleTask = task
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.trainer = trainer
        self.store: ArtifactStore = store
        self.session_id = session_id

    async def run(self, *, start_step: int = 1) -> dict:
        records: list[dict] = self.store.load_step_records(before_step=start_step)
        for step in range(start_step, self.config.trainer.steps + 1):
            targets = self.task.select_train_targets(
                step=step,
                count=self.config.wordle.targets_per_step,
                seed=self.config.wordle.target_seed,
            )
            coalescer = GroupCoalescer(
                min_datums=self.config.streaming.min_datums_per_submission,
                max_groups=self.config.streaming.max_groups_per_submission,
                queue_max_groups=self.config.streaming.queue_max_groups,
            )
            forward_metrics: list[dict[str, float]] = []
            totals: dict[str, float] = {}

            async def submit(item) -> None:
                pending = coalescer.add(item)
                if pending is not None:
                    await submit_batch(pending)

            async def submit_batch(batch) -> None:
                datums, group_metrics = batch
                if not datums:
                    raise RuntimeError(
                        "complete groups produced no trainable assistant turns"
                    )
                forward_metrics.append(await self.trainer.forward_backward(datums))
                for key, value in group_metrics.items():
                    totals[key] = totals.get(key, 0.0) + value

            async def completed(group) -> None:
                await submit(
                    build_group_datums(group, r3_enabled=self.config.r3.enabled)
                )

            trajectories = await rollout_complete_groups(
                task=self.task,
                tokenizer=self.tokenizer,
                sampler=self.sampler,
                targets=targets,
                step=step,
                config=self.config,
                on_group_complete=completed,
            )
            tail = coalescer.flush()
            if tail is not None:
                await submit_batch(tail)
            finite_loss = _finite_metric(forward_metrics, ("loss",))
            if not self.config.correctness.require_finite_loss:
                finite_loss = True
            k3 = _k3_max(forward_metrics)
            ratio = _ratio_error_max(forward_metrics)
            gates = True
            if self.config.correctness.max_k3 is not None:
                gates &= k3 is not None and k3 <= self.config.correctness.max_k3
            if self.config.correctness.max_ratio_error is not None:
                gates &= (
                    ratio is not None
                    and ratio <= self.config.correctness.max_ratio_error
                )
            gate_record = {
                "step": step,
                "finite_loss": finite_loss,
                "k3": k3,
                "ratio_error": ratio,
                "correctness_gates_passed": gates,
                "evaluated_before_optimizer": True,
                "optimizer_step_requested_at_write": False,
            }
            self.store.write_preoptimizer_gate(step, gate_record)
            if not finite_loss or not gates:
                raise RuntimeError(
                    f"step {step} correctness failure: finite_loss={finite_loss}, "
                    f"k3={k3}, ratio={ratio}, gates={gates}; optimizer not requested"
                )
            optimizer_metrics = await self.trainer.optimizer_step(step)
            finite_gradient = _finite_metric([optimizer_metrics], ("grad",))
            if not self.config.correctness.require_finite_gradient:
                finite_gradient = True
            if not finite_gradient:
                raise RuntimeError(
                    f"step {step} optimizer returned a non-finite or missing gradient metric"
                )
            sync = await self.trainer.sync()
            if not sync.get("success"):
                raise RuntimeError(
                    f"step {step} sampler synchronization failed: {sync}"
                )
            checkpoint = None
            if (
                step % self.config.trainer.checkpoint_every == 0
                or step == self.config.trainer.steps
            ):
                checkpoint = await self.trainer.checkpoint(f"wordle-step-{step:08d}")
                self.store.append_checkpoint(
                    {
                        "step": step,
                        "path": checkpoint,
                        "model": self.config.model.model,
                        "model_id": self.config.model.model_id,
                        "session_id": self.session_id,
                        "optimizer": True,
                    }
                )
            trajectory_count = max(int(totals.get("trajectories", 0)), 1)
            metrics = {
                "step": step,
                "targets": targets,
                "trajectory_count": len(trajectories),
                "datum_count": int(totals.get("datums", 0)),
                "reward_mean": totals.get("reward_sum", 0.0) / trajectory_count,
                "exact_match_rate": totals.get("exact_sum", 0.0) / trajectory_count,
                "format_rate": totals.get("format_sum", 0.0) / trajectory_count,
                "valid_guess_rate": totals.get("valid_sum", 0.0) / trajectory_count,
                "forward_backward": forward_metrics,
                "optimizer": optimizer_metrics,
                "learning_rate": learning_rate_at_step(
                    base_learning_rate=self.config.trainer.learning_rate,
                    step=step,
                    total_steps=self.config.trainer.steps,
                    schedule=self.config.trainer.learning_rate_schedule,
                    warmup_steps=self.config.trainer.learning_rate_warmup_steps,
                    min_learning_rate=self.config.trainer.min_learning_rate,
                ),
                "optimizer_complete": True,
                "finite_loss": finite_loss,
                "finite_gradient": finite_gradient,
                "k3": k3,
                "ratio_error": ratio,
                "correctness_gates_passed": gates,
                "sync": sync,
                "final_sync": True,
                "checkpoint": checkpoint,
            }
            self.store.append_metrics(metrics)
            self.store.write_step(step, metrics)
            records.append(metrics)
        audit = terminal_audit(
            records=records,
            expected_steps=self.config.trainer.steps,
            checkpoint_every=self.config.trainer.checkpoint_every,
        )
        self.store.write_terminal_audit(audit)
        if not audit["success"]:
            raise RuntimeError(f"terminal audit failed: {audit}")
        return audit


def _endpoint_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"sync URL must include host and port: {redact_url(url)}")
    return parsed.hostname, parsed.port


async def _run_cli(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    endpoint_updates = {}
    if args.generation_url:
        endpoint_updates["generation_url"] = args.generation_url
    if args.sync_url:
        endpoint_updates["sync_urls"] = args.sync_url
    if endpoint_updates:
        config.endpoints = config.endpoints.model_copy(update=endpoint_updates)
    artifact_updates = {}
    if args.output_dir:
        artifact_updates["output_dir"] = args.output_dir
    if args.resume_from:
        artifact_updates["resume_from"] = args.resume_from
        artifact_updates.setdefault("output_dir", args.resume_from)
    if artifact_updates:
        config.artifacts = config.artifacts.model_copy(update=artifact_updates)
    if not config.endpoints.generation_url or not config.endpoints.sync_urls:
        raise ValueError("generation URL and at least one direct sync URL are required")
    resume = config.artifacts.resume_from is not None

    task = WordleTask(
        targets_path=config.wordle.targets_path,
        legal_guesses_path=config.wordle.legal_guesses_path,
        train_targets=config.wordle.train_targets,
        eval_targets=config.wordle.eval_targets,
        seed=config.wordle.target_seed,
        max_turns=config.wordle.max_turns,
    )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the Wordle example requires xorl-client[examples]") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer or config.model.model
    )

    capability_urls = [config.endpoints.generation_url, *config.endpoints.sync_urls]
    capability_values = await asyncio.gather(
        *(
            _server_info(url, min(config.generation.timeout, 30.0))
            for url in capability_urls
        )
    )
    capabilities = {
        redact_url(url): value
        for url, value in zip(capability_urls, capability_values, strict=True)
    }
    _validate_capabilities(config, capabilities)

    service = ServiceClient(
        base_url=args.trainer_url, timeout=config.generation.timeout
    )
    if config.model.mode == "lora":
        training_client = service.create_lora_training_client(
            base_model=config.model.model,
            rank=config.model.lora_rank,
            model_id=config.model.model_id,
        )
    else:
        training_client = service.create_training_client(
            base_model=config.model.model, model_id=config.model.model_id
        )
    for url in config.endpoints.sync_urls:
        host, port = _endpoint_host_port(url)
        response = training_client.add_inference_endpoint(
            host=host,
            port=port,
            world_size=config.endpoints.sync_world_size,
            sync_weights=not resume,
            buffer_size_mb=config.endpoints.sync_buffer_mb,
        ).result()
        if not response.success:
            raise RuntimeError(
                f"failed to register sync endpoint {redact_url(url)}: {response.message}"
            )
    sampler = SamplingClient(
        base_url=config.endpoints.generation_url,
        model_path=(config.model.model_id if config.model.mode == "lora" else ""),
        model=config.model.model,
        timeout=config.generation.timeout,
    )
    sampler.max_retries = config.generation.max_retries
    store = ArtifactStore(config.artifacts.output_dir)
    session_id = secrets.token_hex(12)
    source = git_source_info(config.artifacts.source_root)
    source.update(
        {
            "dataset_hashes": {
                "targets": file_sha256(config.wordle.targets_path),
                "legal_guesses": file_sha256(config.wordle.legal_guesses_path),
            },
            "endpoints": {
                "trainer": redact_url(args.trainer_url),
                "generation": redact_url(config.endpoints.generation_url),
                "sync": [redact_url(url) for url in config.endpoints.sync_urls],
            },
            "endpoint_capabilities": capabilities,
            "seeds": {
                "target": config.wordle.target_seed,
                "sampling": config.generation.sampling_seed,
            },
        }
    )
    public_config = config.model_dump(mode="json")
    public_config["endpoints"]["generation_url"] = redact_url(
        config.endpoints.generation_url
    )
    public_config["endpoints"]["sync_urls"] = [
        redact_url(url) for url in config.endpoints.sync_urls
    ]
    public_config["run_metadata"] = {
        "session_id": session_id,
        "parent_run": config.artifacts.resume_from,
    }
    store.initialize(run_config=public_config, source_info=source, resume=resume)
    trainer = XorlTrainerBackend(training_client, config)
    start_step = 1
    if resume:
        state = store.load_resume(
            model=config.model.model, model_id=config.model.model_id
        )
        await trainer.restore(state.checkpoint_path)
        restored_sync = await trainer.sync()
        if not restored_sync.get("success"):
            raise RuntimeError(
                f"failed to synchronize restored checkpoint to samplers: {restored_sync}"
            )
        start_step = state.step + 1
        session_id = state.session_id
    runner = ExperimentRunner(
        config=config,
        task=task,
        tokenizer=tokenizer,
        sampler=sampler,
        trainer=trainer,
        store=store,
        session_id=session_id,
    )
    return await runner.run(start_step=start_step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trainer-url", required=True)
    parser.add_argument("--generation-url")
    parser.add_argument("--sync-url", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    return parser


def main() -> None:
    result = asyncio.run(_run_cli(build_parser().parse_args()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
