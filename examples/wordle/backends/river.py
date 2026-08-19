"""River adapter: one full-step request with opaque Router Replay handles."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..metrics import (
    LogprobPair,
    StructuralAlignmentError,
    compute_k3_metrics,
    numeric_metrics,
    require_complete_alignment,
)
from ..training import GroupTrainingBatch, TrainingSample, river_loss_inputs
from .base import (
    BackendStep,
    CheckpointResult,
    ForwardBackwardResult,
    OptimizerResult,
    PublishResult,
    RenderedPrompt,
    SampledTurn,
    SamplingRequest,
    generation_budget,
    rendered_prompt,
)

_MAX_FORWARD_BACKWARD_DATUMS = 8192


@dataclass
class _RiverStepState:
    groups: list[GroupTrainingBatch] = field(default_factory=list)


def _resolved(value: Any, timeout: float) -> Any:
    if (
        hasattr(value, "result")
        and callable(value.result)
        and not hasattr(value, "metrics")
    ):
        try:
            return value.result(timeout=timeout)
        except TypeError:
            return value.result()
    return value


class RiverBackend:
    name = "river"

    @property
    def preserve_full_replay_sequence(self) -> bool:
        return self.config.router_replay.enabled

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        session_context: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        backend = config.backends.river
        if backend is None:
            raise ValueError("configuration has no backends.river section")
        self.backend_config = backend
        self._session_context = session_context

    def render_prompt(self, *, task, target, history) -> RenderedPrompt:
        del target  # The hidden answer must never enter the policy prompt.
        messages = task.prompt_messages(history)
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": True,
            "return_dict": False,
        }
        try:
            tokens = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            kwargs.pop("return_dict")
            tokens = self.tokenizer.apply_chat_template(messages, **kwargs)
        return rendered_prompt(
            self.tokenizer,
            tokens,
            assume_private_think_open=True,
        )

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        return str(self.tokenizer.decode(tokens, skip_special_tokens=False))

    def _sample_sync(self, requests: Sequence[SamplingRequest]) -> list[Any]:
        if not requests:
            return []
        budgets = [
            generation_budget(self.config, len(request.prompt.tokens))
            for request in requests
        ]
        kwargs = {
            "prompt_token_ids": [request.prompt.tokens for request in requests],
            "num_samples": 1,
            # River accepts one budget for the whole request. The minimum keeps
            # every prompt under the common full-sequence max_length contract.
            "max_tokens": min(budgets),
            "temperature": self.config.generation.temperature,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k,
            "seed": requests[0].seed if requests else None,
            "return_prompt_logprobs": False,
            "return_expert_routing": self.config.router_replay.enabled,
            "timeout": self.backend_config.sample_timeout,
        }
        # A raw ``</guess>`` stop is unsafe while the prompt's private thinking
        # block is open. Generate through River's EOS/budget boundary, then let
        # the shared state-aware retention code select the public action.
        pending = self.model.submit_sample(**kwargs)
        groups = _resolved(pending, self.backend_config.sample_timeout)
        if len(groups) != len(requests):
            raise RuntimeError(
                "River sampler cardinality mismatch: "
                f"submitted={len(requests)} returned={len(groups)}"
            )
        return [group[0] if group else None for group in groups]

    async def sample_batch(
        self, requests: Sequence[SamplingRequest]
    ) -> list[SampledTurn]:
        size = self.backend_config.sample_batch_size
        chunks = [
            requests[start : start + size] for start in range(0, len(requests), size)
        ]
        chunked = await asyncio.gather(
            *(asyncio.to_thread(self._sample_sync, chunk) for chunk in chunks)
        )
        samples = [sample for chunk in chunked for sample in chunk]
        result: list[SampledTurn] = []
        for request, sample in zip(requests, samples, strict=True):
            if sample is None:
                raise RuntimeError("River returned an empty sample row")
            result.append(
                SampledTurn(
                    prompt_tokens=list(request.prompt.tokens),
                    output_tokens=[int(value) for value in sample.tokens],
                    logprobs=[float(value) for value in (sample.logprobs or [])],
                    text=str(sample.text or ""),
                    backend_metadata=getattr(sample, "expert_routing", None),
                )
            )
        return result

    def retain_backend_metadata(
        self,
        *,
        request: SamplingRequest,
        metadata: Any,
        original_output_tokens: int,
        retained_output_tokens: int,
    ) -> Any:
        if not self.config.router_replay.enabled:
            return None
        handle = getattr(metadata, "handle", "") if metadata is not None else ""
        if not handle:
            raise ValueError(
                "River Router Replay requires a routing handle on every datum"
            )
        if retained_output_tokens > original_output_tokens:
            raise ValueError("River Router Replay retention exceeds its routing handle")
        return {"expert_routing_handle": handle}

    async def begin_step(self, step_number: int) -> BackendStep:
        return BackendStep(number=step_number, state=_RiverStepState())

    async def add_group(self, step: BackendStep, group: GroupTrainingBatch) -> None:
        if step.forward_complete:
            raise RuntimeError("cannot add a group after forward/backward is sealed")
        step.state.groups.append(group)

    def _wire_datum(self, sample: TrainingSample) -> tuple[dict[str, Any], list[bool]]:
        loss = river_loss_inputs(sample)
        datum: dict[str, Any] = {
            "input_ids": list(sample.tokens),
            "attention_mask": [1] * len(sample.tokens),
            "old_logprobs": loss["old_logprobs"],
            "advantages": loss["advantages"],
        }
        if self.config.router_replay.enabled:
            metadata = sample.backend_metadata
            if not isinstance(metadata, dict) or not metadata.get(
                "expert_routing_handle"
            ):
                raise ValueError("River Router Replay metadata is incomplete")
            handle = metadata["expert_routing_handle"]
            datum["expert_routing_handle"] = (
                handle if isinstance(handle, bytes) else str(handle).encode("utf-8")
            )
        return datum, list(loss["response_mask"])

    def _forward_sync(self, data: list[dict[str, Any]]) -> Any:
        loss_config = self.config.trainer.effective_loss_fn_params(backend="river")
        loss_config["logprob_temperature"] = self.backend_config.logprob_temperature
        result = self.model.forward_backward(
            data,
            loss_fn=self.config.trainer.loss_fn,
            return_logprobs=True,
            zero_out=True,
            force_routing_replay=self.config.router_replay.enabled,
            compute_expert_flip_metric=False,
            timeout=self.backend_config.operation_timeout,
            **loss_config,
        )
        return _resolved(result, self.backend_config.operation_timeout)

    async def finish_forward_backward(self, step: BackendStep) -> ForwardBackwardResult:
        if step.forward_complete:
            raise RuntimeError("forward/backward was already finished for this step")
        samples = [sample for group in step.state.groups for sample in group.samples]
        step.submitted_datums = len(samples)
        step.forward_complete = True
        if not samples:
            alignment = compute_k3_metrics(
                [],
                submitted_datums=0,
                returned_rows=0,
                missing_datums=0,
                mismatched_datums=0,
                trainer_returned_tokens=0,
                prompt_lengths=[],
                response_lengths=[],
            )
            return ForwardBackwardResult(alignment, alignment, 0, 0, 0)
        if len(samples) > _MAX_FORWARD_BACKWARD_DATUMS:
            raise ValueError(
                "River's one-call step exceeds the 8192-datum request limit: "
                f"{len(samples)}"
            )
        wire_and_masks = [self._wire_datum(sample) for sample in samples]
        data = [value for value, _ in wire_and_masks]
        masks = [mask for _, mask in wire_and_masks]
        result = await asyncio.to_thread(self._forward_sync, data)
        step.forward_backward_calls = 1
        rows = getattr(result, "logprobs", None)
        returned_rows = 0 if rows is None else len(rows)
        if rows is None or returned_rows != len(samples):
            alignment = compute_k3_metrics(
                [],
                submitted_datums=len(samples),
                returned_rows=returned_rows,
                missing_datums=max(len(samples) - returned_rows, 0),
                mismatched_datums=abs(len(samples) - returned_rows),
                trainer_returned_tokens=0,
                prompt_lengths=[len(sample.prompt_tokens) for sample in samples],
                response_lengths=[sample.response_tokens for sample in samples],
            )
            require_complete_alignment(alignment)

        pairs: list[LogprobPair] = []
        returned_tokens = 0
        exact = 0
        padded = 0
        trimmed_padding = 0
        for index, (sample, datum, mask, row) in enumerate(
            zip(samples, data, masks, rows, strict=True)
        ):
            values = np.asarray(row, dtype=np.float64).reshape(-1)
            returned_tokens += values.size
            expected = len(datum["input_ids"])
            if values.size < expected:
                raise StructuralAlignmentError(
                    "River trainer logprob row is shorter than its submitted datum: "
                    f"index={index} input={expected} returned={values.size}"
                )
            if values.size == expected:
                exact += 1
            else:
                padded += 1
                trimmed_padding += int(values.size - expected)
            values = values[:expected]
            mask_array = np.asarray(mask, dtype=bool)
            sampled_full = np.asarray(
                river_loss_inputs(sample)["old_logprobs"], dtype=np.float64
            )
            pairs.append(
                LogprobPair(
                    sampled=sampled_full[mask_array].tolist(),
                    trainer=values[mask_array].tolist(),
                )
            )
        alignment = compute_k3_metrics(
            pairs,
            submitted_datums=len(samples),
            returned_rows=returned_rows,
            missing_datums=0,
            mismatched_datums=0,
            trainer_returned_tokens=returned_tokens,
            prompt_lengths=[len(sample.prompt_tokens) for sample in samples],
            response_lengths=[sample.response_tokens for sample in samples],
        )
        require_complete_alignment(alignment)
        metrics = numeric_metrics(getattr(result, "metrics", {}))
        if "loss" not in metrics and "policy_loss_sum" in metrics:
            metrics["loss"] = metrics["policy_loss_sum"] / len(samples)
        metrics.update(alignment)
        metrics.update(
            {
                "client_logprob_exact_length_datums": float(exact),
                "client_logprob_right_padded_datums": float(padded),
                "client_logprob_trimmed_padding_tokens": float(trimmed_padding),
                "client_logprob_requests": 1.0,
            }
        )
        return ForwardBackwardResult(
            metrics=metrics,
            alignment=alignment,
            call_count=1,
            submitted_datums=len(samples),
            returned_rows=returned_rows,
        )

    def _optimizer_sync(self, learning_rate: float) -> Any:
        clip = (
            None
            if self.config.trainer.grad_clip_norm <= 0
            else self.config.trainer.grad_clip_norm
        )
        result = self.model.optim_step(
            lr=learning_rate,
            beta1=self.config.trainer.beta1,
            beta2=self.config.trainer.beta2,
            eps=self.config.trainer.eps,
            weight_decay=self.config.trainer.weight_decay,
            grad_clip_norm=clip,
            timeout=self.backend_config.operation_timeout,
        )
        return _resolved(result, self.backend_config.operation_timeout)

    async def optimizer_step(
        self, step: BackendStep, *, learning_rate: float
    ) -> OptimizerResult:
        if not step.forward_complete or step.optimizer_requested:
            raise RuntimeError(
                "optimizer must run once after forward/backward completes"
            )
        step.optimizer_requested = True
        result = await asyncio.to_thread(self._optimizer_sync, learning_rate)
        step.optimizer_complete = True
        return OptimizerResult(numeric_metrics(getattr(result, "metrics", {})))

    async def publish_policy(self, step: BackendStep) -> PublishResult:
        if not step.optimizer_complete or step.policy_published:
            raise RuntimeError(
                "policy publication requires one completed optimizer update"
            )
        # River sampling reads the live in-memory policy; publication is implicit.
        step.policy_published = True
        return PublishResult(success=True, detail="river_live_policy")

    def _checkpoint_sync(self, name: str) -> Any:
        return self.model.save_weights(
            name, mode="training", timeout=self.backend_config.operation_timeout
        )

    async def checkpoint(self, step: BackendStep, *, name: str) -> CheckpointResult:
        if step.submitted_datums and not step.optimizer_complete:
            raise RuntimeError(
                "checkpoint requested away from a clean optimizer boundary"
            )
        result = await asyncio.to_thread(self._checkpoint_sync, name)
        path = str(getattr(result, "path", result))
        return CheckpointResult(path=path, optimizer=True)

    async def close(self) -> None:
        if self._session_context is not None:
            await asyncio.to_thread(self._session_context.__exit__, None, None, None)
            self._session_context = None


async def create_river_backend(
    config: ExperimentConfig,
    *,
    resume_checkpoint: str | None = None,
    resume_step: int = 0,
) -> RiverBackend:
    backend = config.backends.river
    if backend is None:
        raise ValueError("configuration has no backends.river section")
    api_key = os.environ.get("RIVER_API_KEY")
    if not api_key:
        raise RuntimeError("RIVER_API_KEY is required for the River backend")
    try:
        import river_client as river
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install river-client to use --backend river") from exc
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the Wordle example requires xorl-client[examples]") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer or config.model.model
    )
    client = river.Client(api_key=api_key, endpoint=backend.endpoint)
    context = client.session(project=backend.project, run=backend.run_name)
    session = await asyncio.to_thread(context.__enter__)
    try:
        checkpoint = (
            river.Checkpoint(
                path=resume_checkpoint,
                step=resume_step,
                checkpoint_type="training",
            )
            if resume_checkpoint
            else None
        )
        model = await asyncio.to_thread(
            session.create_model,
            base_model=backend.base_model or config.model.resolved_train_base_model(),
            lora=river.LoraConfig(
                rank=config.model.lora_rank,
                train_attn=config.model.train_attn,
                train_mlp=config.model.train_mlp,
                train_unembed=config.model.train_unembed,
                seed=config.model.lora_seed,
            ),
            checkpoint=checkpoint,
            timeout=backend.operation_timeout,
        )
    except BaseException:
        await asyncio.to_thread(context.__exit__, None, None, None)
        raise
    return RiverBackend(
        model=model,
        tokenizer=tokenizer,
        config=config,
        session_context=context,
    )
