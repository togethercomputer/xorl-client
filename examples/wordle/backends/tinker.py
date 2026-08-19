"""Tinker adapter with strict full-step result pairing and optimizer resume."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..metrics import (
    LogprobPair,
    StructuralAlignmentError,
    compute_k3_metrics,
    numeric_metrics,
    require_complete_alignment,
    tensor_values,
)
from ..training import GroupTrainingBatch, TrainingSample, tinker_loss_inputs
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


@dataclass
class _TinkerStepState:
    groups: list[GroupTrainingBatch] = field(default_factory=list)


def _result(value: Any, timeout: float) -> Any:
    if hasattr(value, "result") and callable(value.result):
        try:
            return value.result(timeout=timeout)
        except TypeError:
            return value.result()
    return value


class TinkerBackend:
    name = "tinker"

    def __init__(
        self,
        *,
        training_client: Any,
        tinker_module: Any,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        self.client = training_client
        self.tinker = tinker_module
        self.tokenizer = tokenizer
        self.config = config
        backend = config.backends.tinker
        if backend is None:
            raise ValueError("configuration has no backends.tinker section")
        self.backend_config = backend
        self.sampling_client: Any = None
        self._sample_semaphore = asyncio.Semaphore(backend.sample_inflight)

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

    def _submit_sample(self, request: SamplingRequest) -> Any:
        budget = generation_budget(self.config, len(request.prompt.tokens))
        params = self.tinker.types.SamplingParams(
            max_tokens=budget,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            seed=request.seed,
        )
        return self.sampling_client.sample(
            prompt=self.tinker.types.ModelInput.from_ints(request.prompt.tokens),
            num_samples=1,
            sampling_params=params,
        )

    async def sample_batch(
        self, requests: Sequence[SamplingRequest]
    ) -> list[SampledTurn]:
        if self.sampling_client is None:
            raise RuntimeError(
                "Tinker sampling policy was not published before rollout"
            )

        async def sample_one(request: SamplingRequest) -> Any:
            async with self._sample_semaphore:
                future = await asyncio.to_thread(self._submit_sample, request)
                return await asyncio.to_thread(
                    _result, future, self.backend_config.sample_timeout
                )

        responses = await asyncio.gather(*(sample_one(request) for request in requests))
        result: list[SampledTurn] = []
        for request, response in zip(requests, responses, strict=True):
            sequences = list(getattr(response, "sequences", []) or [])
            if len(sequences) != 1:
                raise StructuralAlignmentError(
                    "Tinker sample cardinality mismatch: "
                    f"expected one sequence, returned={len(sequences)}"
                )
            sequence = sequences[0]
            output = [int(value) for value in sequence.tokens]
            result.append(
                SampledTurn(
                    prompt_tokens=list(request.prompt.tokens),
                    output_tokens=output,
                    logprobs=[float(value) for value in (sequence.logprobs or [])],
                    text=self.decode_tokens(output),
                    backend_metadata=None,
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
    ) -> None:
        return None

    async def _publish_sampling_client(self) -> None:
        self.sampling_client = await asyncio.to_thread(
            self.client.save_weights_and_get_sampling_client
        )

    async def begin_step(self, step_number: int) -> BackendStep:
        if self.sampling_client is None:
            await self._publish_sampling_client()
        return BackendStep(number=step_number, state=_TinkerStepState())

    async def add_group(self, step: BackendStep, group: GroupTrainingBatch) -> None:
        if step.forward_complete:
            raise RuntimeError("cannot add a group after forward/backward is sealed")
        step.state.groups.append(group)

    def _datum(self, sample: TrainingSample) -> Any:
        return self.tinker.types.Datum(
            model_input=self.tinker.types.ModelInput.from_ints(sample.tokens[:-1]),
            loss_fn_inputs=tinker_loss_inputs(sample),
        )

    def _forward_sync(self, data: list[Any]) -> Any:
        future = self.client.forward_backward(
            data,
            loss_fn=self.config.trainer.loss_fn,
            loss_fn_config=self.config.trainer.effective_loss_fn_params(
                backend="tinker"
            ),
        )
        return _result(future, self.backend_config.operation_timeout)

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
        data = [self._datum(sample) for sample in samples]
        result = await asyncio.to_thread(self._forward_sync, data)
        step.forward_backward_calls = 1
        outputs = list(getattr(result, "loss_fn_outputs", []) or [])
        returned_rows = len(outputs)
        if returned_rows != len(samples):
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
        for index, (sample, output) in enumerate(zip(samples, outputs, strict=True)):
            row_value = (
                output.get("logprobs")
                if isinstance(output, dict)
                else getattr(output, "logprobs", None)
            )
            row = tensor_values(row_value)
            returned_tokens += len(row)
            if len(row) != sample.shifted_length:
                raise StructuralAlignmentError(
                    "Tinker trainer logprob row does not align with its datum: "
                    f"index={index} expected={sample.shifted_length} returned={len(row)}"
                )
            pairs.append(
                LogprobPair(
                    sampled=sample.sampled_logprobs,
                    trainer=row[
                        sample.shifted_response_start : sample.shifted_response_end
                    ],
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
        metrics.update(alignment)
        return ForwardBackwardResult(
            metrics=metrics,
            alignment=alignment,
            call_count=1,
            submitted_datums=len(samples),
            returned_rows=returned_rows,
        )

    def _optimizer_sync(self, learning_rate: float) -> Any:
        future = self.client.optim_step(
            self.tinker.types.AdamParams(
                learning_rate=learning_rate,
                beta1=self.config.trainer.beta1,
                beta2=self.config.trainer.beta2,
                eps=self.config.trainer.eps,
                weight_decay=self.config.trainer.weight_decay,
                grad_clip_norm=self.config.trainer.grad_clip_norm,
            )
        )
        return _result(future, self.backend_config.operation_timeout)

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
        metrics = numeric_metrics(getattr(result, "metrics", {}))
        return OptimizerResult(
            metrics,
            gradient_metrics_reported=any("grad" in key.lower() for key in metrics),
        )

    async def publish_policy(self, step: BackendStep) -> PublishResult:
        if not step.optimizer_complete or step.policy_published:
            raise RuntimeError(
                "policy publication requires one completed optimizer update"
            )
        await self._publish_sampling_client()
        step.policy_published = True
        return PublishResult(success=True, detail="tinker_sampling_snapshot")

    def _checkpoint_sync(self, name: str) -> Any:
        future = self.client.save_state(name=name)
        return _result(future, self.backend_config.operation_timeout)

    async def checkpoint(self, step: BackendStep, *, name: str) -> CheckpointResult:
        if step.submitted_datums and not step.optimizer_complete:
            raise RuntimeError(
                "checkpoint requested away from a clean optimizer boundary"
            )
        result = await asyncio.to_thread(self._checkpoint_sync, name)
        path = str(getattr(result, "path", result))
        return CheckpointResult(path=path, optimizer=True)

    async def close(self) -> None:
        return None


async def create_tinker_backend(
    config: ExperimentConfig, *, resume_checkpoint: str | None = None
) -> TinkerBackend:
    backend = config.backends.tinker
    if backend is None:
        raise ValueError("configuration has no backends.tinker section")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for the Tinker backend")
    os.environ.setdefault("TINKER_TELEMETRY", "0")
    try:
        import tinker
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install xorl-client[examples] to use Tinker") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer or config.model.model
    )
    service = tinker.ServiceClient()
    metadata = {"run_name": backend.run_name, "task": "wordle"}
    if resume_checkpoint:
        training_client = (
            await service.create_training_client_from_state_with_optimizer_async(
                resume_checkpoint,
                user_metadata=metadata,
            )
        )
    else:
        training_client = await service.create_lora_training_client_async(
            base_model=backend.base_model or config.model.resolved_train_base_model(),
            rank=config.model.lora_rank,
            seed=config.model.lora_seed,
            train_mlp=config.model.train_mlp,
            train_attn=config.model.train_attn,
            train_unembed=config.model.train_unembed,
            user_metadata=metadata,
        )
    return TinkerBackend(
        training_client=training_client,
        tinker_module=tinker,
        tokenizer=tokenizer,
        config=config,
    )
