"""XoRL adapter with complete-group streaming and one optimizer update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlsplit

from xorl_client import SamplingClient, SamplingParams, ServiceClient, types
from xorl_client.client.chunked_helpers import combine_fwd_bwd_output_results

from ..artifacts import redact_url
from ..config import ExperimentConfig
from ..metrics import (
    LogprobPair,
    compute_k3_metrics,
    numeric_metrics,
    require_complete_alignment,
    tensor_values,
)
from ..rollout import GroupCoalescer
from ..training import GroupTrainingBatch, TrainingSample, xorl_loss_inputs
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

_ROUTING_SPANS_SCHEMA = "xorl.r3.spans.v1"
_SGLANG_ROUTING_FILE_SCHEMA = "sglang.routed_experts.file.v1"


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def routing_rows(payload: Any) -> int:
    if not isinstance(payload, dict) or payload.get("schema") != _ROUTING_SPANS_SCHEMA:
        return 0
    value = payload.get("rows")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _routing_payload_bytes(payload: Any) -> int:
    if not isinstance(payload, dict) or payload.get("schema") != _ROUTING_SPANS_SCHEMA:
        return 0
    return sum(
        int(span.get("rows", 0)) * int(span.get("row_nbytes", 0))
        for span in payload.get("spans", [])
        if isinstance(span, dict)
    )


def _slice_spans(payload: Any, rows: int) -> list[dict[str, Any]] | None:
    if rows == 0:
        return []
    if rows < 0 or rows > routing_rows(payload):
        return None
    remaining = rows
    result: list[dict[str, Any]] = []
    for span in payload.get("spans", []):
        if remaining == 0:
            break
        if not isinstance(span, dict) or int(span.get("rows", -1)) < 0:
            return None
        take = min(remaining, int(span["rows"]))
        result.append({**span, "rows": take})
        remaining -= take
    return result if remaining == 0 else None


def _routing_payload_from_descriptor(
    descriptor: Any,
    *,
    previous: Any,
    prefix_rows: int,
    rows: int,
    field_name: str,
    dtype: str,
) -> dict[str, Any]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("schema") != _SGLANG_ROUTING_FILE_SCHEMA
    ):
        raise ValueError("Router Replay requires an SGLang packed routing descriptor")
    if descriptor.get("field") != field_name:
        raise ValueError(f"Router Replay descriptor field mismatch for {field_name}")
    fields = descriptor.get("fields")
    field = fields.get(field_name) if isinstance(fields, dict) else None
    shape = field.get("shape") if isinstance(field, dict) else None
    if (
        not isinstance(field, dict)
        or field.get("dtype") != dtype
        or not isinstance(shape, list)
        or len(shape) != 3
        or prefix_rows < 0
        or int(descriptor.get("start_row", -1)) != prefix_rows
        or rows < prefix_rows
    ):
        raise ValueError(f"invalid Router Replay packed descriptor for {field_name}")
    current_rows = rows - prefix_rows
    source_rows = int(shape[0])
    row_nbytes = int(shape[1]) * int(shape[2]) * 4
    if (
        source_rows < current_rows
        or source_rows < 0
        or int(field.get("nbytes", -1)) != source_rows * row_nbytes
        or int(field.get("offset", -1)) < 0
    ):
        raise ValueError(f"invalid Router Replay packed extent for {field_name}")
    spans = _slice_spans(previous, prefix_rows)
    if spans is None:
        if prefix_rows:
            raise ValueError(
                "Router Replay prefix is not backed by prior routing spans"
            )
        spans = []
    if current_rows:
        path = descriptor.get("path")
        error_path = descriptor.get("error_path")
        if not isinstance(path, str) or not path:
            raise ValueError("Router Replay descriptor path must be non-empty")
        if not isinstance(error_path, str) or not error_path:
            raise ValueError("Router Replay error path must be non-empty")
        spans.append(
            {
                "path": path,
                "error_path": error_path,
                "offset": int(field["offset"]),
                "source_row": 0,
                "rows": current_rows,
                "row_nbytes": row_nbytes,
                "source_shape": [int(value) for value in shape],
                "dtype": dtype,
            }
        )
    return {
        "schema": _ROUTING_SPANS_SCHEMA,
        "rows": rows,
        "shape": [rows, int(shape[1]), int(shape[2])],
        "dtype": dtype,
        "spans": spans,
    }


@dataclass
class _XorlStepState:
    coalescer: GroupCoalescer
    batches: list[tuple[list[TrainingSample], Any]] = field(default_factory=list)


class XorlBackend:
    name = "xorl"

    def __init__(
        self,
        *,
        training_client: Any,
        sampling_client: Any,
        tokenizer: Any,
        config: ExperimentConfig,
    ) -> None:
        self.client = training_client
        self.sampler = sampling_client
        self.tokenizer = tokenizer
        self.config = config
        backend = config.backends.xorl
        if backend is None:
            raise ValueError("configuration has no backends.xorl section")
        self.backend_config = backend

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

    async def sample_batch(
        self, requests: Sequence[SamplingRequest]
    ) -> list[SampledTurn]:
        replay = self.config.router_replay.enabled
        prefix_starts: list[int] = []
        params: list[SamplingParams] = []
        for request in requests:
            previous = request.previous_turn
            previous_input = (
                previous.prompt_tokens + previous.output_tokens
                if previous is not None and replay
                else []
            )
            if previous_input:
                previous_input = previous_input[:-1]
            prefix = _common_prefix_len(request.prompt.tokens, previous_input)
            prefix_starts.append(prefix)
            params.append(
                SamplingParams(
                    max_tokens=generation_budget(
                        self.config, len(request.prompt.tokens)
                    ),
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                    top_k=self.config.generation.top_k,
                    ignore_eos=self.config.generation.ignore_eos,
                    no_stop_trim=True,
                    sampling_seed=request.seed,
                    return_routed_experts=replay,
                    return_expert_logits=replay,
                    return_routed_experts_file=replay,
                    routed_experts_start_len=prefix,
                )
            )
        rows = await self.sampler.generate_batch_native_async(
            [request.prompt.tokens for request in requests],
            params,
            return_logprobs=True,
        )
        if len(rows) != len(requests):
            raise RuntimeError(
                "XoRL sampler cardinality mismatch: "
                f"submitted={len(requests)} returned={len(rows)}"
            )
        return [
            SampledTurn(
                prompt_tokens=list(request.prompt.tokens),
                output_tokens=list(row.tokens),
                logprobs=list(row.logprobs or []),
                text=row.text or "",
                backend_metadata=(
                    {
                        "raw": dict(row.meta_info or {}),
                        "prefix_rows": prefix,
                        "previous": (
                            request.previous_turn.backend_metadata
                            if request.previous_turn is not None
                            else None
                        ),
                    }
                    if replay
                    else None
                ),
            )
            for request, row, prefix in zip(requests, rows, prefix_starts, strict=True)
        ]

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
        if not isinstance(metadata, dict):
            raise ValueError("Router Replay metadata is missing")
        raw = metadata.get("raw")
        prefix_rows = int(metadata.get("prefix_rows", -1))
        previous = metadata.get("previous")
        rows = len(request.prompt.tokens) + retained_output_tokens - 1
        previous_experts = (
            previous.get("routed_experts") if isinstance(previous, dict) else None
        )
        previous_logits = (
            previous.get("routed_expert_logits") if isinstance(previous, dict) else None
        )
        result = {
            "routed_experts": _routing_payload_from_descriptor(
                raw.get("routed_experts") if isinstance(raw, dict) else None,
                previous=previous_experts,
                prefix_rows=prefix_rows,
                rows=rows,
                field_name="routed_experts",
                dtype="int32",
            ),
            "routed_expert_logits": _routing_payload_from_descriptor(
                raw.get("expert_logits") if isinstance(raw, dict) else None,
                previous=previous_logits,
                prefix_rows=prefix_rows,
                rows=rows,
                field_name="routed_expert_logits",
                dtype="float32",
            ),
        }
        if result["routed_experts"]["shape"] != result["routed_expert_logits"]["shape"]:
            raise ValueError(
                "Router Replay IDs and selected weights have different shapes"
            )
        payload_bytes = _routing_payload_bytes(
            result["routed_experts"]
        ) + _routing_payload_bytes(result["routed_expert_logits"])
        if payload_bytes > self.config.router_replay.max_payload_bytes:
            raise ValueError(
                "Router Replay payload exceeds max_payload_bytes: "
                f"{payload_bytes} > {self.config.router_replay.max_payload_bytes}"
            )
        return result

    async def begin_step(self, step_number: int) -> BackendStep:
        streaming = self.backend_config.streaming
        return BackendStep(
            number=step_number,
            state=_XorlStepState(
                coalescer=GroupCoalescer(
                    min_datums=streaming.min_datums_per_submission,
                    max_groups=streaming.max_groups_per_submission,
                    queue_max_groups=streaming.queue_max_groups,
                )
            ),
        )

    def _datum(self, sample: TrainingSample) -> types.Datum:
        metadata = sample.backend_metadata
        if self.config.router_replay.enabled:
            if not isinstance(metadata, dict):
                raise ValueError("Router Replay is enabled but a datum has no metadata")
            expected = sample.shifted_length
            for name in ("routed_experts", "routed_expert_logits"):
                if routing_rows(metadata.get(name)) != expected:
                    raise ValueError(
                        f"XoRL {name} rows do not align with shifted model input"
                    )
        else:
            metadata = {}
        return types.Datum(
            model_input=types.ModelInput.from_ints(sample.tokens[:-1]),
            loss_fn_inputs=xorl_loss_inputs(sample),
            routed_experts=metadata.get("routed_experts"),
            routed_expert_logits=metadata.get("routed_expert_logits"),
        )

    async def _submit(
        self, step: BackendStep, groups: Sequence[GroupTrainingBatch]
    ) -> None:
        samples = [sample for group in groups for sample in group.samples]
        if not samples:
            return
        datums = [self._datum(sample) for sample in samples]
        result = await self.client.forward_backward(
            datums,
            self.config.trainer.xorl_loss_name(),
            loss_fn_params=self.config.trainer.effective_loss_fn_params(backend="xorl"),
        )
        state: _XorlStepState = step.state
        state.batches.append((samples, result))
        step.submitted_datums += len(samples)
        step.forward_backward_calls += 1

    async def add_group(self, step: BackendStep, group: GroupTrainingBatch) -> None:
        if step.forward_complete:
            raise RuntimeError("cannot add a group after forward/backward is sealed")
        state: _XorlStepState = step.state
        ready = state.coalescer.add(group)
        if ready is not None:
            await self._submit(step, ready)

    async def finish_forward_backward(self, step: BackendStep) -> ForwardBackwardResult:
        if step.forward_complete:
            raise RuntimeError("forward/backward was already finished for this step")
        state: _XorlStepState = step.state
        tail = state.coalescer.flush()
        if tail is not None:
            await self._submit(step, tail)
        step.forward_complete = True
        if not state.batches:
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

        pairs: list[LogprobPair] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        returned_rows = 0
        returned_tokens = 0
        for samples, result in state.batches:
            outputs = list(getattr(result, "loss_fn_outputs", []) or [])
            returned_rows += len(outputs)
            if len(outputs) != len(samples):
                alignment = compute_k3_metrics(
                    pairs,
                    submitted_datums=step.submitted_datums,
                    returned_rows=returned_rows,
                    missing_datums=max(step.submitted_datums - returned_rows, 0),
                    mismatched_datums=abs(len(samples) - len(outputs)),
                    trainer_returned_tokens=returned_tokens,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                )
                require_complete_alignment(alignment)
            for sample, output in zip(samples, outputs, strict=True):
                row = tensor_values(getattr(output, "logprobs", None))
                returned_tokens += len(row)
                prompt_lengths.append(len(sample.prompt_tokens))
                response_lengths.append(sample.response_tokens)
                if len(row) != sample.shifted_length:
                    alignment = compute_k3_metrics(
                        pairs,
                        submitted_datums=step.submitted_datums,
                        returned_rows=returned_rows,
                        missing_datums=0,
                        mismatched_datums=1,
                        trainer_returned_tokens=returned_tokens,
                        prompt_lengths=prompt_lengths,
                        response_lengths=response_lengths,
                    )
                    require_complete_alignment(alignment)
                trainer = row[
                    sample.shifted_response_start : sample.shifted_response_end
                ]
                pairs.append(
                    LogprobPair(sampled=sample.sampled_logprobs, trainer=trainer)
                )
        combined = combine_fwd_bwd_output_results(
            [result for _, result in state.batches]
        )
        metrics = numeric_metrics(combined.metrics)
        alignment = compute_k3_metrics(
            pairs,
            submitted_datums=step.submitted_datums,
            returned_rows=returned_rows,
            missing_datums=max(step.submitted_datums - returned_rows, 0),
            mismatched_datums=0,
            trainer_returned_tokens=returned_tokens,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
        )
        require_complete_alignment(alignment)
        metrics.update(alignment)
        return ForwardBackwardResult(
            metrics=metrics,
            alignment=alignment,
            call_count=step.forward_backward_calls,
            submitted_datums=step.submitted_datums,
            returned_rows=returned_rows,
        )

    async def optimizer_step(
        self, step: BackendStep, *, learning_rate: float
    ) -> OptimizerResult:
        if not step.forward_complete or step.optimizer_requested:
            raise RuntimeError(
                "optimizer must run once after forward/backward completes"
            )
        step.optimizer_requested = True
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
        step.optimizer_complete = True
        return OptimizerResult(numeric_metrics(getattr(result, "metrics", {})))

    async def sync_samplers(self) -> PublishResult:
        """Sync trainer weights to the sampler fleet.

        With ``sync_pool_per_endpoint`` each registered endpoint pool syncs as
        its own pairwise NCCL group. One group spanning every endpoint is known
        to hang on multi-replica fleets, so world-2 per-pool syncs are the
        proven path (mirrors rl-bench's wordle_convergence harness).
        """
        if not self.backend_config.sync_pool_per_endpoint:
            result = await self.client.sync_weights_to_inference(
                sync_method=self.backend_config.sync_method
            )
            return PublishResult(
                success=bool(getattr(result, "success", False)),
                metrics={
                    "transfer_time": float(getattr(result, "transfer_time", 0.0)),
                    "total_bytes": float(getattr(result, "total_bytes", 0)),
                },
                detail=str(getattr(result, "message", "")),
            )
        total_time = 0.0
        total_bytes = 0.0
        for index in range(max(len(self.backend_config.sync_urls), 1)):
            result = await self.client.sync_weights_to_inference(
                sync_method=self.backend_config.sync_method,
                pools=[f"r{index}"],
                group_name=f"weight_sync_group_r{index}",
            )
            if not getattr(result, "success", False):
                return PublishResult(
                    success=False,
                    metrics={
                        "transfer_time": total_time,
                        "total_bytes": total_bytes,
                    },
                    detail=f"pool r{index}: {getattr(result, 'message', '')}",
                )
            total_time += float(getattr(result, "transfer_time", 0.0))
            total_bytes += float(getattr(result, "total_bytes", 0))
        return PublishResult(
            success=True,
            metrics={"transfer_time": total_time, "total_bytes": total_bytes},
        )

    async def publish_policy(self, step: BackendStep) -> PublishResult:
        if not step.optimizer_complete or step.policy_published:
            raise RuntimeError(
                "policy publication requires one completed optimizer update"
            )
        result = await self.sync_samplers()
        if result.success:
            step.policy_published = True
        return result

    async def checkpoint(self, step: BackendStep, *, name: str) -> CheckpointResult:
        if step.submitted_datums and not step.optimizer_complete:
            raise RuntimeError(
                "checkpoint requested away from a clean optimizer boundary"
            )
        result = await self.client.save_state(name)
        return CheckpointResult(path=str(result.path), optimizer=True)

    async def restore(self, path: str) -> None:
        await self.client.load_state_with_optimizer(path)

    async def close(self) -> None:
        return None


def _endpoint_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"sync URL must include host and port: {redact_url(url)!r}")
    return parsed.hostname, parsed.port


async def create_xorl_backend(
    config: ExperimentConfig, *, resume_checkpoint: str | None = None
) -> XorlBackend:
    backend = config.backends.xorl
    if backend is None:
        raise ValueError("configuration has no backends.xorl section")
    if not backend.trainer_url or not backend.generation_url or not backend.sync_urls:
        raise ValueError(
            "XoRL requires trainer_url, generation_url, and at least one sync_url"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the Wordle example requires xorl-client[examples]") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer or config.model.model
    )
    service = ServiceClient(
        base_url=backend.trainer_url, timeout=config.generation.timeout
    )
    if config.model.mode == "lora":
        training_client = service.create_lora_training_client(
            base_model=config.model.resolved_train_base_model(),
            rank=config.model.lora_rank,
            model_id=config.model.model_id,
            # XoRL owns LoRA initialization server-wide.  The Wordle seed is
            # meaningful for River/Tinker model construction, but forwarding it
            # here turns it into an unsupported per-session structure override.
            lora_seed=None,
            # The strict server manifest owns module selection as well.
            train_mlp=None,
            train_attn=None,
            train_unembed=None,
        )
    else:
        training_client = service.create_training_client(
            base_model=config.model.resolved_train_base_model(),
            model_id=config.model.model_id,
        )
    if resume_checkpoint:
        await training_client.load_state_with_optimizer(resume_checkpoint)
    for index, url in enumerate(backend.sync_urls):
        host, port = _endpoint_host_port(url)
        kwargs: dict[str, Any] = {}
        if backend.sync_pool_per_endpoint:
            # Distinct rendezvous port per endpoint: back-to-back registrations
            # on a shared port hit EADDRINUSE before the previous TCPStore is
            # released. Each pool then syncs as its own pairwise NCCL group.
            kwargs.update(
                pool=f"r{index}",
                group_name=f"weight_sync_group_r{index}",
                master_port=29600 + index,
            )
        registered = training_client.add_inference_endpoint(
            host=host,
            port=port,
            world_size=backend.sync_world_size,
            sync_weights=False,
            buffer_size_mb=backend.sync_buffer_mb,
            **kwargs,
        ).result()
        if not registered.success:
            raise RuntimeError(
                "failed to register XoRL sync endpoint "
                f"{redact_url(url)!r}: {registered.message}"
            )
    sampler = SamplingClient(
        base_url=backend.generation_url,
        # XoRL publishes merged weights into serving engines.  The training
        # session ID is not an SGLang LoRA adapter name and must not become
        # lora_path on generation requests.
        model_path="",
        model=config.model.model,
        timeout=config.generation.timeout,
    )
    sampler.max_retries = config.generation.max_retries
    result = XorlBackend(
        training_client=training_client,
        sampling_client=sampler,
        tokenizer=tokenizer,
        config=config,
    )
    initial_sync = await result.sync_samplers()
    if not initial_sync.success:
        state = "restored checkpoint" if resume_checkpoint else "initial policy"
        raise RuntimeError(
            f"failed to publish XoRL {state}: {initial_sync.detail}"
        )
    return result
