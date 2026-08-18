"""Backend protocol and backend-neutral operation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from ..config import BackendName, ExperimentConfig


def generation_budget(config: ExperimentConfig, prompt_tokens: int) -> int:
    """Return the legal response budget without silently crossing max_length."""

    remaining = int(config.generation.max_length) - int(prompt_tokens)
    if remaining <= 0:
        raise ValueError(
            "rendered Wordle prompt reaches or exceeds generation.max_length: "
            f"prompt_tokens={prompt_tokens} max_length={config.generation.max_length}"
        )
    return min(int(config.generation.max_new_tokens), remaining)


@dataclass(frozen=True)
class RenderedPrompt:
    tokens: list[int]
    text: str | None = None
    private_think_open: bool = False


def rendered_prompt(
    tokenizer: Any,
    tokens: Sequence[int],
    *,
    assume_private_think_open: bool,
) -> RenderedPrompt:
    values = [int(value) for value in tokens]
    try:
        text = str(tokenizer.decode(values, skip_special_tokens=False))
    except (AttributeError, KeyError, TypeError, ValueError):
        text = None
    private_think_open = assume_private_think_open
    if text is not None:
        lower = text.lower()
        private_think_open = private_think_open or (
            lower.rfind("<think>") > lower.rfind("</think>")
        )
    return RenderedPrompt(
        tokens=values,
        text=text,
        private_think_open=private_think_open,
    )


@dataclass
class SampledTurn:
    """Ordinary sampled turn shared by all backends.

    ``backend_metadata`` deliberately remains opaque. River routing handles,
    XoRL routing spans, and future backend metadata never translate through a
    common routing representation.
    """

    prompt_tokens: list[int]
    output_tokens: list[int]
    logprobs: list[float]
    text: str
    backend_metadata: Any = None
    trainable_output_tokens: int | None = None


@dataclass(frozen=True)
class SamplingRequest:
    prompt: RenderedPrompt
    seed: int
    turn: int
    previous_turn: SampledTurn | None = None


@dataclass
class BackendStep:
    number: int
    state: Any = None
    submitted_datums: int = 0
    forward_backward_calls: int = 0
    forward_complete: bool = False
    optimizer_requested: bool = False
    optimizer_complete: bool = False
    policy_published: bool = False


@dataclass(frozen=True)
class ForwardBackwardResult:
    metrics: dict[str, float]
    alignment: dict[str, float]
    call_count: int
    submitted_datums: int
    returned_rows: int


@dataclass(frozen=True)
class OptimizerResult:
    metrics: dict[str, float]
    gradient_metrics_reported: bool = True


@dataclass(frozen=True)
class PublishResult:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True)
class CheckpointResult:
    path: str
    optimizer: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: BackendName
    config: ExperimentConfig

    def render_prompt(
        self,
        *,
        task: Any,
        target: str,
        history: Sequence[tuple[str, str]],
    ) -> RenderedPrompt: ...

    def decode_tokens(self, tokens: Sequence[int]) -> str: ...

    async def sample_batch(
        self, requests: Sequence[SamplingRequest]
    ) -> list[SampledTurn]: ...

    def retain_backend_metadata(
        self,
        *,
        request: SamplingRequest,
        metadata: Any,
        original_output_tokens: int,
        retained_output_tokens: int,
    ) -> Any: ...

    async def begin_step(self, step_number: int) -> BackendStep: ...

    async def add_group(self, step: BackendStep, group: Any) -> None: ...

    async def finish_forward_backward(
        self, step: BackendStep
    ) -> ForwardBackwardResult: ...

    async def optimizer_step(
        self, step: BackendStep, *, learning_rate: float
    ) -> OptimizerResult: ...

    async def publish_policy(self, step: BackendStep) -> PublishResult: ...

    async def checkpoint(self, step: BackendStep, *, name: str) -> CheckpointResult: ...

    async def close(self) -> None: ...
