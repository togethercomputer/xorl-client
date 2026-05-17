"""SampledSequence type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from typing_extensions import Literal

__all__ = ["SampledSequence", "StopReason"]

# Type alias matching tinker's StopReason
StopReason = Literal["length", "stop"]


@dataclass
class SampledSequence:
    """A single sampled sequence from the model."""

    tokens: List[int]
    stop_reason: StopReason = "stop"
    """Reason why sampling stopped: 'length' if max_tokens reached, 'stop' otherwise."""
    logprobs: Optional[List[float]] = None
    text: Optional[str] = None
    prompt_tokens: Optional[List[int]] = None
    """Token IDs for the prompt that produced this sequence, when returned by the backend."""
