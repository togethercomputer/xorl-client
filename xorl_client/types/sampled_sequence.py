"""SampledSequence type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = ["SampledSequence"]


@dataclass
class SampledSequence:
    """A single sampled sequence from the model."""

    tokens: List[int]
    logprobs: Optional[List[float]] = None
    text: Optional[str] = None
