"""SampleResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .sampled_sequence import SampledSequence

__all__ = ["SampleResponse"]


@dataclass
class SampleResponse:
    """Response from sampling operation."""

    sequences: List[SampledSequence]

    @property
    def text(self) -> str:
        """Get text from first sequence."""
        if self.sequences and self.sequences[0].text:
            return self.sequences[0].text
        return ""

    @property
    def tokens(self) -> List[int]:
        """Get tokens from first sequence."""
        if self.sequences:
            return self.sequences[0].tokens
        return []

    @property
    def logprobs(self) -> Optional[List[float]]:
        """Get logprobs from first sequence."""
        if self.sequences:
            return self.sequences[0].logprobs
        return None
