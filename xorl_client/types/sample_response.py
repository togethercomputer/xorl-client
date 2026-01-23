"""SampleResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .sampled_sequence import SampledSequence

__all__ = ["SampleResponse"]


@dataclass
class SampleResponse:
    """Response from sampling operation."""

    sequences: List[SampledSequence]

    prompt_logprobs: Optional[List[Optional[float]]] = None
    """If prompt_logprobs was set to true in the request, logprobs are computed for
    every token in the prompt. The `prompt_logprobs` response contains a float32
    value for every token in the prompt."""

    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None
    """If topk_prompt_logprobs was set to a positive integer k in the request,
    the top-k logprobs are computed for every token in the prompt. The
    `topk_prompt_logprobs` response contains, for every token in the prompt,
    a list of up to k (token_id, logprob) tuples."""

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
