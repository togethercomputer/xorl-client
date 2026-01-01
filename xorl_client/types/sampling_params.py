"""SamplingParams type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["SamplingParams"]


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.stop is not None:
            # Ensure stop contains only valid strings
            # SGLang's tokenizer.encode() will fail on non-string values
            valid_stops = [s for s in self.stop if isinstance(s, str) and s]
            if valid_stops:
                result["stop"] = valid_stops
        if self.stop_token_ids is not None:
            result["stop_token_ids"] = self.stop_token_ids
        return result
