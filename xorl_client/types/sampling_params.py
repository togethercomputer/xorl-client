"""SamplingParams type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["SamplingParams"]


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    max_tokens: int = 128
    """Maximum number of tokens to generate."""

    temperature: float = 1.0
    """Sampling temperature."""

    top_p: float = 1.0
    """Nucleus sampling probability."""

    top_k: int = -1
    """Top-k sampling parameter (-1 for no limit)."""

    stop: Optional[List[str]] = None
    """Stop sequences for generation."""

    stop_token_ids: Optional[List[int]] = None
    """Stop token IDs for generation."""

    ignore_eos: bool = False
    """Continue generation until another stop condition or the token limit."""

    no_stop_trim: bool = False
    """Keep matched stop strings in SGLang's returned text and token IDs."""

    custom_params: Optional[Dict[str, Any]] = None
    """Backend-specific sampling controls passed through to the inference server."""

    return_routed_experts: bool = False
    """For R3 (Rollout Routing Replay) in MoE models."""

    return_expert_logits: bool = False
    """Return selected router weights alongside expert IDs for R3."""

    return_routed_experts_file: bool = False
    """Return packed shared-storage descriptors instead of base64 routing bodies."""

    routed_experts_start_len: int = 0
    """Skip an already stored routing prefix of this many rows."""

    seed: Optional[int] = None
    """Random seed for reproducible generation."""

    sampling_seed: Optional[int] = None
    """Per-request sampling seed for diverse generation across requests."""

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
        if self.ignore_eos:
            result["ignore_eos"] = True
        if self.no_stop_trim:
            result["no_stop_trim"] = True
        if self.custom_params is not None:
            result["custom_params"] = self.custom_params
        if self.seed is not None:
            result["seed"] = self.seed
        if self.sampling_seed is not None:
            result["sampling_seed"] = self.sampling_seed
        return result
