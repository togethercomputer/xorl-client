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

    custom_params: Optional[Dict[str, Any]] = None
    """Backend-specific sampling controls passed through to the inference server."""

    return_routed_experts: bool = False
    """For R3 (Rollout Routing Replay) in MoE models."""

    return_expert_logits: bool = False
    """Return expert routing logits for R3 replay in MoE models."""

    seed: Optional[int] = None
    """Random seed for reproducible generation."""

    sampling_seed: Optional[int] = None
    """Per-request sampling seed for diverse generation across requests."""

    chat_continue_final_message: bool = False
    """For chat_completions api_format: if True, the request asks the backend to
    continue the trailing assistant message rather than open a new turn. Used by
    student-prefill OPD recipes (e.g. Run B's pause-prefilled student) where the
    student's last assistant message contains the prefill content and the model
    should generate from that point."""

    chat_template_kwargs: Optional[Dict[str, Any]] = None
    """For chat_completions api_format: forwarded to the backend's chat-template
    application (e.g. {"enable_thinking": False} to pin Qwen3.x to the closed
    think-block rendering). Without it the SERVER's template default decides —
    found 2026-06-10 to render assistant prefills inside an OPEN <think> block,
    silently flipping every sampled token's context (PTC-118 reproduction gap)."""

    chat_logprob_start_len: Optional[int] = None
    """For chat_completions api_format: logprob_start_len for the request. Set 0
    so the backend returns input_token_logprobs -> input_token_ids (the server's
    ACTUAL rendered prompt). Without it SGLang returns EMPTY input_token_ids and
    the client silently re-renders the prompt locally — the trained context can
    then mismatch the sampled context (same 2026-06-10 finding)."""

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
        if self.custom_params is not None:
            result["custom_params"] = self.custom_params
        if self.seed is not None:
            result["seed"] = self.seed
        if self.sampling_seed is not None:
            result["sampling_seed"] = self.sampling_seed
        return result
