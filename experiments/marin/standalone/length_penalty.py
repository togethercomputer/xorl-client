from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LengthPenaltyConfig:
    max_completion_tokens: int = 3584
    lpw: float = 1.0
    target_length: int = 768
    truncated_penalty: float = -2.0
    min_response_length: int = 16


def length_penalty_reward(
    *,
    correct: bool,
    completion_tokens: int,
    has_box: bool,
    truncated: bool,
    config: LengthPenaltyConfig = LengthPenaltyConfig(),
) -> float:
    """Return the SkyRL AIME reward used by the Marin reference run.

    With ``lpw=0`` this is the legacy ``+1/-1`` reward. With ``lpw>0``,
    wrong complete answers remain ``-1``, length-stopped responses are pushed
    toward ``truncated_penalty``, and correct complete answers receive the
    reference cosine length ramp.
    """

    if config.max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    if config.lpw == 0.0:
        return 1.0 if correct else -1.0
    if truncated:
        return -1.0 + config.lpw * (config.truncated_penalty - (-1.0))
    if not correct:
        return -1.0
    if config.max_completion_tokens <= config.target_length:
        return 1.0

    used = min(max(completion_tokens, 0), config.max_completion_tokens)
    if used < config.min_response_length:
        return 1.0 - config.lpw
    length_frac = (used - config.target_length) / float(config.max_completion_tokens - config.target_length)
    length_frac = min(1.0, max(0.0, length_frac))
    cosine_decay = (1.0 - math.cos(math.pi * length_frac)) / 2.0
    return 1.0 - config.lpw * cosine_decay


def shaped_reward(
    *,
    verifier_reward: float,
    correct: bool,
    completion_tokens: int,
    has_box: bool,
    truncated: bool,
    config: LengthPenaltyConfig = LengthPenaltyConfig(),
) -> float:
    _ = verifier_reward, has_box
    return length_penalty_reward(
        correct=correct,
        completion_tokens=completion_tokens,
        has_box=has_box,
        truncated=truncated,
        config=config,
    )
