"""Small client-side schedules for optimizer parameters."""

from __future__ import annotations

import math
from typing import Literal


def learning_rate_at_step(
    *,
    base_learning_rate: float,
    step: int,
    total_steps: int,
    schedule: Literal["constant", "cosine"] = "constant",
    warmup_steps: int = 0,
    min_learning_rate: float = 0.0,
) -> float:
    """Return the learning rate for a one-indexed optimizer step."""
    if step < 1 or total_steps < 1:
        raise ValueError("step and total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if not 0.0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be between zero and base_learning_rate")
    if schedule not in {"constant", "cosine"}:
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    if warmup_steps and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    if schedule == "constant":
        return base_learning_rate
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return min_learning_rate + 0.5 * (base_learning_rate - min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )
