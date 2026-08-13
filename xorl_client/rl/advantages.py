"""Grouped advantage calculation used by GRPO-style applications."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class GroupStatistics:
    """Population statistics for one advantage-normalization group."""

    count: int
    mean: float
    std: float


def compute_grpo_advantages(
    rewards: Sequence[float],
    *,
    group_ids: Sequence[Hashable],
    normalize: bool = True,
    std_normalization: bool = True,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> list[float] | tuple[list[float], dict[Hashable, GroupStatistics]]:
    """Return group-relative advantages in the original row order.

    The population standard deviation is intentional because every member of
    the complete rollout group is present. Zero-variance groups receive zero
    advantages.
    """

    if len(rewards) != len(group_ids):
        raise ValueError(
            f"rewards/group_ids cardinality mismatch: {len(rewards)} != {len(group_ids)}"
        )
    if not rewards:
        return ([], {}) if return_stats else []
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")

    grouped: dict[Hashable, list[float]] = defaultdict(list)
    values: list[float] = []
    for reward, group_id in zip(rewards, group_ids, strict=True):
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError(f"reward must be finite, got {reward!r}")
        values.append(value)
        grouped[group_id].append(value)

    stats: dict[Hashable, GroupStatistics] = {}
    for group_id, group_values in grouped.items():
        mean = sum(group_values) / len(group_values)
        variance = sum((value - mean) ** 2 for value in group_values) / len(
            group_values
        )
        stats[group_id] = GroupStatistics(
            count=len(group_values), mean=mean, std=math.sqrt(variance)
        )

    advantages: list[float] = []
    for reward, group_id in zip(values, group_ids, strict=True):
        group = stats[group_id]
        value = reward - group.mean if normalize else reward
        if std_normalization:
            value = 0.0 if group.std == 0.0 else value / (group.std + eps)
        advantages.append(float(value))
    return (advantages, stats) if return_stats else advantages
