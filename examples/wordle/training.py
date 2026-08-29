"""Advantages, token retention, and the backend-neutral datum contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from xorl_client.rl import compute_grpo_advantages

from .config import ExperimentConfig


@dataclass(frozen=True)
class TrainingSample:
    group_id: str
    rollout_id: int
    turn: int
    prompt_tokens: list[int]
    output_tokens: list[int]
    sampled_logprobs: list[float]
    advantage: float
    reward: float
    backend_metadata: Any = None
    trainable_output_tokens: int | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_tokens + self.output_tokens

    @property
    def shifted_response_start(self) -> int:
        return len(self.prompt_tokens) - 1

    @property
    def shifted_length(self) -> int:
        return len(self.tokens) - 1

    @property
    def response_tokens(self) -> int:
        if self.trainable_output_tokens is None:
            return len(self.output_tokens)
        return int(self.trainable_output_tokens)

    @property
    def shifted_response_end(self) -> int:
        return self.shifted_response_start + self.response_tokens

    def validate(self) -> None:
        if not self.prompt_tokens:
            raise ValueError("training samples require at least one prompt token")
        if not self.output_tokens:
            raise ValueError("training samples require at least one response token")
        if len(self.output_tokens) != len(self.sampled_logprobs):
            raise ValueError(
                "retained response token/logprob mismatch: "
                f"{len(self.output_tokens)} != {len(self.sampled_logprobs)}"
            )
        if self.response_tokens < 1 or self.response_tokens > len(self.output_tokens):
            raise ValueError(
                "trainable response boundary is outside the sampled output: "
                f"{self.response_tokens} not in [1, {len(self.output_tokens)}]"
            )
        if not all(math.isfinite(float(value)) for value in self.sampled_logprobs):
            raise ValueError("sampled response logprobs must be finite")
        if not math.isfinite(float(self.advantage)):
            raise ValueError("advantage must be finite")


@dataclass
class GroupTrainingBatch:
    samples: list[TrainingSample]
    metrics: dict[str, float]
    trajectories: Sequence[Any] = field(default_factory=tuple)


def reward_key(config: ExperimentConfig) -> str:
    if config.wordle.reward == "exact_match":
        return "exact_match"
    if config.wordle.reward == "wordle":
        return "wordle_reward"
    return "reward"


def retain_training_tokens(
    group: Sequence[Any],
    *,
    config: ExperimentConfig,
    force_keep_zero_variance: bool = False,
) -> GroupTrainingBatch:
    """Apply group-relative advantages to every retained turn in the group.

    Rollout has already trimmed each sampled turn through its first completed
    Wordle action. This function revalidates the paired token/logprob boundary,
    skips zero-advantage trajectories unless replay metadata must be consumed,
    and produces the sole common datum shape from which backend wire requests
    are built.

    Under ``trainer.remove_constant_reward_groups`` a group whose rewards are
    all identical yields no samples (only its metrics), matching the matched-
    Tinker posture's group-level drop. ``force_keep_zero_variance`` overrides
    that for the keep-one fallback: when every group in a step is uniform, one
    group is still trained (with all-zero advantages) so the optimizer step
    happens instead of being skipped.
    """

    if not group:
        return GroupTrainingBatch(samples=[], metrics={"groups": 1.0})
    key = reward_key(config)
    rewards = [
        float(row.reward.get(key, row.reward.get("reward", 0.0))) for row in group
    ]
    group_ids = [row.group_id for row in group]
    advantages = compute_grpo_advantages(
        rewards,
        group_ids=group_ids,
        normalize=True,
        std_normalization=config.trainer.advantage_std_normalization,
    )
    zero_variance = all(value == rewards[0] for value in rewards)
    drop_group = (
        config.trainer.remove_constant_reward_groups
        and zero_variance
        and not force_keep_zero_variance
    )
    samples: list[TrainingSample] = []
    skipped_zero = 0
    retained_zero_replay = 0
    skipped_empty = 0
    truncated = 0
    for trajectory, advantage, trajectory_reward in zip(
        group, advantages, rewards, strict=True
    ):
        trajectory.advantage = float(advantage)
        if drop_group:
            continue
        if (
            abs(float(advantage)) <= 1e-12
            and config.trainer.skip_zero_advantage_trajectories
        ):
            if config.router_replay.enabled:
                retained_zero_replay += len(trajectory.turns)
            else:
                skipped_zero += len(trajectory.turns)
                continue
        for turn in trajectory.turns:
            sampled = turn.sample
            if not sampled.output_tokens:
                skipped_empty += 1
                continue
            item = TrainingSample(
                group_id=trajectory.group_id,
                rollout_id=trajectory.rollout_id,
                turn=turn.turn,
                prompt_tokens=list(sampled.prompt_tokens),
                output_tokens=list(sampled.output_tokens),
                sampled_logprobs=list(sampled.logprobs),
                advantage=float(advantage),
                reward=float(trajectory_reward),
                backend_metadata=sampled.backend_metadata,
                trainable_output_tokens=sampled.trainable_output_tokens,
            )
            item.validate()
            samples.append(item)
            truncated += int(turn.truncated_after_action)

    count = max(len(group), 1)
    metrics: dict[str, float] = {
        "groups": 1.0,
        "trajectories": float(len(group)),
        "datums": float(len(samples)),
        "reward_sum": float(sum(rewards)),
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "advantage_sum": float(sum(float(value) for value in advantages)),
        "advantage_abs_sum": float(sum(abs(float(value)) for value in advantages)),
        "skipped_zero_advantage_turns": float(skipped_zero),
        "retained_zero_advantage_replay_turns": float(retained_zero_replay),
        "skipped_empty_turns": float(skipped_empty),
        "truncated_after_action_turns": float(truncated),
        "sampled_response_tokens": float(sum(item.response_tokens for item in samples)),
        "submitted_response_tokens": float(
            sum(len(item.output_tokens) for item in samples)
        ),
        "prompt_tokens": float(sum(len(item.prompt_tokens) for item in samples)),
        "zero_variance_groups": float(zero_variance),
        "dropped_zero_variance_groups": float(drop_group),
    }
    score_keys = {
        key
        for trajectory in group
        for key, value in trajectory.reward.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for score_key in sorted(score_keys):
        metrics[f"{score_key}_sum"] = float(
            sum(float(trajectory.reward.get(score_key, 0.0)) for trajectory in group)
        )
    metrics["reward_mean"] = metrics["reward_sum"] / count
    return GroupTrainingBatch(samples=samples, metrics=metrics, trajectories=group)


def xorl_loss_inputs(sample: TrainingSample) -> dict[str, list[int] | list[float]]:
    """Next-token-shifted targets with ignored prompt positions."""

    sample.validate()
    targets = [-100] * sample.shifted_length
    logprobs = [0.0] * sample.shifted_length
    advantages = [0.0] * sample.shifted_length
    for offset, position in enumerate(
        range(sample.shifted_response_start, sample.shifted_response_end)
    ):
        targets[position] = sample.tokens[position + 1]
        logprobs[position] = sample.sampled_logprobs[offset]
        advantages[position] = sample.advantage
    return {
        "target_tokens": targets,
        "logprobs": logprobs,
        "advantages": advantages,
    }


def tinker_loss_inputs(sample: TrainingSample) -> dict[str, list[int] | list[float]]:
    """Tinker shifted targets; zero advantage masks prompt predictions."""

    values = xorl_loss_inputs(sample)
    values["target_tokens"] = list(sample.tokens[1:])
    return values


def river_loss_inputs(sample: TrainingSample) -> dict[str, list[Any]]:
    """River full-sequence prediction-position fields and response mask."""

    sample.validate()
    prompt_prefix = len(sample.prompt_tokens) - 1
    response = sample.response_tokens
    trailing = len(sample.output_tokens) - response
    return {
        "old_logprobs": [0.0] * prompt_prefix + list(sample.sampled_logprobs) + [0.0],
        "advantages": (
            [0.0] * prompt_prefix
            + [sample.advantage] * response
            + [0.0] * trailing
            + [0.0]
        ),
        "response_mask": (
            [False] * prompt_prefix + [True] * response + [False] * trailing + [False]
        ),
    }


def merge_metrics(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def merge_metric_diff(
    target: dict[str, float],
    new: dict[str, float],
    old: dict[str, float],
) -> None:
    """Replace an already-merged ``old`` contribution with ``new`` in place."""

    for key in set(new) | set(old):
        delta = float(new.get(key, 0.0)) - float(old.get(key, 0.0))
        if delta:
            target[key] = target.get(key, 0.0) + delta
