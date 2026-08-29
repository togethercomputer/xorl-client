"""Tinker-namespace metric overlay for cross-stack convergence comparisons.

The runner's own step record keeps every existing metric name; this module adds
a flat superset of the metric names the rl-bench Wordle convergence harnesses
log (``quality/*``, ``env/all/*``, ``optim/*``, ``tokens/*``, ``bench/*``,
``perf/*`` on a shared ``global_step`` axis) so W&B curves from this example
overlay directly with harness runs on either stack.

Sign convention: the harness defines ``d = logp_sampler - logp_trainer`` with
``K1 = E[d]``, ``K2 = 0.5 E[d^2]`` and ``K3 = E[exp(-d) - 1 + d]``. This
example's alignment metrics use ``delta = logp_trainer - logp_sampler = -d``,
so ``K1 = -logratio_mean``, ``K2 = 0.5 * sq_logratio_mean`` and ``K3`` is
``k3_mean`` unchanged (the estimator is even in the pairing order used here).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .config import ExperimentConfig
from .training import reward_key

# Tinker buckets uniform-reward groups into good/bad at this threshold
# (metric_util._compute_by_group_metrics); mirrored for env/all/by_group/*.
BY_GROUP_GOOD_THRESH = 0.5


def _metric_leaf(key: str) -> str:
    return key.lower().replace(":", "/").rsplit("/", 1)[-1]


def _find_grad_norm(*records: Mapping[str, float]) -> float | None:
    for record in records:
        for key, value in record.items():
            if _metric_leaf(key) == "grad_norm":
                return float(value)
    for record in records:
        for key, value in record.items():
            leaf = _metric_leaf(key)
            if leaf.endswith("grad_norm") and "clip" not in leaf:
                return float(value)
    return None


def _trajectory_reward(trajectory: Any, key: str) -> float:
    reward = getattr(trajectory, "reward", {}) or {}
    return float(reward.get(key, reward.get("reward", 0.0)))


def step_overlay(
    *,
    config: ExperimentConfig,
    step: int,
    trajectories: Sequence[Any],
    totals: Mapping[str, float],
    forward_metrics: Mapping[str, float],
    alignment: Mapping[str, float],
    optimizer_metrics: Mapping[str, float],
    learning_rate: float,
    optimizer_skipped: bool,
    rollout_wall_s: float,
    train_wall_s: float,
    sync_transfer_s: float,
    publish_wall_s: float,
    step_wall_s: float,
) -> dict[str, float]:
    """Build the harness-named metric superset for one committed step."""

    key = reward_key(config)
    rewards: list[float] = []
    group_rewards: dict[str, list[float]] = defaultdict(list)
    solved = 0
    total_turns = 0
    ac_tokens = 0
    ob_tokens = 0
    generated = 0
    sampled_logprobs: list[float] = []
    score_sums: dict[str, float] = defaultdict(float)
    score_counts: dict[str, int] = defaultdict(int)

    for trajectory in trajectories:
        reward = _trajectory_reward(trajectory, key)
        rewards.append(reward)
        group_rewards[str(getattr(trajectory, "group_id", ""))].append(reward)
        solved += int(bool(getattr(trajectory, "solved", False)))
        for score_key, value in (getattr(trajectory, "reward", {}) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                score_sums[str(score_key)] += float(value)
                score_counts[str(score_key)] += 1
        for turn in getattr(trajectory, "turns", []):
            total_turns += 1
            trainable = int(turn.trainable_output_tokens)
            ac_tokens += trainable
            ob_tokens += len(turn.prompt_tokens)
            generated += len(turn.output_tokens)
            sampled_logprobs.extend(float(v) for v in turn.old_logprobs[:trainable])

    episodes = len(rewards)
    groups = list(group_rewards.values())
    uniform = [g for g in groups if g and all(v == g[0] for v in g)]
    n_good = sum(1 for g in uniform if g[0] >= BY_GROUP_GOOD_THRESH)
    n_bad = len(uniform) - n_good

    total_groups = float(totals.get("groups", len(groups)))
    dropped_groups = float(totals.get("dropped_zero_variance_groups", 0.0))
    trained_tokens = float(totals.get("sampled_response_tokens", 0.0))

    mean_reward = statistics.fmean(rewards) if rewards else 0.0
    metrics: dict[str, float] = {
        "global_step": float(step),
        "quality/train_reward": mean_reward,
        "quality/train_reward_std": (
            statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        ),
        "quality/solve_rate": solved / episodes if episodes else 0.0,
        "env/all/reward/total": mean_reward,
        "env/all/total_episodes": float(episodes),
        "env/all/total_turns": float(total_turns),
        "env/all/ac_tokens_per_turn": ac_tokens / total_turns if total_turns else 0.0,
        "env/all/ob_tokens_per_turn": ob_tokens / total_turns if total_turns else 0.0,
        "env/all/turns_per_episode": total_turns / episodes if episodes else 0.0,
        "env/all/by_group/frac_all_good": n_good / len(groups) if groups else 0.0,
        "env/all/by_group/frac_all_bad": n_bad / len(groups) if groups else 0.0,
        "env/all/by_group/frac_mixed": (
            (len(groups) - len(uniform)) / len(groups) if groups else 0.0
        ),
        "tokens/generated": float(generated),
        "tokens/trained": trained_tokens,
        "bench/total_groups": total_groups,
        "bench/kept_groups": total_groups - dropped_groups,
        "bench/dropped_zero_variance_groups": dropped_groups,
        "bench/zero_variance_drop_rate": (
            dropped_groups / total_groups if total_groups else 0.0
        ),
        "bench/all_groups_uniform": float(bool(groups) and len(uniform) == len(groups)),
        "bench/datums": float(totals.get("datums", 0.0)),
        "bench/skipped_step": float(optimizer_skipped),
        "optim/lr": float(learning_rate),
        "perf/rollout_s": float(rollout_wall_s),
        "perf/train_s": float(train_wall_s),
        "perf/sync_s": float(sync_transfer_s),
        "perf/adapter_publish_s": float(publish_wall_s),
        "perf/step_s": float(step_wall_s),
        "perf/generated_tokens_per_s": (
            generated / rollout_wall_s if rollout_wall_s > 0 else 0.0
        ),
        "perf/trained_tokens_per_s": (
            trained_tokens / train_wall_s if train_wall_s > 0 else 0.0
        ),
    }
    for score_key in sorted(score_sums):
        metrics[f"env/all/{score_key}"] = score_sums[score_key] / score_counts[score_key]
    if sampled_logprobs:
        metrics["optim/entropy"] = -statistics.fmean(sampled_logprobs)

    if float(alignment.get("k3_valid_tokens", 0.0)) > 0 and "k3_mean" in alignment:
        metrics.update(
            {
                "optim/kl_sample_train_v1": -float(alignment["logratio_mean"]),
                "optim/kl_sample_train_v2": 0.5 * float(alignment["sq_logratio_mean"]),
                "optim/kl_sample_train_v3": float(alignment["k3_mean"]),
                "optim/kl_abs_mean": float(alignment["abs_logratio_mean"]),
                "optim/kl_abs_p99": float(alignment["abs_logratio_p99"]),
                "optim/kl_abs_max": float(alignment["abs_logratio_max"]),
                "optim/kl_tokens": float(alignment["k3_valid_tokens"]),
            }
        )

    grad_norm = _find_grad_norm(optimizer_metrics, forward_metrics)
    if grad_norm is not None:
        metrics["optim/grad_norm"] = grad_norm
    if "loss" in forward_metrics:
        metrics["loss"] = float(forward_metrics["loss"])
    return metrics
