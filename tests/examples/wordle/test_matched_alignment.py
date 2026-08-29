"""Matched-Tinker posture: advantages, group drops, overlay, comparison."""

import json
import math
from pathlib import Path

import pytest

from examples.wordle.compare_tinker import build_comparison, load_run
from examples.wordle.config import TrainerConfig, load_config
from examples.wordle.overlay import step_overlay
from examples.wordle.rollout import Trajectory, TurnRecord
from examples.wordle.backends.base import SampledTurn
from examples.wordle.training import (
    merge_metric_diff,
    retain_training_tokens,
)

ROOT = Path(__file__).parents[3]
MATCHED = ROOT / "examples/wordle/configs/tinker_matched.yaml"


def _turn(turn_index: int, *, outputs: int = 3, logprob: float = -0.5) -> TurnRecord:
    return TurnRecord(
        turn=turn_index,
        sample=SampledTurn(
            prompt_tokens=[1, 2, 3, 4],
            output_tokens=list(range(10, 10 + outputs)),
            logprobs=[logprob] * outputs,
            text="<guess>[ABIDE]</guess>",
            trainable_output_tokens=outputs,
        ),
        raw_text="<guess>[ABIDE]</guess>",
        truncated_after_action=False,
        guess="ABIDE",
        single_guess_tag=True,
        format_ok=True,
        strict_format_ok=True,
        valid_guess=True,
        public_constraint_valid=True,
        target_leak=False,
        extra_text=False,
        feedback="",
        solved=False,
    )


def _trajectory(
    group_id: str, rollout_id: int, reward: float, *, solved: bool = False
) -> Trajectory:
    return Trajectory(
        group_id=group_id,
        rollout_id=rollout_id,
        target="ABIDE",
        turns=[_turn(1)],
        terminal=True,
        solved=solved,
        stopped_reason="solved" if solved else "exhausted",
        reward={"reward": reward, "format": 1.0},
    )


def test_matched_config_loads_with_matched_posture():
    config = load_config(MATCHED)
    trainer = config.trainer
    assert trainer.loss_fn == "ppo"
    assert trainer.advantage_std_normalization is False
    assert trainer.skip_zero_advantage_trajectories is False
    assert trainer.remove_constant_reward_groups is True
    assert trainer.learning_rate_schedule == "constant"
    assert trainer.beta2 == pytest.approx(0.95)
    assert trainer.grad_clip_norm == pytest.approx(1.0)
    xorl = trainer.effective_loss_fn_params(backend="xorl")
    assert xorl["eps_clip"] == pytest.approx(0.2)
    assert xorl["eps_clip_high"] == pytest.approx(0.28)
    assert xorl["use_tis"] is False
    assert xorl["compute_kl_stats"] is True
    assert xorl["return_per_token"] is True
    tinker = trainer.effective_loss_fn_params(backend="tinker")
    assert tinker["clip_low_threshold"] == pytest.approx(0.8)
    assert tinker["clip_high_threshold"] == pytest.approx(1.28)
    for key in ("use_tis", "compute_kl_stats", "return_per_token"):
        assert key not in tinker
    assert config.backends.xorl.sync_pool_per_endpoint is True


def test_group_drop_requires_training_zero_advantage_members():
    with pytest.raises(ValueError, match="skip_zero_advantage_trajectories"):
        TrainerConfig(
            remove_constant_reward_groups=True,
            skip_zero_advantage_trajectories=True,
        )


def test_matched_retention_mean_centres_and_keeps_zero_advantage():
    config = load_config(MATCHED)
    group = [
        _trajectory("g0", 0, 1.0, solved=True),
        _trajectory("g0", 1, 0.0),
        _trajectory("g0", 2, 0.5),
    ]
    batch = retain_training_tokens(group, config=config)
    # Mean-centred, not std-normalised: 1.0 - 0.5 = 0.5 exactly.
    advantages = sorted(sample.advantage for sample in batch.samples)
    assert advantages == pytest.approx([-0.5, 0.0, 0.5])
    # The zero-advantage member is trained, not skipped.
    assert len(batch.samples) == 3
    assert batch.metrics["skipped_zero_advantage_turns"] == 0.0
    assert batch.metrics["dropped_zero_variance_groups"] == 0.0


def test_matched_retention_drops_uniform_group_and_force_keep_overrides():
    config = load_config(MATCHED)
    group = [_trajectory("g1", i, 1.0) for i in range(3)]
    dropped = retain_training_tokens(group, config=config)
    assert dropped.samples == []
    assert dropped.metrics["zero_variance_groups"] == 1.0
    assert dropped.metrics["dropped_zero_variance_groups"] == 1.0
    assert dropped.metrics["reward_mean"] == pytest.approx(1.0)

    kept = retain_training_tokens(
        group, config=config, force_keep_zero_variance=True
    )
    assert len(kept.samples) == 3
    assert all(sample.advantage == 0.0 for sample in kept.samples)
    assert kept.metrics["dropped_zero_variance_groups"] == 0.0

    totals: dict[str, float] = {}
    for key, value in dropped.metrics.items():
        totals[key] = totals.get(key, 0.0) + value
    merge_metric_diff(totals, kept.metrics, dropped.metrics)
    assert totals["datums"] == kept.metrics["datums"]
    assert totals["dropped_zero_variance_groups"] == 0.0
    assert totals["groups"] == 1.0


def test_default_config_preserves_grpo_std_and_zero_skip():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    assert config.trainer.advantage_std_normalization is True
    assert config.trainer.skip_zero_advantage_trajectories is True
    assert config.trainer.remove_constant_reward_groups is False
    group = [
        _trajectory("g0", 0, 1.0),
        _trajectory("g0", 1, 0.0),
        _trajectory("g0", 2, 0.5),
    ]
    batch = retain_training_tokens(group, config=config)
    # The mean-reward member has (near-)zero advantage and is skipped.
    assert len(batch.samples) == 2
    assert batch.metrics["skipped_zero_advantage_turns"] == 1.0


def test_step_overlay_maps_harness_names_and_kl_sign():
    config = load_config(MATCHED)
    trajectories = [
        _trajectory("g0", 0, 1.0, solved=True),
        _trajectory("g0", 1, 0.0),
        _trajectory("g1", 2, 1.0, solved=True),
        _trajectory("g1", 3, 1.0, solved=True),
    ]
    logratio_mean = 0.001
    sq_logratio_mean = 4e-6
    alignment = {
        "k3_valid_tokens": 12.0,
        "k3_mean": 2.5e-6,
        "logratio_mean": logratio_mean,
        "sq_logratio_mean": sq_logratio_mean,
        "abs_logratio_mean": 0.0015,
        "abs_logratio_p99": 0.004,
        "abs_logratio_max": 0.005,
    }
    overlay = step_overlay(
        config=config,
        step=7,
        trajectories=trajectories,
        totals={
            "groups": 2.0,
            "datums": 2.0,
            "sampled_response_tokens": 6.0,
            "dropped_zero_variance_groups": 1.0,
        },
        forward_metrics={"loss": 0.25},
        alignment=alignment,
        optimizer_metrics={"grad_norm": 0.75},
        learning_rate=1e-5,
        optimizer_skipped=False,
        rollout_wall_s=10.0,
        train_wall_s=2.0,
        sync_transfer_s=0.5,
        publish_wall_s=0.7,
        step_wall_s=13.0,
    )
    assert overlay["global_step"] == 7.0
    assert overlay["quality/train_reward"] == pytest.approx(0.75)
    assert overlay["quality/solve_rate"] == pytest.approx(0.75)
    assert overlay["env/all/total_episodes"] == 4.0
    assert overlay["env/all/turns_per_episode"] == pytest.approx(1.0)
    assert overlay["env/all/ac_tokens_per_turn"] == pytest.approx(3.0)
    assert overlay["env/all/ob_tokens_per_turn"] == pytest.approx(4.0)
    # g1 is uniform at reward 1.0 (all good); g0 is mixed.
    assert overlay["env/all/by_group/frac_all_good"] == pytest.approx(0.5)
    assert overlay["env/all/by_group/frac_all_bad"] == 0.0
    assert overlay["env/all/by_group/frac_mixed"] == pytest.approx(0.5)
    assert overlay["env/all/format"] == pytest.approx(1.0)
    assert overlay["optim/entropy"] == pytest.approx(0.5)
    assert overlay["optim/kl_sample_train_v1"] == pytest.approx(-logratio_mean)
    assert overlay["optim/kl_sample_train_v2"] == pytest.approx(
        0.5 * sq_logratio_mean
    )
    assert overlay["optim/kl_sample_train_v3"] == pytest.approx(2.5e-6)
    assert overlay["optim/kl_tokens"] == 12.0
    assert overlay["optim/grad_norm"] == pytest.approx(0.75)
    assert overlay["optim/lr"] == pytest.approx(1e-5)
    assert overlay["tokens/generated"] == 12.0
    assert overlay["tokens/trained"] == 6.0
    assert overlay["bench/total_groups"] == 2.0
    assert overlay["bench/kept_groups"] == 1.0
    assert overlay["bench/zero_variance_drop_rate"] == pytest.approx(0.5)
    assert overlay["bench/skipped_step"] == 0.0
    assert overlay["perf/generated_tokens_per_s"] == pytest.approx(1.2)
    assert overlay["perf/trained_tokens_per_s"] == pytest.approx(3.0)
    assert overlay["loss"] == pytest.approx(0.25)
    assert all(
        isinstance(value, float) and math.isfinite(value)
        for value in overlay.values()
    )


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_compare_tinker_report(tmp_path):
    xorl_dir = tmp_path / "xorl"
    tinker_dir = tmp_path / "tinker"
    _write_metrics(
        xorl_dir,
        [
            {
                "step": step,
                "overlay": {
                    "quality/train_reward": 0.1 * step,
                    "quality/solve_rate": 0.05 * step,
                    "optim/kl_sample_train_v3": 1e-6 * step,
                    "optim/grad_norm": 1.0,
                    "loss": -0.1,
                },
            }
            for step in (1, 2, 3)
        ],
    )
    # Legacy-shaped tinker run: no overlay block; nested forward metrics.
    _write_metrics(
        tinker_dir,
        [
            {
                "step": step,
                "reward_mean": 0.1 * step - 0.02,
                "forward_backward": {"k3_mean": 2e-6 * step, "loss": -0.11},
                "optimizer": {"grad_norm": 0.9},
            }
            for step in (1, 2)
        ],
    )
    comparison = build_comparison(
        load_run(xorl_dir), load_run(tinker_dir), last_n=2
    )
    assert comparison["summary"]["steps_compared"] == 2
    assert comparison["summary"]["xorl_steps"] == 3
    per_step = comparison["per_step"]
    assert [entry["step"] for entry in per_step] == [1, 2]
    assert per_step[0]["reward_delta"] == pytest.approx(0.02)
    assert per_step[1]["tinker"]["k3"] == pytest.approx(4e-6)
    assert per_step[1]["tinker"]["grad_norm"] == pytest.approx(0.9)
    assert comparison["summary"]["reward_gap_xorl_minus_tinker"] == pytest.approx(
        0.02
    )
