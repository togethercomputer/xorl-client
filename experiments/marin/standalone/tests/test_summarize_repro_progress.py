from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.marin.standalone.summarize_repro_progress import _read_reference, summarize


def _write_reference(path: Path, rows: int = 2, step_rewards: list[tuple[int, float]] | None = None) -> None:
    fields = [
        "trainer/global_step",
        "loss/avg_final_rewards",
        "reward/avg_pass_at_16",
        "timing/step",
        "timing/generate",
        "timing/policy_train",
        "generate/avg_num_tokens",
    ]
    if step_rewards is None:
        step_rewards = [(idx + 1, -0.1) for idx in range(rows)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for global_step, reward in step_rewards:
            writer.writerow(
                {
                    "trainer/global_step": str(global_step),
                    "loss/avg_final_rewards": str(reward),
                    "reward/avg_pass_at_16": "0.5",
                    "timing/step": "1.0",
                    "timing/generate": "1.0",
                    "timing/policy_train": "1.0",
                    "generate/avg_num_tokens": "100",
                }
            )


def _write_metrics(path: Path, *, same_policy: bool) -> None:
    train_version = "policy-000000" if same_policy else "policy-000001"
    rows = [
        {
            "event": "step_rollout_summary",
            "step": 0,
            "mean_reward": -0.2,
            "pass_at_16": 0.6,
            "truncated_fraction": 0.0,
            "mean_completion_tokens": 100,
            "total_tokens_per_s": 1000,
            "policy_weight_version_at_sampling": "policy-000000",
            "policy_weight_version_at_train": train_version,
        },
        {
            "event": "train_update",
            "step": 0,
            "behavior_k3": 2e-6,
            "derived_train_metrics": {"valid_tokens_per_s": 2000},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _args(metrics_path: Path, reference_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        metrics_jsonl=metrics_path,
        reference_csv=reference_path,
        tail=8,
        max_k3=1e-6,
    )


def test_summarize_repro_progress_fails_same_policy_k3(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    reference_path = tmp_path / "reference.csv"
    _write_reference(reference_path)
    _write_metrics(metrics_path, same_policy=True)

    assert summarize(_args(metrics_path, reference_path)) == 1
    out = capsys.readouterr().out
    assert "policy_lag behavior_k3" in out
    assert "WARNING: latest same-policy behavior_k3" in out


def test_summarize_repro_progress_notes_lagged_k3(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    reference_path = tmp_path / "reference.csv"
    _write_reference(reference_path)
    _write_metrics(metrics_path, same_policy=False)

    assert summarize(_args(metrics_path, reference_path)) == 0
    out = capsys.readouterr().out
    assert "0 -0.200 -0.100 -0.100 0.600 0.500 0.000 100 1000 2000 1 2.00e-06" in out
    assert "NOTE: latest over-threshold behavior_k3 is on a pipelined policy-lag row" in out


def test_read_reference_dedups_restart_segments_keep_last(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.csv"
    # Two restart segments with overlapping global steps (1-3, then 2-4):
    # the resume block's values must win on the overlap.
    _write_reference(
        reference_path,
        step_rewards=[(1, 0.10), (2, 0.20), (3, 0.30), (2, 0.25), (3, 0.35), (4, 0.40)],
    )

    reference = _read_reference(reference_path)

    assert sorted(reference) == [1, 2, 3, 4]
    assert reference[2]["reward"] == 0.25
    assert reference[3]["reward"] == 0.35
    assert reference[4]["reward"] == 0.40


def test_summarize_aligns_our_step_to_reference_global_step(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    reference_path = tmp_path / "reference.csv"
    # Our 0-indexed step 0 must be compared against reference global step 1.
    _write_reference(reference_path, step_rewards=[(1, -0.10), (2, 0.90)])
    _write_metrics(metrics_path, same_policy=False)

    assert summarize(_args(metrics_path, reference_path)) == 0
    out = capsys.readouterr().out
    assert "0 -0.200 -0.100 -0.100" in out
