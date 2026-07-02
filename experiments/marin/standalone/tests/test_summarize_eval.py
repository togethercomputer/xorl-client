from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.marin.standalone.summarize_eval import summarize_run


def _write_jsonl(path: Path, correct_values: list[bool]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, correct in enumerate(correct_values):
            handle.write(json.dumps({"index": index, "correct": correct}) + "\n")


def test_summarize_run_reports_completeness_and_targets(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "sft_math500_seed42.jsonl", [True, False])
    _write_jsonl(tmp_path / "sft_gsm8k_seed42.jsonl", [True, True, False, False])
    _write_jsonl(tmp_path / "sft_aime24_seed42.jsonl", [True] + [False] * 29)

    summary = summarize_run(tmp_path)

    assert not summary["complete"]
    assert summary["tasks"]["math500"]["num_scored"] == 2
    assert summary["tasks"]["math500"]["accuracy_percent"] == 50.0
    assert summary["tasks"]["math500"]["delta_from_target"] == 5.0
    assert not summary["tasks"]["gsm8k-flex"]["complete"]
    assert summary["tasks"]["aime24"]["complete_seed_count"] == 1
    assert summary["tasks"]["aime24"]["mean_accuracy_percent_complete_seeds"] == 100.0 / 30.0
    assert summary["tasks"]["aime24"]["mean_score_complete_seeds"] == 1.0
    assert summary["tasks"]["aime24"]["target_mean_accuracy_percent"] == 4.9
    assert summary["tasks"]["aime24"]["delta_from_target"] == pytest.approx((100.0 / 30.0) - 4.9)


def test_summarize_run_treats_skipped_long_as_accounted_rows(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "sft_math500_seed42.jsonl", [True] * 499)
    with (tmp_path / "summary.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "math500",
                    "seed": 42,
                    "num_scored": 499,
                    "skipped_long": 1,
                    "accuracy_percent": 100.0,
                }
            )
            + "\n"
        )

    summary = summarize_run(tmp_path)
    math500 = summary["tasks"]["math500"]

    assert math500["complete"]
    assert math500["num_scored"] == 499
    assert math500["expected_scored"] == 499
    assert math500["skipped_long"] == 1


def test_summarize_run_accounts_for_repeated_aime_rows(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "sft_aime24_seed42.jsonl", [True] * 15 + [False] * 285)
    with (tmp_path / "summary.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "aime24",
                    "seed": 42,
                    "repeat_count": 10,
                    "repeat_seed_base": 0,
                    "num_scored": 300,
                    "skipped_long": 0,
                    "accuracy_percent": 5.0,
                }
            )
            + "\n"
        )

    summary = summarize_run(tmp_path)
    seed42 = summary["tasks"]["aime24"]["seeds"]["42"]

    assert seed42["complete"]
    assert seed42["expected_rows"] == 300
    assert seed42["repeat_count"] == 10
    assert seed42["correct"] == 15
    assert seed42["accuracy_percent"] == 5.0
    assert seed42["score"] == 1.5
    assert summary["tasks"]["aime24"]["mean_score_complete_seeds"] == 1.5
