from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SFT_BASELINE_TARGETS = {
    "math500": 45.0,
    "gsm8k-flex": 64.1,
    "aime24": 4.9,
}

EXPECTED_ROWS = {
    "math500": 500,
    "gsm8k": 1319,
    "aime24": 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Marin #6279 eval JSONL outputs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full JSON summary.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _summarize_rows(path: Path, *, expected_rows: int) -> dict[str, Any]:
    rows = _read_jsonl(path)
    num_scored = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    accuracy_percent = 100.0 * correct / num_scored if num_scored else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "complete": num_scored == expected_rows,
        "expected_rows": expected_rows,
        "expected_scored": expected_rows,
        "num_scored": num_scored,
        "skipped_long": 0,
        "correct": correct,
        "score": correct,
        "accuracy_percent": accuracy_percent,
    }


def _latest_task_status(run_dir: Path, *, task: str, seed: int) -> dict[str, Any] | None:
    latest = None
    for row in _read_jsonl(run_dir / "summary.jsonl"):
        if row.get("task") == task and row.get("seed") == seed:
            latest = row
    return latest


def _apply_skipped_long(summary: dict[str, Any], status: dict[str, Any] | None) -> dict[str, Any]:
    if not status:
        return summary
    skipped_long = int(status.get("skipped_long") or 0)
    expected_scored = max(0, summary["expected_rows"] - skipped_long)
    return {
        **summary,
        "complete": summary["num_scored"] + skipped_long == summary["expected_rows"],
        "expected_scored": expected_scored,
        "skipped_long": skipped_long,
        "reported_num_scored": status.get("num_scored"),
    }


def _repeat_count_from_status_or_rows(status: dict[str, Any] | None, path: Path) -> int:
    if status and status.get("repeat_count"):
        return max(1, int(status["repeat_count"]))
    rows = _read_jsonl(path)
    repeat_counts = [int(row["repeat_count"]) for row in rows if row.get("repeat_count")]
    if repeat_counts:
        return max(1, max(repeat_counts))
    repeat_indices = [int(row["repeat_index"]) for row in rows if row.get("repeat_index") is not None]
    return max(1, max(repeat_indices) + 1) if repeat_indices else 1


def summarize_run(run_dir: Path) -> dict[str, Any]:
    math500 = _apply_skipped_long(
        _summarize_rows(run_dir / "sft_math500_seed42.jsonl", expected_rows=EXPECTED_ROWS["math500"]),
        _latest_task_status(run_dir, task="math500", seed=42),
    )
    gsm8k = _apply_skipped_long(
        _summarize_rows(run_dir / "sft_gsm8k_seed42.jsonl", expected_rows=EXPECTED_ROWS["gsm8k"]),
        _latest_task_status(run_dir, task="gsm8k", seed=42),
    )

    aime_seed_summaries = {}
    for seed in range(42, 52):
        path = run_dir / f"sft_aime24_seed{seed}.jsonl"
        status = _latest_task_status(run_dir, task="aime24", seed=seed)
        repeat_count = _repeat_count_from_status_or_rows(status, path)
        summary = _apply_skipped_long(
            _summarize_rows(path, expected_rows=EXPECTED_ROWS["aime24"] * repeat_count),
            status,
        )
        summary["repeat_count"] = repeat_count
        summary["score"] = summary["correct"] / repeat_count
        aime_seed_summaries[str(seed)] = summary

    complete_aime = [summary for summary in aime_seed_summaries.values() if summary["complete"]]
    all_aime = list(aime_seed_summaries.values())
    complete_aime_accuracy = [
        summary["accuracy_percent"] for summary in complete_aime if summary["accuracy_percent"] is not None
    ]
    all_aime_accuracy = [summary["accuracy_percent"] for summary in all_aime if summary["accuracy_percent"] is not None]
    complete_aime_scores = [summary["score"] for summary in complete_aime]
    all_aime_scores = [summary["score"] for summary in all_aime if summary["num_scored"]]

    tasks = {
        "math500": {
            **math500,
            "target_accuracy_percent": SFT_BASELINE_TARGETS["math500"],
            "delta_from_target": _delta(math500["accuracy_percent"], SFT_BASELINE_TARGETS["math500"]),
        },
        "gsm8k-flex": {
            **gsm8k,
            "target_accuracy_percent": SFT_BASELINE_TARGETS["gsm8k-flex"],
            "delta_from_target": _delta(gsm8k["accuracy_percent"], SFT_BASELINE_TARGETS["gsm8k-flex"]),
        },
        "aime24": {
            "seeds": aime_seed_summaries,
            "complete": len(complete_aime) == 10,
            "complete_seed_count": len(complete_aime),
            "seed_count_with_rows": len(all_aime_accuracy),
            "mean_accuracy_percent_complete_seeds": statistics.fmean(complete_aime_accuracy)
            if complete_aime_accuracy
            else None,
            "mean_accuracy_percent_available_seeds": statistics.fmean(all_aime_accuracy) if all_aime_accuracy else None,
            "mean_score_complete_seeds": statistics.fmean(complete_aime_scores) if complete_aime_scores else None,
            "mean_score_available_seeds": statistics.fmean(all_aime_scores) if all_aime_scores else None,
            "target_mean_accuracy_percent": SFT_BASELINE_TARGETS["aime24"],
            "delta_from_target": _delta(
                statistics.fmean(complete_aime_accuracy) if complete_aime_accuracy else None,
                SFT_BASELINE_TARGETS["aime24"],
            ),
        },
    }
    return {
        "run_dir": str(run_dir),
        "complete": all(task["complete"] for task in tasks.values()),
        "tasks": tasks,
    }


def _delta(value: float | None, target: float) -> float | None:
    if value is None:
        return None
    return value - target


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_scored(task: dict[str, Any]) -> str:
    scored = f"{task['num_scored']}/{task['expected_scored']}"
    if task.get("skipped_long"):
        scored = f"{scored} (+{task['skipped_long']} skipped)"
    return scored


def print_markdown_summary(summary: dict[str, Any]) -> None:
    tasks = summary["tasks"]
    print(f"run_dir: {summary['run_dir']}")
    print()
    print("| metric | complete | scored | value | target | delta |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for metric in ("math500", "gsm8k-flex"):
        task = tasks[metric]
        print(
            "| "
            f"{metric} | {task['complete']} | {_format_scored(task)} | "
            f"{_format_percent(task['accuracy_percent'])} | "
            f"{_format_percent(task['target_accuracy_percent'])} | "
            f"{_format_percent(task['delta_from_target'])} |"
        )
    aime = tasks["aime24"]
    print(
        "| "
        f"aime24 | {aime['complete']} | {aime['complete_seed_count']}/10 seeds | "
        f"{_format_percent(aime['mean_accuracy_percent_complete_seeds'])} | "
        f"{_format_percent(aime['target_mean_accuracy_percent'])} | "
        f"{_format_percent(aime['delta_from_target'])} |"
    )


def main() -> int:
    args = parse_args()
    summary = summarize_run(args.run_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_markdown_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
