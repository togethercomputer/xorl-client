#!/usr/bin/env python3
"""Compare OPD profile/server timing artifacts from two runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNNER_RE = re.compile(
    r"forward_backward step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+) "
    r"tokens=(?P<tokens>\d+) time=(?P<time>[0-9.]+)s"
)
EXECUTOR_RE = re.compile(
    r"executor forward_backward: pack=(?P<pack>[0-9.]+)s "
    r"backend=(?P<backend>[0-9.]+)s build_output=(?P<build>[0-9.]+)s "
    r"total=(?P<total>[0-9.]+)s \| loss=(?P<loss>[0-9.eE+-]+), "
    r"tokens=(?P<tokens>\d+)"
)


PROFILE_FIELDS = [
    "step_total_s",
    "forward_backward_s",
    "sync_inference_weights_s",
    "optim_step_queued_s",
    "optim_step_s",
    "teacher_prefill_forward_compute_s",
    "teacher_prefill_s",
    "student_sampling_s",
    "teacher_hidden_cache_write_s",
    "prepare_window_s",
    "student_sampling_output_tokens",
    "teacher_prefill_tokens",
    "valid_tokens",
]


SUBPHASE_FIELDS = [
    "opd_profile_forward_loop_total_s",
    "opd_profile_forward_compute_s",
    "opd_profile_backward_compute_s",
    "opd_profile_model_forward_s",
    "opd_profile_loss_compute_s",
    "opd_profile_prefetch_s",
    "opd_profile_hidden_fetch_s",
    "opd_profile_head_prepare_s",
    "opd_profile_kl_compute_s",
    "opd_profile_loss_total_s",
    "opd_profile_input_transfer_s",
    "opd_profile_per_token_collect_s",
    "opd_profile_deferred_k3_s",
    "opd_profile_loss_report_allreduce_s",
    "opd_profile_sp_grad_sync_s",
    "opd_profile_metric_finalize_s",
    "opd_profile_final_synchronize_s",
    "opd_profile_clear_gradients_s",
]


@dataclass
class RunArtifacts:
    label: str
    run_dir: Path
    profile_rows: list[dict[str, Any]]
    runner_records: list[dict[str, Any]]
    executor_records: list[dict[str, Any]]
    step_groups: dict[int, dict[str, Any]]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _pct(delta: float, base: float) -> float:
    return (delta / base * 100.0) if base else 0.0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "sum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "gt10": 0,
            "gt20": 0,
            "gt40": 0,
        }
    sorted_values = sorted(float(v) for v in values)
    return {
        "n": len(sorted_values),
        "sum": sum(sorted_values),
        "mean": statistics.mean(sorted_values),
        "median": statistics.median(sorted_values),
        "p90": _quantile(sorted_values, 0.90),
        "p95": _quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
        "gt10": sum(v > 10.0 for v in sorted_values),
        "gt20": sum(v > 20.0 for v in sorted_values),
        "gt40": sum(v > 40.0 for v in sorted_values),
    }


def _load_profile(path: Path) -> list[dict[str, Any]]:
    profile_path = path / "opd_profile.jsonl"
    rows: list[dict[str, Any]] = []
    with profile_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_server_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runners: list[dict[str, Any]] = []
    executors: list[dict[str, Any]] = []
    with (path / "server.log").open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            runner = RUNNER_RE.search(line)
            if runner:
                runners.append(
                    {
                        "step": int(runner.group("step")),
                        "loss": float(runner.group("loss")),
                        "tokens": int(runner.group("tokens")),
                        "time": float(runner.group("time")),
                    }
                )
            executor = EXECUTOR_RE.search(line)
            if executor:
                executors.append(
                    {
                        "pack": float(executor.group("pack")),
                        "backend": float(executor.group("backend")),
                        "build_output": float(executor.group("build")),
                        "total": float(executor.group("total")),
                        "loss": float(executor.group("loss")),
                        "tokens": int(executor.group("tokens")),
                    }
                )
    return runners, executors


def _group_by_profile_rows(
    profile_rows: list[dict[str, Any]],
    runner_records: list[dict[str, Any]],
    executor_records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    offset = 0
    for row in profile_rows:
        step = int(row["step"])
        count = int(row.get("num_microbatches", 0))
        runner_slice = runner_records[offset : offset + count]
        executor_slice = executor_records[offset : offset + count]
        groups[step] = {
            "runner": runner_slice,
            "executor": executor_slice,
            "runner_stats": _stats([record["time"] for record in runner_slice]),
            "executor_stats": _stats([record["total"] for record in executor_slice]),
            "executor_backend_stats": _stats([record["backend"] for record in executor_slice]),
            "runner_token_sum": sum(int(record["tokens"]) for record in runner_slice),
            "executor_token_sum": sum(int(record["tokens"]) for record in executor_slice),
        }
        offset += count
    return groups


def _load_run(label: str, path: Path) -> RunArtifacts:
    rows = _load_profile(path)
    runners, executors = _parse_server_log(path)
    groups = _group_by_profile_rows(rows, runners, executors)
    return RunArtifacts(label, path, rows, runners, executors, groups)


def _parse_window(spec: str) -> set[int]:
    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            steps.update(range(int(start), int(end) + 1))
        else:
            steps.add(int(part))
    return steps


def _selected_rows(run: RunArtifacts, steps: set[int]) -> list[dict[str, Any]]:
    return [row for row in run.profile_rows if int(row["step"]) in steps]


def _profile_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for field in PROFILE_FIELDS:
        vals = [float(row[field]) for row in rows if field in row and isinstance(row[field], (int, float))]
        if vals:
            means[field] = _mean(vals)
    return means


def _subphase_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for field in SUBPHASE_FIELDS:
        vals = [float(row[field]) for row in rows if field in row and isinstance(row[field], (int, float))]
        if vals:
            means[field] = _mean(vals)
    return means


def _subphase_table(old_means: dict[str, float], clean_means: dict[str, float]) -> str:
    fields_present = [f for f in SUBPHASE_FIELDS if f in old_means or f in clean_means]
    if not fields_present:
        return "_No `opd_profile_*` sub-phase metrics found in either run._"
    lines = [
        "| Sub-phase | Old mean s | Clean mean s | Delta s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in fields_present:
        old_value = old_means.get(field)
        clean_value = clean_means.get(field)
        old_str = "—" if old_value is None else _fmt(old_value)
        clean_str = "—" if clean_value is None else _fmt(clean_value)
        if old_value is not None and clean_value is not None:
            delta_str = _fmt(clean_value - old_value)
        else:
            delta_str = "—"
        lines.append(f"| `{field}` | {old_str} | {clean_str} | {delta_str} |")
    return "\n".join(lines)


def _pooled_microbatch_stats(
    run: RunArtifacts,
    rows: list[dict[str, Any]],
    group_key: str,
    field: str,
) -> dict[str, float]:
    values: list[float] = []
    per_step_sums: list[float] = []
    for row in rows:
        step = int(row["step"])
        records = run.step_groups.get(step, {}).get(group_key, [])
        field_values = [float(record[field]) for record in records]
        values.extend(field_values)
        if field_values:
            per_step_sums.append(sum(field_values))
    result = _stats(values)
    result["mean_step_sum"] = _mean(per_step_sums)
    return result


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _profile_comparison_table(old_means: dict[str, float], clean_means: dict[str, float]) -> str:
    lines = [
        "| Field | Old mean | Clean mean | Delta | Delta % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in PROFILE_FIELDS:
        if field not in old_means or field not in clean_means:
            continue
        old_value = old_means[field]
        clean_value = clean_means[field]
        delta = clean_value - old_value
        lines.append(
            f"| `{field}` | {_fmt(old_value)} | {_fmt(clean_value)} | "
            f"{_fmt(delta)} | {_fmt(_pct(delta, old_value), 1)}% |"
        )
    return "\n".join(lines)


def _microbatch_table(old: dict[str, float], clean: dict[str, float], label: str) -> str:
    fields = ["mean_step_sum", "mean", "median", "p90", "p95", "max", "gt10", "gt20", "gt40"]
    lines = [
        f"| {label} metric | Old | Clean | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in fields:
        old_value = float(old.get(field, 0.0))
        clean_value = float(clean.get(field, 0.0))
        lines.append(f"| `{field}` | {_fmt(old_value)} | {_fmt(clean_value)} | {_fmt(clean_value - old_value)} |")
    return "\n".join(lines)


def _top_microbatches(run: RunArtifacts, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        step = int(row["step"])
        for record in run.step_groups.get(step, {}).get("runner", []):
            item = dict(record)
            item["opd_step"] = step
            records.append(item)
    return sorted(records, key=lambda item: float(item["time"]), reverse=True)[:limit]


def render_report(
    old: RunArtifacts,
    clean: RunArtifacts,
    old_steps: set[int],
    clean_steps: set[int],
    old_label: str,
    clean_label: str,
) -> str:
    old_rows = _selected_rows(old, old_steps)
    clean_rows = _selected_rows(clean, clean_steps)
    old_means = _profile_means(old_rows)
    clean_means = _profile_means(clean_rows)
    old_subphase = _subphase_means(old_rows)
    clean_subphase = _subphase_means(clean_rows)

    old_runner = _pooled_microbatch_stats(old, old_rows, "runner", "time")
    clean_runner = _pooled_microbatch_stats(clean, clean_rows, "runner", "time")
    old_executor = _pooled_microbatch_stats(old, old_rows, "executor", "total")
    clean_executor = _pooled_microbatch_stats(clean, clean_rows, "executor", "total")
    old_backend = _pooled_microbatch_stats(old, old_rows, "executor", "backend")
    clean_backend = _pooled_microbatch_stats(clean, clean_rows, "executor", "backend")

    step_delta = clean_means.get("step_total_s", 0.0) - old_means.get("step_total_s", 0.0)
    fb_delta = clean_means.get("forward_backward_s", 0.0) - old_means.get("forward_backward_s", 0.0)
    teacher_compute_delta = clean_means.get("teacher_prefill_forward_compute_s", 0.0) - old_means.get(
        "teacher_prefill_forward_compute_s", 0.0
    )
    sample_delta = clean_means.get("student_sampling_s", 0.0) - old_means.get("student_sampling_s", 0.0)
    sync_delta = clean_means.get("sync_inference_weights_s", 0.0) - old_means.get("sync_inference_weights_s", 0.0)

    top_old = _top_microbatches(old, old_rows, 5)
    top_clean = _top_microbatches(clean, clean_rows, 5)

    lines = [
        "# OPD Old vs Clean Profiling Gap",
        "",
        "## Inputs",
        "",
        f"- Old run: `{old.run_dir}`",
        f"- Clean run: `{clean.run_dir}`",
        f"- Old window: `{old_label}` ({len(old_rows)} OPD rows)",
        f"- Clean window: `{clean_label}` ({len(clean_rows)} OPD rows)",
        f"- Server records parsed: old `{len(old.runner_records)}` runner / `{len(old.executor_records)}` executor; "
        f"clean `{len(clean.runner_records)}` runner / `{len(clean.executor_records)}` executor.",
        "",
        "## Main Finding",
        "",
        f"Clean is {_fmt(step_delta)}s slower per selected OPD step. "
        f"The direct `forward_backward_s` delta is {_fmt(fb_delta)}s, while sync is {_fmt(sync_delta)}s "
        f"and student sampling is {_fmt(sample_delta)}s. Teacher forward compute is also "
        f"{_fmt(teacher_compute_delta)}s slower, but much of teacher preparation overlaps the trainer work.",
        "",
        "The server microbatch timing shows a broad per-microbatch slowdown rather than one stalled request. "
        f"Runner median is {_fmt(clean_runner['median'])}s clean vs {_fmt(old_runner['median'])}s old; "
        f"executor median is {_fmt(clean_executor['median'])}s clean vs {_fmt(old_executor['median'])}s old.",
        "",
        "## OPD Profile Means",
        "",
        _profile_comparison_table(old_means, clean_means),
        "",
        "## OPD Server Sub-Phase Means (per OPD step, summed across micro-batches)",
        "",
        _subphase_table(old_subphase, clean_subphase),
        "",
        "## Server Microbatch Timings",
        "",
        "Runner `time=` records:",
        "",
        _microbatch_table(old_runner, clean_runner, "Runner"),
        "",
        "Executor `total=` records:",
        "",
        _microbatch_table(old_executor, clean_executor, "Executor"),
        "",
        "Executor `backend=` records:",
        "",
        _microbatch_table(old_backend, clean_backend, "Executor backend"),
        "",
        "## Slowest Runner Microbatches",
        "",
        "| Run | OPD step | Server step | Time s | Logged tokens | Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, records in ((old.label, top_old), (clean.label, top_clean)):
        for record in records:
            lines.append(
                f"| {label} | {record['opd_step']} | {record['step']} | "
                f"{_fmt(float(record['time']))} | {int(record['tokens'])} | {_fmt(float(record['loss']), 4)} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Student sampling and P2P sync are not the current bottleneck; both are faster in the clean run.",
            "- The large wall-time gap tracks trainer forward/backward and a slower teacher hidden-cache forward path.",
            "- The selected trainer config topology is effectively the same: 4 trainer nodes, FSDP=32, EP=8, packing on, "
            "and `recompute_before_dispatch` in both old and clean logs.",
            "- Logged `valid_tokens` and server `tokens` are not directly comparable across the two codepaths: clean logs "
            "roughly 8x the teacher token count while old logs local token counts. Packed batch sizes in server logs remain "
            "around 28k tokens, so use wall-clock timings for this comparison.",
            "- When sub-phase rows are present, divide each `opd_profile_*` by `num_microbatches` (64 here) to get the "
            "per-microbatch wall-clock contribution. Forward+backward MAX-reduced sums can over-count the loop wall "
            "by ~0.4s/call because each rank may be slowest in a different phase.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--old-steps", default="56:101")
    parser.add_argument("--clean-steps", default="1:2")
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--clean-label", default="clean")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    old = _load_run(args.old_label, args.old_run)
    clean = _load_run(args.clean_label, args.clean_run)
    report = render_report(
        old,
        clean,
        _parse_window(args.old_steps),
        _parse_window(args.clean_steps),
        args.old_steps,
        args.clean_steps,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
