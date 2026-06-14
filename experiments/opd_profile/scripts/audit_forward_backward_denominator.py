#!/usr/bin/env python3
"""Audit token/MFU denominators for a captured OPD forward_backward payload.

Server OPD logs valid tokens and wall-time buckets, but not the local trainer's
MFU counter.  This script reconstructs the student-token denominators from a
captured forward_backward request and the server packing/dispatch rules, then
joins them with replay/full-run JSONL timing artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

IGNORE_INDEX = -100


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1))
    return float(ordered[idx])


def _stats(values: list[int | float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "count": float(len(vals)),
        "sum": float(sum(vals)),
        "min": min(vals) if vals else float("nan"),
        "p50": _p50(vals),
        "mean": _mean(vals),
        "p95": _percentile(vals, 0.95),
        "max": max(vals) if vals else float("nan"),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def _flatten_datum(datum: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(datum.get("model_input"), dict):
        flattened.update(datum["model_input"])
    if isinstance(datum.get("loss_fn_inputs"), dict):
        flattened.update(datum["loss_fn_inputs"])
    for key, value in datum.items():
        if key not in ("model_input", "loss_fn_inputs"):
            flattened[key] = value
    return flattened


def _teacher_sort_key(datum: dict[str, Any]) -> int:
    flattened = _flatten_datum(datum)
    teacher_id = flattened.get("teacher_id")
    if teacher_id is None:
        teacher_ids = flattened.get("teacher_ids")
        if hasattr(teacher_ids, "reshape"):
            teacher_ids = teacher_ids.reshape(-1).tolist()
        if isinstance(teacher_ids, list) and teacher_ids:
            while isinstance(teacher_ids[0], list) and teacher_ids[0]:
                teacher_ids = teacher_ids[0]
            teacher_id = teacher_ids[0]
    return int(teacher_id) if teacher_id is not None else 0


def _valid_count(tokens: Any) -> int:
    if tokens is None:
        return 0
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return sum(1 for token in tokens if token != IGNORE_INDEX)


def _position_lengths(position_ids: list[int]) -> list[int]:
    starts = [idx for idx, pos in enumerate(position_ids) if pos == 0]
    if len(starts) <= 1:
        return [len(position_ids)]
    return [end - start for start, end in zip(starts, [*starts[1:], len(position_ids)])]


def _sample_stats(data: list[dict[str, Any]]) -> dict[str, Any]:
    flattened = [_flatten_datum(datum) for datum in data]
    student_lengths = [len(datum.get("input_ids") or []) for datum in flattened]
    valid_lengths = [_valid_count(datum.get("target_tokens", datum.get("labels"))) for datum in flattened]
    teacher_lengths = [len(datum.get("teacher_input_ids") or []) for datum in flattened if datum.get("teacher_input_ids")]
    kept_lengths = [
        len(datum.get("teacher_kept_indices") or []) for datum in flattened if datum.get("teacher_kept_indices")
    ]
    teacher_ids = [_teacher_sort_key(datum) for datum in data]
    return {
        "num_samples": len(data),
        "student_input_tokens": _stats(student_lengths),
        "valid_target_tokens": _stats(valid_lengths),
        "teacher_input_tokens": _stats(teacher_lengths),
        "teacher_kept_rows": _stats(kept_lengths),
        "teacher_id_counts": {str(tid): teacher_ids.count(tid) for tid in sorted(set(teacher_ids))},
    }


def _dispatcher_range(dp_rank: int, base_count: int, remainder: int) -> tuple[int, int]:
    if dp_rank < remainder:
        return dp_rank * (base_count + 1), base_count + 1
    return remainder * (base_count + 1) + (dp_rank - remainder) * base_count, base_count


def _batch_summary(batch: dict[str, Any]) -> dict[str, Any]:
    def _flat_len(value: Any) -> int:
        if value is None:
            return 0
        if hasattr(value, "tolist"):
            value = value.tolist()
        while isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        return len(value) if isinstance(value, list) else 0

    row_len = _flat_len(batch["input_ids"])
    real_len = sum(batch.get("_r3_sample_lengths", []))
    labels = batch.get("labels", [[]])[0]
    position_ids = batch.get("position_ids", [[]])[0]
    return {
        "padded_tokens": row_len,
        "real_student_tokens": real_len,
        "valid_tokens": _valid_count(labels),
        "num_samples": int(batch.get("num_samples", 1)),
        "seqlens": _position_lengths(position_ids),
        "teacher_input_tokens": _flat_len(batch.get("teacher_input_ids")),
        "teacher_kept_rows": _flat_len(batch.get("teacher_kept_indices")),
    }


def _pack_report(
    data: list[dict[str, Any]],
    *,
    sample_packing_sequence_len: int,
    enable_packing: bool,
    sort_by_teacher: bool,
    dp_size: int,
    pad_to_multiple_of: int,
) -> dict[str, Any]:
    from xorl.server.orchestrator.packing import pack_samples

    ordered = sorted(data, key=_teacher_sort_key) if sort_by_teacher else data
    batches = pack_samples(
        ordered,
        max_seq_len=sample_packing_sequence_len,
        enable_packing=enable_packing,
        request_id="denominator-audit",
        pad_to_multiple_of=pad_to_multiple_of,
    )
    infos = [_batch_summary(batch) for batch in batches]
    row_tokens = [info["padded_tokens"] for info in infos]
    real_tokens = [info["real_student_tokens"] for info in infos]
    valid_tokens = [info["valid_tokens"] for info in infos]
    samples_per_row = [info["num_samples"] for info in infos]
    teacher_input_tokens = [info["teacher_input_tokens"] for info in infos]
    teacher_kept_rows = [info["teacher_kept_rows"] for info in infos]

    num_batches = len(infos)
    batches_per_dp_group = math.ceil(num_batches / dp_size) if dp_size > 0 else num_batches
    base_count = num_batches // dp_size
    remainder = num_batches % dp_size

    rank_executed_tokens: list[int] = []
    rank_real_tokens: list[int] = []
    rank_valid_tokens: list[int] = []
    rank_teacher_forward_tokens: list[int] = []
    rank_teacher_kept_rows: list[int] = []
    rank_real_batch_counts: list[int] = []
    rank_dummy_batch_counts: list[int] = []
    execution_seqlens: list[int] = []

    for rank in range(dp_size):
        start_idx, real_count = _dispatcher_range(rank, base_count, remainder)
        selected = infos[start_idx : start_idx + real_count]

        executed = sum(int(info["padded_tokens"]) for info in selected)
        real = sum(int(info["real_student_tokens"]) for info in selected)
        valid = sum(int(info["valid_tokens"]) for info in selected)
        teacher_forward = sum(int(info["teacher_input_tokens"]) for info in selected)
        teacher_kept = sum(int(info["teacher_kept_rows"]) for info in selected)
        for info in selected:
            execution_seqlens.extend(int(seqlen) for seqlen in info["seqlens"])

        dummy_count = max(0, batches_per_dp_group - real_count)
        if dummy_count:
            reference = selected[-1] if selected else infos[0]
            executed += int(reference["padded_tokens"]) * dummy_count
            teacher_forward += int(reference["teacher_input_tokens"]) * dummy_count
            teacher_kept += int(reference["teacher_kept_rows"]) * dummy_count
            execution_seqlens.extend(int(seqlen) for seqlen in reference["seqlens"] for _ in range(dummy_count))

        rank_executed_tokens.append(executed)
        rank_real_tokens.append(real)
        rank_valid_tokens.append(valid)
        rank_teacher_forward_tokens.append(teacher_forward)
        rank_teacher_kept_rows.append(teacher_kept)
        rank_real_batch_counts.append(real_count)
        rank_dummy_batch_counts.append(dummy_count)

    dispatcher_executed_tokens = sum(rank_executed_tokens)
    packed_padded_tokens = sum(row_tokens)
    real_student_tokens_total = sum(real_tokens)
    valid_tokens_total = sum(valid_tokens)
    teacher_forward_tokens_total = sum(rank_teacher_forward_tokens)
    teacher_kept_rows_total = sum(rank_teacher_kept_rows)
    mean_rank_tokens = _mean([float(v) for v in rank_executed_tokens])

    return {
        "sample_packing_sequence_len": sample_packing_sequence_len,
        "enable_packing": enable_packing,
        "sort_by_teacher": sort_by_teacher,
        "dp_size": dp_size,
        "pad_to_multiple_of": pad_to_multiple_of,
        "num_packed_rows": num_batches,
        "batches_per_dp_group": batches_per_dp_group,
        "capacity_tokens": num_batches * sample_packing_sequence_len,
        "packed_padded_tokens_without_dispatch_dummy": packed_padded_tokens,
        "real_student_tokens_without_dispatch_dummy": real_student_tokens_total,
        "valid_tokens": valid_tokens_total,
        "packing_capacity_utilization": real_student_tokens_total / (num_batches * sample_packing_sequence_len),
        "packing_padding_utilization": real_student_tokens_total / packed_padded_tokens,
        "valid_fraction_of_packed_padded_tokens": valid_tokens_total / packed_padded_tokens,
        "dispatcher_executed_tokens": dispatcher_executed_tokens,
        "dispatcher_dummy_batches": sum(rank_dummy_batch_counts),
        "dispatcher_dummy_executed_tokens": dispatcher_executed_tokens - packed_padded_tokens,
        "packed_teacher_input_tokens_without_dispatch_dummy": sum(teacher_input_tokens),
        "dispatcher_teacher_forward_tokens": teacher_forward_tokens_total,
        "dispatcher_dummy_teacher_forward_tokens": teacher_forward_tokens_total - sum(teacher_input_tokens),
        "packed_teacher_kept_rows_without_dispatch_dummy": sum(teacher_kept_rows),
        "dispatcher_teacher_kept_rows": teacher_kept_rows_total,
        "student_plus_teacher_forward_tokens": dispatcher_executed_tokens + teacher_forward_tokens_total,
        "valid_fraction_of_dispatcher_executed_tokens": valid_tokens_total / dispatcher_executed_tokens,
        "valid_fraction_of_student_plus_teacher_forward_tokens": valid_tokens_total
        / (dispatcher_executed_tokens + teacher_forward_tokens_total),
        "rank_executed_tokens": _stats(rank_executed_tokens),
        "rank_real_student_tokens": _stats(rank_real_tokens),
        "rank_valid_tokens": _stats(rank_valid_tokens),
        "rank_teacher_forward_tokens": _stats(rank_teacher_forward_tokens),
        "rank_teacher_kept_rows": _stats(rank_teacher_kept_rows),
        "rank_real_batch_counts": _stats(rank_real_batch_counts),
        "rank_dummy_batch_counts": _stats(rank_dummy_batch_counts),
        "rank_load_imbalance_max_over_mean": max(rank_executed_tokens) / mean_rank_tokens if mean_rank_tokens else 0.0,
        "packed_row_tokens": _stats(row_tokens),
        "packed_row_real_student_tokens": _stats(real_tokens),
        "packed_row_valid_tokens": _stats(valid_tokens),
        "packed_row_teacher_input_tokens": _stats(teacher_input_tokens),
        "packed_row_teacher_kept_rows": _stats(teacher_kept_rows),
        "packed_row_samples": _stats(samples_per_row),
        "execution_seqlens": execution_seqlens,
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    if "warmup" in rows[0]:
        measured = [row for row in rows if not row.get("warmup")]
    elif "profile_warmup" in rows[0]:
        measured = [row for row in rows if not row.get("profile_warmup")]
    else:
        measured = rows[2:] if len(rows) > 2 else rows
    keys = sorted({key for row in measured for key, value in row.items() if isinstance(value, (int, float))})
    return {key: _mean([float(row[key]) for row in measured if isinstance(row.get(key), (int, float))]) for key in keys}


def _load_flops_counter(model_config: str | None):
    if not model_config:
        return None, None
    try:
        from transformers import AutoConfig
        from xorl.utils.count_flops import XorlFlopsCounter
    except Exception as exc:
        return None, f"could not import FLOP counter dependencies: {exc}"

    try:
        parent = AutoConfig.from_pretrained(model_config, trust_remote_code=True)
        cfg = copy.deepcopy(getattr(parent, "text_config", parent))
        parent_model_type = getattr(parent, "model_type", "")
        if parent_model_type == "qwen3_5_moe" or getattr(cfg, "model_type", "") == "qwen3_5_moe_text":
            cfg.model_type = "xorl_qwen3_5_moe"
        if not hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size = getattr(cfg, "shared_expert_intermediate_size", None)
        counter = XorlFlopsCounter(
            cfg,
            gradient_checkpointing_enabled=True,
            gradient_checkpointing_method="recompute_before_dispatch",
        )
        return counter, None
    except Exception as exc:
        return None, f"could not initialize FLOP counter for {model_config}: {exc}"


def _rates_for_timing(
    means: dict[str, float],
    pack: dict[str, Any],
    *,
    counter: Any,
    promised_tflops_per_gpu: float,
    gpus: int,
) -> dict[str, Any]:
    time_key = "server_forward_backward_s" if "server_forward_backward_s" in means else "forward_backward_s"
    if time_key not in means:
        return {}
    dt = means[time_key]
    valid_tokens = means.get("valid_tokens", pack["valid_tokens"])
    dispatcher_tokens = pack["dispatcher_executed_tokens"]
    real_tokens = pack["real_student_tokens_without_dispatch_dummy"]
    packed_tokens = pack["packed_padded_tokens_without_dispatch_dummy"]
    teacher_forward_tokens = pack.get("dispatcher_teacher_forward_tokens", 0)
    combined_forward_tokens = pack.get("student_plus_teacher_forward_tokens", dispatcher_tokens)

    rates: dict[str, Any] = {
        "time_key": time_key,
        "time_s": dt,
        "valid_tokens_per_gpu_s": valid_tokens / dt / gpus,
        "real_student_tokens_per_gpu_s": real_tokens / dt / gpus,
        "packed_padded_tokens_per_gpu_s": packed_tokens / dt / gpus,
        "dispatcher_executed_tokens_per_gpu_s": dispatcher_tokens / dt / gpus,
        "teacher_forward_tokens_per_gpu_s": teacher_forward_tokens / dt / gpus,
        "student_plus_teacher_forward_tokens_per_gpu_s": combined_forward_tokens / dt / gpus,
        "valid_to_dispatcher_executed_token_ratio": valid_tokens / dispatcher_tokens,
        "valid_to_student_plus_teacher_forward_token_ratio": valid_tokens / combined_forward_tokens,
    }

    if counter is not None:
        achieved_tflops, _ = counter.estimate_flops(pack["execution_seqlens"], dt)
        mfu = achieved_tflops / (promised_tflops_per_gpu * gpus)
        rates.update(
            {
                "reconstructed_logical_tflops_total": achieved_tflops,
                "reconstructed_logical_tflops_per_gpu": achieved_tflops / gpus,
                "reconstructed_logical_mfu": mfu,
                "valid_token_scaled_logical_mfu": mfu * valid_tokens / dispatcher_tokens,
            }
        )
    return rates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--sample-packing-sequence-len", dest="seq_lens", type=int, action="append")
    parser.add_argument("--disable-packing", action="store_true")
    parser.add_argument("--disable-opd-sort-by-teacher", action="store_true")
    parser.add_argument("--gpus", type=int, default=32)
    parser.add_argument("--pad-to-multiple-of", type=int, default=128)
    parser.add_argument("--model-config", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--promised-tflops-per-gpu", type=float, default=989.0)
    parser.add_argument("--replay-jsonl", action="append", default=[], help="LABEL=PATH or PATH")
    parser.add_argument("--profile-jsonl", action="append", default=[], help="LABEL=PATH or PATH")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    seq_lens = args.seq_lens or [4096]
    payload = json.loads(args.capture.read_text(encoding="utf-8"))
    request = payload.get("request", payload)
    forward_backward_input = request["forward_backward_input"]
    data = forward_backward_input["data"]
    loss_fn = forward_backward_input.get("loss_fn")
    loss_fn_params = forward_backward_input.get("loss_fn_params") or {}
    sort_by_teacher = (
        loss_fn == "opd_loss"
        and not args.disable_opd_sort_by_teacher
        and bool(loss_fn_params.get("opd_sort_by_teacher", True))
    )

    counter, flops_warning = _load_flops_counter(args.model_config)
    pack_reports = {
        str(seq_len): _pack_report(
            data,
            sample_packing_sequence_len=seq_len,
            enable_packing=not args.disable_packing,
            sort_by_teacher=sort_by_teacher,
            dp_size=args.gpus,
            pad_to_multiple_of=args.pad_to_multiple_of,
        )
        for seq_len in seq_lens
    }
    primary_pack = pack_reports[str(seq_lens[0])]

    timing_summaries: dict[str, Any] = {}
    for kind, values in (("replay", args.replay_jsonl), ("profile", args.profile_jsonl)):
        for labeled in values:
            label, path = _parse_labeled_path(labeled)
            rows = _load_jsonl(path)
            means = _mean_metrics(rows)
            timing_summaries[f"{kind}:{label}"] = {
                "path": str(path),
                "rows": len(rows),
                "means": means,
                "rates_using_primary_capture_denominator": _rates_for_timing(
                    means,
                    primary_pack,
                    counter=counter,
                    promised_tflops_per_gpu=args.promised_tflops_per_gpu,
                    gpus=args.gpus,
                ),
            }

    report = {
        "capture": str(args.capture),
        "metadata": payload.get("metadata", {}),
        "loss_fn": loss_fn,
        "loss_fn_params_keys": sorted(loss_fn_params.keys()),
        "sample_stats_before_packing": _sample_stats(data),
        "pack_reports": {
            key: {k: v for k, v in value.items() if k != "execution_seqlens"} for key, value in pack_reports.items()
        },
        "timing_summaries": timing_summaries,
        "flops_model_config": args.model_config,
        "flops_warning": flops_warning,
        "notes": [
            "reconstructed_logical_mfu uses the repo logical FLOP convention and the capture's packed/dummy execution seqlens",
            "valid_token_scaled_logical_mfu is diagnostic only; it shows how low the number looks if valid target tokens are used as the work denominator",
            "profile JSONL timing rates use the capture denominator unless the profile came from the same captured replay payload",
        ],
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
