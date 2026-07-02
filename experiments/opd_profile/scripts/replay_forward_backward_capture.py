#!/usr/bin/env python3
"""Replay a captured OPD forward_backward payload against a live trainer server.

This bypasses OPD sampling, teacher prefill, inference endpoint registration, and
weight sync. Use it for trainer-only fb A/B tests once a representative payload
has been captured from `examples/on_policy_distillation.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _add_xorl_client_path(path: str) -> None:
    if path:
        sys.path.insert(0, path)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    for candidate in (key, f"{key}:mean", f"{key}:sum"):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metric_first(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _metric(metrics, key)
        if value is not None:
            return value
    return None


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _load_capture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request", payload)
    if "forward_backward_input" not in request:
        raise ValueError(f"{path} does not look like a forward_backward capture")
    return payload, request


def _parse_loss_param(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--loss-param must be KEY=VALUE")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--loss-param key must be non-empty")
    raw = raw.strip()
    lowered = raw.lower()
    if lowered == "true":
        parsed: Any = True
    elif lowered == "false":
        parsed = False
    elif lowered in {"none", "null"}:
        parsed = None
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
    return key, parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:26050")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--repeat-data",
        type=int,
        default=1,
        help="Repeat the captured data list in memory before each replay call. Throughput-only; do not use for correctness.",
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=0,
        help="Replay only the first N captured datums before repeat-data. Throughput fit-ladder only.",
    )
    parser.add_argument("--xorl-client-path", default="/home/apanda/xorl-client")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument(
        "--no-clear-gradients",
        action="store_true",
        help="Do not force profile_clear_gradients_after_backward. Only use when also running optim_step.",
    )
    parser.add_argument("--optim-step", action="store_true")
    parser.add_argument("--skip-session-registration", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--loss-param",
        action="append",
        type=_parse_loss_param,
        default=[],
        help="Override captured loss_fn_params with KEY=VALUE. VALUE is parsed as JSON/bool/null when possible.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0 or args.warmup >= args.iterations:
        raise ValueError("--warmup must be >=0 and < --iterations")
    if args.repeat_data <= 0:
        raise ValueError("--repeat-data must be positive")
    if args.limit_data < 0:
        raise ValueError("--limit-data must be non-negative")

    _add_xorl_client_path(args.xorl_client_path)
    import xorl_client as tomi  # noqa: PLC0415
    from xorl_client.client.training_client import TrainingClient  # noqa: PLC0415

    payload, request = _load_capture(args.capture)
    forward_backward_input = request["forward_backward_input"]
    data = forward_backward_input["data"]
    original_num_datums = len(data)
    if args.limit_data:
        if args.limit_data > original_num_datums:
            raise ValueError(f"--limit-data={args.limit_data} exceeds captured datum count {original_num_datums}")
        data = data[: args.limit_data]
    if args.repeat_data > 1:
        data = [copy.deepcopy(datum) for _ in range(args.repeat_data) for datum in data]
    loss_fn = forward_backward_input.get("loss_fn", "opd_loss")
    loss_fn_params = copy.deepcopy(forward_backward_input.get("loss_fn_params") or {})
    for key, value in args.loss_param:
        loss_fn_params[key] = value
    if not args.no_clear_gradients:
        loss_fn_params["profile_clear_gradients_after_backward"] = True
        loss_fn_params["opd_profile_timings"] = True

    capture_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    model_id = args.model_id or request.get("model_id") or capture_metadata.get("model_id") or "default"
    model_name = args.model_name or request.get("base_model") or capture_metadata.get("model_name") or "default"
    service_client = tomi.ServiceClient(base_url=args.base_url, timeout=args.timeout)
    if not args.skip_session_registration:
        response = service_client.holder.post_sync(
            "/api/v1/create_session",
            {"session_id": model_id, "base_model": model_name},
        )
        print("REGISTER_SESSION " + json.dumps(response, sort_keys=True), flush=True)
    training_client = TrainingClient(
        holder=service_client.holder,
        model_id=model_id,
        base_model=model_name,
    )

    rows: list[dict[str, Any]] = []
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")

    for iteration in range(args.iterations):
        t0 = time.perf_counter()
        output = training_client.forward_backward(data, loss_fn=loss_fn, loss_fn_params=loss_fn_params).result(
            timeout=args.timeout
        )
        api_wall_s = time.perf_counter() - t0
        metrics = output.metrics or {}
        optim_wall_s = 0.0
        optim_metrics: dict[str, Any] = {}
        if args.optim_step:
            opt_t0 = time.perf_counter()
            optim_output = training_client.optim_step(
                tomi.AdamParams(learning_rate=args.learning_rate, grad_clip_norm=args.grad_clip_norm)
            ).result(timeout=args.timeout)
            optim_wall_s = time.perf_counter() - opt_t0
            optim_metrics = optim_output.metrics or {}
        row = {
            "iteration": iteration,
            "warmup": iteration < args.warmup,
            "api_wall_s": api_wall_s,
            "optim_wall_s": optim_wall_s,
            "optim_metrics": optim_metrics,
            "optim_empty_cache_skipped": optim_metrics.get("optim_empty_cache_skipped"),
            "server_forward_backward_s": _metric(metrics, "execution_time"),
            "valid_tokens": _metric(metrics, "valid_tokens"),
            "executor_batches": _metric(metrics, "executor_batches"),
            "executor_original_batches": _metric(metrics, "executor_original_batches"),
            "executor_packed_row_batch_size": _metric(metrics, "executor_packed_row_batch_size"),
            "loss": _metric(metrics, "loss"),
            "opd_profile_forward_compute_s": _metric(metrics, "opd_profile_forward_compute_s"),
            "opd_profile_backward_compute_s": _metric(metrics, "opd_profile_backward_compute_s"),
            "opd_profile_loss_total_s": _metric(metrics, "opd_profile_total_ms"),
            "opd_profile_kl_compute_s": _metric(metrics, "opd_profile_kl_compute_ms"),
            "opd_profile_hidden_fetch_s": _metric(metrics, "opd_profile_hidden_fetch_ms"),
            "opd_profile_model_forward_s": _metric(metrics, "opd_profile_model_forward_ms"),
            "opd_profile_loss_compute_s": _metric(metrics, "opd_profile_loss_compute_ms"),
            "opd_profile_oprd_teacher_forward_s": _metric(metrics, "opd_profile_oprd_teacher_forward_ms"),
            "opd_profile_oprd_layer_fetch_s": _metric(metrics, "opd_profile_oprd_layer_fetch_ms"),
            "opd_profile_clear_gradients_s": _metric(metrics, "opd_profile_clear_gradients_ms"),
            "opd_kl": _metric(metrics, "opd_kl"),
            "opd_weighted_kl": _metric(metrics, "opd_weighted_kl"),
            "opd_hidden_match_loss": _metric(metrics, "opd_hidden_match_loss"),
            "opd_hidden_match_raw_loss": _metric(metrics, "opd_hidden_match_raw_loss"),
            "opd_teacher_weight_mean": _metric(metrics, "opd_teacher_weight_mean"),
            "opd_loss_min": _metric_first(metrics, ("opd_loss_min", "opd_loss_min:min")),
            "opd_loss_max": _metric_first(metrics, ("opd_loss_max", "opd_loss_max:max")),
            "opd_loss_abs_mean": _metric(metrics, "opd_loss_abs_mean"),
            "opd_loss_clamp_frac": _metric(metrics, "opd_loss_clamp_frac"),
            "opd_num_teachers": _metric(metrics, "opd_num_teachers"),
            "loss_param_overrides": dict(args.loss_param),
        }
        for key in (
            "opd_profile_loss_total_s",
            "opd_profile_kl_compute_s",
            "opd_profile_hidden_fetch_s",
            "opd_profile_model_forward_s",
            "opd_profile_loss_compute_s",
            "opd_profile_oprd_teacher_forward_s",
            "opd_profile_oprd_layer_fetch_s",
            "opd_profile_clear_gradients_s",
        ):
            if row[key] is not None:
                row[key] = row[key] * 0.001
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.output_jsonl is not None:
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    measured = [row for row in rows if not row["warmup"]]
    summary = {
        "capture": str(args.capture),
        "capture_metadata": payload.get("metadata", {}),
        "base_url": args.base_url,
        "model_id": model_id,
        "model_name": model_name,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeat_data": args.repeat_data,
        "limit_data": args.limit_data,
        "loss_param_overrides": dict(args.loss_param),
        "effective_loss_fn_params": loss_fn_params,
        "original_num_datums": original_num_datums,
        "num_datums": len(data),
        "mean_api_wall_s": _mean([row["api_wall_s"] for row in measured]),
        "mean_optim_wall_s": _mean([row["optim_wall_s"] for row in measured]),
        "optim_empty_cache_skipped_values": [row["optim_empty_cache_skipped"] for row in measured],
        "p50_api_wall_s": _p50([row["api_wall_s"] for row in measured]),
        "mean_server_forward_backward_s": _mean(
            [
                row["server_forward_backward_s"]
                for row in measured
                if isinstance(row["server_forward_backward_s"], (int, float))
            ]
        ),
        "mean_executor_batches": _mean(
            [row["executor_batches"] for row in measured if isinstance(row["executor_batches"], (int, float))]
        ),
        "mean_executor_original_batches": _mean(
            [
                row["executor_original_batches"]
                for row in measured
                if isinstance(row["executor_original_batches"], (int, float))
            ]
        ),
        "executor_packed_row_batch_size_values": [
            row["executor_packed_row_batch_size"] for row in measured if row["executor_packed_row_batch_size"] is not None
        ],
        "mean_opd_profile_forward_compute_s": _mean(
            [
                row["opd_profile_forward_compute_s"]
                for row in measured
                if isinstance(row["opd_profile_forward_compute_s"], (int, float))
            ]
        ),
        "mean_opd_profile_backward_compute_s": _mean(
            [
                row["opd_profile_backward_compute_s"]
                for row in measured
                if isinstance(row["opd_profile_backward_compute_s"], (int, float))
            ]
        ),
        "mean_clear_gradients_s": _mean(
            [
                row["opd_profile_clear_gradients_s"]
                for row in measured
                if isinstance(row["opd_profile_clear_gradients_s"], (int, float))
            ]
        ),
        "mean_opd_profile_model_forward_s": _mean(
            [
                row["opd_profile_model_forward_s"]
                for row in measured
                if isinstance(row["opd_profile_model_forward_s"], (int, float))
            ]
        ),
        "mean_opd_profile_loss_compute_s": _mean(
            [
                row["opd_profile_loss_compute_s"]
                for row in measured
                if isinstance(row["opd_profile_loss_compute_s"], (int, float))
            ]
        ),
        "mean_opd_profile_oprd_teacher_forward_s": _mean(
            [
                row["opd_profile_oprd_teacher_forward_s"]
                for row in measured
                if isinstance(row["opd_profile_oprd_teacher_forward_s"], (int, float))
            ]
        ),
        "mean_opd_profile_oprd_layer_fetch_s": _mean(
            [
                row["opd_profile_oprd_layer_fetch_s"]
                for row in measured
                if isinstance(row["opd_profile_oprd_layer_fetch_s"], (int, float))
            ]
        ),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
