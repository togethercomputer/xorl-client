"""Replay captured XORL /forward_backward requests against a training server.

The companion capture flag is:

    train_opsd_baseline.py --dump-forward-backward-replay forward_backward_replay.jsonl

The artifact is API-shaped JSONL, so this script does not need Wordle samplers,
teacher generation, SGLang weight sync, or teacher-cache creation. It only needs
the training server and the copied teacher-cache files referenced by the artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import requests


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _raise_on_failed_future(result: dict[str, Any], context: str) -> dict[str, Any]:
    if result.get("type") == "request_failed":
        raise RuntimeError(f"{context} failed: {result.get('error', result)}")
    if result.get("error"):
        raise RuntimeError(f"{context} failed: {result['error']}")
    return result


def wait_for_future(train_url: str, request_id: str, *, timeout: float, poll_interval: float = 1.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _post_json(
            f"{train_url}/api/v1/retrieve_future",
            {"request_id": request_id},
            timeout=120.0,
        )
        if result.get("type") == "try_again":
            time.sleep(poll_interval)
            continue
        return result
    raise TimeoutError(f"Timed out waiting for future {request_id}")


def call_future(
    train_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    context: str,
    submit_timeout: float = 120.0,
    future_timeout: float = 7200.0,
) -> dict[str, Any]:
    future = _post_json(f"{train_url}{endpoint}", payload, timeout=submit_timeout)
    request_id = future.get("request_id")
    if not request_id:
        raise RuntimeError(f"{context} did not return request_id: {future}")
    return _raise_on_failed_future(
        wait_for_future(train_url, str(request_id), timeout=future_timeout),
        context,
    )


def wait_for_training_service(train_url: str, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{train_url}/health", timeout=5)
            if resp.ok and resp.json().get("engine_running"):
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Training service did not become ready at {train_url} within {timeout}s")


def create_model(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": args.train_model_id,
        "base_model": args.model,
        "lora_config": None
        if args.full_weight
        else {
            "rank": args.lora_rank,
            "lora_rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "lora_alpha": args.lora_alpha,
        },
        "optimizer_config": None
        if args.full_weight
        else {
            "type": args.optimizer,
            "learning_rate": args.optim_lr,
            "weight_decay": args.weight_decay,
            "optimizer_dtype": args.optimizer_dtype,
            "betas": [args.beta1, args.beta2],
            "eps": args.eps,
        },
        "zorl_config": {"enabled": False},
    }
    return call_future(
        args.train_url,
        "/api/v1/create_model",
        payload,
        context=f"create_model({args.train_model_id})",
        future_timeout=args.future_timeout,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_base_name(key: str) -> tuple[str, str]:
    if ":" in key:
        base, suffix = key.rsplit(":", 1)
        return base, suffix
    return key, "mean"


def _valid_tokens(metrics: dict[str, Any]) -> float:
    for key in ("global_valid_tokens", "valid_tokens"):
        value = metrics.get(key)
        if _is_number(value):
            return max(float(value), 0.0)
    return 1.0


def summarize_responses(responses: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    passthrough_sums: dict[str, float] = {}
    loss_sum = 0.0
    loss_sum_tokens = 0.0

    for response in responses:
        metrics = response.get("metrics") or {}
        weight = _valid_tokens(metrics)
        for raw_key, value in metrics.items():
            if not _is_number(value):
                continue
            key, reduction = _metric_base_name(raw_key)
            if key == "loss" and reduction == "sum":
                loss_sum += float(value)
                loss_sum_tokens += weight
            elif key.startswith("opd_profile_") and key.endswith(("_s", "_ms")):
                passthrough_sums[key] = passthrough_sums.get(key, 0.0) + float(value)
            elif reduction == "sum" or key in {"valid_tokens", "global_valid_tokens"}:
                passthrough_sums[key] = passthrough_sums.get(key, 0.0) + float(value)
            elif key == "loss":
                sums[key] = sums.get(key, 0.0) + float(value) * weight
                counts[key] = counts.get(key, 0.0) + weight
            else:
                sums[key] = sums.get(key, 0.0) + float(value) * weight
                counts[key] = counts.get(key, 0.0) + weight

    out = {key: total / max(counts[key], 1.0) for key, total in sums.items()}
    if "loss" not in out and loss_sum_tokens > 0:
        out["loss"] = loss_sum / loss_sum_tokens
    out.update(passthrough_sums)
    out["chunks"] = float(len(responses))
    return out


def load_replay_requests(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("requests", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list or a dict with requests")
    else:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))

    requests_out: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Replay record must be an object, got {type(record).__name__}")
        request = record.get("request", record)
        if "forward_backward_input" not in request:
            continue
        requests_out.append({"request": request, "record": record})
    return requests_out


def patch_request(request: dict[str, Any], args: argparse.Namespace, seq_id: int) -> dict[str, Any]:
    payload = copy.deepcopy(request)
    payload["model_id"] = args.train_model_id
    payload["seq_id"] = int(seq_id)
    loss_params = payload.setdefault("forward_backward_input", {}).setdefault("loss_fn_params", {})
    if args.opd_profile_timings:
        loss_params["opd_profile_timings"] = True
    if args.opd_profile_sync_cuda:
        loss_params["opd_profile_sync_cuda"] = True
    if args.microbatch_diagnostic_dir:
        loss_params["diagnostic_microbatch_dump_dir"] = args.microbatch_diagnostic_dir
    if args.microbatch_diagnostic_tensors:
        loss_params["diagnostic_microbatch_dump_tensors"] = True
    if args.microbatch_diagnostic_summary_only:
        loss_params["diagnostic_microbatch_summary_only"] = True
    return payload


def optim_step(args: argparse.Namespace, seq_id: int) -> dict[str, Any]:
    return call_future(
        args.train_url,
        "/api/v1/optim_step",
        {
            "model_id": args.train_model_id,
            "seq_id": int(seq_id),
            "learning_rate": args.optim_lr,
            "gradient_clip": args.gradient_clip,
        },
        context=f"optim_step({seq_id})",
        future_timeout=args.future_timeout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-url", default="http://127.0.0.1:26040")
    parser.add_argument("--artifact", required=True, help="JSONL/JSON replay artifact from train_opsd_baseline.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--train-model-id", default="default")
    parser.add_argument("--full-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-create-model", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Measured replay repeats.")
    parser.add_argument("--warmup-repeats", type=int, default=1, help="Unmeasured replay repeats before measured repeats.")
    parser.add_argument("--max-requests", type=int, default=0, help="Replay only the first N requests; 0 means all.")
    parser.add_argument("--seq-id-start", type=int, default=0)
    parser.add_argument("--optim-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim-lr", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--future-timeout", type=float, default=7200.0)
    parser.add_argument("--opd-profile-timings", action="store_true")
    parser.add_argument("--opd-profile-sync-cuda", action="store_true")
    parser.add_argument("--microbatch-diagnostic-dir", default="")
    parser.add_argument("--microbatch-diagnostic-tensors", action="store_true")
    parser.add_argument("--microbatch-diagnostic-summary-only", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "anyprecision_adamw", "sgd", "signsgd", "muon"])
    parser.add_argument("--optimizer-dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats must be >= 0")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be >= 0")
    if args.microbatch_diagnostic_summary_only and args.optim_step:
        print("[init] microbatch diagnostic summary-only mode disables optim_step", flush=True)
        args.optim_step = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "replay_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    artifact = Path(args.artifact)
    requests_in = load_replay_requests(artifact)
    if args.max_requests > 0:
        requests_in = requests_in[: args.max_requests]
    if not requests_in:
        raise RuntimeError(f"No forward_backward requests found in {artifact}")

    run_config = {
        **vars(args),
        "artifact": str(artifact),
        "output_dir": str(output_dir),
        "requests": len(requests_in),
    }
    (output_dir / "replay_run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    _jsonl(metrics_path, {"event": "init", "time": time.time(), **run_config})

    print(f"[init] waiting for training server at {args.train_url}", flush=True)
    wait_for_training_service(args.train_url, timeout=args.startup_timeout)
    print("[init] training server is ready", flush=True)

    if not args.skip_create_model:
        create_result = create_model(args)
        _jsonl(metrics_path, {"event": "create_model", "time": time.time(), "result": create_result})

    seq_id = int(args.seq_id_start)
    measured_responses: list[dict[str, Any]] = []
    measured_fb_wall_s = 0.0
    total_repeats = args.warmup_repeats + args.repeat

    for replay_idx in range(total_repeats):
        warmup = replay_idx < args.warmup_repeats
        repeat_idx = replay_idx - args.warmup_repeats if not warmup else replay_idx
        repeat_label = "warmup" if warmup else "measured"
        repeat_responses: list[dict[str, Any]] = []
        repeat_fb_wall_s = 0.0

        print(
            f"[{repeat_label} repeat={repeat_idx}] replaying {len(requests_in)} forward_backward requests",
            flush=True,
        )
        for request_idx, entry in enumerate(requests_in):
            payload = patch_request(entry["request"], args, seq_id)
            started = time.perf_counter()
            response = call_future(
                args.train_url,
                "/api/v1/forward_backward",
                payload,
                context=f"forward_backward({repeat_label} repeat {repeat_idx} request {request_idx})",
                future_timeout=args.future_timeout,
            )
            wall_s = time.perf_counter() - started
            repeat_fb_wall_s += wall_s
            repeat_responses.append(response)
            _jsonl(
                metrics_path,
                {
                    "event": "forward_backward",
                    "time": time.time(),
                    "warmup": warmup,
                    "repeat_idx": repeat_idx,
                    "request_idx": request_idx,
                    "seq_id": seq_id,
                    "wall_s": wall_s,
                    "metrics": response.get("metrics") or {},
                    "source": {
                        key: entry["record"].get(key)
                        for key in ("policy_step", "chunk_idx", "request_idx", "datums", "teacher_cache_path")
                    },
                },
            )
            seq_id += 1

        opt_wall_s = 0.0
        opt_result: dict[str, Any] | None = None
        if args.optim_step:
            started = time.perf_counter()
            opt_result = optim_step(args, seq_id)
            opt_wall_s = time.perf_counter() - started
            seq_id += 1

        summary = summarize_responses(repeat_responses)
        valid_tokens = summary.get("global_valid_tokens", summary.get("valid_tokens", 0.0))
        summary["forward_backward_wall_s"] = repeat_fb_wall_s
        summary["forward_backward_requests"] = float(len(repeat_responses))
        summary["optim_step_wall_s"] = opt_wall_s
        if repeat_fb_wall_s > 0:
            summary["valid_tokens_per_s"] = valid_tokens / repeat_fb_wall_s
            summary["forward_backward_ms_per_1k_valid_tokens"] = repeat_fb_wall_s * 1_000_000.0 / max(valid_tokens, 1.0)
        _jsonl(
            metrics_path,
            {
                "event": "replay_repeat",
                "time": time.time(),
                "warmup": warmup,
                "repeat_idx": repeat_idx,
                "summary": summary,
                "optim_result": opt_result,
            },
        )
        print(
            f"[{repeat_label} repeat={repeat_idx}] fb={repeat_fb_wall_s:.2f}s "
            f"tokens={valid_tokens:.0f} tok/s={summary.get('valid_tokens_per_s', 0.0):.2f} "
            f"optim={opt_wall_s:.2f}s",
            flush=True,
        )

        if not warmup:
            measured_responses.extend(repeat_responses)
            measured_fb_wall_s += repeat_fb_wall_s

    final_summary = summarize_responses(measured_responses)
    final_tokens = final_summary.get("global_valid_tokens", final_summary.get("valid_tokens", 0.0))
    final_summary["forward_backward_wall_s"] = measured_fb_wall_s
    final_summary["forward_backward_requests"] = float(len(measured_responses))
    if measured_fb_wall_s > 0:
        final_summary["valid_tokens_per_s"] = final_tokens / measured_fb_wall_s
        final_summary["forward_backward_ms_per_1k_valid_tokens"] = (
            measured_fb_wall_s * 1_000_000.0 / max(final_tokens, 1.0)
        )
    _jsonl(metrics_path, {"event": "done", "time": time.time(), "summary": final_summary})
    print(
        f"[done] measured_fb={measured_fb_wall_s:.2f}s tokens={final_tokens:.0f} "
        f"tok/s={final_summary.get('valid_tokens_per_s', 0.0):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
