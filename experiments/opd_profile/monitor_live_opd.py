#!/usr/bin/env python3
"""Live health monitor for OPD slot runs.

This is intentionally separate from analyze_slot_profile.py: the analyzer gates
completed control rows, while this script answers whether an unfinished run is
still making progress through the sampler/control path.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(
    "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/"
    "er-opd-q36-35b-slots"
)
DEFAULT_RUN_PREFIX = "together-research/xorl-prefill-time-compute"
DEFAULT_METRICS_URL = "http://er-opd-q36-35b-slots-dispatch:29000/metrics"
METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
QUEUE_RE = re.compile(r"#queue-req:\s*(\d+)")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_profile(arg: str) -> Path:
    if arg == "latest":
        profiles = sorted(glob.glob(str(RESULT_ROOT / "*" / "opd_profile.jsonl")))
        if not profiles:
            raise SystemExit(f"no profiles under {RESULT_ROOT}")
        return Path(profiles[-1])
    path = Path(arg)
    if path.is_dir():
        path = path / "opd_profile.jsonl"
    if not path.exists():
        raise SystemExit(f"profile not found: {path}")
    return path


def _resolve_run_path(run: str) -> str:
    if run.count("/") == 2:
        return run
    if "/" in run:
        raise SystemExit(
            "W&B run must be either a bare run id or entity/project/run_id, got "
            f"{run!r}"
        )
    return f"{DEFAULT_RUN_PREFIX}/{run}"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return dict(LABEL_RE.findall(raw))


def _fetch_text(url: str, timeout_s: float) -> str:
    import requests  # noqa: PLC0415

    metrics_url = url if url.endswith("/metrics") else f"{url.rstrip('/')}/metrics"
    response = requests.get(metrics_url, timeout=max(1.0, timeout_s))
    response.raise_for_status()
    return response.text


def _parse_smg_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {
        "router_requests_total": 0.0,
        "router_responses_total": 0.0,
        "connections_active": 0.0,
        "inflight_age_count_gt_30s": 0.0,
        "inflight_age_count_gt_60s": 0.0,
        "inflight_age_count_gt_300s": 0.0,
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        value = _float(match.group("value"))
        labels = _parse_labels(match.group("labels"))
        if name == "smg_router_requests_total":
            totals["router_requests_total"] += value
        elif name == "smg_router_upstream_responses_total":
            totals["router_responses_total"] += value
        elif name == "smg_http_connections_active":
            totals["connections_active"] += value
        elif name == "smg_http_inflight_request_age_count":
            lower = labels.get("gt", "")
            if lower in {"30", "60", "180", "300", "600", "1200"}:
                totals["inflight_age_count_gt_30s"] += value
            if lower in {"60", "180", "300", "600", "1200"}:
                totals["inflight_age_count_gt_60s"] += value
            if lower in {"300", "600", "1200"}:
                totals["inflight_age_count_gt_300s"] += value
    totals["router_request_response_gap"] = (
        totals["router_requests_total"] - totals["router_responses_total"]
    )
    return totals


def _metrics_snapshot(url: str, timeout_s: float) -> dict[str, float]:
    metrics = _parse_smg_metrics(_fetch_text(url, timeout_s))
    metrics["timestamp_s"] = time.time()
    return metrics


def _wandb_summary(run: str) -> dict[str, Any]:
    try:
        import wandb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"available": False, "error": f"import failed: {exc}"}

    try:
        api = wandb.Api(timeout=30)
        wb_run = api.run(_resolve_run_path(run))
    except Exception as exc:  # pragma: no cover - network/API dependent
        return {"available": False, "error": str(exc)}

    summary = dict(wb_run.summary)
    return {
        "available": True,
        "state": wb_run.state,
        "name": wb_run.name,
        "summary_step": summary.get("step"),
        "summary_eval_accuracy": summary.get("eval/accuracy"),
        "summary_buffer_delta": summary.get("eval/buffer_delta"),
        "summary_answer_select_delta": summary.get("eval/answer_logprob_select_delta"),
        "summary_sync_endpoint_success_count": summary.get("sync_endpoint_success_count"),
        "summary_sampler_balance": summary.get("sampler_worker_success_balance_ratio"),
    }


def _native_log_activity(
    pods: list[str],
    *,
    namespace: str,
    since: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    activity: list[dict[str, Any]] = []
    for pod in pods:
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "logs",
                    pod,
                    "-n",
                    namespace,
                    f"--since={since}",
                    "--tail=200",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, timeout_s),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort live monitor
            activity.append({"pod": pod, "available": False, "error": str(exc)})
            continue
        text = result.stdout or ""
        lines = [line for line in text.splitlines() if line.strip()]
        active_lines = [
            line
            for line in lines
            if "Prefill batch" in line
            or "Decode batch" in line
            or 'POST /generate' in line
            or 'POST /v1/chat/completions' in line
        ]
        queue_values = [int(value) for value in QUEUE_RE.findall(text)]
        activity.append(
            {
                "pod": pod,
                "available": result.returncode == 0,
                "returncode": result.returncode,
                "line_count": len(lines),
                "active_line_count": len(active_lines),
                "active": bool(active_lines),
                "last_queue_req": queue_values[-1] if queue_values else None,
                "max_queue_req": max(queue_values) if queue_values else None,
                "last_active_line": active_lines[-1] if active_lines else "",
                "stderr": (result.stderr or "").strip()[:200],
            }
        )
    return activity


def _profile_summary(profile: Path) -> dict[str, Any]:
    rows = _load_rows(profile)
    latest = rows[-1] if rows else {}
    mtime_s = profile.stat().st_mtime
    control_rows = [
        row for row in rows if _float(row.get("eval/control_allowed_by_start_step")) >= 1.0
    ]
    latest_control = control_rows[-1] if control_rows else {}
    return {
        "profile": str(profile),
        "profile_mtime_s": mtime_s,
        "profile_mtime_age_s": time.time() - mtime_s,
        "profile_rows": len(rows),
        "latest_step": latest.get("step"),
        "latest_eval_accuracy": latest.get("eval/accuracy"),
        "latest_sync_success": latest.get("sync_success"),
        "latest_sync_endpoint_success_count": latest.get("sync_endpoint_success_count"),
        "control_rows": len(control_rows),
        "latest_control_step": latest_control.get("step"),
        "latest_control_n": latest_control.get("eval/control_n"),
        "latest_control_buffer_delta": latest_control.get("eval/buffer_delta"),
        "latest_control_buffer_delta_z": latest_control.get("eval/buffer_delta_z"),
        "latest_control_answer_select_delta": latest_control.get("eval/answer_logprob_select_delta"),
        "latest_control_answer_select_z": latest_control.get("eval/answer_logprob_select_delta_z"),
    }


def _rate(first: dict[str, float], last: dict[str, float], key: str) -> float:
    elapsed = max(last["timestamp_s"] - first["timestamp_s"], 1e-9)
    return (last.get(key, 0.0) - first.get(key, 0.0)) / elapsed


def _verdict(
    summary: dict[str, Any],
    latest_metrics: dict[str, float],
    rates: dict[str, float],
    native_activity: list[dict[str, Any]],
) -> str:
    if latest_metrics.get("inflight_age_count_gt_300s", 0.0) > 0.0:
        return "live_attention_aged_inflight"
    if rates and rates.get("router_responses_per_s", 0.0) <= 0.0 and latest_metrics.get("connections_active", 0.0) > 0.0:
        return "live_attention_no_response_progress"
    if latest_metrics.get("connections_active", 0.0) > 0.0:
        return "live_progress"
    if any(bool(item.get("active")) for item in native_activity):
        return "live_progress_native"
    if _float(summary.get("control_rows")) > 0.0:
        return "profile_control_row_present"
    return "idle_or_waiting"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", nargs="?", default="latest", help="profile JSONL, run dir, or latest")
    parser.add_argument("--wandb-run", help="bare run id or entity/project/run_id")
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument("--metrics-timeout", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=1, help="number of SMG metric samples")
    parser.add_argument("--interval-s", type=float, default=0.0, help="sleep between metric samples")
    parser.add_argument("--k8s-namespace", default="apanda")
    parser.add_argument(
        "--native-log-pod",
        action="append",
        default=[],
        help="native SGLang pod to inspect for recent non-SMG answer-logprob activity; repeatable",
    )
    parser.add_argument("--native-log-since", default="2m")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    profile_info = _profile_summary(profile)
    samples = max(1, int(args.samples))
    snapshots: list[dict[str, float]] = []
    for idx in range(samples):
        snapshots.append(_metrics_snapshot(args.metrics_url, args.metrics_timeout))
        if idx + 1 < samples:
            time.sleep(max(0.0, float(args.interval_s)))
    latest_metrics = snapshots[-1]
    rates: dict[str, float] = {}
    if len(snapshots) >= 2:
        rates = {
            "router_requests_per_s": _rate(snapshots[0], snapshots[-1], "router_requests_total"),
            "router_responses_per_s": _rate(snapshots[0], snapshots[-1], "router_responses_total"),
        }
    native_activity = _native_log_activity(
        args.native_log_pod,
        namespace=args.k8s_namespace,
        since=args.native_log_since,
        timeout_s=args.metrics_timeout,
    )
    wandb_info = _wandb_summary(args.wandb_run) if args.wandb_run else {"available": False}
    out = {
        "now_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        **profile_info,
        "smg": latest_metrics,
        "smg_rates": rates,
        "native_activity": native_activity,
        "wandb": wandb_info,
    }
    out["verdict"] = _verdict(out, latest_metrics, rates, native_activity)

    if args.json:
        print(json.dumps(out, sort_keys=True))
        return

    print(f"now={out['now_utc']}")
    print(f"profile={out['profile']}")
    print(
        "profile_rows={profile_rows} latest_step={latest_step} "
        "mtime_age_s={profile_mtime_age_s:.1f} control_rows={control_rows}".format(**out)
    )
    print(
        "latest_acc={latest_eval_accuracy} sync={latest_sync_success}/"
        "{latest_sync_endpoint_success_count}".format(**out)
    )
    print(
        "smg_requests={router_requests_total:.0f} responses={router_responses_total:.0f} "
        "gap={router_request_response_gap:.0f} active={connections_active:.0f} "
        "aged_gt30={inflight_age_count_gt_30s:.0f} aged_gt300={inflight_age_count_gt_300s:.0f}".format(
            **latest_metrics
        )
    )
    if rates:
        print(
            "smg_rate=requests/s {router_requests_per_s:.2f} "
            "responses/s {router_responses_per_s:.2f}".format(**rates)
        )
    if native_activity:
        active_pods = [str(item.get("pod")) for item in native_activity if item.get("active")]
        print(f"native_active_pods={','.join(active_pods) if active_pods else '-'}")
        queue_bits = [
            f"{item.get('pod')}:{item.get('last_queue_req')}/{item.get('max_queue_req')}"
            for item in native_activity
            if item.get("last_queue_req") is not None
        ]
        if queue_bits:
            print(f"native_queue_req_last_max={','.join(queue_bits)}")
    if wandb_info.get("available"):
        print(
            "wandb_state={state} wandb_step={summary_step} wandb_acc={summary_eval_accuracy} "
            "wandb_buffer_delta={summary_buffer_delta}".format(**wandb_info)
        )
    elif args.wandb_run:
        print(f"wandb_unavailable={wandb_info.get('error')}")
    print(f"VERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()
