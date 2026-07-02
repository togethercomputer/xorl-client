"""Client-side on-policy distillation loop for XoRL.

This example keeps OPD orchestration in xorl-client instead of relying on a
repo-local shell/Python driver. It expects:

1. A XoRL student trainer API at ``base_url``.
2. One or more student sampler endpoints in ``inference_base_urls``.
3. A XoRL teacher prefill API at ``teacher_base_url``.

The loop is intentionally small: sample trajectories from the current student,
prefill those trajectories through the teacher into a shared hidden-state cache,
train the student with ``opd_loss``, optionally optimizer-step, and optionally
sync the trainer weights back to the registered sampler endpoints.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import logging
import math
import numbers
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import chz
import requests
import xorl_client as tomi
from xorl_client.client.training_client import TrainingClient


logging.basicConfig(
    level=os.environ.get("OPD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xorl-client-opd")

# httpx logs every request as INFO ("HTTP/1.1 200"). At scale (thousands of
# sampler calls per step) this drowns out every other signal in the log; the
# 21-step Run B at 08:22 emitted 188,430 of 188,574 lines from httpx alone.
# Silence to WARNING by default; override via OPD_HTTPX_LOG_LEVEL=INFO if you
# need raw request traces.
logging.getLogger("httpx").setLevel(os.environ.get("OPD_HTTPX_LOG_LEVEL", "WARNING"))


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _capture_asset_dir(capture_path: Path) -> Path:
    suffix = capture_path.suffix or ".json"
    return capture_path.with_suffix(f"{suffix}.assets")


def _copy_capture_asset(src: str | Path, capture_path: Path, *, label: str) -> str:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"cannot capture missing forward_backward asset {src_path}")
    dst_dir = _capture_asset_dir(capture_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{label}-{src_path.name}"
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def _write_forward_backward_capture(
    *,
    capture_path: Path,
    config: "Config",
    step: int,
    prepare_batch_idx: int,
    train_batch_idx: int,
    train_microbatch_size: int,
    data: list[dict[str, Any]],
    loss_fn: str,
    loss_fn_params: dict[str, Any],
    prepared: "PreparedOpdBatch",
) -> None:
    captured_loss_params = copy.deepcopy(loss_fn_params)
    cache_map = captured_loss_params.get("teacher_hidden_caches")
    if isinstance(cache_map, dict):
        captured_loss_params["teacher_hidden_caches"] = {
            str(key): _copy_capture_asset(
                value,
                capture_path,
                label=f"step{step}-prep{prepare_batch_idx}-train{train_batch_idx}-teacher{key}",
            )
            for key, value in cache_map.items()
        }
    if prepared.layers_cache_path is not None:
        captured_loss_params["opd_oprd_layers_cache_path"] = _copy_capture_asset(
            prepared.layers_cache_path,
            capture_path,
            label=f"step{step}-prep{prepare_batch_idx}-train{train_batch_idx}-oprd-layers",
        )
        captured_loss_params["teacher_layer_hidden_caches"] = {
            "0": {
                "path": captured_loss_params["opd_oprd_layers_cache_path"],
                "tensor_key": "hidden_states_layers",
            }
        }

    _atomic_write_json(
        capture_path,
        {
            "metadata": {
                "format": "xorl.opd.forward_backward_capture.v1",
                "captured_at_unix_s": time.time(),
                "step": step,
                "prepare_batch_idx": prepare_batch_idx,
                "train_batch_idx": train_batch_idx,
                "model_id": config.model_id,
                "model_name": config.model_name,
                "loss_fn": loss_fn,
                "num_datums": len(data),
                "opd_microbatch_size": config.opd_microbatch_size,
                "opd_prepare_batch_size": config.opd_prepare_batch_size,
                "opd_train_microbatch_size": train_microbatch_size,
                "opd_oprd_layers": config.opd_oprd_layers,
                "opd_oprd_num_layers": config.opd_oprd_num_layers,
                "opd_oprd_student_capture": config.opd_oprd_student_capture,
                "recommended_replay_overrides": {
                    "profile_clear_gradients_after_backward": True,
                    "opd_profile_timings": True,
                },
            },
            "request": {
                "model_id": config.model_id,
                "forward_backward_input": {
                    "data": data,
                    "loss_fn": loss_fn,
                    "loss_fn_params": captured_loss_params,
                },
            },
        },
    )


def _interval_union_s(intervals: list[tuple[float, float]]) -> float:
    """Total wall-clock seconds covered by a set of [t0, t1] perf_counter intervals.

    Concurrent prepare batches overlap, so summing per-batch phase durations
    (`student_sampling_s` / `teacher_prefill_s`) wildly overstates the real wall
    (a sum can exceed step_total). This returns the union length — the actual
    wall during which ANY batch was in that phase. All inputs share the process
    perf_counter clock, so they are directly comparable.
    """
    spans = sorted((t0, t1) for t0, t1 in intervals if t1 > t0)
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for t0, t1 in spans[1:]:
        if t0 > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = t0, t1
        else:
            cur_end = max(cur_end, t1)
    total += cur_end - cur_start
    return total


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_p95_max(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(float(value) for value in values)
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return sum(ordered) / len(ordered), ordered[p95_idx], ordered[-1]


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_token_id_list(value: str) -> list[int]:
    """Parse a CLI-friendly token-id list.

    Accepts JSON lists (``"[1,2,3]"``) or comma/whitespace separated IDs
    (``"1,2,3"`` / ``"1 2 3"``). The empty string means "disabled".
    """
    raw = value.strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("token-id JSON must be a list")
        items = parsed
    else:
        items = [part for part in re.split(r"[\s,]+", raw) if part]
    ids: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"token IDs must be integers, got {item!r}")
        try:
            token_id = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token IDs must be integers, got {item!r}") from exc
        if token_id < 0:
            raise ValueError(f"token IDs must be non-negative, got {token_id}")
        ids.append(token_id)
    return ids


def _parse_stop_sequences(value: str) -> list[str]:
    """Parse stop sequences from JSON or a pipe-separated string."""
    raw = value.strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("stop-sequence JSON must be a list")
        items = parsed
    else:
        items = raw.split("|")
    stops: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"stop sequences must be strings, got {item!r}")
        if item:
            stops.append(item)
    return stops


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0 or chunk_size >= len(items):
        return [list(items)]
    return [
        list(items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    ]


def get_inference_urls(inference_base_urls: str, inference_port: int) -> list[str]:
    """Parse comma-separated inference URLs or research-common node suffixes."""
    urls: list[str] = []
    for part in _split_csv(inference_base_urls):
        if part.startswith("http://") or part.startswith("https://"):
            urls.append(part.rstrip("/"))
        else:
            urls.append(f"http://research-common-{part}:{inference_port}")
    if not urls:
        raise ValueError(
            "inference_base_urls must contain at least one URL or host suffix"
        )
    return urls


def get_sampler_metrics_urls(
    sampler_metrics_urls: str,
    inference_urls: list[str],
    metrics_port: int,
) -> list[str]:
    """Parse explicit SMG metrics URLs. Use ``auto``/``infer`` to infer :metrics_port."""
    explicit = _split_csv(sampler_metrics_urls)
    if not explicit:
        return []
    if len(explicit) == 1 and explicit[0].strip().lower() in {"auto", "infer"}:
        explicit = []
    else:
        return [url.rstrip("/") for url in explicit]

    urls: list[str] = []
    for inference_url in inference_urls:
        parsed = urlsplit(inference_url)
        if not parsed.scheme or not parsed.hostname:
            continue
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = f"{hostname}:{metrics_port}"
        urls.append(urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/"))
    return urls


def _parse_prometheus_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    idx = 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " ,":
            idx += 1
        start = idx
        while idx < len(raw) and raw[idx] != "=":
            idx += 1
        if idx >= len(raw):
            break
        key = raw[start:idx].strip()
        idx += 1
        if idx >= len(raw) or raw[idx] != '"':
            break
        idx += 1
        value_chars: list[str] = []
        while idx < len(raw):
            char = raw[idx]
            if char == "\\" and idx + 1 < len(raw):
                value_chars.append(raw[idx + 1])
                idx += 2
                continue
            if char == '"':
                idx += 1
                break
            value_chars.append(char)
            idx += 1
        if key:
            labels[key] = "".join(value_chars)
        while idx < len(raw) and raw[idx] not in ",":
            idx += 1
        if idx < len(raw) and raw[idx] == ",":
            idx += 1
    return labels


def _iter_prometheus_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        metric_part, _, value_part = line.partition(" ")
        if not value_part:
            continue
        try:
            value = float(value_part.split()[0])
        except (IndexError, ValueError):
            continue
        if "{" in metric_part and metric_part.endswith("}"):
            name, raw_labels = metric_part.split("{", 1)
            labels = _parse_prometheus_labels(raw_labels[:-1])
        else:
            name = metric_part
            labels = {}
        samples.append((name, labels, value))
    return samples


def _parse_smg_metrics(text: str) -> dict[str, Any]:
    workers: dict[str, dict[str, float]] = {}
    router_requests_total = 0.0
    router_upstream_responses_total = 0.0
    http_chat_responses_total = 0.0
    http_connections_active = 0.0
    http_inflight_request_age_count = 0.0
    worker_selection_total = 0.0
    worker_pool_size = 0.0
    selection_policies: dict[str, float] = {}

    for name, labels, value in _iter_prometheus_samples(text):
        if name == "smg_worker_cb_outcomes_total":
            worker = labels.get("worker")
            outcome = labels.get("outcome", "unknown")
            if worker:
                workers.setdefault(worker, {})[f"cb_{outcome}"] = value
        elif name == "smg_worker_health":
            worker = labels.get("worker")
            if worker:
                workers.setdefault(worker, {})["health"] = value
        elif name == "smg_router_requests_total":
            router_requests_total += value
        elif name == "smg_router_upstream_responses_total":
            router_upstream_responses_total += value
        elif name == "smg_http_responses_total":
            if labels.get("path") == "/v1/chat/completions":
                http_chat_responses_total += value
        elif name == "smg_http_connections_active":
            http_connections_active = max(http_connections_active, value)
        elif name == "smg_http_inflight_request_age_count":
            http_inflight_request_age_count += value
        elif name == "smg_worker_selection_total":
            worker_selection_total += value
            policy = labels.get("policy")
            if policy:
                selection_policies[policy] = selection_policies.get(policy, 0.0) + value
        elif name == "smg_worker_pool_size":
            worker_pool_size = max(worker_pool_size, value)

    return {
        "workers": workers,
        "router_requests_total": router_requests_total,
        "router_upstream_responses_total": router_upstream_responses_total,
        "http_chat_responses_total": http_chat_responses_total,
        "http_connections_active": http_connections_active,
        "http_inflight_request_age_count": http_inflight_request_age_count,
        "worker_selection_total": worker_selection_total,
        "worker_pool_size": worker_pool_size,
        "selection_policies": selection_policies,
    }


def _fetch_sampler_metrics(urls: list[str], timeout: float) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for base_url in urls:
        metrics_url = base_url.rstrip("/")
        if not metrics_url.endswith("/metrics"):
            metrics_url = f"{metrics_url}/metrics"
        try:
            response = requests.get(metrics_url, timeout=timeout)
            response.raise_for_status()
            snapshots[base_url] = _parse_smg_metrics(response.text)
        except Exception as exc:  # noqa: BLE001
            errors[base_url] = str(exc)
    return {"snapshots": snapshots, "errors": errors}


def _sampler_metrics_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None and after is None:
        return {"sampler_metrics_available": 0.0, "sampler_metrics_url_count": 0}

    before_snapshots = (before or {}).get("snapshots", {})
    after_snapshots = (after or {}).get("snapshots", {})
    before_errors = (before or {}).get("errors", {})
    after_errors = (after or {}).get("errors", {})
    urls = sorted(set(before_snapshots) | set(after_snapshots) | set(before_errors) | set(after_errors))
    worker_deltas: dict[str, float] = {}
    worker_health: dict[str, float] = {}
    router_delta = 0.0
    router_upstream_response_delta = 0.0
    http_chat_response_delta = 0.0
    http_connections_active = 0.0
    http_inflight_request_age_count = 0.0
    selection_delta = 0.0
    worker_pool_size = 0.0
    policy_deltas: dict[str, float] = {}

    for url in urls:
        before_snapshot = before_snapshots.get(url, {})
        after_snapshot = after_snapshots.get(url, {})
        router_delta += float(after_snapshot.get("router_requests_total", 0.0)) - float(
            before_snapshot.get("router_requests_total", 0.0)
        )
        router_upstream_response_delta += float(after_snapshot.get("router_upstream_responses_total", 0.0)) - float(
            before_snapshot.get("router_upstream_responses_total", 0.0)
        )
        http_chat_response_delta += float(after_snapshot.get("http_chat_responses_total", 0.0)) - float(
            before_snapshot.get("http_chat_responses_total", 0.0)
        )
        http_connections_active += float(after_snapshot.get("http_connections_active", 0.0))
        http_inflight_request_age_count += float(after_snapshot.get("http_inflight_request_age_count", 0.0))
        selection_delta += float(after_snapshot.get("worker_selection_total", 0.0)) - float(
            before_snapshot.get("worker_selection_total", 0.0)
        )
        worker_pool_size = max(worker_pool_size, float(after_snapshot.get("worker_pool_size", 0.0)))

        before_policies = before_snapshot.get("selection_policies", {}) or {}
        after_policies = after_snapshot.get("selection_policies", {}) or {}
        for policy in set(before_policies) | set(after_policies):
            policy_deltas[policy] = policy_deltas.get(policy, 0.0) + (
                float(after_policies.get(policy, 0.0)) - float(before_policies.get(policy, 0.0))
            )

        before_workers = before_snapshot.get("workers", {}) or {}
        after_workers = after_snapshot.get("workers", {}) or {}
        for worker in set(before_workers) | set(after_workers):
            before_success = float((before_workers.get(worker, {}) or {}).get("cb_success", 0.0))
            after_success = float((after_workers.get(worker, {}) or {}).get("cb_success", 0.0))
            worker_deltas[worker] = worker_deltas.get(worker, 0.0) + max(after_success - before_success, 0.0)
            if worker in after_workers:
                worker_health[worker] = float((after_workers.get(worker, {}) or {}).get("health", 0.0))

    success_values = [value for value in worker_deltas.values() if value >= 0.0]
    active_values = [value for value in success_values if value > 0.0]
    max_success = max(success_values) if success_values else 0.0
    min_success = min(success_values) if success_values else 0.0
    metrics: dict[str, Any] = {
        "sampler_metrics_available": float(bool(after_snapshots)),
        "sampler_metrics_url_count": len(urls),
        "sampler_metrics_error_count": len(before_errors) + len(after_errors),
        "sampler_router_requests_delta": router_delta,
        "sampler_router_upstream_responses_delta": router_upstream_response_delta,
        "sampler_http_chat_responses_delta": http_chat_response_delta,
        "sampler_new_outstanding_requests": max(router_delta - router_upstream_response_delta, 0.0),
        "sampler_http_connections_active": http_connections_active,
        "sampler_http_inflight_request_age_count": http_inflight_request_age_count,
        "sampler_worker_selection_delta": selection_delta,
        "sampler_worker_pool_size": worker_pool_size,
        "sampler_worker_count": len(worker_deltas),
        "sampler_worker_active_count": len(active_values),
        "sampler_worker_success_delta_total": sum(success_values),
        "sampler_worker_success_delta_min": min_success,
        "sampler_worker_success_delta_max": max_success,
        "sampler_worker_success_balance_ratio": (min_success / max_success if max_success > 0 else 0.0),
        "sampler_worker_health_min": min(worker_health.values()) if worker_health else 0.0,
        "sampler_policy_round_robin_delta": policy_deltas.get("round_robin", 0.0),
        "sampler_policy_round_robin_active": float(policy_deltas.get("round_robin", 0.0) > 0.0),
    }
    for idx, (worker, delta) in enumerate(sorted(worker_deltas.items())):
        metrics[f"sampler_worker_{idx}_success_delta"] = delta
        metrics[f"sampler_worker_{idx}_url"] = worker
        if worker in worker_health:
            metrics[f"sampler_worker_{idx}_health"] = worker_health[worker]
    if after_errors:
        first_url = sorted(after_errors)[0]
        metrics["sampler_metrics_first_error_url"] = first_url
        metrics["sampler_metrics_first_error"] = after_errors[first_url]
    return metrics


def _sampler_quiescence_metrics(
    baseline: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, float]:
    baseline_snapshots = (baseline or {}).get("snapshots", {}) or {}
    current_snapshots = (current or {}).get("snapshots", {}) or {}
    current_errors = (current or {}).get("errors", {}) or {}
    urls = sorted(set(baseline_snapshots) | set(current_snapshots) | set(current_errors))

    new_requests = 0.0
    new_responses = 0.0
    absolute_outstanding = 0.0
    connections_active = 0.0
    inflight_age_count = 0.0

    for url in urls:
        before_snapshot = baseline_snapshots.get(url, {}) or {}
        after_snapshot = current_snapshots.get(url, {}) or {}
        before_requests = float(before_snapshot.get("router_requests_total", 0.0))
        after_requests = float(after_snapshot.get("router_requests_total", 0.0))
        before_responses = float(before_snapshot.get("router_upstream_responses_total", 0.0))
        after_responses = float(after_snapshot.get("router_upstream_responses_total", 0.0))
        new_requests += max(after_requests - before_requests, 0.0)
        new_responses += max(after_responses - before_responses, 0.0)
        absolute_outstanding += max(after_requests - after_responses, 0.0)
        connections_active += float(after_snapshot.get("http_connections_active", 0.0))
        inflight_age_count += float(after_snapshot.get("http_inflight_request_age_count", 0.0))

    return {
        "sampler_quiesce_available": float(bool(current_snapshots)),
        "sampler_quiesce_url_count": float(len(urls)),
        "sampler_quiesce_error_count": float(len(current_errors)),
        "sampler_quiesce_new_requests": new_requests,
        "sampler_quiesce_new_responses": new_responses,
        "sampler_quiesce_new_outstanding": max(new_requests - new_responses, 0.0),
        "sampler_quiesce_absolute_outstanding": absolute_outstanding,
        "sampler_quiesce_connections_active": connections_active,
        "sampler_quiesce_inflight_request_age_count": inflight_age_count,
    }


def _wait_for_sampler_quiescence(
    urls: list[str],
    baseline: dict[str, Any] | None,
    *,
    metrics_timeout: float,
    timeout_s: float,
    poll_s: float,
    max_new_outstanding: float,
    max_connections_active: float,
    max_inflight: float,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    start = time.perf_counter()
    deadline = start + max(timeout_s, 0.0)
    poll_interval = max(poll_s, 0.05)
    polls = 0
    last_snapshot: dict[str, Any] | None = None
    last_metrics: dict[str, float] = {
        "sampler_quiesce_enabled": 1.0,
        "sampler_quiesce_available": 0.0,
        "sampler_quiesce_success": 0.0,
        "sampler_quiesce_wait_s": 0.0,
        "sampler_quiesce_poll_count": 0.0,
    }

    if not urls:
        last_metrics.update(
            {
                "sampler_quiesce_wait_s": 0.0,
                "sampler_quiesce_failure_reason": 1.0,
            }
        )
        return last_metrics, None

    while True:
        polls += 1
        last_snapshot = _fetch_sampler_metrics(urls, metrics_timeout)
        last_metrics = {
            "sampler_quiesce_enabled": 1.0,
            **_sampler_quiescence_metrics(baseline, last_snapshot),
            "sampler_quiesce_wait_s": time.perf_counter() - start,
            "sampler_quiesce_poll_count": float(polls),
        }
        ready = (
            last_metrics["sampler_quiesce_available"] >= 1.0
            and last_metrics["sampler_quiesce_new_outstanding"] <= max_new_outstanding
            and last_metrics["sampler_quiesce_connections_active"] <= max_connections_active
            and last_metrics["sampler_quiesce_inflight_request_age_count"] <= max_inflight
        )
        last_metrics["sampler_quiesce_success"] = float(ready)
        if ready or time.perf_counter() >= deadline:
            return last_metrics, last_snapshot
        time.sleep(poll_interval)


def _wait_for_future(base_url: str, request_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.post(
            f"{base_url}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") == "try_again":
            time.sleep(0.5)
            continue
        if payload.get("type") == "request_failed" or payload.get("error"):
            raise RuntimeError(f"Future {request_id} failed: {payload}")
        return payload
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def _wait_for_xorl(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200 and response.json().get("engine_running"):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"XoRL server at {base_url} not healthy within {timeout}s")


def _ensure_xorl_session(
    base_url: str,
    *,
    model_id: str,
    base_model: str,
    timeout: float,
) -> None:
    """Register model_id for direct /api/v1 calls on a XORL server."""
    response = requests.post(
        f"{base_url}/api/v1/create_session",
        json={"session_id": model_id, "base_model": base_model},
        timeout=min(timeout, 60.0),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Failed to register XORL session {model_id!r} at {base_url}: "
            f"{response.status_code} {response.text}"
        ) from exc
    logger.info(
        "Registered XORL session: base_url=%s model_id=%s response=%s",
        base_url,
        model_id,
        response.text,
    )


def _wait_for_sglang(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server at {base_url} not healthy within {timeout}s")


def _opd_causal_pair(sequence: list[int]) -> tuple[list[int], list[int]]:
    if len(sequence) < 2:
        raise ValueError(
            f"OPD trajectory must contain at least two tokens, got {len(sequence)}"
        )
    return list(sequence[:-1]), list(sequence[1:])


def _teacher_hidden_cache_data(
    sequences: list[list[int]],
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | list[list[int]] | None = None,
    prompt_token_lens: list[int] | None = None,
    student_filler_count: int = 0,
    teacher_cot_mode: str = "replace",
    supervise_student_cot: bool = False,
    mask_answer: bool = False,
    per_sample_filler_count: list[int] | None = None,
    return_teacher_seqs: bool = False,
) -> list[dict[str, Any]]:
    """Build the teacher forward request payload.

    ``teacher_cot_mode`` controls how the teacher's CoT relates to the student's
    pause filler region (length K = student_filler_count at position p):
      * ``"replace"`` (default): teacher_seq = prompt + CoT + answer — the CoT
        takes the place of the student's pause. Answer sits right after CoT.
      * ``"insert"``: teacher_seq = prompt + CoT + pause + answer — the CoT is
        prepended and the student's pause is KEPT, so the answer's immediate
        left-context (the pause buffer) is IDENTICAL to the student's; the only
        difference is the prepended CoT. Cleaner ablation for distillation.
    Either way only prompt + answer positions are kept (CoT and pause masked),
    so the returned cache row count is (p-1)+ans — unchanged for the student
    remap.

    Three ways to extend the teacher input beyond the bare student sequence:

    1. ``teacher_prefix_tokens`` — prepended to the very start of every teacher
       input_ids. Their target positions are set to ``-100`` (IGNORE_INDEX) so
       they don't appear in the returned cache. This is a "system-instruction
       prefix" injection — the teacher reads the prefix BEFORE the user prompt.

    2. ``teacher_filler_tokens`` + ``prompt_token_lens`` — inserted AT the
       user→assistant boundary of each sample. Two forms accepted:
         * ``list[int]``: a single global filler shared by every sample
           (Run A: 100× " pause").
         * ``list[list[int]]``: per-sample filler, length-matched to sequences
           (Run B: per-prompt precomputed CoT tokens).
       Either way, the K filler-predicting target positions are masked with
       ``-100`` so they don't appear in the cache.

    3. ``student_filler_count`` — when > 0, the *student* sequence already
       contains K student-filler tokens (e.g. pause prefill) starting at
       position ``prompt_token_lens[idx]``. The teacher reconstruction REPLACES
       those K positions with ``teacher_filler_tokens[idx]`` rather than
       appending. Final teacher_seq layout per sample:
           student_seq[:p] + teacher_filler_tokens[idx] + student_seq[p + K:]
       This matches Run B: the student sees ``pause × K`` between user and
       answer; the teacher sees per-prompt CoT in the same slot.

    The xorl trainer's ``_split_hidden_cache_rows`` filters hidden states by
    ``target_tokens != IGNORE_INDEX``, so the resulting
    ``cache_indices_by_sample`` returned by the server is a contiguous
    ``range(num_kept)`` per sample regardless of the filler length.
    """
    prefix = list(teacher_prefix_tokens) if teacher_prefix_tokens else []

    per_sample_filler: list[list[int]] | None = None
    if teacher_filler_tokens:
        first = teacher_filler_tokens[0] if teacher_filler_tokens else None
        if isinstance(first, list):
            if len(teacher_filler_tokens) != len(sequences):
                raise ValueError(
                    f"teacher_filler_tokens has {len(teacher_filler_tokens)} entries for {len(sequences)} sequences"
                )
            per_sample_filler = [list(f) for f in teacher_filler_tokens]
        else:
            per_sample_filler = [list(teacher_filler_tokens) for _ in sequences]

    use_filler_insert = per_sample_filler is not None
    if use_filler_insert and not prompt_token_lens:
        raise ValueError(
            "teacher_filler_tokens requires prompt_token_lens for per-sample boundary insertion"
        )
    if use_filler_insert and len(prompt_token_lens) != len(sequences):
        raise ValueError(
            f"prompt_token_lens has {len(prompt_token_lens)} entries for {len(sequences)} sequences"
        )
    if student_filler_count and not use_filler_insert:
        raise ValueError(
            "student_filler_count > 0 requires teacher_filler_tokens (need replacement content)"
        )
    # "buffer = CoT length (K=C)": K_student is per-sample (K_i ~= C_i) instead of
    # the global scalar. Requires the filler-insert path (per-sample CoT content).
    if per_sample_filler_count is not None:
        if not use_filler_insert:
            raise ValueError(
                "per_sample_filler_count requires teacher_filler_tokens (need replacement content)"
            )
        if len(per_sample_filler_count) != len(sequences):
            raise ValueError(
                f"per_sample_filler_count has {len(per_sample_filler_count)} entries "
                f"for {len(sequences)} sequences"
            )

    data: list[dict[str, Any]] = []
    for idx, sequence in enumerate(sequences):
        input_ids, target_tokens = _opd_causal_pair(sequence)

        if use_filler_insert:
            # Build teacher_seq by substituting student's filler (K=student_filler_count
            # tokens at position p) with the per-sample teacher filler, OR by inserting
            # at position p when the student has no filler. The K filler-predicting
            # positions of teacher_target are masked with -100 so they don't appear
            # in the cache.
            p = int(prompt_token_lens[idx])
            sample_filler = per_sample_filler[idx]
            C = len(sample_filler)
            K_student = (
                int(per_sample_filler_count[idx])
                if per_sample_filler_count is not None
                else int(student_filler_count)
            )
            if teacher_cot_mode in ("insert", "match_cot"):
                # prompt + CoT + pause(kept) + answer
                full_teacher_seq = (
                    list(sequence[:p])
                    + sample_filler
                    + list(sequence[p:p + K_student])
                    + list(sequence[p + K_student:])
                )
                if teacher_cot_mode == "match_cot":
                    # Per-position CoT distillation: KEEP the first K_student teacher
                    # CoT positions so the student's K filler hiddens align 1:1 with
                    # the teacher's CoT-i hiddens via the hidden match; mask CoT[K:C]
                    # and the pause. Kept rows = prompt + CoT[0:K] + answer =
                    # (p-1)+K+ans — IDENTICAL count to the pause-kept supervise path,
                    # so the student-side identity remap is unchanged. Requires
                    # K_student <= C and supervise_student_cot=true.
                    if K_student > C:
                        if per_sample_filler_count is not None:
                            # K=C mode: the forced prefill's suffix can make
                            # len(prefill) > len(cot) for very short CoTs. Clamp K to C
                            # (the student aligns its first C buffer positions to the
                            # CoT; the trailing prefill positions fall in the kept tail)
                            # rather than aborting the run.
                            import logging as _logging  # noqa: PLC0415
                            _logging.getLogger("xorl-client-opd").warning(
                                "match_cot K=C: student filler %d > teacher CoT %d for "
                                "sample %d; clamping K_student=C (suffix overshoot on a "
                                "short CoT).",
                                K_student,
                                C,
                                idx,
                            )
                            K_student = C
                        else:
                            raise ValueError(
                                f"teacher_cot_mode='match_cot' requires student_filler_count "
                                f"({K_student}) <= teacher CoT length ({C})"
                            )
                    mask_start = max(0, p - 1) + K_student
                    mask_len = C
                else:
                    # Always mask the teacher-CoT region (the student never produces
                    # it). Mask the pause region too UNLESS supervising it: when
                    # supervise_student_cot is set, the pause hiddens are KEPT in the
                    # cache so the student's pause positions can be distilled against
                    # the teacher's post-CoT pause states.
                    mask_start = max(0, p - 1)
                    mask_len = C if supervise_student_cot else C + K_student
            else:  # "replace": prompt + CoT + answer (CoT in place of pause)
                full_teacher_seq = (
                    list(sequence[:p])
                    + sample_filler
                    + list(sequence[p + K_student:])
                )
                mask_start = max(0, p - 1)
                mask_len = C
            input_ids = full_teacher_seq[:-1]
            target_tokens = list(full_teacher_seq[1:])
            # Mask the filler-predicting positions [mask_start, mask_start+mask_len)
            # so only prompt + answer (+ any kept CoT/pause) positions remain in
            # the returned cache.
            for j in range(mask_start, mask_start + mask_len):
                if 0 <= j < len(target_tokens):
                    target_tokens[j] = -100
            if mask_answer and teacher_cot_mode == "insert":
                # Buffer-only supervision: ALSO mask the answer positions so ONLY the
                # K_student buffer positions are kept in the cache. Forces the student's
                # BUFFER hiddens (not its answer) to carry the teacher's CoT-conditioned
                # reasoning — the one setup that can't internalize the skill via the
                # answer. Pairs with opd_hidden_match_coef>0. (insert mode only; replace
                # has no buffer.)
                for j in range(max(0, p - 1) + C + K_student, len(target_tokens)):
                    if 0 <= j < len(target_tokens):
                        target_tokens[j] = -100

        if prefix:
            input_ids = prefix + input_ids
            target_tokens = [-100] * len(prefix) + target_tokens

        row: dict[str, Any] = {
            "model_input": {"input_ids": input_ids},
            "loss_fn_inputs": {"target_tokens": target_tokens},
        }
        if return_teacher_seqs:
            # Multi-layer OPRD trainer-side teacher forward: ship the teacher input_ids
            # (== the request input_ids above) and the KEPT positions (target != -100,
            # in ascending order) so the trainer can run its own no-grad forward and
            # gather exactly the positions the rank-2 KL cache kept, in the same order.
            row["teacher_input_ids"] = list(input_ids)
            row["teacher_kept_indices"] = [j for j, t in enumerate(target_tokens) if t != -100]
        data.append(row)
    return data


def _opd_loss_data(
    sequences: list[list[int]],
    cache_indices: list[list[int]],
    prompt_token_lens: list[int] | None = None,
    student_filler_count: int = 0,
    supervise_student_cot: bool = False,
    mask_answer: bool = False,
    mask_prompt_positions: bool = False,
    teacher_weights_by_sample: list[list[float]] | None = None,
    hidden_match_weights_by_sample: list[list[float]] | None = None,
    old_logprobs_by_sample: list[list[float]] | None = None,
    mask_zero_weight_positions: bool = False,
    per_sample_filler_count: list[int] | None = None,
    teacher_input_ids_by_sample: list[list[int]] | None = None,
    teacher_kept_indices_by_sample: list[list[int]] | None = None,
    sample_ok_by_sample: list[int] | None = None,
    correct_prefix_only: bool = False,
) -> list[dict[str, Any]]:
    """Build student-side OPD loss request payload.

    Default behavior (Run A, ``student_filler_count == 0``): cache_indices are
    used as-is and must have length ``len(sequence) - 1`` per sample. The
    student input is just ``sequence[:-1]`` with no extra masking.

    Run B (``student_filler_count > 0``): the student sequence is
    ``prompt + student_filler + answer`` (length ``p + K + ans``). The teacher's
    cache only contains ``p - 1 + ans`` rows (no rows for the student's filler
    positions, since the teacher's filler was replaced by its own CoT and got
    masked out). We need to:

      1. Mask the K student-filler-predicting target positions [p-1, p+K-1) so
         the trainer's ``valid_mask = labels != IGNORE_INDEX`` skips them.
      2. Build a cache_indices array of length ``len(input_ids)`` that maps the
         remaining positions to the cache rows that came back from the server:
            * positions [0, p-1)        → cache rows [0, p-1)        (prompt)
            * positions [p-1, p+K-1)    → placeholder 0 (will be masked)
            * positions [p+K-1, L_s)    → cache rows [p-1, p-1+ans)  (answer)
       where ``L_s = p + K + ans - 1``.

    The cache_indices passed in are the server-returned natural ranges (length
    ``p - 1 + ans``); we re-map them per-position for the student.
    """
    if len(cache_indices) != len(sequences):
        raise RuntimeError(
            f"Got {len(cache_indices)} cache-index lists for {len(sequences)} sequences"
        )
    if teacher_weights_by_sample is not None and len(teacher_weights_by_sample) != len(sequences):
        raise RuntimeError(
            f"Got {len(teacher_weights_by_sample)} teacher-weight lists for {len(sequences)} sequences"
        )
    if hidden_match_weights_by_sample is not None and len(hidden_match_weights_by_sample) != len(sequences):
        raise RuntimeError(
            f"Got {len(hidden_match_weights_by_sample)} hidden-match-weight lists for {len(sequences)} sequences"
        )
    if old_logprobs_by_sample is not None and len(old_logprobs_by_sample) != len(sequences):
        raise RuntimeError(
            f"Got {len(old_logprobs_by_sample)} old-logprob lists for {len(sequences)} sequences"
        )

    if per_sample_filler_count is not None and len(per_sample_filler_count) != len(sequences):
        raise ValueError(
            f"per_sample_filler_count has {len(per_sample_filler_count)} entries "
            f"for {len(sequences)} sequences"
        )
    if sample_ok_by_sample is not None and len(sample_ok_by_sample) != len(sequences):
        raise RuntimeError(
            f"Got {len(sample_ok_by_sample)} sample-correctness flags for {len(sequences)} sequences"
        )
    oprd_trainer_forward = teacher_input_ids_by_sample is not None
    if oprd_trainer_forward:
        if teacher_kept_indices_by_sample is None:
            raise ValueError("teacher_input_ids_by_sample requires teacher_kept_indices_by_sample")
        if len(teacher_input_ids_by_sample) != len(sequences) or len(
            teacher_kept_indices_by_sample
        ) != len(sequences):
            raise RuntimeError(
                f"OPRD teacher seqs ({len(teacher_input_ids_by_sample)} ids / "
                f"{len(teacher_kept_indices_by_sample)} kept) misaligned with {len(sequences)} sequences"
            )
    # In the global path the remap is gated on a positive scalar K; with a
    # per-sample K_i list ("buffer = CoT length") the remap is always on (each
    # sample carries its own K_i >= 0).
    use_remap = (per_sample_filler_count is not None) or int(student_filler_count) > 0
    if use_remap and prompt_token_lens is None:
        raise ValueError(
            "student_filler_count > 0 requires prompt_token_lens for per-sample remap"
        )
    if use_remap and len(prompt_token_lens) != len(sequences):
        raise ValueError(
            f"prompt_token_lens has {len(prompt_token_lens)} entries for {len(sequences)} sequences"
        )

    data: list[dict[str, Any]] = []
    for idx, (sequence, indices) in enumerate(zip(sequences, cache_indices)):
        input_ids, target_tokens = _opd_causal_pair(sequence)
        L_s = len(input_ids)
        # Per-sample filler region length: K_i when provided, else the global scalar.
        K = (
            int(per_sample_filler_count[idx])
            if per_sample_filler_count is not None
            else int(student_filler_count)
        )

        if not use_remap:
            if len(indices) != L_s:
                raise RuntimeError(
                    f"Teacher cache index length {len(indices)} does not match OPD input length {L_s}"
                )
            remapped_indices = indices
        else:
            p = int(prompt_token_lens[idx])
            ans = L_s - (p - 1) - K  # = len(sequence) - p - K = answer-token count
            if p < 1:
                raise RuntimeError(f"Sample {idx}: prompt_token_lens={p} too small for remap")
            if ans < 1:
                raise RuntimeError(
                    f"Sample {idx}: derived answer length {ans} from L_s={L_s} p={p} K={K}"
                )
            if supervise_student_cot and mask_answer:
                # Buffer-only: the teacher cache masked BOTH the CoT and the answer,
                # so it keeps only prompt+buffer = (p-1)+K rows. Align prompt+buffer
                # 1:1 and MASK the student's answer positions (no answer supervision →
                # the model cannot internalize the skill via the answer; it must encode
                # into the buffer to satisfy the hidden-match). Pairs with mask_answer
                # on the teacher side + opd_hidden_match_coef>0.
                expected_cache_len = (p - 1) + K
                if len(indices) != expected_cache_len:
                    raise RuntimeError(
                        f"Sample {idx}: teacher cache rows {len(indices)} != expected "
                        f"{expected_cache_len} (buffer-only: p={p}, K={K})"
                    )
                # GLOBAL rows at supervised positions, 0 at masked positions — the
                # KL hidden-fetch consumes these against the GLOBAL teacher cache, so
                # they must NOT be localized (the 06-12 bisect: local rows silently
                # mistrained every masked-K run). The OPRD kept-row gather instead
                # uses the packer-emitted local view, re-based with the explicit
                # teacher_cache_base sent below (min()-inference breaks here because
                # the masked 0-filler makes min()==0 — the ARITH-014 step-0 OOB; see
                # docs/notes/oprd_warm_cache_indices_rebase_bug.md).
                remapped_indices = [0] * L_s
                for j in range(p - 1 + K):
                    remapped_indices[j] = indices[j]
                target_tokens = list(target_tokens)
                for j in range(p - 1 + K, L_s):
                    if 0 <= j < len(target_tokens):
                        target_tokens[j] = -100
            elif supervise_student_cot:
                # Pause-position-supervised variant. The teacher cache KEEPS the
                # pause rows (mask_len=C in _teacher_hidden_cache_data), so the
                # student's input layout (prompt + pause + answer) matches the
                # kept teacher positions 1:1 — the remap is identity and NOTHING
                # is masked → KL is computed on student-CoT (pause) + answer.
                expected_cache_len = (p - 1) + K + ans
                if len(indices) != expected_cache_len:
                    raise RuntimeError(
                        f"Sample {idx}: teacher cache rows {len(indices)} != expected "
                        f"{expected_cache_len} (p={p}, K={K}, ans={ans}); supervise_student_cot "
                        f"requires teacher_cot_mode='insert' with the pause kept (mask_len=C)"
                    )
                remapped_indices = list(indices)
                target_tokens = list(target_tokens)  # no masking: pause is supervised
            else:
                expected_cache_len = (p - 1) + ans
                if len(indices) != expected_cache_len:
                    raise RuntimeError(
                        f"Sample {idx}: teacher cache rows {len(indices)} != expected {expected_cache_len} "
                        f"(p={p}, ans={ans}, K={K})"
                    )
                # Build remapped cache_indices of length L_s, as GLOBAL rows at
                # supervised positions and 0 at the masked filler positions (see the
                # buffer-only branch comment: globals feed the KL hidden-fetch; the
                # OPRD gather's local view comes from teacher_cache_base + packer).
                remapped_indices = [0] * L_s
                for j in range(p - 1):
                    remapped_indices[j] = indices[j]
                for j in range(p - 1 + K, L_s):
                    remapped_indices[j] = indices[(p - 1) + (j - (p - 1 + K))]
                # Mask the K student-filler-predicting target positions.
                mask_start = p - 1
                mask_end = mask_start + K
                target_tokens = list(target_tokens)
                for j in range(mask_start, mask_end):
                    if 0 <= j < len(target_tokens):
                        target_tokens[j] = -100

        if mask_prompt_positions and prompt_token_lens is not None:
            # Reference-OPD semantics (verl GKD: calc_kl_mask zeroes everything
            # before the response): drop the prompt-position KL — region 0 of
            # `opd_region_ids`, an anchor-to-base term the teacher's CoT never
            # conditions — so only buffer/answer positions carry loss. Same
            # boundary as the sft_mode answer-region weights, so SFT and OPD
            # supervise identical positions when K=0.
            prompt_end = min(len(target_tokens), max(0, int(prompt_token_lens[idx]) - 1))
            target_tokens = list(target_tokens)
            for j in range(prompt_end):
                target_tokens[j] = -100

        if correct_prefix_only and sample_ok_by_sample is not None:
            # Correct-prefix filtering (science runbook §7.1): on a floored task
            # (~5% correct), distilling the teacher's conditionals at the ~95%
            # WRONG on-policy prefixes erodes competence (the binding bootstrap
            # constraint). When enabled, supervise ONLY samples whose sampled
            # answer was correct — mask the WHOLE target for any sample that is
            # not verified-correct (sample_ok != 1, incl. -1=unknown) so the KL
            # lands only on the right manifold. Shrinks the effective batch (that
            # is the point); pair with a larger prompts/step if valid-token count
            # gets too small. Reuses the same -100 masking as mask_prompt_positions.
            if int(sample_ok_by_sample[idx]) != 1:
                target_tokens = [-100] * len(target_tokens)

        teacher_weights = (
            list(teacher_weights_by_sample[idx])
            if teacher_weights_by_sample is not None
            else [1.0] * len(target_tokens)
        )
        if len(teacher_weights) != len(target_tokens):
            raise RuntimeError(
                f"Sample {idx}: teacher_weights length {len(teacher_weights)} != target length {len(target_tokens)}"
            )
        hidden_match_weights: list[float] | None = None
        if hidden_match_weights_by_sample is not None:
            hidden_match_weights = list(hidden_match_weights_by_sample[idx])
            if len(hidden_match_weights) != len(target_tokens):
                raise RuntimeError(
                    f"Sample {idx}: hidden_match_weights length {len(hidden_match_weights)} "
                    f"!= target length {len(target_tokens)}"
                )
        if mask_zero_weight_positions:
            target_tokens = list(target_tokens)
            for pos, token in enumerate(target_tokens):
                if token == -100:
                    continue
                teacher_weight = float(teacher_weights[pos])
                hidden_weight = (
                    float(hidden_match_weights[pos])
                    if hidden_match_weights is not None
                    else teacher_weight
                )
                if teacher_weight == 0.0 and hidden_weight == 0.0:
                    target_tokens[pos] = -100
        loss_fn_inputs: dict[str, Any] = {
            "target_tokens": target_tokens,
            "teacher_ids": [0] * len(target_tokens),
            "teacher_weights": teacher_weights,
            "teacher_cache_indices": remapped_indices,
        }
        # Metrics-only KL-split diagnostics: per-position region ids (0=prompt,
        # 1=buffer, 2=answer; same boundaries as _position_weight_bucket_sums) and
        # the sample's broadcast answer-correctness flag (1/0, -1=unknown). The
        # trainer splits the per-token KL by these — they never affect the loss.
        # XORL_OPD_DIAG_DATUM_FIELDS=0 suppresses them for LEGACY trainers
        # (pre-2026-06-09 xorl lacks the _LOSS_EXCLUDE_KEYS entries, so unknown
        # micro-batch keys would reach the model forward as kwargs and crash) —
        # used by repo-state-rewind reproduction runs.
        if os.environ.get("XORL_OPD_DIAG_DATUM_FIELDS", "1").strip().lower() not in {"0", "false", "no", "off"}:
            if prompt_token_lens is not None:
                p_region = max(0, int(prompt_token_lens[idx]))
                prompt_end = min(len(target_tokens), max(0, p_region - 1))
                buffer_end = min(len(target_tokens), prompt_end + max(0, K))
                region_ids = (
                    [0] * prompt_end
                    + [1] * (buffer_end - prompt_end)
                    + [2] * (len(target_tokens) - buffer_end)
                )
            else:
                region_ids = [-1] * len(target_tokens)
            loss_fn_inputs["opd_region_ids"] = region_ids
            sample_ok = (
                int(sample_ok_by_sample[idx]) if sample_ok_by_sample is not None else -1
            )
            loss_fn_inputs["opd_sample_ok"] = [sample_ok] * len(target_tokens)
        if oprd_trainer_forward:
            # Trainer-side teacher forward inputs for this sample. The kept-position
            # count is the teacher's cache-row count; it must equal len(indices) (the
            # server-returned natural row range this sample's remap indexes into) so
            # the trainer's kept-row gather lines up 1:1 with teacher_cache_indices.
            teacher_kept = list(teacher_kept_indices_by_sample[idx])
            if len(teacher_kept) != len(indices):
                raise RuntimeError(
                    f"Sample {idx}: teacher kept positions {len(teacher_kept)} != teacher "
                    f"cache rows {len(indices)}; trainer-forward OPRD alignment broken"
                )
            loss_fn_inputs["teacher_input_ids"] = list(teacher_input_ids_by_sample[idx])
            loss_fn_inputs["teacher_kept_indices"] = teacher_kept
            # Explicit per-sample re-base anchor for the packer's OPRD local view
            # (= the sample's first GLOBAL cache row; indices is the server-returned
            # contiguous ascending range). Additive: old servers ignore it. Without
            # it the packer falls back to min()-inference, which is wrong for the
            # masked variants (their 0-filled positions make min()==0). Wrapped as a
            # 1-element list: the API schema's loss_fn_inputs InputType has no
            # scalar form (bare ints 422 at /forward_backward).
            loss_fn_inputs["teacher_cache_base"] = [int(indices[0]) if len(indices) else 0]
        if hidden_match_weights is not None:
            loss_fn_inputs["hidden_match_weights"] = hidden_match_weights
        if old_logprobs_by_sample is not None:
            old_logprobs = list(old_logprobs_by_sample[idx])
            if len(old_logprobs) != len(target_tokens):
                raise RuntimeError(
                    f"Sample {idx}: old_logprobs length {len(old_logprobs)} != target length {len(target_tokens)}"
                )
            loss_fn_inputs["old_logprobs"] = old_logprobs

        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": loss_fn_inputs,
            }
        )
    return data


def _buffer_position_weights(
    sequence: list[int],
    prompt_token_len: int,
    student_filler_count: int,
    value: float,
) -> list[float]:
    input_len = max(0, len(sequence) - 1)
    weights = [0.0] * input_len
    start = max(0, int(prompt_token_len) - 1)
    end = min(input_len, start + max(0, int(student_filler_count)))
    for idx in range(start, end):
        weights[idx] = float(value)
    return weights


def _answer_position_weights(
    sequence: list[int],
    prompt_token_len: int,
    student_filler_count: int,
    value: float,
) -> list[float]:
    input_len = max(0, len(sequence) - 1)
    weights = [0.0] * input_len
    start = max(0, int(prompt_token_len) - 1 + max(0, int(student_filler_count)))
    for idx in range(start, input_len):
        weights[idx] = float(value)
    return weights


def _position_weight_bucket_sums(
    weights: list[float],
    prompt_token_len: int,
    student_filler_count: int,
) -> dict[str, float]:
    input_len = len(weights)
    prompt_end = min(input_len, max(0, int(prompt_token_len) - 1))
    buffer_end = min(input_len, prompt_end + max(0, int(student_filler_count)))
    buckets = {
        "prompt": weights[:prompt_end],
        "buffer": weights[prompt_end:buffer_end],
        "answer": weights[buffer_end:],
    }
    out: dict[str, float] = {}
    for name, vals in buckets.items():
        out[f"{name}_weight_sum"] = float(sum(vals))
        out[f"{name}_weight_count"] = float(len(vals))
    return out


def _add_position_weights(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise RuntimeError(f"position-weight lengths differ: {len(left)} != {len(right)}")
    return [float(a) + float(b) for a, b in zip(left, right)]


def _with_corrupted_buffer(
    sequence: list[int],
    prompt_token_len: int,
    student_filler_count: int,
    *,
    mode: str = "reverse",
) -> list[int]:
    p = int(prompt_token_len)
    k = int(student_filler_count)
    if k <= 0:
        return list(sequence)
    return (
        list(sequence[:p])
        + _corrupt_token_ids(list(sequence[p : p + k]), mode=mode)
        + list(sequence[p + k :])
    )


def _different_prompt_donor_index(sample_idx: int, total_samples: int, group_size: int) -> int | None:
    if total_samples < 2:
        return None
    G = max(1, int(group_size))
    src_prompt = sample_idx // G
    for offset in range(1, total_samples):
        donor_idx = (sample_idx + offset) % total_samples
        if donor_idx // G != src_prompt:
            return donor_idx
    return None


def _with_mismatched_memory_cache_indices(
    cache_indices_by_sample: list[list[int]],
    prompt_token_lens: list[int],
    sample_idx: int,
    *,
    span: int,
    group_size: int = 1,
) -> tuple[list[int] | None, int | None, int]:
    """Return own cache indices with only the memory-span rows donated by another prompt."""
    total_samples = len(cache_indices_by_sample)
    donor_idx = _different_prompt_donor_index(sample_idx, total_samples, group_size)
    if donor_idx is None:
        return None, None, 0
    if sample_idx >= len(prompt_token_lens) or donor_idx >= len(prompt_token_lens):
        return None, donor_idx, 0
    own_indices = list(cache_indices_by_sample[sample_idx])
    donor_indices = list(cache_indices_by_sample[donor_idx])
    memory_span = max(0, int(span))
    start = max(0, int(prompt_token_lens[sample_idx]) - 1)
    donor_start = max(0, int(prompt_token_lens[donor_idx]) - 1)
    end = start + memory_span
    donor_end = donor_start + memory_span
    if memory_span <= 0 or end > len(own_indices) or donor_end > len(donor_indices):
        return None, donor_idx, 0

    mismatched = list(own_indices)
    mismatched[start:end] = donor_indices[donor_start:donor_end]
    changed_rows = sum(
        int(int(left) != int(right))
        for left, right in zip(own_indices[start:end], mismatched[start:end])
    )
    return mismatched, donor_idx, changed_rows


def _teacher_memory_pair_diagnostics(
    cache_path: Path,
    cache_indices_by_sample: list[list[int]],
    prompt_token_lens: list[int],
    *,
    corrupt_span: int,
    group_size: int = 1,
) -> dict[str, float]:
    """Compare each sample's teacher memory rows against another prompt's rows."""
    metrics: dict[str, float] = {
        "opd_teacher_memory_pair_diag_requested": 1.0,
        "opd_teacher_memory_pair_diag_active": 0.0,
        "opd_teacher_memory_pair_diag_failure": 0.0,
        "opd_teacher_memory_pair_span_tokens": float(max(0, int(corrupt_span))),
        "opd_teacher_memory_pair_sample_count": 0.0,
        "opd_teacher_memory_pair_skipped_samples": 0.0,
        "opd_teacher_memory_pair_cross_token_count": 0.0,
        "opd_teacher_memory_pair_cross_cosine_similarity_mean": 0.0,
        "opd_teacher_memory_pair_cross_cosine_distance_mean": 0.0,
        "opd_teacher_memory_pair_cross_cosine_distance_min": 0.0,
        "opd_teacher_memory_pair_cross_cosine_distance_max": 0.0,
        "opd_teacher_memory_pair_within_adjacent_token_count": 0.0,
        "opd_teacher_memory_pair_within_adjacent_distance_mean": 0.0,
        "opd_teacher_memory_pair_cross_minus_within_distance": 0.0,
    }
    span = max(0, int(corrupt_span))
    total_samples = len(cache_indices_by_sample)
    if span <= 0 or total_samples < 2:
        metrics["opd_teacher_memory_pair_skipped_samples"] = float(total_samples)
        return metrics
    if len(prompt_token_lens) != total_samples:
        raise ValueError(
            f"prompt_token_lens has {len(prompt_token_lens)} entries for "
            f"{total_samples} cache-index samples"
        )

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    tensors = load_file(str(cache_path))
    cache_key = "hidden_states" if "hidden_states" in tensors else next(iter(tensors))
    cache = tensors[cache_key].float()
    cache_rows = int(cache.shape[0])

    cross_similarities: list["torch.Tensor"] = []
    cross_distances: list["torch.Tensor"] = []
    within_distances: list["torch.Tensor"] = []
    sample_pairs = 0
    skipped = 0

    for sample_idx, indices in enumerate(cache_indices_by_sample):
        donor_idx = _different_prompt_donor_index(sample_idx, total_samples, group_size)
        if donor_idx is None:
            skipped += 1
            continue
        start = max(0, int(prompt_token_lens[sample_idx]) - 1)
        end = start + span
        donor_start = max(0, int(prompt_token_lens[donor_idx]) - 1)
        donor_end = donor_start + span
        donor_indices = cache_indices_by_sample[donor_idx]
        if end > len(indices) or donor_end > len(donor_indices):
            skipped += 1
            continue
        own_rows_idx = [int(row) for row in indices[start:end]]
        donor_rows_idx = [int(row) for row in donor_indices[donor_start:donor_end]]
        if not own_rows_idx or not donor_rows_idx:
            skipped += 1
            continue
        if min(own_rows_idx + donor_rows_idx) < 0 or max(own_rows_idx + donor_rows_idx) >= cache_rows:
            skipped += 1
            continue
        own_rows = cache[torch.tensor(own_rows_idx, dtype=torch.long)]
        donor_rows = cache[torch.tensor(donor_rows_idx, dtype=torch.long)]
        if own_rows.shape != donor_rows.shape:
            skipped += 1
            continue
        sims = F.cosine_similarity(own_rows, donor_rows, dim=-1, eps=1e-6)
        distances = 1.0 - sims
        cross_similarities.append(sims.detach())
        cross_distances.append(distances.detach())
        if own_rows.shape[0] > 1:
            within_sims = F.cosine_similarity(own_rows[:-1], own_rows[1:], dim=-1, eps=1e-6)
            within_distances.append((1.0 - within_sims).detach())
        sample_pairs += 1

    metrics["opd_teacher_memory_pair_sample_count"] = float(sample_pairs)
    metrics["opd_teacher_memory_pair_skipped_samples"] = float(skipped)
    if not cross_distances:
        return metrics

    cross_sims = torch.cat(cross_similarities)
    cross_dists = torch.cat(cross_distances)
    metrics["opd_teacher_memory_pair_diag_active"] = 1.0
    metrics["opd_teacher_memory_pair_cross_token_count"] = float(cross_dists.numel())
    metrics["opd_teacher_memory_pair_cross_cosine_similarity_mean"] = float(cross_sims.mean().item())
    metrics["opd_teacher_memory_pair_cross_cosine_distance_mean"] = float(cross_dists.mean().item())
    metrics["opd_teacher_memory_pair_cross_cosine_distance_min"] = float(cross_dists.min().item())
    metrics["opd_teacher_memory_pair_cross_cosine_distance_max"] = float(cross_dists.max().item())

    if within_distances:
        within_dists = torch.cat(within_distances)
        within_mean = float(within_dists.mean().item())
        metrics["opd_teacher_memory_pair_within_adjacent_token_count"] = float(within_dists.numel())
        metrics["opd_teacher_memory_pair_within_adjacent_distance_mean"] = within_mean
        metrics["opd_teacher_memory_pair_cross_minus_within_distance"] = (
            metrics["opd_teacher_memory_pair_cross_cosine_distance_mean"] - within_mean
        )
    return metrics


def _teacher_cache_from_xorl(
    teacher_url: str,
    sequences: list[list[int]],
    cache_path: Path,
    model_id: str,
    timeout: float,
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | list[list[int]] | None = None,
    prompt_token_lens: list[int] | None = None,
    student_filler_count: int = 0,
    teacher_cot_mode: str = "replace",
    supervise_student_cot: bool = False,
    per_sample_filler_count: list[int] | None = None,
) -> dict[str, Any]:
    """Ask the XORL teacher server to write a hidden-state cache on shared storage."""
    response = requests.post(
        f"{teacher_url}/api/v1/forward",
        json={
            "model_id": model_id,
            "forward_input": {
                "data": _teacher_hidden_cache_data(
                    sequences,
                    teacher_prefix_tokens,
                    teacher_filler_tokens,
                    prompt_token_lens,
                    student_filler_count=student_filler_count,
                    teacher_cot_mode=teacher_cot_mode,
                    supervise_student_cot=supervise_student_cot,
                    per_sample_filler_count=per_sample_filler_count,
                ),
                "loss_fn": "teacher_hidden_cache",
                "loss_fn_params": {
                    "teacher_hidden_cache_path": str(cache_path),
                    "teacher_hidden_cache_dtype": "bfloat16",
                },
            },
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Teacher cache request failed: {response.status_code} {response.text}"
        ) from exc
    future = _wait_for_future(
        teacher_url, response.json()["request_id"], timeout=timeout
    )
    cache = future.get("info", {}).get("teacher_hidden_cache")
    if not cache:
        raise RuntimeError(
            f"XORL teacher did not return teacher_hidden_cache metadata: {future}"
        )

    cache_indices = cache.get("cache_indices_by_sample") or []
    if len(cache_indices) != len(sequences):
        raise RuntimeError(
            f"XORL teacher returned {len(cache_indices)} cache-index lists for {len(sequences)}"
        )
    # When the student has its own filler (Run B), the teacher cache rows
    # count is shorter than student input_ids by exactly student_filler_count,
    # because we filtered out the teacher's filler-predicting positions and
    # the student's filler positions get masked client-side. So skip the
    # 1-to-1 length check in that branch.
    if student_filler_count == 0:
        for sequence, indices in zip(sequences, cache_indices):
            input_ids, _ = _opd_causal_pair(sequence)
            if len(indices) != len(input_ids):
                raise RuntimeError(
                    f"XORL teacher returned {len(indices)} indices for input length {len(input_ids)}"
                )
    if cache.get("path") != str(cache_path):
        raise RuntimeError(
            f"XORL teacher wrote unexpected cache path: {cache.get('path')} != {cache_path}"
        )
    return {
        "cache_indices_by_sample": cache_indices,
        "metrics": future.get("metrics", {}),
        "info": future.get("info", {}),
    }


def _post_teacher_cache_with_retry(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    max_attempts: int = 6,
) -> "requests.Response":
    """POST to a teacher ``/teacher_hidden_cache`` endpoint, retrying transient
    failures. The cache write is idempotent (same input_ids -> same hiddens ->
    same file overwritten), so re-POSTing a request that hit a transient SMG/worker
    blip (502/503/504, connection reset, read timeout) is safe — and is the only
    thing between a momentary hiccup in a large prefill burst (1k+ requests/step)
    and a crashed run. Non-transient failures (e.g. 400 cached_kept) are returned
    to the caller's ``raise_for_status`` immediately, not retried.
    """
    import time as _time  # noqa: PLC0415

    # 500 included: a teacher replica's scheduler can momentarily 500 under load
    # (e.g. a kernel recompile); retrying lets it (or, with a non-affinity router,
    # another replica) serve the idempotent request instead of crashing the run.
    transient_status = {500, 502, 503, 504}
    last: str = "unknown"
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt == max_attempts - 1:
                raise
            _time.sleep(min(2.0 * (attempt + 1), 15.0))
            continue
        if response.status_code in transient_status and attempt < max_attempts - 1:
            last = f"{response.status_code} {response.text[:200]}"
            _time.sleep(min(2.0 * (attempt + 1), 15.0))
            continue
        return response
    raise RuntimeError(f"teacher cache POST exhausted {max_attempts} retries: {last}")


def _resolve_oprd_layer_indices(
    spec: str, num_layers: int = 0
) -> list[int] | None:
    """Expand the ``opd_oprd_layers`` config to an explicit, sorted layer-id list.

    Returns None when OPRD is off (empty spec). The SAME explicit list is sent to
    the teacher (capture_layer_indices) and the trainer (opd_oprd_layer_indices) so
    both pick identical decoder layers. ``everyN`` needs ``num_layers`` to expand;
    a comma list (e.g. "0,8,16,24") does not.
    """
    spec = (spec or "").strip().lower()
    if not spec:
        return None
    indices: list[int]
    if spec.startswith("every"):
        try:
            stride = int(spec[len("every") :])
        except ValueError as exc:
            raise ValueError(f"invalid opd_oprd_layers stride spec: {spec!r}") from exc
        if stride <= 0:
            raise ValueError(f"opd_oprd_layers stride must be > 0, got {stride}")
        if num_layers <= 0:
            raise ValueError(
                "opd_oprd_layers='everyN' requires opd_oprd_num_layers > 0 to expand "
                "into explicit layer indices"
            )
        indices = list(range(0, num_layers, stride))
    else:
        indices = [int(tok) for tok in spec.replace(" ", "").split(",") if tok != ""]
    seen: set[int] = set()
    resolved: list[int] = []
    for i in indices:
        if i >= 0 and i not in seen:
            seen.add(i)
            resolved.append(i)
    return sorted(resolved) or None


def _teacher_cache_from_sglang(
    teacher_url: str,
    sequences: list[list[int]],
    cache_path: Path,
    timeout: float,
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | list[list[int]] | None = None,
    prompt_token_lens: list[int] | None = None,
    student_filler_count: int = 0,
    teacher_cot_mode: str = "replace",
    supervise_student_cot: bool = False,
    mask_answer: bool = False,
    per_sample_filler_count: list[int] | None = None,
    capture_layer_indices: list[int] | None = None,
    layers_cache_path: str | None = None,
    use_sglang_layer_cache: bool = False,
) -> dict[str, Any]:
    """Drop-in for ``_teacher_cache_from_xorl`` against the xorl-sglang-internal
    ``/teacher_hidden_cache`` endpoint.

    Builds the IDENTICAL teacher payload via ``_teacher_hidden_cache_data`` (same
    CoT-insertion / pause / supervise masking), then POSTs the per-sample
    ``input_ids`` + ``target_tokens``. The sglang teacher prefills with full
    hidden capture, keeps exactly the ``target != -100`` (CoT/pause/answer)
    positions, and writes the rank-2 KL safetensors cache itself; only metadata
    returns over HTTP (hidden tensors never leave the sglang process). Launch the
    teacher with ``--enable-return-hidden-states --disable-radix-cache
    --chunked-prefill-size >= max_seq_len`` and a shared cache filesystem.

    Multi-layer OPRD (``capture_layer_indices`` set) defaults to the current
    trainer-side no-grad teacher forward: SGLang writes only the rank-2 KL cache
    and returns teacher ``input_ids`` + kept-position indices so the trainer can
    recompute per-layer hiddens. When ``use_sglang_layer_cache`` is true, SGLang
    additionally writes its rank-3 per-layer cache and the trainer reads it
    directly; this is an explicit throughput A/B because it moves work from trainer
    f/b to teacher prefill/cache write.
    """
    oprd_active = bool(capture_layer_indices)
    oprd_trainer_forward = oprd_active and not use_sglang_layer_cache
    data = _teacher_hidden_cache_data(
        sequences,
        teacher_prefix_tokens,
        teacher_filler_tokens,
        prompt_token_lens,
        student_filler_count=student_filler_count,
        teacher_cot_mode=teacher_cot_mode,
        supervise_student_cot=supervise_student_cot,
        mask_answer=mask_answer,
        per_sample_filler_count=per_sample_filler_count,
        return_teacher_seqs=oprd_trainer_forward,
    )
    input_ids = [row["model_input"]["input_ids"] for row in data]
    target_tokens = [row["loss_fn_inputs"]["target_tokens"] for row in data]
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "target_tokens": target_tokens,
        "cache_path": str(cache_path),
        "cache_key": "hidden_states",
        "dtype": "bfloat16",
    }
    if oprd_active and use_sglang_layer_cache:
        if not layers_cache_path:
            raise ValueError("use_sglang_layer_cache requires layers_cache_path")
        payload["capture_layer_indices"] = list(capture_layer_indices or [])
        payload["layers_cache_path"] = str(layers_cache_path)
    response = _post_teacher_cache_with_retry(
        f"{teacher_url}/teacher_hidden_cache",
        payload,
        timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"SGLang teacher cache request failed: {response.status_code} {response.text}"
        ) from exc
    cache = response.json()
    if not isinstance(cache, dict):
        raise RuntimeError(
            f"SGLang teacher returned unexpected payload type: {type(cache).__name__}"
        )
    if cache.get("error"):
        raise RuntimeError(f"SGLang teacher_hidden_cache error: {cache['error']}")

    cache_indices = cache.get("cache_indices_by_sample") or []
    if len(cache_indices) != len(sequences):
        raise RuntimeError(
            f"SGLang teacher returned {len(cache_indices)} cache-index lists for {len(sequences)}"
        )
    # The kept-row count must equal the supervised positions (target != -100) in
    # the rows we sent — this naturally covers supervise_student_cot mode (where
    # the pause block is kept) without a separate expected-count helper.
    for tgt, indices in zip(target_tokens, cache_indices):
        expected = sum(1 for t in tgt if t != -100)
        if len(indices) != expected:
            raise RuntimeError(
                f"SGLang teacher returned {len(indices)} indices; expected {expected} "
                f"supervised positions (target != -100)"
            )
    if cache.get("path") != str(cache_path):
        raise RuntimeError(
            f"SGLang teacher wrote unexpected cache path: {cache.get('path')} != {cache_path}"
        )
    result: dict[str, Any] = {
        "cache_indices_by_sample": cache_indices,
        "layers_cache_path": cache.get("layers_path") if use_sglang_layer_cache else None,
        "metrics": {},
        "info": cache,
    }
    if use_sglang_layer_cache:
        if result["layers_cache_path"] != str(layers_cache_path):
            raise RuntimeError(
                f"SGLang teacher wrote unexpected OPRD layer cache path: "
                f"{result['layers_cache_path']} != {layers_cache_path}"
            )
        expected_layers = len(capture_layer_indices or [])
        actual_layers = int(cache.get("num_capture_layers") or 0)
        if actual_layers != expected_layers:
            raise RuntimeError(
                f"SGLang teacher wrote {actual_layers} OPRD layers; expected {expected_layers}"
            )
    if oprd_trainer_forward:
        # Trainer-side teacher forward inputs: per-sample teacher input_ids + the kept
        # positions (target != -100). The kept count must match the rank-2 cache rows
        # (same target != -100 invariant), guaranteeing 1:1 student↔teacher alignment.
        result["teacher_input_ids_by_sample"] = [row["teacher_input_ids"] for row in data]
        result["teacher_kept_indices_by_sample"] = [row["teacher_kept_indices"] for row in data]
        for kept, indices in zip(result["teacher_kept_indices_by_sample"], cache_indices):
            if len(kept) != len(indices):
                raise RuntimeError(
                    f"OPRD teacher kept-position count {len(kept)} != rank-2 cache rows "
                    f"{len(indices)}; student↔teacher alignment would be broken"
                )
    return result


# ---------------------------------------------------------------------------
# Pipelined two-phase teacher prefill (runbooks/pipelined_teacher_prefill.md).
#
# The single-phase supervise path keeps, per sample, the contiguous rows
#     prompt[0:p-1]  pause[0:K]  answer[0:ans]
# from teacher_seq = prompt + CoT + pause + answer (CoT-predicting positions
# masked). We reproduce that EXACT row set with two radix-served prefills:
#
#   Phase A: teacher_seq_A = prompt + CoT + pause          (no answer)
#            keep = prompt[0:p-1] + pause[0:K]   (mask the C CoT-predicting rows)
#            -> the (p-1)+K "prefix" rows. FIXED per prompt; prefetchable.
#   Phase B: teacher_seq_B = prompt + CoT + pause + answer (the full seq)
#            keep = answer[0:ans] only           (mask prompt+CoT+pause)
#            -> the ans answer rows. radix serves prompt+CoT+pause from A.
#
# A pause position's hidden state attends only to prompt+CoT+pause to its left
# (causal LM), so it is bit-identical between Phase A and the full sequence —
# the answer never influences it. Merging A's prefix rows with B's answer rows
# yields the single-phase cache exactly.
# ---------------------------------------------------------------------------


def _teacher_phase_data(
    sequences: list[list[int]],
    phase: str,
    teacher_prefix_tokens: list[int] | None,
    teacher_filler_tokens: list[int] | list[list[int]] | None,
    prompt_token_lens: list[int] | None,
    student_filler_count: int,
) -> list[dict[str, Any]]:
    """Build per-sample {input_ids, target_tokens} for one pipeline phase.

    Requires the production supervise recipe: per-sample CoT filler
    (``teacher_filler_tokens`` as ``list[list[int]]``), a student pause region of
    ``student_filler_count`` tokens at ``prompt_token_lens[idx]``, and
    ``teacher_cot_mode='insert'`` semantics. ``phase`` is ``"a"`` (prefix:
    prompt+pause kept, answer absent) or ``"b"`` (answer kept, full seq).
    """
    if phase not in ("a", "b"):
        raise ValueError(f"phase must be 'a' or 'b', got {phase!r}")
    prefix = list(teacher_prefix_tokens) if teacher_prefix_tokens else []

    first = teacher_filler_tokens[0] if teacher_filler_tokens else None
    if not (isinstance(first, list)):
        raise ValueError(
            "teacher_pipeline_phase requires per-sample CoT (teacher_filler_tokens as "
            "list[list[int]] — set teacher_cot_json_path)"
        )
    if len(teacher_filler_tokens) != len(sequences):
        raise ValueError(
            f"teacher_filler_tokens has {len(teacher_filler_tokens)} entries for "
            f"{len(sequences)} sequences"
        )
    if not prompt_token_lens or len(prompt_token_lens) != len(sequences):
        raise ValueError(
            "teacher_pipeline_phase requires prompt_token_lens aligned to sequences"
        )

    K = int(student_filler_count)
    data: list[dict[str, Any]] = []
    for idx, sequence in enumerate(sequences):
        p = int(prompt_token_lens[idx])
        cot = list(teacher_filler_tokens[idx])
        C = len(cot)
        prompt = list(sequence[:p])
        pause = list(sequence[p : p + K])
        answer = list(sequence[p + K :])

        if phase == "a":
            # prompt + CoT + pause  (answer omitted)
            full = prompt + cot + pause
            input_ids = full[:-1]
            target_tokens = list(full[1:])
            # Mask the C CoT-predicting positions [p-1, p-1+C); keep prompt + pause.
            # (The final pause position predicts nothing meaningful here, but its
            # hidden state — what we cache — is the post-CoT pause state, identical
            # to the full-sequence pause hidden.)
            for j in range(max(0, p - 1), max(0, p - 1) + C):
                if 0 <= j < len(target_tokens):
                    target_tokens[j] = -100
        else:  # phase == "b": prompt + CoT + pause + answer, keep ONLY answer
            full = prompt + cot + pause + answer
            input_ids = full[:-1]
            target_tokens = list(full[1:])
            # Keep ONLY the answer-predicting positions. The answer block starts at
            # full index (p + C + K); its predictions live at target positions
            # [p + C + K - 1, len). Mask everything before that (prompt+CoT+pause).
            ans_pred_start = p + C + K - 1
            for j in range(0, min(ans_pred_start, len(target_tokens))):
                target_tokens[j] = -100

        if prefix:
            input_ids = prefix + input_ids
            target_tokens = [-100] * len(prefix) + target_tokens

        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {"target_tokens": target_tokens},
            }
        )
    return data


def _post_teacher_phase_cache(
    teacher_url: str,
    data: list[dict[str, Any]],
    cache_path: Path,
    timeout: float,
) -> dict[str, Any]:
    """POST one phase's payload to the sglang ``/teacher_hidden_cache`` endpoint.

    Returns the endpoint metadata (incl. ``cache_indices_by_sample``). Validates
    that the kept-row count per sample equals the supervised positions in the
    payload (target != -100) — same invariant as ``_teacher_cache_from_sglang``.
    """
    input_ids = [row["model_input"]["input_ids"] for row in data]
    target_tokens = [row["loss_fn_inputs"]["target_tokens"] for row in data]
    response = _post_teacher_cache_with_retry(
        f"{teacher_url}/teacher_hidden_cache",
        {
            "input_ids": input_ids,
            "target_tokens": target_tokens,
            "cache_path": str(cache_path),
            "cache_key": "hidden_states",
            "dtype": "bfloat16",
        },
        timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"SGLang teacher phase cache request failed: {response.status_code} {response.text}"
        ) from exc
    cache = response.json()
    if not isinstance(cache, dict) or cache.get("error"):
        raise RuntimeError(f"SGLang teacher_hidden_cache error: {cache}")
    cache_indices = cache.get("cache_indices_by_sample") or []
    if len(cache_indices) != len(data):
        raise RuntimeError(
            f"SGLang teacher returned {len(cache_indices)} cache-index lists for {len(data)}"
        )
    for tgt, indices in zip(target_tokens, cache_indices):
        expected = sum(1 for t in tgt if t != -100)
        if len(indices) != expected:
            raise RuntimeError(
                f"SGLang teacher returned {len(indices)} indices; expected {expected} "
                f"supervised positions (target != -100)"
            )
    if cache.get("path") != str(cache_path):
        raise RuntimeError(
            f"SGLang teacher wrote unexpected cache path: {cache.get('path')} != {cache_path}"
        )
    return cache


def _merge_phase_caches(
    a_path: Path,
    b_path: Path,
    a_indices: list[list[int]],
    b_indices: list[list[int]],
    merged_path: Path,
    cache_key: str = "hidden_states",
) -> list[list[int]]:
    """Concatenate Phase A (prefix: prompt+pause) and Phase B (answer) rows.

    Per sample, the merged cache holds A's rows followed by B's rows, matching the
    single-phase row order ``prompt[0:p-1] pause[0:K] answer[0:ans]``. Returns the
    merged ``cache_indices_by_sample`` (contiguous range per sample), and writes a
    ``[N_total, hidden]`` safetensors at ``merged_path`` under ``cache_key`` so
    forward_backward reads it unchanged.
    """
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    if len(a_indices) != len(b_indices):
        raise RuntimeError(
            f"phase-A/B sample count mismatch: {len(a_indices)} vs {len(b_indices)}"
        )
    a_t = load_file(str(a_path))[cache_key]
    b_t = load_file(str(b_path))[cache_key]
    if a_t.shape[1] != b_t.shape[1]:
        raise RuntimeError(
            f"phase-A/B hidden_size mismatch: {a_t.shape[1]} != {b_t.shape[1]}"
        )

    chunks: list["torch.Tensor"] = []
    merged_indices: list[list[int]] = []
    offset = 0
    for ai, bi in zip(a_indices, b_indices):
        a_rows = a_t[torch.tensor(ai, dtype=torch.long)] if ai else a_t[0:0]
        b_rows = b_t[torch.tensor(bi, dtype=torch.long)] if bi else b_t[0:0]
        sample = torch.cat([a_rows, b_rows], dim=0)
        rows = int(sample.shape[0])
        merged_indices.append(list(range(offset, offset + rows)))
        offset += rows
        chunks.append(sample)

    merged = torch.cat(chunks, dim=0).contiguous()
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{merged_path}.tmp-pid{os.getpid()}"
    save_file({cache_key: merged}, tmp_path)
    os.replace(tmp_path, str(merged_path))
    return merged_indices


# ---------------------------------------------------------------------------
# GROUP TEACHER — shared-prefix two-phase prefill for group sampling (G>1).
#
# The G samples of prompt n share the prompt+CoT+pause prefix and differ only in
# the answer tail. So we prefill that prefix ONCE per prompt (Phase A → the
# prompt+pause rows, which include the supervised-pause hiddens) and extend with
# each of the G answers (Phase B → each answer's rows; the shared prefix is served
# from the teacher's radix cache, so only the answer tokens are freshly computed).
# We then merge, per (prompt n, sample g): that prompt's ONE Phase-A prefix block
# followed by sample g's Phase-B answer block — the exact single-sample
# {prompt, pause, answer} cache for (prompt n, answer_g), in the student's
# kept-position order. This is the SAME Phase-A/Phase-B mechanism as
# `_teacher_phase_data` / `_merge_phase_caches`, with one Phase A shared across G
# Phase Bs (the prior two-phase code merged 1 A with 1 B per sample).
# ---------------------------------------------------------------------------


def _merge_group_phase_caches(
    a_path: Path,
    b_path: Path,
    a_indices: list[list[int]],
    b_indices: list[list[int]],
    group_size: int,
    merged_path: Path,
    cache_key: str = "hidden_states",
) -> list[list[int]]:
    """Merge a per-PROMPT Phase A (length N) with a per-SAMPLE Phase B (length N*G).

    ``a_indices`` has one prefix-row list per prompt (N entries); ``b_indices`` has
    one answer-row list per sample (N*G entries, prompt-major: sample i belongs to
    prompt ``i // G``). The merged cache, per sample i, is prompt ``i//G``'s Phase-A
    prefix rows followed by sample i's Phase-B answer rows — i.e. for sample i the
    rows are ``promptA[i//G]  ⧺  answerB[i]`` in kept-position order. Returns the
    contiguous merged ``cache_indices_by_sample`` (N*G entries) and writes the
    ``[N_total, hidden]`` safetensors so forward_backward reads it unchanged.
    """
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    G = max(1, int(group_size))
    if len(b_indices) != len(a_indices) * G:
        raise RuntimeError(
            f"group phase merge: {len(b_indices)} Phase-B samples != {len(a_indices)} "
            f"prompts × G={G}"
        )
    a_t = load_file(str(a_path))[cache_key]
    b_t = load_file(str(b_path))[cache_key]
    if a_t.shape[1] != b_t.shape[1]:
        raise RuntimeError(
            f"group phase-A/B hidden_size mismatch: {a_t.shape[1]} != {b_t.shape[1]}"
        )

    chunks: list["torch.Tensor"] = []
    merged_indices: list[list[int]] = []
    offset = 0
    for sample_idx, bi in enumerate(b_indices):
        prompt_idx = sample_idx // G
        ai = a_indices[prompt_idx]
        a_rows = a_t[torch.tensor(ai, dtype=torch.long)] if ai else a_t[0:0]
        b_rows = b_t[torch.tensor(bi, dtype=torch.long)] if bi else b_t[0:0]
        sample = torch.cat([a_rows, b_rows], dim=0)
        rows = int(sample.shape[0])
        merged_indices.append(list(range(offset, offset + rows)))
        offset += rows
        chunks.append(sample)

    merged = torch.cat(chunks, dim=0).contiguous()
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{merged_path}.tmp-pid{os.getpid()}"
    save_file({cache_key: merged}, tmp_path)
    os.replace(tmp_path, str(merged_path))
    return merged_indices


def _offset_teacher_cache_refs(data: list[dict[str, Any]], offset: int) -> None:
    """Shift teacher-cache row references after concatenating chunk caches."""
    if offset == 0:
        return
    for row in data:
        loss_inputs = row.get("loss_fn_inputs") or {}
        indices = loss_inputs.get("teacher_cache_indices")
        if indices is not None:
            loss_inputs["teacher_cache_indices"] = [int(idx) + offset for idx in indices]
        base = loss_inputs.get("teacher_cache_base")
        if base is not None:
            loss_inputs["teacher_cache_base"] = [int(base[0]) + offset] if base else []


def _merge_prepared_chunk_batches(
    chunks: list[PreparedOpdBatch],
    merged_path: Path,
    *,
    cache_key: str = "hidden_states",
) -> PreparedOpdBatch:
    """Merge ordered strict-prepare chunks into one trainer-facing prepare batch.

    Each chunk was prepared with identical semantics over a disjoint prompt slice.
    We concatenate the hidden-cache tensors in chunk order, offset every
    teacher-cache row reference by that chunk's starting row, and return a single
    PreparedOpdBatch so the trainer still receives one fb call for the original
    prompt window.
    """
    if not chunks:
        raise ValueError("cannot merge zero prepared chunks")

    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    tensors = [load_file(str(chunk.cache_path))[cache_key] for chunk in chunks]
    hidden_sizes = {int(t.shape[1]) for t in tensors if t.ndim == 2}
    if any(t.ndim != 2 for t in tensors) or len(hidden_sizes) != 1:
        raise RuntimeError("prepared chunk cache tensors must all be rank-2 with identical hidden size")

    merged_t = torch.cat(tensors, dim=0).contiguous()
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{merged_path}.tmp-pid{os.getpid()}"
    save_file({cache_key: merged_t}, tmp_path)
    os.replace(tmp_path, str(merged_path))

    sequences: list[list[int]] = []
    data: list[dict[str, Any]] = []
    completions: list[list[int]] = []
    prompt_texts: list[str] = []
    offset = 0
    for chunk, tensor in zip(chunks, tensors):
        chunk_data = copy.deepcopy(chunk.data)
        _offset_teacher_cache_refs(chunk_data, offset)
        sequences.extend(chunk.sequences)
        data.extend(chunk_data)
        completions.extend(chunk.completions)
        prompt_texts.extend(chunk.prompt_texts)
        offset += int(tensor.shape[0])

    first_layers = chunks[0].oprd_layer_indices
    if any(chunk.layers_cache_path is not None for chunk in chunks):
        raise RuntimeError("strict chunked prepare does not support rank-3 layer caches")
    if any(chunk.oprd_layer_indices != first_layers for chunk in chunks):
        raise RuntimeError("prepared chunk OPRD layer subsets do not match")

    chunk_metrics = _aggregate_prepared_metrics(chunks)
    sample_wall_s = _interval_union_s(
        [
            (float(chunk.metrics["student_sampling_t0"]), float(chunk.metrics["student_sampling_t1"]))
            for chunk in chunks
            if "student_sampling_t0" in chunk.metrics
        ]
    )
    teacher_wall_s = _interval_union_s(
        [
            (float(chunk.metrics["teacher_prefill_t0"]), float(chunk.metrics["teacher_prefill_t1"]))
            for chunk in chunks
            if "teacher_prefill_t0" in chunk.metrics
        ]
    )
    metrics = dict(chunk_metrics)
    metrics.update(
        {
            "student_sampling_s": sample_wall_s or float(chunk_metrics.get("student_sampling_s", 0.0)),
            "student_sampling_output_tok_per_s": (
                float(chunk_metrics.get("student_sampling_output_tokens", 0.0)) / sample_wall_s
                if sample_wall_s > 0.0
                else 0.0
            ),
            "teacher_prefill_s": teacher_wall_s or float(chunk_metrics.get("teacher_prefill_s", 0.0)),
            "teacher_prefill_tok_per_s": (
                float(chunk_metrics.get("teacher_prefill_tokens", 0.0)) / teacher_wall_s
                if teacher_wall_s > 0.0
                else 0.0
            ),
            "strict_prepare_overlap_active": 1.0,
            "strict_prepare_overlap_chunks": float(len(chunks)),
            "strict_prepare_overlap_student_sampling_sum_s": float(chunk_metrics.get("student_sampling_s", 0.0)),
            "strict_prepare_overlap_teacher_prefill_sum_s": float(chunk_metrics.get("teacher_prefill_s", 0.0)),
        }
    )
    for chunk in chunks:
        try:
            Path(chunk.cache_path).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chunk cache cleanup skipped for %s: %s", chunk.cache_path, exc)
    return PreparedOpdBatch(
        sequences=sequences,
        data=data,
        cache_path=merged_path,
        metrics=metrics,
        completions=completions,
        prompt_texts=prompt_texts,
        layers_cache_path=None,
        oprd_layer_indices=first_layers,
    )


@dataclass
class PreparedOpdBatch:
    sequences: list[list[int]]
    data: list[dict[str, Any]]
    cache_path: Path
    metrics: dict[str, Any]
    # Raw student-generated tokens per sample (post-prefix), for in-loop eval.
    completions: list[list[int]] = field(default_factory=list)
    # Decoded user-prompt text per sample (for answer scoring + sample logging).
    prompt_texts: list[str] = field(default_factory=list)
    # Multi-layer OPRD: path to the rank-3 [layers, tokens, d] teacher cache (None
    # when OPRD is off) and the explicit decoder-layer subset that was captured.
    layers_cache_path: Path | None = None
    oprd_layer_indices: list[int] | None = None


# ---------------------------------------------------------------------------
# In-loop eval helpers (added 2026-05-28). Loss convergence is not a success
# signal for OPD; these surface what the student actually generates.
# ---------------------------------------------------------------------------

_MULT_RE = re.compile(r"(\d[\d,]*)\s*[*x×]\s*(\d[\d,]*)")
_ARITH_MARKER = "Evaluate this Python expression."
# Arithmetic answers can be negative — keep the leading -? in every parse.
_ARITH_ANS_RE = re.compile(r"-?\d[\d,]*")


def _safe_eval_arith(expr: str) -> int:
    """Evaluate an integer arithmetic expression (+ - * // % and parens) via an
    AST whitelist — never bare eval on prompt-derived text."""
    node = ast.parse(expr, mode="eval").body

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            value = ev(n.operand)
            return -value if isinstance(n.op, ast.USub) else value
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            return left % right
        raise ValueError(f"disallowed node: {ast.dump(n)}")

    return ev(node)


def _user_prompt_text(prompt: Any) -> str:
    """Best-effort extraction of the user-turn text from a chat prompt."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(
                msg.get("content"), str
            ):
                return msg["content"]
        return " ".join(
            m.get("content", "")
            for m in prompt
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
    if isinstance(prompt, str):
        return prompt
    return ""


_GSM8K_GOLD: dict[str, str] | None = None


def _gsm8k_gold_lookup(prompt_text: str) -> str | None:
    """Gold final-answer for a GSM8K prompt, loaded once from ``$GSM8K_GOLD_JSON``
    (a {prompt_content: answer} map). GSM8K gold is NOT computable from the
    question (unlike multiplication), so it is supplied out-of-band and keyed by
    the exact user-content string the prompt was built with."""
    global _GSM8K_GOLD
    if _GSM8K_GOLD is None:
        path = os.environ.get("GSM8K_GOLD_JSON", "")
        try:
            _GSM8K_GOLD = json.load(open(path)) if path else {}
        except Exception:  # noqa: BLE001 - missing/bad gold map => unscorable
            _GSM8K_GOLD = {}
        logger.info("gsm8k gold map: %d entries from %r", len(_GSM8K_GOLD), path)
    g = _GSM8K_GOLD.get(prompt_text)
    return str(g) if g is not None else None


def _score_answer(prompt_text: str, completion_text: str, task: str) -> bool | None:
    """Return True/False if the completion contains the correct answer, or None
    if the task scorer can't parse the problem (so accuracy skips it)."""
    target = _target_answer_text(prompt_text, task)
    if target is None:
        return None
    if task == "multiplication":
        digits = re.sub(r"[,\s]", "", completion_text)
        return target in digits
    if task == "arithmetic":
        # Substring match is unsafe with signed answers ("13" in "-13"/"113");
        # parse the first signed integer and compare exactly.
        m = _ARITH_ANS_RE.search(completion_text or "")
        if not m:
            return False
        try:
            return int(m.group(0).replace(",", "")) == int(target)
        except ValueError:
            return False
    if task == "gsm8k":
        # GSM8K answers are integers; the post-"Answer:" region is short. Lenient
        # integer-membership match (strip $/commas), like multiplication but
        # integer-boundary-safe.
        nums = re.findall(r"-?\d+", (completion_text or "").replace(",", "").replace("$", ""))
        return target in nums
    return None


def _target_answer_text(prompt_text: str, task: str) -> str | None:
    if task == "none" or not prompt_text:
        return None
    if task == "multiplication":
        m = _MULT_RE.search(prompt_text)
        if not m:
            return None
        a = int(m.group(1).replace(",", ""))
        b = int(m.group(2).replace(",", ""))
        return str(a * b)
    if task == "arithmetic":
        marker_pos = prompt_text.find(_ARITH_MARKER)
        if marker_pos < 0:
            return None
        expr = prompt_text[marker_pos + len(_ARITH_MARKER):].strip().splitlines()[0].strip()
        try:
            return str(_safe_eval_arith(expr))
        except (ValueError, SyntaxError, ZeroDivisionError):
            return None
    if task == "gsm8k":
        return _gsm8k_gold_lookup(prompt_text)
    return None


def _replace_sampled_answers_with_gold(
    sequences: list[list[int]],
    completions: list[list[int]],
    prompts: list[Any],
    prompt_token_lens: list[int],
    filler_token_count: int,
    group_size: int,
    chat_tokenizer: Any,
    task: str,
) -> tuple[list[list[int]], list[list[int]], int, int]:
    """Replace the post-prefill answer tail with deterministic task answers."""
    if len(sequences) != len(prompt_token_lens):
        raise RuntimeError(
            f"sequence/prompt length mismatch: {len(sequences)} != {len(prompt_token_lens)}"
        )
    G = max(1, int(group_size))
    out_sequences: list[list[int]] = []
    out_completions: list[list[int]] = []
    replaced = 0
    skipped = 0
    for idx, sequence in enumerate(sequences):
        prompt = prompts[idx // G]
        target = _target_answer_text(_user_prompt_text(prompt), task)
        if target is None:
            skipped += 1
            out_sequences.append(list(sequence))
            out_completions.append(list(completions[idx]))
            continue
        answer_tokens = [
            int(t) for t in chat_tokenizer.encode(target, add_special_tokens=False)
        ]
        if not answer_tokens:
            skipped += 1
            out_sequences.append(list(sequence))
            out_completions.append(list(completions[idx]))
            continue
        answer_start = int(prompt_token_lens[idx]) + int(filler_token_count)
        out_sequences.append(list(sequence[:answer_start]) + answer_tokens)
        out_completions.append(answer_tokens)
        replaced += 1
    return out_sequences, out_completions, replaced, skipped


def _lead_digit_frac(prompt_text: str, completion_text: str, task: str) -> float | None:
    """Fraction of the answer's leading digits the completion gets right — a
    GRADED counterpart of _score_answer for hard tasks where exact-match is ~0
    because the model is close but not digit-exact (e.g. 5-digit mult → 10-digit
    products: ~4-5 leading digits right, low-order wrong). This makes the buffer's
    contribution visible when binary exact-match can't. None if unparseable."""
    true = _target_answer_text(prompt_text, task)
    if task != "multiplication" or true is None:
        return None
    runs = re.findall(r"\d+", completion_text or "")
    if not runs:
        return 0.0
    cand = max(runs, key=len)  # the model's answer = the longest digit run
    k = 0
    for c1, c2 in zip(true, cand):
        if c1 != c2:
            break
        k += 1
    return k / len(true)


def _space_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _contains_marker(text: str, marker: str) -> bool:
    marker = marker or ""
    if not marker.strip():
        return False
    text_norm = _space_norm(text)
    marker_norm = _space_norm(marker)
    if marker_norm and marker_norm in text_norm:
        return True
    pieces = [p for p in re.split(r"\s+", marker.strip()) if p]
    if len(pieces) < 3:
        return False
    # Filler echoes are often partial ("! | ~ _ ..." without the full cue).
    # Count distinct marker pieces so repeated punctuation does not dominate.
    hits = sum(1 for p in set(pieces) if p and p.lower() in text.lower())
    return hits >= min(3, len(set(pieces)))


def _contains_stop_sequence(text: str, stop_sequences: list[str]) -> bool:
    if not text or not stop_sequences:
        return False
    return any(stop and stop in text for stop in stop_sequences)


def _longest_digit_run(text: str) -> str:
    runs = re.findall(r"\d+", text or "")
    return max(runs, key=len) if runs else ""


def _repeated_numeric_frac(texts: list[str]) -> float:
    keys = [_longest_digit_run(t) for t in texts]
    keys = [k for k in keys if k]
    if not keys:
        return 0.0
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) / len(keys)


def _reasoning_phrase_hit(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)(step-by-step|reasoning|calculation|set up|multiply|we need|let'?s|here'?s|wait)",
            text or "",
        )
    )


def _sample_answer_correctness(
    completions: list[list[int]],
    prompts: list[Any],
    group_size: int,
    chat_tokenizer: Any,
    task: str,
) -> list[int]:
    """Score each prompt-major completion against its prompt's gold answer.

    Returns one flag per completion: 1=correct, 0=wrong, -1=unscorable (no
    tokenizer / unparseable). Sample i belongs to prompt i // group_size.
    """
    if chat_tokenizer is None:
        return [-1] * len(completions)
    flags: list[int] = []
    for i, toks in enumerate(completions):
        text = chat_tokenizer.decode(toks, skip_special_tokens=False) if toks else ""
        prompt = prompts[i // max(1, group_size)] if prompts else None
        verdict = _score_answer(_user_prompt_text(prompt), text, task) if prompt is not None else None
        flags.append(-1 if verdict is None else int(verdict))
    return flags


def _generation_health(
    completions: list[list[int]],
    prompt_texts: list[str],
    chat_tokenizer: Any,
    task: str,
    *,
    max_new_tokens: int = 0,
    filler_marker: str = "",
    answer_cue_marker: str = "",
    stop_sequences: list[str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute generation-health + accuracy from on-policy student completions.

    Returns (metrics, per_sample_rows). `metrics` is wandb-loggable scalars;
    `per_sample_rows` are decoded samples for a wandb.Table / log line.
    """
    n = max(1, len(completions))
    lengths = [len(c) for c in completions]
    empty = sum(1 for L in lengths if L == 0)
    cap_hit = sum(1 for L in lengths if max_new_tokens > 0 and L >= max_new_tokens)
    has_think_close = 0
    has_digit = 0
    filler_leak = 0
    answer_cue_leak = 0
    stop_sequence_seen = 0
    reasoning_phrase = 0
    scored = 0
    correct = 0
    rows: list[dict[str, Any]] = []
    stop_sequences = stop_sequences or []
    for i, toks in enumerate(completions):
        text = (
            chat_tokenizer.decode(toks, skip_special_tokens=False)
            if (chat_tokenizer is not None and toks)
            else ""
        )
        if "</think>" in text:
            has_think_close += 1
        if any(ch.isdigit() for ch in text):
            has_digit += 1
        if _contains_marker(text, filler_marker):
            filler_leak += 1
        if _contains_marker(text, answer_cue_marker):
            answer_cue_leak += 1
        if _contains_stop_sequence(text, stop_sequences):
            stop_sequence_seen += 1
        if _reasoning_phrase_hit(text):
            reasoning_phrase += 1
        ptext = prompt_texts[i] if i < len(prompt_texts) else ""
        verdict = _score_answer(ptext, text, task)
        if verdict is not None:
            scored += 1
            correct += int(verdict)
        rows.append(
            {"prompt": ptext[:120], "completion": text[:200], "len": lengths[i], "correct": verdict}
        )
    metrics = {
        "eval/empty_frac": empty / n,
        "eval/mean_completion_tokens": sum(lengths) / n,
        "eval/max_completion_tokens": float(max(lengths) if lengths else 0),
        "eval/cap_hit_frac": cap_hit / n,
        "eval/has_think_close_frac": has_think_close / n,
        "eval/has_digit_frac": has_digit / n,
        "eval/filler_leak_frac": filler_leak / n,
        "eval/answer_cue_leak_frac": answer_cue_leak / n,
        "eval/stop_sequence_seen_frac": stop_sequence_seen / n,
        "eval/reasoning_phrase_frac": reasoning_phrase / n,
        "eval/num_scored": float(scored),
    }
    if scored > 0:
        # Scored TRAINING completions (the current step's prepare batches) — a
        # health/progress signal, NOT a real eval: it is temperature-sampled, on
        # the rolling train window, and meaningless under gold answer replacement
        # (sft_mode / opd_teacher_answer_source=gold scores the spliced targets).
        # The real eval is eval/accuracy from _heldout_greedy_eval.
        metrics["eval/train_window_accuracy"] = correct / scored
    return metrics, rows


def _eval_generation_max_tokens(config: "Config") -> int:
    configured = int(config.eval_max_new_tokens or 0)
    if configured > 0:
        return configured
    return max(1, int(config.max_new_tokens or 1))


def _corrupt_token_ids(token_ids: list[int], *, mode: str = "reverse") -> list[int]:
    if len(token_ids) <= 1:
        return list(token_ids)
    mode_lc, _preserve_boundary_ws = _corrupt_mode_parts(mode)
    if mode_lc in {"rotate", "roll", "cycle"}:
        return list(token_ids[1:]) + [int(token_ids[0])]
    if mode_lc not in {"reverse", "legacy"}:
        raise ValueError(
            f"unknown corrupt buffer mode {mode!r}; expected reverse, rotate, "
            "or *_preserve_ws variants"
        )
    return list(reversed(token_ids))


def _corrupt_mode_parts(mode: str) -> tuple[str, bool]:
    mode_lc = (mode or "reverse").strip().lower().replace("-", "_")
    preserve_boundary_ws = False
    for suffix in (
        "_preserve_boundary_ws",
        "_preserve_ws",
        "_boundary_ws",
        "_ws",
    ):
        if mode_lc.endswith(suffix):
            mode_lc = mode_lc[: -len(suffix)]
            preserve_boundary_ws = True
            break
    return mode_lc or "reverse", preserve_boundary_ws


def _corrupt_prefill_text(text: str, *, mode: str = "reverse") -> str:
    if not text:
        return ""
    mode_lc, preserve_boundary_ws = _corrupt_mode_parts(mode)
    leading_ws = ""
    working_text = text
    if preserve_boundary_ws:
        leading_ws_match = re.match(r"\s*", text)
        leading_ws = leading_ws_match.group(0) if leading_ws_match else ""
        working_text = text[len(leading_ws) :]
    pieces = re.findall(r"\S+\s*", working_text)
    if mode_lc in {"rotate", "roll", "cycle"}:
        if len(pieces) > 1:
            return leading_ws + "".join(pieces[1:] + pieces[:1])
        return text
    if mode_lc not in {"reverse", "legacy"}:
        raise ValueError(
            f"unknown corrupt buffer mode {mode!r}; expected reverse, rotate, "
            "or *_preserve_ws variants"
        )
    if len(pieces) > 1:
        return leading_ws + "".join(reversed(pieces))
    if preserve_boundary_ws and working_text:
        return leading_ws + working_text[::-1]
    return text[::-1]


def _leading_whitespace(text: str) -> str:
    match = re.match(r"\s*", text or "")
    return match.group(0) if match else ""


def _trailing_whitespace(text: str) -> str:
    match = re.search(r"\s*$", text or "")
    return match.group(0) if match else ""


def _control_prefill_delta_metrics(
    pause_prefill: Any,
    corrupt_prefill: Any,
    *,
    mode: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "eval/control_corrupt_pause_mode_preserve_boundary_ws": float(
            _corrupt_mode_parts(mode)[1]
        )
    }
    if isinstance(pause_prefill, list) and isinstance(corrupt_prefill, list):
        max_len = max(len(pause_prefill), len(corrupt_prefill))
        overlap = min(len(pause_prefill), len(corrupt_prefill))
        changed = abs(len(pause_prefill) - len(corrupt_prefill)) + sum(
            int(int(left) != int(right))
            for left, right in zip(pause_prefill[:overlap], corrupt_prefill[:overlap])
        )
        metrics.update(
            {
                "eval/control_corrupt_pause_changed_tokens": float(changed),
                "eval/control_corrupt_pause_change_frac": changed / max(1, max_len),
                "eval/control_corrupt_pause_len_delta_tokens": float(
                    len(corrupt_prefill) - len(pause_prefill)
                ),
            }
        )
        return metrics

    pause_text = str(pause_prefill or "")
    corrupt_text = str(corrupt_prefill or "")
    max_len = max(len(pause_text), len(corrupt_text))
    overlap = min(len(pause_text), len(corrupt_text))
    changed = abs(len(pause_text) - len(corrupt_text)) + sum(
        int(left != right)
        for left, right in zip(pause_text[:overlap], corrupt_text[:overlap])
    )
    metrics.update(
        {
            "eval/control_corrupt_pause_changed_chars": float(changed),
            "eval/control_corrupt_pause_change_frac": changed / max(1, max_len),
            "eval/control_corrupt_pause_len_delta_chars": float(
                len(corrupt_text) - len(pause_text)
            ),
            "eval/control_corrupt_pause_leading_ws_match": float(
                _leading_whitespace(pause_text) == _leading_whitespace(corrupt_text)
            ),
            "eval/control_corrupt_pause_trailing_ws_match": float(
                _trailing_whitespace(pause_text) == _trailing_whitespace(corrupt_text)
            ),
        }
    )
    return metrics



def _student_memory_prefix_token_count(config: "Config", chat_tokenizer: Any | None) -> int:
    exact_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
    if exact_prefill_ids:
        return len(exact_prefill_ids)
    if not (config.student_prefill_text and config.student_prefill_count > 0):
        return 0
    if chat_tokenizer is None:
        return int(config.student_prefill_count)
    return len(
        chat_tokenizer.encode(
            config.student_prefill_text * config.student_prefill_count,
            add_special_tokens=False,
        )
    )


def _corrupt_span_token_count(config: "Config", chat_tokenizer: Any | None, full_prefix_len: int) -> int:
    span = (config.opd_contrastive_corrupt_buffer_span or "full_prefix").strip().lower()
    if span in {"full", "full_prefix", "forced_prefix"}:
        return int(full_prefix_len)
    if span in {"memory", "memory_only", "pause", "pause_only"}:
        memory_len = _student_memory_prefix_token_count(config, chat_tokenizer)
        memory_len += max(0, int(config.student_generated_memory_tokens or 0))
        return min(int(full_prefix_len), memory_len)
    raise ValueError(
        "opd_contrastive_corrupt_buffer_span must be 'full_prefix' or 'memory_only', "
        f"got {config.opd_contrastive_corrupt_buffer_span!r}"
    )


def _mean_se_z(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    se = math.sqrt(max(variance, 0.0) / len(values))
    z = mean / se if se > 0 else 0.0
    return mean, se, z


async def _maybe_await_future(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _heldout_greedy_eval(
    config: "Config",
    sampling_clients: list[Any],
    eval_prompts: list[Any],
    chat_tokenizer: Any,
) -> dict[str, float]:
    """Greedy accuracy on held-out prompts through the live samplers.

    The REAL per-step eval (deployment condition: answer-cue prefill, greedy,
    stop at newline). Unlike the train-window health metric it is unaffected by
    sampling temperature, gold answer replacement, or the training window.
    """
    n = min(len(eval_prompts), max(0, int(config.eval_heldout_num_problems)))
    if n == 0 or not sampling_clients:
        return {}
    prompts_slice = eval_prompts[:n]
    stop_sequences = _parse_stop_sequences(config.student_stop_sequences)
    prefill = config.student_prefill_suffix or ""
    # Under student_prefill_token_ids the sampling clients run api_format=
    # "generate", which rejects chat-format prompts ("Request must have either
    # 'text' or 'input_ids'" — every heldout request failed on ARITH-012).
    # Build token-ids requests instead: prompt ids + the answer-cue suffix ids
    # (NO buffer — heldout is the no-buffer deployment condition by design;
    # it matched acc_nopause exactly on ARITH-009).
    exact_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
    heldout_suffix_ids: list[int] | None = None
    if exact_prefill_ids:
        if chat_tokenizer is None:
            logger.warning("heldout eval skipped: student_prefill_token_ids requires chat_tokenizer_path")
            return {}
        heldout_suffix_ids = (
            list(chat_tokenizer.encode(prefill, add_special_tokens=False)) if prefill else []
        )
    params = tomi.SamplingParams(
        max_tokens=_eval_generation_max_tokens(config),
        temperature=0.0,
        stop=stop_sequences or None,
        chat_continue_final_message=bool(prefill) and not exact_prefill_ids,
        **_chat_sampling_extras(config),
    )
    sem = asyncio.Semaphore(max(1, int(config.eval_control_max_concurrency or 64)))

    async def _one(i: int, prompt: Any) -> tuple[int, str | None]:
        if heldout_suffix_ids is not None:
            prompt_ids = _prompt_token_ids(prompt, chat_tokenizer)
            sample_prompt = tomi.ModelInput.from_ints(prompt_ids + heldout_suffix_ids)
        else:
            messages = (
                list(prompt)
                if isinstance(prompt, list)
                else [{"role": "user", "content": _user_prompt_text(prompt)}]
            )
            sample_prompt = messages + (
                [{"role": "assistant", "content": prefill}] if prefill else []
            )
        client = sampling_clients[i % len(sampling_clients)]
        async with sem:
            try:
                result = await asyncio.wait_for(
                    client.sample_async(
                        prompt=sample_prompt,
                        sampling_params=params,
                        num_samples=1,
                    ),
                    timeout=max(float(config.request_timeout), 1.0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("heldout eval request %d failed: %s", i, str(exc)[:120])
                return i, None
        seq = result.sequences[0]
        if seq.text:
            return i, seq.text
        text = (
            chat_tokenizer.decode(list(seq.tokens), skip_special_tokens=False)
            if chat_tokenizer is not None and seq.tokens
            else ""
        )
        return i, text
    results = await asyncio.gather(*(_one(i, p) for i, p in enumerate(prompts_slice)))
    scored = correct = failures = cap_hits = 0
    max_tokens = _eval_generation_max_tokens(config)
    for i, text in results:
        if text is None:
            failures += 1
            continue
        verdict = _score_answer(_user_prompt_text(prompts_slice[i]), text, config.eval_task)
        if verdict is not None:
            scored += 1
            correct += int(verdict)
        if chat_tokenizer is not None and len(chat_tokenizer.encode(text, add_special_tokens=False)) >= max_tokens:
            cap_hits += 1
    metrics: dict[str, float] = {
        "eval/heldout_n": float(n),
        "eval/heldout_scored": float(scored),
        "eval/heldout_request_failure_frac": failures / max(1, n),
        "eval/heldout_cap_hit_frac": cap_hits / max(1, n),
    }
    if scored > 0:
        metrics["eval/accuracy"] = correct / scored
    return metrics


def _static_control_arms(
    config: "Config",
    chat_tokenizer: Any,
) -> tuple[tuple[str, Any], ...]:
    """The static pause / nopause (+corrupt_pause) control prefills.

    Shared by the full greedy control eval and the periodic answer-logprob
    scoring. Token-id prefills yield list[int] arms; text prefills yield str.
    """
    exact_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
    if exact_prefill_ids:
        if chat_tokenizer is None:
            raise ValueError("student_prefill_token_ids requires chat_tokenizer_path for control eval")
        suffix_tokens = (
            [int(t) for t in chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False)]
            if config.student_prefill_suffix
            else []
        )
        control_arms: tuple[tuple[str, Any], ...] = (
            ("pause", exact_prefill_ids + suffix_tokens),
            ("nopause", suffix_tokens),
        )
        if config.eval_corrupt_pause_control:
            corrupt_mode = config.eval_corrupt_pause_mode or config.opd_contrastive_corrupt_buffer_mode
            control_arms = (
                *control_arms,
                (
                    "corrupt_pause",
                    _corrupt_token_ids(exact_prefill_ids, mode=corrupt_mode) + suffix_tokens,
                ),
            )
        return control_arms
    pause_prefix = config.student_prefill_text * config.student_prefill_count
    pause_block = pause_prefix + config.student_prefill_suffix
    nopause_block = config.student_prefill_suffix  # answer cue, 0 pause tokens
    control_arms = (("pause", pause_block), ("nopause", nopause_block))
    if config.eval_corrupt_pause_control and pause_prefix:
        corrupt_mode = config.eval_corrupt_pause_mode or config.opd_contrastive_corrupt_buffer_mode
        control_arms = (
            *control_arms,
            (
                "corrupt_pause",
                _corrupt_prefill_text(pause_prefix, mode=corrupt_mode)
                + config.student_prefill_suffix,
            ),
        )
    return control_arms


async def _answer_logprob_control_eval(
    config: "Config",
    sampling_clients: list[Any],
    eval_prompts: list[Any],
    chat_tokenizer: Any,
    control_arms: tuple[tuple[str, Any], ...],
    dynamic_prefills_by_arm: Mapping[str, list[Any]] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    force: bool = False,
) -> dict[str, float]:
    """Score the true answer tokens under each control prefix.

    `force=True` bypasses the eval_answer_logprob_control gate — used by the
    periodic `eval_answer_logprob_every` path, which runs this scoring on its
    own cadence independent of the end-gated control eval.
    """
    if not (config.eval_answer_logprob_control or force):
        return {}
    base_metrics = {
        "eval/answer_logprob_control_active": 1.0,
        "eval/answer_logprob_control_available": 0.0,
        "eval/answer_logprob_client_count": float(len(sampling_clients)),
    }
    if chat_tokenizer is None:
        return base_metrics
    if not sampling_clients:
        return base_metrics

    grouped: dict[int, dict[str, Any]] = {}
    item_meta: dict[tuple[int, int], tuple[str, int, str, int, int, int, int]] = {}
    distractor_enabled = bool(config.eval_answer_logprob_distractor_control)
    distractor_offset = max(1, int(config.eval_answer_logprob_distractor_offset or 1))
    answers_by_prompt: dict[int, str] = {}
    if distractor_enabled:
        for idx, eval_prompt in enumerate(eval_prompts):
            target = _target_answer_text(_user_prompt_text(eval_prompt), config.eval_task)
            if target is not None:
                answers_by_prompt[idx] = target

    def _distractor_answer(prompt_idx: int, answer: str) -> tuple[int, str] | None:
        if not distractor_enabled or len(answers_by_prompt) < 2:
            return None
        n_prompts = max(1, len(eval_prompts))
        for shift in range(distractor_offset, distractor_offset + n_prompts):
            candidate_idx = (prompt_idx + shift) % n_prompts
            candidate = answers_by_prompt.get(candidate_idx)
            if candidate is not None and candidate != answer:
                return candidate_idx, candidate
        return None

    distractor_pairs = 0
    distractor_skipped = 0
    for prompt_idx, prompt in enumerate(eval_prompts):
        prompt_text = _user_prompt_text(prompt)
        answer = _target_answer_text(prompt_text, config.eval_task)
        if answer is None:
            continue
        answer_tokens = [
            int(token)
            for token in chat_tokenizer.encode(answer, add_special_tokens=False)
        ]
        if not answer_tokens:
            continue
        target_answers: list[tuple[str, int, list[int]]] = [("correct", prompt_idx, answer_tokens)]
        distractor = _distractor_answer(prompt_idx, answer)
        if distractor is not None:
            distractor_idx, distractor_text = distractor
            distractor_tokens = [
                int(token)
                for token in chat_tokenizer.encode(distractor_text, add_special_tokens=False)
            ]
            if distractor_tokens:
                target_answers.append(("distractor", distractor_idx, distractor_tokens))
                distractor_pairs += 1
            else:
                distractor_skipped += 1
        elif distractor_enabled:
            distractor_skipped += 1
        prompt_tokens = _prompt_token_ids(prompt, chat_tokenizer)
        client_idx = prompt_idx % max(1, len(sampling_clients))
        group = grouped.setdefault(
            client_idx,
            {
                "client": sampling_clients[client_idx],
                "input_ids_batch": [],
                "starts": [],
            },
        )
        for arm, prefill in control_arms:
            prompt_prefills = dynamic_prefills_by_arm.get(arm) if dynamic_prefills_by_arm else None
            prompt_prefill = prompt_prefills[prompt_idx] if prompt_prefills is not None else prefill
            if isinstance(prompt_prefill, list):
                prefill_tokens = [int(token) for token in prompt_prefill]
            else:
                prefill_tokens = (
                    [
                        int(token)
                        for token in chat_tokenizer.encode(str(prompt_prefill), add_special_tokens=False)
                    ]
                    if prompt_prefill
                    else []
                )
            prefix = prompt_tokens + prefill_tokens
            score_start = max(0, len(prefix) - 1)
            for target_kind, target_prompt_idx, target_tokens in target_answers:
                local_idx = len(group["input_ids_batch"])
                group["input_ids_batch"].append(prefix + target_tokens)
                group["starts"].append(score_start)
                item_meta[(client_idx, local_idx)] = (
                    arm,
                    prompt_idx,
                    target_kind,
                    target_prompt_idx,
                    len(prefix),
                    len(target_tokens),
                    score_start,
                )

    total_items = sum(len(group["input_ids_batch"]) for group in grouped.values())
    if total_items == 0:
        return base_metrics

    configured_batch_size = max(0, int(config.eval_answer_logprob_batch_size or 0))
    configured_max_concurrency = max(0, int(config.eval_answer_logprob_max_concurrency or 0))

    score_specs: list[dict[str, Any]] = []
    max_chunk_size = 0
    for client_idx, group in grouped.items():
        group_size = len(group["input_ids_batch"])
        batch_size = configured_batch_size if configured_batch_size > 0 else group_size
        batch_size = max(1, batch_size)
        for chunk_start in range(0, group_size, batch_size):
            chunk_end = min(group_size, chunk_start + batch_size)
            chunk_size = chunk_end - chunk_start
            max_chunk_size = max(max_chunk_size, chunk_size)
            score_specs.append(
                {
                    "client_idx": client_idx,
                    "client": group["client"],
                    "chunk_start": chunk_start,
                    "input_ids_batch": group["input_ids_batch"][chunk_start:chunk_end],
                    "starts": group["starts"][chunk_start:chunk_end],
                }
            )

    effective_max_concurrency = configured_max_concurrency if configured_max_concurrency > 0 else len(score_specs)
    semaphore = asyncio.Semaphore(effective_max_concurrency) if configured_max_concurrency > 0 else None

    async def _score_chunk(spec: dict[str, Any]) -> Any:
        client = spec["client"]
        if hasattr(client, "score_prompt_logprobs_batch_async"):
            return await client.score_prompt_logprobs_batch_async(
                input_ids_batch=spec["input_ids_batch"],
                logprob_start_lens=spec["starts"],
            )
        if hasattr(client, "score_prompt_logprobs_batch"):
            return await _maybe_await_future(
                client.score_prompt_logprobs_batch(
                    input_ids_batch=spec["input_ids_batch"],
                    logprob_start_lens=spec["starts"],
                )
            )
        raise AttributeError("sampling client does not support prompt logprob scoring")

    async def _score_chunk_timed(spec: dict[str, Any]) -> tuple[float, float, float, Any]:
        queued_t0 = time.perf_counter()

        async def _run_once() -> tuple[float, Any]:
            service_t0 = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    _score_chunk(spec),
                    timeout=max(float(config.request_timeout), 1.0),
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as request failures below
                return _elapsed(service_t0), exc
            return _elapsed(service_t0), response

        if semaphore is not None:
            async with semaphore:
                queue_latency_s = _elapsed(queued_t0)
                service_latency_s, response = await _run_once()
                return queue_latency_s, service_latency_s, queue_latency_s + service_latency_s, response
        try:
            service_latency_s, response = await _run_once()
        except Exception as exc:  # pragma: no cover - _run_once converts scoring errors
            service_latency_s, response = _elapsed(queued_t0), exc
        return 0.0, service_latency_s, service_latency_s, response

    progress_every_s = max(0.0, float(config.eval_control_progress_log_every_s or 0.0))
    progress_t0 = time.perf_counter()
    last_progress_t = progress_t0
    completed_chunks = 0
    completed_items = 0
    exception_failed_items = 0
    timed_responses_by_idx: list[tuple[float, float, float, Any] | None] = [None] * len(score_specs)

    def _emit_answer_logprob_progress(force: bool = False) -> None:
        nonlocal last_progress_t
        if progress_callback is None:
            return
        now = time.perf_counter()
        if not force and progress_every_s <= 0.0:
            return
        if not force and now - last_progress_t < progress_every_s:
            return
        last_progress_t = now
        progress_callback(
            {
                "eval/answer_logprob_progress_active": 1.0,
                "eval/answer_logprob_progress_elapsed_s": now - progress_t0,
                "eval/answer_logprob_progress_total_chunks": float(len(score_specs)),
                "eval/answer_logprob_progress_completed_chunks": float(completed_chunks),
                "eval/answer_logprob_progress_pending_chunks": float(
                    max(0, len(score_specs) - completed_chunks)
                ),
                "eval/answer_logprob_progress_total_requests": float(total_items),
                "eval/answer_logprob_progress_completed_requests": float(completed_items),
                "eval/answer_logprob_progress_completion_frac": (
                    completed_items / max(1, total_items)
                ),
                "eval/answer_logprob_progress_exception_failures": float(exception_failed_items),
                "eval/answer_logprob_progress_exception_failure_frac": (
                    exception_failed_items / max(1, total_items)
                ),
                "eval/answer_logprob_progress_configured_max_concurrency": float(
                    configured_max_concurrency
                ),
                "eval/answer_logprob_progress_max_concurrency": float(
                    effective_max_concurrency
                ),
            }
        )

    async def _score_chunk_indexed(
        idx: int, spec: dict[str, Any]
    ) -> tuple[int, tuple[float, float, float, Any]]:
        return idx, await _score_chunk_timed(spec)

    tasks = [
        asyncio.create_task(_score_chunk_indexed(idx, spec))
        for idx, spec in enumerate(score_specs)
    ]
    _emit_answer_logprob_progress(force=True)
    for task in asyncio.as_completed(tasks):
        idx, timed_response = await task
        timed_responses_by_idx[idx] = timed_response
        completed_chunks += 1
        completed_items += len(score_specs[idx]["input_ids_batch"])
        if isinstance(timed_response[3], Exception):
            exception_failed_items += len(score_specs[idx]["input_ids_batch"])
        _emit_answer_logprob_progress(force=completed_chunks == len(score_specs))
    timed_responses = [
        response for response in timed_responses_by_idx if response is not None
    ]

    scores_by_arm: dict[str, dict[int, float]] = {}
    distractor_scores_by_arm: dict[str, dict[int, float]] = {}
    request_failures = 0
    queue_latencies = [queue_s for queue_s, _service_s, _total_s, _response in timed_responses]
    service_latencies = [service_s for _queue_s, service_s, _total_s, _response in timed_responses]
    chunk_latencies = [total_s for _queue_s, _service_s, total_s, _response in timed_responses]
    for spec, (_queue_s, _service_s, _total_s, response) in zip(score_specs, timed_responses):
        batch_size = len(spec["input_ids_batch"])
        if isinstance(response, Exception):
            request_failures += batch_size
            continue
        try:
            response_len = len(response)
        except TypeError:
            request_failures += batch_size
            continue
        if response_len != batch_size:
            request_failures += batch_size
            continue
        for local_idx, token_logprobs in enumerate(response):
            item_idx = int(spec["chunk_start"]) + local_idx
            (
                arm,
                prompt_idx,
                target_kind,
                _target_prompt_idx,
                prefix_len,
                answer_len,
                score_start,
            ) = item_meta[(int(spec["client_idx"]), item_idx)]
            full_len = prefix_len + answer_len
            if len(token_logprobs) >= full_len:
                answer_logprobs = token_logprobs[prefix_len:full_len]
            else:
                # Native SGLang commonly returns only entries from
                # `logprob_start_len` onward. Its first returned entry is
                # unscored/None, so this probe starts one token before the answer.
                row_offset = max(0, prefix_len - score_start)
                answer_logprobs = token_logprobs[row_offset : row_offset + answer_len]
            if len(answer_logprobs) != answer_len or any(value is None for value in answer_logprobs):
                request_failures += 1
                continue
            mean_logprob = sum(float(value) for value in answer_logprobs) / answer_len
            if target_kind == "distractor":
                distractor_scores_by_arm.setdefault(arm, {})[prompt_idx] = mean_logprob
            else:
                scores_by_arm.setdefault(arm, {})[prompt_idx] = mean_logprob

    metrics = dict(base_metrics)
    metrics["eval/answer_logprob_control_available"] = 1.0
    metrics["eval/answer_logprob_total_requests"] = float(total_items)
    metrics["eval/answer_logprob_request_failure_frac"] = request_failures / max(1, total_items)
    metrics["eval/answer_logprob_distractor_control_active"] = float(distractor_enabled)
    metrics["eval/answer_logprob_distractor_pairs"] = float(distractor_pairs)
    metrics["eval/answer_logprob_distractor_skipped"] = float(distractor_skipped)
    metrics["eval/answer_logprob_configured_batch_size"] = float(configured_batch_size)
    metrics["eval/answer_logprob_batch_size"] = float(max_chunk_size)
    metrics["eval/answer_logprob_chunk_count"] = float(len(score_specs))
    metrics["eval/answer_logprob_configured_max_concurrency"] = float(configured_max_concurrency)
    metrics["eval/answer_logprob_max_concurrency"] = float(effective_max_concurrency)
    metrics["eval/answer_logprob_bounded_concurrency_active"] = float(configured_max_concurrency > 0)
    latency_mean, latency_p95, latency_max = _mean_p95_max(chunk_latencies)
    metrics["eval/answer_logprob_chunk_latency_mean_s"] = latency_mean
    metrics["eval/answer_logprob_chunk_latency_p95_s"] = latency_p95
    metrics["eval/answer_logprob_chunk_latency_max_s"] = latency_max
    metrics["eval/answer_logprob_group_latency_mean_s"] = latency_mean
    metrics["eval/answer_logprob_group_latency_p95_s"] = latency_p95
    metrics["eval/answer_logprob_group_latency_max_s"] = latency_max
    queue_mean, queue_p95, queue_max = _mean_p95_max(queue_latencies)
    metrics["eval/answer_logprob_client_queue_latency_mean_s"] = queue_mean
    metrics["eval/answer_logprob_client_queue_latency_p95_s"] = queue_p95
    metrics["eval/answer_logprob_client_queue_latency_max_s"] = queue_max
    service_mean, service_p95, service_max = _mean_p95_max(service_latencies)
    metrics["eval/answer_logprob_service_latency_mean_s"] = service_mean
    metrics["eval/answer_logprob_service_latency_p95_s"] = service_p95
    metrics["eval/answer_logprob_service_latency_max_s"] = service_max

    for arm, scores in scores_by_arm.items():
        vals = list(scores.values())
        metrics[f"eval/answer_logprob_scored_{arm}"] = float(len(vals))
        metrics[f"eval/answer_logprob_mean_{arm}"] = sum(vals) / len(vals) if vals else 0.0

    select_by_arm: dict[str, dict[int, float]] = {}
    if distractor_enabled:
        for arm, scores in scores_by_arm.items():
            distractor_scores = distractor_scores_by_arm.get(arm, {})
            common = sorted(set(scores) & set(distractor_scores))
            select_values = {
                idx: scores[idx] - distractor_scores[idx]
                for idx in common
            }
            select_by_arm[arm] = select_values
            vals = list(select_values.values())
            select_mean, select_se, select_z = _mean_se_z(vals)
            metrics[f"eval/answer_logprob_distractor_scored_{arm}"] = float(
                len(distractor_scores)
            )
            metrics[f"eval/answer_logprob_select_margin_{arm}"] = select_mean
            metrics[f"eval/answer_logprob_select_margin_se_{arm}"] = select_se
            metrics[f"eval/answer_logprob_select_margin_z_{arm}"] = select_z
            metrics[f"eval/answer_logprob_select_margin_paired_n_{arm}"] = float(
                len(vals)
            )

    def _paired(left: str, right: str) -> tuple[float, float, float, int]:
        left_scores = scores_by_arm.get(left, {})
        right_scores = scores_by_arm.get(right, {})
        common = sorted(set(left_scores) & set(right_scores))
        diffs = [left_scores[idx] - right_scores[idx] for idx in common]
        mean, se, z = _mean_se_z(diffs)
        return mean, se, z, len(diffs)

    margin, margin_se, margin_z, paired_n = _paired("pause", "nopause")
    metrics.update(
        {
            "eval/answer_logprob_margin": margin,
            "eval/answer_logprob_margin_se": margin_se,
            "eval/answer_logprob_margin_z": margin_z,
            "eval/answer_logprob_margin_paired_n": float(paired_n),
        }
    )

    causal_margins = [margin]
    causal_zs = [margin_z]
    if "corrupt_pause" in scores_by_arm:
        corrupt_margin, corrupt_se, corrupt_z, corrupt_n = _paired("pause", "corrupt_pause")
        metrics.update(
            {
                "eval/answer_logprob_vs_corrupt_margin": corrupt_margin,
                "eval/answer_logprob_vs_corrupt_margin_se": corrupt_se,
                "eval/answer_logprob_vs_corrupt_margin_z": corrupt_z,
                "eval/answer_logprob_vs_corrupt_margin_paired_n": float(corrupt_n),
            }
        )
        causal_margins.append(corrupt_margin)
        causal_zs.append(corrupt_z)

    if distractor_enabled:
        def _paired_select(left: str, right: str) -> tuple[float, float, float, int]:
            left_scores = select_by_arm.get(left, {})
            right_scores = select_by_arm.get(right, {})
            common = sorted(set(left_scores) & set(right_scores))
            diffs = [left_scores[idx] - right_scores[idx] for idx in common]
            mean, se, z = _mean_se_z(diffs)
            return mean, se, z, len(diffs)

        select_delta, select_se, select_z, select_n = _paired_select("pause", "nopause")
        metrics.update(
            {
                "eval/answer_logprob_select_delta": select_delta,
                "eval/answer_logprob_select_delta_se": select_se,
                "eval/answer_logprob_select_delta_z": select_z,
                "eval/answer_logprob_select_delta_paired_n": float(select_n),
            }
        )
        select_causal_deltas = [select_delta]
        select_causal_zs = [select_z]
        if "corrupt_pause" in select_by_arm:
            (
                select_corrupt_delta,
                select_corrupt_se,
                select_corrupt_z,
                select_corrupt_n,
            ) = _paired_select("pause", "corrupt_pause")
            metrics.update(
                {
                    "eval/answer_logprob_select_vs_corrupt_delta": select_corrupt_delta,
                    "eval/answer_logprob_select_vs_corrupt_delta_se": select_corrupt_se,
                    "eval/answer_logprob_select_vs_corrupt_delta_z": select_corrupt_z,
                    "eval/answer_logprob_select_vs_corrupt_delta_paired_n": float(
                        select_corrupt_n
                    ),
                }
            )
            select_causal_deltas.append(select_corrupt_delta)
            select_causal_zs.append(select_corrupt_z)
        metrics["eval/answer_logprob_select_causal_delta"] = min(select_causal_deltas)
        metrics["eval/answer_logprob_select_causal_z_min"] = min(select_causal_zs)

    metrics["eval/answer_logprob_causal_margin"] = min(causal_margins)
    metrics["eval/answer_logprob_causal_z_min"] = min(causal_zs)
    return metrics


async def _buffer_control_eval(
    config: "Config",
    sampling_clients: list[Any],
    eval_prompts: list[Any],
    chat_tokenizer: Any,
    logprob_clients: list[Any] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """With/without-pause control eval — the genuine-improvement judge.

    Mirrors filler_tokens_rl's run_filler_eval: sample the SAME held-out prompts
    under two conditions — the trained pause prefill (`pause_text*count + suffix`)
    vs. an answer-cue-only prefill with ZERO pause tokens (`suffix`) — score both
    greedily, and report the delta. `eval/buffer_delta > 0` means the pause
    buffer is actually load-bearing; `~0` means OPD only improved direct
    answering (the no-op failure mode found on opd-8x8: pause==no_pause==82.8%).
    """
    stop_sequences = _parse_stop_sequences(config.student_stop_sequences)
    exact_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
    control_arms = _static_control_arms(config, chat_tokenizer)
    generated_memory_metrics: dict[str, float] = {
        "eval/generated_memory_control_active": 0.0,
    }
    dynamic_prefills_by_arm: dict[str, list[Any]] | None = None
    generated_control_active = bool(config.eval_generated_memory_control)
    generated_memory_tokens = max(
        0,
        int(config.eval_generated_memory_tokens or config.student_generated_memory_tokens or 0),
    )
    if generated_control_active:
        if exact_prefill_ids:
            raise ValueError("eval_generated_memory_control does not support student_prefill_token_ids")
        if chat_tokenizer is None:
            raise ValueError("eval_generated_memory_control requires chat_tokenizer_path")
        if not sampling_clients:
            raise ValueError("eval_generated_memory_control requires sampling clients")
        if generated_memory_tokens <= 0:
            raise ValueError(
                "eval_generated_memory_control requires eval_generated_memory_tokens "
                "or student_generated_memory_tokens > 0"
            )
        if not any(arm == "corrupt_pause" for arm, _ in control_arms):
            control_arms = (*control_arms, ("corrupt_pause", ""))

        seed_block = config.student_prefill_text * config.student_prefill_count
        seed_tokens = (
            [int(t) for t in chat_tokenizer.encode(seed_block, add_special_tokens=False)]
            if seed_block
            else []
        )
        suffix_tokens = (
            [int(t) for t in chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False)]
            if config.student_prefill_suffix
            else []
        )
        memory_params = tomi.SamplingParams(
            max_tokens=generated_memory_tokens,
            temperature=float(config.eval_generated_memory_temperature),
            stop=None,
            chat_continue_final_message=bool(seed_block),
            custom_params={"ignore_eos": True, "min_tokens": generated_memory_tokens},
        )
        memory_concurrency = max(0, int(config.eval_control_max_concurrency or 0))
        memory_semaphore = asyncio.Semaphore(memory_concurrency) if memory_concurrency > 0 else None

        async def _sample_eval_memory(prompt_idx: int, prompt: Any) -> tuple[int, Any, float]:
            if not (
                isinstance(prompt, list)
                and prompt
                and all(isinstance(message, dict) for message in prompt)
            ):
                return prompt_idx, ValueError("generated-memory control requires chat-message prompts"), 0.0
            client = sampling_clients[prompt_idx % len(sampling_clients)]
            memory_prompt = (
                list(prompt) + [{"role": "assistant", "content": seed_block}]
                if seed_block
                else list(prompt)
            )

            async def _run_once() -> tuple[int, Any, float]:
                t0 = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        client.sample(
                            prompt=memory_prompt,
                            sampling_params=memory_params,
                            num_samples=1,
                            return_logprobs=True,
                        ),
                        timeout=max(float(config.request_timeout), 1.0),
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced in metrics
                    return prompt_idx, exc, _elapsed(t0)
                return prompt_idx, response, _elapsed(t0)

            if memory_semaphore is not None:
                async with memory_semaphore:
                    return await _run_once()
            return await _run_once()

        memory_results = await asyncio.gather(
            *[
                _sample_eval_memory(prompt_idx, prompt)
                for prompt_idx, prompt in enumerate(eval_prompts)
            ]
        )
        memory_text_by_prompt = [""] * len(eval_prompts)
        memory_lengths: list[int] = []
        memory_latencies: list[float] = []
        memory_request_failures = 0
        memory_length_failures = 0
        for prompt_idx, response, latency_s in memory_results:
            memory_latencies.append(float(latency_s))
            if isinstance(response, Exception) or not getattr(response, "sequences", None):
                memory_request_failures += 1
                memory_lengths.append(0)
                continue
            sampled = response.sequences[0]
            tokens = _sampled_output_token_ids(sampled, chat_tokenizer)
            memory_lengths.append(len(tokens))
            if len(tokens) != generated_memory_tokens:
                memory_length_failures += 1
            memory_text_by_prompt[prompt_idx] = (
                chat_tokenizer.decode(tokens, skip_special_tokens=False)
                if hasattr(chat_tokenizer, "decode")
                else (getattr(sampled, "text", None) or "")
            )
        shuffle_offset = max(1, int(config.eval_generated_memory_shuffle_offset or 1))
        n_prompts = max(1, len(eval_prompts))
        pause_prefills = [
            seed_block + memory_text + config.student_prefill_suffix
            for memory_text in memory_text_by_prompt
        ]
        corrupt_prefills = [
            seed_block + memory_text_by_prompt[(idx + shuffle_offset) % n_prompts] + config.student_prefill_suffix
            for idx in range(len(eval_prompts))
        ]
        dynamic_prefills_by_arm = {
            "pause": pause_prefills,
            "corrupt_pause": corrupt_prefills,
        }
        memory_latency_mean, memory_latency_p95, memory_latency_max = _mean_p95_max(memory_latencies)
        generated_memory_metrics = {
            "eval/generated_memory_control_active": 1.0,
            "eval/generated_memory_tokens_requested": float(generated_memory_tokens),
            "eval/generated_memory_seed_tokens": float(len(seed_tokens)),
            "eval/generated_memory_suffix_tokens": float(len(suffix_tokens)),
            "eval/generated_memory_shuffle_offset": float(shuffle_offset),
            "eval/generated_memory_request_failure_frac": memory_request_failures / max(1, len(eval_prompts)),
            "eval/generated_memory_length_failure_frac": memory_length_failures / max(1, len(eval_prompts)),
            "eval/generated_memory_actual_tokens_mean": (
                sum(memory_lengths) / len(memory_lengths) if memory_lengths else 0.0
            ),
            "eval/generated_memory_actual_tokens_min": float(min(memory_lengths) if memory_lengths else 0),
            "eval/generated_memory_actual_tokens_max": float(max(memory_lengths) if memory_lengths else 0),
            "eval/generated_memory_latency_mean_s": memory_latency_mean,
            "eval/generated_memory_latency_p95_s": memory_latency_p95,
            "eval/generated_memory_latency_max_s": memory_latency_max,
            "eval/generated_memory_configured_max_concurrency": float(memory_concurrency),
        }

    def _control_prefill_for_prompt(arm: str, default_prefill: Any, prompt_idx: int) -> Any:
        if dynamic_prefills_by_arm is None:
            return default_prefill
        prompt_prefills = dynamic_prefills_by_arm.get(arm)
        if prompt_prefills is None:
            return default_prefill
        return prompt_prefills[prompt_idx]

    accs: dict[str, float] = {}
    leads: dict[str, float] = {}  # mean leading-correct-digit fraction (graded)
    scored_by_arm: dict[str, int] = {}
    correct_by_arm: dict[str, int] = {}
    arm_extra: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    max_tokens = _eval_generation_max_tokens(config)
    request_specs: list[tuple[str, Any, Any, Any, Any, float]] = []
    for arm, prefill in control_arms:
        for i, p in enumerate(eval_prompts):
            prompt_prefill = _control_prefill_for_prompt(arm, prefill, i)
            params = tomi.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                stop=stop_sequences or None,
                chat_continue_final_message=bool(prompt_prefill) and not exact_prefill_ids,
                **_chat_sampling_extras(config),
            )
            if exact_prefill_ids:
                prompt_ids = _prompt_token_ids(p, chat_tokenizer)
                sample_prompt = tomi.ModelInput.from_ints(prompt_ids + list(prompt_prefill))
            else:
                sample_prompt = (
                    list(p) + ([{"role": "assistant", "content": prompt_prefill}] if prompt_prefill else [])
                )
            client = sampling_clients[i % len(sampling_clients)]
            queued_t0 = time.perf_counter()
            request_specs.append((arm, p, client, sample_prompt, params, queued_t0))

    async def _await_sample_timed(
        future: Any,
        queued_t0: float,
        service_t0: float,
        queue_s: float,
    ) -> tuple[float, float, float, Any]:
        try:
            response = await asyncio.wait_for(future, timeout=max(float(config.request_timeout), 1.0))
        except Exception as exc:  # noqa: BLE001 - surfaced as request failures below
            return _elapsed(queued_t0), queue_s, _elapsed(service_t0), exc
        return _elapsed(queued_t0), queue_s, _elapsed(service_t0), response

    configured_max_concurrency = max(0, int(config.eval_control_max_concurrency or 0))
    effective_max_concurrency = (
        min(configured_max_concurrency, len(request_specs))
        if configured_max_concurrency > 0
        else len(request_specs)
    )
    progress_every_s = max(0.0, float(config.eval_control_progress_log_every_s or 0.0))
    progress_t0 = time.perf_counter()
    last_progress_t = progress_t0
    completed_requests = 0
    exception_failed_requests = 0
    timed_responses_by_idx: list[tuple[float, float, float, Any] | None] = [None] * len(request_specs)

    def _emit_control_progress(force: bool = False) -> None:
        nonlocal last_progress_t
        if progress_callback is None:
            return
        now = time.perf_counter()
        if not force and progress_every_s <= 0.0:
            return
        if not force and now - last_progress_t < progress_every_s:
            return
        last_progress_t = now
        progress_callback(
            {
                "eval/control_progress_active": 1.0,
                "eval/control_progress_elapsed_s": now - progress_t0,
                "eval/control_progress_total_requests": float(len(request_specs)),
                "eval/control_progress_completed_requests": float(completed_requests),
                "eval/control_progress_pending_requests": float(
                    max(0, len(request_specs) - completed_requests)
                ),
                "eval/control_progress_completion_frac": (
                    completed_requests / max(1, len(request_specs))
                ),
                "eval/control_progress_exception_failures": float(exception_failed_requests),
                "eval/control_progress_exception_failure_frac": (
                    exception_failed_requests / max(1, len(request_specs))
                ),
                "eval/control_progress_configured_max_concurrency": float(
                    configured_max_concurrency
                ),
                "eval/control_progress_max_concurrency": float(effective_max_concurrency),
            }
        )

    async def _run_request_indexed(
        idx: int,
        request_coro: Any,
    ) -> tuple[int, tuple[float, float, float, Any]]:
        return idx, await request_coro

    if configured_max_concurrency > 0:
        semaphore = asyncio.Semaphore(max(1, effective_max_concurrency))

        async def _submit_bounded(
            client: Any,
            sample_prompt: Any,
            params: Any,
            queued_t0: float,
        ) -> tuple[float, float, float, Any]:
            async with semaphore:
                queue_s = _elapsed(queued_t0)
                service_t0 = time.perf_counter()
                try:
                    future = client.sample(prompt=sample_prompt, sampling_params=params, num_samples=1)
                    return await _await_sample_timed(future, queued_t0, service_t0, queue_s)
                except Exception as exc:  # noqa: BLE001 - surfaced as request failures below
                    return _elapsed(queued_t0), queue_s, _elapsed(service_t0), exc

        request_tasks = [
            asyncio.create_task(
                _run_request_indexed(
                    idx,
                    _submit_bounded(client, sample_prompt, params, queued_t0),
                )
            )
            for idx, (_arm, _prompt, client, sample_prompt, params, queued_t0) in enumerate(
                request_specs
            )
        ]
    else:
        pending: list[tuple[Any, float, float, float]] = []
        for _arm, _prompt, client, sample_prompt, params, queued_t0 in request_specs:
            queue_s = _elapsed(queued_t0)
            service_t0 = time.perf_counter()
            pending.append(
                (
                    client.sample(prompt=sample_prompt, sampling_params=params, num_samples=1),
                    queued_t0,
                    service_t0,
                    queue_s,
                )
            )
        request_tasks = [
            asyncio.create_task(
                _run_request_indexed(
                    idx,
                    _await_sample_timed(future, queued_t0, service_t0, queue_s),
                )
            )
            for idx, (future, queued_t0, service_t0, queue_s) in enumerate(pending)
        ]
    _emit_control_progress(force=True)
    for task in asyncio.as_completed(request_tasks):
        idx, timed_response = await task
        timed_responses_by_idx[idx] = timed_response
        completed_requests += 1
        if isinstance(timed_response[3], Exception):
            exception_failed_requests += 1
        _emit_control_progress(force=completed_requests == len(request_specs))
    timed_responses = [
        response for response in timed_responses_by_idx if response is not None
    ]
    responses_by_arm: dict[str, list[tuple[Any, Any, float, float, float]]] = {
        arm: [] for arm, _ in control_arms
    }
    for (arm, prompt, *_rest), (latency_s, queue_s, service_s, response) in zip(request_specs, timed_responses):
        responses_by_arm[arm].append((prompt, response, latency_s, queue_s, service_s))

    for arm, _prefill in control_arms:
        correct = scored = 0
        lead_sum = 0.0
        lead_n = 0
        texts: list[str] = []
        lengths: list[int] = []
        has_think_close = 0
        filler_leak = 0
        answer_cue_leak = 0
        stop_sequence_seen = 0
        cap_hit = 0
        reasoning_phrase = 0
        filler_marker = config.student_prefill_text
        answer_cue_marker = config.student_prefill_suffix
        if exact_prefill_ids and chat_tokenizer is not None and hasattr(chat_tokenizer, "decode"):
            filler_marker = chat_tokenizer.decode(exact_prefill_ids, skip_special_tokens=False)
        request_failures = 0
        arm_responses = responses_by_arm.get(arm, [])
        request_latencies = [float(latency_s) for _p, _r, latency_s, _queue_s, _service_s in arm_responses]
        queue_latencies = [float(queue_s) for _p, _r, _latency_s, queue_s, _service_s in arm_responses]
        service_latencies = [float(service_s) for _p, _r, _latency_s, _queue_s, service_s in arm_responses]
        for p, r, _latency_s, _queue_s, _service_s in arm_responses:
            if isinstance(r, Exception) or not getattr(r, "sequences", None):
                request_failures += 1
                continue
            seq = r.sequences[0]
            seq_tokens = list(getattr(seq, "tokens", None) or [])
            text = getattr(seq, "text", None) or (
                chat_tokenizer.decode(seq_tokens, skip_special_tokens=True)
                if (chat_tokenizer is not None and seq_tokens) else ""
            )
            texts.append(text or "")
            lengths.append(len(seq_tokens))
            if "</think>" in (text or ""):
                has_think_close += 1
            if _contains_marker(text or "", filler_marker):
                filler_leak += 1
            if _contains_marker(text or "", answer_cue_marker):
                answer_cue_leak += 1
            if _contains_stop_sequence(text or "", stop_sequences):
                stop_sequence_seen += 1
            if len(seq_tokens) >= max_tokens:
                cap_hit += 1
            if _reasoning_phrase_hit(text or ""):
                reasoning_phrase += 1
            ptext = _user_prompt_text(p)
            v = _score_answer(ptext, text, config.eval_task)
            if v is not None:
                scored += 1
                correct += int(v)
            # Graded leading-correct-digit fraction — on hard tasks (5-digit mult)
            # exact-match is ~0 (model close but not digit-exact), so this reveals
            # whether the buffer helps the model compute MORE digits correctly.
            lf = _lead_digit_frac(ptext, text, config.eval_task)
            if lf is not None:
                lead_sum += lf
                lead_n += 1
            if arm == "pause" and len(rows) < 4:
                rows.append({"prompt": ptext[:80],
                             "completion": (text or "").strip()[:60], "correct": v})
        accs[arm] = correct / scored if scored else 0.0
        leads[arm] = lead_sum / lead_n if lead_n else 0.0
        scored_by_arm[arm] = scored
        correct_by_arm[arm] = correct
        denom = max(1, len(texts))
        request_denom = max(1, len(arm_responses))
        latency_mean, latency_p95, latency_max = _mean_p95_max(request_latencies)
        queue_latency_mean, queue_latency_p95, queue_latency_max = _mean_p95_max(queue_latencies)
        service_latency_mean, service_latency_p95, service_latency_max = _mean_p95_max(service_latencies)
        arm_extra[arm] = {
            "mean_completion_tokens": sum(lengths) / denom,
            "cap_hit_frac": cap_hit / denom,
            "has_think_close_frac": has_think_close / denom,
            "filler_leak_frac": filler_leak / denom,
            "answer_cue_leak_frac": answer_cue_leak / denom,
            "stop_sequence_seen_frac": stop_sequence_seen / denom,
            "reasoning_phrase_frac": reasoning_phrase / denom,
            "repeated_numeric_frac": _repeated_numeric_frac(texts),
            "request_failure_frac": request_failures / request_denom,
            "request_latency_mean_s": latency_mean,
            "request_latency_p95_s": latency_p95,
            "request_latency_max_s": latency_max,
            "client_queue_latency_mean_s": queue_latency_mean,
            "client_queue_latency_p95_s": queue_latency_p95,
            "client_queue_latency_max_s": queue_latency_max,
            "service_latency_mean_s": service_latency_mean,
            "service_latency_p95_s": service_latency_p95,
            "service_latency_max_s": service_latency_max,
        }
    pause_acc = accs.get("pause", 0.0)
    nopause_acc = accs.get("nopause", 0.0)
    corrupt_acc = accs.get("corrupt_pause", 0.0)
    pause_n = max(1, scored_by_arm.get("pause", 0))
    nopause_n = max(1, scored_by_arm.get("nopause", 0))
    corrupt_n = max(1, scored_by_arm.get("corrupt_pause", 0))
    delta_se = math.sqrt(
        max(pause_acc * (1.0 - pause_acc), 0.0) / pause_n
        + max(nopause_acc * (1.0 - nopause_acc), 0.0) / nopause_n
    )
    corrupt_delta_se = math.sqrt(
        max(pause_acc * (1.0 - pause_acc), 0.0) / pause_n
        + max(corrupt_acc * (1.0 - corrupt_acc), 0.0) / corrupt_n
    )
    metrics = {
        "eval/acc_pause": pause_acc,
        "eval/acc_nopause": nopause_acc,
        "eval/acc_corrupt_pause": corrupt_acc,
        "eval/buffer_delta": pause_acc - nopause_acc,
        "eval/buffer_delta_se": delta_se,
        "eval/buffer_delta_z": (pause_acc - nopause_acc) / delta_se if delta_se > 0 else 0.0,
        "eval/buffer_vs_corrupt_delta": pause_acc - corrupt_acc,
        "eval/buffer_vs_corrupt_delta_se": corrupt_delta_se,
        "eval/buffer_vs_corrupt_delta_z": (
            (pause_acc - corrupt_acc) / corrupt_delta_se if corrupt_delta_se > 0 else 0.0
        ),
        "eval/lead_pause": leads.get("pause", 0.0),
        "eval/lead_nopause": leads.get("nopause", 0.0),
        "eval/lead_corrupt_pause": leads.get("corrupt_pause", 0.0),
        "eval/buffer_lead_delta": leads.get("pause", 0.0) - leads.get("nopause", 0.0),
        "eval/buffer_vs_corrupt_lead_delta": (
            leads.get("pause", 0.0) - leads.get("corrupt_pause", 0.0)
        ),
        "eval/control_n": float(len(eval_prompts)),
        "eval/control_scored_pause": float(scored_by_arm.get("pause", 0)),
        "eval/control_scored_nopause": float(scored_by_arm.get("nopause", 0)),
        "eval/control_scored_corrupt_pause": float(scored_by_arm.get("corrupt_pause", 0)),
        "eval/control_max_completion_tokens": float(max_tokens),
        "eval/control_total_requests": float(len(request_specs)),
        "eval/control_num_arms": float(len(control_arms)),
        "eval/control_sampler_clients": float(len(sampling_clients)),
        "eval/control_arms_concurrent": 1.0,
        "eval/control_max_concurrency": float(effective_max_concurrency),
        "eval/control_configured_max_concurrency": float(configured_max_concurrency),
        "eval/control_bounded_concurrency_active": float(configured_max_concurrency > 0),
        "eval/control_corrupt_pause_active": float("corrupt_pause" in responses_by_arm),
        "eval/control_corrupt_pause_mode": config.eval_corrupt_pause_mode
        or config.opd_contrastive_corrupt_buffer_mode,
    }
    metrics.update(generated_memory_metrics)
    control_prefills = {
        arm: _control_prefill_for_prompt(arm, prefill, 0)
        for arm, prefill in control_arms
    }
    if "pause" in control_prefills and "corrupt_pause" in control_prefills:
        metrics.update(
            _control_prefill_delta_metrics(
                control_prefills["pause"],
                control_prefills["corrupt_pause"],
                mode=config.eval_corrupt_pause_mode
                or config.opd_contrastive_corrupt_buffer_mode,
            )
        )
    for arm, vals in arm_extra.items():
        for name, value in vals.items():
            metrics[f"eval/{arm}_{name}"] = value
    causal_margins = [
        metrics["eval/buffer_delta"],
    ]
    if "corrupt_pause" in responses_by_arm:
        causal_margins.append(metrics["eval/buffer_vs_corrupt_delta"])
    causal_zs = [
        metrics["eval/buffer_delta_z"],
    ]
    if "corrupt_pause" in responses_by_arm:
        causal_zs.append(metrics["eval/buffer_vs_corrupt_delta_z"])
    cap_fracs = [vals["cap_hit_frac"] for vals in arm_extra.values()]
    request_failure_fracs = [vals["request_failure_frac"] for vals in arm_extra.values()]
    request_latency_means = [vals["request_latency_mean_s"] for vals in arm_extra.values()]
    request_latency_p95s = [vals["request_latency_p95_s"] for vals in arm_extra.values()]
    request_latency_maxes = [vals["request_latency_max_s"] for vals in arm_extra.values()]
    queue_latency_means = [vals["client_queue_latency_mean_s"] for vals in arm_extra.values()]
    queue_latency_p95s = [vals["client_queue_latency_p95_s"] for vals in arm_extra.values()]
    queue_latency_maxes = [vals["client_queue_latency_max_s"] for vals in arm_extra.values()]
    service_latency_means = [vals["service_latency_mean_s"] for vals in arm_extra.values()]
    service_latency_p95s = [vals["service_latency_p95_s"] for vals in arm_extra.values()]
    service_latency_maxes = [vals["service_latency_max_s"] for vals in arm_extra.values()]
    stop_sequence_seen_fracs = [vals["stop_sequence_seen_frac"] for vals in arm_extra.values()]
    metrics.update(
        {
            "eval/buffer_causal_margin": min(causal_margins),
            "eval/buffer_causal_z_min": min(causal_zs),
            "eval/control_cap_hit_frac_max": max(cap_fracs) if cap_fracs else 0.0,
            "eval/control_cap_hit_frac_mean": sum(cap_fracs) / len(cap_fracs) if cap_fracs else 0.0,
            "eval/control_request_failure_frac_max": (
                max(request_failure_fracs) if request_failure_fracs else 0.0
            ),
            "eval/control_request_latency_mean_s": (
                sum(request_latency_means) / len(request_latency_means)
                if request_latency_means
                else 0.0
            ),
            "eval/control_request_latency_p95_s": (
                max(request_latency_p95s) if request_latency_p95s else 0.0
            ),
            "eval/control_request_latency_max_s": (
                max(request_latency_maxes) if request_latency_maxes else 0.0
            ),
            "eval/control_client_queue_latency_mean_s": (
                sum(queue_latency_means) / len(queue_latency_means)
                if queue_latency_means
                else 0.0
            ),
            "eval/control_client_queue_latency_p95_s": (
                max(queue_latency_p95s) if queue_latency_p95s else 0.0
            ),
            "eval/control_client_queue_latency_max_s": (
                max(queue_latency_maxes) if queue_latency_maxes else 0.0
            ),
            "eval/control_service_latency_mean_s": (
                sum(service_latency_means) / len(service_latency_means)
                if service_latency_means
                else 0.0
            ),
            "eval/control_service_latency_p95_s": (
                max(service_latency_p95s) if service_latency_p95s else 0.0
            ),
            "eval/control_service_latency_max_s": (
                max(service_latency_maxes) if service_latency_maxes else 0.0
            ),
            "eval/control_stop_sequence_seen_frac_max": (
                max(stop_sequence_seen_fracs) if stop_sequence_seen_fracs else 0.0
            ),
            "eval/control_stop_sequence_seen_frac_mean": (
                sum(stop_sequence_seen_fracs) / len(stop_sequence_seen_fracs)
                if stop_sequence_seen_fracs
                else 0.0
            ),
        }
    )
    metrics.update(
        await _answer_logprob_control_eval(
            config,
            logprob_clients if logprob_clients is not None else sampling_clients,
            eval_prompts,
            chat_tokenizer,
            control_arms,
            dynamic_prefills_by_arm=dynamic_prefills_by_arm,
            progress_callback=progress_callback,
        )
    )
    return metrics, rows


@chz.chz
class Config:
    base_url: str = "http://127.0.0.1:6000"
    teacher_base_url: str = "http://127.0.0.1:30002"
    # "xorl" (training-framework forward) or "sglang" (xorl-sglang-internal
    # /teacher_hidden_cache microservice — faster prefill, fewer GPUs).
    teacher_backend: str = "xorl"
    inference_base_urls: str = "http://127.0.0.1:30001"
    inference_logprob_base_urls: str = ""  # optional native SGLang /generate endpoints for answer-logprob scoring
    # Dedicated eval pool (2026-06-11): comma-separated SGLang endpoints that
    # serve the held-out/control greedy evals instead of the training samplers.
    # These endpoints must be registered with the trainer under pool="eval"
    # (the generator does this); the per-step weight sync then covers only
    # pools=["default"], and the eval pool is refreshed with a pools=["eval"]
    # sync at eval steps — so its weights stay frozen for the whole eval window.
    eval_inference_base_urls: str = ""
    # Run the eval bundle as a background task (requires eval_inference_base_urls):
    # training continues while the eval runs against the frozen eval pool; results
    # land as a late `row_kind="eval_async"` profile row stamped with the step they
    # evaluate. The terminal control eval (eval_control_start_step) always runs
    # BLOCKING so the autopilot verdict gate is unchanged.
    eval_async: bool = False
    inference_port: int = 30001
    inference_api_format: str = ""
    sampler_metrics_urls: str = ""      # comma-separated SMG Prometheus base URLs; empty disables, "auto" infers :29000
    sampler_metrics_port: int = 29000
    sampler_metrics_timeout: float = 2.0
    sampler_quiesce_before_sync: bool = False
    sampler_quiesce_timeout: float = 120.0
    sampler_quiesce_poll_s: float = 2.0
    sampler_quiesce_max_new_outstanding: float = 0.0
    sampler_quiesce_max_connections_active: float = 0.0
    sampler_quiesce_max_inflight: float = 0.0
    model_name: str = "default"
    model_id: str = "default"
    teacher_model_id: str = "default"
    teacher_head: str = ""
    chat_tokenizer_path: str = ""
    # Server-side chat-template kwargs for ALL chat-completions sampling/eval
    # requests (JSON). Default pins Qwen3.x to the closed-think rendering —
    # the server default renders assistant prefills inside an OPEN <think>
    # block (2026-06-10 PTC-118 repro root cause). Set to "" to defer to the
    # server default (NOT recommended).
    chat_template_kwargs: str = '{"enable_thinking": false}'
    # SFT ablation mode (2026-06-10): identical loop/data/eval/sync, but train with
    # plain teacher-forced cross-entropy on the answer tokens instead of the OPD
    # reverse-KL against the CoT-conditioned teacher. No teacher prefill at all.
    # Combine with opd_teacher_answer_source=gold so the targets are the spliced
    # gold answers (the teacher's answers are 98.7% gold on 4x4). Answers the
    # "is on-policy distillation necessary or does SFT suffice?" question.
    sft_mode: bool = False
    # Per-step REAL eval (2026-06-10): every eval_accuracy_every steps, greedy-decode
    # this many held-out prompts (the control-eval pool tail) through the samplers
    # and report eval/accuracy. Replaces the old semantics where eval/accuracy was
    # the scored TRAINING samples (now eval/train_window_accuracy). 0 disables.
    eval_heldout_num_problems: int = 256
    output_dir: str = "/tmp/xorl-client-opd"
    profile_output: str = ""

    num_steps: int = 2
    profile_warmup_steps: int = 1
    num_prompts: int = 2
    # Per-step prompt window. When 0 (default), every step samples + trains on
    # ALL `num_prompts` (on-policy: same prompt pool re-sampled each step). When
    # > 0, step N uses a rolling, non-overlapping window of this many prompts
    # (prompts[N*pps : (N+1)*pps], wrapping at the end of the pool), so a run of
    # K steps sees K*pps distinct prompts before any repeat. Lets a small
    # per-step batch (fast steps) still cover a large, diverse prompt set.
    prompts_per_step: int = 0
    # In-loop eval (added 2026-05-28). Loss convergence is NOT a success signal
    # for OPD: the student can reward-hack by emitting EOS immediately after the
    # pause prefill → empty completions → trivial KL (loss → 0 while accuracy
    # → 0). These knobs surface generation health + task accuracy + decoded
    # samples every step so the collapse is visible (and auto-abortable).
    eval_health_every: int = 1          # log generation-health metrics every N steps (0=off)
    eval_accuracy_every: int = 0        # dedicated held-out temp=0 accuracy eval every N steps (0=off)
    eval_num_problems: int = 64         # held-out accuracy-eval problem count
    eval_temperature: float = 0.0       # accuracy-eval sampling temperature
    eval_max_new_tokens: int = 0        # 0 -> max_new_tokens
    eval_task: str = "multiplication"   # answer scorer; "none" disables scoring
    eval_log_samples: int = 8           # decoded samples logged to wandb each step
    eval_corrupt_pause_control: bool = True  # add same-length corrupted-pause arm to control eval
    eval_corrupt_pause_mode: str = ""        # empty -> opd_contrastive_corrupt_buffer_mode
    eval_answer_logprob_control: bool = False  # score true-answer logprob under pause/no-pause/corrupt prefixes
    # Run the gold-answer logprob scoring every N steps on the held-out eval set,
    # INDEPENDENT of eval_control_start_step (which end-gates the full greedy
    # control eval). Cheap (prefill-only scoring, no generation), so it gives the
    # gold-logprob (+ distractor-margin) trajectory across the whole run —
    # distinguishing "mass moves away from gold" from "sampling degrades around an
    # intact mode" when accuracy decays. 0 = off.
    eval_answer_logprob_every: int = 0
    eval_answer_logprob_batch_size: int = 0  # 0 keeps legacy one prompt-scoring batch per client
    eval_answer_logprob_max_concurrency: int = 0  # 0 keeps legacy unbounded chunk concurrency
    eval_answer_logprob_distractor_control: bool = False  # also score paired wrong answers for answer-selection margins
    eval_answer_logprob_distractor_offset: int = 1  # wrong answer comes from prompt index + offset modulo eval set
    eval_control_max_concurrency: int = 0  # 0 keeps legacy unbounded control request fanout
    eval_control_start_step: int = 0       # first step allowed to run the held-out control eval
    eval_control_progress_log_every_s: float = 30.0  # W&B/log heartbeat interval during long control evals
    eval_generated_memory_control: bool = False  # control eval uses per-prompt generated memory instead of static filler
    eval_generated_memory_tokens: int = 0  # 0 -> student_generated_memory_tokens
    eval_generated_memory_shuffle_offset: int = 1  # corrupt arm uses prompt index + this offset
    eval_generated_memory_temperature: float = 0.0  # memory generation temperature for eval controls
    eval_abort_empty_frac: float = 0.9  # auto-abort if empty_frac >= this ...
    eval_abort_patience: int = 0        # ... for this many consecutive steps (0=never abort)
    eval_prompts_json_path: str = ""    # held-out eval prompts; falls back to pool tail
    # Optional fb-replay capture. When set, the first matching trainer
    # forward_backward request is dumped as self-contained JSON plus copied
    # teacher-cache assets, so server-only replay benchmarks can bypass
    # sampling, teacher prefill, optimizer, and inference weight sync.
    forward_backward_capture_path: str = ""
    forward_backward_capture_step: int = 0
    forward_backward_capture_train_batch: int = 0
    opd_microbatch_size: int = 0
    opd_prepare_batch_size: int = 0
    opd_prepare_concurrency: int = 1
    # Strict same-step prepare overlap. When >1, a single prepare batch is split
    # into this many prompt chunks, each chunk samples then teacher-prefills as
    # soon as its own samples are ready, and the per-chunk teacher caches are
    # merged back into one cache before the trainer sees the batch. This preserves
    # fresh same-step samples and a single final fb call. Default 0/off.
    opd_strict_prepare_overlap_chunks: int = 0
    prompt_len: int = 32
    prompts_json: str = ""
    prompts_json_path: str = ""
    max_new_tokens: int = 8
    temperature: float = 1.0

    request_timeout: float = 900.0
    endpoint_timeout: float = 7200.0
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    skip_optim_step: bool = False

    sync_weights: bool = True
    sync_method: str = "p2p"
    weight_sync_master_address: str | None = None
    weight_sync_timeout: float = 1800.0

    opd_kl_backend: str = "streaming"
    opd_vocab_chunk_size: int | None = None
    opd_sharded_head_device_cache: bool = True

    # PR #320 (xorl-internal) — VERL-parity OPD loss controls.
    # `opd_loss_mode`: one of `reverse_kl_full` (default), `forward_kl_full`,
    #   `kl`/`k1`/`abs`/`mse`/`k2`/`low_var_kl`/`k3` (single-sample estimators;
    #   trailing `+` applies k2 straight-through gradient trick).
    # `opd_emit_full_vocab_diagnostics`: enable extra metrics (teacher/student
    #   entropy, top1 agreement, loss min/max/abs-mean). Slight per-batch overhead.
    # `opd_use_policy_gradient`: PG mode (PPO advantage = -distillation_loss).
    # `opd_loss_max_clamp`: symmetric per-token clamp.
    opd_loss_mode: str = "reverse_kl_full"
    opd_emit_full_vocab_diagnostics: bool = False
    opd_use_policy_gradient: bool = False
    opd_loss_max_clamp: float | None = None
    # Hidden-state matching coefficient (only meaningful with supervise_student_cot):
    # adds coef * cosine-distance(student_hidden, teacher_hidden) at valid (pause+
    # answer) positions, forcing the pause buffer toward the teacher's post-CoT
    # hidden state — the encoded-reasoning lever logit-KL alone can't provide.
    # 0 = off (default). Try 0.1–1.0; watch eval/buffer_delta + opd_hidden_match_loss.
    opd_hidden_match_coef: float = 0.0
    # KL term weight (default 1.0). Set 0.0 to supervise ONLY on hidden states
    # (loss = kl_loss_weight*KL + hidden_match_coef*hidden_match). Pairs with
    # opd_hidden_match_coef>0 for a hidden-state-primary OPD objective.
    opd_kl_loss_weight: float = 1.0
    # Hidden-match distance: "cosine" (default) or "mse". MSE is magnitude-aware so
    # MSE->0 implies matched logits via the shared head (cosine decouples from accuracy).
    opd_hidden_match_mode: str = "cosine"
    # ---- Multi-layer OPRD (all-layer hidden matching) ----
    # When non-empty, multi-layer OPRD is ON: the trainer matches the student's
    # decoder-layer subset against the SAME subset of teacher-sequence hiddens
    # (self-distillation: teacher = the same frozen weights run on the teacher
    # sequence prompt+CoT+pause+answer), via normalized MSE averaged over layers
    # (REPLACES the single-layer hidden term, scaled by opd_hidden_match_coef).
    # By default the teacher per-layer hiddens are recomputed trainer-side. The
    # optional opd_oprd_cache_backend="sglang" A/B asks SGLang to write a rank-3
    # per-layer cache instead. "" = off (default, byte-identical).
    # Forms: "every4" (every 4th decoder layer) or a comma list e.g. "0,8,16,24".
    # Pairs with opd_hidden_match_coef>0.
    opd_oprd_layers: str = ""
    # Total decoder layers in the model, used ONLY to expand "everyN" into explicit
    # layer indices client-side so the teacher and trainer pick the SAME subset. Not
    # needed for an explicit comma list. 0 = unknown (then "everyN" is rejected).
    opd_oprd_num_layers: int = 0
    # Restrict the OPRD term to the last-k supervised positions per sample (0 = all).
    opd_oprd_last_k: int = 0
    # Multi-layer OPRD teacher layer source. "trainer" preserves the current path:
    # the trainer runs a no-grad teacher forward inside forward_backward. "sglang"
    # asks /teacher_hidden_cache to write the rank-3 layer cache and passes that to
    # the trainer, moving this work out of f/b. Use "sglang" as an explicit
    # throughput/correctness-gated A/B; default remains byte-identical.
    opd_oprd_cache_backend: str = "trainer"
    # Multi-layer OPRD student layer capture. "selected_hooks" captures only
    # supervised-position rows from decoder-layer forward hooks, preserving the
    # same OPRD loss while avoiding full [batch, seq, layers, d] retention.
    # "output_hidden_states" is the legacy full-sequence capture fallback.
    opd_oprd_student_capture: str = "selected_hooks"
    # "buffer = CoT length (K=C)" mode (CLIENT-ONLY orchestration; the trainer is
    # already per-position and needs no change). When True, each prompt's student
    # pause buffer is sized to K_i = C_i = len(that prompt's teacher CoT tokens),
    # replacing the GLOBAL student_filler_count with a PER-SAMPLE K_i, and the
    # teacher cache + loss use teacher_cot_mode="match_cot" + supervise_student_cot
    # to align student-buffer-position-i <-> teacher-CoT-position-i 1:1 (full CoT,
    # no masking). Default False = byte-identical to the global-prefill behavior.
    # NOTE: this is never sent in the trainer params dict.
    opd_buffer_equals_cot: bool = False
    profile_sync_cuda: bool = False

    # Teacher-prefix prepend for context-distillation recipes (e.g., teacher
    # sees an instruction system prompt the student does not). If non-empty,
    # `_teacher_hidden_cache_data` prepends these tokens to the teacher input
    # and marks their target_tokens as IGNORE_INDEX (-100) so the returned
    # `cache_indices_by_sample` aligns 1-to-1 with the student tokens.
    teacher_system_prefix: str = ""
    teacher_system_prefix_path: str = ""

    # Teacher filler-token INSERTION at the user→assistant boundary. The student
    # generates trajectories directly from the user prompt; the teacher sees
    # `teacher_filler_count` repetitions of `teacher_filler_text` inserted right
    # at the position where the assistant turn begins (= prompt_token_len). This
    # gives the teacher a "thinking budget" of fixed-content filler tokens that
    # the student does not get, matching the filler-eval methodology that showed
    # `pause`/`lorem`/`ellipsis` lift Qwen3.6-35B accuracy +4-6pp at 100 fillers.
    teacher_filler_text: str = ""       # e.g. " pause" (single token in Qwen tokenizer)
    teacher_filler_count: int = 0       # number of filler tokens to insert

    # Per-prompt teacher CoT (Run B recipe). When set, OVERRIDES teacher_filler_text/
    # teacher_filler_count with a per-sample filler taken from a JSON file produced
    # by experiments/opd_profile/cot_precompute.py. Schema: list of dicts with at
    # least a "cot" field; order matches the prompts JSON. The CoT text is tokenized
    # with the chat_tokenizer once at startup, and the i-th prompt uses
    # teacher_cot_tokens_by_prompt[i] as its filler insertion.
    teacher_cot_json_path: str = ""

    # Student-side prefill (Run B): the student SAMPLES with a forced assistant
    # prefix. Text mode uses an open assistant turn prefilled with
    # `student_prefill_text` × `student_prefill_count`; exact-token mode uses
    # `student_prefill_token_ids` as the full buffer and sends prompt+IDs through
    # SGLang /generate. The student forward pass (and OPD loss target) sees those
    # forced tokens; the teacher's CoT-filler replaces or conditions those
    # positions when reconstructing teacher_seq.
    student_prefill_text: str = ""
    student_prefill_count: int = 0
    # Exact token IDs for the prefill buffer, before student_prefill_suffix. This
    # avoids printable-text leakage and tokenizer split ambiguity for random-token
    # experiments. Accepts JSON (`[1,2,3]`) or comma/space-separated IDs.
    student_prefill_token_ids: str = ""
    # Answer lead-in appended after the pause buffer (e.g. "</think>Answer: ").
    # Closes the think block and cues the answer so the student actually emits
    # answer tokens; without it the student collapses to EOS (see runbook §3).
    # The full forced prefix (pause + suffix) is the masked filler region K.
    student_prefill_suffix: str = ""
    # Dynamic generated-memory mode. When >0, the student first generates this
    # many memory tokens from the prompt plus optional student_prefill_text/count
    # seed, then answers from that generated memory plus student_prefill_suffix.
    # The generated memory must hit the exact requested length so the existing
    # scalar-K OPD cache remap remains valid.
    student_generated_memory_tokens: int = 0
    # Optional stop sequences applied to both on-policy student rollouts and the
    # pause/no-pause control eval. Accepts JSON (`["\n"]`) or pipe-separated text.
    student_stop_sequences: str = ""
    # How the teacher CoT relates to the student's pause region:
    #   "replace" — teacher_seq = prompt + CoT + answer (CoT in place of pause)
    #   "insert"  — teacher_seq = prompt + CoT + pause + answer (CoT prepended,
    #               pause kept; identical answer left-context to the student).
    teacher_cot_mode: str = "replace"

    # Mask prompt positions (region 0) out of the OPD KL — reference-OPD
    # semantics (verl GKD masks KL to response positions only). The teacher's
    # CoT is inserted AFTER the prompt, so the prompt-position term is a pure
    # anchor-to-base regularizer that dilutes the per-answer-token gradient
    # ~1/frac_answer under global-valid-token normalization (~6-8x on
    # short-answer tasks). True = supervise the same positions as sft_mode
    # (buffer+answer). NB: every run before 2026-06-11 (incl. the 4x4 0.905)
    # effectively ran False — set False explicitly to reproduce those.
    opd_mask_prompt_kl: bool = True

    # Correct-prefix filtering (science runbook §7.1, the surviving bootstrap
    # lever after temperature + OPRD-coef were exhausted). When True, the OPD/KL
    # loss supervises ONLY on-policy samples whose sampled answer scored correct
    # (per eval_task); samples that are wrong or unverifiable are masked out
    # entirely. Rationale: on a floored task ~95% of sampled answer prefixes are
    # wrong and the CoT-teacher is itself confused there, so distilling those
    # conditionals erodes competence — restrict supervision to the right
    # manifold. Default False (no behavior change). Shrinks valid tokens/step,
    # so consider raising prompts/step. After enabling, run the infra §9c step-0
    # KL gate (datum-path change).
    opd_correct_prefix_only: bool = False

    # Expansion-lock filter (filler-RFT pivot, 2026-06-27). Keep a sampled
    # rollout as a positive training datum ONLY when the with-pause answer is
    # correct AND the model FAILS the same problem WITHOUT the pause buffer
    # (greedy no-pause). This trains exactly the filler-DEPENDENT frontier — the
    # runbook's "lock in the pass@8 expansion as reliable pass@1" — instead of
    # plain RFT, which teaches the arithmetic in-weights so acc_nopause rises
    # until the buffer is redundant (buffer_delta collapses; observed run-1).
    # Implemented by sampling one greedy no-pause completion per prompt (the eval
    # "nopause" arm = answer cue only), scoring it, and zeroing sample_ok for any
    # prompt already solved without the buffer so opd_correct_prefix_only masks
    # it. REQUIRES opd_correct_prefix_only=true. Shrinks valid tokens/step (only
    # the gap subset survives) — pair with a larger prompts/step. Adds one greedy
    # no-pause sampling pass per step. Default False (no behavior change).
    opd_only_train_on_pause_gap: bool = False

    # Pause-position-supervised variant (only meaningful with teacher_cot_mode=
    # "insert"). When False (default) the student's pause/CoT region is MASKED
    # out of the loss, so only answer positions are distilled — the buffer is
    # never trained to reason (empirically a no-op: removing the pause changes
    # nothing). When True, the pause region is KEPT in the teacher cache and the
    # student's pause positions are SUPERVISED against the teacher's post-CoT
    # pause hiddens (KL on student-CoT + answer): the student must reconstruct
    # the teacher's reasoning state in its buffer without seeing the CoT. NOTE:
    # logit-KL here can be weak if the teacher's pause-position next-token
    # distribution is trivially "pause" — the full-vocab overlap_ratio logged on
    # those positions tells you whether there is real signal to distill.
    supervise_student_cot: bool = False

    # Buffer-only supervision: mask BOTH the CoT and the answer (teacher cache +
    # student loss), so ONLY the K buffer positions are distilled. Forces the model
    # to encode reasoning into the buffer — it cannot internalize via the answer.
    # Pair with opd_hidden_match_coef>0; requires supervise_student_cot=true + insert.
    opd_supervise_buffer_only: bool = False
    # Add a same-length corrupted-buffer copy of each OPD sample. The corrupt copy
    # has zero KL weight and negative hidden-match weight on buffer positions, so
    # training optimizes real buffer states toward teacher CoT states while pushing
    # corrupted buffer states away.
    opd_contrastive_corrupt_buffer_weight: float = 0.0
    # Corrupt-buffer construction. "reverse" is the legacy aggressive arm;
    # "rotate" preserves the same token multiset/order locality. Text control
    # modes can suffix "_preserve_ws" (for example "rotate_preserve_ws") to keep
    # assistant-prefix boundary whitespace while corrupting the memory pieces.
    opd_contrastive_corrupt_buffer_mode: str = "reverse"
    # Which forced-prefix positions the corrupt hidden contrast touches.
    # "full_prefix" is legacy behavior (pause + answer cue). "memory_only"
    # corrupts/weights only the pause/memory tokens and leaves the answer cue
    # suffix intact.
    opd_contrastive_corrupt_buffer_span: str = "full_prefix"
    # Optional answer-level causal contrast for the same corrupt-buffer pair.
    # When >0, real-buffer answer positions get +weight teacher KL and corrupt-
    # buffer answer positions get -weight teacher KL. This requires answer rows
    # to be present, so use opd_supervise_buffer_only=false. Start small; the
    # negative arm is an anti-target and should usually be paired with
    # opd_loss_max_clamp.
    opd_contrastive_corrupt_answer_weight: float = 0.0
    # Optional positive-only answer KL weight. Unlike
    # opd_contrastive_corrupt_answer_weight, this does not create a corrupt
    # answer anti-target. Use it when the negative/control arm is hidden-only
    # (for example cache mismatch) but answer rows still need direct supervision.
    opd_positive_answer_weight: float = 0.0
    # Positive-only prefill-time-compute OPSD weights. These are the direct
    # "teacher has CoT, student has filler" objective: prompt positions can be
    # zeroed, filler positions get KL/hidden pressure, and answer positions get
    # an optional positive KL target without any corrupt/cache negative arm.
    opd_ptc_positive_buffer_kl_weight: float = 0.0
    opd_ptc_positive_answer_kl_weight: float = 0.0
    opd_ptc_positive_hidden_weight: float = 0.0
    # If true, positions whose KL and hidden weights are both zero are marked
    # IGNORE_INDEX in the student loss payload. Use this for exact PTC masking
    # so prompt rows do not dilute the filler+answer loss normalization.
    opd_mask_zero_weight_positions: bool = False
    # "sampled" keeps the on-policy sampled answer tail. "gold" replaces the
    # answer tail after the forced prefix with the task's deterministic answer
    # before teacher cache construction and student teacher-forced loss.
    opd_teacher_answer_source: str = "sampled"
    # Optional diagnostics for future in-distribution memory-mismatch controls.
    # Loads the teacher hidden cache on the client and compares each prompt's
    # supervised memory rows against another prompt's rows. Does not alter
    # training data.
    opd_teacher_memory_pair_diagnostics: bool = False
    # Optional memory-only cache-mismatch negative. For each sample, append a
    # same-visible-input copy whose teacher cache indices are unchanged except
    # that supervised memory rows come from a different prompt. This applies
    # negative hidden-match weight only on those memory rows and zero KL weight
    # everywhere, so it never anti-trains the answer distribution.
    opd_cache_mismatch_memory_weight: float = 0.0
    # If true, add the cache-mismatch negative weight back onto the positive
    # real-memory hidden-match rows. This keeps the memory hidden-match mass
    # balanced while preserving the cache-mismatch contrast.
    opd_cache_mismatch_balance_positive_hidden: bool = False

    # Pipelined two-phase teacher prefill (see runbooks/pipelined_teacher_prefill.md).
    # When True, the per-step teacher prefill is split into two POSTs against a
    # RADIX-ON teacher (drop --disable-radix-cache):
    #   Phase A = prompt + CoT + pause (no answer); keeps prompt+pause rows. FIXED
    #     per prompt (independent of the on-policy answer), so it is PREFETCHED one
    #     step ahead during the previous step's forward_backward — zero staleness.
    #   Phase B = prompt + CoT + pause + answer; keeps the answer rows only. The
    #     prompt+CoT+pause prefix is served from radix (Phase A warmed it), so only
    #     the ~ans answer tokens are freshly computed. Runs at step N after sampling.
    # The two caches are merged per sample (Phase A's prompt+pause rows, then Phase
    # B's answer rows) into the single per-step cache forward_backward reads — the
    # merged supervision is IDENTICAL to the single-phase {prompt, pause, answer}
    # cache. Only meaningful with teacher_backend=sglang + teacher_cot_mode=insert +
    # supervise_student_cot=true (the production OPD recipe). Default False keeps the
    # current single-phase path so it is A/B-testable and a running job is unaffected.
    teacher_pipeline_phase: bool = False

    # Async-overlapped sampling ("pipeline RL"), modeled on filler_tokens_rl.py's
    # `pipeline_rl` path. Orthogonal to teacher_pipeline_phase (which is DEPRECATED
    # — do not combine; this uses the plain single-phase teacher path only).
    #
    # When False (default): the current serial path is unchanged. Each step does
    #   prepare (student sampling + single-phase teacher prefill)  ->  forward_backward
    #   ->  optim_step  ->  weight sync, strictly in series. step ~= prepare + train.
    #
    # When True: the PREPARE for step N+1 (student sampling + teacher prefill ->
    #   merged teacher cache + _opd_loss_data) is launched as a background asyncio
    #   task that OVERLAPS step N's forward_backward + optim_step + weight sync. Step
    #   N consumes the prepare produced during step N-1. The samples are therefore
    #   1-step-stale (generated against the inference weights synced at the end of
    #   step N-2 / consumed during step N-1's train window) — exactly the accepted
    #   staleness of filler_tokens_rl's pipeline_rl. step ~= max(prepare, train).
    #
    # Step 0 runs prepare synchronously (no prior step to overlap with), mirroring
    # the reference, which only starts its generation worker after the first step.
    # The control eval (eval_accuracy_every) still runs AFTER sync, on the current
    # (not stale) inference weights — identical to the serial path.
    opd_pipeline_rl: bool = False
    # Pipeline lookahead depth for opd_pipeline_rl. 1 = prepare 1 step ahead (the
    # samples are 1-step-stale). >1 prepares that many steps ahead so the teacher
    # never starves between steps (it always has queued work) — fills the per-step
    # teacher idle gap (AMDAHL-002: teachers idle 25-37% waiting for the next step's
    # sampling). Cost: samples up to `depth` steps stale + `depth` steps of prepared
    # batches/caches held. Incompatible with sampler_quiesce_before_sync (which
    # requires no next-step requests in flight during sync).
    opd_pipeline_depth: int = 1

    # Group sampling + group teacher (modeled on filler_tokens_rl.py's group_size).
    # When 1 (default): the current single-sample-per-prompt behavior, byte-for-byte
    #   (each prompt → one on-policy completion → one OPD datum). G=1 takes the exact
    #   same code paths as before — no new branches execute.
    # When G>1: each prompt is sampled G times (G `num_samples=1` calls, exactly like
    #   filler_tokens_rl's `_submit_problem` G-fanout), yielding G on-policy completions
    #   that SHARE the prompt+CoT+pause prefix and differ only in the answer tail.
    #   OPD has NO reward/advantage (it is distillation, not RL): the group simply
    #   produces N·G on-policy samples to distill against the teacher. The OPD loss
    #   then reduces over ALL N·G kept tokens — token-mean, identical to VERL's
    #   `agg_loss(loss_agg_mode="token-mean")` (masked_sum / total_valid_tokens),
    #   which is what the trainer's TokenPartial reducer already does (the extra G-1
    #   samples are simply more rows → more valid tokens in numerator + denominator).
    #
    # GROUP TEACHER (the OPD-specific efficiency): because the G samples of a prompt
    #   share prompt+CoT+pause, the teacher prefills that prefix ONCE per prompt
    #   (Phase A → the pause hiddens for the supervised-pause KL) and extends with each
    #   of the G answers (Phase B → each answer's hiddens; the shared prefix is served
    #   from the teacher's radix cache, so only the answer tokens are freshly computed).
    #   This is the existing two-phase Phase-A/Phase-B mechanism, reused with one Phase
    #   A shared across G Phase-B extensions. Requires teacher_backend=sglang +
    #   teacher_cot_mode=insert + supervise_student_cot=true + a RADIX-ON teacher
    #   (drop --disable-radix-cache) + the prefix-affinity SMG. The merged per-sample
    #   cache is provably the single-sample cache for that exact (prompt, answer_g):
    #   A's prompt+pause rows then B_g's answer rows, in the student's kept-position
    #   order — see _merge_group_phase_caches.
    # Composes with opd_pipeline_rl (group sampling overlaps fwd_bwd) and the
    # supervise / control-eval / checkpoint machinery.
    group_size: int = 1

    # Checkpoint cadence: every `save_every` completed steps (post-optim,
    # post-sync), the OPD client calls TrainingClient.save_state to persist a
    # named checkpoint (`{save_name_prefix}-step{step}`). Set to 0 to disable.
    # The save call blocks the next step until the checkpoint completes; on
    # large MoE models this can add ~30s every 50 steps. Recommended for runs
    # you want resumable / evaluable mid-training.
    save_every: int = 0
    save_name_prefix: str = "opd"

    # When true, ALSO fire an HF-layout safetensors export at each save, via
    # TrainingClient.save_full_weights_safetensors. The client fires it
    # fire-and-forget (does NOT block the step on .result(); the trainer writes
    # it on its background loop) and the run drains any pending exports at the
    # end. The export is directly loadable by SGLang
    # (`--model-path …/safetensors/{name}`) so a finished checkpoint can be
    # served WITHOUT the trainer-load + P2P-sync dance. NOTE: the trainer may
    # serialize the full-weight gather against forward_backward, so the step
    # immediately after a save can still be slower on large MoE models — the
    # win is "directly servable", not "free".
    save_hf_safetensors: bool = False

    # Wandb logging. Disabled by default; set wandb_enabled=true to opt in.
    # WANDB_API_KEY must be set, or set wandb_mode="offline" to log locally only.
    wandb_enabled: bool = False
    wandb_project: str = "xorl-prefill-time-compute"
    wandb_entity: str = "together-research"
    wandb_run_name: str = ""             # auto-generated from output_dir if empty
    wandb_mode: str = "online"           # online | offline | disabled
    wandb_group: str = ""                # optional run grouping


def _maybe_init_wandb(config: Config):
    """Initialize wandb if `wandb_enabled` is set. Returns the run handle or None.

    Imports wandb lazily so the dependency is only required when actually used.
    Logs `OPD step N profile` rows via `wandb.log()` keyed by step number.
    """
    if not config.wandb_enabled:
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        logger.warning("wandb_enabled=true but `import wandb` failed; install wandb or set wandb_enabled=false")
        return None
    if not os.environ.get("WANDB_API_KEY") and config.wandb_mode == "online":
        logger.warning("WANDB_API_KEY not set; falling back to offline mode")
        os.environ.setdefault("WANDB_MODE", "offline")
    elif config.wandb_mode != "online":
        os.environ.setdefault("WANDB_MODE", config.wandb_mode)
    run_name = config.wandb_run_name or f"opd-{Path(config.output_dir).name}"
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=run_name,
        group=config.wandb_group or None,
        config={
            "model_name": config.model_name,
            "teacher_head": config.teacher_head,
            "num_steps": config.num_steps,
            "num_prompts": config.num_prompts,
            "opd_microbatch_size": config.opd_microbatch_size,
            "opd_prepare_batch_size": config.opd_prepare_batch_size,
            "opd_prepare_concurrency": config.opd_prepare_concurrency,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "learning_rate": config.learning_rate,
            "sync_method": config.sync_method,
            "sync_weights": config.sync_weights,
            "opd_loss_max_clamp": config.opd_loss_max_clamp,
            "opd_pipeline_rl": config.opd_pipeline_rl,
            "teacher_filler_text": config.teacher_filler_text,
            "teacher_filler_count": config.teacher_filler_count,
            "teacher_system_prefix": (config.teacher_system_prefix or "")[:200],
            "student_stop_sequences": config.student_stop_sequences,
            "eval_answer_logprob_control": config.eval_answer_logprob_control,
            "eval_answer_logprob_batch_size": config.eval_answer_logprob_batch_size,
            "eval_answer_logprob_max_concurrency": config.eval_answer_logprob_max_concurrency,
            "eval_answer_logprob_distractor_control": config.eval_answer_logprob_distractor_control,
            "eval_answer_logprob_distractor_offset": config.eval_answer_logprob_distractor_offset,
            "eval_control_max_concurrency": config.eval_control_max_concurrency,
            "eval_control_start_step": config.eval_control_start_step,
            "eval_control_progress_log_every_s": config.eval_control_progress_log_every_s,
            "eval_corrupt_pause_mode": config.eval_corrupt_pause_mode,
            "inference_logprob_base_urls": config.inference_logprob_base_urls,
            "sampler_quiesce_before_sync": config.sampler_quiesce_before_sync,
            "sampler_quiesce_timeout": config.sampler_quiesce_timeout,
            "sampler_quiesce_poll_s": config.sampler_quiesce_poll_s,
            "sampler_quiesce_max_new_outstanding": config.sampler_quiesce_max_new_outstanding,
            "sampler_quiesce_max_connections_active": config.sampler_quiesce_max_connections_active,
            "sampler_quiesce_max_inflight": config.sampler_quiesce_max_inflight,
            "opd_supervise_buffer_only": config.opd_supervise_buffer_only,
            "opd_contrastive_corrupt_buffer_weight": config.opd_contrastive_corrupt_buffer_weight,
            "opd_contrastive_corrupt_buffer_mode": config.opd_contrastive_corrupt_buffer_mode,
            "opd_contrastive_corrupt_buffer_span": config.opd_contrastive_corrupt_buffer_span,
            "opd_contrastive_corrupt_answer_weight": config.opd_contrastive_corrupt_answer_weight,
            "opd_positive_answer_weight": config.opd_positive_answer_weight,
            "opd_ptc_positive_buffer_kl_weight": config.opd_ptc_positive_buffer_kl_weight,
            "opd_ptc_positive_answer_kl_weight": config.opd_ptc_positive_answer_kl_weight,
            "opd_ptc_positive_hidden_weight": config.opd_ptc_positive_hidden_weight,
            "opd_mask_zero_weight_positions": config.opd_mask_zero_weight_positions,
            "opd_teacher_answer_source": config.opd_teacher_answer_source,
            "opd_teacher_memory_pair_diagnostics": config.opd_teacher_memory_pair_diagnostics,
            "opd_cache_mismatch_memory_weight": config.opd_cache_mismatch_memory_weight,
            "opd_cache_mismatch_balance_positive_hidden": config.opd_cache_mismatch_balance_positive_hidden,
        },
        reinit=True,
    )
    # Async-eval results land AFTER later training steps have logged, so they
    # can't use the monotonic `step=` axis (wandb drops out-of-order steps).
    # They log under eval_async/* keyed by their own step metric instead; the
    # jsonl profile row (row_kind="eval_async") remains the authoritative record.
    if run is not None:
        try:
            run.define_metric("eval_async/for_step")
            run.define_metric("eval_async/*", step_metric="eval_async/for_step")
        except Exception:  # noqa: BLE001 — older wandb without define_metric
            logger.warning("wandb define_metric unavailable; eval_async/* will use the default axis")
    logger.info("wandb run: %s (project=%s entity=%s)", run.url if run else "n/a", config.wandb_project, config.wandb_entity)
    return run


def _default_prompts(num_prompts: int, prompt_len: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for idx in range(num_prompts):
        base = 1000 + idx * 4096
        prompts.append([base + offset for offset in range(prompt_len)])
    return prompts


def _default_chat_prompts(num_prompts: int) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "role": "user",
                "content": f"Write a concise answer to synthetic OPD prompt {idx}.",
            }
        ]
        for idx in range(num_prompts)
    ]


def _uses_chat_completions(config: Config) -> bool:
    return config.inference_api_format.strip().lower().replace("-", "_") in {
        "chat",
        "openai",
        "openai_chat",
        "chat_completion",
        "chat_completions",
    }


def _sampled_output_token_ids(sampled: Any, chat_tokenizer: Any | None) -> list[int]:
    tokens = getattr(sampled, "tokens", None) or []
    if tokens:
        return [int(token) for token in tokens]
    text = getattr(sampled, "text", None) or ""
    if text and chat_tokenizer is not None:
        return [int(token) for token in chat_tokenizer.encode(text, add_special_tokens=False)]
    return []


def _sampled_output_logprobs(sampled: Any, expected_tokens: list[int], *, context: str) -> list[float]:
    logprobs = getattr(sampled, "logprobs", None)
    if logprobs is None:
        raise RuntimeError(f"{context} did not return logprobs")
    if len(logprobs) != len(expected_tokens):
        raise RuntimeError(
            f"{context} returned {len(logprobs)} logprobs for {len(expected_tokens)} output tokens"
        )
    return [float(value) for value in logprobs]


def _old_logprobs_for_generated_spans(
    sequence_len: int,
    spans: list[tuple[int, list[float]]],
) -> list[float]:
    """Align output-token logprobs to causal OPD target positions.

    A generated token at sequence position ``pos`` is predicted by input position
    ``pos - 1`` in the shifted ``target_tokens`` array.
    """
    target_len = max(0, int(sequence_len) - 1)
    old_logprobs = [0.0] * target_len
    for token_start, span_logprobs in spans:
        target_start = int(token_start) - 1
        for offset, value in enumerate(span_logprobs):
            target_pos = target_start + offset
            if 0 <= target_pos < target_len:
                old_logprobs[target_pos] = float(value)
    return old_logprobs


def _load_prompts(config: Config) -> list[Any]:
    if config.prompts_json_path:
        # Path-based loading avoids ARG_MAX explosion for large prompt sets
        # (passing a 100k+-prompt JSON directly via `prompts_json=...` blows
        # past Linux's ~128 KB MAX_ARG_STRLEN).
        prompts = json.loads(Path(config.prompts_json_path).read_text())
    elif config.prompts_json:
        prompts = json.loads(config.prompts_json)
    elif _uses_chat_completions(config):
        prompts = _default_chat_prompts(config.num_prompts)
    else:
        prompts = _default_prompts(config.num_prompts, config.prompt_len)
    if not isinstance(prompts, list):
        raise ValueError("prompts_json must encode a list of prompts")

    normalized: list[Any] = []
    for prompt in prompts:
        if isinstance(prompt, str):
            normalized.append(prompt)
        elif isinstance(prompt, list) and all(
            isinstance(token, int) for token in prompt
        ):
            normalized.append([int(token) for token in prompt])
        elif (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(message, dict) for message in prompt)
        ):
            normalized.append(prompt)
        else:
            raise ValueError(
                "prompts_json entries must be token-id lists, strings, or chat-message lists"
            )
    return normalized[: config.num_prompts]


def _sample_prompt(prompt: Any) -> Any:
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return tomi.ModelInput.from_ints(prompt)
    return prompt


def _tokenize_teacher_prefix(
    config: Config, chat_tokenizer: Any | None
) -> list[int]:
    """Tokenize the teacher system-prompt prefix.

    Returns an empty list when no prefix is configured. Uses a paired
    [system, dummy-user] render and strips the user portion so the resulting
    tokens are exactly the system header that should be prepended to the
    student's input_ids (which already start with their own user turn).
    """
    text = config.teacher_system_prefix
    if not text and config.teacher_system_prefix_path:
        text = Path(config.teacher_system_prefix_path).read_text(encoding="utf-8")
    if not text:
        return []
    if chat_tokenizer is None:
        raise ValueError(
            "teacher_system_prefix requires a chat_tokenizer (set chat_tokenizer_path)"
        )
    # Many chat templates (Qwen, Llama) error if there's no user turn. Render
    # [system, dummy-user] then split on the dummy-user marker to recover the
    # pure system header.
    dummy_marker = "__XORL_OPD_DUMMY_USER__"
    rendered_full = chat_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": text},
            {"role": "user", "content": dummy_marker},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if dummy_marker not in rendered_full:
        raise RuntimeError(
            f"Tokenizer did not echo the dummy-user marker; rendered={rendered_full!r}"
        )
    # Everything before the dummy user turn is the system header. We also need
    # to drop the dummy "<|im_start|>user\n" preamble that the template emits
    # right before the marker. Easiest: locate the marker and walk backward to
    # the previous "<|im_start|>" (which opens the dummy user turn).
    idx = rendered_full.index(dummy_marker)
    # Find the start of the dummy user's opening token. Common open markers:
    # "<|im_start|>user", "<|start_header_id|>user", "<s>[INST]", etc.
    # Strategy: scan backwards from idx for the LAST occurrence of "<|" or "<s>"
    # marker open.
    cut = idx
    for marker in ("<|im_start|>user", "<|start_header_id|>user", "[INST]"):
        pos = rendered_full.rfind(marker, 0, idx)
        if pos != -1 and pos > cut - len(marker) - 8:
            cut = pos
            break
        if pos != -1:
            cut = min(cut, pos)
    system_prefix_text = rendered_full[:cut]
    encoded = chat_tokenizer.encode(system_prefix_text, add_special_tokens=False)
    return [int(token) for token in encoded]


def _load_chat_tokenizer(config: Config) -> Any | None:
    if not _uses_chat_completions(config):
        return None
    tokenizer_path = config.chat_tokenizer_path or config.model_name
    if not tokenizer_path or tokenizer_path == "default":
        raise ValueError(
            "chat_tokenizer_path must be set when inference_api_format=chat_completions"
        )
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _as_flat_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
    elif hasattr(value, "input_ids"):
        value = value.input_ids
        if hasattr(value, "tolist"):
            value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if isinstance(value, (list, tuple)) and all(
        isinstance(token, numbers.Integral) for token in value
    ):
        return [int(token) for token in value]
    raise ValueError("Tokenizer did not return a flat list of chat prompt token IDs")


def _encode_chat_prompt(prompt: Any, tokenizer: Any) -> list[int]:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif (
        isinstance(prompt, list)
        and prompt
        and all(isinstance(message, dict) for message in prompt)
    ):
        messages = prompt
    else:
        raise ValueError("Chat OPD prompt must be a string or chat-message list")

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return _as_flat_token_ids(token_ids)


def _prompt_token_ids(prompt: Any, tokenizer: Any | None) -> list[int]:
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return [int(token) for token in prompt]
    if tokenizer is None:
        raise ValueError("chat_tokenizer_path is required to render chat prompts as token IDs")
    return _encode_chat_prompt(prompt, tokenizer)


def _sampled_sequence_tokens(
    prompt: Any, sampled: tomi.SampledSequence, chat_tokenizer: Any | None = None
) -> list[int]:
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return list(prompt) + list(sampled.tokens)
    if sampled.prompt_tokens:
        return list(sampled.prompt_tokens) + list(sampled.tokens)
    if sampled.prompt_tokens is not None:
        # EMPTY input_token_ids: the backend accepted the request but did not
        # return its rendered prompt (logprob_start_len missing/ignored). Falling
        # back to a LOCAL re-render here is how the 2026-06-10 train/sample
        # context mismatch slipped in (server rendered an open <think> block;
        # client trained on the closed form) — fail loud instead.
        raise RuntimeError(
            "Chat-completions backend returned EMPTY input_token_ids; the trained "
            "context would silently mismatch the sampled context. Ensure the "
            "request carries logprob_start_len=0 (SamplingParams.chat_logprob_start_len)."
        )
    if chat_tokenizer is not None:
        return _encode_chat_prompt(prompt, chat_tokenizer) + list(sampled.tokens)
    raise RuntimeError(
        "Chat-completions OPD sampling requires backend input_token_ids in each choice "
        "or a configured chat_tokenizer_path"
    )


def _chat_sampling_extras(config: "Config") -> dict[str, Any]:
    """SamplingParams kwargs that pin server-side rendering + ids return.

    chat_logprob_start_len=0 forces the backend to return input_token_ids (the
    server-truth rendered prompt) so OPD sequences are built on what the model
    actually sampled in; chat_template_kwargs pins the template rendering.
    """
    extras: dict[str, Any] = {"chat_logprob_start_len": 0}
    raw = (config.chat_template_kwargs or "").strip()
    if raw:
        extras["chat_template_kwargs"] = json.loads(raw)
    return extras


async def _sample_student_batch(
    sampling_clients: list[tomi.SamplingClient],
    prompts: list[Any],
    max_new_tokens: int,
    temperature: float,
    return_logprobs: bool = False,
    chat_tokenizer: Any | None = None,
    student_prefill_text: str = "",
    student_prefill_count: int = 0,
    student_prefill_token_ids: str = "",
    student_prefill_suffix: str = "",
    student_stop_sequences: str = "",
    student_generated_memory_tokens: int = 0,
    group_size: int = 1,
    request_timeout: float = 900.0,
    per_sample_cot_tokens: list[list[int]] | None = None,
    buffer_equals_cot: bool = False,
    sampling_extras: dict[str, Any] | None = None,
) -> tuple[list[list[int]], list[int], int, list[list[int]], int, list[list[float]] | None, list[int]]:
    """Sample student trajectories.

    With ``student_prefill_count > 0`` and a chat prompt, the student request
    appends an extra assistant message with ``student_prefill_text`` repeated
    ``student_prefill_count`` times and asks the backend to ``continue`` from
    that open turn. The sampled sequence is then ``prompt + assistant_open +
    prefill_tokens + answer``; ``prompt_token_lens[i]`` is the boundary BEFORE
    the prefill (the index where the teacher-CoT replacement should occur).

    ``group_size`` (G) controls group sampling, mirroring
    ``filler_tokens_rl.py``'s ``_submit_problem`` G-fanout: each prompt is sampled
    ``G`` times via ``G`` independent ``num_samples=1`` calls (distinct random
    ``sampling_seed`` per call so the completions differ). The G samples of a
    prompt SHARE the prompt + prefill (pause/CoT region) and differ only in the
    generated answer tail. Returned lists are flattened in PROMPT-MAJOR order:
    ``[p0_g0, p0_g1, ..., p0_gG-1, p1_g0, ...]`` (length ``N*G``), and the final
    return element is ``samples_per_prompt = G`` so callers can re-group by prompt
    (e.g. the group teacher, which prefills each prompt's shared prefix once).
    With ``G == 1`` (default) the call is byte-for-byte the original single-sample
    path: one future per prompt, the original unseeded ``params``, identical
    flattening.

    ``buffer_equals_cot`` + ``per_sample_cot_tokens`` enable the "buffer = CoT
    length (K=C)" mode (text-prefill only): each PROMPT's forced prefill is sized to
    that prompt's CoT length (``pause_count_i = max(1, C_i - len(suffix))``), so
    every sample's filler region has length ``K_i ~= C_i``. The returned
    ``filler_counts`` (the final tuple element, prompt-major, length == len(
    sequences)) carries the actual per-sample tokenized prefill length; in the
    global / flag-off path it is just the scalar ``filler_token_count`` repeated.
    """
    # Turn on the forced-prefix mechanism whenever there's an answer-cue suffix,
    # even with ZERO pause tokens (the no-pause control). At count==0 the prefix
    # is just `student_prefill_suffix` (the "</think>Answer:" cue) — required so
    # the student still emits an answer (no cue => base model loops on filler/EOS).
    # k_filler then = len(cue tokens) and the teacher masks prompt+CoT, keeping
    # cue+answer. The original pause path (count>0) is unchanged.
    exact_prefill_ids = _parse_token_id_list(student_prefill_token_ids)
    stop_sequences = _parse_stop_sequences(student_stop_sequences)
    use_exact_prefill_ids = bool(exact_prefill_ids)
    generated_memory_tokens = max(0, int(student_generated_memory_tokens or 0))
    use_prefill = bool(student_prefill_suffix) or use_exact_prefill_ids or (
        bool(student_prefill_count) and bool(student_prefill_text)
    )
    G = max(1, int(group_size))

    if generated_memory_tokens > 0:
        if chat_tokenizer is None:
            raise ValueError("student_generated_memory_tokens requires chat_tokenizer_path")
        if use_exact_prefill_ids:
            raise ValueError("student_generated_memory_tokens does not support student_prefill_token_ids")
        seed_block = student_prefill_text * student_prefill_count
        seed_tokens = [
            int(t) for t in chat_tokenizer.encode(seed_block, add_special_tokens=False)
        ] if seed_block else []
        suffix_tokens = [
            int(t) for t in chat_tokenizer.encode(student_prefill_suffix, add_special_tokens=False)
        ] if student_prefill_suffix else []
        filler_token_count = len(seed_tokens) + generated_memory_tokens + len(suffix_tokens)

        memory_params = tomi.SamplingParams(
            max_tokens=generated_memory_tokens,
            temperature=temperature,
            stop=None,
            chat_continue_final_message=bool(seed_block),
            custom_params={"ignore_eos": True, "min_tokens": generated_memory_tokens},
        )
        memory_futures = []
        memory_meta: list[tuple[int, int, Any]] = []
        for idx, prompt in enumerate(prompts):
            if not (
                isinstance(prompt, list)
                and prompt
                and all(isinstance(message, dict) for message in prompt)
            ):
                raise ValueError("student_generated_memory_tokens requires chat-message prompts")
            client = sampling_clients[idx % len(sampling_clients)]
            memory_prompt = (
                list(prompt) + [{"role": "assistant", "content": seed_block}]
                if seed_block
                else list(prompt)
            )
            for g in range(G):
                call_params = (
                    memory_params
                    if G == 1
                    else tomi.SamplingParams(
                        max_tokens=generated_memory_tokens,
                        temperature=temperature,
                        stop=None,
                        chat_continue_final_message=bool(seed_block),
                        custom_params={"ignore_eos": True, "min_tokens": generated_memory_tokens},
                        sampling_seed=int.from_bytes(os.urandom(8), "big") >> 1,
                    )
                )
                memory_futures.append(
                    client.sample(
                        prompt=memory_prompt,
                        sampling_params=call_params,
                        num_samples=1,
                        return_logprobs=True,
                    )
                )
                memory_meta.append((idx, g, prompt))

        memory_responses = await asyncio.wait_for(
            asyncio.gather(*memory_futures),
            timeout=max(float(request_timeout), 1.0),
        )
        memory_token_rows: list[list[int]] = []
        memory_logprob_rows: list[list[float]] = []
        memory_text_rows: list[str] = []
        for response in memory_responses:
            if not response.sequences:
                raise RuntimeError("Generated-memory sampler returned no sequences")
            sampled = response.sequences[0]
            tokens = _sampled_output_token_ids(sampled, chat_tokenizer)
            if len(tokens) != generated_memory_tokens:
                raise RuntimeError(
                    "Generated-memory sampler did not produce the requested fixed length: "
                    f"got {len(tokens)} tokens, expected {generated_memory_tokens}"
                )
            memory_token_rows.append(tokens)
            memory_logprob_rows.append(
                _sampled_output_logprobs(sampled, tokens, context="Generated-memory sampler")
            )
            memory_text_rows.append(
                chat_tokenizer.decode(tokens, skip_special_tokens=False)
                if hasattr(chat_tokenizer, "decode")
                else (getattr(sampled, "text", None) or "")
            )

        answer_params = tomi.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences or None,
            chat_continue_final_message=True,
        )
        answer_futures = []
        for row_idx, (prompt_idx, _g, prompt) in enumerate(memory_meta):
            client = sampling_clients[prompt_idx % len(sampling_clients)]
            answer_prompt = list(prompt) + [
                {
                    "role": "assistant",
                    "content": seed_block + memory_text_rows[row_idx] + student_prefill_suffix,
                }
            ]
            answer_futures.append(
                client.sample(
                    prompt=answer_prompt,
                    sampling_params=answer_params,
                    num_samples=1,
                    return_logprobs=return_logprobs,
                )
            )

        answer_responses = await asyncio.wait_for(
            asyncio.gather(*answer_futures),
            timeout=max(float(request_timeout), 1.0),
        )
        sequences: list[list[int]] = []
        prompt_token_lens: list[int] = []
        completions: list[list[int]] = []
        old_logprobs_by_sample: list[list[float]] | None = [] if return_logprobs else None
        for row_idx, response in enumerate(answer_responses):
            prompt_idx, _g, prompt = memory_meta[row_idx]
            if not response.sequences:
                raise RuntimeError("Student answer sampler returned no sequences")
            sampled = response.sequences[0]
            p_tokens = _encode_chat_prompt(prompt, chat_tokenizer)
            answer_tokens = _sampled_output_token_ids(sampled, chat_tokenizer)
            answer_logprobs = (
                _sampled_output_logprobs(sampled, answer_tokens, context="Student answer sampler")
                if return_logprobs
                else []
            )
            completions.append(answer_tokens)
            sequence = (
                p_tokens
                + list(seed_tokens)
                + list(memory_token_rows[row_idx])
                + list(suffix_tokens)
                + answer_tokens
            )
            sequences.append(sequence)
            prompt_token_lens.append(len(p_tokens))
            if old_logprobs_by_sample is not None:
                memory_start = len(p_tokens) + len(seed_tokens)
                answer_start = memory_start + len(memory_token_rows[row_idx]) + len(suffix_tokens)
                old_logprobs_by_sample.append(
                    _old_logprobs_for_generated_spans(
                        len(sequence),
                        [
                            (memory_start, memory_logprob_rows[row_idx]),
                            (answer_start, answer_logprobs),
                        ],
                    )
                )
        # Generated-memory path does not support buffer_equals_cot (no per-sample
        # CoT plumbing here); return a flat per-sample filler-count list = scalar.
        return (
            sequences,
            prompt_token_lens,
            filler_token_count,
            completions,
            G,
            old_logprobs_by_sample,
            [filler_token_count] * len(sequences),
        )

    params = tomi.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop_sequences or None,
        chat_continue_final_message=use_prefill and not use_exact_prefill_ids,
        **(sampling_extras or {}),
    )
    # The forced student prefix is the pause buffer PLUS an answer lead-in
    # suffix (e.g. "\n</think>\n\nAnswer: "). The suffix is what makes the model
    # actually emit answer tokens: with an open `<think>` full of pause and no
    # answer cue, even the BASE model just keeps emitting pause (0/20 on
    # 4dmult), so on-policy rollouts produced no answer tokens and the student
    # collapsed to EOS. With the lead-in, base emits the answer (~11/20), so the
    # student has real answer positions to distill. The ENTIRE prefix
    # (pause + suffix) is the masked "filler" region K — see _opd_loss_data.
    prefill_block = ""
    if use_prefill and chat_tokenizer is None:
        raise ValueError(
            "student_prefill_count > 0 requires a chat_tokenizer (set chat_tokenizer_path)"
        )

    prefill_tokens: list[int] = []
    if use_prefill:
        suffix_tokens = (
            [int(t) for t in chat_tokenizer.encode(student_prefill_suffix, add_special_tokens=False)]
            if student_prefill_suffix
            else []
        )
        if use_exact_prefill_ids:
            prefill_tokens = list(exact_prefill_ids) + suffix_tokens
        else:
            prefill_block = student_prefill_text * student_prefill_count + student_prefill_suffix
            prefill_tokens = [
                int(t)
                for t in chat_tokenizer.encode(prefill_block, add_special_tokens=False)
            ]
        # Build the prefill ONCE, then splice these IDs into the reconstructed
        # student trajectory. Text prefill is invisible in chat-completions
        # output; exact-token prefill is part of the /generate prompt. Either
        # way, this local reconstruction is the trainer's source of truth.
        # The masked filler region K = the full prefill length (pause + suffix).
        # Sanity-check the pause part is still one-token-per-repeat (the remap
        # masks the whole K region, so only the TOTAL length must match what
        # SGLang tokenizes for the assistant content — same assumption as before
        # for the bare-pause case).
        pause_tokens = (
            exact_prefill_ids
            if use_exact_prefill_ids
            else chat_tokenizer.encode(
                student_prefill_text * student_prefill_count, add_special_tokens=False
            )
        )
        if not use_exact_prefill_ids and len(pause_tokens) != student_prefill_count:
            # Multi-token filler (e.g. nato): k_filler downstream uses the FULL
            # tokenized prefix length (len(prefill_tokens)), and the teacher-cache
            # remap masks the K region by total length (mask_len = C [+ K_student]),
            # not per-repeat — so a multi-token filler is correctly aligned. Warn,
            # don't raise. (Single-token pause still satisfies ==, no warning.)
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger("xorl-client-opd").warning(
                f"multi-token filler: {student_prefill_text!r} -> {len(pause_tokens)} tokens "
                f"for count={student_prefill_count}; using full length as K_student "
                f"(remap masks K by length, so alignment is preserved)."
            )

    # "buffer = CoT length (K=C)" mode: size each PROMPT's pause buffer to its own
    # teacher-CoT length. Only the text-prefill path (not exact-ids) supports this —
    # per the spec, exact-ids keeps the global path. When active, build one prefill
    # block/tokenization per prompt: K_i = len(per_sample_cot_tokens[i]); the pause
    # count is K_i minus the suffix length (clamped >= 1), so the TOTAL prefill
    # length per prompt ~= K_i (the masked filler region). The downstream
    # filler_counts list carries the actual tokenized prefill length per sample.
    use_per_sample_prefill = (
        bool(buffer_equals_cot)
        and per_sample_cot_tokens is not None
        and use_prefill
        and not use_exact_prefill_ids
    )
    per_sample_prefill_tokens: list[list[int]] = []
    per_sample_prefill_blocks: list[str] = []
    if use_per_sample_prefill:
        if len(per_sample_cot_tokens) != len(prompts):
            raise ValueError(
                f"per_sample_cot_tokens has {len(per_sample_cot_tokens)} entries for "
                f"{len(prompts)} prompts"
            )
        for cot_tokens in per_sample_cot_tokens:
            k_i = len(cot_tokens)
            pause_count_i = max(1, k_i - len(suffix_tokens))
            block_i = student_prefill_text * pause_count_i + student_prefill_suffix
            tokens_i = [
                int(t)
                for t in chat_tokenizer.encode(block_i, add_special_tokens=False)
            ]
            per_sample_prefill_blocks.append(block_i)
            per_sample_prefill_tokens.append(tokens_i)

    def _sample_messages_with_prefill(prompt: Any, prompt_idx: int) -> Any:
        if not use_prefill:
            return _sample_prompt(prompt)
        if use_exact_prefill_ids:
            p_tokens = _prompt_token_ids(prompt, chat_tokenizer)
            return tomi.ModelInput.from_ints(p_tokens + list(prefill_tokens))
        if not (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(message, dict) for message in prompt)
        ):
            raise ValueError(
                "student_prefill_count > 0 requires chat-message prompts (list of {role, content} dicts)"
            )
        content = (
            per_sample_prefill_blocks[prompt_idx]
            if use_per_sample_prefill
            else prefill_block
        )
        return list(prompt) + [{"role": "assistant", "content": content}]

    # Build one future per (prompt, sample-in-group). For G==1 this reduces to the
    # original single-future-per-prompt loop with the shared unseeded `params`
    # (byte-for-byte). For G>1 each call gets a distinct `sampling_seed` so the G
    # completions of a prompt actually differ — exactly filler_tokens_rl's
    # `_submit_problem`, which fans out `num_samples=1` G times with a random seed
    # per call. The (prompt_idx, group_idx) flattening is prompt-major so the
    # caller can re-group: sample index `i` belongs to prompt `i // G`.
    futures = []
    for idx, prompt in enumerate(prompts):
        client = sampling_clients[idx % len(sampling_clients)]
        sample_prompt = _sample_messages_with_prefill(prompt, idx)
        for _g in range(G):
            call_params = (
                params
                if G == 1
                else tomi.SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=stop_sequences or None,
                    chat_continue_final_message=use_prefill and not use_exact_prefill_ids,
                    **(sampling_extras or {}),
                    sampling_seed=int.from_bytes(os.urandom(8), "big") >> 1,
                )
            )
            futures.append(
                client.sample(
                    prompt=sample_prompt,
                    sampling_params=call_params,
                    num_samples=1,
                    return_logprobs=return_logprobs,
                )
            )

    responses = await asyncio.wait_for(
        asyncio.gather(*futures),
        timeout=max(float(request_timeout), 1.0),
    )
    sequences: list[list[int]] = []
    prompt_token_lens: list[int] = []
    # The raw generated tokens per sample (post-prefill). Threaded out so the
    # training loop can measure generation health (empty/EOS rate, length) and
    # score answers — the signal that would have caught the EOS-collapse.
    completions: list[list[int]] = []
    old_logprobs_by_sample: list[list[float]] | None = [] if return_logprobs else None
    # Per-sample filler-region length K_i (prompt-major, length == len(sequences)).
    # In the global path this is just the scalar prefill length repeated; in the
    # buffer_equals_cot path it is the per-prompt prefill length (= ~K_i = C_i).
    filler_counts: list[int] = []

    def _sample_prefill_tokens(sample_idx: int) -> list[int]:
        # Resolve the spliced prefill IDs for sample `sample_idx` (prompt sample_idx//G).
        if use_per_sample_prefill:
            return per_sample_prefill_tokens[sample_idx // G]
        return prefill_tokens

    # responses are prompt-major: prompt i's G samples are responses[i*G:(i+1)*G].
    for idx, response in enumerate(responses):
        prompt = prompts[idx // G]
        if not response.sequences:
            raise RuntimeError("Student sampler returned no sequences")
        sampled = response.sequences[0]
        sampled_tokens = [int(t) for t in (sampled.tokens or [])]
        sampled_logprobs = (
            _sampled_output_logprobs(sampled, sampled_tokens, context="Student sampler")
            if return_logprobs
            else []
        )
        completions.append(sampled_tokens)
        sample_prefill = _sample_prefill_tokens(idx)
        if use_exact_prefill_ids:
            p_tokens = _prompt_token_ids(prompt, chat_tokenizer)
            sequence = p_tokens + list(sample_prefill) + sampled_tokens
            sequences.append(sequence)
            prompt_token_lens.append(len(p_tokens))
        elif use_prefill:
            # Reconstruct: bare_prompt + prefill + generated. SGLang's response
            # gives us only the post-prefill `sampled.tokens`, so we splice the
            # prefill back in from our tokenized copy (per-sample under K=C).
            p_tokens = _encode_chat_prompt(prompt, chat_tokenizer)
            sequence = p_tokens + list(sample_prefill) + sampled_tokens
            sequences.append(sequence)
            # Boundary p is BEFORE the prefill (where teacher's CoT will be
            # substituted).
            prompt_token_lens.append(len(p_tokens))
        else:
            sequence = _sampled_sequence_tokens(prompt, sampled, chat_tokenizer)
            sequences.append(sequence)
            if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
                prompt_token_lens.append(len(prompt))
            elif sampled.prompt_tokens:
                prompt_token_lens.append(len(sampled.prompt_tokens))
            elif chat_tokenizer is not None:
                prompt_token_lens.append(len(_encode_chat_prompt(prompt, chat_tokenizer)))
            else:
                prompt_token_lens.append(0)
        filler_counts.append(len(sample_prefill) if use_prefill else 0)
        if old_logprobs_by_sample is not None:
            generated_start = int(prompt_token_lens[-1]) + len(sample_prefill)
            old_logprobs_by_sample.append(
                _old_logprobs_for_generated_spans(
                    len(sequence),
                    [(generated_start, sampled_logprobs)],
                )
            )
    # K_eff = full forced-prefix length (pause + suffix); the masked filler
    # region used by the teacher-substitution and the cache-index remap. Scalar
    # (global) for backward-compat; per-sample K_i is in `filler_counts`.
    filler_token_count = len(prefill_tokens) if use_prefill else 0
    return (
        sequences,
        prompt_token_lens,
        filler_token_count,
        completions,
        G,
        old_logprobs_by_sample,
        filler_counts,
    )


@dataclass
class PhaseAResult:
    """Prefetched Phase A (prompt+CoT+pause) cache for one prepare batch.

    Built WITHOUT sampling — purely from the prompt tokens, the per-prompt CoT,
    and the fixed forced-prefix (pause+suffix) tokens — so it can be issued one
    step ahead during the previous step's forward_backward with zero staleness.
    """

    cache_path: Path
    a_indices: list[list[int]]
    prompt_token_lens: list[int]
    forced_prefix_len: int
    metrics: dict[str, Any] = field(default_factory=dict)


# Serializes HF-tokenizer use across the concurrent Phase A prefetch threads.
_PHASE_A_TOKENIZER_LOCK = threading.Lock()


def _forced_prefix_tokens(config: Config, chat_tokenizer: Any) -> list[int]:
    """The fixed student forced-prefix (pause*count + suffix) token ids = K."""
    suffix_tokens = (
        [int(t) for t in chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False)]
        if config.student_prefill_suffix
        else []
    )
    exact_ids = _parse_token_id_list(config.student_prefill_token_ids)
    if exact_ids:
        return exact_ids + suffix_tokens
    block = config.student_prefill_text * config.student_prefill_count
    return [int(t) for t in chat_tokenizer.encode(block, add_special_tokens=False)] + suffix_tokens


def _student_generated_memory_prepare_metrics(config: Config, chat_tokenizer: Any | None) -> dict[str, float]:
    generated_tokens = max(0, int(config.student_generated_memory_tokens or 0))
    if generated_tokens <= 0:
        return {
            "student_generated_memory_active": 0.0,
            "student_generated_memory_tokens": 0.0,
            "student_generated_memory_seed_tokens": 0.0,
            "student_generated_memory_suffix_tokens": 0.0,
            "student_generated_memory_total_filler_tokens": 0.0,
        }
    seed_tokens = _student_memory_prefix_token_count(config, chat_tokenizer)
    suffix_tokens = (
        len(chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False))
        if chat_tokenizer is not None and config.student_prefill_suffix
        else 0
    )
    return {
        "student_generated_memory_active": 1.0,
        "student_generated_memory_tokens": float(generated_tokens),
        "student_generated_memory_seed_tokens": float(seed_tokens),
        "student_generated_memory_suffix_tokens": float(suffix_tokens),
        "student_generated_memory_total_filler_tokens": float(seed_tokens + generated_tokens + suffix_tokens),
    }


def _phase_a_prepare(
    config: Config,
    teacher_url: str,
    prompts: list[Any],
    cots: list[list[int]],
    forced_prefix: list[int],
    output_dir: Path,
    step: int,
    microbatch_idx: int,
    chat_tokenizer: Any,
    teacher_prefix_tokens: list[int] | None,
) -> PhaseAResult:
    """Build + POST Phase A (prompt+CoT+pause, keep prompt+pause) for one batch.

    Synchronous (blocking HTTP); call via ``asyncio.to_thread``. Returns the cache
    path + per-sample kept-row indices for the later merge.
    """
    t0 = time.perf_counter()
    K = len(forced_prefix)
    # Reconstruct the fixed prompt+CoT+pause sequences (no answer). The pause block
    # is the forced prefix appended right after the bare prompt — identical to what
    # _sample_student_batch splices in, but with no sampled answer.
    # NOTE: Phase A batches run as concurrent asyncio.to_thread calls, so guard the
    # HF tokenizer (apply_chat_template/encode is not thread-safe) with a lock.
    sequences: list[list[int]] = []
    prompt_token_lens: list[int] = []
    with _PHASE_A_TOKENIZER_LOCK:
        for prompt in prompts:
            p_tokens = _encode_chat_prompt(prompt, chat_tokenizer)
            sequences.append(list(p_tokens) + list(forced_prefix))
            prompt_token_lens.append(len(p_tokens))
    data = _teacher_phase_data(
        sequences,
        "a",
        teacher_prefix_tokens,
        cots,
        prompt_token_lens,
        student_filler_count=K,
    )
    cache_path = output_dir / f"teacher_hidden_a_step{step}_mb{microbatch_idx}.safetensors"
    cache = _post_teacher_phase_cache(teacher_url, data, cache_path, config.request_timeout)
    return PhaseAResult(
        cache_path=cache_path,
        a_indices=cache.get("cache_indices_by_sample") or [],
        prompt_token_lens=prompt_token_lens,
        forced_prefix_len=K,
        metrics={"phase_a_s": _elapsed(t0)},
    )


async def _prepare_opd_batch_pipelined(
    config: Config,
    sampling_clients: list[tomi.SamplingClient],
    teacher_url: str,
    prompts: list[Any],
    output_dir: Path,
    step: int,
    chat_tokenizer: Any,
    microbatch_idx: int,
    teacher_prefix_tokens: list[int] | None,
    cots: list[list[int]],
    phase_a: PhaseAResult,
) -> PreparedOpdBatch:
    """Two-phase prepare: sample -> Phase B (radix-served) -> merge(A, B)."""
    prepare_t0 = time.perf_counter()
    sample_t0 = time.perf_counter()
    # The DEPRECATED two-phase pipelined path stays single-sample (group_size is
    # not plumbed here; group sampling uses the single-phase _prepare_opd_batch via
    # opd_pipeline_rl). Unpack the trailing group_size (always 1). This path does
    # not use buffer_equals_cot; the per-sample filler_counts is ignored.
    sequences, prompt_token_lens, filler_token_count, completions, _G, old_logprobs_by_sample, _filler_counts = await _sample_student_batch(
        sampling_clients,
        prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        return_logprobs=_uses_chat_completions(config),
        chat_tokenizer=chat_tokenizer,
        student_prefill_text=config.student_prefill_text,
        student_prefill_count=config.student_prefill_count,
        student_prefill_token_ids=config.student_prefill_token_ids,
        student_prefill_suffix=config.student_prefill_suffix,
        student_stop_sequences=config.student_stop_sequences,
        student_generated_memory_tokens=config.student_generated_memory_tokens,
        request_timeout=config.request_timeout,
    )
    sample_s = _elapsed(sample_t0)
    k_filler = filler_token_count or config.student_prefill_count
    gold_answer_replacements = 0
    gold_answer_skipped = 0
    answer_source = (config.opd_teacher_answer_source or "sampled").strip().lower()
    if answer_source in {"gold", "gold_answer", "target", "target_answer"}:
        if config.opd_use_policy_gradient:
            raise ValueError("opd_use_policy_gradient requires sampled/on-policy answers, not gold replacement")
        sequences, completions, gold_answer_replacements, gold_answer_skipped = (
            _replace_sampled_answers_with_gold(
                sequences,
                completions,
                prompts,
                prompt_token_lens,
                k_filler,
                1,
                chat_tokenizer,
                config.eval_task,
            )
        )
    elif answer_source not in {"sampled", "student", "student_sample", "on_policy"}:
        raise ValueError(
            "opd_teacher_answer_source must be 'sampled' or 'gold', "
            f"got {config.opd_teacher_answer_source!r}"
        )
    if config.opd_use_policy_gradient and old_logprobs_by_sample is None:
        raise ValueError("opd_use_policy_gradient requires sampled old_logprobs")

    # Phase A was built from the deterministic prompt+pause; its prompt lengths must
    # match this step's sampled sequences (same prompts, same tokenizer).
    if prompt_token_lens != phase_a.prompt_token_lens:
        raise RuntimeError(
            f"step {step} mb {microbatch_idx}: Phase A prompt_token_lens "
            f"{phase_a.prompt_token_lens[:4]}... != sampled {prompt_token_lens[:4]}... "
            "(prefetched Phase A window must match this step's prompt window)"
        )
    if k_filler != phase_a.forced_prefix_len:
        raise RuntimeError(
            f"step {step} mb {microbatch_idx}: forced-prefix length {k_filler} != "
            f"Phase A's {phase_a.forced_prefix_len}"
        )

    teacher_t0 = time.perf_counter()
    b_path = output_dir / f"teacher_hidden_b_step{step}_mb{microbatch_idx}.safetensors"
    b_data = _teacher_phase_data(
        sequences, "b", teacher_prefix_tokens, cots, prompt_token_lens,
        student_filler_count=k_filler,
    )
    b_cache = await asyncio.to_thread(
        _post_teacher_phase_cache, teacher_url, b_data, b_path, config.request_timeout
    )
    merged_path = output_dir / f"teacher_hidden_step{step}_mb{microbatch_idx}.safetensors"
    merged_indices = await asyncio.to_thread(
        _merge_phase_caches,
        phase_a.cache_path,
        b_path,
        phase_a.a_indices,
        b_cache.get("cache_indices_by_sample") or [],
        merged_path,
    )
    teacher_s = _elapsed(teacher_t0)
    # Phase A + Phase B intermediate caches are consumed; drop them now.
    for _p in (phase_a.cache_path, b_path):
        try:
            Path(_p).unlink(missing_ok=True)
        except Exception as _exc:  # noqa: BLE001
            logger.debug("phase cache cleanup skipped for %s: %s", _p, _exc)

    teacher_tokens = sum(len(sequence) - 1 for sequence in sequences)
    sample_output_tokens = sum(
        max(0, len(sequence) - prompt_len - k_filler)
        for sequence, prompt_len in zip(sequences, prompt_token_lens)
    )
    data = _opd_loss_data(
        sequences,
        merged_indices,
        prompt_token_lens=prompt_token_lens,
        student_filler_count=k_filler,
        supervise_student_cot=config.supervise_student_cot,
        mask_prompt_positions=config.opd_mask_prompt_kl,
        correct_prefix_only=config.opd_correct_prefix_only,
        old_logprobs_by_sample=old_logprobs_by_sample if config.opd_use_policy_gradient else None,
        sample_ok_by_sample=_sample_answer_correctness(
            completions, prompts, 1, chat_tokenizer, config.eval_task
        ),
    )
    metrics = {
        "student_sampling_s": sample_s,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s
        if sample_s > 0
        else 0.0,
        "teacher_prefill_s": teacher_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_s if teacher_s > 0 else 0.0,
        "teacher_prefill_forward_compute_s": 0.0,
        "teacher_hidden_cache_write_s": 0.0,
        "teacher_phase_a_prefetched_s": float(phase_a.metrics.get("phase_a_s", 0.0)),
        "opd_teacher_answer_source_gold": float(gold_answer_replacements > 0),
        "opd_gold_answer_replacements": float(gold_answer_replacements),
        "opd_gold_answer_skipped": float(gold_answer_skipped),
        **_student_generated_memory_prepare_metrics(config, chat_tokenizer),
        "prepare_s": _elapsed(prepare_t0),
    }
    return PreparedOpdBatch(
        sequences=sequences,
        data=data,
        cache_path=merged_path,
        metrics=metrics,
        completions=completions,
        prompt_texts=[_user_prompt_text(p) for p in prompts],
    )


def _strict_prepare_overlap_supported(config: Config, group_size: int) -> bool:
    """Whether chunked strict prepare is semantics-preserving for this recipe."""
    return (
        config.teacher_backend == "sglang"
        and group_size == 1
        and not config.sft_mode
        and not config.opd_buffer_equals_cot
        and int(config.student_generated_memory_tokens or 0) <= 0
        and float(config.opd_contrastive_corrupt_buffer_weight or 0.0) == 0.0
        and float(config.opd_contrastive_corrupt_answer_weight or 0.0) == 0.0
        and float(config.opd_positive_answer_weight or 0.0) == 0.0
        and float(config.opd_ptc_positive_buffer_kl_weight or 0.0) == 0.0
        and float(config.opd_ptc_positive_answer_kl_weight or 0.0) == 0.0
        and float(config.opd_ptc_positive_hidden_weight or 0.0) == 0.0
        and float(config.opd_cache_mismatch_memory_weight or 0.0) == 0.0
        and not bool(config.opd_teacher_memory_pair_diagnostics)
    )


async def _prepare_opd_batch_strict_chunked(
    config: Config,
    sampling_clients: list[tomi.SamplingClient],
    teacher_url: str,
    prompts: list[Any],
    output_dir: Path,
    step: int,
    chat_tokenizer: Any | None,
    microbatch_idx: int,
    teacher_prefix_tokens: list[int] | None,
    teacher_filler_tokens: list[int] | list[list[int]] | None,
    chunk_count: int,
) -> PreparedOpdBatch:
    """Strict same-step prepare overlap via ordered prompt chunks.

    Each chunk runs the normal strict single-phase prepare. Chunks are launched
    concurrently, so a chunk's teacher prefill can start as soon as that chunk's
    same-step samples complete while slower chunks are still sampling. The chunk
    caches are concatenated and cache references are re-based, yielding a single
    PreparedOpdBatch for one downstream trainer fb call.
    """
    prepare_t0 = time.perf_counter()
    if chunk_count <= 1 or len(prompts) <= 1:
        raise ValueError("strict chunked prepare requires chunk_count > 1 and more than one prompt")
    chunk_count = min(int(chunk_count), len(prompts))
    chunk_size = int(math.ceil(len(prompts) / chunk_count))
    prompt_chunks = [
        prompts[start : start + chunk_size]
        for start in range(0, len(prompts), chunk_size)
    ]
    first_filler = teacher_filler_tokens[0] if teacher_filler_tokens else None
    per_prompt_filler = isinstance(first_filler, list)

    def _chunk_filler(start: int, end: int) -> list[int] | list[list[int]] | None:
        if per_prompt_filler:
            return teacher_filler_tokens[start:end]  # type: ignore[index,return-value]
        return teacher_filler_tokens

    base_config = chz.replace(config, opd_strict_prepare_overlap_chunks=0)
    tasks: list[asyncio.Task[PreparedOpdBatch]] = []
    offset = 0
    for chunk_idx, chunk_prompts in enumerate(prompt_chunks):
        start = offset
        end = start + len(chunk_prompts)
        offset = end
        tasks.append(
            asyncio.create_task(
                _prepare_opd_batch(
                    base_config,
                    sampling_clients,
                    teacher_url,
                    chunk_prompts,
                    output_dir,
                    step,
                    chat_tokenizer,
                    (microbatch_idx + 1) * 1000 + chunk_idx,
                    teacher_prefix_tokens,
                    _chunk_filler(start, end),
                )
            )
        )
    chunks = await asyncio.gather(*tasks)
    merged_path = output_dir / f"teacher_hidden_step{step}_mb{microbatch_idx}.safetensors"
    merged = _merge_prepared_chunk_batches(chunks, merged_path)
    sample_t0s = [
        float(chunk.metrics["student_sampling_t0"])
        for chunk in chunks
        if "student_sampling_t0" in chunk.metrics
    ]
    sample_t1s = [
        float(chunk.metrics["student_sampling_t1"])
        for chunk in chunks
        if "student_sampling_t1" in chunk.metrics
    ]
    teacher_t0s = [
        float(chunk.metrics["teacher_prefill_t0"])
        for chunk in chunks
        if "teacher_prefill_t0" in chunk.metrics
    ]
    teacher_t1s = [
        float(chunk.metrics["teacher_prefill_t1"])
        for chunk in chunks
        if "teacher_prefill_t1" in chunk.metrics
    ]
    merged.metrics.update(
        {
            "prepare_s": _elapsed(prepare_t0),
            "strict_prepare_overlap_wall_s": _elapsed(prepare_t0),
        }
    )
    if sample_t0s and sample_t1s:
        merged.metrics["student_sampling_t0"] = min(sample_t0s)
        merged.metrics["student_sampling_t1"] = max(sample_t1s)
    if teacher_t0s and teacher_t1s:
        merged.metrics["teacher_prefill_t0"] = min(teacher_t0s)
        merged.metrics["teacher_prefill_t1"] = max(teacher_t1s)
    return merged


async def _prepare_opd_batch(
    config: Config,
    sampling_clients: list[tomi.SamplingClient],
    teacher_url: str,
    prompts: list[Any],
    output_dir: Path,
    step: int,
    chat_tokenizer: Any | None = None,
    microbatch_idx: int = 0,
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | list[list[int]] | None = None,
) -> PreparedOpdBatch:
    prepare_t0 = time.perf_counter()
    sample_t0 = time.perf_counter()
    G = max(1, int(config.group_size))
    # Per-PROMPT teacher CoT (list[list[int]], one per prompt). Derived purely from
    # teacher_filler_tokens + the prompt list, so it is available BEFORE sampling —
    # the "buffer = CoT length (K=C)" mode needs it to size each prompt's pause
    # buffer. None when there is no per-prompt CoT (global-filler / no-filler).
    per_prompt_cot: list[list[int]] | None = None
    if teacher_filler_tokens and isinstance(teacher_filler_tokens[0], list):
        per_prompt_cot = [list(c) for c in teacher_filler_tokens]
        if len(per_prompt_cot) != len(prompts):
            raise RuntimeError(
                f"teacher_filler_tokens has {len(per_prompt_cot)} entries for "
                f"{len(prompts)} prompts"
            )
    buffer_equals_cot = bool(config.opd_buffer_equals_cot)
    strict_overlap_chunks = int(getattr(config, "opd_strict_prepare_overlap_chunks", 0) or 0)
    if strict_overlap_chunks > 1 and len(prompts) > 1:
        if not _strict_prepare_overlap_supported(config, G):
            raise ValueError(
                "opd_strict_prepare_overlap_chunks is currently supported only for "
                "single-phase sglang teacher strict prepares with group_size=1, "
                "sft_mode=false, no buffer_equals_cot/generated-memory/custom-pair/cache diagnostics"
            )
        return await _prepare_opd_batch_strict_chunked(
            config,
            sampling_clients,
            teacher_url,
            prompts,
            output_dir,
            step,
            chat_tokenizer,
            microbatch_idx,
            teacher_prefix_tokens,
            teacher_filler_tokens,
            strict_overlap_chunks,
        )
    sequences, prompt_token_lens, filler_token_count, completions, _G, old_logprobs_by_sample, filler_counts = await _sample_student_batch(
        sampling_clients,
        prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        return_logprobs=_uses_chat_completions(config),
        chat_tokenizer=chat_tokenizer,
        student_prefill_text=config.student_prefill_text,
        student_prefill_count=config.student_prefill_count,
        student_prefill_token_ids=config.student_prefill_token_ids,
        student_prefill_suffix=config.student_prefill_suffix,
        student_stop_sequences=config.student_stop_sequences,
        student_generated_memory_tokens=config.student_generated_memory_tokens,
        group_size=G,
        request_timeout=config.request_timeout,
        per_sample_cot_tokens=per_prompt_cot,
        buffer_equals_cot=buffer_equals_cot,
    )
    # sequences/prompt_token_lens/completions are now prompt-major and length N*G:
    # sample i belongs to prompt i // G.
    sample_s = _elapsed(sample_t0)
    # PER-SAMPLE filler region length K_i (prompt-major, length == len(sequences)).
    # In the global/flag-off path every entry == filler_token_count, so downstream
    # behaviour is unchanged; under buffer_equals_cot each K_i ~= C_i.
    per_sample_filler_count: list[int] = list(filler_counts)
    # Effective filler length = full forced prefix (pause + suffix). Used as K
    # for both the teacher-CoT substitution and the student cache-index remap.
    # Falls back to student_prefill_count when no prefill is used.
    k_filler = filler_token_count or config.student_prefill_count
    gold_answer_replacements = 0
    gold_answer_skipped = 0
    answer_source = (config.opd_teacher_answer_source or "sampled").strip().lower()
    if answer_source in {"gold", "gold_answer", "target", "target_answer"}:
        if config.opd_use_policy_gradient:
            raise ValueError("opd_use_policy_gradient requires sampled/on-policy answers, not gold replacement")
        if chat_tokenizer is None:
            raise ValueError("opd_teacher_answer_source=gold requires chat_tokenizer_path")
        if buffer_equals_cot:
            # Gold replacement splices the answer at prompt+scalar-k_filler, which is
            # wrong under per-sample K_i. Out of scope for the K=C mode.
            raise ValueError(
                "opd_buffer_equals_cot is incompatible with opd_teacher_answer_source=gold "
                "(per-sample K_i prevents a single-boundary gold splice)"
            )
        sequences, completions, gold_answer_replacements, gold_answer_skipped = (
            _replace_sampled_answers_with_gold(
                sequences,
                completions,
                prompts,
                prompt_token_lens,
                k_filler,
                G,
                chat_tokenizer,
                config.eval_task,
            )
        )
    elif answer_source not in {"sampled", "student", "student_sample", "on_policy"}:
        raise ValueError(
            "opd_teacher_answer_source must be 'sampled' or 'gold', "
            f"got {config.opd_teacher_answer_source!r}"
        )
    if config.opd_use_policy_gradient and old_logprobs_by_sample is None:
        raise ValueError("opd_use_policy_gradient requires sampled old_logprobs")

    # Per-sample answer correctness of the trained-on completion (post any gold
    # replacement): 1=correct, 0=wrong, -1=unscorable. Broadcast per-position by
    # _opd_loss_data so the trainer can split answer-position KL by whether the
    # on-policy prefix being supervised is a correct or wrong answer.
    sample_ok_by_sample: list[int] = _sample_answer_correctness(
        completions, prompts, G, chat_tokenizer, config.eval_task
    )

    # Expansion-lock pivot (opd_only_train_on_pause_gap): restrict positive
    # supervision to the filler-DEPENDENT frontier — problems the model gets
    # RIGHT with the pause buffer but WRONG without it. Sample one greedy
    # no-pause completion per prompt (prefill = answer cue only, the eval
    # "nopause" arm), score it, and zero sample_ok for any sample whose prompt
    # the model already solves no-pause, so opd_correct_prefix_only masks it.
    if config.opd_only_train_on_pause_gap:
        if not config.opd_correct_prefix_only:
            raise ValueError(
                "opd_only_train_on_pause_gap requires opd_correct_prefix_only=true "
                "(the gap filter is applied through the same sample_ok masking)"
            )
        # Mirror the eval "nopause" arm EXACTLY (it works — acc_nopause ~0.5): sample
        # each prompt with an assistant prefill of ONLY the answer cue (no pause buffer),
        # greedy, and score the raw completion. NB: _sample_student_batch returns EMPTY
        # completions at student_prefill_count=0 (its prefix-stripping is pause-path-
        # specific), so we sample directly via client.sample like the eval.
        _np_params = tomi.SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=0.0,
            chat_continue_final_message=bool(config.student_prefill_suffix),
            **_chat_sampling_extras(config),
        )

        async def _nopause_ok(client: Any, p: Any) -> int:
            sp = list(p) + (
                [{"role": "assistant", "content": config.student_prefill_suffix}]
                if config.student_prefill_suffix else []
            )
            try:
                resp = await asyncio.wait_for(
                    client.sample(prompt=sp, sampling_params=_np_params, num_samples=1),
                    timeout=max(float(config.request_timeout), 1.0),
                )
            except Exception:  # noqa: BLE001 - sampling failure => unscorable
                return -1
            seqs = getattr(resp, "sequences", None) or []
            if not seqs:
                return -1
            seq = seqs[0]
            toks = list(getattr(seq, "tokens", None) or [])
            text = getattr(seq, "text", None) or (
                chat_tokenizer.decode(toks, skip_special_tokens=True) if toks else ""
            )
            v = _score_answer(_user_prompt_text(p), text, config.eval_task)
            return -1 if v is None else int(v)

        nopause_ok_by_prompt = list(
            await asyncio.gather(
                *[
                    _nopause_ok(sampling_clients[i % len(sampling_clients)], p)
                    for i, p in enumerate(prompts)
                ]
            )
        )
        n_prompts = len(prompts)
        nopause_correct = sum(
            1 for p in range(n_prompts) if int(nopause_ok_by_prompt[p]) == 1
        )
        kept = 0
        for sample_idx in range(len(sample_ok_by_sample)):
            prompt_idx = sample_idx // G
            if prompt_idx < n_prompts and int(nopause_ok_by_prompt[prompt_idx]) == 1:
                # Already solved without the buffer → not filler-dependent: drop it.
                sample_ok_by_sample[sample_idx] = 0
            elif int(sample_ok_by_sample[sample_idx]) == 1:
                kept += 1
        logger.info(
            "expansion-lock: nopause_solved=%d/%d prompts; kept %d/%d samples "
            "(pause-correct AND nopause-wrong) for supervision",
            nopause_correct,
            n_prompts,
            kept,
            len(sample_ok_by_sample),
        )

    cache_path = (
        output_dir / f"teacher_hidden_step{step}_mb{microbatch_idx}.safetensors"
    )
    # Multi-layer OPRD: resolve the decoder-layer subset once. None => off. Only the
    # single-phase sglang teacher path supports OPRD; the group two-phase path does
    # not (it would need per-phase layer merging), so OPRD requires G==1 / non-group.
    oprd_layer_indices = _resolve_oprd_layer_indices(
        config.opd_oprd_layers, config.opd_oprd_num_layers
    )
    use_sglang_oprd_cache = bool(oprd_layer_indices) and config.opd_oprd_cache_backend == "sglang"
    if config.opd_oprd_cache_backend not in {"trainer", "sglang"}:
        raise ValueError(
            f"opd_oprd_cache_backend must be 'trainer' or 'sglang', got {config.opd_oprd_cache_backend!r}"
        )
    # Default: trainer-side teacher forward writes no rank-3 layers cache. The
    # explicit "sglang" A/B writes the layer cache next to the rank-2 KL cache.
    oprd_layers_cache_path = (
        output_dir / f"teacher_hidden_layers_step{step}_mb{microbatch_idx}.safetensors"
        if use_sglang_oprd_cache
        else None
    )

    # Decide between the GROUP TEACHER (shared-prefix two-phase) and the plain
    # single-phase teacher. The group teacher is only valid (and only a win) for
    # the production supervise recipe — per-prompt CoT, teacher_cot_mode=insert,
    # supervise_student_cot, sglang backend, radix ON. It prefills each prompt's
    # shared prompt+CoT+pause prefix ONCE (Phase A) and extends it with each of the
    # G answers (Phase B, radix-served). For G==1 (or any non-supervise recipe) we
    # take the EXACT original single-phase path on all N*G sequences — each sample
    # gets its own independent, full teacher prefill — which is byte-for-byte the
    # pre-group behavior when G==1.
    # Per-SAMPLE CoT: expand the pre-computed per-PROMPT CoT G times (prompt-major)
    # so the teacher payload aligns 1:1 with the N*G sequences.
    per_sample_cot: list[list[int]] | None = None
    if per_prompt_cot is not None:
        per_sample_cot = [per_prompt_cot[i // G] for i in range(len(sequences))]
    # "buffer = CoT length (K=C)": align student-buffer-position-i <-> teacher-CoT-
    # position-i 1:1 over the FULL CoT. That requires match_cot + supervise_student_cot
    # on the teacher-cache + loss-data calls regardless of the configured mode.
    effective_teacher_cot_mode = config.teacher_cot_mode
    effective_supervise_student_cot = config.supervise_student_cot
    if buffer_equals_cot:
        if per_sample_cot is None:
            raise ValueError(
                "opd_buffer_equals_cot requires per-prompt teacher CoT "
                "(set teacher_cot_json_path so teacher_filler_tokens is per-prompt)"
            )
        if config.opd_supervise_buffer_only:
            # match_cot keeps the full CoT+answer (no answer masking); buffer-only
            # is the insert-mode recipe and is out of scope for K=C.
            raise ValueError(
                "opd_buffer_equals_cot (match_cot, full CoT) is incompatible with "
                "opd_supervise_buffer_only=true"
            )
        effective_teacher_cot_mode = "match_cot"
        effective_supervise_student_cot = True
    use_group_teacher = (
        G > 1
        and config.teacher_backend == "sglang"
        and config.teacher_cot_mode == "insert"
        and config.supervise_student_cot
        and per_sample_cot is not None
        and k_filler > 0
        and not buffer_equals_cot  # group two-phase uses a scalar K; K=C needs per-sample
        and not oprd_layer_indices  # OPRD is single-phase only (per-layer merge unsupported)
    )
    if oprd_layer_indices and config.teacher_backend != "sglang":
        raise ValueError("opd_oprd_layers requires teacher_backend='sglang'")

    teacher_t0 = time.perf_counter()
    teacher_compute_s = 0.0
    teacher_write_s = 0.0
    # Multi-layer OPRD trainer-side teacher forward inputs (None unless OPRD is on; it
    # is single-phase sglang-only, so only that branch populates these).
    teacher_input_ids_by_sample: list[list[int]] | None = None
    teacher_kept_indices_by_sample: list[list[int]] | None = None
    if config.sft_mode:
        # SFT ablation: NO teacher prefill — supervision is plain cross-entropy on
        # the (gold-spliced) answer tokens. Natural cache indices keep the
        # downstream _opd_loss_data plumbing happy; rows are re-mapped to CE
        # datums after it runs (search "sft_mode" below).
        if oprd_layer_indices or buffer_equals_cot or config.opd_use_policy_gradient:
            raise ValueError("sft_mode supports only the plain no-filler path (no OPRD/K=C/PG)")
        cache_indices_by_sample = [list(range(len(s) - 1)) for s in sequences]
    elif use_group_teacher:
        # ── Group teacher: Phase A once per prompt, Phase B once per sample. ──
        a_path = output_dir / f"teacher_hidden_a_step{step}_mb{microbatch_idx}.safetensors"
        b_path = output_dir / f"teacher_hidden_b_step{step}_mb{microbatch_idx}.safetensors"
        # Phase A input: the deterministic prompt+CoT+pause per PROMPT (one row per
        # prompt). Reconstruct from the first sample of each group (its prompt+pause
        # prefix is identical across the group, so any group member works); drop the
        # answer tail (sequence[:p+K]).
        per_prompt_cot = [list(c) for c in teacher_filler_tokens]  # type: ignore[union-attr]
        prefix_sequences: list[list[int]] = []
        prefix_prompt_lens: list[int] = []
        for n in range(len(prompts)):
            rep = n * G  # first sample of prompt n's group
            p = int(prompt_token_lens[rep])
            prefix_sequences.append(list(sequences[rep][: p + k_filler]))
            prefix_prompt_lens.append(p)
        a_data = _teacher_phase_data(
            prefix_sequences,
            "a",
            teacher_prefix_tokens,
            per_prompt_cot,
            prefix_prompt_lens,
            student_filler_count=k_filler,
        )
        a_cache = await asyncio.to_thread(
            _post_teacher_phase_cache, teacher_url, a_data, a_path, config.request_timeout
        )
        # Phase B input: the full prompt+CoT+pause+answer per SAMPLE (N*G rows), keep
        # ONLY answer rows. The shared prefix is radix-served from Phase A.
        b_data = _teacher_phase_data(
            sequences,
            "b",
            teacher_prefix_tokens,
            per_sample_cot,
            prompt_token_lens,
            student_filler_count=k_filler,
        )
        b_cache = await asyncio.to_thread(
            _post_teacher_phase_cache, teacher_url, b_data, b_path, config.request_timeout
        )
        merged_indices = await asyncio.to_thread(
            _merge_group_phase_caches,
            a_path,
            b_path,
            a_cache.get("cache_indices_by_sample") or [],
            b_cache.get("cache_indices_by_sample") or [],
            G,
            cache_path,
        )
        cache_indices_by_sample = merged_indices
        # Phase A/B intermediates are consumed by the merge; drop them now.
        for _p in (a_path, b_path):
            try:
                Path(_p).unlink(missing_ok=True)
            except Exception as _exc:  # noqa: BLE001
                logger.debug("group phase cache cleanup skipped for %s: %s", _p, _exc)
    else:
        # ── Plain single-phase teacher on all N*G sequences (G==1 byte-for-byte). ──
        single_phase_filler: list[int] | list[list[int]] | None
        single_phase_filler = per_sample_cot if per_sample_cot is not None else teacher_filler_tokens
        if config.teacher_backend == "sglang":
            teacher_cache = await asyncio.to_thread(
                _teacher_cache_from_sglang,
                teacher_url,
                sequences,
                cache_path,
                config.request_timeout,
                teacher_prefix_tokens,
                single_phase_filler,
                prompt_token_lens,
                k_filler,
                effective_teacher_cot_mode,
                effective_supervise_student_cot,
                config.opd_supervise_buffer_only,
                per_sample_filler_count if buffer_equals_cot else None,
                oprd_layer_indices,
                str(oprd_layers_cache_path) if oprd_layers_cache_path else None,
                use_sglang_oprd_cache,
            )
        else:
            teacher_cache = await asyncio.to_thread(
                _teacher_cache_from_xorl,
                teacher_url,
                sequences,
                cache_path,
                config.teacher_model_id,
                config.request_timeout,
                teacher_prefix_tokens,
                single_phase_filler,
                prompt_token_lens,
                k_filler,
                effective_teacher_cot_mode,
                effective_supervise_student_cot,
                per_sample_filler_count if buffer_equals_cot else None,
            )
        cache_indices_by_sample = teacher_cache["cache_indices_by_sample"]
        teacher_compute_s = teacher_cache["metrics"].get("teacher_prefill_forward_compute_s", 0.0)
        teacher_write_s = teacher_cache["metrics"].get("teacher_hidden_cache_write_s", 0.0)
        # Multi-layer OPRD: either keep the current trainer-side teacher forward
        # inputs, or use the SGLang rank-3 layer cache when explicitly requested.
        if use_sglang_oprd_cache:
            layer_path = teacher_cache.get("layers_cache_path")
            if not layer_path:
                raise RuntimeError("OPRD SGLang cache backend requested but no layers_cache_path was returned")
            oprd_layers_cache_path = Path(layer_path)
        elif oprd_layer_indices:
            teacher_input_ids_by_sample = teacher_cache.get("teacher_input_ids_by_sample")
            teacher_kept_indices_by_sample = teacher_cache.get("teacher_kept_indices_by_sample")
            if teacher_input_ids_by_sample is None or teacher_kept_indices_by_sample is None:
                raise RuntimeError(
                    "OPRD enabled but the teacher cache did not return teacher_input_ids/"
                    "teacher_kept_indices for the trainer-side forward"
                )
    teacher_s = _elapsed(teacher_t0)
    teacher_tokens = sum(len(sequence) - 1 for sequence in sequences)
    sample_output_tokens = sum(
        max(0, len(sequence) - prompt_len - (per_sample_filler_count[i] if buffer_equals_cot else k_filler))
        for i, (sequence, prompt_len) in enumerate(zip(sequences, prompt_token_lens))
    )

    loss_sequences = sequences
    loss_cache_indices = cache_indices_by_sample
    loss_prompt_token_lens = prompt_token_lens
    # Aligned with loss_sequences; custom-pair datums (negatives/mismatch) get -1.
    loss_sample_ok: list[int] = sample_ok_by_sample
    teacher_weights_by_sample: list[list[float]] | None = None
    hidden_match_weights_by_sample: list[list[float]] | None = None
    contrastive_weight = float(config.opd_contrastive_corrupt_buffer_weight or 0.0)
    answer_contrastive_weight = float(config.opd_contrastive_corrupt_answer_weight or 0.0)
    positive_answer_weight = float(config.opd_positive_answer_weight or 0.0)
    ptc_buffer_kl_weight = float(config.opd_ptc_positive_buffer_kl_weight or 0.0)
    ptc_answer_kl_weight = float(config.opd_ptc_positive_answer_kl_weight or 0.0)
    ptc_hidden_weight = float(config.opd_ptc_positive_hidden_weight or 0.0)
    cache_mismatch_weight = float(config.opd_cache_mismatch_memory_weight or 0.0)
    ptc_positive_active = (
        ptc_buffer_kl_weight > 0.0
        or ptc_answer_kl_weight > 0.0
        or ptc_hidden_weight > 0.0
    )
    balance_cache_mismatch_hidden = bool(
        config.opd_cache_mismatch_balance_positive_hidden and cache_mismatch_weight > 0.0
    )
    positive_hidden_memory_weight = 1.0 + (
        cache_mismatch_weight if balance_cache_mismatch_hidden else 0.0
    )
    contrastive_examples = 0
    answer_contrastive_examples = 0
    positive_answer_examples = 0
    ptc_positive_examples = 0
    cache_mismatch_examples = 0
    cache_mismatch_skipped_examples = 0
    cache_mismatch_span_token_total = 0
    cache_mismatch_changed_rows = 0
    teacher_position_weight_sums = {
        "prompt_weight_sum": 0.0,
        "prompt_weight_count": 0.0,
        "buffer_weight_sum": 0.0,
        "buffer_weight_count": 0.0,
        "answer_weight_sum": 0.0,
        "answer_weight_count": 0.0,
    }
    hidden_position_weight_sums = dict(teacher_position_weight_sums)
    corrupt_mode = (config.opd_contrastive_corrupt_buffer_mode or "reverse").strip().lower()
    corrupt_span = _corrupt_span_token_count(config, chat_tokenizer, k_filler)
    corrupt_changed_tokens = 0
    corrupt_span_token_total = 0
    corrupt_noop_examples = 0
    custom_pair_active = (
        contrastive_weight > 0.0
        or answer_contrastive_weight > 0.0
        or positive_answer_weight > 0.0
        or cache_mismatch_weight > 0.0
        or ptc_positive_active
    )
    corrupt_pair_active = contrastive_weight > 0.0 or answer_contrastive_weight > 0.0
    if oprd_layer_indices and custom_pair_active:
        # The custom-pair path appends extra negative/mismatch datums to
        # loss_sequences that have no matching teacher forward sequence, so the
        # per-datum teacher_input_ids/kept-indices alignment breaks. OPRD trainer
        # forward is supported only on the plain positive (1:1) path.
        raise ValueError(
            "opd_oprd_layers (trainer-side teacher forward) is incompatible with "
            "contrastive/PTC/cache-mismatch custom pairs"
        )
    if custom_pair_active:
        if config.opd_use_policy_gradient and (corrupt_pair_active or cache_mismatch_weight > 0.0):
            raise ValueError(
                "opd_use_policy_gradient only supports sampled positive OPD datums; "
                "corrupt/cache-mismatch datums are not sampled from the behavior policy"
            )
        if not (config.supervise_student_cot and k_filler > 0 and corrupt_span > 0):
            raise ValueError(
                "PTC/contrastive/cache-mismatch memory supervision requires supervise_student_cot=true "
                "and a non-empty student buffer"
            )
        if (
            (
                answer_contrastive_weight > 0.0
                or positive_answer_weight > 0.0
                or ptc_answer_kl_weight > 0.0
            )
            and config.opd_supervise_buffer_only
        ):
            raise ValueError(
                "answer KL weights require "
                "opd_supervise_buffer_only=false so answer rows are present"
            )
        loss_sequences = []
        loss_cache_indices = []
        loss_prompt_token_lens = []
        loss_sample_ok = []
        teacher_weights_by_sample = []
        hidden_match_weights_by_sample = []
        # Parallel to loss_sequences (which appends negative/mismatch datums), so the
        # per-sample K_i list stays aligned for the buffer_equals_cot remap.
        loss_per_sample_filler_count: list[int] = []
        loss_old_logprobs_by_sample: list[list[float]] | None = [] if config.opd_use_policy_gradient else None
        for sample_idx, (sequence, indices, prompt_len) in enumerate(
            zip(sequences, cache_indices_by_sample, prompt_token_lens)
        ):
            # Per-sample filler region length: K_i under buffer_equals_cot, else the
            # global scalar. Used as the buffer/answer boundary so the answer-position
            # weights and bucket splits track each prompt's own K_i.
            k_filler_i = per_sample_filler_count[sample_idx] if buffer_equals_cot else k_filler
            positive_buffer_span = (
                k_filler_i if ptc_positive_active else corrupt_span
            )
            positive_buffer_kl_weight = ptc_buffer_kl_weight if ptc_positive_active else 1.0
            positive_hidden_weight = ptc_hidden_weight if ptc_positive_active else positive_hidden_memory_weight
            positive_answer_kl_weight = (
                answer_contrastive_weight + positive_answer_weight + ptc_answer_kl_weight
            )
            positive_teacher_buffer_weights = _buffer_position_weights(
                sequence,
                prompt_len,
                positive_buffer_span,
                positive_buffer_kl_weight,
            )
            positive_hidden_buffer_weights = _buffer_position_weights(
                sequence,
                prompt_len,
                positive_buffer_span,
                positive_hidden_weight,
            )
            positive_answer_weights = _answer_position_weights(
                sequence,
                prompt_len,
                k_filler_i,
                positive_answer_kl_weight,
            )
            positive_teacher_weights = _add_position_weights(
                positive_teacher_buffer_weights, positive_answer_weights
            )
            loss_sequences.append(sequence)
            loss_cache_indices.append(indices)
            loss_prompt_token_lens.append(prompt_len)
            loss_per_sample_filler_count.append(k_filler_i)
            loss_sample_ok.append(sample_ok_by_sample[sample_idx])
            teacher_weights_by_sample.append(positive_teacher_weights)
            hidden_match_weights_by_sample.append(positive_hidden_buffer_weights)
            if loss_old_logprobs_by_sample is not None:
                if old_logprobs_by_sample is None:
                    raise ValueError("opd_use_policy_gradient requires sampled old_logprobs")
                loss_old_logprobs_by_sample.append(old_logprobs_by_sample[sample_idx])
            if positive_answer_weight > 0.0:
                positive_answer_examples += 1
            if ptc_positive_active:
                ptc_positive_examples += 1
            for key, value in _position_weight_bucket_sums(
                positive_teacher_weights, prompt_len, k_filler_i
            ).items():
                teacher_position_weight_sums[key] += value
            for key, value in _position_weight_bucket_sums(
                positive_hidden_buffer_weights, prompt_len, k_filler_i
            ).items():
                hidden_position_weight_sums[key] += value

            if corrupt_pair_active:
                negative_sequence = _with_corrupted_buffer(
                    sequence,
                    prompt_len,
                    corrupt_span,
                    mode=corrupt_mode,
                )
                positive_span = list(sequence[prompt_len : prompt_len + corrupt_span])
                negative_span = list(negative_sequence[prompt_len : prompt_len + corrupt_span])
                changed_tokens = sum(
                    int(int(left) != int(right))
                    for left, right in zip(positive_span, negative_span)
                )
                corrupt_changed_tokens += changed_tokens
                corrupt_span_token_total += len(positive_span)
                if changed_tokens == 0:
                    corrupt_noop_examples += 1
                zero_weights = [0.0] * (len(negative_sequence) - 1)
                negative_buffer_weights = _buffer_position_weights(
                    negative_sequence,
                    prompt_len,
                    corrupt_span,
                    -contrastive_weight,
                )
                negative_answer_weights = _answer_position_weights(
                    negative_sequence,
                    prompt_len,
                    k_filler_i,
                    -answer_contrastive_weight,
                )
                loss_sequences.append(negative_sequence)
                loss_cache_indices.append(indices)
                loss_prompt_token_lens.append(prompt_len)
                loss_per_sample_filler_count.append(k_filler_i)
                loss_sample_ok.append(-1)
                teacher_weights_by_sample.append(_add_position_weights(zero_weights, negative_answer_weights))
                hidden_match_weights_by_sample.append(negative_buffer_weights)
                contrastive_examples += 1
                if answer_contrastive_weight > 0.0:
                    answer_contrastive_examples += 1

            if cache_mismatch_weight > 0.0:
                mismatch_indices, _donor_idx, changed_rows = _with_mismatched_memory_cache_indices(
                    cache_indices_by_sample,
                    prompt_token_lens,
                    sample_idx,
                    span=corrupt_span,
                    group_size=G,
                )
                if mismatch_indices is None:
                    cache_mismatch_skipped_examples += 1
                    continue
                zero_weights = [0.0] * (len(sequence) - 1)
                mismatch_hidden_weights = _buffer_position_weights(
                    sequence,
                    prompt_len,
                    corrupt_span,
                    -cache_mismatch_weight,
                )
                loss_sequences.append(sequence)
                loss_cache_indices.append(mismatch_indices)
                loss_prompt_token_lens.append(prompt_len)
                loss_per_sample_filler_count.append(k_filler_i)
                loss_sample_ok.append(-1)
                teacher_weights_by_sample.append(zero_weights)
                hidden_match_weights_by_sample.append(mismatch_hidden_weights)
                cache_mismatch_examples += 1
                cache_mismatch_span_token_total += corrupt_span
                cache_mismatch_changed_rows += changed_rows
    else:
        loss_old_logprobs_by_sample = old_logprobs_by_sample if config.opd_use_policy_gradient else None
        # No custom pairs: loss_sequences IS sequences, so the per-sample K_i list
        # (aligned to sequences) is used directly.
        loss_per_sample_filler_count = per_sample_filler_count

    teacher_memory_pair_metrics: dict[str, float] = {}
    if config.opd_teacher_memory_pair_diagnostics:
        try:
            teacher_memory_pair_metrics = _teacher_memory_pair_diagnostics(
                cache_path,
                cache_indices_by_sample,
                prompt_token_lens,
                corrupt_span=corrupt_span,
                group_size=G,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic only; do not abort training
            logger.warning("teacher memory pair diagnostics failed: %s", exc)
            teacher_memory_pair_metrics = {
                "opd_teacher_memory_pair_diag_requested": 1.0,
                "opd_teacher_memory_pair_diag_active": 0.0,
                "opd_teacher_memory_pair_diag_failure": 1.0,
            }

    data = _opd_loss_data(
        loss_sequences,
        loss_cache_indices,
        prompt_token_lens=loss_prompt_token_lens,
        student_filler_count=k_filler,
        supervise_student_cot=effective_supervise_student_cot,
        mask_answer=config.opd_supervise_buffer_only,
        mask_prompt_positions=config.opd_mask_prompt_kl,
        correct_prefix_only=config.opd_correct_prefix_only,
        teacher_weights_by_sample=teacher_weights_by_sample,
        hidden_match_weights_by_sample=hidden_match_weights_by_sample,
        old_logprobs_by_sample=loss_old_logprobs_by_sample,
        mask_zero_weight_positions=config.opd_mask_zero_weight_positions,
        per_sample_filler_count=loss_per_sample_filler_count if buffer_equals_cot else None,
        # Multi-layer OPRD trainer-side teacher forward inputs (None unless OPRD on;
        # OPRD is gated to the plain positive 1:1 path so these align with loss_sequences).
        teacher_input_ids_by_sample=teacher_input_ids_by_sample,
        teacher_kept_indices_by_sample=teacher_kept_indices_by_sample,
        sample_ok_by_sample=loss_sample_ok,
    )
    if config.sft_mode:
        # Re-map each OPD row to a plain cross-entropy datum: supervise EXACTLY the
        # answer-region target positions ([p-1+K, L_s) — same boundary as the OPD
        # answer region) with weights, drop all teacher fields. Combined with
        # opd_teacher_answer_source=gold the targets are the gold answers.
        sft_data = []
        for row, p_len, k_i in zip(data, loss_prompt_token_lens, loss_per_sample_filler_count or [k_filler] * len(data)):
            target_tokens = list(row["loss_fn_inputs"]["target_tokens"])
            answer_start = max(0, int(p_len) - 1 + int(k_i))
            weights = [0] * min(answer_start, len(target_tokens)) + [1] * max(0, len(target_tokens) - answer_start)
            sft_data.append(
                {
                    "model_input": row["model_input"],
                    "loss_fn_inputs": {"target_tokens": target_tokens, "weights": weights},
                }
            )
        data = sft_data

    def _bucket_mean(sums: dict[str, float], bucket: str) -> float:
        count = sums.get(f"{bucket}_weight_count", 0.0)
        return sums.get(f"{bucket}_weight_sum", 0.0) / count if count > 0.0 else 0.0

    metrics = {
        "student_sampling_s": sample_s,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s
        if sample_s > 0
        else 0.0,
        "teacher_prefill_s": teacher_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_s
        if teacher_s > 0
        else 0.0,
        "teacher_prefill_forward_compute_s": teacher_compute_s,
        "teacher_hidden_cache_write_s": teacher_write_s,
        "group_size": float(G),
        "group_teacher_active": float(use_group_teacher),
        "num_samples": float(len(sequences)),
        "num_opd_datums": float(len(data)),
        "opd_contrastive_corrupt_buffer_weight": contrastive_weight,
        "opd_contrastive_corrupt_answer_weight": answer_contrastive_weight,
        "opd_positive_answer_weight": positive_answer_weight,
        "opd_ptc_positive_buffer_kl_weight": ptc_buffer_kl_weight,
        "opd_ptc_positive_answer_kl_weight": ptc_answer_kl_weight,
        "opd_ptc_positive_hidden_weight": ptc_hidden_weight,
        "opd_mask_zero_weight_positions": float(config.opd_mask_zero_weight_positions),
        "opd_teacher_answer_source_gold": float(gold_answer_replacements > 0),
        "opd_gold_answer_replacements": float(gold_answer_replacements),
        "opd_gold_answer_skipped": float(gold_answer_skipped),
        "opd_contrastive_corrupt_examples": float(contrastive_examples),
        "opd_contrastive_corrupt_answer_examples": float(answer_contrastive_examples),
        "opd_positive_answer_examples": float(positive_answer_examples),
        "opd_ptc_positive_examples": float(ptc_positive_examples),
        "opd_teacher_weight_prompt_mean": _bucket_mean(teacher_position_weight_sums, "prompt"),
        "opd_teacher_weight_buffer_mean": _bucket_mean(teacher_position_weight_sums, "buffer"),
        "opd_teacher_weight_answer_mean": _bucket_mean(teacher_position_weight_sums, "answer"),
        "opd_hidden_weight_prompt_mean": _bucket_mean(hidden_position_weight_sums, "prompt"),
        "opd_hidden_weight_buffer_mean": _bucket_mean(hidden_position_weight_sums, "buffer"),
        "opd_hidden_weight_answer_mean": _bucket_mean(hidden_position_weight_sums, "answer"),
        "opd_contrastive_data_multiplier": float(len(data) / max(1, len(sequences))),
        "opd_contrastive_corrupt_span_tokens": float(corrupt_span if contrastive_examples else 0),
        "opd_contrastive_corrupt_span_token_total": float(corrupt_span_token_total),
        "opd_contrastive_corrupt_changed_tokens": float(corrupt_changed_tokens),
        "opd_contrastive_corrupt_change_frac": (
            corrupt_changed_tokens / max(1, corrupt_span_token_total)
        ),
        "opd_contrastive_corrupt_noop_examples": float(corrupt_noop_examples),
        "opd_contrastive_corrupt_noop_frac": (
            corrupt_noop_examples / max(1, contrastive_examples)
        ),
        "opd_contrastive_corrupt_memory_only": float(
            (config.opd_contrastive_corrupt_buffer_span or "").strip().lower()
            in {"memory", "memory_only", "pause", "pause_only"}
        ),
        "opd_cache_mismatch_memory_weight": cache_mismatch_weight,
        "opd_cache_mismatch_balance_positive_hidden": float(balance_cache_mismatch_hidden),
        "opd_cache_mismatch_positive_hidden_boost": (
            cache_mismatch_weight if balance_cache_mismatch_hidden else 0.0
        ),
        "opd_memory_hidden_weight_balance_per_token": (
            positive_hidden_memory_weight
            - (contrastive_weight if corrupt_pair_active else 0.0)
            - (cache_mismatch_weight if cache_mismatch_weight > 0.0 else 0.0)
            if custom_pair_active
            else 0.0
        ),
        "opd_cache_mismatch_examples": float(cache_mismatch_examples),
        "opd_cache_mismatch_skipped_examples": float(cache_mismatch_skipped_examples),
        "opd_cache_mismatch_span_tokens": float(corrupt_span if cache_mismatch_examples else 0),
        "opd_cache_mismatch_span_token_total": float(cache_mismatch_span_token_total),
        "opd_cache_mismatch_changed_cache_rows": float(cache_mismatch_changed_rows),
        "opd_cache_mismatch_change_frac": (
            cache_mismatch_changed_rows / max(1, cache_mismatch_span_token_total)
        ),
        "opd_cache_mismatch_same_visible_input": float(cache_mismatch_examples > 0),
        "opd_cache_mismatch_negative_answer_kl_weight": 0.0,
        **teacher_memory_pair_metrics,
        **_student_generated_memory_prepare_metrics(config, chat_tokenizer),
        "prepare_s": _elapsed(prepare_t0),
        # Absolute perf_counter spans for overlap-aware WALL aggregation across the
        # concurrent prepare batches (the per-batch *_s above are durations that the
        # step row SUMS — a sum overstates wall). See _interval_union_s.
        "student_sampling_t0": sample_t0,
        "student_sampling_t1": sample_t0 + sample_s,
        "teacher_prefill_t0": teacher_t0,
        "teacher_prefill_t1": teacher_t0 + teacher_s,
    }
    return PreparedOpdBatch(
        sequences=sequences,
        data=data,
        cache_path=cache_path,
        metrics=metrics,
        completions=completions,
        prompt_texts=[_user_prompt_text(prompts[i // G]) for i in range(len(sequences))],
        layers_cache_path=oprd_layers_cache_path,
        oprd_layer_indices=oprd_layer_indices,
    )


def _aggregate_prepared_metrics(
    prepared_batches: list[PreparedOpdBatch],
) -> dict[str, Any]:
    sample_s = sum(
        float(batch.metrics.get("student_sampling_s", 0.0))
        for batch in prepared_batches
    )
    sample_output_tokens = sum(
        int(batch.metrics.get("student_sampling_output_tokens", 0))
        for batch in prepared_batches
    )
    teacher_s = sum(
        float(batch.metrics.get("teacher_prefill_s", 0.0)) for batch in prepared_batches
    )
    teacher_tokens = sum(
        int(batch.metrics.get("teacher_prefill_tokens", 0))
        for batch in prepared_batches
    )
    prepare_s = sum(
        float(batch.metrics.get("prepare_s", 0.0)) for batch in prepared_batches
    )
    # Overlap-aware WALL for each prepare phase (union of the concurrent batches'
    # spans), vs the sums above which overstate wall when prepare_concurrency > 1.
    # Read these for attribution; the *_s sums are cumulative across batches.
    sample_wall_s = _interval_union_s(
        [
            (float(batch.metrics["student_sampling_t0"]), float(batch.metrics["student_sampling_t1"]))
            for batch in prepared_batches
            if "student_sampling_t0" in batch.metrics
        ]
    )
    teacher_wall_s = _interval_union_s(
        [
            (float(batch.metrics["teacher_prefill_t0"]), float(batch.metrics["teacher_prefill_t1"]))
            for batch in prepared_batches
            if "teacher_prefill_t0" in batch.metrics
        ]
    )
    num_samples = sum(float(batch.metrics.get("num_samples", 0.0)) for batch in prepared_batches)
    num_opd_datums = sum(float(batch.metrics.get("num_opd_datums", 0.0)) for batch in prepared_batches)
    contrastive_examples = sum(
        float(batch.metrics.get("opd_contrastive_corrupt_examples", 0.0))
        for batch in prepared_batches
    )
    contrastive_weight = max(
        float(batch.metrics.get("opd_contrastive_corrupt_buffer_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    answer_contrastive_examples = sum(
        float(batch.metrics.get("opd_contrastive_corrupt_answer_examples", 0.0))
        for batch in prepared_batches
    )
    answer_contrastive_weight = max(
        float(batch.metrics.get("opd_contrastive_corrupt_answer_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    positive_answer_examples = sum(
        float(batch.metrics.get("opd_positive_answer_examples", 0.0))
        for batch in prepared_batches
    )
    positive_answer_weight = max(
        float(batch.metrics.get("opd_positive_answer_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    ptc_positive_examples = sum(
        float(batch.metrics.get("opd_ptc_positive_examples", 0.0))
        for batch in prepared_batches
    )
    ptc_buffer_kl_weight = max(
        float(batch.metrics.get("opd_ptc_positive_buffer_kl_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    ptc_answer_kl_weight = max(
        float(batch.metrics.get("opd_ptc_positive_answer_kl_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    ptc_hidden_weight = max(
        float(batch.metrics.get("opd_ptc_positive_hidden_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    mask_zero_weight_positions = max(
        float(batch.metrics.get("opd_mask_zero_weight_positions", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    gold_answer_replacements = sum(
        float(batch.metrics.get("opd_gold_answer_replacements", 0.0))
        for batch in prepared_batches
    )
    gold_answer_skipped = sum(
        float(batch.metrics.get("opd_gold_answer_skipped", 0.0))
        for batch in prepared_batches
    )
    teacher_answer_source_gold = max(
        float(batch.metrics.get("opd_teacher_answer_source_gold", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    position_weight_means = {
        metric_name: (
            sum(float(batch.metrics.get(metric_name, 0.0)) for batch in prepared_batches)
            / len(prepared_batches)
            if prepared_batches
            else 0.0
        )
        for metric_name in [
            "opd_teacher_weight_prompt_mean",
            "opd_teacher_weight_buffer_mean",
            "opd_teacher_weight_answer_mean",
            "opd_hidden_weight_prompt_mean",
            "opd_hidden_weight_buffer_mean",
            "opd_hidden_weight_answer_mean",
        ]
    }
    corrupt_span_tokens = max(
        float(batch.metrics.get("opd_contrastive_corrupt_span_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    corrupt_memory_only = max(
        float(batch.metrics.get("opd_contrastive_corrupt_memory_only", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    corrupt_span_token_total = sum(
        float(batch.metrics.get("opd_contrastive_corrupt_span_token_total", 0.0))
        for batch in prepared_batches
    )
    corrupt_changed_tokens = sum(
        float(batch.metrics.get("opd_contrastive_corrupt_changed_tokens", 0.0))
        for batch in prepared_batches
    )
    corrupt_noop_examples = sum(
        float(batch.metrics.get("opd_contrastive_corrupt_noop_examples", 0.0))
        for batch in prepared_batches
    )
    cache_mismatch_weight = max(
        float(batch.metrics.get("opd_cache_mismatch_memory_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    cache_mismatch_balance_positive_hidden = max(
        float(batch.metrics.get("opd_cache_mismatch_balance_positive_hidden", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    cache_mismatch_positive_hidden_boost = max(
        float(batch.metrics.get("opd_cache_mismatch_positive_hidden_boost", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    memory_hidden_weight_balance = max(
        float(batch.metrics.get("opd_memory_hidden_weight_balance_per_token", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    cache_mismatch_examples = sum(
        float(batch.metrics.get("opd_cache_mismatch_examples", 0.0))
        for batch in prepared_batches
    )
    cache_mismatch_skipped_examples = sum(
        float(batch.metrics.get("opd_cache_mismatch_skipped_examples", 0.0))
        for batch in prepared_batches
    )
    cache_mismatch_span_tokens = max(
        float(batch.metrics.get("opd_cache_mismatch_span_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    cache_mismatch_span_token_total = sum(
        float(batch.metrics.get("opd_cache_mismatch_span_token_total", 0.0))
        for batch in prepared_batches
    )
    cache_mismatch_changed_rows = sum(
        float(batch.metrics.get("opd_cache_mismatch_changed_cache_rows", 0.0))
        for batch in prepared_batches
    )
    cache_mismatch_same_visible = max(
        float(batch.metrics.get("opd_cache_mismatch_same_visible_input", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    cache_mismatch_negative_answer_kl_weight = max(
        float(batch.metrics.get("opd_cache_mismatch_negative_answer_kl_weight", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    memory_pair_requested = max(
        float(batch.metrics.get("opd_teacher_memory_pair_diag_requested", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    memory_pair_active = max(
        float(batch.metrics.get("opd_teacher_memory_pair_diag_active", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    memory_pair_failure = max(
        float(batch.metrics.get("opd_teacher_memory_pair_diag_failure", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    memory_pair_sample_count = sum(
        float(batch.metrics.get("opd_teacher_memory_pair_sample_count", 0.0))
        for batch in prepared_batches
    )
    memory_pair_skipped_samples = sum(
        float(batch.metrics.get("opd_teacher_memory_pair_skipped_samples", 0.0))
        for batch in prepared_batches
    )
    memory_pair_cross_tokens = sum(
        float(batch.metrics.get("opd_teacher_memory_pair_cross_token_count", 0.0))
        for batch in prepared_batches
    )
    memory_pair_within_tokens = sum(
        float(batch.metrics.get("opd_teacher_memory_pair_within_adjacent_token_count", 0.0))
        for batch in prepared_batches
    )

    def _weighted_memory_pair_mean(metric_name: str, weight_name: str) -> float:
        numerator = sum(
            float(batch.metrics.get(metric_name, 0.0)) * float(batch.metrics.get(weight_name, 0.0))
            for batch in prepared_batches
        )
        denominator = sum(float(batch.metrics.get(weight_name, 0.0)) for batch in prepared_batches)
        return numerator / denominator if denominator > 0.0 else 0.0

    memory_pair_mins = [
        float(batch.metrics.get("opd_teacher_memory_pair_cross_cosine_distance_min", 0.0))
        for batch in prepared_batches
        if float(batch.metrics.get("opd_teacher_memory_pair_cross_token_count", 0.0)) > 0.0
    ]
    memory_pair_maxes = [
        float(batch.metrics.get("opd_teacher_memory_pair_cross_cosine_distance_max", 0.0))
        for batch in prepared_batches
        if float(batch.metrics.get("opd_teacher_memory_pair_cross_token_count", 0.0)) > 0.0
    ]
    memory_pair_cross_distance_mean = _weighted_memory_pair_mean(
        "opd_teacher_memory_pair_cross_cosine_distance_mean",
        "opd_teacher_memory_pair_cross_token_count",
    )
    memory_pair_within_distance_mean = _weighted_memory_pair_mean(
        "opd_teacher_memory_pair_within_adjacent_distance_mean",
        "opd_teacher_memory_pair_within_adjacent_token_count",
    )
    generated_memory_active = max(
        float(batch.metrics.get("student_generated_memory_active", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    generated_memory_tokens = max(
        float(batch.metrics.get("student_generated_memory_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    generated_memory_seed_tokens = max(
        float(batch.metrics.get("student_generated_memory_seed_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    generated_memory_suffix_tokens = max(
        float(batch.metrics.get("student_generated_memory_suffix_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    generated_memory_total_filler_tokens = max(
        float(batch.metrics.get("student_generated_memory_total_filler_tokens", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    strict_prepare_overlap_active = max(
        float(batch.metrics.get("strict_prepare_overlap_active", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    strict_prepare_overlap_batch_count = sum(
        1.0
        for batch in prepared_batches
        if float(batch.metrics.get("strict_prepare_overlap_active", 0.0)) > 0.0
    )
    strict_prepare_overlap_chunks = max(
        float(batch.metrics.get("strict_prepare_overlap_chunks", 0.0))
        for batch in prepared_batches
    ) if prepared_batches else 0.0
    strict_prepare_overlap_wall_s = sum(
        float(batch.metrics.get("strict_prepare_overlap_wall_s", 0.0))
        for batch in prepared_batches
    )
    strict_prepare_overlap_student_sampling_sum_s = sum(
        float(batch.metrics.get("strict_prepare_overlap_student_sampling_sum_s", 0.0))
        for batch in prepared_batches
    )
    strict_prepare_overlap_teacher_prefill_sum_s = sum(
        float(batch.metrics.get("strict_prepare_overlap_teacher_prefill_sum_s", 0.0))
        for batch in prepared_batches
    )

    return {
        "num_prepare_batches": len(prepared_batches),
        "num_samples": num_samples,
        "num_opd_datums": num_opd_datums,
        "opd_contrastive_corrupt_buffer_weight": contrastive_weight,
        "opd_contrastive_corrupt_answer_weight": answer_contrastive_weight,
        "opd_positive_answer_weight": positive_answer_weight,
        "opd_ptc_positive_buffer_kl_weight": ptc_buffer_kl_weight,
        "opd_ptc_positive_answer_kl_weight": ptc_answer_kl_weight,
        "opd_ptc_positive_hidden_weight": ptc_hidden_weight,
        "opd_mask_zero_weight_positions": mask_zero_weight_positions,
        "opd_teacher_answer_source_gold": teacher_answer_source_gold,
        "opd_gold_answer_replacements": gold_answer_replacements,
        "opd_gold_answer_skipped": gold_answer_skipped,
        "opd_contrastive_corrupt_examples": contrastive_examples,
        "opd_contrastive_corrupt_answer_examples": answer_contrastive_examples,
        "opd_positive_answer_examples": positive_answer_examples,
        "opd_ptc_positive_examples": ptc_positive_examples,
        **position_weight_means,
        "opd_contrastive_data_multiplier": num_opd_datums / max(1.0, num_samples),
        "opd_contrastive_corrupt_span_tokens": corrupt_span_tokens,
        "opd_contrastive_corrupt_span_token_total": corrupt_span_token_total,
        "opd_contrastive_corrupt_changed_tokens": corrupt_changed_tokens,
        "opd_contrastive_corrupt_change_frac": (
            corrupt_changed_tokens / max(1.0, corrupt_span_token_total)
        ),
        "opd_contrastive_corrupt_noop_examples": corrupt_noop_examples,
        "opd_contrastive_corrupt_noop_frac": (
            corrupt_noop_examples / max(1.0, contrastive_examples)
        ),
        "opd_contrastive_corrupt_memory_only": corrupt_memory_only,
        "opd_cache_mismatch_memory_weight": cache_mismatch_weight,
        "opd_cache_mismatch_balance_positive_hidden": cache_mismatch_balance_positive_hidden,
        "opd_cache_mismatch_positive_hidden_boost": cache_mismatch_positive_hidden_boost,
        "opd_memory_hidden_weight_balance_per_token": memory_hidden_weight_balance,
        "opd_cache_mismatch_examples": cache_mismatch_examples,
        "opd_cache_mismatch_skipped_examples": cache_mismatch_skipped_examples,
        "opd_cache_mismatch_span_tokens": cache_mismatch_span_tokens,
        "opd_cache_mismatch_span_token_total": cache_mismatch_span_token_total,
        "opd_cache_mismatch_changed_cache_rows": cache_mismatch_changed_rows,
        "opd_cache_mismatch_change_frac": (
            cache_mismatch_changed_rows / max(1.0, cache_mismatch_span_token_total)
        ),
        "opd_cache_mismatch_same_visible_input": cache_mismatch_same_visible,
        "opd_cache_mismatch_negative_answer_kl_weight": cache_mismatch_negative_answer_kl_weight,
        "opd_teacher_memory_pair_diag_requested": memory_pair_requested,
        "opd_teacher_memory_pair_diag_active": memory_pair_active,
        "opd_teacher_memory_pair_diag_failure": memory_pair_failure,
        "opd_teacher_memory_pair_sample_count": memory_pair_sample_count,
        "opd_teacher_memory_pair_skipped_samples": memory_pair_skipped_samples,
        "opd_teacher_memory_pair_cross_token_count": memory_pair_cross_tokens,
        "opd_teacher_memory_pair_cross_cosine_similarity_mean": _weighted_memory_pair_mean(
            "opd_teacher_memory_pair_cross_cosine_similarity_mean",
            "opd_teacher_memory_pair_cross_token_count",
        ),
        "opd_teacher_memory_pair_cross_cosine_distance_mean": memory_pair_cross_distance_mean,
        "opd_teacher_memory_pair_cross_cosine_distance_min": (
            min(memory_pair_mins) if memory_pair_mins else 0.0
        ),
        "opd_teacher_memory_pair_cross_cosine_distance_max": (
            max(memory_pair_maxes) if memory_pair_maxes else 0.0
        ),
        "opd_teacher_memory_pair_within_adjacent_token_count": memory_pair_within_tokens,
        "opd_teacher_memory_pair_within_adjacent_distance_mean": memory_pair_within_distance_mean,
        "opd_teacher_memory_pair_cross_minus_within_distance": (
            memory_pair_cross_distance_mean - memory_pair_within_distance_mean
            if memory_pair_cross_tokens > 0.0 and memory_pair_within_tokens > 0.0
            else 0.0
        ),
        "student_generated_memory_active": generated_memory_active,
        "student_generated_memory_tokens": generated_memory_tokens,
        "student_generated_memory_seed_tokens": generated_memory_seed_tokens,
        "student_generated_memory_suffix_tokens": generated_memory_suffix_tokens,
        "student_generated_memory_total_filler_tokens": generated_memory_total_filler_tokens,
        "strict_prepare_overlap_active": strict_prepare_overlap_active,
        "strict_prepare_overlap_batch_count": strict_prepare_overlap_batch_count,
        "strict_prepare_overlap_chunks": strict_prepare_overlap_chunks,
        "strict_prepare_overlap_wall_s": strict_prepare_overlap_wall_s,
        "strict_prepare_overlap_student_sampling_sum_s": strict_prepare_overlap_student_sampling_sum_s,
        "strict_prepare_overlap_teacher_prefill_sum_s": strict_prepare_overlap_teacher_prefill_sum_s,
        "student_sampling_s": sample_s,
        "student_sampling_wall_s": sample_wall_s,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s
        if sample_s > 0
        else 0.0,
        "teacher_prefill_s": teacher_s,
        "teacher_prefill_wall_s": teacher_wall_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_s
        if teacher_s > 0
        else 0.0,
        "teacher_prefill_forward_compute_s": sum(
            float(batch.metrics.get("teacher_prefill_forward_compute_s", 0.0))
            for batch in prepared_batches
        ),
        "teacher_hidden_cache_write_s": sum(
            float(batch.metrics.get("teacher_hidden_cache_write_s", 0.0))
            for batch in prepared_batches
        ),
        "prepare_s": prepare_s,
    }


def _loss_mean(output: tomi.ForwardBackwardOutput) -> float:
    losses = [item.loss for item in output.loss_fn_outputs if item.loss is not None]
    return _mean([float(loss) for loss in losses])


def _loss_mean_many(outputs: list[tomi.ForwardBackwardOutput]) -> float:
    losses = [
        float(item.loss)
        for output in outputs
        for item in output.loss_fn_outputs
        if item.loss is not None
    ]
    return _mean(losses)


def _valid_tokens(output: tomi.ForwardBackwardOutput) -> int | float:
    return output.metrics.get(
        "is_valid_tokens:sum",
        output.metrics.get("valid_tokens:sum", output.metrics.get("valid_tokens", 0)),
    )


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    # The xorl-internal trainer applies a reduction suffix to each metric
    # (see ModelRunner._metric_accumulator_key): `:mean` for most OPD fields,
    # `:max` for `opd_num_teachers`, `:sum_max` for `opd_profile_*_ms`. Check
    # all candidates so the OPD client doesn't silently miss a metric just
    # because its reducer differs from the default.
    for candidate in (key, f"{key}:sum", f"{key}:mean", f"{key}:max", f"{key}:sum_max", f"{key}:min"):
        value = metrics.get(candidate)
        if isinstance(value, numbers.Number):
            return float(value)
    return None


def _aggregate_forward_backward_profile_metrics(
    outputs: list[tomi.ForwardBackwardOutput],
) -> dict[str, float]:
    specs = {
        "opd_profile_forward_compute_s": ("opd_profile_forward_compute_s", 1.0),
        "opd_profile_backward_compute_s": ("opd_profile_backward_compute_s", 1.0),
        "opd_profile_input_transfer_s": ("opd_profile_input_transfer_s", 1.0),
        "opd_profile_per_token_collect_s": ("opd_profile_per_token_collect_s", 1.0),
        "opd_profile_deferred_k3_s": ("opd_profile_deferred_k3_s", 1.0),
        "opd_profile_loss_report_allreduce_s": ("opd_profile_loss_report_allreduce_s", 1.0),
        "opd_profile_sp_grad_sync_s": ("opd_profile_sp_grad_sync_s", 1.0),
        "opd_profile_metric_finalize_s": ("opd_profile_metric_finalize_s", 1.0),
        "opd_profile_final_synchronize_s": ("opd_profile_final_synchronize_s", 1.0),
        "opd_profile_forward_loop_total_s": ("opd_profile_forward_loop_total_s", 1.0),
        "opd_profile_prefetch_s": ("opd_profile_prefetch_ms", 0.001),
        "opd_profile_hidden_fetch_s": ("opd_profile_hidden_fetch_ms", 0.001),
        "opd_profile_head_prepare_s": ("opd_profile_head_prepare_ms", 0.001),
        "opd_profile_kl_compute_s": ("opd_profile_kl_compute_ms", 0.001),
        "opd_profile_model_forward_s": ("opd_profile_model_forward_ms", 0.001),
        "opd_profile_loss_compute_s": ("opd_profile_loss_compute_ms", 0.001),
        "opd_profile_loss_total_s": ("opd_profile_total_ms", 0.001),
        "opd_profile_clear_gradients_s": ("opd_profile_clear_gradients_ms", 0.001),
    }
    totals: dict[str, float] = {}
    for output in outputs:
        metrics = output.metrics or {}
        for output_key, (metric_key, scale) in specs.items():
            value = _metric_value(metrics, metric_key)
            if value is not None:
                totals[output_key] = totals.get(output_key, 0.0) + value * scale
    return totals


def _aggregate_opd_loss_metrics(
    outputs: list[tomi.ForwardBackwardOutput],
) -> dict[str, float]:
    """Token-weighted average of per-microbatch OPDLossMetrics fields.

    The xorl-internal OPD loss path (PR #320 + #323) emits a per-microbatch
    `OPDLossMetrics.to_dict()` payload containing the actual distillation
    diagnostics: full-vocab KL, full-vocab entropy per side, top1 agreement,
    estimator-mode abs_loss, loss min/max/abs_mean, multi-teacher weight
    bookkeeping, and PG-mode clipfrac / ppo_kl. They live in each
    `ForwardBackwardOutput.metrics`.

    The xorl-internal trainer averages OVER microbatch tokens (not over
    microbatches), so to recover a step-level scalar we re-weight each
    microbatch's mean by its `valid_tokens` count and divide by the total.
    Mirrors what VERL's aggregator does for `distillation/*` per step.
    """
    # Per-call key inventory — keep in sync with src/xorl/ops/loss/opd_loss.py
    # OPDLossMetrics.to_dict(). Numeric fields are mean-aggregated by valid
    # tokens; "valid_tokens" itself is summed separately by the caller; the
    # discrete `opd_num_teachers` field is taken as the max (sane fallback —
    # ranks with zero valid tokens emit 0).
    SUM_KEYS = ("opd_num_teachers",)
    MEAN_KEYS = (
        "opd_kl",
        "opd_weighted_kl",
        "opd_teacher_weight_mean",
        "opd_teacher_entropy",
        "opd_student_entropy",
        "opd_top1_agreement",
        "opd_abs_loss",
        "opd_loss_min",
        "opd_loss_max",
        "opd_loss_abs_mean",
        "opd_pg_clipfrac",
        "opd_pg_clipfrac_lower",
        "opd_ppo_kl",
        "opd_hidden_match_loss",
        "opd_hidden_match_raw_loss",
        "opd_hidden_match_weight_mean",
        "opd_hidden_match_pos_loss",
        "opd_hidden_match_neg_loss",
        "opd_hidden_match_pos_raw_loss",
        "opd_hidden_match_neg_raw_loss",
        "opd_hidden_match_neg_minus_pos_raw",
        "opd_hidden_match_pos_weight_mean",
        "opd_hidden_match_neg_weight_mean",
        "opd_oprd_loss",
        "opd_oprd_raw_loss",
        "opd_oprd_num_layers",
        # Clamp-frac + region/correctness KL splits. The `*_per_valid` fields are
        # masked sums over TOTAL valid tokens and `opd_frac_*` the matching token
        # fractions, so the valid-token weighting here recomposes them exactly;
        # _derive_opd_split_means turns them into human-readable region means.
        "opd_loss_clamp_frac",
        "opd_kl_prompt_per_valid",
        "opd_kl_buffer_per_valid",
        "opd_kl_answer_per_valid",
        "opd_frac_prompt",
        "opd_frac_buffer",
        "opd_frac_answer",
        "opd_kl_answer_correct_per_valid",
        "opd_kl_answer_wrong_per_valid",
        "opd_frac_answer_correct",
        "opd_frac_answer_wrong",
        "opd_student_entropy_answer_correct_per_valid",
        "opd_student_entropy_answer_wrong_per_valid",
        "opd_teacher_entropy_answer_correct_per_valid",
        "opd_teacher_entropy_answer_wrong_per_valid",
    )

    mean_totals: dict[str, float] = dict.fromkeys(MEAN_KEYS, 0.0)
    weight_total = 0.0
    max_vals: dict[str, float] = dict.fromkeys(SUM_KEYS, 0.0)

    for output in outputs:
        metrics = output.metrics or {}
        valid = _metric_value(metrics, "valid_tokens") or 0.0
        if valid <= 0:
            # Skip ranks/microbatches with zero valid tokens — their metric
            # contributions are unconstrained noise (per OPDLossMetrics docstring).
            continue
        weight_total += valid
        for key in MEAN_KEYS:
            value = _metric_value(metrics, key)
            if value is not None:
                mean_totals[key] += float(value) * valid
        for key in SUM_KEYS:
            value = _metric_value(metrics, key)
            if value is not None:
                max_vals[key] = max(max_vals[key], float(value))

    if weight_total <= 0:
        return {}
    result: dict[str, float] = {k: v / weight_total for k, v in mean_totals.items()}
    for k, v in max_vals.items():
        result[k] = v
    _derive_opd_split_means(result)
    return result


def _derive_opd_split_means(metrics: dict[str, float]) -> None:
    """Turn the exact per-valid split sums into human-readable region means.

    `opd_kl_<region>_mean` = mean KL over that region's tokens only (vs the
    `*_per_valid` raw fields which are normalized by ALL valid tokens). Added
    in-place; 0.0 when the region has no tokens.
    """
    for numerator, frac, out in (
        ("opd_kl_prompt_per_valid", "opd_frac_prompt", "opd_kl_prompt_mean"),
        ("opd_kl_buffer_per_valid", "opd_frac_buffer", "opd_kl_buffer_mean"),
        ("opd_kl_answer_per_valid", "opd_frac_answer", "opd_kl_answer_mean"),
        ("opd_kl_answer_correct_per_valid", "opd_frac_answer_correct", "opd_kl_answer_correct_mean"),
        ("opd_kl_answer_wrong_per_valid", "opd_frac_answer_wrong", "opd_kl_answer_wrong_mean"),
        (
            "opd_student_entropy_answer_correct_per_valid",
            "opd_frac_answer_correct",
            "opd_student_entropy_answer_correct_mean",
        ),
        (
            "opd_student_entropy_answer_wrong_per_valid",
            "opd_frac_answer_wrong",
            "opd_student_entropy_answer_wrong_mean",
        ),
        (
            "opd_teacher_entropy_answer_correct_per_valid",
            "opd_frac_answer_correct",
            "opd_teacher_entropy_answer_correct_mean",
        ),
        (
            "opd_teacher_entropy_answer_wrong_per_valid",
            "opd_frac_answer_wrong",
            "opd_teacher_entropy_answer_wrong_mean",
        ),
    ):
        denom = metrics.get(frac, 0.0)
        metrics[out] = (metrics.get(numerator, 0.0) / denom) if denom > 0.0 else 0.0


# streaming/tilelang gained full-vocab diagnostics via a no-grad streaming pass
# (xorl opd_streaming_kl.streaming_full_vocab_diagnostics); reverse_kl_full only.
_FULL_VOCAB_DIAGNOSTIC_BACKENDS = {"torch_compile", "compile", "auto_chunker", "streaming", "tilelang"}


def _full_vocab_diagnostic_metrics(config: Config) -> dict[str, float]:
    requested = bool(config.opd_emit_full_vocab_diagnostics)
    active = requested and config.opd_kl_backend.strip().lower() in _FULL_VOCAB_DIAGNOSTIC_BACKENDS
    return {
        "opd_full_vocab_diag_requested": float(requested),
        "opd_full_vocab_diag_active_expected": float(active),
        "opd_full_vocab_diag_unavailable_expected": float(requested and not active),
    }


def _weight_sync_master_address(config: Config) -> str | None:
    return (
        config.weight_sync_master_address
        or os.environ.get("XORL_WEIGHT_SYNC_MASTER_ADDRESS")
        or None
    )


_SYNC_ENDPOINT_RE = re.compile(r"\bto\s+(\d+)\s+endpoint")


def _sync_profile_metrics(sync_result: Any) -> dict[str, Any]:
    message = getattr(sync_result, "message", "") or ""
    endpoints = list(getattr(sync_result, "endpoints_synced", []) or [])
    endpoint_count = len(endpoints)
    if endpoint_count == 0:
        match = _SYNC_ENDPOINT_RE.search(message)
        if match:
            endpoint_count = int(match.group(1))
    endpoint_success_count = sum(1 for endpoint in endpoints if bool(getattr(endpoint, "success", False)))
    timing = getattr(sync_result, "timing_breakdown", {}) or {}
    p2p_summaries = [
        summary
        for summary in (getattr(sync_result, "p2p_rank_summaries", []) or [])
        if isinstance(summary, Mapping)
    ]
    transfer_wall_values = []
    for summary in p2p_summaries:
        value = summary.get("transfer_wall_s")
        try:
            transfer_wall_values.append(float(value))
        except (TypeError, ValueError):
            continue

    metrics: dict[str, Any] = {
        "sync_endpoint_count": endpoint_count,
        "sync_endpoint_success_count": endpoint_success_count,
        "sync_endpoint_failure_count": max(endpoint_count - endpoint_success_count, 0),
        "sync_serial_endpoint_sync": float(
            bool(timing.get("serial_endpoint_sync", False))
            or message.lower().startswith("serial endpoint sync")
        ),
        "sync_p2p_rank_summary_count": len(p2p_summaries),
    }
    if endpoints:
        failed = [endpoint for endpoint in endpoints if not bool(getattr(endpoint, "success", False))]
        if failed:
            metrics["sync_first_failed_endpoint"] = f"{failed[0].host}:{failed[0].port}"
            metrics["sync_first_failed_endpoint_message"] = getattr(failed[0], "message", "")
    if transfer_wall_values:
        metrics["sync_p2p_transfer_wall_max_s"] = max(transfer_wall_values)
        metrics["sync_p2p_transfer_wall_mean_s"] = _mean(transfer_wall_values)
    if p2p_summaries:
        endpoint_indices = {
            summary.get("endpoint_index")
            for summary in p2p_summaries
            if summary.get("endpoint_index") is not None
        }
        metrics["sync_p2p_summary_endpoint_count"] = len(endpoint_indices)

    interesting_timing_keys = {
        "serial_endpoint_sync",
        "serial_endpoint_count",
        "health_check_s",
        "backend_init_s",
        "pause_s",
        "transfer_s",
        "complete_s",
        "resume_s",
        "total_handler_s",
        "max_rank_transfer_s",
        "rank_transfer_spread_s",
        "p2p_backend_max_transfer_s",
        "p2p_backend_max_total_s",
    }
    endpoint_timing_suffixes = (
        "/backend_init_s",
        "/transfer_s",
        "/complete_s",
        "/resume_s",
        "/total_handler_s",
        "/p2p_backend_max_transfer_s",
        "/p2p_backend_max_total_s",
    )
    for key, value in timing.items():
        if key not in interesting_timing_keys and not key.endswith(endpoint_timing_suffixes):
            continue
        try:
            metrics[f"sync_timing/{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


async def main(config: Config) -> None:
    if not config.teacher_head:
        raise ValueError(
            "teacher_head must point to the teacher prediction head or teacher model path"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = (
        Path(config.profile_output)
        if config.profile_output
        else output_dir / "opd_profile.jsonl"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("", encoding="utf-8")

    wandb_run = _maybe_init_wandb(config)

    prompts = _load_prompts(config)
    if not prompts:
        raise ValueError("OPD requires at least one prompt")
    chat_tokenizer = _load_chat_tokenizer(config)
    teacher_prefix_tokens = _tokenize_teacher_prefix(config, chat_tokenizer)
    if teacher_prefix_tokens:
        logger.info(
            "Teacher system prefix: %d tokens (%r...)",
            len(teacher_prefix_tokens),
            (config.teacher_system_prefix or "<file>")[:80],
        )
    teacher_filler_tokens: list[int] | list[list[int]] = []
    teacher_filler_by_prompt: list[list[int]] | None = None
    if config.teacher_cot_json_path:
        if chat_tokenizer is None:
            raise ValueError(
                "teacher_cot_json_path requires a chat_tokenizer (set chat_tokenizer_path)"
            )
        cot_data = json.loads(Path(config.teacher_cot_json_path).read_text())
        if not isinstance(cot_data, list):
            raise ValueError(
                f"teacher_cot_json_path file {config.teacher_cot_json_path} must contain a list"
            )
        if len(cot_data) < len(prompts):
            raise ValueError(
                f"teacher_cot_json_path has {len(cot_data)} entries but need {len(prompts)} prompts"
            )
        teacher_filler_by_prompt = []
        for idx, entry in enumerate(cot_data[: len(prompts)]):
            if not isinstance(entry, dict) or "cot" not in entry:
                raise ValueError(
                    f"teacher_cot_json_path entry {idx} missing 'cot' field"
                )
            cot_text = entry["cot"]
            tokens = [
                int(t) for t in chat_tokenizer.encode(cot_text, add_special_tokens=False)
            ]
            if not tokens:
                raise ValueError(
                    f"teacher_cot_json_path entry {idx} produced 0 tokens (empty cot?) "
                    "— precompute the dataset until every entry has a real CoT, or "
                    "filter the bad indices out of both the prompts and cot JSON before "
                    "passing them to this trainer."
                )
            teacher_filler_by_prompt.append(tokens)
        cot_lengths = [len(f) for f in teacher_filler_by_prompt]
        logger.info(
            "Teacher per-prompt CoT loaded: %d entries, min=%d median=%d max=%d tokens",
            len(teacher_filler_by_prompt),
            min(cot_lengths),
            sorted(cot_lengths)[len(cot_lengths) // 2],
            max(cot_lengths),
        )
    elif config.teacher_filler_count > 0 and config.teacher_filler_text:
        if chat_tokenizer is None:
            raise ValueError(
                "teacher_filler_text requires a chat_tokenizer (set chat_tokenizer_path)"
            )
        # Tokenize the filler text repeated `teacher_filler_count` times
        filler_string = config.teacher_filler_text * config.teacher_filler_count
        teacher_filler_tokens = [
            int(t) for t in chat_tokenizer.encode(filler_string, add_special_tokens=False)
        ]
        logger.info(
            "Teacher filler insert: %d tokens from %r × %d",
            len(teacher_filler_tokens),
            config.teacher_filler_text,
            config.teacher_filler_count,
        )
    student_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
    student_stop_sequences = _parse_stop_sequences(config.student_stop_sequences)
    if student_prefill_ids and config.student_prefill_text:
        raise ValueError("set either student_prefill_token_ids or student_prefill_text, not both")
    if config.student_generated_memory_tokens > 0:
        if student_prefill_ids:
            raise ValueError("student_generated_memory_tokens does not support student_prefill_token_ids")
        if chat_tokenizer is None:
            raise ValueError("student_generated_memory_tokens requires a chat_tokenizer_path")
    if config.student_prefill_count > 0 or student_prefill_ids or config.student_prefill_suffix:
        if config.student_prefill_count > 0 and not config.student_prefill_text and not student_prefill_ids:
            raise ValueError(
                "student_prefill_count > 0 requires student_prefill_text or student_prefill_token_ids"
            )
        if chat_tokenizer is None:
            raise ValueError(
                "student prefill requires a chat_tokenizer (set chat_tokenizer_path)"
            )
        suffix_token_count = (
            len(chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False))
            if config.student_prefill_suffix
            else 0
        )
        if student_prefill_ids:
            prefill_token_count = len(student_prefill_ids)
            logger.info(
                "Student exact-token prefill: %d buffer tokens + suffix %r (%d tokens) = %d filler tokens "
                "(K, /generate input_ids)",
                prefill_token_count,
                config.student_prefill_suffix,
                suffix_token_count,
                prefill_token_count + suffix_token_count,
            )
        else:
            prefill_token_count = len(
                chat_tokenizer.encode(
                    config.student_prefill_text * config.student_prefill_count,
                    add_special_tokens=False,
                )
            )
            logger.info(
                "Student prefill: %r × %d + suffix %r → %d pause + %d suffix = %d filler tokens "
                "(K, continue_final_message)",
                config.student_prefill_text,
                config.student_prefill_count,
                config.student_prefill_suffix,
                prefill_token_count,
                suffix_token_count,
                prefill_token_count + suffix_token_count,
            )
            if prefill_token_count != config.student_prefill_count:
                # Multi-token filler (e.g. nato): k_filler uses the full tokenized prefix
                # length and the teacher-cache remap masks the K region by total length,
                # so multi-token is correctly aligned. Warn, don't raise.
                logger.warning(
                    "multi-token filler: %r -> %d tokens for count=%d; using full length "
                    "as K_student (remap masks K by length, alignment preserved).",
                    config.student_prefill_text, prefill_token_count, config.student_prefill_count,
                )
    if config.student_generated_memory_tokens > 0:
        seed_token_count = (
            len(
                chat_tokenizer.encode(
                    config.student_prefill_text * config.student_prefill_count,
                    add_special_tokens=False,
                )
            )
            if chat_tokenizer is not None and config.student_prefill_text and config.student_prefill_count > 0
            else 0
        )
        suffix_token_count = (
            len(chat_tokenizer.encode(config.student_prefill_suffix, add_special_tokens=False))
            if chat_tokenizer is not None and config.student_prefill_suffix
            else 0
        )
        logger.info(
            "Student generated memory: seed %r × %d -> %d tokens + generated %d tokens + suffix %r "
            "(%d tokens) = %d filler tokens (K, dynamic per prompt)",
            config.student_prefill_text,
            config.student_prefill_count,
            seed_token_count,
            config.student_generated_memory_tokens,
            config.student_prefill_suffix,
            suffix_token_count,
            seed_token_count + int(config.student_generated_memory_tokens) + suffix_token_count,
        )
    if config.eval_generated_memory_control:
        eval_generated_tokens = int(config.eval_generated_memory_tokens or config.student_generated_memory_tokens or 0)
        if eval_generated_tokens <= 0:
            raise ValueError(
                "eval_generated_memory_control requires eval_generated_memory_tokens "
                "or student_generated_memory_tokens > 0"
            )
        logger.info(
            "Generated-memory control eval enabled: %d memory tokens, shuffle_offset=%d, temperature=%.3f",
            eval_generated_tokens,
            max(1, int(config.eval_generated_memory_shuffle_offset or 1)),
            float(config.eval_generated_memory_temperature),
        )
    if student_stop_sequences:
        logger.info("Student rollout/control stop sequences: %r", student_stop_sequences)
    student_urls = get_inference_urls(config.inference_base_urls, config.inference_port)
    sampler_metrics_urls = get_sampler_metrics_urls(
        config.sampler_metrics_urls,
        student_urls,
        config.sampler_metrics_port,
    )
    if sampler_metrics_urls:
        logger.info("Sampler metrics enabled: %s", ", ".join(sampler_metrics_urls))

    logger.info("Waiting for trainer: %s", config.base_url)
    _wait_for_xorl(config.base_url, timeout=config.endpoint_timeout)
    teacher_is_sglang = config.teacher_backend == "sglang"
    logger.info(
        "Waiting for teacher (%s): %s", config.teacher_backend, config.teacher_base_url
    )
    if teacher_is_sglang:
        _wait_for_sglang(config.teacher_base_url, timeout=config.endpoint_timeout)
    else:
        _wait_for_xorl(config.teacher_base_url, timeout=config.endpoint_timeout)
    for url in student_urls:
        logger.info("Waiting for student sampler: %s", url)
        _wait_for_sglang(url, timeout=config.endpoint_timeout)
    _ensure_xorl_session(
        config.base_url,
        model_id=config.model_id,
        base_model=config.model_name,
        timeout=config.request_timeout,
    )
    if not teacher_is_sglang:
        _ensure_xorl_session(
            config.teacher_base_url,
            model_id=config.teacher_model_id,
            base_model=config.teacher_head,
            timeout=config.request_timeout,
        )

    service_client = tomi.ServiceClient(
        base_url=config.base_url, timeout=config.request_timeout
    )
    training_client = TrainingClient(
        holder=service_client.holder,
        model_id=config.model_id,
        base_model=config.model_name,
    )
    sampler_api_format = "generate" if student_prefill_ids else (config.inference_api_format or None)
    if student_prefill_ids:
        logger.info("Student exact-token prefill active: using sampler api_format=generate for input_ids")
    sampling_clients = [
        tomi.SamplingClient(
            base_url=url,
            model=config.model_name,
            timeout=config.request_timeout,
            api_format=sampler_api_format,
        )
        for url in student_urls
    ]
    logprob_urls = (
        get_inference_urls(config.inference_logprob_base_urls, config.inference_port)
        if config.inference_logprob_base_urls
        else []
    )
    logprob_clients = [
        tomi.SamplingClient(
            base_url=url,
            model=config.model_name,
            timeout=config.request_timeout,
            api_format="generate",
        )
        for url in logprob_urls
    ]
    if logprob_clients:
        logger.info("Answer logprob scoring uses native SGLang endpoints: %s", ", ".join(logprob_urls))
    eval_urls = (
        get_inference_urls(config.eval_inference_base_urls, config.inference_port)
        if config.eval_inference_base_urls
        else []
    )
    eval_sampling_clients = [
        tomi.SamplingClient(
            base_url=url,
            model=config.model_name,
            timeout=config.request_timeout,
            api_format=sampler_api_format,
        )
        for url in eval_urls
    ]
    if eval_sampling_clients:
        logger.info("Greedy evals use the dedicated eval pool: %s", ", ".join(eval_urls))
    if config.eval_async and not eval_sampling_clients:
        raise ValueError("eval_async=true requires eval_inference_base_urls (a dedicated eval pool)")
    # When a dedicated eval pool exists, the per-step sync must NOT touch it —
    # its weights stay frozen until the eval-step pools=["eval"] refresh.
    train_sync_pools = ["default"] if eval_sampling_clients else None
    train_microbatch_size = config.opd_microbatch_size
    prepare_batch_size = config.opd_prepare_batch_size or train_microbatch_size
    prepare_concurrency = max(1, config.opd_prepare_concurrency)

    prompts_per_step = int(config.prompts_per_step or 0)
    if prompts_per_step < 0:
        raise ValueError(f"prompts_per_step must be >= 0, got {prompts_per_step}")
    if prompts_per_step > len(prompts):
        raise ValueError(
            f"prompts_per_step ({prompts_per_step}) exceeds the loaded prompt pool "
            f"({len(prompts)}); load more prompts or lower prompts_per_step"
        )

    def _slice_window(seq: list, step: int) -> list:
        """Rolling, wrap-around window of `prompts_per_step` items for this step."""
        if prompts_per_step <= 0:
            return seq
        n = len(seq)
        start = (step * prompts_per_step) % n
        end = start + prompts_per_step
        if end <= n:
            return seq[start:end]
        # Window straddles the end of the pool — wrap to the front.
        return seq[start:] + seq[: end - n]

    if prompts_per_step > 0:
        logger.info(
            "Per-step prompt window: %d prompts/step over a pool of %d → "
            "%d distinct prompts before repeat (run is %d steps)",
            prompts_per_step,
            len(prompts),
            len(prompts),
            config.num_steps,
        )

    # Pipelined two-phase teacher prefill: validate the recipe + precompute the
    # fixed forced-prefix tokens. Phase A is built deterministically (no sampling)
    # so it can be prefetched one step ahead.
    pipeline_phase = bool(config.teacher_pipeline_phase)
    # Async-overlapped sampling ("pipeline RL"). Orthogonal to (and mutually
    # exclusive with) the DEPRECATED two-phase teacher_pipeline_phase: the latter
    # caused a training regression, so they must not be combined.
    pipeline_rl = bool(config.opd_pipeline_rl)
    # OPRD is compatible with async-overlapped pipeline_rl because that path calls
    # _prepare_step_batches -> _prepare_opd_batch, the same single-phase prepare
    # used by the serial loop. It remains incompatible with the deprecated
    # teacher_pipeline_phase path below, which uses _prepare_opd_batch_pipelined.
    if pipeline_rl and pipeline_phase:
        raise ValueError(
            "opd_pipeline_rl and teacher_pipeline_phase are mutually exclusive; "
            "opd_pipeline_rl uses the plain single-phase teacher path (the two-phase "
            "teacher_pipeline_phase is deprecated/buggy)."
        )
    if config.student_generated_memory_tokens > 0 and pipeline_phase:
        raise ValueError(
            "student_generated_memory_tokens is incompatible with teacher_pipeline_phase; "
            "generated memory is prompt/sample-specific and cannot use deterministic "
            "Phase A prefixes."
        )

    # Group sampling + group teacher (group_size G). G>1 multiplies the per-step
    # sample count to N*G (N prompts × G completions/prompt) via _prepare_opd_batch,
    # which then uses the shared-prefix group teacher when the supervise recipe is
    # active. The deprecated two-phase teacher_pipeline_phase has its own (single-
    # sample) Phase-A prefetch path and must NOT be combined with group sampling.
    group_size = max(1, int(config.group_size))
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {config.group_size}")
    if group_size > 1 and pipeline_phase:
        raise ValueError(
            "group_size > 1 is incompatible with teacher_pipeline_phase (deprecated); "
            "use opd_pipeline_rl for overlap instead."
        )
    if group_size > 1:
        group_teacher_ready = (
            config.teacher_backend == "sglang"
            and config.teacher_cot_mode == "insert"
            and config.supervise_student_cot
            and teacher_filler_by_prompt is not None
            and config.student_prefill_count > 0
        )
        logger.info(
            "Group sampling ON: group_size=%d → %d samples/prompt; group teacher "
            "(shared prompt+CoT+pause prefix, Phase A once/prompt + Phase B/sample, "
            "radix-served) %s. Total samples/step = N_prompts × %d.",
            group_size,
            group_size,
            "ACTIVE (radix-on teacher required)" if group_teacher_ready else
            "INACTIVE — falling back to independent per-sample teacher prefill "
            "(needs teacher_backend=sglang + teacher_cot_mode=insert + "
            "supervise_student_cot + teacher_cot_json_path + student_prefill_count>0)",
            group_size,
        )
    forced_prefix_tokens: list[int] = []
    if pipeline_phase:
        if config.teacher_backend != "sglang":
            raise ValueError("teacher_pipeline_phase requires teacher_backend=sglang")
        if teacher_filler_by_prompt is None:
            raise ValueError(
                "teacher_pipeline_phase requires per-sample CoT (set teacher_cot_json_path)"
            )
        if config.teacher_cot_mode != "insert" or not config.supervise_student_cot:
            raise ValueError(
                "teacher_pipeline_phase requires teacher_cot_mode=insert + "
                "supervise_student_cot=true (the production OPD recipe)"
            )
        if not config.student_prefill_count or chat_tokenizer is None:
            raise ValueError(
                "teacher_pipeline_phase requires student_prefill_count>0 + chat_tokenizer"
            )
        forced_prefix_tokens = _forced_prefix_tokens(config, chat_tokenizer)
        logger.info(
            "Pipelined two-phase teacher prefill ON: forced-prefix K=%d tokens, "
            "Phase A (prompt+CoT+pause) prefetched 1 step ahead, Phase B (answer) "
            "radix-served at step N. Teacher MUST run with radix ENABLED.",
            len(forced_prefix_tokens),
        )

    # prompt_batches / teacher_filler_batches are (re)computed per step inside
    # the loop when windowing is on; when off, compute once here over the full
    # pool (original behavior). The step-loop closure reads whatever these
    # names hold at the time it is (re)defined each iteration.
    prompt_batches = _chunked(prompts, prepare_batch_size)
    # Per-prompt teacher CoT (Run B): slice in lockstep with prompt_batches so
    # each prepare task sees the CoT-token list aligned to its prompt slice.
    teacher_filler_batches: list[list[list[int]]] | None = None
    if teacher_filler_by_prompt is not None:
        teacher_filler_batches = _chunked(teacher_filler_by_prompt, prepare_batch_size)
        if len(teacher_filler_batches) != len(prompt_batches):
            raise RuntimeError(
                f"prompt_batches ({len(prompt_batches)}) and teacher_filler_batches "
                f"({len(teacher_filler_batches)}) misaligned"
            )

    def _window_batches(step: int) -> tuple[list[list[Any]], list[list[list[int]]] | None]:
        """Return (prompt_batches, teacher_filler_batches) for `step`'s window.

        Mirrors the in-loop windowing so the Phase A prefetcher can compute step
        N+1's window without mutating the loop's own batch variables.
        """
        if prompts_per_step > 0:
            step_prompts = _slice_window(prompts, step)
            pb = _chunked(step_prompts, prepare_batch_size)
            fb_ = None
            if teacher_filler_by_prompt is not None:
                fb_ = _chunked(_slice_window(teacher_filler_by_prompt, step), prepare_batch_size)
            return pb, fb_
        fb_ = (
            _chunked(teacher_filler_by_prompt, prepare_batch_size)
            if teacher_filler_by_prompt is not None
            else None
        )
        return _chunked(prompts, prepare_batch_size), fb_

    def _launch_phase_a(step: int) -> dict[int, "asyncio.Task[PhaseAResult]"]:
        """Kick off Phase A (prompt+CoT+pause) for every prepare batch of `step`.

        Each task runs the blocking phase-A HTTP POST in a worker thread, so they
        proceed concurrently with the current step's forward_backward. Returns a
        {batch_idx: Task} map consumed by the next step's pipelined prepare.
        """
        if step >= config.num_steps:
            return {}
        pb, fb_ = _window_batches(step)
        if fb_ is None:
            raise RuntimeError("teacher_pipeline_phase requires per-prompt CoT batches")
        tasks: dict[int, "asyncio.Task[PhaseAResult]"] = {}
        for bidx, (batch_prompts, batch_cots) in enumerate(zip(pb, fb_)):
            tasks[bidx] = asyncio.create_task(
                asyncio.to_thread(
                    _phase_a_prepare,
                    config,
                    config.teacher_base_url,
                    batch_prompts,
                    batch_cots,
                    forced_prefix_tokens,
                    output_dir,
                    step,
                    bidx,
                    chat_tokenizer,
                    teacher_prefix_tokens,
                )
            )
        return tasks

    async def _prepare_step_batches(step: int) -> list[PreparedOpdBatch]:
        """Run ALL prepare batches for `step` and return them in batch order.

        Used only by the async-overlapped (opd_pipeline_rl) path. This is the
        single-phase prepare (student sampling + single-phase teacher prefill via
        ``_prepare_opd_batch`` -> ``_teacher_cache_from_sglang`` -> merged teacher
        cache + ``_opd_loss_data``). It is the SAME work the serial path does in
        ``_prepare_opd_batch``; the only difference is that here it is run as a
        standalone coroutine (launched one step ahead) and does NOT submit
        forward_backward — fb submission stays in the main loop so a step trains
        on exactly the batches prepared for it.

        Batch order is preserved (indexed insert) so prepared_batches[i] is the
        prepare for prompt window batch i — the teacher targets, student-sampled
        sequences, and _opd_loss_data inside each PreparedOpdBatch all come from
        the SAME _prepare_opd_batch call, i.e. the SAME prompts and the SAME
        sampled sequences.
        """
        step_prompt_batches, step_filler_batches = _window_batches(step)
        results: list[PreparedOpdBatch | None] = [None] * len(step_prompt_batches)
        next_idx = 0
        inflight: dict[asyncio.Task[PreparedOpdBatch], int] = {}

        def _schedule() -> None:
            nonlocal next_idx
            while next_idx < len(step_prompt_batches) and len(inflight) < prepare_concurrency:
                if step_filler_batches is not None:
                    filler_for_batch: list[int] | list[list[int]] = step_filler_batches[next_idx]
                else:
                    filler_for_batch = teacher_filler_tokens
                task = asyncio.create_task(
                    _prepare_opd_batch(
                        config,
                        sampling_clients,
                        config.teacher_base_url,
                        step_prompt_batches[next_idx],
                        output_dir,
                        step,
                        chat_tokenizer,
                        next_idx,
                        teacher_prefix_tokens,
                        filler_for_batch,
                    )
                )
                inflight[task] = next_idx
                next_idx += 1

        _schedule()
        while inflight:
            done, _ = await asyncio.wait(
                inflight.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                bidx = inflight.pop(task)
                results[bidx] = task.result()
                _schedule()
        return [r for r in results if r is not None]

    rows: list[dict[str, Any]] = []
    empty_streak = 0  # consecutive steps with empty_frac >= abort threshold
    # Async-overlapped sampling (opd_pipeline_rl) one-step-ahead prepare handle.
    # Holds the background task that prepares step N+1 while step N trains+syncs;
    # step N consumes the task launched during step N-1. Empty until the first step
    # launches it (step 0 prepares synchronously, mirroring filler_tokens_rl which
    # only starts its generation worker after the first step). With
    # opd_pipeline_depth>1 this holds up to `depth` step-keyed tasks (N+1..N+depth)
    # so the teacher never starves waiting for the next step's sampling.
    pipeline_depth = max(1, int(getattr(config, "opd_pipeline_depth", 1)))
    if pipeline_depth > 1 and config.sampler_quiesce_before_sync:
        raise ValueError(
            "opd_pipeline_depth>1 is incompatible with sampler_quiesce_before_sync "
            "(deeper lookahead keeps next-step requests in flight during sync)"
        )
    pending_prepares: dict[int, "asyncio.Task[list[PreparedOpdBatch]]"] = {}
    # Background eval bundle (eval_async + dedicated eval pool). At most one in
    # flight; it runs against the eval pool's frozen weights (the pool is only
    # re-synced at eval steps, after draining any previous bundle). Results are
    # appended to the profile as a late row_kind="eval_async" row stamped with
    # the step they evaluate.
    pending_eval: "asyncio.Task[dict[str, Any]] | None" = None
    pending_eval_step: int = -1
    pending_eval_t0: float = 0.0
    # One-step-ahead Phase A prefetch (pipelined path only). Issue step 0's Phase A
    # before the loop; thereafter step N's Phase A is launched during step N-1's
    # forward_backward. Keyed by prepare-batch index.
    phase_a_tasks: dict[int, "asyncio.Task[PhaseAResult]"] = (
        _launch_phase_a(0) if pipeline_phase else {}
    )
    # Fire-and-forget HF-safetensors export futures (see config.save_hf_safetensors);
    # drained after the loop so the process doesn't exit before they finish.
    hf_save_futures: list[tuple[str, Any]] = []
    fb_capture_done = False
    fb_capture_path = (
        Path(config.forward_backward_capture_path)
        if config.forward_backward_capture_path
        else None
    )
    if fb_capture_path is not None:
        logger.info(
            "Will capture forward_backward replay payload at step=%d train_batch=%d to %s",
            config.forward_backward_capture_step,
            config.forward_backward_capture_train_batch,
            fb_capture_path,
        )
    # Held-out set for the with/without-pause control eval (genuine-improvement
    # judge). From eval_prompts_json_path if given, else the tail of the pool.
    control_eval_prompts: list[Any] = []
    if config.eval_accuracy_every:
        if config.eval_prompts_json_path:
            control_eval_prompts = json.loads(
                Path(config.eval_prompts_json_path).read_text()
            )[: config.eval_num_problems]
        elif len(prompts) > config.eval_num_problems:
            control_eval_prompts = list(prompts[-config.eval_num_problems:])
        else:
            control_eval_prompts = list(prompts)
        logger.info(
            "Control eval (with/without pause) every %d steps on %d held-out prompts",
            config.eval_accuracy_every, len(control_eval_prompts),
        )
    for step in range(config.num_steps):
        logger.info("=== OPD step %s ===", step)
        step_t0 = time.perf_counter()
        sampler_metrics_before = (
            _fetch_sampler_metrics(sampler_metrics_urls, config.sampler_metrics_timeout)
            if sampler_metrics_urls
            else None
        )

        # Re-slice the per-step prompt window (no-op when prompts_per_step==0).
        if prompts_per_step > 0:
            step_prompts = _slice_window(prompts, step)
            prompt_batches = _chunked(step_prompts, prepare_batch_size)
            if teacher_filler_by_prompt is not None:
                step_filler = _slice_window(teacher_filler_by_prompt, step)
                teacher_filler_batches = _chunked(step_filler, prepare_batch_size)
                if len(teacher_filler_batches) != len(prompt_batches):
                    raise RuntimeError(
                        f"step {step}: prompt_batches ({len(prompt_batches)}) and "
                        f"teacher_filler_batches ({len(teacher_filler_batches)}) misaligned"
                    )
            window_start = (step * prompts_per_step) % len(prompts)
            logger.info(
                "step %d prompt window: [%d:%d) of %d (%d prompts)",
                step,
                window_start,
                window_start + prompts_per_step,
                len(prompts),
                len(step_prompts),
            )

        prepare_window_t0 = time.perf_counter()
        prepared_batches: list[PreparedOpdBatch] = []
        fb_futures: list[Any] = []
        first_fb_submit_t0: float | None = None
        train_microbatch_count = 0

        # Pipelined path: collect this step's prefetched Phase A results (issued
        # during the previous step's fwd_bwd). They are already in-flight; await as
        # the pipelined prepare consumes them.
        step_phase_a: dict[int, PhaseAResult] = {}
        if pipeline_phase:
            for bidx, task in phase_a_tasks.items():
                step_phase_a[bidx] = await task
            next_step_phase_a_launched = False

        next_prepare_idx = 0
        completed_prepare_count = 0
        inflight_prepare: dict[asyncio.Task[PreparedOpdBatch], int] = {}

        # Submit the forward_backward microbatches for one completed prepare batch.
        # Shared by the serial (interleaved) and pipelined paths so a step always
        # trains on exactly the batches prepared for it. `is_last` marks the final
        # prepare batch of the step (used for skip_optim_step gradient-clear).
        def _submit_fb_for_prepared(
            prepared: PreparedOpdBatch,
            *,
            prepare_batch_idx: int,
            is_last: bool,
        ) -> None:
            nonlocal first_fb_submit_t0, train_microbatch_count, fb_capture_done
            loss_params: dict[str, Any] = {
                "teacher_heads": {"0": config.teacher_head},
                "teacher_hidden_caches": {"0": str(prepared.cache_path)},
                "opd_sort_by_teacher": True,
                "opd_kl_backend": config.opd_kl_backend,
                "opd_vocab_chunk_size": config.opd_vocab_chunk_size,
                "opd_sharded_head_device_cache": config.opd_sharded_head_device_cache,
                "opd_loss_mode": config.opd_loss_mode,
                "opd_emit_full_vocab_diagnostics": config.opd_emit_full_vocab_diagnostics,
                "opd_use_policy_gradient": config.opd_use_policy_gradient,
                "opd_loss_max_clamp": config.opd_loss_max_clamp,
                "opd_hidden_match_coef": config.opd_hidden_match_coef,
                "opd_kl_loss_weight": config.opd_kl_loss_weight,
                "opd_hidden_match_mode": config.opd_hidden_match_mode,
                "opd_profile_timings": True,
                "opd_profile_sync_cuda": config.profile_sync_cuda,
                "num_chunks": 8,
            }
            # Multi-layer OPRD: send the explicit layer subset (the SAME list the
            # trainer uses to capture student layers) + the spec string. By default
            # teacher per-layer hiddens come from the trainer-side no-grad forward;
            # the SGLang cache backend passes a rank-3 layer cache here instead.
            if prepared.oprd_layer_indices:
                loss_params["opd_oprd_enabled"] = True
                loss_params["opd_oprd_layers"] = config.opd_oprd_layers
                loss_params["opd_oprd_layer_indices"] = list(prepared.oprd_layer_indices)
                loss_params["opd_oprd_last_k"] = int(config.opd_oprd_last_k)
                loss_params["opd_oprd_student_capture"] = config.opd_oprd_student_capture
                if prepared.layers_cache_path is not None:
                    loss_params["teacher_layer_hidden_caches"] = {
                        "0": {
                            "path": str(prepared.layers_cache_path),
                            "tensor_key": "hidden_states_layers",
                        }
                    }
            train_batches = _chunked(prepared.data, train_microbatch_size)
            for train_idx, train_batch in enumerate(train_batches):
                train_loss_params = dict(loss_params)
                is_last_train_batch = is_last and train_idx == len(train_batches) - 1
                if config.skip_optim_step and is_last_train_batch:
                    train_loss_params["profile_clear_gradients_after_backward"] = True
                if first_fb_submit_t0 is None:
                    first_fb_submit_t0 = time.perf_counter()
                loss_fn = "cross_entropy" if config.sft_mode else "opd_loss"
                request_loss_params = {} if config.sft_mode else train_loss_params
                if (
                    fb_capture_path is not None
                    and not fb_capture_done
                    and step == int(config.forward_backward_capture_step)
                    and train_idx == int(config.forward_backward_capture_train_batch)
                ):
                    _write_forward_backward_capture(
                        capture_path=fb_capture_path,
                        config=config,
                        step=step,
                        prepare_batch_idx=prepare_batch_idx,
                        train_batch_idx=train_idx,
                        train_microbatch_size=train_microbatch_size,
                        data=train_batch,
                        loss_fn=loss_fn,
                        loss_fn_params=request_loss_params,
                        prepared=prepared,
                    )
                    fb_capture_done = True
                    logger.info("Captured forward_backward replay payload to %s", fb_capture_path)
                fb_futures.append(
                    training_client.forward_backward(
                        train_batch,
                        loss_fn=loss_fn,
                        loss_fn_params=request_loss_params,
                    )
                )
                train_microbatch_count += 1

        if pipeline_rl:
            # ── Async-overlapped sampling (pipeline RL) ──
            # Step N consumes the prepare launched during step N-1 (samples are
            # 1-step-stale). Step 0 has no predecessor, so it prepares synchronously
            # — exactly like filler_tokens_rl starts its gen worker only after the
            # first step (filler_tokens_rl.py:2904-2920). The N+1 prepare is launched
            # below, AFTER fb submission, so it overlaps train+sync.
            pending = pending_prepares.pop(step, None)
            if pending is None:
                prepared_batches = await _prepare_step_batches(step)
            else:
                prepared_batches = await pending
            for bidx, prepared in enumerate(prepared_batches):
                _submit_fb_for_prepared(
                    prepared,
                    prepare_batch_idx=bidx,
                    is_last=(bidx == len(prepared_batches) - 1),
                )
        else:
            def schedule_prepare_batches() -> None:
                nonlocal next_prepare_idx
                while (
                    next_prepare_idx < len(prompt_batches)
                    and len(inflight_prepare) < prepare_concurrency
                ):
                    if teacher_filler_batches is not None:
                        filler_for_batch: list[int] | list[list[int]] = teacher_filler_batches[
                            next_prepare_idx
                        ]
                    else:
                        filler_for_batch = teacher_filler_tokens
                    if pipeline_phase:
                        phase_a = step_phase_a.get(next_prepare_idx)
                        if phase_a is None:
                            raise RuntimeError(
                                f"step {step}: missing prefetched Phase A for batch {next_prepare_idx}"
                            )
                        task = asyncio.create_task(
                            _prepare_opd_batch_pipelined(
                                config,
                                sampling_clients,
                                config.teacher_base_url,
                                prompt_batches[next_prepare_idx],
                                output_dir,
                                step,
                                chat_tokenizer,
                                next_prepare_idx,
                                teacher_prefix_tokens,
                                filler_for_batch,  # per-prompt CoT for this batch
                                phase_a,
                            )
                        )
                    else:
                        task = asyncio.create_task(
                            _prepare_opd_batch(
                                config,
                                sampling_clients,
                                config.teacher_base_url,
                                prompt_batches[next_prepare_idx],
                                output_dir,
                                step,
                                chat_tokenizer,
                                next_prepare_idx,
                                teacher_prefix_tokens,
                                filler_for_batch,
                            )
                        )
                    inflight_prepare[task] = next_prepare_idx
                    next_prepare_idx += 1

            schedule_prepare_batches()
            while inflight_prepare:
                done, _ = await asyncio.wait(
                    inflight_prepare.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    bidx = inflight_prepare.pop(task)
                    prepared = task.result()
                    completed_prepare_count += 1
                    schedule_prepare_batches()

                    prepared_batches.append(prepared)
                    _submit_fb_for_prepared(
                        prepared,
                        prepare_batch_idx=bidx,
                        is_last=(completed_prepare_count == len(prompt_batches)),
                    )
        prepare_window_s = _elapsed(prepare_window_t0)

        optim_future = None
        optim_t0 = None
        if not config.skip_optim_step:
            optim_t0 = time.perf_counter()
            optim_future = training_client.optim_step(
                tomi.AdamParams(
                    learning_rate=config.learning_rate,
                    grad_clip_norm=config.grad_clip_norm,
                )
            )

        # Prefetch step N+1's Phase A NOW so it overlaps step N's forward_backward.
        # Phase A is the fixed prompt+CoT+pause prefill (no student weights, no
        # sampled answer) → prefetching it one step early introduces zero staleness.
        if pipeline_phase and not next_step_phase_a_launched:
            phase_a_tasks = _launch_phase_a(step + 1)
            next_step_phase_a_launched = True

        # Async-overlapped sampling: launch step N+1's PREPARE now, AFTER step N's
        # forward_backward + optim_step have been submitted to the trainer. It runs
        # as a background asyncio task that overlaps the awaits below (fb + optim +
        # sync). At launch the inference server still holds the weights synced at
        # the END of step N-1 (step N's sync happens below), so N+1's samples are
        # 1-step-stale — the accepted pipeline-RL staleness (mirrors
        # filler_tokens_rl.py:2880-2920: "generation thread is already working on
        # next batch with stale weights", sync happens after training). Step 0
        # already prepared synchronously above; this seeds the pipeline so step 1
        # onward consumes pending_prepare. Not launched on the final step (nothing
        # to consume it).
        launch_pending_prepare_after_sync = False
        if pipeline_rl and step + 1 < config.num_steps:
            if config.sampler_quiesce_before_sync:
                # Fresh-sampler science cannot have step N+1 requests in flight
                # while step N syncs. Launch the overlap after sync instead.
                # (pipeline_depth>1 is rejected with quiesce — see the depth guard.)
                launch_pending_prepare_after_sync = True
            else:
                # Keep the lookahead window full: launch every step in
                # [step+1, step+pipeline_depth] not already in flight. At depth=1 this
                # is exactly the old single-step launch; deeper keeps the teacher fed.
                for s in range(step + 1, min(step + pipeline_depth + 1, config.num_steps)):
                    if s not in pending_prepares:
                        pending_prepares[s] = asyncio.create_task(_prepare_step_batches(s))

        fb_t0 = first_fb_submit_t0 or time.perf_counter()
        fb_results = await asyncio.gather(*fb_futures)
        fb_s = _elapsed(fb_t0)

        # Delete the teacher hidden-state caches now that forward_backward has
        # consumed them — they're ~136 MB each (kept positions, per prepare
        # batch) and the trainer never re-reads them, so leaving them would pile
        # up ~50 GB/run of dead files on shared storage. Everything downstream
        # (metrics, eval) uses in-memory batch fields, not the cache file.
        for _b in prepared_batches:
            try:
                Path(_b.cache_path).unlink(missing_ok=True)
            except Exception as _exc:  # noqa: BLE001
                logger.debug("teacher cache cleanup skipped for %s: %s", _b.cache_path, _exc)
            # Multi-layer OPRD: drop the rank-3 layer cache too (off => no-op).
            if _b.layers_cache_path is not None:
                try:
                    Path(_b.layers_cache_path).unlink(missing_ok=True)
                except Exception as _exc:  # noqa: BLE001
                    logger.debug(
                        "teacher layer cache cleanup skipped for %s: %s",
                        _b.layers_cache_path,
                        _exc,
                    )

        optim_s = 0.0
        optim_queued_s = 0.0
        if optim_future is not None:
            optim_wait_t0 = time.perf_counter()
            await optim_future
            optim_s = _elapsed(optim_wait_t0)
            optim_queued_s = _elapsed(optim_t0 or time.perf_counter())

        sync_s = 0.0
        sync_result = None
        sync_failure: str | None = None
        sync_exception: BaseException | None = None
        sampler_quiesce_metrics: dict[str, float] = {"sampler_quiesce_enabled": 0.0}
        if config.sync_weights and not config.skip_optim_step:
            if config.sampler_quiesce_before_sync:
                logger.info("Waiting for sampler quiescence before weight sync at step %s", step)
                sampler_quiesce_metrics, _ = _wait_for_sampler_quiescence(
                    sampler_metrics_urls,
                    sampler_metrics_before,
                    metrics_timeout=config.sampler_metrics_timeout,
                    timeout_s=config.sampler_quiesce_timeout,
                    poll_s=config.sampler_quiesce_poll_s,
                    max_new_outstanding=config.sampler_quiesce_max_new_outstanding,
                    max_connections_active=config.sampler_quiesce_max_connections_active,
                    max_inflight=config.sampler_quiesce_max_inflight,
                )
                logger.info(
                    "Sampler quiescence step %s: success=%s new_outstanding=%.0f "
                    "connections=%.0f inflight_age_count=%.0f wait=%.1fs polls=%.0f",
                    step,
                    bool(sampler_quiesce_metrics.get("sampler_quiesce_success", 0.0)),
                    sampler_quiesce_metrics.get("sampler_quiesce_new_outstanding", 0.0),
                    sampler_quiesce_metrics.get("sampler_quiesce_connections_active", 0.0),
                    sampler_quiesce_metrics.get("sampler_quiesce_inflight_request_age_count", 0.0),
                    sampler_quiesce_metrics.get("sampler_quiesce_wait_s", 0.0),
                    sampler_quiesce_metrics.get("sampler_quiesce_poll_count", 0.0),
                )
                if sampler_quiesce_metrics.get("sampler_quiesce_success", 0.0) < 1.0:
                    sync_failure = (
                        "sampler quiescence failed before weight sync: "
                        f"new_outstanding={sampler_quiesce_metrics.get('sampler_quiesce_new_outstanding', 0.0):.0f}, "
                        f"connections_active={sampler_quiesce_metrics.get('sampler_quiesce_connections_active', 0.0):.0f}, "
                        "inflight_age_count="
                        f"{sampler_quiesce_metrics.get('sampler_quiesce_inflight_request_age_count', 0.0):.0f}"
                    )
                    sync_exception = RuntimeError(sync_failure)
                sync_result = SimpleNamespace(
                    success=False,
                    message=sync_failure,
                    transfer_time=0.0,
                    total_bytes=0,
                    num_buckets=0,
                    endpoints_synced=[],
                    timing_breakdown={},
                    p2p_rank_summaries=[],
                )
            if sync_failure is None:
                sync_t0 = time.perf_counter()
                try:
                    sync_result = await asyncio.wait_for(
                        training_client.sync_weights_to_inference(
                            sync_method=config.sync_method,
                            master_address=_weight_sync_master_address(config),
                            timeout=config.weight_sync_timeout,
                            # With a dedicated eval pool, the per-step sync covers
                            # only the training samplers; the eval pool stays on
                            # frozen weights until the eval-step refresh.
                            pools=train_sync_pools,
                        ),
                        timeout=max(float(config.weight_sync_timeout), 1.0) + 30.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    sync_exception = exc
                    sync_failure = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                    logger.exception("Weight sync failed at step %s", step)
                    sync_result = SimpleNamespace(
                        success=False,
                        message=sync_failure,
                        transfer_time=0.0,
                        total_bytes=0,
                        num_buckets=0,
                        endpoints_synced=[],
                        timing_breakdown={},
                        p2p_rank_summaries=[],
                    )
                sync_s = _elapsed(sync_t0)
                if not sync_result.success:
                    sync_failure = sync_result.message or "sync_weights_to_inference failed"

        if launch_pending_prepare_after_sync and sync_failure is None:
            pending_prepares[step + 1] = asyncio.create_task(_prepare_step_batches(step + 1))

        save_s = 0.0
        save_path: str | None = None
        if (
            config.save_every > 0
            and not config.skip_optim_step
            and (step + 1) % config.save_every == 0
        ):
            save_name = f"{config.save_name_prefix}-step{step + 1}"
            save_t0 = time.perf_counter()
            logger.info("Saving checkpoint %s at step %d ...", save_name, step + 1)
            try:
                save_future = training_client.save_state(save_name)
                save_response = await save_future
                save_path = getattr(save_response, "path", None) or str(save_response)
                save_s = _elapsed(save_t0)
                logger.info(
                    "Saved checkpoint %s in %.1fs -> %s",
                    save_name,
                    save_s,
                    save_path,
                )
                if config.save_hf_safetensors:
                    # Fire-and-forget: save_weights_for_sampler writes
                    # SGLang-loadable safetensors (the returned model_path can be
                    # used directly as an sglang --model-path / SamplingClient
                    # source). It's the lightweight per-batch sampler-export path
                    # (NOT save_full_weights_safetensors, which this server build
                    # does not expose — 404). Returns an APIFuture immediately
                    # and runs on the holder's background loop; not awaited here,
                    # drained at end-of-run.
                    try:
                        hf_future = training_client.save_weights_for_sampler(save_name)
                        hf_save_futures.append((save_name, hf_future))
                        logger.info(
                            "Fired async sampler-weights export %s (non-blocking; "
                            "directly SGLang-loadable)",
                            save_name,
                        )
                    except Exception as hf_exc:  # noqa: BLE001
                        logger.error("Failed to fire sampler-weights export %s: %s", save_name, hf_exc)
            except Exception as exc:  # noqa: BLE001
                save_s = _elapsed(save_t0)
                logger.error("Checkpoint save %s failed after %.1fs: %s", save_name, save_s, exc)
                # Don't kill the training run on a save failure — the trainer
                # engine continues; the user can retry the save manually.
                save_path = f"FAILED: {exc}"

        valid_tokens = sum(_valid_tokens(result) for result in fb_results)
        # In-loop generation eval: health (empty/EOS rate, completion length,
        # </think> + digit presence) + task accuracy, computed from the
        # on-policy student completions we already have. This is the REAL
        # success signal — loss can collapse to ~0 while the model emits nothing.
        eval_metrics: dict[str, float] = {}
        eval_rows: list[dict[str, Any]] = []
        if config.eval_health_every and (step % config.eval_health_every == 0):
            all_completions: list[list[int]] = []
            all_prompt_texts: list[str] = []
            for b in prepared_batches:
                all_completions.extend(b.completions)
                all_prompt_texts.extend(b.prompt_texts)
            if all_completions:
                filler_marker = config.student_prefill_text * max(1, config.student_prefill_count)
                exact_prefill_ids = _parse_token_id_list(config.student_prefill_token_ids)
                if exact_prefill_ids and chat_tokenizer is not None and hasattr(chat_tokenizer, "decode"):
                    filler_marker = chat_tokenizer.decode(exact_prefill_ids, skip_special_tokens=False)
                eval_metrics, eval_rows = _generation_health(
                    all_completions,
                    all_prompt_texts,
                    chat_tokenizer,
                    config.eval_task,
                    max_new_tokens=config.max_new_tokens,
                    filler_marker=filler_marker,
                    answer_cue_marker=config.student_prefill_suffix,
                    stop_sequences=student_stop_sequences,
                )

        # Which evals are due this step. The three sub-evals keep their historical
        # cadences/conditions; they now run through one bundle so the whole set can
        # execute synchronously (default) or as a background task against the
        # dedicated frozen-weight eval pool (eval_async).
        control_due = bool(
            config.eval_accuracy_every
            and control_eval_prompts
            and step >= max(0, int(config.eval_control_start_step))
            and (step % config.eval_accuracy_every == 0)
            and not sync_failure
        )
        heldout_due = bool(
            config.eval_accuracy_every
            and control_eval_prompts
            and config.eval_heldout_num_problems > 0
            and (step % config.eval_accuracy_every == 0)
            and not sync_failure
        )
        answer_logprob_due = bool(
            config.eval_answer_logprob_every
            and control_eval_prompts
            and chat_tokenizer is not None
            and (step % config.eval_answer_logprob_every == 0)
            and not sync_failure
        )

        async def _run_eval_bundle(
            eval_step: int, *, run_control: bool, run_heldout: bool, run_answer_logprob: bool
        ) -> dict[str, Any]:
            bundle_metrics: dict[str, Any] = {}
            eval_clients = eval_sampling_clients or sampling_clients
            # With/without-pause control eval: the genuine-improvement judge. Extra
            # greedy sampling on a fixed held-out set under pause vs no-pause prefills.
            if run_control:

                def _log_control_progress(progress_metrics: dict[str, float]) -> None:
                    phase = (
                        "answer_logprob"
                        if progress_metrics.get("eval/answer_logprob_progress_active", 0.0)
                        else "sample"
                    )
                    completed = progress_metrics.get(
                        "eval/answer_logprob_progress_completed_requests",
                        progress_metrics.get("eval/control_progress_completed_requests", 0.0),
                    )
                    total = progress_metrics.get(
                        "eval/answer_logprob_progress_total_requests",
                        progress_metrics.get("eval/control_progress_total_requests", 0.0),
                    )
                    frac = progress_metrics.get(
                        "eval/answer_logprob_progress_completion_frac",
                        progress_metrics.get("eval/control_progress_completion_frac", 0.0),
                    )
                    elapsed_s = progress_metrics.get(
                        "eval/answer_logprob_progress_elapsed_s",
                        progress_metrics.get("eval/control_progress_elapsed_s", 0.0),
                    )
                    logger.info(
                        "[control step %d] %s progress %.0f/%.0f (%.1f%%) elapsed=%.1fs",
                        eval_step,
                        phase,
                        completed,
                        total,
                        100.0 * frac,
                        elapsed_s,
                    )
                    if wandb_run is not None and eval_step == step:
                        wandb_run.log(progress_metrics, step=eval_step)

                ctrl_metrics, _ctrl_rows = await _buffer_control_eval(
                    config,
                    eval_clients,
                    control_eval_prompts,
                    chat_tokenizer,
                    logprob_clients=logprob_clients if logprob_clients else None,
                    progress_callback=_log_control_progress,
                )
                bundle_metrics.update(ctrl_metrics)
                logger.info(
                    "[control step %d] acc_pause=%.3f acc_nopause=%.3f buffer_delta=%+.3f (n=%d)",
                    eval_step, ctrl_metrics["eval/acc_pause"], ctrl_metrics["eval/acc_nopause"],
                    ctrl_metrics["eval/buffer_delta"], int(ctrl_metrics["eval/control_n"]),
                )
            # REAL per-step eval: greedy decode on the held-out tail. This is what
            # eval/accuracy means (2026-06-10); the scored training samples are
            # eval/train_window_accuracy (health only).
            if run_heldout:
                heldout_metrics = await _heldout_greedy_eval(
                    config, eval_clients, control_eval_prompts, chat_tokenizer
                )
                bundle_metrics.update(heldout_metrics)
                if "eval/accuracy" in heldout_metrics:
                    logger.info(
                        "OPD step %s HELD-OUT eval: accuracy=%.3f (n=%d, greedy)",
                        eval_step,
                        heldout_metrics["eval/accuracy"],
                        int(heldout_metrics.get("eval/heldout_scored", 0)),
                    )
            # Periodic gold-answer logprob scoring on the held-out set, independent of
            # the end-gated control eval. Prefill-only (no generation), so it tracks
            # whether probability mass is moving toward/away from the gold answer all
            # run — the decisive signal when eval accuracy peaks then decays.
            if run_answer_logprob:
                answer_logprob_metrics = await _answer_logprob_control_eval(
                    config,
                    logprob_clients if logprob_clients else eval_clients,
                    control_eval_prompts,
                    chat_tokenizer,
                    _static_control_arms(config, chat_tokenizer),
                    force=True,
                )
                bundle_metrics.update(answer_logprob_metrics)
                if answer_logprob_metrics.get("eval/answer_logprob_control_available"):
                    logger.info(
                        "[answer-logprob step %d] %s",
                        eval_step,
                        " ".join(
                            f"{k.removeprefix('eval/answer_logprob_')}={v:.4f}"
                            for k, v in sorted(answer_logprob_metrics.items())
                            if k.startswith("eval/answer_logprob_mean_")
                            or k.startswith("eval/answer_logprob_select_margin_")
                        ),
                    )
            return bundle_metrics

        async def _finalize_pending_eval() -> None:
            nonlocal pending_eval
            if pending_eval is None:
                return
            task, eval_step, t0 = pending_eval, pending_eval_step, pending_eval_t0
            pending_eval = None
            try:
                bundle_metrics = await task
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Async eval for step %s failed", eval_step)
                bundle_metrics = {"eval/async_failed": 1.0}
            late_row = {
                "step": eval_step,
                "row_kind": "eval_async",
                "eval_async_wall_s": _elapsed(t0),
                **bundle_metrics,
            }
            with profile_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(late_row, sort_keys=True) + "\n")
            logger.info("OPD step %s async-eval profile: %s", eval_step, json.dumps(late_row, sort_keys=True))
            if wandb_run is not None:
                # Out-of-order wrt the monotonic step axis — log under the
                # eval_async/* namespace keyed by eval_async/for_step instead
                # (define_metric at init); the jsonl row is authoritative.
                wandb_run.log(
                    {
                        "eval_async/for_step": eval_step,
                        **{
                            f"eval_async/{k.removeprefix('eval/')}": v
                            for k, v in bundle_metrics.items()
                            if isinstance(v, numbers.Number) and not isinstance(v, bool)
                        },
                    }
                )

        any_eval_due = control_due or heldout_due or answer_logprob_due
        if any_eval_due and eval_sampling_clients:
            # Drain the previous bundle BEFORE refreshing the eval pool — its
            # weights must stay frozen while a bundle is still reading them.
            if pending_eval is not None:
                if not pending_eval.done():
                    logger.warning(
                        "Async eval for step %s still running at step %s; awaiting it before the eval-pool refresh "
                        "(eval_async_overrun — consider more eval replicas or a coarser eval cadence)",
                        pending_eval_step,
                        step,
                    )
                    eval_metrics["eval/async_overrun"] = 1.0
                await _finalize_pending_eval()
            eval_pool_sync_t0 = time.perf_counter()
            try:
                # NB: deliberately NO group_name override. The pools filter alone
                # selects the eval endpoints; using the default weight_sync_group
                # rides the battle-tested receiver-side cold-prepare/invalidate
                # path across trainer restarts. A custom "weight_sync_group_eval"
                # left stale receiver session state when the trainer changed and
                # wedged the next transfer for the full timeout (2026-06-11,
                # ARITH-005 step 0: 900 s hang, then completing the half-armed
                # group crashed both eval pods). Timeout capped: a wedged eval
                # refresh should cost minutes, not weight_sync_timeout — the
                # evals are skipped and training continues.
                eval_pool_sync_timeout = min(float(config.weight_sync_timeout), 180.0)
                eval_pool_sync = await asyncio.wait_for(
                    training_client.sync_weights_to_inference(
                        sync_method=config.sync_method,
                        master_address=_weight_sync_master_address(config),
                        timeout=eval_pool_sync_timeout,
                        pools=["eval"],
                    ),
                    timeout=eval_pool_sync_timeout + 30.0,
                )
                eval_pool_sync_ok = bool(eval_pool_sync.success)
            except Exception:  # noqa: BLE001
                logger.exception("Eval-pool weight sync failed at step %s", step)
                eval_pool_sync_ok = False
            if not eval_pool_sync_ok:
                # A failed/timed-out sync leaves the eval pods PAUSED (pause_mode
                # retract fires before the transfer) — unpause them so the next
                # eval cycle can try again instead of every subsequent sync
                # failing against paused receivers (observed 2026-06-12 step 30).
                # Never POST /complete_weights_update here: completing a
                # half-armed group crashes the receiver.
                for eval_url in eval_urls:
                    try:
                        await asyncio.to_thread(
                            requests.post, f"{eval_url}/continue_generation", json={}, timeout=10
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning("continue_generation failed for %s", eval_url)
            eval_metrics["eval/pool_sync_s"] = _elapsed(eval_pool_sync_t0)
            eval_metrics["eval/pool_sync_success"] = float(eval_pool_sync_ok)
            if not eval_pool_sync_ok:
                if control_due:
                    # The verdict-bearing control eval cannot run on stale weights.
                    raise RuntimeError(f"OPD aborting at step {step}: eval-pool weight sync failed")
                logger.error("Skipping step %s evals: eval-pool sync failed (training continues)", step)
                any_eval_due = control_due = heldout_due = answer_logprob_due = False
        if any_eval_due:
            # The control eval is verdict-bearing (the autopilot waits on the final
            # control row), so it always runs blocking; only the periodic
            # held-out/logprob bundles go async.
            if config.eval_async and eval_sampling_clients and not control_due:
                pending_eval = asyncio.create_task(
                    _run_eval_bundle(
                        step,
                        run_control=False,
                        run_heldout=heldout_due,
                        run_answer_logprob=answer_logprob_due,
                    )
                )
                pending_eval_step = step
                pending_eval_t0 = time.perf_counter()
            else:
                eval_metrics.update(
                    await _run_eval_bundle(
                        step,
                        run_control=control_due,
                        run_heldout=heldout_due,
                        run_answer_logprob=answer_logprob_due,
                    )
                )
        sampler_metrics_after = (
            _fetch_sampler_metrics(sampler_metrics_urls, config.sampler_metrics_timeout)
            if sampler_metrics_urls
            else None
        )

        prepared_metrics = _aggregate_prepared_metrics(prepared_batches)
        row: dict[str, Any] = {
            "step": step,
            "profile_warmup": step < config.profile_warmup_steps,
            "eval/control_start_step": float(max(0, int(config.eval_control_start_step))),
            "eval/control_allowed_by_start_step": float(step >= max(0, int(config.eval_control_start_step))),
            "step_total_s": _elapsed(step_t0),
            "opd_microbatch_size": config.opd_microbatch_size,
            "opd_prepare_batch_size": prepare_batch_size,
            "opd_prepare_concurrency": prepare_concurrency,
            "opd_train_microbatch_size": train_microbatch_size,
            "num_microbatches": train_microbatch_count,
            # 1.0 once the pipeline is warm (step consumed an overlapped prepare);
            # 0.0 on the synchronous step-0 prepare or when opd_pipeline_rl is off.
            "pipeline_rl_active": float(pipeline_rl and step > 0),
            "prepare_window_s": prepare_window_s,
            "forward_backward_s": fb_s,
            "optim_step_s": optim_s,
            "optim_step_queued_s": optim_queued_s,
            "sync_inference_weights_s": sync_s,
            "save_checkpoint_s": save_s,
            "save_checkpoint_path": save_path,
            "loss": _loss_mean_many(fb_results),
            "valid_tokens": valid_tokens,
            **prepared_metrics,
            **_aggregate_forward_backward_profile_metrics(fb_results),
            **_aggregate_opd_loss_metrics(fb_results),
            **_full_vocab_diagnostic_metrics(config),
            **eval_metrics,
            **sampler_quiesce_metrics,
            **_sampler_metrics_delta(sampler_metrics_before, sampler_metrics_after),
        }
        if row["step_total_s"] > 0:
            row["valid_tokens_per_step_s"] = valid_tokens / row["step_total_s"]
        if prepare_window_s > 0:
            row["student_sampling_output_tok_per_prepare_window_s"] = (
                row["student_sampling_output_tokens"] / prepare_window_s
            )
            row["teacher_prefill_tok_per_prepare_window_s"] = (
                row["teacher_prefill_tokens"] / prepare_window_s
            )
        if sync_result is not None:
            row.update(
                {
                    "sync_success": sync_result.success,
                    "sync_message": sync_result.message,
                    "sync_transfer_time_s": sync_result.transfer_time,
                    "sync_total_bytes": sync_result.total_bytes,
                    "sync_num_buckets": sync_result.num_buckets,
                }
            )
            row.update(_sync_profile_metrics(sync_result))
        if sync_failure:
            row["sync_failure"] = sync_failure
        rows.append(row)
        with profile_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        # Persist EVERY decoded eval sample (not just the n_show shown in wandb/logs)
        # so per-digit error decomposition over training is possible offline.
        if eval_rows:
            eval_samples_path = profile_path.with_name("eval_samples.jsonl")
            with eval_samples_path.open("a", encoding="utf-8") as handle:
                for sample_row in eval_rows:
                    handle.write(json.dumps({"step": step, **sample_row}, sort_keys=True) + "\n")
        logger.info("OPD step %s profile: %s", step, json.dumps(row, sort_keys=True))
        # In-loop generation summary — the operator-facing success signal.
        if eval_metrics:
            logger.info(
                "OPD step %s eval: accuracy=%s empty_frac=%.3f mean_completion_tokens=%.1f "
                "has_think_close_frac=%.3f has_digit_frac=%.3f (scored=%d)",
                step,
                f"{eval_metrics['eval/train_window_accuracy']:.3f}"
                if "eval/train_window_accuracy" in eval_metrics
                else "n/a",
                eval_metrics.get("eval/empty_frac", 0.0),
                eval_metrics.get("eval/mean_completion_tokens", 0.0),
                eval_metrics.get("eval/has_think_close_frac", 0.0),
                eval_metrics.get("eval/has_digit_frac", 0.0),
                int(eval_metrics.get("eval/num_scored", 0)),
            )
        # Log a few decoded samples (visible in logs + wandb table) so the actual
        # student behaviour is inspectable, not just scalars.
        n_show = int(config.eval_log_samples)
        for r in eval_rows[:n_show]:
            logger.info(
                "OPD step %s sample correct=%s len=%s | %s -> %s",
                step,
                r["correct"],
                r["len"],
                r["prompt"].replace("\n", " "),
                r["completion"].replace("\n", "\\n"),
            )
        if wandb_run is not None:
            # Log every numeric field from the profile row to wandb, keyed by step.
            wandb_metrics = {
                k: v for k, v in row.items()
                if isinstance(v, numbers.Number) and not isinstance(v, bool)
            }
            wandb_metrics["sync_success_int"] = int(bool(row.get("sync_success", True)))
            wandb_run.log(wandb_metrics, step=step)
            # Decoded-sample table so generations are visible in the wandb UI.
            if eval_rows and n_show > 0:
                try:
                    import wandb  # noqa: PLC0415

                    table = wandb.Table(columns=["step", "prompt", "completion", "len", "correct"])
                    for r in eval_rows[:n_show]:
                        table.add_data(step, r["prompt"], r["completion"], r["len"], str(r["correct"]))
                    wandb_run.log({"eval/samples": table}, step=step)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("wandb sample-table log failed at step %s: %s", step, exc)
        if sync_failure:
            raise RuntimeError(f"OPD aborting at step {step}: {sync_failure}") from sync_exception
        # Auto-abort on sustained generation collapse (the EOS reward-hack): if
        # the student emits empty completions at/above the threshold for
        # `eval_abort_patience` consecutive steps, stop — a converging loss here
        # is meaningless and the run is wasting compute.
        if config.eval_abort_patience and "eval/empty_frac" in eval_metrics:
            if eval_metrics["eval/empty_frac"] >= config.eval_abort_empty_frac:
                empty_streak += 1
            else:
                empty_streak = 0
            if empty_streak >= config.eval_abort_patience:
                raise RuntimeError(
                    f"OPD aborting at step {step}: empty_frac "
                    f"{eval_metrics['eval/empty_frac']:.2f} >= {config.eval_abort_empty_frac} "
                    f"for {empty_streak} consecutive steps — student is generating nothing "
                    f"(EOS-collapse / loss reward-hack). Fix the prompt/recipe (see runbook)."
                )
    # Cancel any orphaned lookahead prepares (only possible if the loop exits before
    # consuming them; the final step never launches past num_steps).
    for _pending in pending_prepares.values():
        if not _pending.done():
            _pending.cancel()
            try:
                await _pending
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    pending_prepares.clear()

    # Drain a still-running async eval bundle so its late row lands before the
    # steady summary (the terminal control eval is blocking, so this only fires
    # when the run's tail has no control step). _finalize_pending_eval is the
    # loop-body closure from the last iteration — valid whenever pending_eval is.
    if pending_eval is not None:
        logger.info("Draining the in-flight async eval for step %s before exit...", pending_eval_step)
        await _finalize_pending_eval()

    # Drain any in-flight HF-safetensors exports so the process doesn't exit
    # before they finish writing (they ran concurrently with training).
    if hf_save_futures:
        logger.info("Draining %d async HF-safetensors export(s)...", len(hf_save_futures))
        for name, fut in hf_save_futures:
            try:
                res = await fut
                logger.info(
                    "HF-safetensors export %s complete -> %s",
                    name,
                    getattr(res, "path", None) or str(res),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("HF-safetensors export %s failed: %s", name, exc)

    steady = [row for row in rows if not row["profile_warmup"]]
    if steady:
        steady_summary = {
            "steady/total_s": _mean([row["step_total_s"] for row in steady]),
            "steady/sample_s": _mean([row["student_sampling_s"] for row in steady]),
            "steady/teacher_s": _mean([row["teacher_prefill_s"] for row in steady]),
            "steady/fwd_bwd_s": _mean([row["forward_backward_s"] for row in steady]),
            "steady/sync_s": _mean([row["sync_inference_weights_s"] for row in steady]),
        }
        logger.info(
            "Steady OPD mean: total=%(steady/total_s).3fs sample=%(steady/sample_s).3fs "
            "teacher=%(steady/teacher_s).3fs fwd_bwd=%(steady/fwd_bwd_s).3fs sync=%(steady/sync_s).3fs",
            steady_summary,
        )
        if wandb_run is not None:
            wandb_run.summary.update(steady_summary)
    logger.info("Wrote OPD profile rows to %s", profile_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":

    def _main(config: Config) -> None:
        asyncio.run(main(config))

    chz.nested_entrypoint(_main)
