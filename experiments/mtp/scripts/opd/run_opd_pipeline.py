"""OPD full-pipeline driver: orchestrates a multi-pod cross-node validation.

This script runs *inside* the trainer pod after the xorl training server is
healthy. It:

  1. Reads student/teacher endpoint URLs from OPD_*_BASE_URLS or the shared
     coord dir fallback.
  2. Registers every concrete student SGLang endpoint for trainer weight sync.
  3. Loops: rollout from student SGLang  ->  teacher hidden states from
     teacher XORL  ->  forward_backward(opd_loss)  ->  optim_step  ->
     sync_inference_weights.
  4. Asserts each step succeeds and that rollouts diverge after weight sync.

Inputs (env vars):
  XORL_TRAIN_URL       e.g. http://127.0.0.1:6000
  XORL_TRAINER_NODE_IP advertised master_address used by sync_inference_weights
                       (defaults to $POD_IP, which equals nodeIP under hostNetwork)
  OPD_COORD_DIR        directory holding student.json / teacher.json when
                       explicit OPD_*_BASE_URLS are not set
  OPD_STUDENT_BASE_URLS comma/space separated student native route URLs
  OPD_STUDENT_SYNC_BASE_URLS concrete student endpoints to register for sync
  OPD_TEACHER_BASE_URLS comma/space separated teacher prefill URLs
  OPD_NATIVE_ROUTE_MODE smg, python_router, or direct_fanout
  OPD_NUM_STEPS        default 2
  OPD_MAX_NEW_TOKENS   default 8
  OPD_SKIP_OPTIM_STEP  write profile rows after forward/backward and skip
                       optimizer/sync; useful for throughput profiling

Exit code 0 on success, non-zero on any failure (so k8s Job can detect).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import math
import os
import queue as queue_module
import random
import shutil
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import Semaphore
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import torch
from safetensors.torch import save_file


logging.basicConfig(
    level=os.environ.get("OPD_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("opd-driver")


def _load_endpoint(coord_dir: Path, role: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Wait for and parse coord_dir/<role>.json written by a service pod."""
    path = coord_dir / f"{role}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if payload.get("ready"):
                return payload
        time.sleep(2)
    raise TimeoutError(f"Coord file for {role} did not become ready within {timeout}s: {path}")


def _normalize_base_url(base_url: str) -> str:
    value = str(base_url).strip()
    if not value:
        raise ValueError("endpoint URL must not be empty")
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def _base_url_list(raw: str | None, *, arg_name: str) -> List[str]:
    if raw is None or raw.strip() == "":
        return []
    if raw.lstrip().startswith("["):
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{arg_name} must be valid JSON or comma/space separated URLs") from exc
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"{arg_name} JSON form must decode to a list of strings")
        return [_normalize_base_url(item) for item in values]
    return [_normalize_base_url(item) for item in raw.replace(",", " ").split() if item.strip()]


def _endpoint_base_url(payload: Dict[str, Any], *, role: str) -> str:
    host = payload.get("host")
    port = payload.get("port")
    if not host or port is None:
        raise ValueError(f"{role} coord payload must contain host and port: {payload}")
    return _normalize_base_url(f"http://{host}:{int(port)}")


def _endpoint_payload_from_base_url(
    base_url: str,
    *,
    world_size: int = 8,
    master_address: str | None = None,
) -> Dict[str, Any]:
    parsed = urlparse(_normalize_base_url(base_url))
    if not parsed.hostname:
        raise ValueError(f"endpoint URL must include a host: {base_url!r}")
    if parsed.port is None:
        raise ValueError(f"endpoint URL must include an explicit port: {base_url!r}")
    payload: Dict[str, Any] = {
        "host": parsed.hostname,
        "port": int(parsed.port),
        "world_size": int(world_size),
        "sync_weights": False,
    }
    if master_address:
        payload["master_address"] = master_address
    return payload


class _EndpointUrlRouter:
    def __init__(self, urls: Sequence[str]) -> None:
        normalized = [_normalize_base_url(url) for url in urls]
        if not normalized:
            raise ValueError("Endpoint router requires at least one URL")
        self._urls = normalized
        self._lock = threading.Lock()
        self._next = 0

    @property
    def urls(self) -> List[str]:
        return list(self._urls)

    def next(self) -> str:
        with self._lock:
            url = self._urls[self._next % len(self._urls)]
            self._next += 1
            return url


def _wait_for_xorl(train_url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{train_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("engine_running"):
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"xorl training server at {train_url} not healthy within {timeout}s")


def _wait_for_sglang(base_url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang at {base_url} not healthy within {timeout}s")


def _wait_for_model_listing(base_url: str, *, model_hint: str | None = None, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                text = r.text
                if not model_hint or model_hint in text:
                    return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    suffix = f" containing {model_hint!r}" if model_hint else ""
    raise TimeoutError(f"{base_url}/v1/models did not return a model listing{suffix} within {timeout}s")


def _endpoint_already_registered(payload: Dict[str, Any], host: str, port: int) -> bool:
    message = str(payload.get("message", "")).lower()
    endpoint = payload.get("endpoint") or {}
    try:
        endpoint_port = int(endpoint.get("port", -1))
    except (TypeError, ValueError):
        return False
    return "already registered" in message and endpoint.get("host") == host and endpoint_port == int(port)


def _resolve_endpoint_urls(args: argparse.Namespace, coord_dir: Path) -> Dict[str, Any]:
    student_base_urls = _base_url_list(args.student_base_urls, arg_name="--student-base-urls")
    teacher_base_urls = _base_url_list(args.teacher_base_urls, arg_name="--teacher-base-urls")

    if not student_base_urls:
        student_base_urls = [
            _endpoint_base_url(_load_endpoint(coord_dir, "student", timeout=args.endpoint_timeout), role="student")
        ]
    if not teacher_base_urls:
        teacher_base_urls = [
            _endpoint_base_url(_load_endpoint(coord_dir, "teacher", timeout=args.endpoint_timeout), role="teacher")
        ]

    student_sync_base_urls = _base_url_list(args.student_sync_base_urls, arg_name="--student-sync-base-urls")
    if not student_sync_base_urls:
        student_sync_base_urls = list(student_base_urls)

    student_router_url = _normalize_base_url(args.student_router_url) if args.student_router_url else None
    teacher_router_url = _normalize_base_url(args.teacher_router_url) if args.teacher_router_url else None

    if args.native_route_mode == "smg":
        if student_router_url:
            student_route_urls = [student_router_url]
        elif len(student_base_urls) == 1:
            student_route_urls = list(student_base_urls)
        else:
            raise ValueError("--native-route-mode=smg with multiple student URLs requires --student-router-url")

        if teacher_router_url:
            teacher_route_urls = [teacher_router_url]
        elif len(teacher_base_urls) == 1:
            teacher_route_urls = list(teacher_base_urls)
        else:
            raise ValueError("--native-route-mode=smg with multiple teacher URLs requires --teacher-router-url")
    else:
        student_route_urls = list(student_base_urls)
        teacher_route_urls = list(teacher_base_urls)

    return {
        "student_base_urls": student_base_urls,
        "student_sync_base_urls": student_sync_base_urls,
        "student_route_urls": student_route_urls,
        "student_router_url": student_router_url,
        "teacher_base_urls": teacher_base_urls,
        "teacher_route_urls": teacher_route_urls,
        "teacher_router_url": teacher_router_url,
    }


def _register_inference_endpoint(
    train_url: str,
    endpoint_url: str,
    *,
    master_address: str | None,
    world_size: int = 8,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    endpoint = _endpoint_payload_from_base_url(endpoint_url, world_size=world_size, master_address=master_address)
    r = requests.post(f"{train_url}/add_inference_endpoint", json=endpoint, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("success"):
        return payload
    if _endpoint_already_registered(payload, host=str(endpoint["host"]), port=int(endpoint["port"])):
        log.info("Student SGLang inference endpoint %s already registered; reusing it.", endpoint_url)
        return payload
    raise RuntimeError(f"add_inference_endpoint failed for {endpoint_url}: {payload}")


def _sync_endpoint_success_count(sync_response: Dict[str, Any]) -> int:
    endpoints_synced = sync_response.get("endpoints_synced")
    if isinstance(endpoints_synced, list):
        return len(endpoints_synced)

    endpoint_results = sync_response.get("endpoint_results")
    if isinstance(endpoint_results, list):
        return sum(1 for result in endpoint_results if isinstance(result, dict) and result.get("success") is True)
    return 0


def _parse_prometheus_labels(raw: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def _parse_smg_success_counters(metrics_text: str) -> Dict[str, float]:
    counters: Dict[str, float] = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "smg_router_upstream" not in line:
            continue
        try:
            name_and_labels, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        label_text = ""
        if "{" in name_and_labels and name_and_labels.endswith("}"):
            label_text = name_and_labels[name_and_labels.index("{") + 1 : -1]
        labels = _parse_prometheus_labels(label_text)
        status = " ".join(str(v).lower() for v in labels.values())
        metric_name = name_and_labels.split("{", 1)[0].lower()
        if "success" not in status and "success" not in metric_name and labels.get("code") not in {"200", "2xx"}:
            continue
        worker = (
            labels.get("worker")
            or labels.get("upstream")
            or labels.get("backend")
            or labels.get("worker_url")
            or labels.get("url")
            or f"worker-{len(counters)}"
        )
        counters[worker] = max(value, counters.get(worker, 0.0))
    return counters


def _fetch_smg_success_counters(metrics_url: str | None, *, timeout: float = 5.0) -> Dict[str, float]:
    if not metrics_url:
        return {}
    try:
        r = requests.get(metrics_url, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        log.warning("Could not fetch SMG metrics from %s: %s", metrics_url, exc)
        return {}
    return _parse_smg_success_counters(r.text)


def _smg_success_delta_metrics(
    prefix: str,
    before: Dict[str, float],
    after: Dict[str, float],
) -> Dict[str, Any]:
    if not before and not after:
        return {
            f"{prefix}_smg_worker_count": None,
            f"{prefix}_smg_worker_success_delta_min": None,
            f"{prefix}_smg_worker_success_delta_max": None,
        }
    workers = sorted(set(before) | set(after))
    deltas = [max(0.0, float(after.get(worker, 0.0)) - float(before.get(worker, 0.0))) for worker in workers]
    return {
        f"{prefix}_smg_worker_count": len(workers),
        f"{prefix}_smg_worker_success_delta_min": min(deltas) if deltas else 0.0,
        f"{prefix}_smg_worker_success_delta_max": max(deltas) if deltas else 0.0,
    }


def _add_route_observation_profile(
    row: Dict[str, Any],
    args: argparse.Namespace,
    *,
    student_smg_before: Dict[str, float],
    teacher_smg_before: Dict[str, float],
) -> None:
    row.update(
        _smg_success_delta_metrics(
            "student",
            student_smg_before,
            _fetch_smg_success_counters(args.student_smg_metrics_url),
        )
    )
    row.update(
        _smg_success_delta_metrics(
            "teacher",
            teacher_smg_before,
            _fetch_smg_success_counters(args.teacher_smg_metrics_url),
        )
    )
    row.setdefault("student_gpu_util_mean", None)
    row.setdefault("teacher_gpu_util_mean", None)


def _wait_for_future(train_url: str, request_id: str, timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.post(
            f"{train_url}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("type") == "try_again":
            time.sleep(0.5)
            continue
        if payload.get("type") == "request_failed" or payload.get("error"):
            raise AssertionError(f"Future {request_id} failed: {payload}")
        return payload
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def _training_checkpoint_id(model_id: str, checkpoint_name: str) -> str:
    return f"weights/{model_id}/{checkpoint_name}"


def _save_training_checkpoint(
    train_url: str,
    *,
    model_id: str,
    checkpoint_name: str,
    timeout: float,
) -> Dict[str, Any]:
    save_t0 = time.perf_counter()
    r = requests.post(
        f"{train_url}/api/v1/save_weights",
        json={"model_id": model_id, "path": checkpoint_name},
        timeout=120,
    )
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"save_weights HTTP {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    payload = r.json()
    if "request_id" in payload:
        payload = _wait_for_future(train_url, payload["request_id"], timeout=timeout)
    checkpoint_id = _training_checkpoint_id(model_id, checkpoint_name)
    return {
        "name": checkpoint_name,
        "checkpoint_id": checkpoint_id,
        "path": payload.get("path", f"xorl://{model_id}/weights/{checkpoint_name}"),
        "save_s": _elapsed(save_t0),
    }


def _delete_training_checkpoint(
    train_url: str,
    *,
    model_id: str,
    checkpoint_id: str,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    delete_t0 = time.perf_counter()
    r = requests.post(
        f"{train_url}/api/v1/delete_checkpoint",
        json={"model_id": model_id, "checkpoint_id": checkpoint_id},
        timeout=timeout,
    )
    if r.status_code == 404:
        log.warning("Checkpoint already absent during retention cleanup: %s", checkpoint_id)
        return {"success": False, "checkpoint_id": checkpoint_id, "status_code": 404, "delete_s": _elapsed(delete_t0)}
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"delete_checkpoint HTTP {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    payload = r.json()
    payload["checkpoint_id"] = checkpoint_id
    payload["delete_s"] = _elapsed(delete_t0)
    return payload


def _write_checkpoint_summary(
    path: Path,
    *,
    policy: Dict[str, Any],
    latest_checkpoints: List[Dict[str, Any]],
    best_checkpoint: Dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": policy,
        "latest_checkpoints": latest_checkpoints,
        "best_checkpoint": best_checkpoint,
        "updated_at_unix": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _checkpoint_metric_value(row: Dict[str, Any], metric_name: str) -> float | None:
    candidates = (metric_name, f"{metric_name}:mean", f"{metric_name}_mean")
    for key in candidates:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _is_better_checkpoint_metric(candidate: float, incumbent: float | None, *, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    if mode == "max":
        return candidate > incumbent
    raise ValueError(f"Unsupported checkpoint best mode: {mode}")


def _scalar(value: Any) -> float:
    if isinstance(value, dict) and "data" in value:
        return float(value["data"][0])
    if hasattr(value, "data"):
        return float(value.data[0])
    return float(value)


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _metric(metrics: Dict[str, Any], name: str, default: Any = None) -> Any:
    for key in (name, f"{name}:mean", f"{name}:sum", f"{name}:max"):
        if key in metrics:
            return metrics[key]
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _json_arg_or_none(raw: str | None, *, arg_name: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON; got {raw!r}") from exc


def _dict_json_arg_or_none(raw: str | None, *, arg_name: str) -> Dict[str, Any] | None:
    value = _json_arg_or_none(raw, arg_name=arg_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object; got {type(value).__name__}")
    return value


def _str_list_json_arg_or_none(raw: str | None, *, arg_name: str) -> List[str] | None:
    value = _json_arg_or_none(raw, arg_name=arg_name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{arg_name} must decode to a JSON string list; got {type(value).__name__}")
    return value


def _int_list_json_arg(raw: str | None, *, arg_name: str, default: List[int]) -> List[int]:
    value = _json_arg_or_none(raw, arg_name=arg_name)
    if value is None:
        return list(default)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{arg_name} must decode to a JSON integer list; got {type(value).__name__}")
    if not value:
        raise ValueError(f"{arg_name} must not be empty")
    return [int(item) for item in value]


def _sampling_params(
    *,
    max_new_tokens: int,
    temperature: float,
    extra: Dict[str, Any] | None,
) -> Dict[str, Any]:
    params = dict(extra or {})
    params["temperature"] = temperature
    params["max_new_tokens"] = max_new_tokens
    return params


def _sampling_params_are_argmax(*, temperature: float, sampling_params: Dict[str, Any] | None) -> bool:
    params = sampling_params or {}
    top_k = params.get("top_k")
    top_p = params.get("top_p", 1.0)
    min_p = params.get("min_p", 0.0)
    if float(temperature) != 0.0 and float(params.get("temperature", temperature)) != 0.0:
        return False
    return (top_k is None or int(top_k) == 1) and float(top_p) == 1.0 and float(min_p) == 0.0


def _bool_config_value(config: Dict[str, Any], keys: tuple[str, ...], default: bool) -> bool:
    for key in keys:
        if key in config and config[key] is not None:
            return bool(config[key])
    return default


def _int_config_value(config: Dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in config and config[key] is not None:
            return int(config[key])
    return default


def _write_profile_row(profile_output: Path, row: Dict[str, Any]) -> None:
    with profile_output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _delete_consumed_teacher_cache(cache_path: Path) -> bool:
    try:
        cache_path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.warning("Failed to delete consumed teacher hidden cache %s: %s", cache_path, exc)
        return False


def _wandb_metrics_from_profile_row(row: Dict[str, Any]) -> Dict[str, int | float | bool | str]:
    metrics: Dict[str, int | float | bool | str] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, bool, str)):
            metrics[key] = value
    return metrics


def _wandb_config_from_args(
    args: argparse.Namespace,
    *,
    profile_output: Path,
    prompt_schedule: "_PromptSchedule",
) -> Dict[str, Any]:
    return {
        "coord_dir": args.coord_dir,
        "profile_output": str(profile_output),
        "teacher_backend": args.teacher_backend,
        "teacher_head": args.teacher_head or args.teacher_model_dir,
        "max_opd_steps": args.max_opd_steps,
        "max_new_tokens": args.max_new_tokens,
        "opd_loss_mode": args.opd_loss_mode,
        "optim_learning_rate": args.optim_learning_rate,
        "optim_gradient_clip": args.optim_gradient_clip,
        "opd_emit_full_vocab_diagnostics": args.opd_emit_full_vocab_diagnostics,
        "sync_buffer_size_mb": args.sync_buffer_size_mb,
        "sync_pause_mode": args.sync_pause_mode,
        "sync_quantization": args.sync_quantization,
        "student_temperature": args.student_temperature,
        "student_sampling_params": args.student_sampling_params,
        "student_argmax_rollout": args.student_argmax_rollout,
        "singleshot_mtp": args.singleshot_mtp,
        "pipeline_chunk_size": args.pipeline_chunk_size,
        "pipeline_prefetch_chunks": args.pipeline_prefetch_chunks,
        "pipeline_teacher_concurrency": args.pipeline_teacher_concurrency,
        "prompt_dataset_path": args.prompt_dataset_path,
        "prompt_dataset_type": args.prompt_dataset_type,
        "prompt_dataset_split": args.prompt_dataset_split,
        "prompt_dataset_column": args.prompt_dataset_column,
        "prompt_dataset_num_prompts": args.prompt_dataset_num_prompts,
        "prompt_dataset_prompt_len": args.prompt_dataset_prompt_len,
        "prompt_dataset_offset": args.prompt_dataset_offset,
        "prompt_dataset_epochs": args.prompt_dataset_epochs,
        "prompt_dataset_turn_strategy": args.prompt_dataset_turn_strategy,
        "prompt_dataset_min_target_tokens": args.prompt_dataset_min_target_tokens,
        "prompt_dataset_assistant_offset_tokens": args.prompt_dataset_assistant_offset_tokens,
        "prompt_dataset_im_start_token_id": args.prompt_dataset_im_start_token_id,
        "prompt_dataset_im_end_token_id": args.prompt_dataset_im_end_token_id,
        "prompt_dataset_assistant_role_token_ids": args.prompt_dataset_assistant_role_token_ids,
        "prompt_schedule": prompt_schedule.metadata(),
        "rollout_samples_output": args.rollout_samples_output,
        "rollout_samples_per_step": args.rollout_samples_per_step,
        "rollout_sample_tail_tokens": args.rollout_sample_tail_tokens,
        "rollout_sample_target_tokens": args.rollout_sample_target_tokens,
        "rollout_sample_text_max_chars": args.rollout_sample_text_max_chars,
        "rollout_sample_tokenizer_path": args.rollout_sample_tokenizer_path,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "checkpoint_keep_latest": args.checkpoint_keep_latest,
        "checkpoint_save_best": args.checkpoint_save_best,
        "checkpoint_best_metric": args.checkpoint_best_metric,
        "checkpoint_best_mode": args.checkpoint_best_mode,
        "checkpoint_eval_dataset_split": args.checkpoint_eval_dataset_split,
        "checkpoint_eval_dataset_offset": args.checkpoint_eval_dataset_offset,
        "checkpoint_eval_steps": args.checkpoint_eval_steps,
        "checkpoint_eval_batch_size": args.checkpoint_eval_batch_size,
    }


def _init_wandb(
    args: argparse.Namespace,
    *,
    profile_output: Path,
    prompt_schedule: "_PromptSchedule",
) -> Any | None:
    if args.disable_wandb or not args.wandb_project:
        return None

    import wandb  # noqa: PLC0415

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    init_kwargs: Dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_name or Path(args.coord_dir).name,
        "tags": args.wandb_tags,
        "config": _wandb_config_from_args(args, profile_output=profile_output, prompt_schedule=prompt_schedule),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    run = wandb.init(**init_kwargs)
    log.info(
        "W&B enabled: project=%s name=%s entity=%s mode=%s",
        args.wandb_project,
        init_kwargs["name"],
        args.wandb_entity,
        args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
    )
    return run


def _log_profile_row_to_wandb(wandb_run: Any | None, row: Dict[str, Any], *, log_interval: int) -> None:
    if wandb_run is None:
        return
    step = int(row.get("step", 0))
    if log_interval > 1 and step % log_interval != 0:
        return
    wandb_run.log(_wandb_metrics_from_profile_row(row), step=step)


def _record_profile_row(
    profile_rows: List[Dict[str, Any]],
    profile_output: Path,
    row: Dict[str, Any],
    *,
    wandb_run: Any | None,
    wandb_log_interval: int,
) -> None:
    profile_rows.append(row)
    _write_profile_row(profile_output, row)
    _log_profile_row_to_wandb(wandb_run, row, log_interval=wandb_log_interval)


def _mean(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _sum_metric(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows if isinstance(row.get(key, 0.0), (int, float)))


_OPD_LOSS_PROFILE_METRICS = (
    "opd_kl",
    "opd_weighted_kl",
    "opd_teacher_entropy",
    "opd_student_entropy",
    "opd_top1_agreement",
    "opd_teacher_rollout_token_agreement",
    "opd_student_rollout_token_agreement",
    "opd_hard_teacher_ce",
    "opd_abs_loss",
)

_SINGLESHOT_TRAINER_SUM_METRICS = (
    "opd_singleshot_mtp_micro_batches",
    "opd_singleshot_mtp_batch_tokens",
    "opd_singleshot_mtp_local_valid_tokens",
    "opd_singleshot_mtp_replay_plan_sequences",
    "opd_singleshot_mtp_replay_plan_outputs",
    "opd_singleshot_mtp_replay_plan_packed_tokens",
    "opd_singleshot_mtp_replay_plan_base_context_tokens",
    "opd_singleshot_mtp_replay_plan_base_mask_tokens",
    "opd_singleshot_mtp_replay_plan_virtual_context_tokens",
    "opd_singleshot_mtp_replay_plan_virtual_mask_tokens",
    "opd_singleshot_mtp_replay_plan_virtual_pad_tokens",
    "opd_singleshot_mtp_replay_plan_context_dup_tokens",
    "opd_singleshot_mtp_replay_plan_stateful_replay_tokens",
    "opd_singleshot_mtp_replay_plan_stateful_token_savings",
    "opd_singleshot_mtp_replay_plan_prefix_state_boundaries",
    "opd_singleshot_mtp_replay_plan_prefix_state_branch_sequences",
    "opd_singleshot_mtp_replay_plan_gdn_forward_calls",
    "opd_singleshot_mtp_replay_plan_stateful_schedule_micro_batches",
    "opd_singleshot_mtp_replay_plan_stateful_fallback_micro_batches",
    "opd_singleshot_mtp_replay_plan_stateful_context_segments",
    "opd_singleshot_mtp_replay_plan_stateful_context_packed_tokens",
    "opd_singleshot_mtp_replay_plan_stateful_suffix_segments",
    "opd_singleshot_mtp_replay_plan_stateful_suffix_packed_tokens",
    "opd_singleshot_mtp_replay_plan_stateful_state_slots",
    "opd_singleshot_mtp_replay_plan_stateful_gdn_forward_calls",
    "opd_singleshot_mtp_native_trace_micro_batches",
    "opd_singleshot_mtp_gdn_executed_tokens",
    "opd_singleshot_mtp_attn_visible_pairs",
    "opd_singleshot_mtp_global_batch_tokens",
    "opd_singleshot_mtp_global_nonpad_tokens",
    "opd_singleshot_mtp_global_gdn_executed_tokens",
    "opd_singleshot_mtp_global_attn_visible_pairs",
    "opd_singleshot_mtp_flops_mlp_tf",
    "opd_singleshot_mtp_flops_attn_tf",
    "opd_singleshot_mtp_flops_gdn_tf",
    "opd_singleshot_mtp_flops_lm_head_tf",
    "opd_singleshot_mtp_flops_actual_tf",
    "opd_singleshot_mtp_flops_useful_tf",
)

_SINGLESHOT_TRAINER_MAX_METRICS = (
    "opd_singleshot_mtp_max_padded_seq_len",
    "opd_singleshot_mtp_max_raw_seq_len",
    "opd_singleshot_mtp_max_generated_tokens",
    "opd_singleshot_mtp_max_chunks",
    "opd_singleshot_mtp_replay_plan_max_sequence_len",
    "opd_singleshot_mtp_replay_plan_gdn_cap",
    "opd_singleshot_mtp_replay_plan_gdn_max_chunk_packed_tokens",
    "opd_singleshot_mtp_replay_plan_stateful_context_rounds",
    "opd_singleshot_mtp_flops_world_size",
    "opd_singleshot_mtp_flops_promised_tflops_per_gpu",
    "opd_singleshot_mtp_batch_replication_factor",
    "opd_singleshot_mtp_enabled",
)

_SINGLESHOT_TRAINER_RATIO_METRICS = (
    "opd_singleshot_mtp_replay_plan_tokens_per_local_valid",
    "opd_singleshot_mtp_batch_tokens_per_local_valid",
    "opd_singleshot_mtp_replay_plan_tokens_per_global_valid",
    "opd_singleshot_mtp_batch_tokens_per_global_valid",
    "opd_singleshot_mtp_replay_plan_context_dup_per_local_valid",
    "opd_singleshot_mtp_replay_plan_context_dup_per_global_valid",
    "opd_singleshot_mtp_replay_plan_context_dup_fraction",
    "opd_singleshot_mtp_replay_plan_stateful_tokens_per_local_valid",
    "opd_singleshot_mtp_replay_plan_stateful_tokens_per_global_valid",
    "opd_singleshot_mtp_replay_plan_stateful_token_savings_fraction",
)


def _opd_loss_metrics_from_response(metrics: Dict[str, Any]) -> Dict[str, float]:
    profile_metrics: Dict[str, float] = {}
    for key in _OPD_LOSS_PROFILE_METRICS:
        value = _metric(metrics, key)
        if isinstance(value, (int, float)):
            profile_metrics[key] = float(value)
    return profile_metrics


_singleshot_metrics_missing_logged = False


def _singleshot_trainer_metrics_from_response(metrics: Dict[str, Any]) -> Dict[str, float]:
    profile_metrics: Dict[str, float] = {}
    for key in _SINGLESHOT_TRAINER_SUM_METRICS + _SINGLESHOT_TRAINER_MAX_METRICS + _SINGLESHOT_TRAINER_RATIO_METRICS:
        value = _metric(metrics, key)
        if isinstance(value, (int, float)):
            profile_metrics[key] = float(value)
    if not profile_metrics:
        global _singleshot_metrics_missing_logged
        if not _singleshot_metrics_missing_logged:
            _singleshot_metrics_missing_logged = True
            log.info(
                "No opd_singleshot_mtp_* metrics in FB response; response metric keys: %s",
                sorted(metrics.keys()),
            )
    return profile_metrics


def _add_singleshot_trainer_profile_totals(row: Dict[str, Any], chunk_fb_profiles: List[Dict[str, Any]]) -> None:
    for key in _SINGLESHOT_TRAINER_SUM_METRICS:
        total = _sum_metric(chunk_fb_profiles, key)
        if total:
            row[key] = total
    for key in _SINGLESHOT_TRAINER_MAX_METRICS:
        values = [float(chunk.get(key, 0.0)) for chunk in chunk_fb_profiles if isinstance(chunk.get(key), (int, float))]
        if values:
            row[key] = max(values)

    local_valid = float(row.get("opd_singleshot_mtp_local_valid_tokens", 0.0) or 0.0)
    global_valid = float(row.get("valid_tokens", 0.0) or 0.0)
    replay_tokens = float(row.get("opd_singleshot_mtp_replay_plan_packed_tokens", 0.0) or 0.0)
    batch_tokens = float(row.get("opd_singleshot_mtp_batch_tokens", 0.0) or 0.0)
    stateful_tokens = float(row.get("opd_singleshot_mtp_replay_plan_stateful_replay_tokens", 0.0) or 0.0)
    if local_valid > 0:
        row["opd_singleshot_mtp_replay_plan_tokens_per_local_valid"] = replay_tokens / local_valid
        row["opd_singleshot_mtp_batch_tokens_per_local_valid"] = batch_tokens / local_valid
        row["opd_singleshot_mtp_replay_plan_context_dup_per_local_valid"] = (
            float(row.get("opd_singleshot_mtp_replay_plan_context_dup_tokens", 0.0) or 0.0) / local_valid
        )
        row["opd_singleshot_mtp_replay_plan_stateful_tokens_per_local_valid"] = stateful_tokens / local_valid
    if global_valid > 0:
        row["opd_singleshot_mtp_replay_plan_tokens_per_global_valid"] = replay_tokens / global_valid
        row["opd_singleshot_mtp_batch_tokens_per_global_valid"] = batch_tokens / global_valid
        row["opd_singleshot_mtp_replay_plan_context_dup_per_global_valid"] = (
            float(row.get("opd_singleshot_mtp_replay_plan_context_dup_tokens", 0.0) or 0.0) / global_valid
        )
        row["opd_singleshot_mtp_replay_plan_stateful_tokens_per_global_valid"] = stateful_tokens / global_valid
    if replay_tokens > 0:
        row["opd_singleshot_mtp_replay_plan_context_dup_fraction"] = (
            float(row.get("opd_singleshot_mtp_replay_plan_context_dup_tokens", 0.0) or 0.0) / replay_tokens
        )
        row["opd_singleshot_mtp_replay_plan_stateful_token_savings_fraction"] = (
            float(row.get("opd_singleshot_mtp_replay_plan_stateful_token_savings", 0.0) or 0.0) / replay_tokens
        )

    fb_s = float(row.get("trainer_forward_backward_s", 0.0) or 0.0)
    flops_actual_tf = float(row.get("opd_singleshot_mtp_flops_actual_tf", 0.0) or 0.0)
    flops_useful_tf = float(row.get("opd_singleshot_mtp_flops_useful_tf", 0.0) or 0.0)
    world_size = float(row.get("opd_singleshot_mtp_flops_world_size", 0.0) or 0.0)
    promised_tflops = float(row.get("opd_singleshot_mtp_flops_promised_tflops_per_gpu", 0.0) or 0.0)
    if fb_s > 0 and flops_actual_tf > 0 and world_size > 0:
        tflops_per_gpu = flops_actual_tf / (fb_s * world_size)
        row["opd_singleshot_mtp_tflops_per_gpu"] = tflops_per_gpu
        if math.isfinite(promised_tflops) and promised_tflops > 0:
            row["opd_singleshot_mtp_mfu_actual"] = tflops_per_gpu / promised_tflops
            if flops_useful_tf > 0:
                row["opd_singleshot_mtp_mfu_useful"] = flops_useful_tf / (fb_s * world_size * promised_tflops)
        if flops_useful_tf > 0:
            row["opd_singleshot_mtp_flops_regret_ratio"] = flops_actual_tf / flops_useful_tf


def _weighted_mean_metric(rows: List[Dict[str, Any]], key: str, *, weight_key: str = "valid_tokens") -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key)
        if isinstance(value, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
            weighted_sum += float(value) * float(weight)
            weight_sum += float(weight)
    if weight_sum <= 0:
        return None
    return weighted_sum / weight_sum


def _csv_union_metric(rows: List[Dict[str, Any]], key: str, *, numeric: bool = False) -> str | None:
    values = set()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            values.add(int(item) if numeric else item)
    if not values:
        return None
    return ",".join(str(value) for value in sorted(values))


def _unique_int_metric(rows: List[Dict[str, Any]], key: str) -> int | None:
    values = {int(row[key]) for row in rows if isinstance(row.get(key), (int, float))}
    if len(values) != 1:
        return None
    return next(iter(values))


def _aggregate_native_mtp_trace_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    native_rows = [row for row in rows if row.get("student_sampling_mode") == "sglang_native_mtp"]
    if not native_rows:
        return metrics

    metrics["student_sampling_mtp_native"] = any(bool(row.get("student_sampling_mtp_native")) for row in native_rows)
    for key in ("student_sampling_mtp_k_toks", "student_sampling_mtp_mask_token_id", "student_sampling_top_k"):
        value = _unique_int_metric(native_rows, key)
        if value is not None:
            metrics[key] = value
    for key in (
        "student_sampling_temperature",
        "student_sampling_top_p",
        "student_sampling_min_p",
        "student_sampling_mtp_conf_threshold",
    ):
        values = {float(row[key]) for row in native_rows if isinstance(row.get(key), (int, float))}
        if len(values) == 1:
            metrics[key] = next(iter(values))
    for key in ("student_sampling_mtp_strategy", "student_sampling_mtp_adaptive_window_mode"):
        values = {str(row[key]) for row in native_rows if row.get(key) is not None}
        if len(values) == 1:
            metrics[key] = next(iter(values))
    argmax_values = [
        bool(row["student_sampling_argmax_rollout"])
        for row in native_rows
        if isinstance(row.get("student_sampling_argmax_rollout"), bool)
    ]
    if argmax_values:
        metrics["student_sampling_argmax_rollout"] = all(argmax_values)

    requested_values = [
        bool(row["student_sampling_mtp_debug_trace_requested"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_debug_trace_requested"), bool)
    ]
    if requested_values:
        metrics["student_sampling_mtp_debug_trace_requested"] = any(requested_values)

    for key in (
        "student_sampling_mtp_debug_trace_samples",
        "student_sampling_mtp_debug_trace_steps",
        "student_sampling_mtp_debug_trace_seed_steps",
        "student_sampling_mtp_debug_trace_steady_steps",
        "student_sampling_mtp_debug_trace_cuda_graph_steps",
        "student_sampling_mtp_debug_trace_committed_tokens",
        "student_sampling_mtp_generated_tokens",
        "student_sampling_mtp_supervised_tokens_expected",
        "student_sampling_mtp_bootstrap_tokens",
        "student_sampling_mtp_replay_committed_tokens",
        "student_sampling_mtp_trace_tokens_unused",
    ):
        if any(isinstance(row.get(key), (int, float)) for row in native_rows):
            metrics[key] = int(_sum_metric(native_rows, key))

    q_lens = _csv_union_metric(native_rows, "student_sampling_mtp_debug_trace_q_lens", numeric=True)
    if q_lens is not None:
        metrics["student_sampling_mtp_debug_trace_q_lens"] = q_lens
    max_q_lens = [
        int(row["student_sampling_mtp_debug_trace_max_q_len"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_debug_trace_max_q_len"), (int, float))
    ]
    if max_q_lens:
        metrics["student_sampling_mtp_debug_trace_max_q_len"] = max(max_q_lens)

    phases = _csv_union_metric(native_rows, "student_sampling_mtp_debug_trace_phases")
    if phases is not None:
        metrics["student_sampling_mtp_debug_trace_phases"] = phases
        phase_set = set(phases.split(","))
        metrics["student_sampling_mtp_debug_trace_seen_seed"] = "seed" in phase_set
        metrics["student_sampling_mtp_debug_trace_seen_steady"] = "steady" in phase_set

    effective_k_mins = [
        int(row["student_sampling_mtp_debug_trace_effective_k_min"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_debug_trace_effective_k_min"), (int, float))
    ]
    effective_k_maxes = [
        int(row["student_sampling_mtp_debug_trace_effective_k_max"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_debug_trace_effective_k_max"), (int, float))
    ]
    if effective_k_mins:
        metrics["student_sampling_mtp_debug_trace_effective_k_min"] = min(effective_k_mins)
    if effective_k_maxes:
        metrics["student_sampling_mtp_debug_trace_effective_k_max"] = max(effective_k_maxes)

    effective_k_weighted_sum = 0.0
    effective_k_weight = 0.0
    for row in native_rows:
        mean_value = row.get("student_sampling_mtp_debug_trace_effective_k_mean")
        weight = row.get(
            "student_sampling_mtp_debug_trace_steps", row.get("student_sampling_mtp_debug_trace_samples", 1)
        )
        if isinstance(mean_value, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
            effective_k_weighted_sum += float(mean_value) * float(weight)
            effective_k_weight += float(weight)
    if effective_k_weight > 0:
        metrics["student_sampling_mtp_debug_trace_effective_k_mean"] = effective_k_weighted_sum / effective_k_weight

    weighted_mean_specs = {
        "student_sampling_mtp_debug_trace_commit_len_mean": "student_sampling_mtp_debug_trace_steps",
        "student_sampling_mtp_debug_trace_commit_len_seed": "student_sampling_mtp_debug_trace_seed_steps",
        "student_sampling_mtp_debug_trace_commit_len_steady": "student_sampling_mtp_debug_trace_steady_steps",
        "student_sampling_mtp_debug_trace_pending_confidence_mean": "student_sampling_mtp_debug_trace_steps",
    }
    for key, weight_key in weighted_mean_specs.items():
        weighted_sum = 0.0
        weight_sum = 0.0
        for row in native_rows:
            mean_value = row.get(key)
            weight = row.get(weight_key, row.get("student_sampling_mtp_debug_trace_samples", 1))
            if isinstance(mean_value, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
                weighted_sum += float(mean_value) * float(weight)
                weight_sum += float(weight)
        if weight_sum > 0:
            metrics[key] = weighted_sum / weight_sum
    for source_key, reducer, target_key in (
        ("student_sampling_mtp_debug_trace_commit_len_min", min, "student_sampling_mtp_debug_trace_commit_len_min"),
        ("student_sampling_mtp_debug_trace_commit_len_max", max, "student_sampling_mtp_debug_trace_commit_len_max"),
        (
            "student_sampling_mtp_debug_trace_pending_confidence_min",
            min,
            "student_sampling_mtp_debug_trace_pending_confidence_min",
        ),
    ):
        values = [float(row[source_key]) for row in native_rows if isinstance(row.get(source_key), (int, float))]
        if values:
            metrics[target_key] = reducer(values)

    cuda_graph_all_values = [
        bool(row["student_sampling_mtp_debug_trace_cuda_graph_all"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_debug_trace_cuda_graph_all"), bool)
    ]
    if cuda_graph_all_values:
        metrics["student_sampling_mtp_debug_trace_cuda_graph_all"] = all(cuda_graph_all_values)

    trace_covered_values = [
        bool(row["student_sampling_mtp_replay_trace_covered_all_targets"])
        for row in native_rows
        if isinstance(row.get("student_sampling_mtp_replay_trace_covered_all_targets"), bool)
    ]
    if trace_covered_values:
        metrics["student_sampling_mtp_replay_trace_covered_all_targets"] = all(trace_covered_values)

    return metrics


def _infer_visible_device_count(default: int = 1) -> int:
    override = os.environ.get("OPD_WANDB_NUM_DEVICES")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    return len(devices) if devices else default


def _padded_length(length: int, multiple: int | None) -> int:
    if not multiple:
        return length
    return ((length + multiple - 1) // multiple) * multiple


def _int_schedule_from_config(value: Any, *, key: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            value = json.loads(text)
        elif ".." in text:
            start_text, end_text = text.split("..", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            value = list(range(start, end + step, step))
        elif "-" in text and "," not in text and " " not in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            value = list(range(start, end + step, step))
        else:
            value = [item for item in text.replace(",", " ").split(" ") if item]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list, range string, or comma-separated string")

    schedule = [int(item) for item in value]
    if not schedule:
        raise ValueError(f"{key} must not be empty")
    if any(item <= 0 for item in schedule):
        raise ValueError(f"{key} values must be positive integers, got {schedule}")
    return schedule


def _scheduled_singleshot_mtp_config(
    base: Dict[str, Any] | None,
    *,
    step: int,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    if not isinstance(base, dict):
        return base, {}

    cfg = dict(base)
    metadata: Dict[str, Any] = {}
    schedule_value = cfg.pop("k_toks_schedule", cfg.pop("k_schedule", None))
    if schedule_value is not None:
        schedule = _int_schedule_from_config(schedule_value, key="k_toks_schedule")
        mode = str(cfg.pop("k_toks_schedule_mode", cfg.pop("k_schedule_mode", "cycle"))).lower()
        seed = int(cfg.pop("k_toks_schedule_seed", cfg.pop("k_schedule_seed", 0)) or 0)
        if mode in {"random", "rand", "uniform", "lockstep_random", "lockstep-random"}:
            index = random.Random(seed + step).randrange(len(schedule))
        elif mode in {"cycle", "round_robin", "round-robin"}:
            index = step % len(schedule)
        else:
            raise ValueError(f"Unsupported k_toks_schedule_mode {mode!r}; expected cycle or random")
        cfg["k_toks"] = schedule[index]
        metadata.update(
            {
                "singleshot_mtp_schedule_enabled": True,
                "singleshot_mtp_schedule_mode": mode,
                "singleshot_mtp_schedule_seed": seed,
                "singleshot_mtp_schedule_index": index,
                "singleshot_mtp_schedule_size": len(schedule),
                "singleshot_mtp_schedule_k_toks": int(cfg["k_toks"]),
            }
        )
    else:
        metadata["singleshot_mtp_schedule_enabled"] = False

    prompt_pad_to_multiple = cfg.get("prompt_pad_to_multiple")
    if str(prompt_pad_to_multiple).lower() in {"k", "k_toks", "dynamic_k"}:
        if "k_toks" not in cfg:
            raise ValueError('prompt_pad_to_multiple="k_toks" requires k_toks or k_toks_schedule')
        cfg["prompt_pad_to_multiple"] = int(cfg["k_toks"])
    elif isinstance(prompt_pad_to_multiple, str) and prompt_pad_to_multiple.strip():
        cfg["prompt_pad_to_multiple"] = int(prompt_pad_to_multiple)

    return cfg, metadata


def _add_singleshot_profile_aliases(
    row: Dict[str, Any],
    args: argparse.Namespace,
    *,
    device_count: int,
    cumulative_tokens: int,
    cumulative_sup_tokens: int,
    cumulative_consumed_tokens: int,
) -> None:
    singleshot = args.singleshot_mtp if isinstance(args.singleshot_mtp, dict) else {}
    for key in (
        "k_toks",
        "mask_region_count",
        "truncation_length",
        "offset",
        "pad_to_multiple",
        "static_padded_seq_len",
        "prompt_pad_to_multiple",
        "rollout_replay",
        "flex_block_size",
        "bidirectional_mtp_attention",
        "truncation_side",
        "preserve_context_prefix",
        "sampling_mode",
        "train_rollout_strategy",
        "mtp_conf_threshold",
        "confidence_threshold",
        "conf_adapt_threshold",
        "conf_threshold",
        "mtp_strategy",
        "native_mtp_strategy",
        "validate_native_mtp_trace",
        "native_mtp_debug_trace",
        "native_mtp_debug_max_steps",
    ):
        if key in singleshot:
            row[f"singleshot_mtp_{key}"] = singleshot[key]
    if singleshot:
        strategy, threshold = _native_mtp_strategy_config(singleshot)
        if strategy is not None:
            row["singleshot_mtp_native_strategy"] = json.dumps(strategy, sort_keys=True)
        if threshold is not None:
            row["singleshot_mtp_conf_threshold"] = threshold
    if "k_toks" in singleshot:
        row["k_toks"] = singleshot["k_toks"]
    truncation_length = singleshot.get("truncation_length")
    pad_to_multiple = singleshot.get("pad_to_multiple")
    if isinstance(truncation_length, int):
        row["singleshot_mtp_raw_seq_len"] = truncation_length
        row["singleshot_mtp_padded_seq_len"] = _padded_length(
            truncation_length,
            int(pad_to_multiple) if isinstance(pad_to_multiple, int) else None,
        )

    if "loss" in row:
        row["loss_rank0"] = row["loss"]
    row["iter"] = row.get("step", 0)
    row["epoch"] = row.get("prompt_dataset_epoch_start", 0)

    step_total_s = float(row.get("step_total_s", 0.0) or 0.0)
    row["iter_time"] = step_total_s
    row["time_iter_total"] = step_total_s
    prepare_wait_s = row.get("opd_pipeline_prepare_wait_s")
    if isinstance(prepare_wait_s, (int, float)):
        row["time_data_mask_prep"] = float(prepare_wait_s)
    else:
        row["time_data_mask_prep"] = float(row.get("student_sampling_s", 0.0) or 0.0) + float(
            row.get("teacher_prefill_s", 0.0) or 0.0
        )
    row["time_fwd"] = float(row.get("trainer_forward_loss_s", 0.0) or 0.0)
    row["time_metrics"] = 0.0
    row["time_main_loss"] = float(row.get("trainer_opd_kl_compute_ms", 0.0) or 0.0) / 1000.0
    row["time_prefix_loss"] = 0.0
    row["time_bwd"] = float(row.get("trainer_backward_s", 0.0) or 0.0)
    row["time_post_bwd"] = 0.0
    row["time_grad_clip"] = 0.0
    row["time_optim_step"] = float(row.get("optim_step_roundtrip_s", 0.0) or 0.0)
    row["time_log_calcs"] = 0.0

    row["tokens"] = cumulative_tokens
    row["total_tokens"] = cumulative_tokens
    row["sup_tokens"] = cumulative_sup_tokens
    row["total_sup_tokens"] = cumulative_sup_tokens
    row["consumed_tokens"] = cumulative_consumed_tokens
    row["total_consumed_tokens"] = cumulative_consumed_tokens
    row["learning_rate"] = args.optim_learning_rate
    row["profile_num_devices"] = device_count

    if "opd_kl" in row:
        row["kl_teach_stud"] = row["opd_kl"]
    if "opd_teacher_entropy" in row:
        row["ent_teach"] = row["opd_teacher_entropy"]
    if "opd_student_entropy" in row:
        row["ent_stud"] = row["opd_student_entropy"]
    if "opd_hard_teacher_ce" in row:
        row["ce_teach_stud"] = row["opd_hard_teacher_ce"]
        row["hard_teacher_ce"] = row["opd_hard_teacher_ce"]
    if "opd_teacher_rollout_token_agreement" in row:
        row["teacher_rollout_token_agreement"] = row["opd_teacher_rollout_token_agreement"]
    if "opd_student_rollout_token_agreement" in row:
        row["student_rollout_token_agreement"] = row["opd_student_rollout_token_agreement"]

    mtp_aliases = {
        "student_sampling_mtp_k_toks": "mtp/k_toks",
        "student_sampling_mtp_debug_trace_effective_k_mean": "mtp/effective_k_mean",
        "student_sampling_mtp_debug_trace_commit_len_mean": "mtp/commit_len_mean",
        "student_sampling_mtp_debug_trace_commit_len_seed": "mtp/commit_len_seed",
        "student_sampling_mtp_debug_trace_commit_len_steady": "mtp/commit_len_steady",
        "student_sampling_mtp_partial_blocks": "mtp/partial_blocks",
        "student_sampling_mtp_bootstrap_tokens": "mtp/bootstrap_tokens",
        "student_sampling_mtp_generated_tokens": "mtp/generated_tokens",
        "student_sampling_mtp_supervised_tokens_expected": "mtp/supervised_tokens",
        "student_sampling_mtp_trace_tokens_unused": "mtp/trace_tokens_unused",
        "student_sampling_mtp_replay_trace_covered_all_targets": "mtp/replay_trace_covered_all_targets",
        "student_sampling_mtp_debug_trace_seen_seed": "mtp/native_trace_seen_seed",
        "student_sampling_mtp_debug_trace_seen_steady": "mtp/native_trace_seen_steady",
        "student_sampling_mtp_debug_trace_cuda_graph_all": "mtp/cuda_graph_all",
        "student_sampling_mtp_conf_threshold": "mtp/conf_threshold",
        "student_sampling_argmax_rollout": "mtp/argmax_rollout",
    }
    for source_key, alias_key in mtp_aliases.items():
        if source_key in row:
            row[alias_key] = row[source_key]
    if "valid_tokens" in row:
        row["mtp/supervised_tokens_actual"] = int(row.get("valid_tokens", 0) or 0)

    step_tokens = int(row.get("teacher_prefill_tokens", 0) or 0)
    step_sup_tokens = int(row.get("valid_tokens", 0) or 0)
    step_consumed_tokens = int(row.get("student_sampling_prompt_tokens", 0) or 0) + int(
        row.get("student_sampling_output_tokens", 0) or 0
    )
    if step_total_s > 0:
        row["toks_per_sec_world"] = step_tokens / step_total_s
        row["sup_toks_per_sec_world"] = step_sup_tokens / step_total_s
        row["consumed_toks_per_sec_world"] = step_consumed_tokens / step_total_s
        divisor = max(device_count, 1)
        row["toks_per_sec_per_device"] = row["toks_per_sec_world"] / divisor
        row["sup_toks_per_sec_per_device"] = row["sup_toks_per_sec_world"] / divisor
        row["consumed_toks_per_sec_per_device"] = row["consumed_toks_per_sec_world"] / divisor


def _chunked(items: List[List[int]], chunk_size: int) -> List[List[List[int]]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _infer_local_dataset_type(path: Path) -> str | None:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet"
        if suffix in {".json", ".jsonl"}:
            return "json"
        if suffix == ".csv":
            return "csv"
        if suffix == ".txt":
            return "text"
        if suffix == ".arrow":
            return "arrow"
        return None

    if path.is_dir():
        for pattern, ds_type in (
            ("*.parquet", "parquet"),
            ("*.jsonl", "json"),
            ("*.json", "json"),
            ("*.csv", "csv"),
            ("*.txt", "text"),
            ("*.arrow", "arrow"),
        ):
            if any(path.glob(pattern)):
                return ds_type
    return None


def _load_tokenized_prompt_dataset(path: str, *, ds_type: str | None, split: str):
    from datasets import load_dataset, load_from_disk  # noqa: PLC0415

    dataset_path = Path(path)
    if ds_type in {None, "", "hf_disk"} and dataset_path.is_dir() and (dataset_path / "dataset_info.json").exists():
        return load_from_disk(str(dataset_path))

    resolved_type = ds_type or _infer_local_dataset_type(dataset_path)
    if resolved_type is None:
        raise ValueError(
            f"Could not infer prompt dataset type for {path!r}; pass --prompt-dataset-type "
            "(for example 'parquet', 'json', or 'hf_disk')."
        )
    if resolved_type == "hf_disk":
        return load_from_disk(str(dataset_path))
    if dataset_path.is_dir():
        return load_dataset(resolved_type, data_dir=str(dataset_path), split=split)
    return load_dataset(resolved_type, data_files=str(dataset_path), split=split)


def _iter_local_prompt_rows(path: Path, *, ds_type: str | None, split: str, column: str) -> Iterator[List[int]] | None:
    resolved_type = ds_type or _infer_local_dataset_type(path)
    if resolved_type == "parquet":
        import pyarrow.parquet as pq  # noqa: PLC0415

        def iter_parquet() -> Iterator[List[int]]:
            if path.is_dir():
                split_file = path / f"{split}.parquet"
                files = [split_file] if split_file.exists() else sorted(path.glob("*.parquet"))
            else:
                files = [path]
            try:
                batch_size = max(1, int(os.environ.get("OPD_PROMPT_PARQUET_BATCH_SIZE", "128")))
            except ValueError:
                batch_size = 128
            for file_path in files:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[column]):
                    for values in batch.column(0).to_pylist():
                        if values is not None:
                            yield values

        return iter_parquet()

    if resolved_type == "json" and path.is_file():

        def iter_json() -> Iterator[List[int]]:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    values = row.get(column)
                    if values is not None:
                        yield [int(token_id) for token_id in values]

        return iter_json()

    return None


def _load_prompts_from_dataset(
    *,
    path: str,
    ds_type: str | None,
    split: str,
    column: str,
    num_prompts: int,
    prompt_len: int,
    offset: int,
    turn_strategy: str = "prefix",
    im_start_token_id: int = 151644,
    im_end_token_id: int = 151645,
    assistant_role_token_ids: List[int] | None = None,
    min_target_tokens: int = 1,
    assistant_offset_tokens: int = 0,
) -> tuple[List[List[int]], List[Dict[str, Any]]]:
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if offset < 0:
        raise ValueError(f"prompt dataset offset must be non-negative, got {offset}")

    prompts: List[List[int]] = []
    prompt_metadata: List[Dict[str, Any]] = []
    assistant_role_token_ids = assistant_role_token_ids or [77091]

    for row_idx, token_ids in _iter_prompt_dataset_rows(path=path, ds_type=ds_type, split=split, column=column):
        if row_idx < offset or token_ids is None:
            continue
        prompt_result = _prompt_from_dataset_row(
            token_ids,
            prompt_len=prompt_len,
            turn_strategy=turn_strategy,
            im_start_token_id=im_start_token_id,
            im_end_token_id=im_end_token_id,
            assistant_role_token_ids=assistant_role_token_ids,
            min_target_tokens=min_target_tokens,
            assistant_offset_tokens=assistant_offset_tokens,
        )
        if prompt_result is None:
            continue
        prompt, metadata = prompt_result
        prompts.append(prompt)
        prompt_metadata.append(metadata)
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        raise ValueError(
            f"Prompt dataset {path!r} only yielded {len(prompts)} non-empty prompts from offset {offset}; "
            f"needed {num_prompts}."
        )
    return prompts, prompt_metadata


def _iter_prompt_dataset_rows(
    *,
    path: str,
    ds_type: str | None,
    split: str,
    column: str,
) -> Iterator[tuple[int, Any]]:
    local_rows = _iter_local_prompt_rows(Path(path), ds_type=ds_type, split=split, column=column)
    if local_rows is not None:
        yield from enumerate(local_rows)
        return

    dataset = _load_tokenized_prompt_dataset(path, ds_type=ds_type, split=split)
    for row_idx in range(len(dataset)):
        yield row_idx, dataset[row_idx].get(column)


def _find_assistant_content_spans(
    token_ids: List[int],
    *,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: List[int],
) -> List[tuple[int, int, int]]:
    """Return ``(turn_start, content_start, content_end)`` spans for assistant turns."""

    if not assistant_role_token_ids:
        raise ValueError("assistant_role_token_ids must be non-empty")
    spans: List[tuple[int, int, int]] = []
    role_len = len(assistant_role_token_ids)
    i = 0
    while i + 1 + role_len <= len(token_ids):
        if token_ids[i] != im_start_token_id or token_ids[i + 1 : i + 1 + role_len] != assistant_role_token_ids:
            i += 1
            continue
        content_start = i + 1 + role_len
        if content_start < len(token_ids) and token_ids[content_start] == 198:
            content_start += 1
        content_end = content_start
        while content_end < len(token_ids) and token_ids[content_end] != im_end_token_id:
            content_end += 1
        if content_end > content_start:
            spans.append((i, content_start, content_end))
        i = max(content_end + 1, i + 1)
    return spans


def _assistant_prompt_from_row(
    token_ids: List[int],
    *,
    prompt_len: int,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: List[int],
    min_target_tokens: int,
    assistant_offset_tokens: int,
) -> tuple[List[int], Dict[str, Any]] | None:
    if min_target_tokens <= 0:
        raise ValueError(f"min_target_tokens must be positive, got {min_target_tokens}")
    if assistant_offset_tokens < 0:
        raise ValueError(f"assistant_offset_tokens must be non-negative, got {assistant_offset_tokens}")

    spans = _find_assistant_content_spans(
        token_ids,
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        assistant_role_token_ids=assistant_role_token_ids,
    )
    for turn_start, content_start, content_end in reversed(spans):
        latest_prompt_end = content_end - min_target_tokens
        if latest_prompt_end < content_start:
            continue
        prompt_end = min(content_start + assistant_offset_tokens, latest_prompt_end)
        prompt_start = max(0, prompt_end - prompt_len)
        prompt = token_ids[prompt_start:prompt_end]
        if prompt:
            return prompt, {
                "assistant_turn_start": turn_start,
                "assistant_content_start": content_start,
                "assistant_content_end": content_end,
                "assistant_prompt_start": prompt_start,
                "assistant_prompt_end": prompt_end,
                "assistant_target_remaining": content_end - prompt_end,
            }
    return None


def _prompt_from_dataset_row(
    token_ids: Sequence[Any],
    *,
    prompt_len: int,
    turn_strategy: str,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: List[int],
    min_target_tokens: int,
    assistant_offset_tokens: int,
) -> tuple[List[int], Dict[str, Any]] | None:
    if turn_strategy == "prefix":
        prompt = [int(token_id) for token_id in token_ids[:prompt_len]]
        return (prompt, {}) if prompt else None
    if turn_strategy == "suffix":
        # Tail-align: keep the final ``prompt_len`` tokens. For datasets pre-extracted so each
        # row's prompt already ends at the assistant generation boundary (e.g.
        # coderforge_assistant_turns_*, whose ``prompt_ids`` terminate at
        # ``<|im_start|>assistant\n``), this yields an assistant-aligned prompt without
        # re-deriving turns — the student's first sampled token begins the assistant turn.
        prompt = [int(token_id) for token_id in token_ids[-prompt_len:]]
        return (prompt, {}) if prompt else None
    if turn_strategy == "assistant":
        full_token_ids = [int(token_id) for token_id in token_ids]
        return _assistant_prompt_from_row(
            full_token_ids,
            prompt_len=prompt_len,
            im_start_token_id=im_start_token_id,
            im_end_token_id=im_end_token_id,
            assistant_role_token_ids=assistant_role_token_ids,
            min_target_tokens=min_target_tokens,
            assistant_offset_tokens=assistant_offset_tokens,
        )
    raise ValueError(
        f"Unsupported prompt dataset turn strategy {turn_strategy!r}; expected 'prefix', 'suffix', or 'assistant'"
    )


def _load_prompt_pool_from_dataset(
    *,
    path: str,
    ds_type: str | None,
    split: str,
    column: str,
    prompt_len: int,
    turn_strategy: str = "prefix",
    im_start_token_id: int = 151644,
    im_end_token_id: int = 151645,
    assistant_role_token_ids: List[int] | None = None,
    min_target_tokens: int = 1,
    assistant_offset_tokens: int = 0,
    max_prompts: int | None = None,
) -> tuple[List[List[int]], List[int], List[Dict[str, Any]]]:
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError(f"max_prompts must be positive when set, got {max_prompts}")

    prompts: List[List[int]] = []
    row_indices: List[int] = []
    prompt_metadata: List[Dict[str, Any]] = []
    assistant_role_token_ids = assistant_role_token_ids or [77091]
    collect_metadata = turn_strategy == "assistant"
    for row_idx, token_ids in _iter_prompt_dataset_rows(path=path, ds_type=ds_type, split=split, column=column):
        if token_ids is None:
            continue
        prompt_result = _prompt_from_dataset_row(
            token_ids,
            prompt_len=prompt_len,
            turn_strategy=turn_strategy,
            im_start_token_id=im_start_token_id,
            im_end_token_id=im_end_token_id,
            assistant_role_token_ids=assistant_role_token_ids,
            min_target_tokens=min_target_tokens,
            assistant_offset_tokens=assistant_offset_tokens,
        )
        if prompt_result is None:
            continue
        prompt, metadata = prompt_result
        prompts.append(prompt)
        row_indices.append(row_idx)
        if collect_metadata:
            prompt_metadata.append(metadata)
        if len(prompts) % 10000 == 0:
            log.info("Loaded %d OPD prompts from %s split=%s column=%s", len(prompts), path, split, column)
        if max_prompts is not None and len(prompts) >= max_prompts:
            log.info(
                "Loaded %d OPD prompts from %s split=%s column=%s; stopping early for capped run",
                len(prompts),
                path,
                split,
                column,
            )
            break

    if not prompts:
        raise ValueError(f"Prompt dataset {path!r} yielded no non-empty prompts.")
    return prompts, row_indices, prompt_metadata


def _load_prompts(args: argparse.Namespace) -> tuple[List[List[int]], Dict[str, Any]]:
    if args.prompt_dataset_path:
        prompts, prompt_metadata = _load_prompts_from_dataset(
            path=args.prompt_dataset_path,
            ds_type=args.prompt_dataset_type,
            split=args.prompt_dataset_split,
            column=args.prompt_dataset_column,
            num_prompts=args.prompt_dataset_num_prompts,
            prompt_len=args.prompt_dataset_prompt_len,
            offset=args.prompt_dataset_offset,
            turn_strategy=getattr(args, "prompt_dataset_turn_strategy", "prefix"),
            im_start_token_id=getattr(args, "prompt_dataset_im_start_token_id", 151644),
            im_end_token_id=getattr(args, "prompt_dataset_im_end_token_id", 151645),
            assistant_role_token_ids=getattr(args, "prompt_dataset_assistant_role_token_ids", [77091]),
            min_target_tokens=getattr(args, "prompt_dataset_min_target_tokens", 1),
            assistant_offset_tokens=getattr(args, "prompt_dataset_assistant_offset_tokens", 0),
        )
        metadata: Dict[str, Any] = {
            key: [item[key] for item in prompt_metadata if key in item]
            for key in {
                "assistant_turn_start",
                "assistant_content_start",
                "assistant_content_end",
                "assistant_prompt_start",
                "assistant_prompt_end",
                "assistant_target_remaining",
            }
        }
        return prompts, {
            **metadata,
            "prompt_source": "dataset",
            "prompt_dataset_path": args.prompt_dataset_path,
            "prompt_dataset_type": args.prompt_dataset_type
            or _infer_local_dataset_type(Path(args.prompt_dataset_path)),
            "prompt_dataset_split": args.prompt_dataset_split,
            "prompt_dataset_column": args.prompt_dataset_column,
            "prompt_dataset_offset": args.prompt_dataset_offset,
            "prompt_dataset_prompt_len": args.prompt_dataset_prompt_len,
            "prompt_dataset_turn_strategy": getattr(args, "prompt_dataset_turn_strategy", "prefix"),
            "prompt_dataset_min_target_tokens": getattr(args, "prompt_dataset_min_target_tokens", 1),
            "prompt_dataset_assistant_offset_tokens": getattr(args, "prompt_dataset_assistant_offset_tokens", 0),
            "prompt_count": len(prompts),
            "prompt_lengths": [len(prompt) for prompt in prompts],
        }

    prompts = json.loads(args.prompts_json)
    if not isinstance(prompts, list) or not all(isinstance(prompt, list) for prompt in prompts):
        raise ValueError("--prompts-json must decode to a list of token-id lists")
    normalized = [[int(token_id) for token_id in prompt] for prompt in prompts]
    if not normalized or any(not prompt for prompt in normalized):
        raise ValueError("--prompts-json must contain at least one non-empty prompt")
    return normalized, {
        "prompt_source": "json",
        "prompt_count": len(normalized),
        "prompt_lengths": [len(prompt) for prompt in normalized],
    }


class _PromptSchedule:
    def __init__(
        self,
        *,
        mode: str,
        num_steps: int,
        static_prompts: List[List[int]] | None = None,
        static_metadata: Dict[str, Any] | None = None,
        prompt_pool: List[List[int]] | None = None,
        row_indices: List[int] | None = None,
        prompt_metadata: List[Dict[str, Any]] | None = None,
        base_metadata: Dict[str, Any] | None = None,
        batch_size: int = 0,
        epochs: int = 0,
        offset: int = 0,
    ) -> None:
        self.mode = mode
        self.num_steps = num_steps
        self.static_prompts = static_prompts
        self.static_metadata = static_metadata or {}
        self.prompt_pool = prompt_pool or []
        self.row_indices = row_indices or []
        self.prompt_metadata = prompt_metadata or []
        self.base_metadata = base_metadata or {}
        self.batch_size = batch_size
        self.epochs = epochs
        self.offset = offset
        self.total_examples = len(self.prompt_pool) * self.epochs if self.prompt_pool else 0

    def metadata(self) -> Dict[str, Any]:
        if self.mode == "static":
            return {
                **self.static_metadata,
                "prompt_schedule": "static",
                "prompt_schedule_total_steps": self.num_steps,
            }
        return {
            **self.base_metadata,
            "prompt_schedule": "dataset_epochs",
            "prompt_schedule_total_steps": self.num_steps,
            "prompt_dataset_epochs": self.epochs,
            "prompt_dataset_size": len(self.prompt_pool),
            "prompt_dataset_total_examples": self.total_examples,
            "prompt_dataset_batch_size": self.batch_size,
        }

    def batch_for_step(self, step: int) -> tuple[List[List[int]], Dict[str, Any]]:
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"Prompt schedule step {step} outside [0, {self.num_steps})")

        if self.mode == "static":
            prompts = self.static_prompts or []
            return prompts, {
                **self.static_metadata,
                "prompt_schedule": "static",
                "prompt_schedule_total_steps": self.num_steps,
            }

        dataset_size = len(self.prompt_pool)
        start = step * self.batch_size
        count = min(self.batch_size, self.total_examples - start)
        end = start + count
        pool_indices = [(self.offset + logical_idx) % dataset_size for logical_idx in range(start, end)]
        prompts = [self.prompt_pool[pool_idx] for pool_idx in pool_indices]
        source_row_indices = [self.row_indices[pool_idx] for pool_idx in pool_indices]
        assistant_metadata: Dict[str, Any] = {}
        if self.prompt_metadata:
            selected_metadata = [self.prompt_metadata[pool_idx] for pool_idx in pool_indices]
            for key in {
                "assistant_turn_start",
                "assistant_content_start",
                "assistant_content_end",
                "assistant_prompt_start",
                "assistant_prompt_end",
                "assistant_target_remaining",
            }:
                values = [item[key] for item in selected_metadata if key in item]
                if values:
                    assistant_metadata[key] = values
        return prompts, {
            **self.metadata(),
            **assistant_metadata,
            "prompt_count": len(prompts),
            "prompt_lengths": [len(prompt) for prompt in prompts],
            "prompt_dataset_global_start": start,
            "prompt_dataset_global_end": end,
            "prompt_dataset_epoch_start": start // dataset_size,
            "prompt_dataset_epoch_end": (end - 1) // dataset_size,
            "prompt_dataset_pool_indices": pool_indices,
            "prompt_dataset_row_indices": source_row_indices,
        }


def _build_prompt_schedule(args: argparse.Namespace) -> _PromptSchedule:
    if args.prompt_dataset_epochs < 0:
        raise ValueError(f"prompt dataset epochs must be non-negative, got {args.prompt_dataset_epochs}")

    if args.prompt_dataset_epochs == 0:
        if args.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {args.num_steps}")
        prompts, metadata = _load_prompts(args)
        return _PromptSchedule(
            mode="static",
            num_steps=args.num_steps,
            static_prompts=prompts,
            static_metadata=metadata,
        )

    if not args.prompt_dataset_path:
        raise ValueError("--prompt-dataset-epochs requires --prompt-dataset-path")
    if args.prompt_dataset_num_prompts <= 0:
        raise ValueError(f"prompt dataset batch size must be positive, got {args.prompt_dataset_num_prompts}")
    if args.prompt_dataset_offset < 0:
        raise ValueError(f"prompt dataset offset must be non-negative, got {args.prompt_dataset_offset}")

    max_prompt_pool_size = None
    max_opd_steps = getattr(args, "max_opd_steps", None)
    if max_opd_steps is not None:
        max_prompt_pool_size = args.prompt_dataset_offset + args.prompt_dataset_num_prompts * max_opd_steps

    prompts, row_indices, prompt_metadata = _load_prompt_pool_from_dataset(
        path=args.prompt_dataset_path,
        ds_type=args.prompt_dataset_type,
        split=args.prompt_dataset_split,
        column=args.prompt_dataset_column,
        prompt_len=args.prompt_dataset_prompt_len,
        turn_strategy=getattr(args, "prompt_dataset_turn_strategy", "prefix"),
        im_start_token_id=getattr(args, "prompt_dataset_im_start_token_id", 151644),
        im_end_token_id=getattr(args, "prompt_dataset_im_end_token_id", 151645),
        assistant_role_token_ids=getattr(args, "prompt_dataset_assistant_role_token_ids", [77091]),
        min_target_tokens=getattr(args, "prompt_dataset_min_target_tokens", 1),
        assistant_offset_tokens=getattr(args, "prompt_dataset_assistant_offset_tokens", 0),
        max_prompts=max_prompt_pool_size,
    )
    total_examples = len(prompts) * args.prompt_dataset_epochs
    batch_size = args.prompt_dataset_num_prompts
    num_steps = (total_examples + batch_size - 1) // batch_size
    base_metadata = {
        "prompt_source": "dataset",
        "prompt_dataset_path": args.prompt_dataset_path,
        "prompt_dataset_type": args.prompt_dataset_type or _infer_local_dataset_type(Path(args.prompt_dataset_path)),
        "prompt_dataset_split": args.prompt_dataset_split,
        "prompt_dataset_column": args.prompt_dataset_column,
        "prompt_dataset_offset": args.prompt_dataset_offset,
        "prompt_dataset_prompt_len": args.prompt_dataset_prompt_len,
        "prompt_dataset_turn_strategy": getattr(args, "prompt_dataset_turn_strategy", "prefix"),
        "prompt_dataset_min_target_tokens": getattr(args, "prompt_dataset_min_target_tokens", 1),
        "prompt_dataset_assistant_offset_tokens": getattr(args, "prompt_dataset_assistant_offset_tokens", 0),
    }
    return _PromptSchedule(
        mode="dataset_epochs",
        num_steps=num_steps,
        prompt_pool=prompts,
        row_indices=row_indices,
        prompt_metadata=prompt_metadata,
        base_metadata=base_metadata,
        batch_size=batch_size,
        epochs=args.prompt_dataset_epochs,
        offset=args.prompt_dataset_offset % len(prompts),
    )


class _PreparedOpdChunk:
    def __init__(
        self,
        *,
        chunk_idx: int,
        prompts: List[List[int]],
        sequences: List[List[int]],
        cache_path: Path,
        cache_indices: List[List[int]],
        data: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        sample_metadata: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.chunk_idx = chunk_idx
        self.prompts = prompts
        self.sequences = sequences
        self.cache_path = cache_path
        self.cache_indices = cache_indices
        self.data = data
        self.metrics = metrics
        self.sample_metadata = sample_metadata or [{} for _ in sequences]


def _normalize_generate_batch(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "outputs", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise TypeError(f"Unexpected SGLang /generate response type: {type(payload).__name__}")


def _output_ids(output: Dict[str, Any]) -> List[int]:
    completion = (
        output.get("output_ids") or output.get("token_ids") or output.get("meta_info", {}).get("output_ids") or []
    )
    if isinstance(completion, dict):
        completion = completion.get("output_ids", [])
    return [int(t) for t in completion]


def _top_logprob_token_id(entry: Any) -> int:
    if entry is None:
        raise RuntimeError("SGLang top-logprob entry is None")
    if isinstance(entry, dict):
        for key in ("token_id", "id"):
            if key in entry:
                return int(entry[key])
        raise RuntimeError(f"SGLang top-logprob dict does not contain a token id: {entry}")
    if not isinstance(entry, (list, tuple)) or not entry:
        raise RuntimeError(f"Unexpected SGLang top-logprob entry: {entry!r}")
    top = entry[0]
    if isinstance(top, dict):
        for key in ("token_id", "id"):
            if key in top:
                return int(top[key])
        raise RuntimeError(f"SGLang top-logprob dict does not contain a token id: {top}")
    if isinstance(top, (list, tuple)) and len(top) >= 2:
        return int(top[1])
    raise RuntimeError(f"Unexpected SGLang top-logprob item: {top!r}")


def _output_top_logprob_token_id(output: Dict[str, Any]) -> int:
    meta = output.get("meta_info", {}) if isinstance(output.get("meta_info"), dict) else {}
    output_top = meta.get("output_top_logprobs") or output.get("output_top_logprobs")
    if not output_top:
        raise RuntimeError("SGLang response did not contain output_ids or output_top_logprobs")
    return _top_logprob_token_id(output_top[0])


def _singleshot_mtp_rollout_sampling_enabled(singleshot_mtp: Dict[str, Any] | None) -> bool:
    if not singleshot_mtp:
        return False
    mode = str(singleshot_mtp.get("sampling_mode", "")).lower()
    if mode in {"native", "native_mtp", "sglang", "sglang_mtp", "sglang_native", "sglang_native_mtp"}:
        return False
    if mode in {"prompt_logprob", "prompt-logprob", "masked_prompt_logprob"}:
        return True
    for key in ("sample_rollout", "rollout_sampling", "student_rollout_sampling"):
        if bool(singleshot_mtp.get(key, False)):
            return True
    return mode in {"singleshot_mtp", "mtp"}


def _singleshot_mtp_native_sampling_enabled(singleshot_mtp: Dict[str, Any] | None) -> bool:
    if not singleshot_mtp:
        return False
    mode = str(singleshot_mtp.get("sampling_mode", "")).lower()
    return mode in {"native", "native_mtp", "sglang", "sglang_mtp", "sglang_native", "sglang_native_mtp"}


def _native_mtp_strategy_config(singleshot_mtp: Dict[str, Any]) -> tuple[Any | None, float | None]:
    threshold: float | None = None
    for key in ("mtp_conf_threshold", "confidence_threshold", "conf_adapt_threshold", "conf_threshold"):
        if singleshot_mtp.get(key) is not None:
            threshold = float(singleshot_mtp[key])
            break

    strategy = singleshot_mtp.get("mtp_strategy", singleshot_mtp.get("native_mtp_strategy"))
    if isinstance(strategy, str):
        raw = strategy.strip()
        lower = raw.lower()
        for sep in ("+", ":", "="):
            prefix = f"conf_adapt{sep}"
            if lower.startswith(prefix):
                threshold = float(raw[len(prefix) :])
                return ["conf_adapt", threshold], threshold
        if lower in {"conf_adapt", "none", ""}:
            return (None if lower in {"none", ""} else raw), threshold
    if isinstance(strategy, Sequence) and not isinstance(strategy, (str, bytes, bytearray)):
        items = list(strategy)
        if len(items) == 2 and isinstance(items[0], str) and items[0].strip().lower() == "conf_adapt":
            threshold = float(items[1])
            return ["conf_adapt", threshold], threshold
        return strategy, threshold

    if strategy is None and bool(singleshot_mtp.get("confidence_adaptive", False)):
        if threshold is None:
            raise ValueError("confidence_adaptive native MTP sampling requires a confidence threshold")
        return ["conf_adapt", threshold], threshold
    return strategy, threshold


def _native_sglang_mtp_sampling_params(
    singleshot_mtp: Dict[str, Any],
    sampling_params_extra: Dict[str, Any] | None,
) -> Dict[str, Any]:
    k_toks = int(singleshot_mtp.get("k_toks", 0))
    if k_toks < 1:
        raise ValueError("singleshot_mtp.k_toks must be set to a positive integer for native MTP sampling")

    params = dict(sampling_params_extra or {})
    params["mtp_enabled"] = True
    params["mtp_k"] = k_toks
    params.setdefault("top_k", 1)

    for source_key, target_key in (
        ("mask_token_id", "mtp_mask_id"),
        ("mtp_mask_id", "mtp_mask_id"),
        ("min_mask_token_id", "mtp_min_mask_id"),
        ("mtp_min_mask_id", "mtp_min_mask_id"),
        ("max_mask_token_id", "mtp_max_mask_id"),
        ("mtp_max_mask_id", "mtp_max_mask_id"),
        ("mtp_conf_threshold", "mtp_conf_threshold"),
        ("mtp_temperature", "mtp_temperature"),
        ("mtp_adaptive_window_mode", "mtp_adaptive_window_mode"),
    ):
        if source_key in singleshot_mtp and singleshot_mtp[source_key] is not None:
            params[target_key] = singleshot_mtp[source_key]

    strategy, threshold = _native_mtp_strategy_config(singleshot_mtp)
    if threshold is not None:
        params["mtp_conf_threshold"] = threshold
    if strategy is not None:
        params["mtp_strategy"] = strategy

    trace_enabled = _bool_config_value(
        singleshot_mtp,
        ("native_mtp_debug_trace", "mtp_debug_trace", "debug_trace", "record_native_mtp_debug_trace"),
        True,
    )
    if trace_enabled:
        custom_params = params.get("custom_params")
        if custom_params is not None and not isinstance(custom_params, dict):
            raise ValueError("native MTP sampling requires sampling_params.custom_params to be a JSON object")
        custom_params = dict(custom_params or {})
        custom_params["mtp_debug_trace"] = True
        custom_params["mtp_debug_max_steps"] = _int_config_value(
            singleshot_mtp,
            ("native_mtp_debug_max_steps", "mtp_debug_max_steps", "debug_max_steps"),
            4,
        )
        params["custom_params"] = custom_params

    if k_toks > 1 and params.get("mtp_mask_id") is None and params.get("mtp_min_mask_id") is None:
        raise ValueError("native MTP sampling requires mask_token_id/mtp_mask_id or mtp_min_mask_id when k_toks > 1")
    return params


def _min_native_mtp_debug_steps_for_rollout(max_new_tokens: int, k_toks: int) -> int:
    """Minimum native-MTP trace rows needed to replay a generated continuation."""

    if max_new_tokens <= 1:
        return 1
    k_toks = max(1, int(k_toks))
    # SGLang emits one AR bootstrap token outside the q>1 trace. The seed trace
    # row commits one more token; steady rows then commit up to k_toks tokens.
    return 1 + max(0, max_new_tokens - 2 + k_toks - 1) // k_toks


def _native_mtp_debug_trace_needs_partial_commit_budget(sampling_params: Dict[str, Any]) -> bool:
    strategy = sampling_params.get("mtp_strategy")
    if isinstance(strategy, str):
        return strategy.strip().lower().startswith("conf_adapt")
    if isinstance(strategy, Sequence) and not isinstance(strategy, (str, bytes, bytearray)):
        items = list(strategy)
        return bool(items) and isinstance(items[0], str) and items[0].strip().lower() == "conf_adapt"
    return False


def _ensure_native_mtp_debug_steps_cover_rollout(
    sampling_params: Dict[str, Any],
    *,
    max_new_tokens: int,
    k_toks: int,
) -> Dict[str, Any]:
    custom_params = sampling_params.get("custom_params")
    if not isinstance(custom_params, dict) or not custom_params.get("mtp_debug_trace"):
        return sampling_params

    if _native_mtp_debug_trace_needs_partial_commit_budget(sampling_params):
        required = max(1, int(max_new_tokens))
    else:
        required = _min_native_mtp_debug_steps_for_rollout(max_new_tokens, k_toks)
    current = int(custom_params.get("mtp_debug_max_steps", 0) or 0)
    if current >= required:
        return sampling_params

    adjusted = dict(sampling_params)
    adjusted_custom = dict(custom_params)
    adjusted_custom["mtp_debug_max_steps"] = required
    adjusted["custom_params"] = adjusted_custom
    return adjusted


def _generate_batch_outputs(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    r = requests.post(
        f"{student_url}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": _sampling_params(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                extra=sampling_params_extra,
            ),
            "return_hidden_states": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    outputs = _normalize_generate_batch(r.json())
    if len(outputs) != len(prompts):
        raise RuntimeError(f"SGLang returned {len(outputs)} outputs for {len(prompts)} prompts")
    return outputs


def _sequence_from_generate_output(prompt_ids: List[int], output: Dict[str, Any]) -> List[int]:
    return list(prompt_ids) + _output_ids(output)


def _native_mtp_sample_metadata(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for output in outputs:
        meta = output.get("meta_info", {}) if isinstance(output.get("meta_info"), dict) else {}
        trace = meta.get("mtp_debug_trace")
        if not isinstance(trace, list):
            trace = []
        metadata.append({"mtp_debug_trace": trace})
    return metadata


def _int_list(value: Any) -> List[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: List[int] = []
    for item in value:
        if isinstance(item, bool):
            result.append(int(item))
        elif isinstance(item, (int, float)):
            result.append(int(item))
    return result


def _float_list(value: Any) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: List[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            result.append(float(item))
    return result


def _native_mtp_step_commit_len(step: Dict[str, Any], *, k_toks: int) -> int:
    for key in ("commit_len_runtime", "commit_len"):
        value = step.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    commit_slice = step.get("commit_slice")
    if isinstance(commit_slice, dict) and isinstance(commit_slice.get("len"), (int, float)):
        return max(0, int(commit_slice["len"]))
    value = step.get("mtp_commit_len")
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 1 if str(step.get("phase", "")) == "seed" else max(1, int(k_toks))


def _native_mtp_step_committed_token_ids(step: Dict[str, Any]) -> List[int]:
    for key in ("committed_appended_to_output_ids", "committed_token_ids"):
        committed = _int_list(step.get(key))
        if committed:
            return committed
    return []


def _native_mtp_trace_replay_summary(
    generated_token_ids: Sequence[int],
    steps: Sequence[Dict[str, Any]],
    *,
    k_toks: int,
) -> Dict[str, Any]:
    generated = [int(token_id) for token_id in generated_token_ids]
    bootstrap_token_ids = generated[:1]
    trace_committed_token_ids: List[int] = []
    step_summaries: List[Dict[str, Any]] = []
    cursor = 1 if generated else 0

    for step_idx, step in enumerate(steps):
        commit_len = _native_mtp_step_commit_len(step, k_toks=k_toks)
        committed = _native_mtp_step_committed_token_ids(step)
        committed_slice = committed[:commit_len] if commit_len else []
        remaining = max(0, len(generated) - cursor)
        label_count = min(commit_len, remaining)
        trace_committed_token_ids.extend(committed_slice)
        step_summaries.append(
            {
                "step_idx": int(step.get("step_idx", step_idx))
                if isinstance(step.get("step_idx", step_idx), (int, float))
                else step_idx,
                "phase": step.get("phase"),
                "strategy_kind": step.get("strategy_kind"),
                "commit_len": commit_len,
                "label_count": label_count,
                "committed_token_ids": committed_slice,
                "pending_token_ids": _int_list(step.get("pending_token_ids")),
                "pending_confidences": _float_list(step.get("pending_confidences")),
                "conf_threshold": step.get("conf_threshold"),
            }
        )
        cursor += label_count

    replay_committed_token_ids = bootstrap_token_ids + trace_committed_token_ids
    covered_prefix = replay_committed_token_ids[: len(generated)]
    return {
        "bootstrap_token_ids": bootstrap_token_ids,
        "trace_committed_token_ids": trace_committed_token_ids,
        "replay_committed_token_ids": replay_committed_token_ids,
        "trace_unused_committed_token_count": max(0, len(replay_committed_token_ids) - len(generated)),
        "trace_covered_generated_tokens": len(replay_committed_token_ids) >= len(generated)
        and covered_prefix == generated,
        "trace_commit_lens": [int(step["commit_len"]) for step in step_summaries],
        "trace_label_counts": [int(step["label_count"]) for step in step_summaries],
        "trace_steps": step_summaries,
    }


def _native_mtp_trace_metrics(
    sample_metadata: List[Dict[str, Any]],
    *,
    trace_requested: bool,
    generated_token_ids: Sequence[Sequence[int]] | None = None,
    k_toks: int = 1,
) -> Dict[str, Any]:
    traces = [item.get("mtp_debug_trace") for item in sample_metadata]
    traces = [trace for trace in traces if isinstance(trace, list)]
    nonempty_traces = [trace for trace in traces if trace]
    steps = [step for trace in nonempty_traces for step in trace if isinstance(step, dict)]

    q_lens = sorted(
        {int(step["decode_q_len_per_req"]) for step in steps if isinstance(step.get("decode_q_len_per_req"), int)}
    )
    phases = sorted({str(step["phase"]) for step in steps if step.get("phase") is not None})
    effective_ks = [
        int(step["effective_k_runtime"] if step.get("effective_k_runtime") is not None else step["effective_k"])
        for step in steps
        if step.get("effective_k_runtime") is not None or step.get("effective_k") is not None
    ]
    can_run_graph = [
        bool(step["can_run_cuda_graph"]) for step in steps if isinstance(step.get("can_run_cuda_graph"), bool)
    ]
    commit_lens = [_native_mtp_step_commit_len(step, k_toks=k_toks) for step in steps]
    seed_commit_lens = [
        _native_mtp_step_commit_len(step, k_toks=k_toks) for step in steps if str(step.get("phase", "")) == "seed"
    ]
    steady_commit_lens = [
        _native_mtp_step_commit_len(step, k_toks=k_toks) for step in steps if str(step.get("phase", "")) == "steady"
    ]
    pending_confidences = [confidence for step in steps for confidence in _float_list(step.get("pending_confidences"))]

    metrics: Dict[str, Any] = {
        "student_sampling_mtp_debug_trace_requested": trace_requested,
        "student_sampling_mtp_debug_trace_samples": len(nonempty_traces),
        "student_sampling_mtp_debug_trace_steps": len(steps),
    }
    if q_lens:
        metrics["student_sampling_mtp_debug_trace_q_lens"] = ",".join(str(q_len) for q_len in q_lens)
        metrics["student_sampling_mtp_debug_trace_max_q_len"] = max(q_lens)
    if phases:
        metrics["student_sampling_mtp_debug_trace_phases"] = ",".join(phases)
        metrics["student_sampling_mtp_debug_trace_seen_seed"] = "seed" in phases
        metrics["student_sampling_mtp_debug_trace_seen_steady"] = "steady" in phases
    if effective_ks:
        metrics["student_sampling_mtp_debug_trace_effective_k_mean"] = sum(effective_ks) / len(effective_ks)
        metrics["student_sampling_mtp_debug_trace_effective_k_min"] = min(effective_ks)
        metrics["student_sampling_mtp_debug_trace_effective_k_max"] = max(effective_ks)
    if can_run_graph:
        metrics["student_sampling_mtp_debug_trace_cuda_graph_steps"] = sum(1 for value in can_run_graph if value)
        metrics["student_sampling_mtp_debug_trace_cuda_graph_all"] = all(can_run_graph)
    if commit_lens:
        metrics["student_sampling_mtp_debug_trace_commit_len_mean"] = sum(commit_lens) / len(commit_lens)
        metrics["student_sampling_mtp_debug_trace_commit_len_min"] = min(commit_lens)
        metrics["student_sampling_mtp_debug_trace_commit_len_max"] = max(commit_lens)
        metrics["student_sampling_mtp_debug_trace_committed_tokens"] = sum(commit_lens)
    if seed_commit_lens:
        metrics["student_sampling_mtp_debug_trace_seed_steps"] = len(seed_commit_lens)
        metrics["student_sampling_mtp_debug_trace_commit_len_seed"] = sum(seed_commit_lens) / len(seed_commit_lens)
    if steady_commit_lens:
        metrics["student_sampling_mtp_debug_trace_steady_steps"] = len(steady_commit_lens)
        metrics["student_sampling_mtp_debug_trace_commit_len_steady"] = sum(steady_commit_lens) / len(
            steady_commit_lens
        )
    if pending_confidences:
        metrics["student_sampling_mtp_debug_trace_pending_confidence_mean"] = sum(pending_confidences) / len(
            pending_confidences
        )
        metrics["student_sampling_mtp_debug_trace_pending_confidence_min"] = min(pending_confidences)

    if generated_token_ids is not None:
        replay_summaries = []
        for item, generated in zip(sample_metadata, generated_token_ids):
            trace = item.get("mtp_debug_trace") if isinstance(item, dict) else None
            trace_steps = [step for step in trace if isinstance(step, dict)] if isinstance(trace, list) else []
            replay_summaries.append(_native_mtp_trace_replay_summary(generated, trace_steps, k_toks=k_toks))
        generated_count = sum(len(list(tokens)) for tokens in generated_token_ids)
        bootstrap_count = sum(1 for tokens in generated_token_ids if len(list(tokens)) > 0)
        replay_committed_count = sum(len(summary["replay_committed_token_ids"]) for summary in replay_summaries)
        unused_count = sum(int(summary["trace_unused_committed_token_count"]) for summary in replay_summaries)
        covered_values = [bool(summary["trace_covered_generated_tokens"]) for summary in replay_summaries]
        metrics["student_sampling_mtp_generated_tokens"] = generated_count
        metrics["student_sampling_mtp_supervised_tokens_expected"] = generated_count
        metrics["student_sampling_mtp_bootstrap_tokens"] = bootstrap_count
        metrics["student_sampling_mtp_replay_committed_tokens"] = replay_committed_count
        metrics["student_sampling_mtp_trace_tokens_unused"] = unused_count
        if covered_values:
            metrics["student_sampling_mtp_replay_trace_covered_all_targets"] = all(covered_values)
    return metrics


def _extract_singleshot_mtp_block_tokens(
    output: Dict[str, Any],
    *,
    effective_k: int,
) -> List[int]:
    if effective_k < 1:
        raise ValueError(f"effective_k must be >= 1, got {effective_k}")

    output_ids = _output_ids(output)
    if effective_k == 1:
        if output_ids:
            return [output_ids[0]]
        return [_output_top_logprob_token_id(output)]

    meta = output.get("meta_info", {}) if isinstance(output.get("meta_info"), dict) else {}
    input_top = meta.get("input_top_logprobs") or output.get("input_top_logprobs")
    if input_top is None:
        raise RuntimeError("SGLang MTP rollout response is missing input_top_logprobs")
    if len(input_top) == effective_k:
        input_top = input_top[1:]
    elif len(input_top) != effective_k - 1:
        raise RuntimeError(
            "Unexpected input_top_logprobs length for MTP rollout: "
            f"got={len(input_top)} expected={effective_k} or {effective_k - 1}"
        )

    tokens = [_top_logprob_token_id(entry) for entry in input_top]
    if output_ids:
        tokens.append(output_ids[0])
    else:
        tokens.append(_output_top_logprob_token_id(output))
    if len(tokens) != effective_k:
        raise RuntimeError(f"Extracted {len(tokens)} MTP tokens for effective_k={effective_k}: {tokens}")
    return tokens


def _student_sample_singleshot_mtp_batch(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    *,
    singleshot_mtp: Dict[str, Any],
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> tuple[List[List[int]], Dict[str, Any]]:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    k_toks = int(singleshot_mtp.get("k_toks", 0))
    if k_toks < 1:
        raise ValueError("singleshot_mtp.k_toks must be set to a positive integer for rollout sampling")
    mask_token_id = singleshot_mtp.get("mask_token_id")
    if k_toks > 1 and mask_token_id is None:
        raise ValueError("singleshot_mtp.mask_token_id is required for MTP rollout sampling when k_toks > 1")
    mask_token_id = int(mask_token_id) if mask_token_id is not None else None

    sequences = [list(prompt) for prompt in prompts]
    generated_counts = [0 for _ in prompts]
    generate_calls = 0
    block_count = 0
    partial_block_count = 0

    while any(count < max_new_tokens for count in generated_counts):
        active_indices = [i for i, count in enumerate(generated_counts) if count < max_new_tokens]
        groups: Dict[int, List[int]] = {}
        for idx in active_indices:
            remaining = max_new_tokens - generated_counts[idx]
            effective_k = min(k_toks, remaining)
            groups.setdefault(effective_k, []).append(idx)

        for effective_k, group_indices in groups.items():
            request_inputs: List[List[int]] = []
            logprob_start_lens: List[int] = []
            for idx in group_indices:
                prefix = sequences[idx]
                if not prefix:
                    raise ValueError("MTP rollout sampling requires non-empty prompts")
                masks = [mask_token_id] * (effective_k - 1) if mask_token_id is not None else []
                request_inputs.append(prefix + masks)
                logprob_start_lens.append(max(0, len(prefix) - 1))

            logprob_start_len: int | List[int]
            if all(x == logprob_start_lens[0] for x in logprob_start_lens):
                logprob_start_len = logprob_start_lens[0]
            else:
                logprob_start_len = logprob_start_lens

            r = requests.post(
                f"{student_url}/generate",
                json={
                    "input_ids": request_inputs,
                    "sampling_params": _sampling_params(
                        max_new_tokens=1,
                        temperature=temperature,
                        extra=sampling_params_extra,
                    ),
                    "return_logprob": True,
                    "logprob_start_len": logprob_start_len,
                    "top_logprobs_num": 1,
                    "return_hidden_states": False,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            generate_calls += 1
            outputs = _normalize_generate_batch(r.json())
            if len(outputs) != len(group_indices):
                raise RuntimeError(f"SGLang returned {len(outputs)} outputs for {len(group_indices)} MTP prompts")

            for idx, output in zip(group_indices, outputs):
                tokens = _extract_singleshot_mtp_block_tokens(output, effective_k=effective_k)
                remaining = max_new_tokens - generated_counts[idx]
                tokens = tokens[:remaining]
                sequences[idx].extend(tokens)
                generated_counts[idx] += len(tokens)
                block_count += 1
                if len(tokens) < k_toks:
                    partial_block_count += 1

    return sequences, {
        "student_sampling_mode": "singleshot_mtp_prompt_logprob",
        "student_sampling_temperature": float(temperature),
        "student_sampling_top_k": int((sampling_params_extra or {}).get("top_k", -1) or -1),
        "student_sampling_top_p": float((sampling_params_extra or {}).get("top_p", 1.0) or 1.0),
        "student_sampling_min_p": float((sampling_params_extra or {}).get("min_p", 0.0) or 0.0),
        "student_sampling_argmax_rollout": _sampling_params_are_argmax(
            temperature=temperature,
            sampling_params=sampling_params_extra,
        ),
        "student_sampling_mtp_k_toks": k_toks,
        "student_sampling_mtp_mask_token_id": mask_token_id if mask_token_id is not None else -1,
        "student_sampling_mtp_blocks": block_count,
        "student_sampling_mtp_generate_calls": generate_calls,
        "student_sampling_mtp_partial_blocks": partial_block_count,
    }


def _model_info(base_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    r = requests.get(f"{base_url}/model_info", timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unexpected /model_info response type: {type(payload).__name__}")


def _verify_student_weight_version(
    student_url: str,
    expected_weight_version: str,
    profile_row: Dict[str, Any],
    timeout: float = 30.0,
) -> tuple[bool, str | None]:
    try:
        model_info = _model_info(student_url, timeout=timeout)
    except Exception as exc:
        message = f"Could not fetch SGLang /model_info after sync: {exc}"
        profile_row["student_weight_version_error"] = message
        return False, message

    actual_weight_version = model_info.get("weight_version")
    profile_row["student_weight_version"] = actual_weight_version
    if actual_weight_version != expected_weight_version:
        return (
            False,
            "SGLang weight_version did not match expected sync version: "
            f"expected={expected_weight_version} actual={actual_weight_version}",
        )
    return True, None


def _student_sample_batch(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> List[List[int]]:
    outputs = _generate_batch_outputs(
        student_url,
        prompts,
        max_new_tokens,
        temperature=temperature,
        timeout=timeout,
        sampling_params_extra=sampling_params_extra,
    )
    return [_sequence_from_generate_output(prompt_ids, output) for prompt_ids, output in zip(prompts, outputs)]


def _student_sample(
    student_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> List[int]:
    return _student_sample_batch(
        student_url,
        [prompt_ids],
        max_new_tokens,
        temperature=temperature,
        timeout=timeout,
        sampling_params_extra=sampling_params_extra,
    )[0]


def _student_sample_native_mtp_batch(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    *,
    singleshot_mtp: Dict[str, Any],
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> tuple[List[List[int]], Dict[str, Any]]:
    k_toks = int(singleshot_mtp.get("k_toks", 0))
    sampling_params = _native_sglang_mtp_sampling_params(singleshot_mtp, sampling_params_extra)
    sampling_params = _ensure_native_mtp_debug_steps_cover_rollout(
        sampling_params,
        max_new_tokens=max_new_tokens,
        k_toks=k_toks,
    )
    outputs = _generate_batch_outputs(
        student_url,
        prompts,
        max_new_tokens,
        temperature=temperature,
        timeout=timeout,
        sampling_params_extra=sampling_params,
    )
    sequences = [_sequence_from_generate_output(prompt, output) for prompt, output in zip(prompts, outputs)]
    generated_by_sample = [sequence[len(prompt) :] for sequence, prompt in zip(sequences, prompts)]
    output_lens = [max(0, len(sequence) - len(prompt)) for sequence, prompt in zip(sequences, prompts)]
    block_count = sum((output_len + k_toks - 1) // k_toks for output_len in output_lens)
    partial_block_count = sum(1 for output_len in output_lens if output_len > 0 and output_len % k_toks != 0)
    mask_token_id = sampling_params.get("mtp_mask_id", singleshot_mtp.get("mask_token_id"))
    sample_metadata = _native_mtp_sample_metadata(outputs)
    trace_requested = bool(
        isinstance(sampling_params.get("custom_params"), dict)
        and sampling_params["custom_params"].get("mtp_debug_trace")
    )
    metrics = {
        "student_sampling_mode": "sglang_native_mtp",
        "student_sampling_temperature": float(temperature),
        "student_sampling_top_k": int(sampling_params.get("top_k", -1) or -1),
        "student_sampling_top_p": float(sampling_params.get("top_p", 1.0) or 1.0),
        "student_sampling_min_p": float(sampling_params.get("min_p", 0.0) or 0.0),
        "student_sampling_argmax_rollout": _sampling_params_are_argmax(
            temperature=temperature,
            sampling_params=sampling_params,
        ),
        "student_sampling_mtp_k_toks": k_toks,
        "student_sampling_mtp_mask_token_id": int(mask_token_id) if mask_token_id is not None else -1,
        "student_sampling_mtp_blocks": block_count,
        "student_sampling_mtp_generate_calls": 1,
        "student_sampling_mtp_partial_blocks": partial_block_count,
        "student_sampling_mtp_native": True,
        "_student_sample_metadata": sample_metadata,
    }
    if sampling_params.get("mtp_strategy") is not None:
        metrics["student_sampling_mtp_strategy"] = json.dumps(sampling_params["mtp_strategy"], sort_keys=True)
    if sampling_params.get("mtp_conf_threshold") is not None:
        metrics["student_sampling_mtp_conf_threshold"] = float(sampling_params["mtp_conf_threshold"])
    if sampling_params.get("mtp_adaptive_window_mode") is not None:
        metrics["student_sampling_mtp_adaptive_window_mode"] = str(sampling_params["mtp_adaptive_window_mode"])
    if isinstance(sampling_params.get("custom_params"), dict):
        metrics["student_sampling_mtp_debug_max_steps"] = int(
            sampling_params["custom_params"].get("mtp_debug_max_steps", 0) or 0
        )
    metrics.update(
        _native_mtp_trace_metrics(
            sample_metadata,
            trace_requested=trace_requested,
            generated_token_ids=generated_by_sample,
            k_toks=k_toks,
        )
    )
    return sequences, metrics


def _check_rollout_coherence(
    prompts: List[List[int]],
    sequences: List[List[int]],
    *,
    degenerate_token_ids: set[int],
    max_degenerate_frac: float,
    action: str,
    min_generated_tokens: int = 8,
    context: str = "",
) -> Dict[str, Any]:
    """Guard against degenerate sampler rollouts (e.g. token-0 collapse).

    A frozen-teacher OPD loss stays low on garbage rollouts (the teacher
    "agrees" with the student on the corrupt context), so loss/agreement never
    catch this. Fail loudly here instead of training on garbage.
    """
    degenerate_rollouts = 0
    worst_frac = 0.0
    checked = 0
    for prompt, sequence in zip(prompts, sequences):
        generated = sequence[len(prompt) :]
        if len(generated) < min_generated_tokens:
            continue
        checked += 1
        frac = sum(1 for t in generated if t in degenerate_token_ids) / len(generated)
        worst_frac = max(worst_frac, frac)
        if frac > max_degenerate_frac:
            degenerate_rollouts += 1
    metrics = {
        "rollout_coherence_checked": checked,
        "rollout_coherence_degenerate": degenerate_rollouts,
        "rollout_coherence_worst_degenerate_frac": worst_frac,
    }
    if degenerate_rollouts > 0 and action != "off":
        message = (
            f"Rollout coherence gate: {degenerate_rollouts}/{checked} rollouts have "
            f">{max_degenerate_frac:.0%} degenerate tokens (ids {sorted(degenerate_token_ids)}, "
            f"worst {worst_frac:.0%}){f' [{context}]' if context else ''}. "
            "The student sampler is emitting garbage (see "
            "docs/notes/mtp_student_sampler_token0_collapse_handoff.md); training on these "
            "rollouts is vacuous self-distillation."
        )
        if action == "fail":
            raise RuntimeError(message)
        log.error(message)
    return metrics


def _student_sample_for_opd_batch(
    student_url: str,
    prompts: List[List[int]],
    max_new_tokens: int,
    *,
    singleshot_mtp: Dict[str, Any] | None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    sampling_params_extra: Dict[str, Any] | None = None,
) -> tuple[List[List[int]], Dict[str, Any]]:
    if _singleshot_mtp_native_sampling_enabled(singleshot_mtp):
        assert singleshot_mtp is not None
        return _student_sample_native_mtp_batch(
            student_url,
            prompts,
            max_new_tokens,
            singleshot_mtp=singleshot_mtp,
            temperature=temperature,
            timeout=timeout,
            sampling_params_extra=sampling_params_extra,
        )
    if _singleshot_mtp_rollout_sampling_enabled(singleshot_mtp):
        assert singleshot_mtp is not None
        return _student_sample_singleshot_mtp_batch(
            student_url,
            prompts,
            max_new_tokens,
            singleshot_mtp=singleshot_mtp,
            temperature=temperature,
            timeout=timeout,
            sampling_params_extra=sampling_params_extra,
        )
    return (
        _student_sample_batch(
            student_url,
            prompts,
            max_new_tokens,
            temperature=temperature,
            timeout=timeout,
            sampling_params_extra=sampling_params_extra,
        ),
        {
            "student_sampling_mode": "ar",
            "student_sampling_temperature": float(temperature),
            "student_sampling_top_k": int((sampling_params_extra or {}).get("top_k", -1) or -1),
            "student_sampling_top_p": float((sampling_params_extra or {}).get("top_p", 1.0) or 1.0),
            "student_sampling_min_p": float((sampling_params_extra or {}).get("min_p", 0.0) or 0.0),
        },
    )


def _load_rollout_tokenizer(tokenizer_path: str | None) -> Any | None:
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        log.warning("Cannot decode rollout samples because transformers is not installed.")
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on local HF cache shape
        log.warning("Could not load rollout sample tokenizer from %s: %s", tokenizer_path, exc)
        return None


def _tail_token_ids(token_ids: List[int], max_tokens: int) -> List[int]:
    if max_tokens <= 0:
        return []
    return list(token_ids[-max_tokens:])


def _head_token_ids(token_ids: List[int], max_tokens: int) -> List[int]:
    if max_tokens <= 0:
        return []
    return list(token_ids[:max_tokens])


def _decode_token_ids(tokenizer: Any | None, token_ids: List[int], *, max_chars: int) -> str | None:
    if tokenizer is None:
        return None
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception as exc:  # pragma: no cover - tokenizer-dependent
        return f"<decode failed: {exc}>"
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _sample_metadata_at_step(step_metadata: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    sample_metadata: Dict[str, Any] = {}
    for key, value in step_metadata.items():
        if isinstance(value, list) and sample_idx < len(value):
            item = value[sample_idx]
            if isinstance(item, (int, float, bool, str)):
                sample_metadata[key] = item
    return sample_metadata


def _max_repeated_token_run(token_ids: Sequence[int]) -> int:
    if not token_ids:
        return 0
    max_run = 1
    current_run = 1
    previous = int(token_ids[0])
    for token_id in token_ids[1:]:
        current = int(token_id)
        if current == previous:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
            previous = current
    return max(max_run, current_run)


def _tokenizer_special_token_ids(tokenizer: Any | None) -> set[int]:
    if tokenizer is None:
        return set()
    special_ids: set[int] = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is not None:
            special_ids.add(int(token_id))
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def _tokenizer_eos_token_ids(tokenizer: Any | None) -> set[int]:
    if tokenizer is None:
        return set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, Sequence) and not isinstance(eos, (str, bytes, bytearray)):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def _decoded_token_text(tokenizer: Any | None, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return None


def _rollout_health_for_tokens(
    token_ids: Sequence[int],
    *,
    tokenizer: Any | None,
    generated_text: str | None,
) -> Dict[str, Any]:
    tokens = [int(token_id) for token_id in token_ids]
    token_count = len(tokens)
    special_ids = _tokenizer_special_token_ids(tokenizer)
    eos_ids = _tokenizer_eos_token_ids(tokenizer)
    decoded_tokens = [_decoded_token_text(tokenizer, token_id) for token_id in tokens] if tokenizer is not None else []

    okay_count = 0
    digit_count = 0
    newline_count = 0
    for text in decoded_tokens:
        if text is None:
            continue
        stripped = text.strip().lower()
        if stripped == "okay":
            okay_count += 1
        if stripped.isdigit() and stripped != "":
            digit_count += 1
        if "\n" in text:
            newline_count += 1

    return {
        "rollout_token_count": token_count,
        "rollout_unique_token_count": len(set(tokens)),
        "rollout_max_repeated_token_run": _max_repeated_token_run(tokens),
        "rollout_okay_token_count": okay_count,
        "rollout_digit_token_count": digit_count,
        "rollout_newline_token_count": newline_count,
        "rollout_special_token_count": sum(1 for token_id in tokens if token_id in special_ids),
        "rollout_eos_token_count": sum(1 for token_id in tokens if token_id in eos_ids),
        "rollout_generated_char_count": len(generated_text or ""),
    }


def _finalize_rollout_health_metrics(raw: Dict[str, Any], *, examples: int) -> Dict[str, Any]:
    token_count = int(raw.get("token_count", 0) or 0)
    repeated_sum = float(raw.get("repeated_sum", 0.0) or 0.0)
    char_sum = float(raw.get("char_sum", 0.0) or 0.0)
    metrics: Dict[str, Any] = {
        "rollout/examples_written": examples,
        "rollout/token_count": token_count,
        "rollout/max_repeated_token_run": int(raw.get("max_repeated", 0) or 0),
        "rollout/mean_repeated_token_run": repeated_sum / examples if examples else 0.0,
        "rollout/samples_with_repeat_run_ge_16": int(raw.get("repeat_ge_16", 0) or 0),
        "rollout/unique_token_fraction": (float(raw.get("unique_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/okay_token_rate": (float(raw.get("okay_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/digit_token_rate": (float(raw.get("digit_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/newline_token_rate": (float(raw.get("newline_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/special_token_rate": (float(raw.get("special_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/eos_rate": (float(raw.get("eos_count", 0) or 0) / token_count) if token_count else 0.0,
        "rollout/avg_chars": char_sum / examples if examples else 0.0,
    }
    return metrics


def _rollout_health_metrics_for_sequences(
    prompts: Sequence[Sequence[int]],
    sequences: Sequence[Sequence[int]],
    *,
    tokenizer: Any | None,
    text_max_chars: int,
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "token_count": 0,
        "unique_count": 0,
        "okay_count": 0,
        "digit_count": 0,
        "newline_count": 0,
        "special_count": 0,
        "eos_count": 0,
        "char_sum": 0,
        "repeated_sum": 0,
        "max_repeated": 0,
        "repeat_ge_16": 0,
    }
    examples = 0
    for prompt, sequence in zip(prompts, sequences):
        generated = [int(token_id) for token_id in sequence[len(prompt) :]]
        generated_text = _decode_token_ids(tokenizer, generated, max_chars=text_max_chars)
        health = _rollout_health_for_tokens(generated, tokenizer=tokenizer, generated_text=generated_text)
        examples += 1
        raw["token_count"] += int(health["rollout_token_count"])
        raw["unique_count"] += int(health["rollout_unique_token_count"])
        raw["okay_count"] += int(health["rollout_okay_token_count"])
        raw["digit_count"] += int(health["rollout_digit_token_count"])
        raw["newline_count"] += int(health["rollout_newline_token_count"])
        raw["special_count"] += int(health["rollout_special_token_count"])
        raw["eos_count"] += int(health["rollout_eos_token_count"])
        raw["char_sum"] += int(health["rollout_generated_char_count"])
        raw["repeated_sum"] += int(health["rollout_max_repeated_token_run"])
        raw["max_repeated"] = max(int(raw["max_repeated"]), int(health["rollout_max_repeated_token_run"]))
        if int(health["rollout_max_repeated_token_run"]) >= 16:
            raw["repeat_ge_16"] += 1
    return _finalize_rollout_health_metrics(raw, examples=examples)


def _aggregate_rollout_health_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "token_count": int(_sum_metric(rows, "rollout/token_count")),
        "unique_count": 0,
        "okay_count": 0,
        "digit_count": 0,
        "newline_count": 0,
        "special_count": 0,
        "eos_count": 0,
        "char_sum": 0,
        "repeated_sum": 0,
        "max_repeated": 0,
        "repeat_ge_16": int(_sum_metric(rows, "rollout/samples_with_repeat_run_ge_16")),
    }
    examples = int(_sum_metric(rows, "rollout/examples_written"))
    for row in rows:
        token_count = int(row.get("rollout/token_count", 0) or 0)
        examples_row = int(row.get("rollout/examples_written", 0) or 0)
        raw["unique_count"] += float(row.get("rollout/unique_token_fraction", 0.0) or 0.0) * token_count
        raw["okay_count"] += float(row.get("rollout/okay_token_rate", 0.0) or 0.0) * token_count
        raw["digit_count"] += float(row.get("rollout/digit_token_rate", 0.0) or 0.0) * token_count
        raw["newline_count"] += float(row.get("rollout/newline_token_rate", 0.0) or 0.0) * token_count
        raw["special_count"] += float(row.get("rollout/special_token_rate", 0.0) or 0.0) * token_count
        raw["eos_count"] += float(row.get("rollout/eos_rate", 0.0) or 0.0) * token_count
        raw["char_sum"] += float(row.get("rollout/avg_chars", 0.0) or 0.0) * examples_row
        raw["repeated_sum"] += float(row.get("rollout/mean_repeated_token_run", 0.0) or 0.0) * examples_row
        raw["max_repeated"] = max(int(raw["max_repeated"]), int(row.get("rollout/max_repeated_token_run", 0) or 0))
    return _finalize_rollout_health_metrics(raw, examples=examples)


def _rollout_sample_invariant_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    generated_matches = [
        bool(record["generated_matches_supervised_target"])
        for record in records
        if isinstance(record.get("generated_matches_supervised_target"), bool)
    ]
    if generated_matches:
        metrics["rollout/generated_matches_supervised_target_all"] = all(generated_matches)
        metrics["rollout/generated_supervised_mismatch_count"] = sum(1 for value in generated_matches if not value)

    trace_covered = [
        bool(record["native_mtp_trace_covered_generated_tokens"])
        for record in records
        if isinstance(record.get("native_mtp_trace_covered_generated_tokens"), bool)
    ]
    if trace_covered:
        metrics["rollout/native_trace_covered_generated_all"] = all(trace_covered)
        metrics["rollout/native_trace_coverage_failure_count"] = sum(1 for value in trace_covered if not value)
    if records:
        metrics["rollout/native_trace_unused_committed_token_count"] = sum(
            int(record.get("native_mtp_trace_unused_committed_token_count", 0) or 0)
            for record in records
            if isinstance(record.get("native_mtp_trace_unused_committed_token_count", 0), (int, float))
        )
    return metrics


def _int_from_config(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    return int(value)


def _scalar_int_from_loss_input(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])
    return int(value)


def _mtp_training_selection(
    data: Dict[str, Any],
    singleshot_mtp: Dict[str, Any] | None,
    *,
    ignore_index: int = -100,
) -> Dict[str, Any]:
    if not singleshot_mtp:
        return {}
    if "k_toks" not in singleshot_mtp:
        return {"mtp_selection_error": "singleshot_mtp.k_toks is not set"}

    try:
        from xorl.mtp.singleshot import prepare_singleshot_mtp_batch, prepare_singleshot_mtp_opd_batch  # noqa: PLC0415
    except ImportError as exc:
        repo_src = Path(__file__).resolve().parents[2] / "src"
        if repo_src.exists() and str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
            xorl_pkg = sys.modules.get("xorl")
            xorl_pkg_path = repo_src / "xorl"
            if xorl_pkg is not None and hasattr(xorl_pkg, "__path__") and str(xorl_pkg_path) not in xorl_pkg.__path__:
                xorl_pkg.__path__ = [str(xorl_pkg_path), *list(xorl_pkg.__path__)]
            importlib.invalidate_caches()
            try:
                from xorl.mtp.singleshot import (  # noqa: PLC0415
                    prepare_singleshot_mtp_batch,
                    prepare_singleshot_mtp_opd_batch,
                )
            except ImportError as retry_exc:
                return {"mtp_selection_error": f"could not import SingleShot MTP helper: {retry_exc}"}
        else:
            return {"mtp_selection_error": f"could not import SingleShot MTP helper: {exc}"}

    input_ids = list(data["model_input"]["input_ids"])
    target_tokens = list(data["loss_fn_inputs"]["target_tokens"])
    source_len = len(input_ids)
    truncation_length = singleshot_mtp.get("truncation_length")
    truncation_side = str(singleshot_mtp.get("truncation_side", "left")).lower()
    window_start = max(0, source_len - int(truncation_length)) if truncation_side == "tail" and truncation_length else 0
    loss_start = _scalar_int_from_loss_input(
        data["loss_fn_inputs"].get(str(singleshot_mtp.get("loss_start_key", "mtp_loss_start"))),
        0,
    )

    if bool(singleshot_mtp.get("rollout_replay", False)):
        try:
            prepared = prepare_singleshot_mtp_opd_batch(
                {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "target_tokens": torch.tensor([target_tokens], dtype=torch.long),
                    str(singleshot_mtp.get("loss_start_key", "mtp_loss_start")): torch.tensor([loss_start]),
                },
                k_toks=int(singleshot_mtp["k_toks"]),
                mask_token_id=singleshot_mtp.get("mask_token_id"),
                pad_token_id=singleshot_mtp.get("pad_token_id"),
                pad_to_multiple=singleshot_mtp.get("pad_to_multiple"),
                static_padded_seq_len=singleshot_mtp.get("static_padded_seq_len"),
                prompt_pad_to_multiple=singleshot_mtp.get("prompt_pad_to_multiple"),
                ignore_index=ignore_index,
                loss_start_key=str(singleshot_mtp.get("loss_start_key", "mtp_loss_start")),
                rollout_replay=True,
                flex_block_size=_int_from_config(
                    singleshot_mtp,
                    "flex_block_size",
                    _int_from_config(singleshot_mtp, "flex_attention_block_size", 128),
                ),
            )
        except Exception as exc:
            return {"mtp_selection_error": str(exc)}

        valid = prepared["labels"][0] != ignore_index
        source_indices = [int(x) for x in prepared["_singleshot_mtp_source_indices"][0, valid].tolist()]
        target_ids = [int(x) for x in prepared["labels"][0, valid].tolist()]
        input_slot_ids = [int(x) for x in prepared["input_ids"][0, valid].tolist()]
        mask_slots = prepared["_singleshot_mtp_mask_positions"][0]
        refill_slots = prepared["_singleshot_mtp_refill_positions"][0]
        return {
            "mtp_rollout_replay": True,
            "mtp_raw_seq_len": int(prepared["_singleshot_mtp_raw_seq_len"].item()),
            "mtp_padded_seq_len": int(prepared["_singleshot_mtp_padded_seq_len"].item()),
            "mtp_prefix_length": loss_start + 1,
            "mtp_prompt_length": loss_start + 1,
            "mtp_prompt_pad_count": int(prepared["_singleshot_mtp_prompt_pad_count"].item()),
            "mtp_prompt_padded_length": int(prepared["_singleshot_mtp_prompt_padded_len"].item()),
            "mtp_k_toks": int(singleshot_mtp["k_toks"]),
            "mtp_window_source_start": loss_start,
            "mtp_window_source_end": source_len,
            "mtp_context_prefix_token_count": loss_start + 1,
            "mtp_trainer_input_token_count": int(prepared["input_ids"].shape[1]),
            "mtp_generated_token_count": int(prepared["_singleshot_mtp_generated_token_count"].item()),
            "mtp_chunk_count": int(prepared["_singleshot_mtp_chunk_count"].item()),
            "mtp_loss_start_source_index": loss_start,
            "mtp_loss_start_target_token_index": loss_start + 1,
            "mtp_first_trained_target_is_first_generated": loss_start + 1 == data["prompt_len"],
            "mtp_selected_label_count": len(target_ids),
            "mtp_selected_source_indices_head": _head_token_ids(source_indices, 32),
            "mtp_selected_source_indices_tail": _tail_token_ids(source_indices, 32),
            "mtp_selected_input_slot_token_ids_head": _head_token_ids(input_slot_ids, 32),
            "mtp_selected_target_token_ids": target_ids,
            "mtp_selected_target_token_ids_head": _head_token_ids(target_ids, 128),
            "mtp_selected_target_token_ids_tail": _tail_token_ids(target_ids, 128),
            "mtp_replay_input_token_ids_head": _head_token_ids(
                [int(x) for x in prepared["input_ids"][0].tolist()], 128
            ),
            "mtp_replay_input_token_ids_tail": _tail_token_ids(
                [int(x) for x in prepared["input_ids"][0].tolist()], 128
            ),
            "mtp_replay_position_ids_head": _head_token_ids(
                [int(x) for x in prepared["position_ids"][0].tolist()], 128
            ),
            "mtp_mask_slot_indices_head": _head_token_ids(
                [int(x) for x in mask_slots.nonzero().reshape(-1).tolist()], 64
            ),
            "mtp_refill_slot_indices_head": _head_token_ids(
                [int(x) for x in refill_slots.nonzero().reshape(-1).tolist()], 64
            ),
        }

    try:
        prepared = prepare_singleshot_mtp_batch(
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor([target_tokens], dtype=torch.long),
            k_toks=int(singleshot_mtp["k_toks"]),
            mask_token_id=singleshot_mtp.get("mask_token_id"),
            min_mask_token_id=singleshot_mtp.get("min_mask_token_id"),
            max_mask_token_id=singleshot_mtp.get("max_mask_token_id"),
            truncation_length=truncation_length,
            mask_region_count=_int_from_config(
                singleshot_mtp,
                "mask_region_count",
                _int_from_config(singleshot_mtp, "mask_region_ct", 1),
            ),
            offset=_int_from_config(singleshot_mtp, "offset", 0),
            pad_token_id=singleshot_mtp.get("pad_token_id"),
            prelude_token_ids=singleshot_mtp.get("prelude_token_ids"),
            pad_to_multiple=singleshot_mtp.get("pad_to_multiple"),
            window_start=window_start,
            min_label_source_index=loss_start,
            ignore_index=ignore_index,
        )
    except Exception as exc:
        return {"mtp_selection_error": str(exc)}

    valid = prepared.labels[0] != ignore_index
    source_indices = [int(x) for x in prepared.source_indices[valid].tolist()]
    target_ids = [int(x) for x in prepared.labels[0, valid].tolist()]
    input_slot_ids = [int(x) for x in prepared.input_ids[0, valid].tolist()]
    preserve_context_prefix = bool(singleshot_mtp.get("preserve_context_prefix", False))
    return {
        "mtp_raw_seq_len": prepared.raw_seq_len,
        "mtp_padded_seq_len": prepared.padded_seq_len,
        "mtp_prefix_length": prepared.prefix_length,
        "mtp_window_source_start": window_start,
        "mtp_window_source_end": source_len,
        "mtp_context_prefix_token_count": window_start if preserve_context_prefix else 0,
        "mtp_trainer_input_token_count": (window_start if preserve_context_prefix else 0) + prepared.padded_seq_len,
        "mtp_loss_start_source_index": loss_start,
        "mtp_loss_start_target_token_index": loss_start + 1,
        "mtp_first_trained_target_is_first_generated": loss_start + 1 == data["prompt_len"],
        "mtp_selected_label_count": len(target_ids),
        "mtp_selected_source_indices_head": _head_token_ids(source_indices, 32),
        "mtp_selected_source_indices_tail": _tail_token_ids(source_indices, 32),
        "mtp_selected_input_slot_token_ids_head": _head_token_ids(input_slot_ids, 32),
        "mtp_selected_target_token_ids": target_ids,
        "mtp_selected_target_token_ids_head": _head_token_ids(target_ids, 128),
        "mtp_selected_target_token_ids_tail": _tail_token_ids(target_ids, 128),
    }


def _rollout_sample_records(
    *,
    step: int,
    prepared: _PreparedOpdChunk,
    step_metadata: Dict[str, Any],
    sample_offset: int,
    sample_limit: int,
    tokenizer: Any | None,
    singleshot_mtp: Dict[str, Any] | None,
    tail_tokens: int,
    target_tokens: int,
    text_max_chars: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for local_idx, (prompt, sequence, data) in enumerate(zip(prepared.prompts, prepared.sequences, prepared.data)):
        if len(records) >= sample_limit:
            break
        sample_idx = sample_offset + local_idx
        generated = list(sequence[len(prompt) :])
        teacher_input = data["model_input"]["input_ids"]
        teacher_targets = data["loss_fn_inputs"]["target_tokens"]
        data_for_selection = {
            **data,
            "prompt_len": len(prompt),
        }
        selection = _mtp_training_selection(data_for_selection, singleshot_mtp)
        selected_full = selection.get("mtp_selected_target_token_ids")
        selected_head = selection.get("mtp_selected_target_token_ids_head", [])
        selected_tail = selection.get("mtp_selected_target_token_ids_tail", [])
        sample_info = prepared.sample_metadata[local_idx] if local_idx < len(prepared.sample_metadata) else {}
        native_mtp_trace = sample_info.get("mtp_debug_trace") if isinstance(sample_info, dict) else None
        native_mtp_steps = (
            [step for step in native_mtp_trace if isinstance(step, dict)] if isinstance(native_mtp_trace, list) else []
        )
        native_mtp_phases = sorted({str(step["phase"]) for step in native_mtp_steps if step.get("phase") is not None})
        native_mtp_q_lens = sorted(
            {
                int(step["decode_q_len_per_req"])
                for step in native_mtp_steps
                if isinstance(step.get("decode_q_len_per_req"), int)
            }
        )
        generated_text_full = _decode_token_ids(tokenizer, generated, max_chars=text_max_chars)
        rollout_health = _rollout_health_for_tokens(
            generated,
            tokenizer=tokenizer,
            generated_text=generated_text_full,
        )
        native_mtp_summary = _native_mtp_trace_replay_summary(
            generated,
            native_mtp_steps,
            k_toks=int(singleshot_mtp.get("k_toks", 1)) if isinstance(singleshot_mtp, dict) else 1,
        )
        native_mtp_trace_steps_compact = []
        for trace_step in native_mtp_summary["trace_steps"]:
            committed_token_ids = [int(token_id) for token_id in trace_step.get("committed_token_ids", [])]
            native_mtp_trace_steps_compact.append(
                {
                    **trace_step,
                    "committed_text": _decode_token_ids(
                        tokenizer,
                        committed_token_ids,
                        max_chars=text_max_chars,
                    ),
                    "pending_text": _decode_token_ids(
                        tokenizer,
                        [int(token_id) for token_id in trace_step.get("pending_token_ids", [])],
                        max_chars=text_max_chars,
                    ),
                }
            )
        supervised_target_token_ids = (
            [int(token_id) for token_id in selected_full] if isinstance(selected_full, list) else None
        )
        record: Dict[str, Any] = {
            "step": step,
            "chunk_idx": prepared.chunk_idx,
            "sample_idx": sample_idx,
            **_sample_metadata_at_step(step_metadata, sample_idx),
            "prompt_token_count": len(prompt),
            "generated_token_count": len(generated),
            "sequence_token_count": len(sequence),
            "teacher_prefill_input_token_count": len(teacher_input),
            "teacher_target_token_count": len(teacher_targets),
            "prompt_tail_token_ids": _tail_token_ids(prompt, tail_tokens),
            "generated_token_ids": _head_token_ids(generated, target_tokens),
            "generated_token_ids_full": generated,
            "sequence_tail_token_ids": _tail_token_ids(sequence, tail_tokens),
            "teacher_prefill_input_token_ids_head": _head_token_ids(teacher_input, tail_tokens),
            "teacher_prefill_tail_token_ids": _tail_token_ids(teacher_input, tail_tokens),
            "teacher_target_token_ids_head": _head_token_ids(teacher_targets, target_tokens),
            "teacher_target_token_ids_tail": _tail_token_ids(teacher_targets, target_tokens),
            "prompt_tail_text": _decode_token_ids(
                tokenizer, _tail_token_ids(prompt, tail_tokens), max_chars=text_max_chars
            ),
            "generated_text": _decode_token_ids(
                tokenizer, _head_token_ids(generated, target_tokens), max_chars=text_max_chars
            ),
            "generated_text_full": generated_text_full,
            "sequence_tail_text": _decode_token_ids(
                tokenizer,
                _tail_token_ids(sequence, tail_tokens),
                max_chars=text_max_chars,
            ),
            "teacher_prefill_tail_text": _decode_token_ids(
                tokenizer,
                _tail_token_ids(teacher_input, tail_tokens),
                max_chars=text_max_chars,
            ),
            "teacher_target_text_head": _decode_token_ids(
                tokenizer,
                _head_token_ids(teacher_targets, target_tokens),
                max_chars=text_max_chars,
            ),
            "teacher_target_text_tail": _decode_token_ids(
                tokenizer,
                _tail_token_ids(teacher_targets, target_tokens),
                max_chars=text_max_chars,
            ),
            "native_mtp_debug_trace": native_mtp_trace if isinstance(native_mtp_trace, list) else [],
            "native_mtp_debug_trace_step_count": len(native_mtp_steps),
            "native_mtp_debug_trace_phases": ",".join(native_mtp_phases),
            "native_mtp_debug_trace_q_lens": ",".join(str(q_len) for q_len in native_mtp_q_lens),
            "native_mtp_bootstrap_token_ids": native_mtp_summary["bootstrap_token_ids"],
            "native_mtp_bootstrap_text": _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in native_mtp_summary["bootstrap_token_ids"]],
                max_chars=text_max_chars,
            ),
            "native_mtp_trace_committed_token_ids": native_mtp_summary["trace_committed_token_ids"],
            "native_mtp_trace_committed_text": _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in native_mtp_summary["trace_committed_token_ids"]],
                max_chars=text_max_chars,
            ),
            "native_mtp_replay_committed_token_ids": native_mtp_summary["replay_committed_token_ids"],
            "native_mtp_replay_committed_text": _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in native_mtp_summary["replay_committed_token_ids"]],
                max_chars=text_max_chars,
            ),
            "native_mtp_trace_unused_committed_token_count": native_mtp_summary["trace_unused_committed_token_count"],
            "native_mtp_trace_covered_generated_tokens": native_mtp_summary["trace_covered_generated_tokens"],
            "native_mtp_trace_commit_lens": native_mtp_summary["trace_commit_lens"],
            "native_mtp_trace_label_counts": native_mtp_summary["trace_label_counts"],
            "native_mtp_trace_steps_compact": native_mtp_trace_steps_compact,
            "supervised_target_token_ids": supervised_target_token_ids,
            "supervised_target_text": _decode_token_ids(
                tokenizer,
                supervised_target_token_ids or [],
                max_chars=text_max_chars,
            ),
            "generated_matches_supervised_target": (
                generated == supervised_target_token_ids if supervised_target_token_ids is not None else None
            ),
            **rollout_health,
            **selection,
        }
        if native_mtp_steps:
            first_step = native_mtp_steps[0]
            first_input_row_token_ids = first_step.get("input_row_token_ids") or []
            first_pending_token_ids = first_step.get("pending_token_ids") or []
            record["native_mtp_first_step_phase"] = first_step.get("phase")
            record["native_mtp_first_step_input_row_token_ids"] = first_input_row_token_ids
            record["native_mtp_first_step_pending_token_ids"] = first_pending_token_ids
            record["native_mtp_first_step_committed_token_ids"] = first_step.get("committed_token_ids") or []
            record["native_mtp_first_step_input_row_text"] = _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in first_input_row_token_ids],
                max_chars=text_max_chars,
            )
            record["native_mtp_first_step_pending_text"] = _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in first_pending_token_ids],
                max_chars=text_max_chars,
            )
        if isinstance(selected_head, list):
            record["mtp_selected_target_text_head"] = _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in selected_head],
                max_chars=text_max_chars,
            )
        if isinstance(selected_tail, list):
            record["mtp_selected_target_text_tail"] = _decode_token_ids(
                tokenizer,
                [int(token_id) for token_id in selected_tail],
                max_chars=text_max_chars,
            )
        records.append(record)
    return records


def _append_rollout_sample_records(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _log_rollout_samples_to_wandb(wandb_run: Any | None, records: List[Dict[str, Any]], *, step: int) -> None:
    if wandb_run is None or not records:
        return
    try:
        import wandb  # noqa: PLC0415

        table = wandb.Table(
            columns=[
                "step",
                "sample_idx",
                "prompt_token_count",
                "generated_token_count",
                "mtp_selected_label_count",
                "generated_matches_supervised_target",
                "rollout_max_repeated_token_run",
                "native_mtp_trace_covered_generated_tokens",
                "native_mtp_trace_unused_committed_token_count",
                "first_trained_target_is_first_generated",
                "native_mtp_debug_trace_phases",
                "native_mtp_debug_trace_q_lens",
                "prompt_tail_text",
                "generated_text",
                "supervised_target_text",
                "mtp_selected_target_text_head",
            ]
        )
        for record in records:
            table.add_data(
                record.get("step"),
                record.get("sample_idx"),
                record.get("prompt_token_count"),
                record.get("generated_token_count"),
                record.get("mtp_selected_label_count"),
                record.get("generated_matches_supervised_target"),
                record.get("rollout_max_repeated_token_run"),
                record.get("native_mtp_trace_covered_generated_tokens"),
                record.get("native_mtp_trace_unused_committed_token_count"),
                record.get("mtp_first_trained_target_is_first_generated"),
                record.get("native_mtp_debug_trace_phases"),
                record.get("native_mtp_debug_trace_q_lens"),
                record.get("prompt_tail_text"),
                record.get("generated_text"),
                record.get("supervised_target_text"),
                record.get("mtp_selected_target_text_head"),
            )
        wandb_run.log({"rollout_samples": table, "rollout_sample_count": len(records)}, step=step)
    except Exception as exc:  # pragma: no cover - W&B availability/runtime dependent
        log.warning("Could not log rollout samples to W&B: %s", exc)


def _opd_causal_pair(sequence: List[int]) -> tuple[List[int], List[int]]:
    """Convert a sampled trajectory into causal-LM input/target tokens."""
    if len(sequence) < 2:
        raise ValueError(f"OPD trajectory must contain at least two tokens, got {len(sequence)}")
    return list(sequence[:-1]), list(sequence[1:])


def _teacher_hidden_cache_data(sequences: List[List[int]]) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for seq in sequences:
        input_ids, target_tokens = _opd_causal_pair(seq)
        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {"target_tokens": target_tokens},
            }
        )
    return data


def _teacher_hidden_cache_input_targets(sequences: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
    input_ids_batch: List[List[int]] = []
    target_tokens_batch: List[List[int]] = []
    for seq in sequences:
        input_ids, target_tokens = _opd_causal_pair(seq)
        input_ids_batch.append(input_ids)
        target_tokens_batch.append(target_tokens)
    return input_ids_batch, target_tokens_batch


def _opd_loss_data(
    sequences: List[List[int]],
    cache_indices: List[List[int]],
    *,
    prompt_lengths: List[int] | None = None,
    sample_metadata: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    if len(cache_indices) != len(sequences):
        raise RuntimeError(f"Got {len(cache_indices)} cache-index lists for {len(sequences)} sequences")
    if prompt_lengths is not None and len(prompt_lengths) != len(sequences):
        raise RuntimeError(f"Got {len(prompt_lengths)} prompt lengths for {len(sequences)} sequences")
    if sample_metadata is not None and len(sample_metadata) != len(sequences):
        raise RuntimeError(f"Got {len(sample_metadata)} sample metadata records for {len(sequences)} sequences")

    data: List[Dict[str, Any]] = []
    for sample_idx, (seq, idx) in enumerate(zip(sequences, cache_indices)):
        input_ids, target_tokens = _opd_causal_pair(seq)
        if len(idx) != len(input_ids):
            raise RuntimeError(
                f"Teacher cache index length {len(idx)} does not match OPD input length {len(input_ids)}"
            )
        prompt_len = prompt_lengths[sample_idx] if prompt_lengths is not None else None
        sample_info = sample_metadata[sample_idx] if sample_metadata is not None else {}
        native_mtp_trace = sample_info.get("mtp_debug_trace") if isinstance(sample_info, dict) else None
        loss_fn_inputs: Dict[str, Any] = {
            "target_tokens": target_tokens,
            "teacher_ids": [0] * len(target_tokens),
            "teacher_weights": [1.0] * len(target_tokens),
            "teacher_cache_indices": idx,
            **({"mtp_loss_start": [max(0, int(prompt_len) - 1)]} if prompt_len is not None else {}),
        }
        if isinstance(native_mtp_trace, list):
            loss_fn_inputs["singleshot_mtp_native_trace"] = [json.dumps(native_mtp_trace, separators=(",", ":"))]
        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": loss_fn_inputs,
            }
        )
    return data


def _teacher_hidden_for(
    teacher_url: str, sequences: List[List[int]], timeout: float = 120.0
) -> List[List[List[float]]]:
    """Return per-sample [seq_len, hidden] prefill hidden states from teacher SGLang."""
    r = requests.post(
        f"{teacher_url}/generate",
        json={
            "input_ids": sequences,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_hidden_states": True,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    outputs = _normalize_generate_batch(r.json())
    if len(outputs) != len(sequences):
        raise RuntimeError(f"Teacher SGLang returned {len(outputs)} outputs for {len(sequences)} sequences")

    out_per_sample: List[List[List[float]]] = []
    for seq, output in zip(sequences, outputs):
        chunks = output.get("meta_info", {}).get("hidden_states") or []
        if not chunks:
            raise RuntimeError(f"Teacher SGLang returned no hidden states for seq len {len(seq)}")
        prefill = chunks[0]
        if len(prefill) != len(seq):
            raise RuntimeError(f"Teacher prefill hidden state length {len(prefill)} != input length {len(seq)}")
        out_per_sample.append(prefill)
    return out_per_sample


def _teacher_cache_from_xorl(
    teacher_url: str,
    sequences: List[List[int]],
    cache_path: Path,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Ask a XORL teacher server to write a shared hidden-state cache."""
    data = _teacher_hidden_cache_data(sequences)
    r = requests.post(
        f"{teacher_url}/api/v1/forward",
        json={
            "model_id": "default",
            "forward_input": {
                "data": data,
                "loss_fn": "teacher_hidden_cache",
                "loss_fn_params": {
                    "teacher_hidden_cache_path": str(cache_path),
                    "teacher_hidden_cache_dtype": "bfloat16",
                },
            },
        },
        timeout=60,
    )
    r.raise_for_status()
    future = _wait_for_future(teacher_url, r.json()["request_id"], timeout=timeout)
    cache = future.get("info", {}).get("teacher_hidden_cache")
    if not cache:
        raise RuntimeError(f"XORL teacher did not return teacher_hidden_cache metadata: {future}")
    indices = cache.get("cache_indices_by_sample") or []
    if len(indices) != len(sequences):
        raise RuntimeError(f"XORL teacher returned {len(indices)} cache-index lists for {len(sequences)} sequences")
    for seq, idx in zip(sequences, indices):
        input_ids, _ = _opd_causal_pair(seq)
        if len(idx) != len(input_ids):
            raise RuntimeError(f"XORL teacher returned {len(idx)} cache indices for OPD input length {len(input_ids)}")
    if cache.get("path") != str(cache_path):
        raise RuntimeError(f"XORL teacher wrote unexpected cache path: {cache.get('path')} != {cache_path}")
    return {
        "cache_path": cache_path,
        "cache_indices_by_sample": indices,
        "metrics": future.get("metrics", {}),
        "info": future.get("info", {}),
    }


def _teacher_cache_from_sglang(
    teacher_url: str,
    sequences: List[List[int]],
    cache_path: Path,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Ask a SGLang teacher to prefill and write the shared hidden-state cache."""
    input_ids_batch, target_tokens_batch = _teacher_hidden_cache_input_targets(sequences)
    request_payload = {
        "input_ids": input_ids_batch,
        "target_tokens": target_tokens_batch,
        "cache_path": str(cache_path),
        "cache_key": "hidden_states",
        "dtype": "bfloat16",
    }
    r = requests.post(
        f"{teacher_url}/teacher_hidden_cache",
        json=request_payload,
        timeout=timeout,
    )

    if getattr(r, "status_code", None) == 404:
        log.warning("SGLang teacher lacks /teacher_hidden_cache; falling back to legacy hidden-state JSON path")
        hiddens = _teacher_hidden_for(teacher_url, input_ids_batch, timeout=timeout)
        cache_indices = _save_teacher_cache(hiddens, cache_path)
        return {
            "cache_path": cache_path,
            "cache_indices_by_sample": cache_indices,
            "metrics": {},
            "info": {
                "teacher_hidden_cache": {
                    "path": str(cache_path),
                    "cache_indices_by_sample": cache_indices,
                    "backend": "sglang_generate_legacy",
                }
            },
        }

    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        body = getattr(r, "text", "")
        raise RuntimeError(f"SGLang teacher_hidden_cache request failed: {body[:1000]}") from exc

    cache = r.json()
    if not isinstance(cache, dict):
        raise RuntimeError(f"SGLang teacher_hidden_cache returned non-object payload: {type(cache).__name__}")
    if cache.get("error") or cache.get("success") is False:
        message = cache.get("message") or cache.get("error") or cache
        raise RuntimeError(f"SGLang teacher_hidden_cache failed: {message}")

    indices = cache.get("cache_indices_by_sample") or []
    if len(indices) != len(sequences):
        raise RuntimeError(f"SGLang teacher returned {len(indices)} cache-index lists for {len(sequences)} sequences")
    normalized_indices: List[List[int]] = []
    for input_ids, idx in zip(input_ids_batch, indices):
        if len(idx) != len(input_ids):
            raise RuntimeError(
                f"SGLang teacher returned {len(idx)} cache indices for OPD input length {len(input_ids)}"
            )
        normalized_indices.append([int(i) for i in idx])

    reported_path = cache.get("path")
    if reported_path is not None and reported_path != str(cache_path):
        raise RuntimeError(f"SGLang teacher wrote unexpected cache path: {reported_path} != {cache_path}")
    if not cache_path.exists():
        raise RuntimeError(f"SGLang teacher reported cache path but file does not exist: {cache_path}")

    return {
        "cache_path": cache_path,
        "cache_indices_by_sample": normalized_indices,
        "metrics": {
            "teacher_hidden_cache_tokens": int(cache.get("num_tokens") or 0),
            "teacher_prefill_forward_compute_s": float(cache.get("teacher_prefill_forward_compute_s") or 0.0),
            "teacher_hidden_cache_write_s": float(cache.get("teacher_hidden_cache_write_s") or 0.0),
        },
        "info": {"teacher_hidden_cache": cache},
    }


def _save_teacher_cache(hiddens: List[List[List[float]]], cache_path: Path) -> List[List[int]]:
    """Concatenate per-sample hidden states and return per-sample cache indices."""
    cache_indices: List[List[int]] = []
    chunks = []
    offset = 0
    for h in hiddens:
        chunks.append(torch.tensor(h, dtype=torch.bfloat16))
        cache_indices.append(list(range(offset, offset + len(h))))
        offset += len(h)
    big = torch.cat(chunks, dim=0).contiguous()
    save_file({"hidden_states": big}, str(cache_path))
    return cache_indices


def _prepare_opd_chunk(
    *,
    args: argparse.Namespace,
    student_url: str,
    teacher_url: str,
    prompts: List[List[int]],
    artifacts_dir: Path,
    step: int,
    chunk_idx: int,
    cache_path: Path | None = None,
    teacher_semaphore: Semaphore | None = None,
) -> _PreparedOpdChunk:
    """Sample a prompt chunk and prefill teacher hidden states for that chunk."""
    chunk_t0 = time.perf_counter()
    sample_t0 = time.perf_counter()
    sequences, sampling_metrics = _student_sample_for_opd_batch(
        student_url,
        prompts,
        args.max_new_tokens,
        singleshot_mtp=args.singleshot_mtp,
        temperature=args.student_temperature,
        timeout=args.request_timeout,
        sampling_params_extra=args.student_sampling_params,
    )
    sample_s = _elapsed(sample_t0)
    sample_metadata = sampling_metrics.pop("_student_sample_metadata", [{} for _ in sequences])
    sample_input_tokens = sum(len(prompt) for prompt in prompts)
    sample_output_tokens = sum(max(0, len(seq) - len(prompt)) for seq, prompt in zip(sequences, prompts))

    degenerate_token_ids = {0}
    if isinstance(args.singleshot_mtp, dict) and args.singleshot_mtp.get("mask_token_id") is not None:
        degenerate_token_ids.add(int(args.singleshot_mtp["mask_token_id"]))
    sampling_metrics.update(
        _check_rollout_coherence(
            prompts,
            sequences,
            degenerate_token_ids=degenerate_token_ids,
            max_degenerate_frac=args.rollout_coherence_max_degenerate_frac,
            action=args.rollout_coherence_action,
            context=f"step={step} chunk={chunk_idx}",
        )
    )

    if cache_path is None:
        cache_path = artifacts_dir / f"teacher_hidden_step{step}_chunk{chunk_idx}.safetensors"
    teacher_wait_t0 = time.perf_counter()
    acquired_teacher = False
    if teacher_semaphore is not None:
        teacher_semaphore.acquire()
        acquired_teacher = True
    teacher_queue_wait_s = _elapsed(teacher_wait_t0)
    teacher_t0 = time.perf_counter()
    try:
        if args.teacher_backend == "xorl":
            teacher_cache = _teacher_cache_from_xorl(
                teacher_url,
                sequences,
                cache_path,
                timeout=args.request_timeout,
            )
            cache_indices = teacher_cache["cache_indices_by_sample"]
            teacher_cache_metrics = teacher_cache["metrics"]
            if not cache_path.exists():
                raise RuntimeError(f"XORL teacher reported cache path but file does not exist: {cache_path}")
            teacher_cache_save_s = 0.0
        else:
            teacher_cache = _teacher_cache_from_sglang(
                teacher_url,
                sequences,
                cache_path,
                timeout=args.request_timeout,
            )
            cache_indices = teacher_cache["cache_indices_by_sample"]
            teacher_cache_metrics = teacher_cache["metrics"]
            teacher_cache_save_s = 0.0
    finally:
        if acquired_teacher:
            teacher_semaphore.release()

    teacher_prefill_s = _elapsed(teacher_t0)
    teacher_tokens = sum(len(seq) - 1 for seq in sequences)
    data = _opd_loss_data(
        sequences,
        cache_indices,
        prompt_lengths=[len(prompt) for prompt in prompts],
        sample_metadata=sample_metadata,
    )
    metrics: Dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "chunk_size": len(prompts),
        "chunk_prepare_s": _elapsed(chunk_t0),
        "student_endpoint_url": student_url,
        "teacher_endpoint_url": teacher_url,
        "student_sampling_s": sample_s,
        "student_sampling_batch_size": len(prompts),
        "student_sampling_prompt_tokens": sample_input_tokens,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s if sample_s > 0 else 0.0,
        "teacher_prefill_queue_wait_s": teacher_queue_wait_s,
        "teacher_prefill_s": teacher_prefill_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_prefill_s if teacher_prefill_s > 0 else 0.0,
        "teacher_prefill_forward_compute_s": float(
            _metric(teacher_cache_metrics, "teacher_prefill_forward_compute_s", 0.0)
        ),
        "teacher_hidden_cache_write_s": float(_metric(teacher_cache_metrics, "teacher_hidden_cache_write_s", 0.0)),
        "teacher_cache_save_s": teacher_cache_save_s,
    }
    metrics.update(sampling_metrics)
    return _PreparedOpdChunk(
        chunk_idx=chunk_idx,
        prompts=prompts,
        sequences=sequences,
        cache_path=cache_path,
        cache_indices=cache_indices,
        data=data,
        metrics=metrics,
        sample_metadata=sample_metadata,
    )


def _save_fb_request_payload(
    save_dir: Path, *, data: List[Dict[str, Any]], teacher_head: str, cache_path: Path
) -> None:
    """Persist one forward_backward request (data + teacher hidden cache) for offline replay."""
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(save_dir.glob("fb_request_*.json")))
    cache_file = f"teacher_hidden_{idx}.safetensors"
    shutil.copyfile(cache_path, save_dir / cache_file)
    payload = {"teacher_head": teacher_head, "cache_file": cache_file, "data": data}
    (save_dir / f"fb_request_{idx}.json").write_text(json.dumps(payload))
    log.info("Saved FB request payload %d (%d batches) to %s", idx, len(data), save_dir)


def _run_fb_request_replay(args: argparse.Namespace) -> int:
    """Trainer-only loop over captured forward_backward payloads.

    No samplers or teachers are touched: the rollout data, native MTP traces, and the
    teacher hidden cache all come from the captured payloads, so trainer/CP debugging and
    deterministic FB benchmarking can run against the trainer alone.
    """
    replay_dir = Path(args.fb_request_replay_path)
    payload_paths = sorted(replay_dir.glob("fb_request_*.json"))
    if not payload_paths:
        log.error("No fb_request_*.json payloads under %s", replay_dir)
        return 2
    loaded = []
    for path in payload_paths:
        payload = json.loads(path.read_text())
        cache_path = replay_dir / payload["cache_file"]
        if not cache_path.exists():
            log.error("Missing teacher hidden cache %s for %s", cache_path, path.name)
            return 2
        loaded.append((payload["data"], payload["teacher_head"], cache_path))
    results_path = replay_dir / "replay_results.jsonl"
    log.info(
        "FB replay: %d payload(s) x %d step(s) against %s; results -> %s",
        len(loaded),
        args.num_steps,
        args.train_url,
        results_path,
    )
    for step in range(args.num_steps):
        for payload_idx, (data, teacher_head, cache_path) in enumerate(loaded):
            fb, fb_timing = _run_forward_backward_chunk(
                args=args,
                train_url=args.train_url,
                data=data,
                teacher_head=teacher_head,
                cache_path=cache_path,
                clear_gradients_after_backward=True,
            )
            fb_metrics = fb["metrics"]
            row = {
                "step": step,
                "payload": payload_idx,
                "loss": _scalar(fb["loss_fn_outputs"][0]["loss"]),
                "valid_tokens": int(_metric(fb_metrics, "valid_tokens", 0)),
                "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                **fb_timing,
                **_singleshot_trainer_metrics_from_response(fb_metrics),
            }
            with results_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            log.info(
                "FB replay step=%d payload=%d loss=%.4f valid_tokens=%d trainer=%.3fs forward=%.3fs backward=%.3fs",
                step,
                payload_idx,
                row["loss"],
                row["valid_tokens"],
                row["trainer_forward_backward_s"],
                row["trainer_forward_loss_s"],
                row["trainer_backward_s"],
            )
    log.info("FB replay complete: %d forward_backward calls", args.num_steps * len(loaded))
    return 0


def _run_forward_backward_chunk(
    *,
    args: argparse.Namespace,
    train_url: str,
    data: List[Dict[str, Any]],
    teacher_head: str,
    cache_path: Path,
    clear_gradients_after_backward: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    save_dir = str(getattr(args, "save_fb_request_dir", "") or "")
    if save_dir:
        _save_fb_request_payload(Path(save_dir), data=data, teacher_head=teacher_head, cache_path=Path(cache_path))
    fb_t0 = time.perf_counter()
    loss_fn_params = {
        "teacher_heads": {"0": teacher_head},
        "teacher_hidden_caches": {"0": str(cache_path)},
        "opd_sort_by_teacher": True,
        "opd_loss_mode": args.opd_loss_mode,
        "opd_kl_backend": args.opd_kl_backend,
        "opd_vocab_chunk_size": args.opd_vocab_chunk_size,
        "opd_sharded_head_device_cache": args.opd_sharded_head_device_cache,
        "opd_profile_timings": True,
        "opd_profile_sync_cuda": args.profile_sync_cuda,
        "opd_emit_full_vocab_diagnostics": args.opd_emit_full_vocab_diagnostics,
        "profile_clear_gradients_after_backward": clear_gradients_after_backward,
        "num_chunks": 8,
    }
    if args.singleshot_mtp:
        loss_fn_params["singleshot_mtp"] = args.singleshot_mtp
    r = requests.post(
        f"{train_url}/api/v1/forward_backward",
        json={
            "model_id": "default",
            "forward_backward_input": {
                "data": data,
                "loss_fn": "opd_loss",
                "loss_fn_params": loss_fn_params,
            },
        },
        timeout=120,
    )
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"forward_backward HTTP {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    metrics: Dict[str, Any] = {"forward_backward_enqueue_s": _elapsed(fb_t0)}
    fb_wait_t0 = time.perf_counter()
    fb = _wait_for_future(train_url, r.json()["request_id"], timeout=args.forward_backward_timeout)
    metrics["forward_backward_wait_s"] = _elapsed(fb_wait_t0)
    metrics["forward_backward_roundtrip_s"] = _elapsed(fb_t0)
    return fb, metrics


def _run_forward_chunk(
    *,
    args: argparse.Namespace,
    train_url: str,
    data: List[Dict[str, Any]],
    teacher_head: str,
    cache_path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    forward_t0 = time.perf_counter()
    loss_fn_params = {
        "teacher_heads": {"0": teacher_head},
        "teacher_hidden_caches": {"0": str(cache_path)},
        "opd_sort_by_teacher": True,
        "opd_loss_mode": args.opd_loss_mode,
        "opd_kl_backend": args.opd_kl_backend,
        "opd_vocab_chunk_size": args.opd_vocab_chunk_size,
        "opd_sharded_head_device_cache": args.opd_sharded_head_device_cache,
        "opd_profile_timings": True,
        "opd_profile_sync_cuda": args.profile_sync_cuda,
        "opd_emit_full_vocab_diagnostics": args.opd_emit_full_vocab_diagnostics,
        "num_chunks": 8,
    }
    if args.singleshot_mtp:
        loss_fn_params["singleshot_mtp"] = args.singleshot_mtp
    r = requests.post(
        f"{train_url}/api/v1/forward",
        json={
            "model_id": "default",
            "forward_input": {
                "data": data,
                "loss_fn": "opd_loss",
                "loss_fn_params": loss_fn_params,
            },
        },
        timeout=120,
    )
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"forward HTTP {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    metrics: Dict[str, Any] = {"forward_enqueue_s": _elapsed(forward_t0)}
    forward_wait_t0 = time.perf_counter()
    forward = _wait_for_future(train_url, r.json()["request_id"], timeout=args.forward_backward_timeout)
    metrics["forward_wait_s"] = _elapsed(forward_wait_t0)
    metrics["forward_roundtrip_s"] = _elapsed(forward_t0)
    return forward, metrics


def _evaluate_opd_validation_loss(
    *,
    args: argparse.Namespace,
    train_url: str,
    teacher_head: str,
    eval_prompt_schedule: _PromptSchedule,
    student_router: _EndpointUrlRouter,
    teacher_router: _EndpointUrlRouter,
    artifacts_dir: Path,
    completed_step: int,
) -> Dict[str, Any]:
    eval_steps = min(args.checkpoint_eval_steps, eval_prompt_schedule.num_steps)
    if eval_steps <= 0:
        return {}

    eval_t0 = time.perf_counter()
    weighted_loss_sum = 0.0
    total_valid_tokens = 0
    chunk_count = 0
    sample_s = 0.0
    teacher_s = 0.0
    forward_s = 0.0
    chunk_size = args.pipeline_chunk_size if args.pipeline_chunk_size > 0 else eval_prompt_schedule.batch_size

    for eval_idx in range(eval_steps):
        prompts, _ = eval_prompt_schedule.batch_for_step(eval_idx)
        for chunk_idx, prompt_chunk in enumerate(_chunked(prompts, chunk_size)):
            prepared = _prepare_opd_chunk(
                args=args,
                student_url=student_router.next(),
                teacher_url=teacher_router.next(),
                prompts=prompt_chunk,
                artifacts_dir=artifacts_dir,
                step=completed_step,
                chunk_idx=chunk_idx,
                cache_path=artifacts_dir
                / f"teacher_hidden_val_step{completed_step}_eval{eval_idx}_chunk{chunk_idx}.safetensors",
            )
            try:
                forward, forward_timing = _run_forward_chunk(
                    args=args,
                    train_url=train_url,
                    data=prepared.data,
                    teacher_head=teacher_head,
                    cache_path=prepared.cache_path,
                )
            finally:
                _delete_consumed_teacher_cache(prepared.cache_path)

            forward_metrics = forward["metrics"]
            chunk_loss = _scalar(forward["loss_fn_outputs"][0]["loss"])
            chunk_valid = int(_metric(forward_metrics, "valid_tokens", 0))
            weighted_loss_sum += chunk_loss * chunk_valid
            total_valid_tokens += chunk_valid
            chunk_count += 1
            sample_s += float(prepared.metrics.get("student_sampling_s", 0.0) or 0.0)
            teacher_s += float(prepared.metrics.get("teacher_prefill_s", 0.0) or 0.0)
            forward_s += float(forward_timing.get("forward_roundtrip_s", 0.0) or 0.0)

    val_loss = weighted_loss_sum / total_valid_tokens if total_valid_tokens > 0 else None
    return {
        "val_loss": val_loss,
        "val_valid_tokens": total_valid_tokens,
        "val_steps": eval_steps,
        "val_chunks": chunk_count,
        "val_total_s": _elapsed(eval_t0),
        "val_student_sampling_s": sample_s,
        "val_teacher_prefill_s": teacher_s,
        "val_forward_roundtrip_s": forward_s,
        "val_dataset_split": args.checkpoint_eval_dataset_split,
    }


def _queue_put_until_stopped(
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    item: tuple[str, int, Any],
) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=0.5)
            return True
        except queue_module.Full:
            continue
    return False


def _opd_prepare_worker(
    *,
    worker_idx: int,
    job_queue: queue_module.Queue,
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    args: argparse.Namespace,
    student_router: _EndpointUrlRouter,
    teacher_router: _EndpointUrlRouter,
    artifacts_dir: Path,
    step: int,
    teacher_semaphore: Semaphore,
) -> None:
    """Background chunk-preparation worker, mirroring the xorl-client RL pattern."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not stop_event.is_set():
            try:
                job = job_queue.get(timeout=0.5)
            except queue_module.Empty:
                continue
            if job is None:
                break

            chunk_idx, prompts = job
            try:
                result = loop.run_until_complete(
                    asyncio.to_thread(
                        _prepare_opd_chunk,
                        args=args,
                        student_url=student_router.next(),
                        teacher_url=teacher_router.next(),
                        prompts=prompts,
                        artifacts_dir=artifacts_dir,
                        step=step,
                        chunk_idx=chunk_idx,
                        teacher_semaphore=teacher_semaphore,
                    )
                )
                if not _queue_put_until_stopped(output_queue, stop_event, ("ok", chunk_idx, result)):
                    break
            except Exception as exc:
                log.exception("OPD prepare worker %d failed on chunk %s", worker_idx, chunk_idx)
                _queue_put_until_stopped(output_queue, stop_event, ("error", chunk_idx, exc))
                stop_event.set()
                break
    finally:
        _queue_put_until_stopped(output_queue, stop_event, ("done", worker_idx, None))
        loop.close()


def _opd_step_prompt_chunks(args: argparse.Namespace, prompts: List[List[int]]) -> List[List[List[int]]]:
    """Return the ordered prompt-chunk list for a step.

    Multiple chunks when ``pipeline_chunk_size>0`` and ``len(prompts)>chunk_size``; otherwise a single
    chunk containing every prompt. This mirrors the chunking decision the pipelined path makes so the
    async-overlap path can reuse the same worker-pool prepare for one or many chunks.
    """
    if args.pipeline_chunk_size > 0 and len(prompts) > args.pipeline_chunk_size:
        return _chunked(prompts, args.pipeline_chunk_size)
    return [list(prompts)]


class _PreparedOpdStep:
    """All prepared chunks for a single OPD step, plus the worker-pool sizing used to build them."""

    def __init__(
        self,
        *,
        step: int,
        prompt_chunks: List[List[List[int]]],
        prepared_chunks: List[_PreparedOpdChunk],
        worker_count: int,
        teacher_concurrency: int,
        max_prefetch: int,
        prepare_wait_s: float,
    ) -> None:
        self.step = step
        self.prompt_chunks = prompt_chunks
        self.prepared_chunks = prepared_chunks
        self.worker_count = worker_count
        self.teacher_concurrency = teacher_concurrency
        self.max_prefetch = max_prefetch
        self.prepare_wait_s = prepare_wait_s


def _prepare_opd_step_chunks(
    *,
    args: argparse.Namespace,
    step: int,
    prompts: List[List[int]],
    student_router: "_EndpointUrlRouter",
    teacher_router: "_EndpointUrlRouter",
    artifacts_dir: Path,
) -> _PreparedOpdStep:
    """Prepare (sample + teacher-prefill) ALL chunks of a step via the worker pool, returning them ordered.

    This reuses the exact ``_opd_prepare_worker``/``_prepare_opd_chunk`` machinery the pipelined path uses
    (no duplicated sampling), but drains every chunk into an ordered list instead of interleaving
    forward_backward. It never calls forward_backward and never deletes teacher caches; cache deletion is
    owned exclusively by the foreground fb-consume loop.
    """
    prompt_chunks = _opd_step_prompt_chunks(args, prompts)
    max_prefetch = max(1, min(args.pipeline_prefetch_chunks, len(prompt_chunks)))
    teacher_concurrency = max(1, args.pipeline_teacher_concurrency)
    worker_count = max(1, min(max_prefetch, len(prompt_chunks)))
    teacher_semaphore = Semaphore(teacher_concurrency)
    prepare_wait_s = 0.0

    job_queue: queue_module.Queue = queue_module.Queue()
    output_queue: queue_module.Queue = queue_module.Queue(maxsize=max_prefetch)
    stop_event = threading.Event()
    for chunk_idx, prompt_chunk in enumerate(prompt_chunks):
        job_queue.put((chunk_idx, prompt_chunk))
    for _ in range(worker_count):
        job_queue.put(None)

    workers = [
        threading.Thread(
            target=_opd_prepare_worker,
            kwargs={
                "worker_idx": worker_idx,
                "job_queue": job_queue,
                "output_queue": output_queue,
                "stop_event": stop_event,
                "args": args,
                "student_router": student_router,
                "teacher_router": teacher_router,
                "artifacts_dir": artifacts_dir,
                "step": step,
                "teacher_semaphore": teacher_semaphore,
            },
            name=f"opd-prepare-{step}-{worker_idx}",
            daemon=True,
        )
        for worker_idx in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    pending_chunks: Dict[int, _PreparedOpdChunk] = {}
    prepared_chunks: List[_PreparedOpdChunk] = []
    finished_workers = 0
    try:
        for chunk_idx in range(len(prompt_chunks)):
            wait_t0 = time.perf_counter()
            while chunk_idx not in pending_chunks:
                try:
                    kind, item_idx, payload = output_queue.get(timeout=1.0)
                except queue_module.Empty:
                    if finished_workers >= worker_count and not pending_chunks:
                        raise RuntimeError(f"OPD prepare workers exited before chunk {chunk_idx} became available")
                    continue

                if kind == "ok":
                    pending_chunks[item_idx] = payload
                elif kind == "error":
                    raise RuntimeError(f"OPD prepare worker failed on chunk {item_idx}: {payload}") from payload
                elif kind == "done":
                    finished_workers += 1
                else:
                    raise RuntimeError(f"Unexpected OPD prepare queue item kind: {kind!r}")

            prepared = pending_chunks.pop(chunk_idx)
            chunk_wait_s = _elapsed(wait_t0)
            prepare_wait_s += chunk_wait_s
            prepared.metrics["main_prepare_wait_s"] = chunk_wait_s
            prepared_chunks.append(prepared)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=30.0)
            if worker.is_alive():
                log.warning("OPD prepare worker %s did not stop within timeout", worker.name)

    return _PreparedOpdStep(
        step=step,
        prompt_chunks=prompt_chunks,
        prepared_chunks=prepared_chunks,
        worker_count=worker_count,
        teacher_concurrency=teacher_concurrency,
        max_prefetch=max_prefetch,
        prepare_wait_s=prepare_wait_s,
    )


class _OpdPrefetch:
    """Run ``_prepare_opd_step_chunks`` in a daemon background thread.

    Any ``BaseException`` raised during preparation is captured and re-raised in the main thread at
    ``join()`` (the barrier), so a failed background prefetch surfaces deterministically before optim/sync.
    """

    def __init__(self, target, step: int, prompts: List[List[int]], **kwargs: Any) -> None:
        self.step = step
        self._result: _PreparedOpdStep | None = None
        self._error: BaseException | None = None

        def _run() -> None:
            try:
                self._result = target(step=step, prompts=prompts, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised at join() in the main thread
                self._error = exc

        self._thread = threading.Thread(target=_run, name=f"opd-prefetch-step{step}", daemon=True)
        self._thread.start()

    def join(self) -> _PreparedOpdStep:
        self._thread.join()
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError(f"OPD prefetch for step {self.step} produced no result")
        return self._result


def _add_sync_response_profile(row: Dict[str, Any], sync_response: Dict[str, Any]) -> None:
    endpoints_synced = sync_response.get("endpoints_synced")
    endpoint_count = (
        len(endpoints_synced) if isinstance(endpoints_synced, list) else _sync_endpoint_success_count(sync_response)
    )
    row.update(
        {
            "sync_success": bool(sync_response.get("success", False)),
            "sync_transfer_time_s": float(sync_response.get("transfer_time") or 0.0),
            "sync_total_bytes": int(sync_response.get("total_bytes") or 0),
            "sync_num_parameters": int(sync_response.get("num_parameters") or 0),
            "sync_num_buckets": int(sync_response.get("num_buckets") or 0),
            "sync_endpoint_count": endpoint_count,
            "sync_endpoint_success_count": _sync_endpoint_success_count(sync_response),
        }
    )
    timing = sync_response.get("timing_breakdown") or {}
    if isinstance(timing, dict):
        for key, value in timing.items():
            if isinstance(value, (int, float)):
                row[f"sync_timing_{key}"] = float(value)

    rank_summaries = sync_response.get("p2p_rank_summaries") or []
    if isinstance(rank_summaries, list):
        sender_summaries = [s for s in rank_summaries if isinstance(s, dict) and s.get("has_transfers")]
        row["sync_p2p_rank_count"] = len(rank_summaries)
        row["sync_p2p_sender_count"] = len(sender_summaries)
        transfer_wall = [
            float(s["transfer_wall_s"]) for s in sender_summaries if isinstance(s.get("transfer_wall_s"), (int, float))
        ]
        if transfer_wall:
            row["sync_p2p_max_rank_transfer_s"] = max(transfer_wall)
            row["sync_p2p_min_rank_transfer_s"] = min(transfer_wall)

        backend_transfer = [
            float(s["backend"]["transfer_s"])
            for s in sender_summaries
            if isinstance(s.get("backend"), dict) and isinstance(s["backend"].get("transfer_s"), (int, float))
        ]
        if backend_transfer:
            row["sync_p2p_max_backend_transfer_s"] = max(backend_transfer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-url", default=os.environ.get("XORL_TRAIN_URL", "http://127.0.0.1:6000"))
    parser.add_argument("--coord-dir", default=os.environ.get("OPD_COORD_DIR", "/shared/opd-coord/default"))
    parser.add_argument(
        "--student-base-urls",
        default=os.environ.get("OPD_STUDENT_BASE_URLS"),
        help="Comma/space separated concrete student SGLang URLs. Defaults to OPD_COORD_DIR/student.json.",
    )
    parser.add_argument(
        "--student-sync-base-urls",
        default=os.environ.get("OPD_STUDENT_SYNC_BASE_URLS"),
        help=(
            "Comma/space separated concrete student endpoints to register for weight sync. "
            "Defaults to --student-base-urls."
        ),
    )
    parser.add_argument(
        "--student-router-url",
        default=os.environ.get("OPD_STUDENT_ROUTER_URL"),
        help="Student router URL used for native requests when --native-route-mode=smg.",
    )
    parser.add_argument(
        "--teacher-base-urls",
        default=os.environ.get("OPD_TEACHER_BASE_URLS"),
        help="Comma/space separated concrete teacher prefill URLs. Defaults to OPD_COORD_DIR/teacher.json.",
    )
    parser.add_argument(
        "--teacher-router-url",
        default=os.environ.get("OPD_TEACHER_ROUTER_URL"),
        help="Teacher router URL used for native requests when --native-route-mode=smg.",
    )
    parser.add_argument(
        "--native-route-mode",
        choices=("smg", "python_router", "direct_fanout"),
        default=os.environ.get("OPD_NATIVE_ROUTE_MODE", "direct_fanout"),
        help="How native /generate and /teacher_hidden_cache requests are routed.",
    )
    parser.add_argument(
        "--student-router-policy",
        default=os.environ.get("OPD_STUDENT_SMG_POLICY", os.environ.get("OPD_STUDENT_ROUTER_POLICY", "round_robin")),
        help="Policy label recorded in profile rows for student routing.",
    )
    parser.add_argument(
        "--teacher-router-policy",
        default=os.environ.get("OPD_TEACHER_SMG_POLICY", os.environ.get("OPD_TEACHER_ROUTER_POLICY", "round_robin")),
        help="Policy label recorded in profile rows for teacher routing.",
    )
    parser.add_argument(
        "--student-smg-metrics-url",
        default=os.environ.get("OPD_STUDENT_SMG_METRICS_URL"),
        help="Optional Prometheus metrics URL for the student SMG router.",
    )
    parser.add_argument(
        "--teacher-smg-metrics-url",
        default=os.environ.get("OPD_TEACHER_SMG_METRICS_URL"),
        help="Optional Prometheus metrics URL for the teacher SMG router.",
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get("XORL_TRAINER_NODE_IP") or os.environ.get("POD_IP") or os.environ.get("HOSTNAME_IP"),
    )
    parser.add_argument(
        "--teacher-backend",
        choices=("xorl", "sglang"),
        default=os.environ.get("OPD_TEACHER_BACKEND", "xorl"),
        help="Teacher prefill service type. XORL is the DCP-capable path; SGLang is legacy.",
    )
    parser.add_argument(
        "--teacher-head",
        default=os.environ.get("OPD_TEACHER_HEAD") or os.environ.get("OPD_TEACHER_MODEL_DIR"),
        help="Teacher LM-head source for trainer-side KL. Accepts HF dir, safetensors file, or OPD teacher store.",
    )
    parser.add_argument(
        "--teacher-model-dir",
        default=os.environ.get("OPD_TEACHER_MODEL_DIR"),
        help="Deprecated alias for --teacher-head.",
    )
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("OPD_NUM_STEPS", "2")))
    parser.add_argument(
        "--save-fb-request-dir",
        default=os.environ.get("OPD_SAVE_FB_REQUEST_DIR", ""),
        help="Save every forward_backward request payload (+ teacher hidden cache) to this dir for offline replay.",
    )
    parser.add_argument(
        "--fb-request-replay-path",
        default=os.environ.get("OPD_FB_REQUEST_REPLAY_PATH", ""),
        help=(
            "Trainer-only mode: replay captured fb_request_*.json payloads from this dir num-steps times "
            "(no samplers/teachers needed). Payloads come from --save-fb-request-dir."
        ),
    )
    max_opd_steps_env = os.environ.get("OPD_MAX_OPD_STEPS")
    parser.add_argument(
        "--max-opd-steps",
        type=int,
        default=int(max_opd_steps_env) if max_opd_steps_env else None,
        help="Optional hard cap on OPD training steps, including dataset-epoch schedules.",
    )
    parser.add_argument(
        "--opd-start-step",
        type=int,
        default=int(os.environ.get("OPD_START_STEP", "0")),
        help=(
            "Resume the OPD step loop at this absolute step instead of 0. The prompt schedule is "
            "deterministic per step, so this continues the dataset position. Pair with the trainer's "
            "load_checkpoint_path (via OPD_LOAD_CHECKPOINT_PATH) to resume a relaunched run."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("OPD_MAX_NEW_TOKENS", "8")))
    parser.add_argument(
        "--opd-loss-mode",
        default=os.environ.get("OPD_LOSS_MODE", "reverse_kl_full"),
        help=(
            "OPD loss mode passed to the trainer. Use hard_teacher_ce for the "
            "paper-style SingleShot hard-teacher objective."
        ),
    )
    parser.add_argument("--opd-kl-backend", default=os.environ.get("OPD_KL_BACKEND", "torch_compile"))
    parser.add_argument(
        "--opd-vocab-chunk-size",
        type=int,
        default=int(os.environ.get("OPD_VOCAB_CHUNK_SIZE", "32768")),
    )
    parser.add_argument(
        "--opd-emit-full-vocab-diagnostics",
        action="store_true",
        default=_env_bool("OPD_EMIT_FULL_VOCAB_DIAGNOSTICS"),
        help=(
            "Emit OPD teacher/student entropy and top-1 agreement metrics. Requires a full-vocab backend "
            "such as torch_compile."
        ),
    )
    parser.add_argument(
        "--opd-sharded-head-device-cache",
        action="store_true",
        default=os.environ.get("OPD_SHARDED_HEAD_DEVICE_CACHE", "").lower() in {"1", "true", "yes"},
        help="Cache sharded teacher-head chunks on GPU for one forward/backward lifetime.",
    )
    parser.add_argument(
        "--profile-output",
        default=os.environ.get("OPD_PROFILE_OUTPUT"),
        help="JSONL path for per-step throughput timings. Defaults to OPD_COORD_DIR/artifacts/opd_profile.jsonl.",
    )
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        default=os.environ.get("OPD_PROFILE_SYNC_CUDA", "").lower() in {"1", "true", "yes"},
        help="Synchronize CUDA around trainer-side timed sections for more exact phase timings.",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=int(os.environ.get("OPD_PROFILE_WARMUP_STEPS", "1")),
        help="Keep these initial steps in JSONL but exclude them from the steady-state timing summary.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_INTERVAL_STEPS", "0")),
        help="Save a training checkpoint every N completed OPD optimizer steps. 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-model-id",
        default=os.environ.get("OPD_CHECKPOINT_MODEL_ID", "default"),
        help="Model id passed to save_weights/delete_checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name-prefix",
        default=os.environ.get("OPD_CHECKPOINT_NAME_PREFIX", "opd"),
        help="Prefix for periodic and best checkpoint names.",
    )
    parser.add_argument(
        "--checkpoint-keep-latest",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_KEEP_LATEST", "1")),
        help="Number of periodic checkpoints to retain. Best checkpoints are retained separately.",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=float(os.environ.get("OPD_CHECKPOINT_TIMEOUT", "1800")),
        help="Timeout in seconds while waiting for a save_weights future.",
    )
    parser.add_argument(
        "--checkpoint-save-best",
        action="store_true",
        default=_env_bool("OPD_CHECKPOINT_SAVE_BEST"),
        help="Also retain the best checkpoint by --checkpoint-best-metric on checkpoint steps.",
    )
    parser.add_argument(
        "--checkpoint-best-metric",
        default=os.environ.get("OPD_CHECKPOINT_BEST_METRIC", "loss"),
        help="Profile-row metric used to select the retained best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-best-mode",
        choices=("min", "max"),
        default=os.environ.get("OPD_CHECKPOINT_BEST_MODE", "min"),
        help="Whether lower or higher --checkpoint-best-metric is better.",
    )
    parser.add_argument(
        "--checkpoint-best-min-step",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_BEST_MIN_STEP", "1")),
        help="Minimum completed OPD optimizer step eligible for best-checkpoint selection.",
    )
    parser.add_argument(
        "--checkpoint-eval-dataset-split",
        default=os.environ.get("OPD_CHECKPOINT_EVAL_DATASET_SPLIT", "eval"),
        help="Prompt dataset split used for held-out checkpoint scoring when metric is val_loss.",
    )
    parser.add_argument(
        "--checkpoint-eval-dataset-offset",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_EVAL_DATASET_OFFSET", "0")),
        help="Row offset into the checkpoint eval dataset split.",
    )
    parser.add_argument(
        "--checkpoint-eval-steps",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_EVAL_STEPS", "1")),
        help="Number of eval prompt batches to score on checkpoint steps.",
    )
    parser.add_argument(
        "--checkpoint-eval-batch-size",
        type=int,
        default=int(os.environ.get("OPD_CHECKPOINT_EVAL_BATCH_SIZE", "0")),
        help="Eval prompts per checkpoint eval batch. 0 reuses the training prompt batch size.",
    )
    parser.add_argument(
        "--checkpoint-summary-output",
        default=os.environ.get("OPD_CHECKPOINT_SUMMARY_OUTPUT"),
        help="JSON summary path for latest/best retained checkpoints. Defaults under OPD_COORD_DIR/artifacts.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("OPD_WANDB_PROJECT"),
        help="W&B project for OPD profile metrics. Unset disables W&B logging.",
    )
    parser.add_argument(
        "--wandb-name",
        default=os.environ.get("OPD_WANDB_NAME"),
        help="W&B run name. Defaults to the coord-dir basename when --wandb-project is set.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("OPD_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY"),
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--wandb-tags-json",
        default=os.environ.get("OPD_WANDB_TAGS_JSON", "null"),
        help='JSON-encoded W&B tag list, e.g. ["opd", "mtp"].',
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=int(os.environ.get("OPD_WANDB_LOG_INTERVAL", "1")),
        help="Log every N OPD steps to W&B when --wandb-project is set.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("OPD_WANDB_MODE") or os.environ.get("WANDB_MODE"),
        help="Optional W&B mode override such as online, offline, or disabled.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        default=_env_bool("OPD_WANDB_DISABLED"),
        help="Disable OPD W&B logging even if --wandb-project or OPD_WANDB_PROJECT is set.",
    )
    parser.add_argument(
        "--skip-optim-step",
        action="store_true",
        default=os.environ.get("OPD_SKIP_OPTIM_STEP", "").lower() in {"1", "true", "yes"},
        help="Profile sampling, teacher prefill, and trainer forward/backward without optimizer/sync.",
    )
    parser.add_argument(
        "--pipeline-chunk-size",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_CHUNK_SIZE", "0")),
        help=(
            "If >0, split prompts into chunks and overlap sample+teacher-prefill for later chunks "
            "with trainer forward_backward for earlier chunks. Gradients accumulate and one optim/sync "
            "still happens after all chunks, preserving the optimizer boundary."
        ),
    )
    parser.add_argument(
        "--pipeline-prefetch-chunks",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_PREFETCH_CHUNKS", "2")),
        help="Maximum prepared prompt chunks in flight when --pipeline-chunk-size is enabled.",
    )
    parser.add_argument(
        "--pipeline-teacher-concurrency",
        type=int,
        default=int(os.environ.get("OPD_PIPELINE_TEACHER_CONCURRENCY", "1")),
        help="Maximum concurrent teacher-prefill requests from this driver in pipeline mode.",
    )
    parser.add_argument(
        "--opd-async-sample-overlap",
        action="store_true",
        default=_env_bool("OPD_ASYNC_SAMPLE_OVERLAP", False),
        help=(
            "Opt-in: prepare (sample + teacher-prefill) step N+1 in a background thread while running "
            "forward_backward for step N's prepared chunks, double-buffering across steps. The prefetch is "
            "barrier-joined before optim/sync so no student sampling is in flight during the weight sync."
        ),
    )
    parser.add_argument(
        "--optim-learning-rate",
        type=float,
        default=float(os.environ.get("OPD_OPTIM_LR", "1e-4")),
        help="Learning rate passed to the trainer optim_step API.",
    )
    parser.add_argument(
        "--optim-gradient-clip",
        type=float,
        default=float(os.environ.get("OPD_OPTIM_GRADIENT_CLIP", "1.0")),
        help="Gradient clip value passed to the trainer optim_step API.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("OPD_REQUEST_TIMEOUT", "120")),
        help="HTTP timeout in seconds for student sampling and teacher prefill requests.",
    )
    parser.add_argument(
        "--forward-backward-timeout",
        type=float,
        default=float(os.environ.get("OPD_FORWARD_BACKWARD_TIMEOUT", "600")),
        help="Timeout in seconds while waiting for a trainer forward_backward future.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=float(os.environ.get("OPD_ENDPOINT_TIMEOUT", "600")),
        help="Timeout in seconds while waiting for student/teacher endpoint coord files.",
    )
    parser.add_argument(
        "--allow-unchanged-sync",
        action="store_true",
        default=os.environ.get("OPD_ALLOW_UNCHANGED_SYNC", "").lower() in {"1", "true", "yes"},
        help="For profiling, do not fail if a post-sync greedy sample matches the initial sample.",
    )
    parser.add_argument(
        "--allow-sync-http-error",
        action="store_true",
        default=_env_bool("OPD_ALLOW_SYNC_HTTP_ERROR"),
        help="Keep the legacy behavior of continuing after a non-200 sync_inference_weights response.",
    )
    parser.add_argument(
        "--allow-sync-version-mismatch",
        action="store_true",
        default=_env_bool("OPD_ALLOW_SYNC_VERSION_MISMATCH"),
        help="For profiling only, continue if SGLang does not report the expected post-sync weight_version.",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=float(os.environ.get("OPD_SYNC_TIMEOUT", "1800")),
        help="HTTP timeout in seconds for sync_inference_weights.",
    )
    parser.add_argument(
        "--sync-buffer-size-mb",
        type=int,
        default=int(os.environ.get("OPD_SYNC_BUFFER_SIZE_MB", "4096")),
        help="Weight-sync transfer bucket size.",
    )
    parser.add_argument(
        "--sync-master-port",
        type=int,
        default=int(os.environ.get("OPD_SYNC_MASTER_PORT", "0")),
        help="Weight-sync rendezvous port. Zero lets the trainer pick an ephemeral port.",
    )
    parser.add_argument(
        "--sync-pause-mode",
        choices=("retract", "abort", "in_place"),
        default=os.environ.get("OPD_SYNC_PAUSE_MODE", "retract"),
        help="How SGLang should pause requests during weight sync.",
    )
    parser.add_argument(
        "--sync-flush-cache",
        action="store_true",
        default=_env_bool("OPD_SYNC_FLUSH_CACHE"),
        help="Flush SGLang KV cache after weight sync.",
    )
    parser.add_argument(
        "--sync-weight-version-prefix",
        default=os.environ.get("OPD_SYNC_WEIGHT_VERSION_PREFIX", "opd"),
        help="Prefix for per-step SGLang weight_version values. Set empty string to disable.",
    )
    parser.add_argument(
        "--sync-quantization-json",
        default=os.environ.get("OPD_SYNC_QUANTIZATION_JSON", "null"),
        help="JSON quantization_config to pass to set_sync_quantization and per sync call. Defaults to null/bf16.",
    )
    parser.add_argument(
        "--student-sampling-params-json",
        default=os.environ.get("OPD_STUDENT_SAMPLING_PARAMS_JSON", "null"),
        help="Extra JSON object merged into SGLang student sampling_params for /generate.",
    )
    parser.add_argument(
        "--student-temperature",
        type=float,
        default=float(os.environ.get("OPD_STUDENT_TEMPERATURE", "0.7")),
        help="Student rollout sampling temperature for OPD data collection.",
    )
    parser.add_argument(
        "--student-argmax-rollout",
        action="store_true",
        default=_env_bool("OPD_STUDENT_ARGMAX_ROLLOUT", False),
        help="Force deterministic student rollouts with temperature=0, top_k=1, top_p=1, and min_p=0.",
    )
    parser.add_argument(
        "--rollout-samples-output",
        default=os.environ.get("OPD_ROLLOUT_SAMPLES_OUTPUT"),
        help="Optional JSONL path for decoded rollout/MTP training-window samples. Defaults under OPD_COORD_DIR.",
    )
    parser.add_argument(
        "--rollout-samples-per-step",
        type=int,
        default=int(os.environ.get("OPD_ROLLOUT_SAMPLES_PER_STEP", "1")),
        help="Number of rollout samples to write per OPD step. Set 0 to disable sample artifacts.",
    )
    parser.add_argument(
        "--rollout-sample-tokenizer-path",
        default=os.environ.get("OPD_ROLLOUT_SAMPLE_TOKENIZER_PATH") or os.environ.get("OPD_TOKENIZER_PATH"),
        help="Tokenizer path used to decode rollout sample snippets. Token IDs are still written if unset.",
    )
    parser.add_argument(
        "--rollout-coherence-action",
        choices=("fail", "warn", "off"),
        default=os.environ.get("OPD_ROLLOUT_COHERENCE_ACTION", "fail"),
        help=(
            "What to do when a sampled rollout exceeds the degenerate-token fraction "
            "(token id 0 / MTP mask token). 'fail' aborts the run (default), 'warn' logs an "
            "error, 'off' disables the gate. Guards against sampler token-0 collapse that "
            "loss/agreement cannot catch under frozen-teacher self-distillation."
        ),
    )
    parser.add_argument(
        "--rollout-coherence-max-degenerate-frac",
        type=float,
        default=float(os.environ.get("OPD_ROLLOUT_COHERENCE_MAX_DEGENERATE_FRAC", "0.5")),
        help="Per-rollout degenerate-token fraction above which the coherence gate triggers.",
    )
    parser.add_argument(
        "--rollout-sample-tail-tokens",
        type=int,
        default=int(os.environ.get("OPD_ROLLOUT_SAMPLE_TAIL_TOKENS", "512")),
        help="Number of tail tokens to include for prompt/sequence sample snippets.",
    )
    parser.add_argument(
        "--rollout-sample-target-tokens",
        type=int,
        default=int(os.environ.get("OPD_ROLLOUT_SAMPLE_TARGET_TOKENS", "128")),
        help="Maximum generated/training-target tokens to include in each rollout sample record.",
    )
    parser.add_argument(
        "--rollout-sample-text-max-chars",
        type=int,
        default=int(os.environ.get("OPD_ROLLOUT_SAMPLE_TEXT_MAX_CHARS", "12000")),
        help="Maximum decoded characters per text field in rollout sample records.",
    )
    parser.add_argument(
        "--singleshot-mtp-json",
        default=os.environ.get("OPD_SINGLESHOT_MTP_JSON", "null"),
        help=(
            "JSON object for loss_fn_params.singleshot_mtp. Example: "
            '\'{"k_toks": 4, "mask_token_id": 248063, "attention_bias_dtype": "bfloat16"}\'.'
        ),
    )
    parser.add_argument(
        "--prompts-json",
        default=os.environ.get(
            "OPD_PROMPTS_JSON",
            json.dumps([[5, 6, 7, 8], [11, 12, 13], [21, 22, 23, 24, 25]]),
        ),
        help="JSON-encoded list of prompt token-id lists.",
    )
    parser.add_argument(
        "--prompt-dataset-path",
        default=os.environ.get("OPD_PROMPT_DATASET_PATH"),
        help="Optional local/HF-disk tokenized dataset path to read OPD prompt prefixes from.",
    )
    parser.add_argument(
        "--prompt-dataset-type",
        default=os.environ.get("OPD_PROMPT_DATASET_TYPE"),
        help="Dataset loader type for --prompt-dataset-path, e.g. parquet, json, arrow, or hf_disk. Auto-inferred for local files.",
    )
    parser.add_argument(
        "--prompt-dataset-split",
        default=os.environ.get("OPD_PROMPT_DATASET_SPLIT", "train"),
        help="Dataset split to load when using --prompt-dataset-path.",
    )
    parser.add_argument(
        "--prompt-dataset-column",
        default=os.environ.get("OPD_PROMPT_DATASET_COLUMN", "input_ids"),
        help="Token-id column to slice into rollout prompts.",
    )
    parser.add_argument(
        "--prompt-dataset-num-prompts",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_NUM_PROMPTS", os.environ.get("OPD_NUM_PROMPTS", "2"))),
        help="Number of prompts to load from --prompt-dataset-path.",
    )
    parser.add_argument(
        "--prompt-dataset-prompt-len",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_PROMPT_LEN", os.environ.get("OPD_PROMPT_LEN", "128"))),
        help="Number of prefix tokens to keep per dataset row.",
    )
    parser.add_argument(
        "--prompt-dataset-turn-strategy",
        choices=("prefix", "assistant"),
        default=os.environ.get("OPD_PROMPT_DATASET_TURN_STRATEGY", "prefix"),
        help="How tokenized dataset rows are converted into rollout prompts.",
    )
    parser.add_argument(
        "--prompt-dataset-im-start-token-id",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_IM_START_TOKEN_ID", "151644")),
        help="ChatML <|im_start|> token id used by --prompt-dataset-turn-strategy=assistant.",
    )
    parser.add_argument(
        "--prompt-dataset-im-end-token-id",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_IM_END_TOKEN_ID", "151645")),
        help="ChatML <|im_end|> token id used by --prompt-dataset-turn-strategy=assistant.",
    )
    parser.add_argument(
        "--prompt-dataset-assistant-role-token-ids-json",
        default=os.environ.get("OPD_PROMPT_DATASET_ASSISTANT_ROLE_TOKEN_IDS_JSON", "[77091]"),
        help="JSON integer list for the assistant role tokenization, defaulting to Qwen3's 'assistant'.",
    )
    parser.add_argument(
        "--prompt-dataset-min-target-tokens",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_MIN_TARGET_TOKENS", "1")),
        help="Minimum remaining assistant-turn tokens after the selected prompt endpoint.",
    )
    parser.add_argument(
        "--prompt-dataset-assistant-offset-tokens",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_ASSISTANT_OFFSET_TOKENS", "0")),
        help="Number of assistant content tokens to include before sampling the continuation.",
    )
    parser.add_argument(
        "--prompt-dataset-offset",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_OFFSET", "0")),
        help="Starting row offset for deterministic dataset prompt selection.",
    )
    parser.add_argument(
        "--prompt-dataset-epochs",
        type=int,
        default=int(os.environ.get("OPD_PROMPT_DATASET_EPOCHS", "0")),
        help=(
            "If >0, iterate through the whole prompt dataset this many times. "
            "--prompt-dataset-num-prompts becomes the per-step batch size and --num-steps is ignored."
        ),
    )
    args = parser.parse_args()

    if not args.master_address:
        log.error("master-address must be set (via flag, XORL_TRAINER_NODE_IP, or POD_IP)")
        return 2
    teacher_head = args.teacher_head or args.teacher_model_dir
    if not teacher_head:
        log.error("--teacher-head / OPD_TEACHER_HEAD is required for trainer-side OPD KL")
        return 2
    try:
        sync_quantization = _json_arg_or_none(args.sync_quantization_json, arg_name="--sync-quantization-json")
        args.student_sampling_params = _dict_json_arg_or_none(
            args.student_sampling_params_json,
            arg_name="--student-sampling-params-json",
        )
        args.singleshot_mtp = _dict_json_arg_or_none(args.singleshot_mtp_json, arg_name="--singleshot-mtp-json")
        args.wandb_tags = _str_list_json_arg_or_none(args.wandb_tags_json, arg_name="--wandb-tags-json")
        args.prompt_dataset_assistant_role_token_ids = _int_list_json_arg(
            args.prompt_dataset_assistant_role_token_ids_json,
            arg_name="--prompt-dataset-assistant-role-token-ids-json",
            default=[77091],
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if args.student_argmax_rollout:
        args.student_temperature = 0.0
        args.student_sampling_params = dict(args.student_sampling_params or {})
        args.student_sampling_params.update({"top_k": 1, "top_p": 1.0, "min_p": 0.0})
    if args.wandb_log_interval <= 0:
        log.error("--wandb-log-interval must be positive, got %d", args.wandb_log_interval)
        return 2
    if args.max_opd_steps is not None and args.max_opd_steps <= 0:
        log.error("--max-opd-steps must be positive when set, got %d", args.max_opd_steps)
        return 2
    if args.checkpoint_interval_steps < 0:
        log.error("--checkpoint-interval-steps must be non-negative, got %d", args.checkpoint_interval_steps)
        return 2
    if args.checkpoint_keep_latest < 0:
        log.error("--checkpoint-keep-latest must be non-negative, got %d", args.checkpoint_keep_latest)
        return 2
    if args.checkpoint_timeout <= 0:
        log.error("--checkpoint-timeout must be positive, got %.3f", args.checkpoint_timeout)
        return 2
    if args.checkpoint_best_min_step < 1:
        log.error("--checkpoint-best-min-step must be >= 1, got %d", args.checkpoint_best_min_step)
        return 2
    if args.checkpoint_eval_dataset_offset < 0:
        log.error("--checkpoint-eval-dataset-offset must be non-negative, got %d", args.checkpoint_eval_dataset_offset)
        return 2
    if args.checkpoint_eval_steps < 0:
        log.error("--checkpoint-eval-steps must be non-negative, got %d", args.checkpoint_eval_steps)
        return 2
    if args.checkpoint_eval_batch_size < 0:
        log.error("--checkpoint-eval-batch-size must be non-negative, got %d", args.checkpoint_eval_batch_size)
        return 2
    if args.checkpoint_save_best and args.checkpoint_best_metric == "val_loss" and args.checkpoint_eval_steps <= 0:
        log.error("--checkpoint-best-metric=val_loss requires --checkpoint-eval-steps > 0")
        return 2
    if args.rollout_samples_per_step < 0:
        log.error("--rollout-samples-per-step must be non-negative, got %d", args.rollout_samples_per_step)
        return 2
    if args.rollout_sample_tail_tokens < 0:
        log.error("--rollout-sample-tail-tokens must be non-negative, got %d", args.rollout_sample_tail_tokens)
        return 2
    if args.rollout_sample_target_tokens < 0:
        log.error("--rollout-sample-target-tokens must be non-negative, got %d", args.rollout_sample_target_tokens)
        return 2
    if args.rollout_sample_text_max_chars < 0:
        log.error("--rollout-sample-text-max-chars must be non-negative, got %d", args.rollout_sample_text_max_chars)
        return 2
    if args.opd_emit_full_vocab_diagnostics and str(args.opd_kl_backend).lower() not in {
        "torch_compile",
        "compile",
        "auto_chunker",
    }:
        log.error(
            "--opd-emit-full-vocab-diagnostics requires a full-vocab OPD backend; got --opd-kl-backend=%s",
            args.opd_kl_backend,
        )
        return 2
    if str(args.opd_loss_mode).lower() == "hard_teacher_ce" and str(args.opd_kl_backend).lower() not in {
        "torch_compile",
        "compile",
        "auto_chunker",
    }:
        log.error(
            "--opd-loss-mode=hard_teacher_ce requires a full-vocab OPD backend; got --opd-kl-backend=%s",
            args.opd_kl_backend,
        )
        return 2
    args.sync_quantization = sync_quantization

    coord_dir = Path(args.coord_dir)
    try:
        endpoints = _resolve_endpoint_urls(args, coord_dir)
    except (ValueError, TimeoutError) as exc:
        log.error("Failed to resolve OPD endpoints: %s", exc)
        return 2
    student_sync_base_urls = endpoints["student_sync_base_urls"]
    student_route_urls = endpoints["student_route_urls"]
    teacher_base_urls = endpoints["teacher_base_urls"]
    teacher_route_urls = endpoints["teacher_route_urls"]
    expected_sampler_count = len(student_sync_base_urls)
    expected_teacher_count = len(teacher_base_urls)
    student_router = _EndpointUrlRouter(student_route_urls)
    teacher_router = _EndpointUrlRouter(teacher_route_urls)
    log.info("Student route URLs: %s", student_route_urls)
    log.info("Student sync URLs: %s", student_sync_base_urls)
    log.info("Teacher %s route URLs: %s", args.teacher_backend, teacher_route_urls)
    log.info("Teacher concrete URLs: %s", teacher_base_urls)

    log.info("Waiting for trainer to be healthy at %s", args.train_url)
    _wait_for_xorl(args.train_url)
    if args.fb_request_replay_path:
        # Trainer-only captured-payload replay: skip all sampler/teacher waits and registration.
        return _run_fb_request_replay(args)
    for idx, url in enumerate(student_sync_base_urls):
        log.info("Waiting for student SGLang %d to be healthy at %s", idx, url)
        _wait_for_sglang(url, timeout=args.endpoint_timeout)
    if args.native_route_mode == "smg":
        for url in student_route_urls:
            if url not in student_sync_base_urls:
                log.info("Waiting for student router model listing at %s", url)
                _wait_for_model_listing(url, model_hint="Qwen", timeout=args.endpoint_timeout)
    for idx, url in enumerate(teacher_base_urls):
        if args.teacher_backend == "xorl":
            log.info("Waiting for teacher XORL %d to be healthy at %s", idx, url)
            _wait_for_xorl(url, timeout=args.endpoint_timeout)
        else:
            log.info("Waiting for teacher SGLang %d to be healthy at %s", idx, url)
            _wait_for_sglang(url, timeout=args.endpoint_timeout)
    if args.teacher_backend == "sglang" and args.native_route_mode == "smg":
        for url in teacher_route_urls:
            if url not in teacher_base_urls:
                log.info("Waiting for teacher router model listing at %s", url)
                _wait_for_model_listing(url, model_hint="Qwen", timeout=args.endpoint_timeout)

    log.info("Registering %d student SGLang sync endpoints on trainer", expected_sampler_count)
    try:
        for idx, url in enumerate(student_sync_base_urls):
            log.info("Registering student sync endpoint %d: %s", idx, url)
            _register_inference_endpoint(args.train_url, url, master_address=args.master_address)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 3

    r = requests.post(
        f"{args.train_url}/api/v1/set_sync_quantization",
        json={"quantization": sync_quantization},
        timeout=10,
    )
    r.raise_for_status()

    try:
        prompt_schedule = _build_prompt_schedule(args)
    except (ValueError, KeyError, ImportError) as exc:
        log.error("Failed to load OPD prompts: %s", exc)
        return 2
    run_num_steps = prompt_schedule.num_steps
    if args.max_opd_steps is not None:
        run_num_steps = min(run_num_steps, args.max_opd_steps)
        log.info(
            "Capping OPD run to %d steps from prompt schedule total %d",
            run_num_steps,
            prompt_schedule.num_steps,
        )
    start_step = args.opd_start_step
    if start_step < 0:
        log.error("--opd-start-step must be non-negative, got %d", start_step)
        return 2
    if start_step >= run_num_steps:
        log.info(
            "Nothing to do: --opd-start-step=%d >= run_num_steps=%d; exiting cleanly",
            start_step,
            run_num_steps,
        )
        return 0
    if start_step > 0:
        log.info("Resuming OPD run at step %d (run_num_steps=%d)", start_step, run_num_steps)
    initial_prompts, initial_prompt_metadata = prompt_schedule.batch_for_step(0)
    initial_greedy_prompt = initial_prompts[0]
    log.info("Loaded OPD prompt schedule: %s", json.dumps(prompt_schedule.metadata(), sort_keys=True))
    log.info("Initial OPD prompt batch: %s", json.dumps(initial_prompt_metadata, sort_keys=True))

    eval_prompt_schedule: _PromptSchedule | None = None
    if args.checkpoint_save_best and args.checkpoint_best_metric == "val_loss":
        eval_args = argparse.Namespace(**vars(args))
        eval_args.prompt_dataset_split = args.checkpoint_eval_dataset_split
        eval_args.prompt_dataset_offset = args.checkpoint_eval_dataset_offset
        eval_args.prompt_dataset_epochs = 1
        if args.checkpoint_eval_batch_size > 0:
            eval_args.prompt_dataset_num_prompts = args.checkpoint_eval_batch_size
        try:
            eval_prompt_schedule = _build_prompt_schedule(eval_args)
        except (ValueError, KeyError, ImportError) as exc:
            log.error("Failed to load OPD checkpoint eval prompts: %s", exc)
            return 2
        log.info("Loaded OPD checkpoint eval schedule: %s", json.dumps(eval_prompt_schedule.metadata(), sort_keys=True))
    rollout_history: List[List[List[int]]] = []
    loss_history: List[float] = []
    post_sync_rollouts: List[List[int]] = []

    artifacts_dir = coord_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_output = Path(args.profile_output) if args.profile_output else artifacts_dir / "opd_profile.jsonl"
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.write_text("", encoding="utf-8")
    checkpoint_summary_output = (
        Path(args.checkpoint_summary_output)
        if args.checkpoint_summary_output
        else artifacts_dir / "checkpoint_summary.json"
    )
    latest_checkpoints: List[Dict[str, Any]] = []
    best_checkpoint: Dict[str, Any] | None = None
    best_checkpoint_metric: float | None = None
    checkpoint_policy = {
        "interval_steps": args.checkpoint_interval_steps,
        "model_id": args.checkpoint_model_id,
        "name_prefix": args.checkpoint_name_prefix,
        "keep_latest": args.checkpoint_keep_latest,
        "save_best": args.checkpoint_save_best,
        "best_metric": args.checkpoint_best_metric,
        "best_mode": args.checkpoint_best_mode,
        "best_min_step": args.checkpoint_best_min_step,
        "eval_dataset_split": args.checkpoint_eval_dataset_split,
        "eval_dataset_offset": args.checkpoint_eval_dataset_offset,
        "eval_steps": args.checkpoint_eval_steps,
        "eval_batch_size": args.checkpoint_eval_batch_size or args.prompt_dataset_num_prompts,
    }
    _write_checkpoint_summary(
        checkpoint_summary_output,
        policy=checkpoint_policy,
        latest_checkpoints=latest_checkpoints,
        best_checkpoint=best_checkpoint,
    )
    rollout_samples_output = (
        Path(args.rollout_samples_output) if args.rollout_samples_output else artifacts_dir / "rollout_samples.jsonl"
    )
    if args.rollout_samples_per_step > 0:
        rollout_samples_output.parent.mkdir(parents=True, exist_ok=True)
        rollout_samples_output.write_text("", encoding="utf-8")
        if not args.rollout_sample_tokenizer_path:
            args.rollout_sample_tokenizer_path = teacher_head if Path(teacher_head).is_dir() else None
    rollout_tokenizer = (
        _load_rollout_tokenizer(args.rollout_sample_tokenizer_path) if args.rollout_samples_per_step > 0 else None
    )
    profile_rows: List[Dict[str, Any]] = []
    try:
        wandb_run = _init_wandb(args, profile_output=profile_output, prompt_schedule=prompt_schedule)
    except Exception as exc:
        log.error("Failed to initialize W&B logging: %s", exc)
        return 2

    profile_device_count = _infer_visible_device_count(default=1)
    cumulative_tokens = 0
    cumulative_sup_tokens = 0
    cumulative_consumed_tokens = 0
    base_singleshot_mtp = args.singleshot_mtp

    initial_t0 = time.perf_counter()
    initial_student_url = student_router.next()
    initial_greedy_rollout = _student_sample(
        initial_student_url,
        initial_greedy_prompt,
        args.max_new_tokens,
        temperature=0.0,
        timeout=args.request_timeout,
        sampling_params_extra=args.student_sampling_params,
    )
    log.info(
        "Initial greedy rollout for prompt %s in %.3fs: %s",
        initial_greedy_prompt,
        _elapsed(initial_t0),
        initial_greedy_rollout,
    )
    log.info("Writing OPD throughput profile to %s", profile_output)

    def _prepare_args_for_step(target_step: int) -> argparse.Namespace:
        """Shallow-copy args with the scheduled SingleShot MTP config for ``target_step`` resolved.

        The prepare workers read ``args.singleshot_mtp``; because the async path prefetches step N+1 while
        the foreground still holds step N's config, the prefetch must carry its own resolved copy rather
        than rely on the shared mutable namespace.
        """
        step_args = argparse.Namespace(**vars(args))
        step_cfg, _ = _scheduled_singleshot_mtp_config(base_singleshot_mtp, step=target_step)
        step_args.singleshot_mtp = step_cfg
        return step_args

    async_buffer: _PreparedOpdStep | None = None
    async_prefetch: _OpdPrefetch | None = None

    for step in range(start_step, run_num_steps):
        try:
            step_singleshot_mtp, mtp_schedule_metadata = _scheduled_singleshot_mtp_config(
                base_singleshot_mtp,
                step=step,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            log.error("Failed to resolve SingleShot MTP schedule at step %d: %s", step, exc)
            return 2
        args.singleshot_mtp = step_singleshot_mtp
        prompts, prompt_metadata = prompt_schedule.batch_for_step(step)
        log.info("=== OPD step %d ===", step)
        student_smg_before = _fetch_smg_success_counters(args.student_smg_metrics_url)
        teacher_smg_before = _fetch_smg_success_counters(args.teacher_smg_metrics_url)
        step_t0 = time.perf_counter()
        profile_row: Dict[str, Any] = {"step": step, "profile_warmup": step < args.profile_warmup_steps}
        profile_row.update(prompt_metadata)
        profile_row.update(mtp_schedule_metadata)
        profile_row.update(
            {
                "opd_native_route_mode": args.native_route_mode,
                "student_endpoint_count": expected_sampler_count,
                "teacher_endpoint_count": expected_teacher_count,
                "student_route_endpoint_count": len(student_route_urls),
                "teacher_route_endpoint_count": len(teacher_route_urls),
                "student_router_policy": args.student_router_policy,
                "teacher_router_policy": args.teacher_router_policy,
                "student_smg_metrics_url": args.student_smg_metrics_url,
                "teacher_smg_metrics_url": args.teacher_smg_metrics_url,
                "student_smg_worker_count": None,
                "student_smg_worker_success_delta_min": None,
                "student_smg_worker_success_delta_max": None,
                "teacher_smg_worker_count": None,
                "teacher_smg_worker_success_delta_min": None,
                "teacher_smg_worker_success_delta_max": None,
                "student_gpu_util_mean": None,
                "teacher_gpu_util_mean": None,
                "student_node_group": os.environ.get("STUDENT_NODE_GROUP"),
                "teacher_node_group": os.environ.get("TEACHER_NODE_GROUP"),
                "trainer_node_group": os.environ.get("TRAINER_NODE_GROUP"),
            }
        )
        if isinstance(step_singleshot_mtp, dict) and "k_toks" in step_singleshot_mtp:
            log.info(
                "Resolved SingleShot MTP config for step %d: k_toks=%s prompt_pad_to_multiple=%s",
                step,
                step_singleshot_mtp.get("k_toks"),
                step_singleshot_mtp.get("prompt_pad_to_multiple"),
            )
        profile_row["profile_skip_optim_step"] = bool(args.skip_optim_step)

        pipeline_enabled = args.pipeline_chunk_size > 0 and len(prompts) > args.pipeline_chunk_size
        profile_row["opd_pipeline_enabled"] = bool(pipeline_enabled)

        if args.opd_async_sample_overlap:
            is_final_step = step == run_num_steps - 1
            if async_buffer is None:
                fill_t0 = time.perf_counter()
                async_buffer = _prepare_opd_step_chunks(
                    args=_prepare_args_for_step(step),
                    step=step,
                    prompts=prompts,
                    student_router=student_router,
                    teacher_router=teacher_router,
                    artifacts_dir=artifacts_dir,
                )
                profile_row["opd_async_initial_fill_s"] = _elapsed(fill_t0)
            else:
                profile_row["opd_async_initial_fill_s"] = 0.0

            if not is_final_step:
                next_prompts, _ = prompt_schedule.batch_for_step(step + 1)
                async_prefetch = _OpdPrefetch(
                    _prepare_opd_step_chunks,
                    step + 1,
                    next_prompts,
                    args=_prepare_args_for_step(step + 1),
                    student_router=student_router,
                    teacher_router=teacher_router,
                    artifacts_dir=artifacts_dir,
                )

            prompt_chunks = async_buffer.prompt_chunks
            max_prefetch = async_buffer.max_prefetch
            teacher_concurrency = async_buffer.teacher_concurrency
            worker_count = async_buffer.worker_count
            chunk_prepare_profiles: List[Dict[str, Any]] = []
            chunk_fb_profiles: List[Dict[str, Any]] = []
            all_sequences: List[List[int]] = []
            step_rollout_sample_records: List[Dict[str, Any]] = []
            processed_samples = 0
            total_valid_tokens = 0
            weighted_loss_sum = 0.0

            log.info(
                "Async OPD step %d: %d prompt chunks, chunk_size=%d, prefetch=%d, teacher_concurrency=%d, "
                "prefetching=%s",
                step,
                len(prompt_chunks),
                args.pipeline_chunk_size,
                max_prefetch,
                teacher_concurrency,
                not is_final_step,
            )

            for chunk_idx, prepared in enumerate(async_buffer.prepared_chunks):
                chunk_wait_s = float(prepared.metrics.get("main_prepare_wait_s", 0.0) or 0.0)
                chunk_prepare = dict(prepared.metrics)
                chunk_prepare.update(
                    _rollout_health_metrics_for_sequences(
                        prepared.prompts,
                        prepared.sequences,
                        tokenizer=rollout_tokenizer,
                        text_max_chars=args.rollout_sample_text_max_chars,
                    )
                )
                chunk_prepare["main_prepare_wait_s"] = chunk_wait_s
                chunk_prepare_profiles.append(chunk_prepare)
                all_sequences.extend([list(s) for s in prepared.sequences])
                remaining_samples = max(0, args.rollout_samples_per_step - len(step_rollout_sample_records))
                if remaining_samples:
                    step_rollout_sample_records.extend(
                        _rollout_sample_records(
                            step=step,
                            prepared=prepared,
                            step_metadata=prompt_metadata,
                            sample_offset=processed_samples,
                            sample_limit=remaining_samples,
                            tokenizer=rollout_tokenizer,
                            singleshot_mtp=args.singleshot_mtp,
                            tail_tokens=args.rollout_sample_tail_tokens,
                            target_tokens=args.rollout_sample_target_tokens,
                            text_max_chars=args.rollout_sample_text_max_chars,
                        )
                    )
                processed_samples += len(prepared.prompts)
                log.info(
                    "Prepared OPD chunk %d/%d: sample=%.3fs teacher=%.3fs wait=%.3fs tokens=%d cache=%s",
                    chunk_idx + 1,
                    len(prompt_chunks),
                    prepared.metrics["student_sampling_s"],
                    prepared.metrics["teacher_prefill_s"],
                    chunk_wait_s,
                    prepared.metrics["teacher_prefill_tokens"],
                    prepared.cache_path,
                )

                fb, fb_timing = _run_forward_backward_chunk(
                    args=args,
                    train_url=args.train_url,
                    data=prepared.data,
                    teacher_head=teacher_head,
                    cache_path=prepared.cache_path,
                    clear_gradients_after_backward=args.skip_optim_step and chunk_idx == len(prompt_chunks) - 1,
                )
                cache_deleted = _delete_consumed_teacher_cache(prepared.cache_path)
                fb_metrics = fb["metrics"]
                chunk_loss = _scalar(fb["loss_fn_outputs"][0]["loss"])
                chunk_valid = int(_metric(fb_metrics, "valid_tokens", 0))
                weighted_loss_sum += chunk_loss * chunk_valid
                total_valid_tokens += chunk_valid
                chunk_fb = {
                    "chunk_idx": chunk_idx,
                    **fb_timing,
                    "teacher_cache_deleted": cache_deleted,
                    "loss": chunk_loss,
                    "valid_tokens": chunk_valid,
                    "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                    "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                    "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                    "trainer_opd_total_ms": float(_metric(fb_metrics, "opd_profile_total_ms", 0.0)),
                    "trainer_opd_prefetch_ms": float(_metric(fb_metrics, "opd_profile_prefetch_ms", 0.0)),
                    "trainer_opd_hidden_fetch_ms": float(_metric(fb_metrics, "opd_profile_hidden_fetch_ms", 0.0)),
                    "trainer_opd_head_prepare_ms": float(_metric(fb_metrics, "opd_profile_head_prepare_ms", 0.0)),
                    "trainer_opd_kl_compute_ms": float(_metric(fb_metrics, "opd_profile_kl_compute_ms", 0.0)),
                    "trainer_clear_gradients_ms": float(_metric(fb_metrics, "opd_profile_clear_gradients_ms", 0.0)),
                    **_opd_loss_metrics_from_response(fb_metrics),
                    **_singleshot_trainer_metrics_from_response(fb_metrics),
                }
                chunk_fb_profiles.append(chunk_fb)
                log.info(
                    "forward_backward step=%d chunk=%d/%d loss=%.4f valid_tokens=%s roundtrip=%.3fs "
                    "trainer=%.3fs forward_loss=%.3fs backward=%.3fs opd_kl=%.1fms",
                    step,
                    chunk_idx + 1,
                    len(prompt_chunks),
                    chunk_loss,
                    chunk_valid,
                    chunk_fb["forward_backward_roundtrip_s"],
                    chunk_fb["trainer_forward_backward_s"],
                    chunk_fb["trainer_forward_loss_s"],
                    chunk_fb["trainer_backward_s"],
                    chunk_fb["trainer_opd_kl_compute_ms"],
                )

            async_buffer_next: _PreparedOpdStep | None = None
            barrier_t0 = time.perf_counter()
            if async_prefetch is not None:
                async_buffer_next = async_prefetch.join()
                async_prefetch = None
            profile_row["opd_async_prefetch_wait_s"] = _elapsed(barrier_t0)

            rollout_history.append(all_sequences)
            _append_rollout_sample_records(rollout_samples_output, step_rollout_sample_records)
            _log_rollout_samples_to_wandb(wandb_run, step_rollout_sample_records, step=step)
            loss = weighted_loss_sum / total_valid_tokens if total_valid_tokens > 0 else 0.0
            loss_history.append(loss)
            profile_row.update(
                {
                    "loss": loss,
                    "rollout_samples_written": len(step_rollout_sample_records),
                    "opd_pipeline_chunks": len(prompt_chunks),
                    "opd_pipeline_chunk_size": args.pipeline_chunk_size,
                    "opd_pipeline_prefetch_chunks": max_prefetch,
                    "opd_pipeline_prepare_workers": worker_count,
                    "opd_pipeline_teacher_concurrency": teacher_concurrency,
                    "opd_pipeline_prepare_wait_s": 0.0,
                    "opd_pipeline_chunk_prepare": chunk_prepare_profiles,
                    "opd_pipeline_chunk_forward_backward": chunk_fb_profiles,
                    "student_sampling_s": _sum_metric(chunk_prepare_profiles, "student_sampling_s"),
                    "student_sampling_batch_size": len(prompts),
                    "student_sampling_prompt_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_prompt_tokens")
                    ),
                    "student_sampling_output_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_output_tokens")
                    ),
                    "student_sampling_mtp_generate_calls": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_generate_calls")
                    ),
                    "student_sampling_mtp_blocks": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_blocks")
                    ),
                    "student_sampling_mtp_partial_blocks": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_partial_blocks")
                    ),
                    "teacher_prefill_queue_wait_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_queue_wait_s"),
                    "teacher_prefill_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_s"),
                    "teacher_prefill_tokens": int(_sum_metric(chunk_prepare_profiles, "teacher_prefill_tokens")),
                    "teacher_prefill_forward_compute_s": _sum_metric(
                        chunk_prepare_profiles, "teacher_prefill_forward_compute_s"
                    ),
                    "teacher_hidden_cache_write_s": _sum_metric(chunk_prepare_profiles, "teacher_hidden_cache_write_s"),
                    "teacher_cache_save_s": _sum_metric(chunk_prepare_profiles, "teacher_cache_save_s"),
                    "forward_backward_enqueue_s": _sum_metric(chunk_fb_profiles, "forward_backward_enqueue_s"),
                    "forward_backward_wait_s": _sum_metric(chunk_fb_profiles, "forward_backward_wait_s"),
                    "forward_backward_roundtrip_s": _sum_metric(chunk_fb_profiles, "forward_backward_roundtrip_s"),
                    "trainer_forward_backward_s": _sum_metric(chunk_fb_profiles, "trainer_forward_backward_s"),
                    "trainer_forward_loss_s": _sum_metric(chunk_fb_profiles, "trainer_forward_loss_s"),
                    "trainer_backward_s": _sum_metric(chunk_fb_profiles, "trainer_backward_s"),
                    "trainer_opd_total_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_total_ms"),
                    "trainer_opd_prefetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_prefetch_ms"),
                    "trainer_opd_hidden_fetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_hidden_fetch_ms"),
                    "trainer_opd_head_prepare_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_head_prepare_ms"),
                    "trainer_opd_kl_compute_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_kl_compute_ms"),
                    "trainer_clear_gradients_ms": _sum_metric(chunk_fb_profiles, "trainer_clear_gradients_ms"),
                    "valid_tokens": total_valid_tokens,
                    "opd_async_sample_overlap": True,
                }
            )
            for key in _OPD_LOSS_PROFILE_METRICS:
                weighted_value = _weighted_mean_metric(chunk_fb_profiles, key)
                if weighted_value is not None:
                    profile_row[key] = weighted_value
            _add_singleshot_trainer_profile_totals(profile_row, chunk_fb_profiles)
            sampling_s = profile_row["student_sampling_s"]
            teacher_prefill_s = profile_row["teacher_prefill_s"]
            profile_row["student_sampling_output_tok_per_s"] = (
                profile_row["student_sampling_output_tokens"] / sampling_s if sampling_s > 0 else 0.0
            )
            profile_row["teacher_prefill_tok_per_s"] = (
                profile_row["teacher_prefill_tokens"] / teacher_prefill_s if teacher_prefill_s > 0 else 0.0
            )
            sampling_modes = sorted(
                {
                    str(row["student_sampling_mode"])
                    for row in chunk_prepare_profiles
                    if row.get("student_sampling_mode") is not None
                }
            )
            if sampling_modes:
                profile_row["student_sampling_mode"] = ",".join(sampling_modes)
            profile_row.update(_aggregate_native_mtp_trace_metrics(chunk_prepare_profiles))
            profile_row.update(_aggregate_rollout_health_metrics(chunk_prepare_profiles))
            profile_row.update(_rollout_sample_invariant_metrics(step_rollout_sample_records))
            log.info(
                "Async forward_backward step=%d loss=%.4f valid_tokens=%d prefetch_wait=%.3fs "
                "sample_sum=%.3fs teacher_sum=%.3fs fb_sum=%.3fs",
                step,
                loss,
                total_valid_tokens,
                profile_row["opd_async_prefetch_wait_s"],
                profile_row["student_sampling_s"],
                profile_row["teacher_prefill_s"],
                profile_row["forward_backward_roundtrip_s"],
            )
            async_buffer = async_buffer_next
        elif not pipeline_enabled:
            prepared = _prepare_opd_chunk(
                args=args,
                student_url=student_router.next(),
                teacher_url=teacher_router.next(),
                prompts=prompts,
                artifacts_dir=artifacts_dir,
                step=step,
                chunk_idx=0,
                cache_path=artifacts_dir / f"teacher_hidden_step{step}.safetensors",
            )
            profile_row.update(prepared.metrics)
            rollout_history.append([list(s) for s in prepared.sequences])
            rollout_sample_records = _rollout_sample_records(
                step=step,
                prepared=prepared,
                step_metadata=prompt_metadata,
                sample_offset=0,
                sample_limit=args.rollout_samples_per_step,
                tokenizer=rollout_tokenizer,
                singleshot_mtp=args.singleshot_mtp,
                tail_tokens=args.rollout_sample_tail_tokens,
                target_tokens=args.rollout_sample_target_tokens,
                text_max_chars=args.rollout_sample_text_max_chars,
            )
            _append_rollout_sample_records(rollout_samples_output, rollout_sample_records)
            _log_rollout_samples_to_wandb(wandb_run, rollout_sample_records, step=step)
            profile_row["rollout_samples_written"] = len(rollout_sample_records)
            profile_row.update(_rollout_sample_invariant_metrics(rollout_sample_records))
            profile_row.update(
                _rollout_health_metrics_for_sequences(
                    prepared.prompts,
                    prepared.sequences,
                    tokenizer=rollout_tokenizer,
                    text_max_chars=args.rollout_sample_text_max_chars,
                )
            )
            if rollout_sample_records:
                log.info("Wrote %d rollout sample records to %s", len(rollout_sample_records), rollout_samples_output)
            log.info(
                "Sampled %d sequences from student SGLang in %.3fs (%.1f new tok/s)",
                len(prepared.sequences),
                profile_row["student_sampling_s"],
                profile_row["student_sampling_output_tok_per_s"],
            )
            log.info(
                "Fetched teacher prefill for %d tokens in %.3fs (%.1f tok/s); cache write %.3fs: %s",
                profile_row["teacher_prefill_tokens"],
                profile_row["teacher_prefill_s"],
                profile_row["teacher_prefill_tok_per_s"],
                profile_row.get("teacher_hidden_cache_write_s", profile_row["teacher_cache_save_s"]),
                prepared.cache_path,
            )

            fb, fb_timing = _run_forward_backward_chunk(
                args=args,
                train_url=args.train_url,
                data=prepared.data,
                teacher_head=teacher_head,
                cache_path=prepared.cache_path,
                clear_gradients_after_backward=args.skip_optim_step,
            )
            profile_row["teacher_cache_deleted"] = _delete_consumed_teacher_cache(prepared.cache_path)
            profile_row.update(fb_timing)
            loss = _scalar(fb["loss_fn_outputs"][0]["loss"])
            loss_history.append(loss)
            fb_metrics = fb["metrics"]
            profile_row.update(
                {
                    "loss": loss,
                    "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                    "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                    "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                    "trainer_opd_total_ms": float(_metric(fb_metrics, "opd_profile_total_ms", 0.0)),
                    "trainer_opd_prefetch_ms": float(_metric(fb_metrics, "opd_profile_prefetch_ms", 0.0)),
                    "trainer_opd_hidden_fetch_ms": float(_metric(fb_metrics, "opd_profile_hidden_fetch_ms", 0.0)),
                    "trainer_opd_head_prepare_ms": float(_metric(fb_metrics, "opd_profile_head_prepare_ms", 0.0)),
                    "trainer_opd_kl_compute_ms": float(_metric(fb_metrics, "opd_profile_kl_compute_ms", 0.0)),
                    "trainer_clear_gradients_ms": float(_metric(fb_metrics, "opd_profile_clear_gradients_ms", 0.0)),
                    "valid_tokens": int(_metric(fb_metrics, "valid_tokens", 0)),
                }
            )
            profile_row.update(_opd_loss_metrics_from_response(fb_metrics))
            profile_row.update(_singleshot_trainer_metrics_from_response(fb_metrics))
            _add_singleshot_trainer_profile_totals(profile_row, [dict(profile_row)])
            log.info(
                "forward_backward step=%d loss=%.4f valid_tokens=%s roundtrip=%.3fs trainer=%.3fs "
                "forward_loss=%.3fs backward=%.3fs opd_kl=%.1fms",
                step,
                loss,
                fb_metrics.get("valid_tokens:sum"),
                profile_row["forward_backward_roundtrip_s"],
                profile_row["trainer_forward_backward_s"],
                profile_row["trainer_forward_loss_s"],
                profile_row["trainer_backward_s"],
                profile_row["trainer_opd_kl_compute_ms"],
            )
        else:
            prompt_chunks = _chunked(prompts, args.pipeline_chunk_size)
            max_prefetch = max(1, min(args.pipeline_prefetch_chunks, len(prompt_chunks)))
            teacher_concurrency = max(1, args.pipeline_teacher_concurrency)
            worker_count = max(1, min(max_prefetch, len(prompt_chunks)))
            teacher_semaphore = Semaphore(teacher_concurrency)
            chunk_prepare_profiles: List[Dict[str, Any]] = []
            chunk_fb_profiles: List[Dict[str, Any]] = []
            all_sequences: List[List[int]] = []
            step_rollout_sample_records: List[Dict[str, Any]] = []
            processed_samples = 0
            total_valid_tokens = 0
            weighted_loss_sum = 0.0
            prepare_wait_s = 0.0

            log.info(
                "Pipeline OPD step %d: %d prompt chunks, chunk_size=%d, prefetch=%d, teacher_concurrency=%d",
                step,
                len(prompt_chunks),
                args.pipeline_chunk_size,
                max_prefetch,
                teacher_concurrency,
            )

            job_queue: queue_module.Queue = queue_module.Queue()
            output_queue: queue_module.Queue = queue_module.Queue(maxsize=max_prefetch)
            stop_event = threading.Event()
            for chunk_idx, prompt_chunk in enumerate(prompt_chunks):
                job_queue.put((chunk_idx, prompt_chunk))
            for _ in range(worker_count):
                job_queue.put(None)

            workers = [
                threading.Thread(
                    target=_opd_prepare_worker,
                    kwargs={
                        "worker_idx": worker_idx,
                        "job_queue": job_queue,
                        "output_queue": output_queue,
                        "stop_event": stop_event,
                        "args": args,
                        "student_router": student_router,
                        "teacher_router": teacher_router,
                        "artifacts_dir": artifacts_dir,
                        "step": step,
                        "teacher_semaphore": teacher_semaphore,
                    },
                    name=f"opd-prepare-{worker_idx}",
                    daemon=True,
                )
                for worker_idx in range(worker_count)
            ]
            for worker in workers:
                worker.start()

            pending_chunks: Dict[int, _PreparedOpdChunk] = {}
            finished_workers = 0
            try:
                for chunk_idx in range(len(prompt_chunks)):
                    wait_t0 = time.perf_counter()
                    while chunk_idx not in pending_chunks:
                        try:
                            kind, item_idx, payload = output_queue.get(timeout=1.0)
                        except queue_module.Empty:
                            if finished_workers >= worker_count and not pending_chunks:
                                raise RuntimeError(
                                    f"OPD prepare workers exited before chunk {chunk_idx} became available"
                                )
                            continue

                        if kind == "ok":
                            pending_chunks[item_idx] = payload
                        elif kind == "error":
                            raise RuntimeError(f"OPD prepare worker failed on chunk {item_idx}: {payload}") from payload
                        elif kind == "done":
                            finished_workers += 1
                        else:
                            raise RuntimeError(f"Unexpected OPD prepare queue item kind: {kind!r}")

                    prepared = pending_chunks.pop(chunk_idx)
                    chunk_wait_s = _elapsed(wait_t0)
                    prepare_wait_s += chunk_wait_s

                    chunk_prepare = dict(prepared.metrics)
                    chunk_prepare.update(
                        _rollout_health_metrics_for_sequences(
                            prepared.prompts,
                            prepared.sequences,
                            tokenizer=rollout_tokenizer,
                            text_max_chars=args.rollout_sample_text_max_chars,
                        )
                    )
                    chunk_prepare["main_prepare_wait_s"] = chunk_wait_s
                    chunk_prepare_profiles.append(chunk_prepare)
                    all_sequences.extend([list(s) for s in prepared.sequences])
                    remaining_samples = max(0, args.rollout_samples_per_step - len(step_rollout_sample_records))
                    if remaining_samples:
                        step_rollout_sample_records.extend(
                            _rollout_sample_records(
                                step=step,
                                prepared=prepared,
                                step_metadata=prompt_metadata,
                                sample_offset=processed_samples,
                                sample_limit=remaining_samples,
                                tokenizer=rollout_tokenizer,
                                singleshot_mtp=args.singleshot_mtp,
                                tail_tokens=args.rollout_sample_tail_tokens,
                                target_tokens=args.rollout_sample_target_tokens,
                                text_max_chars=args.rollout_sample_text_max_chars,
                            )
                        )
                    processed_samples += len(prepared.prompts)
                    log.info(
                        "Prepared OPD chunk %d/%d: sample=%.3fs teacher=%.3fs wait=%.3fs tokens=%d cache=%s",
                        chunk_idx + 1,
                        len(prompt_chunks),
                        prepared.metrics["student_sampling_s"],
                        prepared.metrics["teacher_prefill_s"],
                        chunk_wait_s,
                        prepared.metrics["teacher_prefill_tokens"],
                        prepared.cache_path,
                    )

                    fb, fb_timing = _run_forward_backward_chunk(
                        args=args,
                        train_url=args.train_url,
                        data=prepared.data,
                        teacher_head=teacher_head,
                        cache_path=prepared.cache_path,
                        clear_gradients_after_backward=args.skip_optim_step and chunk_idx == len(prompt_chunks) - 1,
                    )
                    cache_deleted = _delete_consumed_teacher_cache(prepared.cache_path)
                    fb_metrics = fb["metrics"]
                    chunk_loss = _scalar(fb["loss_fn_outputs"][0]["loss"])
                    chunk_valid = int(_metric(fb_metrics, "valid_tokens", 0))
                    weighted_loss_sum += chunk_loss * chunk_valid
                    total_valid_tokens += chunk_valid
                    chunk_fb = {
                        "chunk_idx": chunk_idx,
                        **fb_timing,
                        "teacher_cache_deleted": cache_deleted,
                        "loss": chunk_loss,
                        "valid_tokens": chunk_valid,
                        "trainer_forward_backward_s": float(_metric(fb_metrics, "execution_time", 0.0)),
                        "trainer_forward_loss_s": float(_metric(fb_metrics, "opd_profile_forward_compute_s", 0.0)),
                        "trainer_backward_s": float(_metric(fb_metrics, "opd_profile_backward_compute_s", 0.0)),
                        "trainer_opd_total_ms": float(_metric(fb_metrics, "opd_profile_total_ms", 0.0)),
                        "trainer_opd_prefetch_ms": float(_metric(fb_metrics, "opd_profile_prefetch_ms", 0.0)),
                        "trainer_opd_hidden_fetch_ms": float(_metric(fb_metrics, "opd_profile_hidden_fetch_ms", 0.0)),
                        "trainer_opd_head_prepare_ms": float(_metric(fb_metrics, "opd_profile_head_prepare_ms", 0.0)),
                        "trainer_opd_kl_compute_ms": float(_metric(fb_metrics, "opd_profile_kl_compute_ms", 0.0)),
                        "trainer_clear_gradients_ms": float(_metric(fb_metrics, "opd_profile_clear_gradients_ms", 0.0)),
                        **_opd_loss_metrics_from_response(fb_metrics),
                        **_singleshot_trainer_metrics_from_response(fb_metrics),
                    }
                    chunk_fb_profiles.append(chunk_fb)
                    log.info(
                        "forward_backward step=%d chunk=%d/%d loss=%.4f valid_tokens=%s roundtrip=%.3fs "
                        "trainer=%.3fs forward_loss=%.3fs backward=%.3fs opd_kl=%.1fms",
                        step,
                        chunk_idx + 1,
                        len(prompt_chunks),
                        chunk_loss,
                        chunk_valid,
                        chunk_fb["forward_backward_roundtrip_s"],
                        chunk_fb["trainer_forward_backward_s"],
                        chunk_fb["trainer_forward_loss_s"],
                        chunk_fb["trainer_backward_s"],
                        chunk_fb["trainer_opd_kl_compute_ms"],
                    )
            finally:
                stop_event.set()
                for worker in workers:
                    worker.join(timeout=30.0)
                    if worker.is_alive():
                        log.warning("OPD prepare worker %s did not stop within timeout", worker.name)

            rollout_history.append(all_sequences)
            _append_rollout_sample_records(rollout_samples_output, step_rollout_sample_records)
            _log_rollout_samples_to_wandb(wandb_run, step_rollout_sample_records, step=step)
            loss = weighted_loss_sum / total_valid_tokens if total_valid_tokens > 0 else 0.0
            loss_history.append(loss)
            profile_row.update(
                {
                    "loss": loss,
                    "rollout_samples_written": len(step_rollout_sample_records),
                    "opd_pipeline_chunks": len(prompt_chunks),
                    "opd_pipeline_chunk_size": args.pipeline_chunk_size,
                    "opd_pipeline_prefetch_chunks": max_prefetch,
                    "opd_pipeline_prepare_workers": worker_count,
                    "opd_pipeline_teacher_concurrency": teacher_concurrency,
                    "opd_pipeline_prepare_wait_s": prepare_wait_s,
                    "opd_pipeline_chunk_prepare": chunk_prepare_profiles,
                    "opd_pipeline_chunk_forward_backward": chunk_fb_profiles,
                    "student_sampling_s": _sum_metric(chunk_prepare_profiles, "student_sampling_s"),
                    "student_sampling_batch_size": len(prompts),
                    "student_sampling_prompt_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_prompt_tokens")
                    ),
                    "student_sampling_output_tokens": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_output_tokens")
                    ),
                    "student_sampling_mtp_generate_calls": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_generate_calls")
                    ),
                    "student_sampling_mtp_blocks": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_blocks")
                    ),
                    "student_sampling_mtp_partial_blocks": int(
                        _sum_metric(chunk_prepare_profiles, "student_sampling_mtp_partial_blocks")
                    ),
                    "teacher_prefill_queue_wait_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_queue_wait_s"),
                    "teacher_prefill_s": _sum_metric(chunk_prepare_profiles, "teacher_prefill_s"),
                    "teacher_prefill_tokens": int(_sum_metric(chunk_prepare_profiles, "teacher_prefill_tokens")),
                    "teacher_prefill_forward_compute_s": _sum_metric(
                        chunk_prepare_profiles, "teacher_prefill_forward_compute_s"
                    ),
                    "teacher_hidden_cache_write_s": _sum_metric(chunk_prepare_profiles, "teacher_hidden_cache_write_s"),
                    "teacher_cache_save_s": _sum_metric(chunk_prepare_profiles, "teacher_cache_save_s"),
                    "forward_backward_enqueue_s": _sum_metric(chunk_fb_profiles, "forward_backward_enqueue_s"),
                    "forward_backward_wait_s": _sum_metric(chunk_fb_profiles, "forward_backward_wait_s"),
                    "forward_backward_roundtrip_s": _sum_metric(chunk_fb_profiles, "forward_backward_roundtrip_s"),
                    "trainer_forward_backward_s": _sum_metric(chunk_fb_profiles, "trainer_forward_backward_s"),
                    "trainer_forward_loss_s": _sum_metric(chunk_fb_profiles, "trainer_forward_loss_s"),
                    "trainer_backward_s": _sum_metric(chunk_fb_profiles, "trainer_backward_s"),
                    "trainer_opd_total_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_total_ms"),
                    "trainer_opd_prefetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_prefetch_ms"),
                    "trainer_opd_hidden_fetch_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_hidden_fetch_ms"),
                    "trainer_opd_head_prepare_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_head_prepare_ms"),
                    "trainer_opd_kl_compute_ms": _sum_metric(chunk_fb_profiles, "trainer_opd_kl_compute_ms"),
                    "trainer_clear_gradients_ms": _sum_metric(chunk_fb_profiles, "trainer_clear_gradients_ms"),
                    "valid_tokens": total_valid_tokens,
                }
            )
            for key in _OPD_LOSS_PROFILE_METRICS:
                weighted_value = _weighted_mean_metric(chunk_fb_profiles, key)
                if weighted_value is not None:
                    profile_row[key] = weighted_value
            _add_singleshot_trainer_profile_totals(profile_row, chunk_fb_profiles)
            sampling_s = profile_row["student_sampling_s"]
            teacher_prefill_s = profile_row["teacher_prefill_s"]
            profile_row["student_sampling_output_tok_per_s"] = (
                profile_row["student_sampling_output_tokens"] / sampling_s if sampling_s > 0 else 0.0
            )
            profile_row["teacher_prefill_tok_per_s"] = (
                profile_row["teacher_prefill_tokens"] / teacher_prefill_s if teacher_prefill_s > 0 else 0.0
            )
            sampling_modes = sorted(
                {
                    str(row["student_sampling_mode"])
                    for row in chunk_prepare_profiles
                    if row.get("student_sampling_mode") is not None
                }
            )
            if sampling_modes:
                profile_row["student_sampling_mode"] = ",".join(sampling_modes)
            profile_row.update(_aggregate_native_mtp_trace_metrics(chunk_prepare_profiles))
            profile_row.update(_aggregate_rollout_health_metrics(chunk_prepare_profiles))
            profile_row.update(_rollout_sample_invariant_metrics(step_rollout_sample_records))
            log.info(
                "Pipeline forward_backward step=%d loss=%.4f valid_tokens=%d prepare_wait=%.3fs "
                "sample_sum=%.3fs teacher_sum=%.3fs fb_sum=%.3fs",
                step,
                loss,
                total_valid_tokens,
                prepare_wait_s,
                profile_row["student_sampling_s"],
                profile_row["teacher_prefill_s"],
                profile_row["forward_backward_roundtrip_s"],
            )

        cumulative_tokens += int(profile_row.get("teacher_prefill_tokens", 0) or 0)
        cumulative_sup_tokens += int(profile_row.get("valid_tokens", 0) or 0)
        cumulative_consumed_tokens += int(profile_row.get("student_sampling_prompt_tokens", 0) or 0) + int(
            profile_row.get("student_sampling_output_tokens", 0) or 0
        )

        if args.skip_optim_step:
            profile_row.update(
                {
                    "optim_step_enqueue_s": 0.0,
                    "optim_step_wait_s": 0.0,
                    "optim_step_roundtrip_s": 0.0,
                    "sync_inference_weights_s": 0.0,
                    "post_sync_greedy_sample_s": 0.0,
                    "step_total_s": _elapsed(step_t0),
                }
            )
            _add_route_observation_profile(
                profile_row,
                args,
                student_smg_before=student_smg_before,
                teacher_smg_before=teacher_smg_before,
            )
            _add_singleshot_profile_aliases(
                profile_row,
                args,
                device_count=profile_device_count,
                cumulative_tokens=cumulative_tokens,
                cumulative_sup_tokens=cumulative_sup_tokens,
                cumulative_consumed_tokens=cumulative_consumed_tokens,
            )
            _record_profile_row(
                profile_rows,
                profile_output,
                profile_row,
                wandb_run=wandb_run,
                wandb_log_interval=args.wandb_log_interval,
            )
            log.info("Skipping optim_step/sync because --skip-optim-step is set.")
            log.info("OPD throughput step=%d %s", step, json.dumps(profile_row, sort_keys=True))
            continue

        opt_t0 = time.perf_counter()
        r = requests.post(
            f"{args.train_url}/api/v1/optim_step",
            json={
                "model_id": "default",
                "adam_params": {"learning_rate": args.optim_learning_rate, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
                "gradient_clip": args.optim_gradient_clip,
            },
            timeout=120,
        )
        r.raise_for_status()
        profile_row["optim_step_enqueue_s"] = _elapsed(opt_t0)
        opt_wait_t0 = time.perf_counter()
        opt = _wait_for_future(args.train_url, r.json()["request_id"], timeout=300.0)
        profile_row["optim_step_wait_s"] = _elapsed(opt_wait_t0)
        profile_row["optim_step_roundtrip_s"] = _elapsed(opt_t0)
        grad_norm = _scalar(opt["metrics"]["grad_norm"])
        profile_row["grad_norm"] = grad_norm
        log.info(
            "optim_step grad_norm=%.4f roundtrip=%.3fs",
            grad_norm,
            profile_row["optim_step_roundtrip_s"],
        )

        weight_version = f"{args.sync_weight_version_prefix}-step{step}" if args.sync_weight_version_prefix else None
        sync_payload = {
            "master_address": args.master_address,
            "master_port": args.sync_master_port,
            "buffer_size_mb": args.sync_buffer_size_mb,
            "flush_cache": args.sync_flush_cache,
            "pause_mode": args.sync_pause_mode,
            "weight_version": weight_version,
            "quantization": sync_quantization,
        }
        log.info(
            "Calling sync_inference_weights with master_address=%s buffer_size_mb=%d pause_mode=%s weight_version=%s",
            args.master_address,
            args.sync_buffer_size_mb,
            args.sync_pause_mode,
            weight_version,
        )
        sync_t0 = time.perf_counter()
        sync_response: Dict[str, Any] = {}
        try:
            r = requests.post(
                f"{args.train_url}/api/v1/sync_inference_weights",
                json=sync_payload,
                timeout=args.sync_timeout,
            )
            sync_status = r.status_code
            try:
                sync_response = r.json()
            except ValueError:
                sync_response = {"success": False, "message": r.text[:1000]}
        except requests.exceptions.RequestException as exc:
            sync_status = -1
            sync_response = {"success": False, "message": str(exc)}
            log.warning("sync_inference_weights HTTP error: %s", exc)
        profile_row["sync_inference_weights_s"] = _elapsed(sync_t0)
        profile_row["sync_http_status"] = sync_status
        _add_sync_response_profile(profile_row, sync_response)
        log.info(
            "sync_inference_weights returned status=%s success=%s transfer_time=%.3fs bytes=%s buckets=%s in %.1fs",
            sync_status,
            sync_response.get("success"),
            profile_row["sync_transfer_time_s"],
            profile_row["sync_total_bytes"],
            profile_row["sync_num_buckets"],
            profile_row["sync_inference_weights_s"],
        )
        if sync_status != 200 and not args.allow_sync_http_error:
            log.error("sync_inference_weights failed with status=%s response=%s", sync_status, sync_response)
            profile_row["step_total_s"] = _elapsed(step_t0)
            _add_route_observation_profile(
                profile_row,
                args,
                student_smg_before=student_smg_before,
                teacher_smg_before=teacher_smg_before,
            )
            _add_singleshot_profile_aliases(
                profile_row,
                args,
                device_count=profile_device_count,
                cumulative_tokens=cumulative_tokens,
                cumulative_sup_tokens=cumulative_sup_tokens,
                cumulative_consumed_tokens=cumulative_consumed_tokens,
            )
            _record_profile_row(
                profile_rows,
                profile_output,
                profile_row,
                wandb_run=wandb_run,
                wandb_log_interval=args.wandb_log_interval,
            )
            return 6
        if sync_status == 200 and not sync_response.get("success", False):
            log.error("sync_inference_weights returned success=false: %s", sync_response)
            profile_row["step_total_s"] = _elapsed(step_t0)
            _add_route_observation_profile(
                profile_row,
                args,
                student_smg_before=student_smg_before,
                teacher_smg_before=teacher_smg_before,
            )
            _add_singleshot_profile_aliases(
                profile_row,
                args,
                device_count=profile_device_count,
                cumulative_tokens=cumulative_tokens,
                cumulative_sup_tokens=cumulative_sup_tokens,
                cumulative_consumed_tokens=cumulative_consumed_tokens,
            )
            _record_profile_row(
                profile_rows,
                profile_output,
                profile_row,
                wandb_run=wandb_run,
                wandb_log_interval=args.wandb_log_interval,
            )
            return 7

        if (
            profile_row.get("sync_endpoint_count") != expected_sampler_count
            or profile_row.get("sync_endpoint_success_count") != expected_sampler_count
        ):
            log.error(
                "sync_inference_weights did not sync every expected sampler endpoint: "
                "expected=%d count=%s success_count=%s response=%s",
                expected_sampler_count,
                profile_row.get("sync_endpoint_count"),
                profile_row.get("sync_endpoint_success_count"),
                sync_response,
            )
            profile_row["step_total_s"] = _elapsed(step_t0)
            _add_route_observation_profile(
                profile_row,
                args,
                student_smg_before=student_smg_before,
                teacher_smg_before=teacher_smg_before,
            )
            _add_singleshot_profile_aliases(
                profile_row,
                args,
                device_count=profile_device_count,
                cumulative_tokens=cumulative_tokens,
                cumulative_sup_tokens=cumulative_sup_tokens,
                cumulative_consumed_tokens=cumulative_consumed_tokens,
            )
            _record_profile_row(
                profile_rows,
                profile_output,
                profile_row,
                wandb_run=wandb_run,
                wandb_log_interval=args.wandb_log_interval,
            )
            return 9

        if weight_version:
            version_errors: List[str] = []
            for endpoint_idx, url in enumerate(student_sync_base_urls):
                endpoint_profile: Dict[str, Any] = {}
                version_ok, version_error = _verify_student_weight_version(
                    url,
                    weight_version,
                    endpoint_profile,
                    timeout=30.0,
                )
                prefix = f"student_endpoint_{endpoint_idx}"
                for key, value in endpoint_profile.items():
                    profile_row[f"{prefix}_{key}"] = value
                if not version_ok and version_error:
                    version_errors.append(f"{url}: {version_error}")
            if version_errors and not args.allow_sync_version_mismatch:
                log.error("SGLang weight_version verification failed: %s", "; ".join(version_errors))
                profile_row["step_total_s"] = _elapsed(step_t0)
                _add_route_observation_profile(
                    profile_row,
                    args,
                    student_smg_before=student_smg_before,
                    teacher_smg_before=teacher_smg_before,
                )
                _add_singleshot_profile_aliases(
                    profile_row,
                    args,
                    device_count=profile_device_count,
                    cumulative_tokens=cumulative_tokens,
                    cumulative_sup_tokens=cumulative_sup_tokens,
                    cumulative_consumed_tokens=cumulative_consumed_tokens,
                )
                _record_profile_row(
                    profile_rows,
                    profile_output,
                    profile_row,
                    wandb_run=wandb_run,
                    wandb_log_interval=args.wandb_log_interval,
                )
                return 8
            if version_errors:
                log.warning(
                    "SGLang weight_version verification failed but continuing because "
                    "--allow-sync-version-mismatch is set: %s",
                    "; ".join(version_errors),
                )

        post_sync_sample_t0 = time.perf_counter()
        post_sync_student_url = student_router.next()
        sample_after_sync = _student_sample(
            post_sync_student_url,
            initial_greedy_prompt,
            args.max_new_tokens,
            temperature=0.0,
            timeout=args.request_timeout,
            sampling_params_extra=args.student_sampling_params,
        )
        profile_row["post_sync_greedy_sample_s"] = _elapsed(post_sync_sample_t0)
        post_sync_rollouts.append(sample_after_sync)
        if sample_after_sync == initial_greedy_rollout and not args.allow_unchanged_sync:
            log.error(
                "Greedy rollout for prompt %s did not change at all by step %d "
                "(initial=%s, after_sync=%s); the broadcast did not take effect.",
                initial_greedy_prompt,
                step,
                initial_greedy_rollout,
                sample_after_sync,
            )
            return 4
        if sample_after_sync == initial_greedy_rollout:
            log.warning(
                "Greedy rollout for prompt %s did not change by step %d; continuing because "
                "--allow-unchanged-sync is set.",
                initial_greedy_prompt,
                step,
            )
        else:
            log.info(
                "Verified weights changed on student SGLang: initial=%s post_sync=%s",
                initial_greedy_rollout,
                sample_after_sync,
            )
        completed_step = step + 1
        profile_row["checkpoint_due"] = False
        if args.checkpoint_interval_steps and completed_step % args.checkpoint_interval_steps == 0:
            profile_row["checkpoint_due"] = True
            checkpoint_t0 = time.perf_counter()
            checkpoint_prefix = args.checkpoint_name_prefix.strip() or "opd"
            if args.checkpoint_save_best and args.checkpoint_best_metric == "val_loss":
                if eval_prompt_schedule is None:
                    raise RuntimeError("checkpoint best metric val_loss requested but eval prompt schedule is absent")
                val_metrics = _evaluate_opd_validation_loss(
                    args=args,
                    train_url=args.train_url,
                    teacher_head=teacher_head,
                    eval_prompt_schedule=eval_prompt_schedule,
                    student_router=student_router,
                    teacher_router=teacher_router,
                    artifacts_dir=artifacts_dir,
                    completed_step=completed_step,
                )
                profile_row.update(val_metrics)
                log.info(
                    "Checkpoint eval step=%d val_loss=%s valid_tokens=%s eval_total=%.3fs",
                    completed_step,
                    profile_row.get("val_loss"),
                    profile_row.get("val_valid_tokens"),
                    profile_row.get("val_total_s", 0.0),
                )
            latest_name = f"{checkpoint_prefix}-step{completed_step:06d}"
            log.info("Saving periodic checkpoint for completed OPD step %d: %s", completed_step, latest_name)
            latest_entry = _save_training_checkpoint(
                args.train_url,
                model_id=args.checkpoint_model_id,
                checkpoint_name=latest_name,
                timeout=args.checkpoint_timeout,
            )
            latest_entry.update(
                {
                    "step": completed_step,
                    "metric_name": args.checkpoint_best_metric,
                    "metric_value": _checkpoint_metric_value(profile_row, args.checkpoint_best_metric),
                    "kind": "latest",
                }
            )
            latest_checkpoints.append(latest_entry)
            profile_row["checkpoint_latest_name"] = latest_entry["name"]
            profile_row["checkpoint_latest_path"] = latest_entry["path"]
            profile_row["checkpoint_latest_save_s"] = latest_entry["save_s"]

            deleted_latest: List[str] = []
            if args.checkpoint_keep_latest == 0:
                delete_candidates = list(latest_checkpoints)
                latest_checkpoints = []
            else:
                delete_candidates = latest_checkpoints[: -args.checkpoint_keep_latest]
                latest_checkpoints = latest_checkpoints[-args.checkpoint_keep_latest :]
            for old in delete_candidates:
                _delete_training_checkpoint(
                    args.train_url,
                    model_id=args.checkpoint_model_id,
                    checkpoint_id=str(old["checkpoint_id"]),
                )
                deleted_latest.append(str(old["checkpoint_id"]))
            if deleted_latest:
                profile_row["checkpoint_deleted_latest"] = ",".join(deleted_latest)

            metric_value = _checkpoint_metric_value(profile_row, args.checkpoint_best_metric)
            profile_row["checkpoint_best_metric_value"] = metric_value
            if (
                args.checkpoint_save_best
                and completed_step >= args.checkpoint_best_min_step
                and metric_value is not None
                and _is_better_checkpoint_metric(metric_value, best_checkpoint_metric, mode=args.checkpoint_best_mode)
            ):
                best_name = f"{checkpoint_prefix}-best-step{completed_step:06d}"
                log.info(
                    "Saving new best checkpoint at step %d: metric %s=%.6g name=%s",
                    completed_step,
                    args.checkpoint_best_metric,
                    metric_value,
                    best_name,
                )
                previous_best = best_checkpoint
                best_entry = _save_training_checkpoint(
                    args.train_url,
                    model_id=args.checkpoint_model_id,
                    checkpoint_name=best_name,
                    timeout=args.checkpoint_timeout,
                )
                best_entry.update(
                    {
                        "step": completed_step,
                        "metric_name": args.checkpoint_best_metric,
                        "metric_value": metric_value,
                        "kind": "best",
                    }
                )
                best_checkpoint = best_entry
                best_checkpoint_metric = metric_value
                profile_row["checkpoint_best_name"] = best_entry["name"]
                profile_row["checkpoint_best_path"] = best_entry["path"]
                profile_row["checkpoint_best_save_s"] = best_entry["save_s"]
                if previous_best is not None:
                    _delete_training_checkpoint(
                        args.train_url,
                        model_id=args.checkpoint_model_id,
                        checkpoint_id=str(previous_best["checkpoint_id"]),
                    )
                    profile_row["checkpoint_deleted_previous_best"] = previous_best["checkpoint_id"]
            elif args.checkpoint_save_best and metric_value is None:
                log.warning(
                    "Skipping best-checkpoint selection because metric %s is absent from profile row",
                    args.checkpoint_best_metric,
                )

            _write_checkpoint_summary(
                checkpoint_summary_output,
                policy=checkpoint_policy,
                latest_checkpoints=latest_checkpoints,
                best_checkpoint=best_checkpoint,
            )
            profile_row["checkpoint_summary_output"] = str(checkpoint_summary_output)
            profile_row["checkpoint_total_s"] = _elapsed(checkpoint_t0)
        profile_row["step_total_s"] = _elapsed(step_t0)
        _add_route_observation_profile(
            profile_row,
            args,
            student_smg_before=student_smg_before,
            teacher_smg_before=teacher_smg_before,
        )
        _add_singleshot_profile_aliases(
            profile_row,
            args,
            device_count=profile_device_count,
            cumulative_tokens=cumulative_tokens,
            cumulative_sup_tokens=cumulative_sup_tokens,
            cumulative_consumed_tokens=cumulative_consumed_tokens,
        )
        _record_profile_row(
            profile_rows,
            profile_output,
            profile_row,
            wandb_run=wandb_run,
            wandb_log_interval=args.wandb_log_interval,
        )
        log.info("OPD throughput step=%d %s", step, json.dumps(profile_row, sort_keys=True))

    if (
        post_sync_rollouts
        and all(r == initial_greedy_rollout for r in post_sync_rollouts)
        and not args.allow_unchanged_sync
    ):
        log.error(
            "Greedy rollout for prompt %s never diverged from initial across %d steps.\ninitial: %s\npost_sync: %s",
            initial_greedy_prompt,
            len(post_sync_rollouts),
            initial_greedy_rollout,
            post_sync_rollouts,
        )
        return 5

    log.info("OPD pipeline validation succeeded.")
    log.info("loss_history=%s", loss_history)
    summary_keys = [
        "student_sampling_s",
        "teacher_prefill_s",
        "opd_pipeline_prepare_wait_s",
        "forward_backward_roundtrip_s",
        "trainer_forward_backward_s",
        "trainer_forward_loss_s",
        "trainer_backward_s",
        "optim_step_roundtrip_s",
        "sync_inference_weights_s",
        "sync_transfer_time_s",
        "step_total_s",
    ]
    all_summary = {key: _mean(profile_rows, key) for key in summary_keys}
    steady_rows = [row for row in profile_rows if not row.get("profile_warmup")]
    steady_summary = {key: _mean(steady_rows, key) for key in summary_keys}
    log.info(
        "Mean OPD throughput timings across %d steps: %s", len(profile_rows), json.dumps(all_summary, sort_keys=True)
    )
    log.info(
        "Steady-state OPD timings across %d steps (excluded %d warmup): %s",
        len(steady_rows),
        len(profile_rows) - len(steady_rows),
        json.dumps(steady_summary, sort_keys=True),
    )
    log.info("Per-step throughput JSONL: %s", profile_output)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
