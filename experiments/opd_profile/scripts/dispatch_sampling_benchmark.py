#!/usr/bin/env python3
"""Benchmark Dispatch/SGLang sampling throughput for the OPD rollout shape.

The default payload mirrors xorl-client OPD chat-completions sampling: single
chat prompt, ``max_tokens=192``, temperature 1.0, ``n=1``, and
``logprobs=true``.

Typical in-cluster use:

    python experiments/opd_profile/scripts/dispatch_sampling_benchmark.py \
      --base-url http://opd-q36-397b-4t4s-20260515-dispatch:8080 \
      --requests 1024 --concurrency 64,128,256,512
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


@dataclass
class RequestResult:
    request_id: int
    backend: str
    status_code: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    error: str = ""


def _parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for chunk in value.replace(",", " ").split():
        if chunk.strip():
            items.append(int(chunk))
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int((len(ordered) - 1) * pct), len(ordered) - 1)
    return ordered[idx]


def _opd_prompt(idx: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"Write a concise answer to synthetic OPD prompt {idx}.",
        }
    ]


def _longform_prompt(idx: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Write a detailed synthetic reasoning trace for benchmark item "
                f"{idx}. Continue until the answer is complete."
            ),
        }
    ]


def _numbered_prompt(idx: int) -> list[dict[str, str]]:
    a = 100 + (idx * 17) % 900
    b = 100 + (idx * 31) % 900
    return [
        {
            "role": "user",
            "content": (f"Solve {a} * {b}. Show the arithmetic briefly, then give the final integer."),
        }
    ]


def _messages(idx: int, mode: str) -> list[dict[str, str]]:
    if mode == "opd":
        return _opd_prompt(idx)
    if mode == "longform":
        return _longform_prompt(idx)
    if mode == "numbered":
        return _numbered_prompt(idx)
    raise ValueError(f"unknown prompt mode: {mode}")


def _payload(args: argparse.Namespace, idx: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": _messages(idx, args.prompt_mode),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n": 1,
        "logprobs": args.logprobs,
    }
    if args.seed is not None:
        payload["seed"] = args.seed + idx
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.extra_json:
        payload.update(json.loads(args.extra_json))
    return payload


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


def _admin_status_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/admin/status"


async def _fetch_dispatch_status(client: httpx.AsyncClient, base_url: str) -> dict[str, Any] | None:
    try:
        response = await client.get(_admin_status_url(base_url), timeout=10)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def _backend_metrics(status: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    if not status:
        return {}
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for url, data in metrics.items():
        if not isinstance(data, dict):
            continue
        out[url] = {
            "requests": int(data.get("successful_requests") or 0) + int(data.get("failed_requests") or 0),
            "success": int(data.get("successful_requests") or 0),
            "errors": int(data.get("failed_requests") or 0),
            "prompt_tokens": int(data.get("prompt_tokens") or 0),
            "completion_tokens": int(data.get("completion_tokens") or 0),
        }
    return out


def _metric_deltas(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    urls = sorted(set(before) | set(after))
    deltas: dict[str, dict[str, int]] = {}
    for url in urls:
        lhs = before.get(url, {})
        rhs = after.get(url, {})
        deltas[url] = {
            key: int(rhs.get(key, 0)) - int(lhs.get(key, 0))
            for key in (
                "requests",
                "success",
                "errors",
                "prompt_tokens",
                "completion_tokens",
            )
        }
    return deltas


def _backend_urls_from_status(status: dict[str, Any] | None) -> list[str]:
    if not status:
        return []
    urls: list[str] = []
    for backend in status.get("backends") or []:
        if not isinstance(backend, dict):
            continue
        url = backend.get("url")
        healthy = backend.get("healthy", True)
        if isinstance(url, str) and url and healthy:
            urls.append(url)
    return urls


def _backend_label(url: str) -> str:
    if "sglang-" in url:
        return "sglang-" + url.split("sglang-", 1)[1].split(":", 1)[0]
    return url


async def _send_one(
    client: httpx.AsyncClient,
    endpoint: str,
    args: argparse.Namespace,
    idx: int,
    backend_label: str,
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post(
            endpoint,
            json=_payload(args, idx),
            headers={"x-request-id": f"opd-sampling-bench-{idx}"},
        )
        latency_s = time.perf_counter() - start
        try:
            data = response.json()
        except Exception:
            data = {}
        usage = data.get("usage") if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        finish_reason = ""
        try:
            finish_reason = str(data["choices"][0].get("finish_reason") or "")
        except Exception:
            pass
        error = ""
        if response.status_code >= 400:
            error = str(data)[:500]
        return RequestResult(
            request_id=idx,
            backend=backend_label,
            status_code=response.status_code,
            latency_s=latency_s,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            finish_reason=finish_reason,
            error=error,
        )
    except Exception as exc:
        return RequestResult(
            request_id=idx,
            backend=backend_label,
            status_code=0,
            latency_s=time.perf_counter() - start,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason="",
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_once(
    args: argparse.Namespace,
    concurrency: int,
    backend_urls: list[str],
) -> dict[str, Any]:
    limits = _http_limits(args, concurrency)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        before = await _fetch_dispatch_status(client, args.base_url)
        before_metrics = _backend_metrics(before)

        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(args.requests):
            queue.put_nowait(idx)

        results: list[RequestResult] = []
        completed = 0
        started = time.perf_counter()

        async def worker(worker_id: int) -> None:
            nonlocal completed
            while True:
                try:
                    idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if args.mode == "direct":
                    backend_url = backend_urls[idx % len(backend_urls)]
                    url = backend_url.rstrip("/") + "/chat/completions"
                    label = _backend_label(backend_url)
                else:
                    url = _endpoint(args.base_url)
                    label = "dispatch"
                result = await _send_one(client, url, args, idx, label)
                results.append(result)
                completed += 1
                if args.progress_interval and completed % args.progress_interval == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "concurrency": concurrency,
                                "completed": completed,
                                "elapsed_s": elapsed,
                                "requests_per_s": completed / elapsed if elapsed > 0 else 0.0,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                queue.task_done()

        workers = [asyncio.create_task(worker(worker_id)) for worker_id in range(min(concurrency, args.requests))]
        await asyncio.gather(*workers)
        elapsed_s = time.perf_counter() - started
        after = await _fetch_dispatch_status(client, args.base_url)
        after_metrics = _backend_metrics(after)

    successes = [item for item in results if item.status_code == 200]
    errors = [item for item in results if item.status_code != 200]
    latencies = [item.latency_s for item in successes]
    prompt_tokens = sum(item.prompt_tokens for item in successes)
    completion_tokens = sum(item.completion_tokens for item in successes)
    total_tokens = sum(item.total_tokens for item in successes)
    finish_reasons = Counter(item.finish_reason for item in successes)
    backend_counts = Counter(item.backend for item in successes)
    deltas = _metric_deltas(before_metrics, after_metrics)

    summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_mode": args.prompt_mode,
        "logprobs": args.logprobs,
        "no_keepalive": args.no_keepalive,
        "keepalive_expiry": args.keepalive_expiry,
        "elapsed_s": elapsed_s,
        "success": len(successes),
        "errors": len(errors),
        "requests_per_s": len(successes) / elapsed_s if elapsed_s > 0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_s": completion_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "total_tokens_per_s": total_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "mean_completion_tokens": completion_tokens / len(successes) if successes else 0.0,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p90_s": _percentile(latencies, 0.90),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "latency_max_s": max(latencies) if latencies else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "client_backend_counts": dict(sorted(backend_counts.items())),
        "dispatch_backend_deltas": deltas,
        "first_errors": [
            {
                "request_id": item.request_id,
                "backend": item.backend,
                "status_code": item.status_code,
                "latency_s": item.latency_s,
                "error": item.error,
            }
            for item in errors[:10]
        ],
    }
    delta_requests = [data["requests"] for data in deltas.values() if data.get("requests", 0) > 0]
    if delta_requests:
        summary["dispatch_backend_request_min"] = min(delta_requests)
        summary["dispatch_backend_request_max"] = max(delta_requests)
    return summary


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "requests",
        "concurrency",
        "max_tokens",
        "no_keepalive",
        "success",
        "errors",
        "elapsed_s",
        "requests_per_s",
        "completion_tokens",
        "completion_tokens_per_s",
        "mean_completion_tokens",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "latency_max_s",
        "dispatch_backend_request_min",
        "dispatch_backend_request_max",
    ]
    exists = out.exists()
    with out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _http_limits(args: argparse.Namespace, concurrency: int) -> httpx.Limits:
    if args.no_keepalive:
        return httpx.Limits(
            max_connections=max(concurrency * 2, 128),
            max_keepalive_connections=0,
        )
    return httpx.Limits(
        max_connections=max(concurrency * 2, 128),
        max_keepalive_connections=max(concurrency, 64),
        keepalive_expiry=args.keepalive_expiry,
    )


async def _main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10) as client:
        status = await _fetch_dispatch_status(client, args.base_url)
    direct_backend_urls = args.direct_backend or _backend_urls_from_status(status)
    if args.mode == "direct" and not direct_backend_urls:
        raise RuntimeError("direct mode needs --direct-backend or a working Dispatch /admin/status")

    rows: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        if args.cooldown_s > 0 and rows:
            await asyncio.sleep(args.cooldown_s)
        print(
            json.dumps(
                {
                    "event": "start",
                    "mode": args.mode,
                    "requests": args.requests,
                    "concurrency": concurrency,
                    "max_tokens": args.max_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        row = await _run_once(args, concurrency, direct_backend_urls)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        _write_jsonl(args.output_jsonl, [row])
        _write_csv(args.output_csv, [row])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("dispatch", "direct"), default="dispatch")
    parser.add_argument("--direct-backend", action="append", default=[])
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument(
        "--concurrency",
        type=_parse_int_list,
        default=[32, 64, 128, 256],
        help="Comma or space separated concurrency values.",
    )
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--extra-json", default="")
    parser.add_argument("--prompt-mode", choices=("opd", "longform", "numbered"), default="opd")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--keepalive-expiry", type=float, default=5.0)
    parser.add_argument("--no-keepalive", action="store_true")
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--progress-interval", type=int, default=256)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if any(value <= 0 for value in args.concurrency):
        raise SystemExit("--concurrency values must be positive")

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
