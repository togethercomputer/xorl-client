#!/usr/bin/env python3
"""Evaluate Qwen3.6-35B-A3B on the 4-digit multiplication task.

Hits the dispatch in front of the OPD-trained sgLang pods (which hold the
latest in-memory weights post-step-72) and samples two arms:

  - baseline: standard chat completion, model generates answer directly.
  - pause:    assistant message prefilled with " pause" * 100 + continue.

Grades each response by parsing the first integer the model emits and
comparing against the ground-truth product.

Output: per-arm accuracy + a JSON dump of all responses for later analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

INT_RE = re.compile(r"-?\d[\d,]*")


def parse_int(text: str) -> int | None:
    """First integer-looking thing the model emits (ignores formatting commas)."""
    if not text:
        return None
    m = INT_RE.search(text)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def ground_truth(prompt_text: str) -> int | None:
    """Extract 'Calculate: A * B' and compute A*B."""
    m = re.search(r"Calculate:\s*(-?\d+)\s*\*\s*(-?\d+)", prompt_text)
    if not m:
        return None
    return int(m.group(1)) * int(m.group(2))


async def _ask(client: httpx.AsyncClient, endpoint: str, model: str,
               messages: list[dict], extra_body: dict | None,
               max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if extra_body:
        payload.update(extra_body)
    resp = await client.post(f"{endpoint}/chat/completions", json=payload, timeout=600.0)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "tokens": data.get("usage", {}).get("completion_tokens"),
    }


async def _run_arm(name: str, prompts: list[list[dict]], endpoint: str, model: str,
                   pause_count: int, concurrency: int, max_tokens: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    n = len(prompts)
    results: list[dict | None] = [None] * n
    done = 0
    last_log = time.time()
    limits = httpx.Limits(
        max_connections=concurrency + 64,
        max_keepalive_connections=concurrency + 64,
        keepalive_expiry=600.0,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(idx: int, prompt_msgs: list[dict]) -> None:
            nonlocal done, last_log
            async with sem:
                if name == "pause":
                    msgs = list(prompt_msgs) + [
                        {"role": "assistant", "content": " pause" * pause_count},
                    ]
                    extra = {"continue_final_message": True}
                else:
                    msgs = list(prompt_msgs)
                    extra = None
                try:
                    r = await _ask(client, endpoint, model, msgs, extra, max_tokens)
                except Exception as e:
                    r = {"text": "", "finish_reason": f"error:{e}", "tokens": 0}
                # Strip the prefilled pause out of the response text — under
                # continue_final_message the API echoes the assistant message
                # plus the continuation. We only want what the model generated.
                if name == "pause":
                    text_only = r["text"]
                    pause_block = " pause" * pause_count
                    if text_only.startswith(pause_block):
                        text_only = text_only[len(pause_block):]
                    r["text"] = text_only
                gt = ground_truth(prompt_msgs[-1]["content"])
                pred = parse_int(r["text"])
                results[idx] = {
                    "prompt": prompt_msgs,
                    "text": r["text"],
                    "tokens": r["tokens"],
                    "finish_reason": r["finish_reason"],
                    "ground_truth": gt,
                    "pred": pred,
                    "correct": (gt is not None and pred is not None and gt == pred),
                }
                done += 1
                now = time.time()
                if now - last_log > 5 or done == n:
                    correct = sum(1 for r in results if r and r.get("correct"))
                    print(f"[{name}] {done}/{n}  correct={correct} ({100.0*correct/done:.2f}%)", flush=True)
                    last_log = now

        tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arms", default="baseline,pause", help="comma-separated arms to run")
    ap.add_argument("--pause-count", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"Loaded {len(prompts)} prompts. Arms: {args.arms}. Pause count: {args.pause_count}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue
        t0 = time.time()
        results = await _run_arm(arm, prompts, args.endpoint, args.model,
                                 args.pause_count, args.concurrency, args.max_tokens)
        elapsed = time.time() - t0
        correct = sum(1 for r in results if r.get("correct"))
        n = len(results)
        avg_tokens = sum((r.get("tokens") or 0) for r in results) / max(n, 1)
        summary[arm] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / max(n, 1),
            "avg_completion_tokens": avg_tokens,
            "elapsed_s": elapsed,
        }
        out_file = out_dir / f"{arm}.jsonl"
        with out_file.open("w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{arm}] DONE  accuracy={summary[arm]['accuracy']*100:.2f}%  avg_tokens={avg_tokens:.0f}  elapsed={elapsed:.1f}s  -> {out_file}")

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
