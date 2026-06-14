#!/usr/bin/env python3
"""Precompute CoT solutions for the OPD prompt set.

Reads a list of chat-message prompts, sends each to a chat-completions endpoint
asking for a chain-of-thought solution, and writes the per-prompt result to a
JSON file. Used by the Run B OPD setup to supply per-prompt CoT as the teacher's
filler-token sequence.

Output schema (matches the input prompt order):
    [
        {
            "prompt": [{"role": "user", "content": "..."}],
            "cot": "Let me think step by step...",
            "tokens": 187,
            "finish_reason": "stop"
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

SYSTEM = (
    "You are a careful, methodical mathematician. For each problem, work through "
    "the calculation step by step, showing your reasoning clearly. End with the "
    "final answer on its own line in the form 'Answer: <number>'."
)


async def _ask(client: httpx.AsyncClient, endpoint: str, model: str,
               messages: list[dict], max_tokens: int,
               enable_thinking: bool | None = None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, *messages],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if enable_thinking is not None:
        # Pin the template mode explicitly — the server-side default has bitten
        # us before (PTC-118 chat-rendering root cause).
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    resp = await client.post(f"{endpoint}/chat/completions", json=payload, timeout=600.0)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "cot": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "tokens": data.get("usage", {}).get("completion_tokens"),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True, help="JSON list of message arrays")
    ap.add_argument("--output", required=True)
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint root (e.g. http://localhost:30060/v1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on prompts processed (smoke test).")
    ap.add_argument("--enable-thinking", choices=["true", "false"], default=None,
                    help="Pin chat_template_kwargs.enable_thinking (default: omit, server default)")
    args = ap.parse_args()
    enable_thinking = None if args.enable_thinking is None else args.enable_thinking == "true"

    prompts = json.loads(Path(args.prompts).read_text())
    if args.limit is not None:
        prompts = prompts[: args.limit]
    n = len(prompts)
    print(f"Loaded {n} prompts from {args.prompts}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-populate `results` from any existing partial output. We only treat an
    # entry as "done" if it has a non-empty cot (so transient errors retry).
    results: list[dict | None] = [None] * n
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if len(existing) == n:
                kept = 0
                for i, r in enumerate(existing):
                    if isinstance(r, dict) and r.get("cot"):
                        results[i] = r
                        kept += 1
                if kept == n:
                    print(f"Output already complete at {out_path} ({n} entries). Skipping.")
                    return
                print(f"Resuming from partial output: {kept}/{n} entries already done; will fill the rest.")
        except Exception as e:
            print(f"Could not load partial output ({e}); starting fresh.")

    sem = asyncio.Semaphore(args.concurrency)
    done_count = sum(1 for r in results if r is not None)
    last_log = time.time()
    # httpx default is 100 max connections. Bump way up so the concurrency
    # semaphore is the actual gate.
    limits = httpx.Limits(
        max_connections=args.concurrency + 64,
        max_keepalive_connections=args.concurrency + 64,
        keepalive_expiry=600.0,
    )

    # Atomic checkpoint write: serialize to tmp file in same dir, then rename.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    last_checkpoint_count = done_count
    last_checkpoint_t = time.time()

    def checkpoint() -> None:
        tmp_path.write_text(json.dumps(results, ensure_ascii=False))
        tmp_path.replace(out_path)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(idx: int, messages: list[dict]) -> None:
            nonlocal done_count, last_log, last_checkpoint_count, last_checkpoint_t
            # Skip if already done in a prior run
            if results[idx] is not None:
                return
            async with sem:
                try:
                    r = await _ask(client, args.endpoint, args.model, messages, args.max_tokens,
                                   enable_thinking=enable_thinking)
                except Exception as e:
                    r = {"cot": "", "finish_reason": f"error:{e}", "tokens": 0}
                results[idx] = {"prompt": messages, **r}
                done_count += 1
                now = time.time()
                if now - last_log > 5 or done_count == n:
                    avg_tokens = sum((r['tokens'] or 0) for r in results if r) / (done_count or 1)
                    print(f"[{done_count}/{n}] avg tokens={avg_tokens:.0f}", flush=True)
                    last_log = now
                # Periodic checkpoint: every 60s OR every 2000 newly-done entries.
                if (now - last_checkpoint_t > 60) or (done_count - last_checkpoint_count >= 2000):
                    try:
                        checkpoint()
                        print(f"checkpoint: {done_count}/{n} entries written to {out_path}", flush=True)
                        last_checkpoint_count = done_count
                        last_checkpoint_t = now
                    except Exception as ce:
                        print(f"checkpoint failed: {ce}", flush=True)

        tasks = [asyncio.create_task(worker(i, msgs)) for i, msgs in enumerate(prompts)]
        await asyncio.gather(*tasks)

    # Final write
    checkpoint()
    print(f"Wrote {n} CoT entries to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
