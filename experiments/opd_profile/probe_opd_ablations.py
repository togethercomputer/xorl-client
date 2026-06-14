#!/usr/bin/env python3
"""Serve a finished OPD checkpoint and probe its inference behavior.

The trainer process is launched separately with ``--server.load_checkpoint_path``
pointing at the trained DCP, so by the time this driver runs the trainer holds
the TRAINED weights. This driver:

  1. registers a single SGLang endpoint with the trainer,
  2. triggers ONE p2p weight sync (trainer -> sglang), so sglang now serves the
     trained model (bf16 quantized to FP8 on the wire to match the FP8 receiver),
  3. runs an ablation suite to characterise whether the pause buffer is
     load-bearing, and writes a JSON + a human-readable summary.

Arms (all greedy / temperature=0 for a clean, comparable read):
  control_pause100  4dmult, prefill = " pause"*100 + "</think>Answer: "  (trained format)
  no_pause          4dmult, prefill = "</think>Answer: "                 (0 pause — the control)
  pause50/200/400   4dmult, fewer/more pause tokens                      (capacity curve)
  freegen           4dmult, prefill = " pause"*100 (no answer cue), long max_tokens
  offdist_3dmult    3-digit x 3-digit, trained format                    (generalisation)
  offdist_4dadd     4-digit + 4-digit, trained format                    (generalisation)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
from typing import Any

import requests
import xorl_client as tomi
from transformers import AutoTokenizer
from xorl_client.client.training_client import TrainingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s probe: %(message)s")
log = logging.getLogger("probe")
for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

PAUSE = " pause"
SUFFIX = "</think>Answer: "


def wait_health(base_url: str, timeout: float, kind: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5).ok:
                log.info("%s healthy: %s", kind, base_url)
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"{kind} not healthy within {timeout}s: {base_url}")


def ensure_session(base_url: str, model_id: str, base_model: str) -> None:
    r = requests.post(
        f"{base_url}/api/v1/create_session",
        json={"session_id": model_id, "base_model": base_model},
        timeout=60,
    )
    r.raise_for_status()
    log.info("registered session %s @ %s -> %s", model_id, base_url, r.text)


def make_prompt(a: int, b: int, op: str) -> list[dict[str, str]]:
    sym = "*" if op == "mul" else "+"
    return [{"role": "user", "content": f"/no_think Calculate: {a} {sym} {b}"}]


def truth(a: int, b: int, op: str) -> int:
    return a * b if op == "mul" else a + b


def gen_problems(n: int, lo: int, hi: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]


def _completion_text(seq: Any, tok: Any) -> tuple[str, int]:
    """Return (decoded_text, num_tokens) for one sampled sequence."""
    toks = list(getattr(seq, "tokens", None) or [])
    text = getattr(seq, "text", None)
    if not text and toks:
        text = tok.decode(toks, skip_special_tokens=True)
    return (text or ""), len(toks)


async def run_arm(
    name: str,
    client: tomi.SamplingClient,
    problems: list[tuple[int, int]],
    op: str,
    prefill: str,
    max_tokens: int,
    tok: Any,
    n_log: int = 6,
) -> dict[str, Any]:
    params = tomi.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        chat_continue_final_message=bool(prefill),
    )
    futures = []
    for a, b in problems:
        msgs = make_prompt(a, b, op)
        if prefill:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        futures.append(client.sample(prompt=msgs, sampling_params=params, num_samples=1))
    responses = await asyncio.gather(*futures, return_exceptions=True)

    correct = scored = empty = errors = 0
    lens: list[int] = []
    rows: list[dict[str, Any]] = []
    for (a, b), resp in zip(problems, responses):
        if isinstance(resp, Exception) or not getattr(resp, "sequences", None):
            errors += 1
            continue
        text, ntok = _completion_text(resp.sequences[0], tok)
        lens.append(ntok)
        if ntok == 0:
            empty += 1
        digits = re.sub(r"[,\s]", "", text)
        ok = str(truth(a, b, op)) in digits
        scored += 1
        correct += int(ok)
        if len(rows) < n_log:
            rows.append({"a": a, "b": b, "out": text.strip()[:90], "correct": ok})

    acc = correct / scored if scored else 0.0
    res = {
        "arm": name,
        "op": op,
        "prefill_repr": (prefill[:40] + ("..." if len(prefill) > 40 else "")),
        "max_tokens": max_tokens,
        "n": len(problems),
        "scored": scored,
        "errors": errors,
        "accuracy": round(acc, 4),
        "empty_frac": round(empty / max(1, len(problems)), 4),
        "mean_len": round(sum(lens) / len(lens), 2) if lens else 0.0,
        "samples": rows,
    }
    log.info(
        "[%s] acc=%.3f empty=%.3f mean_len=%.1f scored=%d errors=%d",
        name, acc, res["empty_frac"], res["mean_len"], scored, errors,
    )
    for r in rows:
        log.info("   %s %d,%d -> %r", "OK" if r["correct"] else "XX", r["a"], r["b"], r["out"])
    return res


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-url", required=True)
    ap.add_argument("--sglang-host", required=True)
    ap.add_argument("--sglang-port", type=int, default=30060)
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--model-name", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sync-method", default="p2p")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n4", type=int, default=256)
    ap.add_argument("--n3", type=int, default=128)
    ap.add_argument("--nadd", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    wait_health(args.trainer_url, args.timeout, "trainer")
    wait_health(args.sglang_url, args.timeout, "sglang")
    ensure_session(args.trainer_url, "student", args.model_name)

    svc = tomi.ServiceClient(base_url=args.trainer_url, timeout=args.timeout)
    tc = TrainingClient(holder=svc.holder, model_id="student", base_model=args.model_name)

    log.info("registering sglang endpoint %s:%d (world=%d)...",
             args.sglang_host, args.sglang_port, args.world_size)
    tc.add_inference_endpoint(
        host=args.sglang_host, port=args.sglang_port,
        world_size=args.world_size, sync_weights=False,
    ).result()

    log.info("syncing trained weights -> sglang via %s ...", args.sync_method)
    t0 = time.time()
    sync_res = tc.sync_weights_to_inference(sync_method=args.sync_method).result()
    log.info("weight sync done in %.1fs: %s", time.time() - t0, sync_res)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = tomi.SamplingClient(
        base_url=args.sglang_url, model=args.model_name,
        timeout=args.timeout, api_format="chat_completions",
    )

    # Held-out problem sets (fresh random draws; disjoint seeds).
    p4 = gen_problems(args.n4, 1000, 9999, seed=20260529)
    p3 = gen_problems(args.n3, 100, 999, seed=11)
    padd = gen_problems(args.nadd, 1000, 9999, seed=22)

    arms = [
        ("control_pause100", p4, "mul", PAUSE * 100 + SUFFIX, 16),
        ("no_pause",         p4, "mul", SUFFIX, 16),
        ("pause50",          p4, "mul", PAUSE * 50 + SUFFIX, 16),
        ("pause200",         p4, "mul", PAUSE * 200 + SUFFIX, 16),
        ("pause400",         p4, "mul", PAUSE * 400 + SUFFIX, 16),
        ("freegen_pause100", p4[:64], "mul", PAUSE * 100, 320),
        ("offdist_3dmult",   p3, "mul", PAUSE * 100 + SUFFIX, 16),
        ("offdist_4dadd",    padd, "add", PAUSE * 100 + SUFFIX, 16),
    ]

    results = []
    for name, probs, op, prefill, mx in arms:
        log.info("=== arm: %s (n=%d, op=%s, max_tokens=%d) ===", name, len(probs), op, mx)
        results.append(await run_arm(name, client, probs, op, prefill, mx, tok))
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    log.info("================ ABLATION SUMMARY ================")
    log.info("%-20s %8s %8s %9s  %s", "arm", "acc", "empty", "mean_len", "prefill")
    for r in results:
        log.info("%-20s %8.3f %8.3f %9.1f  %s",
                 r["arm"], r["accuracy"], r["empty_frac"], r["mean_len"], r["prefill_repr"])
    log.info("=================================================")
    log.info("results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
