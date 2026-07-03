"""Single-replica scorer throughput harness (turbo-tune before fleet rolls).

Drives ONE sglang scorer directly (no SMG, no trainer) with the exact ZORL
serving structure: a loaded parent LoRA + K virtual rank-r candidates
registered from synthetic seed specs, then N concurrent decode-heavy
/generate requests round-robining across candidates. Reports end-to-end
completions/s and tok/s, and scrapes the pod's decode-batch log lines for
running-req / mamba / KV / cuda-graph state during the window.

Usage (from the hub):
  python bench_scorer_throughput.py \
    --url http://zorl-w35-sglang-32.zorl-w35-sglang-headless.apanda.svc.cluster.local:30000 \
    --parent-path /shared/.../parent_g1 \
    --candidates 128 --concurrency 128 --requests 512 \
    --rank 1 --b-sigma 6e-4 --max-new-tokens 512

Sweep no-restart knobs (candidates, concurrency) in one server life; restart
the pod between restart-knob configs (max_running, mamba slots, graph bs).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xorl_ps_client as ps  # noqa: E402
import zorl_client as zc  # noqa: E402

PROMPT = (
    "You are playing a word puzzle. Think step by step about letter frequencies, "
    "positional constraints, and prior feedback before answering. "
    + "Consider the state of the board carefully. " * 40
    + "Now reason at length about the best next move and explain your strategy."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--parent-path", required=True)
    ap.add_argument("--parent-name", default="bench-parent")
    ap.add_argument("--candidates", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--requests", type=int, default=512)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--b-sigma", type=float, default=6e-4)
    ap.add_argument("--perturbation-mode", default="fresh_ab")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--generation-id", default=None)
    ap.add_argument("--keep", action="store_true", help="skip teardown (reuse candidates)")
    args = ap.parse_args()

    gen_id = args.generation_id or f"bench-{int(time.time())}"
    session_id = "bench-session"

    # Parent load (idempotent) + candidate registration from synthetic seeds.
    try:
        zc.load_lora_adapter(args.url, args.parent_name, args.parent_path)
    except Exception as e:  # noqa: BLE001
        if "already loaded" not in str(e):
            raise
    rng = random.Random(1234)
    specs = [
        {
            "candidate_id": f"{gen_id}-c{i}",
            "lora_name": f"{gen_id}-c{i}",
            "perturbation_index": i // 2,
            "direction": "pos" if i % 2 == 0 else "neg",
            "b_seed": rng.getrandbits(62),
            "a_seed": rng.getrandbits(62),
            "b_sigma": args.b_sigma,
            "perturbation_mode": args.perturbation_mode,
            "rank": args.rank,
        }
        for i in range(args.candidates)
    ]
    t0 = time.time()
    ps.register_zorl_candidates(
        args.url,
        session_id=session_id,
        generation_id=gen_id,
        parent_lora_name=args.parent_name,
        candidates=specs,
        timeout=600.0,
    )
    t_reg = time.time() - t0
    print(f"registered {args.candidates} rank-{args.rank} candidates in {t_reg:.1f}s")

    # Concurrent decode-heavy load, round-robin across candidates.
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.concurrency + 8,
                                            pool_maxsize=args.concurrency + 8, max_retries=0)
    sess.mount("http://", adapter)
    done = 0
    tokens = 0
    errors = 0
    lock = threading.Lock()

    def one(i: int) -> None:
        nonlocal done, tokens, errors
        body = {
            "text": PROMPT,
            "lora_path": specs[i % args.candidates]["lora_name"],
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.7,
                "ignore_eos": True,
            },
        }
        try:
            r = sess.post(f"{args.url}/generate", json=body, timeout=1800)
            r.raise_for_status()
            meta = r.json().get("meta_info", {})
            with lock:
                done += 1
                tokens += int(meta.get("completion_tokens", args.max_new_tokens))
        except Exception:  # noqa: BLE001
            with lock:
                errors += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, range(args.requests)))
    dt = time.time() - t0

    print(json.dumps({
        "candidates": args.candidates, "concurrency": args.concurrency,
        "requests": args.requests, "errors": errors,
        "wall_s": round(dt, 1),
        "completions_per_s": round(done / dt, 2),
        "gen_tok_per_s": round(tokens / dt, 1),
        "register_s": round(t_reg, 1),
    }))

    if not args.keep:
        try:
            ps.abort_replica_generation(args.url, session_id=session_id, generation_id=gen_id)
        except Exception as e:  # noqa: BLE001
            print(f"teardown warning: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
