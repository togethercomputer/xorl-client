#!/usr/bin/env python3
"""Cold no-filler-OPSD multiplication DIRECT-answer baseline (headroom check).

Single-shot DIRECT-answer eval on the BASE Qwen3-30B-A3B-Instruct-2507 (no LoRA,
no ZORL session), using THIS worktree's ``tasks.mult``. For each eval problem we
POST the STUDENT prompt (chat-template user turn + ``Answer: `` -- exactly the
no-filler-OPSD student input, NO CoT, NO filler) at temperature 0 / 64 new
tokens, parse the first integer, and check it against ``A * B``.

This measures Q3-30B's HEADROOM on the task: the gradient no-filler-OPSD arm
lifted Q3.6-35B from cold 0.594 -> ~0.88. If Q3-30B's cold direct accuracy is
~1.0 there is no room to climb (flag it); if it is low/mid, ES has something to
optimize toward.

Reported: direct accuracy (exact match of parsed integer == product),
``empty_frac`` (fraction emitting NO integer -- the EOS-collapse signal), and
``mean_completion_tokens`` (sanity that generations aren't degenerate).

Usage (run from inside a pool replica so the headless DNS resolves):

  kubectl exec zorl-ar-sglang-0 -n apanda -- \\
    /workspace/home/xorl-sglang-internal/.venv/bin/python \\
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/eval_mult_baseline.py 64
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer


# Make ``tasks/`` importable regardless of cwd (mirrors zorl_client.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks import mult  # noqa: E402
from tasks.base import Example  # noqa: E402


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 64
SEED = 9234
DEFAULT_N = 64
MAX_WORKERS = 16

REPLICAS = [f"http://zorl-ar-sglang-{i}.zorl-ar-sglang-headless.apanda.svc.cluster.local:30000" for i in range(16)]


def _generate(url: str, *, input_ids: list[int], temperature: float, max_new_tokens: int) -> dict:
    """POST one /generate to a base-model replica (no LoRA). Returns the raw
    choice dict (text + meta). Retries transient connection/5xx failures."""
    sampling = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "n": 1,
    }
    payload = {"input_ids": list(input_ids), "sampling_params": sampling, "return_logprob": False}
    last_err = None
    for attempt in range(6):
        try:
            with requests.post(
                f"{url}/generate", json=payload, headers={"Connection": "close"}, timeout=300.0
            ) as response:
                if response.status_code >= 400:
                    last_err = RuntimeError(f"/generate {url} HTTP {response.status_code}: {response.text[:300]}")
                    if response.status_code in {429, 503} and attempt + 1 < 6:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise last_err
                data = response.json()
                if isinstance(data, list):
                    data = data[0] if data else {}
                return data or {}
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            if attempt + 1 < 6:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"/generate on {url} failed after retries: {last_err}")


def _completion_token_count(tokenizer, data: dict, text: str) -> int:
    """Best-effort completion token count: prefer the server's
    ``completion_tokens`` meta, else tokenize the returned text."""
    meta = data.get("meta_info") or {}
    for key in ("completion_tokens", "output_tokens"):
        val = meta.get(key)
        if isinstance(val, int) and val >= 0:
            return val
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _eval_one(url: str, tokenizer, example: Example) -> dict[str, float]:
    """Direct-answer eval of one problem: POST the student prompt, parse + score."""
    data = _generate(
        url,
        input_ids=list(example.prompt_ids),
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    text = str((data or {}).get("text", "") or "")
    blob = mult.score_completion(example, text)
    return {
        "exact_match": float(blob["exact_match"]),
        "emitted_number": float(blob["format_rate"]),  # 1.0 iff any integer emitted
        "completion_tokens": float(_completion_token_count(tokenizer, data, text)),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    print(
        f"[init] cold no-filler-OPSD mult DIRECT baseline: model={MODEL} temp={TEMPERATURE} "
        f"max_new_tokens={MAX_NEW_TOKENS} N={n} seed={SEED} replicas={len(REPLICAS)}",
        flush=True,
    )
    print(f"[init] usable aligned+filtered (prompt, CoT) pairs: {mult.NUM_USABLE_PAIRS}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)

    # N distinct eval problems. build_examples splits train/eval disjointly; use
    # train_size=0 and eval_size=n so all N come from the eval slice (and match
    # the seed/split the candidate uses).
    _train, eval_examples = mult.build_examples(tokenizer, train_size=0, eval_size=n, seed=SEED)
    if len(eval_examples) != n:
        raise RuntimeError(f"expected {n} eval examples, got {len(eval_examples)}")

    started = time.time()
    results: list[dict[str, float]] = []

    def run_one(index_example):
        index, example = index_example
        url = REPLICAS[index % len(REPLICAS)]
        return _eval_one(url, tokenizer, example)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n)) as pool:
        futures = {pool.submit(run_one, (i, ex)): i for i, ex in enumerate(eval_examples)}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 16 == 0 or done == n:
                print(f"[progress] {done}/{n} problems evaluated ({time.time() - started:.1f}s)", flush=True)

    count = max(len(results), 1)
    metrics = {
        "n": len(results),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "usable_pairs": mult.NUM_USABLE_PAIRS,
        "direct_accuracy": sum(r["exact_match"] for r in results) / count,
        "empty_frac": 1.0 - (sum(r["emitted_number"] for r in results) / count),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in results) / count,
        "wall_seconds": time.time() - started,
    }
    print("[result] " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
