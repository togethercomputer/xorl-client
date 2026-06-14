#!/usr/bin/env python3
"""Microbenchmark for the xorl teacher-prefill service.

Reproduces the exact shape of work the OPD client sends to the teacher in
Run B: per-prompt CoT-substituted teacher sequences via /api/v1/forward
with loss_fn="teacher_hidden_cache". Lets us sweep batch size, concurrency,
and (paired with `generate_teacher_only.py`) teacher node count.

Inputs:
  --teacher-url:      e.g. http://er-opdb-bench1n-teacher-master:30002
  --prompts-json:     path to filtered prompts JSON (same format as OPD client)
  --cot-json:         path to filtered CoT JSON
  --tokenizer-path:   chat tokenizer path (for both student-prefill and CoT
                      tokenization; matches the model the teacher serves)
  --batch-size:       per-request sample count (microbatch shape)
  --concurrency:      number of in-flight requests
  --num-batches:      total number of batches to submit
  --student-fill:     pause token count to bake into the synthetic student
                      sequence (replaced by CoT in the teacher seq)
  --student-fill-text: text to repeat (default " pause")
  --answer-len:       synthetic answer length per sample (replaces real student
                      sampling — we are NOT testing the student here)
  --cache-dir:        where to write teacher_hidden_cache safetensors (must be
                      a path visible to the teacher pod; /shared/* works)
  --output-json:      optional per-batch latency log

Output:
  Per-batch wall time, then aggregates: total tokens, total wall time, tok/s,
  effective concurrency (sum(per-batch s) / wall), latency p50/p90/p99.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


def _load_chat_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _encode_chat_prompt(messages: list[dict], tokenizer) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    # Newer transformers returns a BatchEncoding (UserDict subclass, not dict)
    # with "input_ids". Older versions return a flat list. Probe by attribute.
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(t) for t in token_ids]


def _build_teacher_payload(
    sample_idx_subset: list[int],
    prompts: list,
    cot_tokens_by_prompt: list[list[int]],
    prefill_tokens: list[int],
    answer_tokens: list[int],
    tokenizer,
) -> tuple[list[dict], int]:
    """Build the same _teacher_hidden_cache_data structure the OPD client uses.

    Returns (payload_data, total_teacher_input_tokens).
    """
    payload = []
    total_tokens = 0
    K = len(prefill_tokens)
    for idx in sample_idx_subset:
        p_tokens = _encode_chat_prompt(prompts[idx], tokenizer)
        p = len(p_tokens)
        # Synthetic student sequence: prompt + pause + answer.
        student_seq = p_tokens + prefill_tokens + answer_tokens
        cot = cot_tokens_by_prompt[idx]
        # Teacher seq = student[:p] + cot + student[p+K:]
        teacher_seq = list(student_seq[:p]) + list(cot) + list(student_seq[p + K:])
        input_ids = teacher_seq[:-1]
        target_tokens = list(teacher_seq[1:])
        # Mask the C cot-predicting positions [p-1, p-1+C).
        C = len(cot)
        for j in range(max(0, p - 1), max(0, p - 1) + C):
            if 0 <= j < len(target_tokens):
                target_tokens[j] = -100
        payload.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {"target_tokens": target_tokens},
            }
        )
        total_tokens += len(input_ids)
    return payload, total_tokens


async def _submit_one(
    client: httpx.AsyncClient,
    teacher_url: str,
    model_id: str,
    payload: list[dict],
    cache_path: str,
    request_timeout: float,
) -> tuple[float, int]:
    """Submit one teacher_hidden_cache forward request. Returns (wall_s, status).

    Polls the future to completion via /api/v1/retrieve_future.
    """
    t0 = time.perf_counter()
    body = {
        "model_id": model_id,
        "forward_input": {
            "data": payload,
            "loss_fn": "teacher_hidden_cache",
            "loss_fn_params": {
                "teacher_hidden_cache_path": cache_path,
                "teacher_hidden_cache_dtype": "bfloat16",
            },
        },
    }
    r = await client.post(f"{teacher_url}/api/v1/forward", json=body, timeout=60)
    r.raise_for_status()
    request_id = r.json()["request_id"]

    # Poll for completion.
    poll_t0 = time.perf_counter()
    while True:
        if time.perf_counter() - poll_t0 > request_timeout:
            raise TimeoutError(f"future {request_id} did not complete within {request_timeout}s")
        r = await client.post(
            f"{teacher_url}/api/v1/retrieve_future",
            json={"request_id": request_id, "timeout_seconds": 30.0},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        ftype = data.get("type")
        if ftype == "try_again":
            await asyncio.sleep(0.5)
            continue
        if ftype == "request_failure":
            raise RuntimeError(f"teacher forward failed: {data}")
        break

    return (time.perf_counter() - t0, 200)


async def _run_bench(
    teacher_url: str,
    model_id: str,
    prompts: list,
    cot_tokens_by_prompt: list[list[int]],
    prefill_tokens: list[int],
    answer_tokens: list[int],
    tokenizer,
    cache_dir: Path,
    batch_size: int,
    concurrency: int,
    num_batches: int,
    request_timeout: float,
) -> dict:
    n = len(prompts)
    indices_pool = list(range(n))
    random.seed(20260527)

    # Pre-build all payloads + cache paths so the timing only covers HTTP work.
    batches = []
    for bidx in range(num_batches):
        idxs = random.sample(indices_pool, batch_size)
        payload, tok_count = _build_teacher_payload(
            idxs, prompts, cot_tokens_by_prompt, prefill_tokens, answer_tokens, tokenizer
        )
        cache_path = str(cache_dir / f"bench_step_b{bidx}.safetensors")
        batches.append((payload, cache_path, tok_count, idxs))
    total_tokens_target = sum(b[2] for b in batches)
    print(
        f"[setup] {num_batches} batches × {batch_size} samples each, "
        f"{total_tokens_target:,} total teacher input tokens, "
        f"concurrency={concurrency}",
        flush=True,
    )

    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
        keepalive_expiry=600.0,
    )
    sem = asyncio.Semaphore(concurrency)
    per_batch_s: list[float] = [0.0] * num_batches
    submit_t0: list[float] = [0.0] * num_batches

    wall_t0 = time.perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(bidx: int):
            payload, cache_path, tok_count, _ = batches[bidx]
            async with sem:
                submit_t0[bidx] = time.perf_counter() - wall_t0
                wall_s, _ = await _submit_one(
                    client, teacher_url, model_id, payload, cache_path, request_timeout
                )
                per_batch_s[bidx] = wall_s
                done_pct = (bidx + 1) / num_batches * 100
                print(
                    f"  [batch {bidx + 1}/{num_batches}] {wall_s:.2f}s ({tok_count:,} tok, "
                    f"{tok_count / wall_s:,.0f} tok/s)  progress={done_pct:.0f}%",
                    flush=True,
                )

        tasks = [asyncio.create_task(worker(b)) for b in range(num_batches)]
        await asyncio.gather(*tasks)

    wall_s = time.perf_counter() - wall_t0
    sum_per_batch_s = sum(per_batch_s)
    effective_concurrency = sum_per_batch_s / wall_s if wall_s > 0 else 0.0
    tok_per_s_aggregate = total_tokens_target / wall_s if wall_s > 0 else 0.0

    sorted_s = sorted(per_batch_s)
    p50 = sorted_s[len(sorted_s) // 2]
    p90 = sorted_s[int(0.9 * len(sorted_s))]
    p99 = sorted_s[int(0.99 * len(sorted_s))]

    summary = {
        "batch_size": batch_size,
        "concurrency": concurrency,
        "num_batches": num_batches,
        "total_tokens": total_tokens_target,
        "wall_s": wall_s,
        "sum_per_batch_s": sum_per_batch_s,
        "effective_concurrency": effective_concurrency,
        "aggregate_tok_per_s": tok_per_s_aggregate,
        "median_batch_s": statistics.median(per_batch_s),
        "mean_batch_s": statistics.mean(per_batch_s),
        "p50_batch_s": p50,
        "p90_batch_s": p90,
        "p99_batch_s": p99,
        "per_batch_s": per_batch_s,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-url", required=True)
    ap.add_argument("--model-id", default="default")
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--cot-json", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--cache-dir", required=True, help="Shared dir for teacher_hidden_cache writes")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=16)
    ap.add_argument("--student-fill-text", default=" pause")
    ap.add_argument("--student-fill", type=int, default=100)
    ap.add_argument("--answer-len", type=int, default=150)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--output-json", default="")
    ap.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Optional cap on # prompts loaded (smoke; 0=all)",
    )
    args = ap.parse_args()

    print(f"[setup] Loading prompts from {args.prompts_json}")
    prompts = json.loads(Path(args.prompts_json).read_text())
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    print(f"[setup] Loading CoT from {args.cot_json}")
    cot_data = json.loads(Path(args.cot_json).read_text())
    cot_data = cot_data[: len(prompts)]

    print(f"[setup] Loading tokenizer from {args.tokenizer_path}")
    tokenizer = _load_chat_tokenizer(args.tokenizer_path)

    print(f"[setup] Tokenizing {len(cot_data)} CoT entries ...")
    cot_tokens_by_prompt: list[list[int]] = []
    for idx, entry in enumerate(cot_data):
        cot_text = entry.get("cot", "")
        if not cot_text:
            raise ValueError(f"CoT entry {idx} empty — filter the dataset first")
        cot_tokens_by_prompt.append(
            [int(t) for t in tokenizer.encode(cot_text, add_special_tokens=False)]
        )
    cot_lens = [len(c) for c in cot_tokens_by_prompt]
    print(
        f"[setup] CoT lengths: min={min(cot_lens)} median={sorted(cot_lens)[len(cot_lens)//2]} max={max(cot_lens)}"
    )

    prefill_block = args.student_fill_text * args.student_fill
    prefill_tokens = [int(t) for t in tokenizer.encode(prefill_block, add_special_tokens=False)]
    if len(prefill_tokens) != args.student_fill:
        print(
            f"WARNING: student_fill_text {args.student_fill_text!r} tokenizes to "
            f"{len(prefill_tokens)} tokens, expected {args.student_fill}"
        )

    # Synthetic answer: a sequence of token IDs from the model's vocab. We use
    # a fixed pattern so every sample has the same answer length, isolating the
    # teacher cost from per-sample answer variance.
    answer_tokens = list(range(1000, 1000 + args.answer_len))

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] teacher_url={args.teacher_url}  batch_size={args.batch_size}  "
        f"concurrency={args.concurrency}  num_batches={args.num_batches}  "
        f"student_fill={args.student_fill}  answer_len={args.answer_len}"
    )

    summary = asyncio.run(
        _run_bench(
            args.teacher_url,
            args.model_id,
            prompts,
            cot_tokens_by_prompt,
            prefill_tokens,
            answer_tokens,
            tokenizer,
            cache_dir,
            args.batch_size,
            args.concurrency,
            args.num_batches,
            args.request_timeout,
        )
    )

    print("\n=== Summary ===")
    for k in [
        "batch_size",
        "concurrency",
        "num_batches",
        "total_tokens",
        "wall_s",
        "sum_per_batch_s",
        "effective_concurrency",
        "aggregate_tok_per_s",
        "median_batch_s",
        "mean_batch_s",
        "p50_batch_s",
        "p90_batch_s",
        "p99_batch_s",
    ]:
        v = summary[k]
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop the per_batch_s list to keep file small unless explicitly requested.
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
