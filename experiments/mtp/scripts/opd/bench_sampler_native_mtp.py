#!/usr/bin/env python3
"""Benchmark SGLang native-MTP student sampling throughput without training.

Mirrors the OPD pipeline's sampling request exactly (same helper, same
singleshot_mtp payload, same defaults: temperature 0.7, top_k 1, mtp debug
trace on) so sampler flag A/Bs can run without touching the trainer. Prompts
come from the same parquet prompt dataset the pipeline uses (prefix strategy).

Example (32-prompt burst, pipeline-equivalent, one replica per chunk):
  python scripts/opd/bench_sampler_native_mtp.py \
    --server-urls http://er-opd-q36-mtp-ss-0605c-sglang-0:30060,http://er-opd-q36-mtp-ss-0605c-sglang-1:30060 \
    --dataset /shared/opd-datasets/.../train.parquet \
    --num-prompts 32 --chunk-size 32 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_opd_pipeline import (  # noqa: E402
    _student_sample_native_mtp_batch,
    _wait_for_sglang,
)


def _load_prefix_prompts(path: str, *, offset: int, count: int, prompt_len: int) -> list[list[int]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    prompts: list[list[int]] = []
    remaining_offset = offset
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=256, columns=["prompt_ids"]):
        if remaining_offset >= batch.num_rows:
            remaining_offset -= batch.num_rows
            continue
        for row in batch.column("prompt_ids").to_pylist()[remaining_offset:]:
            prompt = [int(token_id) for token_id in row[:prompt_len]]
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= count:
                break
        remaining_offset = 0
        if len(prompts) >= count:
            break
    if len(prompts) < count:
        raise RuntimeError(f"Only {len(prompts)} prompts available from offset {offset}; need {count}")
    return prompts


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _sample_chunk(
    url: str,
    prompts: list[list[int]],
    args: argparse.Namespace,
    singleshot_mtp: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    sequences, metrics = _student_sample_native_mtp_batch(
        url,
        prompts,
        args.max_new_tokens,
        singleshot_mtp=singleshot_mtp,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    latency = time.perf_counter() - start
    output_tokens = sum(max(0, len(seq) - len(prompt)) for seq, prompt in zip(sequences, prompts))
    return {
        "url": url,
        "prompt_count": len(prompts),
        "latency_s": latency,
        "output_tokens": output_tokens,
        "output_tok_per_s": output_tokens / latency if latency > 0 else 0.0,
        "trace_steps": metrics.get("student_sampling_mtp_debug_trace_steps"),
        "trace_cuda_graph_steps": metrics.get("student_sampling_mtp_debug_trace_cuda_graph_steps"),
        "trace_cuda_graph_all": metrics.get("student_sampling_mtp_debug_trace_cuda_graph_all"),
        "trace_commit_len_mean": metrics.get("student_sampling_mtp_debug_trace_commit_len_mean"),
        "trace_covered_all_targets": metrics.get("student_sampling_mtp_replay_trace_covered_all_targets"),
        "generated_tokens": metrics.get("student_sampling_mtp_generated_tokens"),
        "trace_committed_tokens": metrics.get("student_sampling_mtp_debug_trace_committed_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-urls", required=True, help="comma-separated replica or router URLs")
    parser.add_argument("--dataset", required=True, help="parquet prompt dataset (prompt_ids column)")
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32, help="prompts per request; chunks round-robin over URLs")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--k-toks", type=int, default=4)
    parser.add_argument("--mask-token-id", type=int, default=248063)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--fresh-prompts-per-round", action="store_true", help="advance dataset offset each round")
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()

    urls = [url.strip().rstrip("/") for url in args.server_urls.split(",") if url.strip()]
    for url in urls:
        _wait_for_sglang(url, timeout=120.0)

    singleshot_mtp = {
        "k_toks": args.k_toks,
        "mask_token_id": args.mask_token_id,
        "sampling_mode": "native",
        "mtp_strategy": ["conf_adapt", args.conf_threshold],
        "mtp_conf_threshold": args.conf_threshold,
        "native_mtp_debug_trace": True,
    }

    rows: list[dict[str, Any]] = []
    for round_idx in range(args.rounds):
        offset = args.dataset_offset + (round_idx * args.num_prompts if args.fresh_prompts_per_round else 0)
        prompts = _load_prefix_prompts(args.dataset, offset=offset, count=args.num_prompts, prompt_len=args.prompt_len)
        chunks = _chunked(prompts, args.chunk_size)
        assignments = [(urls[(round_idx + i) % len(urls)], chunk) for i, chunk in enumerate(chunks)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            chunk_rows = list(pool.map(lambda pair: _sample_chunk(pair[0], pair[1], args, singleshot_mtp), assignments))
        wall = time.perf_counter() - start
        output_tokens = sum(row["output_tokens"] for row in chunk_rows)
        row = {
            "round": round_idx,
            "wall_s": wall,
            "output_tokens": output_tokens,
            "agg_output_tok_per_s": output_tokens / wall if wall > 0 else 0.0,
            "trace_covered_all_targets": all(bool(c["trace_covered_all_targets"]) for c in chunk_rows),
            "trace_cuda_graph_steps": sum(int(c["trace_cuda_graph_steps"] or 0) for c in chunk_rows),
            "trace_steps": sum(int(c["trace_steps"] or 0) for c in chunk_rows),
            "chunks": chunk_rows,
        }
        rows.append(row)
        print(
            f"round {round_idx}: wall {wall:.1f}s, {output_tokens} tok, "
            f"{row['agg_output_tok_per_s']:.1f} tok/s agg, covered={row['trace_covered_all_targets']}, "
            f"cuda_graph_steps={row['trace_cuda_graph_steps']}/{row['trace_steps']}",
            flush=True,
        )
        for chunk in chunk_rows:
            print(
                f"  {chunk['url']}: {chunk['prompt_count']}p {chunk['latency_s']:.1f}s "
                f"{chunk['output_tok_per_s']:.1f} tok/s commit_len={chunk['trace_commit_len_mean']}",
                flush=True,
            )

    summary = {
        "server_urls": urls,
        "num_prompts": args.num_prompts,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "conf_threshold": args.conf_threshold,
        "rounds": rows,
        "steady_agg_tok_per_s": (
            sum(row["agg_output_tok_per_s"] for row in rows[1:]) / max(1, len(rows) - 1) if len(rows) > 1 else None
        ),
        "trace_covered_all_rounds": all(row["trace_covered_all_targets"] for row in rows),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rounds"}, indent=2, sort_keys=True))
    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
