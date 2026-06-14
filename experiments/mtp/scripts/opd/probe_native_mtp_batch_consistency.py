#!/usr/bin/env python3
"""Probe batched-decode consistency of native-MTP sampling (corruption check).

Context: docs/notes (xorl-apanda-dev-opd-port) sglang_batched_decode_corruption_handoff.md
found flashinfer-backend generation corruption that grows with server-side batch size /
concurrent load. Native MTP requires the flashinfer backend, so the fa3 mitigation does
not apply; this probe instead measures the failure directly on the MTP path:

Same prompts, argmax (top_k=1), deterministic-inference server: a serial bs=1 pass and a
batched single-request pass must produce IDENTICAL token ids. Divergence rate, divergence
position, and degeneration stats (max repeated run, unique-token fraction) quantify any
batch-dependent corruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_sampler_native_mtp import _load_prefix_prompts  # noqa: E402
from run_opd_pipeline import (  # noqa: E402
    _student_sample_native_mtp_batch,
    _wait_for_sglang,
)


def _max_repeated_run(tokens: list[int]) -> int:
    best = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        previous = token
        best = max(best, current)
    return best


def _generated(sequences: list[list[int]], prompts: list[list[int]]) -> list[list[int]]:
    return [seq[len(prompt) :] for seq, prompt in zip(sequences, prompts)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-offset", type=int, default=5000)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--k-toks", type=int, default=4)
    parser.add_argument("--mask-token-id", type=int, default=248063)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    _wait_for_sglang(args.server_url, timeout=120.0)
    singleshot_mtp = {
        "k_toks": args.k_toks,
        "mask_token_id": args.mask_token_id,
        "sampling_mode": "native",
        "mtp_strategy": ["conf_adapt", args.conf_threshold],
        "mtp_conf_threshold": args.conf_threshold,
        "native_mtp_debug_trace": True,
    }
    prompts = _load_prefix_prompts(
        args.dataset, offset=args.dataset_offset, count=args.num_prompts, prompt_len=args.prompt_len
    )

    def sample(batch: list[list[int]]) -> list[list[int]]:
        sequences, _ = _student_sample_native_mtp_batch(
            args.server_url,
            batch,
            args.max_new_tokens,
            singleshot_mtp=singleshot_mtp,
            temperature=0.7,  # top_k=1 (pipeline default) -> argmax regardless
            timeout=args.timeout,
        )
        return sequences

    serial = []
    for prompt in prompts:
        serial.extend(sample([prompt]))
    batched = sample(prompts)

    serial_gen = _generated(serial, prompts)
    batched_gen = _generated(batched, prompts)
    rows = []
    for idx, (s, b) in enumerate(zip(serial_gen, batched_gen)):
        first_div = next((pos for pos, (x, y) in enumerate(zip(s, b)) if x != y), None)
        if first_div is None and len(s) != len(b):
            first_div = min(len(s), len(b))
        rows.append(
            {
                "prompt_idx": idx,
                "match": first_div is None,
                "first_divergence_pos": first_div,
                "serial_len": len(s),
                "batched_len": len(b),
                "serial_max_repeat": _max_repeated_run(s),
                "batched_max_repeat": _max_repeated_run(b),
                "serial_unique_frac": len(set(s)) / max(1, len(s)),
                "batched_unique_frac": len(set(b)) / max(1, len(b)),
                "serial_token_ids": s,
                "batched_token_ids": b,
            }
        )
    matches = sum(1 for row in rows if row["match"])
    summary = {
        "server_url": args.server_url,
        "num_prompts": args.num_prompts,
        "exact_match": matches,
        "exact_match_frac": matches / len(rows),
        "min_first_divergence": min(
            (row["first_divergence_pos"] for row in rows if row["first_divergence_pos"] is not None), default=None
        ),
        "batched_degenerate_count": sum(
            1 for row in rows if row["batched_max_repeat"] >= 16 and row["serial_max_repeat"] < 16
        ),
        "rows": rows,
    }
    print(
        f"exact_match {matches}/{len(rows)}; "
        f"min_first_div={summary['min_first_divergence']}; "
        f"batched-only degenerates={summary['batched_degenerate_count']}"
    )
    for row in rows:
        if not row["match"]:
            print(
                f"  prompt {row['prompt_idx']}: div@{row['first_divergence_pos']} "
                f"len {row['serial_len']}->{row['batched_len']} "
                f"maxrep {row['serial_max_repeat']}->{row['batched_max_repeat']} "
                f"uniq {row['serial_unique_frac']:.2f}->{row['batched_unique_frac']:.2f}"
            )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
