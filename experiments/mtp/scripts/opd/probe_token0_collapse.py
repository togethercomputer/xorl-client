#!/usr/bin/env python3
"""Minimal repro probe for the MTP draft token-0 collapse (handoff
docs/notes/mtp_student_sampler_token0_collapse_handoff.md).

Sends fresh-assistant-turn ("suffix"-style) prompts and mid-stream
("prefix"-style) prompts through the native-MTP sampling path and reports the
token-0 fraction + first generated ids per prompt, plus the native trace's
first-step fields when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_opd_pipeline import _student_sample_native_mtp_batch, _wait_for_sglang  # noqa: E402


SUFFIX_USER_MSGS = [
    "What is 17 * 23? Think step by step.",
    "Name three prime numbers between 50 and 70.",
    "Explain in one sentence why the sky is blue.",
    "What is the capital of Australia?",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k-toks", type=int, default=4)
    parser.add_argument("--mask-token-id", type=int, default=248063)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--mode", choices=["suffix", "prefix", "both"], default="both")
    parser.add_argument("--batch", action="store_true", help="send all prompts in one batch (vs serial)")
    parser.add_argument("--no-mtp", action="store_true", help="control: plain decode without MTP strategy")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts: list[tuple[str, list[int]]] = []
    if args.mode in ("suffix", "both"):
        for msg in SUFFIX_USER_MSGS:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            prompts.append((f"suffix:{msg[:32]}", ids))
    if args.mode in ("prefix", "both"):
        for msg in SUFFIX_USER_MSGS[:2]:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            # prefix-style: include some already-generated assistant tokens so the
            # draft engages mid-stream rather than on a fresh turn boundary
            ids = ids + tokenizer.encode("Let me work through this carefully. First, I will", add_special_tokens=False)
            prompts.append((f"prefix:{msg[:32]}", ids))

    singleshot_mtp = {
        "k_toks": args.k_toks,
        "mask_token_id": args.mask_token_id,
        "sampling_mode": "native",
        "mtp_strategy": ["conf_adapt", args.conf_threshold],
        "mtp_conf_threshold": args.conf_threshold,
        "native_mtp_debug_trace": True,
    }

    _wait_for_sglang(args.server_url, timeout=60.0)

    def run(batch: list[list[int]]) -> tuple[list[list[int]], dict]:
        if args.no_mtp:
            from run_opd_pipeline import _generate_batch_outputs, _sequence_from_generate_output

            outputs = _generate_batch_outputs(
                args.server_url,
                batch,
                args.max_new_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            seqs = [_sequence_from_generate_output(p, o) for p, o in zip(batch, outputs)]
            return seqs, {}
        return _student_sample_native_mtp_batch(
            args.server_url,
            batch,
            args.max_new_tokens,
            singleshot_mtp=singleshot_mtp,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    rows = []
    if args.batch:
        seqs, metrics = run([ids for _, ids in prompts])
        gens = [seq[len(ids):] for seq, (_, ids) in zip(seqs, prompts)]
        for (label, _), gen in zip(prompts, gens):
            rows.append((label, gen, metrics))
    else:
        for label, ids in prompts:
            seqs, metrics = run([ids])
            rows.append((label, seqs[0][len(ids):], metrics))

    out = []
    for label, gen, metrics in rows:
        frac0 = sum(1 for t in gen if t == 0) / max(1, len(gen))
        text = tokenizer.decode(gen[:48])
        print(f"[{label}] len={len(gen)} frac_tok0={frac0:.2f} ids[:16]={gen[:16]}")
        print(f"    text: {text!r}")
        meta = metrics.get("_student_sample_metadata")
        if meta:
            m0 = meta[0] if isinstance(meta, list) else meta
            for key in (
                "native_mtp_bootstrap_token_ids",
                "native_mtp_first_step_phase",
                "native_mtp_first_step_input_row_token_ids",
                "native_mtp_first_step_committed_token_ids",
                "native_mtp_first_step_pending_token_ids",
            ):
                if isinstance(m0, dict) and key in m0:
                    print(f"    {key}: {m0[key]}")
        out.append({"label": label, "len": len(gen), "frac_tok0": frac0, "ids": gen, "text": text})

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
