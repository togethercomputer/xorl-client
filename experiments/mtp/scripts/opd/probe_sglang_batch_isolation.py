#!/usr/bin/env python3
"""Compare singleton and batched SGLang generation for the same prompts.

This diagnostic intentionally does no training. It checks whether a batched
`/generate` call preserves per-prompt state/order compared with issuing one
request per prompt. The first target is AR and native-MTP k=1 because k=1 should
be the cleanest parity check after the native pending-token replay fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_opd_pipeline import (  # noqa: E402
    _student_sample_batch,
    _student_sample_native_mtp_batch,
    _wait_for_sglang,
)


def _decode(tokenizer: Any, token_ids: list[int], *, max_chars: int) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[:max_chars]


def _full_prompt_tokens(record: dict[str, Any]) -> list[int] | None:
    prompt_len = int(record.get("prompt_token_count") or 0)
    prompt_tail = [int(x) for x in record.get("prompt_tail_token_ids") or []]
    if prompt_len > 0 and len(prompt_tail) == prompt_len:
        return prompt_tail

    sequence_len = int(record.get("sequence_token_count") or 0)
    sequence_tail = [int(x) for x in record.get("sequence_tail_token_ids") or []]
    if prompt_len > 0 and sequence_len > 0 and len(sequence_tail) == sequence_len:
        return sequence_tail[:prompt_len]
    return None


def _load_prompts_from_rollouts(path: Path, *, limit: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = _full_prompt_tokens(record)
            if not prompt:
                continue
            key = tuple(prompt)
            if key in seen:
                continue
            seen.add(key)
            prompts.append(
                {
                    "source_step": record.get("step"),
                    "source_sample_idx": record.get("sample_idx"),
                    "prompt_token_ids": prompt,
                }
            )
            if len(prompts) >= limit:
                break
    if not prompts:
        raise RuntimeError(f"No full prompts found in rollout artifact: {path}")
    return prompts


def _parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = []
    for mode in modes:
        if mode == "ar":
            continue
        if not mode.startswith("native_k"):
            invalid.append(mode)
            continue
        try:
            k_toks = int(mode.removeprefix("native_k"))
        except ValueError:
            invalid.append(mode)
            continue
        if k_toks < 1:
            invalid.append(mode)
    if invalid:
        raise ValueError("Unsupported modes " f"{sorted(set(invalid))}; expected ar or native_k<positive-int>")
    return modes


def _native_trace_from_metrics(metrics: dict[str, Any], prompt_idx: int) -> list[dict[str, Any]]:
    sample_metadata = metrics.get("_student_sample_metadata")
    if not isinstance(sample_metadata, list) or prompt_idx >= len(sample_metadata):
        return []
    sample_meta = sample_metadata[prompt_idx]
    if not isinstance(sample_meta, dict):
        return []
    trace = sample_meta.get("mtp_debug_trace")
    if not isinstance(trace, list):
        return []
    return [step for step in trace if isinstance(step, dict)]


def _pending_replay_check(trace: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    for prev, current in zip(trace, trace[1:]):
        saved = prev.get("pending_saved_for_next_step")
        row = current.get("input_row_token_ids")
        if not isinstance(saved, list) or not isinstance(row, list) or not saved:
            continue
        if [int(x) for x in row[: len(saved)]] != [int(x) for x in saved]:
            mismatches.append(
                {
                    "previous_step_idx": prev.get("step_idx"),
                    "current_step_idx": current.get("step_idx"),
                    "saved_pending": saved,
                    "next_input_prefix": row[: len(saved)],
                }
            )
    return {
        "checked": bool(trace),
        "ok": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
    }


def _run_mode(
    *,
    mode: str,
    server_url: str,
    prompts: list[list[int]],
    max_new_tokens: int,
    mask_token_id: int,
    temperature: float,
    sampling_extra: dict[str, Any],
    timeout: float,
) -> tuple[list[list[int]], dict[str, Any]]:
    start = time.perf_counter()
    if mode == "ar":
        sequences = _student_sample_batch(
            server_url,
            prompts,
            max_new_tokens,
            temperature=temperature,
            timeout=timeout,
            sampling_params_extra=sampling_extra,
        )
        return sequences, {
            "latency_s": time.perf_counter() - start,
            "student_sampling_mode": "ar",
        }

    if mode.startswith("native_k"):
        k_toks = int(mode.removeprefix("native_k"))
        sequences, metrics = _student_sample_native_mtp_batch(
            server_url,
            prompts,
            max_new_tokens,
            singleshot_mtp={
                "k_toks": k_toks,
                "mask_token_id": int(mask_token_id),
                "sampling_mode": "native",
                "native_mtp_debug_trace": True,
                "native_mtp_debug_max_steps": 256,
            },
            temperature=temperature,
            timeout=timeout,
            sampling_params_extra=sampling_extra,
        )
        metrics["latency_s"] = time.perf_counter() - start
        return sequences, metrics

    raise ValueError(f"Unsupported mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--rollout-samples-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--modes", default="ar,native_k1")
    parser.add_argument("--mask-token-id", type=int, default=151662)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sampling-params-json", default='{"top_k":1,"top_p":1.0,"min_p":0.0}')
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--text-max-chars", type=int, default=1200)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415

    _wait_for_sglang(args.server_url, timeout=args.timeout)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_records = _load_prompts_from_rollouts(Path(args.rollout_samples_jsonl), limit=args.prompt_limit)
    prompts = [record["prompt_token_ids"] for record in prompt_records]
    modes = _parse_modes(args.modes)
    sampling_extra = json.loads(args.sampling_params_json)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for mode in modes:
            singleton_sequences: list[list[int]] = []
            singleton_metrics: list[dict[str, Any]] = []
            for prompt in prompts:
                sequences, metrics = _run_mode(
                    mode=mode,
                    server_url=args.server_url,
                    prompts=[prompt],
                    max_new_tokens=args.max_new_tokens,
                    mask_token_id=args.mask_token_id,
                    temperature=args.temperature,
                    sampling_extra=sampling_extra,
                    timeout=args.timeout,
                )
                singleton_sequences.append(sequences[0])
                singleton_metrics.append(metrics)

            batched_sequences, batched_metrics = _run_mode(
                mode=mode,
                server_url=args.server_url,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                mask_token_id=args.mask_token_id,
                temperature=args.temperature,
                sampling_extra=sampling_extra,
                timeout=args.timeout,
            )

            mismatches = 0
            native_replay_failures = 0
            for prompt_idx, (record, prompt, singleton_sequence, batched_sequence) in enumerate(
                zip(prompt_records, prompts, singleton_sequences, batched_sequences)
            ):
                singleton_generated = [int(x) for x in singleton_sequence[len(prompt) :]]
                batched_generated = [int(x) for x in batched_sequence[len(prompt) :]]
                generated_match = singleton_generated == batched_generated
                if not generated_match:
                    mismatches += 1

                singleton_trace = _native_trace_from_metrics(singleton_metrics[prompt_idx], 0)
                batched_trace = _native_trace_from_metrics(batched_metrics, prompt_idx)
                singleton_replay = _pending_replay_check(singleton_trace)
                batched_replay = _pending_replay_check(batched_trace)
                if mode != "ar" and (not singleton_replay["ok"] or not batched_replay["ok"]):
                    native_replay_failures += 1

                row = {
                    "mode": mode,
                    "prompt_idx": prompt_idx,
                    "source_step": record.get("source_step"),
                    "source_sample_idx": record.get("source_sample_idx"),
                    "prompt_token_count": len(prompt),
                    "prompt_tail_text": _decode(tokenizer, prompt[-128:], max_chars=args.text_max_chars),
                    "singleton_generated_token_ids": singleton_generated,
                    "batched_generated_token_ids": batched_generated,
                    "singleton_generated_text": _decode(
                        tokenizer, singleton_generated, max_chars=args.text_max_chars
                    ),
                    "batched_generated_text": _decode(
                        tokenizer, batched_generated, max_chars=args.text_max_chars
                    ),
                    "generated_match": generated_match,
                    "singleton_native_mtp_debug_trace": singleton_trace,
                    "batched_native_mtp_debug_trace": batched_trace,
                    "singleton_native_mtp_debug_trace_step_count": len(singleton_trace),
                    "batched_native_mtp_debug_trace_step_count": len(batched_trace),
                    "singleton_pending_replay": singleton_replay,
                    "batched_pending_replay": batched_replay,
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

            summaries[mode] = {
                "prompt_count": len(prompts),
                "singleton_batch_match_all": mismatches == 0,
                "singleton_batch_mismatch_count": mismatches,
                "native_replay_failure_count": native_replay_failures,
                "batched_metric_student_sampling_mode": batched_metrics.get("student_sampling_mode"),
                "batched_metric_latency_s": batched_metrics.get("latency_s"),
            }
            if mode != "ar":
                for key, value in batched_metrics.items():
                    if isinstance(value, (bool, int, float, str)):
                        summaries[mode][f"batched_metric/{key}"] = value

    summary = {
        "server_url": args.server_url,
        "model_path": args.model_path,
        "rollout_samples_jsonl": args.rollout_samples_jsonl,
        "output_jsonl": str(output_path),
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "sampling_params": sampling_extra,
        "modes": summaries,
        "all_modes_match": all(mode_summary["singleton_batch_match_all"] for mode_summary in summaries.values()),
        "total_mismatches": sum(int(mode_summary["singleton_batch_mismatch_count"]) for mode_summary in summaries.values()),
        "total_native_replay_failures": sum(int(mode_summary["native_replay_failure_count"]) for mode_summary in summaries.values()),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_on_mismatch and (summary["total_mismatches"] or summary["total_native_replay_failures"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
