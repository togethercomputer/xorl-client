#!/usr/bin/env python3
"""Probe frozen SGLang native-MTP generation on saved OPD prompts.

This diagnostic intentionally does no training. It compares ordinary greedy AR
generation with SGLang native MTP generation for several k values on the exact
prompt tokens captured in rollout artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
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


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _max_repeated_run(tokens: list[int]) -> int:
    best = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        best = max(best, current)
    return best


def _token_rates(tokenizer: Any, token_ids: list[int]) -> dict[str, float]:
    if not token_ids:
        return {
            "digit_token_rate": 0.0,
            "newline_token_rate": 0.0,
            "okay_token_rate": 0.0,
            "special_token_rate": 0.0,
            "unique_token_fraction": 0.0,
        }
    decoded = [_decode(tokenizer, [token_id], max_chars=64) for token_id in token_ids]
    digit = 0
    newline = 0
    okay = 0
    special = 0
    for text in decoded:
        stripped = text.strip()
        if stripped and stripped.isdigit():
            digit += 1
        if text and set(text) <= {"\n", "\r"}:
            newline += 1
        if stripped.lower() == "okay":
            okay += 1
        if text.startswith("<|") and text.endswith("|>"):
            special += 1
    denom = len(token_ids)
    return {
        "digit_token_rate": digit / denom,
        "newline_token_rate": newline / denom,
        "okay_token_rate": okay / denom,
        "special_token_rate": special / denom,
        "unique_token_fraction": len(set(token_ids)) / denom,
    }


def _completion_metrics(tokenizer: Any, token_ids: list[int], text: str) -> dict[str, Any]:
    repeated = _max_repeated_run(token_ids)
    return {
        "generated_token_count": len(token_ids),
        "generated_char_count": len(text),
        "max_repeated_token_run": repeated,
        "repeat_run_ge_16": repeated >= 16,
        **_token_rates(tokenizer, token_ids),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(rows)}
    numeric_keys = [
        "generated_token_count",
        "generated_char_count",
        "max_repeated_token_run",
        "digit_token_rate",
        "newline_token_rate",
        "okay_token_rate",
        "special_token_rate",
        "unique_token_fraction",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_max"] = max(values)
    summary["samples_with_repeat_run_ge_16"] = sum(1 for row in rows if row.get("repeat_run_ge_16"))
    generated = [tuple(row.get("generated_token_ids", [])) for row in rows]
    summary["unique_completion_count"] = len(set(generated))
    for key, value in rows[0].get("mode_metrics", {}).items():
        if isinstance(value, (bool, int, float, str)):
            summary[f"mode_metric/{key}"] = value
    return summary


def _run_ar_mode(
    *,
    server_url: str,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    sampling_extra: dict[str, Any],
    timeout: float,
) -> tuple[list[list[int]], dict[str, Any]]:
    start = time.perf_counter()
    sequences = _student_sample_batch(
        server_url,
        prompts,
        max_new_tokens,
        temperature=temperature,
        timeout=timeout,
        sampling_params_extra=sampling_extra,
    )
    return sequences, {"latency_s": time.perf_counter() - start, "student_sampling_mode": "ar"}


def _run_native_mtp_mode(
    *,
    server_url: str,
    prompts: list[list[int]],
    max_new_tokens: int,
    k_toks: int,
    mask_token_id: int,
    temperature: float,
    sampling_extra: dict[str, Any],
    timeout: float,
    strategy: list[Any] | None = None,
    adaptive_window_mode: str | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    singleshot = {
        "k_toks": int(k_toks),
        "mask_token_id": int(mask_token_id),
        "sampling_mode": "native",
        "native_mtp_debug_trace": True,
        "native_mtp_debug_max_steps": 256,
    }
    if strategy is not None:
        singleshot["mtp_strategy"] = strategy
    if adaptive_window_mode:
        singleshot["mtp_adaptive_window_mode"] = adaptive_window_mode
    start = time.perf_counter()
    sequences, metrics = _student_sample_native_mtp_batch(
        server_url,
        prompts,
        max_new_tokens,
        singleshot_mtp=singleshot,
        temperature=temperature,
        timeout=timeout,
        sampling_params_extra=sampling_extra,
    )
    metrics["latency_s"] = time.perf_counter() - start
    return sequences, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--rollout-samples-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k-values", default="1,2,4,8,16")
    parser.add_argument("--conf-adapt-thresholds", default="0.9")
    parser.add_argument("--conf-adapt-k", type=int, default=16)
    parser.add_argument("--conf-adapt-window-mode", default="hf_exact")
    parser.add_argument("--mask-token-id", type=int, default=151662)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sampling-params-json", default='{"top_k":1,"top_p":1.0,"min_p":0.0}')
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--text-max-chars", type=int, default=1200)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415

    _wait_for_sglang(args.server_url, timeout=args.timeout)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_records = _load_prompts_from_rollouts(Path(args.rollout_samples_jsonl), limit=args.prompt_limit)
    prompts = [record["prompt_token_ids"] for record in prompt_records]
    sampling_extra = json.loads(args.sampling_params_json)
    k_values = _parse_int_list(args.k_values)
    thresholds = _parse_float_list(args.conf_adapt_thresholds)

    modes: list[tuple[str, Any]] = [("ar", None)]
    modes.extend((f"native_k{k}", {"k": k, "strategy": None}) for k in k_values)
    modes.extend(
        (
            f"conf_adapt_k{args.conf_adapt_k}_t{str(threshold).replace('.', 'p')}",
            {"k": args.conf_adapt_k, "strategy": ["conf_adapt", threshold]},
        )
        for threshold in thresholds
        if not math.isnan(threshold)
    )

    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ar_generated_by_prompt: dict[int, list[int]] = {}
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for mode_name, mode_cfg in modes:
            if mode_name == "ar":
                sequences, mode_metrics = _run_ar_mode(
                    server_url=args.server_url,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    sampling_extra=sampling_extra,
                    timeout=args.timeout,
                )
            else:
                sequences, mode_metrics = _run_native_mtp_mode(
                    server_url=args.server_url,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    k_toks=int(mode_cfg["k"]),
                    mask_token_id=args.mask_token_id,
                    temperature=args.temperature,
                    sampling_extra=sampling_extra,
                    timeout=args.timeout,
                    strategy=mode_cfg.get("strategy"),
                    adaptive_window_mode=args.conf_adapt_window_mode if mode_cfg.get("strategy") else None,
                )

            for prompt_idx, (record, prompt, sequence) in enumerate(zip(prompt_records, prompts, sequences)):
                generated = [int(x) for x in sequence[len(prompt) :]]
                text = _decode(tokenizer, generated, max_chars=args.text_max_chars)
                native_trace = []
                if mode_name != "ar":
                    sample_metadata = mode_metrics.get("_student_sample_metadata")
                    if isinstance(sample_metadata, list) and prompt_idx < len(sample_metadata):
                        sample_meta = sample_metadata[prompt_idx]
                        if isinstance(sample_meta, dict) and isinstance(sample_meta.get("mtp_debug_trace"), list):
                            native_trace = sample_meta["mtp_debug_trace"]
                row = {
                    "mode": mode_name,
                    "prompt_idx": prompt_idx,
                    "source_step": record.get("source_step"),
                    "source_sample_idx": record.get("source_sample_idx"),
                    "prompt_token_count": len(prompt),
                    "prompt_tail_text": _decode(tokenizer, prompt[-128:], max_chars=args.text_max_chars),
                    "generated_token_ids": generated,
                    "generated_text": text,
                    "mode_metrics": mode_metrics,
                    "native_mtp_debug_trace": native_trace,
                    "native_mtp_debug_trace_step_count": len(native_trace),
                    **_completion_metrics(tokenizer, generated, text),
                }
                if mode_name == "ar":
                    ar_generated_by_prompt[prompt_idx] = generated
                    row["matches_ar"] = True
                elif prompt_idx in ar_generated_by_prompt:
                    row["matches_ar"] = generated == ar_generated_by_prompt[prompt_idx]
                rows_by_mode[mode_name].append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "server_url": args.server_url,
        "model_path": args.model_path,
        "rollout_samples_jsonl": args.rollout_samples_jsonl,
        "output_jsonl": str(output_path),
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "sampling_params": sampling_extra,
        "mode_order": [mode for mode, _ in modes],
        "modes": {mode: _mode_summary(rows) for mode, rows in rows_by_mode.items()},
    }
    for mode, rows in rows_by_mode.items():
        if mode == "ar":
            continue
        match_values = [bool(row.get("matches_ar")) for row in rows if "matches_ar" in row]
        if match_values:
            summary["modes"][mode]["matches_ar_all"] = all(match_values)
            summary["modes"][mode]["matches_ar_count"] = sum(1 for value in match_values if value)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
