"""Standalone greedy-eval harness for Countdown 24-puzzle adapters.

Loads a saved LoRA from disk via SGLang's /load_lora_adapter, runs greedy
inference (temperature=0) on the 8 puzzles, prints exact-match counts. Used
to compare ZORL vs GRPO (or any saved adapter) on the SAME evaluation
protocol — bypasses the flaky NCCL-based sync_inference_weights path.

Usage:
  python eval_countdown.py \
      --infer-url http://zorl-30b-password-sglang.apanda.svc.cluster.local:30000 \
      --lora-name eval-zorl-probe-16 \
      --lora-path /shared/path/to/saved/lora/dir \
      --label "ZORL probe 16"

The LoRA path must be in PEFT format (adapter_config.json + adapter_model.safetensors).
"""

from __future__ import annotations

import argparse
import sys
import time

import requests
from run_countdown_test import (
    COUNTDOWN_PUZZLES,
    QUESTION_META,
    SYSTEM_PROMPT,
    _expression_uses_numbers_exactly_once,
    _extract_expression,
    _question_text,
    _safe_eval_expr,
)


def load_lora_on_sglang(infer_url, lora_name, lora_path, *, timeout=300):
    payload = {"lora_name": lora_name, "lora_path": lora_path, "pinned": False}
    resp = requests.post(
        f"{infer_url}/load_lora_adapter",
        json=payload,
        headers={"Connection": "close"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        msg = (data.get("error_message") or "").lower()
        # Idempotent: if the adapter is already loaded under the same name, treat as success.
        if "already" in msg and "loaded" in msg:
            return data
        raise RuntimeError(f"load_lora_adapter returned {data}")
    return data


def unload_lora_on_sglang(infer_url, lora_name, *, timeout=120):
    try:
        requests.post(
            f"{infer_url}/unload_lora_adapter",
            json={"lora_name": lora_name},
            headers={"Connection": "close"},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        print(f"  (cleanup) unload_lora_adapter failed: {type(exc).__name__}: {exc}")


def greedy_query(infer_url, served_model_name, prompt, *, lora_name=None, max_new_tokens=64, timeout=120):
    """Single chat completion at temperature=0."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": served_model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": int(max_new_tokens),
        "stream": False,
    }
    if lora_name is not None:
        payload["lora_path"] = lora_name
    resp = requests.post(
        f"{infer_url}/v1/chat/completions",
        json=payload,
        headers={"Connection": "close"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_greedy_eval(infer_url, served_model_name, *, lora_name, label, max_new_tokens):
    print(f"\n=== Greedy eval: {label} ===")
    if lora_name:
        print(f"  LoRA: {lora_name}")
    else:
        print("  LoRA: <base model only>")
    correct = 0
    rows = []
    for question_key, _solution in [(f"q_{i:02d}", s) for i, (_n, _t, s) in enumerate(COUNTDOWN_PUZZLES)]:
        meta = QUESTION_META[question_key]
        prompt = _question_text(meta)
        try:
            answer = greedy_query(
                infer_url,
                served_model_name,
                prompt,
                lora_name=lora_name,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            answer = f"<<ERROR: {type(exc).__name__}: {exc}>>"
        parsed = _extract_expression(answer)
        uses_each_once = parsed is not None and _expression_uses_numbers_exactly_once(parsed, meta["numbers"])
        eval_value = _safe_eval_expr(parsed) if uses_each_once else None
        match = eval_value is not None and abs(eval_value - float(meta["target"])) < 1e-6
        correct += int(match)
        rows.append((question_key, meta["numbers"], meta["target"], answer, parsed, eval_value, match))
        nums = "+".join(str(n) for n in meta["numbers"])
        flag = "OK  " if match else "FAIL"
        print(
            f"  [{flag}] {question_key} ({nums}->{meta['target']}): '{answer[:80]}' parsed='{parsed}' eval={eval_value}"
        )
    print(f"  Greedy score: {correct}/{len(COUNTDOWN_PUZZLES)}")
    return correct, rows


def main():
    parser = argparse.ArgumentParser(description="Greedy temp=0 eval for Countdown adapters")
    parser.add_argument("--infer-url", required=True, help="SGLang base URL (e.g. http://...:30000)")
    parser.add_argument("--lora-name", required=False, default=None, help="LoRA name to register on SGLang")
    parser.add_argument("--lora-path", required=False, default=None, help="On-disk path to saved PEFT LoRA dir")
    parser.add_argument("--label", required=False, default="adapter", help="Display label")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keep-loaded", action="store_true", default=False, help="Skip the /unload at the end")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        default=False,
        help="Assume the LoRA is already registered; skip /load_lora_adapter",
    )
    args = parser.parse_args()

    if args.lora_name is None and args.lora_path is not None:
        parser.error("--lora-path requires --lora-name")
    if args.lora_name is not None and args.lora_path is None and not args.skip_load:
        parser.error("--lora-name without --lora-path requires --skip-load (assume already registered)")

    if args.lora_name is not None and not args.skip_load:
        print(f"Loading LoRA '{args.lora_name}' from '{args.lora_path}'...")
        t0 = time.time()
        load_lora_on_sglang(args.infer_url, args.lora_name, args.lora_path)
        print(f"Loaded in {time.time() - t0:.1f}s")
    try:
        run_greedy_eval(
            args.infer_url,
            args.served_model_name,
            lora_name=args.lora_name,
            label=args.label,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        if args.lora_name is not None and not args.keep_loaded:
            print(f"Unloading LoRA '{args.lora_name}'...")
            unload_lora_on_sglang(args.infer_url, args.lora_name)


if __name__ == "__main__":
    sys.exit(main())
