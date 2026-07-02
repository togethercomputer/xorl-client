"""Score fixed on-policy texts on a sglang sampler under base / repacked-GDN-LoRA
variants (prefill logprobs), for validating fused-GDN LoRA serving.

Usage:
  python score_gdn_lora.py --server http://...:30000 --out results.json \
      [--adapter NAME=/path ...] [--label base]

Each --adapter NAME=/path is loaded via /load_lora_adapter, scored, unloaded.
Scoring with no adapters (base only) is always included, labeled --label.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

GEN = Path(
    "/shared/apanda/wordle-sft-runs/20260702T034135Z-grpo-wq36-2n-k3lora-head-wd2qf-wordle-r16-grpo2node/generations.jsonl"
)
N_TEXTS = 8
MAX_CHARS = 4000


def collect_texts() -> list[str]:
    texts = []
    for line in GEN.open():
        d = json.loads(line)
        if d.get("event") == "grpo_student_turn" and d.get("step", 0) >= 13:
            t = d.get("raw_text") or ""
            if len(t) > 500:
                texts.append(t[:MAX_CHARS])
        if len(texts) >= N_TEXTS:
            break
    assert len(texts) >= N_TEXTS, f"only {len(texts)} texts"
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=/path")
    ap.add_argument("--label", default="base", help="label for the no-adapter scores")
    args = ap.parse_args()
    smp = args.server.rstrip("/")

    texts = collect_texts()
    print(f"scoring {len(texts)} texts on {smp}")

    def score(text: str, lora_name: str | None) -> list[float]:
        payload = {
            "text": text,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        if lora_name:
            payload["lora_path"] = lora_name
        r = requests.post(f"{smp}/generate", json=payload, timeout=300)
        r.raise_for_status()
        meta = r.json()["meta_info"]
        return [tok[0] for tok in meta["input_token_logprobs"] if tok[0] is not None]

    def flush():
        requests.post(f"{smp}/flush_cache", timeout=60)

    results: dict[str, list[list[float]]] = {}

    # base (no adapter)
    t0 = time.time()
    results[args.label] = [score(t, None) for t in texts]
    print(f"{args.label}: scored in {time.time() - t0:.1f}s")
    flush()

    for spec in args.adapter:
        name, path = spec.split("=", 1)
        r = requests.post(
            f"{smp}/load_lora_adapter",
            json={"lora_name": name, "lora_path": path},
            timeout=600,
        )
        print(f"load {name}: {r.status_code} {r.text[:300]}")
        r.raise_for_status()
        t0 = time.time()
        results[name] = [score(t, name) for t in texts]
        print(f"{name}: scored in {time.time() - t0:.1f}s")
        r = requests.post(f"{smp}/unload_lora_adapter", json={"lora_name": name}, timeout=120)
        print(f"unload {name}: {r.status_code}")
        flush()

    Path(args.out).write_text(json.dumps(results))
    print(f"wrote {args.out}")

    def diff(a_name: str, b_name: str):
        A, B = results[a_name], results[b_name]
        deltas = []
        for sa, sb in zip(A, B):
            n = min(len(sa), len(sb))
            deltas += [abs(x - y) for x, y in zip(sa[:n], sb[:n])]
        return (
            sum(deltas) / len(deltas),
            max(deltas),
            sum(1 for d in deltas if d > 1e-9) / len(deltas),
        )

    names = list(results)
    print(f"\n{'pair':52s} {'mean|dlp|':>12s} {'max|dlp|':>12s} {'frac>0':>8s}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m, x, f = diff(names[i], names[j])
            print(f"{names[i]} vs {names[j]:34s} {m:12.6g} {x:12.6g} {f:8.3f}")


if __name__ == "__main__":
    main()
