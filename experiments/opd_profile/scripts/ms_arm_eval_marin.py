#!/usr/bin/env python3
"""ATTEMPT-1 marin arm eval instrument (arms A/B + their within-substrate C0
and the raw floor). PRODUCTION path only: prefill token ids -> sglang
/generate (DC-chain launch flags) -> greedy retries=0. No sandbox code in the
model path — prompt construction reuses the recur_loop token conventions the
arms trained with (think-wrapped PAUSE slots; 'Answer:' prefill token fact).

Dose rows (win-condition clause 1): --doses "0,1,2,4" = slot multipliers of
the item's base k (n_ops for cb arith; band depth for recur substitution);
dose 0 = ABLATED (native empty-think c0 layout — the recurrence track's
ablation convention).

Schemas:
  cb    : comp-bench jsonl (problem/answer/n_ops/depth) — numeric scoring
          (exact + graded prox .2), gold = int answer.
  recur : recur_stage0 jsonl (prompt/gold/band) — substitution transfer row;
          gold is a symbol string (e.g. 'PT'); exact = symbol match.

Output: one per-item JSONL + summary JSON per (tag, dataset, dose) under
--out-dir (FILES-ONLY discipline).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RL_CKPT = "/shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B"
PAUSE_ID = 1981
START_THINK_ID = 128002
END_THINK_ID = 128003
NL_ID = 198
NL2_ID = 271


def gen(url, ids, max_tokens):
    payload = {"input_ids": ids,
               "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0}}
    req = urllib.request.Request(f"{url}/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["text"]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def score_numeric(text, gold):
    m = re.search(r"-?\d+", text.replace(",", ""))
    v = int(m.group(0)) if m else None
    if v == gold:
        return 1.0, 1.0, v
    if v is not None:
        rel = abs(v - gold) / max(abs(gold), 1)
        if rel < 0.2:
            return 0.0, 1.0 - rel / 0.2, v
    return 0.0, 0.0, v


def score_symbol(text, gold):
    m = re.search(r"\b([A-Z]{2})\b", text)
    v = m.group(1) if m else None
    return (1.0 if v == gold else 0.0), (1.0 if v == gold else 0.0), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--schema", choices=["cb", "recur"], required=True)
    ap.add_argument("--band-filter", type=int, default=None,
                    help="recur schema: keep rows with band == this")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--doses", default="0,1,2,4",
                    help="slot multipliers of base k; 0 = ablated empty-think")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    from recur_loop import load_tokenizer
    from experiments.marin.standalone.prompts import render_chat_prompt
    tok = load_tokenizer(RL_CKPT)
    ans_ids = tok.encode("Answer:", add_special_tokens=False)  # NOT 'Answer: ' (token fact)

    rows = [json.loads(l) for l in open(args.data)]
    if args.schema == "recur" and args.band_filter is not None:
        rows = [r for r in rows if r["band"] == args.band_filter]
    rows = rows[args.offset: args.offset + args.n]
    assert rows, "no rows selected"

    def base_k(r):
        return r["n_ops"] if args.schema == "cb" else r["band"]

    def prompt_text(r):
        return r["problem"] if args.schema == "cb" else r["prompt"]

    def gold(r):
        return int(r["answer"]) if args.schema == "cb" else str(r["gold"])

    cache = {}

    def prefill(r, mult):
        key = id(r)
        if key not in cache:
            cache[key] = tok.encode(
                render_chat_prompt(tok, prompt_text(r), add_boxed_instruction=True),
                add_special_tokens=False)
        p = cache[key]
        if mult == 0:  # ablated: native empty think (recur c0 convention)
            return p + [START_THINK_ID, NL2_ID, END_THINK_ID, NL2_ID] + ans_ids
        k = mult * base_k(r)
        return (p + [START_THINK_ID, NL_ID] + [PAUSE_ID] * k
                + [NL_ID, END_THINK_ID, NL2_ID] + ans_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    dtag = os.path.basename(args.data).replace(".jsonl", "")
    if args.band_filter is not None:
        dtag += f"_d{args.band_filter}"
    scorer = score_numeric if args.schema == "cb" else score_symbol

    for mult in [int(x) for x in args.doses.split(",")]:
        def one(r):
            out = gen(args.url, prefill(r, mult), args.max_tokens)
            e, g, v = scorer(out, gold(r))
            it = {"idx": r.get("idx", r.get("sig")), "gold": gold(r), "exact": e,
                  "graded": round(g, 4), "pred": v, "k": mult * base_k(r),
                  "raw": out[:120]}
            if args.schema == "cb":
                it["n_ops"] = r["n_ops"]
                it["depth"] = r.get("depth")
            return it

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            items = list(ex.map(one, rows))
        stem = f"{args.model_tag}__{dtag}__dose{mult}x"
        with open(f"{args.out_dir}/{stem}.jsonl", "w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        n = len(items)
        exact = sum(i["exact"] for i in items) / n
        rep = {"model_tag": args.model_tag, "data": args.data, "dose_mult": mult,
               "n": n, "exact": round(exact, 4), "exact_ci95": wilson(exact, n),
               "graded": round(sum(i["graded"] for i in items) / n, 4),
               "parse_fail": sum(1 for i in items if i["pred"] is None)}
        with open(f"{args.out_dir}/{stem}.summary.json", "w") as fh:
            json.dump(rep, fh, indent=2)
        print(json.dumps(rep))


if __name__ == "__main__":
    main()
