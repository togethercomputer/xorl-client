#!/usr/bin/env python3
"""W2-2: arm B's mandated offline QUANTIZED-DECODE diagnostic
(ATTEMPT1_PREREG §WAVE-2; the B-G3 escalation follow-up). No new training.

Decodes the banked eval items with the quantizer ACTIVE (the mode the arm was
TRAINED in: two-pass prefill, slots snapped before the reader pass) on the
banked armBL checkpoint, and pairs the results item-by-item against the
banked PRODUCTION rows (sglang, unsnapped). Also runs dose0 (no slots), where
quantized == plain forward — the sandbox-vs-sglang instrument-calibration row.

Single GPU, sandbox code (diagnostic exception — it measures the
sandbox-vs-production gap itself). Outputs per-item JSONLs (tag
{arm_tag}-qdec) + a paired comparison JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from mechanism_forward import FeatureConfig, MarinMechModel, marin_annotate_spans
from ms_arm_mask_check import load_sandbox_sd
from recur_loop import build_batch, load_tokenizer

RL_CKPT = "/shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B"


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="banked armBL sandbox ckpt (file or sharded dir)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--doses", default="0,1,2,4")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arm-tag", default="armBL")
    ap.add_argument("--production-dir", required=True,
                    help="banked production eval dir (armB_ext/evals) for pairing")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quantize", default="hard")
    ap.add_argument("--quantize-metric", default="logits")
    args = ap.parse_args()

    device = "cuda:0"
    tok = load_tokenizer(RL_CKPT)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(RL_CKPT, dtype=torch.bfloat16,
                                              attn_implementation="sdpa")
    model = MarinMechModel(hf)
    sd = load_sandbox_sd(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing[:3], unexpected[:3])
    model = model.to(device).eval()

    rows = [json.loads(l) for l in open(args.data)][: args.n]
    dtag = os.path.basename(args.data).replace(".jsonl", "")
    os.makedirs(args.out_dir, exist_ok=True)
    comparison = {"ckpt": args.ckpt, "data": args.data, "doses": {}}

    for mult in [int(x) for x in args.doses.split(",")]:
        feats = (FeatureConfig(quantize_slots=args.quantize,
                               quantize_metric=args.quantize_metric)
                 if mult > 0 else FeatureConfig())
        arm = "r1" if mult > 0 else "c0"
        items_out = []
        for c0 in range(0, len(rows), args.batch):
            chunk = rows[c0: c0 + args.batch]
            items = [{"prompt": r["problem"], "gold": str(r["answer"]),
                      "band": r["n_ops"], "k": mult * r["n_ops"],
                      "sig": f"qd{r['idx']}", "idx": r["idx"]} for r in chunk]
            b = build_batch(tok, items, arm, device, with_answer=False)
            marin_annotate_spans(tok, b, items, arm)
            with torch.no_grad():
                gen = model.generate_greedy_masked(b, feats, max_new_tokens=16)
            for i, r in enumerate(chunk):
                text = tok.decode(gen[i]).split("<|")[0]
                e, g, v = score_numeric(text, int(r["answer"]))
                items_out.append({"idx": r["idx"], "gold": r["answer"], "exact": e,
                                  "graded": round(g, 4), "pred": v,
                                  "k": mult * r["n_ops"], "n_ops": r["n_ops"],
                                  "raw": text[:80]})
        stem = f"{args.arm_tag}-qdec__{dtag}__dose{mult}x"
        with open(f"{args.out_dir}/{stem}.jsonl", "w") as fh:
            for it in items_out:
                fh.write(json.dumps(it) + "\n")
        n = len(items_out)
        q_exact = sum(i["exact"] for i in items_out) / n
        entry = {"n": n, "quantized_exact": round(q_exact, 4),
                 "quantized_graded": round(sum(i["graded"] for i in items_out) / n, 4)}
        # pair against the banked production row
        prod_path = f"{args.production_dir}/{args.arm_tag}__{dtag}__dose{mult}x.jsonl"
        if os.path.exists(prod_path):
            prod = {str(json.loads(l)["idx"]): json.loads(l) for l in open(prod_path)}
            mine = {str(i["idx"]): i for i in items_out}
            common = set(prod) & set(mine)
            b_ = sum(1 for i in common if mine[i]["exact"] > prod[i]["exact"])
            c_ = sum(1 for i in common if mine[i]["exact"] < prod[i]["exact"])
            entry.update({"production_exact": round(
                sum(prod[i]["exact"] for i in common) / max(1, len(common)), 4),
                "mcnemar_q_only": b_, "mcnemar_prod_only": c_,
                "mcnemar_p": round(mcnemar_p(b_, c_), 5), "n_paired": len(common)})
        comparison["doses"][f"{mult}x"] = entry
        print(json.dumps({f"dose{mult}x": entry}))

    # preregistered adjudication (W2-2 execution prereg): serve-gap fires iff
    # any nonzero dose has quantized - production >= +0.05 with McNemar p<.05
    fires = any(
        e.get("production_exact") is not None
        and e["quantized_exact"] - e["production_exact"] >= 0.05
        and e.get("mcnemar_p", 1) < 0.05
        and e.get("mcnemar_q_only", 0) > e.get("mcnemar_prod_only", 0)
        for m, e in comparison["doses"].items() if m != "0x")
    comparison["reading"] = ("SERVE-GAP (quantized >> production: B's null was the "
                             "serve mismatch; serving-side snap worth speccing)"
                             if fires else
                             "MECHANISM-NULL (quantized ~= production: the quantizer "
                             "mechanism itself is null at this scale)")
    out_p = f"{args.out_dir}/qdec_comparison_{dtag}.json"
    with open(out_p, "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps({"reading": comparison["reading"]}))
    print(f"-> {out_p}")


if __name__ == "__main__":
    main()
