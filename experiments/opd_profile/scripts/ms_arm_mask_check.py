#!/usr/bin/env python3
"""ARM-A mask-verification row (preregistered gate): on N real band items,
under the TRAINING mask (answer->problem blocked), the recomputed answer-row
attention mass over the problem span must be EXACTLY 0.0 at every layer —
checked on the INIT weights and on the trained arm checkpoint. Also runs the
masked-eval diagnostic (generate_greedy_masked) on the same items: greedy
answers with the training mask kept during decode vs the unmasked production
behavior (the prereg's caveat row for train/eval mask mismatch).

Single GPU; sandbox code (diagnostic instrument, NOT the eval path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from mechanism_forward import (FeatureConfig, MarinMechModel, rows_probs_from_capture,
                               need_from_feats)
from mechanism_sandbox_train import RL_CKPT  # noqa: F401  (path constant)
from recur_loop import build_batch, load_tokenizer
from mechanism_forward import marin_annotate_spans


def load_sandbox_sd(ckpt: str) -> dict:
    """Sandbox ckpts are a single .safetensors file (<12GB) OR a sharded dir
    with model.safetensors.index.json (the armAL-final crash: safe_open on a
    directory raises OSError 19)."""
    import json as _json
    from pathlib import Path
    from safetensors import safe_open
    p = Path(ckpt)
    if p.is_dir():
        wmap = _json.load(open(p / "model.safetensors.index.json"))["weight_map"]
        opened, sd = {}, {}
        for k, shard in wmap.items():
            if shard not in opened:
                opened[shard] = safe_open(p / shard, framework="pt", device="cpu")
            sd[k] = opened[shard].get_tensor(k)
        return sd
    f = safe_open(p, framework="pt", device="cpu")
    return {k: f.get_tensor(k) for k in f.keys()}


def load_model(ckpt: str | None, device):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        RL_CKPT, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = MarinMechModel(hf)
    if ckpt:
        sd = load_sandbox_sd(ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        assert not [m for m in missing], missing[:5]
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="sandbox safetensors (None = init weights)")
    ap.add_argument("--data", required=True, help="cb-band jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--mask-blocks", default="answer->problem")
    ap.add_argument("--dose-mult", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda:0"
    tok = load_tokenizer(RL_CKPT)
    rows = [json.loads(l) for l in open(args.data)][args.offset: args.offset + args.n]
    items = [{"prompt": r["problem"], "gold": str(r["answer"]), "band": r["n_ops"],
              "k": args.dose_mult * r["n_ops"], "sig": f"mc{r['idx']}",
              "idx": r["idx"]} for r in rows]
    model = load_model(args.ckpt, device)
    n_l = model.n_layers
    model.install_hooks(attn_layers=list(range(n_l)))
    feats = FeatureConfig(mask_blocks=args.mask_blocks, attn_target_coef=1.0,
                          attn_layers=",".join(map(str, range(n_l))))
    need = {"attn_layers": set(range(n_l))}

    batch = build_batch(tok, items, "r1", device, with_answer=True)
    marin_annotate_spans(tok, batch, items, "r1")
    with torch.no_grad():
        out = model(batch, feats, need)
    per_layer = {}
    worst = 0.0
    for l in range(n_l):
        mx = 0.0
        for b in range(len(items)):
            sp = batch["spans"][b]
            rows_t = torch.arange(sp["answer"][0], sp["answer"][1], device=device)
            probs = rows_probs_from_capture(out, out["capture"], l, b, rows_t, "marin")
            p0, p1 = sp["problem"]
            mx = max(mx, float(probs[:, :, p0:p1].abs().max()))
        per_layer[l] = mx
        worst = max(worst, mx)

    # masked-eval diagnostic: greedy WITH the training mask maintained
    masked_answers, unmasked_answers = [], []
    for it in items:
        b1 = build_batch(tok, [it], "r1", device, with_answer=False)
        marin_annotate_spans(tok, b1, [it], "r1")
        with torch.no_grad():
            g_m = model.generate_greedy_masked(b1, feats, max_new_tokens=16)
            g_u = model.generate_greedy_masked(b1, FeatureConfig(), max_new_tokens=16)
        masked_answers.append(tok.decode(g_m[0]).split("<|")[0].strip())
        unmasked_answers.append(tok.decode(g_u[0]).split("<|")[0].strip())

    rep = {"tag": args.tag, "ckpt": args.ckpt, "mask_blocks": args.mask_blocks,
           "n_items": len(items),
           "answer_to_problem_mass_max": worst,
           "per_layer_max": per_layer,
           "PASS_exact_zero": worst == 0.0,
           "masked_vs_unmasked_greedy": [
               {"idx": it["idx"], "gold": it["gold"], "masked": m, "unmasked": u}
               for it, m, u in zip(items, masked_answers, unmasked_answers)]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({"tag": args.tag, "mass_max": worst, "pass": worst == 0.0}))


if __name__ == "__main__":
    main()
