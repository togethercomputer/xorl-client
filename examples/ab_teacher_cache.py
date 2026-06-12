#!/usr/bin/env python
"""Numerical A/B gate for the OPD teacher hidden-state cache.

Runs the SAME fixed token sequences through both teacher backends and compares
the cached hidden states they produce:

  (a) the xorl training-framework teacher -> POST /api/v1/forward (teacher_hidden_cache)
  (b) the xorl-sglang-internal teacher     -> POST /teacher_hidden_cache

Both write a ``[N_rows, hidden]`` safetensors cache (tensor key ``hidden_states``);
this script loads both, aligns rows per sample via each backend's
``cache_indices_by_sample``, and reports per-row cosine similarity. Both store the
post-final-norm head input, so with matched bf16 dtype the cosine must be ~1.0. If
the aggregate drops below the threshold, investigate dtype / TP / final-norm before
trusting the sglang teacher in a run.

It calls the real production helpers in ``on_policy_distillation.py`` so it
validates the actual client code path.

Run with the OPD repo on PYTHONPATH, e.g.:
    PYTHONPATH=/home/apanda/xorl-client-opd \
    <venv>/bin/python examples/ab_teacher_cache.py \
        --xorl-url http://teacher-xorl:30002 \
        --sglang-url http://teacher-sglang:30000 \
        --num-seqs 8 --min-len 64 --max-len 256
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _load_example():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "on_policy_distillation.py")
    spec = importlib.util.spec_from_file_location("opd_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opd_example"] = module  # so @dataclass / chz can resolve the module
    spec.loader.exec_module(module)
    return module


def _build_sequences(args):
    if args.seqs_file:
        import json

        with open(args.seqs_file) as f:
            seqs = [[int(t) for t in seq] for seq in json.load(f)]
        for seq in seqs:
            if len(seq) < 2:
                raise ValueError("every sequence in --seqs-file needs >= 2 tokens")
        return seqs
    rng = random.Random(args.seed)
    return [
        [rng.randint(1, args.vocab_size - 1) for _ in range(rng.randint(args.min_len, args.max_len))]
        for _ in range(args.num_seqs)
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xorl-url", required=True, help="xorl teacher base URL (/api/v1/forward)")
    p.add_argument("--sglang-url", required=True, help="xorl-sglang teacher base URL (/teacher_hidden_cache)")
    p.add_argument("--teacher-model-id", default="default", help="model_id for the xorl teacher session")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--min-len", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seqs-file", default=None, help="optional JSON: list of token-id lists")
    p.add_argument("--prompt-frac", type=float, default=0.5,
                   help="fraction of each sequence treated as the prompt prefix (rest = CoT+answer)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--cos-threshold", type=float, default=0.99)
    p.add_argument("--max-frac-below", type=float, default=0.01)
    args = p.parse_args()

    if args.min_len < 2 or args.max_len < args.min_len:
        p.error("require 2 <= --min-len <= --max-len")

    opd = _load_example()
    seqs = _build_sequences(args)
    # Treat a leading fraction of each sequence as the prompt prefix; the rest is the
    # supervised CoT+answer. Clamp so every sample keeps >= 1 row (needs >= 2 gen tokens).
    prompt_lens = [max(0, min(int(len(s) * args.prompt_frac), len(s) - 2)) for s in seqs]
    print(f"[ab] {len(seqs)} sequences, lengths {[len(s) for s in seqs]}, prompt_lens {prompt_lens}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="ab_teacher_cache_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    xorl_path = out_dir / "teacher_cache_xorl.safetensors"
    sglang_path = out_dir / "teacher_cache_sglang.safetensors"

    print(f"[ab] xorl   teacher -> {args.xorl_url}")
    xorl_res = opd._teacher_cache_from_xorl(
        args.xorl_url, seqs, xorl_path, args.teacher_model_id, args.timeout, prompt_token_lens=prompt_lens
    )
    print(f"[ab] sglang teacher -> {args.sglang_url}")
    sglang_res = opd._teacher_cache_from_sglang(
        args.sglang_url, seqs, sglang_path, args.timeout, prompt_token_lens=prompt_lens
    )

    xt = load_file(str(xorl_path))["hidden_states"].float()
    st = load_file(str(sglang_path))["hidden_states"].float()
    print(f"[ab] xorl cache {tuple(xt.shape)}  sglang cache {tuple(st.shape)}")
    if xt.shape[1] != st.shape[1]:
        print(f"[ab] FAIL: hidden_size mismatch {xt.shape[1]} != {st.shape[1]} (check sglang TP all-gather)")
        return 1

    xi_all = xorl_res["cache_indices_by_sample"]
    si_all = sglang_res["cache_indices_by_sample"]
    cosines = []
    for i, seq in enumerate(seqs):
        xi, si = xi_all[i], si_all[i]
        if len(xi) != len(si):
            print(f"[ab] FAIL: sample {i} kept-row count differs xorl={len(xi)} sglang={len(si)}")
            return 1
        a = xt[torch.tensor(xi, dtype=torch.long)]
        b = st[torch.tensor(si, dtype=torch.long)]
        cosines.append(F.cosine_similarity(a, b, dim=-1))

    cos = torch.cat(cosines)
    mean, cmin = float(cos.mean()), float(cos.min())
    median = float(cos.median())
    p01 = float(torch.quantile(cos, 0.01))
    frac_below = float((cos < args.cos_threshold).float().mean())
    print("[ab] cosine(kept-position hiddens):")
    print(f"       mean={mean:.6f} median={median:.6f} min={cmin:.6f} p01={p01:.6f}")
    print(f"       fraction below {args.cos_threshold}: {frac_below:.4%}  (rows={cos.numel()})")

    if mean >= args.cos_threshold and frac_below <= args.max_frac_below:
        print(f"[ab] PASS: backends match (mean cosine {mean:.6f}).")
        return 0
    print("[ab] FAIL: backends diverge -- investigate dtype / TP / final-norm.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
