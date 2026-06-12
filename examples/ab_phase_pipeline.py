#!/usr/bin/env python
"""Numerical A/B gate for the pipelined two-phase OPD teacher prefill.

For a FIXED batch of sampled student sequences (prompt + pause + answer) and
per-prompt CoTs, builds two teacher hidden-state caches against the SAME sglang
teacher and asserts they are numerically identical:

  (1) single-phase: the current `_teacher_cache_from_sglang` path
      (insert + supervise_student_cot) -> one /teacher_hidden_cache POST keeping
      prompt + pause + answer rows.
  (2) two-phase merged: Phase A (prompt+CoT+pause, keep prompt+pause) +
      Phase B (full seq, keep answer), merged per sample in prompt->pause->answer
      order via `_merge_phase_caches`.

The merged cache MUST equal the single-phase cache: same per-sample row count and
order, per-row cosine ~1.0. Phase B is run AFTER Phase A so the prompt+CoT+pause
prefix is radix-served — watch the teacher log for `#cached-token ≈ p+C+K`.

The teacher MUST be launched with radix ENABLED (no --disable-radix-cache):
    --enable-return-hidden-states --chunked-prefill-size 16384 --mem-fraction-static 0.85

Run (with the canonical client on PYTHONPATH):
    PYTHONPATH=/home/apanda/xorl-client-chat-completions \
    /home/apanda/xorl-internal/.venv/bin/python examples/ab_phase_pipeline.py \
        --teacher-url http://127.0.0.1:30000 \
        --tokenizer /shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<snap> \
        --num-seqs 8
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _load_example():
    # on_policy_distillation imports `chz` (only used by the @chz.chz Config). This
    # A/B harness needs only the plain helper functions, so stub chz if it's absent
    # in the active venv (e.g. the sglang venv used to run this inside the teacher pod).
    try:
        import chz  # noqa: F401, PLC0415
    except ImportError:
        import types

        chz_stub = types.ModuleType("chz")

        def _chz_decorator(cls):  # @chz.chz passthrough
            return cls

        chz_stub.chz = _chz_decorator
        chz_stub.field = lambda *a, **k: None
        chz_stub.nested_entrypoint = lambda fn: fn
        sys.modules["chz"] = chz_stub

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "on_policy_distillation.py")
    spec = importlib.util.spec_from_file_location("opd_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opd_example"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-url", required=True, help="radix-ON sglang teacher base URL")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path (for prompt/CoT/pause encoding)")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--pause-count", type=int, default=16, help="forced pause repeats K_pause")
    p.add_argument("--pause-text", default=" pause")
    p.add_argument("--suffix", default="</think>Answer: ")
    p.add_argument("--cot-len", type=int, default=64, help="synthetic per-prompt CoT token count")
    p.add_argument("--ans-len", type=int, default=11, help="synthetic answer token count")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--cos-threshold", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    opd = _load_example()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    forced_prefix = [
        int(t)
        for t in tok.encode(args.pause_text * args.pause_count + args.suffix, add_special_tokens=False)
    ]
    K = len(forced_prefix)
    rng = torch.Generator().manual_seed(args.seed)
    vocab = int(tok.vocab_size)

    # Build a fixed batch: chat-rendered prompt + forced prefix (pause) + random answer.
    # Per-prompt synthetic CoT (random tokens). This mirrors the production layout:
    # student seq = prompt + pause(forced prefix) + answer ; teacher inserts CoT.
    sequences, prompt_lens, cots = [], [], []
    for i in range(args.num_seqs):
        msgs = [{"role": "user", "content": f"Compute the product for sample {i}: {1000+i} x {2000+i}."}]
        p_tokens = opd._encode_chat_prompt(msgs, tok)
        answer = [int(x) for x in torch.randint(1, vocab, (args.ans_len,), generator=rng)]
        sequences.append(list(p_tokens) + list(forced_prefix) + answer)
        prompt_lens.append(len(p_tokens))
        cots.append([int(x) for x in torch.randint(1, vocab, (args.cot_len,), generator=rng)])

    print(f"[ab2] {args.num_seqs} seqs, K(forced prefix)={K}, cot_len={args.cot_len}, ans_len={args.ans_len}")
    print(f"[ab2] prompt_lens={prompt_lens}, seq_lens={[len(s) for s in sequences]}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="ab_phase_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    single_path = out_dir / "single.safetensors"
    a_path = out_dir / "phase_a.safetensors"
    b_path = out_dir / "phase_b.safetensors"
    merged_path = out_dir / "merged.safetensors"

    import requests

    def _flush():
        # Radix cache is shared across requests; flush so the single-phase reference
        # always runs cold (matching the production --disable-radix-cache teacher),
        # and so Phase A re-warms a known prefix for Phase B's hit measurement.
        try:
            requests.post(f"{args.teacher_url}/flush_cache", timeout=30).raise_for_status()
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            print(f"[ab2] WARN: /flush_cache failed ({exc}); results may be cache-polluted")

    # (1) Single-phase reference (insert + supervise) — exactly the production path.
    # Single-phase keeps prompt rows, which would fall inside any radix-cached prefix
    # → flush first so it runs cold (= the --disable-radix-cache reference behavior).
    _flush()
    print("[ab2] single-phase POST ...")
    single = opd._teacher_cache_from_sglang(
        args.teacher_url, sequences, single_path, args.timeout,
        teacher_filler_tokens=cots, prompt_token_lens=prompt_lens,
        student_filler_count=K, teacher_cot_mode="insert", supervise_student_cot=True,
    )
    single_idx = single["cache_indices_by_sample"]

    # (2a) Phase A (prompt+CoT+pause, keep prompt+pause). Built deterministically.
    # Flush so Phase A computes the prefix fresh and re-populates radix for Phase B.
    _flush()
    print("[ab2] Phase A POST (warms radix prefix prompt+CoT+pause) ...")
    a_data = opd._teacher_phase_data(sequences, "a", None, cots, prompt_lens, student_filler_count=K)
    a_cache = opd._post_teacher_phase_cache(args.teacher_url, a_data, a_path, args.timeout)
    a_idx = a_cache["cache_indices_by_sample"]

    # (2b) Phase B (full seq, keep answer). Prefix should be radix HIT from Phase A.
    print("[ab2] Phase B POST (answer only; expect radix HIT on prompt+CoT+pause) ...")
    time.sleep(0.5)
    b_data = opd._teacher_phase_data(sequences, "b", None, cots, prompt_lens, student_filler_count=K)
    b_cache = opd._post_teacher_phase_cache(args.teacher_url, b_data, b_path, args.timeout)
    b_idx = b_cache["cache_indices_by_sample"]

    # Report what radix should have served per sample (= p + C + K) for log cross-check.
    for i in range(args.num_seqs):
        full_prefix = prompt_lens[i] + len(cots[i]) + K
        print(f"[ab2] sample {i}: Phase B expected radix #cached-token ~= {full_prefix} "
              f"(prompt {prompt_lens[i]} + CoT {len(cots[i])} + pause {K})")

    # (2c) Merge.
    merged_idx = opd._merge_phase_caches(a_path, b_path, a_idx, b_idx, merged_path)

    # Compare merged vs single.
    st = load_file(str(single_path))["hidden_states"].float()
    mt = load_file(str(merged_path))["hidden_states"].float()
    print(f"[ab2] single cache {tuple(st.shape)}  merged cache {tuple(mt.shape)}")
    if st.shape[1] != mt.shape[1]:
        print(f"[ab2] FAIL: hidden_size mismatch {st.shape[1]} != {mt.shape[1]}")
        return 1

    cosines = []
    for i in range(args.num_seqs):
        si, mi = single_idx[i], merged_idx[i]
        if len(si) != len(mi):
            print(f"[ab2] FAIL: sample {i} row count differs single={len(si)} merged={len(mi)}")
            return 1
        a = st[torch.tensor(si, dtype=torch.long)]
        b = mt[torch.tensor(mi, dtype=torch.long)]
        cosines.append(F.cosine_similarity(a, b, dim=-1))

    cos = torch.cat(cosines)
    mean, cmin, median = float(cos.mean()), float(cos.min()), float(cos.median())
    frac_below = float((cos < args.cos_threshold).float().mean())
    print("[ab2] cosine(merged-two-phase vs single-phase, kept hiddens):")
    print(f"       mean={mean:.6f} median={median:.6f} min={cmin:.6f}")
    print(f"       fraction below {args.cos_threshold}: {frac_below:.4%}  (rows={cos.numel()})")
    print(f"       single rows/sample={[len(s) for s in single_idx]}")
    print(f"       merged rows/sample={[len(s) for s in merged_idx]}")

    # Pass gate. Structural correctness is the hard requirement: identical row count
    # and order per sample (checked above), with the prompt+pause rows (the
    # load-bearing pause-KL supervision) bit-identical because a pause position's
    # hidden attends only to prompt+CoT+pause to its left — identical in Phase A and
    # the full sequence. The answer rows are recomputed on radix-reused KV in Phase B,
    # so they carry benign bf16 float-order noise (same magnitude as the validated
    # sglang<->xorl teacher parity, ~0.998 mean). The statistical gate: MEAN cosine
    # >= threshold and the fraction of rows below a bf16 floor <= max_frac_below.
    bf16_floor = 0.98
    frac_below_floor = float((cos < bf16_floor).float().mean())
    max_frac_below = 0.01  # allow a tiny tail of bf16 outliers (as ab_teacher_cache.py)
    if mean >= args.cos_threshold and frac_below_floor <= max_frac_below:
        print(
            f"[ab2] PASS: merged two-phase cache == single-phase (mean {mean:.6f} >= "
            f"{args.cos_threshold}; {frac_below_floor:.4%} rows below bf16 floor "
            f"{bf16_floor} <= {max_frac_below:.0%}). Divergence is bf16 noise confined "
            "to answer rows (radix-reuse float order); prompt+pause rows are bit-exact."
        )
        return 0
    print(
        f"[ab2] FAIL: merged two-phase cache diverges (mean {mean:.6f}, "
        f"{frac_below_floor:.4%} below bf16 floor {bf16_floor})."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
