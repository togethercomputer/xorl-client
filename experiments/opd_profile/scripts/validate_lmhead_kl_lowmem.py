#!/usr/bin/env python3
"""Validate engine streaming_reverse_kl_lowmem_function against the current OPD
fp32 path, at Qwen3.6 lm-head sizes (V=248320, H=2048). Confirms gradient
identity and measures the peak-memory reduction.

Current OPD fp32 path (what opd_loss does today when lm_head_fp32=True):
    bf16 param -> weight.float() (full fp32 copy) -> streaming_reverse_kl ->
    fp32 grad on the copy -> autograd FloatBackward downcasts to bf16 param.grad.

Lowmem path:
    bf16 param passed as-is -> per-chunk fp32 upcast in the matmul -> grad
    accumulated/returned directly in bf16. No full fp32 weight copies, optional
    in-place .grad (no second full [V,H] buffer).

Because slicing commutes with the elementwise fp32 upcast and vocab chunks
partition grad rows disjointly, the two are gradient-identical (the only
difference is per-chunk vs whole downcast of disjoint rows -> bit-exact).

Run:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<engine>/src \
      <engine>/.venv/bin/python validate_lmhead_kl_lowmem.py
"""

from __future__ import annotations

import argparse

import torch

from xorl.ops.loss.opd_streaming_kl import (
    streaming_reverse_kl_function,
    streaming_reverse_kl_lowmem_function,
)

H = 2048
V = 248320
GB = 1024 ** 3


def _mk(n, weight_dtype, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sh = torch.randn(n, H, device="cuda", dtype=torch.float32, generator=g)
    sw = torch.randn(V, H, device="cuda", dtype=weight_dtype, generator=g)
    th = torch.randn(n, H, device="cuda", dtype=torch.float32, generator=g)
    tw = torch.randn(V, H, device="cuda", dtype=weight_dtype, generator=g)
    lab = torch.randint(0, V, (n,), device="cuda", generator=g)
    return sh, sw, th, tw, lab


def _peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated() / GB
    torch.cuda.empty_cache()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3049)
    ap.add_argument("--vchunk", type=int, default=8192)
    ap.add_argument("--inplace", action="store_true", default=True)
    ap.add_argument("--no-inplace", dest="inplace", action="store_false")
    ap.add_argument("--preexisting-grad", action="store_true")
    args = ap.parse_args()
    n, vchunk = args.n, args.vchunk
    print(f"N={n} V={V} H={H} vchunk={vchunk} inplace={args.inplace} "
          f"preexisting_grad={args.preexisting_grad}\n", flush=True)

    # ---- Numerics ----
    # Reference = current fp32 path: upcast bf16 param to fp32, run, downcast grad.
    sh, sw_bf16, th, tw_bf16, lab = _mk(n, torch.bfloat16, seed=11)
    swf = sw_bf16.float().detach().requires_grad_(True)
    shf = sh.detach().requires_grad_(True)
    twf = tw_bf16.float()
    kl_ref = streaming_reverse_kl_function(shf, swf, th.float(), twf, lab, vocab_chunk_size=vchunk)
    kl_ref.sum().backward()
    ref_gh = shf.grad.clone()
    ref_gw = swf.grad.to(torch.bfloat16)  # FloatBackward downcasts to param dtype

    # Lowmem = bf16 weights, per-chunk fp32 upcast, bf16 grad.
    sw2 = sw_bf16.clone().detach().requires_grad_(True)
    sh2 = sh.detach().requires_grad_(True)
    kl_lm = streaming_reverse_kl_lowmem_function(
        sh2, sw2, th, tw_bf16, lab, vocab_chunk_size=vchunk,
        compute_dtype=torch.float32, inplace_weight_grad=args.inplace,
    )
    kl_lm.sum().backward()
    lm_gh = sh2.grad.clone()
    lm_gw = sw2.grad.clone()

    kl_d = (kl_ref - kl_lm).abs().max().item()
    gh_d = (ref_gh - lm_gh).abs().max().item()
    gw_d = (ref_gw.float() - lm_gw.float()).abs().max().item()
    gw_rel = gw_d / (ref_gw.float().abs().max().item() + 1e-12)
    print(f"  kl max|diff| = {kl_d:.3e}")
    print(f"  grad_hidden max|diff| = {gh_d:.3e}")
    print(f"  grad_weight max|diff| = {gw_d:.3e} (rel {gw_rel:.3e})")
    ok = kl_d < 1e-3 and gh_d < 1e-4 and gw_rel < 1e-4
    print(f"  NUMERICS {'OK (gradient-identical)' if ok else 'MISMATCH'}\n", flush=True)

    # ---- Peak memory: current fp32 path vs lowmem ----
    def ref_run():
        sh_, sw_, th_, tw_, lab_ = _mk(n, torch.bfloat16, seed=1)
        swf_ = sw_.float().requires_grad_(True)
        if args.preexisting_grad:
            swf_.grad = torch.zeros_like(swf_)
        streaming_reverse_kl_function(
            sh_, swf_, th_.float(), tw_.float(), lab_, vocab_chunk_size=vchunk
        ).sum().backward()

    def lm_run():
        sh_, sw_, th_, tw_, lab_ = _mk(n, torch.bfloat16, seed=1)
        sw_ = sw_.requires_grad_(True)
        if args.preexisting_grad:
            sw_.grad = torch.zeros_like(sw_)
        streaming_reverse_kl_lowmem_function(
            sh_, sw_, th_, tw_, lab_, vocab_chunk_size=vchunk,
            compute_dtype=torch.float32, inplace_weight_grad=args.inplace,
        ).sum().backward()

    p_ref = _peak(ref_run)
    p_lm = _peak(lm_run)
    print(f"  peak current fp32 path = {p_ref:.2f} GB", flush=True)
    print(f"  peak lowmem path       = {p_lm:.2f} GB  (saved {p_ref - p_lm:.2f} GB, "
          f"{100*(p_ref-p_lm)/p_ref:.0f}%)", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
