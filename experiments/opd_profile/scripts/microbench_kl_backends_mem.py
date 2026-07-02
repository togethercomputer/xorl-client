#!/usr/bin/env python3
"""Compare OPD reverse-KL backends' PEAK MEMORY at the lm_head_fp32 regime.

Directive (2026-06-14): keep the lm-head in fp32 (reverse-KL accuracy on rare
near-certain tokens) and find which loss mode minimizes the 1-node memory blocker
(the 1.89 GiB fp32 lm-head grad_weight, AMDAHL-033).

Backends, all at Qwen3.6 lm-head sizes (V=248320, H=2048), N=3049 (OPD batch):
  1. streaming (baseline OPD path): weights pre-cast to fp32 -> full fp32 copies +
     full fp32 grad_weight buffer.
  2. streaming_lowmem (PR #373): bf16 weights, per-chunk fp32 upcast, native-dtype
     in-place grad -> no full fp32 copies, no full grad buffer. Bit-exact fp32.
  3. compiled (torch_compile auto_chunker): lm_head_fp32=True -> ALSO weight.float()
     (full fp32 copy); chunks TOKENS to avoid full logits.

Reports peak GB per backend (no-grad-accum and grad-accum worst case) + numerics
vs the streaming-fp32 reference. Single GPU.
"""
from __future__ import annotations
import argparse, torch

H, V, GB = 2048, 248320, 1024 ** 3


def _mk(n, seed=11):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sh = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    sw = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, generator=g)
    th = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    tw = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, generator=g)
    lab = torch.randint(0, V, (n,), device="cuda", generator=g)
    return sh, sw, th, tw, lab


def _peak(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated() / GB
    torch.cuda.empty_cache()
    return p, out


def run_streaming_fp32(sh, sw, th, tw, lab, vchunk, preexist):
    from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_function
    swf = sw.float().detach().requires_grad_(True)
    shf = sh.float().detach().requires_grad_(True)
    if preexist:
        swf.grad = torch.zeros_like(swf)
    kl = streaming_reverse_kl_function(shf, swf, th.float(), tw.float(), lab, vocab_chunk_size=vchunk)
    kl.sum().backward()
    return kl.detach(), shf.grad.detach().clone(), swf.grad.to(torch.bfloat16).detach().clone()


def run_lowmem(sh, sw, th, tw, lab, vchunk, preexist):
    from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_lowmem_function
    sw2 = sw.clone().detach().requires_grad_(True)
    sh2 = sh.float().detach().requires_grad_(True)
    if preexist:
        sw2.grad = torch.zeros_like(sw2)
    kl = streaming_reverse_kl_lowmem_function(sh2, sw2, th, tw, lab, vocab_chunk_size=vchunk,
                                              compute_dtype=torch.float32, inplace_weight_grad=True)
    kl.sum().backward()
    return kl.detach(), sh2.grad.detach().clone(), sw2.grad.detach().clone()


def run_compiled(sh, sw, th, tw, lab, num_chunks, preexist):
    from xorl.ops.loss.compiled_cross_entropy import compiled_reverse_kl_function
    sw2 = sw.clone().detach().requires_grad_(True)
    sh2 = sh.clone().detach().requires_grad_(True)
    if preexist:
        sw2.grad = torch.zeros_like(sw2.float())
    kl = compiled_reverse_kl_function(sh2, sw2, th, tw, lab, num_chunks=num_chunks,
                                      lm_head_fp32=True, teacher_lm_head_fp32=True)
    kl.sum().backward()
    gw = sw2.grad
    return kl.detach(), sh2.grad.detach().clone(), (gw.to(torch.bfloat16).detach().clone() if gw is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3049)
    ap.add_argument("--vchunk", type=int, default=8192)
    ap.add_argument("--num-chunks", type=int, default=64)
    args = ap.parse_args()
    n = args.n
    print(f"device={torch.cuda.get_device_name()} | V={V} H={H} N={n} | lm_head_fp32 regime\n", flush=True)
    wf = V * H * 4 / GB
    print(f"(one fp32 [V,H] weight = {wf:.2f} GB; baseline holds student+teacher copies + grad_weight)\n", flush=True)

    rows = []
    for preexist in (False, True):
        tag = "grad-accum (pre-existing .grad)" if preexist else "no grad-accum"
        print(f"==== {tag} ====", flush=True)
        # reference = streaming fp32 (current OPD path)
        sh, sw, th, tw, lab = _mk(n)
        try:
            p_s, (kl_ref, gh_ref, gw_ref) = _peak(lambda: run_streaming_fp32(sh, sw, th, tw, lab, args.vchunk, preexist))
            print(f"  streaming fp32 (baseline)   peak = {p_s:6.2f} GB", flush=True)
        except RuntimeError as e:
            print(f"  streaming fp32 (baseline)   OOM/ERR {str(e)[:50]}"); kl_ref = None; p_s = float('nan')
        # lowmem
        sh, sw, th, tw, lab = _mk(n)
        try:
            p_l, (kl_l, gh_l, gw_l) = _peak(lambda: run_lowmem(sh, sw, th, tw, lab, args.vchunk, preexist))
            msg = f"  streaming_lowmem (PR#373)   peak = {p_l:6.2f} GB"
            if kl_ref is not None:
                dkl = (kl_ref - kl_l).abs().max().item(); dgw = (gw_ref.float() - gw_l.float()).abs().max().item()
                msg += f"   | vs baseline: max|Δkl|={dkl:.2e} max|Δgrad_w|={dgw:.2e}  saved {p_s-p_l:.2f} GB"
            print(msg, flush=True)
        except RuntimeError as e:
            print(f"  streaming_lowmem (PR#373)   OOM/ERR {str(e)[:50]}")
        # compiled
        sh, sw, th, tw, lab = _mk(n)
        try:
            p_c, (kl_c, gh_c, gw_c) = _peak(lambda: run_compiled(sh, sw, th, tw, lab, args.num_chunks, preexist))
            msg = f"  compiled (auto_chunker fp32) peak = {p_c:6.2f} GB"
            if kl_ref is not None:
                dkl = (kl_ref - kl_c).abs().max().item(); msg += f"   | vs baseline: max|Δkl|={dkl:.2e}"
            print(msg, flush=True)
        except Exception as e:
            print(f"  compiled (auto_chunker fp32) OOM/ERR {str(e)[:80]}", flush=True)
        print("", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
