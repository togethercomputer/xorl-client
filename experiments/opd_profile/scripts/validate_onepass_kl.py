#!/usr/bin/env python3
"""Validate the streaming reverse-KL against a brute-force full-logit reference,
and time fwd+bwd. Run the SAME script with different PYTHONPATH worktrees to
compare the one-pass forward (kl-fused) vs the two-pass forward (apanda-dev):

  # one-pass (fused) worktree
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src \
    /home/apanda/xorl-internal/.venv/bin/python validate_onepass_kl.py --tag onepass
  # two-pass (baseline) worktree
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/apanda/xorl-opd-throughput-20260614/src \
    /home/apanda/xorl-internal/.venv/bin/python validate_onepass_kl.py --tag twopass

Correctness (small N, exact reference) must hold; timing (large N) is the win.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_function

H = 2048
V = 248320
GB = 1024 ** 3


def reference_kl(sh, sw, th, tw, labels, ignore_index):
    """Brute-force full-logit reverse-KL (the ground truth)."""
    s_logits = (sh @ sw.t()).float()
    t_logits = (th @ tw.t()).float()
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    valid = (labels != ignore_index).to(kl.dtype)
    return kl * valid


def check_correctness(n, vchunk, device):
    torch.manual_seed(0)
    sh = (torch.randn(n, H, device=device, dtype=torch.float32) * 0.1)
    th = (torch.randn(n, H, device=device, dtype=torch.float32) * 0.1)
    sw0 = (torch.randn(V, H, device=device, dtype=torch.float32) * 0.02)
    tw = (torch.randn(V, H, device=device, dtype=torch.float32) * 0.02)
    labels = torch.randint(0, V, (n,), device=device, dtype=torch.long)
    labels[: n // 8] = -100  # some ignored tokens

    # streaming path
    sh_s = sh.clone().requires_grad_(True)
    sw_s = sw0.clone().requires_grad_(True)
    kl_s = streaming_reverse_kl_function(
        student_hidden_states=sh_s, student_weight=sw_s,
        teacher_hidden_states=th, teacher_weight=tw,
        labels=labels, ignore_index=-100, vocab_chunk_size=vchunk,
    )
    kl_s.sum().backward()

    # reference path
    sh_r = sh.clone().requires_grad_(True)
    sw_r = sw0.clone().requires_grad_(True)
    kl_r = reference_kl(sh_r, sw_r, th, tw, labels, -100)
    kl_r.sum().backward()

    kl_err = (kl_s - kl_r).abs().max().item()
    kl_rel = ((kl_s - kl_r).abs() / kl_r.abs().clamp_min(1e-6)).max().item()
    gw_err = (sw_s.grad - sw_r.grad).abs().max().item()
    gw_scale = sw_r.grad.abs().max().item()
    gh_err = (sh_s.grad - sh_r.grad).abs().max().item()
    gh_scale = sh_r.grad.abs().max().item()
    print(f"[correctness N={n} vc={vchunk}]")
    print(f"  kl   max|abs|={kl_err:.3e}  max|rel|={kl_rel:.3e}")
    print(f"  gradW max|abs|={gw_err:.3e}  (scale {gw_scale:.3e}, rel {gw_err/max(gw_scale,1e-9):.3e})")
    print(f"  gradH max|abs|={gh_err:.3e}  (scale {gh_scale:.3e}, rel {gh_err/max(gh_scale,1e-9):.3e})")
    ok = kl_rel < 1e-3 and gw_err / max(gw_scale, 1e-9) < 1e-3 and gh_err / max(gh_scale, 1e-9) < 1e-3
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def time_fwd_bwd(n, vchunk, device, iters=5):
    torch.manual_seed(0)
    sh = (torch.randn(n, H, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    th = (torch.randn(n, H, device=device, dtype=torch.float32) * 0.1)
    sw = (torch.randn(V, H, device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    tw = (torch.randn(V, H, device=device, dtype=torch.float32) * 0.02)
    labels = torch.randint(0, V, (n,), device=device, dtype=torch.long)

    def run():
        if sw.grad is not None:
            sw.grad = None
        if sh.grad is not None:
            sh.grad = None
        kl = streaming_reverse_kl_function(
            student_hidden_states=sh, student_weight=sw,
            teacher_hidden_states=th, teacher_weight=tw,
            labels=labels, ignore_index=-100, vocab_chunk_size=vchunk,
        )
        kl.sum().backward()

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    peak = (torch.cuda.max_memory_allocated() - base) / GB
    ts.sort()
    return ts[len(ts) // 2], peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="?")
    ap.add_argument("--vchunk", type=int, default=32768)
    ap.add_argument("--time-ns", type=int, nargs="+", default=[8192])
    args = ap.parse_args()
    device = "cuda"
    print(f"=== tag={args.tag} V={V} H={H} vchunk={args.vchunk} ===", flush=True)
    check_correctness(512, args.vchunk, device)
    for n in args.time_ns:
        t, p = time_fwd_bwd(n, args.vchunk, device)
        print(f"[timing N={n}] fwd+bwd median = {t:.1f} ms  peak = {p:.2f} GB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
