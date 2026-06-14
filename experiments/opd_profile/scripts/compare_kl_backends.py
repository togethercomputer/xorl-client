#!/usr/bin/env python3
"""Head-to-head: OPD full-vocab reverse-KL backends at Qwen3.6 lm-head sizes.

Compares the three engine reverse-KL implementations that opd_loss_function
dispatches to, at the OPD default (lm_head_fp32=True, teacher_lm_head_fp32=True):

  - compiled       : compiled_reverse_kl_function (torch_compile, num_chunks)
  - streaming      : streaming_reverse_kl_function (caller upcasts weights to fp32)
  - streaming_lowmem: streaming_reverse_kl_lowmem_function (native-dtype weights,
                      per-chunk fp32 upcast; the AMDAHL-029..033 memory fix)

For each, over a sweep of valid-token counts N, reports fwd+bwd wall time and
PEAK CUDA memory. The peak-memory delta is the lever: a leaner KL frees GB that
can be spent on removing gradient-checkpoint recompute (the ~27%-of-fb OPD cost).

Single GPU, no trainer. Run e.g.:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/apanda/xorl-opd-throughput-20260614/src \
      /home/apanda/xorl-internal/.venv/bin/python compare_kl_backends.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from xorl.ops.loss.compiled_cross_entropy import compiled_reverse_kl_function
from xorl.ops.loss.opd_streaming_kl import (
    streaming_reverse_kl_function,
    streaming_reverse_kl_lowmem_function,
)

# Qwen3.6-35B-A3B lm-head.
H = 2048
V = 248320
PEAK_TFLOPS_BF16 = 989.0
GB = 1024 ** 3


def _make_inputs(n: int, device: str, weight_dtype: torch.dtype):
    """Realistic student/teacher hidden + tied-shape lm-head weights."""
    torch.manual_seed(0)
    # hidden states in bf16 (native activation dtype), require grad on student.
    student_hidden = torch.randn(n, H, device=device, dtype=torch.bfloat16) * 0.1
    teacher_hidden = torch.randn(n, H, device=device, dtype=torch.bfloat16) * 0.1
    student_w = (torch.randn(V, H, device=device, dtype=weight_dtype) * 0.02)
    teacher_w = (torch.randn(V, H, device=device, dtype=weight_dtype) * 0.02)
    labels = torch.randint(0, V, (n,), device=device, dtype=torch.long)
    return student_hidden, teacher_hidden, student_w, teacher_w, labels


def _time_fwd_bwd(fn, weight, iters):
    """Median fwd+bwd ms + peak mem GB. weight is the leaf we backprop into."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    # warmup (compile / autotune)
    for _ in range(2):
        if weight.grad is not None:
            weight.grad = None
        loss = fn().sum()
        loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        if weight.grad is not None:
            weight.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = fn().sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = (torch.cuda.max_memory_allocated() - base) / GB
    times.sort()
    return times[len(times) // 2], peak


def bench_compiled(n, device, num_chunks, iters):
    sh, th, sw, tw, lab = _make_inputs(n, device, torch.bfloat16)
    sw.requires_grad_(True)
    sh.requires_grad_(True)

    def fn():
        return compiled_reverse_kl_function(
            student_hidden_states=sh, student_weight=sw,
            teacher_hidden_states=th, teacher_weight=tw,
            labels=lab, ignore_index=-100, num_chunks=num_chunks,
            lm_head_fp32=True, teacher_lm_head_fp32=True,
        )

    return _time_fwd_bwd(fn, sw, iters)


def bench_streaming(n, device, vchunk, iters):
    # The engine upcasts weights+hidden to fp32 before calling (lm_head_fp32 path).
    sh, th, sw, tw, lab = _make_inputs(n, device, torch.bfloat16)
    sw_f = sw.float().detach().requires_grad_(True)
    sh_f = sh.float().detach().requires_grad_(True)
    th_f = th.float()
    tw_f = tw.float()

    def fn():
        return streaming_reverse_kl_function(
            student_hidden_states=sh_f, student_weight=sw_f,
            teacher_hidden_states=th_f, teacher_weight=tw_f,
            labels=lab, ignore_index=-100, vocab_chunk_size=vchunk,
        )

    return _time_fwd_bwd(fn, sw_f, iters)


def bench_lowmem(n, device, vchunk, iters):
    # Native-dtype weights, per-chunk fp32 upcast inside the kernel.
    sh, th, sw, tw, lab = _make_inputs(n, device, torch.bfloat16)
    sw.requires_grad_(True)
    sh_f = sh.float().detach().requires_grad_(True)  # engine upcasts hidden when lm_head_fp32
    th_f = th.float()

    def fn():
        return streaming_reverse_kl_lowmem_function(
            student_hidden_states=sh_f, student_weight=sw,
            teacher_hidden_states=th_f, teacher_weight=tw,
            labels=lab, ignore_index=-100, vocab_chunk_size=vchunk,
            compute_dtype=torch.float32, inplace_weight_grad=False,
        )

    return _time_fwd_bwd(fn, sw, iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-chunks", type=int, default=8, help="compiled backend chunks")
    ap.add_argument("--vchunks", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = "cuda"
    wbytes_fp32 = V * H * 4 / GB
    print(f"Qwen3.6 lm-head V={V} H={H}; fp32 weight = {wbytes_fp32:.2f} GB each "
          f"(student+teacher = {2*wbytes_fp32:.2f} GB)", flush=True)
    print(f"{'N':>7} {'backend':<28} {'fwd+bwd ms':>11} {'peak GB':>9}", flush=True)
    results = []
    for n in args.ns:
        rows = []
        try:
            t, p = bench_compiled(n, device, args.num_chunks, args.iters)
            rows.append((f"compiled(nchunks={args.num_chunks})", t, p))
        except Exception as e:
            rows.append((f"compiled(nchunks={args.num_chunks})", float("nan"), float("nan")))
            print(f"  compiled FAILED: {type(e).__name__}: {e}", flush=True)
        for vc in args.vchunks:
            try:
                t, p = bench_streaming(n, device, vc, args.iters)
                rows.append((f"streaming(vc={vc})", t, p))
            except Exception as e:
                rows.append((f"streaming(vc={vc})", float("nan"), float("nan")))
                print(f"  streaming vc={vc} FAILED: {type(e).__name__}: {e}", flush=True)
            try:
                t, p = bench_lowmem(n, device, vc, args.iters)
                rows.append((f"streaming_lowmem(vc={vc})", t, p))
            except Exception as e:
                rows.append((f"streaming_lowmem(vc={vc})", float("nan"), float("nan")))
                print(f"  lowmem vc={vc} FAILED: {type(e).__name__}: {e}", flush=True)
        for name, t, p in rows:
            print(f"{n:>7} {name:<28} {t:>11.1f} {p:>9.2f}", flush=True)
            results.append({"n": n, "backend": name, "ms": t, "peak_gb": p})
        print("", flush=True)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"H": H, "V": V, "results": results}, f, indent=2)
        print(f"wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
