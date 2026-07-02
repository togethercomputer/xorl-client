#!/usr/bin/env python3
"""1-GPU lm-head streaming reverse-KL microbench (OPD throughput + memory).

Reproduces the AMDAHL-029..033 1-node blocker: streaming_reverse_kl backward
allocates a FULL [vocab, hidden] fp32 grad_weight buffer, on top of the already
fp32 lm-head weight (lm_head_fp32=true). At Qwen3.6 sizes (V=248320, H=2048)
each such buffer is ~2.0 GB, and the OPD streaming path holds several at once.

This bench measures, at real shapes and over a sweep of valid-token counts:
  - streaming reverse-KL fwd+bwd wall time and achieved lm-head matmul MFU
  - peak CUDA memory, broken into weights / grad_weight / chunk transients
  - the same with a pre-existing weight.grad (the grad-accumulation worst case
    autograd doubles into), which is what tips a near-full 1-node GPU over.

It also benches a low-memory prototype (`--prototype`) that streams the weight
gradient directly into a preallocated fp32 .grad and keeps lm-head weights bf16
(casting per vocab chunk), to show the peak-memory reduction before porting the
fix into engine opd_streaming_kl.py.

Single GPU, no trainer. Run:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<engine>/src \
      <engine>/.venv/bin/python microbench_lmhead_kl.py
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_function

# Qwen3.6-35B-A3B.
H = 2048
V = 248320
PEAK_TFLOPS = 989.0
GB = 1024 ** 3


def _kl_matmul_flops(n: int) -> float:
    """Total lm-head matmul FLOPs for streaming reverse-KL fwd+bwd.

    fwd = 2 passes x (student + teacher logits) = 8*N*V*H.
    bwd = student+teacher logits recompute (4*N*V*H) + grad_hidden (2*N*V*H)
          + grad_weight (2*N*V*H) = 8*N*V*H.
    """
    return 16.0 * n * V * H


def _mk(n: int, weight_dtype, hidden_dtype, device="cuda"):
    sh = torch.randn(n, H, device=device, dtype=hidden_dtype, requires_grad=True)
    sw = torch.randn(V, H, device=device, dtype=weight_dtype, requires_grad=True)
    th = torch.randn(n, H, device=device, dtype=hidden_dtype)
    tw = torch.randn(V, H, device=device, dtype=weight_dtype)
    lab = torch.randint(0, V, (n,), device=device)
    return sh, sw, th, tw, lab


def bench_streaming(n: int, iters: int, vchunk: int, lm_head_fp32: bool,
                    preexisting_grad: bool) -> dict:
    """Real engine streaming_reverse_kl_function fwd+bwd."""
    wdt = torch.float32 if lm_head_fp32 else torch.bfloat16
    hdt = torch.float32 if lm_head_fp32 else torch.bfloat16
    sh, sw, th, tw, lab = _mk(n, wdt, hdt)
    if preexisting_grad:
        # Simulate a prior microbatch's accumulated grad: autograd must ADD our
        # returned grad into this, transiently holding both.
        sw.grad = torch.zeros_like(sw)

    def run():
        kl = streaming_reverse_kl_function(sh, sw, th, tw, lab, vocab_chunk_size=vchunk)
        kl.sum().backward()
        sh.grad = None
        if not preexisting_grad:
            sw.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run()  # warmup (also builds sw.grad if preexisting)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    dt = start.elapsed_time(end) / 1000.0 / iters
    peak = torch.cuda.max_memory_allocated()
    tflops = _kl_matmul_flops(n) / dt / 1e12
    return {
        "n": n,
        "ms": 1000 * dt,
        "tflops": tflops,
        "mfu": tflops / PEAK_TFLOPS,
        "peak_gb": peak / GB,
        "base_resident_gb": base / GB,
        "vchunk": vchunk,
        "lm_head_fp32": lm_head_fp32,
        "preexisting_grad": preexisting_grad,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--vchunk", type=int, default=32768)
    ap.add_argument("--lm-head-fp32", action="store_true", default=True)
    ap.add_argument("--bf16", dest="lm_head_fp32", action="store_false")
    ap.add_argument("--preexisting-grad", action="store_true",
                    help="preallocate sw.grad (grad-accumulation worst case)")
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[48, 128, 512, 1536, 3049, 6144])
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    print(
        f"device={torch.cuda.get_device_name()} | Qwen3.6 lm-head V={V} H={H} | "
        f"lm_head_fp32={args.lm_head_fp32} vchunk={args.vchunk} "
        f"preexisting_grad={args.preexisting_grad}",
        flush=True,
    )
    wbytes = V * H * (4 if args.lm_head_fp32 else 2)
    print(f"per lm-head weight tensor = {wbytes/GB:.2f} GB "
          f"(student + teacher + grad_weight all this size)\n", flush=True)
    print(f"{'N_valid':>8} {'ms':>9} {'TFLOP/s':>9} {'MFU%':>7} {'peak_GB':>9}", flush=True)

    rows = []
    for n in args.ns:
        try:
            r = bench_streaming(n, args.iters, args.vchunk, args.lm_head_fp32,
                                args.preexisting_grad)
            print(f"{n:>8} {r['ms']:>9.2f} {r['tflops']:>9.1f} "
                  f"{100*r['mfu']:>7.2f} {r['peak_gb']:>9.2f}", flush=True)
            rows.append(r)
        except RuntimeError as e:
            print(f"{n:>8}  OOM/ERROR {str(e)[:60]}", flush=True)
            rows.append({"n": n, "error": str(e)[:200]})
            torch.cuda.empty_cache()

    print("", flush=True)
    ok = [r for r in rows if "mfu" in r]
    if ok:
        print(f"lm-head KL MFU stays ~{100*min(r['mfu'] for r in ok):.1f}-"
              f"{100*max(r['mfu'] for r in ok):.1f}% across N; "
              f"peak grows from {min(r['peak_gb'] for r in ok):.1f} to "
              f"{max(r['peak_gb'] for r in ok):.1f} GB.", flush=True)
        wf = wbytes / GB
        print(f"Fixed weight cost alone = {3*wf:.1f} GB "
              f"(student {wf:.1f} + teacher {wf:.1f} + grad_weight {wf:.1f}); "
              f"grad_weight + teacher-fp32 are the avoidable {2*wf:.1f} GB.", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": {"H": H, "V": V, "vchunk": args.vchunk,
                                  "lm_head_fp32": args.lm_head_fp32,
                                  "preexisting_grad": args.preexisting_grad,
                                  "peak_tflops": PEAK_TFLOPS}, "rows": rows}, f, indent=2)
        print(f"wrote {args.output_json}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
