#!/usr/bin/env python3
"""1-GPU MoE expert-FFN GEMM microbench for Qwen3.6-35B-A3B (OPD throughput).

Root-cause probe for the ~1% executed MFU on the OPD/prefill stack. The student
is an A3B MoE: hidden=2048, moe_intermediate=512, 256 experts, top-8. With
balanced routing each expert sees only

    M_per_expert = tokens_in_EP_group * top_k / num_experts

tokens. At the OPD operating point (~2.3k real student tokens/rank) that is
~72 tokens/expert, i.e. a [72,2048]@[2048,512] GEMM -- far too skinny to fill
an H100 tensor core. This script sweeps M_per_expert and reports achieved
TFLOP/s and MFU for the SwiGLU expert FFN (gate/up/down) fwd+bwd, both as a
batched grouped GEMM (bmm, the kernel ceiling) and as a per-expert python loop
(the launch-overhead floor). It demonstrates that the GEMMs themselves reach
>10% MFU once M is large -- the deficit is small-M, not the kernel -- and prints
the tokens/rank needed to clear a target MFU.

Single GPU, no trainer, no OPD stack. Run:

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<engine>/src \
      <engine>/.venv/bin/python microbench_moe_gemm.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch

# Qwen3.6-35B-A3B (config.json: qwen3_5_moe_text).
H = 2048  # hidden_size
I = 512  # moe_intermediate_size (per expert)
E = 256  # num_experts
TOP_K = 8  # num_experts_per_tok
L = 40  # num_hidden_layers
# H100 SXM bf16 dense tensor-core peak (TFLOP/s).
PEAK_TFLOPS = 989.0


def _expert_ffn_flops(m_per_expert: int, n_experts: int) -> float:
    """fwd+bwd FLOPs for SwiGLU FFN over n_experts each seeing m tokens.

    gate[H->I] + up[H->I] + down[I->H] = 6*M*H*I fwd; bwd ~= 2x fwd.
    """
    fwd = 6.0 * m_per_expert * H * I * n_experts
    return 3.0 * fwd  # fwd + 2x bwd


def _time(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def bench_grouped_bmm(m: int, n_experts: int, iters: int, dtype) -> float:
    """Batched expert FFN via bmm -- the grouped-GEMM kernel ceiling (uniform M)."""
    dev = "cuda"
    x = torch.randn(n_experts, m, H, device=dev, dtype=dtype, requires_grad=True)
    wg = torch.randn(n_experts, H, I, device=dev, dtype=dtype, requires_grad=True)
    wu = torch.randn(n_experts, H, I, device=dev, dtype=dtype, requires_grad=True)
    wd = torch.randn(n_experts, I, H, device=dev, dtype=dtype, requires_grad=True)

    def run():
        g = torch.bmm(x, wg)
        u = torch.bmm(x, wu)
        h = torch.nn.functional.silu(g) * u
        out = torch.bmm(h, wd)
        out.sum().backward()
        x.grad = wg.grad = wu.grad = wd.grad = None

    return _time(run, iters)


def bench_loop(m: int, n_experts: int, iters: int, dtype) -> float:
    """Per-expert python loop -- the launch-overhead floor."""
    dev = "cuda"
    xs = [torch.randn(m, H, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wg = [torch.randn(H, I, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wu = [torch.randn(H, I, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wd = [torch.randn(I, H, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]

    def run():
        acc = 0.0
        for i in range(n_experts):
            g = xs[i] @ wg[i]
            u = xs[i] @ wu[i]
            h = torch.nn.functional.silu(g) * u
            acc = acc + (h @ wd[i]).sum()
        acc.backward()
        for i in range(n_experts):
            xs[i].grad = wg[i].grad = wu[i].grad = wd[i].grad = None

    return _time(run, iters)


def mfu(m: int, n_experts: int, dt: float) -> tuple[float, float]:
    tflops = _expert_ffn_flops(m, n_experts) / dt / 1e12
    return tflops, tflops / PEAK_TFLOPS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=8, help="expert-parallel degree -> local experts = E/ep")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--loop", action="store_true", help="also run per-expert loop floor (slow)")
    ap.add_argument("--target-mfu", type=float, default=0.10)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    n_local = E // args.ep
    print(
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} peak={PEAK_TFLOPS}TF | "
        f"Qwen3.6-A3B MoE H={H} I={I} E={E} top_k={TOP_K} | EP={args.ep} local_experts={n_local}",
        flush=True,
    )
    print(
        "M_per_expert maps to real tokens/rank by: tokens = M * E / top_k "
        f"= M * {E // TOP_K}\n",
        flush=True,
    )

    rows = []
    header = f"{'M/exp':>6} {'tok/rank':>9} {'grouped ms':>11} {'TFLOP/s':>9} {'MFU%':>7}"
    if args.loop:
        header += f" {'loop ms':>9} {'loopMFU%':>9}"
    print(header, flush=True)

    for m in [8, 16, 32, 64, 72, 128, 256, 512, 1024, 2048, 4096]:
        try:
            dt = bench_grouped_bmm(m, n_local, args.iters, dtype)
            tflops, frac = mfu(m, n_local, dt)
            tokens = m * E // TOP_K
            line = f"{m:>6} {tokens:>9} {1000*dt:>11.3f} {tflops:>9.1f} {100*frac:>7.2f}"
            row = {"m_per_expert": m, "tokens_per_rank": tokens, "grouped_ms": 1000 * dt,
                   "grouped_tflops": tflops, "grouped_mfu": frac}
            if args.loop:
                dtl = bench_loop(m, n_local, max(3, args.iters // 4), dtype)
                tflopsl, fracl = mfu(m, n_local, dtl)
                line += f" {1000*dtl:>9.2f} {100*fracl:>9.2f}"
                row["loop_ms"] = 1000 * dtl
                row["loop_mfu"] = fracl
            print(line, flush=True)
            rows.append(row)
        except RuntimeError as e:
            print(f"{m:>6}  ERROR {str(e)[:60]}", flush=True)
            torch.cuda.empty_cache()

    # Find the M where grouped MFU crosses the target.
    knee = next((r for r in rows if r["grouped_mfu"] >= args.target_mfu), None)
    print("", flush=True)
    if knee:
        print(
            f"KNEE: grouped expert FFN crosses {100*args.target_mfu:.0f}% MFU at "
            f"M_per_expert={knee['m_per_expert']} -> {knee['tokens_per_rank']} real tokens/rank.",
            flush=True,
        )
    else:
        print(f"No swept M reached target MFU {100*args.target_mfu:.0f}%.", flush=True)
    op = next((r for r in rows if r["m_per_expert"] == 72), None)
    if op:
        print(
            f"OPD operating point (~2.3k tok/rank, M~72): grouped MFU={100*op['grouped_mfu']:.2f}% "
            f"-- this is the small-GEMM floor the stack runs at.",
            flush=True,
        )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": {"H": H, "I": I, "E": E, "top_k": TOP_K, "ep": args.ep,
                                  "local_experts": n_local, "dtype": args.dtype,
                                  "peak_tflops": PEAK_TFLOPS}, "rows": rows}, f, indent=2)
        print(f"wrote {args.output_json}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
