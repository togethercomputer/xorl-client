#!/usr/bin/env python3
"""1-GPU MoE expert-FFN GEMM microbench for Qwen3.6-35B-A3B (OPD throughput).

GEMM-size probe for the OPD/prefill stack. The student is an A3B MoE:
hidden=2048, moe_intermediate=512, 256 experts, top-8. With balanced routing
each expert sees

    M_per_expert = tokens_in_EP_group * top_k / num_experts

tokens, i.e. a [M, 2048] @ [2048, 512] GEMM. This script sweeps M and reports
achieved TFLOP/s and MFU for the SwiGLU expert FFN (gate/up/down) fwd+bwd, as a
batched grouped GEMM (bmm ceiling), torch._grouped_mm (engine `native`
primitive), and a per-expert python loop (launch-overhead floor).

KEY RESULT (and a correction to a first-pass conclusion): with expert
parallelism the all-to-all gathers the WHOLE EP group's tokens before the expert
GEMM, so M depends on the global batch, not per-rank tokens. At the real 1-node
config (expert_parallel_size=8, full ~71804-token OPRD batch) M ≈ 2244 →
**~45% MFU** — the MoE expert GEMM is NOT the bottleneck. M≈72 (~12% MFU) is the
EP=1 regime only. The grouped kernel matches the bmm ceiling within ~1pp, so
there is no MoE-GEMM win to chase at EP=8; the ~1% executed MFU is a phase-mix /
dummy-rank-occupancy problem (teacher fwd + KL + clear-grad + comms), not GEMM
size. See THROUGHPUT_MICROBENCH_RUNBOOK.md.

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
IDIM = 512  # moe_intermediate_size (per expert)
E = 256  # num_experts
TOP_K = 8  # num_experts_per_tok
L = 40  # num_hidden_layers
# H100 SXM bf16 dense tensor-core peak (TFLOP/s).
PEAK_TFLOPS = 989.0


def _expert_ffn_flops(m_per_expert: int, n_experts: int) -> float:
    """fwd+bwd FLOPs for SwiGLU FFN over n_experts each seeing m tokens.

    gate[H->IDIM] + up[H->IDIM] + down[IDIM->H] = 6*M*H*IDIM fwd; bwd ~= 2x fwd.
    """
    fwd = 6.0 * m_per_expert * H * IDIM * n_experts
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
    wg = torch.randn(n_experts, H, IDIM, device=dev, dtype=dtype, requires_grad=True)
    wu = torch.randn(n_experts, H, IDIM, device=dev, dtype=dtype, requires_grad=True)
    wd = torch.randn(n_experts, IDIM, H, device=dev, dtype=dtype, requires_grad=True)

    def run():
        g = torch.bmm(x, wg)
        u = torch.bmm(x, wu)
        h = torch.nn.functional.silu(g) * u
        out = torch.bmm(h, wd)
        out.sum().backward()
        x.grad = wg.grad = wu.grad = wd.grad = None

    return _time(run, iters)


def bench_grouped_mm(m: int, n_experts: int, iters: int, dtype) -> float:
    """Production primitive: torch._grouped_mm (cuBLAS/CUTLASS), as the engine's
    `native` MoE backend uses it. Uniform M per expert (ragged offs, contiguous).
    """
    dev = "cuda"
    total = m * n_experts
    x = torch.randn(total, H, device=dev, dtype=dtype, requires_grad=True)
    gate_up = torch.randn(n_experts, H, 2 * IDIM, device=dev, dtype=dtype, requires_grad=True)
    down = torch.randn(n_experts, IDIM, H, device=dev, dtype=dtype, requires_grad=True)
    offs = (torch.arange(1, n_experts + 1, device=dev, dtype=torch.int32) * m)

    def run():
        gu = torch._grouped_mm(x, gate_up, offs=offs)
        g, u = gu.chunk(2, dim=-1)
        h = (torch.nn.functional.silu(g) * u).contiguous()
        out = torch._grouped_mm(h, down, offs=offs)
        # grouped_mm backward rejects the 0-stride grad from .sum(); use a
        # contiguous grad_output instead.
        out.backward(torch.ones_like(out))
        x.grad = gate_up.grad = down.grad = None

    return _time(run, iters)


def bench_loop(m: int, n_experts: int, iters: int, dtype) -> float:
    """Per-expert python loop -- the launch-overhead floor."""
    dev = "cuda"
    xs = [torch.randn(m, H, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wg = [torch.randn(H, IDIM, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wu = [torch.randn(H, IDIM, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]
    wd = [torch.randn(IDIM, H, device=dev, dtype=dtype, requires_grad=True) for _ in range(n_experts)]

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
    ap.add_argument("--grouped-mm", action="store_true",
                    help="also run torch._grouped_mm (the engine's native production primitive)")
    ap.add_argument("--target-mfu", type=float, default=0.10)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    n_local = E // args.ep
    print(
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} peak={PEAK_TFLOPS}TF | "
        f"Qwen3.6-A3B MoE H={H} IDIM={IDIM} E={E} top_k={TOP_K} | EP={args.ep} local_experts={n_local}",
        flush=True,
    )
    # M_per_expert = (tokens in the EP group) * top_k / E. With expert parallelism
    # the all-to-all gathers ALL ep ranks' tokens before the expert GEMM, so for a
    # fixed per-rank token count M grows with ep:
    #   tokens_per_rank = M * E / (top_k * ep);  ep_group_tokens = M * E / top_k.
    # ep=1 (pure DP) is the small-M regime; ep=8 (1 node) makes M ~ep*larger.
    tok_per_rank_factor = E / (TOP_K * args.ep)
    print(
        f"M_per_expert <-> tokens: ep_group_tokens = M * {E // TOP_K}; "
        f"per-rank tokens = M * {tok_per_rank_factor:g} (EP={args.ep})\n",
        flush=True,
    )

    rows = []
    header = f"{'M/exp':>6} {'tok/rank':>9} {'bmm ms':>9} {'bmmMFU%':>8}"
    if args.grouped_mm:
        header += f" {'gmm ms':>8} {'gmmMFU%':>8}"
    if args.loop:
        header += f" {'loop ms':>9} {'loopMFU%':>9}"
    print(header, flush=True)

    for m in [8, 16, 32, 64, 72, 128, 256, 512, 1024, 2048, 4096]:
        try:
            dt = bench_grouped_bmm(m, n_local, args.iters, dtype)
            tflops, frac = mfu(m, n_local, dt)
            tokens = int(round(m * tok_per_rank_factor))  # per-rank tokens at this EP
            line = f"{m:>6} {tokens:>9} {1000*dt:>9.3f} {100*frac:>8.2f}"
            row = {"m_per_expert": m, "tokens_per_rank": tokens, "grouped_ms": 1000 * dt,
                   "grouped_tflops": tflops, "grouped_mfu": frac}
            if args.grouped_mm:
                dtg = bench_grouped_mm(m, n_local, args.iters, dtype)
                tflopsg, fracg = mfu(m, n_local, dtg)
                line += f" {1000*dtg:>8.3f} {100*fracg:>8.2f}"
                row["grouped_mm_ms"] = 1000 * dtg
                row["grouped_mm_mfu"] = fracg
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
            f"M_per_expert={knee['m_per_expert']} ({knee['tokens_per_rank']} tokens/rank at EP={args.ep}).",
            flush=True,
        )
    else:
        print(f"No swept M reached target MFU {100*args.target_mfu:.0f}%.", flush=True)
    # Real OPD operating point: the full 64-sample OPRD batch is ~71804 real
    # student tokens; with expert parallelism the whole EP group's tokens are
    # gathered before the expert GEMM, so M ~= ep_group_tokens * top_k / E.
    op_group_tokens = 71804
    op_m = op_group_tokens * TOP_K / E
    nearest = min(rows, key=lambda r: abs(r["m_per_expert"] - op_m)) if rows else None
    print(
        f"OPD operating point: full OPRD batch ~{op_group_tokens} tokens; at EP={args.ep} the "
        f"EP group gathers them all -> M_per_expert ~= {op_m:.0f}.",
        flush=True,
    )
    if nearest:
        print(
            f"  nearest swept M={nearest['m_per_expert']} -> grouped MFU "
            f"~{100*nearest['grouped_mfu']:.0f}%. So the MoE expert GEMM is only the "
            f"bottleneck when M is small (EP=1 / sparse per-rank batches), NOT at EP=8.",
            flush=True,
        )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": {"H": H, "IDIM": IDIM, "E": E, "top_k": TOP_K, "ep": args.ep,
                                  "local_experts": n_local, "dtype": args.dtype,
                                  "peak_tflops": PEAK_TFLOPS}, "rows": rows}, f, indent=2)
        print(f"wrote {args.output_json}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
