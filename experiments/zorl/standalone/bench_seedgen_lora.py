#!/usr/bin/env python
"""Prototype + microbench: seed-materialized (in-kernel) fresh_ab LoRA tiles.

Question: for ZORL fresh_ab candidates (pure seeded Gaussian A/B), can the
multi-LoRA kernels *generate* their A/B tiles in-registers from a counter-based
RNG instead of loading them from the HBM LoRA pool — and at which population
sizes is that faster?

This emulates the two population-scaled stages of the MoE LoRA delta at
Qwen3.6-35B-A3B geometry (the per-expert tiles; the expert-shared outer factors
are per-lora and small):

  EXPAND  gate_up per-expert B tile [OUT=1024, R=16]:  out[t] += h[t] @ B^T
  SHRINK  down    per-expert A tile [R=16, IN=512]:    h[t]   = x[t] @ A^T

Tokens are grouped moe_align-style into BLOCK_M blocks per (lora, expert) pair;
each block consumes its pair's full tile — exactly the unit of traffic the
question is about.

Variants (same block schedule, same bf16 tile values, fp32 accum):
  LOAD    tl.load from a [L*E, ...] bf16 pool (what csgmv/virtual-experts do)
  GEN     tl.randn per element (mode 1)
  GEN4X   tl.randn4x — one Philox call per 4 values (mode 2)
  RAD4X   Rademacher ±1 from randint4x sign bits (mode 3)
  GFAST4X gaussian via fast-math Box-Muller MUFU intrinsics (mode 4)

Seed-spec v2 candidate mapping (what the PS fold would also implement):
  seed(l, e, module) = splitmix32(candidate_seed_l, module_id, e)   [host, int32]
  plain: tile[k] = f(seed, k)                 k = row-major tile offset
  4x:    tile[k] = f4x(seed, k // 4)[k % 4]
The fill kernel materializes a pool with the SAME device mapping (this is the
"PS side"). Parity gates: (1) tile VALUES are bit-identical across chunkings
(the seed-transport contract); (2) GEMM outputs are bit-equal where the dot
lowering matches, else <=1-ULP ties on <0.01% of elements.

VERDICT (H100, triton 3.5.1, 2026-07-02): in-kernel generation is NOT faster
at any population size on kernel time — gaussian ~0.5-0.8x of LOAD, Rademacher
0.75-1.03x — HBM delivers bf16 tiles faster than SMs can Philox them, and the
ratio holds under a memory-bound overlap proxy too. The seeded path's real
wins are structural: no pool (0.64 GB/candidate at this geometry), no
max_loras_per_batch cap, no fill/eviction churn (fill measured ~2.5 ms per
module-layer at L=256).

Run (any idle H100):
  CUDA_VISIBLE_DEVICES=7 /home/apanda/xorl-sglang-zorl/.venv/bin/python \
      experiments/zorl/standalone/bench_seedgen_lora.py
"""

import argparse
import json
import math
import time

import torch
import triton
import triton.language as tl
import triton.testing
from triton.language.extra import libdevice

# ---------------------------------------------------------------- geometry
E = 256          # experts per layer
TOPK = 8
R = 16           # LoRA rank
GATEUP_OUT = 1024  # 2 * moe_intermediate (512)
DOWN_IN = 512      # moe_intermediate
SIGMA = 1.5e-4   # live b_sigma; scalar at materialization (sign folds in too)


# ------------------------------------------------------------- generator
@triton.jit
def _gen_tile(seed, row_starts, col_start, NC: tl.constexpr,
              GMODE: tl.constexpr, NROUNDS: tl.constexpr):
    """fp32 noise tile chunk; element (i, c) has tile offset
    row_starts[i] + col_start + c, c in [0, NC).

    GMODE 0: gaussian, one Philox counter per element (ctr = offset).
    GMODE 1: gaussian, 4 values per counter (ctr = offset//4, lane = offset%4);
             requires row_starts/col_start multiples of 4 and NC % 4 == 0.
    GMODE 2: Rademacher ±1 (sign bit of randint4x), same 4x-packed mapping.
    GMODE 3: gaussian via fast-math Box-Muller (MUFU intrinsics), 4x-packed.
             ~2^-21 rel err vs libm — irrelevant for ES noise; bit-reproducible
             because both ends run this same function.
    Each GMODE is a *different* seed-spec mapping; the PS-side fill must use
    the same one.
    """
    if GMODE == 0:
        c = col_start + tl.arange(0, NC)
        vals = tl.randn(seed, row_starts[:, None] + c[None, :], NROUNDS)
    else:
        q = col_start // 4 + tl.arange(0, NC // 4)
        ctr = row_starts[:, None] // 4 + q[None, :]
        if GMODE == 1:
            r0, r1, r2, r3 = tl.randn4x(seed, ctr, NROUNDS)
        elif GMODE == 2:
            i0, i1, i2, i3 = tl.randint4x(seed, ctr, NROUNDS)
            r0 = tl.where(i0 >= 0, 1.0, -1.0)
            r1 = tl.where(i1 >= 0, 1.0, -1.0)
            r2 = tl.where(i2 >= 0, 1.0, -1.0)
            r3 = tl.where(i3 >= 0, 1.0, -1.0)
        else:
            i0, i1, i2, i3 = tl.randint4x(seed, ctr, NROUNDS)
            u0 = tl.uint_to_uniform_float(i0)
            u1 = tl.uint_to_uniform_float(i1)
            u2 = tl.uint_to_uniform_float(i2)
            u3 = tl.uint_to_uniform_float(i3)
            TWO_PI: tl.constexpr = 6.2831853071795864
            m0 = tl.sqrt(-2.0 * libdevice.fast_logf(u0))
            m2 = tl.sqrt(-2.0 * libdevice.fast_logf(u2))
            r0 = m0 * libdevice.fast_cosf(TWO_PI * u1)
            r1 = m0 * libdevice.fast_sinf(TWO_PI * u1)
            r2 = m2 * libdevice.fast_cosf(TWO_PI * u3)
            r3 = m2 * libdevice.fast_sinf(TWO_PI * u3)
        vals = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))
    return vals


@triton.jit
def _fill_pool_kernel(
    pool_ptr, seed_ptr, sigma,
    NROW: tl.constexpr, NCOL: tl.constexpr,
    BR: tl.constexpr, BC: tl.constexpr, GMODE: tl.constexpr, NROUNDS: tl.constexpr,
):
    """Materialize pool[le] tiles from seeds with the same mapping (= PS side)."""
    le = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    seed = tl.load(seed_ptr + le)
    rows = pid_r * BR + tl.arange(0, BR)
    vals = _gen_tile(seed, rows * NCOL, pid_c * BC, BC, GMODE, NROUNDS) * sigma
    offs = le * NROW * NCOL + rows[:, None] * NCOL + pid_c * BC + tl.arange(0, BC)[None, :]
    tl.store(pool_ptr + offs, vals.to(tl.bfloat16))


# ------------------------------------------------------------- GEMM kernels
@triton.jit
def _expand_kernel(
    h_ptr, out_ptr, pool_ptr, seed_ptr, sorted_ids_ptr, block_le_ptr,
    D, sigma,
    OUT: tl.constexpr, RANK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    GEN: tl.constexpr, GMODE: tl.constexpr, NROUNDS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    le = tl.load(block_le_ptr + pid_m)
    tok = tl.load(sorted_ids_ptr + pid_m * BM + tl.arange(0, BM))
    mask_m = tok < D
    r = tl.arange(0, RANK)
    h = tl.load(h_ptr + tok[:, None] * RANK + r[None, :],
                mask=mask_m[:, None], other=0.0)                    # [BM, R] bf16
    n = pid_n * BN + tl.arange(0, BN)
    if GEN:
        seed = tl.load(seed_ptr + le)
        b = (_gen_tile(seed, n * RANK, 0, RANK, GMODE, NROUNDS) * sigma).to(tl.bfloat16)
    else:
        b = tl.load(pool_ptr + le * OUT * RANK + n[:, None] * RANK + r[None, :])
    acc = tl.dot(h, tl.trans(b))                                    # [BM, BN] fp32
    tl.store(out_ptr + tok[:, None] * OUT + n[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None])


@triton.jit
def _shrink_kernel(
    x_ptr, h_ptr, pool_ptr, seed_ptr, sorted_ids_ptr, block_le_ptr,
    D, sigma,
    IN: tl.constexpr, RANK: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    GEN: tl.constexpr, GMODE: tl.constexpr, NROUNDS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    le = tl.load(block_le_ptr + pid_m)
    tok = tl.load(sorted_ids_ptr + pid_m * BM + tl.arange(0, BM))
    mask_m = tok < D
    r = tl.arange(0, RANK)
    seed = 0
    if GEN:
        seed = tl.load(seed_ptr + le)
    acc = tl.zeros((BM, RANK), dtype=tl.float32)
    for k0 in range(0, IN, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + tok[:, None] * IN + k[None, :],
                    mask=mask_m[:, None], other=0.0)                # [BM, BK] bf16
        if GEN:
            a = (_gen_tile(seed, r * IN, k0, BK, GMODE, NROUNDS) * sigma).to(tl.bfloat16)
        else:
            a = tl.load(pool_ptr + le * RANK * IN + r[:, None] * IN + k[None, :])
        acc += tl.dot(x, tl.trans(a))                               # [BM, R]
    tl.store(h_ptr + tok[:, None] * RANK + r[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None])


# ------------------------------------------------------------- raw probe
@triton.jit
def _probe_kernel(src_ptr, out_ptr, sigma,
                  BLOCK: tl.constexpr, GEN: tl.constexpr, GMODE: tl.constexpr,
                  NROUNDS: tl.constexpr):
    pid = tl.program_id(0)
    if GEN:
        rows = pid * BLOCK + tl.arange(0, 1) * 0  # single "row" at block start
        v = _gen_tile(1234, rows * 0 + pid * BLOCK, 0, BLOCK, GMODE, NROUNDS) * sigma
        tl.store(out_ptr + pid, tl.sum(v))
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src_ptr + offs).to(tl.float32)
        tl.store(out_ptr + pid, tl.sum(v))


# ------------------------------------------------------------- host side
def splitmix32(x: torch.Tensor) -> torch.Tensor:
    """Deterministic int64->int31 mix (the host half of the v2 seed spec)."""
    x = (x.to(torch.int64) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFF
    return ((x ^ (x >> 31)) & 0x7FFFFFFF).to(torch.int32)


def make_seed_table(pop: int, module_id: int, device) -> torch.Tensor:
    cand = torch.arange(pop, dtype=torch.int64, device=device)
    exp = torch.arange(E, dtype=torch.int64, device=device)
    key = (cand[:, None] * (E * 8) + exp[None, :] * 8 + module_id)
    return splitmix32(key).reshape(-1).contiguous()   # [pop*E] int32


def build_dispatch(m_tokens: int, pop: int, bm: int, device, gen: torch.Generator):
    """moe_align-style grouping: dispatched rows sorted by (lora, expert),
    padded per group to BLOCK_M. Returns (sorted_ids, block_le, D, n_pairs)."""
    lora = torch.arange(m_tokens, device=device, dtype=torch.int64) % pop
    scores = torch.rand(m_tokens, E, device=device, generator=gen)
    experts = scores.topk(TOPK, dim=-1).indices                  # [M, TOPK]
    le = (lora.repeat_interleave(TOPK) * E + experts.reshape(-1))  # [D]
    D = le.numel()
    order = torch.argsort(le, stable=True)
    # rows are unique dispatched-row ids (like the real dispatch buffer), so
    # blocks never store to the same row — keeps both variants race-free.
    le_sorted, tok_sorted = le[order], order
    uniq, counts = torch.unique_consecutive(le_sorted, return_counts=True)
    blocks_per = torch.ceil(counts / bm).to(torch.int64)
    n_blocks = int(blocks_per.sum())
    sorted_ids = torch.full((n_blocks * bm,), D, dtype=torch.int32, device=device)
    block_le = torch.zeros(n_blocks, dtype=torch.int32, device=device)
    # scatter each group's rows into its padded span
    grp_starts = torch.cumsum(counts, 0) - counts                 # in sorted rows
    blk_starts = torch.cumsum(blocks_per, 0) - blocks_per         # in blocks
    pos_in_grp = torch.arange(D, device=device) - grp_starts.repeat_interleave(counts)
    dst = (blk_starts.repeat_interleave(counts) * bm + pos_in_grp)
    sorted_ids[dst] = tok_sorted.to(torch.int32)
    blk_le = uniq.repeat_interleave(blocks_per).to(torch.int32)
    block_le[: blk_le.numel()] = blk_le
    return sorted_ids, block_le, D, int(uniq.numel()), n_blocks


def run_expand(h, out, pool, seeds, sorted_ids, block_le, D, mode, bm, bn,
               nrounds=10):
    n_blocks = block_le.numel()
    grid = (n_blocks, GATEUP_OUT // bn)
    _expand_kernel[grid](
        h, out, pool, seeds, sorted_ids, block_le, D, SIGMA,
        OUT=GATEUP_OUT, RANK=R, BM=bm, BN=bn,
        GEN=mode > 0, GMODE=max(mode - 1, 0), NROUNDS=nrounds,
    )


def run_shrink(x, h, pool, seeds, sorted_ids, block_le, D, mode, bm, bk,
               nrounds=10):
    grid = (block_le.numel(),)
    _shrink_kernel[grid](
        x, h, pool, seeds, sorted_ids, block_le, D, SIGMA,
        IN=DOWN_IN, RANK=R, BM=bm, BK=bk,
        GEN=mode > 0, GMODE=max(mode - 1, 0), NROUNDS=nrounds,
    )


def fill_pool(pool, seeds, nrow, ncol, gmode, br=None, bc=None, nrounds=10):
    br = br or min(nrow, 64)
    bc = bc or min(ncol, 128)
    grid = (seeds.numel(), nrow // br, ncol // bc)
    _fill_pool_kernel[grid](pool, seeds, SIGMA, NROW=nrow, NCOL=ncol,
                            BR=br, BC=bc, GMODE=gmode, NROUNDS=nrounds)


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--tokens", default="256,1024")
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    rng = torch.Generator(device=device)

    # ---------------- parity gates
    # Gate 1 (the seed-transport contract): the generator mapping is a pure
    # function of (seed, tile offset) — materializing with different block
    # chunkings (= what the PS-side fill vs the in-GEMM generation do) must
    # produce bit-identical tile VALUES.
    # Gate 2: LOAD(filled pool) vs GEN GEMM outputs. Bit-equal when the
    # compiled dot lowering matches (plain mode here); the 4x mapping compiles
    # to a different accumulation order, so allow <=1-ULP ties on a tiny
    # fraction of elements (serve-side bf16 eval noise; the fold is exact fp32
    # on the PS regardless).
    print("== parity gates (pop=8, tokens=256) ==")
    rng.manual_seed(0)
    sorted_ids, block_le, D, n_pairs, _ = build_dispatch(256, 8, args.bm, device, rng)
    h_in = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
    x_in = torch.randn(D + 1, DOWN_IN, device=device, generator=rng).bfloat16()
    for gen_mode, mname in ((1, "gen"), (2, "gen4x"), (3, "rad4x"), (4, "gfast4x")):
        gmode = gen_mode - 1
        seeds_b = make_seed_table(8, 0, device)
        seeds_a = make_seed_table(8, 1, device)
        pool_b = torch.empty(8 * E, GATEUP_OUT, R, device=device, dtype=torch.bfloat16)
        pool_a = torch.empty(8 * E, R, DOWN_IN, device=device, dtype=torch.bfloat16)
        fill_pool(pool_b.view(8 * E, -1), seeds_b, GATEUP_OUT, R, gmode)
        fill_pool(pool_a.view(8 * E, -1), seeds_a, R, DOWN_IN, gmode)
        # gate 1: refill with different chunking, must be bit-identical
        alt_b = torch.empty_like(pool_b)
        alt_a = torch.empty_like(pool_a)
        fill_pool(alt_b.view(8 * E, -1), seeds_b, GATEUP_OUT, R, gmode, br=128, bc=16)
        fill_pool(alt_a.view(8 * E, -1), seeds_a, R, DOWN_IN, gmode, br=16, bc=64)
        ok_vals = torch.equal(pool_b, alt_b) and torch.equal(pool_a, alt_a)
        assert ok_vals, "VALUE PARITY FAIL: generator mapping is chunking-dependent"
        del alt_b, alt_a
        outs, hs = [], []
        for mode in (0, gen_mode):
            out = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
            hh = torch.zeros(D + 1, R, device=device, dtype=torch.bfloat16)
            run_expand(h_in, out, pool_b, seeds_b, sorted_ids, block_le, D, mode, args.bm, args.bn)
            run_shrink(x_in, hh, pool_a, seeds_a, sorted_ids, block_le, D, mode, args.bm, args.bk)
            outs.append(out); hs.append(hh)
        stats = []
        for load_t, gen_t in ((outs[0], outs[1]), (hs[0], hs[1])):
            d = (load_t.float() - gen_t.float()).abs()
            nd = int((d > 0).sum())
            rel = float((d / load_t.float().abs().clamp_min(1e-12))[d > 0].max()) if nd else 0.0
            stats.append((nd, d.numel(), rel))
        nz = outs[1].abs().sum().item()
        print(f"  {mname}: tile values bit-equal=True; gemm diffs "
              f"expand {stats[0][0]}/{stats[0][1]} shrink {stats[1][0]}/{stats[1][1]} "
              f"(max rel {max(s[2] for s in stats):.1e}, out |sum|={nz:.3e})")
        assert nz > 0
        for nd, n_total, rel in stats:
            assert nd <= n_total * 1e-3 and rel < 5e-2, "GEMM PARITY FAIL beyond ULP ties"
        del pool_b, pool_a

    # ---------------- raw probe: pure gen throughput vs pure HBM read
    print("== raw probe (read-only vs generate, 2^28 values) ==")
    n = 1 << 28
    block = 1024
    src = torch.randn(n, device=device).bfloat16()
    part = torch.empty(n // block, device=device, dtype=torch.float32)
    probe = {}
    for name, gm, nr in (("load", 0, 10), ("gen", 1, 10), ("gen4x", 2, 10),
                         ("gen4x_r7", 2, 7), ("rad4x", 3, 10), ("rad4x_r7", 3, 7),
                         ("gfast4x", 4, 10), ("gfast4x_r7", 4, 7)):
        ms = bench(lambda gm=gm, nr=nr: _probe_kernel[(n // block,)](
            src, part, SIGMA, BLOCK=block, GEN=gm > 0, GMODE=max(gm - 1, 0),
            NROUNDS=nr))
        rate = n / (ms * 1e-3) / 1e12
        probe[name] = (ms, rate)
        extra = f"  ({n * 2 / (ms * 1e-3) / 1e12:.2f} TB/s)" if gm == 0 else ""
        print(f"  {name:9s} {ms:7.3f} ms   {rate:6.2f} T values/s{extra}")
    del src, part

    # ---------------- population sweep
    results = []
    pops = [int(p) for p in args.pops.split(",")]
    toks = [int(t) for t in args.tokens.split(",")]
    print(f"== sweep: decode tokens x population  (BM={args.bm} BN={args.bn} BK={args.bk}) ==")
    hdr = (f"{'M':>5} {'L':>4} {'pairs':>6} {'blocks':>6} {'tileMB':>7} {'fill':>7} "
           f"{'exp_load':>9} {'exp_gf4':>8} {'exp_rad':>8} "
           f"{'shr_load':>9} {'shr_gf4':>8} {'shr_rad':>8} "
           f"{'spd(gf4)':>8} {'spd(rad)':>8}")
    print(hdr)
    for m_tokens in toks:
        for pop in pops:
            rng.manual_seed(1000 + pop)
            sorted_ids, block_le, D, n_pairs, n_blocks = build_dispatch(
                m_tokens, pop, args.bm, device, rng)
            seeds_b = make_seed_table(pop, 0, device)
            seeds_a = make_seed_table(pop, 1, device)
            pool_b = torch.empty(pop * E, GATEUP_OUT, R, device=device, dtype=torch.bfloat16)
            pool_a = torch.empty(pop * E, R, DOWN_IN, device=device, dtype=torch.bfloat16)
            fill_ms = bench(lambda: fill_pool(pool_b.view(pop * E, -1), seeds_b,
                                              GATEUP_OUT, R, 1))
            fill_pool(pool_a.view(pop * E, -1), seeds_a, R, DOWN_IN, 1)
            h_in = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
            x_in = torch.randn(D + 1, DOWN_IN, device=device, generator=rng).bfloat16()
            out = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
            hh = torch.zeros(D + 1, R, device=device, dtype=torch.bfloat16)
            row = {"tokens": m_tokens, "pop": pop, "pairs": n_pairs,
                   "blocks": n_blocks, "fill_ms": fill_ms,
                   "tile_mb": n_blocks * (GATEUP_OUT * R + R * DOWN_IN) * 2 / 1e6}
            for name, gm in (("load", 0), ("gfast4x", 4), ("rad4x", 3)):
                row[f"expand_{name}"] = bench(lambda gm=gm: run_expand(
                    h_in, out, pool_b, seeds_b, sorted_ids, block_le, D, gm,
                    args.bm, args.bn))
                row[f"shrink_{name}"] = bench(lambda gm=gm: run_shrink(
                    x_in, hh, pool_a, seeds_a, sorted_ids, block_le, D, gm,
                    args.bm, args.bk))
            base = row["expand_load"] + row["shrink_load"]
            row["speedup_gfast4x"] = base / (row["expand_gfast4x"] + row["shrink_gfast4x"])
            row["speedup_rad4x"] = base / (row["expand_rad4x"] + row["shrink_rad4x"])
            results.append(row)
            print(f"{m_tokens:>5} {pop:>4} {n_pairs:>6} {n_blocks:>6} "
                  f"{row['tile_mb']:>7.1f} {fill_ms*1e3:>6.0f}u "
                  f"{row['expand_load']*1e3:>8.1f}u {row['expand_gfast4x']*1e3:>7.1f}u "
                  f"{row['expand_rad4x']*1e3:>7.1f}u "
                  f"{row['shrink_load']*1e3:>8.1f}u {row['shrink_gfast4x']*1e3:>7.1f}u "
                  f"{row['shrink_rad4x']*1e3:>7.1f}u "
                  f"{row['speedup_gfast4x']:>7.2f}x {row['speedup_rad4x']:>7.2f}x")
            del pool_b, pool_a, h_in, x_in, out, hh
            torch.cuda.empty_cache()

    # ---------------- overlap: marginal cost next to a memory-bound base proxy
    # Real decode runs the LoRA delta alongside HBM-saturating base MoE GEMMs.
    # Measure makespan of (proxy || lora-variant) minus proxy alone: an
    # ALU-bound generator can hide under the proxy's memory-boundness, while
    # tile loads compete for the same bandwidth.
    print("== overlap vs memory-bound base proxy (805MB read/layer-ish) ==")
    proxy = torch.randn(805 * 1000 * 1000 // 2, device=device).bfloat16()
    pblock = 8192
    ppart = torch.empty(proxy.numel() // pblock, device=device, dtype=torch.float32)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def run_proxy():
        _probe_kernel[(proxy.numel() // pblock,)](
            proxy, ppart, SIGMA, BLOCK=pblock, GEN=False, GMODE=0, NROUNDS=10)

    def timed(fn, iters=100):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    overlap = {}
    for m_tokens, pop in ((1024, 8), (1024, 128)):
        rng.manual_seed(1000 + pop)
        sorted_ids, block_le, D, n_pairs, n_blocks = build_dispatch(
            m_tokens, pop, args.bm, device, rng)
        seeds_b = make_seed_table(pop, 0, device)
        seeds_a = make_seed_table(pop, 1, device)
        pool_b = torch.empty(pop * E, GATEUP_OUT, R, device=device, dtype=torch.bfloat16)
        pool_a = torch.empty(pop * E, R, DOWN_IN, device=device, dtype=torch.bfloat16)
        fill_pool(pool_b.view(pop * E, -1), seeds_b, GATEUP_OUT, R, 1)
        fill_pool(pool_a.view(pop * E, -1), seeds_a, R, DOWN_IN, 1)
        h_in = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
        x_in = torch.randn(D + 1, DOWN_IN, device=device, generator=rng).bfloat16()
        out = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
        hh = torch.zeros(D + 1, R, device=device, dtype=torch.bfloat16)

        def combo(mode):
            with torch.cuda.stream(s1):
                run_proxy()
            with torch.cuda.stream(s2):
                run_expand(h_in, out, pool_b, seeds_b, sorted_ids, block_le, D,
                           mode, args.bm, args.bn)
                run_shrink(x_in, hh, pool_a, seeds_a, sorted_ids, block_le, D,
                           mode, args.bm, args.bk)

        t_proxy = timed(run_proxy)
        res = {"proxy_alone": t_proxy}
        for name, gm in (("load", 0), ("gen4x", 2), ("gfast4x", 4), ("rad4x", 3)):
            res[name] = timed(lambda gm=gm: combo(gm)) - t_proxy
        overlap[f"M{m_tokens}_L{pop}"] = res
        print(f"  M={m_tokens} L={pop} pairs={n_pairs}: proxy {t_proxy*1e3:.0f}u; "
              f"marginal: load {res['load']*1e3:+.0f}u  gen4x {res['gen4x']*1e3:+.0f}u  "
              f"gfast4x {res['gfast4x']*1e3:+.0f}u  rad4x {res['rad4x']*1e3:+.0f}u")
        del pool_b, pool_a, h_in, x_in, out, hh
        torch.cuda.empty_cache()
    del proxy, ppart

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"probe": probe, "sweep": results, "overlap": overlap},
                      f, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
