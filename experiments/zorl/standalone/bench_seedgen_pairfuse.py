#!/usr/bin/env python
"""Steps 1-3 on top of bench_seedgen_lora.py (run that first for context):

1. ANTITHETIC PAIR-SHARING in the block sort: a +/- pair shares its noise, so
   sort dispatched rows by (pair, expert) instead of (candidate, expert) and
   ride the candidate's sign on h at the expand (sign commutes through the
   dot; A stages have no sign at all). Halves the tile working set and merges
   +/- blocks; the block-count gain is collision-limited at sparse decode and
   approaches 2x as (owner, expert) slots saturate.
2. FUSION into the adjacent base GEMM: gate_up LoRA expand fused into the
   gate_up base-weight stream (bench_seedgen_lora already models this),
   plus a fused down-base + per-expert shrink kernel here. NOTE the two
   per-expert LoRA stages live in DIFFERENT modules separated by the
   activation + base down GEMM, so they cannot chain in one launch; the
   fusion win is hiding gen under each base stream and reusing its inputs.
3. ZERO-POOL FULL LAYER: all four MoE LoRA stages (shared gate_up shrink,
   per-expert gate_up expand, per-expert down shrink, shared down expand)
   with tiles generated in-kernel — no LoRA pool bytes at all — vs the
   all-pool load baseline. A stages use fast-math gaussian (Rademacher-A is
   not science-gated); B stages use Rademacher.

Stage inputs (h_g, y, h_d) are independent random tensors — this measures
kernel cost, not a numerically-chained layer (the chain needs activation +
routing-weight modeling that doesn't change tile traffic). The full-layer
floor overstates real base traffic (a fused-by-(pair,expert) kernel
re-streams base tiles per block, L2-mitigated); the same floor sits under
both variants, so the MARGINALS are the comparison.

VERDICT (H100, 2026-07-02, C=128 candidates = 64 pairs):
- step 1 block ratio: 1.12x (M=1024) -> 1.47x (4096) -> 1.96x (16384):
  collision-limited at sparse decode, full 2x at (pair, expert) saturation.
- standalone crossover at saturation: pair-sort + generation beats today's
  candidate-sort + pool loads 1.20x (582u vs 700u at M=16384).
- full-layer (4 stages, fused base streams) delta marginal, all-gen vs
  all-load: 1.39x WORSE at M=1024, 1.11x better at 4096, 1.56x better at
  16384 — with 0 MB pool vs 814 MB/layer (39 GB/model: the load baseline
  cannot even exist at this population without seeded tiles).

Run:
  CUDA_VISIBLE_DEVICES=7 /home/apanda/xorl-sglang-zorl/.venv/bin/python \
      experiments/zorl/standalone/bench_seedgen_pairfuse.py
"""

import argparse
import json

import torch
import triton
import triton.language as tl

from bench_seedgen_lora import (  # noqa: F401
    DOWN_IN, E, GATEUP_OUT, R, SIGMA, TOPK,
    _gen_tile, bench, fill_pool, make_seed_table, run_expand, run_shrink,
    splitmix32,
)

HID = 2048  # hidden size (shared-stage K dim / down expand out dim)


# ------------------------------------------------------------- dispatch
def build_dispatch2(m_tokens, n_cands, bm, device, gen, pair_shared):
    """Like bench_seedgen_lora.build_dispatch but keyed by pair when
    pair_shared, and returns per-slot signs aligned with sorted_ids."""
    cand = torch.arange(m_tokens, device=device, dtype=torch.int64) % n_cands
    owner = cand // 2 if pair_shared else cand
    sign = (1 - 2 * (cand % 2)).to(torch.bfloat16) if pair_shared else None
    scores = torch.rand(m_tokens, E, device=device, generator=gen)
    experts = scores.topk(TOPK, dim=-1).indices
    le = owner.repeat_interleave(TOPK) * E + experts.reshape(-1)
    D = le.numel()
    order = torch.argsort(le, stable=True)
    le_sorted = le[order]
    uniq, counts = torch.unique_consecutive(le_sorted, return_counts=True)
    blocks_per = torch.ceil(counts / bm).to(torch.int64)
    n_blocks = int(blocks_per.sum())
    sorted_ids = torch.full((n_blocks * bm,), D, dtype=torch.int32, device=device)
    signs = torch.ones(n_blocks * bm, dtype=torch.bfloat16, device=device)
    grp_starts = torch.cumsum(counts, 0) - counts
    blk_starts = torch.cumsum(blocks_per, 0) - blocks_per
    pos = torch.arange(D, device=device) - grp_starts.repeat_interleave(counts)
    dst = blk_starts.repeat_interleave(counts) * bm + pos
    sorted_ids[dst] = order.to(torch.int32)
    if pair_shared:
        signs[dst] = sign.repeat_interleave(TOPK)[order]
    block_le = uniq.repeat_interleave(blocks_per).to(torch.int32)
    return sorted_ids, block_le, signs, D, int(uniq.numel()), n_blocks


def build_token_groups(m_tokens, n_cands, bm, device, pair_shared):
    """Group whole tokens (not dispatched rows) by owner for the shared
    stages. Returns (sorted_tok, block_owner, signs, n_blocks)."""
    cand = torch.arange(m_tokens, device=device, dtype=torch.int64) % n_cands
    owner = cand // 2 if pair_shared else cand
    sign = (1 - 2 * (cand % 2)).to(torch.bfloat16)
    order = torch.argsort(owner, stable=True)
    o_sorted = owner[order]
    uniq, counts = torch.unique_consecutive(o_sorted, return_counts=True)
    blocks_per = torch.ceil(counts / bm).to(torch.int64)
    n_blocks = int(blocks_per.sum())
    sorted_tok = torch.full((n_blocks * bm,), m_tokens, dtype=torch.int32,
                            device=device)
    signs = torch.ones(n_blocks * bm, dtype=torch.bfloat16, device=device)
    grp_starts = torch.cumsum(counts, 0) - counts
    blk_starts = torch.cumsum(blocks_per, 0) - blocks_per
    pos = torch.arange(m_tokens, device=device) - grp_starts.repeat_interleave(counts)
    dst = blk_starts.repeat_interleave(counts) * bm + pos
    sorted_tok[dst] = order.to(torch.int32)
    signs[dst] = sign[order]
    block_owner = uniq.repeat_interleave(blocks_per).to(torch.int32)
    return sorted_tok, block_owner, signs, n_blocks


# ------------------------------------------------------------- kernels
@triton.jit
def _fused_shrink_kernel(
    y_ptr, h_ptr, base_ptr, pool_ptr, seed_ptr, sorted_ids_ptr, block_le_ptr,
    D, sigma,
    EXP: tl.constexpr, IN: tl.constexpr, RANK: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BASE_ELEMS: tl.constexpr,
    SPLIT: tl.constexpr, LORA: tl.constexpr, GEN: tl.constexpr,
    GMODE: tl.constexpr, NROUNDS: tl.constexpr,
):
    """Down-proj side of step 2: stream this block's down base-expert chunk
    and compute the per-expert LoRA shrink h = y @ A^T in the same kernel
    (pid_k 0 does the shrink; all pid_k stream a base slice)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    le = tl.load(block_le_ptr + pid_m)
    e = (le % EXP).to(tl.int64)
    bsum = 0.0
    for i in range(0, BASE_ELEMS, 4096):
        off = i + tl.arange(0, 4096)
        v = tl.load(base_ptr + e * (BASE_ELEMS * SPLIT) + pid_k * BASE_ELEMS + off)
        bsum += tl.sum(v.to(tl.float32))
    if LORA and pid_k == 0:
        tok = tl.load(sorted_ids_ptr + pid_m * BM + tl.arange(0, BM))
        mask_m = tok < D
        r = tl.arange(0, RANK)
        seed = 0
        if GEN:
            seed = tl.load(seed_ptr + le)
        acc = tl.zeros((BM, RANK), dtype=tl.float32)
        for k0 in range(0, IN, BK):
            k = k0 + tl.arange(0, BK)
            y = tl.load(y_ptr + tok[:, None] * IN + k[None, :],
                        mask=mask_m[:, None], other=0.0)
            if GEN:
                a = (_gen_tile(seed, r * IN, k0, BK, GMODE, NROUNDS)
                     * sigma).to(tl.bfloat16)
            else:
                a = tl.load(pool_ptr + le * RANK * IN + r[:, None] * IN + k[None, :])
            acc += tl.dot(y, tl.trans(a))
        tl.store(h_ptr + tok[:, None] * RANK + r[None, :],
                 (acc + bsum * 1e-20).to(tl.bfloat16), mask=mask_m[:, None])
    else:
        tl.store(h_ptr + D * RANK + pid_m, (bsum * 1e-20).to(tl.bfloat16),
                 mask=(pid_k == 0) & (LORA == 0))


@triton.jit
def _shared_shrink_kernel(
    x_ptr, h_ptr, pool_ptr, seed_ptr, sorted_tok_ptr, block_owner_ptr,
    M, sigma,
    HIDDEN: tl.constexpr, RANK: tl.constexpr, BM: tl.constexpr,
    BK: tl.constexpr, GEN: tl.constexpr, GMODE: tl.constexpr,
    NROUNDS: tl.constexpr,
):
    """Shared (expert-independent) gate_up shrink: h_g = x @ A_g^T,
    one A_g [RANK, HIDDEN] per owner, token blocks grouped by owner."""
    pid_m = tl.program_id(0)
    owner = tl.load(block_owner_ptr + pid_m).to(tl.int64)
    tok = tl.load(sorted_tok_ptr + pid_m * BM + tl.arange(0, BM))
    mask_m = tok < M
    r = tl.arange(0, RANK)
    seed = 0
    if GEN:
        seed = tl.load(seed_ptr + owner)
    acc = tl.zeros((BM, RANK), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + tok[:, None] * HIDDEN + k[None, :],
                    mask=mask_m[:, None], other=0.0)
        if GEN:
            a = (_gen_tile(seed, r * HIDDEN, k0, BK, GMODE, NROUNDS)
                 * sigma).to(tl.bfloat16)
        else:
            a = tl.load(pool_ptr + owner * RANK * HIDDEN
                        + r[:, None] * HIDDEN + k[None, :])
        acc += tl.dot(x, tl.trans(a))
    tl.store(h_ptr + tok[:, None] * RANK + r[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None])


@triton.jit
def _shared_expand_kernel(
    h_ptr, out_ptr, pool_ptr, seed_ptr, sorted_tok_ptr, block_owner_ptr,
    sign_ptr, M, sigma,
    HIDDEN: tl.constexpr, RANK: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, GEN: tl.constexpr, GMODE: tl.constexpr,
    NROUNDS: tl.constexpr, SIGNED: tl.constexpr,
):
    """Shared down expand: out = sign * (h_d @ B_d^T), one B_d [HIDDEN, RANK]
    per owner, token blocks grouped by owner."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    owner = tl.load(block_owner_ptr + pid_m).to(tl.int64)
    tok = tl.load(sorted_tok_ptr + pid_m * BM + tl.arange(0, BM))
    mask_m = tok < M
    r = tl.arange(0, RANK)
    h = tl.load(h_ptr + tok[:, None] * RANK + r[None, :],
                mask=mask_m[:, None], other=0.0)
    if SIGNED:
        sg = tl.load(sign_ptr + pid_m * BM + tl.arange(0, BM),
                     mask=mask_m, other=1.0).to(tl.bfloat16)
        h = h * sg[:, None]
    n = pid_n * BN + tl.arange(0, BN)
    if GEN:
        seed = tl.load(seed_ptr + owner)
        b = (_gen_tile(seed, n * RANK, 0, RANK, GMODE, NROUNDS)
             * sigma).to(tl.bfloat16)
    else:
        b = tl.load(pool_ptr + owner * HIDDEN * RANK
                    + n[:, None] * RANK + r[None, :])
    acc = tl.dot(h, tl.trans(b))
    tl.store(out_ptr + tok[:, None] * HIDDEN + n[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None])


# ------------------------------------------------------------- host helpers
def make_owner_seed_table(pop, module_id, device):
    """Seeds for expert-shared tiles: one per owner (module ids 2+ so they
    never collide with the per-expert tables)."""
    key = torch.arange(pop, dtype=torch.int64, device=device) * (E * 8) + module_id
    return splitmix32(key).contiguous()


def run_fused_shrink(y, h, base, pool, seeds, sorted_ids, block_le, D, lora,
                     mode, bm, bk, base_elems, split=4):
    grid = (block_le.numel(), split)
    _fused_shrink_kernel[grid](
        y, h, base, pool, seeds, sorted_ids, block_le, D, SIGMA,
        EXP=E, IN=DOWN_IN, RANK=R, BM=bm, BK=bk, BASE_ELEMS=base_elems,
        SPLIT=split, LORA=lora, GEN=mode > 0,
        GMODE=max(mode - 1, 0), NROUNDS=10,
    )


def run_shared_shrink(x, h, pool, seeds, sorted_tok, block_owner, M, mode,
                      bm, bk):
    grid = (block_owner.numel(),)
    _shared_shrink_kernel[grid](
        x, h, pool, seeds, sorted_tok, block_owner, M, SIGMA,
        HIDDEN=HID, RANK=R, BM=bm, BK=bk,
        GEN=mode > 0, GMODE=max(mode - 1, 0), NROUNDS=10,
    )


def run_shared_expand(h, out, pool, seeds, sorted_tok, block_owner, signs, M,
                      mode, bm, bn):
    grid = (block_owner.numel(), HID // bn)
    _shared_expand_kernel[grid](
        h, out, pool, seeds, sorted_tok, block_owner,
        signs if signs is not None else h, M, SIGMA,
        HIDDEN=HID, RANK=R, BM=bm, BN=bn,
        GEN=mode > 0, GMODE=max(mode - 1, 0), NROUNDS=10,
        SIGNED=signs is not None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=128)
    ap.add_argument("--tokens", default="1024,4096,16384")
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    rng = torch.Generator(device=device)
    C = args.cands
    P = C // 2
    report = {}

    # ============ parity: candidate-sort LOAD(+/- pool) == pair-sort GEN+sign
    print("== parity: pair-shared sign vs per-candidate tiles (M=512) ==")
    rng.manual_seed(7)
    sc, blc, _sg, D, _, _ = build_dispatch2(512, C, args.bm, device, rng, False)
    rng.manual_seed(7)
    sp, blp, sgp, D2, _, _ = build_dispatch2(512, C, args.bm, device, rng, True)
    assert D == D2
    seeds_b = make_seed_table(P, 0, device)
    pair_pool = torch.empty(P * E, GATEUP_OUT, R, device=device, dtype=torch.bfloat16)
    fill_pool(pair_pool.view(P * E, -1), seeds_b, GATEUP_OUT, R, 2)  # rad4x
    cand_pool = pair_pool.view(P, E, GATEUP_OUT, R).repeat_interleave(2, dim=0)
    cand_pool[1::2].neg_()
    cand_pool = cand_pool.reshape(C * E, GATEUP_OUT, R).contiguous()
    h_in = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
    out_c = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
    out_p = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
    run_expand(h_in, out_c, cand_pool, seeds_b, sc, blc, D, 0, args.bm, args.bn)
    run_expand(h_in, out_p, pair_pool, seeds_b, sp, blp, D, 3, args.bm, args.bn,
               signs=sgp)
    ok = torch.equal(out_c, out_p)
    print(f"  candidate-sort LOAD(+/-) == pair-sort GEN(rad4x)+sign: {ok}")
    assert ok and out_p.abs().sum() > 0
    del pair_pool, cand_pool

    # ============ step 1: pair-sort vs candidate-sort, standalone kernels
    print(f"== step 1: antithetic pair-sharing (C={C} candidates = {P} pairs) ==")
    print(f"{'M':>6} {'sort':>9} {'blocks':>7} {'exp_load':>9} {'exp_rad':>8} "
          f"{'shr_load':>9} {'shr_gf4':>8} {'tot_ld':>7} {'tot_gen':>8}")
    step1 = []
    for m_tokens in [int(t) for t in args.tokens.split(",")]:
        row_m = {}
        for pair_shared in (False, True):
            rng.manual_seed(100 + m_tokens)
            si, bl, sg, D, n_grp, nb = build_dispatch2(
                m_tokens, C, args.bm, device, rng, pair_shared)
            K = P if pair_shared else C
            seeds_b = make_seed_table(K, 0, device)
            seeds_a = make_seed_table(K, 1, device)
            pool_b = torch.empty(K * E, GATEUP_OUT, R, device=device,
                                 dtype=torch.bfloat16)
            pool_a = torch.empty(K * E, R, DOWN_IN, device=device,
                                 dtype=torch.bfloat16)
            fill_pool(pool_b.view(K * E, -1), seeds_b, GATEUP_OUT, R, 2)
            fill_pool(pool_a.view(K * E, -1), seeds_a, R, DOWN_IN, 3)
            h_in = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
            y_in = torch.randn(D + 1, DOWN_IN, device=device, generator=rng).bfloat16()
            out = torch.zeros(D + 1, GATEUP_OUT, device=device, dtype=torch.bfloat16)
            hh = torch.zeros(D + 1, R, device=device, dtype=torch.bfloat16)
            signs = sg if pair_shared else None
            r = {
                "expand_load": bench(lambda: run_expand(
                    h_in, out, pool_b, seeds_b, si, bl, D, 0, args.bm, args.bn,
                    signs=signs)),
                "expand_rad4x": bench(lambda: run_expand(
                    h_in, out, pool_b, seeds_b, si, bl, D, 3, args.bm, args.bn,
                    signs=signs)),
                "shrink_load": bench(lambda: run_shrink(
                    y_in, hh, pool_a, seeds_a, si, bl, D, 0, args.bm, args.bk)),
                "shrink_gfast": bench(lambda: run_shrink(
                    y_in, hh, pool_a, seeds_a, si, bl, D, 4, args.bm, args.bk)),
                "blocks": nb,
            }
            name = "pair" if pair_shared else "candidate"
            row_m[name] = r
            tot_ld = r["expand_load"] + r["shrink_load"]
            tot_gen = r["expand_rad4x"] + r["shrink_gfast"]
            print(f"{m_tokens:>6} {name:>9} {nb:>7} "
                  f"{r['expand_load']*1e3:>8.1f}u {r['expand_rad4x']*1e3:>7.1f}u "
                  f"{r['shrink_load']*1e3:>8.1f}u {r['shrink_gfast']*1e3:>7.1f}u "
                  f"{tot_ld*1e3:>6.0f}u {tot_gen*1e3:>7.0f}u")
            del pool_b, pool_a, h_in, y_in, out, hh
            torch.cuda.empty_cache()
        c, p = row_m["candidate"], row_m["pair"]
        print(f"       -> block ratio {c['blocks']/p['blocks']:.2f}x; "
              f"pair-gen vs candidate-load speedup "
              f"{(c['expand_load']+c['shrink_load'])/(p['expand_rad4x']+p['shrink_gfast']):.2f}x")
        step1.append({"tokens": m_tokens, **{k: row_m[k] for k in row_m}})
    report["step1"] = step1

    # ============ steps 2+3: fused per-expert stages + shared stages,
    # full-layer zero-pool vs all-load (pair sort throughout)
    print(f"== steps 2+3: full-layer delta, pair sort, fused base streams ==")
    gu_elems = (2 * 1024 * 1024 // 2) // (GATEUP_OUT // args.bn)  # 2MB fp16/8
    dn_elems_total = 1 * 1024 * 1024 // 2                          # 1MB fp16
    dn_split = 4
    dn_elems = dn_elems_total // dn_split
    base_gu = torch.randn(E * (2 * 1024 * 1024 // 2), device=device).half()
    base_dn = torch.randn(E * dn_elems_total, device=device).half()
    from bench_seedgen_lora import run_fused  # noqa: E402
    step23 = []
    for m_tokens in [int(t) for t in args.tokens.split(",")]:
        rng.manual_seed(100 + m_tokens)
        si, bl, sg, D, n_grp, nb = build_dispatch2(
            m_tokens, C, args.bm, device, rng, True)
        st, bo, sgt, nbt = build_token_groups(m_tokens, C, args.bm, device, True)
        seeds_b = make_seed_table(P, 0, device)
        seeds_a = make_seed_table(P, 1, device)
        seeds_ag = make_owner_seed_table(P, 2, device)
        seeds_bd = make_owner_seed_table(P, 3, device)
        # pools (load baseline)
        pool_b = torch.empty(P * E, GATEUP_OUT, R, device=device, dtype=torch.bfloat16)
        pool_a = torch.empty(P * E, R, DOWN_IN, device=device, dtype=torch.bfloat16)
        pool_ag = torch.empty(P, R, HID, device=device, dtype=torch.bfloat16)
        pool_bd = torch.empty(P, HID, R, device=device, dtype=torch.bfloat16)
        fill_pool(pool_b.view(P * E, -1), seeds_b, GATEUP_OUT, R, 2)
        fill_pool(pool_a.view(P * E, -1), seeds_a, R, DOWN_IN, 3)
        fill_pool(pool_ag.view(P, -1), seeds_ag, R, HID, 3)
        fill_pool(pool_bd.view(P, -1), seeds_bd, HID, R, 2)
        pool_bytes = 2 * (pool_b.numel() + pool_a.numel()
                          + pool_ag.numel() + pool_bd.numel())
        # activations
        x_in = torch.randn(m_tokens + 1, HID, device=device, generator=rng).bfloat16()
        hg = torch.zeros(m_tokens + 1, R, device=device, dtype=torch.bfloat16)
        hgd = torch.randn(D + 1, R, device=device, generator=rng).bfloat16()
        y_in = torch.randn(D + 1, DOWN_IN, device=device, generator=rng).bfloat16()
        hd_flat = torch.zeros((D + 1) * R + nb + 1, device=device, dtype=torch.bfloat16)
        hd_tok = torch.randn(m_tokens + 1, R, device=device, generator=rng).bfloat16()
        out_gu = torch.zeros((D + 1) * GATEUP_OUT + nb, device=device,
                             dtype=torch.bfloat16)
        out_dn = torch.zeros(m_tokens + 1, HID, device=device, dtype=torch.bfloat16)

        def full_layer(load):
            m_a = 0 if load else 4     # A stages: load or gfast gaussian
            m_b = 0 if load else 3     # B stages: load or rademacher
            run_shared_shrink(x_in, hg, pool_ag, seeds_ag, st, bo, m_tokens,
                              m_a, args.bm, args.bk)
            run_fused(hgd, out_gu, base_gu, pool_b, seeds_b, si, bl, D, 1,
                      m_b, args.bm, args.bn, gu_elems, signs=sg)
            run_fused_shrink(y_in, hd_flat, base_dn, pool_a, seeds_a, si, bl,
                             D, 1, m_a, args.bm, args.bk, dn_elems, dn_split)
            run_shared_expand(hd_tok, out_dn, pool_bd, seeds_bd, st, bo, sgt,
                              m_tokens, m_b, args.bm, args.bn)

        t_load = bench(lambda: full_layer(True))
        t_gen = bench(lambda: full_layer(False))
        # base-stream-only floor (no lora at all) for marginals
        def floor():
            run_fused(hgd, out_gu, base_gu, pool_b, seeds_b, si, bl, D, 0,
                      0, args.bm, args.bn, gu_elems)
            run_fused_shrink(y_in, hd_flat, base_dn, pool_a, seeds_a, si, bl,
                             D, 0, 0, args.bm, args.bk, dn_elems, dn_split)
        t_floor = bench(floor)
        row = {"tokens": m_tokens, "blocks": nb, "tok_blocks": nbt,
               "pool_mb": pool_bytes / 1e6, "floor_ms": t_floor,
               "layer_load_ms": t_load, "layer_gen_ms": t_gen,
               "marginal_load_ms": t_load - t_floor,
               "marginal_gen_ms": t_gen - t_floor}
        step23.append(row)
        print(f"  M={m_tokens:>5} blocks={nb:>6}: base floor {t_floor*1e3:>5.0f}u; "
              f"full-layer delta marginal: all-load {row['marginal_load_ms']*1e3:+6.0f}u "
              f"(pool {pool_bytes/1e6:.0f} MB/layer)  "
              f"all-gen {row['marginal_gen_ms']*1e3:+6.0f}u (pool 0 MB)")
        del pool_b, pool_a, pool_ag, pool_bd, x_in, hg, hgd, y_in
        del hd_flat, hd_tok, out_gu, out_dn
        torch.cuda.empty_cache()
    report["step23"] = step23

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
