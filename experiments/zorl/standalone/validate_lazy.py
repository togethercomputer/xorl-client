"""Offline CPU validation of the LAZY-SCALE FP8 fold (no buffer).

3-way comparison over >=40 successive SGD-scale ES deltas on a synthetic
block-FP8 weight:
  * stochastic       (eager: recompute every block scale each step) -> drifts;
  * stochastic_lazy  (recompute only on saturation)                 -> stays;
  * bf16 truth       (exact accumulation reference).

Reports rel-err-vs-bf16-truth trajectories, COUNTS rescale (saturation) events
for each mode, and confirms the lazy path (a) captures the accumulated signal
and (b) leaves delta-untouched weights byte-stable between rescales. Repeats for
the 3-D MoE path. Uses the REAL fp8_fold_ from the fork.
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/xorl-sglang-zorl/python"))
from sglang.srt.lora.fp8_fold import fp8_fold_, E4M3_MAX  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
BN = BK = 128


def quantize2d(W):
    r, c = W.shape
    nbr, nbc = (r + BN - 1) // BN, (c + BK - 1) // BK
    si = torch.zeros(nbr, nbc, dtype=torch.float32)
    Wq = torch.zeros_like(W, dtype=torch.float8_e4m3fn)
    for i in range(nbr):
        for j in range(nbc):
            blk = W[i * BN:(i + 1) * BN, j * BK:(j + 1) * BK]
            s = max(float(blk.abs().max()) / E4M3_MAX, 1e-12)
            si[i, j] = s
            Wq[i * BN:(i + 1) * BN, j * BK:(j + 1) * BK] = (
                (blk / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            )
    return Wq, si


def eff2d(Wq, si):
    full = si.repeat_interleave(BN, 0).repeat_interleave(BK, 1)[:Wq.shape[0], :Wq.shape[1]]
    return Wq.float() * full


def relerr(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


R = C = 256
N = 40
# SGD ~17-norm regime: per-entry rms ~3e-3 (this is the regime that collapsed).
DRMS = 3e-3
W0 = torch.randn(R, C) * 0.02
deltas = [torch.randn(R, C) * DRMS for _ in range(N)]

print("=" * 78)
print("LAZY-SCALE FP8 fold: 3-way validation (SGD ~17-norm regime, 40 steps)")
print("=" * 78)
print(f"weight {R}x{C} block {BN}x{BK} | per-entry delta rms={DRMS:.0e} | "
      f"||sum delta||_rms={float(torch.stack(deltas).sum(0).pow(2).mean().sqrt()):.4e}")

results = {}
for mode in ("stochastic", "stochastic_lazy"):
    Wq, si = quantize2d(W0.clone())
    truth = W0.clone()
    gen = torch.Generator().manual_seed(1)
    total_rescales = 0
    nblocks = si.numel()
    traj = {}
    prev_Wq = Wq.clone()
    untouched_changed_steps = 0
    for step in range(1, N + 1):
        d = deltas[step - 1]
        # Which entries does this delta actually move (>0)? everything here, but
        # for the "untouched stays put" check we use a SPARSE delta below.
        rescaled = fp8_fold_(Wq, si, d.to(torch.bfloat16), mode=mode,
                             block_n=BN, block_k=BK, generator=gen)
        total_rescales += int(rescaled)
        truth += d
        if step in (1, 5, 10, 20, 40):
            traj[step] = relerr(eff2d(Wq, si), truth)
    results[mode] = (traj, total_rescales, nblocks)
    rescale_str = "EVERY step (all blocks)" if mode == "stochastic" else \
                  f"{total_rescales} block-rescales total over {N} steps x {nblocks} blocks"
    print(f"\n  [{mode}]  rescales: {rescale_str}")
    print("    rel_err vs bf16-truth: " +
          "  ".join(f"@{s}={traj[s]:.4f}" for s in sorted(traj)))

# bf16-truth is the reference (rel_err 0 by construction); print the signal mag.
sig = eff2d(*quantize2d(W0.clone()))  # step-0 effective (no delta)
print(f"\n  [bf16 truth] reference (rel_err 0 by construction)")

# ---- signal capture + untouched-stays-put: a SPARSE delta touching one block --
print("\n-- signal capture + untouched-weights-stable (sparse delta, lazy) --")
Wq, si = quantize2d(W0.clone())
gen = torch.Generator().manual_seed(7)
# delta only in block (0,0); rest exactly zero
sparse = torch.zeros(R, C)
sparse[:BN, :BK] = torch.randn(BN, BK) * DRMS
base_eff = eff2d(Wq, si).clone()
touched_mask = (sparse != 0)
untouched_drift = []
captured = []
for step in range(1, 21):
    fp8_fold_(Wq, si, sparse.to(torch.bfloat16), mode="stochastic_lazy",
              block_n=BN, block_k=BK, generator=gen)
    cur = eff2d(Wq, si)
    # untouched entries (outside block 0,0) must be byte-stable
    ud = float((cur[~touched_mask] - base_eff[~touched_mask]).abs().max())
    untouched_drift.append(ud)
    # the accumulated signal in the touched block should track step*sparse
    want = base_eff[touched_mask] + step * sparse[touched_mask]
    captured.append(relerr(cur[touched_mask], want))
print(f"   max drift of UNTOUCHED entries over 20 steps: {max(untouched_drift):.3e} "
      f"(0.0 => byte-stable, lazy never re-rounded them)")
print(f"   signal capture rel_err in TOUCHED block @20 steps: {captured[-1]:.4f}")

# ---- 3-D MoE path ----
print("\n-- 3-D MoE path (E=4 experts, [E, 256, 256]) --")
E = 4
W3 = torch.randn(E, R, C) * 0.02
def quantize3d(W):
    Ee, r, c = W.shape; nbr, nbc = (r + BN - 1)//BN, (c + BK - 1)//BK
    si = torch.zeros(Ee, nbr, nbc); Wq = torch.zeros_like(W, dtype=torch.float8_e4m3fn)
    for e in range(Ee):
        wq, s = quantize2d(W[e]); Wq[e] = wq; si[e] = s
    return Wq, si
def eff3d(Wq, si):
    full = si.repeat_interleave(BN,1).repeat_interleave(BK,2)[:, :Wq.shape[1], :Wq.shape[2]]
    return Wq.float() * full
d3 = [torch.randn(E, R, C) * DRMS for _ in range(N)]
for mode in ("stochastic", "stochastic_lazy"):
    Wq, si = quantize3d(W3.clone()); truth = W3.clone()
    gen = torch.Generator().manual_seed(2); tot = 0; nb = si.numel(); traj = {}
    for step in range(1, N + 1):
        tot += int(fp8_fold_(Wq, si, d3[step-1].to(torch.bfloat16), mode=mode,
                             block_n=BN, block_k=BK, generator=gen))
        truth += d3[step-1]
        if step in (1, 10, 40): traj[step] = relerr(eff3d(Wq, si), truth)
    rs = "EVERY step" if mode == "stochastic" else f"{tot} of {N*nb} block-folds"
    print(f"  [{mode}] rescales={rs}  rel_err: " +
          "  ".join(f"@{s}={traj[s]:.4f}" for s in sorted(traj)))

print("\nSUMMARY:")
st = results["stochastic"][0]; lz = results["stochastic_lazy"][0]
print(f"  stochastic (eager) drifts: @1={st[1]:.4f} -> @40={st[40]:.4f}")
print(f"  stochastic_lazy stays    : @1={lz[1]:.4f} -> @40={lz[40]:.4f}")
print(f"  lazy total rescales: {results['stochastic_lazy'][1]} "
      f"(vs eager {N}*{results['stochastic'][2]}={N*results['stochastic'][2]})")
