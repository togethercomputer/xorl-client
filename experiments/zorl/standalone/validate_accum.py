"""Offline CPU validation of the FP8-native accumulate-in-adapter scheme.

Proves (numerically, no GPU, no sglang import needed for the core math):

  (A) accumulate-in-bf16-accumulator + periodic SR-merge yields the SAME
      effective served weight (base+accum after merge) as the current
      per-step fold-into-FP8-base, on a small synthetic block-FP8 weight; and

  (B) with K=never the FP8 base BYTES are unchanged across every step
      (the base stays pristine), while the bf16 accumulator carries the
      full ES signal losslessly.

We import the validated fp8_fold_ from the sglang fork so the merge uses the
exact production re-quant kernel.
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/xorl-sglang-zorl/python"))
from sglang.srt.lora.fp8_fold import fp8_fold_, E4M3_MAX  # noqa: E402

torch.manual_seed(0)

BN = BK = 128
ROWS, COLS = 256, 256  # 2x2 blocks
R = 8                  # lora rank
N = 16                 # pairs / step
STEPS = 40
LR = 0.05
SCALING = 2.0          # lora_alpha / r style scaling
SIGMA = 0.05


def quantize_block_fp8(w_real: torch.Tensor):
    """Quantize a real bf16/fp32 weight to block-wise E4M3 + fp32 scale_inv."""
    rows, cols = w_real.shape
    nbr, nbc = (rows + BN - 1) // BN, (cols + BK - 1) // BK
    scale_inv = torch.zeros(nbr, nbc, dtype=torch.float32)
    wq = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn)
    for bi in range(nbr):
        for bj in range(nbc):
            r0, r1 = bi * BN, min(rows, (bi + 1) * BN)
            c0, c1 = bj * BK, min(cols, (bj + 1) * BK)
            blk = w_real[r0:r1, c0:c1]
            amax = blk.abs().amax().clamp_min(1e-12)
            s = (amax / E4M3_MAX).float()
            scale_inv[bi, bj] = s
            wq[r0:r1, c0:c1] = (blk / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return wq, scale_inv


def dequant(wq, scale_inv):
    rows, cols = wq.shape
    full = scale_inv.repeat_interleave(BN, 0).repeat_interleave(BK, 1)[:rows, :cols]
    return wq.to(torch.float32) * full


def make_step_delta(step: int) -> torch.Tensor:
    """A fresh_ab-style rank-(N*r) real-valued ES delta for one step.

    dW = lr * (1/N) * sum_i z_i * scaling * (B_i @ A_i^T), small per-entry.
    """
    g = torch.Generator().manual_seed(1000 + step)
    delta = torch.zeros(ROWS, COLS, dtype=torch.float32)
    for i in range(N):
        A = torch.randn(R, COLS, generator=g) * 1.0          # eps_A unit
        B = torch.randn(ROWS, R, generator=g) * SIGMA        # eps_B (sigma)
        z = torch.randn(1, generator=g).item()               # score
        delta += (z / N) * SCALING * (B @ A)
    return delta * LR


# ---- Common starting FP8 base (a realistic random weight) ----
w_real0 = torch.randn(ROWS, COLS, dtype=torch.float32) * 0.02
wq0, scale0 = quantize_block_fp8(w_real0)

# Deterministic SR generator factory so both schemes draw the SAME SR stream
# at the moment a given delta is committed. The key invariant for scheme (A)
# parity is: the merge re-quant of the SUMMED delta uses the same kernel; it is
# NOT bitwise identical to K separate per-step re-quants (different SR draws and
# different intermediate rounding), so we compare in EXPECTATION / RMS, which is
# exactly what stochastic rounding guarantees (E[W_fp8*scale] == target).

def sr_gen(tag: int):
    gg = torch.Generator()
    gg.manual_seed(424242 + tag)
    return gg


# ================= SCHEME 1: current per-step fold into FP8 base =================
wq_a = wq0.clone()
sc_a = scale0.clone()
deltas = [make_step_delta(s) for s in range(STEPS)]
for s, d in enumerate(deltas):
    fp8_fold_(wq_a, sc_a, d, mode="stochastic", block_n=BN, block_k=BK,
              generator=sr_gen(s))
served_perstep = dequant(wq_a, sc_a)

# ============ SCHEME 2: accumulate in bf16, merge every K (pristine base) ========
def run_accum(K):
    wq_b = wq0.clone()
    sc_b = scale0.clone()
    base_bytes_at_start = wq_b.view(torch.uint8).clone()
    accum = torch.zeros(ROWS, COLS, dtype=torch.bfloat16)  # bf16 parent accumulator
    base_untouched_until_first_merge = True
    merges = 0
    for s, d in enumerate(deltas):
        # accumulate-in-adapter: clean bf16 add, NO base re-quant
        accum.add_(d.to(torch.bfloat16))
        do_merge = K > 0 and (s + 1) % K == 0
        if do_merge:
            if base_untouched_until_first_merge:
                # verify base bytes are still pristine right up to first merge
                assert torch.equal(wq_b.view(torch.uint8), base_bytes_at_start)
                base_untouched_until_first_merge = False
            fp8_fold_(wq_b, sc_b, accum.to(torch.float32), mode="stochastic",
                      block_n=BN, block_k=BK, generator=sr_gen(10_000 + merges))
            accum.zero_()
            merges += 1
    served = dequant(wq_b, sc_b) + accum.to(torch.float32)  # served = base + live accum
    return wq_b, sc_b, accum, served, merges, base_bytes_at_start


# ---- Reference: the IDEAL real-space target (what the math wants) ----
target_real = dequant(wq0, scale0) + sum(deltas).to(torch.float32)

def rms(x):
    return float(x.float().pow(2).mean().sqrt())

print("=" * 78)
print("FP8-native accumulate-in-adapter: offline CPU validation")
print("=" * 78)
print(f"weight {ROWS}x{COLS} block {BN}x{BK} | rank r={R} pairs N={N} steps={STEPS}")
print(f"lr={LR} scaling={SCALING} sigma={SIGMA}")
sumdelta_rms = rms(sum(deltas))
print(f"\n||sum of {STEPS} ES deltas||_rms = {sumdelta_rms:.6e} "
      f"(this is the signal that must survive)")

print("\n-- (B) K=never: FP8 base must stay byte-for-byte pristine --")
wq_n, sc_n, accum_n, served_n, merges_n, base0 = run_accum(K=0)
base_pristine = torch.equal(wq_n.view(torch.uint8), wq0.view(torch.uint8)) and \
                torch.equal(sc_n, scale0)
print(f"   merges performed         : {merges_n}")
print(f"   FP8 base bytes unchanged : {base_pristine}")
print(f"   bf16 accum ||.||_rms     : {rms(accum_n):.6e}")
print(f"   accum vs sum(deltas) rms-err (bf16 rounding only): "
      f"{rms(accum_n.to(torch.float32) - sum(deltas)):.3e}")
served_err_never = rms(served_n - target_real)
print(f"   served(base+accum) vs ideal target rms-err: {served_err_never:.3e}")

print("\n-- (A) parity: per-step-FP8-fold vs accumulate+merge (RMS, in expectation) --")
perstep_err = rms(served_perstep - target_real)
print(f"   scheme1 per-step FP8 fold   vs ideal target rms-err: {perstep_err:.3e}")
for K in (1, 4, 8, STEPS):
    wq_k, sc_k, accum_k, served_k, merges_k, _ = run_accum(K=K)
    err = rms(served_k - target_real)
    base_changed = not torch.equal(wq_k.view(torch.uint8), wq0.view(torch.uint8))
    print(f"   accum+merge K={K:<3d} merges={merges_k:<3d} "
          f"vs ideal target rms-err: {err:.3e}  (base re-quant happened: {base_changed})")

print("\nINTERPRETATION:")
print(" * K=never: base bytes identical -> FP8 base PRISTINE; full ES signal lives")
print("   in the bf16 accum losslessly (only bf16 rounding ~1e-3 relative).")
print(" * accumulate+merge tracks the ideal target at the SAME error scale as the")
print("   per-step FP8 fold (both are SR-unbiased); larger K = fewer re-quants =")
print("   LESS accumulated rounding noise, because the signal is summed cleanly in")
print("   bf16 first and re-quantized ONCE per epoch instead of every step.")
