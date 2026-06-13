# FP8 base + bf16 LoRA for fresh_ab ES — design & implementation

**Status:** IMPLEMENTED then ABANDONED (2026-06-13). Components A (FP8 scoring
GEMM), B (stochastic-rounding fold), and C (error-feedback) were all built,
unit/GEMM-parity tested on CPU + GPU, and validated end-to-end on the live 35B
pool (SR fold correct — `update_norm` matched bf16; OOM/timeout fixed). **But the
core premise failed: FP8 did not speed up scoring** (ZORL teacher-forced scoring
is short-sequence, not MoE-GEMM-bound — measured FP8 t_score ≈ bf16 ~40s), and the
FP8 fold made the apply ~20× worse, for a net ~3.3× per-step *regression*. Effort
dropped, GPUs relinquished. The code is sound and reusable if a long-sequence
workload ever needs it — but A/B-confirm FP8 *scoring* is faster first. See the
§8.0 VERDICT in `FP8_BASE_FRESH_AB_IMPLEMENTATION.md` for the numbers.

- Branch: `apanda-dev-fp8-fresh-ab` in the SGLang fork (worktree
  `/home/apanda/xorl-sglang-fp8-fresh-ab`, off `apanda-dev-fold-gpu-gdn`).
- New module: `python/sglang/srt/lora/fp8_fold.py` (SR + EF requant primitives).
- Edits: `lora/layers.py` (Component A), `lora/lora_manager.py` (Component B/C).
- Tests: `test/registered/lora/test_fp8_fold.py` (16, CPU) +
  `test_fp8_moe_gemm_parity.py` (2, GPU). bf16 path proven byte-identical
  (existing `test_zorl_fresh_ab.py` 31/31 still green).

---

## Original design (as authored)

**Status:** design (not yet built). Author handoff for an implementing agent/engineer.
**Goal:** serve the Qwen3.6-35B-A3B base in FP8 to ~2× the scoring forward (the dominant
per-step cost), while keeping the EggRoll/`fresh_ab` ES update mathematically intact.
**Headline result this builds on:** `fresh_ab` + resampled SFT reaches gradient-SFT parity
(held-out 0.9219 vs 0.93 reference) on pure inference infra. See `OPSD_ZORL_RUNBOOK.md` and
memory `zorl-es-levers-audit`. The FP8 win compounds into the planned characterization sweep
(halves its GPU-hours).

---

## 0. TL;DR

- Base stays **FP8 E4M3, block-wise 128×128** (the checkpoint's native format). Candidates stay
  **bf16 LoRA** on top. Scoring forward = FP8 MoE GEMM + bf16 LoRA delta added to its output.
- The **fold (apply) is the only hard part.** A `fresh_ab` step folds ΔW directly into the base;
  the per-entry update is **~0.1–0.15 ULP** at the FP8 block scale, so a deterministic
  `W_fp8 += delta` rounds ~90% of the ES signal to zero. **Fix: stochastic rounding (SR)** on the
  requantize — unbiased in expectation, which is all ES accumulation needs. This is the same
  primitive EggRoll uses for int8 *pretraining*.
- **Error-feedback (EF)** is a strictly-lower-variance refinement (keep the rounding residual,
  add it back next step) at the cost of a residual buffer. Ship SR first (memory-free), enable EF
  only if the parity gate shows a gap.
- **Parity gate:** an FP8+SR run MUST reach the bf16 plateau (~0.90 held-out on 4×4-mult) before
  any FP8 result is trusted.

---

## 1. Why naive FP8 + fold fails (the numbers)

The FP8 checkpoint (`models--Qwen--Qwen3.6-35B-A3B-FP8`): `quant_method=fp8`, `fmt=e4m3`,
`weight_block_size=[128,128]`, `activation_scheme=dynamic`. Each weight tensor `W` stores an
`F8_E4M3` payload + a bf16 `weight_scale_inv` of shape `[ceil(rows/128), ceil(cols/128)]`;
`W_real[i,j] = W_fp8[i,j] * scale_inv[i//128, j//128]`.

Magnitude budget (measured, 4×4-mult, `fresh_ab` at the proven lr=3.8e-4):
- Per-step folded `‖ΔW‖_F ≈ 17.1` (the `update_norm` metric), spread over ~3.2×10¹⁰ targeted
  dense weight entries (experts gate_up+down on 40 layers + attn qkv/o on 10) ⇒
  **per-entry update RMS ≈ 1×10⁻⁴**.
- A 128×128 block of expert weights has std ~0.01, max ~0.04; the block scale maps ~0.04 → E4M3
  max (448) ⇒ `scale_inv ≈ 9×10⁻⁵`. A 0.01-magnitude entry is fp8-value ~111, in E4M3's
  `[64,128)` exponent bucket where the mantissa step is `2^6·2^-3 = 8` fp8-units ⇒
  **real-units ULP ≈ 8 × 9×10⁻⁵ ≈ 7×10⁻⁴**.
- So **per-step delta / ULP ≈ 1×10⁻⁴ / 7×10⁻⁴ ≈ 0.14**. Deterministic round-to-nearest sends
  every sub-0.5-ULP delta to zero; the few that cross a boundary jump a full ULP (biased
  overshoot). The ES signal — which lives entirely in the *accumulation* of these tiny steps
  (per-step gradient cosine is ~1.4×10⁻⁴; see E5 in the runbook) — is destroyed.

Keeping a **bf16 master** of the target weights to fold into avoids this but costs ~64 GB (the
full model) — it defeats the FP8 memory win and most of the point. Rejected as the default.

---

## 2. Architecture

```
                        ┌─────────────────────────────────────────────┐
   scoring (hot path)   │  x @ W_base[FP8 128×128]   (fast FP8 GEMM)    │
   per candidate        │     +  scaling · (x@A)@Bᵀ  (bf16 LoRA delta)  │
                        └─────────────────────────────────────────────┘
                                         │  reward = mean answer logprob
                                         ▼
   apply (per step)     ΔW = (lr/N) Σᵢ zᵢ·scaling·ε_B,i·ε_A,iᵀ     [bf16, computed as today]
                                         │
                        per touched 128×128 block:
                          w_bf16 = dequant(W_fp8 block, scale_inv)        # transient
                          w_bf16 += ΔW_block  (+ EF residual, §5)
                          scale_inv' = recompute_block_scale(w_bf16)      # §4.3
                          W_fp8 block = quantize_SR(w_bf16, scale_inv')   # §4.2
                          (EF: residual = w_bf16 − dequant(W_fp8', scale_inv'))   # §5
```

What stays FP8: the resident base weights (`w13_weight`, `w2_weight` on `FusedMoEWithLoRA`;
attn `qkv_proj`/`o_proj`; GDN `in_proj_*`/`out_proj`). What's bf16: the candidate LoRA adapters
(unchanged), the transient per-block dequant during the fold, and (if EF) the residual buffer.
No persistent bf16 master of the base.

---

## 3. Component A — FP8 base GEMM + bf16 LoRA delta (scoring)

The `fresh_ab` scoring forward already runs through `FusedMoEWithLoRA._forward_with_lora_virtual`
(`lora/layers.py:1636`): base gate_up GEMM (`_base_gate_up_gemm`, `:1251`) → LoRA delta added via
`merged_experts_fused_moe_lora_add` with `fuse_add_to_output=True` → activation → base down GEMM
(`_base_down_gemm`, `:1290`) → LoRA delta → reduce.

The LoRA delta path is precision-agnostic (it's a separate bf16 grouped-GEMM whose result is
added into the base GEMM output by the `FUSE_ADD_TO_OUTPUT` epilogue in
`fused_moe_triton_kernels.py:611`). **The only change is the base GEMM itself:** `_base_gate_up_gemm`
/ `_base_down_gemm` must dispatch the FP8 block-wise MoE kernel (SGLang already has
`w8a8_block_fp8` / blockwise-FP8 fused-MoE kernels) when the base layer is FP8-quantized, instead
of the bf16 GEMM. Activations are quantized dynamically (per `activation_scheme=dynamic`) inside
that kernel.

Integration checklist:
- Detect FP8 base (`w13_weight.dtype == torch.float8_e4m3fn` + presence of `w13_weight_scale_inv`)
  and route the two base GEMMs to the blockwise-FP8 path.
- Confirm `fuse_add_to_output` accumulates the bf16 LoRA delta into the FP8 kernel's (bf16/fp32)
  output buffer **before** any requantization — the delta must not be FP8-rounded.
- Attention/GDN LoRA path: those base modules are also FP8; the standard SGMV LoRA add is already
  bf16-delta-into-output, same requirement.
- LoRA candidate **scoring is unaffected numerically** — verify with a logprob-parity check
  (FP8-base + bf16-LoRA candidate vs bf16-base + same candidate should agree to FP8 forward
  tolerance, ~1% relative on the answer logprobs; the ES *ranking* must be preserved, which is the
  real requirement — check pair-delta Spearman ρ ≥ 0.95 across the two).

Expected win: ~2× on the base MoE GEMM, which dominates scoring (~40 s of a ~52 s step) ⇒ step
~52 s → ~30 s. The LoRA delta and the fold are unaffected.

---

## 4. Component B — stochastic-rounding fold (apply)

The `fresh_ab` apply currently runs `_apply_zorl_rewards_fresh_ab` (`lora/lora_manager.py:2345`),
which (in the `gpu_direct` path) builds per-chunk noise via `_zorl_build_fresh_ab_chunk_adapter`
(`:2262`) and folds through `_zorl_fold_chunk_adapter_direct` (`:2844`) →
`_fold_standard_module_from_tensors` (`:2989`) / `_fold_moe_module_from_tensors` (`:3089`). The
MoE fold today does `w13[e].add_(B_gu@A_gu)` / `w2[e].add_(B_dn@A_dn)` directly on the resident
weight (bmm-batched over experts). For FP8 these `add_`s become the dequant→add→SR-requant cycle.

### 4.1 Gate
New env `XORL_ZORL_FP8_FOLD=stochastic` (default `off` = bf16 base, current behavior). Only
meaningful when the base is FP8. Composes with `XORL_ZORL_FRESH_AB_FOLD=gpu_direct`.

### 4.2 Stochastic rounding
For a target real value `v` and quantization step `q` (= `scale_inv` for the block), let
`f = v/q`, `f0 = floor(f)`. Round up with probability `f − f0`:
```
W_fp8 = f0 + (rand_uniform[0,1) < (f − f0) ? 1 : 0)     # then clamp to E4M3 grid
```
E[`W_fp8`·q] = v exactly. Use a counter-based RNG seeded per (step, block, lane) for
reproducibility (the run must be re-derivable from seeds, like the rest of ZORL). Note E4M3 is
**not uniform-step** — the mantissa step doubles each exponent octave — so SR must be done against
the *local* fp8 grid: decompose `v/scale_inv` into its E4M3 exponent bucket and stochastically
round the 3-bit mantissa. A correct helper: compute the two bracketing E4M3 values `lo ≤ v' ≤ hi`,
round to `hi` with probability `(v'−lo)/(hi−lo)`, else `lo`. (`lo`/`hi` come from
`torch`'s fp8 cast of `v'` and `nextafter` in fp8 space, or a small LUT over the 256 E4M3 codes.)

### 4.3 Per-block rescale
Each 128×128 block carries one `scale_inv`. After folding, the block max may have drifted; recompute
`scale_inv' = blockmax(|w_bf16|) / 448` (E4M3 max) before requantizing, and write it back to
`weight_scale_inv`. Per step the drift is ~0.1 ULP so rescale is usually a no-op, but over a full
run the weights move — recompute every fold (it's a cheap per-block max + divide, negligible vs
the GEMM). Guard against degenerate blocks (all-zero → keep prior scale).

### 4.4 Where it lands in code
- `_fold_moe_module_from_tensors`: replace the two `w13[...].add_(delta)` / `w2[...].add_(delta)`
  with `fp8_fold_(weight_fp8, scale_inv, delta, mode)` where `mode ∈ {bf16_add, stochastic,
  stochastic_ef}`. The delta is already the per-expert bmm result (`B@A`); dequant the affected
  blocks, add, SR-requant, write back weight + scale.
- `_fold_standard_module_from_tensors`: same for the 2-D attn/GDN weights.
- Vectorize over the block grid (don't Python-loop blocks): dequant the whole tensor to bf16
  (transient, one tensor at a time to bound memory), add the dense delta, recompute all block
  maxes, SR-requant the whole tensor. Transient bf16 = one weight tensor at a time (~tens of MB),
  not the whole model.

---

## 5. Component C — error-feedback (EF) refinement

SR is unbiased but adds per-step variance `~ (ULP)²/12` per entry. EF removes most of it: carry the
rounding residual and fold it back next step.

```
# per block, per step:
target   = dequant(W_fp8, scale_inv) + ΔW_block + residual_block     # residual from last step
scale'   = recompute_block_scale(target)
W_fp8'   = quantize(target, scale')          # deterministic round-to-nearest is fine WITH EF
residual_block = target − dequant(W_fp8', scale')                    # the part we dropped
```
With EF the cumulative quantization error is bounded by one ULP *total* (not per-step), so the
accumulated weight tracks the true bf16 trajectory to ~1 ULP regardless of run length —
strictly better than SR's random walk. EF can use round-to-nearest (deterministic) since the
residual carries the bias forward; SR + EF together is belt-and-suspenders and unnecessary.

**Cost:** a residual buffer the size of the target weights. Options, cheapest first:
- **bf16 residual over targets ≈ 64 GB** — too big (defeats FP8).
- **fp8/int8 residual ≈ 16–32 GB** — the residual is itself ~1 ULP magnitude, so a *separate*
  fp8 (or int8) buffer scaled to ULP range represents it fine; 16 GB at int8. Fits alongside the
  32 GB FP8 base on 2×80 GB TP=2 with KV headroom.
- **rank-restricted residual** — keep only a low-rank sketch of the residual (the fold delta is
  itself low-rank per step); cheaper but lossy. Probably unnecessary.

**Recommendation:** ship **SR-only** first (zero buffer). The ES accumulation already averages
over hundreds of steps, so SR's variance is small next to ES estimation noise. Enable EF
(int8 residual) only if the parity gate shows SR plateauing below bf16.

---

## 6. Numerical analysis / why this is safe

- ES needs the *accumulated* update direction, not per-step fidelity (per-step cosine to the true
  gradient is ~1.4×10⁻⁴; the method works by averaging over ~300 steps against a batch-coherent
  target, coherence 0.882 — see E5). SR preserves E[ΔW] per step ⇒ E[ΣΔW] is the true accumulation.
- SR random-walk error after T steps: per entry ~ULP·√(T·var_frac) where var_frac ≤ 1/12. At
  ULP 7×10⁻⁴, T~1000 ⇒ ~7×10⁻⁴·√83 ≈ 6×10⁻³ accumulated rounding noise vs an accumulated *signal*
  of order (per-step 1×10⁻⁴)·(coherent fraction)·T. Estimate the SNR before trusting; if marginal,
  EF collapses the rounding error to ~1 ULP total. (This back-of-envelope is exactly what the
  parity gate measures empirically — don't rely on it alone.)
- E4M3 dynamic range is fine for the *weights* (block-scaled). The risk is precision, addressed by
  SR/EF. Activations are already dynamic-FP8 in the checkpoint's scheme.

---

## 7. Parity gate (mandatory before trusting any FP8 result)

1. Stand up one FP8 pool (`--model …-FP8`, FP8 fused-MoE GEMM enabled) + the SR fold.
2. Run `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE` (the parity recipe, seed-matched) on it.
3. Gate: held-out probe must reach **≥ 0.88 by step ~300 and plateau ~0.90**, matching the bf16
   run's trajectory within probe noise (±2.6 pts). Compare *curves*, not just peak.
4. Secondary checks: (a) scoring logprob parity vs bf16 (pair-delta Spearman ρ ≥ 0.95);
   (b) `update_norm` per step within ~5% of bf16; (c) no slow divergence past step 300
   (the signature of biased rounding — would indicate the SR/rescale is wrong).
5. If SR fails the gate → enable EF (int8 residual) and re-run. If EF fails → the FP8 GEMM
   scoring (Component A) can still ship *alone* with the bf16 fold (keep base bf16 for the fold
   target via a different mechanism), but that loses the memory win; reassess.

---

## 8. Implementation plan

Branch `apanda-dev-fp8-fresh-ab` off current `apanda-dev` HEAD; worktree; GPU tests on idle GPUs
only; no kubectl. Land behind env flags, default-off (bf16 path byte-identical).

1. **Component A** (FP8 GEMM + bf16 LoRA add) — `lora/layers.py` `_base_gate_up_gemm`/
   `_base_down_gemm` FP8 dispatch; verify `fuse_add_to_output` accumulates pre-requant.
   Test: logprob/pair-delta parity vs bf16 on a fixed adapter.
2. **Component B** (SR fold) — `fp8_fold_` helper (E4M3-correct SR + per-block rescale);
   wire into `_fold_moe_module_from_tensors` / `_fold_standard_module_from_tensors` behind
   `XORL_ZORL_FP8_FOLD=stochastic`. Tests: (i) SR unbiasedness — over many folds of a known small
   delta into an fp8 block, mean reconstructed ≈ deterministic bf16 accumulate (fp32 reference);
   (ii) rescale correctness when a block max grows; (iii) momentum=0 bf16 path unchanged.
3. **Parity gate run** (§7) — the real validation.
4. **Component C** (EF, int8 residual) — only if §7 needs it. Adds a residual buffer keyed per
   target weight + the EF update in `fp8_fold_`.

Env flags: `XORL_ZORL_FP8_FOLD ∈ {off, stochastic, stochastic_ef}` (default off);
pool serves `--model …-FP8` with the FP8 fused-MoE path enabled.

---

## 9. Risks / open questions

- **SGLang FP8 fused-MoE + LoRA composition.** The LoRA-MoE virtual-experts path (this fork's
  addition) was written against the bf16 base GEMM. Confirm the blockwise-FP8 MoE kernel exposes
  the same output buffer the `fuse_add_to_output` epilogue writes into, or adapt.
- **E4M3 non-uniform step in SR.** Must round against the local fp8 grid (§4.2), not a fixed step,
  or low-magnitude blocks get biased. The LUT-over-256-codes helper is the safe implementation.
- **Per-block rescale churn.** Recomputing `scale_inv` every fold is correct but interacts with any
  cached/compiled GEMM that assumed static scales — verify the FP8 kernel reads scales live.
- **Is it worth it post-parity?** Parity is already reached in bf16. FP8's value is the ~2×
  scoring speedup compounding into the characterization sweep (fold-norm × population × rank ×
  batch, ~25–30 runs). If the sweep is descoped, FP8 priority drops accordingly.
- **GDN + FP8.** The GDN-mixer adapter (`-hybridattn-gdn`) targets `in_proj_*`/`out_proj`; those
  base weights are also FP8 and fold via `_fold_standard_module_from_tensors`, so they're covered
  by the same SR path — but include a GDN layer in the fold unit tests.
