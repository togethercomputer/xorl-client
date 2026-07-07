# MECHANISM SANDBOX (2026-07-06) — standalone HF-side FSDP trainer with full forward-graph control

**Track:** MECHANISM-SANDBOX (prefill-time-compute program). **Status: BUILT +
CPU-gated; GPU validation jobs STAGED** (coordinator executes the run.sh write —
see §7). This is the enabling artifact for the next mechanism wave
(RECONCILIATION_PREFILL_DEBATE_20260706.md §3 mechanisms 1–2 + the win-condition
dose-response requirement); it runs NO mechanism arms itself — attempt-#1
preregistration is separate.

## 1. Why this exists

The xorl engine cannot express the next wave's objectives (WORDLE_P3_TRACK §V4
feasibility audit): (a) slot-position **K/V matching** — no grad-preserving
k/v capture reaches any loss path; (b) **attention-target** losses — attention
probabilities are never materialized (FA/varlen); custom **attention masks**
and **discretized slot write-back** have no engine surface at all. The V4 audit
ranked these as "engine surgery / deepest surgery"; the reconciliation instead
sanctions a standalone HF/torch harness with the recurrence track's
shared-forward + parity discipline (stage-0 validated that pattern end-to-end
on marin at ~zero overhead).

Feature list is derived from the reconciliation:
- §2 "what exploiting the tower would require" → composable codes
  (value/hidden supervision, quantized write-back = the discretization gift),
  attention routing (attention-target loss), read bottlenecks (mask control);
- §3 win condition clause 1 → **trained k-dose-response requires per-example
  k-SAMPLING** (a fixed-k model cannot exhibit a dose curve);
- SCRAMBLED_ALIGNMENT (reading [B]: buffer = unordered content reservoir) →
  **SET-matching** (order-free assignment) as a first-class supervision mode;
- ATTENTION_AUDIT (answer RECOMPUTES over raw values; slots not fan-in read at
  full-attn layers) → the bottleneck arm (block answer→problem) and the
  attention-target loss aim exactly at that read path.

## 2. What was built (all additive; commit set below)

| file | role |
|---|---|
| `scripts/mechanism_forward.py` | ONE shared implementation of every feature: FeatureConfig, STE quantizer, Hungarian/Sinkhorn set-matcher, mask builders, hook-based activation capture (CaptureMixin), MarinMechModel (dense backend over recur_loop.RecurLayer — layer math NOT forked), Q35BMech (hybrid backend over the unmodified HF Qwen3_5MoeForCausalLM), loss assembly, batch builders, GDN leak probe, strict 35B text-only loader |
| `scripts/mechanism_sandbox_train.py` | torchrun FSDP trainer, `--model marin\|q35b`; marin `--arm c0` is bit-compatible (rng/data/recipe) with `recur_stage0_train.py` = gate (i); q35b = FSDP bf16-compute/fp32-master + activation checkpointing + tokens/s + peak-mem telemetry = gate (ii) |
| `scripts/mechanism_export_hf.py` | sandbox ckpt → **production-servable HF dir** (marin: HF layout + aux files; q35b: base-layout hfconv-style dir with visual/mtp copied verbatim + `preprocessor_config.json` — the patched-converter lesson, verified present in `convert_sampler_export_to_hf.py` lines 106-114). Two-way STRICT key audits (the "Unexpected key" doctrine) |
| `scripts/mechanism_sandbox_gates.py` | unit gates U1–U6 (below) |
| `scripts/mechanism_parity_compare.py` | gate (i) verdict instrument (JSONL-only) |
| `scripts/comp_bench_eval.py` | ADDITIVE `--pause-slots` / `--pause-slots-per-op` (default-off = bit-identical): slotted prefill `prompt_ids + [1873]*k + ANS_PREFIX_IDS` == buffer_sft `prefill_ids`; enables the eval-time k-dose rows on the production instrument |
| `ms_jobs/ms_gate_chain.sh` + `ms_jobs/run_sh_rendered_c4.sh` | staged GPU validation chain (DONE-file guarded, resumable) + rendered run.sh |

**Eval path is 100% production:** train here → `mechanism_export_hf.py` →
sglang serve → `comp_bench_eval.py [--pause-slots]` (35B) /
`eval_math_suite.py --prompt-style pause-slots` (marin, dc_arm_chain pattern).
Zero eval-time dependence on sandbox code.

### FSDP discipline (the design's load-bearing constraint)
Under FULL_SHARD a unit's params exist only during its own forward. Therefore
(i) all parameter-touching compute (layer forwards, lm_head/embedding calls,
the quantizer's `logits` metric) runs INSIDE the root module forward;
(ii) loss assembly afterwards consumes only **captured activations** — q/k are
captured at the `q_norm`/`k_norm` module outputs (= post-norm PRE-RoPE, the
K/V-loss capture point), v at `v_proj`, hiddens at decoder-layer boundaries,
rotary cos/sin from the rotary module. Attention rows are recomputed from
those captures (verified == HF eager attention weights, gate U4b, max Δ 0.0).
Activation checkpointing EXCLUDES capture layers (a checkpointed layer re-runs
its forward in backward; the first pass is no-grad and would poison captures).

## 3. Feature matrix (v1)

| feature | marin 9.7B dense | Qwen3.6-35B-A3B hybrid | unit gate |
|---|---|---|---|
| (a) slot VALUE targets (input≠label CE remap, arbitrary positions) | ✅ exact | ✅ exact (incl. built-in `trace-digits` labels from the comp-bench generator's `trace`) | U5 |
| (a) slot HIDDEN targets, layer-selectable (-1 = post-norm final) | ✅ | ✅ (hooked decoder-layer outputs) | U5 |
| (a) SET-matching (Hungarian, rectangular m≤k; Sinkhorn variant, detached plan) | ✅ | ✅ | U1 (brute-force verified) |
| (a) per-example k-SAMPLING (`k_choices` = multiples of base k or absolute) | ✅ | ✅ (deterministic in seed/step/row) | offline check |
| (b) mask control: block (query→key) region pairs; regions = problem/prompt/slots/answer | ✅ **EXACT** (all 37 layers; bit-invariance proven) | ⚠️ full-attn layers ONLY (10/40) — see §4 GDN caveat; leak probe included | U3 |
| (c) quantized slot write-back: hard STE snap (bit-exact rows) / top-k soft mixture; per-pass reader blocks (`quantize_passb_blocks`) for the strict discrete bottleneck | ✅ | ✅ (metric `logits` under FSDP; guard raises on sharded-weight misuse) | U2 (bit-level), U3(c) |
| (d) K/V matching at chosen layers/slot positions, post-q/k-norm PRE-RoPE | ✅ | ✅ full-attn layers only (asserted: GDN layers have NO K/V interface) | U5 |
| (d) attention-target KL (answer rows over slots, head-mean, renormalized) | ✅ | ✅ full-attn layers only | U4b + U5 |
| parity path (all features off) | == plain HF forward (U4: max Δlogit **0.0** fp32 CPU; real-ckpt top-1 gate staged) | hooks-installed forward **BIT-identical** to plain HF (torch.equal); custom-4D-causal-mask path Δ **0.0** | U4 |

**Definitional conventions (frozen):** "layer l hidden" = output of decoder
layer l (0-based); final = post-final-norm. K/V capture point = after q/k head
RMSNorm, before RoPE (phase-free; targets from any position layout are
comparable). `kv_rope="post"` is NOT implemented — post-RoPE targets are only
meaningful if teacher positions == student positions; rather than silently
mis-phase, the assembly asserts `pre` (documented decision, revisit only with
position-aligned captures). Attention-target student statistic = head-mean of
the recomputed post-softmax row restricted to slot cols and renormalized;
target = uniform-over-slots (built-in) or per-item `attn_target` from the
teacher bundle.

**Teacher bundle schema** (`--slot-teacher-dir`, one `th_{idx:06d}.pt` per
item, the P3-capture precedent): `hiddens` [m,H]; `k_l{L}`/`v_l{L}`
[m, n_kv_heads, head_dim] (post-norm pre-RoPE); `attn_target` [k].

## 4. The 35B hybrid caveat (feature b), stated precisely

Qwen3.6-35B-A3B: 40 layers, full attention ONLY at [3,7,11,15,19,23,27,31,35,39]
(config-derived, matches the attention audit). The other 30 are GDN
(gated-delta-rule linear attention): a **causal recurrent scan with
input-dependent decay plus a depthwise conv (kernel 4)**. Two consequences:

1. **No mask can constrain GDN.** The HF GDN path ignores 4D masks entirely
   (`apply_mask_to_padding_states` requires 2D). A "blocked" span's content
   enters the recurrent state and reaches every later position, and the conv
   additionally bleeds ≤3 positions across any span boundary. **On the 35B the
   bottleneck guarantee is RESTRICTED to the 10 full-attention layers.** The
   tiny-hybrid gate demonstrates the leak (blocked full-attn mass exactly 0.0
   while answer states still move under problem substitution: max|Δh| 0.024 on
   the tiny model); `gdn_leak_probe` measures it on the REAL model per context
   (final-hidden cos/L2 + answer-logit KL + greedy flip rate) — staged in U6.
   Bottleneck ARMS on the 35B must report the measured leak next to any claim;
   marin-dense is the exact-guarantee substrate (U3: bit-identical answer
   states under substitution, Δ = 0.0).
2. **Batch layouts must be right-pad-ONLY** (mid-sequence pads would corrupt
   the GDN scan — pads still decay the recurrent state). `build_batch_q35`
   uses a contiguous per-row layout; trailing pads cannot affect real
   positions in a causal scan.

**Eval-time mask caveat (applies to both models):** sglang serving is
UNMASKED — the mask is a TRAINING-time constraint shaping the circuit;
production evals measure the unmasked model (removal/dose instruments).
`MarinMechModel.generate_greedy_masked` exists as an offline DIAGNOSTIC
(maintains blocks during decode) for mask-on/mask-off contrasts; it is not an
eval instrument.

## 5. Quantizer semantics (feature c)

Two-pass write-back: **pass A (writer)** = plain forward; slot final hiddens
snap to token embeddings (hard: STE, forward value BIT-exactly the embedding
row — `e + (h - h.detach())`, gated at torch.equal level; topk: softmax(tau)
mixture, fully differentiable). **Pass B (reader)** = full forward with slot
INPUT embeddings replaced by the snapped vectors — the decode-time token
write-back gift (reconciliation §2: discretization as per-step error
correction), differentiable end-to-end. Slots compute in parallel within a
pass (Jacobi, 1 iteration in v1 — slot i does NOT see slot j's snapped token;
documented limitation, iterable later). Slot supervision (value/hidden) reads
PASS-A states (the writer); answer CE reads pass B. The **strict discrete
bottleneck** composes (b)+(c): `mask_blocks="answer->problem"` +
`quantize_passb_blocks="slots->problem"` — writer slots read the problem,
reader slots see only the snapped tokens, so the answer's entire problem
channel is the discrete token choice (gate U3(c): writer problem mass 7.59,
reader problem mass exactly 0.0; marin-exact, 35B GDN caveat applies).
Under FSDP the snap metric is `logits` (module-call safe); `cosine`/`dot`
require the unsharded embedding and are guarded against silent misuse.

## 6. Gates

### CPU unit gates — RUN, ALL PASS (2026-07-06, marin venv, banked at `/shared/apanda/filler_grpo/mechanism_sandbox/gates/gates_cpu_all.json`)

| gate | result |
|---|---|
| U1 set-matcher | 200/200 brute-force cost-equality trials; permutation recovery exact; order-invariance ≤1e-6; Sinkhorn low-τ plan concentrates on the permutation; grads flow |
| U2 quantizer | hard snap **bit-equal** to embedding rows (cosine AND logits metrics); STE grad **== identity** (torch.equal); topk differentiable; sharded-weight guard fires |
| U3 dense mask | blocked answer→problem attention probs **exactly 0.0** at every layer; FULL bottleneck (answer->prompt;slots->prompt) ⇒ answer states **BIT-invariant** (Δ=0.0) to problem substitution vs open-mask control Δ=2.45; strict quantize bottleneck: pass-A problem mass 7.59 / pass-B **0.0** |
| U3 hybrid GDN leak | full-attn blocked mass exactly 0.0 AND substitution still moves answer states (max Δh 0.024 on tiny hybrid) — the caveat is real and the probe detects it |
| U4 parity | dense MechModel vs HF: max Δlogit **0.0** (fp32 CPU), top-1 1.0; hybrid hooks-installed forward **torch.equal** to plain HF; 4D-causal-mask Δ **0.0** |
| U4b attn recompute | captured-activation row recompute == HF eager attn_weights, max Δ **0.0** |
| U5 loss plumbing | ALL features simultaneously on (quantize+value+hidden-set-match+kv+attn): losses finite on both backends; grads present on k_proj/q_proj/embedding as expected; kv loss exactly 0.01 for a +0.1-perturbed self-teacher (sanity-exact) |

Offline construction checks (same session): q35 batch **bit-compatible** with
buffer_sft/comp_bench conventions (seq, CE shift, ANS_PREFIX_IDS, PAUSE 1873);
problem-span frame-diff decodes to the problem text on both tokenizers;
trace-digit labels decode correctly (`-051` for −51, 4-wide sign+digits);
k-sampling deterministic. 35B strict text-only load: PASS on the real
snapshot (34.66B params, full-attn [3,7,...,39], untied lm_head, spot tensor
bit-equal, 28 s).

### GPU gates — STAGED (chain `ms_jobs/ms_gate_chain.sh`, run.sh rendered for er-buffer-c4)

- **(i) marin parity:** features-off `--model marin --arm c0 --task sub`
  500 steps, seed 42 — same rng stream, same `build_batch`, same recipe as
  `recur_stage0_train.py` (data sequence identical by construction; no new
  parameters exist). PREREGISTERED TOLERANCES (`mechanism_parity_compare.py`):
  step-0 |Δce| ≤ 0.02 (identical weights+batch; residual = kernel-shape noise
  — the sandbox runs one full-sequence pass where recur phase-split
  prefix/tail); mean |Δce| steps 0-49 ≤ 0.05; final per-band val CE |Δ| ≤ 0.10
  (B1 same-seed relaunch precedent). Ref = banked
  `/shared/apanda/filler_grpo/recur_stage0/logs/train_c0_sub.jsonl`.
- **(ii) 35B feasibility smoke:** 50 steps, band 5-7 (13k-pool sequential
  slices, bs 64 global), answer-CE only; then a 5-step ALL-features-on
  overhead smoke (value+mask+quantize+attn on layers 19,23). Reports tokens/s
  + `torch.cuda.max_memory_allocated/reserved` per step (logged in the train
  JSONL). **Memory arithmetic on record (8×80GB, FULL_SHARD):** fp32 master
  17.5 G/GPU + AdamW 35 + grads {fp32: 17.5 | bf16: 8.75} ⇒ persistent
  {70 | 61.25} G/GPU + bf16 gather transients (~2×0.9B-param units) +
  activations (tiny at seq ~130-250). The fp32-reduce config is MARGINAL by
  design; the chain runs the ladder fp32-reduce → bf16-reduce → adafactor
  (~35 G/GPU, no momentum) and records the first success. Throughput unknowns
  (torch-fallback GDN kernels — no FLA in the venv; HF looped MoE experts) are
  what the smoke measures. **If all three rungs fail, v1 is declared
  marin-only** — stated in advance as an acceptable outcome; the feature set
  is fully validated on marin regardless.
- **(iii) real-model unit gates (U6):** marin MechModel-vs-HF top-1 (recur
  S0.2 bar: 100%); 35B strict-load + greedy coherence + hook bit-identity +
  real-layer mask-prob inspection (must be exactly 0.0) + `gdn_leak_probe`
  n=8 real band items.
- **Eval-path proof:** marin export → transformers reload + sglang serve
  probe; q35b export → hfconv-style dir (preprocessor_config.json asserted)
  + sglang TP2 serve probe.

### RESULTS (GPU gates) — appended from the banked JSONs; FILES-ONLY discipline

**Chain run 1 (deployed by coordinator, bust 1783375627; started 22:07:08Z
07-06; exited rc=5 at P4 22:17:28Z — resumable):**

- **P1 CPU gates on c4: ALL PASS** (numbers identical to the dev-box run;
  `gates/gates_cpu.json`).
- **P2 U6 real-marin parity: PASS — top-1 100% AND max|Δlogit| = 0.0** on the
  real 9.7B checkpoint (bit-exact, exceeds the recur S0.2 bar).
- **P3 GATE (i) marin 500-step parity (`parity/parity_report.json`):** step-0
  |Δce| = **0.0008** (PASS ≤0.02); early-mean |Δce| steps 0-49 = **0.0023**
  (PASS ≤0.05); endpoint final-val criterion **FAILED as preregistered**
  (d16 Δ0.156, d32 Δ0.122 > 0.10). Labeled post-hoc addendum
  (`parity/parity_addendum_posthoc.json`): the reference run's OWN
  adjacent-checkpoint val swings exceed 0.10 at 3/6 bands (d2 0.107, d24
  0.110, d32 0.136) — the criterion sits below the endpoint statistic's
  self-noise; the smoothed last-3-vals statistic passes ≤0.10 at EVERY band
  (max 0.085); train-CE |Δ| plateaus at ~0.025-0.03 mean (bounded drift).
  **COORDINATOR ADJUDICATION (recorded verbatim):** "the formal endpoint
  criterion FAILED as preregistered and that stands in the record — but the
  criterion is adjudicated MIS-CALIBRATED (set below the reference run's own
  adjacent-checkpoint val noise, which your addendum documents at 0.107-0.136
  on 3/6 bands), and the totality (step-0 0.0008, early tracking 0.0023,
  bounded CE drift, smoothed last-3 ≤0.085 all bands) is accepted as adequate
  parity. HOWEVER, the binding design consequence for attempt-#1 — record
  this as the adjudication's condition: ALL mechanism-vs-baseline comparisons
  will be WITHIN-SUBSTRATE (any sandbox-trained mechanism arm is compared
  against a sandbox-trained C0 at matched config, never against the
  engine-trained C0 curve directly; the engine C0 budget curve remains the
  program-level reference but the arm-level McNemar pairs are same-trainer).
  This neutralizes residual trainer deltas entirely and is better design
  regardless. Proceed with the chain; no re-run of gate (i) needed."
- **P3b marin export:** strict two-way key audit PASS; transformers reload +
  greedy sample coherent. sglang serve probe **ok=0 (non-fatal)**: the BARE
  venv launch crashed with an illegal memory access in the piecewise-CUDA-
  graph `fused_add_rmsnorm` kernel — an sglang LAUNCH-CONFIG issue, not an
  export defect (reload audit clean). FIX adopted into the chain: the
  DC-chain's proven marin invocation (repo PYTHONPATH +
  `--disable-piecewise-cuda-graph --disable-custom-all-reduce
  --attention-backend fa3 --dtype bfloat16 --trust-remote-code`,
  SGLANG_DISABLE_ROPE_COMPILE=1). Mechanism-arm evals must use the DC-chain
  `launch_sampler` invocation verbatim.
- **P4 U6 real-q35b: FAILED run 1 with a REAL BUG the tiny fp32 CPU gates
  could not catch** — SDPA rejects additive masks whose dtype ≠ query dtype
  ("invalid dtype for bias"): `build_mask4d` hardcoded fp32 while the real
  model computes bf16; masks are built INSIDE forward so FSDP's root
  input-cast would never have fixed it in training either. **FIXED at
  source** (`Q35BMech.mask_dtype`, derived from embed dtype unwrapped /
  set to MixedPrecision.param_dtype by the trainer) + a bf16 masked-forward
  regression added to gate U3-hybrid (now in the standard CPU suite,
  passing). Chain resumes at P4 on redeploy (P1-P3 DONE-marked; P3b marker
  cleared to re-prove the serve probe with the fixed flags).

**Chain run 2 (redeploy after the P4 fix; started 22:24:59Z, COMPLETE
22:42:44Z rc=0; c4 released, nothing serving):**

- **P3b (re-proof with the fixed sglang flags): marin serve probe ok=1** —
  the marin production path (export → reload → sglang → /generate) is proven
  end-to-end.
- **P4 U6 real-q35b: PASS** (`gates/gates_u6_q35b.json`): strict load audit
  clean; coherence sample sane; **hook bit-identity TRUE on the real model**;
  **blocked full-attn prob max = 0.0** on real band contexts. **GDN leak,
  real band-5-7 items (n=8), full bottleneck blocks
  (answer->prompt;slots->prompt): mean answer-hidden cosine 0.778, mean
  logit KL 0.319, greedy argmax flip rate 62.5%.** PREREG-CAVEAT VERBATIM:
  on the 35B the unmaskable GDN pathway carries enough blocked-problem
  content to flip the answer argmax on 5/8 real items — mask control there
  is a soft pressure on 10/40 layers, NOT a bottleneck guarantee; strict
  discrete-bottleneck arms are MARIN-ONLY.
- **P5 GATE (ii): PASS ON THE FIRST RUNG — 35B training is FEASIBLE.**
  fp32-reduce config (fp32 master + AdamW-fused + fp32 grads + activation
  ckpt), 50/50 steps, band 5-7, global bs 64: **avg 2826 tok/s (~2.5 s/step),
  peak 75.5 GB allocated / 83.3 GB reserved per GPU** (decimal GB; marginal
  as predicted but stable across 50 steps + the 3-step exportsrc rerun); CE
  3.14 -> 1.11, val CE 2.98 -> 0.95 (healthy, consistent with the engine C0
  ladder's trajectory). bf16-reduce / adafactor rungs untested (not needed
  features-off).
- **P5 feature-on overhead smoke: OOM at the fp32-reduce rung** (all features
  on: trace-digit slots k=4·n_ops, mask blocks, two-pass hard quantize, attn
  targets at layers 19/23 excluded from ckpting; failed allocating 1.56 GiB
  with ~74 GiB PyTorch-allocated). CONSEQUENCE for arms: features-OFF (C0
  controls) may run fp32-reduce; feature-ON 35B arms must use the
  bf16-reduce rung (~9 GB more headroom) — follow-up smoke STAGED
  (`ms_jobs/ms_featon_bf16_job.sh` + `run_sh_rendered_c4_featon.sh`, ~10-15
  min, includes a per-rank-bs-4 fallback) and should run before attempt-#1
  finalizes 35B arm configs.
- **P5b: q35b export + serve PASS** — 3-step ckpt run (same 75.5 GB profile)
  → full-state save → hfconv-style export (67G, strict two-way audit,
  preprocessor_config.json present) → **sglang TP2 serve probe ok=1**. The
  35B production eval path is proven end-to-end; raw sandbox ckpt deleted
  after export as designed.

**GATE MATRIX SUMMARY: every gate green** — (i) parity PASS-as-adjudicated
(within-substrate comparison condition binding), (ii) 35B feasible at
fp32-reduce features-off / bf16-reduce required feature-on (staged smoke),
(iii) all unit gates PASS incl. real-model hook bit-identity + exact mask
zeros + measured GDN leak; both production serve paths proven. **v1 supports
BOTH models**; the marin-only fallback clause is retired except for
strict-bottleneck arms (GDN caveat above).

## 7. Ops — the staged deploy (coordinator action required)

My permission context is expected to deny control-plane writes (DC-ARM
precedent: a coordinator message is not user consent). Everything is staged:

1. **Stack:** er-buffer-c4-trainer-head — status re-verified
   `exited_at=2026-07-06T09:40:58Z rc=0` (scrambled-alignment's clean v5
   release; campaign doc says all 8 GPUs 0 MiB at handover). Claim written to
   the campaign-doc SHARED-FILES block (MECHANISM-SANDBOX track). c2/c6 are
   equivalent substitutes if c4 is taken — the chain is stack-agnostic
   (only `STACK=` in the run.sh names it).
2. **The exact write:** copy
   `experiments/opd_profile/ms_jobs/run_sh_rendered_c4.sh`
   → `/shared/opd-control/er-buffer-c4/trainer-head/run.sh`
   as a SINGLE write, replacing `REPLACE_WITH_EPOCH_AT_WRITE_TIME` in the
   final `# bust` line with the current epoch (single sha change = single
   wrapper restart), after re-verifying the status file at write time
   (the 03:32Z race lesson).
3. **Expected runtime ~2-2.5 h** (P1 2m; P2 6m; P3 25m; P3b 15m; P4 25m;
   P5 60m incl. two possible OOM retries + fp32 load/broadcast; P5b 35m).
   Chain is resumable (DONE files under
   `/shared/apanda/filler_grpo/mechanism_sandbox/state/`); it kills its own
   process trees BY NAME (sglang.launch_server / torch.distributed.run /
   mechanism_sandbox_train — the scrambled-alignment teardown lesson) and
   fail-fasts if GPUs aren't clear (<1 GiB) at start. Disk guard 250G before
   every heavy phase; the q35b export source ckpt is deleted after
   conversion (peak transient ~140G).
4. **Nothing serves after exit**; c4 releases in the SHARED-FILES block when
   the verdict files are banked.

## 8. v1 support statement + known limitations

- **v1 supports:** both models end-to-end through training; all four features
  gated + unit-gated on both backends (35B pending the staged real-model
  gates); production export/serve path for both; eval-time k-dose rows via
  the comp_bench_eval extension; trained k-dose via per-example sampling.
- **35B training feasibility is UNPROVEN until gate (ii) runs** — the memory
  ladder is designed so at least the adafactor rung should fit (35 G/GPU
  persistent); if all rungs fail, **v1 is marin-only** and the doc will say so
  here explicitly.
- Quantizer is single-Jacobi-iteration (slots don't see each other's snapped
  tokens); `kv_rope=post` unimplemented (deliberate, §3); Sinkhorn plan is
  detached (DETR-style); 35B masked-decode diagnostic not implemented (marin
  only); mask blocks are training-time only (eval unmasked, §4); marin
  `--arm mech` uses the recur think-block slot layout (eval_math_suite
  pause-slots convention), 35B uses the buffer_sft bare-slots layout — per
  their respective production eval instruments.
- The 35B trainer trains the TEXT model only; visual/mtp ride along verbatim
  at export (identical to every hfconv dir sglang already serves).

## 9. Artifacts

- Code: `experiments/opd_profile/scripts/mechanism_{forward,sandbox_train,sandbox_gates,parity_compare,export_hf}.py`, `experiments/opd_profile/ms_jobs/`.
- CPU gate reports + all future GPU verdicts: `/shared/apanda/filler_grpo/mechanism_sandbox/`.
- Conventions inherited: PAUSE 1873 / ANS_PREFIX [198,15666,25,220] / ops6
  sysprompt (buffer_sft.py); marin pause-slots think-block + `Answer:`
  prefill token fact (recur S0.1.6); comp-bench data + trace targets
  (`/shared/apanda/filler_grpo/comp_bench/data/`).
