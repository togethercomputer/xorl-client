# GOAL: Apply ZORL (Evolution Strategies) to Wordle, beating the GRPO baseline

> Persisted from the approved plan (`~/.claude/plans/ok-so-now-i-glowing-rossum.md`).
> Companion docs: `ZORL_BEAT_093_{SCIENCE,INFRA,THROUGHPUT}.md` (the prior `mult`
> ES campaign), `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` (Wordle GRPO/eval).

## Context
We ran **ZORL** — a parameter-efficient *evolution-strategies* trainer (a rank-16
LoRA optimized on a **frozen** base via antithetic population perturbation +
seed-space fold, served by sglang pools, driven by a standalone client) — against
the **`mult`** task (prior best probe 0.9297, later shown to be a high-variance
draw, not a stable target). We now **re-point ZORL at Wordle**, reusing everything
learned, with an existing **GRPO (full-weight RL) run as the baseline to beat**
(held-out **0.5625**).

## 1. What we've been doing with ES (ZORL)
Optimize a rank-16 LoRA on a frozen base. Per step: sample N=128 antithetic pairs
(`fresh_ab`), score each candidate, fold a score-weighted update into the parent in
**seed space** (`gpu_direct`); no backprop — the "gradient" is the ES estimate.
Served by 8×TP2 sglang StatefulSets; standalone client
(`experiments/zorl/standalone/zorl_client.py`) drives load→score→apply→probe.
Science from the `mult` campaign: the 0.93 "ceiling" is an ES step-size/**drift**
problem, and **run-to-run variance (~±0.05–0.10) swamps config effects** — single-run
comparisons are noise-limited.

## 2. What we implemented from HyperscaleES (EggRoll), GPL-clean
- `fresh_ab` = EggRoll paired low-rank antithetic outer-product perturbation.
- Seed-space fold (`XORL_ZORL_FRESH_AB_FOLD=gpu_direct`).
- Trust region `max_update_norm` (EggRollBS `clip_by_global_norm`).
- Per-project baseline standardization (`project_baseline_standardized`).
- LR schedules: cosine + a new **delayed-decay** (`--lr-hold-steps`).
- Deliberately not ported (until now): Adam/optax. See `[[zorl-eggroll-optimizer]]`.

## 3. What ran successfully on `mult`
ES climbed cold 0.74 → **0.9297** (rank-16 `fresh_ab`, raw scoring, 128 pairs,
lr 3.8e-4, sigma 1.5e-4, seed 9246) via **teacher-forced SFT logprob scoring** +
greedy exact-match probe. Caveat: five faithful reproductions landed 0.83–0.88 —
ES *can* climb a held-out metric on a frozen base, but **the signal must correlate
with the eval, and single runs are noisy.**

## 4. What Wordle is (the new target)
Multi-turn (≤6 turns) word-guessing; "solve" = exact word in ≤6 turns.
- **GRPO baseline to beat:** Qwen3.6-35B-A3B, full-weight, from base, EP8 + **muon**
  (lr 5e-5), dense `wordle_retrieval_reward`, prompt
  `public_reasoning_constraints_think`, group/train 8 / pool 512 / seed 9234.
  Won at step 20: **36/64 = 0.5625** held-out (128-game: 0.5391).
- Floors (floor-protocol): base 0.219 · **base+think 0.469** · SFT+think 0.500 ·
  scaffold ceiling 0.969.
- **Eval discipline:** floor-protocol on a **separate idle serve** (never the live
  sampler), seed-777 `[0:64]`, temp 0.2, `max_new_tokens 12288`,
  `public_reasoning_constraints_think`; `eval_ckpt_generic.sh` +
  `eval_failure_taxonomy.py`. Rollout solve **under-reads** held-out.
- **ZORL Wordle task already exists** (`tasks/wordle.py`, multi-turn,
  `rollout_completion`, shaped reward). **But prior ES-Wordle (W000–W018) solved
  0 games** (teacher-forced logprob rose, exact_match stayed 0).

## 5. How to target Wordle with ES
Fix the reasons prior ES-Wordle got 0 solve:
1. **rollout-solve scoring** (ES fitness = actual solving), not teacher-forced logprob.
2. **GRPO 35B substrate + the floor-protocol probe** so cold ≈ 0.47 (real signal),
   not the prior 30B/rank-4 degenerate 0-solve probe.
3. **Judge by held-out floor-eval, multi-seed** (the `mult` variance lesson).

**Config:** clone the `mult` sglang pool → Qwen3.6-35B-A3B + LoRA; standalone client
`TASK=wordle`; `fresh_ab` rank 16; rollout-solve reward; floor-protocol probe;
constant LR first (delayed-decay later); cold base init.
**Target:** **beat GRPO 0.5625** (0.469 is the must-pass en-route gate), on
multi-seed floor-eval distributions.

## Workstream B: Muon-in-AB-space for fresh_ab (NOT Adam)
**Adam in ab-space — no:** its second moment `EMA(g⊙g)` is per-coordinate;
squaring a low-rank matrix is full-rank → can't live in ab-space (full Adam only
fits the tiny `b_only` LoRA path).
**Muon in ab-space — yes, and it matches the GRPO winner's optimizer:**
- Persistent state = **momentum only** → use the coherent **truncated seed-replay
  momentum** (low-rank), so the memory wall that kills Adam doesn't apply.
- The fresh_ab update is genuinely low-rank (Σ of 128 rank-16 outer products,
  rank ≤2048 vs module dims ≥5120). Newton–Schulz uses only rank-preserving
  matmuls → run it either **factored in ab-space** or by **transiently
  materializing one module's update** (reuse `xorl/optim/muon.py`).
- Cheap first cut: HyperscaleES `alteggroll`'s `sign(A)@sign(B)ᵀ` (≈free).
Build order: `muon_sign` (smoke) → transient-materialize `muon` (primary) →
factored ab-space (optional). Env-gated `optimizer={sgd|muon_sign|muon}`,
additive/default-off.

**FEASIBILITY VALIDATED (GPU harness, route-b):** NS-orthogonalizing a synthetic
rank-2048 low-rank `G` is correct (active SVs flatten ~0.87, tail ~0.04, stable
bf16) for 2D modules and 3D MoE (batched NS). Per-ES-step cost ≈ **4.8 s over 48
layers vs t_score ≈150 s → ~3%, negligible.** Reuse
`torch.optim._muon._zeropower_via_newtonschulz` (2D) + xorl
`_batched_zeropower_via_newtonschulz` (MoE) + `_adjust_lr` (spectral LR). No
persistent per-coordinate state → the Adam memory wall is avoided. Implementation
subtleties found: (1) per-pair score coefficients are baked into the B buffer, so
`muon_sign` must sign the *noise* with scores kept separate; (2) the 35B is MoE so
the bulk (gate_up/down) needs the 3D batched-NS path. Hook: in
`_fold_standard_module_from_tensors`/`_fold_moe_module_from_tensors`, force
single-pass so `delta` is the full per-module update, orthogonalize, then
`base_weight.add_(lr_scale * O)`.

## Execution
**B (offline first):** implement env-gated optimizer in the fresh_ab fold; unit-test
the Muon-in-ab-space update vs dense xorl Muon on a synthetic rank-2048 G;
`py_compile` + existing fresh_ab test. Then the Wordle run uses `fresh_ab`+`muon`.
**A (provision + launch):** clone the `mult` pool spec → `zorl-ar-sglang-w` (8×TP2,
Qwen3.6-35B-A3B + LoRA), match the GRPO sampler flags; launch the standalone client
(`TASK=wordle`, rollout-solve, floor-protocol probe, rank-16 `fresh_ab`,
`SAMPLE_EVAL_INTERVAL=0`); **do not share the GRPO sampler**; append launch status
to `/shared/apanda/wordle-coord/messages.jsonl`. Eval landed checkpoints on a
separate idle serve; ≥2–3 seeds; compare to 0.469 / 0.5625.

## First-launch findings (2026-06-24) — Muon ran; 2 fixable blockers surfaced
Launched ES-Wordle + Muon-in-ab-space on pool W (`OPT=muon` confirmed). Muon fold
executed (math validated separately) but step 1 surfaced two issues; run stopped,
pool W freed:
1. **Muon fold OOM** at `mem-fraction-static=0.85` (`tried 652MiB, 137MiB free`).
   The single-pass muon materializes the full per-module fp32 delta + bf16 NS gram
   on top of the 35B+KV — no headroom. The standalone harness ran on an empty GPU.
   **Fix:** re-provision pool W at `mem-fraction 0.70` (GRPO sampler uses 0.70);
   optionally also free the fp32 delta before the NS and loop MoE experts (smaller
   per-expert gram) to cut the transient.
2. **Prompt-protocol gap (the bigger one):** cold solve **1/64 = 0.0156, NEGATIVE
   reward** — below even base-one-line (0.219). `tasks/wordle.py` forces
   `enable_thinking=False` and the run used a 1024-tok budget, whereas the GRPO
   0.469 base floor used `public_reasoning_constraints_THINK` (Qwen native
   thinking) + a large budget. With thinking off + tiny budget the base produces
   mostly invalid/truncated guesses → ES has ~no solving signal (the prior-failure
   regime, re-confirmed). **Fix:** add a thinking-enabled prompt path to
   `tasks/wordle.py` (match GRPO's `_think`) + bump rollout budget (≥2048–4096),
   and verify the cold probe reproduces ≈0.47 BEFORE trusting any ES climb.

Next: (a) reproduce base ≈0.47 under the ES probe with the thinking protocol (the
gating prerequisite), (b) re-provision pool W at 0.70 with muon, (c) relaunch;
watch step-1 t_score (multi-turn rollout-solve cost is the feasibility risk —
may need fewer pairs/examples).

## Verification
Floor-eval each ES LoRA checkpoint (`eval_ckpt_generic.sh`, 64 games seed-777
`[0:64]`) + `eval_failure_taxonomy.py`; compare held-out solve to base+think 0.469
(must clear) and GRPO 0.5625 (headline); multi-seed distributions, not single runs;
never claim reproduction from rollout metrics.
