# Wordle Science Canonical Runbook (Qwen3.6-35B-A3B)

**Last rewritten: 2026-06-25 (apanda).** This is a full rewrite. The prior body (a chronological
archive of 06-08 → 06-14 handoffs plus a "STATE OF THE SCIENCE" written before the format bug was
found) was deleted — it was confounded and stale. Git history preserves it (`git log -p
experiments/wordle/OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md`) if you need the old narrative.
Infra / how-to-run lives in the sibling **`THROUGHPUT_DEBUGGING_HANDOFF.md`**. SGLang↔xorl logprob
parity lives in **`SGLANG_XORL_PARITY.md`**.

---

## ⏩ 2026-06-30 UPDATE — read this FIRST (supersedes the 06-26 §4 k3 conclusion)

Three findings this session: **live on-policy k3 is SOLVED (it was never GDN-gated)**, the **climb
reproduces at the k3 floor** across three concurrent stacks, and **sampler-count + pipeline-RL are the
throughput levers**. Where this conflicts with §4 below, this wins.

### A. 🟢 LIVE k3 SOLVED — ~3e-4 (no-BI) / ~2.6e-4 (BI), NOT 0.055. Root cause was STALE SOURCE.
§4's "k3~0.055 is GDN-gated, accepted floor" was an **artifact of the trainer importing `xorl` from the
wrong tree** (`apanda-dev/src`) + scoring the wrong logprob surface. Fixing both drops live k3 ~200×:
- **Pin the engine source**: `XORL_SRC=/home/apanda/xorl-qwen-k3-reconciliation/src` (validated
  k3-reconciliation tree, HEAD 98eb289e + ~20 dirty live-path files), `PYTHONPATH` it AHEAD of the venv's
  editable xorl, and **assert `xorl.__file__` at startup**. Stale-`apanda-dev` sat at k3 0.003; same config
  on the pinned tree → ~3e-4.
- **Behavior-logprob alignment** (RL-correct for temp≠1): sampler returns `log_softmax(logits/0.7)`,
  trainer `--logprob-temperature 0.7` matching `--student-temperature 0.7`. (NOT `SGLANG_RETURN_ORIGINAL_LOGPROB`.)
- **R3 routing replay**: `--return-routed-experts --return-expert-logits` (decode-route expert IDs **+
  float routing weights** — both matter: static IDs-only 1.9e-4, IDs+weights 1.5e-4).
- **Live, sustained over the climb: no-BI k3 ~3-5e-4, ratio_mean=1.0, ratio_max bounded (<~7).** k3 rises
  mildly with distance-from-base (parity characterized near base) — benign, NOT divergence.

### B. ⚖️ Batch-invariant vs no-BI (the §4 "BI crash-loops" is also stale)
`--rl-on-policy-target xorl-batch-invariant` runs **stable** in the synchronous IS setup (no crash-loop this
session). It gives the **lowest k3 (~2.6e-4)** but deterministic+batch-invariant kernels → **~1.4× slower**
(step 609s vs ~490s at 4 samplers). **no-BI (stochastic flashinfer) is the practical winner** — k3 ~3e-4 is
already the floor and it's faster. BI is the "absolute lowest k3" reference. (The only BI-related crash this
session was the pipeline flush-cache issue, §D.)

### C. 🟢 THE CLIMB REPRODUCES at the k3 floor (confirms §2 — IS is the lever)
Three concurrent single-node EP8 stacks, all IS-loss from base, climbing (as of 2026-06-30):
- **no-BI** (`k3diag-head`, validated src, k3 3e-4): in-training exact **0→0.67 @ s73**, reward +0.75, still rising.
- **9dxtb** (`grpo-wq36-1n-isr3k3-head`, apanda-dev src, k3 3.6e-3): peak exact **0.80 @ s120**, reward +0.79.
  (Higher k3 climbs fine too — low k3 is the honest on-policy target, not a hard precondition for the climb.)
- **BI@8** (`k3bi-head`, validated src, k3 2.6e-4, 8 samplers): just crossed positive (exact 0.19 @ s13). Early.
⚠️ These are **NOT step-matched** (9dxtb is furthest along). The honest comparison is the **retries=0 held-out
eval at matched/peak checkpoints** — the next agent's job. §3's over-training-decline watch still applies.

### D. 🚀 Throughput levers (with caveats)
- **Scale samplers** for rollout-bound runs: BI **4→8 halved rollout (710→379s), ~1.6× step**. 3 coordinated
  edits — StatefulSet `replicas`, SMG `WORKER_URLS`, builder `SAMPLER_INDICES` (→regenerate trainer) — then full rebuild.
- **Pipeline RL** (`--pipeline-rl`) overlaps rollout with train (step-2 wall-clock 227s vs 487s serial).
  **REQUIRES `--no-weight-sync-flush-cache`** — else the flush races the bg worker's in-flight generation →
  `AssertionError: Cache flush failed` → all samplers SIGQUIT. Only helps when rollout<train (scale samplers
  to flip it train-bound). Keep IS loss (§2); pipeline's 1-step staleness is a tradeoff for from-base acquisition.
- **Capacity**: trainer = one whole 8-GPU node (anti-affinity keeps trainers apart); samplers = 2-GPU TP2
  (volcano bin-pack); SMG = CPU. **~3 stacks fit comfortably**; a 4th hits whole-node fragmentation (trainer
  Pending) then RDMA pinned-memory `Cannot allocate memory [12]` on packed nodes (we dropped a 4th pipeline stack).

### E. 🔧 INFRA REPRODUCTION — full stack handoff
Complete verified "stand up a stack" doc (source provenance, model, config, builders, the 4 manifests/stack,
the gated launch protocol, the clone workflow, 8 gotchas):
**`/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md`** (+ `STACK_launch.template.sh`, `STACK_rebuild.template.sh`).

### ▶ NEXT AGENT — start at `experiments/wordle/NEXT_AGENT_START_2026_06_30.md`
3 stacks climbing overnight. Next: watch the §3 peak/decline, checkpoint the peak, run the **retries=0 held-out
floor-eval** on peak checkpoints, and answer the open Q: **does the low-k3 (BI/validated) path climb better or
more stably than the higher-k3 (9dxtb) path on the honest held-out eval?**

---

## ⏩ 2026-06-26 UPDATE — read this BEFORE the older sections below

The format-bug fix (below) is validated and in. Since then we found **two things that reframe every
number in this runbook**, identified **the real training lever**, and opened a **parity workstream**.
Where this conflicts with older sections, this wins.

### 1. 🛑 THE EVAL RETRY-CRUTCH — all held-out numbers were inflated (MAJOR)
The floor-eval (`eval_wordle_sglang.py` via `shard_eval.py`) was run with **`--invalid-retries 2`**:
on an invalid turn it silently **re-prompted in `enable_thinking=False` (no-think) mode** up to 2× and
recovered a guess. So the held-out numbers measured "play **with** a no-think safety net," not pure
think skill. Measured clean (retries=0, the protocol that matches how training actually works):
- **base+think, retries=0 = 0.00** (0/64). The historical "**base+think = 0.469**" was ENTIRELY the
  no-think retry. The real base floor is **zero**.
- → All prior held-out absolutes (0.469, GRPO 0.55, OPSD/POPE ladder) are **retry-propped**;
  comparable to each other (all retries=2) but the absolutes are inflated.
- **The eval scripts now default to `--invalid-retries 0`** (env-overridable `EVAL_INVALID_RETRIES=2`).
  Training rollout was always retries=0 (invalid = terminal), so train and eval now measure the same
  thing. This also explains the old "in-training (~0.05) vs held-out (0.55)" gap — it was the retry.
- **Consequence:** GRPO's true gain is **large** (base 0.00 → ptmqx ~0.48 expected at retries=0), not
  the marginal +0.08 the retry protocol implied. Re-eval ptmqx/checkpoints at retries=0 for honest numbers.
  See memory `wordle-eval-retry-crutch`.

### 2. ⭐ THE LOSS IS THE LEVER — use unclipped `importance_sampling` from base
ptmqx (the "0.55 win") used the **default `--loss-fn importance_sampling`** (unclipped IS policy
gradient), NOT policy_loss. A `--pipeline-rl --loss-fn policy_loss --eps-clip 0.2` run **plateaued**
(in-training solve stuck ~0.05–0.10, reward never positive over 102 steps; held-out step-25 = 0.29 vs
ptmqx 0.55). **Flipping ONLY the loss to `importance_sampling` recovered the climb** (in-training
0.05 → 0.17–0.33, reward went positive). Mechanism (`ops/loss/policy_loss.py`): PPO clipping zeroes the
gradient once a token's ratio exceeds `1+eps`, which **caps how fast the policy can up-weight the RARE
high-advantage solve trajectories** — fine for refining a dense skill, throttling for *bootstrapping* a
sparse one from base. `pipeline_rl` (1-step-stale) compounds it. **Takeaway: from base, use unclipped
`importance_sampling`, on-policy (drop `--pipeline-rl`).** See memory `wordle-pipeline-rl-throughput`.

### 3. ⚠️ OVER-TRAINING COLLAPSE — early-stop at the peak
The IS run on the 4096 pool **peaked ~0.48 in-training @ ~step 76, then DECLINED to ~0.16 @ step 128**
— a format/validity regression (single_tag 0.96→0.75, non-dictionary guesses 3.2%→7.8%, invalid 0.50→0.81)
**while k3 stayed flat** (so NOT a KL/optimizer blowup). Cause: the unclipped IS keeps amplifying the
large `is_ratio_max` tail (20–45) on the over-confident policy (entropy collapsed to ~0.03) → it drifts
into confident confabulation past its peak. **Best checkpoint = the peak (~step 75), not the final.**
Fixes: the parity work below (shrinks the ratio_max tail) + early-stop; or a light stabilizer
(KL-to-ref / entropy floor / loose clip).

**⏩ UPDATE (2026-06-27, `isr3` = IS + the numerics-only parity recipe, full 128 steps, 4096 disjoint
pool):** isr3 behaved BETTER but NOT per the mechanism above, so the tail-amplification story is now
**partly contradicted**. isr3 climbed to a **higher, later in-training peak (~0.58 @ s119 vs ~0.48 @ s76)**
and declined only **mildly** afterward (~0.58 → 0.365 @ s128) instead of cratering to ~0.16; at s128 its
**validity held** (`valid_guess_rate` 0.82, invalid 0.59) where the earlier run's regressed (invalid →0.81).
BUT isr3's final `is_ratio_max` had **rebounded to ~19** (NOT a sustained ~3; and isr3 never ran
batch-invariant — only the trainer numerics knobs, which alone barely move the tail) and entropy still
collapsed (~0.044). **So the no-crater is real but is NOT explained by a sustained tail reduction** — the
actual differentiator is *sustained validity*, and isr3-vs-earlier may differ in more than parity. Mechanism
**unresolved**; confirm by comparing the two runs' validity/ratio trajectories (wandb `8r1ur5zb`).
**Early-stop near the peak STILL applies** (isr3 DID decline 0.58→0.365). Held-out floor-eval of s125
@ retries=0 is in progress to pin the honest number + whether held-out over-trains like in-training.

### 4. 🔧 SGLang↔xorl PARITY — DONE: k3 is GDN-gated; batch-invariant reverted
On-policy k3 should be ~0 but sat at **0.05–0.07, ratio_max 20–45** = train/inference logprob mismatch.
We turned on every parity knob (routed-experts R3 replay — had to finish the half-done wiring in
`train_grpo_wordle.py`; batch-invariant samplers `--rl-on-policy-target xorl-batch-invariant` +
`--enable-return-routed-experts`; xorl numerics `rope/activation/attention_cast`; env `NCCL_ALGO=Ring`,
`CUBLAS_WORKSPACE_CONFIG`). **Outcome (full recipe in `SGLANG_XORL_PARITY.md` FINAL OUTCOME): the TAIL
dropped (ratio_max 12→3) but the MEAN k3 did NOT (~0.055) → it's the GDN/FlashQLA linear-attention layers
(batch-invariant kernels + R3 don't cover the GDN/mamba path). k3→0 is GDN-gated, not a sampler/numerics
knob.** AND the batch-invariant samplers **crash-loop** (destabilizing → trainer NaN), so they were
**REVERTED to the stable flashinfer config**. `flash_attention_deterministic` was also removed (crashes
hdim-256). **Stable recipe = `importance_sampling` on flashinfer samplers; k3~0.055 is the accepted floor
(it didn't block the earlier 0→0.48 climb).** Remaining levers (GDN workstream): `routed_expert_logits`
(needs xorl_client upgrade) + GDN parity itself.

**⏩ 06-27 refinement (k3 agent, full detail in `SGLANG_XORL_PARITY.md`):** the tail is now root-caused to
SGLang's **Triton packed-decode GDN kernel** picking a different argmax than prefill (a SGLang
generation-vs-prefill *self-consistency* failure, NOT xorl's forward); a **Qwen3-Coder-30B non-GDN control
is clean (k3 ~0.0003)** → confirms GDN-specificity (and implies the ZORL coder-30B path shouldn't hit this
k3 issue); residual drift localized to ~layer0 GDN. **Actionable training lever:** the IS `old_logprob` is
the SGLang *generation* logprob, which is the wrong surface on GDN-decode tokens — switching it to
**fixed-sequence/prefill scoring** (or fixing the kernel) would cut the `is_ratio_max` tail (the one that
rebounded to ~19 on isr3, §3 UPDATE). Owned by the k3/GDN agent.

**Operational gotchas** (also in the infra runbook):
bounce the SMG with the samplers (else 503→NaN); `kubectl apply` resets `replicas`; single-node EP8
fallback when the 2-node can't place.

### 5. CURRENT LIVE EXPERIMENTS (2026-06-26) — single-node EP8, separate pools
Both runs fell back to **single-node EP8** (`--nnodes 1`, `data_parallel_replicate_size 1`): the 2-node
worker sat Pending on capacity and never rendezvoused. Builders `build_grpo_wq36_1node_{isr3,clophi}.py`;
configs `grpo-ep8x1node-muon-lowlr-{isr3,clophi}.yaml`.
- **`grpo-wq36-1n-isr3`** — IS loss (`importance_sampling`) + numerics-only parity, 4096 **disjoint** pool
  (`WORDLE_TRAIN_EXCLUDE`). **COMPLETED all 128 steps (2026-06-27).** Reproduced AND EXCEEDED the earlier
  IS climb: in-training peak **~0.58 @ s119** (vs earlier ~0.48 @ s76), then a **mild** decline to 0.365 @
  s128 — NOT the earlier 0.16 crater (mechanism unresolved — see §3 UPDATE; the `is_ratio_max` tail
  rebounded to ~19, so it's NOT the hypothesized tail-reduction). Checkpoints s25/50/75/100/125 + final
  (196 GB each). Floor-eval of s125 @ retries=0 running for the honest held-out number. Samplers
  `sampler-b-1..7` (flashinfer), router `wordle-grpo-smg`.
- **`grpo-wq36-1n-clophi`** — `policy_loss` **Clip-Higher** (`eps_low 0.2`, `eps_high 0.5`). Tests
  whether decoupled clipping matches unclipped IS. **STATUS (2026-06-27): built + verified-correct, then
  TORN DOWN — the cluster re-saturated (no free 8-GPU IB node; one launch hit a node-local optim_step
  CUDA-OOM on h100-097) so its 14-GPU sampler-c pool was idling. Queued to run SEQUENTIALLY after isr3
  peaks/frees its node.** Sequential is the SIMPLE path: once isr3's trainer stops, **reuse isr3's freed
  `sampler-b` + `wordle-grpo-smg`** — bounce both first (warm-cache relaunch hang + SMG-bounce rule) then
  launch the STOCK `build_grpo_wq36_1node_clophi.py` UNCHANGED (it's already wired to sampler-b/smg).
  🛑 The sampler-b wiring is a TRAP only for PARALLEL operation: to run clophi ALONGSIDE another trainer
  you must repoint it to a separate `-c` pool (recipe in the infra runbook) and `grep -E
  'sampler-b|wordle-grpo-smg\.apanda'` the manifest clean first, or you re-trigger the `RemoteDisconnected`
  sampler-sharing crash that killed both runs once.
- These supersede the "leading lever = SFT warmstart" hypothesis in the old TL;DR — the working path is
  **GRPO + importance_sampling + the format fix**, which climbs (~0.21–0.48 in-training on the hard pool).
  SFT-warmstart remains a *valid* idea (clean format → spend budget on retrieval) but is not the active line.

### 6. ▶ REPLICATION CHECKLIST (what another agent — e.g. zorl — needs to reproduce the climb)
Port from this fork (`/home/apanda/xorl-client-wordle-science-20260614`). All are necessary; #1 is the
load-bearing confound and #2 is the dominant lever.
1. **The format fix (4 parts) — the #1 confound.** `tasks/wordle.py`: `has_single_guess_tag` must count
   **5-letter** guesses (`_GUESS_RE`, ~L348), and `wordle_retrieval_reward` (~L616) must use partial
   credit + graded format (per-turn `invalid_rate` penalty + `format_weight·format_rate`, NOT the
   `−0.8·any(invalid)` cliff). `train_grpo_wordle.py`: decouple `valid_guess` from single-tag (lenient
   `extract_guesses`, last 5-letter guess plays). The `_think` user message must force `</think>`
   termination every turn. Without this ~91% of turns die on a parser bug → you measure format, not
   Wordle (this is exactly the coder-30B 0.0156 turn-2 collapse — the non-terminating-think half).
2. **Loss = `--loss-fn importance_sampling`, on-policy (NO `--pipeline-rl`).** `policy_loss`+clip plateaus
   from base (clip caps up-weighting the rare high-advantage solve trajectories). Dominant lever.
3. **`--reward-key wordle_retrieval_reward` + `--wordle-prompt-style public_reasoning_constraints_think`.**
   The reward VERIFIES each guess (legal + consistent-with-all-clues + solve), dense per-turn → unfakeable.
4. **EP8 single-node, muon bf16, `muon_lr 5e-5`, from base** (EP4 rollout fb HANGS; muon 2e-4 diverges;
   adamw OOMs EP8). Client `--lr 5e-6`.
5. **Eval at retries=0** (`EVAL_INVALID_RETRIES=0`, now the default). `--invalid-retries 2` is a no-think
   crutch that inflated every historical number; clean base+think = 0.00.
6. **Early-stop at the peak (~s75).** The IS run over-trains: 0.48 → 0.16 by s128 (format/validity regression).
7. *Optional* parity knobs (shrink the ratio_max tail only; mean k3 is GDN-floored): `rmsnorm_mode native`,
   `activation_native`, `rope_native`, `attention_cast_bf16`. NOT `flash_attention_deterministic` (crashes
   hdim-256); skip batch-invariant (REVERTED — crash-loops → trainer NaN). Full detail in `SGLANG_XORL_PARITY.md`.
8. Infra: ONE sampler pool per trainer (never shared); bounce the SMG whenever you bounce the samplers.

### Honest capability ladder (retries=0)
base one-line ≈ 0 · **base+think (retries=0) = 0.00** (was a 0.469 retry-artifact) · **GRPO+IS clean ≈ 0.48
expected** (ptmqx, re-eval pending) · candidate-scaffold-given ≈ 0.97 (inference-only ceiling, retrieval removed).

---

## TL;DR (read this first)

- **Goal:** get Qwen3.6-35B-A3B to *legitimately* play Wordle under the think contract — think
  privately, then emit exactly one public line `<reasoning>…</reasoning><guess>WORD</guess>`.
- **The one hard skill** (durable, probe-confirmed): **constrained-vocabulary RETRIEVAL.** The base
  model derives the green/present/absent constraints *correctly* but cannot retrieve a real 5-letter
  word consistent with all of them — it loops "TITAN? no… TITAN? no…" and never finds the word. A
  candidate-list scaffold lifts solve to ~0.97 (filter/pick works); no-scaffold stays low (endgame
  retrieval fails). Closing that gap is the whole problem.
- **🛑 A FORMAT BUG CONFOUNDS ALMOST EVERY PRIOR `_think`-FROM-BASE RUN (found 2026-06-25).** ~91% of
  training turns were killed at step 1 on a parser bug, so those runs' reward/solve curves measured
  **format compliance, not Wordle skill.** Every training-dynamics conclusion and most held-out eval
  numbers below the probe are **suspect until re-run with the fixed parser.** Details next section.
- **Fix is in this fork and validated at step-0** (base valid-guess rate 0.09 → 0.55). The full
  re-validation run has NOT been done yet — that is the immediate next deliverable.
- **Live now:** 2-node p2p GRPO under the fixed parser + partial-credit reward (`grpo-wq36-2n-head-j7r4n`).
  It plays now, but the climb past the format floor is slow. **Leading hypothesis for the real lever:
  SFT warmstart** from a clean-format checkpoint so GRPO spends its budget on Wordle, not on closing tags.

---

## 🛑 The format bug — what it confounds, what survives

**The bug** (`experiments/wordle/standalone/tasks/wordle.py`). With the
`public_reasoning_constraints_think` prompt on a from-base thinking model:

1. `has_single_guess_tag` counted matches of `_GUESS_TAG_RE` (`<guess\b[^>]*>.*?</guess>` — ANY
   content). The model **echoes the prompt's literal 4-letter `<guess>WORD</guess>` template** and
   lists candidate words in its CoT → ≥2 tags → `stopped_reason=not_exactly_one_guess_tag` → turn
   killed. (Of ~999 killed turns, 424 had ZERO real tags — pure ramble; the rest had 2–23.)
2. `valid_guess` REQUIRED exactly one tag, and an invalid turn ENDED the game → the model died around
   turn 1–2 of every game and never got to play the endgame where retrieval matters.

Net: step-0 base `valid_guess_rate ≈ 0.09`. **The training reward/solve curves of `_think`-from-base
runs were dominated by format death, not Wordle skill.**

**What this confounds (do NOT trust without a re-run):**
- All **training-dynamics** reads from `_think`-from-base GRPO runs. The "muon metastable divergence"
  story (attempt-13 / POPE peaking ~s25 then degrading) was **largely the format-death worsening** —
  `reason_chars` exploded 40→181 (more rambling over training → more kills), not a clean optimizer
  signal.
- Most **held-out eval verdicts**, including the headline "GRPO + retrieval reward → 0.56" and the
  POPE "scaffold-fade is not a retrieval win." The eval path (temp 0.2) may share the parser bug, and
  there is a separately-documented sample_eval format-metric bug. These numbers may be real or may be
  format-suppressed — **re-run with the fixed parser before citing them.**
- The "~0.56 is a ceiling" claim. Only two arms were ever cleanly eval'd, one slice each, all pre-fix.

**What SURVIVES the bug (still trustworthy):**
- **The retrieval diagnosis.** The PROBE RESULT was measured on the LIVE base model with no training
  (teacher endpoint, `probe_wordle_enumeration.py`): asked to enumerate consistent words for real
  mid-game states, think-mode validity was **0.18** (82% of listed words violate the given
  constraints), recall 0.33. For `divot` it derived the constraints flawlessly then looped
  "TITAN? No. TITAN? No…" and never found BIGOT/DIVOT/PIVOT; for `manes` (33 valid answers) it found
  0. **Logic intact; constrained-vocabulary retrieval is the deficit.** No format/parser in this
  path — this stands.
- **The scaffold contrast.** Candidate-scaffold → ~0.97 vs no-scaffold low: this is a prompt-input
  manipulation, robust to the training bug. Retrieval is the bottleneck; supplying the list removes it.
- **The confabulation mechanism for imitation** (directionally — see "Methods" below).
- The answer pool is the installed **`wordle-python` legal list (~4,266 words), not a small NYT-style
  list** — so rare-word retrieval is a genuine part of the difficulty.

---

## The fix (in this fork, validated step-0; full re-run pending)

Four parts, all landed in `experiments/wordle/standalone/{tasks/wordle.py, train_grpo_wordle.py}`:

1. **`has_single_guess_tag` counts 5-letter guesses** (`_GUESS_RE`, `wordle.py:348`), not any
   `<guess>…</guess>` block → the 4-letter template echo no longer counts as a tag.
2. **Rollout decouples `valid_guess` from single-tag** (`train_grpo_wordle.py`): take the LAST
   5-letter `<guess>` via `extract_guesses` (lenient); a legal, consistent guess PLAYS regardless of
   stray tags. Format is a reward signal, not a kill switch.
3. **Reward = partial credit + graded format** in `wordle_retrieval_reward` (the ACTUAL `--reward-key`,
   `wordle.py:616`). NOTE: `_wordle_reward_components` is only a logged metric, not the live reward —
   the fix has to be in `wordle_retrieval_reward`. Replaced the `−0.8·any(invalid)` cliff with a
   per-turn invalid-RATE penalty (`invalid_rate = 1 − valid_guess_rate`) and added
   `format_weight·format_rate` (default `format_weight=0.5`). The `any()` cliff also mis-credits:
   GRPO advantages are per-trajectory and **shared across all turns**, so one bad turn punished the
   good turns in the same game. Partial credit gives a smooth, monotonic gradient.
4. **The `_think` prompt must force termination on EVERY turn.** Native `<think>` alone lets a base
   model ramble past the token budget and emit nothing — and this bites on turns 2+ as a
   low-temperature repetition loop, not just turn 1 (the ZORL coder-30B run hit 0.0156 from exactly
   this; a model swap alone does not fix a non-terminating think). The per-turn user message must say
   "think briefly, close `</think>`, then output exactly one line, one short sentence."

VALIDATED: step-0 base `valid_rate` 0.09 → **0.55**; partial-credit gradient is smooth. NOT yet
validated: that the held-out solve rate climbs materially, and whether the pre-fix eval numbers
re-stand. **That re-validation is the #1 next task.**

---

## Methods tried — honest confidence levels

Read this as "what we believe and how much," not "settled results." Confidence is downgraded wherever
a claim depended on `_think`-from-base training dynamics or pre-fix evals.

| Method | What we observed | Confidence |
|---|---|---|
| **Base + think** | The think contract itself helps over one-line (early floor-protocol: 0.219 → 0.469). | **High** (also a prompt-only effect, robust to the bug). |
| **SFT on gold reasoning traces** | Confabulation: learns the FORM ("Only N words fit; X is best") but writes constraints then guesses words that violate them / non-words. The headline "SFT-48" warmstart was *also* a near-no-op (adamw lr 1e-6, loss flat) — so "SFT can't teach enumeration" is **not cleanly proven**; a properly-tuned retrieval-SFT was never completed. | **Medium** (confabulation is real and on-policy-probe-consistent; the magnitude/eval numbers are pre-fix). |
| **OPSD / on-policy distillation** (every teacher: answer-leak, public-reframe, candidate-scaffold, GRPO-teacher) | Same confabulation, worse held-out. The cleanest argument is mechanistic: you can't distill a skill the student LACKS via token-KL — the signal encoding the missing skill is exactly the part the student can't reproduce, so "match harder" = "confabulate harder"; and on the student's own rollout the correct teacher token often *agrees* (KL→0, dead signal). | **Medium-high for the mechanism, low for the exact eval deltas** (numbers are pre-fix). |
| **GRPO + verifying retrieval reward** | The reward VERIFIES each guess (legal + consistent-with-all-clues + solve) and is DENSE per-turn, so it can't be faked like an imitation target and has nonzero advantage before the model solves games. From base, EP8, muon bf16 **lr 5e-5** (lr is the key lever — muon 2e-4 diverged). **This is the only thing that produced legitimate constraint-tracking play.** | **The DIRECTION is high confidence; the specific 0.56 held-out number is SUSPECT (pre-fix, re-run pending).** |
| **OPSD-on-GRPO** (warmstart student from a GRPO checkpoint) | Once the student is already near a good teacher, OPSD *preserves* skill rather than collapsing (gentle lr stays ≥ base). Best 128-game 0.586 vs GRPO 0.539 = within noise. Defensible role: **post-GRPO preservation, not skill acquisition.** | **Low** (entirely pre-fix evals, within-noise gaps). |
| **POPE / scaffold-fade GRPO** | Faded the candidate scaffold out over the first ~12 steps. Peaked ~s25 near baseline then collapsed; transcript analysis said it never internalized candidate-consistency. **Now flagged confounded** — the scaffold mostly masked the format bug (POPE started ~0.125 vs plain ~0.008 *because* the scaffold steered output past the parser). | **Low — re-run with fixed parser before any verdict.** |

**The defensible verdict (empirical, not a theorem):** GRPO + a verifying reward is the only recipe
that produced legitimate play; CE/KL imitation confabulated. The retrieval bottleneck is real and
probe-confirmed. But the *quantitative* picture (exact solve rates, whether 0.56 is a ceiling, whether
OPSD/POPE beat GRPO) is **on hold until the fixed parser re-validates the eval harness.**

---

## Path forward (ranked)

1. **Re-validate the harness + re-baseline (do this first).** Run the fixed parser through
   `eval_ckpt_generic.sh` on (a) base+think and (b) the preserved GRPO-0.56 checkpoint. Establish
   whether the eval path shared the bug and what the *true* base / GRPO floors are. Everything else is
   gated on trustworthy numbers. The eval harness now auto-runs `eval_failure_taxonomy.py` +
   `wordle_panel.py` (behavior/strategy/optimal-play metrics; `cand_in_set_rate` is the sharpest
   single signal) — always report the taxonomy, not just exact solve.
2. **SFT warmstart, then GRPO (leading hypothesis for the real lever).** The current GRPO-from-base
   spends most of its budget learning to close tags, not to play. An SFT checkpoint that already emits
   clean format (a `SCI-WORDLE-RETRIEVAL-SFT` run) lets GRPO spend its budget on retrieval. This is the
   single most likely unlock and is the natural next experiment after re-baselining. The broad
   algo-think gold (`/shared/apanda/wordle-data/algo_think_v2_broad_*/gold.jsonl`, 15,986 turns, full
   word-list coverage, floor-eval targets held out) is the SFT data; tune the LR so the loss actually
   moves (the prior SFT-48 was a 1e-6 no-op).
3. **Larger / disjoint train pool.** Runs used `train_pool_size=512` (~11% of 4,266 legal words). The
   task supports a provably-disjoint split: `--train-pool-size 4096` +
   `WORDLE_TRAIN_EXCLUDE_SEED=777 WORDLE_TRAIN_EXCLUDE_COUNT=170` reserves the floor-eval set out of
   training. (Bumping the pool WITHOUT the exclude makes the floor eval ~96% seen and invalid.)
4. **Reward shaping for constraint adherence.** Transcript analysis (pre-fix, diagnostic) showed
   gray-letter reuse rising as runs progressed. Penalize constraint violations directly (esp. reusing
   a known-absent letter) and/or KL-to-ref to stop the feedback-tracking circuit from drifting.
5. **Reward-gated / inverted SD (AntiSD / Rebellious-Student)** — reinforce the student's OWN correct
   rollouts, distill the teacher only onto wrong ones; JSD not KL; entropy gate. Not ruled out; fork
   the loss, do NOT edit the shared engine `opd_loss.py`.

---

## Promotion / gate protocol (load-bearing — do not skip)

- **Always FLOOR-EVAL** with the full-game multi-turn harness (`eval_ckpt_generic.sh`, EP8 sync). The
  in-trainer rollout per-completion solve (0.03–0.16 at temp 0.7) badly under-reads the real held-out
  solve — never gate on it.
- **PROMOTION RULE:** a checkpoint beats the incumbent only if its 128-game floor-eval is **≥ 0.66 on
  one slice OR ≥ 0.60 replicated on two disjoint 128-game slices.** 128-game 1σ ≈ ±0.044, so a +6/128
  gap is noise and is NOT a promotion.
- **🛑 RETRIES (most important — see 06-26 UPDATE #1):** the eval scripts now default to
  `--invalid-retries 0` (the honest, train-matching protocol; no no-think crutch). The historical gate
  numbers (0.469 / 0.55 / the 0.66 & 0.60 thresholds) were measured at **retries=2** and are inflated —
  RE-GROUND the thresholds at retries=0 before using them (base+think = 0.00 at retries=0, so any real
  GRPO gain shows clearly). Set `EVAL_INVALID_RETRIES=2` only to reproduce the old retry-propped numbers.
- **Eval temperature:** `shard_eval` default temp 0.7 (`EVAL_TEMP`); historical baselines were temp 0.2.
  Always state BOTH temp and retries when quoting a held-out number.
- Read transcripts every gate. Numbers hide collapse and confabulation.
- GRPO is ~11 min/step; a 6h cap buys ~20–26 steps. Use `save_interval ≤ 5` and do not expect the
  configured `--steps`.

---

## Key assets & checkpoints (on disk)

- **Preserved GRPO checkpoint** (the "0.56" run, NOW pending re-eval):
  `…ptmqx…/server_output/weights/default/step-000020` and `step-000030` (also `GRPO_WIN_step20_0.56.txt`).
- **Broad algo-think SFT gold** (for the SFT-warmstart experiment):
  `/shared/apanda/wordle-data/algo_think_v2_broad_20260613T215318Z/gold.jsonl`
  (15,986 turns, full coverage, seed-777 floor-eval targets excluded, 0 leakage verified).
- **Model:** `Qwen/Qwen3.6-35B-A3B`, resolved snapshot
  `…/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`. Always pass
  the **resolved snapshot dir** as `model_path` (an unresolved name deadlocks cross-node weight load).
- **Analysis tooling:** `/shared/apanda/wordle-sft-runs/wordle_panel.py` (behavior + reasoning +
  optimal-play + `info_regret_bits`), `eval_failure_taxonomy.py`, `eval_ckpt_generic.sh`,
  `probe_wordle_enumeration.py`.
- **Eval-path gotcha:** `convert_dcp_to_hf.py` produced GARBAGE weights for the GRPO DCP checkpoints
  (served model emitted repeated "!!!", 0/128 all-malformed). Use the **EP8 `eval_ckpt_generic.sh`
  sync** path (matches the 8-shard DCP, ~6 min) — do not trust the DCP→HF conversion for these.

---

## Live run + how to reproduce

- **Live (2026-06-26):** see 06-26 UPDATE #5 — `grpo-wq36-1n-isr3` (IS, **single-node EP8**) climbing on
  the 4096 disjoint pool; `grpo-wq36-1n-clophi` (policy_loss Clip-Higher) launching on its OWN sampler-c
  pool. The 06-25 `j7r4n` and the `policy_loss` siblings are DONE/superseded — the loss-lever finding
  moved the line to `importance_sampling`. Generation via `wordle-grpo-smg`; samplers
  `wordle-sci-opsd-sampler-b-1..7` run **flashinfer** (batch-invariant was REVERTED — crash-looped → NaN; b-0 skipped).
- **Build/launch + full infra recipe:** `THROUGHPUT_DEBUGGING_HANDOFF.md` and the single-node builders
  `/shared/apanda/wordle-sft-runs/build_grpo_wq36_1node_{isr3,clophi}.py`. Configs
  `/shared/apanda/wordle-sft-runs/configs/grpo-ep8x1node-muon-lowlr-{isr3,clophi}.yaml`. (2-node builders
  `…_2node_*` exist but fell back — worker Pending on capacity.)
- **Science knobs:** `--reward-key wordle_retrieval_reward`,
  `--wordle-prompt-style public_reasoning_constraints_think`, muon bf16 `muon_lr 5e-5`, from base.
  `grpo_rl_shim.py` ports the absent `xorl_client.rl`. Topology is **EP8** (EP4's rollout
  forward_backward HANGS — alltoall "pair closure" / deepep "CPU recv timeout").
- Default-off flags preserved so the proven path is untouched: `--wordle-scaffold-fade-steps` (POPE),
  `--resume-from-step`, `--student-generation-workers`.

---

## Where things live + coordination

| What | Home | Path |
|---|---|---|
| Harness + science (this runbook, trainers, eval, probes) | `xorl-client` `apanda-dev` | `experiments/wordle/` |
| This science fork (live work) | branch `science/wordle-retrieval-sft-20260614` | `/home/apanda/xorl-client-wordle-science-20260614` |
| Engine (imported as `xorl`; p2p/Mooncake) | **`xorl-apanda-dev`** src on PYTHONPATH | `/home/apanda/xorl-apanda-dev/src` |
| Compiled deps venv (borrowed) | `xorl-internal` | `/home/apanda/xorl-internal/.venv` |
| Run artifacts, builders, configs, analysis | weka `/shared` | `/shared/apanda/wordle-sft-runs/`, `/shared/apanda/wordle-data/` |

**Coordination with the throughput agent** (`/shared/apanda/wordle-coord/`): before a training launch,
read `PROMOTED.json` and use its `config_path`; overlay only science env (`--reward-key`, prompt
style, gold data). Append launches/gates/blockers to `messages.jsonl` and poll it.

---

## Hard rules / constraints (do not violate)

- Write outputs ONLY under `/shared/apanda/wordle-sft-runs` (and `/shared/apanda/wordle-data`).
- NEVER share a sampler between two trainers (each full-weight-syncs its own policy → crashes it).
- NEVER touch another experiment's shared stack (the `er-opd-q36*` / coder runs, etc.). Only delete
  pods/jobs YOU created this session.
- Node exclusions: the preemptive banned-node list (`research-common-h100-005/-050/-080/-089/-113/-116/-118`)
  was **CLEARED by the user 2026-06-26** (ample capacity). Schedule anywhere; only reactively avoid a node
  that actually hardware-faults (import preflight fail / "no accelerator available" / DeepEP-NCCL startup timeout).
- Do NOT break the shared venv (`/home/apanda/xorl-internal/.venv`) or the shared engine loss
  (`xorl-apanda-dev` / `xorl-internal` `opd_loss.py`). Fork losses; never edit the shared one.
- `/shared` (473T weka) is contested and has hit 100% — use `save-interval 25` and prune old
  checkpoints (196GB per DCP).
