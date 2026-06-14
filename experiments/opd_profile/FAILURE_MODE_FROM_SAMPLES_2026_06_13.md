# OPD/OPRD failure-mode diagnosis from samples + code + RiM (2026-06-13)

Triangulated from the three angles the user asked for (metrics ↔ code ↔ samples), plus the
`~/xorl-rim-repro` comparison. Companion to `autoresearch/CANONICAL_SCIENCE_RUNBOOK.md` §1e / `ARITH_OPSD_STATUS_2026_06_11.md` §6.

## 0. TL;DR

The wall is **computational depth, not the objective.** On ops6 nested arithmetic (6 serial
binary reductions per problem), a single forward pass — pause buffer or not, any distillation
objective — collapses to **magnitude/leading-digit regression toward the conditional mean of the
answer distribution.** Proven directly from samples: answers with `|gold| ≥ 10000` are solved
**0 times across every step of every run**; `gold = 0` alone accounts for **30–67% of all
"correct" answers.** Every objective (reverse-KL, forward-KL trace, hidden-MSE/OPRD, SFT, any
coef/temperature) hits the same ceiling because none of them gives the buffer a *serial
computation* to perform — they supervise the buffer with a vacuous target, the CoT prose trace,
or the answer's magnitude statistics. The one mechanism with a positive prior for forcing serial
latent computation — RiM's grounded memory tokens — has **never been tested on a non-saturated
task**, and our arithmetic task is exactly the testbed it lacked.

## 1. The failure mode, seen in the samples

Source: `eval_samples.jsonl` (per-step on-policy T=1.0 training samples, scored vs AST gold). Runs:
ARITH-005 (answer-only reverse-KL warm OPD, null 0.094), ARITH-014D (OPRD coef-1, best 0.145),
ARITH-020 (OPRD c1, 201-step extend, in-flight), ARITH-007F/010 (pause↔CoT logprob trace-match).

### 1a. Format is learned fast; the value is never computed
- **Step 0** (base / warm seed at T=1.0): bimodal — a chunk emits 64-token digit garbage
  (`-220000000000…`, random digit strings); the rest emit short numbers, often near-miss
  (gold=-455→`-505`, gold=4808→`4507`).
- **By step ~8** the garbage is gone — the model has learned the *format* (short int + EOS).
  From here on essentially all outputs are short plausible numbers that are **wrong by a little**:
  gold=244→`238`, 391→`401`, -1531→`-1495`, 290→`279`, 28322→`28655`, 641136→`628247`.
  → **magnitude + leading digits ≈ right; low-order digits / carries wrong.** Approximation, not computation.

### 1b. Accuracy is entirely carried by small-magnitude / zero answers
Accuracy stratified by `|gold|` (any run, any step, n≈64/step):

| `|gold|`  | solved |
|-----------|--------|
| `= 0`     | ~100% — also **30–67% of all correct answers** (degenerate "guess 0" attractor) |
| `< 100`   | ~5–30% |
| `< 1000`  | ~0–30% |
| `< 10000` | ~0–1 of ~13 |
| `≥ 10000` | **0 / (14,15,7,…) — never, at every step of every run** |

The ~0.10–0.23 headline accuracy is "small/zero answers that can be guessed or approximated." The
problems that genuinely need a multi-digit carry-chain are categorically unreachable in one pass.

### 1c. Objective only changes *how* it fails, not *whether*
- **Answer-only reverse-KL (005):** sign accuracy decays to a coin-flip (95%→~50%); the model emits
  ever-more degenerate small/round numbers (gold=-2710→`0`, -772221→`7001`, -6162→`0`). Reverse-KL
  clamp is gradient-dead (clamp_frac→0.99, code-confirmed) on wrong-prefix positions.
- **OPRD hidden-MSE (014D):** *preserves/recovers* sign (→70–86% late) and magnitude
  (gold=10565→`5922`, 1155→`1292`) — the restorative gradient that buys +0.05 over the null — **but
  still can't get exact values**, because the teacher's *answer-position hidden* encodes answer
  magnitude/structure, not a re-runnable algorithm. This is the user's intuition exactly: minimizing
  hidden-state MSE pulls the answer toward the *average numerical token* (right scale, no computation).
- **CoT-trace logprob match (007F/010):** the student learns to imitate the CoT trace at buffer
  positions (buffer-KL converges) **and corrupts the answer** — accuracy ≤ base, and CoT prose leaks
  into the answer slot (step 20: `0` (since float is effectively integer 0).`). Imitating the trace ≠ doing the computation.

## 2. The user's four intuitions, adjudicated against the evidence

1. **"After pause pause pause the student only wants to predict pause."** — **CONFIRMED at the
   mechanism level (code).** In the default `supervise_student_cot` path the teacher's target at each
   pause position is *the teacher's own next pause token*; the KL literally trains "predict pause."
   The client docstring itself flags this as "weak / possibly vacuous." (In eval the model doesn't
   emit "pause" because the forced `</think>Answer:` suffix overrides it — the symptom is in the
   *gradient*, not the output: the buffer never receives a useful learning signal.)
   Ref: `examples/on_policy_distillation.py` `_teacher_hidden_cache_data` (mask logic ~ :759), docstring ~ :3830.

2. **"Min-MSE of hidden states → avg over all numerical tokens; maybe use random filler (but tried)."**
   — **CONFIRMED both clauses.** (a) OPRD/hidden-MSE samples are textbook magnitude regression (§1c).
   (b) Random filler *was* tried (ARITH-012/012A): buffer content is irrelevant under SFT
   (rand ≈ pause ≈ none ≈ 0.22). The token identity was never the issue — the issue is the buffer has
   no *per-step computational target*, regardless of which token fills it.

3. **"Having a CoT to match to would help."** — **REFUTED as implemented; reframed as the new bet.**
   Matching the CoT *prose trace* (logprob, 007F/010) or its *hidden states* (008/011/014D) both fail
   (§1c). What was *never* tried: grounding the buffer to the **intermediate computation VALUES** (the
   reduction tree), which is the RiM recipe (§4).

4. **"Are we correctly implementing: prefill pause, sample answer, supervise pause vs CoT in logprob
   space?"** — **NO, on two counts.** (a) The *default* path supervises pause ↔ teacher's-own-pause
   (vacuous), not pause ↔ CoT. (b) The version the user describes — pause-position-i ↔ CoT-token-i
   logprob-KL — exists only under `opd_buffer_equals_cot`/`match_cot`, and **it was tested as
   ARITH-007F/010 and is a confirmed dead end** (converges the trace imitation, destroys the answer).
   So this specific idea is already falsified; it is not a path forward.

## 3. Why depth is the binding constraint (mechanistic reconciliation)

ops6 = 6 serial reductions; each needs the *output* of earlier reductions as input (a reduction DAG).
A transformer forward pass has fixed depth and the standard pause buffer provides **no serial
dependency across buffer positions that carries a partial result** — uniform pause tokens give the
attention map nothing to route a running value through (the ledger's "attention-map worry", D). So the
network can at best learn a *parallel statistical approximation* of the answer from surface features
(operand magnitudes, op count) → magnitude regression → the exact ceiling we see. The teacher escapes
this by spending ~2600 CoT tokens of *serial autoregressive* scratch. Internalizing that into prefill
requires the buffer to actually *carry and transform partial results step by step* — which needs (i)
buffer positions with a real per-step target and (ii) a structure forcing computation to route through
them. That is precisely RiM.

## 4. RiM comparison — the untested quadrant with a positive prior

`~/xorl-rim-repro` ("Reasoning in Memory", Aichberger & Hochreiter 2605.30343): K trainable
memory-token embeddings after the question + a **block-causal forcing mask** (a readout can't see prior
written reasoning, so block *t* must *encode* step *t*) + **dense grounding to the per-step reasoning
values**. Plain weighted CE, no teacher. Results: mechanism fires at every scale (blocks vs no-blocks
+29.5pp@1B, **+22.3pp@30B**), beats SFT-no-CoT at 1B (+6.9pp) but **loses at 30B (−9.3pp) only because
GSM8K is saturated there** (SFT-no-CoT already 56%). **The decisive test — RiM vs SFT on a hard,
non-saturated task — was never run.** Our ops6 is that task: SFT ceiling ~0.23, teacher 0.96.

Critically, RiM differs from *everything* in the OPD program on exactly the axis the samples implicate:
it gives the memory tokens (a) dedicated trainable embeddings (not frozen pause), (b) a forcing mask
(serial routing), and (c) **dense grounding to intermediate computation values** (not pause, not CoT
prose, not answer hiddens). The OPD program has tried every *other* combination and they all no-op.

### The grounding signal is free and exact
For synthetic arithmetic we have an oracle: the AST yields the exact ordered reduction trace with
intermediate values — **every ops6 problem = exactly 6 clean serial steps** (validated on the pool;
e.g. `((40+(40+40))--62) % (-89-(84-56))` → `40+40=80, 40+80=120, 120--62=182, 84-56=28, -89-28=-117,
182%-117=-52`). No teacher-prose parsing needed; dense, noise-free per-step targets for all 7,892 problems.

## 5. Proposed experiment ladder (genuinely new — value grounding)

**Rung A — Numeric-CoT SFT (upper bound, cheap, existing stack).** SFT the model to autoregressively
emit the 6-step reduction trace (numbers only, e.g. `-7826;-7797;-7865;-1;-6;0` then `Answer:`) then the
answer. Tests whether the *content* (intermediate values) carries the capability. Prediction: approaches
teacher ~0.9 — establishing that the serial values are the missing signal and bounding the prefill version.
*This is not prefill-time compute (it's autoregressive scratch), but it is the necessary control: it
isolates "does grounding to intermediate values teach the algorithm" from "can it be compressed into one pass."*

**Rung B — RiM-grounded prefill memory (the real test).** Port RiM's recipe onto this task: K=8 trainable
memory blocks after the prompt, block-causal forcing mask, each block's readout grounded to one reduction
value (CE), answer read out after the last block in a **single prefill** (no autoregressive trace at
inference). `~/xorl-rim-repro` already has the Qwen3-MoE machinery (all-expert LoRA, 4-D mask verified
applied). Prediction: if it lifts the `≥1k`/`≥10k` magnitude buckets at all, it is the first method to do
so — the program's first genuine prefill-time-compute signal on a depth-limited task.

**Gating:** the decisive metric is the **per-magnitude-bucket accuracy** (§1b), NOT aggregate accuracy —
aggregate is dominated by the gold=0 attractor and will hide a real lift on the hard buckets. Compare
against the budget-matched SFT control and the teacher ceiling.

### What NOT to run (closed by this analysis)
- More OPRD coef / temperature points (§1e: exhausted, c1=0.145 is favorable-variance peak).
- Correct-prefix filtering *as a standalone lever*: §1b predicts it trains only on the easy small/zero
  bucket → will tie SFT. (Still worth it only as a *component* of value-grounded objectives.)
- CoT-prose trace matching in any space (007F/010/008/011 — falsified 4 ways).

## 6. LIVE EXPERIMENT (launched 2026-06-13 ~18:0xZ): RiM value-grounded prefill compute on ops6

Running the test RiM never got — value-grounded prefill compute on a hard, non-saturated task — on the
isolated RiM repro stack (`~/xorl-rim-repro`, NOT the OPD client, which has another agent's uncommitted
fb-capture work). Model: **Qwen3-30B-A3B** (RiM-proven mask substrate; NOT Q3.6-35B — internal
comparison, model caveat stated). Two budget-matched arms, all-expert LoRA r128 + expert r8, 4×H100 DDP:

- **RiM arm** (`k8s/rim-qwen3-30b-arith-rim.yaml`): K=8 memory blocks + block-causal forcing mask;
  stage-1 grounds each block to one **AST reduction value** (the dense per-step target the OPD buffer
  never had), stage-2 answer refinement. s1=3000 + s2=1500. Eval: K=8 blocks (prefill compute) AND
  no-blocks control (same weights).
- **SFT-no-CoT baseline** (`k8s/rim-qwen3-30b-arith-baseline.yaml`): budget-matched single stage 4500,
  same trainable surface, prompt→answer (the 0.23-equivalent for THIS model).

Data adapter: `experiments/rim/arith_data.py` + `--task arithmetic` (train) / `--arith_eval` (eval), fed
by the reduction-trace JSONs (`/shared/opd-coord/arith_ops6_*_redtrace.json`). Eval n=1024 disjoint,
exact-integer match, `--samples_out` for per-|gold|-bucket analysis.

**Decision gate (NOT aggregate accuracy — it's dominated by the gold=0 attractor):** does the RiM arm
lift the `|gold|≥1k`/`≥10k` magnitude buckets that single-pass OPD/SFT NEVER solve (§1b)? Compare
RiM-blocks vs no-blocks (mechanism fires?) vs SFT-baseline (beats the single-pass ceiling?). If yes →
first prefill-compute signal on a depth-limited task → port to Q3.6-35B. If no → prefill compute on deep
arithmetic is genuinely hard at this depth; learn the limit.
