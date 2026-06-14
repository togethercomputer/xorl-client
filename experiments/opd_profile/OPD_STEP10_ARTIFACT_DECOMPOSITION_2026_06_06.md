# OPD step-10 "prefill-time-compute" signal — decomposition (2026-06-06)

**Question (user):** are we overfitting to a phenomenon that emerges within ~10
steps, and is there a simpler explanation than "filler tokens carry distillable
prefill-time compute"?

**Short answer: yes, to both.** Across every recent fla run the generation-level
pause-vs-no-pause lift is **within noise**, and the only "significant" metric the
autopilot verdict relies on (`vs_corrupt`) is an **OOD generation-derailment
artifact** whose magnitude is unstable (swings 10× across near-identical recipes).

Tool: `experiments/opd_profile/autoresearch/da_decompose.py <PTC-id | profile.jsonl>`.

---

## 1. Cross-run table (step 10, fla backend, control n=1024×3 arms)

| run | recipe | acc_pause | acc_nopause | acc_corrupt | Δ pause−nopause (z) | Δ pause−corrupt (z) |
|---|---|---|---|---|---|---|
| PTC-079 | alt-filler replicate | 0.744 | 0.727 | 0.672 | **+0.018 (z=0.90)** | +0.072 (z=3.6) |
| PTC-080 | alt-filler replicate | 0.664 | 0.657 | 0.560 | **+0.007 (z=0.33)** | +0.104 (z=4.9) |
| PTC-081 | hidden-OFF replicate | 0.789 | 0.784 | 0.098 | **+0.005 (z=0.27)** | +0.691 (z=43.9) |
| — for ref: PTC-044 orig (flashqla) | promoted | 0.705 | 0.530 | 0.627 | +0.175 (z=8.27) | +0.078 (z=3.8) |

**Finding A — the actual claim is unsupported.** "Prefill-time compute" means the
filler should let the model compute more and answer *better than answering
directly*. That is exactly `buffer_delta = acc_pause − acc_nopause`. It is **noise
in all three fla runs** (z = 0.90, 0.33, 0.27). The filler does nothing over a
direct answer. The big PTC-044 delta (+0.175, z=8.27) **does not replicate** on fla
(~25× smaller) — either a flashqla/fla numerical difference or a noise outlier;
either way it is not a stable effect.

---

## 2. The `vs_corrupt` margin is a derailment artifact, not compute

The verdict gate fires on `vs_corrupt` (and a teacher-forced answer-logprob
margin). But `vs_corrupt` is not a stable measure of anything — generation
diagnostics on the corrupt arm explain it entirely:

| run | corrupt acc | corrupt mean compl. tokens | corrupt cap-hit frac | corrupt repeated-numeric | vs_corrupt Δ |
|---|---|---|---|---|---|
| PTC-079 | 0.672 | 8.9 (≈ pause 9.2) | 0.002 | 0.001 | +0.072 |
| PTC-080 | 0.560 | 9.2 (≈ pause 9.1) | 0.007 | 0.001 | +0.104 |
| PTC-081 | **0.098** | **44.0** (pause 8.8) | **0.623** | **0.157** | **+0.691** |

When corrupt filler is tolerated by the decoder (079/080: ~9 tokens, same as
pause), `vs_corrupt` is small (+0.07–0.10). When corrupt filler **derails the
decoder** (081: 44 tokens, 62% hit the token cap, 16% spew repeated numerics),
accuracy collapses to 9.8% and `vs_corrupt` explodes to +0.69 (z=44). Same step,
near-identical recipe, **10× swing** in the load-bearing metric.

So `vs_corrupt` measures **how badly injecting OOD garbage tokens before "Answer:"
breaks generation/format** — a property of how fragile the current checkpoint's
decoding is — *not* whether the coherent filler performs useful latent
computation. The answer-logprob `vs_corrupt` margin (z up to 98) is the same
artifact in logprob space (corrupt mean logprob −1.43 vs pause −0.09).

---

## 3. Two simpler explanations fully account for the "signal"

1. **Filler ≈ direct answer (no compute).** `acc_pause ≈ acc_nopause` everywhere
   (Δ within noise). The model is not using the filler to compute; it answers the
   4-digit multiplication about as well (or as badly) either way.
2. **vs_corrupt = OOD-derailment.** Corrupting the filler injects garbage into
   context, which sometimes makes the decoder ramble past the answer / hit the
   token cap. That mechanical failure — not a missing reasoning substrate — is the
   "gap" the verdict reads as a positive signal.

Neither requires "distillable prefill-time compute." This is consistent with the
prior cross-model finding that the filler effect is format-specific generation
behaviour, not distillable reasoning
(`[[project_opd_filler_mechanism_mismatch]]`).

---

## 4. Meta: the autopilot verdict gate is mis-targeted

The scorecard's `weak/strong_ptc_signal` verdicts gate on `vs_corrupt` +
teacher-forced answer-logprob margin — **both dominated by the corrupt-arm
artifact** — while `buffer_delta` (the real pause-vs-no-pause generation claim) is
noise. The loop has therefore been promoting/triaging recipes largely on a
derailment artifact. If we keep scoring, the gate should be **`buffer_delta` (z≥3
sustained), not `vs_corrupt`**, and should require the corrupt arm to NOT be
derailing (corrupt mean-completion-tokens ≈ pause) before `vs_corrupt` is admitted
as evidence.

---

## 5. What PTC-088 / PTC-089 add (queued, pri 130/129; will fold in next)

- **PTC-088** (`eval_control_start_step:0`, 31 steps): is even this tiny picture
  present at **step 0 (untrained)**? If `acc_pause ≈ acc_nopause` and the
  vs_corrupt artifact already exist before any gradient step → 100% prompt-format,
  0% learned. Also: does training **lower acc_nopause** over 0→30 (manufacturing
  any delta by damaging the baseline), and does any signal **decay** past step 10?
- **PTC-089** (`learning_rate:0`): the airtight untrained reference — any margin at
  steps 0/5/10 with zero weight updates is purely harness + prompt format.

Expected (given §1–§3): both confirm the step-10 numbers are essentially the
untrained model's, i.e. the phenomenon does not emerge *from training* at all.

---

## 7. PTC-088 RESULT (step-0 + full trajectory) — delta is collapse-dominated + a confounded pause gain

PTC-088 ran with `eval_control_start_step:0`, full 31-step trajectory complete
(control rows at 0,5,…,30). This is the decisive control:

| step | acc_pause | acc_nopause | acc_corrupt | Δ pause−nopause (z) | Δ pause−corrupt (z) |
|---|---|---|---|---|---|
| **0 (untrained)** | 0.641 | **0.688** | 0.657 | **−0.047 (z=−2.25)** | −0.017 (z=−0.8) |
| 5 | 0.624 | 0.596 | 0.576 | +0.028 (z=1.31) | +0.048 (z=2.2) |
| 10 | 0.696 | 0.265 | 0.497 | +0.432 (z=21.7) | +0.199 (z=9.4) |
| 15 | 0.788 | 0.048 | 0.687 | +0.740 (z=51.4) | +0.102 (z=5.3) |
| 20 | 0.799 | **0.009** | 0.543 | +0.790 (z=61.4) | +0.256 (z=12.8) |
| 25 | 0.850 | 0.113 | 0.779 | +0.736 (z=49.3) | +0.070 (z=4.1) |
| 30 | 0.846 | 0.098 | 0.828 | +0.748 (z=51.2) | **+0.018 (z=1.1)** |

The delta has **two superimposed components** — be precise, do not over-claim:

1. **At step 0 the effect is absent** (pause is slightly WORSE, −0.047) → not a pure
   prompt-format artifact; it emerges *with training*.
2. **A catastrophic, clearly-artifactual no-pause collapse** (dominant): `acc_nopause`
   falls 0.688 → 0.009 (step 20), ~0.10 by step 30, and the no-pause arm **rambles**
   (17.7 tok vs pause 9.1 at step 30). OPD supervises only the filler+"Answer:"
   positions, so the student becomes **dependent on that scaffold** and loses the
   ability to answer directly — the EOS/OOD-fragility failure mode as a trajectory.
3. **A real pause-arm gain, best explained by FORMAT ADAPTATION**: `acc_pause` rises
   0.641 → 0.846 (+0.205). (An earlier "acc_pause is ~flat" note was an over-read of
   step 10 alone.) Most parsimonious account: the model learns to emit a clean answer
   in the `filler+"Answer:"` format — the *same* adaptation that collapses `acc_nopause`
   (it over-fits to the scaffold and rambles without it). Both arms move for one format
   reason; neither requires latent compute. Alternatives (filler-as-compute, train/eval
   leakage) are not separately verifiable from logs — the control eval logs only
   progress, not eval problems — but are not load-bearing: PTC-089 (lr=0) shows the
   whole effect is training-induced. NOT, on this evidence, demonstrably "prefill-time
   compute." A fair no-filler-trained baseline would attribute it definitively.

**So `pause − nopause` is not a valid measure of prefill compute**: the arms are
trained unequally (pause trained, no-pause destroyed), so the delta conflates a
baseline-collapse artifact (dominant) with a confounded in-format gain. Attributing
the pause gain needs a **fair no-filler-trained baseline** (same OPD objective with
"Answer:" but no filler, and/or a no-pause-trained control) + a verified held-out eval.

**Unified picture across runs — every "significant" metric is a training-induced
fragility or a confound, not demonstrated compute:**
- PTC-081 (hidden-off): the *corrupt* arm derails → `vs_corrupt` inflated.
- PTC-088 (helpful filler): the *no-pause* arm collapses → `pause−nopause` inflated;
  `vs_corrupt` here is unstable (step 20 z=12.8 → step 30 z=1.1) — same noise.
- The pause-arm accuracy gain is real but **unattributed** (needs the fair baseline);
  no run yet isolates filler-compute from in-format training + baseline collapse.

**The autopilot scored PTC-088 step-10 `strong_ptc_signal` *because* the baseline
collapsed** (its pause-vs-nopause gate passed). The verdict gate literally rewards
baseline destruction. (§4 meta-finding, confirmed.)

**PTC-089 (lr=0) — CLOSING CONTROL, confirmed.** With zero weight updates the run
is the untrained model at every step (weights never move), so its step-0 control is
the whole answer: acc_pause 0.626 / acc_nopause 0.670 / acc_corrupt 0.649,
Δ pause−nopause = **−0.044 (z=−2.08)** — pause slightly WORSE than no-pause, no
corrupt margin, no logprob margin, all arms ~10 tokens (no derailment). This is
**identical to PTC-088's untrained step-0** (Δ −0.047). Autopilot correctly scored
it `science_reject` (no signal). **Conclusion: the +0.75 delta in PTC-088 is 100%
training-induced — lr=0 produces no lift whatsoever.** The "prefill-time compute"
signal is entirely a product of OPD training reshaping the model (dominated by
no-pause baseline collapse + a confounded in-format pause gain), not a property the
filler confers at inference.

## 8. FAIR NO-FILLER BASELINE (PTC-091, the missing control) — filler buys NOTHING

Ran the control the review called for: train the student to answer DIRECTLY (no
filler — `student_prefill_count=0`, all custom-pair weights 0, `positive_answer=0`,
`supervise_student_cot=true` = base on-policy distillation), 31-step trajectory,
vs PTC-088 (9-symbol filler). Per-step `eval/accuracy`:

| step | filler (PTC-088) | no-filler (PTC-091) |
|---|---|---|
| 0 | 0.484 | 0.516 |
| 4 | 0.414 | 0.758 |
| 10 | 0.570 | 0.766 |
| 16 | 0.648 | 0.805 |
| 24 | 0.844 | 0.852 |
| 28 | 0.859 | 0.805 |
| **mean step≥10** | **0.715** | **0.768** |

**The no-filler baseline converges to the same accuracy — faster, and slightly
higher on average.** The filler-trained model is *slower* early (busy learning the
scaffold format); both reach ~0.80–0.85. So the `acc_pause` gain in the filler runs
is **task-learning + format-adaptation reachable with zero filler**, NOT distillable
prefill-time compute. This **definitively closes H1** — the filler provides no
compute benefit and if anything slows convergence. (Config note: a truly empty
buffer is only valid with all custom-pair weights 0 + `positive_answer=0`; any of
those >0 makes the OPD pipeline require a non-empty buffer. Per-step accuracy is
noisy; clean confirmation = PTC-091 step-30 control n=1024 + PTC-092/093 lr-sweep.
Stack note: ran crash-free after the §7c sglang-0 sampler refresh cleared the stale
Mooncake registry.)

## 9. CAN A BETTER OBJECTIVE MAKE THE FILLER WORK? (user #4) — NO; pushing harder backfires

Tried to strengthen the "filler encodes the teacher's CoT computation" objective. Both
levers FAIL, monotonically worse the harder you push (eval/accuracy mean step≥10):

| config | mean acc | vs no-filler 0.771 |
|---|---|---|
| filler + hidden-match=2 (standard) | 0.715 | −0.056 |
| filler + hidden-match=8 (PTC-098) | 0.654 | −0.117 |
| filler + buffer-only (PTC-099, supervise ONLY the buffer) | **0.290** | **−0.481** |

Forcing the student's filler hiddens to match the teacher's post-CoT hiddens harder
(hm=8), or supervising ONLY the filler so it must carry everything (buffer-only),
both DEGRADE the model rather than installing useful compute — buffer-only catastrophically.
So no objective in this family makes the filler substitute for the CoT; the harder you
constrain the filler to do the teacher's work, the worse generation gets. Combined with
the fair no-filler baseline (§8: no-filler matches/beats filler with ZERO filler), the
verdict is firm: **the filler buys nothing and cannot be made to.** Still pending: filler
LR sweep (1e-6/5e-7), longer training (step-60), filler-position control.

## 6. Bottom line for the research program

The "OPD prefill-time-compute" effect, as currently measured, is **not a
reproducible generation-level lift**. Before investing more filler-type sweeps:
re-target the metric to `buffer_delta` with a non-derailing corrupt control, get
the PTC-088/089 step-0/lr-0 baselines, and treat any future "signal" as real only
if pause beats *no-pause* (not just corrupt) on generation accuracy, reproducibly,
above the untrained baseline.
