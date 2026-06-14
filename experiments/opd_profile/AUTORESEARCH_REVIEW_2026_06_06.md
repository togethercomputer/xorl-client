# OPD prefill-time-compute autoresearch — full experiment review (2026-06-06)

Programme: `prefill_time_compute_opsd` on `er-opd-q36-35b-slots` (Qwen3.6-35B-A3B OPD
self-distillation). 89 ideas in `ideas.yaml`; **67 launched at least once**, 22 never
ran. Goal of the programme: show that `prompt + filler + "Answer:"` answers **better**
than `prompt + "Answer:"` — i.e. the filler carries distillable prefill-time compute.

> **READ THIS FIRST — the scoreboard is mostly artifact.** The loop's operational
> metric is `delta = acc_pause − acc_nopause`, gated together with `vs_corrupt` and a
> **teacher-forced** `answer_logprob` margin. The deep-analysis controls (PTC-088 step-0
> trajectory, PTC-089 lr=0, the alt/hidden-off replicates) showed these gates **reward
> two trivial training artifacts, not compute**:
> 1. **No-pause baseline collapse** — OPD supervises only the `filler+"Answer:"`
>    positions, so training destroys the model's ability to answer *directly*; the
>    delta grows because `acc_nopause` craters, not because pause improves.
> 2. **Corrupt-arm derailment** — corrupting the filler injects OOD tokens that make
>    the decoder ramble/hit the token cap; `vs_corrupt` measures decoder fragility,
>    not latent computation, and is wildly unstable (swings 10× across replicates).
>
> So of the **22 `strong_signal` verdicts**, essentially all are one or both artifacts.
> The corrected read is in §3–§5. Full decomposition: `OPD_STEP10_ARTIFACT_DECOMPOSITION_2026_06_06.md`.

---

## 1. The decisive controls (what actually settles it)

| Idea | Setup | Result | What it proves |
|---|---|---|---|
| **PTC-089** | promoted recipe, **lr = 0** | step0 `delta=−0.044` (z=−2.1), pause 0.63 / nopause 0.67 | With NO weight updates there is **no signal** → the effect is 100% training-induced, not a base-model/prompt-format property. |
| **PTC-088** | promoted recipe, **step-0→30 trajectory** | step0 `−0.05` → step10 `+0.43` → step30 `+0.75`; pause 0.64→0.85, **nopause 0.69→0.10** | The growing delta is **manufactured by no-pause baseline collapse** (nopause rambles: 17.7 tok vs 8.9). pause rises too (format adaptation / confound), but the headline delta is the collapse. |
| **PTC-061** | promoted recipe, **step-15** | `delta=+0.765` (z=55.5), pause 0.80 / **nopause 0.04** | Independent confirmation: training longer → nopause →4% → delta balloons. NOT "genuine compute revealed by longer training." |
| **PTC-079/080** | *alt* 9-symbol filler replicates | `delta=+0.018 / +0.007` (z≈0.3–0.9 = NOISE) | The pause-vs-nopause delta **does not replicate** across an equivalent filler set. |
| **PTC-081/082** | hidden-OFF replicates | `delta≈0` (z≈0.3–1.3) but `vs_corrupt=+0.65–0.69` | The "signal" survives only as the corrupt-derailment artifact (corrupt arm rambled 44 tok / 62% cap-hit in 081). |

---

## 2. Per-experiment ledger (67 launched, by phase)

Format: `id | verdict | step delta(z) [pause/nopause]`. "REJ"=rejected/science_reject,
"STR"=strong_signal, "WEAK"=weak_signal, "INFRA"=infra_invalid (crash/cap-hit, no science).

### Phase 1 — objective-formulation search (PTC-001–019): mostly REJECTED
The early loop searched OPSD loss formulations (positive filler-KL, hidden matching,
gold-answer tails, RiM-style generated memory, fixed slots). **None gave a clean
pause-vs-nopause win**; several were strongly NEGATIVE (pause *worse* than nopause):
- High-capacity positive-KL filler: PTC-007 `−0.089`, 008 `−0.086`, 009 `−0.082`, 010 `−0.105` (all step-4, z≈−4 to −5) → long positive-only filler-KL HURTS.
- Generated-memory / RiM-style: PTC-011 `+0.027`(incon), 012 `+0.020`(REJ), 013 `−0.152`(REJ), 014 `−0.008`, 015 `+0.023`, 018 `−0.004` → generated-memory carries some causal answer signal but no operational lift.
- 001/003/006 (positive-KL + hidden + gold-answer): `delta≈−0.005`, `vs_corrupt≈+0.80` → the FIRST sign the "signal" lives entirely in `vs_corrupt`, not delta.

### Phase 2–3 — the "AM" recipe + ablations (PTC-020–036)
"AM" = prompt + arbitrary short filler + "Answer:" with hidden-match + corrupt-negatives.
- **PTC-020** step5 `+0.046`(z2.1) → first positive; **PTC-021** step10 **`+0.229`(z11.2)** [0.76/0.54] → first "strong" (note nopause already low).
- PTC-022 (batch×2): `−0.033` REJ. PTC-023 (no-hidden, s5) `+0.079`(z3.6) STR. PTC-024/025 (no corrupt) WEAK/REJ → loop concluded "corrupt necessary". PTC-026 (no-hidden s10) `+0.001` → "unstable by s10".
- PTC-029/030 (high-capacity random filler) `−0.130 / −0.087` REJ.
- PTC-035 (pipeline no-hidden s5) `+0.003` WEAK; **PTC-036** (full-AM pipeline s5) `+0.063`(z2.9) → promoted.

### Phase 4 — PTC-044 promotion + single-knob wave (PTC-044, 050–062): ALL "STRONG"
**PTC-044** s10 `+0.175`(z8.3) [0.71/0.53] → the promoted recipe. Then single-knob sweeps,
**every one `strong_signal`** but with deltas ranging 0.05–0.77 and **always nopause ≪ pause**:
- hidden-coef: 050(0.5)`+0.256`, 053(1.0)`+0.132`, 054(4.0)`+0.051`.
- corrupt-answer: 055(0.25)`+0.300`, 056(0.0625)`+0.115`.
- corrupt-buffer: 057(0.5)`+0.104`, 058(2.0) INFRA.
- lr: 059(5e-6)`+0.205`, 060(2e-6)`+0.140`.
- **steps: 061(step-15) `+0.765` [0.80/0.04]** ← the collapse, undisguised.
- batch: 062(256)`+0.378`.
Interpretation: the "knob doesn't matter, it's always strong" pattern is itself the
tell — the gate fires on baseline collapse regardless of the knob.

### Phase 5 — filler-content sweep (PTC-063–087): INCOHERENT
User-requested "which filler type works." Results do **not** form a coherent pattern —
the hallmark of noise in a broken metric:
- **Inert / REJ:** 9 dots (063 `+0.006`), diff-punct (064 `−0.015`), NL "think" (065 `0.000`), neutral-9sym (085 `−0.052`), 27-uniform (068 `−0.054`), 100-uniform (070 `−0.059`), 100-rand-symbols (071 `−0.078`), 100-rand-words (072 `−0.072`).
- **"STRONG" (artifact):** orig-9sym (044/076 `+0.166`/077 `+0.310`/078 `+0.130`), 100-orig-sym (083 `+0.302`), 100-from-orig-set (084 `+0.140`), 100-rand-numbers (073 `+0.154`, 087 `+0.233`), **9-rand-numbers (086 `+0.510`!)**, 54-uniform (069 `+0.301`).
- **Self-contradiction:** 9 dots inert but **9 random numbers gave the LARGEST delta (0.51)**; 100-uniform rejects but 54-uniform is "strong"; 100-random-symbols rejects but 100-orig-symbols is "strong". The RUNBOOK's "variety matters, count hurts" story does **not** hold. These are baseline-collapse magnitudes, not content effects.

### Phase 6 — deep-analysis controls (PTC-076–082, 088, 089): see §1.

---

## 3. Hypotheses — validated / invalidated

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | Filler carries distillable prefill-time compute (`pause` beats `nopause`). | **INVALIDATED** | Δ is noise at step0/lr=0 (089: −0.04); when positive it's no-pause collapse (088/061); doesn't replicate (079/080). |
| **H2** | Hidden-state matching is necessary. | **INVALIDATED / not load-bearing** | No-hidden gave early lift (023) and replicates behave the same (075, 081/082). |
| **H3** | Corrupt-negative pressure is necessary. | **CIRCULAR** | "Necessary" only because the `vs_corrupt` gate needs the corrupt arm; corrupt shapes the artifact, doesn't prove compute. |
| **H4** | Filler content/variety drives the effect. | **INVALIDATED** | Content sweep (§Phase 5) is self-contradictory; magnitudes track baseline collapse, not content. |
| **H5** | Longer training reveals genuine compute (not an EOS/format artifact). | **INVALIDATED** | step-15 (061) and step-30 (088) make the delta BIGGER **via nopause→0.04–0.10**, not via pause-arm compute. |
| **Meta** | The autopilot's verdict gate measures compute. | **FALSE** | 22 `strong_signal` ≈ all baseline-collapse + corrupt-derailment. Gate should be `buffer_delta` with a non-derailing corrupt control, vs a fair no-filler-trained baseline. |

**One unresolved, genuinely-open thread:** `acc_pause` *does* rise with training
(0.64→0.85 over 30 steps). That is either (a) format adaptation (learning to emit a
clean answer in the scaffold), (b) real filler-compute, or (c) train/eval leakage on
the random 4-digit mults. We could not separate these — the one experiment that would
(a fair *no-filler-trained* baseline at matched steps) was never completed
(PTC-090 sync-crashed/ESCALATEd). This is the crux for any future work.

---

## 4. On "train longer" (your forward question)

Your intuition — latent reasoning shouldn't be expected in a few steps — is sound a
priori. But **we already have the longer-horizon data, and it does not rescue H1; it
amplifies the artifact.** The PTC-088 trajectory (one run, control every 5 steps):

| step | acc_pause | acc_nopause | Δ |
|---|---|---|---|
| 0 (untrained) | 0.64 | 0.69 | −0.05 |
| 10 | 0.70 | 0.27 | +0.43 |
| 15 | 0.79 | 0.05 | +0.74 |
| 20 | 0.80 | 0.01 | +0.79 |
| 30 | 0.85 | 0.10 | +0.75 |

`acc_nopause` collapses toward 0 (the model forgets how to answer without the scaffold);
the headline delta grows purely from that. So **training longer with the current
setup makes the metric look better while measuring more baseline destruction.**

**What longer training is actually worth doing — but only with the right controls.**
The reason the within-model `pause − nopause` delta is invalid is that the two arms are
*not trained equally*: OPD trains the pause format and never trains (in fact sabotages)
the no-pause format. To test whether longer training elicits real compute you must
compare against a **fairly-trained no-filler baseline**, not the same model's sabotaged
no-pause arm:

1. Train the filler recipe AND a **matched no-filler control** (same objective, same
   steps, `"Answer:"` with no buffer) for a long horizon (e.g. 50–100 steps).
2. Score both on a **verified held-out** problem set (rule out leakage) using
   generation accuracy (`buffer_delta`-style), and require the corrupt arm to be
   **non-derailing** (corrupt completion-length ≈ pause) before admitting `vs_corrupt`.
3. Real prefill-compute ⟺ filler-trained `acc` > no-filler-trained `acc` on held-out,
   sustained and growing with steps. If they converge, the pause-arm gain was format
   adaptation/leakage, not compute.

Without (1)–(2), more steps = more artifact. With them, longer training is exactly the
right next experiment.

---

## 5. Recommendations going forward

1. **Stop scoring on `vs_corrupt` + teacher-forced logprob.** Re-gate on held-out
   generation `buffer_delta` AND a non-derailing-corrupt requirement; treat any
   `strong_signal` from the old gate as suspect.
2. **Build the fair no-filler-trained baseline** (the one missing control) and run BOTH
   arms to a long horizon (50–100 steps) — this is the only way "train longer" answers
   H1.
3. **Fix/verify the eval split** (held-out vs the 8185 training prompts) so the
   `acc_pause` rise can be attributed.
4. **Harden the stack** before a long run — tonight's recurring Mooncake
   `Peer-nic-not-found` bursts (1.5–4h) repeatedly killed runs at the step-10 sync;
   long runs need either the §7c sampler-refresh hygiene or a more robust sync path.
5. **Consider the task** — 4-digit multiplication may be the wrong probe (saturated /
   leakable). A held-out, harder, non-memorizable task would make `acc_pause` gains
   interpretable.

**Bottom line:** the programme as run did **not** demonstrate prefill-time compute —
the positive results are training-induced baseline collapse + corrupt-derailment, and
they grow (not shrink) with longer training. The single honest open question is whether
the modest `acc_pause` rise is compute vs format-adaptation, which needs the fair
no-filler baseline we never got to run.
