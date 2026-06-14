# OPD Prefill-Time-Compute — Canonical Runbook

> **SUPERSEDED (2026-06-09) by the canonical pair:** operations → `CANONICAL_INFRA_RUNBOOK.md` (clean operate-the-machinery procedure for the current merged DeepEP+triton 2×TP=2 stack); science → `CANONICAL_SCIENCE_RUNBOOK.md` (findings + eval methodology). This file remains the deep historical ledger; consult it for detail the pair doesn't carry.

**Consolidates** the operational `RUNBOOK.md`, the full experiment ledger (PTC-020 → PTC-121 on 4×4; PTC-300→302/302L/302N on 5×5), the cross-model fleet eval, the RiM repro, and the infra/teacher/sync/dataset docs (full index in O11).
**Stack:** `er-opd-q36-35b-slots`. **Originally** Qwen3.6-35B-A3B self-distill, 4-digit×4-digit. **As of 2026-06-09 REPOINTED to Qwen3.5-35B-A3B, 5-digit×5-digit** (see HANDOFF + S11). **Last updated:** 2026-06-09.

**Navigation — two parts:**
- **PART I — SCIENCE** (S1–S11): what the research found + how to measure it correctly.
- **PART II — INFRASTRUCTURE & OPERATIONS** (O1–O12): how to run the loop + the stack.

---

## HANDOFF (2026-06-09) — read this first

**One-line state:** the trainer-side **all-layer OPRD** machinery (the "manufacture a load-bearing buffer via training" approach = the S10h K=C ablation realized) is **built, validated end-to-end under packing, and running on Q3.5-35B-A3B 5×5** — but the early "buffer is load-bearing" reading was the **documented false positive** (S4#2), so the live experiment is now the **decisive no-filler control** (does 5×5 no-filler OPSD plateau high like 4×4, or low?).

**What this session did:**
1. **Repointed the stack** Q3.6/4×4 → **Q3.5-35B-A3B / 5×5** (5×5 is the runbook's "direct-fails" regime, base ~0.195; 4×4 was too easy → buffer moot, S6). Generator `MODEL_PATH` + config `qwen3_5_35b_a3b_opd_opdb_8node.yaml`. Data: `/shared/opd-coord/randnum_5digit_q35_le6000_{prompts,cot}.json` (5179 ex, C 1143–6000).
2. **Built + validated trainer-side all-layer OPRD** (S10h done; full build + 7 bugs fixed in **O12**). `opd_oprd_num_layers=40`, `opd_oprd_loss` reported, no OOB/desync/compile-wedge under multi-sample packing.
3. **Science (S11):** all-layer OPRD at coef=100 over 16→41 steps showed `buffer_delta` rising to z=2.4 — **but the absolutes expose it as the S4 confound**: acc_pause 0.10→0.13, acc_nopause 0.06→**0.047** (the starved one-armed no-pause arm collapsing), BOTH below base ~0.195. **Not** a load-bearing buffer. Also: our 5×5 runs are ~20× **under-trained** vs the protocol that reached 0.88 on 4×4 (PTC-118, 128×101).
4. **Conceptual reframe (S11):** the 4×4 no-filler 0.88 IS real latent reasoning — but *amortized/parametric* (CoT baked into weights, fixed single-pass depth), NOT *test-time-adaptive* prefill-compute. The buffer only has a job where single-pass latent capacity hits a CEILING. So the decisive measurement is the **no-filler plateau on a direct-fail task**.

**KEY RESULT — `PTC-302N` no-filler control DONE (5×5, 64×81, lr=3e-6):** 5×5 no-filler OPSD **plateaus LOW** — rose 0.19 (base) → ~0.33 in ~10 steps, then **flat for 70 steps** (mean 0.29 over steps≥40); step-80 control eval n=128 **`acc_nopause=0.375`**, `empty_frac=0`. **vs 4×4's 0.88 and the teacher's ~0.99.** So single-pass latent capacity is **insufficient on 5×5** → there is a real **0.37→0.99 gap a buffer could fill** → unlike 4×4, the buffer is **NOT automatically moot on 5×5**. This is the green light: an *adequately-trained* buffer arm whose absolute `acc_pause` beats **0.375** (NOT the delta) would be the first un-confounded prefill-compute win. (Caveat: 64×81 ≈ 40% of the 0.88-protocol's exposures; the 70-step flat trajectory says ~0.37 is near-plateau, but a 128×101 no-filler nails the exact ceiling.)
**NEXT INFRA (in progress):** the buffer arm's K=C pause rollouts are long → a single TP=8 student sampler bottlenecks fwd-bwd. Move the student sampler to **N×TP=2 replicas behind SMG** (`dispatch`, `round_robin`) with `serial_endpoint_sync` to each — needs a generator change (samplers are hardcoded TP=8/gpu_count=8) + verify the P2P weight-sync handler maps shards to a TP=2 receiver. See O12 / `HOWTO_NOFILLER_OPSD_TRAINING.md` §3.

**Uncommitted code (working tree, 3 repos) — see O12:**
- `xorl-apanda-dev-opd-port` (branch `codex/opd-port-20260602`): `src/xorl/server/orchestrator/packing.py`, `src/xorl/server/runner/model_runner.py`, `src/xorl/models/transformers/qwen3_5_moe/modeling_qwen3_5_moe.py`, `src/xorl/models/outputs.py`, `src/xorl/ops/loss/opd_loss.py`, the Q3.5 config, candidates `PTC-300/301/302/302L/302N`.
- `xorl-client-chat-completions` (branch `apanda/retry-502-503-504`): `examples/on_policy_distillation.py` (OPRD layer-index resolve + loss_params + pipeline guard).
- `xorl-sglang-internal`: dormant OPRD hook (flag-gated off, harmless).

**NEXT STEPS (priority):**
1. Read the `PTC-302N` no-filler plateau. If low → run an adequately-trained (≥64×81) **buffer arm** (`PTC-302`-style, all-layer OPRD coef≈100) for a fair head-to-head vs the no-filler plateau. If high → buffer moot on 5×5; pivot to 6×6 / a genuinely direct-fail task (S6, S10b) or accept the negative.
2. The *un-confounded* objective is **both-arm training** (S10 / plan Exp 2): supervise both pause and no-pause arms so neither collapses → `buffer_delta` becomes trustworthy. Needs the client per-sample-K refactor.
3. OPS hazards this session (O12): trainer restart re-stales sglang-0 Mooncake P2P (`ret=-1`) → recreate sglang-0+dispatch (keep teachers warm); a CUDA device-assert kills a bare pod permanently (next deploy hangs in rendezvous → `kubectl get pods` FIRST); run OPRD eager (compile wedges in `_sfdp_init`).

---
---

# PART I — SCIENCE (what we learned)

## S1. BOTTOM LINE (the scientific answer)

**On Q3.6-35B-A3B / 4-digit-mult, no filler variant beats the no-filler (direct-answer) baseline — at any content, count, position, objective, learning rate, or training length.**

- The fair no-filler baseline trains to **~0.77 by step-30** and **~0.81 by step-100** (`eval/accuracy`). It rises with training.
- Every filler variant (visible symbols, random tokens, lorem, 10–1000 tokens, before/after the question) and even the principled **per-position CoT-distillation (`match_cot`)** at best **converges *to* this baseline with long training; none exceeds it.**
- The autopilot's `strong_ptc_signal` verdicts are a **confounded artifact** (the pause-vs-nopause delta is inflated by the *no-pause* arm collapsing, not by pause improving). **Always read `eval/accuracy`, never the verdict.**

This matches the cross-model prior (S7): a genuine distillable filler lift survives **only on a base model with a responsive (~0.4–0.5) baseline** (Q3-235B base 4×4), not on this instruct-tuned 35B where the baseline is already strong.

## S2. The question, setup & objective

**Operational goal:** does `prompt + filler + "Answer:"` answer *better* than `prompt + "Answer:"` after on-policy distillation against a CoT-informed teacher?

- Student emits filler then an answer; distilled (reverse-KL) against the teacher's answer. Optional **hidden-match** aligns student filler hiddens to teacher hiddens.
- **Headline metric = `eval/accuracy`** (greedy generation correctness), logged every `eval_accuracy_every` steps. The "control" eval (pause vs no-pause vs corrupt + answer-logprob) feeds the autopilot verdict but is **confounded** — diagnostic only.
- Data: `/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json` (prompts) + `..._cot_nonempty.json` (teacher CoT, ~228–2048 tok, median 2048).

**OPSD objective** (what the PTC autoresearch loop actually trains): teacher sees `prompt+CoT+filler`, student sees `prompt+filler` only; the KL is evaluated **only on the student's filler+answer positions** (prompt masked from the denominator). Knobs: `opd_ptc_positive_buffer_kl_weight=1.0`, `opd_mask_zero_weight_positions=true`. This is the "force the buffer to be load-bearing" framing the PTC-0xx candidates explore.

## S3. COMPLETE EXPERIMENT LEDGER

### S3a. Old AM-recipe evidence (PTC-020–062) — established the recipe family
| Knob | result (`eval/acc`) | takeaway |
|---|---|---|
| Full AM, short filler (021/044/052) | 0.65–0.70 | the promoted recipe; "strong" verdict but acc ≈ baseline |
| hidden-match coef 0.5→4.0 (050/053/054) | 0.72 → 0.66 → **0.46** | higher coef HURTS; low coef best |
| corrupt-answer 0.0625→0.25 (055/056) | 0.55 / **0.38** | too-low corrupt-answer hurts |
| lr 2e-6 / 5e-6 (060/059) | 0.56 / **0.80** | lr matters a lot; ~5e-6 good at step-10 |

### S3b. Filler-surface sweep (PTC-063–087)
- Short *varied* filler (9-symbol, alt-sym, NL "think"): **0.65–0.72**. 9 dots ≈ 0.70.
- **More filler hurts:** dots ×3/×6/×11 → 0.55 / 0.23 / 0.45.
- **Random content hurts:** rand-100-symbols 0.21, rand-100-numbers 0.20, rand-100-numbers-rep **0.15**.
- Effect tracks *symbol variety as mild format-recovery*, NOT capacity/compute.

### S3c. Artifact decomposition + fair baselines (PTC-088–102) — the pivot
- **PTC-088** filler 9-sym step-30: **0.715** (peak 0.859).
- **PTC-091 fair no-filler (lr3e-6): 0.771** ← the baseline that reframed everything.
- lr=0 frozen control (089) = no signal → effect is 100% training-induced.
- Conclusion: PTC-088's acc gain is **format adaptation**, not compute (the same adaptation that raises `acc_pause` collapses `acc_nopause` → manufactured delta).

### S3d. LR matrix (filler vs no-filler, step-30)
| lr | filler | no-filler |
|---|---|---|
| 5e-7 | 0.438 | 0.689 |
| 1e-6 | 0.538 | 0.769 |
| **3e-6** | 0.715 | **0.771** |
| 1e-5 | 0.629 | 0.635 |
| 3e-5 | (collapses) | 0.193 |

→ **no-filler ≥ filler at every lr; 3e-6 optimal; >1e-5 destabilizes.**

### S3e. Content × count sweep (step-30, `eval/acc` mean≥10)
| | 10 tok | 100 tok | 1000 tok |
|---|---|---|---|
| 9-symbol | 0.715 | 0.548 | 0.342 |
| lorem (NL) | 0.751 | 0.480 | 0.327 |
| random ids | 0.583 | 0.324 | **0.074** |

→ **more tokens hurt monotonically; natural-language > random; nothing beats no-filler (0.77).**

### S3f. match_cot — per-position CoT distillation (the principled "filler IS the CoT" test)
*New `teacher_cot_mode=match_cot`: keeps the first-K teacher CoT positions in the cache, aligns student-filler-i ↔ teacher-CoT-i via hidden-match.*
| variant | result | note |
|---|---|---|
| K-63 (116) | 0.493 | smaller cap = less signal |
| K-126 (120) | 0.423 | |
| K-200 (113, step-30) | 0.550 | bigger K mildly better |
| coef-8 (115) | **0.289** | harder match HURTS |
| K-200 step-60 (114) | 0.663, **peak 0.758** | climbs with training |
| K-200 step-100 (119) | 0.602, peak 0.781 | climbs to ~0.76 ~step-90, then plateaus |

→ best config (K-200, coef-2) **converges to ~0.76 ≈ no-filler with ~90 steps; never exceeds it.** Higher coef / smaller K underperform.

### S3g. Train-longer baselines — the decisive comparison
| run | step-30 | step-60 | step-100 (mean / peak) |
|---|---|---|---|
| no-filler | 0.771 | 0.795 | **0.810 / 0.914** |
| filler 9-sym | 0.715 | 0.80 (096) | 0.781 / 0.914 |

→ **both rise to ~0.81 with step-100; no-filler ≥ filler at every length.** The earlier "0.77" was just under-trained no-filler. Position (filler before vs after the question, PTC-100) = no effect (0.76).

## S4. Scientific conclusions (priors going forward)
1. **No prefill-time-compute lift on this model/task.** The filler is at best a format crutch the baseline doesn't need.
2. **The verdict ≠ the result.** `strong_ptc_signal`/`promote_retest` are confounded by no-pause collapse. Decide on `eval/accuracy` mean(step≥10) + the full trajectory.
3. **Single-step snapshots mislead** — fillers spike above baseline at a lucky mid-training step then decline; the *trajectory* and the matched-length comparison matter.
4. **"More" always hurts**; variety/NL is mildly less harmful than random; forcing the buffer harder (coef↑, buffer-only) backfires.
5. **To revisit the hypothesis**, change the *model* (base model w/ responsive baseline, see S7), not more filler knobs here. True `match_cot` at K=C (full CoT length) needs a per-sample rollout-prefill change (deferred).

## S5. Eval methodology + the loss≠success lesson (CRITICAL)

**Loss convergence does NOT mean success.** The 2026-05-28 run converged loss 0.26→0.019 and checkpointed cleanly but scored **0%** — it **reward-hacked via early EOS**: the OPD loss distills teacher hiddens at the student's *self-sampled* answer positions with the pause masked and no `min_new_tokens`, so the lowest-loss action is to emit `<eos>` right after the pause → zero answer positions → trivial KL → the model learns "after pause, stop" (premature-EOS everywhere).
- **Root fix = prompt format:** the trained prefix `<think>\n pause×100` has no answer cue, so even the base model produces no answer to imitate. Fix = `student_prefill_suffix="</think>Answer: "` (close think + answer lead-in); the masked filler region extends over pause+suffix. Belt-and-suspenders: teacher-force/GT the answer target, floor generation with `min_new_tokens`, add an answer-token CE term.
- **In-loop eval (the success gate, free):** `eval/accuracy` (operands parsed from prompt, comma-robust product match), `eval/empty_frac`, `eval/mean_completion_tokens`, `eval/has_digit_frac`; `eval/samples` table + per-step decoded samples; **auto-abort** if `empty_frac≥thresh` for N steps. **Gate on generation, not loss.** Healthy: empty_frac~0, mean_completion~11, accuracy ≥ base and rising. Collapse signature: loss DOWN while `mean_completion_tokens`→cap (e.g. 3e-5 lr collapse: 39-digit garbage products, lead-frac drops).
- **The in-loop with/without-pause control** (`eval/acc_pause`, `eval/acc_nopause`, `eval/buffer_delta`) is the genuine-improvement judge — but its verdict is confounded (S4); read `eval/accuracy`.
- **Faithful eval must reproduce the exact training prefix.** The `tomi` filler-RL harness defaults to `/no_think … Answer:` which does NOT match the `<think>… pause` OPD format → false 0%. Use the training format (direct `/generate` probe to a live sglang pod) or set `--prompt-format training`; tomi needs `TOMI_USE_LITE_QWEN_RENDERER=1`. Base references (4dmult 10-shot): baseline 59.4%, pause 63.6%, CoT 91.3%.

## S6. Pre-PTC history & the buffer-no-op falsification (why we are where we are)

**The encoded-reasoning hypothesis was already falsified for this recipe in 2026-05-29**, before the PTC autoresearch loop:
- Served the finished `opd-8x8-step50` checkpoint and ran with/without-pause ablations: **no_pause 82.8%, pause50/100/200/400 = 82.8/82.4/81.2/81.2 (flat)**; uncued freegen continues "pause" (0%); off-distribution 3-digit-mult/4-digit-add 99/100%. → OPD lifted *direct* arithmetic (59.4%→82.8% no-pause) and generalized — **a real result — but NOT via latent reasoning in the pause tokens.** Root cause by construction: the OPD loss masks the pause positions (only answer hiddens distilled → buffer never gets a reasoning gradient).
- `supervise_student_cot=true` (keep pause in teacher cache) + `opd_hidden_match_coef` (cosine-match student pause hiddens to teacher's) were built to fix this. Verdict over long runs: `buffer_delta`≈0 (logit-KL too weak — teacher's pause next-token is trivially "pause") and hidden-match at coef=0.5 made direct acc *worse*.
- **Headline conclusion: the buffer won't be load-bearing on a task the model can solve DIRECTLY.** 4dmult is too easy (96% single-pass) → no gradient pressure to use the pause. For encoded reasoning to emerge, the task must sit where DIRECT fails but buffer-assisted succeeds (≥6-digit mult / multi-step, base direct <~40%). 5-digit was only marginally buffer-positive (+1pp, within noise).

**Pre-PTC A–X operational ledger** (older naming; `pause_vs_nopause` gate, `buffer_delta=acc_pause−acc_nopause`):
| Run | Result | Note |
|---|---|---|
| AH | delta +0.27 z 4.06 but n=96, corrupt cap-hit 0.94 | first strong signal, too small/degenerate |
| AL | delta +0.115 z 2.41, answer-logprob +0.22 z 17 | real positive, rejected by old corrupt-control gate |
| **AM** | delta +0.094 z 4.35 @ n=1024; post-hoc answer-logprob +0.178 z 36.8 | strongest *operational* result; the AM recipe |
| AN | delta +0.039, answer-selection negative | weaker; AM raised true-answer likelihood but not correct-vs-wrong margin |
| AO | delta −0.032, no-corrupt/cache-mismatch | clean rejection of that recipe |

This is why the autoresearch loop (PTC-020+) was built and why the conclusion (no filler advantage) is now over-determined: the buffer was shown content-free in 2026-05-29, and the 2026-06-07 matrix confirmed no filler/objective/length beats the no-filler baseline.

**Scoring modes:** `pause_vs_nopause` (operational gate, current) vs `causal_control` (stricter mechanism gate — boundary-identical corrupt/shuffle + correct-vs-distractor answer selection; needed only for a prompt-specific *memory* claim, which AM never proved). **Generated-memory / RiM** is a separate branch (the student *generates* a ~100-tok memory) that needs PG/task-reward credit assignment, not supervised weight sweeps — early positives decayed. (RiM proper: S9.)

## S7. Cross-model native filler/CoT eval (the "fleet") — does the paper's effect exist at all?

The filler-token claim was evaluated **natively (no training)** across a model fleet via the SMG router (`eval_filler_fleet.py`; per-model summaries under `experiments/opd_profile/results/filler_fleet/`). This is the source of the "needs a responsive baseline" prior.

**Methodology (the crux):** two regimes per model — `raw_nothink` (soft text `/no_think`, the paper's regime, which *suppresses* the baseline) vs `chat_hardoff` (hard `enable_thinking=false` kwarg = genuine-capability baseline). A lift that exists in raw_nothink but is *beaten by the chat_hardoff baseline* is **format-recovery, not compute.** **Diagnostic: pass@1↑ without pass@8↑ = compute (distillable); pass@8↑ without pass@1↑ = search (not distillable).** Use **N≥200** (n=80 too noisy), free generation (not prefill), T=1.0.

**Tomi multimodel filler×CoT table** (N=1000, T=1.0, 100 filler tokens, CoT-aware regrader; baseline = `/no_think` + `--prefill-answer`):
| model | task/shot | baseline | best filler (lift) | CoT |
|---|---|---:|---|---:|
| Qwen3.6-35B-A3B | 4dmult-10 | 59.4% | pause (+4.2) | 91.3% |
| Qwen3.5-35B-A3B | 4dmult-10 | 55.6% | **pause (+14.9)** | 92.9% |
| Qwen3.5-397B-A17B | 4dmult-10 | 25.8% | **ellipsis (+20.1)** | 97.3% |
| all models | 4dmult-0 / arith / varcount | — | (fillers ~never help) | 84–99% |

**Patterns:** (1) **CoT beats every filler at every (model,task,shot)** — smallest margin ~22pp. (2) **0-shot 4dmult: no filler helps any model.** (3) The big raw-regime lifts are **format-recovery** (beaten by the clean baseline). **Two artifact mechanisms:** *A — anti-EOS* (397B 10-shot: baseline emits `<eos>` after `Answer:`, 449/1000 empty; fillers prevent it → +20.1pp ≈ recovered empties); *B — digit-refinement* (Q3.5-35B: only 3/1000 empty, fillers give extra attention steps that flip wrong→right digits, +9pp). Fillers help only when the failure is generation- or attention-budget-related, never pure capability.

**The ONE genuine cell** (5-model × 3-difficulty × 2-regime, N=100×K=8): **Qwen3-235B-A22B *base*, 4×4, chat_hardoff: 0.46 → pause 0.59 (+0.13 pass@1), pass@8 0.81→0.77 (−0.04)** = compute concentration, not search; matches the paper's "+13pp", cross-confirmed independently.
- **Format-recovery only:** Q3.5-35B 4×4 raw +0.18 but clean 0.85 ≫ 0.76; Q3.6-35B raw +0.09 but clean 0.84 ≫ 0.69.
- **Q3.6 responsive-zone census (2026-06-08, N=200×K=8) — ruled out at EVERY difficulty:** 4×4 chat_hardoff base 0.825 (saturated, pause +0.005); **5×5 base 0.195 (has headroom) but pause HURTS −0.02 p@1 / −0.085 p@8**; 6×6 floored (0.015). raw_nothink same (5×5 pause −0.025). No Q3.6 (task,difficulty) cell shows a genuine pause lift — where Q3.6 has room (5×5) the filler *distracts*. Confirms harder-task-on-Q3.6 is dead; the responsive target is the **model** (Q3-235B base 4×4), not task hardness. (`results/filler_fleet/q36-responsive-census/`.)
- **Instruction-tuning kills it:** Q3-235B-**Instruct-2507** chat_hardoff baseline 0.73 (RLHF-saturated), pause −0.02 — the paper's own variant doesn't reproduce.
- **397B BF16 saturated:** baseline 0.98–0.99 everywhere, pause +0 pass@1.
- **Unifying pattern:** a distillable lift needs the rare intersection of (a) large model, (b) NOT instruction-tuned, (c) baseline in the **responsive zone ~0.4–0.5** (not saturated/floored). Only Q3-235B base 4×4 qualifies. (`OPD_FILLER_FLEET_FINDINGS_2026_06_01.md`, `encoded_reasoning_final_reconciliation_20260601.md`.)

## S8. Can the native 235B lift be *distilled into a buffer*? (235B OPD, Config A/B) — NO

The lift exists *natively* in Q3-235B base (S7) — can OPD train it into a buffer encoding? **Tested 2026-06-02 (`er-opd-235b-clean4d`); both configs NEGATIVE:**
- **Config A** (`opd_supervise_buffer_only=true`, answer masked → buffer-only channel, the unambiguous test): losses fall but **`acc_pause` drops 0.396→0.333 / 30 steps** (`acc_nopause` pinned ~1/96 → positive `buffer_delta` is artifactual).
- **Config B** (`opd_supervise_buffer_only=false`, supervise buffer+answer): no-pause climbs 0.104→0.365 (answer supervision works) but pause advantage collapses → **`buffer_delta=−0.010`**; pause ends *worse* than step 0.
- **Pattern:** objective optimization decouples from answer behavior (losses↓, accuracy↓). Post-CoT teacher hiddens are unnatural targets for the student buffer context. The paper's own RL section reports the same (filler RL lifts pass@8 not pass@1 = search). **So the native 235B-base lift is NOT trainable into a buffer** (`buffer_delta` plateaus ≈0–0.22, no emergence). WANDB: A `jzc4g717`, B `b8zejpa3`. (Proven 235B trainer config: O3.)

## S9. RiM (Reasoning-in-Memory) reproduction — same saturation verdict, different method

Separate repro (`~/xorl-rim-repro/experiments/rim/RESULTS.md`) of "Unlocking the Working Memory of LLMs for Latent Reasoning": fixed memory-block tokens `[<b><m>…</b>]` after the question, one forward pass under a **block-causal 4-D mask** that forbids readouts from attending to prior reasoning → the block must carry the computation. LoRA + memory-token embeddings; base frozen; GSM8K-Aug.

| | Llama-3.2-1B | Qwen3-30B-A3B |
|---|---|---|
| RiM (K=8 blocks) | **33.7%** | **46.7%** |
| blocks removed (control) | 4.2% | 24.4% |
| **SFT-w/o-CoT baseline** | 26.8% | **56.0%** |
| RiM vs SFT | **+6.9pp (wins)** | **−9.3pp (loses)** |
| block mechanism | +29.5pp ✓ | +22.3pp ✓ |

- **The block mechanism is REAL at both scales** (blocks ≫ no-blocks; 4-D-mask intervention selftest passes). At 30B it required LoRA'ing **all 128 experts** (attn-only froze the MoE compute → blocks *hurt*; fix = `rank_pattern={"gate_up_proj":r,"down_proj":r}` → 732M trainable, 4-GPU DDP at r≤8).
- **But RiM beats plain SFT only at 1B; at 30B it loses (46.7 < 56.0)** — a strong 30B base SFT'd to answer GSM8K directly already hits 56%, leaving no latent-reasoning headroom. **GSM8K is saturated at scale.**
- **Same lesson as OPD:** latent/encoded reasoning pays off only where the model **can't** answer directly. The decisive test (RiM vs SFT on a *non-saturated* hard set) wasn't run. Infra: run **non-privileged** (privileged exposes all 8 host GPUs → collision/OOM); 4-GPU pods schedule under contention.

## S10. Novel experiments to run next

Do **not** queue another Q3.6-35B / 4-digit filler-content sweep. That cell is answered. W&B and local profiles agree on the current anchor: no-filler step-100 (`PTC-118`, W&B `ko5kko3m`) finishes at `eval/accuracy=0.8828` with `acc_pause≈acc_nopause`, while 9-symbol filler step-100 (`PTC-121`, W&B `2x1bjcaq`) finishes lower at `0.8281` despite a large `buffer_delta` caused by no-pause degradation. The newest 1000-token runs are also negative (`lorem=0.5156`, `9sym=0.4375`, `random=0.2188`).

**Success gates for any new experiment:**
- Primary: filler-conditioned `eval/accuracy` must beat the matched no-filler trajectory on the same model/task/training budget, not just produce a positive `buffer_delta`.
- Mechanism: no-pause/corrupt collapse is not success. Require `acc_pause` itself to improve against step 0 and against no-filler, with `empty_frac≈0`, stable completion length, and no cap-hit artifact.
- Generalization: test on held-out direct-fail prompts and the full distribution. A recipe that only improves the mined training slice is not a reusable prefill-time-compute mechanism.
- Native eval: always report pass@1 and pass@8. pass@1 up with pass@8 flat means compute concentration/reliability; pass@8 up means actual capability expansion; empty-rate drops mean anti-EOS, not reasoning.

### S10a. Responsive-zone census before training

**Goal:** find a model/task cell where filler helps natively for the right reason before spending OPD training budget.

Run `eval_filler_fleet.py` at larger N on Q3-235B base, Q3.5/Q3.6 35B, and any available non-instruct checkpoints across 4×4, 5×5, 6×6, and one non-arithmetic multi-step task. The target zone is no-filler pass@1 around 0.25-0.55 with nonzero pass@8 headroom. Include pause, ellipsis, lorem, and fixed random-token fillers; extend the script for lorem/random if needed. Promote only cells where filler pass@1 rises by >=5pp without an empty-rate or formatting explanation. This directly tests the user's prior that Q3-235B benefits because it sits in a responsive band, while Q3.6-35B/4×4 is already too strong.

**Immediate command shape:**
```bash
python experiments/opd_profile/eval_filler_fleet.py \
  --port <router-port> --model <served-model-id> --run-id <model>-responsive-census \
  --nprob 400 --ksamp 8 --difficulties 4x4,5x5,6x6 --filler pause --workers 128
```

### S10b. Direct-fail hard mining

**Hypothesis:** the gap decays because uniform training quickly teaches the easy direct-answer route. The buffer only has a chance if training/eval is concentrated on prompts where direct answer fails but filler/CoT succeeds.

Build a mined split:
1. Sample each prompt with no filler, pause/lorem/random filler, and CoT/teacher.
2. Keep prompts where no-filler is wrong, filler or CoT is correct, and at least one resample shows pass@8 headroom.
3. Train two matched runs: filler-only path vs no-filler direct path on the same mined prompts.
4. Evaluate on held-out mined prompts and the full unmined distribution.

Success requires filler to beat no-filler on held-out mined prompts and not collapse on the full distribution. This is the cleanest test of "filler as extra compute" without asking the model to use a buffer on examples it already solves directly.

### S10c. Native-paused teacher distillation

**Hypothesis:** Config A/B failed because the teacher target was `prompt+CoT+buffer`, not the model's actual native filler computation. Distill the native behavior that works instead of an unnatural post-CoT hidden state.

For Q3-235B base first, collect pairs where `prompt+pause+Answer:` is correct and `prompt+Answer:` is wrong. Use the same base model, with pause, as the teacher distribution over answer tokens; train the student pause path by answer KL/CE only, with no hidden-match to post-CoT states. Control with an identical no-filler distillation run. If this cannot preserve the native 235B pause lift, OPD hidden-state variants are not the bottleneck.

**Gate:** pause-path accuracy on held-out native-positive/direct-negative prompts must stay above both the original no-filler baseline and a no-filler distillation baseline after 30/100 steps.

### S10d. Reward/DPO/GRPO on the filler path

**Hypothesis:** supervised KL is the wrong credit-assignment signal; the model needs sequence-level pressure that says "the answer after these filler tokens is correct."

Train only prompts rendered with filler and answer suffix, using task reward or pairwise preference:
- positive: correct answer after filler;
- negative: wrong answer after the same filler, no-filler answer, or shuffled-memory answer.

Freeze the base or use a small LoRA first, then unfreeze only if the path works. Keep no-filler as eval-only or as a separately trained matched baseline; do not co-train direct no-filler examples in the same optimizer stream, because that recreates the gap-decay failure.

**Implementation status:** core training arguments have task-reward plumbing, but the autoresearch candidate path is still OPD-supervised. Treat this as a code-change experiment, not a YAML-only candidate.

### S10e. Filler-conditioned adapter / direct-path freeze

**Hypothesis:** the filler advantage disappears because updates improve the direct route faster than the filler route. A controlled engineering test can ask whether extra token compute can help when the direct route is not allowed to move.

Freeze the base weights and train LoRA/adapters active only on filler-token and post-filler answer positions. Compare:
- no-filler frozen baseline;
- filler-conditioned adapter;
- always-on adapter with no filler.

This is not final scientific evidence by itself, because it can manufacture a conditional route. It is useful as a feasibility test: if even a filler-conditioned adapter cannot beat the frozen direct baseline on a responsive hard set, static filler tokens are unlikely to carry useful arithmetic state.

### S10f. Mechanistic hidden-state probes before another long run

**Goal:** determine whether filler hiddens contain prompt-specific computation or just formatting state.

For candidate checkpoints and native models:
- Patch filler hiddens across prompts before answer generation; answer accuracy should fall if the buffer is causal.
- Shuffle or noise only filler positions, leaving prompt and answer suffix unchanged.
- Train lightweight probes from filler positions for operands, partial products, carries, and final answer digits.
- Track whether probe accuracy grows before answer accuracy; if not, hidden-match losses are probably optimizing geometry unrelated to arithmetic.

This can be run against saved profiles/checkpoints and a live SGLang endpoint before launching another full training wave.

### S10g. Algorithmic latent-state scaffold

**Hypothesis:** arbitrary filler tokens need an easier curriculum than "match a post-CoT hidden." Give the filler positions explicit arithmetic state targets, then remove the scaffold.

For multiplication, supervise auxiliary probes on filler positions to predict partial products/carries/final digit chunks while the visible tokens remain pause/lorem/random. Anneal the auxiliary weight down and keep the final answer reward/CE. This tests whether the architecture can store useful state in filler slots at all. If it works, try replacing task-specific state labels with teacher-derived latent labels; if it fails, the fixed-token buffer is not a viable scratchpad for this model/task.

### S10h. True full-CoT alignment (`match_cot` at K=C) — ✅ BUILT + RUN 2026-06-09 → see S11

The current `match_cot` caps at a fixed K because `K <= min(CoT length)`; K≈200 converges toward no-filler and never beats it. A fair "filler is the CoT prefix" test needs per-sample rollout-prefill lengths and ragged hidden matching so each example can use its full CoT length C.

**DONE (2026-06-09):** realized as the trainer-side **all-layer OPRD** build (O12) on **Q3.5-35B-A3B 5×5** (per-sample K=C buffer + all-40-layer hidden match + KL anchor). Required code change shipped through client/cache/collator(packer)/loss. **Result (S11):** runs end-to-end, but at fixed-K small-buffer it floored (eval→0) and at K=C all-layer it produced the documented `buffer_delta` false positive (no-pause arm collapse), absolutes below base. The principled supervised-distillation ablation is no longer "deferred" — it's executed; the open question moved to the **no-filler plateau on a direct-fail task** (S11).

**Bottom line:** the next real science should either (1) find a responsive native cell and preserve that native filler behavior, or (2) switch to a non-saturated task/reward objective where the direct route cannot immediately solve the problem. More static filler surface sweeps on Q3.6-35B/4×4 are confirmatory only.

## S11. 5×5 Q3.5 all-layer OPRD — "manufacture a load-bearing buffer via training" (2026-06-09)

**Setup.** Stack repointed to **Qwen3.5-35B-A3B, 5-digit×5-digit** (base direct ~0.195 = "direct-fails" regime, unlike saturated 4×4). Self-distill: teacher = same model + privileged CoT; student = `prompt + K pause tokens + "Answer:"`. New objective = **all-layer OPRD** (On-Policy Representation Distillation, arxiv 2606.06021): per-layer normalized MSE between student and teacher residual-stream hiddens at aligned positions, composed as `L = kl_loss_weight·KL + opd_hidden_match_coef·OPRD`. Build + flags in **O12**.

**Arc of findings:**
1. **Fixed small-K (256) buffer, any objective floors.** Pure-KL, cosine-hidden (kl_w=0 decouples → eval 0), MSE-hidden, OPD baseline — all eval→0 on 5×5; 256 ≪ ~7800-tok CoT. Cosine's magnitude-blindness decouples from generation; keep KL as anchor.
2. **K=C (buffer = CoT length), last-layer match PLATEAUS** (PTC-300/301): hidden-match MSE 10.4→6, `opd_kl` flat — last-layer alone can't drive the pause buffer to the teacher's CoT-conditioned state.
3. **All-layer OPRD (every1 = 40 layers), coef=100, 16→41 steps:** `opd_oprd_raw` optimizes (0.021→0.012), `opd_kl` couples down a little (4.97→4.6), `buffer_delta` rises to **z=2.4 (significant)** and `buffer_vs_corrupt` to z=4.4 — *looks* like a manufactured load-bearing buffer.
4. **❌ It's the S4 false positive.** Absolutes (n=128): `acc_pause` 0.102→**0.133**, `acc_nopause` 0.0625→**0.047** — the rising delta is the **starved one-armed no-pause arm collapsing**, and BOTH arms sit BELOW the untrained base (~0.195). Per-position pre-norm residual MSE is intrinsically tiny (~0.02, ~700× below KL/token) because the pause-buffer and CoT residuals are close in absolute terms; the real divergence is post-norm/logit (KL's domain). **All-layer OPRD does not break the plateau.**
5. **Under-training caveat (decisive).** The 0.88 no-filler result was 4×4 at **128 prompts × 101 steps** (~12.9k exposures); our 5×5 runs were 16×41 (~656). So 0.13 is *not* a plateau — it's ~20× under-trained.

**Conceptual reframe (the useful output):** the 4×4 no-filler→0.88 (S1) IS latent reasoning — the student compresses ~2728 serial CoT steps into one ~40-layer forward pass — but it is **amortized/parametric** (algorithm in the weights, fixed single-pass depth), NOT **test-time-adaptive prefill-compute** (filler tokens buying extra serial compute per query). The buffer only has a job where **single-pass latent capacity hits a ceiling**. Therefore the decisive measurement is the **no-filler plateau on a direct-fail task**: if 5×5 no-filler OPSD (adequately trained) climbs to ~0.88, single pass suffices → buffer moot (like 4×4); if it plateaus below the teacher, there is headroom a buffer (extra latent depth) could fill. **`PTC-302N` (no-filler, 64×81) is that measurement.** **RESULT: 5×5 no-filler OPSD plateaus at ~0.37** (n=128 `acc_nopause=0.375`; rose 0.19→~0.33 in 10 steps then flat 70 steps), ≪ teacher 0.99 and ≪ 4×4's 0.88. **5×5 single-pass latent capacity is insufficient → real headroom → the buffer is NOT moot on 5×5** (unlike 4×4). The clean next test: an adequately-trained buffer arm whose **absolute** `acc_pause` beats 0.375. Gate on `eval/accuracy`/`acc_pause`, never on `buffer_delta`.

---
---

# PART II — INFRASTRUCTURE & OPERATIONS (how to run it)

## O1. The autoresearch loop (controller / autopilot / supervisor)

**Stack / paths**
```
STACK=er-opd-q36-35b-slots ; NS=apanda
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
CONTROLLER / IDEAS / CANDIDATES / SCORECARDS under experiments/opd_profile/autoresearch/
W&B: together-research/xorl-prefill-time-compute
```
**The loop:** `ideas.yaml` (queued candidates + gates) → autopilot renders each via the generator → trainer writes `opd_profile.jsonl` → autopilot scores → advances. **`autopilot_supervisor.py`** owns autopilot liveness (pidfile `logs/supervisor.pid`, polls 60s, bounded-retries recent crashes, kills duplicate autopilots). **NEVER hand-launch `controller.py autopilot`** — it races the supervisor and corrupts `ideas.yaml`. Relaunch the *supervisor* only if dead:
```bash
cd /home/apanda/xorl-apanda-dev-opd-port && setsid nohup python \
 experiments/opd_profile/autoresearch/autopilot_supervisor.py \
 >> experiments/opd_profile/autoresearch/logs/autopilot_supervisor.nohup 2>&1 < /dev/null &
```
**Liveness check (avoid the self-match trap):** match the *python* exe, exclude bash wrappers —
`pgrep -f autopilot_supervisor.py | while read p; do [ "$(basename "$(tr '\0' '\n' </proc/$p/cmdline|head -1)")" != bash ] && echo x; done | grep -c x`. A bare `pgrep -f 'controller.py autopilot'` also matches your own shell command → false positives.

**Candidate schema** (`candidates/PTC-*.yaml`): required `id, base_config, trainer_config, buffer_label, student_prefill_{text|token_ids}, student_prefill_count, teacher_cot_json_path, prompts_json_path, num_prompts, client_args`. **Verdicts:** `strong_ptc_signal`/`promote_retest`/`weak_ptc_signal` (deferred → operator promotes), `science_reject` (auto-advances), `infra_invalid`/`incomplete`/`inconclusive`. Post-run gate tooling: `analyze_slot_profile.py <profile> --min-control-n 1024 --require-answer-select-control --expected-sync-endpoints 2 --require-serial-sync …` + `audit_wandb_profile.py`.

## O2. CONFIG RULES (hard-won — these stall the loop if violated)

**`eval_control_start_step` must satisfy BOTH:**
1. **Be a multiple of `eval_accuracy_every` (=5)** — else the control eval never fires → no scorecard → run is mis-marked `infra_invalid` → **re-queue/cycle loop** (burned ~3 PTC-114 reruns).
2. **Equal the last step (`num_steps − 1`)** for any run that may earn a *deferred* verdict (strong/weak/promote). The autopilot's `--terminal-only-after-final-control` waits for a control at the final step; if control is earlier, it **stalls forever** (burned PTC-117). `science_reject` auto-advances regardless, which is why some earlier runs survived.

**Combined rule:** **`num_steps ≡ 1 (mod eval_accuracy_every)` and `eval_control_start_step = num_steps − 1`.** Safe: step-31→ctrl-30, step-61→ctrl-60, step-101→ctrl-100. UNSAFE: step-60→ctrl-59 (not %5) or ctrl-55 (not last). Verify before queuing: `cs==ns-1 and cs % eval_accuracy_every == 0`.

**match_cot:** requires `K (student_filler_count) ≤ min teacher-CoT length`. Min CoT here ≈ 228, so **K ≤ ~200** (count≈22 of the 9-token text). K>C raises `ValueError` per-prompt → crash. (Full K=C needs per-sample rollout prefill — not yet built.)

**Other gotchas:** `student_prefill_text` and `student_prefill_token_ids` are mutually exclusive (set one). `student_prefill_count` = repetitions of the text (9-symbol text → K = 9·count + 2). Trajectory runs: read `eval/accuracy`, not the verdict.

## O3. Stack, topology, launch & weight-sync

**Full-run topology (the OPD training stack, ~21 pods / ~14 nodes, `node-group=nccl`):**
- **Trainer:** 1 head + 7 workers, FSDP=64, EP=8 (`configs/qwen3_6_35b_a3b_opd_opdb_8node.yaml`).
- **Teacher:** see O4. Historically a 2/4-node xorl FSDP teacher; the faster path is an SGLang teacher.
- **Samplers (student):** SGLang TP=2 pods; FP8 receiver checkpoint when syncing FP8. Layout `spare-teacher1` = `sglang-0` + a replica on the teacher-1 node (so there are **2 sampler roles: sglang-0 AND sglang-1**).
- **Dispatch:** 1 CPU pod routing the samplers (Python `dispatch` or SMG — O4).

**The autoresearch stack `er-opd-q36-35b-slots`** is the always-warm reprogrammable variant: dispatch + sglang-0/1 + 2 teachers (teacher-sglang-0/1) + teacher-smg + trainer-head + 7 workers, driven by control files under `CONTROL_ROOT` (the generator writes per-role `run.sh`/`stop`; pods are bare with a control loop). `q36_35b_reprogrammable_slots.py` subcommands: `render-control`, `write-control` (all roles), `write-trainer-control`, `write-student-inference-control`, `write-dispatch-control`, `write-teacher-control`, `stop-control --remove-run`, `stop-trainer-control --remove-run`, `status`.

**Weight sync (P2P/Mooncake RDMA) — the validated 8× speedup (warm 127s→16s, step 583s→45s):**
| Knob | Effect |
|---|---|
| `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu` (default when CUDA) | CPU→GPU block-FP8 quant, 85s→10.7s/sync |
| `XORL_WEIGHT_SYNC_BATCH_DENSE=1` (+ MOE) | batch per-layer dense RDMA into one flush |
| conditional post-process cache-invalidation (sglang) | warm-cache survives FP8 sync, backend_init 34s→0.05s |
| `XORL_P2P_CPU_POOL_MIN_BYTES=0` | Bug-7: routes 9 tiny tensors off the buggy GPU-direct path |
| `XORL_P2P_HANDSHAKE_BASE_PORT` (→ `MC_HANDSHAKE_PORT=base+gpu_id`) | pins the Mooncake session so the receiver peer registry doesn't go stale across restarts (kills "Peer nic not found" bursts) |

**Required trainer/sender env** (the generator injects these): `XORL_P2P_CPU_POOL_MIN_BYTES=0`, `XORL_WEIGHT_SYNC_BATCH_MOE=1`, `XORL_WEIGHT_SYNC_BATCH_DENSE=1`, `XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES=134217728`, `XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES=1073741824`, `XORL_WEIGHT_SYNC_QUANTIZATION={"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]}`, `P2P_TRAINER_HOSTNAME=$(POD_IP)`, `NCCL_SOCKET_IFNAME=bond0`, and **`unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF`** (expandable-segments breaks Mooncake registration >~20 MiB). **Trainer pods that init Mooncake MUST NOT set `NCCL_IB_GID_INDEX`/`NCCL_IB_HCA`** (forces a GID path that fails with no IPoIB netdev; and `NCCL_IB_GID_INDEX` survives the container `~/.bashrc`).
- **Never skip `/prepare_weights_update` on warm syncs** (`XORL_P2P_SKIP_CACHED_PREPARE=1`) → `ret=-1` × 50 retries → 735s failure. It re-arms the receiver RDMA buffers.
- **FP8-on-wire only works when sglang serves the FP8 model.** `--fp8-sync-quantization` + bf16 sglang → tensor_map size-mismatch crash at step 0. Pair FP8 sync only with `--sglang-model-path <…-FP8>`.
- **The real low-sync lever is shard placement**, not quant: 2 shards bf16 = 2.4s; 4-shards-on-1-node bf16 = 33s (RDMA ingress contention). Spread shards across nodes. Sampling is not the step bottleneck (teacher is), so few shards is fine.

**Cold bring-up ≈ 9 min** (5 min multi-node rendezvous + imports + NCCL/IB, then ~2 min HF base load + ~2 min DCP overwrite). `load_weights_mode=skip` skips the redundant HF load (~2 min) **but** had a silent buffer-reinit bug (RoPE `inv_freq` left as garbage by `to_empty()` → gibberish, loss 14 vs 12); fixed in the server checkpoint path (PR #329 commit 709ffa9a) — validate step-1 loss matches `all_ranks` if using skip.

**Servable checkpoints:** use `TrainingClient.save_weights_for_sampler(name)` (client flag `save_hf_safetensors=true`), NOT `save_full_weights_safetensors` (404s on this build). DCP `save_state` is resume-only (not sglang-loadable).

**Proven 235B OPD trainer config** (any future 235B run, S8): **EP=8 intra-node** (`ep_intranode=true`; EP=64 = 35-min/step cross-node all-to-all) ~61s fwd_bwd; **shard=64** (shard=1 deadlocks OPD all_reduce); **packing ≤512** (adamw +18 GB at step-1); `operation_timeout` 1800→7200 (`server/launcher.py`).

## O4. Teacher subsystem (the step bottleneck)

Teacher **prefill dominates the step** (~99–206 s depending on CoT length & topology; fwd_bwd is only ~22–46 s). Three levers, in order of payoff:

**(a) SGLang teacher (validated, ~5× on 1/16 the GPUs) — the big win.** Replace the xorl FSDP teacher (uses the *training* framework for an inference task) with SGLang TP=2 replicas exposing a custom `/teacher_hidden_cache` endpoint that prefills and writes ONLY the kept positions (prompt+pause+answer, ~130/seq; the ~2k-tok CoT is masked out) straight to the shared safetensors cache in-process (no HTTP hidden transfer). Config: `teacher_backend=sglang`, `teacher_base_url=…`.
- **No representation-alignment needed:** SGLang `CaptureHiddenMode.FULL` returns the post-final-norm hidden (Qwen3MoeModel applies `self.norm` inside `self.model`; the `before_norm` override is EAGLE-only). xorl cache stores the head input + `opd_streaming_kl` does a pure `teacher_hidden @ head_weight` matmul (no norm) → also post-final-norm. **A/B gate passed: cosine 0.998** (`examples/ab_teacher_cache.py`).
- **Launch:** `--tp-size 2 --dtype bfloat16 --enable-return-hidden-states --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.85 --skip-server-warmup`. **`--chunked-prefill-size 16384` is mandatory** (default 2048 returns only the last chunk's hiddens → "N kept positions served from radix cache" error, which MISATTRIBUTES the cause — it's the chunk size, not radix). **Keep `--disable-radix-cache`** (radix serves kept positions from cache → no fresh hidden → same error on repeated CoTs).
- **Throughput:** ~3000–3400 tok/s on one TP=2 replica vs ~690 tok/s on the 32-GPU xorl teacher.

**(b) SMG router for the teacher** (load-balance prefill across N replicas): add a `/teacher_hidden_cache` **passthrough route** to SMG (`round_robin`/`power_of_two`, NOT cache_aware — opaque body; forward raw bytes, ≥600s timeout). Point `teacher_base_url` at the teacher-SMG; shrink `opd_prepare_batch_size=16 opd_prepare_concurrency=8` to fill replicas. (Spec in `smg_teacher_route.md`; load-balances necessary work, doesn't reduce FLOPs.) **SMG sampler routing:** `/health` always returns OK with 0 workers → wait for `/v1/models` to list the model before sending traffic; `cache_aware` is wrong for slow (>30s/req) models → use `power_of_two`; concurrency ≈ 340×N_shards.

**(c) Pipelined two-phase prefill (`opd_pipeline_rl=true`):** split teacher seq into Phase A `prompt+CoT+pause` (fixed → **prefetch 1 step ahead** during fwd_bwd, keeps pause hiddens, zero staleness) + Phase B `prompt+CoT+pause+answer` (on-policy, radix-serves the prefix, keeps only ~11 answer hiddens). Merge pause(A)+answer(B) caches → identical supervision, step ~52s→~25s. **This path keeps the answer on-policy.** (Note: `opd_pipeline_rl=true` uses the plain single-phase teacher cache path in the current client; the deprecated `teacher_pipeline_phase` is the explicit two-phase variant.)

**Teacher FSDP topology:** `ep_fsdp` must divide `hidden=7168=2^10·7` → use **2/4/8 nodes** (ep_fsdp 2/4/8); **6 and 7 crash** in `fully_shard(experts)`. 2-node teacher needs `engine_connect_host: 127.0.0.1` (else 300s rank0-address wait) and the teacher-master launcher must pass `--server.model_path`/`--server.tokenizer_path` (else HF-name broadcast-resolve path deadlocks vs workers).

## O5. GDN linear-attention backend (`fla` vs `flashqla`)

Qwen3.6-35B-A3B has `GatedDeltaNet` linear-attention layers. `XORL_GDN_BACKEND` selects the chunk kernel:
- **`fla` (default, Triton, tilelang-free):** always works. **USE THIS.**
- **`flashqla` (TileLang, Hopper-only):** fwd 4.5–5.6× / bwd 1.3–2.2× faster *on the kernel*, but: (1) needs a specific tilelang wheel (currently **BROKEN** in the venv — `libtvm_compiler.so: undefined symbol _ZN3tvm3ffi9ReprPrintERKNS0_3AnyE`); (2) **silently falls back to FLA when `ulysses_parallel_size>1`** — and the OPD config sets ulysses=8, so flashqla gives **0 speedup** even when it loads. End-to-end win is small anyway (GDN is a subset of layers). **Keep `fla` until tilelang is repaired AND ulysses=1.** `PTC-044` ran on flashqla; the whole PTC-052+ wave switched to `fla` after a deterministic step-0 `exitcode 127` crash.

## O6. CoT datasets (`/shared/opd-coord/`)

Each file is a JSON list of `{prompt, cot, finish_reason, tokens}`. Prompt sets are pairwise-disjoint 4-digit×4-digit pairs: `randnum_4digit_8192{,_v2..v5}.json`, `..._81920_v6.json`, unions `..._16384_combined`, `..._32768_combined`, `..._122880_combined` (122,880 unique pairs total). **CoTs:** the matching `..._cot_mt8192.json` (Q3.6-35B-A3B teacher, BF16 TP=2, mean ~2728 tok, ~1.2% truncation). **Always use the `mt8192` CoT files**, never the old `_cot.json` (mt=2048, amputated reasoning, 80% truncation). The current autoresearch loop uses `randnum_4digit_8192_nonempty_cot.json` (min CoT ~228, median 2048). Other teachers' CoTs exist (Q3-235B base: mean ~518 tok; Q3.5-35B-A3B 5-digit: `..._mt16384`, 0% trunc). Empty-CoT entries hard-abort the trainer → filter prompts+CoT at the same (index-aligned) indices. To add a teacher: TP = `ceil(2·params_GB/80)`, mem-frac 0.85 (<50B), SMG router. Index: `runbooks/cot_dataset_index.md`.

## O7. Failure modes + recovery

- **Transient 504** on the weight-sync HTTP path (`Bulk SGLang endpoint weight sync failed`, `curl 504`): a single/sparse 504 crashes the attempt (rc=1) → **supervisor bounded-retry relaunches it → it recovers.** RIDE IT OUT; do NOT force a stack restart for a single transient (happened ~3× in one night, all self-cleared). Only escalate if it recurs across all retries (ESCALATE line).
- **Recurring sync 504 / peer-nic bursts / stuck `weight_sync_group`** (a real wedge): the trainer crash/restart churn leaves stale sampler state (paused-gen, stuck P2P group, dispatch circuit-breaker). **Recreate the inference stack together:** `controller.py launch --id <PTC> --restart-inference` → `stop-control --remove-run` (all 14 roles incl. **both** samplers sglang-0 *and* sglang-1, dispatch, teachers, trainer) → `write-control` recreates all → wait ready. Pause the supervisor first (no race). Teachers reload (~10 min, mem-frac-static **0.8** is the proven value). The full "everything created together" recreate avoids the partial-state hazards. Belt: POST `/complete_weights_update` (`group_name=weight_sync_group`) to clear a stuck group before re-registration. See `[[project_opd_inference_stack_restart_recipe]]`.
- **P2P handshake pin:** `XORL_P2P_HANDSHAKE_BASE_PORT` (O3) — re-port it if adopting the apanda-dev merge (it's in `src/.../p2p.py` on `baf6dcce`).
- **flashqla/tilelang is broken** (O5). Keep `gdn_backend: fla`.
- **Clean-idle end-state:** when the queue drains to only gate-blocked ideas, the supervisor logs "queue drained … idle, not relaunching" and stays quiet. **This is correct — do NOT manufacture confirmatory busywork.** The mission is complete; the right next step is a new research question (S10).
- **Rapid relaunch** → ZMQ `Address already in use` (orchestrator port held by leftover proc; wait for 0 pods +60s) and `UnexpectedAdmissionError` (sampler node grabbed by another tenant; `nodeName` bypasses the scheduler — pick a node with REAL free GPUs across ALL namespaces, "0 apanda pods" ≠ free). **sglang OOM = node co-location** (teacher FSDP grabs all 8 GPUs, then sglang OOMs and exits 0 → looks "Completed"); pin samplers to distinct truly-free nodes.

## O8. Performance knobs (step 583→437→~45s warm)
- **Per-call defrag gating** — `gc.collect()+empty_cache()` fired per HTTP call (×64 microbatches) burned ~300ms×64 ≈ 19s/step pure overhead; gate it once-per-step (set `_allocator_dirty` after sync) → outer overhead 225s→8s/step.
- **`enable_forward_prefetch=true`** — overlaps teacher-cache prefetch: model_forward −17%, loss_compute −41%.
- **`TRITON_CACHE_DIR=/tmp/triton-cache-<role>-<revision>` + `rm -rf`** per revision (stale Triton cache across code changes).
- Per-microbatch cost reference: model_forward ~2.0s, KL ~0.68s, backward ~3.9s.

## O9. Comprehensive footgun list
1. **Loss ≠ success** — gate on `eval/accuracy`/generation (S5). The headline lesson.
2. **The autopilot verdict is confounded** — `strong_ptc_signal` is the no-pause-collapse artifact; read `eval/accuracy` (S4).
3. **`eval_control_start_step` must be `%eval_every` AND `=num_steps−1`** or the loop cycles/stalls (O2).
4. **Never hand-launch `controller.py autopilot`** (races the supervisor → duplicate autopilots corrupt `ideas.yaml`); and `pgrep -f 'controller.py autopilot'` matches your own shell → use the python-exe filter (O1).
5. **Transient 504 on weight-sync** → ride out via supervisor bounded retry; only recreate the stack for a recurring wedge (O7).
6. `XORL_P2P_SKIP_CACHED_PREPARE` → 735s failure. Never use.
7. **FP8 sync only with FP8 sglang**; bf16 sglang + `--fp8-sync-quantization` → step-0 size-mismatch crash.
8. **4-shards-on-1-node** = 33s sync (RDMA contention); spread shards across nodes.
9. **`flashqla` is bypassed under ulysses>1** (→ 0 speedup) and currently broken; keep `fla` (O5).
10. **Teacher ep_fsdp must divide 7168** (2/4/8 nodes; 6/7 crash).
11. **sglang teacher needs `--chunked-prefill-size 16384` + `--disable-radix-cache`** (the "radix" error misattributes the chunk-size cause).
12. **Don't set `NCCL_IB_GID_INDEX`/`NCCL_IB_HCA` on Mooncake trainer pods** (and `NCCL_IB_GID_INDEX` survives `~/.bashrc`). **`unset PYTORCH_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF`** — expandable-segments breaks Mooncake P2P registration >~20 MiB (pair with `XORL_P2P_CPU_POOL_MIN_BYTES=0` for Qwen3.5 small/odd-dtype entries).
13. **Rapid relaunch** → ZMQ `Address already in use` + `UnexpectedAdmissionError` (O7).
14. **sglang OOM = node co-location** (O7).
15. **lr ≤ 1e-5** on this regime (3e-5 collapses generation while loss drops).
16. **SMG `/health` always OK** with 0 workers → wait for `/v1/models`; `cache_aware` wrong for slow models → `power_of_two`; concurrency ≈ 340×N_shards (O4).
17. **Empty-CoT entries hard-abort** the trainer → filter prompts+CoT at aligned indices (O6). **Empty-rank metric seeding:** `LossMetrics.to_dict()` must emit ALL keys with defaults on every rank — dict-keyed Gloo all-reduce deadlocks on mismatched key sets (bit `opd_num_teachers` when None under packing).
18. **`generate_opd_manifest.py` post-edits** (older full-run generator): SMG swap (+ `/v1/models` wait), inject `supervise_student_cot`/`eval_accuracy_every`/`save_hf_safetensors` chz flags, drop `node-pool=compute`, pass `--fp8-sync-quantization` only with FP8 sglang.
19. **Don't touch anything `MTP`.** All GPU pods need `team: turbo` on the pod template (not the controller); don't set Volcano scheduler fields manually. Use **non-privileged + `rdma/infiniband` + `IPC_LOCK`** (not `privileged`, which leaks all 8 host GPUs).
20. **Sampler `--max-running-requests`:** the OLD Q36 runbook mandated `1` (batched GDN decode → repeated-suffix correctness failures); the LATER lesson (`LESSON_sglang_sampler_flags.md`) is that this serialized student eval (~30 min) and the right fix is **`1024` + cuda-graph + overlap-schedule** (correctness handled without serializing). **Use the modern flags.**
21. **Known-bad nodes:** `h100-{014,050,113,087}` (stuck SM / chronic IB). Most other "bad node" symptoms are the transient SM-sweep race — retry, don't permanently exclude.
22. **Hidden-match reporting bug:** `opd_hidden_match_loss` logs 0 even when active → verify via the `loss ≠ opd_kl` gap. **Multi-token filler:** `K=int(student_filler_count)` ⇒ single-token only; multi-token needs the per-sample `teacher_filler_tokens: list[list[int]]` path. `student_prefill_suffix="</think>Answer: "` un-suppresses the baseline. Instruct-2507 snapshot ships only `merges.txt` → `hf_hub_download` its `tokenizer.json` or get `!`-garbage.
23. **xorl vs VERL:** xorl has **no top-k KL mode** (full-vocab reverse-KL only) but uniquely has `opd_profile_*_ms` sub-phase profiling. Qwen3-MoE checkpoint boundaries must sit **inside** decoder layers (matched to `recompute_before_dispatch`) or MoE all-to-all replays with wrong packed-token metadata.

## O10. Repo / merge state (2026-06-07)
- Loop runs on branch `codex/opd-port-20260602` @ **`baf6dcce`** (pre-merge; has the OPD/P2P working code + the pin).
- The **`origin/apanda-dev` merge is resolved on branch `merge-apanda-dev` (43bfe780)** — apanda-dev base + hidden-match ported onto its refactored `opd_loss.py`; local p2p/model_runner/inference_endpoints additions dropped in favor of upstream. **UNTESTED against the live OPD flow.** Adopt only after testing + re-porting the P2P pin. Detail: `MERGE_APANDA_DEV_NOTES_2026_06_07.md`.

## O11. Source docs consolidated here (archive)
**Standalone how-to (independent interest):** `experiments/opd_profile/HOWTO_NOFILLER_OPSD_TRAINING.md` — the self-contained recipe for the fair **no-filler (no-CoT) OPSD** baseline (the one that reaches ~0.88 on 4×4; verified rising on 5×5 Q3.5). Read that if all you want is to run direct-answer self-distillation.

This runbook supersedes/consolidates (consult for full historical detail; this is the canonical entry point):
- **Loop/science:** `autoresearch/RUNBOOK.md` (banners here), `AUTORESEARCH_REVIEW_2026_06_06.md`, `OPD_STEP10_ARTIFACT_DECOMPOSITION_2026_06_06.md`, `PREFILL_TIME_COMPUTE_AUTORESEARCH_{RUNBOOK,SPEC,COMPREHENSIVE_ARCHIVE}_2026_06_0{3,4}.md`, `PREFILL_TIME_COMPUTE_OPSD_{MEMO,RUNBOOK}_2026_06_03.md`
- **Encoded-reasoning findings:** `ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`, `HANDOFF_OPD_ENCODED_REASONING.md`, `OPD_ENCODED_REASONING_FINDINGS_2026_06_01.md`, `OPD_FILLER_FLEET_FINDINGS_2026_06_01.md`, `OPD_CONFIG_A_B_RUNBOOK_2026_06_02.md`, `/old-data/apanda/tomi/outputs/{multimodel_filler_and_cot_summary_20260528,encoded_reasoning_final_reconciliation_20260601}.md`
- **Cross-method:** `~/xorl-rim-repro/experiments/rim/RESULTS.md` (RiM)
- **Infra:** `runbooks/opd_complete_runbook.md` + `runbooks/{flashqla_gdn_kernels,sglang_teacher_design,cot_dataset_index,smg_router_swap,smg_teacher_route,pipelined_teacher_prefill,run_b_opd_2node_teacher}.md`, `K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md`, `Q36_35B_OPD_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_02.md`, `P2P_CRASH_DIAGNOSIS.md`, `MERGE_APANDA_DEV_NOTES_2026_06_07.md`
- **Perf/diff:** `OPD_PROFILE_GAP_2026_05_27{,_V2,_V3}.md`, `OPD_VERL_DIFF_2026_05_27.md`, `MAINLINE_DIFF_2026_05_26.md`, `trials.md`, `LESSON_sglang_sampler_flags.md`

## O12. Trainer-side all-layer OPRD machinery (built 2026-06-09)

**What it is.** A self-distillation hidden-state objective: the trainer runs a **second no-grad forward on the teacher sequence** (`prompt+CoT+pause+answer`), captures per-layer `output_hidden_states`, and adds a normalized per-layer MSE between student and teacher hiddens at aligned (match_cot) positions, composed with the existing KL. Replaces the SGLang per-layer cache (the Q3.5 teacher is a multimodal-wrapper class; per-layer SGLang capture was off-table). Enables the S10h K=C / all-layer ablation.

**Flags (candidate `client_args` + top-level):**
- `opd_oprd_layers`: `everyN` (e.g. `every1`=all 40, `every4`=10) or explicit comma list → which decoder layers to match.
- `opd_oprd_num_layers`: total decoder layers (Q3.5 = **40**; required to expand `everyN`).
- `opd_oprd_last_k`: restrict the match to the last-k valid positions per row (memory bound). NOTE: per-position pre-norm residual MSE is uniform (~0.02) across the buffer, so last_k mostly trades memory, not signal.
- `opd_hidden_match_coef`: weight of the OPRD term (1.0 ≈ 0.1% of loss since residual MSE ≪ KL; **coef≈100** makes it ~12%). `opd_hidden_match_mode`: `cosine` (decouples) | `mse`. `opd_kl_loss_weight`: 0 = hidden-only (decouples → don't).
- `opd_buffer_equals_cot: true` → per-sample K=C buffer.

**Files (uncommitted working tree):**
- `models/transformers/qwen3_5_moe/modeling_qwen3_5_moe.py`: inner model collects `all_hidden_states`; **outer `Qwen3_5MoeForCausalLM.forward` must pass `hidden_states=outputs.hidden_states`** (was dropped → OPRD silently off). `models/outputs.py`: `hidden_states` field on `MoeModelOutput`/`MoeCausalLMOutput`.
- `server/runner/model_runner.py`: `_trainer_teacher_kept_layers` (no-grad teacher fwd, block-diagonal via `cu_seq_lens` from `teacher_position_ids`), the **teacher forward HOISTED before the 0-valid early-return** (uniform FSDP collectives — else dummy ranks desync), student-layer capture, `_resolve_opd_oprd_layer_indices`, OPRD metric accumulation (`opd_oprd_loss/raw/num_layers`), host-side bounds asserts on the gathers, `teacher_position_ids` in the opd_loss model_inputs exclude set.
- `server/orchestrator/packing.py`: under packing, concat `teacher_input_ids`; offset `teacher_kept_indices` by cum teacher-token-len; **re-base GLOBAL `teacher_cache_indices` per sample (`- min`) then offset by cum num_kept**; build per-sample `teacher_position_ids`; pop `_oprd_cum_*` in finalize.
- `ops/loss/opd_loss.py`: `use_oprd` branch (multi-layer MSE) + `kl_loss_weight`/`hidden_match_mode` params.
- client `examples/on_policy_distillation.py`: resolve `oprd_layer_indices`, add to `loss_params`, **guard forcing `pipeline_rl=False` when `opd_oprd_layers` set** (the pipelined prepare path doesn't populate `teacher_cache_indices`).

**Config requirement (eager).** Run OPRD with `rmsnorm_mode: native` + `ce_mode: eager` (`enable_compile: false`). The no-grad teacher forward triggers an **inference-mode inductor compile that wedges in `_sfdp_init` >20 min**; compile gives ~no speedup on this backward-bound model anyway.

**The 7 bugs fixed (each surfaced only after the prior, since OPRD had never executed end-to-end):** (1) outer-forward `hidden_states` drop; (2) compile wedge → eager; (3) collective desync (teacher forward inside the per-teacher loop, skipped by 0-valid ranks → hoist before early-return); (4) packer dropped/mis-offset teacher fields under packing; (5) `teacher_cache_indices` are GLOBAL cache rows → re-base per sample; (6) cross-sample teacher attention → block-diagonal `cu_seq_lens`; (7) `opd_oprd_loss/num_layers` not accumulated → always reported 0 (looked like OPRD-off). Validated: `opd_oprd_num_layers=40`, `opd_oprd_loss==opd_hidden_match_loss`, no OOB.

**OPS hazards (this session):**
- **`pipeline_rl` + no-OPRD:** the pipelined prepare path doesn't populate `teacher_cache_indices` for the no-filler/no-OPRD case → `opd_loss requires teacher_cache_indices` crash. Set `opd_pipeline_rl: false` (single-phase) for no-filler control runs.
- **P2P re-stage on restart:** each trainer restart can stale sglang-0's Mooncake buffers → `[P2P] batch_transfer_sync ret=-1` at step-0 sync. Recover/avoid by recreating sglang-0 **and** dispatch (`kubectl delete pod … --force` + `kubectl apply -f <rendered manifest>`); keep teachers warm; use the `dedicated` 1-sampler layout (not `spare-teacher1`, which overloads teacher-sglang-1). Proactively recreate before a long run.
- **CUDA device-assert kills a bare pod permanently.** An OOB gather (the OPRD bugs) raises a device-side assert that corrupts the CUDA context → that pod's process exits → bare Pods don't restart → it sits in `Error` → **every subsequent trainer deploy hangs in the 64-rank rendezvous** ("Waiting for rank 0 ready", GPU mem≈627 MiB). When init hangs, **`kubectl get pods -l stack=… ` FIRST**; delete+apply the dead pod (others no-op/stay warm). Fix the assert or it dies again next step. (A host-side `ValueError` does NOT kill the pod — clean recovery.)
- **Iteration:** `num_steps=2 num_prompts small` for path validation; `prompts_per_step=1` (or `enable_packing: false`) sidesteps multi-sample packing while debugging; le6000 packs ~1 sample/micro-batch (~4 s/sample) → 64 prompts/step ≈ 255 s/step ≈ 5.7 h for 81 steps.
