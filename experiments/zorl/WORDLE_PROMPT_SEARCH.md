# Wordle prompt search — fixing the cold-base collapse for ZORL

**Date:** 2026-06-26
**Substrate:** Qwen3.6-35B-A3B (FP8), frozen base, **no LoRA** — i.e. the exact base
ZORL starts from. Served by the 40×TP2 `zorl-ar-sglang-w` pool.
**Goal:** find the system prompt that gives the cold base the best, most *reliable*
multi-turn Wordle play, so ZORL (evolution strategies on a rank-16 LoRA) has a real
floor to climb from instead of being capped at ~0.05.

---

## 0. HANDOFF — BOTTOM LINE (2026-06-28)

**Verdict: parameter-efficient ES (rank-16 `fresh_ab` LoRA on a *frozen* Qwen3.6-35B-A3B
base) does NOT reproduce GRPO's Wordle climb, and we have now ruled out every cheap lever.**
Across reward, learning rate, fitness-noise, model scale, and think-vs-no-think, the held-out
solve rate does not climb meaningfully off the cold floor. The reward fix (§10) was *necessary*
(it finally gave ES a non-flat train gradient) but *not sufficient* (the gradient doesn't
transfer to held-out solving). **This is a capability limit of the substrate, not a tuning miss.**

### What was tried, and what each showed (the lever map)
| Lever | Run | Result |
|---|---|---|
| Prompt search (no-think + commit) | §3 rounds 1–3 | Fixed the *format* collapse (cold ~0.19 probe / ~0.28 rollout, terminate ~99%) — a real floor to climb from. |
| Wrong reward (`_wordle_shaped_reward`) | early ES | **Flat fitness** (`reward_mean≈0.0004`) → no gradient → wander. Root cause of no-lift (§10). |
| Muon LR sweep (4×, ROLL-DENSE) | §8.1 | Held-out flat across all LR — **LR is not the lever.** `update_norm=23800·lr`; lr 1.2e-5 → un≈0.29. |
| Fitness noise (train64 / roll4) | §8 | Held-out flat — sharper gradient estimate doesn't help; the gap is generalization, not estimator variance. |
| Bigger base (Qwen3-235B-A22B-Instruct) | §9 | Baseline **≈ 35B, not better** — reasoning recipe > parameter count for this task. |
| **Correct reward, no-think** (`wordle_retrieval_reward`) | `sm4c9` / RETRIEVAL-W | **The only config that moved**: train `reward_mean` *climbed* 0.756→0.943; held-out **faintly drifted up** but weak/borderline over ~30 steps. Best ES substrate found. |
| **GRPO-match: correct reward, THINK** | `5hsph` / THINK-GRPOMATCH-W | **Flat on BOTH train and held-out** (19 steps): probe stuck at the collapse floor (~0.016–0.047), train `reward_mean` flat ~0.44. The 98%-invalid think collapse *starves the ES gradient*. Stopped 2026-06-28. |

### Why ES can't do what GRPO does here
The one hard skill is **constrained-vocabulary retrieval** (find a real word consistent with the
accumulated green/yellow/grey clues). The cold base derives the constraints *correctly* but then
**loops/rambles enumerating candidates without committing** — the "AMENS → no. AMENS → no." collapse
(see the think-sample inspection, §11). Clean turns are mostly *openers* (no constraints → easy
commit); the collapse is concentrated on the *retrieval* turns. GRPO bootstraps out of this with
**full-weight RL + unclipped importance sampling over ~75 steps**. A frozen-base rank-16 LoRA via
ES has neither the capacity nor the on-policy credit assignment to *instill* that retrieval skill —
it can only reweight what the base already does, and the base mostly collapses on the hard turns.

### Where to take it next (recommended, in priority order)
1. **Warm-start the base** — serve a *solving* full-weight checkpoint (e.g. the GRPO winner, or an
   SFT-on-solutions ckpt) as the ES base, then let ES refine *from above* the retrieval floor.
   This is the untested capability lever and the most likely to work: ES is good at *refining* a
   competent policy, bad at *creating* a missing skill. (Warm-start via the served BASE weights, NOT
   a LoRA-adapter warm-start — `mult` showed exported adapters don't carry the learning.)
2. **Candidate-scaffold** — the reference scaffold that *hands* the model the consistent candidate
   list hits ~0.969; if the product goal is solve-rate (not "ES learns retrieval unaided"), this is
   the cheap ceiling. Orthogonal to ES.
3. **Accept the negative result** — ES-on-frozen-base is the wrong tool for instilling Wordle
   retrieval; it remains useful where the base is already competent and you want cheap refinement
   (`mult` 0.74→0.93). Document and move on.

**Do NOT re-run** reward/LR/scale/think sweeps on the frozen base — they're exhausted. The forward-
pass (HyperscaleES) population trick also does NOT apply here (KV-stateful decode; see the plan-file
EVALUATION and `[[zorl-throughput-levers]]`).

**Infra/launch/serving knowledge for picking this up:** see the companion **`ZORL_WORDLE_INFRA.md`**
(pool spec, ES launch, reward/harness changes, the 235B serve recipe, Volcano binpack, bottleneck
measurements, wandb logging, current cluster state). Detail for the science claims above is in §3–§11.

---

## 1. The problem this solves

Cold ZORL held-out solve was **flat at ~0.05** over 6 ES steps (vs the GRPO full-weight
baseline **0.5625**). The wall was **not** ZORL — it was the **base model's prompt
behavior**:

- With the native `<think>` block enabled (`public_reasoning_constraints_think`), the
  base model **enumerates candidate words unboundedly** inside `<think>` and never
  commits. It hits the token budget → `finish_reason=length` → no `</think>`, no
  `<guess>` → the turn scores as `invalid_action`.
- Measured on the broken prompt: `invalid_action ≈ 0.9`, `valid_guess_rate ≈ 0.57`,
  `turns_used ≈ 2.4`. Turn 0 usually works; **turn 2+ collapses.**

Representative collapse (turn 1, after a good turn-0 guess `CRANE → XGXXX`):

```
So the pattern is _ R _ _ _.  Known absent: C, A, N, E.
Common words fitting this pattern:
BROAD? BROOD? BRICK? BRINE? BROIL? PRIDE?
... BRISK (B R I S K) - Valid.  BRITS - Valid.  BROOD - Valid.  BROOK - Valid.
BROOM - Valid.  BROTH - Valid.  BROWS - Valid.  ...   [runs to 6144 tokens, never guesses]
```

It parses the constraints **correctly** — it just can't stop listing words and commit.

---

## 2. Method

Parallel prompt sweep across all 40 replicas (`prompt_search*.py`). For each
`(variant, target)`: play a ≤6-turn game on the frozen base; `temp 0.7`,
`max_new_tokens 6144`, `repetition_penalty 1.1` (same as the training rollout).

A **variant** = `(label, enable_thinking, system-prompt augmentation)`. Two families:
- **`T_*`** = native think ON (`enable_thinking=True`) + an instruction trying to make
  the think terminate/commit.
- **`F_*`** = native think OFF (`enable_thinking=False`) + a "reason in one sentence,
  then guess, don't enumerate" instruction.

**Metrics:**
- **solve** = fraction of games solved within 6 turns
- **valid-turn** = fraction of turns that produced a valid guess
- **terminate** = fraction of turns that ended with `finish_reason=stop` (i.e. *not*
  the runaway-to-budget collapse)

---

## 3. Results

### Round 1 — 11 variants × 26 words

| variant | think | solve | valid-turn | terminate |
|---|---|---|---|---|
| T_strict | ON | 30.8% | 75.0% | 69.4% |
| F_direct | OFF | 26.9% | 85.7% | 99.0% |
| F_base | OFF | 23.1% | 80.6% | 100.0% |
| F_cap | OFF | 23.1% | 80.2% | 97.8% |
| T_2sent | ON | 23.1% | 70.1% | 70.1% |
| T_cap40 | ON | 23.1% | 70.1% | 67.2% |
| F_noenum | OFF | 19.2% | 81.8% | 100.0% |
| T_commit | ON | 19.2% | 66.7% | 65.1% |
| T_noenum | ON | 19.2% | 65.0% | 61.7% |
| F_strict | OFF | 15.4% | 81.2% | 99.0% |
| **T_cur** (the prompt the broken run used) | ON | 15.4% | 65.6% | 62.5% |

### Round 2 — 7 variants × 50 words (confirmation + refinements)

| variant | think | solve | valid-turn | terminate |
|---|---|---|---|---|
| **F_strat** | OFF | **28.0%** | 81.3% | 98.9% |
| F_cap | OFF | 20.0% | 82.1% | 99.0% |
| F_allclue | OFF | 18.0% | 81.0% | 99.5% |
| F_base | OFF | 16.0% | 80.2% | 98.4% |
| F_tight | OFF | 16.0% | 79.9% | 98.0% |
| T_strict | ON | 16.0% | 67.2% | 65.6% |
| F_direct | OFF | 12.0% | 79.8% | 99.5% |

### Round 3 — *in flight* (5 variants × 50 words)

Testing whether a **better** think prompt can beat no-think: `T_struct` (fixed
bounded template), `T_caphard` (hard <30-word, one-candidate cap), `T_fewshot`
(few-shot short-think example) vs `F_strat`/`F_cap` baselines. Results appended below
when done.

---

## 4. Findings

1. **Turning OFF the native `<think>` fixes the collapse.** Every `F_*` variant
   terminates **~98–100%** and produces **~80–82% valid turns**, stable across both
   rounds. The unbounded enumeration *was* the native think.

2. **Cold solve roughly 2–5×'d:** from ~0.05 (broken run) / 0.15 (`T_cur`) up to
   **~0.28** (`F_strat`), with valid-turn ~0.57 → ~0.81.

3. **Think doesn't help this cold base — even controlling for the collapse.** The fair
   comparison conditions on termination (a collapsed turn *can't* solve):

   | | solve | terminate | **solve-given-terminate** |
   |---|---|---|---|
   | F_strat (no-think) | 28% | 99% | **~28%** |
   | T_strict (think) | 16% | 66% | **~24%** |

   So on the games where think *does* finish, it still solves no better than no-think.
   The base's `<think>` is unproductive enumeration, not useful deduction. (Round-1's
   T_strict=30.8% was n=26 noise; at n=50 it fell to 16%.) Round-3 stress-tests this
   with stronger think prompts.

4. **The exact solve ranking among `F_*` is noisy** (12–28% at n≤50; F_direct swung
   27%→12% between rounds). What is *stable* is the terminate/valid floor (~99%/~80%).
   Pick by reliability, not by a noisy 4-point solve gap.

---

## 5. Decision

**Lock the no-think `F_strat` prompt as the ZORL base** (pending round-3 confirmation):

- *"Pick the single most likely real five-letter answer that fits EVERY clue (prefer
  common words). Give ONE short sentence of reasoning, then output `<guess>[WORD]</guess>`.
  Do NOT enumerate or test lists of candidate words."*

**Harness change** (`tasks/wordle.py`, `_turn_system_prompt`): `public_reasoning_constraints`
now returns `PUBLIC_REASONING_STRICT_SYSTEM_PROMPT + _FSTRAT_COMMIT`.
**Launch change:** `WORDLE_PROMPT_STYLE` `public_reasoning_constraints_think` →
`public_reasoning_constraints` (native think OFF). With no-think generations being short
(~100–300 tokens), `ROLLOUT_MAX_NEW_TOKENS` can drop 6144 → ~2048 for faster ES steps.

**Then restart ZORL from this fixed base** — the held-out probe should now read ~0.28
(real headroom) instead of ~0.05, giving ES an actual gradient toward solving.

---

## 7. RECONCILIATION — the 0.469 reference (rounds 4–5)

Rounds 1–3 ran at `max_new_tokens=6144`, `temp 0.7`, a non-harness think augmentation,
and a hand-picked word set — **none of which is the floor-protocol** that scored 0.469.
Round-4 reproduces the *exact* floor protocol (`public_reasoning_constraints_think`,
`temp 0.2`, `max_new_tokens 12288`) on 64 held-out `WORD_LIST` seed-777 words:

| arm (64 games) | solve | valid-turn | terminate |
|---|---|---|---|
| nothink (12288 / temp 0.2) | **18.8%** | 83.5% | 95.5% |
| think_ref (12288 / temp 0.2) | **1.6%** | 54.7% | 55.4% |
| think_6144 (temp 0.2) | **1.6%** | 55.0% | 53.6% |

**The 0.469 does NOT reproduce: pure think gets 1.6%, and the 12288 budget changes
nothing** (think is identical at 6144 and 12288 — it rambles to fill *any* budget,
terminate stuck ~55%). So the 6144 cap was never the cause.

**Why the gap — two different rollout paths in `tasks/wordle.py`:**
- **`rollout_completion`** (the **training rollout** ES optimizes): an invalid turn is
  **terminal — no retries** ("matches the gradient GRPO arm", line 1046–1050). My
  search mirrors this → pure think collapses to 1.6%.
- **`rollout_capture`** (the **eval** path): `invalid_retries=2`; on an invalid guess it
  **re-prompts with `enable_thinking=False`** (no-think) up to twice (line 1160–1191).

So **the 0.469 reference is an eval number propped up by a no-think retry fallback** —
when the think rambles past budget, the eval silently re-prompts in no-think mode and
recovers a clean guess. The reference is *already* leveraging no-think. The ES training
rollout has no such crutch, which is the real reason cold ZORL-with-think was stuck at
~0.05. **Round-5 (retry loop replayed on the same 64 words) confirms it:**

| arm (with retries) | solve | valid-turn |
|---|---|---|
| think_retry2 (think → no-think retry ×2) | **51.6%** | 90.9% |
| nothink_retry2 (no-think → retry ×2) | **32.8%** | 93.1% |

The retry fallback lifts think 1.6% → **51.6%** — reproducing (slightly exceeding) the
0.469 reference. **Reconciliation complete: the 0.469/0.5625 protocol is think+retry, and
our stack reproduces it.**

### Strategic consequence (the headline)

- **With retries, think (51.6%) BEATS no-think (32.8%)** — a committed reasoned guess is
  better than a direct one. No-think only wins on the *bare* training rollout (no net).
- **The cold base on GRPO's exact protocol already scores ~0.52** — essentially *at*
  GRPO's 0.5625. So **full-weight GRPO added only ~5 points over the untrained base**, and
  that margin is *inside* the ±0.05–0.10 run-to-run noise. "Beat 0.5625" is therefore a
  tiny, noisy target on this protocol — though the candidates-scaffold ceiling (0.969)
  shows there is real headroom above 0.52 if ES can climb.
- **Implication for the ES setup:** to compare to GRPO, eval MUST be think+retry (cold
  probe then reads ~0.52, not ~0.05). For a clean signal the training rollout should match
  (think+retry) — but that is far more expensive than the throughput-maxxed no-think/6144
  config (12288 think budget + up to 2 retries/turn). Train-cost vs signal-fidelity is the
  open decision.

**Corrected conclusion:** for the **no-retry training rollout that ES actually optimizes**,
no-think is not just better — it's the *only* thing that produces signal (18.8–28% vs
1.6%). Think "helps" only inside the retry-eval, and only because the retry is no-think.
`temp 0.7` no-think (28%, rounds 2–3) > `temp 0.2` no-think (18.8%) → use temp 0.7 for
the rollout. **Eval/probe must use the floor-protocol *with retries* for an apples-to-apples
comparison to GRPO's 0.5625** (train and eval measure different things — the plan's
"rollout solve under-reads held-out solve").

## 8. ES lift result — run ZORL-WORDLE-MUON-FRESH-R16-NOTHINK-W (2026-06-26)

Launched on the no-think no-retry fixed protocol (pop-640 fresh_ab+muon, lr 5e-5,
MAX_UPDATE_NORM 0, train_size 16, rollout temp0.7/max2048, probe temp0.2, job
`...fresh-resample-rmn5p`). Held-out probe (64 games) trajectory:

| step | exact_rate |
|---|---|
| **cold (0)** | **0.1875** (12/64) |
| 1 | 0.0625 |
| 2 | 0.0625 |
| 3 | 0.0781 |
| 4 | 0.1250 |
| 5 | 0.0312 |
| 6 | 0.0938 |

**Cold baseline validated at 0.1875 (≈ the predicted no-think ~0.19) — foundation is
sound.** But the first ES update knocked the parent off it (0.19→0.06, update_norm 1.19,
no trust region) and steps 1–6 random-walk in 0.03–0.125, **all below cold → no lift.**
Same non-generalizing-direction failure as `mult`/W000–018, now on a clean substrate
(`best_cand`≈0.25 on the train batch, but the fold doesn't transfer). `reward_mean≈0` +
train_size 16 = noisy fitness; no anti-decay guard lets the parent drift below cold.
**Decision (user): let it run longer** before tuning (anti-decay rescue =
ELITIST_ROLLBACK + bigger train_size + lower lr + trust region is the queued lever if
the wander persists).

**Extended to step 49 (2026-06-26):** probe stays a flat random walk between 0.031 and
0.203, **mean ≈ 0.09 — below cold**; best single step 0.2031 (step 21) is within 1σ of
cold (≈±0.05 on 64 games), i.e. not a real lift. No upward trend over 49 steps →
**conclusive null on a clean substrate: ES does not lift held-out Wordle solve here.**
More steps won't change this; the anti-decay rescue is the lever that might. Run left
going per user; monitor pings only on a genuine lift (probe ≥ 0.25).

### Phase-1 muon LR sweep (2026-06-26) — porting the mult-0.93 recipe

Mult-0.93 recipe = SGD lr 3.8e-4, sigma 1.5e-4, raw, constant-LR, MAX_UPDATE_NORM 0,
teacher-forced train256. **Key meta-lessons from the 0.93 post-mortem** (memory
[[zorl-beat-093]]): run-to-run VARIANCE (~0.06–0.10) exceeds config effects (single-run
comparisons measure noise); and elitist rollback is a TRAP (breaks accumulation). User
chose: **muon** (keep), ablate the **LR** (derive, don't guess) at lower-noise fitness.

**LR derivation (agent, verified):** muon uses `match_rms_adamw` (scale `0.2·√max(A,B)`);
since NS is scale-invariant, `update_norm = 23800·lr` (calibrated: 5e-5→1.19). SGD's
3.8e-4 is NOT transferable (muon discards gradient magnitude). Target update_norm ~0.3 →
lr ~1.2e-5. Swept LO 6e-6 / MID 1.2e-5 / HI 2.5e-5, partitioned 13/13/14 on a pristine
pool, no-think train32/roll1, EVAL 128.

**Result (~20 steps/arm):** `update_norm` validated to the digit (0.14/0.29/0.60 ✓).
But **no lift** — all three arms flat noise centered on cold (~0.10 on EVAL128; the cold
×3 = variance sample ≈ ±0.026): LO mean .10, MID mean .097, HI mean .103, no trend; the
0.148–0.156 "bests" were single-step noise spikes. **LR is not the lever** (flat across
4×). One positive: lower LR stopped the lr-5e-5 below-cold drift — traded "degrades" for
"flat-at-cold." → **Phase 2** (per user): MID lr 1.2e-5 + full `train64/roll4` ROLL-DENSE
(16× fitness samples) on all 40 replicas — the last untested lever (fitness noise). If
that's also flat, the ES direction simply doesn't generalize here (not tuning/noise).

## 9. Bigger-base test — Qwen3-235B-A22B-Instruct-2507 baseline (2026-06-27)

Hypothesis: the 35B-A3B base is too weak at Wordle; a 6×-larger instruct model would be a
much stronger substrate. Served BF16/TP8 from `/shared/huggingface` (after a long serve
detour — the FP8 copy on `research-coder-data` failed every which way: FP8+TP8 quant tiling,
mmap SIGBUS, no-mmap OOM, watchdog/restart; the OPD team's proven `q3-235bi-teach` config —
BF16, `/shared/huggingface`, TP8, graph-capture+custom-all-reduce disabled — is what works).

| protocol | 35B-A3B | **235B-Instruct** |
|---|---|---|
| no-think, no-retry (ES rollout) | 0.10–0.19 | **0.125** |
| no-think, +retry (eval) | 0.328 | **0.359** |
| think+retry | 0.52 | — (Instruct can't think) |

**Result: NOT a stronger substrate — basically tied with the 35B.** 6× the parameters buys
~nothing on *direct* (no-think) Wordle (valid-turn is high, 84–98%, so it forms guesses fine;
the deduction quality just isn't better). **For Wordle, reasoning >> scale:** the 35B *with*
think+retry (0.52) beats the 235B *without* think (0.36). The Instruct-2507 is the
non-thinking variant, so it can't use that lever, and the reasoning variant
(Thinking-2507) is not available locally. Conclusion: Wordle on a frozen base is
reasoning-bound, not scale-bound (nor ES-tuning-bound, per §8). Scale is not the fix.

## 10. REWARD FIX — ZORL was using the wrong fitness (2026-06-27)

Reviewed the GRPO-*converging* science runbook (`OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md`).
**Root cause of the ES no-lift: ZORL's ES fitness was the wrong reward.** ZORL's
`rollout_completion` used `_wordle_shaped_reward` (0.55·solve + 0.15·format + 0.2·valid +
0.15·info + 0.1·turn − 0.45·invalid) — which the science fork itself labels the "legacy
`reward` metric" and replaced. Its flaws (env-audit): **under-prices solving** (a clean
non-solve banks ~40% of a solve from saturated format/valid proxies) and has **NO signal for
the bottleneck skill — constrained-vocabulary retrieval.** As an ES fitness it's ~flat across
the population (`reward_mean≈0.0004`, every valid non-solve ≈0.45) → no gradient → the wander.

The converging GRPO run uses **`wordle_retrieval_reward`** (`--reward-key`, wordle.py:616):
SOLVE dominates (2.0·solve + fast-solve bonus, ~3–5× a non-solve) + **dense per-turn retrieval
signal**: +0.1 if guess still possible, **−0.2 if it violates a known constraint** (the exact
failure mode), +0.1·log(candidate narrowing) — present *before* solving. Verified port:
solve 3.94 ≫ narrowing non-solve 1.44 ≫ invalid-heavy 0.26, and non-solves are now
*differentiated* by narrowing/consistency (the discriminative gradient ES never had).
Bonus: the runbook's dominant GRPO lever was unclipping the loss (PPO clip throttles the rare
high-advantage solve); **ES has no clip** → ES + this reward sidesteps that.

**Action:** ported `wordle_retrieval_reward` into ZORL `tasks/wordle.py` + switched
`rollout_completion`'s ES fitness to it (env-gated `XORL_WORDLE_REWARD`, default `retrieval`;
`shaped` reverts). Relaunched ES: **ZORL-WDL-NOTHINK-RETRIEVAL-W** (job `...4fzmn`), no-think,
lr 1.2e-5 (derived, un≈0.29), pop-640, on a Volcano-binpacked 35B pool (4 TP2/node). Watching
whether the probe finally climbs above cold (~0.19) — the make-or-break test of the reward fix.

## 11. GRPO-MATCH (think) run + FINAL VERDICT (2026-06-28)

After the reward fix (§10), the no-think + retrieval run (`sm4c9`) was the *only* config to move
(train `reward_mean` climbed 0.756→0.943; held-out faintly drifted). To remove the last confound
("ES deviates from GRPO"), we matched GRPO's protocol **exactly**: `public_reasoning_constraints_think`
(cleaned so `_FSTRAT_COMMIT` is appended only for no-think — see `_turn_system_prompt(enable_thinking=)`),
`wordle_retrieval_reward`, 2048 tokens, train32/pool4096, no-retry, lr 1.2e-5, pop-960 (PAIRS=12) at
conc-96. GRPO-parity reward breakdown logged (`wr_solve, wr_narrowing, wr_consistency, wr_violate_turns,
wr_length_bonus, wr_format, wr_invalid_penalty`) — flows to wandb via the stdout forwarder's generic
`key=value` parser (project `zorl`, run `ZORL-WDL-THINK-GRPOMATCH-W`).

**Result (19 steps, then stopped): flat on BOTH axes.**
- Held-out probe `exact_rate`: `0.0156 0.0469 0.0156 0.0312 0.0156 0.0156 0.0312 0.0312 0.0000 0.0469 0.0312 0.0156 0.0312 0.0156 0.0312 0.0312 0.0156 0.0312 0.0312` — **no trend, stuck at the think-collapse floor.**
- Train `reward_mean`: ~0.37–0.56, bouncing ~0.44, **no climb**; `best_cand` ~0.69–0.95, no climb.
- ~15.7 min/step (think is long). `update_norm` 0.29 (healthy), `restarts=0`.

**Why think is WORSE than no-think for ES (the key mechanism):** the think prompt collapses ~98% of
*games* (`invalid_action≈0.984`). Most candidates collapse *similarly*, so there is little fitness
**difference** across the population → the ES gradient is ~flat (`reward_mean` doesn't move). No-think
keeps games terminating (~99%), so candidates *differ* in solve/narrowing → a climbable train signal.
**No-think gives ES a climbable substrate; think-from-collapse does not.** (GRPO survives think because
full-weight RL + unclipped IS can claw out of the collapse over ~75 steps; ES cannot.)

### Think-sample inspection (2026-06-28) — what "collapse" is, concretely
Played 24 think games on the live pool, classified each turn CLEAN (`finish=stop` + closed `</think>`
+ valid guess) vs COLLAPSED. **26/53 turns (49%) were CLEAN, 27/53 collapsed, 0/24 games solved.**
- **CLEAN turns are mostly openers** — short think (~1250 tok), closes `</think>`, commits
  (e.g. `<reasoning>SLATE is a strong common opener…</reasoning><guess>[SLATE]</guess>`). The model
  *can* terminate-and-commit when there's nothing to retrieve against.
- **COLLAPSED turns are the retrieval turns** — the model derives the constraints correctly
  (`A _ _ N _`, "E is X") but then **enumerates and loops without committing** —
  `…AMEND → E is X. So no. AMENS → no. AMENS → no. AMENS → no.…` — rambling to the 2048-token budget
  (`finish=length`, no `</think>`, no guess). This is exactly the runbook's "one hard skill =
  constrained-vocabulary retrieval." `valid_guess_rate≈0.53` (openers clean) but games collapse
  because one bad retrieval turn ends the game.

**This is the whole story in one observation:** the non-collapsed mode exists but mostly on the *easy*
turns; the thing training must instill is *retrieval on constrained turns*, a genuine capability gap.
ES on a frozen rank-16 LoRA can reweight the base's behavior but cannot create this skill. → **warm-start
from a base that can already retrieve** (§0, direction 1) is the lever, not more ES tuning.

## 6. Caveats

- Small n (26–50 games/variant); solve has high variance, terminate/valid do not.
- This is the **cold base, no LoRA** — it measures the *floor* ZORL climbs from, not
  ZORL itself. The open question remains whether parameter-efficient ES can lift ~0.28
  → the GRPO 0.5625 (the §4-plan target), judged on multi-seed held-out floor-eval.
  **(2026-06-28: answered — no; see §0 + §11.)**
- If round-3 shows a think prompt that clears ~95% terminate **and** >28% solve, revisit
  the lock in favor of that think variant.
