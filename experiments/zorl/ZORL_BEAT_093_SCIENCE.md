# ZORL — Beat 0.93: SCIENCE

Goal: beat the ZORL `mult` held-out greedy probe best of **0.9297**
(128-example eval, exact-match rate).

## 1. The baseline and what "0.93" really is

Best prior run `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE` (seed 9246): raw score
strategy, `fresh_ab` perturbation, resample-train-each-step, rank 16 / alpha 16,
128 antithetic pairs, `lr=3.8e-4`, `sigma=1.5e-4`, `MAX_UPDATE_NORM=0`,
`momentum=0`.

**0.93 is NOT a capability ceiling and NOT a runtime cutoff.** Full trajectory
(`results-record/.../qnjc6-gqqf4`, ran to step 421):

```
step  25  0.8125     step 200  0.9141
step 100  0.8438     step 235  0.9297   <- PEAK
step 150  0.8984     step 300  0.8984
step 175  0.9062     step 400  0.8125   <- DRIFTED back toward cold
```

Classic evolution-strategies failure: with constant step-size (sigma) and
constant LR, once the parent is near the optimum the per-step ES gradient
estimate is noise-dominated, so the fixed-size update **random-walks the parent
off the peak**. The recipe reaches 0.93; it cannot *hold* it. ⇒ the lever is
**step-size control**, not more pairs/data/time.

## 2. Already exhausted (don't repeat — all peaked < 0.93)

From `/shared/apanda/zorl-consolidated-runs/results-record/zorl_autoresearch`
(~40 runs):

| lever | result |
|---|---|
| momentum (B07 / MOM512 / STAT256-MOM) | 0.78–0.84 — hurts fresh_ab |
| rank fitness shaping (STAT256-RANK) | 0.78 |
| more pairs (P256 / RESAMPLE256) | 0.79–0.86 |
| bigger train set (TS512 / STAT1024) | 0.91 / 0.75 |
| higher LR (1e-3 / 3e-3) | 0.79 / 0.66 |
| GDN layers / LOZO / combos / stationary | 0.73–0.88 |
| **FRESH-RESAMPLE (the recipe above)** | **0.9297** |

## 3. The LR-decay-strength sweep (this campaign)

All fresh climbs from the standard init adapter, `raw`, `MAX_UPDATE_NORM=0`, **no
elitist rollback**. Cosine `lr_decay_steps/lr_min_frac`:

| run | schedule | LR at step 235 | outcome |
|---|---|---|---|
| R3 LRDECAY2 | sharp 350/0.12 | ~57% | plateaued ~0.88 (decay bit mid-climb) — killed |
| R1c LRDECAY | medium 600/0.2 | ~73% | plateaued ~0.87 |
| R4 LRDECAY3 | near-const 1000/0.3 | ~91% | climbing, ~0.87 @ step 150 (control) |

**Lesson: any LR decay that bites *during* the climb (steps 0–235) suppresses the
peak.** Keep LR ≳85% until ~step 235.

## 4. The key insight → delayed decay (lead bet R5)

- Constant LR → reaches 0.93 then **drifts** (peak = 0.93, doesn't exceed).
- Decay-from-step-0 → **suppresses** the climb (peaks < 0.90).
- To **exceed** 0.93 you must hold full LR to the peak, **then** shrink the step
  to refine without drifting = **delayed decay**.

Implemented a new schedule `--lr-hold-steps N` (hold `lr` for N steps, then cosine
over `--lr-decay-steps` to floor; default 0 = legacy). **R5 HOLDDECAY-M**:
hold 200 → cosine 200 → floor 0.15 (100% LR through the climb, 94% @ 235, 27% @
350, 15% @ 400). This is the only configuration in the sweep that can structurally
*exceed* 0.93. Status: climbing on the prior best's pace at full LR.

## 5. Optimizer / rollback findings

- **No Adam.** ZORL's seed-space fold is SGD + optional first-moment momentum
  only. HyperscaleES `EggRollBS` supports `optax.adamw`, but the prior best
  reached 0.93 with plain SGD, so Adam is not necessary; it's the one un-ported
  EggRollBS capability and a large engineering cost in seed space. See
  `[[zorl-eggroll-optimizer]]`.
- **Momentum is contraindicated for `fresh_ab`** (velocity across per-pair
  subspaces is undefined; the prior best explicitly ran momentum=0). Confirmed:
  all momentum runs underperformed.
- **Elitist rollback breaks the climb.** It reverts every probe-regressing step,
  destroying the ES accumulation of the small persistent signal (the prior best
  *dipped* early — step 5 = 0.71 — yet kept climbing). Tested: it restored at
  step 5. Do not use it during a fresh climb. (It *might* fit a warm-start
  refine-and-preserve, but warm-start is not viable — see §6.)

## 6. Open questions / caveats

- **Warm-start is not viable** from the saved adapter. `fresh_ab` folds the ES
  learning into the **served base weights** (in place), not the parent LoRA, so
  `best-adapter-fresh-resample-step288` scores ≈init (cold 0.7266) when loaded on
  a pristine pool. The 0.93 lives in the original pool's mutated base (gone). Only
  a **fresh climb** reaches 0.93.
- **RESOLVED (2026-06-16): 0.93 did not reproduce; the bottleneck is VARIANCE.**
  A faithful constant-LR reproduction (REPRO-M, exact config, seed 9246) peaked
  **0.8438**. Across five fresh runs — REPRO-M 0.844, R5 0.828, R4 0.867, R1c
  0.875, R3 0.883 — none cleared 0.884 vs the original 0.9297. The constant-LR
  *exact reproduction was the lowest*, below the decay runs, so the ~0.06–0.10
  run-to-run spread is **variance** (scoring nondeterminism + ES trajectory
  divergence over 150+ steps), **larger than any decay-schedule effect.**
  Consequences: (1) single-run config comparisons here are noise-limited — the
  decay sweep (and much of the prior campaign) measured noise; (2) **0.93 is the
  favorable tail of a distribution centered ~0.85–0.88, not a stable target.**
  Beating it by config tuning on single runs is chasing luck. The real problem is
  probe/run variance, not the algorithm.

## 7. Recommended next experiments (priority) — VARIANCE FIRST

The reproduction verdict (§6) means config tuning is premature until variance is
understood and reduced. New priority order:

1. **Quantify the variance (cheap, fast).** Probe one fixed parent adapter N times
   (no training) to separate pure scoring-noise variance from ES-trajectory
   variance. Tells us whether the lever is the eval (measurement) or the dynamics.
2. **Cut per-probe noise:** larger eval set (256/512 — shrink TRAIN_POOL to fit
   the 8192 usable-pair budget) and/or `PROBE_N>1` averaging. Note: changes
   comparability with the 128-example 0.93 number — re-baseline.
3. **Reliable point estimates:** run configs as small seed-ensembles (n≥3) and
   compare *distributions*, not single runs. Only then is any config effect
   (decay, rank, sigma) measurable above the ±0.05 noise.
4. Then, if a config reliably beats the ~0.86–0.88 center: delayed decay (the
   structural anti-drift idea) and higher rank (r32) remain the best bets — but
   judge them on ensembles, not single draws.
