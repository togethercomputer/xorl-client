# Beat 0.93 — evidence-based plan (2026-06-14, agent)

## The decisive evidence
Prior best `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE` probe trajectory (full log,
`results-record/.../qnjc6-gqqf4`):

```
step  25  0.8125
step 100  0.8438
step 150  0.8984
step 200  0.9141
step 235  0.9297   <- PEAK
step 250  0.8984
step 300  0.8984
step 350  0.8750
step 400  0.8125   <- DECAYED back toward cold
```

**0.93 is NOT a capability ceiling and NOT a time cutoff** (it ran to step 421).
It is the classic ES failure mode: with constant step-size (sigma) and constant
LR, once the gradient signal vanishes near the optimum, the fixed-size ES update
random-walks the parent *away* from the peak. The recipe can clearly reach 0.93;
it just cannot *hold* it.

## What is already exhausted (don't repeat — all scored < 0.93)
From `results-record/zorl_autoresearch` peak probes:
- momentum: FRESH-MOM-B07 0.781, MOM512 0.836, STAT256-MOM 0.820 (momentum hurts fresh_ab)
- rank fitness shaping: STAT256-RANK 0.781
- more pairs: FRESH-P256 0.859, RESAMPLE256 0.789
- bigger train set: TS512 0.906, STAT1024 0.750
- LR up: LR1E3 0.789, LR3E3 0.656 (3.8e-4 is near-optimal)
- GDN 0.867, LOZO 0.727, COMBO-* 0.80-0.84, stationary 0.80-0.84
- FRESH-RESAMPLE (raw, resample, fresh_ab, r16, 128 pairs, lr3.8e-4, sigma1.5e-4) = 0.9297 BEST

## The plan (priority order)

### R1 — FRESH-RESAMPLE-REFINE  [primary, NO CODE, highest EV]
Proven recipe + the two anti-decay levers NEVER used on the winner, both already
implemented, both directly targeting the post-peak decay:
- `ELITIST_ROLLBACK=1` (+ `SNAPSHOT_ID=best`): snapshot on new best, restore on
  regression. Server-backed (`/snapshot_zorl_parent`,`/restore_zorl_parent`).
  Would have held 0.9297 instead of decaying to 0.81. Compatible because
  `MERGE_EVERY_STEPS=0`.
- `LR_SCHEDULE=cosine`, GENTLE + LATE: `LR_DECAY_STEPS=500`, `LR_MIN_FRAC=0.3`
  (LR stays ~full through the ~235-step climb, then eases to refine — opposite of
  HYPER-L which decayed to 10% by step 120 AND broke the score signal).
- KEEP `UPDATE_STRATEGY=raw` (HYPER-L's project_baseline_standardized collapses
  signal via its per-project std-floor — confirmed flat-at-cold for 37 steps).
- `MAX_UPDATE_NORM=0`, `MAX_RUNTIME_SECONDS=64800` (18h, room past step 235).
- Else identical to proven recipe (seed 9246).

### R2 — REFINE-WARM  [fast refinement test, NO CODE]
Warm-start from the saved champion to test pure refinement without re-climbing:
- `ADAPTER_DIR=/shared/apanda/zorl-consolidated-runs/best-adapter-fresh-resample-step288`
- `LEARNING_RATE=0.00019` (half), `ELITIST_ROLLBACK=1`, raw, constant or gentle cosine.
- Cheap signal on whether reduced-step refinement breaks 0.93 from the peak.

### R3 — higher rank r32  [moderate effort: needs r32 init adapter]
fresh_ab perturbation is rank-(N*r)/step; r32 doubles per-step capacity. Servers
already run `--max-lora-rank 32`. Needs an r32 init adapter (current is r16).

### Code stretch (only if R1/R2 plateau)
- sigma annealing schedule (ES step-size adaptation; client+server change).
- per-module RMS preconditioning in the fresh_ab fold — fresh_ab-COHERENT
  "Adam-lite" (moments in weight space, not seed space). NOT full per-coordinate
  Adam (gpu_direct fold never materializes dense per-module deltas; per-coord
  would undo that efficiency). See [[zorl-eggroll-optimizer]].
- `EVAL_SIZE` 256/512: 0.9297 = 119/128; probe granularity 0.0078 may be hiding
  sub-resolution gains.

## BLOCKER
All experiments require resetting pool L (delete HYPER-L job + delete the 8
`zorl-ar-sglang-l` pods so they reload a pristine base — ZORL merges mutate the
served base in place). The auto-mode guardrail DENIED this destructive/interfering
action (correctly — it's not specifically authorized). Need either explicit
"reset pool L and launch R1" from the user, or a Bash permission rule, OR stand up
a fresh additive pool (16 free GPUs exist) to avoid touching pool L.

## Exact R1 launch (after pool L reset + 8x /health=200)
```
python /home/apanda/xorl-client/experiments/zorl/autoresearch/controller.py launch \
  --candidate /home/apanda/xorl-client/experiments/zorl/autoresearch/candidates/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE.yaml \
  --set-env INFER_URL='<8 pool-L urls>' \
  --set-env RESULT_ROOT=/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-REFINE-L \
  --set-env WANDB_NAME=MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-REFINE-L \
  --set-env ELITIST_ROLLBACK=1 --set-env SNAPSHOT_ID=best \
  --set-env LR_SCHEDULE=cosine --set-env LR_DECAY_STEPS=500 --set-env LR_MIN_FRAC=0.3 \
  --set-env MAX_RUNTIME_SECONDS=64800
```
(raw strategy + MAX_UPDATE_NORM=0 are candidate defaults; do NOT override them.)
