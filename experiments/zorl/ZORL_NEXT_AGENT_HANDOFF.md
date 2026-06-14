# ZORL — Next-Agent Handoff (2026-06-14)

Start here. This orients you across the post-consolidation homes and tells you the one
experiment to run next. Science detail lives in `OPSD_ZORL_RUNBOOK.md` §0b (same dir).

## Where the work lives (3 homes — the consolidation split everything out)

| Home | What | Notes |
|---|---|---|
| **`~/xorl-sglang-zorl`** (worktree, `zorl` branch) | **The real ZORL engine — your primary workspace, already set up & venv-linked.** Edit ZORL serving/kernel code here: `/start_zorl_*` endpoints, virtual-experts MoE-LoRA kernels, GPU-direct fold, seed-space momentum, GDN-layer LoRA wrapping (`sglang/srt/zorl/`, `lora/`, `managers/`, `model_executor/`). Isolated from the shared `~/xorl-sglang-internal` checkout (which all *other* experiments' pods serve from — don't touch it). | The 4 ZORL pools currently serve from the **shared** `~/xorl-sglang-internal` (content-identical to `zorl` right now → fine for the ceiling run, which needs no code change). **To deploy serving-code edits:** point the ZORL pool manifests' `workingDir`/`PYTHONPATH` at `/workspace/home/xorl-sglang-zorl` and restart those pods. FP8 variant: `~/xorl-sglang-fp8-fresh-ab`. |
| **`~/xorl-client/experiments/zorl/`** | **The harness** — `standalone/zorl_client.py` (HTTP client), `standalone/init_zorl_adapter.py`, `autoresearch/` (`controller.py`, `overnight_rank_search.sh`, `candidates/`), and the runbooks. | How you drive runs. This file lives here. |
| **`~/xorl-infra/{k8s,configs}/zorl/`** | **Deploy manifests** — pool StatefulSets `qwen3_6-35b-a3b-zorl-sglang-{m,f,i,l}-tp2-shard.yaml`, client job yaml, serving configs. | `team: turbo` baked in; bad nodes h100-073/105 excluded. |

**DEAD:** `src/xorl/server/zorl.py` in xorl-internal — an engine-native ZORL path that was
superseded by the fork; **unused, no PR**. (Preserved in branch history if ever wanted.) Any
*trainer*-side hooks → focused PR off `apanda-dev` on an `exp/zorl` branch. The old
`xorl-apanda-dev-zorl-consolidated` worktree is **prunable — don't use it.**

A run touches all three: **harness (client) → SGLang ZORL endpoints (fork) → k8s (infra).**

## The method (one paragraph)

EggRoll-style ES on a frozen/served base: antithetic **fresh_ab** paired-outer-product LoRA
perturbations (fresh A *and* B per pair → accumulated update is rank N·r), scored by
teacher-forced SFT logprob of the answer (`score_mode=sft`, zero sampling noise), z-scored pair
deltas → `W += lr·(1/N)Σ zᵢεᵢ`, then `merge_zorl_parent_into_base` folds parent into base. All
on inference infra — no trainer, no backprop.

## Current state — the goal is MET

- **Gradient-SFT parity reached:** held-out **0.9297** (≈ the 0.93 backprop reference) on
  Qwen3.6-35B-A3B, pure inference. Recipe: **rank-16, batch-256, resample-on, abs-lr 5.37e-4,
  ~280 steps / 12 h** (`MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE`).
- **lr is RANK-FREE.** σ ∝ 1/√r already normalizes the perturbation, so ‖update‖ ∝ lr alone —
  optimal **abs-lr ≈ 5.37e-4** for every rank. Keep σ = 1.5e-4·√(16/r); do NOT scale lr by rank.
- **early-climb ≠ converged ceiling.** 1 h trials rank *climb-rate* (r8 > r16 > r32 at ~8 probes)
  but that is NOT the ceiling — the only converged run (r16) hit 0.93. r8's ceiling is **unknown**.
- **Seed variance on a 1 h trial ≈ ±0.03** — replicate before ranking; never trust n=1.
- Artifacts preserved (worktree pruned): **`/shared/apanda/zorl-consolidated-runs/`** —
  `results-record/` (all logs incl. the 0.93 probe trail, CSVs, metrics) +
  `best-adapter-fresh-resample-step288/` (the 0.93 model, ~493 MB).

## ▶ DO THIS NEXT — the r8-vs-r16 ceiling run

The decisive open experiment. 4 pools (M/F/I/L) are held warm for it. Clean **2×2 (rank × batch)**
at the established optimum, **12 h cap** so each reaches the ~280-step plateau:

| pool | rank | batch | σ | abs-lr | role |
|---|---|---|---|---|---|
| M | 8 | 256 | 2.12e-4 | 5.37e-4 | r8 ceiling (the unknown) |
| F | 8 | 512 | 2.12e-4 | 5.37e-4 | r8 + batch (stability lever) |
| I | 16 | 256 | 1.5e-4 | 5.37e-4 | **re-confirm 0.9297** (anchor) |
| L | 16 | 512 | 1.5e-4 | 5.37e-4 | r16 + batch |

All: fresh_ab, resample-on, `score_mode=sft`, 128 pairs (`PAIRS_PER_SHARD=16`),
`MAX_RUNTIME_SECONDS=43200`. Adapters at `/shared/zorl/init-adapters/qwen3_6-35b-a3b-r{8,16}-eggroll-hybridattn`.

**Settles:** r8 arms ≥0.92 ⇒ r8's early lead holds to the ceiling (r8 wins: faster *and* equal).
r8 plateaus ~0.85–0.88 ⇒ capacity tradeoff, r16 owns the ceiling. Pool I reproducing 0.93 is the
trust anchor.

**Launch:** restart each pool's pods first (pristine base — merges mutate the served base in
place), wait `/health` 200, then one `controller.py launch` per pool with the rank/batch/lr/12 h
overrides (the per-pool targeting in `candidates/SEARCH-BASE-<POOL>.yaml` still applies).

## Ops / gotchas (hard-won)

- **One client per pool.** Pristine base = restart pods. `XORL_ZORL_FRESH_AB_FOLD=gpu_direct`,
  `XORL_ZORL_NOISE_DEVICE=gpu`, virtual-experts stack, mem-frac 0.85, `--max-lora-rank 32`.
- **Outputs → `/shared`, never the checkout.** The 296 GB of intermediate adapter exports were
  junk — keep logs/CSVs + the single best adapter, not every step's checkpoint.
- **k8s only** for training (never `torchrun` on the shared dev pod); `team: turbo` on every GPU pod.
- StatefulSets are `updateStrategy: OnDelete` — `kubectl apply` is inert until you delete pods
  (never RollingUpdate under a live run; it rolls the whole fleet).
- `wait_pool_healthy` must gate on real `/health` 200, not the 1/1 readiness probe (flips ~28 s
  in, long before the 35B model loads). Don't `pkill -f` the search driver (self-matches).
