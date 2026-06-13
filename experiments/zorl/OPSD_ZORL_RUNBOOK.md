# OPSD-via-ZORL — Canonical Runbook

ES (EggRoll-style) reproduction of the no-filler OPSD-mult result, on the standalone SGLang
stack. **§0b is the current canonical state (2026-06-13)** — it supersedes §0a/§0 and the
parts of §2 it contradicts (read it first). Earlier sections are kept for provenance.
Prior revision 2026-06-11 (evening): rung 3 VALIDATED, throughput solved, conclusions in §2.

**Worktrees:**
- `/home/apanda/xorl-apanda-dev-zorl-consolidated` — the standalone ES stack (this runbook,
  `experiments/zorl/standalone/…`, the serving pools). Everything below lives here.
- `/home/apanda/xorl-apanda-dev-opd-port` — the gradient/OPD server-training stack (rungs 1–2,
  EP=4×8). Reference only; see §8.

**SGLang fork:** `/home/apanda/xorl-sglang-internal` @ `apanda-dev`, HEAD `bc1ffb583` ("Fix LoRA
serving on Qwen3.5/3.6-MoE hybrid models"). Pool pods exec from this PVC tree.

**Key memories:** `zorl-es-stat64-generalization`, `zorl-es-batched-scoring-oom-fix`,
`project_zorl_es_sft_validation`, `project_shared_outer_lora_moe_35b_hang`.

---

## 0b. UPDATE (2026-06-13) — fresh_ab ES, the HP-search campaign, and the early-climb≠ceiling lesson

**This is the current canonical state. It supersedes §0a/§0 and the parts of §2 it contradicts
(esp. §2.1's "ES requires stationarity").**

### Estimator: fresh_ab (paired-outer-product EggRoll) is now the default
True fresh A *and* B per antithetic pair → each step's accumulated update is rank (N·r), not
rank-r. Replaces the old `b_only` / shared-A `a_and_b` (the NULL experiment flagged in §0a).
Serving = the **virtual-expert MoE-LoRA kernel stack** (user directive: always use it):
`--lora-use-virtual-experts`, GDN-layer LoRA wrapping, GPU-direct fold. SGLang fork `apanda-dev`
advanced past `e8cbba250` → batched MoE fold bmm `89cc195c6`, GPU-direct fold `b591ddcc1`,
fresh_ab momentum `6b8c0bfb3`. (FP8 base + bf16 LoRA is a separate WIP branch
`apanda-dev-fp8-fresh-ab` — see `FP8_BASE_FRESH_AB_IMPLEMENTATION.md`; not in the search path.)

### §2.1 REVISED — ES does NOT require a stationary objective; that was a batch-size confound
The old "resampling kills accumulation" conclusion conflated *resampling* with *small batches*.
With fresh_ab + an adequate batch, **resampling is fine**: the project-best **0.9297** (≈ the
0.93 gradient-SFT reference) came from `…-FRESH-RESAMPLE` (`RESAMPLE_TRAIN_EACH_STEP=1`,
rank-16, batch-256, 12 h / ~280 steps, run 2026-06-12). What ES needs is batch *coherence*
(enough pairs that the averaged pair-signal clears the noise floor), NOT a frozen set. The E5
audit measured weight-space cos(ES,∇) ≈ 1.4e-4 with D_eff ≈ full dense dim and batch-gradient
coherence 0.882 at batch-256 — coherence, not stationarity, is the lever.

### lr/rank double-counting → lr is RANK-FREE; optimal abs-lr ≈ 5.37e-4
σ ∝ 1/√r already normalizes the per-pair perturbation displacement (‖σ·ABᵀ‖_F ≈ σ·d·√r), so
‖update‖ ∝ lr **alone**. Scaling lr *also* by √(16/r) (the old displacement-match) double-counts
the rank correction and under-steps high ranks (r32/lr1× topped out ~0.797; r32 at abs-lr
5.37e-4 climbed to the leader band). **Rule: keep σ ∝ 1/√r (load-bearing); make lr
rank-independent.** Empirical optimum abs-lr ≈ **5.37e-4** (= old r8/lr1×). r16/r32 are
lr-insensitive across [3.8e-4, 5.4e-4] (~flat ceiling); 1.07e-3 overshoots, 1.9e-4 undershoots.

### The HP-search campaign (overnight, 4 pools M/F/I/L)
Driver `autoresearch/overnight_rank_search.sh <POOL> 8 SEARCH-BASE-<POOL>`, one per pool. Each
trial: restart pods (pristine base) → launch a **1 h-capped** trial of a sampled config → record
peak held-out probe → repeat. Candidate `SEARCH-BASE-<POOL>` = fresh_ab + resample + `score_mode=sft`.

**Grid = 54 cells:** rank {8,16,32} × batch {256,512,768} × pairs_per_shard {8,16}
(→ total_pairs 64/128 over 8 replicas) × lr_mult {0.5,1,2} (× rank-free base 5.37e-4 →
{2.7e-4, 5.37e-4, 1.07e-3}). σ = 1.5e-4·√(16/r) per rank (r8 2.12e-4, r16 1.5e-4, r32 1.06e-4).
seed = 9300 + POOL_OFFSET + trial. A 4-entry PRIORITY queue front-loads r16/r32 at the optimum
(rank-isolation vs the r8 leader) before random sampling resumes.

**1 h leaderboard (early-climb peak, matched b768/p64, abs-lr 5.37e-4):**

| rank | peak (mean) | n | spread |
|---|---|---|---|
| r8  | 0.867 | 1 | — |
| r16 | 0.826 | 4 | 0.797–0.852 |
| r32 | 0.799 | 4 | 0.773–0.813 |

Ordering r8 > r16 > r32 in early climb. **Seed variance on a 1 h trial ≈ ±0.03** (r16 spanned
0.055) — larger than several inter-config gaps; replicate before ranking, never trust n=1.

### ⚠ CENTRAL LESSON — early-climb ≠ converged ceiling
The 1 h search ranks **climb-rate in ~8 probes**, NOT the converged ceiling. r8 wins the early
climb (0.867 @ 8 probes), but there are **zero** converged r8 runs. The only converged
(12 h / ~280-step) run is **rank-16 → 0.9297**, higher than r8's entire early-climb peak.
Classic capacity tradeoff: low rank climbs fast / may plateau low; high rank slow-starts /
higher ceiling. So "rank-8 wins" is an **early-stopping artifact, not a ceiling result** — do
NOT pick the production rank from 1 h trials. Decisive open experiment: a **long uncapped run**,
r8-long (does its early lead beat 0.93?) vs re-confirm r16-long (reproduce 0.9297). The r32
"0.867" seen earlier was a 4-probe fast-start at b768/p128 — dismissed once run to 7 probes (0.80).

### Infra notes for the search pools
- 4 pools `zorl-ar-sglang-{m,f,i,l}`, 8×TP=2 each, virtual-experts stack, `updateStrategy:
  OnDelete`, `--max-lora-rank 32 --max-loras-per-batch 16`, mem-frac 0.85. Manifests
  `k8s/qwen3_6-35b-a3b-zorl-sglang-<pool>-tp2-shard.yaml`.
- **Bad nodes** excluded via nodeAffinity NotIn: **h100-105** (NCCL `ncclCommInitRank` "unhandled
  cuda error"), **h100-073** (crashloop). Add new bad nodes there if pods crashloop.
- **Driver gotchas burned during the campaign** (all fixed in-script, all worth re-checking on any
  fork): APP name must be lowercased (`${POOL,,}` — pod names are lowercase; a capital grep matched
  0 pods → silent `pool_unhealthy` spin forever; *trace a spinning loop with `bash -x` FIRST*).
  The launched-job regex must match the candidate's actual job name (`zorl-ar-search-base-<pool>-…`,
  NOT `multopsd`). `wait_pool_healthy` gates on real `/health 200`, not the 1/1 readiness probe
  (which flips ~28 s in, long before the 35B model loads). The lr fix is live (`base_lr=0.000537`,
  rank-free); **editing the script requires restarting the loops** — bash buffers the `while` body,
  so a running loop won't re-read the file (kill specific PIDs + relaunch in background).
- **NEVER `pkill -f overnight_rank_search`** — it self-matches and kills the watchdog/driver. Kill
  specific PIDs excluding `$$`. **Never touch other users'/agents' pods** (`er-opd-*`, Wordle SMGs,
  FP8 pool). No `git stash -u` in background tasks.

---

## 0a. UPDATE (2026-06-11 ~21:00Z) — NEW KERNEL STACK + E-WAVE

**The serving stack moved to virtual-expert MoE-LoRA kernels** (user directive: always use it
from now on). `xorl-sglang-internal` @ `apanda-dev` is now `e8cbba250` (local, unpushed):
`02cde2bec` virtual-experts + GDN-layer LoRA wrapping, `e8cbba250` server-side ES momentum.
Validated: 9/9 fp32-ref unit tests, 108/108 kernel regression, zero-B parity, exact
determinism, **2.04× production-shape scoring** (82→40 s for 32 cands × 256 ex). Cross-path
logprob diffs vs the chunked path are bf16 small-delta rounding (ES pair-signal corr 0.974);
within-path determinism is exact. New-pool serving flags: `--lora-use-virtual-experts
--max-loras-per-batch 33` (still `--disable-cuda-graph`; incompatible with fully-sharded-loras).

**Pools C/D + STAT256/N512 jobs DELETED (user-authorized); pools E/F/G (8×TP=2 each, new
stack) serve the E-wave** launched ~20:50Z, all 12 h caps, recipes = STAT256 unless noted:
- **E1** `MULTOPSD-EGGROLL-35B-RESAMPLE256` (pool E) — resampled TS=256 from the 8192 pool;
  decides stationarity-vs-batch-size. Descends like STAT256 ⇒ "ES needs large batches, not a
  frozen set".
- **E3** `MULTOPSD-EGGROLL-35B-STAT256-RANK` (pool F) — rank fitness shaping vs z-score.
- **E2** `MULTOPSD-EGGROLL-35B-STAT256-MOM` (pool G) — server momentum β=0.9 + lr=0.003
  (displacement-matched) + merge-every-16 (B-velocity reset at merge; verified ratios
  1.000/1.345/1.570 vs theory on pool G). Watch `pre_momentum_update_norm` vs `update_norm`.

GDN-mixer LoRA serving support EXISTS on the new stack but needs the init adapter regenerated
with `in_proj_qkvz in_proj_ba out_proj` targets + those names in `--lora-target-modules`
(NOTE: adding tensors changes every tensor's noise stream — new sessions only). Next code item:
E4 paired-outer-product EGGROLL estimator (true fresh-A,B; the current `a_and_b` flag is a NULL
experiment — parent B=0 post-merge + separate ΔA/ΔB updates ≠ Σzᵢε_Bε_Aᵀ). Memory:
`zorl-es-levers-audit`.

## 0. CURRENT STATE (2026-06-11 ~18:00Z)

**The ladder is complete:**

| Rung | Method | Status | Result |
|------|--------|--------|--------|
| 1 | gradient full-FT | DONE | fast rise 0.508→0.625 (EP=4×8, §8) |
| 2 | LoRA (attention-only, gradient) | DONE | held-out 0.664→0.680 |
| 3 | **ES (EggRoll, merge/step)** | **VALIDATED** | held-out 0.74→**0.82–0.836 sustained** on stationary SFT (§2) |

**Live right now (3 runs, monitors armed):**
- `MULTOPSD-EGGROLL-35B-STAT256` (pool C) — fixed 256 examples; step ~480; held-out steady
  0.80–0.836. The strongest run.
- `MULTOPSD-EGGROLL-35B-STAT64-N512` (pool D) — fixed 64 examples, 512 pairs; step ~180;
  held-out 0.82+, exceeded STAT64's ceiling.
- `MULTOPSD-EGGROLL-35B-TFB8` (main pool) — resampled control; step ~230; flat 0.73–0.78.

All run to 12 h caps unless stopped:
`kubectl delete job zorl-ar-multopsd-eggroll-35b-{stat256-5wv9f,stat64-n512-dm924,tfb8-f98th}`.
Pools C/D (32 GPUs) when done: `kubectl delete statefulset zorl-ar-sglang-{c,d}; kubectl delete
svc zorl-ar-sglang-{c,d}-headless`.

---

## 1. THE EXPERIMENT (science setup)

**Task:** 4-digit × 4-digit multiplication. Student sees `chat(user_problem) + "Answer: "` (no
CoT) and must emit the product. Data: `/shared/opd-coord/randnum_4digit_8192.json` (+ aligned
CoTs for the OPSD-KL variants). Verifiable.

**ES objective = SFT loss:** candidates are scored by the mean teacher-forced logprob of the
ground-truth answer digits (= −SFT loss), `--score-mode sft`. Smooth, deterministic (temp-0
input logprobs — there is NO sampling noise in scores; pair deltas are exact). OPSD-KL
(`opsd_kl_full`, §9) remains available but SFT is the validation objective.

**ES algorithm (EggRoll):** antithetic pairs of rank-16 LoRA-B perturbations (`b_only`,
unit-gaussian noises, σ=0.012), z-scored pair deltas, update `W += lr·(1/N)Σ zᵢεᵢ` (lr=0.03),
then `merge_zorl_parent_into_base` every step (folds parent into base, re-inits LoRA-A → the
accumulated update is full-rank though each step's noise is low-rank).

**Adapter:** `/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll-hybridattn` (attn on the 10
full-attention layers + shared-outer experts on all 40; D_B ≈ 1.7e8). The serving of this
adapter was fixed @ `bc1ffb583` (§6).

**Readouts (the important part):**
- **Held-out probe** — greedy exact-match on 128 *disjoint* problems, every `PROBE_INTERVAL`
  steps. THE success metric. Cold baseline ≈ 0.742–0.758. Binomial 1σ ≈ ±0.036.
- **Train probe** — `standalone/probe_train_batch.py` re-derives the deterministic frozen train
  batch (`build_examples(seed)` + `train_pool[:n]`) and greedy-scores the CURRENT parent on it,
  read-only. Decomposes loss descent into memorization (train↑ only) vs generalization (both↑).
  `--urls` to point at any pool; omit `--parent-lora-name` to probe a served base.
  **Baseline gotcha:** the 256-example train slice is HARDER than the eval set — cold 0.660,
  not ~0.74. Always measure the train slice's own cold baseline before interpreting.

---

## 2. SCIENTIFIC CONCLUSIONS (2026-06-11)

### 2.1 ES requires a STATIONARY objective — resampling kills accumulation
> **⚠ SUPERSEDED by §0b (2026-06-13): this was a batch-size confound.** With fresh_ab + an
> adequate batch, resampling works — the 0.9297 best was a resampled run. What ES needs is batch
> *coherence*, not a frozen set. The text below describes the small-batch regime only.

Per-step ES updates have cosine ~√(N/D) with the true gradient ≈ 0.003–0.006 here; signal exists
only as accumulation over hundreds of steps of the SAME objective.
- **Resampled (fresh batch each step): flat.** Old pop-2044 run: 88 steps @490 s, probe
  0.734–0.773 no trend. TFB8 continuation (batch=8, 896 pairs): flat through step 230+. Train
  reward oscillates wildly step-to-step (non-stationarity signature).
- **Stationary (frozen train set): descends smoothly and monotonically every time.**
- (Contrast: gradient-SFT thrives on resampled batches — 0.93 held-out in 15 steps — because
  its per-step gradient is high-quality. ES's is mostly noise; only accumulation saves it.)
- Note `RESAMPLE_TRAIN_EACH_STEP=1` is *epoch-cycling* over a 512 pool (each batch recurs every
  pool/train_size steps), not iid — at TRAIN_SIZE=8 the 64-step revisit period was too long.

### 2.2 With stationarity, ES generalizes — and train-set size controls how much
EggRoll-ES at full 35B-hybridattn dimensionality extracts a real generalizing direction:

| run | frozen train | pairs | train exact (cold→) | held-out greedy (cold→) | memorize:generalize |
|---|---|---|---|---|---|
| STAT64 | 64 | 128 | 0.734→0.922 @300 (saturated) | 0.742→peak 0.8125 (@115–145), plateau ~0.775 | ≈6:1 @300 |
| STAT256 | 256 | 128 | 0.660→0.813 @300 (still moving) | 0.750→**0.80–0.836 sustained** through 480+ | ≈2:1 |
| N512 | 64 | 512 | (same recipe as STAT64) | hit STAT64's peak @100, **exceeded it @140 (0.820–0.828)** | — |

- **Train-size scaling:** 4× the frozen set ⇒ ~2.5× higher generalizing share AND a higher
  held-out ceiling (+7–8.6 pts vs +3–6). A frozen set is effectively full-batch optimization;
  bigger batch ⇒ the cheapest descent direction is shared circuitry, not per-answer lookup.
  NOT saturated at 256 → STAT512/1024 is the obvious next point.
- **Population scaling:** z-scored update norm halves at 4× pairs (577.7 = 1155/√4, exactly as
  theory predicts: signal displacement per step is N-independent, noise ∝ 1/√N). Result: same
  loss curve, but held-out rises ~3× sooner and passes STAT64's ceiling — the STAT64 plateau
  was partly ES-estimation noise, not purely a data ceiling.
- **Loss descent ≠ learning answers.** Early descent is probability sharpening on
  already-correct answers (STAT64 @45: loss down 0.23, train exact UNCHANGED). Argmax flips
  start later (@100+). Always read the train/held-out probes, not the loss.
- `pair_delta_std` decays as the run saturates (0.089→0.003) — a useful saturation signal.
  `pair_delta_mean ≈ 0` is EXPECTED (zero-mean ⟨ε,∇f⟩), not a failure.

### 2.3 Open frontier
> **⚠ UPDATED by §0b (2026-06-13): best ES is now 0.9297** (fresh_ab + resample, rank-16,
> 12 h / ~280 steps) — at the 0.93 reference. The decisive open experiment is a long uncapped
> r8-vs-r16 ceiling comparison (see §0b "early-climb ≠ converged ceiling"). Below is the 06-11 list.

Gradient-SFT reference is 0.93; best ES so far 0.836. Levers, in expected order of value:
1. **STAT512/1024** (scaling curve toward the reference; ~2–4 min/step with batching).
2. **Pairs ↑ at fixed train size** (N512 showed ceiling+speed gains; loaded-loras has room).
3. **Slow-cycle resampling** (large TRAIN_SIZE + occasional re-draws) — tests ES on slowly
   varying objectives, the bridge to data that doesn't fit one static batch.
4. lr/σ tuning past the proven 0.03/0.012 (untouched in these runs).

---

## 3. INFRASTRUCTURE & THROUGHPUT

### 3.1 Pools (all Qwen3.6-35B-A3B bf16, TP=2/replica, fork `bc1ffb583`)
| pool | manifest | replicas | notes |
|---|---|---|---|
| `zorl-ar-sglang` (main) | `k8s/qwen3_6-35b-a3b-zorl-sglang-tp2-shard.yaml` | 16 (14 healthy; 4 & 6 CSI-stuck) | mem-frac **0.90** |
| `zorl-ar-sglang-c` | `k8s/qwen3_6-35b-a3b-zorl-sglang-c-tp2-shard.yaml` | 8 | mem-frac **0.85** (OOM headroom, §5) |
| `zorl-ar-sglang-d` | `k8s/qwen3_6-35b-a3b-zorl-sglang-d-tp2-shard.yaml` | 8 | clone of C |

Serving config (validated): `--max-lora-rank 16 --max-loras-per-batch 32 --max-loaded-loras 160
--max-running-requests 128` (effective mrr ≈ 110), triton LoRA backend, `hybrid_shared` +
`--experts-shared-outer-loras`, cuda-graph off. LoRA slots are pre-allocated and ~free; the
memory consumers are GDN/mamba state + KV (scale with mrr) and score-time logits (§5).
Per-pod DNS: `zorl-ar-sglang[-c|-d]-N.zorl-ar-sglang[-c|-d]-headless.apanda.svc.cluster.local:30000`.

**Clone-pool pattern:** `sed 's/zorl-ar-sglang/zorl-ar-sglang-X/g'` on the manifest + adjust
replicas → own pool on free fragmented GPUs in ~7 min. Used for C/D so experiments run in
parallel with the main pool's one-client lock (and gives a pristine base — EggRoll merges
MUTATE the served base; only a pod restart resets it). Startup ~6–7 min, `Parallel` pod
management. CSI mount throttle can stick pods in ContainerCreating; force-delete to retry.

### 3.2 Throughput (measured 2026-06-11)
**Cost model: step time scales with DISTINCT-LoRA applications (= requests per candidate), not
sequences.** The shared-outer LoRA-MoE forward applies per-expert SGMV over 256 experts once
per distinct LoRA per request; batching a candidate's examples into one request amortizes it.

| config | pairs | train ex | batch | s/step |
|---|---|---|---|---|
| legacy batch=1 (pop-2044 run) | 1022 | 8 | 1 | **490** |
| TFB8 (same pool, same science) | 896 | 8 | 8 | **69 (7.1×)** |
| STAT64 (pool C) | 128 | 64 | 32 | **26** |
| STAT256 (pool C) | 128 | 256 | 32 | **65** |
| N512 (pool D) | 512 | 64 | 32 | **135** |

`t_apply` ≈ 2 s (GPU noise path), merge ≈ 2 s, probe (128 greedy) ≈ 30–60 s per interval.

### 3.3 Concurrency sizing (MANDATORY with batched scoring)
- `SCORE_MAX_WORKERS_PER_OWNER × TEACHER_FORCED_BATCH_SIZE ≤ effective mrr (~110)`
- distinct LoRAs in flight per replica (= workers/owner) `≤ max-loras-per-batch (32)`
- **`TEACHER_FORCED_LOGPROB_TRIM=1` always** (§5). Proven shapes: batch=8/workers=12,
  batch=32/workers=3. (Wordle's "batch 16 unstable" was 64 global workers with NO per-owner
  semaphore = 1024 in-flight seqs; the per-owner cap is what makes batching safe.)

---

## 4. HOW TO RUN

### 4.1 Launch (one ZORL client per pool at a time — it holds a server-side session)
```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/autoresearch
python3 controller.py launch --candidate candidates/<CAND>.yaml \
  --set-env WANDB_NAME=<name> [--set-env KEY=VAL ...]   # --dry-run / render --output to verify
```
Candidates (all batched + trim, EggRoll merge-every-step, σ=0.012, lr=0.03):
- `MULTOPSD-EGGROLL-35B-STAT64.yaml` — frozen 64, 128 pairs, pool C.
- `MULTOPSD-EGGROLL-35B-STAT256.yaml` — frozen 256, 128 pairs, pool C.
- `MULTOPSD-EGGROLL-35B-STAT64-N512.yaml` — frozen 64, 512 pairs, pool D.
- `MULTOPSD-EGGROLL-35B-TFB8.yaml` — resampled TS=8, 896 pairs, main pool (the control).
Population = 2 × NUM_SHARDS(auto=#URLs) × PAIRS_PER_SHARD; keep ≤ loaded-loras×replicas.
The client job runs `standalone/zorl_client.py` from THIS worktree on the PVC (code/candidate
edits are live for new launches). Client jobs are `backoffLimit: 0` — a pool crash fails them
terminally (no zombie crashloop).

### 4.2 Monitor
- Step lines in `kubectl logs` / `RESULT_ROOT/<run>/zorl_client.log`:
  `step N: reward_mean … used_pairs … t_score … probe_reward=… exact_rate=…`.
- W&B project `zorl`, group `MULTOPSD-EGGROLL-35B`.
- Adapter exports every `EXPORT_INTERVAL=32` steps under `RESULT_ROOT/<run>/exports/`.
- Train-vs-held-out decomposition: `standalone/probe_train_batch.py` (§1). Run it at a few
  step milestones; it's read-only and safe against a live pool.
- `kubectl logs -f` streams break silently after ~1–2 h — poll pod phase to detect real
  termination, don't trust stream EOF.

### 4.3 Hygiene
- Pristine base needs a pool pod restart (merges mutate the base in place). A run's step-0
  cold probe re-measures the actual starting baseline either way.
- Don't route ES through the SMG (round-robin, no LoRA affinity → scatter + churn). `owner`
  routing is correct.
- Never `torchrun`/heavy work on the shared dev pod; k8s jobs only.

---

## 5. EFFICIENCY FINDINGS (what made the 7–19× and what bit us)

1. **Batched teacher-forced scoring** (`TEACHER_FORCED_BATCH_SIZE>1`): all of a candidate's
   examples in one `/generate` (list `input_ids` + per-seq `logprob_start_len`) → LoRA applied
   once per request. THE lever; 490→69 s/step at identical science config.
2. **The OOM landmine:** score-time input-logprobs materialize
   `[in-flight seqs × prompt positions × vocab]` fp32 at `logits[input_logprob_indices]`
   (logits_processor.py:378). 96 seqs × 39 positions = 2.30 GiB vs 2.29 GiB free at mem-frac
   0.90 → ALL replicas of pool C OOM-crashed simultaneously on the first wave (in-process ZORL
   sessions + LoRAs wiped; the client then sees "LoRA never loaded" / "Unknown ZORL session").
   **Fix:** `--teacher-forced-logprob-trim` (logprobs only over the answer suffix + 8-token
   margin ≈ 16 positions; suffix-based extraction makes it index-safe) — plumbed through the
   client job yaml as `TEACHER_FORCED_LOGPROB_TRIM` (default 0, set 1 always) — plus mem-frac
   0.85 on scoring pools and the §3.3 sizing rule.
3. **Per-owner semaphores** (`SCORE_MAX_WORKERS_PER_OWNER`) are what make batch sizes >8 safe.
4. `XORL_ZORL_NOISE_DEVICE=gpu` keeps apply ~2 s at pop ≤1024.

---

## 6. THE 35B SHARED-OUTER LoRA FIX (RESOLVED @ `bc1ffb583`)

Serving the attn+experts LoRA on Qwen3.6-35B looked like a forward hang; it was 4 stacked bugs
(full writeup `SHARED_OUTER_LORA_MOE_35B_FORWARD_HANG.md`): `_lora_pattern` matched nothing on
this arch (nothing got LoRA-wrapped), gated-qkv buffer sizing (out 9216 not 5120), GDN-layer
adapter weights crashing the load (now skipped), PID-1 SIGQUIT masking the crash. **Use the
`…-hybridattn` adapter**; the generic `init_zorl_adapter.py` qkv sizing builds a rejected
ungated shape on this model — derive attn-only variants from hybridattn instead of rebuilding.

---

## 7. SFT SCORE MODE (`--score-mode sft`)

`sft` reuses the teacher-forced answer-logprob machinery with reward = mean answer logprob
(= −SFT loss) via `use_logprob_reward`; in `probe_parent` sft deliberately probes by greedy
exact-match (headline stays accuracy). `tasks/mult.py::build_teacher_forced_example` builds
`chat(user)+"Answer: "+product` with `teacher_target_token_count` metadata; extraction takes
the LAST target_count logprobs (trim-safe).

---

## 8. GRADIENT STACK (rungs 1–2, EP=4×8) — reference

Worktree `/home/apanda/xorl-apanda-dev-opd-port`, stack `zorl-repro-q36-35b`. EP=4×8 = 32 GPUs
as 8×4-GPU pods for fragmented capacity; both rungs reproduced the fast rise. Gradient-SFT
reference on this task: ~0.93 held-out in ~15 steps. Detail:
`project_opd_ep4x8_fit_fragmented_capacity`.

---

## 9. FULL-VOCAB OPSD KL (alternative objective, server-side)

`sglang/srt/zorl/opd_kl.py` + `/generate` field `opd_kl={role,key,cot_len,mode,…}` computes
full-vocab KL on-GPU (teacher hidden-state caching, scalar reward back). `forward_kl_full` =
default. For ES, SFT > KL (KL was flatter). Design: `OPSD_ZORL_FULLVOCAB_KL_DESIGN.md`.
`merge_zorl_parent_into_base` (the EggRoll fold) also lives in the fork.

---

## 10. GOTCHAS (operational, still current)

- One ZORL client per pool. Pool re-tune (mem-frac etc.) = edit manifest + apply + delete pods.
- mrr is the GDN/mamba+KV memory lever; keep 64–128 for scoring. Loaded-loras are ~free.
- CSI mount throttle on many simultaneous pod starts; force-delete stuck pods to retry.
- `kubectl logs -f` breaks silently on long follows; verify pod phase before declaring a run dead.
- Train-slice cold baselines differ from eval (256-slice = 0.660 vs eval 0.74–0.76) — measure
  before claiming train-side gains.
- Agent-ops note: the permission layer may block deleting cluster resources not created in the
  current session — clone-pool (§3.1) + self-launched jobs avoid the constraint entirely.
- Don't touch other agents' stacks (`er-opd-*`, Wordle SMGs). No `git stash -u` in bg tasks.

---

## 11. FILES

- `standalone/zorl_client.py` — ES client (batched scoring + trim + `--merge-every-steps`).
- `standalone/probe_train_batch.py` — train-batch generalization probe (NEW).
- `standalone/tasks/mult.py` — task + teacher-forced example builder.
- `k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml` — client job (trim plumbed).
- `k8s/qwen3_6-35b-a3b-zorl-sglang{,-c,-d}-tp2-shard.yaml` — the three pools.
- `k8s/cal_pod_template.yaml` — single-replica calibration pods.
- `autoresearch/candidates/MULTOPSD-EGGROLL-35B*.yaml` — the experiment matrix.
- Results/exports: `experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B*/`.
