# ZORL: Evolution Strategies for Frontier-Scale LLMs — The Canonical Document

*2026-07-02. Audience: someone building an explainer (webpage/talk) from this material.
It covers the science and the systems end-to-end; it is not a runbook (those are
linked at the bottom). Everything here is sourced from code and measured runs —
file:line anchors and run URLs are given throughout.*

---

## 0. One paragraph

ZORL ("Zeroth-Order RL") trains a frozen large language model with **Evolution
Strategies**: no backprop, no gradients — only forward inference and scalar rewards.
The core trick that makes this affordable is that **every candidate model, every
update, and even optimizer momentum is just a random seed**: a candidate is a seeded
low-rank perturbation the inference engine regenerates on demand, so an entire
population of models is *communicated* as a handful of integers and *served* by one
inference fleet at once. Around that idea we built a full system — multi-LoRA
population serving inside SGLang, MoE "virtual experts" grouped-GEMM kernels,
CUDA-graph-compatible serving, a Muon (Newton–Schulz) optimizer that runs in
LoRA space, and an fp32-master parameter server — that took ES from a
research-preview toy to training a **35B-parameter MoE** on real tasks, matching a
gradient-descent (SFT) reference on 4×4-digit multiplication (**0.74 → 0.93**
held-out exact-match, vs the gradient ceiling ≈ 0.93).

---

## Part I — The science premise: ES and EGGROLL

### 1.1 Evolution Strategies in one equation

ES treats the model as a black box. Perturb the weights with random noise, measure
scalar fitness (reward) on the task, and move the weights toward the perturbations
that scored well. The estimator (as stated in the reference repo
`~/HyperscaleES`, `eggroll.ipynb`):

```
∇θ E[F(θ + σ·ε₂ε₁ᵀ)]  =  (1/σ) · E[ F(θ + σ·ε₂ε₁ᵀ) · ε₂ε₁ᵀ ]
```

Sample N perturbations, score each, and form the **fitness-weighted sum of the
perturbation directions**. No backward pass ever runs; the model only does
inference. Two standard refinements:

- **Antithetic pairs**: evaluate each noise draw at +σ and −σ. The pair's score
  difference `r⁺ − r⁻` isolates the effect of the direction and cancels the
  first-order noise — a much lower-variance estimator.
- **Advantage normalization**: z-score the raw rewards (optionally per prompt-group,
  GRPO-style) before weighting.

### 1.2 EGGROLL: the low-rank trick

The naive perturbation of a weight matrix `W ∈ R^{out×in}` costs `O(out·in)` memory
*per candidate* — hopeless for a population on an LLM. **EGGROLL** ("Evolution
Guided General Optimization via Low-rank Learning", `~/HyperscaleES`,
`src/hyperscalees/noiser/eggroll.py`) replaces the dense noise with a **rank-r outer
product**:

```
ΔW = σ · B · Aᵀ        (A ∈ R^{in×r}, B ∈ R^{out×r}, entries ~ N(0,1) from a seed)
```

Three consequences:

1. **Memory per candidate is O((in+out)·r)** instead of O(in·out) — and since A,B
   are generated from a PRNG seed, you don't even store *that*: you store the seed.
2. **The perturbed forward pass never materializes ΔW**: `x·(W+ΔW)ᵀ = x·Wᵀ + (x·B)·Aᵀ`
   — two skinny matmuls bolted onto the base GEMM. This is *exactly* the shape of
   LoRA inference, which is the observation ZORL's serving stack is built on.
3. **Low-rank steps, full-rank learning**: each generation's update
   `G = (1/N)·Σᵢ cᵢ·σ·(Bᵢ·Aᵢᵀ)` has rank ≤ N·r, and it is added into the **dense**
   base weights — so across generations the accumulated change spans full rank.
   Low-rank is a per-step probe geometry, not a capacity cap (provided the update
   lands in the base — see §4.2).

### 1.3 Why anyone should want this

- **No backprop** → no optimizer state sharding, no activation memory, no
  backward kernels. Training hardware = inference hardware.
- **Trivially parallel**: candidates are independent; scoring scales linearly with
  inference replicas.
- **Reward can be anything scalar** — non-differentiable, sparse, multi-turn,
  end-to-end (a whole Wordle game, a verified integer product).
- **Communication is seeds** (§3.4): the entire distributed-training communication
  problem collapses to integers + scalar rewards.

---

## Part II — Why this is hard to scale to real tasks

`~/HyperscaleES` is an honest research preview (its README says so), and its gaps
define exactly the problem ZORL solves:

| HyperscaleES (reference) | What real tasks need |
|---|---|
| Weights fully **replicated** per GPU (`shard_map` over the population only) — the model must fit on one GPU | 35B+ MoE models served TP≥2, tens of replicas |
| Hand-rolled token-by-token `jax.lax.scan` generation (`llm_experiments/utils.py:31-61`) | A real inference engine: paged KV-cache, continuous batching, CUDA graphs — 10–100× throughput |
| RWKV-only model zoo (0.1B–14B), softmax-attention transformers absent | Production transformer/MoE checkpoints (Qwen3.6-35B-A3B) |
| Single-turn "bandit" tasks (one prompt → one generation → scalar) | Multi-turn, agentic rollouts (6-turn Wordle with environment feedback) |
| Fitness scored in Python loops on CPU | Server-side scoring (teacher-forced logprobs, full-vocab KL) at fleet scale |
| bf16 weights, no quantization; naive `.astype(dtype)` update application | FP8-served models; **sub-ULP update arithmetic** (Part V — this one is subtle and it silently kills the whole method) |
| One process = one experiment | A population *shared* across an inference fleet, with an update authority |

The one-line version: **EGGROLL made the math cheap; nothing made the *system*
exist.** ES at scale is an inference-systems problem (serve N models at once, fast)
plus a numerics problem (apply microscopically small updates without losing them),
plus an algorithm-engineering problem (make the estimator behave on real reward
surfaces). ZORL is those three things.

---

## Part III — The ZORL system

### 3.1 Architecture

```
             driver (CPU pod) — zorl_client.py / run_wordle_zorl_xorl_ps.py
             owns the loop: start generation → score → send rewards
                   │                                   │
        (seeds only│lockstep)                  (scalar rewards only)
                   ▼                                   ▼
   ┌───────────────────────────┐      ┌─────────────────────────────────┐
   │ PARAMETER SERVER           │      │ SCORER FLEET                    │
   │ xorl trainer (fp32 master, │      │ N× SGLang replicas (TP=2 each), │
   │ Muon optimizer, seed       │─────▶│ FP8 base + multi-LoRA; each one │
   │ authority)                 │ sync │ serves the WHOLE population as  │
   │                            │      │ seeded LoRA candidates          │
   └───────────────────────────┘      └───────────────┬─────────────────┘
                                                       │ /generate round-robin
                                                 SMG router (Rust)
```

- The **population lives on the inference fleet**: every replica reconstructs every
  candidate from seeds, so scoring requests round-robin freely.
- The **parameter server** is the only writer. It holds the fp32 master weights,
  reconstructs the same perturbations from the same seeds, folds the reward-weighted
  update through a real optimizer (Muon), and syncs the served weights.
- The **driver** never touches weights at all. Its entire uplink is
  `{generation seeds}` down and `{candidate_id → reward}` up.

### 3.2 Multi-LoRA population serving (one replica = the whole population)

A candidate is `(parent, seed, σ, sign)` — nothing else. Inside SGLang
(`python/sglang/srt/lora/lora_manager.py` on the `zorl-ps-fp32` branch of the fork):

- **Virtual candidates / virtual slots.** Registering a population creates
  seed-only specs (`create_zorl_lora_candidates`, `lora_manager.py:2736`) —
  `{parent, b_seed, a_seed, direction, σ, mode}` with no weight bytes. A candidate
  occupies a real slot in the GPU LoRA memory pool **only while a batch references
  it** (`_materialize_zorl_candidate`, `:2668`); the pool holds
  `max_loras_per_batch` residents regardless of how many candidates are registered.
  N candidates registered, K resident: the classic virtual-memory move, applied to
  model weights.
- **Seeded backend** (`XORL_ZORL_SEEDED_BACKEND`): even the CPU-side adapter is
  stripped to a seed-only stub (`_make_zorl_seeded_stub`, `:2698`) — after the GPU
  buffers are seed-filled, `layer.weights = {}`. Host memory for the population is
  ~zero; eviction and re-load just re-runs the RNG.
- **Routing**: a scoring request selects its model by `lora_path=<candidate name>`
  in the ordinary `/generate` payload. With every replica reconstructing the whole
  population, a compiled Rust router (SMG) round-robins requests across the fleet.

### 3.3 Making it fast

**Serving side** (measured ~10× stack, `ZORL_WORDLE_FRESH_AB_FP8_ALGORITHM_AND_INFRA.md` §3.2):

- **MoE virtual experts** (`--lora-use-virtual-experts`, `lora/layers.py:916-1740`):
  the killer problem is MoE + many LoRAs. Chunked-SGMV kernels reload a LoRA tile
  *per dispatched token*; with a population resident that's prohibitive. The fix
  treats each `(lora, expert)` pair as a **virtual expert** and runs the LoRA delta
  through the same grouped-GEMM machinery as the base MoE: tokens are
  block-sorted by `(lora, expert)` (`moe_align_block_size`), each weight tile loads
  once per block, and the delta adds directly into the base GEMM's output buffer.
  Cost becomes independent of how many adapters are in the batch — "a handful of
  kernel launches regardless" (`layers.py:1680`).
- **CUDA graphs**: virtual-expert routing is GPU-only (no CPU sync), verified
  capture-safe on Qwen3.6-35B-A3B (~2.8k tok/s with graphs on); requires
  `--disable-custom-all-reduce`.
- **FP8 base + multi-LoRA**: the scorers serve a block-FP8 (E4M3, 128×128-block)
  base for memory/throughput; block scales and quant metadata are threaded through
  the hand-rolled fused-MoE GEMM calls (`TritonMoeQuantInfo`, `layers.py:891-905`)
  so LoRA deltas add onto an FP8 base correctly.
- **GPU noise generation** (`XORL_ZORL_NOISE_DEVICE=gpu`): one batched CUDA `randn`
  per pair instead of per-tensor CPU draws — **~480×** on the noise core
  (~62 s → ~0.13 s per step).
- Throughput anchor: ~133k tok/s aggregate across a 16-replica pool at
  population 1024 (~4.1k tok/s per GPU) on the 35B MoE.

**Update side** (the fold used to cost 170–226 s/step; `MUON_APPLY_PROFILE.md`):

- **Newton–Schulz was never the bottleneck** — batched over experts it is ~0.13 s
  (<1%). The cost was regenerating each pair's noise once *per module* (~60×
  redundancy, ~38,400 regenerations/step).
- **O(1)-in-N accumulation**: build each module's dense `G` incrementally
  (`dense_accum`), orthogonalize once, apply, free — peak memory is one module's G,
  independent of population size.
- **Band-major fold** (`XORL_ZORL_MUON_FOLD_BAND`): draw each pair's noise once per
  *band of layers* instead of per module — bit-exact to the naive order, ~6–60×
  fewer redraws, pushing the fold toward a ~30–60 s floor. (The xorl-PS successor
  folds once with a draw-once loop and is expected in single-digit seconds.)

### 3.4 Seeds are the entire communication layer

This deserves its own subsection because it's the deepest systems consequence of
the method. In ZORL **no weight bytes ever cross the network during training**:

- **Candidate distribution**: the driver tells PS + replicas
  "generation g, base seed S". Both sides derive
  `seed_i = f(base_seed, family, generation, pair_index)` and regenerate identical
  `A_i, B_i`. Shipping a population of 128 rank-16 candidates = shipping one
  integer.
- **The update**: the driver sends back `{candidate_id: reward}` — a few KB of
  floats. The PS reconstructs every ΔWᵢ from seeds and folds
  `G = Σ cᵢ·ΔWᵢ` locally.
- **Momentum is seed replay** (`lora_manager.py:3362-3441`): instead of a dense
  velocity tensor (which for a 35B model would be another 70 GB), fresh_ab momentum
  keeps a bounded history of `(b_seed, a_seed, weight)` triplets and re-folds each
  live historical step at `lr·βᵃᵍᵉ`. The optimizer state *is also seeds*.
- The only bulk transfer in the system is the **serving sync** after a base fold
  (fresh_ab mode): the PS pushes the changed served view to replicas — and even
  that is sparse (§5.4) or a dense RDMA push reusing the proven GRPO weight-sync
  path (~5–10 s for the whole 35B to a fleet).

---

## Part IV — The update algorithm

### 4.1 One ES step, end to end

```
1. driver: start_zorl_generation(seed=S, gen=g)        [seeds only, lockstep]
2. replicas: regenerate the N-pair population from seeds
3. driver: score all candidates on a train shard (round-robin) → rewards rᵢ
4. driver: POST rewards → PS
5. PS:  cᵢ = z-scored pair advantages (antithetic: signal is r⁺−r⁻)
        G  = (1/N)·Σᵢ cᵢ·scaling·(Bᵢ·Aᵢᵀ)          [reconstructed from seeds]
        step = Muon: p += lr_adj · NewtonSchulz(G)     [exact, in fp32 master]
        sync the served view to replicas
6. every K steps: greedy held-out probe → the honest learning curve
```

### 4.2 The perturbation-mode family

All modes share antithetic pairing (a pair shares its noise, differs in sign) and
the reward-weighted fold; they differ in *what* is perturbed and *where* the update
lands (`lora_manager.py:2297-2422`, `:2888+`):

- **`b_only` (parent-perturb).** A LoRA parent (A fixed, B learned) is the model
  state; candidates perturb B only: `B = parent_B ± σ·ε_B`. The update lands in the
  parent's B. Simple, no base sync needed — but **capacity-capped at rank r**: the
  model can never leave the r-dimensional B-subspace (with fixed A). This was the
  original production path and its rank cap is the confound fresh_ab removes.
- **`fresh_ab` (EGGROLL proper).** Candidates *replace* the factors with fresh
  seeded gaussians each generation: `A = ε_A` (pair-shared), `B = ±σ·ε_B`. The
  parent's B stays identically zero forever — the adapter is pure scratch. The
  update `G = Σ cᵢ·(ε_B,ᵢ·ε_A,ᵢᵀ)` has rank ≤ N·r, which **cannot fit in a rank-r
  adapter**, so it folds **directly into the base weights**. Full-rank accumulation,
  no cap; the price is the precision problem of Part V and a serving sync per step.
- **`a_and_b`.** Perturb both factors around the parent. Supported; not the
  production path.
- **Noise variants**: shared-A-per-step, fixed-A, and **Rademacher-B**
  (`XORL_ZORL_RADEMACHER_B`: B-noise projected to ±1 — SPSA-style, 1-bit-friendly,
  and it climbed just as well: 0.89 on mult).
- **Update shaping** (client `--update-strategy`): `raw` (server-side z-score),
  `centered`, `rank`, and `project_baseline*` variants (candidate − parent, with
  ReLU/standardization options).
- **Elitist rollback**: snapshot the parent (and fp32 master) at the best probe;
  roll back on regression — ES's cheap answer to late-run reward decay.

### 4.3 Muon in LoRA space

The reward-weighted sum `G` is anisotropic and ill-conditioned (a few
high-advantage directions dominate). **Muon** — orthogonalize the update via
Newton–Schulz before applying — whitens the spectrum, and is a natural fit because
it is *stateless* (momentum-off): no full-weight optimizer state to shard, which
matters when your "trainer" is an inference server. (Adam is deliberately
unsupported on fresh_ab: a full-rank second moment can't live in AB-space.)

Implementation (`lora_manager.py:1171-1266`, ported from xorl's optimizer):

- **What space NS runs in**: for `b_only`, the update lives in B-space — NS on
  `[out, r]`. For `fresh_ab`, the dense per-module `G` `[out, in]` (or `[E, out, in]`
  batched over MoE experts) is materialized and orthogonalized — correct because
  N·r ≥ every module dimension.
- **Gram Newton–Schulz**: for rectangular/low-rank G, iterate on the small Gram
  matrix `R = X·Xᵀ` (m×m, m = min dim) and materialize `Q·X` once at the end —
  quintic, 5 steps, coefficients (3.4445, −4.775, 2.0315), bf16 compute
  (direction cosine vs fp32: 0.997).
- **LR transfer**: `match_rms_adamw` scaling — multiply the unit-singular-value
  update by `0.2·√max(rows, cols)` so its per-element RMS matches AdamW's, letting
  Muon **reuse the AdamW/GRPO learning rate** directly (Moonshot's "Muon is
  Scalable" rule). Getting this wrong (the Keller-Jordan `√(rows/cols)` scale,
  ~9–15× smaller) was a real flat-curve bug.
- **Numerical parity**: the xorl-trainer Muon and the sglang in-server fold were
  validated bit-for-bit against each other in fp32 (cos = 1.0000, relerr ≈ 1e-6;
  `PS_AS_XORL_TRAINER_R1_VERDICT.md`).

---

## Part V — Why an fp32 parameter server (the precision story)

This is the least obvious and most load-bearing part of the whole project.

### 5.1 The sub-ULP problem

A converged ES step is *tiny*. At the live recipe, the per-weight Muon step RMS is
**~1.6e-6**. The bf16 ULP at a typical weight magnitude (0.02) is **~1.22e-4** —
the update is **~75× below one ULP**. `w.add_(update.to(bf16))` therefore rounds
back to `w` almost every time: measured on CPU, a bf16 base **retains only 2–5%**
of the intended cumulative movement, at any step count
(`FP32_MASTER_WEIGHTS_DESIGN.md:68-101`). The learning curve looks like the
algorithm failed; actually the *arithmetic* failed. (An earlier SGD recipe with
~60× larger steps sat *at* the ULP and climbed — which masked the bug for weeks.)

FP8 is worse: on a block-FP8 base each step is ~0.1 ULP, deterministic rounding
zeroes ~90% of the signal, and even stochastic rounding drifts (per-step block
re-quantization re-rounds untouched entries; rel-err 0.05 → 0.21 over 40 steps;
live A/B: FP8 fold collapsed 0.69 → −0.57 while bf16 reproduced the 0.93 climb).

### 5.2 The fix: exact accumulation, downcast view

Keep an **fp32 master** of every folded weight; accumulate the update there
*exactly*; the served bf16/FP8 weight is a **downcast view** of the master.
Retention goes from 2–5% to ~100% — sub-ULP steps sum in fp32 and cross serving
ULPs when they've genuinely accumulated. Interim mitigations that also work and are
implemented: **stochastic-rounding folds** into bf16/FP8 (unbiased against the true
non-uniform E4M3 grid, `fp8_fold.py`), with a "lazy-scale" mode that keeps block
scales fixed so untouched entries re-encode exactly.

### 5.3 Where the master lives: the parameter server *is* a trainer

The full fp32 master of the folded set is 64.8 GB/rank — it doesn't fit beside the
serving weights. More importantly, "hold fp32 masters + run a real optimizer + sync
weights to inference replicas" is *exactly what an RL training server already is*.
So the PS is an **xorl trainer** (branch `zorl-ps`, `src/xorl/server/zorl.py`):
FSDP fp32 masters for free, the original Muon (not a re-port), DCP checkpointing
(elitist rollback = checkpoint restore), and the proven GRPO p2p RDMA weight-sync
(reshards trainer-layout → each replica's TP rank; full 35B in ~5–10 s). The
first sglang-hosted PS implementation exists and is CPU-validated but is
deprecated in favor of this (`PS_AS_XORL_TRAINER_DESIGN.md` §9 has the full
argument). The R1 gate — xorl Muon ≡ sglang fold, seed→noise streams bit-identical,
fresh_ab fold parity cos = 1.0 — passed on 2026-06-30/07-02.

### 5.4 Syncing an FP8 fleet without corrupting it

For fresh_ab the served base changes every step. Two transports:

- **FP8-view sparse diff**: RTN-quantize the fp32 master onto the replicas' current
  128×128 block grid (bump a block's scale only on saturation) and diff against the
  last-synced view. Unmoved blocks reproduce bit-for-bit → empty diff; only weights
  that actually crossed an FP8 code ship. Replicas end up serving
  `quantize_fp8(master)` **bit-for-bit identically** — no drift, cross-replica
  equality is checkable by hashing.
- **Dense p2p RDMA push** with trainer-side block-FP8 quantization (E4M3 + fp32
  scales) — the same path GRPO uses, ~5–10 s for the full model.

---

## Part VI — Results

### 6.1 The multiplication testbed

Task: `Calculate: A * B` for 4-digit A, B (single-shot; `tasks/mult.py`). Why it's
the right ES testbed: the reward is **verifiable** (exact integer match); the
training score is **deterministic** (teacher-forced mean logprob of the true
product digits at temp-0 → antithetic pair deltas carry zero sampling noise); the
held-out probe (greedy exact-match on 128 disjoint problems) directly measures
generalization; and there's a **known gradient ceiling** to compare against
(gradient-SFT reaches ~0.93). "MULTOPSD" = multiplication + on-policy
self-distillation: the same weights, prompted with a private CoT scratchpad, act as
teacher for the direct-answer student.

### 6.2 The ladder (Qwen3.6-35B-A3B, cold direct accuracy ≈ 0.74)

| Rung | Method | Held-out exact-match |
|---|---|---|
| 1 | Gradient full-FT (early) | 0.508 → 0.625 |
| 2 | Gradient LoRA (attn-only) | 0.664 → 0.680 |
| 3 | **ZORL b_only + merge-every-step** (STAT256) | 0.74 → **0.80–0.836** sustained |
| 4 | **ZORL fresh_ab + resampled batches** (FRESH-RESAMPLE) | 0.74 → **0.9297** |
| — | Gradient-SFT reference | ~0.93 |

**ES matched the gradient reference on a 35B MoE, using only forward passes.**
The winning recipe: `fresh_ab`, rank 16, σ = 1.5e-4, lr = 3.8e-4 (plain SGD-raw
fold), 128 pairs (256 candidates), train batch 256 **resampled each step** from an
8k pool, ~280 steps / 12 h on an SGLang pool. The two biggest wins the search
found: fresh_ab beat b_only ~2× per-step (rank freedom), and resampling beat a
frozen train set (what ES needs is *batch coherence* — measured gradient coherence
0.882 at batch 256 — not a stationary objective).

Robustness reproductions from the same family (all in the results tree, all
climbing to the 0.84–0.90 band): SGD-REPRO 0.75 → 0.898, Rademacher-B noise
0.74 → 0.891, shared-A 0.75 → 0.883, LR-decay variants 0.72–0.73 → 0.87–0.88,
8-shard scoring 0.74 → 0.859. The effect is not a lucky seed.

### 6.3 The promising-runs script (wandb)

`experiments/zorl/wandb_filter_promising.py` scans
`https://wandb.ai/together-research/zorl`, picks each run's primary score metric,
and ranks by **the run's own improvement** (best − first), so warm starts don't
masquerade as learning:

```
python experiments/zorl/wandb_filter_promising.py --top 40 --json promising.json
```

Top of the current leaderboard (2026-07-02; note `state=crashed` usually means the
autoresearch controller reaped the pod, not that the science failed — read the
curve, not the state):

| run | metric | first → best | steps |
|---|---|---|---|
| MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE | eval/exact_rate | 0.711 → **0.930** | 420 |
| MULTOPSD-EGGROLL-35B-FRESH-GDN | eval/exact_rate | 0.672 → 0.867 | 282 |
| CEILING-L-r16-b512 | eval/exact_rate | 0.703 → 0.898 | 376 |
| ZORL-MULT-SGD-REPRO-X | eval/exact_rate | 0.727 → 0.898 | 680 |
| ZORL-MULT-RADEMACHERB-BF16-Z | eval/exact_rate | 0.734 → 0.891 | 181 |
| MULTOPSD-EGGROLL-35B-STAT256-MOM | eval/exact_rate | 0.664 → 0.820 | 1013 |

(Runs named `GRPO-*` in the same project are the gradient-RL Wordle *baselines*,
not ZORL — the shared project is intentional so the curves overlay.)

### 6.4 Honest negatives and open problems

- **Muon-lr variants on mult were flat** (lr 2.5–5e-5 with Muon underperformed
  plain SGD at 3.8e-4 there) — the mult win is an SGD-raw result; Muon's value
  case is Wordle-scale steps where its conditioning matters.
- **FP8-substrate training collapsed** until the precision work of Part V; the
  0.93 climb ran on a bf16 base. This is *why* the fp32 master exists.
- **KL objective was flatter than SFT** for ES scoring ("For ES, SFT > KL").
- **Wordle (multi-turn) has not yet beaten its GRPO reference** (~0.67–0.68
  honest solve-rate). The b_only frozen-base arm exhausted its levers (rank cap +
  the bf16 sub-ULP drop were both implicated); fresh_ab-through-the-fp32-PS is the
  current live experiment. ES-vs-GRPO on multi-turn tasks is the open frontier,
  not a settled win.
- **Early climb-rate is a bad model-selector**: a 1 h HP search ranked r8 > r16,
  but the only *converged* run (r16 → 0.9297) beat r8's entire early peak. Seed
  variance on 1 h trials is ±0.03. Never trust n=1.

---

## Appendix

**Key code** (repos: `togethercomputer/xorl-sglang-internal` branch `zorl-ps-fp32`;
`togethercomputer/xorl-internal` branch `zorl-ps`; `xorl-client` / `xorl-infra`
branch `zorl-consolidation`):

| What | Where |
|---|---|
| ES core: modes, fold, Muon, seeded backend | sglang `python/sglang/srt/lora/lora_manager.py` |
| MoE virtual experts + FP8 quant threading | sglang `python/sglang/srt/lora/layers.py`, `lora/triton_ops/virtual_experts.py` |
| FP8 stochastic-rounding fold | sglang `python/sglang/srt/lora/fp8_fold.py` |
| fp32 master + FP8-view sync (deprecated PS) | sglang `python/sglang/srt/lora/zorl_fp32_master.py` |
| PS-as-trainer: seed authority, fold, endpoints | xorl `src/xorl/server/zorl.py`, `server/runner/model_runner.py` |
| Driver / tasks / probe | this repo: `experiments/zorl/standalone/{zorl_client.py, run_wordle_zorl_xorl_ps.py, tasks/}` |
| Run filter | this repo: `experiments/zorl/wandb_filter_promising.py` |
| Manifests / recipes | xorl-infra `k8s/zorl/`, `configs/zorl/` |

**Deeper reading** (in the dev worktree `experiments/zorl/`):
`ZORL_WORDLE_FRESH_AB_FP8_ALGORITHM_AND_INFRA.md` (algorithm+infra overview),
`PS_AS_XORL_TRAINER_DESIGN.md` + `_R1_VERDICT.md` (the PS pivot),
`FP32_MASTER_WEIGHTS_DESIGN.md` (the sub-ULP analysis),
`FP8_NATIVE_ADAPTER_ACCUM.md` + `FP8_VIEW_SYNC_DESIGN.md` (FP8 numerics),
`MUON_APPLY_PROFILE.md` (fold performance),
`OPSD_ZORL_RUNBOOK.md` (the multiplication campaign of record),
`ZORL_WORDLE_GRPO_FAITHFULNESS.md` (honest-eval methodology).

**Glossary.** *ZORL*: zeroth-order RL, this project. *EGGROLL*: low-rank seeded
ES perturbations (`~/HyperscaleES`). *fresh_ab / b_only*: perturbation modes
(§4.2). *Virtual candidates*: seed-only LoRA registrations. *Virtual experts*:
(lora, expert) pairs routed through grouped GEMM. *Muon / Gram-NS*: orthogonalized
update via Newton–Schulz on the Gram matrix. *fp32 master*: exact accumulator
behind the served low-precision view. *SMG*: the Rust request router.
*Antithetic pair*: ±σ evaluations sharing one noise draw. *OPSD*: on-policy
self-distillation (CoT-teacher → direct-answer student, same weights).
