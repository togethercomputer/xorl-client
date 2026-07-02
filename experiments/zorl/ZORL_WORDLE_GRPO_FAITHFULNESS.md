# ZORL-Wordle ↔ GRPO-Wordle Faithfulness Audit

**Date: 2026-07-01 (apanda).** Purpose: state, honestly and axis-by-axis, whether the
ZORL (Evolution-Strategies) Wordle run is a *faithful replication* of the canonical
GRPO Wordle experiment, differing **only** in the optimization algorithm. Reference:
`/home/apanda/xorl-client-wordle-science-20260614/experiments/wordle/WORDLE_RECIPE.md`
(GRPO canonical, 2026-07-01).

## Verdict (read first)

**We are NOT yet confident this is a faithful match.** The *model, prompt, reward, task
code, and rollout temperature are identical*, and the held-out-parity machinery exists in
code. But the current live run (`zorl-wordle-ps-35b-trainer`, cfg
`qwen3_6_35b_a3b_zorl_wordle_ps_muon.yaml`) has **six unintended deviations** from the
recipe — most importantly a **tiny ES population (8 pairs)**, a **retry crutch on the
reward**, **too few rollouts per candidate**, and an **eval set that is NOT GRPO's
held-out set**. These must be closed before any ZORL-vs-GRPO number is meaningful.

The intended (and only legitimate) difference is the optimizer:
**GRPO unclipped-importance-sampling policy gradient → ZORL reward-weighted LoRA-B
perturbation fold (Muon).** Everything else should match.

---

## 1. Faithful today (matched — do not touch)

| Axis | GRPO recipe | ZORL now | Match |
|---|---|---|---|
| Model | Qwen3.6-35B-A3B snapshot `995ad96e…` | same snapshot | ✅ |
| Prompt | `public_reasoning_constraints_think`, `scaffold-fade-steps 0` (static) | `public_reasoning_constraints_think` (static; no fade) | ✅ |
| Reward | `wordle_retrieval_reward` (verify each guess vs answer), `tasks/wordle.py` | same function, same file (breakdown commented "GRPO-parity"); `XORL_WORDLE_REWARD` default = retrieval | ✅ |
| Rollout task | multi-turn Wordle, verifiable per-guess retrieval | same `tasks/wordle.py` rollout | ✅ |
| Sampling temp | student-temperature 0.7 | rollout-temperature 0.7 | ✅ |
| Base floor | base+think = **0.00** (retries=0) | cold probe = **0.00** | ✅ |
| Held-out word set | `_pick_targets(seed=777)` reserved, kept out of train | **machinery exists** (`WORDLE_TRAIN_EXCLUDE_SEED/COUNT`, WORD_LIST byte-identical) | ⚠️ present but **OFF** (see gap #5) |

## 2. Deliberate algorithm differences (the whole point — expected, not gaps)

| Axis | GRPO | ZORL-ES | Why it must differ |
|---|---|---|---|
| Update signal | per-token policy gradient, backprop through 35B | reward-weighted perturbation fold `G=Σ cᵢ·ΔWᵢ`, forward-only | ES is zeroth-order (no gradients) |
| On-policy correction | **unclipped importance sampling** (the acquisition lever) | none | ES uses only the scalar reward |
| Parameterization | full-weight from base | **LoRA rank-16** (fp32 master lands small updates) | ES needs a low-dim search space |
| Trainer | EP8, fwd+bwd+optimizer | parameter server, **no backprop** (EP4 fine); fold applied in ~3s | ES does inference + a cheap fold |
| Weight delivery to samplers | p2p full-weight sync | candidate **LoRA adapters exported + loaded** | one adapter per ES candidate |
| k3 / logprob-temp / routed-expert IDs / KL stats | required (correctness for IS) | **dropped entirely** | ES never reads logprobs |
| "group" semantics | group-size 16 = rollouts of one prompt for **relative advantage** | population of **antithetic perturbation pairs** scored across train puzzles | different statistical object (see §3) |
| LR / schedule | muon, lr 5e-6, cosine, warmup 8 | muon-fold, muon_lr 1e-2, sigma 0.05 | full-weight vs LoRA-B scale; ES-tuned |

Note the LR values are **not** comparable across parameterizations — GRPO's 5e-6 is a
full-weight gradient step; ZORL's 1e-2 is a Muon step on a LoRA-B perturbation fold. These
are tuned per-algorithm and are not a faithfulness axis.

## 3. The ES "optimization budget" — GRPO analogs and current gaps

ES statistical quality is governed by three axes that have no single GRPO knob but map to
GRPO's rollout budget. **Total rollouts/step = num_pairs·2 · train_size · rollouts_per_puzzle.**

| ES axis | Role (variance of the ES update) | GRPO analog | GRPO value | **ZORL now** | Recommended |
|---|---|---|---|---|---|
| `num_pairs` (population) | # of antithetic perturbation directions averaged into `G`; **the dominant ES signal-to-noise lever** | (none — GRPO is single-policy) | — | **8 (16 cands)** ❌ far too small | **≥ 32–64 pairs** (the mult run that *climbs* uses **128**) |
| `rollouts_per_puzzle` | samples per candidate per puzzle → reliability of each candidate's reward | group-size (rollouts/prompt) | **16** | **4** ❌ | **8–16** |
| `train_size` (puzzles/step) | coverage; denoises pair-deltas across puzzles | train-size | **32** (pool 4096) | **16** (pool **256**) ❌ | **32**, pool ≥ 4096 |

At Wordle's ~2048-token think rollouts this is the **cost tension**: step-time ≈ linear in
`num_pairs·train_size·rollouts_per_puzzle`. Current step ≈ 22 min at 8·16·4 = 1024
rollouts. Faithful settings (32 pairs · 32 · 8 = 16384) are ~16× → not affordable naively.
**This is the central feasibility question the doc surfaces:** ES needs a large population
*and* enough samples per candidate, but that multiplies against Wordle's long rollouts. We
must either (a) accept fewer steps with a big population, (b) scale the sampler pool wide,
or (c) shorten rollouts — and decide deliberately, not by leaving small defaults in place.

## 4. Unintended gaps (must fix before any ZORL-vs-GRPO claim)

1. **Population too small** — `num_pairs=8`. Biggest lever. → 32–64+.
2. **Retry crutch on the reward** — `--invalid-retries 2` feeds **both** training reward
   and probe. The recipe is explicit: retries inflate every number; honest gate = **0**.
   ES is currently optimizing the *crutched* objective (lean on re-prompts instead of
   valid first-try guesses). → `--invalid-retries 0` for training **and** eval.
3. **Too few rollouts per candidate** — `rollouts_per_puzzle=4` (GRPO group-size 16). Noisy
   per-candidate reward → noisy `G`. → 8–16.
4. **Undersized train coverage** — `train_size=16` / `train_pool_size=256` vs GRPO 32 /
   4096. → 32 / ≥4096.
5. **Eval set is NOT GRPO's held-out** — `WORDLE_TRAIN_EXCLUDE_SEED/COUNT` unset ⇒ eval is a
   disjoint slice of the train pool, not the seed-777 reserved set. → set
   `WORDLE_TRAIN_EXCLUDE_SEED=777 WORDLE_TRAIN_EXCLUDE_COUNT=170`, `eval_size ≥ 128`.
6. **No trained-parent held-out eval** — driver only probes the **cold base once**
   (`run_wordle_zorl_xorl_ps.py` §2 comment: "trained parent eval … a follow-up"). We
   currently watch only the *candidate mean on train* — a training proxy, not the honest
   held-out curve. → add a periodic parent held-out eval (retries=0, NG≥128) by serving a
   zero-perturbation parent adapter each N steps.

## 5. Eval protocol parity (the one true gate)

GRPO honest gate: seed-777 held-out, **NG≥128**, **retries=0**, `EVAL_INVALID_RETRIES=0`;
base+think=0.00; honest ceiling **~0.67–0.68** (peak in-training ~0.78 @ ~s116). Our
earlier "0.5625 target / 0.469 en-route" were **retry-crutch-era numbers** — superseded.
ZORL must be measured on the identical held-out words at retries=0, NG≥128, to be
comparable. The code supports this (gap #5); the run must enable it.

## 6. Recommended change set (apply together, then relaunch)

Training (what we optimize): `NUM_PAIRS=32+`, `ROLLOUTS_PER_PUZZLE=8`, `TRAIN_SIZE=32`,
`TRAIN_POOL_SIZE=4096`, `INVALID_RETRIES=0`.
Eval (honest measurement): `WORDLE_TRAIN_EXCLUDE_SEED=777`, `WORDLE_TRAIN_EXCLUDE_COUNT=170`,
`EVAL_SIZE=128`, retries=0; add periodic trained-parent held-out eval.
Decide the step-time/population tradeoff explicitly (§3) and, if needed, widen the sampler
pool so a large population is affordable.

**Bottom line:** the *scientific* setup (model/prompt/reward/task/eval-word-set) is faithful
or faithfully-supportable; the *current knob values* are not. Until §4 is closed, treat all
live ZORL Wordle numbers as not-comparable to GRPO's 0.67–0.68.

## 7. As-built configuration + design-choice justification (2026-07-01, applied)

This is the deployed config and *why* each choice was made (some deviate from §6's first-cut
recommendation). **Gaps #1–#5 are closed; gap #6 (a true trained-parent held-out eval) is
NOT — it is approximated by a biased proxy. Feasibility (§4 was never a "gap" but a risk) is
NOT solved.** The honest caveats and unresolved risks are catalogued in §8 — read it before
trusting any ZORL-vs-GRPO number this config produces.

| Axis | GRPO | ZORL as-built | Justification |
|---|---|---|---|
| Model / prompt / reward / task | Qwen3.6-35B-A3B / constraints_think / `wordle_retrieval_reward` | **identical** | faithful (§1) |
| Objective honesty | retries=0 | **`--invalid-retries 0`** (train + eval) | optimize the honest objective, not the retry crutch |
| Held-out set | seed-777, 170 reserved, NG≥128 | **`WORDLE_TRAIN_EXCLUDE_SEED=777 COUNT=170`, EVAL_SIZE=128** | identical held-out words (WORD_LIST byte-identical); train pool = 4266−170 = **4096** (= GRPO's `train-pool-size`) |
| Held-out metric | **solve rate** (`exact_match`) | solve rate of the **candidate mean** (per-puzzle `exact_match`) — a BIASED proxy, not the parent (§8.1) | composite reward is not comparable to GRPO solve-rate; but candidate-mean ≠ parent (O(σ²) bias, unknown sign) — a true parent eval is still owed |
| train_size (puzzles/step) | 32 | **32** | GRPO parity |
| **Population** | (n/a — single policy) | **requested 64 pairs (128 candidates)** — delivered count UNVERIFIED (§8.4) | ES SNR scales with population — the *dominant* ES lever; the mult run that climbs used 128–256 candidates |
| **rollouts / candidate / puzzle** | group-size 16 (one policy) | **2** (→ train32×2 = 64 samples/candidate) — RISKY, unvalidated for Wordle (§8.5) | §6 said 8; cut to 2 by *analogy* to the single-turn mult run (rollouts=1). Multi-turn Wordle is higher-variance; needs a per-pair SNR check before trusting |
| Serving precision | bf16 | **bf16** | FP8 base + bf16 LoRA **attempted and blocked** — LoRA loads on FP8 but generate crashes at `lora/layers.py:1264 _base_gate_up_gemm`: Triton `Unsupported rhs dtype fp8e4nv` (the MoE-LoRA fp8 GEMM). Needs a Triton upgrade or the DeepGEMM fp8 path — a separate infra task, not a config switch |
| Concurrency / pool | N×TP2 samplers behind SMG | **32 TP2 scorers, `--score-max-workers 1024`** | measured GPU util was ~30–40% at 16 workers/scorer (multi-turn rollouts stall at turn boundaries); 32 rollouts/scorer fills the batch → fuller GPU use. 32 scorers repurposed from the (passed) mult sanity |
| Optimizer | unclipped-IS policy gradient, full-weight | Muon ES fold, LoRA rank-16, b_only, muon_lr 1e-2, sigma 0.05 | the one intended difference (§2); LR not comparable across parameterizations |

**Total rollouts/step** = 2·64 · 32 · 2 = **8192** (vs GRPO's 512) — the structural ~16× ES
overhead of population scoring (§3), which is why the sampler pool is 2× wide and heavily
concurrent. This config makes the *metric* honest (solve-rate, retries=0, GRPO held-out) and
raises GPU utilization; it does **not** make the comparison feasible on its own (§8.3) and
does not guarantee — or even make likely — the ES parent-climb toward 0.67–0.68.

**Deferred lever:** FP8 base + bf16 LoRA (~1.5× + KV headroom for even higher concurrency)
once the Triton/DeepGEMM fp8-GEMM path is available on the scorer build.

## 8. Honest caveats & unresolved risks (peer review, 2026-07-01)

Recorded verbatim so the doc doesn't oversell itself. Ranked by how much each could
undermine an eventual ZORL-vs-GRPO claim.

**8.1 — The held-out number is a biased proxy, not the parent.** §7 evaluates the
candidate MEAN of `exact_match`, never the actual parent. For antithetic pairs the mean
cancels the O(σ) term but keeps an **O(σ²) curvature bias of unknown magnitude AND sign** —
and σ=0.05 B-only perturbations are deliberately sized large enough to move reward, so the
bias is not negligible for a near-discrete solve-rate. "σ²-accurate" (earlier wording) was
wrong. **To make the "vs 0.67–0.68" claim, we must serve a zero-perturbation parent-B
adapter and eval it at least once.** Not yet built.

**8.2 — The experiment does NOT isolate "the optimizer" (deepest issue).** The headline
"differ only in the optimization algorithm" is not true: ZORL co-varies **rank-16 B-only
LoRA vs full-weight** (a severe capacity restriction) and **EP4 vs EP8**. If ZORL
underperforms, the gap is confounded — "ES vs unclipped-IS PG" cannot be separated from
"tiny-LoRA vs full-weight." §2 discloses LoRA, but the top-line framing undersells it. **The
missing control is GRPO trained on the SAME rank-16 B-only LoRA** — without it, no clean
attribution to the optimizer is possible.

**8.3 — Feasibility is not established.** ~56 min/step (prior comparable run: load ~410s +
score ~2950s) ⇒ ~**12 steps** in the 11h cap; GRPO reached 0.67 around **step 116**. Worse,
the driver **cold-starts (B=0) every job** (`load_checkpoint_path: ""`, fresh `create_model`,
no parent checkpoint/resume), so runs **cannot be chained** to accumulate steps. An ES climb
to a GRPO-comparable solve rate in ~12 updates (48–64 pairs, σ=0.05, rank-16) is a very large
ask. Parent-B checkpoint save + resume is a prerequisite for any serious attempt.

**8.4 — As-built population is unverified.** §7 asserts 64 pairs / 128 candidates; the live
step-1 count must be read back (`cands=` / `used_pairs=`) to confirm the PS/scorer pool
actually delivers it (per-scorer `max-loras-per-batch`=8, `max-loaded-loras`=48 vs 4
candidates/scorer at 32 scorers — should fit, but confirm, don't assert). The earlier
`cands=96 used_pairs=48` run was `NUM_PAIRS=48` by config (not a capped 64).

**8.5 — rollouts_per_puzzle=2 is a cross-task extrapolation.** Cut 8→2 by analogy to the
*single-turn* mult run; multi-turn Wordle is higher-variance. At 64 samples/candidate,
solve-rate std ≈ 0.05–0.06; a σ=0.05 B-only antithetic pair-delta may sit **below** that
noise floor, in which case adding pairs just averages noise. This is the single riskiest
knob and needs a **per-pair SNR check** (pair-delta magnitude vs per-candidate std) before
being trusted.

**8.6 — Cold-probe hygiene.** The base probe uses `lora_path=None`, which sglang
`--enable-lora` rejects (HTTP 400 per example) → the reported `cold=0.00` is *errors counted
as 0*, not a genuine base generation. GRPO science says the true base is 0.00, so the
conclusion is unaffected, but the ✅ in §1 overstates what the probe measures.
