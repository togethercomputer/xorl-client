# xorl vs SkyRL: Pareto-dominance evidence (marin #6279 repro)

**Audience note:** this doc is the source-of-truth data pack for an agent building a website /
slide for a technical talk. Every number below has provenance; do not round differently or drop
the footnotes marked ⚠. The story fits one "evidence" slide plus an optional backup slide.

---

## 1. Problem setup (say this fast)

Two codebases train **the same model on the same data with the same RL recipe**; we compare
outcomes end to end.

- **Task:** GRPO math RL (forced-thinking) on a dense Qwen3 ~9.7B SFT checkpoint
  (`laion/delphi-1e22-p33m67-...-wc386k_lr1e5-sft`), dataset RLVR-MATH-7500, boxed-answer
  verifier + length-penalty reward, no KL / no entropy reg, lr 1e-5, clip 0.5, bs 256 × G=16,
  ~145 steps, 4k token budget (512 prompt + 3584 gen), temp 0.7 / top_p 1.0.
- **Baseline system:** marin's published `rlvr7500_w1` run on **SkyRL** (colocated
  policy+ref+rollout, 16 nodes × 8×A100 = 128 GPUs). Public repo:
  `laion/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B` (training_logs = ground truth).
- **Our system:** the same recipe reproduced on **xorl** disaggregated server-mode RL
  (8×H100 trainer + 8×1-H100 SGLang samplers = 16 GPUs total).

**The API framing (core message):** the RL loop is a plain Python **client** written against a
**Tinker-compatible API** — `forward_backward(...)`, `optim_step(...)`,
`save_weights_for_sampler(...)`, `sample(...)` — and the backends are **just servers**: an xorl
training server and SGLang inference servers, each launched from a one-file k8s manifest. The
driver holds the science (rewards, advantages, data); the engines are swappable infrastructure
behind the API. Nothing about the RL algorithm lives inside the training engine.

## 2. The claim (headline table — this is the slide)

Same model, same data, same recipe. xorl pareto-dominates on every axis:

| axis | SkyRL (reference) | xorl (ours) | factor |
|---|---:|---:|---:|
| GPUs | 128× A100 | 16× H100 | **8× fewer** |
| wall-clock per step | 1,027 s | 198 s | **5.2× faster** |
| total train wall | ~41 h | 8.0 h | **5.1× faster** |
| GPU-hours | ~5,290 | ~128 | **~41× less** ⚠a |
| sampler↔trainer logprob gap (mean \|log r\|) | 1.3e-3 | ~1.7e-4 | **~8× tighter** ⚠b |
| …same gap as KL (K3 estimator) | ~8e-7 (implied) | 1.4e-8 | **~60× tighter** ⚠b |
| PPO clipping active? | yes (clip ratio ~3.0e-4 every step) | **never** (high/low clip fraction = 0.0 for all 146 steps) | — |
| training reward, last-10 mean | +0.247 | **+0.436** | ahead on 132/145 steps ⚠c |
| MATH500 lift (same-harness ΔSFT) | +8.4 | **+12.3** | ~1.5× |
| gsm8k-flex lift | +4.4 | **+8.8** | ~2× |
| AIME24 lift | **−3.4** (regressed) | ≈ +1 ⚠d | sign flip |

One-line causal chain for the talk: **better numerics → truly on-policy training (ratio ≈ 1, no
clipping, no off-policy correction needed) → cleaner gradients → higher reward → bigger eval
lifts — and it's simultaneously 5× faster on 8× fewer GPUs.**

## 3. Per-axis evidence

### 3.1 Speed

- Ours: mean **197.6 s/step**, median 164 s (W&B row timestamps of `nw155nmj`); total
  **8.01 h** for 146 steps on 16 H100s. Phase split: rollout ~147 s (75%), train ~45 s,
  weight-sync **3.5 s**, checkpoint ~13 s every 10th step. Rollout throughput ~17.8k tok/s.
- SkyRL: mean **1,026.6 s/step** (their `timing/step`); ~41 h for 145 steps on 128 A100s.
  Phase split is *inverted*: `policy_train` 783 s (76%), a separate 129 s
  `fwd_logprobs_values_reward` pass (recomputing rollout logprobs — xorl's drgrpo loss computes
  behavior logprobs inside the same forward-backward, so this pass doesn't exist for us),
  generate 87 s, `sync_weights` 21.8 s (vs our 3.5 s KV-cache-preserving P2P sync).

### 3.2 Numerics (the differentiator)

The quantity both frameworks log: the gap between the logprobs the **sampler** assigned during
generation and the logprobs the **trainer** recomputes for the same tokens. This gap is exactly
the off-policy error that PPO-style clipping exists to paper over.

- SkyRL: `policy/log_ratio_abs_mean` = **1.30e-3** (mean over all 211 logged steps);
  `ppo_clip_ratio` ≈ 3.0e-4 — clipping fires every step.
- xorl: behavior K3 mean **1.51e-8** / median 1.40e-8 / worst step 8.1e-8 (all 146 steps);
  implied \|log r\| ≈ 1.7e-4; `ratio_mean` = 0.99996; clip high/low fractions **identically 0.0**.
- How: train/serve bit-alignment — batch-invariant matmul/reduction kernels shared with the
  sampler (`XORL_BATCH_INVARIANT_MATMUL=1` + `SGLANG_BATCH_INVARIANT_OPS`), SGLang-ordered
  residual RMSNorm in the trainer (`rmsnorm_mode: sglang`), fp32 lm-head on both sides, fp32
  sampler log_softmax, eager RoPE, ieee tril. All upstreamed (xorl PRs #431/#433, sglang #52).
- Bonus result the numerics enabled us to *see*: with pipelining (policy lag 1) the identical
  recipe **collapsed at step ~21** (K3 spiked 1e-6 → 8e-2, reward → −0.91). K3 was the leading
  indicator. Fully on-policy (lag 0) sailed through. SkyRL's run also needed a stabilization fix
  (entropy-bonus explosion) before its published result.

### 3.3 Training reward

Correctly aligned per-step comparison (⚠c): ours ahead on **132/145** steps (all 13 exceptions
in the first ~20 noisy steps), mean delta **+0.148**, last-10 mean **0.436 vs 0.247**. Our
late-run rewards (~0.43–0.52) exceed even SkyRL's single best step (0.345 @ its step 142).

### 3.4 Final evals (same-harness ΔSFT; ⚠d)

| metric | SFT base (ours) | trained (ours) | our Δ | SkyRL Δ |
|---|---:|---:|---:|---:|
| MATH500 | 48.69 | **60.97** | **+12.28** | +8.4 (45.0→53.4) |
| gsm8k-flex | 65.28 | **74.07** | **+8.79** | +4.4 (64.1→68.5) |
| AIME24 (10 seeds × 300) | ~2.6–3.1 ⚠d | 3.77 ± 0.47 | **≈ +1** | −3.4 (4.9→1.50) |

The AIME sign flip has a mechanism worth one sentence: SkyRL's run lost 3.4 points to CoT
truncation at the 4k ceiling; our policy's completions stayed ~450 tokens (concise-correct via
the same length penalty), avoiding the cliff entirely.

## 4. The API slide (one diagram)

```
   RL driver (plain Python, ~1 file)               Tinker-compatible API
   rewards · advantages · data · eval    ──────►   forward_backward / optim_step /
                                                   save_weights_for_sampler / sample
                                                          │
              ┌───────────────────────────────────────────┴──────────────┐
              ▼                                                          ▼
   xorl training server (1 k8s manifest)                8× SGLang samplers + router (2 manifests)
   8×H100, FSDP2, drgrpo loss                           1×H100 each; NCCL/P2P weight sync,
                                                        KV cache preserved across syncs (3.5 s)
```

Talking points: the entire stack is `kubectl apply` of 3 manifests (or one rebuild script);
the driver never links against the engine; swapping the training backend = pointing the client
at a different server URL. The same driver code shape runs against any Tinker-compatible
backend.

## 5. Chart specs for the website (all data already in W&B, project `together-research/xorl-marin-rl-6279`)

1. **Reward-vs-step overlay** (the money chart): runs `nw155nmj` (ours) +
   `reference-rlvr7500_w1-full` / id `3comlb0c` (SkyRL, imported). Key `rollout/mean_reward`
   on both, x-axis `policy_step`. Optionally shade the first ~20 steps (noisy region).
2. **Per-step wall time overlay**: runs `marin6279-repro-nopipe-timing` / id `7odwm4bz` (ours,
   retro-derived) + `3comlb0c` (SkyRL). Key `timing/step`, x-axis `policy_step`. Log-scale y
   makes the 5× gap readable. Backup: stacked-bar phase split (ours: `timing/generate`,
   `timing/policy_train`, `timing/sync_weights`; theirs: same keys).
3. **Numerics bar**: two bars on log scale — 1.3e-3 vs 1.7e-4 (mean |log ratio|), captioned with
   "PPO clipping: fires every step vs never fires". (Data: SkyRL `policy/log_ratio_abs_mean` in
   `3comlb0c`; ours from `train/behavior_k3` in `nw155nmj`, converted via \|log r\| ≈ √(2·K3).)
4. **Eval delta table** — render §3.4 as-is.
5. Do NOT use run `totwpo79` (tagged `superseded`; truncated single-segment reference).

## 6. Honesty footnotes (⚠ keep these on or near the slide)

- **⚠a Hardware:** SkyRL ran A100s, we ran H100s. Discount per-GPU ~2–3×; the
  hardware-normalized compute advantage is still **~15–20×**, and the per-step wall (5.2×) and
  GPU-count (8×) comparisons are raw facts.
- **⚠b Units:** SkyRL logs mean \|log ratio\|; we log K3 (≈ E[(log r)²]/2). Both directions of
  conversion shown; use "~8× tighter logprob agreement / ~60× lower divergence" — do not quote
  "5 orders of magnitude" (that compares mismatched units).
- **⚠c Alignment:** the reference `metrics.csv` concatenates 3 SLURM restart segments with
  overlapping steps; all comparisons here dedup by `trainer/global_step` (keep-last) and align
  our 0-indexed step s to their step s+1. The imported W&B reference run already handles this.
- **⚠d Eval honesty:** deltas are same-harness (our grader/template/sampling for both our SFT
  base and our trained model; theirs from their published harness) — absolute numbers across
  harnesses are not comparable (our harness reads MATH500 base as 48.69 vs their 45.0). Our SFT
  AIME baseline is incomplete (283/3000 generations, 9/10 seeds → 2.6–3.1 range), so quote the
  AIME delta as "≈ +1, small/noisy" — the robust claim is "did not regress, unlike the
  reference". Trained-side AIME is complete (3,000 generations).
- Reference identification: the baseline is marin issue #6279's `rlvr7500_w1` leg (lpw=1.0) —
  the strongest RLVR-MATH leg of their sweep (their w0 leg scored lower: MATH500 +6.4, AIME 0.0).
  We compared against their best, not a strawman.

## 7. Provenance index

| artifact | where |
|---|---|
| Our training run (reward, K3, ratios) | W&B `together-research/xorl-marin-rl-6279/nw155nmj` |
| SkyRL reference (reward + timing, deduped import) | W&B `.../3comlb0c` (`reference-rlvr7500_w1-full`) |
| Our per-step timing (retro-derived) | W&B `.../7odwm4bz` (`marin6279-repro-nopipe-timing`) |
| Raw driver metrics | `/shared/xorl-marin-rl-6279/runs/stack/20260701T193444Z-marin6279-repro-stack/metrics.jsonl` |
| SkyRL raw logs | `/shared/xorl-marin-rl-6279/checkpoints/delphi-...-rlvr7500_w1-think-140-10B/training_logs/{metrics.csv,report.md}` |
| Our evals (raw JSONL) | `/shared/xorl-marin-rl-6279/evals/k8s/{20260628T101054Z-baseline-eval,20260702T033930Z-trained-eval}` ⚠ the `sft_*` files inside the trained-eval dir are mislabeled duplicates — use the baseline dir for SFT |
| SkyRL evals | marin issue #6279 (penfever's `rlvr7500_w1` comment) + gist `penfever/19ea9149ae9cc02a21a9e2c9cec753c8` |
| Launch manifests (the "just servers" claim) | xorl-infra `k8s/marin/` (README has the full env profile + pinned SHAs) |
| Client driver + eval harness | xorl-client `experiments/marin/standalone/` |
| Full RCA + run ledger | `experiments/marin_rl_6279/{RUNBOOK,HANDOFF}.md` (branch `feature/marin-rl-6279`) |
