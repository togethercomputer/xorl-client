# LoRA serving bug: magnitude-proportional k3 growth (root-caused + fixed, 2026-07-02)

**TL;DR.** During LoRA-16 GRPO on Qwen3.6-35B-A3B, k3 grew proportionally to LoRA magnitude
because the SGLang sampler was silently *not applying* two of the four trained tensor groups —
and actively *corrupting* a third. Three distinct defects, all confirmed empirically with a
filtered-adapter A/B on a single sampler. Fixed via two SGLang commits + a MoE-experts-only
recipe config. The original `GRPO-WQ36-LORA16-IS-control` baseline is **confounded** (§5).

## 1. Symptom

k3 (`loss/is_kl_k3_debug_abs_logratio_mean`) starts at the floor (~2.8e-4 step 1, zero-init
LoRA-B ⇒ sampler≡trainer) and climbs monotonically with the applied delta: run `wd2qf`
(2026-07-02) went 0.0064 → 0.018 over 15 steps. The sampler was not sampling from the trained
policy, which corrupts GRPO itself (behavior ≠ policy).

## 2. What the trainer exports vs what the sampler applied

Adapter `policy-000015` of run `wd2qf` (default target set `[q,k,v,o,gate,up,down]_proj`),
61,840 tensors:

| group | tensors | sampler behavior (before fixes) |
|---|---|---|
| `mlp.experts.<E>.{gate,up}_proj` (per-expert, 40 layers) | 40,960 | applied ✓ |
| `mlp.experts.<E>.down_proj` (per-expert, 40 layers) | 20,480 | **WIPED** by bug 2 ✗ |
| `mlp.shared_expert.down_proj` (40 layers) | 80 | **misapplied to expert-0 of every layer** ✗ |
| `linear_attn.{q,k,v,o}_proj` (30 GDN layers) | 240 | dropped (warning in sampler log) ✗ |
| `self_attn.{q,k,v,o}_proj` (10 full-attn layers) | 80 | applied ✓ (stacked qkv mapping) |

Mechanisms (all in `xorl-sglang-internal`):

- **Bug 1 — GDN drop.** SGLang's GDN block (`models/qwen3_5.py`, `Qwen3_5GatedDeltaNet`) has
  only fused `in_proj_qkvz`/`in_proj_ba`/`out_proj`. The trainer
  (`xorl/ops/linear_attention/layers/gated_deltanet.py:99-149`) has separate
  `q/k/v/o_proj`. LoRA wrapping matches leaf names ⇒ the 240 GDN tensors never attach.
  The drop *is* logged ("contains weights for modules that are not LoRA-wrapped … skipped") —
  it scrolled by unread in sampler pod logs during every run.
- **Bug 2 — shared_expert misroute → per-expert down_proj wipe.** In
  `lora/mem_pool.py::load_lora_weight_to_buffer`, `mlp.shared_expert.down_proj` matches
  neither `experts\.(\d+)\.` nor `experts\.shared\.`, so it fell into the "legacy shared MoE"
  branch and was appended to the `down_proj_moe` dict as `shared_list`. The load branch gave
  `shared_list` priority ⇒ **all 256 per-expert down_proj A/B tensors discarded on every
  layer**, expert-0 slot overwritten with the shared-expert tensor. Silent because
  `shared_expert_intermediate_size == moe_intermediate_size == 512` (shapes fit).
- **Bug 3 — shared expert unwrapped.** `Qwen3_5MoeForConditionalGeneration._lora_pattern`
  deliberately excluded `mlp.shared_expert.*` ("no current adapter targets it" — false: the
  engine's indirect target matching adapts `shared_expert.down_proj`, and only that, because
  the trainer's shared expert has fused `gate_up_proj`).

## 3. Discriminating experiment (evidence)

Filtered copies of `policy-000015` served on one TP2 sampler
(`/shared/apanda/wordle-sft-runs/lora_debug/`): v0 = full, v1 = −`linear_attn.*`,
v2 = v1 −`shared_expert.*`, v3 = v2 −`self_attn.*`. Scored 16 on-policy texts (prefill
logprobs), pairwise mean |Δ logprob| per token:

| pair | pre-fix | post-fix (both commits) |
|---|---|---|
| v0 vs v1 (GDN effect) | **0 (bitwise)** | 0 (bitwise) — GDN still dropped by design; recipe excludes it |
| v1 vs v2 (shared_expert effect) | **0.036** (= the wipe) | 0.036 (= shared correctly applied; per-expert intact) |
| v2 vs v3 (self_attn effect) | 0.035 | 0.035 (unchanged, regression check ✓) |
| base vs v0 (whole adapter) | 0.033 | 0.035 |

Pre-fix, removing the shared_expert tensors changed logprobs as much as the entire adapter —
that was the per-expert wipe. Post-fix the load log shows no unwrapped/mixed warnings for MoE
weights; only the expected GDN skip on layers `[0,1,2,4,…,38]`.

## 4. Fixes

1. **SGLang `f26350543`** (mem_pool): weights addressing a LoRA-wrapped dense module (matched
   by layer-local path) route to that module's dense buffer, never the `*_moe` shared path.
   Defense-in-depth: mixed per-expert+shared dicts now load per-expert and warn. CPU tests:
   `test/registered/lora/test_hybrid_shared_moe_weight_loading.py` (also repairs 2
   pre-existing red tp=1 cases whose harness predated the unwrapped-drop pass).
2. **SGLang `c45f64fff`** (qwen3_5): LoRA-wrap `mlp.shared_expert.{gate_up_proj,down_proj}`;
   dense mlp buffers sized with `shared_expert_intermediate_size`. The exported
   `shared_expert.down_proj` now applies exactly as trained.
3. **Recipe** (`configs/wordle/grpo-ep8x1node-muon-lora16-moeonly.yaml`, xorl-infra
   `1143917`): `lora_target_modules: [gate_proj, up_proj, down_proj]` — per-expert MoE +
   `shared_expert.down_proj`, i.e. exactly the set the sampler now applies exactly. Attention
   targets removed on purpose: leaf-name matching cannot include `self_attn.*` without also
   training the un-servable `linear_attn.*` (trainer-side matcher is leaf-name only).

**Tier-2 follow-up (attention capacity) — VALIDATED 2026-07-02 (background agent):** the
export-side repack works with **zero SGLang changes**. `in_proj_qkvz` is a plain
`[q|k|v|z]` row-concat (qwen3_5.py:380-398,1907-1910 — no head interleaving), each slice
TP-sharded independently. Repack = A_fused rowstack(A_q,A_k,A_v) [3r×2048], B_fused
[12288×3r] block-diagonal (z rows zero), `o_proj`→`out_proj`. Validated on a dedicated
sampler: adapters attach (no skip warnings), zero-B reproduces base bitwise, ×64-amplified
deltas agree with a merged-weights oracle to bf16-rounding level (the un-amplified trained
GDN deltas are *below the base weights' bf16 half-ulp*, so naive folding destroys them —
LoRA serving is more faithful than folding). Constraint: SGLang has no per-module ranks in
one adapter — zero-pad all modules to uniform r=3r (α scaled to keep α/r), bitwise-validated;
inflates per-expert MoE buffers at high rank, so per-module rank support is the long-term
fix. Tools (client hub `27cab4a`): `repack_gdn_lora.py`, `fold_gdn_lora_into_ckpt.py`,
`score_gdn_lora.py`; sampler yaml `gdnlora-sampler.yaml` (xorl-infra `aec14c5`). Recipe to
re-enable attention: post-export repack + sampler `--max-lora-rank ≥3r` + `in_proj_qkvz
out_proj` in `--lora-target-modules`; final validation = live k3 gate with a GDN-inclusive
adapter.

## 5. The original k3lora baseline is confounded

Every prior LoRA-16 run (`GRPO-WQ36-LORA16-IS-control`, runs kb2d6/fnfwm/k49sw/29rtm/wd2qf/
hw79r; `server_output_k3lora*`) used the default 7-name target set ⇒ trained GDN attention +
shared_expert the sampler never (correctly) applied, and lost per-expert down_proj serving
entirely. Its "low step-1 k3" was the zero-B illusion. Rollouts sampled ≠ policy trained, so
its training curves and any conclusions drawn from them (incl. the river-port comparison
anchor `GRPO-WQ36-LORA16-IS-river` vs "control") should be re-anchored on the fixed MoE-only
run. k3 numbers from those runs measure the serving bug, not sampler-trainer kernel parity.

## 6. Validation run — GATE PASSED (25 steps, k3 at floor)

`GRPO-WQ36-LORA16-IS-moeonly` (wandb, project xorl-wordle; run dir
`20260702T103508Z-…-k3loramoe-…`): builder `build_k3lora_moe.py`, engine = fresh
`origin/apanda-dev` worktree, client = `~/xorl-client` hub, samplers = `wordle-k3lora-smp`
×4 on sglang `c45f64fff` + `k3lora-smg`. Step-0 export verified: exactly the servable set
(61,440 per-expert + 80 shared_expert.down_proj tensors, zero attention), clean sampler
load (no unwrapped/mixed warnings).

**k3 (`loss/is_kl_sample_train_k3:mean`), fixed vs old bugged run:**

| step | fixed | old (bugged) |
|---|---|---|
| 1 | 3.0e-4 | 2.8e-4 (zero-B illusion) |
| 5 | 3.3e-4 | 3.5e-4 |
| 9 | 3.5e-4 | 6.4e-4 |
| 12 | 3.1e-4 | 1.18e-3 |
| 15 | 3.4e-4 | 2.02e-3 |
| 20 | 3.5e-4 | — |
| **25** | **3.6e-4** | — |

Whole run inside 2.7–3.9e-4 (the known live floor band) — never above 4e-4 vs the ≤1e-3
gate. **Magnitude-matched**: the fixed run's MoE B²-norm at s10 (7.6) is 2.3× the old run's
at s12 (3.4, where its k3 was already 1.18e-3); by s15 the fixed adapter held the floor at
9.1. The magnitude-proportional divergence is gone. Reward moved off the floor
(5/512 → 41/512 peak at s23, reward −0.222 → −0.105 best) but climbs far slower than
full-weight (54/512 at s9, 215/512 at s18 on the k3pnr3 recipe) — muon 5e-5 was inherited
from the k3-diagnostic control config and is the full-weight lr; muon's orthogonalized
update is also a poor fit for rank-16 LoRA factor shapes.

## 7. AdamW recipe + GDN serving at magnitude (three-run discriminator, 2026-07-02 evening)

Muon 5e-5 (inherited from the diagnostic control config) learns far too slowly for LoRA —
muon's orthogonalized update is built for square full-rank matrices, not rank-16 factors.
Switched to the LoRA-16 recipe lr: **adamw 5e-4, wd 0, cosine (warmup 8)**. Sampler side
for GDN serving: `--max-lora-rank 48` + `in_proj_qkvz out_proj` targets, mem-fraction 0.85,
**1 LoRA slot** (the rank-48 per-expert pool is ~3× larger; 0.70/2-slot OOMs KV sizing).
Server config must be adamw too — server-level muon kwargs merge into the session optimizer
and crash `AdamW.__init__`.

**`GRPO-WQ36-LORA16-ADAMW-gdnfull`** (full 7-name targets + per-step `--gdn-repack`,
~20s/step overhead): learning exploded — 63/512 at s3, **104/512 at s4, 163/512 (reward
+0.24) at s5** — full-weight took 17-18 steps to reach these levels. But k3 *compounded*:
2.5→4.2→9.1→22.6→80→129 (×1e-4) over s1-6, ratio_max →9.8, and s6 reward halved (76/512).
Stopped at s6 to discriminate serving-bug vs lr-effect.

**Discriminator `GRPO-WQ36-LORA16-ADAMW-moeonly`** (same adamw 5e-4, but ONLY the
serving-proven MoE targets): k3 tracked the gdnfull curve almost exactly
(2.7/5.2/12.3/26.8/47.9 ×1e-4 over s1-5) with the same reward trajectory (161/512@s5,
dip@s6, recovery@s7). **Verdict: the k3 elevation is NOT attention-serving — the fused-GDN
repack path is vindicated** (its load is clean, step-0 k3 at floor, and the offline ×64
fold-oracle already bounded mapping error at bf16 level). The elevation is an honest
property of aggressive adamw updates: they concentrate on high-gradient directions and
sharpen the policy fast (that's why it learns 5× faster), which amplifies the same
serving/kernel noise floor into larger logprob divergence — magnitude is not the driver
(the muon run reached 7× more B²-norm at floor k3). Both runs were also still inside lr
warmup while k3 accelerated. Calibration: full-weight GRPO trained to ~0.8 in-training at a
constant k3≈5.5e-2 — 4-25× above these levels.

**5e-4 outcome (`vhsq7`, stopped at s23): fast climb, then genuine divergence.** Peaked
234/512 in-training at s14 (full-weight needed ~18 steps for less), but k3 kept compounding
(1e-2 @s6 → **0.196 @s23**, ratio_max →120) and reward swung 234→27→105. At 5e-4 the
policy moves so fast that training goes seriously off-policy — the same "5e-4 too hot"
failure the river port hit, just slower to bite on xorl. Stopped; per-step exports kept.

**Honest held-out eval of the salvaged 5e-4 policies** (NG=128, seed-777 slices, retries=0,
temp 0.2, `eval_wordle_sglang.py --lora-path`, adapters re-served via the fused repack —
`evals/lora-gdnfull-vhsq7/`):

| policy (export) | held-out exact | trained-slice exact |
|---|---|---|
| policy-000013 (s14 peak source) | 0.297 | 0.313 |
| policy-000015 | **0.313** | 0.375 |

Base+think honest floor = **0.00** ⇒ +31pp held-out from 14-15 LoRA steps; held-out ≈
trained ⇒ generalization, not memorization. **LoRA GRPO on this stack is train/serve-correct
and learns.** Definition-of-done: the ≤1e-3/≥25-step serve-correctness gate = §6 muon run;
training gains = this eval.

**Go-forward: `GRPO-WQ36-LORA16-ADAMW2E4-gdnfull`** (adamw 2e-4, run `pzjbt`, own pool
`wordle-k3lora2-smp` at mem-fraction 0.80 — 0.85 left too little runtime headroom over the
rank-48 pool on one node and OOM-looped) — expect the fast climb with bounded off-policy
drift. Lessons: LoRA lr ladder on this task is muon 5e-5 ≪ (too slow), adamw 5e-4 ≫ (diverges
by ~s20), adamw 2e-4 = target.

## 8. AdamW lr ladder (2026-07-03) — honest held-out evals (NG=128, retries=0, seed-777)

| run | in-training peak | k3 trajectory | held-out exact | trained exact |
|---|---|---|---|---|
| muon 5e-5 (moeonly) | 41/512 @s23 | floor 3e-4 flat (25-step gate) | — | — |
| adamw 5e-4 (gdnfull, `vhsq7`) | 234/512 @s14 | 1e-2@s6 → **0.196@s23** (diverged, stopped) | 0.297–0.313 | 0.313–0.375 |
| **adamw 2e-4 (gdnfull, `28swv`)** | **256/512 @s14** | 2.6e-3@s7 → 4.2e-2@s14, **plateaued/declining** 2.9e-2@s17 | **0.500** (policy-000013) | 0.477 |
| adamw 5e-5 (gdnfull, `bwmpr`) | running | — | — | — |

The 2e-4 peak policy (14 LoRA steps) matches the full-weight k3pnr3 in-training peak
(0.5 @s38) on the honest held-out gate in a third of the steps, with format rate 0.99 and
held-out ≥ trained. Stopped at s18 (user call: still oscillating 121-256 post-peak; the
5e-5 run tests the steadier end of the ladder). All per-step exports of every run are kept
under the respective `server_output_k3lora_gdn*/sampler_weights/` for salvage evals via
`repack_gdn_lora.py` + `--lora-path`.

**Provenance note (applies to ALL these runs and the k3-comparison six runs):** `--reward-key
wordle_retrieval_reward` silently falls back to the SHAPED reward (no `wr_*` keys logged);
the optimization target was the shaped reward throughout. Solve counts (`exact=N/512`) and
the held-out evals above are unaffected (real solves).
