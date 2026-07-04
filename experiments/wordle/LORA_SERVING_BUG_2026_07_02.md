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
| adamw 5e-5 (gdnfull, `gh6mj`, throughput fixes applied) | 191/512 @s29, still climbing @s39 | 2.8e-4@s1 → 4.3e-3@s36, no destabilization | 0.297 (policy-000039) | 0.328 |

The 2e-4 peak policy (14 LoRA steps) matches the full-weight k3pnr3 in-training peak
(0.5 @s38) on the honest held-out gate in a third of the steps, with format rate 0.99 and
held-out ≥ trained. Stopped at s18 (user call: still oscillating 121-256 post-peak). The
5e-5 run (`gh6mj`) tested the steadier end of the ladder: it never destabilized (k3 flat,
in-training reward climbing to 191/512 by s29 with no sign of a plateau by s39) but, being
the lowest lr, it also converges slower per step — 0.297 held-out at s39 is *below* the
2e-4 run's 0.500 at s14, consistent with "not done climbing yet" rather than a worse
ceiling. `gh6mj` was stopped by an unrelated `/shared` disk-full crash at step 40 (see §9),
not a training failure or divergence. All per-step exports of every run are kept
under the respective `server_output_k3lora_gdn*/sampler_weights/` for salvage evals via
`repack_gdn_lora.py` + `--lora-path` (except the five superseded runs deleted 2026-07-04 to
relieve a cluster-wide disk-full incident — see §9).

**Provenance note (applies to ALL these runs and the k3-comparison six runs):** `--reward-key
wordle_retrieval_reward` silently falls back to the SHAPED reward (no `wr_*` keys logged);
the optimization target was the shaped reward throughout. Solve counts (`exact=N/512`) and
the held-out evals above are unaffected (real solves).

## 9. Throughput: topology bug fix (real, ~2×) + sampler-fleet fix (real, ~30%); recompute
## method changed nothing on real data (synthetic benchmark misled)

**Bug found via `nvidia-smi` on the live `bwmpr` (5e-5) trainer pod**: GPUs 0-3 at 95-96%
util, GPUs 4-7 at 0%. `grpo-ep8x1node-lora16-gdnfull.yaml` had `expert_parallel_size: 4` /
`data_parallel_shard_size: 4` — a stale comment-driven leftover from an earlier 4-GPU-node
version of this recipe ("EP4 ... -> 4-GPU node") that nobody updated when it moved to the
current 8-GPU pod. World size = 4, not 8; every LoRA run today (`vhsq7`, `28swv`, `bwmpr`)
ran on half the paid-for trainer. **Fixed to `expert_parallel_size: 8` /
`data_parallel_shard_size: 8`** in both config copies (not touching the then-live pod).

**Throughput-tuner sweep** (`xorl-throughput-tuner` skill; two k8s smoke jobs, dummy data,
8 steps, corrected EP8/dp_shard8 topology, LoRA-16 GDN-full targets) on two free nodes:

| candidate | warm MFU (synthetic dummy data) |
|---|---|
| `recompute_full_layer`, micro_batch_size=4 (matches the CLI harness's own batching) | **~4.8%** (steps 5-8: 4.83/4.82/4.81/4.80%) |
| `recompute_before_dispatch`, mbs=4 | OOM (matches the doc's prior finding) |
| `recompute_before_dispatch`, mbs=1 | **~6.5%** (steps 2,3,4,7,8: 6.46-6.62%) — ~35% faster |

`micro_batch_size` defaults to **1** in `Arguments` and the wordle recipe never overrides
it, so production's real per-forward_backward shape already matches the mbs=1 benchmark,
not mbs=4 — de-risking the switch to `recompute_before_dispatch` in production.

**Live validation (relaunch `gh6mj`, replacing `bwmpr`, same 5e-5 lr):** step 1 completed
with no OOM on real ragged Wordle data — confirms `recompute_before_dispatch` is safe in
production. But the fb speedup **did not transfer**: warm tok/s/GPU at step 2 was **437**
(`gh6mj`, recompute_before_dispatch) vs **436.5** (`bwmpr`, recompute_full_layer) —
statistically identical (~1% MFU either way). This matches a precedent already in
`THROUGHPUT_DEBUGGING_HANDOFF.md`: synthetic-data MFU on this model overstates real MFU by
roughly an order of magnitude (real per-rank shapes are tiny/ragged — short Wordle turns,
imbalanced across ranks — which the recompute method doesn't address). Keeping
`recompute_before_dispatch` since it's proven safe and equally fast, but it is NOT the lever
the synthetic sweep predicted.

**What actually sped up the live run (~30-33%, real and reproducible):** an 8-worker SMG
(`k3lora-combined-smg`) fronting both idle sampler pools (`wordle-k3lora-smp` +
`wordle-k3lora2-smp`), replacing the prior 4-direct-samplers-no-router setup. Diagnosed
cause: samplers ran at `#running-req: 1-4` out of a 256-request cap (active-drain — a batch
of 512 candidates drains to a handful of stragglers well before the per-turn barrier
releases everyone), so ~75% of step wall-clock was rollout with samplers mostly idle. Step
dt: `bwmpr` 1796-2041s/step → `gh6mj` 1200-1468s/step. `wordle-k3pnr3` (dead trainer,
`UnexpectedAdmissionError`, orphaned 8-sampler+SMG fleet) was also torn down, freeing 16 GPUs.

**`gh6mj` result: 39/40 steps completed cleanly, killed by an unrelated infra incident.**
Reward climbed to 191/512 by s29 and was still rising at s39 (170-184/512, no plateau, no
divergence) — the throughput fixes held for the full run. Step 40 itself hit
`OSError: [Errno 28] No space left on device` writing generation logs: `/shared` was at
**100% capacity cluster-wide** (473T total, 166G free) at the time, likely affecting other
users' jobs too. Root cause on our side: ~471GB of superseded, already-salvage-evaled LoRA
run exports (`server_output_k3lora{,_moe,_moe2,_gdn,_gdn2e4}`) accumulating under
`/shared/apanda/wordle-sft-runs/` — non-fused per-step adapter exports (~2.5GB × 40 steps ×
5 finished runs) were never pruned (only the `-fused` repack dirs get auto-pruned per
`export_and_load_sampler`'s cleanup). Deleted with user approval 2026-07-04; `/shared` back
to 36T free. `gh6mj`'s own step-000039 (fused, still loaded on all 8 samplers) was verified
intact before and after the cleanup and used for the final held-out eval (§8).
**Lesson for future long runs on this recipe: prune non-fused `policy-NNNNNN` export dirs
periodically, not just the fused ones — they aren't needed once the fused copy is loaded.**

## 10. xorl vs. river cross-check (2026-07-04) — the gap is run length, not a capability gap

A parallel river-client port of this exact recipe (`train_grpo_wordle_river.py`,
`RIVER_VS_XORL.md`, both in `xorl-client-wordle-science-20260614`) reached **57.1% held-out
at step 90** and **54.1% at step 128** (its winning config `GRPO-WQ36-LORA16-IS-river-shaped`,
`--lr 5e-5 --steps 128`, eval on all 170 reserved words at temperature=1.0). At first glance
this looks like xorl trails badly (0.30-0.50 vs 0.54-0.57). Checked for real causes rather
than assuming a capability gap:

- **Same held-out set — ruled out as a confound.** Both stacks import `tasks/wordle.py` from
  sibling checkouts of the same repo; verified directly: `WORD_LIST` size 4266
  (`wordle-python` source) and the first 5 of the seed-777/count-170 shuffle
  (`relax, yearn, weeny, years, fluke`) are **identical** on both. NG=128 (xorl, a random
  128-word subset of the 170) vs 170 (river, the full set) is a minor sampling-noise
  difference, not a systematic one.
- **Eval temperature differs** (river 1.0, xorl 0.2) but this should if anything favor xorl
  (lower temp → more confident guesses from a competent policy), so it doesn't explain xorl
  trailing.
- **Train temperature differs**: river used `--temperature 1.0 --logprob-temperature 1.0`
  throughout; xorl's recipe uses 0.7/0.7, inherited from an early SGLang-numerical-parity
  choice, never tuned for GRPO learning quality. Real, untested lever — lower rollout
  diversity could mean weaker per-step advantage signal — but see below, xorl doesn't look
  weaker per-step.
- **The actual answer: xorl was never run anywhere near river's step count.** River's
  winning run used 128 steps, peaking at 90. Every xorl LoRA-GDN run to date stopped
  early for an unrelated reason — 5e-4 diverged (~s16-20), 2e-4 was manually stopped at s18
  out of caution (still oscillating, not degrading), 5e-5 (`gh6mj`, the SAME lr as river's
  winner) hit an infra crash at s40 (§9) while still climbing. **None of them ran out of
  learning signal; all of them were cut off by something else first.** The 2e-4 run's
  0.500 held-out at just 14 steps — 1/6th of river's step-90 checkpoint — for only 7 points
  less solve rate is the strongest evidence xorl's per-step learning rate is comparable to
  or better than river's, not worse.

**Follow-up (not yet run): let one xorl config (5e-5 or 2e-4, GDN-full targets, current
throughput-fixed stack) run the full 90-128 steps without interruption**, ideally also
trying `--student-temperature 1.0 --logprob-temperature 1.0` to match river's recipe
exactly, before drawing any conclusion about a ceiling difference between the two stacks.
