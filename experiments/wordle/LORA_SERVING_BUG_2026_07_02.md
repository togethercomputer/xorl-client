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

**Tier-2 follow-up (attention capacity, not required to unblock):** SGLang already supports
GDN LoRA in the *fused* layout (`in_proj_qkvz`/`in_proj_ba`/`out_proj` are in the wrap
pattern). An export-side repack — fold A_q/A_k/A_v row-stacked into a rank-3r
`in_proj_qkvz` LoRA (B block-placed into q/k/v output slices, z rows zero, o→`out_proj`) —
would restore attention adaptation exactly, without SGLang changes.

## 5. The original k3lora baseline is confounded

Every prior LoRA-16 run (`GRPO-WQ36-LORA16-IS-control`, runs kb2d6/fnfwm/k49sw/29rtm/wd2qf/
hw79r; `server_output_k3lora*`) used the default 7-name target set ⇒ trained GDN attention +
shared_expert the sampler never (correctly) applied, and lost per-expert down_proj serving
entirely. Its "low step-1 k3" was the zero-B illusion. Rollouts sampled ≠ policy trained, so
its training curves and any conclusions drawn from them (incl. the river-port comparison
anchor `GRPO-WQ36-LORA16-IS-river` vs "control") should be re-anchored on the fixed MoE-only
run. k3 numbers from those runs measure the serving bug, not sampler-trainer kernel parity.

## 6. Validation run

`GRPO-WQ36-LORA16-IS-moeonly` (wandb, project xorl-wordle), 40 steps, launched 2026-07-02
~10:45Z: builder `k8s/wordle/builders/build_k3lora_moe.py`, engine = fresh
`origin/apanda-dev` worktree (`~/xorl-lora-fix`), client = `~/xorl-client` hub, samplers =
`wordle-k3lora-smp` ×4 (restarted on sglang apanda-dev `c45f64fff`) + `k3lora-smg`.
Gate: k3 stays ≤ ~1e-3 while `update_norm`/LoRA magnitude grows over ≥25 steps; then honest
held-out eval (retries=0, NG≥128).

**Result: (in progress — see below / metrics.jsonl of the run dir)**
