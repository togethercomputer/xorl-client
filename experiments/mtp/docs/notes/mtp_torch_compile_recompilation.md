# Qwen3.6 MTP/OPD — torch.compile recompilation status

_Last updated: 2026-06-06. Stack: `er-opd-q36-mtp-ss-0605c` — Qwen3.6-35B-A3B SingleShot MTP OPD, 4 nodes × 8 H100._

## HANDOFF — task for the next agent (start here)

> ### ⚠️ UPDATE 2026-06-06 20:40 UTC — recompile fix works, but COMPILED PROD RUN FAILS AT SCALE
>
> The recompile-stability fix is real (0 recompiles), but the **full prod-scale
> 4-node compiled run does NOT run**. Two relaunches with `enable_compile: true`
> + `flex_attention` (runs `20260606T194644Z` and `20260606T203005Z`, full prod
> config: 32 prompts/step, ep=8 alltoall) **both crashed on the first
> forward_backward** with `NCCL Error 1: unhandled cuda error` across multiple
> trainer ranks (1,2,4,5), thrown inside the **compiled MoE expert-parallel
> alltoall dispatch**:
> `experts.py::_ep_forward → alltoall.py::token_pre_all2all → comm.py::all_to_all
> → dist.all_to_all_single → NCCL unhandled cuda error`. The stuck collective
> then hung the step until the 1800s fb timeout → `rc=1`.
>
> **Why the smoke missed it:** the validating smoke (`20260606T193414Z`) was
> *reduced scale* — step 0→1 in ~24s and `forward_loss=1.2s`, i.e. tiny token
> volume. It exercised compile/recompile stability but not the prod-scale MoE
> alltoall traffic that triggers the CUDA error. The recompile grep was clean;
> the EP-dispatch path was effectively untested under load.
>
> **New task = make the compiled MoE EP-dispatch work at full scale (or exclude
> it from compile).** The `_AllToAll` custom autograd Function calling
> `dist.all_to_all_single` inside the compiled decoder/MoE graph is the suspect —
> compiling across the EP collective. Likely fix: `@torch.compiler.disable` the
> EP dispatch (`experts._ep_forward` / `alltoall.token_pre_all2all`) so the
> collective runs outside the graph (kernels/GEMMs still compile), or move the
> compile boundary above the dispatch. Also worth trying `ep_dispatch: deepep`
> (base yaml's value, overridden to alltoall by the persisted args) — a different
> dispatch path that may compile cleanly — and a single-rank `NCCL_DEBUG=INFO`
> repro to get the underlying CUDA error.
>
> **Live state:** the base yaml is still on the proven eager fallback
> (`attn_implementation: eager`, `enable_compile: false`). The production-scale
> GDN/MoE dispatch OOM caused by dense static replay plans is fixed and validated
> in that eager fallback (`20260606T225949Z`, one full 32-prompt step). Do not
> flip the base config back to `flex_attention`/`enable_compile: true` until a
> separate full-scale compile-on validation passes. The small 4-node compile
> smoke remains useful evidence, but it did not exercise the 32-prompt
> `static_padded_seq_len=768` replay capacity.

**Goal:** keep the Qwen3.6 SingleShot MTP OPD trainer compiled without per-step
`torch._dynamo recompile_limit` churn or eager fallback.

**State as of 2026-06-06 19:36 UTC:** compile stability is fixed in the 4-node
trainer smoke. Run `20260606T193413Z` completed two OPD forward/backward calls
with `enable_compile: true`, `pad_to_multiple=128`, and auto
`static_padded_seq_len=256`; the workers logged `torch.compile applied to 30
decoder layers`, the head exited `rc=0`, and a recompile-only grep over all four
trainer logs returned zero matches for `__recompiles`, `Recompiling function`,
`recompile_limit`, `attention_mask.linear_plan`, `kwargs['request_id']`, and
Flex `BlockMask` guards. Step 0 took the expected initial compile hit
(`forward_loss=17.283s`); step 1 reused the compiled graphs
(`forward_loss=1.224s`).

**What already works (don't redo):**
- Full-attention kernels via `attn_implementation: flex_attention` + a
  **tensor-parameterized static `BlockMask`** (`BlockMask.from_kv_blocks`,
  `src/xorl/mtp/singleshot.py` ~L362/L401): rollout variability lives in tensor
  data read by `mask_mod`; the block-table shape is static per padded bucket.
  Validated locally — `tests/mtp/test_singleshot.py::test_rollout_replay_static_block_mask_reuses_compiled_flex_graph`
  compiles once, no recompiles.
- Linear-attention/GDN decoder layers compile via
  `torch.compile(..., dynamic=True)` in
  `src/xorl/distributed/torch_parallelize.py::_compile_decoder_layers` (~L120).
  Full-attention decoder layers are skipped when they use FlexAttention because
  the Python `BlockMask` object at the decoder-layer boundary specializes per
  layer/mask object and can hit Dynamo's shared `recompile_limit`. This still
  keeps the actual FlexAttention kernel compiled.

**GDN replay-plan fix:** the previous surviving
churn was the **linear-attention / GatedDeltaNet** decoder layer. Evidence
(4-node run `...002029Z`, 2026-06-06): `recompile_limit (8)` hit at step 1,
frame `[0/8]`; `config.text_config.layer_types = [linear, linear, linear, full,
...]`, so layer 0 was a `linear_attention` (GatedDeltaNet) layer, not flex. The
first GDN replay-plan attempt still had a variable active-branch dimension; a
local reproducer hit `attention_mask.linear_plan.sequence_cols` size guards when
generated length changed. The current patch pads `sequence_*` and `output_*`
plan tensors to a fixed `static_padded_seq_len` capacity, with
`sequence_valid`/`output_valid` carrying the active branch contents. The compiled
GDN wrapper consumes those tensors directly instead of constructing Python
virtual branch lists inside the traced region.

**Decoder-boundary and metadata fixes:** the same live
logs also showed full-attention layers hitting `recompile_limit` at
`src/xorl/models/module_utils.py:2842` because the compiled decoder layer guarded
on `kwargs['attention_mask']` being a fresh `torch.nn.attention.flex_attention.BlockMask`.
Later FlexAttention frames guarded on raw key lengths (`expected 764, actual
528/536`), indicating that run was not on the final 128-bucketed path. Current
compile selection skips Flex full-attention decoder layers and compiles only the
linear-attention decoder layers; the full-attention kernel itself remains
compiled by the Flex backend. The final smoke exposed one more non-mask guard:
transport metadata `request_id` reached the compiled decoder/MoE wrappers and
specialized on the per-call UUID. `ModelRunner._model_inputs_for_loss` now strips
`request_id`, `batch_id`, `num_samples`, and `_shifted` from every model call
while keeping them in the micro-batch for packing/logging/response code.

**Ruled out (both fail identically — it is NOT a tensor-shape problem):**
`torch.compile(dynamic=True)` alone → `recompile_limit 8` at step 1;
`+ pad_to_multiple=128` (seq buckets) → same. Don't re-try shape-only levers.

**Concrete next steps (cheapest first):**
1. Resume the real trainer with the current patch set, not the older
   flex-only/GDN-only snapshots. Keep `pad_to_multiple=128` and the launcher
   auto-computed `static_padded_seq_len`; override only if the configured
   prompt/new-token budget exceeds the computed bound.
2. Grep the fresh long-run trainer logs for `__recompiles`,
   `Recompiling function`, `recompile_limit`, `attention_mask.linear_plan`,
   `kwargs['request_id']`, and `BlockMask`. The 2-step smoke is clean; the long
   run is still the authority for production stability.
3. If the long run stays clean, move on to throughput levers. The compile work
   removes one-time churn/eager fallback; it is not expected to materially change
   the backward-dominated step time.

**Honest expectation (set before optimizing):** the throughput motivation is
weak. Measured eager forward ≈ compiled forward (~48 s); a step is ~335 s and is
**backward-dominated** (`time_bwd` ~265 s, FSDP-comm-bound) — compile touches
only the forward. So `sup_tok/s/GPU ≈ 1.3` is the OPD workload's nature (small
supervised signal/step + inference-in-the-loop), **not** the compile failure, and
fixing compile will mostly remove the one-time recompile churn / be cleaner — it
will **not** materially raise throughput. If raising throughput is the real goal,
the levers are batch/packing, sampling overlap, and the backward FSDP comms — not
torch.compile. (MFU here is ~0.02%.)

**How to test + restart (recipe):** see "Status / results" and the generator
`experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`. Local:
`pytest tests/mtp/test_singleshot.py tests/distributed/test_torch_parallelize_policies.py`.
4-node relaunch (after a fix): bounce inference for fresh Mooncake state, then the
trainer — `stop-trainer-control --remove-run` → `write-student-inference-control`
→ wait all 4 endpoints healthy (`/get_model_info`=200, teacher `/health`=200) →
`write-trainer-control` (args persisted at `/tmp/mtp_relaunch_args.txt`: p2p +
alltoall, which the generator keeps; `attn_implementation`/`enable_compile` come
from the base yaml `examples/server/opd_singleshot_mtp_qwen36/qwen3_6_35b_a3b_student_mtp_4node_ep8.yaml`).
Then grep the run's `server.log` for `recompile_limit` (0 = fixed). Gotcha: the
run-dir RUN_ID is ~1 s off the control-revision timestamp (e.g. control
`…002028Z` → run dir `…002029Z`), and the `Resolved SingleShot MTP config` log
line does not print `pad_to_multiple`.

## TL;DR

`torch.compile(dynamic=True)` and shape padding alone were **not sufficient**:
the Qwen3.6 SingleShot MTP run still hit Dynamo's `recompile_limit` because
rollout replay changed metadata/control flow, not just tensor shapes. The active
fix keeps model compile enabled without compiling across Python mask objects:
full-attention layers use the compiled `flex_attention` kernel with a static
full `BlockMask` table and tensor-backed `mask_mod`, but their decoder-layer
wrappers stay eager; GatedDeltaNet decoder layers remain compiled and use a
static-capacity `RolloutReplayLinearPlan` instead of Python virtual branch
construction inside the traced region. Server transport metadata is stripped
before every model call so per-request UUIDs do not become Dynamo guards.

## What gets compiled

`enable_compile` does **not** wrap the whole model. In
`src/xorl/distributed/torch_parallelize.py` (two sites, FSDP2 + PP paths) it
compiles **each decoder layer module** whose class is in
`model._no_split_modules` (+ any `basic_modules`):

```python
for fqn, mod in model.named_modules():
    if mod.__class__.__name__ in target_classes:
        compiled_mod = torch.compile(mod, dynamic=True)   # was: torch.compile(mod)
        setattr(parent, child_name, compiled_mod)
```

So the compiled region is the **per-layer decoder block forward** (attention /
linear-attention + MoE MLP), repeated for every layer. Qwen3.6-35B-A3B is a
hybrid stack — full-attention layers and `linear_attention` (GatedDeltaNet)
layers — and the inner linear-attention kernels are already excluded from
compile via `@torch.compiler.disable` (see
`src/xorl/ops/linear_attention/...`), so dynamo traces the layer wrapper around
them. The loss-side ops (`compiled_cross_entropy`, `policy_loss`) are compiled
separately and already use `dynamic=True`.

## Why the shapes are dynamic

This is on-policy distillation with SingleShot MTP, not fixed-length
pretraining. Every OPD step:

- The **student samples a fresh completion** per prompt; generated length varies
  (e.g. `mtp/generated_tokens` ranged ~1480–1750 over a 32-prompt step).
- The trainer forward/backward then runs over **prompt + generated** tokens.
  With `enable_packing: false`, microbatches are **not** packed/padded to a
  fixed length, so the SingleShot sequence length (`padded_seq_len`) changes per
  chunk/step. (`prompt_pad_to_multiple` defaults to `k_toks`=4, and the
  full-sequence `pad_to_multiple` is `None` → effectively no bucketing.)
- `torch.compile` (static) installs a guard on the input tensor's sequence
  dimension. A new length ⇒ guard miss ⇒ recompile. ~8 distinct lengths in step
  0 ⇒ `recompile_limit (8)` ⇒ dynamo stops compiling that frame and runs eager.

Evidence (pre-fix run `q36mtp-20260605T205142Z`):

```
[rank*]:W ... [0/8] torch._dynamo hit config.recompile_limit (8)   # once, step 0 (21:03:33)
```
- One-time burst across the 8 ranks, then **no further compile activity** →
  eager for steps 1–N.
- Per-step (eager) timings were flat (no warm-up speedup): `time_fwd` ~35–67 s
  (mean ~48 s), `time_bwd` ~220–289 s (mean ~266 s), `step_total` ~351 s.
- Throughput ~1.6 tok/s/GPU total, ~1.25 tok/s/GPU supervised (32 trainer GPUs).
  MFU ~0.02% — backward is FSDP-comm-bound and eager; compile was never the
  dominant cost, but the recompile churn was pure waste.

## What we tried

1. **`dynamic=True` on the per-layer `torch.compile`** (applied 2026-06-05 in
   `torch_parallelize.py::_compile_decoder_layers`, called from both PP and
   non-PP paths). This is retained because it is a numerics-neutral improvement
   over static shape guards for workloads that do compile, but the Qwen3.6
   SingleShot MTP run below showed it is **not sufficient** for this workload:
   the churn persisted because the mask structure is data-dependent, not just
   shape-dependent.

2. **Pad the SingleShot sequence to a multiple of N** (`pad_to_multiple` in
   `src/xorl/mtp/singleshot.py`, or `pad_to_multiple_of` in the packing path,
   default 128). This buckets the seq dim so only a few distinct shapes exist,
   but the 128-token padding run still hit the same recompile limit because the
   guard churn was not seq-shape-driven.
3. **Raise `torch._dynamo.config.recompile_limit`.** Doesn't help — with truly
   variable lengths it just churns more recompiles instead of falling back.
4. **`enable_compile: false`.** Operational fallback that removed the churn by
   not compiling decoder layers. This is no longer the target end state.
5. **Tensor-parameterized static FlexAttention BlockMask.** Port the core idea
   from `~/singleshot` PR #4: do not bake rollout layout into Python closures or
   compressed sparse block metadata. XORL's rollout replay mask now uses tensor
   metadata (`token_kind`, source indices, `block_ids`) plus a static full block
   table, so changing a rollout changes tensor contents rather than the traced
   graph shape/context.

## Status / results

- `dynamic=True` picked up on the trainer relaunch at **2026-06-05 22:42 UTC**
  (run `q36mtp-20260605T224206Z-2s2t`); `torch.compile applied to 40 decoder
  layers`.

**Microbenchmark result (step 0–1 of run 224206Z) — `dynamic=True` is NOT
sufficient:**

- ❌ Still recompiled to the limit: `[0/8] torch._dynamo hit
  config.recompile_limit (8)` fired again at 22:55:41 (during step 1) → eager
  fallback, same as before.
- Net step throughput unchanged: step 0 `time_fwd` 58.4s, `time_bwd` 263.6s,
  `step_total` 402s, sync 12.9s.
- This first *looked* like a ~2× forward win while compiled (chunk 1 25.8s
  compiling, chunks 2–4 ~10.9s), but the later `enable_compile:false` control run
  showed eager chunks 2–4 are **also ~8.5–10s** and eager chunk 1 is ~20.5s — so
  compile gave no real per-chunk speedup; the slow first chunk was warmup. (See
  Conclusion for the corrected numbers.)

**Why `dynamic=True` didn't stop it:** 8 distinct recompiles persist even with a
symbolic seq dim ⇒ the churn is **not** a single tensor dimension. Prime
suspect: the SingleShot **flex_attention 4D bias / block mask** — its block
structure is data-dependent on the rollout (prompt/gen boundaries, number of
mask blocks, partial blocks), and `dynamic=True` can't symbolicize a
data-dependent mask. Multiple dynamic axes (batch × seq) or 0/1 specialization
could also contribute.

**Padding attempt (run `q36mtp-20260605T230724Z`, `pad_to_multiple: 128` +
`dynamic=True`):** ❌ also hit `recompile_limit (8)` at step 1 (23:20:32) — the
exact same failure point as `dynamic=True` alone. The per-chunk forward seq is
narrow (`tokens=` 4444–4600 ⇒ ~2 buckets at 128), so if the churn were
seq-driven this would have fixed it. It did not.

## Conclusion

Both shape-level fixes failed identically:
- `torch.compile(dynamic=True)` (symbolic for **all** tensor dims) → recompile_limit 8.
- `dynamic=True` + `pad_to_multiple=128` (fixed seq buckets) → recompile_limit 8.

⇒ The original recompilation was **data-dependent, not tensor-shape-driven**:
the SingleShot rollout replay mask changed the FlexAttention/block-mask
structure as the rollout changed. The compile-stable fix is to make that
structure static and move the rollout variability into tensors.

**Fix applied:** enable compile for the 4-node config and use
`attn_implementation: flex_attention` for full-attention layers. Rollout replay
`BlockMask` construction now defaults to a static full block table via
`BlockMask.from_kv_blocks`; the table shape is fixed for a padded bucket and
`mask_mod` reads the per-rollout tensors to filter individual positions. The
current compile selector skips the full-attention decoder-layer wrapper under
FlexAttention so Dynamo does not guard on fresh `BlockMask` Python objects.

**Local validation:** a small CUDA FlexAttention probe passes fresh rollout
`BlockMask` objects with the same static table shape but different tensor
contents through `torch.compile(dynamic=True)`. Dynamo reports one frame and one
unique graph, with no recompile log entries. This is covered by
`tests/mtp/test_singleshot.py::test_rollout_replay_static_block_mask_reuses_compiled_flex_graph`.

**GatedDeltaNet follow-up:** the live run still recompiled after the
FlexAttention fix, which pointed at Qwen3.6's hybrid `linear_attention` layers.
The first GDN replay-plan patch removed Python list construction from the traced
region, but local variable-length replay reproduced the remaining guard:
`attention_mask.linear_plan.sequence_cols` changed at dim 0 as the number of
active virtual branches changed. That path now has the same static-boundary
treatment as FlexAttention: batch preparation pads replay-plan branch/output
slots to a capacity derived from `k_toks` and the padded sequence bucket, and
`GatedDeltaNet._forward_with_replay_plan` masks inactive slots with
`sequence_valid`/`output_valid`. Production-scale batches cannot materialize the
full static rectangle: `B=32,S=768,k=4` would gather 14,180,352 hidden rows before
MoE dispatch. The current path keeps the dense compiled replay only for small
plans and sends large plans through a packed `cu_seqlens` chunk path capped by
`replay_plan_max_packed_tokens` (default 65,536). That large-plan path is
`torch.compiler.disable` by design to avoid the OOM.

**Local validation:** fresh focused CUDA probes with `TORCH_LOGS=recompiles`
passed for both:
- `tests/mtp/test_singleshot.py::test_rollout_replay_static_block_mask_reuses_compiled_flex_graph`
- `tests/models/test_qwen3_5_singleshot_mask.py::test_qwen36_linear_attention_decoder_layer_compile_boundary_is_stable_on_cuda`
- `tests/models/test_qwen3_5_singleshot_mask.py::test_qwen36_mixed_flex_model_compile_keeps_blockmask_outside_decoder_layer_on_cuda`
- `tests/ops/test_linear_attention_singleshot_mask.py::test_gated_deltanet_static_replay_plan_capacity_avoids_shape_recompiles_on_cuda`
- `tests/ops/test_linear_attention_singleshot_mask.py::test_gated_deltanet_replay_plan_runs_packed_chunks_not_dense_capacity`

The GDN probe varies generated lengths across a fixed `static_padded_seq_len`;
active branch/output counts change, but `linear_plan` tensor shapes stay fixed.
The dense compiled GDN probe remains clean for small plans. The packed-chunk
regression verifies large padded plans do not expand to dense static capacity.
The mixed Flex/GDN probe compiles only the linear-attention decoder layer and
keeps the full-attention `BlockMask` at the eager wrapper boundary; no recompile
log entries appear in the small-plan case.

## 4-node live verification (2026-06-06) — flex-only caveat confirmed

Relaunched the live 4-node run (`q36mtp-20260606T002029Z-2s2t`) with the fix
active: `attn_implementation: flex_attention`, `enable_compile: true`, the static
`from_kv_blocks` BlockMask, `torch.compile(...dynamic=True)` (`torch.compile
applied to 40 decoder layers`), keeping `sync_inference_method: p2p` +
`ep_dispatch: alltoall` to isolate the compile change.

Result: **still hit `recompile_limit (8)` at step 1** (00:34 UTC), frame
`[0/8]`, then eager fallback — i.e. the flex-mask fix alone did not make the
4-node run compile-stable. Steady throughput was unchanged from eager.

Root cause was initially two-part. The first frame (`src/xorl/models/module_utils.py:2842`)
guarded on `kwargs['attention_mask']` being a fresh Flex `BlockMask` at the
compiled decoder-layer boundary. Separately, the first GDN replay-plan attempt
left the number of active virtual branches in tensor shapes. The current code
addresses both and also strips transport metadata before model calls; the
19:34 UTC smoke below supersedes this failed flex-only result.

## 4-node live verification (2026-06-06 19:34 UTC) — current patch stable

Relaunched a two-step trainer-only smoke using the same warm stack and current
patches:

- control logs:
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260606T193413Z-run.log`
  and worker logs `20260606T193412Z/20260606T193413Z`
- config summary included `mtp_pad_to_multiple=128` and
  `mtp_static_padded_seq_len=256`
- workers logged `torch.compile applied to 30 decoder layers`
- head completed:
  - step 0: `roundtrip=23.375s`, `forward_loss=17.283s`, `backward=5.445s`
  - step 1: `roundtrip=6.245s`, `forward_loss=1.224s`, `backward=3.930s`
  - `OPD pipeline validation succeeded.`
  - `trainer-head cleanup rc=0`

Recompile-only grep over all head/worker logs returned no matches for
`__recompiles`, `Recompiling function`, `guard failure`, `recompile_limit`,
`attention_mask.linear_plan`, `kwargs['request_id']`, `tensor 'key' size
mismatch`, or `BlockMask`. Worker-side Gloo tracebacks immediately after success
are shutdown noise from the head exiting and stopping the trainer control, not a
compile failure.
