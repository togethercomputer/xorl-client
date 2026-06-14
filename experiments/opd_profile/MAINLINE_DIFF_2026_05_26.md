# OPD Mainline Integration Diff - 2026-05-26

State recorded after the mainline OPD smoke run completed on 2026-05-26,
then amended after production-scale OPD convergence relaunches diagnosed the
checkpointing, KL-backend, and empty-rank metric-reducer issues.

## Result

The closest-to-main OPD state that completed a real smoke run is:

- `xorl`: `origin/main` `be68d08772fde80a45c597fa432f011679d621e0` plus the PR stack and local fixes listed below.
- `xorl` branch: `codex/opd-mainline-run-20260526`
- `xorl` tested runtime commit: `fb01c53bbbd218b228561724f60e2a1c5f4678a5`
- `xorl` active convergence test: `efc925b1f415a128b0b7481d0492728ef825ffd2` plus the local OPD metric-reducer patch.
- `xorl-client`: `opd-client-pipeline-additions` at `74380ca424fbd8f8b683f3979699a12e1d96c3cf`
- `sglang`: `4ece41975433d104af05573db957199a5b75399d`

Smoke run:

- Workload: `er-opdmain-052620`
- Trainer pod: `er-opdmain-052620-trainer-head-pf7nb`
- Run dir: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260526T211016Z-er-opdmain-052620-trainer-head-pf7nb`
- Wandb: `https://wandb.ai/together-research/xorl-prefill-time-compute/runs/fdvua0wx`
- Model/config: Qwen3.6-35B-A3B self-distill smoke, 4-node trainer, 1-node teacher, 8 SGLang endpoints.
- Scale: `num_prompts=128`, `num_steps=2`, `opd_microbatch_size=128`, `opd_prepare_batch_size=128`, `opd_prepare_concurrency=1`, `max_new_tokens=192`.

The run completed both OPD steps and wrote `opd_profile.jsonl`.

| Step | Total s | Fwd/Bwd s | Sync s | Loss | Valid tokens | Sync result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 153.552 | 106.546 | 27.906 | 0.125352 | 222208 | `sync_success=true`, 31333 params to 8 endpoints |
| 1 | 36.771 | 11.305 | 5.652 | 0.192205 | 206392 | `sync_success=true`, 31333 params to 8 endpoints |

The earlier apparent hang was not a trainer deadlock. `xorl-client` PR #7 loaded
all 8192 JSON prompts from `/shared/opd-coord/randnum_4digit_8192.json` even
when the smoke config set `num_prompts=128`. The local `xorl-client` fix
`74380ca` caps JSON and JSON-path prompts by `num_prompts`; after that fix the
smoke produced exactly one teacher cache file per step:

- `teacher_hidden_step0_mb0.safetensors`
- `teacher_hidden_step1_mb0.safetensors`

## Production-Scale Convergence Relaunch

The first production-scale OPD run was launched on the closest-to-main branch:

- Workload: `er-opdmain-052620`
- Manifest: `experiments/opd_profile/k8s/generated/er-opdmain-converge-052621.yaml`
- Trainer head pod: `er-opdmain-052620-trainer-head-twczs`
- Run dir: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260526T215209Z-er-opdmain-052620-trainer-head-twczs`
- Wandb: `https://wandb.ai/together-research/xorl-prefill-time-compute/runs/ivlax8z6`
- Scale: `num_prompts=8192`, `num_steps=400`,
  `opd_microbatch_size=128`, `opd_prepare_batch_size=128`,
  `opd_prepare_concurrency=4`, `max_new_tokens=192`.
- Teacher/student setup: `/shared/opd-coord/randnum_4digit_8192.json`,
  `teacher_filler_text=" pause"`, `teacher_filler_count=100`,
  P2P weight sync enabled.

The first convergence attempt on this manifest hung after partial step 0. The
trainer server had accepted the forward/backward and optimizer requests, but
worker ranks 24-31 logged `torch.utils.checkpoint.CheckpointError` from full
layer recomputation replaying MoE all-to-all with different packed-token tensor
metadata. The relaunch sets:

```yaml
enable_gradient_checkpointing: true
gradient_checkpointing_method: recompute_before_dispatch
```

This is a config-only fix. It keeps dispatch/expert/combine activations instead
of replaying the MoE all-to-all path during activation checkpoint recompute.

After isolating the failure, the branch also fixes the underlying full-layer
checkpoint boundary for Qwen3 MoE and Qwen3.5/3.6 MoE. The old implementation
wrapped the entire FSDP decoder-layer call from the outer model loop. On this
OPD workload that allowed checkpoint recompute to see different
FSDP/mixed-precision metadata, first at zero-centered RMSNorm and then in the
MoE all-to-all saved tensors. Full-layer checkpointing now lives inside the
decoder layer forward path, matching the ownership model already used by
`recompute_before_dispatch`.

The relaunched run completed the first three full production-scale OPD steps:

| Step | Total s | Fwd/Bwd s | Sync s | Loss | Valid tokens | Sync result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 741.475 | 688.696 | 26.672 | 0.136764 | 13,140,144 | `sync_success=true`, 31333 params to 8 endpoints |
| 1 | 673.882 | 642.722 | 4.998 | 0.201531 | 14,138,136 | `sync_success=true`, 31333 params to 8 endpoints |
| 2 | 686.841 | 655.071 | 4.995 | 0.328379 | 14,108,328 | `sync_success=true`, 31333 params to 8 endpoints |

Step 3 started immediately after the step-2 sync. Early step-3 microbatch
losses were around `0.15`, matching the expected recovery after the warmup
spike from the older successful run. No recent worker `CheckpointError`, OOM,
or runtime error was present after the `recompute_before_dispatch` relaunch.

Follow-up throughput diagnosis: the old successful run passed
`opd_kl_backend=streaming`, but that working snapshot did not implement the
streaming OPD KL backend and therefore used the compiled KL path. The mainline
branch does implement `streaming`, so the same manifest exercised a slower
pure-PyTorch streaming KL path. The generated smoke/convergence manifests were
updated to set `opd_kl_backend=torch_compile` explicitly while keeping
`gradient_checkpointing_method=recompute_before_dispatch`; this isolates the
speed regression check from the full-layer activation-checkpoint fix.

The first `torch_compile` KL relaunch reached trainer compute but hung before
writing an OPD profile row. `py-spy` showed ranks 0-23 blocked in
`ModelRunner._finalize_loss_metrics` while ranks 24-31 had already advanced to
`RunnerDispatcher._sync_error_state`. Empty OPD ranks had no local
micro-batches or metric accumulators, so they skipped the OPD loss-metric
all-reduces and entered the next Gloo collective.

This is not an entirely novel reducer idea, but the exact fix is not present in
the current OPD path on the relevant open PR heads. `origin/main`,
`origin/apanda/pr-step-phase-timing` (#292), and
`origin/apanda/pr-moe-act-recompute` (#291) still have the early
`if not accumulated: return` in `_finalize_loss_metrics`. The closest existing
work is #127 / `origin/feature/custom-loss-functions`, which uses identity
elements for empty ranks in older KL/ratio reducers; it predates the current
namespaced OPD metrics such as `opd_num_teachers:max` and the current
`_finalize_loss_metrics` path. The local patch ports that invariant by seeding
zero-valued OPD metric accumulators on empty ranks before finalization.

A broader `gh api` scan of all 66 open PR heads on 2026-05-27 found no PR with
`_ensure_opd_loss_metric_accumulators`, and no PR whose `model_runner.py`
removed the early `if not accumulated:` return. Sixteen open PR heads already
contain the current `opd_num_teachers:max` metric shape, including #291/#292,
but they still retain the empty-rank early return.

The first relaunch after that patch proved the reducer fix but was not a valid
speed comparison:

- Trainer head pod: `er-opdmain-052620-trainer-head-v722m`
- Run dir: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260526T233728Z-er-opdmain-052620-trainer-head-v722m`
- Runtime: `opd_kl_backend=torch_compile`,
  `gradient_checkpointing_method=recompute_before_dispatch`,
  `efc925b1` plus the local OPD metric-reducer patch.
- Result: step 0 wrote an OPD profile row with `sync_success=true`, so the
  previous metric collective mismatch was cleared. It is not speed-comparable:
  the long-lived SGLang pods had already received weights from an earlier
  partial run, and step-0 sampling produced only 339,944 student output tokens
  instead of the old/full-run ~1.55M token workload.

The stack was then fully torn down and relaunched from
`er-opdmain-converge-052621.yaml` so the SGLang samplers start from fresh model
weights:

- Trainer head pod: `er-opdmain-052620-trainer-head-7rnvm`
- Run dir: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260526T235245Z-er-opdmain-052620-trainer-head-7rnvm`
- Sampler pods restarted at `2026-05-26T23:52:37Z` and all eight SGLang
  services answered `/health` by `2026-05-26T23:57:36Z`.
- This is the run to use for the next apples-to-apples speed and convergence
  comparison.

The clean relaunch completed warmup step 0 and measured step 1 with full token
volume and successful P2P sync:

| Step | Warmup | Total s | Fwd/Bwd s | Sync s | Loss | Student output tokens | Teacher prefill tokens | Sync result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | yes | 767.845 | 659.608 | 31.759 | 0.125568 | 1,572,458 | 1,777,258 | `sync_success=true` |
| 1 | no | 667.113 | 635.784 | 4.710 | 0.192965 | 1,446,754 | 1,651,554 | `sync_success=true` |

This proves the `torch_compile` KL path plus the empty-rank metric fix runs
through a clean production-scale OPD step, but it does not recover the old
speed. Compared with the previous mainline `streaming` run, measured step 1 is
only about 1.4% faster end-to-end and 1.5% faster in forward/backward. Compared
with the old successful run's stable mean, it is still about 43.5% slower
end-to-end and 45.7% slower in forward/backward:

| Run | Basis | Total s | Fwd/Bwd s | Notes |
| --- | --- | ---: | ---: | --- |
| Old working OPD | mean steps 56-101 | 464.895 | 436.506 | Effective compiled KL path despite manifest label. |
| Mainline streaming | mean steps 1-3 | 676.684 | 645.199 | Explicit `streaming` backend on mainline. |
| Clean mainline compile | measured step 1 | 667.113 | 635.784 | Valid full-token run, no reducer deadlock. |

The remaining 50% speed target therefore is not explained by the new streaming
KL backend alone. The next bottleneck to isolate is the current
`ModelRunner`/OPD loss path and its per-microbatch timing delta versus the old
working snapshot.

A profiling comparison was added in
`experiments/opd_profile/OPD_PROFILE_GAP_2026_05_27.md`, generated by
`experiments/opd_profile/analyze_opd_run_gap.py`. It compares old stable OPD
steps 56-101 with clean compile steps 1-2. The clean run is 184.971s slower per
selected OPD step; 181.670s of that is direct `forward_backward_s`. Server
microbatch timing shows a broad per-microbatch slowdown, not a single stall:
runner median is 5.990s clean versus 3.290s old, and executor median is 9.544s
clean versus 6.667s old. Student sampling and P2P sync are faster in the clean
run, so they are not the current speed bottleneck. Teacher hidden-cache forward
compute is also much slower, which points at shared model forward / OPD server
execution paths rather than SGLang sampling or weight sync.

## Current Delta From Main

The current branch head is `origin/main` plus 31 commits, plus the uncommitted
OPD metric-reducer patch. The first 27 commits are the smoke-tested
runtime/library and smoke-manifest stack; later commits add reports, convergence
manifests, and the full-layer checkpoint-boundary fix.

```text
20159836 fix(p2p): remove unstable small-entries GPU-direct path (Bug 7)
da32bb3a feat(weight-sync): XORL_P2P_SKIP_PARAM_PATTERNS + skip-moe-experts knobs
c9f96531 fix(weight-sync): wire SKIP_MOE_EXPERTS + fix glob anchoring + tests
b2ccbc83 feat(trainer): step-phase + per-component timing and memory profiling
9e847136 chore(lint): apply ruff-format to trainer.py
35c6eddf revert: drop unrelated clip_gradients early-return
33101ed2 test(trainer): cover step-phase summary + per-component timer helpers
1c255e14 feat(trainer): skip gradient-norm computation when max_grad_norm <= 0
975d2f6a feat(moe): gradient_checkpointing_method=moe_act recomputes activation only
954a17f2 chore(lint): remove unused os import in triton.py
f375fb69 fix(moe): wire moe_act flag and define _ROUTING_WEIGHTS_BEFORE_DOWN
f4dd5334 perf(moe): tunable accum dtype + chunked scatter for DeepEP grad/combine
932a6fd1 refactor(moe): cache scatter knobs + factor combine path + add CPU tests
8d92ce35 fix(moe): always attach R3 in post_init, not only via grad-ckpt
5bae200a experiment(opd): add mainline smoke run manifest
e28ec291 experiment(opd): align smoke sglang node selectors
e04f7b75 experiment(opd): move smoke sglang shard off occupied node
dac62418 experiment(opd): move smoke trainer rank off unreachable node
7ab1449b fix(server): keep forward evals out of inference_mode
29f0ceee experiment(opd): move smoke teacher off occupied node
c1e281b4 fix(server): filter masked teacher cache rows
0d73e6b1 experiment(opd): move smoke pods off occupied nodes
e2f53a17 experiment(opd): let scheduler place smoke fallback pods
2032db86 experiment(opd): widen smoke teacher placement
01992f6e experiment(opd): schedule smoke teacher on nccl nodes
2a10e800 fix(opd): default missing streaming vocab chunk size
fb01c53b fix(server): use returned metric ops in forward loop
41f16033 docs(opd): record mainline integration diff
7b67ee20 experiment(opd): add convergence validation manifest
88400e13 experiment(opd): avoid full-layer recompute in convergence run
efc925b1 fix(moe): keep full-layer checkpoints inside decoder layers
```

Untracked run outputs remain intentionally uncommitted:

- `experiments/encoded_reasoning/`
- `experiments/opd_profile/results/`

## Applied PR Set

| PR | Included commits | Role in working smoke |
| --- | --- | --- |
| xorl-internal #309 | `20159836` | Removes the unstable small-entries P2P GPU-direct path. This replaces the old manifest-only workaround `XORL_P2P_CPU_POOL_MIN_BYTES=0`. |
| xorl-internal #293 | `da32bb3a`, `c9f96531` | Adds P2P skip-pattern and skip-MoE-experts controls used by the OPD sync path. |
| xorl-internal #292 | `b2ccbc83`, `9e847136`, `35c6eddf`, `33101ed2` | Adds step-phase timing, per-component timing, memory profiling, and tests. |
| xorl-internal #296 | `1c255e14` | Skips gradient norm computation when `max_grad_norm <= 0`. |
| xorl-internal #291 | `975d2f6a`, `954a17f2`, `f375fb69` | Adds `gradient_checkpointing_method=moe_act` and wires MoE activation recompute. |
| xorl-internal #289 | `f4dd5334`, `932a6fd1` | Adds DeepEP accum dtype tuning and chunked scatter/combine helpers. |
| xorl-internal #308 | `8d92ce35` | Always attaches R3 routing replay during `post_init`. |
| xorl-client #7 | `4e120cb` | Adds OPD teacher filler/system-prefix, `prompts_json_path`, and wandb logging. |

## Not Applied

| PR | Status | Reason |
| --- | --- | --- |
| xorl-internal #298 | Not applied | Muon/quack tuned env work; not required for this OPD smoke. |
| xorl-internal #302 | Not applied | K3/static-trace tooling; useful for validation workflows but not required for OPD runtime. |

## Local Fixes Beyond PR Heads

These are the runtime fixes still needed beyond the applied PR set:

| Commit | Repo | Fix |
| --- | --- | --- |
| `7ab1449b` | `xorl` | Keeps forward/eval teacher-cache paths out of `torch.inference_mode()` so later autograd paths remain valid. |
| `c1e281b4` | `xorl` | Filters masked teacher-cache rows before writing packed hidden caches. |
| `2a10e800` | `xorl` | Defaults missing OPD streaming vocab chunk size so `None` does not break the streaming KL path. |
| `fb01c53b` | `xorl` | Uses the returned OPD metric ops inside the forward loop instead of an undefined `metric_ops` variable. This fixed the previous `NameError` failure. |
| `efc925b1` | `xorl` | Moves Qwen3/Qwen3.5 MoE `recompute_full_layer` checkpoint boundaries inside decoder layers so recompute stays inside the layer FSDP/mixed-precision context. |
| uncommitted | `xorl` | Seeds zero-valued OPD loss-metric accumulators on empty ranks so all ranks enter the same `_finalize_loss_metrics` collectives. This is a local port of the older empty-rank reducer invariant, not present in current main/#291/#292. |
| `74380ca` | `xorl-client` | Caps JSON and JSON-path prompt loading by `num_prompts`. This turned the intended 128-prompt smoke from an 8192-prompt run into the actual one-microbatch smoke. |
| uncommitted | `xorl-client` | Aggregates server-returned OPD profile sub-phase metrics into `opd_profile.jsonl` for the next clean run. |

The remaining `experiment(opd)` commits in this branch are smoke-manifest and
node-placement changes only. They are not runtime library changes.

## Difference From The Old Working Snapshot

Old snapshot:

- Branch: `codex/er-dispatch-working-snapshot-20260526`
- Snapshot commits: `ed8d1aff` and `5692b0aa`
- Shape: older OPD/P2P branch lineage plus applied stash overlays.
- PR #309 was not present as code; the successful run used
  `XORL_P2P_CPU_POOL_MIN_BYTES=0`.
- Several PR semantics were present in adapted/stash form rather than clean PR
  heads.

This mainline branch:

- Starts from current `origin/main` `be68d087`.
- Applies clean PR-head commits for #289, #291, #292, #293, #296, #308, and #309.
- Does not depend on applying the old stashes.
- Adds the local `xorl` runtime fixes and one local `xorl-client` fix needed
  to make the mainline OPD smoke complete and the production-scale run progress.

The old stashes still exist in the shared stash list and were not dropped or
rewritten:

- `stash@{0}`: `wip-pre-main-reset-2026-05-26`
- `stash@{1}`: `session-baseline-pre-cherrypick`

## Files Changed Versus Main

Runtime/library and tests:

- `src/xorl/arguments.py`
- `src/xorl/distributed/moe/deepep.py`
- `src/xorl/models/base.py`
- `src/xorl/models/layers/moe/backend/__init__.py`
- `src/xorl/models/layers/moe/experts.py`
- `src/xorl/models/module_utils.py`
- `src/xorl/models/transformers/qwen3_moe/modeling_qwen3_moe.py`
- `src/xorl/ops/loss/opd_loss.py`
- `src/xorl/ops/loss/opd_streaming_kl.py`
- `src/xorl/ops/moe/quack.py`
- `src/xorl/ops/moe/triton.py`
- `src/xorl/server/runner/model_runner.py`
- `src/xorl/server/server_arguments.py`
- `src/xorl/server/weight_sync/backends/p2p.py`
- `src/xorl/server/weight_sync/handler.py`
- `src/xorl/trainers/model_builder.py`
- `src/xorl/trainers/per_component_timer.py`
- `src/xorl/trainers/trainer.py`
- `src/xorl/trainers/training_utils.py`
- `tests/distributed/test_deepep_scatter_helpers.py`
- `tests/ops/loss/test_opd_loss.py`
- `tests/server/runner/test_opd_runner.py`
- `tests/server/weight_sync/test_handler_config.py`
- `tests/server/weight_sync/test_p2p_async_api.py`
- `tests/server/weight_sync/test_p2p_backend_protocol.py`
- `tests/trainers/test_per_component_timer.py`
- `tests/trainers/test_step_phase_timing.py`
- `tests/trainers/test_training_utils.py`

Experiment artifacts:

- `experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdfix_4node_pack4096.yaml`
- `experiments/opd_profile/configs/qwen3_6_35b_a3b_teacher_1node_ep8.yaml`
- `experiments/opd_profile/k8s/generated/er-opdmain-052620.yaml`
- `experiments/opd_profile/k8s/generated/er-opdmain-converge-052621.yaml`
- `experiments/opd_profile/MAINLINE_DIFF_2026_05_26.md`

## Verification

Completed cluster verification:

- The trainer head pod completed.
- Both OPD profile rows were written.
- Both P2P syncs completed with `sync_success=true`.
- The prompt-cap fix was verified by one teacher-cache file per OPD step.

Targeted local verification passed in this worktree:

```bash
PYTHONPATH=$PWD/src python3 -m pytest -q \
  tests/server/runner/test_opd_runner.py::test_forward_uses_no_grad_not_inference_mode \
  tests/server/runner/test_opd_runner.py::test_teacher_hidden_cache_filters_masked_targets_in_packed_batch \
  tests/server/runner/test_opd_runner.py::test_teacher_hidden_cache_filters_with_gathered_sp_labels \
  tests/server/runner/test_opd_runner.py::test_forward_loop_accumulates_opd_metrics_without_metric_ops \
  tests/server/runner/test_opd_runner.py::test_opd_metric_accumulators_cover_empty_rank \
  tests/ops/loss/test_opd_loss.py::test_opd_streaming_backend_accepts_none_vocab_chunk_size \
  tests/ops/loss/test_opd_loss.py::test_opd_streaming_backends_match_reference

PYTHONPATH=$PWD/src python3 -m pytest -q \
  tests/server/runner/test_opd_runner.py \
  tests/models/test_gradient_checkpointing.py

python3 -m py_compile \
  src/xorl/server/runner/model_runner.py \
  src/xorl/ops/loss/opd_streaming_kl.py \
  src/xorl/ops/loss/opd_loss.py

git diff --check origin/main..HEAD -- src tests experiments/opd_profile
```

Result: targeted OPD/loss tests passed; full targeted runner/checkpointing suite
passed with `28 passed`; `py_compile` passed; `git diff --check` passed.

Conflict-marker sanity also passed:

```bash
grep -RInE '^(<<<<<<<|>>>>>>>|=======$)' src tests experiments/opd_profile --exclude-dir=results
```

`xorl-client` verification for the local prompt-cap fix:

```bash
python3 -m pytest -q tests/test_on_policy_distillation_example.py
```

Result: `17 passed`.
