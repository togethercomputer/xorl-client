# OPD Throughput Microbench Runbook

Last updated: 2026-06-15 15:57 UTC.

This is the focused handoff for Agent #1 throughput work on the filler-token /
prefill-time-compute OPD setup. It exists because the full infra runbook is dense
and because the next throughput loop should not require the full 32-GPU OPD
stack.

## Start Here

Work from:

```bash
cd /home/apanda/xorl-opd-prefill
```

Read these in order:

1. `AGENT_COORDINATION.md`
2. this file
3. `experiments/opd_profile/autoresearch/CANONICAL_INFRA_RUNBOOK.md` §0 and §7e

Core rule: **iterate on one node first.** The stack is far below acceptable MFU,
so the next throughput agent should build and optimize a 1-node reproducer or
surrogate for the fwd/bwd shape before spending 32 H100s. Use 4-node replay only
as a later fidelity/promotion gate when a 1-node candidate has a reason to
survive.

## What Is Already Known

- The full OPD science stack is not needed for fwd/bwd attribution. AMDAHL-021
  captured a static OPD fwd/bwd payload and replayed it through the trainer API
  without samplers, teacher prefill, endpoint registration, optimizer, or weight
  sync.
- The 4-node trainer-only replay baseline measured
  `server_forward_backward_s=4.4588`, with forward/backward/clear-grad split
  around `1.60 / 1.58 / 0.63 s`.
- The sorted AMDAHL-021 replay split showed the forward bucket was not just the
  student model: `model_forward_s=0.8056`,
  `oprd_teacher_forward_s=0.8498`, `loss_compute_s=0.9071`,
  `backward_compute_s=1.5860`, `clear_gradients_s=0.5882`.
- MFU remains bad even on executed tokens: the denominator audit reconstructs
  about `1.37%` logical MFU over dispatcher-executed student tokens, and only
  `0.039%` when scaled by valid answer tokens. Do not call this acceptable.
- AMDAHL-028 rejected a dominant fixed server/API tax: minimal dummy rows were
  neutral (`4.4612 s`), and repeated static data improved executed throughput
  only from `520.9 -> 584.4 -> 775.5 tok/s/GPU` at repeat 1/2/4.
- AMDAHL-025 was a replay-only false positive: pack2304 improved static replay
  (`4.1632 s`) but AMDAHL-026 regressed the real full strict run
  (`forward_backward_s=11.03 s` vs AMDAHL-020's `4.21 s`).
- EP=1 was tested in replay and lost (`5.8706 s`), so do not rerun it as the
  next obvious lever.
- No-checkpoint was tested in warmed replay and lost/was neutral
  (`4.6051 s`), so do not spend the next pass there.

## Engine Branch State — Vocab-Parallel KL / lm-head TP (2026-06-15 04:13Z)

Draft PR #375 (`throughput/opd-vocab-parallel-kl`) is pushed through
`706256f4` with opt-in model-runner wiring for
`opd_kl_backend: vocab_parallel`. It avoids the full lm-head module anchor on the
VP path, resolves the local DTensor lm-head row range, loads the matching
teacher-store vocab rows via `TeacherHeadShardView.load_rows_device(start, end)`,
keeps zero-token ranks in the same teacher/group collectives, and uses a VP OPD
loss helper that preserves full KL gradient scale while dividing the detached KL
value for reporting. The latest pushed follow-up also adds DP-sourced
no-CP lm-head TP for trainer-server replay. The same-topology no-CP loss drift
was later traced to dtype materialization: the VP DTensor lm-head shard loaded
as fp32 while the normal gathered FSDP/mixed-precision path consumed bf16-rounded
lm-head weights. The local patch now casts the VP student shard to the hidden /
model compute dtype before loss use. This fixes the limit8 parity blocker, but
the 4-node all-layer OPRD replay still reconstructs only about `1.04%` logical
MFU, so this is not promoted.

Additional server/topology support now included in the same branch:

- `ServerArguments` accepts and exports `lm_head_tensor_parallel_size`,
  `fsdp_sharded_lm_head_loss`, and
  `fsdp_sharded_lm_head_loss_num_chunks`.
- `ModelRunner` forwards `fsdp_sharded_lm_head_loss` into the shared
  `build_training_model()` path.
- `build_training_model()` forwards `fsdp_sharded_lm_head_loss` to
  `build_parallelize_model()`.
- Server backward syncs lm-head-TP replica gradients, but no-ops when the
  replica group world size is `1` (`lm_head_tensor_parallel_size == CP size`).
  This avoids an unnecessary NCCL all-reduce of the huge local lm-head grad.
- PR #375 follow-up: `parallel_state.py` can now source
  `lm_head_tensor_parallel_size` from the DP-shard axis when CP/Ulysses/Ring are
  disabled. For the AMDAHL-045 4-node topology this gives
  `source_axis=dp`, `source_replica=4`, `mesh=(4,8)` with
  `data_parallel_shard_size=32`, `ulysses_parallel_size=1`, and
  `lm_head_tensor_parallel_size=8`.
- PR #375 follow-up: `TeacherActivationCache.get(..., cache_device=True)` can
  keep one activation or layer cache resident on the target GPU. The runner
  exposes `opd_teacher_activation_cache_device_cache`,
  `opd_teacher_hidden_cache_device_cache`, and
  `opd_teacher_layer_cache_device_cache`; defaults are false.
- PR #375 follow-up: the SGLang layer-cache gather now fetches only valid teacher
  rows (`teacher_cache_indices[teacher_mask]`) instead of gathering the padded
  full batch and filtering afterward.
- PR #375 follow-up: OPRD all-layer hidden MSE is computed in layer chunks via
  `_oprd_hidden_distance()` so the loss no longer materializes all selected
  layers as fp32 at once.
- PR #375 follow-up: rank-3 SGLang layer-cache reads now use
  `TeacherActivationCache.get_layer_slice()` plus a loss-level
  `teacher_layer_fetcher`, so the trainer fetches/compares teacher OPRD layers
  by chunk instead of first materializing the full `[valid, layers, hidden]`
  teacher tensor. This preserves the full-tensor path for existing callers.
  `opd_oprd_layer_chunk_size` now controls the streamed layer width; default is
  `4`, so existing configs keep the same behavior while replay candidates can
  trade memory against slice-launch overhead.
- Draft SGLang PR #48 (`throughput/teacher-hidden-cache-tensor-output`,
  commit `c8552f4fa`) makes `/teacher_hidden_cache` set
  `teacher_hidden_cache_tensor_output=true`, so the scheduler returns selected
  hidden rows as CPU tensors instead of nested Python float lists. The cache
writer accepts tensor chunks and splits all-layer payloads per sample before
concatenation, avoiding the wide `base + L*hidden` Python-list expansion that
likely blocked real all-40 capture. This has now been validated by a real
full64 all-layer cache capture and one-node trainer replay: the capture wrote
rank-2 hidden `[3542,2048]` plus rank-3 all-layer `[40,3542,2048]`, and the
real-cache replay measured `server_forward_backward_s=4.9408` over 3 warmed
iterations. It is still not promoted because the static/K3 gate and a
  same-workload 4-node real-cache gate are pending. A later one-node
  `opd_oprd_layer_chunk_size` sweep found the default chunk size `4` is still the
  fastest stable setting on the real all-layer cache replay: chunk `4` measured
  `server_forward_backward_s=4.6591`, chunk `8` regressed to `4.9491`, and chunk
  `16` OOMed in FSDP pre-backward all-gather after one measured row. The
  subsequent AMDAHL-048 2-node retry completed at
  `server_forward_backward_s=4.4441`, a small improvement over the one-node
  chunk4 replay but still not a promotion or a 4-node/K3 gate. AMDAHL-049
  rejected pack2304 for this real-cache path, and AMDAHL-050 rejected disabling
  the runner allocator flushes: it removed the replay clear-gradient cache flush
  but OOMed on the next measured request under the current near-full memory
  envelope. AMDAHL-051 then screened manual FSDP module prefetch on the same
  no-CP/lm-head-TP real-cache path; it was neutral (`4.6506s` vs `4.6591s`) and
  changed the reported loss by about `0.0012`, so it is not a promotion.
  AMDAHL-052 then retried the fatter-call idea on the current real-cache path
  with `--repeat-data 2`; it packed 128 samples into 43 rows but OOMed during the
  warmup FSDP pre-backward all-gather while allocating `970 MiB`, so it is
  rejected without a measured replay row. AMDAHL-053 split CPU GC from CUDA cache
  release and skipped only CPU `gc.collect()` during warmed replay. That reduced
  explicit `clear_gradients_s` (`0.1783s` vs `0.4339s`) but regressed total
  server/API wall (`5.4222s`/`5.6626s` vs `4.6591s`/`4.9797s`) with time moving
  into model forward/backward, so keep the default CPU GC path for this workload.
  AMDAHL-054 then tried reducing DeepEP reserved SMs from `36` to `24` on the
  same real-cache no-CP/lm-head-TP VP-KL path. It regressed server wall
  (`4.7599s` vs `4.6591s`) and slightly shifted loss (`2.3569047` vs
  `2.3550419`), so keep `deepep_num_sms=36` for this workload.
  AMDAHL-055 then tried the custom MoE BF16-a2a expert-gradient reduce hook on
  the same one-node path, but it failed the launch gate before replay: singleton
  expert FSDP on one node used `mp_policy.reduce_dtype=torch.bfloat16`, while
  `moe_grad_reduce_mode=bf16_a2a_fp32_sum` requires FP32 reduce buffers. Do not
  retry that hook as a one-node screen without an engine/topology fix.
  AMDAHL-056 then tried plain global `fsdp_reduce_dtype: bf16` on the same
  one-node path. It preserved the loss but regressed server/API wall
  (`4.8815s`/`5.1059s` versus `4.6591s`/`4.9797s`), so do not promote BF16 FSDP
  reduce-scatter for this real-cache workload.
	  AMDAHL-057 then switched expert dispatch from DeepEP to `alltoall` on the same
	  one-node path. It reached warmup forward/backward but OOMed before writing a
	  replay row in FSDP pre-backward all-gather while allocating `970 MiB`, so do
	  not promote `ep_dispatch=alltoall` for this memory envelope.
		  AMDAHL-058 then tested deferring the detached scalar loss-report all-reduce
		  from once per packed microbatch to once per `forward_backward` call. It
		  preserved loss but regressed server/API wall (`4.8040s`/`5.0065s` versus
		  `4.6591s`/`4.9797s`), so the reporting-path patch was reverted/not promoted.
	  AMDAHL-059 then tested EP4 x ep_fsdp2 instead of the baseline EP8 singleton
	  expert-FSDP shape on the same DeepEP/SMS36 no-CP/lm-head-TP path. It required
	  a default-preserving `load_checkpoint_optimizer` knob because the EP8 DCP
	  optimizer state is shape-incompatible with EP4; the retry skipped optimizer
	  state only for this server-only topology screen. EP4 measured slightly faster
	  wall (`4.5983s`/`4.8492s` versus `4.6591s`/`4.9797s`) but shifted reported
	  loss/KL by `+0.0181`, so it is rejected and not a promotion candidate.
  AMDAHL-060 then spent the 2-node memory headroom on
  `reshard_after_forward:false` to test whether avoiding FSDP re-gather/reshard
  traffic helps the current 2-node replay. It was flat on server wall
  (`4.4440s` versus AMDAHL-048 `4.4441s`), with time moving from backward into
  model forward and clear-gradients, so it is rejected and not a promotion.
  AMDAHL-061 then retested `enable_forward_prefetch:true` in the 2-node topology.
  It was a real speed win (`3.6775s`, -17.25% server wall versus AMDAHL-048) but
  shifted same-capture KL/loss by about `+0.00458`, so it is rejected for
  promotion and should be treated as a correctness-risk speed lever until the KL
  drift is explained. AMDAHL-062 then split manual FSDP prefetch direction into
  default-preserving `enable_forward_prefetch` and `enable_backward_prefetch`.
  The 2-node candidate kept forward prefetch off and enabled backward prefetch
  only. It preserved same-capture loss/KL exactly, but measured only a small
  speed win versus AMDAHL-048 (`4.2808s`, -3.67% server wall) and is still
  16.4% slower than the rejected AMDAHL-061 server wall. Treat backward-only
  prefetch as a correctness-clean small lever, not the missing 10% MFU fix.
  AMDAHL-063 then measured the matching forward-only isolation screen
  (`enable_forward_prefetch:true`, `enable_backward_prefetch:false`). It
  preserved AMDAHL-048 loss/KL/hidden exactly and measured `4.1402s` mean
  server wall, so forward-only prefetch is a correctness-clean small speed lever
  on this replay. The rejected AMDAHL-061 drift is therefore not explained by
  forward-only manual prefetch alone; it came from the old coupled
  forward+backward prefetch mode or from state specific to that run.

Validation that has passed:

- `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python tests/ops/loss/test_vp_kl_gathered.py`
  - original gathered VP-KL: hidden-grad rel `8.655e-07`, weight-grad rel
    `1.487e-06`
  - uneven counts `(7,0,13,3)`: hidden-grad rel `1.211e-06`, weight-grad rel
    `2.487e-06`, gathered rows `[23]`
  - weighted OPD helper: hidden-grad rel `7.654e-07`, weight-grad rel
    `2.698e-06`, all-reduced reported-loss error `1.550e-06`
  - DTensor row metadata check: ranges
    `[(0,6), (6,12), (12,18), (18,23)]` for a 23-row vocab over 4 ranks
- `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/trainers/test_training_utils.py::test_sync_lm_head_tp_gradient_skips_single_rank_replica_group tests/trainers/test_fp8_model_builder.py::test_build_training_model_threads_sharded_lm_head_loss_to_parallelize tests/server/runner/test_model_runner_fp8_training.py::test_model_runner_threads_sharded_lm_head_loss_to_model_builder tests/server/test_server_arguments.py::test_load_server_arguments_threads_lm_head_tp_loss_fields -q`
  passed `4` tests.
- `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_lm_head_tp_fsdp_e2e.py tests/distributed/test_lm_head_tp_ep_parallel_state.py -q`
  passed `3` distributed tests.
- Local post-`5c17f409` no-CP lm-head TP validation:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py -q`
  passed `5` tests, including CP-sourced and DP-sourced no-CP group membership
  plus FSDP-sharded lm-head loss/hidden-grad/weight-grad equivalence for
  `dp=4, cp=1, lm_head_tp=2`.
- Regression validation after the local no-CP change:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_lm_head_tp_ep_parallel_state.py tests/distributed/test_lm_head_tp_loss.py tests/trainers/test_training_utils.py::test_sync_lm_head_tp_gradient_skips_single_rank_replica_group tests/trainers/test_fp8_model_builder.py::test_build_training_model_threads_sharded_lm_head_loss_to_parallelize tests/server/runner/test_model_runner_fp8_training.py::test_model_runner_threads_sharded_lm_head_loss_to_model_builder tests/server/test_server_arguments.py::test_load_server_arguments_threads_lm_head_tp_loss_fields -q`
  passed `6` tests.
- Earlier PR validation also passed
  `pytest tests/utils/test_distillation_teacher_cache.py tests/ops/loss/test_opd_loss.py -q`.
- Latest layer-cache / chunked-OPRD validation passed after the streaming
  layer-slice follow-up:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m py_compile src/xorl/distillation/teacher_cache.py src/xorl/ops/loss/opd_loss.py src/xorl/server/runner/model_runner.py tests/ops/loss/test_opd_loss.py tests/server/runner/test_model_runner_opd_layer_cache.py tests/utils/test_distillation_teacher_cache.py`
  and
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/ops/loss/test_opd_loss.py tests/server/runner/test_model_runner_opd_layer_cache.py tests/utils/test_distillation_teacher_cache.py -q`
  (`35 passed`, `16 warnings`). The distributed/VP-KL regression
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py tests/ops/loss/test_vp_kl_gathered.py -q`
  still passed (`6 passed`, `16 warnings`).
- Latest SGLang tensor-output cache validation passed:
  `python -m py_compile python/sglang/srt/entrypoints/teacher_hidden_cache.py python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/tokenizer_manager.py python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/scheduler_output_processor_mixin.py test/srt/zorl/test_teacher_hidden_cache_writer.py`
  and
  `PYTHONPATH=/home/apanda/xorl-sglang-internal/python /home/apanda/xorl-sglang-internal/.venv/bin/python -m pytest test/srt/zorl/test_teacher_hidden_cache_writer.py -q`
  (`2 passed`, warnings only). The endpoint test verifies the generated request
  flag, kept-position indices, sorted capture-layer indices, tensor metadata
  handling, and rank-2/rank-3 safetensors output.
- Packaging update: xorl draft PR #375 is open at commit `706256f4`; SGLang draft
  PR #48 is open at commit `c8552f4fa`. Both are draft because static/K3
  correctness and same-workload 4-node replay gates are still pending.
- Live teacher validation now passed on a temporary slots teacher pod pinned to
  `research-common-h100-110.cloud.together.ai` (`node-group=nccl`). Earlier
  2026-06-15 00:30Z capacity accounting had found no ready 8-GPU node, but the
  later node-110 window was used only for `teacher-sglang-0` capture and then
  only for `trainer-head` replay. Both pods were deleted afterward; slots now
  show only dispatch + teacher-smg, and the separate `er-opd-q36-35b-sci` stack
  was not touched.
- After AMDAHL-045 exposed the no-CP loss mismatch, the VP wrapper test was
  extended and rerun:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python tests/ops/loss/test_vp_kl_gathered.py`
  passed, including a new two-group DP-sourced lm-head TP case. That miniature
  case validates local hidden gradients, replica-summed vocab-shard weight
  gradients, and world-reduced reported loss against a full-vocab reference
  (`hidden grad rel=2.019e-06`, `weight grad rel=1.950e-06`,
  reported-loss abs err `1.110e-06`). So the live AMDAHL-044/045 loss mismatch
  is not explained by a simple VP loss-wrapper duplicate/reduction bug.
- Production-path no-CP OPD FSDP validation was added and rerun:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py -q`
  passed `6` tests. The new case uses a real FSDP-sharded `lm_head.weight`
  DTensor, `ModelRunner._opd_student_vocab_shard()` row-range extraction,
  matching teacher row slicing, OPD token weights, no-CP DP-sourced lm-head TP,
  and `sync_lm_head_tp_gradient()`. It matches a full-vocab reference for raw
  reported loss, local hidden gradients, and replica-summed weight-shard
  gradients. This closes the local production-wrapper gap, but the live trainer
  parity gate below still fails.
- A further local OPD kernel parity case compares
  `opd_vocab_parallel_loss_function` directly against the full-vocab
  `streaming_reverse_kl_function` with hidden-match disabled, token weights,
  clamp `10.0`, and uneven valid counts `(7,0,13,3)`. It passed with local
  hidden-grad max rel `4.661e-07`, weight-shard grad max rel `6.645e-07`,
  world-reduced reported-loss abs err `9.239e-07`, and KL metric abs err
  `4.090e-08`. This makes the live mismatch unlikely to be pure VP helper
  arithmetic when weights and hiddens are identical.
- Dtype-fix local validation passed after adding the opt-in lm-head fingerprint
  dump and effective-shard cast:
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python -m py_compile src/xorl/server/runner/model_runner.py`
  and
  `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python tests/ops/loss/test_vp_kl_gathered.py`.
  The latter includes gathered, uneven, weighted, DP-sourced lm-head TP,
  streaming-vs-VP, and DTensor row helper cases.
- Kubernetes auth is working on the normal context
  `apanda-admin@research-common-h100` (`kubectl auth can-i get pods -n apanda`
  returns `yes`). AMDAHL-046/044 parity ran on nccl nodes with explicit
  trainer-head `nodeSelector: node-group=nccl`; the 22:32Z enriched follow-up
  used only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`. Cleanup completed with trainer control stopped
  via `--remove-run`, the trainer-head pod deleted, no slots trainer pods
  remaining, and the isolated `er-opd-q36-35b-sci` stack untouched.

AMDAHL-042 trainer-server evidence:

- Teacher-store rows are at
  `/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json`
  for Qwen3.6-35B-A3B `lm_head.weight` shape `[248320, 2048]`, BF16, 8 row
  shards.
- 1-node topology config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp.yaml`
  with `ulysses_parallel_size=8`, `lm_head_tensor_parallel_size=8`,
  `fsdp_sharded_lm_head_loss=true`, `cp_fsdp_mode=all`,
  `enable_forward_prefetch=false`, and `load_weights_mode=skip`.
- Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-042-OPRD-PREP64-1NODE-LMHEADTP-VPKL.yaml`.
- Live trainer-head-only run on
  `research-common-h100-110.cloud.together.ai`, engine commit `5c17f409`,
  run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T204321Z-serveronly-configAMDAHL-042-OPRD-PREP64-1NODE-LMHEADTP-VPKL-er-opd-q36-35b-slots-trainer-head`.
- Successful replay artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-limit8-serveronly-1x-20260614T2045-size1skip.jsonl`.
  Result: `server_forward_backward_s=9.7336`, `api_wall_s=9.8009`,
  `model_forward_s=6.0761`, `kl_compute_s=0.1389`,
  `backward_compute_s=2.2063`, `clear_gradients_s=0.5099`,
  `valid_tokens=68`.
- The first same-server full64 replay after `limit-data=8` OOMed before KL in
  DeepEP forward dispatch metadata
  (`permutation_metadata_for_experts -> torch.argsort(sort_keys) -> CUDA OOM`,
  GPUs around `78-81 GiB`). That result is **superseded** by the fresh-server
  full64-first diagnostic below; do not treat the OOM as intrinsic to prep64 on
  this topology.
- Fresh-server full64-first run on
  `research-common-h100-099.cloud.together.ai`, run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T205035Z-serveronly-configAMDAHL-042-OPRD-PREP64-1NODE-LMHEADTP-VPKL-er-opd-q36-35b-slots-trainer-head`.
  It fit full64 cold (`server_forward_backward_s=138.7807`,
  `model_forward_s=85.0501`, `backward_compute_s=51.1890`,
  `kl_compute_s=0.4527`, `valid_tokens=515`), then stabilized after warmup at
  mean `server_forward_backward_s=24.8447`, `model_forward_s=9.3195`,
  `backward_compute_s=13.9033`, `kl_compute_s=0.1587`, `valid_tokens=515`
  over 3 full64 iterations.
- Fresh-server artifacts:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-full64-first-serveronly-1x-20260614T2054.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-limit8-after-full64-serveronly-1x-20260614T2056.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-full64-after-full64-limit8-serveronly-1x-20260614T2057.jsonl`,
  and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-full64-warmed-serveronly-3x-20260614T2058.jsonl`.
- Earlier AMDAHL-042 retries exposed and fixed three startup/runtime gaps:
  unknown server config fields, `build_training_model()` missing
  `fsdp_sharded_lm_head_loss`, and an unnecessary size-1 lm-head replica
  all-reduce that tried to allocate a 512 MiB NCCL buffer.
- Cleanup completed after the AMDAHL-042 diagnostics:
  `stop-trainer-control --remove-run`, killed local replay/port-forward/tail
  helpers, deleted `er-opd-q36-35b-slots-trainer-head`, and confirmed the
  separate `er-opd-q36-35b-sci` science stack was not touched.

AMDAHL-043 4-node fidelity evidence:

- 4-node topology config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp.yaml`
  with `data_parallel_shard_size=4`, `ulysses_parallel_size=8`,
  `lm_head_tensor_parallel_size=8`, `fsdp_sharded_lm_head_loss=true`,
  `cp_fsdp_mode=all`, `enable_forward_prefetch=false`, and
  `load_weights_mode=skip`.
- Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-043-OPRD-PREP64-4NODE-LMHEADTP-VPKL.yaml`.
- Trainer-only stack used only slots trainer pods with explicit
  `nodeSelector: node-group=nccl`: head on `research-common-h100-099`, workers
  on `110`, `089`, and `077`. No slots sampler/teacher/dispatch controls were
  changed, and `er-opd-q36-35b-sci` was not touched.
- Run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T205956Z-serveronly-configAMDAHL-043-OPRD-PREP64-4NODE-LMHEADTP-VPKL-er-opd-q36-35b-slots-trainer-head`.
- Fresh full64 4-node replay fit but was cold:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-4node-full64-serveronly-1x-20260614T2102.jsonl`,
  `server_forward_backward_s=135.5214`, `api_wall_s=135.9142`,
  `model_forward_s=84.5115`, `backward_compute_s=47.5739`,
  `kl_compute_s=0.2324`, `valid_tokens=515`.
- Warmed full64 4-node replay:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-4node-full64-warmed-serveronly-3x-20260614T2110.jsonl`,
  mean `server_forward_backward_s=15.4878`, `api_wall_s=15.7324`,
  `model_forward_s=4.6513`, `backward_compute_s=9.4123`,
  `kl_compute_s=0.0672`, `valid_tokens=515`. This is about `1.6x` faster than
  the warmed 1-node lm-head-TP replay (`24.8447s -> 15.4878s`), with forward
  around `2.0x` faster and backward only around `1.5x` faster.
- As of pushed commit `5c17f409`, a YAML-only "drop Ulysses but keep lm-head
  TP" follow-up was not legal: `parallel_state.py` required
  `lm_head_tp_size>1` to be carved from CP. The local AMDAHL-044/045 follow-up
  below implements the required code path by carving lm-head TP from the DP
  axis when CP is disabled.

AMDAHL-044/045 no-CP lm-head-TP evidence:

- 1-node topology config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml`
  with `data_parallel_shard_size=8`, `ulysses_parallel_size=1`,
  `ringattn_parallel_size=1`, `tensor_parallel_size=1`,
  `lm_head_tensor_parallel_size=8`, `fsdp_sharded_lm_head_loss=true`,
  `expert_parallel_size=8`, `enable_forward_prefetch=false`, and
  `load_weights_mode=skip`.
- 4-node topology config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml`
  with the same no-CP lm-head-TP settings and `data_parallel_shard_size=32`.
- Candidates:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-044-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL.yaml`
  and
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-045-OPRD-PREP64-4NODE-LMHEADTP-NOCP-VPKL.yaml`.
- AMDAHL-044 1-node topology log confirmed
  `lm_head_tp_size=8`, `source_axis=dp`, `source_replica=1`, `mesh=(1,8)`.
  Cold full64 fit with `server_forward_backward_s=141.6945`,
  `model_forward_s=87.9340`, `backward_compute_s=51.7090`,
  `kl_compute_s=0.1708`, `valid_tokens=515`, and GPU memory peaking around
  `79-80 GiB`. Warmed 3x full64 mean:
  `server_forward_backward_s=4.7197`, `api_wall_s=5.0063`,
  `model_forward_s=1.2169`, `backward_compute_s=2.3254`,
  `kl_compute_s=0.0289`, `valid_tokens=515`.
- AMDAHL-044 artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-1node-full64-warmed-serveronly-3x-20260614T2121.jsonl`.
- AMDAHL-045 4-node trainer-only stack used only slots trainer pods with
  explicit `nodeSelector: node-group=nccl`: head on
  `research-common-h100-110`, workers on `099`, `088`, and `081`. No slots
  sampler/teacher/dispatch controls were changed, and `er-opd-q36-35b-sci` was
  not touched.
- AMDAHL-045 topology log confirmed `lm_head_tp_size=8`, `source_axis=dp`,
  `source_replica=4`, `mesh=(4,8)`. Cold full64 fit with
  `server_forward_backward_s=81.9382`, `api_wall_s=82.2425`,
  `model_forward_s=51.9088`, `backward_compute_s=27.8803`,
  `kl_compute_s=0.2408`, `valid_tokens=515`, and loss `1.8479248`.
- AMDAHL-045 warmed 3x full64 replay:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-4node-full64-warmed-serveronly-3x-20260614T2127.jsonl`,
  mean `server_forward_backward_s=4.1209`, `api_wall_s=4.3762`,
  `model_forward_s=0.8699`, `backward_compute_s=1.9017`,
  `kl_compute_s=0.0121`, `valid_tokens=515`.
- AMDAHL-045 cold artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-4node-full64-first-serveronly-1x-20260614T2125.jsonl`.
- Cleanup completed after AMDAHL-045: killed the local port-forward, stopped
  trainer control with `--remove-run`, deleted trainer-head/workers, verified no
  slots trainer pods remained, and left the science stack untouched.

This is **not promoted**. The branch/local follow-up now proves the
lm-head-only-TP VP-KL path can execute real trainer-server OPD replays, that
prep64 can fit, and that the no-CP variant removes the Ulysses8/backward tax
from raw replay wall time. However, AMDAHL-044/045 report losses
`1.8495717` / `1.8479248` while AMDAHL-042/043 VP-KL reported `1.8686199` on
the same replay. A focused two-group DP-sourced VP-OPD unit test passes, and the
only YAML difference between AMDAHL-042/043 and AMDAHL-044/045 is topology
(`Ulysses8 + DP1/4` vs no-CP `DP8/32`). Treat the CP-vs-no-CP loss comparison
as a topology-parity warning, not yet proof of a VP-KL arithmetic bug. AMDAHL-046
was the narrower same-topology gate:
`experiments/opd_profile/autoresearch/candidates/AMDAHL-046-OPRD-PREP64-1NODE-NOCP-STREAMING-VS-VPKL.yaml`.
It used the existing no-CP/full-lm-head trainer config
`/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch.yaml`
to compare pure streaming-KL `--limit-data 8` against pure no-CP VP-KL
`--limit-data 8` before changing engine code further. That gate ran and
**failed loss equivalence**:

- AMDAHL-046 streaming full-lm-head no-CP limit8:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-limit8-serveronly-1x-20260614T2155.jsonl`,
  loss `0.6243904829`, valid tokens `397`,
  `server_forward_backward_s=80.2944`, `model_forward_s=50.8974`,
  `backward_compute_s=28.2316`, `kl_compute_s=0.0533`.
- AMDAHL-044 no-CP lm-head-TP VP-KL limit8:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-limit8-serveronly-1x-20260614T2158.jsonl`,
  loss `0.6256709099`, valid tokens `397`,
  `server_forward_backward_s=8.0331`, `model_forward_s=5.5086`,
  `backward_compute_s=0.9694`, `kl_compute_s=0.1690`.
- VP minus streaming loss is `+0.001280427` (`0.205%` relative to streaming).
  Treat this as a live correctness blocker, not acceptable drift.

Enriched follow-up (2026-06-14 22:32Z) narrowed the failure:

- The replay script now records OPD submetrics plus `loss_param_overrides` and
  `effective_loss_fn_params`. With explicit
  `opd_hidden_match_coef=0.0`, `opd_oprd_enabled=false`,
  `opd_emit_full_vocab_diagnostics=false`, and `opd_kl_backend=streaming`, the
  no-CP full-lm-head server reported the same KL-only loss as above:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-hf-hidden0-enriched-serveronly-1x-20260614T221914Z.jsonl`,
  loss `0.6243904829`, `opd_kl=0.6243904652`,
  `opd_hidden_match_loss=0`, `opd_profile_oprd_teacher_forward_s=0`.
- Swapping only `teacher_heads` from the HF snapshot to
  `/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json`
  produced the exact same streaming KL/loss:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-teacherstore-hidden0-enriched-serveronly-1x-20260614T222041Z.jsonl`.
  The teacher store also matches sampled HF `lm_head.weight` row windows
  exactly (`max_abs=0.0`), so teacher-store corruption is not the drift source.
- Turning hidden-match on with the same streaming backend gave loss
  `4.0513830` with `opd_hidden_match_loss=3.4269925`
  (`replay-streaming-nocp-teacherstore-hidden1-enriched-serveronly-1x-20260614T222050Z.jsonl`).
  The hidden-match contribution is orders of magnitude larger than the
  `+0.001280427` VP drift, so the old mismatch is not a hidden-match/config
  artifact.
- No-CP lm-head-TP VP-KL with the same teacher-store and hidden-match disabled
  still reported loss `0.6256709099`,
  `opd_kl=0.6256708486`, `opd_hidden_match_loss=0`, and OPRD timings `0`:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-hidden0-enriched-serveronly-1x-20260614T222357Z.jsonl`.
  VP hidden-match-on was additive (`4.0526638`, same hidden loss).
- Attempting `opd_kl_backend=streaming` on the lm-head-TP server failed with
  the expected mixed Tensor/DTensor error in `opd_streaming_kl.py`
  (`student_hidden_states @ student_weight[...]`), then left the server in a
  bad state for later traffic (DeepEP timeout). Do not use that probe as a
  supported path; restart the trainer after any such failure.
- DCP metadata for the step101 student checkpoint stores `model.lm_head.weight`
  as 64 contiguous row chunks of 3880 rows over global shape `[248320,2048]`.
  The 1-node no-CP lm-head-TP server uses `lm_head_tp_size=8`,
  `source_axis=dp`, `source_replica=1`, `mesh=(1,8)`. The remaining suspect is
  live student `lm_head` DTensor materialization/loading or body topology
  effects under the lm-head-TP server, not teacher store, hidden-match config,
  or standalone VP OPD arithmetic.

Dtype-fix follow-up (2026-06-14 23:24Z) supersedes the live mismatch blocker
above:

- The opt-in lm-head fingerprint dump showed the decisive difference: streaming
  no-CP full-lm-head consumed `student_weight_local.dtype=torch.bfloat16` with
  shape `[248320,2048]`, while no-CP lm-head-TP VP-KL consumed local DTensor
  shards as `torch.float32` with shape `[31040,2048]`, placements `['R','S(0)']`,
  and mesh `[1,8]`. Shard row ranges were correct
  (`[0,31040)`, `[31040,62080)`, ..., `[217280,248320)`), so this was not a
  row-order or teacher-row slicing bug.
- Fixed path: `ModelRunner._opd_student_lm_head_weight_for_loss()` now uses the
  original DTensor shard for gradient flow but casts the effective student
  weight used by VP-KL to the hidden/model compute dtype when needed. The debug
  JSONL records both raw shard and effective-for-loss fingerprints.
- Same-server hidden0 limit8 parity now passes on the current launch/config/code
  state. Streaming no-CP full-lm-head:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-lmheaddebug-hidden0-serveronly-1x-20260614T2245.jsonl`,
  loss `0.6181625724`, `opd_kl=0.6181626116`, valid tokens `397`.
  The no-debug repeat
  `replay-streaming-nocp-nodebug-hidden0-serveronly-1x-20260614T2248.jsonl`
  matched exactly, so the debug hook did not perturb loss. Pre-fix VP was
  `0.6195043921` in
  `replay-vpkl-lmheadtp-nocp-lmheaddebug-hidden0-serveronly-1x-20260614T2253.jsonl`.
  Post-fix VP was `0.6181625128` in
  `replay-vpkl-lmheadtp-nocp-dtypefix-hidden0-serveronly-1x-20260614T2305.jsonl`,
  within about `6e-8` of streaming. Do not compare these rows directly to the
  older `0.624390...` 22:32Z rows; compare only same-server rows from the same
  launch/config/code state.
- 1-node no-CP lm-head-TP VP-KL full64 all-layer OPRD replay still fits after
  the dtype fix:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-full64-warmed-serveronly-4x-20260614T2307.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=7.6398`, `model_forward_s=1.1788`,
  `backward_compute_s=2.7703`, `kl_compute_s=0.0538`,
  `oprd_teacher_forward_s=2.3955`, valid tokens `3049`, loss `0.5793913`,
  `opd_kl=0.5689359`, `opd_hidden_match_loss=0.0104553`.
- 4-node no-CP lm-head-TP VP-KL full64 all-layer OPRD replay ran trainer-only
  on nccl nodes (`head=092`, workers `110/077/089`) and completed:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-4node-full64-warmed-serveronly-4x-20260614T2321.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=5.8661`, `model_forward_s=0.9778`,
  `backward_compute_s=1.7877`, `kl_compute_s=0.0173`,
  `oprd_teacher_forward_s=1.1042`, valid tokens `3049`, loss `0.5791455`,
  `opd_kl=0.5686667`, `opd_hidden_match_loss=0.0104788`.
- Denominator audit:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
  The capture packs 64 samples into 22 rows, with
  `dispatcher_executed_tokens=107648`,
  `real_student_tokens_without_dispatch_dummy=71804`, valid tokens `3049`, and
  `student_plus_teacher_forward_tokens=458518`. Rates were
  `valid_tokens_per_gpu_s=16.24`,
  `dispatcher_executed_tokens_per_gpu_s=573.46`,
  `student_plus_teacher_forward_tokens_per_gpu_s=2442.62`,
  reconstructed logical TFLOPS/GPU `10.29`, reconstructed logical MFU
  `0.010405` (`~1.04%`), and valid-token-scaled logical MFU `0.000295`.

SGLang layer-cache follow-up (2026-06-14 23:43Z) is an engine
micro-optimization, not a science promotion:

- Replaying the older AMDAHL-031 every-4th-layer SGLang OPRD cache payload on
  the current AMDAHL-044 no-CP lm-head-TP VP engine now fits on one node. Without
  the new device-cache flag, the measured mean was
  `server_forward_backward_s=4.9560`, `model_forward_s=1.1769`,
  `backward_compute_s=2.3690`, `oprd_teacher_forward_s=0.0`,
  `oprd_layer_fetch_s=0.2847`, `kl_compute_s=0.1948`, valid tokens `515`, loss
  `1.8533004`. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-warmed-serveronly-4x-20260614T2340.jsonl`.
- Local engine change: `TeacherActivationCache.get(..., cache_device=True)` can
  keep one full activation/layer cache resident on the target device. Runner
  flags are `opd_teacher_activation_cache_device_cache`,
  `opd_teacher_hidden_cache_device_cache`, and
  `opd_teacher_layer_cache_device_cache`; defaults are false. Focused
  validation: `/home/apanda/xorl-internal/.venv/bin/python -m py_compile
  src/xorl/distillation/teacher_cache.py src/xorl/server/runner/model_runner.py`
  and `PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src
  /home/apanda/xorl-internal/.venv/bin/python -m pytest
  tests/utils/test_distillation_teacher_cache.py -q` (`15 passed`).
- With `opd_teacher_layer_cache_device_cache=true`, the same replay measured
  mean `server_forward_backward_s=4.3206`, `model_forward_s=1.0629`,
  `backward_compute_s=2.2234`, `oprd_teacher_forward_s=0.0`,
  `oprd_layer_fetch_s=0.00138`, `kl_compute_s=0.0301`, valid tokens `515`, loss
  `1.8533004`. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-layerdevcache-warmed-serveronly-4x-20260614T2343.jsonl`.
- Interpretation: the old AMDAHL-031 full-lm-head-gradient OOM is gone in the
  no-CP lm-head-TP VP engine, and layer-cache device residency removes the CPU
  layer-cache transfer from the hot path. This is still reduced-layer every4
  OPRD with only `515` valid tokens, so it is not a 10% MFU route by itself and
  does not change the promoted science config.

Valid-only layer-cache gather + chunked OPRD MSE follow-up (2026-06-15
00:13Z) is a fit/stability result, not a promotion:

- Valid-only gather plus chunked OPRD MSE stabilized the every4 replay that had
  previously produced only a partial measured row in this variant. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0009.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.7859`, `model_forward_s=1.2912`,
  `backward_compute_s=2.4126`, `oprd_teacher_forward_s=0.0`,
  `oprd_layer_fetch_s=0.000733`, `kl_compute_s=0.0367`, valid tokens `515`,
  loss `1.8533004`. This is stable, but it is slower than the 23:43Z
  device-cache-only replay (`4.3206s`), so it is not the next speed winner by
  itself.
- A derived all-40-layer synthetic layer-cache stress replay now completes on
  the same 1-node no-CP lm-head-TP VP engine. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-synthlayercache-all40-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0013.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.8691`, `model_forward_s=1.3279`,
  `backward_compute_s=2.5173`, `oprd_teacher_forward_s=0.0`,
  `oprd_layer_fetch_s=0.000758`, `kl_compute_s=0.0549`, valid tokens `3049`,
  loss `5.6481571`.
- Caveat: AMDAHL-047 is explicitly synthetic. Its rank-3 layer cache repeats
  the final hidden-state cache for every layer, so this proves the performance
  path and memory behavior only. It is not a loss/science validation artifact and
  cannot replace a real all-layer SGLang-cache capture or a K3/static gate.
- Cleanup after the 00:13Z replay was clean: `stop-trainer-control --remove-run`
  exited the trainer server (`rc=0`, `stopped_at=2026-06-15T00:12:30Z`), only
  `er-opd-q36-35b-slots-trainer-head` was deleted, and `kubectl get pods`
  showed only slots dispatch + teacher-smg for this stack.

Real all-layer SGLang cache capture + replay (2026-06-15 00:56Z) replaces the
synthetic-cache caveat for the one-node path, but is still not a promotion:

- Full64 direct recache artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-realalllayer-sglangcache-full64-20260615T004904Z.json`.
  Assets:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-realalllayer-sglangcache-full64-20260615T004904Z.hidden.safetensors`
  (`[3542,2048]`, 14M) and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-realalllayer-sglangcache-full64-20260615T004904Z.layers.safetensors`
  (`[40,3542,2048]`, 554M). SGLang PR #48 tensor-output path reported
  `teacher_prefill_forward_compute_s=8.9454`,
  `teacher_hidden_cache_write_s=0.7722`, local wall `10.001s`, `171748`
  teacher input tokens, max teacher sequence `4228`, and max kept rows `112`.
  Cache indices matched the reconstructed AMDAHL-031 row offsets. The real
  capture has `3542` cache rows, not the synthetic stress capture's `3049`,
  because the direct recache reconstructed exact prompt+answer kept-row offsets
  for the 64 samples.
- Successful replay artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-warmed-serveronly-4x-20260615T005428Z.jsonl`.
  It ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-110.cloud.together.ai`, with xorl PR #375 commit
  `706256f4` and SGLang PR #48 commit `c8552f4fa`. The effective replay
  overrides were `opd_kl_backend="vocab_parallel"`,
  `opd_emit_full_vocab_diagnostics=false`,
  `teacher_heads={"0":"/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json"}`,
  `opd_teacher_layer_cache_device_cache=true`,
  `opd_sharded_head_device_cache=true`, and `opd_oprd_layer_chunk_size=4`.
  Warmup paid the one-time teacher-store/device-cache cost
  (`server_forward_backward_s=89.0883`). Mean over 3 measured iterations:
  `server_forward_backward_s=4.9408`, `api_wall_s=5.2502`,
  `model_forward_s=1.2082`, `forward_compute_s=1.2754`,
  `backward_compute_s=2.5217`, `kl_compute_s=0.0414`,
  `loss_compute_s=0.0649`, `hidden_fetch_s=0.0092`,
  `oprd_layer_fetch_s=0.000436`, `oprd_teacher_forward_s=0.0`,
  `clear_gradients_s=0.3842`, valid tokens `515`, loss `2.3550419`,
  `opd_kl=2.3337434`, and `opd_hidden_match_loss=0.0212984`.
- Operationally important failed probes before the successful replay: forcing
  VP-KL while leaving `opd_emit_full_vocab_diagnostics=true` fails because the
  VP backend intentionally rejects full-vocab diagnostics, and forcing VP-KL
  while leaving `teacher_heads` as the HF snapshot path fails because VP requires
  an OPD teacher-store entry. The successful run disabled full-vocab diagnostics
  and pointed `teacher_heads[0]` at the prepared teacher-store manifest.
- Cleanup after the real-cache run was clean: local port-forward was killed,
  `stop-trainer-control --remove-run` was issued, only
  `er-opd-q36-35b-slots-trainer-head` was deleted, and `kubectl get pods`
  showed only slots dispatch + teacher-smg for this stack. The separate
  `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-048 2-node scale-up retry + layer-chunk sweep (2026-06-15 01:27Z) is
still not a promotion:

- Added a 2-node replay candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-048-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml`.
  The config keeps the AMDAHL-045 no-CP/lm-head-TP body and changes
  `data_parallel_shard_size` to `16` for a 2-node trainer-server replay. Render
  and server-side dry-run passed with explicit checkout overrides
  (`OPD_XORL_REPO=/home/apanda/xorl-opd-kl-fused`,
  `OPD_XORL_CLIENT_REPO=/home/apanda/xorl-opd-prefill`,
  `OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal`).
- The first AMDAHL-048 launch did not produce a 2-node throughput row. At launch
  time worker-1 scheduled on `research-common-h100-077`, but head stayed Pending
  because `research-common-h100-110` was claimed by
  `er-opd-q36-mtp-perf-replay-trainer-head` immediately after the capacity check.
  I stopped trainer control with `--remove-run`, deleted the temporary head/worker
  pods, killed the local `kubectl wait`, and left slots at dispatch + teacher-smg
  only.
- I then ran a one-node real-cache `opd_oprd_layer_chunk_size` sweep on
  `research-common-h100-077` using the same capture and the AMDAHL-044
  trainer-server body. Chunk `4` artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk4-warmed-serveronly-4x-20260615T010816Z.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.6591`, `api_wall_s=4.9797`,
  `model_forward_s=1.1515`, `forward_compute_s=1.2140`,
  `backward_compute_s=2.3476`, `kl_compute_s=0.0353`,
  `loss_compute_s=0.0596`, `oprd_layer_fetch_s=0.000343`,
  `clear_gradients_s=0.4339`, loss `2.3550419`.
- Chunk `8` artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk8-warmed-serveronly-4x-20260615T010816Z.jsonl`.
  Mean over 3 measured iterations: `server_forward_backward_s=4.9491`,
  `api_wall_s=5.2323`, `backward_compute_s=2.6161`,
  `oprd_layer_fetch_s=0.000351`, loss `2.3550419`. This is slower than chunk
  `4`.
- Chunk `16` artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk16-warmed-serveronly-4x-20260615T010816Z.jsonl`.
  It wrote only one measured row after warmup (`server_forward_backward_s=4.6654`,
  `api_wall_s=4.8758`, loss `2.3550419`) and then OOMed on rank 2 in FSDP
  pre-backward all-gather while trying to allocate `970 MiB`. Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T010520Z-serveronly-configAMDAHL-044-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-er-opd-q36-35b-slots-trainer-head/server.log`.
  The OOM leaves chunk `16` rejected for this workload.
- Cleanup after the sweep completed: local replay loop killed, trainer control
  stopped with `--remove-run`, `er-opd-q36-35b-slots-trainer-head` deleted, local
  port-forward killed, and `kubectl get pods` showed only slots dispatch +
  teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack was not
  touched.
- The AMDAHL-048 retry completed after kube auth/capacity recovered. The first
  head pin to `research-common-h100-092` hit kubelet admission
  `Requested: 8, Available: 4`, so the accepted run used head
  `research-common-h100-077` and worker-1 `research-common-h100-110`; no slots
  sampler/teacher/dispatch roles were touched. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-chunk4-warmed-serveronly-4x-20260615T012513Z.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.4441`, `api_wall_s=4.7269`,
  `model_forward_s=1.1004`, `forward_compute_s=1.1472`,
  `backward_compute_s=2.0665`, `kl_compute_s=0.0256`,
  `loss_compute_s=0.0475`, `oprd_layer_fetch_s=0.000236`,
  `clear_gradients_s=0.6290`, `oprd_teacher_forward_s=0.0`, valid tokens `515`,
  loss `2.3550419`. This is only about `4.6%` faster than the one-node chunk4
  row (`4.6591s`) despite doubling trainer GPUs, so it does not alter the
  underfilled/MFU conclusion.
- Cleanup after the 2-node retry completed: trainer control stopped with
  `--remove-run`, temporary trainer-head/worker-1 pods deleted, local port-forward
  killed, and `kubectl get pods` showed only slots dispatch + teacher-smg for this
  stack. The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-049 pack2304 zero-dummy screen (2026-06-15 01:38Z) is rejected for the
real-cache path:

- Added a one-node pack2304 replay candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-049-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-PACK2304.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_pack2304.yaml`.
  The only intended topology change from the current AMDAHL-044/048 body is
  `sample_packing_sequence_len: 2304`; no CP, `lm_head_tensor_parallel_size=8`,
  VP-KL, SGLang layer-cache device residency, and `opd_oprd_layer_chunk_size=4`
  are unchanged.
- The screen came from the row-shape audit
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_realcache_pack_sweep_20260615.json`.
  At the 32-GPU gate, pack2304/3072 yields `32` packed rows and `0` dummy rows,
  compared with seq4096's `22` packed rows and `10` dummy rows. Caveat: this
  real-cache capture lacks `_r3_sample_lengths`, so audit fields derived from
  "real student tokens without dispatch dummy" are zero; the packed-row, dummy-row,
  dispatcher-executed-token, and valid-fraction fields are still the useful signal.
- The one-node trainer-only replay ran only `er-opd-q36-35b-slots-trainer-head`
  on `research-common-h100-077`, using xorl PR #375 `706256f4` and SGLang PR #48
  `c8552f4fa`. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-pack2304-chunk4-warmed-serveronly-4x-20260615T013525Z.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=5.5459`, `api_wall_s=5.8138`,
  `model_forward_s=1.4871`, `forward_compute_s=1.5736`,
  `backward_compute_s=2.8861`, `kl_compute_s=0.0449`,
  `loss_compute_s=0.0831`, `oprd_layer_fetch_s=0.000436`,
  `clear_gradients_s=0.4280`, `oprd_teacher_forward_s=0.0`, valid tokens `515`,
  loss `2.3666090`.
- This is slower than the one-node pack4096/chunk4 row (`4.6591s`) and slower
  than the AMDAHL-048 2-node pack4096 row (`4.4441s`). The zero-dummy row shape
  does not compensate for the shorter pack shape's worse forward/backward profile
  on this real-cache workload. Do not retry or promote pack2304 unless a later
  full-OPD prepare path changes the generated sample distribution.
- Cleanup after AMDAHL-049 completed: trainer control stopped with
  `--remove-run`, temporary trainer-head pod deleted, local port-forward killed,
  and `kubectl get pods` showed only slots dispatch + teacher-smg for this stack.
  The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-050 allocator-flush/no-defrag screen (2026-06-15 01:51Z) is rejected for
the real-cache path:

- Added a one-node tooling candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-050-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-NODEFRAG.yaml`
  that reuses the current pack4096/chunk4 no-CP lm-head-TP VP-KL trainer config
  and changes only replay loss params:
  `forward_backward_defrag=false` and
  `profile_clear_gradients_empty_cache=false`.
- The local xorl engine change under test is default-preserving: `forward_backward`
  still runs the old `gc.collect()` / `torch.cuda.empty_cache()` defrag path unless
  the loss param or `XORL_FORWARD_BACKWARD_DEFRAG=0` disables it, and the
  profile-clear cache flush remains enabled unless
  `profile_clear_gradients_empty_cache=false` is passed. Validation before launch:
  xorl `git diff --check`, top-level `git diff --check`, and
  `py_compile` for `/home/apanda/xorl-opd-kl-fused/src/xorl/server/runner/model_runner.py`.
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-089`, using xorl PR #375 base `706256f4` plus the local
  default-preserving allocator-flush controls, SGLang PR #48 `c8552f4fa`, and the
  same real all-layer SGLang-cache capture as AMDAHL-048/049. Partial artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-nodefrag-chunk4-warmed-serveronly-4x-20260615T014448Z.jsonl`.
  It wrote one very slow warmup (`server_forward_backward_s=139.2982`) and two
  measured rows before failure. Mean over the two completed measured rows:
  `server_forward_backward_s=4.8437`, `api_wall_s=5.0955`,
  `model_forward_s=2.2231`, `forward_compute_s=2.3064`,
  `backward_compute_s=2.4803`, `kl_compute_s=0.0452`,
  `loss_compute_s=0.0803`, `oprd_layer_fetch_s=0.000379`,
  `clear_gradients_s=0.0039`, `oprd_teacher_forward_s=0.0`, valid tokens `515`,
  loss `2.3550419`.
- The third measured request OOMed in FSDP pre-backward all-gather on ranks 3/4
  trying to allocate `970 MiB`; rank 3 had only `515.62 MiB` free and rank 4 had
  `593.62 MiB` free, with about `3.9 GiB` PyTorch reserved-but-unallocated on
  each. `nvidia-smi` at failure showed the trainer ranks around `78.7-81.0 GiB`
  used. So disabling allocator defrag is unsafe for the current all-layer
  real-cache memory shape even though it nearly eliminates the replay
  `clear_gradients_s` timer. Keep the default defrag path enabled; do not retry
  or promote the no-defrag replay knobs unless the memory envelope is changed.
- Cleanup after AMDAHL-050 completed: the hung replay client and local port-forward
  were killed, trainer control was stopped with `--remove-run`, the temporary
  trainer-head pod was deleted, and `kubectl get pods` showed only slots dispatch
  + teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack was not
  touched.

AMDAHL-051 manual FSDP prefetch screen (2026-06-15 02:03Z) is neutral/rejected
for the real-cache path:

- Added a one-node forward-prefetch candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-051-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-PREFETCH.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_prefetch_lmheadtp_nocp.yaml`.
  The only intended topology/scheduling change from the current
  AMDAHL-044/047 pack4096 chunk4 body is `enable_forward_prefetch: true`; no CP,
  `lm_head_tensor_parallel_size=8`, VP-KL, real all-layer SGLang cache, and
  `opd_oprd_layer_chunk_size=4` are unchanged.
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`, using xorl PR #375 base `706256f4` plus the local
  default-preserving allocator-flush controls, SGLang PR #48 `c8552f4fa`, and the
  same real all-layer SGLang-cache capture as AMDAHL-048/049/050. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-prefetch-chunk4-warmed-serveronly-4x-20260615T020000Z.jsonl`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.6506`, `api_wall_s=4.8910`,
  `model_forward_s=1.1424`, `forward_compute_s=1.2069`,
  `backward_compute_s=2.3933`, `kl_compute_s=0.0360`,
  `loss_compute_s=0.0629`, `oprd_layer_fetch_s=0.000336`,
  `clear_gradients_s=0.3838`, `oprd_teacher_forward_s=0.0`, valid tokens `515`,
  loss `2.3562496`.
- This is not a meaningful speed win over the one-node pack4096/chunk4 baseline
  (`server_forward_backward_s=4.6591`, loss `2.3550419`), and backward compute
  slightly regressed (`2.3933s` vs `2.3476s`). Because the timing delta is within
  run-to-run noise and the reported loss shifts by about `0.0012`, do not promote
  forward-prefetch for the no-CP/lm-head-TP real-cache path without a later
  static/K3 pass plus a stronger same-workload 4-node speed result.
- Cleanup after AMDAHL-051 completed: trainer control stopped with
  `--remove-run`, temporary trainer-head pod deleted, local port-forward killed,
  and `kubectl get pods` showed only slots dispatch + teacher-smg for this stack.
  The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-052 repeat-data=2 fatter-call screen (2026-06-15 02:12Z) is rejected for
the real-cache path:

- Added throughput-only candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-052-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-REPEAT2.yaml`
  using the existing one-node no-prefetch no-CP/lm-head-TP trainer config. The
  replay used the same real all-layer SGLang-cache capture as AMDAHL-048/049/050/051,
  the same VP-KL/layer-cache/chunk4 loss-param overrides, and only changed the
  replay harness input to `--repeat-data 2`.
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`, using xorl PR #375 base `706256f4` plus the local
  default-preserving allocator-flush controls and SGLang PR #48 `c8552f4fa`.
  Startup run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T020711Z-serveronly-configAMDAHL-052-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-REPEAT2-er-opd-q36-35b-slots-trainer-head`.
  Empty output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-repeat2-chunk4-warmed-serveronly-4x-20260615T021000Z.jsonl`.
- Server evidence: `[SequentialPacker] Packed 128 samples into 43 batches (80.7%
  utilization, 142080 tokens)`, then rank 5 OOMed in FSDP pre-backward all-gather
  while allocating `970 MiB` with only about `798 MiB` free. No warmup or measured
  replay rows were written.
- Conclusion: repeated captured data is not a viable one-node fix for the current
  real-cache memory envelope. Do not spend a 4-node gate or K3/static pass on
  AMDAHL-052. If larger generated batches are revisited, they need a memory change
  or topology change first, not simple repeat-data on the current one-node path.
- Cleanup after AMDAHL-052 completed: the hung local replay client and
  port-forward were killed, trainer control was stopped with `--remove-run`, the
  temporary trainer-head pod was deleted, and `kubectl get pods` showed only slots
  dispatch + teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack
  was not touched.

AMDAHL-053 CPU-GC split screen (2026-06-15 02:22Z) is rejected for the
real-cache path:

- Added default-preserving xorl runner cleanup controls that split CPU
  `gc.collect()` from CUDA `torch.cuda.empty_cache()`:
  `forward_backward_gc_collect`, `forward_backward_empty_cache`, and
  `profile_clear_gradients_gc_collect`. Defaults preserve the old behavior.
  Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-053-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-GCSKIP.yaml`.
  The replay kept the current one-node no-prefetch no-CP/lm-head-TP VP-KL config,
  real all-layer SGLang cache, chunk `4`, and only passed loss params
  `forward_backward_gc_collect=false` and
  `profile_clear_gradients_gc_collect=false` to skip CPU GC.
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-089`, using xorl PR #375 base `706256f4` plus the local
  default-preserving cleanup split, SGLang PR #48 `c8552f4fa`, and the same real
  all-layer SGLang-cache capture as AMDAHL-048/049/050/051/052. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-gcskip-chunk4-warmed-serveronly-4x-20260615T021900Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T021555Z-serveronly-configAMDAHL-053-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-GCSKIP-er-opd-q36-35b-slots-trainer-head/server.log`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=5.4222`, `api_wall_s=5.6626`,
  `model_forward_s=2.3896`, `backward_compute_s=2.7226`,
  `kl_compute_s=0.0604`, `loss_compute_s=0.0922`,
  `oprd_layer_fetch_s=0.000378`, `clear_gradients_s=0.1783`, valid tokens `515`,
  loss `2.3550419`.
- The same-workload one-node chunk4 baseline was
  `server_forward_backward_s=4.6591`, `api_wall_s=4.9797`,
  `model_forward_s=1.1515`, `backward_compute_s=2.3476`,
  `clear_gradients_s=0.4339`, loss `2.3550419`. Skipping CPU GC makes the
  explicit clear-gradient timer smaller but the total request is slower, mostly
  through model forward/backward. Keep default CPU GC enabled; do not promote
  AMDAHL-053 or spend 4-node/static/K3 gates on this variant.
- Cleanup after AMDAHL-053 completed: the local port-forward was killed, trainer
  control was stopped with `--remove-run`, the temporary trainer-head pod was
  deleted, and `kubectl get pods` showed only slots dispatch + teacher-smg for
  this stack. The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-054 DeepEP SMS24 screen (2026-06-15 02:34Z) is rejected for the
real-cache path:

- Added a one-node DeepEP reserved-SM candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-054-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEEPEP24.yaml`
  plus infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep24_noprefetch_lmheadtp_nocp.yaml`.
  The only intended trainer-topology change versus the chunk4 baseline config is
  `deepep_num_sms: 24` instead of `36`; no sampler/teacher/dispatch roles were
  restamped.
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`, using xorl PR #375 base `706256f4` plus local
  default-preserving cleanup controls, SGLang PR #48 `c8552f4fa`, and the same
  real all-layer SGLang-cache capture as AMDAHL-048/049/050/051/052/053.
  Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-deepep24-chunk4-warmed-serveronly-4x-20260615T023000Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T022624Z-serveronly-configAMDAHL-054-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEEPEP24-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T022624Z-run.log`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.7599`, `api_wall_s=4.9905`,
  `model_forward_s=1.2075`, `backward_compute_s=2.4477`,
  `kl_compute_s=0.0443`, `loss_compute_s=0.0673`,
  `oprd_layer_fetch_s=0.000360`, `clear_gradients_s=0.3433`, valid tokens `515`,
  loss `2.3569047`.
- The same-workload one-node SMS36 chunk4 baseline was
  `server_forward_backward_s=4.6591`, `api_wall_s=4.9797`,
  `model_forward_s=1.1515`, `backward_compute_s=2.3476`,
  `clear_gradients_s=0.4339`, loss `2.3550419`. SMS24 reduces the explicit
  clear-gradient timer but worsens model forward/backward and total server wall
  by `2.16%`, with a small loss shift. Keep `deepep_num_sms=36`; do not promote
  AMDAHL-054 or spend 4-node/static/K3 gates on this variant.
- Cleanup after AMDAHL-054 completed: the local port-forward was killed, trainer
  control was stopped with `--remove-run`, the temporary trainer-head pod was
  deleted, and `kubectl get pods` showed only slots dispatch + teacher-smg for
  this stack. The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-055 MoE BF16-a2a reduce screen (2026-06-15 02:42Z) is rejected at the
launch gate for the one-node real-cache path:

- Added a one-node MoE expert-gradient reduce candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-055-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-MOEBF16A2A.yaml`
  plus infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_moebf16a2a.yaml`.
  The intended change versus the chunk4/SMS36 baseline was only
  `moe_grad_reduce_mode: bf16_a2a_fp32_sum`, using the existing engine hook that
  stochastic-rounds expert gradients to BF16 for transport and sums locally in
  FP32.
- Preflight passed before launch: candidate/config YAML parsed, client and infra
  `git diff --check` passed, distributed BF16-a2a reduce tests passed
  (`tests/distributed/test_torch_parallelize_policies.py::test_resolve_fsdp_reduce_dtype`,
  `tests/distributed/test_bf16_a2a_reduce.py`,
  `tests/distributed/test_bf16_a2a_fsdp_hook.py`: `4 passed`), and
  `tests/server/test_server_arguments.py::test_load_server_arguments_preserves_runner_compatibility_fields`
  passed. A separate attempt to include
  `tests/test_arguments.py::test_parse_args_accepts_fsdp_reduce_dtype` failed in
  this venv during import with missing `triton._C.libtriton.linear_layout`; this
  is an environment issue and was not used as an AMDAHL-055 gate.
- The trainer startup used only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`. Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T023950Z-run.log`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T023950Z-serveronly-configAMDAHL-055-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-MOEBF16A2A-er-opd-q36-35b-slots-trainer-head/server.log`.
  No replay JSONL was written. All ranks failed during engine init with:
  `ValueError: moe_grad_reduce_mode='bf16_a2a_fp32_sum' requires FSDP mp_policy.reduce_dtype=torch.float32, got torch.bfloat16`.
- Interpretation: this hook is not a valid one-node screen for the current
  topology. On the one-node `expert_parallel_size=8`, `data_parallel_shard_size=8`
  path, expert FSDP is effectively singleton for the relevant expert units and
  the engine uses BF16 reduce policy already. The BF16-a2a hook may still be a
  multi-node communication idea, but it needs either a topology where expert FSDP
  has real reduce-scatter groups or an engine change that makes singleton groups
  no-op cleanly. Do not spend another one-node replay on AMDAHL-055 as written.
- Cleanup after AMDAHL-055 completed: the local health poll/port-forward were
  gone, trainer control was stopped with `--remove-run`, the temporary
  trainer-head pod was deleted, and `kubectl get pods` showed only slots dispatch
  + teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack was not
  touched.

AMDAHL-056 BF16 FSDP reduce screen (2026-06-15 02:55Z) is rejected for the
real-cache path:

- Added a one-node FSDP reduce-dtype candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-056-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-FSDPBF16REDUCE.yaml`
  plus infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_fsdpbf16reduce.yaml`.
  The intended trainer change versus the chunk4/SMS36 baseline was only
  `fsdp_reduce_dtype: bf16`.
- Preflight passed before launch: candidate/config YAML parsed, client and infra
  `git diff --check` passed, and the reduce-dtype/server-argument selectors
  passed:
  `tests/distributed/test_torch_parallelize_policies.py::test_resolve_fsdp_reduce_dtype`
  plus
  `tests/server/test_server_arguments.py::test_load_server_arguments_preserves_runner_compatibility_fields`
  (`2 passed`).
- The replay ran only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`, using xorl PR #375 base `706256f4`, SGLang PR #48
  `c8552f4fa`, and the same real all-layer SGLang-cache capture as
  AMDAHL-048/049/050/051/052/053/054/055. Artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-fsdpbf16reduce-chunk4-warmed-serveronly-4x-20260615T025041Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T024526Z-serveronly-configAMDAHL-056-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-FSDPBF16REDUCE-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T024526Z-run.log`.
  Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.8815`, `api_wall_s=5.1059`,
  `model_forward_s=1.2253`, `backward_compute_s=2.3802`,
  `kl_compute_s=0.0440`, `loss_compute_s=0.0709`,
  `oprd_layer_fetch_s=0.000417`, `clear_gradients_s=0.6332`, valid tokens `515`,
  loss `2.3550419`.
- The same-workload chunk4/SMS36 baseline was
  `server_forward_backward_s=4.6591`, `api_wall_s=4.9797`,
  `model_forward_s=1.1515`, `backward_compute_s=2.3476`,
  `clear_gradients_s=0.4339`, loss `2.3550419`. BF16 FSDP reduce preserves the
  loss but regresses server wall by `4.77%` and API wall by `2.53%`; do not
  promote AMDAHL-056 or spend 4-node/static/K3 gates on this variant.
- Cleanup after AMDAHL-056 completed: the local port-forward was killed, trainer
  control was stopped with `--remove-run`, the temporary trainer-head pod was
  deleted, and `kubectl get pods` showed only slots dispatch + teacher-smg for
  this stack. The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-057 alltoall dispatch screen (2026-06-15 03:10Z) is rejected for the
real-cache path:

- Added a one-node expert-dispatch candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-057-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-ALLTOALL.yaml`
  plus infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_alltoall_noprefetch_lmheadtp_nocp.yaml`.
  The intended trainer change versus the chunk4/SMS36 baseline was only
  `ep_dispatch: alltoall` instead of DeepEP.
- Preflight passed before launch: candidate/config YAML parsed, client and infra
  `git diff --check` passed, the server-argument selector passed
  (`1 passed`), and the rendered trainer-head control pointed at
  `/home/apanda/xorl-opd-kl-fused` and `/home/apanda/xorl-opd-prefill`.
- The replay used only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-089`, using xorl PR #375 base `706256f4`, SGLang PR #48
  `c8552f4fa`, and the same real all-layer SGLang-cache capture as
  AMDAHL-048/049/050/051/052/053/054/055/056. Empty output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-alltoall-chunk4-warmed-serveronly-4x-20260615T030520Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T030227Z-serveronly-configAMDAHL-057-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-ALLTOALL-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T030227Z-run.log`.
- Evidence: the server packed 64 samples into 22 batches (78.8% utilization,
  71040 tokens), then ranks 2 and 6 failed during warmup in FSDP pre-backward
  all-gather at `local_loss_sum.backward()` while allocating `970 MiB`. Rank 2
  had only `445.62 MiB` free; rank 6 had only `325.62 MiB` free. No warmup or
  measured replay rows were written.
- Interpretation: swapping DeepEP for alltoall does not solve the current
  one-node memory/throughput envelope. It appears worse on headroom than the
  DeepEP/SMS36 baseline, which fits and measures `4.6591s`. Do not spend
  4-node/static/K3 gates on AMDAHL-057 as written.
- Cleanup after AMDAHL-057 completed: the local replay client terminated, the
  local port-forward was killed, trainer control was stopped, the temporary
  trainer-head pod was deleted, and `kubectl get pods` showed only slots
  dispatch + teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack
  was not touched.

AMDAHL-058 deferred loss-report reduce screen (2026-06-15 03:22Z) is rejected
for the real-cache path:

- Added a one-node engine reporting candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-058-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEFERLOSSREDUCE.yaml`.
  The trainer config remained the current DeepEP/SMS36 baseline
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml`.
  The intended engine-only change was to accumulate detached fp32 scalar loss
  reports across packed microbatches and all-reduce once per `forward_backward`
  call. Gradients and optimizer state were unchanged by the experiment.
- Preflight passed before launch: engine `py_compile` passed, client/engine
  `git diff --check` passed, candidate YAML parsed, the server-argument selector
  passed (`1 passed`), and the model-runner layer-cache tests passed with pytest
  plugin autoload disabled (`2 passed`) to avoid a local Torch `_inductor_test`
  duplicate-registration import issue.
- The replay used only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-092`, using xorl PR #375 base `706256f4`, SGLang PR #48
  `c8552f4fa`, and the same real all-layer SGLang-cache capture as
  AMDAHL-048/049/050/051/052/053/054/055/056/057. Output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-deferlossreduce-chunk4-warmed-serveronly-4x-20260615T031650Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T031508Z-serveronly-configAMDAHL-058-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEFERLOSSREDUCE-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T031508Z-run.log`.
- Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.803969`, `api_wall_s=5.006521`,
  `model_forward_s=1.195422`, `forward_compute_s=1.267063`,
  `backward_compute_s=2.419367`, `loss_compute_s=0.068802`,
  `clear_gradients_s=0.500089`, valid tokens `515`, loss `2.355041742`.
- The same-workload chunk4/SMS36 baseline was
  `server_forward_backward_s=4.6591`, `api_wall_s=4.9797`,
  `model_forward_s=1.1515`, `backward_compute_s=2.3476`,
  `clear_gradients_s=0.4339`, loss `2.3550419`. AMDAHL-058 is `3.109%` slower
  on server wall and `0.539%` slower on API wall with no loss change. Repeated
  detached scalar reporting all-reduces are not the live bottleneck in this path.
- The local engine reporting patch was reverted/not promoted after measurement.
  Cleanup after AMDAHL-058 completed: the local port-forward was killed, trainer
  control was stopped, the temporary trainer-head pod was deleted, and
  `kubectl get pods` showed only slots dispatch + teacher-smg for this stack. The
  separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-059 EP4 DeepEP topology screen (2026-06-15 03:39Z) is rejected for the
real-cache path:

- Added one-node topology candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-059-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-EP4DEEPEP.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_ep4.yaml`.
  The workload stayed the current pack4096/chunk4 no-CP lm-head-TP VP-KL
  real-cache replay; only the expert mesh changed from EP8 x ep_fsdp1 to EP4 x
  ep_fsdp2, while keeping DeepEP/SMS36.
- First launch reached model/mesh setup (`DeviceMesh((ep=4, ep_fsdp=2))` and
  DP-sourced lm-head TP), then failed before replay because the DCP checkpoint's
  optimizer state was shaped for the baseline EP8 local expert slice
  (`saved [32,2048,1024]`, EP4 current `[64,2048,1024]`). Evidence:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T032818Z-serveronly-configAMDAHL-059-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-EP4DEEPEP-er-opd-q36-35b-slots-trainer-head/server.log`.
- Added default-true `load_checkpoint_optimizer` plumbing in the engine arguments,
  server arguments, initial server checkpoint load, and trainer checkpoint load.
  The EP4 retry config alone sets `load_checkpoint_optimizer:false`; normal
  science/training configs still default true. Preflight before retry: `py_compile`
  passed, client/infra/engine `git diff --check` passed, candidate/config YAML
  parsed, checkpoint runner tests passed (`3 passed` with pytest plugin autoload
  disabled), and server-argument selectors passed (`2 passed` with pytest plugin
  autoload disabled).
- Retry replay used only `er-opd-q36-35b-slots-trainer-head` on
  `research-common-h100-077`, xorl PR #375 base `706256f4`, SGLang PR #48
  `c8552f4fa`, and the same real all-layer SGLang-cache capture as
  AMDAHL-048/049/050/051/052/053/054/055/056/057/058. Output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-ep4deepep-chunk4-warmed-serveronly-4x-20260615T033549Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T033344Z-serveronly-configAMDAHL-059-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-EP4DEEPEP-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T033344Z-run.log`.
- Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.598297`, `api_wall_s=4.849234`,
  `model_forward_s=1.071934`, `forward_compute_s=1.142842`,
  `backward_compute_s=2.187091`, `loss_compute_s=0.068660`,
  `clear_gradients_s=0.704342`, valid tokens `515`, loss `2.373135030`,
  `opd_kl=2.351824299`, `opd_hidden_match_loss=0.021310837`.
- The same-workload chunk4/SMS36 baseline was
  `server_forward_backward_s=4.659086`, `api_wall_s=4.979698`,
  `model_forward_s=1.151450`, `backward_compute_s=2.347599`,
  `clear_gradients_s=0.433852`, loss `2.355041921`, `opd_kl=2.333743375`.
  AMDAHL-059 is `1.305%` faster on server wall and `2.620%` faster on API wall,
  but shifts total loss by `+0.018093` and KL by `+0.018081`. The loss drift
  fails the promotion gate; do not promote EP4 or spend 4-node/static/K3 gates on
  it without first explaining and eliminating that drift.
- Cleanup after AMDAHL-059 completed: local port-forward/replay helpers were
  gone, trainer control was stopped with `--remove-run`, the temporary
  trainer-head pod was deleted, and `kubectl get pods` showed only slots dispatch
  + teacher-smg for this stack. The separate `er-opd-q36-35b-sci` stack was not
  touched.

AMDAHL-060 two-node `reshard_after_forward:false` screen (2026-06-15 03:55Z) is
rejected for the real-cache path:

- Added two-node topology candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-060-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-RESHARDOFF.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_noprefetch_lmheadtp_nocp_reshardoff.yaml`.
  This keeps the AMDAHL-048 2-node no-CP/lm-head-TP/DeepEP/SMS36 workload and
  changes only `reshard_after_forward:false`.
- Operational note: the generator emits no trainer node selector, but the cluster
  admission path defaulted unspecific trainer pods to `node-group=default`. The
  AMDAHL-060 launch therefore filtered the rendered manifest to just
  `trainer-head`, `trainer-worker-1`, and the trainer-head service, then added
  explicit `nodeSelector: {node-group: nccl, node-pool: compute}` for the two
  trainer pods. The accepted run scheduled head on
  `research-common-h100-110` and worker-1 on `research-common-h100-092`. No
  sampler/teacher/dispatch roles were restamped.
- A first replay client used bad shell quoting for the `teacher_heads` JSON and
  was killed after it hung post-request; it wrote an empty artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-reshardoff-chunk4-warmed-serveronly-4x-20260615T034923Z.jsonl`.
  Ignore that file. The corrected replay passed
  `TEACHER_HEADS_JSON='{"0":"/shared/apanda/opd_teacher_stores/qwen3_6_35b_a3b_lm_head_shard32768/manifest.json"}'`
  through the pod environment so `teacher_heads` parsed as a JSON dict.
- Corrected output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-reshardoff-chunk4-warmed-serveronly-4x-20260615T035332Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T034833Z-serveronly-configAMDAHL-060-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-RESHARDOFF-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control logs:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T034833Z-run.log`,
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-worker-1/logs/20260615T034834Z-run.log`.
- Server log evidence confirms world size 16, `DeviceMesh((dp_shard=16))`,
  `DeviceMesh((ep=8, ep_fsdp=2))`, DP-sourced lm-head TP, manual FSDP prefetch
  disabled, and normal optimizer state load (`load_optimizer=True`).
- Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.443982`, `api_wall_s=4.643382`,
  `model_forward_s=1.249282`, `forward_compute_s=1.297781`,
  `backward_compute_s=1.738238`, `kl_compute_s=0.025777`,
  `loss_compute_s=0.048206`, `clear_gradients_s=0.762611`,
  `oprd_layer_fetch_s=0.000239`, valid tokens `515`, loss `2.355553985`,
  `opd_kl=2.334234441`, `opd_hidden_match_loss=0.021319744`.
- AMDAHL-048 2-node baseline was
  `server_forward_backward_s=4.444137`, `api_wall_s=4.726916`,
  `model_forward_s=1.100357`, `backward_compute_s=2.066451`,
  `clear_gradients_s=0.629015`, loss `2.355041921`,
  `opd_kl=2.333743553`, `opd_hidden_match_loss=0.021298385`.
  AMDAHL-060 is only `0.003%` faster on server wall and `1.767%` faster on API
  wall, while moving time from backward (-15.9%) into model forward (+13.5%) and
  clear-gradients (+21.2%). The same-capture loss/KL shift is small
  (`+0.000512` / `+0.000491`) but the wall-time result is not meaningful.
  Do not promote `reshard_after_forward:false` or spend 4-node/static/K3 gates on
  AMDAHL-060.
- Cleanup after AMDAHL-060 completed: trainer control stopped with
  `--remove-run`, child processes reached `stopped_at`, temporary trainer-head
  and trainer-worker-1 pods were deleted, local replay/port-forward helpers were
  gone, and `kubectl get pods` showed only slots dispatch + teacher-smg for this
  stack. The separate `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-061 two-node `enable_forward_prefetch:true` screen (2026-06-15 04:05Z) is
rejected for promotion despite a strong replay speed win:

- Added two-node topology candidate
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-061-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-PREFETCH.yaml`
  and infra config
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_prefetch_lmheadtp_nocp.yaml`.
  This keeps the AMDAHL-048 2-node no-CP/lm-head-TP/DeepEP/SMS36 workload and
  changes only `enable_forward_prefetch:true`.
- Operational note: the launch used the same filtered trainer-only manifest path
  as AMDAHL-060 because unspecific trainer pods can land on `node-group=default`.
  The accepted run applied only the trainer-head service/pod and
  trainer-worker-1 pod, each with explicit `nodeSelector: {node-group: nccl,
  node-pool: compute}`. It scheduled head on `research-common-h100-077` and
  worker-1 on `research-common-h100-089`. No sampler/teacher/dispatch roles were
  restamped.
- Output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-prefetch-chunk4-warmed-serveronly-4x-20260615T040241Z.jsonl`.
  Server log:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T040018Z-serveronly-configAMDAHL-061-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-PREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
  Run control logs:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T040018Z-run.log`,
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-worker-1/logs/20260615T040018Z-run.log`.
- Server log evidence confirms world size 16, `DeviceMesh((dp_shard=16))`,
  `DeviceMesh((ep=8, ep_fsdp=2))`, DP-sourced lm-head TP mesh `(2,8)`, and
  normal optimizer state load (`load_optimizer=True`).
- Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=3.677469`, `api_wall_s=3.863640`,
  `model_forward_s=0.806941`, `forward_compute_s=0.854842`,
  `backward_compute_s=1.689998`, `kl_compute_s=0.023734`,
  `loss_compute_s=0.046555`, `clear_gradients_s=0.520730`,
  `oprd_layer_fetch_s=0.000241`, valid tokens `515`, loss `2.359620273`,
  `opd_kl=2.338320834`, `opd_hidden_match_loss=0.021299212`.
- Versus AMDAHL-048, AMDAHL-061 is `17.25%` faster on server wall and `18.26%`
  faster on API wall; model forward is `26.67%` lower, backward is `18.22%`
  lower, and clear-gradients is `17.22%` lower. However, the same-capture loss/KL
  shift is `+0.004578` / `+0.004577` while hidden loss is unchanged. Do not
  promote `enable_forward_prefetch:true` or spend 4-node/static/K3 gates on
  AMDAHL-061 as-is; first explain and eliminate the KL drift.
- Cleanup after AMDAHL-061 completed: trainer control stopped with
  `--remove-run`, child processes reached `stopped_at` at `04:04:49Z` and
  `04:04:53Z`, temporary trainer-head and trainer-worker-1 pods were deleted,
  local replay/port-forward helpers were gone, and `kubectl get pods` showed
  only slots dispatch + teacher-smg for this stack. The separate
  `er-opd-q36-35b-sci` stack was not touched.

AMDAHL-062 two-node backward-only FSDP prefetch screen (2026-06-15 04:33Z)
completed and is not a promotion:

- Engine change under test: `/home/apanda/xorl-opd-kl-fused` now exposes
  `enable_backward_prefetch` as a default-preserving override. When unset, it
  follows `enable_forward_prefetch` and preserves existing behavior. The new
  candidate sets `enable_forward_prefetch:false` and
  `enable_backward_prefetch:true` to isolate backward prefetch overlap from the
  forward prefetch path that produced AMDAHL-061's KL/loss drift.
- Candidate/config:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH.yaml`
  and
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_backprefetch_lmheadtp_nocp.yaml`.
  The workload otherwise matches AMDAHL-048/061: 2-node no-CP/lm-head-TP
  VP-KL, DeepEP/SMS36, real full64 all-layer SGLang cache, layer chunk `4`, and
  the same teacher-store/device-cache replay overrides.
- Validation/preflight passed before launch was deferred:
  `py_compile` for the touched xorl engine files, targeted server-argument test
  `test_load_server_arguments_threads_backward_prefetch_override`,
  candidate/config YAML parse, render-control grep for the intended repo roots,
  `--nnodes 2`, and the new trainer config, plus server-side dry-run for only
  trainer-head service/pod and trainer-worker-1 pod with explicit
  `nodeSelector: {node-group: nccl, node-pool: compute}`.
- Local follow-up while capacity was closed: the manual prefetch scheduling loop
  is now factored into `_configure_manual_fsdp_prefetch()` and covered by
  CPU-only policy tests. A second hardening pass added
  `_coerce_optional_bool_config()` so string config values such as `"false"` or
  `"off"` cannot be interpreted as truthy Python objects and silently enable a
  disabled prefetch direction. Validation:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src /home/apanda/xorl-internal/.venv/bin/python3 -m pytest /home/apanda/xorl-opd-kl-fused/tests/distributed/test_torch_parallelize_policies.py -q`
  passed `28` tests, including bool-string parsing, forward-only, backward-only,
  both-directions, and no-op cases; the server-argument threading test still
  passed.
- Launch/placement notes: after kube auth was refreshed, capacity opened and the
  run launched on the slots stack. Admission defaulted unpinned GPU pods toward
  `node-group=default`, so the 8-GPU teacher/trainer pods were recreated with
  explicit `nodeSelector: {node-group: nccl, node-pool: compute}`. The student
  sampler pods first landed on cordoned nodes `064`/`096`; `064` failed CUDA
  initialization and `096` was also unusable for the SGLang child despite
  `nvidia-smi` visibility, so the samplers were finally hostname-pinned to
  `research-common-h100-077`. The replay itself used the trainer-server-only
  path and did not need live sampler traffic.
- Output JSONL:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-backprefetch-chunk4-warmed-serveronly-4x-20260615T043109Z.jsonl`.
  Server-only run dir:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T042714Z-serveronly-configAMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH-er-opd-q36-35b-slots-trainer-head`.
- Mean over 3 measured iterations after 1 warmup:
  `server_forward_backward_s=4.280844`, `api_wall_s=4.477853`,
  `model_forward_s=1.097722`, `backward_compute_s=1.684526`,
  `loss_compute_s=0.041398`, `clear_gradients_s=0.533312`, valid tokens `515`,
  loss `2.355041921`, `opd_kl=2.333743553`,
  `opd_hidden_match_loss=0.021298385`.
- Versus AMDAHL-048, AMDAHL-062 is `3.67%` faster on server wall and `5.27%`
  faster on API wall while matching AMDAHL-048 same-capture loss/KL/hidden loss.
  Versus AMDAHL-061, it is `16.41%` slower on server wall and `15.90%` slower on
  API wall. Conclusion: backward-only prefetch is correctness-clean but recovers
  only a small part of the rejected full-prefetch speed win, so it is not worth a
  standalone 4-node/static/K3 spend as the next 10% MFU lever.
- Cleanup after AMDAHL-062 completed: `stop-control --remove-run` was written,
  temporary slots `sglang-0`, `sglang-1`, `teacher-sglang-0`, `trainer-head`,
  and `trainer-worker-1` pods were deleted, and `kubectl get pods` showed only
  slots dispatch + teacher-smg for this stack. The separate science stack was
  not touched.

AMDAHL-063 two-node forward-only FSDP prefetch isolation screen
(2026-06-15 04:52Z) is measured and correctness-clean:

- Purpose: isolate the manual FSDP prefetch direction that produced the
  rejected AMDAHL-061 speed/KL tradeoff. This candidate keeps the same
  AMDAHL-048/061/062 workload and sets `enable_forward_prefetch:true`,
  `enable_backward_prefetch:false`.
- Candidate/config:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH.yaml`
  and
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
- Preflight passed: candidate/config YAML parse, render-control check for the
  intended repo roots and `--nnodes 2`, filtered trainer-only server-side
  dry-run with explicit `nodeSelector: {node-group: nccl, node-pool: compute}`,
  and repo `diff --check`. The underlying engine policy tests and
  server-argument threading test are the same AMDAHL-062 hardening gate listed
  above.
- Launch/placement notes: refreshed kube auth worked (`kubectl auth can-i get
  pods` and `create pods` both returned `yes`). The first retry reached API
  admission but lost the apparent `110/089` window to
  `wordle-rpl2n-sms20-0440`, so the pending pods were deleted. A follow-up
  retry launched `trainer-head` on `research-common-h100-110` and
  `trainer-worker-1` on `research-common-h100-077`.
- Artifact/server:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-chunk4-warmed-serveronly-4x-20260615T044953Z.jsonl`
  and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T044707Z-serveronly-configAMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
- Mean over `3` measured rows after `1` warmup:
  `server_forward_backward_s=4.140186`, `api_wall_s=4.335127`,
  `forward_compute_s=0.863331`, `backward_compute_s=2.069476`,
  `loss_compute_s=0.045991`, `clear_gradients_s=0.633195`, valid `515`,
  loss `2.355041921`, `opd_kl=2.333743553`, hidden loss `0.021298385`.
- Versus AMDAHL-048, AMDAHL-063 is `6.84%` faster on server wall and `8.29%`
  faster on API wall while matching the AMDAHL-048 loss/KL/hidden tuple exactly.
  Versus AMDAHL-062, it is `3.29%` faster on server wall and `3.19%` faster on
  API wall. Versus rejected AMDAHL-061, it is still `12.58%` slower on server
  wall and `12.20%` slower on API wall.
- Conclusion: forward-only manual prefetch is correctness-clean for this
  same-capture replay, so AMDAHL-061's KL/loss drift is not forward-only
  prefetch by itself. Treat AMDAHL-063 as a small diagnostic speed lever only;
  do not promote or spend static/K3 plus same-workload 4-node gates on it unless
  a larger correctness-clean throughput stack needs this knob. Cleanup complete:
  trainer controls were stopped, temporary trainer-head/worker-1 pods were
  deleted, and the slots stack was verified back to only dispatch + teacher-smg.
  No science-stack pod was touched.

AMDAHL-064 two-node repeat-data=2 fatter-call fit/throughput screen
(2026-06-15 05:02Z) fit and improved valid-token throughput, but is not
correctness-clean:

- Purpose: test whether the current 2-node no-CP/lm-head-TP VP-KL real-cache
  path can absorb a fatter 128-datum static f/b call after the one-node
  AMDAHL-052 repeat2 screen OOMed. This reuses the AMDAHL-063 loss-clean
  forward-only prefetch config and runs the same AMDAHL-047 full64 capture with
  `--repeat-data 2`.
- Candidate/config:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2.yaml`
  and
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
- Launch/placement notes: kube auth worked, the same filtered trainer-only
  manifest was used with explicit
  `nodeSelector: {node-group: nccl, node-pool: compute}`, and the run placed
  `trainer-head` on `research-common-h100-089` and `trainer-worker-1` on
  `research-common-h100-092`. No slots sampler/teacher/dispatch role was
  restamped, and the separate science stack was not touched.
- Artifact/server:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-chunk4-warmed-serveronly-3x-20260615T045836Z.jsonl`
  and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T045625Z-serveronly-configAMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-er-opd-q36-35b-slots-trainer-head/server.log`.
- Mean over `2` measured rows after `1` warmup:
  `server_forward_backward_s=5.503175`, `api_wall_s=5.994858`,
  `forward_compute_s=1.221080`, `backward_compute_s=3.036948`,
  `loss_compute_s=0.076166`, `clear_gradients_s=0.495572`, valid tokens
  `1030`, loss `2.355820715`, `opd_kl=2.334518359`, hidden loss
  `0.021302296`.
- Throughput signal: AMDAHL-064 reaches `187.16` valid tokens/server-second and
  `171.81` valid tokens/API-second. Versus AMDAHL-048, server wall is `23.83%`
  longer but valid/server-token rate is `61.51%` higher; versus AMDAHL-063,
  server wall is `32.92%` longer but valid/server-token rate is `50.46%` higher.
- Correctness caveat: repeating the exact capture should not change the
  token-weighted reported loss/KL/hidden tuple, but AMDAHL-064 shifted from the
  AMDAHL-048/063 tuple `2.355041921 / 2.333743553 / 0.021298385` to
  `2.355820715 / 2.334518359 / 0.021302296`. That is about `+0.000779` loss,
  `+0.000775` KL, and `+0.0000039` hidden loss.
- Conclusion: AMDAHL-064 is a useful fit/throughput signal for fatter calls, but
  it is not a science/default promotion. The next engine target should audit
  packing and token-weighted metric/loss aggregation across packed microbatches
  before promoting repeat2, prepare128, or any larger-current-call path.
  Cleanup complete: trainer controls were stopped, temporary trainer-head and
  worker-1 pods were deleted, and the slots stack was verified back to only
  dispatch + teacher-smg.

AMDAHL-065 repeat2 separator diagnostic (2026-06-15 05:13Z) completed and
is not correctness-clean:

- Purpose: split the AMDAHL-064 question into packing-neighbor sensitivity vs
  larger-call/aggregation effects. The custom replay used the same AMDAHL-063/064
  two-node forward-only prefetch trainer config, but fed the original AMDAHL-047
  full64 capture, then one artificial no-loss 4096-token separator datum, then
  the same 64 samples again. The separator forces the sequential packer to close
  the first copy's final packed microbatch before starting the second copy.
- Candidate/config:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR.yaml`
  and
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`.
- Launch/placement notes: kube auth worked, the filtered trainer-only manifest
  used explicit `nodeSelector: {node-group: nccl, node-pool: compute}`, and the
  run placed `trainer-head` on `research-common-h100-077` and `trainer-worker-1`
  on `research-common-h100-110`. No slots sampler/teacher/dispatch role was
  restamped, and the separate science stack was not touched.
- Artifact/server:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-separator-chunk4-warmed-serveronly-3x-20260615T050943Z.jsonl`
  and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T050707Z-serveronly-configAMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR-er-opd-q36-35b-slots-trainer-head/server.log`.
- Mean over `2` measured rows after `1` warmup:
  `server_forward_backward_s=5.892664`, `api_wall_s=6.389314`,
  `forward_compute_s=1.262510`, `backward_compute_s=3.069160`,
  `loss_compute_s=0.062265`, `clear_gradients_s=0.883305`, valid tokens
  `1030`, loss `2.356476724`, `opd_kl=2.335177020`, hidden loss
  `0.021299774`.
- Throughput signal: AMDAHL-065 reaches `174.79` valid tokens/server-second and
  `161.21` valid tokens/API-second. It remains `40.52%` higher valid/server
  throughput than AMDAHL-063, but is `6.61%` lower than AMDAHL-064 because the
  separator adds packed work without adding valid target tokens.
- Correctness caveat: preserving the first 64-sample pack boundary did not
  restore the AMDAHL-048/063 tuple. The AMDAHL-065 tuple shifted from
  `2.355041921 / 2.333743553 / 0.021298385` to
  `2.356476724 / 2.335177020 / 0.021299774`, about `+0.001435` loss,
  `+0.001433` KL, and `+0.0000014` hidden loss. It is also higher than
  AMDAHL-064 by about `+0.000656` loss and `+0.000659` KL, so the no-loss
  separator is not a clean corrective workload.
- Conclusion: AMDAHL-065 rejects the simple "repeat2 drift is only the first-copy
  boundary co-pack" explanation. It does not prove that packed-neighbor
  invariance is harmless, because the artificial separator changed the workload
  and objective reporting enough to move the tuple further. Treat the next engine
  target as a direct token-weighted loss/metric aggregation and packed-layout
  invariance audit, preferably with a replay that compares per-original-sample
  numerators/denominators rather than adding separator rows. Cleanup complete:
  trainer controls were stopped, temporary trainer-head and worker-1 pods were
  deleted, and the slots stack was verified back to only dispatch + teacher-smg.

AMDAHL-066 lm-head/VP-loss/packed-sample sidecar diagnostic (2026-06-15 06:12Z) COMPLETE/REJECTED:

- Purpose: turn the AMDAHL-064/065 repeat-data drift from a tuple-level symptom
  into direct evidence. The replay writes lm-head shard fingerprints,
  vocab-parallel loss numerator rows, and packed-sample segment rows, then
  compare same vocab shards across lm-head replica groups (`rank 0 vs 8`, `1 vs
  9`, etc.), compare `model_runner_local_*_contribution` against each TP group's
  KL sums, and map each packed segment back to original cache ranges/valid
  labels/weights.
- Engine branch/PR: fresh xorl-internal worktree
  `/home/apanda/xorl-opd-repeat2-diagnostics`, branch
  `codex/opd-repeat2-diagnostics-20260615`, commit `28354ca7`, draft PR #376
  stacked on PR #375. It includes the already-validated AMDAHL-062/063 replay
  hardening (`enable_backward_prefetch`, `load_checkpoint_optimizer`, and
  replay defrag toggles) plus new fixed-key OPD metrics
  `opd_vocab_parallel_group_tokens`, `opd_vocab_parallel_kl_sum`, and
  `opd_vocab_parallel_weighted_kl_sum`. These metrics are defaults-on in
  `OPDLossMetrics` to preserve distributed metric key-set symmetry, but the
  aggregate JSONL sidecar is opt-in via `opd_debug_vocab_parallel_loss_path`.
  The latest commit also adds opt-in `opd_debug_packed_sample_path`, which writes
  per-position-reset segment rows with teacher id, cache-index stats, input/label
  summaries, teacher weights, region/sample-ok counts, and vocab shard metadata.
  The branch also adds a default-safe post-load lm-head-TP parameter replica sync:
  modules marked `_xorl_fsdp_sharded_lm_head_loss` broadcast their local vocab
  shard over `lm_head_tp_replica_group` after server `ModelRunner` loads and
  local `Trainer` checkpoint resume loads. This keeps the parameter replica
  invariant aligned with the existing replica-gradient sum before AMDAHL-066
  compares shard fingerprints across ranks.
- Validation passed on the fresh branch:
  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/server/runner/test_opd_runner.py::test_vocab_parallel_loss_debug_writer_records_local_contribution tests/ops/loss/test_opd_verl_parity.py::test_metrics_dict_has_stable_key_set_across_loss_modes -q`
  (`2 passed`);
  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python tests/ops/loss/test_vp_kl_gathered.py`
  (all cases PASS; debug numerator rows `[23]`, KL sum err `1.818e-06`,
	  weighted KL sum err `1.729e-06`);
	  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest /home/apanda/xorl-opd-repeat2-diagnostics/tests/server/runner/test_opd_runner.py -q`
	  (`17 passed`, including the packed-sample segment writer);
	  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/distributed/test_torch_parallelize_policies.py tests/server/test_server_arguments.py tests/server/runner/test_checkpoint_loading.py -q`
	  (`55 passed, 30 skipped`);
  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest /home/apanda/xorl-opd-repeat2-diagnostics/tests/trainers/test_training_utils.py::test_sync_lm_head_tp_parameters_broadcasts_marked_module /home/apanda/xorl-opd-repeat2-diagnostics/tests/trainers/test_training_utils.py::test_sync_lm_head_tp_gradient_skips_single_rank_replica_group /home/apanda/xorl-opd-repeat2-diagnostics/tests/server/runner/test_checkpoint_loading.py -q`
  (`5 passed`);
  `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest /home/apanda/xorl-opd-repeat2-diagnostics/tests/distributed/test_lm_head_tp_fsdp_e2e.py -q`
  (`4 passed`);
  `py_compile` for the touched server/trainer/training-utils and lm-head-TP test
  files; and `git diff --check`.
- Runtime: refreshed kube auth worked (`can-i create/get pods -n apanda` =
  `yes`). The filtered trainer-only manifest used explicit
  `nodeSelector: {node-group: nccl, node-pool: compute}` and ran only
  `trainer-head` on `research-common-h100-077` plus `trainer-worker-1` on
  `research-common-h100-092`. No slots sampler/teacher/dispatch restamp and no
  science-stack touch. Cleanup is complete: trainer controls stopped, temporary
  trainer pods deleted, local port-forward killed, and slots verified back to
  only dispatch + teacher-smg.
- Run log:
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T060158Z-run.log`
  (xorl-internal commit `28354ca792c87ab00d39c1b06a09c3cb97b40802`; client
  commit `730379d4f335d92cad5ff2994ada434c2b5d6eb2`; sglang commit
  `c8552f4fa186b31ab49605fa170fefc5c4639a15`; run dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T060158Z-serveronly-configAMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-er-opd-q36-35b-slots-trainer-head`).
- Repeat1 sidecar artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-sidecar66-chunk4-warmed-serveronly-3x-20260615T060545Z.jsonl`,
  debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl066-debug-20260615T060545Z`.
  Mean over 2 measured rows after warmup: `server_forward_backward_s=4.092009`,
  `api_wall_s=4.414162`, `forward_compute_s=0.876625`,
  `backward_compute_s=2.043065`, `loss_compute_s=0.095551`,
  `clear_gradients_s=0.598869`, valid `515`, loss `2.354916632`,
  `opd_kl=2.333631971`, hidden loss `0.021284351`.
- Repeat2 sidecar artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-sidecar66-repeat2-chunk4-warmed-serveronly-2x-20260615T060828Z.jsonl`,
  debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl066-repeat2-debug-20260615T060828Z`.
  One measured row after warmup: `server_forward_backward_s=5.650213`,
  `api_wall_s=6.055651`, `forward_compute_s=1.274800`,
  `backward_compute_s=3.021867`, `loss_compute_s=0.148465`,
  `clear_gradients_s=0.759470`, valid `1030`, loss `2.356349051`,
  `opd_kl=2.335068045`, hidden loss `0.021281254`.
- Sidecar analysis:
  - Lm-head TP replica mismatch is ruled out for this run: `student_weight_local`
    and `student_weight_for_loss` fingerprints match exactly across `rank 0/8`,
    `1/9`, ..., `7/15` (`max_abs=0`).
  - VP-loss numerator aggregation is ruled out: within each complete 8-rank
    lm-head-TP group, the sum of `model_runner_local_kl_contribution` equals
    `opd_vocab_parallel_kl_sum` to roundoff, and `kl_sum / group_valid_tokens`
    equals the reported `opd_kl_group_mean`.
  - The repeat2 drift follows packed-layout changes. Repeat1 complete TP groups
    have token counts/means `(143, 2.939771)`, `(176, 2.023700)`, and
    `(196, 2.169704)`; their token-weighted mean is `2.333631971`. Repeat2 has
    `(206, 2.295636)`, `(213, 2.063958)`, `(113, 3.630783)`,
    `(194, 2.342567)`, `(264, 1.781739)`, and `(40, 3.937013)`, with
    token-weighted mean `2.335068045`.
  - Packed-sample rows show copy-boundary co-packing. Example: repeat2 batch 21
    on rank 7 contains segment 0 from original-tail cache `3513-3541`, segment 1
    from second-copy cache `44-47`, and segment 2 from second-copy cache `95-96`,
    all under group token count `206`.
- Verdict: AMDAHL-066 rejects repeat2/prepare128 promotion and keeps §1
  unchanged. The next useful target is per-segment/per-original-sample KL
  numerator attribution, or a direct packed-layout invariance fix, before any
  fatter-call science/default use.

AMDAHL-067..070 repeat2 hidden-state attribution (2026-06-15 07:21Z)
COMPLETES the first drift root-cause pass and rejects repeat2/prepare128:

- AMDAHL-067 reran the AMDAHL-066 sidecars with per-segment KL numerator rows
  on PR #376 commit `11029aa5`. It showed that repeat2 duplicate placements of
  the same original cache/label segment are not invariant; the largest spreads
  clustered on source rank `r` versus `r+8` with the same lm-head group rank
  under the EP8 x eFSDP2 layout. This moved the suspect before VP aggregation:
  source-rank-sensitive student forward/MoE/DeepEP behavior rather than a pure
  lm-head shard or detached metric-reduction bug.
- AMDAHL-068 tried the obvious topology isolation, `expert_parallel_size:16`
  (`ep_fsdp=1`) with DeepEP SMS36, but rejected at launch/runtime with the
  DeepEP IBGDA assertion `num_rc_per_pe == num_channels or num_rc_per_pe >=
  num_sms` in `internode.cu:492`. No replay row was produced.
- AMDAHL-069 lowered the EP16 diagnostic to `deepep_num_sms:24`. It ran and was
  raw-fast (`server_forward_backward_s=3.869048` repeat1; repeat2
  `5.187816`), but rejected on correctness: repeat1 loss/KL shifted versus the
  AMDAHL-063 same-capture tuple by about `+0.021273` / `+0.021255`, and repeat2
  still shifted versus that repeat1 by about `-0.001475` / `-0.001468`.
- AMDAHL-070 added per-segment student-hidden fingerprints to the packed-sample
  sidecar on PR #376 commit `69cbe49f`, with local validation:
  `tests/server/runner/test_opd_runner.py -q` (`17 passed`), `py_compile`,
  `git diff --check`, and
  `tests/ops/loss/test_vp_kl_gathered.py` (all subtests PASS). It reused the
  AMDAHL-063 two-node forward-only-prefetch, no-CP, lm-head-TP, VP-KL trainer
  config and wrote rank-suffixed sidecars.
- AMDAHL-070 artifacts:
  repeat1 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-hidden70-chunk4-warmed-serveronly-3x-20260615T071012Z.jsonl`,
  repeat1 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl070-hidden-debug-20260615T071012Z`,
  repeat2 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-hidden70-repeat2-chunk4-warmed-serveronly-2x-20260615T071306Z.jsonl`,
  repeat2 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl070-hidden-repeat2-debug-20260615T071306Z`,
  run log
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T070619Z-run.log`.
  The earlier `20260615T070851Z` debug attempt used the capture defaults
  (`opd_kl_backend=streaming` and full-vocab teacher head) and failed with a
  mixed Tensor/DTensor matmul; ignore it for the AMDAHL-070 verdict.
- AMDAHL-070 repeat1 metrics, measured over two rows after warmup:
  `server_forward_backward_s=4.648698`, `api_wall_s=4.852244`,
  `forward_compute_s=0.930723`, `backward_compute_s=2.045822`,
  valid `515`, loss `2.354453623`, `opd_kl=2.333146060`, hidden loss
  `0.021307328`. Repeat2 measured one row after warmup:
  `server_forward_backward_s=5.563359`, `api_wall_s=6.202537`,
  `forward_compute_s=1.286163`, `backward_compute_s=3.005776`,
  valid `1030`, loss `2.354963422`, `opd_kl=2.333680207`, hidden loss
  `0.021283326`. The same-server repeat2 shift is about `+0.000510` loss and
  `+0.000534` KL versus repeat1, so this remains non-promotable.
- AMDAHL-070 sidecar analysis used only the last measured request from each
  replay: repeat1 had `64` unique sample-identity keys; repeat2 had `128`
  segment rows, `64` unique keys, and exact multiplicity `2` for every key.
  For duplicate repeat2 keys, `17/64` had different `segment_student_hidden`
  sample hashes. Among the `20` duplicates that stayed on the same
  lm-head group rank, `8` had hidden-hash mismatches and material local-KL
  spreads. In contrast, same group-rank duplicates with the same hidden hash
  only showed roundoff KL spreads (`1.9e-6` to `3.8e-6`).
- Representative failing duplicate: cache/label key
  `(3513,3541,29,102283,16,271,29,8281)` was placed on rank `7` batch `21`
  and rank `15` batch `42` with the same group rank `7`; its hidden hash
  differed, hidden sample-sum spread was `38.149170`, and local weighted-KL sum
  spread was `1.533176`. Another same-group example
  `(2605,2606,2,5211,15,198,2,213)` on ranks `13` and `5` had hidden hash
  mismatch and KL-sum spread `0.616126`.
- Verdict: the repeat-data/fatter-call path is a real feed-rate lever but is
  still correctness-rejected. AMDAHL-070 moves the root cause before VP-KL/loss:
  duplicated samples can produce different student hidden states when source
  placement changes, especially across the `r`/`r+8` eFSDP/EP pattern. The next
  engine target is source-rank-invariant student forward/MoE/DeepEP behavior
  (or a routing/dispatch replay that proves the exact divergence point), not
  another loss aggregation rewrite. Cleanup complete: trainer controls stopped,
  temporary trainer-head/worker-1 pods deleted, local port-forward killed, and
  slots verified back to dispatch + teacher-smg only.

**2026-06-15 07:29Z follow-up:** PR #376 now has commit `ee3e8b8f`
(`opd: hash packed sample debug inputs`). It extends the packed-sample sidecar
with exact SHA-256 digests over dtype, shape, and bytes for segment
`input_ids`, labels, target tokens, position ids, teacher ids/cache rows,
teacher weights, region ids, and `opd_sample_ok`. Local validation passed with
`PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src python -m pytest
tests/server/runner/test_opd_runner.py -q` (`17 passed`), `ruff check` on the
touched files, and `git diff --check`. A 2-node trainer-only AMDAHL-071 replay
was attempted by rendering only the slots `trainer-head` service/pod and
`trainer-worker-1` pod with `OPD_XORL_REPO=/home/apanda/xorl-opd-repeat2-diagnostics`,
but both pods stayed Pending with scheduler reason `0/54 nodes are available:
17 node(s) didn't match Pod's node affinity/selector, 37 Insufficient
nvidia.com/gpu`. The pending pods were deleted and local waits were killed; no
replay rows were produced, no science stack was touched, and the denominator
MFU remains `0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`).

**2026-06-15 08:33Z follow-up:** AMDAHL-073/074 did not change the current
promotable MFU. AMDAHL-073 reached TileLang/TVM setup but failed before replay
with a `libtvm_compiler.so` symbol lookup error for
`_ZN3tvm3ffi9ReprPrintERKNS0_3AnyE`; no replay rows were produced and there is
no promotion. AMDAHL-074 is a default-preserving cleanup/allocator candidate on
PR #376 (`d4b5afe3`, `d20dfb32`, `db8ec2c7`) plus infra candidate key
`skip_optim_empty_cache:true`. It adds `XORL_SKIP_EMPTY_CACHE_AFTER_OPTIM_STEP`
support, exposes `optim_empty_cache_skipped` through sync and async optimizer
responses, and keeps the default path unchanged.

AMDAHL-074 same-server replay evidence on the AMDAHL-063 two-node forward-prefetch
real all-layer SGLang-cache workload:

- Baseline artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-currentbaseline74-realcache-cmp-chunk4-warmed-serveronly-4x-20260615T081347Z.jsonl`
  measured `server_forward_backward_s=4.219886`, `api_wall_s=4.495177`,
  and `clear_gradients_s=0.649127` over three measured rows.
- Replay-only no-GC/no-empty-cache stability artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-nodefrag74-realcache-stability12-chunk4-warmed-serveronly-12x-20260615T081714Z.jsonl`
  measured `server_forward_backward_s=3.991819`, `api_wall_s=4.269315`, and
  `clear_gradients_s=0.003191` over 11 measured rows after one warmup, with the
  same loss/KL/hidden tuple `2.367357433 / 2.346086165 / 0.021271308` and
  `valid_tokens=515`.
- Patched normal optimizer-step proof artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-nodefrag74-optimstep-proof-db8-chunk4-2x-20260615T083011Z.jsonl`
  measured one post-warmup row with `server_forward_backward_s=4.163849`,
  `api_wall_s=4.354535`, `optim_wall_s=0.044204`, and
  `optim_empty_cache_skipped_values=[true]`. This proves the normal async
  optimizer response/JSONL path carries the skip flag.

Verdict: AMDAHL-074 is real but modest, about a 5% two-node server/API wall win
for the cleanup-heavy replay path. It is not the missing 10% MFU lever and is
not a science/default promotion until same-capture static/K3 and same-workload
4-node real-cache MFU/fragmentation gates pass. Cleanup after the proof is
complete: local port-forward stopped, trainer controls stopped, temporary slots
trainer-head/worker-1 pods deleted, and the science stack was not touched.

Conclusion: the no-CP VP-KL dtype drift is fixed, and a real all-layer
SGLang-cache full64 one-node replay now fits with trainer-side OPRD teacher
forward removed. The default chunk size `4` remains the best stable
one-node setting found for the real-cache workload; larger layer chunks are not
the missing 10% MFU lever. This is still only a speed/fit milestone: the last
4-node denominator audit remains about `1.04%` logical MFU, AMDAHL-048's 2-node
retry improved one-node chunk4 only modestly, AMDAHL-049 rejects pack2304 for
the current real-cache workload, AMDAHL-050 rejects no-defrag allocator-flush
overrides under the current memory envelope, AMDAHL-051 does not produce a
meaningful forward-prefetch win, AMDAHL-052 OOMs before producing a repeat-data=2
warmup row, AMDAHL-053 shows skipping only CPU GC regresses total wall despite
shrinking the explicit clear-gradient timer, AMDAHL-054 shows reducing DeepEP
	reserved SMs from 36 to 24 regresses the same real-cache path, this real-cache
	path has not yet run the same-workload 4-node gate, AMDAHL-055 is not launchable
	 as a one-node custom MoE reduce-hook screen, AMDAHL-056 shows plain BF16 FSDP
	 reduce-scatter regresses the same path, AMDAHL-057 shows alltoall dispatch OOMs
	 before writing a warmup row, AMDAHL-058 shows deferred loss-report reduction
	 regresses server/API wall, AMDAHL-059 shows EP4 has a small wall-time win but
	 shifts same-capture KL/loss by about `+0.0181`, AMDAHL-060 shows 2-node
	 `reshard_after_forward:false` is flat on server wall, AMDAHL-061 shows
	 2-node `enable_forward_prefetch:true` is fast but shifts same-capture KL/loss
	 by about `+0.00458`, AMDAHL-062 backward-only prefetch is correctness-clean
	 but only modestly faster than AMDAHL-048 (`4.2808s`, -3.67% server wall),
  AMDAHL-063 forward-only prefetch is also correctness-clean and a somewhat
	 larger small win (`4.1402s`, -6.84% server wall) but still far smaller than
	 the rejected AMDAHL-061 coupled-prefetch speed, AMDAHL-064 repeat-data=2
	 fits and improves valid-token throughput but shifts same-capture loss/KL/hidden,
	 AMDAHL-065's separator diagnostic preserves a 64-sample boundary but shifts
		 loss/KL/hidden further, AMDAHL-066's follow-up lm-head/VP-loss/packed-sample
		 sidecar diagnostic rules out lm-head replica mismatch and VP numerator
		 aggregation drift but rejects repeat2/prepare128 because the KL/loss shift
		 follows packed-layout group changes, AMDAHL-067 maps the duplicate-sample
		 drift to source-rank placement, AMDAHL-068/069 reject the EP16 escape hatch,
		 and AMDAHL-070 confirms the material repeat2 drift is already present in
		 student hidden states before VP-KL/loss, AMDAHL-071 was scheduler-blocked,
		 AMDAHL-073 was dependency-blocked before replay, and AMDAHL-074 proves an
		 optimizer-empty-cache skip path with only a modest 2-node cleanup win,
	 and static/K3 correctness remains pending.
AMDAHL-044/045/047/048/049/050/051/052/053/054/055/056/057/058/059/060/061/062/063/064/065/066/067/068/069/070/071/073/074
remain speed/fit milestones, measured rejects, launch-gate rejections, or
diagnostics only.
Current cleanup check at 2026-06-15 08:33Z: Kubernetes auth works. AMDAHL-074
completed, no local replay/port-forward helpers are running, and no slots trainer
pods are running; the slots stack is back to dispatch + teacher-smg only. The
separate `er-opd-q36-35b-sci` stack was not touched.
Next throughput work should make the student forward/MoE/DeepEP path
source-rank invariant for repeated/fatter calls, or build the routing/dispatch
replay that pinpoints exactly where duplicate hidden states diverge. Do not
promote repeat2/prepare128 until that is fixed; only then spend a same-workload
4-node real-cache gate.

**2026-06-15 13:25Z K3/router follow-up:** The current audited/promotable MFU is
unchanged at `0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`)
from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
The latest work was a correctness diagnostic, not a throughput replay. The
router/expert capture artifacts for the row2 worst trace show the first large
dense drift at layer `15` MoE/MLP, and layer-15 router selection has a real
expert-set replacement at row `2069` (`148 -> 49`) rather than a row-alignment
bug. Forcing only layer `15` selected experts from the SGLang router tensor dump
while regathering weights from XoRL logits improved the one-trace row2 K3 from
`mean=0.005890282`, `p95=0.002430974`, `max=0.622587` to `mean=0.000494166`,
`p95=0.002081735`, `max=0.0168205`. The forced run exactly matched SGLang
selected experts at layer `15` (`slot_match=row_exact=1.0`) while leaving layers
`14` and `16` natural. Artifact dir:
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_force_sg_router_l15_row2_20260615T1317Z/`.
Conclusion: the low current MFU is still the measured systems state, but the
quack+DeepEP SMS36 K3 blocker now has a causal path through layer-15 MoE router
top-k/expert-choice drift. Next K3 target is a real engine fix for Qwen3.6 MoE
routing determinism/parity under the production EP8 path, followed by the full
worst-4/static gate before any throughput promotion.

**2026-06-15 13:46Z route-replay follow-up:** Route replay is not currently a
safe bridge around the K3 blocker, and the current MFU remains unchanged at
`0.010405` logical MFU. Artifact dir:
`/shared/opd-control/er-opd-q36-35b-slots/k3/r3_route_replay_row2_20260615T1334Z/`.
Refreshing the row2 worst trace captured nonempty SGLang `routed_experts`, but
`expert_logits=[]` for both prefill and generation. The refreshed generation
also mismatched the older trace at output position `4`, so this bundle is only a
diagnostic prefill-source artifact, not a gate. Replaying selected experts
without SGLang routing weights catastrophically failed
(`/shared/opd-control/er-opd-q36-35b-slots/k3/r3_route_replay_row2_20260615T1334Z/k3_r3_prefill_result.json`):
worst token K3 `27109118748`, SGLang target logprob `-0.0023805`, XoRL target
logprob `-24.0255`, XoRL target rank `244587`.

Tooling was hardened in `/home/apanda/xorl-opd-repeat2-diagnostics` and pushed
to PR #376 branch `codex/opd-repeat2-diagnostics-20260615` at `23450e38` so
empty route-weight payloads no longer look valid: trace refresh now records
accurate `has_expert_logits=false` provenance and clears stale fields, while
`compare_static_traces.py --xorl-use-trace-routed-expert-logits` now fails before
forwarding with `sglang_expert_logits is list(len=0)`. Validation passed
`py_compile`, `ruff check`, `git diff --check`, a focused helper smoke, and a
real-trace fail-fast smoke. Do not run another route-replay K3 from this static
trace. Either launch a dedicated SGLang reference that actually returns
nonempty expert weights, or fix Qwen3.6 MoE/router parity directly and then
rerun the one-trace K3 before any full worst-4/static or 4-node MFU gate.

**2026-06-15 13:56Z MoE attribution follow-up:** Engine PR #376 branch
`codex/opd-repeat2-diagnostics-20260615` is pushed to `563567ad`, adding
`experiments/k3_tests/diagnose_qwen36_moe_component_attribution.py` and a
corrected offline attribution artifact:
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_moe_component_attribution_row2069_20260615T135538Z.json`.
Validation passed `py_compile`, `ruff check`, `git diff --check`, and the
real tensor-dump diagnostic run. The row2069 layer-15 MLP delta is almost
entirely routed-expert output (`experts` mean abs `0.0030646` vs `mlp`
`0.0030937`, ratio `0.9906`); the weighted shared-expert contribution is small
(`0.0002457`, ratio `0.0794`), and both SGLang and XoRL `mlp = experts +
shared_expert_weighted` reconstruction closes to BF16 noise. Simulated top-k
variants on XoRL layer-15 logits do not provide a safe engine fix: tie-low /
stable-low improve exact ordering (`~0.26` row-exact vs stored `0.208`) but do
not improve unordered expert-set match (`~0.617-0.618`), while tie-high recovers
expert `148` for row2069 but worsens layer-wide unordered match to `0.563`.
Conclusion: do not implement router-FP32, simple tie-policy, or shared-expert
arithmetic changes as the next K3 candidate. The remaining useful path is
upstream hidden/residual parity before the layer-15 router logits, or a
dedicated SGLang reference that returns nonempty routing weights for route
replay.

**2026-06-15 14:03Z router-gate input attribution:** Engine PR #376 branch
`codex/opd-repeat2-diagnostics-20260615` is pushed to `c178bb6a`, adding
`experiments/k3_tests/diagnose_qwen36_router_gate_input_attribution.py`.
Corrected artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_router_gate_input_attribution_row2069_20260615T140249Z.json`
loads the real checkpoint gate weight
`model.language_model.layers.15.mlp.gate.weight` from
`model-00011-of-00026.safetensors` and recomputes layer-15 router logits from
each engine's captured `post_attention_norm`. Validation passed `py_compile`,
`ruff check`, `git diff --check`, and the real tensor-dump diagnostic run.
Under the primary BF16 recompute path, SGLang recompute-vs-captured-logits top-k
slot match is `0.999483` and XoRL is `0.999138`; recomputed cross-engine top-k
slot match is `0.661305`, essentially the same as captured-logits cross-engine
slot match `0.660846` (absolute delta `0.000460`). The full-layer
`post_attention_norm` input delta before the gate is already large
(`mean_abs=0.05554`, `rms_abs=0.09777`, `p95=0.17773`); row2069 input delta is
`mean_abs=0.02430`. Row2069 reproduces the captured divergence exactly: SGLang
logits select expert `148` in slot 8 while XoRL selects `49`. Conclusion: do
not implement gate-weight loading, gate matmul, router-FP32, or simple top-k
tie-policy changes as the next K3 candidate. The router mismatch is explained
by hidden/input drift arriving at layer-15 MoE; the next useful engine target is
upstream residual/hidden parity before the layer-15 `post_attention_norm`
tensor, not another router or route-replay launch.

**2026-06-15 14:06Z layer-15 full-attention attribution:** Reused existing
`experiments/k3_tests/diagnose_qwen36_full_attention_path.py` on the same
layer-14/16 tensor dumps to check whether the layer-15 gate-input drift is
introduced by full-attention arithmetic. Artifacts:
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer15_row2069_bf16_sdpa_20260615T140515Z.json`
and
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer15_row2069_fp32_eager_20260615T140548Z.json`.
The BF16/SDPA run shows full-layer captured attention drift mean abs
`0.0011042` and recomputed attention drift `0.0011020`; captured
`post_attention_norm` drift `0.0555438` and recomputed `0.0555308`. Row2069
attention drift is `0.0005799` captured vs `0.0005747` recomputed, while
`post_attention_norm` remains `0.0243048` captured vs `0.0240554` recomputed.
The FP32/eager sanity check does not remove the gap (`post_attention_norm`
full-layer `0.0555438` captured vs `0.0553535` recomputed; row2069 `0.0243048`
vs `0.0232686`). Conclusion: do not chase layer-15 q/k/v/o projection mapping,
RoPE, attention backend, attention output gate, residual add, or post-attention
RMSNorm as a standalone fix. Layer-15 attention/norm recomputation reproduces
the captured drift from the captured inputs; the remaining fix target is earlier
residual-stream divergence before the layer-15 inputs.
Missing gates before any science/default use:

- a static/K3 correctness gate joined to the dtype-fix throughput artifact
- a same-workload 4-node real-cache full-run or replay comparison that actually
  reaches the throughput target or beats the current promoted science path under
  the same correctness requirements

Until those gates pass, do **not** put `opd_kl_backend: vocab_parallel`,
`lm_head_tensor_parallel_size`, or the no-CP DP-sourced lm-head TP topology in
§1 or science defaults.

## 1-Node Microbench Ladder — Results (2026-06-14, Agent #1)

Built the cheap end of the ladder as **single-GPU** microbenches (no trainer, no
32-GPU stack) using the real Qwen3.6-35B-A3B shapes
(`H=2048`, `moe_intermediate=512`, `E=256`, `top_k=8`, `V=248320`, `40` layers).
Scripts live in `experiments/opd_profile/scripts/`. Ran on one free H100 with the
engine venv (`/home/apanda/xorl-internal/.venv`) and the engine worktree on
`PYTHONPATH` (see Artifacts). These resolve the root-cause question and produce a
validated, gradient-identical memory fix for the AMDAHL-029..033 blocker.

### MoE expert-GEMM size sweep (`microbench_moe_gemm.py`)

The student is an A3B MoE. Each expert is a `[M, 2048] @ [2048, 512]` SwiGLU
FFN where `M_per_expert = tokens_in_EP_group * top_k / E`. Sweeping M
(grouped `bmm` ceiling; `torch._grouped_mm` = engine `native` primitive;
per-expert loop floor), `MFU = achieved / 989 TF` bf16, EP=8 local_experts=32:

| M/expert | grouped `bmm` MFU | `torch._grouped_mm` MFU | per-expert **loop** floor |
|---|---|---|---|
| 8  | 1.4%  | 1.3%  | 0.07% |
| 64 | 11.1% | 10.4% | 0.53% |
| 128 | 19.0% | 18.9% | 1.05% |
| 256 | 28.7% | 30.4% | 2.12% |
| 512 | 37.0% | 39.1% | 4.22% |
| 1024 | 42.2% | 44.0% | 8.58% |
| 2048 | 44.6% | 46.1% | 17.2% |
| 4096 | 45.6% | 48.5% | 32.1% |

Conclusions:

1. **CORRECTION (important): the MoE expert GEMM is NOT the bottleneck at the
   real EP=8 operating point.** `M_per_expert = ep_group_tokens * top_k / E`, and
   with expert parallelism the all-to-all gathers the WHOLE EP group's tokens
   before the expert GEMM. The 1-node trainer config
   (`..._1node_warm009_deepep36.yaml`) is `expert_parallel_size=8`,
   `data_parallel_shard_size=8`, `moe_implementation=quack`, `ep_dispatch=deepep`
   ("deepep36" = `deepep_num_sms=36`, **not** EP=36). The full 64-sample OPRD
   batch is ~71804 real student tokens, so `M ≈ 71804*8/256 ≈ 2244` → **~45% MFU**
   on the expert GEMM. A small `M≈72` only happens at **EP=1** (pure DP), i.e.
   `tokens_per_rank/32` — and EP=1 was already tested and lost. An earlier version
   of this section wrongly mapped the operating point to `M≈72/12%` using EP=1
   semantics; the corrected number is `M≈2244/~45%`.
2. The grouped kernel is already at the ceiling: `torch._grouped_mm` (the engine
   `native` primitive) tracks `bmm` within ~1pp across the whole sweep. So when M
   *is* small, the deficit is small-M, not a fixable kernel inefficiency — and a
   per-expert loop would be catastrophic (0.6% at M=72, ~20× worse). **Do not
   regress onto an eager/loop expert path; do not chase a "better MoE kernel" at
   the EP=8 operating point — there is no MoE-GEMM win to get there.**
3. Where small GEMMs *would* bite: EP=1, or very sparse per-rank batches (few
   real tokens packed). Keep packing dense and EP≥8 and the expert GEMM stays in
   the 40-45% band. The lever "more tokens/expert" is real but already satisfied
   at EP=8 — it is NOT the explanation for ~1.37%.
4. So what IS the ~1.37%? It is **student-model FLOPs ÷ total wall time**, and the
   model GEMMs (MoE ~45%, lm-head KL ~20% — see below) are fine. The 4.46 s wall
   is dominated by work that is NOT student-model FLOPs: teacher forward (0.85 s),
   KL/loss (0.91 s), clear-grad (0.59 s), comms (DeepEP all-to-all), and — at 32
   GPUs — an **85% dummy-rank waste** (5 packed rows spread
   over 32 DP ranks; `dispatcher_dummy_executed_tokens=425088` of `497280`). So
   the path to 10%+ is **NOT** a MoE-GEMM change: kill dummy-rank waste
   (1-node / pack so rows ≈ DP size), cut clear-grad (0.59 s ≈ 13% of wall), and
   stop recomputing the teacher every step (0.85 s; the OPRD cache should serve
   it). Keep packing dense so M stays ≥256; that is already true at EP=8.

### lm-head streaming-KL is a MEMORY problem, not a small-GEMM problem (`microbench_lmhead_kl.py`)

`streaming_reverse_kl` fwd+bwd at real `V=248320`, `H=2048`, fp32 (the
`lm_head_fp32=true` regime), sweeping valid-token count N:

| N_valid | MFU | peak GB (fp32) |
|---|---|---|
| 128 | 9.1% | 6.1 |
| 1536 | 20.8% | 7.3 |
| 3049 (full batch) | 21.7% | 8.8 |
| 6144 | 22.1% | 11.9 |

The lm-head KL runs at **20%+ MFU** even at small N (the `V`-dimension is huge, so
the GEMM is not skinny) — it is **not** a small-GEMM problem. It IS the 1-node
**memory** blocker: at N=3049 the ~8.8 GB peak is dominated by **5.7 GB of weight
tensors** (student 1.9 + teacher 1.9 + a full fp32 `grad_weight` buffer 1.9), and
the trainer's grad-accumulation case (pre-existing `.grad`) pushes it to **10.7 GB**
(autograd doubles the returned grad against the existing `.grad`). This is exactly
the AMDAHL-033 "OOMed on full lm-head gradient allocation" failure. Free, numerics-
neutral lever found: `vocab_chunk_size=8192` cuts peak 8.8→6.6 GB for ~5% slower KL.

### Validated memory fix: `streaming_reverse_kl_lowmem` (engine, gradient-identical)

New engine path (`opd_streaming_kl.py`, opt-in via `opd_streaming_lowmem=true`):
keep the lm-head weights in their native (bf16) dtype and **upcast each vocab
chunk to fp32 inside the kernel** instead of holding two full fp32 weight copies;
return/accumulate the weight grad in the native dtype (optional in-place into the
leaf `.grad`, no second full `[V,H]` buffer). Because slicing commutes with the
elementwise upcast and vocab chunks partition grad rows disjointly, it is
**gradient-identical** to the current fp32 path (`validate_lmhead_kl_lowmem.py`:
`max|Δ|` = 0 on kl, grad_hidden, AND grad_weight). Measured peak reduction at
N=3049, vchunk=8192:

| case | current fp32 path | lowmem | saved |
|---|---|---|---|
| no grad-accum | 19.9 GB | 15.3 GB | **4.6 GB (23%)** |
| grad-accum (pre-existing `.grad`) | 21.8 GB | 15.3 GB | **6.5 GB (30%)** |

(absolute numbers inflated by resident comparison tensors; the **delta** is the
clean signal). Wired through `model_runner` as `opd_streaming_lowmem`; unit test
`test_opd_streaming_lowmem_matches_streaming` asserts loss + grad identity; full
`tests/ops/loss/test_opd_loss.py` (15) passes. Engine branch
`throughput/opd-lmhead-moe-gemm-20260614` off `origin/apanda-dev @ 609bed76`.

## Failed 1-Node Attempts

The desired cheap target is one-node OPD replay. The direct full-prep64 path is
currently blocked by the full lm-head FSDP all-gather, but smaller rungs fit and
are sufficient for the next inner loop. The failures are useful because they
locate the next work:

| candidate | result |
|---|---|
| AMDAHL-029 | 1-node trainer-side OPRD replay OOMed in `_trainer_teacher_kept_layers` before any replay row. |
| AMDAHL-030 | Moved OPRD hidden cache to SGLang rank-3, but all-40-layer cache saturated teacher memory and wrote no capture. |
| AMDAHL-031 | Every-4th-layer SGLang cache captured successfully, but 1-node trainer replay OOMed before a row. |
| AMDAHL-032 | Selected hooks removed full hidden retention, but all-layer prep64 still OOMed in model forward/final-norm. |
| AMDAHL-033 | Every-4th-layer cache + selected hooks reached streaming-KL backward, then OOMed on full lm-head gradient allocation in `opd_streaming_kl.py`. |
| AMDAHL-033 KL staging retry | Still OOMed because current OPD uses `lm_head_fp32=true`; the weight was already fp32 before grad allocation. |
| AMDAHL-033 root-caused + fixed (2026-06-14) | The blocker was concretely the fp32 lm-head copies + full fp32 `grad_weight` buffer (~5.7 GB at N=3049, 10.7 GB with grad-accum). `streaming_reverse_kl_lowmem` (per-chunk fp32 upcast, native-dtype grad, in-place optional) reclaims 4.6-6.5 GB **gradient-identical** — validated 1-GPU and then confirmed by AMDAHL-034 on the 1-node slot. |
| AMDAHL-034 (2026-06-14, cycle 2) | **RAN** (concurrent agent, ~04:31Z). The lowmem fix **WORKED end-to-end: it cleared the AMDAHL-033 lm-head grad OOM** — the replay got *past* streaming-KL backward into FSDP backward prefetch, then OOMed on a **~970 MiB all-gather with <1 GiB free per rank** (a NEW, different blocker than 033). Replay output `fb_replay/replay-amdahl034-1node-lowmem-serveronly-6x.jsonl` is empty (OOM before any row). → AMDAHL-035 disables FSDP fwd/bwd prefetch (`enable_forward_prefetch` gate) as the narrowest fit probe. The lowmem-fix validation (AMDAHL-033 root-cause) is now confirmed on real 1-node hardware; the remaining 1-node fit gap is FSDP all-gather headroom, not the lm-head grad. |
| AMDAHL-035 full / limit32 (2026-06-14 live slot) | `enable_forward_prefetch=false` disables XORL manual module prefetch lists but does **not** remove PyTorch FSDP2's pre-backward unshard. Full prep64 and `--limit-data 32` still OOM on the same ~970 MiB all-gather. |
| AMDAHL-035 limit8 / limit16 / limit24 | Fit on the 1-node slot and produced phase rows. `limit24` is the current largest fitting rung: `server_forward_backward_s=2.8989`, `forward=0.7920`, `backward=1.1761`, `clear=0.3815`, `model_forward=0.6562`, `loss_compute=0.1354`, `oprd_layer_fetch=0.0859`, `valid_tokens=160`. |
| AMDAHL-035 limit28 | Before the zero-anchor fix, it got past packing (`28` samples -> `10` packed batches, 75.9% utilization, 31104 tokens) and OOMed on `student_weight.float().sum() * 0.0`, which materialized a full fp32 lm-head copy. |
| AMDAHL-036 limit28 anchorfix | Engine commit `e123b782` fixed the full-fp32 zero anchor and passed unit tests, but `limit28` then reached the same ~970 MiB lm-head/FSDP all-gather. This isolates the remaining blocker to the full lm-head module unshard, not the lowmem KL path or zero anchors. |

Failed / discarded hypotheses this cycle (recorded so they are not retried):

- *"clear-grad (0.59 s) can be cut with `set_to_none`/fused zeroing."* **Closed** —
  `model_runner` already calls `zero_grad(set_to_none=True)` everywhere
  (`4020/4024/4256/4260`). The 0.59 s is sync/allocator attribution at the backward
  tail, not zeroing work; there is no fused-zeroing win to get.
- *"The teacher-forward (0.85 s) bucket is wasted recompute that the OPRD cache can
  remove."* **Already removed in AMDAHL-033/034** — with `opd_oprd_cache_backend:
  sglang` the client sets `oprd_trainer_forward = False`
  (`on_policy_distillation.py:1643`) and the trainer fetches the every4 cache. So
  this bucket is not a remaining lever in the every4 config; do not re-attack it.
- *"Fuse the streaming-KL forward to one pass (cache the logsumexp) to cut the KL."*
  **Deprioritized** — algebraically valid (single-pass `KL = C/Z_s − s_logz +
  t_logz`, ~25% of the KL matmul) but NOT bit-exact (~1e-6 drift → needs gating)
  and worth only ~0.03 s (KL is ~0.12 s fwd+bwd in isolation). The backward must
  recompute teacher logits regardless. The KL's value is its memory fix, not latency.

- *"The lm-head streaming KL is a small-GEMM / low-MFU hotspot."* **False** — it
  runs 20%+ MFU at realistic N (the V=248320 dimension keeps the GEMM fat). It is
  purely a memory problem.
- *"In-place `.grad` accumulation alone unblocks AMDAHL-033."* **Partly** — it only
  saves the autograd *doubling* (~1.9 GB), which appears in grad-accumulation, not
  in the single-microbatch first-allocation that 033 likely hit. The decisive
  lever is the **per-chunk fp32 upcast** (avoids the full fp32 weight copies),
  with in-place grad as a bonus.
- *"`lm_head_fp32=false` is the memory fix."* **Avoided** — the lowmem path gets
  the memory back with bit-exact fp32 numerics, so the gated `lm_head_fp32=false`
  numerics change is unnecessary for the memory goal.
- *"Going to 1 node shrinks the MoE GEMMs (smaller M)."* **False** — EP gathers the
  whole EP group's tokens before the expert GEMM, so M depends on packing density,
  not node count. 1 node helps by removing dummy-rank waste, not by changing M.
- *"Small MoE expert GEMMs (M≈72) are the cause of ~1% MFU."* **False at the EP=8
  operating point** (this was my first-pass conclusion, corrected same day). M≈72
  is the EP=1 mapping; with `expert_parallel_size=8` the all-to-all gathers the
  whole ~71804-token OPRD batch, so M≈2244 → ~45% MFU on the expert GEMM. The
  small-GEMM curve is real but the stack does not operate on its bad end. The ~1%
  is student-FLOP ÷ total-wall (teacher fwd + KL + clear-grad + comms) plus the
  32-GPU dummy-rank waste — a phase-mix/occupancy problem, not a GEMM-size one.

## Current 1-Node Target (2026-06-14 Live Slot Validation)

The cheap throughput inner loop is real now: use the reprogrammable slot
`er-opd-q36-35b-slots` and replay the captured fwd/bwd payload through the
trainer API on **one node**. Do not move back to 32 GPUs just to make the payload
fit. The point of this track is to remove the 1-node memory/phase-mix blockers
first, then promote.

Validated live setup:

- Engine: `/home/apanda/xorl-opd-throughput-20260614`, branch
  `throughput/opd-lmhead-moe-gemm-20260614`, PR #373.
- Client: `/home/apanda/xorl-opd-prefill`, branch `exp/opd-prefill`.
- Infra: `/home/apanda/xorl-infra`, branch `opd-battery-consolidation`.
- Slot: `er-opd-q36-35b-slots`, role `trainer-head`.
- Capture:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-031-oprd-prep64-deepep36-1node-sglangcache-every4.json`.
- Replay flags:
  `--loss-param opd_streaming_lowmem=true --loss-param opd_vocab_chunk_size=8192`.
- Config:
  `configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch.yaml`.

### What The Live Slot Proved

1. `opd_streaming_lowmem` cleared the AMDAHL-033 full lm-head grad OOM on real
   1-node hardware. The remaining failure moved later, into FSDP all-gather.
2. `enable_forward_prefetch=false` is not enough. It disables XORL manual module
   prefetch lists, but PyTorch FSDP2 still performs a pre-backward unshard.
3. Full prep64 and `--limit-data 32` fail on a ~970 MiB all-gather with less than
   1 GiB free per rank. `970 MiB ~= 248320 * 2048 * 2 bytes`, i.e. one full bf16
   lm-head shard/materialization boundary.
4. `--limit-data 24` is the largest fitting rung today and is the right profiling
   rung until the all-gather is removed.
5. `--limit-data 28` exposed and then validated a separate zero-anchor bug:
   `student_weight.float().sum() * 0.0` materialized a full fp32 lm-head copy
   (~1.89 GiB). Engine commit `e123b782` replaces that with scalar-slice fp32
   anchors in `model_runner.py` and `opd_loss.py`; after the fix, `limit28`
   reaches the same ~970 MiB all-gather instead of dying at the anchor.

Fit ladder from the live slot:

| replay | result | mean server fwd/bwd | notes |
|---|---:|---:|---|
| `limit8` | fit | `2.2370 s` | `valid_tokens=68`; forward/backward/clear `0.5749 / 0.8887 / 0.3894 s`. |
| `limit16` | fit | `2.2075 s` | `valid_tokens=105`; forward/backward/clear `0.4925 / 0.8663 / 0.3656 s`. |
| `limit24` | fit | `2.8989 s` | `valid_tokens=160`; packed `24 -> 8` rows, 81.2% utilization, 26624 tokens; forward/backward/clear `0.7920 / 1.1761 / 0.3815 s`. |
| `limit28` pre-anchor | fail | n/a | OOM on full fp32 zero anchor (`student_weight.float().sum()`). |
| `limit28` post-anchor | fail | n/a | OOM on ~970 MiB full lm-head FSDP all-gather. |
| `limit32` | fail | n/a | Same ~970 MiB full lm-head FSDP all-gather. |
| `limit64` / full prep64 | fail | n/a | Same all-gather class; lowmem KL itself is no longer the first OOM. |

### Next Engineering Target

Attack the full lm-head module unshard. The suspected edge is
`_lm_head_forward_anchor(hidden_states, student_lm_head)`: even though streaming
OPD KL consumes the lm-head weight tensor directly, this one-token module forward
exists to preserve graph/FSDP hook ordering and appears to force the full lm-head
unshard. The next candidate should be an opt-in probe that either:

1. skips/replaces `_lm_head_forward_anchor` for the streaming OPD KL path with a
   graph edge that does not call `student_lm_head.forward`, or
2. implements true sharded/vocab-parallel OPD KL weight-gradient handling so the
   full lm-head is never materialized on a rank.

Promotion order:

1. Re-run `limit28`; it must get past the current all-gather.
2. Re-run `limit32`, then full `prep64`.
3. Capture 6 replay rows on full prep64 and report phase breakdown plus executed
   MFU.
4. Only after full prep64 fits and improves on one node, promote to 4-node replay
   to measure scaling/dummy-rank behavior. Do not add nodes to hide the 1-node
   lm-head all-gather.

Do **not** treat `opd_streaming_lowmem` as a numerics change: it is bit-exact vs
the current fp32 path. The in-place-`.grad` mode of the lowmem Function is OFF in
the `model_runner` wiring (returns a native-dtype grad, standard autograd); flip
it on only after checking FSDP/DTensor `.grad` semantics.

Closed/deprioritized levers from this pass:

- `enable_forward_prefetch=false`: tried; does not remove the FSDP2 all-gather.
- clear-grad/fused zeroing: `model_runner` already uses
  `zero_grad(set_to_none=True)`; the bucket is sync/allocator attribution, not
  literal zeroing work.
- teacher-forward recompute: with `opd_oprd_cache_backend: sglang`, the trainer
  fetches every-4th-layer cache instead of running trainer-side teacher forward.
- KL forward fusion: algebraically possible but not bit-exact and worth only
  about 0.03 s in the isolated bench; the KL's value here is the memory fix.
- small MoE GEMM, EP=1, no-checkpoint, pack2304, pause-token trimming: already
  measured negative, irrelevant at EP=8, or science-recipe changes rather than
  throughput fixes.

## 2026-06-14 (cycle 3) — standalone-trainer MFU ceiling, CP verdict, fp32 lm-head memory recipe

**Methodology note (honest).** This cycle did NOT extend the AMDAHL-021 *server-replay*
ladder directly, for two reasons: (a) the real 1-node OPD *server* `forward_backward`
is blocked by a rank-0 **dispatch deadlock** (orchestrator→ZMQ→Rank0Protocol never
broadcasts the fb command at world_size=8; works at ≥2 nodes and for
`register_session`) — root-caused by the live-slot agent via faulthandler, so the
1-node server-replay inner loop cannot run regardless of memory; and (b) the work was
redirected (by the owner) to the MFU-ceiling, CP, and loss-mode-memory questions. So
the ladder here is the **bare `xorl.cli.train` standalone trainer** (Tier-0 ceiling
probe in §7e) on the real Qwen3.6 shapes + a single-GPU lm-head loss-mode microbench.
These directly attack the AMDAHL-029..033 **streaming-KL/lm-head gradient memory**
blocker the goal names. Full per-attempt detail + raw numbers:
`/shared/apanda/tput-mine/NOTES.md`. Bench stacks (mine, isolated): `q36-tput-mine`
(1-node), `q36-tput-2node`, `q36-tput-4node`.

### AMDAHL-021 payload reproduction (re-run this cycle, grounds the memory attack)

Re-ran `audit_forward_backward_denominator.py` on the AMDAHL-021 captured payload
(`fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json`) — reproduces its exact
packed/dummy/valid shape and the low executed-MFU (output:
`fb_replay/mfu_denominator_audit_cycle3_20260614.json`):

- 22 packed rows @4096 = 73,088 row-padded tokens; +10 dispatcher dummy rows
  (+34,560 tok) → 107,648 dispatcher-executed student tokens.
- **3,049 valid target tokens (2.83% of executed)** → **reconstructed executed-MFU
  = 1.37%** (0.039% valid-scaled), `server_forward_backward_s = 4.46` (4-node replay
  baseline; loss split fwd 1.60 / bwd 1.58 / clear-grad 0.63 / kl 0.046 s).

This is the prescribed "start from the AMDAHL-021 payload, preserve packed/dummy/valid
shape, reproduce the low executed-MFU" step. **The memory attack below is at this exact
payload's lm-head shape:** the streaming-KL/lm-head loss runs over the **3,049 valid
tokens** at `V=248320, H=2048`, which is precisely the `N=3049` used in the
`microbench_kl_backends_mem.py` / `validate_lmhead_kl_lowmem.py` comparisons — so the
lowmem fix and the loss-mode memory table are measured on the AMDAHL-021 lm-head
gradient state (AMDAHL-033), not a synthetic surrogate. The 1-node *server* replay of
this payload remains blocked by the orchestrator→rank0 dispatch deadlock (live-slot
agent), so the lm-head memory blocker is attacked via the single-GPU bare-tensor
microbench at the payload's shape rather than the server replay.

### Reproduced the low 1-node executed-MFU AND root-caused it (it is NOT GEMM-starving)

Standalone-trainer fwd/bwd MFU at the OPD topology (EP=8, quack, deepep36, recompute,
synthetic balanced routing, MFU is recompute-fair), sweeping node count / sharding:

| topology | dp_shard | tokens/rank | peak mem | MFU | note |
|---|---|---|---|---|---|
| 1-node | 8 | 16,384 | **69 GB** | **~5%** | flat 4.3% (8k) → 5.3% (16k); does NOT climb |
| 2-node | 16 | 16,384 | 40 GB | **~10.6%** | `nocp_mbs4` best |
| 4-node | 32 | 16,384 | 28 GB | ~10.0% | `nocp_mbs4` |
| 4-node | 32 | 32,768 | 39 GB | ~10.5% | `nocp_mbs8` — bigger GEMM ~neutral |

**Root cause of the flat ~5% at 1 node: MEMORY PRESSURE, not small GEMMs.** At
dp_shard=8 the peak is 69/80 GB (86%), leaving no headroom for FSDP forward-prefetch
to overlap → comms serialize → ~5% regardless of tokens/rank, async-combine, or
offload. dp_shard≥16 (≥2 nodes) drops peak to ~40 GB → prefetch overlaps → **2×
MFU (~10%)**. The model's clean fwd/bwd ceiling is **~10–10.6%** and it **plateaus**
there (bigger GEMMs/more nodes/CP do not beat it). The lever is *enough FSDP sharding
for memory headroom*, not node count. (Independently corroborated by the Wordle
agent's raw-transformer ~9% server-path peak.)

### CP (context/ulysses parallelism) MONOTONICALLY HURTS here — not the lever

| config | MFU |
|---|---|
| 2-node no-CP mbs4 | ~10.6% |
| 2-node CP=2 mbs4 | ~6–7.5% |
| 2-node CP=2 mbs8 | ~9.4% |
| 4-node CP=2 mbs8 | ~8.9% |
| 4-node CP=4 mbs8 | ~6.4% |

CP splits the sequence, but OPD samples are short (~1138 tokens) so there is nothing
to split; CP only adds all-to-all/all-gather comms. More CP = lower MFU. **Do not use
CP for this short-sequence MoE.** (offload and `deepep_async_combine` also did not
help: ~5.5–7.4% and neutral, respectively.)

### fp32 lm-head memory — which loss mode helps (owner directive: keep fp32)

Keep `lm_head_fp32=true` (reverse-KL accuracy on rare near-certain tokens) and pick
the loss mode that minimizes the 1.89 GiB fp32 `grad_weight` blocker. Single-GPU
microbench (`microbench_kl_backends_mem.py`, V=248320 H=2048 N=3049, fp32):

| reverse-KL backend | peak (no accum) | peak (grad-accum) | numerics vs fp32 baseline |
|---|---|---|---|
| `streaming` (baseline OPD) | 9.63 GB | 13.27 GB | reference |
| **`streaming_lowmem`** (PR #373) | **5.86 GB** | **7.79 GB** | **bit-exact (max\|Δ\|=0)** |
| `compiled` / auto_chunker ("fused") | 16.35 GB | OOM / dtype-error | NOT bit-exact (Δkl=2.3e-3) |

**The "fused" compiled/auto_chunker path is WORSE on memory** (1.7× baseline — it still
does `student_weight.float()` full fp32 copies plus compile/chunk intermediates), is
not bit-exact, and errors in the grad-accum case. **`streaming_lowmem` is the only mode
that keeps lm-head fp32 *bit-exact* while cutting memory.** Do NOT drop to bf16-head
(the Wordle agent's +16% lever) — it carries the rare-token KL accuracy risk the owner
wants to avoid.

**FSDP-safe by ENGINE DESIGN (definitively resolves the prior "lowmem multi-rank
unverified / DTensor slicing may break" caveat).** The lm-head weight is a **full
local tensor at the loss, not a sharded DTensor** — by deliberate engine design, not
luck: `torch_parallelize.py:381-385` groups `norm + lm_head` into one FSDP unit with
`reshard_after_forward=False`, "so that when `norm.forward()` runs FSDP all-gathers
both, and **they stay gathered so external `compute_loss()` can access `lm_head.weight`
without a redundant all-gather**." The vocab-sharded loss path (`fsdp_sharded_lm_head_loss`)
is opt-in and requires CP + `dp_size=1`, which OPD does not use. So the streaming KL
(baseline AND lowmem) always slices a **full local** `student_weight[start:end]` — never
a DTensor. (This also corrects Agent A's suspected mechanism: it is the norm+lm_head
FSDP grouping that keeps the weight gathered, not `weight.float()`.) Additionally the
wired mode uses `inplace_weight_grad=False` → returns a native-dtype grad through
standard autograd (same reduction path as baseline), so even the grad accumulation is
identical. Confirmed bit-exact in this mode (`validate_lmhead_kl_lowmem.py --no-inplace
--preexisting-grad`: max|Δ|=0, saves 5.59 GB). A 4-node trainer-only replay with
`opd_streaming_lowmem=true` is now only optional belt-and-suspenders, not a correctness
gate.

**Memory recipe (keep fp32):** `opd_kl_backend=streaming` + `lm_head_fp32=true` +
`opd_streaming_lowmem=true` (inplace off). Bit-exact, ~5.6 GB saved, FSDP-safe.

### Production recipe status toward the ~10% ceiling (this cycle + live-slot + Wordle agents)

The standalone model ceiling (~10.6%) is proven, but OPD has not reached it. The
latest 4-node trainer-only replay (AMDAHL-041 below) rejects the simple
"0-dummy / bigger prepare batch is enough" hypothesis: fatter repeated payloads
raise logical MFU only from ~1.42% to ~1.83%. Keep the already-measured good
substrate choices (`>=2` nodes for memory headroom, no-CP, dense microbatching,
lm-head fp32 + `streaming_lowmem`), but do **not** promote a larger
`opd_prepare_batch_size` by static replay alone. The live work remains engine
and loss-shape work: remove the lm-head all-gather memory cap for 1-node/2-node
fit tests, then attack the OPRD teacher-forward / hidden-match phase mix that
dominates at larger repeated batches.

## 2026-06-14 (cycle 4, overnight, Agent #1) — lm-head KL engine work

Goal this cycle: engine changes to raise MFU, prove 10% at 4-node. DeepEP
`low_latency_mode` was triaged OUT first: it is decode-oriented (fixed-size RDMA
buffers, FP8, forward-only) and **OPD is FP8-off**, so the all-to-all already
sits at ~19% (Wordle's own FP8-off result), not the bottleneck. Refocused on the
two real OPD-specific costs: the full-vocab KL on the giant lm-head, and the
full-lm-head FSDP all-gather memory blocker.

**lm-head KL backend comparison** (1-GPU microbench, V=248320 H=2048, fp32 lm-head,
`compare_kl_backends.py`): at N=8192 fwd+bwd —
`compiled` 138 ms / **19.1 GB** (materializes full [N,V] logits; `num_chunks` has
no effect — auto_chunker not reducing memory in torch 2.10); `streaming` 303 ms /
10.0 GB; `streaming_lowmem` 311 ms / **9.45 GB** (leanest). vc=32768 beats 65536 on
memory at equal speed. So `compiled` is fast-but-hungry; streaming variants are
slow-but-lean. (Consistent with the runbook note that KL *compute* is tiny at the
current limit24 batch — this matters only once the batch grows.)

**One-pass fused streaming reverse-KL** — landed as PR #374
(branch `throughput/opd-fused-kl`). The streaming forward did 2 vocab passes
(logsumexp + KL); fused to 1 via `KL = A/Z_s - logZ_s + logZ_t`,
`A = Σ exp(s_v-s_max)(s_v-t_v)` (online-accumulable). Backward untouched →
gradient-identical. Validated vs brute-force full-logit reference (kl rel 2.7e-4,
grad rel 1.2e-4): **~23% faster** fwd+bwd (315→243 ms @ N=8192), same 8.0 GB peak.
`validate_onepass_kl.py`. NB: a side win, not the MFU lever (KL compute is small
at limit24).

**Vocab-parallel reverse-KL kernel** — the real lever for the all-gather blocker.
Validated, branch `throughput/opd-vocab-parallel-kl` (`vocab_parallel_reverse_kl.py`
+ multi-process gloo test `test_vocab_parallel_reverse_kl.py`): each rank uses only
its [V/world,H] lm-head shard, computes full-vocab KL via all_reduce of tiny [N,1-3]
stats — **no full lm-head, no full logits anywhere**, and each rank's logits are
world× smaller (one fast matmul, no chunk loop). Matches the full-vocab reference to
float32 precision (kl rel 3e-5, grad rel 1e-6) across 4 ranks.

**Integration constraint discovered (the hard part, NOT yet wired):** under FSDP2
the lm-head shard group ALSO data-shards the tokens — each rank has different tokens
*and* a different vocab slice, which is the wrong layout for vocab-parallel. The fix
is to **all_gather hidden states (cheap: weight 1.89 GB ≫ activations ~72 MB), keep
the weight sharded** — i.e. invert FSDP's gather-weight/shard-activation for the
lm-head only. Then VP-KL applies directly and comms drop ~12×. The remaining risk is
the autograd: the activation all_gather must inject grad only into the LOCAL token
slice (no double-count), and the lm-head shard grad must be supplied to FSDP2 without
fighting its gather/reduce-scatter hooks. This is the next engine step; the kernel is
de-risked. Skip `_lm_head_forward_anchor`'s gather on this path.

**Scale note for the 10%-at-4-node goal:** the blockers differ by scale, but the
initial 4-node feeding diagnosis was too optimistic. 1-node is still
lm-head-all-gather memory blocked (the VP-KL path above). 4-node does have some
row under-fill at prep64 (`22` packed rows over `32` ranks), but AMDAHL-041 shows
that repeated feeding alone only modestly raises executed-token throughput and
does not approach the 10% MFU ceiling. Treat bigger prepare batches as a
candidate to full-run gate, not as a promoted fix.

**LIVE 1-node trainer-replay evidence (2026-06-14 ~10:5xZ, er-opd-tput-apanda-0614,
8 GPU, expandable_segments ON, base config recompute_before_dispatch + compiled KL,
amdahl-031 capture).** Steady-state `server_forward_backward` mean over a `--limit-data`
sweep (3 iters, 1 warmup each):

| limit-data | fb wall | model_fwd | backward | kl_compute |
|---|---|---|---|---|
| 8  | 4.93 s | 0.71 s | 1.91 s | 0.000 s |
| 16 | 4.88 s | 0.66 s | 2.09 s | 0.000 s |
| 24 | 5.06 s | 0.71 s | 2.49 s | 0.09 s |

**The fb wall is ~FLAT (4.9–5.1 s) while the data triples (8→24 datums).** The fb is
fixed-overhead / dummy-padding dominated, NOT data-proportional, at these batches: real
datums fill otherwise-padded/dummy slots almost for free. So throughput (datums/s) and
MFU climb ~LINEARLY with batch size up to the memory cap — limit8→24 is ~3× the
throughput at the same wall. Extrapolating, fitting full prep64 (64 datums) at ~the same
~5 s would be ~2.7× the MFU of limit24 just from filling the pack. The cap is the
lm-head FSDP all-gather (full prep64 OOMs; reproduced live — a too-big capture OOM'd
mid-DeepEP-dispatch → rank desync → "DeepEP CPU recv timeout" + CUDA crash). KL compute
is confirmed negligible (0.00–0.09 s), and clear_gradients (~1.28 s, ~25% at limit24) is
a replay artifact (forced per-fb; amortized over grad-accum in real training; already
`set_to_none`). **Net: the #1 1-node MFU lever is raising the memory cap so a bigger
batch fits — exactly what the validated vocab-parallel reverse-KL kernel enables (drop
the full lm-head gather). The integration is the next deep change (task #13).**

**VP-KL integration — autograd fully de-risked offline (branch `throughput/opd-vocab-parallel-kl`).**
Two multi-process (gloo) tests now pass to float32 precision vs single-process full-vocab
references: (1) `test_vocab_parallel_reverse_kl` — the kernel's cross-rank vocab-parallel
reductions (kl rel 3e-5, grad rel 1e-6); (2) `test_vp_kl_gathered` — the FSDP glue
`vocab_parallel_reverse_kl_gathered` (gather activations, shard weights): the local
token-slice hidden grad matches the reference slice (rel 8.7e-7, **no cross-rank
double-count**) and the weight-shard grad matches (rel 1.5e-6). So the hard part (autograd
correctness across the data+vocab-sharded group) is proven. **Remaining model_runner
wiring (needs the live multi-rank trainer to develop, since it depends on the live FSDP2
DTensor placements / EP×FSDP mesh): (a) get the lm-head local shard via `lm_head.weight`
DTensor `.to_local()` + the `fsdp_mesh` group + vocab offset; (b) match the teacher shard
via `head_manager.sharded_view`; (c) pad uneven per-rank token counts to a uniform count +
mask; (d) supply the local-shard grad back to FSDP2 as the DTensor grad; (e) skip
`_lm_head_forward_anchor`'s gather; gate behind `opd_kl_backend=vocab_parallel` (opt-in →
cannot affect the science default path) and validate loss-match vs the full-gather path.**

**2026-06-14 19:35Z follow-up:** engine draft PR #375 now carries the VP-KL
kernel/glue branch (`throughput/opd-vocab-parallel-kl`, commit `08080a73`,
stacked on PR #374). It hardens the gathered-activation helper for the real OPD
case where local valid-token counts differ across ranks, including zero-token
ranks, and materializes functional collectives so the gloo tests no longer exit
with outstanding-work warnings. Validation:
`test_vocab_parallel_reverse_kl.py` passes (kl rel `3.051e-05`, hidden-grad rel
`8.040e-07`, weight-grad rel `1.061e-06`); `test_vp_kl_gathered.py` passes the
equal-count case (hidden-grad rel `8.655e-07`, weight-grad rel `1.487e-06`) and
the new uneven case `counts=(7,0,13,3)` (hidden-grad rel `1.211e-06`,
weight-grad rel `2.487e-06`, gathered rows `[23]`). Ruff check/format also
pass on the touched files. **Still not promoted:** model_runner wiring,
trainer replay, and K3/static gate are pending, so `opd_kl_backend=vocab_parallel`
is an engine building block only.

**AMDAHL-041 4-node feeding validation (2026-06-14 19:40-19:52Z): simple feeding is
NOT the 10% fix.** Kube auth recovered and an explicit `nodeSelector: node-group=nccl`
survived server admission, so the trainer-only stack ran on full idle nccl nodes
`099/110/092/088` without touching `er-opd-q36-35b-sci`. Engine repo:
`/home/apanda/xorl-opd-throughput-20260614 @ 27b1694`; candidate:
`AMDAHL-021-OPRD-PREP64-DEEPEP36-FB-CAPTURE`; config:
`qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36.yaml`. Controls were stopped and
the four slots trainer pods were deleted after the sweep.

| repeat-data | prompt-equivalent | packed rows | dummy rows | fb wall | valid tok/s/GPU | executed tok/s/GPU | reconstructed logical MFU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 22 | 10 | 4.286 s | 22.23 | 784.83 | 1.424% |
| 2 | 128 | 43 | 21 | 7.578 s | 25.15 | 895.78 | 1.625% |
| 4 | 256 | 86 | 10 | 10.710 s | 35.59 | 946.77 | 1.718% |
| 8 | 512 | 171 | 21 | 20.330 s | 37.49 | 1008.96 | 1.831% |

Interpretation: static repeated feeding helps only modestly. Valid-token rate is
~1.69× better at repeat8 than repeat1 despite 8× more valid tokens, and logical MFU
only improves ~29% relative (1.42% → 1.83%). The base capture actually packs to
`22` rows at `seq_len=4096`, not the earlier misremembered `~5` rows; repeat2/repeat8
still add dummy rows because `43`/`171` rows round up to `64`/`192` dispatch slots.
The phase mix scales with the fatter payload (`opd_profile_oprd_teacher_forward_s`
mean `0.824 → 5.085`, backward `1.577 → 9.734`, loss compute `0.882 → 5.361`),
while KL remains small (`0.050 → 0.277`) and clear-grad stays around `0.46-0.51s`.
**No §1 promotion:** do not bump `opd_prepare_batch_size` for science from this
static replay. Next target returns to engine/loss-shape work: finish VP-KL
model-runner wiring for the lm-head all-gather cap, then use the fitted 1-node/2-node
path to test whether fewer trainer ranks plus the real generated batch improves
tokens/rank and phase mix.

**AMDAHL-042/043 lm-head-TP VP-KL trainer replay (2026-06-14 20:25-21:05Z):
fit-safe, not yet fast enough.** PR #375 at `5c17f409` now supports
`lm_head_tensor_parallel_size` in the server path and threads
`fsdp_sharded_lm_head_loss` into model construction. The 1-node Ulysses8 +
lm-head-TP topology can run the real full64 VP-KL replay after a fresh-server
warmup: warmed mean `server_forward_backward_s=24.8447`, with
`model_forward_s=9.3195`, `backward_compute_s=13.9033`, and
`kl_compute_s=0.1587`. The 4-node fidelity variant (`data_parallel_shard_size=4`,
`ulysses_parallel_size=8`, `lm_head_tensor_parallel_size=8`) also fits full64:
the warmed 3-iteration mean is `server_forward_backward_s=15.4878`,
`model_forward_s=4.6513`, `backward_compute_s=9.4123`,
`kl_compute_s=0.0672`. This is a real fit/memory milestone and a `1.6x` wall-time
improvement over the 1-node lm-head-TP run, but it does **not** beat the current
promoted strict 4-node trainer path; the next engine target is to preserve the
lm-head materialization fix without paying the Ulysses8/backward tax. This is
code work, not a YAML-only sweep: the current `parallel_state.py` requires
`lm_head_tp_size>1` to be carved out of the CP axis, so a no-CP/lower-CP
lm-head-TP config is invalid until that constraint changes. K3/static gate any
promotable result.

## Do Not Spend The Next Cycle On

- Context/ulysses parallelism (CP) for this short-sequence MoE — measured to hurt
  monotonically (2-node CP2 ~7%, 4-node CP4 ~6.4% vs no-CP ~10%).
- Adding trainer nodes before the 1-node lm-head all-gather is fixed.
- Full OPD science launches as the first test.
- KL/top-k loss micro-optimizations; KL compute was tiny in replay.
- Pause-token trimming as a throughput fix; that is a science recipe change.
- No-checkpoint, EP=1, or pack2304 promotion without new evidence; those have
  measured negative or misleading results above.

## Artifacts

- Capture payload:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json`
- Baseline replay output:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-serveronly-6x.jsonl`
- Breakdown replay output:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-breakdown-sortedmetrics-serveronly-5x.jsonl`
- Denominator audit:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_20260613.json`
- AMDAHL-041 4-node repeat-data replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl041-4node-repeat1-serveronly-4x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl041-4node-repeat2-serveronly-4x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl041-4node-repeat4-serveronly-4x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl041-4node-repeat8-serveronly-3x.jsonl`.
- AMDAHL-041 denominator audits:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_amdahl041_repeat1_20260614.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_amdahl041_repeat2_20260614.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_amdahl041_repeat4_20260614.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_amdahl041_repeat8_20260614.json`.
- AMDAHL-042/043 lm-head-TP VP-KL replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-full64-warmed-serveronly-3x-20260614T2058.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-4node-full64-serveronly-1x-20260614T2102.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-4node-full64-warmed-serveronly-3x-20260614T2110.jsonl`.
- AMDAHL-044/045/046 dtype-fix replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-streaming-nocp-lmheaddebug-hidden0-serveronly-1x-20260614T2245.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-hidden0-serveronly-1x-20260614T2305.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-full64-warmed-serveronly-4x-20260614T2307.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-dtypefix-4node-full64-warmed-serveronly-4x-20260614T2321.jsonl`.
- Dtype-fix 4-node denominator audit:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
- AMDAHL-031 SGLang-layer-cache no-CP VP replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-warmed-serveronly-4x-20260614T2340.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-layerdevcache-warmed-serveronly-4x-20260614T2343.jsonl`.
- AMDAHL-047 synthetic all-layer layer-cache capture and valid-only/chunked replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-oprd-prep64-deepep36-alllayer-synthlayercache.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-sglangcache-every4-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0009.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-synthlayercache-all40-1node-validonly-chunkedmse-layerdevcache-warmed-serveronly-4x-20260615T0013.jsonl`.
- Real all-layer SGLang-cache capture and chunk-size replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-047-realalllayer-sglangcache-full64-20260615T004904Z.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-warmed-serveronly-4x-20260615T005428Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk4-warmed-serveronly-4x-20260615T010816Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk8-warmed-serveronly-4x-20260615T010816Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-chunk16-warmed-serveronly-4x-20260615T010816Z.jsonl`.
- AMDAHL-048 2-node candidate/config and completed retry output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-048-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-chunk4-warmed-serveronly-4x-20260615T012513Z.jsonl`.
- AMDAHL-049 pack2304 candidate/config/audit and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-049-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-PACK2304.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_pack2304.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_realcache_pack_sweep_20260615.json`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-pack2304-chunk4-warmed-serveronly-4x-20260615T013525Z.jsonl`.
- AMDAHL-050 no-defrag candidate and rejected partial output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-050-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-NODEFRAG.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-nodefrag-chunk4-warmed-serveronly-4x-20260615T014448Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T014208Z-serveronly-configAMDAHL-050-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-NODEFRAG-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-051 forward-prefetch candidate/config and neutral output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-051-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-PREFETCH.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_prefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-prefetch-chunk4-warmed-serveronly-4x-20260615T020000Z.jsonl`.
- AMDAHL-052 repeat-data=2 candidate and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-052-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-REPEAT2.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-repeat2-chunk4-warmed-serveronly-4x-20260615T021000Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T020711Z-serveronly-configAMDAHL-052-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-REPEAT2-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-053 GC-split candidate and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-053-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-GCSKIP.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-gcskip-chunk4-warmed-serveronly-4x-20260615T021900Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T021555Z-serveronly-configAMDAHL-053-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-GCSKIP-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-054 DeepEP SMS24 candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-054-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEEPEP24.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep24_noprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-deepep24-chunk4-warmed-serveronly-4x-20260615T023000Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T022624Z-serveronly-configAMDAHL-054-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEEPEP24-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-055 MoE BF16-a2a reduce candidate/config and launch-rejected logs:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-055-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-MOEBF16A2A.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_moebf16a2a.yaml`,
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T023950Z-run.log`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T023950Z-serveronly-configAMDAHL-055-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-MOEBF16A2A-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-056 BF16 FSDP reduce candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-056-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-FSDPBF16REDUCE.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_fsdpbf16reduce.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-fsdpbf16reduce-chunk4-warmed-serveronly-4x-20260615T025041Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T024526Z-serveronly-configAMDAHL-056-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-FSDPBF16REDUCE-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-057 alltoall dispatch candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-057-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-ALLTOALL.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_alltoall_noprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-alltoall-chunk4-warmed-serveronly-4x-20260615T030520Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T030227Z-serveronly-configAMDAHL-057-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-ALLTOALL-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-058 deferred loss-report reduce candidate and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-058-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEFERLOSSREDUCE.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-deferlossreduce-chunk4-warmed-serveronly-4x-20260615T031650Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T031508Z-serveronly-configAMDAHL-058-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-DEFERLOSSREDUCE-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-059 EP4 DeepEP topology candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-059-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-EP4DEEPEP.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch_lmheadtp_nocp_ep4.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-1node-ep4deepep-chunk4-warmed-serveronly-4x-20260615T033549Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T033344Z-serveronly-configAMDAHL-059-OPRD-PREP64-1NODE-LMHEADTP-NOCP-VPKL-EP4DEEPEP-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-060 2-node reshard-off candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-060-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-RESHARDOFF.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_noprefetch_lmheadtp_nocp_reshardoff.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-reshardoff-chunk4-warmed-serveronly-4x-20260615T035332Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T034833Z-serveronly-configAMDAHL-060-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-RESHARDOFF-er-opd-q36-35b-slots-trainer-head/server.log`.
  Empty bad-shell-quoting artifact to ignore:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-reshardoff-chunk4-warmed-serveronly-4x-20260615T034923Z.jsonl`.
- AMDAHL-061 2-node forward-prefetch candidate/config and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-061-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-PREFETCH.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_prefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-prefetch-chunk4-warmed-serveronly-4x-20260615T040241Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T040018Z-serveronly-configAMDAHL-061-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-PREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-062 2-node backward-only prefetch candidate/config and measured output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_backprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-backprefetch-chunk4-warmed-serveronly-4x-20260615T043109Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T042714Z-serveronly-configAMDAHL-062-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-BACKPREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-063 2-node forward-only prefetch candidate/config and measured output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-chunk4-warmed-serveronly-4x-20260615T044953Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T044707Z-serveronly-configAMDAHL-063-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-064 2-node repeat-data=2 fatter-call fit/throughput candidate and
  caveated output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-chunk4-warmed-serveronly-3x-20260615T045836Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T045625Z-serveronly-configAMDAHL-064-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-065 2-node repeat2 separator diagnostic candidate and rejected output:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR.yaml`,
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp.yaml`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-repeat2-separator-chunk4-warmed-serveronly-3x-20260615T050943Z.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T050707Z-serveronly-configAMDAHL-065-OPRD-PREP128-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-REPEAT2-SEPARATOR-er-opd-q36-35b-slots-trainer-head/server.log`.
- AMDAHL-066 2-node sidecar diagnostics on PR #376 commit `28354ca7`:
  repeat1 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-sidecar66-chunk4-warmed-serveronly-3x-20260615T060545Z.jsonl`,
  repeat1 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl066-debug-20260615T060545Z`,
  repeat2 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-sidecar66-repeat2-chunk4-warmed-serveronly-2x-20260615T060828Z.jsonl`,
  repeat2 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl066-repeat2-debug-20260615T060828Z`,
  run log
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T060158Z-run.log`.
- AMDAHL-070 2-node hidden-fingerprint diagnostics on PR #376 commit
  `69cbe49f`:
  repeat1 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-hidden70-chunk4-warmed-serveronly-3x-20260615T071012Z.jsonl`,
  repeat1 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl070-hidden-debug-20260615T071012Z`,
  repeat2 output
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-hidden70-repeat2-chunk4-warmed-serveronly-2x-20260615T071306Z.jsonl`,
  repeat2 debug dir
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl070-hidden-repeat2-debug-20260615T071306Z`,
  run log
  `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/20260615T070619Z-run.log`.
- Replay script:
  `experiments/opd_profile/scripts/replay_forward_backward_capture.py`
- Denominator audit script:
  `experiments/opd_profile/scripts/audit_forward_backward_denominator.py`
- Candidate YAMLs:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-021-*.yaml`
  through `AMDAHL-065-*.yaml`. AMDAHL-066/067/070 are sidecar diagnostics that
  reuse the AMDAHL-063 trainer config and have no dedicated candidate YAML.
  AMDAHL-047
  is a derived capture artifact, not a
  candidate YAML; AMDAHL-048 is a 2-node tooling candidate, AMDAHL-049 is a
  pack2304 tooling candidate, AMDAHL-050 is a no-defrag tooling candidate, and
  AMDAHL-051 is a forward-prefetch tooling candidate. AMDAHL-052 is a
  repeat-data=2 tooling candidate that OOMed before writing a replay row.
  AMDAHL-053 is a CPU-GC-skip tooling candidate rejected by measured regression.
  AMDAHL-054 is a DeepEP SMS24 tooling candidate rejected by measured regression.
  AMDAHL-055 is a MoE BF16-a2a reduce tooling candidate rejected at launch gate
  on the one-node topology. AMDAHL-056 is a BF16 FSDP reduce tooling candidate
	  rejected by measured regression. AMDAHL-057 is an alltoall dispatch tooling
	  candidate rejected by warmup OOM. AMDAHL-058 is a deferred loss-report reduce
	  engine-screen candidate rejected by measured regression; its local engine
	  reporting patch was reverted/not promoted after the run. AMDAHL-059 is an EP4
	  DeepEP topology candidate rejected because its small speed win came with
	  same-capture KL/loss drift. AMDAHL-060 is a 2-node
	  `reshard_after_forward:false` tooling candidate rejected because server wall
	  was flat versus AMDAHL-048. AMDAHL-061 is a 2-node forward-prefetch tooling
	  candidate rejected because its strong replay speed win came with
	  same-capture KL/loss drift. AMDAHL-062 is a measured 2-node backward-only
	  prefetch tooling candidate: correctness-clean versus AMDAHL-048, but only a
	  modest wall-time win and not a standalone promotion. AMDAHL-063 is a
	  measured 2-node forward-only prefetch isolation candidate: correctness-clean
	  versus AMDAHL-048 and faster than backward-only, but still only a small
	  diagnostic speed lever and not a standalone promotion. AMDAHL-064 is a
	  2-node repeat-data=2 fatter-call screen that fits and improves valid-token
	  throughput, but shifts same-capture loss/KL/hidden and is not a promotion.
	  AMDAHL-065 is a 2-node repeat2 separator diagnostic that preserves the first
	  copy's pack boundary but shifts loss/KL/hidden further, so it is rejected and
	  points the next audit at token-weighted aggregation plus packed-layout
	  invariance. AMDAHL-067 is the per-segment KL attribution diagnostic that
	  maps duplicate-sample drift to source-rank placement; AMDAHL-068/069 are
	  EP16 escape-hatch diagnostics rejected by DeepEP launch failure and
	  loss/KL drift respectively; AMDAHL-070 is the hidden-fingerprint diagnostic
	  confirming the material repeat2 drift starts in student hidden states.
  These have replay metrics or fit/fail evidence only for the real-cache
  trainer-server path.

### cycle-3 artifacts (2026-06-14) — MFU ceiling + loss-mode memory

- KL-backend fp32 memory comparison bench:
  `experiments/opd_profile/scripts/microbench_kl_backends_mem.py` (streaming vs
  streaming_lowmem vs compiled/auto_chunker; the loss-mode-memory table above).
- Standalone-trainer MFU sweep (bare `xorl.cli.train`, OPD topology), configs +
  per-rank run.sh + results under `/shared/apanda/tput-mine/` (`configs/`,
  `NOTES.md` = full per-attempt ledger). Bench stacks: pods `q36-tput-mine`,
  `q36-tput-2node`, `q36-tput-4node` (control dirs `/shared/opd-control/q36-tput-*`).
- Engine worktree with research branch + lowmem cherry-pick (for a real-OPD replay):
  `/home/apanda/xorl-opd-mine-tput` (`throughput/opd-mine-tput-20260614`).

### 1-node microbench ladder artifacts (2026-06-14)

- MoE GEMM root-cause bench:
  `experiments/opd_profile/scripts/microbench_moe_gemm.py`
- lm-head streaming-KL memory/MFU bench:
  `experiments/opd_profile/scripts/microbench_lmhead_kl.py`
- lowmem-fix numerics+memory validator:
  `experiments/opd_profile/scripts/validate_lmhead_kl_lowmem.py`
- Microbench result JSONs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/microbench/`
  (`moe_gemm_ep8_20260614.json`, `moe_gemm_ep1_20260614.json`,
  `lmhead_kl_fp32_20260614.json`, `lmhead_kl_bf16_20260614.json`,
  `lmhead_kl_fp32_preexist_20260614.json`). Cycle-2 reproduction (2026-06-14
  ~04:31Z, confirms committed `61c90e6e` reproduces exactly):
  `moe_gemm_ep8_repro_20260614.json`, `lmhead_kl_fp32_repro_20260614.json`,
  `lmhead_kl_fp32_preexist_repro_20260614.json`.
- Engine fix branch: `xorl-internal` `throughput/opd-lmhead-moe-gemm-20260614`
  off `origin/apanda-dev @ 609bed76` (worktree
  `/home/apanda/xorl-opd-throughput-20260614`). Commits:
  `61c90e6e` lowmem streaming reverse-KL,
  `a4b0ec48` streaming diagnostics dtype-match,
  `e123b782` scalar-slice fp32 zero anchors. Files:
  `src/xorl/ops/loss/opd_streaming_kl.py` (`streaming_reverse_kl_lowmem_function`),
  `src/xorl/ops/loss/opd_loss.py` (`streaming_lowmem` param),
  `src/xorl/server/runner/model_runner.py` (`opd_streaming_lowmem` plumbing),
  `tests/ops/loss/test_opd_loss.py` (`test_opd_streaming_lowmem_matches_streaming`).
- Run recipe (single GPU):
  `CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=/home/apanda/xorl-opd-throughput-20260614/src
  /home/apanda/xorl-internal/.venv/bin/python experiments/opd_profile/scripts/<bench>.py`
- Live 1-node replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit8-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit16-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit24-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit28-lowmem-noprefetch-serveronly-3x.jsonl`
  (pre-anchor fail), and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl036-limit28-anchorfix-lowmem-noprefetch-serveronly-3x.jsonl`
  (post-anchor all-gather fail).
- Live 1-node server log dirs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T043652Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T044158Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`, and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T045116Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`.

## AMDAHL-075 Row-Batch Diagnostic (2026-06-15 08:58Z)

Current audited/promotable 4-node MFU is still unchanged:
`0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
The newer real-cache pack-sweep audit remains a side-path, not a 4-node
promotion (`0.008962` 1-node and `0.009395` 2-node).

AMDAHL-075 tested packed-row batching on the current 2-node real all-layer
SGLang-cache replay, using PR #376. Commit `10d12e74` proved the orchestrator
could reduce `22 -> 6` backend batches for rowbatch4, but the live replay failed
before rows because the stacked implementation created batch dimension >1 and
the Qwen3.6 packed-varlen short-conv path requires `batch_size=1`. Commit
`eddf68c5` changed grouping to concatenate packed rows into one longer
packed-varlen row while preserving `cu_seq_lens`.

The concat retry is rejected. Rowbatch4 completed at
`server_forward_backward_s=4.345614` but shifted loss/KL/hidden to
`2.371532` / `2.350189` / `0.0213423`. Rowbatch2 completed at
`server_forward_backward_s=2.831341`; using the AMDAHL-048 2-node denominator,
that raw time would imply about `0.014747` logical MFU (`~1.47%`) and
`14.58` logical TFLOPS/GPU, but it also shifted loss/KL/hidden to
`2.358434` / `2.337124` / `0.0213099` versus the same-capture baseline
`2.355042` / `2.333744` / `0.0212984`.

Verdict: the stack is still around 1% logical MFU by the current promotable
audit, and this is not only an external data-pipeline starvation problem. Even
trainer-only replay on a static payload is underfeeding the GPU because the
trainer executes too many small packed-row/model-communication units. Reducing
backend batch count can improve raw wall time, but the concat form changes the
student/OPD numeric path, likely through packed-layout or source-rank-sensitive
GDN/MoE/DeepEP behavior. The next useful target is a packed-layout/source-rank
invariance fix that keeps row-batch speed without changing loss/KL, not a
larger science run or more sampler data.

## AMDAHL-076/077 Rank-Local Row-Batch Follow-Up (2026-06-15 09:24Z)

Current audited/promotable 4-node MFU is still unchanged:
`0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.

AMDAHL-076 tested whether segment-looping the packed GDN output restored
correctness for the AMDAHL-075 rowbatch2 concat path. It did not. Candidate
`experiments/opd_profile/autoresearch/candidates/AMDAHL-076-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-ROWBATCH2-GDNSEGLOOP.yaml`
exported `XORL_GDN_PACKED_SEGMENT_LOOP=true`; artifact
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-76-gdnsegloop-chunk4-warmed-serveronly-4x-20260615T090421Z.jsonl`
measured `server_forward_backward_s=3.272135`, loss/KL/hidden
`2.358434` / `2.337124` / `0.0213099`, the same invalid tuple as AMDAHL-075
rowbatch2 and slower than that raw rowbatch2 screen. Verdict: GDN segment-loop
is falsified as the correctness fix.

AMDAHL-077 changed the implementation instead: apply packed-row batching only
after each trainer rank has selected its original source-rank slice. Engine PR
#376 commit `a9f2e2c1` adds `opd_packed_row_batch_scope: rank_local`, shared
packed-row helpers, and runner-side grouping/padding. Local validation passed:
`py_compile`, focused request-processor/dispatcher tests, full touched
request-processor/dispatcher tests (`25 passed`), dispatcher-forward test, ruff,
and diff-check.

Live two-node replay used trainer-only pods on `research-common-h100-089/110`
with candidate
`experiments/opd_profile/autoresearch/candidates/AMDAHL-077-OPRD-PREP64-2NODE-LMHEADTP-NOCP-VPKL-FWDPREFETCH-ROWBATCH2-RANKLOCAL.yaml`.
Server logs confirmed the desired semantics: RequestProcessor deferred
rowbatching to rank-local runner slices, and ranks with two local real rows
logged `2 -> 1` local real batches; executor-global rows stayed at `22`.

Same-server no-rowbatch control artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch77-sameserver-baseline-chunk4-warmed-serveronly-3x-20260615T091857Z.jsonl`.
Mean over 2 measured rows: `server_forward_backward_s=3.548186`,
`api_wall_s=3.826919`, loss/KL/hidden
`2.361179` / `2.339847` / `0.0213319`.

Rank-local rowbatch2 stability artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-77-ranklocal-stability12-chunk4-serveronly-20260615T091946Z.jsonl`.
Mean over 11 measured rows: `server_forward_backward_s=2.936666`,
`api_wall_s=3.184268`, loss/KL/hidden
`2.361179` / `2.339847` / `0.0213319`. This is a clean same-server
`17.23%` server-wall win. Using the AMDAHL-048 2-node denominator, it implies
raw 2-node logical MFU `0.014218` (`~1.42%`) and logical TFLOPS/GPU `14.06`.

Verdict: rank-local rowbatch fixes the source-rank ownership bug in the
executor-global rowbatch idea and is the first correctness-clean rowbatch speed
lever on this path. It still does not change the current promotable MFU because
it is a 2-node server-only proxy without static/K3 or same-workload 4-node audit.
The next promotion gate is a static/K3 check plus 4-node real-cache replay with
the same rank-local rowbatch implementation. The broader diagnosis remains:
this is not mainly "samplers cannot feed the trainer"; the trainer server can
replay a static payload and still lands near 1% because the internal execution
unitization is too small and communication-heavy.

### AMDAHL-077 Static/K3 Gate (2026-06-15 10:33Z)

AMDAHL-077 is **not promotable**: the first OPD-aligned quack+DeepEP static gate
failed the strict mean-K3 threshold.

Engine/tooling state:

- Engine branch `/home/apanda/xorl-opd-repeat2-diagnostics`
  `codex/opd-repeat2-diagnostics-20260615` is pushed through `dc58e5ee`.
- Commit `dc58e5ee` adds K3 launcher node-selector knobs and explicit config
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/configs/qwen3_6_35b_ep8_deepep_sms36_nocompile_noasync.yaml`.
- K3 config: `moe_implementation: quack`, `ep_dispatch: deepep`,
  `deepep_num_sms: 36`, `deepep_async_combine: false`,
  `enable_compile: false`, `expert_parallel_size: 8`.

Artifacts:

- Static traces:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k3/sglanggen_trace_refresh_20260615T033726Z/q36-coderforge-pilot-opd32-sglanggen-static-traces-20260615T033726Z.json`
- K3 result:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_result.json`
- Joined gate:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_gated_summary.json`
- Diagnosis:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_diagnosis.json`
- Worst-4 repro:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/q36-opd077-quack-deepep-sms36-k3-repro-worst4.json`

Gate result:

- Coverage: `32/32` prompts, `2820/2820` tokens.
- Failed: `mean_k3=0.0016289913158513175 > 0.001`.
- Passed: `p95=0.0031932317246325496 < 0.01`.
- Max K3: `0.6225874560430769`.
- Worst token: trace `train.parquet:rg0:row2`, position `22`, token
  `17880`, SGLang logprob `-2.6560185`, xorl logprob `-3.5976868`.
- Diagnosis: shift `0` is the best alignment (`mean=0.00162899`); shift
  `-3..+3` all regress badly, so this is not a simple label offset.

Verdict: do **not** spend a same-workload 4-node replay on AMDAHL-077 until the
K3 mismatch is fixed and re-gated. The next useful work is to debug the worst-4
static repro against model/kernel differences, likely starting with the
Qwen3.6 linear-attention/GDN and MoE/DeepEP paths. The speed evidence still says
rank-local rowbatch is the right feed-rate lever once correctness clears.

### AMDAHL-077 K3 Diagnostics After Failed Gate (2026-06-15 11:00Z)

The first localization round ruled out several cheap explanations for the
AMDAHL-077 K3 failure, but did not clear the gate. Engine branch
`/home/apanda/xorl-opd-repeat2-diagnostics`
`codex/opd-repeat2-diagnostics-20260615` is pushed through `f36f5ed7`.
That commit exposes existing model-runner diagnostics through static replay:
reference logits, loss-logprob deltas, hidden-state summaries, and
hidden-component summaries. Validation passed `py_compile`, `ruff check`,
`git diff --check`, and a local monkeypatch smoke that confirmed the diagnostic
loss parameters are threaded into the xorl model runner.

Reduced worst-4 controls:

- `lm_head_fp32:true`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/lmheadfp32_worst4_20260615T1039Z/k3_gated_summary.json`
  failed with `mean_k3=0.0076940066`.
- `router_fp32:true`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/routerfp32_worst4_20260615T1043Z/k3_gated_summary.json`
  failed and regressed to `mean_k3=0.0158171338`; the same worst token worsened
  to `k3=1.668889`.
- `flash_attention_deterministic:true`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/fa3det_worst4_20260615T1046Z/k3_gated_summary.json`
  failed with `mean_k3=0.0078013087`, matching the original selected worst-4
  profile, so deterministic FA3 is not the fix.

One-trace component diagnostic:

- Artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/components_row2_20260615T1055Z/k3_gated_summary.json`
- Coverage: reduced smoke only, `1/4` repro prompts and `128/385` tokens.
- Result: failed `mean_k3=0.005890282137603278 > 0.001`, with
  `p95=0.0024309740226052604 < 0.01`.
- Worst token remains `train.parquet:rg0:row2`, position `22`, token `17880`:
  SGLang-generation logprob `-2.6560185`, xorl logprob `-3.5976868`.
- Loss path ruled out for that token: `xorl_loss_logprob` equals the emitted
  xorl target logprob, and explicit FP32 reference lm-head scoring is only
  `0.0103006` logprob away from xorl while the SGLang-generation gap is
  `0.941668`.

Reference-mode audit: the static trace also carries SGLang prefill logprobs,
but switching the full 32-trace offline comparison from generation logprobs to
prefill logprobs is not a pass. Prefill improves some tokens but worsens the
aggregate worst case, so this is not simply the wrong SGLang reference mode.

Verdict: the remaining blocker is SGLang-vs-xorl model-forward/reference parity,
not data volume, target extraction, lm-head dtype, router dtype, or deterministic
FA3. The next useful diagnostic is a SGLang activation/component reference for
the same worst token and layer path, then a targeted engine fix and full static
gate rerun before any 4-node rowbatch replay.

### AMDAHL-077 SGLang Tensor-Dump Reference (2026-06-15 11:15Z)

The SGLang-side activation reference for the AMDAHL-077 worst trace now exists.
This still does **not** promote AMDAHL-077; it gives the next model-path
debugging artifact.

Tooling landed in the active engine branch
`codex/opd-repeat2-diagnostics-20260615` at `e73469d3`:

- `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/compare_hidden_component_artifacts.py`
- `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/sglang_debug_dump_to_artifact.py`
- `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/prepare_sglang_component_tensor_dump.py`

Validation:

- `py_compile` passed for the three scripts.
- `ruff check` passed for the three scripts.
- Direct utility smoke passed: build SGLang artifact from synthetic tensors,
  compare component summaries, and prepare a component tensor dump.

Live SGLang diagnostic:

- Trace: `train.parquet:rg0:row2` from the AMDAHL-077 worst-4 repro bundle.
- Request mode: SGLang prefill scoring of full prompt+target.
- Node: `research-common-h100-110`, standalone temporary SGLang pod only.
- Layers/components: layers `0,10,20,30,38,39`, `num_layers=40`,
  `hidden_dim=2048`, all component tensors.
- Launcher caveat: the richer debug-dump launcher's rendered manifest
  hardcoded `nodeSelector: node-group: default`, so the first pod failed
  `NodeAffinity` before running. The applied manifest changed only that selector
  to `node-group: nccl`; the pod then ran, served the request, and was deleted.

Artifacts:

- Summary:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/diagnostic_summary.json`
- Converted SGLang component artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/sglang_component_artifact.json`
- xorl-vs-SGLang summary diff:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/xorl_vs_sglang_component_compare.json`
- Normalized full SGLang tensor bundle:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/sglang_component_tensors.pt`
- Applied manifest:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/logs/sglang-pod.applied-nccl-nodegroup.yaml`

Result:

- Summary diff matched `128/128` component-token rows and `9216` components.
- Largest sampled deltas are in late-layer component summaries, especially
  layer `39` `post_attention_norm` / `shared_expert_input` at the K3-worst
  token, and layer `38`/`39` downstream components across other high-impact
  tokens.
- The summary diff is **diagnostic only**. Sampled component summaries can be
  sensitive to tensor source/layout and are not enough to choose a model-code
  fix.

Next exact diagnostic: rerun xorl on the same one-trace repro with
`--xorl-diagnostic-hidden-component-path` for the same layers and compare the
full xorl component tensors against `sglang_component_tensors.pt`. Do this
before touching Qwen3.6 model math.

### AMDAHL-077 Full Component Tensor Compare (2026-06-15 11:35Z)

The full xorl component dump for the same AMDAHL-077 worst trace is complete.
This still does **not** promote AMDAHL-077; it turns the K3 failure into a
model-forward parity target rather than a feed-rate target.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `f017deb6`.
- New utility:
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/compare_component_tensor_dumps.py`.
- Validation passed:
  `py_compile`,
  `ruff check /home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/compare_component_tensor_dumps.py`,
  `git diff --check -- experiments/k3_tests/compare_component_tensor_dumps.py`,
  and direct full-tensor compare against the saved artifacts.

Artifacts:

- xorl one-trace K3/component run:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/k3_result.json`
- xorl full component tensors:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/tensors/xorl_components.rank0.pt`
  through
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/tensors/xorl_components.rank7.pt`
- Full compare JSON:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/xorl_rank0_vs_sglang_component_tensor_compare.json`
- Concise summary:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/xorl_rank0_vs_sglang_component_tensor_summary.json`

Checks before interpreting the compare:

- SGLang raw TP rank check: `input_layernorm`, `post_attention_layernorm`, and
  `layer_output` tensors are replicated across TP ranks for inspected layers, so
  using rank0 for those residual-stream components is not an obvious converter
  artifact. TP-partial expert/shared-expert tensors differ by rank as expected.
- xorl raw rank check: inspected full component tensors are replicated across
  rank0/rank1, so rank0 is representative for this diagnostic. Labels differ by
  rank, but the compared hidden/component tensors do not.
- The current SGLang debug prefill/scoring logprob at the K3-worst token is
  also far above xorl (`-2.5399` versus xorl `-3.5977`), so the failure is not
  explained by the old generation-vs-prefill trace-field split.

Result:

- Compare counts: reference `78` tensor keys, candidate `72`, common `72`.
  Missing xorl keys are only the SGLang-only `raw_mlp` component for layers
  `0,10,20,30,38,39`; there are no shape mismatches.
- K3-worst token: `train.parquet:rg0:row2`, output position `22`, absolute
  token position `2070`, compare row `2069`, token id `17880`. K3 gap remains
  the xorl logprob being lower by about `0.9417`.
- At row `2069`, layer-output drift grows monotonically across sampled layers:
  layer0 mean abs `8.64e-05`, layer10 `3.84e-04`, layer20 `0.00380`,
  layer30 `0.00846`, layer38 `0.03204`, layer39 `0.03940`.
- The biggest row-2069 component deltas by mean abs are late-layer norms:
  layer38 `post_attention_norm` mean abs `0.2572`, layer39
  `post_attention_norm` `0.2289`, layer38 `input_norm` `0.1562`, and layer30
  `post_attention_norm` `0.1555`.
- Large label-window outliers also occur away from the K3-worst token, especially
  layer20/30 `input_norm` and `post_attention_norm` / `shared_expert_input`
  rows at hidden index `2045` (for example row `2158` layer30
  `post_attention_norm` max abs `30.0`).

Interpretation:

- The loss/logit extraction path is already ruled out, and the full tensor
  compare now shows a real model-forward residual-stream divergence accumulating
  through the sampled Qwen3.6 layers.
- The sampled layers `0,10,20,30,38` are linear-attention layers and layer `39`
  is full attention. The next highest-value target is Qwen3.6
  linear-attention/GDN parity against SGLang, not more data-pipeline volume or a
  4-node throughput spend.
- A local SGLang-venv parity attempt did not produce evidence because importing
  the SGLang reference path became stuck in D-state before GPU allocation. Do
  not count that as pass/fail; if kernel-level parity is needed, run it in a
  clean pod/venv or use already-running SGLang debug infrastructure.

### AMDAHL-077 Long-Sequence GDN Parity Probe (2026-06-15 11:43Z)

The direct Qwen3.6 GatedDeltaNet parity probe is complete. This does **not**
promote AMDAHL-077; it rules out the gross GDN mapping and packed-boundary
theories for the current K3 failure.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `1027e55a`.
- `compare_qwen36_gdn_parity.py` now supports 2176-token component summaries
  with `kthvalue` percentile reporting, and its top-level diff uses the actual
  `GatedDeltaNet` module output against a SGLang-style reference path.
- Validation passed:
  `py_compile`,
  `ruff check experiments/k3_tests/compare_qwen36_gdn_parity.py`, and
  `git diff --check -- experiments/k3_tests/compare_qwen36_gdn_parity.py`.

Artifacts:

- Layer0, actual K3-length sequence:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_module_layer0_seq2176_20260615T113935Z.json`
- Layer38, actual K3-length sequence:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_module_layer38_seq2176_20260615T113935Z.json`
- Short controls:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_layer0_seq64_20260615T113606Z.json`
  and
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_layer38_seq128_20260615T113632Z.json`

Results:

- Layer0 seq2176 passed: `module_vs_reference_final` max abs `0.046875`,
  mean abs `0.0003519343`, p95 `0.001220703125`; packed projection/conv are
  exact, and the core chunk diff max is `0.0009765625`.
- Layer38 seq2176 passed: `module_vs_reference_final` max abs `0.01171875`,
  mean abs `0.0006630136`, p95 `0.001953125`; packed projection/conv are exact,
  and the core chunk diff max is `0.000732421875`.
- The K3 repro is a single 2176-token sequence with valid labels from rows
  `2047..2174`, not a packed multi-segment case. Do not chase
  `XORL_GDN_PACKED_SEGMENT_LOOP` or copy-boundary segmentation for this K3 miss.

Interpretation:

- Do not spend more time on gross Qwen3.6 GDN weight mapping, grouped-head
  layout, convolution layout, or packed multi-segment boundary hypotheses for
  AMDAHL-077. Single-layer GDN matches the SGLang-style reference within
  BF16-scale tolerance at the real sequence length.
- The full-model failure remains real: residual-stream drift accumulates across
  many layers before the final logprob gap. The next target is accumulated
  numeric/model-path drift in the trainable xorl FLA/MoE/RMSNorm path versus
  SGLang's inference path.
- XoRL's trainable FLA chunk path is materially different from SGLang's fused
  inference-only `chunk_gated_delta_rule_fwd_intra`. Simply importing the
  SGLang chunk is not a training-safe fix because it does not represent the
  trainable backward path used by xorl.
- A local FlashQLA execution check failed with a TileLang
  `libtvm_compiler.so` symbol mismatch. That is a local dependency failure, not
  evidence for or against FlashQLA as a K3 fix.

Next exact target: narrow the trainable FLA/MoE/RMSNorm accumulated
SGLang-vs-xorl residual-stream drift on the AMDAHL-077 one-trace repro, then
rerun the one-trace K3 smoke and the full 32-trace AMDAHL-077 K3 gate. Only
after K3 passes should AMDAHL-077 get a same-workload 4-node MFU replay.

### AMDAHL-077 Real-Input GDN Probe + Live MFU Refresh (2026-06-15 11:53Z)

Current MFU answer:

- The current audited/promotable 4-node OPD MFU is still `0.010405` logical MFU
  (`~1.04%`; valid-token-scaled `0.000295`) from
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
- The newer pack-sweep audit is only 1-node/2-node and does not supersede the
  4-node number: `0.008962` for `replay:1node`, `0.009395` for
  `replay:2node`.
- Live `er-opd-q36-35b-sci-trainer-head` emitted no OPD throughput rows in the
  last 30 minutes when checked at 11:52Z.
- Live `er-opd-q36-mtp-ss-0605c-trainer-head` is a different MTP singleshot
  run, not a §1 OPD-slot promotion. Its latest completed logged step `3089`
  reports `opd_singleshot_mtp_mfu_actual=0.0205736` (`~2.06%`),
  `opd_singleshot_mtp_mfu_useful=0.00207627`, and
  `opd_singleshot_mtp_tflops_per_gpu=20.3473`; the last ten completed steps
  were roughly `2.05%..2.45%` actual singleshot-MTP MFU.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `1c83a09b`.
- `compare_qwen36_gdn_parity.py` now accepts saved component tensors as the
  hidden-state input and can compare the module/reference output against an
  external captured tensor, so the probe can use the exact SGLang or xorl
  activations from the K3-worst trace.
- The harness can run xorl `GatedDeltaNet(mode="fused_recurrent")` for
  diagnostic comparison.
- `fused_recurrent_gated_delta_rule_fwd` now applies the same default scale
  guard as the chunk path (`scale = K**-0.5` when omitted); before this, the
  recurrent diagnostic crashed with `NoneType` scale.
- Validation passed `py_compile`, `ruff check`, and `git diff --check` on the
  touched harness and fused-recurrent kernel files.

Real-input artifacts:

- SGLang input/output, layer0 chunk:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_sglang_layer0_20260615T1144Z.json`
- XoRL input/output, layer0 chunk:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_xorl_layer0_20260615T1145Z.json`
- SGLang input/output, layer38 chunk:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_sglang_layer38_20260615T1145Z.json`
- XoRL input/output, layer38 chunk:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_xorl_layer38_20260615T1145Z.json`
- SGLang input/output, layer0 recurrent:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_sglang_layer0_recurrent_20260615T115157Z.json`
- SGLang input/output, layer38 recurrent:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_parity_realinput_sglang_layer38_recurrent_20260615T115223Z.json`

Results:

- Layer0 with SGLang captured input/output passes. Chunk mode has
  `module_vs_external_output` max/mean abs `0.0078125` / `3.643e-05`;
  recurrent mode has `0.0078125` / `3.717e-05`, so recurrent is not better.
- Layer0 with xorl captured input/output confirms the xorl module matches xorl's
  own captured attention closely: `module_vs_external_output` max/mean abs
  `0.00390625` / `1.029e-06`.
- Layer38 with SGLang captured input/output passes. Chunk mode has
  `module_vs_external_output` max/mean abs `0.03125` / `4.376e-04`;
  recurrent mode has `0.046875` / `4.306e-04`, only a marginal mean improvement
  and worse max.
- Layer38 with xorl captured input/output again shows xorl self-consistency:
  `module_vs_external_output` max/mean abs `0.015625` / `2.072e-05`.
- `FLA_TRIL_PRECISION=tf32` was neutral in this harness for SGLang-input layer0
  and layer38; do not spend a cluster K3 cycle on that flag alone.

Interpretation:

- Do not promote anything from this probe. It is diagnostic-only and leaves the
  current audited/promotable 4-node MFU at `0.010405`.
- The GDN recurrent path is not the missing K3 fix. It now runs, but it does not
  materially reduce the captured SGLang-output gap.
- The gross Qwen3.6 GDN mapping, projection/conv layout, grouped-head chunk
  layout, TF32 chunk precision, packed multi-segment boundary, and recurrent vs
  chunk choice are all low-probability explanations for the AMDAHL-077 K3 miss.
- The next useful target remains accumulated model-path drift after BF16-scale
  attention/GDN differences: RMSNorm amplification, MoE path/accumulation, or a
  later full-attention/logit-path source. A full K3 rerun should wait for a
  candidate change in that path.

### AMDAHL-077 Component Drift Attribution (2026-06-15 11:56Z)

The saved xorl and SGLang component tensors now point away from an RMSNorm
implementation bug and away from MoE as the primary amplifier.

Inputs:

- XoRL full tensor dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/tensors/xorl_components.rank0.pt`
- SGLang tensor dump:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/sglang_component_tensors.pt`
- Checkpoint weights:
  `/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`

RMSNorm finding:

- Qwen3.6 uses the Qwen3.5 MoE zero-centered RMSNorm path, so the checkpoint
  formula is `rms_norm(x, 1 + weight, eps=1e-6)`.
- Applying that formula to xorl saved tensors reconstructs xorl
  `input_norm`/`post_attention_norm` exactly to BF16 storage precision.
- Applying the same formula to SGLang saved tensors leaves only small mean
  residuals, typically `~5e-4..1.6e-3`, far below the cross-engine norm deltas.
  That is not large enough to explain the K3 miss by itself.

Amplification pattern at the K3-worst row `2069`:

| layer | residual/input delta before post-attn norm | post-attn norm delta | amplification | attention delta | mlp delta | layer-output delta |
|---|---:|---:|---:|---:|---:|---:|
| 0 | `4.78e-05` | `2.12e-03` | `44.3x` | `4.99e-05` | `6.64e-05` | `8.64e-05` |
| 10 | `3.29e-04` | `1.09e-02` | `33.0x` | `1.88e-04` | `1.83e-04` | `3.84e-04` |
| 20 | `3.53e-03` | `9.90e-02` | `28.0x` | `1.33e-03` | `1.92e-03` | `3.80e-03` |
| 30 | `8.00e-03` | `1.55e-01` | `19.4x` | `2.88e-03` | `4.08e-03` | `8.46e-03` |
| 38 | `2.72e-02` | `2.57e-01` | `9.45x` | `8.55e-03` | `1.71e-02` | `3.20e-02` |
| 39 | `3.11e-02` | `2.29e-01` | `7.37x` | `5.72e-03` | `2.31e-02` | `3.94e-02` |

MoE finding:

- In the sampled layers, the MoE/MLP block mostly compresses the norm input
  delta rather than amplifying it. At row `2069`, `mlp_delta / post_norm_delta`
  is only about `0.017..0.101` across layers `10..39`.
- The layer-output delta is mostly residual carry-forward plus a smaller MLP
  contribution. The internal equations `mlp ~= experts + shared_expert_weighted`
  and `layer_output ~= post_attention_residual + mlp` remain consistent to BF16
  storage/reduction tolerance on both engines.

Interpretation:

- Do not treat this as a data-pipeline feed problem. Cached server-only replay
  and component dumps already show the model path drifting before any external
  feeding bottleneck would matter.
- Do not chase a plain xorl RMSNorm implementation bug. XoRL's zero-centered
  RMSNorm reconstruction is exact; SGLang's saved norm tensors are close enough
  to the same formula that the huge cross-engine norm deltas are explained by
  input/residual differences being amplified by RMSNorm.
- The next candidate needs to reduce the upstream residual/attention differences
  before repeated norms amplify them, or prove a later logit-path issue after
  the sampled layer-output drift. A K3 rerun without such a candidate is wasted.

### AMDAHL-077 GDN Input-Sensitivity Closure (2026-06-15 12:08Z)

This pass asks whether the sampled GatedDeltaNet layers themselves turn the
captured `input_norm` drift into the captured attention drift, or whether there
is still a local GDN formula mismatch worth fixing.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `3c2cbad6`.
- New diagnostic:
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/diagnose_qwen36_gdn_input_sensitivity.py`.
  It loads saved xorl/SGLang component tensors, recomputes Qwen3.6 GDN from
  each side's captured `input_norm`, and compares recomputed deltas to captured
  attention deltas.
- Validation passed:
  `py_compile`, `ruff check`, `git diff --check`, and a real CUDA tensor rerun
  on layers `0,10,20,30,38`, row `2069`.

Artifacts:

- Active-branch reproducible run:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_input_sensitivity_active_layers0_10_20_30_38_row2069_20260615T120524Z.json`.
- Sibling full diagnostic run with the wider component schema:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_input_sensitivity_layers0_10_20_30_38_row2069_20260615T120254Z.json`.
- TP8 out-proj arithmetic screen:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_input_sensitivity_tp8_layers0_10_20_30_38_row2069_20260615T120642Z.json`.
- Raw SGLang TP-rank closure screen for layer0:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_gdn_input_sensitivity_tp8_rankdump_layer0_row2069_20260615T120735Z.json`.

Row-2069 active-branch result summary:

| layer | row input_norm max delta | captured attention max delta | xorl candidate recompute vs xorl captured | SGLang candidate recompute vs xorl captured |
|---|---:|---:|---:|---:|
| 0 | `0.0` | `0.0009765625` | `0.000244140625` | `0.000244140625` |
| 10 | `0.130859375` | `0.00537109375` | `0.000244140625` | `0.00048828125` |
| 20 | `0.5` | `0.01171875` | `0.000244140625` | `0.0003204345703125` |
| 30 | `0.5224609375` | `0.068359375` | `0.000244140625` | `0.001953125` |
| 38 | `0.7578125` | `0.15625` | `0.0009765625` | `0.001953125` |

Interpretation:

- Recomputed GDN from each side's captured `input_norm` explains the captured
  attention deltas to BF16-scale closure on the K3-worst row. This points away
  from a local GDN projection/conv/core/norm/out-proj formula fix.
- Layer0 has identical captured `input_norm` at row `2069` but still has a
  `0.0009765625` attention delta. The raw SGLang TP-rank dump sum exactly
  matches SGLang's captured attention (`dump_out_proj_sum_minus_reference=0.0`),
  while offline TP8 recompute is worse (`0.00390625`). So a simple
  TP-chunked-out-proj emulation is not the fix either.
- A durable rank-local follow-up below supersedes the earlier scratch suspicion
  that `linear_attn.in_proj_qkvz` was mismatched: `qkvz` is exact against the
  SGLang dump on the sampled layers/ranks. SGLang's fused `ba` projection differs
  from XoRL's split `b_proj`/`a_proj`, but making `ba` merged is not enough to
  close the attention residual and is not a K3 fix.
- The next useful target is the earliest small serving-runtime/model-arithmetic
  residual drift that repeated RMSNorm amplifies, or a later logit-path proof.
  Do not launch another full K3 pod until there is a candidate in that path.

### AMDAHL-077 SGLang TP-Intermediate Projection Correction (2026-06-15 12:19Z)

This pass turns the raw SGLang TP-rank intermediate read into a reusable offline
analyzer and rejects the obvious merged-projection code candidate before any K3
pod spend.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `a8aca956`.
- New diagnostic:
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/diagnose_qwen36_sglang_tp_intermediates.py`.
  It loads the raw SGLang `Pass00002.pt` rank dumps, recomputes TP-local
  Qwen3.6 GDN projections/core/norm/out-proj partials from HF weights, and
  compares split-BA versus merged-BA recompute against SGLang and XoRL captured
  attention.
- Validation passed: `py_compile`, `ruff check`, `git diff --check`, plus real
  CUDA artifact generation for layers `0,10,20,30,38`, row `2069`.

Artifacts:

- Main multi-layer artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_sglang_tp_intermediates_layers0_10_20_30_38_row2069_20260615T121827Z.json`.
- Focused layer0 artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_sglang_tp_intermediates_layer0_row2069_20260615T121751Z.json`.

Durable summary:

- `max_rank_xorl_split_qkvz_minus_live=0.0` and
  `max_rank_merged_qkvz_minus_live=0.0` across the sampled layers/ranks. Do not
  chase a QKVZ loader/order bug next.
- `max_rank_xorl_split_ba_minus_live=0.0625`, while
  `max_rank_merged_ba_minus_live=0.0`. SGLang's fused `ba` projection is exactly
  reproduced by a single merged GEMM, but XoRL's split `b_proj`/`a_proj` is only
  BF16-close.
- Despite that, merged-BA does not close the sampled attention residual:

| layer | live TP out-proj sum vs SGLang attention | split-BA recompute vs SGLang attention | merged-BA recompute vs SGLang attention | merged minus split | XoRL captured vs SGLang attention |
|---|---:|---:|---:|---:|---:|
| 0 | `0.0009765625` | `0.0009765625` | `0.0009765625` | `0.0001220703125` | `0.0009765625` |
| 10 | `0.000244140625` | `0.0003662109375` | `0.0003662109375` | `0.0001220703125` | `0.00537109375` |
| 20 | `0.0009765625` | `0.0009765625` | `0.0009765625` | `0.00048828125` | `0.01171875` |
| 30 | `0.00048828125` | `0.001953125` | `0.001953125` | `0.000244140625` | `0.068359375` |
| 38 | `0.015625` | `0.015625` | `0.015625` | `0.0009765625` | `0.15625` |

Interpretation:

- A merged `b/a` projection path would better emulate one SGLang intermediate,
  but it is not a sufficient training-engine fix for the K3 failure. Do not
  implement or K3-test merged-BA as a standalone candidate.
- The remaining useful K3 target is either deeper TP-local serving arithmetic
  inside the GDN core/norm/out-proj path that is not reproduced by the offline
  checkpoint recompute, or a later full-attention/logit-path proof. The next pod
  launch should carry a concrete candidate in one of those paths.

### AMDAHL-077 Offline Logit-Path Attribution (2026-06-15 12:28Z)

This pass adds the later logit-path proof without launching another pod. It
scores the saved SGLang and XoRL component dumps through the same reconstructed
final zero-centered RMSNorm plus the row-sharded OPD teacher LM head.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `f314eac7` with new diagnostic
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/diagnose_qwen36_logit_path.py`.
- Validation passed: `py_compile`, `ruff check`, and `git diff --check`.
- BF16 same-scorer artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_logit_path_row2_top5_offsets_bf16_20260615T1227Z.json`.
- FP32 exact-row stability artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_logit_path_row2_top5_exact_fp32_20260615T1228Z.json`.

Results on the K3-worst token `train.parquet:rg0:row2`, output position `22`,
absolute position `2070`, scored row `2069`, token `17880`:

- The local BF16 scorer on XoRL hidden reproduces the emitted XoRL logprob
  exactly: `-3.597686767578125`.
- The same scorer on SGLang hidden gives target logprob `-2.5375614166259766`;
  local XoRL-minus-SGLang target logprob delta is `-1.0601253509521484`.
- The FP32 exact-row pass is stable: local delta `-1.091294288635254`.
- This is the same order as the reported generation-reference K3 gap
  (`xorl_minus_sglang=-0.9416682720184326`), while the loss-path reference
  delta remains tiny (`~0.0103`). The scorer/lm-head extraction path is therefore
  not the primary K3 blocker.
- Hidden drift is already present before final scoring: at row `2069`,
  `layer39.layer_output` XoRL-minus-SGLang max/mean/rms abs is
  `0.1669921875 / 0.0394028127 / 0.0491390340`; after final norm it amplifies
  to max/mean/rms abs `2.1015625 / 0.4916196764 / 0.6162380576`.

Interpretation:

- Do not spend another K3 pod on lm-head FP32, loss extraction, final RMSNorm, or
  a local scoring-path rewrite. The saved XoRL hidden plus current LM-head path
  already reproduces the emitted XoRL logprob.
- The remaining blocker is upstream hidden-state divergence before final norm,
  amplified by repeated RMSNorm and final norm. The next engine candidate should
  target the earliest small serving-runtime/model-arithmetic residual drift
  still visible after the GDN input-sensitivity and TP-intermediate closures, or
  produce a similarly direct proof for final full-attention/layer-39 arithmetic.
  Do not rerun full static K3 until such a candidate exists.

### AMDAHL-077 Offline Full-Attention Path Attribution (2026-06-15 12:37Z)

This pass closes the later layer-39 full-attention proof without launching
another pod. It reconstructs Qwen3.6 layer `39` from saved XoRL and SGLang
component dumps using checkpoint `q_proj`/`k_proj`/`v_proj`/`o_proj`, q/k RMSNorm,
XoRL RoPE helpers, causal attention, the attention output gate, the residual
add, and post-attention RMSNorm.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `15e0adb1` with new diagnostic
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/diagnose_qwen36_full_attention_path.py`.
- Validation passed: `py_compile`, `ruff check`, `git diff --check`, BF16 SDPA
  GPU artifact generation, and FP32 eager GPU artifact generation.
- BF16 SDPA artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer39_top5_bf16_sdpa_20260615T1235Z.json`.
- FP32 eager artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer39_top5_fp32_eager_20260615T1237Z.json`.

Results on the K3-worst row `2069`:

- XoRL captured attention is reproduced at BF16 scale. BF16 SDPA
  recomputed-minus-captured attention on row `2069` has max/mean/rms abs
  `0.0009765625 / 0.00001644 / 0.00006522`; FP32 eager is also small at
  `0.0028348 / 0.0002531 / 0.0003341`.
- SGLang captured attention is also close enough to make the formula diagnostic
  valid. BF16 SDPA row `2069` recomputed-minus-captured attention has
  max/mean/rms abs `0.03125 / 0.0001866 / 0.0007614`; FP32 eager is
  `0.017473 / 0.0003279 / 0.0005781`.
- The cross-engine attention delta is explained by the recomputed path, not by
  a missing layer-39 arithmetic term. BF16 SDPA row `2069` captured
  XoRL-minus-SGLang attention mean abs is `0.0057213`, while recomputed
  XoRL-minus-SGLang attention mean abs is `0.0057310`; their delta residual has
  mean abs `0.0001920`. FP32 eager agrees: captured mean abs `0.0057213`,
  recomputed mean abs `0.0057092`, delta residual mean abs `0.0004217`.
- The large row `2069` drift is already upstream of layer-39 full attention:
  `layer_input` XoRL-minus-SGLang mean abs is `0.032037`, `input_norm` mean abs
  is `0.124005`, attention mean abs is only `0.005721`, and
  `post_attention_norm` is then amplified to mean abs `0.228897`.

Interpretation:

- Do not chase layer-39 full attention, q/k norm, RoPE, attention output gate,
  o-proj, residual add, or post-attention RMSNorm as the next standalone K3 fix.
  The saved tensors plus checkpoint reconstruction show that layer-39 attention
  propagates the existing hidden drift rather than creating the K3 gap.
- The remaining K3 blocker is earlier residual-stream divergence before layer
  `39`, with repeated norms amplifying it. The next useful target is the earliest
  small drift still visible before the sampled late layers, especially the live
  serving-runtime arithmetic/intermediate gap around linear-attention layers
  that the GDN input-sensitivity and TP-intermediate probes narrowed but did not
  turn into a patch. No same-workload 4-node replay is warranted until the
  AMDAHL-077 K3 gate passes.

### AMDAHL-077 Offline Layer-Transition Drift Audit (2026-06-15 12:42Z)

This pass audits the saved XoRL and SGLang component tensors for residual-add,
MLP-add, layer handoff, and norm-amplification accounting. It launched no pods
and still does **not** promote AMDAHL-077.

Tooling update:

- Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
  `e4a05abd` with new diagnostic
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/diagnose_qwen36_layer_transition_drift.py`.
- Validation passed: `py_compile`, `ruff check`, `git diff --check`, and a real
  offline run against the saved row2 component tensors.
- Artifact:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/q36_layer_transition_drift_labels_row2069_20260615T1239Z.json`.

Key row `2069` result:

- The sampled layer algebra is clean to BF16-scale residuals. For layers
  `0,10,20,30,38,39`, `delta(post_attention_residual)` closes against
  `delta(layer_input) + delta(attention)` with row-2069 mean residuals
  `3.66e-05`, `4.35e-05`, `9.45e-05`, `1.03e-04`, `3.65e-04`, `4.11e-04`.
  `delta(layer_output)` closes against `delta(post_attention_residual) +
  delta(mlp)` with row-2069 mean residuals `3.66e-05`, `5.12e-05`,
  `1.15e-04`, `1.18e-04`, `5.28e-04`, `5.97e-04`.
- The layer `38 -> 39` handoff is exact on both sides:
  `model.layers.39.layer_input - model.layers.38.layer_output` has zero
  max/mean/rms for XoRL and SGLang.
- The row-2069 residual-stream delta is small at sampled layer `10` and already
  much larger by sampled layer `20`: layer-input mean abs moves from
  `0.000301` at layer `10` to `0.003985` at layer `20`, then `0.008221` at
  layer `30`, `0.027379` at layer `38`, and `0.032037` at layer `39`.
- RMSNorm amplifies existing residual drift rather than creating a formula bug:
  row-2069 post-attention-norm/post-attention-residual mean-abs ratios are
  `44.31x`, `32.97x`, `28.01x`, `19.42x`, `9.44x`, `7.37x` for layers
  `0,10,20,30,38,39`.
- MLP/expert deltas are downstream and smaller than norm amplification on the
  sampled rows: row-2069 `mlp_delta/post_attention_norm_delta` is only
  `0.0168..0.1010` for layers `10..39`.

Interpretation:

- Do not chase sampled-layer residual-add formulas, MLP-add formulas, or the
  `38 -> 39` layer-output handoff as standalone fixes.
- The next diagnostic target is denser activation/component coverage between
  sampled layers `10` and `20` (and, secondarily, `20` to `30`) on the same
  one-trace repro, or an engine-side candidate that specifically changes the
  earlier residual-stream arithmetic before layer `20`. A full K3 rerun or
  same-workload 4-node replay is still premature.

### AMDAHL-077 Dense Layer-10-20 Component Audit (2026-06-15 13:06Z)

This pass ran the denser coverage requested by the 12:42Z audit. It used one
standalone SGLang tensor-dump pod on `research-common-h100-077` and one
standalone XoRL static-replay pod on `research-common-h100-092`; both pods were
cleaned up. It still does **not** promote AMDAHL-077 and does not touch
`er-opd-q36-35b-sci`.

Tooling update:

- Added launcher
  `/home/apanda/xorl-opd-repeat2-diagnostics/experiments/k3_tests/launch_sglang_debug_dump.py`
  to launch only SGLang tensor dumps from an existing static trace. It is pushed
  on engine branch `codex/opd-repeat2-diagnostics-20260615` at commit
  `b5bada7d`.
- The first 077 launcher attempt wrote the SGLang artifact and rank dumps, then
  failed after artifact creation because the wrapper expected
  `metadata.dump_file` while this artifact stores `config.dump_file`. The
  pushed wrapper accepts both layouts, and validation passed: `py_compile`,
  `ruff check`, and `git diff --check`.
- The normalized SGLang tensor bundle was prepared from `Pass00003` without
  relaunching SGLang.

Artifacts:

- SGLang dense reference:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_layers10_20_20260615T1250Z/`.
- SGLang normalized tensor bundle:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_layers10_20_20260615T1250Z/sglang_component_tensors.pt`.
- XoRL dense component dump and one-trace K3 replay:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/`.
- Full tensor compare:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/xorl_rank0_vs_sglang_component_tensor_compare.json`.
- Dense layer-transition audit:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/q36_layer_transition_drift_layers10_20_row2069_20260615T1302Z.json`.

K3 status:

- The one-trace XoRL replay still fails exactly as expected:
  `mean_k3=0.005890282`, `p95=0.002430974`, `max_k3=0.622587`, worst token
  `train.parquet:rg0:row2` position `22`, token `17880`. This is a diagnostic
  replay, not a gate pass.

Dense row `2069` result:

- Layer-output drift is modest through layer `14`, then jumps at layer `15`:
  row-2069 `layer_output` mean abs is `0.0008016` at layer `14`,
  `0.0032233` at layer `15`, `0.0034440` at layer `16`, and `0.0039853` at
  layer `19`.
- The layer-15 jump is in the MoE/MLP contribution, not residual-add algebra:
  row-2069 layer-15 `experts` mean abs is `0.0030646`, `mlp` mean abs is
  `0.0030937`, and `layer_output` mean abs is `0.0032233`; meanwhile both
  engines close `layer_output - post_attention_residual - mlp` to BF16-scale
  noise (`~8.7e-05` SGLang mean residual, `~3.9e-05` XoRL mean residual).
- Norm amplification remains real and large after the jump: row-2069
  `post_attention_norm_delta/post_attention_residual_delta` is `29.0x` at
  layer `15`, `26.7x` at layer `16`, `26.7x` at layer `17`, `26.0x` at
  layer `18`, `25.3x` at layer `19`, and `28.0x` at layer `20`.
- The dumps do not include router top-k / expert-choice metadata, so do not
  claim routing is proven yet. The narrow next target is layer-15 expert/MLP
  arithmetic or adding router/expert-choice capture around layers `14-16` to
  decide whether the expert-output delta is routing, expert-kernel arithmetic,
  or input-sensitive amplification.

Interpretation:

- The current K3 blocker is no longer an undifferentiated "somewhere between
  layer 10 and 20" residual drift. For the worst row, the first large step in
  dense coverage is layer `15` MLP/expert output, followed by repeated RMSNorm
  amplification.
- Do not rerun the layer `10-20` SGLang/XoRL tensor dumps unless the diagnostic
  payload changes. The next useful work is either an engine diagnostic patch
  that records router top-k/expert choices and expert pre/post tensors around
  layers `14-16`, or a narrowly scoped candidate that changes the layer-15 MoE
  arithmetic path and then reruns the one-trace K3 smoke. A full 32-trace K3
  gate or same-workload 4-node replay is still premature.

## AMDAHL-078/079 DP-Replicated No-CP lm-head TP Topology (2026-06-15 10:16Z)

Current audited/promotable 4-node MFU is still unchanged:
`0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.

Engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through commit
`a9753836`. Commit `bba35412` adds the narrow topology enablement: no-CP
lm-head TP can now compose with `data_parallel_replicate_size > 1` by requiring
`lm_head_tp_size` to divide `data_parallel_shard_size` and carving lm-head TP
groups inside each `dp_shard` row. Commit `a9753836` adds startup mesh
provenance for the body FSDP mesh, DP replica/shard meshes, EP-FSDP mesh, and
lm-head mesh, plus stronger HSDP mesh assertions in the parallel-state test. For
a 4-rank HSDP unit test with `dp_replicate=2`, `dp_shard=2`, `lm_head_tp=2`, TP
groups are `[0,1]` and `[2,3]`, while replica groups are `[0,2]` and `[1,3]`.

Validation on the engine checkout:

- `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src python -m py_compile src/xorl/distributed/parallel_state.py tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py`
- `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src python -m pytest tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py -q` -> `9 passed`
- Follow-up provenance patch: `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src python -m pytest tests/distributed/test_lm_head_tp_parallel_state.py -q` -> `3 passed`
- `ruff check src/xorl/distributed/parallel_state.py tests/distributed/test_lm_head_tp_parallel_state.py tests/distributed/test_lm_head_tp_fsdp_e2e.py`
- `ruff check src/xorl/distributed/torch_parallelize.py tests/distributed/test_lm_head_tp_parallel_state.py`
- `git diff --check`

Prepared AMDAHL-078 as the 4-node diagnostic candidate:

- Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-078-OPRD-PREP64-4NODE-DPREP2-LMHEADTP-NOCP-VPKL-FWDPREFETCH-ROWBATCH2.yaml`
- Trainer config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_forwardprefetch_lmheadtp_nocp_dprep2.yaml`
- Key knobs: `data_parallel_replicate_size:2`,
  `data_parallel_shard_size:16`, `lm_head_tensor_parallel_size:8`,
  `expert_parallel_size:8`, `moe_implementation:quack`,
  `ep_dispatch:deepep`, `deepep_num_sms:36`,
  `enable_forward_prefetch:true`, `enable_backward_prefetch:false`,
  rank-local `opd_packed_row_batch_size=2`,
  `opd_kl_backend:vocab_parallel`, `opd_emit_full_vocab_diagnostics:false`,
  `opd_teacher_layer_cache_device_cache:true`, and
  `opd_sharded_head_device_cache:true`.
- YAML parse and `render-control --trainer-nodes 4` passed and showed the
  expected repo roots, `--nnodes 4`, config path, `XORL_GDN_BACKEND=fla`,
  `XORL_SKIP_EMPTY_CACHE_AFTER_OPTIM_STEP=true`, and rowbatch client args.

Prepared AMDAHL-079 as the 2-node live topology smoke:

- Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-079-OPRD-PREP64-2NODE-DPREP2-LMHEADTP-NOCP-VPKL-FWDPREFETCH-ROWBATCH2.yaml`
- Trainer config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_2node_warm009_deepep36_forwardprefetch_lmheadtp_nocp_dprep2.yaml`
- Key knobs: `data_parallel_replicate_size:2`,
  `data_parallel_shard_size:8`, `lm_head_tensor_parallel_size:8`, same Quack +
  DeepEP + forward-prefetch + rank-local rowbatch2 stack as AMDAHL-078.
- YAML parse, `render-control --trainer-nodes 2`, `git diff --check`, and a
  server-side dry-run of the filtered trainer-only manifest with explicit
  `nodeSelector: {node-group:nccl, node-pool:compute}` all passed.

The first live AMDAHL-079 attempt was capacity-gated, but the corrected retry
completed on trainer-only pods with head on `research-common-h100-092` and
worker-1 on `research-common-h100-110`. Server startup proved the intended HSDP
mesh:

- Body mesh: `DeviceMesh((dp_replicate=2, dp_shard=8), 'cuda', stride=(8, 1))`.
- lm-head TP: `lm_head_tp_size=8`, `source_axis=dp_shard`,
  `source_replica=1`, `mesh=(2, 8)`.
- FSDP topology log:
  `lm_head_mesh=shape=(2, 8) names=('replica', 'lm_head_tp') size=16`.

Corrected AMDAHL-079 replay artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-hsdp2node-amdahl079-corrected-rowbatch2-chunk4-warmed-serveronly-3x-20260615T1012Z.jsonl`.
The replay used explicit VP-KL and device-cache overrides:
`opd_kl_backend=vocab_parallel`, `opd_emit_full_vocab_diagnostics=false`,
`fsdp_sharded_lm_head_loss_num_chunks=8`,
`opd_teacher_layer_cache_device_cache=true`,
`opd_sharded_head_device_cache=true`, `opd_oprd_layer_chunk_size=4`,
`opd_packed_row_batch_size=2`, and
`opd_packed_row_batch_scope=rank_local`.

Measured rows after the 80s warmup:

- Mean server F/B: `3.674399s`.
- Mean API wall: `3.942664s`.
- Mean forward/backward/clear-grad: `0.768869s` / `1.572231s` /
  `0.615128s`.
- Mean KL compute: `0.019663s`.
- Loss/KL/hidden: `2.361356` / `2.340025` / `0.0213316`.
- Valid tokens: `515`.
- Approx raw 2-node logical MFU by scaling the AMDAHL-077 proxy denominator:
  `0.011363` (`~1.14%`).

Verdict: AMDAHL-079 is topology-valid and forward/backward-clean, but it is not
a throughput promotion. It is slower than the same-server AMDAHL-077 rank-local
rowbatch2 path (`2.936666s`, raw 2-node logical MFU estimate `~1.42%`) and has a
slightly different loss/KL tuple from that same-server AMDAHL-077 comparison.
Do not spend scarce 4-node time on AMDAHL-078 as the next speed bet unless the
goal is only topology proof; the near-term speed path should stay on the
non-HSDP rank-local rowbatch2 lineage and attack the dominant forward/backward
and clear-gradient costs.

Caveat: this patch now proves local distributed tests plus live 2-node startup
and F/B replay for the HSDP topology. It still does not prove a 4-node MFU win,
and the measured 2-node result argues against promoting this topology as the
next speed lever.

## AMDAHL-080 1-Node HSDP Mesh-Provenance Smoke (2026-06-15 10:16Z)

Current audited/promotable 4-node MFU is still unchanged:
`0.010405` logical MFU (`~1.04%`; valid-token-scaled `0.000295`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.

AMDAHL-080 is the one-node topology-provenance smoke for the same HSDP idea:

- Candidate:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-080-OPRD-PREP64-1NODE-DPREP2-LMHEADTP4-NOCP-VPKL-FWDPREFETCH-ROWBATCH2.yaml`
- Trainer config:
  `/home/apanda/xorl-infra/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_forwardprefetch_lmheadtp_nocp_dprep2.yaml`
- Key knobs: `data_parallel_replicate_size:2`,
  `data_parallel_shard_size:4`, `lm_head_tensor_parallel_size:4`,
  `expert_parallel_size:8`, Quack + DeepEP, forward-prefetch on,
  backward-prefetch off, rank-local rowbatch2.

The initial live startup succeeded and proved the intended mesh shape on
`research-common-h100-110`. Server log:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260615T094637Z-serveronly-configAMDAHL-080-OPRD-PREP64-1NODE-DPREP2-LMHEADTP4-NOCP-VPKL-FWDPREFETCH-ROWBATCH2-er-opd-q36-35b-slots-trainer-head/server.log`.
Relevant log facts: world size 8, body mesh
`DeviceMesh((dp_replicate=2, dp_shard=4), 'cuda', stride=(4, 1))`,
EP mesh `DeviceMesh((ep=8, ep_fsdp=1), 'cuda', stride=(1, 8))`, and
lm-head TP was sourced from `dp_shard` with
`lm_head_tp_size=4`, `source_replica=1`, `mesh=(2, 4)`. The new provenance log
reported `lm_head_mesh=shape=(2, 4) names=('replica', 'lm_head_tp') size=8`.

The first replay did not produce throughput rows and failed with
`aten.mm.default got mixed torch.Tensor and DTensor` in
`src/xorl/ops/loss/opd_streaming_kl.py:92`. That was not the intended HSDP
VP-KL path: the replay command omitted the explicit
`opd_kl_backend=vocab_parallel` override, so it exercised the unsupported
streaming fallback on the DTensor lm-head shard. The AMDAHL-078/079/080 YAMLs
now make the intended VP-KL knobs explicit.

Corrected AMDAHL-080 reruns no longer hit the Tensor/DTensor streaming failure.
They reached real forward/backward with the intended HSDP mesh, but one-node
full64 HSDP is memory-tight:

- Rowbatch2 corrected replay artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp4-hsdp1node-amdahl080-corrected-chunk4-warmed-serveronly-2x-20260615T1000Z.jsonl`.
  It OOMed in Quack backward while allocating `442 MiB`.
- Rowbatch1 corrected replay artifact:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp4-hsdp1node-amdahl080-corrected-rowbatch1-chunk4-warmed-serveronly-2x-20260615T1008Z.jsonl`.
  It OOMed in Quack backward while allocating `198 MiB`.

Verdict: AMDAHL-080 proves HSDP startup and the corrected VP-KL path but rejects
the one-node HSDP full64 fit envelope. It is not an MFU datapoint and should not
be promoted.

Cleanup is complete: local port-forward was closed, trainer control was stopped,
temporary trainer pods were deleted, and the slots stack is back to dispatch +
teacher-smg only. No `er-opd-q36-35b-sci` pods were touched.

## Next Agent Prompt

```text
/goal Continue OPD filler-token throughput from @experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_RUNBOOK.md.
Start in /home/apanda/xorl-opd-prefill. Use the one-node
`er-opd-q36-35b-slots` trainer-server replay loop first; do not launch a full OPD
science run first and do not require 32 H100s to make progress.

Current state: the current audited/promotable 4-node MFU remains `0.010405`
logical MFU (`~1.04%`). The latest raw 2-node candidate estimate is AMDAHL-077
rank-local rowbatch2 at `0.014218` logical MFU (`~1.42%`), but it failed the
OPD-aligned quack+DeepEP SMS36 static gate (`mean_k3=0.001628991 > 0.001`, full
coverage), so it is not promoted and should not get a same-workload 4-node replay
until the K3 repro is fixed. Follow-up K3 controls ruled out lm-head FP32,
router FP32, deterministic FA3, loss-logprob extraction, and a simple
generation-vs-prefill reference-mode swap; the next K3 target is
SGLang-vs-xorl model-forward activation/reference parity on the worst token.
The SGLang tensor-dump reference now exists at
`/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_20260615T1110Z/sglang_component_tensors.pt`;
the matching xorl full component dump and compare are also complete at
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/`,
with concise summary
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_20260615T1120Z/xorl_rank0_vs_sglang_component_tensor_summary.json`.
The compare shows model-forward residual-stream drift accumulating across
sampled Qwen3.6 layers: at the K3-worst compare row `2069`, sampled
`layer_output` mean abs grows from layer0 `8.64e-05` to layer39 `0.03940`, with
largest row-2069 component deltas in late-layer norms/shared-expert-input.
SGLang TP-rank and xorl rank checks ruled out the obvious rank0-only diagnostic
artifact for residual-stream components. Long-sequence and real-input GDN
diagnostics now rule out gross GDN mapping, recurrent-vs-chunk, TF32, and simple
TP8 out-proj chunking as fixes. Recomputed GDN from captured `input_norm`
explains the row-2069 attention drift to BF16-scale closure, so the next target
is the earliest small serving-runtime/model-arithmetic residual drift that
repeated RMSNorm amplifies, or a later logit-path proof. The TP-intermediate
diagnostic commit is `a8aca956`, adding
`experiments/k3_tests/diagnose_qwen36_sglang_tp_intermediates.py`; artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_sglang_tp_intermediates_layers0_10_20_30_38_row2069_20260615T121827Z.json`
shows TP-local `qkvz` projection is exact against SGLang, SGLang's fused `ba`
projection is exactly reproduced by a merged GEMM, but merged `ba` does not
reduce the attention residual enough to be a K3 fix. Do not implement merged-BA
or chase qkvz loader/order as the next standalone candidate. The latest pushed
engine diagnostic commit is `f314eac7`, adding the offline logit-path proof
`experiments/k3_tests/diagnose_qwen36_logit_path.py`; artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_logit_path_row2_top5_offsets_bf16_20260615T1227Z.json`
shows the same reconstructed final RMSNorm plus row-sharded teacher LM head
exactly reproduces XoRL's emitted worst-token logprob (`-3.597686767578125`)
and still gives a same-scorer XoRL-minus-SGLang hidden gap of
`-1.0601253509521484` at row `2069`; FP32 exact-row artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_logit_path_row2_top5_exact_fp32_20260615T1228Z.json`
is stable at `-1.091294288635254`. Do not chase lm-head extraction, final
RMSNorm, or local scoring as the next standalone fix. The latest pushed engine
diagnostic commit is `15e0adb1`, adding
`experiments/k3_tests/diagnose_qwen36_full_attention_path.py`; artifacts
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer39_top5_bf16_sdpa_20260615T1235Z.json`
and
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer39_top5_fp32_eager_20260615T1237Z.json`
show layer-39 full attention also propagates existing hidden drift rather than
creating the K3 gap. At row `2069`, BF16 SDPA recomputed attention closes XoRL
captured attention to max/mean abs `0.0009765625` / `0.00001644`, and the
XoRL-minus-SGLang recomputed attention delta (`0.0057310` mean abs) matches the
captured attention delta (`0.0057213` mean abs). Do not chase layer-39 full
attention, q/k norm, RoPE, attention output gate, o-proj, residual add, or
post-attention RMSNorm as the next standalone fix. Do not rerun the completed
tensor-dump/logit-path/full-attention compares unless changing the diagnostic
itself. The latest pushed engine diagnostic commit is `e4a05abd`, adding
`experiments/k3_tests/diagnose_qwen36_layer_transition_drift.py`; artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_layer_transition_drift_labels_row2069_20260615T1239Z.json`
shows sampled-layer residual-add and MLP-add algebra are clean to BF16 scale,
the `38 -> 39` handoff is exact on both sides, and row-2069 layer-input drift is
small at sampled layer `10` (`0.000301` mean abs) but already much larger by
sampled layer `20` (`0.003985` mean abs). Do not chase residual-add formulas,
MLP-add formulas, or the layer `38 -> 39` handoff as standalone fixes. The next
useful runtime diagnostic was completed at 13:06Z: dense SGLang/XoRL component
coverage for layers `10-20` on the same one-trace repro. SGLang artifact dir:
`/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_layers10_20_20260615T1250Z/`;
XoRL artifact dir:
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/`;
dense transition audit:
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/q36_layer_transition_drift_layers10_20_row2069_20260615T1302Z.json`.
The one-trace replay still fails (`mean_k3=0.005890282`, `p95=0.002430974`,
`max_k3=0.622587`). Dense coverage narrows the first large row-2069 jump to
layer `15`: `layer_output` mean abs moves from `0.0008016` at layer `14` to
`0.0032233` at layer `15`, with layer-15 `experts=0.0030646` and
`mlp=0.0030937`; the layer-output residual algebra closes to BF16-scale noise
on both engines. The dumps do not include router top-k/expert-choice metadata,
so do not claim routing is proven. Do not rerun the completed layer `10-20`
tensor dumps unless changing the diagnostic payload. The next useful work is an
engine diagnostic patch that records router top-k/expert choices and expert
pre/post tensors around layers `14-16`, or a narrowly scoped candidate that
changes layer-15 MoE/expert arithmetic and then reruns the one-trace K3 smoke.
The no-CP
lm-head-TP VP-KL dtype drift is fixed;
AMDAHL-031 every4 SGLang layer-cache replay fits with GPU device cache;
valid-only gather + chunked/streamed OPRD MSE stabilizes the path; SGLang PR #48
real all-layer cache capture now works; the real full64 all-layer cache replay
fits on one node with trainer-side OPRD forward removed. Latest same-workload
chunk sweep: chunk `4` is the best stable one-node setting
(`server_forward_backward_s=4.6591`, loss `2.3550419`), chunk `8` is slower
(`4.9491`), and chunk `16` OOMs in FSDP pre-backward all-gather after one
measured row.

Rejected/negative screens on this same real-cache path: AMDAHL-048 2-node retry
completed at `server_forward_backward_s=4.4441`, only a modest improvement over
one-node. AMDAHL-049 pack2304 is rejected (`5.5459s`, loss `2.3666090`).
AMDAHL-050 no-defrag reduced `clear_gradients_s` to ~0.004s but OOMed on the
third measured request, so keep allocator defrag enabled. AMDAHL-051
`enable_forward_prefetch=true` is neutral/rejected (`4.6506s` vs `4.6591s`, loss
shifted to `2.3562496`). AMDAHL-052 `--repeat-data 2` OOMed before a warmup row.
AMDAHL-053 CPU-GC-skip regressed server/API wall to `5.4222s`/`5.6626s`.
AMDAHL-054 `deepep_num_sms=24` regressed server wall to `4.7599s` and shifted
loss to `2.3569047`, so keep `deepep_num_sms=36`. AMDAHL-055
`moe_grad_reduce_mode=bf16_a2a_fp32_sum` failed engine init because expert FSDP
reduce policy was already `torch.bfloat16`; do not retry it as a one-node screen
without an engine/topology fix. AMDAHL-056 `fsdp_reduce_dtype=bf16` preserved
loss but regressed server/API wall to `4.8815s`/`5.1059s`. AMDAHL-057
`ep_dispatch=alltoall` OOMed during warmup before writing a replay row.
AMDAHL-058 deferred detached loss-report all-reduce to once per
`forward_backward`; it preserved loss but regressed server/API wall to
`4.803969s`/`5.006521s`, so the local reporting-path engine patch was reverted
and should not be repeated/promoted. AMDAHL-059 EP4 DeepEP had a small one-node
wall-time win but shifted KL/loss by about `+0.0181`. AMDAHL-060 2-node
`reshard_after_forward:false` was flat versus AMDAHL-048 (`4.443982s` versus
`4.444137s`). AMDAHL-061 2-node `enable_forward_prefetch=true` was fast
(`3.677469s`, -17.25% server wall versus AMDAHL-048) but shifted same-capture
KL/loss by about `+0.00458`, so treat forward-prefetch as a correctness-risk
speed lever until that drift is explained. AMDAHL-062 split the engine control
into forward and backward manual prefetch and measured backward-only prefetch on
the same 2-node replay. It preserved loss/KL exactly (`2.355041921` /
`2.333743553`) and improved server wall to `4.280844s` versus AMDAHL-048
`4.444137s`, but that is only -3.67% and remains far slower than the rejected
coupled-prefetch row. AMDAHL-063 measured forward-only prefetch on the same
2-node replay. It also preserved loss/KL/hidden exactly and improved server
wall to `4.140186s` (-6.84% versus AMDAHL-048), but remains far slower than
rejected AMDAHL-061 and is not a standalone promotion. AMDAHL-064 repeated the
same capture to 128 datums on the AMDAHL-063 2-node config. It fit and improved
valid/server-token rate to `187.16` tokens/s (`+50.46%` versus AMDAHL-063), but
shifted same-capture loss/KL/hidden by about `+0.000779` / `+0.000775` /
`+0.0000039`, so it is a packing/aggregation diagnostic, not a promotion.
AMDAHL-065 then inserted a no-loss 4096-token separator between the two repeated
copies to preserve the first 64-sample pack boundary. It still shifted the tuple
to loss `2.356476724`, KL `2.335177020`, hidden `0.021299774` and slowed the
fatter-call throughput to `174.79` valid/server-token/s, so the separator
diagnostic is rejected and does not isolate the issue to only boundary co-pack.
AMDAHL-075 rowbatch2 showed the strongest raw trainer replay speed so far on the
same 2-node real-cache path (`server_forward_backward_s=2.831341`, about
`1.47%` raw logical MFU by the AMDAHL-048 denominator), but the packed-row concat
implementation shifted loss/KL/hidden, so it is rejected until the
packed-layout/source-rank invariance problem is fixed. AMDAHL-076 tested
`XORL_GDN_PACKED_SEGMENT_LOOP=true` as the rowbatch correctness fix and was
falsified (`server_forward_backward_s=3.272135`, same invalid loss/KL tuple as
AMDAHL-075 rowbatch2). AMDAHL-077 moved rowbatching to the rank-local runner
slice after each rank selected its original source-rank work. Same-server
no-rowbatch control was `server_forward_backward_s=3.548186`; rank-local
rowbatch2 stability was `2.936666` over 11 measured rows with matching
loss/KL/hidden, so it is correctness-clean relative to that control and implies
about `1.42%` raw 2-node logical MFU. The follow-up static gate failed
(`mean_k3=0.001628991 > 0.001`, `p95=0.003193 < 0.01`), with worst-4 repro
bundle at
`/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/q36-opd077-quack-deepep-sms36-k3-repro-worst4.json`;
debug that before any science/default promotion or 4-node spend.
AMDAHL-078/079/080 add a topology-enablement path for DP-replicated no-CP lm-head
TP: engine branch `codex/opd-repeat2-diagnostics-20260615` is pushed through
commit `a9753836` (`bba35412` topology enablement plus `a9753836` startup mesh
provenance). It allows `data_parallel_replicate_size>1` by grouping lm-head TP
inside `dp_shard`; local distributed validation passed (`9 passed`, focused
provenance follow-up `3 passed`, ruff, diff-check). AMDAHL-078 is
the prepared 4-node candidate (`dp_replicate=2`, `dp_shard=16`,
`lm_head_tp=8`, rank-local rowbatch2, explicit VP-KL and device-cache loss
knobs). Corrected AMDAHL-080 one-node HSDP no longer hits the earlier
Tensor/DTensor streaming fallback once `opd_kl_backend=vocab_parallel` is set,
but full64 OOMs in Quack backward even with rowbatch1/2, so it is a one-node fit
rejection rather than an MFU datapoint. Corrected AMDAHL-079 two-node HSDP fits
and proves live startup/F/B for `dp_replicate=2`, `dp_shard=8`, `lm_head_tp=8`,
but it is slower than AMDAHL-077: `server_forward_backward_s=3.674399`,
`api_wall_s=3.942664`, loss/KL/hidden `2.361356` / `2.340025` / `0.0213316`,
raw 2-node proxy MFU `~1.14%`. Do not promote HSDP topology as the next speed
lever; only run AMDAHL-078 4-node if topology proof is explicitly needed.

Current capacity/cleanup as of 2026-06-15 13:06Z: Kubernetes auth works.
2026-06-15 14:19Z follow-up: no promotion. The captured-input GDN sweep over
SGLang inputs for layers `10/12/13/14` rejects porting SGLang's fused-intra
`chunk_gated_delta_rule` as the next XoRL candidate. Artifact dir:
`/shared/opd-control/er-opd-q36-35b-slots/k3/gdn_parity_sglang_input_layers10_14_20260615T140940Z/`.
For layers `12/13/14`, the XoRL module was slightly closer to captured SGLang
attention than the SGLang-style reference; layer `10` only failed the max
threshold while keeping the same mean-scale conclusion. Layer-11 full-attention
attribution also rejects that path:
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_full_attention_path_layer11_row2069_bf16_sdpa_20260615T141132Z.json`
recomputes attention/post-attention norm from captured inputs and reproduces the
captured drift rather than removing it. Exact R3 route replay remains
diagnostic-only until the SGLang reference is launched with nonempty expert
weights: the current routed trace has nonempty `sglang_routed_experts` but
`sglang_expert_logits=[]`, and the live science SGLang run script has
`--enable-return-routed-experts` but not `--enable-return-expert-logits`. Do not
mutate `er-opd-q36-35b-sci`; if exact-R3 is retried, launch an owned reference
endpoint with `--enable-return-expert-logits` and regenerate the trace. Engine
branch `/home/apanda/xorl-opd-repeat2-diagnostics` now has a narrow exact-R3
fix pending packaging: `_shard_and_slice_batches()` slices
`routed_expert_logits` with the same `_r3_datum_offset/_r3_datum_count` metadata
as `routed_experts`, preventing selected-expert and weight payloads from
desynchronizing after local sharding. Validation passed:
`PYTHONPATH=src python -m py_compile src/xorl/server/runner/runner_dispatcher.py tests/server/runner/test_runner_dispatcher.py`,
`PYTHONPATH=src ruff check ...`, and
`PYTHONPATH=src pytest -q tests/server/runner/test_runner_dispatcher.py tests/server/orchestrator/test_request_processor.py -q`
(`26 passed`). This is correctness infrastructure for a future exact-R3 gate,
not an MFU promotion.

Current capacity/cleanup as of 2026-06-15 14:19Z: Kubernetes auth works.
AMDAHL-080/079 corrected trainer pods, the standalone AMDAHL-077 full K3 pod,
the reduced diagnostic pods, and the temporary SGLang tensor-dump pod were
cleaned up; the dense 10-20 SGLang and XoRL diagnostic pods on `077` and `092`
were also cleaned up; the TP-intermediate, logit-path, full-attention, and
layer-transition follow-ups apart from those dense tensor dumps were offline-only;
slots cleanup is complete with only dispatch + teacher-smg remaining, and the
separate `er-opd-q36-35b-sci` stack was not touched. Next target is the
AMDAHL-077 K3 repro, not a 4-node replay: either launch an owned SGLang
reference with `--enable-return-expert-logits` for exact-R3 trace regeneration,
or add deeper residual/component coverage before layer-15 inputs and produce a
candidate that reduces the upstream hidden drift. Run the same-workload 4-node
rowbatch replay only after the K3 gate passes. In parallel, keep attacking
dominant model forward/backward and clear-gradient costs. The measured evidence
does not support blaming only the external data pipeline: even server-only static
replays with cached inputs stay near 1% logical MFU because the trainer's
internal execution units are too small/communication-heavy.

Do not promote no-CP VP-KL/lm-head-TP, layer-cache flags, pack2304,
no-defrag replay knobs,
no-CP forward-prefetch, repeat-data=2, repeat2 separator, prepare128,
CPU-GC-skip, DeepEP SMS24, one-node MoE BF16-a2a reduce-hook, BF16 FSDP
reduce-scatter, alltoall dispatch, deferred loss-report reduce, EP4 DeepEP,
`reshard_after_forward:false`, 2-node coupled-prefetch, forward-only prefetch,
backward-only prefetch, executor-global packed-row batching, or rank-local
packed-row batching to science defaults until static/K3 and the same-workload
4-node real-cache throughput gate pass. The current rank-local rowbatch K3 status
is failed, not pending. Also do not repeat the already-falsified K3 quick fixes:
lm-head FP32, router FP32, deterministic FA3, loss-logprob extraction, final
RMSNorm/lm-head scoring, qkvz loader/order, merged BA alone, or a
generation-vs-prefill reference-mode swap. 2026-06-15 14:51Z exact-R3 update:
an owned SGLang route refresh with `--enable-return-expert-logits` produced real
nonzero route weights
(`/shared/opd-control/er-opd-q36-35b-slots/k3/r3_exact_refresh_20260615T1437Z/q36-opd077-row2-routed-trace-expertweights.json`;
prefill weights `696320/696320` nonzero, generation weights `696000/696000`
nonzero, generation mismatches `0`), but exact-R3 is REJECTED as the next parity
bridge. Prefill-source exact replay
(`/shared/opd-control/er-opd-q36-35b-slots/k3/r3_exact_k3_20260615T1442Z/k3_r3_exact_result.json`)
failed with `mean_k3=158.892269`, `p95=762.595881`, `max=9673.068879`;
generation-source exact replay
(`/shared/opd-control/er-opd-q36-35b-slots/k3/r3_generation_exact_k3_20260615T1449Z/k3_r3_generation_exact_result.json`)
failed worse with `mean_k3=307.219859`, `p95=721.843827`,
`max=27261.159191`. Both fail at row2 position `22` token `17880`, and the
diagnosis says shift `0` is best while SGLang prefill/generation
self-consistency is small (`mean_abs=0.031674`, `max_abs=1.035477`). Forcing
SGLang routes/weights into the current XoRL hidden stream makes this trace much
worse, so do not spend more K3 pods on exact-R3 without a new model-forward
hypothesis. 2026-06-15 14:55Z MFU/feed-rate clarification: current audited
promotable 4-node logical MFU remains `0.010405075057463931` (`~1.04%`;
valid-token-scaled `0.0002947112240841216`). This is not simply external
data-loader starvation: the denominator audit is mostly server-resident
forward/backward (`server_forward_backward_s=5.866114` versus
`api_wall_s=6.216350`), and cached-input server-only replays stay near this
MFU scale. Trainer feed-rate work did find useful levers, especially
rank-local rowbatch in AMDAHL-077 (`~17%` 2-node server-wall win and raw
MFU estimate around `0.0142`), but they are blocked from promotion by the
static K3/model-forward parity failure. Do not turn the next step into another
data-volume or exact-R3 run; get a one-trace K3-moving upstream hidden/residual
candidate, or a correctness-cleared batching path, before spending more 4-node
capacity. Record every run and failed
hypothesis back into this runbook and
/shared/apanda/opd_throughput_science_channel.md.

2026-06-15 15:08Z router-policy diagnostic update: engine branch
`/home/apanda/xorl-opd-repeat2-diagnostics` is pushed through commit
`9b1f9616` with two default-off diagnostics: `XORL_GDN_FUSED_PROJECTIONS=1`
and `XORL_MOE_ROUTER_TOPK_POLICY={logits,stable_low_id,tie_low_id,tie_high_id}`.
Local validation passed for the router/GDN tests, py_compile, ruff, and
diff-check. The fused GDN projection path was screened offline on real
SGLang-captured inputs and is rejected as a standalone K3 candidate: layer-10
mean `module_vs_external_output` only moved from `0.0000673366` to
`0.0000672519`, and layer-14 from `0.0000807448` to `0.0000804435`. The
tie-high router policy was then K3-smoked on the minimized row2 trace because it
can recover the missing high-ID expert seen in the layer-15 row2069 router
report, but it failed and regressed: artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/router_tiehigh_row2_20260615T1504Z/k3_result.json`
has `mean_k3=0.010588831`, `p95=0.002059908`, `max=1.313526992` with the same
worst token (`train.parquet:rg0:row2`, position `22`, token `17880`). Do not
spend worst-4/full K3 pods on `tie_high_id`; simple deterministic/tie-biased
router top-k is not the model-forward parity fix.
```

2026-06-15 15:28Z router-margin and layer-scoped fp32 update: current audited
promotable 4-node logical MFU remains `0.010405075057463931` (`~1.04%`;
valid-token-scaled `0.0002947112240841216`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
The newer real-cache pack sweep is not a 4-node promotion; it has only 1-node
`0.008961757` and 2-node `0.009395210`.

New offline diagnostic
`/shared/opd-control/er-opd-q36-35b-slots/k3/q36_router_margin_delta_row2069_projection_20260615T151507Z.json`
attributes the layer-15 router flip on row2069 to upstream hidden drift rather
than gate-weight loading or router matmul arithmetic. For the key missing/added
expert pair (`148` vs `49`), the reference margin is about `+0.06045` fp32 while
the XoRL candidate margin is only `+0.00555` fp32; BF16 then rounds it to a
tie/zero-margin selection and loses expert `148`. The dominant harmful term is
layer-15 `post_attention_norm` (`hidden_delta_margin_projection_fp32=-0.05491`);
layer-14 input/norm projections are the only smaller coherent contributors.
Layer 13 and earlier are much smaller in the projection table, so a broader
router-fp32 sweep would be weakly motivated without a new hypothesis.

Engine branch `/home/apanda/xorl-opd-repeat2-diagnostics` now has default-off
diagnostic plumbing pending packaging for `XORL_MOE_ROUTER_FP32_LAYERS`, with
`experiments/k3_tests/launch_k3_test.py --router-fp32-layers` and pod-template
env export. Local validation before K3 passed `py_compile`, `ruff check`,
`git diff --check`, and `PYTHONPATH=src pytest -q tests/models/test_topk_router.py`
(`21 passed`). One-trace K3 smokes on `research-common-h100-092` improved the
known row2 failure but still failed the strict mean gate:

- `XORL_MOE_ROUTER_FP32_LAYERS=15`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/routerfp32l15_row2_20260615T1520Z/k3_result.json`
  had mean `0.002720670`, p95 `0.002430185`, max `0.242282049`; worst token was
  still row2 position `22`, token `17880`, with XoRL logprob `-3.279907` vs
  SGLang `-2.656018`.
- `XORL_MOE_ROUTER_FP32_LAYERS=14-15`:
  `/shared/opd-control/er-opd-q36-35b-slots/k3/routerfp32l14_15_row2_20260615T1524Z/k3_result.json`
  had mean `0.00202256`, p95 `0.002690`, max `0.134542887`; worst token remained
  row2 position `22`, token `17880`, with XoRL logprob `-3.133524`.

Interpretation: layer-scoped router fp32 is useful as a diagnostic because it
moves the failure in the expected direction (`0.622587` baseline max down to
`0.242282` and then `0.134543`), but it is not a promotable fix. The forced
SGLang layer-15 routing diagnostic remains the upper bound (`max=0.01682`) and
is itself reference-routing-only, so the real fix must reduce the upstream
hidden/residual drift before the layer-15 router input. Do not spend worst-4,
full K3, or 4-node replay capacity on router-fp32 layer scopes. All K3 pods from
this ladder were cleaned up; no science or slots stack was restamped.

2026-06-15 15:45Z GDN beta-rounding update: no promotion and no 4-node replay.
Current audited/promotable 4-node logical MFU remains exactly
`0.010405075057463931` (`~1.04%`) from
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`.
The newer real-cache pack sweep is still only 1-node/2-node and does not
supersede this value. Repeat-data AMDAHL-041 remains the measured feed-rate
ceiling for "just feed more work" at 4 nodes: repeat1/2/4/8 reconstructed
logical MFU `0.0142401777`, `0.0162517014`, `0.0171767331`, and
`0.0183050777`. That is useful signal but saturates under `2%`, so the path to
`10%` needs either a correctness-cleared global batching/topology change or a
real per-token/model-engine speedup, not more samples alone.

Direct SGLang runtime evidence now shows `fused_gdn_gating` computes `beta` as a
bf16-rounded sigmoid and then returns it as fp32. Artifact:
`/shared/opd-control/er-opd-q36-35b-slots/k3/gdn_parity_sglang_input_layers10_14_beta_round_20260615T1539Z/sglang_fused_gdn_gating_dtype_check.json`
reports `max_abs_beta_minus_bf16_rounded_sigmoid=0.0` and
`max_abs_beta_minus_full_fp32_sigmoid=0.0019377470`. Engine diagnostics branch
`/home/apanda/xorl-opd-repeat2-diagnostics` is pushed through commit
`742022f4` (`k3: align GDN parity beta gating with SGLang`) so the offline
GDN parity reference now uses the same beta rounding. Validation passed:
`py_compile`, `ruff check`, `git diff --check`, and
`PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src pytest -q tests/experiments/test_qwen36_gdn_parity.py`
(`1 passed`).

The corrected reference did not create a promotable GDN fix. Patched layer-14
captured-input parity artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/gdn_parity_sglang_input_layers10_14_beta_round_20260615T1539Z/layer14_sginput_vs_sgattention_beta_round.json`
passes local reference closure but leaves the external SGLang-output gap at the
same scale: `module_vs_external_output.mean_abs=8.044e-05`,
`reference_vs_external_output.mean_abs=8.023e-05`, and `core.mean_abs=7.84e-08`.
Interpretation: the beta-rounding update fixes the diagnostic model of SGLang's
kernel, but local GDN projection/conv/core/norm/out-proj arithmetic is still
rejected as the missing K3/MFU promotion path. Next target remains upstream
hidden/residual parity before the layer-15 router input, or a separate
correctness-cleared batching path; do not spend worst-4/full K3 or 4-node replay
capacity on beta-rounded GDN alone.

2026-06-15 15:57Z Quack+DeepEP force-generic update: no promotion and no
4-node replay. Engine diagnostics branch
`/home/apanda/xorl-opd-repeat2-diagnostics` is pushed through commit `1f3dc161`
(`k3: thread DeepEP diagnostic envs through launcher`), which adds default-off
K3 launcher/template plumbing for `XORL_QUACK_DEEPEP_FORCE_GENERIC` and
`XORL_DEEPEP_PARITY_DIAGNOSTIC*` envs plus focused validation coverage. Local
validation passed: `py_compile`, `ruff check`, `git diff --check`, focused
render/helper test `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src
pytest -q tests/experiments/test_launch_k3_deepep_env.py` (`2 passed`). The
broad `tests/experiments/test_k3_static_traces.py -k 'deepep_parity or
xorl_pod_template'` remains module-skipped in this tree (`0 collected / 1
skipped`) because the full experiment harness import set is incomplete here.

The new force-generic EP8 one-trace K3 diagnostic rejected the fused
Quack+DeepEP no-permute path as the row2 K3 fix. Artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/force_generic_row2_20260615T1554Z/k3_result.json`
used `XORL_QUACK_DEEPEP_FORCE_GENERIC=1` on `train.parquet:rg0:row2` and
returned the same K3 profile as the original baseline: mean `0.005890282`,
p95 `0.002430974`, max `0.622587456`, worst token position `22`, token
`17880`, XoRL logprob `-3.597687` vs SGLang `-2.656018`. The all-rank DeepEP
reference diagnostic emitted 24 records (`post_dispatch`/`post_compute`/
`post_combine` for ranks `0-7`); all post-combine records had
`quack_deepep_no_permute=false` and `result_reference.status=pass` with
`max_abs=0.000440076`, `mean_abs=3.3819e-05`, and `p95_abs=0.000100011` on the
sampled 64-row reference. Interpretation: generic DeepEP combine/reference is
locally sane, but disabling no-permute does not move K3 at all. Continue
targeting upstream hidden/residual drift before layer-15 router input, or a
source-rank-preserving batching design; do not spend worst-4/full K3 or 4-node
replay capacity on force-generic/no-permute alone. Pod cleanup was confirmed;
no science stack was touched.

2026-06-15 16:34Z dense layer-0-10 component audit: no promotion and no
4-node replay. The standalone SGLang/XoRL component dumps for
`train.parquet:rg0:row2`, layers `0-10`, completed and all temporary pods were
cleaned up; no `er-opd-q36-35b-slots` trainer/teacher roles and no
`er-opd-q36-35b-sci` roles were restamped. Artifacts:
`/shared/opd-control/er-opd-q36-35b-slots/k3/sglang_debug_row2_layers0_10_20260615T1607Z/`
and
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/`.
The XoRL one-trace K3 replay reproduced the row2 baseline: mean
`0.005890282`, p95 `0.002430974`, max `0.622587456`, worst token row2 position
`22`, token `17880`.

The dense transition audit
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/q36_layer_transition_drift_layers0_10_row2069_20260615T1607Z.json`
shows row2069 residual drift starts immediately rather than waiting for the
layer-15 router: layer0 input is identical, but layer0 attention mean abs is
`4.990e-05`, post-attention residual is `4.776e-05`, and post-attention
RMSNorm amplifies that to `0.0021167`. By layer10, layer-output drift is still
only `0.0003836`, while post-attention RMSNorm drift is already `0.0108637`.
This matches the broader pattern seen in the sampled-layer audit: small
residual differences are repeatedly amplified by RMSNorm before the layer-15
router margin becomes fragile.

Full-attention recompute checks on the two full-attention layers in this window
reject early attention internals as the next standalone code fix. Artifacts:
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/q36_full_attention_path_layer3_rows2047_2070_bf16_sdpa_20260615T1607Z.json`
and
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/q36_full_attention_path_layer7_rows2047_2070_bf16_sdpa_20260615T1607Z.json`.
For row2069, layer3 captured/recomputed attention mean abs is
`7.1967e-05` / `6.6984e-05`, and layer7 is `1.0346e-04` /
`1.0151e-04`; recomputed post-attention norm also stays at the captured drift
scale. So do not chase layer3/layer7 q/k/v mapping, RoPE, q/k norm, attention
backend, output gate, o-proj, residual add, or post-attention RMSNorm as
standalone K3 fixes.

The layers0-10 GDN input-sensitivity artifact
`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/q36_gdn_input_sensitivity_layers0_10_20260615T1607Z.json`
also closes local linear-attention arithmetic to BF16-scale on captured inputs
(`record_count=1170`, `missing_record_count=2`; candidate-vs-captured maxes:
XoRL candidate `0.0009765625`, XoRL reference `0.00390625`, SGLang candidate
`0.00390625`, SGLang reference `0.0078125`). Interpretation: the local
full-attention and GDN formulas are not the current promotion path. Next target
is either an upstream residual-stream parity candidate before the layer-15
router input, or a correctness-cleared batching/source-rank-preserving design.
Current audited/promotable 4-node logical MFU remains `0.010405075057463931`
(`~1.04%`); AMDAHL-077-style feed-rate wins still cannot promote until the
static K3/model-forward gate passes.

2026-06-15 16:35Z packed-row provenance follow-up: no promotion and no pod
launch. Engine PR #376 branch `/home/apanda/xorl-opd-repeat2-diagnostics`
is pushed through commit `7cc54ab7` (`Add packed row source provenance
diagnostics`). The patch preserves original packed-row source provenance through
`opd_packed_row_batch_size` grouping:

- `packed_row_source_batch_ids`
- `packed_row_source_request_ids`
- `packed_row_source_num_samples`
- `packed_row_source_token_spans`
- `packed_row_source_group_size`

These fields are marked as packed-row metadata, excluded from row-compatibility
checks, and kept as Python metadata during runner tensor conversion. The
`opd_debug_packed_sample_path` sidecar now emits the full source metadata and
`segment_source_overlaps`, so a rowbatch/repeat replay can map each emitted
teacher-valid segment back to original source packed rows and token spans. This
directly targets the current repeat-data/rowbatch evidence, where copy-boundary
co-packing and source-rank/layout changes are the leading suspects.

Validation passed:

- `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m pytest tests/server/runner/test_runner_dispatcher.py tests/server/runner/test_opd_runner.py tests/server/orchestrator/test_request_processor.py::test_packed_row_batching_groups_single_row_packed_batches tests/server/orchestrator/test_request_processor.py::test_packed_row_batching_defers_to_rank_local_runner_by_default -q`
  -> `28 passed`.
- `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m py_compile src/xorl/server/runner/utils/batch_utils.py src/xorl/server/runner/model_runner.py src/xorl/server/runner/runner_dispatcher.py`.
- `PYTHONPATH=/home/apanda/xorl-opd-repeat2-diagnostics/src /home/apanda/xorl-internal/.venv/bin/python -m ruff check src/xorl/server/runner/utils/batch_utils.py src/xorl/server/runner/model_runner.py tests/server/runner/test_runner_dispatcher.py tests/server/runner/test_opd_runner.py`.
- `git diff --check`.

Verdict: this is a diagnostic/support patch, not an MFU promotion. Current
audited/promotable 4-node logical MFU remains `0.010405075057463931`
(`~1.04%`). The next useful replay, when capacity and a candidate justify it,
should run the packed-sample sidecar with this provenance enabled so the
AMDAHL-077/rank-local rowbatch path can distinguish source-rank preservation
from packed-layout dependence before spending another 4-node gate. Static K3
still blocks promoting rowbatch or quack+DeepEP as a science default.

2026-06-15 16:43Z AMDAHL-081 provenance replay: complete, diagnostic-only, no
promotion. I ran a two-node trainer-only slots replay with XoRL PR #376 commit
`7cc54ab7` on `research-common-h100-092` and
`research-common-h100-110`, then stopped trainer control, deleted
`er-opd-q36-35b-slots-trainer-head` and `trainer-worker-1`, and confirmed the
stack is back to only dispatch + teacher-smg. No `er-opd-q36-35b-sci` role was
restamped.

Primary artifact:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-81-provenance-real47-chunk4-serveronly-20260615T163845Z.jsonl`.
Sidecar dir:
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl081-provenance-real47-debug-20260615T163845Z`.

The measured row after warmup was `server_forward_backward_s=2.894965`,
`api_wall_s=3.094273`, valid `515`, loss `2.366714001`,
`opd_kl=2.345451044`, hidden loss `0.021262966`, forward compute
`0.663631`, backward compute `1.374354`, and model forward `0.625148`.
The first warmup row was slow (`server_forward_backward_s=20.937023`), so use
the measured row only for this smoke.

The provenance sidecars prove rank-local rowbatch grouping is active on the real
AMDAHL-047 capture: 16 sidecar files, 58 rows total, 30 rows with
`packed_row_source_*`, and all 30 provenance rows have
`packed_row_source_group_size=2`. Source batch IDs cover 12 source batches and
the common source spans are `[0,3328]` and `[3328,6656]`, e.g. rank0 emits
`packed_row_source_batch_ids=[0,1]` and spans `[[0,3328],[3328,6656]]`.

Verdict: AMDAHL-081 confirms the debug metadata now exposes source-row grouping
and can support the packed-layout/source-rank investigation, but it is not a
new MFU result and does not change the promotion state. Current audited and
promotable 4-node logical MFU remains `0.010405075057463931` (`~1.04%`;
valid-token-scaled `0.0002947112240841216`). AMDAHL-077/rank-local rowbatch
still requires a passing static K3/model-forward gate before any 4-node
promotion replay or science-default use.
