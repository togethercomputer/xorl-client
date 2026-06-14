# SingleShot MTP OPD - MoE dispatch OOM handoff

## Current Status

**Status (2026-06-06 23:07Z): fixed and live-validated.** Prod stack
`er-opd-q36-mtp-ss-0605c` (Qwen3.6-35B-A3B SingleShot MTP OPD, 4 x 8 H100,
EP=8) now completes the formerly failing full-scale first `forward_backward`
with `prompt_len=512`, `max_new_tokens=64`, `prompts_per_step=32`, and
`static_padded_seq_len=768`.

Validation run:

- head log:
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260606T225949Z-run.log`
- worker logs:
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-{1,2}/logs/20260606T225949Z-run.log`
  and
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-3/logs/20260606T225950Z-run.log`
- trainer-head exited `rc=0`; workers stopped after cleanup.
- grep over head/worker logs returned no old OOM signatures:
  `out of memory`, `CUDA out of memory`, `NCCL WARN`,
  `Failed to CUDA calloc`, `alltoall_pre_dispatch`, `token_pre_all2all`,
  or `DistBackendError`.

Successful full-scale validation evidence:

```text
23:04:28 forward_backward step=0 chunk=1/4 loss=0.6412 valid_tokens=4096 roundtrip=152.075s
23:05:42 forward_backward step=0 chunk=2/4 loss=0.3737 valid_tokens=3608 roundtrip=74.565s
23:06:42 forward_backward step=0 chunk=3/4 loss=0.5700 valid_tokens=3592 roundtrip=59.766s
23:07:15 forward_backward step=0 chunk=4/4 loss=0.5269 valid_tokens=4096 roundtrip=33.510s
23:07:15 Pipeline forward_backward step=0 loss=0.5314 valid_tokens=15392
23:07:15 OPD pipeline validation succeeded.
23:07:15 loss_history=[0.5314316955354382]
```

Worker-side Gloo `Connection closed by peer` tracebacks after 23:07:16 are
shutdown noise from the head exiting and stopping trainer control after success,
not the MoE dispatch OOM.

## Root Cause

The OOM was caused by the previous compile-stability fix for GatedDeltaNet replay
plans. For the production args, the launcher computes `static_padded_seq_len=768`.
The old GDN replay path materialized:

```python
sequence_hidden = hidden_states[plan.sequence_rows, plan.sequence_cols]
```

where `plan.sequence_rows` had shape:

```text
sequence_capacity = batch * (ceil(seq_len / k_toks) * (k_toks - 1) + 1)
                  = 32 * (ceil(768 / 4) * 3 + 1)
                  = 18464

dense virtual rows = 18464 * 768 = 14,180,352 hidden rows
```

At Qwen3.6 hidden size, that temporary alone is tens of GB before GDN's q/k/v,
chunk-kernel workspaces, autograd state, and the following MoE dispatch. The
first MoE all-to-all then surfaced the pressure as an NCCL raw cudaMalloc failure
for a tiny comm buffer:

```text
include/alloc.h:316 NCCL WARN Cuda failure 'out of memory'
include/alloc.h:324 NCCL WARN Failed to CUDA calloc 2097152 / 10485760 bytes
torch.distributed.DistBackendError: NCCL error ... unhandled cuda error
```

So the bug was not infrastructure, DeepEP/alltoall choice, allocator
fragmentation, or bad GPU assignment. It was a dense virtual replay expansion
inside the GDN path.

## Code Fix

Implemented in
`src/xorl/ops/linear_attention/layers/gated_deltanet.py`:

- Small replay plans still use the dense compiled path, preserving the local
  compile-stability behavior.
- Large replay plans switch to a memory-capped packed execution path:
  - gather only active virtual tokens via `sequence_valid`
  - run GDN over `[1, packed_tokens, hidden]` with `cu_seqlens`
  - chunk by `replay_plan_max_packed_tokens` (default `65536`)
  - scatter only valid outputs back to `[batch, seq, hidden]`
- The large packed path is intentionally `torch.compiler.disable`: it keeps the
  outer model usable and removes the OOM. Do not reintroduce the full
  `[sequence_capacity, static_seq_len, hidden]` gather for production-scale
  batches.

Regression added in
`tests/ops/test_linear_attention_singleshot_mask.py`:

- `test_gated_deltanet_replay_plan_runs_packed_chunks_not_dense_capacity`
  verifies a padded plan with static capacity executes packed chunks instead of
  a dense rectangular capacity batch.

## Validation

Local:

```bash
PYTHONPATH=src pytest \
  tests/ops/test_linear_attention_singleshot_mask.py \
  tests/models/test_qwen3_5_singleshot_mask.py \
  tests/mtp/test_singleshot.py::test_prepare_singleshot_rollout_replay_static_padded_seq_len_stabilizes_linear_plan_shape \
  tests/mtp/test_singleshot.py::test_prepare_singleshot_rollout_replay_static_padded_seq_len_rejects_small_capacity \
  -q
```

Result: `14 passed`.

CUDA compile probe:

```bash
PYTHONPATH=src TORCH_LOGS=recompiles pytest \
  tests/ops/test_linear_attention_singleshot_mask.py::test_gated_deltanet_static_replay_plan_capacity_avoids_shape_recompiles_on_cuda \
  -q -s
```

Result: passed.

Live:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-trainer-control $(cat /tmp/mtp_relaunch_args.txt) \
  --num-steps 1 \
  --prompt-dataset-epochs 0 \
  --max-opd-steps 1 \
  --prompts-per-step 32 \
  --prompt-len 512 \
  --max-new-tokens 64 \
  --skip-optim-step \
  --checkpoint-interval-steps 0 \
  --checkpoint-keep-latest 0 \
  --no-checkpoint-save-best \
  --checkpoint-eval-steps 0 \
  --checkpoint-eval-batch-size 0 \
  --forward-backward-timeout 2400 \
  --checkpoint-timeout 2400
```

Result: one full 32-prompt production-scale OPD step completed; no old MoE
dispatch OOM signatures appeared.

## Relaunch Guidance

The trainer is currently stopped after the successful one-step validation. The
inference stack was not restarted for this fix.

To resume the real run:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
mapfile -t ARGS < /tmp/mtp_relaunch_args.txt
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-trainer-control "${ARGS[@]}"
```

Keep these existing fixes:

- Trainer `run.sh` derives `CUDA_VISIBLE_DEVICES` from the device-plugin
  assignment. Do not hardcode CVD.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is harmless but was not the
  OOM fix.
- Trainer pods are bare pods with `restartPolicy: Never`; `write-trainer-control`
  restarts the process in place on the same nodes. To force reschedule, delete
  the four trainer pods and re-apply the rendered manifest, leaving inference
  pods untouched.

## Remaining Non-Blockers

The full-scale validation surfaced a separate Dynamo recompile in
`src/xorl/ops/loss/compiled_cross_entropy.py::_compute_hard_teacher_ce_with_diag`
on worker-1 ranks:

```text
tensor 'labels' size mismatch at index 0. expected 64, actual 3
```

That is not the MoE dispatch OOM and did not block the successful validation.
Treat it as a separate cleanup item if compile-noise cleanup becomes the next
task.
