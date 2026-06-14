# Handoff: conf=0.3 compile-on forward_backward OOM masked as timeout

**Date:** 2026-06-07
**Stack:** `er-opd-q36-mtp-ss-0605c`
**Run:** `q36mtp-20260607T053023Z-2s2t`
**Status:** blocking until the selected-output GDN replay fix is live-validated at
`max_new_tokens=256` / `static_padded_seq_len=1536`.

## TL;DR

The conf=0.3 / `max_new_tokens=256` compile-on run did **not** just hang in
Inductor for 1800s. The head/server logs make it look like a
`forward_backward` timeout, but the worker logs show the real first failure:
multiple trainer ranks hit CUDA OOM at **05:37:45Z**, about 44s after the first
`forward_backward` began. The orchestrator then waited until its 1800s
operation timeout and surfaced a 504 at **06:07:00Z**.

The OOM occurs in backward recompute through:

```text
GatedDeltaNet._forward_with_replay_plan_packed_chunks
  -> _forward_standard
  -> FusedRMSNormGated.forward
  -> rms_norm_gated
  -> y * weight.float()
torch.OutOfMemoryError: Tried to allocate 1016.00 MiB
```

Later A/Bs showed the first-FB allocation does **not** move with
`linear_replay_plan_max_packed_tokens`, compile on/off, pipeline chunk size, or
AdamW-vs-Muon. Optimizer states are lazily allocated on the first optimizer step,
and this run never reaches that step. The one lever that has moved the footprint
is the SingleShot padded sequence length: `max_new_tokens=64` / pad 768 fit,
while `max_new_tokens=256` / pad 1536 is about 250 MiB over the first-FB memory
limit.

## Evidence

Head log:

```text
/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260607T053023Z-run.log

05:37:00 Prepared OPD chunk 1/4 ... tokens=4871
06:07:00 Future future_699b5dbbd1e3 failed:
         {'error': '504: Forward-backward timeout after 1800.0s', 'category': 'server'}
```

Server log:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260607T053023Z-2s2t/server.log

05:31:12 torch.compile applied to 30 decoder layers
05:37:01 POST /api/v1/forward_backward HTTP/1.1 200 OK
05:37:01 Full-weights session started: default
05:37:xx torch/_inductor/compile_fx.py warnings + Online softmax warnings
06:07:00 Timeout waiting for response (... timeout=1800.0s)
06:07:00 [TIMING] engine forward_backward: execute=1800.1155s
```

Worker logs:

```text
/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-1/logs/20260607T053023Z-run.log
/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-2/logs/20260607T053023Z-run.log
/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-3/logs/20260607T053022Z-run.log

05:37:45 Rank 10/12/11/13/9/14/8/15: CUDA out of memory.
05:37:45 Rank 30/31/28/27/29/25/26/24: CUDA out of memory.
```

The failure pattern is replicated across worker ranks. The server log contains
Inductor warnings but no `recompile_limit`, `Recompiling function`, guard
failure, `attention_mask.linear_plan`, or `BlockMask` evidence for this run.

## Corrected Diagnosis

The previous version of this handoff over-attributed the incident to dynamic
Flex masks and compile churn. Current code does not match that framing:

- `src/xorl/distributed/torch_parallelize.py` compiles decoder layers with
  `torch.compile(..., dynamic=True)`, but skips full-attention decoder wrappers
  when `attn_implementation == "flex_attention"`.
- The run logged `torch.compile applied to 30 decoder layers`, which are the
  Qwen3.6 linear-attention/GatedDeltaNet decoder layers, not all 40 layers.
- The large GDN replay packed path is `@torch.compiler.disable`, so the OOMing
  `_forward_with_replay_plan_packed_chunks` body is outside Dynamo.
- The failing stack is in backward checkpoint recompute, not in MoE dispatch and
  not in a FlexAttention `BlockMask` guard.

The immediate cause is the first forward/backward memory envelope at
`static_padded_seq_len=1536`. At OOM time the live allocation is model weights +
gradients + GDN backward-recompute buffers over the static padded sequence.
Changing optimizer cannot help before the first optimizer step, and the packed
token cap did not change the measured ~70 GiB FB-time allocation.

## Fix Applied In This Checkout

This checkout now keeps compile enabled and reduces the peak allocation inside
the GatedDeltaNet packed replay path:

- `src/xorl/ops/linear_attention/layers/gated_deltanet.py` still runs the
  recurrent GDN scan over all packed replay-history tokens, because selected
  outputs depend on that history.
- The packed replay path now computes `output_indices` for the positions that
  are actually scattered back to the model sequence and passes those indices into
  `_forward_standard`.
- `_forward_standard` now selects recurrent outputs before `FusedRMSNormGated`
  and computes `g_proj` only for the selected output positions. This changes the
  failing `rms_norm_gated -> y * weight.float()` allocation from
  replay-history-token scale to selected-output-token scale.
- Chunks with no selected outputs are skipped.
- `OPD_TRAINER_ENABLE_COMPILE` now defaults to `true`; compile remains
  overrideable but is no longer the primary OOM lever.
- `OPD_GDN_REPLAY_PLAN_MAX_PACKED_TOKENS` now defaults to the YAML value
  `16384`; lower values remain overrideable for A/Bs, but the observed FB OOM
  did not move with 16k or 8k.
- Added `--mtp-max-static-padded-seq-len` to
  `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`, but its
  default is `0` (disabled). Use it only as a fallback fit guard or bisection
  knob if the selected-output fix still OOMs.
- The earlier server-arg plumbing for `linear_replay_plan_max_packed_tokens` and
  `--operation-timeout "${OPD_SERVER_OPERATION_TIMEOUT}"` remains in place.

## Relaunch / Verification

Re-render the trainer control from the same args:

```bash
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-trainer-control $(grep -v '^$' /tmp/mtp_relaunch_args_0607b.txt)
```

Default expected behavior after this patch for the previous
`max_new_tokens=256` relaunch args:

```bash
OPD_REQUESTED_MAX_NEW_TOKENS=256
OPD_EFFECTIVE_MAX_NEW_TOKENS=256
OPD_MAX_NEW_TOKENS=256
OPD_MTP_MAX_STATIC_PADDED_SEQ_LEN=0
OPD_MTP_STATIC_PADDED_SEQ_LEN=1536
OPD_TRAINER_ENABLE_COMPILE=true
OPD_GDN_REPLAY_PLAN_MAX_PACKED_TOKENS=16384
OPD_SERVER_OPERATION_TIMEOUT=${OPD_FORWARD_BACKWARD_TIMEOUT}
```

Fallback if pad1536 still OOMs:

```bash
--mtp-max-static-padded-seq-len=1280
```

For `prompt_len=512`, `k_toks=4`, and `pad_to_multiple=128`, that fallback guard
derives effective `max_new_tokens=193` and `static_padded_seq_len=1280`.

Use compile-off only as an A/B control after the model-path fix is tested:

```bash
OPD_TRAINER_ENABLE_COMPILE=false
```

Success criteria:

- first chunk returns from `forward_backward` instead of hanging until a 504;
- worker logs have zero `CUDA out of memory`, `Failed to CUDA calloc`, and
  `NCCL unhandled cuda error`;
- compile logs have zero `recompile_limit`, `Recompiling function`, guard
  failure, `attention_mask.linear_plan`, and `BlockMask` churn;
- profile rows advance past step 0.

Do not rely only on `server.log`; for this incident it missed the actionable OOM
until cleanup. Always grep head plus all worker logs:

```bash
rg -a -n "CUDA out of memory|Failed to CUDA calloc|recompile_limit|Recompiling|guard failure|BlockMask|linear_plan|Timeout|forward_backward" \
  /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/<stamp>-run.log \
  /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-worker-*/logs/<stamp>*-run.log \
  /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/<run-id>/server.log
```

## Remaining Question

If pad1536 still OOMs after the selected-output replay fix, step down the guard,
not the optimizer:

1. `--mtp-max-static-padded-seq-len=1280`: should derive effective
   `max_new_tokens=193` for `prompt_len=512`, `k_toks=4`.
2. `--mtp-max-static-padded-seq-len=1152`: should derive effective
   `max_new_tokens=161` for `prompt_len=512`, `k_toks=4`.
3. `--mtp-max-static-padded-seq-len=1024`: should derive effective
   `max_new_tokens=129`.
4. Only after the smallest fit point is known, retest compile-off to measure
   compile overhead; do not treat it as the primary fix unless it changes the
   first-FB allocation.
