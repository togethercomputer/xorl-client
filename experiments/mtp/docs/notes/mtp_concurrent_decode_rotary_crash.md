# SGLang native-MTP concurrent decode rotary crash

**Owner ask:** make the OPD student sampler work with
`--max-running-requests > 1`. Today Qwen3.6 native-MTP/ConfAdapt sampling is
effectively pinned to serial decode, which makes OPD rollout sampling the
dominant step-time cost.

## Summary

Concurrent native-MTP decode crashes because the Qwen3.6 text path is routed
through SGLang's Qwen3-VL/MRoPE model wrapper. The generic decode position
builder already creates one flattened position per decode query row for
`decode_q_len_per_req > 1`, but the MRoPE-specific override then replaces it
with only one position per request.

For the observed crash:

- `batch_size = 16`
- `decode_q_len_per_req = 4`
- `query.shape[0] = 16 * 4 = 64`
- `forward_batch.mrope_positions.shape = (3, 16)`
- `cos/sin` therefore have 16 rows, while the query has 64 rows

That mismatch reaches `apply_rotary_emb` and fails with `64 vs 16`.

## Exact failure

```text
RuntimeError: The size of tensor a (64) must match the size of tensor b (16)
              at non-singleton dimension 0
```

The SGLang scheduler reports `Scheduler hit an exception`; the OPD trainer then
sees `RemoteDisconnected`, reports `OPD prepare worker failed on chunk N`, and
exits.

Evidence:

- CUDA graph disabled:
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/sglang-0/logs/20260607T024815Z-run.log`
- CUDA graph enabled:
  `/shared/opd-control/er-opd-q36-mtp-ss-0605c/sglang-0/logs/20260607T023846Z-run.log`

Both runs crash at the first multi-request MTP decode step, after prefills ramp
the running batch to 16 requests. Both logs emit:

```text
Bypassing q>1 decode cuda graph replay.
reason=adaptive_hf_exact_flag_disabled
payload={'decode_q_len_per_req': 4, 'mtp_phase': 'seed', ...}
```

So this is not a CUDA graph replay bug; the eager q>1 decode path also crashes.

## Key traceback

```text
scheduler.run_batch
  -> tp_worker.forward_batch_generation
  -> model_runner._forward_raw
  -> forward_decode
  -> models/qwen3_vl.py forward
  -> general_mm_embed_routine
  -> language_model(...)
  -> models/qwen3_5.py self_attention
  -> q, k = self.rotary_emb(positions, q, k)
  -> layers/rotary_embedding/mrope.py forward_native
  -> query_rot = apply_rotary_emb(query_rot, cos, sin, ...)
  -> layers/rotary_embedding/utils.py apply_rotary_emb
  -> o1 = x1 * cos - x2 * sin
```

## Root Cause

The generic `ForwardBatch` position path is already MTP-aware:

```python
if (
    ret.forward_mode.is_decode()
    and ret.decode_q_len_per_req > 1
    and len(batch.input_ids) == len(batch.seq_lens) * ret.decode_q_len_per_req
):
    start_positions = batch.seq_lens - ret.decode_q_len_per_req
    pos_offsets = torch.arange(ret.decode_q_len_per_req, device=device, dtype=torch.int64)
    ret.positions = (start_positions.unsqueeze(1).to(torch.int64) + pos_offsets).reshape(-1)
```

For `batch_size=16` and `q_len=4`, this produces 64 flattened positions:

```text
[req0_p0, req0_p1, req0_p2, req0_p3,
 req1_p0, req1_p1, req1_p2, req1_p3,
 ...]
```

But because Qwen3.6 is loaded through the Qwen3-VL/MRoPE path, the model later
does:

```python
if self.is_mrope_enabled and forward_batch.mrope_positions is not None:
    positions = forward_batch.mrope_positions
```

That override is built by `_compute_mrope_positions`. In decode mode, the
text-only / RL-on-policy branch ignores `decode_q_len_per_req` and emits only a
single `(3, 1)` position column per request:

```python
mrope_positions_list[batch_idx] = torch.full(
    (3, 1),
    self.seq_lens_cpu[batch_idx] - 1,
    dtype=torch.int64,
)
```

After concatenation across 16 requests, `mrope_positions` is `(3, 16)`, not
`(3, 64)`.

`MRotaryEmbedding.forward_native` indexes the RoPE cache with those positions:

```python
cos_sin = self.cos_sin_cache[positions]
cos, sin = cos_sin.chunk(2, dim=-1)
...
seq_len_q = query.shape[0]  # 64
query = query.view(seq_len_q, -1, self.head_size)
query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
```

`apply_rotary_emb` requires `x`, `cos`, and `sin` to agree on `num_tokens`.
Here `x` has 64 rows and `cos/sin` have 16.

## Correct Fix

Fix the MRoPE position construction, not the rotary kernel.

In
`~/xorl-sglang-internal/python/sglang/srt/model_executor/forward_batch_info.py`,
teach `ForwardBatch._compute_mrope_positions` that decode batches can have
`decode_q_len_per_req > 1`.

For text-only MRoPE decode, the MRoPE positions should mirror the already
computed flattened decode positions. Prefer a vectorized fast path before the
per-request loop:

```python
treat_as_text_mrope = (
    get_global_server_args().rl_on_policy_target is not None
    or all(mm_input is None for mm_input in batch.multimodal_inputs)
)
if (
    self.forward_mode.is_decode()
    and int(self.decode_q_len_per_req) > 1
    and self.positions is not None
    and treat_as_text_mrope
):
    self.mrope_positions = self.positions.reshape(1, -1).repeat(3, 1).to(torch.int64)
    return
```

The final shape must be:

```text
mrope_positions.shape == (3, batch_size * decode_q_len_per_req)
```

For this OPD crash case, that means `(3, 64)`.

Mixed text+image batches need the same invariant, but they must apply the
request-specific MRoPE deltas per query row rather than using this text-only
fast path.

Do **not** fix this by repeating the existing `cos/sin` rows inside
`mrope.py`. Repeating would make the shape pass, but it would rotate all `k`
query rows for a request at the same absolute position. The MTP query rows are
different decode positions and must receive their own RoPE rows.

## Files To Inspect

SGLang fork:

- `~/xorl-sglang-internal/python/sglang/srt/model_executor/forward_batch_info.py`
  - `ForwardBatch.init_new`: generic q>1 decode `ret.positions` builder
  - `ForwardBatch._compute_mrope_positions`: incorrect MRoPE decode override
- `~/xorl-sglang-internal/python/sglang/srt/models/qwen3_vl.py`
  - replaces `positions` with `forward_batch.mrope_positions`
- `~/xorl-sglang-internal/python/sglang/srt/layers/rotary_embedding/mrope.py`
  - crash site after indexing RoPE cache with too few positions
- `~/xorl-sglang-internal/python/sglang/srt/layers/rotary_embedding/utils.py`
  - `apply_rotary_emb` contract: `x`, `cos`, `sin` share `num_tokens`

This repo:

- `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`
  - student-slot launcher used for the repro
- `docs/notes/mtp_singleshot_port.md`
  - broader SingleShot MTP implementation context
- `docs/notes/mtp_torch_compile_recompilation.md`
  - separate trainer-side compile noise; not this crash

## Reproduction

Stack:

- `er-opd-q36-mtp-ss-0605c`
- model: `Qwen/Qwen3.6-35B-A3B`
- student TP: 2
- native MTP / ConfAdapt
- `k_toks=4`
- `OPD_SINGLESHOT_MTP_JSON` includes:
  - `"sampling_mode": "native"`
  - `"mtp_strategy": ["conf_adapt", 0.6]`
  - `"rollout_replay": true`
  - `"native_mtp_debug_trace": true`

Observed behavior:

- `--max-running-requests 1`: OPD sampling works, but rollout is serialized.
- `--max-running-requests 256` with CUDA graph disabled:
  crashes at the first multi-request q>1 decode step.
- `--max-running-requests 256` with CUDA graph enabled:
  same crash. CUDA graph capture succeeds, but q>1 replay is bypassed before
  the crash because the adaptive exact-q graph flag is disabled.

A minimal repro should be possible by launching the Qwen3.6 SGLang server with
native MTP enabled and sending two concurrent `/generate` requests that both
reach the seed MTP decode step with `decode_q_len_per_req=4`.

## Constraints Already Ruled Out

- **CUDA graph is not the cause.** The crash reproduces with CUDA graph on and
  off. The relevant q>1 decode path is eager in both repro logs.
- **Overlap schedule is a separate guard.** Without
  `--disable-overlap-schedule`, `/generate` returns HTTP 400 with:

  ```text
  Temporary phase-2A policy: MTP decode currently requires
  --disable-overlap-schedule while static/conf_adapt CUDA graph stability is
  being finalized.
  ```

  Keep overlap disabled until that policy is intentionally lifted.
- **The generic 1D decode positions are not the broken part.** They are already
  expanded for `q_len>1`. The MRoPE replacement tensor is the broken part.

## Validation Plan

1. Add a unit-level shape test around `ForwardBatch._compute_mrope_positions`
   for text-only decode with:
   - `batch_size=2`
   - `decode_q_len_per_req=4`
   - `batch.seq_lens` already advanced by 4
   - expected `mrope_positions.shape == (3, 8)`
   - expected rows equal the generic flattened positions repeated across the
     three MRoPE axes.
2. Run a two-request SGLang repro against Qwen3.6 native MTP and confirm the
   first batched seed decode no longer crashes.
3. Run an OPD smoke with `--max-running-requests 2` or `4` before jumping to
   `256`.
4. Confirm native-MTP debug trace still records:
   - `decode_q_len_per_req`
   - `phase`
   - emitted / committed token windows
   - no change in serialized `max-running-requests=1` output for a fixed seed.
5. Run the static K3 gate before promoting the SGLang change.

## Impact

This is the current blocker for concurrent student rollout sampling. The trainer
side has other throughput issues, but this crash forces the student sampler into
serial request execution. Until the MRoPE q>1 decode position builder is fixed,
keep the OPD student at `--max-running-requests 1`.
