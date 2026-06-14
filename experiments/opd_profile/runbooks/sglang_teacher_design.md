# SGLang teacher-prefill microservice — build spec (2026-05-29)

## Why
Teacher prefill dominates the OPD step: **99–139 s** (vs fwd_bwd ~37–46 s, sync ~2.5 s,
cache write ~0.5–1 s). The xorl teacher uses the *training* framework (FSDP forward,
`data_parallel_shard_size=16`, 2 nodes) for an inference-only task. 5-digit CoTs are
~5 k tokens (3.25× the 4-digit ~1.5 k), so this only gets worse. SGLang's prefill
(paged KV, fused kernels, continuous batching) is the lever.

## Key facts established (de-risked)
- **No representation-alignment step needed.** SGLang `CaptureHiddenMode.FULL` returns
  the POST-final-norm hidden (Qwen3MoeModel applies `self.norm` inside `self.model`;
  the LogitsProcessor `before_norm` override is EAGLE-only and Qwen3MoE.forward never
  passes it). The xorl teacher cache stores the head input and `opd_streaming_kl` does
  a pure `teacher_hidden @ head_weight` matmul (no norm) → ALSO post-final-norm. They
  match directly.
- **Files, not RDMA.** Cache write is 0.4–1.1 % of the teacher step; the bottleneck is
  prefill *compute*. Keep the existing safetensors cache (trainer reads it unchanged).
  Cache is small + constant (~136 MB/batch; kept rows = prompt+pause+answer ~130/seq,
  the CoT is masked OUT regardless of its length).
- **No HTTP hidden transfer.** Do the cache-write INSIDE the sglang process (hiddens
  never leave it) — reuse xorl's writer.
- **Cache hygiene** (added 2026-05-29): the OPD client now unlinks each step's caches
  after forward_backward; the sglang path inherits this (same cache_path).

## Cache format (what to produce — must match exactly)
`save_file({"teacher_hidden_states": hidden_cache}, cache_path)` where
`hidden_cache` = `[N_total_kept_rows, hidden_size]` bf16, kept rows = the non-masked
predicting positions per sample concatenated; plus return metadata
`{path, tensor_key:"teacher_hidden_states", dtype, num_tokens, hidden_size,
cache_indices_by_sample}` where `cache_indices_by_sample[i] = range(start_i, start_i+rows_i)`
(contiguous per sample). See `model_runner.py::_write_teacher_hidden_cache` +
`_merge_teacher_hidden_cache_payloads`. The kept-position mask logic is in the OPD
client `_teacher_hidden_cache_data` (insert mode: prompt + CoT + pause + answer,
`mask_len = C` if supervise_student_cot else `C + K`).

## Architecture
- **Topology:** SGLang TP=2 × N replicas behind SMG (`power_of_two` policy — cache_aware
  imbalanced on the 235B per cot_dataset_index). Teacher is throughput-bound + forward-
  only → many small DP replicas beat one big TP shard. Q3.6-35B-A3B BF16 (match the
  trainer's teacher dtype for numerical parity).
- **Custom endpoint** (xorl-sglang-internal, e.g. `/teacher_hidden_cache`): receives
  `{input_ids_per_sample, prompt_lens, filler_count, teacher_filler_tokens,
  teacher_cot_mode, supervise_student_cot, cache_path, cache_key, dtype}`. For each
  sample build the teacher seq (prompt + CoT + pause + answer) exactly as
  `_teacher_hidden_cache_data` does, run the engine forward with
  `CaptureHiddenMode.FULL`, take per-position hiddens, apply the kept-position mask,
  concat, `save_file`, return `cache_indices_by_sample`. Both repos are on PYTHONPATH
  so it can `from xorl.server.runner... import` the writer/mask helpers (or vendor them).
- **Client:** add `_teacher_cache_from_sglang(sglang_url, sequences, cache_path, ...)`
  mirroring `_teacher_cache_from_xorl` but POSTing to the SMG router's endpoint instead
  of the xorl teacher `/api/v1/forward`. Gate via a `teacher_backend: str = "xorl"|"sglang"`
  config flag in `_prepare_opd_batch`.

## Validation gate (do BEFORE trusting it in a run)
**Numerical A/B:** same fixed token sequences through (a) the xorl teacher
(`teacher_hidden_cache` /forward) and (b) the sglang endpoint; cosine(kept-position
hiddens) must be ~1.0 (bf16/TP numerical match). If <0.99, investigate dtype/TP/norm
before any training run.

## Throughput target
xorl 2-node teacher ≈ 520 tok/s prefill. Measure sglang TP=2 prefill tok/s/replica;
N replicas → N×. Goal: teacher prefill ≤ fwd_bwd (~40 s) so the step stops being
teacher-bound.

## Status
Foundation ready (free nodes, format, no-alignment proof, files decision, reuse path,
cache cleanup). Remaining: implement the endpoint + client `_teacher_cache_from_sglang`
+ run the A/B gate, then a full OPD run with `teacher_backend=sglang`. Tracked: task #27.
