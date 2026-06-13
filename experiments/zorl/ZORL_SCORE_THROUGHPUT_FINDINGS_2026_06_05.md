# ZORL score-throughput investigation

Date: 2026-06-05
Companion to: `ZORL_PERFORMANCE_HANDOFF_RUNBOOK_2026_06_05.md`,
`ZORL_APPLY_GPU_OPTIMIZATION_2026_06_05.md`

## TL;DR

Scoring (~235 s/step, ~69% of the ZORL-WORDLE-014 step) is **not** throughput-
bound — a TP=2 H100 replica only pushes ~8–20k tok/s of prefill for a 3B-active
MoE that should do far more. Three things throttle it, and one hard kernel bug
caps how far you can push:

| lever | effect | status |
|---|---|---|
| fewer distinct LoRAs + bigger single-LoRA batch | ~2× (9.9k→20k tok/s/replica) | **validated** |
| drop serialization flags (`CUDA_LAUNCH_BLOCKING=1`, `--disable-overlap-schedule`) | extra throughput | recommended |
| trim teacher-forced logprobs to the target suffix | less logit memory/compute | **implemented** (flag, default off) |
| **~64-seq/prefill Triton CUDA OOB** in the MoE LoRA path | hard ceiling → 4× blocked | **root-caused, needs kernel fix** |

## How it was measured

Dedicated 1-replica TP=2 benchmark pool (separate from the live run; the live
wordle run was never touched):

```
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-bench-sglang.yaml   (all throttle flags parameterized)
experiments/zorl/standalone/bench_zorl_sglang.py score            (driver; reachable by pod IP from the dev pod)
```

A single replica is enough: the per-replica prefill throughput, the LoRA crash,
and the buffer limits are all per-replica. The benchmark workload
`--num-pairs 32 --num-shards 1 --train-size 128` reproduces the exact production
per-replica load (64 candidates × 128 examples = 8192 seqs / 1.95M input tokens).

`score_max_workers_per_owner` (= **C**, distinct LoRAs co-batched per replica)
and `--teacher-forced-batch-size` (= **S**, examples per request) are the client
packing knobs; in-flight seqs/prefill ≈ C×S.

## Result 1 — fewer LoRAs + bigger batch ≈ 2× (validated, FAST flags)

| config | distinct LoRAs/prefill | tok/s/replica |
|---|---|---|
| C=4, S=8  (production packing) | 4 | 9 912 |
| C=2, S=16 | 2 | 10 570 |
| C=1, S=32 | 1 | 13 957 (64 cand) / **19 961** (16 cand) |

All stable (0 restarts, 0 5xx). Routing **one LoRA per prefill with a bigger
example batch** beats four LoRAs × small batch, because the multi-LoRA
shared-outer kernel overhead dominates at small batch. Production t_score ~235 s
→ optimized ~100–140 s ≈ **~1.7–2.4×** (single-replica bench vs the multi-replica
production number, so directional).

Recommended scoring config (stays under the crash line below):
`score_max_workers_per_owner=1–2`, `teacher_forced_batch_size=32`.

## Result 2 — serialization flags throttle the server

The production SGL pool launches with:

- `CUDA_LAUNCH_BLOCKING=1` — serializes *every* CUDA kernel (no async overlap)
- `--disable-overlap-schedule` — no CPU/GPU scheduling overlap
- `--disable-cuda-graph` — (prefill-heavy, so smaller effect)
- `SGLANG_DEBUG_SHARED_OUTER=1` — **no-op** (unreferenced in the fork; drop it)

The first three were added defensively against the crash below. Dropping the
first two (overlap on, no launch-blocking) is part of the recommended config;
keep batches under the crash line so the guards aren't needed.

## Result 3 — teacher-forced logprob trim (implemented, flag, default off)

Teacher-forced scoring sent `logprob_start_len=0` (logprobs over the whole ~238-
token prompt) but the client only consumes the last ~60 target-token logprobs.
`zorl_client.py` now has `--teacher-forced-logprob-trim` (+`-margin`), which sets
a per-sequence `logprob_start_len = prompt_len - target_tokens - margin`. The
prompt is still fully prefilled (so target logprob *values* are unchanged); only
the prompt's `[positions, vocab≈150k]` logit rows are skipped. Same scores, less
score-time logit memory/compute. Default off; validate on an idle pool
(`bench_zorl_sglang.py score --logprob-start-len ...`) before enabling in prod.

## Result 4 — the hard ceiling: ~64-seq/prefill Triton CUDA OOB

S=32 (32 seqs/prefill) is rock-stable; **S=64 (~15.2k tokens) reliably crashes**
the scheduler, independent of concurrency, mem-fraction, or prefill-size. Ruled
out: logprob/logits memory (trimming didn't help), host-RAM OOM (exit reason is
`Completed`/0, not `OOMKilled`), mem-fraction (crashes at 0.6 and 0.85 alike).

Root cause (async-reported traceback, FAST flags):

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  lora/layers.py::_forward_with_lora -> _base_down_gemm -> invoke_fused_moe_kernel -> fused_moe_kernel
SIGQUIT received ... one child failed   (-> exit 0 "Completed")
```

i.e. a Triton kernel buffer OOB in the MoE LoRA forward at ~64 seqs (the report
site is the *next* kernel launch; the real OOB is likely an earlier LoRA SGMV /
shared-outer kernel). This is exactly why production caps batch=8/4-LoRAs and
serializes. **Fixing this kernel OOB is the path to 4×+** and to safely raising
`--chunked-prefill-size` / `--max-running-requests`. It needs a synchronous
compute-sanitizer repro to pin the exact kernel + buffer (the hard device abort
gives no clean traceback under `CUDA_LAUNCH_BLOCKING=1`). Related:
`project_chunked_sgmv_oob` memory.

Follow-up (not done, premature until the OOB is fixed): make
`layers.py::_ensure_moe_lora_buffers` grow on demand instead of the fixed
`_MAX_NUM_DISPATCHED=131072` (= 16384 × topk) so prefill size can scale.

## mem-fraction note

For this teacher-forced / `max_new_tokens=1` (prefill-dominated) workload the KV
pool is barely used, so a big `--mem-fraction-static` is wasteful and dangerous:
0.85 left only ~8 GB working memory and OOM-crashed even small requests. Keep
≤0.6; spend the headroom on bigger prefills (once the kernel OOB is fixed), not
KV.

## Do-now vs needs-kernel-fix

- **Do now (≈2×, low risk):** scoring config `per_owner=1–2`, `batch=32`, drop
  `CUDA_LAUNCH_BLOCKING`/`--disable-overlap-schedule`/`SGLANG_DEBUG_SHARED_OUTER`;
  optionally `--teacher-forced-logprob-trim` after a correctness check.
- **Needs kernel fix (≈4×):** the ~64-seq MoE-LoRA Triton OOB.
