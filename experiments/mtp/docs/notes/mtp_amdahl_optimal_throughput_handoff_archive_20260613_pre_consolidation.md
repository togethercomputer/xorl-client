# Handoff: drive OPD-MTP to Amdahl-optimal throughput

**Date:** 2026-06-13
**Stack:** `er-opd-q36-mtp-ss-0605c` (Qwen3.6-35B-A3B, native SingleShot-MTP, ConfAdapt)
**Live code:** `/home/apanda/xorl-mtp-commitlen-fix-20260612` (b0cfc6de) — trainer + samplers PYTHONPATH here
**Analysis worktree:** `/home/apanda/xorl-mtp-singleshot-port-20260602` (5f3ff0bd) — NOT what the trainer runs

## Latest override (2026-06-13 22:30Z)

The decoupled trainer-coalescing path was retested under EP32 `alltoall`/triton
to isolate whether the earlier failure was a DeepEP-specific limit. It completed
cleanly, but it is **not a throughput winner** in the current live state.

Artifacts:

| artifact | path / run | result |
|---|---|---|
| full pipeline coalesce-2 alltoall validation | `q36mtp-20260613T221209Z-2s1t`, W&B `qm62obaw`, trainer log `20260613T221209Z-run.log` | Clean capped run from step 500 through 507, `trainer-head cleanup rc=0`. |
| profile JSONL | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T221209Z-2s1t/artifacts/opd_profile.jsonl` | 8 rows, `OPD_PIPELINE_CHUNK_SIZE=32`, `OPD_TRAINER_COALESCE_CHUNKS=2`, one trainer f/b group per step. |

Run shape: resumed from the step-500 checkpoint, `--k-toks 2`,
`--prompts-per-step 64`, `--pipeline-chunk-size 32`,
`--pipeline-prefetch-chunks 2`, `--pipeline-teacher-concurrency 2`,
`--trainer-coalesce-chunks 2`, EP32 `alltoall`/triton,
`recompute_before_dispatch`, `OPD_START_STEP=500`, `OPD_MAX_OPD_STEPS=508`.

Comparison window below excludes cold step 500 and terminal step 507; use steps
501-506 for all three rows.

| shape | step wall | trainer FB | prefetch wait | sampling | teacher | sync | actual MFU | useful MFU | consumed tok/s | commit_len | sample tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk 32, q-band + clean capture | 32.16s | 22.18s | 3.07s | 18.22s | 24.75s | 3.12s | 1.42% | 0.30% | 1,524 | 1.80 | 830 |
| chunk 64, direct validation | 40.90s | 25.74s | 8.71s | 18.60s | 15.88s | 2.92s | 0.98% | 0.14% | 1,172 | 1.75 | 813 |
| chunk 32 prepare + coalesce-2 alltoall trainer | 59.37s | 27.31s | 23.00s | 74.08s | 17.34s | 3.24s | 0.93% | 0.14% | 735 | 1.07 | 139 |

Interpretation:

- The 22:00Z coalescing failure was DeepEP-specific. The same decoupled
  coalescing idea can complete under `alltoall`/triton, so the payload
  concatenation and server semantics are viable enough to measure.
- Do **not** promote coalesce-2 from this run. In the fair warm window its single
  coalesced trainer f/b call averages 27.31s, slower than the earlier
  two-call chunk-32 baseline at 22.18s total. End-to-end wall regresses to
  59.37s.
- The wall-clock comparison is also not same-workload: the sampler state moved
  substantially. The q-band + clean capture had MTP `commit_len` about 1.80 and
  sampler output about 830 tok/s; this later checkpoint has `commit_len` about
  1.07 and sampler output about 139 tok/s. Student confidence can evolve during
  training, but higher confidence is not the same thing as higher MTP
  multi-token acceptance. Future throughput promotion must compare either the
  same checkpoint/sampler state or trainer-only replays of captured payloads
  from the same run.
- The current measured throughput winner remains **q-banding + clean-region +
  `recompute_before_dispatch` at the canonical 32-prompt pipeline chunk**.
  Coalescing is now an optional trainer-only research branch, not the immediate
  science-resume control.

Operational state at 22:25Z: `trainer-head` exited cleanly and trainer workers
are stopped. Dispatch, `sglang-0/1`, and `teacher-sglang-0` remain live. The
current rendered trainer control is the alltoall coalesce-2 capped validation
control, not science training control. Before EXP-1 or any k=2 continuation,
render fresh non-replay, non-profiler training control from the step-500
checkpoint with the canonical 32-prompt chunk shape, EP32, and
`OPD_TRAINER_COALESCE_CHUNKS=1`.

## Latest override (2026-06-13 22:12Z)

The updated OPSD low-MFU note suggested auditing the `G` lever
(`tokens/rank/forward = N / (G * F)`). I tested that directly on the MTP
capture, but only as a trainer-only topology probe:

| artifact | path / run | result |
|---|---|---|
| EP8 trained-checkpoint replay attempt | `q36mtp-20260613T220042Z-2s1t`, trainer log `20260613T220041Z-run.log` | Failed during engine init before replay rows. |
| EP8 base-weight alltoall replay | `q36mtp-20260613T220513Z-2s1t`, trainer log `20260613T220513Z-run.log` | Completed cleanly, 4 replay rows, `trainer-head cleanup rc=0`. |
| replay results | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z_subset_fast8_slow11_ep8_1node_base_alltoall_20260613T220512Z/replay_results.jsonl` | EP8, 1 node, alltoall/triton, base HF weights, no checkpoint load. |

The trained-checkpoint attempt proves a real resume blocker, not a performance
result. Loading the step-500 EP32 DCP checkpoint into EP8 fails on optimizer
state shape mismatch:

```text
ValueError: Size mismatch between saved torch.Size([8, 2048, 1024]) and current: torch.Size([32, 2048, 1024])
for optimizer.state.model.layers.0.mlp.experts.gate_up_proj.momentum_buffer
```

For the base-weight replay, the warmed EP8 rows were:

| topology | warmed trainer FB for payload 11 + 8 | valid target tok/s | actual MFU proxy | useful MFU proxy | GPU-seconds for same two payloads |
|---|---:|---:|---:|---:|---:|
| EP32/alltoall replay baseline | 18.21s | 1,356 | 1.59% | 0.328% | 583 |
| EP8/alltoall base-weight replay | 26.41s | 935 | 3.73% | 0.551% | 211 |

Interpretation:

- Shrinking `G` from 32 to 8 improves per-GPU efficiency and uses ~36% of the
  GPU-seconds for this replay pair, but it is **45% slower in wall-clock** for
  the same two payloads. It is not an immediate throughput winner.
- This was base HF weights with no optimizer state, so it is only a topology
  timing probe. It is not evidence that an EP8 step-500 science resume is valid.
- To test EP8 as a real science continuation, add a model-only checkpoint load
  path or an optimizer-state conversion path; direct EP32->EP8 optimizer resume
  fails today.
- The current measured throughput winner remains q-banding + clean-region +
  `recompute_before_dispatch` at the canonical 32-prompt pipeline chunk. The
  most promising throughput code path is still trainer-call amortization, but
  it must avoid the DeepEP timeout and preserve full-pipeline overlap.

Operational state at 22:12Z: `trainer-head` exited cleanly from the EP8 replay;
trainer workers are stopped. Dispatch, `sglang-0/1`, and `teacher-sglang-0`
remain live. The current rendered trainer control is now the EP8 base-weight
replay control (`OPD_FB_REQUEST_REPLAY_PATH=...ep8_1node_base_alltoall_20260613T220512Z`,
`OPD_LOAD_CHECKPOINT_PATH=''`, `trainer_nodes=1`,
`trainer_expert_parallel_size=8`, `trainer_ep_dispatch=alltoall`), not science
training control. Before EXP-1 or any k=2 continuation, render fresh non-replay,
non-profiler training control from the step-500 checkpoint with the canonical
32-prompt chunk shape, EP32, and `OPD_TRAINER_COALESCE_CHUNKS=1`.

## Latest override (2026-06-13 22:00Z)

Trainer-side chunk coalescing has now been tried in the full pipeline and is
**not promotable as implemented**. This was the decoupled version of the
chunk-64 idea: keep sampler/teacher preparation at two 32-prompt chunks, then
merge the prepared trainer payloads into one 64-prompt `forward_backward`
request.

Artifact:

| artifact | path / run | result |
|---|---|---|
| full pipeline trainer-coalescing validation | `q36mtp-20260613T215028Z-2s1t`, W&B `75424yvk`, trainer log `20260613T215027Z-run.log` | Failed on the first coalesced step-500 trainer future. |
| profile JSONL | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T215028Z-2s1t/artifacts/opd_profile.jsonl` | 0 rows; no throughput window. |
| coalesced cache | `.../q36mtp-20260613T215028Z-2s1t/artifacts/teacher_hidden_step500_coalesced0.safetensors` | Written successfully, 177 MB; coalescing prep completed before the failure. |

Run shape: `--k-toks 2`, `--prompts-per-step 64`,
`OPD_PIPELINE_CHUNK_SIZE=32`, `OPD_PIPELINE_PREFETCH_CHUNKS=2`,
`OPD_TRAINER_COALESCE_CHUNKS=2`, `OPD_START_STEP=500`,
`OPD_MAX_OPD_STEPS=508`. The head log confirms the intended shape:
`Async OPD step 500: 2 prompt chunks, chunk_size=32, prefetch=2`, then the
prepared chunks were `21,757` and `23,469` sampled tokens. The merged teacher
hidden cache was written, so the failure is inside the 64-prompt trainer
`forward_backward`, not in sampler/teacher prep or cache concatenation.

Failure:

```text
AssertionError: Future future_d5eb6687411f failed:
{'error': '500: Engine error: Operation failed: DeepEP error: timeout (dispatch CPU)', 'category': 'server'}
```

Worker logs show the same `DeepEP error: timeout (dispatch CPU)` across ranks
0-31 at `21:57:09Z`. `trainer-head` exited `rc=1`; trainer workers stopped.
Dispatch, student samplers, and teacher services remained live.

Interpretation:

- This does **not** erase the earlier trainer-only combined replay result. It
  says the current production EP32/DeepEP path cannot simply accept this merged
  64-prompt payload in the full pipeline.
- Do **not** rerun `--trainer-coalesce-chunks 2` as a throughput validation
  without first isolating the DeepEP dispatch timeout on the same merged payload.
- Do **not** promote global `pipeline_chunk_size=64`; that was already rejected
  by wall-clock regression.
- The measured throughput winner remains **q-banding + clean-region +
  `recompute_before_dispatch` at the canonical 32-prompt pipeline chunk**.

The updated OPSD low-MFU note still helps, but only as a decomposition checklist:
separate denominator, dummy-fill/packing, above-model server overhead, and
in-model comm/GEMM shape. For MTP, the most recent evidence says trainer-call
amortization is the plausible lever, but the immediate implementation now hits
DeepEP dispatch limits before it can produce rows.

Operational state at 22:00Z: trainer slots are stopped after the failed
coalescing run; dispatch, `sglang-0/1`, and `teacher-sglang-0` are live. The
current rendered trainer control is the failed coalescing validation
(`OPD_PROMPT_DATASET_NUM_PROMPTS=64`, `OPD_TRAINER_COALESCE_CHUNKS=2`,
`OPD_MAX_OPD_STEPS=508`), not science training control. Before EXP-1 or any k=2
science continuation, render fresh non-replay, non-profiler training control
from the step-500 checkpoint with the canonical 32-prompt chunk shape and
`OPD_TRAINER_COALESCE_CHUNKS=1`. Samplers remain k=2; no sampler reprogram is
needed for EXP-1.

## Latest override (2026-06-13 21:30Z)

Full-pipeline validation of the combined-replay idea is now complete, and it
**rejects direct `--pipeline-chunk-size 64` as a promotable setting**. The
trainer-only replay win below is real, but changing the live pipeline chunk from
32 to 64 regresses end-to-end wall-clock and MFU.

Artifact:

| artifact | path / run | result |
|---|---|---|
| full pipeline chunk-64 validation | `q36mtp-20260613T211900Z-2s1t`, W&B `i3ji92nq`, trainer log `20260613T211859Z-run.log` | Clean capped run from step 500 through 507, `trainer-head cleanup rc=0`. |
| profile JSONL | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T211900Z-2s1t/artifacts/opd_profile.jsonl` | 8 rows, chunk size 64, one pipeline chunk per step. |

Comparison window: chunk-64 excludes cold step 500 and terminal step 507
(steps 501-506). The chunk-32 capture baseline is the q-band + clean-region
capture run `q36mtp-20260613T204443Z-2s1t`, steps 501-509.

| shape | step wall | trainer FB | prefetch wait | sampling | teacher | sync | actual MFU | useful MFU | consumed tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk 32, q-band + clean capture | 32.21s | 22.20s | 3.09s | 18.55s | 24.78s | 3.15s | 1.45% | 0.30% | 1,531 |
| chunk 64, q-band + clean validation | 40.90s | 25.74s | 8.71s | 18.60s | 15.88s | 2.92s | 0.98% | 0.14% | 1,172 |

Interpretation:

- The replay-only 64-sample payload still proves that same-shape trainer-call
  amortization can help. It does **not** prove that simply increasing the live
  pipeline chunk is good.
- Direct chunk-64 changes the pipeline topology from two prepared chunks
  (`prefetch_chunks=2`, `prepare_workers=2`) to one prepared chunk
  (`prefetch_chunks=1`, `prepare_workers=1`). The measured prefetch wait rises
  from ~3.09s to ~8.71s, and live trainer FB rises rather than falls.
- The current measured winner remains **q-banding + clean-region +
  `recompute_before_dispatch` at the canonical 32-prompt pipeline chunk**.
- The next throughput implementation, if we pursue this line, should decouple
  trainer coalescing from sampler/teacher chunking: keep 32-prompt
  sampler/teacher prepare parallelism, then coalesce ready trainer payloads into
  fewer/larger `forward_backward` submissions. Do not promote global
  `pipeline_chunk_size=64`.

Operational state at 21:29Z: the capped chunk-64 validation exited cleanly.
`trainer-head` exited with rc=0 and trainer workers are stopped; dispatch,
student samplers, and teacher services remain live. The current rendered trainer
control is the chunk-64 capped validation control (`OPD_PIPELINE_CHUNK_SIZE=64`,
`OPD_MAX_OPD_STEPS=508`). Before science EXP-1 or any k=2 continuation, render
fresh non-replay, non-profiler training control from the step-500 checkpoint
with the canonical 32-prompt chunk shape. Do not just unpause/restart the
current control. Samplers remain k=2; no sampler reprogram is needed for EXP-1.

## Latest override (2026-06-13 21:20Z)

Superseded by the 21:30Z full-pipeline validation above; retained here for the
trainer-only combined replay artifact and interpretation.

The combined-payload replay has now been measured. It is the first direct
positive signal for trainer-call amortization, but it is still a trainer-only
replay result and must not be promoted without a full-pipeline validation.

Artifact:

| artifact | path / run | result |
|---|---|---|
| combined 64-sample replay payload | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z_combo_slow11_fast8` | Concatenates captured payloads 11 + 8, offsets teacher-cache indices, `teacher_hidden_0.safetensors` has 47,547 rows. |
| replay run | `q36mtp-20260613T211419Z-2s1t`, trainer log `20260613T211418Z-run.log` | Correct q-band + clean baseline flags: EP32, alltoall/triton, `dp_shard`, stateful replay on, `recompute_before_dispatch`, no profiler. Clean completion. |
| replay rows | `.../replay_results_combo64_20260613T211418Z.jsonl` | 3 calls: step0 cold 52.02s, step1 9.91s, step2 10.31s. |

Comparison against the earlier sequential compile-on replay of the same two
payloads (`replay_results_compile_on_20260613T205525Z.jsonl`, post-warm
payload 11 + payload 8):

| shape | warmed trainer FB | valid tokens/s | actual MFU proxy | useful MFU proxy |
|---|---:|---:|---:|---:|
| sequential two calls | 18.21s | 1,356 | 1.59% | 0.328% |
| combined one call | 10.11s | 2,440 | 2.44% | 0.364% |

Interpretation: warmed combined replay gets ~1.8x more valid target tokens/s and
~1.5x actual-MFU proxy versus issuing the two payloads as separate trainer calls.
This validates the Amdahl hypothesis that per-call FSDP/EP communication and
movement are being amortized poorly. It also explains why
`fsdp_defer_grad_sync` did not look immediately useful for the existing payloads:
single-payload calls are one micro-batch; the win comes from fewer
forward_backward calls, not from the current defer flags.

Do **not** over-read this as a production win yet:

- The first combined call was very slow: 52.02s forward/backward
  (31.49s forward, 21.53s backward), worse than the earlier first sequential
  slow+fast pair. The warmed calls are the useful signal.
- This was replay-only: no student sampling, teacher prefill, optimizer step, or
  weight sync. Full-pipeline chunk/coalescing has to preserve pipeline overlap
  and science semantics.
- The combined request still reported `opd_singleshot_mtp_micro_batches=1.0`;
  this is not evidence that FSDP defer flags help. It is evidence that fewer,
  larger `forward_backward` submissions can amortize the current per-call
  overhead.

Historical next throughput test, now completed and rejected above: run a full
q-band + clean-region validation with a larger trainer forward_backward chunk
shape (effectively coalescing the two 32-prompt chunks into one 64-prompt
trainer call) and compare end-to-end step wall, sampling/teacher/FB/sync, and
actual / useful MFU. The promotion gate is full-pipeline wall-clock, not
replay-only valid/s.

Historical operational state after the replay: trainer slots were stopped cleanly
(`trainer-head exited_at=21:17:44Z rc=0`). The current rendered trainer control
was the **combined replay** control, not science training control.

## Latest override (2026-06-13 21:15Z)

The updated OPSD low-MFU note is useful, but the key correction is that its
16k-token/rank GEMM knee is a **stack-specific executed-token bound**, not a
direct MTP prescription. On the current q-band + clean-region MTP replay, the
captured payloads already run roughly at that GDN-token knee:

| payload | warmed trainer FB | global GDN executed / rank | base batch tokens / rank | valid targets / rank |
|---|---:|---:|---:|---:|
| slow live payload 11, replay payload 0 | 10.20s | 20.3k | 3.8k | 434 |
| fast live payload 8, replay payload 1 | 8.01s | 15.7k | 3.8k | 337 |

So the low useful MFU is not explained by "MTP is merely below the OPSD
small-GEMM knee." The current evidence says: GDN replay is already large enough
to be near the in-model token knee, useful-token density is still sparse, and
the first-call torch profile is dominated by FSDP/EP communication and movement
(`record_param_comms`, NCCL all-gather/all-to-all, copies), not OPD KL/top-k.
Because each captured q-band + clean-region replay call reported
`opd_singleshot_mtp_micro_batches=1.0`, the existing
`--trainer-fsdp-defer-grad-sync` / `--trainer-fsdp-defer-reshard-after-backward`
knobs cannot help those single-payload calls; they only become relevant if a
larger/coalesced forward_backward call creates multiple micro-batches.

Prepared but **not run**, to avoid colliding with the science agent's k=2
resume window: a combined 64-sample replay payload at
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z_combo_slow11_fast8`.
It concatenates captured payloads 11 and 8, offsets teacher-cache indices, and
writes a combined `teacher_hidden_0.safetensors` with 47,547 rows. Sequential
warmed replay of the same two payloads costs about 18.2s (10.20s + 8.01s). Run
this combined replay only after coordinating with science; it occupies the
trainer slots even though it does not reprogram samplers.

Operational state checked at `2026-06-13T21:12:15Z`: the capped perf replay had
already exited cleanly (`trainer-head exited_at=21:07:14Z rc=0`), and trainer
pods are only idle slots. The current trainer `run.sh` is still the one-call
profiler replay (`XORL_PROFILE_SERVER_FB=1`,
`OPD_FB_REQUEST_REPLAY_PATH=...subset_slow11_profile`). Before EXP-1 or any
science resume, render fresh non-replay, non-profiler `recompute_before_dispatch`
trainer control. Do not merely unpause/restart the existing trainer control.

## Latest override (2026-06-13 21:10Z)

This section supersedes the 20:45Z next-step guidance. The OPSD low-MFU
microbench is useful as a decomposition pattern, but **do not transplant its
numeric factors to SingleShot-MTP**. MTP is already executing large replay/GDN
work per trainer call; its useful-token density is low and its live f/b timing
is transient/bimodal.

New q-band + clean-region capture and replay artifacts:

| artifact | path / run | result |
|---|---|---|
| full pipeline capture | `q36mtp-20260613T204443Z-2s1t`, W&B `9t25o44t` | Clean 10-row step-500 replay-window run, 20 saved f/b payloads. |
| saved payloads | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z` | `fb_request_0..19.json` plus teacher hidden caches. |
| fast/slow replay subset | `.../fb_capture_qband_clean_step500_20260613T204431Z_subset_fast8_slow11` | Payload 8 is a fast live chunk; payload 11 is the slowest live chunk. |
| compile-on replay | `replay_results_compile_on_20260613T205525Z.jsonl` | Slow payload post-warm mean 10.20s; fast payload post-warm mean 8.01s. |
| no-compile replay | `replay_results_no_compile_20260613T205911Z.jsonl` | Slow payload post-warm mean 9.59s; fast payload post-warm mean 8.54s. No material speed win. |
| first-call torch profile | `q36mtp-20260613T210316Z-2s1t/server_output/fb_profiles/fb_keyavg_call0.txt` and `fb_trace_call0.json.gz` | One profiled slow payload; profiler wall inflated to 105.5s, but phase metrics stayed forward 12.49s / backward 8.25s. |

Capture-window summary, excluding step-500 warmup (`q36mtp-20260613T204443Z-2s1t`,
steps 501-509): mean step wall 32.21s, trainer FB 22.20s, forward 7.38s,
backward 12.41s, sync 3.15s, teacher 24.78s, sampling 18.55s,
actual/useful MFU 1.45% / 0.30%. The live chunks were bimodal: most chunks were
7.8-9.0s trainer time, but payloads 4/11/19 were 24.0-25.7s. OPD loss/KL stayed
small; the slow path was extra transformer/GDN/comm forward+backward time.

The replay result is the key new evidence: the captured "slow" payload is not
intrinsically a 25s payload. With the same checkpoint, q-banding, clean-region,
and `recompute_before_dispatch`, payload 11 ran 20.7s only on the first replay
call, then 9.6-10.8s on repeats. Disabling trainer compile did not remove the
first-call behavior and did not improve warmed throughput. Therefore do **not**
promote no-compile, and do not explain the live slow chunks as fixed bad
payloads or OPD KL/top-k.

Profiler interpretation: the rank-0 key-average table for the first slow replay
call is dominated by communication and movement, not the OPD head. Top CUDA rows
include `record_param_comms` (4.18s self CUDA), NCCL all-gather (2.16s),
NCCL all-to-all (1.93s), `aten::copy_` (1.08s), GDN backward/forward rows
(~0.64s / ~0.53s), and `aten::mm` (~0.65s). Treat the 105.5s profiled replay
wall as profiler/export overhead; use the phase metrics and key-average table
for attribution.

Actionable next target:

- Keep **q-banding + clean-region + `recompute_before_dispatch`** as the current
  measured winner.
- Do **not** retry `no_recompute`, no-compile, OPD KL/top-k, or
  `--gdn-replay-plan-max-packed-tokens` as the next throughput lever.
- The next plausible speed work is trainer-call efficiency: reduce first-use
  FSDP/EP communication spikes and per-call all-gather/all-to-all overhead, or
  test coalescing larger captured f/b payloads so communication is amortized
  over more useful targets. Correctness-gate any code change that touches replay
  semantics.

Operational state: the current rendered trainer control is the one-call
profiler replay (`XORL_PROFILE_SERVER_FB=1`, `OPD_FB_REQUEST_REPLAY_PATH=...subset_slow11_profile`).
Render fresh non-profiler `recompute_before_dispatch` control before any further
science or promotion run.

## Latest override (2026-06-13 20:45Z)

This section supersedes the 20:20Z next-step guidance. The OPSD pass helped by
pointing us at synchronized phase timing, not by giving us a direct config
transplant. We ran that diagnostic on the current MTP winner and confirmed the
same shape: **OPD KL/top-k is not the bottleneck.**

New diagnostic runs:

| config | run dir / W&B | rows | outcome |
|---|---|---|---|
| q-banding + clean-region, wall timing | `q36mtp-20260613T200325Z-2s1t` / `tu5ffak6` | 502-514, no terminal | Still the measured winner: 34.62s wall, 18.55s sampling, 25.00s teacher, 24.98s FB, 3.37s sync, 1460 consumed tok/s. |
| q-banding + clean-region, CUDA-synced profile | `q36mtp-20260613T201834Z-2s1t` / `krofvfsq` | 502-508 | Clean completion. 35.34s wall, 19.36s sampling, 25.20s teacher, 26.96s FB, 3.16s sync. Inside trainer FB: `trainer_forward_loss_s=9.20s`, `trainer_backward_s=15.10s`, `trainer_opd_total_ms=379ms`, `trainer_opd_kl_compute_ms=140ms`. |
| q-banding + clean-region + `no_recompute` | `q36mtp-20260613T203140Z-2s1t` / `h57g4v8g` | 0 | Failed on the first step-500 FB: rank 5 OOM in the stateful GDN suffix path (`chunk_delta_h.py`), 1.40 GiB allocation with only 287 MiB free. Terminated manually, `trainer-head cleanup rc=143`. |

Interpretation and hard gates:

- Keep **q-banding + clean-region + `recompute_before_dispatch`** as the
  current throughput winner.
- Do **not** chase OPD KL/top-k as a throughput lever. On the synced MTP profile,
  KL is about 0.14s/step and total OPD head/KL work is about 0.38s/step; the
  expensive region is transformer/GDN forward, backward, recompute, and comm.
- Do **not** promote or retry `no_recompute` at the current 64-prompt q-band +
  clean-region shape. It does not fit.
- Do **not** spend a run sweeping `--gdn-replay-plan-max-packed-tokens` while
  `linear_replay_plan_use_stateful_prefix_cache=true`: the active
  `_forward_with_replay_plan_stateful_prefix_cache` path bypasses that cap. The
  profile still reports fallback-style `gdn_forward_calls`, but actual stateful
  execution is fixed at 2 context rounds plus one suffix call
  (`stateful_gdn_forward_calls=3`, `fallback_micro_batches=0` on the failed
  no-recompute chunk).
- The next meaningful MFU work is below the OPD layer: warmed kernel/FSDP comm
  profiling of the q-band+clean winner, or a code change that reduces the
  stateful GDN suffix memory/work without changing the training signal.

Operational state after the failed test: `supervisor.pause` remains present.
The failed no-recompute run was killed and no run-specific trainer processes
remain in the trainer pods. The current desired trainer control still encodes
the failed `no_recompute` diagnostic, so render fresh `recompute_before_dispatch`
control before the next trainer launch.

## Latest override (2026-06-13 20:20Z)

This section supersedes the 19:45Z alignment-only note. **The current measured
winner is canonical q-banding + clean-region replay**, not any standalone
`XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN` setting.

The OPSD throughput pass in
`/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607` transfers one important
lesson: do not chase OPD KL/top-k as the next speed lever until CUDA-synchronized
phase timing says it is the bottleneck. Their corrected profile showed
`opd_profile_backward_compute_s=969.91s` of `opd_pipeline_forward_backward_s=1015.77s`,
with OPD loss/KL only ~6.09s/~5.40s. The same principle applies here: our
promotion runs should compare wall-clock without extra synchronization, and
diagnostic relaunches should enable `OPD_PROFILE_SYNC_CUDA=1` when splitting the
trainer forward/backward phases. The reprogrammable trainer generator now bakes
`OPD_PROFILE_SYNC_CUDA` into `run.sh` when exported at render time.

Validated runs from the step-500 filesystem DCP:

| config | run dir | window | step wall | prefetch wait | sampling | teacher | FB | sync | actual/useful MFU | consumed tok/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline align=256 | `q36mtp-20260613T110939Z-2s1t` | 508-521 | 62.53s | 32.52s | 85.15s | 18.34s | 23.63s | 3.24s | 1.59% / 0.29% | 730 | old reference |
| q-banding only | `q36mtp-20260613T194614Z-2s1t` | 502-514, no terminal | 43.33s | 0.19s | 19.01s | 25.62s | 36.00s | 3.92s | 0.88% / 0.18% | 1147 | sampling fixed, FB inflated |
| q-banding + clean-region | `q36mtp-20260613T200325Z-2s1t` | 502-514, no terminal | 34.62s | 3.03s | 18.55s | 25.00s | 24.98s | 3.37s | 1.38% / 0.28% | 1460 | **new measured winner** |
| align=128 long | `q36mtp-20260613T190017Z-2s1t` | 508-530, no terminal | 61.46s | 8.26s | 83.86s | 17.92s | 46.75s | 3.28s | 0.58% / 0.14% | 738 | not promoted |

Interpretation:

- q-banding is the real sampling lever: it moved sampling from ~85s aggregate to
  ~19s and removed the prefetch bubble, but by itself it pushed FB from ~24s to
  ~36s.
- clean-region replay is the offset for that FB inflation: on top of q-banding it
  brought FB back to ~25s while preserving the sampling win.
- The new combined config is close to Amdahl-balanced for this shape: sampling
  ~18.6s, teacher ~25.0s, FB ~25.0s, sync ~3.4s. It roughly doubles consumed
  tok/s vs the baseline and cuts step wall by ~45%.
- MFU is not "solved" by this alone. Useful MFU returns near baseline rather than
  jumping, because the win is mostly reducing idle/prefetch and replay waste. The
  next MFU diagnostic should use CUDA-synchronized phase timing or a warmed
  kernel profiler on the q-band+clean winner; do not spend time on OPD KL/top-k
  unless those timers contradict the current evidence.

Controls used for the winner:

- samplers already reprogrammed with `--student-mtp-canonical-q-banding`
  (`--enable-mtp-adaptive-hf-exact-canonical-q-banding`, q-lens `2,3`);
- trainer `XORL_SINGLESHOT_MTP_CLEAN_REPLAY_CONTEXT=1`;
- trainer `XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN=''` (default 256);
- trainer `OPD_PROFILE_SYNC_CUDA=''` for promotion-comparable wall timing;
- `OPD_START_STEP=500`, `OPD_MAX_OPD_STEPS=516`, same step-500 DCP as baseline.

Both q-banding-only and q-banding+clean completed cleanly with
`trainer-head cleanup rc=0`. Exclude step 500 warmup and step 515 terminal when
using these runs for promotion comparisons.

## Previous override (2026-06-13 19:45Z)

This section supersedes the 18:35Z align=128-winner note. **Do not promote
`XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN=128` as the permanent default on
stack `er-opd-q36-mtp-ss-0605c`.** The longer validation shows it mostly trades
sampler prefetch wait for slower FB/replay execution, with essentially flat wall
clock and much worse useful MFU.

Long validation run: `q36mtp-20260613T190017Z-2s1t`, same step-500 filesystem DCP
as the baseline, `STUDENT_REPLICAS=2`, alltoall/triton trainer, no q-banding, no
clean-region flag, `XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN=128`,
`OPD_MAX_OPD_STEPS=532`. It completed cleanly with `trainer-head cleanup rc=0`
and wrote 32 profile rows. Exclude row 500 as warmup and row 531 as the capped
terminal row.

| config | run dir | window | step wall | prefetch wait | FB | sampling | actual/useful MFU | consumed tok/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline align=256 | `q36mtp-20260613T110939Z-2s1t` | 508-521 | 62.53s | 32.52s | 23.63s | 85.15s | 1.59% / 0.29% | 730 | default still defensible |
| align=128 long | `q36mtp-20260613T190017Z-2s1t` | 508-530, no terminal | 61.46s | 8.26s | 46.75s | 83.86s | 0.58% / 0.14% | 738 | not promoted |
| align=128 short | `q36mtp-20260613T182155Z-2s1t` | 501-506, no terminal | 57.13s | 14.27s | 36.80s | 69.68s | 0.70% / 0.18% | 762 | favorable slice, not reproduced |
| align=64 short | `q36mtp-20260613T180946Z-2s1t` | 501-506, no terminal | 59.00s | 21.04s | 31.50s | 74.04s | 0.64% / 0.20% | 740 | close but weak |
| align=32 short | `q36mtp-20260613T175558Z-2s1t` | 501-506, no terminal | 61.35s | 10.86s | 44.02s | 75.57s | 0.41% / 0.15% | 712 | rejected |
| 4 samplers, align=128 | `q36mtp-20260613T184125Z-4s1t` | 501-506, no terminal | 67.51s | 22.97s | 29.90s | 77.14s | 0.90% / 0.24% | 646 | rejected: sync ~11.4s |

Important correction: the earlier 8-row A/Bs ended at `OPD_MAX_OPD_STEPS=508`.
Their final row 507 is a capped terminal-row artifact with no next-step prefetch
pressure. Do not use the earlier 501-507 or last-three-row averages as promotion
evidence; use 501-506 for those short runs.

**Interpretation:** align=128 reduces replay tokens (`~1.50M -> ~0.99M` global
GDN tokens on the 508-521 comparison), but it increases stateful replay grouping
work from 6 calls / 2 context rounds to about 7.9 calls / 3 context rounds. The
result is lower prefetch wait and lower arithmetic utilization: FB becomes the
pole, useful MFU roughly halves, and wall clock barely moves. The low MFU is
therefore not explained by a single "too many FLOPs" story. Baseline wastes
useful MFU on duplicated GDN context; align=128 removes some duplicate tokens but
turns FB into a lower-utilization, launch/roundtrip-heavy pole.

**Current code/control state:**

- Live code remains patched in `/home/apanda/xorl-mtp-commitlen-fix-20260612/src/xorl/mtp/singleshot.py`
  with env-gated `XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN`; default remains 256.
- Local control generator
  `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py` bakes
  `XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN` into trainer `run.sh`.
- Unit validation in the live code tree:
  `PYTHONPATH=/home/apanda/xorl-mtp-commitlen-fix-20260612/src /home/apanda/xorl-mtp-commitlen-fix-20260612/.venv/bin/python -m pytest /home/apanda/xorl-mtp-commitlen-fix-20260612/tests/mtp/test_singleshot.py -q`
  -> `41 passed`.
- The stack is back to 2 student samplers; `sglang-2/3` pods and Services were deleted after the failed
  4-sampler A/B. `supervisor.pause` is still present.
- Current trainer slot status is stopped/exited cleanly, but the current trainer `run.sh` is still the
  completed bounded align=128 validation (`OPD_START_STEP=500`, `OPD_MAX_OPD_STEPS=532`). Write fresh
  control before unpausing or continuing science.

**Next gate:** do not spend K3 on align=128 as a default because it did not pass
the throughput promotion gate. If someone wants to promote a non-default align
setting later, run the static K3/logprob replay gate after launching an isolated
xorl endpoint; do not run it concurrently with a throughput validation.

## Mission

Push OPD-MTP throughput to **Amdahl-optimal** and *fully validate* the best setup with measured numbers.
"Amdahl-optimal" = balance the per-step poles (sampling / teacher / FB / weight-sync) so no single one
dominates, and every deployed lever is measured end-to-end, not assumed. The validated levers below are NOT
yet combined or deployed — that is the job.

## Current measured state (the Amdahl baseline)

k=2 prod run (infra agent, 2026-06-13 ~10:4x), and k=4 numbers from the throughput A/Bs earlier today:

| pole | k=2 (current) | k=4 (earlier) | notes |
|---|---|---|---|
| **step wall** | **~62 s** | ~79 s | what we minimize |
| student sampling (sum / 2 chunks) | ~85–95 s | ~120–140 s | **the binding pole** |
| teacher prefill | ~16–24 s | ~16–24 s | overlapped |
| forward_backward (fb_sum) | ~20 s | ~50 s | **hidden under sampling** |
| weight sync | ~3 s | ~3 s | serial |
| MFU actual / useful | 0.96% / 0.17% | ~3.0% / 0.13% | over wall-clock |
| achieved TFLOPS/GPU | 9.5 / 989 | ~36 / 989 | H100 peak 989 |

**The step is SAMPLING-BOUND.** With async prefetch overlap, FB (≈20 s) and teacher (≈20 s) hide under the
sampler wall (≈62 s), so the 32 trainer GPUs sit idle ~⅔ of each step waiting on the 2 student samplers.
That idle is *why* MFU is ~1% — not an FB inefficiency. **Corollary that drives lever ranking:** anything
that only shrinks the *hidden* FB time does NOT move wall-clock until sampling drops below FB.

## Levers, ranked by WALL-CLOCK impact (with measured evidence)

### 1. Faster sampling — highest wall-clock impact (attack this first)
- **q-banding** — VALIDATED this session (the crash is fixed). ConfAdapt fragments the MTP decode batch into
  per-q_len buckets {4,5,6,7} → near-serial decode (~100 tok/s). Banding pins steady steps to q=2k−1 so all
  requests share one bucket. Measured A/B (k=4, 2 clean steps): q collapsed `{4,5,6,7}→{4,7}`,
  `canonical_q_banding_applied=True` (169/229), **tok/s ~100→330–442 (~4×)**, sample_sum ~120–140 s→33–45 s,
  prefetch_wait ~22–35 s→**0**. Crash (`Invalid verified MTP commit length verified=2/planned=1`) **FIXED** by
  the q-banding agent (per-request hf-exact verify; `xorl-sglang-internal::mtp_decode_verify.py`) — ran ~27 min
  of decode with zero crashes. **Caveat: q-banding INFLATES fb_sum** (k=4: ~50→70 s) because q=7 padding adds
  GDN-replay positions. So banding alone gave only ~5% net step-time at k=4 (the win was traded into FB) —
  see lever 3. Enable: launcher flag `--student-mtp-canonical-q-banding`.
- **Scale samplers 2→4** — VALIDATED NEGATIVE on the current stack. `q36mtp-20260613T184125Z-4s1t`
  kept align=128 and added two non-privileged student sampler pods (`team: turbo`, `rdma/infiniband: 1`,
  `IPC_LOCK`, no hardcoded CVD). Dispatch saw 4 workers, but rows 501-506 were worse than 2-sampler:
  67.51s wall, 77.14s sampling, and sync exploded to 11.42s. Do not retry plain 4-sampler scaling until
  the sync path/topology is changed or isolated from the step wall.

### 2. Higher commit_len — science lever, compounds sampling
- commit_len≈1 today (draft heads under-trained) means ~k draft+verify passes per committed token. Raising
  acceptance → fewer decode steps/token → faster sampling. The k=2 experiment is the science attempt to
  unlock this. Out of pure-throughput scope, but it is the *structural* sampling win and it shrinks FB too
  (fewer replay blocks). Track `mtp/commit_len_steady`; coordinate with the science agent.

### 3. clean-region FB (~26×) — EFFICIENCY win now, SPEED win only after sampling is fixed
- VALIDATED + **proven prediction-safe** this session (see `mtp_clean_region_replay_semantics_question.md`
  PROOF RESULT): committed-context sorted+deduped → GDN stateful path nests → ~26× fewer GDN tokens; the ~12%
  GDN-state delta does NOT flip the supervised argmax (current==clean 8/8; 0.500 vs 0.476 vs sampler).
- **But at the current sampling-bound config it barely moves wall-clock** — it cuts fb_sum (k=4 ~50→~?, k=2
  ~20→~1 s) but FB is already hidden under sampling. It LOWERS actual-MFU waste (regret 5.6× → ~1×) without
  changing wall.
- **Why it still matters for Amdahl-optimal:** (a) once levers 1–2 drop sampling *below* FB, FB becomes the
  pole and clean-region is what keeps it from being the new bottleneck; (b) q-banding inflates FB, and
  clean-region is exactly the offset. So the optimal config is **q-banding + clean-region together** — banding
  for sampling, clean-region to absorb the FB it adds. Deploy clean-region flag-gated; harness +
  `build_clean_plan` are in `experiments/opd_profile/` (validated offline + the live argmax proof).
- One open rigor item: the proof judged top-1 argmax (stable); the OPD KL *loss distribution* still shifts by
  the state delta. A KL-distribution check would upgrade "prediction-safe" to "loss-target-safe" before a
  permanent deploy.

## The plan to reach + validate Amdahl-optimal

1. **Re-measure the clean k=2 baseline** (per-pole: sample_sum wall, teacher, fb_sum, weight-sync, step) so
   you have the reference. Use `opd_profile.jsonl` + the trainer log `Async forward_backward` lines.
2. **Deploy q-banding** (coordinated restart) → re-measure. Expect sampler wall to ~⅓, step → toward
   max(teacher, fb_sum_banded). Confirm `canonical_q_banding_applied=True` and no commit-verify crash over
   ≥30 min.
3. **Deploy clean-region on top** → re-measure. Expect fb_sum (now inflated by banding) to collapse → step →
   toward the sampler wall again, now much lower. This is the combined win.
4. **If still sampling-bound after q-banding**, do not repeat plain 2→4 sampler scaling. First change or
   isolate the sync path/topology, then re-test sampler scaling with `sync_inference_weights_s` reported as a
   first-class pole.
5. **Amdahl check each step:** report the per-pole breakdown and which pole binds. Stop when the poles are
   balanced (no single pole > ~1.5× the others) or further lever cost exceeds the gain. Target: get useful
   MFU off the ~0.1–0.2% floor by (a) removing trainer idle (faster sampling) and (b) removing FB regret
   (clean-region). Document the final config + the measured per-pole numbers as the validated best setup.
6. **Re-baseline science metrics** if clean-region is deployed permanently (the ~12% state change means the
   per-offset draft-acceptance numbers and the k=2 result must be re-measured on the corrected replay) —
   coordinate with the science agent.

## Operational gotchas (learned the hard way this session — read before touching the stack)

- **`OPD_XORL_REPO` defaults to the ANALYSIS worktree.** A manual `write-trainer-control` without
  `export OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612` runs the trainer on the WRONG (older)
  code. The supervisor sets it; manual calls must too.
- **Resume needs `OPD_START_STEP` + `OPD_LOAD_CHECKPOINT_PATH` exported** before `write-trainer-control`, else
  `load_checkpoint_path=''` → base weights at step 0 (run.sh hardcodes `OPD_START_STEP=0` as the default;
  the exported values get baked into the regenerated run.sh).
- **Pod `HOME` ≠ `/home/apanda`**, so `~/.shell_env` is NOT sourced in pods — inject env by baking it into the
  launcher run.sh template, not `.shell_env`.
- **Every sampler reprogram wedges the trainer** (in-flight weight sync to restarting samplers) → the
  supervisor recovers by resuming from the latest checkpoint. To A/B a sampler change: `touch supervisor.pause`
  FIRST (so its crash-recovery can't regenerate the sampler control and clobber your flag), then
  `stop-trainer-control → write-student-inference-control <flag> → write-trainer-control` (with the two env
  exports above), and **unpause when done.** The supervisor's recovery regenerates samplers from
  `launch_args_*.txt` (no banding) — banding/clean-region must be in the args file or re-applied each time.
- **`stop-trainer-control` needs `--stack`** or it hits an (empty) default stack.
- **Checkpoint cadence:** checkpoints at step%100; the run keeps resetting to the step-200 ckpt on each
  disruption, so it rarely advances past ~200 during heavy experimentation. Minimize restarts; batch changes
  into ONE coordinated restart.

## Coordination state at handoff (2026-06-13 ~10:43Z)

- **Infra agent owns the stack** — launched the science **k=2** prod run at 10:39 (apanda /loop directive).
  Samplers are baseline (banding reverted). Trainer on commitlen-fix.
- **`supervisor.pause` is IN PLACE** (I left it from the q-banding A/B). The k=2 run currently has NO
  auto-recovery. The infra agent must unpause AFTER updating `launch_args_*.txt` to k=2 (else the supervisor
  recovers with k=4 args). Flagged in AGENT_NOTES — confirm it's resolved before assuming the run is protected.
- **q-banding crash fix** is uncommitted in `xorl-sglang-internal` working tree (live on sampler restart; ask
  the q-banding agent to commit so it survives a `git reset`).

## Validated assets to reuse (don't re-derive)
- Findings + per-pole analysis: `docs/notes/mtp_throughput_findings_20260613.md`
- clean-region proof + verdict: `docs/notes/mtp_clean_region_replay_semantics_question.md` (+ science answer doc)
- q-banding fix details: `docs/notes/mtp_qbanding_sglang_fix.md`
- proof/clean-plan harness: `experiments/opd_profile/clean_region_proof.py`, `_clean_plan_gen.py`,
  `clean_region_proof_harness.py`
- memory: `project_mtp_clean_region_replay_signal_change`, `project_mtp_opd_throughput_sampling_bound`
