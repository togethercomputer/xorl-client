# Superseded

This incremental findings log is superseded by
`docs/notes/mtp_amdahl_optimal_throughput_handoff.md`, which consolidates the
current control, tried levers, failures, and next-agent runbook. Keep this file
for raw historical detail only.

# MTP throughput findings — measured report

**Date:** 2026-06-13 ~04:05 UTC
**Agent direction:** throughput
**Stack:** `er-opd-q36-mtp-ss-0605c`
**Live run analyzed:** `q36mtp-20260613T014150Z-2s1t` (steps 100–197, alive at time of write)
**Repo HEAD:** `5f3ff0bd` on `apanda-dev-mtp`
**Scope:** read-only analysis only. The live run was NOT interrupted (science checkpoint not yet
secured — see "Operational state"). No trainer/sampler control was written.

---

## LATEST OVERRIDE 2026-06-13 22:30Z: alltoall coalescing completes, but does not win

Retested the decoupled trainer-coalescing path under EP32 `alltoall`/triton:
`q36mtp-20260613T221209Z-2s1t`, W&B `qm62obaw`, trainer log
`20260613T221209Z-run.log`, profile
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T221209Z-2s1t/artifacts/opd_profile.jsonl`.
It completed cleanly through steps 500-507 (`trainer-head cleanup rc=0`) with
`pipeline_chunk_size=32`, `trainer_coalesce_chunks=2`, one coalesced trainer
f/b group per step, `alltoall`/triton, and `recompute_before_dispatch`.

Measured warm window, excluding cold step 500 and terminal step 507:

| shape | steps | wall | trainer FB | prefetch wait | sampling | teacher | sync | consumed tok/s | actual/useful MFU | commit_len | sample tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk 32 q-band + clean | 501-506 | 32.16s | 22.18s | 3.07s | 18.22s | 24.75s | 3.12s | 1,524 | 1.42% / 0.30% | 1.80 | 830 |
| direct chunk 64 | 501-506 | 40.90s | 25.74s | 8.71s | 18.60s | 15.88s | 2.92s | 1,172 | 0.98% / 0.14% | 1.75 | 813 |
| chunk 32 prepare + coalesce-2 alltoall trainer | 501-506 | 59.37s | 27.31s | 23.00s | 74.08s | 17.34s | 3.24s | 735 | 0.93% / 0.14% | 1.07 | 139 |

Verdict: the earlier coalesce-2 failure was DeepEP-specific, not a proof that
payload concatenation is invalid. But coalesce-2 under alltoall is still **not
promotable**: the coalesced trainer call is slower than the old two-call
chunk-32 baseline, and the full step regresses badly.

Important caveat for future comparisons: sampler behavior is a moving workload
as the student trains. This later checkpoint no longer matches the q-band clean
capture: MTP `commit_len` fell from about 1.80 to about 1.07 and sampler output
fell from about 830 tok/s to about 139 tok/s. Student confidence may be changing
for science reasons, but higher confidence is not identical to higher MTP
offset acceptance. Throughput candidates should be promoted only by same
checkpoint/same sampler-state comparisons, or by trainer-only replay where
sampling is removed.

Operational note: the trainer slot is stopped cleanly after this validation.
Dispatch, `sglang-0/1`, and `teacher-sglang-0` remain live. The current rendered
trainer control is alltoall coalesce-2 capped validation control, not EXP-1
science control. Before EXP-1, render fresh non-replay, non-profiler step-500
control with the canonical 32-prompt EP32 shape and
`OPD_TRAINER_COALESCE_CHUNKS=1`.

## LATEST OVERRIDE 2026-06-13 22:12Z: EP8 replay improves GPU efficiency, not wall throughput

The updated OPSD low-MFU note is useful as a decomposition checklist, so I
tested its `G` lever on the captured MTP payloads. This was a trainer-only
topology probe, not a science continuation.

Artifacts:

| artifact | run / path | result |
|---|---|---|
| EP8 trained-checkpoint replay attempt | `q36mtp-20260613T220042Z-2s1t`, trainer log `20260613T220041Z-run.log` | Failed during engine init before replay rows. |
| EP8 base-weight alltoall replay | `q36mtp-20260613T220513Z-2s1t`, trainer log `20260613T220513Z-run.log` | Completed cleanly; 4 replay rows; `trainer-head cleanup rc=0`. |
| replay JSONL | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z_subset_fast8_slow11_ep8_1node_base_alltoall_20260613T220512Z/replay_results.jsonl` | EP8, 1 node, alltoall/triton, base HF weights, no checkpoint load. |

The trained-checkpoint attempt exposed a real resume blocker: direct EP32 DCP
optimizer state does not load into EP8. The representative error is:

```text
ValueError: Size mismatch between saved torch.Size([8, 2048, 1024]) and current: torch.Size([32, 2048, 1024])
for optimizer.state.model.layers.0.mlp.experts.gate_up_proj.momentum_buffer
```

For the base-weight replay, ignoring the first compile/warmup row, EP8 was:

| topology | warmed trainer FB for payload 11 + 8 | valid target tok/s | actual MFU proxy | useful MFU proxy | GPU-seconds for same two payloads |
|---|---:|---:|---:|---:|---:|
| EP32/alltoall replay baseline | 18.21s | 1,356 | 1.59% | 0.328% | 583 |
| EP8/alltoall base-weight replay | 26.41s | 935 | 3.73% | 0.551% | 211 |

Verdict: reducing `G` is a GPU-efficiency lever here, not the next wall-clock
throughput lever. EP8 uses far fewer GPU-seconds on this replay pair, but it is
~45% slower wall-clock and cannot currently resume the trained step-500
optimizer state. Treat EP8 as an optional future cost-efficiency track after a
model-only/optimizer-conversion load path exists, not as the immediate EXP-1
science path.

Operational note: the trainer slot is stopped cleanly after the EP8 replay.
Dispatch, `sglang-0/1`, and `teacher-sglang-0` remain live. The current rendered
trainer control is EP8 base replay control (`OPD_FB_REQUEST_REPLAY_PATH=...ep8_1node_base_alltoall_20260613T220512Z`,
`OPD_LOAD_CHECKPOINT_PATH=''`, `trainer_nodes=1`,
`trainer_expert_parallel_size=8`, `trainer_ep_dispatch=alltoall`), not science
training control. Before EXP-1, render fresh non-replay, non-profiler step-500
control with the canonical 32-prompt EP32 shape and
`OPD_TRAINER_COALESCE_CHUNKS=1`.

## LATEST OVERRIDE 2026-06-13 22:00Z: Trainer-side coalescing fails in DeepEP

The decoupled coalescing implementation was validated in the full pipeline and
failed before producing profile rows. This test preserved sampler/teacher
parallelism at two 32-prompt chunks and only coalesced the prepared trainer
payloads into one 64-prompt `forward_backward` request:
`q36mtp-20260613T215028Z-2s1t`, W&B `75424yvk`, trainer log
`20260613T215027Z-run.log`, profile
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T215028Z-2s1t/artifacts/opd_profile.jsonl`.

Run shape: `k_toks=2`, `prompts_per_step=64`, `pipeline_chunk_size=32`,
`prefetch=2`, `teacher_concurrency=2`, `trainer_coalesce_chunks=2`,
`OPD_START_STEP=500`, `OPD_MAX_OPD_STEPS=508`. The log confirmed
`Async OPD step 500: 2 prompt chunks, chunk_size=32, prefetch=2`, with prepared
chunks of `21,757` and `23,469` sampled tokens. The merged cache was written at
`.../q36mtp-20260613T215028Z-2s1t/artifacts/teacher_hidden_step500_coalesced0.safetensors`
(177 MB), so the failure occurred inside trainer `forward_backward`, not during
sampling, teacher prefill, or hidden-cache concatenation.

Failure:

```text
AssertionError: Future future_d5eb6687411f failed:
{'error': '500: Engine error: Operation failed: DeepEP error: timeout (dispatch CPU)', 'category': 'server'}
```

Worker logs report `DeepEP error: timeout (dispatch CPU)` across ranks 0-31 at
`21:57:09Z`. `trainer-head` exited `rc=1`; trainer workers stopped; the profile
file has 0 rows.

Verdict: the coalescing code path is **not promotable as-is**. The earlier
trainer-only combined replay remains a useful signal that fewer/larger
`forward_backward` submissions can amortize communication, but the current
production EP32/DeepEP path cannot just consume the merged 64-prompt payload.
Do not rerun `--trainer-coalesce-chunks 2` as a throughput test until the DeepEP
dispatch timeout is isolated. The measured winner remains q-banding +
clean-region + `recompute_before_dispatch` at the canonical 32-prompt pipeline
chunk.

Operational note: dispatch, `sglang-0/1`, and `teacher-sglang-0` remain live.
The current rendered trainer control is failed coalescing validation control
(`OPD_PROMPT_DATASET_NUM_PROMPTS=64`, `OPD_TRAINER_COALESCE_CHUNKS=2`,
`OPD_MAX_OPD_STEPS=508`), not EXP-1 science control. Before any k=2 continuation,
render fresh non-replay, non-profiler training control from the step-500
checkpoint with `prompts_per_step=32` and `OPD_TRAINER_COALESCE_CHUNKS=1`.

## LATEST OVERRIDE 2026-06-13 21:30Z: Full-pipeline chunk-64 validation rejects direct promotion

The trainer-only combined replay result below was real, but its first
full-pipeline validation did **not** transfer. We ran a capped q-band +
clean-region validation with `--pipeline-chunk-size 64`:
`q36mtp-20260613T211900Z-2s1t`, W&B `i3ji92nq`, trainer log
`20260613T211859Z-run.log`, profile
`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T211900Z-2s1t/artifacts/opd_profile.jsonl`.
It completed cleanly through steps 500-507 (`trainer-head cleanup rc=0`).

Measured window, excluding cold step 500 and terminal step 507: steps 501-506
mean 40.90s wall, 25.74s trainer FB, 10.05s forward, 14.20s backward, 18.60s
sampling, 15.88s teacher, 2.92s sync, 8.71s prefetch wait, 1,172 consumed
tok/s, actual/useful MFU 0.98% / 0.14%. The comparable chunk-32 q-band + clean
capture window (`q36mtp-20260613T204443Z-2s1t`, steps 501-509) was 32.21s wall,
22.20s trainer FB, 18.55s sampling, 24.78s teacher, 3.15s sync, 3.09s prefetch
wait, 1,531 consumed tok/s, actual/useful MFU 1.45% / 0.30%.

Verdict: keep q-banding + clean-region + `recompute_before_dispatch` at the
canonical 32-prompt pipeline chunk as the measured winner. Do not promote global
`pipeline_chunk_size=64`. The replay result still points to trainer-call
amortization as a promising code path, but the next implementation should keep
32-prompt sampler/teacher prepare parallelism and coalesce ready trainer
payloads separately. Directly widening the live pipeline chunk changes prefetch
topology (`prefetch_chunks=2`, `prepare_workers=2` -> 1/1), increases prefetch
wait, and does not reduce live trainer FB.

Operational note: the capped chunk-64 run is stopped cleanly. The current
rendered trainer control is chunk-64 capped validation control, not science
training control. EXP-1 can reuse the warm k=2 stack only after rendering fresh
non-replay, non-profiler training control from the step-500 checkpoint with the
canonical 32-prompt chunk shape. The samplers remain k=2; no sampler reprogram
is needed for EXP-1.

## LATEST OVERRIDE 2026-06-13 21:20Z: Combined replay validates call-amortization, replay-only

Superseded by the 21:30Z full-pipeline validation above; retained here for the
trainer-only combined replay artifact and interpretation.

Measured the prepared combined 64-sample replay payload
`.../fb_capture_qband_clean_step500_20260613T204431Z_combo_slow11_fast8` with
the correct q-band + clean baseline trainer flags (`q36mtp-20260613T211419Z-2s1t`,
trainer log `20260613T211418Z-run.log`). It completed cleanly and preserved
results to `replay_results_combo64_20260613T211418Z.jsonl`.

Result: the cold combined call was bad (52.02s trainer; 31.49s forward,
21.53s backward), but warmed calls were strong: 9.91s and 10.31s. Against the
earlier sequential compile-on replay of the same two payloads, warmed combined
replay improved trainer FB from 18.21s to 10.11s for the same 24,680 valid
targets. Valid target throughput rose from ~1,356/s to ~2,440/s; actual MFU
proxy rose from ~1.59% to ~2.44%; useful MFU proxy rose only modestly
(~0.328% to ~0.364%).

Interpretation: the replay validates trainer-call amortization as a real lever.
The bottleneck is not fixed by OPD KL/top-k, no-compile, or no-recompute. It is
also not solved by existing FSDP defer flags for the current single-payload
calls, because both old and combined replay rows still report
`opd_singleshot_mtp_micro_batches=1.0`. The promising lever is fewer/larger
`forward_backward` submissions, followed by a full-pipeline validation to see
whether the wall-clock gain survives sampling, teacher prefill, optimizer, and
weight sync.

Historical operational note after the replay: trainer slots were stopped
cleanly, but the rendered trainer control was the combined replay control.

## LATEST OVERRIDE 2026-06-13 21:15Z: OPSD's token knee is not the MTP bottleneck by itself

The updated OPSD microbench note narrows its claim: the ~16k tokens/rank knee is
an in-model executed-token bound, not a universal server-RL tax. On the current
MTP q-band + clean replay payloads, GDN executed work is already near that knee
per rank: payload 11 is ~20.3k global GDN tokens/rank and payload 8 is ~15.7k,
while base batch tokens are only ~3.8k/rank and valid targets are ~337-434/rank.
That makes MTP different from the OPSD small-batch story. Low useful MFU here is
a stacked effect of sparse valid-token density plus trainer communication /
movement, with the first-call profile dominated by FSDP/EP all-gather,
all-to-all, and copies rather than OPD KL/top-k.

One more lever audit: `fsdp_defer_grad_sync` and
`fsdp_defer_reshard_after_backward` are real communication-amortization knobs,
but the captured q-band + clean replay calls report
`opd_singleshot_mtp_micro_batches=1.0`, so they cannot help those single-payload
calls. They become worth testing only if larger/coalesced forward_backward
payloads produce multiple micro-batches.

Prepared but not launched: a 64-sample combined replay payload
`.../fb_capture_qband_clean_step500_20260613T204431Z_combo_slow11_fast8`
combines payloads 11 and 8 with adjusted teacher-cache indices and a concatenated
47,547-row hidden cache. Sequential warmed replay of those two payloads costs
~18.2s; the combined replay is the right next trainer-only test once science is
not about to resume k=2.

Operational correction: at `2026-06-13T21:12:15Z`, no trainer process was active;
the last profiler replay had exited cleanly at `21:07:14Z`. The current trainer
control is still profiler replay control, so EXP-1 must render fresh non-replay
trainer control before resuming from the step-500 checkpoint.

## LATEST OVERRIDE 2026-06-13 21:10Z: MTP replay says the OPSD factors do not transfer directly

The OPSD low-MFU microbench is useful as a checklist: separate valid-token
accounting, server overhead, and in-model executed-token size. It is not a
numeric recipe for SingleShot-MTP. On the current q-band + clean-region winner,
MTP already executes large GDN replay work per f/b call while useful target
tokens remain sparse.

New measured artifacts:

| artifact | run / path | result |
|---|---|---|
| full q-band + clean capture | `q36mtp-20260613T204443Z-2s1t`, W&B `9t25o44t` | Clean 10-row run, 20 captured f/b payloads. |
| capture dir | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/fb_capture_qband_clean_step500_20260613T204431Z` | Payloads `0..19`. |
| replay subset | `..._subset_fast8_slow11` | Fast live payload 8 plus slowest live payload 11. |
| compile-on replay | `replay_results_compile_on_20260613T205525Z.jsonl` | Slow payload post-warm 10.20s; fast payload post-warm 8.01s. |
| no-compile replay | `replay_results_no_compile_20260613T205911Z.jsonl` | Slow payload post-warm 9.59s; fast payload post-warm 8.54s. |
| first-call profile | `q36mtp-20260613T210316Z-2s1t/server_output/fb_profiles/` | `fb_keyavg_call0.txt` + Chrome trace for the first slow call. |

Capture-window means over steps 501-509: step wall 32.21s, trainer FB 22.20s,
forward 7.38s, backward 12.41s, sync 3.15s, teacher 24.78s, sampling 18.55s,
actual/useful MFU 1.45% / 0.30%.

The new evidence changes the bottleneck story. Live f/b chunks are bimodal
(mostly 7.8-9.0s, with payloads 4/11/19 at 24.0-25.7s), but replay shows the
slowest payload is not permanently slow: it is ~20s on the first call and
~10s after warmup. `--no-trainer-enable-compile` did not remove that first-call
slow path and did not improve warmed replay. The profiler key-average table for
the first slow call is dominated by communication/movement: `record_param_comms`
4.18s self CUDA, NCCL all-gather 2.16s, NCCL all-to-all 1.93s, and `aten::copy_`
1.08s; GDN forward/backward and GEMMs are below those rows. The 105.5s profiled
wall includes profiler/export overhead; use the 12.49s forward / 8.25s backward
phase metrics and the key-average table for attribution.

Updated verdicts:

- Keep q-banding + clean-region + `recompute_before_dispatch` as the measured
  winner.
- Do not promote no-compile or retry no-recompute at this shape.
- Do not chase OPD KL/top-k or pause/think token deletion for throughput.
- Next useful work is trainer-call efficiency: first-use FSDP/EP communication
  spikes, all-gather/all-to-all amortization, or a measured larger-payload
  coalescing experiment. Correctness-gate semantic replay changes.

Operational note: current rendered trainer control is the profiler replay. Write
fresh non-profiler `recompute_before_dispatch` control before the next run.

---

## LATEST OVERRIDE 2026-06-13 20:45Z: OPSD helps, but no-recompute does not fit

The OPSD-Wordle pass was useful as a diagnostic pattern: use corrected
CUDA-synchronized phase timing before blaming OPD KL/top-k. Applying that here
confirms the same broad conclusion for SingleShot-MTP: **the bottleneck is
trainer forward/backward/recompute/comm, not OPD KL/top-k.**

New runs from the step-500 DCP:

| config | run dir / W&B | rows | result |
|---|---|---|---|
| q-banding + clean-region, wall timing | `q36mtp-20260613T200325Z-2s1t` / `tu5ffak6` | 502-514, no terminal | Current winner: 34.62s wall, 1460 consumed tok/s. |
| q-banding + clean-region, `OPD_PROFILE_SYNC_CUDA=1` | `q36mtp-20260613T201834Z-2s1t` / `krofvfsq` | 502-508 | 35.34s wall; FB 26.96s; `trainer_forward_loss_s=9.20s`, `trainer_backward_s=15.10s`, OPD total 0.379s, KL compute 0.140s. Clean completion. |
| q-banding + clean-region + `no_recompute` | `q36mtp-20260613T203140Z-2s1t` / `h57g4v8g` | 0 | Failed first step-500 FB with rank-5 CUDA OOM in stateful GDN suffix (`chunk_delta_h.py`): 1.40 GiB allocation, 287 MiB free. Terminated, `trainer-head cleanup rc=143`. |

Updated verdicts:

- Keep **q-banding + clean-region + `recompute_before_dispatch`** as the measured
  winner.
- Do not chase OPD KL/top-k: the synced profile puts KL at about 0.14s/step and
  total OPD head/KL work at about 0.38s/step.
- Do not retry or promote `no_recompute` at the current shape; it does not fit.
- Do not sweep `--gdn-replay-plan-max-packed-tokens` while stateful prefix cache
  is enabled. The active stateful GDN replay path bypasses that cap; the failed
  no-recompute chunk had `fallback_micro_batches=0` and
  `stateful_gdn_forward_calls=3`.

Operational note: the no-recompute run was killed and no run-specific trainer
processes remain. The current desired trainer control still encodes the failed
no-recompute diagnostic; render fresh `recompute_before_dispatch` control before
the next trainer launch.

---

## LATEST OVERRIDE 2026-06-13 20:20Z — q-banding + clean-region is the measured winner

The 19:45Z alignment-only correction still stands, but it is no longer the
latest throughput conclusion. Canonical q-banding was re-enabled on the two
student samplers and then measured both alone and with clean-region replay from
the same step-500 filesystem DCP. **The current best measured config is
q-banding + clean-region replay.**

The relevant OPSD-Wordle throughput pass in
`/home/apanda/xorl-opsd-wordle-apanda-dev-run-20260607` helps here mainly as a
profiling discipline: their CUDA-synchronized run showed backward/recompute/comm,
not OPD KL/top-k, was the bottleneck (`1015.77s` f/b, `969.91s` backward,
~`6.09s` OPD loss, ~`5.40s` KL). The SingleShot-MTP driver already sends
`opd_profile_timings=True`; `OPD_PROFILE_SYNC_CUDA=1` is now baked by
`q36_singleshot_reprogrammable_slots.py` when exported for a diagnostic relaunch.
Do not enable it for promotion wall-clock comparisons unless the goal is exact
phase attribution rather than raw throughput.

| config | run dir | rows | step wall | prefetch wait | sampling | teacher | FB | sync | actual/useful MFU | consumed tok/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline align=256 | `q36mtp-20260613T110939Z-2s1t` | 508-521 | 62.53s | 32.52s | 85.15s | 18.34s | 23.63s | 3.24s | 1.59% / 0.29% | 730 | old reference |
| q-banding only | `q36mtp-20260613T194614Z-2s1t` | 502-514, no terminal | 43.33s | 0.19s | 19.01s | 25.62s | 36.00s | 3.92s | 0.88% / 0.18% | 1147 | sampling fixed, FB inflated |
| q-banding + clean-region | `q36mtp-20260613T200325Z-2s1t` | 502-514, no terminal | 34.62s | 3.03s | 18.55s | 25.00s | 24.98s | 3.37s | 1.38% / 0.28% | 1460 | **new winner** |
| align=128 long | `q36mtp-20260613T190017Z-2s1t` | 508-530, no terminal | 61.46s | 8.26s | 83.86s | 17.92s | 46.75s | 3.28s | 0.58% / 0.14% | 738 | not promoted |

The combined run completed cleanly (`OPD pipeline validation succeeded`,
`trainer-head cleanup rc=0`). Exclude step 500 warmup and step 515 terminal. The
combined config cuts wall by ~45% and roughly doubles consumed tok/s versus the
baseline. It also explains why both halves are needed: q-banding fixes sampling
but inflates FB; clean-region absorbs that FB inflation while keeping the
sampling win.

MFU caveat: the combined config is a wall-clock/throughput winner, not proof that
useful MFU is fully fixed. Useful MFU is back near baseline (~0.28%) rather than
materially higher. The next MFU diagnostic should profile the q-band+clean
winner with `OPD_PROFILE_SYNC_CUDA=1` or a warmed kernel profiler focused on the
trainer fwd/bwd phase. Per the OPSD result, do not prioritize OPD KL/top-k unless
sync timing says the KL region is significant.

## LATEST OVERRIDE 2026-06-13 19:45Z — stateful capture alignment is not a promoted fix

The 18:35Z note that called align=128 the measured winner is superseded. A longer validation from the same
step-500 DCP completed cleanly and showed that align=128 mostly shifts the bottleneck from sampler prefetch
wait to lower-utilization FB/replay. Wall time is essentially flat versus baseline, while useful MFU is much
worse.

All runs below use stack `er-opd-q36-mtp-ss-0605c`, the same step-500 filesystem DCP from
`q36mtp-20260613T110939Z-2s1t`, alltoall/triton trainer, and no q-banding. The long validation run was
`q36mtp-20260613T190017Z-2s1t` with `STUDENT_REPLICAS=2`,
`XORL_SINGLESHOT_MTP_STATEFUL_CAPTURE_ALIGN=128`, and `OPD_MAX_OPD_STEPS=532`; it wrote 32 profile rows and
ended with `trainer-head cleanup rc=0`.

| config | run dir | rows | step wall | prefetch wait | FB | sampling | stateful calls / rounds | GDN tokens | actual/useful MFU | consumed tok/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline align=256 | `q36mtp-20260613T110939Z-2s1t` | 508-521 | 62.53s | 32.52s | 23.63s | 85.15s | 6.0 / 2 | 1.50M | 1.59% / 0.29% | 730 | default still defensible |
| align=128 long | `q36mtp-20260613T190017Z-2s1t` | 508-530, no terminal | 61.46s | 8.26s | 46.75s | 83.86s | 7.9 / 3 | 1.00M | 0.58% / 0.14% | 738 | not promoted |
| align=128 short | `q36mtp-20260613T182155Z-2s1t` | 501-506, no terminal | 57.13s | 14.27s | 36.80s | 69.68s | 7.7 / 3 | 0.82M | 0.70% / 0.18% | 762 | favorable slice only |
| align=64 short | `q36mtp-20260613T180946Z-2s1t` | 501-506, no terminal | 59.00s | 21.04s | 31.50s | 74.04s | 11.7 / 5 | 0.64M | 0.64% / 0.20% | 740 | weak |
| align=32 short | `q36mtp-20260613T175558Z-2s1t` | 501-506, no terminal | 61.35s | 10.86s | 44.02s | 75.57s | 18.2 / 9 | 0.55M | 0.41% / 0.15% | 712 | rejected |
| 4 samplers, align=128 | `q36mtp-20260613T184125Z-4s1t` | 501-506, no terminal | 67.51s | 22.97s | 29.90s | 77.14s | 7.7 / 3 | 0.85M | 0.90% / 0.24% | 646 | rejected: sync ~11.4s |

Important measurement correction: the short A/Bs were capped at `OPD_MAX_OPD_STEPS=508`, so row 507 is a
terminal-row artifact with no next-step prefetch pressure. Exclude row 507 when using those runs for
promotion decisions. The earlier 501-507 and last-three-row averages overstated the align=128 win.

**Interpretation:** baseline has low useful MFU because most FB FLOPs are duplicated GDN replay, but it keeps
FB relatively efficient: the wall is dominated by prefetch wait. Align=128 removes about one third of the GDN
tokens, but increases replay grouping to about 7.9 GDN calls and 3 context rounds, making FB the pole. That
explains the lower actual/useful MFU despite fewer executed replay tokens. This is a bottleneck trade, not a
robust throughput fix.

**Validation state:** live SingleShot MTP unit tests passed (`41 passed`) after adding the env-gated alignment
knob, and the long throughput validation completed cleanly. K3/logprob replay was not run for align=128
because the candidate did not pass the throughput promotion gate.

## TL;DR

1. **Wall decomposes as `step (~80s) ≈ fb_sum (~52s) + prefetch_wait (~25s) + overhead (~3s)`.**
   Both poles are large. FB is the useful-MFU killer; prefetch_wait is the sampler pole.
2. **useful MFU ≈ 0.13%, actual MFU ≈ 3.0%, FLOPs regret ≈ 22–24x.** FB FLOPs are **95% GDN
   (linear-attention) replay**, of which **98.65% is context duplication**.
3. **Stateful GDN replay is in pure-fallback (174/182 = 96% of micro-batches).** Root-caused: the
   native-MTP recompute windows slide *backward* at commit boundaries under faithful
   `commit_len_runtime` (commit-len fix `d54ae9a1`), so each block's visible context is **not a
   contiguous prefix** of the canonical context. The guard at `singleshot.py:1075` correctly returns
   `None`. **This is not a bug to "fix" by relaxing the guard** — doing so would corrupt the OPD
   training signal (proven below).
4. **Canonical q-banding is OFF** (`canonical_q_banding_applied=False` on all 255 trace steps;
   q-lengths fragmented across `{4,5,6,7}`; sampling ≈ **100 tok/s** aggregate). This is the
   prefetch_wait pole.

**Verdicts:**
- **DO NOT PROMOTE** any stateful-replay guard relaxation / pos_id dedup (corrupts training).
- **RECOMMENDED next promoted change: re-enable canonical q-banding** (`--student-mtp-canonical-q-banding`).
  Correctness-neutral for the replay, already validated (prior bench ~2,800 tok/s aggregate), targets the
  ~25s prefetch_wait pole. Gated on the science checkpoint + infra coordination (sampler restart). Exact
  commands below. **Not executed now** — the run is protected.
- **Useful-MFU lever (future work, not promotable without offline validation):** a committed-trunk
  state-capture replay redesign (offline projection ~6x fewer GDN tokens), or — out of throughput scope —
  raising `commit_len` (science: under-trained draft heads).

---

## 1. Wall breakdown (measured, steps 190–197 from the trainer log + steps 176–190 from `opd_profile.jsonl`)

Per-step async pipeline summary line (`trainer-head/logs/20260613T014150Z-run.log`):

```text
step  fb_sum   prefetch_wait  sample_sum  teacher_sum   valid_tokens
190   51.5s    17.4s          145.7s      24.7s         66332
191   52.3s    31.2s          111.8s      15.6s         53514
192   55.0s    27.2s          141.7s      16.5s         62665
193   52.4s    30.2s          131.2s      16.8s         64092
194   43.9s    13.0s          105.9s      15.5s         47658
195   46.1s    27.4s           90.6s      22.6s         41501
196   53.3s    20.6s          123.8s      22.2s         55469
197   50.0s    17.9s          100.6s      15.5s         46120
```

`step_total ≈ fb_sum + prefetch_wait + sync(~3s)`. `prefetch_wait` is the trainer stalling for the next
step's first chunk because sampling does not finish within the FB-overlap budget (`sample_sum` per step
≈ 100–145s aggregate across 2 chunks; chunk-2 prefetches under chunk-1 FB at `wait≈0`, chunk-1 stalls at
`wait≈70s` head-of-step but is mostly hidden by the prior step's FB — the unhidden residue is
`prefetch_wait`).

MFU / replay telemetry (`opd_profile.jsonl`, mean over steps 176–190):

```text
opd_singleshot_mtp_mfu_actual                  0.0301   (~3.0% hardware MFU)
opd_singleshot_mtp_mfu_useful                  0.00134  (~0.13% useful MFU)
opd_singleshot_mtp_flops_regret_ratio          22.5
opd_singleshot_mtp_replay_plan_context_dup_fraction   0.9865
opd_singleshot_mtp_gdn_executed_tokens         464,679 / micro-batch
```

FLOP attribution (representative step 190): `flops_gdn_tf=56,407` of `flops_actual_tf=59,189` →
**95.3% of FB FLOPs are GDN replay**; `flops_useful_tf=2,420` (4%). The low useful MFU is entirely the
GDN context-dup, not attention/MLP/lm_head.

---

## 2. Canonical q-banding: OFF (confirmed), is the sampler pole

- Args file `experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt` contains
  `--student-enable-cuda-graph` and `--student-mtp-adaptive-cuda-graph` but **NOT**
  `--student-mtp-canonical-q-banding`.
- Trace (`rollout_samples.jsonl`, sample 0, 255 steps): `canonical_q_banding_applied=False` on **all**
  steps; `canonical_q_len=None` on all; `decode_q_len_per_req` distribution `{4:40, 5:85, 6:129, 7:1}`.
- `student_sampling_output_tok_per_s ≈ 100` aggregate (vs the resolved sampler bench's ~2,800 aggregate
  with q-banding active). cuda-graph IS on (`cuda_graph_all=True`), but q-len fragmentation prevents
  cross-request batching.

**Measured before/after:** not available — q-banding requires a sampler-control rewrite that restarts
samplers and disturbs the live run; the run is protected (§Operational state). The before-state (100 tok/s,
fragmented q) is measured; the after-state must be obtained from a live A/B once cleared.

**The fix is a one-flag change, fully wired and de-risked:** add `--student-mtp-canonical-q-banding` to
`experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt` (launcher flag at
`q36_singleshot_reprogrammable_slots.py:1853`, default OFF, env `OPD_STUDENT_MTP_CANONICAL_Q_BANDING`).
It emits `--enable-mtp-adaptive-hf-exact-canonical-q-banding --mtp-adaptive-canonical-q-lens 7` (=2k−1,
k=4) to the student sampler. The in-code rationale (`q36_singleshot_reprogrammable_slots.py:686-689`)
describes the EXACT measured symptom: *"conf_adapt fragments the MTP decode batch into per-q_len buckets
(largest bucket runs, the rest pause), collapsing 32-prompt bursts to near-serial decode. Banding maps
every steady step to q_len=2k-1 so all requests share one bucket."* It is designed to pair with
`--student-mtp-adaptive-cuda-graph` (already in the args), which pre-captures graphs for both q_lens
(seed k=4, steady 2k−1=7). No code change, no correctness surface on the FB replay — just the missing
launcher flag, applied via `write-student-inference-control` (restarts samplers) once cleared.

**Tradeoff to measure in the A/B (not in prior docs):** banding pads every decode to `q=2k-1=7`, which
also lands in the FB replay (more mask tokens/block). Sampling win (≈28x) should dominate, and FB is
already context-dup-bound (not mask-count-bound), so net step time should drop — but the A/B must report
`fb_sum` and `prefetch_wait`, not only `tok/s`.

---

## 3. Why stateful GDN replay falls back (root cause — definitive, from real trace data)

`build_rollout_replay_linear_plan` builds, per masked block, a branch whose visible context is
`{context tokens with key_source_indices ≤ query_context_source}` in dense order
(`singleshot.py:906`). `_build_rollout_replay_stateful_schedule` then requires every branch's visible
context to be a **contiguous prefix** of the canonical context sequence (`singleshot.py:1075`:
`sequences[i][1][:clen] != context_tokens[c][:clen] -> return None`), because prefix-state reuse is only
valid for nested branch prefixes.

Native-MTP decode under faithful `commit_len_runtime` (the commit-len fix, `d54ae9a1`, present in live
HEAD) produces recompute windows that **slide backward** between consecutive steps:

```text
step 0 (seed)   recompute_len=1  refill pos_ids = [511]
step 1 (steady) recompute_len=3  refill pos_ids = [512,513,514]
step 2 (steady) recompute_len=2  refill pos_ids = [513,514]   <- drops back from 514 to 513
step 3          recompute_len=3  refill pos_ids = [514,515,516]
step 4          recompute_len=3  refill pos_ids = [515,516,517] <- starts below step 3's 516
```

so the context `key_source_indices` are **non-monotonic** in dense order → a later context token can be
excluded by a block's threshold while an earlier one beyond it is included → not a prefix.

**Measured on the live run's own trace (`rollout_samples.jsonl`, offline reconstruction of the guard):**

| sample/step | #blocks | blocks violating prefix guard | first bad block |
|---|---|---|---|
| s0 step100 | 255 | **122** | 3 |
| s0 step101 | 14  | 3   | 1 |
| s0 step102 | 244 | 136 | 10 |
| s0 step103 | 244 | 118 | 8 |

The guard returns at the FIRST violating branch, and ~half of every row's blocks violate it → **every
micro-batch falls back**. Server log: `174 × schedule_micro_batches=0 fallback_micro_batches=1` vs
`8 × schedule_micro_batches=1`.

**The fallback is CORRECT — relaxing it would corrupt training.** The backward-slide is not noise; the
recompute windows re-process genuine *speculative draft* tokens:

- 215/257 distinct refill pos_ids carry **conflicting tokens** across blocks (the same sequence position
  is re-processed with different speculative tokens in different steps).
- **57%** of recompute-context tokens do **not** match the final committed token at that position
  (e.g. step 1 re-processes pos 513 as token `198` while the committed token is `</think>`=248069).

So each block's visible context is a genuinely different token sequence (committed prefix + that block's
own speculative tail). Treating branches as shared prefixes — or deduping by pos_id — would replay the
**wrong GDN hidden state** and produce a wrong OPD training signal. The 98.65% context-dup is the
faithful cost of replaying `commit_len≈1` native speculative decode.

**Why this regressed:** the stateful prefix-cache (historical win ~1.66–2x, FB 304→45s) was designed for
clean committed-sequence replay where branches ARE nested prefixes. The native-sampler-trace replay
honoring `commit_len_runtime` (post-verify, =1 due to heavy draft rejection on under-trained MTP heads)
introduced backward-sliding speculative recompute, which the prefix assumption cannot represent. It is
**structurally inapplicable while `commit_len≈1`**, independent of any further worktree change.

---

## 4. Verdict

### DO NOT PROMOTE
- Any relaxation of the `singleshot.py:1075` prefix guard, pos_id dedup, or "treat branches as shared
  prefixes" shortcut. Evidence: §3 (57% speculative-divergent tokens; 215/257 conflicting positions).
  It would silently corrupt the OPD signal — exactly the kind of "looks faster, trains wrong" trap the
  promotion gate (`generated_supervised_mismatch_count=0`) exists to catch.

### RECOMMENDED NEXT PROMOTED CHANGE (gated, not executed now)
- **Re-enable canonical q-banding** on the student sampler. It attacks the prefetch_wait pole (~25s/step),
  is correctness-neutral for the FB replay, and is already validated (prior bench ~2,800 tok/s aggregate
  vs current ~100). Projected effect: prefetch_wait → ~0 ⇒ step ~80s → ~55s (≈31%). Does NOT improve
  useful MFU (FB is unchanged) — it is a step-time / sampler-throughput fix, not an MFU fix.

  **Blocked until:** (a) the current-run checkpoint exists (currently `checkpoint_summary.json` shows
  `best_checkpoint: null, latest_checkpoints: []` — the step-100/200 checkpoint is not yet reflected),
  and (b) the science resume handoff is secured, and (c) infra coordinates the sampler restart. Do not
  write student-inference-control before then.

  **Turn-key command (matches the infra handoff's documented procedure, `mtp_infra_agent_handoff.md:286,297`
  — that agent already has this lever queued; this report supplies the measured justification):**
  ```bash
  cd /home/apanda/xorl-mtp-singleshot-port-20260602
  PY=.venv/bin/python
  G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
  A="$(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)"
  # only AFTER the science checkpoint + resume handoff are secured:
  $PY "$G" write-student-inference-control $A --student-mtp-canonical-q-banding
  ```
  A/B success criteria: `student_sampling_mtp_debug_trace_q_lens` collapses (steady → 7); `tok/s` rises
  far above 100 (bench ~2800); `prefetch_wait` → ~0 and step → ~55s; **`rollout/native_trace_coverage_failure_count`
  stays 0** and `student_sampling_mtp_replay_trace_covered_all_targets` stays true. Report `fb_sum` and
  `prefetch_wait` in the A/B, not only `tok/s` (banding pads q to 7, which also lands in FB replay; net
  should still win since FB is dup-bound, not mask-count-bound). Do NOT enable overlap schedule for MTP
  (SGLang hard-rejects MTP requests under overlap).

### USEFUL-MFU LEVER (future work, do-not-promote-without-validation)
- **Committed-trunk state-capture replay.** Redefine the stateful canonical context as the *committed
  trunk* (one token per committed position, the shared nested prefix), capture GDN state at coarse
  boundaries along it, and per-block replay only `[trunk remainder from nearest boundary] +
  [block speculative tail] + [mask]`. Offline projection over 8 live rows: GDN tokens
  **1,038,672 → 171,371 (~6.1x)**, context-dup 0.996 → 0.977. Caveats: the per-block committed-remainder
  term dominates (commit_len=1 ⇒ a supervised block at every position), so the win is ALIGN-sensitive and
  far below the broken stateful path's naive 80x claim; and it is correctness-critical (the supervised
  position's required state must be reconstructed exactly). Needs a from-scratch offline correctness
  validation (`native_trace_coverage_failure_count=0`, `generated_supervised_mismatch_count=0`) before any
  live profile. **Note the coupling:** raising `commit_len` (science: train the draft heads) reduces the
  block count directly and shrinks both this dup AND the speculative re-processing — it is the larger
  lever, but out of throughput scope.

---

## ADDENDUM 2026-06-13 ~05:40Z — MFU lever found + validated offline (clean N-region replay)

After apanda cleared live disruption + clarified the intended SingleShot signal (refer to `~/singleshot`:
`litgpt/mtp.py::interleaved_mtp_mask_mod_factory`, `litgpt/truncate_and_mask_debug_rewrite.py`), the real
useful-MFU lever became clear. It is NOT a "trade correctness for speed" change — it is a **correctness +
throughput** fix.

**The bug.** The live native-trace rollout replay (`_prepare_singleshot_mtp_native_trace_replay_opd_batch`)
makes each MTP attempt's masks attend to `{context with key_source ≤ T}` (`singleshot.py:906`, identical to
the flex mask at `:698`). Under commit_len≈1 this pulls in **other attempts' speculative refills** (verified:
block 10 attends to 23 refills from blocks 0–9 — the same positions with *different, rejected* draft tokens).
The SGLang sampler does NOT do this — it decodes incrementally over the committed KV + its own draft window;
rejected drafts are rolled back. So the replay's cross-attempt over-inclusion **diverges from the sampler's
actual states** (a correctness bug) AND is exactly the non-monotonic `key_source` that breaks the stateful
prefix guard → 96% fallback → 98.65% GDN dup → useful MFU 0.13%.

**The fix.** Restructure the replay to the reference SingleShot N-region signal: one shared **committed NTP
chain** (the accepted rollout tokens, one token/position, monotonic key_source) + **per-attempt mask windows**
that attend to the committed prefix up to their frontier + their own window — NOT other attempts. This
provably nests (monotonic key_source → threshold yields a contiguous prefix) → the stateful GDN path
activates.

**Offline validation (real rollout `q36mtp-20260613T014150Z-2s1t`, 4 rows):**
- Clean layout **nests** (`stateful_schedule` non-None) at every capture-align; current live layout falls
  back 174/182.
- Clean-layout DENSE (no stateful): 491,749 GDN tokens / 4 rows ≈ current live (~462k/microbatch).
- Clean + stateful, sweeping `_STATEFUL_REPLAY_CAPTURE_ALIGN` (currently 256, tuned for the old higher-
  commit_len workload):

  | align | ctx rounds | GDN tokens (4 rows) | reduction |
  |------:|-----------:|--------------------:|----------:|
  | 256 | 2 | 101,349 | 4.9× |
  | 64 | 5 | 29,541 | 16.6× |
  | **32** | **9** | **17,573** | **28×** |
  | 16 | 17 | 11,589 | 42× |

- GDN stateful executor is numerically sound (existing `test_gated_deltanet_*stateful*` pass on GPU).

GDN is 95% of FB FLOPs and memory-bound (time ∝ tokens), so ~26× fewer GDN tokens ⇒ FB ~52s → ~5s ⇒
useful MFU ~0.13% → ~1.5–2%, and step becomes sampling-bound (then q-banding closes it out).

**Status:** implementation in progress (restructure `_prepare_singleshot_mtp_native_trace_replay_opd_batch`
behind a flag; the intricate part is preserving exact teacher-cache/source-index/MTP-head alignment while
moving supervised positions to mask windows). Then offline-validate teacher alignment + live A/B on the pool.

## ADDENDUM 2026-06-13 ~06:00Z — q-banding A/B on the REAL POOL: NEGATIVE result

apanda cleared live disruption. Applied `--student-mtp-canonical-q-banding` via
`write-student-inference-control` (samplers rebooted with banding flags;
`Capture MTP q>1 ... static_q_lens=[4,7] capture_q_lens=[4,5,6,7]`). Trainer wedged on the in-flight
step-263 weight sync → supervisor auto-recovered (resume from step-200 ckpt; q-banding preserved) into a
new run dir `q36mtp-20260613T054455Z-2s1t`.

**Measured q-banding ON (steps 201–203) vs baseline:**

| metric | baseline (off) | q-banding ON |
|---|---|---|
| step_total_s | ~80 | ~83–93 |
| sample_sum_s | ~124 | ~135–143 |
| tok/s | ~100 | ~88–100 |
| fb_sum_s | ~52 | ~65 |
| `canonical_q_banding_applied` | — | **False (249/249)** |
| `decode_q_len_per_req` | {4,5,6,7} | {4:52,5:44,6:51,7:102} (still fragmented) |

**Verdict: q-banding does NOT help on the current live stack — slightly hurts.** The flag captured the
q=7 cuda-graphs but the runtime banding never fired (`canonical_q_banding_applied=False`, `canonical_q_len=None`),
so sampling stayed ~100 tok/s. It only skewed the q-distribution toward 7, which RAISED fb_sum (~52→65s).
Root cause: the live samplers run `xorl-sglang-internal`; the SGL agent's ~2,800 tok/s validation used a
*snapshot* repo (`xorl-sglang-sglagent-20260610`) with a gate fix that is NOT (fully) in the live build —
only the gdn q>1 conv fix was upstreamed, not the runtime q-banding. **Do not chase q-banding further on
this stack** unless the samplers are pointed at the snapshot sglang (OPD_SGLANG_REPO) — separate rabbit hole.

This redirects the throughput effort entirely to the **clean-region MFU fix** (FB lever, sampler-independent),
which is the validated ~26× GDN-token reduction. Operational note: every sampler reprogram wedges the
trainer → supervisor resumes from the step-200 checkpoint (resets progress to 200). So minimize sampler
reprograms; deploy the clean-region fix in ONE coordinated trainer restart (dropping the inert q-banding flag).

## ADDENDUM 2026-06-13 ~06:30Z — clean-region NUMERICAL validation: it CHANGES the signal (~12%)

The live worktree (`xorl-mtp-commitlen-fix-20260612`, b0cfc6de) is more advanced than the analysis worktree:
its `build_rollout_replay_linear_plan` already (a) separates committed vs stale (`_STALE_KEY_SOURCE`) context,
(b) builds a single committed-context sequence, (c) builds per-window branches that reference the committed
prefix + only that window's new members. So "clean-region" is largely already there.

**Reproduced the real fallback (real rollout, live `_prepare`):** the committed context is 790 tokens /
**767 distinct (1.03× dup)** — nearly clean — but **NOT trajectory-sorted in dense order** (~20 local
inversions from recompute-window overlaps). That disorder trips the prefix guard → 96% fallback → 162K-token
dense replay → 98.65% dup → useful MFU 0.13%. So the remaining fix is narrow: **sort + dedup the committed
context into trajectory order.**

**GDN numerical comparison (small GDN layer, real committed-context tokens, GPU):**

```
dup traj_slots with CONFLICTING token values: 0      -> dedup is value-safe (no info loss)
FINAL recurrent state ||dense - clean|| / ||dense|| = 0.117   (11.7%)
per-position output rel-diff: mean 3.3%, max 57%, 28.5% of positions > 1e-3
```

**Verdict: clean-region is NOT bit-identical — it shifts GDN states ~3–12% at supervised positions.** It is
a real training-signal change (ordering + dedup of the committed-context GDN scan), NOT a free speedup.

**But the change is very likely a CORRECTNESS FIX, not a regression:** the SGLang sampler's committed GDN
state advances in sequence order, each committed token once — exactly the clean (sorted+deduped) layout, NOT
the current scrambled+duplicated dense order. So the current replay feeds GDN a slightly-wrong committed-state
ordering vs what the sampler actually had; clean-region corrects it (and unlocks nesting → ~26× MFU). To
*confirm* clean==sampler (not just clean!=current), the next step is to validate replay states against the
sampler's recorded trace logits, or confirm the sampler's `native_mtp_state_cache_steps` committed-cache
semantics. This is a science-correctness call (coordinate w/ science agent) — it is NOT a pure-throughput knob.

## Operational state (why nothing was promoted live)

- Run `q36mtp-20260613T014150Z-2s1t` is **alive** (step 197 at 04:04 UTC; profile written 04:03:50).
- Infra agent is ON WATCH (`AGENT_NOTES.md` 2026-06-13T03:59:54Z: "everything healthy").
- **No checkpoint is saved yet** for this run (`checkpoint_summary.json`: empty), so the science
  diagnostic's resume point does not exist. Interrupting now would lose progress. Per the handoff, do not
  rewrite sampler/trainer control until the science checkpoint + resume handoff are secured.

---

## Exact repro commands (read-only; the infra agent can re-run without guesswork)

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
RUN=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t

# (a) wall + MFU + dup, last 15 rows
.venv/bin/python - <<'PY'
import json, pathlib, statistics as st
rows=[json.loads(l) for l in pathlib.Path(f"{__import__('os').environ.get('RUN','')}/artifacts/opd_profile.jsonl").open()] if False else \
     [json.loads(l) for l in pathlib.Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/artifacts/opd_profile.jsonl").open()]
for r in rows[-15:]:
    print(r["step"],"fb",round(r.get("trainer_forward_backward_s",0),1),
          "samp",round(r.get("student_sampling_s",0),1),
          "tok/s",round(r.get("student_sampling_output_tok_per_s",0),1),
          "mfu_use",round(r.get("opd_singleshot_mtp_mfu_useful",0),5),
          "regret",round(r.get("opd_singleshot_mtp_flops_regret_ratio",0),1),
          "ctxdup",round(r.get("opd_singleshot_mtp_replay_plan_context_dup_fraction",0),4))
PY

# (b) pipeline poles (fb_sum / prefetch_wait / sample_sum)
grep -E "Async forward_backward step=" /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T014150Z-run.log | tail -10

# (c) stateful fallback ratio
rg -o "schedule_micro_batches=[0-9]+ fallback_micro_batches=[0-9]+" "$RUN/server.log" | sort | uniq -c

# (d) q-banding off + q-len fragmentation (from the native trace)
.venv/bin/python - <<'PY'
import json,pathlib,collections
s=json.loads(pathlib.Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/artifacts/rollout_samples.jsonl").open().readline())
qb=collections.Counter(st.get("canonical_q_banding_applied") for st in s["native_mtp_debug_trace"])
ql=collections.Counter(st.get("decode_q_len_per_req") for st in s["native_mtp_debug_trace"])
print("canonical_q_banding_applied:",dict(qb)," decode_q_len_per_req:",dict(ql))
PY
```

The guard-violation reconstruction and the speculative-divergence / committed-trunk projection scripts
used in §3–§4 are inline in this session's transcript (offline, operate only on `rollout_samples.jsonl`).
