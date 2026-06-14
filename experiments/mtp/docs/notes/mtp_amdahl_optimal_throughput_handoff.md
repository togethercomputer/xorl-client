# Handoff: OPD-MTP throughput improvement runbook

Date: 2026-06-13
Stack: `er-opd-q36-mtp-ss-0605c`
Model: Qwen3.6-35B-A3B, native SingleShot-MTP, ConfAdapt, k=2
Live trainer code: `/home/apanda/xorl-mtp-commitlen-fix-20260612`
Analysis checkout: `/home/apanda/xorl-mtp-singleshot-port-20260602`
SGLang code: `/home/apanda/xorl-sglang-internal`

Post-consolidation canonical locations:

- engine landing base: `/home/apanda/xorl-stage-mtp` branch `pr/mtp` / `mtp-merge-apanda-dev` at `3a6a6a02`
- client docs and non-k8s harness: `/home/apanda/xorl-client/experiments/mtp`
- k8s/manifests: `/home/apanda/xorl-infra/k8s/opd_profile`
- run outputs: `/shared`

This replaces the old incremental override log. The pre-consolidation version is
archived at:

`docs/notes/mtp_amdahl_optimal_throughput_handoff_archive_20260613_pre_consolidation.md`

## Current conclusion

The current promotable throughput control is:

- student samplers: k=2 with canonical q-banding enabled
- trainer: clean-region replay context enabled
- trainer checkpointing: `recompute_before_dispatch`
- trainer topology: EP32, `alltoall` dispatch, triton MoE
- pipeline: 64 prompts per step, two 32-prompt prepare chunks, no trainer coalescing
- checkpoints disabled for short validation

Do not chase OPD KL/top-k. CUDA-synced profiling put the model-side time in
trainer forward/loss, backward/recompute, and communication. OPD loss/KL is
small relative to f/b.

Do not compare full-stack throughput runs across different sampler states as if
they are same-workload A/Bs. The student changes the workload over time. In the
late runs, commit length and sampler tok/s changed enough to dominate wall time.
Use either same-checkpoint/same-sampler-state full-stack runs or trainer-only
replays of captured payloads.

## Latest validation

The stack was reprogrammed to the current q-band + clean-region control and the
capped validation completed cleanly:

| item | value |
|---|---|
| run | `q36mtp-20260613T223322Z-2s1t` |
| W&B | `btqzrlm6` |
| profile | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T223322Z-2s1t/artifacts/opd_profile.jsonl` |
| trainer log | `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T223321Z-run.log` |
| control root | `/shared/opd-control/er-opd-q36-mtp-ss-0605c` |
| supervisor | paused |
| terminal log | `OPD pipeline validation succeeded`; W&B synced |

The first row was a cold/low-accept row:

| step | wall | trainer f/b | sample | teacher | sync | consumed tok/s | commit_len |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 112.00s | 64.05s | 35.91s | 16.57s | 12.05s | 413 | 1.156 |

The warm windows show the sampler-state effect directly:

| window | wall | trainer f/b | sample | teacher | prefetch wait | sync | consumed tok/s | sample tok/s | commit_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 501-509 | 37.83s | 28.30s | 19.65s | 25.43s | 3.06s | 2.99s | 1333 | 772 | 1.686 |
| 502-509 | 35.02s | 25.10s | 19.18s | 25.93s | 3.44s | 2.99s | 1406 | 802 | 1.761 |

Step 501 was still a transition row (`commit_len=1.089`). By step 502, commit
length had climbed to the 1.73-1.80 range, which is why the steadier 502-509
window is the better comparison to earlier q-band + clean runs. This validates
the control and confirms the user's hypothesis: as the student changes,
confidence and sampler output change too.

The live control summary currently says:

- `trainer_ep_dispatch=alltoall`
- `trainer_moe_implementation=triton`
- `trainer_gradient_checkpointing_method=recompute_before_dispatch`
- `trainer_expert_parallel_size=32`
- `gdn_replay_plan_use_stateful_prefix_cache=True`
- `trainer_clean_replay_context=True`
- `prompts_per_step=64`
- `pipeline_chunk_size=32`
- `trainer_coalesce_chunks=1`
- `k_toks=2`
- `max_opd_steps=510`
- `checkpoint_interval_steps=0`
- `checkpoint_save_best=False`

## How to inspect the live run

```bash
cd /home/apanda/xorl-mtp-commitlen-fix-20260612

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py status \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)

tail -n 120 /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T223321Z-run.log

python - <<'PY'
import json, statistics
from pathlib import Path
p = Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T223322Z-2s1t/artifacts/opd_profile.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print("rows", len(rows))
for row in rows:
    print({
        "step": row.get("iter", row.get("step")),
        "wall": round(row.get("iter_time", 0.0), 2),
        "trainer_fb": round(row.get("trainer_forward_backward_s", 0.0), 2),
        "prefetch_wait": round(row.get("opd_async_prefetch_wait_s", 0.0), 2),
        "sample_s": round(row.get("student_sampling_s", 0.0), 2),
        "sample_tok_s": round(row.get("student_sampling_output_tok_per_s", 0.0), 1),
        "commit_len": round(row.get("mtp/commit_len_mean", 0.0), 3),
        "consumed_tok_s": round(row.get("consumed_toks_per_sec_world", 0.0), 1),
    })
PY
```

## How to relaunch the current canonical validation

Use this only if the current run has exited or must be re-rendered. Keep the
supervisor paused while doing diagnostic throughput runs.

```bash
cd /home/apanda/xorl-mtp-commitlen-fix-20260612

export OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612
export OPD_START_STEP=500
export OPD_LOAD_CHECKPOINT_PATH=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T110939Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000500

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-student-inference-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)

python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py \
  write-trainer-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt) \
  --max-opd-steps 510 \
  --checkpoint-interval-steps 0 \
  --no-checkpoint-save-best
```

The canonical args file now includes `--k-toks 2`,
`--student-mtp-canonical-q-banding`, `--trainer-clean-replay-context`,
`--trainer-ep-dispatch alltoall`, `--trainer-moe-implementation triton`,
`--prompts-per-step 64`, and `--trainer-coalesce-chunks 1`.

## Evidence table

Warm windows exclude cold step 500 and terminal cleanup rows unless noted.

| experiment | run / artifact | result | decision |
|---|---|---|---|
| k=2 science baseline, no q-band/clean | `q36mtp-20260613T110939Z-2s1t`, steps 508-521 | 62.53s/step, sampling 85.15s, teacher 18.34s, trainer f/b 23.63s, consumed 730 tok/s | Baseline was sampling-bound. |
| q-banding only | `q36mtp-20260613T194614Z-2s1t` | 43.33s/step, prefetch wait 0.19s, sampling 19.01s, trainer f/b 36.00s, consumed 1147 tok/s | Fixed sampling; exposed/inflated trainer f/b. Needs clean-region. |
| q-banding + clean-region | `q36mtp-20260613T200325Z-2s1t`, W&B `tu5ffak6` | 34.62s/step, sampling 18.55s, teacher 25.00s, trainer f/b 24.98s, sync 3.37s, consumed 1460 tok/s | Best completed full-stack result. |
| q-band + clean CUDA-synced profile | `q36mtp-20260613T201834Z-2s1t`, W&B `krofvfsq` | 35.34s/step, trainer f/b 26.96s, forward/loss 9.20s, backward 15.10s, OPD total 0.379s, KL 0.140s | Bottleneck is model f/b and comm/recompute, not KL/top-k. |
| no recompute | `q36mtp-20260613T203140Z-2s1t`, W&B `h57g4v8g` | OOM on first step-500 f/b in stateful GDN suffix path; zero profile rows | Do not retry as-is. |
| q-band + clean capture | `q36mtp-20260613T204443Z-2s1t` | Warm steps 501-506: 32.16s/step, trainer f/b 22.18s, sampler 830 tok/s, commit_len 1.80 | Strong q-band + clean reference window. |
| trainer-only two-payload replay | `fb_capture_qband_clean_step500_20260613T204431Z`, payloads 11 + 8 | Sequential warmed two calls: 18.21s trainer f/b, 1356 valid tok/s, actual MFU proxy 1.59% | Model-side replay harness works; useful for same-payload A/B. |
| combined 64-sample trainer replay | `q36mtp-20260613T211419Z-2s1t` | One combined call warmed ~10.11s, 2440 valid tok/s, actual MFU proxy 2.44% | Positive trainer-call amortization signal, trainer-only only. |
| full pipeline chunk size 64 | `q36mtp-20260613T211900Z-2s1t`, W&B `i3ji92nq` | 40.90s/step, trainer f/b 25.74s, prefetch wait 8.71s, consumed 1172 tok/s | Do not promote; loses prepare parallelism. |
| trainer coalesce-2 with DeepEP | `q36mtp-20260613T215028Z-2s1t`, W&B `75424yvk` | Failed first coalesced f/b with `DeepEP error: timeout (dispatch CPU)` | DeepEP cannot handle this merged k=2 payload as-is. |
| EP8/G-shrink replay | `q36mtp-20260613T220513Z-2s1t` plus replay JSONL | EP8 base-weight replay: 26.41s vs EP32 replay 18.21s; per-GPU MFU proxy improves, wall throughput worsens | Cost-efficiency future branch, not wall-clock fix. |
| EP32 checkpoint into EP8 | `q36mtp-20260613T220042Z-2s1t` | Optimizer state shape mismatch on expert momentum buffer | Needs model-only load or optimizer conversion before EP shrink science run. |
| trainer coalesce-2 with alltoall/triton | `q36mtp-20260613T221209Z-2s1t`, W&B `qm62obaw` | Clean, but warm 59.37s/step, trainer f/b 27.31s, sampler 139 tok/s, commit_len 1.07 | Not promotable; sampler state changed and trainer did not improve. |
| current q-band + clean validation | `q36mtp-20260613T223322Z-2s1t`, W&B `btqzrlm6` | Completed cleanly. Warm 501-509: 37.83s/step, trainer f/b 28.30s, sample 19.65s, teacher 25.43s, sync 2.99s, consumed 1333 tok/s, commit_len 1.686. Steadier 502-509: 35.02s/step, trainer f/b 25.10s, consumed 1406 tok/s, commit_len 1.761. | Confirms current control and sampler confidence drift; do not over-rank against older sampler states. |

## Why OPSD microbench still helps, but only as a checklist

The OPSD low-MFU microbench decomposes visible MFU into:

- denominator choice: executed tokens vs valid target tokens
- dummy-fill / rank occupancy
- above-model server overhead
- in-model small-GEMM and communication shape

Do not copy the OPSD numeric answer to MTP. MTP q-band + clean-region payloads
already execute large GDN replay work per call, and trainer-only replay showed
that some slow live calls warm away. For MTP, the latest profile points to
backward/recompute/FSDP/EP communication rather than OPD KL/top-k or a fixed bad
payload.

## Clean-region replay status

Clean-region replay is enabled by:

```bash
XORL_SINGLESHOT_MTP_CLEAN_REPLAY_CONTEXT=1
```

The proof run showed clean-region and current ordering are top-1
prediction-equivalent for the sampled proof batch:

- current top-1 agreement: 0.500
- clean top-1 agreement: 0.476
- current == clean in 8/8 examples

Caveat: this is an argmax-level proof. It does not prove the full KL/logit
distribution is identical. Treat it as safe enough for throughput validation,
not as a mathematical equivalence proof for all science claims.

The live launcher now has `--trainer-clean-replay-context` and validates that it
is only used with stateful GDN replay prefix cache.

## SGLang dependency

Canonical q-banding previously crashed the sampler with:

```text
ValueError: Invalid verified MTP commit length. verified=2 planned=1 phase=steady
```

The SGLang fix is now committed and pushed on `apanda-dev`:

```text
18c2357de Fix per-request MTP draft verification
```

It adds per-request `mtp_hf_exact_accept_len` verification and unit tests in
`test/srt/zorl/test_mtp_decode_verify.py`. The MTP engine landing should be
paired with this SGLang commit.

## Rules for the next throughput agent

1. Do not promote from cross-state full-stack A/Bs.
   If commit_len or sampler tok/s changed, the workload changed.

2. Do not retry no-recompute without a memory plan.
   It OOMed in stateful GDN suffix replay on the first f/b.

3. Do not promote global `pipeline_chunk_size=64`.
   It reduced prepare parallelism and regressed wall-clock.

4. Do not retry `trainer_coalesce_chunks=2` on DeepEP.
   The merged k=2 payload hit DeepEP dispatch CPU timeout.

5. If revisiting coalescing, start from alltoall/triton and compare same
   captured payloads first.
   The alltoall full-stack coalesce run completed but was not faster.

6. Use trainer-only replay for model-side changes.
   It removes sampler drift and isolates f/b, FSDP, EP dispatch, GDN suffix, and
   compile effects.

7. Preserve q-banding + clean-region + recompute_before_dispatch as the baseline
   until a same-workload run beats it.

8. Keep outputs under `/shared`.
   The OPD generator now sets future `WANDB_DIR=${RUN_DIR}/artifacts/wandb`, and
   local-benchmark k8s rendering defaults to `/shared/xorl-local-benchmark`.

## Next useful work

1. If optimizing throughput next, capture current-step f/b payloads and replay
   them trainer-only before touching full-stack controls.

2. Candidate model-side code targets:
   reduce stateful GDN suffix work, reduce FSDP all-gather/reduce-scatter
   frequency, or find a safe coalescing implementation that does not hit DeepEP
   timeout and does not lose sampler/teacher prepare overlap.

3. Candidate science path:
   if science wants EXP-1 continuation, use q-band + clean only if they accept
   that sampler behavior is now part of the training state. Otherwise render a
   science-control run that preserves their intended sampler setting and do not
   compare its wall-clock to q-band runs as a throughput A/B.

## Consolidation notes

The throughput harness edits are not engine code. Engine landing is covered by
`CONSOLIDATION_HANDOFF.md`; the paired SGLang q-banding fix is already pushed.
Post-B2 cleanup remains gated on the MTP engine PR landing into `apanda-dev`.
