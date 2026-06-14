# Superseded

Use `docs/notes/mtp_amdahl_optimal_throughput_handoff.md` as the current
throughput runbook. This earlier handoff is retained for historical context.

# Handoff: MTP throughput / low-MFU investigation

> **STATUS 2026-06-13 ~04:10Z — ANSWERED (read-only; run NOT interrupted).**
> Full measured report: **`docs/notes/mtp_throughput_findings_20260613.md`**.
> - Wall: `step ~80s = fb_sum ~52s + prefetch_wait ~25s + ~3s`. useful MFU ~0.13%, regret ~22x; FB is
>   95% GDN replay @ 98.65% context-dup.
> - **P1 q-banding: confirmed OFF** (`canonical_q_banding_applied=False` all steps; q-len `{4,5,6,7}`;
>   ~100 tok/s). RECOMMENDED next promoted change — re-enable `--student-mtp-canonical-q-banding`
>   (correctness-neutral, validated ~2800 tok/s bench). Gated on the science checkpoint + infra restart.
> - **P2 stateful fallback: root-caused.** Backward-sliding speculative recompute under faithful
>   `commit_len_runtime` (`d54ae9a1`) breaks the prefix guard (`singleshot.py:1075`). The fallback is
>   CORRECT — 57% of recompute tokens are speculative-divergent; **DO NOT relax/dedup the guard** (would
>   corrupt the OPD signal). Useful-MFU lever = committed-trunk replay redesign (~6x offline projection,
>   needs validation) or raising commit_len (science).

**Date:** 2026-06-13
**Agent direction:** throughput
**Stack:** `er-opd-q36-mtp-ss-0605c`
**Repo:** `/home/apanda/xorl-mtp-singleshot-port-20260602`

## Mission

Explain why OPD-MTP useful MFU is still extremely low and produce the next measured throughput fix. Do not make the science call about whether SingleShot MTP is learnable, and do not own deployment recovery except for carefully scoped throughput experiments coordinated with the infra agent.

The current live run already emits true SingleShot-MTP FLOP/MFU fields. Use those fields, plus sampler timing and replay-plan telemetry, instead of older proxy-only docs.

## Current live state to anchor on

As of the read-only inspection on 2026-06-13 around 03:20 UTC:

- Live run: `q36mtp-20260613T014150Z-2s1t`
- Artifacts: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/`
- Trainer log: `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T014150Z-run.log`
- Server log: `<run-dir>/server.log`
- Profile rows: `<run-dir>/artifacts/opd_profile.jsonl`
- Rollout trace: `<run-dir>/artifacts/rollout_samples.jsonl`
- Trainer config rendered at `/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/generated_trainer_config.yaml`

Rendered trainer config highlights:

```yaml
moe_implementation: triton
ep_dispatch: deepep
deepep_num_sms: 48
expert_parallel_size: 32
data_parallel_shard_size: 32
ulysses_parallel_size: 1
batch_parallel_mode: dp_shard
gradient_checkpointing_method: recompute_before_dispatch
fsdp_reduce_dtype: fp32
enable_packing: true
sample_packing_sequence_len: 4608
linear_replay_plan_max_packed_tokens: 16384
linear_replay_plan_use_stateful_prefix_cache: true
lr: 1.0e-05
muon_lr: 0.001
```

Current long-run args are in:

```text
experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt
```

Important drift from older handoffs:

- Older MFU docs promoted `quack + deepep sms24` for this workstream. The current live run and args file use `triton + deepep sms48`.
- Older sampler docs said q-banding was the key sampler fix. The current args include `--student-enable-cuda-graph` and `--student-mtp-adaptive-cuda-graph`, but do **not** include `--student-mtp-canonical-q-banding`.
- The current run uses `prompt_dataset_turn_strategy=suffix`, `prompts_per_step=64`, and `mtp_static_padded_seq_len=2304`.

Science coordination constraint:

- The next science diagnostic is Qwen3.6 live-stack low-k/curriculum from the current live run's checkpoint.
- Do not interrupt the current run for sampler-control rewrites until that current-run checkpoint exists and the science
  resume handoff is secured, unless apanda explicitly clears the throughput interruption sooner.
- Read-only throughput analysis can proceed immediately against `opd_profile.jsonl`, `rollout_samples.jsonl`, and
  `server.log`.

## Current measured symptoms

Recent profile rows from steps 154-165:

```text
step_total_s:           75.9-91.8
trainer_forward_backward_s: 48.8-70.4
student_sampling_s:     113.5-143.9 aggregate across two 32-prompt chunks
teacher_prefill_s:      15.7-24.0
opd_singleshot_mtp_mfu_actual: roughly 0.024-0.034
opd_singleshot_mtp_mfu_useful: roughly 0.0010-0.0014
mtp/commit_len_steady:  roughly 1.03-1.05
```

Representative step 165:

```text
step_total_s=75.93
trainer_forward_backward_s=68.72
student_sampling_s=134.54
teacher_prefill_s=23.18
opd_singleshot_mtp_mfu_actual=0.02447
opd_singleshot_mtp_mfu_useful=0.00105
opd_singleshot_mtp_flops_regret_ratio=23.32
student_sampling_output_tok_per_s=100.34
sync_inference_weights_s=3.07
```

That means the actual hardware MFU is only about 2-3%, and the useful MFU is about 0.1%. The wall is not a single component: FB is large, sampler throughput is far below the resolved sampler-bench numbers, and useful FLOPs are diluted by replay/context work.

## First conclusion: do not trust the old one-line bottleneck

Earlier docs correctly found and fixed several historical issues:

- EP-group batch replication was fixed by `batch_parallel_mode=dp_shard`.
- Statefulness and sample packing improved some replay profiles.
- Sampler cuda graphs and q-banding were shown to be required for fast native-MTP sampling.
- Ordinary non-MTP 32-GPU training can reach roughly 14.7% logical MFU on this model, so the hardware is not inherently limited to 0.1% useful MFU.

But the current live run has two fresh red flags:

1. **Sampler-side q-banding appears absent.** Current sampler traces report `student_sampling_mtp_debug_trace_q_lens: "4,5,6,7"` and only about `100 tok/s` aggregate at step 165. The resolved sampler handoff said canonical q-banding to `2*k_toks-1` was the fix that restored cross-request batching and produced about 2,800 tok/s aggregate in a bench. If q-banding is missing, sampling is again badly bottlenecked.

2. **Stateful GDN replay is mostly falling back.** Although `linear_replay_plan_use_stateful_prefix_cache: true`, the server log repeatedly shows `stateful schedule_micro_batches=0 fallback_micro_batches=1`. Some chunks do get `schedule_micro_batches=1`, but most recent chunks are fallback. The live rows still show context duplication around 98.5-98.8% and large GDN executed-token counts. The commit-len fix worktree may have invalidated the prefix assumption that stateful replay depends on.

The throughput agent should validate these two before reopening topology sweeps.

## Priority 1: sampler q-banding check

Goal: determine whether current sampling is slow because canonical q-banding is missing or not active.

Evidence to collect:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
RUN=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t

.venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/artifacts/opd_profile.jsonl")
for line in list(p.open())[-20:]:
    r = json.loads(line)
    print(
        r["step"],
        "step_s", round(r.get("step_total_s", 0), 2),
        "sample_s", round(r.get("student_sampling_s", 0), 2),
        "tok_s", round(r.get("student_sampling_output_tok_per_s", 0), 2),
        "q_lens", r.get("student_sampling_mtp_debug_trace_q_lens"),
        "cuda_all", r.get("student_sampling_mtp_debug_trace_cuda_graph_all"),
    )
PY
```

Check current args:

```bash
rg -n "student-(mtp-canonical-q-banding|enable-cuda-graph|mtp-adaptive-cuda-graph|max-total-tokens|max-running-requests)" \
  experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt
```

If q-banding is absent, the likely experiment is to add `--student-mtp-canonical-q-banding`, rewrite student inference
control, and run a short end-to-end comparison. Coordinate this with the infra agent first because
`write-student-inference-control` restarts samplers and will disturb the live run. Do not perform that write before the
science current-run checkpoint is secured unless apanda explicitly clears it.

Success criteria:

- `student_sampling_mtp_debug_trace_q_lens` should collapse to the canonical q length instead of `"4,5,6,7"`.
- `student_sampling_output_tok_per_s` should move far above the current about 100 tok/s aggregate; prior bench evidence says the target is in the thousands aggregate if the same SGL path is active.
- `rollout/native_trace_coverage_failure_count` stays 0 and `student_sampling_mtp_replay_trace_covered_all_targets` stays true.

Do not enable overlap schedule for MTP. The sampler handoff says SGLang hard-rejects MTP requests when overlap scheduling is enabled.

## Priority 2: stateful replay fallback root cause

Goal: explain why the current run has stateful replay enabled but usually falls back to packed replay.

Fast evidence:

```bash
RUN=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t
rg "stateful schedule_micro_batches" "$RUN/server.log" | tail -80

.venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/artifacts/opd_profile.jsonl")
for line in list(p.open())[-12:]:
    r = json.loads(line)
    print(
        r["step"],
        "fallback", r.get("opd_singleshot_mtp_replay_plan_stateful_fallback_micro_batches"),
        "stateful", r.get("opd_singleshot_mtp_replay_plan_stateful_schedule_micro_batches"),
        "ctx_dup", round(r.get("opd_singleshot_mtp_replay_plan_context_dup_fraction", 0), 4),
        "gdn_exec", r.get("opd_singleshot_mtp_gdn_executed_tokens"),
        "regret", round(r.get("opd_singleshot_mtp_flops_regret_ratio", 0), 2),
    )
PY
```

Likely code paths:

- `src/xorl/mtp/singleshot.py`
  - `build_rollout_replay_linear_plan`
  - `_build_rollout_replay_stateful_schedule`
  - prefix-validity guard that returns `None`
  - replay visibility and emit supervision changes in `/home/apanda/xorl-mtp-commitlen-fix-20260612`
- `src/xorl/ops/linear_attention/layers/gated_deltanet.py`
  - `_forward_with_replay_plan_stateful_prefix_cache`
  - `_forward_with_replay_plan_packed_chunks`
- `src/xorl/server/runner/model_runner.py`
  - `[singleshot-mtp-profile]` logging

Hypothesis to test:

The commit-len fix made replay visibility faithful by hiding stale rejected drafts. That may make some branch-visible contexts no longer prefixes of their context sequences, so the existing stateful schedule guard falls back. If true, stateful replay has to be generalized to the new trajectory-slot visibility, or the throughput path needs a new single-pass state capture design.

Promotion gate:

- No correctness relaxation. The run must keep `rollout/native_trace_coverage_failure_count=0`, `rollout/generated_supervised_mismatch_count=0`, and `OPD pipeline validation succeeded`.
- Compare `trainer_forward_backward_s / valid_tokens`, `opd_singleshot_mtp_mfu_actual`, `opd_singleshot_mtp_mfu_useful`, and `flops_regret_ratio`, not raw step time only.

## Priority 3: profile the current stack if still ambiguous

Use profiling only on short, non-destructive runs. Do not turn on the profiler for normal training; it can inflate wall time heavily.

Before any profile run:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
CTL=/shared/opd-control/er-opd-q36-mtp-ss-0605c
tail -20 "$CTL/AGENT_NOTES.md"
.venv/bin/python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py status --stack er-opd-q36-mtp-ss-0605c
```

If the live science run is still active, do not interrupt it without explicit coordination. If cleared, pause the supervisor and run a one-step profiler:

```bash
touch /shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.pause

XORL_PROFILE_SERVER_FB=1 XORL_PROFILE_SERVER_FB_SKIP=0 XORL_PROFILE_SERVER_FB_COUNT=1 \
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
.venv/bin/python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py write-trainer-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt) \
  --num-steps 1 --max-opd-steps 1 --prompts-per-step 8 --pipeline-chunk-size 8 \
  --pipeline-prefetch-chunks 1 --pipeline-teacher-concurrency 1 \
  --skip-optim-step --checkpoint-interval-steps 0 --no-checkpoint-save-best
```

Expected artifacts:

```text
<run-dir>/server_output/fb_profiles/fb_trace_call0.json.gz
<run-dir>/server_output/fb_profiles/fb_keyavg_call0.txt
<run-dir>/artifacts/opd_profile.jsonl
```

## What not to spend first effort on

- Do not reopen EP width, CP, or alltoall by default. Current live config is already EP32, dp_shard, non-CP.
- Do not assume "FSDP comm is the wall" from the old profile without checking current run telemetry first. The current live rows show major GDN replay duplication and slow sampler throughput.
- Do not use local ordinary-training sweep winners as direct OPD recipes. They prove the hardware and model can run much higher MFU, but OPD has replay, teacher, sampler, weight sync, and sparse useful-supervision effects.
- Do not scale samplers just because sampling is slow until you check q-banding. The cheap fix is likely activating the q-banding path that was already validated.

## Deliverable

Produce a short measured report with:

- Current wall breakdown over at least 10 recent rows.
- Whether canonical q-banding is active and its measured before/after effect.
- Why stateful replay falls back under the current commit-len fix worktree.
- One promoted change or one explicit "do not promote" verdict with exact evidence.
- Exact commands and artifacts used so the infra agent can reproduce without guesswork.
