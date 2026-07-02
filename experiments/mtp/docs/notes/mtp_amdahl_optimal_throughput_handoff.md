# MTP Throughput Handoff - What Worked And What Did Not

Last updated: 2026-06-15 16:54Z

This is a summary handoff for the MTP MFU/throughput work on
`er-opd-q36-mtp-ss-0605c`. It replaces the prior chronological runbook. The
removed long-form version is preserved at
`archive/mtp_amdahl_optimal_throughput_handoff_20260615T1654Z_chronological.md`.

Authoritative config summary:
`/shared/opd-control/er-opd-q36-mtp-ss-0605c/MTP_CONFIG_OF_RECORD.md`

## Plain-English Summary

I improved the measurement setup more than the live config. The biggest concrete
accomplishment was building a replay harness that let us compare trainer changes
on identical captured workloads without disturbing the science run.

The strongest safe performance signal is topology: EP8 is faster than EP32 in
trainer-only replay because expert all-to-all stays within a node. That has not
been promoted because it changes the shared live trainer topology.

The biggest raw trainer f/b speedups came from shorter static shapes, but those
changed packing or microbatch geometry and produced measurable loss deltas. They
are useful evidence, not a live config.

Most execution knobs were noise or negative once tested on the same workload.
The current live stack remains v2: EP32/alltoall/triton, clean-region ON, skip
outer trainer f/b defrag ON, static4352, defer OFF.

Current measured full-stack MFU is from `q36mtp-20260615T115717Z-2s1t` step 3504:
actual MFU `2.246%`, useful MFU `0.228%`. The last-20-step mean is actual
`2.103%`, useful `0.211%`. A newer run directory
`q36mtp-20260615T161813Z-2s1t` failed distributed rendezvous and produced no MFU
rows.

## Current Promoted Live Config

| Area | Value |
|---|---|
| Engine | `/home/apanda/xorl-mtp` on `apanda-dev-mtp` |
| Active rung | k=8, files64 Coderforge stream, confidence threshold 0.3 |
| Trainer topology | 4 nodes / 32 GPUs, EP32 |
| Dispatch and MoE | `alltoall`, `triton` |
| Checkpointing | `recompute_before_dispatch` |
| Clean-region replay | ON |
| Skip outer f/b defrag | ON |
| Stateful GDN prefix cache | ON |
| Static padded seq len | 4352 in the latest promoted live profile |
| Pipeline | 64 prompts/step, 2x32 chunks, coalesce chunks 1 |
| Defer grad sync | OFF |
| Dynamic static flags | OFF |

## Measurement Method That Worked

The reliable method was isolated trainer-only replay of captured
`/forward_backward` payloads:

- k=8 capture: `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z`
- k=4 capture: `/shared/opd-control/er-opd-q36-mtp-perf-replay/k4_files64_capture_clean_20260614T234233Z`

This avoided a key failure mode in earlier comparisons: the student sampler
changes the workload over time, so ordinary full-stack A/Bs can confuse sampler
state with trainer speed.

## What Worked

### EP8 Topology

EP8 is the real same-workload throughput lever found in this pass. It is a pure
layout change: training math is unchanged, but the MoE all-to-all is intra-node
instead of cross-node.

| Test | Result | Status |
|---|---|---|
| k=4, 4-node EP8 vs EP32 | EP8 f/b 10.8664s vs EP32 13.6694s; MFU 2.063% vs 1.640% | validated replay win |
| k=8, 2-node EP8 no-defer | f/b 10.8665s, roundtrip 12.0561s, pseudo actual MFU 3.048% | validated replay win |

Not promoted because it changes live topology and needs a deliberate shared-stack
relaunch.

### Shorter Static Shapes

Shorter static shape was the largest raw trainer-only f/b speedup. It is not
live-safe yet because the faster forms change pack/microbatch geometry and move
loss.

| Candidate | Result versus static4352 repeat | Loss delta | Status |
|---|---:|---:|---|
| static2304, two-sample pack | f/b -45.6%, useful MFU 1.093% vs 0.595% | 0.004722 | not promoted |
| static1920, two-sample pack | f/b -48.6%, useful MFU 1.159% vs 0.595% | 0.007348 | not promoted |
| static2048, two-sample pack | f/b 7.8042s, slower than static1920 | near static1920 | not promoted |
| static1920, one-sample pack | f/b -16.5%, useful MFU +19.7% | 0.002707 | not promoted |
| static1664 | failed on `raw_seq_len=1795` > static limit | n/a | rejected |

The one-sample static1920 bracket kept the same microbatch count as static4352,
which showed that most of the loss drift came from pack/microbatch geometry.

### Default-Off Dynamic Static Support

Engine and generator support was added for dynamic static padding and dynamic
static packing, but all flags default to off.

Touched engine areas:

- `src/xorl/server/orchestrator/packing.py`
- `src/xorl/server/orchestrator/request_processor.py`
- `src/xorl/server/runner/model_runner.py`
- generator copies in the engine and infra checkouts

Focused validation:

```bash
PYTHONPATH=/home/apanda/xorl-mtp/src uv run pytest \
  tests/server/orchestrator/test_packing.py \
  tests/server/orchestrator/test_request_processor.py \
  tests/server/runner/test_opd_runner.py \
  tests/experiments/test_q36_singleshot_reprogrammable_slots.py -q
```

Observed result: `102 passed, 16 warnings`.

Measured dynamic candidates:

| Candidate | Result | Status |
|---|---|---|
| conservative dynamic static, static4352 cap | f/b 11.8890s vs 14.8162s, geometry preserved, max loss delta 0.003138 | not promoted |
| dynamic static packing, static4352 cap | f/b 9.0066s vs 14.8162s, max loss delta 0.005774 | not promoted |

## What Did Not Work

| Area | Result | Status |
|---|---|---|
| clean-region as a speed lever | Clean-region is semantically required, but the earlier huge speed win was actually stateful GDN cache | keep ON for semantics, not speed |
| defer grad sync / reshard | Fresh k=8 EP8x2 and current EP32 same-workload gates were slower than no-defer | OFF |
| `XORL_GDN_STATEFUL_INPLACE_TABLE_UPDATES=1` | Same-lane retry made the signal neutral/noise | OFF |
| `GDN_CAP=32768` | Slower or no useful gain | default cap |
| compact replay plan | Small/noisy; did not survive as a promotion candidate | not promoted |
| align/capture boundary variants | k=4 positives did not become a clean k=8 live-safe result; loss drift appeared | not promoted |
| 64-prompt trainer coalesce | Lost prepare parallelism in full-pipeline context | not promoted |
| no recompute | OOM in stateful GDN suffix path | rejected |
| DeepEP at small-k geometry | Timeout in dispatch CPU path at k=2 | alltoall remains promoted |

## MFU Ceiling Evidence

10% MFU was not reached. At the OPD natural 64-prompt batch, the evidence points
to too little useful work per GPU for a 32-GPU trainer. The best current
full-stack measurement is about 2.1-2.25% actual MFU and 0.21-0.23% useful MFU.
The best k=8 one-node replay pseudo actual MFU observed during static-shape
experiments was about 3.24%, but that was not promotable.

Historical large-batch 4-node EP8 evidence reached around 3.44% MFU, not 10%.

## Artifact Index

Key k=8 replay files:

- Default static4352 repeat:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_defaultalign_static4352_sync_nccl_repeat_20260615T1459Z.jsonl`
- Static2304:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_defaultalign_static2304_sync_nccl_20260615T1450Z.jsonl`
- Static1920:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_defaultalign_static1920_sync_nccl_20260615T1603Z.jsonl`
- Static1920 one-sample pack:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_defaultalign_static1920_pack1920_sync_nccl_20260615T1632Z.jsonl`
- Conservative dynamic static:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_dynstatic_static4352cap_sync_nccl_20260615T1528Z.jsonl`
- Dynamic static packing:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/replay_results_k8files64_v2_ep8_1node_dynstaticpack_static4352cap_sync_nccl_20260615T1544Z.jsonl`
- EP8x2 defer gate:
  `/shared/opd-control/er-opd-q36-mtp-perf-replay/k8_files64_capture_step3100_20260615T1132Z/compare_k8_ep8x2_defer_vs_nodefer.json`

Live full-stack MFU profile:

- `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260615T115717Z-2s1t/artifacts/opd_profile.jsonl`

## Handoff Boundaries

- Treat v2 as the only promoted live configuration.
- Treat EP8 and static-shape results as evidence, not live configuration.
- Treat dynamic static flags as default-off experimental code.
- Treat full-stack comparisons across different sampler states as non-decisive
  unless the workload is explicitly controlled.
