# MTP Infra Handoff - Current Control Summary

Last updated: 2026-06-15 16:54Z

This is the client-side infra handoff for `er-opd-q36-mtp-ss-0605c`. It replaces
the prior chronological deployment log. The old form is archived at
`archive/mtp_infra_agent_handoff_20260615T1654Z_chronological.md`.

Infra source of truth:
`/home/apanda/xorl-infra-opd-wordle-pr1-20260614/k8s/opd_profile/INFRA_RUNBOOK.md`

Promoted config source of truth:
`/shared/opd-control/er-opd-q36-mtp-ss-0605c/MTP_CONFIG_OF_RECORD.md`

## Current Stack State

The live stack name is `er-opd-q36-mtp-ss-0605c`.

As of the latest check, Kubernetes still had these live stack pods up: dispatch,
`sglang-0`, `sglang-1`, `teacher-sglang-0`, `teacher-smg`, trainer head, and
three trainer workers. No trainer-only perf-replay pod or service remained for
`stack=er-opd-q36-mtp-perf-replay,role-class=trainer`.

The newest run directory,
`q36mtp-20260615T161813Z-2s1t`, failed engine initialization because only 2 of 4
distributed clients joined rendezvous. It has no MFU profile rows. The latest
usable full-stack MFU result is the prior run
`q36mtp-20260615T115717Z-2s1t`, step 3504: actual MFU `2.246%`, useful MFU
`0.228%`.

## Canonical Control Inputs

| Item | Value |
|---|---|
| Engine repo for control writes | `/home/apanda/xorl-mtp` |
| Required engine env | `OPD_XORL_REPO=/home/apanda/xorl-mtp` |
| SGLang repo | `/home/apanda/xorl-sglang-internal` |
| Infra repo | `/home/apanda/xorl-infra-opd-wordle-pr1-20260614` |
| Infra working dir | `/home/apanda/xorl-infra-opd-wordle-pr1-20260614/k8s/opd_profile` |
| Args file | `launch_args_er-opd-q36-mtp-ss-0605c.txt` |
| Control root | `/shared/opd-control/er-opd-q36-mtp-ss-0605c` |
| Output root | `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c` |

The current args file encodes the promoted v2 trainer shape:

- `--trainer-nodes 4`
- `--trainer-expert-parallel-size 32`
- `--trainer-ep-dispatch alltoall`
- `--trainer-moe-implementation triton`
- `--trainer-gradient-checkpointing-method recompute_before_dispatch`
- `--trainer-clean-replay-context`
- `--trainer-skip-forward-backward-defrag`
- `--gdn-replay-plan-use-stateful-prefix-cache`
- `--prompts-per-step 64`
- `--pipeline-chunk-size 32`
- `--pipeline-prefetch-chunks 2`
- `--student-mtp-canonical-q-banding`
- `--k-toks 8`

## What Infra Must Not Accidentally Promote

The throughput investigation produced several measured candidates that should not
appear in a live control render unless the config-of-record is explicitly
updated:

| Candidate | Infra status |
|---|---|
| EP8 trainer topology | validated replay win, not live-promoted |
| defer grad sync / reshard | rejected for current k=8 EP8x2 and current EP32 |
| shorter static shapes such as 2304/2048/1920 | faster replay, not live-promoted due geometry/loss deltas |
| dynamic static padding / packing | code exists, default-off, not live-promoted |
| capture-align or optimized boundary flags | not live-promoted |
| `XORL_GDN_STATEFUL_INPLACE_TABLE_UPDATES=1` | neutral/noise, not live-promoted |
| `GDN_CAP=32768` | slower/no benefit, not live-promoted |

## Safety Model

Control writes are destructive because they rewrite slot `run.sh` files and cause
pods to re-exec. Before any write, inspect the shared state and announce in
`$CTL/AGENT_NOTES.md` if the operation changes the live stack.

Use this shell shape for inspection from the infra checkout:

```bash
cd /home/apanda/xorl-infra-opd-wordle-pr1-20260614/k8s/opd_profile
PY=/home/apanda/xorl-mtp/.venv/bin/python
G=q36_singleshot_reprogrammable_slots.py
A="$(cat launch_args_er-opd-q36-mtp-ss-0605c.txt)"
CTL=/shared/opd-control/er-opd-q36-mtp-ss-0605c
RB=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c
```

Read-only state checks:

```bash
$PY "$G" status --stack er-opd-q36-mtp-ss-0605c
kubectl get pods -n apanda -o wide | rg 'er-opd-q36-mtp-ss-0605c|NAME'
tail -80 "$CTL/AGENT_NOTES.md"
```

If writing trainer or sampler controls, set the repo env explicitly:

```bash
export OPD_XORL_REPO=/home/apanda/xorl-mtp
export OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal
```

Do not rely on generator defaults for repo selection, start step, checkpoint, or
learning rates.

## Cluster Policy

Every pod that requests `nvidia.com/gpu` must carry `team: turbo` on the pod
template labels. Do not manually override Kyverno-injected `schedulerName` or
Volcano queue labels unless there is a specific scheduling reason.

Do not use privileged pods, do not hardcode `CUDA_VISIBLE_DEVICES`, and keep the
existing RDMA / `IPC_LOCK` settings for trainer and P2P receiver paths.

## Known Infra Failure Modes

| Failure | Symptom | Infra handling |
|---|---|---|
| Bare-pod controller death from `/shared` read blip | pod enters `Error` or `Failed`; log mentions `sha256sum run.sh` | delete and reapply that Pod doc; rebuilt pods have tolerant polling in the current generator |
| Destructive trainer-control defaults | cold start, wrong worktree, wrong LR, or missing checkpoint resume | always export `OPD_XORL_REPO`, `OPD_SGLANG_REPO`, start step, checkpoint path, and LR env before writes |
| DeepEP small-k timeout | `DeepEP error: timeout (dispatch CPU)` | current promoted args use `alltoall` |
| P2P weight-sync wedge | `ret=-1`, `status=-1`, repeated batch transfer failures | restart samplers plus trainer together |
| Rendezvous partial join | server log reports fewer than 4 trainer clients joined | do not count that run; inspect trainer worker logs and relaunch only from explicit control state |

## Throughput Work Outcome For Infra

The main infra-relevant conclusion is negative: do not bake experimental
throughput flags into the live generator or args file just because they exist in
the code. The live args remain the promoted v2 shape. The code has default-off
knobs for dynamic static work, but current live controls should not enable them.

The previous perf-replay resources were cleaned up. Replay artifacts remain under
`/shared/opd-control/er-opd-q36-mtp-perf-replay`.
