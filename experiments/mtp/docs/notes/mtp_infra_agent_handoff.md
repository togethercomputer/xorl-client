# Handoff: MTP infra / deployment pilot runbook

**Date:** 2026-06-13
**Agent direction:** infra
**Stack:** `er-opd-q36-mtp-ss-0605c`
**Repo:** `/home/apanda/xorl-mtp-singleshot-port-20260602`

---

## ⚡ SESSION ADDENDUM — 2026-06-13 (infra agent, pre-consolidation). READ FIRST.

Everything below the `---` is the original runbook (still valid). This section captures what changed and what was learned this session. Full chronological ledger: `mtp_infra_ops_log.md`.

### Current control-plane state (as of ~19:00Z)
- **k=2 science run WOUND DOWN** at user direction; stack handed to the **perf agent** for efficiency work. Perf agent is actively running throughput sweeps (2-sampler vs 4-sampler configs, resuming the k=2 step-500 ckpt).
- **Supervisor PAUSED** (`supervisor.pause` present) so it doesn't fight the perf agent. Do not un-pause until a training run is the intended steady state again.
- **Args file** `launch_args_er-opd-q36-mtp-ss-0605c.txt` now carries **`--k-toks 2`** and **`--trainer-ep-dispatch alltoall`** (see below). If you resume normal k=4 prod, revert both.
- **Science progress preserved on disk:** `q36mtp-20260613T110939Z-2s1t/.../q36mtp-coderforge-v1-step000500` (latest) and `...best-step000300` (val_loss 0.2254). Resume Experiment A from these via `OPD_START_STEP` + `OPD_LOAD_CHECKPOINT_PATH`.
- A session-scoped **bare-pod resurrector** Monitor was running (see below); it dies with that session. The permanent fix is now in the generator — the next-gen agent does NOT need to re-create the resurrector once pods are rebuilt from the patched generator.

### 🔴 #1 NEW FAILURE MODE: bare-pod controller death from a transient `/shared` NFS read failure
**Symptom:** a slot pod (dispatch or any trainer slot) goes `Error`/exit 1; pod log ends with `sha256sum: <slot>/run.sh: No such file or directory`. Hit dispatch @06:04 and trainer-worker-1 @17:21 and trainer-head @18:58 this session.
**Root cause:** the in-pod slot controller poll loop runs under `set -euo pipefail` and did `sha256sum run.sh` with no retry. A momentary NFS read failure (NOT a write race — `write_executable` is atomic tmp+replace, and `stop-trainer-control` defaults `remove_run=False`) makes `sha256sum` fail → `set -e` exits PID 1 → bare pod (`restartPolicy: Never`) stays dead forever.
**The supervisor CANNOT recover this** — it writes a control script, but with no live controller nothing reads it; worse, `samplers_ready()` reads stale "ready" lines and starts the trainer against a dead slot.
**Manual recovery (proven):** `kubectl delete pod <stack>-<slot>` then `kubectl apply` that slot's Pod doc from `render-manifest --output /tmp/x.yaml` (filter docs by `metadata.name`). A trainer worker rejoins the rendezvous if recreated within the torchrun timeout; a dead trainer-HEAD or a wedged run needs a full clean restart after.
**PERMANENT FIX — APPLIED THIS SESSION** in `q36_singleshot_reprogrammable_slots.py` (the slot-controller poll loop): the `sha256sum` is now wrapped in `set +e` and gated on a non-empty `new_hash`, so a transient read failure skips the poll instead of killing the controller. **Takes effect only on pods rebuilt from the patched generator** — currently-running pods still have the old loop, so the resurrector / manual recovery is still needed until a full re-deploy. Validated: py_compile + `bash -n` on the rendered controller.
**Resurrector (session-scoped stopgap):** a `Monitor` polling for `Error`/`Failed` allowlisted pods and auto-`delete`+`apply` from pre-rendered `$CTL/_resurrect/<slot>.pod.yaml` (per-pod 5min cooldown, cap 4). Safe because `Error` uniquely means controller death (intentional stops / inner-process crashes keep the pod `Running`). Gap: only the 2-sampler set + dispatch + trainer×4; does NOT cover `sglang-2/3` from 4-sampler runs.

### 🟠 #2 TRAP: `write-trainer-control` defaults are destructive
With NO env set, the generator renders `OPD_START_STEP=0` (cold start from BASE weights), `OPD_XORL_REPO=/home/apanda/xorl-mtp-singleshot-port-20260602` (the analysis checkout, NOT `commitlen-fix`), `OPD_FULL_FT_LR=1e-6`, `OPD_MUON_LR=1e-5`. The throughput agent hit this twice (cold-started on the wrong worktree). **ALWAYS export** `OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612 OPD_FULL_FT_LR=0.00001 OPD_MUON_LR=0.001 OPD_START_STEP=<latest-ckpt-step> OPD_LOAD_CHECKPOINT_PATH=<that-ckpt>` before `write-trainer-control` (the supervisor's own env already carries these — that's why supervisor recoveries are correct).

### 🟠 #3 DeepEP dispatch timeout at k=2 geometry → use alltoall
At k=2 the trainer's first fb reproducibly died with `DeepEP error: timeout (dispatch CPU)` (deepep was fine at k=4 for 50+ steps — it's the smaller k=2 replay geometry, static_padded 1280 vs 2304). Fix: `--trainer-ep-dispatch alltoall` (science-neutral — identical MoE numerics, robust NCCL all-to-all). Baked into the args file. If a future config re-hits a deepep dispatch timeout, switch to alltoall.

### Consolidation note
Per `CONSOLIDATION_HANDOFF.md`, this whole `experiments/opd_profile/k8s/` + `docs/notes/mtp_infra_*` harness is "harness, not engine" → destined for **`xorl-infra`** (k8s/configs) with outputs to `/shared`. The slot generator (with the controller fix + the WANDB_DIR patch) should move as one unit. The `opd_profile` harness is shared with opd-port — collapse to one copy.

---

## Mission

Own the deployment control plane so the throughput and science agents can run experiments without losing the stack. You may run anything, but control writes are destructive: they re-exec pods and can kill an in-flight run. Your job is to make the stack pilotable and recoverable, not to make the science or MFU call.

## Current stack state

Read-only inspection on 2026-06-13:

```text
live run: q36mtp-20260613T014150Z-2s1t
trainer started: 2026-06-13T01:41:50Z
run dir: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t
control root: /shared/opd-control/er-opd-q36-mtp-ss-0605c
args file: experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt
```

Current controller status:

```text
dispatch: running since 2026-06-12T23:58:03Z
sglang-0: running since 2026-06-12T23:58:03Z
sglang-1: running since 2026-06-12T23:58:03Z
teacher-sglang-0: running since 2026-06-05T20:48:43Z
teacher-smg: running since 2026-06-05T04:53:47Z
trainer-head: running since 2026-06-13T01:41:50Z
trainer-worker-1: running since 2026-06-13T01:41:51Z
trainer-worker-2: running since 2026-06-13T01:41:49Z
trainer-worker-3: running since 2026-06-13T01:41:50Z
```

Current Kubernetes placement:

```text
dispatch:         research-common-h100-053
sglang-0:         research-common-h100-041
sglang-1:         research-common-h100-114
teacher-sglang-0: research-common-h100-079
teacher-smg:      research-common-h100-125
trainer-head:     research-common-h100-071
trainer-worker-1: research-common-h100-117
trainer-worker-2: research-common-h100-094
trainer-worker-3: research-common-h100-068
```

The controller status files still mention stale `sglang-2`/`sglang-3` slots from earlier experiments. The actual Kubernetes stack currently has only `sglang-0` and `sglang-1`.

## Current rendered run semantics

This block is provenance for the currently running trainer, not the template for the next science diagnostic. The next
diagnostic must use the resume template below and should not copy this older step-100 checkpoint path.

The live trainer `run.sh` has:

```text
XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612
SGLANG_REPO=/home/apanda/xorl-sglang-internal
OPD_FULL_FT_LR=0.00001
OPD_MUON_LR=0.001
OPD_START_STEP=100
OPD_LOAD_CHECKPOINT_PATH=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260612T204932Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000100
```

The generated trainer config confirms:

```yaml
lr: 1.0e-05
muon_lr: 0.001
optimizer: muon
load_checkpoint_path: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260612T204932Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000100
```

Do not `uv sync` a fresh venv in `/home/apanda/xorl-mtp-commitlen-fix-20260612`. That worktree should use the default proven venv symlink so DeepEP and package versions match the live stack.

## The control-plane model

Bare pods run slot controllers. Each slot polls:

```text
/shared/opd-control/er-opd-q36-mtp-ss-0605c/<slot>/desired.sha256
```

When a `write-*-control` command writes a new `run.sh` and hash, the in-pod controller re-execs that role. No image rebuild is needed. Uncommitted code in the selected `XORL_REPO` is live on next re-exec.

Generator:

```text
experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
```

Subcommands:

```text
render-manifest
render-control
write-control
write-trainer-control
write-student-inference-control
write-dispatch-control
write-teacher-control
render-native-route-probe
native-route-probe
stop-control
stop-trainer-control
status
```

Set common shell:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python
G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
A="$(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)"
CTL=/shared/opd-control/er-opd-q36-mtp-ss-0605c
RB=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c
```

Status:

```bash
$PY "$G" status --stack er-opd-q36-mtp-ss-0605c
kubectl get pods -o wide | rg 'er-opd-q36-mtp-ss-0605c|NAME'
```

## Safety protocol before any control write

1. Read the coordination file:

```bash
tail -80 "$CTL/AGENT_NOTES.md"
```

2. Check whether the trainer is active:

```bash
$PY "$G" status --stack er-opd-q36-mtp-ss-0605c
LOG=$(ls -t "$CTL"/trainer-head/logs/*-run.log | head -1)
tail -60 "$LOG"
```

3. Check for a finished or crashed run:

```bash
rg -n "OPD pipeline validation succeeded|trainer-head cleanup rc=|Traceback|CUDA out of memory|DistBackendError|ret=-1|status=-1|Forward-backward timeout" "$LOG" | tail -40
```

4. Announce the operation in `AGENT_NOTES.md` before writing controls.

5. If doing manual changes, pause the supervisor first:

```bash
touch "$CTL/supervisor.pause"
```

Remove the pause only when you deliberately want the supervisor to resume recovery:

```bash
rm -f "$CTL/supervisor.pause"
```

## Canonical recovery / relaunch recipe

Most failures are recovered by restarting trainer and samplers together while leaving teachers warm:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python
G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
A="$(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)"

$PY "$G" stop-trainer-control $A
$PY "$G" write-student-inference-control $A
# Wait for dispatch log to report all expected student backends ready.
$PY "$G" write-trainer-control $A
```

Why samplers are restarted with the trainer:

- A trainer-only restart can leave samplers in paused generation or a stale `weight_sync_group`.
- Stale Mooncake state can produce p2p `ret=-1` / `status=-1`.
- Recreating dispatch plus samplers clears that state.

Teachers usually stay warm because they are HTTP prefill workers with no weight-sync receiver state.

## Supervisor

Supervisor script:

```text
experiments/opd_profile/k8s/opd_trainer_supervisor.sh
```

Current supervisor files:

```text
/shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.pid
/shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.log
/shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.pause
```

Run it with `bash`; it may not have the executable bit:

```bash
nohup bash experiments/opd_profile/k8s/opd_trainer_supervisor.sh \
  >> /shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.log 2>&1 < /dev/null &
```

The supervisor:

- polls every 120s,
- treats 1500s of head-log staleness as a stall,
- greps head, worker logs, and newest `server.log` for crash signatures,
- recovers by `stop-trainer-control -> write-student-inference-control -> write-trainer-control`,
- resumes from the newest run directory that actually has an on-disk DCP `.metadata`,
- recreates samplers before trainer.

Important: if you launch an LR/worktree experiment through environment overrides, make sure the rendered `run.sh` or the supervisor environment carries the same `OPD_XORL_REPO`, `OPD_FULL_FT_LR`, and `OPD_MUON_LR`. Otherwise recovery can silently relaunch the wrong experiment.

## Science live diagnostic relaunch template

Use only when the science agent explicitly requests the next live low-k/curriculum diagnostic. Apanda resolved that
science should **not** re-run the Qwen3-4B paper-style harness first, and that the live diagnostic can resume from the
current checkpoint.

Here, "current checkpoint" means the newest validated DCP checkpoint from the current live run, not the older
`q36mtp-20260612T204932Z-2s1t/.../q36mtp-coderforge-v1-step000100` checkpoint. At the time this handoff was updated,
no DCP `.metadata` had been found yet under the current live run's `server_output/weights/default`; if that is still
true, keep the current run alive until a current-run checkpoint exists, or coordinate an intentional checkpoint save.

Find the checkpoint before stopping anything:

```bash
CURRENT_RUN=$RB/q36mtp-20260613T014150Z-2s1t
find "$CURRENT_RUN/server_output/weights/default" -maxdepth 2 -type f -name .metadata -print | sort
```

Select the newest current-run checkpoint and derive the resume step:

```bash
CKPT_META=$(find "$CURRENT_RUN/server_output/weights/default" -maxdepth 2 -type f -name .metadata -print | sort | tail -1)
test -n "$CKPT_META"
CKPT=$(dirname "$CKPT_META")
STEP=$(basename "$CKPT" | sed -E 's/.*step0*([0-9]+)$/\1/')
test -n "$STEP"
printf 'resume step=%s checkpoint=%s\n' "$STEP" "$CKPT"
```

Then relaunch with the science-requested low-k/curriculum arguments:

```bash
touch "$CTL/supervisor.pause"
$PY "$G" stop-trainer-control $A

export OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612
export OPD_FULL_FT_LR=0.00001
export OPD_MUON_LR=<science-requested-lr>
export OPD_START_STEP=$STEP
export OPD_LOAD_CHECKPOINT_PATH=$CKPT

# Apply the science-requested k=2 or random-[2,4] controls through the generator/args file path that
# actually renders into run.sh and generated_trainer_config.yaml.
nohup bash experiments/opd_profile/k8s/sync_launch_0605c.sh > /tmp/relaunch_mtp_live_diag.out 2>&1 < /dev/null &
```

Gotchas:

- `pkill -f sync_launch` and `pkill -f opd_trainer_supervisor` can match your shell. Kill by PID.
- The sync-launch "STEP 0 fb PASSED" can be a stale-log false match on resume runs. Verify generated config and
  `=== OPD step $OPD_START_STEP`.
- Keep `OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612` for commit-len science A/Bs.
- Do not restart from the older step-100 checkpoint unless apanda explicitly authorizes that comparison.
- Do not assume an env var changes `k`; inspect the rendered `run.sh`, args file, and generated trainer config to prove
  the requested low-k/curriculum control is active.

## Throughput experiment support

The throughput agent may ask you to:

- add `--student-mtp-canonical-q-banding` to a sampler-control write,
- run a short non-destructive `--skip-optim-step` trainer profile,
- rewrite only student inference control,
- render a 4-sampler manifest.

Sampler control writes are disruptive. Use the safety protocol first.

If testing q-banding:

```bash
touch "$CTL/supervisor.pause"
$PY "$G" write-student-inference-control $A --student-mtp-canonical-q-banding
```

This command shape may need the full args rewritten into the args file for supervisor consistency. Do not leave the live args file and control scripts disagreeing if the experiment is promoted.

For profiler runs, prefer short and non-destructive:

```bash
XORL_PROFILE_SERVER_FB=1 XORL_PROFILE_SERVER_FB_SKIP=0 XORL_PROFILE_SERVER_FB_COUNT=1 \
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
$PY "$G" write-trainer-control $A \
  --num-steps 1 --max-opd-steps 1 --prompts-per-step 8 --pipeline-chunk-size 8 \
  --pipeline-prefetch-chunks 1 --pipeline-teacher-concurrency 1 \
  --skip-optim-step --checkpoint-interval-steps 0 --no-checkpoint-save-best
```

## Four-sampler scale-up

Do not scale samplers just because `commit_len` is flat. This is gated on the science agent confirming learnability, or the throughput agent proving a sampler bottleneck after q-banding.

If scaling is approved:

1. Render a 4-replica manifest to a temp file:

```bash
$PY "$G" render-manifest $A --student-replicas 4 --output /tmp/er-opd-q36-mtp-ss-0605c-4s.yaml
```

2. Extract only new `sglang-2` and `sglang-3` Service/Pod docs. Do not blindly apply the full manifest over running pods.

3. Apply the new docs:

```bash
kubectl apply -f /tmp/sglang-2-3-only.yaml
```

4. Wait for both new samplers to load the model and dispatch to report readiness.

5. Rewrite student and trainer controls with `--student-replicas 4`, then update the args file if promoted.

Weight sync to more than two p2p receivers is config-supported but should be treated as a validation point.

## Non-negotiable cluster rules

For all GPU pod templates:

- include label `team: turbo`,
- do not set `privileged: true`,
- do not hardcode `CUDA_VISIBLE_DEVICES`,
- request `rdma/infiniband: 1` plus `IPC_LOCK` for trainer and p2p receiver samplers,
- do not set `NCCL_IB_GID_INDEX` or `NCCL_IB_HCA` on pods that initialize Mooncake,
- let Kyverno inject `schedulerName: volcano` and the queue label.

If a pod gets `NCCL error: unhandled cuda error` at init, delete and recreate it once before adding node excludes. Avoid known-bad `research-common-h100-105` if the failure repeats.

## Failure signatures

Search across head, workers, and server log:

```bash
RUN=$(ls -td "$RB"/q36mtp-* | head -1)
rg -n "cleanup rc=[1-9]|DistBackendError|DistStoreError|CUDA out of memory|Failed to CUDA calloc|NCCL watchdog|Forward-backward timeout|Engine Core initialization timeout|ret=-1|status=-1|ReadTimeout" \
  "$CTL"/trainer-head/logs/*-run.log \
  "$CTL"/trainer-worker-*/logs/*-run.log \
  "$RUN/server.log" | tail -100
```

Common recoveries:

- Greedy rollout `ReadTimeout`: recreate dispatch+samplers+trainer.
- p2p `ret=-1` / `status=-1`: recreate samplers, confirm `expandable_segments` is off, inspect `server.log`.
- Trainer `cleanup rc!=0`: run canonical recovery.
- Pod `Error`: delete pod and re-apply the matching manifest doc.
- Pending GPU pod during contention: delete/recreate may place it elsewhere.

## Deliverable

Maintain a short live ops note with:

- current active run id and step,
- whether supervisor is paused or armed,
- exact control writes performed,
- args file changes,
- pod/node changes,
- recovery attempts and their result,
- any mismatch between args file, `last_singleshot_config.txt`, and generated `run.sh`.

## Open coordination item

- Keep the current science run alive until a current-run checkpoint exists or science/infra intentionally stops it for
  the live low-k/curriculum diagnostic.
- If throughput wants q-banding restored, do not interrupt the current run until the current-run checkpoint and science
  diagnostic handoff are secured, unless apanda explicitly clears the sampler-control rewrite sooner.
