# ZORL — Beat 0.93: INFRA

How the serving pools, client jobs, and code are wired, plus the operational
lessons from this campaign.

## 1. Serving pools (sglang)

Each "pool" is an 8-replica StatefulSet `zorl-ar-sglang-<id>` in namespace
`apanda`, headless service `zorl-ar-sglang-<id>-headless`, port 30000. Each
replica is TP=2 (16 GPUs/pool) serving `Qwen/Qwen3.6-35B-A3B` with LoRA enabled:
`--lora-backend triton --max-lora-rank 32 --max-loras-per-batch 16
--lora-moe-format hybrid_shared --experts-shared-outer-loras
--lora-use-virtual-experts --disable-cuda-graph --mem-fraction-static 0.85`.

**Patched checkout (critical):** pods run from `workingDir
/workspace/home/xorl-sglang-zorl`, `SGLANG_REPO=/workspace/home/xorl-sglang-zorl`,
`PYTHONPATH=${SGLANG_REPO}/python`, but `SGLANG_PYTHON=/workspace/home/
xorl-sglang-internal/.venv/bin/python` (internal venv as **dependency provider
only**). Do NOT re-apply the infra-repo manifest — its `last-applied-config`
still points at `xorl-sglang-internal` and would revert the serving source.
Confirmed patched code is live because step 1 emits `unclipped_update_norm` /
`update_clip_scale` (only in the patched `lora_manager.py`).

**Live pools this session:**
- `zorl-ar-sglang-l` — original (prior agent's HYPER-L run). HYPER-L completed at
  its 12h cap; **pool L pods are now idle, holding 16 GPUs**. Reset is
  guardrail-blocked (not agent-created) — see §5.
- `zorl-ar-sglang-m` — clone, running **R5** (delayed decay).
- `zorl-ar-sglang-n` — clone, running **R4** (near-constant control).

## 2. Cloning a pool (additive, the way to add capacity)

Done without touching shared infra:
```
kubectl -n apanda get sts zorl-ar-sglang-l -o json > /tmp/poolL_sts.json
kubectl -n apanda get svc zorl-ar-sglang-l-headless -o json > /tmp/poolL_svc.json
# python: drop status + metadata.{uid,resourceVersion,generation,creationTimestamp,
#   managedFields, annotations.last-applied}; string-replace zorl-ar-sglang-l ->
#   zorl-ar-sglang-<new>; replicas=8; drop svc clusterIP/ipFamilies/etc.
kubectl -n apanda apply -f /tmp/pool<new>_svc.json
kubectl -n apanda apply -f /tmp/pool<new>_sts.json
```
The clone inherits the patched checkout. Verify 8x `/health=200` before launching
a client (model load ~250–340s). Helper: `/home/apanda/poolm_health_wait.sh`.

## 3. Launching a client (one run per pool)

Controller renders the base manifest + `--set-env` overrides into a Job:
```
python experiments/zorl/autoresearch/controller.py launch \
  --candidate .../candidates/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE.yaml \
  --set-env INFER_URL='<8 pool URLs>' --set-env RESULT_ROOT=... --set-env WANDB_NAME=... \
  --set-env LR_SCHEDULE=cosine --set-env LR_HOLD_STEPS=200 --set-env LR_DECAY_STEPS=200 \
  --set-env LR_MIN_FRAC=0.15 --set-env MAX_RUNTIME_SECONDS=86400
```
Staged launch scripts: `/home/apanda/launch_{R1c,R4,R5,...}_*.sh`.

**One ZORL client per pool.** `fresh_ab` folds updates into the served base
weights *in place*, so two concurrent sessions on one pool corrupt each other.
`MAX_UPDATE_NORM=0` is converted client-side to `None` (no clip) and the patched
server skips clipping — the proven recipe runs unchanged on patched servers.

## 4. Failure mode learned: node loss → session loss → full pool reset

A node (`research-common-h100-058`) went NotReady; TaintManager evicted its 3
pool-M pods; they rescheduled on recovery. **But a restarted serving pod loses its
in-memory ZORL session** → the client's `/apply_zorl_rewards` to that pod fails
with `Unknown ZORL session`, and the pool's base is now **inconsistent** (5 pods
mutated, 3 pristine). The run cannot hot-recover. Required action:

```
kubectl -n apanda delete job <client-job>                 # mine this session
kubectl -n apanda delete pod -l app=zorl-ar-sglang-<id>   # reset ALL pods -> pristine
# wait 8x /health=200, then relaunch the client (fresh session on all 8)
```
Cost: lose the run's progress (restart from cold). Mitigation idea (not done):
checkpoint/restore session state, or run pools at higher priority to avoid
preemption.

## 5. Guardrails / permissions

The auto-mode classifier **blocks deleting pods/jobs the agent did not create
this session** (shared pool-L serving pods, HYPER-L's job) — correctly, as
destructive/interfering actions. The agent's *own* pools (M/N) and jobs are
freely resettable. Consequence: **pool L's 16 idle GPUs cannot be reclaimed by
the agent** — the user should delete the `zorl-ar-sglang-l` pods (or add a Bash
permission rule). On an oversubscribed cluster (≈100 GPUs of pending demand
observed) this idle capacity matters. R4-style additive pools were used instead
of touching pool L.

## 6. Code changes this campaign (additive, default-off)

Delayed-decay LR schedule — runs execute the **consolidated** client
`/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/
zorl_client.py` (identical to the `xorl-client` copy; base manifest runs it
in-place, no overwrite at launch):
- `zorl_client.py`: added `--lr-hold-steps` arg + `_lr_for_step` honors it
  (`if step < hold: return lr`). Default 0 = legacy cosine-from-0.
- `k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml`: added
  `LR_HOLD_STEPS` env default + `--lr-hold-steps "${LR_HOLD_STEPS}"`.
Verified: client accepts the arg (reaches `[init]`), curve correct, py_compile OK.

## 7. Monitoring harness (this session)

Background waiters under `/home/apanda/`:
- `milestone_wait.sh <rundir_file> <job> <pod> <label> [deadline]` — wakes only on
  a WIN (>0.9297), a fatal error (traceback/CUDA/OOM/`[done]`), pod-not-Running,
  or a periodic checkpoint. Transient "connection refused" is NOT treated as fatal
  (a sustained outage surfaces as no step progress at the checkpoint).
- `poolm_health_wait.sh` / `pooln_health_wait.sh` — wait for 8x `/health=200`.
- `hyperl_liveness.sh` — death-only watch.
Run dirs tracked in `/home/apanda/R{1,3}_rundir.txt`.
