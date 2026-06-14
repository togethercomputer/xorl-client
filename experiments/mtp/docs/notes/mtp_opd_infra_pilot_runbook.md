# MTP OPD — infra / kickoff pilot runbook

**Stack:** `er-opd-q36-mtp-ss-0605c` — Qwen3.6-35B-A3B SingleShot-MTP on-policy distillation (OPD), server-training mode, coderforge dataset.
**Audience:** the agent piloting the infra / launch / recovery of this stack. **Scope: operations only.** Performance/MFU/profiling is a *separate* agent's job — see `docs/notes/mtp_opd_mfu_profiling_handoff.md`; do NOT do perf work from here, and do NOT enable the profiler on normal runs.
**Last validated:** 2026-06-08 — full OPD loop runs end-to-end on 4 nodes via p2p sync (step → optim → sync 200 → step), losses healthy.

---

## 1. What the stack is

| Role | Pods | GPUs | Parallelism | Notes |
|---|---|---|---|---|
| Trainer | trainer-head + worker-1/2/3 | 32 (4×8) | FSDP=32, EP=8 (deepep), TP/PP/CP=1, Quack MoE, DeepEP SMS24 | runs the OPD HTTP server; the thing being trained |
| Student samplers | sglang-0, sglang-1 | 4 (2×2) | TP=2 each | SGLang, SingleShot-MTP ConfAdapt decode; generate rollouts; p2p weight-sync receivers |
| Teachers | teacher-sglang-0/1 | 4 (2×2) | TP=2 each | SGLang prefill; distillation targets (HTTP, no weight sync) |
| dispatch | 1 | 0 (CPU) | — | routes student gen across samplers |
| teacher-smg | 1 | 0 (CPU) | — | routes teacher requests |

~40 H100s total. Model: Qwen3.6-35B-A3B MoE (~3B active), GatedDeltaNet linear-attn + full-attn hybrid.

## 2. Where everything lives

- **Repo / worktree:** `/home/apanda/xorl-mtp-singleshot-port-20260602` (branch `codex/mtp-singleshot-port-20260602`). **The k8s pods run THIS worktree** via `XORL_REPO=<repo>` + `PYTHONPATH=$XORL_REPO/src` — so uncommitted edits deploy on the next controller re-exec (no commit/build needed). Several fix files are uncommitted working-tree edits; that's intended.
- **venv / python:** `<repo>/.venv/bin/python`.
- **Generator (the control tool):** `<repo>/experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`.
- **Launch args:** persisted in the repo at `experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt` (the working copy was `/tmp/mtp_relaunch_args_0607b.txt`, which is ephemeral). Use the repo copy: `A="$(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)"`. Full config also embedded in §4.
- **Control dir:** `/shared/opd-control/er-opd-q36-mtp-ss-0605c/<role>/` — each role has `run.sh`, `desired.sha256`, `status`, `logs/<rev>Z-run.log`.
- **Run dir / server logs:** `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-<rev>Z-2s2t/` → `server.log` (trainer SERVER ranks), `server_output/` (DCP checkpoints under `weights/default/<name>/`), `artifacts/` (`checkpoint_summary.json`, teacher caches).
- **YAML (untracked):** `examples/server/opd_singleshot_mtp_qwen36/qwen3_6_35b_a3b_student_mtp_4node_ep8.yaml` (muon lr 1e-5, all_ranks, `load_checkpoint_path: ""`).

## 3. Control-plane model (reprogrammable slots)

Bare pods run a controller daemon that polls its `desired.sha256` and re-execs `run.sh` when the hash changes. `write_control` stamps a fresh `revision` timestamp into each `run.sh`, so **every `write-*-control` forces a re-exec** (even with identical config). `run_opd_pipeline.py` (the OPD client/orchestrator) runs on the trainer-head pod and is invoked **with no CLI args** — all config is env, baked into `run.sh` by the generator.

**Generator subcommands** (`<venv> <generator> <cmd> $ARGS`):
- `write-trainer-control` — (re)launch the 4 trainer pods (head + workers).
- `write-student-inference-control` — (re)launch dispatch + sglang-0..N (the samplers).
- `write-dispatch-control` — (re)launch just dispatch.
- `write-teacher-control` — (re)launch teacher-smg + teacher-sglang-0..N.
- `stop-trainer-control` / `stop-control` — write stop sentinels so controllers don't respawn.
- `render-manifest --output <f>` — emit the full k8s manifest (pods+services); used to create pods.
- `render-control --role <r> --output <f>` — dry-render a role's run.sh (no deploy).
- `status` — print role statuses.

`$ARGS` = the contents of the args file: `A="$(cat /tmp/mtp_relaunch_args_0607b.txt)"` then `… $A`.

## 4. The known-good launch config (embed / persist this)

```
--stack er-opd-q36-mtp-ss-0605c
--student-replicas 2 --student-gpus 2 --teacher-replicas 2 --teacher-gpus 2 --trainer-nodes 4
--student-node-group default --teacher-node-group default --trainer-node-group default --router-node-group default
--student-smg-policy round_robin --teacher-smg-policy round_robin --native-route-mode direct_fanout
--trainer-ep-dispatch deepep --trainer-moe-implementation quack --trainer-deepep-num-sms 24
--trainer-gradient-checkpointing-method recompute_full_layer --trainer-fsdp-reduce-dtype fp32 --no-trainer-enable-packing
--sync-inference-method p2p                                                  # Mooncake RDMA (validated); nccl_broadcast is the fallback
--prompt-dataset-path /shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream
--prompt-dataset-type parquet --prompt-dataset-split train --prompt-dataset-column prompt_ids --prompt-dataset-turn-strategy prefix
--prompt-dataset-min-target-tokens 1 --prompt-dataset-assistant-offset-tokens 0 --prompt-dataset-assistant-role-token-ids-json [77091]
--prompt-dataset-offset 0 --prompt-dataset-epochs 3 --max-opd-steps 10000    # 3 epochs = 12004 steps; 10000 caps at ~2.5 epochs
--prompt-len 512 --max-new-tokens 256 --prompts-per-step 32
--no-skip-optim-step --forward-backward-timeout 2400.0
--checkpoint-interval-steps 10 --checkpoint-keep-latest 1 --checkpoint-save-best --checkpoint-best-metric val_loss
--checkpoint-best-mode min --checkpoint-best-min-step 100 --checkpoint-timeout 2400.0 --checkpoint-name-prefix q36mtp-coderforge-v1
--checkpoint-eval-dataset-split eval --checkpoint-eval-dataset-offset 0 --checkpoint-eval-steps 1 --checkpoint-eval-batch-size 32
--pipeline-chunk-size 4 --pipeline-prefetch-chunks 4 --pipeline-teacher-concurrency 2
--conf-threshold 0.3 --opd-async-sample-overlap
```

Notes: `--opd-async-sample-overlap` = sample step N+1 during step N fb (1-step-stale rollouts, ≈on-policy at lr 1e-5); validated, keep on. `--checkpoint-interval-steps 10` is frequent (was 100; lowered for resume work) — raise if checkpoint eval overhead matters. The trainer config sets `load_weights_mode: all_ranks` + `load_checkpoint_path: ""` (DCP load disabled — workaround; see §8).

## 5. Kickoff / launch (the canonical recovery recipe)

This is also the recovery procedure for almost every failure — recreate the inference stack and trainer **together** (teachers stay warm). Run on the dev pod:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python ; G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
A="$(cat /tmp/mtp_relaunch_args_0607b.txt)"

$PY $G stop-trainer-control $A                 # 1. stop trainer (no respawn while we reset samplers)
$PY $G write-student-inference-control $A       # 2. recreate dispatch + both samplers (clears paused-gen / stuck weight_sync_group / stale Mooncake)
#    3. WAIT until BOTH samplers log "server is fired up and ready to roll!" AND dispatch logs
#       "student backend ...sglang-{0,1}... ready after N polls" (x2). Samplers reload the 35B model (~2-8 min).
$PY $G write-trainer-control $A                 # 4. start the trainer (picks up current worktree code + baked env)
```

Cold start from scratch (no pods yet): `render-manifest --output /tmp/m.yaml $A` then `kubectl apply -f /tmp/m.yaml` (creates all pods+services), then the same `write-*-control` sequence. (Teachers: `write-teacher-control $A`.)

**Always recreate samplers before relaunching the trainer.** A trainer-only restart leaves samplers in paused-generation with a stuck `weight_sync_group` and the dispatch circuit-breaker open → the new trainer's first greedy rollout 900s-ReadTimeouts or step-0 crashes. Recreating them together avoids this. Teachers do NOT need recreation (HTTP prefill, no weight sync).

## 6. Critical env / settings that MUST hold (don't re-break these)

The generator bakes these; if you edit the generator or YAML, preserve them:
- **`expandable_segments` OFF on the trainer.** `OPD_TRAINER_EXPANDABLE_SEGMENTS` defaults 0. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **breaks Mooncake P2P registration >~20 MiB → silent weight-sync hang at `model.layers.0` → 1800s timeout.** This was the hardest bug; do not turn it on. (It was originally added to dodge an NCCL `alltoall_pre_dispatch` calloc; that calloc is fine at the current conf0.3/mnt256 memory headroom.)
- **P2P / Mooncake env** (trainer): `XORL_P2P_CPU_POOL_MIN_BYTES=0` (Bug-7: keeps tiny tensors off the buggy GPU-direct path), `XORL_WEIGHT_SYNC_BATCH_DENSE=1` + `_MOE=1`, `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu`, `XORL_P2P_HANDSHAKE_BASE_PORT=16400` (→ `MC_HANDSHAKE_PORT=base+gpu_id`). Cold sync sends `p2p_invalidate_cache=True` to re-arm the receiver — load-bearing, don't disable.
- **NO `NCCL_IB_GID_INDEX` / `NCCL_IB_HCA` on Mooncake-init pods (the trainer).** They force a user-specified-GID path that fails when HCAs have no IPoIB netdev; Mooncake auto-discovers cleanly. (SGLang samplers don't set them either.)
- **RDMA pods (trainer, p2p receivers):** `rdma/infiniband: 1` + `securityContext.capabilities.add: [IPC_LOCK]`, **NOT `privileged: true`** (privileged leaks all 8 host GPUs → contention → NCCL init errors), and **never hardcode `CUDA_VISIBLE_DEVICES`** (the device-plugin sets it).
- **`team: turbo`** label on every GPU pod template (Volcano queue). Don't set `schedulerName`/`queue-name` manually (Kyverno injects them).
- **Distributed-init timeouts:** `XORL_PROCESS_GROUP_TIMEOUT_MINUTES=60`, `XORL_ENGINE_READY_TIMEOUT_S=5400` (90 min) — the head's engine starts ~6 min after workers; without these the workers `DistBackendError wait timeout`.
- **`moe_implementation: quack`** with **`deepep_num_sms: 24`** for the DeepEP trainer path.

## 7. Monitoring (watch HEAD + ALL WORKERS + server.log)

- **`trainer-head/logs/<rev>-run.log` = the OPD CLIENT** (`run_opd_pipeline.py`): step markers (`=== OPD step N`), `Async forward_backward step=N …`, `Calling sync_inference_weights`, `sync_inference_weights returned status=… success=…`, `optim_step grad_norm=`.
- **`{RUN_DIR}/server.log` = the trainer SERVER ranks** (model_runner): per-module `WeightSync timing`, the actual fb internals, DCP/NCCL init. The head run.log does NOT contain these — to debug a sync/fb stall you MUST read `server.log` (find RUN_DIR: `ls -t .../q36mtp-*/server.log | head -1`).
- **`trainer-worker-{1,2,3}/logs/<rev>-run.log`** = the other trainer ranks (8–31). Watch these too.
- **Lesson:** a head-only / OOM-only watcher MISSES engine-init / NCCL / DistStore / `cleanup rc=` failures — a 14-hour-dead run went unnoticed once. Tail head + all 3 workers + server.log.
- **Healthy heartbeat:** `=== OPD step N` → fb chunks → `optim_step grad_norm=…` → `sync_inference_weights returned status=200 success=True … in ~2s` (warm; ~11s cold first sync) → `=== OPD step N+1`. p2p sync moves ~69 GB / 107 buckets.
- **Failure signatures to grep:** `cleanup rc=[1-9]`, `DistBackendError`, `DistStoreError`, `CUDA out of memory`, `Failed to CUDA calloc`, `NCCL error: unhandled cuda` (bad node / contention), `NCCL watchdog`, `ret=-1`/`status=-1` (sync wedge), `Forward-backward timeout`, `Engine Core initialization timeout`, `ReadTimeout` (900s on greedy rollout = wedged sampler).

## 8. Recovery playbook (common failures → fix)

- **Sampler wedged** (greedy-rollout 900s ReadTimeout, or sync can't reach a sampler): run the §5 recipe (recreate samplers+dispatch+trainer). Most common failure; almost always the answer.
- **Bad-node NCCL error** (`NCCL error: unhandled cuda error` at a sampler's `Init torch distributed`): the pod landed on a bad/contended node. `kubectl delete pod <name>` + `kubectl apply -f <its manifest doc>` to reschedule. **h100-105 is known NCCL-bad** (also historically h100-005/036/061, but many such excludes are stale SM-sweep-race victims — retry once before excluding). If it re-lands on a bad node and re-fails, add a `nodeAffinity` excluding that hostname (label = the full node name, e.g. `research-common-h100-105.cloud.together.ai`).
- **p2p sync wedge** (`status=-1`, 1800s, `bytes=0`): root causes are FIXED (expandable_segments off + cold-prepare `p2p_invalidate_cache` re-arm, both in the worktree). If it recurs: confirm `expandable_segments` is off in the rendered `run.sh`, recreate samplers (clears stale Mooncake registration), and read `server.log` to see which `model.layers.N` it stalls at. Full history: `docs/notes/mtp_p2p_weight_sync_wedge_handoff.md`.
- **Trainer crash / `cleanup rc=`:** bare pods do NOT reschedule the process on crash — re-run the §5 recipe. If a pod itself goes `Error` (controller died), `kubectl delete pod <name>` + re-apply its manifest doc.
- **Pod stuck `Pending`/contention:** another tenant grabbed GPUs the device-plugin counted free; `kubectl delete` + recreate → scheduler places it elsewhere.
- **DCP load disabled:** the run uses `load_weights_mode: all_ranks` + `load_checkpoint_path: ""` (loads base HF). The 32-rank DCP-load path was disabled as a Gloo-hang workaround; the fix landed (`checkpointer.py`) but is **unvalidated** — don't switch to DCP load without testing. Base DCP exists at `/shared/huggingface/Qwen3.6-35B-A3B-xorl-dcp-ep8-20260522`.

## 9. History (resolved — context so you don't re-break or mis-diagnose)

The apanda-dev merge broke OPD server init + weight sync; all resolved in this worktree (uncommitted edits). Five init regressions (PG timeout, grouped-load `--model_path` forwarding, rank-0 address skew, DCP-Gloo, quack-EP) → `docs/notes/merge_opd_distributed_init_regressions_handoff.md`. P2P sync wedge (expandable_segments + cold-prepare invalidate + orphaned fused-cat) → `docs/notes/mtp_p2p_weight_sync_wedge_handoff.md`. Don't reintroduce the env in §6.

## 10. Parked work (NOT for the infra pilot unless asked)

- **Crash-resume + auto-relaunch supervisor:** code done (`run_opd_pipeline.py --opd-start-step`, generator bakes `OPD_START_STEP`/`OPD_LOAD_CHECKPOINT_PATH`, `experiments/opd_profile/k8s/opd_trainer_supervisor.sh`); validation paused at the DCP-load step. To enable resume you must validate DCP save→load (load_checkpoint_path = `{RUN_DIR}/server_output/weights/default/<name>`, model+optimizer via `load_state`).
- **Sampler scale-up to 4 replicas:** warranted (sampling is the step wall at 2 replicas) but needs new pods (sglang-2/3) on good nodes (avoid h100-105) + p2p sync to 4 receivers (untested at 4). Render the 4-replica manifest, apply ONLY the sglang-2/3 docs (not the whole manifest — it would disturb running pods).

## 11. Hand-off boundary
Performance / MFU / kernel profiling is a **separate agent**. The profiler hook (`XORL_PROFILE_SERVER_FB`, default-off) and the MFU diagnosis live in `docs/notes/mtp_opd_mfu_profiling_handoff.md`. Do not enable profiling on normal training runs (it inflates step wall ~30×). If perf changes land, they deploy via the same worktree on the next `write-trainer-control`.
