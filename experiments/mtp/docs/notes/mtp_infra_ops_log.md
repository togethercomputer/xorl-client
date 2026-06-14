# MTP infra ops log — `er-opd-q36-mtp-ss-0605c`

Live ops note maintained by the infra agent. Newest entries at top.
Handoff: `docs/notes/mtp_infra_agent_handoff.md`.

## Live status

**2026-06-13 ~22:35Z — CONSOLIDATION WRAP-UP (infra handoff to next-gen agents).** My infra-side contributions, done:
1. **Permanent fix for the NFS-blip bare-pod death** applied to `q36_singleshot_reprogrammable_slots.py` (controller poll loop: `sha256sum` under `set +e` + non-empty-hash guard). py_compile + `bash -n` validated. Effective on the next generator-rebuilt deploy; running pods keep the old loop (resurrector still covers them this session).
2. **Runbook updated** — `mtp_infra_agent_handoff.md` has a "SESSION ADDENDUM (READ FIRST)" with current state + the 3 failure modes (NFS-blip pod death / `write-trainer-control` dangerous defaults / DeepEP-timeout-at-k=2→alltoall).
3. **Memory** `project_mtp_dispatch_bare_pod_death_recovery` refined (confirmed NFS-blip root cause, fix applied).
4. **Coordinated** with perf agent via AGENT_NOTES (shared generator changed — controller loop only).
5. Per `CONSOLIDATION_HANDOFF.md`: this `experiments/opd_profile/k8s/` + `mtp_infra_*` docs harness → **`xorl-infra`**; move the generator (controller fix + WANDB_DIR patch) as one unit; outputs → `/shared`.
Stack: still the perf agent's (throughput sweeps, supervisor PAUSED, 9/9 pods Running). Science ckpts safe on disk. Infra is in a clean, documented, handoff-ready state.

### (prior) live status history

**2026-06-13 ~16:57Z — k=2 RUN WOUND DOWN; stack handed to PERF AGENT (apanda directive).** Trainer stopped at step 521 (4 slots stopped). Science progress preserved: latest ckpt `q36mtp-20260613T110939Z-2s1t/...step000500`, best `...best-step000300` (val_loss 0.2254). Inference stack (sglang-0/1 @k=2, teachers, dispatch) left WARM. **Supervisor PAUSED** (no auto-recovery). Args file = k=2 + **alltoall** EP-dispatch. My run-monitors STOPPED (cron `b477b736` deleted, Monitor v2 `b3q62q8di` stopped) so they don't false-alarm on the intentional idle. **Resurrector `bj9cgjs52` STILL ON** (infra hygiene — auto-recreates Error/controller-dead pods). Baseline that perf is improving: ~62s/step, ~22 tok/s/gpu, MFU ~0.96%, SAMPLING-bound (fb ~20s overlapped under ~85-95s sampling). I'm in light-touch standby as control-plane owner.

### (prior) k=2 run live status

**2026-06-13 ~11:19Z — k=2 SCIENCE RUN LIVE.** Run `q36mtp-20260613T110939Z-2s1t`, resumed step-200, **k_toks=2 + alltoall** EP-dispatch (deepep timed out at k=2 geometry — switched, baked into args), LRs 1e-5/muon 1e-3, ConfAdapt 0.3, no banding. Step 200 done (loss 4.09), sync status=200. Supervisor ARMED (auto-recovers k=2+alltoall). Resurrector live for NFS pod deaths. This is the science verdict's Experiment A. Readout (science): `analyze_mtp_draft_acceptance.py --max-offset 1`. Cron `b477b736` now serves as the 10-min health check on this run.

**~13:07Z — step-300 checkpoint landed** (`...step000300` + `...best-step000300`, val_loss 0.2254). Run now SELF-RESUMABLE at k=2 step-300 (supervisor `latest_checkpoint` resolves to this run dir now). Steady ~10 steps/10min, loss descending 4.09→~3.0 by step 305. Healthy.


- **Mode (from 2026-06-13 ~07:36Z):** stack under ACTIVE multi-agent control. apanda directive (asleep): no questions, best judgment, **KEEP GPUs HOT**, check any run I kick off. Throughput/SGL/science agents run coordinated A/Bs (clean-region proof DONE; q-banding A/B in progress @10:12Z). Supervisor gets paused/unpaused by them around each restart.
- **My role now:** do NOT interfere with active A/B control writes. Guard against the stack being left **idle with supervisor paused** (crashed A/B / forgotten unpause) — that's the keep-GPUs-hot failure. Restore production (resume@200, no banding) only if abandoned.
- **Resume invariant:** all recoveries use `OPD_XORL_REPO=xorl-mtp-commitlen-fix-20260612`, lr 1e-5/muon 1e-3, `OPD_START_STEP=<latest on-disk ckpt>` (step 200 today). The supervisor's env carries these correctly; manual `write-trainer-control` MUST export them (defaults = cold-start step 0 + base weights + wrong worktree/LR — throughput agent hit this twice, now corrected).
- **Supervisor:** PID 2113266 alive; pause toggles frequently during A/Bs (expected). Escalation sentinel absent.
- **Monitors running:** (1) `b3q62q8di` v2 alerting — supervisor death/escalation, dead pods, idle-stack guard (head stale >30min), 30-min heartbeat. (2) `bj9cgjs52` **bare-pod resurrector** — auto-recreates any allowlisted pod in `Error`/`Failed` (=controller died from NFS blip; intentional stops/inner crashes keep pod Running) from `$CTL/_resurrect/<slot>.pod.yaml`; per-pod 5min cooldown + cap 4 then alerts. Covers the supervisor's blind spot (it can't revive dead bare pods).
- **Cron `b477b736`** (every 10m): "check throughput done → kick off k=2". k=2 already launched; subsequent fires just verify health.
- **⚠ Known fragility (CONFIRMED root cause, hit dispatch @06:04 + trainer-worker-3 @10:41):** ALL bare-pod controllers (dispatch + 4 trainer slots, `restartPolicy: Never`) die on a transient `/shared` **NFS read failure**: the poll loop runs `sha256sum run.sh` under `set -e` with no retry, so one failed read kills the controller permanently. NOT a write race — `write_executable` is atomic (tmp+replace) and `stop-trainer-control` has `remove_run=False`. Supervisor CANNOT recover a dead bare pod (writes control, no live controller reads it). **Recovery = delete + re-apply that Pod doc** (`render-manifest --output /tmp/x.yaml`, filter docs by `metadata.name`), then it rejoins (trainer worker rejoins the rendezvous if done within the torchrun timeout). Will recur overnight → mitigation = fast pod-Error detection + auto/manual recreate. Proper fix (tooling owner): `set +e` + retry around the `sha256sum`, or `restartPolicy: OnFailure`. See [[project_mtp_dispatch_bare_pod_death_recovery]].
- **First current-run checkpoint:** ✅ step-200 DCP on disk — and it already earned its keep (the 05:42 recovery resumed from it).
- **Pods:** all 9 Running (dispatch, sglang-0/1, teacher-sglang-0, teacher-smg, trainer-head + worker-1/2/3)

## Config provenance (live rendered trainer-head run.sh)

```
XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612
OPD_FULL_FT_LR=0.00001   OPD_MUON_LR=0.001   optimizer=muon
OPD_START_STEP=100
OPD_LOAD_CHECKPOINT_PATH=.../q36mtp-20260612T204932Z-2s1t/.../q36mtp-coderforge-v1-step000100
```
Args file (`launch_args_er-opd-q36-mtp-ss-0605c.txt`) consistent with this. LR/start/ckpt are env-driven (not in args file) — matches handoff.

## Watch criteria → action

- supervisor PID dead → relaunch supervisor (`nohup bash .../opd_trainer_supervisor.sh >> supervisor.log ...`)
- `supervisor.escalated` appears → 8 recoveries failed; human-class novel failure, investigate logs before any control write
- head log stale >1500s AND supervisor not recovering → manual canonical recovery
- pods Pending/Error/NCCL-init → delete+recreate matching manifest doc (avoid h100-105)
- recovery thrash (repeated relaunch, run keeps dying same signature) → pause supervisor, root-cause

## PENDING ACTION — kick off k=2 prod run when throughput agent finishes (apanda /loop, 2026-06-13 ~10:20Z)

Cron `b477b736` (every 10m) checks throughput-agent done-ness; when done, launch the science verdict's **Experiment A: k=2 bootstrap** (`mtp_science_verdict_20260613.md` §6). Recipe (coordinated restart — samplers MUST match trainer k):

```
# 0. read throughput agent's final note: did they KEEP banding? is clean-region deployed (perf-agent env)? PRESERVE that state.
# 1. bake k=2 into the args file so the SUPERVISOR recovers k=2 (not default k=4):
#    add "--k-toks\n2" to launch_args_er-opd-q36-mtp-ss-0605c.txt  (also keep any banding flag the throughput agent promoted)
# 2. coordinated restart (supervisor already paused by throughput agent; keep it paused until trainer healthy):
export OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612 OPD_FULL_FT_LR=0.00001 OPD_MUON_LR=0.001
export OPD_START_STEP=200 OPD_LOAD_CHECKPOINT_PATH=.../q36mtp-20260613T014150Z-2s1t/.../q36mtp-coderforge-v1-step000200
$PY $G stop-trainer-control $A --k-toks 2
$PY $G write-student-inference-control $A --k-toks 2     # samplers at k=2 (cuda-graph k-list=2)
# wait both samplers "fired up and ready" + dispatch "ready after N polls"
$PY $G write-trainer-control $A --k-toks 2               # trainer OPD_MTP_K_TOKS=2, static_padded_seq_len=2k-1=3
# verify step completes + sync status=200, then rm supervisor.pause
```
Keep LRs/ConfAdapt-0.3/hard_teacher_ce/emit_window unchanged (clean Experiment A). Do NOT switch clean-region mid-run (science: avoid confound). Readout = `analyze_mtp_draft_acceptance.py --max-offset 1`: offset-1 conf slope >0.3 + mask-slot top1 >15% + commit_len>1.10. Judge DRAFT side, NOT val_loss/commit_len alone. Ensure `rollout_samples.jsonl` captured from the resume step. Budget ~150–300 steps.

## Event log

- **2026-06-13 ~10:34Z [infra] — q-banding A/B crashed → launching science k=2 prod run.** Throughput agent's banding A/B (steps 200-202): banding samplers healthy (commit-len ValueError FIXED, sample_sum ~33-45s vs ~120s baseline = the throughput delta they wanted), but worker-1 rank-2 hit `CUDA error: unspecified launch failure` @ step 202 chunk2 (10:31:13, rc=1); head+workers2/3 hung in the collective; supervisor PAUSED (theirs) → no auto-recovery → GPUs idle. Per apanda /loop ("keep GPUs hot, kick off k=2 when throughput done"), TOOK OVER: baked `--k-toks 2` into args file (supervisor-safe), coordinated restart to the science verdict's **Experiment A (clean k=2 bootstrap, NO banding, resume step-200, LRs unchanged)**. static_padded auto-shrinks ~2304→~1277 (2k-1 geometry, generator-derived, no fb-hang). clean-region NOT deployed (throughput reverted proof edits) → k=2 on current pipeline (science-acceptable; don't switch clean-region mid-run). Sequence: stop-trainer → samplers@k=2 (reloading) → [pending] trainer@k=2 → verify → unpause. CUDA launch-failure was on node h100-117 (worker-1); if it recurs there post-restart, exclude/recreate that pod.

- **2026-06-13T06:12–06:31Z [infra manual recovery]** — DISPATCH BARE-POD DEATH. Sequence: (1) ~06:04 dispatch controller died `Error`/exit 1 with `sha256sum: dispatch/run.sh: No such file` — transient run.sh read failure under the controller's `set -e` poll loop (`[ -f ]` then `sha256sum`, no retry); `restartPolicy: Never` → pod stayed dead. (2) Supervisor's 06:05 recovery rewrote dispatch *control* (no live controller to read it) and falsely read stale "ready" lines from the old dispatch log → started trainer (head log 060611Z) against a dead dispatch → `curl: (28)` timeouts, zero progress. (3) My monitor's pod-check caught `dispatch=Error` at 06:12. **Manual fix:** paused supervisor → `stop-trainer-control` → `write-student-inference-control` (fresh samplers, clears stale Mooncake) → `kubectl delete pod dispatch` + `kubectl apply` dispatch Service+Pod from `render-manifest` (new IP 10.42.18.96/h100-046) → verified both samplers "fired up and ready" → `write-trainer-control` with **verified** resume env (worktree `xorl-mtp-commitlen-fix-20260612`, lr 1e-5/muon 1e-3, `OPD_START_STEP=200`, step000200 ckpt — guarded against the generator's dangerous defaults: unset env → cold-start step 0 + base weights + 10–100× wrong LR). **Validated:** step 200 fb (loss 4.59), sync status=200/107 buckets, step 201 running. Supervisor re-armed 06:31Z. ~58 steps re-lost (258→200, same as the prior wedge resume point). Note: also caught + corrected a false-positive in my own watcher (curl:(28)/(7) are normal trainer boot polling the local OPD server on :26050, not a failure).
- **2026-06-13T05:42Z [supervisor auto-recovery, infra verified]** — P2P weight-sync WEDGE. worker-1 log: `batch_transfer_sync to 10.42.16.201:16307 (sglang-1) failed: ret=-1 ... after 50 attempts` (~05:40, cached_prepare/adopted_from_rank0 — stale Mooncake state, the documented mode). Crash was around step ~258. Supervisor recovery attempt 1/8: RESUME from step-200 ckpt → stop trainer → recreate dispatch+samplers (ready 05:44:53, ~2m46s) → trainer restart OPD_START_STEP=200. New head log `20260613T054454Z`. By 06:02: step 205, weight sync restored (status=200, 3s, 107 buckets), no new crash sigs. Net: ~58 steps lost, clean. NO manual action needed — supervisor handled it correctly; I verified only.
- **2026-06-13T04:31Z [infra]** — Step-200 checkpoint landed (current_run_ckpts 0→2). Run self-resumable; a crash now resumes from its own progress, not the old step-100 ckpt. Step 215, ~90s/step, healthy.
- **2026-06-13T03:59Z [infra]** — Came on watch. Full read-only inspection: stack healthy, run at step ~191, supervisor armed, no sentinel, no pause, args/control consistent. No control writes. Established monitoring loop.
