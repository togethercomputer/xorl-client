# P2P Weight-Sync Crash — Diagnosis, Evidence, and Robustness Plan

**Stack:** `er-opd-q36-35b-slots` (Qwen3.6-35B-A3B OPD, P2P/Mooncake weight sync)
**Window:** 2026-06-05 (recurred ~5× during the PTC filler/dig-in sweep)
**Status:** intermittent, recoverable by relaunch; no durable fix landed yet.

---

## 1. Summary

The trainer (sender) periodically fails to push updated weights to the SGLang
student receiver over **Mooncake RDMA**, then **exits rc=1** (all ranks stop). The
sender logs `batch_transfer_sync ... ret=-1 ... after 50 attempts`; the receiver
logs `Peer nic not found in that server: <trainer-ip>:<port>@mlx5_X`. It is
**intermittent** — most syncs succeed, the same recipe ran clean many times — and
the failing HCA **varies** (`mlx5_2`, `mlx5_5`), so it is **not a single dead NIC**.
The best-supported root cause is a **Mooncake RDMA topology/QP handshake mismatch**
between sender and receiver (the receiver can't resolve the peer NIC the trainer
advertises), likely aggravated by the **hardcoded `--mooncake-ib-device` GPU→HCA
mapping** and by **stale Mooncake state accumulating across back-to-back trainer
restarts**.

---

## 2. Failure signature

**Sender — trainer-head** (`h100-096`, `10.42.75.69`), run.log:
```
Syncing registered SGLang endpoints sync_method=p2p serial_endpoint_sync=false
{"success":false,"message":"Weight sync failed: [P2P] batch_transfer_sync to
  10.42.68.44:16577 failed: ret=-1 (bucket 1, chunk 0..8 of 42 buffers,
  sizes=[262144,...], session_info={'world_rank':0,'session_id':'10.42.68.44:16577',
  'endpoint_idx':0}, after 50 attempts)", ...}
Bulk SGLang endpoint weight sync failed; attempting endpoint recovery before exit
```
→ trainer-head process exits **rc=1**; the 7 trainer-worker pods then `stopped`. The
autopilot reports `launch/health failure ... trainer-head:... rc=1`.

**Receiver — sglang-0** (`h100-110`, `10.42.68.44`), pod log:
```
E rdma_endpoint.cpp:219] Peer nic not found in that server: 10.42.75.69:16650@mlx5_5
```
(seen as `@mlx5_2` earlier, `@mlx5_5` later — **the HCA varies**.)

Key reads:
- It fails on **bucket 1, chunk 0..8** (early in the transfer) — a *connection/QP
  setup* failure, not a mid-stream data error.
- `after 50 attempts` — Mooncake's internal retry is exhausted, then the trainer
  gives up and exits. 50 retries is not enough to ride out the flaky window.
- "Peer nic not found" = the receiver's topology view of the sender does **not
  contain the HCA the sender is transferring on**. A naming/registration/topology
  mismatch, not "the link is down."

---

## 3. Evidence ledger (this session)

| run | event (UTC) | outcome | note |
|---|---|---|---|
| PTC-052/050/053/054/055/056 | — | **synced clean** | path works the large majority of the time |
| PTC-057 | head rc=1 @14:25:55 | recovered | 1 relaunch → **2nd attempt succeeded** |
| PTC-058 | P2P ret=-1 (`@mlx5_2`) | not retried | SMG layer died ~same window (see §6) |
| PTC-063 | rc=1 @18:13 and @18:44 | recovered | failed 2 retries, **succeeded after SMG recreate** |
| PTC-076/077 | — | synced clean | orig replicates on the new sampler |
| PTC-078 | head rc=1 @23:42:01 | not retried | had 2 orig replicates already |
| receiver | `Peer nic not found @mlx5_5` @23:28 | — | failing HCA varies vs the earlier `@mlx5_2` |

Pattern: **intermittent and bursty** (clean for hours, then several in a row),
**worsening** over the session, **HCA varies**, **always recoverable by relaunch**
(a fresh trainer Mooncake session often re-establishes cleanly).

---

## 4. Diagnosis — ranked hypotheses

1. **Mooncake RDMA topology/QP handshake mismatch (most supported).** The receiver
   can't resolve the trainer's advertised HCA (`Peer nic not found @mlx5_X`). The
   sender hardcodes a GPU→HCA map
   `--mooncake-ib-device '{"0":"mlx5_2","1":"mlx5_3","2":"mlx5_1","3":"mlx5_5",...}'`;
   if the sender's per-GPU NIC choice isn't registered/visible in the receiver's
   peer metadata at transfer time, QP setup fails → ret=-1. The varying HCA + early
   bucket failure fit a metadata/registration problem, not a dead link.
2. **Stale Mooncake state across back-to-back restarts.** Each candidate relaunch
   re-inits the trainer's Mooncake session ~30s after the previous run tore down. If
   the receiver's RDMA buffers/QPs from the prior session aren't cleanly re-armed
   (the `/prepare_weights_update` re-arm is load-bearing — see refs), the next sync's
   handshake can fail. Consistent with "relaunch into an idle/clean receiver fixes it"
   and with the documented sync-wedge recovery (recreate samplers+dispatch+trainer).
3. **Flaky/contended IB fabric on the trainer node (`h100-096`).** Most crashes
   originate from the same sender node; transient SM-sweep / GID races
   (`local GID -1`) and HCA contention are known on this cluster. The varying HCA
   argues against one permanently-bad NIC but is consistent with intermittent fabric
   flakiness on that node/path.
4. **Insufficient retry budget.** 50 `batch_transfer_sync` attempts then hard-exit
   gives the transient no time to clear; a longer/backed-off retry might ride through.

These are not mutually exclusive; (1)+(2) together best explain "intermittent,
varying HCA, recovers on a clean relaunch."

---

## 5. Why it's slow to detect (operational pain)

- The autopilot uses `--launch-grace-seconds 1800`. A crash at startup (head exits
  ~3–17 min in) is **not flagged until the 1800s grace expires** → **~25–28 min of
  idle** per crash before the autopilot marks `infra_invalid` and stops.
- The autopilot **stops** on `infra_invalid` (it does not auto-relaunch), so each
  crash needs a manual/tripwire-driven relaunch.
- Mitigation already in use: a **tripwire** (`tail -F autopilot.log | grep
  'launch/health failure|infra_invalid|no queued idea'`) that wakes the operator
  when the autopilot logs the failure — but that still waits on the grace.

---

## 6. Related/secondary failure (don't conflate)

Separately, the `dispatch` + `teacher-smg` pods (both on `h100-114`, bare pods, no
owner → no self-heal) were **terminated/evicted** ~19:01, after which every trainer
launch **hung** at `Waiting for SMG dispatch model registration` (not a P2P crash —
a *hang*). Recovered by recreating just those two pods (extract from rendered
manifest → delete Failed → re-apply → model re-registers). This is a node/eviction
issue, distinct from the RDMA sync crash, but both feed the "stack is fragile under
churn" picture.

---

## 7. Robustness plan (ranked by impact/effort)

1. **Auto-retry the P2P-sync crash in the autopilot (lowest effort, highest payoff).**
   Treat the `Bulk SGLang endpoint weight sync failed` / `batch_transfer_sync ret=-1`
   / trainer-head `rc=1` signature as a **retryable infra failure**: on detection,
   `stop-trainer-control` → re-queue the candidate (bump `infra_retry_count`) →
   relaunch, up to N (e.g. 3) retries. It's intermittent and a clean relaunch usually
   works (PTC-057, PTC-063), so bounded auto-retry would have absorbed ~all of these
   without operator involvement. (Current code `stop`s on `infra_invalid`; this is the
   gap.)
2. **Cut detection latency.** A worker/head that **exits non-zero** is a definitive
   failure — flag it immediately instead of waiting out `--launch-grace-seconds 1800`.
   Add a startup-phase health check that trips on `rc!=0` regardless of grace (grace
   should only cover "slow to produce a profile," not "already crashed").
3. **Raise Mooncake retry/backoff.** Bump `batch_transfer_sync` attempts well above
   50 with exponential backoff so a transient handshake window clears without the
   trainer exiting.
4. **Attack the handshake root cause.** Try dropping the hardcoded
   `--mooncake-ib-device` map (let Mooncake auto-discover; the explicit map may
   advertise NICs the receiver can't resolve), and verify sender/receiver IB topology
   metadata agree. Move the stack to the **non-privileged `rdma/infiniband` +
   `IPC_LOCK`** pod spec (see CLAUDE.md "Do NOT use privileged") — cleaner IB exposure
   may reduce handshake flakiness.
5. **Node hygiene / restack.** The trainer-head on `h100-096` and the SMG pods on
   `h100-114` were the hotspots; a clean restack onto known-good nodes (and making the
   inference pods owned/self-healing rather than bare) removes both failure classes.
6. **Land the desync fix.** The working-tree patch in
   `[[project_opd_p2p_sync_wedge_collective_desync]]` makes all ranks raise together
   on a partial P2P failure (avoids the separate 1800s orchestrator hang). Land it so
   a sync failure fails *fast and cleanly* instead of wedging.

**Recommended first step:** implement #1 (bounded auto-retry in the autopilot) + #2
(immediate rc!=0 detection) — together they make the recurring crash a non-event
operationally, independent of whether the RDMA root cause is ever fixed.

---

## 7b. UPDATE 2026-06-06 — supervisor IMPLEMENTED + validated; new 504 variant; duplicate-autopilot race

Robustness item #1 (auto-retry) is now LIVE as `autoresearch/autopilot_supervisor.py`
(detached, pidfile `logs/supervisor.pid`): polls 60s, and when no autopilot is
alive it re-queues recently (<3h) crashed `infra_invalid` ideas (bounded
`infra_retry_count<3`) and relaunches the autopilot; escalates in-log after 3.

**First live save (validated):** `PTC-089` trainer-head exited rc=1 at
2026-06-06T01:37:15Z; the autopilot only declared the failure ~01:50 (it waits out
`--launch-grace-seconds 1800` from the 01:20 launch) and exited; the supervisor
detected it at 01:51:16, re-queued PTC-089 (retry 1/3), relaunched, and PTC-089 was
back to step 0 by 01:57 — no operator action. Confirms bounded auto-retry absorbs
the recurring crash.

**New crash variant:** this one was NOT the classic Mooncake `batch_transfer_sync
ret=-1`. Trainer log showed `curl: (22) ... error: 504` then `Bulk SGLang endpoint
weight sync failed; attempting endpoint recovery before exit`. So the sync-crash
family also presents as an **HTTP 504 (gateway timeout)** on a weight-sync call, not
only the RDMA ret=-1. Same exit path, same supervisor handling. (It is infra, not
lr=0-specific — PTC-089's only diff from the clean PTC-088 was `learning_rate:0`,
which cannot cause a sync 504.)

**Duplicate-autopilot race (FIXED):** the autopilot exits on crash and has no
single-instance guard, so a **concurrent cron-tick agent** following the old
RUNBOOK §0a ("restart autopilot if dead") relaunched it at the same time as the
supervisor → briefly TWO autopilots (would race `ideas.yaml`). `ideas.yaml` was
checked intact (one `launched`), no corruption. Fixes: (a) supervisor now runs
`enforce_single_autopilot()` each poll (kills duplicates, keeps oldest);
(b) RUNBOOK §0a rewritten — agents must NEVER manually launch the autopilot; only
the supervisor owns its liveness; agents only relaunch the *supervisor* if it (not
the autopilot) is dead (pidfile-guarded, safe).

## 7c. UPDATE 2026-06-06 03:33 — degradation went PERSISTENT (stale receiver Mooncake registry)

The crash stopped being bursty and became **persistent**: no run has completed since
PTC-078 (~02:17). Every trainer session now dies at its ~step-10 weight sync (PTC-089
01:37; PTC-082 02:34, 03:05 — i.e. PTC-082 is at supervisor retry 2/3). Signature:
trainer `curl 504` → "Bulk SGLang endpoint weight sync failed"; **receiver sglang-0
logs `Peer nic not found in that server: 10.42.75.69:16512@mlx5_3` on repeat** (HCA
varies: mlx5_2/3/5 across the night). ALL pods are Running/Ready (sglang-0 up 23h, no
OOM/restart) — so it is NOT a dead pod; it is the **receiver's Mooncake peer registry
gone stale** after 23h + a dozen trainer-session restarts tonight (hypothesis #2,
now dominant). The bounded supervisor relaunch can't clear it because the staleness
is on the *receiver* side, not the trainer.

**READY-TO-RUN RECOVERY (trigger: PTC-082 hits retry 3/3 = `ESCALATE`, OR no completion
for >90 min).** Refresh the student sampler to clear its Mooncake state (in-design,
no pod delete), coordinated so the supervisor doesn't relaunch a doomed trainer mid-refresh:
```bash
cd /home/apanda/xorl-apanda-dev-opd-port
AR=experiments/opd_profile/autoresearch ; GEN=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
# 1. stop supervisor so it won't relaunch the autopilot mid-refresh
kill -TERM "$(cat $AR/logs/supervisor.pid)" ; rm -f $AR/logs/supervisor.pid
# 2. stop autopilot (no new trainer launches) + halt current trainer
pkill -f 'controller.py autopilot' ; python $GEN stop-trainer-control
# 3. refresh the sampler (restarts sglang-0 → clears stale Mooncake peer registry)
python $GEN write-student-inference-control --sampler-replicas 2 --sampler-layout spare-teacher1
# 4. wait until sglang-0 is ready=true again (re-registers Mooncake clean)
kubectl -n apanda get pods -l stack=er-opd-q36-35b-slots | grep sglang-0
# 5. reset the stuck idea to queued, then restart the supervisor (it relaunches the autopilot into the refreshed stack)
#    (edit ideas.yaml: PTC-082 status launched/infra_invalid -> queued), then:
setsid nohup python $AR/autopilot_supervisor.py >> $AR/logs/autopilot_supervisor.nohup 2>&1 < /dev/null &
```
Verify success: the next launched run reaches a clean step-10 sync (`Weight sync
complete: … GB`) instead of `curl 504`. If it still 504s after a sampler refresh,
escalate to recreating sglang-0's pod, or drop the hardcoded `--mooncake-ib-device`
map (§7 #4). Do at most ONE refresh attempt unattended; if it doesn't take, stop and
hand off — don't thrash the receiver.

**Why not done unattended at 03:33:** the deep-analysis program (the stack's purpose)
is COMPLETE; remaining queue is low-value; PTC-082 is retry 2/3 (not yet the RUNBOOK's
`ESCALATE` trigger); the cluster is non-idle (partial work each cycle). A failed
unattended sampler restart could idle the whole stack (worse). So: follow the
documented ESCALATE trigger rather than preempt with risky surgery.

**OUTCOME 2026-06-06 04:33 — the burst SELF-CLEARED; §7c recovery was NOT needed and
NOT run.** When the recovery was attempted at 04:03 it was DENIED by the auto-mode
permission classifier (disruptive shared-infra surgery outside the monitor scope) —
correctly, as it turned out. PTC-082's 4th retry (03:50) hit a good handshake window
and completed at 04:06 (weak_signal); PTC-083 completed 04:21 (strong_signal); the
queue resumed. So the supervisor's bounded retry rode out a ~90-min bursty window
WITHOUT the sampler restart. **Reinforced lesson:** for the bursty Mooncake
`Peer-nic-not-found` failure, prefer patience (bounded retry) over the §7c sampler
restart — reserve §7c for a burst that genuinely never clears (a real `ESCALATE` with
zero completions across all retries), and even then surface to the user for
authorization rather than auto-running it.

## 7d. ROOT-CAUSE FIX (2026-06-06): pin the Mooncake handshake port (no worker restart)

**Root cause confirmed:** the trainer's Mooncake session id is `{hostname}:{rpc_port}`,
and the rpc/handshake port is **auto-assigned (ephemeral)** — a NEW port every trainer
restart (`p2p.py` reads it back via `get_rpc_port()` after `initialize(hostname,
"P2PHANDSHAKE", "rdma", ib)`). The SGLang receiver caches/accumulates peer metadata keyed
on the trainer session; across restarts the old session goes stale → "Peer nic not found".
There is **no API to clear just the receiver's peer registry** (the Mooncake engine is a
process-global singleton; the pybind binding exposes no peer-teardown), so the only clear
today is a worker restart (§7c, 5-min reload).

**The fix — stable session id via a pinned handshake port (env var, no model reload):**
Mooncake honors `MC_HANDSHAKE_PORT` (confirmed in `engine.so`; the "Ignore value from
environment variable MC_HANDSHAKE_PORT" string is the generic invalid-value path, shared by
~20 MC_* vars, not "always ignored"). Pinning it makes the session id stable across restarts
→ the receiver's peer metadata stays valid → the bursts never happen. Implemented opt-in in
`p2p.py` `PeerTransferEngine.__init__`: if `XORL_P2P_HANDSHAKE_BASE_PORT` is set, it does
`MC_HANDSHAKE_PORT = base + gpu_id` **per rank** (each trainer pod has 8 GPUs/ranks on one
host, so a single fixed port would collide — `+gpu_id` gives 8 distinct, stable ports).
Default (unset) = current auto-assign, so it is INERT until enabled.

**VALIDATED 2026-06-07 00:33 — THE PIN WORKS; BURSTS ELIMINATED.** First pinned launch
PTC-094 ran the FULL 31 steps with no crash (pre-pin, runs died ~every 17 min at the
step-10 sync); the 094→095 trainer RESTART (same pinned port) launched clean; sglang-0
logged ZERO `Peer nic not found` and there were no sync crashes/re-queues across the
whole pinned window. No TIME_WAIT bind failure. Manual §7c sampler refreshes are RETIRED;
keep `XORL_P2P_HANDSHAKE_BASE_PORT=16400` as the default. (Keep watching across more
restarts, but the mechanism is confirmed.)

**ENABLED 2026-06-06 23:35 (user-authorized):** `XORL_P2P_HANDSHAKE_BASE_PORT=16400` is now
the default in the trainer env (`q36_35b_reprogrammable_slots.py`), so every NEW trainer
launch pins ranks to 16400..16407. Takes effect on the next launch (PTC-094; the running
PTC-099 keeps its old auto-port). VALIDATION IN PROGRESS — watch the next launches for:
(a) trainer log `pinning Mooncake handshake port to 1640X`; (b) clean first sync; (c) NO
`Peer nic not found` on sglang-0 across the 094→095→… restarts (the real test — stable
session id); (d) NO `unable to find available tcp port` (TIME_WAIT bind fail). If (d)
bites, override `XORL_P2P_HANDSHAKE_BASE_PORT=` (empty) to revert to auto, or try
`MC_LEGACY_RPC_PORT_BINDING`. No sampler refresh needed — the first pinned launch
establishes a fresh stable session the receiver keeps.

**(original) To enable + VALIDATE (do AFTER the current baseline wave; needs an empirical check):**
1. Set `XORL_P2P_HANDSHAKE_BASE_PORT=16400` (or any free base) on the trainer pods — add it
   to the trainer env in `q36_35b_reprogrammable_slots.py`. Ranks then bind 16400..16407.
2. Launch ONE candidate; confirm trainer log shows `pinning Mooncake handshake port to ...`
   and the run syncs clean.
3. **The real test:** force a trainer restart mid-run (stop-trainer-control → relaunch) and
   confirm the next sync does NOT log "Peer nic not found" on sglang-0 (i.e. the receiver's
   cached peer is still valid because host:port is unchanged). If clean → the fix works;
   wire it into the default trainer env so the §7c restart is never needed again.

**Caveat — TIME_WAIT:** a pinned port may be in TIME_WAIT briefly after a restart → bind
fail ("unable to find available tcp port"). Trainer engine init takes minutes (TIME_WAIT is
~60s) so usually clear by rebind; if it bites, Mooncake has `MC_LEGACY_RPC_PORT_BINDING`
(SO_REUSEADDR-style) to investigate. This is why it ships opt-in + needs the restart test
before becoming default.

## 8. Cross-references

- `[[project_opd_p2p_sync_wedge_collective_desync]]` — partial-failure desync → 1800s
  hang; working-tree except-handler fix; recover by recreating samplers+dispatch+trainer.
- `[[feedback_opd_no_skip_cached_prepare]]` — `/prepare_weights_update` re-arms the
  receiver's Mooncake RDMA buffers; load-bearing, skipping → ret=-1 ×50.
- `[[feedback_mooncake_requires_privileged]]` (now: use `rdma/infiniband` + `IPC_LOCK`,
  NOT privileged) + CLAUDE.md "Do NOT use privileged on GPU workloads".
- `[[project_opd_inference_stack_restart_recipe]]` — stale-state restart hazards.
- `[[project_apanda_multinode_network_blockers]]` / `[[feedback_nccl_ib_env_vars]]` —
  bad-IB-HCA / SM-sweep `local GID -1` race; NCCL_IB env caveats (and that they break
  Mooncake init — `[[feedback_nccl_ib_breaks_mooncake_init]]`).
- Operational tripwire pattern: `tail -F $(cat /tmp/ptc-autopilot-current-log) | grep
  -E 'launch/health failure|infra_invalid|no queued idea'`.
