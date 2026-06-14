# MTP OPD — P2P weight-sync RDMA wedge (handoff)

**Date:** 2026-06-08 ~03:35Z
**Stack:** `er-opd-q36-mtp-ss-0605c` (Qwen3.6-35B-A3B SingleShot-MTP OPD, 4-node trainer)
**Branch:** `codex/mtp-singleshot-port-20260602` (worktree `/home/apanda/xorl-mtp-singleshot-port-20260602`)
**Author of this note:** monitoring/ops agent. Companion to
`docs/notes/merge_opd_distributed_init_regressions_handoff.md` (the 5 init regressions, all RESOLVED).
This doc covers the **one remaining blocker**: the on-policy weight sync.

---

## ✅ RESOLVED (2026-06-08 17:16) — root cause was `expandable_segments`, NOT the p2p.py code

The hang was the trainer running `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which **breaks
Mooncake P2P registration >~20 MiB** (canonical runbook §O3, marked mandatory-not-to-use). My MTP
generator re-introduced it to dodge the NCCL `alltoall_pre_dispatch` 2 MB calloc (under `NCCL_CUMEM_ENABLE=0`
+ MTP static-padded dispatch); the working canonical (`baf6dcce`, non-MTP) never sets it. The dense
`lm_head` (CPU pinned pool) survived, but the `>20 MiB` MoE **expert** buffers in expandable CUDA segments
broke registration → silent `copy_`/RDMA hang in `_direct_ep_transfer_experts` (watchdog: `p2p.py:2035`,
workers in native RDMA, 37 `cuda` frames).

**Fix:** made expandable **default-off** in the trainer shells of
`experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py` (opt back in via
`OPD_TRAINER_EXPANDABLE_SEGMENTS=1`). **Verified end-to-end:** step-0 fb completed with **no** calloc
failure (conf0.3/mnt256 has headroom), then `sync_inference_weights returned status=200 success=True
transfer_time=3.822s bytes=69.3 GB buckets=107 in 11.4s` — every module flowed (layer 0 = 5 ms), and
`=== OPD step 1 ===` started. **Full OPD loop running on p2p, 4 nodes.**

Two real *latent* merge regressions remain to land in the apanda-dev PR (neither was the hang, both
confirmed/predicted): (1) `p2p_invalidate_cache` cold-prepare receiver re-arm (restored, confirmed on the
wire — but Run 3 still hung *with it on*, proving it wasn't the hang); (2) the orphaned
`_slice_qwen_linear_attention_fused_param` fused-cat (glm5 rebase `955191dd`; no live receiver counterpart;
would surface as a *size-mismatch raise* once arming/registration is healthy — bypass it gated default-on).

---

## TL;DR (historical — see RESOLVED above)

The full OPD loop is **one step from running on 4 nodes**. Everything works **except the
trainer→sglang weight sync over p2p (Mooncake RDMA)**, which **hangs in the RDMA data plane**
and never returns. It has now wedged **3 consecutive times**. The handshake-pin fix (init-regression
#4, already re-ported and live) did **not** resolve it — the hang is downstream of the handshake.

- **Training is healthy and fits.** Step 0 fb+optim complete on 4 nodes, conf0.3/mnt256, no OOM
  (`loss=0.8756`, `grad_norm=41.07`, `valid_tokens=47040`). Reproduced 3×.
- **Samplers/dispatch/teachers are healthy.** Recreated fresh at 02:54Z; both samplers serve at
  256-concurrency; dispatch sees both backends.
- **The sync is the sole blocker.** It hangs after the control plane succeeds; orchestrator times
  out at 1800s and the trainer exits.

**Primary hypothesis:** the apanda-dev merge's FP8 commit `0acb8618`
(*"feat(fp8): FP8 low-precision full-stack training, QARL fake-quant, and weight-sync fp8_kv_cache"*)
**rewrote the p2p weight-sync path** and regressed the multi-node sender/RDMA-transfer flow. This is a
**new regression (call it #7)**, distinct from the 5 init regressions in the companion doc.

---

---

## UPDATE — Run 2 (Fix 1 + Fix 2 applied): STILL WEDGES, now precisely localized

A second agent ruled out the rank-divergent-collective theory (sender init / EP-MoE materialization
gating are byte-identical to `baf6dcce`; materialization is local) and restored **two merge-deleted
safety nets in `p2p.py`**: **Fix 1** = bounded drain timeout in `_wait_all_pending` (`:1574`,
`XORL_P2P_PENDING_TRANSFER_TIMEOUT_S`, our env=120s) + `FutureTimeoutError` import (`:27`); **Fix 2** =
small-entry registration fallback `if not out: return intervals` in `_intervals_per_cuda_segment`
(`:2330`). Both verified present and live. **Re-ran 4-node p2p (`035503Z`) → wedged identically**
(`status=-1 bytes=0 buckets=0 transfer_time=0.000s in 1800.0s`, `cleanup rc=6`).

### *** Where the sender logs actually are (critical for all future diagnosis) ***

The head pod's `…/trainer-head/logs/<rev>-run.log` is **only the OPD client** (`run_opd_pipeline.py`,
run.sh:332) — it shows just `Calling sync_inference_weights` → 1800s HTTP timeout. **The trainer SERVER
(ranks 0–31, incl. the senders 0–7) logs to `${RUN_DIR}/server.log`** (run.sh:315), where
`RUN_DIR=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/<stack>/<RUN_ID>/`
(`RUN_ID = q36mtp-<rev>Z-2s2t`). **Read `server.log`, not run.log, to see the senders.**

### Exact hang point (from `server.log`, run `035503Z`)

```
04:05:36  Rank 0: Module (root): 1 params → Broadcasting 1 params, 1017.1 MB → 1 transfer bucket
04:05:37  Rank 0: [WeightSync timing] (root): unshard=80 broadcast=1049 ... total=1145ms     ← ROOT/lm_head DONE ✓
04:05:37  Rank 0: Module model.layers.0: 16 params → Broadcasting 16 params, 74.8 MB
04:05:37  [P2P] CPU scratch pool 1/2 registered 2.15 GB ...  ×8 ranks
04:05:37  [P2P] using 2 concurrent Mooncake workers          ×8 ranks   (p2p.py:1572, _ensure_transfer_executor)
   ...    (SILENCE on every rank for 30 min — no slice log, no Fix-2 fallback log, no Fix-1 timeout, no ret=-1)
04:35:30  signal 15 → killed
```

**The dense path works; the FIRST small-param linear-attention layer hangs.** The root module
(`lm_head`, 1 dense param, 1017 MB) transfers fine. **`model.layers.0` — Qwen3.6's first GatedDeltaNet
linear-attention layer (16 small params: `A_log`, `dt_bias`, `conv1d`, norms, 74.8 MB) — hangs** right
after the transfer executor spins up, before any transfer completes or any drain happens.

### What this rules in / out

- **Fix 1's 120s drain timeout NEVER fires** → the hang is **upstream of `_wait_all_pending`**, in
  transfer **item-build / submission** for layer 0 (or the worker-thread submit blocks before the drain).
  It is *not* the async-future-drain path.
- **Fix 2 does not prevent it** → either the small/sliced linear-attn params still build a bad transfer
  item, or the hang is in item construction itself (not registration).
- **Confirmed mechanism for the global wedge:** non-sender ranks (8–31) all reach `Processing PP stage 0
  (83 modules)` then block at the next all-rank `unshard()` barrier (handler.py:~1453) waiting for the
  stalled senders → all 32 ranks wedge → 1800s orchestrator timeout. (The `unshard()` lock-step theory
  is correct.)

### Prime suspect (promoted) + next step

`_slice_qwen_linear_attention_fused_param` (`p2p.py:~2505`) — **new functional code** that runs for
exactly these linear-attention fused params, deliberately untouched by Fix 1/2. It (or the small-param
interval/registration path it feeds) is the last unaudited thing between "using 2 concurrent Mooncake
workers" and the first completion. **Next step:** add per-step logging in `transfer_bucket` (`:1745`) /
`_do_async_transfer` (`:618`) / `_run_async_transfer_items` (`:471`) / `_intervals_per_cuda_segment`
(`:2272`) / `_slice_qwen_linear_attention_fused_param` (`:2505`) to pin the exact blocking call for
`model.layers.0`, and separately determine why a stalled worker future doesn't trip Fix 1's
`_wait_all_pending` timeout (is the wait reached at all for the direct/sync transfer path vs the async
one?). A fast bisect: temporarily force the linear-attn layer down the same dense/broadcast path the
root module used (which works) to confirm the slice/small-entry path is the trigger.

---

## UPDATE — Run 3 ROOT CAUSE FOUND + fixed (cold-prepare receiver re-arm dropped by FP8 merge)

The Run-2 localization (layer-0 hang, bytes=0, no ret=-1, Fix-1 timeout never fires) led to the exact
blocking call and the real regression:

- **The blocking call:** `prior_future.result()` at `p2p.py:1911` (now `:~1935`) — a **second, separate
  no-timeout futures wait** in `transfer_bucket`, distinct from `_wait_all_pending`. It's the per-bucket
  CPU-pool-slot reuse drain: before overwriting pool slot N it blocks on the *prior* bucket's Mooncake
  worker future. lm_head (1017 MB) is chunked into ~8 buckets across 2 pool slots; when `model.layers.0`
  reuses a slot whose prior chunk's `batch_transfer_sync_write` never completed, this `.result()`
  deadlocks **in the synchronous per-module loop, before any async future exists** — which is exactly why
  Fix-1's `_wait_all_pending` timeout never fired (it's never reached).
- **Why the prior transfer never completes (ROOT CAUSE):** the FP8 commit `0acb8618` **deleted the
  cold-prepare receiver cache-invalidation**. Baseline `baf6dcce` set
  `invalidate_receiver_cache = not request_cached_prepare and _env_flag(...COLD_PREPARE, True)` and sent
  `payload["p2p_invalidate_cache"] = True` on a COLD `/prepare`. On a cold 4-node sync the trainer no
  longer tells the receiver to drop its stale Mooncake registration and re-arm its RDMA buffers, so the
  sender's `batch_transfer_sync_write` blocks with **no completion and no `ret=-1`** — the silent
  data-plane wedge. (Matches `feedback_opd_no_skip_cached_prepare`: the /prepare call is load-bearing for
  re-arming the receiver.) In multi-sender direct-EP mode rank 0 does the prepare via
  `_initialize_single_sender` (`_initialize_multi_sender:1256`), so this one fix covers the 8-sender path.

**Fixes applied (all in `p2p.py`, all clean reverts/guards of merge-dropped code):**
1. **Restored cold-prepare cache invalidation** — `invalidate_receiver_cache` (`:1374`) + `payload["p2p_invalidate_cache"]=True` (`:1395`) in `_initialize_single_sender`. *This is the substantive fix that should close the loop.*
2. **Bounded the per-bucket reuse drain** — `prior_future.result(timeout=XORL_P2P_PENDING_TRANSFER_TIMEOUT_S)` + raise on timeout (`:~1935`), so any residual wedge fails in ~120s with an attributable per-rank error (surfaced symmetrically via the `_gather_p2p_transfer_statuses` except-path) instead of a 30-min silent hang.
3. (Run-2) bounded `_wait_all_pending` + restored `_intervals_per_cuda_segment` fallback (still valid).

**Status:** compile + ruff clean; weight_sync tests 257 passed (1 pre-existing unrelated failure). **Needs a
live 4-node sync.** Expected: the loop closes (Fix 1 re-arms the receiver). If it still wedges, Fix 2 now
fails it fast at the *exact* bucket with a per-rank error, and the remaining suspect is the receiver-side
honoring of `p2p_invalidate_cache` in `xorl-sglang-internal` (verify the sampler image implements it).
`_slice_qwen_linear_attention_fused_param` was **ruled out** (with `XORL_P2P_CPU_POOL_MIN_BYTES=0` every
layer-0 param takes the large/CPU-pool path; the slice runs before the hang and guards size mismatches).

---

## What works (do not re-investigate)

| Component | State | Evidence |
|---|---|---|
| 4-node distributed init | ✅ | PG init, engine up, all init regressions #1–#5 fixed/worked-around |
| Prompt load + schedule | ✅ | 128,034 prompts, schedule + eval schedule built |
| Initial greedy rollout | ✅ | reached `=== OPD step 0 ===` (this is the call that 900s-timed-out when samplers were wedged) |
| Step-0 forward_backward | ✅ | `loss=0.8756 valid_tokens=47040 fb_sum=209.4s sample_sum=208.0s` — fits, no OOM |
| Optimizer step (Muon) | ✅ | `optim_step grad_norm=41.0748 roundtrip=3.912s` |
| Samplers (sglang-0/1) | ✅ | `server is fired up`, `max_running_requests=256, context_len=262144`; MTP/ConfAdapt decode |
| Dispatch | ✅ | both student backends `ready after N polls` |
| Trainer RDMA pod setup | ✅ | `rdma/infiniband:1` + `IPC_LOCK`, **not** privileged, **no** `NCCL_IB_GID_INDEX/HCA`, no hardcoded `CUDA_VISIBLE_DEVICES` (correct per CLAUDE.md Mooncake rules) |

---

## The blocker — p2p sync hangs in the RDMA data plane

### Timeline (single sync, step 0 → step 1)

```
03:07:55  Pipeline forward_backward step=0 loss=0.8756  (fb done)
03:07:59  optim_step grad_norm=41.0748                  (optim done)
03:07:59  [head] Calling sync_inference_weights master_address=10.42.21.204 buffer_size_mb=4096 pause_mode=retract weight_version=opd-step0
03:07:59  [workers, ranks 8-31] WeightSync sync_method=p2p, endpoints=2, cache_invalidation_mode=auto, fp8_kv_cache_enabled=False
03:07:59  [workers] P2P direct-EP sender ranks=(0,1,2,3,4,5,6,7)  ← SENDERS are head ranks 0-7
03:07:59  [workers] P2P Mooncake trainer binding: gpu_id=1 ib_device=mlx5_3  (rank 9), gpu_id=4 mlx5_9 (r12), gpu_id=6 mlx5_6 (r14), ...
03:07:59  [workers, non-senders] Skipping EP MoE tensor materialization on non-sender P2P rank
03:07:59  [workers, non-senders] Processing PP stage 0 (83 modules, remote=False)   ← workers then BLOCK here
03:08:01  [sglang-0 RECEIVER] [P2P] receiver register mode=allocator: 206 CUDA allocator segment regions
03:08:01  [sglang-0 RECEIVER] [P2P] receiver registering 206 memory regions strict=True chunk_size=4096
03:08:03  [sglang-0 RECEIVER] POST /prepare_weights_update HTTP/1.1 200 OK
   ...    (SILENCE — no transfer, no ret=-1, no progress on any rank, for 30 min)
03:37:59  [head] sync_inference_weights returned status=-1 success=False transfer_time=0.000s bytes=0 buckets=0 in 1800.1s
03:37:59  [head] ERROR sync_inference_weights failed status=-1 response={'success': False,
                 'message': "HTTPConnectionPool(host='127.0.0.1', port=26050): Read timed out. (read timeout=1800.0)"}
03:37:59  trainer-head cleanup rc=6
```

### *** KEY: the transfer never started (bytes=0, buckets=0, transfer_time=0.000s) ***

The final signature is decisive: **`bytes=0 buckets=0 transfer_time=0.000s`**. The sync hung in
**sender-side SETUP, before a single bucket was shipped** — it is **NOT** an RDMA-write/route problem
(no write was ever issued). The reported failure is a **1800s read-timeout on the client→trainer-server
HTTP call** (`run_opd_pipeline.py` POSTs the trainer's own server at `127.0.0.1:26050`; the server-side
WeightSync handler never returned because the head sender ranks are blocked). So the receiver's
`/prepare` 200 was reached, but the trainer's sender path then deadlocked **before** entering the
per-bucket `transfer_bucket` loop.

This **downgrades** the "RDMA route/HCA" hypothesis and **upgrades** "sender-side deadlock in
materialization / engine construction / a cross-rank collective during sync setup" — i.e. the
FP8-rewrite or an EP-MoE materialization barrier. Look at what the sender does *between*
`/prepare_weights_update` and the first `transfer_bucket` call.

### What is confirmed healthy vs hung

- **Control plane = OK.** Receiver registered its 206 CUDA allocator regions and `/prepare_weights_update`
  returned **200** (from `10.42.21.204` = head/rank-0). So the HTTP coordination, Mooncake registration,
  and handshake **all completed** — the sender successfully told the receiver to arm.
- **Sender setup = HUNG, before any transfer.** After `/prepare` 200, the sender **never ships a single
  bucket**: final return is `bytes=0 buckets=0 transfer_time=0.000s` (see KEY box below). No `batch_transfer_sync`
  is ever issued, no `ret=-1` fires — the deadlock is *upstream* of the per-bucket transfer loop.
- **Sender (head ranks 0-7) logged nothing after `Calling sync_inference_weights`.** No
  `P2P Mooncake trainer binding`, no `Processing PP stage`, no transfer log on the head — while the
  *non-sender* worker ranks logged their bindings and reached the PP-stage barrier. So the **senders are
  stuck very early** (EP-MoE materialization, Mooncake engine construction, or the first
  `batch_transfer_sync`), or their logs are buffered behind a blocking C++ Mooncake call.

### Topology of this sync

- 4-node trainer, FSDP=32, EP=8. **Sender ranks = 0–7 (the head node)**, "direct-EP" mode
  (`supports_direct_ep_transfer`, `p2p.py:2237`). Each head rank ships its EP shard.
- 2 receiver endpoints (sglang-0, sglang-1), each TP=2 → 4 receiver GPUs. Receiver IB devices
  `mlx5_9` (cuda:0) / `mlx5_6` (cuda:1); trainer binds per-GPU IB devices (`mlx5_3`, `mlx5_6`, `mlx5_9`, …).
- `buffer_size_mb=4096`, `pause_mode=retract`, `weight_version=opd-step0`.

---

## Why it is NOT the handshake pin (rule out #4 first)

Init-regression **#4** was: the merge dropped the `p2p.py` code mapping
`XORL_P2P_HANDSHAKE_BASE_PORT → MC_HANDSHAKE_PORT = base + gpu_id`. That was **re-ported and is live**:

- `src/xorl/server/weight_sync/backends/p2p.py:262–291` reads `XORL_P2P_HANDSHAKE_BASE_PORT` and sets
  `MC_HANDSHAKE_PORT` **before** the TransferEngine is constructed.
- Env is set: `XORL_P2P_HANDSHAKE_BASE_PORT=16400` (trainer `run.sh:39`).

Evidence the handshake is **not** the problem: the receiver **registered + `/prepare` 200'd**, which
requires a working sender↔receiver connection. The hang is *after* a successful handshake, in the
actual RDMA write. (Note: the receiver Transfer Engine listens on auto-assigned ports `:15591/:15871`,
not 16400+id — but since registration/prepare succeeded, port discovery is dynamic and not the issue.)

---

## Primary hypothesis — FP8 weight-sync merge regression (commit `0acb8618`)

`git log --oneline` head: `0acb8618 feat(fp8): FP8 low-precision full-stack training, QARL fake-quant,
and weight-sync fp8_kv_cache`. This commit **modified the weight-sync handler/p2p path**. Evidence it
is active in our run: the handler now logs **new fields** that did not exist pre-merge —
`cache_invalidation_mode=auto, fp8_kv_cache_enabled=False, fp8_kv_cache_postprocess_required=False,
fp8_kv_cache_static_scales=False, quantization=None`.

Even with FP8 *disabled*, the rewrite may have changed the **sender materialization / bucketing /
transfer-issue order**, introducing a multi-node deadlock or a never-completing async transfer. This is
the same failure *class* as the 5 init regressions: the merge silently changed coordination code.

**Suggested first action:** diff the weight-sync sender path against pre-merge `baf6dcce`:

```bash
git diff baf6dcce..HEAD -- src/xorl/server/weight_sync/ src/xorl/server/runner/ | less
# focus: p2p.py transfer_bucket / _do_async_transfer / _run_async_transfer_items / supports_direct_ep_transfer
#        handler.py around the direct-EP sender path (see code pointers below)
```

If the diff shows a sender/async-transfer or cache-invalidation change, that is the lead. If the sender
path is byte-identical to `baf6dcce` (as grouped/DCP module bodies were — that was a red herring there),
pivot to the secondary hypotheses.

---

## Secondary hypotheses (if not the FP8 commit)

1. **Multi-node trainer P2P is known-flaky on this stack.** Memory `project_p2p_2node_tp8_findings`:
   *"2-node TP=8 receiver works end-to-end but lands at 44s with silent data corruption from Mooncake
   `ret=-1` retry path."* This is a **4-node** trainer (FSDP=32) → unprecedented; the all-prior-working
   P2P numbers (`project_p2p_sync_8s_target_hit`, 5–8s) were **1-node** trainers. The direct-EP sender
   fan-out from 8 head ranks → 4 receiver GPUs across nodes may expose an RDMA QP/route problem.
2. **EP-MoE materialization collective deadlock on senders.** Sender ranks 0–7 materialize EP shards
   while ranks 8–31 "skip"; if materialization involves a collective with a rank-divergent signature
   (the recurring bug class — see `feedback_dist_allreduce_dict_keyed`), the senders hang before transfer.
   The head logging *nothing* after `Calling sync_inference_weights` is consistent with this.
3. **Async-transfer submit/poll never completing.** `p2p.py:471 _run_async_transfer_items` /
   `:618 _do_async_transfer` use a bounded submit/poll on Mooncake's batch transfer; if a completion is
   silently dropped (no `ret=-1`), the poll loop blocks until the orchestrator timeout. Worth adding a
   per-transfer watchdog/log to localize.

---

## Full config (exact)

Launch args file (not in repo): `/tmp/mtp_relaunch_args_0607b.txt`. Key values:

```
--trainer-nodes 4   --trainer-ep-dispatch deepep   --trainer-moe-implementation triton
--trainer-gradient-checkpointing-method recompute_full_layer   --trainer-fsdp-reduce-dtype fp32
--no-trainer-enable-packing   --sync-inference-method p2p
--prompt-len 512   --max-new-tokens 256   --prompts-per-step 32   --conf-threshold 0.3
--pipeline-chunk-size 4   --pipeline-prefetch-chunks 4   --pipeline-teacher-concurrency 2
--forward-backward-timeout 2400.0   --student-replicas 2 (TP=2)   --teacher-replicas 2 (TP=2)
```

YAML (untracked): `examples/server/opd_singleshot_mtp_qwen36/qwen3_6_35b_a3b_student_mtp_4node_ep8.yaml`
— `optimizer: muon`, `muon_lr: 1.0e-5`, `muon_momentum: 0.95`, `load_weights_mode: all_ranks`,
`load_checkpoint_path: ""` (DCP disabled as a workaround; the DCP-Gloo fix is now live so canonical
32-rank DCP can be restored once sync works), `linear_replay_plan_max_packed_tokens: 16384`.

P2P / weight-sync env (trainer pod + `run.sh`, all correct/required):
```
XORL_P2P_HANDSHAKE_BASE_PORT=16400          XORL_P2P_CPU_POOL_MIN_BYTES=0   (Bug-7 fix)
XORL_WEIGHT_SYNC_BATCH_DENSE=1              XORL_WEIGHT_SYNC_BATCH_MOE=1
XORL_P2P_FP8_QUANTIZE_DEVICE=gpu            XORL_P2P_TRANSFER_RETRIES=50
XORL_P2P_PENDING_TRANSFER_TIMEOUT_S=120     XORL_P2P_CPU_SCRATCH_POOL_BYTES=2147483648
XORL_WEIGHT_SYNC_BUCKET_BYTES=1073741824    XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES=134217728
XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES=1073741824   XORL_WEIGHT_SYNC_MASTER_ADDRESS=$(POD_IP)
# NOT set (correct): NCCL_IB_GID_INDEX, NCCL_IB_HCA  (would break Mooncake init)
```

---

## Code pointers

- `src/xorl/server/weight_sync/backends/p2p.py`
  - `:240 _DirectMooncakeTransferEngine` — wrapper; `:262–291` handshake-port pin (fix #4, OK);
    `:315 batch_transfer_sync` → `batch_transfer_sync_write`.
  - `:440 _run_sync_transfer_items` (retry loop, `XORL_P2P_TRANSFER_RETRIES`), `:471 _run_async_transfer_items`,
    `:558 _transfer_small_entries`, `:618 _do_async_transfer`, `:701` label `"batch_transfer_sync"`.
  - `:1545 register_memory` (CPU scratch pool), `:1560 _ensure_transfer_executor`,
    `:1596 flush_pending_transfers`, `:1745 transfer_bucket` (**ships one `batch_transfer_sync` per session**),
    `:2237 supports_direct_ep_transfer`.
- `src/xorl/server/weight_sync/handler.py` — direct-EP sender path; key log lines seen:
  `:748` WeightSync banner, `:1121` "P2P Mooncake trainer binding", `:1154` "P2P direct-EP sender ranks",
  `:1351` "Skipping EP MoE tensor materialization on non-sender", `:1394` "Processing PP stage".
- Receiver side is in `xorl-sglang-internal` (`python/sglang/srt/...`, `[P2P] receiver register`).

---

## Reproduce / relaunch (ops)

Control plane = `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py` writing per-role
`run.sh`+`desired.sha256` under `/shared/opd-control/er-opd-q36-mtp-ss-0605c/`; bare pods run controller
daemons that re-exec on hash change (`write_control` stamps a fresh `revision` timestamp → hash always
changes → forced restart). Pods run **this worktree** via `XORL_REPO`+`PYTHONPATH`, so code edits deploy
without a commit.

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python ; G=experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
A="$(cat /tmp/mtp_relaunch_args_0607b.txt)"

# Full recovery recipe (REQUIRED before any sync retry — samplers wedge in paused-gen across crashes):
$PY $G stop-trainer-control $A                  # stop trainer first (no respawn)
$PY $G write-student-inference-control $A        # recreate dispatch + sglang-0 + sglang-1 (fresh Mooncake)
#   wait until BOTH samplers log "server is fired up" and dispatch logs both backends "ready after N polls"
$PY $G write-trainer-control $A                  # then start trainer

# Status / logs:
$PY $G status
H=/shared/opd-control/er-opd-q36-mtp-ss-0605c
cat $H/<role>/status                             # running_since / pid / current log path
```

**Watch HEAD + ALL WORKER logs** for the sync (a head-only watcher misses worker-side wedges). Key
signatures: `sync_inference_weights returned status=200 success=True` (✅ loop closed),
`ret=-1` / `batch_transfer` / `pending transfer` (retry path), silent hang → 1800s `cleanup rc=`.

---

## UPDATE — Run 3 LIVE RESULT (2026-06-08 16:08, `155749Z`): Fix 1 works, but layer-0 hang PERSISTS as a separate bug

Ran the 3-fix build live on 4 nodes. **Fix 1 (cold-prepare receiver re-arm) is empirically confirmed
working** — sglang-0 receiver log:
```
16:08:35 [P2P tp_rank=0/1] invalidating receiver warm cache because prepare requested p2p_invalidate_cache=True
16:08:36 [P2P] receiver register mode=allocator: 206 CUDA allocator segment regions   ← re-armed fresh
16:08:40 POST /prepare_weights_update 200 OK
```
So the dropped cold-prepare invalidation was a real bug and is fixed (receiver invalidates + re-registers).

**BUT the sync still wedged at the identical point** — `model.layers.0`. Sender `server.log`:
```
16:08:44 Rank 0: [WeightSync timing] (root): broadcast=1062ms total=1150ms          ← lm_head (1 dense param, 1017 MB) DONE ✓
16:08:44 Rank 0: Module model.layers.0: 16 params → Broadcasting 16 params, 74.8 MB
   ...    SILENCE 8.5+ min against the freshly re-armed receiver → 1800s timeout
```

**Decisive new facts:**
1. **The receiver re-armed AND the dense `lm_head` write completed (1.15s)** — yet the 74.8 MB layer-0
   transfer (16 small linear-attn params: `A_log`/`dt_bias`/`conv1d`/norms) still hangs. So cold-prepare
   was **not** the cause of the layer-0 hang; it is a **distinct, still-open bug** in the small/linear-attn
   param transfer.
2. **Neither Fix 2 (`prior_future.result`, 120s) nor Fix 3 (`_wait_all_pending`, 120s) fired** (8.5 min, no
   timeout). → The hang is **not in the async-drain path** those bound — it's in a **synchronous transfer
   call or pre-submission build** for the small params (e.g. `_run_sync_transfer_items` :440 whose retry
   loop only re-fires if `batch_transfer_sync` *returns*; a C++ `batch_transfer_sync_write` that never
   returns blocks forever with no bound). 
3. The dense vs small split is the crux: **1017 MB dense `lm_head` succeeds; 74.8 MB of small linear-attn
   params hangs.** The earlier ruling-out of the small-param/`_slice_qwen_linear_attention_fused_param`
   path was premature — this is where it hangs, on a re-armed receiver.

**Next leads:** (a) determine sync-vs-async path for the layer-0 small params and bound the synchronous
`batch_transfer_sync_write` (a C++ call that never returns needs a wall-clock watchdog, not a future
timeout); (b) verify the 206 receiver-registered allocator segments actually *cover* the destination
addresses for the small linear-attn params (an RDMA write to an unregistered remote region posts but never
completes — the classic silent hang); (c) the fast bisect still applies — force layer 0 down the same
dense/broadcast path `lm_head` uses (which works) to confirm the small-param path is the trigger.

---

## Fallback to unblock training tonight (if p2p can't be fixed quickly)

`--sync-inference-method nccl_broadcast` is the default, **Mooncake-free** backend (rank-0 NCCL broadcast
→ sglang `update_weights_from_distributed`). **Its sync has never actually been tested** on this stack —
the one nccl_broadcast attempt died earlier on the *initial greedy rollout* (900s ReadTimeout) because the
samplers were wedged at the time; the samplers are now fresh, so the nccl_broadcast **sync itself is
untested and may work**. **The user explicitly prefers p2p and rejected nccl_broadcast**, so treat this
only as a debugging A/B (does *any* backend close the loop?) or a last-resort to keep training moving while
p2p is root-caused — not as the production choice. To A/B: edit the args file
`--sync-inference-method nccl_broadcast`, run the full recovery recipe above.

---

## Related

- `docs/notes/merge_opd_distributed_init_regressions_handoff.md` — the 5 init regressions (RESOLVED).
- Memory: `project_apanda_dev_merge_opd_init_regressions`, `project_p2p_2node_tp8_findings`,
  `project_opd_inference_stack_restart_recipe`, `project_opd_p2p_sync_wedge_collective_desync`,
  `feedback_p2p_cpu_pool_min_bytes_bug7`, `feedback_opd_no_skip_cached_prepare`,
  `feedback_mooncake_requires_privileged` (→ use rdma/infiniband + IPC_LOCK, not privileged).
- Canonical OPD setup: `xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch/CANONICAL_RUNBOOK.md` §9 (P2P env).
```
