# Handoff — pipeline_rl + p2p weight sync fails to init on Qwen3-235B GRPO (2026-06-29)

Audience: a new agent picking up the pipeline_rl throughput work. Self-contained. The K3 work is
done (see `K3_GRPO_235B_HANDOFF_20260628.md`); this is purely about enabling `pipeline_rl=true` for
the ~40% step-time win without breaking the p2p weight sync.

---

## ✅ ROOT CAUSE + FIX (2026-06-29 ~07:1x) — RESOLVED: it was a CUDA-12/13 lib mismatch, NOT async/pipeline

**The swallowed exception was already in `server.log` (no instrumentation needed).** Grepping the
failing run's engine log (`RUN_DIR/server.log`, the `20260629T064928Z` serial run) showed:

```
[P2P] mooncake-transfer-engine is not installed. ... fall back to sync_inference_method='nccl_broadcast'.
[P2P] underlying ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
[P2P] rank 0 reported initialize() failure
[P2P] direct-EP initialize failed on ranks [0,1,2,3,4,5,6,7]
```

The handler swallows it: `handler.py:1267 if not backend.initialize(): return {"success": False,
"message": "Failed to initialize p2p backend"}` (the boolean is the only thing that bubbles to
`sync_result.message`). But the backend itself DID log the real reason — at
`p2p.py:_make_local_engine` (`from mooncake.engine import TransferEngine` → ImportError).

**Root cause (and the real regression):** the current stack runs under the
`xorl-qwen-k3-reconciliation/.venv-cu132-latest-probe` venv, which is a **CUDA 13** env
(`nvidia/cu13/lib/libcudart.so.13`; `torch 2.12.1+cu132`). But the installed **mooncake wheel's
`engine.so` was built for CUDA 12** — its rpath is `$ORIGIN:$ORIGIN/../mooncake_transfer_engine.libs`
and it needs `libcudart.so.12`, which does NOT exist anywhere in this venv or on the node (only
`libcudart.so.13` is present). So `import mooncake.engine` raises ImportError → p2p backend init
returns False → "Failed to initialize p2p backend" at step 1. **This is exactly the
`XORL_REPO=xorl-qwen-k3-reconciliation` regression hypothesized at the top of this doc:** the OLD
"serial works (470 GB)" config ran under `xorl-apanda-dev-opd-port/.venv`, which is a **CUDA-12** venv
(`nvidia/cuda_runtime/lib/libcudart.so.12`) where mooncake imported fine. It was NEVER an async /
blocking-`.result()` / pinned-pool / pipeline issue — those §4 hypotheses are all wrong. It fails
identically in serial and pipeline because mooncake simply can't import in this venv.

**The fix (verified):** add a cu12-only shim dir to `LD_LIBRARY_PATH` so `libcudart.so.12` resolves
for mooncake WITHOUT shadowing torch's cu13 libs.
- Shim dir: `/home/apanda/xorl-qwen-k3-reconciliation/.mooncake-cu12-shim/` containing ONLY a copy of
  `libcudart.so.12` (copied from the opd-port cu12 venv). It is a real file copy (not a symlink), so it
  survives even if the opd-port venv is removed. It lives on the shared `/home/apanda` NFS, so all 8
  trainer nodes see it.
- run.sh change (head + all 7 workers): append it at the **END** of `LD_LIBRARY_PATH`:
  `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/apanda/xorl-qwen-k3-reconciliation/.mooncake-cu12-shim"`
  Appending at the end + the dir containing only `libcudart.so.12` (a soname absent from every cu13 dir)
  guarantees it cannot shadow any lib torch uses.
- **Verified in-pod:** `LD_LIBRARY_PATH=...:<shim> python -c "import torch; from mooncake.engine import
  TransferEngine"` → `torch 2.12.1+cu132 cuda 13.2 avail True` AND `mooncake OK alongside torch`, on
  BOTH the head pod and a worker pod (NFS-shared shim confirmed visible everywhere).

**PRIMARY durable fix (deployed, env-independent — survives ANY run.sh rewrite):** drop a copy of
`libcudart.so.12` directly into mooncake's own rpath dir:
`<cu132-venv>/lib/python3.12/site-packages/mooncake_transfer_engine.libs/libcudart.so.12`.
`engine.so`'s rpath is `$ORIGIN:$ORIGIN/../mooncake_transfer_engine.libs`, so it resolves the lib from
there with NO `LD_LIBRARY_PATH` needed. Verified: `from mooncake.engine import TransferEngine; import
torch` → `rpath-only OK; torch cuda 13.2 True` on both head and worker pods (NFS-shared, all 8 nodes
covered). This is the fix that makes the GRPO run resilient regardless of which run.sh the sweep/agent
leaves behind. The `.mooncake-cu12-shim` + LD_LIBRARY_PATH export in run.sh is kept as a redundant
belt-and-suspenders backup.

**Even more permanent (for whoever rebuilds the venv):** `pip install nvidia-cuda-runtime-cu12` into the
cu132 venv (adds `libcudart.so.12` under site-packages/nvidia), or install a mooncake wheel built
against CUDA 13. The rpath copy is the zero-rebuild fix deployed now.

### ⚠️ Scheduling collision (2026-06-29 ~07:07) — a concurrent EP-dispatch K3 sweep seized the slots
Right as I was relaunching GRPO with the fix, another agent launched
`experiments/k3_tests/run_q235_ep_dispatch_sweep.py --matrix 8:alltoall,8:deepep` (pid 2737637). That
driver **rewrites all 8 slot `run.sh` files** per variant (static-K3 forward/backward replay; NOT
weight-sync), holds the 8 trainer nodes, and `stop_slots` between variants. It backs up the original
run.sh into `.../k3_ep_dispatch_sweep/20260629T070732Z/control_backups/` (those backups DO contain my
shim fix) but **never restores them**. To not stomp the other agent's live experiment, I let the sweep
finish, then restore the GRPO run.sh from `control_backups/` and do the sequenced relaunch. See the
"Current state" section at the bottom for live status.

---

## ⚠️ UPDATE (2026-06-29, later): NOT pipeline-specific — SERIAL fails too

A serial relaunch (`pipeline_rl=false`, **clean** group) ALSO died at step 1 with `Weight sync failed:
Failed to initialize p2p backend` (plain, not "Pipeline"; 260 s). So the p2p backend init is broken in
**both serial and pipeline** under the CURRENT stack config (`XORL_REPO=/home/apanda/xorl-qwen-k3-reconciliation`
+ the sibling's recent sampler/handler changes: `--enable-return-expert-logits`,
`SGLANG_RETURN_ORIGINAL_LOGPROB=1`, etc.). The "serial works (470 GB)" below was the EARLIER config
(opd-port engine, fewer sampler flags) — today's serial step-1 sync was never actually exercised until
now (the validation run exited at step 0). **So the real bug is the p2p TransferEngine init in the
current config, regardless of pipeline_rl.** First thing to bisect: did the sibling's weight_sync/handler
or sampler-flag changes regress the p2p init? Compare the current `xorl-qwen-k3-reconciliation` handler
+ sampler flags against the older opd-port config that synced 470 GB. The §2-§4 pipeline analysis below
still applies, but treat the init failure as config-wide, not pipeline-only.

## 0. TL;DR

- **Goal:** run the 235B GRPO filler run with `pipeline_rl=true` (overlap the next batch's decode with
  the current fwd/bwd → ~2.5 min/step vs ~4.5 serial, ~40% faster).
- **Blocker:** in pipeline mode, the post-training weight sync fails at **step 1** with
  `Pipeline weight sync failed: Failed to initialize p2p backend`. The serial p2p sync works fine
  (470 GB @ 37 GB/s). Reproduced 3×.
- **Ruled out:** (a) **engine-busy** — added `/pause_generation` (retract) before the sync; the sampler
  logs confirm the pause fires (200 OK) yet the sync still fails instantly. (b) **stale p2p group** —
  cleared `weight_sync_group` via `/complete_weights_update` so the sampler was verified clean, and it
  STILL fails. So it's neither the pause nor the stale group.
- **Where it actually fails:** **trainer-side**, before contacting the sampler (sampler log shows
  NOTHING at the failure timestamp). The p2p backend init is a mooncake **TransferEngine handshake +
  ~8.6 GB pinned-pool registration** (`server/weight_sync/handler.py` ~L1195-1300) that succeeds in
  serial (`await`, idle) but fails in pipeline (blocking `.result()` while the generation thread runs).
- **Current state:** fell back to **serial** (`pipeline_rl=false`) so the overnight run is banked
  (works, best k3 2.3e-4, ~12 h). pipeline_rl is the open item.

---

## 1. Repro / config

- Stack: `er-opd-q235-fillerrft-slots` (ns `apanda`). Trainer 8-node EP8 (`XORL_REPO=/home/apanda/xorl-qwen-k3-reconciliation`,
  has the K3 packing fix); sampler 1-node TP8 (full recon recipe + `--enable-return-routed-experts`);
  rollouts **direct to the sampler** (`inference_base_urls=…-sglang-0:30060`, dispatch bypassed).
- Client: `/home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py`.
  Flip in `/shared/opd-control/er-opd-q235-fillerrft-slots/trainer-head/run.sh`: `pipeline_rl=true`.
- To (re)launch: edit the head run.sh (one atomic edit = one head re-exec), wait ~55 s for its rdzv
  store, then append a marker to each `trainer-worker-{1..7}/run.sh` (sequenced; static torchrun
  rendezvous on head `:29610` — verify it's free first). Engine logs → `RUN_DIR/server.log`; client
  logs → the head run.sh log; metrics → `RUN_DIR/grpo/*/metrics.jsonl`.

The failure (head log):
```text
[~420s] Step 1: Pipeline weight sync failed: Failed to initialize p2p backend
```
Note: step 0's sync is skipped (the original cold-EP-MoE deadlock fix `elif config.use_full_weights:`),
so **step 1 is the FIRST real p2p sync** in both serial and pipeline. Serial step-1 succeeds; pipeline
step-1 fails.

---

## 2. What I tried (all in `filler_tokens_rl.py`)

The pipeline sync block (the `if config.pipeline_rl and generation_queue is not None:` branch, ~L3000,
note line 2988's comment "use blocking `.result()` to avoid cross-event-loop deadlock; generation runs
in a separate thread"):

1. **Pause/continue around the sync (client-level, per owner's correct instinct).** Added helpers
   `_pause_inference(config, mode="retract")` + `_continue_inference(config)` (stdlib `urllib` POST to
   `get_inference_urls(config)` → `/pause_generation` {mode:retract} and `/continue_generation`).
   Wrapped: `_pause_inference()` → `sync_weights_to_inference(p2p).result()` → `finally _continue_inference()`.
   **Result:** sampler logs `POST /pause_generation 200` + `/continue_generation 200` (pause works), but
   the sync still fails instantly. ⇒ engine-busy is NOT the cause.
2. **Clear the stale `weight_sync_group`** before retry (the sibling's fix, no SGLang restart):
   `POST …:30060/complete_weights_update -d '{"group_name":"weight_sync_group","flush_cache":false,"transport":"p2p","run_post_process_weights":false}'`.
   Verified the sampler returned "No P2P weight update in progress" (clean) + `gen=200`, then re-ran.
   **Result:** still `Failed to initialize p2p backend` at step 1. ⇒ pre-existing stale group is NOT
   the cause (it's a *secondary cascade*: each failed sync leaves a group → next sync hits "already in
   progress"; the clear breaks the cascade but not the root failure).

The pause + clear are both still worth KEEPING (pause avoids weight read/write races during the
transfer; clear self-heals the cascade) — they're in the client now but commented as the pipeline path.

---

## 3. The actual failure (where to look)

- Sampler log shows **nothing** at the failure timestamp → the trainer never reaches the sampler. So
  `Failed to initialize p2p backend` is raised **trainer-side**, inside `sync_weights_to_inference` →
  the weight-sync handler's p2p backend init.
- Trainer p2p init: `src/xorl/server/weight_sync/handler.py` ~L1195-1300 — module-level cache
  `_cached_p2p_backend` (key = sync_method, endpoints, group_name, master addr/port, buffer_size,
  world_size, rank, backend cfg). On a key match + `is_alive` it reuses; else it **destroys the old and
  creates a new backend** — "TransferEngine handshake + ~8.6 GB CPU pinned pool registration." That
  create is what fails in pipeline. **It never succeeds once in pipeline, so the cache never helps.**
- The exact `Failed to initialize p2p backend` string is NOT in the `.py` grep (likely in the mooncake
  TransferEngine wrapper / a backend module, or constructed). **First action: instrument the create
  path to log the real exception** — the message is swallowed into `sync_result.message`. Find where
  `sync_result.success=False, message="Failed to initialize p2p backend"` originates (training_client
  → handler → backend create) and log the underlying error.

---

## 4. Hypotheses (ranked) + suggested fixes

1. **Blocking `.result()` context breaks the async TransferEngine init.** Serial `await`s the sync in
   the event loop; pipeline calls `.result()` (blocking) from the main thread while generation runs in
   another thread. If the TransferEngine handshake needs the event loop / an async step, the blocking
   call starves it → init fails. **Test:** instrument the init exception (§3); if it's an event-loop /
   "no running loop" error, run the pipeline sync on the loop (or do the FIRST init on the loop).
2. **Pre-warm the p2p backend at step 0 (most promising fix).** Initialize + cache the backend ONCE
   while the system is idle (no overlap), so step 1+ pipeline syncs **reuse the cached backend** (the
   `_cached_p2p_backend` path, no re-create). Options: (a) make the FIRST pipeline sync non-overlapped
   (await, sampler idle) then overlap from step 2; (b) a one-shot warmup sync at startup; (c) check
   `GRPO_FORCE_COLD_FULL_WEIGHT_SYNC` (default false) — but cold step-0 sync historically deadlocked on
   EP-MoE materialization, so warm it AFTER step-0 fwd/bwd, not before.
3. **Host-memory / pinned-pool contention.** The 8.6 GB pinned-pool registration during pipeline (the
   generation thread + overlap active) may fail to register. **Test:** the instrumented exception (§3)
   will show if it's a pinned-alloc / RDMA-register failure; if so, pre-warm (#2) sidesteps it (register
   once when idle).

The pause(retract)+continue and the stale-group clear should stay (resilience), but they are NOT the
fix — the fix is making the FIRST p2p backend create succeed (or be reused) in the pipeline context.

---

## 5. Resilience note (for the overnight serial run too)

Stale `weight_sync_group` forms whenever a sync is interrupted, and the NEXT sync then fails with
"already in progress" → the run dies and stays dead. Consider adding a **self-healing clear** before
each sync (POST `/complete_weights_update` — no-op "No update in progress" if clean) so an unattended
run can recover from a single hiccup. Not yet added.

---

## 6. Current state

- Serial overnight run relaunching now (`pipeline_rl=false`, N=2500 ~156 steps ~12 h, save_every=50,
  save_final=true, wandb `grpo-235b-filler-fullrun-plrl-20260629`), best k3 ~2.3e-4. Sampler clean.
- pipeline_rl step 0 always runs fine (k3 ~0.00042, overlap visible: "Sampling batch 1" + "batch 2"
  concurrently) — only the step-1 sync init fails. So the overlap machinery itself works; only the
  p2p-init-under-pipeline is broken.
- Scratchpad watchers/patterns: `/tmp/claude-0/-home-apanda-xorl-opd-prefill/08702b41-075c-4bf2-8502-fd28f7e7cc4d/scratchpad/k3_fullrun.sh`
  (re-trigger workers + watch steps/k3/overlap). Sibling p2p notes:
  `xorl-qwen-k3-reconciliation/docs/notes/k3_235b_live_investigation_2026_06_28.md` (~L866 documents the
  stale-group clear). Memory: `wordle-live-k3-floor.md`, `reprogrammable-slots-235b-launch.md`.
