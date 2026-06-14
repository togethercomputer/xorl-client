# Production-scale SingleShot-MTP OPD run: launch handoff

**Audience:** the agent kicking off (and babysitting) the long coderforge MTP-OPD training
run on stack `er-opd-q36-mtp-ss-0605c`.
**Code line:** `apanda-dev-mtp` (this worktree, @ `58ab72bb` or later). All CP/perf fixes,
the FSDP flush machinery, metric canonicalization, and the determinism gate are on this
branch; the loss path is byte-identical to what produced the oracles below.
**Companion docs:** `mtp_opd_mfu_profiling_handoff.md` §-1 (perf/correctness ledger),
`gdn_cp_handoff.md` (CP work — NOT needed for this run), the OPD canonical runbook
(ops rules), `sgl_sampler_throughput_handoff.md` (sampler side, RESOLVED).

---

## 1. The config to launch

`experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt` is the promoted
production config — use it verbatim:

- 4 trainer nodes / 32 GPUs: **EP=32 (deepep, SMS24) × eFSDP=1, CP=1, dp_shard=32**,
  `batch_parallel_mode: dp_shard`
- `gradient_checkpointing_method: no_recompute`, fp32 grad-reduce, packing OFF, quack MoE
- `--trainer-fsdp-defer-grad-sync` (+ reshard deferral) — fine at this 1536-token shape;
  **do NOT carry these to longer-context configs** (defer_reshard keeps all 35B params
  gathered ⇒ OOM at 4k+; see main handoff §-1.16)
- `--opd-async-sample-overlap`, pipeline chunk 32, conf 0.3, static-padded 1536
  (prompt 512 + 256 gen), 32 prompts/step, p2p weight sync
- checkpoints: every 10 steps, keep 1, save-best on val_loss (min, min_step 100),
  eval split per the args file

Expected steady-state at this config (validated 2026-06-09/10):
- FB ≈ **12.6s** for ~9.6k unique supervised tokens (1.31 ms/token, ~23.8 tok/s/GPU)
- sampling ≈ 2-3s at 2 student replicas (post-SGL-fix ~2800 tok/s agg) → **step ≈ FB-bound
  ~13s** + weight-sync ~15.8s on sync steps
- MFU sanity (logged per FB as `[singleshot-mtp-profile] flops`): actual ≈ 0.43%,
  useful ≈ 0.11%, regret ≈ 4.06 vs 989 TFLOPS/GPU promised
- loss oracle sanity on the replay payload: 1.0598 (this exact stack/kernels)

## 2. Pre-flight checklist (in order)

1. **The trainer control is currently a REPLAY config** (p8k CP1 cell, skip-optim,
   `OPD_FB_REQUEST_REPLAY_PATH` baked). You MUST rewrite it:
   `python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
   write-trainer-control $(cat experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)`
   with a CLEAN env (no `OPD_FB_REQUEST_REPLAY_PATH`, no `OPD_SAVE_FB_REQUEST_DIR`, no
   `XORL_CP_STATEFUL_DEBUG_SYNC`). Optionally keep `XORL_TRAINER_DETERMINISTIC=1` if you
   want bitwise-reproducible steps (small perf cost; default off for prod).
2. **Check AGENT_NOTES.md first** (`/shared/opd-control/er-opd-q36-mtp-ss-0605c/AGENT_NOTES.md`)
   and add a line announcing the launch — protocol is: check newest trainer-head run log
   for an unfinished run before ANY control write (a write kills in-flight runs).
3. **Trainer nodes**: 4 nodes were healthy as of 2026-06-10 (trainer-worker-2 was
   recreated once — bare pods in Error do NOT restart; `kubectl get pods` first, delete +
   re-apply from `render-manifest` if any are Error). Boot-rendezvous can wedge after
   rapid control churn — symptom is "Waiting for rank 0 ready" with workers stalled after
   CUDA selection; recover with stop-trainer-control + rewrite.
4. **Samplers/teachers**: leave them alone unless dead — they run the SGL agent's tuned
   revision (snapshot repo via OPD_SGLANG_REPO, cuda-graph + q-banding + static k-list,
   `--max-running-requests 32`). A bare `write-student-inference-control` from the
   default args would DOWNGRADE them (lose the snapshot/banding flags) — coordinate via
   AGENT_NOTES with the SGL agent if a sampler change is needed. Revive a dead sampler by
   revision bump only with their flag set preserved.
5. **Weight sync env** (already baked in the launcher): `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu`,
   `XORL_WEIGHT_SYNC_BATCH_DENSE=1`, `XORL_P2P_CPU_POOL_MIN_BYTES=0`, handshake base port
   pin. NEVER skip the cached `/prepare_weights_update` (it re-arms Mooncake buffers).
   Trainer pods must NOT set `NCCL_IB_GID_INDEX`/`NCCL_IB_HCA` (breaks Mooncake init).
6. **supervisor.pause** is in place on the control root — remove it only when you are
   ready for the supervisor to manage relaunches (resume-capable supervisor is the OTHER
   agent's work on `apanda-dev-mtp-resume`, deployed via `OPD_XORL_REPO` override; Test A
   validated, Test B pending — coordinate before relying on auto-resume).
7. **Eval cadence rule** (canonical runbook): `eval_control_start_step` must be divisible
   by `eval_every` AND equal `num_steps − 1` (`num_steps ≡ 1 mod 5`) or the loop stalls.

## 3. First-hour watch items (specific, each has bitten before)

- **Step-1 memory**: `no_recompute` was validated under `--skip-optim-step`; the first
  long run adds AdamW state. Per-rank activations are one ~1536-token micro-batch so it
  should hold, but watch `nvidia-smi`/OOM on step 1-2. Fallback if it OOMs:
  `recompute_before_dispatch` (costs ~2x FB — acceptable, still ~17x over legacy).
- **Watch head + ALL worker logs** (a head-only watcher once missed a 14h-dead run).
  Worker slot logs: `/shared/opd-control/<stack>/trainer-worker-{1..3}/logs/`.
- **P2P sync wedge**: if `batch_transfer_sync ... ret=-1` retries appear, the fix for the
  collective desync is in-tree, but the operational cause is stale Mooncake state —
  recreate samplers+dispatch+trainer together (restart recipe memory), keep teachers warm.
- **First checkpoint** (step 10): confirm DCP save completes within `checkpoint_timeout`
  and the keep-latest pruning works before walking away.
- **Throughput regression check**: steady FB should be ≤ ~13s. If it's ~16s+, suspect the
  no_recompute fix changed memory/recompute behavior post-merge — the quick A/B is one
  replay cell against `fb_capture_b32_v1` (expect ≤ 1.31 ms/unique-valid; the fix that
  made no_recompute REAL landed after the 1.31 measurement, so faster is plausible,
  slower is a red flag).

## 4. Known-good invariants and red flags

- Loss at step 0 on coderforge should land ~1.0-1.1 (the replay oracle band); NaN/0.0 or
  >2 ⇒ stop and check sampler chat-template + `input_token_ids` fail-loud paths (PTC-118
  class).
- `opd_singleshot_mtp_enabled=1.0` must appear in step results — its absence means the
  replay prep silently didn't run.
- `opd_num_teachers:max` and `opd_profile_*` metrics flow through the `:max`/`:sum_max`
  reduction suffixes — the client-side `_metric_value` fix is in xorl-client-internal
  PR #2 (4b58a2d); make sure the client in use has it or dashboards silently read 0.0.
- Batched-decode + think-open chat-rendering sampler bugs are fixed in the running
  sampler revision — do not roll samplers back past 2026-06-10 ~06:32 rev.

## 5. What's intentionally NOT in this run

- **CP**: correct at all widths but not a throughput win at this shape (main handoff
  §-1.16); config stays CP=1. The CP work continues separately per `gdn_cp_handoff.md`.
- **Packing**: measured negative at this shape (b64_packed cell, 3.61 ms/token).
- **bf16 grad-reduce**: measured negative twice; stays fp32.
- **Sampler scale-up 2→4 replicas (Phase 2)** and supervisor kill-recover (Test B): the
  resume agent's queue; current 2 replicas already keep steps FB-bound.

## 6. If throughput work resumes mid-run

The next real lever is **prefix-sharing N continuations per prompt** (amortizes the ~⅔ of
FLOPs spent on context; attacks regret 4.06 directly) — design notes in the main handoff.
Topology is done: EP32+dp_shard is the measured optimum on this hardware; don't reopen
EP width or CP for this shape without new evidence.
