# apanda-dev merge ADOPTED on the OPD stack + 2×TP=2 SMG samplers — handoff (2026-06-09)

**One-line state:** `origin/apanda-dev` is merged into `codex/opd-port-20260602` and **adopted live** (the running stack now runs the merged code); the student samplers were re-architected from 1×TP=8 to **2×TP=2 behind SMG**; every merge-specific surface is validated (engine init, dual-endpoint registration, **2-endpoint p2p weight sync**, and the re-ported all-layer-OPRD forward_backward executes). **The only open item is a compute-cost blocker, not a merge bug:** the OPRD **K=C** buffer-arm (`PTC-302B`) fb chunk exceeds the 1800s operation timeout, so that specific config can't make progress as-is. The no-filler control (`PTC-302N`) completed earlier and is unaffected.

Supersedes `MERGE_APANDA_DEV_NOTES_2026_06_07.md` (the untested 2026-06-07 attempt). Memory: `project_apanda_dev_merge_adopted_opd_2xtp2`.

---

## 0. Current stack state (as of 2026-06-09 ~04:20 UTC)

- Stack `er-opd-q36-35b-slots`, ns `apanda`. Live worktree `/home/apanda/xorl-apanda-dev-opd-port` is **fast-forwarded to the merged code** (branch `codex/opd-port-20260602`, tip = merge + 3 fix commits).
- **Samplers:** `sglang-0`, `sglang-1` — **TP=2 each, non-privileged + rdma/infiniband**, co-located on one `nccl` node (h100-096), warm, serving `Qwen3.5-35B-A3B`. SMG `dispatch` round_robin over both.
- **Teachers:** `teacher-sglang-0/1` + `teacher-smg` — warm (untouched throughout).
- **Trainer:** STOPPED (`stop-trainer-control --remove-run` was issued to end the OPRD timeout-retry loop). Relaunch with `write-trainer-control` once the next config is chosen.
- **Autopilot supervisor:** STOPPED (was idle; stopped to avoid interference during the cutover). Restart only if you want the autoresearch queue driven again.

## 1. What the merge did

`origin/apanda-dev` advanced 20 commits past the 2026-06-07 base and now contains the **upstreamed** OPD distributed-init fixes (`9b5ec4e8`) + p2p cold-sync fixes (`29024521`) + the handshake-port pin + FP8 stack + GLM-5 + DSv4 + the slime-parity `rl/` module. The MTP stack (`/home/apanda/xorl-mtp-singleshot-port-20260602`) runs this exact commit and validates 2×TP=2 p2p sync, so the merged sync subsystem is trusted.

**Resolution philosophy (the key decision):**
- **OPD-loss subsystem = OURS.** Upstream gutted `opd_loss.py` (512→214 lines) into a new `rl/` module the research client doesn't use. Took ours for `opd_loss.py`, `compiled_cross_entropy.py`, `opd_streaming_kl.py` (the cohesive set `opd_loss` imports). `loss_output`/`reducers` are identical ours==theirs; `per_token_ce` diverges but is NOT in the `opd_loss` import graph.
- **Server infra = THEIRS.** `p2p.py`, `handler.py`, `inference_endpoints.py`, `launcher.py`, `runner_dispatcher.py`, tests → upstream (byte-identical to the MTP-proven code). Multi-endpoint sync fans out to all receivers; `XORL_SERIAL_INFERENCE_ENDPOINT_SYNC` is a fallback.
- **`model_runner.py`** — ours for the 8 OPD-loss-method conflicts (OPRD teacher-forward, hidden-match, diagnostics, full metric set + `denom = max(valid_count,1)`); theirs for all infra. The forward-call conflict was **merged**: request `output_hidden_states` when EITHER OPRD capture or upstream's `diagnostic_hidden_states` is active.
- **`qwen3_5_moe` modeling** — kept OUR per-layer-output `hidden_states` convention (40 entries, index i = layer i output, pre-final-norm) since OPRD is the sole consumer; removed the auto-merged duplicate before-append + final-norm-append.

Validated: py_compile, ruff(E9/F8) clean, all server/loss/model modules import, `loss_fn_params` flows free-form to the OPRD params, API route set identical to baf6dcce.

Branches: merge prepared in isolated worktree `/home/apanda/xorl-opd-merge` (`merge-apanda-dev-opd-20260609` = `d5aaf58d` + `1b9bc6fe`), then the live branch was `git merge --ff-only`'d to it + 1 more fix commit.

## 2. Three runtime breaks hit at step 0 (all fixed)

These are the tail of the "OPD-loss subsystem diverged" issue — upstream changed loss helpers my `opd_loss` depends on:

1. **`moe_implementation: quack` → `triton`** in `configs/qwen3_5_35b_a3b_opd_opdb_8node.yaml`. apanda-dev's quack dropped EP support (`ValueError: moe_implementation='quack' does not support Expert Parallelism`). 302N ran on the *old* code where quack+EP worked. Fix = triton (canonical runbook O9 #19; the MTP stack uses triton too).
2. **`opd_streaming_kl.py`** — upstream dropped the `if vocab_chunk_size is None` guard → `TypeError: '<=' not supported between NoneType and int`. Restored ours.
3. **`compiled_cross_entropy.py`** — upstream removed the forward-KL/diagnostic functions (moved to `rl/`) → ImportError for `compiled_forward_kl_full_function`. Restored ours.

## 3. The 2×TP=2 SMG generator port

The OPD slots generator (`experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py`, untracked) hardcoded **1× TP=8** samplers (privileged, `CUDA_VISIBLE_DEVICES=0..7`, 8-GPU mooncake map). Ported the MTP generator's pattern:
- `SAMPLER_GPUS = 2`; `sampler_script` now uses `select_assigned_cuda_devices 2` + runtime `mooncake_ib_device_json()` + `--tp-size 2` (dropped the hardcoded CVD + 8-GPU map).
- `pod()` gained a `privileged: bool` param → sampler pods render **non-privileged + `securityContext.capabilities.add: [IPC_LOCK]` + `rdma/infiniband: 1`**, `gpu_count=2`, no hardcoded CVD. (All `nccl` nodes advertise `rdma/infiniband`, confirmed.) This is the CLAUDE.md-correct pattern and lets two TP=2 samplers co-locate on one node with **distinct device-plugin GPUs** (no collision — verified: distinct UUIDs).
- SMG `dispatch` auto-uses `round_robin` for >1 endpoint.

**Sampler memory tuning for TP=2:** 35B at TP=2 = ~35 GB weights/GPU; OOM'd at `mem-fraction-static 0.8`. Tuned to **`mem-fraction-static 0.85`, `max-running-requests 256`, `max-total-tokens 16384`, `cuda-graph-max-bs 64`** (light sampling doesn't need the TP=8-era footprint). Both samplers came up clean after this.

## 4. The cutover sequence (what was done, in order)

1. Stopped the idle autopilot supervisor (`pkill -f autopilot_supervisor.py`).
2. `PTC-302N` (no-filler control) **completed on its own** at 64×81 → `eval/accuracy ≈ 0.359` (W&B `im1f07mj`) — preserved, not killed. (Still under-trained vs the 128×101 plateau the HANDOFF wants.)
3. Created candidates `PTC-302N2` (= 302N + serial sync) and `PTC-302B` (= PTC-302 buffer arm + serial sync + `opd_pipeline_rl: false` + 16×81 + `eval_control_start_step: 80`).
4. `git merge --ff-only merge-apanda-dev-opd-20260609` in the live worktree (untracked generator/config/candidates preserved).
5. `kubectl delete pod …-sglang-0` (old TP=8) → applied the 2×TP=2 manifest (new sglang-0 + sglang-1) → `write-student-inference-control --sampler-replicas 2 --sampler-layout dedicated`.
6. `write-trainer-control --candidate PTC-302B --num-steps 81 --prompts-per-step 16 --sampler-replicas 2 --sampler-layout dedicated`.

Then iterated through the three step-0 fixes (§2) + a rapid-relaunch hang (§5), each time relaunching the trainer (samplers stayed warm — host-side errors don't corrupt the CUDA context, so no sampler recreation needed).

## 5. The OPRD K=C timeout blocker (the open item)

After the three fixes, step 0 reached the OPRD forward_backward and the gang ran at **100% GPU** — but the fb chunk hit the **1800s operation timeout** exactly (`[TIMING] engine forward_backward: execute=1800.1168s`), then the client retried → futile timeout-retry loop (0 profile rows).

**Why:** `PTC-302B` uses `opd_buffer_equals_cot: true` (K=C — the pause buffer = the full CoT length, up to ~6000 tokens) + `opd_oprd_layers: every1` (all 40 layers) + `last_k 2000` at `prompts_per_step 16`. The packer makes **7 batches of ~8192-token sequences** (vs 302N's ~100-token samples), each needing a student forward (40-layer hidden capture) **plus a teacher 2nd no-grad forward** at 8192 tokens. At ~4 min/sample × 7 samples, one fb chunk exceeds 30 min → timeout. So the K=C all-layer arm at 16 prompts/step is effectively multi-day / infeasible as configured.

**Not a merge bug:** every merge-specific surface worked (engine init in 14 polls, both endpoints registered + synced in 18.6s, OPRD config active `hidden_match_coef=100.0`, the fb starts/packs/computes). It's the OPRD compute cost. (Unconfirmed whether the merge also added an OPRD per-step slowdown vs the old S11 16×41 runs — no S11 step-time baseline was captured; worth checking if pursuing the real arm.)

### Options to make the buffer arm run (pick one, then relaunch)
- **(A) Minimal OPRD validation first [recommended to close validation]:** `prompts_per_step=2, num_steps=6, opd_oprd_layers=every4 (10 layers), opd_oprd_last_k=512, request_timeout=3600`. ~10-15 min/step, ~1h. Confirms `opd_oprd_loss` is emitted end-to-end on merged code, then decide the real budget.
- **(B) Real K=C arm, slow:** keep K=C + every1, set `request_timeout=3600+`, `prompts_per_step=4`. Scientifically faithful (the HANDOFF's exact arm) but ~75 min/step → 81 steps is multi-day.
- **(C) Lighter buffer arm:** abandon K=C → fixed K (~256-512 pause tokens) + every4 + last_k=512 → tractable real run (minutes/step), but changes the science from "K=C all-layer."

(Knobs live in `experiments/opd_profile/autoresearch/candidates/PTC-302B.yaml`: `default_num_steps`, `default_prompts_per_step`, `request_timeout`, and under `client_args`: `opd_buffer_equals_cot`, `opd_oprd_layers`, `opd_oprd_last_k`. Keep the O2 rule: `eval_control_start_step == num_steps-1` and `% eval_accuracy_every(5)`.)

## 6. Operational reference

**Relaunch the trainer** (samplers/teachers stay warm):
```bash
cd /home/apanda/xorl-apanda-dev-opd-port
PY=.venv/bin/python ; G=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
$PY $G write-trainer-control --candidate experiments/opd_profile/autoresearch/candidates/<CAND>.yaml \
   --num-steps <N> --prompts-per-step <P> --sampler-replicas 2 --sampler-layout dedicated
```
**Recreate samplers** (only if wedged): `$PY $G write-student-inference-control --sampler-replicas 2 --sampler-layout dedicated`. To change sampler pod specs (TP/gpu), delete the sglang pods + re-apply `render-manifest --sampler-replicas 2 --sampler-layout dedicated`.

**Monitor:** head client log `/shared/opd-control/er-opd-q36-35b-slots/trainer-head/logs/<rev>-run.log` (step markers); server ranks `RUN_DIR/server.log` (fb internals, `[TIMING] … forward_backward`); profile rows `RUN_DIR/opd_profile.jsonl` (`eval/accuracy`, `opd_oprd_loss`). Watch head **+ all workers** + server.log.

**Gotchas hit this session:**
- **Rapid trainer relaunches (3×) → engine-init hang** 13 min at "Waiting for rank 0 ready" with **4 MiB GPU** (vs the dead-pod 627 MiB tell). Fix: `stop-trainer-control --remove-run`, wait for GPU to clear to 0 MiB, relaunch unhurried → engine-ready in 14 polls. Don't rapid-fire relaunches.
- **TP=2 35B OOM** at mem-frac 0.8 → see §3 tuning.
- **Benign FSDP warning** "1 of 2 modules … did not run forward before backward [FSDPLinear(…→248320)]" — the lm_head; intrinsic to the streaming-KL path (KL does `hidden @ head_weight` directly, bypassing the lm_head module forward; grads still flow). Not an error.
- The generator + config + candidates are **untracked** in the live worktree (as the experiment tooling always is). The merged *tracked* code is committed on `codex/opd-port-20260602`.

## 6a. UPDATE (2026-06-09, later) — root cause corrected + deepep/quack switch

- **The "OPRD K=C timeout" was NOT compute-bound (§5 superseded).** A second agent found it was a **collective desync**: merge regression `955191dd` dropped the `_ensure_opd_loss_metric_accumulators` seeding call, so under packing some ranks skip seeding the OPD metric keys → dict-keyed all-reduce keyset mismatch → those ranks block → fb sits to the 1800s timeout (the `feedback_dist_allreduce_dict_keyed` failure class). **Fix:** restored the `_ensure_opd_loss_metric_accumulators` call in `model_runner.py` + added the `opd_oprd_*` keys to its canonical set (on the worktree import path). So the K=C arm should land once the fix is live; the A/B/C "reduce the budget" options in §5 are NOT needed for correctness (only if you separately want faster steps).
- **EP backend switched to DeepEP + quack** (per user, 2026-06-09): `qwen3_5_35b_a3b_opd_opdb_8node.yaml` now `ep_dispatch: deepep`, `moe_implementation: quack`, `deepep_num_sms: 24` (was alltoall/triton/72). Selects the `quack_deepep_no_permute` fused path (`experts.py:779`). quack+EP is only supported via DeepEP (quack+alltoall+EP is the unsupported combo that forced the earlier triton workaround). DeepEP wheel confirmed available (same image as the MTP stack, which runs deepep). `train_router:false` + `deepep_async_combine:false` satisfy the guards. **First deepep run on this stack — watch step 0 for DeepEP buffer init.**
- A `recover_dead_bare_pods()` auto-restart handler was added to `autopilot_supervisor.py` (recreates terminal-dead bare pods, bounded 3×/pod, ESCALATE past bound; runs before relaunch). Staged for the supervisor's next start.

## 6b. UPDATE (2026-06-09, overnight) — RUN WORKING on deepep+triton

- **The OPRD K=C buffer arm (PTC-302B) is TRAINING** on `ep_dispatch: deepep` + `moe_implementation: triton` + `deepep_num_sms: 24`, 2×TP=2 SMG samplers. Step 0 fb **90.6s** / step_total **146s** (incl. triton warmup); steps 1+ ~80–90s; `opd_oprd_num_layers: 40`, `opd_oprd_loss` emitted, `sync_endpoint_success_count: 2`. **The prior 1800s "timeout" was 100% the collective desync (regression 955191dd), not compute** — confirmed: the same K=C config now completes fb in ~90s. ~81 steps ≈ ~2h.
- **quack is NOT usable for EP — and it's an upstream bug, not the merge:** upstream `backend/__init__.py` imports `QuackEPGroupGemmMoeAct`, but that symbol is defined in **neither** ours nor theirs `quack.py` → ImportError → quack drops out of `EP_EXPERT_COMPUTE` → quack+EP rejected by the `experts.py:745` gate (regardless of dispatch). So the user-requested `moe_implementation: quack` was switched to **triton** (the MTP-proven EP MoE backend). To actually get quack+EP, that missing-symbol bug must be fixed (define/port `QuackEPGroupGemmMoeAct` into `quack.py`, or drop its import in backend) — separate task.
- **Config (live):** `moe_implementation: triton`, `ep_dispatch: deepep`, `deepep_num_sms: 24`. Generator sampler tuning for TP=2 (mem-frac 0.85 / max-running-requests 256 / max-total-tokens 16384 / cuda-graph-max-bs 64).
- **Validated combos this session:** deepep+triton no-filler probe (PTC-302Np, 2×2) completed rc=0; deepep+triton OPRD K=C (PTC-302B) training. Do NOT restart the autopilot supervisor while a manual candidate runs — it would launch a queued idea and clobber the manual run.

## 6c. UPDATE (2026-06-09, overnight COMPLETE) — two clean OPRD runs end-to-end

Both OPRD K=C buffer-arm runs **completed cleanly (`cleanup rc=0`)** on the merged deepep+triton 2×TP=2 stack, zero errors, zero pod restarts:
- **PTC-302B** (16×81): completed; loss 5.72→4.86.
- **PTC-302Bx** (16×321 ≈ 5,136 exposures — the exposure-matched "adequately-trained" buffer arm vs 302N's 64×81): completed step 320 + final n=128 control; loss 5.72→4.93; ~70s/step (~6h).

**Science read (gate on loss, NOT in-loop eval):** loss dropped fast to ~5.0 by step ~40 then **plateaued flat through step 320**; `opd_oprd_loss` 0.0075→~0.003 (OPRD term optimized). The step-320 control shows `eval/acc_pause = eval/acc_nopause = buffer_delta = 0.0` — **this is the documented static-buffer-eval artifact** (the in-loop eval uses a static buffer, not K=C; the K=C eval path is a follow-up), NOT a training failure. Net: the all-layer OPRD buffer arm trains stably but the loss plateau is consistent with S11 (OPRD doesn't break the single-pass plateau). A proper **K=C eval** is the follow-up needed to actually score the buffer arm head-to-head vs 302N (0.359).

**Infra goal: fully met.** The run that kept crashing (quack-EP, desync 1800s timeout, deepep init, TP=2 OOM, rapid-relaunch hangs) now runs reliably to completion. Stack left warm (samplers/teachers up) for the next experiment; trainer idle.

## 6d. CRITICAL (2026-06-09) — our "OPRD" ≠ the paper's OPRD

Reviewing the OPRD paper (method + experiments) against our PTC-302B/302Bx K=C buffer arm: **we implemented the OPRD *mechanism* (per-layer hidden MSE) but the wrong *objective*.** This is why "all-layer OPRD doesn't help" — it's the encoded-reasoning buffer experiment, not OPRD.

Paper OPRD: student samples a real on-policy response `ŷ`; **both student and teacher run on the SAME ŷ (same tokens/positions)**; MSE on the **last-k=2000 RESPONSE tokens**, all layers; **stronger teacher** (JustRL-1.5B) vs weaker student (R1-distill-1.5B), same backbone; **OPRD-only (μ=0)**. Closes the student→teacher gap because the teacher has something real to teach on the shared tokens.

Our K=C arm: `match_cot` matches **student pause-buffer-i ↔ teacher CoT-i** = *different tokens, different sequences* (cross-token cram-CoT-into-buffer); supervises the **buffer**, not the response tail; **self-distill** (no genuinely-stronger teacher; only privileged CoT *context*); **KL + 100·OPRD**, not OPRD-only; 5×5 mult (≈10-token answer, no long student CoT). → not a valid test of the paper's OPRD; its failure matches the prior buffer-no-op findings (S6/S11).

**Fix (morning design call):** point the hidden-match at the **shared response/ANSWER positions** (same `ŷ` tokens), matching student-answer hiddens to the CoT-informed-teacher answer hiddens (distill the CoT benefit into the answer computation) — a code change, not a knob. Needs a teacher that's genuinely better at the supervised positions (stronger checkpoint, or the CoT-context teacher at the answer), and ideally a task where the student emits a long CoT so "last-k response" has substance (5×5-mult is a poor testbed). Did NOT re-architect overnight (needs design + can't verify blind). Kept the clean no-filler baseline (PTC-302Nx) training instead.

## 6e. No-filler plateau result (decisive S11 data point) — 2026-06-09 overnight

**PTC-302Nx** (no-filler control, 64×101 ≈ 6.5k exposures, deepep+triton) completed `rc=0`: `eval/accuracy` plateaus **~0.35** (noisy n=64; range 0.25–0.48, peak 0.48 @ step77; final control acc_pause 0.34 / acc_nopause 0.37; loss 0.62→0.25). **This is far below the 4×4 no-filler 0.88** → per S11, **5×5 has real single-pass headroom** (the direct-answer baseline does NOT internalize the 5×5 multiply algorithm), the regime where a *correctly-implemented* OPRD/buffer (see §6d) could actually help. Running **PTC-302Nxx** (128×101 ≈ 13k, the exact 4×4-matched budget) to settle whether 5×5 no-filler reaches a higher plateau at full budget or stays low.

**Overnight tally: 4 runs, all `cleanup rc=0`, zero infra failures** on the merged deepep+triton 2×TP=2 stack: 302B (K=C buffer 16×81), 302Bx (K=C buffer 16×321), 302Nx (no-filler 64×101), 302Nxx (no-filler 128×101, running). The merge + infra are production-stable; the open science items are (a) the correct-OPRD redesign (§6d) and (b) the full-budget no-filler plateau (302Nxx).

## 6f. No-filler 5×5 OVERFITS (peak-then-decay) — key science (2026-06-09)

302Nxx (no-filler, full 128×101 budget) `eval/accuracy` **peaks 0.477 @ step 15 then trends DOWN to ~0.16–0.26** while **loss falls monotonically** (0.64→0.25). It's overfitting / loss≠accuracy (S5), **NOT** EOS-collapse: `empty_frac=0.00`, `mean_completion_tokens≈11` (stable), `has_digit_frac=1.00` throughout — valid numeric answers that get increasingly wrong. Unlike 4×4 no-filler (monotonic→0.88), **5×5 is beyond single-pass capacity**: OPD pushes the student to match the teacher's answer distribution but it can't do 5-digit multiply in one pass, so held-out acc decays as it over-distills. ⇒ the no-filler 5×5 ceiling is **~0.48, early (step~15), unstable** (use early-stop + lower lr), and this is the **strongest headroom evidence yet** for the buffer/extra-serial-compute hypothesis a correct OPRD (§6d) targets.

## 7. Validated vs open
- ✅ Merge resolved + compiles + imports + route-parity; OPD-loss subsystem coherent (ours); infra = MTP-proven upstream.
- ✅ 2×TP=2 non-priv SMG samplers schedule, load, serve, round_robin.
- ✅ Engine init, dual-endpoint registration, **2-endpoint p2p weight sync**, OPRD config active, OPRD fb executes on a healthy 64-rank gang past all 3 fixes.
- ❌ OPEN: an OPRD fb *completing* — blocked only by the K=C compute cost vs the 1800s timeout (§5). Pick option A/B/C and relaunch.
