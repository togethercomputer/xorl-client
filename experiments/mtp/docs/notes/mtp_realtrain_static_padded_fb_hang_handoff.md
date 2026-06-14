# SingleShot-MTP OPD: real-training step-0 FB hang — `static_padded_seq_len` too small → collective desync

> **RESOLVED 2026-06-11 (trainer/CP agent), both fixes on `apanda-dev-mtp`:**
>
> - **(B) Fail-loud gate landed.** `ModelRunner._sync_micro_batch_prep_error` all-reduces an
>   error flag (same group as `_count_global_valid_tokens`, via `all_reduce_metadata_tensor`)
>   between SingleShot replay prep and `count_valid_tokens` in `_forward_loop`. A per-rank prep
>   `ValueError` now raises on **all** ranks immediately; the dispatcher's existing post-execution
>   `_sync_error_state` then returns `Cross-rank error: rank N: …` to the client in seconds
>   instead of the 1800 s PG-timeout hang. Unit tests in `tests/server/runner/test_opd_runner.py`.
> - **(A) Sizing root cause was deeper than the override.** The generator's
>   `mtp_raw_seq_len_for_max_new_tokens` budgeted `k_toks` per trace row, but a replayed
>   ConfAdapt row is recompute-prefix + k drafts with `q_len ≤ 2k−1` (canonical q banding pins
>   steady rows at exactly 7 for k=4) — so even *dropping* the override would have produced
>   1536 < observed 1552. Fixed the formula (and its inverse used by the
>   `--mtp-max-static-padded-seq-len` OOM guard) to `2k−1` per row; prod shape (prompt 512,
>   mnt 256, k 4) now sizes to **2304**. The `--mtp-static-padded-seq-len 1536` override is
>   removed from `launch_args_er-opd-q36-mtp-ss-0605c.txt`.
> - **Memory coupling resolved per §A caveat:** the launch args flip
>   `--trainer-gradient-checkpointing-method no_recompute → recompute_before_dispatch`
>   (~2x FB, still ~17x over legacy — prod handoff §3's documented fallback). `no_recompute`
>   was already memory-marginal at 1536 once step-1 optimizer state lands (the config is Muon,
>   not AdamW — momentum buffers + AdamW moments only for non-matrix params), and a one-rank OOM under deepep
>   presents as a silent hang; do not re-tighten without a measured step-1 memory headroom check.
> - Expect steady FB ≈ 25s (2x the 12.6s no_recompute estimate) until packing
>   (`recompute_before_dispatch` + per-layer reshard, see `mtp_opd_workload_analysis.md` §3)
>   is promoted.

**Date:** 2026-06-11
**Stack:** `er-opd-q36-mtp-ss-0605c` · **Code line:** `apanda-dev-mtp` (run from `apanda-dev-mtp-resume` worktree, `OPD_XORL_REPO=/home/apanda/xorl-resume`)
**Failing run:** `…/q36mtp-20260611T052153Z-2s2t/` (server.log) + `…/trainer-head/logs/20260611T052152Z-run.log`
**Reporter:** resume/ops agent. **Owner:** trainer/CP agent (FB internals + static-padding ↔ memory tradeoff + the prod EP=32 config).

---

## TL;DR

The **first real-training step** on the promoted prod config (EP=32 / deepep / quack) hangs ~30 min at the step-0 forward-backward and dies with `504: Forward-backward timeout after 2400.0s`. Root cause:

- One rank's **sampled rollout has a SingleShot replay layout `raw_seq_len=1552` > `--mtp-static-padded-seq-len 1536`** → that rank raises `ValueError` in replay-batch prep and **never reaches the `count_valid_tokens` global-valid-token all_reduce**; the other 31 ranks block in that all_reduce for the **1800 s** PG timeout → `DistStoreError` → gloo dispatcher "Application timeout caused pair closure" cascade.

Two independent issues:
- **(A) Config — static padding too small for real rollouts.** `1536` is *below* the worst-case replay layout for this prompt/gen shape; it must be raised (or computed, not overridden).
- **(B) Robustness — a single-rank replay-prep error desyncs a collective.** A `ValueError` on one rank should fail loud on *all* ranks immediately, not hang the others for 30 min at `count_valid_tokens`.

**Not** a deepep/quack hang and **not** the determinism-gate caveat. **Samplers are fine** (the separate `--student-max-total-tokens 32768` KV-pool OOM was already fixed).

## Why the perf agent's validation missed it

The EP=32/quack config was validated only as **replay / `--skip-optim-step` cells with a fixed saved payload** (`OPD_FB_REQUEST_REPLAY_PATH`) whose layout fit in 1536. Replay never exercises the real path: *sample 32 fresh variable-length rollouts → distribute across 32 ranks → build per-rank replay layout → `count_valid_tokens` all_reduce → FB → optim*. Real rollouts have variable replay layouts and the tail exceeds 1536. This is the first real-training step on the integration line. (The prod handoff §3 already hedged that `no_recompute` was "validated under `--skip-optim-step`".)

## Symptom (logs)

Client (`run_opd_pipeline`, trainer-head run.log):
```
05:28:38  Async OPD step 0: 1 prompt chunks … Prepared OPD chunk 1/1: sample=24.832s teacher=8.073s tokens=20822
   …(stuck ~30 min)…
06:08:?   AssertionError: Future … failed: {'error': '504: Forward-backward timeout after 2400.0s', 'category':'server'}
          trainer-head cleanup rc=1
```
Server (server.log): 31 ranks
```
Error executing command forward_backward: wait timeout after 1800000ms, keys: /default_pg/0//3
  … runner_dispatcher.py:549 _handle_forward_backward → model_runner.py:4319 forward_backward
  → _count_global_valid_tokens (model_runner.py:2332) → count_valid_tokens (trainers/training_utils.py:156)
```
then the gloo command-broadcast cascade (`runner_dispatcher.py:317 broadcast_object_list … Application timeout caused pair closure`).

## Root cause (exact)

`server.log` lines ~836–870 — **Rank 6** raised, *before* the all_reduce:
```
ValueError: static_padded_seq_len is too small for SingleShot replay: static=1536 raw_seq_len=1552
  File src/xorl/mtp/singleshot.py:1573  _resolve_padded_seq_len
  ← singleshot.py:2535  _prepare_singleshot_mtp_native_trace_replay_opd_batch
  ← singleshot.py:2755  prepare_singleshot_mtp_rollout_replay_opd_batch
  ← model_runner.py:2539  _prepare_singleshot_mtp_micro_batches
  ← model_runner.py:3668  _forward_loop
  ← model_runner.py:4319  forward_backward
```
So rank 6's prompt produced a replay layout needing **1552** tokens (512 prompt + mask blocks + re-prefill of accepted tokens), exceeding the 1536 static cap → raised in micro-batch prep → it skipped the `count_valid_tokens` all_reduce that the other 31 ranks then blocked on. A `Rank 6: Cross-rank error detected: static_padded_seq_len is too small` line exists but only appears **after** the 1800 s timeout — too late to prevent the hang.

## Fixes (two, independent)

### A. Config — raise the static padding (immediate unblock)
`--mtp-static-padded-seq-len 1536` (in `launch_args_er-opd-q36-mtp-ss-0605c.txt`) is set **below** the worst-case replay layout for prompt-len 512 + max-new-tokens 256. The generator can compute the correct upper bound — see `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`:
- `mtp_raw_seq_len_for_max_new_tokens()` / `mtp_static_padded_seq_len()` (≈ lines 108–128) compute the raw upper bound from `max_new_tokens`.
- The hard `--mtp-static-padded-seq-len 1536` override **bypasses** that and pins it too low.

Options:
1. **Drop the `--mtp-static-padded-seq-len 1536` override** and let the generator size it from `max_new_tokens=256` (gives the correct upper bound), or
2. set `--mtp-static-padded-seq-len` to the computed worst case (≥ ~1664–2048; 1552 observed is only one tail sample, pick the true max), or
3. set `--mtp-max-static-padded-seq-len` and let the OOM-guard cap drive both `effective_max_new_tokens` and the static shape.

**CAVEAT (memory):** a larger static padding grows the static micro-batch shape → more activation memory. The prod config uses `no_recompute`, which the handoff §3 already flags as memory-marginal on step 1 (AdamW state). Raising static padding likely needs `recompute_before_dispatch` (which the perf agent's later packing work already recommends). So this couples to the packing/recompute prod-config decision.

### B. Robustness — don't let one rank's replay-prep error hang the collective
A `ValueError` in `_resolve_padded_seq_len` / replay prep on a single rank causes a 1800 s hang on every other rank's `count_valid_tokens` all_reduce. Desired: detect the per-rank prep failure and **all-reduce an error flag *before* `count_valid_tokens`** so all ranks raise together immediately (fail-loud), or handle the oversized rollout gracefully (clamp/skip it; the rank participates in `count_valid_tokens` with 0 valid tokens). The existing "Cross-rank error detected" path fires after the timeout — it needs to gate the collective, not follow it. Same class as the known empty-rank/dict-keyed all_reduce desyncs (`feedback_dist_allreduce_dict_keyed`, `project_opd_p2p_sync_wedge_collective_desync`). Code: `model_runner.py` `forward_backward` / `_forward_loop` / `_count_global_valid_tokens` (2332), `count_valid_tokens` (`trainers/training_utils.py:156`), and the cross-rank-error path in `runner_dispatcher.py`.

## Repro
1. Samplers up with `--student-max-total-tokens 32768` (else step-0 sampling OOMs — separate, fixed).
2. `write-trainer-control $(cat launch_args_er-opd-q36-mtp-ss-0605c.txt)` with clean env (no replay vars) — i.e., **real training, not `--skip-optim-step`, not replay**.
3. Step 0 samples 32 prompts, builds replay layouts; whichever rank draws a prompt whose layout > 1536 raises → ~30 min `DistStoreError`, ~40 min client `504 Forward-backward timeout 2400s`.

## Scope / impact
Blocks the resume agent's **Test B** (supervisor kill-recover) and **Phase 2** (sampler 2→4) and the **committed long run** — all need a working real-training step. The fix (A) is a one-line config change but couples to the memory/packing decision; (B) is a real robustness bug worth fixing so any future bad-rollout doesn't burn 30 min.
