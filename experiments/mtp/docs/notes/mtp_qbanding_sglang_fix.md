# SGLang fix needed: canonical q-banding crashes native-MTP decode

**Date:** 2026-06-13
**From:** throughput agent
**To:** SGL agent (sglang sampler owner)
**Repo:** `/home/apanda/xorl-sglang-internal` (branch `apanda-dev`) — the build the live samplers run
**Stack:** `er-opd-q36-mtp-ss-0605c` (Qwen3.6-35B-A3B native MTP, ConfAdapt, k=4)
**Severity:** q-banding is **unusable** on the live build — it crashes both samplers ~7 min into decode,
which then crashes the trainer. Blocks the sampler-side throughput win (the validated ~2,800 tok/s).

---

## RESOLVED 2026-06-13 (code fix landed; live A/B re-test pending)

**Root cause (confirmed + sharpened).** The native-MTP q>1 hf-exact verification in
`model_runner.py` computed the verified commit length, the pending window, and the GDN/Mamba
state-advance boundary **uniformly from request 0's commit window** (`commit_start_uniform` /
`commit_len_uniform`). That is only correct when every row in the decode batch shares a commit
window. The batch builder (`schedule_batch.py:2874/2879`) enforces a uniform `q_len` and `emit`
window but **deliberately keeps `commit_start`/`commit_len` per-request** (`:2905-2906`, no
uniformity check). Without banding, uniform `q_len` ⟹ uniform `a_prev` ⟹ uniform commit windows, so
the uniform shortcut happened to be correct. **With banding**, `recompute_len`/`q_len`/`emit` are
forced to the canonical value but `commit_len = a_prev` and `commit_start = recompute_len − a_prev`
still **vary per row** (each request accepted a different count last step). A row with a smaller
`commit_len` was verified against row 0's wider window, so `accept_len` could exceed its own planned
`commit_len` → `verified=2 > planned=1` at `scheduler_output_processor_mixin.py:710`. (The same
row-0 bug silently advanced every row's Mamba state to row-0's boundary — the fix-direction-#2
caveat — so clamping the crash alone would have left state corruption.) The bug only surfaces once
requests diverge in acceptance *and* co-occur in one banded batch, i.e. minutes into decode.

**Fix (landed on `apanda-dev`).** Made the verification fully per-request:
- New pure helper `python/sglang/srt/managers/mtp_decode_verify.py :: mtp_hf_exact_accept_len(...)`
  vectorizes the hf-exact prefix-match per row, masking columns past each row's own `commit_len−1`
  so `accept_len[i] ≤ commit_len[i] == planned[i]` **by construction** — the
  `verified ≤ planned` invariant now holds without relaxing the scheduler check.
- `model_runner.py` native-MTP decode path: normalize the per-request commit metadata up front, then
  use per-request `commit_start_t`/`commit_len_t` for `accept_len`, `pending_start`, and the Mamba
  `accepted_steps` advance (replacing all three `commit_start_uniform`/`commit_len_uniform` uses).
  The downstream `fused_mamba_state_scatter_with_mask` already indexes per row by `accepted_steps[i]`,
  so each row now advances to its own boundary; the index range is unchanged (`≤ recompute_len−1`,
  the same values row 0 already exercised) → no new OOB.
- Verified bit-identical to the old code in the non-banded case (parity test) and against a
  brute-force reference in the banded case.

**Tests.** `test/srt/zorl/test_mtp_decode_verify.py` (CPU-only, importlib-loaded like
`test_opd_kl_streaming.py`): non-banded↔uniform parity, banded per-request correctness +
`accept_len ≤ commit_len` invariant, a concrete `verified=2/planned=1` repro, and empty/all-single
edge cases. All pass (`python -m pytest test/srt/zorl/test_mtp_decode_verify.py`).

**Deployment.** Samplers import sglang from the mounted host path
`/workspace/xorl-sglang-internal` (pure-Python, loaded at runtime) — the fix takes effect on the
next sglang **process** restart in the sampler pods; no image rebuild needed.

**Remaining: live A/B re-test (not yet done).** Validating the ~2,800 tok/s win requires enabling
the banding flag on the live stack, which (per the operational note at the bottom of this doc)
needs the supervisor-pause + coordinated `stop-trainer-control → write-student-inference-control
+banding → write-trainer-control → unpause` sequence and will perturb the running production
trainer. Left for an explicitly-authorized run.

---

## Symptom

With canonical q-banding enabled (`--enable-mtp-adaptive-hf-exact-canonical-q-banding
--mtp-adaptive-canonical-q-lens 7`, i.e. the launcher's `--student-mtp-canonical-q-banding`), the samplers
boot fine — banding confirmed active:

```
server_args: enable_mtp_adaptive_hf_exact_canonical_q_banding=True, mtp_adaptive_canonical_q_lens=[7]
Capture MTP q>1 cuda graph runners. static_q_lens=[4, 7] adaptive_hf_exact_q_lens=[4, 7] capture_q_lens=[4, 7]
```

…then both TP ranks die `rc=1` after ~7 min of decode:

```
File ".../managers/scheduler_output_processor_mixin.py", line 711, in process_batch_result_decode
    raise ValueError(
ValueError: Invalid verified MTP commit length. rid=..., verified=2, planned=1, phase=steady.
```

Live evidence: `/shared/opd-control/er-opd-q36-mtp-ss-0605c/sglang-0/logs/20260613T071839Z-run.log`
(boot 07:18:39, crash 07:27:17). Reproduced cleanly: enable banding, let it decode a few minutes.

## Root cause

`scheduler_output_processor_mixin.py:706-716` enforces the invariant that hf-exact verification only
**truncates** the planned commit (verified ≤ planned):

```python
# hf-exact draft verification can truncate the commit below the planned length
# (commits past the first draft mismatch were conditioned on rejected tokens).
if mtp_commit_len_runtime is not None and i < len(mtp_commit_len_runtime):
    commit_len_verified = int(mtp_commit_len_runtime[i])
    if not (1 <= commit_len_verified <= commit_len_this_step):   # planned = commit_len_this_step
        raise ValueError("Invalid verified MTP commit length. verified=..., planned=...")
    commit_len_this_step = commit_len_verified
```

For a ConfAdapt steady step the **planned** commit is pinned to the previous accepted length:
`commit_len_this_step == a_prev` (enforced at `:666`). But q-banding upsizes the decode q-len to the
canonical value (7) and recomputes the window geometry in `schedule_batch.py`:

```python
# schedule_batch.py  mtp_step_layout / mtp_select_canonical_q_len_for_step
recompute_len = canonical_q_len - attempt_k + 1     # banded: q -> 7
commit_start  = recompute_len - a_prev
canonical_q_banding_applied = True
```

So under banding the model verifies drafts over the **wider banded window** (q=7) and hf-exact can accept
**more** committed tokens than `a_prev` (`verified=2 > planned=1`). The "truncate-only" invariant at `:710`
is violated → crash. In short: **the planned-commit / `a_prev` accounting is computed pre-banding, but the
verified commit comes from the post-banding wider window — the two are inconsistent.**

(Without banding, q = `a_prev + k - 1` tracks `a_prev`, so verified ≤ a_prev = planned holds — which is why
baseline is stable and only banding trips this.)

## Fix direction (SGL agent to decide the right one)

The banded-window commit accounting must be made consistent. Candidate fixes:

1. **Make the planned commit length banding-aware.** When `canonical_q_banding_applied`, the planned
   `commit_len_this_step` / the `commit_len == a_prev` check (`:666`) and the verified-commit bound (`:710`)
   should be computed against the **banded** window geometry (the extra `recompute_len - a_prev` prefix the
   banding re-feeds), not the pre-banding `a_prev`. i.e. allow `verified` up to the banded committable span.
2. **Clamp instead of crash**, if a verified commit beyond `a_prev` is semantically valid under banding
   (commit `min(verified, banded_committable)` and advance state accordingly) — but only if that matches what
   the GDN/KV state actually advanced; otherwise it corrupts state. Needs your read on the decode path.
3. **Confirm against the snapshot.** The SGL agent's snapshot repo `xorl-sglang-sglagent-20260610` is where
   banding hit ~2,800 tok/s in the original bench. If that snapshot does NOT crash here, the banding↔commit
   fix likely lives there and was only partially upstreamed into `xorl-sglang-internal` (the gdn q>1 conv fix
   was upstreamed per AGENT_NOTES 2026-06-10 ~03:00; the commit-verification reconciliation may not have
   been). Diffing `scheduler_output_processor_mixin.py` + `schedule_batch.py` (the `mtp_step_layout` /
   `mtp_select_canonical_q_len_for_step` / commit-verify regions) between the snapshot and live should reveal
   the missing piece quickly.

## Why this matters (throughput context)

q-banding is the sampler-side throughput lever: ConfAdapt fragments the MTP decode batch into per-q_len
buckets (q ∈ {4,5,6,7}) → near-serial decode (~100 tok/s aggregate); banding pins steady steps to q=7 so all
requests share one bucket (bench: ~2,800 tok/s). On the live build it is currently a **crash**, not a no-op.
Until this is fixed, q-banding cannot be A/B'd or used.

NOTE for whoever re-tests after the fix: applying the banding flag wedges the trainer on the in-flight weight
sync, and the trainer-supervisor's crash-recovery **regenerates the sampler control from the args file
(without the banding flag)** — so a naive A/B silently reverts banding. Re-test with `touch
$CTL/supervisor.pause` first, then a coordinated `stop-trainer-control → write-student-inference-control
+banding → write-trainer-control`, and **unpause when done**. (See
`docs/notes/mtp_throughput_findings_20260613.md` for the full operational story.)
