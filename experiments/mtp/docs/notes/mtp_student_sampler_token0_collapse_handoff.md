# SGLang MTP student sampler collapses to token-0 (`!`) — OPD rollouts are corrupt

**Date:** 2026-06-11
**Stack:** `er-opd-q36-mtp-ss-0605c` (Qwen3.6-35B-A3B SingleShot-MTP OPD)
**Sampler:** SGLang from `/home/apanda/xorl-sglang-internal`, served `Qwen/Qwen3.6-35B-A3B`
**Reporter:** infra/resume agent. **Owner:** perf/SGL sampler agent (MTP native decode internals).
**Status when filed:** prod trainer STOPPED (was training on garbage); samplers + teachers left WARM for repro.

---

## TL;DR

The student SGLang **MTP speculative-draft decode collapses to token id 0 (`!`)**. Prefill/bootstrap
is fine (the first sampled token is real), but the **first MTP draft step reads garbage and the
lm-head argmaxes to token 0**, then every steady step commits 4× token-0. The sampled rollout is
`<first real token>` + `!`×255. Because OPD is on-policy (the student's rollout *is* the supervised
target) the trainer then distills on `<think>!!!!…`. The loss (~0.13) and `opd_top1_agreement` (~0.99)
look healthy but are **vacuous**: the teacher is the *same frozen base model*, so it also predicts
token-0 on the corrupt context → student and teacher "agree" on garbage.

This is **not new and not specific to one config** — it is an intermittent MTP-draft corruption present
across the entire run history. It was never caught because everyone (perf agent + me) monitored
throughput / FB time / loss / no-crash, never **rollout token coherence**.

## Exact mechanism (from `<run>/artifacts/rollout_samples.jsonl`)

For a current rollout (run `q36mtp-20260611T204611Z-2s2t`):

```
native_mtp_bootstrap_token_ids        : [248068]                      # <think>  — REAL
native_mtp_first_step_phase           : seed
native_mtp_first_step_input_row_token_ids : [248068, 248063, 248063, 248063]  # <think> + <|fim_pad|>×3 (k=4 mask slots)
native_mtp_first_step_committed_token_ids : [0]                       # committed token 0 = '!'
native_mtp_first_step_pending_token_ids   : [0, 0, 0, 0]             # model predicts 0 at ALL draft positions
native_mtp_trace_commit_lens          : [1, 4, 4, 4, 4, ...]          # seed=1 then steady=4, every commit is token-0
native_mtp_debug_trace_q_lens         : 4,7                            # seed q=4, steady q=7 (canonical q-banding, k=4)
mtp_k_toks=4   mask_token_id=248063 (<|fim_pad|>)   token 0 = '!'   248068 = '<think>'
```

So given `[<think>, <|fim_pad|>, <|fim_pad|>, <|fim_pad|>]` on top of the cached prompt, the model
returns `[0,0,0,0]`. Argmax→token-0 is the classic signature of a **garbage / zero hidden state at the
draft positions** (draft slots reading uninitialized or wrong KV). The prefill path that produced the
bootstrap `<think>` is unaffected — only the speculative-draft positions are corrupt.

## Severity is config-dependent (prefix vs suffix) — and historically pervasive

`avg_frac_tok0` = fraction of generated tokens equal to id 0, over the first ≤30 samples per run
(`rollout_samples.jsonl`), oldest→newest:

| run | sample[:30] | avg_frac_tok0 |
|---|---|---|
| 20260605T045349Z (early, prefix) | `.g. files in .gitignore, node_modules…` | 0.00 |
| 20260606T002029Z (perf "validation" ckpt run, n=187) | `.g. files in .gitignore…` | **0.13** |
| 20260608T025738Z | `.g. files!!!!!!!…` | **0.98** |
| 20260609T034743Z | `.g. files!!!!!!!…` | **0.98** |
| … (prefix runs range 0.00 ↔ 0.98 intermittently) | | 0.00–0.98 |
| 20260611T195155Z (**suffix**) | `<think>!!!!!!!…` | **0.99** |
| 20260611T204611Z (**suffix**) | `<think>!!!!!!!…` | **0.99** |

- **Intermittent across all history** (0.00–0.98) even under `prefix` → the MTP draft has been
  producing variably-corrupt rollouts the whole time; the OPD training data has been contaminated.
- **`suffix` makes it ~99%**: a fresh assistant turn (prompt ends at `<|im_start|>assistant\n`) fails
  at the *first* draft step after a 1-token bootstrap. `prefix` got ~32 coherent bootstrap/mid-stream
  tokens before the draft, so the contamination was lower (and sometimes 0). The trigger correlates
  with **how soon the draft engages after a fresh boundary**, not with the first token value (samples
  starting with `The`=760 collapse identically to `<think>`).

## Config in play (so it can be reproduced / bisected)

Sampler run.sh flags (from `launch_args_er-opd-q36-mtp-ss-0605c.txt`):
`--student-enable-cuda-graph --student-mtp-canonical-q-banding --student-mtp-adaptive-cuda-graph
--student-max-running-requests 32 --student-max-total-tokens 32768`, MTP strategy
`["conf_adapt", 0.3]`, `k_toks=4`, `temperature=0.7 top_k=1`, `prompt-len 512 max-new-tokens 256`.

**Overlap is already OFF** (`disable-overlap` in the sampler run.sh) — so this is **NOT** the known
overlap/batched-decode `prepare_for_decode seq_lens` corruption (`project_sglang_batched_decode_corruption_bisect`).
Prime suspects, in order:
1. **`--student-mtp-adaptive-cuda-graph`** — a captured graph for the seed/steady q-band (q=4 / q=7)
   shape may read stale/uninitialized draft-slot KV. Toggle OFF first.
2. **`--student-mtp-canonical-q-banding`** — the q-len banding that pins steady rows at q=7.
3. The MTP draft KV/position handling for the **first draft step after a short bootstrap** (the
   fresh-turn case `suffix` exposes). Related throughput notes: `project_sgl_sampler_throughput_mtp_buckets`
   ("conf_adapt bucket fragmentation + banding/static-graph"; "static knob gates flashinfer q>1 capture").

## Suggested bisection (decisive, ~10 min each on one sampler)

1. Recreate ONE student sampler with `--student-mtp-adaptive-cuda-graph` removed; send one `suffix`
   prompt (or any prompt that opens a fresh assistant turn) with the MTP strategy; check
   `generated_token_ids` for the token-0 run. If coherent → adaptive-cuda-graph capture is the bug.
2. If still corrupt, also drop `--student-mtp-canonical-q-banding`. If coherent → banding.
3. If still corrupt with both off (plain MTP draft), the bug is in the draft KV/seq_len handling for
   short-bootstrap first steps — inspect the draft-position KV init / attention mask for the seed step.

## Needed regardless of root cause: a rollout-coherence gate

Add a cheap guard so this can never silently train again: fail (or warn loudly) when a sampled
rollout's token-0 fraction (or `<|fim_pad|>`/special-token fraction) exceeds a threshold, in the
sampler or in `run_opd_pipeline` right after `_student_sample_*`. Today loss/agreement do not catch it
(frozen-teacher self-distillation → vacuous agreement on garbage).

## Repro pointers

- Corrupt (suffix, 0.99): `…/q36mtp-20260611T204611Z-2s2t/artifacts/rollout_samples.jsonl`
- Lower-contamination (prefix, 0.13): `…/q36mtp-20260606T002029Z-2s2t/artifacts/rollout_samples.jsonl`
- WandB (data derived from garbage, ignore): `together-research/singleshot/runs/hs3pshn1`
- `suffix` turn-strategy is correct for assistant-alignment (added @ 3615f3c1) and is **not** the
  bug — it only *exposes* the draft corruption. Keep it; fix the sampler, then re-validate rollout
  coherence before resuming.

## Impact

Blocks the prod run, Test B (kill-recover), Phase 2 (sampler 2→4) — all need coherent rollouts.
The samplers + teachers are warm for immediate repro; the trainer is stopped.

---

## ROUND 2 (2026-06-11) — batched native-MTP eager decode STILL collapses (distinct, intermittent)

After deploying the fix (sglang e447f5e2a + dropping the 6 MTP q>1 graph flags so native MTP runs
eager), the probe shows the GRAPH-path fix works for SERIAL decode but a SECOND bug remains in
**batched (bs>1 × q>1) native-MTP decode** — which is exactly the OPD workload (32 prompts/step,
`--student-max-running-requests 32`).

**Discriminators (probe `scripts/opd/probe_token0_collapse.py` against prod sampler, k=4, conf 0.3):**
- `--mode both` (SERIAL): reliably clean, frac_tok0=0.00, coherent reasoning. ✓
- `--mode both --no-mtp --batch` (plain batched decode, no MTP): reliably clean. ✓
- `--mode both --batch` (batched native-MTP): **intermittently collapses to token-0.** 5 consecutive
  identical runs, clean/collapsed counts: 3/3, 1/5, 0/6, 6/0, 3/3. Non-deterministic **per batch AND
  per prompt within a batch** (e.g. `0.00 0.98 0.00 0.93 0.91 0.00`). NOT a warmup transient (a fully
  clean run followed fully collapsed runs).
- Independent of cuda-graph: collapses both with `--cuda-graph-max-bs 64` and fully eager
  (`--disable-cuda-graph`). So this is NOT bug #1/#4 (graph path) — it's the **eager batched MTP path**.

**So:** `e447f5e2a` fixed the graphed-q>1 path (serial is clean), but batched native-MTP decode has a
separate race/state bug (looks like cross-request state bleed in the GDN varlen kernels for bs>1×q>1).
The agent's "serial and concurrent ×8 — frac_tok0=0.00" did not reproduce here; suspect their
"concurrent" was N independent single-request decodes, not one batched decode of N requests.

**Repro:** `python scripts/opd/probe_token0_collapse.py --server-url http://<sampler>:30060
--mode both --batch --k-toks 4 --conf-threshold 0.3 --max-new-tokens 96` (run ~5×; watch for
collapsed batches). Prod samplers are warm, fully eager.

**Stopgap (NOT used — too slow):** `--student-max-running-requests 1` forces serial decode (clean)
but serializes the 32-prompt sample → ~6-10× step-time blowup → impractical for the 2.35-day run.
Open question for SGL: is there a SAFE small batch size (2/4) or is any bs>1 racy?

**State:** trainer STILL stopped; samplers fully eager + warm for repro; launch_args at
`--student-max-running-requests 32` + fully-eager (no cuda-graph). Run remains blocked.

---

## ROUND 3 (2026-06-12) — MTP trace commit-length inconsistency blocks forward_backward

With graphs back on (sglang 9190ee358) the OPD trainer reclaim loads + samples cleanly (token-0
collapse is GONE), but **step-0 forward_backward fails deterministically** in the trace replay:

```
RuntimeError: Operation failed: native MTP trace committed-token row is shorter than the
supervised commit slice: row=0 step=1 got=1 required=2     (src/xorl/mtp/singleshot.py:2493)
```
(got=1 every time; required=2/3/4 varying, across steps 1,2,...).

**Mechanism:** the new hf-exact draft acceptance truncates commits at the first draft mismatch, so the
sampler commits ~1 token/step in this workload. But the emitted trace is INTERNALLY INCONSISTENT: the
trainer reads `len(committed_tokens)=1` (`_trace_committed_tokens`, singleshot.py) while it derives
`label_count` from the trace's `commit_slice.len` (=2/3/4). singleshot.py:2492-2504 then (correctly)
rejects it — that validation ALSO checks the committed tokens equal the OPD supervision targets, so it
guards against training on misaligned data.

**This is the sampler↔trainer trace contract, not a trainer config.** Do NOT disable
`validate_native_mtp_trace` (default True) — it would index `committed[:label_count]` on a shorter row
and train on misaligned tokens (the same garbage-supervision class as Round 1/2). The SGL agent's xorl
commit (397a1f62) was the coherence gate + probe + docs; the trainer's trace PARSER (singleshot.py) was
not updated for the new commit semantics.

**Fix needed (SGL/MTP-code):** reconcile the contract. Either (a) the sampler emits
`commit_slice.len == len(committed_tokens)` after hf-exact truncation, or (b) `_trace_committed_tokens`
reads the post-truncation committed row (and `label_count` is capped by it) so the trainer supervises
exactly the actually-committed tokens. The SGL agent's note "the trainer replays the actual rows
anyway" implies (b), but the cursor/loss_start/target alignment must be verified — not an infra patch.

**Sanity check for SGL:** got=1 EVERYWHERE means the MTP drafts are being rejected to a single token
in the OPD workload, even though the probe showed commit_len_mean~4. Confirm whether that's expected
(hf-exact rejecting all drafts at temp 0.7 / conf 0.3) or a second symptom — if MTP commits only 1
token/step, speculation isn't buying anything and the throughput premise weakens.

**Repro:** reclaim -0605c OPD trainer (triton, samplers on 9190ee358 graphs-on) → step-0 fb fails
immediately with the above. Trainer STOPPED; samplers/teacher warm.

---

## ROUND 3 RESOLVED (2026-06-12, SGL agent) — trainer trace parser now honors post-verify runtime fields

**Root cause:** the trace step carries the commit twice. `commit_slice` is written at *prepare* time
(schedule_batch.py) with the PLANNED commit; `commit_len_runtime`/`commit_start_runtime` +
`committed_appended_to_output_ids` are written together in the *post-verify* upsert
(scheduler_output_processor_mixin.py) and are always mutually consistent — sglang 9190ee358's hf-exact
verification truncates `commit_len_runtime` (model_runner.py `accept_len`) and the output processor
raises if the committed list disagrees. The trainer's `_trace_commit_len`/`_trace_commit_start`
preferred the planned `commit_slice` over the runtime keys — inconsistent with `_trace_recompute_len`
and `_trace_attempt_k`, which already prefer `*_runtime` first.

**Fix (this repo):** `_trace_commit_len`/`_trace_commit_start` now prefer `commit_len_runtime`/
`commit_start_runtime`, falling back to `commit_slice` then bare keys (old traces unaffected — pre-
truncation, runtime == planned). `label_count`, cursor advance, supervised positions, and
`native_commit_positions` all follow the actually-committed prefix; the `validate_native_mtp_trace`
equality check against OPD targets stays load-bearing and still fails loudly on genuinely inconsistent
traces (test pinned). Tests: `test_prepare_singleshot_opd_batch_replays_hf_exact_truncated_commits`
(reproduces the exact prod error pre-fix) +
`test_prepare_singleshot_opd_batch_rejects_truncated_commit_without_runtime_len`.

**got=1 everywhere: ANSWERED — expected at training start, not a second symptom.**
- The hf-exact truncation is NEW in 9190ee358 (`git log -S mtp_commit_len_runtime` → only that
  commit). Every earlier measurement (probe commit_len_mean~4) reflects the old semantics where the
  full planned slice was committed regardless of draft match — that over-commit was part of the
  garbage-decode problem.
- The OPD pipeline samples at top_k=1 (effectively greedy), so draft vs verify is deterministic
  argmax-vs-argmax. The drafts are mask-conditioned MTP-head predictions from a student that has NOT
  been MTP-trained yet — this run exists to distill that capability. Acceptance ≈ 0 at step 0 is the
  honest baseline; `mtp/commit_len_steady` is now a true acceptance metric and should climb as
  training progresses. Sampler-side speculation throughput will be ~AR until it does.
- Sampler logs (er-opd-q36-mtp-ss-0605c-sglang-0, [MTP_DEBUG_FIRST]) show coherent seed-step
  drafts/commits — no corruption.

**State:** fix committed on apanda-dev-mtp in the shared worktree the trainer PYTHONPATHs
(/home/apanda/xorl-mtp-singleshot-port-20260602/src) — live on next trainer start, no rebuild.
Run #10 is UNBLOCKED; prod-run agent relaunches via their staged write-trainer-control.
