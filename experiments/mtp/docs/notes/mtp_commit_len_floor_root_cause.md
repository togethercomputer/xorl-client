# ROOT CAUSE: OPD-MTP `commit_len` floor — replay context leak + unsupervised draft heads

**Date:** 2026-06-12 · **Continues:** `mtp_commit_len_floor_debug_handoff.md` · **Run:** `q36mtp-20260612T095624Z-2s1t`
**Verdict:** two compounding training-side bugs in `src/xorl/mtp/singleshot.py`. The sampler is fine (H4 ruled out
by metrics below); the loss mode is fine (H1 refuted — see §1). The student is being trained on a **corrupted
conditioning distribution** (§2) and the **positions that produce drafts receive zero gradient** (§3).

---

## 1. H1 (forward-KL mode-covering) is REFUTED — the loss is already mode-seeking

`opd_hard_teacher_ce` is only emitted when `loss_mode == "hard_teacher_ce"` (`src/xorl/ops/loss/opd_loss.py:613`),
and the run logs set it (equal to `loss`/`opd_kl`/`ce_teach_stud` — they all alias the same per-token quantity in
this mode). So the active objective is **CE against the teacher argmax** — exactly the "smallest fix" the handoff
proposed under H1. Mode-covering is not the problem.

## 2. H6 (new): the replay attends to REJECTED DRAFTS that the runtime evicted

### The smoking-gun metrics (steps 308–399, trainer-head log)

| metric | value | meaning |
|---|---|---|
| `opd_student_rollout_token_agreement` | **0.63–0.68, flat** | student argmax vs its OWN GREEDY rollout tokens (`top_k=1`) |
| `opd_teacher_rollout_token_agreement` | **0.88–0.89** | frozen teacher argmax vs the student's rollout |
| `opd_top1_agreement` | 0.65–0.69, flat | student vs teacher argmax at supervised positions |
| `ent_stud` / `ent_teach` | 1.4–1.7 / 0.52–0.64 | student ~3× teacher entropy at the SAME positions |

A model replaying **its own greedy rollout** must agree with it ≈1.0 if the replay reproduces the decode-time
conditioning. It agrees **0.65** — while the *frozen teacher* (clean AR prefill over the committed text) agrees
**0.89**. The student predicts its own output *worse than a different model does*. The replay forward is therefore
conditioned on something the runtime never saw. This also explains why step-0 self-distill CE was ~2.6 instead of
≈ teacher −log p(argmax) ≈ 0.5.

### The bug

The replay visibility rule (three copies: `SingleShotRolloutReplayMask.allowed()` @ `singleshot.py:698`, the flex
`mask_mod` @ `:1430`, and the GDN linear-plan context selection @ `:878,:906`) is:

```python
source_visible = key_is_context & (k_source <= q_context)
```

where `k_source`/`q_context` are **trace position ids**. In the hf-exact got=1 regime, each decode window replays
the previous step's **rejected pending drafts as REFILL (real) tokens**, and the committed token of the next window
sits at the **same position id** as the rejected draft it superseded (see the fixture in
`tests/mtp/test_singleshot.py:858`: rejected draft `88` at position 4, committed `52` also at position 4). So:

- the query at committed `52` (q_context=4) sees rejected `88` (k_source=4 ≤ 4) — **stale-draft leak**: every
  later query attends every rejected draft ever produced (~1.8 garbage tokens/position at effective_k≈2.8, got=1);
- the query at `88` (q_context=4) sees the *future* committed `52` (k_source=4 ≤ 4) — **future-token leak**;
- the GDN flat context sequence (`:878`) packs prompt + ALL refills — committed and rejected interleaved — into one
  recurrent scan, so the linear-attention state at every supervised position has absorbed all the garbage.

At runtime the verify-style KV state (sglang `9190ee358`) **overwrites** the rejected slot with the committed token
before attention — the runtime context is clean committed-prefix-only. So the trainer optimizes
p(next | committed prefix + all rejected drafts) while decode acceptance tests p(next | committed prefix). The loss
descends (the student slowly learns to predict through garbage) but transfers nothing reliable to the clean-context
decode distribution: acceptance stays at the floor, `ent_stud` stays ~3× teacher (contradictory context tokens),
flat `top1_agreement` ~0.67.

## 3. H2 confirmed, stronger than stated: draft-emitting positions get ZERO gradient — structurally

Labels are placed only at the commit slice (`label_count = min(commit_len_runtime, …)` @ `:2491`, label write @
`:2619-2625`), and the commit slice is **validated to lie inside the recompute prefix** (`:2472`). The emit window —
the positions whose logits actually produce drafts at runtime (default `[recompute_len-1, attempt_k)` = last refill
+ mask slots, `:130`) — **never overlaps the supervised set** (commit ends at offset `commit_len-1 < recompute_len-1`
at got=1). So:

- The supervised positions train the **verify side** (clean-context next-token).
- The **draft side** (mask slots + the pending-context last-refill position) is never trained — not "only when
  rejected" as the handoff guessed, but *never, at any acceptance level*.

Acceptance = (draft argmax == verify argmax). Training only one side of that equality cannot raise it; the shared
trunk gives at most slow incidental transfer (the observed ent_stud drift 1.70→1.44 over 90 steps with zero
commit_len movement).

## 4. The fix (both parts required)

**(A) Faithful replay conditioning** — re-key visibility on a clean *trajectory-slot* axis (prompt token i → slot i,
generated[j] → slot prompt_len+j; window offset o ↔ slot prompt_len+label_start−1+o, verified on both test
fixtures). Stale refills (window input token ≠ trajectory token at its slot) get a sentinel key-source → invisible
cross-block; window queries get `q_context = window_start_slot − 1` → future tokens excluded; a new same-window
causal clause keeps pending drafts visible *inside* their own window (runtime-faithful — pending drafts ARE context
for in-window queries and for next-draft emission). GDN flat context = prompt + committed refills only; stale and
mask queries route through per-block branch sequences.

**(B) Supervise the emit window** — label each emit offset `e` with `generated[label_start+e]` (the trajectory token
its runtime draft targeted), teacher hidden gathered at the same trajectory index (clean teacher context). With (A),
the emit positions reproduce the runtime *draft* conditioning, so hard-CE pushes draft argmax → teacher argmax while
the commit positions push verify argmax → teacher argmax — both sides of the acceptance test converge to the same
token. (B) without (A) trains drafts on polluted context; (A) without (B) trains only the verify side.

### Ceiling note
`teacher_rollout_token_agreement ≈ 0.89` (greedy self-rollout vs same-weights teacher prefill) bounds per-token
acceptance ≈ 0.85–0.9 even after perfect distillation (decode-vs-prefill numerics at `pending_confidence ≈ 0.48`
near-ties). That's commit_len ≈ 3+ at k=4 — a fine target, not a floor explanation.

## 5. Fix landed — branch `fix/mtp-replay-visibility-emit-supervision` @ `b0cfc6de`

Worktree `/home/apanda/xorl-mtp-commitlen-fix-20260612` (created off `apanda-dev-mtp` @ `ac79f331`).
Changes: `src/xorl/mtp/singleshot.py` (slot-axis visibility + sentinel stale keys + same-window clause in
`allowed()`/flex `mask_mod`, committed-only GDN flat context + per-window branches, emit-window labels),
`src/xorl/server/runner/model_runner.py` (`supervise_emit_window` config key, default **on**),
`tests/mtp/test_singleshot.py` (leak-regression asserts; conditioning-parity test proving replay commit
positions exactly reproduce a clean causal pass over the committed text; clean-GDN-plan test). 39/39 mtp +
32/32 opd-runner + 48/48 batch-utils/payload tests pass; ruff clean.

### Launching the A/B
- **Do NOT merge into the `apanda-dev-mtp` worktree while the live run is up** — the trainer pods import
  `PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src` live, and the supervisor restarts crashed
  trainers; a merge would hot-apply the fix mid-run. Launch the A/B from the fix worktree
  (`OPD_XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612`; run `uv sync` there first or point
  `PYTHON_BIN` at the dev venv), or merge only when deliberately relaunching.
- `singleshot_mtp.supervise_emit_window: false` restores the old labeling for a (B)-only ablation; the
  visibility fix (A) is unconditional.
- Expect at step 0 of the fixed run: CE near the teacher's −log p(argmax) (~0.5) at commit positions instead
  of 2.6, `student_rollout_token_agreement` ≈ 1.0, `mtp/supervised_tokens_actual` (= `valid_tokens`) ≈ k× the
  expected `mtp/supervised_tokens` (the expected-vs-actual metrics legitimately diverge now — they count
  commit-only vs commit+emit). Success metric: `commit_len_steady` climbing off 1.0x within tens of steps as
  emit-position CE drops; per-token acceptance ceiling ≈ 0.85–0.9 (§4 ceiling note) → commit_len ≈ 3 at k=4.

## 6. Don't break
Per the handoff: validate offline / as a separate A/B run. The live run keeps producing valid teacher supervision;
the fix changes loss semantics (`supervised_tokens` grows ~k×) and replay masks, so it must NOT be hot-applied.

## 7. A/B RESULT + CORRECTION (2026-06-13): under-trained, not a loss-mode problem

The fix ran 50 steps at scale (`q36mtp-20260612T204931Z-2s1t`, student synced every step). Aggregate readout
looked negative — commit_len flat ~1.03, top1 ~0.26, ent_stud ~3.0, hard_teacher_ce 6.1→5.4 — and was initially
read as "the fix doesn't lift commit_len ⇒ the lever is the loss mode (H1)." **Per-offset analysis of the sampler
trace (`scripts/opd/analyze_mtp_draft_acceptance.py` on `rollout_samples.jsonl`) overturns that reading:**

- **The aggregate metrics mislead.** They blend the VERIFY positions (sharp: confidence median 0.76, fine) with
  the hard offset-2..k DRAFTS. Split out, the gate is the **offset-1 draft: confidence median 0.30 (diffuse),
  acceptance 3.3%.** offset-2/3 are ~0.1%.
- **The loss is already correct.** Active mode is `hard_teacher_ce` — one-hot CE on the teacher argmax, i.e.
  already a sharpening objective. And `teacher argmax == verify token` ≈ 88.7% (= `teacher_rollout_token_agreement`),
  so the training target ≈ the thing acceptance checks. A reverse-KL / temperature-sharpen / entropy-penalty change
  is circular or contraindicated (it would sharpen a still-diffuse draft toward the same target).
- **It's under-trained.** LR is a flat **1e-6** (grad_norm 200–560). The baseline took ~hundreds of steps at this
  same LR to train the EASIER verify positions (cold→top1 0.74). The fix run's draft positions at step 50 are
  statistically identical to the baseline's NEVER-trained drafts (conf 0.30 vs 0.28; accept 3.3% vs 3.5%), with a
  faint positive slope within the run (offset-1 accept 0.024→0.029, conf 0.295→0.309 over 40 steps) = training
  works, ~100× too slowly.

**⇒ Next experiment (infra-side, NO `opd_loss.py` change):** raise the MTP/draft LR (1e-6 is far too low) and/or
run hundreds–thousands of steps; judge by **offset-1 draft acceptance / offset-1 draft CE** (the script), not the
aggregate top1/ent_stud that caused the misreads.

**Remaining real risk = H3 (architectural), not H1.** Single-shot/parallel MTP drafts from MASK slots and predicts
2+ tokens ahead with no intermediate token (offset-1 effectively predicts t+1 from context ending at t−1). It may
never sharpen enough to match the AR verify. If a properly-resourced run drives offset-1 draft CE down while
acceptance stays pinned → H3 confirmed → the fix is EAGLE-style **autoregressive drafting** (sampler/model change),
still not a loss term.
