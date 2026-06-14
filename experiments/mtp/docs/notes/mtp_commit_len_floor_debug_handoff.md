# Debug handoff: OPD-MTP `commit_len` won't rise off the floor (~1.0) while loss descends

> **REOPENED 2026-06-12 — the replay-visibility + emit-window fix did NOT lift commit_len in a 50-step A/B**
> (`fix/mtp-replay-visibility-emit-supervision` @ `b0cfc6de`, run `q36mtp-20260612T204931Z-2s1t`): sampler-side
> commit_len flat ~1.03, aggregate top1 ~0.26, ent_stud ~3.0, hard_teacher_ce 6.1→5.4. The two replay bugs were
> real and the fix is correct, but it did not move acceptance in 50 steps.
>
> > **CORRECTION 2026-06-13 — per-offset trace analysis says the 50-step A/B was UNDER-TRAINED, not falsified,
> > and "switch the loss mode" is the wrong lever.** Don't act on the loss-mode prescription below until a
> > properly-resourced run rules out under-training. Evidence (from `rollout_samples.jsonl` via the new
> > `scripts/opd/analyze_mtp_draft_acceptance.py` — the aggregate metrics MISLED everyone, including me twice):
> > - The aggregate `top1`/`ent_stud` are dominated by the verify positions + the hard offset-2..k drafts.
> >   Split out, the **verify positions are sharp (conf median 0.76) and fine**; the **offset-1 DRAFT is diffuse
> >   (conf median 0.30) and accepted 3.3%** — the draft side is the gate, and it is barely trained.
> > - The active loss is ALREADY `hard_teacher_ce` (one-hot CE on the teacher argmax) — i.e. already a sharpening
> >   objective. "Add hard-target CE / reverse-KL / entropy penalty" is therefore circular or chases sharpness the
> >   diffuse draft can't yet supply. Also `teacher argmax == verify token` ~88.7% → the TARGET is fine (teacher ≈
> >   the thing acceptance checks); it is NOT a forward-vs-reverse-KL problem.
> > - **Why under-trained:** LR is a flat **1e-6** (grad_norm 200–560). The BASELINE took ~hundreds of steps at
> >   this same LR to train the EASY verify positions (cold→top1 0.74). The draft positions are HARDER and were
> >   cold at step 0; at step 50 they are statistically identical to the baseline's NEVER-trained drafts
> >   (conf 0.30 vs 0.28, accept 3.3% vs 3.5%) — with a faint positive slope within the run (offset-1 accept
> >   0.024→0.029, conf 0.295→0.309 over 40 steps). That slope = training works, just ~100× too slowly.
> > - **⇒ Next experiment is NOT a loss change (infra-side, no `opd_loss.py` edit):** raise the MTP/draft LR
> >   (1e-6 is far too low) and/or run hundreds–thousands of steps, and judge by **offset-1 draft acceptance /
> >   offset-1 draft CE** (run `analyze_mtp_draft_acceptance.py`), NOT aggregate top1/ent_stud.
> > - **Real remaining risk = H3 (architectural), not H1.** Single-shot/parallel MTP drafts from MASK slots and
> >   predicts 2+ tokens ahead with no intermediate token (offset-1 effectively predicts t+1 from context ending
> >   at t−1), so it may never sharpen enough to match the AR verify. If a properly-resourced run shows offset-1
> >   draft CE falling while acceptance stays pinned → it's H3, and the fix is EAGLE-style AUTOREGRESSIVE drafting
> >   (sampler/model change), not a loss term. A reverse-KL/entropy loss is contraindicated either way.
> > Full data: `mtp_commit_len_floor_root_cause.md` / memory `mtp-commit-len-floor-root-cause`.

**Date:** 2026-06-12 · **Stack:** `er-opd-q36-mtp-ss-0605c` · **Run:** `q36mtp-20260612T095624Z-2s1t`
(WandB `together-research/singleshot`) · **Branch:** `apanda-dev-mtp` (worktree
`/home/apanda/xorl-mtp-singleshot-port-20260602`, has `.venv`).
**Your job:** figure out *why* MTP draft acceptance (`commit_len_steady`) stays pinned at ~1.0 and what
would make it climb. This is the whole point of MTP-OPD — distilling speculation so decode is cheap.

---

## Symptom

Over steps 0→~380 (the run is healthy, checkpoints at 100/200/300, supervisor armed, 0 crashes):

- **`mtp/commit_len_steady` ≈ 1.03–1.05, flat the entire run** (started ~1.04 at step 0, still ~1.04 now).
- **`mtp/effective_k_mean` ≈ 2.89** — the model DOES draft ~3 tokens/step (k_toks=4); they're just rejected.
  So ~1 of ~2.9 drafted tokens is accepted ⇒ **draft acceptance ≈ 35%, the rest argmax-mismatch the verify.**
- **Loss IS descending**: `loss = ce_teach_stud = kl_teach_stud = opd_kl` went ~2.6 (cold) → ~1.2 and is
  leveling. So the supervised objective is improving, but it is **not translating into draft acceptance.**

The SGL agent (who fixed the sampler, sglang `9190ee358`) framed it: hf-exact acceptance commits up to the
first draft mismatch; `commit_len` climbing = distillation working; **pinned at 1 after meaningful training =
the real red flag.** We are now at "meaningful training" (loss halved) with no lift — so this is worth a dig.

## The headline clue: the student stays FLAT while the teacher is PEAKY

From the per-step throughput JSON (trainer-head run.log, `OPD throughput step=N {...}`):

```
ce_teach_stud = kl_teach_stud = opd_kl = loss ≈ 1.23   # the supervised objective (forward KL teacher||student)
ent_teach ≈ 0.45     # frozen teacher (AR, full context): confident / peaky
ent_stud  ≈ 1.40     # student MTP-draft (mask-conditioned): flat / uncertain  <-- 3x the teacher's entropy
effective_k_mean ≈ 2.89   commit_len_steady ≈ 1.04   k_toks = 4
```

hf-exact acceptance needs the **draft's argmax** to equal the verify/teacher token. A distribution can get
closer in KL (loss ↓) while its **top-1 stays unstable** if it remains high-entropy. The student's entropy
(1.40) is ~3× the teacher's (0.45) and is **not collapsing toward the teacher's** — so the draft argmax is
unreliable ⇒ rejected ⇒ `commit_len≈1`. **The loss is mode-covering the student broad, not making it
confident.** That's the leading hypothesis.

## Hypotheses, ranked, with how to test

### H1 (primary): the loss is mode-covering (forward KL) → student never sharpens → bad draft argmax
`ce_teach_stud == kl_teach_stud` ⇒ the objective is **forward KL `KL(teacher‖student)`**, which is
mean-/mode-COVERING: it rewards the student for putting mass on *all* the teacher's modes, i.e. staying
broad. That's the opposite of what acceptance needs (a peaky student whose argmax matches). The persistent
`ent_stud≈1.4 ≫ ent_teach≈0.45` is the signature.
- **Test:** plot `ent_stud` and `ent_stud-ent_teach` vs step (it should be flat/not-shrinking). Confirm the
  loss mode (`OPD_LOSS_MODE` env / `src/xorl/ops/loss/opd_loss.py`; the run's metric also reports
  `hard_teacher_ce` in some configs — check which is actually the trained loss here).
- **Fix directions to try:** add a mode-SEEKING term (reverse KL `KL(student‖teacher)`), or distill on a
  **temperature-sharpened teacher**, or add a **top-1 / hard-target CE** term on the teacher's argmax (directly
  optimizes the thing acceptance checks), or an **entropy penalty** on the student draft. Any of these pushes
  the student peaky. Smallest first: hard-target CE on the teacher argmax, or teacher-temperature < 1.

### H2: only the COMMITTED token is supervised → the k−1 rejected draft positions get no gradient
The Round-3 fix (`d54ae9a1`) set `label_count = min(commit_len_runtime, …)` in
`src/xorl/mtp/singleshot.py` (~line 2489) so the trainer supervises exactly the **actually-committed** tokens
(=1 at `got=1`). If the k−1 *rejected* draft positions (the mask slots the model also predicted) are **not**
in the supervised set, their MTP heads never get gradient → they can't learn to predict acceptably → they stay
rejected → `commit_len` can't rise. **Chicken-and-egg: rejected ⇒ unsupervised ⇒ stays rejected.**
- **Test:** in `singleshot.py`, trace what positions enter the loss vs what the model drafts. Compare
  `mtp/supervised_tokens` (≈3971) to `mtp/generated_tokens` (≈3971) and to `effective_k_mean`×steps — does the
  supervised set cover the **full draft window (k positions per step)** or only the **committed prefix**? If
  only committed, the draft heads for j≥2 are undertrained.
- **Fix direction:** supervise the full k-draft window with the teacher's distribution at each draft offset
  (train all MTP heads), regardless of runtime acceptance — independent of H1. (Reconcile with the Round-3
  validation, which currently asserts committed==target only on the committed prefix.)

### H3: MTP-draft (mask-conditioned, multi-offset) is intrinsically harder than teacher AR → ceiling on argmax-match
The teacher predicts token t from full causal context; the student MTP head predicts t+j from `[…, <think>,
<fim_pad>×k]` mask slots. Predicting offset-2..k from masks is genuinely harder → higher entropy → a real
acceptance ceiling even with perfect distillation. May cap `commit_len` below k regardless of H1/H2.
- **Test:** measure per-offset draft accuracy (offset-1 vs offset-2..k argmax-match to teacher) on a held
  checkpoint. If offset-1 accepts well but 2..k never do, this is the ceiling.

### H4: acceptance/metric plumbing (rule out)
Confirm the sampler's hf-exact acceptance actually *can* accept >1 on this branch (it's new in `9190ee358`).
- **Test:** `scripts/opd/probe_token0_collapse.py` reports per-prompt `commit_len`/trace; run it against a
  sampler **synced to the step-300 student weights** and a hand-crafted prompt where the model is confident
  (e.g. a memorized continuation) — if even there `commit_len==1`, the acceptance path is suspect, not training.

### H5: just needs more steps (least likely)
Loss has *plateaued* (~1.2) while `commit_len` is flat ⇒ more steps of the same objective probably won't move
it. Worth a glance at the trend but don't bank on it.

## Where to look

- **Metrics (per step):** trainer-head run.log lines `OPD throughput step=N {…json…}` —
  `commit_len_steady`, `effective_k_mean`, `ent_stud`, `ent_teach`, `ce_teach_stud`, `kl_teach_stud`,
  `supervised_tokens`, `flops_regret_ratio` (≈23 — the perf cost of commit_len≈1). Also the per-step JSONL
  `…/q36mtp-20260612T095624Z-2s1t/artifacts/opd_profile.jsonl`. WandB project `singleshot`.
- **Loss:** `src/xorl/ops/loss/opd_loss.py` (forward KL / hard_teacher_ce / hidden_match; which is active here?).
- **Supervision / trace replay:** `src/xorl/mtp/singleshot.py` — `label_count`, `_trace_committed_tokens`,
  `commit_slice`/`*_runtime` (lines ~90–130, ~2480–2510); the Round-3 fix is `d54ae9a1`.
- **Sampler acceptance:** sglang `/home/apanda/xorl-sglang-internal` @ `9190ee358` (hf-exact draft acceptance,
  verify-style state); root-cause memo `docs/notes/mtp_token0_collapse_root_cause.md` (xorl `d39a0c91`).
- **Rollout samples (what the student drafts vs commits):** `…/artifacts/rollout_samples.jsonl`
  (`native_mtp_debug_trace`, `commit_lens`, `pending_token_ids` per step — shows drafted-vs-committed directly).
- **Probe tool:** `scripts/opd/probe_token0_collapse.py` (sends native-MTP requests, reports per-prompt commit).
- **Checkpoints to probe (the student at various training amounts):**
  `…/q36mtp-20260612T095624Z-2s1t/server_output/weights/default/{…-best-step000200,…-step000300}` (DCP).

## Suggested first 3 moves

1. **Confirm H1 cheaply:** plot `ent_stud` vs step from `opd_profile.jsonl`. Flat-and-high (≈1.4, not shrinking
   toward 0.45) ⇒ the student isn't sharpening ⇒ the objective is the problem. (~5 min, no GPU.)
2. **Settle H2:** read `singleshot.py` around `label_count` and confirm whether the loss covers the full
   k-draft window or only the committed prefix. If only committed, that's likely *the* bug and is fixable.
3. **Probe per-offset acceptance** on the step-300 checkpoint (H3 vs H1): does offset-1 accept and 2..k never?
   → intrinsic ceiling. Does *nothing* accept even when confident? → H1/loss. Use the rollout trace +
   `probe_token0_collapse.py` against a step-300-synced sampler.

## Don't break

The run is **healthy and training** (loss descending, supervisor armed, checkpointing). `commit_len≈1` makes
it FLOPS-inefficient (`flops_regret_ratio≈23`) but it is producing real, coherent supervision and a descending
loss — it is NOT failing. Debug with **probes / offline checkpoint analysis**; don't perturb the live trainer
or the launch_args without coordinating (the infra/resume agent owns the live stack + the supervisor). If you
want a loss-mode A/B, do it as a *separate* short run, not by editing the live one.
