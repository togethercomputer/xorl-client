# MTP science — next experiments after k=2 bootstrap success (2026-06-13)

**Context:** k=2 bootstrap validated live — offset-1 draft acceptance 3%→~78%, commit_len 1.03→1.85,
conf 0.29→0.84, threshold unchanged 0.3 (runs 110939Z/194614Z/200325Z). H3 ceiling refuted; replay
transfers to sampler. See `mtp_science_verdict_20260613.md` (top banner). These are the next arms,
ordered by disruption to the shared stack.

**Stack-sharing note:** the samplers are warm at **k=2**. Continuing k=2 needs NO sampler change.
Widening to k>2 requires reprogramming the samplers (`--mtp-static-cuda-graph-k-list`) — coordinate
with the perf agent (it breaks an in-flight k=2 microbenchmark). The perf agent's capped capture runs
(run_num_steps=510) finish quickly; grab the stack between them.

---

## EXP-1 (LOW disruption, do first): continue k=2 — is 1.85 a plateau or still climbing?

**Question:** the k=2 climb was slow-creep-then-phase-transition (~step 500). Is commit_len still
rising past 1.85, or plateaued? This bounds how much pure k=2 buys before we need to widen.

**Config:** identical to the live k=2 run — resume from the latest k=2 checkpoint
(`q36mtp-...-step000500` or later), `OPD_MTP_K_TOKS=2`, `muon_lr=1e-5`, `lr=1e-6`, ConfAdapt thr 0.3,
hard_teacher_ce, supervise_emit_window. **No sampler reprogram** (already k=2). Run ~100–200 steps.

**Readout (every ~20 steps):** `analyze_mtp_draft_acceptance.py --max-offset 1` →
offset-1 acceptance + conf median; `commit_len_steady`; `opd_top1_agreement`; `ent_stud`.
**Stop criteria:** STILL CLIMBING (offset-1 accept slope > 0 over 100 steps) → keep going / k=2 has
more headroom. PLATEAUED (accept flat at ~0.78, commit ~1.85 for ≥100 steps) → k=2 is saturated;
move to EXP-2 (widen). Either way, confirms whether to widen now or train k=2 longer.

## EXP-2 (MED disruption, the decisive widen test): k=2 → k=4 curriculum

**Question:** can offset-2/3 (never supervised under k=2 — only 1 mask slot exists at k=2) be
warmed up by widening? This is the real test of the full MTP objective (multi-token commit).

**Why a curriculum, not a cold k=4:** the original k=4 cold-start is exactly what failed (the stuck
014150Z run). Resume from the **k=2-trained checkpoint** so offset-1 is already strong (78% accept),
then widen so offset-2/3 bootstrap on top of a working offset-1 — instead of three cold offsets
competing for gradient.

**Config:** resume from the best k=2 checkpoint. `OPD_MTP_K_TOKS=4` (trainer) AND sampler
`--mtp-static-cuda-graph-k-list 4` / `--mtp-adaptive-cuda-graph-kmax-list 4` (must match — coordinate
the sampler reprogram). Keep muon_lr=1e-5, lr=1e-6, thr 0.3. Re-derive `static_padded_seq_len` for
k=4 geometry. (Optional intermediate: random k∈[2,4] if a schedule knob exists, closer to the paper.)

**Readout:** `analyze_mtp_draft_acceptance.py --max-offset 3` →
- offset-1 should HOLD ~0.78 (not regress when masks 2/3 are added);
- offset-2/3 acceptance + conf should RISE from ~0 over training (the bootstrap signal);
- commit_len should climb past 1.85 toward 2+.
**Stop criteria:** offset-2 acceptance clear positive slope over ~150 steps → widen further (k=8).
offset-1 REGRESSES when widening → the multi-mask config interferes; back off to k=3 or a slower
curriculum. offset-2 dead-flat while offset-1 holds → offset-2 needs more steps or its own warmup.

## EXP-3 (INDEPENDENT, free nodes, forward-only): offset-2/3 cold-start probe

**Only if the shared stack stays busy.** Load the k=2 checkpoint on 1 node (8 GPU), run the SingleShot
replay forward at k=4 (3 mask slots) on recorded prompts, measure offset-1/2/3 draft argmax-match to
the AR-verify WITHOUT training. Characterizes the EXP-2 starting point (likely confirms offset-2/3 are
cold, since k=2 never presented 2+ mask slots — so mostly a sanity check, lower value than EXP-1/2).
Reuse the `ModelRunner`-load path (it loads cleanly standalone; `experiments/opd_profile/` harness as
a base). No sampler/teacher/weight-sync → zero collision with the shared stack.

## Cross-cutting: judge by the draft side + teacher quality, not loss

`val_loss` (0.22) is driven by easy committed positions — not a draft signal. Promote on per-offset
acceptance + ConfAdapt effective-k + teacher-scored rollout quality (does the accepted multi-token
draft preserve the teacher's intended continuation), per the handoff. Throughput (clean-region /
q-banding) only matters for affording more steps; the step is sampling-bound, so sampling is the
wall-clock lever.
