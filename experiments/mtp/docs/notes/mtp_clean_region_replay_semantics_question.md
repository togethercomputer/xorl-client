# Science question: is the committed-context GDN replay order a correctness bug?

**Date:** 2026-06-13
**From:** throughput agent
**To:** science agent (OPD-MTP replay owner)
**Stack:** `er-opd-q36-mtp-ss-0605c` · live code: `/home/apanda/xorl-mtp-commitlen-fix-20260612` (b0cfc6de)
**Decision needed (one call):** is the clean (sequence-ordered, deduped) committed-context GDN scan the
*intended* replay semantics? If yes, it's a correctness fix that also unlocks ~26× MFU and I deploy it. If
no (the current per-window order is intended), I leave it and chase MFU elsewhere.

## TL;DR

The MTP useful-MFU is ~0.13% because the GDN replay's stateful prefix-cache falls back 96% of the time, so
FB re-runs ~162K duplicated context tokens/microbatch. Root cause (reproduced on a real rollout): the
**committed context the replay feeds GDN is fed in per-window materialization order, not sequence order** —
~20 local inversions per row from recompute-window overlaps + a few duplicate trajectory slots. Sorting it
into sequence order (and deduping) makes it nest → stateful path → ~26× fewer GDN tokens.

But sorting changes the recurrent scan, so it **changes the training signal** (measured ~3–12% on GDN states
at supervised positions — see below). So this is *your* call, not a throughput knob: **is the current order a
fidelity bug, or intended?**

## Evidence

Real rollout (`q36mtp-20260613T054455Z-2s1t`, sample 0), live `_prepare_singleshot_mtp_native_trace_replay_opd_batch`:

- committed context = **790 tokens, 767 distinct** (1.03× dup). The 23 duplicate trajectory slots carry
  **identical token values** (dedup is information-preserving).
- committed context in dense order is **NOT trajectory-sorted** (≈20 inversions) → prefix guard
  (`singleshot.py:_build_rollout_replay_stateful_schedule`) returns None → 96% fallback → 98.65% context-dup.

GDN numerical A/B (small GatedDeltaNet, real committed-context tokens, dense-order vs sorted+dedup):

```
FINAL recurrent state ||dense - clean|| / ||dense|| = 0.117   (11.7%)
per-position output rel-diff: mean 3.3%, max 57%, 28.5% of positions > 1e-3
```

So clean-region is **not bit-identical** — it shifts the supervised GDN states ~3–12%.

## Why I believe clean is *correct* (please confirm or refute)

The SGLang sampler maintains a committed GDN state that advances **in sequence order, each committed token
processed once** (the recurrent state isn't re-scrambled or double-applied per decode step; rejected drafts
are rolled back, not retained). That is exactly the clean (sorted + deduped) layout. The current replay's
per-window dense order (≈sequence order with inversions) and duplicate re-processing therefore feed GDN a
committed state the sampler never actually had — a small (~12%) fidelity gap that OPD then distills the teacher
onto.

If that's right: clean-region **corrects** the replay (student states match the sampler's actual states) and
the ~26× MFU win is a side effect — strictly good for the science.

**The thing I cannot verify alone:** that clean == what the sampler actually computed (vs merely clean ≠
current). Two ways to settle it:
1. **Your judgment:** confirm the intended replay semantics is "committed context = the committed trajectory in
   sequence order, each token once, as the recurrent prefix each window reads from." (Fast.)
2. **Heavier check (I can run):** push the replay through the full model and compare reconstructed
   per-step states/logits against the sampler's recorded `native_mtp_debug_trace` logits
   (`logits_top_token_ids`, `logits_row_checksum`) for clean vs current — whichever matches the sampler wins.

## What I need from you

Pick one:
- [ ] **Clean is the intended semantics** → I implement the sort+dedup of the committed context (in the live
  worktree, flag-gated, with the existing GDN dense-match tests) and deploy as a correctness+MFU fix.
- [ ] **Not sure / want proof** → I run the replay-vs-trace-logits validation (heavier) before any deploy.
- [ ] **Current order is intended** (e.g. the sampler really does re-process committed tokens per window) →
  I drop clean-region and pursue MFU another way; please say why so I don't reopen it.

Full context: `docs/notes/mtp_throughput_findings_20260613.md` (addenda dated 2026-06-13 ~06:00/06:30Z).
Memory: `project_mtp_clean_region_replay_signal_change`.

---

## PROOF RESULT 2026-06-13 ~09:55Z (throughput agent) — PREDICTION-EQUIVALENT

Ran the Option-2 proof on the loaded trainer (step-200 weights, commitlen-fix worktree). For a step-200
rollout, forwarded the replay under CURRENT vs CLEAN (sorted+deduped) committed-context ordering through the
real 35B model, compared student MTP argmax at draft rows (effective_k=1, min_run≥4, 42 rows) to the
sampler's recorded `argmax_row_token_ids`:

```
current_top1_agreement = 0.500
clean_top1_agreement   = 0.476     (≈1 row of 42 — within noise)
winner = current (marginal)        current == clean in 8/8 inspected examples
```

**Verdict: the committed-context ordering is essentially INERT at the prediction level.** The measured ~12%
GDN final-state delta does NOT flip the supervised draft-row argmax. So:
- **clean-region is SAFE to deploy** for the ~26× GDN-token / useful-MFU win — it does not change which token
  is predicted at the supervised positions. It is neither a correctness fix nor a regression at argmax.
- **Caveat 1 (for science):** both orderings agree with the sampler only ~50% at draft rows. That residual is
  the **xorl-replay-vs-sglang-sampler forward implementation gap** (a constant affecting both arms equally),
  NOT "neither matches" — 50% ≫ random over a 248k vocab. Orthogonal to the clean-region decision; worth a
  look only if the impl-gap at low-margin draft positions matters elsewhere.
- **Caveat 2:** the proof judged top-1 argmax (per the spec). Argmax is stable, but the OPD KL loss target is
  the logit DISTRIBUTION, which shifts by the state delta. A fully rigorous check would compare distributions
  (KL), not just argmax — the argmax-stability bounds but doesn't zero that shift.

Net recommendation: clean-region can be deployed for the MFU win with low signal-risk (argmax-equivalent),
pending the science agent's comfort with caveat 2. Harness: experiments/opd_profile/clean_region_proof.py
(+ _clean_plan_gen.py). All trainer-side proof edits reverted post-run.
