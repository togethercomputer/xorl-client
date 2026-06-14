# MTP science verdict — why `commit_len` is not lifting (2026-06-13)

> ## ⏩ UPDATE 2026-06-13 ~20:20Z — k=2 BOOTSTRAP VALIDATED LIVE; verdict's core thesis CONFIRMED
>
> The stack advanced well past this doc's `014150Z`/step-200 baseline. The resolved **k=2 bootstrap**
> (Experiment A) was run and **works**, at unchanged ConfAdapt threshold 0.3:
>
> | run (k=2, thr 0.3) | offset-1 acceptance | offset-1 conf median | commit_len |
> |---|---|---|---|
> | `014150Z` (this doc's stuck baseline) | 0.031 | 0.292 | 1.03 |
> | `110939Z` step 520 | 0.747 | 0.865 | — |
> | `194614Z` | 0.814 | 0.826 | 1.72 |
> | `200325Z` | 0.776 | 0.844 | 1.73–1.85 |
>
> Within the long run, offset-1 acceptance creeps 0.024→0.10 (steps 200→480) then sharply climbs to
> ~0.75 by ~step 500 — a bootstrap phase-transition, reproduced across 3 runs. **offset-1 draft
> acceptance: 3% → ~78%. commit_len: 1.03 → 1.85. confidence now correlates with correctness.**
>
> This empirically settles the two open questions:
> - **Architectural ceiling (H3) is REFUTED.** The mask-conditioned offset-1 draft *can* be trained to
>   match the AR-verify (78% accept). The prior "ceiling → EAGLE" verdict was wrong; the lever was the
>   cold-start (k=4 dilution), exactly as argued in §5/§6.
> - **The ~50% replay-vs-sampler forward gap (§5b caveat 1) is BENIGN (outcome A, flatness).** Training
>   on the *current dense-order* replay transferred to the *sampler's* commit_len (1.85). A real
>   forward-kernel ceiling (B) could not have produced that. The 50% was diffuse-draft argmax
>   instability; it vanished as drafts sharpened (conf 0.29→0.84). So the planned offline
>   replay-vs-sampler GPU validation is moot — the live k=2 success is the definitive proof.
>
> **Next science step:** widen the k curriculum (k=2 → [2,4] → [2,8] → [2,16]) from the k=2-trained
> checkpoint; judge offset-2/3 acceptance + teacher-scored quality. Throughput (clean-region/q-banding)
> now matters for affording the token budget — but note the step is **sampling-bound** (FB ~20s
> overlapped vs sample ~85-95s), so the FB/MFU win is hidden; sampling (q-banding / more samplers) is
> the real wall-clock lever. Everything below is the original step-200 analysis, retained for context.


**Author:** science agent (continuation of `docs/notes/mtp_science_agent_handoff.md`)
**Run:** `q36mtp-20260613T014150Z-2s1t` (er-opd-q36-mtp-ss-0605c), resumed from prior run's
step-100 checkpoint, now at **step ~200**.
**Fix repo:** `/home/apanda/xorl-mtp-commitlen-fix-20260612` @ `fix/mtp-replay-visibility-emit-supervision` `b0cfc6de`
**Reference:** `/home/apanda/singleshot`

> One-line verdict: **The wiring is correct** (no separate-draft-module mismatch, target is
> correctly aligned, emit-window supervision is active). The mask-conditioned draft slots are
> **functionally untrained**, and the dominant causes are (1) the run is at **~2% of the paper's
> optimization budget** and (2) **fixed `k=4` diffuses the limited gradient across three
> simultaneously-cold mask offsets**, so the *first* mask slot — the gate for `commit_len` 1→2 —
> never concentrates enough signal to clear the 0.3 ConfAdapt threshold. Next move is the
> resolved **k=2 bootstrap from the step-200 checkpoint**, not "more steps at k=4."

---

## 1. Reference-parity findings (`~/singleshot`)

- **Inference (`generate/base_mtp.py::extend_w_mask`):** to draft `k` tokens the model is fed
  `[last_real_token, MASK×(k-1)]` and reads the last `k` logits. **offset-1 (the first draft) is
  ordinary next-token prediction from a *real* token; only offsets ≥2 are MASK-conditioned.**
- **Training (`pretrain.py:1318-1432`):** one student forward over `[prefix(P), mask(K)]`;
  `pred_pos_mask[:, P-1:P+K]` supervises **every** draft slot (`k_toks = K+1` per region,
  *independent of acceptance*). Target = **teacher argmax under student-forcing**
  (`hard_teacher_supervision=True`, `pt_ce_plus_ent_loss(... labels_teacher=hard_teach_preds ...)`),
  teacher = frozen student-init. ConfAdapt is described as an *inference* policy.
- Successful recipe: random offsets and **random `k∈[2,16]`**; AdamW, 2000 warmup, constant peak
  LR ~1e-5 for the whole model; ~100k iters / ~500M supervised MTP tokens.

## 2. Live-run mechanism audit — the big one (rules out the separate-module mismatch)

The handoff's central open question was whether the deployment-shaped replay "still supplies
gradient to the draft slots that need to learn." **It does.**

- sglang launch (`/shared/opd-control/.../sglang-0/run.sh:152`) has **no `--speculative-algorithm`**.
  The EAGLE3-style `DFlashWorker` (separate 1-layer draft module consuming aux hidden states) in
  `xorl-sglang-internal` is **dead code for this run**. The active path is the fork's custom
  "native MTP" (`--enable-mtp-static-q-len-cuda-graph --mtp-static-cuda-graph-k-list 4`), which
  drafts by running the **main 40-layer model** over `[verified_token, MASK×(k-1)]` and reading the
  main `lm_head` — i.e. the SingleShot scheme.
- The xorl trainer model (`Qwen3_5MoeForCausalLM`) is the **plain 40-layer main model + lm_head;
  it contains no MTP/nextn/draft module**. The SingleShot replay trains the same main-model
  mask-conditioned logits the rollout uses. Weight sync covers these params.
- `validate_native_mtp_trace=true` + profile `replay_trace_covered_all_targets=True`,
  `trace_tokens_unused=2` → the replay forward **reproduces the native rollout's committed
  trajectory**. Train/infer are consistent on the committed path.
- Qwen3.6 *does* ship a built-in MTP module (`text_config.mtp_num_hidden_layers=1`,
  `mtp_use_dedicated_embeddings=False`) — it is simply **not used** here. (The config's
  1152/2048 block is the vision tower; red herring.)

**Conclusion:** there is no "trainer trains X, rollout drafts from Y" mismatch. The gradient
reaches the parameters that produce the rollout drafts.

## 3. Emit-window supervision audit — active and correctly aligned

- `supervise_emit_window=True` by default (`singleshot.py:2687`). Profile `valid/generated ≈ 4.5`
  (e.g. step 190: valid 66332, generated 14563) → the trainer supervises far more than committed
  tokens.
- Label/teacher alignment is correct: both commit and emit paths set
  `labels[dst]=generated[target_offset]` **and** `source_indices[dst]=loss_start+target_offset`
  (same index for label and teacher gather). A slot at trajectory position `p` is supervised to
  predict `generated[p+1]` with the teacher gathered AR-aligned at `p+1`. Empirically confirmed:
  `opd_teacher_rollout_token_agreement ≈ 0.91` — at the supervised positions the teacher argmax
  *does* predict the trajectory token, so the target the student is asked to match is meaningful
  and achievable-with-context.

**The target is not the bug.**

## 4. Latest per-offset metrics (step ~200, `analyze_mtp_draft_acceptance.py`)

```
commit_len dist:  {1: 15183, 2: 486, 3: 26, 4: 1}  (mean 1.0345)
verify-token (pc[0], real-context next-token) confidence: median 0.738   ← healthy
offset-1 draft (pc[1], FIRST mask slot): acceptance 0.031  conf median 0.292
offset-2 draft (pc[2]):                  acceptance 0.002  conf median 0.275
offset-3 draft (pc[3]):                  acceptance 0.000  conf median 0.260
```

Profile trends over steps 100→200 (≈100 steps):
- `loss` **descends** 5.18 → 4.19
- `opd_top1_agreement` **flat** ~0.27 · `student_rollout_token_agreement` **flat** ~0.25 ·
  `teacher_rollout_token_agreement` flat ~0.91 · `ent_stud` **flat** ~3.05 ·
  `commit_len_steady` **flat** ~1.03

**Decomposition (the key read):** `student_rollout_token_agreement ≈ 0.25` ≈ exactly the commit
fraction (1 of 4 slots) ⇒ **commit positions agree ~100%, mask positions ~0%.** Back out from
`top1_agreement≈0.27 = ¼·0.91 + ¾·x` ⇒ **mask-slot student↔teacher argmax agreement ≈ 6%, dead
flat.** `val_loss=0.2245` (best-ckpt metric) vs training loss 4.2 confirms the committed/easy
positions are nearly solved while the mask positions carry ~all the loss — and that **checkpoint
selection is not tracking draft quality at all.**

So `loss` drops almost entirely on the easy committed positions (plus microscopic mass shifts on
mask slots); the mask-conditioned draft is not functionally improving.

## 5. Highest-probability reason deeper drafts are not learning

Ranked, evidence-backed:

1. **Gross under-training + wrong cold-start (primary).** ~11M supervised tokens ≈ **2.2% of the
   paper's ~500M**. With fixed `k=4`, every step trains offsets 1/2/3 *simultaneously*, all cold.
   Their confidences are nearly equal and barely separated (0.293 / 0.275 / 0.260) — the *first*
   mask slot is **not** being preferentially learned, and it sits at 0.292, just under the 0.3
   ConfAdapt gate. `commit_len` cannot reach 2 until offset-1 clears that gate. The paper
   randomizes `k∈[2,16]` (heavy small-`k`/offset-1 coverage); fixed `k=4` is *not* reference parity
   and is the hardest possible cold-start.
2. **Mask-token-embedding optimization asymmetry (secondary, cheap to test).** `mask_token_id=248063`
   is the repurposed `<|fim_pad|>` token — real pretrained embedding, learnable, but in the **AdamW
   group at `lr=1e-5`** while the body trains under **Muon at `muon_lr=1e-3`**. The single most
   important parameter for mask-conditioning crawls 100× slower than the body. The reference trains
   the (fresh) MTP token aggressively and the whole model uniformly at 1e-5. The dead-flat
   `ent_stud` is the yellow flag that a pure k=4 continuation may *not* converge.

Not the cause: separate-module mismatch (§2), target misalignment (§3), frozen/missing mask token
(real learnable token), or loss mode (already hard-teacher CE; do not rewrite per handoff).

## 5b. Is it a "never-confident → never-supervised" cycle? (trace-level audit)

Asked directly: with the 0.3 ConfAdapt gate, is the model never confident enough to draft
multiple tokens, so the deeper slots never get supervision? **Trace-level answer: no — and the
binding gate is not confidence at all.** Three findings from `rollout_samples.jsonl`
(28,837 steady steps):

1. **Supervision is decoupled from acceptance — deeper slots ARE supervised ~every step.** The
   trainer's emit window is sized by `attempt_k` (=k=4), not `commit_len`. Replicating the
   emit-window labeling (`singleshot.py:2687-2705`) on one 256-token generation (254 steps): the
   trainer labels draft depths 0/1/2/3 in **253/252/251/250 of 254 steps** (emit:commit label
   ratio **3.96**). The original "commit-only → deeper slots starved" bug (the literal version of
   the cycle) is fixed.
2. **The 0.3 confidence gate is NOT the bottleneck.** Conditioning offset-1 (first mask) acceptance
   on its own confidence:
   ```
   offset-1 conf ≤0.3     → accepted  266/11221 = 2.4%
   offset-1 0.3<conf≤0.5  → accepted  332/7857  = 4.2%
   offset-1 conf >0.5     → accepted  126/2555  = 4.9%
   ```
   Even **highly confident** offset-1 drafts (>0.5) are accepted only **4.9%** of the time.
   Confidence barely moves acceptance (2.4%→4.9%). Concrete confidently-rejected examples:
   `conf=[1.0, 0.92, 0.04, 0.23] commit_len=1` (offset-1 "pyth" conf 0.92, rejected);
   `[1.0, 0.62, 0.36] commit_len=1`; `[0.99, 0.52, 0.35] commit_len=1`.
3. **The binding gate is the exact-match verify (`adaptive_window_mode="hf_exact"`).** A draft
   commits only if its argmax equals the model's own AR-verify token. The mask-conditioned draft is
   **confidently predicting the WRONG token** — an argmax-correctness problem, not a confidence or
   supervision problem. This is the genuine 2-ahead MTP difficulty: predict t+2 from a mask without
   seeing t+1.

There IS a real inference-side collapse — `effective_k` (drafts the rollout bothers to *propose*)
falls to 1 when recent acceptance is low (2112 collapse-runs, **max run 251 steps** = a whole
generation that never attempted >1 draft; 5476 steps in runs ≥3). But this is a *symptom* of
non-acceptance (an inference-efficiency adaptation), it still proposes offset-1 in 75% of steps,
and it does **not** gate the training supervision.

**Implication for §6:** the success metric for the k=2 experiment should be **offset-1
acceptance / draft-argmax==verify-argmax and mask-slot student↔teacher top1 agreement**, not draft
confidence. Confidence rising without argmax-match would not lift `commit_len`.

## 6. Next MTP-preserving experiment — **Experiment A: k=2 bootstrap** (resolved priority)

Resume from the **current run's** checkpoint (exists now):
`.../q36mtp-20260613T014150Z-2s1t/server_output/weights/default/q36mtp-coderforge-v1-step000200`.
(Do **not** fall back to the old step-100 checkpoint.)

Config deltas vs current (coordinate launch mechanics with the infra agent):
- `OPD_MTP_K_TOKS=2` (trainer) **and** sglang `--mtp-static-cuda-graph-k-list 2`
  / `--mtp-adaptive-cuda-graph-kmax-list 2` (must match, else the cuda-graph capture mismatches k).
- Keep: `hard_teacher_ce`, `supervise_emit_window=True`, native rollout + replay,
  `mask_token_id=248063`, ConfAdapt `0.3`, optimizer/LRs unchanged for the clean A run.
- `static_padded_seq_len` will shrink with k=2 geometry; re-derive (k=2 → 1 mask/window).

**Why:** k=2 has a single mask slot, so 100% of the mask-slot gradient lands on offset-1 — the
exact prerequisite for `commit_len` 1→2.

**Stop criteria / readout** (use `analyze_mtp_draft_acceptance.py --max-offset 1`; ensure
`rollout_samples.jsonl` is captured from the resume step so the early→late slope is measurable —
in the current run early `n=0` made the slope verdict degenerate):
- **SUCCESS:** offset-1 (pc[1]) conf median rises **>0.3** with a clear positive slope, mask-slot
  student↔teacher top1 climbs **>15%**, `commit_len` mean **>1.10**, `ent_stud` starts dropping.
  → then **Experiment B**: widen via `k∈[2,4]→[2,8]→[2,16]` curriculum.
- **PARTIAL:** offset-1 conf/agreement shows a clear positive slope but not yet >0.3 over ~150–300
  steps → keep going / start the curriculum; method is working, just slow.
- **FAILURE:** offset-1 conf/agreement **dead-flat** at k=2 over ≥150 steps → optimization is the
  blocker, not the schedule. Run **Experiment D** before any architectural conclusion: bump the
  embedding/lm_head (mask-token) LR or move the mask-token row to a faster param group, and/or a
  short reference-parity AdamW-uniform-1e-5 param-group diagnostic.

Budget for the diagnostic: ~150–300 steps (k=2 steps are cheaper than k=4). Judge by the **draft
side** (offset-1 confidence slope + teacher-scored quality), never `commit_len` or aggregate
`top1`/`ent_stud` alone.

## 7. Operational notes for whoever launches

- A current-run checkpoint exists (`step000200`, val_loss 0.2245). `best`==`latest` (val_loss is
  driven by easy committed positions, so it is not a draft-quality signal — don't trust it for
  promotion).
- Other agents share this stack; this verdict does **not** unilaterally relaunch the live run.
  The k=2 launch must be coordinated with infra (matching trainer `k=2` and sglang
  `mtp-static-cuda-graph-k-list 2`).
- Lowering ConfAdapt threshold (0.3→0.25) would mechanically raise `commit_len` by accepting the
  0.29-confidence offset-1 drafts, but that is an inference-policy hack that admits low-quality
  drafts; it is not a training fix and degrades trajectory quality.
