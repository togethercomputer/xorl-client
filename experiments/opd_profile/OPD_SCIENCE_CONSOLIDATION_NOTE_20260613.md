# OPD science — consolidation capture note (2026-06-13 ~22:45Z)

For the consolidation (CONSOLIDATION_HANDOFF.md). The perf agent owns the engine/branch
commits; this note inventories the **science-side uncommitted artifacts** from the overnight
OPD wave so they travel correctly to `xorl-client` (Part 2) and nothing is dropped. None of
these are engine files — they don't touch the `runner_dispatcher.py` merge hotspot.

## Findings (already written into the canonical runbooks — verify they survive the move)
- **Science runbook `autoresearch/CANONICAL_SCIENCE_RUNBOOK.md` §1e + §7** — the 2026-06-13
  lever-1 × OPRD-coef wave: lowering prepare temperature is NEGATIVE; OPRD coef peaks at
  coef-1=0.145 but is SFT-dominated; extend-c1 (201 steps) does not reproduce it; **no
  on-policy distillation variant beats answer-only SFT (~0.23) on the floored ops6 task.**
  Surviving lever = correct-prefix filtering (now built, see below).
- **Infra runbook `autoresearch/CANONICAL_INFRA_RUNBOOK.md` §7d-addendum** — measured
  64-prompt buffer-arm throughput (plain step ~25–27 s, run-avg ~36–38 s, 101-step ~61–64 min,
  fb ~17.7 s, ~120 tok/s/GPU — trainer MFU, not allocation, is the residual lever).
- **Ledger `ARITH_OPSD_STATUS_2026_06_11.md` §6** — full per-run table (ARITH-015→020 controls)
  + the quack_linear/SFT incompatibility + the CPF-knob build.

## Uncommitted artifacts in THIS repo (capture / migrate to xorl-client per Part 2)
- `experiments/opd_profile/autoresearch/CANONICAL_SCIENCE_RUNBOOK.md` (M — §1e/§7)
- `experiments/opd_profile/autoresearch/CANONICAL_INFRA_RUNBOOK.md` (M — §7d-addendum; NB also
  carries the perf agent's §0/§7d framing edits — complementary sections)
- `experiments/opd_profile/ARITH_OPSD_STATUS_2026_06_11.md` (M — §6)
- `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py` (M — **`temperature` is now a
  templated candidate knob**; the generator had hardcoded `temperature=1.0` and silently ignored
  the candidate key. Backward-compatible default 1.0.)
- `experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_4node_warm009.yaml` (M — `ce_mode:
  quack_linear`, OPRD/KL path)
- `experiments/opd_profile/configs/qwen3_6_35b_a3b_opd_opdb_4node_adamw.yaml` (M — `ce_mode:
  compiled`; **SFT is INCOMPATIBLE with quack_linear** — needs return_per_token; do not revert)
- New candidates (untracked): `autoresearch/candidates/ARITH-015-WARM-T03`, `-016-OPRD-T03`,
  `-017-OPRD-C3`, `-018-OPRD-C10`, `-019-OPRD-C05`, `-020-OPRD-C1-LONG`, `-021-CPF-OPRD-WARM`,
  `E0A-SFT-LONG`.yaml
- `autoresearch/overnight_supervisor.sh` (untracked — back-to-back queue driver w/ recovery)

## Uncommitted in `xorl-client` (apanda-dev) — the CPF knob (mixed into a larger in-flight diff)
The perf/coordinator should preserve these specific additions when committing
`examples/on_policy_distillation.py` + `tests/test_on_policy_distillation_example.py`:
- Config field **`opd_correct_prefix_only: bool = False`**.
- Param `correct_prefix_only` on `_opd_loss_data(...)` + the masking block (masks the whole
  target `-100` for any sample with `sample_ok != 1`), wired at BOTH call sites.
- Tests: `test_correct_prefix_only_masks_incorrect_and_unknown_samples`,
  `test_correct_prefix_only_off_is_a_no_op` (full file: 69 tests pass).
- Client-side only, no server loss-op change. The §7.1 surviving bootstrap lever.

## Ready-to-run — BUT read §1f first (the depth diagnosis supersedes "CPF is the headline")
`ARITH-021-CPF-OPRD-WARM.yaml` (CPF + OPRD c1 + warm009, T=0.6) remains a clean, ready probe,
but the science agent's **§1f (samples-driven depth diagnosis, 2026-06-13 22:00Z)** reframes its
role: the wall is single-pass **computational depth / answer magnitude** (`|gold|≥10k` solved 0×
across every run; the model regresses magnitude, not value), and **every objective — incl. CPF —
hits it**. §1f.3 predicts CPF will **likely also tie SFT**, because the correct on-policy samples
CPF keeps are ~all small/zero-magnitude (the easy bucket), so it attacks the bootstrap/objective,
not depth. So: run ARITH-021 only to *confirm* that prediction (gate on **per-`|gold|`-bucket
accuracy**, never aggregate — the gold=0 guess attractor masks the hard buckets), not as a
headline win. **The live direction is the science agent's value-grounded PREFILL compute (RiM-style)
— give the buffer an actual serial computation to perform** — which is what my plan called Track A
(latent serial depth). §1f also corrected two of my earlier framings: (b) pause-i↔CoT-i logprob
matching was NOT untested — it = ARITH-007F/010, already falsified; (a) the "predicts only pause"
vacuity is the `insert`+`supervise_student_cot` recipe, not the `replace`+masked answer-only nulls.
Defer to §1f / `FAILURE_MODE_FROM_SAMPLES_2026_06_13.md` over any "next experiment" framing in my
§1e/§7 edits where they differ.

## Memories written (durable across sessions)
`project_opd_correct_prefix_filter_knob`, `project_quack_linear_incompatible_sft`,
`project_opd_generator_temperature_hardcoded`, + the arith-workstream index entry.
