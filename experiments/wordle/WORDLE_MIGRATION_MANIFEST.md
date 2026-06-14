# Wordle harness — migration manifest (2026-06-13)

Scope: the **Wordle OPSD/GRPO science harness** only (my domain). Complements the broader
`CONSOLIDATION_HANDOFF.md` Part 2. Destinations per that handoff:
- harness code + science docs → **`xorl-client/experiments/wordle/`**
- k8s manifests + trainer configs → **`xorl-infra`**
- run artifacts (gitignored) → **`/shared/apanda/<run>/`**

Engine code (`src/xorl/**`, incl. `ops/loss/{opd_loss,importance_sampling_loss}.py`) **stays in
the xorl engine repo** — the harness imports it (`xorl`, `xorl_client`). Do NOT move engine code.

## → xorl-client/experiments/wordle/  (harness code + science)
Core (load-bearing, committed on `exp/opsd-wordle` @ c79d25f2):
- `standalone/tasks/wordle.py` — task: prompts, feedback, candidates, **`extract_action_text`** (the
  multi-turn rollout fix — load-bearing for OPSD AND GRPO).
- `standalone/train_opsd_baseline.py` — OPSD trainer (asymmetric_opsd; teacher styles incl.
  `public_policy_hint` and new `public_candidates_only`; reason-first; full-weight sync).
- `standalone/train_grpo_wordle.py` — GRPO trainer (full-weight-ported; **needs `xorl_client.rl`**
  installed — currently missing; the only blocker to running GRPO).
- `standalone/eval_wordle_sglang.py` — floor-protocol harness eval (the gate; 64 games, seed 777).
- `standalone/test_wordle_opsd_prompts.py` — prompt/parse unit tests (31/31 green).
Supporting science scripts:
- `standalone/probe_wordle_teacher.py` — teacher-prompt logprob probe (no trainer needed).
- `standalone/probe_wordle_enumeration.py` — enumeration probe.
- `standalone/generate_wordle_gold_sft.py` — gold SFT data gen (Kimi / self-scaffold).
- `standalone/generate_wordle_algo_think_gold.py` — algorithmic think-gold gen.
Science docs / runbooks:
- `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` — **authoritative**; read its "🎯 HANDOFF — START HERE".
- `THROUGHPUT_DEBUGGING_HANDOFF.md` — perf/throughput notes (coordinate w/ perf agent; may live in xorl-infra).
- this file.

## → xorl-infra  (k8s manifests + trainer configs)
k8s (q36 = current target; q3-coder-30b = legacy reference):
- Serving: `qwen3-6-35b-a3b-opsd-wordle-{sglang-pool,smg,teacher-sglang}.yaml`,
  `qwen35-35b-a3b-wordle-eval-sglang.yaml`.
- Trainers: `qwen3-6-35b-a3b-opsd-wordle-{think-canonical,reasonfirst-candonly-fwdkl,fwdkl-control,
  guessonly,fullft-8gpu,fullft-4gpu,fullft-2x4,fullft-2x4-think-gang,baseline-4gpu,algosft,algosft-4gpu}.yaml`.
  NOTE: `think-canonical` (reverse-KL + answer-giving teacher) is the KNOWN-COLLAPSING negative
  control; `reasonfirst-candonly-fwdkl` is the live hypothesis (forward-KL + reason-first).
- Legacy 30B: `qwen3-coder-30b-a3b-{opsd,grpo}-wordle-*.yaml`.
configs/ (trainer YAML): `qwen3_6_35b_a3b_opsd_wordle_*.yaml` (the `*_warmstart_sft48*` are the
warm-start-from-SFT variants); legacy `qwen3_coder_30b_a3b_zorl_wordle_*.yaml`.

## → /shared/apanda/<run>/  (gitignored run artifacts — ~2T, reclaim disk)
- `results/opsd_wordle_native_baseline/` — **1.9T** (all OPSD/SFT run dirs incl. checkpoints).
  Warm-start ckpt to KEEP/locate: `…/20260612T160347Z-…sft_gold/server_output/weights/default/step-000048`.
- `results/wordle_gold_sft/` (50M, gold data), `results/wordle_teacher_cot/` (776K, reason-first CoT
  cache `q36_cot_cache_v0_*`), `results/wordle_eval/` (25M, eval transcripts).
- Fix the launch template so future runs write `RESULT_ROOT` under `/shared`, not the checkout.

## Notes
- Working-tree backup of this session's science: `/shared/apanda/consolidation-staging/20260613/wordle/session-2026-06-13-pm/` (patch + files + README).
- **Committed Wordle science @ `fe5b7bc8` (migrate with the harness):** `generate_wordle_algo_think_gold.py`,
  `probe_wordle_enumeration.py`, `k8s/…-algosft.yaml`, `k8s/…-algosft-4gpu.yaml`, and the runbook's
  PROBE RESULT section. These are the enumeration-RETRIEVAL workstream, NOT perf-agent work.
- **Key dataset (→ /shared):** broad algo-think gold `results/wordle_gold_sft/algo_think_v2_broad_20260613T215318Z/
  gold.jsonl` (15,986 turns, full word-list coverage, seed-777 floor-eval held out). The SFT that consumes it
  (`…-algosft-4gpu.yaml`) HANGS at the first forward_backward (4-GPU EP=4 + fp8 + compiled; cold eval=0.443 OK)
  — open item, see runbook. Resume after consolidation with the fp8/compile workarounds.
- `replay_*.py`, `*_quack/_nockpt/_ep1` configs, and the THROUGHPUT doc are the perf agent's / still-uncommitted —
  coordinate before moving those.
