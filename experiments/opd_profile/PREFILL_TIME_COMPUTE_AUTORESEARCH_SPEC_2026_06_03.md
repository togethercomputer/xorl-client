# Prefill-Time-Compute Autoresearch Spec - 2026-06-03

## Goal

Build an autoresearch programme that can make larger, hypothesis-level changes toward latent or encoded reasoning in the model, with automatic launch, monitoring, scoring, and queue advancement.

The primary target is OPSD-style prefill-time-compute distillation. The teacher sees CoT before the filler span; the student sees filler tokens; the objective is evaluated only on student filler and answer positions.

## State Files

- `experiments/opd_profile/autoresearch/ideas.yaml`: persistent idea queue.
- `experiments/opd_profile/autoresearch/candidates/*.yaml`: runnable candidate definitions.
- `experiments/opd_profile/autoresearch/runs.jsonl`: launch and score events written by the controller.
- `experiments/opd_profile/autoresearch/scorecards/`: generated JSON and Markdown scorecards.

The queue is append-only in spirit. Rejected ideas stay in the file with their verdict and rationale so that later ideas can refer to them.

## Candidate Schema

Required:

- `id`: stable candidate id, for example `PTC-001`.
- `base_config`: historical launcher substrate, normally `AN` for the current Q36 35B slots.
- `trainer_config`: xorl trainer YAML path.
- `buffer_label`: run-name label.
- `student_prefill_text` or `student_prefill_token_ids`.
- `student_prefill_count`.
- `teacher_cot_json_path`.
- `prompts_json_path`.
- `num_prompts`.
- `client_args`: extra `on_policy_distillation.py` config overrides.

Common autoresearch defaults:

- `default_num_steps`
- `default_prompts_per_step`
- `eval_num_problems`
- `eval_accuracy_every`
- `eval_control_start_step`
- `wandb_group`

The generator preserves the historical `--config A..AO` path but also accepts `--candidate <yaml>` for render, full control, and trainer-only control commands.

## Controller Commands

```bash
python experiments/opd_profile/autoresearch/controller.py next
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 --dry-run
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 --sampler-replicas 2 --sampler-layout spare-teacher1
python experiments/opd_profile/autoresearch/controller.py monitor --profile latest --json
python experiments/opd_profile/autoresearch/controller.py score --profile latest --idea-id PTC-001
python experiments/opd_profile/autoresearch/controller.py advance --id PTC-001 --profile latest
```

Launch uses the reusable slot controller. It stops only trainer slots, writes a candidate-driven trainer control script, and leaves sampler/teacher slots reusable.

## Scoring Gates

The controller emits one of these verdicts:

- `infra_invalid`: sync failed, endpoint sync failed, request failures occurred, sampler quiesce failed, cap hits are too high, or answer-logprob evaluation failed.
- `incomplete`: no usable control row or too few scored control examples.
- `strong_ptc_signal`: pause beats no-pause and corrupt controls with significant accuracy and answer-logprob support.
- `promote_retest`: accuracy delta is promising and answer-logprob support is strong enough for immediate retest.
- `weak_ptc_signal`: answer-logprob support is strong and accuracy delta is nonnegative, but not enough for promotion.
- `science_reject`: completed cleanly but does not support the hypothesis.
- `inconclusive`: completed enough to inspect, but missed both reject and promotion thresholds.

Default eval size is `N=1000`. Anything below 900 scored control examples is `incomplete` unless the controller threshold is explicitly lowered.

## First Programme

The first queue is deliberately focused:

1. `PTC-001`: pure positive OPSD KL on filler plus sampled answer, prompt positions masked out.
2. `PTC-002`: same as PTC-001 but replace sampled answer tail with gold answer before teacher KL.
3. `PTC-003`: add filler hidden matching to the gold-answer objective.
4. `PTC-004`: batch/eval scale-up of the best earlier PTC candidate.

Filler type, decode compression, and prefill-length scaling remain in the backlog, but they should not dominate until there is a causal signal worth amplifying.

