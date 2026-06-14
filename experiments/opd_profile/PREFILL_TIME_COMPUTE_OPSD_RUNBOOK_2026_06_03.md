# Prefill-Time-Compute OPSD Runbook - 2026-06-03

## Render A Candidate

```bash
python experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py render-control \
  --candidate experiments/opd_profile/autoresearch/candidates/PTC-001.yaml \
  --num-steps 9 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-001-trainer-head.yaml
```

Check the rendered trainer command for:

- `opd_ptc_positive_buffer_kl_weight=1.0`
- `opd_mask_zero_weight_positions=true`
- `opd_teacher_answer_source=sampled` or `gold`
- `eval_num_problems=1000`
- candidate-specific `wandb_group=q36-ptc-autoresearch`

## Launch

Dry run:

```bash
python experiments/opd_profile/autoresearch/controller.py launch --id PTC-001 --dry-run
```

Actual trainer launch:

```bash
python experiments/opd_profile/autoresearch/controller.py launch \
  --id PTC-001 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

The controller calls `stop-trainer-control --remove-run` and then writes trainer control scripts for the candidate. It does not recreate the Kubernetes Pods.

## Monitor

```bash
python experiments/opd_profile/autoresearch/controller.py monitor --profile latest --json
```

This wraps the existing live monitor and appends the snapshot to the autoresearch event log.

## Score And Advance

```bash
python experiments/opd_profile/autoresearch/controller.py score --profile latest --idea-id PTC-001
python experiments/opd_profile/autoresearch/controller.py advance --id PTC-001 --profile latest
```

Scorecards are written under `experiments/opd_profile/autoresearch/scorecards/`. The queue status is advanced only by the `advance` command.

## Manual Analysis

For deeper completed-run analysis:

```bash
python experiments/opd_profile/analyze_slot_profile.py latest \
  --min-control-n 900 \
  --min-answer-logprob-margin 0.0 \
  --min-answer-logprob-z 10.0
```

For W&B/profile consistency:

```bash
python experiments/opd_profile/audit_wandb_profile.py latest --wandb-run <run-id>
```

