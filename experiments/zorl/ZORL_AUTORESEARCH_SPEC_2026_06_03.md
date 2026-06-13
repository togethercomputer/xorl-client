# ZORL Autoresearch Spec - 2026-06-03

## State Files

- `experiments/zorl/autoresearch/ideas.yaml`: persistent science queue.
- `experiments/zorl/autoresearch/candidates/*.yaml`: runnable candidate definitions.
- `experiments/zorl/autoresearch/runs.jsonl`: launch and score events.
- `experiments/zorl/autoresearch/renders/`: generated Kubernetes manifests.
- `experiments/zorl/autoresearch/scorecards/`: generated JSON and Markdown scorecards.

## Candidate Schema

Required:

- `id`: stable candidate id.
- `kind`: `sglang_controller`, `smoke`, or `countdown`.
- `base_manifest`: Kubernetes YAML to render.

Common fields:

- `container`: container name to receive env/CLI patches.
- `k8s.service_name`: service name for multi-document SGLang manifests.
- `k8s.app_label`: app label used by Job pods and Service selectors.
- `k8s.node_name`: exact node name for long-lived SGLang pods.
- `env`: container environment overrides.
- `cli_args`: extra harness CLI flags injected before `${EXTRA_ARGS[@]}`.
- `result_root`: local result root used by `score --result latest`.
- `score_source`: `smoke_summary` or `countdown_log`.
- `score_gates`: thresholds for status advancement.

## Controller Commands

```bash
python experiments/zorl/autoresearch/controller.py next
python experiments/zorl/autoresearch/controller.py render --id ZORL-001
python experiments/zorl/autoresearch/controller.py launch --id ZORL-001 --dry-run
python experiments/zorl/autoresearch/controller.py monitor --idea-id ZORL-001 --result latest
python experiments/zorl/autoresearch/controller.py score --idea-id ZORL-001 --result latest
python experiments/zorl/autoresearch/controller.py advance --id ZORL-001 --result latest
```

Infrastructure candidates can be rendered directly without entering the science queue:

```bash
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-047.yaml
```

Override a trainer's SGL service at render/launch time:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-001 \
  --set-env INFER_URL=http://zorl-ar-sglang-117.apanda.svc.cluster.local:30000
```

## Scoring Gates

Smoke summaries produce:

- `smoke_pass`: all expected records exist, update norms are nonzero, and exported checkpoint metadata has ZORL enabled.
- `infra_invalid`: missing exports, missing ZORL metadata, or zero update norms.
- `incomplete`: fewer records than the smoke run requested.

Countdown logs produce:

- `infra_invalid`: fatal harness error before any usable score, or failed sync with no parent/sampler score.
- `incomplete`: no exact score, or too few completed generations for the candidate gate.
- `strong_zorl_signal`: best exact count meets `strong_exact`.
- `promote_retest`: best exact count meets `promote_exact`.
- `weak_zorl_signal`: best exact count beats the baseline or meets `weak_exact`.
- `science_reject`: completed cleanly without beating baseline.
- `inconclusive`: partial usable score that misses completion and rejection gates.
