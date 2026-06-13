# ZORL Autoresearch Comprehensive Runbook - 2026-06-04

Last updated: 2026-06-04T17:53:24Z

This document is the self-contained operating and research runbook for the
current ZORL autoresearch loop in:

```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated
```

It covers the research goal, current architecture, all run results observed so
far, the mechanics for bringing up the SGLang/SMG serving pool and trainer
jobs, how to score and advance the autoresearch queue, and the current
interpretation of what has and has not worked.

## Research Goal

The immediate research goal is to determine whether the ZORL evolution-strategy
update path can produce reliable, repeatable improvements on the 32-example
Countdown task using Qwen3-30B-A3B with LoRA perturbations and rollout-based
reward. The operational goal is to iterate overnight without repeatedly
relaunching expensive model servers.

The loop is testing these questions:

- Can parent weights improve over the fixed baseline exact score of `1/32`?
- Which knobs make that improvement repeatable: rollout averaging, number of
  perturbation pairs, learning rate / update scale, validity/value shaping, or
  optimizer momentum?
- Can the system separate real parent improvement from noisy best-candidate
  spikes?
- Can malformed Countdown expressions be reduced without destroying exact
  reward signal?
- Can the serving pool be driven at a sane throughput instead of spending hours
  per generation window?

The primary success signal is parent-probe exact score, not only best candidate
exact score. Best-candidate spikes can be useful, but parent probes tell us
whether the ES update moved the trainable parent in the right direction.

## Current State

As of this update:

- Long-lived SGLang workers are running for nodes `001`, `047`, and `117`.
- SMG is running in front of those workers with round-robin routing.
- Trainers use direct SGL URLs for control/sync and use SMG as the routed reward
  generation endpoint.
- `ZORL-009` is the active trainer run.
- `ZORL-010` is queued as the next throughput-isolation run.

Active run:

```text
ZORL-009
job: zorl-ar-zorl-009-vdxds
pod: zorl-ar-zorl-009-vdxds-5v2c2
state: Running, ready=true, restarts=0
run_dir: experiments/zorl/results/zorl_autoresearch/ZORL-009/20260604T155758Z-zorl-ar-zorl-009-vdxds-5v2c2-trainer-only
current parsed result: generation 1, parent probe 3/32, candidate 1.25/32
```

Current queue source of truth:

```bash
experiments/zorl/autoresearch/ideas.yaml
```

Scorecards:

```bash
experiments/zorl/autoresearch/scorecards/
```

## Architecture

The autoresearch setup separates model serving from trainer jobs.

Long-lived serving:

- `SGL-001`: TP=2 SGLang worker on `research-common-h100-001.cloud.together.ai`
- `SGL-047`: TP=2 SGLang worker on `research-common-h100-047.cloud.together.ai`
- `SGL-117`: TP=2 SGLang worker on `research-common-h100-117.cloud.together.ai`
- `SMG-ZORL`: CPU-side SMG router with round-robin policy over the three SGL
  workers.

Trainer jobs:

- Short-lived Kubernetes Jobs.
- Use 4 trainer GPUs on the scheduled node.
- Start a local xorl training server.
- Register and sync LoRA adapters against direct SGLang worker URLs.
- Send rollout scoring/generation traffic through SMG:
  `http://zorl-ar-smg.apanda.svc.cluster.local:30000`.

Why SMG matters:

- One occupied GPU does not imply serving must be down.
- The serving pool is long-lived and trainer-only jobs come and go.
- SMG gives one stable routed reward endpoint while direct worker services remain
  available for sync/control paths.

Important service URLs:

```text
http://zorl-ar-sglang-001.apanda.svc.cluster.local:30000
http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000
http://zorl-ar-sglang-117.apanda.svc.cluster.local:30000
http://zorl-ar-smg.apanda.svc.cluster.local:30000
```

## GPU Scheduling Requirement

Every pod requesting GPUs must have the pod-template label:

```yaml
team: turbo
```

For Jobs, Deployments, and StatefulSets this belongs under:

```yaml
spec:
  template:
    metadata:
      labels:
        team: turbo
```

Do not manually override `schedulerName` or Volcano queue labels unless there is
a specific reason. Kyverno injects those from the team label.

Always inspect rendered GPU manifests before launching:

```bash
manifest=$(ls -t experiments/zorl/autoresearch/renders/*zorl-009.yaml | head -n1)
python - <<'PY' "$manifest"
import sys, yaml
obj = yaml.safe_load(open(sys.argv[1]))
print(obj["spec"]["template"]["metadata"]["labels"])
PY
```

Use `kubectl create --dry-run=server`, not `kubectl apply --dry-run=server`, for
manifests that use `generateName`.

```bash
kubectl create --dry-run=server -f "$manifest"
```

## Important Files

Autoresearch controller and queue:

```text
experiments/zorl/autoresearch/controller.py
experiments/zorl/autoresearch/ideas.yaml
experiments/zorl/autoresearch/runs.jsonl
experiments/zorl/autoresearch/candidates/
experiments/zorl/autoresearch/renders/
experiments/zorl/autoresearch/scorecards/
```

Serving manifests:

```text
experiments/zorl/autoresearch/candidates/SGL-001.yaml
experiments/zorl/autoresearch/candidates/SGL-047.yaml
experiments/zorl/autoresearch/candidates/SGL-117.yaml
experiments/zorl/autoresearch/candidates/SMG-ZORL.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-smg-router.yaml
```

Countdown trainer harness:

```text
experiments/zorl/run_countdown_test.py
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-countdown-trainer-only.yaml
```

Older related docs:

```text
experiments/zorl/ZORL_HANDOFF_RUNBOOK_2026_06_03.md
experiments/zorl/ZORL_AUTORESEARCH_RUNBOOK_2026_06_03.md
experiments/zorl/ZORL_AUTORESEARCH_SPEC_2026_06_03.md
experiments/zorl/ZORL_AUTORESEARCH_RESEARCH_MEMO_2026_06_03.md
experiments/zorl/standalone/RUNBOOK.md
```

## Operating The Serving Pool

Render an SGL worker:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-047.yaml
```

Launch the rendered manifest after inspection:

```bash
manifest=$(ls -t experiments/zorl/autoresearch/renders/*sgl-047.yaml | head -n1)
kubectl create --dry-run=server -f "$manifest"
kubectl create -f "$manifest"
```

Repeat for `SGL-117` and `SGL-001` if they are not already running.

Check SGL workers:

```bash
kubectl get pods -n apanda -l app=zorl-ar-sglang-047 -o wide
kubectl logs -n apanda -l app=zorl-ar-sglang-047 --tail=80
```

Probe from inside the cluster:

```bash
kubectl run -n apanda -it --rm zorl-sgl-probe \
  --image=curlimages/curl --restart=Never -- \
  http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000/v1/models
```

Render and launch SMG:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SMG-ZORL.yaml

manifest=$(ls -t experiments/zorl/autoresearch/renders/*smg*.yaml | head -n1)
kubectl create --dry-run=server -f "$manifest"
kubectl create -f "$manifest"
```

Check SMG routing:

```bash
kubectl logs -n apanda -l app=zorl-ar-smg --tail=120 \
  | rg 'Worker selected|Backend request completed|ERROR|WARN|422|503|504'
```

Healthy SMG logs should show `Worker selected` and `Backend request completed`
with `status=200` across `zorl-ar-sglang-001`, `047`, and `117`.

## Launching Trainer Candidates

List the next queued candidate:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py next
```

Render a candidate:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py render --id ZORL-009
```

Dry-run the rendered Job:

```bash
manifest=$(ls -t experiments/zorl/autoresearch/renders/*zorl-009.yaml | head -n1)
kubectl create --dry-run=server -f "$manifest"
```

Launch:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py launch --id ZORL-009
```

Check the trainer pod:

```bash
kubectl get pods -n apanda -l job-name=<job-name> -o wide
kubectl get pod -n apanda <pod-name> \
  -o jsonpath='{.status.phase}{"\n"}{range .status.containerStatuses[*]}{.name}{" ready="}{.ready}{" restarts="}{.restartCount}{"\n"}{end}'
```

Trainer startup logs should show:

```text
Direct SGLang ... reachable after 1 polls.
Routed reward endpoint ... registered a model after 1 polls.
Running ZORL password test...
ZORL training ...
```

## Monitoring Without Burning Context

Use `wait-eval` to wake only on useful boundaries:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py wait-eval \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log" \
  --poll-seconds 60 \
  --min-generation-delta 5 \
  --min-step-delta 5 \
  --min-record-delta 1 \
  --wake-on-probe
```

This prints JSON only when one of these happens:

- completed generation count advances enough,
- parent/probe rows increase,
- parsed record count advances enough,
- terminal or failure verdict appears,
- timeout fires, if configured.

For a manual snapshot:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py monitor \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log" \
  --json
```

Do not rely on `--result latest` if several runs exist under one candidate root.
Use the explicit `job.log` path from the active run directory.

## Scoring And Advancing

Write a scorecard without changing queue state:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py score \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log"
```

Advance queue state from a score:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py advance \
  --id ZORL-009 \
  --result "$run_dir/job.log"
```

Statuses used so far:

- `complete`: infra smoke passed.
- `weak_signal`: exact improved over baseline but did not pass promote gate.
- `promote_retest`: best parent or best exact reached promote threshold.
- `rejected`: manual science rejection after comparison to stronger prior runs.
- `launched`: active run.
- `queued`: not launched yet.

The controller's generic verdict may say `weak_zorl_signal`; still record a
manual `science_reject` in `ideas.yaml` when the result is a regression against a
better controlled predecessor.

## Stopping Trainer Jobs

Stop only trainer Jobs when a science decision is made. Leave SGL/SMG running.

```bash
kubectl delete job -n apanda <trainer-job-name> --ignore-not-found
```

Examples already used:

```bash
kubectl delete job -n apanda zorl-ar-zorl-005-szpgk --ignore-not-found
kubectl delete job -n apanda zorl-ar-zorl-006-wpd2k --ignore-not-found
kubectl delete job -n apanda zorl-ar-zorl-007-7cnqf --ignore-not-found
kubectl delete job -n apanda zorl-ar-zorl-008-ncfl7 --ignore-not-found
```

## Results So Far

All exact counts are over 32 Countdown prompts unless otherwise noted.

| ID | Status | Key recipe | Best candidate | Best parent | Gen | Verdict / decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ZORL-000 | complete | Multi-session plumbing smoke | n/a | n/a | n/a | Smoke passed: records, updates, and exports present |
| ZORL-001 | promote_retest | lr 1e-2, sigma 0.1, pairs 8, rollouts 4 | 2.5/32 | 3/32 | 16 | First credible Countdown signal, but malformed outputs persisted |
| ZORL-005 | weak_signal | ZORL-001 plus validity/value shaping and intended max update cap | 1.75/32 | 1/32 | 8 | Weak; shaping without effective update shrink regressed parent |
| ZORL-006 | promote_retest | ZORL-005 shaping, lr shrunk to 5.4e-3 | 2.25/32 | 3/32 | 8 | Best completed branch so far; parent promote by gen8 |
| ZORL-007 | rejected | ZORL-006 with validity 0.30, value 0.05 | 1/32 | 1/32 | 1 | Rejected early; heavier validity shaping regressed exact |
| ZORL-008 | rejected | ZORL-006 with rollouts 8 and steps 96 | 2/32 | 2/32 | 8 | Failed retest; deeper rollout averaging underperformed ZORL-006 |
| ZORL-009 | launched | ZORL-006 with pairs 16, rollouts 4 | 1.25/32 | 3/32 | 1 | Active; strong early parent probe, needs gen8 validation |
| ZORL-010 | queued | ZORL-009 with score workers 16 | n/a | n/a | n/a | Throughput-isolation follow-up |

Detailed scorecard metrics:

| Scorecard | Verdict | Gen | Best exact | Candidate | Parent | Rollout exact max | Update norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260604T002324Z-zorl-000-smoke-pass | smoke_pass | n/a | n/a | n/a | n/a | n/a | n/a |
| 20260604T053142Z-zorl-001-promote-retest | promote_retest | 16 | 3/32 | 2.5/32 | 3/32 | 0.0293 | 1183.18 |
| 20260604T080330Z-zorl-005-weak-zorl-signal | weak_zorl_signal | 8 | 1.75/32 | 1.75/32 | 1/32 | 0.0283 | 1183.76 |
| 20260604T103449Z-zorl-006-promote-retest | promote_retest | 8 | 3/32 | 2.25/32 | 3/32 | 0.0278 | 1183.11 |
| 20260604T155708Z-zorl-008-weak-zorl-signal | weak_zorl_signal | 8 | 2/32 | 2/32 | 2/32 | 0.0298 | 1183.18 |
| ZORL-009 live monitor | incomplete | 1 | 3/32 | 1.25/32 | 3/32 | 0.0181 | 836.86 |

## Interpretation

### What Looks Promising

`ZORL-006` is the best completed science branch. It combines:

- LR shrink to `5.4e-3`,
- sigma `0.08`,
- pairs `8`,
- rollouts per puzzle `4`,
- validity weight `0.15`,
- value weight `0.10`,
- momentum `0.0`.

This reached parent `3/32` by generation 8, earlier than ZORL-001's generation
16 promote signal.

`ZORL-009` is the best active hypothesis. It keeps the ZORL-006 reward and LR
balance, but spends the same 4096-rollout generation budget on more perturbation
pairs:

```text
ZORL-006: 8 pairs x 4 rollouts
ZORL-008: 8 pairs x 8 rollouts
ZORL-009: 16 pairs x 4 rollouts
```

ZORL-009 gen1 already has parent `3/32` and update norm around `837`, which is
both better parent signal and smaller update norm than the repeated `~1183`
updates in earlier runs. Do not declare it solved at gen1; the important next
gate is whether it holds or improves at gen8.

### What Did Not Work

The max-update cap in ZORL-005 did not work as intended. The deployed
`/home/apanda/xorl-internal` server accepts the extra field without applying it,
so `max_update_norm=800` was ignored and update norm stayed around `1183`.
Learning-rate shrink was used as the effective workaround in ZORL-006.

Heavier validity shaping did not work. ZORL-007 increased validity weight from
`0.15` to `0.30` and reduced value weight from `0.10` to `0.05`; exact signal
regressed immediately to baseline-level `1/32`, with no meaningful validity
improvement.

Deeper rollout averaging did not work as a retest. ZORL-008 doubled rollouts per
puzzle from `4` to `8` with the ZORL-006 recipe, but only reached `2/32` by gen8
and did not reproduce ZORL-006's parent `3/32`.

Malformed-expression drift remains unresolved. Runs with exact improvements still
produce many outputs with duplicated fragments, comments, invalid parentheses,
extra equations, and non-expression text. Prefix scores remain `0.0000` in the
logged best candidates and parent probes.

### Throughput Finding

The current Countdown scoring path underfeeds the serving pool.

Measured examples:

```text
ZORL-008 gen1: score_rollouts=4096, t_score=1975.6s, rollouts_per_s=2.07
ZORL-008 gen4: score_rollouts=4096, t_score=2179.2s, rollouts_per_s=1.88
ZORL-008 gen8: score_rollouts=4096, t_score=2237.2s, rollouts_per_s=1.83
ZORL-009 gen1: score_rollouts=4096, t_score=2023.1s, rollouts_per_s=2.02
```

This is not because three SGL nodes cannot serve Qwen3-30B-A3B. SMG logs show
round-robin routing and 200 responses across the workers. The likely limiter is
the Countdown scoring harness:

- `ZORL_SCORE_MAX_WORKERS` is `4` in all current Countdown candidates.
- Traffic goes through many chat-completions calls.
- Batch/concurrency is too low for the live SGL pool.

`ZORL-010` is queued to isolate this: same recipe as ZORL-009, but
`ZORL_SCORE_MAX_WORKERS=16`.

## Candidate Queue

Current active/queued candidates:

| ID | State | Purpose |
| --- | --- | --- |
| ZORL-002 | queued | Older more-pairs recipe, but it changes LR/rollouts and lacks ZORL-006 shaping |
| ZORL-003 | queued | Momentum retest |
| ZORL-004 | queued | Long scale retest of older baseline |
| ZORL-009 | launched | Best active science run: pairs 16, rollouts 4, ZORL-006 shaping |
| ZORL-010 | queued | Same as ZORL-009 but score workers 16 for throughput |
| ZORL-OPD-000 | queued | Standalone OPD multiplication smoke |
| ZORL-OPD-001 | queued | OPD 3-digit multiplication science run after smoke |

Recommended ordering:

1. Let ZORL-009 reach gen8/parent-probe unless it fails.
2. If ZORL-009 holds parent `>=3/32` or improves, promote it as the current best
   science branch.
3. If ZORL-009 is scientifically good but still scores at `~2` rollouts/s, run
   ZORL-010 before longer overnight sweeps.
4. If ZORL-009 regresses to parent `<=2/32`, do not spend more on pair count yet;
   revisit whether ZORL-006 was variance or whether parent probe sampling is too
   noisy.
5. Do not run ZORL-007-style heavier validity shaping again unless the objective
   changes from exact improvement to output-format repair.

## Decision Rules At Eval Boundaries

At gen8 or a new parent probe:

- Parent `>=3/32`: promote/retest. This matches the promote gate and is the most
  important signal.
- Candidate `>2.25/32` with parent still `2/32`: weak-to-interesting, but do not
  promote without a parent probe unless the run is still cheap to continue.
- Parent `<=2/32` and candidate `<=2/32`: reject or mark weak; stop the trainer
  to save GPUs.
- Prefix still `0`: note that exact improved but output protocol remains broken.
- Throughput still `~2` rollouts/s: run or prioritize a score-worker/concurrency
  follow-up before scaling steps.

When stopping a trainer, do not delete SGL or SMG.

## Known Issues And Workarounds

### `max_update_norm` Is Ignored

Symptom:

```text
ZORL-005 passed max_update_norm=800 but update_norm stayed around 1183.
```

Cause:

The deployed server ignores that extra API/config field.

Workaround:

Express the intended update shrink through the honored learning-rate path.
ZORL-006 changed LR from `8e-3` to `5.4e-3`.

### Result Path Ambiguity

`monitor --result latest` can select a stale log if several runs live under the
same candidate root.

Use an explicit log path:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py monitor \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log" \
  --json
```

### Parent Probe Rows Can Lag

Sometimes a generation summary lands before the parent probe row. After an eval
boundary, wait briefly and run `monitor` again before deciding whether parent
improved.

### Scoring Is Slow

Current observed end-to-end throughput is about `2` scored rollouts/s for
4096-rollout generation windows. This makes gen8 decisions take hours.

Use `wait-eval` instead of manual polling. Use ZORL-010 or a similar
concurrency-only candidate to test whether higher `ZORL_SCORE_MAX_WORKERS`
improves throughput.

### Malformed Countdown Outputs

The model often emits:

- duplicated expression fragments,
- invalid parentheses,
- explanatory text,
- `= 24` decorations,
- numbers not in the prompt,
- expressions that accidentally solve while violating format expectations.

This is not solved by heavier validity shaping so far.

## Useful Current Commands

Active ZORL-009 monitor:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py monitor \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log" \
  --json
```

Attach low-noise watcher:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py wait-eval \
  --idea-id ZORL-009 \
  --result "$run_dir/job.log" \
  --poll-seconds 60 \
  --min-generation-delta 5 \
  --min-step-delta 5 \
  --min-record-delta 1 \
  --wake-on-probe
```

Trainer health:

```bash
kubectl get pod -n apanda zorl-ar-zorl-009-vdxds-5v2c2 \
  -o jsonpath='{.status.phase}{"\n"}{range .status.containerStatuses[*]}{.name}{" ready="}{.ready}{" restarts="}{.restartCount}{"\n"}{end}'
```

SMG activity:

```bash
kubectl logs -n apanda -l app=zorl-ar-smg --since=5m --tail=160 \
  | rg 'Worker selected|Backend request completed|ERROR|WARN|422|503|504'
```

Latest scoring summaries:

```bash
run_dir=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-009/* | head -n1)
rg -n '^\\s+Generation|^\\s+Parent probe|Traceback|ERROR|Exception|422|503|504' \
  "$run_dir/job.log" "$run_dir/letter_count_test.log" "$run_dir/server.log" \
  | tail -n 120
```

Stop active trainer only:

```bash
kubectl delete job -n apanda zorl-ar-zorl-009-vdxds --ignore-not-found
```

## How To Add A New Candidate

Add a candidate YAML under:

```text
experiments/zorl/autoresearch/candidates/
```

Then add an idea entry to:

```text
experiments/zorl/autoresearch/ideas.yaml
```

Use `append-idea` for simple entries:

```bash
PYTHONPATH=src python experiments/zorl/autoresearch/controller.py append-idea \
  --id ZORL-011 \
  --candidate candidates/ZORL-011-some-recipe.yaml \
  --hypothesis "Short falsifiable hypothesis." \
  --rationale "Why this is the next controlled lever." \
  --priority 79
```

For controlled science, change one main variable at a time. The strongest current
baseline to branch from is ZORL-006/ZORL-009, not ZORL-001.

## Current Working Hypotheses

Most promising:

- More perturbation-pair coverage is better than deeper rollout averaging for
  this task and current reward.
- LR shrink is the practical way to reduce update size until `max_update_norm`
  is implemented or accepted by the server.
- Small validity/value shaping can coexist with exact reward, but only at the
  ZORL-006 weights (`0.15` validity, `0.10` value).

Likely false or currently unsupported:

- More rollout averaging alone improves the parent update.
- Heavier validity shaping improves exact while fixing malformed output.
- The serving pool is saturated by model size; observed throughput points to
  low scorer concurrency instead.

Open:

- Whether ZORL-009's gen1 parent `3/32` persists at gen8.
- Whether `ZORL_SCORE_MAX_WORKERS=16` improves rollouts/s without increasing
  failures or hurting determinism.
- Whether a separate output-protocol repair objective is needed after exact
  signal is reliable.
- Whether OPD multiplication should be launched after Countdown throughput is
  fixed or in parallel as a separate task family.

## Completion Checklist For Future Operators

Before calling a run successful:

- Score with an explicit `job.log` path.
- Confirm parent probe rows, not only candidate exact.
- Inspect at least one best-candidate block for malformed outputs.
- Confirm pod restarts are zero or explain restarts.
- Confirm SGL/SMG logs do not show sustained 422/503/504 errors.
- Write or update the scorecard.
- Update `ideas.yaml` with status, verdict, reason, result path, and scorecard.
- Stop trainer jobs that have reached a decision.
- Leave long-lived SGL/SMG up for the next candidate.
