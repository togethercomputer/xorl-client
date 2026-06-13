# ZORL Consolidated Handoff Runbook

Date: 2026-06-03

This is the single handoff document for the consolidated ZORL worktree.
It reconciles the prior ZORL worktree, the standalone dispatch worktree,
the ZORL autoresearch queue, and the latest `origin/apanda-dev` changes.

## 0. Branch State

Worktree:

```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated
```

Branch:

```bash
git rev-parse --abbrev-ref HEAD
# codex/zorl-consolidated-apanda-dev-20260603
```

Base:

```bash
git rev-parse HEAD
# b909c549376d02af84eb88c3aabe2631d8e0d318
```

That HEAD is `origin/apanda-dev` after the 2026-06-03 refresh. The
consolidated ZORL changes are staged on top of it; no commit has been
made in this worktree. There should be no merge in progress, no
untracked files, and no unstaged changes.

Check before doing anything else:

```bash
git status --short --branch --untracked-files=all
git rev-parse -q --verify MERGE_HEAD || true
git diff --name-only
```

## 1. Source Inputs

Consolidated inputs:

- `/home/apanda/zorl`: original ZORL feature worktree, including tracked
  ZORL server/runtime changes, experiment configs, k8s manifests, and
  the new ZORL autoresearch controller/docs/candidates.
- `/home/apanda/xorl-internal-zorl-dispatch-feature`: standalone SGLang
  ZORL runbook/client/tasks/generated manifests and the newer
  `experiments/zorl/run_countdown_test.py`.
- `origin/apanda-dev` at `b909c549`: new upstream OPD hidden-match,
  inference weight-sync hardening, sparse-delta manifest/serial sync,
  and server gradient-checkpointing updates.

Important reconciliation choices:

- `experiments/zorl/run_countdown_test.py` uses the dispatch-feature
  version because it carries the newer standalone/native SGL path and
  already includes ZORL score-normalization hooks.
- Conflicts in OPD/sparse-delta/weight-sync files were resolved by
  keeping the latest `origin/apanda-dev` behavior.
- `src/xorl/server/weight_sync/handler.py` keeps both sides: it first
  syncs live adapter state for a `model_id`, then follows the latest
  sparse-delta path/manifest validation and dispatch.
- ZORL server/API/runtime additions were preserved in
  `src/xorl/server/zorl.py`, `src/xorl/server/api_server/api_types.py`,
  `src/xorl/server/runner/model_runner.py`,
  `src/xorl/server/runner/runner_dispatcher.py`, and related tests.

## 2. File Map

Start here:

- `experiments/zorl/README.md`: short layout and validated recipes.
- `experiments/zorl/ZORL_HANDOFF_RUNBOOK_2026_06_03.md`: this file.
- `experiments/zorl/standalone/RUNBOOK.md`: detailed standalone SGLang
  runbook from the dispatch worktree.
- `experiments/zorl/ZORL_AUTORESEARCH_RUNBOOK_2026_06_03.md`: narrower
  autoresearch queue/controller runbook.

Core ZORL experiment files:

- `experiments/zorl/run_countdown_test.py`: Countdown harness with
  `gradient`, `zorl`, and `grpo` modes.
- `experiments/zorl/run_coderforge_zorl_smoke.py`: smoke test for ZORL
  plumbing.
- `experiments/zorl/eval_countdown.py`: greedy temp=0 adapter evaluator.
- `experiments/zorl/configs/*.yaml`: local/server recipe configs.
- `experiments/zorl/k8s/*.yaml`: static k8s jobs/services.
- `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml`:
  CPU-side standalone ZORL client job for SGLang-native tasks such as
  OPD multiplication.
- `experiments/zorl/k8s/generated/*.yaml`: historical/generated
  standalone manifests from dispatch runs.

Standalone SGLang ZORL:

- `experiments/zorl/standalone/init_zorl_adapter.py`: builds the
  zero-B, kaiming-A starting LoRA adapter.
- `experiments/zorl/standalone/zorl_client.py`: thin HTTP orchestrator
  for `/start_zorl_session`, `/start_zorl_generation`,
  `/apply_zorl_rewards`, snapshots, and probes.
- `experiments/zorl/standalone/tasks/*.py`: task plugins for Countdown,
  GSM8K, alphabet sort, OPD multiplication, and Wordle.

Autoresearch:

- `experiments/zorl/autoresearch/controller.py`: queue/controller CLI.
- `experiments/zorl/autoresearch/ideas.yaml`: current queue.
- `experiments/zorl/autoresearch/candidates/SGL-001.yaml`
- `experiments/zorl/autoresearch/candidates/SGL-047.yaml`
- `experiments/zorl/autoresearch/candidates/SGL-117.yaml`
- `experiments/zorl/autoresearch/candidates/ZORL-*.yaml`

Server integration:

- `src/xorl/server/zorl.py`: session state, candidate LoRA construction,
  and ES update helpers.
- `src/xorl/server/api_server/api_types.py`: ZORL request/response API
  types.
- `src/xorl/server/api_server/training_ops.py`: API routes for ZORL
  generation/reward operations.
- `src/xorl/server/runner/model_runner.py`: server-side ZORL runtime
  hooks.
- `src/xorl/server/runner/runner_dispatcher.py`: rank-wide ZORL command
  dispatch.
- `tests/experiments/test_zorl_countdown_update.py`: Countdown update
  regression.

## 3. Autoresearch Loop

Available long-lived SGL nodes:

- `001`: `research-common-h100-001.cloud.together.ai`
- `047`: `research-common-h100-047.cloud.together.ai`
- `117`: `research-common-h100-117.cloud.together.ai`

Candidate service URLs:

- `http://zorl-ar-sglang-001.apanda.svc.cluster.local:30000`
- `http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000`
- `http://zorl-ar-sglang-117.apanda.svc.cluster.local:30000`

Default trainer and standalone-client candidates point at
`zorl-ar-sglang-047`. Override `INFER_URL` at render/launch time to use
another SGL node.

List the next queued idea:

```bash
python experiments/zorl/autoresearch/controller.py next
```

Render a node-pinned long-lived SGLang candidate:

```bash
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SGL-047.yaml \
  --output /tmp/zorl-ar-sglang-047.yaml

kubectl create -f /tmp/zorl-ar-sglang-047.yaml
```

Dry-run a trainer launch:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-001 \
  --dry-run
```

Launch against a specific SGL controller:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-001 \
  --set-env INFER_URL=http://zorl-ar-sglang-047.apanda.svc.cluster.local:30000
```

Monitor, score, and advance the queue:

```bash
python experiments/zorl/autoresearch/controller.py monitor \
  --idea-id ZORL-001 \
  --result latest \
  --tail 120

python experiments/zorl/autoresearch/controller.py score \
  --idea-id ZORL-001 \
  --result latest \
  --json

python experiments/zorl/autoresearch/controller.py advance \
  --id ZORL-001 \
  --result latest
```

Dry-run the standalone ZORL-OPD smoke after an SGL controller is healthy:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-OPD-000 \
  --dry-run
```

## 4. Current Science Queue

Current queued candidates:

- `ZORL-000`: smoke gate for multi-session ZORL plumbing.
- `ZORL-001`: Countdown modal rollout baseline.
- `ZORL-002`: more perturbation pairs.
- `ZORL-003`: SGD momentum.
- `ZORL-004`: longer scale retest.
- `ZORL-OPD-000`: standalone OPD multiplication smoke using synthetic
  2-digit operands.
- `ZORL-OPD-001`: standalone OPD multiplication 3-digit science run gated
  by `ZORL-OPD-000`.

Known Countdown baseline from the consolidated notes:

```text
STEPS=64
TRAINER_MODE=zorl
LORA_RANK=4
LORA_ALPHA=4
LEARNING_RATE=1e-2
ZORL_NUM_PAIRS=8
ZORL_ROLLOUTS_PER_PUZZLE=4
ZORL_ROLLOUT_TEMPERATURE=0.6
ZORL_B_SIGMA=0.1
ZORL_SCORE_BATCH_SIZE=128
ZORL_SCORE_MAX_WORKERS=4
ZORL_SGD_MOMENTUM=0.0
ZORL_PARENT_PROBE_INTERVAL=8
SYNC_QUANT=none
```

Known standalone SGL recipe from `standalone/RUNBOOK.md`:

```text
b_sigma=0.012
lr=0.01
max_update_norm=20000
rollout_temperature=0.6
probe_temperature=0.6
```

## 5. Standalone SGLang Path

Use this when running ZORL with no trainer process. The server owns the
parent LoRA, candidate perturbations, ES update, and snapshots.

One-time adapter init:

```bash
python experiments/zorl/standalone/init_zorl_adapter.py \
  --model /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<sha> \
  --output-dir /shared/zorl/init-adapters/qwen3-30b-a3b-r16 \
  --rank 16 \
  --alpha 16
```

Run the client:

```bash
TS=$(date +%s)
PYTHONUNBUFFERED=1 python -u -m experiments.zorl.standalone.zorl_client \
  --task countdown \
  --infer-url http://<sglang-service>:30000 \
  --adapter-dir /shared/zorl/init-adapters/qwen3-30b-a3b-r16 \
  --model /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<sha> \
  --parent-lora-name "zorl-${TS}/parent" \
  --session-id "zorl-${TS}" \
  --steps 25 \
  --num-pairs 16 \
  --b-sigma 0.012 \
  --lr 0.01 \
  --max-update-norm 20000 \
  --train-size 16 \
  --eval-size 32 \
  --rollout-temperature 0.6 \
  --probe-temperature 0.6 \
  --probe-n 16 \
  --score-max-workers 32 \
  --probe-interval 1
```

Always use `PYTHONUNBUFFERED=1` and `python -u` for long runs; otherwise
`nohup` logs can stay empty for hours.

## 6. ZORL-OPD / ZORL-OPSD Next Steps

This consolidation provides the long-lived SGL controller machinery and
existing ZORL Countdown/password/standalone tasks. The first ZORL-OPD path
is now wired through the standalone SGLang client:

- task plugin: `experiments/zorl/standalone/tasks/opd_multiplication.py`
- client job: `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml`
- smoke candidate: `experiments/zorl/autoresearch/candidates/ZORL-OPD-000-smoke.yaml`
- 3-digit candidate:
  `experiments/zorl/autoresearch/candidates/ZORL-OPD-001-opd-multiplication-3digit.yaml`

Render and dry-run the smoke:

```bash
python experiments/zorl/autoresearch/controller.py launch \
  --id ZORL-OPD-000 \
  --dry-run
```

Recommended next-agent path:

1. Launch or verify at least one SGL-001/047/117 controller first.
2. Run and advance `ZORL-OPD-000`; it must pass before `ZORL-OPD-001`
   becomes runnable.
3. Run `ZORL-OPD-001` against the same SGL controller, overriding
   `INFER_URL` if needed.
4. Promote only if `score` reports at least `weak_signal` or
   `promote_retest`.
5. For ZORL-OPSD, follow the same surface choice: use standalone SGL if
   the reward can be computed from HTTP generations, otherwise add a
   trainer-coupled harness.

## 7. Verification Commands

Pin imports to this worktree first. The current shell environment may
otherwise import `xorl` from `/home/apanda/xorl-internal`.

```bash
export PYTHONPATH=/home/apanda/xorl-apanda-dev-zorl-consolidated/src
```

Syntax:

```bash
python -m py_compile \
  experiments/zorl/run_countdown_test.py \
  experiments/zorl/run_coderforge_zorl_smoke.py \
  experiments/zorl/eval_countdown.py \
  experiments/zorl/autoresearch/controller.py \
  experiments/zorl/standalone/init_zorl_adapter.py \
  experiments/zorl/standalone/zorl_client.py \
  experiments/zorl/standalone/tasks/*.py \
  src/xorl/server/zorl.py \
  src/xorl/server/api_server/api_types.py \
  src/xorl/server/runner/model_runner.py \
  src/xorl/server/runner/runner_dispatcher.py \
  tests/experiments/test_zorl_countdown_update.py
```

Conflict and whitespace checks:

```bash
rg -n '^<<<<<<<|^=======$|^>>>>>>>' experiments/zorl src tests
git diff --cached --check
```

Focused tests to run before committing if the environment has test
dependencies:

```bash
PYTHONPATH=/home/apanda/xorl-apanda-dev-zorl-consolidated/src \
pytest -q \
  tests/experiments/test_zorl_countdown_update.py \
  tests/server/api_server/test_api_types.py \
  tests/server/test_server_arguments.py
```

## 8. Operational Guardrails

- Do not launch trainer or standalone-client candidates until at least one SGL-001/047/117
  controller is serving `/health`.
- For long-lived controllers, prefer one SGL pod per available node and
  route clients by overriding `INFER_URL`.
- In the `apanda` namespace, the home PVC is `home-apanda` and is mounted
  at `/workspace/home`; the older placeholder `workspace-home` PVC is not
  present.
- Avoid deleting old `experiments/zorl/results/` artifacts during scoring;
  the controller resolves `latest` by result mtime.
- If a run wedges, keep the rendered manifest and logs. They are the
  easiest way to reconstruct env overrides.
- Do not clean `/home/apanda/xorl-internal` as part of this handoff
  without explicit approval. It was observed dirty during consolidation,
  and some of those changes may predate this work.
