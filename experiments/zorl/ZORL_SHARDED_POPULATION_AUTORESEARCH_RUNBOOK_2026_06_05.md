# ZORL Sharded-Population Autoresearch Canonical Runbook

Last updated: 2026-06-05 04:20 UTC

Workspace:

```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated
```

This is the canonical operating document for the current ZORL autoresearch loop.
It supersedes the older 2026-06-03 and 2026-06-04 runbooks. Sharded population
ZORL is now implemented and live: many TP=2 SGLang replicas jointly materialize
and score a much larger ES population while applying the same global parent
update on every worker.

The primary science direction is no longer Countdown. Countdown is retained as a
regression/reference task because GRPO and early ZORL did not show reliable
progress there. The active target is to make ZORL work well on tasks where dense
or semi-dense supervision is available, especially Wordle OPSD teacher-forced
training and possibly GSM8K shaped/hinted variants.

## Current System Facts

Live serving topology as observed on 2026-06-05 04:15 UTC:

```text
StatefulSet: zorl-ar-sglang
replicas: 16
per replica: Qwen/Qwen3-30B-A3B-Instruct-2507, TP=2, LoRA rank 4
service: zorl-ar-sglang-headless
direct endpoints:
  http://zorl-ar-sglang-{0..15}.zorl-ar-sglang-headless.apanda.svc.cluster.local:30000

SMG service: http://zorl-ar-smg.apanda.svc.cluster.local:30000
SMG policy: round_robin
active sharded candidate scoring: direct owner routing, not SMG
```

Current SGLang pool launch limits:

```text
--max-loaded-loras 80
--max-loras-per-batch 64
--max-running-requests 64
--max-queued-requests 512
```

Replicated topology is still supported, but it is no longer the scale path. In
replicated mode every worker has the same candidate LoRA population, so SMG can
round-robin `/generate` requests. In sharded mode every worker has the parent
and only its local candidate pairs; candidate scoring must route to the owner
URL until SMG has LoRA-aware routing.

Current scale run:

```text
candidate: ZORL-WORDLE-013
job: zorl-ar-zorl-wordle-013-tph4n
task: wordle
score_mode: teacher_forced
trace style: hinted_cot
steps: 256 planned
num_pairs: 512
population: 1024
shards: 16
pairs_per_shard: 32
local candidates per worker: 64
train_size per step: 128
train_pool_size: 512
eval_size: 128
resample_train_each_step: true
W&B project: together-research/zorl
W&B run: ZORL-WORDLE-013-tph4n
W&B URL: https://wandb.ai/together-research/zorl/runs/zorl-wordle-013-tph4n
```

First-probe status after step 16:

```text
cold probe reward: 0.0096
step-16 probe reward: 0.0140
reward gain: +0.0044 absolute, about +46% relative
exact rate: 0.0
train reward mean: 0.0101 at step 1 -> 0.0141 at step 16
best candidate max through step 16: 0.0205
update rows: 16/16 positive
pair usage: used_pairs=512, dropped_pairs=0 on every completed step
average score time: 539.4s
average apply time: 112.5s
average total step time: 651.8s, about 10.9 minutes
controller verdict: incomplete only because min_steps=32
```

This result says the large sharded update path is healthy and the dense
teacher-forced reward is still climbable at population 1024 with a larger,
resampled train pool. It still does not prove free-form Wordle play: exact rate
is zero and rollout probes must be tracked separately before claiming solved
Wordle.

## Key Constraint: SMG Round-Robin Is Not Enough For Sharded LoRAs

For population size 1024:

```text
population = 1024 candidates
num_pairs = 512 antithetic pairs
current shard: 16 replicas * 64 candidate LoRAs per replica = 1024 candidates
```

This is conceptually viable, but only if candidate ownership is explicit.

Current replicated design:

```text
worker 0 has candidates 0..1023
worker 1 has candidates 0..1023
...
SMG can route any request anywhere
```

Sharded design:

```text
worker 0 has 32 antithetic pairs = 64 candidates
worker 1 has 32 antithetic pairs = 64 candidates
...
worker 15 has 32 antithetic pairs = 64 candidates
parent LoRA exists on every worker
```

In the sharded design, a generation request for candidate 173 must go to the
worker that materialized candidate 173. Blind SMG round-robin will fail because
most workers will not have that `lora_path`.

There are two acceptable routing designs:

1. **Direct owner routing, current implementation.**
   The standalone client records the owner URL for each candidate returned by
   `/start_zorl_generation` and sends scoring requests directly to that owner.
   Parent probes can use direct URLs or SMG because parent LoRA is replicated to
   all workers. W011, W012, and W013 use owner routing.

2. **LoRA-aware SMG routing, later implementation.**
   SMG owns or receives a `lora_path -> worker_url` map and routes generation
   requests to the worker that has the requested LoRA. This keeps one stable
   reward endpoint, but requires router changes.

Direct owner routing remains the recommended production path until SMG has
explicit LoRA ownership routing.

## Correctness Invariants For Sharded Population

The important invariant is that all workers end each ZORL step with identical
parent LoRA weights.

That requires:

- Every worker starts from the same parent LoRA.
- Every worker uses the same global session seed, generation index, family
  index, `b_sigma`, perturbation mode, and global `num_pairs`.
- Antithetic pair `+` and `-` stay on the same worker for scoring locality and
  simpler failure accounting.
- The coordinator collects rewards for the full global population.
- The same full global reward payload is sent to `/apply_zorl_rewards` on every
  worker.
- Every worker reconstructs the full global update from the same pair seeds and
  scores, even if it only materialized a candidate shard for scoring.
- Generation cleanup deletes only locally materialized candidate adapters.
- If one worker fails to apply the global update, the run must stop; continuing
  with divergent parents invalidates the experiment.

Do not implement sharding by letting each worker apply only its local shard
update. That produces different parents on different replicas.

## Implemented Sharded Population Design

### Phase 0: Stabilize Current Replicated Loop

Replicated topology remains available while sharding is selected by explicit
flags. Current code paths:

```text
experiments/zorl/standalone/zorl_client.py
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-smg-router.yaml
src/xorl/server/zorl.py
src/xorl/server/runner/model_runner.py
```

Current client features that should remain:

- Multiple direct `--infer-url` values for replicated control.
- Optional `--reward-infer-url` for SMG routed scoring.
- Wordle `--score-mode teacher_forced`.
- Wordle `--wordle-teacher-trace-style hinted_cot`.
- Client-side update strategies:
  - `raw`
  - `centered`
  - `rank`
  - `project_baseline`
  - `project_baseline_positive`
  - `project_baseline_standardized`

Keep these tests green:

```bash
ruff check experiments/zorl/standalone/zorl_client.py \
  experiments/zorl/standalone/wandb_log_forwarder.py \
  experiments/zorl/autoresearch/controller.py \
  tests/experiments/test_zorl_autoresearch_controller.py

PYTHONPATH=$PWD/src:$PWD pytest tests/experiments/test_zorl_autoresearch_controller.py -q
```

### Phase 1: Global-Plan, Local-Materialization Semantics

The native ZORL generation endpoint now supports each worker creating a full
global generation plan while materializing only its local candidate shard.

`/start_zorl_generation` request:

```json
{
  "session_id": "zorl-wordle-...",
  "num_pairs": 512,
  "preload_candidates": false,
  "materialization": {
    "mode": "pair_shard",
    "shard_index": 0,
    "num_shards": 16
  }
}
```

The server:

- Builds the full global `ZORLGenerationPlan` with the requested global
  `num_pairs`.
- Store the full plan as active generation.
- Materialize only candidate pairs whose `perturbation_index % num_shards ==
  shard_index`, or use contiguous pair ranges if `mode=pair_range`.
- Return only locally materialized candidates, each annotated with ownership.

Response fields:

```json
{
  "generation_id": "...-g000000",
  "global_num_pairs": 512,
  "global_population": 1024,
  "shard_index": 0,
  "num_shards": 16,
  "local_num_pairs": 32,
  "candidates": [
    {
      "candidate_id": "...-p0000+",
      "perturbation_index": 0,
      "direction": "positive",
      "lora_name": "zorl/...",
      "owner_url": "http://zorl-ar-sglang-0..."
    }
  ]
}
```

Server code locations:

```text
src/xorl/server/zorl.py
  ZORLSessionState.begin_generation
  ZORLGenerationPlan

src/xorl/server/runner/model_runner.py
  start_zorl_generation
  apply_zorl_rewards
  _cleanup_zorl_generation_exports
```

Internal representation:

- `ZORLGenerationPlan.candidates` remains the full global candidate list.
- Add `materialized_candidate_ids` or return-time filtering in
  `ModelRunner.start_zorl_generation`.
- Keep `active_generation` full-size so `apply_zorl_rewards` can reconstruct the
  global update on every worker.

Compatibility:

- If no `materialization` field is present, preserve current full
  materialization behavior.
- If one direct URL is provided, current single-worker runs remain unchanged.
- If multiple direct URLs are provided with no sharding flag, current replicated
  mode remains unchanged.

### Phase 2: Client-Side Shard Coordinator

`experiments/zorl/standalone/zorl_client.py` has a sharded population mode.

CLI flags:

```text
--population-sharding replicated|pair_shard
--num-shards auto|N
--pairs-per-shard optional int
--candidate-routing owner|smg
```

Defaults:

```text
population_sharding=replicated
candidate_routing=owner when sharding is enabled
num_shards=len(control_infer_urls)
```

Client behavior in sharded mode:

1. Load parent adapter on all control URLs.
2. Start session on all control URLs with identical global config.
3. For each generation, call `/start_zorl_generation` on every worker with:

   ```text
   num_pairs = global num_pairs
   shard_index = worker index
   num_shards = number of workers
   ```

4. Combine all returned local candidate lists into one global candidate list.
5. Validate:
   - Same `generation_id` on all workers.
   - No duplicate candidate ids.
   - Every perturbation pair has both `+` and `-`.
   - Expected global candidate count is `2 * num_pairs`.
6. Score each candidate on its owner URL.
7. Transform rewards if `--update-strategy` is not `raw`.
8. Send the full transformed reward list to `/apply_zorl_rewards` on every
   control URL.
9. Validate returned update metrics agree across workers within tolerance:
   - `used_pairs`
   - `dropped_pairs`
   - `update_norm`
   - `pair_delta_mean`
   - `pair_delta_std`
10. Flush caches on all workers.
11. Probe parent on SMG or all direct URLs.

Do not route sharded candidate scoring through the current round-robin SMG.
Use owner URLs until SMG has explicit LoRA ownership routing.

### Phase 3: Add LoRA-Aware SMG Routing

This is optional after direct owner routing works.

Required SMG behavior:

- Accept a route table mapping `lora_path` or `lora_name` prefixes to worker
  URLs.
- For `/generate`, inspect `lora_path`.
- If `lora_path` maps to an owner, route there.
- If `lora_path` is the replicated parent, use normal load balancing.
- If no route is found, either reject with a clear error or fall back only when
  the route table marks the LoRA as replicated.

Possible route-table source:

- Static config file written by the standalone client per generation.
- HTTP registration endpoint called by the standalone client.
- Embedded candidate ownership metadata from `/start_zorl_generation`.

Do not rely on hash-based routing unless the same hash function also controls
candidate materialization. Explicit maps are safer.

### Phase 4: Scale Serving Pool For Large Populations

A sharded run with 64 local candidates per worker needs at least 65 loaded LoRA
slots per worker if the parent counts toward `max-loaded-loras`. Keep extra
headroom for snapshots, transient loads, and cleanup lag.

Recommended SGLang caps for 64 local candidates:

```text
SGLANG_MAX_LOADED_LORAS=80
SGLANG_MAX_LORAS_PER_BATCH=64
SGLANG_MAX_RUNNING_REQUESTS=64
SGLANG_MAX_QUEUED_REQUESTS=512
```

If SGLang memory allows, prefer:

```text
SGLANG_MAX_LOADED_LORAS=96
```

This gives room for parent, snapshots, transient loads, and cleanup lag.

For population 256:

```text
num_pairs=128
num_shards=4
local_pairs_per_worker=32
local_candidates_per_worker=64
```

For population 320:

```text
num_pairs=160
num_shards=5
local_pairs_per_worker=32
local_candidates_per_worker=64
```

For population 512:

```text
num_pairs=256
num_shards=8
local_pairs_per_worker=32
local_candidates_per_worker=64
```

For population 1024:

```text
num_pairs=512
num_shards=16
local_pairs_per_worker=32
local_candidates_per_worker=64
```

The live pool currently has 16 replicas and supports the W013 population-1024
run with `PAIRS_PER_SHARD=32`.

### Phase 5: Correctness Tests

Add tests before launching a large run.

Unit tests:

- `ZORLSessionState.begin_generation` produces identical global candidate ids
  for sharded and unsharded plans.
- Shard assignment covers every perturbation index exactly once.
- `+` and `-` for a perturbation pair are assigned to the same shard.
- `apply_zorl_rewards` with full global rewards on a locally materialized shard
  reconstructs the same update as an unsharded full-materialization plan.
- Missing one side of a pair increments `dropped_pairs`.
- Duplicate candidate ownership is rejected by the client.

Client tests:

- Combine shard responses into global candidate list.
- Route candidate scoring to `owner_url`.
- Reject inconsistent `generation_id` across workers.
- Reject incomplete population count.
- Verify global apply metrics across workers.
- Parse and forward standalone logs to W&B without dropping cold/probe/step
  metrics.

Integration smoke:

```text
2 workers
num_pairs=4
population=8
local population=4 per worker
train_size=2
steps=2
score_mode=teacher_forced
```

Gate:

- Both workers apply the same global update.
- Parent probes remain callable from both workers.
- Final parent update metrics match an equivalent replicated 1-worker run for
  the same seed and rewards, within numeric tolerance.

## ES Algorithm Plan

The current raw update sends candidate rewards to the server and lets the server
compute antithetic pair deltas followed by standard z-score normalization. It
works for W012/W013 and has nonzero pair deltas at population 1024, but it may
still overfit teacher traces or become noisy at larger train sizes.

Keep these strategy branches:

```text
raw
  Historical behavior. Server computes pair deltas and standardizes.

centered
  Client subtracts population mean reward before server pair differencing.

rank
  Client converts candidate rewards to centered rank z-scores.
  Useful when reward scale changes over training.

project_baseline
  Client scores parent on the train set, then uses candidate - parent per
  project. Tests whether the update should focus on improvements over the
  current parent rather than absolute teacher likelihood.

project_baseline_positive
  Uses max(0, candidate - parent) per project. More conservative, but can drop
  negative gradient signal.

project_baseline_standardized
  Parent and candidates define per-project score statistics. Candidate project
  scores are standardized before aggregation. Best first serious variant for
  larger Wordle runs.
```

Recommended current science sequence:

1. Let W013 continue at least through the 32-step gate.
2. Score W013 and update `ideas.yaml`/scorecards with the explicit result path.
3. Compare W013 probe reward at steps 16, 32, 48, and 64 against W012's
   population-256 trajectory.
4. If W013 remains healthy, test `rank` or `project_baseline_standardized` at
   population 1024 using the same 16-worker pool.
5. Add rollout probes before claiming actual Wordle improvement.
6. If step time is too high, test lower `TRAIN_SIZE` or higher
   `SCORE_MAX_WORKERS` only after preserving pair correctness (`used_pairs`,
   `dropped_pairs`, metric agreement).

Do not promote a run on teacher-forced reward alone. A wrong global update or a
reward-only teacher trace improvement makes downstream claims uninterpretable.

## Wordle Reward And Teacher

Wordle has two reward modes in the current standalone loop.

### Rollout Mode

The model plays a six-turn Wordle game. It emits guesses in:

```text
<guess>WORD</guess>
```

The harness supplies feedback:

```text
G = correct letter and position
Y = letter present but wrong position
B = letter absent
```

Rollout reward is:

```text
0.4 * solved + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus
```

This is the score that matters for actual free-form play.

### OPSD Teacher-Forced Mode

In `teacher_forced` mode, the teacher trace is constructed by the task plugin,
not generated by the model during training.

Current Wordle teacher creation:

```text
task file: experiments/zorl/standalone/tasks/wordle.py
function: build_teacher_forced_example
style: hinted_cot or guess_only
```

For `hinted_cot`, the context includes the private target word:

```text
Private hint: the target word is CRATE.
```

The teacher text includes a compact hidden rationale and two guesses:

```text
<think>... target word is CRATE ...</think>
<guess>STARE</guess>
<guess>CRATE</guess>
```

The reward is the mean input-token likelihood of the teacher target suffix:

```text
reward = exp(mean teacher target token logprob)
```

This answers the prior question directly: yes, in the current OPSD path we give
the teacher target in the context and score the model on a plausible teacher
trace. This is dense distillation pressure, not a free-form solve metric. Always
track rollout exact separately before claiming Wordle is solved.

## Canonical Operating Runbook

### Repo And Environment

Use:

```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated
```

For Python commands, the Kubernetes job defaults to:

```text
PYTHON_BIN=/workspace/home/xorl-internal/.venv/bin/python
PYTHONPATH=/workspace/home/xorl-apanda-dev-zorl-consolidated:/workspace/home/xorl-apanda-dev-zorl-consolidated/src
```

Local checks:

```bash
PYTHONPATH=$PWD/src:$PWD ruff check \
  experiments/zorl/standalone/zorl_client.py \
  experiments/zorl/standalone/wandb_log_forwarder.py \
  experiments/zorl/autoresearch/controller.py \
  tests/experiments/test_zorl_autoresearch_controller.py

PYTHONPATH=$PWD/src:$PWD pytest tests/experiments/test_zorl_autoresearch_controller.py -q
```

### GPU Scheduling

Every pod requesting GPUs must have:

```yaml
spec:
  template:
    metadata:
      labels:
        team: turbo
```

For the ZORL SGLang StatefulSet, the label is already in:

```text
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml
```

Do not manually set Volcano queue labels or scheduler name unless explicitly
needed. Kyverno injects them from `team: turbo`.

### Serving Pool Files

Serving candidate:

```text
experiments/zorl/autoresearch/candidates/SGL-POOL.yaml
```

Serving manifest:

```text
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml
```

SMG candidate:

```text
experiments/zorl/autoresearch/candidates/SMG-ZORL.yaml
```

SMG manifest:

```text
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-smg-router.yaml
```

Standalone client manifest:

```text
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml
```

### Check Serving Pool

```bash
kubectl get pods,statefulset,svc,jobs -n apanda -o wide \
  | rg 'zorl-ar-(sglang|smg|zorl-wordle)'
```

Expected stable serving state:

```text
statefulset.apps/zorl-ar-sglang 16/16
pod/zorl-ar-sglang-0  Running
...
pod/zorl-ar-sglang-15 Running
pod/zorl-ar-smg-* Running
service/zorl-ar-sglang-headless
service/zorl-ar-smg
```

Check SGLang launch flags:

```bash
kubectl get statefulset zorl-ar-sglang -n apanda -o yaml \
  | rg -n 'max-loras|max-loaded|max-running|enable-lora|lora-moe|replicas|team' -C 2
```

Check SMG routing:

```bash
kubectl logs -n apanda -l app=zorl-ar-smg --since=5m --tail=160 \
  | rg 'Worker selected|Backend request completed|ERROR|WARN|status=200'
```

### Render And Launch A Candidate

Render:

```bash
python experiments/zorl/autoresearch/controller.py render \
  --id ZORL-WORDLE-013 \
  --output /tmp/zorl-wordle-013.yaml
```

Dry run:

```bash
kubectl create --dry-run=server -n apanda -f /tmp/zorl-wordle-013.yaml
```

Launch:

```bash
python experiments/zorl/autoresearch/controller.py launch --id ZORL-WORDLE-013
```

The controller records `last_manifest` and uses `kubectl create` internally.
Use `kubectl create`, not `kubectl apply`, for generated-name Jobs when
launching a rendered manifest manually.

### Candidate YAML Fields

Important env keys for standalone ZORL:

```yaml
env:
  TASK: wordle
  SCORE_MODE: teacher_forced
  WORDLE_TEACHER_TRACE_STYLE: hinted_cot
  INFER_URL: "<direct worker URLs>"
  REWARD_INFER_URL: "http://zorl-ar-smg.apanda.svc.cluster.local:30000"
  STEPS: "64"
  NUM_PAIRS: "16"
  TRAIN_SIZE: "16"
  TRAIN_POOL_SIZE: "16"
  RESAMPLE_TRAIN_EACH_STEP: "0"
  EVAL_SIZE: "32"
  B_SIGMA: "0.05"
  LEARNING_RATE: "0.01"
  MAX_UPDATE_NORM: "20000"
  UPDATE_STRATEGY: raw
  PERTURBATION_MODE: b_only
  ELITIST_ROLLBACK: "1"
  SCORE_MAX_WORKERS: "64"
  PROBE_INTERVAL: "4"
  WANDB_PROJECT: zorl
  WANDB_NAME: ZORL-WORDLE-013
  WANDB_GROUP: ZORL-WORDLE-013
  WANDB_JOB_TYPE: zorl-standalone-wordle
```

Interpretation:

```text
population = 2 * NUM_PAIRS
```

For replicated current pool:

```text
NUM_PAIRS must fit per worker: parent + 2 * NUM_PAIRS <= max_loaded_loras
```

With the current `max_loaded_loras=80` default, `NUM_PAIRS=39` is the safer
replicated upper bound if the parent counts against the cap. For sharded mode,
the local cap is `parent + 2 * local_num_pairs`.

For sharded client runs, set:

```yaml
env:
  POPULATION_SHARDING: pair_shard
  NUM_SHARDS: auto
  PAIRS_PER_SHARD: "32"
  CANDIDATE_ROUTING: owner
```

For W013, the concrete data/population settings are:

```yaml
env:
  STEPS: "256"
  NUM_PAIRS: "512"
  PAIRS_PER_SHARD: "32"
  TRAIN_SIZE: "128"
  TRAIN_POOL_SIZE: "512"
  RESAMPLE_TRAIN_EACH_STEP: "1"
  EVAL_SIZE: "128"
  SCORE_MAX_WORKERS: "512"
  PROBE_INTERVAL: "16"
```

`TRAIN_POOL_SIZE` controls the larger fixed train pool. `TRAIN_SIZE` controls
the per-step batch size. With `RESAMPLE_TRAIN_EACH_STEP=1`, the standalone
client cycles deterministically through shuffled train batches so adjacent
steps do not reuse the same tiny train set.

### W&B Logging

Current project:

```text
project: together-research/zorl
run: ZORL-WORDLE-013-tph4n
url: https://wandb.ai/together-research/zorl/runs/zorl-wordle-013-tph4n
```

The standalone client writes text logs. W&B logging is handled by:

```text
experiments/zorl/standalone/wandb_log_forwarder.py
```

The forwarder parses `zorl_client.log`, backfills completed cold/step metrics,
then follows the log and records future steps. The base standalone Kubernetes
manifest starts it automatically when `WANDB_PROJECT` is nonempty.

Attach W&B to an already-running run:

```bash
kubectl exec -n apanda zorl-ar-zorl-wordle-013-tph4n-mjg9p -- /bin/bash -lc '
set -euo pipefail
REPO_ROOT=/workspace/home/xorl-apanda-dev-zorl-consolidated
RUN_DIR=${REPO_ROOT}/experiments/zorl/results/zorl_autoresearch/ZORL-WORDLE-013/20260605T010823Z-zorl-ar-zorl-wordle-013-tph4n-mjg9p-wordle
PYTHON_BIN=/workspace/home/xorl-internal/.venv/bin/python
cd "${REPO_ROOT}"
nohup "${PYTHON_BIN}" -u experiments/zorl/standalone/wandb_log_forwarder.py \
  --log-file "${RUN_DIR}/zorl_client.log" \
  --project zorl \
  --name ZORL-WORDLE-013-tph4n \
  --run-id zorl-wordle-013-tph4n \
  --group ZORL-WORDLE-013 \
  --job-type zorl-standalone-wordle \
  --follow \
  --poll-seconds 30 \
  > "${RUN_DIR}/wandb_forwarder.log" 2>&1 &
echo $! > "${RUN_DIR}/wandb_forwarder.pid"
'
```

Verify W&B forwarding:

```bash
kubectl exec -n apanda zorl-ar-zorl-wordle-013-tph4n-mjg9p -- /bin/bash -lc '
RUN_DIR=/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/ZORL-WORDLE-013/20260605T010823Z-zorl-ar-zorl-wordle-013-tph4n-mjg9p-wordle
ps -p "$(cat "${RUN_DIR}/wandb_forwarder.pid")" -o pid,stat,etime,cmd
cat "${RUN_DIR}/zorl_client.log.wandb-state.json"
tail -40 "${RUN_DIR}/wandb_forwarder.log"
'
```

### Monitor A Run

Find the latest result directory:

```bash
latest=$(ls -td experiments/zorl/results/zorl_autoresearch/ZORL-WORDLE-013/*-wordle | head -1)
echo "$latest"
tail -n 120 "$latest/job.log"
```

Check pod/job:

```bash
kubectl get pods,jobs -n apanda -o wide | rg 'zorl-ar-zorl-wordle-013|zorl-ar-sglang|zorl-ar-smg'
```

Score a result:

```bash
python experiments/zorl/autoresearch/controller.py score \
  --idea-id ZORL-WORDLE-013 \
  --result "$latest/job.log" \
  --json
```

When scoring a specific run after a newer run has started, pass the explicit log
path. Do not use `latest` across different ideas unless the result root is
unambiguous.

Look for:

```text
cold: reward_mean=...
step N/M: reward_mean=... best_cand=... update_norm=...
pair_delta_std=...
zero_score_pairs=0
dropped_pairs=0
probe_reward=...
exact_rate=...
[done] M steps in ...
```

For W013 specifically, the first 16-step probe should show:

```text
cold: reward_mean=0.0096 exact_rate=0.0000 exact_count=0.0/128
step 16/256: reward_mean=0.0141 ... used_pairs=512 ... dropped_pairs=0 ...
  t_score=534.9s t_apply=110.3s, probe_reward=0.0140 exact_rate=0.0000
```

For teacher-forced Wordle, a strong reward gain is not equivalent to solving.
The run is only actual Wordle progress if rollout probes improve exact solves or
free-form rollout reward on held-out eval improves.

### Scoring Gates

For standalone logs, scorecards are produced by:

```text
experiments/zorl/autoresearch/controller.py
```

Typical gate meanings:

```text
weak_reward_gain: minimum probe reward gain for weak signal
promote_reward_gain: retest-worthy gain
strong_reward_gain: strong reward climb
weak_exact_gain/promote_exact_gain/strong_exact_gain: exact-count gains
min_steps: minimum completed steps before accepting update evidence
min_update_norm_positive_rows: require actual nonzero updates
expected_total: expected eval denominator
```

For OPSD teacher-forced Wordle, reward gates are primary science signal, but
exact gates should remain visible to prevent confusing distillation with solved
play.

### Failure Modes

#### SMG round-robin with sharded LoRAs

Symptom:

```text
worker returns missing LoRA or unknown lora_path
```

Cause:

```text
SMG routed a candidate request to a worker that did not materialize that candidate.
```

Fix:

```text
Use direct owner routing or implement LoRA-aware SMG routing.
```

#### Parent divergence

Symptom:

```text
same parent probe differs substantially across direct workers after apply
```

Cause:

```text
workers applied different reward payloads or local-only updates.
```

Fix:

```text
stop the run, do not score it as valid, restart from a clean parent session.
```

#### Loaded-LoRA cap

Symptom:

```text
candidate loading fails near generation start
```

Cause:

```text
parent + local candidates + snapshots exceed --max-loaded-loras
```

Fix:

```text
reduce NUM_PAIRS or raise SGLANG_MAX_LOADED_LORAS and restart the StatefulSet.
```

#### Zero pair deltas

Symptom:

```text
pair_delta_std=0.0000
update_norm=0.00
```

Cause:

```text
reward ties across antithetic pairs, too sparse reward, or too small perturbation.
```

Fixes:

```text
use shaped/teacher-forced reward
increase train set only after nonzero deltas exist
try B_SIGMA=0.05 for Wordle-style teacher-forced branches
try update_strategy=rank or project_baseline_standardized
```

#### Greedy exact does not move

Symptom:

```text
teacher-forced reward climbs, exact_rate remains 0
```

Cause:

```text
OPSD teacher trace likelihood improved, but free-form policy did not learn a robust play policy.
```

Fixes:

```text
add rollout probes on held-out Wordle
mix teacher-forced and rollout scoring in later algorithm branch
increase train_size and steps
consider teacher traces without explicit target in final context once dense learning is established
```

### Cleanup

Completed Jobs are TTL-controlled, but old pods can be inspected until removed.

Delete a failed standalone job:

```bash
kubectl delete job -n apanda <job-name>
```

Do not delete the SGLang StatefulSet or SMG job unless intentionally restarting
the serving pool.

Restart serving pool after manifest changes:

```bash
kubectl rollout restart statefulset/zorl-ar-sglang -n apanda
kubectl rollout status statefulset/zorl-ar-sglang -n apanda
```

If changing SMG worker URLs or policy, delete and recreate the SMG Job because
it is a Job, not a Deployment:

```bash
kubectl delete job -n apanda -l app=zorl-ar-smg
python experiments/zorl/autoresearch/controller.py render \
  --candidate experiments/zorl/autoresearch/candidates/SMG-ZORL.yaml \
  --output /tmp/zorl-smg.yaml
kubectl create --dry-run=server -n apanda -f /tmp/zorl-smg.yaml
kubectl create -n apanda -f /tmp/zorl-smg.yaml
```

## Recommended Next Work

1. Keep W013 running through at least step 32, then score with the explicit log
   path and advance the idea state.
2. Keep the W&B forwarder alive; it should continue logging to
   `together-research/zorl`, run `ZORL-WORDLE-013-tph4n`.
3. Compare W013 against W012:

   ```text
   W012: population 256, train_size 16, 16 steps, reward gain +0.0045
   W013: population 1024, train_size 128 from pool 512, step-16 gain +0.0044
   ```

4. If W013 remains healthy at step 32, queue a same-scale algorithm variant:
   `UPDATE_STRATEGY=rank` or `project_baseline_standardized`.
5. Add a held-out rollout/free-form Wordle probe before claiming actual Wordle
   play improvement.
6. If the 10.9 min/step runtime is too high, test a separate speed candidate
   with lower `TRAIN_SIZE` or adjusted `SCORE_MAX_WORKERS`; do not change W013
   mid-run.
7. Preserve the 16-replica SGLang pool while W013 is active. Scaling down will
   invalidate direct endpoint assumptions for the running job.

## Completion Criteria For Sharded Population

The sharded-population implementation is operationally complete for the current
direct-owner-routing design. Evidence:

```text
num_pairs larger than one worker cap:
  W013 requests NUM_PAIRS=512 and population=1024.

candidate materialization distributed across workers:
  W013 uses 16 direct workers with PAIRS_PER_SHARD=32.

candidate scoring routes to owner:
  W011/W012/W013 use CANDIDATE_ROUTING=owner.

parent probes work after global updates:
  W012 completed probes through step 16.
  W013 completed cold and step-16 probes on eval_size=128.

all workers apply same global payload:
  W013 step lines show used_pairs=512 for global population 1024.
  Server logs show /apply_zorl_rewards with num_rewards=1024.

per-worker apply metrics agree:
  client validates used_pairs, dropped_pairs, update_norm, pair_delta_mean, and
  pair_delta_std across workers before printing the step line.

2-worker sharded smoke:
  W011 passed the sharded smoke path.

4-worker scale probe:
  W012 completed 16/16 steps and scored promote_retest.

16-worker scale run:
  W013 completed the first 16-step probe with zero dropped pairs and positive
  updates on all completed rows.

scorecard/controller path:
  W012 scorecard exists and ideas.yaml records promote_retest.
  W013 scores as incomplete until min_steps=32, which is expected.

runbook/manifests/tests:
  tests/experiments/test_zorl_autoresearch_controller.py covers sharded
  generation combination, owner routing, train resampling, rendered flags, and
  W&B log parsing.
```

Remaining non-blocking work is science and productization, not sharded
correctness: LoRA-aware SMG routing, rollout probes, and algorithm variants.

## W014 Serving Fix: SMG Owner Routing And Export

W014 is the recovery/fix run for the W013 scoring bottleneck. The W013 client
used direct owner routing, but the first thread-pool wave was candidate-major
and therefore only warmed a small number of owners. The W014 setup keeps
owner-routed candidate scoring but sends requests through SMG with an
`X-SMG-Target-Worker` header naming the candidate owner's direct SGL URL.

SGL pool caps from `qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml`:

```text
replicas:                 16
tp per replica:            2
max-running-requests:     64
max-queued-requests:     512
max-loras-per-batch:      64
max-loaded-loras:         80
base dtype:              bfloat16
LoRA dtype:              bfloat16
```

The base model is currently BF16, not FP8. The initial adapter tensors were
verified as BF16, so this pool is BF16 base plus BF16 LoRA until the SGLang
manifest is deliberately changed and validated for FP8.

W014 population sizing:

```text
pairs per shard:          32
candidate LoRAs per SGL:  64
global pairs:            512
global population:      1024
train examples/step:     128, resampled from a pool of 512
steps:                   384
score_max_workers:      4096
```

The LoRA cap matters: 32 antithetic pairs per SGL produce 64 local candidate
LoRAs, exactly matching `max-loras-per-batch=64`. `SCORE_MAX_WORKERS=4096`
maps to about 256 in-flight score requests per SGL when owner-round-robin order
is used across 16 replicas. That is 4x `max-running-requests=64`, enough to keep
the scheduler fed while staying below each replica's running+queued capacity of
576 requests. If the relaunched run still underfills SGL, the next operational
knob is `SCORE_MAX_WORKERS=8192`, or about 512 in-flight per SGL.

Persistence:

```text
SGL endpoint:      POST /export_zorl_parent
client flags:      --export-dir, --export-interval
default interval:  every 16 steps, plus best and final exports
format:            adapter_config.json + adapter_model.safetensors
```

The export path writes a PEFT/SGL reloadable adapter layout, not SGL's internal
fused tensor layout: `qkv_proj.lora_A` is reduced back to one copy, and
`gate_up_proj` is split back to `gate_proj` and `up_proj` when the adapter
config uses split gate/up targets.

Launch order:

```bash
cargo build --manifest-path /workspace/home/smg-together-thunderagent-port/Cargo.toml
kubectl delete job -n apanda -l app=zorl-ar-smg
kubectl delete service -n apanda zorl-ar-smg
python experiments/zorl/autoresearch/controller.py launch \
  --candidate experiments/zorl/autoresearch/candidates/SMG-ZORL.yaml

kubectl delete job -n apanda -l autoresearch-id=ZORL-WORDLE-013
kubectl apply -n apanda -f experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml
kubectl rollout restart -n apanda statefulset/zorl-ar-sglang
kubectl rollout status -n apanda statefulset/zorl-ar-sglang --timeout=45m

python experiments/zorl/autoresearch/controller.py launch \
  --candidate experiments/zorl/autoresearch/candidates/ZORL-WORDLE-014-opsd-tf-sharded-smg-export-pop1024.yaml
```

W013 cannot be exported in place because its live SGL processes do not have the
disk export endpoint loaded. Treat W014 as a restart unless a separate manual
live-process extraction is added and validated.
