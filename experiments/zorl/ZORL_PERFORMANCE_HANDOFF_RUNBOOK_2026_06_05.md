# ZORL Performance Handoff Runbook

Last updated: 2026-06-05

Workspace:

```bash
cd /home/apanda/xorl-apanda-dev-zorl-consolidated
```

This runbook is for the next agent optimizing ZORL score and apply
throughput. It is intentionally not an autoresearch/controller guide. The next
agent should first build a standalone microbenchmark and use that benchmark as
the eval/hillclimb target before changing the production loop.

## Current Live Run

Leave this run running unless explicitly asked to stop it:

```text
namespace: apanda
job: zorl-ar-zorl-wordle-014-zwmq5
pod: zorl-ar-zorl-wordle-014-zwmq5-k9dxr
task: Wordle teacher-forced OPSD
W&B project: zorl
run id/name root: zorl-wordle-20260605T090513Z-zorl-ar-zorl-wordle-014-zwmq5-k9dxr
```

Current launch shape:

```text
model: Qwen/Qwen3-30B-A3B-Instruct-2507
base dtype: bfloat16
LoRA dtype/rank: bf16 rank 4
SGL replicas: 16
SGL TP per replica: 2
total sampler GPUs: 32
population: 1024 candidates = 512 antithetic pairs
population sharding: pair_shard
pairs per shard: 32
local candidate LoRAs per SGL: 64
train examples per step: 128
train pool: 512, resampled every step
eval examples: 128
steps planned: 384
score mode: teacher_forced
trace style: hinted_cot
score workers: 64 global
score workers per owner: 4
teacher_forced_batch_size: 8
preload_candidates: false
export_interval: 16
probe_interval: 16
```

Useful status commands:

```bash
kubectl get job,pod -n apanda | rg 'zorl-ar-zorl-wordle-014|zorl-ar-sglang|zorl-ar-smg'

kubectl logs -n apanda zorl-ar-zorl-wordle-014-zwmq5-k9dxr --tail=120

kubectl get pods -n apanda -l app=zorl-ar-sglang \
  -o jsonpath='{range .items[*]}{.metadata.name} {.status.containerStatuses[0].restartCount} {.status.containerStatuses[0].ready}{"\n"}{end}'

kubectl exec -n apanda zorl-ar-smg-nqfg9-zb2kw -- \
  curl -fsS --max-time 60 http://127.0.0.1:8080/health_generate
```

Observed stable performance for the current shape:

```text
step 1: t_score=243.0s, t_apply=107.9s, used_pairs=512, dropped_pairs=0
step 2: t_score=237.5s, t_apply=125.6s, used_pairs=512, dropped_pairs=0
later steady state: t_score about 233-236s, t_apply about 104-130s
```

Effective score throughput math:

```text
sequences per step = 1024 candidates * 128 examples = 131072 sequences
observed teacher-forced input length = about 238 tokens/sequence
tokens per step = about 31.2M input tokens
aggregate score throughput = 31.2M / 235s = about 133k tokens/s
per SGL replica = about 8.3k tokens/s
per GPU = about 4.1k tokens/s
```

This is the number to beat. Optimize against the same population, train size,
LoRA rank, and prompt length unless intentionally sweeping one of those axes.

## Architecture

ZORL here means standalone zeroth-order RL on top of SGLang-native LoRA
updates. There is no xorl trainer process in the active Wordle run.

The control client is:

```text
experiments/zorl/standalone/zorl_client.py
```

The client talks to SGLang over HTTP:

```text
/load_lora_adapter
/start_zorl_session
/start_zorl_generation
/generate
/apply_zorl_rewards
/snapshot_zorl_parent
/restore_zorl_parent
/export_zorl_parent
/abort_zorl_generation
/unload_lora_adapter
/flush_cache
```

The current Kubernetes launch files are:

```text
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-smg-router.yaml
experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-standalone-client-job.yaml
experiments/zorl/autoresearch/candidates/ZORL-WORDLE-014-opsd-tf-sharded-smg-export-pop1024.yaml
```

The SGLang ZORL implementation lives in the local fork:

```text
/home/apanda/xorl-sglang-internal/python/sglang/srt/entrypoints/http_server.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/managers/io_struct.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/managers/tokenizer_communicator_mixin.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/managers/scheduler.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/managers/tp_worker.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/model_executor/model_runner.py
/home/apanda/xorl-sglang-internal/python/sglang/srt/lora/lora_manager.py
```

Request path:

```text
HTTP server
  -> TokenizerManager / tokenizer_communicator_mixin.py
  -> Scheduler
  -> TP worker
  -> ModelRunner
  -> LoRAManager
```

The standalone client coordinates the global loop; the SGLang LoRA manager owns
the parent adapter, candidate adapters, random perturbation reconstruction,
update application, snapshots, export, and cleanup.

## ZORL Algorithm

Each step uses antithetic finite differences on the LoRA weights:

```text
for pair i in 0..num_pairs-1:
  positive candidate = parent + sigma * epsilon_i
  negative candidate = parent - sigma * epsilon_i

score both candidates on the train examples
pair_delta_i = reward_positive_i - reward_negative_i
normalize pair deltas
update = mean_i(normalized_delta_i * epsilon_i)
parent <- parent + lr * clip(update, max_update_norm)
```

Current production mode perturbs LoRA B only:

```text
perturbation_mode: b_only
b_sigma: 0.05
learning_rate: 0.01
max_update_norm: 20000
```

The SGLang code also supports `a_and_b`, but that is not the current stable
Wordle path.

## Sharded Population Semantics

The population is globally defined but locally materialized.

For the active run:

```text
global pairs: 512
global population: 1024
num_shards: 16
pairs_per_shard: 32
local candidates per shard: 64
```

Every SGL worker:

1. Loads the same parent LoRA.
2. Starts the same session with the same `session_id`, `seed`, `num_pairs`,
   `b_sigma`, and `perturbation_mode`.
3. Builds the same global candidate plan for each generation.
4. Materializes only its local pair shard for scoring.
5. Receives the full global reward payload during apply.
6. Reconstructs the same full global update from seeds and rewards.
7. Applies that same update to its local parent.

Correctness invariant: all SGL replicas must finish every step with identical
parent LoRA weights. Do not change apply to use local shard-only rewards unless
you are intentionally changing the algorithm; that would make replicas diverge.

The client combines shard responses in `start_zorl_generation_all`. Each
returned candidate includes an `owner_url`. Score requests for a candidate must
go to the owner. The current stable route is `owner_via_smg`, where requests go
to the SMG service with:

```text
X-SMG-Target-Worker: http://zorl-ar-sglang-N.zorl-ar-sglang-headless.apanda.svc.cluster.local:30000
```

Blind SMG round-robin is not correct for sharded candidate LoRAs. The parent
LoRA is replicated and can be probed through SMG, but candidate LoRAs are
owner-local.

## Score Phase

The relevant client functions are:

```text
score_candidates
_ordered_score_batches
_candidate_score_route
_candidate_score_owner_key
input_logprobs_batch_with_lora
```

Teacher-forced Wordle uses `task.build_teacher_forced_example(...)` from:

```text
experiments/zorl/standalone/tasks/wordle.py
```

The teacher-forced payload is a batched `/generate` call:

```json
{
  "input_ids": [[...], [...]],
  "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
  "return_logprob": true,
  "return_text_in_logprobs": false,
  "logprob_start_len": 0,
  "lora_path": "<candidate lora name>"
}
```

The score is derived from the target-token suffix of
`meta_info.input_token_logprobs`. The current reward is approximately:

```text
reward = exp(mean target-token logprob)
```

Current score scheduler behavior:

- Build all `(candidate, example_batch)` jobs.
- Batch examples per candidate using `--teacher-forced-batch-size`.
- Use owner-round-robin ordering so each thread-pool wave spreads across SGL
  owners.
- Use a global `ThreadPoolExecutor(max_workers=score_max_workers)`.
- Gate each candidate-owner with `score_max_workers_per_owner`.
- Route candidate requests through SMG with `X-SMG-Target-Worker`.

Measured stability limits:

```text
safe:   score_max_workers_per_owner=4, teacher_forced_batch_size=8
unsafe: score_max_workers_per_owner=8, teacher_forced_batch_size=8
unsafe: score_max_workers_per_owner=4, teacher_forced_batch_size=16
unsafe: preload_candidates=true with 64 local candidates per SGL
```

Failure signature for unsafe settings:

```text
SGL /generate returns 500 or SMG returns 503 no_available_workers
SGL logs show running_phase_sigquit_handler, CancelledError, SystemExit(0)
Kubernetes restart reason often appears as Completed, exit code 0
```

Why score takes long:

```text
requests per step = 1024 candidates * ceil(128 examples / 8) = 16384 requests
requests per SGL = 64 candidates * 16 example-batches = 1024 requests
current cap = 4 concurrent candidate requests per SGL
observed t_score = about 235s
```

SGL logs show batches such as:

```text
Prefill batch, #new-seq: 8,  #new-token: about 1900
Prefill batch, #new-seq: 16, #new-token: about 3800
Prefill batch, #new-seq: 24, #new-token: about 5700
```

So the stable path does not fill the 16k prefill budget. The next optimization
target is to safely increase in-flight work without triggering the current LoRA
Triton/SGL failure path.

## Apply Phase

The relevant server function is:

```text
/home/apanda/xorl-sglang-internal/python/sglang/srt/lora/lora_manager.py::apply_zorl_rewards
```

Inputs:

```json
{
  "session_id": "...",
  "generation_id": "...",
  "candidate_rewards": [
    {
      "candidate_id": "...",
      "reward_mean": 0.0123,
      "num_rollouts": 1,
      "project_metrics": {}
    }
  ],
  "learning_rate": 0.01,
  "max_update_norm": 20000
}
```

Tokenizer manager fills in the active generation's full candidate spec list
before forwarding to the TP worker.

Apply implementation details:

1. Build `reward_by_candidate`.
2. Validate one score-normalization mode across rewards.
3. Group candidate specs by antithetic pair.
4. Compute raw pair scores: `positive_reward - negative_reward`.
5. Normalize pair scores, currently standard z-score unless disabled by the
   client-side transform strategy.
6. For each used pair, regenerate LoRA noise from the stored seeds.
7. Accumulate a float32 update tensor per LoRA tensor.
8. Compute update norm and optional clip scale.
9. Move parent weights to CPU float32, add update, cast back to original dtype,
   and store contiguous tensors on the parent adapter.
10. Evict the parent from the LoRA memory pool so the next generation uses the
    updated parent.
11. Unload candidate LoRAs and clear the active generation.

Current apply cost is about 105-130 seconds per step. This is large enough to
deserve its own benchmark. It is also easier to isolate than score, because it
does not need `/generate`.

> UPDATE 2026-06-05: apply was diagnosed and optimized. It is **CPU-`randn`
> bound** (~88% of apply is per-tensor Gaussian sampling; the metadata rebuild
> and reassembly are negligible). Fix = generate the noise on GPU with one
> batched `randn` per pair (~480x on the core; full apply should drop to single
> digits). Implemented in the fork behind `XORL_ZORL_NOISE_DEVICE` (default
> `cpu` = unchanged). Offline correctness gates pass. See
> `ZORL_APPLY_GPU_OPTIMIZATION_2026_06_05.md` and the two new benches
> `experiments/zorl/standalone/bench_zorl_apply_math.py` (offline math/perf) and
> `experiments/zorl/standalone/bench_zorl_sglang.py` (cluster gen/score/apply).
> The cluster integration gate (16-replica TP=2 checksum agreement) is NOT yet
> run — do it in a non-production window before setting `=gpu` in production.

Likely apply bottlenecks:

- Every SGL replica recomputes the full 512-pair global update, not just its
  local shard.
- Noise generation is done pair-by-pair and tensor-by-tensor in Python.
- Updates are accumulated in host-visible PyTorch tensors and parent weights
  are moved through CPU float32 before being cast back.
- Cleanup after apply unloads up to 64 materialized candidates per SGL.
- The same apply work is repeated across 16 replicas to preserve identical
  parent state.

Any apply optimization must preserve exact update semantics unless the change
is explicitly gated by a numerical correctness benchmark.

## Candidate LoRA Lifecycle

Generation registers virtual candidates:

```text
lora_path = "__zorl__"
zorl_virtual_candidates[lora_id] = {
  parent_lora_id,
  b_seed,
  a_seed,
  direction,
  b_sigma,
  perturbation_mode
}
```

If `preload_candidates=false`, candidates remain virtual until SGL prepares a
batch containing their `lora_id`. `fetch_new_loras` calls
`_materialize_zorl_candidate`, which builds a candidate adapter as
`parent +/- noise`.

If `preload_candidates=true`, generation materializes every local candidate
immediately and calls `fetch_new_loras` for the full local set. This was unsafe
with 64 local candidates per SGL in the current Qwen3-30B-A3B setup.

Cleanup paths:

- Normal step cleanup happens inside `/apply_zorl_rewards`.
- Failed or interrupted generation should use `/abort_zorl_generation`.
- Parent cleanup uses `/unload_lora_adapter`.

Manual cleanup template:

```bash
SESSION="zorl-wordle-..."
GEN="${SESSION}-family-000000-g000000"
PARENT="${SESSION}/parent"

kubectl exec -n apanda zorl-ar-smg-nqfg9-zb2kw -- sh -lc '
for i in $(seq 0 15); do
  url="http://zorl-ar-sglang-${i}.zorl-ar-sglang-headless.apanda.svc.cluster.local:30000"
  curl -sS -X POST "$url/abort_zorl_generation" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"'"$SESSION"'\",\"generation_id\":\"'"$GEN"'\"}" \
    --max-time 120 || true
  curl -sS -X POST "$url/unload_lora_adapter" \
    -H "Content-Type: application/json" \
    -d "{\"lora_name\":\"'"$PARENT"'\"}" \
    --max-time 120 || true
done
'
```

## Export And Recovery

The client supports exports with:

```text
--export-dir <dir>
--export-interval <N>
```

Current run exports:

```text
exports/best
exports/step-000016
exports/step-000032
...
```

Each export currently reports:

```text
num_tensors=480
num_bytes=120717312
```

Export is implemented by `/export_zorl_parent`; it cannot run while a
generation is active. The client exports after apply/probe boundaries.

For a resumed experiment, load the exported adapter directory as the next
run's `ADAPTER_DIR`. Do not assume SGL in-memory state survives pod restarts.

## Microbenchmark First

The next agent should build a standalone benchmark, not tune through the
autoresearch controller. Put it somewhere like:

```text
experiments/zorl/standalone/bench_zorl_sglang.py
```

It should run against existing SGL/SMG services and emit JSONL/CSV with enough
metrics to hillclimb:

```text
run_id
model
replicas
tp
population
pairs_per_shard
train_size
teacher_forced_batch_size
score_max_workers
score_max_workers_per_owner
preload_candidates
candidate_routing
score_wall_s
score_sequences
score_input_tokens
score_tokens_per_s
apply_wall_s
used_pairs
dropped_pairs
update_norm
restart_count_before
restart_count_after
http_4xx_count
http_5xx_count
smg_503_count
```

The benchmark should have three modes.

### Mode 1: Generation Benchmark

Purpose: isolate `/start_zorl_generation` and virtual candidate registration.

Procedure:

1. Load parent on one or all direct SGL URLs.
2. Start a ZORL session.
3. Call `/start_zorl_generation` with fixed `num_pairs`, `materialization`, and
   `preload_candidates`.
4. Record candidate counts, `candidate_creation_metadata`, and wall time.
5. Abort generation and unload parent.

Sweep:

```text
pairs_per_shard: 1, 2, 4, 8, 16, 32
preload_candidates: false, true
```

Do not promote `preload_candidates=true` until it passes scoring concurrency;
it can pass generation and still crash under `/generate`.

### Mode 2: Score Benchmark

Purpose: isolate teacher-forced `/generate` throughput under candidate LoRAs.

Procedure:

1. Set up one generation exactly like production.
2. Build a fixed list of input ID sequences with the same length distribution
   as Wordle teacher-forced traces. Use real `wordle.py` examples if possible;
   synthetic fixed-length token IDs are acceptable only for quick scheduler
   tests.
3. Issue batched `/generate` requests with `return_logprob=true`,
   `max_new_tokens=1`, `logprob_start_len=0`, and candidate `lora_path`.
4. Route via SMG target header or direct owner URL.
5. Record wall time, request latencies, status codes, SGL restart counts, and
   effective tokens/s.
6. Abort generation and unload parent.

Required baseline:

```text
num_shards=16
pairs_per_shard=32
population=1024
train_size=128
teacher_forced_batch_size=8
score_max_workers=64
score_max_workers_per_owner=4
candidate_routing=owner_via_smg
preload_candidates=false
```

Useful single-SGL smoke before full pool:

```text
one SGL, pairs_per_shard=32, local candidates=64
N concurrent candidate requests, each with batch=8
N=1 must pass
N=4 passed
N=8 failed in full-pool production
```

Primary sweep axes:

```text
score_max_workers_per_owner: 1, 2, 4, then maybe 6 only after stability work
teacher_forced_batch_size: 4, 8, 12, 16
request ordering: owner_round_robin vs grouped-by-owner
route: owner_via_smg vs direct owner
candidate reuse: same 4 LoRAs repeated vs rotating across 64 local LoRAs
preload/lazy: false vs partial preload if implemented
```

Stop a candidate immediately if any SGL restart count increases.

### Mode 3: Apply Benchmark

Purpose: isolate `/apply_zorl_rewards`.

Procedure:

1. Load parent and start session on all replicas.
2. Start sharded generation but do not score.
3. Build deterministic synthetic rewards for every global candidate.
4. Call `/apply_zorl_rewards` on every direct SGL URL with the same payload.
5. Validate all responses agree on `used_pairs`, `dropped_pairs`,
   `update_norm`, `pair_delta_mean`, and `pair_delta_std`.
6. Repeat for multiple steps or reset session between trials.

Required baseline:

```text
num_pairs=512
candidate_rewards=1024
score_normalization=standard
learning_rate=0.01
max_update_norm=20000
perturbation_mode=b_only
```

Sweep:

```text
num_pairs: 32, 64, 128, 256, 512, 1024
perturbation_mode: b_only first; a_and_b only after b_only is characterized
parallel apply fanout: sequential urls vs ThreadPoolExecutor over 16 urls
candidate cleanup wait timeout
noise/update implementation variants
```

Metrics:

```text
apply_wall_s end-to-end from client
server-side apply time if instrumented
candidate cleanup count
skipped_candidate_cleanup
used_pairs/dropped_pairs
update_norm
parent checksum before/after if implemented
```

## Correctness Gates

Do not promote a performance change without a correctness gate matching the
surface area of the change.

Score-path changes:

- Same candidate, same inputs, same `lora_path` should return the same
  teacher-forced rewards within tolerance before and after the change.
- Coverage must include at least one parent LoRA and multiple candidate LoRAs.
- If routing changes, prove every candidate request goes to the owner that has
  the candidate.

Apply-path changes:

- For a fixed generation and fixed synthetic rewards, compare parent export
  tensors before and after the optimization against the current implementation.
- Compare `used_pairs`, `dropped_pairs`, `zero_score_pairs`, `pair_delta_mean`,
  `pair_delta_std`, `update_norm`, `grad_norm`, and `update_clip_scale`.
- Validate all SGL replicas produce identical parent checksums after apply.

Integration gate:

- Run at least two full ZORL steps with population 1024, train size 128, no SGL
  restarts, `used_pairs=512`, and `dropped_pairs=0`.
- Keep W&B logging enabled for production launches.
- Keep exports enabled so work is not lost.

If model numerics, routing, LoRA kernels, or update math change, use the
existing K3/static-trace style correctness workflow or build an equivalent
trace replay for the teacher-forced ZORL path.

## Known Problems And Hypotheses

### Full local preload is unsafe

`preload_candidates=true` with 64 local candidate LoRAs per SGL crashed the
Triton LoRA path under concurrent scoring. Lazy virtual candidates are stable
at the current cap. A promising optimization is partial preload or staged
preload, but it needs a benchmark that combines preload plus scoring.

### Per-owner concurrency is the hard stability limit

The stable cap is four candidate requests per SGL. Eight per SGL restarted
workers. Batch size 16 also restarted workers even with four requests per SGL.

The next agent should determine whether the real limit is:

- unique LoRAs concurrently materialized,
- total active LoRAs in the memory pool,
- total sequences per prefill batch,
- total prompt tokens per prefill batch,
- Triton MoE LoRA kernel shape diversity,
- or SGL scheduler/HTTP cancellation behavior.

Design the score benchmark to isolate those axes.

### Apply recomputes full update on every replica

This is correct but expensive. Possible optimization directions:

- Vectorize noise generation and accumulation.
- Cache deterministic noise tensors for a generation or across generations.
- Accumulate on GPU instead of CPU if memory permits.
- Avoid repeated CPU round trips for parent weights.
- Compute update once and broadcast the updated parent or delta to all SGL
  replicas, then verify identical parent checksums.
- Reduce cleanup overhead by deferring candidate unloads safely.

Any cross-replica update sharing must preserve the invariant that every worker
serves the same parent before the next generation.

### The current score batches underfill prefill

Stable batches often show 8-24 sequences and 1.9k-5.7k tokens per prefill on
SGL0, below the configured 16k prefill cap. More work per batch would help, but
naive batch 16 was unstable. This points to either LoRA-kernel instability or
memory-pool/scheduler pressure before pure prefill capacity is reached.

## Launch And Verification Commands

Launch the current production candidate:

```bash
python -m experiments.zorl.autoresearch.controller launch \
  --candidate experiments/zorl/autoresearch/candidates/ZORL-WORDLE-014-opsd-tf-sharded-smg-export-pop1024.yaml
```

Focused local checks after client edits:

```bash
uv run ruff check \
  experiments/zorl/standalone/zorl_client.py \
  tests/experiments/test_zorl_autoresearch_controller.py

python -m pytest tests/experiments/test_zorl_autoresearch_controller.py -q
```

Health check:

```bash
kubectl exec -n apanda zorl-ar-smg-nqfg9-zb2kw -- \
  curl -fsS --max-time 60 http://127.0.0.1:8080/health_generate
```

SGL restart check:

```bash
kubectl get pods -n apanda -l app=zorl-ar-sglang \
  -o jsonpath='{range .items[*]}{.metadata.name} {.status.containerStatuses[0].restartCount} {.status.containerStatuses[0].ready}{"\n"}{end}'
```

SGL scoring activity:

```bash
kubectl logs -n apanda zorl-ar-sglang-0 --tail=80 \
  | rg 'Prefill batch|POST /generate|ERROR|SIGQUIT|RuntimeError|illegal'
```

## Do Not Regress These

- Population 1024 support with sharded ownership.
- Resampled 128-example train batch from a 512-example pool.
- W&B project `zorl`.
- Export support for `best` and periodic step adapters.
- Owner-routed candidate scoring through SMG target headers.
- `used_pairs=512`, `dropped_pairs=0` on successful production steps.
- No SGL restart count changes during a promoted benchmark or production run.
- Parent update agreement across all SGL replicas.

