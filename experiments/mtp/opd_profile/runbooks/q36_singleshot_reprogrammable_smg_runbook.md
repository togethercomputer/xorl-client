# Qwen3.6 SingleShot MTP Reprogrammable SMG Runbook

Status: draft operating plan, updated 2026-06-05 after capacity and generator
audit.

This runbook replaces the one-off three-pod SingleShot MTP ladder with a
reprogrammable-slot workflow that can scale student sampling, teacher prefill,
trainer forward/backward, and weight sync independently. It is intentionally
self-contained: it summarizes the research goal, data artifacts, current
baseline evidence, infrastructure, verification checks, and the next launch
ladder.

## Goal

Train Qwen3.6-35B-A3B SingleShot MTP on Coderforge assistant-turn prompts with
enough infrastructure parallelism to find the actual bottleneck. The target
steady-state loop is:

```text
student native-MTP rollout fanout
  -> teacher hidden-cache prefill fanout
  -> XORL forward_backward(opd_loss), possibly chunked/pipelined
  -> optimizer step
  -> all active student sampler endpoints receive fresh weights
```

The immediate goal is not to maximize context length. The immediate goal is to
make a 1024 prompt / 64 generated-token run throughput-informative by removing
the known sampler/router and serial-driver artifacts.

## Baseline Evidence

The latest one-off baseline run was `q36k4d06042335`:

- Coord dir:
  `/shared/opd-coord/q36k4d06042335`
- Profile:
  `/shared/opd-coord/q36k4d06042335/artifacts/qwen36_confadapt_4node_opd_profile.jsonl`
- Dataset:
  `/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream`
- Fit knobs:
  `OPD_PROMPT_DATASET_PROMPT_LEN=1024`, `OPD_MAX_NEW_TOKENS=64`,
  `OPD_OPTIM_LR=1e-6`, `OPD_FULL_FT_LR=1e-6`
- Topology:
  4 trainer nodes plus one 2-GPU student SGLang and one 2-GPU teacher SGLang.

Latest measured rows:

| Step | Total s | Student s | Teacher s | Fwd/Bwd Roundtrip s | Sync s | Pipeline | Prompts | Sampler Endpoints |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 356.44 | 3.63 | 118.10 | 165.15 | 61.92 | false | 1 | 1 |
| 1 | 109.69 | 1.24 | 0.63 | 81.08 | 21.38 | false | 1 | 1 |
| 2 | 362.28 | 3.92 | 0.61 | 326.77 | 24.17 | false | 1 | 1 |
| 3 | 142.62 | 1.42 | 0.59 | 116.65 | 18.48 | false | 1 | 1 |

What this proves:

- The 1024/64 fit point runs and validates native trace coverage.
- The first row is cold-start/compile/cache-heavy and should not be used alone.
- Even after warmup, the run is not throughput-shaped: one prompt per optimizer
  step, one student endpoint, one teacher endpoint, `opd_pipeline_enabled=false`,
  and one synced sampler endpoint.
- The current bottleneck is not a single phase. Warm rows still spend about
  81-327s in trainer forward/backward and about 21-24s in sync, while sampling
  and teacher prefill are artificially small because there is only one prompt.
- The step-2 trainer log shows repeated RDMA endpoint/device errors during
  weight sync before a nominally successful sync. Treat this as a P2P health
  warning for the one-off topology, not as a promotable throughput result.
- Repeating this one-off topology will not answer the scale question.
- Kubernetes resources for `q36k4d06042335` were deleted after row 3 to release
  the 36 GPUs. Artifacts remain under `/shared/opd-coord/q36k4d06042335`.

Known failed memory point:

- `1536/64` OOMed in `q36k4c06042308` in the fused linear-attention norm/gate
  path. Do not retry `1536/64` until a memory lever changes.

## Live Reprogrammable Evidence

The active `er-opd-q36-35b-slots` fleet is already proving some of the
infrastructure we need, but it is running the chat-completions PTC loop from
`/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`,
not this repo's native SingleShot MTP driver.

Latest observed non-SingleShot profile rows from the trainer-head slot log:

| Step | Total s | Fwd/Bwd s | Prepare Window s | Sync s | Samples | Sampler Endpoints | Balance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 127.31 | 105.19 | 119.26 | 2.97 | 256 | 2 | 1.00 |
| 2 | 123.08 | 100.38 | 115.27 | 2.81 | 256 | 2 | 1.00 |
| 3 | 115.58 | 89.40 | 107.25 | 2.97 | 256 | 2 | 1.00 |
| 4 | 112.82 | 86.97 | 104.53 | 2.91 | 256 | 2 | 1.00 |

What this proves:

- SMG `round_robin` over two sampler endpoints works in the warm fleet.
- Both registered sampler endpoints can receive fresh weights and report
  `sync_endpoint_success_count=2`.
- The active PTC stack uses an 8-node, 64-GPU trainer because that is how the
  existing reprogrammable fleet was provisioned. This should not be copied as
  the default SingleShot MTP trainer size.
- Two endpoints are still not the target scale. Even this non-SingleShot loop
  spends about 105s in the prepare window and about 87s in trainer
  forward/backward at step 4.
- Do not treat these rows as native SingleShot MTP results. Treat them as
  operational evidence for the reprogrammable substrate.

## Research Hypotheses

1. The one-off pod topology hides the real throughput frontier because the OPD
   driver is serial and underfilled.
2. SingleShot MTP quality runs need multiple serialized student samplers behind
   a router rather than higher per-pod SGLang concurrency. Prior Qwen3.6
   reprogrammable-slot runs found batched student decoding can produce invalid
   repeated/numeric artifacts; keep each sampler conservative and scale by
   replica count.
3. Student routing must be load-balanced for on-policy science runs. Use SMG
   `round_robin` when there is more than one student sampler. `cache_aware` can
   collapse same-prefix rollout traffic onto one backend.
4. Teacher hidden-cache prefill should be independently fanout-capable. If the
   current SingleShot MTP driver cannot route `/teacher_hidden_cache` through
   SMG, add a native endpoint fanout or SMG route before claiming teacher scale.
5. Pipeline mode is required once `prompts_per_step > chunk_size`. The useful
   question is which phase dominates when we overlap chunk preparation with
   trainer forward/backward and sync all active sampler endpoints once per
   optimizer step.

## Data Plan

Current validated dataset checkpoint:

```text
/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream
```

Summary:

- Source: first 16 Coderforge tokenized parquet shards from
  `/shared/huggingface/hub/datasets--togethercomputer--CoderForge-Preview/snapshots/060fca96cf723b2ebab3181e9e59fafd273df3cb/trajectories-tokenized_qwencoder`
- Retokenizer:
  `/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
- Rows: 130429 assistant turns.
- Train rows: 128034.
- Eval rows: 2395.
- Malformed marker rate: 0.0.
- Prompt truncation rate: about 0.937.
- Target truncation rate: about 0.223.
- Prompt column: `prompt_ids`.
- Prompt tail was spot-checked to end in the Qwen3.6 assistant header
  `[248045, 74455, 198]`.

Use this dataset for throughput-shape smokes. Do not use the old pilot64 bucket
for quality. Before any long quality run, materialize the full all-shard dataset
with the same builder and streaming writer.

Throughput smoke settings:

```text
prompt_dataset_path=/shared/opd-datasets/coderforge_assistant_turns_ctx8k_tgt256_qwen36_files16_stream
prompt_dataset_type=parquet
prompt_dataset_column=prompt_ids
prompt_dataset_prompt_len=1024
max_new_tokens=64
min_target_tokens=1
lr=1e-6
k_toks=4
conf_threshold=0.6
```

Quality settings stay gated until throughput is sane:

```text
prompt_dataset_prompt_len=8192
max_new_tokens=256
```

## Infrastructure Model

The current warm reprogrammable fleet is in the separate operator repo:

```text
/home/apanda/xorl-apanda-dev-opd-port
```

The generator and operational references there are:

```text
experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md
experiments/opd_profile/runbooks/smg_router_swap.md
experiments/opd_profile/runbooks/pipelined_teacher_prefill.md
```

The active stack name is:

```text
er-opd-q36-35b-slots
```

Current warm slot roles:

| Role | Purpose | GPU |
|---|---|---:|
| `sglang-0` | student SGLang, port 30060 | 8 |
| `sglang-1` | optional dedicated second student sampler | 8 when rendered |
| `dispatch` | SMG router for student sampling, port 8080 | 0 |
| `teacher-sglang-0` | teacher hidden-cache SGLang, port 30000 | 8 |
| `teacher-sglang-1` | spare teacher or second student in `spare-teacher1` layout | 8 |
| `teacher-smg` | SMG router for teacher services, port 8080 | 0 |
| `trainer-head` | XORL API/trainer head | 8 |
| `trainer-worker-1..7` | XORL trainer workers | 56 |

The already-running warm stack holds 88 GPUs. The one-off `q36k4d` run held
another 36 GPUs while active, then was deleted after baseline row 3. At the
2026-06-05 capacity snapshot, the cluster had 397 allocatable GPUs, 306 GPUs
used by live pods, and 91 GPUs free, with seven completely free 8-GPU nodes:
`research-common-h100-036`, `research-common-h100-040`,
`research-common-h100-047`, `research-common-h100-049`,
`research-common-h100-071`, `research-common-h100-116`, and
`research-common-h100-117`.

All seven fully free nodes currently report `node-group=default`. The current
operator generator renders:

- dedicated `sglang-*` student pods with `nodeSelector: node-group=nccl`;
- trainer pods with `nodeSelector: node-group=nccl`;
- `teacher-sglang-*` pods with `nodeSelector: node-group=default`;
- the `teacher-sglang-1` spare role on `default`, which can already act as a
  second student sampler under `spare-teacher1`.

Important correction: we can use either `node-group=nccl` or
`node-group=default` for every GPU worker class: student SGLang samplers,
teacher SGLang hidden-cache servers, and XORL trainer head/workers. The node
pool should be an explicit placement knob, not encoded as a semantic property of
the role. CPU-only SMG/router pods can also run on either pool, though `default`
is usually preferable to avoid consuming scarce NCCL placement.

Implication: the current free full `default` nodes are valid capacity for
additional students, teachers, or even trainer workers once the new SingleShot
generator exposes per-role node-group controls and the generated stack passes
route, P2P weight-sync, and throughput probes. Do not request many new
dedicated `sglang-*` student pods with the old generator unchanged; it will
target `nccl` nodes and may sit pending despite usable `default` capacity.

All GPU pod templates must include:

```yaml
metadata:
  labels:
    team: turbo
```

Do not manually set `schedulerName` or Volcano queue labels unless explicitly
overriding the cluster queue behavior.

## Capacity-Driven Scaling Plan

The research question is not "can one more sampler help?" The useful question is
"which component saturates first as we add enough student and teacher endpoints
to keep trainer forward/backward continuously fed?" The plan is therefore to
scale roles independently and stop when the profile says another role, not the
current one, is the limiter.

Current hard generator limits:

- `sampler_layout=spare-teacher1` supports at most two student sampler
  endpoints: `sglang-0` plus `teacher-sglang-1`.
- `sampler_layout=dedicated` can render more `sglang-*` endpoints, but they are
  currently forced to `node-group=nccl` by the old generator even though
  `default` nodes are valid for student workers.
- Teacher fanout is capped at the two hard-coded `teacher-sglang-0/1` roles; in
  `spare-teacher1`, `teacher-sglang-1` is not available as a teacher.
- `teacher-smg` routes only to `teacher-sglang-0` in `spare-teacher1`.

Required generator extension before using the seven free full nodes well:

```text
student_replicas=<N>
teacher_replicas=<M>
trainer_nodes=4
student_node_group=default|nccl
teacher_node_group=default|nccl
trainer_node_group=default|nccl
router_node_group=default|nccl
student_smg_policy=round_robin
teacher_smg_policy=round_robin
native_route_mode=smg|python_router|direct_fanout
```

The rendered roles should become:

```text
sglang-0..N-1              student native-MTP samplers
student-smg or dispatch    SMG router over all student samplers
teacher-sglang-0..M-1      teacher hidden-cache prefill servers
teacher-smg                SMG router over all teacher servers, if native route works
trainer-head/worker-*      4-node XORL trainer to start; scale only from evidence
```

For the 1024 prompt / 64 generated-token SingleShot MTP ladder, use four
trainer nodes, 32 GPUs, as the default trainer allocation. The one-off
`q36k4a/q36k4b/q36k4d` family already proved this fit point runs on a 4-node
trainer; the 1536/64 OOM is a memory-boundary data point, not evidence that the
1024/64 throughput ladder needs 64 trainer GPUs. Spend the extra GPUs on
student and teacher endpoints first.

Feasible fresh-stack role targets with the current 91 free GPUs:

| Target | Trainer GPUs | Student Endpoints | Teacher Endpoints | Total GPUs | Increment Over S0 | Purpose |
|---|---:|---:|---:|---:|---:|---|
| S0 | 32 | 2 | 1 | 56 | 0 | native route compatibility on a right-sized trainer |
| S1 | 32 | 2 | 3 | 72 | +16 | prove teacher fanout and teacher-SMG/direct fanout |
| S2 | 32 | 4 | 2 | 80 | +24 | first balanced student+teacher throughput smoke |
| S3 | 32 | 6 | 4 | 112 | +56 | use most of the currently free full-node budget for inference fanout |
| S4 | 32 | 8 | 4 | 128 | +72 | only after S3 shows student sampling remains limiting |

Start S0/S1 for route correctness, not because two samplers are enough. Move to
S2/S3 as soon as native `/generate` and `/teacher_hidden_cache` routing is
proven. Keep the trainer at 32 GPUs for the first scaled inference rungs so the
profile can tell us whether inference or training is the limiter. Increase to a
64-GPU trainer only if S3/S4 show inference phases are hidden and
`forward_backward_roundtrip_s` dominates steady-state wall time after pipeline
overlap is working.

Scaling decision rules:

- Add student endpoints when `student_sampling_s` or student SMG queue/load is
  the largest non-overlapped component, or when per-endpoint sampler GPU util is
  high while trainer waits.
- Add teacher endpoints when `teacher_prefill_s`, teacher cache write time, or
  pipeline prepare wait dominates, or when teacher GPU util is high while
  trainer waits.
- Increase `OPD_PIPELINE_PREFETCH_CHUNKS` or
  `OPD_PIPELINE_TEACHER_CONCURRENCY` before adding teacher GPUs if teacher GPUs
  are underutilized.
- Increase `pipeline_chunk_size` when trainer call overhead dominates many
  chunked `forward_backward` calls.
- Add trainer nodes or change trainer topology only after S2/S3 show
  inference-side phases are no longer the dominant wall-time limiter.

## Reprogrammable Slot Operations

The slot pods are long-lived GPU allocations. A pod can be `Running` even when
its child workload has stopped. Inspect the control status, not only pod phase.

Set standard variables in the operator repo:

```bash
cd /home/apanda/xorl-apanda-dev-opd-port
GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
MANIFEST=experiments/opd_profile/k8s/generated/er-opd-q36-35b-slots.yaml
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
NS=apanda
STACK=er-opd-q36-35b-slots
```

Check slot child processes:

```bash
python "$GENERATOR" status
```

Render/apply the warm slot manifest only when the fleet needs to be scheduled or
moved:

```bash
python "$GENERATOR" render-manifest --output "$MANIFEST" \
  --sampler-replicas 2 --sampler-layout spare-teacher1
kubectl apply -n "$NS" -f "$MANIFEST"
```

Normal recipe iteration should keep warm inference roles alive and rewrite only
trainer control scripts:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" status
python "$GENERATOR" write-trainer-control \
  --config <CONFIG> \
  --num-steps <N> \
  --prompts-per-step <P> \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

Restart student samplers plus dispatch:

```bash
python "$GENERATOR" write-student-inference-control \
  --sampler-replicas 2 --sampler-layout spare-teacher1
```

Restart dispatch only:

```bash
python "$GENERATOR" write-dispatch-control \
  --sampler-replicas 2 --sampler-layout spare-teacher1
```

Stop trainer roles only:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
```

Stop all child workloads while keeping GPU slots allocated:

```bash
python "$GENERATOR" stop-control --remove-run
```

Delete Kubernetes controllers only when deliberately freeing GPUs.

## SMG Router Requirements

SMG binary:

```text
/home/apanda/smg-together-thunderagent-port/target/debug/smg
```

Student dispatch policy:

- one student endpoint: `cache_aware` is allowed.
- more than one student endpoint: use `round_robin`.

`spare-teacher1` student endpoints:

```text
http://er-opd-q36-35b-slots-sglang-0:30060
http://er-opd-q36-35b-slots-teacher-sglang-1:30000
```

Teacher hidden-cache endpoint in `spare-teacher1`:

```text
http://er-opd-q36-35b-slots-teacher-sglang-0:30000
```

Do not route teacher hidden-cache requests to `teacher-sglang-1` while it is
acting as a student sampler.

SMG startup check must wait for model discovery, not only `/health`:

```bash
kubectl exec -n "$NS" "$STACK-dispatch" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:8080/v1/models'
```

Expected model:

```text
Qwen/Qwen3.6-35B-A3B
```

Per-backend SGLang checks:

```bash
kubectl exec -n "$NS" "$STACK-sglang-0" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:30060/v1/models'
kubectl exec -n "$NS" "$STACK-teacher-sglang-1" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:30000/v1/models'
```

Native SingleShot MTP compatibility gate:

The current local driver, `scripts/opd/run_opd_pipeline.py`, calls raw SGLang
endpoints:

```text
/generate
/model_info
/teacher_hidden_cache
```

The documented SMG path was validated for OpenAI-compatible
`/v1/chat/completions`. Before routing SingleShot MTP through SMG, run a native
compatibility probe:

1. POST one native-MTP `/generate` payload through SMG dispatch.
2. Verify the response includes the same native trace fields seen from direct
   SGLang.
3. POST one `/teacher_hidden_cache` payload through teacher SMG if teacher
   routing is desired.
4. If either native route is unsupported, do not fake the result. Add one of:
   a native SGLang-path proxy route to SMG, a thin Python native router, or
   multi-endpoint fanout directly inside `run_opd_pipeline.py`.

## Pipeline Driver Plan

Local pipeline-capable driver:

```text
scripts/opd/run_opd_pipeline.py
```

Relevant knobs:

```text
OPD_PIPELINE_CHUNK_SIZE
OPD_PIPELINE_PREFETCH_CHUNKS
OPD_PIPELINE_TEACHER_CONCURRENCY
OPD_PROMPT_DATASET_NUM_PROMPTS
```

Pipeline mode activates only when:

```text
OPD_PIPELINE_CHUNK_SIZE > 0
and number_of_prompts_this_step > OPD_PIPELINE_CHUNK_SIZE
```

The one-off q36k4d run had `OPD_PROMPT_DATASET_NUM_PROMPTS=1` and
`OPD_PIPELINE_CHUNK_SIZE=1`, so pipeline mode was impossible.

Initial pipeline ladder after native route compatibility:

| Rung | Scale Target | Prompts/step | Chunk size | Prefetch chunks | Teacher concurrency | Student Endpoints | Teacher Endpoints | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P0 | S0 | 2 | 1 | 2 | 1 | 2 | 1 | prove native fanout and sync both existing samplers |
| P1 | S1 | 4 | 1 | 2 | 2 | 2 | 3 | prove teacher fanout hides teacher prefill |
| P2 | S2 | 8 | 1 | 4 | 2 | 4 | 2 | first balanced student+teacher throughput smoke |
| P3 | S2 | 8 | 2 | 4 | 2 | 4 | 2 | reduce fwd/bwd call count per step |
| P4 | S3 | 16 | 2 | 4 | 4 | 6 | 4 | use the currently feasible seven-node expansion |
| P5 | S3 | 32 | 4 | 4-8 | 4 | 6 | 4 | only if P4 remains trace-clean and inference-fed |

Each rung keeps:

```text
prompt_len=1024
max_new_tokens=64
k_toks=4
conf_threshold=0.6
lr=1e-6
```

Only increase prompt length or generated length after the throughput substrate
has stable correctness and clear phase timings.

Do not stop at P0/P1 if they pass. They are compatibility gates. The first rung
that can answer the user's sampler-saturation concern is P2, and the first rung
that uses the currently free full-node budget well is P4.

## Required Code/Manifest Changes

The current reprogrammable generator drives
`/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`.
For SingleShot MTP we need a reprogrammable trainer script that runs this repo's
`scripts/opd/run_opd_pipeline.py` or an equivalent native-MTP-aware driver.

Minimum implementation work:

1. Port or derive a generator in this repo:

   ```text
   experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py
   ```

   It can start from:

   ```text
   /home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
   ```

2. Preserve the slot-agent control model:

   ```text
   /shared/opd-control/<stack>/<role>/run.sh
   /shared/opd-control/<stack>/<role>/stop
   /shared/opd-control/<stack>/<role>/status
   /shared/opd-control/<stack>/<role>/logs/
   ```

3. Add SingleShot-specific trainer controls:

   ```text
   OPD_PROMPT_DATASET_PATH
   OPD_PROMPT_DATASET_TYPE=parquet
   OPD_PROMPT_DATASET_COLUMN=prompt_ids
   OPD_PROMPT_DATASET_PROMPT_LEN=1024
   OPD_PROMPT_DATASET_NUM_PROMPTS=<P>
   OPD_MTP_K_TOKS=4
   OPD_MTP_CONF_THRESHOLD=0.6
   OPD_MAX_NEW_TOKENS=64
   OPD_FULL_FT_LR=0.000001
   OPD_OPTIM_LR=0.000001
   OPD_PIPELINE_CHUNK_SIZE=<C>
   OPD_PIPELINE_PREFETCH_CHUNKS=<K>
   OPD_PIPELINE_TEACHER_CONCURRENCY=<T>
   ```

4. Add role-count and routing controls:

   ```text
   STUDENT_REPLICAS=<N>
   TEACHER_REPLICAS=<M>
   TRAINER_NODES=4
   STUDENT_NODE_GROUP=default|nccl
   TEACHER_NODE_GROUP=default|nccl
   TRAINER_NODE_GROUP=default|nccl
   ROUTER_NODE_GROUP=default|nccl
   STUDENT_SMG_POLICY=round_robin
   TEACHER_SMG_POLICY=round_robin
   OPD_STUDENT_BASE_URLS=http://...
   OPD_TEACHER_BASE_URLS=http://...
   OPD_NATIVE_ROUTE_MODE=smg|python_router|direct_fanout
   ```

   The current `spare-teacher1` layout is only a compatibility bridge. It is
   not the target topology.

5. Register every student sampler endpoint with the trainer and require sync
   success for every endpoint before continuing:

   ```text
   sync_endpoint_count == expected_sampler_count
   sync_success == true
   sync_endpoint_success_count == expected_sampler_count
   ```

6. For multi-endpoint sync, keep serial endpoint sync enabled until combined
   P2P sync is revalidated:

   ```text
   XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1
   ```

7. Expose node-pool placement for all role classes, not only student and
   teacher roles:

   ```text
   STUDENT_NODE_GROUP=default|nccl
   TEACHER_NODE_GROUP=default|nccl
   TRAINER_NODE_GROUP=default|nccl
   ROUTER_NODE_GROUP=default|nccl
   ```

   The generated manifest should record the selected node group per role in a
   label or env var so profile artifacts can be interpreted against placement.
   `default` and `nccl` are both valid pools for student, teacher, and trainer
   GPU workers; choose based on live availability and measured sync/throughput,
   not role name.

8. Emit or preserve profile fields:

   ```text
   opd_pipeline_enabled
   opd_pipeline_chunks
   opd_pipeline_chunk_size
   opd_pipeline_prepare_wait_s
   opd_pipeline_chunk_prepare
   opd_pipeline_chunk_forward_backward
   student_sampling_s
   teacher_prefill_s
   forward_backward_roundtrip_s
   sync_inference_weights_s
   sync_endpoint_count
   sync_success
   rollout/native_trace_covered_generated_all
   rollout/native_trace_coverage_failure_count
   rollout/max_repeated_token_run
   rollout/unique_token_fraction
   mtp/replay_trace_covered_all_targets
   student_endpoint_count
   teacher_endpoint_count
   student_router_policy
   teacher_router_policy
   student_smg_worker_success_delta_min
   student_smg_worker_success_delta_max
   teacher_smg_worker_success_delta_min
   teacher_smg_worker_success_delta_max
   student_gpu_util_mean
   teacher_gpu_util_mean
   student_node_group
   teacher_node_group
   trainer_node_group
   ```

9. Add a native-route probe command to the generator so each rendered stack can
   validate direct backend, student router, direct teacher, and teacher router
   behavior before a long run starts. The probe should fail closed if native
   trace fields or hidden-cache metadata are missing.

## Profiling Procedure

For every rung, collect:

```bash
PROFILE=<run-dir>/opd_profile.jsonl
python - <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print("rows", len(rows))
for r in rows[-5:]:
    print({
        "step": r.get("step"),
        "total": r.get("step_total_s"),
        "sample": r.get("student_sampling_s"),
        "teacher": r.get("teacher_prefill_s"),
        "fb": r.get("forward_backward_roundtrip_s"),
        "sync": r.get("sync_inference_weights_s"),
        "pipeline": r.get("opd_pipeline_enabled"),
        "chunks": r.get("opd_pipeline_chunks"),
        "sync_endpoints": r.get("sync_endpoint_count"),
        "trace_ok": r.get("rollout/native_trace_covered_generated_all"),
        "repeat": r.get("rollout/max_repeated_token_run"),
        "unique": r.get("rollout/unique_token_fraction"),
    })
PY "$PROFILE"
```

SMG model and metrics:

```bash
kubectl exec -n "$NS" "$STACK-dispatch" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:8080/v1/models'
kubectl exec -n "$NS" "$STACK-dispatch" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:29000/metrics | grep smg_router_upstream'
```

For teacher SMG, repeat the same model and metrics checks against
`$STACK-teacher-smg` when teacher fanout is enabled.

Per-sampler load:

```bash
kubectl exec -n "$NS" "$STACK-sglang-0" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:30060/get_load || true'
kubectl exec -n "$NS" "$STACK-teacher-sglang-1" -- \
  bash -lc 'curl -m 5 -fsS http://127.0.0.1:30000/get_load || true'
```

GPU utilization:

```bash
kubectl exec -n "$NS" "$STACK-sglang-0" -- \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
kubectl exec -n "$NS" "$STACK-teacher-sglang-0" -- \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

For S2/S3, sample every student and teacher role, not just `sglang-0` and
`teacher-sglang-0`. The promotion table must include:

```text
student_worker_count
student_worker_success_delta_min/max
teacher_worker_count
teacher_worker_success_delta_min/max
per-role GPU util min/mean/max
pending/running request counts from /get_load
```

Trainer health:

```bash
tail -160 "$(ls -td "$CONTROL_ROOT/trainer-head/logs/"*-run.log | head -1)"
```

Promote a rung only if:

- all expected pods and child processes are healthy;
- every registered sampler endpoint receives synced weights;
- no native trace coverage failure;
- no failed `/model_info`, `/generate`, or `/teacher_hidden_cache` route;
- at least two non-warmup profile rows exist;
- the profile row says the intended pipeline and sampler count were used;
- the profile row says the intended teacher endpoint count was used;
- SMG worker deltas show no endpoint is idle unless deliberately held out;
- repeated-token metrics are understood, not ignored.

## Failure Handling

If SMG `/v1/models` returns `unknown`:

```bash
python "$GENERATOR" write-dispatch-control \
  --sampler-replicas 2 --sampler-layout spare-teacher1
```

If a sampler has stale P2P receiver state:

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" write-student-inference-control \
  --sampler-replicas 2 --sampler-layout spare-teacher1
```

Symptoms include:

```text
Peer nic not found
received packet mismatch
batch_transfer failed
A P2P weight update for group 'weight_sync_group' is already in progress
```

If multi-endpoint sync fails:

- keep `XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1`;
- write a failure row with `sync_success=false`;
- stop the trainer roles;
- restart student inference before the next science attempt.

If `1536/64` or `8192/256` OOMs:

- do not retry unchanged;
- reduce SingleShot truncation length or `pad_to_multiple`;
- test a memory-changing topology;
- keep the throughput substrate fixed while changing the memory lever.

If the native SMG route is unsupported:

- do not launch a long run through chat-completions as a substitute;
- implement native `/generate` and `/teacher_hidden_cache` routing or direct
  endpoint fanout in `run_opd_pipeline.py`;
- validate route parity on one prompt before reusing the warm stack.

If expanded student pods remain pending:

- check the rendered per-role node-group setting, not just the role name;
- if only `default` nodes are free, render students, teachers, trainers, or any
  subset of them with `*_NODE_GROUP=default`;
- if only `nccl` nodes are free, render the corresponding roles with
  `*_NODE_GROUP=nccl`;
- keep either placement only if P2P weight sync, native generation probes,
  teacher hidden-cache probes, and steady-state profile rows pass.

## Proposed Launch Ladder

### Phase A: Cleanup and Baseline Closeout

Status: complete as of 2026-06-04.

1. Captured four `q36k4d` profile rows.
2. Stopped/deleted `q36k4d` one-off resources after enough baseline evidence
   was recorded.
3. Recorded the cleanup and metrics in
   `docs/notes/singleshot_mtp_research_runbook.md`.

### Phase B: Native Route Compatibility

1. With warm reprogrammable slots up, probe direct student SGLang native MTP.
2. Probe student SMG dispatch with the same native MTP payload.
3. Probe direct teacher `/teacher_hidden_cache`.
4. Probe teacher SMG with the same hidden-cache payload if teacher fanout is in
   scope.
5. Decide: SMG native route works, or implement adapter/fanout.

### Phase C: Two-Sampler Reprogrammable Smoke

Use a right-sized S0 stack when possible. If the existing warm PTC stack is
reused for convenience, remember that its 64-GPU trainer is inherited
infrastructure, not the target SingleShot trainer allocation. This phase is
required, but it is not a throughput answer.

```text
sampler_replicas=2
sampler_layout=spare-teacher1
student dispatch policy=round_robin
teacher=teacher-sglang-0 direct
prompts_per_step=2
pipeline_chunk_size=1
pipeline_prefetch_chunks=2
pipeline_teacher_concurrency=1
num_steps=3
```

Pass gate:

```text
sync_endpoint_count=2
sync_success=true
opd_pipeline_enabled=true
rollout/native_trace_covered_generated_all=true
mtp/replay_trace_covered_all_targets=true
```

### Phase D: Generator Extension For Real Fanout

Implement and dry-run the SingleShot generator extension:

```text
student_replicas=4, teacher_replicas=2, student_node_group=default
student_replicas=6, teacher_replicas=4, student_node_group=default
student_replicas=6, teacher_replicas=4, student_node_group=nccl
trainer_nodes=4, trainer_node_group=default
trainer_nodes=4, trainer_node_group=nccl
trainer_nodes=8, trainer_node_group=default
trainer_nodes=8, trainer_node_group=nccl
```

Dry-run gates:

```text
all GPU pods have team=turbo
student, teacher, and trainer pods can select either default or nccl pools
selected node group is explicit in generated metadata or env
trainer role count defaults to 4 nodes for 1024/64 SingleShot smokes
student and teacher roles render unique Services and slot roles
student SMG lists all student URLs
teacher SMG lists all teacher URLs when native teacher routing is enabled
trainer control registers exactly the expected student endpoints
sync endpoint count equals student_replicas
no generated role reuses teacher-sglang-1 as both student and teacher
```

Render server-side first:

```bash
python "$GENERATOR" render-manifest --output "$MANIFEST" \
  --student-replicas 4 --teacher-replicas 2 \
  --trainer-nodes 4 \
  --student-node-group default \
  --teacher-node-group default \
  --trainer-node-group default
kubectl apply -n "$NS" --dry-run=server -f "$MANIFEST"
```

### Phase E: Pipeline Throughput Smoke

Run 5-10 steps each:

```text
P1/S1: prompts=4, chunk=1, prefetch=2, teacher_concurrency=2, students=2, teachers=3
P2/S2: prompts=8, chunk=1, prefetch=4, teacher_concurrency=2, students=4, teachers=2
P3/S2: prompts=8, chunk=2, prefetch=4, teacher_concurrency=2, students=4, teachers=2
P4/S3: prompts=16, chunk=2, prefetch=4, teacher_concurrency=4, students=6, teachers=4
```

Compare:

```text
mean step_total_s
mean forward_backward_roundtrip_s
mean sync_inference_weights_s
mean student_sampling_s
mean teacher_prefill_s
prepare_wait_s
SMG request deltas
sampler GPU utilization
teacher GPU utilization
repeat and trace metrics
```

### Phase F: Memory Boundary Revisit

Only after Phase E is stable:

```text
1024/96
1024/128
1536/64 with an explicit memory lever
8192/256 only with a validated memory plan
```

## Open Implementation Questions

1. Does the deployed SMG binary proxy raw SGLang `/generate` and
   `/teacher_hidden_cache`, or only OpenAI-compatible chat completions?
2. Should the SingleShot MTP reprogrammable path use a ported generator in this
   repo or continue to operate from `/home/apanda/xorl-apanda-dev-opd-port`?
3. Which live node-pool mix should the next rendered stack use for student,
   teacher, and trainer roles, given that both `default` and `nccl` are valid
   for all GPU worker classes?
4. Should teacher hidden-cache requests go through teacher SMG, or should
   teacher fanout be implemented in `run_opd_pipeline.py` with explicit endpoint
   selection?
5. Can sync cost be reduced by syncing every N chunks while preserving on-policy
   semantics, or must every optimizer step sync all samplers before any new
   rollout?

Until those are answered, the next productive action is the native SMG route
compatibility probe plus the `q36_singleshot_reprogrammable_slots.py`
generator extension for independent student and teacher replica counts.
