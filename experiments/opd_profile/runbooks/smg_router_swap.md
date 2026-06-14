# SMG router swap-in (Run B routing layer)

Replace the Python `dispatch` proxy with SMG (Shepherd Model Gateway, Rust) in
front of N sglang shards. Validated on the OPD CoT precompute pipeline on
2026-05-28 — got 3× per-GPU throughput at 4 shards and unlocked
linear-ish scaling beyond 8 shards (dispatch hit a hard wall at ~12 shards
where adding shards actually *reduced* total throughput).

## Why swap

Dispatch saturates at ~280–460 tok/s/GPU and gets WORSE per-GPU as you add
shards (negative scaling). SMG with `--policy cache_aware` consistently hits
1300–1500 tok/s/GPU at the same sglang topology because it routes
prefix-similar requests to the same shard, maximizing KV-cache hits. For the
Run B 8192-prompt CoT generation, the shared system message + chat template
prefix is exactly the workload `cache_aware` is built for.

| Run | Router | Shards × TP | Per-GPU |
|---|---|---|---|
| v1 | dispatch | 12×2 | 80 tok/s/GPU |
| v2 | dispatch | 8×2 | 277 |
| v3 | dispatch | 4×2 | 462 |
| v4 | dispatch | 8×2 | 282 |
| **v5** | **SMG cache_aware, conc=512** | **3×2** | **~1466** |

## Binary

```
/home/apanda/smg-together-thunderagent-port/target/debug/smg
```

Pre-built debug binary, version 1.4.1. Mounted via the home PVC so it's visible
from every apanda-namespace pod that mounts `/home/apanda`. Debug build is
fine — routing isn't CPU-bound; the bottleneck is on the sglang backend.

## Minimal launch

```bash
"${SMG_BIN}" launch \
  --host 0.0.0.0 --port 8080 \
  --policy cache_aware \
  --worker-urls http://run-sglang-0:30060 http://run-sglang-1:30060 ...
```

OpenAI-compatible chat completions at `/v1/chat/completions`. Prometheus
metrics at `:29000/metrics`.

## K8s pod spec (drop-in dispatch replacement)

The full SMG_TEMPLATE_HEAD is in
`experiments/opd_profile/k8s/generate_cot_fanout.py` (the `--router smg`
branch). The key container spec:

```yaml
containers:
- name: smg
  image: nvcr.io/nvidia/pytorch:26.02-py3
  env:
  - name: HOME
    value: /home/apanda
  ports:
  - containerPort: 8080
  volumeMounts:
  - name: home
    mountPath: /home/apanda
  command:
  - /bin/bash
  - -lc
  - |
    set -euo pipefail
    [ -f "${HOME}/.shell_env" ] && source "${HOME}/.shell_env"
    SMG_BIN=/home/apanda/smg-together-thunderagent-port/target/debug/smg
    WORKER_URLS="http://run-sglang-0:30060 http://run-sglang-1:30060 ..."  # MUST be quoted
    exec "${SMG_BIN}" launch \
      --host 0.0.0.0 --port 8080 \
      --policy cache_aware \
      --worker-urls ${WORKER_URLS}
```

Service name and port can match the existing `dispatch` Service (8080) → the
client doesn't need to change.

To regenerate a manifest:
```
python3 experiments/opd_profile/k8s/generate_cot_fanout.py \
  --run-name <run-name> \
  --sglang-nodes <node1>,<node2>,... \
  --dispatch-node <router-node> \
  --router smg \
  --prompts <prompts.json> \
  --cot-output <output.json> \
  --max-tokens 8192 \
  --client-concurrency 512 \
  --output <manifest.yaml>
```

## Critical client-side fix: wait for /v1/models, not /health

SMG's `/health` always returns OK even when zero workers are registered. If
your client only checks `/health` and starts sending traffic, it'll burn 100s
to 1000s of requests as immediate 404s
(`error_type=no_workers, model_not_found`) before sglang finishes loading.

**Symptom:** client log shows `avg tokens=0` for many hundreds of prompts;
SMG metrics show `smg_router_upstream_responses_total{status_code="404",error_code="model_not_found"}`
in the thousands.

**Fix:** wait until SMG has discovered the model. The fanout generator's
CLIENT_TEMPLATE already does this:

```bash
for i in $(seq 1 600); do
  if ! curl -m 2 -fsS http://run-dispatch:8080/health >/dev/null 2>&1; then
    sleep 5; continue
  fi
  # SMG must have discovered at least one worker serving the model
  if curl -m 2 -fsS http://run-dispatch:8080/v1/models 2>/dev/null \
       | grep -q 'Qwen3.6-35B-A3B'; then
    break
  fi
  sleep 5
done
```

## Concurrency knob — measured saturation point

Direct sweep on **3 sglang TP=2 shards (6 GPUs)**, Qwen3.6-35B-A3B, 4-digit
multiplication prompts (avg 2700 tokens out):

| conc | per-GPU tok/s | GPU util | busiest shard load |
|---|---|---|---|
| 192 | 665 | — | 102 running, 0 wait (client-conc bottleneck) |
| 512 | 1466 | 62% | 197 running, 0 wait |
| **1024** | **2270** | **62%** | **437 running, 181 wait — sweet spot** |
| 1536 | 1790–1963 | 63% | regression, queue piles up |

**~2270 tok/s/GPU is the wall** at this topology. Adding more concurrency past
1024 doesn't help — GPU util stays stuck at ~62% because:

1. **TP=2 NCCL all-reduce overhead** is the dominant idle source. Each decode
   step requires an inter-GPU reduce; ~30–50% of each rank's wall time is
   comm-bound. Visible as GPU 1, 3, 5 sitting at ~46% util while their pair
   GPUs 0, 2, 4 hit 85–89%.
2. **Cache_aware imbalance** caps the busiest shard. One shard always runs
   ~50% more requests than the others (KV cache stickiness is the feature).
   Past 1024, the imbalance turns into queueing.

To push past 2270 tok/s/GPU you need **more shards**, not more concurrency.
The throughput is per-shard bounded.

**Concurrency rule of thumb (revised):** `client_concurrency ≈ 340 × N_shards`
for TP=2 sglang at max_running_requests=256. That keeps busiest shard at
~437/256 (~70% overload absorbed into the wait queue) which is the
empirical sweet spot. Above 340/shard, queueing latency dominates.

## Diagnostics

```bash
# per-shard load
kubectl -n apanda exec <RUN>-sglang-<i> -- curl -s http://localhost:30060/get_load

# SMG model registration
kubectl -n apanda exec <RUN>-dispatch -- curl -s http://localhost:8080/v1/models

# SMG throughput (deltas)
kubectl -n apanda exec <RUN>-dispatch -- curl -s http://localhost:29000/metrics \
  | grep 'smg_router_upstream_responses_total.*status_code="200"'
# Take two snapshots 60s apart, multiply delta by avg-tokens, divide by num_gpus.

# GPU util on a sglang node
kubectl -n apanda exec <RUN>-sglang-0 -- nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used --format=csv
```

## Gotchas

1. **`WORKER_URLS` must be quoted in bash.** Unquoted multi-URL strings get
   split by IFS and the second URL becomes a command. The first version of
   my generator missed this — caught it because smg started routing to only
   the first worker. Generator fix is in place.

1b. **`--policy cache_aware` is the WRONG default for big/slow models.** With
   long per-request latency (Q3-235B-A22B at TP=8 = ~30-90 s/request), the
   cache_aware feedback loop puts ~100% of new requests onto whichever shard
   first built up a prefix cache. One shard goes to 479+ running while the
   other idles at 30. Use `--policy power_of_two` for slow models (load-aware
   with cache hint) or `round_robin` when balance trumps cache.
   - Diagnostic: `kubectl exec <RUN>-sglang-{N} -- curl :30060/get_load` and
     compare `num_reqs`. If one shard is >2× another, change policy.
   - Q3.6-35B-A3B at TP=2 was OK with cache_aware because per-request latency
     is short enough that the imbalance equilibrates. For per-request >30s,
     the imbalance compounds catastrophically.

2. **GPU pair conflicts.** The fanout generator hardcodes CUDA pairs
   `0,1` `2,3` `4,5` `6,7`. If the node has another tenant's GPU process on
   any of those pairs, that sglang pod will hit `UnexpectedAdmissionError`.
   Delete it and proceed — SMG routes to the remaining healthy backends.

3. **Cache_aware imbalance is the feature.** Don't switch to `round_robin`
   or `random` to "fix" the load imbalance. The imbalance is what gives the
   3× speedup. The price is sometimes one shard gets bigger batches and
   slightly higher per-token latency, but total throughput is higher.

4. **Apanda admission webhook injects nodeSelector=default.** The fanout
   generator auto-detects node-group via `_node_group()` and sets the right
   selector. Don't try to use nccl-tainted nodes (014, 040, 059, 071, 073,
   087, 092) — kubelet will reject with NodeAffinity error.

## Related

- Generator: `experiments/opd_profile/k8s/generate_cot_fanout.py` (`--router smg|dispatch`)
- Memory note: `project_run_b_cot_mt8192` (now lists v1–v5 disjoint CoT pool)
- Upstream SMG: https://github.com/togethercomputer/together-smg
