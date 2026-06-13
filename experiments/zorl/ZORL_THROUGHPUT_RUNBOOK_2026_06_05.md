# ZORL throughput runbook

Date: 2026-06-05
Scope: ZORL standalone (SGLang-native LoRA, no xorl trainer) step time.
Detail docs (read for the full reasoning):
- `ZORL_APPLY_GPU_OPTIMIZATION_2026_06_05.md` (apply phase)
- `ZORL_SCORE_THROUGHPUT_FINDINGS_2026_06_05.md` (score phase)
- `ZORL_PERFORMANCE_HANDOFF_RUNBOOK_2026_06_05.md` (the original handoff)

The live run (`zorl-ar-zorl-wordle-014-zwmq5`, population 1024) was kept running
untouched throughout this investigation. All code edits take effect only on the
next process launch (they don't disturb running pods).

---

## 0. Where the time goes (ZORL-WORDLE-014, per step)

```
t_score ~= 234 s   (~69%)   teacher-forced /generate under candidate LoRAs
t_apply ~= 105-130 s (~31%)  /apply_zorl_rewards (antithetic ES update)
```

Both phases were profiled. Neither is compute-bound; both were throttled by
overheads/limits that can be removed.

---

## 1. Tools (build/benchmark first — do not tune the production loop blind)

| tool | what it isolates | needs cluster? |
|---|---|---|
| `experiments/zorl/standalone/bench_zorl_apply_math.py` | apply noise/update math (correctness + perf) | no (1 GPU on dev pod) |
| `experiments/zorl/standalone/bench_zorl_sglang.py` | gen / score / apply against a live SGL pool | yes |
| `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-bench-sglang.yaml` | 1-replica TP=2 benchmark pool, all throttle flags parameterized | yes (2 GPUs) |

Deploy the benchmark pool (separate from production, `team: turbo`):

```bash
kubectl apply -f experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-bench-sglang.yaml
# wait for readiness, then drive it by pod IP (the dev pod reaches SGL pod IPs directly):
IP=$(kubectl get pods -n apanda -l app=zorl-bench-sglang \
       --field-selector=status.phase=Running -o jsonpath='{.items[0].status.podIP}')
curl -s -o /dev/null -w '%{http_code}\n' http://$IP:30000/health_generate   # 200 = ready
# flip a throttle flag and restart in place:
kubectl set env deploy/zorl-bench-sglang -n apanda LAUNCH_BLOCKING=1 DISABLE_OVERLAP=1
kubectl rollout restart deploy/zorl-bench-sglang -n apanda
# when done, free the GPUs:
kubectl delete deploy zorl-bench-sglang -n apanda
```

Per-replica score workload that matches production exactly:
`--num-pairs 32 --num-shards 1 --train-size 128` (64 candidates × 128 examples =
8192 seqs / 1.95M input tokens).

---

## 2. APPLY phase — FIXED (GPU batched-flat noise)

### Diagnosis
Apply is ~88% irreducible CPU `torch.randn`: each of up to 512 pairs regenerates
~240 per-tensor noise tensors (rank-4 LoRA-B = 39.5M floats/pair → ~20B Gaussian
floats/step). Metadata rebuild / reassembly are negligible. Restructuring the CPU
path is bit-exact but useless; `.add_(noise, alpha=)` *inside* the randn loop is
even ~3× slower (OpenMP thread-launch overhead).

### Fix (implemented, flag-gated, default off)
`xorl-sglang-internal/.../lora/lora_manager.py`: generate the antithetic noise on
the **GPU** with one batched `randn` per pair. Because `_zorl_normalized_b_noises`
is shared by candidate materialization (score-time) and apply, a single switch
keeps both sides consistent.

- Env: `XORL_ZORL_NOISE_DEVICE` — `cpu` (default, unchanged) or `gpu`.
- Additive methods: `_zorl_noise_device`, `_zorl_b/a_noise_layout`,
  `_zorl_normalized_b/a_noises_gpu`, `_zorl_build_update_gpu`; device branches in
  `_build_zorl_candidate_adapter` + `apply_zorl_rewards`. CPU path byte-identical.
- k8s: commented `XORL_ZORL_NOISE_DEVICE: gpu` block in
  `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml`.

### Validation status
- Offline gates **PASS** (`bench_zorl_apply_math.py`, 1×H100): CPU path
  byte-identical (sha256 unchanged); GPU apply==materialize bit-exact; determinism;
  antithetic; **426.9× on the core** (CPU 17.2s → GPU 0.040s at 128 pairs).
- Fork unit tests added (`test/registered/lora/test_zorl_lora_manager.py`),
  11/11 pass.

### STILL NEEDS (before enabling in prod)
**Cluster integration gate** on an idle 16-replica TP=2 pool (non-production
window):
1. Set `XORL_ZORL_NOISE_DEVICE=gpu` on the SGL pods; restart.
2. `bench_zorl_sglang.py apply` — confirm `apply_wall_s` drops ~105-130s → single
   digits, `used_pairs=512`, `dropped_pairs=0`, all replicas agree on
   `update_norm`/`pair_delta_*`, `restart_delta=0`.
3. Two full steps + cross-replica parent-checksum agreement (export from 2
   replicas, diff). Only then flip the default.

Run it:
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.zorl.standalone.bench_zorl_apply_math \
  --num-pairs 128 --layers 48 --repeat 1 --skip-profile        # offline gate
python -m experiments.zorl.standalone.bench_zorl_sglang apply \
  --control-urls "$DIRECT_SGL_URLS" --session-id bench-$(whoami) \
  --parent-lora <exports/best> --num-pairs 512 --num-shards 16 \
  --noise-device-hint gpu --out apply_bench.jsonl              # cluster gate
```

---

## 3. SCORE phase — FIXED (kernel OOB closed 2026-06-05)

Score is **not** throughput-bound (~8-20k tok/s/replica for a 3B-active MoE on
2×H100 that should do far more). Three throttles + one hard kernel bug — the
kernel bug (§3d, the 4× blocker) is now **fixed**.

### 3a. FIXED/recommended: fewer distinct LoRAs + bigger single-LoRA batch (~2×, validated)
`score_max_workers_per_owner` (= **C**, distinct LoRAs co-batched per replica) and
`--teacher-forced-batch-size` (= **S**, examples/request) are the packing knobs.

| config | LoRAs/prefill | tok/s/replica |
|---|---|---|
| C=4, S=8 (production) | 4 | 9 912 |
| C=2, S=16 | 2 | 10 570 |
| C=1, S=32 | 1 | 13 957 (64 cand) / 19 961 (16 cand) |

All 0-restart / 0-5xx. **Recommended scoring config (post-§3d-fix): `per_owner=1`,
`batch=64`** — S=64 used to crash the scheduler (§3d) and is now safe.

Post-fix re-measurement on a fresh `zorl-bench-sglang` pool (TP=2, FAST flags:
LAUNCH_BLOCKING=0, overlap on; `per_owner=1`, 16 candidates × train_size=128,
seq_len=238, 2048 seqs/config), 2026-06-05:

| config | seqs/prefill (≈C×S) | tok/s/replica | 5xx | restarts |
|---|---|---|---|---|
| S=8 (old production packing) | 8 | 7 636 | 0 | 0 |
| S=32 (previous safe ceiling) | 32 | 15 254 | 0 | 0 |
| **S=64 (was a hard crash pre-fix)** | 64 | **17 151** | 0 | 0 |

S=64 is the fastest stable config (2.25× over S=8); S=32→S=64 is a modest +12%
(prefill is near-saturated by S=32), so the bulk of the score win is still the
S=8→S=32 jump — but S=64 is now available *and* crash-free, which is what lets you
also drop the §3b serialization guards. Apply via the autoresearch candidate /
client args; validate two production steps with `used_pairs=512, dropped_pairs=0`,
no SGL restarts, before promoting.

### 3b. FIXED/recommended: drop the serialization flags
Production SGL launches with throttles added defensively against §3d:
- `CUDA_LAUNCH_BLOCKING=1` — serializes **every** CUDA kernel. Drop it.
- `--disable-overlap-schedule` — no CPU/GPU overlap. Drop it.
- `--disable-cuda-graph` — minor for prefill-heavy; keep for now.
- `SGLANG_DEBUG_SHARED_OUTER=1` — **no-op** (unreferenced in fork). Drop it.

Keep batches under the §3d crash line so these guards aren't needed.

### 3c. IMPLEMENTED (flag, default off, needs cluster validation): logprob trim
Teacher-forced scoring sent `logprob_start_len=0` (logprobs over the whole ~238-
token prompt) but the client uses only the last ~60 target-token logprobs.
`zorl_client.py` now has `--teacher-forced-logprob-trim` (+ `-margin`): per-seq
`logprob_start_len = prompt_len - target_tokens - margin`. The prompt is still
fully prefilled, so target logprob *values* are unchanged — only the prompt's
`[positions, vocab≈150k]` logit rows are skipped. Same scores, less score-time
logit memory/compute.
- Validate before prod: on an idle pool, score one candidate with trim on/off and
  confirm identical rewards (`bench_zorl_sglang.py score --logprob-start-len ...`
  was added to test this; the dedicated correctness check loads a parent and
  compares last-N logprobs at start_len 0 vs trimmed).

### 3d. FIXED — the ~64-seq/prefill Triton CUDA OOB (the 4× blocker)
S=32 (32 seqs/prefill) was rock-stable; **S=64 (~15.2k tokens) reliably crashed**
the scheduler, independent of concurrency, mem-fraction, and prefill-size.

**Root cause (found by code inspection + a bit-exact GPU repro, 2026-06-05):** a
grid_z batching bug in the chunked-SGMV *expand* kernel launcher
(`triton_ops/chunked_sgmv_expand.py`). CUDA grid Z (segments) is capped at 65535,
so for `num_dispatched > 65535` the launch loops with `seg_offset > 0`. In that
loop `seg_indptr`/`weight_indices` are per-SEGMENT arrays and are correctly sliced
by `seg_offset`, but `input_map` (the MoE shared-outer dispatch remap) is a
per-ROW array indexed inside the kernel by the **global** logical row
`s_offset_logical` — exactly like the un-sliced `permutation`. The launcher
nonetheless passed `input_map[seg_offset:]`, double-counting `seg_offset` against
an already-global index → out-of-bounds read on the second grid_z batch.

The cliff lands precisely between S=32 and S=64:
`num_dispatched = seqs × ~238 tok × topk(8)` → S=32 = 60 928 (< 65535, one batch,
fine) vs S=64 = 121 856 (> 65535, two batches → OOB). That matches the observed
"independent of concurrency / mem-fraction / prefill-size" signature (it depends
only on `num_dispatched`). The reported crash site (`_base_down_gemm ->
fused_moe_kernel`) is the *next* kernel launch after the async OOB in the
preceding shared-outer gate_up **expand**, exactly as suspected. Related:
`chunked_sgmv_oob` history (a different, earlier OOB in the same path).

**Fix (landed in the fork working tree):** pass `input_map` **un-sliced** in
`chunked_sgmv_lora_expand_forward` (consistent with `permutation`); only the
per-segment arrays stay sliced. One-line change; for `seg_offset==0`
(num_dispatched ≤ 65535, i.e. all previously-validated configs incl. S=32) it is
byte-identical, so no regression risk to the stable path.

**compute-sanitizer notes (read before re-trying memcheck):**
- On the **full SGLang server** `compute-sanitizer --tool memcheck
  --target-processes all` floods `cudaErrorNoKernelImageForDevice (error 209)` on
  `cudaFuncGetAttributes` (even with `CUDA_MODULE_LOADING=EAGER`) — the
  flashinfer/many-kernel SM90 context never resolves a device image under the
  sanitizer, so it can't reach/observe the OOB. Don't waste time memchecking the
  whole server here.
- On the **minimal standalone repro** (`/tmp/test_gridz_oob.py`, single process,
  the lone Triton chunked-SGMV kernel, sgl_kernel stubbed) memcheck works cleanly
  and pins it exactly. On the buggy code it reports, for the >65535 case:
  ```
  Invalid __global__ read of size 4 bytes
    at _chunked_lora_expand_kernel+0x5b0 in chunked_sgmv_expand.py:118
    Access ... is out of bounds, 85 bytes after the nearest allocation of size 2097152 bytes
  ```
  Line 118 is the `tl.load(input_map + s_offset_logical, ...)` read — i.e. the
  `input_map` over-read from the `seg_offset` double-count, on the fixed code this
  report is gone. Lesson: memcheck the isolated kernel, not the full server.

**Validation (independently reproduced):**
- Bit-exact GPU repro (`/tmp/test_gridz_oob.py`, 1×H100): compares the input_map
  path vs the permutation path on pre-gathered rows (must be identical).
  - BUGGY (`input_map[seg_offset:]`): num_dispatched 60 928 → bit-exact;
    70 000 → **silent corruption** max|Δ|=1.43 (no crash, wrong LoRA output!);
    121 856 (=S=64) → `Triton Error [CUDA]: an illegal memory access`.
  - FIXED (`input_map` un-sliced): all three bit-exact (max|Δ|=0).
  The silent-corruption band (65 535 < num_dispatched ≲ crash) is the important
  part: before the fix, ANY prefill over the cliff that didn't fault returned
  wrong scores. Production (S=8 → 60 928 dispatched) sat just under it.
- Durable regression test added:
  `test/registered/lora/test_chunked_sgmv_backend.py::TestChunkedSGMV::test_expand_input_map_crosses_grid_z_boundary`.
- Cluster (incidental): a `zorl-cs-sglang` TP=2 pod that launched *after* the fix
  hit disk served a 64-seq / 15 232-token prefill → `POST /generate 200 OK`,
  0 restarts (it tripped the 20 s detokenizer health-check once during first-time
  Triton compile of the larger grid_z path, then recovered).

**S=64 is now unlocked** (num_dispatched=121 856 < `_MAX_NUM_DISPATCHED`=131 072, so
the pre-allocated buffers already cover it). To push past ~S=68
(num_dispatched > 131 072) still needs the grow-on-demand follow-up below.

**Follow-up (now unblocked, not yet done):** make
`lora/layers.py::_ensure_moe_lora_buffers` grow on demand instead of the fixed
`_MAX_NUM_DISPATCHED=131072` (=16384×topk) so `--chunked-prefill-size` /
`--max-running-requests` can be raised for concurrency beyond S=64. Must keep the
`_TRITON_OOB_PAD` over-read margin and stay safe under CUDA-graph capture (no
resize mid-capture).

### 3e. Concurrency ceiling (C-sweep) — bounded by fused-prefill GPU memory, not the kernel

With §3d fixed + grow-on-demand, the score-batch ceiling is no longer the kernel
OOB — it is **GPU activation memory for the *fused* prefill**, governed by
**C×S** (= seqs co-batched per forward, capped by `chunked_prefill_size`), not C
or S alone. Single-replica TP=2 sweep (chunked-prefill 32768, mem 0.60,
2026-06-06): every config with **C×S ≤ 64 was clean**; every config with **C×S ≥
128 crashed** (fused ≈30k tok / ~243k dispatched = the OOM edge). C=64 only
"passed" because it always ran on the freshly-restarted server after C=32 crashed
(sweep-order artifact).

Robustly-stable distinct-LoRAs/replica at S samples/LoRA (keep C×S ≤ ~64):
`S=4 → C≈16`, `S=8 → C≈8`, `S=16 → C≈4`. To handle *more* LoRAs at a given S:
lower `chunked_prefill_size` (less fusion/forward), raise mem-fraction, or add
replicas. Throughput-optimal stays `C=1` + large S (the per-distinct-LoRA
shared-outer overhead dominates), but `C=8/S=8` (C×S=64) is the sweet spot when
you need many concurrent distinct LoRAs.

### Promoted to production: ZORL-WORDLE-015 (2026-06-06)

Relaunched the live Wordle run with all of the above. Candidate
`ZORL-WORDLE-015-opsd-tf-sharded-smg-export-pop1024.yaml`: `per_owner=8` /
`batch=8` (`SCORE_MAX_WORKERS=128` so 16 owners × 8 are realized), `hinted_cot`
teacher-forced OPSD reward. The `zorl-ar-sglang` StatefulSet was restarted to pick
up the §3d fix + grow-on-demand (PYTHONPATH from the fork working tree) and had
the §3b serialization guards dropped (`CUDA_LAUNCH_BLOCKING` /
`SGLANG_DEBUG_SHARED_OUTER` / `--disable-overlap-schedule`). Step 1 (pop 1024):
`used_pairs=512 dropped_pairs=0`, **0 SGL restarts**, `t_score=169s` (vs ~234s at
the old C=4/S=8 + serialization), `t_apply=120s` (CPU noise — apply GPU-noise §2
not yet enabled). C=8/S=8 = 121 856 dispatched (>65 535) — the exact config
run-014's notes flagged as "8 concurrent restarted the workers"; now clean on the
fixed kernel.

Gotchas hit during the relaunch (for next time):
- `zorl-ar-smg` is a `backoffLimit:0 restartPolicy:Never` **Job**, not a
  Deployment — deleting its pod kills it permanently; recreate from
  `qwen3-coder-30b-a3b-zorl-smg-router.yaml`.
- Node **h100-105** failed NCCL `ncclCommInitRank` ("unhandled cuda error") under
  tenant contention; added a `nodeAffinity NotIn` exclusion on the SGL StatefulSet.
- Restarting the SGL pool *while a client is mid-startup* desyncs the ZORL session
  (a replica that wasn't warmed when `/start_zorl_session` broadcast → apply 400
  "Unknown ZORL session"). Bring the pool fully up *before* launching the client.

### Hazard: mem-fraction
This is a teacher-forced / `max_new_tokens=1` (prefill-dominated) workload — the KV
pool is barely used. A big `--mem-fraction-static` is wasteful AND dangerous:
0.85 left only ~8 GB working memory and OOM-crashed even small requests. Keep
`mem_fraction_static <= 0.6`; spend headroom on bigger prefills (after §3d is
fixed), not KV.

---

## 4. Quick status table

| item | phase | status | gate to promote |
|---|---|---|---|
| GPU batched-flat noise (`XORL_ZORL_NOISE_DEVICE=gpu`) | apply | implemented, offline-validated (426×) | cluster checksum agreement on 16-replica pool |
| fewer-LoRAs + bigger-batch (`per_owner=1–2`, `batch=32`) | score | validated (~2×) | 2 prod steps, used_pairs=512, no restarts |
| drop `CUDA_LAUNCH_BLOCKING` / `--disable-overlap-schedule` / `SGLANG_DEBUG_SHARED_OUTER` | score | identified | same as above |
| logprob trim (`--teacher-forced-logprob-trim`) | score | implemented (flag, off) | trim-on==off reward check on idle pool |
| **~64-seq MoE-LoRA Triton CUDA OOB** (input_map grid_z slice) | score | **FIXED** (bit-exact repro + regression test; S=64 200 OK in-cluster) | promote: drop the §3b serialization guards, raise score batch to S=64 |
| grow-on-demand `_MAX_NUM_DISPATCHED` | score | follow-up (now unblocked) | needed only for S>~68 (num_dispatched>131072) |

---

## 5. Verification / monitoring commands

```bash
# live run health
kubectl logs -n apanda zorl-ar-zorl-wordle-014-zwmq5-k9dxr --tail=3 | rg 'step '
# SGL restart counts (must not increase during a promoted benchmark/run)
kubectl get pods -n apanda -l app=zorl-ar-sglang \
  -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount} {end}'; echo
# scheduler crash signature to watch for
kubectl logs -n apanda <sgl-pod> --tail=80 | rg 'illegal memory|SIGQUIT|Completed'
```

## 6. Gotchas hit during this investigation
- Editing the fork / `zorl_client.py` is safe for the live run (loaded in memory;
  effective next launch only).
- `pkill -f 'bench_zorl_sglang'` can match the monitor's own `tail` and return
  exit 144 — kill by explicit PID instead.
- The benchmark pool got wedged/crash-looping after pushing past §3d repeatedly;
  delete + redeploy if it stops returning `health_generate=200`.
