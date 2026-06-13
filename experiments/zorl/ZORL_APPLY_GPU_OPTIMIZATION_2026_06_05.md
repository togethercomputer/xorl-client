# ZORL apply-phase optimization: GPU batched-flat noise

Date: 2026-06-05
Author handoff target: next ZORL perf agent
Companion to: `ZORL_PERFORMANCE_HANDOFF_RUNBOOK_2026_06_05.md`

## TL;DR

`/apply_zorl_rewards` (~105-130 s/step, ~31% of the ZORL-WORDLE-014 step wall)
is **bound by CPU `torch.randn`**, not by Python overhead or the work that was
the obvious target (per-pair metadata rebuild). The fix is to generate the
antithetic perturbation noise on the **GPU** with a single batched `randn` per
pair. It is implemented in the SGLang fork, **gated by an env var, default off**
(zero change to the current CPU behavior), and ready for a cluster integration
gate.

- Offline microbench: `experiments/zorl/standalone/bench_zorl_apply_math.py`
- Cluster harness: `experiments/zorl/standalone/bench_zorl_sglang.py`
- Fork change: `xorl-sglang-internal/python/sglang/srt/lora/lora_manager.py`

## Diagnosis (microbench, CPU)

The apply update is, per pair, `_zorl_normalized_b_noises(seed)` (≈240 per-tensor
`torch.randn`, rank-4 LoRA-B = 39.5M floats/pair) accumulated into the update.
At population 1024 that is ~20 billion Gaussian floats generated on the CPU per
replica per step.

Measured cost split (48 layers, real production shapes):

| component | share |
|---|---|
| per-tensor `randn` (irreducible on CPU) | ~88% |
| per-pair metadata rebuild | 0.026 s (negligible) |
| transpose/cat reassembly | 0.089 s (negligible) |

So the "obvious" optimization (hoist the metadata rebuild, accumulate raw,
reassemble once) is **bit-exact but useless** — it does not touch the dominant
cost. Confirmed in the bench: the restructured CPU path is no faster (and a
naive `.add_(noise, alpha=)` *inside* the randn loop is ~3x **slower** due to
OpenMP thread-launch overhead at high core counts — generate-all-then-accumulate
avoids it).

CPU floor ≈ 120 ms/pair ≈ 62 s for 512 pairs, regardless of restructuring.

## Why GPU, and why it is correct

CPU and CUDA RNG streams differ, and even on one device a single flat `randn`
does not equal a sequence of per-tensor `randn`. So GPU noise is **not**
bit-identical to the current CPU noise — it is a different (equally valid)
Gaussian draw. ZORL/ES only needs *some* random perturbation direction, so a
different draw is algorithmically fine, **provided** the noise a candidate was
materialized/scored with is exactly the noise the update regenerates.

That invariant holds for free here because **one function**
(`_zorl_normalized_b_noises` / its GPU twin) is shared by both candidate
materialization (`_build_zorl_candidate_adapter`, score time) and the apply
update. A single switch, `XORL_ZORL_NOISE_DEVICE`, drives both, so within a step
they always agree.

Cross-replica "identical parent" invariant: every replica/TP rank derives noise
purely from `(seed, shape)` with a deterministic CUDA generator on identical
hardware (all H100/sm90 + same image), so all replicas produce the same update.
The client's existing `_validate_apply_results_agree` (update_norm / pair_delta
within tol) is the runtime safety net.

GPU speed (H100, this exact workload):

| path | 512 pairs |
|---|---|
| CPU per-tensor (current) | ~62 s (noise floor) + weight update |
| GPU batched-flat (new) | **0.128 s** noise+accumulate (~480x) |

Per-tensor GPU is 10x slower than batched (kernel-launch bound) → the batched
flat scheme is what the implementation uses.

## What changed in the fork (additive, default off)

`lora_manager.py`, all behind `XORL_ZORL_NOISE_DEVICE` (default `cpu`):

- `_zorl_noise_device()` — resolve `cpu` (default) vs `cuda`.
- `_zorl_b_noise_layout` / `_zorl_a_noise_layout` — precompute the generation
  plan once (sorted raw stream order + transpose/cat reassembly plan).
- `_zorl_normalized_b_noises_gpu` / `_zorl_normalized_a_noises_gpu` — GPU twin of
  the noise functions: one batched `randn(total, seed)` per pair, sliced and
  reassembled.
- `_zorl_build_update_gpu` — apply accumulator: sum `score*weight_scale*randn`
  in raw layout (one add per pair), reassemble the normalized update once.
- `_build_zorl_candidate_adapter` and `apply_zorl_rewards` each get a
  `device == "cuda"` branch; the CPU branch is **byte-for-byte the original
  code** (regression-gated, see below).

The legacy CPU noise functions (`_zorl_normalized_b_noises`,
`_zorl_normalized_a_noises`) are **untouched**.

## Correctness gates

Offline (`bench_zorl_apply_math.py`, run on a dev pod with 1 H100):

1. **CPU zero-regression** — the bench extracts the real fork helpers via AST and
   hashes the resulting parent LoRA-B weights. The hash is identical before and
   after the change (`sha256 a490aaa1…` at 32 pairs/48 layers). The additive edits
   do not alter the CPU path.
2. **Restructured-CPU bit-exactness** — the layout-hoisted raw-accumulate variant
   is bit-exact (`max|Δ|=0`, all metrics agree) vs the golden CPU loop.
3. **GPU apply == materialize** — `_zorl_build_update_gpu` equals the per-pair
   accumulation of `_zorl_normalized_b_noises_gpu(seed_i)` (the materialization
   noise), bit-exact within GPU. This is the key invariant.
4. **GPU determinism** — `_zorl_normalized_b_noises_gpu(seed)` is identical across
   calls (cross-replica determinism proxy).
5. **Antithetic** — `+/-` directions are exact negatives about the parent.

Run it:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.zorl.standalone.bench_zorl_apply_math \
  --num-pairs 128 --layers 48 --repeat 1 --skip-profile
```

Confirmed output (1x H100, 128 pairs, 48 layers, 2026-06-05):

```text
=== correctness gate (b_only) ===
  max |Δ| parent LoRA-B weight (golden vs optimized): 0.000e+00
  bit-exact parent weights: True
  all metrics agree exactly: True
  CORRECTNESS GATE: PASS
=== timing ===
  golden    apply:   17.004s        # restructured CPU is no faster:
  optimized apply:   18.054s        #   apply is randn-bound, not overhead-bound
=== GPU fast-path validation ===
  (1) apply==materialize  max|Δ|=0.000e+00  bit-exact=True
  (2) determinism same-seed identical: True
  (3) antithetic +/- exact negation: True
  (4) CPU golden apply: 17.229s   GPU build+norm: 0.040s   speedup: 426.9x
  GPU VALIDATION: PASS
```

## Required integration gate before promoting (NOT yet run)

This cannot be validated offline; it needs the real 16-replica TP=2 pool in a
**non-production window** (do not collide with the live run):

1. Bring up an idle ZORL SGL pool (or wait for one) with
   `XORL_ZORL_NOISE_DEVICE=gpu` on the SGL pods (add to the env block in
   `experiments/zorl/k8s/qwen3-coder-30b-a3b-zorl-sglang-tp2-shard.yaml`).
2. `bench_zorl_sglang.py apply` — confirm:
   - `apply_wall_s` drops from ~105-130 s toward single digits.
   - `used_pairs=512`, `dropped_pairs=0`.
   - all replicas agree on `update_norm` / `pair_delta_*`
     (`_validate_apply_results_agree`).
   - `restart_delta=0`.
3. Two full ZORL steps (population 1024, train 128) with no SGL restarts and
   parent-checksum agreement across replicas (export from 2 replicas, diff).

```bash
python -m experiments.zorl.standalone.bench_zorl_sglang apply \
  --control-urls "$DIRECT_SGL_URLS" --session-id bench-$(whoami) \
  --parent-lora /path/to/exports/best --num-pairs 512 --num-shards 16 \
  --noise-device-hint gpu --out apply_bench.jsonl
```

Only after this gate passes should `XORL_ZORL_NOISE_DEVICE=gpu` be set on a
production launch.

## Notes / next levers

- Score (~235 s, ~69% of step) is the bigger fish but the harder one — it is
  `/generate`-throughput-bound under the LoRA Triton stability cap (4 in-flight
  candidate requests/replica). `bench_zorl_sglang.py score` is the tool to
  isolate which axis (unique LoRAs, pool size, prefill tokens, kernel shape
  diversity) is the real limit. Untouched here.
- Alternative apply lever (not taken): shard the 512-pair update across the 16
  replicas + NCCL all-reduce. Keeps the exact CPU noise (no RNG change) but needs
  a cross-replica process group the independent SGL servers don't currently have
  — more invasive than the GPU switch.
