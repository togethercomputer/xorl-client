# Qwen3.6-35B-A3B training config guide — 65k seq len on H100

**What this is:** a decision guide for picking the highest-throughput *training* config for
Qwen3.6-35B-A3B (GDN-hybrid MoE) at long context. Built from a measured 16×H100 sweep at 65k
total seq len (64k ISL + 1k OSL), 2026-06-12, branch `apanda-dev-mtp` (post quack-fix #358 and
CP-compile-fix #366). Synthetic dummy data, balanced routing. Numbers are throughput-only — no
K3/correctness gate was run (see Caveats).

Model shape that drives every decision below: **40 layers = 30 GDN linear-attention + 10 full
attention** (`full_attention_interval=4`); full-attn has **16 query heads but only 2 KV heads**,
head_dim 256; hidden 2048; MoE 256 experts / top-8 / `moe_intermediate=512`; vocab 248320;
`tie_word_embeddings=false`.

---

## TL;DR — recommended config

**`ulysses=1, dp_shard=16, EP=8, triton MoE, mbs=1, recompute_full_layer, compile on`**
→ **91.2K tok/s (5.70K tok/s/GPU), 11.5 s/step, MFU 19.5%, 52.7 GB peak** on 16×H100.

Config file: `experiments/local_benchmark/q36_65k_sweep/configs/L09_u1dp16_triton_mbs1.yaml`.

The single most important finding: **at 65k on this model, do NOT use sequence parallelism.**
Pure data parallelism (FSDP shard the *batch*, not the sequence) is fastest because one 65k
sequence already fits in one rank's memory, and Ulysses only adds overhead without reducing the
per-rank attention FLOPs (see "Sequence parallelism" below). Scale `dp_shard` to your GPU count.

---

## Config decision guide

| knob | recommended @ 65k | when to change |
|---|---|---|
| `ulysses_parallel_size` | **1** | Raise only when one sequence no longer fits one rank (≳128–256k). Then prefer **2** (= KV-head count, no replication), composed with ring for the rest. |
| `data_parallel_shard_size` | **= world_size** (16) | This is your throughput lever — scale with GPU count. Each rank trains whole sequences independently. |
| `expert_parallel_size` | **8** (intranode) | EP must divide 256 experts and ideally fit within a node (NVLink). EP=8 keeps dispatch on-node; EP=16/32 spans nodes → slower all-to-all (cross-node EP all-to-all has been a large regression on other Qwen MoE configs — measure before raising it). |
| `moe_implementation` | **triton** | `quack` is correct post-#358 and ties triton within 1% at u1; pick quack only if a future kernel change makes it clearly faster. Avoid quack at ulysses>1 (jitter, below). |
| `ep_dispatch` | **deepep**, `deepep_num_sms: 48` | sms48 > sms24 by ~1–3% at this scale. Single-node jobs can't use deepep → fall back to `alltoall`. |
| `micro_batch_size` | **1** | mbs=1 already saturates the GPU at 65k (one 65k seq = 65537 tokens of work). mbs≥2 OOMs at u1. Larger mbs only helps at *small* ulysses-sharded seq lengths (it was a workaround for u8's per-rank starvation — irrelevant once you drop ulysses). |
| `gradient_checkpointing_method` | **recompute_full_layer** (mandatory) | At 65k, both `no_recompute` and `recompute_before_dispatch` **OOM at every topology** (incl. Ulysses — see re-audit). Full-layer recompute is the only mode that fits; lighter modes only become viable at shorter context. |
| `attn_implementation` | **flash_attention_3** | Always, for long context — FA3 keeps attention memory O(L), never materializes the L×L scores. `flex_attention` was 2.6× slower here and disagreed on loss; don't use it for throughput. |
| `optimizer` | **muon**, `momentum=0.0` | User directive (Muon for all new configs). No-momentum keeps optimizer state small; `muon_lr ≈ 10× lr`. AdamW is ~2% faster per-step but disallowed for new configs and uses more state. |
| `ce_mode` | **quack_linear** (was: compiled) | **+63% tok/s vs `compiled` at 65k** — see ce_mode ablation below. The 248k-vocab CE is ~⅓ of the step; `quack_linear` (chunked cuBLAS + CuTeDSL CE) is faster and −12 GB. `compiled` also runs near-OOM (61 GB → allocator churn). Avoid `fused_quack` (OPD selected-token path; OOMs on dense CE). |
| `enable_compile` | **true** | ~consistent win; excludes a ~10–15 step warmup. Requires the #366 fix to be present (else GDN-CP numerics break under compile — see Correctness fix). |

---

## The knobs that matter most, in depth

### Sequence parallelism (Ulysses) — leave it OFF at 65k

This is the counter-intuitive result, so here is the full reasoning.

**Throughput rises monotonically as ulysses shrinks** (matched 1.05M tokens/step):

| topology | mbs | tok/s | tok/s/GPU | MFU | peak |
|---|---:|---:|---:|---:|---:|
| **u1 dp16** (winner) | 1 | **91.2K** | **5.70K** | 19.5% | 52.7 GB |
| u2 dp8 | 2 | 82.9K | 5.18K | 17.7% | 52.7 GB |
| u8 dp2 | 8 | 81.8K | 5.11K | 17.5% | 52.8 GB |
| u8 dp2 | 1 | 60.0K | 3.75K | 12.8% | 24.8 GB |

**Why Ulysses can't win here (profiled, u8 vs u1, one step, rank0 GPU-kernel time):**

| category | u8 | u1 | Δ |
|---|---:|---:|---:|
| elementwise/**copy** | 3.50s | 2.46s | **+1.04s** |
| nccl sendrecv (a2a + conv halos) | 0.22s | 0 | +0.22s |
| nccl allgather/reduce-scatter (FSDP) | 1.42s | 1.60s | −0.18s |
| attention kernels | 2.66s | 2.71s | ~0 |
| GEMM | 2.82s | 2.84s | ~0 |
| GDN kernels | 0.66s | 0.66s | 0 |
| **total** | **13.79s** | **12.68s** | **+1.11s** |

- **Same FLOPs either way.** Ulysses shards attention *heads* (after an all-to-all); dp shards
  *sequences*. At matched tokens-in-flight each rank does the same L²×heads work — confirmed:
  attention kernel time is identical to within 2%. So there is no compute win to offset the
  overhead.
- **~95% of the overhead is data movement, not comms** (+1.04s of `copy_` vs +0.22s NCCL). The
  copies come from: (1) **GQA KV replication** — only 2 KV heads, so u8 repeats K/V 4× before the
  a2a (u16 would repeat 8×); (2) a2a pack/unpack `.contiguous()`; (3) per-layer GDN-CP boundary
  all-gather + 3 short-conv halo exchanges ×30 GDN layers ×fwd/recompute/bwd.
- The overhead is **flat in ulysses degree** (u2 ≈ u8 step time), i.e. a per-CP-activation tax,
  which is why u2 doesn't rescue it.

**When you WILL need Ulysses:** when a *single* sequence no longer fits one rank. Activation
memory is O(L) (~268 MB/layer boundary × 40 ≈ 10.7 GB at 65k); it scales linearly, so somewhere
around **128–256k** context mbs1/u1 OOMs and you must shard the sequence. At that point use
**ulysses=2** (matches the 2 KV heads → zero replication) composed with ring attention for
degree beyond 2, and pick up the copy-path fixes listed under "Open optimizations".

#### Re-audit (2026-06-13): is "DP > Ulysses" a benchmark artifact? No.

Prompted by the (very reasonable) prior that Ulysses usually beats DP, the whole comparison was
re-audited on the **post-#366 fixed kernel**. Three things were checked:

1. **Is the throughput metric biased against Ulysses?** No. `tokens_per_sec` reduces
   `_original_position_ids` (cloned full, never sp-sliced) over `dp_group`, which sums exactly
   one full copy per dp-group = the true global batch tokens. Verified arithmetically: u1 (90K ×
   11.5 s) and u8 (81.8K × 12.8 s) both process ~1.05M tokens/step — the gap is real wall-clock
   on identical work, not an accounting error. (FLOP counter reports *logical* FLOPs regardless of
   recompute, so recompute cost surfaces only in tok/s — the honest metric.)
2. **Did the #366 compile bug taint the original ranking?** No. The matched comparison
   (u8 mbs8, u1 mbs1) never tripped the trace — re-running on the fixed kernel reproduced
   **u1 90.2K vs u8 81.6K, 0 CompilationErrors** (identical to pre-fix). The bug only hit the
   mbs1/mbs4 u8 runs, which were never the comparison points.
3. **Was Ulysses given its real advantage — sharding activations to drop recompute?** This was
   the genuine gap in the first sweep (everything used `recompute_full_layer`). Tested directly:

   | run | topology | recompute | result |
   |---|---|---|---|
   | M1 | u1 dp16 mbs1 | full | **90.2K tok/s** ✓ |
   | M2 | u8 dp2 mbs8 | full | 81.6K tok/s ✓ |
   | M3 / M3b / M3c | u8 dp2 mbs8 / 4 / 2 | **none** | **OOM** (69 GB) at every batch |
   | M7 | u8 dp2 mbs8 | before_dispatch | **OOM** (69 GB) |
   | (L11) | u1 dp16 mbs1 | before_dispatch | OOM |

   **`no_recompute` OOMs at every Ulysses degree.** `recompute_full_layer` is *mandatory* at 65k
   for DP and Ulysses alike — it was never a handicap imposed only on Ulysses. So Ulysses cannot
   deploy its classic "shard activations → skip recompute" win here, for two structural reasons:
   - **Matched-throughput memory is topology-invariant** (peak ≈ 52.7 GB at u1mbs1, u2mbs2,
     u8mbs8). Ulysses shards the MLP/MoE activations by `cp`, but holding tokens-in-flight
     constant forces `mbs` up by the same factor, which re-multiplies them. Net per-rank
     activation is unchanged.
   - **The post-all-to-all attention activation is NOT sequence-sharded** — after the a2a each
     rank holds the full 65k sequence (for its head subset), so the dominant 65k activation term
     is identical to DP's.

**Why your historical experience still holds — Ulysses wins in a different regime.** Ulysses beats
DP when (a) attention is a large fraction of the model — **dense / all-full-attention** models,
not a 30/40-GDN hybrid where only 10 layers are quadratic; (b) there are **many KV heads**, so no
4–8× GQA replication tax; and crucially (c) the alternative to Ulysses is *not* "fit the whole
sequence with DP + recompute" but **"can't fit at all"** — i.e. the context is long enough that
even mbs1 DP OOMs, so sequence-sharding is the only way to run and recompute can't rescue DP. On
this GDN-hybrid MoE at 65k, none of those hold: attention is a minority of layers, KV heads = 2,
and one sequence fits comfortably — so DP wins. Cross the ~128–256k line and (c) flips, which is
exactly where Ulysses becomes mandatory here too.

**Would packing into one long sequence + ulysses=16 be faster? No.** Document-masked packing
keeps per-doc attention FLOPs identical to dp16, so there's still no FLOP win — and u16 makes the
overhead strictly *worse*: KV replicated 8×, the CP group now spans both nodes (IB instead of
NVLink for every per-layer halo/all-gather), and only 1 query head per rank kills FA3 occupancy.
Packing also reclaims nothing here (fixed 65k docs, one per rank, zero padding).

### MoE implementation — triton, but quack is no longer disqualified

Post-#358 quack produces finite loss everywhere and **ties triton within 1%** on the winning
topology (89.7K vs 91.2K at u1). At ulysses=8 it showed 2.3–4.6 s/step jitter (28–58K) — partly
the now-fixed compile warmup (#366), partly chunked-path/comm interaction. Triton is uniformly
steady, so it stays the default; revisit quack only if a kernel change makes it clearly faster.

### Micro-batch & recompute — both pinned by the 65k memory budget

At 65k, one sequence (65537 tokens) already fills the GPU's compute, and mbs≥2 OOMs at u1.
`recompute_before_dispatch` keeps the pre-MoE-dispatch activations alive and **OOMs even at
mbs1**; `recompute_full_layer` (keep only the layer-boundary tensor, recompute the rest in
backward) is the only method that fits. Both knobs only open up at shorter context / sharded
sequences — at 8k, mbs=10 and lighter recompute were the levers (see the 8k sweep doc).

---

## ce_mode ablation (2026-06-13): quack_linear is a +63% win

After merging `apanda-dev` (for #367's fused CE etc.), ablated `ce_mode` and `moe_implementation`
on 1 node (8×H100, u1 dp8 mbs1, 65537 pack — relative kernel comparison; absolute tok/s lower than
16-GPU since params shard only 8-way). The cross-entropy over **248320 vocab × 65k tokens** turned
out to be the dominant cost — and the default `compiled` mode is the bottleneck:

| ce_mode | moe | tok/s (dp8) | peak | notes |
|---|---|---:|---:|---|
| compiled (default) | triton | 28.5K (noisy 25–30) | 61 GB | torch.compile auto-chunker; near-OOM → allocator churn → 17–20 s/step jitter |
| **quack_linear** | triton | **46.4K (+63%)** | **49 GB** | chunked cuBLAS + CuTeDSL CE; rock-steady 11.28 s/step; loss == compiled ✓ |
| compiled | quack | 28.9K | 61 GB | **MoE kernel makes ~no difference — the step is CE-bound, not MoE-GEMM-bound** |
| quack_linear | triton, **mbs2** | 28.9K | 68.6 GB | mbs2 fits but regresses (near-OOM, super-linear 37 s/step) — the freed memory doesn't convert to throughput via batch |
| fused_quack | triton | **OOM (now FIXED, PR #369)** | — | #367 `fused_selected_logprob_ce` is a *chunked dense-CE drop-in* (NOT OPD-only). Root cause of the OOM (traceback → `causallm_loss.py:538`): `causallm_loss_function` had no `fused_quack` branch, so it fell through to the eager `hidden@weight.t()` full-logits path (60.6 GB). The chunked path was only wired into the other entry point (`compute_per_token_ce`). PR #369 adds the branch + regression tests. |

**Takeaways:** (1) switch `ce_mode: compiled → quack_linear` for a large win at long-context/large-vocab;
(2) the MoE implementation (quack vs triton) is *irrelevant* at 65k because CE dominates — quack is
healthy post-#358/#366 but not a throughput lever here; (3) `fused_quack` is the wrong tool for dense CE.
**16-GPU confirmation pending** (cluster contention) — but the CE cost is per-token and independent of
FSDP degree, so the relative win is expected to hold; absolute numbers TBD.

## Why 65k fits on one rank at all (no SP needed)

Long context costs compute O(L²) but memory only O(L) — and the O(L) terms are small once
everything else is sharded or chunked:

- **Model state**: `cp_fsdp_mode=all` shards params/grads/opt-state across the full 16-rank mesh
  in every topology. ~35B params bf16 → ~4.4 GB/rank, similar for grads; Muon at momentum=0 has
  near-zero optimizer-state footprint.
- **Activations are O(L)**, and `recompute_full_layer` keeps just one boundary tensor/layer:
  65537×2048×2 B ≈ 268 MB × 40 ≈ **10.7 GB**, plus one layer's recompute working set in backward.
- **The two classic memory bombs never materialize**: FlashAttention-3 tiles the 65k×65k scores
  (attention memory O(L), not O(L²)); `ce_mode=compiled` chunks the lm_head so the
  65537×248320 fp32 logit tensor (**~65 GB**) is never built.

Sum ≈ the measured 52.7 GB. The "long context ⇒ needs sequence parallelism" intuition is a
pre-FlashAttention / inference-KV-cache reflex; it doesn't apply to flash-attention training
until the linear activation term itself stops fitting.

---

## Measured sweep (full evidence)

Steady-state mean of steps 3–9, 10-step runs, 16×H100, 65537 packing, balanced routing.

| run | topology | moe | mbs | tok/s | tok/s/GPU | MFU | peak | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| **L09** | **u1 dp16** | **triton** | 1 | **91.2K** | **5.70K** | 19.5% | 52.7 GB | **winner** |
| L13 | u1 dp16 | quack | 1 | 89.7K | 5.61K | 19.2% | 52.7 GB | quack ties, −1% |
| L12 | u2 dp8 | triton | 2 | 82.9K | 5.18K | 17.7% | 52.7 GB | |
| L08 | u8 dp2 | triton | 8 | 81.8K | 5.11K | 17.5% | 52.8 GB | |
| L06 | u8 dp2 | triton | 4 | 74.9K | 4.68K | 16.0% | 32.0 GB | |
| L02 | u8 dp2 | triton | 1 | 60.0K | 3.75K | 12.8% | 24.8 GB | |
| L01 | u8 dp2 | quack | 1 | 40.8K (jittery) | 2.55K | 8.7% | 24.8 GB | finite loss; warmup jitter |
| L11 | u1 dp16, rec-before-dispatch | triton | 1 | **OOM** | — | — | — | |

`results/q36_65k_sweep/sweep_results.json`; raw logs under
`/shared/opd-control/q36-65k-bench/rank0/logs/`.

---

## Correctness fix landed during this sweep (#366, upstream)

The `'int' object has no attribute 'to'` CompilationError seen in the u8 logs was **a silent
correctness bug**, not just noise. `merge_fwd_bwd_kernel`
(`ops/linear_attention/ops/cp/chunk_delta_h.py`) called `.to(tl.int32)` on a `do_not_specialize`
scalar; torch.compile's `identify_mutated_tensors` traces it as a Python int → CompilationError
→ Dynamo "assume every input mutated" fallback, which **on torch 2.10 drops the kernel's writes**
(verified by a minimal eager-vs-compiled store-parity repro). Result: the GDN context-parallel
boundary-state correction wrote zeros, so any compiled GDN-CP run on a shape that tripped the
trace ran with **truncated recurrence at CP boundaries** — wrong numerics, no crash, loss still
descends.

Fix: `tl.cast(pre_or_post_num_ranks, tl.int32)` (commit `0a988d64` on `apanda-dev-mtp`; merged
upstream as **PR #366 / `82ece3a7`** on `apanda-dev`). Validation: 56→0 warnings; compiled
step-1 loss now matches eager to 4 decimals (12.2861 vs 12.2859; the broken path gave 12.1652);
steady-state throughput unchanged (the merge kernel is cheap — impact was numerics-only).

**Implications:** the L09 winner (u1, zero warnings) never hit this path and is unaffected. Any
*historical* compiled GDN-CP run whose logs show `identify_mutated_tensors` / `'int' object has
no attribute 'to'` warnings is numerics-suspect; ulysses=1 runs are clean. Worth adding a
compiled-vs-eager step-1 parity smoke to the CP test suite.

---

## Open optimizations (would shrink the Ulysses gap if SP is needed at longer context)

- Avoid materializing the GQA KV replication — kv-head-aware a2a, or all-gather KV instead of
  replicate-then-a2a when `kv_heads < ulysses`.
- Fuse the 3 short-conv halo exchanges per GDN layer into one collective.
- Overlap the GDN boundary all-gather with interior compute.

---

## Caveats

- **Throughput only — no K3/correctness gate run.** Gate before promoting any of these as a
  production recipe.
- **ISL/OSL:** the benchmark computes loss over all 65537 positions. A real 64k-ISL/1k-OSL job
  masks labels to the last ~1k tokens, which only slightly changes CE *backward* cost (forward
  logits are still computed for all positions); the trunk fwd+bwd — the dominant cost at 65k — is
  identical, so the reported tok/s holds within a few percent, counting all 65k positions.
- **Synthetic balanced routing.** `XORL_MOE_SYNTHETIC_ROUTING=balanced` is required for dummy
  data (else identical sequences collapse the router onto a few experts → EP-rank OOM). Real
  data's routing imbalance can shift MoE/EP cost.

---

## Reproduction / infra

- Stack: 2 bare pods `q36-65k-bench-rank{0,1}`, manifest
  `experiments/local_benchmark/q36_65k_sweep/pods.yaml`. Gotchas learned: pods need
  **`runtimeClassName: nvidia`** (else `Failed to open libnvidia-ml.so.1` → NCCL init fails), and
  must **pre-set `nodeSelector: node-group: nccl`** (a mutating webhook injects `node-group=default`
  → affinity pend on nccl-group nodes). hostNetwork+hostIPC, `rdma/infiniband: 1` + IPC_LOCK,
  `team: turbo`, slot-agent loop polling `/shared/opd-control/q36-65k-bench/<rank>/run.sh`.
- Launch: `experiments/local_benchmark/q36_65k_sweep/launch_65k.py launch <name> <abs-config>`
  (master = h100-092 hostname :29871, hostNetwork).
- Pods deleted after the sweep (16 GPUs returned to the pool).
- Related: 8k-seqlen / 32-GPU recipe in `qwen36_35b_32gpu_throughput_sweep_20260612.md`.
