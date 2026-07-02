# ZORL-Wordle (fresh_ab, fp32-master PS, FP8 scorers) — algorithm + infra

**What this is.** Beat GRPO's held-out Wordle solve-rate (~0.67–0.68 honest,
retries=0) using **Evolution Strategies (ZORL)** on a frozen base model, trained
through an **fp32-master parameter server** that folds a **`fresh_ab` (EGGROLL)**
update into the base weights — while the inference scorers keep serving a fast
**block-FP8** model. Base + think alone ≈ 0.00, so every point is learned.

Date: 2026-07-02. Status: **code-complete + CPU-validated; deploy pending.**

---

## 0. Repos / worktrees

| Piece | Path | Branch |
|---|---|---|
| **xorl** (client, `tasks/wordle.py`, manifests, docs) | `/home/apanda/xorl-apanda-dev-zorl-consolidated` | `zorl-ps-fp32-client` |
| **sglang fork** (PS + FP8-view numerical code + scorers) | `/home/apanda/xorl-sglang-zorl` | `zorl-ps-fp32` |

The k8s scorer pods import sglang from a **working-tree checkout**, so editing the
fork + restarting pods deploys. The FP8 serving replicas currently in the cluster
(`zorl-w35-sglang`) import from `/home/apanda/xorl-sglang-internal` — that tree
already contains the in-tree sparse-delta *receiver* (`weight_sync/sparse_delta.py`),
so replicas need **no** new code; only the **PS** must run the `zorl-ps-fp32` tree.

---

## 1. The algorithm — ZORL Evolution Strategies, `fresh_ab` mode

ZORL is black-box ES over the model's weights, realized cheaply as **seeded
low-rank perturbations** that the inference engine can regenerate on demand.

### 1.1 `fresh_ab` (EGGROLL) perturbations
- The learned state lives in the **fp32 base master** on the PS. The LoRA parent's
  `B` is **identically zero forever** (`parent_B ≡ 0`); the adapter is only a
  scratch space for the per-candidate perturbation.
- Each generation draws a **population of `N` antithetic pairs** (`num_pairs`).
  Candidate `i` is defined *entirely by a seed*: fresh seeded factors
  `A_i, B_i` (rank `r`) give a rank-`r` weight perturbation `ΔW_i = σ · B_i A_i`
  applied on top of the current base. Antithetic pairs share `A_i` and use `±B_i`.
- Because a candidate is `(parent, seed, σ, sign)`, **no adapter bytes are shipped** —
  the PS and every replica reconstruct the identical `ΔW_i` from the seed (seed
  authority is structural: `seed = f(base_seed, family, generation, pair_index)`).

### 1.2 The ES fold (reward → weight update)
1. Score every candidate on a shard of training puzzles → scalar reward `R_i`.
2. Form centered/normalized advantages `c_i` (reward-weighting; antithetic pairs
   give a low-variance gradient estimate).
3. Combine into a base-shaped update `G = Σ_i c_i · ΔW_i` (full-rank in aggregate —
   the population spans up to `N·r` directions, which is *why* the update must land
   in the base, not a rank-`r` adapter).
4. Precondition with **Muon** (Newton–Schulz orthogonalization; `muon_lr`) and
   accumulate **exactly in the fp32 master**: `master += muon_lr · NS(G)`.
5. The next generation perturbs around the new base. Base otherwise **frozen**
   (no backprop, no gradients — pure forward scoring).

**Why fp32 master (not bf16/fp8 base).** A per-step Muon step is ~1.6e-6 per weight,
~75× below the bf16 ULP and far below the fp8 grid, so adding it straight into a
bf16/fp8 base retains only ~2–12 % (a flat trainer / quality collapse). The fp32
master accumulates it ~100 %; the served low-precision base is a *downcast view* of
the master. This is the whole point of the parameter server.

### 1.3 Contrast with `b_only` (the path this replaces)
The previous Wordle run was `b_only`: fixed `A`, an evolving **rank-16 LoRA-`B`**,
base frozen, Muon on `B` only. It is capacity-capped at rank 16 (the GRPO-parity
confound) and never uses the base-merge. `fresh_ab` removes the rank cap (full-rank
accumulation into the base) — the change the PS was built for.

---

## 2. The task / objective — faithful GRPO-Wordle match

`experiments/zorl/standalone/tasks/wordle.py` (multi-turn, `is_multi_turn=True`):
- **Env**: standard Wordle; per turn the model emits `<guess>` (optionally after
  `<think>`), the env returns per-letter feedback; up to 6 turns.
- **Reward**: solve-rate shaped reward (`_wordle_shaped_reward` / retrieval reward),
  scored by `score_completion` / `rollout_completion`.
- **Faithfulness knobs** (match the GRPO recipe): large population (`num_pairs=64`),
  train shard per step (`train_size`), `invalid_retries=0` (honest gate — an invalid
  guess is *not* silently retried), held-out eval on a disjoint seed set
  (`--probe-interval`, `--eval-size`), train/eval split by excluded seed.
- Full justification + honest caveats: `experiments/zorl/ZORL_WORDLE_GRPO_FAITHFULNESS.md`.

---

## 3. The infrastructure

```
                    ┌─────────────────────────────────────────────┐
   driver           │  zorl_client.py  (--task wordle              │
 (1 pod, CPU-ish)   │   --perturbation-mode fresh_ab --ps-url …)   │
                    └───────┬─────────────────────────┬────────────┘
                            │ start_zorl_generation    │ score candidates
                            │ (lockstep, on PS+replicas)│ (via SMG round-robin)
                            ▼                           ▼
        ┌───────────────────────────┐      ┌──────────────────────────────┐
        │  fp32-master PS            │      │  SMG router (Rust)            │
        │  sglang, TP=2, 2×H100      │      │  /generate → 32 workers RR    │
        │  FP8 base + fp32 master    │      └───────────────┬──────────────┘
        │  XORL_ZORL_FP32_MASTER=1   │                      ▼
        │  XORL_ZORL_PS_SYNC_VIEW=fp8│      ┌──────────────────────────────┐
        │  runs the fresh_ab fold    │      │  FP8 scorer pool              │
        │                            │      │  zorl-w35-sglang ×32 (TP=2)   │
        │  ── FP8-view sparse diff ──┼─────▶│  block-FP8 + cuda-graph +     │
        │  /update_weights_from_     │ push │  virtual-experts; in-tree     │
        │   sparse_delta (per rank)  │      │  sparse-delta receiver        │
        └────────────────────────────┘      └──────────────────────────────┘
```

### 3.1 fp32-master PS (the trainer)
- One sglang server, **TP=2** (2×H100; the 70 GB model won't fit TP=1 alongside the
  band-staged fold), FP8 base, LoRA enabled. **Never serves user traffic** — it only
  holds the authoritative fp32 master and runs the fold.
- Env: `XORL_ZORL_FP32_MASTER=1`, `XORL_ZORL_PS_SYNC_VIEW=fp8`,
  `XORL_ZORL_FRESH_AB_FOLD=gpu_direct`, `XORL_ZORL_OPTIMIZER=muon`,
  `XORL_ZORL_PS_DELTA_DIR=<shared FS>` (both PS + replicas mount it).
- Joins the SAME session/generation as the replicas in lockstep (same seed →
  identical perturbations, no seed forwarding).
- Endpoints: `/start_zorl_session`, `/ps_apply_and_broadcast` (fold → build FP8-view
  diff → push → commit), `/ps_rebroadcast` (after an elitist rollback).

### 3.2 FP8 scorer pool (inference)
- `zorl-w35-sglang` StatefulSet, **32 replicas**, TP=2 each, block-FP8
  (`Qwen3.6-35B-A3B-FP8`, e4m3 [128,128]) + `--dtype bfloat16` compute.
- Serving throughput stack (measured ~10×): **cuda-graph** (needs
  `--disable-custom-all-reduce`, `--mem-fraction-static 0.85`), **FP8 + multi-LoRA**
  (triton quant-info threaded through the MoE GEMM), **virtual-experts**.
- Each replica reconstructs the whole candidate population from seeds → scoring can
  round-robin freely; the in-tree receiver `apply_sparse_delta_file` scatters the
  PS's sync into the served `weight` + `weight_scale_inv`.

### 3.3 SMG router
- Compiled Rust gateway in front of the 32 workers, `--policy round_robin`
  (`owner_via_smg` pinning trips the per-worker circuit breaker — use RR). LoRA
  load/unload go **direct to workers**, not through SMG.

### 3.4 Driver
- `experiments/zorl/standalone/zorl_client.py` — opens the session on PS + replicas,
  and per generation: plan candidates from seeds → score on the pool (SMG RR) →
  `POST /ps_apply_and_broadcast` → held-out probe every `--probe-interval`.

---

## 4. The FP8-VIEW sparse sync (the enabling contribution)

The PS accumulates in fp32 but the scorers serve block-FP8. Naïvely scattering a
bf16 delta into fp8 corrupts the per-block scales, and folding straight into the fp8
base random-walks it to collapse (`FP8_NATIVE_ADAPTER_ACCUM.md`). Fix: **diff in the
model's own FP8 view.** Design doc: `experiments/zorl/FP8_VIEW_SYNC_DESIGN.md`.

Per folded tensor, each step:
1. Fold `ΔW` into the exact fp32 master (unchanged).
2. **RTN-quantize the master onto the replicas' current block grid** (grid-preserving:
   bump a block's scale only on saturation) → `(new_w8, new_scale_inv)`. Because it's
   deterministic and grid-preserving, an *unmoved* block reproduces the base
   bit-for-bit → its diff is empty (sparse from step 1, no dense resync).
3. Diff `new_w8` (fp8 bytes) and `new_scale_inv` (fp32) vs the last-synced FP8 view →
   ship both, keyed by the **real resident names** (`…weight`, `…weight_scale_inv`;
   MoE fused `…gate_up_proj.*`/`…down_proj.*`). No fused-split → no block misalignment.
4. Replica overwrites exactly the fp8 bytes + block scales that moved →
   `served == quantize_fp8(master)` **bit-for-bit**, cross-replica identical.

No drift: the master is the exact accumulator; the fp8 is a *derived view* whose
rounding is bounded (≤ ½ ULP) and non-accumulating. Under TP=2 each PS rank folds
its own shard and writes a per-rank delta file → replica-rank-`r` reads `delta[r]`
(1:1, no gather).

**Code** (all on the `zorl-ps-fp32` sglang tree):
- `python/sglang/srt/lora/fp8_fold.py` — `quantize_block_fp8_rtn`,
  `quantize_block_fp8_to_grid`, `dequant_block_fp8`.
- `python/sglang/srt/lora/zorl_fp32_master.py` — `Fp32MasterStore.master_for_fp8`
  (dequant-init), `fold_into_master_fp8`, `diff_fp8_view_since_last_sync`,
  `commit_fp8_sync`; generic `sparse_tensor_diff`.
- `python/sglang/srt/lora/zorl_ps_sparse_sync.py` — packer now carries fp8 (uint8 view).
- `python/sglang/srt/lora/lora_manager.py` — `XORL_ZORL_PS_SYNC_VIEW=fp8` branches in
  fold / build (`_zorl_ps_build_sparse_sync_fp8`) / commit.
- Test: `test/srt/zorl/test_zorl_fp8_view_sync.py` — 5/5 incl. the real in-tree
  receiver; 32 zorl tests pass, no regression.
  Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=python
  /home/apanda/xorl-sglang-internal/.venv/bin/python -m pytest
  test/srt/zorl/test_zorl_fp8_view_sync.py`.

---

## 5. One ES step, end to end

```
driver: start_zorl_generation(seed=S, gen=g)  on PS + all replicas   [lockstep]
replicas: reconstruct the N-pair population from seeds (no bytes moved)
driver: score each candidate on a train shard (SMG round-robin) → rewards R_i
driver: POST /ps_apply_and_broadcast {gen, rewards, lr, replica_urls} → PS
   PS: fresh_ab Muon fold into the fp32 master (exact accumulation)
   PS: build the FP8-VIEW sparse diff (weight fp8 + weight_scale_inv), per TP rank
   PS: POST /update_weights_from_sparse_delta {delta_paths} → each replica
       replica: scatter into served fp8 base (TP-shard 1:1)  → all serve the new base
   PS: commit (advance the FP8-view baseline)
driver: every --probe-interval, held-out probe → solve-rate curve
```

---

## 6. Deploy checklist + live gate

1. Bring up the PS (TP=2, env above), same model-path/sha as the replicas.
2. Reuse the 32 `zorl-w35-sglang` FP8 replicas + SMG (no change).
3. `zorl_client.py --task wordle --perturbation-mode fresh_ab --ps-url <PS>
   --infer-url <32 replicas> --optimizer muon --num-pairs 64 --invalid-retries 0
   --probe-interval …`.
4. Tear down the b_only `zorl-wordle-ps-35b-trainer-vgw5b`. **Never** touch the live
   GRPO sampler `wordle-sci-opsd-sampler-b`; append status to
   `/shared/apanda/wordle-coord/messages.jsonl`.
5. **Live gate**: PS `update_norm > 0`; FP8-view diff `nnz > 0` (weights actually
   flip fp8 codes); cross-replica byte-identity after a sync; held-out solve-rate
   clears the cold floor and climbs toward GRPO's ~0.67–0.68.

Related: `PS_FP32_MASTER_DESIGN.md`, `PS_FP32_MASTER_DEPLOY_GUIDE.md`,
`FP8_VIEW_SYNC_DESIGN.md`, `FP8_NATIVE_ADAPTER_ACCUM.md`, `ZORL_WORDLE_GRPO_FAITHFULNESS.md`.
