# FP8 base + bf16 LoRA for fresh_ab ES — implementation handoff

> **STATUS: ABANDONED (2026-06-13).** Built, tested, and validated end-to-end on
> the live 35B pool — the SR fold is numerically correct (`update_norm` matched
> bf16, cold probe 0.7031) and the OOM/timeout were fixed. **But FP8 is a net
> throughput *regression* for this workload** (~3.3× slower/step), so the effort
> was dropped and the GPUs relinquished. Root cause: ZORL teacher-forced scoring
> is short-sequence and **not** MoE-GEMM-bound, so FP8 doesn't speed up scoring
> (the design's core premise), while the FP8 fold makes the apply ~20× worse. See
> the §8.0 VERDICT for the numbers and the (marginal) best-case. The code stands
> if a long-sequence-scoring workload ever wants it — but A/B-confirm FP8 *scoring*
> is faster for the target workload **before** spending GPUs again. The pool
> (`zorl-ar-sglang-n`) and parity jobs are torn down. Branch
> `apanda-dev-fp8-fresh-ab` is left uncommitted in the worktree.

Implements `FP8_BASE_FRESH_AB_DESIGN.md`. Code lives in the **SGLang fork**, not
the xorl repo. Built 2026-06-13.

- **Branch:** `apanda-dev-fp8-fresh-ab` (off `apanda-dev-fold-gpu-gdn`, the
  fresh_ab + gpu_direct-fold superset stack).
- **Worktree:** `/home/apanda/xorl-sglang-fp8-fresh-ab`.

## What changed

| File | Change |
|------|--------|
| `python/sglang/srt/lora/fp8_fold.py` | **NEW.** `fp8_fold_()` — dequant → add Δ → per-block rescale → requant, with E4M3-correct stochastic rounding (`_stochastic_round_e4m3`, exact via `searchsorted` over the 127 positive E4M3 codes) and an error-feedback variant. **Memory-bounded:** chunks the fold over experts / 128-row blocks (`_fp8_fold_dense_` + `fp8_fold_budget_units`) so the fp32 transient stays under `XORL_ZORL_FP8_FOLD_CHUNK_MB` instead of allocating ~16× the fp8 weight whole. |
| `python/sglang/srt/lora/layers.py` | **Component A.** `_prepare_moe_routing` detects a block-FP8 base (`_base_moe_fp8_blockscale`) and selects an FP8-tuned config; `_base_gate_up_gemm`/`_base_down_gemm` pass `use_fp8_w8a8=True` + `block_shape` + the block weight scale (`A_scale=None` → the triton kernel quantizes activations inline). Output buffer stays bf16, so the LoRA delta accumulates pre-requant. |
| `python/sglang/srt/lora/lora_manager.py` | **Components B/C.** `_fold_moe_module_from_tensors` / `_fold_standard_module_from_tensors` route to `fp8_fold_` when the base is `float8_e4m3fn` and `XORL_ZORL_FP8_FOLD != off`. The MoE fold loops over **expert chunks**, building each chunk's `B@A` delta on the fly (never the whole-tensor fp32 delta). Mode getter `_zorl_fp8_fold_mode`, RNG `_zorl_fp8_fold_generator`, EF buffer `_zorl_fp8_ef_residual`. |
| `test/registered/lora/test_fp8_fold.py` | **NEW** (CPU, 20 tests): SR unbiasedness across magnitudes incl. subnormals; sub-ULP accumulation under SR/EF vs. signal loss under naive; multi-block + partial-edge-block; rescale-on-growth; EF tighter than SR; ue8m0/dtype/mode guards; manager-fold integration; reproducibility; **chunked-fold invariance + budget helper + per-expert-chunk MoE accumulation**. |
| `test/registered/lora/test_fp8_moe_gemm_parity.py` | **NEW** (GPU, 2 tests): FP8 base GEMM vs bf16 GEMM of the same weights — gate_up rel err <3%, down <5%, per-token output correlation >0.99; bf16 base stays on the non-FP8 path. |

## Env flags (all default-off → bf16 path byte-identical)

```
XORL_ZORL_FP8_FOLD = off | stochastic | stochastic_ef     # default off
XORL_ZORL_FP8_FOLD_SEED = <int>                            # SR reproducibility (default 0)
XORL_ZORL_FP8_FOLD_CHUNK_MB = <int>                        # fold transient peak budget, MiB (default 512, floor 64)
```

- `stochastic` — SR, no extra buffer. **The recommended default for FP8.**
- `stochastic_ef` — deterministic requant + error-feedback. Carries a residual
  buffer per target weight (currently **bf16**; the design's int8 residual is a
  future memory optimization — bf16 is correct but ~64 GB at 35B, so EF at full
  scale needs the int8 buffer or the residual won't fit). Enable only if the
  parity gate shows SR plateauing below bf16.
- `XORL_ZORL_FP8_FOLD_CHUNK_MB` — caps the fold's transient float32 working set.
  The fold dequant→add→requant of a multi-GB MoE weight stack would otherwise
  allocate ~16× the fp8 weight at once (measured: a 192 MB w13 → 10.9 GB
  transient) and OOM next to the KV pool. The fold is chunked over experts /
  128-row blocks to keep peak under this budget (192 MB w13 → 171 MB transient,
  a 64× reduction; the MoE delta `B@A` is also built per chunk, never whole).
  Default 512 MiB; lower it on a tight pool.
- Composes with `XORL_ZORL_FRESH_AB_FOLD=gpu_direct` (the per-chunk fold the
  35B runs use). Both fold paths (`buffer` + `gpu_direct`) route through the same
  two `_fold_*_from_tensors` helpers, so both are covered.

Only meaningful when the served base is **block-wise FP8 with plain (non-ue8m0)
scales** — i.e. serve via the **triton** FP8 fused-MoE path, not deep_gemm/UE8M0.
Both Component A and the fold fail loud (`RuntimeError`) if they see
`weight_scale_inv.format_ue8m0 == True`.

## Validated locally

```
# CPU (SR numerics + chunking + manager integration + bf16 regression):
PYTHONPATH=<wt>/python pytest test/registered/lora/test_fp8_fold.py \
    test/registered/lora/test_zorl_fresh_ab.py -q      # 51 passed

# GPU (FP8 GEMM parity), on an idle device:
CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=<wt>/python \
    pytest test/registered/lora/test_fp8_moe_gemm_parity.py -q   # 2 passed
```

Memory-bounding measured directly on GPU: folding a 192 MB fp8 w13 stack peaks at
**10.9 GB** transient whole-tensor vs **171 MB** at the default 512 MiB budget (64×).

Key empirical confirmation of the design's central claim (from the SR tests): a
1e-4 per-entry delta (~0.14 ULP at the expert-block scale) folded 400× accumulates
to the full `0.04` signal under SR (rel-err 0.1%) but only `~0.001` (2.4%) under
naive round-to-nearest — i.e. naive destroys ~97% of the ES signal, SR preserves it.

## §7 parity gate — the remaining (operator) validation

This is the mandatory at-scale check before trusting any FP8 result; it is a
multi-hour cluster run, not done here.

1. **Build the SGLang fork** from `apanda-dev-fp8-fresh-ab` into the serving image.
2. **Serve the FP8 base** `/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8`
   via the SGLang pool, forcing the **triton** FP8 fused-MoE path (so scales stay
   non-ue8m0), with the virtual-expert LoRA-MoE path enabled, and set on the
   server process:
   ```
   XORL_ZORL_FP8_FOLD=stochastic
   XORL_ZORL_FRESH_AB_FOLD=gpu_direct
   ```
3. **Run the parity recipe** `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE` (seed-matched to
   the bf16 reference) via the standalone client
   (`experiments/zorl/standalone/zorl_client.py --perturbation-mode fresh_ab`),
   pointing `--model` at the FP8 checkpoint.
4. **Gate (compare curves, not just peak):** held-out probe ≥ 0.88 by step ~300,
   plateau ~0.90, matching the bf16 trajectory within probe noise (±2.6 pts).
   Secondary: scoring logprob pair-delta Spearman ρ ≥ 0.95 vs bf16; `update_norm`
   within ~5% of bf16; no slow divergence past step 300 (the signature of biased
   rounding).
5. **If SR fails** → set `XORL_ZORL_FP8_FOLD=stochastic_ef` (needs the residual
   buffer; see flag note) and re-run. If EF also fails → ship Component A (FP8
   scoring) alone with a bf16 fold target, losing the memory win.

Expected scoring win (design §3): step ~52 s → ~30 s (~2× on the base MoE GEMM,
which dominates scoring). The fold cost is unchanged.

---

## §8.0 Live status (2026-06-13, this session)

The §8.1 OOM blocker is **fixed and validated end-to-end on the live pool**. The
parity gate is now RUNNING (`MULTOPSD-EGGROLL-35B-FP8-PARITY` on pool N).

First ES step on the FP8 base + SR fold:
- cold probe `reward_mean=0.7031` (90/128) — **FP8 scoring (Component A) correct**.
- `step 1: update_norm=17.12 pair_delta_std=0.0678 used_pairs=128 t_score=50.4s t_apply=130.9s`
- **`update_norm=17.12` matches the bf16 reference ~17** — the SR fold reproduces
  the bf16 update magnitude (secondary gate criterion (b) met on step 1).
- The apply **completes** (no OOM, under the 300 s client timeout).

Two fixes were needed beyond the original code:
1. **OOM** in `/apply_zorl_rewards`: the fold built the whole MoE fp32 delta +
   whole-tensor dequant (~16× the fp8 weight). Fixed by chunking the fold over
   experts + building `B@A` per chunk (`XORL_ZORL_FP8_FOLD_CHUNK_MB`).
2. **>310 s apply timeout**: a too-small chunk budget (128 MiB → ~128 Python
   expert-chunks/projection × 40 layers × ~20 passes) starved throughput. Fixed
   by raising the budget to **4096 MiB** (cl≈64 → 4 chunks/projection), fitting
   the ~14 GB free at mem-fraction 0.78. `t_apply` 300 s+ → 131 s.

**VERDICT (2026-06-13) — NOT A THROUGHPUT WIN for this workload; gate torn down,
GPUs relinquished.** Steady-state over 4 live steps:

```
FP8  step 2: t_score=48.4s  t_apply=130.7s
FP8  step 3: t_score=41.3s  t_apply=130.4s   → ~172 s/step
FP8  step 4: t_score=42.6s  t_apply=130.3s
bf16 reference (doc):  ~40 s score + single-digit apply ≈ 52 s/step
```

The design's premise fails on two counts:
1. **Scoring is not faster.** FP8 `t_score≈42 s` ≈ bf16 `~40 s`. The design assumed
   scoring is dominated by the base MoE GEMM (§3: "~40 s of a ~52 s step"), so FP8
   would halve it. But the teacher-forced SFT scoring uses **short trimmed
   sequences** (~16 answer positions × batch 32), so the MoE GEMM is *not* the
   bottleneck — dispatch / LoRA-SGMV / sampling overhead is. Speeding up the GEMM
   doesn't move `t_score`.
2. **The apply is ~20× worse.** FP8 fold = dequant → SR-requant, done once per
   fold-pass (~20 passes, pairs chunked by noise memory) × 40 layers ≈ 130 s;
   bf16 fold = `add_`, single-digit seconds.

Net **~3.3× slower** (172 vs 52 s/step). Even the best-case apply optimization
(accumulate the bf16 delta per module across passes, then dequant→requant **once**
per module per step — restructure `_apply_zorl_rewards_fresh_ab` to module-outer /
pass-inner, ~40×2 requants/step instead of ~20×40×2) only recovers the apply to
~5 s → ~47 s/step ≈ bf16's 52 s: a marginal ~10%, **not** the promised 2×, because
the dominant cost (scoring) isn't GEMM-bound and FP8 doesn't help it.

**The fold numerics are correct** — `update_norm=17.12` matched bf16 ~17 and the
cold probe was 0.7031 — so the SR fold (Components A/B/C) is sound and the OOM /
timeout were fixed. FP8 is simply not worth it *for this short-sequence ZORL
scoring workload*. It might help a **long-sequence-scoring** workload where the
MoE GEMM actually dominates — but only revive after an A/B that first confirms
FP8 *scoring* is materially faster for the target workload. Do not re-run the
convergence gate on this premise.

## §8 Operations / infra — how to run this on the cluster (READ FIRST)

The parity gate is already provisioned and running — you are inheriting live state,
not starting from scratch. Below is the current state, then the cluster gotchas that
will bite you if you spin up or restart pools.

### 8.1 Current state (as left for you)

- **Pool `zorl-ar-sglang-n`** — 8 replicas, TP=2, serving the FP8 checkpoint
  (`--model-path Qwen/Qwen3.6-35B-A3B-FP8`) **from this worktree's code**
  (`workingDir`/`PYTHONPATH` = `/workspace/home/xorl-sglang-fp8-fresh-ab`), with
  `XORL_ZORL_FP8_FOLD=stochastic` + `XORL_ZORL_FRESH_AB_FOLD=gpu_direct` +
  `XORL_ZORL_NOISE_DEVICE=gpu`. Manifest:
  `experiments/zorl/k8s/qwen3_6-35b-a3b-zorl-sglang-n-tp2-shard.yaml`.
- **Job `…-fp8-parity-<id>`** is RUNNING on pool N — the parity gate, candidate
  `autoresearch/candidates/MULTOPSD-EGGROLL-35B-FP8-PARITY.yaml` (= the bf16
  FRESH-RESAMPLE recipe: fresh_ab, resample, 128 pairs, batch 256, **seed 9246** to
  curve-match the bf16 reference). Watch it; if it diverges you own the fix.
- The bf16 reference to compare against is the historical `FRESH-RESAMPLE` curve in
  W&B project `zorl` (held-out ~0.88 by step 300, plateau ~0.90, peak 0.93). A live
  bf16 `HERO` run (pool M) is unrelated — don't touch it.
- The FP8 branch is **uncommitted** in the worktree; pods exec live from the worktree
  PVC, so this is fine for validation. **A code fix = edit in the worktree + restart
  the pool pods** (`kubectl delete pods -l app=zorl-ar-sglang-n`; OnDelete pool, see
  8.3) — no image rebuild, no commit needed. Commit only when the gate passes.

- **KNOWN BLOCKER (first launch, 2026-06-13) — FIXED IN CODE (2026-06-13):** the
  gate's first run got its cold probe fine (0.7031 — so FP8 scoring works) then
  **OOM'd in the first `/apply_zorl_rewards`**: `CUDA out of memory … 791 MiB
  free … 73.49 GiB allocated`. Root cause was the fold materializing the whole
  MoE delta (`B@A`, fp32, ~4× the fp8 weight) **and** a whole-tensor fp32
  dequant — peak ~16× the fp8 weight (measured: a 192 MB w13 → **10.9 GB**
  transient), with no headroom left after pool N's `--mem-fraction-static 0.85`
  grew the KV pool to fill the (smaller) FP8 base.
  **Fix shipped:** the fold is now **chunked over experts / 128-row blocks**, and
  the MoE delta is built per chunk (never whole). Peak transient is bounded by
  `XORL_ZORL_FP8_FOLD_CHUNK_MB` (default 512 MiB; same 192 MB w13 → **171 MB**, a
  64× reduction). To relaunch: pull the worktree fix onto pool N
  (`kubectl delete pods -l app=zorl-ar-sglang-n`), optionally also drop
  `--mem-fraction-static` to ~0.70 / lower `XORL_ZORL_FP8_FOLD_CHUNK_MB` for extra
  headroom, then relaunch `MULTOPSD-EGGROLL-35B-FP8-PARITY`. The earlier
  `…-fp8-parity-…` job is `Failed` (pre-fix); relaunch fresh. Watch that the first
  `/apply` now completes and `update_norm` matches the bf16 ~17.

- **KNOWN BLOCKER #2 (2026-06-13, OPEN) — the OOM fix traded memory for LATENCY.**
  With the chunked fold in place the OOM is gone, but the next launch
  (`…-fp8-parity-q9pwb`) got its cold probe (0.7031) then the first
  `/apply_zorl_rewards` **timed out**: client log `TimeoutError: 8 (of 8) futures
  unfinished` — the apply exceeded the client's 300 s `_post_all` timeout on all 8
  replicas. The per-block dequant→add→**E4M3-searchsorted SR**→requant loop over a
  35B MoE base's thousands of 128×128 blocks is too slow at the small (memory-safe)
  chunk size. This is the open blocker. Fix directions (your call, you own the fold
  perf/memory tradeoff): **(a)** vectorize the SR requant — `_stochastic_round_e4m3`'s
  `searchsorted` over 127 codes should run batched over all blocks of a tensor at
  once, not per-block in Python; **(b)** raise `XORL_ZORL_FP8_FOLD_CHUNK_MB` (fewer,
  larger chunks → fewer Python iterations) up to whatever headroom a lowered
  `--mem-fraction-static` (~0.6–0.7) allows — there's a perf↔memory knee between
  BLOCKER #1 (OOM) and #2 (timeout) to find; **(c)** raise the client apply timeout
  (`apply_zorl_rewards_all` / `_post_all` `timeout=300.0` in
  `standalone/zorl_client.py`) as a stopgap so you can at least measure per-step
  fold latency and confirm convergence while (a)/(b) land. Measure the actual fold
  wall-time first (server logs the apply duration) to know how far (a)/(b) must close
  the gap. Pool N is up (8/8); the `…-q9pwb` job is `Failed`.

### 8.2 The worktree + venv gotcha (this WILL trip you)

The Python venv lives ONLY in the main repo: `/workspace/home/xorl-sglang-internal/.venv`.
Worktrees do **not** have their own `.venv`. So in any pool manifest:
- `SGLANG_PYTHON` must stay `/workspace/home/xorl-sglang-internal/.venv/bin/python`.
- Only `workingDir` and `PYTHONPATH` point at the worktree
  (`/workspace/home/xorl-sglang-fp8-fresh-ab[/python]`).
- The `git -C … rev-parse HEAD` echo should point at the main repo (a worktree's
  `.git` is a file, not a dir, and resolves oddly in-pod — harmless echo, but don't
  let it confuse you). Symptom of getting this wrong: pods CrashLoop with
  `.venv/bin/python: No such file or directory`.

### 8.3 Cluster gotchas (each one cost real time this session)

1. **StatefulSets MUST use `updateStrategy: OnDelete`.** With the default
   `RollingUpdate`, a `kubectl apply` of any template change (even an env var)
   **rolls every replica under the live run and kills it.** With OnDelete, applies are
   inert until you explicitly `kubectl delete pod`. All current pools are OnDelete;
   keep it that way. If you must `kubectl patch` it on a live SS, the JSON form is
   `[{"op":"replace","path":"/spec/updateStrategy","value":{"type":"OnDelete"}}]`.
2. **Bad nodes.** `research-common-h100-073` and `research-common-h100-064` reliably
   crash sglang at startup (SIGQUIT / "No accelerator available"). New pools catch
   them. Exclude via `nodeAffinity → requiredDuringScheduling → NotIn` (pool N's
   manifest already excludes 073/105; add 064 and any new offender as a sibling list
   item under `values:` — watch the YAML indent, it's a common mis-edit). A pod that
   crashlooped onto a bad node won't move on its own; `kubectl delete pod` it to
   reschedule (it may land on the same bad node — repeat, or exclude the node).
3. **The client is `backoffLimit: 0` and owner-routed across all replicas**, so it
   **fails terminally if even one replica isn't reachable at startup** ("did not
   become reachable in time"). Never point the client at a pool until it is fully
   `N/N` ready. Use a health-settle loop that also kicks crashloopers before launch:
   ```bash
   until [ "$(kubectl get pods --no-headers | grep zorl-ar-sglang-n | grep -c 1/1)" -eq 8 ]; do
     for p in $(kubectl get pods --no-headers -o wide | grep zorl-ar-sglang-n \
                  | grep -E "CrashLoop|h100-073|h100-064" | awk '{print $1}'); do
       kubectl delete pod "$p" --wait=false; done
     sleep 30
   done
   ```
   If the pool is stuck short on capacity (pods `Pending`, no node), you can instead
   **descope the client to the healthy replicas**: set the candidate's `INFER_URL` to
   only the `1/1` pods and `NUM_SHARDS: auto` adapts (`pairs_per_shard × n_healthy`).
4. **Shared 400-GPU cluster, gang-scheduled.** Other tenants come and go; a pool that
   fit a minute ago may go `Pending`. Don't over-provision: a tear-down-then-create
   can lose the freed GPU to another tenant before your pool gangs in. Check headroom
   first: `kubectl get pods -A -o json | <sum nvidia.com/gpu over Running+Pending>`.
5. **`kubectl logs -f` silently stops streaming after ~1–2 h.** Poll (`--tail=N`) and
   check pod phase to detect real termination; don't trust stream EOF.

### 8.4 Launch / monitor / iterate

- **Launch** (when pool is N/N): from `experiments/zorl/autoresearch`,
  `python3 controller.py launch --candidate candidates/MULTOPSD-EGGROLL-35B-FP8-PARITY.yaml`
  (`--dry-run` to inspect the rendered job first). One client per pool — it holds a
  server-side ZORL session.
- **Reading the gate** from `kubectl logs job/<…-fp8-parity-…>`:
  - **Throughput**: the `t_score=…s t_apply=…s` on each step line. Compare `t_score`
    to the bf16 ~40 s; the win is in scoring, not the fold.
  - **Convergence**: `probe_reward=` every PROBE_INTERVAL. Gate ≥0.88 by step ~300,
    plateau ~0.90.
  - **Biased-rounding signature**: `update_norm` drifting >5% from the bf16 ~17, or a
    held-out that climbs then **slowly declines past step 300**. If you see it, the SR
    grid/rescale is suspect (recall the max-E4M3-code=126 clamp class of bug) →
    switch to `stochastic_ef` and/or re-examine `fp8_fold.py`.
- **Iterate a fix**: edit in the worktree → `kubectl delete pods -l app=zorl-ar-sglang-n`
  (recreates on the fixed code) → wait N/N → relaunch the candidate. No rebuild.
- **Pristine base for a clean re-run**: the fresh_ab fold mutates the served base
  in place, so a re-run needs fresh pods — `kubectl delete pods -l app=zorl-ar-sglang-n`
  resets the base; the client's step-0 cold probe re-measures the true baseline anyway.
- **Don't** route through any SMG (no LoRA affinity); `candidate_routing: owner` is
  correct and already set. **Don't** `git stash -u` in the worktree. **Don't** touch
  pools/jobs you didn't create (e.g. the `hero` run on pool M, or `fresh-*` refs).
