# Marin-math ZORL ES arm — launch runbook

ES fallback/parallel science arm to the Wordle ES line, on the Marin #6279 math
task: dense Qwen3 ~9.7B (`delphi-1e22-p33m67-wc386k_lr1e5-sft`), RLVR-MATH-7500,
the exact GRPO-repro prompt/verifier/reward (`experiments/marin`). Prepared
2026-07-05 on branches `zorl-marin-es` (xorl-client + xorl-infra). Nothing here
is deployed; the wordle arm's live resources are untouched (all names/ports are
disjoint).

## What this arm is

| piece | file |
|---|---|
| task module | `experiments/zorl/standalone/tasks/marin_math.py` (`--task marin_math`) |
| driver | `experiments/zorl/standalone/run_wordle_zorl_xorl_ps.py` (task-generic; `--dry-run` added) |
| scorers (64x TP1) | xorl-infra `k8s/zorl/qwen3-9b-marin-zorl-sglang.yaml` |
| SMG router | xorl-infra `k8s/zorl/zorl-marin9b-smg-router.yaml` |
| PS + driver pod (4 GPU) | xorl-infra `k8s/zorl/qwen3-9b-marin-zorl-ps-trainer.yaml` |
| PS config | xorl-infra `configs/zorl/qwen3_9b_marin_zorl_ps_muon.yaml` |
| tests | `experiments/zorl/standalone/tests/` (18) + marin's own suite (93, untouched) |

Reward semantics are IDENTICAL to `train_marin_grpo.py` (imports marin's
`grade_reference_final_answer` + `shaped_reward`, lpw=1.0, max 3584, target 768,
truncated −2 / wrong −1 / correct cosine ramp in [0,1], `min_response_length=16`
zero-bonus clause). The harness `score_result` hook feeds SGLang's exact
`completion_tokens`/`finish_reason` into the reward — same inputs as the GRPO
driver used.

## Dense-vs-MoE simplifications taken (each verified in source)

* **Dropped scorer flags** (MoE/GDN-only in xorl-sglang-zorl):
  `--lora-use-virtual-experts`, `--lora-moe-format hybrid_shared`,
  `--experts-shared-outer-loras`, `--max-mamba-cache-size` (dense Qwen3 has no
  GDN — checkpoint `config.json` `layer_types` = 37x `full_attention`),
  `SGLANG_ZORL_MOE_POOL_SLOTS` (raises outside inline mode),
  `SGLANG_ZORL_RANK1_INLINE_GEN`.
* **Inline-gen eligibility finding**: rank-1 inline AND pool_gen are MoE-only
  (`zorl_rank1_inline_state.py` plans fail "non-MoE A target" for anything
  outside `.mlp.experts.`; `fill_pool_slot` writes `*_moe` pool buffers). Dense
  candidates use the **materialized** seed→adapter path
  (`_build_zorl_candidate_adapter`), which is module-generic. Validated live in
  the local smoke (below).
* **Dropped PS-config keys**: `moe_implementation`, `ep_dispatch`,
  `moe_hybrid_shared_lora`, `lora_export_format` (dense default = `peft`), EP
  sizing. **Added `merge_qkv: false`** — REQUIRED: xorl's dense attention keeps
  a fused `qkv_proj` at TP1 by default and LoRA target matching is by leaf
  name, so q/k/v LoRA would be *silently skipped* (only o_proj+MLP would
  attach).
* **Sync**: `sync_inference_method: p2p` with `--sync-quantization bf16`
  (driver default fp8 is overridden in the manifest). The fold-aware
  sparse-delta fast path is **fp8-only** (`handler.py`: "fold-aware sparse
  delta requires fp8 sync quantization") — inert for this bf16 fleet, so all
  `XORL_SPARSE_DELTA_*` envs are dropped. p2p on this exact dense model →
  single-GPU sglang fleet is the proven marin-GRPO path
  (`configs/marin/qwen3_9p7b_server_rl.yaml`).
* **Prime overlay: not applicable.** `scripts/zorl_fold_prime_overlay.py`
  (xorl-zorl-ps) reconciles `quantize(bf16 master)` vs a pre-quantized FP8
  checkpoint's bytes; with bf16 receivers booting the same bf16 checkpoint the
  PS loads there is nothing to reconcile, and its only consumer (fold-aware
  sparse delta) is disabled at bf16 anyway. If this arm is ever moved to an
  FP8-quantized serving fleet, build the overlay then:

  ```bash
  # (xorl-zorl-ps checkout; only for a future FP8-serving variant — NOT needed for this launch)
  XORL_P2P_FP8_QUANTIZE_DEVICE=gpu python scripts/zorl_fold_prime_overlay.py \
    --bf16-snapshot /shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-wc386k_lr1e5-sft \
    --fp8-snapshot  <the-fp8-quantized-copy> \
    --out /shared/apanda/zorl-marin-ps/prime-overlays/prime_overlay_delphi97b_gpu.pt
  ```
* **Numerics flags**: kept the sampling-relevant marin fleet flags (`fa3`,
  `--disable-custom-all-reduce`); intentionally omitted the batch-invariant /
  fp32-lm-head stack — that exists for GRPO sampler↔trainer *logprob* parity,
  and ES consumes only sampled text + reward.
* **Kept model-agnostic**: `philox_subseed_v2` noise layout (+ same
  `XORL_ZORL_NOISE_DEVICE=gpu` on BOTH manifests — flip together or the fold
  silently diverges from serving), seeds candidate transport, fresh_ab rank-1,
  `idle_session_timeout: 604800`, SMG router pattern, `EVAL_INTERVAL=0`.

## Pre-launch validation already done (2026-07-05, local GPU 2)

* `PYTHONPATH=. pytest experiments/marin/standalone/tests experiments/zorl/standalone/tests`
  → 111 passed (93 marin untouched + 18 new; interpreter
  `/home/apanda/xorl-internal/.venv/bin/python` — same venv the pods use, has
  `math_verify`/`datasets`/`transformers`).
* Driver `--dry-run` with the real checkpoint tokenizer + offline RLVR cache:
  pool 7168 / held-out 127 (one of 128 reserved dropped for prompt length),
  delphi chat template + `<|start_think|>\n` prefill render confirmed.
* Local sglang smoke on the EXACT PVC code (`/home/apanda/xorl-sglang-zorl` ==
  pod `/workspace/home/xorl-sglang-zorl`) with the exact manifest flags:
  server boots; `/generate` + `score_result` grade correctly (wrong −1.0,
  correct 650-token answer +1.0); B=0 rank-1 dense PEFT parent loads;
  `/register_zorl_candidates` materializes fresh_ab candidates from seeds over
  q/k/v/o+gate/up/down; a σ=0.3 candidate visibly diverges from base under
  greedy (noise genuinely applied); `/abort_zorl_generation` cleans up.

## Launch sequence (in order)

Capacity first: 64x1 GPU scorers + 4 GPU PS = 68 GPUs. Check free capacity
(`/gpu-usage`) and the wordle arm's footprint before creating the fleet.

```bash
# 0. One-time: stable client checkout for the pods (safe metadata-only op on
#    the primary; the live wordle run's /workspace/home/xorl-client is untouched).
git -C /home/apanda/xorl-client worktree add --detach \
    /home/apanda/xorl-client-zorl-marin-es zorl-marin-es
# xorl-infra side already exists: /home/apanda/xorl-infra-zorl-marin-es (branch zorl-marin-es).

cd /home/apanda/xorl-infra-zorl-marin-es/k8s/zorl

# 1. Scorer fleet (64x TP1). ~19.4GB weights each; ready in ~3-6 min.
kubectl apply -f qwen3-9b-marin-zorl-sglang.yaml
kubectl get pods -l app=zorl-marin9b-sglang   # wait for 64/64 Ready

# 2. SMG router (create, not apply — the Job uses generateName).
kubectl create -f zorl-marin9b-smg-router.yaml
curl -s http://zorl-marin9b-smg.apanda.svc.cluster.local:30000/model_info  # from any pod: router up

# 3. PRE-FLIGHT PARITY GATE (recommended; ~15 min): one 1-step gate run with
#    CANDIDATE_TRANSPORT=path, NUM_PAIRS=1, STEPS=1, MUON_LR=0 overridden via
#    env on a copy of the trainer Job. The PS then EXPORTS the candidate pair
#    from ITS noise streams; greedy-generate on a scorer with the exported
#    adapter vs the seed-registered twin of the same (b_seed, a_seed, sign)
#    must match token-for-token. This closes the ONE link the local smoke
#    could not: PS-fold noise == scorer-served noise for the dense fused-qkv /
#    split-gate-up stream naming (designed-in on both sides — server/zorl.py
#    "Seed -> noise contract (sglang-aligned)" vs sglang
#    _zorl_raw_b_entries_for_weight — but never yet exercised on a dense run).

# 4. Trainer + driver (Job with generateName):
kubectl create -f qwen3-9b-marin-zorl-ps-trainer.yaml
kubectl get jobs -l app=zorl-marin9b-ps
# logs: kubectl logs -f job/<name>   (driver log also lands in
# /shared/apanda/zorl-marin-ps/<RUN_ID>/driver.log, server.log alongside)

# 5. W&B forwarding (optional, wordle pattern): tail the driver log into the
#    marin project so the ES curve overlays the GRPO reference:
python experiments/zorl/standalone/wandb_log_forwarder.py \
    --log /shared/apanda/zorl-marin-ps/<RUN_ID>/driver.log \
    --project xorl-marin-rl-6279 --name zorl-es-marin9b-r1p1024
```

Teardown (never `kubectl delete` anything of the wordle arm):
`kubectl delete job -l app=zorl-marin9b-ps; kubectl delete sts,svc -l app=zorl-marin9b-sglang; kubectl delete job,svc -l app=zorl-marin9b-smg`.

## Step-0 acceptance gates

1. Cold base+think probe (automatic, printed by the driver): reward_mean over
   the 127 held-out examples at temp 0.7. The SFT checkpoint solves a decent
   fraction of RLVR (local smoke: 1/2 on the first example), so expect roughly
   −0.5…0.0 shaped reward — NOT −1.0 (all-wrong ⇒ prompt bug) and NOT 0.0-with-
   errors (probe path bug).
2. Step 1 `[scoring]` progress lines: units/s > 0 across all 64 owners; WARN
   lines about score failures should be ~0.
3. `apply_rewards` returns `update_norm` > 0 and `used_pairs` == 512.
4. After the first sync (~full p2p push), a fixed greedy prompt against any
   scorer must CHANGE vs pre-sync (weights actually landed) — and step-2 cold
   candidates must still score sanely (no wordle attempt-8-style corruption:
   if mean reward collapses toward −1/−2 right after the first sync, suspect
   fold/serving noise mismatch and stop).

## What to compare against

* W&B `together-research/xorl-marin-rl-6279`: run `nw155nmj` = the GRPO repro
  (train shaped-reward trajectory; last-10 mean **+0.436**), `3comlb0c` = the
  imported SkyRL reference (**+0.247** last-10, per-step peak +0.345; dedup by
  `trainer/global_step` keep-last). The ES per-step `mean=` (candidate mean ≈
  parent estimate, antithetic pairs cancel the O(σ) term) is the comparable
  curve; `best=` is the population max (optimistic).
* Same reward scale ([−2, 1], truncation −2, wrong −1) — curves are directly
  overlayable.
* Held-out: `EVAL_INTERVAL` is 0 by default (a full pop-1024 probe = 131k
  generations). For a periodic honest curve set `EVAL_INTERVAL=16` +
  `--eval-max-pairs 16` (32 cands × 127 examples ≈ 4k generations ≈ 1-2 min).

## Throughput estimate (64 scorers, pop-1024, single-turn)

Per step: 1024 candidates × 32 examples × 1 rollout = **32,768 generations**,
~60-300 prompt + ~650-1400 completion tokens each (local smoke; truncations cap
at 3584) ⇒ ~35-45M completion tokens/step.

* scoring: 64 × ~3-5k tok/s aggregate decode (9.7B bf16 TP1, 48-64 deep,
  rank-1 triton LoRA) ⇒ **~2.5-5 min/step** (worse in early steps if many
  3584-token truncations).
* candidate registration: 1024 seed-materialized adapters/replica (~5.7 MB
  rank-1 each, GPU philox) ⇒ ~0.5-1.5 min, overlapped per replica.
* fold (Muon, module-major, 512 pairs rank-1 dense) ⇒ ~1-2 min.
* p2p full bf16 push 19.4 GB → 64 receivers (no fold-aware delta at bf16;
  wordle measured 34.7 GB → 32 recv at 201.7 s) ⇒ **~2-4 min/step**.

Total ≈ **6-12 min/step ⇒ 128 steps ≈ 13-26 h**; `MAX_RUNTIME_SECONDS=82800`
(23 h) stops cleanly at a step boundary if the pessimistic end materializes.
Levers if too slow: `NUM_PAIRS=256` (halves scoring), `TRAIN_SIZE=16`,
`ROLLOUT_MAX_NEW_TOKENS=2048` (changes the reward's truncation point — then
also set `ZORL_MARIN_MAX_COMPLETION_TOKENS=2048`, and note the GRPO curve was
run at 3584).

## Open risks

1. **PS↔scorer noise-stream parity on dense (the main one).** Both sides
   implement the same sglang-aligned contract (fused `qkv_proj.lora_{A,B}`
   draw, fused `gate_up_proj.lora_A`, per-module gate/up B, 1:1 o/down), but
   no dense run has exercised it end-to-end. Mitigation: the pre-flight parity
   gate (launch step 3) + acceptance gate 4. A mismatch would look exactly like
   the wordle attempt-8/9 failure (reward-uncorrelated noise folded into base).
2. **fp32 master sizing**: 9.7B fp32 ≈ 38.8 GB ⇒ ~9.7 GB/rank at FSDP shard 4,
   plus full-gradient unshard transients and p2p staging — expected fine on
   4×80 GB, but watch rank-0 peak on the first apply+sync; escalate to
   `TRAINER_GPUS=8` / `data_parallel_shard_size: 8` if it OOMs.
3. **p2p step cost**: bf16 full push every step is the price of dropping the
   (fp8-only) fold-aware delta. If it dominates, the upgrade path is an FP8
   serving fleet + prime overlay (command above) — a separate validation cycle.
4. **max_loaded_loras host memory**: 1088 rank-1 dense adapters ≈ 6 GB/scorer
   pod on top of sglang — inside the 64 Gi request, but a scorer OOMKill
   pattern points here first (drop to NUM_PAIRS=256 ⇒ 512+1 loaded).
5. **Dataset edge cases**: RLVR rows without a parseable gold answer are
   skipped by marin's loader; one reserved held-out example is
   overlength-skipped (127 served). Deterministic, logged at task init.
6. **rank-1 + FSDP shard-4 export**: LoRA-A `[1, in]` shards to zero rows on
   ranks 1-3; the wordle rank-1 arm ran the same geometry through the same
   export path (all_ranks adapter load) without issue, but it is listed here
   because the wordle note about `rank >= dp_shard` was written for MoE-B
   export. First `parent_path` export failing would surface it immediately.
