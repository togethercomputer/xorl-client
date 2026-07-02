# K3 Replay Handoff — drive down live training k3 for GRPO Qwen3.6-35B-A3B

**Date:** 2026-06-28
**Mission:** Drive down the **live training** importance-sampling k3 (`train/loss/is_kl_sample_train_k3:mean`) for our exact GRPO Qwen3.6-35B-A3B Wordle setup, using **targeted offline tests** that faithfully replicate the live path. The central open question: **routing replay measures k3≈0.35 (worse than no-replay 0.053) even though every component tested in isolation is correct.** Find the misalignment by building a test that proves routing replay works end-to-end, then make the client match it — adding the live confounders (triton MoE, packing, SMG, per-row seeds, batched decode) one at a time.

Do NOT just turn replay off and call it done — that's the current production fallback, not the answer. The point is to understand and fix the replay path, and to separately characterize the single-sequence-vs-batched k3 gap.

---

## TL;DR of where it stands

- **no-replay** batch-invariant floor: **k3 ≈ 0.053–0.060** (run `GRPO-WQ36-1node-ISR3-parity` ran to step 128 at 0.0605). This is the current production config and the live floor.
- **replay, alignment-fixed**: **k3 ≈ 0.35**, `ratio_mean=0.83`, `ratio_max=41` (run `GRPO-WQ36-1node-MINK3-replay-fixed`, step 1). **Worse** than no-replay.
- **replay, pre-fix (broken packing)**: k3 ≈ 0.22.
- **Static reconciled floor** (the k3-reconciliation agent, see their `STAGE_SUMMARY.md`): Coder-30B + **eager** MoE + decode-route replay + temp=0 → **5.94e-5**; q36 GDN reconciled (no replay, temp=0) → **3.12e-4**. These are **temp=0 / eager-MoE / single-sequence** — almost certainly **unreachable live** (we sample at temp 0.7, use triton MoE, and pack).

**The paradox that must be explained (it's the whole ballgame):** if no-replay k3 is 0.053, the trainer's hidden states are *demonstrably close* to the sampler's. Forcing the sampler's **actual** experts onto near-identical hidden states must give k3 ≈ 0.05, not 0.35. A 7× jump is only possible if **the experts being replayed are not the sampler's experts for those tokens** — i.e. a real misalignment. But every component below tested correct. The bug is in the **real-model end-to-end** path that component tests don't cover. **Your job is to find it with a faithful end-to-end replay test.**

---

## The exact setup (replicate this faithfully)

### Model
Qwen3.6-35B-A3B — **GatedDeltaNet (linear attn) + MoE hybrid**, 40 hidden layers (all MoE), 256 experts, top-8. `num_experts_per_tok=8`, `num_hidden_layers=40`. Routing blob from `/generate` `meta_info["routed_experts"]` is int32, shape **`[P+out−1, 40, 8]`** (prefill on the P prompt tokens + decode on out−1 generated tokens; the final generated token is never forwarded so has no routing row). Verified empirically.

### Trainer (xorl), 1 node EP8
Config: `/shared/apanda/wordle-sft-runs/configs/grpo-ep8x1node-muon-lowlr-isr3k3.yaml`
- `attn_implementation: flash_attention_3`, **`moe_implementation: triton`**, **`ep_dispatch: alltoall`**
- `expert_parallel_size: 8`, `tensor_parallel_size: 1`, `data_parallel_shard_size: 8` (no DP data split)
- `rmsnorm_mode: native`, **`ce_mode: compiled`**, `lm_head_fp32: true`, `router_fp32: true`, `enable_high_precision_for_bf16` always-on
- `enable_gradient_checkpointing: true`, `gradient_checkpointing_method: recompute_full_layer` (slow; k3-neutral — switch to `recompute_before_dispatch` for speed)
- optimizer **muon bf16** (adamw fp32 OOMs EP8; EP4 OOMs/hangs the GRPO fb — keep EP8+muon)

Builder: `/shared/apanda/wordle-sft-runs/build_grpo_wq36_1node_isr3k3.py` →
`--group-size 16 --train-size 32` (=512 rollouts/step), `--max-length 6144`, `--student-temperature 0.7`, `--student-max-new-tokens 2048`, `--student-generation-batch-size 16 --student-generation-workers 48`, `--sampler-world-size 2`, `--lr 5e-6 --lr-schedule cosine --lr-warmup-steps 8`, `--loss-fn importance_sampling --per-rollout-seed`, `--gradient-clip 1.0`, `--compute-kl-stats`. (`--return-routed-experts` toggles replay on.)

### Samplers (SGLang), TP2 × 8 replicas
Manifest: `/shared/apanda/wordle-sft-runs/launch/wordle-sci-opsd-sampler-b.yaml`. PYTHONPATH = `/workspace/home/xorl-sglang-internal/python` (the merged PR#52 source).
Flags: `--tp 2 --sampling-backend flashinfer --rl-on-policy-target xorl-batch-invariant --enable-fp32-lm-head --enable-fp32-router --enable-return-routed-experts --trust-remote-code`, `mem-fraction-static 0.70`, `max-total-tokens 16384`, `cuda-graph-max-bs 64`, `max-running-requests 256`.
Envs: `SGLANG_FLA_TRIL_PRECISION=ieee`, `SGLANG_DISABLE_ROPE_COMPILE=1`, `SGLANG_RMSNORM_FP32_WEIGHT_MUL=1`.
**Note:** `rl-on-policy-target` forces `enable_deterministic_inference=True`. Per-row sampling needs `multinomial_with_seed`; there was a NaN-argmax edge (hash==0xFFFFFFFF → +inf gumbel → NaN) fixed with `nan_to_num` in `sampler.py` — confirm it's present in the deployed sglang source or per-row seeds blow k3 to 100+.

### SMG (Shepherd Model Gateway / model_gateway)
Deployed binary: **`/workspace/home/smg-together-thunderagent-port/target/debug/smg`** (our patched debug build), `--policy round_robin --backend sglang`, worker URLs = the `wordle-sci-opsd-sampler-b-*` samplers (overridden via `WORKER_URLS` env; the script's default `opsd-wordle-q36-sglang-*` is NOT what's used). Service: `wordle-grpo-smg:8080`.
Our patches (vs base `thunderagent-policy-port`): (1) `sampling_params` widened to untagged `Single(SamplingParams) | Batch(Vec<SamplingParams>)` enum + `representative()` helper (gRPC takes first row); (2) `return_routed_experts` / `return_expert_logits` added as `#[serde(default)] bool` typed fields (without them `serde_json::to_value(typed_req)` dropped the flag → no routing emitted). Response is **raw-bytes passthrough** so per-row routing survives. These are on the `xorl` branch of `togethercomputer/together-smg`.

### Client
`experiments/wordle/standalone/train_grpo_wordle.py` (+ `grpo_rl_shim.py` providing `build_policy_loss_inputs`). Key behaviors:
- Sends **`input_ids`** (not text) → sampler prefills the *exact* prompt tokens (no local-renderer / tokenizer divergence; this is why our prompt is byte-identical, unlike the 235B filler experiment which re-rendered with the wrong model and needed a "sampler prompt tokens" fix).
- Batched generation: splits 512 rollouts into batches of 16, dispatches concurrently (ThreadPoolExecutor, 48 workers) through the SMG; each batch → round-robin → one sampler.
- Per-row `sampling_params` list with unique `sampling_seed` per row (diversity under deterministic samplers).
- Routing decode + slice + pad lives at `train_grpo_wordle.py` ~lines 648–700: `_decode_routing(b64, int32, L=40, K=8)` → `arr.reshape(-1, L, K)`; slice `experts_arr[:_keep]` where `_keep = len(prompt_ids)+len(output_ids)`; then the **per-datum pad** (added this session) that extends each datum's routing to `_keep` by duplicating the last (irrelevant) row, so the packed concat stays aligned.

---

## What has been RULED OUT (with the test that proved it)

Each of these is **correct** — do not re-litigate without new evidence; instead build *on top* of them:

1. **Triton MoE kernel applies replayed routing correctly.** `/shared/apanda/wordle-sft-runs/test_triton_replay.py` (run on 1 GPU): forcing the natural experts via the full replay machinery is a **bit-exact no-op** (`max_abs=0.0`) for both eager and triton; forcing *arbitrary* experts gives **eager==triton bit-exact**; order-invariant. So the kernel computes any forced routing correctly, and the live 0.35 is **not** triton-specific (eager would give the same).
2. **`route()` replay logic** (`xorl/models/layers/moe/moe_block.py`): `_regather_routing` (softmax→gather→renorm) is algebraically equal to the natural `TopKRouter` path; weights pair correctly with experts; expert combine is an order-independent sum. No systematic weight error.
3. **Packing concat** (`server/runner/utils/routing_replay_handler.py::_build_per_mb_routing`): with the per-datum pad, the synthetic marked-routing test (`scratchpad/test_routing_align.py`) shows forward-position p gets routing-row p across packed micro-batches (`aligned=True`); the raw/no-pad case correctly reproduces the original 0.22 cumulative-shift bug.
4. **SMG per-row batched routing**: batched request (distinct prompt lengths) through `wordle-grpo-smg:8080` → every row's routing length == its own `(P+out−1)×320`, signatures distinct. Correct.
5. **SGLang continuous-batching routing capture**: same prompt+seed **alone vs inside a batch** → bit-identical generated tokens AND bit-identical routing (0/8640 element mismatches). Correct.
6. **Decode layout**: client `_decode_routing` and engine `_decode_routing_array`/`_infer_shape` both reshape `[T, L, K]` C-order and agree.
7. **Weight sync is bit-identical** (RDMA copy; EP8→TP2 reshards layout not values). Not a source of k3. (Earlier "weight-sync floor" claims are WRONG — do not chase weight-sync fidelity.)

---

## Your job — the targeted test ladder

Build a **faithful end-to-end replay test** and walk confounders in one at a time. The decisive first rung answers everything:

**Rung 0 — real-model single-sequence replay (DO THIS FIRST).**
Captured clean single-sequence traces already exist: `/shared/apanda/wordle-sft-runs/k3_offline_traces.json` (3 traces: `prompt_ids`, `output_ids`, `full_ids`, `prompt_len`, `sglang_generation_logprobs`, `sglang_decode_routed_experts_b64`; lengths verified `(P+out−1)×320`). Run them through the **actual q36 xorl forward with routing replay** and compute k3 vs `sglang_generation_logprobs`.
- Harness: `xorl-qwen-k3-reconciliation/experiments/k3_tests/compare_static_traces.py --xorl-url <server> --reference-logprobs generation` with env `K3_REPLAY_ROUTING=1 K3_ROUTING_SOURCE=decode`. Launch the xorl server via `launch_k3_test.py --model qwen3.6-35b` (profile uses `configs/qwen3_6_35b_ep8_deepep.yaml`, 8 GPUs) — **but change its `moe_implementation` to match the live trainer (`triton`/`alltoall`), and add a `recompute_before_dispatch` variant**; the shipped profile uses `quack`+`deepep`, which is a *third* kernel and not what we run.
- **Interpretation:** low k3 (~0.05 or better) ⇒ single-sequence replay is correct, so the live 0.35 is in **multi-datum assembly** → go to Rung 2. High k3 (~0.35) ⇒ routing content/alignment is wrong even for one clean sequence → instrument the trainer's datum construction (Rung 1).
- **CRITICAL methodology the user asked for:** the test must drive the **client's own routing-prep code** (`_decode_routing` + slice + pad from `train_grpo_wordle.py`), not just the raw trace blob, so you're testing *our client*, not the harness's idealized path. Feed the client-prepped routing into the engine and compare against feeding the raw blob — any divergence is a client bug.

**Rung 1 — eager vs triton, single sequence.** If Rung 0 is high, run the same trace with `moe_implementation=eager` (aten-hookable, reconciled — should hit the agent's ~6e-5) vs `triton`. Isolates whether the (lateral, ~1-ulp) kernel numeric gap matters at temp 0.7 vs whether the content is wrong.

**Rung 2 — batched / packed (multi-datum), triton, replay.** Pack 2+ traces into one micro-batch (use the real `SequentialPacker` from `xorl/server/orchestrator/packing.py`, sequential — no length sort), build the `routed_experts` list the way the client does, run `_build_per_mb_routing` + the forward + replay. Compare per-datum k3 to Rung 0. If it jumps here, the bug is the **routed_experts-list ↔ packed-datum order** correspondence (the one thing the synthetic packing test *assumed*: that the list order matches the micro-batch datum order). Verify the order with a marked-routing assertion at runtime (like the 235B `[ALIGN-FIX]` log, but checking datum-to-row *order*, not just token *count* — count-match misses an order swap of equal-length datums).

**Rung 3 — full live path.** Dispatch through the SMG with per-row seeds and the concurrent ThreadPoolExecutor (batch-size 16, 48 workers), reassemble, and confirm `routed_experts[i]` still belongs to datum i after the concurrent gather (mapping must be by index, not completion order — logprobs being aligned at no-replay 0.053 suggests it is, but verify for routing specifically).

**Separately: the single-sequence-vs-batched k3 issue (independent of replay).** Characterize whether *batching/packing itself* raises k3 vs single-sequence, with replay OFF — i.e. is the xorl forward truly batch-invariant under packing at 6144 length (the `down_proj` K≥6144 cuBLAS-vs-triton divergence, `XORL_BATCH_INVARIANT_MATMUL=1`, is K3-neutral per the agent but untested live; `ce_mode=eager` is the one unpulled parity lever — agent says K3-neutral at temp=0, untested at temp 0.7). Measure no-replay k3 single-seq vs packed on the same data.

### How to verify "routing replay actually works" cleanly
The gold-standard unit test the user wants: take ONE real trace, run the xorl forward (a) **no replay** (trainer's own routing) and (b) **replay** forcing the sampler's experts. If alignment is correct, (b) should be **≤** (a) in k3 (replay can only help or no-op, never hurt, when the experts are right and the hidden states are close). If (b) > (a), the replayed experts are wrong for those positions — bisect with the marked-routing assertion to find where the index mapping breaks.

---

## Harness, files, paths

- **k3 reconciliation harness** (read `STAGE_SUMMARY.md` and `routereplay_findings.md` first — they have the operator-parity table, the §5 "replay forces experts not the GEMM order; eager harness ≠ production grouped kernel" caveat, and the §6 "the floor is bf16 tie degeneracy; below ~1e-4 has no training benefit" conclusion): `/home/apanda/xorl-qwen-k3-reconciliation/experiments/k3_tests/` — `compare_static_traces.py`, `launch_k3_test.py`, `make_static_traces.py`, `static_trace_utils.py`, `configs/qwen3_6_35b_ep8_deepep.yaml`.
  - `--local-forward` does **NOT** apply replay (plain HF forward) — replay needs the `--xorl-url` server path (`routing_replay_applied = not local_forward`).
- **Engine MoE/replay code** (`/home/apanda/xorl-internal/src/xorl/`): `models/layers/moe/{moe_block.py (route, _regather_routing, forward), router.py, routing_replay.py, experts.py, backend/{eager,triton,quack,native}.py}`; `server/runner/utils/routing_replay_handler.py` (`fill_routing_replay`, `_build_per_mb_routing`); `server/orchestrator/packing.py` (`SequentialPacker`); `server/runner/runner_dispatcher.py` (`_select_and_prepare_batches`, lines ~578–800).
- **Client**: `experiments/wordle/standalone/train_grpo_wordle.py` (routing prep ~648–700; sends `input_ids`; batched dispatch ~573), `grpo_rl_shim.py`.
- **Tests written this session** (reuse/extend): `/shared/apanda/wordle-sft-runs/test_triton_replay.py`, `scratchpad/test_routing_align.py` (in this session's scratchpad — recreate from the rungs above if gone), `k3_offline_traces.json` (3 captured traces).
- **wandb**: project `together-research/xorl-wordle`. Key metrics: `train/loss/is_kl_sample_train_k3:mean`, `train/loss/is_ratio_max:mean`, `train/loss/is_ratio_mean:mean`. Compare runs `*-noreplay-climb`, `*-replay-fixed`, `ISR3-parity`, `ISR3-K3RECON-batchinv-fp32`.

---

## Constraints (read before touching infra)

- Write all outputs ONLY under `/shared/apanda/wordle-sft-runs`.
- NEVER share a sampler pool between two trainers. NEVER touch other experiments' shared stacks: `er-opd-q36*`, `zorl-ar-*`, `q36mtp`/MTP, `opsd-wordle-q36*`. Only delete pods/jobs you created.
- Do NOT break the shared venv (`/home/apanda/xorl-internal/.venv`) or the shared SMG binary (restore any patched binary; 3 other live SMGs depend on the shared one — `opsd-wordle-q36-smg`, `q36mtp-teacher-smg`, `zorl-ar-smg`).
- The current production run is `GRPO-WQ36-1node-MINK3-noreplay-climb` (no-replay, parity levers on) — the practically-correct fallback. Don't kill it for test GPUs; the cluster has ample free GPUs (launch the xorl test server on its own allocation; inference-only q36 fits on 2 GPUs TP2).
- Cluster: namespace `apanda`, nodes `node-group=default`, image `nvcr.io/nvidia/pytorch:26.02-py3`, banned nodes h100-005/050/080/116/105.

## Secondary levers (after replay is understood, or if replay proves fundamentally unhelpful live)
- **Temperature ablation**: knob is `--student-temperature` (currently 0.7). Run 2 steps at ~0.05 (near-argmax) to test whether the live floor is temp-0.7 exposure of the irreducible paged-vs-contiguous FA3 attention boundary (§7 item 5, "no flag") — argmax hides the lateral 1-ulp diffs that temp 0.7 samples into.
- **eps_low / eps_high (the user's standing question)**: the agent's §6 says the residual high-k3 tokens are exactly what TIS/clipping is for. If replay can't reach the static floor live (likely, given the attention boundary), the right lever for *training stability* is tuning the IS clip bounds, not more parity. Worth an explicit ablation: does widening/narrowing eps change the climb stability at the same k3?
