# K3 reconciliation handoff — Qwen3-235B GRPO filler-token run (2026-06-28)

Audience: a new agent taking over the K3 (train↔serve logprob parity) work on the live 235B GRPO run.
This doc is self-contained. Read it top to bottom before touching anything.

---

## 0. TL;DR

- **Run:** Qwen3-235B-A22B GRPO, the paper's "Emergent Filler Token Preferences During RL" repro,
  on the reprogrammable-slots stack `er-opd-q235-fillerrft-slots` (ns `apanda`). Trainer = 8 nodes
  EP8; sampler = 1 node TP8; rollouts go **direct to the sampler** (SMG dispatch bypassed).
- **What we fixed:** live behavior-K3 (`kl/kl_sample_train_k3:mean`) went from **~2.7 → ~0.2 (≈14×)**
  via **two client-side routing-replay alignment fixes** (NOT engine/kernel changes). The MoE routing
  replay is confirmed consuming the sampler's routing in the deployed kernel.
- **RESOLVED (2026-06-29) — the "other stuff" was a packing loss-mask bug, K3 0.31 → 2.3e-4.**
  Packed `target_tokens` were padded with **token id 0** (a valid token) while the RL/IS loss keys off
  `target_tokens` (not `labels`). Those pad rows became fake "predict token 0, old_logprob=0.0"
  comparisons → the entire huge live-K3 tail (top-K3 tokens were all `target_id=0, old_logprob=0.0`).
  **Fix:** pad `target_tokens` with `IGNORE_INDEX` (+ valid-token accounting prefers `target_tokens`).
  After fix: K3 **0.00022960** (≈ the static floor), `ratio_mean=0.9998`, reward 0.584. So the kernel
  reconciliation recipe (§3) + routing replay (§4) were ALL correct; the residual was purely this mask
  bug. NOT attention, NOT weight-sync, NOT TP8/EP8 — those suspects (§5/§6) are moot.
  Found+fixed by the sibling k3 agent in `xorl-qwen-k3-reconciliation`; the live stack uses that repo so
  the fixed run already exists:
  `.../er-opd-q235-fillerrft-slots/20260629T045832Z-...-trainer-head/grpo/filler-rl-multiplication_4digit-8f6f1d2e/metrics.jsonl`.
  Ported into `xorl-apanda-dev-opd-port` too: `server/orchestrator/packing.py` (target_tokens→IGNORE_INDEX)
  + `trainers/training_utils.py` (count_valid_tokens / count_active_microbatches prefer target_tokens).
- **The run trains fine** — reward climbing (now 0.584), `eps_clip=0.2` bounds off-policy; with K3 now
  ~2e-4 the run is genuinely on-policy. §5/§6 below are kept for history but are superseded by this.

---

## 1. The stack (where everything lives)

- **Control dir (reprogrammable slots):** `/shared/opd-control/er-opd-q235-fillerrft-slots/<role>/run.sh`
  Roles: `sglang-0` (sampler), `dispatch` (SMG, **now bypassed for rollouts**), `trainer-head`,
  `trainer-worker-1..7`. To restart a role: append a line / edit its `run.sh` (hash change → slot
  agent re-execs the child; the pod keeps the node). **Never `kubectl delete` slot pods** (loses the
  9-node capacity). Logs: `<role>/logs/*-run.log`; status: `<role>/status`.
- **Trainer engine:** `/home/apanda/xorl-apanda-dev-opd-port` (XORL_REPO). 8-node mesh: EP=8,
  ulysses=8, dp_replicate=8, dp_shard=1, ep_fsdp=8 (world 64). Engine logs → `RUN_DIR/server.log`
  (NOT the head run.sh log — the head log is the launcher+client/`filler_tokens_rl.py` output).
- **Trainer config:** `/shared/apanda/filler_grpo/trainer_grpo_k3.yaml`.
- **Client (GRPO harness):** `/home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py`.
- **Sampler engine:** `/home/apanda/xorl-sglang-internal` (branch `apanda-dev`, carries PR #52
  "env-gated train/serve numerical-alignment knobs").
- **Results / metrics:** `/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-q235-fillerrft-slots/<RUN_ID>/`
  — `server.log` (engine), `grpo/*/metrics.jsonl` (per-step metrics incl. `kl/kl_sample_train_k3:mean`),
  `grpo/*/wandb/`. wandb project `together-research/xorl-prefill-time-compute`; each trainer re-trigger
  spawns a NEW wandb run id (named `grpo-235b-filler-mult4`) — sort by updated/step to find the live one.
- **Skills (READ THESE):** `/home/apanda/xorl-qwen-k3-reconciliation/skills/xorl-train-serve-parity/SKILL.md`
  (achieve parity — the recipe + divergence chain + localization workflow) and
  `.../skills/xorl-k3-correctness-check/SKILL.md` (measure parity — static-trace workflow).
  Full reference: `.../experiments/k3_tests/results/STAGE_SUMMARY.md` and
  `.../docs/notes/k3_zero_kld_handoff_2026_06_28.md`.

---

## 2. What K3 we are optimizing

`kl/kl_sample_train_k3:mean` = Schulman K3 = `mean(exp(log_ratio) − log_ratio − 1)`,
`log_ratio = sampler_logprob − trainer_recomputed_logprob`, on the **decoded/completion tokens** (the
IS denominator). This IS the "behavior K3" the skill targets (verified in
`xorl-apanda-dev-opd-port/src/xorl/ops/loss/importance_sampling_loss.py`). It is the training quantity;
do NOT chase prefill K3.

K3 trajectory this session (live, 235B):
- broken routing replay: **~2.7**
- + client routing-length pad (fix #1): **~0.9**
- + sampler-prompt-token alignment (fix #2): **~0.19–0.26** (current)

---

## 3. The reconciliation recipe (ALL applied + verified active)

Verified via the sampler's actual cmdline + the trainer config — these are NOT the residual:

**Sampler (`sglang-0/run.sh`), TP8:**
`--tp-size 8 --enable-return-routed-experts --rl-on-policy-target xorl-batch-invariant
--enable-fp32-lm-head --enable-fp32-router --disable-overlap-schedule --disable-cuda-graph
--disable-custom-all-reduce --attention-backend fa3 --sampling-backend flashinfer`
plus env `SGLANG_DISABLE_ROPE_COMPILE=1` and `SGLANG_RMSNORM_FP32_WEIGHT_MUL=1` (both confirmed in
`/proc/<pid>/environ`). `SGLANG_FLA_TRIL_PRECISION=ieee` is N/A (GDN-only; 235B is not GDN).

**Trainer (`trainer_grpo_k3.yaml`):** `router_fp32: true`, `lm_head_fp32: true`,
`rmsnorm_mode: native`, `ce_mode: eager`, `attn_implementation: flash_attention_3`.
`enable_high_precision_for_bf16()` is called unconditionally (`model_runner.py:371` — tf32 off,
bf16-reduced-accum off). Also for the OOM fix: `enable_gradient_checkpointing: true`,
`sample_packing_sequence_len: 4096`.

---

## 4. The TWO client-side fixes that took K3 2.7→0.2 (the real wins)

Both are in `filler_tokens_rl.py`. Root cause class: the engine + RoutingReplay are correct (the K3
agent measured 5.94e-5 statically); the bugs were in how THIS client prepared the routing/prompt.

**Fix #1 — routing length under packing (~line 1994, the datum-build loop).**
The sampler records routing for `prompt+gen−1` positions (the final generated token is never
forwarded). A single datum's off-by-one is absorbed by micro-batch end-pad (why the static test was
clean), but the trainer PACKS many datums (`sample_packing_sequence_len=4096, pad_to_multiple_of=128`)
and `RoutingReplayHandler._build_per_mb_routing` concatenates per-datum routing back-to-back, so each
short datum shifts every later datum cumulatively → wholesale misroute → K3~2.7. **Fix:** decode each
datum's `routed_experts` (int32, shape `[ntok, 94 MoE layers, 8 topk]`, stride 752), pad(repeat last
row)/truncate to `model_input.length`, re-encode base64. → 2.7→0.9.

**Fix #2 — prompt mismatch (~line 1640 + ~line 1777).**
`api_format=generate` does NOT echo prompt token IDs, so the client fell back to the LOCAL renderer
(`renderer_model_name=Qwen3-32B` + the enable-thinking block), which differs from the sampler's actual
prefill by a fixed ~4 tokens. The trainer then scored a different prompt than the sampler saw → both
the routing boundary AND the completion logprobs misalign. **Fix:** (a) pass `logprob_start_len=0` to
the rollout `client.sample(...)` so `meta_info.input_token_logprobs` is populated (per-prefill-token
`[logprob, token_id, text]`); (b) extract the sampler's EXACT prefill token IDs (index `[1]`) and use
them as `prompt_tokens` (was `getattr(sequence, "prompt_tokens", None)` → None for generate). → 0.9→0.2.
After this, `[ALIGN-FIX]` logs `routing_ntok == fwd_len` exactly (e.g. 953→953) — fix #1's pad becomes
a no-op safety net.

**Verification (engine instrumentation, in `models/layers/moe/moe_block.py`, route() replay_forward
branch):** `[REPLAY-VERIFY]` logs to `server.log` — confirmed `replay_forward` fires across all 94 MoE
layers and the sampler's routing **overrides** the trainer's own router on **0.8–15% of tokens/layer**
(the borderline flips). So replay IS consumed in the deployed kernel. (This + the `_REPLAY_VERIFY_N`
counter + `import logging/logger` are debug-only; remove or keep — they're cheap, first-6-calls-only.)

---

## 5. Dead ends / things already ruled out

- **SMG dispatch strips `routed_experts` on `/generate`.** The `isr3k3` patched binary
  (`/shared/apanda/wordle-sft-runs/smg-isr3k3-bin`) only preserves them on `/v1/chat/completions`, not
  `/generate`. **Resolution:** bypass the dispatch — `inference_base_urls=sglang-0:30060` direct (single
  sampler, so the router is unnecessary; also removes the dispatch's stale-health 503 failure mode).
  The direct sampler `/generate` returns `routed_experts` intact (proven: PRESENT len=48128).
- **fp32 / batch-invariant / RoPE / RMSNorm / ce_eager** — each verified ACTIVE but none independently
  moved live K3 off ~2 until the routing+prompt alignment was fixed. They are necessary (static recipe)
  but were not the live driver.
- **EP8 sampler (to match the trainer's EP8 expert layout).** ABANDONED. (a) Owner's steer: TP8-vs-EP8
  is NOT the source. (b) The build can't do single-node EP8 anyway: `deepep` is **not installed** in
  the sampler venv (`ModuleNotFoundError: deep_ep`); `flashinfer` a2a needs multi-node NVLink and dies
  intra-node (`MnnvlMemory has no attribute 'ptr'`, requires `--moe-runner-backend flashinfer_cutlass`);
  `mori`/`nixl` absent; `mooncake` importable but RDMA/multi-node-oriented. NOTE: `deepep` IS installed
  in the **xorl/trainer** env, so install instructions exist if EP8 is ever revisited — but per the
  owner it's not needed. `return_routed_experts` capture is at `topk.py` (runner-independent), so an EP
  runner would NOT have broken replay.

---

## 6. THE OPEN PROBLEM: the ~0.2 residual ("other stuff")

The static recipe → ~6e-5 (identical weights, STAGE_SUMMARY). Live → ~0.2. The gap is live-specific and
per the owner is NOT the parallelism layout. Prime suspects, and how to localize each:

### 6.0 CROSS-FINDINGS from the q36 sibling investigation (READ FIRST — they reframe this section)
The sibling agent ran the SAME class of problem on q36 (Qwen3.6-35B-A3B, EP8 trainer / TP2 sampler).
Their results (memory `wordle-live-k3-floor.md`; handoff `experiments/wordle/K3_REPLAY_HANDOFF_2026_06_28.md`)
directly change the 235B priorities:
1. **Routing replay HURTS, root cause UNRESOLVED.** With the SAME client routing-alignment fixes we used
   (per-datum pad + byte-identical prompt), q36 measured **k3=0.35 WITH replay vs 0.053 no-replay** —
   replay LOSES ~7×. Every component tests correct OFFLINE (triton kernel applies forced routing
   bit-exact; `route()`/`_regather_routing` no-op for same experts; packing concat aligned; patched SMG
   returns correct per-row routing; SGLang capture bit-identical alone-vs-batched) — yet end-to-end it
   misaligns. **IMPLICATION for 235B: we never measured the no-replay baseline WITH the prompt fix.** Our
   earlier "R3-ON == R3-OFF ≈ 2" was PRE-prompt-fix. The post-fix no-replay number is UNMEASURED — replay
   may be hurting on 235B too, i.e. our "0.2 with replay" could be beatable / replay may be the wrong lever.
   → **TOP PRIORITY: measure 235B no-replay (return_routed_experts=false) WITH the prompt fix.** If
   no-replay < 0.2, turn replay OFF (it's hurting) and the residual is purely attention/distribution.
2. **Weight-sync is NOT the floor — CROSS IT OFF.** `--enable-rdma-weight-updates` is a bit-identical
   COPY (EP→TP reshards layout, not values), so sampler & trainer hold identical weights; q36 step-1
   (freshest sync, no optimizer drift) is already at floor. So suspect #2 below (weight-sync fidelity)
   is essentially refuted by the sibling — deprioritize it.
3. **The floor is temp/distribution + the irreducible paged-vs-contiguous FA3 attention** (STAGE_SUMMARY
   §7 item 5, "no flag" — sampler paged-KV vs trainer contiguous-KV). That, not kernel precision or
   weight-sync, is what's left after the recipe.
4. **The decisive UNTESTED rung (both models): real-model single-sequence replay of offline traces
   through the xorl forward** — isolates the replay misalignment from packing/batching/SMG. q36 uses
   `/shared/apanda/wordle-sft-runs/k3_offline_traces.json`; build the 235B equivalent. This is the single
   highest-value experiment for the "does replay actually help, and if not why" question.

1. **FA3 paged-vs-contiguous attention (handoff's #1 open).** Both engines use fa3, but the sampler
   reads paged-KV (`flash_attn_with_kvcache`) and the trainer uses contiguous `flash_attn_varlen_func`
   → different block/reduction order → ~1-ulp attention-output diff. "Irreducible by flags; needs
   unifying the attention kernel." Localize with the **per-layer hidden-state diff** (skill §"Localize
   a residual"): dump xorl per-op hidden states via `model_runner.py` diagnostic params
   (`diagnostic_hidden_components` / `_layers` / `_path`), compare to SGLang's per-layer states on the
   same prompt; the attention-output residual should be the first to exceed bf16 round-off.
2. **Weight-sync fidelity — LIKELY REFUTED by the sibling (§6.0 item 2):** RDMA weight update is a
   bit-identical copy and q36 step-1 freshest-sync is already at floor. Only revisit if the 235B EP8→TP8
   reshard is shown to differ from q36's EP8→TP2 (unlikely to matter for VALUES). Deprioritized.
3. **Sampling/logprob/temperature.** temp=1.0 (so temp-scaling is a no-op) and the PR-52 fp32
   temp-scaled-logprob path is active via `rl-on-policy-target`; per-row random `sampling_seed` is set
   per sample (was implicated in the wordle k3→112 blow-up under deterministic mode, but here K3 is
   stable ~0.2 — likely not dominant now). Lower priority.

### Recommended first move (cheapest, highest signal) — UPDATED per §6.0
**(a) Measure the 235B no-replay baseline WITH the prompt fix.** Set `return_routed_experts=false`
(R3 off) keeping everything else, run a few steps, compare k3 to the 0.2 replay number. The sibling
showed replay HURTS on q36 — if 235B no-replay < 0.2, **turn replay off** and the residual is attention/
distribution only. This is one trainer re-trigger (no sampler reload). DO THIS FIRST.
**(b) Log the per-sample K3 distribution** (TAIL vs SYSTEMATIC). The client already captures per-sample
k3 (`filler_tokens_rl.py:2978` `loss_output.k3` → `all_step_samples[...]["k3"]`). Log median/p95/max/
frac>1.0 per step:
- **Tail** (median≈0, high p95/max on a few tokens) ⇒ near-argmax-tie flips ⇒ attention (#1) or
  residual routing misalignment — chase the offline-trace replay (§6.0 item 4) + per-layer hidden diff.
- **Systematic** (all samples ≈0.2) ⇒ distribution/attention global term (weight-sync is refuted, §6.0).

### Definitive localization (needs GPUs for the xorl replay)
Run the skill's static-trace workflow against THIS sampler:
`experiments/k3_tests/make_static_traces.py --sglang-url http://...sglang-0:30060` (capture decode
logprobs + routing) → `compare_static_traces.py --reference-logprobs generation` (replay through xorl).
If static behavior-K3 ≈ 6e-5, the kernel recipe is perfect and the entire live 0.2 is
weight-sync/distribution. If static is also high, a kernel lever is still off and the per-layer diff
finds it. (Logistics: all GPUs are in the slots; the xorl replay needs a trainer instance — may need to
pause the live run or borrow it between steps.)

---

## 7. Operational notes / gotchas

- **Re-trigger needs a fresh sampler ONLY if a sync was attempted.** A stale `weight_sync_group`
  wedges the sampler engine (`/health` 200 but `/generate` hangs; `complete_weights_update`/
  `destroy_weights_update_group` both 400 "nothing in progress" — only a reload clears it). If a run
  dies BEFORE any sync, the sampler p2p state is clean and you can re-trigger the trainer alone.
- **The watcher stale-metrics trap:** `ls -t .../metrics.jsonl | head -1` can grab a PRIOR run's file
  before the new run writes its own. Derive `RUN_DIR` from the head log (`grep ^run_dir=`) and read
  that run's metrics, or you'll report stale K3.
- **Engine logs (incl. `[REPLAY-VERIFY]`, `R3:`) go to `server.log`; client logs (`[ALIGN-FIX]`) go to
  the head run.sh log.** Grep the right file.
- **Sampler reload ≈ 13 min** (235B shard load). Probe `/generate` (200) after "fired up and ready" —
  HTTP `/health` can be 200 while the engine is wedged.
- Scratchpad orchestration scripts from this session:
  `/tmp/claude-0/-home-apanda-xorl-opd-prefill/08702b41-075c-4bf2-8502-fd28f7e7cc4d/scratchpad/`
  (`k3_*.sh` — reload+probe+re-trigger+watch patterns; reuse them).
- Memory: `~/.claude/.../memory/wordle-live-k3-floor.md` (k3 knowledge, incl. the corrected note that
  the 235B k3~2 was a client routing/prompt alignment bug, not a weight-sync floor) and
  `reprogrammable-slots-235b-launch.md` (the slots infra + bring-up gauntlet).

---

## 8. Current live state (as of handoff)

- Sampler reverted to **TP8** (the EP8 attempts crash-looped it); run being restored to k3≈0.2 with
  correct replay. Sampler run.sh has the full recon recipe (§3).
- Client `filler_tokens_rl.py` has both routing fixes (§4) + the `[ALIGN-FIX]` log; engine
  `moe_block.py` has the `[REPLAY-VERIFY]` debug. trainer config has the OOM fix (recompute+pack4096).
- Next action per §6: log per-sample K3 distribution → localize tail vs systematic → chase attention
  (per-layer hidden diff) or weight-sync accordingly.
