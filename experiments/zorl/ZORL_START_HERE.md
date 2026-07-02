# ZORL — START HERE (for a new agent)

*2026-07-02. Single pointer into the ZORL (zeroth-order RL / Evolution Strategies)
work after the get-right consolidation. Read this first, then `ZORL_CANONICAL.md`.*

## What ZORL is (one line)

Train a frozen LLM with Evolution Strategies — no backprop, only forward
inference + scalar rewards — where every candidate model and every update is a
**random seed**, an entire population is served at once by a multi-LoRA SGLang
fleet, and the reward-weighted update is folded (Muon) into an fp32-master
parameter server. Full story: `ZORL_CANONICAL.md` (same directory).

## Where the code lives (four checkouts — the work is cross-repo)

Each repo consolidated to a single **hub checkout on `apanda-dev`** (2026-07-02);
the per-stream side worktrees/branches are historical.

| Role | Directory (hub) | Branch |
|---|---|---|
| **client / driver** (start here) | `/home/apanda/xorl-client` | `apanda-dev` |
| **PS / trainer** (fold, seeds, endpoints) | `/home/apanda/xorl-zorl-ps` | `zorl-ps` |
| **scorers** (multi-LoRA SGLang) | `/home/apanda/xorl-sglang-zorl` | `zorl-ps-fp32` |
| **infra** (k8s/configs) | `/home/apanda/xorl-infra` | `apanda-dev` |

**Start in `/home/apanda/xorl-client`** (the unified hub: `experiments/wordle/`
recipe+docs+orchestration, `experiments/zorl/` the ZORL work, everything else).
Key files:
- `experiments/zorl/standalone/run_wordle_zorl_xorl_ps.py` — the PS-mode driver (the live run).
- `experiments/zorl/standalone/zorl_client.py` — scoring / rollout / task glue.
- `experiments/zorl/standalone/xorl_ps_client.py` — PS HTTP client.
- `experiments/zorl/standalone/tasks/` — wordle, mult, etc.
- `experiments/zorl/wandb_filter_promising.py` — rank runs by own-metric gain.

⚠️ **Do NOT develop in `/home/apanda/xorl-apanda-dev-zorl-consolidated`** — it is
SUPERSEDED. The design docs still live there (see below) but the code homes are
the four repos above.

## Docs to read, in order

1. `experiments/zorl/ZORL_CANONICAL.md` — science + systems overview (this repo).
2. `/home/apanda/xorl-client-wordle-science-20260614/experiments/wordle/WORDLE_RECIPE.md`
   — the GRPO recipe we are matching. **Matching it exactly is the science win.**
3. `/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/PS_AS_XORL_TRAINER_DESIGN.md`
   + `PS_AS_XORL_TRAINER_R1_VERDICT.md` — why the PS is an xorl trainer + the numerical parity gate.
4. `…/experiments/zorl/FP32_MASTER_WEIGHTS_DESIGN.md` — the sub-ULP precision story (why fp32 master).
5. `…/experiments/zorl/ZORL_WORDLE_FRESH_AB_FP8_ALGORITHM_AND_INFRA.md` — algorithm + infra topology.
6. Auto-memory (loaded each session): `zorl-get-right-consolidation`,
   `zorl-grpo-match-directive`, `mooncake-p2p-preference`, `zorl-beat-093`,
   `zorl-wordle-think-budget-independent`, `zorl-bf16-fold-drops-small-updates`.

## The current live run (state as of 2026-07-02)

**Goal:** the GRPO-matched Wordle science run — `fresh_ab`-into-base through the
xorl PS at GRPO's own recipe (muon lr 5e-6 cosine warmup 8, steps 128, same
prompt/reward/eval), candidates as seeds, base sync via Mooncake p2p fp8.

- Launch manifest: `xorl-infra` `k8s/zorl/qwen3_6-35b-a3b-zorl-wordle-ps-trainer-grpo-match.yaml`
  (self-contained: waits for scorers → starts PS → runs driver). Scorer pool
  `qwen3_6-35b-a3b-zorl-wordle-w35-sglang.yaml` (32×TP2 FP8) + `zorl-w35-smg-router.yaml`.
- **Status (2026-07-03, overnight): ONE code-level blocker remains** (expert
  weight-name fusion in the sync). 4 debug scorers left UP (warm, all env fixes
  applied) for fast iteration; trainer killed. ~11 more bugs cleared past the
  original 8 — the fresh_ab fold works end-to-end AND the full 35B model
  transfers over Mooncake RDMA; only the receiver's expert tensor-map naming
  rejects the pushed weights. **Debug at 4 replicas** (`SGLANG_REPLICAS=4`,
  `NUM_PAIRS=8`, `TRAIN_SIZE=8`, `STEPS=4`) — cheap ~15-min cycle to the sync.
- **Bug ladder cleared** (all committed/pushed):
  1–8. (original: max_sampling_loras, EP=1, fold OOM, lora_path null, idempotent
  registration, abort wedge, MTP guard, nccl→p2p — see git history).
  9. **Serving-corruption exposure** — the parent adapter carried `linear_attn.*`
  + `shared_expert.down_proj` LoRA the scorers never applied (Muon folded that
  noise into the base). Fix: MoE-only targets in the **PS server config**
  (`configs/zorl/…ps_muon.yaml`, infra `673b941`) — NOT create_model (the
  `LoRAConfigRequest` extras are schema-discarded) — plus sglang cherry-picks
  `f26350543`+`c45f64fff` on `zorl-ps-fp32`. Step-0 gate: grep the PS server.log
  missing-keys for `linear_attn` (must be absent).
  10. Scorers had **no `/dev/infiniband`** — grant `rdma/infiniband` + IPC_LOCK
  (infra `004b1c1`).
  11. Duplicate **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (deprecated
  name) overrode the launch export → VMM segments → `Bad address [14]` on RDMA
  register. Both env names must be `:False` (`004b1c1`).
  12. **GDN reshard gap**: `linear_attn.A_log` (32,)→(16,) TP1→TP2 has no locator;
  skip it (`XORL_WEIGHT_SYNC_SKIP_PARAM_PATTERNS=linear_attn`) — never folded
  under MoE-only, so correct.
  13. **p2p status-gather deadlock** (the old "hang at rendezvous"): py-spy showed
  the whole model transfers, then all sender ranks hang in
  `_gather_p2p_transfer_statuses`→`all_gather_object` (NCCL PG unusable after
  out-of-band RDMA). Fix: gloo group, `XORL_P2P_STATUS_GATHER_BACKEND=gloo`
  (`zorl-ps cfb4b26c9`). This UNMASKED blocker #14 (turned 30-min hangs into
  instant clear errors).
  14. Warm-cache `prepare_weights_update` 400 on reused scorers → **full teardown**
  (STS+Services+SMG, wait all gone), not pod-bounce.
- **PROVEN end-to-end:** EP=1 PS load, registration, seed transport, fleet
  scoring, honest cold floor ≈ 0.008–0.03, the **fresh_ab fold** (160 mods /
  120 base params / ~15 s, MoE-only), AND the **full 35B RDMA transfer**
  (root 1017 MB + 809 MB/1546-param bucket, py-spy-confirmed complete).

### ⛔ THE SINGULAR REMAINING BLOCKER — expert weight-name fusion mismatch
Identical for BOTH p2p and sparse_delta (so one fix unblocks everything). The
base-weight sync EMITS per-expert **unfused** names
`model.layers.N.mlp.experts.{i}.gate_proj.weight` (`xorl-zorl-ps`
`handler.py:2894`, the EP-gather bucket path), but the FP8-FusedMoE receiver's
tensor-map (`_build_hf_tensor_map_locators`/`_emit_fused_moe_locators` in sglang
`model_executor/model_runner.py`) keys experts by **fused per-layer**
`model.layers.N.mlp.experts.gate_up_proj` (source `[E, 2I, H]`, gate rows
`[0:I]`, up `[I:2I]`) + `experts.down_proj` `[E, H, I]` — no expert index. So
the pushed tensors match no locator → p2p `receiver tensor_map is incomplete`,
sparse_delta `did not match a local parameter or any local tensor-map locator`.
**Refined root-cause (2026-07-03, static analysis exhausted):** the sender emits
`model.layers.0.mlp.experts.{i}.gate_proj.weight`; the receiver returns "no
receiver locator" for it. Ruled OUT: `language_model` prefix (sglang strips it,
qwen3_5.py:1158); isinstance gate (served class is `DeepEPMoE(FusedMoE)` —
subclass, passes); the FP8 skip-warnings (none fired). The receiver's
`_emit_fused_moe_locators` is *designed* to emit exactly that per-expert name
(model_runner.py ~4383) — so the mismatch is a subtle one that needs a **live
locator-name dump** to pin: likely the block-scale early-return
(`w13.element_size()<2 and not block_scale_locators` → the FP8 scale layout
isn't the expected 3D block → whole expert emission skipped silently — CHECK
THIS FIRST, it's the leading suspect), or a `module_name`/index-range subtlety.
**Exact next diagnostic:** add a `logger.info` in `_emit_fused_moe_locators`
right after the `w13/block_scale` checks dumping `module_name`, whether it
returned early, and the first per-expert `hf_name`; restart ONE scorer; run a
4-replica debug sync; read the log. THEN fix (options below) with a
`validate_only` dry-run + a logprob-delta correctness check before trusting the
scatter — FP8 expert slices have no ES-side k3 gate, so a wrong mapping is
silent corruption.
**Fix options** (one unblocks BOTH transports): (A, sender) fuse gate+up per
layer into `experts.gate_up_proj` `[E,2I,H]` + fused `weight_scale_inv`
(fused name exists at model_runner.py:1071); (B, receiver) fix whatever the
dump reveals (un-skip the block-scale path for this FP8 scale layout, or add
per-expert locators). `lora_export_format` does NOT help (on-disk adapter
export only, not the sync path).

### ⚠️ Cluster caveat
The PS pod was killed by a clean external SIGTERM ("Normal Killing", NOT OOM) =
preemption/node-reclaim; the manifest is `priorityClassName: normal` on a
contended cluster. Even a working sync may be fragile — consider a higher
priority class or fewer concurrent stacks.

- **Gate (unchanged):** held-out solve-rate (probe every 10 steps, seed-777
  NG=128, retries=0) climbing toward GRPO's 0.65–0.68 honest ceiling.
- **To resume:** `kubectl scale sts -n apanda zorl-w35-sglang --replicas=32`
  (or 2 to isolate), recreate SMG (`zorl-w35-smg-router.yaml`), then
  `kubectl create -f k8s/zorl/qwen3_6-35b-a3b-zorl-wordle-ps-trainer-grpo-match.yaml`.

## Operating rules (learned the hard way)

- **Match GRPO, don't improvise** — any prompt/reward/eval/capacity/lr deviation
  turns a negative into an uninterpretable confound (`zorl-grpo-match-directive`).
- **Bulk weight moves via Mooncake p2p**; HTTP only for seeds/rewards (`mooncake-p2p-preference`).
- **Never touch other streams' workloads**; append status to
  `/shared/apanda/wordle-coord/messages.jsonl`.
- **Full teardown, not pod-bounce**, when relaunching onto reused scorers
  (warm-cache p2p hang). Scorer StatefulSet is `OnDelete` — pods must be deleted
  to pick up new code/spec.
- `think`-style prompts plateaued ~1.6% on the capacity-capped frozen-base arm
  (`zorl-wordle-think-budget-independent`); fresh_ab-at-GRPO-lr is the test of
  whether *capacity* was the wall. Keep `think` to match GRPO.
