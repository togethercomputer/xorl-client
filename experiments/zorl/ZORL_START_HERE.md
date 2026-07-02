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
- **Status (2026-07-03): the expert weight-name blocker is ROOT-CAUSED and
  FIXED in code** (sglang `zorl-ps-fp32` commit `62892dc26`, pushed);
  function-level validated. Remaining: restart the 4 scorers to pick up the
  fix + run the debug sync cycle (see the resolved-blocker section below —
  the agent session was permission-blocked from `kubectl delete pod`).
  4 debug scorers UP but running PRE-fix code; trainer killed. ~11 more bugs
  cleared past the original 8 — the fresh_ab fold works end-to-end AND the
  full 35B model transfers over Mooncake RDMA. **Debug at 4 replicas**
  (`SGLANG_REPLICAS=4`, `NUM_PAIRS=8`, `TRAIN_SIZE=8`, `STEPS=4`) — cheap
  ~15-min cycle to the sync.
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

### ✅ RESOLVED (2026-07-03) — expert weight-name blocker was the LoRA-wrapper hop
**Root cause:** the scorers serve MoE-expert LoRA (`--lora-target-modules
gate_proj up_proj down_proj`), so `get_lora_layer` (isinstance match — catches
`DeepEPMoE` too) wraps every `FusedMoE` in `FusedMoEWithLoRA` and the real
module sits at `model.layers.N.mlp.experts.base_layer`. The receiver's
`_emit_fused_moe_locators` built hf_names from the RAW `module_name`, keying
every expert locator `...experts.base_layer.{i}.gate_proj.weight[...]` while
the sender pushes `...experts.{i}.gate_proj.weight` → zero matches on all 1540
expert tensors (run l9lrt 19:59: `receiver tensor_map is incomplete ...
'model.layers.0.mlp.experts.0.gate_proj.weight': no receiver locator; … 1535
more`). NOT the block-scale early-return (no skip-warnings because emission ran
to completion — just under wrong names). `8e44c42cf` stripped `.base_layer.`
for every dense branch via `_hf_name`; the fused-MoE branch was the one place
that bypassed it. The earlier "receiver wants fused gate_up_proj" framing was
wrong — the receiver emits BOTH fused and per-expert names; both were polluted.
**Fix:** sglang `zorl-ps-fp32` commit `62892dc26` (pushed) — strip the
`.base_layer` suffix/hop into `hf_module_name` for all fused + per-expert +
scale hf_names (raw `module_name` kept for `consumed_param_names`), plus an
INFO line per FusedMoE dumping the resolved hf prefix + first per-expert name.
**Validated** by executing the emit function (mock FP8 module, real code):
pre-fix = 18/18 locators polluted at `...experts.base_layer`; post-fix = exact
sender names (weights + `weight_scale_inv` + fused), wrapped ≡ unwrapped.
**Remaining (operator step — agent was permission-blocked from pod deletes):**
1. `kubectl delete pod -n apanda zorl-w35-sglang-0 zorl-w35-sglang-1
   zorl-w35-sglang-2 zorl-w35-sglang-3` — STS is OnDelete; fresh pods pick the
   fixed code off the PVC. Wait 4/4 Ready + `/health` 200.
2. `kubectl create -f
   ~/xorl-infra/k8s/zorl/generated/zorl-w35-grpomatch-debug4-20260703.yaml`
   (pre-patched debug knobs: SGLANG_REPLICAS=4, NUM_PAIRS=8, TRAIN_SIZE=8,
   STEPS=4).
3. Gate the sync: scorer logs must show `[P2P tensor_map] FusedMoE ... hf
   prefix 'model.layers.N.mlp.experts'` (no `base_layer`), trainer server.log
   no `tensor_map is incomplete`. If prepare 400s/wedges on the reused-fleet
   state → full teardown per rule below, recreate at 4, retry.
4. Correctness gate before trusting the scatter (FP8 expert slices have no
   ES-side k3 gate): receiver content sniff on an expert tensor, post-sync
   `/generate` sanity, honest floor ≈ 0.008–0.03 (not gibberish/zero).

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
