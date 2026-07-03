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
- **Status (2026-07-03): the expert weight-name blocker is FIXED and
  LIVE-VALIDATED.** Two sglang `zorl-ps-fp32` commits (`62892dc26` routed
  experts, `cfd740741` shared_expert gate/up — both the same LoRA-wrapper
  `.base_layer` disease, see below). Debug run `bpn2s`
  (4 scorers, NUM_PAIRS=8, TRAIN_SIZE=8, STEPS=4): the post-fold sync now
  completes end-to-end — **34.7 GB / 61943 params / 43 buckets in ~202 s**
  (step-2 repeat sync 193 s, warm cache), receivers `prepare→complete` 200,
  zero locator errors, and the post-sync held-out probe sits at the honest
  cold floor (`parent_solve_rate=0.0137`, composite 0.397 ≈ cold 0.451) —
  the FP8 expert scatter lands intact. The full GRPO-match run is UNBLOCKED:
  scale scorers to 32 and launch per "To resume" below.
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

### ✅ RESOLVED + LIVE-VALIDATED (2026-07-03) — the LoRA-wrapper hop, twice
**Root cause:** the scorers serve MoE-expert LoRA (`--lora-target-modules
gate_proj up_proj down_proj`), so `get_lora_layer` (isinstance match — catches
`DeepEPMoE` too) wraps target modules in LoRA wrappers, nesting the real module
at `<name>.base_layer`. The receiver's locator builder derived HF names from
the RAW module path in two places:
1. `_emit_fused_moe_locators` keyed every routed-expert locator
   `...experts.base_layer.{i}.gate_proj.weight[...]` while the sender pushes
   `...experts.{i}.gate_proj.weight` → zero matches on all 1540 expert tensors
   (run l9lrt: `'model.layers.0.mlp.experts.0.gate_proj.weight': no receiver
   locator; … 1535 more`). Fixed in `62892dc26`.
2. The `MergedColumnParallelLinear` branch fed the wrapped path into
   `_guess_merged_subnames`, whose leaf is now `base_layer` → sub-names
   `base_layer_shard{0,1}` → `shared_expert.gate_proj/up_proj` (+scales) had
   no locator (run j6cft — surfaced only after fix 1 cleared the experts).
   Fixed in `cfd740741` (also hardens QKV/MergedRepeated/qkvz/conv1d paths).
NOT the block-scale early-return (no skip-warnings because emission ran to
completion — just under wrong names). `8e44c42cf` had stripped `.base_layer.`
for the plain-suffix branches via `_hf_name`; these two leaf-derivation sites
bypassed it. General lesson: when a receiver name-map misses names it was
"designed to emit", suspect wrapper-injected module-path hops and validate by
EXECUTING the emit path with the wrapped name.
**Live validation (debug run `bpn2s`, 4 scorers, NUM_PAIRS=8, TRAIN_SIZE=8,
STEPS=4):**
- Locator dump (added in `62892dc26`): `FusedMoE
  'model.layers.N.mlp.experts.base_layer' -> hf prefix
  'model.layers.N.mlp.experts'`, `block_scale_locators=True`, first per-expert
  name exact — the wrapper hop confirmed live and stripped.
- Post-fold sync: **34.7 GB / 61943 params / 43 buckets, 202 s** (repeat sync
  193 s on the warm cache); receivers `prepare→complete_weights_update` 200;
  zero `tensor_map is incomplete` / `no receiver locator` anywhere.
- Correctness: post-sync held-out probe `parent_solve_rate=0.0137` (honest
  cold floor 0.008–0.03), composite 0.397 ≈ cold 0.451; ES population sane
  (best 1.043 / mean 0.631); all 4 scorer logs error-free. A wrong FP8
  expert scatter would have collapsed all of these.
- Benign log noise to expect: boot-window ZMQ `Host unreachable` health
  retries, and endpoint-registration auto-sync rejected by the intentional
  MTP guard (the real sync path is `sync_inference_weights`).

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
