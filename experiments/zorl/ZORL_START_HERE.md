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
- **Status (2026-07-02): PAUSED for handoff. GPUs freed** (scorer StatefulSet
  scaled to 0, trainer + SMG deleted). Eight launch attempts; a bug ladder was
  cleared, and everything up to and INCLUDING the fold works. One blocker remains.
- **Bug ladder cleared** (each found + fixed live, all committed/pushed):
  1. stale `max_sampling_loras` config field (dropped from PS config)
  2. fresh_ab rejects EP>1 → PS runs `expert_parallel_size: 1` (FSDP-only; PS does no fwd/bwd)
  3. fold host-OOM ~96 GB/rank → chunk-streamed GPU fold, bit-exact (`xorl-zorl-ps` `0d384d676`)
  4. `lora_path: null` 400s the cold probe → omit key / no `str(None)` (client)
  5. driver seed-registration not idempotent → retry + unload-reload (`1769898`)
  6. server `abort_zorl_generation` wedge (unload-missing-adapter left session active) → best-effort (`sglang 91862e50f`)
  7. MTP guard blocked fp8 sync → scoped MTP-excluded escape (`xorl-zorl-ps 399d8fe62`)
  8. sync used `nccl_broadcast` (68-rank collective timeout) → `sync_inference_method: p2p`
- **PROVEN end-to-end:** EP=1 PS load, 32-receiver registration, seed transport
  (no adapter export), fleet scoring, honest cold floor ≈ 0.00–0.02, and — the
  hardest — the **fresh_ab fold at 35B**: 320 modules / 280 base params / 40
  chunks / `update_norm≈9.12e4` / ~28 s, reproduced across 3 attempts.

### ⛔ THE ONE REMAINING BLOCKER — p2p fp8 weight-sync hangs at rendezvous
After the fold, the post-apply sync (`_sync_inference_weights_after_zorl_apply`
→ `sync_inference_weights`, `sync_method=p2p`) to the **32 block-FP8 TP2 scorers**
**hangs at init**: `handler.py:751` logs `sync_method=p2p, endpoints=32,
quantization=fp8` at T0 and then transfers **nothing** for ~21 min (no
`prepare_weights_update`, no bucket transfer, no per-endpoint success) until the
pod is killed. This transport is UNEXERCISED: GRPO's proven p2p goes trainer→
**bf16** samplers at ~7 replicas; **trainer-side-fp8-quantize → 32 fp8 receivers
over Mooncake** is new. Debug path (use the `debug-distributed-hang` skill):
py-spy the hung PS sender (which rank/collective it blocks in), inspect each
receiver's `/prepare_weights_update`, verify Mooncake/NIXL session init +
`--enable-rdma-weight-updates` on the receivers, and the FSDP-4 → 32×TP2 (64-rank)
reshard plan. **Recommended first move: isolate to 1–2 replicas** (`SGLANG_REPLICAS=2`)
to make rendezvous fast and py-spy-able before debugging at full fan-out.
Alternatives if p2p proves intractable: (a) the in-tree `delta_packed_v1`
sparse-delta receiver the scorers already have (the deprecated sglang-PS path
used it — proven receiver contract, slower); (b) bf16 dense p2p if the fp8-quantize
step is the hang.

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
