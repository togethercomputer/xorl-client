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
- **Bug ladder cleared so far** (each found and fixed live): stale `max_sampling_loras`
  config field → EP>1 unsupported for fresh_ab (PS runs EP=1) → fold host-OOM
  (now chunk-streamed, bit-exact) → driver registration idempotency → server-side
  `abort_zorl_generation` wedge (unload-missing-adapter left the session active).
- **Proven components:** EP=1 PS load, 32-receiver p2p registration, seed
  transport (no adapter export), fleet scoring, cold floor ≈ 0.008–0.023 (honest).
- **Still unproven (the remaining firsts):** the fresh_ab fold at 35B scale and
  the post-apply Mooncake fp8 base sync — watch the first `[step 1]` line's
  `update_norm` and `sync` result.
- **Gate:** held-out solve-rate (probe every 10 steps, seed-777 NG=128,
  retries=0) climbing toward GRPO's 0.65–0.68 honest ceiling.

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
