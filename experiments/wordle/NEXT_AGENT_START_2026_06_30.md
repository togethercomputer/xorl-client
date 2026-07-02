# Wordle Science — Next Agent Start (2026-06-30)

You're picking up a **working** GRPO full-weight RL line on Qwen3.6-35B-A3B. The hard problems are solved:
format bug (fixed), eval retry-crutch (fixed), **live on-policy k3 (~3e-4, was 0.055)**, and infra
(reproducible). Three stacks are climbing overnight. Your job is to take them to **honest held-out numbers**
and answer the open **does-low-k3-help-the-climb** question.

## Read first (in order)
1. `experiments/wordle/OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` — the **2026-06-30 UPDATE at the top** (full science state; it supersedes the older §4 k3 conclusion).
2. `/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md` — infra: how to stand up / rebuild / health-check a stack (paths, builders, the gated launch protocol, 8 gotchas). Templates beside it.
3. Memories: `wordle-live-k3-floor` (SOLVED), `wordle-pipeline-rl-throughput`, `sampler-p2p-warmcache-relaunch-hang`, `wordle-eval-retry-crutch`, `wordle-think-format-bug`.

## Current state (2026-06-30) — 3 disjoint single-node EP8 stacks, all IS-loss, climbing
| stack | trainer pod | wandb | src | k3 | progress |
|---|---|---|---|---|---|
| no-BI | `k3diag-head` | GRPO-WQ36-K3FIX-overnight-v2 | validated | ~3e-4 | exact 0.67 @ s73, rising |
| BI@8 | `k3bi-head` | GRPO-WQ36-K3BI-lowest | validated, batch-invariant, 8 samplers | ~2.6e-4 (lowest) | exact 0.19 @ s13, early |
| 9dxtb | `grpo-wq36-1n-isr3k3-head` | GRPO-WQ36-1node-ISR3K3-behavior-... | apanda-dev | ~3.6e-3 | peak exact 0.80 @ s120 |

Each owns its own samplers + SMG + trainer (fully disjoint). Health one-liner in HANDOFF.md §8.
`--save-interval 200` → no mid-run checkpoints land before step 200; **lower it or grab the peak step
manually** if you want the peak checkpoint (see task 1).

## Your tasks (ranked)
1. **Watch for the over-training peak/decline** (runbook §3). The IS climb peaks (~s75–120) then can regress
   (format/validity, while k3 stays flat). Note/checkpoint each run's peak — the **best ckpt is the peak, not
   the final**. Consider dropping `--save-interval` so the peak is captured.
2. **Honest held-out floor-eval at `EVAL_INVALID_RETRIES=0`** on the peak checkpoints. base+think clean = 0.00;
   the real GRPO gain is large (don't use the retries=2 crutch — it inflated every historical number).
3. **Open science Q — does low k3 help the climb?** Compare no-BI (3e-4) vs BI (2.6e-4) vs 9dxtb (3.6e-3) on the
   **held-out eval at matched/peak checkpoints** — NOT raw in-training numbers (not step-matched). Is the honest
   on-policy (low-k3) path better/more stable, or is the climb mostly loss/format-driven and k3 just a diagnostic?
4. **(Optional) Pipeline-RL** if capacity frees: `--pipeline-rl --no-weight-sync-flush-cache` (the flag is
   mandatory — see §D), scale samplers so rollout<train (train-bound). Capacity-bound now (3 stacks fill the
   cluster; a 4th hit whole-node + RDMA-memory walls this session).

## Hard rules (do not violate)
- Outputs only under `/shared/apanda/wordle-sft-runs`. **One sampler pool per trainer** (never shared).
- A trainer relaunch onto reused/crashed samplers can hang step-0 sync or fail p2p-init → **full teardown
  (delete StatefulSet+Deployment+Services, wait all pods gone), not a pod-bounce** (HANDOFF.md gotcha #4).
- Don't touch other experiments' stacks (er-opd-q235*, q3coder30b*, zorl-ar*) or overwrite the shared
  `smg-isr3k3-bin`. Only delete pods/jobs you create.

## Coordination
Coordination channel + promoted-config protocol are in the runbook's "Where things live + coordination"
section (`/shared/apanda/wordle-coord/` — `PROMOTED.json`, `messages.jsonl`). Check it if a throughput/k3
agent is co-active.
