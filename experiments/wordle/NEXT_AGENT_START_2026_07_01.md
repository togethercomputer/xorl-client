# Wordle Science — Next Agent Start (2026-07-01)

The GRPO full-weight RL line on Qwen3.6-35B-A3B is **scientifically wrapped up for the k3 question**.
All three overnight stacks finished and were evaluated on the honest gate. Your job is downstream:
pick a next lever (below), and optionally finish the deferred pipeline-RL throughput run.

## Read first
1. `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` — **2026-07-01 UPDATE at the top** (this session's results).
2. Memory `wordle-eval-pipeline-and-k3-not-a-lever` (the eval-pipeline fix + the k3-not-a-lever result +
   the NG=64 noise caveat). Plus `wordle-live-k3-floor`, `wordle-eval-retry-crutch`.
3. `/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md` — infra (stand up / rebuild / health-check).

## What was established this session (2026-07-01)
- **k3 is a DIAGNOSTIC, not a LEVER.** Honest retries=0 held-out (NG=128, seed-777): 9dxtb (high-k3
  3.6e-3) **0.672**, k3diag (low-k3 3e-4) **0.648**, k3bi (BI lowest-k3 2.6e-4) **0.680** — all tied
  ~0.65–0.68, no monotonic relation to k3. base+think floor = 0.00. The GRPO gain is large and real; the
  k3 regime doesn't move final policy quality.
- **Held-out ≈ in-training at retries=0** (no train/test gap when measured honestly — the old gap was the
  retry crutch). Use **NG≥128** for path comparisons (NG=64 is too noisy — temp-0.7 sampling variance).
- **Fixed the honest-eval pipeline** (broken 06-26→06-30): stripped `engine_connect_host` in
  `prepare_checkpoint_eval_serving.py` (was forcing an orchestrator↔worker ZMQ port mismatch).
- Evaluated checkpoints (DCP, EP8-sync-served): 9dxtb s100/s125/final, k3diag final, k3bi final. Raw eval
  logs in the session scratchpad; TAXONOMY/BEHAVIOR panels emitted per eval.

## Current cluster state
- **No live wordle trainers** — 9dxtb (`server_output_grpo-wq36-1n-mink3`), k3diag (`server_output_k3fix_v2`),
  k3bi (`server_output_k3bi`) all COMPLETED; final checkpoints on disk (196GB each).
- **Idle sampler pools remain** (trainers done): `wordle-k3bi-smp` (8×TP2=16 GPU) + `k3bi-smg`. (The
  sampler-b + k3diag pools were already freed this session.) Free these to return capacity — but confirm
  the exact targets (the safety classifier requires explicitly-named prior-session pods).
- Cluster is **RDMA-saturated** by other experiments (`marin6279-repro` 48 samplers, `er-opd-q235-fillerrft-slots`,
  `q36mtp`, `zorl-wordle-ps-35b`, + other users). Whole-node fragmentation is common.

## Your options (pick by interest)
1. **Deferred pipeline-RL throughput run** (`k3pipe`, fully set up). Only launches when cluster RDMA load
   drops (sampler Mooncake reg fails `-202` under current saturation; 6 attempts). Relaunch:
   `kubectl apply` k3pipe-sampler(8)→smg(+svc)→trainer (gated; samplers boot ~5min, watch step-0 sync).
   Measures pipeline step-time vs ~487s serial + whether 1-step staleness hurts the from-base climb.
2. **Push the policy-quality frontier** (the honest ceiling is ~0.67–0.68). The behavioral panels show
   residual `const_violation` ~0.34–0.44 and `non_dict` guesses ~15–20% — the reasoning still leaves
   derived-constraint violations. Levers: better reward shaping (penalize constraint violations /
   non-dictionary guesses), a stabilizer to hold the in-training peak (0.78–0.80) instead of the mild
   decline to 0.61–0.68 by s128 (early-stop / KL-to-ref / entropy floor), or SFT warmstart.
3. **Capture the true peak.** In-training peaked ~0.78–0.80 (~s116–120) but those checkpoints weren't saved
   (`save_interval 200`; 9dxtb's save-25 missed s120). Re-run one recipe with `save_interval 25` and eval the
   peak — likely the real held-out high-water is >0.68.

## Throughput (code-derived 2026-07-01, needs one measurement)
- **Don't touch `--student-generation-workers` (already fine at 48).** Verified in code: `_nw =
  min(len(batch_specs), workers)`; turn-1 = `512/16 = 32` batch_specs < 48 → all 512 seqs already dispatch;
  raising it is a no-op (the default 16 was the real cap, bumping to 48 uncapped it). See runbook §D + HANDOFF #9.
- **Real rollout bottleneck = turn-synchronous barrier + `max_new_tokens=2048` straggler tail + active-drain
  across turns.** Levers: **pipeline-RL** (option 1, hides barrier idle) and **trimming `max_new_tokens`**.
- **⚠️ QUICK-CHECK for the next training run:** the ~64/sampler turn-1 peak is *predicted from dispatch code, not
  measured* (prior "6–14 running-req" telemetry was eval/idle samplers, not training rollout). Over one training
  step, log sampler `#running-req`/KV-usage to confirm the peak-64→drain pattern before trusting it.

## Hard rules
- Outputs only under `/shared/apanda/wordle-sft-runs`. One sampler pool per trainer; restart samplers +
  bounce SMG before relaunching onto a reused pool (warm-cache p2p hang).
- Don't touch other experiments (`er-opd-q235*`, `q36mtp*`, `marin6279*`, `zorl-*`) or the shared `smg-isr3k3-bin`.
  Deleting any prior-session pool needs explicitly-named confirmation (safety classifier).
