# Wordle Science/Infra — Next Agent Start (2026-07-02)

> **✅ CONSOLIDATION EXECUTED (2026-07-02, later the same day):** the code-audit/upstreaming work
> described below is DONE. See the status table at the top of `UPSTREAMING_SCOPE.md` (versioned copy
> in this directory). Summary: SGLang wordle patches pushed on `apanda-dev` (`e9593887f`); client
> core change was already upstream (`bcd037d`); science branch pushed to `internal` with docs + river
> port + eval orchestration tooling; engine parity work is **PR #430** on togethercomputer/xorl-internal
> (R3 transport kept as upstream's `2093a6c82` side_payloads version — it supersedes the k3-recon-line
> externalize implementation); infra k3* builders/manifests/configs on `xorl-infra` branch
> `wordle-consolidation`; SMG source found+pushed (`together-smg` branch `xorl` @ `a83f1a6f`).

Supersedes `NEXT_AGENT_START_2026_07_01.md`. This handoff is oriented for **auditing the code/script
updates** (engine training+inference, client, orchestration) plus the science + one in-flight run.

## ⭐ If you're auditing the code updates — start here
The **code-change map is `/shared/apanda/wordle-sft-runs/HANDOFF/UPSTREAMING_SCOPE.md`** (read its
2026-07-02 UPDATE first). It enumerates every change by area, with file lists + upstream targets:
- **§1 SGLang** (`xorl-sglang-internal` apanda-dev): 11 files (~146 ins) — `return_expert_logits` in OpenAI
  protocol, selective batch-invariant ops, weight-version request routing.
- **§2 Client** (`xorl-client-internal` apanda-dev): 1 core file — `chunked_helpers.py` (+23, R3 payload byte
  estimate). Everything under `experiments/wordle/` (incl. `train_grpo_wordle.py` +597 with the GRPO loop,
  `--pipeline-rl`/`--logprob-temperature`/R3 flags + the ThreadPoolExecutor rollout parallelism) stays on the
  science branch.
- **§3 Engine (train+inference)** (`xorl-internal` apanda-dev): the big one. ⚠️ HEAD advanced `98eb289e →
  be40a275c` — the `src/` diff GREW (dense-K3 work added 25 files/+3,442). Re-scope vs current HEAD:
  `git -C ~/xorl-qwen-k3-reconciliation diff origin/apanda-dev..be40a275c -- src/`. Groups: batch-invariant
  ops, MoE experts/block, R3 routing-replay + `model_runner`/`runner_dispatcher`, losses (IS/CE/policy),
  attention/rope/norm, server args/launcher.
- **§4 Orchestration scripts** (`/shared/apanda/wordle-sft-runs`, NEW section): builders
  (`build_k3{diag,bi,pipe,pnr3}.py`), eval tooling (`prepare_checkpoint_eval_serving.py`,
  `eval_ckpt_{generic,multi}.sh`, `run_bigeval_serial.sh`, `shard_eval.py`), manifests. **Contains one real
  BUG FIX** (`engine_connect_host` strip in prepare.py — see below).

## Science conclusions (all recorded; see `WORDLE_RECIPE.md` + runbook §1/07-01)
- **k3 is a diagnostic, not a lever.** Honest held-out (retries=0, NG=128×5 tries): high-k3 9dxtb **0.712**,
  low-k3 k3diag **0.686**, lowest-k3 k3bi **0.704** — all tied ~0.70 (vs base 0.00), non-monotonic in k3.
  Held-out ≈ in-training at retries=0; small train-vs-held-out gap (real generalization).
- **Pipeline-RL:** reaches the **same ~0.80 in-training peak as on-policy** (k3pipe 0.805@s120) → 1-step
  staleness costs no quality/acquisition; **modest ~10-25% throughput win** (not the runbook's old 2×; rollout
  ≫ train so overlap hides only train). Peeling R3 is ~100-140s/step faster (transport artifact; a Mooncake-R3
  transport PR on apanda-dev should recover it).
- **Throughput bottleneck** is the **turn-synchronous rollout barrier + straggler tail + active-drain**, NOT
  `--student-generation-workers` (a no-op above ~32; turn-1 already dispatches all 512). Levers: pipeline-RL,
  trim `max_new_tokens`. (Correction recorded in runbook §D / HANDOFF #9.)
- **Eval-pipeline bug (fixed):** `engine_connect_host` bled from training configs into the eval launcher →
  ZMQ `:5556` vs `:25670` wedge that killed every honest eval 06-26→06-30. Fix = strip it in prepare.py.

## 🔴 IN-FLIGHT — k3pnr3-v2 (the one open question)
Testing whether **pipeline staleness on a genuinely high-k3 base destabilizes**. Run 1 (`k3pnr3`) climbed
healthily to peak 0.5@s38 (k3 rising 2.6e-3→7e-3, ratio_max spiking to 43.9) then **died at s39 on a flaky
sampler** (smp-0 crash-loop on node 040) — an INFRA crash, so the "does it destabilize past s39" question was
UNTESTED (my earlier "no destabilization" was overstated — corrected in EVAL_RESULTS). **v2 relaunch is live**
(`GRPO-WQ36-K3PNR3-noR3-highk3-pipelinerl-v2`, wandb, `server_output_k3pnr3v2`): same R3-peeled high-k3 recipe,
`save-interval 25` (peaks captured + resumable), samplers steered off node 040. **Watch it climb to ~s128 —
does the rising k3 / wide ratio_max eventually crater it, or climb fine like the low-k3 pipeline?** Health:
`grep '\[train step=' <run_dir>/trainer_job.log` + k3/ratio_max from metrics.jsonl.

## Doc map
`WORDLE_RECIPE.md` (entry point) · `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` (science log, read dated
UPDATEs top-down) · `HANDOFF/HANDOFF.md` (stand-up-a-stack) · `HANDOFF/UPSTREAMING_SCOPE.md` (**code-change
map**) · `THROUGHPUT_DEBUGGING_HANDOFF.md` · `SGLANG_XORL_PARITY.md`. Memories: `wordle-eval-pipeline-and-k3-not-a-lever`,
`wordle-pipeline-rl-throughput`, `wordle-live-k3-floor`, `wordle-upstreaming-scope`.

## Hard rules
Outputs only under `/shared/apanda/wordle-sft-runs`; one sampler pool per trainer; don't touch other
experiments (`er-opd-q235*`, `q36mtp*`, `marin6279*`, `zorl-*`, `wordle-k3lora-*` = co-active agent); don't
overwrite `smg-isr3k3-bin`; delete only pods/pools you created. Node health is transient — avoid a bad node
reactively, don't hardcode blacklists.
