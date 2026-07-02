# Wordle GRPO — Canonical Recipe (START HERE)

**Last updated: 2026-07-01 (apanda).** Single entry point for the working Wordle RL line on
**Qwen3.6-35B-A3B**. This is a *map + recipe summary*; the deep detail lives in the linked docs.
If something here conflicts with an older doc, this wins (and the dated UPDATEs at the top of the
canonical runbook win over its older body).

---

## TL;DR — what works

Full-weight **server-mode GRPO from base**, **unclipped importance-sampling** loss, on-policy, with a
verifiable **retrieval reward**, on a single-node **EP8** trainer feeding **TP2 SGLang samplers** behind an
**SMG** router. From a **0.00** base floor it reaches **~0.65–0.68 held-out solve rate** (honest,
`retries=0`). Driving live train/inference logprob mismatch (**k3**) from ~0.055 → ~3e-4 is *correctness
hygiene, not a performance lever* — it does not change final policy quality.

### The result (honest held-out, `retries=0`, seed-777, NG=128)
| run | k3 regime | src | held-out val acc |
|---|---|---|---|
| 9dxtb | high 3.6e-3 | apanda-dev | 0.672 (86/128) |
| k3diag | low 3e-4 | k3-recon | 0.648 (83/128) |
| k3bi | BI, lowest 2.6e-4 | k3-recon | 0.680 (87/128) |
| base+think | — | — | **0.00** |

All three **tied ~0.65–0.68** (≈0.5σ), no monotonic relation to k3 → **k3 is a diagnostic, not a lever.**
Held-out ≈ in-training at `retries=0` (the old "in-training 0.05 vs held-out 0.55" gap was the retry
crutch). See `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` (2026-07-01 UPDATE) +
memory `wordle-eval-pipeline-and-k3-not-a-lever`.

---

## The recipe (what to actually set)

**1. Source provenance (THE crux — get wrong and live k3 sits at ~0.003).**
Trainer imports engine from the validated **k3-reconciliation** tree, borrows the venv only for compiled deps:
```
XORL_SRC=/home/apanda/xorl-qwen-k3-reconciliation/src   # PYTHONPATH ahead of the venv's editable xorl
XORL_VENV_REPO=/home/apanda/xorl-internal               # .venv = prebuilt deep_ep/mooncake/FA3
XORL_CLIENT_REPO=/home/apanda/xorl-client-wordle-science-20260614
```
Preflight asserts `xorl.__file__` resolves under `xorl-qwen-k3-reconciliation/src`. Detail: `HANDOFF/HANDOFF.md` §1.

**2. Model + engine config.**
- Model (resolved snapshot dir, required so all ranks direct-load): `…/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
- Config: `/shared/apanda/wordle-sft-runs/configs/grpo-ep8x1node-muon-lowlr-isr3k3.yaml` — `expert_parallel_size 8`,
  `tensor_parallel_size 1`, `moe_implementation triton`, `optimizer muon` (bf16). EP8 works; **EP4 OOMs/hangs**;
  adamw fp32 state OOMs EP8. No `lr_warmup`/cosine in the YAML (crashes engine scheduler — LR is client-side).

**3. The GRPO loop (client — `experiments/wordle/standalone/train_grpo_wordle.py`).** Canonical flags:
- `--loss-fn importance_sampling` (**unclipped IS — the lever**; `policy_loss`+clip throttles from-base acquisition), on-policy (no `--pipeline-rl` for from-base).
- `--reward-key wordle_retrieval_reward` (verifies each guess vs the answer; in `tasks/wordle.py`).
- `--wordle-prompt-style public_reasoning_constraints_think --wordle-scaffold-fade-steps 0`.
- `--group-size 16 --train-size 32 --train-pool-size 4096 --steps 128`, `--student-temperature 0.7`.
- Low-k3 numerics (correctness): `--logprob-temperature 0.7` (matches rollout temp), `--return-routed-experts --return-expert-logits` (R3 decode-route IDs + float weights), `--compute-kl-stats`.
- `--optimizer muon --optimizer-dtype bf16 --lr 5e-6 --lr-schedule cosine --lr-warmup-steps 8`.
Full invocation + wiring: `HANDOFF/HANDOFF.md` §3.

**4. Infra topology (a "stack" = 3 disjoint k8s workloads: samplers → SMG → trainer).**
Single-node EP8 trainer (one whole 8-GPU node) + N×TP2 SGLang samplers behind an SMG router. Generation
routes through SMG (`--infer-url`); p2p weight-sync goes **direct** to each sampler (`--sampler-load-url ×N`).
Builders: `build_k3diag.py` (no-BI, faster) / `build_k3bi.py` (batch-invariant, lowest k3, ~1.4× slower).
**One sampler pool per trainer, never shared.** Stand-up + the 8 gotchas + templates: `HANDOFF/HANDOFF.md`.

---

## Components — how to use & launch each

A **stack** wires 4 components (+ launch orchestration). Each row → its files, how it's launched/invoked,
and the deep doc. Generation flows **client → SMG → samplers**; p2p weight-sync goes **client → each
sampler directly** (`--sampler-load-url ×N`).

| component | role | key files | launch / invoke | deep doc |
|---|---|---|---|---|
| **Engine** (xorl server) | full-weight **EP8 trainer server** — fwd/bwd, optimizer, weight-sync backend | config `configs/grpo-ep8x1node-muon-lowlr-isr3k3.yaml`; source pinned via `XORL_SRC` (§1 above) | `python -m xorl.server.launcher --mode auto --config <cfg> --nnodes 1 --master-addr $POD_IP --api-port 26070 …` — runs **inside the trainer pod's HEAD_CMD** (emitted by `build_<name>.py`). For **eval serving** it's the EP8 sync job stood up by `prepare_checkpoint_eval_serving.py`. | `HANDOFF/HANDOFF.md` §1–2 |
| **Client** (GRPO loop) | drives rollouts + loss + weight-sync; owns all RL flags | `standalone/train_grpo_wordle.py` (+ `train_opsd_baseline.py` helpers, `tasks/wordle.py` task+reward) | invoked in the trainer HEAD_CMD *after* the engine is ready; canonical invocation + wiring in HANDOFF §3 | `HANDOFF/HANDOFF.md` §3 |
| **SGLang samplers** | TP2 inference workers — rollout generation **and** p2p weight-sync receivers | `launch/<name>-sampler.yaml` (StatefulSet; the SGLang launch flags/parity live here) | `kubectl apply -f launch/<name>-sampler.yaml` → wait all N `/health`=200 | `HANDOFF/HANDOFF.md` §4 (sampler parity flags) |
| **SMG router** | load-balances generation across samplers; patched to accept per-row `sampling_params` | `launch/<name>-smg.yaml` + `<name>-smg-svc.yaml`; binary `SMG_BIN=/shared/apanda/wordle-sft-runs/smg-isr3k3-bin` | `kubectl apply` both → wait `/v1/models`=200; **set `WORKER_URLS` explicitly** (gotcha #3) | `HANDOFF/HANDOFF.md` §4; memory `smg-perrow-sampling-params` |
| **Launch orchestration** | generates + applies the stack in the right order | builder `build_<name>.py` (→ writes `launch/<name>-trainer.yaml`); `STACK_launch.template.sh` / `STACK_rebuild.template.sh` | **gated sequence: samplers → SMG → trainer**, watch step-0 sync; clone a stack per §6 | `HANDOFF/HANDOFF.md` §5–6 |

## How to run / evaluate

- **Stand up a stack from scratch:** `/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md` (+ `STACK_launch.template.sh`,
  `STACK_rebuild.template.sh`). Gated launch: samplers → SMG → trainer, watch step-0 sync.
- **Honest held-out eval (the one true gate):** `eval_ckpt_generic.sh <CKPT> <LABEL> <CFG> [NG] [R]` — serves the DCP
  checkpoint (private TP2 sglang + EP8 weight-sync job) and runs `shard_eval.py` on seed-777 held-out at
  **`EVAL_INVALID_RETRIES=0`** (base+think = 0.00; the old retries=2 crutch inflated every historical number).
  For big-N / multi-try paired evals: `eval_ckpt_multi.sh` (syncs once, runs many shard_evals) + `run_bigeval_serial.sh`.
  **Use NG≥128** — NG=64 is too noisy (temp-0.7 run-to-run variance).

---

## Gotchas that cost real time (each has a pointer)

- **Eval retry-crutch** — `--invalid-retries 2` silently re-prompted no-think → inflated every held-out number; the honest gate is `retries=0`. → memory `wordle-eval-retry-crutch`.
- **`_think` format bug** — a parser bug counted template-echo as the guess tag, killing ~91% of turns; confounded early "muon divergence" conclusions. → memory `wordle-think-format-bug`.
- **Eval-pipeline `engine_connect_host` bug (fixed 2026-07-01)** — training configs' `engine_connect_host` bled into the eval launcher → ZMQ `:5556` vs worker `:25670` mismatch → every honest eval dead 06-26→06-30. Fix: strip it in `prepare_checkpoint_eval_serving.py`. → memory `wordle-eval-pipeline-and-k3-not-a-lever`.
- **Warm-cache relaunch hang** — relaunching a trainer onto reused/crashed samplers wedges step-0 p2p (`Failed to initialize p2p backend`); do a FULL teardown, not a pod-bounce. → memory `sampler-p2p-warmcache-relaunch-hang`, `HANDOFF/HANDOFF.md` gotcha #4.
- **Capacity walls** — a 4th concurrent stack hits (a) whole-node fragmentation (EP8 trainer needs a *contiguous* 8-GPU node) and (b) RDMA pinned-memory `-202` (Mooncake receiver registration fails under cluster-wide RDMA saturation). ~3 stacks fit.
- **Flaky free nodes** — the cluster's *free* 8-GPU nodes are often free *because* they're faulty (NVLink `Invalid peer access` on EP8 init, or a physically-7-GPU node). The eval sync job blacklists known-bad nodes in `prepare_checkpoint_eval_serving.py`.
- **Eval checkpoint path must be ABSOLUTE** — a relative `--ckpt` arg lands in the config as-is (`prepare.py` can't map it) → sync pod can't find it → `AssertionError: metadata is None`.
- **Rollout throughput** — `--student-generation-workers` is a no-op above ~32 (turn-1 dispatches all 512 seqs at `min(512/16, 48)=32`; the default 16 was the real cap, already uncapped at 48). The real bottleneck is the **turn-synchronous barrier + `max_new_tokens` straggler tail + active-drain**; real levers are **pipeline-RL** and **trimming `max_new_tokens`**. → runbook §D, `HANDOFF/HANDOFF.md` #9.
- Pipeline-RL requires `--no-weight-sync-flush-cache`; `flash_attention_deterministic` crashes hdim-256; `kubectl apply` resets sampler `replicas`; bounce SMG when you bounce samplers. → `THROUGHPUT_DEBUGGING_HANDOFF.md`, `HANDOFF/HANDOFF.md`.

---

## Doc map

| doc | what it covers |
|---|---|
| **THIS FILE** | canonical recipe summary + entry point |
| `OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md` | science runbook — **read the dated UPDATEs at the top first** (07-01 = honest numbers + k3-not-a-lever; 06-30 = live-k3 solved; 06-26 = retry-crutch + IS-is-the-lever) |
| `/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md` | **infra reproduction** — stand up / rebuild / health-check a stack; source pin; builders; 4 manifests/stack; 8 gotchas; templates |
| `/shared/apanda/wordle-sft-runs/HANDOFF/UPSTREAMING_SCOPE.md` | **code-change map** — every engine (train+inference) / client / SGLang / orchestration-script update → its apanda-dev branch + file list (for auditing/upstreaming). Read its 2026-07-02 UPDATE. |
| `NEXT_AGENT_START_2026_07_02.md` | **latest handoff** — code-audit map pointer + science summary + the in-flight k3pnr3-v2 destabilization test |
| `THROUGHPUT_DEBUGGING_HANDOFF.md` | throughput/MFU narrative; the rollout-was-serial fix; multi-sampler + SMG; pipeline-RL |
| `SGLANG_XORL_PARITY.md` | SGLang↔xorl logprob parity deep-dive (k3 root-causing) |
| `NEXT_AGENT_START_2026_07_01.md` | latest state + next levers (policy-quality frontier, capture-the-peak, deferred pipeline-RL) |
| `K3_REPLAY_HANDOFF_2026_06_28.md` | k3 replay test ladder (historical) |
| `HANDOFF/UPSTREAMING_SCOPE.md` | which fixes go to which apanda-dev branches |
| `WORDLE_MIGRATION_MANIFEST.md` | how this experiment tree was migrated |

**Key memories:** `wordle-eval-pipeline-and-k3-not-a-lever` (result + eval fix), `wordle-live-k3-floor`
(how k3 was driven low), `wordle-eval-retry-crutch`, `wordle-think-format-bug`, `wordle-grpo-the-path`,
`wordle-pipeline-rl-throughput`, `sampler-p2p-warmcache-relaunch-hang`, `smg-perrow-sampling-params`.

---

## Open / next levers
The honest ceiling is ~0.67–0.68. Behavioral panels show residual constraint-violations (~0.34–0.44) and
non-dictionary guesses (~15–20%). Levers: reward shaping (penalize violations / non-dict guesses), a
stabilizer to hold the in-training peak (~0.78–0.80 @ ~s116–120) instead of the mild decline to s128,
SFT warmstart, or **capture the true peak** (re-run with `save_interval 25` — the peak checkpoints
weren't saved). Pipeline-RL (throughput) is set up (`k3pipe`) but deferred on cluster capacity. Details in
`NEXT_AGENT_START_2026_07_01.md`.
