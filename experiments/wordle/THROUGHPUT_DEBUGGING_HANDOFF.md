# Wordle trainer infra + throughput handoff (Qwen3.6-35B-A3B)

> **⏩ 2026-06-30:** the authoritative "stand up a stack from scratch" doc is now
> **`/shared/apanda/wordle-sft-runs/HANDOFF/HANDOFF.md`** (source provenance, model, config, builders, the
> 4 manifests/stack, the gated launch protocol + templates, and 8 hard-won gotchas — incl. the
> source-pinning that fixed live k3, `--no-weight-sync-flush-cache` for pipeline, full-teardown-not-bounce,
> sampler scaling 4→8, and the whole-node/RDMA capacity limits). Use that for reproduction; the sections
> below remain the deeper throughput/MFU narrative.

**Last rewritten: 2026-06-25 (apanda).** Full rewrite. The prior body was a deep diagnostic ledger of
the OPSD "why are we at 0.13% MFU" investigation (replay harnesses, bare-tensor A/Bs, dozens of dated
job IDs). That investigation is **closed and mostly superseded** — the workload moved from single-node
OPSD to **2-node p2p GRPO**, and much of the old "low MFU" headline turned out to be the trainer
*idling on a serial rollout* (and on format-death turns — see the science runbook), not a pure trainer
pathology. Git history preserves the old ledger (`git log -p
experiments/wordle/THROUGHPUT_DEBUGGING_HANDOFF.md`). This doc keeps the recipe that works and the
findings that are still true.

Stack: Qwen3.6-35B-A3B (40 layers, 3:1 GDN-linear-attn:full-attn, 256 experts / 8 active, hidden 2048,
vocab 248320). Full-weight **server-mode GRPO**, EP=8, p2p weight sync, multi-sampler generation behind SMG.

---

## The working recipe: 2-node p2p GRPO (2×8 H100)

- **Builder:** `/shared/apanda/wordle-sft-runs/build_grpo_wq36_2node.py` — emits a head Job + a
  headless head Service + a worker Pod.
- **Server config:** `/shared/apanda/wordle-sft-runs/configs/grpo-ep8x2node-muon-lowlr.yaml`
  (EP8 · dp_replicate2 · dp_shard8 = world 16; muon bf16 `muon_lr 5e-5`; `moe_implementation: triton`,
  `ep_dispatch: deepep`; `enable_packing: true`, `sample_packing_sequence_len: 8192`;
  `gradient_checkpointing_method: recompute_full_layer`; `sync_inference_method: p2p`;
  `engine_connect_host: 127.0.0.1`).
- **Topology launch:** head runs `xorl.server.launcher --nnodes 2 --master-addr $POD_IP`; the worker
  runs `torch.distributed.run … runner_dispatcher` and rendezvous to the head Service.
- **Source/venv split (critical):** put **`/home/apanda/xorl-apanda-dev/src`** on `PYTHONPATH`
  (p2p/Mooncake-capable) and borrow **`/home/apanda/xorl-internal/.venv`** for the compiled deps
  (deep_ep / mooncake / FA3). `xorl-internal` HEAD stripped p2p down to nccl_broadcast-only, so its
  *source* can't do p2p, but its *venv* has the right binaries. A startup preflight prints
  `xorl.__file__` to confirm the source is apanda-dev.

### The 8 bring-up bugs (fix in this order — each was a separate failed init)

1. Worker `runner_dispatcher` needs the **CONFIG positional** arg (not only `--model_path`).
2. Config must NOT carry non-`ServerArguments` fields — `enable_fp8_training` and
   `gradient_checkpointing_method` are **rejected by the xorl-internal ServerArguments schema** at
   engine init. (Set gradient checkpointing where the schema accepts it; do not leave the rejected
   keys in the server YAML.)
3. `model_path` MUST be the **resolved snapshot dir** (`…/snapshots/995ad96…`). An unresolved name
   deadlocks cross-node `_broadcast_object_list_weight_load`.
4. `engine_connect_host: 127.0.0.1` — without it, ranks wait 300s for the rank-0 address.
5. Client needs `--output-dir`, `--train-model-id default`, and `--model` == the server's base model.
6. Full-weight `create_model` must send `zorl_config: None` (not `{"enabled": false}` — it rejects a
   non-None zorl_config in full-weight mode).
7. `sync_inference_method: p2p` requires the apanda-dev source (see the split above).
8. Do NOT put `lr_warmup` / cosine in the server YAML — the server builds the optimizer + LR scheduler
   at engine init and crashes on them. **LR is scheduled client-side.** Also: muon must stay (EP4 OOMs/
   hangs the GRPO fwd/bwd; adamw fp32 optimizer state OOMs EP8).

### NCCL / IB

- Do **NOT** set `NCCL_IB_GID_INDEX` or `NCCL_IB_HCA=^mlx5_*` — they break the world process group.
  The node's bashrc already owns the IB env; let it. (Precedent: the working OPD 2-node teacher run.)
- DeepEP needs `expandable_segments` OFF (`PYTORCH_CUDA_ALLOC_CONF` must not enable it).

---

## THE real throughput bug: the rollout was serial

This was the actual fix, and it reframes the old "0.13% MFU" story. The within-turn batch loop
dispatched **one `_generate_batch_with_logprobs` at a time**, so the 7 samplers + SMG sat idle and
multi-sampler scaling did nothing.

**Fix:** a `ThreadPoolExecutor` over the within-turn batches (ported from the OPSD path), exposed as
`--student-generation-workers`. With it the samplers + SMG saturate → **rollout 1188s → 241s (~5×)**.

After the fix the run is **ROLLOUT-bound**, with a **warm forward/backward of ~65s**. Do NOT read MFU
off the cold step-1 fwd/bwd (~290s) — that's autotune/JIT warmup. Packing helps the warm fwd/bwd, but
`pack16384 + recompute_before_dispatch` OOMs the Triton MoE → use **`pack8192 + recompute_full_layer`**.

---

## Samplers + SMG

- **Samplers:** `wordle-sci-opsd-sampler-b` StatefulSet (TP=2, serving `Qwen/Qwen3.6-35B-A3B`). Scale
  replicas as needed. Replicas **1–7 are live; replica 0 is a crashed pod we intentionally skip**
  (`SAMPLER_INDICES=range(1,8)` in the builder).
- **SMG router:** `build_grpo_smg.py` → `wordle-grpo-smg` (clones the OPSD SMG Deployment+Service;
  `WORKER_URLS` = samplers 1–7; `--policy round_robin --backend sglang`).
- **Wiring:** generation goes through SMG via `--infer-url
  http://wordle-grpo-smg.apanda.svc.cluster.local:8080`; **p2p weight-sync goes DIRECT to each sampler**
  via `--sampler-load-url ×N` (the client auto-registers them). The multi-endpoint p2p sync is
  **parallel** (`XORL_P2P_MOONCAKE_WORKERS=8`, `XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=0`) — it is NOT
  serial. (`XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1` would force serial; it's README-only in apanda-dev.)
- **Sampler relaunch gotcha:** relaunching a full-weight trainer onto a *reused, warm-cache* sglang
  sampler hangs/fails the step-0 weight sync (`Failed to initialize p2p backend` — the Mooncake backend
  is bound to the dead trainer's session). **Restart the samplers** before relaunching the trainer.
  Relaunch sequence: (1) `kubectl rollout restart statefulset wordle-sci-opsd-sampler-b` (or re-apply
  the manifest) for a fresh p2p backend; (2) wait for 1–7 Ready; (3) launch the trainer.
  `podManagementPolicy=Parallel` → the restart bounces ~all pods at once (fast, but watch capacity).

### 2026-06-26 additions
- **🛑 manifest-`replicas` scale-down gotcha:** `kubectl apply -f launch/wordle-sci-opsd-sampler-b.yaml`
  **RESETS `spec.replicas` to whatever the file says.** The live pool had been manually scaled to 8, but
  the file said `replicas: 1` → an apply silently scaled it **8→1** and killed b-1..7 (only b-0 survived).
  **Keep the manifest's `replicas: 8` in sync** (now fixed in the file), or `kubectl scale … --replicas=8`
  right after an apply. Symptom: samplers "0/7 running" + only b-0 present.
- **🛑 Bounce the SMG whenever you bounce the samplers.** `wordle-grpo-smg` caches sampler backends;
  after a sampler restart it returns **503** on the stale endpoints, which silently **starves the trainer
  into NaN** (datums→0, loss=nan). Always `kubectl rollout restart deployment wordle-grpo-smg` after any
  sampler restart, and wait for it Ready before launching the trainer.
- **🛑 `flash_attention_deterministic: true` crashes the trainer at hdim-256** (Qwen3.6): "Deterministic
  backward not supported for hdim 256" at the first forward_backward. Keep it OFF in the trainer config.
- **Batch-invariant parity mode was TRIED then REVERTED** (see `SGLANG_XORL_PARITY.md` FINAL OUTCOME):
  `--rl-on-policy-target xorl-batch-invariant` (+ `--enable-return-routed-experts`) dropped the k3 *tail*
  (ratio_max 12→3) but NOT the *mean* (~0.055, GDN/FlashQLA-floored), AND the deterministic samplers
  **crash-loop** (exitCode 1 under load) → trainer NaN. Net: ineffective + destabilizing → the sampler
  manifest is reverted to the **stable flashinfer config** (no rl-on-policy-target, no return-routed-experts).
  k3→0 is a GDN-parity problem, not a sampler-flag one. Full knobs: **`SGLANG_XORL_PARITY.md`**.
- **2-node capacity wall → single-node EP8 fallback.** The shared cluster often can't place two free
  8-GPU nodes (the 2-node worker sits Pending → rendezvous times out). `build_grpo_wq36_1node_isr3.py`
  (`--nnodes 1`, config `dp_replicate 1`/`dp_shard 8`) is scientifically equivalent (ptmqx was
  single-node EP8) and needs only one node. Even single-node can queue a long time on a saturated cluster.
- **Eval serving capacity:** `eval_ckpt_generic.sh` stands up a private TP=2 sglang + an **8-GPU EP8
  sync** job. On the contended cluster the 8-GPU sync often sits **Pending** behind the live trainer +
  samplers + other experiments — it only schedules when an 8-GPU node frees. The eval scripts now
  default to **`--invalid-retries 0`** (the honest gate; see the science runbook's retry-crutch finding);
  `EVAL_INVALID_RETRIES=2` reproduces the old retry-propped numbers.

### Running a SECOND trainer in parallel — separate sampler pool (2026-06-26)
🛑 ONE sampler pool per trainer. Two trainers full-weight-syncing the same samplers → `[P2P]
/complete_weights_update failed … RemoteDisconnected`, which crashes BOTH runs (this happened once). To
run a second run (e.g. `clophi`) ALONGSIDE the live one, stand up an independent pool + router +
repointed trainer, touching ZERO of the live run's resources:
1. **sampler-c pool:** `launch/wordle-sci-opsd-sampler-c.yaml` — a separate StatefulSet
   (`wordle-sci-opsd-sampler-c`, headless svc `…-c-headless`, TP2). Set `replicas` to match the first
   pool (7), `kubectl apply`, wait for `wordle-sci-opsd-sampler-c-0..N` Ready. Separate StatefulSet → does
   NOT touch sampler-b.
2. **separate router `wordle-grpo-smg-c`:** clone the live SMG (`kubectl get deploy/svc wordle-grpo-smg
   -o json`), set `NAME=wordle-grpo-smg-c` and `WORKER_URLS` = the sampler-c endpoints, apply. Reusing the
   first router would load-balance generation onto both pools → still shared.
3. **repoint the trainer builder:** in `build_grpo_wq36_1node_clophi.py` set `SAMPLER` /
   `SAMPLER_INDICES` / the `SAMPLER_LOAD_URLS` f-string / `SMG_URL` ALL to the `-c` pool + `-c` router.
   The stock file ships **pre-wired to sampler-b + wordle-grpo-smg** — easy to miss.
4. **🛑 PRE-APPLY GREP (mandatory):** `grep -E 'sampler-b|wordle-grpo-smg\.apanda' <generated manifest>`
   must be EMPTY before `kubectl apply`. Any match = still pointed at the live pool → abort and fix.
5. **Health gate:** head pod Running → "engine ready" → train step ≥2 with datums ~1000+ (NOT ~150) and a
   finite reward. The SMG-bounce rule applies to the `-c` resources ONLY — never the live `-b`/no-suffix ones.

---

## Disk (weka `/shared`) — operational

- `/shared` is 473T, contested, and **has hit 100% mid-run.** DCP checkpoints are **196GB each**; a
  worker died with `OSError: No space left on device` at a step-50 save.
- Use **`save-interval 25`** and prune old checkpoints (the watchdog keeps the latest few). Free space
  before relaunching after a disk-full death; reschedule the worker.
- Write outputs only under `/shared/apanda/wordle-sft-runs`.

## Node exclusions

The preemptive banned-node list (`research-common-h100-005 / -050 / -080 / -089 / -113 / -116 / -118`,
historically flaky — DeepEP `CPU recv timeout`, optim_step OOMs) was **CLEARED by the user 2026-06-26**
(ample capacity; the bans were starving scheduling). The builders' `BANNED` lists / sampler-manifest
nodeAffinity should be emptied for new launches. Only reactively avoid a node that *actually*
hardware-faults on this run (import preflight fail / "no accelerator available" / DeepEP-NCCL startup timeout).

---

## Still-true MFU findings (compressed from the closed OPSD investigation)

These were measured carefully and remain valid as background. They are **secondary** now that the run
is rollout-bound — fix rollout/sampler saturation first; only chase these if the warm fwd/bwd becomes
the wall.

- **The OPD/KL loss is NOT the bottleneck.** A CUDA-synchronized one-step profile attributed ~95% of
  fwd/bwd to the backward bucket (transformer + recompute + comm), ~1.9% to forward, and only ~0.5% to
  the full-vocab KL region. Cheaper/top-k KL would save almost nothing.
- **The residual deficit is the model-runner fwd/bwd on tiny, imbalanced per-rank shapes.** A
  bare-tensor replay (direct `ModelRunner.forward_backward()`, no HTTP/sampler/teacher/weight-sync)
  measured **~0.4% MFU warm** on the captured shapes — close to the warmed *API* replay, proving the
  tax is NOT above-model machinery. The captured request gave each rank one real ~0.5k–3.5k-token
  sample (not 4096-padded, no dummy ranks). Coalescing to ~8k input tokens/rank lifts it only to
  ~0.7% MFU — still ~9× below the sibling synthetic 8k/rank point (~6.6%). 16k/rank OOMs in GDN
  `l2norm_bwd` Triton autotune. So the deficit is deeper than "too few tokens/rank."
- **EP=1 is WORSE than EP=8** on the exact same tensors (warm: ~14s vs ~8s). Removing the EP all-to-all
  is not the fix.
- **Quack MoE barely beats Triton** (~0.51% vs ~0.47% input-MFU warm) — a useful default, not the
  root cause. The active config uses `triton` (it didn't crash and is the validated path); quack is a
  small win if you want it.
- **Bigger fwd/bwd calls are worse** in the long-think regime (padding) — the sibling stack's
  "raise the FSDP chunk cap" fix is for SHORT completions and does NOT transfer here.
- **⚠️ Re-measure before trusting old absolute MFU numbers.** The old "0.13% MFU" headline was the
  *whole step* including the serial-rollout idle and the format-death turns. With the rollout fix +
  the format fix in place, the step-level MFU should be re-measured; the warm fwd/bwd (~65s) is the
  honest trainer-compute number to optimize.
- **Replay harness exists** if you need it again: `--dump-forward-backward-replay` captures real
  `/forward_backward` requests; `replay_microbatch_tensors.py` replays the post-slice tensors directly
  under `torchrun`. (Built for OPSD; the same path works for any server-mode trainer.)

---

## Bottom line

The p2p GRPO recipe above works. Live (2026-06-26) is **single-node EP8** (`grpo-wq36-1n-isr3`, climbing;
`grpo-wq36-1n-clophi` on its own sampler-c pool) — the 2-node variant is equivalent but its worker sits
Pending on a saturated cluster, so single-node is the default. The throughput lever that actually mattered
was making the **rollout parallel** (ThreadPoolExecutor + multi-sampler + SMG), which moved the run from
idle-bound to rollout-bound with a ~65s warm fwd/bwd. The remaining trainer-compute deficit (model-runner
on small per-rank shapes) is real but secondary; do not treat any pre-rollout-fix absolute MFU as a floor
— re-measure on the current path.
