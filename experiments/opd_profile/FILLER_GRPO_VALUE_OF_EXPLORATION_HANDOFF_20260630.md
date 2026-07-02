# Handoff: 235B GRPO filler-token "Value of Exploration" experiment — current state 2026-06-30

Entry point for the next agent on the 235B GRPO filler-token work. Ties together the experiment,
the live stack, the engine, the low-k3 recipe, the R3/save sagas, staged fixes, and next steps.
Detailed docs are linked at the bottom.

## Goal
Reproduce the paper's filler-token RL results on Qwen3-235B-A22B (4-digit multiplication). Two threads:
1. **Emergent filler preferences during RL** — which filler strategies the model comes to prefer
   (logged via `strategy/*` + `strategy_correct/*`).
2. **The Value of Exploration** (current focus) — the **prescribed-vs-generated** contrast: a model that
   GENERATES its own filler keeps improving over RL, while a model given PRESCRIBED (forced) filler
   plateaus and its zero-advantage (ZA) count grows. **Readout = `za/count` + `reward/mean` over steps.**

## Experiment status (as of 2026-06-30 ~07:55Z)
- **Prescribed-random arm** (forced random-number filler): ran to ~step 97, then hit the shared-FS R3
  stall. Showed the predicted shape: reward 0.46→0.58 then flat ~0.55, `za`~7/16, format 1.0, correct
  ~0.5 — "initial improvement then saturate." Partial; worth a clean re-run for the full curve.
- **Prescribed-LOREM arm** (forced lorem filler): client dict-with-shape fix applied (R3 saga #4 RESOLVED);
  the Mooncake attempt then SEGFAULTED live (R3 saga #5) → **pivoted to `filesystem` transport** and
  **RELAUNCHED 2026-06-30 ~07:46Z. RUNNING + validated at step 0/1: k3=2.74e-4, format=1.0, reward∈{0,1}
  (reward_format=0/reward_correct=1), za/count=4 (r0=3,r1=1).** Babysitting to completion (~156 steps,
  N=2500). wandb `grpo-235b-filler-prescribed-lorem-20260630`. NB step 0→1 ≈ 4 min (cold start incl).
- **Generated arm** (model generates its own filler): **NOT YET RUN — the headline contrast.** Cleanest
  design = **flip ONLY `prescribe_filler=true→false`** and KEEP everything else identical to the lorem arm
  (incl. `advantage_weighting=uniform`, `filler_token_type=lorem` demos, `num_fewshot=10`, reward_format=0/
  correct=1, N/group/batch). Rationale: with `uniform`, the prescribed arm's filler is prefill (not in
  `sampled_tokens`) so advantage lands on answer tokens only; the generated arm's filler IS sampled so
  advantage reaches the **filler tokens** → it can learn/explore useful filler, and za stays alive because
  filler varies across group samples (vs prescribed: identical forced filler → za grows). That asymmetry IS
  the Value-of-Exploration mechanism. Do NOT use `answer_only` for generated — it would zero the filler-token
  gradient and defeat the experiment. **Keep `num_fewshot=10`** (0-shot collapses the format). Same
  full-R3-filesystem setup. The only run.sh edits: `prescribe_filler=false` + new wandb name. Most important next.

## Live stack: `er-opd-q235-fillerrft-slots`
- 10 standalone pods: `trainer-head` + `trainer-worker-1..7` (8 trainer nodes, EP8) + `sglang-0`
  (TP8 sampler) + `dispatch` (CPU SMG router). = 9 GPU nodes. No teacher.
- **Reprogrammable slots**: swap the workload by rewriting
  `/shared/opd-control/er-opd-q235-fillerrft-slots/<role>/run.sh` — the slot agent re-execs its child on
  run.sh hash change (no pod reschedule), and auto-runs the run.sh on pod startup.
- **Re-acquire recipe** (stack deleted; needs ~9 whole free 8-GPU nodes):
  ```
  OPD_STACK=er-opd-q235-fillerrft-slots OPD_SAMPLER_GPUS=8 \
    /home/apanda/xorl-internal/.venv/bin/python \
    /home/apanda/xorl-infra/k8s/opd_profile/q36_35b_reprogrammable_slots.py \
    --model q235 --trainer-nodes 8 --teacher-replicas 1 --teacher-route direct \
    render-manifest --sampler-replicas 1 --output /tmp/q235-raw.yaml
  # drop the 2 teacher docs (stock generator forces --teacher-replicas>=1; our stack has no teacher):
  #   yaml: keep docs where 'teacher' not in metadata.name  ->  /tmp/q235-reacquire.yaml  (10 pods, 9 GPU)
  kubectl apply -f /tmp/q235-reacquire.yaml
  ```
  `OPD_STACK` env sets the stack name (defaults to the q36 name). xorl-infra is on branch
  `opd-battery-consolidation` with the q235 generator port (uncommitted). The control dir survives a pod
  delete, so re-acquired pods read the existing run.sh. `kubectl apply`/`delete` on these pods is gated by
  the auto-mode classifier — needs explicit user authorization or the user runs it.
- **Do NOT `kubectl delete` slot pods to "reset"** (loses scarce whole-node capacity). Exception: a
  deliberate full release (done 2026-06-30 to free capacity for another effort; then re-acquired).

## Engine
- **DEPLOYED: branch `k3-recon-r3-mooncake-20260630` @ `081f59a7`** in `/home/apanda/xorl-qwen-k3-reconciliation`
  (working tree checked out on it; the run.sh's `XORL_REPO` points here). Has the k3-reconciliation work
  + the 3-way `r3_payload_transport` selector. **The live run uses `filesystem` transport, NOT mooncake**
  (mooncake segfaults live — R3 saga #5); this engine SHA supports both. Verify the live trainer logs this
  SHA at launch.
- Rollback: `k3-recon-r3-hangfix-merge-20260630` (`98eb289e`, shared-FS R3 only);
  `k3-recon-wip-preserve-20260630` (`78f6d76c`, raw k3-work snapshot).
- Venv `.venv-cu132-latest-probe`. NB `/home/apanda/xorl-opd-prefill` is the CLIENT/example repo, NOT the
  engine — don't confuse them; verify the deployed engine SHA before acting.

## Low-k3 recipe (reconciliation)
k3 = sampler-vs-trainer logprob divergence on the answer tokens; must be ~e-4 for valid on-policy GRPO.
- Sampler (`sglang-0/run.sh`): `--enable-fp32-lm-head --enable-fp32-router --attention-backend fa3
  --enable-return-routed-experts --enable-return-expert-logits` + env `SGLANG_DISABLE_ROPE_COMPILE=1
  SGLANG_RMSNORM_FP32_WEIGHT_MUL=1 SGLANG_RETURN_ORIGINAL_LOGPROB=1`. (Batch-invariant NOT needed — dropped,
  k3 held.)
- Trainer (`/shared/apanda/filler_grpo/trainer_grpo_k3.yaml`): `router_fp32:true lm_head_fp32:true
  ce_mode:eager rmsnorm_mode:native enable_compile:false ep_dispatch:alltoall` + the **packing-pad
  IGNORE_INDEX fix** (THE big k3 lever — packed `target_tokens` padded with IGNORE_INDEX, not 0).
- **R3 routing replay** (FULL routing, NOT answer-only): use **`r3_payload_transport: filesystem`** for FULL runs
  (+ `r3_payload_dir: <sharedFS>/routing_payloads`, `r3_payload_keep:false`, NO `namespace_prefix`) — it carried the
  156-step prescribed-lorem run + the 10-shot run to step ~98-155. **Mooncake is validated only to step 1 and
  DEADLOCKS at step 2** (saga #6): launching a `mooncake_master` + wiring MOONCAKE_* via a `head_ip` file gets
  step 0/1 (k3=5.1e-4, store 100%, no segfault), but the mega-filler run then hung at step-2 fwd/bwd (Mooncake
  R3-store vs p2p-Mooncake weight-sync contention). The whole R3+Mooncake stack IS on apanda-dev (PR #426). Both keep FULL routing.
- Client: `XORL_INFERENCE_API_FORMAT=generate` (the prescribe token-id prefill needs /generate;
  chat_completions rejects ModelInput), `temperature=1.0`, no top-p/k truncation, `return_routed_experts=true`.
- **Live k3 ~1e-3 (bounces 1e-4–5e-3). Step-0 reads transient-high (~1e-2) then drops by step 2 — do not
  judge R3 reconciliation from step 0.**

## The R3 saga (why Mooncake) — see also memory `q235-r3-externalize-payloads-hang-fix`
Full R3 over the long 10-shot prompts (~1645 tok, ~0.6GB routing/step) deadlocked the fwd/bwd:
1. inline Gloo `broadcast_object_list` of 0.6GB → deadlock at bring-up;
2. shared-FS externalization (`externalize_r3_payloads`) → fixed the broadcast but the shared-FS payload
   I/O **stalled under FS contention** (hung at step ~98 → 504 timeout);
3. **Mooncake/RDMA transport (`r3_payload_transport: mooncake`)** → the ENGINE fix (validated frozen-replay
   mean_k3 2.4e-4). `externalize_r3_payloads` is replaced by the 3-way `r3_payload_transport`
   {inline|filesystem|mooncake}; `r3_payload_dir` is filesystem-only (errors under mooncake).
4. **RESOLVED (2026-06-30): the CLIENT now sends dict-with-shape, not bare base64.**
   `filler_tokens_rl.py::_align_routing_payload` now returns `{"data": <base64>, "shape": [rows, 94, 8]}`
   (rank-3) for BOTH `routed_experts` (int32) and `routed_expert_logits` (float32). Verified vs the engine
   validator `side_payloads.py::_decode_sglang_routing_dict` (needs `data` + rank-3 int `shape`). The dict
   form is accepted by ALL decode paths (`routing_replay_handler._decode_routing_array` takes dict-or-bare;
   filesystem pickles the item as-is and decodes on worker load) → the client is now transport-agnostic.
5. **Mooncake FAILS LIVE → use `filesystem` (2026-06-30).** With the client fix, the Mooncake 500 cleared and
   step 0 sampled fine, but step-0 fwd/bwd hit a `coro_rpc_client … Connection refused` storm → the Go
   transfer engine **SEGFAULTED (engine exit -11)** → launcher tore everything down. Root cause:
   `MooncakeStoreConfig.from_env` defaults to a master at `localhost:50051` + http metadata at `:8090`, and
   **this stack never launches a `mooncake_master`/`mooncake_http_metadata_server`** (binaries exist in the
   venv bin, but nothing starts them; cross-node TCP also needs per-node `MOONCAKE_LOCAL_HOSTNAME` the
   ref-broadcast doesn't carry). The frozen-replay validation had a master; the live slots don't. (Superseded by #6.)
6. **Mooncake WORKS to step 1 (launch the master) but DEADLOCKS at step 2 → use filesystem for full runs.** Ported the master-launch recipe from the
   validated static stack `er-opd-q235-r3moon-r2` into the live run.sh: HEAD launches `mooncake_master
   -enable_http_metadata_server -http_metadata_server_host=0.0.0.0 -http_metadata_server_port=8090 -rpc_port=50051
   -port=50051 &` (master :50051 + metadata :8090 in one process), publishes `POD_IP` → shared `head_ip` file,
   exports `MOONCAKE_PROTOCOL=tcp MOONCAKE_LOCAL_HOSTNAME=${POD_IP} MOONCAKE_MASTER_SERVER=${POD_IP}:50051
   MOONCAKE_METADATA_SERVER=http://${POD_IP}:8090/metadata MC_STORE_MEMCPY=0 MC_IB_PORT=1`, kills master on cleanup;
   each WORKER reads `head_ip` + points `MOONCAKE_*_SERVER` at the head's pod IP (Service does NOT expose 50051/8090).
   Config: `mooncake` + `r3_payload_namespace_prefix` (NO `r3_payload_dir`). **Validated live: master Clients=65,
   PutStart=448/448, Get=3274/3274 (100%), workers `Loaded R3 … Mooncake slice`, routing replay applies, k3=5.1e-4,
   NO segfault.** Gotchas: needs 9 HEALTHY nodes (a rank stuck on cordoned h100-058 failed distributed-init twice —
   move the pod off cordoned nodes); kill any leaked stale `mooncake_master` before relaunch (else :50051 bind
   conflict). **⚠️ step-2 deadlock:** validated only to step 1 — the ~1000-tok mega-filler run completed step 0/1
   then HUNG at step-2 fwd/bwd (uniform 120W/0%-util collective deadlock after R3 pre-populated in 0.032s; likely
   Mooncake R3-store vs the p2p-Mooncake weight-sync contending once the pipeline saturates). `filesystem` (+
   `r3_payload_dir`, drop namespace_prefix) is the ROBUST full-run transport (156-step lorem + 10-shot to ~step 155);
   use it for full runs. NB #428 (a reland attempt) was CLOSED as a no-op duplicate — the R3+Mooncake stack is
   already merged on apanda-dev via PR #426.
- Do NOT switch to answer-only routing (full routing required; engine pads short routing with dummy
  experts at wrong positions).
- Do NOT use `ep_dispatch: deepep` for the LIVE run (deadlocks the p2p weight-sync). deepep is the
  frozen-trace k3 optimum (4.5e-5 @ EP8) but the live deadlock is the cross-node EP16+ NVSHMEM path;
  EP8-intranode works. A default-off quiesce hook is staged (`deepep-livesync-fix-20260630`). See memory
  `q235-deepep-deadlocks-live-sync`.

## save_state / checkpoints
- Save failures were **disk-full**: `server_output` hit the 10 TB `StorageLimit` (11 TB of stale
  checkpoints from 4 prior runs — ~1.3 TB each). Deleted those 4 (10.4 TB freed) 2026-06-30.
- **Retention fix staged** (branch `savestate-fix-20260630`): `max_checkpoints_to_keep` (default 0=off)
  prunes this-run's oldest on a 507. Set it (e.g. 3) + relaunch to enable. NOT in the Mooncake engine —
  merge if wanted. Disk fills fast (1.3 TB/ckpt × save_every=50) — deploy this before a long run.
- Checkpoints are DCP (via `save_state` → `/api/v1/save_weights`); SGLang eval needs DCP→HF conversion.
  Saves are best-effort (try/except `[SAVE-SKIP]` in the harness) — a save failure won't kill the run.

## Staged fixes (branches in the engine repo, NOT in the deployed Mooncake engine)
- `savestate-fix-20260630` — checkpoint retention (`max_checkpoints_to_keep`).
- `r3-strict-validation-20260630` — fail-loud R3 row validation (gated `r3_strict_row_validation`, default
  off; asserts `routing_ntok == fwd_len`, raises instead of silent pad/truncate).
- `deepep-livesync-fix-20260630` — default-off quiesce/teardown hook for deepep+sync (future cross-node).

## Config knobs (run.sh client args + trainer_grpo_k3.yaml)
- `prescribe_filler` — true = prefill forced filler + sample answer ONLY (uniform advantage); false =
  free-form generation (answer_only advantage).
- `filler_token_type` (random_numbers / lorem / counting / nato / ...), `mixed_filler`.
- `num_fewshot=10` (paper). **0-shot collapses the format — demos are load-bearing** (the model can't
  bootstrap valid filler at 0-shot via GRPO at clipfrac ~0.005).
- reward: `reward_format` / `reward_correct` + `require_format`. For the matched contrast we used
  `reward_format=0 reward_correct=1 require_format=true` → reward = formatted·correct ∈ {0,1} → clean ZA
  buckets (r0=never-solved, r1=always-solved).
- `group_size=8 batch_size=16 learning_rate=5e-6 loss_fn=policy_loss`. N=2500 is the DATASET size →
  ~156 steps/epoch (NOT 2500 steps).
- Harness: `/home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py` (GRPO client —
  prescribe_filler impl, the save_state fix, the generate-api requirement; git-ignored in that repo).

## Open items / next steps
0. **[RESOLVED] R3 transport** — client dict-with-shape fix DONE (saga #4); Mooncake segfaults live (saga #5)
   so the live transport is **`r3_payload_transport: filesystem`** (running, k3=2.74e-4). Babysitting
   prescribed-lorem to completion via the run watchdog.
1. **Generated arm** — the headline Value-of-Exploration contrast (config above). Highest value next.
2. Let prescribed-lorem finish; compare `za`/reward vs prescribed-random (content effect — expect ~no
   difference per the inference map's "budget dominates content").
3. Deploy `savestate-fix` (retention) — disk is a recurring blocker; and `r3-strict-validation` for a
   blessed lowest-k3 run. (Merge them into the Mooncake engine.)
4. Post-hoc eval for acc_pause/acc_nopause (runs use `eval_every=0`): DCP→HF a checkpoint + run the filler
   eval. (DCP→HF has been finicky — see `wordle-pope-result` memory.)

## Pointers
- `HOWTO_235B_OPD_RFT_BRINGUP.md` — bring-up failure modes + recipe (§6-10 cover K3 / mooncake-p2p /
  pipeline_rl / EP-dispatch / slots).
- `R3_EXTERNALIZE_PAYLOADS_HANGFIX_HANDOFF_20260630.md` — R3 hang fix detail (NOTE: Mooncake now
  supersedes the shared-FS transport described there).
- `FILLER_235B_EXPLOIT_RUNBOOK_20260620.md` — filler science findings + the PHASE-2B GRPO section.
- `AGENT_COORDINATION.md` — cross-agent coordination + current-run state.
- Memories: `q235-r3-externalize-payloads-hang-fix`, `q235-deepep-deadlocks-live-sync`,
  `reprogrammable-slots-235b-launch`, `wordle-live-k3-floor`.
