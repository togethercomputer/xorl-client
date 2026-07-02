# Handoff: R3 routing-replay hang fix for the 235B trainer (externalize_r3_payloads)

> **SUPERSEDED 2026-06-30 — use Mooncake transport.** The shared-FS `externalize_r3_payloads` transport
> described below fixed the inline-broadcast deadlock but then STALLED the live fwd/bwd under shared-FS
> contention (the live run hung at step ~98 → 504 timeout). The current fix is **Mooncake/RDMA transport**:
> `r3_payload_transport: mooncake` (the bool `externalize_r3_payloads` was replaced by the 3-way
> `r3_payload_transport` {inline|filesystem|mooncake} selector), on engine branch
> `k3-recon-r3-mooncake-20260630` (commit `081f59a7`), validated frozen-replay mean_k3 2.4e-4. See
> `FILLER_GRPO_VALUE_OF_EXPLORATION_HANDOFF_20260630.md`. The diagnosis/mechanism below is still accurate;
> only the out-of-band transport changed (shared-FS → Mooncake).

Date: 2026-06-30. Context: the 235B GRPO "prescribed filler" (Value-of-Exploration) experiment needs full R3
routing replay (for low k3 / valid on-policy GRPO) at **10-shot** prompts, but full R3 over those long prompts
**deadlocked** the forward/backward. This is the fix, where it lives, how to enable it, and how it was verified.

## Symptom (the hang)
- Live prescribed arm: 10-shot random-number prompts (~1645 tok), 128 datums/step → routing payload
  (`routed_experts` + `routed_expert_logits`) ≈ **0.6 GB/step**.
- With R3 ON, the fwd/bwd **deadlocked** right after `R3: Pre-populated routing weights`: GPUs pinned at **100%
  util but ~120 W** (NCCL/Gloo collective spin-wait, NOT compute — H100 TDP ~700 W), engine `server.log` frozen
  18+ min. The watchdog fired a staleness alert.
- It did NOT hang with: free-form short prompts (~430 tok, inline R3, 156 steps) or the static replay (16 datums).
  So it is specifically the **long-prompt × full-routing data volume** broadcast inline through the Gloo command
  path (`dist.broadcast_object_list`) that deadlocks.

## Root cause + fix
The large routing payload was broadcast **inline** to worker ranks via the Gloo command path; serializing ~0.6 GB
through `broadcast_object_list` deadlocks at that volume. The fix keeps **FULL-length routing** (required for low
k3 — do NOT go sparse/answer-only; the R3 handler positions routing against forward rows, not an answer mask) but
moves the large arrays **out-of-band**:
- Orchestrator writes per-datum routing arrays to a shared filesystem (`r3_payload_dir`, default
  `<output_dir>/routing_payloads`) under **uuid-unique per-request dirs**, with crash-safe cleanup, and broadcasts
  only small refs (`__xorl_routing_payload_ref__`).
- Each worker lazily rehydrates its DP/SP slice from the shared FS.

Important nuance: the *previously deployed* engine already had the externalization machinery but **always-ON and
without uuid-unique dirs** → payload-file collisions across the 128 datums → it still hung. The actual fix is the
**gate + uuid-unique dirs + crash-safe cleanup**.

## Where it is implemented
- **Upstream branch:** `origin/fix/q36-live-k3-behavior-replay` (apanda's own branch, authored 2026-06-29 — the
  day after the deployed SHA). NOT on `apanda-dev` (which had not advanced past the deployed `1bf57eae`).
- **Fix commits:**
  - `be3fe480` "Add behavior logprob routing replay support" — the external-payload transport machinery + threads
    `routed_expert_logits` (behavior logprobs).
  - `5b463f7a` "Gate external R3 payload transport" — gates it behind `externalize_r3_payloads` (default **False**)
    + `r3_payload_dir` + `keep_r3_payloads`; adds uuid-unique dirs + crash-safe (finally) cleanup; demotes chatty
    R3 logs to debug.
- **Files (transport path):** `src/xorl/server/orchestrator/request_processor.py` (`_externalize_routing_payloads`,
  `unique_request_id = …uuid4().hex[:12]`, `ROUTING_PAYLOAD_REF_KEY`), `orchestrator.py` (gates
  `routing_payload_dir` on the flag), `runner/runner_dispatcher.py` (`_is_routing_payload_ref`,
  `_load_routing_payload_slice`), `runner/utils/routing_replay_handler.py`, `server/launcher.py`,
  `server/server_arguments.py` (the 3 fields). (Diff: 19 files, ~+968 lines incl. tests.)

## Deployed state
- **Engine repo:** `/home/apanda/xorl-qwen-k3-reconciliation` (the TRAINER engine; `xorl-opd-prefill` is the
  client/example repo — verify the deployed SHA via the run.sh `git -C $XORL_REPO rev-parse HEAD` line).
- **Deployed branch/SHA:** `k3-recon-r3-hangfix-merge-20260630` @ `98eb289e` — `fix/q36-live-k3-behavior-replay`
  merged onto a snapshot of the (uncommitted) k3-reconciliation engine work.
- **Preservation / rollback branch:** `k3-recon-wip-preserve-20260630` @ `78f6d76c` (full deployed working-tree
  snapshot, incl. the 175 uncommitted k3 changes + R3 + k3_tests artifacts). The original deployed branch
  `docs/qwen-k3-reconciliation-20260627` @ `1bf57eae` is also intact.
- **k3 reconciliation work preserved:** moe_block / modeling_qwen3 / normalization / rope are 0-diff vs the
  preservation snapshot (the merge only took upstream server/handler changes).

## How to enable (REQUIRED — off by default)
Set in the trainer config (`/shared/apanda/filler_grpo/trainer_grpo_k3.yaml`, a flat ServerArguments config):
```yaml
externalize_r3_payloads: true   # r3_payload_dir defaults to <output_dir>/routing_payloads (must be shared FS)
```
Without this flag the merged engine reverts to the inline ~0.6 GB Gloo broadcast and **the hang returns**. Keep R3
on the client side (`return_routed_experts=true return_expert_logits=true`).

## Verification (live, 2026-06-30, prescribed arm)
- Hang cleared: step 0/1/2 fwd/bwd complete (GPUs draw real power; past the prior `R3: Pre-populated routing
  weights` deadlock). Engine log: `External R3 routing payload transport enabled: dir=…/routing_payloads`.
- Full routing intact → **k3 reconciles: step 0→1→2 = 1.08e-2 → 6.76e-4 → 1.51e-4** (target band per the static
  replay = 7.8e-5–1.3e-4). NB **step-0 k3 reads transient-high (~1e-2) then drops by step 2** — cold start; do not
  judge R3 from step 0.
- `format_correct_rate=1.0`, p2p weight sync ~8.3 s steady-state, `za/count` ~6–9/16.
- To see the demoted R3 apply logs (`Externalized R3 routing payload …`, per-rank `Loaded R3 … slice […]`), set
  env `XORL_R3_VERBOSE_LOGGING=1` on the trainer.

## Related
- Memory `q235-r3-externalize-payloads-hang-fix`. The complementary live-MoE-comms gotcha is
  `q235-deepep-deadlocks-live-sync` (deepep deadlocks the live p2p sync — keep `ep_dispatch: alltoall`).
- k3 recipe detail: `docs/notes/k3_235b_live_investigation_2026_06_28.md` (full-routing + answer-only-loss),
  `docs/notes/q235_ep_dispatch_k3_sweep_2026_06_29.md`.
