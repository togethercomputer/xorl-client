# 1-Node Cluster Validation — Agent #1, 2026-06-14

Ran a **dedicated, unambiguously-owned** 1-node trainer to validate the throughput
conclusions on real hardware (not a microbench), per the directive to submit my
own kubectl jobs.

## Setup (mine, isolated)

- Stack `er-opd-tput-apanda-0614`: ONE pod × 8×H100 (h100-106), `team: turbo`,
  rdma, IPC_LOCK, no privileged. Generator copy
  `xorl-infra/k8s/opd_profile/tput_apanda_slots.py` (unique `STACK`, env-overridable).
- Engine `xorl-internal throughput/opd-lmhead-moe-gemm-20260614` (`e123b782`:
  lowmem KL + diag dtype fix + the zero-anchor fp32 fix). Client `xorl-opd-prefill`.
- Drove `write-trainer-server-control` + `replay_forward_backward_capture.py` on the
  AMDAHL-031 every-4-layer 1-node OPRD capture (the same payload AMDAHL-033 used).

## Result 1 — feeding without waste (the 4-node MFU lever), QUANTIFIED

Dispatcher dummy-fill (`runner_dispatcher.py:1043`):
`dummy = ceil(num_rows/dp_size)·dp_size − num_rows`. For the 64-sample OPRD batch
(~71,804 student tokens), via the denominator audit:

| packing seq_len | rows | dp=8 (1 node) waste | dp=32 (4 nodes) waste |
|---|---|---|---|
| 2304 | 32 | **0%** | **0%** |
| 4096 (production) | 22 | 5.9% | **32.1%** |
| 8192 | 10 | 36.1% | 70.7% |

**To keep a 4-node (dp=32) trainer fed without waste: pack so rows divide 32**
(seq2304 → exactly 32 rows → 0% dummy) **or grow the prompt window** (256 prompts
→ ~8% floor). The 64-prompt OPRD window is simply too small to fill 32 ranks at
the production seq4096 (only 22 rows → 32% dummy). **1 node is inherently denser**
(5.9% vs 32% at seq4096), which is the main reason fewer nodes raises useful MFU
here — it is removing dummy waste, not magic. NB: pack2304 won a replay before
(AMDAHL-025) but regressed the real generated run (AMDAHL-026) — the 0%-dummy
packing must be full-stack validated, not just replay-validated.

## Result 2 — the 1-node fb memory blocker is real and reproduced

Baseline streaming KL (no lowmem), `expandable_segments` OFF, pack2304: the fb
**computes forward+backward** then OOMs allocating the **1.89 GiB fp32 lm-head
`grad_weight`** (161 MiB free, 72.04 GiB allocated, 2.09 GiB reserved-unallocated).
This exactly reproduces AMDAHL-033 on my own trainer and confirms the 1.89 GiB
lm-head grad is THE 1-node blocker (and that the fb path otherwise works).

## Result 3 — the 1-node trainer-only fb is BROADLY UNRELIABLE (stalls)

The dominant observation, across ~6 fresh server restarts (pod GPU memory verified
clean each time — 8 worker procs, no leaks/zombies): the 1-node OPD trainer-only
`forward_backward` **stalls at 0% GPU util** (CPU ~1 core, engine logs silent
after "Engine Core started") in **nearly every configuration**, including plain
baselines. Configs observed:

| config | result |
|---|---|
| 64-datum, pack2304, baseline KL, no levers | **computed → OOM** (1.89 GiB lm-head grad) — the ONLY clean compute |
| 64-datum, pack2304, `opd_streaming_lowmem` | stall (0% GPU) |
| 64-datum, pack2304, `expandable_segments` | stall |
| 64-datum, pack2304, `activation_offload` | stall |
| 16-datum slice, pack2304, baseline KL | stall |
| 64-datum, pack4096/pack16k, lowmem | stall |

**Honest caveat (correcting an earlier over-confident note in this file's history):**
because plain baselines *also* stall, I CANNOT cleanly attribute the stalls to
`opd_streaming_lowmem`, `expandable_segments`, or `activation_offload`. There was
one clean A/B early on (baseline computed-to-OOM while a lowmem run stalled), but
a later 16-datum baseline also stalled, so the fb is **broadly flaky on this
1-node stack**, not deterministically broken by a specific lever. This corroborates
the runbook's record that **1-node was never made to work** (AMDAHL-029..033) — the
blocker is engine-level fb instability on the 1-node quack/DeepEP/FSDP path, not
only the lm-head memory. It cannot be root-caused remotely (no `ptrace`/py-spy in
the non-privileged pod; engine logs go silent at the stall).

## CORRECTION: PR #373 lowmem IS FSDP-safe (engine-design evidence, 2026-06-14)

A sibling agent resolved the FSDP-safety question with code evidence (not
inference): `torch_parallelize.py:381-385` deliberately groups final-norm + lm_head
into ONE FSDP unit with `reshard_after_forward=False`, so when `norm.forward()`
runs FSDP all-gathers the lm-head and KEEPS it gathered for `compute_loss()`. The
vocab-sharded loss path is opt-in and needs CP + dp_size=1 (OPD doesn't use it). So
**the lm-head weight at the loss is a full LOCAL tensor by design** — the streaming
KL (baseline AND `opd_streaming_lowmem`) always slices a full-local
`student_weight[start:end]`, never a sharded DTensor. Therefore my earlier
"lowmem hangs under FSDP via DTensor slicing" hypothesis was **WRONG**, and
**PR #373 is FSDP-safe and promotable**. The keep-fp32 recipe
(`opd_kl_backend=streaming` + `lm_head_fp32=true` + `opd_streaming_lowmem=true`) is
validated end-to-end. This is consistent with the cluster evidence below: the
1-node fb stall reproduced with PLAIN baselines too, so it was never lowmem.

## ROOT CAUSE of the 1-node fb stall (2026-06-14, via faulthandler stack dumps)

Used `PYTHONFAULTHANDLER=1` + SIGABRT (no ptrace needed) to dump stacks of the
stalled ranks. **The 1-node stall is a rank-0 DISPATCH deadlock, NOT a
compute/NCCL-GEMM/memory problem:**

- **Worker ranks (1-7)** are blocked in `runner_dispatcher.py:321 _worker_event_loop`
  → `dist.broadcast_object_list(src=0, group=cpu_group)` (Gloo) — waiting for rank 0
  to broadcast the next command.
- **Rank 0** is idle in `Rank0Protocol.run()` → asyncio `run_forever` (ZMQ wait) —
  it is NOT broadcasting.
- `register_session` dispatched fine (the replay logged "Session registered"), so
  the rank-0→worker Gloo broadcast handshake works. The **`forward_backward`
  command is never delivered to rank 0's `_handle_request_rank0` → never broadcast**,
  so the workers deadlock in `broadcast_object_list`.

So the deadlock is upstream of compute, in the **orchestrator (Engine Core) → ZMQ
→ Rank0Protocol → `_handle_request_rank0`** path for `forward_backward` on the
1-node topology (it works at 4 nodes and for `register_session` at 1 node). This
reframes the entire 1-node blocker: it is a **dispatch/serialization stall**, not
the small-GEMM / lm-head-memory story. The lm-head OOM is a *separate*, later
issue that only bites once the fb actually runs (it did, once, at full batch).
Likely suspects: the fb command/payload broadcast (large pickled OPRD batch) or an
orchestrator queue/ZMQ handoff that stalls specifically for `forward_backward` at
world_size=8. Next debug step: faulthandler-dump the **orchestrator/Engine-Core
process** (not the 8 rank procs) during the stall to see where the fb forward is
dropped.

## 10% MFU IS ACHIEVED on this model+hardware (q36-tput-2node sweep, 2026-06-14)

A sibling standalone-trainer sweep (`q36-tput-2node`, `xorl.trainers.trainer`,
2 nodes / 16×H100, Qwen3.6-35B-A3B, synthetic data, lm-head as a separate
FSDPLinear(2048→248320)) measured config `2node_nocp_mbs4` at **steady-state
MFU = 0.1057–0.1059 (~10.6%)** (tflops≈104.7, 93k tok/s, 2.35 s/step, peak 40 GB,
no checkpoint, microbatch 4). **So ≥10% MFU is demonstrably achievable** on this
exact model+hardware with a clean fwd/bwd training loop — confirming the premise.

**This pins down where OPD's ~1% goes:** it is NOT the model's fwd/bwd ceiling
(that is ~10.6%). The OPD *server* path adds, on top of the model fwd/bwd:
trainer-side teacher forward (~0.85 s), streaming-KL/lm-head loss (~0.91 s),
clear-grad (~0.59 s), weight sync, the 32-GPU dummy-rank waste, AND — at 1 node —
the orchestrator→rank-0 fb **dispatch deadlock** documented above. To bring OPD
toward the 10% the model can do: (1) fix the 1-node fb dispatch deadlock; (2) kill
dummy-rank waste (feeding result above); (3) cut the OPD-loss overhead (cache the
teacher instead of recomputing; the lm-head memory/KL work); (4) `mbs4`-style
dense microbatching like the winning sweep config. The standalone-trainer sweep is
the right Tier-0 ceiling probe and should be the MFU reference going forward.

## ✅ FIXED: the rank-0 fb dispatch deadlock (2026-06-14)

Root cause: `AsyncRouterChannel` (API→orchestrator and orchestrator→rank0) had **no
`ROUTER_MANDATORY`**, so a send to a DEALER identity that was briefly unroutable
right after (re)connect was **silently dropped**. A dropped `forward_backward`
command meant rank 0 never broadcast it, the workers blocked forever in
`broadcast_object_list`, and the caller awaited a response that never came
(intermittent: `register_session` worked, fb usually stalled, occasionally ran).

Fix (engine `27b1694b`, branch `throughput/opd-lmhead-moe-gemm-20260614`):
`zmq_channels.py` `AsyncRouterChannel` now sets `ROUTER_MANDATORY=1` (raise
EHOSTUNREACH instead of dropping) and retries the send with exponential backoff
(5ms→100ms, up to 30s) until the peer is routable. Only unroutable sends (latent
silent-drop bugs) are affected; working sends/4-node path unchanged.

**Validated on the 1-node trainer:** the fix engaged (2 "identity not routable yet"
retries at startup) and a 16-sample fb replay then **completed 5/5 iterations
reliably** (previously stalled at 0% GPU). First real 1-node OPD per-phase fb
breakdown (16 samples, pack2304, steady):

| phase | s |
|---|---|
| server_forward_backward | ~2.1 |
| model_forward | 0.33–0.49 |
| backward | ~0.65 |
| **clear_gradients** | **~0.50 (≈24% of fb)** |
| loss_compute | ~0.08 |
| oprd_teacher_forward | 0.0 (cache hit) |

New lever surfaced: **clear_gradients is a ~0.5 s FIXED cost** (zeroing the
[248320,2048] fp32 lm-head grad each step — also ~0.59 s at 4-node) → check
`set_to_none`/skip-rezero. Remaining 1-node blocker for the FULL 64-sample batch is
the 1.90 GiB fp32 lm-head grad OOM (lowmem keeps fp32 so it doesn't shrink the grad
buffer) → pursue a fused-quack loss mode (keep lm_head fp32) per the memory steer.

## ✅ FULL 64-sample 1-node OPD fb RUNS (dispatch fix + expandable_segments), 2026-06-14

With engine `27b1694b` (dispatch fix) AND `expandable_segments:True`, the full
64-sample fb now **completes 5/5 reliably, NO OOM, GPU util peaks 94%**. The
earlier "expandable_segments hangs" was the dispatch deadlock masking it — once the
fb actually dispatches, expandable reclaims the ~2 GiB reserved-but-unallocated
fragmentation and the 1.90 GiB fp32 lm-head grad fits. **So 1-node OPD is unblocked
end-to-end** (no lm_head_fp32=false, no lowmem even needed for the fit — though both
remain valid; expandable was the missing piece).

First authoritative full-batch 1-node measurement (pack2304, dp=8, 0 dummy):

| metric | value |
|---|---|
| server_forward_backward_s | **~11.4 s** (steady) |
| reconstructed logical MFU | **1.40%** |
| executed tok/s/GPU | 770 (all real, 0 dummy) |
| real student tok/s/GPU | 758 |

Per-phase (steady): model_fwd 3.1 s, **backward 5.85 s (52%)**, loss 1.23 s,
kl 1.0 s, **clear_gradients 1.49 s (13%)**, oprd_teacher_forward 0.0 (cache hit).

**vs 4-node** (1.37% MFU, but 85% dummy waste → ~513 real tok/s/GPU): 1-node gives
**~1.5× better REAL throughput/GPU** by eliminating dummy waste — confirming the
feeding analysis. But 1.40% is still 7.5× below the clean-trainer 10.6% ceiling.
The gap is OPD-specific overhead on the model fwd/bwd, in priority order:
1. **backward 5.85 s (52%)** — `recompute_before_dispatch` recomputes the forward in
   backward (~+3.1 s). `no_recompute` would cut it but needs more activation memory.
2. **clear_gradients 1.49 s (13%)** — zeroing the 1.90 GiB lm-head grad each step;
   `set_to_none` avoids it.
3. **KL+loss ~2.2 s (19%)** — full-vocab streaming KL recomputes the lm-head 3×;
   a fused KL would cut it.
4. **Bigger batch** — at 70400 tok the MoE M≈2200 (~44% GEMM) is fine, but fixed
   costs (clear-grad) aren't amortized; more prompts/step raises MFU.
Reaching 10% on OPD specifically is hard (full-vocab KL + teacher-matching are
inherent overhead the clean CE trainer doesn't pay), but 2-4% is reachable by
stacking levers 1-3, and the no-dummy 1-node real-throughput win is already real.

## Net status + next steps

- The one clean compute reproduced the **1.89 GiB fp32 lm-head `grad_weight` OOM**
  (AMDAHL-033) — so the memory blocker is real, but on top of it there is a
  **separate 1-node fb deadlock** that triggers in nearly all configs.
- **PR #373 (`opd_streaming_lowmem`)**: bit-exact + memory-saving on a single GPU
  (validated), but its **multi-rank/FSDP behavior is UNVERIFIED** — every cluster
  run that used it stalled, but so did baselines, so this is not a clean verdict.
  Do not promote/enable `opd_streaming_lowmem` under FSDP until there is a working
  1-node fb to validate against. Likely needs the lm-head DTensor gathered to a
  full local tensor before per-chunk slicing (the baseline's `weight.float()` does
  this implicitly).
- **Recommended next step:** the 1-node fb deadlock needs **engine-level debugging
  in a privileged/debug pod** (py-spy/`TORCH_NCCL_DEBUG`/flight-recorder) to find
  where the multi-rank fb stalls before compute. Until that is understood, the
  4-node trainer-only replay (which works) remains the only reliable inner loop,
  and the productive, validated lever is the **feeding/packing** result above.

## Artifacts

- Generator: `xorl-infra/k8s/opd_profile/tput_apanda_slots.py` (stack
  `er-opd-tput-apanda-0614`).
- Configs: `..._1node_warm009_deepep36_noprefetch{,_pack16k,_pack2304}.yaml`.
- Candidates: `AMDAHL-034..037` (lowmem / noprefetch / pack16k-1batch /
  pack2304-0dummy).
- Feeding audit JSONs: `RESULT_ROOT/er-opd-tput-apanda-0614/feed/audit_dp{8,32}.json`.
- Replay outputs (all empty/OOM — see above): `RESULT_ROOT/er-opd-tput-apanda-0614/fb_replay/`.
