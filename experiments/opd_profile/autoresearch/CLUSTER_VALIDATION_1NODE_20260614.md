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
