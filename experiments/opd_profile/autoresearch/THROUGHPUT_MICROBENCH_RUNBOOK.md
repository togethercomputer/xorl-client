# OPD Throughput Microbench Runbook

Last updated: 2026-06-14 UTC.

This is the focused handoff for Agent #1 throughput work on the filler-token /
prefill-time-compute OPD setup. It exists because the full infra runbook is dense
and because the next throughput loop should not require the full 32-GPU OPD
stack.

## Start Here

Work from:

```bash
cd /home/apanda/xorl-opd-prefill
```

Read these in order:

1. `AGENT_COORDINATION.md`
2. this file
3. `experiments/opd_profile/autoresearch/CANONICAL_INFRA_RUNBOOK.md` §0 and §7e

Core rule: **iterate on one node first.** The stack is far below acceptable MFU,
so the next throughput agent should build and optimize a 1-node reproducer or
surrogate for the fwd/bwd shape before spending 32 H100s. Use 4-node replay only
as a later fidelity/promotion gate when a 1-node candidate has a reason to
survive.

## What Is Already Known

- The full OPD science stack is not needed for fwd/bwd attribution. AMDAHL-021
  captured a static OPD fwd/bwd payload and replayed it through the trainer API
  without samplers, teacher prefill, endpoint registration, optimizer, or weight
  sync.
- The 4-node trainer-only replay baseline measured
  `server_forward_backward_s=4.4588`, with forward/backward/clear-grad split
  around `1.60 / 1.58 / 0.63 s`.
- The sorted AMDAHL-021 replay split showed the forward bucket was not just the
  student model: `model_forward_s=0.8056`,
  `oprd_teacher_forward_s=0.8498`, `loss_compute_s=0.9071`,
  `backward_compute_s=1.5860`, `clear_gradients_s=0.5882`.
- MFU remains bad even on executed tokens: the denominator audit reconstructs
  about `1.37%` logical MFU over dispatcher-executed student tokens, and only
  `0.039%` when scaled by valid answer tokens. Do not call this acceptable.
- AMDAHL-028 rejected a dominant fixed server/API tax: minimal dummy rows were
  neutral (`4.4612 s`), and repeated static data improved executed throughput
  only from `520.9 -> 584.4 -> 775.5 tok/s/GPU` at repeat 1/2/4.
- AMDAHL-025 was a replay-only false positive: pack2304 improved static replay
  (`4.1632 s`) but AMDAHL-026 regressed the real full strict run
  (`forward_backward_s=11.03 s` vs AMDAHL-020's `4.21 s`).
- EP=1 was tested in replay and lost (`5.8706 s`), so do not rerun it as the
  next obvious lever.
- No-checkpoint was tested in warmed replay and lost/was neutral
  (`4.6051 s`), so do not spend the next pass there.

## 1-Node Microbench Ladder — Results (2026-06-14, Agent #1)

Built the cheap end of the ladder as **single-GPU** microbenches (no trainer, no
32-GPU stack) using the real Qwen3.6-35B-A3B shapes
(`H=2048`, `moe_intermediate=512`, `E=256`, `top_k=8`, `V=248320`, `40` layers).
Scripts live in `experiments/opd_profile/scripts/`. Ran on one free H100 with the
engine venv (`/home/apanda/xorl-internal/.venv`) and the engine worktree on
`PYTHONPATH` (see Artifacts). These resolve the root-cause question and produce a
validated, gradient-identical memory fix for the AMDAHL-029..033 blocker.

### Root cause of low MFU = small MoE expert GEMMs (`microbench_moe_gemm.py`)

The student is an A3B MoE. With balanced routing each expert sees only
`M_per_expert = tokens_in_EP_group * top_k / E` tokens, i.e. a
`[M, 2048] @ [2048, 512]` GEMM. Sweeping M (grouped `bmm`, the kernel ceiling),
`MFU = achieved / 989 TF` bf16:

| M/expert | tokens/rank | grouped MFU (EP=8) | grouped MFU (EP=1) | per-expert **loop** floor |
|---|---|---|---|---|
| 8  | 256   | 1.4%  | 2.2%  | 0.07% |
| 64 | 2048  | 10.8% | 13.9% | 0.53% |
| **72** | **2304** | **11.7%** | **15.2%** | **0.59%** |
| 128 | 4096 | 18.8% | 22.8% | 1.05% |
| 256 | 8192 | 28.7% | 33.6% | 2.12% |
| 512 | 16384 | 37.0% | 40.9% | 4.22% |
| 1024 | 32768 | 42.1% | 44.2% | 8.58% |

Conclusions, in order of importance:

1. **It is small GEMMs, and it is fixable.** A grouped expert GEMM crosses 10% MFU
   at `M=64` (≈2048 real tokens/rank) and reaches **37% at M=512**. The OPD
   operating point (~2.3k tok/rank → M≈72) sits at ~12-15% on the ideal grouped
   curve. The lever is **more real tokens per expert per forward**: denser
   packing, larger microbatch, larger EP group, and fewer DP/dummy ranks. EP
   *increases* M for fixed per-rank tokens (the all-to-all gathers the EP group's
   tokens before the expert GEMM), so collapsing to 1 node does NOT shrink M as
   long as packing stays dense.
2. **A per-expert Python/loop path would be catastrophic** (0.6% at M=72, ~20×
   worse than grouped, and ~matches the observed ~1% number). The engine MoE uses
   real grouped-GEMM kernels (`triton`/`native` `torch._grouped_mm`/`quack`,
   default `triton`), so production is on the grouped curve, not the loop floor —
   **do not** regress onto an eager/loop expert path.
   - Confirmed with the engine's actual production primitive: `torch._grouped_mm`
     (the `native` backend, SwiGLU gate_up+down fwd+bwd) tracks the `bmm` ceiling
     within ~1pp at every M — **11.3% vs 11.8% at M=72**, 39% vs 37% at M=512,
     48% vs 46% at M=4096. So the small-M deficit is **not** a kernel
     inefficiency; the grouped kernel is already at the ceiling. The only lever is
     bigger M. (`microbench_moe_gemm.py --grouped-mm`.)
3. The headline "~1.37% executed MFU" is **student-model FLOPs ÷ total wall time**.
   The FLOP counter counts only the student model, but the 4.46 s wall also pays
   for teacher forward (0.85 s), KL/loss (0.91 s), backward (1.59 s), clear-grad
   (0.59 s), and — at 32 GPUs — an **85% dummy-rank waste** (5 packed rows spread
   over 32 DP ranks; `dispatcher_dummy_executed_tokens=425088` of `497280`). So
   the path to 10%+ is: kill dummy-rank waste (1-node/dense packing) + grow M +
   trim clear-grad/teacher-recompute wall time.

### lm-head streaming-KL is a MEMORY problem, not a small-GEMM problem (`microbench_lmhead_kl.py`)

`streaming_reverse_kl` fwd+bwd at real `V=248320`, `H=2048`, fp32 (the
`lm_head_fp32=true` regime), sweeping valid-token count N:

| N_valid | MFU | peak GB (fp32) |
|---|---|---|
| 128 | 9.1% | 6.1 |
| 1536 | 20.8% | 7.3 |
| 3049 (full batch) | 21.7% | 8.8 |
| 6144 | 22.1% | 11.9 |

The lm-head KL runs at **20%+ MFU** even at small N (the `V`-dimension is huge, so
the GEMM is not skinny) — it is **not** a small-GEMM problem. It IS the 1-node
**memory** blocker: at N=3049 the ~8.8 GB peak is dominated by **5.7 GB of weight
tensors** (student 1.9 + teacher 1.9 + a full fp32 `grad_weight` buffer 1.9), and
the trainer's grad-accumulation case (pre-existing `.grad`) pushes it to **10.7 GB**
(autograd doubles the returned grad against the existing `.grad`). This is exactly
the AMDAHL-033 "OOMed on full lm-head gradient allocation" failure. Free, numerics-
neutral lever found: `vocab_chunk_size=8192` cuts peak 8.8→6.6 GB for ~5% slower KL.

### Validated memory fix: `streaming_reverse_kl_lowmem` (engine, gradient-identical)

New engine path (`opd_streaming_kl.py`, opt-in via `opd_streaming_lowmem=true`):
keep the lm-head weights in their native (bf16) dtype and **upcast each vocab
chunk to fp32 inside the kernel** instead of holding two full fp32 weight copies;
return/accumulate the weight grad in the native dtype (optional in-place into the
leaf `.grad`, no second full `[V,H]` buffer). Because slicing commutes with the
elementwise upcast and vocab chunks partition grad rows disjointly, it is
**gradient-identical** to the current fp32 path (`validate_lmhead_kl_lowmem.py`:
`max|Δ|` = 0 on kl, grad_hidden, AND grad_weight). Measured peak reduction at
N=3049, vchunk=8192:

| case | current fp32 path | lowmem | saved |
|---|---|---|---|
| no grad-accum | 19.9 GB | 15.3 GB | **4.6 GB (23%)** |
| grad-accum (pre-existing `.grad`) | 21.8 GB | 15.3 GB | **6.5 GB (30%)** |

(absolute numbers inflated by resident comparison tensors; the **delta** is the
clean signal). Wired through `model_runner` as `opd_streaming_lowmem`; unit test
`test_opd_streaming_lowmem_matches_streaming` asserts loss + grad identity; full
`tests/ops/loss/test_opd_loss.py` (15) passes. Engine branch
`throughput/opd-lmhead-moe-gemm-20260614` off `origin/apanda-dev @ 609bed76`.

## Failed 1-Node Attempts

The desired cheap target is one-node OPD replay, but the direct path is currently
blocked by memory. The failures are useful because they locate the next work:

| candidate | result |
|---|---|
| AMDAHL-029 | 1-node trainer-side OPRD replay OOMed in `_trainer_teacher_kept_layers` before any replay row. |
| AMDAHL-030 | Moved OPRD hidden cache to SGLang rank-3, but all-40-layer cache saturated teacher memory and wrote no capture. |
| AMDAHL-031 | Every-4th-layer SGLang cache captured successfully, but 1-node trainer replay OOMed before a row. |
| AMDAHL-032 | Selected hooks removed full hidden retention, but all-layer prep64 still OOMed in model forward/final-norm. |
| AMDAHL-033 | Every-4th-layer cache + selected hooks reached streaming-KL backward, then OOMed on full lm-head gradient allocation in `opd_streaming_kl.py`. |
| AMDAHL-033 KL staging retry | Still OOMed because current OPD uses `lm_head_fp32=true`; the weight was already fp32 before grad allocation. |
| AMDAHL-033 root-caused + fixed (2026-06-14) | The blocker is concretely the fp32 lm-head copies + full fp32 `grad_weight` buffer (~5.7 GB at N=3049, 10.7 GB with grad-accum). `streaming_reverse_kl_lowmem` (per-chunk fp32 upcast, native-dtype grad, in-place optional) reclaims 4.6-6.5 GB **gradient-identical** — validated 1-GPU. Still needs an 8-GPU 1-node trainer run to confirm it clears the OOM end-to-end (blocked on free GPUs, not on code). |

Failed / discarded hypotheses this cycle (recorded so they are not retried):

- *"The lm-head streaming KL is a small-GEMM / low-MFU hotspot."* **False** — it
  runs 20%+ MFU at realistic N (the V=248320 dimension keeps the GEMM fat). It is
  purely a memory problem.
- *"In-place `.grad` accumulation alone unblocks AMDAHL-033."* **Partly** — it only
  saves the autograd *doubling* (~1.9 GB), which appears in grad-accumulation, not
  in the single-microbatch first-allocation that 033 likely hit. The decisive
  lever is the **per-chunk fp32 upcast** (avoids the full fp32 weight copies),
  with in-place grad as a bonus.
- *"`lm_head_fp32=false` is the memory fix."* **Avoided** — the lowmem path gets
  the memory back with bit-exact fp32 numerics, so the gated `lm_head_fp32=false`
  numerics change is unnecessary for the memory goal.
- *"Going to 1 node shrinks the MoE GEMMs (smaller M)."* **False** — EP gathers the
  whole EP group's tokens before the expert GEMM, so M depends on packing density,
  not node count. 1 node helps by removing dummy-rank waste, not by changing M.

## Next Throughput Target

Build a **1-node fwd/bwd microbench ladder** that starts cheap and only adds OPD
fidelity when needed:

1. Bare 1-node synthetic `xorl.cli.train` sweep to find the tokens/rank knee for
   this model, balanced MoE routing, and checkpoint family. This is a ceiling
   probe, not a proof of OPD behavior.
2. 1-node bare-tensor replay from the AMDAHL-021 payload: bypass HTTP/server
   request parsing if needed, but preserve packed rows, dummy-row participation,
   valid-token sparsity, and student-token shapes.
3. Add trainer-side OPRD teacher-forward/cache behavior.
4. Add streaming KL/lm-head gradient behavior in the smallest sharded/chunked
   form that reproduces the memory/latency bucket.
5. Only after the 1-node reproducer shows a real win, confirm on the 4-node
   trainer-only replay, then on a short full OPD promotion gate if the generated
   batch shape changed.

The likely next code target is the lm-head/streaming-KL memory path that blocked
AMDAHL-033. Prefer sharded or chunked vocab-parallel OPD KL weight-gradient
handling over recipe changes. `lm_head_fp32=false` may be a fit probe, but it is a
numerics change and needs static/K3 gating before promotion.

### Updated next steps (2026-06-14, after the microbench ladder)

The memory blocker is now root-caused and a gradient-identical fix is validated
on one GPU. The remaining throughput work splits cleanly:

1. **Unblock + measure 1 node (gated on 8 free GPUs).** Re-run the AMDAHL-033
   every-4th-layer config with `opd_streaming_lowmem: true` and
   `opd_vocab_chunk_size: 8192` (candidate `AMDAHL-034`). Expect the lm-head OOM
   to clear (~4-6 GB reclaimed). Capture a real 1-node executed MFU number — this
   is the first true 1-node fwd/bwd data point.
2. **Grow M to climb the MoE curve.** Once 1 node runs, push real tokens/rank up
   (denser packing seq_len so packed-row count ≈ DP size → no dummy ranks; larger
   `opd_microbatch_size`/`opd_prepare_batch_size`). Target M≥256-512 (8k-16k
   tok/rank) where the grouped expert GEMM is 30-40% MFU. Verify with
   `microbench_moe_gemm.py` for the chosen token count first (free), then in the
   trainer.
3. **Trim non-GEMM wall time.** `clear_gradients_s≈0.59 s` (13% of wall) is pure
   overhead — check for `set_to_none`/fused zeroing. Teacher forward (0.85 s,
   3.3× student tokens) is recomputed each step; the OPRD cache is meant to avoid
   this — confirm it is actually hit, not recomputed.
4. **Only then** confirm a 1-node win on the 4-node trainer-only replay, then a
   short full OPD promotion gate if the batch shape changed.

Do **not** treat `opd_streaming_lowmem` as a numerics change — it is bit-exact vs
the current fp32 path. The in-place-`.grad` mode of the lowmem Function is OFF in
the `model_runner` wiring (returns a native-dtype grad, standard autograd); flip
it on only after checking FSDP/DTensor `.grad` semantics.

## Do Not Spend The Next Cycle On

- Adding trainer nodes.
- Full OPD science launches as the first test.
- KL/top-k loss micro-optimizations; KL compute was tiny in replay.
- Pause-token trimming as a throughput fix; that is a science recipe change.
- No-checkpoint, EP=1, or pack2304 promotion without new evidence; those have
  measured negative or misleading results above.

## Artifacts

- Capture payload:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json`
- Baseline replay output:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-serveronly-6x.jsonl`
- Breakdown replay output:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl021-breakdown-sortedmetrics-serveronly-5x.jsonl`
- Denominator audit:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_20260613.json`
- Replay script:
  `experiments/opd_profile/scripts/replay_forward_backward_capture.py`
- Denominator audit script:
  `experiments/opd_profile/scripts/audit_forward_backward_denominator.py`
- Candidate YAMLs:
  `experiments/opd_profile/autoresearch/candidates/AMDAHL-021-*.yaml`
  through `AMDAHL-034-*.yaml`

### 1-node microbench ladder artifacts (2026-06-14)

- MoE GEMM root-cause bench:
  `experiments/opd_profile/scripts/microbench_moe_gemm.py`
- lm-head streaming-KL memory/MFU bench:
  `experiments/opd_profile/scripts/microbench_lmhead_kl.py`
- lowmem-fix numerics+memory validator:
  `experiments/opd_profile/scripts/validate_lmhead_kl_lowmem.py`
- Microbench result JSONs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/microbench/`
  (`moe_gemm_ep8_20260614.json`, `moe_gemm_ep1_20260614.json`,
  `lmhead_kl_fp32_20260614.json`, `lmhead_kl_bf16_20260614.json`,
  `lmhead_kl_fp32_preexist_20260614.json`)
- Engine fix branch: `xorl-internal` `throughput/opd-lmhead-moe-gemm-20260614`
  off `origin/apanda-dev @ 609bed76` (worktree
  `/home/apanda/xorl-opd-throughput-20260614`). Files:
  `src/xorl/ops/loss/opd_streaming_kl.py` (`streaming_reverse_kl_lowmem_function`),
  `src/xorl/ops/loss/opd_loss.py` (`streaming_lowmem` param),
  `src/xorl/server/runner/model_runner.py` (`opd_streaming_lowmem` plumbing),
  `tests/ops/loss/test_opd_loss.py` (`test_opd_streaming_lowmem_matches_streaming`).
- Run recipe (single GPU):
  `CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=/home/apanda/xorl-opd-throughput-20260614/src
  /home/apanda/xorl-internal/.venv/bin/python experiments/opd_profile/scripts/<bench>.py`

## Next Agent Prompt

```text
/goal Work on OPD filler-token throughput from @experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_RUNBOOK.md. Start in /home/apanda/xorl-opd-prefill. Do not launch a full OPD science run first and do not require 32 H100s for the inner loop. Build/repair the 1-node fwd/bwd microbenchmark ladder: start from the AMDAHL-021 captured payload, preserve the packed/dummy/valid-token shape, reproduce the low executed-MFU behavior on one node, then attack the memory blocker from AMDAHL-029..033, especially streaming-KL/lm-head gradient state. Use the existing 4-node trainer-only replay only as a fidelity check after a 1-node candidate shows a real win. Record every attempt and failed hypothesis back into this runbook.
```
