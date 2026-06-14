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

### MoE expert-GEMM size sweep (`microbench_moe_gemm.py`)

The student is an A3B MoE. Each expert is a `[M, 2048] @ [2048, 512]` SwiGLU
FFN where `M_per_expert = tokens_in_EP_group * top_k / E`. Sweeping M
(grouped `bmm` ceiling; `torch._grouped_mm` = engine `native` primitive;
per-expert loop floor), `MFU = achieved / 989 TF` bf16, EP=8 local_experts=32:

| M/expert | grouped `bmm` MFU | `torch._grouped_mm` MFU | per-expert **loop** floor |
|---|---|---|---|
| 8  | 1.4%  | 1.3%  | 0.07% |
| 64 | 11.1% | 10.4% | 0.53% |
| 128 | 19.0% | 18.9% | 1.05% |
| 256 | 28.7% | 30.4% | 2.12% |
| 512 | 37.0% | 39.1% | 4.22% |
| 1024 | 42.2% | 44.0% | 8.58% |
| 2048 | 44.6% | 46.1% | 17.2% |
| 4096 | 45.6% | 48.5% | 32.1% |

Conclusions:

1. **CORRECTION (important): the MoE expert GEMM is NOT the bottleneck at the
   real EP=8 operating point.** `M_per_expert = ep_group_tokens * top_k / E`, and
   with expert parallelism the all-to-all gathers the WHOLE EP group's tokens
   before the expert GEMM. The 1-node trainer config
   (`..._1node_warm009_deepep36.yaml`) is `expert_parallel_size=8`,
   `data_parallel_shard_size=8`, `moe_implementation=quack`, `ep_dispatch=deepep`
   ("deepep36" = `deepep_num_sms=36`, **not** EP=36). The full 64-sample OPRD
   batch is ~71804 real student tokens, so `M ≈ 71804*8/256 ≈ 2244` → **~45% MFU**
   on the expert GEMM. A small `M≈72` only happens at **EP=1** (pure DP), i.e.
   `tokens_per_rank/32` — and EP=1 was already tested and lost. An earlier version
   of this section wrongly mapped the operating point to `M≈72/12%` using EP=1
   semantics; the corrected number is `M≈2244/~45%`.
2. The grouped kernel is already at the ceiling: `torch._grouped_mm` (the engine
   `native` primitive) tracks `bmm` within ~1pp across the whole sweep. So when M
   *is* small, the deficit is small-M, not a fixable kernel inefficiency — and a
   per-expert loop would be catastrophic (0.6% at M=72, ~20× worse). **Do not
   regress onto an eager/loop expert path; do not chase a "better MoE kernel" at
   the EP=8 operating point — there is no MoE-GEMM win to get there.**
3. Where small GEMMs *would* bite: EP=1, or very sparse per-rank batches (few
   real tokens packed). Keep packing dense and EP≥8 and the expert GEMM stays in
   the 40-45% band. The lever "more tokens/expert" is real but already satisfied
   at EP=8 — it is NOT the explanation for ~1.37%.
4. So what IS the ~1.37%? It is **student-model FLOPs ÷ total wall time**, and the
   model GEMMs (MoE ~45%, lm-head KL ~20% — see below) are fine. The 4.46 s wall
   is dominated by work that is NOT student-model FLOPs: teacher forward (0.85 s),
   KL/loss (0.91 s), clear-grad (0.59 s), comms (DeepEP all-to-all), and — at 32
   GPUs — an **85% dummy-rank waste** (5 packed rows spread
   over 32 DP ranks; `dispatcher_dummy_executed_tokens=425088` of `497280`). So
   the path to 10%+ is **NOT** a MoE-GEMM change: kill dummy-rank waste
   (1-node / pack so rows ≈ DP size), cut clear-grad (0.59 s ≈ 13% of wall), and
   stop recomputing the teacher every step (0.85 s; the OPRD cache should serve
   it). Keep packing dense so M stays ≥256; that is already true at EP=8.

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
- *"Small MoE expert GEMMs (M≈72) are the cause of ~1% MFU."* **False at the EP=8
  operating point** (this was my first-pass conclusion, corrected same day). M≈72
  is the EP=1 mapping; with `expert_parallel_size=8` the all-to-all gathers the
  whole ~71804-token OPRD batch, so M≈2244 → ~45% MFU on the expert GEMM. The
  small-GEMM curve is real but the stack does not operate on its bad end. The ~1%
  is student-FLOP ÷ total-wall (teacher fwd + KL + clear-grad + comms) plus the
  32-GPU dummy-rank waste — a phase-mix/occupancy problem, not a GEMM-size one.

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
2. **Get a per-phase wall breakdown on 1 node, then attack the biggest non-GEMM
   phase — NOT the MoE GEMM.** The MoE expert GEMM is ~45% MFU at EP=8 and the
   lm-head KL ~20%; there is no GEMM-size win at this operating point. The wall is
   eaten by: clear-grad (0.59 s ≈ 13% — check `set_to_none`/fused zeroing first,
   cheapest possible win), teacher forward (0.85 s — confirm the OPRD cache is hit
   instead of recomputing the teacher every step), the streaming KL (0.91 s — it
   recomputes teacher logits 3×; caching teacher logsumexp/logits could cut it),
   and comms. Keep packing dense (rows ≈ DP size) so there are no dummy ranks and
   M stays ≥256 — verify with `microbench_moe_gemm.py` for the chosen token count.
3. **Mind the denominator.** "Executed MFU" is student-model FLOPs over the full
   fwd/bwd wall (which includes teacher fwd + KL + clear-grad). Report a phase
   breakdown alongside any MFU number, or the headline will keep looking like a
   "small-GEMM" problem when it is a phase-mix / dummy-waste problem.
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
