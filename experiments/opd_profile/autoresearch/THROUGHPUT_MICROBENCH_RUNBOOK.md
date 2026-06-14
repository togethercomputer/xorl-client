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

The desired cheap target is one-node OPD replay. The direct full-prep64 path is
currently blocked by the full lm-head FSDP all-gather, but smaller rungs fit and
are sufficient for the next inner loop. The failures are useful because they
locate the next work:

| candidate | result |
|---|---|
| AMDAHL-029 | 1-node trainer-side OPRD replay OOMed in `_trainer_teacher_kept_layers` before any replay row. |
| AMDAHL-030 | Moved OPRD hidden cache to SGLang rank-3, but all-40-layer cache saturated teacher memory and wrote no capture. |
| AMDAHL-031 | Every-4th-layer SGLang cache captured successfully, but 1-node trainer replay OOMed before a row. |
| AMDAHL-032 | Selected hooks removed full hidden retention, but all-layer prep64 still OOMed in model forward/final-norm. |
| AMDAHL-033 | Every-4th-layer cache + selected hooks reached streaming-KL backward, then OOMed on full lm-head gradient allocation in `opd_streaming_kl.py`. |
| AMDAHL-033 KL staging retry | Still OOMed because current OPD uses `lm_head_fp32=true`; the weight was already fp32 before grad allocation. |
| AMDAHL-033 root-caused + fixed (2026-06-14) | The blocker was concretely the fp32 lm-head copies + full fp32 `grad_weight` buffer (~5.7 GB at N=3049, 10.7 GB with grad-accum). `streaming_reverse_kl_lowmem` (per-chunk fp32 upcast, native-dtype grad, in-place optional) reclaims 4.6-6.5 GB **gradient-identical** — validated 1-GPU and then confirmed by AMDAHL-034 on the 1-node slot. |
| AMDAHL-034 (2026-06-14, cycle 2) | **RAN** (concurrent agent, ~04:31Z). The lowmem fix **WORKED end-to-end: it cleared the AMDAHL-033 lm-head grad OOM** — the replay got *past* streaming-KL backward into FSDP backward prefetch, then OOMed on a **~970 MiB all-gather with <1 GiB free per rank** (a NEW, different blocker than 033). Replay output `fb_replay/replay-amdahl034-1node-lowmem-serveronly-6x.jsonl` is empty (OOM before any row). → AMDAHL-035 disables FSDP fwd/bwd prefetch (`enable_forward_prefetch` gate) as the narrowest fit probe. The lowmem-fix validation (AMDAHL-033 root-cause) is now confirmed on real 1-node hardware; the remaining 1-node fit gap is FSDP all-gather headroom, not the lm-head grad. |
| AMDAHL-035 full / limit32 (2026-06-14 live slot) | `enable_forward_prefetch=false` disables XORL manual module prefetch lists but does **not** remove PyTorch FSDP2's pre-backward unshard. Full prep64 and `--limit-data 32` still OOM on the same ~970 MiB all-gather. |
| AMDAHL-035 limit8 / limit16 / limit24 | Fit on the 1-node slot and produced phase rows. `limit24` is the current largest fitting rung: `server_forward_backward_s=2.8989`, `forward=0.7920`, `backward=1.1761`, `clear=0.3815`, `model_forward=0.6562`, `loss_compute=0.1354`, `oprd_layer_fetch=0.0859`, `valid_tokens=160`. |
| AMDAHL-035 limit28 | Before the zero-anchor fix, it got past packing (`28` samples -> `10` packed batches, 75.9% utilization, 31104 tokens) and OOMed on `student_weight.float().sum() * 0.0`, which materialized a full fp32 lm-head copy. |
| AMDAHL-036 limit28 anchorfix | Engine commit `e123b782` fixed the full-fp32 zero anchor and passed unit tests, but `limit28` then reached the same ~970 MiB lm-head/FSDP all-gather. This isolates the remaining blocker to the full lm-head module unshard, not the lowmem KL path or zero anchors. |

Failed / discarded hypotheses this cycle (recorded so they are not retried):

- *"clear-grad (0.59 s) can be cut with `set_to_none`/fused zeroing."* **Closed** —
  `model_runner` already calls `zero_grad(set_to_none=True)` everywhere
  (`4020/4024/4256/4260`). The 0.59 s is sync/allocator attribution at the backward
  tail, not zeroing work; there is no fused-zeroing win to get.
- *"The teacher-forward (0.85 s) bucket is wasted recompute that the OPRD cache can
  remove."* **Already removed in AMDAHL-033/034** — with `opd_oprd_cache_backend:
  sglang` the client sets `oprd_trainer_forward = False`
  (`on_policy_distillation.py:1643`) and the trainer fetches the every4 cache. So
  this bucket is not a remaining lever in the every4 config; do not re-attack it.
- *"Fuse the streaming-KL forward to one pass (cache the logsumexp) to cut the KL."*
  **Deprioritized** — algebraically valid (single-pass `KL = C/Z_s − s_logz +
  t_logz`, ~25% of the KL matmul) but NOT bit-exact (~1e-6 drift → needs gating)
  and worth only ~0.03 s (KL is ~0.12 s fwd+bwd in isolation). The backward must
  recompute teacher logits regardless. The KL's value is its memory fix, not latency.

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

## Current 1-Node Target (2026-06-14 Live Slot Validation)

The cheap throughput inner loop is real now: use the reprogrammable slot
`er-opd-q36-35b-slots` and replay the captured fwd/bwd payload through the
trainer API on **one node**. Do not move back to 32 GPUs just to make the payload
fit. The point of this track is to remove the 1-node memory/phase-mix blockers
first, then promote.

Validated live setup:

- Engine: `/home/apanda/xorl-opd-throughput-20260614`, branch
  `throughput/opd-lmhead-moe-gemm-20260614`, PR #373.
- Client: `/home/apanda/xorl-opd-prefill`, branch `exp/opd-prefill`.
- Infra: `/home/apanda/xorl-infra`, branch `opd-battery-consolidation`.
- Slot: `er-opd-q36-35b-slots`, role `trainer-head`.
- Capture:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/amdahl-031-oprd-prep64-deepep36-1node-sglangcache-every4.json`.
- Replay flags:
  `--loss-param opd_streaming_lowmem=true --loss-param opd_vocab_chunk_size=8192`.
- Config:
  `configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_1node_warm009_deepep36_noprefetch.yaml`.

### What The Live Slot Proved

1. `opd_streaming_lowmem` cleared the AMDAHL-033 full lm-head grad OOM on real
   1-node hardware. The remaining failure moved later, into FSDP all-gather.
2. `enable_forward_prefetch=false` is not enough. It disables XORL manual module
   prefetch lists, but PyTorch FSDP2 still performs a pre-backward unshard.
3. Full prep64 and `--limit-data 32` fail on a ~970 MiB all-gather with less than
   1 GiB free per rank. `970 MiB ~= 248320 * 2048 * 2 bytes`, i.e. one full bf16
   lm-head shard/materialization boundary.
4. `--limit-data 24` is the largest fitting rung today and is the right profiling
   rung until the all-gather is removed.
5. `--limit-data 28` exposed and then validated a separate zero-anchor bug:
   `student_weight.float().sum() * 0.0` materialized a full fp32 lm-head copy
   (~1.89 GiB). Engine commit `e123b782` replaces that with scalar-slice fp32
   anchors in `model_runner.py` and `opd_loss.py`; after the fix, `limit28`
   reaches the same ~970 MiB all-gather instead of dying at the anchor.

Fit ladder from the live slot:

| replay | result | mean server fwd/bwd | notes |
|---|---:|---:|---|
| `limit8` | fit | `2.2370 s` | `valid_tokens=68`; forward/backward/clear `0.5749 / 0.8887 / 0.3894 s`. |
| `limit16` | fit | `2.2075 s` | `valid_tokens=105`; forward/backward/clear `0.4925 / 0.8663 / 0.3656 s`. |
| `limit24` | fit | `2.8989 s` | `valid_tokens=160`; packed `24 -> 8` rows, 81.2% utilization, 26624 tokens; forward/backward/clear `0.7920 / 1.1761 / 0.3815 s`. |
| `limit28` pre-anchor | fail | n/a | OOM on full fp32 zero anchor (`student_weight.float().sum()`). |
| `limit28` post-anchor | fail | n/a | OOM on ~970 MiB full lm-head FSDP all-gather. |
| `limit32` | fail | n/a | Same ~970 MiB full lm-head FSDP all-gather. |
| `limit64` / full prep64 | fail | n/a | Same all-gather class; lowmem KL itself is no longer the first OOM. |

### Next Engineering Target

Attack the full lm-head module unshard. The suspected edge is
`_lm_head_forward_anchor(hidden_states, student_lm_head)`: even though streaming
OPD KL consumes the lm-head weight tensor directly, this one-token module forward
exists to preserve graph/FSDP hook ordering and appears to force the full lm-head
unshard. The next candidate should be an opt-in probe that either:

1. skips/replaces `_lm_head_forward_anchor` for the streaming OPD KL path with a
   graph edge that does not call `student_lm_head.forward`, or
2. implements true sharded/vocab-parallel OPD KL weight-gradient handling so the
   full lm-head is never materialized on a rank.

Promotion order:

1. Re-run `limit28`; it must get past the current all-gather.
2. Re-run `limit32`, then full `prep64`.
3. Capture 6 replay rows on full prep64 and report phase breakdown plus executed
   MFU.
4. Only after full prep64 fits and improves on one node, promote to 4-node replay
   to measure scaling/dummy-rank behavior. Do not add nodes to hide the 1-node
   lm-head all-gather.

Do **not** treat `opd_streaming_lowmem` as a numerics change: it is bit-exact vs
the current fp32 path. The in-place-`.grad` mode of the lowmem Function is OFF in
the `model_runner` wiring (returns a native-dtype grad, standard autograd); flip
it on only after checking FSDP/DTensor `.grad` semantics.

Closed/deprioritized levers from this pass:

- `enable_forward_prefetch=false`: tried; does not remove the FSDP2 all-gather.
- clear-grad/fused zeroing: `model_runner` already uses
  `zero_grad(set_to_none=True)`; the bucket is sync/allocator attribution, not
  literal zeroing work.
- teacher-forward recompute: with `opd_oprd_cache_backend: sglang`, the trainer
  fetches every-4th-layer cache instead of running trainer-side teacher forward.
- KL forward fusion: algebraically possible but not bit-exact and worth only
  about 0.03 s in the isolated bench; the KL's value here is the memory fix.
- small MoE GEMM, EP=1, no-checkpoint, pack2304, pause-token trimming: already
  measured negative, irrelevant at EP=8, or science-recipe changes rather than
  throughput fixes.

## 2026-06-14 (cycle 3) — standalone-trainer MFU ceiling, CP verdict, fp32 lm-head memory recipe

**Methodology note (honest).** This cycle did NOT extend the AMDAHL-021 *server-replay*
ladder directly, for two reasons: (a) the real 1-node OPD *server* `forward_backward`
is blocked by a rank-0 **dispatch deadlock** (orchestrator→ZMQ→Rank0Protocol never
broadcasts the fb command at world_size=8; works at ≥2 nodes and for
`register_session`) — root-caused by the live-slot agent via faulthandler, so the
1-node server-replay inner loop cannot run regardless of memory; and (b) the work was
redirected (by the owner) to the MFU-ceiling, CP, and loss-mode-memory questions. So
the ladder here is the **bare `xorl.cli.train` standalone trainer** (Tier-0 ceiling
probe in §7e) on the real Qwen3.6 shapes + a single-GPU lm-head loss-mode microbench.
These directly attack the AMDAHL-029..033 **streaming-KL/lm-head gradient memory**
blocker the goal names. Full per-attempt detail + raw numbers:
`/shared/apanda/tput-mine/NOTES.md`. Bench stacks (mine, isolated): `q36-tput-mine`
(1-node), `q36-tput-2node`, `q36-tput-4node`.

### AMDAHL-021 payload reproduction (re-run this cycle, grounds the memory attack)

Re-ran `audit_forward_backward_denominator.py` on the AMDAHL-021 captured payload
(`fb_replay/amdahl-021-oprd-prep64-deepep36-strictchunk4.json`) — reproduces its exact
packed/dummy/valid shape and the low executed-MFU (output:
`fb_replay/mfu_denominator_audit_cycle3_20260614.json`):

- 22 packed rows @4096 = 73,088 row-padded tokens; +10 dispatcher dummy rows
  (+34,560 tok) → 107,648 dispatcher-executed student tokens.
- **3,049 valid target tokens (2.83% of executed)** → **reconstructed executed-MFU
  = 1.37%** (0.039% valid-scaled), `server_forward_backward_s = 4.46` (4-node replay
  baseline; loss split fwd 1.60 / bwd 1.58 / clear-grad 0.63 / kl 0.046 s).

This is the prescribed "start from the AMDAHL-021 payload, preserve packed/dummy/valid
shape, reproduce the low executed-MFU" step. **The memory attack below is at this exact
payload's lm-head shape:** the streaming-KL/lm-head loss runs over the **3,049 valid
tokens** at `V=248320, H=2048`, which is precisely the `N=3049` used in the
`microbench_kl_backends_mem.py` / `validate_lmhead_kl_lowmem.py` comparisons — so the
lowmem fix and the loss-mode memory table are measured on the AMDAHL-021 lm-head
gradient state (AMDAHL-033), not a synthetic surrogate. The 1-node *server* replay of
this payload remains blocked by the orchestrator→rank0 dispatch deadlock (live-slot
agent), so the lm-head memory blocker is attacked via the single-GPU bare-tensor
microbench at the payload's shape rather than the server replay.

### Reproduced the low 1-node executed-MFU AND root-caused it (it is NOT GEMM-starving)

Standalone-trainer fwd/bwd MFU at the OPD topology (EP=8, quack, deepep36, recompute,
synthetic balanced routing, MFU is recompute-fair), sweeping node count / sharding:

| topology | dp_shard | tokens/rank | peak mem | MFU | note |
|---|---|---|---|---|---|
| 1-node | 8 | 16,384 | **69 GB** | **~5%** | flat 4.3% (8k) → 5.3% (16k); does NOT climb |
| 2-node | 16 | 16,384 | 40 GB | **~10.6%** | `nocp_mbs4` best |
| 4-node | 32 | 16,384 | 28 GB | ~10.0% | `nocp_mbs4` |
| 4-node | 32 | 32,768 | 39 GB | ~10.5% | `nocp_mbs8` — bigger GEMM ~neutral |

**Root cause of the flat ~5% at 1 node: MEMORY PRESSURE, not small GEMMs.** At
dp_shard=8 the peak is 69/80 GB (86%), leaving no headroom for FSDP forward-prefetch
to overlap → comms serialize → ~5% regardless of tokens/rank, async-combine, or
offload. dp_shard≥16 (≥2 nodes) drops peak to ~40 GB → prefetch overlaps → **2×
MFU (~10%)**. The model's clean fwd/bwd ceiling is **~10–10.6%** and it **plateaus**
there (bigger GEMMs/more nodes/CP do not beat it). The lever is *enough FSDP sharding
for memory headroom*, not node count. (Independently corroborated by the Wordle
agent's raw-transformer ~9% server-path peak.)

### CP (context/ulysses parallelism) MONOTONICALLY HURTS here — not the lever

| config | MFU |
|---|---|
| 2-node no-CP mbs4 | ~10.6% |
| 2-node CP=2 mbs4 | ~6–7.5% |
| 2-node CP=2 mbs8 | ~9.4% |
| 4-node CP=2 mbs8 | ~8.9% |
| 4-node CP=4 mbs8 | ~6.4% |

CP splits the sequence, but OPD samples are short (~1138 tokens) so there is nothing
to split; CP only adds all-to-all/all-gather comms. More CP = lower MFU. **Do not use
CP for this short-sequence MoE.** (offload and `deepep_async_combine` also did not
help: ~5.5–7.4% and neutral, respectively.)

### fp32 lm-head memory — which loss mode helps (owner directive: keep fp32)

Keep `lm_head_fp32=true` (reverse-KL accuracy on rare near-certain tokens) and pick
the loss mode that minimizes the 1.89 GiB fp32 `grad_weight` blocker. Single-GPU
microbench (`microbench_kl_backends_mem.py`, V=248320 H=2048 N=3049, fp32):

| reverse-KL backend | peak (no accum) | peak (grad-accum) | numerics vs fp32 baseline |
|---|---|---|---|
| `streaming` (baseline OPD) | 9.63 GB | 13.27 GB | reference |
| **`streaming_lowmem`** (PR #373) | **5.86 GB** | **7.79 GB** | **bit-exact (max\|Δ\|=0)** |
| `compiled` / auto_chunker ("fused") | 16.35 GB | OOM / dtype-error | NOT bit-exact (Δkl=2.3e-3) |

**The "fused" compiled/auto_chunker path is WORSE on memory** (1.7× baseline — it still
does `student_weight.float()` full fp32 copies plus compile/chunk intermediates), is
not bit-exact, and errors in the grad-accum case. **`streaming_lowmem` is the only mode
that keeps lm-head fp32 *bit-exact* while cutting memory.** Do NOT drop to bf16-head
(the Wordle agent's +16% lever) — it carries the rare-token KL accuracy risk the owner
wants to avoid.

**FSDP-safe by ENGINE DESIGN (definitively resolves the prior "lowmem multi-rank
unverified / DTensor slicing may break" caveat).** The lm-head weight is a **full
local tensor at the loss, not a sharded DTensor** — by deliberate engine design, not
luck: `torch_parallelize.py:381-385` groups `norm + lm_head` into one FSDP unit with
`reshard_after_forward=False`, "so that when `norm.forward()` runs FSDP all-gathers
both, and **they stay gathered so external `compute_loss()` can access `lm_head.weight`
without a redundant all-gather**." The vocab-sharded loss path (`fsdp_sharded_lm_head_loss`)
is opt-in and requires CP + `dp_size=1`, which OPD does not use. So the streaming KL
(baseline AND lowmem) always slices a **full local** `student_weight[start:end]` — never
a DTensor. (This also corrects Agent A's suspected mechanism: it is the norm+lm_head
FSDP grouping that keeps the weight gathered, not `weight.float()`.) Additionally the
wired mode uses `inplace_weight_grad=False` → returns a native-dtype grad through
standard autograd (same reduction path as baseline), so even the grad accumulation is
identical. Confirmed bit-exact in this mode (`validate_lmhead_kl_lowmem.py --no-inplace
--preexisting-grad`: max|Δ|=0, saves 5.59 GB). A 4-node trainer-only replay with
`opd_streaming_lowmem=true` is now only optional belt-and-suspenders, not a correctness
gate.

**Memory recipe (keep fp32):** `opd_kl_backend=streaming` + `lm_head_fp32=true` +
`opd_streaming_lowmem=true` (inplace off). Bit-exact, ~5.6 GB saved, FSDP-safe.

### Convergent production recipe toward the ~10% ceiling (this cycle + live-slot + Wordle agents)

≥2 nodes (memory headroom for FSDP prefetch + sidesteps the 1-node fb dispatch
deadlock) · no-CP · dense `mbs4` · lm-head fp32 + `streaming_lowmem` · 0-dummy packing
(pack so rows divide dp_size) · cache the teacher forward. The model ceiling (~10.6%)
is proven; OPD reaches it by removing these server overheads (all engine/recipe work,
targets now precise). Open nice-to-have: confirm `streaming_lowmem` on the 4-node
trainer-only replay (the working multi-rank fb) before promoting PR #373.

## 2026-06-14 (cycle 4, overnight, Agent #1) — lm-head KL engine work

Goal this cycle: engine changes to raise MFU, prove 10% at 4-node. DeepEP
`low_latency_mode` was triaged OUT first: it is decode-oriented (fixed-size RDMA
buffers, FP8, forward-only) and **OPD is FP8-off**, so the all-to-all already
sits at ~19% (Wordle's own FP8-off result), not the bottleneck. Refocused on the
two real OPD-specific costs: the full-vocab KL on the giant lm-head, and the
full-lm-head FSDP all-gather memory blocker.

**lm-head KL backend comparison** (1-GPU microbench, V=248320 H=2048, fp32 lm-head,
`compare_kl_backends.py`): at N=8192 fwd+bwd —
`compiled` 138 ms / **19.1 GB** (materializes full [N,V] logits; `num_chunks` has
no effect — auto_chunker not reducing memory in torch 2.10); `streaming` 303 ms /
10.0 GB; `streaming_lowmem` 311 ms / **9.45 GB** (leanest). vc=32768 beats 65536 on
memory at equal speed. So `compiled` is fast-but-hungry; streaming variants are
slow-but-lean. (Consistent with the runbook note that KL *compute* is tiny at the
current limit24 batch — this matters only once the batch grows.)

**One-pass fused streaming reverse-KL** — landed as PR #374
(branch `throughput/opd-fused-kl`). The streaming forward did 2 vocab passes
(logsumexp + KL); fused to 1 via `KL = A/Z_s - logZ_s + logZ_t`,
`A = Σ exp(s_v-s_max)(s_v-t_v)` (online-accumulable). Backward untouched →
gradient-identical. Validated vs brute-force full-logit reference (kl rel 2.7e-4,
grad rel 1.2e-4): **~23% faster** fwd+bwd (315→243 ms @ N=8192), same 8.0 GB peak.
`validate_onepass_kl.py`. NB: a side win, not the MFU lever (KL compute is small
at limit24).

**Vocab-parallel reverse-KL kernel** — the real lever for the all-gather blocker.
Validated, branch `throughput/opd-vocab-parallel-kl` (`vocab_parallel_reverse_kl.py`
+ multi-process gloo test `test_vocab_parallel_reverse_kl.py`): each rank uses only
its [V/world,H] lm-head shard, computes full-vocab KL via all_reduce of tiny [N,1-3]
stats — **no full lm-head, no full logits anywhere**, and each rank's logits are
world× smaller (one fast matmul, no chunk loop). Matches the full-vocab reference to
float32 precision (kl rel 3e-5, grad rel 1e-6) across 4 ranks.

**Integration constraint discovered (the hard part, NOT yet wired):** under FSDP2
the lm-head shard group ALSO data-shards the tokens — each rank has different tokens
*and* a different vocab slice, which is the wrong layout for vocab-parallel. The fix
is to **all_gather hidden states (cheap: weight 1.89 GB ≫ activations ~72 MB), keep
the weight sharded** — i.e. invert FSDP's gather-weight/shard-activation for the
lm-head only. Then VP-KL applies directly and comms drop ~12×. The remaining risk is
the autograd: the activation all_gather must inject grad only into the LOCAL token
slice (no double-count), and the lm-head shard grad must be supplied to FSDP2 without
fighting its gather/reduce-scatter hooks. This is the next engine step; the kernel is
de-risked. Skip `_lm_head_forward_anchor`'s gather on this path.

**Scale note for the 10%-at-4-node goal:** the blockers differ by scale. 1-node is
gather-blocked (the above). 4-node is UNDER-FILL-blocked: the 64-prompt OPRD batch
packs to ~22 real rows spread over 32 ranks → heavy dummy waste, not the lm-head
gather. The 4-node MFU lever is feeding (bigger prepare batch / more prompts so rows
divide 32), a recipe/batching change — distinct from the 1-node memory fix.

**LIVE 1-node trainer-replay evidence (2026-06-14 ~10:5xZ, er-opd-tput-apanda-0614,
8 GPU, expandable_segments ON, base config recompute_before_dispatch + compiled KL,
amdahl-031 capture).** Steady-state `server_forward_backward` mean over a `--limit-data`
sweep (3 iters, 1 warmup each):

| limit-data | fb wall | model_fwd | backward | kl_compute |
|---|---|---|---|---|
| 8  | 4.93 s | 0.71 s | 1.91 s | 0.000 s |
| 16 | 4.88 s | 0.66 s | 2.09 s | 0.000 s |
| 24 | 5.06 s | 0.71 s | 2.49 s | 0.09 s |

**The fb wall is ~FLAT (4.9–5.1 s) while the data triples (8→24 datums).** The fb is
fixed-overhead / dummy-padding dominated, NOT data-proportional, at these batches: real
datums fill otherwise-padded/dummy slots almost for free. So throughput (datums/s) and
MFU climb ~LINEARLY with batch size up to the memory cap — limit8→24 is ~3× the
throughput at the same wall. Extrapolating, fitting full prep64 (64 datums) at ~the same
~5 s would be ~2.7× the MFU of limit24 just from filling the pack. The cap is the
lm-head FSDP all-gather (full prep64 OOMs; reproduced live — a too-big capture OOM'd
mid-DeepEP-dispatch → rank desync → "DeepEP CPU recv timeout" + CUDA crash). KL compute
is confirmed negligible (0.00–0.09 s), and clear_gradients (~1.28 s, ~25% at limit24) is
a replay artifact (forced per-fb; amortized over grad-accum in real training; already
`set_to_none`). **Net: the #1 1-node MFU lever is raising the memory cap so a bigger
batch fits — exactly what the validated vocab-parallel reverse-KL kernel enables (drop
the full lm-head gather). The integration is the next deep change (task #13).**

## Do Not Spend The Next Cycle On

- Context/ulysses parallelism (CP) for this short-sequence MoE — measured to hurt
  monotonically (2-node CP2 ~7%, 4-node CP4 ~6.4% vs no-CP ~10%).
- Adding trainer nodes before the 1-node lm-head all-gather is fixed.
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
  through `AMDAHL-035-*.yaml`

### cycle-3 artifacts (2026-06-14) — MFU ceiling + loss-mode memory

- KL-backend fp32 memory comparison bench:
  `experiments/opd_profile/scripts/microbench_kl_backends_mem.py` (streaming vs
  streaming_lowmem vs compiled/auto_chunker; the loss-mode-memory table above).
- Standalone-trainer MFU sweep (bare `xorl.cli.train`, OPD topology), configs +
  per-rank run.sh + results under `/shared/apanda/tput-mine/` (`configs/`,
  `NOTES.md` = full per-attempt ledger). Bench stacks: pods `q36-tput-mine`,
  `q36-tput-2node`, `q36-tput-4node` (control dirs `/shared/opd-control/q36-tput-*`).
- Engine worktree with research branch + lowmem cherry-pick (for a real-OPD replay):
  `/home/apanda/xorl-opd-mine-tput` (`throughput/opd-mine-tput-20260614`).

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
  `lmhead_kl_fp32_preexist_20260614.json`). Cycle-2 reproduction (2026-06-14
  ~04:31Z, confirms committed `61c90e6e` reproduces exactly):
  `moe_gemm_ep8_repro_20260614.json`, `lmhead_kl_fp32_repro_20260614.json`,
  `lmhead_kl_fp32_preexist_repro_20260614.json`.
- Engine fix branch: `xorl-internal` `throughput/opd-lmhead-moe-gemm-20260614`
  off `origin/apanda-dev @ 609bed76` (worktree
  `/home/apanda/xorl-opd-throughput-20260614`). Commits:
  `61c90e6e` lowmem streaming reverse-KL,
  `a4b0ec48` streaming diagnostics dtype-match,
  `e123b782` scalar-slice fp32 zero anchors. Files:
  `src/xorl/ops/loss/opd_streaming_kl.py` (`streaming_reverse_kl_lowmem_function`),
  `src/xorl/ops/loss/opd_loss.py` (`streaming_lowmem` param),
  `src/xorl/server/runner/model_runner.py` (`opd_streaming_lowmem` plumbing),
  `tests/ops/loss/test_opd_loss.py` (`test_opd_streaming_lowmem_matches_streaming`).
- Run recipe (single GPU):
  `CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=/home/apanda/xorl-opd-throughput-20260614/src
  /home/apanda/xorl-internal/.venv/bin/python experiments/opd_profile/scripts/<bench>.py`
- Live 1-node replay outputs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit8-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit16-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit24-lowmem-noprefetch-serveronly-3x.jsonl`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl035-limit28-lowmem-noprefetch-serveronly-3x.jsonl`
  (pre-anchor fail), and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-amdahl036-limit28-anchorfix-lowmem-noprefetch-serveronly-3x.jsonl`
  (post-anchor all-gather fail).
- Live 1-node server log dirs:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T043652Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`,
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T044158Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`, and
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260614T045116Z-serveronly-configAMDAHL-035-OPRD-PREP64-1NODE-LOWMEM-KL-FB-NOPREFETCH`.

## Next Agent Prompt

```text
/goal Work on OPD filler-token throughput from @experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_RUNBOOK.md. Start in /home/apanda/xorl-opd-prefill. Use the one-node reprogrammable slot `er-opd-q36-35b-slots` for the inner loop; do not launch a full OPD science run first and do not require 32 H100s to make progress. Current state: PR #373 has the lowmem KL, diagnostics dtype, and zero-anchor fixes; `--limit-data 24` is the largest fitting 1-node replay; `limit28`, `limit32`, and full prep64 fail on a ~970 MiB full lm-head FSDP all-gather after lowmem KL has already cleared the earlier lm-head grad OOM. First target: remove/replace `_lm_head_forward_anchor` for streaming OPD KL or implement true sharded/vocab-parallel OPD KL so `limit28 -> limit32 -> prep64` fit on one node. Do not retry EP=1, no-checkpoint, pack2304, clear-grad zeroing, small MoE GEMM, or KL micro-opts without new evidence. Record every run and failed hypothesis back into this runbook.
```
