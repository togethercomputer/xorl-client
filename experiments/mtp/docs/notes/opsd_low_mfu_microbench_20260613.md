# Why server-RL (OPSD) trainer MFU is 0.1% — isolated by microbenchmark (2026-06-13)

**Question:** the OPSD-Wordle trainer (Qwen3.6-35B-A3B, EP=8, 8×H100, full-weight server-mode)
runs at ~0.13% MFU. Is that the GEMM physics (small batches → tiny memory-bound kernels) or the
server pipeline? You can't tell from the live stack because RB/chunk/packing all interact with the
orchestration.

**Method (cheap, faithful):** the profile already proved fb is 91–95% of the step, the loss/KL is
0.5%, and EP=8 is intra-node. So the bottleneck is reproducible with **1 node, bare
`xorl.cli.train` (pure fwd+bwd+optim), static synthetic data, EP=8 — and ZERO weight-sync /
sampler / teacher / dispatch / HTTP.** This replicates their *exact trainer topology* (8×H100,
EP=8) minus the orchestration the profile already showed is hidden/negligible. Then sweep one
variable: **tokens per rank per forward** (micro_batch × seq, seq fixed at 1024), recompute_full,
quack_linear, balanced synthetic routing.

## Result: a clean two-factor decomposition of the 0.1%

Bare local fwd_bwd, EP=8, 1 node, MFU vs tokens/rank/forward:

| tokens/rank | bare fwd_bwd MFU | tok/s |
|---:|---:|---:|
| 1,024 (≈ OPSD per-call shape) | **1.85%** | 8.2K |
| 8,192 | 6.6% | 29K |
| 16,384 | **11.85%** | 52K |
| 32,768 | ~10.7% (noisy; within plateau) | ~47K |
| 65,536 | 13.7% | 60.5K |

(≥16k tok/rank sits in an ~11–14% plateau band; the 32k point is within run-to-run variance of 16k/65k, so the knee is ~16k.)

(All at ~48.9 GB — seq=1024 so memory is weight-dominated; the batch sweep barely moves peak mem.)

This bare-fwd_bwd curve is the **in-model executed-token MFU** — real, and the part of the gap that
is GEMM physics (knee ~16k tok/rank). What it is NOT is a clean measurement of the server's visible
"0.1%". My first draft framed the server-vs-bare gap as a single "~14× pipeline tax"; that was an
over-claim — it mixed three distinct things. **Correction (2026-06-13), after the OPD-port and OPSD
agents replayed real captured batches** (`xorl-apanda-dev-opd-port` denominator audit;
`xorl-opsd-wordle-...` API f/b replay):

The visible "0.1%" decomposes into THREE separable factors, and **which one dominates is
stack-specific** — do not assume the OPSD answer applies to OPD or vice versa:

1. **MFU denominator: executed vs valid tokens.** MFU must state its denominator. On the **OPD-port**
   stack the captured batch executes 107,648 tokens but only **3,049 are valid targets (~3%)** →
   ~1.37% over executed tokens, ~0.039% over valid tokens. The valid/executed ratio explains the gap
   *between those two numbers* — but **1.37% executed-MFU is itself too low and must not be waved
   through as "fine"**: 107,648 / 32 ranks ≈ 3.4k tokens/rank, where the clean microbench gives
   ~3–4%, and the model's healthy large-batch MFU is ~10–14%. So OPD's executed-MFU is ~2–3× below
   even the small-batch curve (cross-node comm at 4 nodes + dummy-fill rank-imbalance + tokens
   fragmented into many sub-forwards) and ~7–10× below the big-batch regime (too few tokens/forward).
   The "0.04%" is three stacked factors — denominator (~3% valid) × executed-MFU-below-curve (~2–3×)
   × small-batch ceiling (~3–4% vs ~10–14%) — **not "just accounting."** (On **OPSD-Wordle** the
   denominator is the opposite: `valid/train=99.6%`.)
2. **Dispatcher dummy-fill.** With `enable_packing:false` the orchestrator makes one micro-batch per
   sample, and the dispatcher slices micro-batches across ranks; if a request has fewer real
   micro-batches than ranks, some ranks run dummy/no-valid work. Executed-token MFU stays decent but
   useful work is diluted. Reducing dummy-fill (rank-occupancy-aware packing) is science-neutral.
3. **Above-model server overhead.** This is what my "pipeline tax" gestured at but did not prove.
   The discriminating evidence: **OPD-port** trainer-only replay (4.46s) ≈ full fwd/bwd (4.21s) → on
   OPD there is **NO** above-model tax. **OPSD-Wordle** API replay (keeps collation/packing, drops
   sampler/teacher/sync) reproduces ~0.01% at dense valid tokens → there IS a real cost there, but it
   has not yet been localized to above-model vs in-model (see "decisive open experiment").

So my microbench's role is narrower than first stated: it bounds the **in-model executed-token MFU
ceiling** (and the ~16k knee). It does NOT by itself explain a stack's visible 0.1% — you must also
measure that stack's valid/executed ratio, its dummy-fill fraction, and whether a bare-tensor replay
of its real microbatch is fast. The number below is the GEMM-physics term only:

- **GEMM-physics headroom ≈ 6–7×** (1.85% → ~12–13.7% executed-token MFU by feeding more tokens per
  forward). Small-batch kernels are memory-bound; more tokens/forward amortizes them. **Knee at
  ~16k tokens/rank** — 16k gets 86% of the 65k ceiling, so the target is ~16k tokens/rank/
   forward, NOT the full 65k. Achievable by coalescing fb calls / ragged packing to ~16k real
   tokens — no need for longer sequences, and **no need to cut pause/think tokens** (those are
   forwarded through the same GEMMs regardless; masking them only zeroes the 0.5% loss term).

## Decisive open experiment: bare-TENSOR replay (the one nobody has run)

Both agents converge here. OPSD's **API** replay keeps the request-processor / collator / packing
layer, so it can't tell whether the cost is *above* the model or *in* it. The missing benchmark:
capture the microbatch tensors **after** server collation/packing and replay those exact tensors
through a **bare** model fwd/bwd (no server entrypoint at all).
- bare-tensor replay **fast** while API replay slow → the cost is **above** the model
  (collation/packing/dispatch) → fix there. (This is OPD's answer already: replay≈full ⇒ no tax.)
- bare-tensor replay **also slow** on the same tensors → it's **in-model** (kernels/comm or the
  dummy-fill *shape*) → my GEMM-physics curve + dummy-fill apply.
My synthetic microbench is one end of this (bare model, *synthetic* dense tensors); the OPSD API
replay is the other (real tensors, *with* the server layer). The bare-real-tensor replay is the
middle that attributes the gap, and it's still unrun for OPSD.

**MTP applicability note (2026-06-13):** the later SingleShot-MTP captured-batch
replay on stack `er-opd-q36-mtp-ss-0605c` showed why the OPSD factors should not
be copied numerically. MTP's q-band + clean-region payloads already execute
large GDN replay work per call, and the slow live f/b chunks mostly warmed away
under trainer-only replay. A first-call torch profiler pointed at FSDP/EP
communication (`record_param_comms`, NCCL all-gather, all-to-all) rather than
OPD KL/top-k or a fixed bad payload. Use this OPSD note as the decomposition
checklist, not as a 14x/6.5x prescription for MTP.

Follow-up MTP `G`-lever probe (2026-06-13 22:12Z): replaying the same two
captured MTP payloads on EP8/1-node/alltoall with base HF weights completed, but
warmed wall time was 26.41s versus 18.21s on the EP32 replay baseline. Per-GPU
MFU proxy improved (actual 3.73% vs 1.59%; useful 0.551% vs 0.328%) and GPU-sec
fell, but wall throughput fell. Direct EP32 step-500 checkpoint resume into EP8
also failed on optimizer-state shard shape mismatch. For MTP, shrinking `G` is
therefore a cost-efficiency/optional future track, not the immediate wall-clock
throughput fix.

Follow-up MTP coalescing probe (2026-06-13 22:30Z): the earlier coalesce-2
failure was DeepEP-specific. Retesting the same decoupled idea with EP32
`alltoall`/triton completed cleanly (`q36mtp-20260613T221209Z-2s1t`, W&B
`qm62obaw`, 8 profile rows), but it still did not beat the measured q-band +
clean 32-chunk baseline. Warm steps 501-506 were 59.37s/step with 27.31s trainer
f/b, versus 32.16s/step and 22.18s trainer f/b for the earlier 32-chunk baseline.
That comparison is not purely architectural: the sampler workload also changed
as the student trained (`commit_len` about 1.07 and sample output about 139 tok/s
versus about 1.80 and 830 tok/s in the earlier capture). For MTP, treat sampler
state as a moving workload variable; promote throughput changes only on
same-checkpoint/same-sampler-state comparisons or trainer-only replays.

## How to actually make the GEMMs bigger (there are only three knobs)

Per-GPU MFU is a function of **tokens/rank/forward**, `T = N / (G × F)`:
- **N** = real tokens processed per optimizer step (Σ rollout lengths in the step),
- **G** = ranks in the DP/EP mesh (GPUs),
- **F** = number of forward_backward calls the step is split into.

The curve above is MFU(T): 1k→1.85%, 8k→6.6%, **16k→11.85% (knee)**, ~13% plateau. To move up the
curve you MUST change N, G, or F — there is no fourth lever, and "small GEMMs" is just `T` being
small. Each stack has been stuck because it changed none of them.

1. **Drop F — pack + coalesce (FREE, science-neutral, do this first).** One sequence per forward
   (`enable_packing:false`) makes F huge and T tiny (~1k → 1.85%). Pack many rollout sequences into
   one varlen sequence with a block-diagonal mask + `cu_seqlens` (per-sequence attention and
   per-sequence loss/advantage are unchanged → **identical science**), targeting ~16k real
   tokens/rank/forward. This alone is the 1.85% → ~12% jump. Set `enable_packing:true` and a packing
   length ~16384; the varlen collator (`sequence_shard_collator`) already supports it and MTP packing
   is validated.
2. **Drop G — use fewer GPUs (FREE, science-neutral, and frees GPUs).** MFU is *per GPU*; throwing
   GPUs at a token-starved RL batch makes T *smaller*. **OPD runs G=32 on N≈108k executed tokens →
   T=3.4k (≈2–3% curve). Same step on G=8 → T=13.5k (the knee, ~11%)** — a ~4–5× per-GPU MFU win that
   also returns 24 GPUs. 32 GPUs for ~3k *valid* tokens/step (≈95 valid tok/rank) is the
   over-provisioning that's been read as a "throughput bug." For OPD the move is **8 GPUs + packing**,
   not optimizing the 32-GPU kernels.
3. **Raise N — bigger rollout batch (a real RL knob, but NOT the pause/think axis).** Accumulate more
   prompts/samples per optimizer step before fwd/bwd. This raises T by gradient accumulation over more
   real sequences; it changes gradient variance (a legitimate batch-size choice) but is independent of
   pause/think-token composition. Use this when packing+fewer-GPUs still leaves T below the knee.

Decision rule: **pack first (always), then shrink G to put T at ~16k, and only raise N if you're
still starved.** OPSD (dense valid) is mostly knob 1; OPD (sparse valid, 32-GPU) is knobs 1+2.

## What this means for the fix (science-preserving)

- **Cutting pause/think tokens is NOT the lever** (and on OPSD it wouldn't even help the denominator
  — valid/train is already 99.6%). Tokens ride the same GEMMs whether masked or not.
- **State the MFU denominator.** Always report executed-token MFU *and* valid-token MFU; they differ
  by the valid/executed ratio. A "0.1%" that's really 1.4%-executed × 3%-valid (OPD) needs a
  different fix (dummy-fill / batch construction) than a "0.1%" with dense valid tokens (OPSD).
- **Per-stack levers:**
  - OPD-style (sparse valid, no above-model tax): reduce dummy-fill / improve rank occupancy and
    valid-token density of the captured batch; the model path is already fine.
  - OPSD-style (dense valid, API replay slow): run the bare-tensor replay first; if the tax is
    above-model, fix collation/packing/dispatch; if in-model, apply the GEMM-physics knee.
  - Either way, the GEMM-physics knee (~16k real tokens/rank/forward via rank-occupancy-aware
    ragged packing) raises executed-token MFU and is science-neutral.
- **Don't bother with:** quack vs triton MoE (kernel-neutral here), the loss/KL (0.5%), or
  no-recompute (the 1k→65k climb is GEMM size, not recompute).

## The reusable cheap microbench (for the OPSD agents)

The synthetic sweep above already localizes the two factors. To attack the pipeline tax on the
*exact real batch* without the live stack:
1. **Capture once** (~10 lines, env-gated): in the server `forward_backward`, `torch.save` one
   step's micro-batch dict (`input_ids`, `position_ids`, `labels`, `cu_seqlens`/mask) to a `.pt`.
   Run ONE OPSD step; never touch sampler/teacher/sync again.
2. **Replay offline**: a bare 1-node harness loads the `.pt`, runs fwd+bwd+optim in a loop, times
   it, splits attention / MoE-dispatch / GDN / comm. Every fix (ragged packing, EP degree, chunk
   size, coalescing) becomes a 5-minute repeatable A/B on the real shape, no orchestration, no
   sampling noise.

## Repro

Configs: `experiments/local_benchmark/q36_65k_sweep/configs/MB_ep8_seq1024_mbs{1,8,16,32,64}.yaml`
(EP=8, dp_shard=8, ulysses=1, seq 1024, recompute_full_layer, quack_linear). 1-node pod
`pod_1node.yaml`; launch via `launch_1node.py`. Node h100-092, 2026-06-13.
