# Qwen3.6-35B-A3B 32×H100 regular-training throughput sweep (2026-06-12)

**Goal:** highest training tok/s/GPU for ordinary (non-MTP) training on the 4
`er-opd-q36-mtp-ss-0605c` trainer pods (research-common-h100-098/094/117/071),
branch `apanda-dev-mtp`, synthetic balanced routing, dummy 8193-token packed data.

## Winner

`experiments/local_benchmark/q36_32gpu_sweep/configs/c14_muon_triton_ep8_mbs10_sms48.yaml`

| | |
|---|---|
| tokens/sec (32 GPU) | **236.1K** steady (steps 3–10) |
| tokens/sec/GPU | **7.38K** |
| step time | 11.10 s (mbs10 × 8193 × 32 ranks = 2.62M tok/step) |
| TFLOPS/GPU / MFU | 145.2 / 14.7% (logical accounting) |
| peak mem | 58.9 GB, 0 allocator retries |
| loss | finite, well-behaved |

Key fields: `moe_implementation: triton` + `ep_dispatch: deepep` (`deepep_num_sms: 48`,
buffer 2 GB), `attn_implementation: flash_attention_3`, EP=8 (`ep_intranode: true`),
FSDP2 `dp_shard=32`, `micro_batch_size: 10`, `recompute_full_layer`, compile on,
**Muon no-momentum** (`muon_momentum: 0.0`, `gram_newton_schulz` + quack NS kernels,
`optimizer_dtype/update_dtype: bf16`, `max_grad_norm: 0`), 8193 sequential packing.
Env: `XORL_MOE_SYNTHETIC_ROUTING=balanced`, `NCCL_SOCKET_IFNAME=^lo,docker`,
NCCL socket tuning (NSOCKS 8 / NTHREADS 4 / BUFFSIZE 8M / GDR 2), expandable_segments
left at the `xorl.cli.train` default (on; no Mooncake in this path).

Launch harness: `experiments/local_benchmark/q36_32gpu_sweep/launch_sweep.py launch <name> <abs-config>`
(writes torchrun run.sh into the 4 trainer control slots; master port must be 29610).
Full results: `experiments/local_benchmark/results/q36_32gpu_sweep/sweep_results.json`.

## Sweep table (steady-state mean, steps 3–10, 12-step runs)

| run | config delta | tok/s | s/step | loss |
|---|---|---:|---:|---|
| **c14** | **muon-nomom, triton MoE, mbs10, sms48** | **236.1K** | 11.10 | ok |
| c10 | muon, triton, mbs10, sms24 | 233.6K | 11.22 | ok |
| c01g | adamw, triton, mbs8, sms24 | 233.5K | 8.98 | ok |
| c12 | muon, triton, mbs8, sms48 | 232.5K | 9.02 | ok |
| c01i | muon, triton, mbs8, sms24 | 230.4K | 9.10 | ok |
| c02v | + FP8 dense-only (excl lm_head+experts) | 174.1K | 12.05 | ok |
| c13 | muon, triton, mbs12 | 133.8K | 23–27 | ok (allocator cliff) |
| c01b/d | quack MoE, adamw | 111–112K | 18.8 | **nan@2** |
| c01f | quack MoE + flex_attention | 65.4K | 32.0 | **nan@2** |
| c02u | FP8 MoE (quack glue, triton_grouped) | 12.3K | 170 | **nan@2** |
| c11 | recompute_before_dispatch, mbs8 | OOM | — | — |
| c02t | FP8 incl lm_head | OOM (fp32 logits) | — | — |

## Findings

1. **`moe_implementation: quack` is broken on this branch for local training at EP=8**:
   backward produces nan grads (step-1 loss finite 10.65, nan from step 2) *and* runs
   2.1× slower than triton (111K vs 233K). The OPD server path on this branch reportedly
   runs quack — the local-trainer EP no-permute path is the suspect. Needs a code fix +
   K3 parity check before quack is usable here.
2. **FP8 training does not pay off for 35B-A3B at this shape (today, this branch).**
   - Full FP8 OOMs via the fp8 lm_head fp32-output logits path.
   - FP8 MoE inherits the quack glue → nan, and all FP8 grouped backends are extremely
     slow at moe_intermediate=512 expert shapes (triton_grouped: 170 s/step).
   - FP8 dense-only is numerically fine but **25% slower** than bf16 (quant overhead >
     GEMM savings at these dense sizes).
3. **Muon without momentum** (gram-NS, quack NS kernels, bf16 states) costs ~2% vs AdamW
   at equal mbs but its smaller optimizer state (12.7 vs 16.2 GB) buys mbs10, which
   nets out ahead. mbs12 fits but collapses (allocator pressure cliff; same pattern as
   the May seed's mbs10 collapse under AdamW).
4. `deepep_num_sms: 48` > 24 by ~1–3%. `recompute_before_dispatch` does not fit at mbs8/8k.
5. flex_attention is ~2.6× slower than FA3 here, and its step-1 loss (2.78) disagrees
   with FA3/seed (10.65) — backend loss mismatch unresolved; a K3 static-trace replay
   should arbitrate which is correct before promoting either for real training.

## Caveats vs the May 261K seed

The May reference (261K, af98064) used `deepep_async_combine: true` when async combine
was honored (now known numerics-unsafe and force-disabled without
`XORL_DEEPEP_UNSAFE_ASYNC_COMBINE=1`), was K3-FAILED on numerics, and ran on a different
vetted node set. Repro attempts on af98064 on *these* nodes failed (2× NCCL
unhandled-cuda-error at init, 1× broadcast-load OOM), so 261K was never validated here.
236.1K with finite loss and sync combine is the best *honest* number measured on this
hardware. This sweep is throughput-only — no K3 gate was run; run one before promoting
the winner as a production training recipe.
