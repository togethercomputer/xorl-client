# OPD Old vs Clean Profiling Gap

## Inputs

- Old run: `/home/apanda/xorl-internal-dispatch-run/experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdp2-052500/20260525T150718Z-er-opdp2-052500-trainer-head-vdnxw`
- Clean run: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260527T032328Z-er-opdmain-052620-trainer-head-8klbk`
- Old window: `56:101` (46 OPD rows)
- Clean window: `1` (1 OPD rows)
- Server records parsed: old `6529` runner / `6529` executor; clean `128` runner / `128` executor.

## Main Finding

Clean is -27.147s slower per selected OPD step. The direct `forward_backward_s` delta is -67.100s, while sync is -0.424s and student sampling is -301.226s. Teacher forward compute is also 188.512s slower, but much of teacher preparation overlaps the trainer work.

The server microbatch timing shows a broad per-microbatch slowdown rather than one stalled request. Runner median is 5.515s clean vs 3.290s old; executor median is 5.561s clean vs 6.667s old.

## OPD Profile Means

| Field | Old mean | Clean mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| `step_total_s` | 464.895 | 437.749 | -27.147 | -5.8% |
| `forward_backward_s` | 436.506 | 369.406 | -67.100 | -15.4% |
| `sync_inference_weights_s` | 6.095 | 5.671 | -0.424 | -7.0% |
| `optim_step_queued_s` | 127.778 | 77.077 | -50.701 | -39.7% |
| `optim_step_s` | 1.848 | 2.823 | 0.976 | 52.8% |
| `teacher_prefill_forward_compute_s` | 44.197 | 232.708 | 188.512 | 426.5% |
| `teacher_prefill_s` | 166.528 | 540.760 | 374.233 | 224.7% |
| `student_sampling_s` | 1150.471 | 849.245 | -301.226 | -26.2% |
| `teacher_hidden_cache_write_s` | 14.176 | 11.905 | -2.271 | -16.0% |
| `prepare_window_s` | 331.022 | 355.001 | 23.979 | 7.2% |
| `student_sampling_output_tokens` | 1550319.196 | 1455084.000 | -95235.196 | -6.1% |
| `teacher_prefill_tokens` | 1755119.196 | 1659884.000 | -95235.196 | -5.4% |
| `valid_tokens` | 1755119.196 | 13279072.000 | 11523952.804 | 656.6% |

## OPD Server Sub-Phase Means (per OPD step, summed across micro-batches)

| Sub-phase | Old mean s | Clean mean s | Delta s |
| --- | ---: | ---: | ---: |
| `opd_profile_forward_loop_total_s` | — | 361.694 | — |
| `opd_profile_forward_compute_s` | — | 137.625 | — |
| `opd_profile_backward_compute_s` | — | 239.614 | — |
| `opd_profile_model_forward_s` | — | 108.120 | — |
| `opd_profile_loss_compute_s` | — | 29.586 | — |
| `opd_profile_prefetch_s` | — | 2.355 | — |
| `opd_profile_hidden_fetch_s` | — | 5.995 | — |
| `opd_profile_head_prepare_s` | — | 0.006 | — |
| `opd_profile_kl_compute_s` | — | 22.816 | — |
| `opd_profile_loss_total_s` | — | 29.477 | — |
| `opd_profile_input_transfer_s` | — | 0.154 | — |
| `opd_profile_per_token_collect_s` | — | 0.000 | — |
| `opd_profile_deferred_k3_s` | — | 0.000 | — |
| `opd_profile_loss_report_allreduce_s` | — | 0.253 | — |
| `opd_profile_sp_grad_sync_s` | — | 0.001 | — |
| `opd_profile_metric_finalize_s` | — | 0.590 | — |
| `opd_profile_final_synchronize_s` | — | 0.001 | — |

## Server Microbatch Timings

Runner `time=` records:

| Runner metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 223.831 | 362.800 | 138.969 |
| `mean` | 3.497 | 5.669 | 2.171 |
| `median` | 3.290 | 5.515 | 2.225 |
| `p90` | 4.040 | 6.170 | 2.130 |
| `p95` | 4.490 | 6.270 | 1.780 |
| `max` | 5.810 | 9.530 | 3.720 |
| `gt10` | 0.000 | 0.000 | 0.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `total=` records:

| Executor metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 436.275 | 369.272 | -67.003 |
| `mean` | 6.817 | 5.770 | -1.047 |
| `median` | 6.667 | 5.561 | -1.106 |
| `p90` | 7.454 | 6.211 | -1.243 |
| `p95` | 7.982 | 6.314 | -1.668 |
| `max` | 10.327 | 12.873 | 2.545 |
| `gt10` | 1.000 | 1.000 | 0.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `backend=` records:

| Executor backend metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 435.933 | 368.971 | -66.962 |
| `mean` | 6.811 | 5.765 | -1.046 |
| `median` | 6.662 | 5.557 | -1.105 |
| `p90` | 7.449 | 6.206 | -1.244 |
| `p95` | 7.977 | 6.310 | -1.667 |
| `max` | 9.481 | 12.865 | 3.384 |
| `gt10` | 0.000 | 1.000 | 1.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

## Slowest Runner Microbatches

| Run | OPD step | Server step | Time s | Logged tokens | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| old-stable | 64 | 4116 | 5.810 | 27165 | 0.0018 |
| old-stable | 68 | 4353 | 5.800 | 27520 | 0.0021 |
| old-stable | 95 | 6115 | 5.770 | 26987 | 0.0017 |
| old-stable | 101 | 6502 | 5.690 | 27409 | 0.0019 |
| old-stable | 89 | 5725 | 5.670 | 27591 | 0.0015 |
| run2-prefetch-defrag | 1 | 64 | 9.530 | 209640 | 0.1861 |
| run2-prefetch-defrag | 1 | 109 | 7.650 | 206440 | 0.1889 |
| run2-prefetch-defrag | 1 | 86 | 7.180 | 209640 | 0.1940 |
| run2-prefetch-defrag | 1 | 115 | 6.270 | 199032 | 0.1979 |
| run2-prefetch-defrag | 1 | 76 | 6.180 | 210480 | 0.1825 |

## Three-Run Summary (step_total, forward_backward — step 1 / non-warmup row)

| Run | step_total s | fb_s | loop_total | model_fwd | bwd | outer overhead | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old-stable (mean 56:101) | 464.9 | 436.5 | — | — | — | ~213 | reference fast run |
| run1-baseline (mainline) | 662.1 | 627.9 | 402.7 | 130.6 | 248.8 | 225.2 | first run with new sub-phase instrumentation |
| run2-prefetch-defrag | 437.7 | 369.4 | 361.7 | 108.1 | 239.6 | 7.7 | enable_forward_prefetch=true + defrag gating |

Run 2 is now **27s faster than the old fast snapshot** on step_total and **67s faster** on forward_backward.

## What Changed Between Run 1 and Run 2

Two fixes landed locally; both are recorded against the same codebase tip + new run dir.

1. **Per-call defrag gating** (`model_runner.py`, `weight_sync/handler.py`):
   - `gc.collect()` + `torch.cuda.empty_cache()` at the top of every `forward_backward` (cost ~300ms/call × 64 microbatches/step = ~19s/step alone) was reduced to a once-per-step defrag gated by `_allocator_dirty`. The flag is set at init and after the weight-sync handler completes, and cleared the first time forward_backward defrags after that. Optim_step now also runs `gc.collect()` (it already had `empty_cache()`), absorbing what used to be the next-step defrag.
   - Net result: per-call outer overhead dropped from **225s → 8s/step** (the dominant Run 1 → Run 2 win).

2. **`enable_forward_prefetch: true`** in both OPD trainer and teacher configs:
   - `ServerArguments.enable_forward_prefetch` defaults to `False`, overriding `model_builder`'s `True` default. Same pattern documented at [[glm5-forward-prefetch-regret]] cost GLM-5 ~19%. Setting it true in the student config recovers:
     - model_forward: 130.6 → 108.1 s/step (-17%)
     - loss_compute: 50.2 → 29.6 s/step (-41%, likely benefiting from better stream alignment)
   - The teacher (1-node FSDP=8) doesn't benefit much from prefetch because intra-node all-gather is already overlapped via NVLink. Teacher's forward is still 5× slower than the old snapshot (232 vs 44 s/step) — separate remaining regression, but not the step-time bottleneck.

## Remaining Bottlenecks for Future Work

- **Backward dominates** at 239.6 s/step (3.74 s/microbatch). 65% of forward_backward. FSDP reduce-scatter and activation recompute (`recompute_before_dispatch`) sit in this bucket.
- **Teacher forward is still 5× slower** vs old snapshot (232 vs 44 s/step). Doesn't block step time today (overlaps with FB), but if FB drops below prepare_window (~355 s) the teacher becomes the critical path. Unexplained by prefetch.
- Optim_step queued time is 77s — down from 297s in Run 1, still higher than old 127s. Likely a downstream effect of overlapping client futures with FB; not the primary lever.

## Interpretation

- Student sampling and P2P sync are not the current bottleneck; both are faster in the clean run.
- The large wall-time gap tracks trainer forward/backward and a slower teacher hidden-cache forward path.
- The selected trainer config topology is effectively the same: 4 trainer nodes, FSDP=32, EP=8, packing on, and `recompute_before_dispatch` in both old and clean logs.
- Logged `valid_tokens` and server `tokens` are not directly comparable across the two codepaths: clean logs roughly 8x the teacher token count while old logs local token counts. Packed batch sizes in server logs remain around 28k tokens, so use wall-clock timings for this comparison.
- When sub-phase rows are present, divide each `opd_profile_*` by `num_microbatches` (64 here) to get the per-microbatch wall-clock contribution. Forward+backward MAX-reduced sums can over-count the loop wall by ~0.4s/call because each rank may be slowest in a different phase.

