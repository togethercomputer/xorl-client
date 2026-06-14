# OPD Old vs Clean Profiling Gap

## Inputs

- Old run: `/home/apanda/xorl-internal-dispatch-run/experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdp2-052500/20260525T150718Z-er-opdp2-052500-trainer-head-vdnxw`
- Clean run: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260526T235245Z-er-opdmain-052620-trainer-head-7rnvm`
- Old window: `56:101` (46 OPD rows)
- Clean window: `1:2` (2 OPD rows)
- Server records parsed: old `6529` runner / `6529` executor; clean `211` runner / `211` executor.

## Main Finding

Clean is 184.971s slower per selected OPD step. The direct `forward_backward_s` delta is 181.670s, while sync is -1.316s and student sampling is -315.338s. Teacher forward compute is also 171.451s slower, but much of teacher preparation overlaps the trainer work.

The server microbatch timing shows a broad per-microbatch slowdown rather than one stalled request. Runner median is 5.990s clean vs 3.290s old; executor median is 9.544s clean vs 6.667s old.

## OPD Profile Means

| Field | Old mean | Clean mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| `step_total_s` | 464.895 | 649.867 | 184.971 | 39.8% |
| `forward_backward_s` | 436.506 | 618.176 | 181.670 | 41.6% |
| `sync_inference_weights_s` | 6.095 | 4.779 | -1.316 | -21.6% |
| `optim_step_queued_s` | 127.778 | 311.007 | 183.229 | 143.4% |
| `optim_step_s` | 1.848 | 2.420 | 0.573 | 31.0% |
| `teacher_prefill_forward_compute_s` | 44.197 | 215.647 | 171.451 | 387.9% |
| `teacher_prefill_s` | 166.528 | 472.737 | 306.210 | 183.9% |
| `student_sampling_s` | 1150.471 | 835.133 | -315.338 | -27.4% |
| `teacher_hidden_cache_write_s` | 14.176 | 11.204 | -2.971 | -21.0% |
| `prepare_window_s` | 331.022 | 334.080 | 3.058 | 0.9% |
| `student_sampling_output_tokens` | 1550319.196 | 1462740.000 | -87579.196 | -5.6% |
| `teacher_prefill_tokens` | 1755119.196 | 1667540.000 | -87579.196 | -5.0% |
| `valid_tokens` | 1755119.196 | 13340320.000 | 11585200.804 | 660.1% |

## Server Microbatch Timings

Runner `time=` records:

| Runner metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 223.831 | 396.370 | 172.539 |
| `mean` | 3.497 | 6.193 | 2.696 |
| `median` | 3.290 | 5.990 | 2.700 |
| `p90` | 4.040 | 6.950 | 2.910 |
| `p95` | 4.490 | 7.400 | 2.910 |
| `max` | 5.810 | 7.770 | 1.960 |
| `gt10` | 0.000 | 0.000 | 0.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `total=` records:

| Executor metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 436.275 | 610.756 | 174.481 |
| `mean` | 6.817 | 9.543 | 2.726 |
| `median` | 6.667 | 9.544 | 2.877 |
| `p90` | 7.454 | 10.285 | 2.831 |
| `p95` | 7.982 | 10.695 | 2.713 |
| `max` | 10.327 | 11.541 | 1.214 |
| `gt10` | 1.000 | 33.000 | 32.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `backend=` records:

| Executor backend metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 435.933 | 610.461 | 174.528 |
| `mean` | 6.811 | 9.538 | 2.727 |
| `median` | 6.662 | 9.539 | 2.877 |
| `p90` | 7.449 | 10.280 | 2.831 |
| `p95` | 7.977 | 10.691 | 2.713 |
| `max` | 9.481 | 11.537 | 2.056 |
| `gt10` | 0.000 | 32.000 | 32.000 |
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
| clean-compile | 2 | 183 | 7.770 | 211936 | 0.2425 |
| clean-compile | 1 | 112 | 7.730 | 206312 | 0.1787 |
| clean-compile | 1 | 69 | 7.710 | 201000 | 0.1987 |
| clean-compile | 2 | 170 | 7.680 | 210232 | 0.2335 |
| clean-compile | 1 | 106 | 7.610 | 202776 | 0.1936 |

## Interpretation

- Student sampling and P2P sync are not the current bottleneck; both are faster in the clean run.
- The large wall-time gap tracks trainer forward/backward and a slower teacher hidden-cache forward path.
- The selected trainer config topology is effectively the same: 4 trainer nodes, FSDP=32, EP=8, packing on, and `recompute_before_dispatch` in both old and clean logs.
- Logged `valid_tokens` and server `tokens` are not directly comparable across the two codepaths: clean logs roughly 8x the teacher token count while old logs local token counts. Packed batch sizes in server logs remain around 28k tokens, so use wall-clock timings for this comparison.
- These pre-instrumentation artifacts do not expose the OPD loss sub-phases in the client profile. The server already computes `opd_profile_*` metrics, but the OPD client only wrote aggregate fwd/bwd wall time. A follow-up xorl-client patch now aggregates those metrics into `opd_profile.jsonl`, so the next profiling run should surface forward, backward, hidden-cache fetch, head prepare, and KL compute timings per OPD step.

