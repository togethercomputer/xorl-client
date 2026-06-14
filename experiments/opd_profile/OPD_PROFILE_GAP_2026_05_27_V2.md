# OPD Old vs Clean Profiling Gap

## Inputs

- Old run: `/home/apanda/xorl-internal-dispatch-run/experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdp2-052500/20260525T150718Z-er-opdp2-052500-trainer-head-vdnxw`
- Clean run: `experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/er-opdmain-052620/20260527T011754Z-er-opdmain-052620-trainer-head-ng5m2`
- Old window: `56:101` (46 OPD rows)
- Clean window: `1` (1 OPD rows)
- Server records parsed: old `6529` runner / `6529` executor; clean `128` runner / `128` executor.

## Main Finding

Clean is 197.237s slower per selected OPD step. The direct `forward_backward_s` delta is 191.374s, while sync is 0.883s and student sampling is -298.917s. Teacher forward compute is also 188.830s slower, but much of teacher preparation overlaps the trainer work.

The server microbatch timing shows a broad per-microbatch slowdown rather than one stalled request. Runner median is 6.180s clean vs 3.290s old; executor median is 9.927s clean vs 6.667s old.

## OPD Profile Means

| Field | Old mean | Clean mean | Delta | Delta % |
| --- | ---: | ---: | ---: | ---: |
| `step_total_s` | 464.895 | 662.132 | 197.237 | 42.4% |
| `forward_backward_s` | 436.506 | 627.880 | 191.374 | 43.8% |
| `sync_inference_weights_s` | 6.095 | 6.978 | 0.883 | 14.5% |
| `optim_step_queued_s` | 127.778 | 297.887 | 170.109 | 133.1% |
| `optim_step_s` | 1.848 | 1.542 | -0.306 | -16.6% |
| `teacher_prefill_forward_compute_s` | 44.197 | 233.027 | 188.830 | 427.3% |
| `teacher_prefill_s` | 166.528 | 549.254 | 382.726 | 229.8% |
| `student_sampling_s` | 1150.471 | 851.554 | -298.917 | -26.0% |
| `teacher_hidden_cache_write_s` | 14.176 | 12.014 | -2.161 | -15.2% |
| `prepare_window_s` | 331.022 | 357.266 | 26.245 | 7.9% |
| `student_sampling_output_tokens` | 1550319.196 | 1457024.000 | -93295.196 | -6.0% |
| `teacher_prefill_tokens` | 1755119.196 | 1661824.000 | -93295.196 | -5.3% |
| `valid_tokens` | 1755119.196 | 13294592.000 | 11539472.804 | 657.5% |

## OPD Server Sub-Phase Means (per OPD step, summed across micro-batches)

| Sub-phase | Old mean s | Clean mean s | Delta s |
| --- | ---: | ---: | ---: |
| `opd_profile_forward_loop_total_s` | — | 402.658 | — |
| `opd_profile_forward_compute_s` | — | 180.004 | — |
| `opd_profile_backward_compute_s` | — | 248.787 | — |
| `opd_profile_model_forward_s` | — | 130.561 | — |
| `opd_profile_loss_compute_s` | — | 50.197 | — |
| `opd_profile_prefetch_s` | — | 2.331 | — |
| `opd_profile_hidden_fetch_s` | — | 3.300 | — |
| `opd_profile_head_prepare_s` | — | 0.006 | — |
| `opd_profile_kl_compute_s` | — | 43.443 | — |
| `opd_profile_loss_total_s` | — | 50.079 | — |
| `opd_profile_input_transfer_s` | — | 0.145 | — |
| `opd_profile_per_token_collect_s` | — | 0.000 | — |
| `opd_profile_deferred_k3_s` | — | 0.000 | — |
| `opd_profile_loss_report_allreduce_s` | — | 0.234 | — |
| `opd_profile_sp_grad_sync_s` | — | 0.001 | — |
| `opd_profile_metric_finalize_s` | — | 0.715 | — |
| `opd_profile_final_synchronize_s` | — | 0.001 | — |

## Server Microbatch Timings

Runner `time=` records:

| Runner metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 223.831 | 402.750 | 178.919 |
| `mean` | 3.497 | 6.293 | 2.796 |
| `median` | 3.290 | 6.180 | 2.890 |
| `p90` | 4.040 | 6.720 | 2.680 |
| `p95` | 4.490 | 6.830 | 2.340 |
| `max` | 5.810 | 8.660 | 2.850 |
| `gt10` | 0.000 | 0.000 | 0.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `total=` records:

| Executor metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 436.275 | 627.743 | 191.468 |
| `mean` | 6.817 | 9.808 | 2.992 |
| `median` | 6.667 | 9.927 | 3.260 |
| `p90` | 7.454 | 10.414 | 2.961 |
| `p95` | 7.982 | 10.545 | 2.563 |
| `max` | 10.327 | 11.160 | 0.833 |
| `gt10` | 1.000 | 23.000 | 22.000 |
| `gt20` | 0.000 | 0.000 | 0.000 |
| `gt40` | 0.000 | 0.000 | 0.000 |

Executor `backend=` records:

| Executor backend metric | Old | Clean | Delta |
| --- | ---: | ---: | ---: |
| `mean_step_sum` | 435.933 | 627.455 | 191.522 |
| `mean` | 6.811 | 9.804 | 2.993 |
| `median` | 6.662 | 9.922 | 3.260 |
| `p90` | 7.449 | 10.410 | 2.961 |
| `p95` | 7.977 | 10.540 | 2.563 |
| `max` | 9.481 | 11.154 | 1.673 |
| `gt10` | 0.000 | 21.000 | 21.000 |
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
| clean-mainline-v2 | 1 | 64 | 8.660 | 209336 | 0.1965 |
| clean-mainline-v2 | 1 | 65 | 8.560 | 207032 | 0.1885 |
| clean-mainline-v2 | 1 | 78 | 7.440 | 206856 | 0.1935 |
| clean-mainline-v2 | 1 | 112 | 6.830 | 211728 | 0.1874 |
| clean-mainline-v2 | 1 | 109 | 6.820 | 203616 | 0.2015 |

## Per-Microbatch Breakdown (clean, 64 microbatches per step, profile_sync_cuda=true)

Divide each clean sub-phase by 64 to get per-call (per-microbatch HTTP forward_backward) contribution:

| Bucket | Clean s/call | Notes |
| --- | ---: | --- |
| Executor `total=` mean | 9.808 | full server endpoint, includes serialization + gc.collect + cuda.empty_cache + adapter switch + validate_token_ids + sync |
| Runner `time=` mean | 6.293 | forward_loop wall, MAX over ranks |
| `forward_loop_total_s / 64` | 6.291 | matches runner mean — same measurement |
| `forward_compute_s / 64` | 2.813 | _compute_micro_batch_loss (forward only) |
| ↳ `model_forward_s / 64` | 2.040 | `self.model(**model_inputs)` |
| ↳ `loss_compute_s / 64` | 0.784 | _compute_opd_micro_batch_loss |
|   ↳ `kl_compute_s / 64` | 0.679 | opd_loss_function (KL, torch_compile backend) |
|   ↳ `hidden_fetch_s / 64` | 0.052 | teacher hidden cache read |
|   ↳ `prefetch_s / 64` | 0.036 | head/cache prefetch overlap |
|   ↳ `head_prepare_s / 64` | 0.0001 | teacher LM head fetch |
| `backward_compute_s / 64` | 3.887 | `loss.backward()`, includes FSDP all-gather + reduce-scatter |
| `loss_report_allreduce_s / 64` | 0.004 | post-backward loss reduce |
| `metric_finalize_s / 64` | 0.011 | end-of-loop OPD metric allreduce (now empty-rank safe) |
| `input_transfer_s / 64` | 0.002 | host→device pinned transfers |
| Sum forward+backward+loss_report+metric+input | 6.717 | (MAX-reduce double-count; `forward_loop_total / 64 = 6.291`) |
| Executor minus runner | 3.515 | per-call overhead OUTSIDE `_forward_loop` |

The +2.99s/call executor delta vs old (9.81 - 6.82) is dominated by the +2.80s/call runner delta (6.29 - 3.50). Only ~0.19s/call of the slowdown is in the per-call outer overhead.

## Where The +2.80s/Microbatch Runner Delta Lives

The old fast snapshot did not record per-phase sub-times, so per-phase deltas are inferred. What we know:

- The clean forward_compute MAX-sum (2.81 s/call) and backward_compute MAX-sum (3.89 s/call) cover the entire forward_loop wall (6.29 s/call) with the expected +0.4 s "MAX-sums-exceed-loop-wall" margin from different ranks being slowest in different phases.
- KL compute is small (0.68 s/call) and not a leading suspect.
- `model_forward_s` per call (2.04 s) is 72% of `forward_compute_s` per call (2.81 s).
- Backward per call (3.89 s) is the single biggest bucket.
- Teacher prefill forward compute is also 4.3× slower (44.2 → 233.0 s/step). The teacher uses `_forward_teacher_hidden_cache`, a distinct path from `_forward_loop`, but invokes the same `self.model(...)` forward. The teacher slowdown corroborates that the model forward itself is slower, not OPD-specific code.

Most likely contributors to investigate next, in order:

1. **Backward path differences** (+~1.5–1.8 s/call inferred): MoE recompute mode (`recompute_before_dispatch`), R3 routing handler attachment (now in `post_init`), FSDP all-gather/reduce-scatter scheduling.
2. **Model forward path differences** (+~1.2–1.4 s/call inferred): same MoE and R3 considerations on the forward side; activation offloader merge; any new pre-/post-forward hook overhead.
3. **DeepEP accum dtype / chunked scatter** (#289 commits): could shift dispatch/combine time independently of recompute mode.

The per-microbatch view rules out:

- KL backend choice (already torch_compile in both effective paths, 0.68 s/call now is fine).
- Sync, optim, sampling, teacher hidden-cache write — all flat or faster.
- Per-call HTTP/dispatch overhead — only 0.19 s/call of the slowdown lives outside `_forward_loop`.

## Interpretation

- Student sampling and P2P sync are not the current bottleneck; both are faster in the clean run.
- The large wall-time gap tracks trainer forward/backward and a slower teacher hidden-cache forward path.
- The selected trainer config topology is effectively the same: 4 trainer nodes, FSDP=32, EP=8, packing on, and `recompute_before_dispatch` in both old and clean logs.
- Logged `valid_tokens` and server `tokens` are not directly comparable across the two codepaths: clean logs roughly 8x the teacher token count while old logs local token counts. Packed batch sizes in server logs remain around 28k tokens, so use wall-clock timings for this comparison.
- When sub-phase rows are present, divide each `opd_profile_*` by `num_microbatches` (64 here) to get the per-microbatch wall-clock contribution. Forward+backward MAX-reduced sums can over-count the loop wall by ~0.4s/call because each rank may be slowest in a different phase.
- Next concrete experiment: bisect among #289 (DeepEP accum dtype), #291 (MoE act recompute), #308 (R3 in post_init) by reverting each one onto this branch tip and re-running the 1+1 profile run. Comparing model_forward_s and backward_compute_s across runs will localize the regressor.

