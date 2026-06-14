# OPD: VERL vs. xorl — metrics & options diff (2026-05-27)

High-level comparison of the OPD implementations in the two codebases,
anchored on VERL **PR #6469** ("[fsdp, megatron, trainer] feat: add top-k
distillation overlap metrics", commit `9c38b8bb`, merged 2026-05-26).

- VERL tree: `/home/apanda/verl` @ tip-of-main (`7c3118e5`).
- xorl tree: `/home/apanda/xorl-opd-mainline-run` @ `codex/opd-mainline-run-20260526`
  (HEAD `6bfab2a4`).

Tagged file paths are repo-relative (i.e. relative to each repo's root).

---

## 1. Headline

- PR #6469 is narrow: it adds **two diagnostic metrics** on VERL's
  `forward_kl_topk` path. No new loss mode, no new CLI flag, no gradient
  change. 7 files, +213/−17 lines.
- xorl OPD has **no top-k loss mode today** — only full-vocab reverse-KL
  (`KL(student‖teacher)` across the whole vocabulary). PR #6469's two new
  metrics are *defined against a top-k subset*, so they have no direct
  analog in xorl. Porting them is cheap in terms of code but requires first
  picking what the "top-k" set even means in a full-vocab loss.

## 2. What PR #6469 actually adds

Two metric keys, both **logging-only** (detached, marked non-differentiable
in the Megatron autograd.Function):

| Key                                             | Definition                                                                                          |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `actor/distillation/overlap_ratio`              | Response-masked mean of `|TopK_teacher ∩ TopK_student| / k`.                                        |
| `actor/distillation/overlap_token_advantage`    | Mean over response positions with ≥1 overlap of `-(p_teacher * (logp_teacher − logp_student))`.     |

Touched files (VERL repo paths; line ranges are post-PR):

- `verl/trainer/distillation/fsdp/losses.py:66–94` — adds `student_topk_ids`,
  builds the overlap mask, returns `overlap_count` + `overlap_token_advantage`.
- `verl/trainer/distillation/megatron/losses.py:97, 175–213, 221–230, 294–311` —
  reconstructs the global student top-k from per-TP-rank candidates (local top-k
  → all_gather → re-topk), all-reduces per-token KL across TP, then computes the
  same diagnostics; the custom autograd.Function returns 5 outputs instead of 3
  and marks the two new ones non-differentiable.
- `verl/trainer/distillation/losses.py:308–349` — treats the two new tensors as
  *optional* on `model_output` (backwards compat), pads to response shape, and
  emits `distillation/overlap_ratio` + `distillation/overlap_token_advantage`
  into the metric dict that the actor wraps under `actor/`.
- `docs/algo/opd.md:589–614, 704–710, 766–771` — documents the new metrics and
  notes they're logging-only.
- Tests: `tests/workers/test_distillation_topk_symmetry_on_cpu.py`,
  `tests/utils/test_special_megatron_kl_loss_tp.py` (overlap-symmetry assertions).
- `README.md` — adds the OPD paper to the project list.

**No new config field** — `distillation_loss.topk` is reused. Existing top-k
runs start emitting the new metrics for free.

## 3. Loss-mode landscape (broader OPD parity)

| Capability                                                          | VERL                                                                                                              | xorl                                                                                                              |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `forward_kl_topk` (top-k KL, used by PR #6469)                      | `verl/trainer/distillation/losses.py:294` (registry) + `fsdp/losses.py:24` + `megatron/losses.py:259`            | **absent**                                                                                                        |
| Reverse-KL estimator family (`k1`, `k2`, `k3`, `abs`, `mse`, `low_var_kl`) | `verl/trainer/distillation/losses.py:359–388` (single dispatch over `loss_mode`)                            | **absent** (only one KL formulation)                                                                              |
| Full-vocab `KL(student‖teacher)` (analytic, no estimator)           | not a separate mode                                                                                               | `src/xorl/ops/loss/opd_loss.py:69` (`opd_loss_function`), backends in `compiled_cross_entropy.py` + `opd_streaming_kl.py` |
| Multi-teacher routing                                               | `distillation.teacher_models{}` + `teacher_key` (`verl/trainer/config/distillation/distillation.yaml:65,113`)     | `teacher_id(s)` tensor in micro-batch, unique-id grouping in `_compute_opd_micro_batch_loss` (`model_runner.py:1790–1972`) |
| Task-reward mixing                                                  | `use_task_rewards`, `distillation_loss_coef` (`distillation.yaml:30,34`)                                          | indirect — single `distillation_loss_weight` (`src/xorl/arguments.py:1239`)                                       |
| Policy-gradient / PPO-style OPD                                     | `use_policy_gradient`, `policy_loss_mode`, `clip_ratio{,_low,_high}` (`distillation.yaml:43–55`)                  | **absent**                                                                                                        |
| Streaming / chunked KL backends                                     | not exposed at config level                                                                                       | `opd_kl_backend` ∈ {`torch_compile`, `streaming`, `tilelang`} (`model_runner.py:1836`)                            |
| Sharded teacher LM-head + async prefetch                            | implicit (handled at engine level)                                                                                | `opd_async_prefetch`, `opd_sharded_head_{cpu,device}_cache` (`model_runner.py:1838, 1841, 1842`)                  |

## 4. Metric-keys diff (the headline)

Final keys as logged by each codebase. VERL's loss function emits
`distillation/...`; the Actor worker prefixes `actor/` before logging.

| Concept                                                   | VERL (post-PR #6469)                                                                                  | xorl                                                                                              |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| KL loss value                                             | `distillation/kl_loss` (estimator path: `losses.py:354`; top-k path emits the loss tensor at `:355`) | `opd_kl` (`src/xorl/ops/loss/opd_loss.py:25`, aggregated in `model_runner.py:1953`)               |
| Student top-k probability mass                            | `distillation/student_mass`, `_min`, `_max` (`losses.py:344–346`)                                     | **absent** (full-vocab, no truncation to measure)                                                 |
| Teacher top-k probability mass                            | `distillation/teacher_mass`, `_min`, `_max` (`losses.py:347–349`)                                     | **absent**                                                                                        |
| **Top-k overlap ratio** (PR #6469)                        | `distillation/overlap_ratio` (`losses.py:331`)                                                        | **absent**                                                                                        |
| **Overlap token advantage** (PR #6469)                    | `distillation/overlap_token_advantage` (`losses.py:334–338`)                                          | **absent**                                                                                        |
| Per-token teacher-weighted KL                             | absent                                                                                                | `opd_weighted_kl` (`opd_loss.py:26`, `model_runner.py:1954`)                                      |
| Mean teacher weight in batch                              | absent                                                                                                | `opd_teacher_weight_mean` (`opd_loss.py:27`, `model_runner.py:1955`)                              |
| #unique teachers contributing to step                     | absent                                                                                                | `opd_num_teachers` (`opd_loss.py:20,30`, `model_runner.py:1956`)                                  |
| Sub-phase timing (prefetch / hidden-fetch / head-prepare / kl-compute / total) | absent                                                                                       | `opd_profile_*_ms` (`model_runner.py:1559–1567`, written at `1961–1965`)                          |
| Valid-token count                                         | response-masked everywhere (not logged as a key)                                                      | `valid_tokens` (`opd_loss.py:16`, `model_runner.py:1952`)                                         |

**Net:** VERL leans on top-k *distributional* diagnostics (mass + overlap).
xorl leans on *teacher-weighting* diagnostics (weighted KL, weight mean,
#teachers) and **sub-phase wall-time profiling** that VERL does not have.

## 5. Config-knob diff

### VERL — `DistillationLossConfig` (`verl/trainer/config/distillation/distillation.yaml:17–55`; dataclass `verl.workers.config.DistillationLossConfig`)

| Field                              | Default     | Notes                                                                  |
|------------------------------------|-------------|------------------------------------------------------------------------|
| `loss_mode`                        | `k3`        | Selects estimator (k1/k2/k3/abs/mse/low_var_kl) or `forward_kl_topk`. |
| `topk`                             | `32`        | k for the top-k path. PR #6469's metrics reuse this.                  |
| `use_task_rewards`                 | `true`      | Mix RL task reward + distillation loss.                               |
| `distillation_loss_coef`           | `1.0`       | Coef on distillation term when mixing.                                |
| `loss_max_clamp`                   | `null`      | Per-token loss ceiling.                                               |
| `log_prob_min_clamp`               | `null`      | Stability clamp on `log p`/`log q` differences.                       |
| `use_policy_gradient`              | `false`     | Treat distillation as a reward signal.                                |
| `policy_loss_mode`                 | `"vanilla"` | PG variant when above is true.                                        |
| `clip_ratio` / `_low` / `_high`    | `0.2`       | PPO clipping when in PG mode.                                         |

Outer namespace: `teacher_models{}` map + `teacher_key`
(`distillation.yaml:65, 113`).

### xorl — `DistillationArguments` (`src/xorl/arguments.py:1222–1247`)

| Field                       | Default        | Notes                                                                |
|-----------------------------|----------------|----------------------------------------------------------------------|
| `enable_distillation`       | `False`        | Master toggle.                                                       |
| `teacher_model_path`        | `None`         | HF ID / local path.                                                  |
| `distillation_loss_type`    | `"forward_kl"` | **Declared but unused** in current OPD path (full-vocab reverse-KL). |
| `distillation_loss_weight`  | `1.0`          | Single scalar weight.                                                |
| `stream_num_workers`        | `32`           | I/O workers for cached teacher activations.                          |

### xorl — runtime params via `loss_fn_params` dict (`src/xorl/server/runner/model_runner.py:1836–1844, 1935–1937`)

Not surfaced as a typed config dataclass — threaded loosely through
`forward_backward(..., loss_fn="opd_loss", loss_fn_params=...)`:

- `opd_kl_backend` / `kl_backend` ∈ {`torch_compile`, `streaming`, `tilelang`} (default `torch_compile`)
- `opd_vocab_chunk_size` / `vocab_chunk_size` (default `32768`)
- `num_chunks` (torch.compile auto-chunker; default 8)
- `teacher_lm_head_fp32` (default `True`)
- `teacher_id` / `teacher_ids` (int or Tensor)
- `teacher_weight` / `teacher_weights` (per-token scalars)
- `opd_async_prefetch` (default `True`)
- `opd_sharded_head_cpu_cache` (default `True`), `opd_sharded_head_device_cache` (default `False`)
- `opd_profile_timings` (default `False`), `opd_profile_sync_cuda` (default `False`)

## 6. Why the overlap metrics don't carry over directly

- VERL's overlap statistic measures how well student top-k tracks teacher
  top-k. That diagnostic is **most meaningful when the loss itself is
  computed over a top-k subset** — i.e. when the loss already pretends the
  rest of the vocab doesn't exist. The metric tells you whether the student
  is allocating its top-k slots to the same tokens the teacher does.
- xorl's `opd_loss_function` (`src/xorl/ops/loss/opd_loss.py:69`) is
  full-vocabulary: every token contributes to KL. Computing top-k overlap
  would still be cheap (one `topk()` on each side, one `intersection` mask)
  but the interpretation is different — it's a *post-hoc* distributional
  agreement metric, not "are we even looking at the same tokens during
  optimization?".
- The closest existing xorl analog is `opd_teacher_weight_mean`, but that
  characterizes the *weighting scheme*, not distributional alignment.

## 7. Cheap port path (out of scope, noted for follow-up)

If we want PR #6469-style overlap diagnostics in xorl:

1. In `opd_loss_function` (`src/xorl/ops/loss/opd_loss.py:69`), after computing
   `student_log_probs` and `teacher_log_probs`, take `torch.topk(...).indices`
   from each at the same `k` and build the overlap mask.
2. Extend `OPDLossMetrics` (`opd_loss.py:14–31`) with two scalars and update
   `to_dict()`.
3. Aggregate per-teacher contributions in `_compute_opd_micro_batch_loss`
   (`model_runner.py:1945–1957`) the same way `kl_sum` and `weighted_kl_metric`
   are aggregated.
4. The streaming and tilelang backends
   (`src/xorl/ops/loss/opd_streaming_kl.py`) currently *don't materialize full
   softmax* — they save only normalization stats. A top-k extractor would need
   to be added to the streaming forward (cheap: `topk` per vocab chunk →
   global merge), or the metric could be computed only under the
   `torch_compile` backend and reported as `0.0` otherwise.
5. **Per [[feedback-dist-allreduce-dict-keyed]]**: any new metric key must be
   present in *every* return path of `OPDLossMetrics.to_dict()` with a default
   value (don't conditionally omit), otherwise dict-keyed `dist.all_reduce`
   can deadlock when ranks have different key sets.

## 8. References

- VERL PR #6469 commit: `9c38b8bb1876a81273d76de3e79328b2dd2b7b32`
  ("[fsdp, megatron, trainer] feat: add top-k distillation overlap metrics").
- VERL OPD doc: `docs/algo/opd.md` (last updated 05/26/2026 in this PR).
- Paper cited by PR #6469: Li, Yaxuan, et al. "Rethinking On-Policy
  Distillation of Large Language Models", arXiv:2604.13016, 2026.
- xorl OPD source of truth: `src/xorl/ops/loss/opd_loss.py`,
  `src/xorl/ops/loss/compiled_cross_entropy.py`,
  `src/xorl/ops/loss/opd_streaming_kl.py`,
  `src/xorl/server/runner/model_runner.py:1790–1972`.
