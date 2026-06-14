# MTP OPD Efficiency / MFU Handoff

**Last updated:** 2026-06-09 (evening corrections pass)
**Checkout:** `/home/apanda/xorl-mtp-singleshot-port-20260602`
**Branch:** `codex/mtp-singleshot-port-20260602`
**Stack:** `er-opd-q36-mtp-ss-0605c`
**Workload:** Qwen3.6-35B-A3B SingleShot-MTP OPD, 4-node trainer, FSDP shard=32, EP=32, Quack + DeepEP/SMS24,
coderforge dataset, `conf_threshold=0.3`, `max_new_tokens=256`, `static_padded_seq_len=1536`,
sample-packing `4096`, compile on (linear-attention compile off).

---

## -1. 2026-06-09 EVENING CORRECTIONS — read before acting on §4.7/§5.0

A later code-reading pass found that the topology conclusions below compare **inflated metrics** and
must not be used for promotion decisions.

1. **The OPD batch is replicated across each EP group.** The server dispatcher slices request batches
   by the **`ep_fsdp` coordinate, not dp_shard** (`runner_dispatcher.py::_batch_parallel_rank_and_size`):
   every rank of an EP group processes the same batch slice, and the MoE all-to-all then also duplicates
   routed-expert work once per peer copy. At EP32 (`ep_fsdp=1`) **all 32 ranks process the entire batch**
   (verified: head-rank `opd_singleshot_mtp_batch_tokens` = 49152 = the full 32-prompt step; EP16 = 24576;
   EP8 = 12288).
2. **`valid_tokens` is inflated by the replication factor** (it is an all-reduce SUM of per-rank counts):
   32x at EP32, 16x at EP16, 8x at EP8. Recomputed on *unique* valid tokens, the §4.7 ranking inverts:

   | Config | FB ms per unique valid token |
   |---|---:|
   | EP8 / DeepEP24 (steady, `…170713Z`) | **13.8** |
   | EP16 / DeepEP24 (steady, `…172711Z`) | 24.7 |
   | EP32 / DeepEP24 ("winner", `…174640Z`) | 44.6 |

   **EP8 is the best of the measured legacy configs; the promoted EP32 default is the worst.** EP32's real
   effect (no expert FSDP comm at `ep_fsdp=1`) is real but swamped by 32x redundant work. True hardware MFU
   of the EP32 steady state is ~0.3% (≈2.9 TFLOPS/GPU); goodput (unique-work) MFU is ~0.01%.
3. **The §4.7 "CP=2 is a bust" cell was confounded**: every CP2 run used EP8. CP×FSDP and EP×eFSDP compose
   freely (EP is a separate mesh over the same ranks — `parallel_state.py:741`), and under batch replication
   CP is the only mechanism that divides the replicated batch across ranks. The missing CP2+EP32 cell was
   run (`q36mtp-20260609T200315Z`): its first FB **timed out at 2400s** (vs 164s non-CP warmup) — the
   current CP path (per-layer GDN state handoffs, per-micro-batch overhead) is not viable at this shape
   regardless of the EP confound.
4. **FSDP comm scales with micro-batch count**: no gradient-sync deferral existed, so fp32 gradients
   reduce-scatter on every micro-batch backward (16/step at EP32) and params re-all-gather per micro-batch
   forward+backward (the §4.5 profile's 164 RS / 326 AG calls = 41 FSDP modules × 4 micro-batches × {1,2}).
5. **Fixes landed (flag-gated, defaults preserve old behavior):**
   - `batch_parallel_mode: dp_shard` — distinct batch slice per (dp, cp) coordinate even under EP
     (launcher `--trainer-batch-parallel-mode dp_shard`; pair with `--no-trainer-enable-packing
     --pipeline-chunk-size 32` so the 32-prompt step feeds all 32 ranks). Gradient math is unchanged:
     grads reduce as SUMs and optim_step normalizes by accumulated `global_valid_tokens`, so the
     replication factor cancels identically in both modes. Guarded against the streaming/tilelang
     sharded-teacher-head KL backends, which do require EP-group-identical batches.
   - `fsdp_defer_grad_sync` / `fsdp_defer_reshard_after_backward` — reduce-scatter (and re-all-gather)
     only on the last micro-batch of each FB call.
   - True-MFU instrumentation: `XorlFlopsCounter.estimate_singleshot_mtp_training_tflops` +
     `opd_singleshot_mtp_flops_{actual,useful}_tf`, `…_mfu_{actual,useful}`,
     `…_batch_replication_factor`, `…_flops_regret_ratio` in `opd_profile.jsonl` (replication-aware:
     `useful` counts each unique token once).
   - Sweep driver: `experiments/opd_profile/k8s/q36_singleshot_topology_sweep.py` (`run`/`harvest`),
     summaries include `fb_ms_per_unique_valid`.
6. **Revised priority order:** (a) `dp_shard` batch slicing (up to ~8-32x), (b) FSDP comm amortization
   (deferral + bf16-reduce paired test), (c) pads/recompute/CP mechanics. GDN replay work (§5.2/§5.3) is
   now third-order: stateful replay already collapsed executed GDN tokens to ≈ non-pad tokens.
7. **Measured dp_shard results (3-step smokes, steady-2, clean `OPD pipeline validation succeeded`,
   losses in family):**

   | Config | FB ms per unique valid | vs legacy promoted EP32 |
   |---|---:|---:|
   | EP8 + dp_shard (`q36mtp-20260609T205504Z`) | 3.03 (converged step 1.24) | 14.7x |
   | **EP32 + dp_shard (`q36mtp-20260609T210556Z`)** | **1.57** (best step 1.04) | **28x** |

   Launch shape: `--trainer-batch-parallel-mode dp_shard --no-trainer-enable-packing
   --pipeline-chunk-size 32` (32 prompts -> one micro-batch per rank). With real DP, EP32's zero
   expert-FSDP-comm advantage finally materializes, so EP32 > EP8 again — but only under dp_shard.
   The step bottleneck is now **student sampling** (~37-52s vs FB ~8-16s): next levers are
   `--opd-async-sample-overlap`, sampler throughput flags, and batch scaling
   (`ep32_dpshard_b64_packed`), plus `no_recompute` now that per-rank activations are one
   1536-token micro-batch (`ep32_dpshard_norecompute`).

8. **CP2 timeout root cause (not a fundamental CP property):** the CP GDN executor
   (`gated_deltanet.py::_forward_with_cp_replay_mask`) re-ran `_cp_replay_plan_local_segments` —
   Python loops + `.cpu().tolist()` over the **entire replay plan** — once per GDN layer per
   micro-batch per forward/recompute/backward (~960 invocations at EP32's replicated batch volume),
   which is the >2400s "timeout". Fixed by memoizing the layer-invariant local-segment plan on the
   micro-batch's plan object (`_cp_replay_plan_local_segments_cached`).
   **Both structural CP fixes are now implemented:**
   - *Stateful schedule under CP*: `shard_rollout_replay_stateful_schedule` (mtp/singleshot.py)
     splits the stateful schedule's segments into per-rank owner runs chained through relay slots
     and assigns dependency-ordered execution phases;
     `GatedDeltaNet._forward_with_cp_replay_plan_stateful` executes one packed-varlen call per
     phase per rank with a per-phase slot-table delta SUM all-reduce (`_AllReduceSumAcrossCp`,
     gradient = SUM all-reduce). Falls back to the naive CP executor when the schedule cannot be
     sharded. Gated by the same `replay_plan_use_stateful_prefix_cache` flag.
   - *Pipelined phases instead of serialized whole-rank turns*: within a phase all ranks run
     concurrently; suffix pieces are scheduled at the earliest phase their state allows
     (zero-state suffixes run in phase 0), so suffix replay overlaps later ranks' context rounds.
     Slot flow is provably low-rank -> high-rank (owners are monotonic along chains).
   - Tests: `tests/ops/test_linear_attention_singleshot_mask.py` — transform partition/symmetry
     properties, and forward **and gradient** parity of the CP executor vs dense replay on a
     sequential 2-rank simulation (the monotone slot flow makes single-process simulation exact).
   Live validation cells: `ep32_cp2_dpshard`, then `ep32_cp2` (legacy shape).
9. **Instrumented MFU on the dp_shard winner** (`q36mtp-20260609T212546Z` repeat; its row's
   `batch_replication_factor=32` is a since-fixed telemetry bug — dp_shard mode was misdetected;
   divide its `flops_regret_ratio` by 32 and multiply `mfu_useful` by 32): hardware MFU ≈ 1.2%
   (12.0 TFLOPS/GPU), goodput MFU ≈ 0.2%, true regret ≈ 6.2x — now dominated by padding
   (~2.1x pad expansion at static 1536 unpacked) and launch/comm-bound execution, not replication.
   Legacy path for contrast (`ep32_epfsdp_defer` row): regret 148.6x, goodput MFU 0.0074%.

10. **Amdahl-balanced topology design** (async training on 1-step-stale samples:
    `--opd-async-sample-overlap` makes step ≈ max(T_sample, T_teacher, T_FB)). Measured
    coefficients at b=32 prompts/step on the dp_shard EP32 trainer: T_FB ≈ 8s, T_teacher ≈ 9s
    (2 TP=2 replicas — already balanced), T_sample ≈ 43-50s on 2 TP=2 student replicas at
    ~68 new tok/s/replica. Balance condition: `n_sampler_replicas ≈ new_tok_per_step /
    (replica_rate x T_FB)`.
    - At the current per-replica rate: ~11 TP=2 replicas (22 GPUs) against the 32-GPU trainer →
      ~12.5 unique sup tok/s/GPU end-to-end (vs ~2.2 today, sync with 2 replicas).
    - The cheaper lever is per-replica rate: students run `--disable-cuda-graph
      --disable-overlap-schedule` (for the native-MTP debug trace). If graphs/overlap can be
      re-enabled (or the trace made optional for steady training), 3x rate → ~4 replicas balance.
    - The stack has 4 student slots (`sglang-0..3`) but only 2 pods exist; scaling beyond 2
      requires new sampler pods (2 GPUs each, non-privileged template).
    - Teacher prefill stays balanced (~2400 tok/s on 2 replicas) until FB drops below ~9s or
      batch grows; then scale teachers alongside.
    Validation cell: `ep32_dpshard_async` (same shape as the winner + async overlap; expect
    steady step_total ≈ T_sample).

11. **Wave-4 results + promotion:** `ep32_dpshard_norecompute` = **1.31 ms/unique-valid (new best**,
    regret 3.85); bf16-reduce negative again (drop); async overlap validated (~2.8x end-to-end at 2
    sampler replicas, sampler-gated per `opd_async_prefetch_wait_s`). The long-run args file
    (`launch_args_er-opd-q36-mtp-ss-0605c.txt`) is updated to dp_shard + no_recompute + packing-off +
    chunk 32 + async overlap. Caveat: no_recompute was validated under `--skip-optim-step`; per-rank
    activations under dp_shard are one ~1536-token micro-batch so it should hold with optimizer
    state, but the first long run should watch step-1 memory.

12. **CP x DeepEP-internode is a device/transport-level incompatibility (open kernel issue).**
    Evidence chain from deterministic replay bisects (2-node CP2+EP16, captured payload
    `fb_capture_b32_v1`, per-phase `torch.cuda.synchronize()` debug in the stateful executor via
    `XORL_CP_STATEFUL_DEBUG_SYNC=1`):
    - The stateful CP executor's kernels are **clean** — every phase compute and relay passes an
      explicit device sync (test G/H runs logged through layer 38 of 40).
    - The crash is in DeepEP's internode path: `unspecified launch failure` on odd CP ranks +
      `DeepEP error: timeout (dispatch CPU)` on peers, or a hard SIGABRT/exit-255 on one rank.
      The failure point is **nondeterministic** (layer-0 dispatch in one run, backward region in
      another) — a use-time race, not a logic error.
    - Not executor-specific: the naive CP executor crashes identically (bisect A').
    - Not compile skew (crashes with compile off), not recompute (crashes with no_recompute),
      not init ordering (eager DeepEP buffer init before any CP collective — `model_runner`
      `Eagerly initialized DeepEP buffer` — fired and did not help).
    - Consistent history: CP + EP8 (intranode NVLink dispatch, no IBGDA) always worked;
      CP + EP16/EP32 (internode IBGDA) never did, with either executor.
    ~~Working hypothesis: concurrent CP-group NCCL kernels interfere with DeepEP's IBGDA
    GPU-initiated transport at use time.~~ **SUPERSEDED by §-1.14: the "transport race" was the
    §-1.13 backward asymmetries.** With the anchor discipline in place, CP2+EP16 DeepEP/IBGDA
    passes cleanly — no `ep_dispatch: alltoall` restriction. The evidence chain above stands as
    the failure signature to recognize if a future CP collective violates the anchor rule.

13. **CP-stateful executor: RESOLVED 2026-06-10 (~07:17Z).** First clean CP run:
    `q36mtp-20260610T071510Z` (CP2+EP16/alltoall, dp_shard, true no_recompute, FSDP deferral) —
    3/3 replay FBs, rc=0, steady FB **10.6-10.9s** for 5773 unique valid tokens
    (**1.84 ms/unique-valid**, ~1.2-1.4x the non-CP dp_shard winner at this short shape — CP's
    payoff is long-context), loss deterministic across repeat steps (1.0554; 0.4% from the
    cross-kernel-stack oracle 1.0598). The five stacked root causes, in discovery order:
    1. Per-layer Python/`tolist` plan recomputation → memoized (`_cp_replay_plan_local_segments_cached`).
    2. Relay = 4 independent autograd collectives → packed into one flat collective.
    3. `no_recompute` silently checkpointing attn+norm+router (`_moe_forward` `_selective` bug) →
       fixed + `tests/models/test_moe_gradient_checkpointing_methods.py`. **Affects non-CP too** —
       re-measure the promoted norecompute cell.
    4. Q/KV a2a backwards unordered on the pair communicator → anchor-chained in
       `UlyssesSyncStrategy` and in `UlyssesAsyncStrategy`'s module-projection fallback
       (the async main path routes q/k/v through one autograd Function — deterministic
       internal order — and was validated live by the §-1.14 DeepEP retest, which ran
       with `XORL_ULYSSES_FORCE_SYNC` unset).
    5. **THE key one:** relay backward nodes were (a) *unrecorded* on ranks whose state tables
       hadn't been written yet (`new_zeros` ⇒ requires_grad=False ⇒ no autograd node), and
       (b) *pruned* on ranks without local consumers, and (c) *unordered* — each asymmetry
       deadlocks the pair communicator in backward. Fix: the 3-property anchor discipline in
       `_forward_with_cp_replay_plan_stateful` (seed tables from `hidden_states`; accumulate each
       relay output into the layer output; chain anchors sequentially). This is also the
       retroactive explanation of the naive executor's `handoff_anchor` design.
    **Rule for any future CP collective in this codebase:** its input must require grad on every
    rank, its output must reach every rank's loss (`+ anchor*0`), and collectives must be chained
    for deterministic backward order. Debug kit: `XORL_CP_STATEFUL_DEBUG_SYNC=1` (per-phase/mb/loss
    markers + backward collective traces), `XORL_ULYSSES_FORCE_SYNC=1`, the FB replay harness.

14. **CP x DeepEP retest: PASSED (2026-06-10 ~07:24Z).** `20260610T072134Z` (CP2+EP16,
    `ep_dispatch: deepep` SMS24, `moe_implementation: quack`, dp_shard, no_recompute, FSDP
    deferral, async Ulysses, debug syncs off) — 3/3 replay FBs, rc=0, steady FB 14.8-15.6s,
    loss **1.0591 deterministic vs the 1.0598 non-CP oracle on the same quack/DeepEP stack
    (0.07%)** — i.e. near-exact parity once the kernel stack is matched; the earlier 0.4-1.2%
    spreads were triton-vs-quack numerics, not CP error. Verdict: the §-1.12 IBGDA aborts were
    the §-1.13 backward asymmetries corrupting in-flight transport; **CP composes with DeepEP
    internode** and no alltoall fallback is required. At this short shape (5773 valid tokens)
    alltoall (10.6s) beat DeepEP (15.6s) under CP2 — internode DeepEP overhead isn't amortized
    at 32 prompts — so pick dispatch by shape, not by stability. CP4 cell pending.

15. **CP4 cell: BLOCKED on an NCCL-level large-reduce-scatter wedge; FSDP grad-sync flush built
    and validated at CP2 (2026-06-10 ~10:46Z).** The CP4 investigation, in bisect order:
    - CP4 (ulysses=4, dp_shard=4, EP16) hangs in whatever backward first contains FSDP
      collectives: deferral ON → mb 7 (the sync micro-batch); deferral OFF → mb 0. Reproduces on
      deepep/quack AND alltoall/triton — not dispatch-specific. CPU 4-rank executor sim has exact
      fwd+grad parity and symmetric relay counts — executor logic is CP4-clean.
    - Initial mechanism (FSDP hook-driven reduce-scatters enqueue in rank-divergent order
      relative to the rank-specific CP executor backward graphs → multi-communicator NCCL
      deadlock) motivated **`fsdp_defer_grad_sync_flush`**: every micro-batch backward runs with
      gradient sync disabled and the accumulated grads are reduce-scattered in a manual
      post-loop pass (`xorl.distributed.fsdp2.flush_deferred_grad_sync` — per-unit
      `post_backward()` in registration order, mirroring FSDP2's root final callback; grad
      parity + state-machine hygiene unit-tested in `tests/distributed/test_fsdp2_grad_sync_flush.py`).
      With the flush, **all 8 CP4 micro-batch backwards complete** — the in-backward deadlock is
      gone.
    - The "residual wedge" was NOT NCCL-level (superseded): per-unit markers + per-rank
      faulthandler dumps + a deliberately-wrong-dtype zero-fill probe named it as **FSDP
      reduce-scatter SIZE mismatch from rank-divergent gradient coverage**. FSDP sizes the flat
      RS buffer from only the params that received grads; at narrow CP slices the coverage
      diverges (a 384-token CP4 slice can be pure padding — real seqs ≈1013 of 1536 — so that
      rank's GDN executor computes nothing and NO linear_attn param gets a grad; other ranks
      were missing the router-gate grad). Mismatched flat sizes hang `reduce_scatter_tensor`
      silently. CP2's 768-token slices always contain real segments — immune by shape, not by
      design. **Fix: `_zero_fill_missing_grads` in the flush (zeros in the REDUCE dtype —
      param-dtype fills raise FSDP's mixed-dtype assertion, which was itself the confirming
      signal).**
    - That unblocked compute but exposed one more deadlock in result delivery: rank-divergent
      OPD metric key sets (`opd_profile_*:sum_max` exists only on ranks that ran the valid-token
      KL path) made `_finalize_loss_metrics`' stacked all_reduce count differ across ranks
      (9 ranks reducing, 7 already in the dispatcher's all_gather). **Fix: cross-rank key-set
      canonicalization** (all_gather the key/op union over the reduce group, op-neutral fills,
      SORTED stacking order — unsorted stacks silently mix metrics — and empty-dict ranks still
      join). Same bug class as the old Bug 5 dict-keyed all_reduce.
    - Last anomaly: CP4-triton loss wobbled per step (1.0182/1.0247/1.0172 variants), confined
      to mb 7 (largest valid-token count), per-sequence. Schedules + shard invariants PROVEN
      clean offline on all 32 real payload batches at cp2/4/8 (invariant checker drives
      `prepare_singleshot_mtp_rollout_replay_opd_batch` on `fb_request_0.json`: single-writer
      slots, relay-once, no reuse, no trash reads). `torch.use_deterministic_algorithms(True)`
      (new env gate `XORL_TRAINER_DETERMINISTIC=1` → baked on both trainer nodes; setup.py)
      pins it to **1.0182×3 deterministic** with no op warning — i.e. an atomics-based CUDA
      kernel variant, not a logic bug.
    - **Status: CP1/CP2/CP4 all validated 3/3 rc=0 on the integration line (apanda-dev-mtp).**
      CP4-deepep 1.0453×3 (matches pre-merge bit-for-bit), CP4-triton 1.0182×3 under the
      determinism gate. The cross-(CP-width × kernel-stack) loss spread (1.018–1.060) is bf16
      reassociation from re-partitioned recurrent chains (fp32 CPU sim is EXACT at cp2/cp4) —
      treat it as the numerics band, never as A/B signal across CP widths. **Required CP
      config: `fsdp_defer_grad_sync` + `fsdp_defer_grad_sync_flush`** (a runtime warning now
      fires for CP-without-flush; in-backward RS remains exposed to the grad-coverage hazard).
      Debug kit: `XORL_CP_STATEFUL_DEBUG_SYNC=1` (markers, per-rank faulthandler dumps to the
      run dir), `XORL_TRAINER_DETERMINISTIC=1`. Worker pods now bake the debug env too
      (a head-only view hid worker state all session). CP8 cell: see AGENT_NOTES for the
      latest run.

16. **Production-shape CP sweep (2026-06-10 ~23:34Z): CP is the memory lever, not a throughput
    win, on this workload.** Synthetic long-context payloads built from `fb_capture_b32_v1` by
    prepending real prompt context (same rollouts/teacher hiddens/5773 supervised tokens;
    `fb_capture_b32_p4k_v1` = 4095-token, `fb_capture_b32_p8k_v1` = 8191-token context).
    Steady FB seconds, 16 GPUs, triton/alltoall, equal tokens/rank across widths:

    | context | CP1 | CP2 | CP4 | run artifacts (steady steps 1-2) |
    |---|---|---|---|---|
    | 1.5k (short cfg: no_recompute+defer_reshard; mixed stacks) | 6.4–6.9 | 10.6 | 18.8–19.0 | CP1 = matched non-CP control (loss 1.0472); CP2 = `q36mtp-20260610T071510Z` (alltoall, 1.0554); CP4 = `q36mtp-20260610T182050Z`/`182559Z` (triton/alltoall, 1.0182); CP4-deepep = `181207Z` (26.0–26.4s, 1.0453) |
    | 4k (recompute_before_dispatch, triton/alltoall) | **12.5–13.2** | 24.6 | 53.6–53.9 | CP1 `224310Z`; CP2 `223850Z`; CP4 `224610Z` |
    | 8k (recompute_before_dispatch, triton/alltoall) | **17.9** | 31.3–33.4 | — | CP1 `233059Z`; CP2 `232552Z` |

    All rows: 32 prompts, 5773 supervised tokens, equal tokens/rank across widths; payloads
    `fb_capture_b32_v1` / `_p4k_v1` / `_p8k_v1`, rows appended to each payload dir's
    `replay_results.jsonl`; run dirs `q36mtp-<id>-2s2t` under the stack result root. The 1.5k
    row mixes dispatch stacks (historical cells) — the 4k/8k rows are uniform triton/alltoall.

    Memory ladder at 4k+: the short-shape config does NOT transfer — `defer_reshard` keeps all
    35B params gathered (~70GB/rank, OOM) and even with reshard, `no_recompute` OOMs; all
    long-context cells need `recompute_before_dispatch`. CP1 then fits through the production
    max (8k) and wins every cell.

    **Why CP has worse economics here than on standard benchmarks:** Qwen3.6 is 30/40
    GatedDeltaNet linear-attention layers and 10/40 full-attention layers. For softmax
    attention, Ulysses shards quadratic attention work and pays pre/post all-to-alls
    (`strategy.py`). For GDN, xorl must preserve the recurrent prefix state across rank
    boundaries: in the standard path each rank computes compact boundary state/gradient
    summaries, all-gathers/merges them to reconstruct its corrected initial state, then runs
    the local interior (`flashqla_cp.py`); in the SingleShot stateful replay executor the
    equivalent is dependency-ordered phases with per-phase state relays. Either way CP adds
    synchronization and boundary/prefix-state reconstruction work while the computation being
    sharded is mostly linear-cost — there are no quadratic savings to pay for it. The
    full-attention layers still benefit, but they are only 25% of token-mixing layers. (The
    FlashQLA cross-node profile shows the same economics: ~2x single-node wins collapse to
    ~1.0-1.06x at high Ulysses widths — `xorl-flashqla/.../FLASHQLA_CROSS_NODE_PROFILE.md`.)
    Structural follow-up if long-context CP throughput ever matters: the gated-delta state
    recurrence is associative, so a cross-rank parallel scan over chunk states could cut the
    boundary-reconstruction depth (kernel project). Until then: use dp_shard (CP1) + recompute
    at long context; reach for CP2/CP4 only when a shape exceeds CP1's memory even with
    recompute.

Measured results accumulate in `topology_sweep_results.jsonl` next to the run dirs
(`fb_ms_per_unique_valid` is the comparison metric; legacy rows need de-inflation by the
replication factor as in §-1.2).

---

## 0. TL;DR for the next agent

- The trainer forward/backward (FB) of this workload is the optimization target. The previous track
  identified **SingleShot GDN (linear-attention) replay context-duplication** as the dominant waste.
- This session **implemented packed-varlen stateful GDN replay** (process each prompt/refill context
  once, capture recurrent + short-conv state, replay only branch suffixes). It is correctness-tested
  (exact dense match incl. gradients) and **validated on the real 4-node smoke**.
- **Result: FB 75.2s → 45.2s (1.66x) vs the documented cap-16384 baseline; 94.4s → 45.2s (2.09x) on a
  matched same-code A/B.** Gated behind `linear_replay_plan_use_stateful_prefix_cache` (default **false**;
  cap-16384 path is the automatic fallback). See §4 for the exact numbers and run IDs.
- The FB op-level profile is now captured. The remaining CUDA time is **FSDP param communication, not GDN**:
  `record_param_comms` is 21.9s / 87.5% self CUDA on the profiled rank, split into 12.2s reduce-scatter
  (`f32`) and 9.7s all-gather. GDN kernels are subsecond-scale in the same dump. See §4.5.
- Three follow-up FSDP smokes were run. **DP replication / shard narrowing is negative** (`replicate=2`,
  `shard=16`: 81.7s FB, 2.49 ms/valid). **BF16 reduce is not promoted**: it validated, but a same-day FP32
  stateful smoke was faster (`28.4s` FB, `43880` valid, `0.65` ms/valid), proving rollout/replay variance is
  too large for a one-run dtype call. See §4.6 and §5.1.
- Topology follow-up: **Ulysses CP is now unblocked for rollout-replay SingleShot-MTP OPD** and validated by
  a 2-rank H100 Qwen3.6-style smoke against a full-sequence reference. Ring CP, PP, and TP remain blocked.
  A 35B OPD CP=2/stateful FB profile was captured with **Quack + DeepEP/SMS24** on
  `q36mtp-20260609T090846Z-2s2t` (artifact paths in §4.5.1), but it is profile-only evidence, not a throughput
  baseline. A full 32-prompt CP2 throughput attempt (`q36mtp-20260609T164154Z-2s2t`) was non-competitive
  (`216.3s` FB for chunk 1 only; chunk 2 did not finish before stop), so the run-kicking config should stay on
  the non-CP path (`--trainer-ulysses-parallel-size 1`). Under the required Quack + DeepEP24 defaults, the
  measured topology winner is **EP32**: `q36mtp-20260609T174640Z-2s2t` steady rows total `309312` valid tokens,
  `430.99s` FB (`1.39` ms/valid), and `73.88` supervised tok/s/GPU end-to-end. EP16 is the runner-up
  (`1.54` ms/valid, `53.05` supervised tok/s/GPU), and EP8 trails both. EP32/alltoall is a historical
  diagnostic control and is slow (`148.5s` FB); EP16/EP32 DeepEP at SMS72 fail in the internode DeepEP kernel.
  See §4.7.

---

## 1. What this workload actually does (so the doc is self-contained)

OPD = on-policy distillation. Per training step:

1. **Student sampling** (SGLang, native MTP): the student generates rollouts for 32 prompts. Native MTP
   commits a variable number of tokens per step ("commit groups"); a long generation (`max_new_tokens=256`)
   produces ~hundreds of commit groups.
2. **Teacher prefill** (SGLang): teachers prefill the rollouts and cache hidden states.
3. **Trainer FB** (xorl server, the thing we optimize): the student model runs forward+backward over the
   rollouts and computes the MTP distillation loss (KL to teacher) **at the mask/commit positions only**.
   `--skip-optim-step` is set for profiling (no weight update, no checkpoint → non-destructive).

The model is a **hybrid linear-attention transformer**: most layers are GatedDeltaNet (GDN, linear
attention) and a minority are full softmax attention, interleaved with MoE FFNs (A3B = ~3B active of 35B,
expert-parallel via DeepEP across the 4 nodes).

### Why "SingleShot replay" exists at all (the crux)

A SingleShot-MTP rollout interleaves prompt/refill context with mask/commit tokens. Each mask token must
attend **causally to a specific prefix** (the tokens committed before it). For **softmax attention** this is
free: one forward with a block-diagonal/causal mask gives every query its correct visibility in a single
pass. For **linear attention (GDN) this is impossible**: a GDN layer's output at position *t* is a function
of the recurrent **state** entering *t*, which is path-dependent and sequential — there is no per-query mask.

So to evaluate GDN at a mask token with the correct visibility, the framework **replays a virtual sequence**
= `[visible prefix context] + [branch's mask/commit suffix]` through GDN, and scatters the suffix outputs
back. Across all mask blocks of a row this re-feeds the prompt/refill context many times. **That replay
expansion is the root of the original low MFU**, and everything below is about shrinking it.

---

## 2. History (how we got here)

| Run | Shape | Replay setting | FB (s) | Notes |
|---|---|---|---:|---|
| `…20260608T232600Z` | 32p/2chunk | cap `1024` | 458.19 | kernel-storm: 540 GDN calls/layer, micro-kernel launch flood |
| `…20260608T235749Z` | 16p/1chunk | cap `16384` | 73.41 | raising the packed-token cap removed launch fragmentation (6.1x) |
| `q36mtp-20260609T000211Z` | 32p/2chunk | cap `16384` | **75.16** | **documented baseline** (`385651` replay tok, `98.1%` duplicated context, 27 GDN calls) |
| `…20260609T002602Z` | 32p/2chunk | stateful **boundary-loop** | 329.0 | negative result: 473 per-boundary `_forward_standard` launches |

The cap-1024→16384 fix made the *expanded* work launch-efficient (big GEMMs instead of a micro-kernel
storm) but did **not** remove the context duplication: at cap-16384, `98%` of the `385651` replay tokens are
duplicated context, for only ~`52024` supervised tokens. The boundary-loop stateful experiment proved the
state-reuse idea is correct but is launch-bound if you capture one state per branch boundary.

---

## 3. Root-cause analysis: why MFU is (still) low

The FB does far more FLOPs — and far more memory traffic and comms — than the supervised signal requires.
There is **no single cause**; there is a stack of them. The replay-duplication item (3a) is the one this
session attacked. The rest are still open and are why MFU remains low after a 66% FB win.

### 3a. GDN replay context duplication — *the contributor we reduced*
Each branch re-feeds its visible prefix context through every GDN layer. At cap-16384 baseline: `385651`
replay tokens for `52024` supervised, `98%` duplicated. This is pure redundant work. Stateful replay
(§4) processes context once and replays only suffixes, cutting GDN token volume ~5–56x depending on
alignment. **Addressed this session, but not to the theoretical minimum** (see 3b/3g).

### 3b. GDN is memory-bound, not compute-bound — *intrinsic, not yet addressed*
Even *minimal* GDN replay is low-MFU. The chunked gated-delta-rule recurrence is dominated by reading and
writing the `[H, K, V]` recurrent state and the short-conv state, plus small per-token elementwise ops — not
large GEMMs. So GDN layers have low arithmetic intensity and low MFU **regardless of how few tokens they
process**. Reducing GDN token count helps wall-clock, but a model that is mostly GDN layers will never reach
high MFU on the GDN portion. (The full-attention + MoE-GEMM layers are the compute-bound, high-MFU part.)
This caps how much the replay optimization alone can buy.

### 3c. Static padding for compile stability — *open*
`static_padded_seq_len=1536` pads every sample to a fixed length so the replay plan tensors have static
shapes and `torch.compile` produces one graph (no recompiles on variable rollout length). Sample-packing
(`4096`) packs 2 static slots per row and recovers utilization to ~75% (handoff §"Generic Sample Packing"),
i.e. ~25% of the dense (MoE/attention) forward is spent on pad tokens. The static padding is a compile-graph
tax; whether it can be relaxed (dynamic shapes on the linear-attn path is disabled anyway) is open.

### 3d. Backward recompute — *open*
`gradient_checkpointing_method=recompute_full_layer` recomputes the full forward during backward (~2x
forward FLOPs). It is required for memory headroom (35B weights + replay expansion), but it means
`trainer_backward_s` ≈ `trainer_forward_loss_s` + recompute. In the stateful run backward (20.4s) ≈ forward
(18.9s). Selective/partial checkpointing (checkpoint only the expensive layers, or skip recompute on the
cheap GDN replay path) is unexplored.

### 3e. MoE expert-parallel comms — *open*
EP=8 over 4 nodes (DeepEP) does an all-to-all dispatch + combine per MoE layer. Prior profiling on a related
30B-A3B model found combine ≈7 ms/layer (NCCL all-to-all + fp32 unpermute). For a 35B MoE this is real
non-FLOP wall-clock that lowers MFU and is **independent of the replay work** — so as replay shrinks, MoE
comms becomes a larger share of FB.

### 3f. FSDP param all-gather / reduce-scatter — *measured dominant after stateful replay*
The stateful FB profile (`q36mtp-20260609T025404Z`, §4.5) says the current largest rank-0 CUDA bucket is
FSDP param communication, not replay: 21.9s self CUDA in `record_param_comms`, with 12.2s in
`nccl:_reduce_scatter_base` / `ReduceScatter_Sum_f32` and 9.7s in `nccl:_all_gather_base`. This explains why
GDN-specific work now has lower leverage. A first DP-replication smoke was negative (§4.6), while BF16 reduce
is inconclusive after same-day FP32/BF16 smokes and must be tested with a tighter repeated comparison before
promotion.

### 3g. Residual replay duplication from coarse capture — *partially open, this session's tradeoff*
The stateful fix snaps state-capture boundaries to a coarse alignment (`_STATEFUL_REPLAY_CAPTURE_ALIGN=256`)
to avoid a launch per boundary. The cost is that each branch suffix re-processes its `<256`-token
intra-block context remainder. In the validated run the suffix carried ~67k tokens of which only ~1.4k are
true mask payload — i.e. **most of the "suffix" is still re-processed context**. The theoretical minimum
(context once + masks ≈ 6–7k GDN tokens) requires single-pass intermediate-state capture (§5.2), which we
did **not** build.

### 3h. Supervision density (a ceiling, not a bug)
Only mask/commit positions are supervised, but the prompt-context forward is mandatory to build the GDN
state / attention KV. So a large fraction of the *necessary* FB is context that produces no supervised loss.
This bounds "useful-FLOPs / total-FLOPs" from above and is structural to OPD on long prompts; the lever is on
the sampling side (share prefixes across N continuations per prompt) rather than FB.

**Bottom line:** 3a was the headline waste and is now reduced. The next measured wall is FSDP parameter
communication (§3f). GDN replay is no longer the dominant kernel bucket; single-pass capture / align sweeps
are still valid cleanup work, but they are not the highest-probability MFU lever until the FSDP comm path is
addressed.

---

## 4. What this session built and measured

All changes are gated behind `linear_replay_plan_use_stateful_prefix_cache` (default **false**); the
cap-16384 packed-chunk path is the automatic fallback (the stateful method returns `None` when no schedule is
available, e.g. a branch whose visible context is not a prefix of its context sequence).

### 4.1 The implementation (3 pieces)

1. **Per-segment packed short-conv cache** — `src/xorl/ops/linear_attention/modules/short_conv.py`.
   The `cu_seqlens` (packed-varlen) path now honors a per-segment `cache` (`[num_segments, hidden, width]`)
   via an offset extended-buffer layout (each segment gets its own `kernel-1` prefix slots; no segment's
   conv window can read another's tokens) and emits a per-segment `output_final_state`
   (`_segment_conv_final_state`). Required so suffix replay sees the correct conv history at a boundary.

2. **Stateful replay schedule** — `src/xorl/mtp/singleshot.py`.
   `build_rollout_replay_linear_plan` now also derives a `RolloutReplayStatefulSchedule` (once, eagerly, on
   CPU; moved via `.to()`): round-major context segments (per-segment `state_in`/`state_out` slot, packed
   rows/cols, `cu`, round offsets) + a packed suffix (per-branch state slot, packed rows/cols/`cu`, output
   packed-index + scatter coords). A **prefix-validity guard** returns `None` (→ fallback) if any branch's
   visible context is not a prefix of its context sequence.
   - **Coarse capture (`_STATEFUL_REPLAY_CAPTURE_ALIGN=256`):** each branch's capture boundary is snapped
     DOWN to a multiple of 256, so hundreds of per-commit boundaries collapse to `O(ceil(L/256))` capture
     rounds (independent of branch count). The branch suffix prepends its `<256`-token intra-block context
     remainder so it re-derives the exact pre-mask state from the captured coarse-boundary state (chaining is
     exact → numerically identical to dense). Larger align = fewer rounds (launches) but more re-processed
     context.

3. **Packed-varlen executor with a tensor state table** — `src/xorl/ops/linear_attention/layers/gated_deltanet.py`,
   `_forward_with_replay_plan_stateful_prefix_cache`. Batched context rounds (one packed `_forward_standard`
   call per round) capture per-segment recurrent + conv state into **tensor slot-tables** written with
   `index_copy` and read with `index_select` (slot 0 = the implicit zero state). Then **one packed suffix
   call** replays all branch suffixes from their gathered initial states. One `.tolist()` per FB total (no
   per-round host syncs, no Python dict / per-element `cat`).

### 4.2 Telemetry (so future runs are measurable)
`summarize_rollout_replay_linear_plan` + `model_runner.py` + `run_opd_pipeline.py` emit the actual stateful
schedule counts: `[singleshot-mtp-profile] stateful schedule_micro_batches=… fallback_micro_batches=…
max_context_rounds=… context_segments=… context_packed_tokens=… suffix_segments=… suffix_packed_tokens=…
state_slots=… gdn_forward_calls_per_layer=…`, and the same as `opd_singleshot_mtp_replay_plan_stateful_*`
keys in `opd_profile.jsonl`.

### 4.3 The two dead-ends that shaped the design (don't repeat them)
- **Per-boundary capture is launch-bound.** Capturing one state per distinct branch boundary on a ~250-step
  generation = 248 nested boundaries → 248 sequential GDN launches. FB = 304s (telemetry
  `stateful_context_rounds=248`). This is the same failure as the old boundary-loop (329s). *Fix:* coarse
  alignment → 3 rounds.
- **Per-layer Python state assembly is overhead-bound.** With coarse alignment (3 rounds) but a Python dict +
  N-way `cat` over ~92 branches to build the suffix initial state (plus `.tolist()`/`.item()` syncs per
  round), FB = 93.6s — still worse than baseline, and it ran *slower than the GDN compute it replaced*. *Fix:*
  tensor slot-table (index_select/index_copy). FB 93.6s → 45.2s **on a larger suffix (67k vs 42k tokens)**,
  which proves the per-layer assembly overhead — not GDN FLOPs — was the wall at that stage.

### 4.4 Measured A/B (Q3.6-35B-A3B, 4-node, 32-prompt/2-chunk smoke)

| Run | Stateful | FB (s) | fwd (s) | bwd (s) | valid tok | ms/valid | sup tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cap-16384 baseline (documented `q36mtp-20260609T000211Z`) | off | 75.16 | 25.8 | 45.5 | 52024 | 1.44 | 402.68 |
| matched baseline, current code (`q36mtp-20260609T015642Z`) | off | 94.39 | 38.7 | 48.9 | 57824 | 1.63 | 462.23 |
| **stateful tensor-table, align=256 (`q36mtp-20260609T015144Z`)** | **on** | **45.20** | **18.9** | **20.4** | 42344 | **1.07** | 407.37 |
| stateful + BF16 FSDP reduce (`q36mtp-20260609T030727Z`) | on | 51.12 | 21.2 | 23.3 | 50944 | 1.00 | 568.65 |
| stateful + FP32 FSDP reduce (`q36mtp-20260609T031333Z`) | on | **28.40** | 12.2 | 11.7 | 43880 | **0.65** | **640.99** |

- vs matched baseline (same code, controls for stochastic-rollout variance): **2.09x FB**, **1.52x per
  valid-token** (1.07 vs 1.63 ms/valid).
- vs documented cap-16384 baseline: **1.66x FB** (45.20 ≤ the 60s "meaningful win" gate; < the 75.16s
  promotion bar). Backward fell the most (20.4 vs 48.9s).
- Correctness: focused CPU tests match dense replay **exactly** incl. gradient parity (faithful
  `cu_seqlens`-aware fake kernel), multi-row, coarse-boundary capture, short-conv per-segment cache; the real
  run reported `OPD pipeline validation succeeded` with **0 stateful fallbacks** on all 4 micro-batches.
- Telemetry on the winning run: `stateful_context_rounds=3`, `gdn_forward_calls≈14` (vs 248/345 for the
  per-boundary attempt). Baseline GPU path is byte-identical (the schedule is still *built* when the flag is
  off — `stateful_schedule_micro_batches=4` on the baseline run — but never *used*), so the 94.4 vs 75.2 is
  rollout variance (per-valid 1.63 vs 1.44, within the documented FB variance), not a regression.
- The two same-day dtype rows are not a controlled same-rollout A/B (student sampling is stochastic). The
  FP32 row beating BF16 means **do not promote BF16 reduce** from the current evidence; use repeated paired
  smokes or a deterministic rollout path before changing defaults.

### 4.5 FB op-level profile of the stateful path
Captured on `q36mtp-20260609T025404Z-2s2t`:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/
  q36mtp-20260609T025404Z-2s2t/server_output/fb_profiles/fb_keyavg_call1.txt
```

Use the key-average dump for attribution; the profiled chunk's wall time is inflated by profiler export/capture.
The unprofiled chunk in the same run was still representative (`trainer=37.68s`, `valid_tokens=21032`,
`stateful_context_rounds=2`, `stateful_gdn_forward_calls=6`).

Top CUDA buckets in `fb_keyavg_call1.txt`:

| Bucket | Self CUDA | Share | Calls | Interpretation |
|---|---:|---:|---:|---|
| `record_param_comms` | 21.877s | 87.53% | 496 | FSDP param communication wrapper |
| `nccl:_reduce_scatter_base` / `ReduceScatter_Sum_f32` | 12.153s | 48.63% | 164 | gradient reduce-scatter, currently fp32 |
| `nccl:_all_gather_base` / `AllGather_RING_LL` | 9.721s | 38.90% | 326 | FSDP param all-gather |
| `_FusedUnpermuteAndCombine` | 265ms | 1.06% | 160 | MoE combine |
| `_FusedDispatchAndPermute` | 221ms CUDA / 8.18s CPU | 0.88% CUDA | 160 | MoE dispatch, notable CPU overhead |
| `ChunkGatedDeltaRuleFunctionBackward` | 159ms | 0.64% | 210 | GDN backward |
| `ChunkGatedDeltaRuleFunction` | 130ms | 0.52% | 420 | GDN forward |

So the old GDN micro-kernel storm is gone. The remaining MFU problem is dominated by FSDP comm, especially
fp32 reduce-scatter. MoE dispatch/combine and GDN kernels are visible but not first-order in this profile.

### 4.5.1 CP=2/stateful FB profile captured with Quack + DeepEP/SMS24

The previously missing CP/stateful FB profile was captured on the real k8s stack with Quack MoE and DeepEP at
24 SMs:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/
  q36mtp-20260609T090846Z-2s2t/server_output/fb_profiles/fb_trace_call0.json.gz
  q36mtp-20260609T090846Z-2s2t/server_output/fb_profiles/fb_keyavg_call0.txt
  q36mtp-20260609T090846Z-2s2t/artifacts/opd_profile.jsonl
```

Run shape: CP/Ulysses=2, DP shard=16, EP8, Quack MoE, DeepEP dispatch with `deepep_num_sms=24`, stateful replay
enabled, 8 prompts in one chunk, profiler `skip=0,count=1`. Generated-config checks from the run:

```text
moe_implementation: quack
ep_dispatch: deepep
deepep_num_sms: 24
ulysses_parallel_size: 2
linear_replay_plan_use_stateful_prefix_cache: true
linear_replay_plan_max_packed_tokens: 16384
enable_packing: true
sample_packing_sequence_len: 4096
fsdp_reduce_dtype: fp32
enable_compile: true
compile_linear_attention: false
expert_parallel_size: 8
```

The run completed cleanly (`trainer-head cleanup rc=0`, `OPD pipeline validation succeeded`). The profiled FB
reported `trainer_forward_backward_s=167.742`, `forward_loss=33.804`, `backward=37.877`, `valid_tokens=6100`,
`forward_backward_roundtrip_s=167.870`. Stateful replay was active with `fallback_micro_batches=0`,
`stateful_replay_tokens=2457`, `stateful_token_savings=221118`, and `gdn_forward_calls_per_layer=3` in the
stateful schedule.

Key `fb_keyavg_call0.txt` buckets:

| Bucket | Self CUDA | Calls / notes |
|---|---:|---|
| `Lazy Function Loading` | 94.898s | profiler/lazy-load artifact on the first profiled call |
| `record_param_comms` | 34.263s | FSDP communication wrapper |
| `nccl:_reduce_scatter_base` / `ReduceScatter_Sum_f32` | 23.914s | gradient reduce-scatter, fp32 |
| `nccl:_all_gather_base` / `AllGather_RING_LL` | 9.232s / 9.407s | FSDP param all-gather |
| `ChunkGatedDeltaRuleFunctionBackward` | 4.622s | GDN backward bucket |
| `ChunkGatedDeltaRuleFunction` | 4.227s | GDN forward bucket; CUDA total inflated by lazy-load attribution |
| `QuackEPDeepEPNoPermute` | 499.5ms | Quack + DeepEP expert compute/dispatch wrapper |
| DeepEP combine/dispatch kernels | 461.8ms / 182.3ms | visible DeepEP intranode combine/dispatch kernels |

The full 32-prompt, two-chunk Quack + DeepEP24 profile attempt (`q36mtp-20260609T080402Z-2s2t`) timed out on
the first FB after 3600s and produced no profile artifacts. Do **not** treat that as license to switch to
alltoall; the successful 8-prompt capture proves the intended Quack + DeepEP24 CP/stateful path works, while
the full-size profiled first call needs a smaller capture shape or further DeepEP/Quack debugging.

Older CP/stateful profile-only artifact: `q36mtp-20260609T073613Z-2s2t` captured DeepEP/SMS24 before the
Quack default flip. Keep it as historical evidence only; use `q36mtp-20260609T090846Z-2s2t` for the current
Quack + DeepEP24 profile.

### 4.6 FSDP comm follow-ups measured after the profile

| Run | FSDP setting | FB (s) | fwd (s) | bwd (s) | valid tok | ms/valid | Result |
|---|---|---:|---:|---:|---:|---:|---|
| `q36mtp-20260609T030203Z` | `replicate=2`, `shard=16`, reduce fp32 | 81.72 | 35.2 | 40.3 | 32776 | 2.49 | **negative** |
| `q36mtp-20260609T030727Z` | `replicate=1`, `shard=32`, reduce bf16 | 51.12 | 21.2 | 23.3 | 50944 | 1.00 | inconclusive |
| `q36mtp-20260609T031333Z` | `replicate=1`, `shard=32`, reduce fp32 | 28.40 | 12.2 | 11.7 | 43880 | 0.65 | **best same-day smoke** |

The DP-replication result validates that the new launcher knobs work and fit in memory, but it is not a
performance path for this workload. Narrowing the FSDP shard group likely does not reduce per-rank param
all-gather volume enough to offset the extra residency / algorithm effects.

The BF16-reduce run used the same stateful replay path and generated config:

```text
data_parallel_replicate_size: 1
data_parallel_shard_size: 32
fsdp_reduce_dtype: bf16
linear_replay_plan_use_stateful_prefix_cache: true
```

Both dtype runs reported `OPD pipeline validation succeeded`. Because rollout stochasticity changes
valid-token count and replay shape, and because the follow-up FP32 smoke beat BF16, do not flip the long-run
default from these one-off smokes. Run the promotion check in §5.1.

### 4.7 Topology follow-up: EP32/DeepEP24 is the measured winner

Recommended 32-rank trainer topology:

```text
pp = 1
tp = 1
ulysses = 1
ring = 1
data_parallel_shard_size = 32
data_parallel_replicate_size = 1
expert_parallel_size = 32
ep_dispatch = deepep
deepep_num_sms = 24
```

The main mesh uses `world_size = pp * dp * ring * ulysses * tp`, so the 4-node stack is one 32-rank FSDP
data-parallel shard. EP is overlaid as a separate `(ep, ep_fsdp)` mesh inside each PP stage:
`ep_fsdp = 32 / expert_parallel_size`. For Qwen3.6-35B-A3B the model has 256 experts, so EP8, EP16, and EP32
all satisfy the divisibility checks.

The available topology knobs are narrower than they first look:

- **CP / Ulysses:** unblocked for rollout-replay SingleShot MTP OPD after the 2026-06-09 CP patch. The
  dispatcher defers generic sequence sharding for SingleShot OPD, `ModelRunner` builds the full replay layout
  first and then shards local sequence/loss/cache tensors, Qwen full-attention layers build full structured
  masks under CP, and GDN now executes the replay plan as true rank-local Ulysses segments: each rank packs
  only the global replay columns it owns, passes recurrent + short-conv replay-state tables to the next CP
  rank via autograd-aware point-to-point sends/receives, and scatters only local outputs. There is no
  full-hidden `gather_outputs` fallback on the GDN replay path. Validation:
  `tests/distributed/test_qwen3_5_singleshot_cp.py` (2xH100) forbids the old full-hidden gather, compares
  gathered CP output and reduced OPD loss to a full-sequence reference, and backprops a suffix-only scalar so
  rank-0 gradients must arrive through the sharded replay-state handoff. Do not promote CP without a future
  35B throughput run that beats the EP32/DeepEP24 default.
- **CP / Ring:** still blocked for Qwen3.6 linear attention (`ringattn_size > 1` raises before SingleShot prep).
- **PP:** blocked for SingleShot MTP OPD.
- **TP:** blocked by `opd_loss` (`opd_loss does not yet support tensor parallelism`).
- **DP/HSDP:** works as a launcher knob, but the `replicate=2, shard=16` smoke in §4.6 was negative.
- **EP:** legal to vary; larger EP reduces expert FSDP (`ep_fsdp`) but makes EP dispatch more internode.

Measured topology matrix:

| Candidate | `ep_fsdp` | Dispatch / SMS | Run | Result |
|---|---:|---|---|---|
| **EP32** | 1 | **Quack + DeepEP / 24** | `q36mtp-20260609T174640Z-2s2t` | **current winner**: 3-step run, clean validation; steady rows total `309312` valid, `430.99s` FB, `1.39` ms/valid FB, `717.68` FB-valid tok/s world, `73.88` sup tok/s/GPU end-to-end |
| EP32 | 1 | Quack + DeepEP / 24 | `q36mtp-20260609T174045Z-2s2t` | 1-step warmup smoke, clean validation; `164.25s` FB, `237824` valid, `0.69` ms/valid |
| EP16 | 2 | Quack + DeepEP / 24 | `q36mtp-20260609T172711Z-2s2t` | runner-up current-compliant 3-step run, clean validation; steady rows total `113152` valid, `174.36s` FB, `1.54` ms/valid FB, `648.98` FB-valid tok/s world, `53.05` sup tok/s/GPU end-to-end |
| EP16 | 2 | Quack + DeepEP / 24 | `q36mtp-20260609T172115Z-2s2t` | 1-step warmup smoke, clean validation; `131.48s` FB, `81744` valid, `1.61` ms/valid |
| EP8 | 4 | Quack + DeepEP / 24 | `q36mtp-20260609T170713Z-2s2t` | slower current-compliant 3-step run, clean validation; steady rows total `79688` valid, `137.08s` FB, `1.72` ms/valid FB, `44.25` sup tok/s/GPU end-to-end |
| EP8 | 4 | Quack + DeepEP / 24 | `q36mtp-20260609T165857Z-2s2t` | current-compliant 1-step warmup, clean validation; `133.88s` FB, `46120` valid, `2.90` ms/valid |
| EP8 | 4 | DeepEP / 72 historical | `q36mtp-20260609T031333Z-2s2t` | historical best same-day stateful smoke: `28.40s` FB, `43880` valid; not current-compliant because future/current configs are standardized on Quack + DeepEP/SMS24 |
| EP8 + HSDP | 4 | DeepEP / 72, `replicate=2, shard=16` | `q36mtp-20260609T030203Z-2s2t` | validates but negative: `81.72s` FB, `32776` valid |
| EP16 | 2 | DeepEP / 72 | `q36mtp-20260609T035339Z-2s2t` | fails in DeepEP internode kernel |
| EP32 | 1 | DeepEP / 72 | `q36mtp-20260609T034149Z-2s2t` | fails in DeepEP internode kernel |
| EP32 | 1 | alltoall | `q36mtp-20260609T034743Z-2s2t` | validates but slow: `148.51s` FB, `180224` valid |
| EP16 | 2 | DeepEP / 48 | `q36mtp-20260609T040010Z-2s2t` | validates: `90.31s` FB, `66880` valid |
| EP8 + CP2 | 4 | Quack + DeepEP / 24 | `q36mtp-20260609T090846Z-2s2t` | FB profile captured; profile-only 8-prompt shape, not throughput-promoted |
| EP8 + CP2 | 4 | Quack + DeepEP / 24 | `q36mtp-20260609T164154Z-2s2t` | full 32-prompt throughput attempt is negative/non-competitive: chunk 1 `216.30s` FB for `16384` valid (`13.2` ms/valid), chunk 2 ran >8 min without a JSONL row and was stopped |

The CP patch's correctness evidence is the focused 2-rank H100 model+OPD-loss smoke plus CPU/server prep
tests, and the CP2 8-prompt profile proves the real Quack + DeepEP24 CP path can execute. It is not the
fastest operational topology on the full 32-prompt OPD shape.

The DeepEP failure signature at SMS72:

```text
DeepEP/csrc/kernels/internode.cu:492,
condition: ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms
CUDA error: unspecified launch failure
```

So wider EP was not simply "set EP32" under the old SMS72 setting: EP16/EP32 at SMS72 move DeepEP dispatch
internode and fail. Lowering DeepEP to SMS24 makes the current Quack + DeepEP path validate, and the
current-compliant EP32/SMS24 3-step run beats EP16 and EP8 on both FB ms/valid and end-to-end supervised
tok/s/GPU.

Recommendation: **promote non-CP EP32/Quack/DeepEP24 as the run-kicking topology.** Do not promote CP2,
EP32/alltoall, EP16/SMS48, or any SMS72 internode DeepEP result. Do not promote another topology change
without a clean 3-step win over EP32 on both FB ms/valid and end-to-end supervised tok/s/GPU.

---

## 5. What to do next (in priority order)

### 5.0 Topology conclusion / launch gate
The topology result is actionable and should be used for the next kicked-off throughput run:

- Keep **non-CP EP32/Quack/DeepEP/SMS24** as the working default. For the run-kicking agent, use
  `--trainer-ulysses-parallel-size 1`, `--trainer-expert-parallel-size 32`, `--trainer-ep-dispatch deepep`,
  `--trainer-moe-implementation quack`, and `--trainer-deepep-num-sms 24`.
- Evidence: EP32/SMS24 steady rows (`q36mtp-20260609T174640Z-2s2t`) beat EP16/SMS24 steady rows
  (`q36mtp-20260609T172711Z-2s2t`) on FB ms/valid (`1.39` vs `1.54`), FB-valid tok/s world (`717.68` vs
  `648.98`), and end-to-end supervised tok/s/GPU (`73.88` vs `53.05`). EP8/SMS24 is slower again
  (`1.72` ms/valid, `44.25` supervised tok/s/GPU). The OPD loop does not emit a true MFU field
  (`mfu`/`flops`/`tflops` absent from the run artifacts), so this handoff uses FB ms/valid and supervised tok/s
  as the throughput/MFU proxy.
- Do not use the CP path for the fastest throughput run. CP is enabled by `--trainer-ulysses-parallel-size 2`;
  it is useful for the already-captured op profile and future debugging, but the measured full 32-prompt CP2
  throughput shape is negative.
- Do not switch to alltoall to work around DeepEP issues. If a profile fails, shrink the capture shape or debug
  Quack + DeepEP24 directly. Only use alltoall as an explicitly labelled diagnostic control.
- Do not run EP16/EP32 DeepEP at SMS72; that exact path fails in the internode DeepEP kernel.
- Future generated and checked-in OPD/MTP configs should use `moe_implementation: quack`, `ep_dispatch: deepep`,
  and `deepep_num_sms: 24` unless a run is explicitly marked as a diagnostic control.
- If continuing topology, require a repeated 3-step win over EP32 before considering another promotion.

The reason to stay conservative: EP changes can reduce expert FSDP communication, but the captured profile's
dominant CUDA bucket is dense FSDP param all-gather/reduce-scatter. Wider EP also increases internode expert
dispatch pressure.

### 5.1 Validate FSDP reduce dtype with repeated paired smokes
The profile points at fp32 reduce-scatter, but the one-off BF16 smoke did **not** survive a same-day FP32
comparison. Next step: run a tighter FP32-vs-BF16 comparison before changing defaults. Suggested check:

1. Run 3-5 one-step stateful smokes with `--trainer-fsdp-reduce-dtype fp32` and 3-5 with `bf16`, same prompt
   dataset offset and same launcher args, all `--skip-optim-step`.
2. Compare `trainer_forward_backward_s / valid_tokens`, not raw FB seconds, because stochastic MTP rollouts
   change valid-token count and replay suffix volume.
3. If BF16 wins consistently and `OPD pipeline validation succeeded` remains clean, try one short non-skip
   multi-step run from the same checkpoint and compare loss/grad-norm traces to FP32 reduce.
4. Only then promote `--trainer-fsdp-reduce-dtype bf16` in the stack launcher or trainer YAML.

Do **not** spend more time on `--trainer-data-parallel-replicate-size 2 --trainer-data-parallel-shard-size 16`
for this workload unless a later profile changes the diagnosis; the measured smoke is a clear negative.

### 5.2 Single-pass intermediate-state capture (removes §3g, and the context rounds)
The current path still pays ~3 sequential context rounds + a suffix whose bulk is re-processed context. The
theoretical minimum is: process the full context **once** and read the recurrent state at every needed
boundary from a single pass, then replay only the true mask suffixes. The chunk GDN kernel already computes
per-chunk states internally: `chunk_gated_delta_rule_fwd_h` returns `h` of shape `[B, NT, H, K, V]`
(`NT = ceil(T/64)`) in `src/xorl/ops/linear_attention/ops/common/chunk_delta_h.py`. Plan:
1. Plumb `h` out of `chunk_gated_delta_rule` (the autograd `Function` currently returns only `o, final_state`)
   **with backward support** — gradients must flow captured-state → context tokens, so `h` cannot be detached.
   This is the hard part (a new differentiable output on the kernel `Function`).
2. For each boundary `b`, take the chunk-boundary state `h[b//64]` and apply a short intra-chunk correction
   (a batched `fused_recurrent` over the `≤63` remaining tokens, per-segment initial state) to get the exact
   state at `b`. The short-conv state at `b` is just the last `kernel-1` context tokens — a gather, no kernel.
3. Replay only the true mask suffixes (`≈1.4k` tokens) from those states.
This collapses the context cost to ~1 pass and the suffix to the mask payload, eliminating 3g. Risk: the
kernel `Function` backward change; validate against the dense path with the existing faithful-fake tests
first, then on the smoke.

### 5.3 Sweep `_STATEFUL_REPLAY_CAPTURE_ALIGN` (now lower priority)
The profiler says GDN is no longer dominant, so this is no longer the first lever. It is still cheap cleanup:
we validated 256 (3 rounds, 45.2s). Try 512 and 128 on the smoke to find the local optimum before investing
in 5.1. (Constant in `src/xorl/mtp/singleshot.py`.)

### 5.4 Address the remaining non-replay buckets
- **FSDP param comms (3f):** repeated reduce-dtype comparison first; then inspect whether all-gather can be
  overlapped or bucketed differently. DP replication / shard 16 has already failed this smoke.
- **MoE comms (3e):** keep Quack + DeepEP/SMS24 for this workstream; only revisit `alltoall` as an explicitly
  labeled diagnostic control if a future profile makes expert dispatch/combine the bottleneck.
- **Recompute (3d):** selective checkpointing — the GDN replay layers may not need full recompute.
- **Padding (3c):** measure pad-token fraction in the profile; if high, revisit static padding / packing.

### 5.5 Promote when satisfied
The A/B gate ("beat cap-16384, FB ≤ 60s, higher sup tok/s") is **met**. The flag is kept default false out of
caution (one 32-prompt smoke + exact unit tests). To promote: set the launcher flag
`--gdn-replay-plan-use-stateful-prefix-cache` (env `OPD_GDN_REPLAY_PLAN_USE_STATEFUL_PREFIX_CACHE=true`) or
flip the `trainer.yaml` env default. Recommended pre-promotion check: a multi-step run confirming the loss
curve matches the cap-16384 path (the math is provably identical, so they should overlap).

---

## 6. Reproduce / operate (self-contained)

### Pin the checkout
```bash
export PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src
```
Without this, pytest can import `xorl` from a sibling checkout and give false confidence.

### How a run is launched
`write-trainer-control` renders `run.sh` + `desired.sha256` under
`/shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/`; an in-pod poller re-execs `run.sh` (full server
boot + the 1-step profiling pipeline, ~8–10 min). The trainer pods mount the `home-apanda` PVC at
`/workspace` with `PYTHONPATH=$XORL_REPO_ROOT/src`, so **edits to this checkout are live for the next launch —
no image rebuild**. Profiling runs use `--skip-optim-step` (no weight update, no checkpoint) → non-destructive.
The stack is currently single-tenant/free, but still check `kubectl get pods | grep er-opd-q36-mtp-ss-0605c`
and that the trainer is idle (last `…-run.log` ends `trainer-head cleanup rc=0`) before launching. If the
supervisor is armed, pause it before profiling:

```bash
touch /shared/opd-control/er-opd-q36-mtp-ss-0605c/supervisor.pause
```

### Reusable long-run launch args
The reusable args file for the free single-tenant stack is:

```text
experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt
```

It is updated to the measured handoff path: non-CP (`--trainer-ulysses-parallel-size 1`), EP32,
Quack MoE, DeepEP/SMS24, sample packing, compile on, linear-attention compile off, stateful GDN replay, and
static padding 1536. It intentionally keeps long-run settings (`--no-skip-optim-step`, checkpointing,
`--opd-async-sample-overlap`) rather than the profiling-only `--skip-optim-step` settings below.

To kick the long run from those args:

```bash
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py write-trainer-control \
  $(tr '\n' ' ' < experiments/opd_profile/k8s/launch_args_er-opd-q36-mtp-ss-0605c.txt)
```

### Baseline (cap-16384, stateful off)
```bash
env -u XORL_PROFILE_SERVER_FB -u XORL_PROFILE_SERVER_FB_SKIP -u XORL_PROFILE_SERVER_FB_COUNT \
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py write-trainer-control \
  --stack er-opd-q36-mtp-ss-0605c \
  --trainer-nodes 4 --student-replicas 2 --teacher-replicas 2 \
  --student-gpus 2 --teacher-gpus 2 \
  --num-steps 1 --prompts-per-step 32 --prompt-dataset-epochs 1 --max-opd-steps 1 \
  --prompt-len 512 --max-new-tokens 256 --conf-threshold 0.3 --mtp-static-padded-seq-len 1536 \
  --pipeline-chunk-size 16 --pipeline-prefetch-chunks 2 --pipeline-teacher-concurrency 2 \
  --trainer-ulysses-parallel-size 1 \
  --trainer-expert-parallel-size 32 \
  --trainer-ep-dispatch deepep --trainer-moe-implementation quack --trainer-deepep-num-sms 24 \
  --trainer-enable-packing --trainer-sample-packing-sequence-len 4096 \
  --trainer-fsdp-reduce-dtype fp32 \
  --trainer-enable-compile --no-trainer-compile-linear-attention \
  --gdn-replay-plan-max-packed-tokens 16384 \
  --skip-optim-step \
  --checkpoint-interval-steps 0 --no-checkpoint-save-best \
  --checkpoint-timeout 2400 --forward-backward-timeout 2400 --checkpoint-eval-batch-size 32
```

### Stateful (this session's path)
Same command **plus** `--gdn-replay-plan-use-stateful-prefix-cache`. Generated-config checks:
```text
linear_replay_plan_max_packed_tokens: 16384
linear_replay_plan_use_stateful_prefix_cache: true   # false for baseline
expert_parallel_size: 32                              # measured current winner; EP16 is runner-up
ep_dispatch: deepep                                   # alltoall only for explicitly labelled diagnostic controls
deepep_num_sms: 24                                    # current DeepEP default for future runs
fsdp_reduce_dtype: fp32                               # bf16 for §4.6 candidate
enable_compile: true
compile_linear_attention: false
enable_packing: true
sample_packing_sequence_len: 4096
```

### Recommended throughput handoff: non-CP EP32 / Quack / DeepEP24
This is the current best measured configuration under the required Quack + DeepEP24 defaults. It **does not**
use the CP path. CP is only selected when `--trainer-ulysses-parallel-size 2` is set; this command pins
`--trainer-ulysses-parallel-size 1` so the trainer stays on the non-CP/Ulysses path.

```bash
env -u XORL_PROFILE_SERVER_FB -u XORL_PROFILE_SERVER_FB_SKIP -u XORL_PROFILE_SERVER_FB_COUNT \
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py write-trainer-control \
  --stack er-opd-q36-mtp-ss-0605c \
  --trainer-nodes 4 --student-replicas 2 --teacher-replicas 2 \
  --student-gpus 2 --teacher-gpus 2 \
  --num-steps 3 --prompts-per-step 32 --prompt-dataset-epochs 3 --max-opd-steps 3 \
  --prompt-len 512 --max-new-tokens 256 --conf-threshold 0.3 --mtp-static-padded-seq-len 1536 \
  --pipeline-chunk-size 16 --pipeline-prefetch-chunks 2 --pipeline-teacher-concurrency 2 \
  --trainer-ulysses-parallel-size 1 \
  --trainer-expert-parallel-size 32 \
  --trainer-ep-dispatch deepep --trainer-moe-implementation quack --trainer-deepep-num-sms 24 \
  --trainer-enable-packing --trainer-sample-packing-sequence-len 4096 \
  --trainer-fsdp-reduce-dtype fp32 \
  --trainer-enable-compile --no-trainer-compile-linear-attention \
  --gdn-replay-plan-max-packed-tokens 16384 --gdn-replay-plan-use-stateful-prefix-cache \
  --skip-optim-step \
  --checkpoint-interval-steps 0 --no-checkpoint-save-best \
  --checkpoint-timeout 2400 --forward-backward-timeout 1800 --checkpoint-eval-batch-size 32
```

Measured run: `q36mtp-20260609T174640Z-2s2t`, clean validation. Excluding the warmup row, the two steady rows
total `309312` valid tokens with `430.99s` total trainer FB (`1.39` ms/valid FB). Aggregate steady
forward/backward throughput is `717.68` valid tok/s world; aggregate steady end-to-end throughput is `73.88`
supervised tok/s/GPU (`591.07` world). The OPD loop does not emit true model MFU in this path
(`mfu`/`flops`/`tflops` absent from the run artifacts), so use FB ms/valid and supervised tok/s as the
throughput/MFU proxy unless adding FLOPs instrumentation.

Do not hand off the CP2 32-prompt command as the fastest config. The full-shape CP2 attempt
`q36mtp-20260609T164154Z-2s2t` reached `216.30s` FB for chunk 1 (`16384` valid; `13.2` ms/valid), and chunk 2
did not produce a JSONL row before stop.

### FB profile capture that succeeded
The full 32-prompt profile attempt can timeout under profiler. For an actual CP/stateful FB op profile, use the
same stateful command but capture the first FB and shrink to one 8-prompt chunk:

```bash
XORL_PROFILE_SERVER_FB=1 XORL_PROFILE_SERVER_FB_SKIP=0 XORL_PROFILE_SERVER_FB_COUNT=1 \
PYTHONPATH=/home/apanda/xorl-mtp-singleshot-port-20260602/src \
python experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py write-trainer-control \
  --stack er-opd-q36-mtp-ss-0605c \
  --trainer-nodes 4 --student-replicas 2 --teacher-replicas 2 \
  --student-gpus 2 --teacher-gpus 2 \
  --num-steps 1 --prompts-per-step 8 --prompt-dataset-epochs 1 --max-opd-steps 1 \
  --prompt-len 512 --max-new-tokens 256 --conf-threshold 0.3 --mtp-static-padded-seq-len 1536 \
  --pipeline-chunk-size 8 --pipeline-prefetch-chunks 1 --pipeline-teacher-concurrency 1 \
  --trainer-ulysses-parallel-size 2 \
  --trainer-expert-parallel-size 8 \
  --trainer-ep-dispatch deepep --trainer-moe-implementation quack --trainer-deepep-num-sms 24 \
  --trainer-enable-packing --trainer-sample-packing-sequence-len 4096 \
  --trainer-fsdp-reduce-dtype fp32 \
  --trainer-enable-compile --no-trainer-compile-linear-attention \
  --gdn-replay-plan-max-packed-tokens 16384 --gdn-replay-plan-use-stateful-prefix-cache \
  --skip-optim-step \
  --checkpoint-interval-steps 0 --no-checkpoint-save-best \
  --checkpoint-timeout 2400 --forward-backward-timeout 1800 --checkpoint-eval-batch-size 8
```

Expected artifacts:

```text
<run-dir>/server_output/fb_profiles/fb_trace_call0.json.gz
<run-dir>/server_output/fb_profiles/fb_keyavg_call0.txt
<run-dir>/artifacts/opd_profile.jsonl
```

### BF16 FSDP reduce check
Same stateful command **plus**:

```bash
--trainer-fsdp-reduce-dtype bf16
```

Measured BF16 run: `q36mtp-20260609T030727Z-2s2t` (`51.12s` FB, `50944` valid, `1.00` ms/valid,
validation succeeded). Same-day FP32 follow-up `q36mtp-20260609T031333Z-2s2t` was faster (`28.40s` FB,
`43880` valid, `0.65` ms/valid), so BF16 is **not promoted** without the repeated §5.1 comparison.

### DP replication / shard narrowing negative control
Same stateful command **plus**:

```bash
--trainer-data-parallel-replicate-size 2 --trainer-data-parallel-shard-size 16
```

Measured negative run: `q36mtp-20260609T030203Z-2s2t` (`81.72s` FB, `32776` valid, `2.49` ms/valid,
validation succeeded). This knob is kept in the launcher for future FSDP experiments, but it is not the next
path for this workload.

### EP topology smokes
Same stateful command, plus the relevant topology flags:

```bash
# Wider EP through DeepEP. Keep SMS24 as the baseline unless explicitly sweeping SMS as a diagnostic.
--trainer-ep-dispatch deepep --trainer-expert-parallel-size 32 --trainer-deepep-num-sms 24
```

Measured topology runs are in §4.7. Do not promote another topology change without a clean 3-step win over
the EP32/DeepEP24 default and a clean `OPD pipeline validation succeeded`.

### Watch a run
```bash
LOG=$(ls -t /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/*-run.log | head -1)
# terminal markers: "OPD pipeline validation succeeded" | crash sigs | "trainer-head cleanup rc="
grep -nE "OPD pipeline validation succeeded|trainer-head cleanup rc=|Traceback|CUDA out of memory|Assertion failed" "$LOG"
```
Headline timings + replay/stateful counters are in the big `OPD throughput step` JSON line in the run log and
in `<run-dir>/artifacts/opd_profile.jsonl` (`trainer_forward_backward_s`, `…stateful_context_rounds`, etc.).
Result dir: `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/<run-id>/`.

### Local tests + lint (run with the pinned PYTHONPATH; lint/format with CI's pinned ruff 0.11.4)
```bash
PYTHONPATH=$PYTHONPATH pytest \
  tests/models/test_qwen3_5_linear_attention_mapping.py \
  tests/ops/test_linear_attention_singleshot_mask.py \
  tests/mtp/test_singleshot.py \
  tests/server/runner/test_opd_runner.py tests/server/runner/test_model_runner_profile.py \
  tests/models/test_qwen3_5_singleshot_mask.py tests/server/test_server_arguments.py -q
uvx ruff@0.11.4 check  <touched files>
uvx ruff@0.11.4 format --check <touched files>
```
Note: the local environment's ruff is newer than CI's pin (0.11.4) and disagrees on a few wraps — always
format with `uvx ruff@0.11.4` to match CI. The repo test
`tests/test_example_assets.py::test_examples_and_experiments_avoid_personal_absolute_paths` fails on
pre-existing `/home/apanda` paths in `experiments/local_benchmark/**` and `qwen36_singleshot_mtp/**` — not
related to this work.

---

## 7. Code map

- Replay plan + stateful schedule + alignment constant: `src/xorl/mtp/singleshot.py`
  (`RolloutReplayLinearPlan`, `RolloutReplayStatefulSchedule`, `build_rollout_replay_linear_plan`,
  `_build_rollout_replay_stateful_schedule`, `_STATEFUL_REPLAY_CAPTURE_ALIGN`,
  `summarize_rollout_replay_linear_plan`, `shard_singleshot_mtp_opd_batch`).
- GDN replay dispatch + packed-varlen stateful executor:
  `src/xorl/ops/linear_attention/layers/gated_deltanet.py`
  (`_forward_with_replay_plan`, `_forward_with_replay_plan_stateful_prefix_cache`, `_index_state`,
  `_forward_with_replay_plan_packed_chunks` = cap-16384 fallback, `_forward_with_cp_replay_mask`,
  `_forward_standard`).
- Qwen3.6 SingleShot CP mask plumbing: `src/xorl/models/transformers/qwen3_5/modeling_qwen3_5.py` and
  `src/xorl/models/transformers/qwen3_5_moe/modeling_qwen3_5_moe.py`
  (`_mask_input_for_structured_cp`, linear-attention replay masks preserved under `cp_context`).
- Server CP prep path: `src/xorl/server/runner/runner_dispatcher.py` defers generic CP sharding for SingleShot
  OPD; `src/xorl/server/runner/model_runner.py` shards prepared replay batches and keeps ring CP fail-fast.
- Short-conv packed per-segment cache + final state: `src/xorl/ops/linear_attention/modules/short_conv.py`
  (`_segmented_depthwise_causal_conv_per_segment_prefix`, `_segment_conv_final_state`).
- Chunk GDN kernel (per-chunk `h` lives here, for §5.2):
  `src/xorl/ops/linear_attention/ops/gated_delta_rule/chunk.py`,
  `src/xorl/ops/linear_attention/ops/common/chunk_delta_h.py`.
- Server FB replay telemetry + FB profiler: `src/xorl/server/runner/model_runner.py`
  (`_singleshot_mtp_replay_profile`, `_log_singleshot_mtp_replay_profile`, the `XORL_PROFILE_SERVER_FB` block).
- OPD profile aggregation (JSONL keys): `scripts/opd/run_opd_pipeline.py`.
- Trainer launcher / control generator: `experiments/opd_profile/k8s/q36_singleshot_reprogrammable_slots.py`
  (flags `--gdn-replay-plan-max-packed-tokens`, `--gdn-replay-plan-use-stateful-prefix-cache`,
  `--trainer-expert-parallel-size`, `--trainer-deepep-num-sms`, DP shard/replicate controls).
- Static trainer manifest (env defaults): `experiments/opd_profile/k8s/qwen36_singleshot_mtp_coderforge_confadapt_4node/trainer.yaml`.

## 8. Tests added/changed this session

- `tests/models/test_qwen3_5_linear_attention_mapping.py`: per-segment packed short-conv cache vs a Python
  loop over single-segment `ShortConvolution(x_i, cache=cache_i)` (incl. sub-kernel segments) + no-cache
  per-segment final-state.
- `tests/ops/test_linear_attention_singleshot_mask.py`: faithful `cu_seqlens`-aware fake chunk kernel
  (per-segment `initial_state`/`final_state`); stateful==dense for single-row, multi-row (gate+conv4),
  coarse-boundary capture (`_STATEFUL_REPLAY_CAPTURE_ALIGN` monkeypatched small), and gradient parity.
- `tests/mtp/test_singleshot.py`, `tests/server/runner/test_opd_runner.py`: existing summary/profile contracts
  still pass; new stateful telemetry keys are additive; CP prep now verifies local sequence/loss/cache shards
  plus full replay masks.
- `tests/distributed/test_qwen3_5_singleshot_cp.py`: 2xH100 Ulysses CP smoke for a tiny Qwen3.6-style
  SingleShot replay batch; forbids the old full-hidden `gather_outputs` path, compares gathered CP output and
  reduced OPD loss with a full-sequence reference, and checks suffix-only backward gradients through the
  sharded replay-state handoff.
- `tests/ops/test_linear_attention_singleshot_mask.py`, `tests/models/test_qwen3_5_singleshot_mask.py`,
  `tests/server/runner/test_runner_dispatcher.py`: sharded CP replay segment planning, full structured mask
  shape, and dispatcher deferral coverage.
- `tests/experiments/test_q36_singleshot_reprogrammable_slots.py`: generator coverage for DP shard/replicate,
  EP size, and `deepep_num_sms` render/summary/validation knobs used by the topology smokes.
