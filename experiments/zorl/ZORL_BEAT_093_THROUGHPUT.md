# ZORL — Beat 0.93: THROUGHPUT

Per-step cost of a ZORL ES step and where the time goes. Numbers measured this
campaign on the cloned 8×TP2 pools (Qwen3.6-35B-A3B, 128 pairs / 256 candidates,
TRAIN_SIZE=256, TEACHER_FORCED_BATCH_SIZE=32). See also the older
`ZORL_THROUGHPUT_RUNBOOK_2026_06_05.md` and
`ZORL_SCORE_THROUGHPUT_FINDINGS_2026_06_05.md`.

## 1. Per-step breakdown (measured)

```
t_score ≈ 150–158 s   (scoring 256 candidates × 256 train examples, teacher-forced SFT)
t_apply ≈  31 s       (fresh_ab fold, raw recipe: MAX_UPDATE_NORM=0, momentum=0)
       ≈  52 s        (when MAX_UPDATE_NORM>0: a dry-run measure pass DOUBLES the fold)
step    ≈ 180–210 s   end-to-end  -> ~410–480 steps in a 24h cap; ~205 steps in 12h
```

**Scoring dominates (~3–5× the fold).** It is raw inference compute: 256 distinct
candidate LoRAs each evaluated on 256 examples (teacher-forced log-probs),
sharded across 8 servers with owner-routing, 3 workers/owner × batch 32.

## 2. Confirmed throughput facts

- **The trust-region dry-run is the single biggest avoidable fold cost.** With
  `MAX_UPDATE_NORM>0`, the fold runs twice (measure pass `apply_update=False`,
  then real fold) → t_apply ~52s vs ~31s. The beat-0.93 recipe uses
  `MAX_UPDATE_NORM=0`, so it pays the cheaper single-pass fold. (HYPER-L paid the
  double cost; ~207s/step vs ~180s/step here.)
- **`gpu_direct` fold beats the staging path.** `XORL_ZORL_FRESH_AB_FOLD=gpu_direct`
  folds chunk noise straight into base-weight shards (no memory-pool round-trip);
  norm stays a device tensor so the loop avoids host syncs. Already enabled on the
  pools.
- **`TEACHER_FORCED_LOGPROB_TRIM=1` is mandatory** with batched scoring: without
  it the input-logprob gather materializes `[seqs × positions × vocab]` fp32
  (~2.3 GiB for 96 in-flight seqs) and OOM-crashed all 8 replicas at mem-fraction
  0.90. With trim (answer suffix + 8 margin) the wave is ~0.9 GiB.
- **batch=32 + per-owner semaphore (3 workers)** keeps ~96 in-flight seqs ≤ the
  effective max-running-requests and ≤ `max-loras-per-batch=32` distinct LoRAs —
  the per-owner cap is what makes batch 32 safe (global 64-worker pools OOM'd).

## 3. Why throughput matters for the science

The prior best peaked at **step 235** and was still meaningfully training to step
421. The beat-0.93 (delayed-decay) plan needs the run to climb to step ~235 **then**
refine through ~step 350–400. At ~180s/step that climb+refine window is ~12–18h —
comfortably inside the 24h cap, but **scoring speed is the gating resource**: any
reduction in t_score directly buys more refinement steps and lets more
configurations be swept per pool-day.

## 4. Efficiency levers (ranked; not yet pursued this campaign)

1. **Scoring is the prize (~80% of step time).** It is teacher-forced prefill, so
   CUDA-graph (currently disabled for dynamic LoRA) wouldn't help much; the wins
   are in batching/scheduling: larger safe batch, fewer redundant prefixes,
   better owner load-balancing. Needs profiling on a live server (avoid
   perturbing an active run).
2. **Avoid `MAX_UPDATE_NORM>0` unless clipping is actually needed** — it doubles
   the fold for no benefit when the natural update norm (~17) is below the cap.
3. **Reduce TRAIN_SIZE** trades scoring time for a noisier ES gradient — the
   campaign settled on 256; smaller was not clearly better.
4. **Node-failure resilience** (see INFRA §4): a lost node forces a full pool
   reset + cold restart, wasting all accumulated steps — the largest *effective*
   throughput hit observed this session. Session checkpoint/restore or
   higher-priority pods would prevent it.

## 5. Open profiling TODO

No live torch-profiler trace was captured this campaign (didn't want to perturb
running climbs). To genuinely cut t_score, capture one scoring wave with
`sglang.profiler` against an idle clone pool and analyze the prefill kernels /
LoRA SGMV overlap (see the `sglang-torch-profiler-analysis` skill).
