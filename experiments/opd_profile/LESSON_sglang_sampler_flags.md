# Lesson: don't cargo-cult conservative SGLang sampler flags

**Date:** 2026-06-05 · **Stack:** `er-opd-q36-35b-slots` (Qwen3.6-35B-A3B OPD) ·
**File:** `experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py` → `sampler_script`

## Symptom
The OPD step-10 **control eval** (1024 problems × 3 conditions = 3072 student
generations) took **~30 minutes**. A batched H100 sglang server should do that in
**~1–2 minutes** — i.e. it was ~15–30× too slow, on *every* run, for a long time.

## Root cause
The **student sampler** (the P2P weight-sync receiver) was launched maximally
conservatively:

```
--max-running-requests 1      # serial — ONE generation at a time
--disable-overlap-schedule
--disable-cuda-graph
```

The eval client correctly fired 128 concurrent requests (`asyncio.gather` +
`Semaphore(128)`), but `--max-running-requests 1` funneled them single-file
(~0.6 s/completion × 3072 ≈ 30 min). The **teacher** sampler, by contrast, ran
`--max-running-requests 128`. The slow flags were **inherited/cargo-culted** from
an older "deterministic receiver" config and never re-justified.

This is the *same failure mode* as `privileged: true` on GPU pods (see CLAUDE.md):
a conservative workaround gets forwarded into new templates and silently kills
performance, and agents keep "fixing around" the symptom instead of the cause.

## Fix
In `sampler_script`: enable cuda-graph, enable overlap-schedule, set
`--max-running-requests 1024`. Applied live via the reprogrammable control loop
(`write-student-inference-control`) — **no pod recreation**. sglang-0 reloaded
`ready=true`; cuda-graph captured cleanly on the Qwen3.6 GDN/linear-attention
model (the feared incompatibility never materialized). Eval: **~30 min → ~1–2 min**,
and per-step on-policy sampling during training sped up too.

## Takeaways
1. **"Slow but working" is a bug.** "It finishes eventually" hid a 15–30×
   regression. Sanity-check throughput against a back-of-envelope (3072 short
   generations on an H100 = minutes, not 30) instead of accepting the wall time.
2. **Question inherited infra defaults.** `--max-running-requests 1`,
   `--disable-cuda-graph`, `--disable-overlap-schedule`, `privileged: true` —
   all justified once "for stability/determinism" — were cargo-culted forward.
   Re-justify each per workload; don't copy.
3. **Determinism belongs (if anywhere) on the training path, not the eval.** A
   control eval just needs accurate greedy generations; batch-1 determinism buys
   nothing there and costs ~30×.
4. **When inference is mysteriously slow, read the launch args first:**
   `kubectl logs <sampler-pod> | grep -o 'max_running_requests=[0-9]*'` (and
   `disable_cuda_graph`, `disable_overlap_schedule`).

## Caveat / open item
These flags also govern the student sampler *during training* (P2P weight-sync
pause + on-policy generation). cuda-graph/overlap/1024 came up healthy serving;
the remaining check is that a full OPD run still **syncs weights cleanly and
produces a sane loss/eval** with the new flags (verify on the first run that uses
them). If a subtle train/infer-match or sync-pause regression appears, revert
`--disable-cuda-graph` first (overlap + high `max-running-requests` give most of
the speedup on their own).
