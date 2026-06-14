# Pipelined two-phase teacher prefill (hide teacher prefill under fwd_bwd)

**Owner:** (to assign) · **Status:** spec, not implemented · 2026-05-29

## Goal

Take the OPD step from ~52s → ~25s (fwd_bwd-bound) by hiding the teacher prefill,
**while keeping the student answer on-policy.** Today (run `er-opd5smg-053007`):
teacher-prefill wall ~24s + fwd_bwd ~23s run **serially** = ~52s/step. The teacher
prefill is ~99% the **CoT** (fixed per prompt) and ~1% the **answer** (on-policy).

## Design: split the teacher prefill into two phases

For each sample, teacher seq = `prompt + CoT + pause + answer`. Split it:

- **Phase A** = `prompt + CoT + pause`. **Fixed** (CoT is the dataset entry, pause is
  constant) — depends on nothing in the current step. `target_tokens` keep = **pause**
  positions (mask prompt+CoT). Returns the **pause hiddens**. **Prefetch Phase A for
  step N+1 during step N's fwd_bwd.** No staleness: these hiddens are independent of
  the on-policy answer (a pause position attends only to `prompt+CoT+pause` to its
  left — standard causal LM, verified assumption).
- **Phase B** = `prompt + CoT + pause + answer`. On-policy (needs the sampled answer).
  `target_tokens` keep = **answer** positions only (mask prompt+CoT+**pause**). With
  **radix on**, the `prompt+CoT+pause` prefix is served from Phase A's KV, so only the
  ~11 answer tokens are freshly computed. Returns the **answer hiddens**. Runs in step
  N's prepare (after sampling), cheap.

**Merge** Phase A's pause-hidden cache + Phase B's answer-hidden cache into the one
per-step teacher cache the fwd_bwd reads. Per sample, concat **pause rows (A) then
answer rows (B)**, in the student's kept-position order (pause precedes answer). The
KL is then on pause+answer — **identical supervision to today** (the pause is NOT
dropped; it's just computed in Phase A, where it's fresh).

### Why masking pause in Phase B is safe (and why radix now works)
The total kept set = A's pause ∪ B's answer = today's {pause, answer}. Masking pause
in B only avoids re-emitting it (A already did) — and it's exactly what lets radix
serve `prompt+CoT+pause` as a cached prefix in B without a `cached_kept` error (the
only fresh/kept positions in B are the answer). Each phase's kept positions are always
in the freshly-computed extension.

## Implementation

### Teacher (manifest, all 8 replicas)
- **Re-enable radix**: drop `--disable-radix-cache` (keep `--chunked-prefill-size
  16384`, `--enable-return-hidden-states`). Phase B must reuse Phase A's KV.
- **Radix capacity is the risk**: one step's prefixes = 128 × ~11k ≈ **1.4M tokens**
  vs KV budget ~1.8M. One-step-ahead means N+1's Phase-A prefixes may coexist with
  N's still-live prefixes → eviction/thrash. Mitigations: raise `--mem-fraction-static`
  toward 0.9, and/or limit prefetch depth to 1 step, and/or shrink `prompts_per_step`/
  batch. Measure radix hit-rate in Phase B (sglang logs `#cached-token` — should be
  ~the prompt+CoT+pause length on a hit; 0 = miss = eviction).

### Client (`/home/apanda/xorl-client-chat-completions/examples/on_policy_distillation.py`)
1. **Split `_teacher_cache_from_sglang`** into `_teacher_cache_phase_a` and
   `_teacher_cache_phase_b`, each building `target_tokens` for its kept set via
   `_teacher_hidden_cache_data` then POSTing `/teacher_hidden_cache`:
   - Phase A: build the `prompt+CoT+pause` sequence (no answer), keep=pause
     (`supervise_student_cot` semantics → mask prompt+CoT). Write to a phase-A cache file.
   - Phase B: build `prompt+CoT+pause+answer`, keep=answer only (mask prompt+CoT+pause).
     Write to a phase-B cache file.
2. **Cache merge** `_merge_phase_caches(a_path, b_path) -> merged_path`: per sample,
   load A rows (pause) + B rows (answer), concat in pause→answer order, write merged
   safetensors + `cache_indices_by_sample`. fwd_bwd reads the merged cache (unchanged).
3. **Prefetch orchestration** in the main loop: maintain a one-step-ahead future for
   Phase A. During step N's fwd_bwd, kick off Phase A for step N+1's prompt window
   (prompts known; CoTs are the dataset lookup). At step N+1: sample → Phase B (cheap,
   radix-served) → merge(prefetched A, B) → fwd_bwd. Phase A for N+2 launches during
   N+1's fwd_bwd. Keep `opd_prepare_concurrency` for intra-phase concurrency.
4. **On-policy guarantee**: Phase A uses no student weights (pure teacher forward of a
   fixed sequence) → prefetching it 1 step early introduces **zero** staleness. Phase B
   uses the current step's fresh sample → answer stays on-policy.
5. Gate the whole thing behind a config flag, e.g. `teacher_pipeline_phase=true`
   (default false → current single-phase path), so it's A/B-testable.

## Validation (before trusting in a run)
1. **Numerical equivalence**: for a fixed batch, the merged two-phase cache must equal
   the single-phase cache — cosine(kept hiddens) ~1.0 per sample, same row counts/order.
   (Adapt `examples/ab_teacher_cache.py`.) This is the correctness gate.
2. **Radix hit**: Phase B sglang logs show `#cached-token ≈ prompt+CoT+pause len`
   (hit), not 0 (eviction).
3. **Step time**: ~52s → ~25–30s; teacher wall no longer on the critical path.
4. **buffer_delta unchanged** vs run 1 at matched steps (the supervision is identical).

## Scope notes
- This is the on-policy version of the user's idea (Phase A ahead, Phase B at step N).
- The cheaper-but-stale alternative (prepare the *entire* step N+1 — sample + full
  teacher prefill — during N's fwd_bwd) is simpler but makes samples 1-step-stale; we
  rejected it to keep the answer on-policy.
- For the **no-pause run** (`student_prefill_count=0`), Phase A becomes a pure
  `prompt+CoT` KV-warm (nothing kept) and Phase B is the only kept (answer) — even simpler.
- Don't split fwd_bwd for "Phase B during student-forward" (the original point 2):
  Phase B is already ~free once the CoT is cached; the trainer-side fwd_bwd split is
  high-effort, low-ROI. Skip.
