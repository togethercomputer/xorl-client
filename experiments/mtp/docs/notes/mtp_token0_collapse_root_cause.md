# MTP token-0 collapse — root cause: q>1 decode CUDA graphs were never valid on hybrid (GDN) models

> **ROUND 2 UPDATE (2026-06-12, sglang `9190ee358`):** the batched-eager collapse
> ("bug #12", intermittent per-prompt token-0 under bs>1 eager native MTP) is the SAME
> stride-1 query_start_loc root cause: request *i* in a batched q>1 decode processed flat
> token *i*, which lies inside request 0's row — cross-request state bleed for every request
> but the first. Additionally, native MTP was **never actually speculating**: the stride-1
> truncation degraded it to serial decode (1 token/step at q tokens of compute), and the
> commit path trusted drafts without verification. `9190ee358` rebuilds q>1 decode on the
> eagle-style verify kernels (per-step intermediate Mamba states + post-step scatter at the
> verified boundary) and adds hf-exact draft prefix-match acceptance. Validated: serial AND
> batched (16×256tok) coherent, zero degenerate rollouts, run-to-run deterministic —
> eager+cache 18.7 tok/s aggregate, **graphs-on 74.0 tok/s (4.0×)**.
>
> **Final prod recipe:** restart samplers on `9190ee358` WITH
> `--enable-mtp-static-q-len-cuda-graph --mtp-static-cuda-graph-k-list 4
> --enable-mtp-adaptive-hf-exact-q-len-cuda-graph --mtp-adaptive-cuda-graph-kmax-list 4`
> (banding still OFF; the graph flags also auto-size the required Mamba speculative state
> cache). Eager-only servers must set `--native-mtp-state-cache-steps 7`; batched q>1
> decode without the cache now fails loudly instead of silently corrupting.

**Date:** 2026-06-12 (investigation night of 06-11)
**Follow-up to:** `mtp_student_sampler_token0_collapse_handoff.md`
**Repo:** `/home/apanda/xorl-sglang-internal` (fixes on working tree on top of `e8cbba250`)
**Debug env:** local TP=2 server on dev-pod GPUs 2,3 (`/tmp/mtp-token0-debug/launch.sh`), prod flags incl.
`--enable-mtp-{static,adaptive-hf-exact}-q-len-cuda-graph --mtp-static-cuda-graph-k-list 4
--mtp-adaptive-cuda-graph-kmax-list 4 --enable-mtp-adaptive-hf-exact-canonical-q-banding --mtp-adaptive-canonical-q-lens 7`

## TL;DR

The handoff's bisection guess (adaptive-cuda-graph / banding) was directionally right but the mechanism
is now precisely known. Three independent bugs, all confirmed by experiment:

1. **GDN q>1 decode graphs computed garbage from capture** (primary).
   `MambaAttnBackendBase._capture_metadata/_replay_metadata` always loaded the **stride-1**
   `cached_cuda_graph_decode_query_start_loc` for DECODE-mode graphs. Native-MTP decode graphs have
   q tokens/request (q=4 seed, q=7 banded steady); the GDN varlen-conv + delta-rule kernels were told
   `query_start_loc=[0,1,2,...]` while bs·q tokens were present → draft-position hidden states are
   garbage/zero → lm-head argmaxes to token 0 (`!`) on the corrupt server state, or to incoherent
   tokens on a fresh server. **Every q>1 graph replay was corrupt; every eager step was fine.**
   This is why `k_toks` 2/3 (never hits captured q-lens) decode cleanly while k=4 (q=4 seed graph,
   q=7 banded steady graph) collapses at the first draft step.

2. **`decode_q_len_per_req` was never plumbed through `HybridLinearAttnBackend`** (replay misrouting).
   `CudaGraphRunner` only forwards `decode_q_len_per_req` to backends whose
   `init_forward_metadata_{capture,replay}_cuda_graph` signature accepts it. The hybrid wrapper didn't,
   so the inner flashinfer backend fell back to `decode_cuda_graph_q_len_per_req.get(bs, 1)` — a dict
   that each successive runner capture **overwrites** (q=1 → 4 → 7, last-write-wins = 7 for every bs).
   Plain q=1 decode replay therefore updated the *(bs,7)* decode-as-prefill wrappers instead of its own
   q=1 decode wrappers → the replayed q=1 graph ran on a stale capture-time plan. Benign at tiny
   seq-lens (single KV split) — which is why short probes on a fresh server look fine — catastrophic at
   prod context lengths and after enough drift: this is the wedged-prod "plain decode also emits token-0
   with uniform logits over ids [1,0,2,4,3]" state observed on both samplers.

3. **Canonical q-banding's recompute row duplicated the pending token** (exposed once #1 was fixed).
   `prepare_for_decode` builds the banded steady row as `output_ids[-needed:] + pending`, but commit
   already appends the pending tokens to `output_ids`, so the row came out e.g.
   `[198, 8160, 579, 579]` instead of `[…, 198, 8160, 579]` — the duplicate is then committed and the
   rollout degenerates into loops ("Here's a a a a…").
   **Residual (not yet fixed):** the banded recompute region is appended at *fresh* positions
   (`seq_lens.add_(q)`, no rewind by `recompute_len − a_prev`), so the re-fed committed tokens appear
   twice in the attention view at shifted positions. hf-exact requires rewinding the KV base for the
   overlap. Until that lands, **canonical banding should stay OFF in prod**.

Also fixed: `MambaAttnBackendBase.init_cuda_graph_state` appended a fresh set of per-bs metadata
tensors on every `CudaGraphRunner` construction (3× for the MTP capture set) — now idempotent.

## Why history was intermittent (0.00 ↔ 0.98 per run, binary per rollout)

conf_adapt shrinks the window on low confidence, so a step's q-len lands on a captured graph
(corrupt) or on eager fallback (clean) depending on content/confidence. A rollout survives iff its
steps stay eager. The 06-10 snapshot made seed q=4 graph-eligible (`static_q_lens=[4,7]` in the
capture banner) → suffix-mode rollouts corrupt at the *first* draft step → ~99%.
Loss/`opd_top1_agreement` stayed "healthy" because the frozen teacher agrees with the student on the
corrupt context (vacuous self-distillation) — hence the new rollout-coherence gate (below).

## Experimental ledger (key probes)

| probe | server | result |
|---|---|---|
| plain greedy, k=2, k=3 | wedged prod samplers | token-0 from step 1 (uniform logits) — both samplers, serial and batched ×8 |
| k=4 / k=5 | wedged prod | token-0 at first graphed step; eager steps (k=5 seed q=5) emit real tokens |
| `/flush_cache` | wedged prod | does NOT clear the state |
| pre-sync rollouts (run 204611Z step 0, 20:52 < first weight sync 20:53) | prod | already 0.99 corrupt → weight sync exonerated as the *primary* cause |
| plain greedy | fresh debug server, prod flags | clean (short ctx — see bug 2 caveat) |
| k=4 | fresh debug, unfixed | first 2 tokens real then incoherent soup; trace: every `graph=True` step garbage, `graph=False` steps healthy (logit l2 ~1000s vs uniform) |
| k=4 | debug + GDN qsl fix | graphed steps commit *real* tokens; rollout loops "a a a" (bug 3) |
| trace `input_row_token_ids` | debug + fix | banded steady row `[198, 8160, 579, 579]` → dup pending token confirmed |

## Fixes (sglang working tree, to be committed on apanda-dev)

- `layers/attention/hybrid_linear_attn_backend.py`
  - `_decode_query_start_loc_source(q)`: per-q cached stride-q `query_start_loc` for DECODE graphs;
    used by `_capture_metadata` and `_replay_metadata` (incl. padding fill `(bs−pad)·q`).
  - `init_cuda_graph_state`: idempotent across multiple CudaGraphRunner constructions.
  - `MambaAttnBackendBase` + `HybridLinearAttnBackend` `init_forward_metadata_{capture,replay}_cuda_graph`
    now accept `decode_q_len_per_req` and forward it to children that accept it (capture also derives
    q = num_tokens // bs when not passed).
- `managers/schedule_batch.py`: banded steady-row build excludes the pending tail from the
  committed-context slice (+ fail-loud invariant that pending == tail of output_ids).

## xorl pipeline: rollout-coherence gate (landed in worktree)

`scripts/opd/run_opd_pipeline.py`: after every `_student_sample_for_opd_batch`, each rollout with ≥8
generated tokens is checked for degenerate-token fraction (token id 0 + the MTP mask token id).
`--rollout-coherence-action {fail,warn,off}` (default **fail**) and
`--rollout-coherence-max-degenerate-frac` (default 0.5); env `OPD_ROLLOUT_COHERENCE_*`.
Metrics: `rollout_coherence_{checked,degenerate,worst_degenerate_frac}`.

## Post-fix validation ledger (debug servers C/D/E/F)

- Server D (qsl fix + dup fix, banding ON): banded rows now built correctly
  (`[…,198,8160,579]` sliding, commits match eager ground truth for the first tokens), but long
  generations still degrade — the no-rewind position flaw (bug 3 residual) stands. Banding stays OFF.
- Server E (qsl fix, graphs ON, banding OFF): no token-0, no loops, but **token stutter**
  ("Here's 's", "?capital capital") — a 4th latent bug in the graphed DECODE q>1 path, previously
  unobservable because the path always emitted token-0 garbage. Strong suspect: GDN recurrent-state
  anchoring across q>1 *decode* steps — `GDNAttnBackend.forward_decode` (varlen-conv +
  `kernel_dispatcher.decode`) writes post-row state incl. mask slots, while the eager EXTEND path
  (`kernel_dispatcher.extend` + `_track_mamba_state_extend`) keeps state correct.
- Server F (fixed code, **NO MTP graph flags** → all-eager MTP): **fully coherent**. k=4 suffix and
  prefix probes, 96-192 tokens, serial and concurrent ×8: frac_tok0 = 0.00, fluent reasoning,
  no stutter. This is the validated prod configuration.

## Prod recommendation (when resuming the MTP-OPD stack)

1. Restart both student samplers (touch `run.sh` revision) so they pick up the fixed sglang code —
   the wedged in-memory state cannot be cleared via HTTP (`/flush_cache` does not help).
2. **Remove ALL MTP q>1 graph flags** from the student sampler launch for now:
   `--enable-mtp-static-q-len-cuda-graph --mtp-static-cuda-graph-k-list 4
   --enable-mtp-adaptive-hf-exact-q-len-cuda-graph --mtp-adaptive-cuda-graph-kmax-list 4
   --enable-mtp-adaptive-hf-exact-canonical-q-banding --mtp-adaptive-canonical-q-lens 7`.
   Native MTP then decodes eagerly — validated correct. The graphed q>1 path needs the GDN
   state-anchoring fix (and banding additionally needs the position rewind) before re-enabling;
   note these graphs NEVER produced correct rollouts, so disabling them does not regress any
   actually-achieved quality, only the (corrupt) throughput numbers.
3. Re-validate with `scripts/opd/probe_token0_collapse.py --server-url http://<sampler>:30060` —
   suffix-mode frac_tok0 must be 0.00 and text coherent for ≥256 tokens before the trainer restarts.
4. The pipeline coherence gate (default fail) protects any future regression of this class.

## Follow-ups for the perf agent (re-enabling q>1 graphs)

1. GDN state anchoring for DECODE q>1 graphs: the state slot must end the step at the commit anchor
   (end of recompute region), not after the mask slots; mirror whatever keeps the eager EXTEND path
   correct (`_track_mamba_state_extend` / chunked-extend `h` handling). This is the stutter bug.
2. Banding position rewind: rewind per-request seq_lens/positions by `recompute_len − a_prev` for the
   overlap region (and re-point req_to_token for the overlap positions); then re-run the throughput
   matrix from `project_sgl_sampler_throughput_mtp_buckets`.
3. The fresh-server "stale q=1 plan, benign at short ctx" → fully-wedged-uniform-logits transition on
   prod was never reduced to a single step; with the q plumbing fixed the misrouting is gone — treat
   as closed unless it reproduces post-fix.
