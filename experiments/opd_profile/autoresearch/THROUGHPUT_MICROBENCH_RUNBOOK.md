# OPD Throughput Handoff

Current as of 2026-06-15 23:01Z.

This file is now the canonical summary of the OPD MFU/throughput work. The
old chronological ledger was archived at:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_HISTORY_20260615.md`

## Current Answer

The current audited/promotable 4-node logical MFU is still:

- Logical MFU: `0.010405075057463931` (`~1.04%`).
- Valid-token-scaled logical MFU: `0.0002947112240841216` (`~0.029%`).
- Source:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`

The newer real-cache pack sweep is not a 4-node promotion. It has only:

- 1-node logical MFU `0.008961757161845595`.
- 2-node logical MFU `0.009395209742988759`.
- Source:
  `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_realcache_pack_sweep_20260615.json`

In simple terms: we did not get to 10% MFU. We proved that the low MFU is not
just external data-loader starvation. Even server-resident replay with cached
inputs stays around the same scale, and fixed-payload repeat data only raised
the measured 4-node logical MFU to `0.018305077674613597` at repeat8. That
rejects repeat-data on the current 64-prompt shape; it does not reject the
newly authorized real larger-batch track, where `--prompts-per-step` itself
increases and therefore raises real tokens per rank. The larger-batch track is
documented in:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/LARGER_BATCH_PACKER_RUNBOOK.md`

The trainer is spending most of its work on full long-sequence model
forward/backward for very sparse OPD supervision.

## What Was Accomplished

1. **Made the MFU denominator explicit.**

   The work split OPD throughput accounting into dispatcher-executed tokens,
   real student tokens, teacher-forward tokens, and valid target tokens. This
   closed the confusion where the run looked like "1% MFU" by one denominator
   and much worse by useful-token accounting.

2. **Moved the 64-prompt runtime config from slow/four-call behavior to the
   current fast tiers.**

   The current 64-prompt science-facing tiers remain:

   | Tier | Config | Key knobs | Wall result | Safety |
   | --- | --- | --- | --- | --- |
   | Fastest | AMDAHL-014 | `moe_implementation: quack`, `ep_dispatch: deepep`, `deepep_num_sms: 24`, `opd_prepare_batch_size: 64`, `opd_microbatch_size: 64`, `opd_pipeline_rl: true` | `step_total_s~13.22` | one-step-stale samples; not K3-cleared |
   | Strict fresh, fastest | AMDAHL-020 | quack + DeepEP SMS36, prep64/micro64, strict prepare chunk4, 1 sampler | `step_total_s~13.41` | fresh; quack+DeepEP K3 failed |
   | Strict fresh, 2 samplers | AMDAHL-018 | quack + DeepEP SMS36, prep64/micro64, strict prepare chunk4, 2 samplers | `step_total_s~14.03` | fresh; quack+DeepEP K3 failed |
   | Correctness fallback | AMDAHL-008 | safe dispatch, prep64 | baseline | use when static/K3 correctness is mandatory |

   These tiers are still throughput recommendations, not a fully gated 10% MFU
   result.

3. **Fixed major memory/fit blockers in the OPD loss path.**

   Useful engine work included:

   - streaming/low-memory reverse KL;
   - vocab-parallel KL/lm-head TP plumbing;
   - dtype fixes for the 4-node no-CP lm-head-TP replay;
   - all-layer SGLang-cache replay support;
   - selected-layer OPRD capture/fetch improvements;
   - skip-empty-cache-after-optimizer-step proof;
   - packed-row source provenance diagnostics.

   The dtype-fixed 4-node replay is the current audited value, but it is still
   only `~1.04%` logical MFU.

4. **Found the strongest raw trainer-side speed lever so far: rank-local
   packed-row batching.**

   AMDAHL-077 changed row batching so each trainer rank groups only rows that
   already belong to that rank. This avoided the earlier source-rank ownership
   bug and produced the strongest comparable 2-node raw speed result:

   - Same-server no-rowbatch control:
     `server_forward_backward_s=3.548186`.
   - Rank-local rowbatch2:
     `server_forward_backward_s=2.936666` over 11 measured rows.
   - Server wall improvement: `17.23%`.
   - Raw 2-node logical MFU estimate: `0.014218` (`~1.42%`).
   - Artifact:
     `/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-77-ranklocal-stability12-chunk4-serveronly-20260615T091946Z.jsonl`

   This is a real systems win, but it is not promotable because the static K3
   gate fails.

5. **Turned the AMDAHL-077 K3 failure from an unknown into a localized model-path
   parity problem.**

   The current full static gate for quack+DeepEP SMS36 failed:

   - Artifact:
     `/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_gated_summary.json`
   - Status: `fail`.
   - Coverage: `32/32` prompts, `2820/2820` tokens.
   - Mean K3: `0.001628991`, above the `1e-3` threshold.
   - p95 K3: `0.003193`, below the `1e-2` threshold.
   - Max K3: `0.622587456`.
   - Worst trace: `train.parquet:rg0:row2`, output position `22`, token `17880`.

   Follow-up diagnostics showed:

   - exact SGLang R3 routes/weights made K3 much worse, so route replay is not
     the bridge;
   - router FP32 moved the failure but did not pass;
   - GDN beta rounding matched SGLang but did not solve the residual drift;
   - full-attention recomputation reproduced existing drift rather than creating
     it;
   - the first nonzero row2069 drift appears at layer0 attention/post-attention
     residual and is then repeatedly amplified by RMSNorm.

6. **Kept science/throughput coordination clean.**

   The shared science channel was kept separate from this microbench handoff:

   `/shared/apanda/opd_throughput_science_channel.md`

   No faster config was promoted as strict-fresh plus K3-gated. The science
   side should therefore treat AMDAHL-008 as the correctness fallback and the
   AMDAHL-018/020 strict-fresh tiers as speed candidates with current K3 risk.

## Next Track: Real Larger Batch

The active next throughput hypothesis is not another fixed-64 packing or repeat
probe. It is:

```text
real_tokens_per_rank = total_real_tokens_per_step / dp_size
```

The current 64-prompt OPD step has only roughly `2.2k` real student tokens per
rank on the 4-node dp32 trainer. The target is a true `256-512` prompts/step
experiment with `sample_packing_sequence_len: 16384`, `sample_packing_strategy:
balanced_dp` or `best_fit`, and `sample_packing_on_oversized: error`, using the
server packer from merged PR #383. This is a new experiment, not a current MFU
claim.

Important boundaries:

- `balanced_dp` and `best_fit` are still negative/diagnostic at the old
  64-prompt batch because they make rows too small.
- The packer strategy fields live in the trainer server config, not candidate
  `client_args`.
- The dirty local `/home/apanda/xorl-apanda-dev` checkout was not sufficient at
  the time of this update; use a fresh/synced engine worktree that contains PR
  #383 before launching.
- Do offline `packing_microbench.py` sizing first, then a 1-2 step fit smoke,
  then a short throughput smoke, then the K3 gate.
- Do not promote without same-workload throughput, `dispatcher_dummy_batches ==
  0`, no oversized/dropped samples, and passing K3.

Use `LARGER_BATCH_PACKER_RUNBOOK.md` for the exact commands and config/candidate
shape.

## What Did Not Work

These were tested or diagnosed and should not be presented as current
solutions.

| Attempt | Result |
| --- | --- |
| Fixed-64 repeat data | AMDAHL-041 repeat1/2/4/8 only reached logical MFU `0.0142401777`, `0.0162517014`, `0.0171767331`, `0.0183050777`; it saturates below 2%. This does not rule out a true larger `--prompts-per-step` experiment. |
| pack2304 | Regressed on real replay (`5.5459s`) and is not a stable promotion. |
| no-defrag/no empty-cache as a main lever | No-defrag OOMed; skip-empty-cache was a modest cleanup win only. |
| CPU GC skip | Regressed server/API wall. |
| DeepEP SMS24 on the real-cache replay path | Regressed versus SMS36 for this path. |
| EP4 DeepEP / alltoall dispatch | Either loss/KL drift or slower path; not a promotion. |
| `reshard_after_forward:false` | Flat versus AMDAHL-048. |
| forward-only/backward-only FSDP prefetch | Forward prefetch was a useful small win, but not a correctness-cleared standalone promotion. |
| global/executor packed-row batching | Shifted loss/KL or broke source-rank invariants. |
| rank-local rowbatch2 | Fastest raw 2-node lever so far, but blocked by failed K3 gate. |
| HSDP / DP-replicated no-CP lm-head TP | Topology can start, but the 2-node corrected run was slower (`3.674399s`, `~1.14%` raw 2-node MFU) and 1-node full64 OOMed. |
| exact-R3 route replay | Made K3 far worse; do not spend more pods on it without a new model-forward hypothesis. |
| lm-head FP32, final RMSNorm/lm-head scoring, loss extraction | Ruled out as primary K3 causes. |
| router tie policy / router FP32 as a fix | Diagnostic only; did not clear K3. |
| GDN beta rounding / fused GDN projections / recurrent GDN | Useful diagnostics, not a K3/MFU promotion. |
| shared-prefix repacking for the current AMDAHL-047/077 capture | Low leverage: the real capture has 64 unique prompt prefixes and zero shared-prefix token saving under current grouping. |

## Current Engine Branch State

Engine worktree:

`/home/apanda/xorl-opd-repeat2-diagnostics`

Branch:

`codex/opd-repeat2-diagnostics-20260615`

Diagnostics PR:

`https://github.com/togethercomputer/xorl-internal/pull/376` is closed as a
superseded diagnostics branch.

Clean replacement PRs:

- `https://github.com/togethercomputer/xorl-internal/pull/381` -
  lm-head TP checkpoint-load parameter sync.
- `https://github.com/togethercomputer/xorl-internal/pull/380` -
  replay checkpoint/prefetch/defrag knobs.

Merged upstream dependency for the larger-batch track:

- `https://github.com/togethercomputer/xorl-internal/pull/383` -
  dp-aware/best-fit server packing plus fail-loud oversized policy, merged into
  `apanda-dev` at `4f9f90cb2fcab48f0c15f14a975a178974e84e84`.

Current head:

`7cc54ab7 Add packed row source provenance diagnostics`

Important commits on the branch include:

- packed row source provenance diagnostics;
- DeepEP diagnostic env threading for K3 launchers;
- GDN beta-rounding parity harness update;
- router margin FP32 diagnostics;
- router top-k policy diagnostics;
- SGLang route trace refresh launcher;
- route replay guards for zero SGLang routing weights.

No engine change from the diagnostics branch is a 10% MFU promotion. The branch
is useful as diagnostic history and as the source of the rejected raw rowbatch
candidate, but it is not the merge target.

## Current Cluster State

As of the last check, the slots stack has no trainer pods running. Only these
slots pods remain:

- `er-opd-q36-35b-slots-dispatch`
- `er-opd-q36-35b-slots-teacher-smg`

The science stack is separate and was not restamped by this throughput work.

## Promotion Rules

Do not promote a speed result to science/defaults unless all of these are true:

1. It is strict fresh-sample if science requires strict fresh semantics.
2. It has a same-workload throughput artifact at the intended node count.
3. It has a static K3 gate joined to the speed artifact.
4. The K3 gate has full intended coverage and passes:
   - mean K3 `<= 1e-3`;
   - p95 K3 `<= 1e-2`.
5. The output says which knobs are used, not just "see AMDAHL-NN".
6. For the larger-batch packer path, the run reports
   `dispatcher_dummy_batches`, oversized-sample policy/outcome, dropped-sample
   count, real tokens per rank, and the exact packer strategy.

The current fastest raw rowbatch path fails rule 4.

## Artifact Index

Current audited 4-node MFU:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_dtypefix_4node_20260614.json`

Non-promotable 1-node/2-node real-cache pack sweep:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_realcache_pack_sweep_20260615.json`

AMDAHL-077 rank-local rowbatch speed artifact:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-77-ranklocal-stability12-chunk4-serveronly-20260615T091946Z.jsonl`

AMDAHL-081 rowbatch provenance artifact:

`/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/replay-vpkl-lmheadtp-nocp-realalllayer-sglangcache-full64-2node-forwardprefetch-rowbatch2-81-provenance-real47-chunk4-serveronly-20260615T163845Z.jsonl`

AMDAHL-077 failed full static K3 gate:

`/shared/opd-control/er-opd-q36-35b-slots/k3/amdahl077_quack_deepep_sms36_20260615T1028Z/k3_gated_summary.json`

Dense layer0-10 transition drift audit:

`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers0_10_20260615T1607Z/q36_layer_transition_drift_layers0_10_row2069_20260615T1607Z.json`

Dense layer10-20 transition drift audit:

`/shared/opd-control/er-opd-q36-35b-slots/k3/xorl_fullcomponents_row2_layers10_20_20260615T1300Z/q36_layer_transition_drift_layers10_20_row2069_20260615T1302Z.json`

Shared science channel:

`/shared/apanda/opd_throughput_science_channel.md`

Larger-batch packer runbook:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/LARGER_BATCH_PACKER_RUNBOOK.md`

Chronological archive:

`/home/apanda/xorl-opd-prefill/experiments/opd_profile/autoresearch/THROUGHPUT_MICROBENCH_HISTORY_20260615.md`
