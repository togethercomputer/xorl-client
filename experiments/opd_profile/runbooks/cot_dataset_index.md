# CoT precompute dataset index

Central pointer to all teacher-generated CoT JSONs on `/shared/opd-coord/`. Each
file is a JSON list of `{prompt, cot, finish_reason, tokens}` entries (same
schema as `cot_precompute.py` writes). All prompts in v1..v6 are pairwise
**disjoint** 4-digit × 4-digit multiplication pairs.

## Prompt sets

| File | Count | Seed | Notes |
|---|---:|---|---|
| `randnum_4digit_8192.json` | 8,192 | — | Original (v1) prompts, mtime 2026-05-25 |
| `randnum_4digit_8192_v2.json` | 8,192 | 20260528 | Disjoint from v1 |
| `randnum_4digit_8192_v3.json` | 8,192 | 20260530 | Disjoint from v1∪v2 |
| `randnum_4digit_8192_v4.json` | 8,192 | 20260601 | Disjoint from v1∪v2∪v3 |
| `randnum_4digit_8192_v5.json` | 8,192 | 20260603 | Disjoint from v1∪v2∪v3∪v4 |
| `randnum_4digit_81920_v6.json` | 81,920 | 20260605 | Disjoint from v1..v5 |
| `randnum_4digit_16384_combined.json` | 16,384 | — | v1∪v2 union, built by another agent |
| `randnum_4digit_32768_combined.json` | 32,768 | — | v1..v4 union, built by another agent |
| `randnum_4digit_122880_combined.json` | 122,880 | 20260606 | v1..v6 merged + shuffled |

Pool total: **122,880** unique disjoint `(a,b)` pairs verified.

## CoT datasets — Qwen3.6-35B-A3B teacher (BF16, TP=2)

All generated at `--max-tokens 8192` with the system message asking for
step-by-step reasoning. Mean ~2720 tokens/prompt, ~1.2% truncation. **335M
total teacher tokens.**

| File | Prompts source | Entries | Mean tokens | Trunc | Wall | Topology |
|---|---|---:|---:|---:|---|---|
| `randnum_4digit_8192_cot_mt8192.json` | v1 | 8,192 | 2728 | 1.35% | ~2.5h | dispatch, 12 sglang TP=2 |
| `randnum_4digit_8192_v2_cot_mt8192.json` | v2 | 8,192 | 2757 | 1.34% | ~1.5h | dispatch, 8 sglang TP=2 |
| `randnum_4digit_8192_v3_cot_mt8192.json` | v3 | 8,192 | 2736 | 1.10% | ~1.7h | dispatch, 4 sglang TP=2 |
| `randnum_4digit_8192_v4_cot_mt8192.json` | v4 | 8,192 | 2740 | 1.22% | ~1.4h | dispatch, 8 sglang TP=2 |
| `randnum_4digit_8192_v5_cot_mt8192.json` | v5 | 8,192 | 2718 | 1.07% | ~30min | **SMG**, 3 sglang TP=2 |
| `randnum_4digit_81920_v6_cot_mt8192.json` | v6 | 81,920 | 2724 | 1.17% | ~6.5h | SMG, 3 sglang TP=2 |
| `randnum_4digit_16384_combined_cot_mt8192.json` | 16384_combined | 16,384 | — | — | — | (merged by another agent) |
| `randnum_4digit_32768_combined_cot_mt8192.json` | 32768_combined | 32,768 | — | — | — | (merged by another agent) |
| `randnum_4digit_122880_combined_cot_mt8192.json` | 122880_combined | 122,880 | 2728 | 1.19% | (merge) | v1..v6 merged + shuffled |

**For Run B / OPD training**: point `prompts_json_path` + `teacher_cot_json_path` at the matching pair. The deterministic-shuffled 122880_combined files are the recommended training pool since the OPD client doesn't shuffle itself.

## CoT datasets — Qwen3-235B-A22B teacher (base, BF16, TP=8)

Started 2026-05-29. Largest filler-effect model per
`/old-data/apanda/tomi/outputs/multimodel_filler_and_cot_summary_20260528.md`
(strict-graded +11.2pp on 4dmult-0shot with nato filler; +4.6pp on 10shot).
Heavy reasoning leak on arith (>30% baseline discard rate) so this CoT
captures the model's natural reasoning.

Observed behavior: emits `<think>\n\n</think>` (empty think) followed by
Markdown step-by-step distributive-property breakdown. **Mean ~525 tok/CoT**,
much shorter than Q3.6-35B-A3B's ~2725 — base 235B reasoning is more
compact. Achieves ~100-150 tok/s/GPU at TP=8 with `power_of_two` policy.

**Initial cache_aware policy failed** — one shard got 16× more load than the
other (sglang-1: 479 running, sglang-0: 30 running). Restarted with
`power_of_two`. See updated `smg_router_swap.md` for the policy lesson.

| File | Prompts source | Entries | Mean tokens | Trunc | Status |
|---|---|---:|---:|---:|---|
| `randnum_4digit_16384_q3_235b_cot_mt8192.json` | 16384_combined | 16,384 | 518 | 0% | **done** (15979 valid + 405 errors). Max 1389 tok — base 235B never approaches 8192-cap |

## CoT datasets — Qwen3.5-35B-A3B teacher (BF16, TP=2)

Started 2026-05-29. The "+14.94pp clean pause filler" model per the summary
doc — Mechanism B (digit refinement) on 4dmult-10shot. **Pivoted to 5-digit
multiplication** to extend beyond the 4-digit pool.

### New prompt set: 5-digit multiplication

| File | Count | Seed | Notes |
|---|---:|---|---|
| `randnum_5digit_81920.json` | 81,920 | 20260607 | 5-digit × 5-digit (10000-99999) |

| File | Prompts source | Entries | Mean tokens | Trunc | Status |
|---|---|---:|---:|---:|---|
| `randnum_5digit_81920_q35_35b_cot_mt8192_partial.json` | randnum_5digit_81920 | 10924 valid / 14046 written | 7624 (median 8192) | **57.8%** hit 8192 cap | **superseded**. Truncation too high — most CoTs would need 9-15k tokens. |
| `randnum_5digit_81920_q35_35b_cot_mt16384.json` | randnum_5digit_81920 | 81920 total: **55,874 valid**, 26,046 errors | 7952 mean, 7803 median, max 12143 | **0%** (huge win vs mt=8192's 57.8%) | **done**, 1.1 GB. Errors are empty-cot transient failures; resume-relaunching would retry just those. |

**Q3.5-35B-A3B 5-digit truncation lesson**: at mt=8192, 57.8% of CoTs hit the cap with median = 8192. The model goes into deep step-by-step distributive-property decomposition that frequently needs 9000-14000 tokens. Bumped to mt=16384 for the canonical run — expected to drop truncation to <10%.

## Planned teachers (queue)

| Teacher | Topology | Active params | Per-GPU est | Notes |
|---|---|---|---|---|
| Qwen3.5-397B-A17B | TP=16 / 2 nodes | 17B | ~250 tok/s/GPU | Largest CLEAN filler effect (+20.1pp ellipsis on 4dmult-10) — Mechanism A (anti-EOS) |
| Qwen3.5-35B-A3B | TP=2 / 1 node | 3B | ~2200 tok/s/GPU | +14.94pp pause on 4dmult-10 — Mechanism B (digit refinement) |
| Qwen3-235B-A22B-Instruct-2507 | TP=8 / 1 node | 22B | ~300 tok/s/GPU | Instruct-tuned variant; cleaner CoT vs base |
| Qwen3-30B-A3B-Instruct-2507 | TP=2 / 1 node | 3B | ~2200 tok/s/GPU | CoT crushes filler (99.3% CoT vs base; cheap to add) |

## How CoTs are used in OPD/Run B

`xorl-client-internal/examples/on_policy_distillation.py` reads
`teacher_cot_json_path` and aligns entries 1:1 with the prompt list. Each
prompt's CoT is used as the teacher's *filler tokens* during distillation —
the teacher sees `[prompt + CoT + student_rollout]` while the student sees
`[prompt + " pause"*N + student_rollout]`. Teacher hidden states at the
assistant positions are the distillation target. See
`runbooks/smg_router_swap.md` for the routing-layer setup that makes these
runs efficient.

## How to add a new teacher

1. Find HuggingFace snapshot under `/shared/huggingface/hub/models--Qwen--<name>/snapshots/<hash>/`
2. Decide TP size: `2 * model_params_GB / 80` GB/GPU. Aim for ≤70 GB/GPU at BF16.
   - <30B params: TP=1
   - 30-50B (incl MoE): TP=2
   - 100-300B: TP=8 (single node)
   - 300B+: TP=16 (cross-node, slower)
3. Copy `cot-q3-235b-052901.yaml` as a starting point; update `--tp-size`,
   `--model-path`, `--served-model-name`, run name, output path.
4. Set sglang's `--mem-fraction-static`: 0.85 for <50B, 0.88 for 50-200B,
   0.90+ for 200B+ (memory-tight). Cap `--max-running-requests` at 128 for
   200B+ models to leave KV cache room.
5. SMG router + client pattern is identical to existing manifests; just
   update worker-urls and prompts/output paths.
6. Add the new CoT file to this index when complete.
