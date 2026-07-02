# OPD Larger-Batch Packer Runbook

Current as of 2026-06-15 23:01Z.

This is the next throughput track for the OPD slots stack. It integrates the
larger-batch guidance from
`/home/apanda/xorl-apanda-dev/docs/notes/throughput_agent_larger_batch_brief.md`.
It is not a measured promotion yet.

## Status

The current audited 4-node logical MFU remains `0.010405075057463931`
(`~1.04%`). The larger-batch track is a new experiment plan, not a replacement
for that number.

The packer dependency exists upstream: PR #383,
`Add dp-aware / best-fit packing strategies + fail-loud oversized policy to server
packer`, is merged into `apanda-dev` at merge commit
`4f9f90cb2fcab48f0c15f14a975a178974e84e84`.

Do not assume the dirty local `/home/apanda/xorl-apanda-dev` checkout is usable
for this track. At the time this runbook was written it was on `pr373-review`
and did not expose `sample_packing_strategy` in the visible source. Use a fresh
engine worktree or sync a clean one, then verify the fields exist before
launching anything.

```bash
ENGINE=/home/apanda/xorl-opd-largebatch-packer
BASE=/home/apanda/xorl-apanda-dev

git -C "$BASE" fetch origin apanda-dev
git -C "$BASE" worktree add "$ENGINE" origin/apanda-dev

rg -n "sample_packing_strategy|sample_packing_on_oversized|packing_microbench|datum_order" \
  "$ENGINE/src/xorl/server" \
  "$ENGINE/experiments/local_benchmark"
```

## Why This Is Different

The rejected AMDAHL-041 and repeat-data ladder used the same 64-prompt payload
or repeated cached data. That did not change the core identity:

```text
real_tokens_per_rank = total_real_tokens_per_step / dp_size
```

At 64 OPD prompts, the trainer has only about `70k-107k` real student tokens
spread over 32 ranks, or roughly `2.2k` real tokens per rank. Packing can reduce
idle ranks and row fragmentation, but it cannot make that fixed token budget
large enough to hit the model's 16k-token-per-rank knee.

This track is different because it raises `--prompts-per-step` itself. The first
real target is `256-512` prompts per step on the 4-node, dp32 trainer, paired
with 16k packed rows and the PR #383 packer. The packer is the enabler for the
larger batch; `balanced_dp` or `best_fit` at the old 64-prompt batch is still a
known negative/diagnostic path.

## Required Server Config

Candidate YAML cannot currently override server packing fields. The slot
generator passes candidate `client_args` to the OPD client, while
`sample_packing_sequence_len` and `sample_packing_strategy` come from the trainer
config file under `/home/apanda/xorl-infra/configs/opd_profile/`.

Before adding a runnable AMDAHL candidate, create a trainer config copy from the
current 4-node no-CP lm-head-TP base:

```bash
INFRA=/home/apanda/xorl-infra
BASE_CFG="$INFRA/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp_nocp.yaml"
LB_CFG="$INFRA/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp_nocp_pack16k_balanceddp.yaml"

cp "$BASE_CFG" "$LB_CFG"
```

The copied config must explicitly contain:

```yaml
sample_packing_sequence_len: 16384
enable_packing: true
sample_packing_strategy: balanced_dp
sample_packing_on_oversized: error
```

Do not add `dp_size` manually; PR #383 auto-plumbs it from the server topology.
If `balanced_dp` is too aggressive in the fit smoke, switch only the strategy to
`best_fit` and keep `sample_packing_on_oversized: error`.

## Candidate Shape

Create the runnable candidate only after the server config exists. Start from
the AMDAHL-045/048 family, not from rowbatch diagnostics, and change only the
batch/packing track fields:

```yaml
id: AMDAHL-082-OPRD-PREP256-4NODE-LMHEADTP-NOCP-VPKL-PACK16K-BALANCEDDP
base_config: AM
trainer_config: configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp_nocp_pack16k_balanceddp.yaml
default_num_steps: 1
default_prompts_per_step: 256
profile_warmup_steps: 0
opd_prepare_batch_size: 256
opd_microbatch_size: 256
```

Keep these inherited OPD knobs from the current lm-head-TP/vocab-parallel path:

```yaml
client_args:
  opd_oprd_layers: every1
  opd_oprd_num_layers: 40
  opd_oprd_student_capture: selected_hooks
  opd_hidden_match_mode: mse
  opd_oprd_cache_backend: sglang
  opd_strict_prepare_overlap_chunks: 0
  opd_streaming_lowmem: true
  opd_vocab_chunk_size: 8192
  opd_mask_prompt_kl: true
```

Do not enable `opd_packed_row_batch_size` on this first track. The point is to
fill ranks with larger rows, not to reuse the K3-fragile rowbatch/coalescing
path.

## Experiment Ladder

Use this order. Do not skip directly to a long science run.

1. Offline sizing, no GPUs:

```bash
ENGINE=/home/apanda/xorl-opd-largebatch-packer
PY=/home/apanda/xorl-internal/.venv/bin/python
AUDIT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay/mfu_denominator_audit_realcache_pack_sweep_20260615.json
OUT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/fb_replay

cd "$ENGINE"
"$PY" experiments/local_benchmark/packing_microbench.py \
  --validate-audit "$AUDIT"
"$PY" experiments/local_benchmark/packing_microbench.py \
  --regime opd_4x --dp-size 32 \
  --output-json "$OUT/packing_microbench_opd_4x_$(date -u +%Y%m%dT%H%M%SZ).json"
"$PY" experiments/local_benchmark/packing_microbench.py \
  --regime large_batch --dp-size 32 \
  --output-json "$OUT/packing_microbench_large_batch_$(date -u +%Y%m%dT%H%M%SZ).json"
```

The built-in `opd_4x` and `large_batch` regimes are sizing checks. If a real
256/512-prompt capture exists, run `--capture <payload-or-audit> --dp-size 32`
and prefer that result.

2. Dry render before writing live control:

```bash
CLIENT=/home/apanda/xorl-opd-prefill
INFRA=/home/apanda/xorl-infra
ENGINE=/home/apanda/xorl-opd-largebatch-packer
PY=/home/apanda/xorl-internal/.venv/bin/python
CAND="$CLIENT/experiments/opd_profile/autoresearch/candidates/AMDAHL-082-OPRD-PREP256-4NODE-LMHEADTP-NOCP-VPKL-PACK16K-BALANCEDDP.yaml"

export OPD_XORL_CLIENT_REPO="$CLIENT"
export OPD_XORL_REPO="$ENGINE"
export OPD_XORL_INFRA_REPO="$INFRA"
export OPD_SGLANG_REPO=/home/apanda/xorl-sglang-internal

"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 \
  render-control \
  --candidate "$CAND" \
  --num-steps 1 \
  --prompts-per-step 256 \
  --sampler-replicas 2 \
  --sampler-layout dedicated \
  --role trainer-head \
  --output /tmp/opd_largebatch_trainer_head.yaml

rg -n "pack16k_balanceddp|prompts_per_step=256|opd_prepare_batch_size=256|opd_microbatch_size=256" \
  /tmp/opd_largebatch_trainer_head.yaml
rg -n "sample_packing_sequence_len|sample_packing_strategy|sample_packing_on_oversized" \
  "$INFRA/configs/opd_profile/qwen3_6_35b_a3b_opd_opdb_4node_warm009_deepep36_noprefetch_lmheadtp_nocp_pack16k_balanceddp.yaml"
```

3. Fit smoke, 1-2 steps:

```bash
"$PY" "$INFRA/k8s/opd_profile/q36_35b_reprogrammable_slots.py" \
  --model q36 --trainer-nodes 4 \
  write-trainer-control \
  --candidate "$CAND" \
  --num-steps 1 \
  --prompts-per-step 256 \
  --sampler-replicas 2 \
  --sampler-layout dedicated
```

Required fit-smoke observations:

- no oversized-sample error;
- `dispatcher_dummy_batches == 0` in the packing/dispatcher metrics;
- peak memory fits at roughly 16k packed tokens per row;
- no rowbatch/coalescing knob is active;
- no silent dropped sample count.

4. Throughput smoke:

Use 2-3 warmup steps and 5-10 measured steps. If B=256 fits and is stable, test
B=512 before changing topology. Report logical MFU, real tokens per rank,
`server_forward_backward_s`, prepare visibility, dummy count, dropped count, and
peak memory.

5. K3 correctness gate:

Any promoted `best_fit` or `balanced_dp` result must pass the static-trace K3
gate and be joined to the throughput artifact. Required thresholds are mean K3
`<= 1e-3`, p95 `<= 1e-2`, and full intended coverage. Use the
`xorl-k3-correctness-check` workflow and `experiments/local_benchmark/k3_gate.py`.

## Promotion Rules

Promote nothing to science/defaults unless all of these are true:

1. Same-workload throughput beats the current 64-prompt path at the intended
   node count.
2. The run uses real larger `--prompts-per-step`, not repeated cached data.
3. `dispatcher_dummy_batches == 0`.
4. `sample_packing_on_oversized: error` produced no oversized failures and no
   silent drops.
5. K3 passes with full coverage.
6. The science owner accepts the changed effective batch/on-policy freshness.

## Decision Boundaries

- If the 256/512 prompt fit smoke exposes prepare as the new wall, then scale
  sampler/teacher replicas. Do not over-provision generation before f/b exposes
  it.
- Keep per-row packing around 16k. Do not chase 24k-32k rows; prior evidence says
  that region is the OOM boundary.
- Do not use this track to relaunch AMDAHL-077 rowbatch. Rowbatch remains a
  separate raw speed diagnostic with failed K3.
- Coordinate any science-facing batch-size change through
  `/shared/apanda/opd_throughput_science_channel.md`, but do not restamp the
  live science stack for sizing or fit smokes.
