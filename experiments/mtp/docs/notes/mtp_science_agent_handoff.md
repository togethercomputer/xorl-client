# Handoff: MTP science agent

**Date:** 2026-06-13
**Direction:** science: make actual SingleShot MTP training work
**Stack:** `er-opd-q36-mtp-ss-0605c`
**XoRL repo:** `/home/apanda/xorl-mtp-singleshot-port-20260602`
**Live fix repo:** `/home/apanda/xorl-mtp-commitlen-fix-20260612`
**Reference repo:** `/home/apanda/singleshot`

## Mission

Your job is to explain why the current Qwen3.6 SingleShot-MTP run is not lifting `commit_len`, then propose and run
MTP-preserving fixes. Do not propose an EAGLE-style or autoregressive draft-model pivot as the answer. The target is the
paper's actual standalone SingleShot MTP training recipe, ported correctly and scaled to the current OPD deployment.

The current run is not evidence that SingleShot MTP is impossible. It is evidence that this port/config has not
bootstrapped deeper draft positions yet. Treat a flat `commit_len` curve as a debugging signal: find the mismatch with
the paper/reference implementation, the missing gradient signal, or the inadequate training schedule.

## Non-negotiables

- Do not change the science recommendation to EAGLE, speculative decoding, or a separate draft model.
- Do not declare the objective dead from a 50-80 step Qwen3.6 OPD continuation. The paper runs around 100k steps and
  reports stabilization around 50k steps.
- Keep the hard-teacher objective as the default unless you can show a reference-parity failure. The paper and original
  repo favor hard teacher argmax supervision over ground-truth labels, soft teacher, static-k-only training,
  bidirectional MTP attention, and prefix loss.
- Judge progress by the draft side: offset-1+ confidence/acceptance, ConfAdapt effective k, teacher-scored rollout
  quality, and per-position supervision coverage. Aggregate `top1_agreement`, aggregate `ent_stud`, and raw
  `commit_len` alone have been misleading.

## Reference facts from `~/singleshot`

Read these before changing code:

```text
/home/apanda/singleshot/icml_main.tex
/home/apanda/singleshot/litgpt/pretrain.py
/home/apanda/singleshot/litgpt/mtp.py
/home/apanda/singleshot/litgpt/model.py
/home/apanda/singleshot/litgpt/generate/base_mtp.py
/home/apanda/singleshot/litgpt/args.py
/home/apanda/singleshot/launch_exps_tuo_su.py
```

Key paper/source points:

- The student and teacher initialize from the same checkpoint; the teacher stays frozen and the student is fully
  trainable.
- The student proposes `k` tokens in one forward pass from prefix plus MTP mask slots. The teacher scores the student's
  proposed continuation under AR teacher forcing.
- Main objective is hard teacher argmax CE on the student-forced teacher targets. In the reference code this is
  `hard_teacher_supervision=True` and `pt_ce_plus_ent_loss(... labels_teacher=hard_teach_preds ...)`.
- Training uses causal blocked MTP attention, not bidirectional attention.
- The successful recipe randomizes both offsets and `k`, using `k in [2,16]`; `M=N/(2*k_max)` gives 5 MTP regions for
  MetaMath `N=160`.
- The paper's positive runs use much more optimization than the live OPD probe: roughly 100k iterations and about 500M
  supervised MTP tokens in expectation.
- ConfAdapt is the practical inference frontier. The original `generate/base_mtp.py` accepts the longest contiguous
  prefix whose top-1 confidences exceed the threshold, falling back to one token if the first draft token is below
  threshold.

Important distinction: the paper trains MTP slots directly under teacher supervision; ConfAdapt is described as an
inference/evaluation policy in the paper. The XoRL OPD path uses native ConfAdapt rollout and trace replay. That is a
reasonable deployment-shaped adaptation, but the science agent must prove it still supplies gradient to the draft slots
that need to learn.

## Current live run

The live run at the time of this handoff:

```text
run_id: q36mtp-20260613T014150Z-2s1t
run_dir: /shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t
trainer_log: /shared/opd-control/er-opd-q36-mtp-ss-0605c/trainer-head/logs/20260613T014150Z-run.log
rollout_samples: <run_dir>/artifacts/rollout_samples.jsonl
opd_profile: <run_dir>/artifacts/opd_profile.jsonl
```

Rendered live semantics:

```text
XORL_REPO=/home/apanda/xorl-mtp-commitlen-fix-20260612
branch=fix/mtp-replay-visibility-emit-supervision
commit=b0cfc6de
OPD_LOSS_MODE=hard_teacher_ce
lr=1e-5
muon_lr=1e-3
k_toks=4
mtp_strategy=["conf_adapt", 0.3]
sampling_mode=native
rollout_replay=true
validate_native_mtp_trace=true
static_padded_seq_len=2304
prompt_dataset_turn_strategy=suffix
prompt_len=512
prompts_per_step=64
optimizer=muon
moe_implementation=triton
ep_dispatch=deepep
deepep_num_sms=48
```

Latest quick read performed while writing this handoff:

```text
rollout records: 77
steady MTP steps: 13163
commit_len mean: 1.0349
commit_len dist: {1: 12729, 2: 409, 3: 24, 4: 1}
verify-token confidence median: 0.742

offset-1 acceptance: 308/9920 = 0.031
offset-1 confidence median: 0.293, mean 0.318
offset-2 acceptance: 13/7863 = 0.002
offset-3 acceptance: 0/6237 = 0.000
```

This says the live draft side is still weak. It does not say "pivot away from SingleShot MTP."

## Resolved launch direction for the next diagnostic

Apanda resolved the ordering: do **not** re-run the Qwen3-4B/MetaMath paper-style harness first. The next science
diagnostic should be a Qwen3.6 live-stack low-k/curriculum run from the current live run's own checkpoint.

At the time of the latest inspection, the live run had advanced beyond the original step-100 resume point, but no DCP
`.metadata` was present yet under the current run's `server_output/weights/default` directory. Before relaunching the
live diagnostic, coordinate with infra to wait for or create a checkpoint inside the current run directory, then resume
from that exact checkpoint. Do not fall back to the older
`q36mtp-20260612T204932Z-2s1t/.../q36mtp-coderforge-v1-step000100` checkpoint unless apanda explicitly authorizes a
restart-from-step-100 comparison.

## First hypothesis to audit: emit-window supervision

The original commit-only replay path cannot teach deeper drafts when `commit_len` is stuck near 1. The live fix branch
adds `supervise_emit_window=True` by default:

```text
/home/apanda/xorl-mtp-commitlen-fix-20260612/src/xorl/mtp/singleshot.py:2687
```

That code supervises each emit-window draft logit with the trajectory token it was trying to draft. Without it, only the
already-accepted commit slice receives loss, so the offset-1+ logits are mostly starved and `commit_len` has no reason to
increase.

Do this first:

1. Confirm the live run really used the fix branch and default `supervise_emit_window=True`.
2. Confirm `valid_tokens` is larger than committed/generated tokens by about the expected emit-window factor.
3. Spot-check a replay batch: labels, teacher hidden gather, `native_emit_positions`, and `_singleshot_mtp_source_indices`
   must align to the same trajectory token.
4. If alignment is wrong, fix this before any LR or long-run experiment.

Recent profile rows are consistent with emit-window supervision being active: step 176 has `valid_tokens=63327` while
`student_sampling_mtp_generated_tokens=13964`, so the trainer is supervising more than just committed tokens. That is a
good sign, not a proof of semantic alignment.

## High-probability gaps versus the paper recipe

Use this checklist before proposing a new objective:

- **Training budget:** the current run has millions of supervised tokens, not the paper's hundreds of millions. A 60-step
  LR continuation can reveal a wiring bug, but cannot falsify the method.
- **`k` schedule:** live Qwen3.6 is fixed `k=4`; the paper trains random `k in [2,16]`. A curriculum such as fixed `k=2`,
  then random `[2,4]`, then `[2,8]`, then `[2,16]` is MTP-preserving and closer to the paper than declaring failure.
- **Offset/randomization:** paper training randomizes offsets. Native OPD replay uses real generation traces, not the
  static `truncate_and_mask` data path; verify the replay gives equivalent coverage over prefix positions.
- **Teacher forcing semantics:** paper teacher is conditioned on the student's proposed continuation. Live OPD should
  prefill teacher hidden states over the generated student continuation and gather the teacher target at the same draft
  trajectory slot. Audit this explicitly.
- **Readout determinism:** paper training uses deterministic student argmax to materialize proposals. Live profile says
  `student_sampling_argmax_rollout=false` but also uses `top_k=1`; verify SGLang's native MTP route is effectively greedy
  for MTP proposals and not injecting sampling noise into the training target.
- **Mask token:** paper adds learned MTP special token(s), initialized from embedding statistics. Verify Qwen3.6's
  `mask_token_id=248063` is present, trainable, synchronized, and not frozen or treated as a degenerate/special stop.
- **Optimizer mismatch:** paper uses AdamW with 2000 warmup and constant peak LR around `1e-5`. Live Qwen3.6 uses Muon
  with a special `muon_lr`; this may be fine for scale, but it is not reference parity. If optimization looks suspect,
  propose an MTP-preserving AdamW or param-group diagnostic, not a new architecture.
- **Data mismatch:** paper's Qwen3-4B result is MetaMathQA, short `N=160`, BOS + `input + "\n\n" + response`, no Qwen chat
  template. Live Qwen3.6 is long Coderforge assistant-turn data with suffix prompts. Domain/data may slow bootstrap.

## Metrics to report

Every science update should include:

- run id, branch, commit, step range, and rendered `singleshot_mtp` config;
- `valid_tokens`, generated tokens, and the valid/generated ratio;
- per-offset draft acceptance and confidence for offset 1/2/3;
- ConfAdapt effective k distribution and confidence distribution;
- teacher rollout quality or teacher NLL if available;
- whether emit-window labels and teacher-hidden gather were spot-checked;
- training budget so far versus the paper's rough 500M supervised-token scale.

Do not promote a run only because loss decreases. Loss can drop while deeper draft slots stay useless.

## Commands

Set common paths:

```bash
cd /home/apanda/xorl-mtp-singleshot-port-20260602
PY=.venv/bin/python
RUN=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t
CTL=/shared/opd-control/er-opd-q36-mtp-ss-0605c
FIX=/home/apanda/xorl-mtp-commitlen-fix-20260612
REF=/home/apanda/singleshot
```

Find the current-run checkpoint to resume from:

```bash
find "$RUN/server_output/weights/default" -maxdepth 2 -type f -name .metadata -print | sort
```

If that prints nothing, do not silently resume from the older step-100 checkpoint. Ask infra to wait for or force a
current-run DCP checkpoint first.

Refresh the per-offset read:

```bash
$PY scripts/opd/analyze_mtp_draft_acceptance.py "$RUN/artifacts/rollout_samples.jsonl" --max-offset 3
```

Inspect current rendered config:

```bash
rg -n "OPD_LOSS_MODE|OPD_SINGLESHOT_MTP_JSON_DEFAULT|OPD_FULL_FT_LR|OPD_MUON_LR|XORL_REPO" \
  "$CTL/trainer-head/run.sh"
rg -n "lr:|muon_lr:|moe_implementation:|ep_dispatch:|deepep_num_sms:" \
  "$CTL/trainer-head/generated_trainer_config.yaml"
git -C "$FIX" rev-parse --abbrev-ref HEAD
git -C "$FIX" rev-parse --short HEAD
```

Check whether labels exceed committed rollout tokens:

```bash
$PY - <<'PY'
import json, pathlib
p = pathlib.Path("/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_singleshot/er-opd-q36-mtp-ss-0605c/q36mtp-20260613T014150Z-2s1t/artifacts/opd_profile.jsonl")
for line in list(p.open())[-10:]:
    r = json.loads(line)
    gen = r.get("student_sampling_mtp_generated_tokens") or r.get("rollout/token_count")
    valid = r.get("valid_tokens")
    print(r["step"], "valid", valid, "generated", gen, "valid/generated", (valid / gen if valid and gen else None),
          "commit", r.get("mtp/commit_len_steady"))
PY
```

Run the static geometry parity preflight against the reference implementation:

```bash
PYTHONPATH=src $PY scripts/opd/preflight_mtp_static_batch.py \
  --num-cases 128 \
  --batch-size 4 \
  --sequence-len 160 \
  --source-len 176 \
  --mask-region-count 5 \
  --k-min 2 \
  --k-max 16 \
  --offset-mode random-negative \
  --min-mask-token-id 248063 \
  --max-mask-token-id 248094 \
  --pad-to-multiple 128 \
  --output-json /tmp/mtp_static_parity_summary.json \
  --output-jsonl /tmp/mtp_static_parity_cases.jsonl
```

Preflight replay on live rollout samples:

```bash
PYTHONPATH=$FIX/src $PY scripts/opd/preflight_mtp_replay.py \
  --rollout-samples-jsonl "$RUN/artifacts/rollout_samples.jsonl" \
  --output-dir /tmp/mtp_replay_science_audit \
  --num-samples 4 \
  --k-toks 4 \
  --mask-token-id 248063 \
  --pad-to-multiple 128 \
  --flex-smoke-repeats 1
```

Run focused tests before and after code edits:

```bash
PYTHONPATH=src pytest tests/mtp/test_singleshot.py -q
PYTHONPATH=src pytest tests/distributed/test_qwen3_5_singleshot_cp.py -q
PYTHONPATH=src pytest tests/server/runner/test_opd_runner.py::test_singleshot_mtp_allows_linear_attention_after_mask_support -q
```

## MTP-preserving experiment menu

Pick one small diagnostic at a time. Coordinate launch mechanics with the infra agent. The immediate next live diagnostic
is Experiment A or A->B from the current-run checkpoint, not the Qwen3-4B static harness.

### Experiment A: reference-aligned low-k bootstrap

Run `k=2` with emit-window supervision and the same hard-teacher objective, resuming from the current Qwen3.6 live-run
checkpoint. If offset-1 confidence rises here but not at `k=4`, bootstrap with a curriculum instead of a fixed `k=4`
long run.

### Experiment B: random-k curriculum

Add or use a schedule that samples `k` from `[2,4]` first, then widens. This is closer to the paper than fixed `k=4`, and
it avoids asking cold offset-2/3 slots to learn before offset-1 works.

### Experiment C: paper-style static small-model reproduction fallback

Do not run this first. Keep the Qwen3-4B/MetaMath static harness as a fallback if the live low-k/curriculum diagnostic
and replay-alignment audit fail to explain the flat commit length, or if apanda explicitly asks for a small-model
reference reproduction. When used, its purpose is to verify that the current XoRL code still reproduces the
paper-positive behavior under short contexts and random offsets/k.

### Experiment D: optimizer reference check

If emit-window alignment is correct and low-k still does not move, run a bounded optimizer diagnostic: AdamW-style
reference param groups or a Muon parameter-group audit. The goal is to verify that MTP token embeddings, lm head, and
draft-relevant parameters receive meaningful updates.

## What not to do

- Do not rewrite `opd_loss.py` first. The active loss is already hard teacher CE.
- Do not switch to ground-truth suffix labels as the main route; the paper ablation says that is worse.
- Do not add prefix NTP loss as the main route; the paper ablation says it underperforms.
- Do not use bidirectional MTP attention as the main route; it is not the paper default and gives marginal/no gain there.
- Do not treat `commit_len` as the only success metric. Use ConfAdapt effective k and teacher-scored quality too.
- Do not stop at "more steps" without first auditing emit-window label/teacher alignment.

## Deliverable

Produce a concise verdict doc with:

- exact reference-parity findings against `~/singleshot`;
- live-run branch/config and latest per-offset metrics;
- emit-window supervision audit result;
- the highest-probability reason deeper drafts are not learning;
- the next MTP-preserving experiment, with exact config, stop criteria, and expected readout.

## Resolved decisions from apanda

- Prioritize the Qwen3.6 live-stack low-k/curriculum diagnostic first.
- Do not re-run the Qwen3-4B/MetaMath paper-style harness first.
- Resume the live diagnostic from the current live run's checkpoint. If no current-run DCP checkpoint is present yet,
  wait for or create one before relaunching.
- Do not restart from the older step-100 checkpoint unless apanda explicitly authorizes that comparison.
