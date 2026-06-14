# Prefill-Time-Compute Autoresearch Runbook - 2026-06-04

This is the active handoff for the OPD / prefill-time-compute research loop.
It is intentionally not a complete historical log. Older runbooks remain the
archive; this file should answer: what is true now, what is missing, what is
unnecessary, what hypotheses we are testing, and how to run the loop without
active babysitting.

The exhaustive companion archive is
`experiments/opd_profile/PREFILL_TIME_COMPUTE_AUTORESEARCH_COMPREHENSIVE_ARCHIVE_2026_06_04.md`.
Use that file when you need the full historical ledgers, source runbooks,
candidate YAMLs, run logs, and scorecards in one place.

## 0. Current State

Primary objective:

- Train a model so `prompt + filler tokens + Answer:` gives better answer
  performance than `prompt + Answer:`.
- The filler does not need to be semantically meaningful. The immediate win is
  operational prefill performance, not a complete prompt-specific memory proof.
- Mechanism diagnostics matter, but should not veto a clean pause-vs-no-pause
  operational win.

Current queue head:

```bash
python experiments/opd_profile/autoresearch/controller.py next
```

selects `PTC-023`.

Last inspected service state in this revision:

- `dispatch`, `sglang-0`, `teacher-sglang-0`, and `teacher-sglang-1` are warm.
- `trainer-head` and `trainer-worker-1..7` are stopped after PTC-020.
- The current sampler layout is `spare-teacher1`: `sglang-0` plus
  `teacher-sglang-1` used as two student sampler endpoints.

The immediate research branch is:

1. `PTC-023`: AM minus hidden matching.
2. `PTC-024`: AM minus corrupt-negative training, hidden matching kept.
3. `PTC-025`: AM minus both hidden matching and corrupt-negative training.
4. `PTC-021`: composite-AM stability retest.
5. `PTC-022`: composite-AM larger-batch retest.

The ablation trio are six-step screens with a step-5 1k control. They are not
stability proofs. A positive ablation should get its own longer stability
follow-up before being treated as scale-ready.

## 1. Important Paths

```bash
REPO=/home/apanda/xorl-apanda-dev-opd-port
cd "$REPO"

STACK=er-opd-q36-35b-slots
NS=apanda
CONTROL_ROOT=/shared/opd-control/er-opd-q36-35b-slots
RESULT_ROOT=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots

GENERATOR=experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py
CONTROLLER=experiments/opd_profile/autoresearch/controller.py
IDEAS=experiments/opd_profile/autoresearch/ideas.yaml
CANDIDATES=experiments/opd_profile/autoresearch/candidates
SCORECARDS=experiments/opd_profile/autoresearch/scorecards
```

Model and data:

```bash
MODEL=/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PROMPTS_JSON=/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json
COT_JSON=/shared/opd-coord/randnum_4digit_8192_cot_nonempty.json
NUM_PROMPTS=8185
```

Primary files:

- `experiments/opd_profile/autoresearch/controller.py`
- `experiments/opd_profile/autoresearch/ideas.yaml`
- `experiments/opd_profile/autoresearch/candidates/PTC-*.yaml`
- `experiments/opd_profile/autoresearch/runs.jsonl`
- `experiments/opd_profile/autoresearch/scorecards/*.json`

## 2. Decision Standards

Operational metric:

```text
buffer_delta = acc_pause - acc_nopause
```

Operational claim:

- The pause/filler condition improves answer performance versus no pause.
- Promotion-quality evidence is approximately `buffer_delta >= 0.06` and
  `buffer_delta_z >= 3.0` at `n ~= 1000`.
- A retest-worthy result is approximately `buffer_delta >= 0.04` and
  `buffer_delta_z >= 2.0`, especially with positive answer-logprob support.

Mechanism claim:

- The pause/filler contains prompt-specific useful state.
- This needs boundary-identical corrupt/shuffle controls and positive
  correct-vs-distractor answer-selection behavior.
- AM does not yet prove this stronger claim.

Scoring modes:

- `pause_vs_nopause`: the operational gate. Corrupt-control artifacts and
  answer-selection do not veto a pause-vs-no-pause win.
- `causal_control`: the older stricter mechanism gate. Corrupt-control and
  answer-logprob criteria can veto promotion.

Use `pause_vs_nopause` for the current AM branch.

## 3. What AM Showed

AM was an answer-causal static-filler recipe, not a filler-surface search.

Student visible prefix:

```text
prompt + " ! | ~ _ * ^ # @ " + "Answer: "
```

Teacher context:

```text
prompt + teacher CoT + same visible pause/answer tail
```

Key AM settings:

- `opd_supervise_buffer_only=false`
- `opd_hidden_match_coef=2.0`
- `opd_contrastive_corrupt_buffer_weight=1.0`
- `opd_contrastive_corrupt_answer_weight=0.125`
- `opd_loss_max_clamp=5.0`
- `learning_rate=3e-6`
- `max_new_tokens=64`
- `eval_control_start_step=5`
- `eval_num_problems=1024`

AM only had a large operational control at step 5. Steps 0-4 only had small
ordinary eval rows, so the first reliable pause-vs-no-pause signal appeared at
the final scheduled control:

```text
acc_pause      = 0.6494
acc_nopause    = 0.5557
buffer_delta   = +0.0938
buffer_delta_z = 4.35
```

Post-hoc chunked answer-logprob scoring of the same final sampler weights was
also positive:

```text
answer_logprob_margin = +0.1784
answer_logprob_z      = 36.76
```

Why AM was not promoted at the time:

- The old gate was optimized for a prompt-specific memory claim, not the narrower
  operational claim.
- The legacy `rotate` corrupt control changed the assistant continuation
  boundary and produced cap/format artifacts.
- The in-loop answer-logprob scorer sent oversized batches and got HTTP 503s.
  Chunked post-hoc scoring fixed that.
- The answer-selection distractor probe was genuinely negative. AM raised
  absolute true-answer likelihood but did not improve the correct-vs-wrong
  likelihood margin.

Current interpretation:

- AM should be treated as the strongest operational prefill-performance hit.
- AM should not be cited as a complete prompt-specific memory proof.

## 4. Current Evidence Ledger

Decision-critical completed runs:

| Run | Result | Interpretation |
| --- | --- | --- |
| AH | `delta=+0.2708`, `z=4.06`, answer-logprob `+0.2234`, `z=5.33`, but `n=96` and corrupt cap-hit `0.9375` | First strong answer-causal signal; too small and corrupt-degenerate for promotion. |
| AI | `delta=+0.0938`, `z=1.91`, answer-logprob `+0.2163`, `z=16.67` at `n=192` | Exact-match underpowered but directionally strong. |
| AL | `delta=+0.1146`, `z=2.41`, answer-logprob `+0.2225`, `z=17.04`, corrupt cap-hit `0.5833` | Real positive, rejected by old corrupt-control health gate. |
| AM | `delta=+0.0938`, `z=4.35` at `n=1024`; post-hoc answer-logprob `+0.1784`, `z=36.76` | Strongest operational result. |
| AN | `delta=+0.0391`, `z=1.77`, answer-logprob `+0.1077`, `z=20.64`, answer-selection negative | Useful diagnostic, weaker than AM. |
| AO | `delta=-0.0322`, answer-logprob `-0.0069`; no corrupt-negative training, cache-mismatch objective | Clean rejection of that no-corrupt/cache-mismatch recipe. |
| PTC-020 | `acc_pause=0.6191`, `acc_nopause=0.5732`, `delta=+0.0459`, `z=2.12`, answer-logprob `+0.1110`, `z=23.76` | Current-stack AM retest is positive but weaker than AM; justifies ablations and stability follow-up. |

PTC-020 artifacts:

```text
/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots/20260604T191133Z-configPTC-020-er-opd-q36-35b-slots-trainer-head/opd_profile.jsonl
experiments/opd_profile/autoresearch/scorecards/20260604T194427Z-PTC-020-step5-promote_retest.json
```

PTC-020 also validated the current stack:

- FlashQLA candidate rendering.
- `sync_method=p2p`.
- Parallel endpoint sync: `serial_endpoint_sync=false`.
- Step-5 sync success to both endpoints.
- No sampled-control or answer-logprob request failures.

Runtime reference:

- AM took about 56 minutes wall-clock to finish through step 5 on the old serial
  endpoint sync path with three sampled control arms.
- PTC-020 took about 27 minutes through step 5 on the current parallel P2P path.
- PTC-023/024/025 skip corrupt exact-match eval, so they should be at least in
  the PTC-020 runtime class, but the actual runtime is missing until one finishes.

## 5. Active Hypotheses

### H1: Operational AM-Style Prefill Performance

Arbitrary static filler tokens can become a useful learned pre-answer state when
the training objective applies answer-level pressure after the pause.

Evidence for:

- AH, AI, AL, AM, and PTC-020 all point in this direction.
- AM is strong at `n=1024`.
- PTC-020 reproduced the direction on the current stack.

Evidence against or caveats:

- The effect was weaker in PTC-020 than in AM.
- AM-style positives are not yet proven stable past the first large control.
- Answer-selection remains negative.

Next evidence needed:

- Component ablations PTC-023/024/025.
- Stability retest PTC-021.
- Batch-size retest PTC-022 if stability holds.

### H2: Hidden Matching May Be Unnecessary

AM used `opd_hidden_match_coef=2.0`, but prior no-hidden results do not isolate
AM-minus-hidden. PTC-023 is the clean ablation.

Prediction:

- If hidden matching is unnecessary, PTC-023 should retain a positive step-5
  pause-vs-no-pause delta with answer-logprob support.

What would falsify it:

- PTC-023 cleanly rejects while PTC-020 remains positive and reproducible.

### H3: Corrupt-Negative Training May Be Unnecessary Or Misleading

AM used corrupt buffer and corrupt-answer penalties. The user does not care
about the corrupt arm as an end goal, and the corrupt control has known
boundary/cap artifacts.

PTC-024 tests a practical no-corrupt replacement:

- keep hidden matching;
- set corrupt weights to zero;
- use `opd_positive_answer_weight=0.125` so the answer path still receives
  explicit non-corrupt pressure.

PTC-025 tests the minimal no-hidden/no-corrupt version with the same positive
answer pressure.

Caveat:

- This is not a perfect mathematical deletion of the corrupt arm. If corrupt
  answer contrast is removed without any positive answer-side replacement, the
  run may no longer test the same answer-causal idea. PTC-024/025 are therefore
  "replace corrupt negative with positive answer pressure" ablations.

### H4: Six-Step Runs Are Screens, Not Stability Proofs

AM showed operational signs of life only at its first scheduled large control,
step 5. Therefore six-step candidates are useful for fast AM-like screening.

They are not sufficient to prove:

- persistence through step 10 or later;
- resistance to optimizer drift;
- scale-readiness;
- batch-size robustness.

If PTC-023/024/025 are positive, add or launch corresponding stability
follow-ups. For a true step-10 readout, do not let the autopilot stop the run
after a positive step-5 verdict.

### H5: Prompt-Specific Mechanism Remains Open

AM may improve formatting, confidence, calibration, or a generic
answer-producing mode. It may also encode prompt-specific state, but that is not
proven.

Required evidence:

- boundary-identical corrupt or shuffle controls;
- positive correct-vs-distractor answer-selection behavior;
- generated-memory or cache-mismatch controls that preserve the visible prompt.

Do not spend the next run on this unless the operational branch stabilizes.

### H6: Generated-Memory / RiM Needs Different Credit Assignment

Generated 100-token memory runs produced early positives, but supervised weight
sweeps did not stabilize them.

Evidence:

- PTC-011/012/014 had early pause-vs-no-pause positives.
- Those signals decayed by later controls, and answer-logprob was often
  negative.
- PTC-019 was the first sampled-token PG/K1 run with old-logprob plumbing, but
  it was infra-invalid.

Next useful generated-memory run:

- a clean sampled-token PG/RiM retest or a true task-reward objective;
- not another supervised answer-weight interpolation.

## 6. Current Queue

Queue source:

```bash
experiments/opd_profile/autoresearch/ideas.yaml
```

Current active order:

| ID | Priority | Candidate | Purpose |
| --- | ---: | --- | --- |
| PTC-023 | 92 | `candidates/PTC-023.yaml` | AM minus hidden matching. |
| PTC-024 | 91 | `candidates/PTC-024.yaml` | AM minus corrupt negatives, hidden kept. |
| PTC-025 | 90 | `candidates/PTC-025.yaml` | AM minus hidden and corrupt negatives. |
| PTC-021 | 89 | `candidates/PTC-021.yaml` | Composite-AM stability retest to step 10. |
| PTC-022 | 88 | `candidates/PTC-022.yaml` | Composite-AM batch-size scale-up. |

Candidate details:

| ID | Key changes | Control schedule | Interpret as |
| --- | --- | --- | --- |
| PTC-023 | `opd_hidden_match_coef=0.0`; corrupt buffer/answer contrast kept | step-5 1k | Does AM need hidden matching? |
| PTC-024 | hidden kept; corrupt weights zero; `opd_positive_answer_weight=0.125` | step-5 1k | Does AM need corrupt-negative training? |
| PTC-025 | hidden zero; corrupt weights zero; `opd_positive_answer_weight=0.125` | step-5 1k | Is a minimal positive-answer static-filler recipe enough? |
| PTC-021 | same as PTC-020; `default_num_steps=11` | step 5 and step 10 if not stopped early | Does composite AM persist? |
| PTC-022 | same as PTC-020 but `default_prompts_per_step=256` | step-5 1k | Does larger per-step batch reduce variance? |

All active AM-style candidates use:

- `base_config: AM`
- `gdn_backend: flashqla`
- `score_mode: pause_vs_nopause`
- `score_min_control_n: 1000`
- `eval_num_problems: 1024`
- `eval_answer_logprob_batch_size: 64`
- `eval_answer_logprob_max_concurrency: 2`
- `opd_pipeline_rl: false`
- `sampler_quiesce_before_sync: true`
- `sync_method: p2p`
- `serial_endpoint_sync: false`

PTC-023/024/025 additionally set:

```yaml
client_args:
  eval_corrupt_pause_control: false
```

That makes them cheaper operational screens. Corrupt exact-match is not part of
their scoring gate.

## 7. What Information Is Missing

Missing science results:

- PTC-023/024/025 outcomes.
- A true step-10 stability row for the composite AM recipe.
- Step-10 stability rows for any positive AM ablation.
- A batch-size retest result for composite AM or a winning ablation.
- A clean sampled-token PG/RiM generated-memory result.
- A prompt-specific mechanism proof with boundary-identical corrupt/shuffle and
  positive answer-selection behavior.

Missing operational data:

- Actual runtime of PTC-023/024/025 after corrupt free-generation eval is
  disabled.
- Whether the autopilot should gain a "terminal only after final control" option
  for multi-control stability runs. Right now, positive terminal verdicts can
  stop or return after the first step-5 control unless configured carefully.
- Whether PTC-024/025 positive-answer replacement is the best no-corrupt
  counterpart to AM. It is a pragmatic ablation, not a perfect objective
  identity.

Missing documentation after future runs:

- Scorecard paths for PTC-023/024/025.
- Exact control row metrics for every new completed run.
- Any failed launch/infra event from `runs.jsonl`.
- Any changed queue ordering after new evidence.

## 8. What Information Is Unnecessary Here

Keep out of this active runbook unless it changes a current decision:

- Full 235B per-step ledgers.
- Exhaustive A-X filler-surface tables.
- Raw W&B run IDs for old rejected runs.
- Long explanations of obsolete NCCL-broadcast or serialized endpoint paths,
  except as fallback notes.
- Detailed historical analyzer failures for runs whose conclusion is already
  captured.
- Repeated command variants for old configs.
- Filler-surface archaeology unless a new objective attaches to it.

Where to look if that detail is needed:

- `experiments/opd_profile/OPD_CONFIG_A_B_RUNBOOK_2026_06_02.md`
- `experiments/opd_profile/Q36_35B_OPD_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_02.md`
- `experiments/opd_profile/K8S_REPROGRAMMABLE_SLOTS_RUNBOOK_2026_06_03.md`
- `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RESEARCH_MEMO_2026_06_03.md`
- `experiments/opd_profile/PREFILL_TIME_COMPUTE_OPSD_RUNBOOK_2026_06_03.md`

## 9. Operating Procedure

### Check State

```bash
python "$GENERATOR" status
python "$CONTROLLER" next
```

If a run is launched, inspect the latest profile:

```bash
python "$CONTROLLER" monitor --profile latest --json
```

### Render A Candidate

```bash
python "$GENERATOR" render-control \
  --candidate "$CANDIDATES/PTC-023.yaml" \
  --num-steps 6 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-023-trainer-head.yaml
```

Check:

```bash
rg "XORL_GDN_BACKEND=flashqla|sync_method=p2p|eval_num_problems=1024|eval_control_start_step=5|opd_hidden_match_coef|opd_contrastive_corrupt|opd_positive_answer_weight" /tmp/ptc-023-trainer-head.yaml
```

### Launch The Next Candidate

Dry run:

```bash
python "$CONTROLLER" launch --dry-run
```

Actual launch:

```bash
python "$CONTROLLER" launch \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

Explicit launch:

```bash
python "$CONTROLLER" launch \
  --id PTC-023 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

### Autopilot For Fast Screens

Use this for PTC-023/024/025 if you want it to launch the next queued idea after
a clean science rejection and stop on a positive screen:

```bash
python "$CONTROLLER" autopilot \
  --launch-next \
  --poll-seconds 300 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --terminal-action stop_advance \
  --launch-grace-seconds 600
```

This is appropriate because PTC-023/024/025 are six-step screens. A positive
screen should stop for review and follow-up candidate creation.

### Autopilot Caution For Stability Runs

PTC-021 has `default_num_steps=11` and `eval_control_start_step=5`. That means
it can emit a positive verdict at step 5 before the step-10 stability row exists.

For a true step-10 stability run, do one of the following:

- monitor manually and do not run the autopilot with
  `--terminal-action stop_advance`; or
- launch a dedicated stability candidate with `eval_control_start_step: 10`; or
- update the controller to support "terminal only after final scheduled control"
  before relying on unattended multi-control stability.

Do not claim a stability result from PTC-021 unless the profile actually contains
the later control row.

### Score And Advance Manually

```bash
python "$CONTROLLER" score --profile latest --idea-id PTC-023 --json
python "$CONTROLLER" advance --id PTC-023 --profile latest
```

Scorecards are written under:

```bash
experiments/opd_profile/autoresearch/scorecards/
```

### Stop Wedged Trainer Roles

```bash
python "$GENERATOR" stop-trainer-control --remove-run
python "$GENERATOR" status
```

Normal science iteration should relaunch trainer roles only; do not restart
student inference unless endpoint health or routing is actually bad.

## 10. Infrastructure Invariants

- Every GPU pod must carry `team: turbo`.
- Use `--sampler-layout spare-teacher1` until a dedicated second student sampler
  is intentionally scheduled and healthy.
- Keep each SGLang sampler internally serialized with `--max-running-requests 1`.
  Batched SGLang decoding previously caused repeated-suffix correctness
  failures.
- Use `sync_method=p2p` with `serial_endpoint_sync=false` for current promoted
  paths.
- Keep serialized P2P and NCCL broadcast as fallbacks only.
- Keep `opd_pipeline_rl=false` and `sampler_quiesce_before_sync=true` until a
  pipelined path has its own freshness proof.
- Keep answer-logprob scoring chunked:
  `eval_answer_logprob_batch_size=64`,
  `eval_answer_logprob_max_concurrency=2`.
- Treat corrupt fixed-token exact-match controls as diagnostics, not gate
  vetoes, unless the corrupt boundary/cap health is known clean.

## 11. Candidate Authoring Rules

For AM-style operational candidates:

```yaml
base_config: AM
gdn_backend: flashqla
score_mode: pause_vs_nopause
score_min_control_n: 1000
student_prefill_text: " ! | ~ _ * ^ # @ "
student_prefill_count: 1
student_prefill_suffix: "Answer: "
student_stop_sequences: '["\n"]'
default_prompts_per_step: 128
eval_num_problems: 1024
eval_accuracy_every: 5
eval_control_start_step: 5
eval_answer_logprob_control: true
eval_answer_logprob_batch_size: 64
eval_answer_logprob_max_concurrency: 2
opd_pipeline_rl: false
sampler_quiesce_before_sync: true
sync_method: p2p
serial_endpoint_sync: false
request_timeout: 1200
weight_sync_timeout: 900
opd_loss_max_clamp: 5.0
```

For fast ablation screens, use:

```yaml
default_num_steps: 6
client_args:
  eval_corrupt_pause_control: false
```

For stability follow-ups, prefer a dedicated candidate whose control schedule
cannot be mistaken for a step-5-only screen. If using an 11-step run, verify the
actual profile contains the step-10 row before advancing the science conclusion.

For generated-memory candidates, only use lower eval sizes when generated-memory
sampling makes full 1k controls too expensive:

```yaml
eval_num_problems: 256
score_min_control_n: 250
client_args:
  student_generated_memory_tokens: 100
  eval_generated_memory_control: true
  eval_generated_memory_tokens: 100
```

Do not add another generated-memory supervised weight sweep unless there is a new
mechanistic reason.

## 12. Quick Verification Commands

Static checks after controller or generator edits:

```bash
python -m py_compile "$CONTROLLER" "$GENERATOR"
uv run ruff check "$CONTROLLER" "$GENERATOR"
```

Queue and render checks:

```bash
python "$CONTROLLER" next
python "$CONTROLLER" launch --id PTC-023 --dry-run

python "$GENERATOR" render-control \
  --candidate "$CANDIDATES/PTC-023.yaml" \
  --num-steps 6 \
  --prompts-per-step 128 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1 \
  --role trainer-head \
  --output /tmp/ptc-023-check.yaml

rg "config=PTC-023|base_config=AM|XORL_GDN_BACKEND=flashqla|sync_method=p2p|serial_endpoint_sync=false" /tmp/ptc-023-check.yaml
```

Profile summary snippet:

```bash
python - <<'PY'
import glob, json, os
root = "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots"
paths = glob.glob(root + "/*/opd_profile.jsonl")
path = max(paths, key=os.path.getmtime)
rows = [json.loads(line) for line in open(path) if line.strip()]
controls = [r for r in rows if "eval/acc_pause" in r]
print(path)
print("rows", len(rows), "last_step", rows[-1].get("step"))
if controls:
    r = controls[-1]
    for key in [
        "step",
        "eval/control_n",
        "eval/acc_pause",
        "eval/acc_nopause",
        "eval/buffer_delta",
        "eval/buffer_delta_z",
        "eval/answer_logprob_margin",
        "eval/answer_logprob_margin_z",
    ]:
        print(key, r.get(key))
PY
```

## 13. Current Next Action

Launch PTC-023 when ready:

```bash
python "$CONTROLLER" launch \
  --id PTC-023 \
  --sampler-replicas 2 \
  --sampler-layout spare-teacher1
```

Then either:

- run the fast-screen autopilot command above; or
- wait for the step-5 control row, score it, and advance manually.

Interpretation after PTC-023:

- Positive: hidden matching is likely not necessary for the fast AM-like
  operational effect. Add a no-hidden stability follow-up.
- Negative: hidden matching may be important, or PTC-020 may have been a weak
  transient. Continue to PTC-024/025 only if the question is component
  attribution, not immediate scale-up.
