# OPD Config A/B Runbook and Handoff

Date: 2026-06-02

This document is the future-agent runbook for the Qwen3-235B OPD encoded-reasoning experiment after both Config A and Config B were run. It supersedes the live-run portions of `ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`, but that master runbook remains the source of truth for the original setup, OPD mechanics, prompt construction, and earlier negative controls.

## 0. TL;DR

Both Config A and Config B optimize the OPD training objective, but neither produces a stable encoded-reasoning improvement.

The important new result is that `eval/acc_pause` goes down over time in both configurations.

- Config A is the clean buffer-only test: answer masked, hidden/KL supervision on the NATO buffer. Hidden loss, KL, and total loss fall quickly, but `acc_pause` drops from 0.396 at step 0 to 0.333 at step 30. `acc_nopause` remains pinned at 0.010, so the buffer delta stays positive mostly because the no-pause baseline is dead.
- Config B is the answer-unmasked fallback: the answer path is supervised too. It proves direct answer learning is possible because `acc_nopause` jumps from 0.104 to 0.365 by step 10. But over the full 400-step run, the pause advantage collapses and the final controlled eval is `acc_pause=0.302`, `acc_nopause=0.313`, `buffer_delta=-0.010`.
- Combined verdict: do not rerun Config A or Config B as-is. The objective is learnable, but the learned behavior is not the desired pause-buffer reasoning channel. The strongest observation is not just "no improvement"; it is that optimizing this OPD target appears to degrade pause accuracy.

## 1. Source Documents

- `experiments/opd_profile/ENCODED_REASONING_MASTER_RUNBOOK_2026_06_02.md`
  - Section 3: OPD mechanics, prompt/teacher/student construction, Config A/B definitions.
  - Section 5.4: bring-up hazards.
  - Section 6: restart procedure.
- `experiments/opd_profile/HANDOFF_OPD_ENCODED_REASONING.md`
  - Older handoff. Useful for background, but parts are stale.
  - In particular, the warning that multi-token filler silently breaks K is outdated for the current code path; current OPD uses `k_filler=len(prefill_tokens)` and aligns the NATO buffer correctly.

## 2. Final Artifact Locations

Experiment family:

```bash
BASE=/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-235b-clean4d
```

Config A:

```bash
RUN_A=$BASE/20260602T055219Z-er-opd-235b-clean4d-trainer-head-zcmzv
PROFILE_A=$RUN_A/opd_profile.jsonl
WANDB_A=jzc4g717
```

Config B:

```bash
RUN_B=$BASE/20260602T065545Z-configb-er-opd-235b-clean4d-trainer-head-96lwh
PROFILE_B=$RUN_B/opd_profile.jsonl
WANDB_B=b8zejpa3
```

Manifest used for the 235B run:

```bash
experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

The manifest was switched from Config A to Config B by changing:

```yaml
opd_supervise_buffer_only: false
save_name_prefix: opd-235b-clean4d-nato-configb
wandb_run_name: opd-235b-clean4d-nato-0shot-hm-configb
```

It retained:

```bash
XORL_TORCHRUN_RDZV_CONF=timeout=1800
save_every=0
opd_kl_backend=streaming
```

## 3. Config Definitions

Both configs used the viable 235B setup from the master runbook:

- Model: Qwen3-235B-A22B base.
- Task: clean 4-digit multiplication.
- Prompting: 0-shot.
- Buffer: NATO filler repeated 3 times, about 102 tokens.
- Packing/sequence length: full CoT path with packing length 1024.
- Hidden match coefficient: `opd_hidden_match_coef=1.0`.
- Teachers and dispatch: default pool.
- Trainer and Mooncake samplers: `node-group=nccl`.

Config A:

```yaml
opd_supervise_buffer_only: true
```

Meaning: mask answer tokens, supervise only the buffer hidden/KL path. This is the cleanest test of whether teacher CoT state can be distilled into the buffer.

Config B:

```yaml
opd_supervise_buffer_only: false
```

Meaning: unmask answer tokens. This is the fallback that allows direct answer supervision in addition to buffer supervision.

## 4. Results

### 4.1 Config A: Buffer-Only

`opd_profile.jsonl` has 31 rows, steps 0 through 30.

Controlled eval rows:

```text
step  acc_pause  acc_nopause  buffer_delta  hidden_loss  kl_loss   loss
0     0.395833   0.010417     0.385417      0.147964     0.319312  0.467275
5     0.375000   0.010417     0.364583      0.132743     0.280177  0.412920
10    0.354167   0.010417     0.343750      0.105457     0.130634  0.236091
15    0.343750   0.010417     0.333333      0.086055     0.116803  0.202858
20    0.322917   0.010417     0.312500      0.074220     0.106577  0.180797
25    0.354167   0.010417     0.343750      0.066588     0.105162  0.171750
30    0.333333   0.010417     0.322917      0.060300     0.099516  0.159816
```

Interpretation:

- The objective is being optimized: hidden loss, KL, and total loss fall strongly.
- Pause accuracy deteriorates from 0.396 to 0.333 over 30 steps.
- No-pause accuracy remains pinned at about 1/96.
- The positive `buffer_delta` is not evidence of improved encoded reasoning; it mostly reflects that no-pause remains dead while pause gets worse.

Config A is negative.

### 4.2 Config B: Answer-Unmasked Fallback

`opd_profile.jsonl` has 400 rows, steps 0 through 399. The trainer head completed cleanly.

Selected controlled eval rows:

```text
step  acc_pause  acc_nopause  buffer_delta  hidden_loss  kl_loss   loss
0     0.458333   0.104167     0.354167      0.139197     0.285339  0.424616
5     0.479167   0.291667     0.187500      0.120272     0.182158  0.303331
10    0.468750   0.364583     0.104167      0.101936     0.128065  0.229966
15    0.447917   0.385417     0.062500      0.084370     0.111642  0.196073
20    0.447917   0.385417     0.062500      0.074292     0.097694  0.172989
25    0.333333   0.270833     0.062500      0.066902     0.097709  0.164848
50    0.354167   0.281250     0.072917      0.058570     0.097771  0.156692
100   0.395833   0.385417     0.010417      0.050861     0.105290  0.156489
150   0.364583   0.322917     0.041667      0.047898     0.094566  0.142950
200   0.260417   0.187500     0.072917      0.044760     0.090619  0.135301
250   0.333333   0.343750    -0.010417      0.044112     0.098121  0.141942
300   0.281250   0.270833     0.010417      0.045112     0.099894  0.144077
350   0.291667   0.291667     0.000000      0.042054     0.081940  0.124075
375   0.343750   0.270833     0.072917      0.044752     0.079852  0.124498
380   0.375000   0.322917     0.052083      0.044326     0.079438  0.124132
385   0.385417   0.354167     0.031250      0.042140     0.085337  0.127461
390   0.354167   0.322917     0.031250      0.043429     0.076300  0.119737
395   0.302083   0.312500    -0.010417      0.042797     0.106779  0.149587
```

Final profile tail:

```text
388 hidden=0.043054 KL=0.082356 loss=0.125566
389 hidden=0.042049 KL=0.073804 loss=0.115958
390 acc_pause=0.354167 acc_nopause=0.322917 delta=0.031250 hidden=0.043429 KL=0.076300 loss=0.119737
391 hidden=0.041627 KL=0.071867 loss=0.113440
392 hidden=0.042627 KL=0.077904 loss=0.120480
393 hidden=0.041830 KL=0.080724 loss=0.123602
394 hidden=0.043798 KL=0.080174 loss=0.124017
395 acc_pause=0.302083 acc_nopause=0.312500 delta=-0.010417 hidden=0.042797 KL=0.106779 loss=0.149587
396 hidden=0.043201 KL=0.100874 loss=0.144006
397 hidden=0.044495 KL=0.090304 loss=0.134811
398 hidden=0.043318 KL=0.077833 loss=0.120783
399 hidden=0.040818 KL=0.069142 loss=0.110081
```

W&B final summary:

```text
eval/acc_pause=0.30208
eval/acc_nopause=0.31250
eval/buffer_delta=-0.01042
eval/accuracy=0.65625
```

Interpretation:

- Direct answer supervision works initially: no-pause accuracy rises sharply by step 10.
- The pause advantage collapses quickly.
- Over the full run, both pause and no-pause accuracy drift down or oscillate below early values.
- The final pause accuracy is worse than the starting pause accuracy, despite lower hidden/KL/loss.

Config B is negative.

## 5. Scientific Interpretation

The key finding is the mismatch between objective optimization and answer behavior.

The training losses say the student is becoming closer to the teacher targets under the OPD objective. The evals say that this does not translate into robust arithmetic accuracy, and the pause condition becomes worse over time.

Working hypotheses:

1. Teacher-target mismatch / representation poisoning.
   - The teacher hidden targets are conditioned on a full CoT before the NATO buffer.
   - The student sees only prompt plus NATO buffer.
   - Matching the teacher's post-CoT state may push the student into an unnatural representation that is close by hidden-space MSE/KL but not causally useful for answer generation.

2. The buffer target is not a causal scratchpad.
   - Native filler lift may be a runtime/prompting property of the pretrained model.
   - Weight updates that force hidden-state imitation may destroy the pretrained behavior that made the pause useful.

3. Answer-unmasked training learns direct-answer behavior, not encoded reasoning.
   - Config B's early no-pause jump shows answer supervision can train the answer path.
   - But this erases the pause/no-pause gap and does not preserve the original pause advantage.

4. Formatting and continuation artifacts may be contaminating the training signal.
   - Late samples observed during Config B included repeated `Answer:` segments, answer text glued near NATO filler, and other continuation artifacts.
   - The model may be learning to continue the packed training format rather than learning arithmetic through a buffer.

5. Eval noise exists but does not explain the full pattern.
   - Controlled eval uses `n=96`, so individual points have several percentage points of noise.
   - The Config B trend over 400 steps and the Config A monotonic early loss/accuracy mismatch are large enough to treat as real.

## 6. Do Not Repeat

Do not relaunch Config A as-is.

Do not relaunch Config B as-is.

Do not spend another long 400-step 235B run on the same settings without a diagnostic change that specifically targets the observed failure mode.

Do not interpret positive `buffer_delta` in Config A as success unless `acc_pause` itself improves against its own step-0 baseline. In Config A, `buffer_delta` is positive because `acc_nopause` is dead.

Do not trust objective loss alone. For this experiment, `opd_hidden_match_loss`, `opd_kl`, and `loss` can all improve while the behavior of interest deteriorates.

## 7. Recommended Next Work

Start with artifact analysis before launching another large run.

### 7.1 Analyze Existing Profiles

Plot these metrics for both Config A and Config B:

```bash
jq -r 'select(.["eval/acc_pause"] != null) |
  [.step,
   .["eval/acc_pause"],
   .["eval/acc_nopause"],
   .["eval/buffer_delta"],
   .opd_hidden_match_loss,
   .opd_kl,
   .loss,
   .["eval/lead_pause"],
   .["eval/lead_nopause"],
   .["eval/has_think_close_frac"]] | @tsv' "$PROFILE_B"
```

Focus on:

- `acc_pause` vs step.
- `acc_nopause` vs step.
- `buffer_delta` vs step.
- Hidden/KL/loss vs step.
- `has_think_close_frac`, `lead_pause`, and `lead_nopause` if populated.

### 7.2 Inspect Samples

Inspect early, middle, and late generated samples from W&B or local artifacts if available. Classify failures by:

- Wrong arithmetic with otherwise clean format.
- Repeated `Answer:`.
- NATO leakage into answer region.
- Missing or malformed final answer.
- CoT continuation after answer.
- Prompt or packing continuation artifacts.

This should be done before designing the next config because the failure class determines whether to attack formatting, objective design, LR, or eval.

### 7.3 Next Config Candidates

Use short diagnostic runs first: 30 to 60 steps, controlled eval every 5 steps. Only extend to hundreds of steps if `acc_pause` improves against step 0 and the pause/no-pause gap remains meaningful.

Candidate C: KL-only, answer masked.

```yaml
opd_supervise_buffer_only: true
opd_hidden_match_coef: 0.0
```

Purpose: isolate whether hidden-state MSE is causing representation poisoning. If pause degradation disappears, hidden-match is the likely problem. If degradation remains, the issue is broader than hidden MSE.

Candidate D: hidden-only or reduced KL diagnostic.

Purpose: isolate whether KL on the buffer logits is responsible for drift. This is lower priority than KL-only because Config A already included both hidden and KL.

Candidate E: answer-only supervised control.

Purpose: quantify how much of Config B is ordinary direct answer learning and whether pause degradation occurs without the OPD buffer losses.

Candidate F: teacher without CoT.

Teacher path should be prompt plus NATO plus answer, not prompt plus CoT plus NATO plus answer.

Purpose: test whether the post-CoT teacher state is fundamentally unreachable from the student context.

Candidate G: lower LR Config B.

Purpose: test whether the long-run decline is generic optimizer damage. Use only after sample inspection, because Config B already failed the pause-buffer objective.

Candidate H: larger control eval.

Increase controlled eval N from 96 to about 400 for short runs if throughput permits. This reduces uncertainty when deciding whether a 5 to 10 point change is real.

## 8. Operational State

Config B completed, but at last inspection several pods were still running. Preserve artifacts before cleanup if needed.

Last known pod state:

```text
er-opd-235b-clean4d-dispatch              1/1 Running
er-opd-235b-clean4d-sglang-0              1/1 Running
er-opd-235b-clean4d-sglang-1              1/1 Running
er-opd-235b-clean4d-teacher-sglang-0      1/1 Running
er-opd-235b-clean4d-teacher-sglang-1      1/1 Running
er-opd-235b-clean4d-teacher-smg           1/1 Running
er-opd-235b-clean4d-trainer-head-96lwh    0/1 Completed
er-opd-235b-clean4d-trainer-worker-1..7   1/1 Running
```

If the next agent is not immediately relaunching on the same teachers, free everything:

```bash
kubectl delete job -n apanda er-opd-235b-clean4d-trainer-head --wait=true --timeout=300s || true
kubectl delete pod -n apanda \
  er-opd-235b-clean4d-trainer-worker-1 \
  er-opd-235b-clean4d-trainer-worker-2 \
  er-opd-235b-clean4d-trainer-worker-3 \
  er-opd-235b-clean4d-trainer-worker-4 \
  er-opd-235b-clean4d-trainer-worker-5 \
  er-opd-235b-clean4d-trainer-worker-6 \
  er-opd-235b-clean4d-trainer-worker-7 \
  er-opd-235b-clean4d-sglang-0 \
  er-opd-235b-clean4d-sglang-1 \
  er-opd-235b-clean4d-dispatch \
  er-opd-235b-clean4d-teacher-sglang-0 \
  er-opd-235b-clean4d-teacher-sglang-1 \
  er-opd-235b-clean4d-teacher-smg \
  --wait=true --timeout=300s || true
```

If immediately relaunching with the same teacher setup, keep teachers warm and delete only trainer, workers, samplers, and dispatch:

```bash
kubectl delete job -n apanda er-opd-235b-clean4d-trainer-head --wait=true --timeout=300s || true
kubectl delete pod -n apanda \
  er-opd-235b-clean4d-trainer-worker-1 \
  er-opd-235b-clean4d-trainer-worker-2 \
  er-opd-235b-clean4d-trainer-worker-3 \
  er-opd-235b-clean4d-trainer-worker-4 \
  er-opd-235b-clean4d-trainer-worker-5 \
  er-opd-235b-clean4d-trainer-worker-6 \
  er-opd-235b-clean4d-trainer-worker-7 \
  er-opd-235b-clean4d-sglang-0 \
  er-opd-235b-clean4d-sglang-1 \
  er-opd-235b-clean4d-dispatch \
  --wait=true --timeout=300s || true
```

Poll until old objects are gone:

```bash
kubectl get pods -n apanda | grep -E 'er-opd-235b-clean4d-(trainer|sglang|dispatch)' || true
```

## 9. Relaunch Procedure

1. Edit `experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml`.
2. Give the new run a unique config name in `RUN_ID`, `save_name_prefix`, and `wandb_run_name`.
3. Keep trainer and Mooncake samplers on the `nccl` pool.
4. Keep teachers and dispatch off known-bad default nodes.
5. Retain the longer rendezvous timeout unless using Volcano gang scheduling:

```bash
XORL_TORCHRUN_RDZV_CONF=timeout=1800
```

6. Dry-run server-side:

```bash
kubectl apply -n apanda --dry-run=server -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

7. Apply:

```bash
kubectl apply -n apanda -f experiments/opd_profile/k8s/generated/er-opd-235b-clean4d.yaml
```

8. Watch the pod bring-up:

```bash
kubectl get pods -n apanda | awk '/er-opd-235b-clean4d/ {print $1,$2,$3,$4}'
```

9. Confirm the run reaches step 0. The historical failure mode was that an 8-node trainer gang started partially, some workers scheduled late, the torch rendezvous hit the old 901s timeout, and the job cascaded. Do not judge a new science config until it reaches step 0.

Cluster note: all GPU pods must have `team: turbo` on the pod template labels. Do not manually override Volcano-injected scheduler fields unless there is a specific reason.

## 10. Monitoring Commands

Set the active run dir:

```bash
RUN_DIR=/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/er-opd-235b-clean4d/<run-id>
PROFILE=$RUN_DIR/opd_profile.jsonl
```

Recent profile rows:

```bash
jq -r '[.step,
        .["eval/acc_pause"],
        .["eval/acc_nopause"],
        .["eval/buffer_delta"],
        .opd_hidden_match_loss,
        .opd_kl,
        .loss,
        .step_total_s] | @tsv' "$PROFILE" | tail -20
```

Controlled eval only:

```bash
jq -r 'select(.["eval/acc_pause"] != null) |
  [.step,
   .["eval/acc_pause"],
   .["eval/acc_nopause"],
   .["eval/buffer_delta"],
   .["eval/accuracy"],
   .["eval/lead_pause"],
   .["eval/lead_nopause"],
   .["eval/has_think_close_frac"]] | @tsv' "$PROFILE"
```

Trainer log summary:

```bash
tail -120 "$RUN_DIR/trainer_job.log" | grep -E 'control step|=== OPD step|Traceback|ERROR|OPD run completed'
```

Health check from the trainer head, while running:

```bash
kubectl exec -n apanda <trainer-head-pod> -- curl -fsS http://127.0.0.1:26050/health
```

Pod state:

```bash
kubectl get pods -n apanda | awk '/er-opd-235b-clean4d/ {print $1,$2,$3,$4}'
```

## 11. Bring-Up Hazards

The main operational hazard is not the science; it is synchronized 8-node 235B bring-up.

Observed historical failure:

- 8-node trainer gang on a contended 10-node `nccl` pool.
- Some workers scheduled late.
- Torch rendezvous hit the 901s timeout.
- The run cascaded through about six failures before a clean synchronized bring-up.

Mitigations:

- Free the whole `nccl` pool before relaunch when possible.
- Delete the old trainer job with `--wait=true`.
- Poll until old worker/sampler/dispatch pods are actually gone before applying the manifest.
- Keep `XORL_TORCHRUN_RDZV_CONF=timeout=1800`.
- Consider Volcano gang scheduling if repeated partial-start failures return.
- Keep trainer and Mooncake samplers off the uncurated default pool.
- Avoid known-bad default nodes for teachers/dispatch if the master runbook still lists them as bad.

## 12. Decision Criteria For Future Runs

A future config is promising only if all of these are true:

- It reaches step 0 cleanly.
- `acc_pause` improves against its own step-0 baseline, not just against `acc_nopause`.
- `buffer_delta` remains positive because pause improves, not because no-pause collapses.
- Formatting/sample inspection does not show repeated `Answer:`, NATO leakage, or packed-continuation artifacts dominating generations.
- Hidden/KL/loss improvement correlates with behavior, rather than moving in the opposite direction.

Stop early if:

- `acc_pause` drops by more than about 5 points by step 20 to 30.
- `buffer_delta` collapses to zero while loss keeps improving.
- Samples show format corruption becoming common.
- The run has not reached step 0 because of rendezvous/scheduling failure; fix infra first and do not count it as a science result.

