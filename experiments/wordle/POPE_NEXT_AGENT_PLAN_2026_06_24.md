# POPE Wordle Next-Agent Plan - 2026-06-24

## Purpose

Take over the live POPE Wordle experiment without launching a new training run. The job is to verify the live state, wait for/check saved checkpoints, run held-out floor evals with failure taxonomy, and decide whether POPE actually internalized retrieval after the scaffold fades out.

## Scope Boundaries

- Do not launch another training job unless the user explicitly asks.
- Do not delete or restart the live POPE job unless it has failed and the user approves cleanup/relaunch.
- Do not edit the shared eval scripts in place.
- Do not touch the existing dirty repo edits unless they are directly needed for eval/handoff.
- Keep `sample_eval` off. Use the existing held-out floor-eval harness.
- Append important launch/eval/blocker updates to `/shared/apanda/wordle-coord/messages.jsonl`.

## Verified State

Last checked from `/home/apanda/xorl-client-wordle-science-20260614` at `2026-06-24T15:38:28Z`.

Repo:

- Worktree: `/home/apanda/xorl-client-wordle-science-20260614`
- Branch: `science/wordle-retrieval-sft-20260614`
- HEAD: `cb24f20`
- Existing dirty files were present before this doc was written:
  - `experiments/wordle/OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md`
  - `experiments/wordle/standalone/tasks/wordle.py`
  - `experiments/wordle/standalone/train_grpo_wordle.py`
  - `experiments/wordle/standalone/train_opsd_baseline.py`
  - untracked `experiments/wordle/standalone/grpo_rl_shim.py`
  - untracked `experiments/wordle/standalone/test_wordle_retrieval_reward.py`

Live POPE run:

- Kubernetes job: `grpo-pope-wordle-q36-9nltc`
- Kubernetes pod: `grpo-pope-wordle-q36-9nltc-mhtvj`
- Namespace: `apanda`
- Pod node at verification: `research-common-h100-044.cloud.together.ai`
- Pod creation timestamp: `2026-06-24T15:30:02Z`
- Run dir: `/shared/apanda/wordle-sft-runs/20260624T153014Z-grpo-pope-wordle-q36-9nltc-mhtvj-wordle-r16-asymmetric_opsd`
- W&B run id from `job.log`: `ubrg0x27`
- W&B run name: `GRPO-POPE-WORDLE-Q36-frombase-scaffold-fade12-muon5e5`

Important discrepancy: a prior note said the live job was on `h100-114`; the verified pod above was on `h100-044`. Re-check at takeover instead of assuming node placement.

Run configuration:

- Script: `/workspace/home/xorl-client-wordle-science-20260614/experiments/wordle/standalone/train_grpo_wordle.py`
- Trainer config: `/shared/apanda/wordle-sft-runs/configs/grpo-ep8-muon-lowlr.yaml`
- Build script: `/shared/apanda/wordle-sft-runs/build_grpo_pope.py`
- Manifest: `/shared/apanda/wordle-sft-runs/launch/grpo-pope-wordle-q36.yaml`
- Sampler endpoint: `http://wordle-sci-opsd-sampler-b-0.wordle-sci-opsd-sampler-b-headless.apanda.svc.cluster.local:30000`
- Model: `Qwen/Qwen3.6-35B-A3B`
- Reward: `wordle_retrieval_reward`
- Base prompt style: `public_reasoning_constraints_think`
- Scaffold prompt style: `public_reasoning_constraints_candidates_think`
- `wordle_scaffold_fade_steps`: `12`
- `save_interval`: `5`
- `steps`: `60`
- `train_size`: `8`
- `group_size`: `8`
- Rollouts per step: `64`
- Optimizer: `muon`, `optimizer_dtype=bf16`, CLI `lr=5e-6`
- Word list: `wordle-python`, `4266` legal guesses

Current run progress at verification:

- `job.log` showed server/sampler ready, step-0 full-weight sync completed, and:
  - `[POPE] step 1: scaffold_fraction=1.000 -> 64/64 rollouts use 'public_reasoning_constraints_candidates_think'`
- `metrics.jsonl` only contained `init` and `create_model`.
- `generations.jsonl` had step-1 rollout turns.
- No checkpoint directory existed yet under `server_output/weights`.

## POPE Mechanism To Validate

The code computes:

```text
scaffold_fraction = max(0.0, 1.0 - float(step - 1) / float(scaffold_fade_steps))
```

With `fade_steps=12`, this means:

- step 1: fraction `1.000`, all or nearly all rollouts scaffolded
- step 5: fraction `0.667`, mostly scaffolded
- step 10: fraction `0.250`, partly scaffolded
- step 12: fraction `0.083`, almost faded
- step 13+: fraction `0.000`, fully post-fade

Therefore step-5 and step-10 checkpoints are curriculum diagnostics. Step-15, step-20, and step-25 are the first real tests of whether retrieval internalized after the candidate list disappeared.

## Baselines And Gates

Existing held-out results to compare against:

- `grpo056-128`: `69/128 = 0.5391`
  - taxonomy: `max_turns=9`, `invalid_death=50`, `malformed=16`, `non_dict=38`, `repeat=1`, `violation=67`
- `ong-revg2-s8-128`: `75/128 = 0.5859`
  - taxonomy: `max_turns=13`, `invalid_death=40`, `malformed=16`, `non_dict=29`, `repeat=0`, `violation=55`
- `grpo-v2a-s25-128`: `63/128 = 0.4922`
- `grpo-strong-s25-128`: `6/128 = 0.0469`

Promotion/claim threshold:

- Strong single-slice win: `>=85/128` on seed-777 `[0:128]`.
- Replicated moderate win: `>=0.60` on two disjoint 128-game slices.
- Anything around `75/128` is interesting but not decisive, because it is close to the current OPSD-on-GRPO best and within small-n noise.

Failure taxonomy is required for any claim. A POPE checkpoint that improves exact only by leaning on malformed/invalid behavior is not a scientific win.

## First Actions For The Next Agent

Start here:

```bash
cd /home/apanda/xorl-client-wordle-science-20260614
date -u +%Y-%m-%dT%H:%M:%SZ
git status --short
kubectl get job grpo-pope-wordle-q36-9nltc -n apanda -o wide
kubectl get pod grpo-pope-wordle-q36-9nltc-mhtvj -n apanda -o wide
```

Then inspect logs/artifacts:

```bash
RUN=/shared/apanda/wordle-sft-runs/20260624T153014Z-grpo-pope-wordle-q36-9nltc-mhtvj-wordle-r16-asymmetric_opsd
tail -n 120 "$RUN/job.log"
tail -n 120 "$RUN/metrics.jsonl"
find "$RUN/server_output" -maxdepth 5 -type d -path '*weights*' -print
find "$RUN/server_output/weights/default" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort
```

If the job failed, do not relaunch automatically. Collect the failure first:

```bash
kubectl logs job/grpo-pope-wordle-q36-9nltc -n apanda --tail=300
tail -n 200 "$RUN/job.log"
tail -n 200 "$RUN/server.log"
tail -n 200 "$RUN/metrics.jsonl"
```

## Eval Commands

Use the generic eval harness for each saved checkpoint. The expected checkpoint path is:

```text
/shared/apanda/wordle-sft-runs/20260624T153014Z-grpo-pope-wordle-q36-9nltc-mhtvj-wordle-r16-asymmetric_opsd/server_output/weights/default/step-000005
```

Concrete first evals:

```bash
ROOT=/shared/apanda/wordle-sft-runs
RUN=/shared/apanda/wordle-sft-runs/20260624T153014Z-grpo-pope-wordle-q36-9nltc-mhtvj-wordle-r16-asymmetric_opsd
CFG=/shared/apanda/wordle-sft-runs/configs/grpo-ep8-muon-lowlr.yaml

bash "$ROOT/eval_ckpt_generic.sh" "$RUN/server_output/weights/default/step-000005" pope-s05-128 "$CFG" 128 2 | tee "$ROOT/logs/eval-pope-s05-128.log"
python "$ROOT/eval_failure_taxonomy.py" pope-s05-128
cat "$ROOT/evals/pope-s05-128/RESULT.json"
cat "$ROOT/evals/pope-s05-128/TAXONOMY.json"

bash "$ROOT/eval_ckpt_generic.sh" "$RUN/server_output/weights/default/step-000010" pope-s10-128 "$CFG" 128 2 | tee "$ROOT/logs/eval-pope-s10-128.log"
python "$ROOT/eval_failure_taxonomy.py" pope-s10-128
cat "$ROOT/evals/pope-s10-128/RESULT.json"
cat "$ROOT/evals/pope-s10-128/TAXONOMY.json"

bash "$ROOT/eval_ckpt_generic.sh" "$RUN/server_output/weights/default/step-000015" pope-s15-postfade-128 "$CFG" 128 2 | tee "$ROOT/logs/eval-pope-s15-postfade-128.log"
python "$ROOT/eval_failure_taxonomy.py" pope-s15-postfade-128
cat "$ROOT/evals/pope-s15-postfade-128/RESULT.json"
cat "$ROOT/evals/pope-s15-postfade-128/TAXONOMY.json"
```

For step-20 and step-25, repeat the same pattern with labels `pope-s20-postfade-128` and `pope-s25-postfade-128`.

If a post-fade checkpoint gets `>=0.60` but `<85/128`, run a second disjoint slice before calling it a win. Do not edit the shared script in place; copy it and change only the target offset:

```bash
cp "$ROOT/eval_ckpt_generic.sh" /tmp/eval_ckpt_generic_off128.sh
perl -0pi -e 's/--offset 0/--offset 128/' /tmp/eval_ckpt_generic_off128.sh
bash /tmp/eval_ckpt_generic_off128.sh "$RUN/server_output/weights/default/step-000015" pope-s15-postfade-128b "$CFG" 128 2 | tee "$ROOT/logs/eval-pope-s15-postfade-128b.log"
python "$ROOT/eval_failure_taxonomy.py" pope-s15-postfade-128b
```

## How To Interpret Outcomes

- Step-5 high, step-15 low: the model leaned on the scaffold and did not internalize retrieval.
- Step-5 high, step-15/20/25 also high: POPE likely worked; require taxonomy and replication before claiming success.
- Step-5 low: scaffolded rollouts did not produce useful policy updates, or the base model/scaffold interaction is dominated by format/invalid failures.
- Step-15 exact improves but taxonomy worsens: not a promotion; inspect malformed/non-dictionary/repeated/constraint-violation examples.
- Step-15 exact near `69/128` with better taxonomy: useful signal, but not a solve-rate win.

The most important failure categories are:

- `invalid_death`: game ended because the model emitted invalid/unparseable output.
- `games_with_non_dictionary`: legal format but not a Wordle dictionary word.
- `games_with_repeated_guess`: retrieval loop/repetition.
- `games_with_constraint_violation`: guessed a legal word inconsistent with public feedback.
- `max_turns_loss`: valid play but failed to solve; this is the real retrieval/planning residue once invalid behavior is controlled.

## Coordination Update Template

Append a compact JSON line after each evaluated checkpoint:

```bash
python - <<'PY'
import json, datetime, pathlib
label = "pope-s15-postfade-128"
root = pathlib.Path("/shared/apanda/wordle-sft-runs/evals") / label
result = json.load(open(root / "RESULT.json"))
taxonomy = json.load(open(root / "TAXONOMY.json"))
msg = {
    "ts": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "from": "science",
    "to": "science",
    "kind": "status",
    "event": "pope_checkpoint_eval",
    "label": label,
    "run_dir": "/shared/apanda/wordle-sft-runs/20260624T153014Z-grpo-pope-wordle-q36-9nltc-mhtvj-wordle-r16-asymmetric_opsd",
    "result": result,
    "taxonomy": taxonomy,
    "decision": "pending_or_promote_or_reject",
    "text": f"{label}: {result['solved']}/{result['games']} exact={result['exact']:.4f}; taxonomy={taxonomy}",
}
with open("/shared/apanda/wordle-coord/messages.jsonl", "a") as f:
    f.write(json.dumps(msg, sort_keys=True) + "\n")
PY
```

Update `decision` before appending.

## Final Deliverable Expected From The Next Agent

Report:

1. Live job status and whether it completed/failed.
2. Which checkpoints were evaluated.
3. `RESULT.json` values for each label.
4. `TAXONOMY.json` values for each label.
5. Whether post-fade POPE beat:
   - GRPO baseline `grpo056-128 = 69/128`
   - OPSD-on-GRPO best artifact `ong-revg2-s8-128 = 75/128`
6. Whether the result is a real retrieval improvement, a scaffold-only effect, or inconclusive.
7. Exact artifact paths for logs, evals, and any blocker.
