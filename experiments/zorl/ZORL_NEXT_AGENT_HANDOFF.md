# ZORL - Next-Agent Handoff (2026-06-14 20:50 UTC)

Start in `/home/apanda/xorl-sglang-zorl`.

Active task: monitor the live HyperscaleES-derived ZORL run that was launched on
pool L using the previous best FRESH-RESAMPLE params. Do not restart pool L or
delete the client job unless the run has failed or the user explicitly asks.

Hard constraint: do not edit the shared `/home/apanda/xorl-sglang-internal`
checkout. The live SGLang pods use its venv as a dependency provider only; the
patched source is served from `/workspace/home/xorl-sglang-zorl/python`.

## Live Run

Kubernetes:

```bash
kubectl -n apanda get job zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct -o wide
kubectl -n apanda get pod zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct-749t6 -o wide
```

Current observed state at handoff time:

- Job: `zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct`
- Pod: `zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct-749t6`
- Pod node: `research-common-h100-005.cloud.together.ai`
- W&B name: `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L`
- Result root:
  `/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L`
- Active run dir:
  `/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L/20260614T204115Z-zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct-749t6-mult`
- Rendered manifest:
  `/home/apanda/xorl-client/experiments/zorl/autoresearch/renders/20260614T204114Z-multopsd-eggroll-35b-fresh-resample.yaml`

The first update completed successfully. Latest important log lines:

```text
cold: reward_mean=0.7422 exact_rate=0.7422 exact_count=95.0/128
step 1/100000: reward_mean=-0.3828 best_cand=-0.2961 update_norm=16.97 pair_delta_mean=-0.0359 pair_delta_std=0.9908 unclipped_update_norm=16.97 grad_norm=44671.02 update_clip_scale=1.0000 zero_score_pairs=0 dropped_pairs=0 score_normalization=none update_strategy=project_baseline_standardized parent_train_reward_mean=-0.4021 transformed_reward_mean=0.0013 transformed_reward_std=0.5330 used_pairs=128 t_score=156.6s t_apply=52.0s
```

This confirms the live server is returning the new trust-region metrics
(`unclipped_update_norm`, `update_clip_scale`) and the run is using
`project_baseline_standardized`.

## Monitor Commands

Follow the client:

```bash
kubectl -n apanda logs -f zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct-749t6
```

Tail the durable log:

```bash
RUN=/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L/20260614T204115Z-zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct-749t6-mult
tail -n 200 "$RUN/zorl_client.log" | rg "cold:|step [0-9]+/|probe|update_clip_scale|unclipped_update_norm|ERROR|Traceback|Exception"
tail -n 80 "$RUN/wandb_forwarder.log"
```

Check pool L health:

```bash
for i in 0 1 2 3 4 5 6 7; do
  code=$(kubectl -n apanda exec zorl-ar-sglang-l-$i -- sh -lc 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health' 2>/dev/null || true)
  printf '%s:%s ' "$i" "$code"
done
echo
```

Verify the serving pods are using the patched checkout:

```bash
kubectl -n apanda logs zorl-ar-sglang-l-0 --tail=80 | rg "sglang_repo=|/workspace/home/xorl-sglang-zorl|Server args"
kubectl -n apanda get sts zorl-ar-sglang-l -o yaml | rg "workingDir|SGLANG_REPO|SGLANG_PYTHON|PYTHONPATH|xorl-sglang-zorl" -C 2
```

Expected current pool L spec:

- `workingDir: /workspace/home/xorl-sglang-zorl`
- `SGLANG_REPO=/workspace/home/xorl-sglang-zorl`
- `SGLANG_PYTHON=/workspace/home/xorl-sglang-internal/.venv/bin/python`
- `PYTHONPATH=${SGLANG_REPO}/python...`

The live StatefulSet was patched in-cluster. The infra repo manifest has not
been made authoritative for this change; re-applying the old manifest may point
the pool back at `/workspace/home/xorl-sglang-internal`.

## Run Config

This is the prior best FRESH-RESAMPLE config with HyperscaleES-derived controls:

- Candidate: `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE`
- Rank/alpha: `LORA_RANK=16`, `LORA_ALPHA=16`
- Adapter: `/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll-hybridattn`
- Model: `Qwen/Qwen3.6-35B-A3B`
- Actual ES population: `num_pairs=128` via `NUM_SHARDS=auto`, `PAIRS_PER_SHARD=16`, 8 pool-L servers
- Train batch: `TRAIN_SIZE=256`, `TRAIN_POOL_SIZE=8064`, `RESAMPLE_TRAIN_EACH_STEP=1`
- Eval: `EVAL_SIZE=128`, `PROBE_INTERVAL=5`, `PROBE_TEMPERATURE=0.0`, `PROBE_N=1`
- Scoring: `SCORE_MODE=sft`, `TEACHER_FORCED_BATCH_SIZE=32`, `TEACHER_FORCED_LOGPROB_TRIM=1`
- ES scale: `B_SIGMA=0.00015`, `LEARNING_RATE=0.00038`
- HyperscaleES controls: `UPDATE_STRATEGY=project_baseline_standardized`,
  `MAX_UPDATE_NORM=17.12`, `LR_SCHEDULE=cosine`, `LR_MIN_FRAC=0.1`,
  `LR_DECAY_STEPS=120`
- Runtime cap: `MAX_RUNTIME_SECONDS=43200`

The shell startup line says `pairs=8` because that is the pre-sharding shell
variable, but the client log is authoritative: it initialized
`num_pairs=128`.

## What Was Changed

Edited in `/home/apanda/xorl-sglang-zorl`:

- `python/sglang/srt/lora/lora_manager.py`
- `test/registered/lora/test_zorl_fresh_ab.py`

Implemented HyperscaleES-derived ideas without copying GPL code:

- `fresh_ab` now supports positive `max_update_norm`.
- A dry-run fold pass computes the full update norm without mutating base
  weights.
- If the dry-run norm exceeds the cap, current and momentum fold coefficients
  are scaled before the real fold.
- Metrics now include `max_update_norm`, `update_clip_scale`,
  `unclipped_update_norm`, `unclipped_pre_momentum_update_norm`, and
  `fold_measure_passes`.
- Fold helpers accept `apply_update=False` so measurement uses the same kernels
  and tensor paths as the real update.
- Existing no-cap behavior remains single-pass.
- Tests now cover clipping behavior and invalid non-positive caps.

Already-existing client features are being used for the rest:

- `project_baseline_standardized` handles per-project standardization /
  baseline subtraction.
- `LR_SCHEDULE=cosine` handles annealing.
- Fresh-ab seed-space fold remains the update mechanism; no dense Adam state was
  added.

## Validation Done

Local checks:

```bash
git diff --check -- python/sglang/srt/lora/lora_manager.py test/registered/lora/test_zorl_fresh_ab.py
```

passed.

Per `DEBUG_ENV_WORKFLOW.md`, pycompile was run inside an interactive login shell:

```bash
cd /home/apanda/xorl-sglang-zorl
bash -li
if type sslm_sgl_env >/dev/null 2>&1; then sslm_sgl_env; else echo 'sslm_sgl_env not defined; using current interactive login shell'; fi
python -m py_compile python/sglang/srt/lora/lora_manager.py test/registered/lora/test_zorl_fresh_ab.py
exit
```

passed. `sslm_sgl_env` was not defined in this environment.

Narrow pytest was not runnable in the available local envs:

- `/home/apanda/xorl-internal/.venv/bin/python` failed collection due an
  `sgl_kernel` undefined symbol:
  `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`
- system `/usr/bin/python3.12` has compatible torch but lacks `pybase64`

Live validation is stronger for this change: step 1 of the active run completed
and returned the new clipping metrics from the patched SGLang servers.

## Relaunch Only If Needed

Do not relaunch while the active job is healthy. If the job fails and the user
wants a relaunch, reset the serving base first because ZORL merges mutate the
served base in place:

```bash
kubectl -n apanda delete job zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct
kubectl -n apanda delete pod -l app=zorl-ar-sglang-l
```

Wait until all eight `/health` checks return `200`, then launch:

```bash
python /home/apanda/xorl-client/experiments/zorl/autoresearch/controller.py launch \
  --candidate /home/apanda/xorl-client/experiments/zorl/autoresearch/candidates/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE.yaml \
  --set-env INFER_URL='http://zorl-ar-sglang-l-0.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-1.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-2.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-3.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-4.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-5.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-6.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000 http://zorl-ar-sglang-l-7.zorl-ar-sglang-l-headless.apanda.svc.cluster.local:30000' \
  --set-env RESULT_ROOT=/workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L \
  --set-env WANDB_NAME=MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L \
  --set-env MAX_RUNTIME_SECONDS=43200 \
  --set-env MAX_UPDATE_NORM=17.12 \
  --set-env UPDATE_STRATEGY=project_baseline_standardized \
  --set-env LR_SCHEDULE=cosine \
  --set-env LR_MIN_FRAC=0.1 \
  --set-env LR_DECAY_STEPS=120
```

## Prior Science Context

- Previous best overall run: `MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE`, rank 16,
  batch 256, resample-on, 128 pairs, `lr=0.00038`, `sigma=0.00015`, seed `9246`.
  Best probe was `0.9297` around step 280 / 12 h.
- Best adapter from that run:
  `/shared/apanda/zorl-consolidated-runs/best-adapter-fresh-resample-step288`
- Prior overnight r8-vs-r16 ceiling attempt is no longer the active task.
  M/F/I were manually stopped by request; L completed but only reached about
  `0.8984`, so it did not beat the previous `0.9297`.
- The old lesson still matters: early climb is not a converged ceiling, and the
  live HyperscaleES run should be judged by held-out probe trajectory, not step
  1 training reward.

## Suggested Next-Agent Prompt

```text
cd /home/apanda/xorl-sglang-zorl

Continue the ZORL HyperscaleES handoff in
/home/apanda/xorl-client/experiments/zorl/ZORL_NEXT_AGENT_HANDOFF.md.
Do not touch /home/apanda/xorl-sglang-internal. Monitor the live Kubernetes job
zorl-ar-multopsd-eggroll-35b-fresh-resample-w4jct and its run dir under
MULTOPSD-EGGROLL-35B-FRESH-RESAMPLE-HYPER-L. Confirm it keeps producing probe
metrics, watch update_clip_scale/unclipped_update_norm, and only relaunch after
resetting pool L if the job has failed or the user explicitly asks.
```
