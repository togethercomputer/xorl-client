# Qwen3.6 OPD SingleShot MTP Smoke

This example runs the existing OPD driver with a Qwen3.6-35B-A3B teacher and
student, while the trainer replays native SGLang MTP rollouts under the
SingleShot MTP layout. The driver defaults to `conf_adapt` with threshold
`0.6`, native debug traces, and trainer-side trace validation.

The Qwen3.6 stack mixes full-attention and `linear_attention` layers. The smoke
therefore exercises both the eager 4D mask path and the linear-attention
SingleShot branch fallback.

## Trainer Config

Use `qwen3_6_35b_a3b_student_mtp.yaml` for the student trainer. It pins
`ulysses_parallel_size: 1` and `attn_implementation: eager`, because the MTP
batch path currently passes a dense 4D attention bias.

Start the trainer with the normal server launcher, for example:

```bash
python -m xorl.server.launcher \
  --mode auto \
  --config examples/server/opd_singleshot_mtp_qwen36/qwen3_6_35b_a3b_student_mtp.yaml \
  --api-port 6000
```

The teacher service should be a separate AR Qwen3.6 server that can write an
OPD hidden cache (`OPD_TEACHER_BACKEND=xorl`) or return hidden states
(`OPD_TEACHER_BACKEND=sglang`). The student sampler should be a SGLang build
with native MTP request knobs.

## Driver Smoke

After `student.json` and `teacher.json` are present in `OPD_COORD_DIR`, run:

```bash
XORL_TRAIN_URL=http://127.0.0.1:6000 \
OPD_COORD_DIR=/shared/opd-coord/qwen36-mtp-smoke \
OPD_TEACHER_HEAD=/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
examples/server/opd_singleshot_mtp_qwen36/run_driver.sh
```

Set `OPD_MTP_MASK_TOKEN_ID` to a valid in-vocabulary special or reserved token
for the tokenizer if the default `248063` (`<|fim_pad|>` in the Qwen3.6
tokenizer) is not suitable. Override the ConfAdapt method through
`OPD_SINGLESHOT_MTP_JSON`, for example:

```bash
OPD_SINGLESHOT_MTP_JSON='{"k_toks":2,"mask_token_id":248063,"sampling_mode":"native","train_rollout_strategy":"confidence","mtp_strategy":["conf_adapt",0.6],"rollout_replay":true,"validate_native_mtp_trace":true,"native_mtp_debug_trace":true}'
```

The default script sets `OPD_SKIP_OPTIM_STEP=1` so the first pass exercises
student rollout, AR teacher hidden-cache generation, and MTP OPD
forward/backward without weight sync. Clear that flag only after the smoke is
healthy.

## Kubernetes Smoke

The runnable cluster manifests are in
`experiments/opd_profile/k8s/qwen36_singleshot_mtp/`. They are based on the
existing OPD k8s pipeline shape, add `team: turbo` to GPU pod templates, use the
Qwen3.6 HF snapshot and EP8 DCP checkpoint, and pass `OPD_SINGLESHOT_MTP_JSON`
to the local OPD driver.

```bash
OPD_MANIFEST_DIR=experiments/opd_profile/k8s/qwen36_singleshot_mtp \
scripts/opd/submit_k8s_pipeline.sh q36-mtp-smoke-$(date -u +%m%d%H%M)
```
