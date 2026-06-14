# OPD Pipeline Validation

`run_opd_pipeline.py` drives the full on-policy distillation loop against an
already-running XORL trainer, student SGLang server, and teacher prefill
service. The default teacher backend is XORL so the teacher can load DCP
checkpoints via `load_weights_mode: skip`.

```text
student SGLang -> teacher XORL hidden cache -> xorl forward_backward(opd_loss)
              <- xorl sync_inference_weights / sampler refresh
```

The script expects:

- `XORL_TRAIN_URL`: xorl training server URL.
- `OPD_COORD_DIR`: shared directory containing `student.json` and `teacher.json`.
- `OPD_TEACHER_HEAD`: teacher LM-head source for trainer-side KL. This may be
  an OPD teacher-store directory, a safetensors file, or an HF model directory.
- `OPD_TEACHER_BACKEND`: `xorl` by default; set `sglang` for the legacy teacher
  hidden-state path.
- Student and teacher coord files shaped like `{"ready": true, "host": "...", "port": 30001}`.

Each run writes a per-step throughput profile to
`$OPD_COORD_DIR/artifacts/opd_profile.jsonl` unless `OPD_PROFILE_OUTPUT` or
`--profile-output` is set. The profile records student sampling, teacher
prefill, teacher-cache save, trainer forward/loss, trainer backward, OPD KL,
optimizer, weight sync, and total step time. Set `OPD_PROFILE_SYNC_CUDA=1` or
pass `--profile-sync-cuda` when exact CUDA phase boundaries are more important
than profiling overhead. The driver batches student sampling and teacher
hidden-cache prefill requests by default, and excludes the first step from the
steady-state timing summary unless `OPD_PROFILE_WARMUP_STEPS` or
`--profile-warmup-steps` says otherwise.

By default, each weight sync sends a per-step `weight_version` to SGLang and
fails the run if `/model_info` does not report that exact version after sync.
Use `OPD_ALLOW_SYNC_VERSION_MISMATCH=1` or `--allow-sync-version-mismatch`
only for profiling environments where the connected SGLang build cannot report
sync acknowledgements.

Set `OPD_PIPELINE_CHUNK_SIZE` or pass `--pipeline-chunk-size` to split each
OPD step into prompt chunks. In this mode the driver prepares later chunks
with background student-sampling and teacher-prefill requests while the trainer
runs `forward_backward` on earlier chunks. Gradients accumulate across chunk
`forward_backward` calls, and the driver still runs one optimizer step and one
sampler sync after all chunks for the policy version are consumed. Use
`OPD_PIPELINE_PREFETCH_CHUNKS` and `OPD_PIPELINE_TEACHER_CONCURRENCY` to bound
client-side prefetch and teacher request concurrency.

`submit_k8s_pipeline.sh` can apply a cluster-specific three-job manifest set,
but those manifests are not committed because they depend on namespace, PVC,
image, node, and security-policy details. Keep them in an operator-owned runbook
or private environment repo and pass their directory with `OPD_MANIFEST_DIR`:

```bash
OPD_MANIFEST_DIR=/path/to/opd/manifests scripts/opd/submit_k8s_pipeline.sh
```

The manifest helper prefers `teacher-xorl.yaml` when present and falls back to
`teacher-sglang.yaml` for older runs.
