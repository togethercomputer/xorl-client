# Endpoint-driven Wordle GRPO

This example runs the Wordle RL experiment against trainer and SGLang/SMG
endpoints that you have already started. It does not create Kubernetes objects,
discover pods, schedule GPUs, or manage sampler processes.

Install the example dependencies and run:

```bash
pip install -e '.[examples]'
python -m examples.wordle.train \
  --config examples/wordle/configs/importance_sampling.yaml \
  --trainer-url http://trainer:8000 \
  --generation-url http://sampler-router:30000 \
  --sync-url http://sampler-a:30000 \
  --sync-url http://sampler-b:30000 \
  --output-dir artifacts/wordle/run-001
```

The generation URL may be a router, but every `--sync-url` must be a direct
sampler endpoint with an explicit port. The runner registers those direct
endpoints with the trainer, performs an initial weight sync, then preserves this
ordering for every step: complete grouped rollouts, forward/backward,
optimizer step, sampler sync, checkpoint, and step artifacts. It also performs
a final sync after the last optimizer step.

Presets

- `importance_sampling.yaml` expects ordinary native SGLang `/generate`,
  decision-token logprobs, LoRA weight synchronization, and XoRL's
  `importance_sampling` trainer loss. It sends no R3 fields.
- `zero_k3.yaml` additionally requires the already-running trainer and sampler
  to use a coherent exact or batch-invariant path. It gates recorded K3 and
  ratio error; it does not launch or reconfigure endpoints.
- `r3.yaml` documents the future routed-index and selected-router-weight seam.
  Loading it currently fails closed because that public transport has not
  landed; it is not advertised as a runnable or production-ready preset.

When `/server_info` is available, its response is recorded and demonstrably
incompatible capabilities fail before training. An unavailable capability
endpoint is recorded as unavailable, not silently treated as proof of support.
No client-side stale-weight-version filter is part of this example.

Data and held-out safety

The shipped compact vocabulary and its provenance are documented in
`data/README.md`. Both input files are hashed into `source_info.json`. The task
creates deterministic disjoint training and held-out pools, and training target
selection uses shuffled epochs without replacement. Replace the file paths in a
preset to use a larger appropriately licensed vocabulary. Held-out hooks expose
only `task.pools.held_out`; training selection cannot draw from that tuple.

Artifacts and resume

The output directory is the source of truth and contains `run_config.json`,
`source_info.json`, `metrics.jsonl`, per-step JSON under `steps/`,
`checkpoints.jsonl`, and `terminal_audit.json`. URLs are saved without userinfo
or query strings. Resume restores optimizer state from the last checkpoint and
validates model, model ID, session, and completed step before advancing:

```bash
python -m examples.wordle.train \
  --config examples/wordle/configs/importance_sampling.yaml \
  --trainer-url http://trainer:8000 \
  --generation-url http://sampler-router:30000 \
  --sync-url http://sampler-a:30000 \
  --resume-from artifacts/wordle/run-001
```

W&B may consume `metrics.jsonl`, but publication is optional and is not used to
decide whether a run completed.
