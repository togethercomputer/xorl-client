# Endpoint-driven Wordle GRPO

This example runs the Wordle RL experiment against trainer and SGLang/SMG
endpoints that you have already started. It does not create Kubernetes objects,
discover pods, schedule GPUs, or manage sampler processes.

Install the example dependencies from a checkout or installed distribution and run:

```bash
pip install -e '.[examples]'
python -m examples.wordle.train \
  --backend xorl \
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
ordering for every step: complete grouped rollouts, forward/backward, durable
pre-optimizer correctness gate, optimizer step, sampler sync, checkpoint, and
step artifacts. It also performs a final sync after the last optimizer step.

`model.model` identifies the sampler model and remains the tokenizer fallback.
Set `model.train_base_model` only when the trainer endpoint expects a different
base-model identifier or checkpoint path; otherwise it defaults to `model.model`.

Presets

- `importance_sampling.yaml` expects ordinary native SGLang `/generate`,
  decision-token logprobs, LoRA weight synchronization, and XoRL's
  `importance_sampling` trainer loss. It sends no R3 fields.
- `cispo.yaml` sends `loss_fn: cispo` with explicit absolute importance-ratio
  bounds `clip_low_threshold: 0.0` and `clip_high_threshold: 4.0`. It requires
  a trainer endpoint built with the corresponding XoRL server-side CISPO loss.
- `zero_k3.yaml` additionally requires the already-running trainer and sampler
  to use a coherent exact or batch-invariant path. Its pre-optimizer gate
  requires literal K3 `0.0` and ratio error `0.0`; it does not launch or
  reconfigure endpoints. Other presets do not inherit this literal-zero rule.
- `r3.yaml` streams routed indices and selected-router weights through shared
  binary spans. It requires XoRL `7f7c688e2bd80228d051d9a9e7962878b1c27890`
  (PR #57) and xorl-sglang `e46810ef5711456d1a7c5bb55b23a8d00d970aaa`
  (PR #21), or later compatible commits preserving those public schemas. Start
  SGLang with `--enable-return-routed-experts`,
  `--enable-return-expert-logits`, and
  `--routed-experts-side-channel-dir <shared-root>`. The shared root must be
  mounted at the same absolute path in SGLang and XoRL, and XoRL must include
  it in `XORL_R3_SHARED_ROOTS`.

`q36.yaml` is the shared Qwen3.6 configuration for all three backends. Every
shipped preset uses the exact production dictionary and ordered
`data/train_targets.txt` and `data/eval_targets.txt` split. The smaller presets
shorten the run geometry only; they do not substitute a different train/eval
split.

When `/server_info` is available, its response is recorded and demonstrably
incompatible capabilities fail before training. An unavailable capability
endpoint is recorded as unavailable, not silently treated as proof of support.
No client-side stale-weight-version filter is part of this example.

Data and held-out safety

The exact production files are documented in `data/README.md`. The loader
preserves the ordered 4,000-target training pool and ordered 170-target held-out
panel, rejects count mismatches, overlap, duplicates, or non-dictionary targets,
and records every dataset hash in `source_info.json`. Training target selection
then uses continuous shuffled epochs without replacement. Held-out hooks expose
only `task.pools.held_out`; training selection cannot draw from that tuple.

Artifacts and resume

The output directory is the source of truth and contains `run_config.json`,
`source_info.json`, `metrics.jsonl`, per-step JSON under `steps/`,
`checkpoints.jsonl`, and `terminal_audit.json`. URLs are saved without userinfo
or query strings. Resume restores optimizer state from the last checkpoint and
validates model, model ID, session, and completed step before advancing:

```bash
python -m examples.wordle.train \
  --backend xorl \
  --config examples/wordle/configs/importance_sampling.yaml \
  --trainer-url http://trainer:8000 \
  --generation-url http://sampler-router:30000 \
  --sync-url http://sampler-a:30000 \
  --resume-from artifacts/wordle/run-001
```

W&B may consume `metrics.jsonl`, but publication is optional and is not used to
decide whether a run completed.

Matched Tinker comparison

`configs/tinker_matched.yaml` pins the matched-Tinker convergence posture used
by the rl-bench Wordle harnesses: clipped PPO with ratio bounds [0.8, 1.28]
(XoRL offsets `eps_clip=0.2` / `eps_clip_high=0.28`, Tinker absolute
thresholds), constant LR 1e-5, Adam (0.9, 0.95, 1e-8), grad clip 1.0, no
weight decay, and — via three trainer flags — the matched advantage semantics:

- `advantage_std_normalization: false` — group mean-centred advantages with no
  per-group std rescaling.
- `skip_zero_advantage_trajectories: false` — every member of a kept group is
  trained. Zero-advantage episodes contribute no policy gradient, but they do
  count in XoRL's valid-token loss denominator, so dropping them silently
  rescales the gradient relative to Tinker.
- `remove_constant_reward_groups: true` — groups whose rewards are all equal
  are dropped wholesale; if every group in a step is uniform, one group is
  kept (all-zero advantages) so the optimizer step still runs and Adam state
  stays step-aligned across backends.

Datum shape: this example submits one datum per turn with the re-rendered
prompt masked out, while the harnesses pack an episode into one prefix-extended
sequence. The trained-token multiset and advantages are identical, so under
XoRL's valid-token-sum reduction the gradients match; the difference is compute
layout, not semantics.

Every committed step also carries an `overlay` block: a flat superset of the
rl-bench/Tinker metric names (`quality/train_reward`, `quality/solve_rate`,
`env/all/*`, `optim/kl_sample_train_v1/v2/v3`, `optim/entropy`, `tokens/*`,
`bench/*`, `perf/*`, `global_step`). W&B logs these keys unprefixed against a
`global_step` axis so curves overlay directly with harness runs; all original
metric names remain unchanged, so the two schemas coexist as a superset. Note
the KL sign convention: the harness defines `d = logp_sampler - logp_trainer`,
this example's internal `logratio_*` metrics use the opposite sign, and the
overlay does the mapping (`v1 = -logratio_mean`, `v2 = sq_logratio_mean / 2`,
`v3 = k3_mean`, which is symmetric).

To compare runs, execute the same matched config on both backends and align
them:

```bash
python -m examples.wordle.train --backend xorl \
  --config examples/wordle/configs/tinker_matched.yaml \
  --trainer-url http://trainer:8000 --generation-url http://router:30000 \
  --sync-url http://sampler-a:30000 --output-dir artifacts/wordle/matched-xorl
python -m examples.wordle.train --backend tinker \
  --config examples/wordle/configs/tinker_matched.yaml \
  --output-dir artifacts/wordle/matched-tinker   # needs TINKER_API_KEY
python -m examples.wordle.compare_tinker \
  --xorl-dir artifacts/wordle/matched-xorl \
  --tinker-dir artifacts/wordle/matched-tinker
```

`compare_tinker.py` prints a per-step table (reward, K3, grad norm for both
stacks) and writes `comparison.json` with tail-mean reward gaps. Passing
`--launch` runs the two `train` invocations itself before comparing. The
matched config also enables `sync_pool_per_endpoint`, registering each sampler
as its own pairwise NCCL pool (`r0`, `r1`, ...) with a distinct rendezvous
port — a single sync group spanning a multi-replica fleet is known to hang.
