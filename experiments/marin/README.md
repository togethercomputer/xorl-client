# marin — marin #6279 `rlvr7500_w1` GRPO math-RL repro (client side)

Client-side driver, verifier, and eval harness for the reproduction of marin issue #6279's
`rlvr7500_w1` run: forced-thinking GRPO math RL on the `laion/delphi-...-wc386k_lr1e5-sft` dense
Qwen3 ~9.7B SFT checkpoint, against an xorl training server + SGLang samplers (disaggregated
server-mode RL).

Result: reproduced and exceeded the reference reward trajectory (last-10 mean +0.436 vs reference
+0.247; reference per-step peak +0.345; ahead on 132/145 correctly-aligned steps). The reference
`metrics.csv` concatenates 3 restart segments — always dedup by `trainer/global_step` keep-last;
`summarize_repro_progress.py` and `import_reference_wandb.py` handle this.
Key finding: pipeline-RL (policy-lag 1) collapsed at step ~21 (off-policy lag × length-penalty →
box-cliff death spiral); the fix is fully on-policy operation. Full RCA + run ledger live in the
engine repo at `experiments/marin_rl_6279/{RUNBOOK,HANDOFF}.md` (branch `feature/marin-rl-6279`).

## Layout

```
standalone/
  train_marin_grpo.py       RL driver: rollout (G/prompt via SMG) → boxed reward + length
                            penalty → GRPO group advantages → forward_backward(drgrpo) →
                            optim_step → weight sync → metrics/W&B
  eval_math_suite.py        MATH500 / gsm8k (strict+flex) / AIME24 (seeds 42-51) eval client,
                            temp 0.7 / top_p 1.0, 4k budget; prints the ΔSFT table
  advantages.py             GRPO group-relative advantages + Datum construction + client chunking
  dataset.py                RLVR-MATH-7500 + eval-set loaders
  length_penalty.py         Concise-correct reward shaping (lpw=1.0)
  prompts.py                Forced-thinking prefill (<|start_think|>\n) + prompt rendering
  summarize_eval.py         Eval-run summarizer
  summarize_repro_progress.py  Reward-trajectory vs reference comparator
  tasks/verifier.py         Minerva/boxed `Answer: \boxed{...}` grader (same fn for RL + eval)
  data/                     aime24_evalchemy.jsonl, delphi_v0.jinja2
  tests/                    CPU test suite (88 tests)
```

## Running

The stack is three groups of plain servers plus this driver (the RL loop never links against
the engine — it talks to a Tinker-compatible API: `forward_backward` / `optim_step` /
`save_weights_for_sampler` / `sample`):

1. **Samplers**: 8× single-GPU SGLang servers behind a round-robin router, launched with
   `--rl-on-policy-target xorl-batch-invariant --enable-fp32-lm-head --attention-backend fa3
   --disable-custom-all-reduce` and env `SGLANG_BATCH_INVARIANT_OPS=mean,rms_norm`,
   `SGLANG_RMSNORM_FP32_WEIGHT_MUL=1`, `SGLANG_DISABLE_ROPE_COMPILE=1`,
   `SGLANG_FLA_TRIL_PRECISION=ieee`.
2. **Trainer**: one 8-GPU [xorl](https://github.com/togethercomputer/xorl) server
   (`python -m xorl.server.launcher --mode auto --config <cfg>.yaml`), FSDP2 shard 8, with
   `lm_head_fp32: true`, `rmsnorm_mode: sglang`, env `XORL_BATCH_INVARIANT_MATMUL=1`, and
   KV-cache-preserving P2P weight sync (`cache_invalidation_mode=none`). All engine-side
   features used here are in the public xorl repo (`apanda-dev`).
3. **Driver**: `standalone/train_marin_grpo.py` from this directory — drgrpo
   (`beta=0`, clip 0.2/0.28, `kl_type=k3`, `logprob_temperature=0.7`), fully on-policy.

This numerics profile is what holds the sampler↔trainer K3 at ~1e-8 (clipping never fires);
drop pieces of it and you reintroduce off-policy error. Training curves + the imported SkyRL
reference trajectory are public in
[W&B](https://wandb.ai/together-research/xorl-marin-rl-6279): runs `nw155nmj` (training),
`3comlb0c` (reference), `7odwm4bz` (timing).

Extra deps beyond `xorl_client`: `math_verify` (grader), `transformers` (tokenizer), `datasets`
(HF data loading).

Tests: `PYTHONPATH=. python -m pytest experiments/marin/standalone/tests` from the repo root,
using an interpreter with the deps above.
