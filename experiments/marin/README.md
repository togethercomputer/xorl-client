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

Launch manifests + rebuild script are in xorl-infra `k8s/marin/` (see its README for the pinned
SGLang SHA `362725903`, the full numerics env profile, and the stack topology). Engine-side
changes are upstreamed as xorl-internal PRs #431/#432/#433.

Extra deps beyond `xorl_client`: `math_verify` (grader), `transformers` (tokenizer), `datasets`
(HF data loading). The k8s driver pods satisfy these via the engine checkout's venv.

Tests: `PYTHONPATH=. python -m pytest experiments/marin/standalone/tests` from the repo root,
using an interpreter with the deps above.
