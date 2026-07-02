# ZORL experiments

For score/apply throughput optimization, start with
`ZORL_PERFORMANCE_HANDOFF_RUNBOOK_2026_06_05.md`. It describes the current
large-population Wordle run, the SGLang-native ZORL implementation, and the
microbenchmark/eval that the next agent should build before hillclimbing.

For the older consolidated handoff state, merge decisions, autoresearch loop,
and next-agent tasks, see `ZORL_HANDOFF_RUNBOOK_2026_06_03.md`.

Recipe configs, k8s manifests, and the Countdown harness that the
ZORL (zeroth-order RL with antithetic ES gradients) feature was
developed and validated against. Companion server-side code lives in
`src/xorl/server/zorl.py` and `src/xorl/server/api_server/training_ops.py`;
the password harness changes are in `examples/server/password_memorization/run_password_test.py`.

## Layout

```
experiments/zorl/
├── README.md                          # this file
├── eval_countdown.py                  # standalone greedy-temp=0 evaluator
├── run_countdown_test.py              # Countdown harness (gradient | zorl | grpo modes)
├── run_coderforge_zorl_smoke.py       # CI smoke for the ZORL plumbing
├── standalone/                        # SGLang-native ZORL client, tasks/, PS drivers
├── autoresearch/                      # queue/controller/candidates for ZORL autoresearch
├── results/                           # gitignored — local run outputs
└── scripts/, sweep/                   # (existing)

NOTE (2026-07-02 get-right): k8s manifests and training-recipe YAML now live in
the xorl-infra repo (`k8s/zorl/`, `configs/zorl/`), not here. The ZORL server
module lives in the xorl repo (`src/xorl/server/zorl.py`, branch `zorl-ps`).
The autoresearch controller resolves `base_manifest: k8s/...` via a repo-root
`k8s` symlink (gitignored): create it once per checkout with
`ln -s ../xorl-infra/k8s/zorl k8s` (assumes a sibling xorl-infra checkout).
```

## Autoresearch

The autoresearch setup lives in `experiments/zorl/autoresearch/`. It can render
node-pinned long-lived SGLang controllers for nodes `047`, `117`, and `001`,
then launch and score queued ZORL trainer or standalone-client candidates.

Start from:

```
python experiments/zorl/autoresearch/controller.py next
python experiments/zorl/autoresearch/controller.py launch --id ZORL-001 --dry-run
python experiments/zorl/autoresearch/controller.py launch --id ZORL-OPD-000 --dry-run
```

Operator docs:

- `ZORL_AUTORESEARCH_RESEARCH_MEMO_2026_06_03.md`
- `ZORL_AUTORESEARCH_SPEC_2026_06_03.md`
- `ZORL_AUTORESEARCH_RUNBOOK_2026_06_03.md`

## Two validated recipes

**Qwen3-8B password memorization (rank=1).**
`reward_mode=teacher_forced + lr=1e-1 + momentum=0.9 + num_pairs=64 + steps=48`
→ 3/3 exact at parent probe ~40 in ~6.5 min on 1×H100. Use the
`qwen3-coder-30b-a3b-zorl-password-trainer-only.yaml` shape (point at
`qwen3_8b_zorl_password_r1.yaml`).

**Qwen3-Coder-30B-A3B Countdown 24-puzzle.**
Modal-style multi-rollout: `--zorl-num-pairs 8 --zorl-rollouts-per-puzzle 4
--zorl-rollout-temperature 0.6 --zorl-score-batch-size 128 --zorl-score-max-workers 4
--zorl-b-sigma 0.1 --learning-rate 1e-2 --zorl-sgd-momentum 0.0`,
`steps=64`, `LORA_RANK=4`. Cold base 1/8 → 2/8 greedy at probe ≥ 16 (held through 64).

Reference numbers and the SGLang stability flags
(`--disable-overlap-schedule + --max-running-requests 64 + --max-queued-requests 256`)
that prevent the silent decode-wedge under sustained multi-LoRA + temp>0
traffic are baked into `qwen3-coder-30b-a3b-zorl-password-sglang-tp8.yaml`.

## Operator notes for the k8s manifests

The yamls assume the current `apanda` namespace storage layout:

- A `home-apanda` PersistentVolumeClaim mounted at `/workspace/home`
  for the user's repo checkouts and HF cache.
- A `shared-data` PVC mounted at `/shared` for cross-pod artifacts
  (HF datasets cache, SGLang model paths, etc.).
- The `compute` node-pool with `node-group: default` taint, and an
  `nccl` node-group toleration for the multi-rank trainer manifests.

If your cluster uses different names, edit the yamls in place or
overlay via kustomize / yq before applying. There is no namespace
field — pods land in your active `kubectl` context's namespace.

## Eval

`eval_countdown.py` loads any saved adapter directory into a running
SGLang via `/load_lora_adapter` and runs greedy temp=0 on the 8-puzzle
eval set. Used to confirm parent-probe scores out-of-band from the
training loop:

```
python experiments/zorl/eval_countdown.py \
    --infer-url http://<sglang-service>:30000 \
    --lora-name probe-gen32 \
    --lora-path /shared/path/to/saved/adapter
```
