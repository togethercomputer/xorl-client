# HANDOFF - MTP consolidation battery

Last updated: 2026-06-14

## Current Status

- MTP engine landing PR: `togethercomputer/xorl-internal#371`, branch `pr/mtp`.
- Landing base: `/home/apanda/xorl-stage-mtp`, branch `mtp-merge-apanda-dev`, commit `3a6a6a02`.
- Required apanda-dev base: `609bed76`.
- SGLang paired fix: `/home/apanda/xorl-sglang-internal` branch `apanda-dev` contains
  `18c2357de Fix per-request MTP draft verification`; current head observed as `96e6109ca`.
- `pr/mtp` was fast-forwarded and pushed from `4070d9f5` to `3a6a6a02` after validation.
- PR #371 is no longer draft; `gh pr ready 371` completed and GitHub reports `mergeStateStatus=CLEAN`.
- Validation evidence was posted to PR #371:
  `https://github.com/togethercomputer/xorl-internal/pull/371#issuecomment-4700289006`.

## Coordinator Rules

- Branch off and merge into the new `apanda-dev@609bed76`.
- For MTP, do not start new work from the old `pr/mtp` head. New MTP work starts from
  `mtp-merge-apanda-dev@3a6a6a02`.
- Keep PR #371 as the MTP engine landing PR. It now carries the apanda-dev merge, subsystem,
  review fixes, and validation base.
- Outputs go to `/shared`, never into the checkout.
- Canonical artifact split:
  - MTP docs and client/harness files: `/home/apanda/xorl-client/experiments/mtp`.
  - MTP k8s/manifests: `/home/apanda/xorl-infra/k8s/opd_profile`.
  - SGLang runtime changes: `/home/apanda/xorl-sglang-internal`.
- Extract validated engine changes as focused PRs immediately; do not accumulate more engine work in the
  experiment checkout.

## Validation Completed

CPU layout gate on the landing base:

```bash
cd /home/apanda/xorl-stage-mtp
PYTHONPATH=/home/apanda/xorl-stage-mtp/src python -m pytest tests/mtp/test_singleshot_layout.py -q
```

Result: `17 passed in 0.10s`.

GPU MTP gate against the landing source:

```bash
cd /home/apanda/xorl-stage-mtp
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/apanda/xorl-stage-mtp/src:/home/apanda/xorl-stage-mtp/tests/distributed \
  python -m pytest \
  /home/apanda/xorl-mtp-singleshot-port-20260602/tests/distributed/test_qwen3_5_singleshot_cp.py -q
```

Result: `1 passed in 206.46s`.

Note: `mtp-merge-apanda-dev` only carries `tests/mtp/test_singleshot_layout.py` as the explicit MTP test file.
The GPU smoke above uses the richer experimental SingleShot CP replay test file while forcing
`/home/apanda/xorl-stage-mtp/src` onto `PYTHONPATH`, so the source under test is the landing branch.

Fast-forward and push performed after those gates:

```bash
git -C /home/apanda/xorl-stage-mtp switch pr/mtp
git -C /home/apanda/xorl-stage-mtp merge --ff-only mtp-merge-apanda-dev
git -C /home/apanda/xorl-stage-mtp push origin pr/mtp
```

Final stage status: `/home/apanda/xorl-stage-mtp` on `pr/mtp`, clean, matching `origin/pr/mtp@3a6a6a02`.

## Artifact Consolidation

Imported to `/home/apanda/xorl-client/experiments/mtp`:

- `docs/notes/mtp*.md`
- `docs/notes/singleshot_mtp_research_runbook.md`
- `docs/notes/opsd_low_mfu_microbench_20260613.md`
- `docs/notes/qwen36_35b_*throughput_sweep_20260612.md`
- `scripts/opd/`
- `examples/server/opd_singleshot_mtp_qwen36/`
- non-k8s `experiments/opd_profile/` harness files and runbooks

K8s state:

- `/home/apanda/xorl-infra/k8s/opd_profile` already contains the MTP k8s files.
- A byte comparison against `/home/apanda/xorl-mtp-singleshot-port-20260602/experiments/opd_profile/k8s`
  found no missing or differing non-bytecode files.
- This pass added `/home/apanda/xorl-infra/k8s/opd_profile/README.md` and updated the root infra README to
  spell out the current no-privileged/no-manual-CUDA-visible-devices policy.
- `/home/apanda/xorl-infra` was already on local branch `opd-battery-consolidation` with commit
  `5b8b486 Populate k8s manifests + configs from opd/mtp worktrees`.

## Science Continuation

Continue k=2 curriculum science on `mtp-merge-apanda-dev@3a6a6a02`, not on the old pre-merge PR head.

Recommended order:

1. Continue the k=2 run first to determine whether commit_len has plateaued near 1.85 or is still climbing.
2. Only after coordinating sampler reprogramming, run the k=2 -> k=4 widening test.
3. Use offline offset-2/3 probes only if the shared stack is busy.

Operational caveat: sampler outputs drift as the student becomes more confident. Full-stack throughput runs across
different sampler states are not clean A/B comparisons; prefer bounded same-state comparisons and profile components
inside a single run.

## Source Checkout Notes

- `/home/apanda/xorl-mtp-singleshot-port-20260602` remains the historical experiment checkout.
- The engine landing branch is now `/home/apanda/xorl-stage-mtp`, not this checkout.
- Keep future run outputs in `/shared`.
- Do not delete or rewrite the historical experiment notes here unless explicitly asked; canonical follow-up work
  should use the imported client artifacts and infra manifests listed above.
