# MTP Experiment Artifacts

This directory is the xorl-client home for MTP science docs and client-side harness files imported during
the 2026-06-14 consolidation pass.

Start with:

- `CONSOLIDATION_HANDOFF.md` for branch, validation, and ownership state.
- `docs/notes/mtp_amdahl_optimal_throughput_handoff.md` for the current throughput conclusion.
- `docs/notes/mtp_science_next_experiments_20260613.md` for the k=2 continuation and k=2 -> k=4 widening plan.
- `scripts/opd/` for OPD/MTP client harness scripts.
- `opd_profile/` for non-k8s profile harness notes and analysis helpers.

K8s manifests and generators are intentionally not duplicated here. Use
`/home/apanda/xorl-infra/k8s/opd_profile` as the canonical k8s location.

Run outputs must go under `/shared`, not into this repository.
