"""Import the Marin/SkyRL reference trajectory into W&B for visual comparison.

The reference ``training_logs/metrics.csv`` concatenates one block per SLURM
restart segment, so ``trainer/global_step`` overlaps across blocks (w1: 1-82,
51-134, 101-145). This importer dedups by global step keeping the LAST
(resume-authoritative) occurrence and logs the full trajectory as a single
run, aligned to the repro driver's 0-indexed ``policy_step`` axis
(``policy_step = global_step - 1``).

Besides the raw CSV column names, the reference reward/pass@16 are also logged
under the driver's key names (``rollout/mean_reward``, ``rollout/pass_at_16``)
so they overlay directly against a repro run in the same panel with
``policy_step`` as the x-axis.

Usage:
    python -m experiments.marin_rl_6279.import_reference_wandb \
        [--csv PATH] [--project xorl-marin-rl-6279] [--entity together-research] \
        [--name reference-rlvr7500_w1-full] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_REFERENCE_CSV = Path(
    "/shared/xorl-marin-rl-6279/checkpoints/"
    "delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B/training_logs/metrics.csv"
)
ALIAS_KEYS = {
    "loss/avg_final_rewards": "rollout/mean_reward",
    "reward/avg_pass_at_16": "rollout/pass_at_16",
}


def read_deduped_reference(path: Path) -> tuple[dict[int, dict[str, float]], list[str]]:
    """Return {global_step: numeric row} keeping the last occurrence per step."""
    rows: dict[int, dict[str, float]] = {}
    segments: list[str] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            log_file = row.get("log_file", "")
            if log_file and (not segments or segments[-1] != log_file):
                segments.append(log_file)
            global_step = int(float(row["trainer/global_step"]))
            numeric: dict[str, float] = {}
            for key, value in row.items():
                if key == "log_file" or value in (None, ""):
                    continue
                try:
                    numeric[key] = float(value)
                except ValueError:
                    continue
            rows[global_step] = numeric
    return rows, segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--project", default="xorl-marin-rl-6279")
    parser.add_argument("--entity", default="together-research")
    parser.add_argument("--name", default="reference-rlvr7500_w1-full")
    parser.add_argument("--tags", nargs="*", default=["reference", "rlvr7500-w1", "marin-rl-6279", "full-dedup"])
    parser.add_argument("--dry-run", action="store_true", help="Print what would be logged without touching W&B.")
    args = parser.parse_args()

    reference, segments = read_deduped_reference(args.csv)
    steps = sorted(reference)
    print(f"reference: {len(steps)} deduped steps ({steps[0]}..{steps[-1]}) from {len(segments)} segments")
    if args.dry_run:
        for step in (steps[0], steps[len(steps) // 2], steps[-1]):
            print(step, {k: reference[step][k] for k in ("loss/avg_final_rewards", "reward/avg_pass_at_16")})
        return 0

    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        tags=args.tags,
        config={
            "note": (
                "Full reference trajectory imported from the downloaded Marin/SkyRL training logs, "
                "deduped across restart segments by trainer/global_step (keep-last). "
                "Supersedes the segment-1-only import (totwpo79)."
            ),
            "source": str(args.csv),
            "segments": segments,
            "dedup": "trainer/global_step keep-last",
            "step_axis": "policy_step = global_step - 1 (matches the repro driver's 0-indexed steps)",
        },
    )
    run.define_metric("policy_step")
    run.define_metric("*", step_metric="policy_step")
    for step in steps:
        payload = dict(reference[step])
        for src, alias in ALIAS_KEYS.items():
            if src in payload:
                payload[alias] = payload[src]
        payload["policy_step"] = step - 1
        run.log(payload, commit=True)
    run.finish()
    print(f"logged {len(steps)} steps to {args.entity}/{args.project}/{run.id} ({args.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
