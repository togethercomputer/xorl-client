"""Retro-log a repro run's per-step timing breakdown as a companion W&B run.

The RL driver did not emit a per-step ``timing/*`` family the way the SkyRL
reference logs do, so a finished repro run has no panel-comparable timing
series. This tool reconstructs one offline and logs it as a NEW run:

- ``timing/generate``      = ``rollout_wall_s`` from the run's metrics.jsonl
- ``timing/sync_weights``  = ``sync_future_done.wall_s``
- ``timing/save_hf_model`` = ``checkpoint_future_done.wall_s`` (sparse)
- ``timing/policy_train``  = residual train-phase wall, derived from the W&B
  timestamps of successive rollout summaries in the SOURCE run:
  ``ts(s+1) - ts(s) - rollout(s+1) - sync(s) - ckpt(s)``
- ``timing/step``          = generate + policy_train + sync + ckpt

Key names deliberately match the SkyRL reference import
(``reference-rlvr7500_w1-full``) so both overlay in one panel with
``policy_step`` as the x-axis. The last step has no successor timestamp, so
its ``timing/policy_train``/``timing/step`` are omitted.

Usage:
    python -m experiments.marin_rl_6279.import_repro_timing_wandb \
        [--metrics-jsonl PATH] [--source-run entity/project/run_id] \
        [--name NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_METRICS_JSONL = Path(
    "/shared/xorl-marin-rl-6279/runs/stack/20260701T193444Z-marin6279-repro-stack/metrics.jsonl"
)
DEFAULT_SOURCE_RUN = "together-research/xorl-marin-rl-6279/nw155nmj"


def read_phase_walls(path: Path) -> dict[str, dict[int, float]]:
    """Per-step phase walls + train throughput from the driver metrics.jsonl."""
    phases: dict[str, dict[int, float]] = {
        "rollout": {},
        "sync": {},
        "checkpoint": {},
        "fb_execution_sum": {},
        "train_valid_tokens_per_s": {},
    }
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            step = row.get("step", row.get("policy_step"))
            if not isinstance(step, int):
                continue
            if event == "step_rollout_summary" and row.get("rollout_wall_s") is not None:
                phases["rollout"][step] = float(row["rollout_wall_s"])
            elif event == "sync_future_done" and row.get("wall_s") is not None:
                phases["sync"][step] = float(row["wall_s"])
            elif event == "checkpoint_future_done" and row.get("wall_s") is not None:
                phases["checkpoint"][step] = float(row["wall_s"])
            elif event == "train_update":
                derived = row.get("derived_train_metrics")
                if isinstance(derived, dict):
                    if derived.get("forward_backward_execution_time_s") is not None:
                        phases["fb_execution_sum"][step] = float(derived["forward_backward_execution_time_s"])
                    if derived.get("valid_tokens_per_s") is not None:
                        phases["train_valid_tokens_per_s"][step] = float(derived["valid_tokens_per_s"])
    return phases


def read_rollout_timestamps(source_run: str) -> dict[int, float]:
    """policy_step -> unix timestamp of that step's rollout-summary W&B row."""
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    run = api.run(source_run)
    stamps: dict[int, float] = {}
    for row in run.scan_history(keys=["policy_step", "rollout/mean_reward", "_timestamp"]):
        step = row.get("policy_step")
        ts = row.get("_timestamp")
        if step is None or ts is None:
            continue
        stamps.setdefault(int(step), float(ts))
    return stamps


def build_timing_series(
    phases: dict[str, dict[int, float]], stamps: dict[int, float]
) -> dict[int, dict[str, float]]:
    series: dict[int, dict[str, float]] = {}
    steps = sorted(phases["rollout"])
    for step in steps:
        payload: dict[str, float] = {"timing/generate": phases["rollout"][step]}
        if step in phases["sync"]:
            payload["timing/sync_weights"] = phases["sync"][step]
        if step in phases["checkpoint"]:
            payload["timing/save_hf_model"] = phases["checkpoint"][step]
        if step in phases["fb_execution_sum"]:
            payload["timing/policy_train_rank_sum"] = phases["fb_execution_sum"][step]
        if step in phases["train_valid_tokens_per_s"]:
            payload["train/valid_tokens_per_s"] = phases["train_valid_tokens_per_s"][step]
        nxt = step + 1
        if step in stamps and nxt in stamps and nxt in phases["rollout"]:
            residual = (
                stamps[nxt]
                - stamps[step]
                - phases["rollout"][nxt]
                - phases["sync"].get(step, 0.0)
                - phases["checkpoint"].get(step, 0.0)
            )
            if residual >= 0:
                payload["timing/policy_train"] = residual
                payload["timing/step"] = (
                    payload["timing/generate"]
                    + residual
                    + phases["sync"].get(step, 0.0)
                    + phases["checkpoint"].get(step, 0.0)
                )
        series[step] = payload
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-jsonl", type=Path, default=DEFAULT_METRICS_JSONL)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--project", default="xorl-marin-rl-6279")
    parser.add_argument("--entity", default="together-research")
    parser.add_argument("--name", default="marin6279-repro-nopipe-timing")
    parser.add_argument("--tags", nargs="*", default=["timing-retro", "marin-rl-6279", "repro"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phases = read_phase_walls(args.metrics_jsonl)
    stamps = read_rollout_timestamps(args.source_run)
    series = build_timing_series(phases, stamps)
    step_walls = [p["timing/step"] for p in series.values() if "timing/step" in p]
    print(
        f"{len(series)} steps; timing/step available for {len(step_walls)}: "
        f"mean {statistics.mean(step_walls):.1f}s median {statistics.median(step_walls):.1f}s"
    )
    if args.dry_run:
        for step in (min(series), max(k for k, v in series.items() if "timing/step" in v)):
            print(step, {k: round(v, 1) for k, v in series[step].items()})
        return 0

    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        tags=args.tags,
        config={
            "note": (
                "Per-step timing breakdown reconstructed offline from the driver metrics.jsonl "
                "plus the source run's W&B row timestamps. timing/policy_train is the residual "
                "wall between successive rollout summaries minus rollout/sync/checkpoint walls; "
                "timing/policy_train_rank_sum is the server-reported forward-backward execution "
                "time SUMMED across DP ranks (not wall). Key names match the SkyRL reference "
                "import (reference-rlvr7500_w1-full) for direct panel overlay."
            ),
            "source_run": args.source_run,
            "metrics_jsonl": str(args.metrics_jsonl),
            "step_axis": "policy_step (matches the source run)",
        },
    )
    run.define_metric("policy_step")
    run.define_metric("*", step_metric="policy_step")
    for step in sorted(series):
        run.log({"policy_step": step, **series[step]}, commit=True)
    run.finish()
    print(f"logged {len(series)} steps to {args.entity}/{args.project}/{run.id} ({args.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
