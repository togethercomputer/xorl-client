"""Summarize a live Marin #6279 repro run against the downloaded reference metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_CSV = Path(
    "/shared/xorl-marin-rl-6279/checkpoints/"
    "delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B/training_logs/metrics.csv"
)


def _read_jsonl_metrics(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rollouts: list[dict[str, Any]] = []
    train_updates: list[dict[str, Any]] = []
    syncs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            if not isinstance(event, str):
                continue
            counts[event] += 1
            if event == "step_rollout_summary":
                rollouts.append(row)
            elif event == "train_update":
                train_updates.append(row)
            elif event == "weight_sync":
                syncs.append(row)
    return rollouts, train_updates, syncs, counts


def _read_reference(path: Path) -> dict[int, dict[str, float]]:
    """Read the reference metrics.csv keyed by ``trainer/global_step``.

    The reference CSV concatenates one block per SLURM restart segment, so
    global steps overlap across blocks (e.g. 1-82, 51-134, 101-145). Later
    rows are the resume-authoritative ones; keep the LAST occurrence of each
    global step. Row position must never be used as the step axis.
    """
    rows: dict[int, dict[str, float]] = {}
    with path.open() as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            global_step = int(float(row.get("trainer/global_step", idx + 1)))
            rows[global_step] = {
                "step": float(global_step),
                "reward": float(row["loss/avg_final_rewards"]),
                "pass16": float(row["reward/avg_pass_at_16"]),
                "step_s": float(row["timing/step"]),
                "generate_s": float(row["timing/generate"]),
                "policy_train_s": float(row["timing/policy_train"]),
                "avg_tokens": float(row["generate/avg_num_tokens"]),
            }
    return rows


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "-"


def _fmt_sci(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return "-"


def _train_by_step(train_updates: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in train_updates:
        step = row.get("step")
        if isinstance(step, int):
            by_step[step] = row
    return by_step


def _rollout_by_step(rollouts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rollouts:
        step = row.get("step")
        if isinstance(step, int):
            by_step[step] = row
    return by_step


def _policy_lag(rollout: dict[str, Any] | None) -> int | None:
    if not rollout:
        return None
    sampling_version = rollout.get("policy_weight_version_at_sampling")
    train_version = rollout.get("policy_weight_version_at_train")
    if isinstance(sampling_version, str) and isinstance(train_version, str):
        return int(sampling_version != train_version)
    lag = rollout.get("pipeline_policy_lag")
    if lag is None:
        return None
    try:
        return int(lag)
    except (TypeError, ValueError):
        return None


def _is_same_policy(rollout: dict[str, Any] | None) -> bool:
    lag = _policy_lag(rollout)
    return lag is None or lag == 0


def _events_by_step(path: Path) -> dict[int, Counter[str]]:
    by_step: dict[int, Counter[str]] = defaultdict(Counter)
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            event = row.get("event")
            if isinstance(step, int) and isinstance(event, str):
                by_step[step][event] += 1
    return by_step


def summarize(args: argparse.Namespace) -> int:
    metrics_path = Path(args.metrics_jsonl)
    reference_path = Path(args.reference_csv)
    rollouts, train_updates, syncs, counts = _read_jsonl_metrics(metrics_path)
    reference = _read_reference(reference_path) if reference_path.exists() else {}
    train_by_step = _train_by_step(train_updates)
    rollout_by_step = _rollout_by_step(rollouts)
    by_step = _events_by_step(metrics_path)

    print(f"metrics_jsonl: {metrics_path}")
    print(f"mtime_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(metrics_path.stat().st_mtime))}")
    print(
        "events: "
        f"rollout_records={counts.get('rollout', 0)} "
        f"rollout_summaries={len(rollouts)} train_updates={len(train_updates)} weight_syncs={len(syncs)}"
    )
    if by_step:
        tail = {step: dict(counter) for step, counter in sorted(by_step.items())[-args.tail :]}
        print(f"by_step_tail: {json.dumps(tail, sort_keys=True)}")
    if syncs:
        last_sync = syncs[-1]
        result = last_sync.get("result") if isinstance(last_sync.get("result"), dict) else {}
        print(
            "last_sync: "
            f"step={last_sync.get('step')} "
            f"success={result.get('success')} "
            f"flush_cache={result.get('flush_cache')} "
            f"cache_invalidation_mode={result.get('cache_invalidation_mode')} "
            f"waited_for_prefetch={last_sync.get('pipeline_waited_for_prefetch')}"
        )

    print()
    print(
        "step ours_reward ref_reward delta_reward ours_pass16 ref_pass16 "
        "ours_trunc mean_completion_toks rollout_tokps train_tokps policy_lag behavior_k3"
    )
    for rollout in rollouts[-args.tail :]:
        step = rollout.get("step")
        if not isinstance(step, int):
            continue
        # Our driver steps are 0-indexed; the reference global_step is 1-indexed,
        # so our step s is the same nth policy update as reference step s + 1.
        ref = reference.get(step + 1, {})
        train = train_by_step.get(step, {})
        derived = train.get("derived_train_metrics") if isinstance(train.get("derived_train_metrics"), dict) else {}
        policy_lag = _policy_lag(rollout)
        reward = rollout.get("mean_reward")
        ref_reward = ref.get("reward")
        delta_reward = float(reward) - float(ref_reward) if reward is not None and ref_reward is not None else None
        print(
            f"{step} "
            f"{_fmt_float(reward)} "
            f"{_fmt_float(ref_reward)} "
            f"{_fmt_float(delta_reward)} "
            f"{_fmt_float(rollout.get('pass_at_16'))} "
            f"{_fmt_float(ref.get('pass16'))} "
            f"{_fmt_float(rollout.get('truncated_fraction'))} "
            f"{_fmt_int(rollout.get('mean_completion_tokens'))} "
            f"{_fmt_int(rollout.get('total_tokens_per_s'))} "
            f"{_fmt_int(derived.get('valid_tokens_per_s'))} "
            f"{_fmt_int(policy_lag)} "
            f"{_fmt_sci(train.get('behavior_k3'))}"
        )

    same_policy_violations: list[tuple[int, float]] = []
    lagged_exceedances: list[tuple[int, float]] = []
    for train in train_updates:
        step = train.get("step")
        k3 = train.get("behavior_k3")
        if not isinstance(step, int) or k3 is None:
            continue
        try:
            k3_value = float(k3)
        except (TypeError, ValueError):
            continue
        if k3_value <= args.max_k3:
            continue
        rollout = rollout_by_step.get(step)
        if _is_same_policy(rollout):
            same_policy_violations.append((step, k3_value))
        else:
            lagged_exceedances.append((step, k3_value))
    if lagged_exceedances:
        step, k3_value = lagged_exceedances[-1]
        print(
            "NOTE: latest over-threshold behavior_k3 is on a pipelined policy-lag row: "
            f"step {step} behavior_k3 {k3_value:.3e} exceeds --max-k3 {args.max_k3:.3e}."
        )
    if same_policy_violations:
        step, k3_value = same_policy_violations[-1]
        print(
            "WARNING: latest same-policy behavior_k3 "
            f"step {step} value {k3_value:.3e} exceeds --max-k3 {args.max_k3:.3e}"
        )
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path, help="Path to the live run metrics.jsonl file.")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--tail", type=int, default=8, help="Number of recent rollout rows to print.")
    parser.add_argument("--max-k3", type=float, default=1e-6, help="Warn and exit nonzero if latest K3 exceeds this.")
    return parser.parse_args()


def main() -> int:
    return summarize(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
