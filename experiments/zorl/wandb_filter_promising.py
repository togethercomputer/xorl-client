"""Filter the together-research/zorl wandb project down to promising runs.

A run is "promising" when its best score metric shows a real climb over the
run's own starting value (ES on a frozen base starts at the cold floor, so
absolute value alone conflates warm starts with learning). We therefore rank
by IMPROVEMENT (best - first) on the run's primary score metric, with a floor
on steps so single-probe noise doesn't rank.

Usage:
  python experiments/zorl/wandb_filter_promising.py [--entity together-research]
      [--project zorl] [--min-steps 10] [--min-gain 0.02] [--top 40]
      [--json out.json]

Output: a ranked table (and optional JSON) of run name/id/url, config summary
(task, perturbation mode, rank, sigma, lr, pairs, optimizer, model), metric
name, first -> best values, gain, steps, state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import wandb

# Score metrics we recognize, in priority order: the first present in a run's
# history/summary becomes its primary metric.
METRIC_PRIORITY = [
    "probe/solve_rate",
    "probe/exact_rate",
    "probe/reward_mean",
    "eval/solve_rate",
    "eval/exact_rate",
    "eval/reward_mean",
    "parent/exact_rate",
    "parent/reward_mean",
    "train/exact_rate",
    "train/reward_mean",
    "reward_mean",
    "exact_rate",
    "solve_rate",
]

CONFIG_KEYS = [
    "task", "perturbation_mode", "lora_rank", "b_sigma", "sigma",
    "learning_rate", "muon_lr", "num_pairs", "optimizer", "model",
    "update_strategy", "score_mode",
]


def pick_metric(keys: set[str]) -> str | None:
    for m in METRIC_PRIORITY:
        if m in keys:
            return m
    # fall back to anything that looks like a score
    for k in sorted(keys):
        if re.search(r"(solve|exact|reward)", k) and not re.search(
            r"(std|min|max|count|time|norm)", k
        ):
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="together-research")
    ap.add_argument("--project", default="zorl")
    ap.add_argument("--min-steps", type=int, default=10)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    runs = api.runs(f"{args.entity}/{args.project}", per_page=200)

    rows = []
    for run in runs:
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        metric = pick_metric(set(summary.keys()))
        if metric is None:
            continue
        steps = run.summary.get("_step", 0) or 0
        if steps < args.min_steps:
            continue
        # One cheap history pass over just the metric column: first + best.
        first = best = None
        try:
            for row in run.scan_history(keys=[metric], page_size=500):
                v = row.get(metric)
                if v is None:
                    continue
                v = float(v)
                if first is None:
                    first = v
                best = v if best is None else max(best, v)
        except Exception as e:  # noqa: BLE001 — a single bad run must not kill the sweep
            print(f"  [warn] history read failed for {run.id}: {e}", file=sys.stderr)
        if first is None or best is None:
            continue
        gain = best - first
        if gain < args.min_gain:
            continue
        cfg = {k: run.config.get(k) for k in CONFIG_KEYS if run.config.get(k) is not None}
        rows.append({
            "name": run.name,
            "id": run.id,
            "url": run.url,
            "state": run.state,
            "steps": int(steps),
            "metric": metric,
            "first": round(first, 4),
            "best": round(best, 4),
            "gain": round(gain, 4),
            "config": cfg,
        })

    rows.sort(key=lambda r: r["gain"], reverse=True)
    rows = rows[: args.top]

    width = max((len(r["name"]) for r in rows), default=10)
    print(f"{'run':<{width}}  {'metric':<22} {'first':>7} {'best':>7} {'gain':>7} {'steps':>6}  state")
    for r in rows:
        print(f"{r['name']:<{width}}  {r['metric']:<22} {r['first']:>7} {r['best']:>7} "
              f"{r['gain']:>7} {r['steps']:>6}  {r['state']}")
        print(f"{'':<{width}}  {r['url']}  {json.dumps(r['config'])}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {len(rows)} runs -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
