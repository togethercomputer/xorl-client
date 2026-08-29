"""Compare a Wordle XoRL run against a Tinker run of the same configuration.

Offline by default: point ``--xorl-dir`` and ``--tinker-dir`` at the artifact
directories of two completed runs (each holding the ``metrics.jsonl`` the
runner writes) and this script aligns them step-by-step, prints a table, and
writes ``comparison.json``.

With ``--launch`` it first runs both backends sequentially through
``examples.wordle.train`` with one shared config (use
``configs/tinker_matched.yaml`` for the matched posture). The XoRL leg needs
``--trainer-url``/``--generation-url``/``--sync-url``; the Tinker leg needs
``TINKER_API_KEY``. Endpoints are supplied, not provisioned.

Example:
    python -m examples.wordle.compare_tinker \
        --xorl-dir artifacts/wordle/matched-xorl \
        --tinker-dir artifacts/wordle/matched-tinker
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_COMPARED_METRICS = ("reward", "solve_rate", "k3", "grad_norm", "loss")


def _leaf(key: str) -> str:
    return key.lower().replace(":", "/").rsplit("/", 1)[-1]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _extract_row(record: dict[str, Any]) -> dict[str, float] | None:
    """Normalize one metrics.jsonl record to the compared metric names.

    Prefers the harness-named ``overlay`` block; falls back to the legacy
    record fields so runs from before the overlay existed still compare.
    """
    step = record.get("step")
    if not isinstance(step, int):
        return None
    overlay = record.get("overlay") or {}
    forward = record.get("forward_backward") or {}
    optimizer = record.get("optimizer") or {}

    row: dict[str, float] = {"step": float(step)}

    reward = _number(overlay.get("quality/train_reward"))
    if reward is None:
        reward = _number(record.get("reward_mean"))
    if reward is not None:
        row["reward"] = reward

    solve = _number(overlay.get("quality/solve_rate"))
    if solve is not None:
        row["solve_rate"] = solve

    k3 = _number(overlay.get("optim/kl_sample_train_v3"))
    if k3 is None:
        k3 = _number(forward.get("k3_mean"))
    if k3 is not None:
        row["k3"] = k3

    grad = _number(overlay.get("optim/grad_norm"))
    if grad is None:
        for source in (optimizer, forward):
            for key, value in source.items():
                if _leaf(key) == "grad_norm":
                    grad = _number(value)
                    break
            if grad is not None:
                break
    if grad is not None:
        row["grad_norm"] = grad

    loss = _number(overlay.get("loss"))
    if loss is None:
        loss = _number(forward.get("loss"))
    if loss is not None:
        row["loss"] = loss
    return row


def load_run(directory: Path) -> dict[int, dict[str, float]]:
    metrics_path = directory / "metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"no metrics.jsonl under {directory}")
    rows: dict[int, dict[str, float]] = {}
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = _extract_row(record)
        if row is not None:
            rows[int(row["step"])] = row
    if not rows:
        raise SystemExit(f"{metrics_path} contains no comparable step records")
    return rows


def _tail_mean(values: list[float], n: int) -> float | None:
    return statistics.fmean(values[-n:]) if values else None


def build_comparison(
    xorl: dict[int, dict[str, float]],
    tinker: dict[int, dict[str, float]],
    *,
    last_n: int,
) -> dict[str, Any]:
    common = sorted(set(xorl) & set(tinker))
    per_step = []
    for step in common:
        entry: dict[str, Any] = {"step": step, "xorl": xorl[step], "tinker": tinker[step]}
        if "reward" in xorl[step] and "reward" in tinker[step]:
            entry["reward_delta"] = xorl[step]["reward"] - tinker[step]["reward"]
        per_step.append(entry)

    def series(run: dict[int, dict[str, float]], key: str) -> list[float]:
        return [run[step][key] for step in common if key in run[step]]

    summary: dict[str, Any] = {
        "steps_compared": len(common),
        "xorl_steps": len(xorl),
        "tinker_steps": len(tinker),
        "last_n": last_n,
    }
    for name, run in (("xorl", xorl), ("tinker", tinker)):
        rewards = series(run, "reward")
        k3s = series(run, "k3")
        summary[name] = {
            "final_reward": rewards[-1] if rewards else None,
            f"mean_reward_last_{last_n}": _tail_mean(rewards, last_n),
            "max_k3": max(k3s) if k3s else None,
            "final_grad_norm": (series(run, "grad_norm") or [None])[-1],
        }
    x_tail = _tail_mean(series(xorl, "reward"), last_n)
    t_tail = _tail_mean(series(tinker, "reward"), last_n)
    if x_tail is not None and t_tail is not None:
        summary["reward_gap_xorl_minus_tinker"] = x_tail - t_tail
    return {"per_step": per_step, "summary": summary}


def print_table(comparison: dict[str, Any]) -> None:
    def fmt(value: Any, spec: str = "8.4f") -> str:
        return format(value, spec) if isinstance(value, (int, float)) else " " * 8

    print(
        f"{'step':>4}  {'reward_x':>8} {'reward_t':>8} {'delta':>8}  "
        f"{'k3_x':>9} {'k3_t':>9}  {'gnorm_x':>8} {'gnorm_t':>8}"
    )
    for entry in comparison["per_step"]:
        x, t = entry["xorl"], entry["tinker"]
        print(
            f"{entry['step']:>4}  {fmt(x.get('reward'))} {fmt(t.get('reward'))} "
            f"{fmt(entry.get('reward_delta'))}  {fmt(x.get('k3'), '9.2e')} "
            f"{fmt(t.get('k3'), '9.2e')}  {fmt(x.get('grad_norm'), '8.3f')} "
            f"{fmt(t.get('grad_norm'), '8.3f')}"
        )
    print(json.dumps(comparison["summary"], indent=2, sort_keys=True))


def launch(backend: str, args: argparse.Namespace, output_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "examples.wordle.train",
        "--backend",
        backend,
        "--config",
        args.config,
        "--output-dir",
        str(output_dir),
    ]
    if backend == "xorl":
        if not (args.trainer_url and args.generation_url and args.sync_url):
            raise SystemExit(
                "--launch with the xorl leg requires --trainer-url, "
                "--generation-url, and at least one --sync-url"
            )
        command += ["--trainer-url", args.trainer_url]
        command += ["--generation-url", args.generation_url]
        for url in args.sync_url:
            command += ["--sync-url", url]
    print(f"[compare] launching {backend}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xorl-dir", required=True, type=Path)
    parser.add_argument("--tinker-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="comparison.json path (default: <xorl-dir>/comparison.json)")
    parser.add_argument("--last-n", type=int, default=10,
                        help="steps in the tail-mean reward summary")
    parser.add_argument("--launch", action="store_true",
                        help="run both backends via examples.wordle.train first")
    parser.add_argument("--skip-xorl", action="store_true",
                        help="with --launch: reuse the existing xorl run")
    parser.add_argument("--skip-tinker", action="store_true",
                        help="with --launch: reuse the existing tinker run")
    parser.add_argument("--config", default=None,
                        help="shared experiment config (required with --launch)")
    parser.add_argument("--trainer-url", default=None)
    parser.add_argument("--generation-url", default=None)
    parser.add_argument("--sync-url", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.launch:
        if not args.config:
            raise SystemExit("--launch requires --config")
        if not args.skip_xorl:
            launch("xorl", args, args.xorl_dir)
        if not args.skip_tinker:
            launch("tinker", args, args.tinker_dir)
    comparison = build_comparison(
        load_run(args.xorl_dir),
        load_run(args.tinker_dir),
        last_n=args.last_n,
    )
    print_table(comparison)
    output = args.output or (args.xorl_dir / "comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(f"[compare] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
