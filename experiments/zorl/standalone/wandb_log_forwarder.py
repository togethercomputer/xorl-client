"""Forward standalone ZORL text logs to Weights & Biases.

This is intentionally decoupled from ``zorl_client.py`` so it can attach to an
already-running Kubernetes job, backfill completed steps, and keep following the
log without restarting training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any


_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s,\]]+)")
_COLD_RE = re.compile(
    r"cold:\s+reward_mean=([+-]?[0-9.eE-]+)\s+exact_rate=([+-]?[0-9.eE-]+)\s+"
    r"exact_count=([+-]?[0-9.eE-]+)/(\d+)"
)
_STEP_RE = re.compile(r"step\s+(\d+)/(\d+):\s+(.*)")
_TRAIN_BATCH_RE = re.compile(r"train_batch\s+step=(\d+):\s+size=(\d+)\s+pool=(\d+)\s+preview=(.*)")


def _coerce(value: str) -> Any:
    value = value.strip()
    if value.endswith("s") and len(value) > 1:
        maybe_number = value[:-1]
        try:
            return float(maybe_number)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        as_float = float(value)
    except ValueError:
        return value
    if not math.isfinite(as_float):
        return value
    if as_float.is_integer() and not any(ch in value.lower() for ch in (".", "e")):
        return int(as_float)
    return as_float


def _parse_kv_fragment(text: str) -> dict[str, Any]:
    return {key: _coerce(value) for key, value in _KV_RE.findall(text)}


def _numeric(mapping: dict[str, Any], key: str) -> float | int | None:
    value = mapping.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"logged_cold": False, "logged_steps": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"logged_cold": False, "logged_steps": []}
    state.setdefault("logged_cold", False)
    state.setdefault("logged_steps", [])
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")


def _run_config(lines: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line in lines:
        for key in (
            "run_id",
            "run_dir",
            "task",
            "infer_url",
            "reward_infer_url",
            "population_sharding",
            "model_path",
            "adapter_dir",
            "seed",
        ):
            prefix = f"{key}="
            if line.startswith(prefix):
                config[key] = line.split("=", 1)[1].strip()
        if line.startswith("recipe:"):
            config.update({f"recipe/{key}": value for key, value in _parse_kv_fragment(line).items()})
        elif line.startswith("data:"):
            config.update({f"data/{key}": value for key, value in _parse_kv_fragment(line).items()})
        elif line.startswith("[init] hyperparams:"):
            config.update({f"init/{key}": value for key, value in _parse_kv_fragment(line).items()})
    return config


def _cold_metrics(line: str) -> dict[str, Any] | None:
    match = _COLD_RE.search(line)
    if not match:
        return None
    return {
        "eval/cold_reward_mean": float(match.group(1)),
        "eval/cold_exact_rate": float(match.group(2)),
        "eval/cold_exact_count": float(match.group(3)),
        "eval/total": int(match.group(4)),
    }


def _step_metrics(line: str, pending_batches: dict[int, dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    match = _STEP_RE.search(line)
    if not match:
        return None
    step = int(match.group(1))
    planned = int(match.group(2))
    values = _parse_kv_fragment(match.group(3))
    metrics: dict[str, Any] = {
        "progress/step": step,
        "progress/planned_steps": planned,
    }
    metrics.update(pending_batches.pop(step, {}))

    key_map = {
        "reward_mean": "train/reward_mean",
        "best_cand": "train/best_candidate_reward",
        "update_norm": "update/update_norm",
        "pair_delta_mean": "update/pair_delta_mean",
        "pair_delta_std": "update/pair_delta_std",
        "unclipped_update_norm": "update/unclipped_update_norm",
        "grad_norm": "update/grad_norm",
        "update_clip_scale": "update/clip_scale",
        "zero_score_pairs": "population/zero_score_pairs",
        "dropped_pairs": "population/dropped_pairs",
        "used_pairs": "population/used_pairs",
        "t_score": "timing/score_sec",
        "t_apply": "timing/apply_sec",
        "probe_reward": "eval/probe_reward",
        "exact_rate": "eval/exact_rate",
        "teacher_forced_finite_rate": "eval/teacher_forced_finite_rate",
        "teacher_forced_logprob": "eval/teacher_forced_logprob",
        "teacher_forced_prob": "eval/teacher_forced_prob",
        "teacher_forced_token_count": "eval/teacher_forced_token_count",
    }
    for source_key, metric_key in key_map.items():
        value = _numeric(values, source_key)
        if value is not None:
            metrics[metric_key] = value

    score_sec = _numeric(values, "t_score")
    apply_sec = _numeric(values, "t_apply")
    if score_sec is not None and apply_sec is not None:
        metrics["timing/step_sec"] = float(score_sec) + float(apply_sec)
    return step, metrics


def _parse_log(path: Path, state: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]], bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    logged_steps = {int(step) for step in state.get("logged_steps", [])}
    pending_batches: dict[int, dict[str, Any]] = {}
    cold: dict[str, Any] = {}
    step_events: list[tuple[int, dict[str, Any]]] = []
    done = False

    for line in lines:
        batch = _TRAIN_BATCH_RE.search(line)
        if batch:
            step = int(batch.group(1))
            pending_batches[step] = {
                "data/train_batch_size": int(batch.group(2)),
                "data/train_pool_size": int(batch.group(3)),
                "data/train_batch_preview": batch.group(4).strip(),
            }
            continue

        if not state.get("logged_cold"):
            cold_metrics = _cold_metrics(line)
            if cold_metrics is not None:
                cold = cold_metrics
                continue

        parsed_step = _step_metrics(line, pending_batches)
        if parsed_step is not None:
            step, metrics = parsed_step
            if step not in logged_steps:
                step_events.append((step, metrics))
            continue

        if line.startswith("[done]"):
            done = True

    return cold, step_events, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT") or "zorl")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--job-type", default=os.environ.get("WANDB_JOB_TYPE") or "zorl-standalone")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--state-file", type=Path, default=None)
    args = parser.parse_args()

    import wandb  # noqa: PLC0415

    if not args.log_file.exists():
        raise SystemExit(f"log file does not exist: {args.log_file}")
    state_file = args.state_file or args.log_file.with_suffix(args.log_file.suffix + ".wandb-state.json")
    state = _load_state(state_file)
    config = _run_config(args.log_file.read_text(encoding="utf-8", errors="replace").splitlines())
    name = args.name or str(config.get("run_id") or args.log_file.parent.name)
    run_id = args.run_id or re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:128]
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=name,
        id=run_id,
        resume="allow",
        group=args.group,
        job_type=args.job_type,
        config=config,
    )

    while True:
        cold, step_events, done = _parse_log(args.log_file, state)
        if cold and not state.get("logged_cold"):
            run.log(cold, step=0)
            run.summary.update(cold)
            state["logged_cold"] = True

        logged_steps = {int(step) for step in state.get("logged_steps", [])}
        for step, metrics in step_events:
            run.log(metrics, step=step)
            run.summary.update(metrics)
            logged_steps.add(step)
        state["logged_steps"] = sorted(logged_steps)
        _save_state(state_file, state)

        if done or not args.follow:
            break
        time.sleep(max(args.poll_seconds, 1.0))

    run.finish()


if __name__ == "__main__":
    main()
