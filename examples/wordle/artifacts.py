"""Append-safe local artifacts and optimizer-aware resume validation."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .config import WandbConfig


def redact_url(value: str) -> str:
    if not value:
        return ""
    has_scheme = "://" in value
    parts = urlsplit(value if has_scheme else f"//{value}")
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    if not has_scheme:
        return f"{host}{parts.path}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path: Path, value: object, *, exclusive: bool = False) -> None:
    """Create or atomically replace one durable JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        return
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ResumeState:
    step: int
    checkpoint_path: str
    model: str
    model_id: str
    session_id: str
    backend: str = ""


def _resume_config(value: dict) -> dict:
    """Remove only operational fields that legitimately change on resume."""

    normalized = json.loads(json.dumps(value))
    normalized.pop("run_metadata", None)
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("output_dir", None)
        artifacts.pop("resume_from", None)
    normalized.pop("wandb", None)
    return normalized


class ArtifactStore:
    def __init__(self, output_dir: str | Path):
        self.root = Path(output_dir).resolve()
        self.steps = self.root / "steps"
        self.pipeline = self.root / "pipeline"

    def initialize(self, *, run_config: dict, source_info: dict, resume: bool) -> None:
        if resume:
            if not (self.root / "run_config.json").is_file():
                raise ValueError(
                    f"resume directory has no run_config.json: {self.root}"
                )
            stored_config = json.loads(
                (self.root / "run_config.json").read_text(encoding="utf-8")
            )
            if _resume_config(stored_config) != _resume_config(run_config):
                raise ValueError(
                    "resume configuration differs from the immutable original run"
                )
            stored_source_path = self.root / "source_info.json"
            if not stored_source_path.is_file():
                raise ValueError(
                    f"resume directory has no source_info.json: {self.root}"
                )
            if source_info:
                stored_source = json.loads(
                    stored_source_path.read_text(encoding="utf-8")
                )
                for key in ("commit", "dataset_hashes", "implementation_hashes"):
                    if (
                        key in source_info
                        and stored_source.get(key) != source_info[key]
                    ):
                        raise ValueError(f"resume source identity differs for {key}")
            return
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.root}")
        self.steps.mkdir(parents=True, exist_ok=True)
        _json(self.root / "run_config.json", run_config, exclusive=True)
        _json(self.root / "source_info.json", source_info, exclusive=True)

    def append_metrics(self, metrics: dict) -> None:
        _append(self.root / "metrics.jsonl", metrics)

    def append_turn(self, record: dict) -> None:
        _append(self.root / "turns.jsonl", record)

    def write_step(self, step: int, record: dict) -> None:
        self.steps.mkdir(parents=True, exist_ok=True)
        _json(self.steps / f"step-{step:08d}.json", record, exclusive=True)

    def write_pipeline_rollout(self, step: int, record: dict) -> None:
        """Durably preserve a queued rollout before its predecessor commits."""

        path = self.pipeline / f"step-{step:08d}.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise ValueError(
                    f"queued pipeline rollout differs from its durable copy: step={step}"
                )
            return
        _json(path, record, exclusive=True)

    def load_pipeline_rollout(self, step: int) -> dict | None:
        path = self.pipeline / f"step-{step:08d}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"queued pipeline rollout is not an object: step={step}")
        return value

    def discard_pipeline_rollout(self, step: int) -> None:
        """Remove a queued rollout only after that step's commit marker is durable."""

        path = self.pipeline / f"step-{step:08d}.json"
        if not path.exists():
            return
        path.unlink()
        _fsync_directory(self.pipeline)

    def write_preoptimizer_gate(self, step: int, record: dict) -> Path:
        """Write an immutable, attempt-specific pre-optimizer receipt."""
        self.steps.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            path = self.steps / (
                f"step-{step:08d}-preoptimizer-attempt-{attempt:04d}.json"
            )
            if not path.exists():
                break
            attempt += 1
        _json(
            path,
            {**record, "attempt": attempt},
            exclusive=True,
        )
        return path

    def append_checkpoint(self, record: dict) -> None:
        _append(self.root / "checkpoints.jsonl", record)

    def load_step_records(self, *, before_step: int | None = None) -> list[dict]:
        records = []
        if not self.steps.is_dir():
            return records
        for path in sorted(self.steps.glob("step-*.json")):
            if not path.name[5:-5].isdigit():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if before_step is None or int(record["step"]) < before_step:
                records.append(record)
        return records

    def write_terminal_audit(self, record: dict) -> None:
        _json(self.root / "terminal_audit.json", record)

    def load_resume(self, *, model: str, model_id: str) -> ResumeState:
        checkpoint_file = self.root / "checkpoints.jsonl"
        if not checkpoint_file.is_file():
            raise ValueError("resume requires checkpoints.jsonl")
        records = [
            json.loads(line)
            for line in checkpoint_file.read_text().splitlines()
            if line
        ]
        if not records:
            raise ValueError("resume checkpoint ledger is empty")
        required = {"step", "path", "model", "model_id", "session_id", "optimizer"}
        run_config = json.loads(
            (self.root / "run_config.json").read_text(encoding="utf-8")
        )
        run_session = (run_config.get("run_metadata") or {}).get("session_id")
        if not run_session:
            raise ValueError("resume run configuration has no session identity")

        committed = self.load_step_records()
        latest_committed = max(
            (int(record.get("step", -1)) for record in committed), default=-1
        )
        selected: dict | None = None
        for record in reversed(records):
            if not required.issubset(record):
                raise ValueError("resume checkpoint metadata is incomplete")
            if record["model"] != model or record["model_id"] != model_id:
                raise ValueError("resume checkpoint model/session identity mismatch")
            if record["session_id"] != run_session:
                raise ValueError("resume checkpoint session identity mismatch")
            if not record["optimizer"]:
                raise ValueError("resume checkpoint does not declare optimizer state")
            step = int(record["step"])
            step_path = self.steps / f"step-{step:08d}.json"
            if not step_path.is_file():
                # Expected crash window: the checkpoint ledger is durable but
                # the immutable completed-step commit marker is not.
                continue
            step_record = json.loads(step_path.read_text(encoding="utf-8"))
            if int(step_record.get("step", -1)) != step:
                raise ValueError(
                    "resume declared step does not match its step artifact"
                )
            step_checkpoint = step_record.get("checkpoint")
            if step_checkpoint is not None and step_checkpoint != record["path"]:
                raise ValueError(
                    "resume checkpoint path differs from its step artifact"
                )
            selected = record
            break
        if selected is None:
            raise ValueError("resume checkpoint step has no completed step artifact")
        if latest_committed != int(selected["step"]):
            raise ValueError(
                "exact resume requires a checkpoint for the latest completed step: "
                f"latest_completed={latest_committed} checkpoint={selected['step']}"
            )
        return ResumeState(
            step=int(selected["step"]),
            checkpoint_path=str(selected["path"]),
            model=str(selected["model"]),
            model_id=str(selected["model_id"]),
            session_id=str(selected["session_id"]),
            backend=str(selected.get("backend", "")),
        )


class WandbSink:
    """Optional common W&B schema; local artifacts remain authoritative."""

    def __init__(self, config: WandbConfig, *, run_config: dict):
        self.run = None
        self.wandb = None
        self.log_sample_count = config.log_samples
        if not config.project:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("W&B logging requires xorl-client[examples]") from exc
        self.wandb = wandb
        self.run = wandb.init(
            project=config.project,
            entity=config.entity or None,
            name=config.name or None,
            group=config.group or None,
            tags=config.tags or None,
            job_type="train",
            config=run_config,
        )

    def log_step(self, record: dict) -> None:
        if self.run is None:
            return
        step = int(record["step"])
        scalar: dict[str, int | float] = {}

        def collect(prefix: str, value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    collect(f"{prefix}/{key}" if prefix else str(key), child)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                key = prefix if prefix.startswith("hybrid/") else f"train/{prefix}"
                scalar[key] = value

        collect("", record)
        self.run.log(scalar, step=step)

    def log_samples(self, *, step: int, trajectories: list[object]) -> None:
        if self.run is None or self.wandb is None or self.log_sample_count <= 0:
            return
        table = self.wandb.Table(
            columns=[
                "target",
                "rollout_id",
                "history",
                "solved",
                "reward",
                "advantage",
                "last_text",
            ]
        )
        for trajectory in trajectories[: self.log_sample_count]:
            turns = list(getattr(trajectory, "turns", []))
            table.add_data(
                str(getattr(trajectory, "target", "")),
                int(getattr(trajectory, "rollout_id", 0)),
                json.dumps(getattr(trajectory, "history", [])),
                bool(getattr(trajectory, "solved", False)),
                float(getattr(trajectory, "reward", {}).get("reward", 0.0)),
                float(getattr(trajectory, "advantage", 0.0)),
                str(getattr(turns[-1], "text", ""))[:1000] if turns else "",
            )
        self.run.log({"train/samples": table}, step=step)

    def finish(self, audit: dict) -> None:
        if self.run is None:
            return
        for key, value in audit.items():
            if isinstance(value, (str, int, float, bool)):
                self.run.summary[f"terminal/{key}"] = value
        self.run.finish()


def git_source_info(root: str | Path) -> dict:
    root = str(Path(root).resolve())

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", root, *args], check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "repository_root": run("rev-parse", "--show-toplevel"),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def terminal_audit(
    *, records: list[dict], expected_steps: int, checkpoint_every: int
) -> dict:
    completed = {
        int(record["step"])
        for record in records
        if record.get("optimizer_complete")
        or (
            record.get("optimizer_skipped")
            and record.get("optimizer_skip_reason") == "no_retained_datums"
        )
    }
    expected = set(range(1, expected_steps + 1))
    missing = sorted(expected - completed)
    failed = [
        int(record["step"])
        for record in records
        if (
            not record.get("final_sync")
            or not record.get("finite_loss")
            or not record.get("structural_alignment_valid", True)
            or not record.get("correctness_gates_passed", True)
        )
    ]
    required_checkpoints = {
        step
        for step in expected
        if step % checkpoint_every == 0 or step == expected_steps
    }
    checkpointed = {
        int(record["step"]) for record in records if record.get("checkpoint")
    }
    result = {
        "success": not missing and not failed and required_checkpoints <= checkpointed,
        "expected_steps": expected_steps,
        "completed_steps": sorted(completed),
        "missing_steps": missing,
        "failed_steps": failed,
        "missing_checkpoints": sorted(required_checkpoints - checkpointed),
        "final_sync": bool(records and records[-1].get("final_sync")),
    }
    return result
