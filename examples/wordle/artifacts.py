"""Append-safe local artifacts and optimizer-aware resume validation."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def redact_url(value: str) -> str:
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _json(path: Path, value: object, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append(path: Path, value: object) -> None:
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


class ArtifactStore:
    def __init__(self, output_dir: str | Path):
        self.root = Path(output_dir).resolve()
        self.steps = self.root / "steps"

    def initialize(self, *, run_config: dict, source_info: dict, resume: bool) -> None:
        if resume:
            if not (self.root / "run_config.json").is_file():
                raise ValueError(
                    f"resume directory has no run_config.json: {self.root}"
                )
            return
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.root}")
        self.steps.mkdir(parents=True, exist_ok=True)
        _json(self.root / "run_config.json", run_config, exclusive=True)
        _json(self.root / "source_info.json", source_info, exclusive=True)

    def append_metrics(self, metrics: dict) -> None:
        _append(self.root / "metrics.jsonl", metrics)

    def write_step(self, step: int, record: dict) -> None:
        self.steps.mkdir(parents=True, exist_ok=True)
        _json(self.steps / f"step-{step:08d}.json", record, exclusive=True)

    def append_checkpoint(self, record: dict) -> None:
        _append(self.root / "checkpoints.jsonl", record)

    def load_step_records(self, *, before_step: int | None = None) -> list[dict]:
        records = []
        if not self.steps.is_dir():
            return records
        for path in sorted(self.steps.glob("step-*.json")):
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
        last = records[-1]
        required = {"step", "path", "model", "model_id", "session_id", "optimizer"}
        if not required.issubset(last):
            raise ValueError("resume checkpoint metadata is incomplete")
        if last["model"] != model or last["model_id"] != model_id:
            raise ValueError("resume checkpoint model/session identity mismatch")
        run_config = json.loads(
            (self.root / "run_config.json").read_text(encoding="utf-8")
        )
        run_session = (run_config.get("run_metadata") or {}).get("session_id")
        if not run_session or last["session_id"] != run_session:
            raise ValueError("resume checkpoint session identity mismatch")
        if not last["optimizer"]:
            raise ValueError("resume checkpoint does not declare optimizer state")
        step_path = self.steps / f"step-{int(last['step']):08d}.json"
        if not step_path.is_file():
            raise ValueError("resume checkpoint step has no completed step artifact")
        step_record = json.loads(step_path.read_text())
        if int(step_record.get("step", -1)) != int(last["step"]):
            raise ValueError("resume declared step does not match its step artifact")
        return ResumeState(
            step=int(last["step"]),
            checkpoint_path=str(last["path"]),
            model=str(last["model"]),
            model_id=str(last["model_id"]),
            session_id=str(last["session_id"]),
        )


def git_source_info(root: str | Path) -> dict:
    root = str(Path(root).resolve())

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", root, *args], check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "repository_root": root,
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def terminal_audit(
    *, records: list[dict], expected_steps: int, checkpoint_every: int
) -> dict:
    completed = {
        int(record["step"]) for record in records if record.get("optimizer_complete")
    }
    expected = set(range(1, expected_steps + 1))
    missing = sorted(expected - completed)
    failed = [
        int(record["step"])
        for record in records
        if not record.get("final_sync")
        or not record.get("finite_loss")
        or not record.get("finite_gradient")
        or not record.get("correctness_gates_passed")
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
