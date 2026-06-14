#!/usr/bin/env python3
"""Small autoresearch controller for prefill-time-compute OPSD candidates."""

from __future__ import annotations

import argparse
import calendar
import glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
OPD_PROFILE_DIR = ROOT.parent
IDEAS_PATH = ROOT / "ideas.yaml"
RUN_LOG = ROOT / "runs.jsonl"
SCORECARD_DIR = ROOT / "scorecards"
GENERATOR = OPD_PROFILE_DIR / "k8s" / "q36_35b_reprogrammable_slots.py"
MONITOR = OPD_PROFILE_DIR / "monitor_live_opd.py"
STACK = "er-opd-q36-35b-slots"
NAMESPACE = "apanda"
RESULT_ROOT = Path(
    "/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/"
    f"{STACK}"
)
CONTROL_ROOT = Path(f"/shared/opd-control/{STACK}")
TRAINER_ROLES = ["trainer-head", *[f"trainer-worker-{i}" for i in range(1, 8)]]
INFERENCE_ROLES = ["dispatch", "sglang-0", "teacher-sglang-0", "teacher-sglang-1", "teacher-smg"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_utc(raw: str) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ideas": []}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    data.setdefault("ideas", [])
    if not isinstance(data["ideas"], list):
        raise SystemExit(f"{path}: ideas must be a list")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def append_event(event: dict[str, Any]) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_utc": utc_now(), **event}
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def existing_control_score_keys() -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not RUN_LOG.exists():
        return keys
    with RUN_LOG.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") not in {"score", "autopilot_score"}:
                continue
            score = event.get("score")
            if not isinstance(score, dict):
                continue
            verdict = str(score.get("verdict", ""))
            if not verdict or verdict == "incomplete":
                continue
            metrics = score.get("metrics")
            if not isinstance(metrics, dict):
                continue
            keys.add(
                (
                    str(event.get("idea_id")),
                    str(score.get("profile")),
                    str(metrics.get("step", "na")),
                    verdict,
                )
            )
    return keys


def idea_by_id(data: dict[str, Any], idea_id: str) -> dict[str, Any]:
    for idea in data["ideas"]:
        if str(idea.get("id")) == idea_id:
            return idea
    raise SystemExit(f"unknown idea id {idea_id!r}")


def candidate_path_for(idea: dict[str, Any]) -> Path:
    raw = idea.get("candidate")
    if not raw:
        raise SystemExit(f"idea {idea.get('id')} has no candidate")
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def load_candidate(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: candidate must be a YAML mapping")
    return data


def dependencies_satisfied(data: dict[str, Any], idea: dict[str, Any]) -> bool:
    complete_states = {"weak_signal", "promote_retest", "strong_signal", "complete", "rejected"}
    by_id = {str(item.get("id")): item for item in data["ideas"]}
    requires_statuses = idea.get("requires_statuses") or {}
    if not isinstance(requires_statuses, dict):
        raise SystemExit(f"idea {idea.get('id')} requires_statuses must be a mapping")
    for dep in idea.get("requires", []) or []:
        dep_idea = by_id.get(str(dep))
        if dep_idea is None:
            return False
        dep_status = str(dep_idea.get("status"))
        allowed_statuses = requires_statuses.get(str(dep))
        if allowed_statuses is not None:
            if isinstance(allowed_statuses, str):
                allowed = {item.strip() for item in allowed_statuses.split(",") if item.strip()}
            elif isinstance(allowed_statuses, list):
                allowed = {str(item).strip() for item in allowed_statuses if str(item).strip()}
            else:
                raise SystemExit(
                    f"idea {idea.get('id')} requires_statuses[{dep!r}] must be a list or comma-separated string"
                )
            if dep_status not in allowed:
                return False
        elif dep_status not in complete_states:
            return False
    return True


def next_idea(data: dict[str, Any]) -> dict[str, Any] | None:
    if any(str(idea.get("status", "")).strip().lower() == "launched" for idea in data["ideas"]):
        return None
    runnable = [
        idea
        for idea in data["ideas"]
        if str(idea.get("status", "queued")) in {"queued", "retest"}
        and dependencies_satisfied(data, idea)
    ]
    if not runnable:
        return None
    return sorted(runnable, key=lambda item: (-int(item.get("priority", 0)), str(item.get("id"))))[0]


def launched_idea(data: dict[str, Any]) -> dict[str, Any] | None:
    launched = [idea for idea in data["ideas"] if str(idea.get("status", "")).strip().lower() == "launched"]
    if len(launched) > 1:
        ids = ", ".join(str(idea.get("id")) for idea in launched)
        raise SystemExit(f"multiple launched ideas: {ids}")
    return launched[0] if launched else None


def launch_stamp(idea: dict[str, Any]) -> str | None:
    raw = str(idea.get("last_launched_utc", "")).strip()
    if not raw:
        return None
    parsed_epoch = parse_utc(raw)
    if parsed_epoch is None:
        return None
    parsed = time.gmtime(parsed_epoch)
    return time.strftime("%Y%m%dT%H%M%SZ", parsed)


def launch_age_s(idea: dict[str, Any]) -> float | None:
    raw = str(idea.get("last_launched_utc", "")).strip()
    launched_at = parse_utc(raw) if raw else None
    if launched_at is None:
        return None
    return max(0.0, time.time() - launched_at)


def run_dir_epoch(path: Path) -> float | None:
    run_stamp = path.parent.name.split("-", 1)[0]
    return parse_utc(f"{run_stamp[:4]}-{run_stamp[4:6]}-{run_stamp[6:8]}T{run_stamp[9:11]}:{run_stamp[11:13]}:{run_stamp[13:15]}Z")


def profile_for_idea(idea: dict[str, Any], requested: str) -> Path:
    if requested != "latest":
        return resolve_profile(requested)
    idea_id = str(idea.get("id", "")).strip()
    launched_at = parse_utc(str(idea.get("last_launched_utc", "")).strip())
    if idea_id:
        matches = [Path(p) for p in glob.glob(str(RESULT_ROOT / f"*config{idea_id}-*" / "opd_profile.jsonl"))]
        if launched_at is not None:
            matches = [
                path
                for path in matches
                if (run_epoch := run_dir_epoch(path)) is not None and run_epoch >= launched_at - 60.0
            ]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
        if launched_at is not None:
            raise SystemExit(f"profile not available yet for {idea_id} launched at {idea.get('last_launched_utc')}")
    stamp = launch_stamp(idea)
    if stamp:
        matches = [Path(p) for p in glob.glob(str(RESULT_ROOT / f"{stamp}*" / "opd_profile.jsonl"))]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    return resolve_profile("latest")


def role_statuses() -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    statuses: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            statuses[parts[0]] = parts[1] if len(parts) > 1 else ""
    if result.returncode != 0:
        statuses["_error"] = (result.stderr or result.stdout or f"status rc={result.returncode}").strip()
    return statuses


def k8s_pod_statuses() -> dict[str, str]:
    result = subprocess.run(
        ["kubectl", "-n", NAMESPACE, "get", "pods", "-l", f"stack={STACK}", "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {"_error": (result.stderr or result.stdout or f"kubectl rc={result.returncode}").strip()}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"_error": f"kubectl returned invalid JSON: {exc}"}

    statuses: dict[str, str] = {}
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        role = labels.get("slot-role")
        if not role:
            continue
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        phase = str(status.get("phase") or "Unknown")
        if metadata.get("deletionTimestamp"):
            phase = "Terminating"
        statuses[str(role)] = phase
    return statuses


def trainer_health_failures(idea: dict[str, Any], *, launch_grace_s: float) -> list[str]:
    age = launch_age_s(idea)
    statuses = role_statuses()
    if "_error" in statuses:
        return [f"status_command_failed={statuses['_error']}"]

    running_roles = [role for role in TRAINER_ROLES if "running_since=" in statuses.get(role, "")]
    if age is not None and age < launch_grace_s and not running_roles:
        return []

    failures: list[str] = []
    required_roles = running_roles if age is not None and age < launch_grace_s else TRAINER_ROLES
    for role in required_roles:
        status = statuses.get(role, "")
        if "running_since=" not in status:
            failures.append(f"{role}:{status or 'missing'}")

    pod_statuses = k8s_pod_statuses()
    if "_error" in pod_statuses:
        failures.append(f"k8s_status_failed={pod_statuses['_error']}")
    else:
        for role in required_roles:
            phase = pod_statuses.get(role)
            if phase != "Running":
                failures.append(f"{role}:pod_{phase or 'missing'}")
    return failures


def wait_for_roles_stopped(roles: list[str], *, timeout_s: float = 240.0, poll_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    last_running: list[str] = []
    while time.time() < deadline:
        statuses = role_statuses()
        if "_error" in statuses:
            raise SystemExit(f"status command failed while waiting for slot stop: {statuses['_error']}")
        last_running = [
            role
            for role in roles
            if "running_since=" in statuses.get(role, "")
        ]
        if not last_running:
            return
        time.sleep(poll_s)
    raise SystemExit(f"control slots did not stop within {timeout_s:.0f}s: {', '.join(last_running)}")


def wait_for_stop_files_consumed(roles: list[str], *, timeout_s: float = 30.0, poll_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    pending: list[str] = []
    while time.time() < deadline:
        pending = [role for role in roles if (CONTROL_ROOT / role / "stop").exists()]
        if not pending:
            return
        time.sleep(poll_s)
    raise SystemExit(f"control stop files were not consumed within {timeout_s:.0f}s: {', '.join(pending)}")


def wait_for_trainer_slots_stopped(*, timeout_s: float = 240.0, poll_s: float = 2.0) -> None:
    wait_for_roles_stopped(TRAINER_ROLES, timeout_s=timeout_s, poll_s=poll_s)


def resolve_profile(arg: str) -> Path:
    if arg == "latest":
        profiles = [Path(p) for p in glob.glob(str(RESULT_ROOT / "*" / "opd_profile.jsonl"))]
        if not profiles:
            raise SystemExit(f"no profiles under {RESULT_ROOT}")
        return max(profiles, key=lambda path: path.stat().st_mtime)
    path = Path(arg)
    if path.is_dir():
        path = path / "opd_profile.jsonl"
    if not path.exists():
        raise SystemExit(f"profile not found: {path}")
    return path


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def buffer_delta_z(row: dict[str, Any]) -> float:
    if "eval/buffer_delta_z" in row:
        return f(row, "eval/buffer_delta_z")
    pause = f(row, "eval/acc_pause")
    nopause = f(row, "eval/acc_nopause")
    pause_n = max(1.0, f(row, "eval/control_scored_pause", f(row, "eval/control_n")))
    nopause_n = max(1.0, f(row, "eval/control_scored_nopause", f(row, "eval/control_n")))
    se = math.sqrt(
        max(pause * (1.0 - pause), 0.0) / pause_n
        + max(nopause * (1.0 - nopause), 0.0) / nopause_n
    )
    return (pause - nopause) / se if se > 0 else 0.0


def latest_control(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    controls = [row for row in rows if "eval/buffer_delta" in row or "eval/acc_pause" in row]
    return controls[-1] if controls else None


def failure_frac_max(row: dict[str, Any], *, include_corrupt: bool = True, include_answer_logprob: bool = True) -> float:
    keys = [
        "eval/pause_request_failure_frac",
        "eval/nopause_request_failure_frac",
        "eval/generated_memory_request_failure_frac",
        "eval/generated_memory_length_failure_frac",
    ]
    if include_corrupt:
        keys.extend(
            [
                "eval/corrupt_pause_request_failure_frac",
                "eval/control_request_failure_frac_max",
            ]
        )
    if include_answer_logprob:
        keys.append("eval/answer_logprob_request_failure_frac")
    return max(f(row, key) for key in keys)


def cap_hit_max(row: dict[str, Any], *, include_corrupt: bool = True) -> float:
    keys = [
        "eval/pause_cap_hit_frac",
        "eval/nopause_cap_hit_frac",
    ]
    if include_corrupt:
        keys.extend(
            [
                "eval/corrupt_pause_cap_hit_frac",
                "eval/control_cap_hit_frac_max",
            ]
        )
    return max(f(row, key) for key in keys)


def infra_failures(
    row: dict[str, Any], *, include_corrupt: bool = True, include_answer_logprob: bool = True
) -> list[str]:
    failures: list[str] = []
    if row.get("sync_success") is False or str(row.get("sync_success")).lower() == "false":
        failures.append("sync_success=false")
    if f(row, "sync_endpoint_failure_count") > 0:
        failures.append(f"sync_endpoint_failure_count={f(row, 'sync_endpoint_failure_count'):.0f}")
    request_failure_frac = failure_frac_max(
        row, include_corrupt=include_corrupt, include_answer_logprob=include_answer_logprob
    )
    cap_hit_frac = cap_hit_max(row, include_corrupt=include_corrupt)
    if request_failure_frac > 0.0:
        failures.append(f"request_failure_frac_max={request_failure_frac:.4f}")
    if cap_hit_frac > 0.50:
        failures.append(f"cap_hit_frac_max={cap_hit_frac:.4f}")
    if "sampler_quiesce_success" in row and f(row, "sampler_quiesce_success") < 1.0:
        failures.append("sampler_quiesce_success<1")
    if f(row, "sampler_quiesce_new_outstanding") > 0.0:
        failures.append("sampler_quiesce_new_outstanding>0")
    if f(row, "sampler_quiesce_connections_active") > 0.0:
        failures.append("sampler_quiesce_connections_active>0")
    if f(row, "sampler_quiesce_inflight_request_age_count") > 0.0:
        failures.append("sampler_quiesce_inflight_request_age_count>0")
    worker_count = f(row, "sampler_worker_count")
    active_count = f(row, "sampler_worker_active_count")
    if worker_count >= 2.0 and active_count < 2.0:
        failures.append(f"sampler_worker_active_count={active_count:.0f}<2")
    return failures


def score_profile(profile: Path, *, min_control_n: float = 900.0, score_mode: str = "causal_control") -> dict[str, Any]:
    rows = load_rows(profile)
    control = latest_control(rows)
    if control is None:
        return {
            "profile": str(profile),
            "verdict": "incomplete",
            "reason": "no control rows",
            "rows": len(rows),
        }

    control_n = f(control, "eval/control_n")
    score_mode = score_mode.strip().lower().replace("-", "_")
    pause_vs_nopause_mode = score_mode in {"pause_vs_nopause", "prefill_performance"}
    metrics = {
        "step": control.get("step"),
        "score_mode": score_mode,
        "control_n": control_n,
        "acc_pause": f(control, "eval/acc_pause"),
        "acc_nopause": f(control, "eval/acc_nopause"),
        "acc_corrupt": f(control, "eval/acc_corrupt_pause"),
        "delta": f(control, "eval/buffer_delta"),
        "delta_z": buffer_delta_z(control),
        "vs_corrupt_delta": f(control, "eval/buffer_vs_corrupt_delta"),
        "vs_corrupt_z": f(control, "eval/buffer_vs_corrupt_delta_z"),
        "answer_logprob_margin": f(control, "eval/answer_logprob_margin"),
        "answer_logprob_z": f(control, "eval/answer_logprob_margin_z"),
        "answer_select_delta": f(control, "eval/answer_logprob_select_delta"),
        "answer_select_z": f(control, "eval/answer_logprob_select_delta_z"),
        "failure_frac_max": failure_frac_max(
            control,
            include_corrupt=not pause_vs_nopause_mode,
            include_answer_logprob=not pause_vs_nopause_mode,
        ),
        "cap_hit_frac_max": cap_hit_max(control, include_corrupt=not pause_vs_nopause_mode),
    }
    failures = infra_failures(
        control,
        include_corrupt=not pause_vs_nopause_mode,
        include_answer_logprob=not pause_vs_nopause_mode,
    )
    if failures:
        verdict = "infra_invalid"
        reason = "; ".join(failures)
    elif control_n < min_control_n:
        verdict = "incomplete"
        reason = f"eval/control_n={control_n:.0f} < {min_control_n:.0f}"
    elif pause_vs_nopause_mode:
        logprob_note = ""
        if metrics["answer_logprob_margin"] > 0.0 and metrics["answer_logprob_z"] >= 2.0:
            logprob_note = " with positive answer-logprob support"
        elif "eval/answer_logprob_margin" in control:
            logprob_note = " without positive answer-logprob support"
        if metrics["delta"] >= 0.06 and metrics["delta_z"] >= 3.0:
            verdict = "strong_ptc_signal"
            reason = f"pause-vs-nopause accuracy gate passed{logprob_note}"
        elif metrics["delta"] >= 0.04 and metrics["delta_z"] >= 2.0:
            verdict = "promote_retest"
            reason = f"pause-vs-nopause accuracy warrants retest{logprob_note}"
        elif metrics["delta"] > 0.0:
            verdict = "weak_ptc_signal"
            reason = f"pause beats no-pause but below scale gate{logprob_note}"
        else:
            verdict = "science_reject"
            reason = "pause does not beat no-pause accuracy"
    elif (
        metrics["delta"] >= 0.06
        and metrics["delta_z"] >= 3.0
        and metrics["vs_corrupt_delta"] > 0.0
        and metrics["vs_corrupt_z"] >= 2.0
        and metrics["answer_logprob_margin"] > 0.0
        and metrics["answer_logprob_z"] >= 10.0
    ):
        verdict = "strong_ptc_signal"
        reason = "accuracy, corrupt-control, and answer-logprob gates passed"
    elif (
        metrics["delta"] >= 0.04
        and metrics["delta_z"] >= 2.0
        and metrics["answer_logprob_margin"] > 0.0
        and metrics["answer_logprob_z"] >= 10.0
    ):
        verdict = "promote_retest"
        reason = "accuracy delta and answer-logprob support warrant retest"
    elif (
        metrics["delta"] >= 0.0
        and metrics["answer_logprob_margin"] > 0.0
        and metrics["answer_logprob_z"] >= 10.0
    ):
        verdict = "weak_ptc_signal"
        reason = "answer-logprob support with nonnegative accuracy delta"
    elif metrics["delta"] <= 0.0 and (metrics["answer_logprob_margin"] <= 0.0 or metrics["answer_logprob_z"] < 2.0):
        verdict = "science_reject"
        reason = "no positive accuracy or answer-logprob signal"
    else:
        verdict = "inconclusive"
        reason = "completed cleanly but missed both reject and promotion thresholds"

    return {
        "profile": str(profile),
        "run_dir": str(profile.parent),
        "rows": len(rows),
        "control_rows": sum(1 for row in rows if "eval/buffer_delta" in row or "eval/acc_pause" in row),
        "verdict": verdict,
        "reason": reason,
        "metrics": metrics,
    }


def write_scorecard(score: dict[str, Any], *, idea_id: str | None) -> tuple[Path, Path]:
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    stem_id = idea_id or "unknown"
    step = score.get("metrics", {}).get("step", "na")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = f"{stamp}-{stem_id}-step{step}-{score['verdict']}"
    json_path = SCORECARD_DIR / f"{stem}.json"
    md_path = SCORECARD_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
    metrics = score.get("metrics", {})
    lines = [
        f"# {stem_id} scorecard",
        "",
        f"- verdict: `{score['verdict']}`",
        f"- reason: {score['reason']}",
        f"- profile: `{score['profile']}`",
        f"- control_n: `{metrics.get('control_n')}`",
        f"- acc_pause / acc_nopause / acc_corrupt: `{metrics.get('acc_pause')}` / `{metrics.get('acc_nopause')}` / `{metrics.get('acc_corrupt')}`",
        f"- delta / z: `{metrics.get('delta')}` / `{metrics.get('delta_z')}`",
        f"- vs_corrupt_delta / z: `{metrics.get('vs_corrupt_delta')}` / `{metrics.get('vs_corrupt_z')}`",
        f"- answer_logprob_margin / z: `{metrics.get('answer_logprob_margin')}` / `{metrics.get('answer_logprob_z')}`",
        f"- answer_select_delta / z: `{metrics.get('answer_select_delta')}` / `{metrics.get('answer_select_z')}`",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def command_next(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = next_idea(data)
    if idea is None:
        print("no queued ideas")
        return
    print(yaml.safe_dump(idea, sort_keys=False).strip())


def command_launch(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = idea_by_id(data, args.id) if args.id else next_idea(data)
    if idea is None:
        raise SystemExit("no queued ideas")
    candidate_path = candidate_path_for(idea)
    candidate = load_candidate(candidate_path)
    num_steps = args.num_steps or int(candidate.get("default_num_steps", idea.get("default_num_steps", 9)))
    prompts_per_step = args.prompts_per_step or int(
        candidate.get("default_prompts_per_step", idea.get("default_prompts_per_step", 128))
    )
    if args.restart_inference:
        stop_cmd = [
            sys.executable,
            str(GENERATOR),
            "stop-control",
            "--remove-run",
            "--sampler-replicas",
            str(args.sampler_replicas),
            "--sampler-layout",
            args.sampler_layout,
        ]
        write_cmd = [
            sys.executable,
            str(GENERATOR),
            "write-control",
            "--candidate",
            str(candidate_path),
            "--num-steps",
            str(num_steps),
            "--prompts-per-step",
            str(prompts_per_step),
            "--sampler-replicas",
            str(args.sampler_replicas),
            "--sampler-layout",
            args.sampler_layout,
        ]
        wait_roles = [*INFERENCE_ROLES, *TRAINER_ROLES]
    else:
        stop_cmd = [sys.executable, str(GENERATOR), "stop-trainer-control", "--remove-run"]
        write_cmd = [
            sys.executable,
            str(GENERATOR),
            "write-trainer-control",
            "--candidate",
            str(candidate_path),
            "--num-steps",
            str(num_steps),
            "--prompts-per-step",
            str(prompts_per_step),
            "--sampler-replicas",
            str(args.sampler_replicas),
            "--sampler-layout",
            args.sampler_layout,
        ]
        wait_roles = TRAINER_ROLES
    if args.dry_run:
        print(" ".join(stop_cmd))
        print(" ".join(write_cmd))
        return
    subprocess.run(stop_cmd, check=True)
    wait_for_roles_stopped(wait_roles)
    wait_for_stop_files_consumed(wait_roles)
    subprocess.run(write_cmd, check=True)
    idea["status"] = "launched"
    idea["last_launched_utc"] = utc_now()
    idea["last_num_steps"] = num_steps
    idea["last_prompts_per_step"] = prompts_per_step
    save_yaml(IDEAS_PATH, data)
    append_event(
        {
            "event": "launch",
            "idea_id": idea.get("id"),
            "candidate": str(candidate_path),
            "num_steps": num_steps,
            "prompts_per_step": prompts_per_step,
            "sampler_replicas": args.sampler_replicas,
            "sampler_layout": args.sampler_layout,
            "restart_inference": bool(args.restart_inference),
        }
    )


def command_monitor(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(MONITOR), args.profile, "--samples", str(args.samples)]
    if args.json:
        cmd.append("--json")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout.rstrip())
    append_event({"event": "monitor", "profile": args.profile, "output": result.stdout.rstrip()})


def command_score(args: argparse.Namespace) -> None:
    profile = resolve_profile(args.profile)
    min_control_n = args.min_control_n
    score_mode = "causal_control"
    if min_control_n is None and args.idea_id:
        data = load_yaml(IDEAS_PATH)
        idea = idea_by_id(data, args.idea_id)
        candidate = load_candidate(candidate_path_for(idea))
        min_control_n = float(candidate.get("score_min_control_n", 900.0))
        score_mode = str(candidate.get("score_mode", score_mode))
    if min_control_n is None:
        min_control_n = 900.0
    score = score_profile(profile, min_control_n=min_control_n, score_mode=score_mode)
    record_score = score.get("verdict") != "incomplete" or args.record_incomplete
    json_path = None
    md_path = None
    if record_score:
        json_path, md_path = write_scorecard(score, idea_id=args.idea_id)
        append_event({"event": "score", "idea_id": args.idea_id, "score": score, "scorecard": str(json_path)})
    if args.json:
        print(json.dumps(score, indent=2, sort_keys=True))
        return
    print(f"verdict={score['verdict']} reason={score['reason']}")
    if record_score:
        print(f"scorecard_json={json_path}")
        print(f"scorecard_md={md_path}")
    else:
        print("scorecard_recorded=false")


def command_advance(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    idea = idea_by_id(data, args.id)
    profile = resolve_profile(args.profile)
    min_control_n = args.min_control_n
    score_mode = "causal_control"
    if min_control_n is None:
        candidate = load_candidate(candidate_path_for(idea))
        min_control_n = float(candidate.get("score_min_control_n", 900.0))
        score_mode = str(candidate.get("score_mode", score_mode))
    else:
        candidate = load_candidate(candidate_path_for(idea))
        score_mode = str(candidate.get("score_mode", score_mode))
    score = score_profile(profile, min_control_n=min_control_n, score_mode=score_mode)
    json_path, _ = write_scorecard(score, idea_id=args.id)
    status_by_verdict = {
        "strong_ptc_signal": "strong_signal",
        "promote_retest": "promote_retest",
        "weak_ptc_signal": "weak_signal",
        "science_reject": "rejected",
        "infra_invalid": "infra_invalid",
        "incomplete": "incomplete",
        "inconclusive": "inconclusive",
    }
    idea["status"] = status_by_verdict.get(str(score["verdict"]), "inconclusive")
    idea["last_verdict"] = score["verdict"]
    idea["last_reason"] = score["reason"]
    idea["last_profile"] = str(profile)
    idea["last_scorecard"] = str(json_path)
    idea["last_scored_utc"] = utc_now()
    save_yaml(IDEAS_PATH, data)
    append_event({"event": "advance", "idea_id": args.id, "score": score, "status": idea["status"]})
    print(f"{args.id}: {idea['status']} ({score['verdict']})")


def csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def mark_idea(data: dict[str, Any], idea: dict[str, Any], *, status: str, reason: str, profile: str | None = None) -> None:
    idea["status"] = status
    idea["last_verdict"] = status
    idea["last_reason"] = reason
    if profile:
        idea["last_profile"] = profile
    idea["last_scored_utc"] = utc_now()
    save_yaml(IDEAS_PATH, data)


def configured_num_steps(idea: dict[str, Any], candidate: dict[str, Any]) -> int:
    for key in ("last_num_steps", "default_num_steps"):
        raw = idea.get(key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return int(candidate.get("default_num_steps", 9))


def reached_final_control(idea: dict[str, Any], candidate: dict[str, Any], score: dict[str, Any]) -> bool:
    metrics = score.get("metrics")
    if not isinstance(metrics, dict):
        return False
    raw_step = metrics.get("step")
    try:
        step = int(float(raw_step))
    except (TypeError, ValueError):
        return False
    return step >= configured_num_steps(idea, candidate) - 1


def maybe_launch_next_from_queue(
    args: argparse.Namespace,
    *,
    sampler_replicas: int,
    sampler_layout: str,
) -> bool:
    data = load_yaml(IDEAS_PATH)
    if next_idea(data) is None:
        print("autopilot: no queued idea to launch", flush=True)
        append_event({"event": "autopilot_idle", "message": "no queued idea to launch"})
        return False
    command_launch(
        argparse.Namespace(
            id=None,
            dry_run=False,
            restart_inference=False,
            num_steps=None,
            prompts_per_step=None,
            sampler_replicas=sampler_replicas,
            sampler_layout=sampler_layout,
        )
    )
    time.sleep(args.post_launch_sleep_s)
    return True


def command_autopilot(args: argparse.Namespace) -> None:
    launch_verdicts = csv_set(args.launch_next_verdicts)
    stop_only_verdicts = csv_set(args.stop_only_verdicts)
    terminal_verdicts = csv_set(args.terminal_verdicts)
    seen_controls: set[tuple[str, str, str, str]] = existing_control_score_keys()
    deferred_terminals: set[tuple[str, str, str, str]] = set()
    polls = 0
    while args.max_polls <= 0 or polls < args.max_polls:
        polls += 1
        data = load_yaml(IDEAS_PATH)
        idea = launched_idea(data)
        if idea is None:
            idea = next_idea(data)
            if idea is None:
                message = "autopilot: no launched or queued ideas"
                print(message, flush=True)
                append_event({"event": "autopilot_idle", "message": message})
                return
            if not args.launch_next:
                message = f"autopilot: next queued idea is {idea.get('id')}; launch disabled"
                print(message, flush=True)
                append_event({"event": "autopilot_idle", "message": message, "idea_id": idea.get("id")})
                return
            command_launch(
                argparse.Namespace(
                    id=idea.get("id"),
                    dry_run=False,
                    restart_inference=False,
                    num_steps=None,
                    prompts_per_step=None,
                    sampler_replicas=args.sampler_replicas,
                    sampler_layout=args.sampler_layout,
                )
            )
            time.sleep(args.post_launch_sleep_s)
            continue

        idea_id = str(idea.get("id"))
        candidate = load_candidate(candidate_path_for(idea))
        min_control_n = float(candidate.get("score_min_control_n", 900.0))
        score_mode = str(candidate.get("score_mode", "causal_control"))
        try:
            profile = profile_for_idea(idea, args.profile)
            score = score_profile(profile, min_control_n=min_control_n, score_mode=score_mode)
        except SystemExit as exc:
            health_failures = trainer_health_failures(idea, launch_grace_s=args.launch_grace_seconds)
            if health_failures:
                reason = "; ".join(health_failures)
                print(f"autopilot: launch/health failure for {idea_id}: {reason}", flush=True)
                subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
                mark_idea(data, idea, status="infra_invalid", reason=reason)
                append_event({"event": "autopilot_infra_invalid", "idea_id": idea_id, "reason": reason})
                return
            message = f"autopilot: profile/score unavailable for {idea_id}: {exc}"
            print(message, flush=True)
            append_event({"event": "autopilot_wait", "idea_id": idea_id, "message": message})
            time.sleep(args.poll_seconds)
            continue

        metrics = score.get("metrics", {})
        step = str(metrics.get("step", "na"))
        verdict = str(score["verdict"])
        key = (idea_id, str(score["profile"]), step, verdict)
        final_control_reached = reached_final_control(idea, candidate, score)
        if verdict != "incomplete" and key not in seen_controls:
            seen_controls.add(key)
            json_path, md_path = write_scorecard(score, idea_id=idea_id)
            append_event(
                {
                    "event": "autopilot_score",
                    "idea_id": idea_id,
                    "score": score,
                    "scorecard": str(json_path),
                }
            )
            print(
                f"autopilot: {idea_id} step={step} verdict={verdict} "
                f"reason={score['reason']} scorecard={md_path}",
                flush=True,
            )

        if verdict == "incomplete":
            health_failures = trainer_health_failures(idea, launch_grace_s=args.launch_grace_seconds)
            if health_failures:
                reason = "; ".join(health_failures)
                print(f"autopilot: launch/health failure for {idea_id}: {reason}", flush=True)
                subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
                mark_idea(data, idea, status="infra_invalid", reason=reason, profile=str(score["profile"]))
                append_event({"event": "autopilot_infra_invalid", "idea_id": idea_id, "reason": reason})
                return

        if verdict in launch_verdicts:
            print(f"autopilot: stopping rejected run {idea_id} and launching next queued idea", flush=True)
            subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
            command_advance(argparse.Namespace(id=idea_id, profile=str(score["profile"]), min_control_n=None))
            if args.launch_next:
                if maybe_launch_next_from_queue(
                    args,
                    sampler_replicas=args.sampler_replicas,
                    sampler_layout=args.sampler_layout,
                ):
                    continue
            return

        if verdict == "inconclusive" and final_control_reached:
            print(
                f"autopilot: final inconclusive control for {idea_id}; "
                "marking non-promotable run rejected and launching next queued idea",
                flush=True,
            )
            subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
            command_advance(argparse.Namespace(id=idea_id, profile=str(score["profile"]), min_control_n=None))
            data = load_yaml(IDEAS_PATH)
            idea = idea_by_id(data, idea_id)
            if str(idea.get("status")) == "inconclusive":
                idea["status"] = "rejected"
                save_yaml(IDEAS_PATH, data)
            if args.launch_next:
                if maybe_launch_next_from_queue(
                    args,
                    sampler_replicas=args.sampler_replicas,
                    sampler_layout=args.sampler_layout,
                ):
                    continue
            return

        if verdict in stop_only_verdicts:
            print(f"autopilot: stopping {idea_id} on {verdict}; not launching next", flush=True)
            subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
            command_advance(argparse.Namespace(id=idea_id, profile=str(score["profile"]), min_control_n=None))
            return

        if verdict in terminal_verdicts:
            if args.terminal_only_after_final_control and not final_control_reached:
                if key not in deferred_terminals:
                    deferred_terminals.add(key)
                    print(
                        f"autopilot: terminal verdict for {idea_id} at step={step} deferred until final control",
                        flush=True,
                    )
                    append_event(
                        {
                            "event": "autopilot_terminal_deferred",
                            "idea_id": idea_id,
                            "verdict": verdict,
                            "profile": str(score["profile"]),
                            "step": step,
                            "configured_num_steps": configured_num_steps(idea, candidate),
                        }
                    )
                if args.max_polls > 0 and polls >= args.max_polls:
                    break
                time.sleep(args.poll_seconds)
                continue
            if args.terminal_action in {"stop_advance", "stop_advance_launch_next"}:
                print(f"autopilot: terminal verdict for {idea_id}: {verdict}; stopping and advancing state", flush=True)
                subprocess.run([sys.executable, str(GENERATOR), "stop-trainer-control"], check=False)
                command_advance(argparse.Namespace(id=idea_id, profile=str(score["profile"]), min_control_n=None))
                append_event(
                    {
                        "event": "autopilot_terminal_stop",
                        "idea_id": idea_id,
                        "verdict": verdict,
                        "profile": str(score["profile"]),
                        "terminal_action": args.terminal_action,
                    }
                )
                if args.terminal_action == "stop_advance_launch_next" and args.launch_next:
                    if maybe_launch_next_from_queue(
                        args,
                        sampler_replicas=args.sampler_replicas,
                        sampler_layout=args.sampler_layout,
                    ):
                        continue
                return
            print(f"autopilot: terminal verdict for {idea_id}: {verdict}; leaving run state unchanged", flush=True)
            return

        if args.max_polls > 0 and polls >= args.max_polls:
            break
        time.sleep(args.poll_seconds)


def command_append_idea(args: argparse.Namespace) -> None:
    data = load_yaml(IDEAS_PATH)
    if any(str(idea.get("id")) == args.id for idea in data["ideas"]):
        raise SystemExit(f"idea {args.id!r} already exists")
    idea = {
        "id": args.id,
        "status": "queued",
        "priority": args.priority,
        "candidate": args.candidate,
        "hypothesis": args.hypothesis,
    }
    if args.rationale:
        idea["rationale"] = args.rationale
    data["ideas"].append(idea)
    save_yaml(IDEAS_PATH, data)
    append_event({"event": "append_idea", "idea": idea})
    print(f"appended {args.id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    next_cmd = sub.add_parser("next")
    next_cmd.set_defaults(func=command_next)

    launch = sub.add_parser("launch")
    launch.add_argument("--id")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--restart-inference", action="store_true")
    launch.add_argument("--num-steps", type=int)
    launch.add_argument("--prompts-per-step", type=int)
    launch.add_argument("--sampler-replicas", type=int, default=2)
    launch.add_argument("--sampler-layout", choices=("dedicated", "spare-teacher1"), default="spare-teacher1")
    launch.set_defaults(func=command_launch)

    monitor = sub.add_parser("monitor")
    monitor.add_argument("--profile", default="latest")
    monitor.add_argument("--samples", type=int, default=1)
    monitor.add_argument("--json", action="store_true")
    monitor.set_defaults(func=command_monitor)

    score = sub.add_parser("score")
    score.add_argument("--profile", default="latest")
    score.add_argument("--idea-id")
    score.add_argument("--min-control-n", type=float)
    score.add_argument("--json", action="store_true")
    score.add_argument("--record-incomplete", action="store_true")
    score.set_defaults(func=command_score)

    advance = sub.add_parser("advance")
    advance.add_argument("--id", required=True)
    advance.add_argument("--profile", default="latest")
    advance.add_argument("--min-control-n", type=float)
    advance.set_defaults(func=command_advance)

    autopilot = sub.add_parser("autopilot")
    autopilot.add_argument("--profile", default="latest")
    autopilot.add_argument("--poll-seconds", type=float, default=300.0)
    autopilot.add_argument("--max-polls", type=int, default=0)
    autopilot.add_argument("--launch-next", action="store_true")
    autopilot.add_argument("--launch-next-verdicts", default="science_reject")
    autopilot.add_argument("--stop-only-verdicts", default="infra_invalid")
    autopilot.add_argument(
        "--terminal-verdicts",
        default="strong_ptc_signal,promote_retest,weak_ptc_signal",
    )
    autopilot.add_argument(
        "--terminal-action",
        choices=("leave_running", "stop_advance", "stop_advance_launch_next"),
        default="leave_running",
        help="what to do when a terminal verdict is reached",
    )
    autopilot.add_argument(
        "--terminal-only-after-final-control",
        action="store_true",
        help="defer terminal verdict handling until the latest control is from the final configured step",
    )
    autopilot.add_argument("--sampler-replicas", type=int, default=2)
    autopilot.add_argument("--sampler-layout", choices=("dedicated", "spare-teacher1"), default="spare-teacher1")
    autopilot.add_argument("--post-launch-sleep-s", type=float, default=30.0)
    autopilot.add_argument("--launch-grace-seconds", type=float, default=300.0)
    autopilot.set_defaults(func=command_autopilot)

    append_idea = sub.add_parser("append-idea")
    append_idea.add_argument("--id", required=True)
    append_idea.add_argument("--candidate", required=True)
    append_idea.add_argument("--hypothesis", required=True)
    append_idea.add_argument("--rationale")
    append_idea.add_argument("--priority", type=int, default=50)
    append_idea.set_defaults(func=command_append_idea)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
