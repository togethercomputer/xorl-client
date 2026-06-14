#!/usr/bin/env python3
"""Emit hourly reminders for manual autoresearch queue checks.

This process intentionally does not inspect the queue. It only records when the
next human/agent check is due so queue inspection stays on a 60 minute cadence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "logs" / "hourly_wakeup.jsonl"
DEFAULT_STATE = ROOT / "logs" / "hourly_wakeup_state.json"
DEFAULT_DUE_FILE = ROOT / "logs" / "hourly_check_due.txt"
DEFAULT_MESSAGE = (
    "Autoresearch hourly check due: inspect loop health, completed scorecards, "
    "current experiment results, and queue hypothesis logic before editing ideas.yaml."
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"expected UTC timestamp string, got {type(value).__name__}")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, object] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def write_due_file(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def advance_state_after_due(args: argparse.Namespace, state: dict[str, object] | None, due_time: datetime) -> None:
    now = datetime.now(UTC)
    interval = timedelta(seconds=args.interval_seconds)
    next_wakeup = due_time + interval
    while next_wakeup <= now:
        next_wakeup += interval

    raw_wakeups = 0 if state is None else state.get("wakeups_emitted", 0)
    try:
        wakeups = int(float(raw_wakeups)) + 1
    except (TypeError, ValueError):
        wakeups = 1

    atomic_write_json(
        args.state,
        {
            "pid": os.getpid(),
            "started_utc": (state or {}).get("started_utc", utc_now()),
            "updated_utc": utc_now(),
            "next_wakeup_utc": next_wakeup.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval_seconds": args.interval_seconds,
            "wakeups_emitted": wakeups,
            "message": args.message,
            "last_due_utc": due_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wait_mode_fallback_emitted": True,
        },
    )


def run_command(command: str, *, log_path: Path) -> None:
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
    append_jsonl(
        log_path,
        {
            "event": "wakeup_command",
            "time_utc": utc_now(),
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=3600.0)
    parser.add_argument("--first-delay-seconds", type=float, default=None)
    parser.add_argument("--max-wakeups", type=int, default=0, help="0 means run until signaled")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--due-file", type=Path, default=DEFAULT_DUE_FILE)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument(
        "--wait-until-due",
        action="store_true",
        help=(
            "Block in the foreground until the due file exists or the recorded "
            "next_wakeup_utc has elapsed. Use this from an agent turn so the "
            "turn does not return before a check is allowed."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling cadence for --wait-until-due.",
    )
    parser.add_argument(
        "--command",
        default="",
        help="Optional shell command to run after each wakeup. Leave empty for reminder-only mode.",
    )
    return parser.parse_args()


def wait_until_due(args: argparse.Namespace) -> int:
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    first_delay = args.interval_seconds if args.first_delay_seconds is None else args.first_delay_seconds
    if first_delay < 0:
        raise SystemExit("--first-delay-seconds must be non-negative")

    append_jsonl(
        args.log,
        {
            "event": "wait_started",
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "state": str(args.state),
            "due_file": str(args.due_file),
        },
    )

    initialized_state = False
    while True:
        if args.due_file.exists():
            append_jsonl(
                args.log,
                {
                    "event": "wait_ready",
                    "time_utc": utc_now(),
                    "pid": os.getpid(),
                    "reason": "due_file_present",
                    "due_file": str(args.due_file),
                },
            )
            print(f"hourly check due: due file present at {args.due_file}", flush=True)
            return 0

        state = read_json(args.state)
        if state is None:
            if initialized_state:
                next_wakeup = datetime.now(UTC)
            else:
                next_wakeup_ts = time.time() + first_delay
                next_wakeup = datetime.fromtimestamp(next_wakeup_ts, UTC)
                initialized_state = True
                atomic_write_json(
                    args.state,
                    {
                        "pid": os.getpid(),
                        "started_utc": utc_now(),
                        "updated_utc": utc_now(),
                        "next_wakeup_utc": utc_from_timestamp(next_wakeup_ts),
                        "interval_seconds": args.interval_seconds,
                        "wakeups_emitted": 0,
                        "message": args.message,
                        "wait_mode_initialized": True,
                    },
                )
        else:
            next_wakeup = parse_utc_timestamp(state["next_wakeup_utc"])

        now = datetime.now(UTC)
        if now >= next_wakeup:
            write_due_file(args.due_file, args.message)
            advance_state_after_due(args, state, next_wakeup)
            append_jsonl(
                args.log,
                {
                    "event": "wait_ready",
                    "time_utc": utc_now(),
                    "pid": os.getpid(),
                    "reason": "next_wakeup_elapsed",
                    "next_wakeup_utc": next_wakeup.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "due_file": str(args.due_file),
                },
            )
            print(f"hourly check due: elapsed {next_wakeup:%Y-%m-%dT%H:%M:%SZ}", flush=True)
            return 0

        sleep_s = min(args.poll_seconds, max(0.1, (next_wakeup - now).total_seconds()))
        time.sleep(sleep_s)


def main() -> int:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.wait_until_due:
        return wait_until_due(args)

    first_delay = args.interval_seconds if args.first_delay_seconds is None else args.first_delay_seconds
    if first_delay < 0:
        raise SystemExit("--first-delay-seconds must be non-negative")

    stop = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True
        append_jsonl(args.log, {"event": "stopping", "time_utc": utc_now(), "signal": signum})

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    started = utc_now()
    append_jsonl(
        args.log,
        {
            "event": "started",
            "time_utc": started,
            "pid": os.getpid(),
            "interval_seconds": args.interval_seconds,
            "first_delay_seconds": first_delay,
            "message": args.message,
            "command_configured": bool(args.command),
        },
    )

    wakeups = 0
    delay = first_delay
    while not stop:
        next_wakeup_ts = time.time() + delay
        next_wakeup_utc = utc_from_timestamp(next_wakeup_ts)
        atomic_write_json(
            args.state,
            {
                "pid": os.getpid(),
                "started_utc": started,
                "updated_utc": utc_now(),
                "next_wakeup_utc": next_wakeup_utc,
                "interval_seconds": args.interval_seconds,
                "wakeups_emitted": wakeups,
                "message": args.message,
            },
        )

        remaining = delay
        while remaining > 0 and not stop:
            sleep_s = min(remaining, 5.0)
            time.sleep(sleep_s)
            remaining -= sleep_s
        if stop:
            break

        wakeups += 1
        due_payload = {
            "event": "hourly_check_due",
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "wakeups_emitted": wakeups,
            "message": args.message,
        }
        append_jsonl(args.log, due_payload)
        write_due_file(args.due_file, args.message)
        print("\a" + args.message, flush=True)
        if args.command:
            run_command(args.command, log_path=args.log)
        if args.max_wakeups and wakeups >= args.max_wakeups:
            break
        delay = args.interval_seconds

    append_jsonl(args.log, {"event": "exited", "time_utc": utc_now(), "wakeups_emitted": wakeups})
    return 0


if __name__ == "__main__":
    sys.exit(main())
