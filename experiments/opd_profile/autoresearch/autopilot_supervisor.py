#!/usr/bin/env python3
"""Self-healing supervisor for the PTC autopilot.

The autopilot (`controller.py autopilot`) EXITS (returns) on any trainer
health/P2P failure — it marks the running idea `infra_invalid` and stops. While
the operator is offline that idles every GPU until the next manual relaunch
(the documented recurring P2P crash; see P2P_CRASH_DIAGNOSIS.md). This watcher
keeps the autopilot alive and bounded-retries recently-crashed ideas so a
transient Mooncake handshake failure becomes a non-event.

Behaviour, every POLL seconds:
  - FIRST, recover any dead bare stack pod (delete + re-apply its manifest). A
    trainer rank that exits non-zero (CUDA assert, or stopping a wedged trainer)
    leaves its restartPolicy=Never Pod in Failed/Error forever; the 64-rank
    rendezvous then hangs and every relaunch stalls in bring-up. Recreating that
    one Pod (others stay warm) is the documented fix — now automatic.
  - If an autopilot process is already running: do nothing (beyond pod recovery).
  - Else, if there is runnable work (queued/launched, or a recently-crashed
    infra_invalid idea we can re-queue): re-queue the recent infra_invalid idea
    (bounded by MAX_RETRY_PER_IDEA), then relaunch the autopilot with the exact
    production args.
  - Else (queue genuinely drained): log idle and wait — do NOT busy-relaunch.

It only re-queues ideas that crashed RECENTLY (last_launched within
REQUEUE_RECENCY_S) so stale/science infra_invalid ideas are left alone, and it
escalates loudly in the log once an idea exhausts its retries (likely needs the
inference-stack recreate, not a bare relaunch).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

AR = Path("/home/apanda/xorl-apanda-dev-opd-port/experiments/opd_profile/autoresearch")
REPO = Path("/home/apanda/xorl-apanda-dev-opd-port")
IDEAS = AR / "ideas.yaml"
LOGDIR = AR / "logs"
WATCH_LOG = LOGDIR / "autopilot_supervisor.log"
AUTOPILOT_LOG = LOGDIR / "autopilot-supervised.log"

POLL = 60                      # seconds between checks
BOOT_GRACE = 45                # seconds to let a freshly-launched autopilot appear
MAX_RETRY_PER_IDEA = 3         # bounded re-queue of a crashed idea
REQUEUE_RECENCY_S = 3 * 3600   # only re-queue ideas that crashed within 3h

AUTOPILOT_ARGS = [
    sys.executable, "experiments/opd_profile/autoresearch/controller.py", "autopilot",
    "--launch-next", "--poll-seconds", "300", "--max-polls", "288",
    "--sampler-replicas", "2", "--sampler-layout", "spare-teacher1",
    "--terminal-action", "stop_advance_launch_next",
    "--terminal-only-after-final-control", "--launch-grace-seconds", "1800",
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} [supervisor] {msg}"
    print(line, flush=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a") as f:
        f.write(line + "\n")


def _start_ticks(pid: int) -> int:
    """Process start time in clock ticks (field 22 of /proc/pid/stat); 0 if gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


def live_autopilots() -> list[int]:
    """PIDs of genuine python `controller.py autopilot` processes (excludes bash wrappers)."""
    out = subprocess.run(["pgrep", "-f", "controller.py autopilot"], capture_output=True, text=True).stdout
    pids = []
    for tok in out.split():
        try:
            pid = int(tok)
        except ValueError:
            continue
        try:
            argv = [c for c in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00") if c]
        except OSError:
            continue
        exe = (argv[0].decode(errors="ignore") if argv else "").lower()
        if "python" in exe and any(b"controller.py" in a for a in argv) and any(b"autopilot" in a for a in argv):
            pids.append(pid)
    return pids


def enforce_single_autopilot() -> int:
    """Kill duplicate autopilots (keep the OLDEST), so a concurrent cron-agent relaunch
    can never race the supervisor's autopilot into queue-state corruption. Returns the
    survivor pid (0 if none)."""
    pids = live_autopilots()
    if len(pids) <= 1:
        return pids[0] if pids else 0
    keep = min(pids, key=_start_ticks)  # oldest = lowest start ticks
    for p in pids:
        if p != keep:
            try:
                os.kill(p, 15)
                log(f"killed DUPLICATE autopilot pid={p} (kept oldest pid={keep}) — concurrent relaunch race")
            except (ProcessLookupError, PermissionError, OSError) as e:
                log(f"could not kill duplicate autopilot pid={p}: {e!r}")
    return keep


def autopilot_running() -> bool:
    return bool(live_autopilots())


def _age_seconds(ts: str | None) -> float:
    if not ts:
        return 1e18
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return 1e18


def requeue_recent_infra_invalid() -> bool:
    """Flip recently-crashed infra_invalid ideas back to queued (bounded). Returns True if changed."""
    data = yaml.safe_load(IDEAS.read_text())
    changed = False
    for it in data.get("ideas", []):
        if str(it.get("status")) != "infra_invalid":
            continue
        if _age_seconds(it.get("last_launched_utc")) > REQUEUE_RECENCY_S:
            continue  # stale / non-transient failure — leave it
        rc = int(it.get("infra_retry_count", 0))
        if rc < MAX_RETRY_PER_IDEA:
            it["status"] = "queued"
            it["infra_retry_count"] = rc + 1
            log(f"re-queued {it.get('id')} after infra_invalid (retry {rc + 1}/{MAX_RETRY_PER_IDEA}); "
                f"reason was: {str(it.get('last_reason'))[:120]}")
            changed = True
        else:
            log(f"ESCALATE: {it.get('id')} exhausted {MAX_RETRY_PER_IDEA} infra retries — leaving "
                f"infra_invalid. A bare relaunch is not clearing it; likely needs inference-stack "
                f"recreate (dispatch+teacher-smg+samplers) per the restart recipe.")
    if changed:
        IDEAS.write_text(yaml.safe_dump(data, sort_keys=False))
    return changed


def has_work() -> bool:
    """True if a run is in flight OR the controller reports a dependency-satisfied
    runnable idea. Uses `controller.py next` as the runnability oracle so the
    supervisor does NOT relaunch-churn when only ungateable (requires-unsatisfied)
    ideas remain. CONSERVATIVE: on any error, returns True (prefer relaunching over
    a silent idle when work might exist)."""
    try:
        data = yaml.safe_load(IDEAS.read_text())
        if any(str(it.get("status")) == "launched" for it in data.get("ideas", [])):
            return True
        r = subprocess.run(
            [sys.executable, "experiments/opd_profile/autoresearch/controller.py", "next"],
            capture_output=True, text=True, cwd=str(REPO), timeout=90,
        )
        out = (r.stdout + r.stderr).lower()
        return "no queued idea" not in out  # controller prints 'no queued ideas' when drained
    except Exception as e:  # never silently idle on a checker error
        log(f"has_work runnability check errored ({e!r}) — assuming work to be safe")
        return True


def relaunch() -> None:
    AUTOPILOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = AUTOPILOT_LOG.open("a")
    subprocess.Popen(AUTOPILOT_ARGS, cwd=str(REPO), stdout=logf,
                     stderr=subprocess.STDOUT, start_new_session=True)
    log(f"relaunched autopilot (cwd={REPO})")


# ---------------------------------------------------------------------------
# Dead bare-pod recovery (gang-rendezvous unblock)
# ---------------------------------------------------------------------------
# A trainer rank that exits non-zero (CUDA device-side assert, or stopping a
# wedged trainer) leaves its bare Pod (restartPolicy=Never, no controller) in
# Failed/Error. It never self-heals, so the 64-rank rendezvous waits forever
# and every subsequent relaunch hangs in early bring-up ("Engine Core
# initialization timeout", GPUs idle ~4 MiB). The fix is always: delete that one
# Pod and re-apply its rendered manifest; the other pods stay warm. This encodes
# it so the supervisor clears it with no operator in the loop.
NAMESPACE = "apanda"
STACK = "er-opd-q36-35b-slots"
GENERATOR = REPO / "experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py"
# Keep render args in sync with AUTOPILOT_ARGS (only sampler pods vary by these;
# trainer/teacher/dispatch docs are identical regardless).
SAMPLER_REPLICAS = "2"
SAMPLER_LAYOUT = "spare-teacher1"
DEAD_POD_PHASES = {"Failed", "Unknown"}   # a bare Pod cannot recover from these
DEAD_WAITING_REASONS = {"CrashLoopBackOff"}
MAX_POD_RECREATE = 3                       # bound recreate-loops per pod per session
_pod_recreate_count: dict[str, int] = {}


def _scan_stack_pods() -> tuple[list[str], set[str]]:
    """Return (dead bare-Pod names, healthy pod names).

    Dead = a bare Pod (no ownerReferences -> no controller to self-heal) whose
    phase is terminal-bad, or a container stuck CrashLoopBackOff. STRICT: a
    Running+Ready pod is never reported dead. Healthy = Running and every
    container ready (used to reset the per-pod recreate budget on recovery).
    """
    try:
        r = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-l", f"stack={STACK}", "-o", "json"],
            capture_output=True, text=True, timeout=60,
        )
        items = json.loads(r.stdout or "{}").get("items", [])
    except Exception as e:  # kubectl/json hiccup — treat as "nothing to recover"
        log(f"dead-pod scan: kubectl get failed ({e!r}) — skipping this poll")
        return [], set()
    dead, healthy = [], set()
    for pod in items:
        meta = pod.get("metadata", {})
        name = meta.get("name", "")
        if not name:
            continue
        status = pod.get("status", {})
        phase = status.get("phase", "")
        cstatuses = status.get("containerStatuses", []) or []
        if phase == "Running" and cstatuses and all(cs.get("ready") for cs in cstatuses):
            healthy.add(name)
        if meta.get("ownerReferences"):
            continue  # managed by a controller — let it self-heal
        bad = phase in DEAD_POD_PHASES
        for cs in cstatuses:
            if (cs.get("state", {}).get("waiting") or {}).get("reason", "") in DEAD_WAITING_REASONS:
                bad = True
        if bad:
            dead.append(name)
    return dead, healthy


def recover_dead_bare_pods() -> bool:
    """Delete+recreate any dead bare stack Pod from a freshly-rendered manifest.

    Returns True if any Pod was recreated (caller should grant bring-up grace).
    Bounded by MAX_POD_RECREATE per pod; escalates loudly past the bound (a pod
    that keeps dying is a real fault — HBM ECC / recurring assert — not a
    transient rendezvous death).
    """
    dead, healthy = _scan_stack_pods()
    # Reset the recreate budget for any pod that came back healthy.
    for name in list(_pod_recreate_count):
        if name in healthy:
            _pod_recreate_count.pop(name, None)
    if not dead:
        return False

    manifest = Path("/tmp/supervisor_stack_manifest.yaml")
    try:
        subprocess.run(
            [sys.executable, str(GENERATOR), "render-manifest",
             "--sampler-replicas", SAMPLER_REPLICAS, "--sampler-layout", SAMPLER_LAYOUT,
             "--output", str(manifest)],
            cwd=str(REPO), capture_output=True, text=True, timeout=120, check=True,
        )
        by_name = {
            d.get("metadata", {}).get("name"): d
            for d in yaml.safe_load_all(manifest.read_text())
            if d and d.get("kind") == "Pod"
        }
    except Exception as e:
        log(f"dead-pod recovery: render-manifest failed ({e!r}) — cannot recreate {dead}")
        return False

    recreated = False
    for name in dead:
        n = _pod_recreate_count.get(name, 0)
        if n >= MAX_POD_RECREATE:
            log(f"ESCALATE: {name} dead again after {n} recreations this session — NOT "
                f"recreating. Likely a real fault (HBM ECC / recurring CUDA assert / bad "
                f"node), not a transient rendezvous death; needs operator triage.")
            continue
        doc = by_name.get(name)
        if doc is None:
            log(f"dead-pod recovery: {name} absent from rendered manifest — skipping")
            continue
        podfile = Path(f"/tmp/supervisor_recreate_{name}.yaml")
        podfile.write_text(yaml.safe_dump(doc, sort_keys=False))
        try:
            subprocess.run(["kubectl", "delete", "pod", "-n", NAMESPACE, name,
                            "--grace-period=5", "--ignore-not-found"],
                           capture_output=True, text=True, timeout=60)
            subprocess.run(["kubectl", "apply", "-f", str(podfile)],
                           capture_output=True, text=True, timeout=60, check=True)
            _pod_recreate_count[name] = n + 1
            recreated = True
            log(f"RECOVERED dead bare pod {name} (delete+recreate {n + 1}/{MAX_POD_RECREATE}) "
                f"— unblocks the gang rendezvous so the next relaunch can bring up the trainer")
        except Exception as e:
            log(f"dead-pod recovery: failed to recreate {name} ({e!r})")
    return recreated


PIDFILE = LOGDIR / "supervisor.pid"


def _is_live_supervisor(pid: int, me: int) -> bool:
    """True iff `pid` is a live, distinct python process actually running this script.
    Uses /proc cmdline so the launching bash wrapper (whose cmdline contains the
    script name) is never mistaken for a second supervisor."""
    if pid == me:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
    except OSError:
        return False
    argv = [c.decode(errors="ignore") for c in cmd if c]
    exe = (argv[0] if argv else "").lower()
    return "python" in exe and any("autopilot_supervisor.py" in a for a in argv[1:])


def main() -> None:
    # single-instance guard via pidfile (robust against the launching shell wrapper)
    me = os.getpid()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        try:
            prev = int(PIDFILE.read_text().strip())
        except (ValueError, OSError):
            prev = -1
        if _is_live_supervisor(prev, me):
            log(f"another supervisor already running (pid={prev}); exiting")
            return
    PIDFILE.write_text(str(me))
    log(f"supervisor start pid={me} poll={POLL}s max_retry/idea={MAX_RETRY_PER_IDEA} recency={REQUEUE_RECENCY_S}s")
    idle_logged = False
    while True:
        try:
            # Clear dead bare pods FIRST so a stuck gang rendezvous can re-form
            # before (and independent of) any autopilot relaunch decision.
            if recover_dead_bare_pods():
                time.sleep(BOOT_GRACE)  # let the recreated pod schedule + rejoin
            if autopilot_running():
                enforce_single_autopilot()  # kill any concurrent-relaunch duplicate
                idle_logged = False
            else:
                requeued = requeue_recent_infra_invalid()
                if has_work():
                    log("autopilot not running but work is queued — relaunching")
                    relaunch()
                    idle_logged = False
                    time.sleep(BOOT_GRACE)
                elif requeued:
                    log("re-queued crashed idea(s) — relaunching")
                    relaunch()
                    idle_logged = False
                    time.sleep(BOOT_GRACE)
                elif not idle_logged:
                    log("queue drained and no recent crash to retry — idle, not relaunching")
                    idle_logged = True
        except Exception as e:  # never let the supervisor itself die
            log(f"supervisor loop error: {e!r}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
