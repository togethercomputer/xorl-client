#!/usr/bin/env bash
# supervise_overnight.sh - keep an overnight ZORL autoresearch run alive by
# warm-resuming from the best export after any crash.
#
# Usage:
#   supervise_overnight.sh JOB_PREFIX CANDIDATE_PATH RESULT_ROOT
#
# Example:
#   experiments/zorl/standalone/supervise_overnight.sh \
#     zorl-ar-multopsd-004 \
#     experiments/zorl/autoresearch/candidates/MULTOPSD-004-rank4-fullpool-bigpop-overnight.yaml \
#     /home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/zorl_autoresearch/MULTOPSD-004
#
# Logic:
#   - Poll every POLL_SECONDS. Find the current k8s Job whose name starts with
#     JOB_PREFIX (controller.py uses generateName "<JOB_PREFIX>-...").
#   - If that job has a Running pod, do nothing (idempotent - never spawn a
#     second client; only one client may hold the sglang pools at a time).
#   - If the latest pod log shows a clean "[done]" or "[max-runtime]", the run
#     finished on purpose -> exit 0.
#   - If the job is gone, or its pod is Error/Failed without a clean marker,
#     treat it as a crash: delete the dead job, pick the most recently modified
#     RESULT_ROOT/*/exports/best containing adapter_model.safetensors, and
#     relaunch warm via --set-env ADAPTER_DIR=<that dir>. If no usable best
#     export exists yet, relaunch cold (no --set-env).
#   - Stop after RELAUNCH_CAP relaunches (exit 1).
#
# Intended to be run as a long-lived background task. set +e keeps a transient
# kubectl error from killing the loop.

set +e
set -u

NAMESPACE="${NAMESPACE:-apanda}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RELAUNCH_CAP="${RELAUNCH_CAP:-25}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONTROLLER="${REPO_ROOT}/experiments/zorl/autoresearch/controller.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

JOB_PREFIX="${1:-}"
CANDIDATE="${2:-}"
RESULT_ROOT="${3:-}"

if [[ -z "${JOB_PREFIX}" || -z "${CANDIDATE}" || -z "${RESULT_ROOT}" ]]; then
  echo "usage: $0 JOB_PREFIX CANDIDATE_PATH RESULT_ROOT" >&2
  exit 2
fi

if [[ ! -f "${CONTROLLER}" ]]; then
  echo "controller not found at ${CONTROLLER}" >&2
  exit 2
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] supervise_overnight: $*"; }

relaunch_count=0

# Latest job name (most recently created) whose name starts with JOB_PREFIX.
current_job() {
  kubectl get jobs -n "${NAMESPACE}" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
    | grep -E "^${JOB_PREFIX}(-|$)" \
    | tail -n 1
}

# Newest pod name for a given job (via job-name label).
pod_for_job() {
  local job="$1"
  kubectl get pods -n "${NAMESPACE}" \
    -l "job-name=${job}" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
    | tail -n 1
}

pod_phase() {
  local pod="$1"
  kubectl get pod -n "${NAMESPACE}" "${pod}" \
    -o jsonpath='{.status.phase}' 2>/dev/null
}

# Returns 0 (true) if the pod log shows a clean done/max-runtime marker.
pod_clean_exit() {
  local pod="$1"
  local logs
  logs="$(kubectl logs -n "${NAMESPACE}" "${pod}" --tail=400 2>/dev/null)"
  echo "${logs}" | grep -qE '\[done\]|\[max-runtime\]'
}

# Most recently modified RESULT_ROOT/*/exports/best dir that holds a usable
# adapter_model.safetensors. Empty string if none.
latest_best_export() {
  local best="" newest=0 d mt
  shopt -s nullglob
  for d in "${RESULT_ROOT}"/*/exports/best; do
    [[ -f "${d}/adapter_model.safetensors" ]] || continue
    mt="$(stat -c %Y "${d}/adapter_model.safetensors" 2>/dev/null)"
    [[ -n "${mt}" ]] || continue
    if (( mt > newest )); then
      newest="${mt}"
      best="${d}"
    fi
  done
  shopt -u nullglob
  echo "${best}"
}

relaunch() {
  if (( relaunch_count >= RELAUNCH_CAP )); then
    log "relaunch cap ${RELAUNCH_CAP} reached; giving up (exit 1)"
    exit 1
  fi

  # Idempotency guard: never relaunch while a Running pod still exists.
  local job pod phase
  job="$(current_job)"
  if [[ -n "${job}" ]]; then
    pod="$(pod_for_job "${job}")"
    if [[ -n "${pod}" ]]; then
      phase="$(pod_phase "${pod}")"
      if [[ "${phase}" == "Running" || "${phase}" == "Pending" ]]; then
        log "skip relaunch - job ${job} pod ${pod} is ${phase}"
        return 0
      fi
    fi
    log "deleting dead job ${job}"
    kubectl delete job -n "${NAMESPACE}" "${job}" --wait=true >/dev/null 2>&1
  fi

  local best
  best="$(latest_best_export)"
  relaunch_count=$((relaunch_count + 1))
  if [[ -n "${best}" ]]; then
    log "relaunch #${relaunch_count} WARM from ${best}"
    "${PYTHON_BIN}" "${CONTROLLER}" launch \
      --candidate "${CANDIDATE}" \
      --set-env "ADAPTER_DIR=${best}"
  else
    log "relaunch #${relaunch_count} COLD (no best export yet)"
    "${PYTHON_BIN}" "${CONTROLLER}" launch \
      --candidate "${CANDIDATE}"
  fi
}

log "starting supervisor: prefix=${JOB_PREFIX} candidate=${CANDIDATE} result_root=${RESULT_ROOT} cap=${RELAUNCH_CAP} poll=${POLL_SECONDS}s"

while true; do
  job="$(current_job)"

  if [[ -z "${job}" ]]; then
    log "no job matching ${JOB_PREFIX}; relaunching"
    relaunch
    sleep "${POLL_SECONDS}"
    continue
  fi

  pod="$(pod_for_job "${job}")"
  if [[ -z "${pod}" ]]; then
    log "job ${job} has no pod yet; waiting"
    sleep "${POLL_SECONDS}"
    continue
  fi

  phase="$(pod_phase "${pod}")"
  case "${phase}" in
    Running|Pending)
      # Healthy or scheduling. Even if Running, a clean-exit marker can appear
      # right before the pod terminates - honor it.
      if pod_clean_exit "${pod}"; then
        log "clean exit detected on ${pod} (job ${job}); done (exit 0)"
        exit 0
      fi
      ;;
    Succeeded)
      log "pod ${pod} Succeeded (job ${job}); done (exit 0)"
      exit 0
      ;;
    Failed|"")
      if pod_clean_exit "${pod}"; then
        log "pod ${pod} ended after a clean marker; done (exit 0)"
        exit 0
      fi
      log "pod ${pod} phase='${phase}' without clean marker; treating as crash"
      relaunch
      ;;
    *)
      # Unknown/transient phase; re-check next tick.
      log "pod ${pod} phase='${phase}'; rechecking next tick"
      ;;
  esac

  sleep "${POLL_SECONDS}"
done
