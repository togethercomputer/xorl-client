#!/usr/bin/env bash
# Overnight watchdog for the 235B GRPO filler run on stack er-opd-q235-fillerrft-slots.
# Modes:
#   gate -> exit when step-0/1 fwd/bwd validated (Mooncake R3 fix works) or a failure.
#   run  -> exit when "GRPO run completed", a failure, or a hang; heartbeat at WD_MAX_SECS.
# Exit codes: 0=gate-passed  2=run-completed  10=R3-mooncake-error  11=fatal(traceback/oom/proc-died)
#             12=hang(no log progress + GPU spin)  13=heartbeat/max-secs reached
set -uo pipefail

STACK="${STACK:-er-opd-q235-fillerrft-slots}"
RR=/shared/opd-coord/encoded_reasoning/results/qwen3_235b_self_distill/${STACK}
CTL=/shared/opd-control/${STACK}
WD_MODE="${WD_MODE:-gate}"
WD_MAX_SECS="${WD_MAX_SECS:-1500}"     # overall watchdog lifetime (gate default 25m)
WD_HANG_SECS="${WD_HANG_SECS:-900}"    # no-log-progress window that triggers a hang probe (15m)
POLL=20

start=$(date -u +%s)
# newest head RUN_DIR
RUN_DIR="$(ls -dt ${RR}/2026*-configb-*trainer-head/ 2>/dev/null | head -1)"
CLOG="${RUN_DIR}trainer_job.log"
SLOG="${RUN_DIR}server.log"
HEAD_STATUS="${CTL}/trainer-head/status"

emit(){ echo "[WD $(date -u +%H:%M:%SZ)] $*"; }
emit "mode=${WD_MODE} run_dir=${RUN_DIR}"

last_size=0; last_change=$(date -u +%s)
prev_step=-1

while true; do
  now=$(date -u +%s); elapsed=$((now-start))
  # RECOMPUTE run dir each poll (head may write its RUN_DIR after arm; avoids empty-dir false hang)
  RUN_DIR="$(ls -dt ${RR}/2026*-configb-*trainer-head/ 2>/dev/null | head -1)"; CLOG="${RUN_DIR}trainer_job.log"; SLOG="${RUN_DIR}server.log"

  # --- combined log size for progress/hang detection ---
  csz=$(stat -c %s "$CLOG" 2>/dev/null || echo 0)
  ssz=$(stat -c %s "$SLOG" 2>/dev/null || echo 0)
  tot=$((csz+ssz))
  if [ "$tot" -ne "$last_size" ]; then last_size=$tot; last_change=$now; fi

  # --- run completion FIRST (before fatal checks) — a completed run tears the engine down normally
  # ("All components stopped"), which must NOT be misread as a fatal engine death. ---
  if grep -qa "GRPO run completed" "$CLOG" 2>/dev/null; then
    emit "DONE: GRPO run completed at step ${prev_step}"; exit 2
  fi

  # --- fatal / R3 signatures (both logs) ---
  if grep -qa "requires dict metadata with shape" "$CLOG" "$SLOG" 2>/dev/null; then
    emit "FAIL: R3 Mooncake dict error recurred"; tail -8 "$CLOG" 2>/dev/null; exit 10
  fi
  if grep -qaE "CUDA out of memory|OutOfMemoryError|torch.OutOfMemory" "$CLOG" "$SLOG" 2>/dev/null; then
    emit "FAIL: OOM"; grep -aE "out of memory|OutOfMemory" "$CLOG" "$SLOG" 2>/dev/null | tail -4; exit 11
  fi
  # engine death (segfault / process exit / launcher teardown) — caught in server.log even while
  # the head run.sh stays alive with the client retrying. THIS is what we missed last round.
  if grep -qaE "Segfault encountered|Engine process exited with code|All components stopped|Stopping all components\.\.\." "$SLOG" 2>/dev/null; then
    emit "FAIL: engine died (segfault/exit/teardown) — server.log:"
    grep -aE "Segfault|exited with code|coro_rpc.*Connection refused|got signal" "$SLOG" 2>/dev/null | tail -8
    exit 11
  fi

  # --- head process died? ---
  if grep -qa "^exited_at=" "$HEAD_STATUS" 2>/dev/null; then
    emit "FAIL: head slot child exited: $(cat "$HEAD_STATUS")"
    grep -aE "Traceback|Error|error|Engine error|RequestFailed" "$CLOG" 2>/dev/null | tail -8
    exit 11
  fi

  # --- progress: latest step ---
  curstep=$(grep -aoE "Step [0-9]+ rewards" "$CLOG" 2>/dev/null | grep -aoE "[0-9]+" | tail -1)
  curstep="${curstep:-}"
  if [ -n "$curstep" ] && [ "$curstep" != "$prev_step" ]; then
    prev_step="$curstep"
    line=$(grep -aE "Step ${curstep} rewards" "$CLOG" 2>/dev/null | tail -1)
    emit "progress: ${line}"
  fi

  # --- run completion ---
  if grep -qa "GRPO run completed" "$CLOG" 2>/dev/null; then
    emit "DONE: GRPO run completed at step ${prev_step}"; exit 2
  fi

  # --- gate pass: step 0 fwd/bwd survived (we reached step 1) ---
  if [ "$WD_MODE" = "gate" ]; then
    if grep -qaE "Step 1 rewards|Step 1:" "$CLOG" 2>/dev/null; then
      emit "GATE PASSED: reached step 1 (R3 Mooncake fix validated; no 500)"
      grep -aiE "k3|kl/|reward|format" "$CLOG" 2>/dev/null | tail -12
      exit 0
    fi
  fi

  # --- hang probe ---
  if [ $((now-last_change)) -ge "$WD_HANG_SECS" ]; then
    emit "STALL: no log growth for $((now-last_change))s — probing GPU power on worker-1"
    pw=$(kubectl exec ${STACK}-trainer-worker-1 -- bash -lc "nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv,noheader 2>/dev/null | head -2" 2>/dev/null)
    emit "GPU(w1): ${pw}"
    # spin signature = high util + low power; treat sustained stall as hang regardless
    emit "FAIL: HANG (stall ${WD_HANG_SECS}s)"; tail -6 "$CLOG" 2>/dev/null; exit 12
  fi

  # --- heartbeat / max lifetime ---
  if [ "$elapsed" -ge "$WD_MAX_SECS" ]; then
    emit "HEARTBEAT: ${elapsed}s elapsed (max ${WD_MAX_SECS}); last_step=${prev_step}; re-assess"
    tail -4 "$CLOG" 2>/dev/null
    exit 13
  fi

  sleep "$POLL"
done
