#!/bin/bash
# overnight_supervisor.sh — drive the OPD science queue back-to-back, keep GPUs
# busy all night, recover from failures. Read-only-supervised by the agent.
# Queue is hardcoded; to change it, stop this script and restart with edits.
# Logs: /tmp/overnight_supervisor.log ; live state: /tmp/overnight_state.txt
set -uo pipefail
cd /home/apanda/xorl-apanda-dev-opd-port
PY=.venv/bin/python
GEN="experiments/opd_profile/k8s/q36_35b_reprogrammable_slots.py"
GFLAGS="--model q36 --trainer-nodes 4 --teacher-replicas 1 --teacher-route direct --eval-replicas 0"
SFLAGS="--sampler-replicas 2 --sampler-layout dedicated"
CAND=experiments/opd_profile/autoresearch/candidates
CTRL=/shared/opd-control/er-opd-q36-35b-slots
RD=/shared/opd-coord/encoded_reasoning/results/qwen3_6_35b_self_distill/er-opd-q36-35b-slots
LOG=/tmp/overnight_supervisor.log
STATE=/tmp/overnight_state.txt

# Ordered queue: CANDIDATE:NUM_STEPS:PROMPTS_PER_STEP
# 2026-06-13 14:43 SWAP: R1-R4 (015/016/017/018) DONE. Remaining = the promising
# probes (coef-0.5, extend-c1) then the SFT anchor. E0a runs on compiled (SFT
# can't use quack_linear). OPRD probes (019/020) are quack_linear-safe.
QUEUE=(
  "ARITH-019-OPRD-C05:101:64"
  "ARITH-020-OPRD-C1-LONG:201:64"
  "E0A-SFT-LONG:321:128"
)
# Infinite keepalive run after the queue (never leave GPUs idle):
KEEPALIVE="E0A-SFT-LONG:321:128"

log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
setstate(){ echo "[$(date -u +%H:%M:%SZ)] $*" > "$STATE"; }

samplers_ready(){
  for s in sglang-0 sglang-1; do
    kubectl exec -n apanda er-opd-q36-35b-slots-$s -- curl -s --max-time 5 \
      http://localhost:30060/v1/models 2>/dev/null | grep -q '"id"' || return 1
  done; return 0
}
wait_samplers(){
  local t=0
  while ! samplers_ready; do sleep 20; t=$((t+20)); [ $t -gt 900 ] && { log "WARN samplers not ready after 900s"; return 1; }; done
  log "samplers ready"; return 0
}
recreate_samplers(){
  log "recreating samplers (clean Mooncake)"
  $PY $GEN $GFLAGS write-student-inference-control $SFLAGS >>"$LOG" 2>&1 || true
  sleep 40; wait_samplers
}
stop_trainer(){
  $PY $GEN $GFLAGS stop-trainer-control --remove-run >>"$LOG" 2>&1 || true
  sleep 45
}

# run_one CAND STEPS PPS  -> 0 completed, 1 hard-fail, 2 launch-fail, 3 stall
run_one(){
  local cand=$1 steps=$2 pps=$3
  log "LAUNCH $cand steps=$steps pps=$pps"
  setstate "LAUNCHING $cand steps=$steps pps=$pps"
  $PY $GEN $GFLAGS write-trainer-control --candidate "$CAND/$cand.yaml" \
    --num-steps "$steps" --prompts-per-step "$pps" $SFLAGS >>"$LOG" 2>&1 || { log "write-trainer-control errored for $cand"; return 2; }
  local start; start=$(date +%s)
  sleep 60
  local HL el rows mtime now d
  while true; do
    HL=$(ls -t $CTRL/trainer-head/logs/*.log 2>/dev/null | head -1)
    el=$(( $(date +%s) - start ))
    # completion
    if [ -n "$HL" ] && grep -qE "OPD config .* completed|cleanup rc=0|RUN COMPLETE" "$HL" 2>/dev/null; then
      log "$cand COMPLETED (${el}s)"; return 0; fi
    # hard failure signatures (head log tail)
    if [ -n "$HL" ] && tail -60 "$HL" 2>/dev/null | grep -qiE "Traceback|CUDA error|CUDA out of memory|RuntimeError|AssertionError|Unknown ce_mode|not supported with|aborting training|\[P2P\].*ret=-1.*after 50"; then
      log "$cand HARD-FAIL signature (${el}s): $(tail -60 "$HL" | grep -iE 'Traceback|CUDA|RuntimeError|AssertionError|ce_mode|not supported|abort|ret=-1' | tail -1)"; return 1; fi
    # launch-fail: 14min in, still no step-0 and no profile row
    d=$(ls -td $RD/*config${cand}*trainer-head 2>/dev/null | head -1)
    rows=0; [ -n "$d" ] && [ -f "$d/opd_profile.jsonl" ] && rows=$(grep -c '^{' "$d/opd_profile.jsonl" 2>/dev/null || echo 0)
    if [ $el -gt 840 ] && [ "${rows:-0}" -lt 1 ] && ! grep -qE "OPD step 0|=== OPD step" "$HL" 2>/dev/null; then
      log "$cand LAUNCH-FAIL (no step-0 after ${el}s)"; return 2; fi
    # stall: head log silent >40min (end-of-run control legitimately silent <=25min)
    mtime=$(stat -c %Y "$HL" 2>/dev/null || echo 0); now=$(date +%s)
    if [ $((now-mtime)) -gt 2400 ]; then log "$cand STALL (head log silent $(((now-mtime)/60))min)"; return 3; fi
    setstate "RUNNING $cand ${el}s rows=${rows:-0} | $(tail -1 "$HL" 2>/dev/null | cut -c1-90)"
    sleep 60
  done
}

drive(){
  local entry=$1 cand steps pps rc attempts
  cand=${entry%%:*}; rest=${entry#*:}; steps=${rest%%:*}; pps=${rest##*:}
  attempts=0
  while [ $attempts -lt 3 ]; do
    run_one "$cand" "$steps" "$pps"; rc=$?
    if [ $rc -eq 0 ]; then
      log "$cand DONE (clean). settling before next."; setstate "DONE $cand"; sleep 45; return 0
    fi
    attempts=$((attempts+1))
    log "$cand failed rc=$rc attempt=$attempts; recover (stop+recreate samplers) and retry"
    setstate "RECOVERING $cand rc=$rc attempt=$attempts"
    stop_trainer; recreate_samplers
    if [ $rc -eq 1 ]; then
      # hard-fail twice in a row likely a config problem; widen breath
      sleep 30
    fi
  done
  log "$cand GAVE UP after 3 attempts; advancing to keep GPUs busy"
  setstate "GAVE-UP $cand"
  return 1
}

log "=== overnight supervisor START ; queue: ${QUEUE[*]} ==="
for entry in "${QUEUE[@]}"; do
  drive "$entry"
done
log "=== queue exhausted; entering KEEPALIVE ($KEEPALIVE) loop ==="
while true; do
  drive "$KEEPALIVE" || { log "keepalive drive failed; full recovery"; stop_trainer; recreate_samplers; }
done
