#!/usr/bin/env bash
# Overnight random hyperparameter search for fresh_ab ES (rank x batch x population x lr).
# One driver instance per pool. Each trial: restart pods (pristine base) -> launch a 1h
# run with a randomly-sampled config -> wait -> record peak held-out probe -> repeat.
# Defensive: a failed trial logs and continues; never crashes the loop.
#
# Usage: overnight_rank_search.sh <POOL_LETTER> <NREPLICAS> <BASE_CANDIDATE>
set -uo pipefail
POOL="$1"; NREP="$2"; BASE="$3"
APP="zorl-ar-sglang-${POOL,,}"
HERE="/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/autoresearch"
RESULTS="/home/apanda/xorl-apanda-dev-zorl-consolidated/experiments/zorl/results/overnight_search_${POOL}.csv"
cd "$HERE"
[ -f "$RESULTS" ] || echo "ts,pool,trial,rank,batch,pairs_per_shard,total_pairs,lr,sigma,peak_probe,n_probes,job,status" > "$RESULTS"

# grid
RANKS=(8 16 32); BATCHES=(256 512 768); PPS=(8 16); LRMULT=(0.5 1 2)
pick() { local a=("$@"); echo "${a[$((RANDOM % ${#a[@]}))]}"; }

# Priority queue (rank batch pps lrm): configs the rank-free-lr fix made newly
# meaningful. r8 was unaffected (its base was already abs-lr 5.37e-4), but r16
# (was 3.8e-4) and r32 (was 2.7e-4) only now reach the optimum at lrm1x. Run them
# matched to the leader batch (b768/p64) for a clean rank-isolation vs r8's 0.867,
# plus a lighter b512/p64 for a many-probe (un-truncated) ceiling read on r32.
PRIORITY=(
  "16 768 8 1"
  "32 768 8 1"
  "16 512 8 1"
  "32 512 8 1"
)

wait_pool_healthy() {  # return 0 only when N/N pods AND owner (pod-0) serves (/health 200).
  # The 1/1 readiness probe flips ~28s in, long before the 35B model loads, so gating on
  # 1/1 alone launches the client into a dead server. Gate on real /health.
  local deadline=$((SECONDS + 1200))
  while true; do
    ready=$(kubectl get pods --no-headers 2>/dev/null | grep "$APP" | grep -c '1/1')
    if [ "$ready" -ge "$NREP" ]; then
      code=$(timeout 15 kubectl exec "${APP}-0" -- curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health 2>/dev/null)
      [ "$code" = "200" ] && return 0
    fi
    for p in $(kubectl get pods --no-headers -o wide 2>/dev/null | grep "$APP" \
                 | grep -E "CrashLoop|Error|h100-073|h100-064" | awk '{print $1}'); do
      kubectl delete pod "$p" --wait=false >/dev/null 2>&1
    done
    [ $SECONDS -gt $deadline ] && return 1
    sleep 30
  done
}

TRIAL=0
while true; do
  TRIAL=$((TRIAL+1))
  if [ "$TRIAL" -le "${#PRIORITY[@]}" ]; then
    read -r rank batch pps lrm <<< "${PRIORITY[$((TRIAL-1))]}"
    echo "[priority $TRIAL/${#PRIORITY[@]}]"
  else
    rank=$(pick "${RANKS[@]}"); batch=$(pick "${BATCHES[@]}")
    pps=$(pick "${PPS[@]}");    lrm=$(pick "${LRMULT[@]}")
  fi
  total_pairs=$((pps * NREP))
  # lr is RANK-FREE. sigma ~ 1/sqrt(r) already normalizes the perturbation
  # displacement, so ||update|| ~ lr alone; scaling lr by sqrt(16/r) too
  # double-counts the rank correction and under-steps at high rank (r32/lr1x
  # topped out ~0.797, r32 at abs lr 5.37e-4 tied the leader at 0.867).
  # Empirical optimum abs lr ~5.37e-4 (= old r8/lr1x); LRMULT explores around it.
  # sigma KEEPS the 1/sqrt(r) match (load-bearing rank normalization).
  base_lr=0.000537
  lr=$(python3 -c "print(f'{$base_lr*$lrm:.6g}')")
  sigma=$(python3 -c "import math;print(f'{0.00015*math.sqrt(16/$rank):.6g}')")
  adapter="/shared/zorl/init-adapters/qwen3_6-35b-a3b-r${rank}-eggroll-hybridattn"
  seed=$((9300 + POOL_OFFSET + TRIAL))
  name="SEARCH-${POOL}-t${TRIAL}-r${rank}-b${batch}-p${total_pairs}-lr${lrm}"
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  echo "[$ts][$POOL t$TRIAL] rank=$rank batch=$batch pairs=$total_pairs lr=$lr sigma=$sigma"

  if ! wait_pool_healthy; then
    echo "$ts,$POOL,$TRIAL,$rank,$batch,$pps,$total_pairs,$lr,$sigma,,,N/A,pool_unhealthy" >> "$RESULTS"
    sleep 60; continue
  fi
  # pristine base for a clean per-config baseline: restart pods, wait healthy
  kubectl delete pods -l app="$APP" --wait=false >/dev/null 2>&1
  sleep 20
  if ! wait_pool_healthy; then
    echo "$ts,$POOL,$TRIAL,$rank,$batch,$pps,$total_pairs,$lr,$sigma,,,N/A,restart_failed" >> "$RESULTS"
    continue
  fi

  out=$(python3 controller.py launch --candidate "candidates/${BASE}.yaml" \
        --set-env LORA_RANK=$rank --set-env LORA_ALPHA=$rank \
        --set-env ADAPTER_DIR="$adapter" \
        --set-env TRAIN_SIZE=$batch --set-env PAIRS_PER_SHARD=$pps \
        --set-env LEARNING_RATE=$lr --set-env B_SIGMA=$sigma \
        --set-env SEED=$seed --set-env WANDB_NAME="$name" 2>&1)
  job=$(echo "$out" | grep -oE "zorl-ar-search-base-[a-z0-9-]+" | head -1)
  if [ -z "$job" ]; then
    echo "$ts,$POOL,$TRIAL,$rank,$batch,$pps,$total_pairs,$lr,$sigma,,,launch_failed,launch_failed" >> "$RESULTS"
    echo "  launch failed: $(echo "$out"|tail -1)"; continue
  fi
  echo "  launched $job (1h cap)"
  # wait for terminal state (job self-caps at 3600s; +900s slack for startup/probe)
  jdeadline=$((SECONDS + 5400))
  while true; do
    st=$(kubectl get job "$job" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
    [ -n "$st" ] && break
    [ $SECONDS -gt $jdeadline ] && { st="watchdog_timeout"; break; }
    sleep 60
  done
  log=$(kubectl logs job/"$job" --tail=2000 2>/dev/null)
  peak=$(echo "$log" | grep -oE "probe_reward=[0-9.]+" | grep -oE "[0-9.]+" | sort -rn | head -1)
  nprobes=$(echo "$log" | grep -c "probe_reward=")
  echo "$(date -u +%Y%m%dT%H%M%SZ),$POOL,$TRIAL,$rank,$batch,$pps,$total_pairs,$lr,$sigma,${peak:-NA},${nprobes:-0},$job,$st" >> "$RESULTS"
  echo "  trial done: peak=${peak:-NA} status=$st"
  kubectl delete job "$job" --wait=false >/dev/null 2>&1
done
