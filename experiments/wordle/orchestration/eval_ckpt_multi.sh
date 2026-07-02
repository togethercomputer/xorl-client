#!/bin/bash
# Sync a checkpoint ONCE, then run MANY shard_evals against it (different offset/count/label),
# then teardown. For big-N and multi-try evals without paying the ~7min weight-sync each time.
# Usage: eval_ckpt_multi.sh <CKPT_DIR> <LABEL> <CONFIG_YAML> <REPLICAS> "<spec>"
#   spec = ';'-separated  offset:count:suffix   (shard_eval runs per entry, label=<LABEL>-<suffix>)
#   e.g. "0:170:ho1;0:170:ho2;0:170:ho3;0:170:ho4;0:170:ho5;170:512:train"
# EVAL_TEMP (default 0.7) and EVAL_INVALID_RETRIES (default 0) honored by shard_eval.
set -uo pipefail
CKPT="${1:?ckpt dir}"; LABEL="${2:?label}"; CFG="${3:?config yaml}"; R="${4:-2}"; SPEC="${5:?spec}"
PY=/home/apanda/xorl-internal/.venv/bin/python
ROOT=/shared/apanda/wordle-sft-runs
[ -d "$CKPT" ] || { echo "[$LABEL] no ckpt $CKPT"; exit 1; }
echo "=== [$LABEL] prepare serve+sync ($CKPT, cfg=$(basename $CFG), R=$R) ==="
OUT=$("$PY" "$ROOT/prepare_checkpoint_eval_serving.py" "$CKPT" --label "$LABEL" --replicas "$R" --config-path "$CFG" --apply 2>&1)
APP=$(echo "$OUT" | grep -oE "statefulset.apps/[a-z0-9-]+" | head -1 | cut -d/ -f2)
SERVE="$ROOT/eval-serving/$LABEL"
echo "[$LABEL] app=$APP"
"$PY" - "$SERVE/private_sglang.yaml" <<'PYEOF'
import yaml,sys
p=sys.argv[1]; docs=[d for d in yaml.safe_load_all(open(p)) if d]
for o in docs:
    if o.get("kind")=="StatefulSet":
        sg=next(c for c in o["spec"]["template"]["spec"]["containers"] if c["name"]=="sglang")
        env=sg.setdefault("env",[]); s=False
        for e in env:
            if e.get("name")=="SGLANG_MEM_FRACTION_STATIC": e["value"]="0.70"; s=True
        if not s: env.append({"name":"SGLANG_MEM_FRACTION_STATIC","value":"0.70"})
yaml.safe_dump_all(docs, open(p,"w"), sort_keys=False)
PYEOF
kubectl apply -f "$SERVE/private_sglang.yaml" -n apanda >/dev/null 2>&1
SJ="$SERVE/sync_job.log"
for i in $(seq 1 40); do
  SYNCPOD=$(kubectl get pods -n apanda 2>/dev/null | grep "${APP}-sync" | awk '{print $1}' | head -1)
  [ -n "$SYNCPOD" ] && break; sleep 3
done
echo "[$LABEL] sync pod=$SYNCPOD -> tailing to $SJ"
( kubectl logs -f "$SYNCPOD" -n apanda > "$SJ" 2>&1 ) & TAILPID=$!
echo "=== [$LABEL] wait for REAL sync ==="
ok=0
for i in $(seq 1 240); do
  grep -q "Checkpoint weights are loaded into" "$SJ" 2>/dev/null && { echo "[$LABEL] SYNC OK (~$((i*15))s)"; ok=1; break; }
  grep -q "sync_inference_weights failed" "$SJ" 2>/dev/null && { echo "[$LABEL] SYNC FAILED"; grep -A1 "sync_inference_weights failed" "$SJ"|tail -1|cut -c1-160; break; }
  sleep 15
done
if [ "$ok" = 1 ]; then
  IFS=';' read -ra ENTRIES <<< "$SPEC"
  for ent in "${ENTRIES[@]}"; do
    OFF=$(echo "$ent" | cut -d: -f1); CNT=$(echo "$ent" | cut -d: -f2); SUF=$(echo "$ent" | cut -d: -f3)
    echo "=== [$LABEL-$SUF] shard_eval offset=$OFF num-games=$CNT ==="
    "$PY" "$ROOT/shard_eval.py" --app "$APP" --replicas "$R" --num-games "$CNT" --offset "$OFF" --label "$LABEL-$SUF"
  done
fi
echo "=== [$LABEL] teardown ==="
kubectl delete statefulset "$APP" -n apanda 2>/dev/null | head -1
kubectl delete svc "${APP}-headless" -n apanda 2>/dev/null | head -1
kubectl delete job -n apanda $(kubectl get jobs -n apanda 2>/dev/null | grep "${APP}-sync" | awk '{print $1}') 2>/dev/null | head -1
echo "[$LABEL] done"
