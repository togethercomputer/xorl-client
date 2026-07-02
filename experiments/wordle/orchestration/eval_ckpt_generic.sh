#!/bin/bash
# Generic multi-turn floor-eval of ANY checkpoint: prepare serve+sync (mem 0.70 + REAL sync
# detection) -> sharded held-out [0:NG] multi-turn eval -> teardown. Reads transcripts after.
# Usage: eval_ckpt_generic.sh <CKPT_DIR> <LABEL> <CONFIG_YAML> [NG] [REPLICAS]
set -uo pipefail
CKPT="${1:?ckpt dir}"; LABEL="${2:?label}"; CFG="${3:?config yaml}"; NG="${4:-64}"; R="${5:-2}"
PY=/home/apanda/xorl-internal/.venv/bin/python
ROOT=/shared/apanda/wordle-sft-runs
[ -d "$CKPT" ] || { echo "[$LABEL] no ckpt $CKPT"; exit 1; }
echo "=== [$LABEL] prepare serve+sync ($CKPT, cfg=$(basename $CFG), R=$R) ==="
OUT=$("$PY" "$ROOT/prepare_checkpoint_eval_serving.py" "$CKPT" --label "$LABEL" --replicas "$R" --config-path "$CFG" --apply 2>&1)
APP=$(echo "$OUT" | grep -oE "statefulset.apps/[a-z0-9-]+" | head -1 | cut -d/ -f2)
SERVE="$ROOT/eval-serving/$LABEL"
echo "[$LABEL] app=$APP"
# lower replica mem to 0.70 (the weight-apply OOMs at 0.85)
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
# tail the sync POD logs into sync_job.log (prepare does NOT write it) so the wait-loop works
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
  echo "=== [$LABEL] multi-turn held-out eval (NG=$NG) ==="
  "$PY" "$ROOT/shard_eval.py" --app "$APP" --replicas "$R" --num-games "$NG" --offset 0 --label "$LABEL"
  echo "=== [$LABEL] failure taxonomy -> TAXONOMY.json ==="
  "$PY" "$ROOT/eval_failure_taxonomy.py" "$LABEL" || true
  echo "=== [$LABEL] behavioral panel (behavior+reasoning+optimal-play+regret) -> BEHAVIOR.json ==="
  "$PY" "$ROOT/wordle_panel.py" --eval "$LABEL" || true
fi
echo "=== [$LABEL] teardown ==="
kubectl delete statefulset "$APP" -n apanda 2>/dev/null | head -1
kubectl delete svc "${APP}-headless" -n apanda 2>/dev/null | head -1
kubectl delete job -n apanda $(kubectl get jobs -n apanda 2>/dev/null | grep "${APP}-sync" | awk '{print $1}') 2>/dev/null | head -1
echo "[$LABEL] done"
