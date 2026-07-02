#!/bin/bash
# Run the paired tightened-stderr eval for the 3 models SERIALLY (one EP8 sync engine at a time —
# 3 concurrent EP8 syncs crashed: NVLink hardware fault on a bad node + concurrent-init storm).
set -uo pipefail
ROOT=/shared/apanda/wordle-sft-runs
CFG=$ROOT/configs/grpo-ep8x1node-muon-lowlr-isr3k3.yaml
SPEC="0:170:ho1;0:170:ho2;0:170:ho3;0:170:ho4;0:170:ho5;170:512:train"
export EVAL_TEMP=0.7 EVAL_INVALID_RETRIES=0
for entry in "bigeval2-9dxtb:server_output_grpo-wq36-1n-mink3" \
             "bigeval2-k3diag:server_output_k3fix_v2" \
             "bigeval2-k3bi:server_output_k3bi"; do
  LBL=${entry%%:*}; DIR=${entry##*:}
  echo "############################## $LBL START $(date -u +%H:%M:%S) ##############################"
  # ABSOLUTE checkpoint path — prepare.py's host_to_pod_path can't map a relative path, so a relative
  # arg lands in the config as-is and the sync pod (different workingDir) can't find it -> "metadata is None".
  "$ROOT/eval_ckpt_multi.sh" "$ROOT/$DIR/weights/default/final" "$LBL" "$CFG" 4 "$SPEC"
  echo "############################## $LBL DONE  $(date -u +%H:%M:%S) ##############################"
done
echo "ALL BIGEVAL DONE"
