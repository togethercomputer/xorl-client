#!/usr/bin/env bash
# Launch filler sweeps inside the 235B sglang pod (repo mounted at /home/apanda).
# Usage: run_filler_sweeps.sh {smoke|F1|F2|F3} [pod] [served-model]
set -uo pipefail
PHASE="${1:?phase}"
POD="${2:-flx-q3-235b-base-sglang-0}"
MODEL="${3:-Qwen/Qwen3-235B-A22B}"
PY=/home/apanda/xorl-sglang-internal/.venv/bin/python
S=experiments/opd_profile/eval_filler_sweep.py

run() {  # run-id + extra args
  local rid="$1"; shift
  echo "### $rid : $*"
  kubectl exec -n apanda "$POD" -c sglang -- bash -lc "cd /home/apanda/xorl-opd-prefill && \
    $PY $S --host 127.0.0.1 --port 30060 --model '$MODEL' --run-id '$rid' $*"
}

case "$PHASE" in
  smoke)
    run smoke --tasks mult:4 --regimes raw_nothink --filler-kind pause \
      --filler-counts 0,100 --nprob 8 --ksamp 2 --workers 16 ;;
  F1)  # filler-budget K sweep on the known cell
    run F1-budget-base-mult4 --tasks mult:4 --regimes raw_nothink,chat_hardoff \
      --filler-kind pause --filler-counts 0,25,50,100,200,400 --nprob 200 --ksamp 8 --workers 96 ;;
  F2)  # filler content-type sweep at K=100 AND K=400 (F1 found lift grows with budget).
       # Strong/bounded kinds first; fibonacci EXCLUDED (unbounded element size -> ctx overflow at K=400;
       # fib@K100 already salvaged from the v1 partial = 0.562 raw, ~base).
    run F2-content-base-mult4 --tasks mult:4 --regimes raw_nothink,chat_hardoff \
      --filler-kinds nato,random_numbers,pause,counting,lorem,states \
      --filler-counts 0,100,400 --nprob 200 --ksamp 8 --workers 96 ;;
  F3)  # new task families, strongest filler, find goal-(b) targets (headroom p@8>>p@1, filler-flat).
       # Key readout: base p@1 vs base p@8 per task. Trimmed to the hard families + n=150 for speed.
    run F3-newtasks-base --tasks mult:5,mult:6,add:16:3,add:24:3,prod3:3,poly:3 \
      --regimes raw_nothink,chat_hardoff --filler-kind nato \
      --filler-counts 0,100,400 --nprob 150 --ksamp 8 --workers 96 ;;
  F4)  # HIGH-N (n=1000) confirmation of the pass@8 rise at the strongest cell (pause K400, both regimes).
       # base+filler measured IN THE SAME RUN (run-to-run base variance ~0.10 -> only within-run Δ trustworthy).
    run F4-confirm-pass8 --tasks mult:4 --regimes raw_nothink,chat_hardoff \
      --filler-kinds pause --filler-counts 0,400 --nprob 1000 --ksamp 8 --workers 96 ;;
  F5)  # few-shot count x filler interaction (does in-context demonstration amplify the lift?)
    for ns in 0 5 10 20; do
      run "F5-shots${ns}-base-mult4" --tasks mult:4 --regimes raw_nothink,chat_hardoff \
        --filler-kinds nato --filler-counts 0,400 --nfewshot "$ns" --nprob 150 --ksamp 8 --workers 96
    done ;;
  F6)  # problem-conditional filler: does filler built from the question's own numbers beat generic?
    run F6-operands-base-mult4 --tasks mult:4 --regimes raw_nothink,chat_hardoff \
      --filler-kinds nato,operands,pause --filler-counts 0,100,400 --nprob 200 --ksamp 8 --workers 96 ;;
  F7)  # confirm the two F3 signals at higher n: mult:5 (filler lifts p@8) + add:16:3 (filler hurts, high-WM)
    run F7-confirm-newtasks --tasks mult:5,add:16:3 --regimes raw_nothink,chat_hardoff \
      --filler-kind nato --filler-counts 0,100,400 --nprob 400 --ksamp 8 --workers 96 ;;
  F8)  # Instruct-2507 at the strong lever (0-shot + 10-shot, K=0,400) — does the zero-lift model lift here?
    for ns in 0 10; do
      run "F8-instruct2507-shots${ns}" --tasks mult:4 --regimes raw_nothink,chat_hardoff \
        --filler-kinds pause,nato --filler-counts 0,400 --nfewshot "$ns" --nprob 200 --ksamp 8 --workers 96
    done ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac
