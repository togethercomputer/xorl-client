#!/usr/bin/env bash
# Sweep the teacher-prefill bench across {batch_size, concurrency}.
# Set TEACHER_URL + OUT_DIR before invoking.
#
# Usage:
#   TEACHER_URL=http://127.0.0.1:30002 \
#   OUT_DIR=experiments/opd_profile/results/teacher_bench/1n \
#   bash experiments/opd_profile/run_teacher_bench_sweep.sh
set -euo pipefail

TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:30002}"
OUT_DIR="${OUT_DIR:-experiments/opd_profile/results/teacher_bench/$(date -u +%Y%m%dT%H%M%SZ)}"
PROMPTS="${PROMPTS:-/shared/opd-coord/randnum_4digit_filtered_8185.json}"
COT="${COT:-/shared/opd-coord/randnum_4digit_filtered_8185_cot.json}"
TOKENIZER="${TOKENIZER:-/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}"
CACHE_DIR="${CACHE_DIR:-/shared/opd-coord/teacher_bench_cache}"
NUM_BATCHES="${NUM_BATCHES:-8}"
PYTHON="${PYTHON:-/home/apanda/xorl-internal/.venv/bin/python}"
BENCH_SCRIPT=experiments/opd_profile/teacher_prefill_bench.py

mkdir -p "$OUT_DIR" "$CACHE_DIR"

# Default sweep: (batch, conc) pairs
# matching Run B's settings (128, 4) + smaller batch + higher conc variants.
SWEEPS="${SWEEPS:-128x4 128x8 64x8 64x4 32x8}"

echo "=== teacher bench sweep ==="
echo "  teacher_url: $TEACHER_URL"
echo "  out_dir:     $OUT_DIR"
echo "  num_batches: $NUM_BATCHES"
echo "  sweeps:      $SWEEPS"

for pair in $SWEEPS; do
  bs="${pair%x*}"
  cc="${pair#*x}"
  out_json="$OUT_DIR/b${bs}_c${cc}.json"
  log="$OUT_DIR/b${bs}_c${cc}.log"
  echo ""
  echo "=== batch=$bs concurrency=$cc ==="
  "$PYTHON" "$BENCH_SCRIPT" \
    --teacher-url "$TEACHER_URL" \
    --prompts-json "$PROMPTS" \
    --cot-json "$COT" \
    --tokenizer-path "$TOKENIZER" \
    --cache-dir "$CACHE_DIR" \
    --batch-size "$bs" \
    --concurrency "$cc" \
    --num-batches "$NUM_BATCHES" \
    --output-json "$out_json" 2>&1 | tee "$log" | tail -20
done

echo ""
echo "=== sweep done ==="
echo "Summary:"
$PYTHON - <<PY
import json, glob
from pathlib import Path
out_dir = Path("$OUT_DIR")
rows = []
for f in sorted(out_dir.glob("b*_c*.json")):
    s = json.loads(f.read_text())
    rows.append((s["batch_size"], s["concurrency"], s["wall_s"], s["aggregate_tok_per_s"],
                 s["median_batch_s"], s["effective_concurrency"]))
print(f"{'bs':>4} {'cc':>3} {'wall_s':>8} {'tok/s':>8} {'med_b_s':>8} {'eff_cc':>7}")
for bs, cc, wall, tps, med, ec in rows:
    print(f"{bs:>4} {cc:>3} {wall:>8.1f} {tps:>8,.0f} {med:>8.2f} {ec:>7.2f}")
PY
