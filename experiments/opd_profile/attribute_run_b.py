"""Compare Run B steady-state profile to Run A baseline.

Usage:
    python3 attribute_run_b.py <run_b_profile.jsonl> [--last N]

Prints a per-component table: Run A median (last 20 steps) vs Run B median
(last N rows, default 5), with delta and pct-of-step. Highlights the new
bottleneck.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


COMPONENTS = [
    ("step_total_s", "total"),
    ("prepare_window_s", "prepare wall"),
    ("forward_backward_s", "fwd_bwd wall"),
    ("student_sampling_s", "student sum"),
    ("teacher_prefill_s", "teacher sum"),
    ("teacher_prefill_forward_compute_s", "teacher compute"),
    ("teacher_hidden_cache_write_s", "teacher cache write"),
    ("opd_profile_forward_compute_s", "trainer fwd"),
    ("opd_profile_backward_compute_s", "trainer bwd"),
    ("opd_profile_kl_compute_s", "trainer KL"),
    ("opd_profile_loss_compute_s", "trainer loss"),
    ("sync_inference_weights_s", "sync"),
    ("optim_step_s", "optim"),
    ("optim_step_queued_s", "optim queue"),
]

RUN_A_PROFILE = (
    "experiments/encoded_reasoning/results/qwen3_30b_a3b_full_weight_real_reward/"
    "er-opda-052700/20260527T093543Z-er-opda-052700-trainer-head-8bpnx/opd_profile.jsonl"
)


def steady_state_rows(path: str, last: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if not r.get("profile_warmup", False)]
    return rows[-last:] if last and len(rows) >= last else rows


def median(rows: Iterable[dict], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows if key in r]
    return statistics.median(vals) if vals else 0.0


def fmt(s: float) -> str:
    if s >= 100:
        return f"{s:>8.1f}s"
    if s >= 10:
        return f"{s:>8.2f}s"
    return f"{s:>8.3f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_b_profile", help="path to Run B opd_profile.jsonl")
    ap.add_argument("--last", type=int, default=5, help="steady-state row count (default 5)")
    ap.add_argument("--run-a", default=RUN_A_PROFILE, help="Run A baseline profile")
    args = ap.parse_args()

    run_a_rows = steady_state_rows(args.run_a, last=20)
    run_b_rows = steady_state_rows(args.run_b_profile, last=args.last)

    print(
        f"Run A: {len(run_a_rows)} rows from {Path(args.run_a).name}\n"
        f"Run B: {len(run_b_rows)} rows from {Path(args.run_b_profile).name}\n"
    )

    a_total = median(run_a_rows, "step_total_s")
    b_total = median(run_b_rows, "step_total_s")
    a_prepare = median(run_a_rows, "prepare_s")
    b_prepare = median(run_b_rows, "prepare_s")

    print(f"{'component':<22} {'Run A':>10} {'Run B':>10} {'delta':>10} {'B % of step':>12}")
    print("-" * 72)
    for key, label in COMPONENTS:
        a = median(run_a_rows, key)
        b = median(run_b_rows, key)
        d = b - a
        pct = (b / b_total * 100) if b_total > 0 else 0
        # Mark large deltas
        flag = ""
        if abs(d) / max(a, 1.0) > 0.20:
            flag = " ★"
        print(f"{label:<22} {fmt(a)} {fmt(b)} {fmt(d):>10} {pct:>10.1f}%{flag}")

    print()
    print(f"step_total:   Run A {a_total:.1f}s → Run B {b_total:.1f}s "
          f"(delta {b_total - a_total:+.1f}s = {(b_total/a_total - 1) * 100:+.1f}%)")
    print(f"prepare_sum:  Run A {a_prepare:.1f}s → Run B {b_prepare:.1f}s "
          f"(prepare wall: A={median(run_a_rows, 'prepare_window_s'):.1f}s "
          f"B={median(run_b_rows, 'prepare_window_s'):.1f}s)")

    # Tokens + throughput
    a_tt = median(run_a_rows, "teacher_prefill_tokens")
    b_tt = median(run_b_rows, "teacher_prefill_tokens")
    a_ttps = median(run_a_rows, "teacher_prefill_tok_per_s")
    b_ttps = median(run_b_rows, "teacher_prefill_tok_per_s")
    a_st = median(run_a_rows, "student_sampling_output_tokens")
    b_st = median(run_b_rows, "student_sampling_output_tokens")
    a_nb = median(run_a_rows, "num_prepare_batches")
    b_nb = median(run_b_rows, "num_prepare_batches")
    print()
    print(f"teacher tokens/step:    A {a_tt:>10.0f}  B {b_tt:>10.0f}")
    print(f"teacher tok/s (sum):    A {a_ttps:>10.0f}  B {b_ttps:>10.0f}")
    print(f"student tokens/step:    A {a_st:>10.0f}  B {b_st:>10.0f}")
    print(f"prepare batches/step:   A {a_nb:>10.0f}  B {b_nb:>10.0f}")

    # Per-prep avg
    if b_nb > 0:
        b_avg_teacher_s = median(run_b_rows, "teacher_prefill_s") / b_nb
        b_avg_teacher_compute_s = median(run_b_rows, "teacher_prefill_forward_compute_s") / b_nb
        b_avg_student_s = median(run_b_rows, "student_sampling_s") / b_nb
        print()
        print(f"Run B per-prepare avg:")
        print(f"  teacher_s            {b_avg_teacher_s:.2f}s")
        print(f"  teacher_compute_s    {b_avg_teacher_compute_s:.2f}s")
        print(f"  teacher overhead     {b_avg_teacher_s - b_avg_teacher_compute_s:.2f}s "
              f"({(1 - b_avg_teacher_compute_s/b_avg_teacher_s)*100:.0f}% of teacher_s)")
        print(f"  student_s            {b_avg_student_s:.2f}s")


if __name__ == "__main__":
    main()
