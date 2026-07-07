#!/usr/bin/env python3
"""Gate (i) verdict: does the features-off marin sandbox run reproduce the
banked recur_stage0 C0 arm's loss curve?

Preregistered tolerances (MECHANISM_SANDBOX_20260706.md §gates):
  - step-0 ce_mean |delta| <= 0.02  (same weights, same batch; residual =
    kernel-shape noise: the sandbox runs ONE full-sequence pass where recur's
    phase-split ran prefix/tail separately — same math, different SDPA tiles)
  - mean |delta ce_mean| over steps 0-49 <= 0.05
  - final val CE per band |delta| <= 0.10 (weight-trajectory noise compounds;
    B1 same-seed relaunch precedent)
All numbers recomputed from the two JSONL logs (FILES-ONLY).
"""
from __future__ import annotations

import argparse
import json


def load(path, kind):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("kind") == kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/shared/apanda/filler_grpo/recur_stage0/logs/train_c0_sub.jsonl")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_t = {r["step"]: r for r in load(args.ref, "train")}
    new_t = {r["step"]: r for r in load(args.new, "train")}
    common = sorted(set(ref_t) & set(new_t))
    assert common, "no overlapping steps"
    deltas = {s: abs(ref_t[s]["ce_mean"] - new_t[s]["ce_mean"]) for s in common}
    d0 = deltas.get(0)
    early = [deltas[s] for s in common if s < 50]
    mean_early = sum(early) / len(early)

    ref_v = load(args.ref, "val")
    new_v = load(args.new, "val")
    last_ref = ref_v[-1] if ref_v else {}
    last_new = new_v[-1] if new_v else {}
    val_deltas = {}
    for k, v in last_ref.items():
        if k.startswith("d") and k in last_new:
            val_deltas[k] = abs(v - last_new[k])

    verdict = {
        "n_common_steps": len(common),
        "step0_delta": d0,
        "mean_abs_delta_steps_0_49": mean_early,
        "max_abs_delta_steps_0_49": max(early),
        "mean_abs_delta_all": sum(deltas.values()) / len(deltas),
        "final_val_deltas": val_deltas,
        "ref_final_val": {k: v for k, v in last_ref.items() if k.startswith("d")},
        "new_final_val": {k: v for k, v in last_new.items() if k.startswith("d")},
        "gates": {
            "step0_le_0.02": (d0 is not None and d0 <= 0.02),
            "early_mean_le_0.05": mean_early <= 0.05,
            "final_val_le_0.10": all(v <= 0.10 for v in val_deltas.values()) if val_deltas else None,
        },
    }
    verdict["pass"] = all(v for v in verdict["gates"].values() if v is not None)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
