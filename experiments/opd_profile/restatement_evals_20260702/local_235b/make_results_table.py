#!/usr/bin/env python3
"""Render the condition x task accuracy table (with 95% CIs) from summary JSON."""
import json
import sys

CONDS = ["baseline", "counting", "restate1", "restate2", "restate4", "restate8",
         "counting+restate", "wrongq-restate", "structured-restate"]
TASKS = ["mult4", "arithmetic", "varcount"]


def main(path):
    s = json.load(open(path))
    cells = {(c["task"], c["condition"]): c for c in s["cells"]}
    base = {t: cells.get((t, "baseline")) for t in TASKS}
    print(f"Model: {s['model']}  (nshots={s['nshots']}, seed={s['seed']})\n")
    hdr = "| condition | " + " | ".join(TASKS) + " |"
    print(hdr)
    print("|" + "---|" * (len(TASKS) + 1))
    for cond in CONDS:
        row = [cond]
        for t in TASKS:
            c = cells.get((t, cond))
            if not c or c["n"] == 0:
                row.append("--")
                continue
            cell = f"{100*c['acc']:.1f} ±{100*c['ci95']:.1f}"
            b = base.get(t)
            if b and cond != "baseline":
                cell += f" ({100*(c['acc']-b['acc']):+.1f})"
            if c.get("n_err"):
                cell += f" [err={c['n_err']}]"
            row.append(cell)
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
