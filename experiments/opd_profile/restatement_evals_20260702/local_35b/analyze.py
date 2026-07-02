#!/usr/bin/env python3
"""Assemble final tables from the samples jsonls; extra diagnostics for the wrongq control
(does the model answer the RESTATED (wrong) question instead of the original?)."""
import json, math, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vendored_filler_harness as H
from run_restatement_eval import wilson_ci, load_oldgen_problems


def rows_of(path):
    return [json.loads(l) for l in open(path)]


def table(samples_path, problems=None, wrongq_offset=157):
    recs = rows_of(samples_path)
    conds = {}
    for r in recs:
        conds.setdefault(r["condition"], []).append(r)
    out = []
    for c, rs in conds.items():
        ok = [r for r in rs if not r.get("error")]
        k = sum(r["correct"] for r in ok)
        n = len(ok)
        lo, hi = wilson_ci(k, n)
        row = {"condition": c, "acc": k / n if n else float("nan"), "k": k, "n": n,
               "ci": (lo, hi), "err": len(rs) - n}
        if c.startswith("wrongq") and problems is not None:
            N = 300
            m = sum(1 for r in ok
                    if r["pred"] == str(problems[(r["idx"] + wrongq_offset) % N]["answer"]))
            row["answered_wrongq"] = m / n if n else float("nan")
        out.append(row)
    return out


def fmt(rows, order=None):
    if order:
        pos = {c: i for i, c in enumerate(order)}
        rows.sort(key=lambda r: pos.get(r["condition"], 99))
    lines = [f"| condition | acc | 95% CI | n | notes |", "|---|---|---|---|---|"]
    for r in rows:
        note = ""
        if "answered_wrongq" in r:
            note = f"answered the restated-wrong question {r['answered_wrongq']:.3f} of the time"
        if r["err"]:
            note += f" errors={r['err']}"
        lines.append(f"| {r['condition']} | {r['acc']:.3f} | [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['n']} | {note} |")
    return "\n".join(lines)


ORDER = ["B_harness_nofiller_norestate", "B_clean_norestate", "C_restate1_nofiller",
         "D_mega8192_norestate", "A_mega8192_restate", "A_sys100_mega8192_restate",
         "restate2", "restate4", "restate8", "wrongq_restate1", "structured_restate1",
         "mega512_norestate", "mega512_restate",
         "baseline", "restate1", "counting200", "counting200_restate"]

if __name__ == "__main__":
    oldp = load_oldgen_problems(300)[:300]
    hardp = H.generate_variable_reassignment_problems(400, seed=12345)[:300]
    for path in sorted(glob.glob(os.path.join(HERE, "samples_*/**/*_samples.jsonl"), recursive=True)) + \
                sorted(glob.glob(os.path.join(HERE, "samples_*/*_samples.jsonl"))):
        pass
    seen = set()
    for path in sorted(glob.glob(os.path.join(HERE, "samples_*", "*_samples.jsonl"))):
        if path in seen:
            continue
        seen.add(path)
        probs = None
        if "oldgen" in path:
            probs = oldp
        elif "count_reassignments" in os.path.basename(path):
            probs = hardp
        print(f"\n## {os.path.relpath(path, HERE)}")
        print(fmt(table(path, problems=probs), ORDER))
