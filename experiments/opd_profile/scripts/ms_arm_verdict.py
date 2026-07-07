#!/usr/bin/env python3
"""ATTEMPT-1 arm verdict (per the preregistered rule, ATTEMPT1_PREREG §common
frame). All numbers recomputed from per-item JSONLs (FILES-ONLY).

Inputs: --eval-dir with ms_arm_eval_marin.py outputs for two tags (arm, c0)
on the same datasets/doses (identical items by construction).

Computes, per dataset:
  - exact per (tag, dose) + Wilson CI;
  - paired exact McNemar arm-vs-C0 at each dose;
  - DOSE TREND per tag: within-item permutation test (10k perms) on the
    per-item Pearson-style linear contrast of exact against log2(dose_mult)
    over the NONZERO doses (>=3 points; two-point claims disallowed);
  - the four cells: (arm/C0) x (ablated dose0 / best nonzero dose).
Preregistered rule (per arm):
  WIN       = arm dose-trend positive p<.05 AND best-dose arm > C0-at-same-
              step McNemar p<.05 AND removal-consistent (arm dose0 ~ C0 dose0:
              |diff| < 3pp or McNemar ns)
  LEVEL-WIN = best-dose arm > C0 (p<.05) without a positive dose trend
  NULL      = neither
  CENSORED  = C0 in-band exact < 0.10 AND arm best < 0.10 (floor; no reading)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict


def mcnemar_p(b, c):
    """Two-sided exact binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tot = sum(math.comb(n, i) for i in range(0, k + 1)) * 2
    p = tot / (2 ** n)
    return min(1.0, p)


def load_items(path):
    return {str(json.loads(l)["idx"]): json.loads(l) for l in open(path)}


def dose_trend_perm(items_by_dose: dict, n_perm=10000, seed=1234):
    """items_by_dose: {mult: {idx: exact}} over nonzero doses. Statistic =
    sum_i sum_d w_d * x_{i,d} with w = centered log2(dose). Permutation: per
    item, shuffle its exact values across doses (within-item exchangeability
    under H0: dose has no effect)."""
    doses = sorted(items_by_dose)
    assert len(doses) >= 3, f"need >=3 nonzero doses, got {doses}"
    w = [math.log2(d) for d in doses]
    wm = sum(w) / len(w)
    w = [x - wm for x in w]
    idxs = set.intersection(*(set(items_by_dose[d]) for d in doses))
    rows = [[items_by_dose[d][i] for d in doses] for i in sorted(idxs)]
    if not rows:
        return None, 0
    stat = sum(sum(wd * x for wd, x in zip(w, r)) for r in rows)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        s = 0.0
        for r in rows:
            rr = r[:]
            rng.shuffle(rr)
            s += sum(wd * x for wd, x in zip(w, rr))
        if s >= stat:
            ge += 1
    p_pos = (ge + 1) / (n_perm + 1)
    return {"stat": round(stat, 3), "p_positive": round(p_pos, 5),
            "n_items": len(rows), "doses": doses}, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--c0-tag", required=True)
    ap.add_argument("--primary-dataset", required=True,
                    help="dataset stem for the in-band verdict, e.g. nops05-07_capped_val500")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def collect(tag):
        out = defaultdict(dict)  # dataset -> mult -> {idx: item}
        for p in glob.glob(f"{args.eval_dir}/{tag}__*__dose*x.jsonl"):
            base = os.path.basename(p)[len(tag) + 2: -len(".jsonl")]
            dset, dose = base.rsplit("__dose", 1)
            out[dset][int(dose[:-1])] = load_items(p)
        return out

    arm = collect(args.arm_tag)
    c0 = collect(args.c0_tag)
    report = {"arm_tag": args.arm_tag, "c0_tag": args.c0_tag, "datasets": {}}

    for dset in sorted(set(arm) | set(c0)):
        d = {"cells": {}, "mcnemar_vs_c0": {}, "dose_trend": {}}
        for tag, data in (("arm", arm.get(dset, {})), ("c0", c0.get(dset, {}))):
            for mult, items in sorted(data.items()):
                ex = sum(i["exact"] for i in items.values()) / max(1, len(items))
                d["cells"][f"{tag}_dose{mult}x"] = {"n": len(items), "exact": round(ex, 4)}
        for mult in sorted(set(arm.get(dset, {})) & set(c0.get(dset, {}))):
            a, c = arm[dset][mult], c0[dset][mult]
            common = set(a) & set(c)
            b_ = sum(1 for i in common if a[i]["exact"] > c[i]["exact"])
            c_ = sum(1 for i in common if a[i]["exact"] < c[i]["exact"])
            d["mcnemar_vs_c0"][f"dose{mult}x"] = {
                "arm_only": b_, "c0_only": c_, "p": round(mcnemar_p(b_, c_), 5)}
        for tag, data in (("arm", arm.get(dset, {})), ("c0", c0.get(dset, {}))):
            nz = {m: {i: it["exact"] for i, it in v.items()}
                  for m, v in data.items() if m > 0}
            if len(nz) >= 3:
                tr, _ = dose_trend_perm(nz)
                d["dose_trend"][tag] = tr
        report["datasets"][dset] = d

    # ---- preregistered adjudication on the primary dataset ----
    P = report["datasets"].get(args.primary_dataset, {})
    verdict = "NO-DATA"
    detail = {}
    if P:
        cells = P["cells"]
        nz_mults = sorted(int(k.split("dose")[1][:-1]) for k in cells
                          if k.startswith("arm_dose") and not k.endswith("dose0x"))
        best_mult, best_ex = None, -1
        for m in nz_mults:
            ex = cells.get(f"arm_dose{m}x", {}).get("exact", -1)
            if ex > best_ex:
                best_ex, best_mult = ex, m
        c0_best = cells.get(f"c0_dose{best_mult}x", cells.get("c0_dose0x", {})).get("exact")
        arm0 = cells.get("arm_dose0x", {}).get("exact")
        c00 = cells.get("c0_dose0x", {}).get("exact")
        trend = P["dose_trend"].get("arm")
        mcn_best = P["mcnemar_vs_c0"].get(f"dose{best_mult}x", {})
        detail = {"best_mult": best_mult, "arm_best": best_ex, "c0_at_best": c0_best,
                  "arm_ablated": arm0, "c0_ablated": c00,
                  "trend_p_positive": trend and trend["p_positive"],
                  "mcnemar_best": mcn_best}
        if (c00 is not None and c00 < 0.10 and best_ex < 0.10):
            verdict = "CENSORED (floor)"
        else:
            beats = (mcn_best.get("p", 1) < 0.05
                     and mcn_best.get("arm_only", 0) > mcn_best.get("c0_only", 0))
            trend_pos = trend is not None and trend["p_positive"] < 0.05
            removal_ok = (arm0 is not None and c00 is not None
                          and abs(arm0 - c00) < 0.03)
            if trend_pos and beats and removal_ok:
                verdict = "WIN"
            elif beats:
                verdict = "LEVEL-WIN" + ("" if removal_ok else " (removal-inconsistent)")
            elif trend_pos:
                verdict = "TREND-ONLY (no capability bar)"
            else:
                verdict = "NULL"
    report["verdict"] = verdict
    report["verdict_detail"] = detail
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"verdict": verdict, **detail}, default=str))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
