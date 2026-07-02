#!/usr/bin/env python3
"""Filler-token sweep eval: parametrized filler budget K + multiple task families.

Generalizes eval_filler_fleet.py (paper-faithful 10-shot prefill methodology) to:
  - sweep the filler budget K (--filler-counts "0,25,50,100,200"; 0 == base/no-filler);
  - support several arithmetic task families (--tasks "mult:4,mult:5,add:8:4,prod3:3,poly:3");
  - report pass@1 (mean per-sample accuracy + SE over problems) AND pass@k (any-of-k),
    so we can see whether filler moves p@1 toward a FIXED p@k ceiling or RAISES the ceiling.

Hit a single sglang replica directly (run inside the worker pod): --host 127.0.0.1 --port 30060.

Task family specs (question text -> single integer answer):
  mult:N        N-digit x N-digit multiplication            "What is a * b?"
  add:M:K       sum of M K-digit numbers                    "What is n1 + n2 + ... + nM?"
  prod3:K       product of 3 K-digit numbers                "What is a * b * c?"
  poly:K        a*b + c*d with K-digit operands             "What is a*b + c*d?"
"""
import argparse, json, math, os, re, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filler_content as fc  # vendored from tomi harness (faithful nato/fibonacci/random/...)

ANS_RE = re.compile(r"([\-+]?[0-9][0-9,]*)")


def _rng_int(rng, k):
    lo = 10 ** (k - 1)
    hi = 10 ** k - 1
    return rng.randint(lo, hi)


def make_task(spec):
    """Return (desc, gen) where gen(rng) -> (question_str, answer_int)."""
    parts = spec.split(":")
    fam = parts[0]
    if fam == "mult":
        k = int(parts[1])
        desc = f"You will be given a {k}-digit multiplication problem."
        def gen(rng):
            a, b = _rng_int(rng, k), _rng_int(rng, k)
            return f"What is {a} * {b}?", a * b
        return desc, gen
    if fam == "add":
        m, k = int(parts[1]), int(parts[2])
        desc = f"You will be given an addition problem with {m} numbers."
        def gen(rng):
            ns = [_rng_int(rng, k) for _ in range(m)]
            return "What is " + " + ".join(str(n) for n in ns) + "?", sum(ns)
        return desc, gen
    if fam == "prod3":
        k = int(parts[1])
        desc = "You will be given a product of three numbers."
        def gen(rng):
            a, b, c = _rng_int(rng, k), _rng_int(rng, k), _rng_int(rng, k)
            return f"What is {a} * {b} * {c}?", a * b * c
        return desc, gen
    if fam == "poly":
        k = int(parts[1])
        desc = "You will be given an arithmetic expression."
        def gen(rng):
            a, b, c, d = (_rng_int(rng, k) for _ in range(4))
            return f"What is {a}*{b} + {c}*{d}?", a * b + c * d
        return desc, gen
    raise ValueError(f"unknown task family: {spec}")


def sysp(desc):
    return f"/no_think\n{desc}\n\nFormat:\nAnswer: [number]"


def fillerstr(n, kind="pause"):
    """One deterministic filler string of n elements (seeded so random-content types
    are reproducible + thread-safe when precomputed before the pool)."""
    if n <= 0:
        return ""
    random.seed((hash((kind, n)) & 0x7FFFFFFF))
    return fc.generate_filler_tokens(n, token_type=kind)


def render_raw(msgs, prefill):
    parts = []
    for idx, m in enumerate(msgs):
        nl = "\n" if idx > 0 else ""
        parts.append(f"{nl}<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append(f"\n<|im_start|>assistant\n{prefill}")
    return "".join(parts)


def post(url, body, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30060)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tasks", default="mult:4")
    ap.add_argument("--filler-counts", default="0,100")
    ap.add_argument("--filler-kind", default="pause")
    ap.add_argument("--filler-kinds", default="", help="CSV of filler content types to sweep (overrides --filler-kind)")
    ap.add_argument("--nprob", type=int, default=200)
    ap.add_argument("--ksamp", type=int, default=8)
    ap.add_argument("--nfewshot", type=int, default=10)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--regimes", default="raw_nothink,chat_hardoff")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default="experiments/opd_profile/results/filler_sweep")
    args = ap.parse_args()

    chat_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    comp_url = f"http://{args.host}:{args.port}/v1/completions"
    out_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    sf = open(os.path.join(out_dir, "samples.jsonl"), "w")

    regimes = args.regimes.split(",")
    tasks = args.tasks.split(",")
    fcounts = [int(x) for x in args.filler_counts.split(",")]
    kinds = [k for k in args.filler_kinds.split(",") if k] or [args.filler_kind]
    # deterministic problem pool per task (few-shot demos + test problems)
    pools, descs = {}, {}
    for t in tasks:
        desc, gen = make_task(t)
        descs[t] = desc
        pr = random.Random(hash((args.seed, t)) & 0xFFFFFFFF)
        pools[t] = [gen(pr) for _ in range(args.nprob + args.nfewshot)]

    # precompute one filler string per (kind,count); fcount==0 -> "" (kind-independent base)
    # precompute static filler per (kind,count); "operands" is per-question (handled in call())
    FILLER = {(k, fc_n): ("" if fc_n == 0 else fillerstr(fc_n, k))
              for k in kinds if k != "operands" for fc_n in fcounts}

    def _operand_filler(qtext, n):
        # problem-conditional filler: the question's own numbers, repeated to ~n elements
        nums = re.findall(r"-?\d+", qtext)
        if not nums:
            return ""
        out = []
        i = 0
        while len(out) < n:
            out.append(nums[i % len(nums)]); i += 1
        return " ".join(out)

    def call(regime, desc, fs, q, kind, fcount, seed):
        if fcount == 0:
            f = ""
        elif kind == "operands":
            f = _operand_filler(q, fcount)  # per-question filler (built per demo too, below)
        else:
            f = FILLER[(kind, fcount)]
        msgs = [{"role": "system", "content": sysp(desc)}]
        for (fq, fa) in fs:
            fdemo = _operand_filler(fq, fcount) if (kind == "operands" and fcount) else f
            asst = f"{fdemo}\nAnswer: {fa}" if fdemo else f"Answer: {fa}"
            msgs.append({"role": "user", "content": fq})
            msgs.append({"role": "assistant", "content": asst})
        msgs.append({"role": "user", "content": q})
        prefill = f"{f}\nAnswer: " if f else "Answer: "
        if regime == "chat_hardoff":
            m2 = msgs + [{"role": "assistant", "content": prefill}]
            body = {"model": args.model, "messages": m2, "max_tokens": args.max_tokens,
                    "temperature": 1.0, "seed": seed, "continue_final_message": True,
                    "add_generation_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            return post(chat_url, body)["choices"][0]["message"]["content"] or ""
        body = {"model": args.model, "prompt": render_raw(msgs, prefill),
                "max_tokens": args.max_tokens, "temperature": 1.0, "seed": seed,
                "stop": ["<|im_end|>"]}
        return post(comp_url, body)["choices"][0]["text"] or ""

    def job(spec):
        regime, t, kind, fcount, idx, s = spec
        fs = pools[t][:args.nfewshot]
        q, ans = pools[t][args.nfewshot + idx]
        rec = {"regime": regime, "task": t, "kind": kind, "fcount": fcount, "idx": idx, "seed": s,
               "target": ans, "completion": None, "correct": False, "error": None}
        try:
            txt = call(regime, descs[t], fs, q, kind, fcount, s)
            m = ANS_RE.search(txt)
            got = m.group(1).replace(",", "") if m else ""
            rec["completion"] = txt[:80]
            rec["correct"] = (got == str(ans))
        except Exception as e:
            rec["error"] = str(e)[:200]
        return rec

    # build cells: fcount==0 -> a single shared "base" cell (kind label "base"); else per kind
    cells = []
    for rg in regimes:
        for t in tasks:
            if 0 in fcounts:
                cells.append((rg, t, "base", 0))
            for k in kinds:
                for fc in fcounts:
                    if fc != 0:
                        cells.append((rg, t, k, fc))
    specs = [(rg, t, k, fc, idx, s) for (rg, t, k, fc) in cells
             for idx in range(args.nprob) for s in range(args.ksamp)]
    print(f"[{args.run_id}] {len(specs)} requests, {len(cells)} cells -> {args.host}:{args.port} ({args.model})", flush=True)
    # agg[(regime,task,kind,fcount)][target] = [correct,...]
    agg, done, errs = {}, 0, 0
    # out-of-order (as_completed) so one slow/hung request can't wedge aggregation
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for fut in as_completed(futs):
            rec = fut.result()
            sf.write(json.dumps(rec) + "\n")
            agg.setdefault((rec["regime"], rec["task"], rec["kind"], rec["fcount"]), {}) \
               .setdefault(rec["target"], []).append(rec["correct"])
            done += 1
            errs += 1 if rec["error"] else 0
            if done % 2000 == 0:
                print(f"  ...{done}/{len(specs)} (errs={errs})", flush=True)
    sf.close()

    def stats(cellkey):
        bp = agg.get(cellkey, {})
        n = len(bp) or 1
        per_prob = [sum(v) / len(v) for v in bp.values()]  # mean over k samples per problem
        p1 = sum(per_prob) / n                              # pass@1 = single-sample success prob
        var = sum((x - p1) ** 2 for x in per_prob) / max(1, n - 1)
        se = math.sqrt(var / n)
        pk = sum(1 for v in bp.values() if any(v)) / n      # pass@k = any-of-k per problem
        return p1, se, pk, len(bp)

    summary = {"run_id": args.run_id, "model": args.model, "nprob": args.nprob,
               "ksamp": args.ksamp, "kinds": kinds, "errs": errs, "cells": []}
    print(f"\n=== {args.run_id} ({args.model})  kinds={kinds} errs={errs} ===", flush=True)
    pk_lbl = "p@" + str(args.ksamp)
    for rg in regimes:
        print(f"-- regime={rg} --", flush=True)
        print(f"   {'task':<10} {'kind':<14} {'K':>5} {'p@1':>7} {'se':>6} {'Δp@1':>7} {pk_lbl:>7} {'Δ'+pk_lbl:>7} {'n':>5}", flush=True)
        for t in tasks:
            b1, bse, bpk, bn = stats((rg, t, "base", 0)) if (rg, t, "base", 0) in agg else (float("nan"),) * 4
            if (rg, t, "base", 0) in agg:
                summary["cells"].append({"regime": rg, "task": t, "kind": "base", "fcount": 0,
                                         "pass1": b1, "se": bse, "passk": bpk, "nprob": bn})
                print(f"   {t:<10} {'base':<14} {0:>5} {b1:>7.3f} {bse:>6.3f} {'':>7} {bpk:>7.3f} {'':>7} {bn:>5}", flush=True)
            for k in kinds:
                for fc in fcounts:
                    if fc == 0:
                        continue
                    p1, se, pk, nn = stats((rg, t, k, fc))
                    summary["cells"].append({"regime": rg, "task": t, "kind": k, "fcount": fc,
                                             "pass1": p1, "se": se, "passk": pk, "nprob": nn})
                    dp1 = p1 - b1 if b1 == b1 else float("nan")
                    dpk = pk - bpk if bpk == bpk else float("nan")
                    print(f"   {t:<10} {k:<14} {fc:>5} {p1:>7.3f} {se:>6.3f} {dp1:>+7.3f} {pk:>7.3f} {dpk:>+7.3f} {nn:>5}", flush=True)
        print(flush=True)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"samples + summary -> {out_dir}\nDONE", flush=True)


if __name__ == "__main__":
    main()
