#!/usr/bin/env python3
"""Filler-token eval client (dual-regime, pass@k) against an SMG router fleet.

Hits the SMG cache_aware router (load-balances across N sglang replicas), runs
the 4dmult few-shot grid in two regimes (chat enable_thinking=False; raw /no_think
completions) x {base, pause} x difficulties, and logs EVERY per-sample result to
samples.jsonl plus an aggregate summary.json. Paper-faithful prefill methodology.

Usage:
  python eval_filler_fleet.py --port 8080 --model Qwen/Qwen3-235B-A22B \
    --run-id q3-235b-base --nprob 100 --ksamp 8 --workers 96
"""
import argparse, json, os, re, random, urllib.request
from concurrent.futures import ThreadPoolExecutor

ANS_RE = re.compile(r"([\-+]?[0-9][0-9,]*)")
DIFFS = {
    "4x4": ("You will be given a 4-digit multiplication problem.", (1000, 9999), (1000, 9999)),
    "5x5": ("You will be given a 5-digit multiplication problem.", (10000, 99999), (10000, 99999)),
    "6x6": ("You will be given a 6-digit multiplication problem.", (100000, 999999), (100000, 999999)),
}


def sysp(desc):
    return f"/no_think\n{desc}\n\nFormat:\nAnswer: [number]"


def fillerstr(n, kind="pause"):
    if kind == "pause":
        return ("pause " * n).strip()
    if kind == "ellipsis":
        return ("... " * n).strip()
    if kind == "counting":
        return " ".join(str(i) for i in range(1, n + 1))
    return ("pause " * n).strip()


def render_raw(msgs, prefill):
    parts = []
    for idx, m in enumerate(msgs):
        nl = "\n" if idx > 0 else ""
        parts.append(f"{nl}<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append(f"\n<|im_start|>assistant\n{prefill}")
    return "".join(parts)


def post(url, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--nprob", type=int, default=100)
    ap.add_argument("--ksamp", type=int, default=8)
    ap.add_argument("--nfewshot", type=int, default=10)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--regimes", default="chat_hardoff,raw_nothink")
    ap.add_argument("--difficulties", default="4x4,5x5,6x6")
    ap.add_argument("--filler", default="pause")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default="experiments/opd_profile/results/filler_fleet")
    args = ap.parse_args()

    chat_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    comp_url = f"http://{args.host}:{args.port}/v1/completions"
    out_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    sf = open(samples_path, "w")

    regimes = args.regimes.split(",")
    diffs = args.difficulties.split(",")
    rng = random.Random(args.seed)
    # fixed problem pools per difficulty (deterministic from seed)
    pools = {}
    for d in diffs:
        _desc, ra, rb = DIFFS[d]
        pools[d] = [(rng.randint(*ra), rng.randint(*rb)) for _ in range(args.nprob + args.nfewshot)]

    def call(regime, desc, fs, a, b, use_filler, seed):
        f = fillerstr(100, args.filler) if use_filler else ""
        msgs = [{"role": "system", "content": sysp(desc)}]
        for (x, y) in fs:
            asst = f"{f}\nAnswer: {x*y}" if f else f"Answer: {x*y}"
            msgs.append({"role": "user", "content": f"What is {x} * {y}?"})
            msgs.append({"role": "assistant", "content": asst})
        msgs.append({"role": "user", "content": f"What is {a} * {b}?"})
        prefill = f"{f}\nAnswer: " if f else "Answer: "
        if regime == "chat_hardoff":
            m2 = msgs + [{"role": "assistant", "content": prefill}]
            body = {"model": args.model, "messages": m2, "max_tokens": 24, "temperature": 1.0,
                    "seed": seed, "continue_final_message": True, "add_generation_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            return post(chat_url, body)["choices"][0]["message"]["content"] or ""
        else:
            body = {"model": args.model, "prompt": render_raw(msgs, prefill), "max_tokens": 24,
                    "temperature": 1.0, "seed": seed, "stop": ["<|im_end|>"]}
            return post(comp_url, body)["choices"][0]["text"] or ""

    def job(spec):
        regime, d, use_filler, idx, seed = spec
        desc, _ra, _rb = DIFFS[d]
        fs = pools[d][:args.nfewshot]
        a, b = pools[d][args.nfewshot + idx]
        try:
            txt = call(regime, desc, fs, a, b, use_filler, seed)
        except Exception as e:
            return {"regime": regime, "diff": d, "arm": "pause" if use_filler else "base",
                    "a": a, "b": b, "target": a * b, "seed": seed, "completion": None,
                    "correct": False, "error": str(e)[:200]}
        m = ANS_RE.search(txt)
        got = m.group(1).replace(",", "") if m else ""
        return {"regime": regime, "diff": d, "arm": "pause" if use_filler else "base",
                "a": a, "b": b, "target": a * b, "seed": seed, "completion": txt[:80],
                "correct": got == str(a * b), "error": None}

    specs = [(rg, d, uf, idx, s)
             for rg in regimes for d in diffs for uf in (False, True)
             for idx in range(args.nprob) for s in range(args.ksamp)]
    print(f"[{args.run_id}] {len(specs)} requests -> router :{args.port}  ({args.model})")
    agg = {}  # (regime,diff,arm) -> {prod: [bool,...]}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(job, specs):
            sf.write(json.dumps(rec) + "\n")
            key = (rec["regime"], rec["diff"], rec["arm"])
            agg.setdefault(key, {}).setdefault(rec["target"], []).append(rec["correct"])
            done += 1
            if done % 2000 == 0:
                print(f"  ...{done}/{len(specs)}")
    sf.close()

    summary = {"run_id": args.run_id, "model": args.model, "nprob": args.nprob,
               "ksamp": args.ksamp, "cells": []}
    print(f"\n=== {args.run_id}  ({args.model}) ===")
    for rg in regimes:
        print(f"-- regime={rg} --")
        for d in diffs:
            row = {}
            for arm in ("base", "pause"):
                bp = agg.get((rg, d, arm), {})
                n = len(bp) or 1
                p1 = sum(v[0] for v in bp.values()) / n
                p8 = sum(any(v) for v in bp.values()) / n
                row[arm] = (p1, p8, len(bp))
                summary["cells"].append({"regime": rg, "diff": d, "arm": arm,
                                         "pass1": p1, "pass8": p8, "nprob": len(bp)})
            (b1, b8, bn), (q1, q8, qn) = row["base"], row["pause"]
            print(f"   {d}: base p@1={b1:.3f} p@8={b8:.3f} | pause p@1={q1:.3f}({q1-b1:+.3f}) "
                  f"p@8={q8:.3f}({q8-b8:+.3f}) | n{bn}")
        print()
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"samples -> {samples_path}\nsummary -> {summary_path}\nDONE")


if __name__ == "__main__":
    main()
