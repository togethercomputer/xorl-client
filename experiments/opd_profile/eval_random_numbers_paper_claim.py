#!/usr/bin/env python3
"""Quick paper-claim verification: random_numbers filler at 10-shot V3 4×4.

Paper claim (appendix): Q3-235B-Instruct, 10-shot V3, random_numbers n=100 → +4.1σ
(baseline 0.661, expected ~0.682). SE at our scale (N=100×8=800) is ~1.7pp on p@1,
which makes a +0.7pp lift hard to resolve cleanly — but direction is informative.

Tests:
  base (no filler) vs random_numbers (n=100), at 10-shot
  regimes: raw_nothink (paper) + chat_hardoff (clean)
  difficulty: 4x4 (paper's target cell)
"""
import argparse, json, os, re, random, urllib.request
from concurrent.futures import ThreadPoolExecutor

ANS_RE = re.compile(r"([\-+]?[0-9][0-9,]*)")


def sysp(desc):
    return f"/no_think\n{desc}\n\nFormat:\nAnswer: [number]"


def random_numbers_filler(n_tokens):
    """Generate random-number filler matching paper's spec (space-separated 0-999).
    Each digit is roughly 1 token in Qwen tokenizers, plus a separator. So we
    generate ~n_tokens worth: each number is on average ~2 tokens, so n/2 numbers."""
    rng = random.Random(42)  # deterministic per-call
    out = []
    for _ in range(n_tokens):  # over-generate; truncate by char-length proxy
        out.append(str(rng.randint(0, 999)))
        if sum(len(s) + 1 for s in out) >= n_tokens * 4:  # ~4 chars/token avg
            break
    return " ".join(out)


def render_raw(msgs, prefill):
    parts = []
    for idx, m in enumerate(msgs):
        nl = "\n" if idx > 0 else ""
        parts.append(f"{nl}<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append(f"\n<|im_start|>assistant\n{prefill}")
    return "".join(parts)


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--nprob", type=int, default=100)
    ap.add_argument("--ksamp", type=int, default=8)
    ap.add_argument("--nfewshot", type=int, default=10)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--filler-tokens", type=int, default=100)
    ap.add_argument("--out-dir", default="/old-data/apanda/tomi/outputs/random_numbers_paper_verification")
    args = ap.parse_args()

    chat_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    comp_url = f"http://{args.host}:{args.port}/v1/completions"
    out_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    sf = open(samples_path, "w")

    desc = "You will be given a 4-digit multiplication problem."
    rng = random.Random(args.seed)
    pool = [(rng.randint(1000, 9999), rng.randint(1000, 9999))
            for _ in range(args.nprob + args.nfewshot)]

    def call(regime, fs, a, b, use_filler, seed):
        f = random_numbers_filler(args.filler_tokens) if use_filler else ""
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
        regime, use_filler, idx, seed = spec
        fs = pool[:args.nfewshot]
        a, b = pool[args.nfewshot + idx]
        try:
            txt = call(regime, fs, a, b, use_filler, seed)
        except Exception as e:
            return {"regime": regime, "arm": "rnums" if use_filler else "base",
                    "a": a, "b": b, "target": a * b, "seed": seed, "completion": None,
                    "correct": False, "error": str(e)[:200]}
        m = ANS_RE.search(txt)
        got = m.group(1).replace(",", "") if m else ""
        return {"regime": regime, "arm": "rnums" if use_filler else "base",
                "a": a, "b": b, "target": a * b, "seed": seed, "completion": txt[:80],
                "correct": got == str(a * b), "error": None}

    specs = [(rg, uf, idx, s) for rg in ("chat_hardoff", "raw_nothink")
             for uf in (False, True) for idx in range(args.nprob) for s in range(args.ksamp)]
    print(f"[{args.run_id}] {len(specs)} requests → {args.host}:{args.port} ({args.model})")
    agg = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(job, specs):
            sf.write(json.dumps(rec) + "\n")
            key = (rec["regime"], rec["arm"])
            agg.setdefault(key, {}).setdefault(rec["target"], []).append(rec["correct"])
            done += 1
            if done % 500 == 0:
                print(f"  ...{done}/{len(specs)}")
    sf.close()

    summary = {"run_id": args.run_id, "model": args.model, "filler": "random_numbers",
               "filler_tokens": args.filler_tokens, "nshots": args.nfewshot,
               "nprob": args.nprob, "ksamp": args.ksamp, "cells": []}
    print(f"\n=== {args.run_id}  ({args.model}, 4x4, {args.nfewshot}-shot) ===")
    for rg in ("chat_hardoff", "raw_nothink"):
        print(f"-- regime={rg} --")
        row = {}
        for arm in ("base", "rnums"):
            bp = agg.get((rg, arm), {})
            n = len(bp) or 1
            p1 = sum(v[0] for v in bp.values()) / n
            p8 = sum(any(v) for v in bp.values()) / n
            row[arm] = (p1, p8, len(bp))
            summary["cells"].append({"regime": rg, "arm": arm, "pass1": p1, "pass8": p8, "nprob": len(bp)})
        (b1, b8, bn), (r1, r8, rn) = row["base"], row["rnums"]
        print(f"   base p@1={b1:.3f} p@8={b8:.3f} | rnums p@1={r1:.3f}({r1-b1:+.3f}) "
              f"p@8={r8:.3f}({r8-b8:+.3f}) | n{bn}")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"samples → {samples_path}\nsummary → {summary_path}\nDONE")


if __name__ == "__main__":
    main()
