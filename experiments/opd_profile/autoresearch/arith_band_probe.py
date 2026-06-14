#!/usr/bin/env python3
"""Bucketed base-model difficulty probe on the nested-arithmetic task.

Buckets /old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl by
(op-count, parse depth) and measures Qwen3.6 base greedy accuracy per bucket in
two arms against the pristine teacher endpoints:

  direct — `/no_think` + `Answer: ` prefill, single forward pass
  cot    — thinking mode, generous token budget, parse last int after </think>

Band-selection rule (HANDOFF §2a): pick the band where cot >= ~0.8 and
direct <= ~0.1 — maximal transferable headroom with tolerable teacher noise.

Usage: python arith_band_probe.py [--n 64] [--max-cot-tokens 4096] [--workers 16]
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

DATA = "/old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl"
PREFIX = "Evaluate this Python expression. "
URLS = [
    "http://er-opd-q36-35b-slots-teacher-sglang-0:30000",
    "http://er-opd-q36-35b-slots-teacher-sglang-1:30000",
]
MODEL = "Qwen/Qwen3.6-35B-A3B"
# Answers can be negative — every regex on this task needs the leading -?
ANS_RE = re.compile(r"-?\d[\d,]*")

# Buckets: primary by op-count, plus shallow/deep depth slices at the extremes
BUCKETS = {
    "ops5": lambda o, d: o == 5,
    "ops6": lambda o, d: o == 6,
    "ops7": lambda o, d: o == 7,
    "ops5_d3": lambda o, d: o == 5 and d == 3,
    "ops6_d5p": lambda o, d: o == 6 and d >= 5,
    "ops7_d5p": lambda o, d: o == 7 and d >= 5,
}


def expr_stats(expr: str) -> tuple[int, int]:
    tree = ast.parse(expr, mode="eval").body

    def depth(n, d=0):
        if isinstance(n, ast.BinOp):
            return max(depth(n.left, d + 1), depth(n.right, d + 1))
        if isinstance(n, ast.UnaryOp):
            return depth(n.operand, d)
        return d

    ops = sum(isinstance(x, ast.BinOp) for x in ast.walk(tree))
    return ops, depth(tree)


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def probe_direct(url: str, problem: str) -> bool:
    resp = _post(url, {
        "model": MODEL, "max_tokens": 64, "temperature": 0.0, "top_p": 1.0,
        "top_k": -1, "n": 1, "logprobs": True, "logprob_start_len": 0,
        "messages": [
            {"role": "user", "content": "/no_think " + problem},
            {"role": "assistant", "content": "Answer: "},
        ],
        "stop": ["\n"], "continue_final_message": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = resp["choices"][0]["message"]["content"]
    m = ANS_RE.search(text)
    gold = safe_eval(problem[len(PREFIX):])
    return bool(m) and int(m.group(0).replace(",", "")) == gold


def probe_cot(url: str, problem: str, max_tokens: int) -> tuple[bool, bool]:
    """Returns (correct, truncated)."""
    resp = _post(url, {
        "model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
        "top_p": 1.0, "top_k": -1, "n": 1,
        "messages": [{"role": "user", "content": problem}],
        "chat_template_kwargs": {"enable_thinking": True},
    })
    choice = resp["choices"][0]
    text = choice["message"]["content"] or ""
    truncated = choice.get("finish_reason") == "length"
    tail = text.split("</think>")[-1]
    ints = ANS_RE.findall(tail)
    gold = safe_eval(problem[len(PREFIX):])
    ok = bool(ints) and int(ints[-1].replace(",", "")) == gold
    return ok, truncated


def safe_eval(expr: str) -> int:
    node = ast.parse(expr, mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            return left % right
        raise ValueError(f"disallowed node: {ast.dump(n)}")

    return ev(node)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-cot-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--output", default="/tmp/arith_band_probe_results.json")
    args = ap.parse_args()

    by_bucket: dict[str, list[str]] = collections.defaultdict(list)
    for line in open(DATA):
        d = json.loads(line)
        ops, dep = expr_stats(d["problem"][len(PREFIX):])
        for name, sel in BUCKETS.items():
            if sel(ops, dep):
                by_bucket[name].append(d["problem"])

    rng = random.Random(args.seed)
    samples = {b: rng.sample(probs, args.n) for b, probs in by_bucket.items()}

    jobs = []  # (bucket, arm, problem)
    for b, probs in samples.items():
        for i, p in enumerate(probs):
            jobs.append((b, "direct", p, URLS[i % len(URLS)]))
            jobs.append((b, "cot", p, URLS[(i + 1) % len(URLS)]))

    results: dict[str, dict[str, list]] = collections.defaultdict(lambda: {"direct": [], "cot": [], "trunc": []})

    def run(job):
        b, arm, p, url = job
        try:
            if arm == "direct":
                return b, arm, probe_direct(url, p), False
            ok, tr = probe_cot(url, p, args.max_cot_tokens)
            return b, arm, ok, tr
        except Exception as e:
            print(f"ERROR {b}/{arm}: {e}", flush=True)
            return b, arm, None, False

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for b, arm, ok, tr in ex.map(run, jobs):
            if ok is not None:
                results[b][arm].append(ok)
                if arm == "cot":
                    results[b]["trunc"].append(tr)
            done += 1
            if done % 50 == 0:
                print(f"[{done}/{len(jobs)}]", flush=True)

    print(f"\n{'bucket':<12} {'direct':>8} {'cot':>8} {'trunc%':>7} {'gap':>6}")
    summary = {}
    for b in BUCKETS:
        r = results[b]
        di = sum(r["direct"]) / max(len(r["direct"]), 1)
        co = sum(r["cot"]) / max(len(r["cot"]), 1)
        tr = sum(r["trunc"]) / max(len(r["trunc"]), 1)
        print(f"{b:<12} {di:>8.3f} {co:>8.3f} {tr:>7.1%} {co - di:>6.2f}")
        summary[b] = {"direct": di, "cot": co, "trunc": tr,
                      "n_direct": len(r["direct"]), "n_cot": len(r["cot"])}
    json.dump(summary, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
