#!/usr/bin/env python3
"""Greedy direct-arm accuracy on NEVER-TRAINED nested-arithmetic problems.

Arithmetic counterpart of fresh_pair_probe.py: run against whatever weights the
samplers currently hold (probe-while-hot at the end of a run; weights are not
checkpointed unless save_every is set). Asserts the probe set is disjoint from
the training pool before scoring anything.

Usage: python arith_fresh_probe.py --band ops6 [--n 256] [--url http://...:30060]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PREFIX = "Evaluate this Python expression. "
ANS_RE = re.compile(r"-?\d[\d,]*")
MODEL = "Qwen/Qwen3.6-35B-A3B"


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
        raise ValueError("disallowed node")

    return ev(node)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="ops6")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--url", default="http://er-opd-q36-35b-slots-sglang-0:30060")
    ap.add_argument("--train-pool", default=None,
                    help="defaults to /shared/opd-coord/arith_<band>_8192_prompts.json")
    ap.add_argument("--fresh-pool", default=None,
                    help="defaults to /shared/opd-coord/arith_<band>_fresh_1024.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    train_pool = args.train_pool or f"/shared/opd-coord/arith_{args.band}_8192_prompts.json"
    fresh_pool = args.fresh_pool or f"/shared/opd-coord/arith_{args.band}_fresh_1024.json"
    train_set = {p[0]["content"] for p in json.load(open(train_pool))}
    fresh = [p[0]["content"] for p in json.load(open(fresh_pool))][: args.n]
    overlap = train_set & set(fresh)
    assert not overlap, f"fresh probe contaminated: {len(overlap)} overlapping prompts"

    def run(user_content: str) -> bool | None:
        payload = {
            "model": MODEL, "max_tokens": 64, "temperature": 0.0, "top_p": 1.0,
            "top_k": -1, "n": 1, "logprobs": True, "logprob_start_len": 0,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "Answer: "},
            ],
            "stop": ["\n"], "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(args.url + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            text = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            return None
        gold = safe_eval(user_content[user_content.find(PREFIX) + len(PREFIX):].strip())
        m = ANS_RE.search(text or "")
        return bool(m) and int(m.group(0).replace(",", "")) == gold

    with ThreadPoolExecutor(args.workers) as ex:
        results = [r for r in ex.map(run, fresh) if r is not None]
    ok = sum(results)
    print(f"FRESH-PROBE ({args.band}) greedy accuracy: {ok}/{len(results)} = {ok/max(len(results),1):.3f} "
          f"(requested {len(fresh)})")


if __name__ == "__main__":
    main()
