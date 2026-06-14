#!/usr/bin/env python3
"""Build train/eval/fresh-probe pools for the nested-arithmetic OPSD task.

From /old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl (100k unique
problems), selects a difficulty band and writes four DISJOINT pool files to
/shared/opd-coord/ (the contamination rule is absolute — see HANDOFF §2c):

  arith_<band>_8192_prompts.json      train pool, chat format, /no_think prefix
  arith_<band>_8192_cotprompts.json   same problems, NO /no_think (thinking-mode
                                      CoT precompute requests), index-aligned
  arith_<band>_eval_1024.json         in-loop eval set (client eval_prompts_json_path)
  arith_<band>_fresh_1024.json        endpoint fresh-probe set

Band selectors match arith_band_probe.py buckets (ops5/ops6/ops7/...), plus
unions like ops56. Usage: python arith_prep_pools.py --band ops6
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

DATA = "/old-data/apanda/no_cot_math_public/arithmetic_problems.jsonl"
PREFIX = "Evaluate this Python expression. "
OUT_DIR = Path("/shared/opd-coord")

BANDS = {
    "ops5": lambda o, d: o == 5,
    "ops6": lambda o, d: o == 6,
    "ops7": lambda o, d: o == 7,
    "ops56": lambda o, d: o in (5, 6),
    "ops67": lambda o, d: o in (6, 7),
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

    return sum(isinstance(x, ast.BinOp) for x in ast.walk(tree)), depth(tree)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True, choices=sorted(BANDS))
    ap.add_argument("--train-n", type=int, default=8192)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--fresh-n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    sel = BANDS[args.band]
    band_problems = []
    for line in open(DATA):
        d = json.loads(line)
        ops, dep = expr_stats(d["problem"][len(PREFIX):])
        if sel(ops, dep):
            band_problems.append(d["problem"])

    need = args.train_n + args.eval_n + args.fresh_n
    if len(band_problems) < need:
        raise SystemExit(f"band {args.band} has {len(band_problems)} problems < {need}")
    rng = random.Random(args.seed)
    rng.shuffle(band_problems)
    train = band_problems[: args.train_n]
    evalset = band_problems[args.train_n: args.train_n + args.eval_n]
    fresh = band_problems[args.train_n + args.eval_n: need]

    # The contamination rule is absolute — assert, don't trust the slicing.
    assert not set(train) & set(evalset), "train/eval overlap"
    assert not set(train) & set(fresh), "train/fresh overlap"
    assert not set(evalset) & set(fresh), "eval/fresh overlap"

    def chat(problems: list[str], no_think: bool) -> list:
        pre = "/no_think " if no_think else ""
        return [[{"role": "user", "content": pre + p}] for p in problems]

    outs = {
        f"arith_{args.band}_{args.train_n}_prompts.json": chat(train, True),
        f"arith_{args.band}_{args.train_n}_cotprompts.json": chat(train, False),
        f"arith_{args.band}_eval_{args.eval_n}.json": chat(evalset, True),
        f"arith_{args.band}_fresh_{args.fresh_n}.json": chat(fresh, True),
    }
    for name, payload in outs.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(payload, ensure_ascii=False))
        print(f"wrote {path} ({len(payload)} prompts)")


if __name__ == "__main__":
    main()
