#!/usr/bin/env python3
"""Filter a precomputed arithmetic CoT file to teacher-correct entries.

The teacher is only ~0.6-0.8 on this task (unlike 98.7% on 4x4), so training
data must be filtered to correct CoTs and the prompts json rebuilt to the
surviving subset so prompts/CoT stay index-aligned (the `_nonempty` convention,
HANDOFF §2d). Also reports the truncation rate (Run-B lesson: a tight budget
once silently amputated 80% of CoTs — re-precompute with a bigger budget if >2%).

Usage:
  python arith_filter_cot.py --prompts /shared/opd-coord/arith_<b>_8192_prompts.json \
      --cot /shared/opd-coord/arith_<b>_8192_cot_raw.json \
      --out-prefix /shared/opd-coord/arith_<b>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

PREFIX = "Evaluate this Python expression. "
ANS_RE = re.compile(r"-?\d[\d,]*")


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
    ap.add_argument("--prompts", required=True, help="training pool json (/no_think chat format)")
    ap.add_argument("--cot", required=True, help="raw cot_precompute.py output, index-aligned with --prompts")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_nonempty_{prompts,cot}.json")
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    cots = json.loads(Path(args.cot).read_text())
    if len(prompts) != len(cots):
        raise SystemExit(f"length mismatch: {len(prompts)} prompts vs {len(cots)} cots")

    kept_prompts, kept_cots = [], []
    n_empty = n_trunc = n_wrong = n_unparsed = 0
    for prompt, entry in zip(prompts, cots):
        text = (entry or {}).get("cot") or ""
        if not text.strip():
            n_empty += 1
            continue
        if (entry.get("finish_reason") or "") == "length":
            n_trunc += 1
            continue
        user = prompt[0]["content"]
        gold = safe_eval(user[user.find(PREFIX) + len(PREFIX):].strip())
        tail = text.split("</think>")[-1]
        ints = ANS_RE.findall(tail)
        if not ints:
            n_unparsed += 1
            continue
        if int(ints[-1].replace(",", "")) != gold:
            n_wrong += 1
            continue
        # Convention: cot text ends with "Answer: <gold>" (the student_prefill
        # suffix cue). Teacher-correct here, so appending the gold is consistent.
        if not re.search(r"Answer:\s*-?[\d,]+\s*$", text):
            text = text.rstrip() + f"\nAnswer: {gold}"
        kept_prompts.append(prompt)
        kept_cots.append({**entry, "cot": text, "prompt": prompt})

    n = len(prompts)
    print(f"total={n} kept={len(kept_prompts)} ({len(kept_prompts)/n:.1%}) "
          f"empty={n_empty} truncated={n_trunc} ({n_trunc/n:.1%}) wrong={n_wrong} unparsed={n_unparsed}")
    if n_trunc / n > 0.02:
        print("WARNING: truncation >2% — raise --max-tokens and re-precompute (Run-B lesson)")

    pp = Path(f"{args.out_prefix}_nonempty_prompts.json")
    cp = Path(f"{args.out_prefix}_nonempty_cot.json")
    pp.write_text(json.dumps(kept_prompts, ensure_ascii=False))
    cp.write_text(json.dumps(kept_cots, ensure_ascii=False))
    lens = sorted(e["tokens"] or 0 for e in kept_cots)
    print(f"wrote {pp} + {cp} ({len(kept_prompts)} entries; cot tokens "
          f"min={lens[0]} med={lens[len(lens)//2]} max={lens[-1]})")


if __name__ == "__main__":
    main()
