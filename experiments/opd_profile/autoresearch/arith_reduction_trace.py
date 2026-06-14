"""Generate exact serial reduction traces for ops6 nested-arithmetic problems.

For each problem the AST yields the ordered list of binary-reduction steps
(innermost-first) with exact intermediate values — dense, noise-free per-step
grounding targets for value-grounded prefill-compute experiments (RiM-style
memory grounding / numeric-CoT SFT). No teacher-prose parsing.

Every ops6 expression decomposes into exactly 6 serial steps.

Usage:
    python arith_reduction_trace.py \
        --prompts /shared/opd-coord/arith_ops6_nonempty_prompts.json \
        --out     /shared/opd-coord/arith_ops6_reduction_trace.json
"""
import argparse
import ast
import json
import re


_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Mod: "%"}


def reduction_trace(expr):
    """Return (steps, final) where steps = [(subexpr_str, value), ...] innermost-first."""
    node = ast.parse(expr, mode="eval").body
    steps = []

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else +v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = _OPS[type(n.op)]
            if op == "+":
                r = a + b
            elif op == "-":
                r = a - b
            elif op == "*":
                r = a * b
            elif op == "//":
                r = a // b
            else:
                r = a % b
            steps.append((f"{a} {op} {b}", r))
            return r
        raise ValueError(f"unsupported node {type(n)}")

    return steps, ev(node)


def extract_expr(prompt):
    if isinstance(prompt, list):
        prompt = prompt[0]["content"]
    if isinstance(prompt, dict):
        prompt = prompt.get("content", "")
    m = re.search(r"expression\.\s*(.+)$", prompt.strip())
    s = m.group(1).strip() if m else prompt.strip()
    return re.sub(r"^/no_think\s*", "", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = json.load(open(args.prompts))
    out, n_ok, n_bad, lens = [], 0, 0, []
    for i, p in enumerate(pool):
        expr = extract_expr(p)
        try:
            steps, final = reduction_trace(expr)
        except Exception as e:
            n_bad += 1
            out.append({"index": i, "expr": expr, "error": str(e)})
            continue
        n_ok += 1
        lens.append(len(steps))
        out.append(
            {
                "index": i,
                "expr": expr,
                "final": final,
                "steps": [{"sub": s, "val": v} for s, v in steps],
                "values": [v for _, v in steps],          # ordered intermediate values
                "trace": ";".join(str(v) for _, v in steps),  # compact numeric-CoT target
            }
        )
    json.dump(out, open(args.out, "w"))
    import statistics

    print(f"wrote {args.out}: ok={n_ok} bad={n_bad}")
    if lens:
        print(f"steps/expr: min={min(lens)} med={statistics.median(lens)} max={max(lens)}")


if __name__ == "__main__":
    main()
