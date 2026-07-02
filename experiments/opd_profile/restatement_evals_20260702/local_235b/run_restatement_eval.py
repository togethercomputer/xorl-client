#!/usr/bin/env python3
"""Restatement-vs-filler prefill evals against a LOCAL sglang sampler.

Tests the hypothesis that the exploitable mechanism of prefill-time compute is
attention-convenient re-placement of the question (visible re-encoding), not
invisible computation over filler tokens.

Methodology matches paper.tex ("Not All LLM Reasoning is Visible in the
Chain-of-Thought"): 10-shot, each few-shot example carries the SAME condition's
prefill content in the assistant turn followed by "Answer: <gold>"; the final
question gets the same prefill + "Answer:" and the model completes only the
answer.

Serving: a standalone sglang server (apanda-eval-q235-sglang pod, 8xH100 tp8,
port 30060) serving Qwen3-235B-A22B-Instruct-2507 from /shared/huggingface.
Requests go to POST /generate with the Qwen <|im_start|> chat template rendered
manually (mirrors eval_random_numbers_paper_claim.py::render_raw); assistant
prefill = the raw prompt simply ends with the prefill text after the assistant
<|im_start|> header. No external API is used.

Decoding: temperature=0, max_new_tokens=16, stop on newline. Parse = first
integer (commas/sign allowed) after the last "Answer:" in the returned text.

Tasks (paper Appendix A definitions, seed 12345, same problems across conditions):
  mult4      -- "What is XXXX times YYYY?", random 4-digit operands.
  arithmetic -- 5-7 nested ops (+ - * // %), operands uniform in [-99, 99],
                gold computed with PYTHON semantics for // and % (floor division
                and sign-of-divisor modulus).
  varcount   -- short code snippet (10-14 assignment lines with reuse, plus a
                couple of print() lines that only read); question: how many
                distinct variables are assigned a value.

Conditions (identical problems per task):
  baseline, counting (1..100, ~200 tok), restate1/2/4/8, counting+restate,
  wrongq-restate (a seed-shifted different problem's question), structured-restate.
"""
import argparse
import json
import math
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ANS_RE = re.compile(r"([\-+]?[0-9][0-9,]*)")
SEED = 12345
WRONGQ_SEED_SHIFT = 1000
NSHOTS = 10

SYSTEM_PROMPT = (
    "You will be given a math problem. Answer immediately using the format "
    "`Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing "
    "else. No explanation, no words, no reasoning, just the number."
)

# Paper's headline filler: counting 1..100 ("~200 tokens"). Measured with the
# served Qwen3-235B-Instruct-2507 tokenizer this is 291 tokens (Qwen3 does not
# merge spaces into digit tokens). We keep 1..100 verbatim to match the paper's
# stated condition rather than truncating to exactly 200 tokens.
COUNTING_FILLER = " ".join(str(i) for i in range(1, 101))


# ---------------------------------------------------------------- tasks

def gen_mult4(rng):
    a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
    return {"question": f"What is {a} times {b}?", "gold": a * b,
            "meta": {"a": a, "b": b}}


def _arith_lit(v):
    return f"({v})" if v < 0 else str(v)


# Op weights are the one free parameter the paper leaves unspecified ("5 to 7
# nested arithmetic operations (+ - * // %) with operands sampled uniformly
# from [-99,99]"). Uniform op choice makes // and % shrink intermediate values
# so answers are small and the task is too easy (Qwen3-235B 10-shot baseline
# 23.0% vs the paper's 11.0%). Calibrated on 2026-07-02 against the paper's
# baseline: weights +:3 -:3 *:3 //:0.5 %:0.5 give 8.2% baseline (within the
# +-4pp validation gate of 11.0%). See RESULTS.md validation-gate section.
ARITH_OPS = ["+", "-", "*", "//", "%"]
ARITH_OP_WEIGHTS = [3.0, 3.0, 3.0, 0.5, 0.5]


def _build_arith_tree(rng, k):
    """Random binary expr tree with k internal (op) nodes. Returns nested tuples."""
    if k == 0:
        return rng.randint(-99, 99)
    lk = rng.randint(0, k - 1)
    op = rng.choices(ARITH_OPS, weights=ARITH_OP_WEIGHTS)[0]
    return (op, _build_arith_tree(rng, lk), _build_arith_tree(rng, k - 1 - lk))


def _render_arith(node, root=False):
    if isinstance(node, int):
        return _arith_lit(node)
    op, l, r = node
    s = f"{_render_arith(l)} {op} {_render_arith(r)}"
    return s if root else f"({s})"


def _eval_arith(node):
    if isinstance(node, int):
        return node
    op, l, r = node
    lv, rv = _eval_arith(l), _eval_arith(r)
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "//":
        return lv // rv  # Python floor division
    return lv % rv       # Python modulus (sign of divisor)


def _arith_subexprs(node, acc):
    """Post-order (innermost-first) list of rendered subexpressions."""
    if isinstance(node, int):
        return
    op, l, r = node
    _arith_subexprs(l, acc)
    _arith_subexprs(r, acc)
    acc.append(f"{_render_arith(l)} {op} {_render_arith(r)}")


def gen_arith(rng):
    while True:
        tree = _build_arith_tree(rng, rng.randint(5, 7))
        try:
            gold = _eval_arith(tree)
        except ZeroDivisionError:
            continue
        if abs(gold) > 10**12:  # keep answers human-scale (rare with */ nesting)
            continue
        expr = _render_arith(tree, root=True)
        return {"question": f"What is {expr}?", "gold": gold,
                "meta": {"expr": expr, "tree": repr(tree)}}


VARNAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n", "p", "q",
            "r", "s", "t", "u", "v", "w", "x", "y", "z"]


def gen_varcount(rng):
    """Dozen-ish lines of simple assignments with some reuse + read-only prints.

    Gold = number of distinct variables that appear on the LHS of an
    assignment. print() lines only read variables and do not count.
    """
    pool = rng.sample(VARNAMES, k=rng.randint(6, 10))
    n_assign = rng.randint(10, 14)
    assigned, lines = [], []
    for _ in range(n_assign):
        if assigned and rng.random() < 0.35:
            var = rng.choice(assigned)          # reassignment (reuse)
        else:
            unused = [v for v in pool if v not in assigned]
            var = rng.choice(unused) if unused else rng.choice(assigned)
        if assigned and rng.random() < 0.5:
            src = rng.choice(assigned)
            rhs = f"{src} {rng.choice(['+', '-', '*'])} {rng.randint(1, 9)}"
        else:
            rhs = str(rng.randint(0, 99))
        if var not in assigned:
            assigned.append(var)
        lines.append(f"{var} = {rhs}")
    for _ in range(rng.randint(1, 3)):          # read-only lines (not assignments)
        pos = rng.randint(1, len(lines))
        seen = [ln.split(" = ")[0] for ln in lines[:pos] if " = " in ln]
        lines.insert(pos, f"print({rng.choice(seen)})")
    snippet = "\n".join(lines)
    q = (f"{snippet}\n\nHow many distinct variables are assigned a value in "
         f"this code?")
    return {"question": q, "gold": len(set(assigned)),
            "meta": {"snippet": snippet}}


GENERATORS = {"mult4": gen_mult4, "arithmetic": gen_arith, "varcount": gen_varcount}


def make_pool(task, n, seed):
    rng = random.Random(seed)
    return [GENERATORS[task](rng) for _ in range(n)]


# ---------------------------------------------------------- structured restate

def structured_restate(task, prob):
    if task == "mult4":
        a = " ".join(str(prob["meta"]["a"]))
        b = " ".join(str(prob["meta"]["b"]))
        return f"What is {a} times {b}?"
    if task == "arithmetic":
        tree = eval(prob["meta"]["tree"])  # ints/tuples only (our own repr)
        acc = []
        _arith_subexprs(tree, acc)
        return "\n".join(acc)
    # varcount: snippet with line numbers prepended
    numbered = "\n".join(f"{i+1}: {ln}"
                         for i, ln in enumerate(prob["meta"]["snippet"].split("\n")))
    return (f"{numbered}\n\nHow many distinct variables are assigned a value in "
            f"this code?")


CONDITIONS = ["baseline", "counting", "restate1", "restate2", "restate4",
              "restate8", "counting+restate", "wrongq-restate",
              "structured-restate"]


def prefill_body(cond, task, prob, wrong_prob):
    """Content placed in the assistant turn before '\\nAnswer:'. '' = none."""
    q = prob["question"]
    if cond == "baseline":
        return ""
    if cond == "counting":
        return COUNTING_FILLER
    if cond.startswith("restate"):
        k = int(cond[len("restate"):])
        return "\n".join([q] * k)
    if cond == "counting+restate":
        return COUNTING_FILLER + "\n" + q
    if cond == "wrongq-restate":
        return wrong_prob["question"]
    if cond == "structured-restate":
        return structured_restate(task, prob)
    raise ValueError(cond)


# ---------------------------------------------------------------- prompting

def build_messages(task, cond, fewshot, fs_wrong, prob, wrong_prob, sys_prompt):
    msgs = [{"role": "system", "content": sys_prompt}]
    for fs, fsw in zip(fewshot, fs_wrong):
        body = prefill_body(cond, task, fs, fsw)
        asst = f"{body}\nAnswer: {fs['gold']}" if body else f"Answer: {fs['gold']}"
        msgs.append({"role": "user", "content": fs["question"]})
        msgs.append({"role": "assistant", "content": asst})
    msgs.append({"role": "user", "content": prob["question"]})
    body = prefill_body(cond, task, prob, wrong_prob)
    prefill = f"{body}\nAnswer:" if body else "Answer:"
    return msgs, prefill


def render_raw(msgs, prefill):
    """Qwen im_start/im_end rendering (== eval_random_numbers_paper_claim.py)."""
    parts = []
    for idx, m in enumerate(msgs):
        nl = "\n" if idx > 0 else ""
        parts.append(f"{nl}<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append(f"\n<|im_start|>assistant\n{prefill}")
    return "".join(parts)


# ------------------------------------------------------------ sglang client

def post(url, body, max_retries=8):
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (409, 429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) + random.random() * 2)
                continue
            raise RuntimeError(f"HTTP {code}: {e.read().decode()[:300]}") from None
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError):
            if attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) + random.random() * 2)
                continue
            raise


def call_model(gen_url, msgs, prefill):
    body = {"text": render_raw(msgs, prefill),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16,
                                "stop": ["\n", "<|im_end|>"]}}
    r = post(gen_url, body)
    return r["text"] or ""


def parse_answer(txt):
    """First integer after the last 'Answer:' if present, else first integer."""
    tail = txt.rsplit("Answer:", 1)[-1] if "Answer:" in txt else txt
    m = ANS_RE.search(tail)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", "").replace("+", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------- runner

def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))


def run_cell(task, cond, args, out_dir):
    tag = f"{args.model_tag}__{task}__{cond}"
    out_path = os.path.join(out_dir, f"{tag}.jsonl")
    done_idx = set()
    if os.path.exists(out_path):  # resume support
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done_idx.add(rec["idx"])
                except json.JSONDecodeError:
                    pass

    pool = make_pool(task, args.nprob + NSHOTS, SEED)
    wrong_pool = make_pool(task, args.nprob + NSHOTS, SEED + WRONGQ_SEED_SHIFT)
    fewshot, fs_wrong = pool[:NSHOTS], wrong_pool[:NSHOTS]

    lock = threading.Lock()
    f = open(out_path, "a")

    def job(i):
        if i in done_idx:
            return None
        prob, wrong = pool[NSHOTS + i], wrong_pool[NSHOTS + i]
        msgs, prefill = build_messages(task, cond, fewshot, fs_wrong, prob,
                                       wrong, args.system_prompt)
        rec = {"task": task, "condition": cond, "idx": i,
               "question": prob["question"][:400], "gold": prob["gold"]}
        try:
            txt = call_model(args.gen_url, msgs, prefill)
            parsed = parse_answer(txt)
            rec.update({"completion": txt[:120], "parsed": parsed,
                        "correct": parsed == prob["gold"], "error": None})
        except Exception as e:
            rec.update({"completion": None, "parsed": None, "correct": False,
                        "error": str(e)[:300]})
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = [r for r in ex.map(job, range(args.nprob)) if r is not None]
    f.close()

    # aggregate from the file (includes resumed rows)
    recs = {}
    with open(out_path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None or rec["idx"] not in recs:
                recs[rec["idx"]] = rec
    vals = list(recs.values())
    n_err = sum(1 for r in vals if r.get("error"))
    n_ok = len(vals) - n_err
    acc = (sum(1 for r in vals if r.get("correct")) / n_ok) if n_ok else 0.0
    cell = {"model": args.model, "task": task, "condition": cond,
            "n": n_ok, "n_err": n_err, "acc": acc, "ci95": ci95(acc, n_ok),
            "secs": round(time.time() - t0, 1)}
    print(f"[{tag}] acc={acc:.3f} ±{cell['ci95']:.3f} n={n_ok} err={n_err} "
          f"({cell['secs']}s)", flush=True)
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="sglang server host (pod IP)")
    ap.add_argument("--port", type=int, default=30060)
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--tasks", default="mult4,arithmetic,varcount")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--nprob", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-think-prefix", action="store_true",
                    help="prepend /no_think to the system prompt (Qwen)")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    args.system_prompt = ("/no_think\n" + SYSTEM_PROMPT if args.no_think_prefix
                          else SYSTEM_PROMPT)
    args.gen_url = f"http://{args.host}:{args.port}/generate"

    tasks = args.tasks.split(",")
    conds = args.conditions.split(",")
    summary_path = os.path.join(args.out_dir, f"summary__{args.model_tag}.json")
    cells = []
    if os.path.exists(summary_path):
        cells = json.load(open(summary_path)).get("cells", [])

    for task in tasks:
        for cond in conds:
            if any(c["task"] == task and c["condition"] == cond and
                   c["n"] >= args.nprob and c["n_err"] == 0 for c in cells):
                print(f"skip {task}/{cond} (already complete)", flush=True)
                continue
            cell = run_cell(task, cond, args, args.out_dir)
            cells = [c for c in cells
                     if not (c["task"] == task and c["condition"] == cond)]
            cells.append(cell)
            json.dump({"model": args.model, "endpoint": args.gen_url,
                       "system_prompt": args.system_prompt, "nshots": NSHOTS,
                       "seed": SEED, "cells": cells},
                      open(summary_path, "w"), indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
