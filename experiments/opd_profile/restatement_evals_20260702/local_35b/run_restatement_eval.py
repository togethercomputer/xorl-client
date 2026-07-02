#!/usr/bin/env python3
"""Step-0 restatement 2x2 completion + dose-response + controls on count_reassignments
(plus mult4/mult5 second task) against a local sglang serving base Qwen3.6-35B-A3B.

Rendering is TOKEN-FOR-TOKEN the training harness's prescribe path
(filler_tokens_rl.py, patched Qwen3 renderer):
    text = tokenizer.apply_chat_template([system, user(question)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    full = text + filler_str + "\nAnswer: "
with num_fewshot=0 and the harness system prompt
    get_system_prompt(problem_type, mixed_filler=False, filler_token_type="lorem", filler_tokens=0)
(the B/C/D run.sh args). filler_str per condition mirrors the prescribe site:
    mega-filler conditions: generate_mega_filler(K) + ("\n"+question if restate else "")
    harness-B          : generate_filler_tokens(0, "lorem", tokenizer)  # NB: floors at 1 lorem word
Problems: generate_variable_reassignment_problems(n+100, seed=12345)[:N] (num_fewshot=0 ->
training's eval problems are exactly [0:N]); SAME problems across all conditions.
Decode: greedy (temperature 0), answer-only (max 16 new tokens, stop "\n").
"""
import argparse, json, math, os, random, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vendored_filler_harness as H

SNAPSHOT = "/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
INT_RE = re.compile(r"[\-+]?\d[\d,]*")


def post_generate(url, text, temperature=0.0, lenient=False, timeout=600):
    body = {
        "text": text,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": 32 if lenient else 16,
            "stop": [] if lenient else ["\n"],
            "skip_special_tokens": True,
        },
    }
    if temperature > 0:
        # match training sampling (top_k/top_p defaults come from generation_config on the
        # chat path only; /generate uses these explicitly like the xorl sampler params)
        body["sampling_params"]["top_k"] = 20
        body["sampling_params"]["top_p"] = 0.95
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def target_var(problem_text):
    m = re.search(r"the variable `(\w+)` reassigned", problem_text)
    return m.group(1)


def code_lines(problem_text):
    m = re.search(r"```python\n(.*?)\n```", problem_text, re.DOTALL)
    return m.group(1).split("\n")


def structured_restate(problem_text):
    """Relevant-line extraction scaffold: restate the question in its exact format, but with
    the code block filtered to only the lines that mention the target variable (LHS or RHS).
    Same shape as a restatement (content control: is FULL-question content needed, or only
    the target-relevant lines?)."""
    t = target_var(problem_text)
    kept = [ln for ln in code_lines(problem_text) if re.search(rf"\b{re.escape(t)}\b", ln)]
    return (
        f"In the following code, how many times is the variable `{t}` reassigned "
        f"(assigned a new value after its first assignment)? Reply with just the number.\n\n"
        f"```python\n" + "\n".join(kept) + "\n```"
    )


# ---------------------------------------------------------------------------
# Condition table. Each entry: (system_prompt_kind, filler_builder(problems, i, tok) -> filler_str)
# Prompt = chat_template(sys, user=problems[i]) + filler_str + "\nAnswer: "
# system_prompt_kind: "ft0" = harness B/C/D system prompt (filler_tokens=0);
#                     "ft100" = the megafiller (A) run.sh system prompt (filler_tokens=100 ->
#                               counting-filler instruction; lorem isn't in FILLER_INSTRUCTIONS).
# ---------------------------------------------------------------------------

def build_cr_conditions():
    q = lambda P, i: str(P[i]["problem"])
    return {
        # (cheap-to-build conditions first: mega-8192 blobs cost ~2.4s each to build, so they
        #  stream last while the server chews through the cheap ones)
        # --- validation gate ---
        "B_harness_nofiller_norestate": ("ft0", lambda P, i, tok: H.generate_filler_tokens(0, "lorem", tok)),
        "C_restate1_nofiller":          ("ft0", lambda P, i, tok: H.generate_mega_filler(0, tok) + "\n" + q(P, i)),
        # --- clean-baseline control (harness B floors at 1 lorem word; this is the true empty prefill) ---
        "B_clean_norestate":            ("ft0", lambda P, i, tok: ""),
        # --- restatement dose-response ---
        "restate2":                     ("ft0", lambda P, i, tok: ("\n" + q(P, i)) * 2),
        "restate4":                     ("ft0", lambda P, i, tok: ("\n" + q(P, i)) * 4),
        "restate8":                     ("ft0", lambda P, i, tok: ("\n" + q(P, i)) * 8),
        # --- content-vs-position control: restate a DIFFERENT problem's question ---
        "wrongq_restate1":              ("ft0", lambda P, i, tok: "\n" + str(P[(i + 157) % len(P)]["problem"])),
        # --- relevant-line extraction scaffold ---
        "structured_restate1":          ("ft0", lambda P, i, tok: "\n" + structured_restate(q(P, i))),
        # --- filler size: is 8192 vs 512 different at all? ---
        "mega512_restate":              ("ft0", lambda P, i, tok: H.generate_mega_filler(512, tok) + "\n" + q(P, i)),
        "mega512_norestate":            ("ft0", lambda P, i, tok: H.generate_mega_filler(512, tok)),
        # --- the missing 2x2 cell + A reproduction (mega-8192; slow builds last) ---
        "D_mega8192_norestate":         ("ft0", lambda P, i, tok: H.generate_mega_filler(8192, tok)),
        "A_mega8192_restate":           ("ft0", lambda P, i, tok: H.generate_mega_filler(8192, tok) + "\n" + q(P, i)),
        "A_sys100_mega8192_restate":    ("ft100", lambda P, i, tok: H.generate_mega_filler(8192, tok) + "\n" + q(P, i)),
    }


def build_mult_conditions():
    q = lambda P, i: str(P[i]["problem"])
    return {
        "baseline":            ("ft0", lambda P, i, tok: ""),
        "restate1":            ("ft0", lambda P, i, tok: "\n" + q(P, i)),
        "restate4":            ("ft0", lambda P, i, tok: ("\n" + q(P, i)) * 4),
        "counting200":         ("ft0", lambda P, i, tok: H.generate_filler_tokens(200, "counting", tok)),
        "counting200_restate": ("ft0", lambda P, i, tok: H.generate_filler_tokens(200, "counting", tok) + "\n" + q(P, i)),
    }


def load_oldgen_problems(n, seed=12345):
    """The EXACT problems the handoff's step-0 2x2 (B=0.42, C=0.75, A=0.73) was measured on.
    The generator defaults were hardened on 2026-07-02 ~01:09Z (ZA fix; counts 5-13); the 2x2
    references predate that (old counts 1-7). The old texts+answers were recovered verbatim from
    the old C (restate) run's sample dumps, whose prescribed prefill contains the full question:
    /shared/opd-coord/.../er-opd-q36-restate/20260701T235857Z-.../samples/step_*.jsonl
    (2500 unique problems, 0 ground-truth mismatches vs an independent recount)."""
    rows = json.load(open(os.path.join(HERE, "old_easy_problems_extracted.json")))
    rows.sort(key=lambda r: (r["src_step"], r["src_idx"]))
    assert seed == 12345
    return rows


TASKS = {
    "count_reassignments": (H.generate_variable_reassignment_problems, build_cr_conditions),
    "count_reassignments_oldgen": (lambda n, seed: load_oldgen_problems(n, seed), build_cr_conditions),
    "multiplication_4digit": (lambda n, seed: H.generate_multiplication_problems(n, num_digits=4, seed=seed), build_mult_conditions),
    "multiplication_5digit": (lambda n, seed: H.generate_multiplication_problems(n, num_digits=5, seed=seed), build_mult_conditions),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30061)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--task", default="count_reassignments")
    ap.add_argument("--conditions", default="", help="CSV subset of condition names (default: all)")
    ap.add_argument("--out", default=os.path.join(HERE, "samples"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ksamp", type=int, default=1,
                    help="samples per problem (>1 only for the temp-1.0 harness-protocol gate)")
    ap.add_argument("--lenient", action="store_true",
                    help="derailment ablation: no stop-newline, 32 tokens, parse first int anywhere")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)
    url = f"http://{args.host}:{args.port}/generate"

    gen, build_conds = TASKS[args.task]
    # load_problems generates n+100 and training (num_fewshot=0) uses [0:N] as eval problems
    problems = gen(args.n + 100, seed=args.seed)[: args.n]
    conds = build_conds()
    if args.conditions:
        keep = args.conditions.split(",")
        missing = [c for c in keep if c not in conds]
        assert not missing, f"unknown conditions: {missing}"
        conds = {k: conds[k] for k in keep}

    ptype = args.task.replace("_oldgen", "")
    sys_ft0 = H.get_system_prompt(ptype, mixed_filler=False, filler_token_type="lorem", filler_tokens=0)
    sys_ft100 = H.get_system_prompt(ptype, mixed_filler=False, filler_token_type="lorem", filler_tokens=100)
    sysmap = {"ft0": sys_ft0, "ft100": sys_ft100}

    def render_base(system_prompt, question):
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    os.makedirs(args.out, exist_ok=True)
    preview_path = os.path.join(args.out, f"{args.task}_prompt_previews.txt")
    pf = open(preview_path, "a")

    # Prompt building runs in the MAIN thread only (global-random filler generators are not
    # thread-safe), with a deterministic per-(condition, idx) seed for the filler content.
    # Jobs are STREAMED to the request pool as they are built: mega-8192 blobs cost ~2.4s each
    # to build (the harness's per-element tokenizer loop), so building all up front would idle
    # the server for ~35 min.
    def gen_jobs():
        for cname, (skind, fbuild) in conds.items():
            for i in range(args.n):
                random.seed(hash((args.task, cname, i)) & 0x7FFFFFFF)
                filler = fbuild(problems, i, tok)
                full = render_base(sysmap[skind], str(problems[i]["problem"])) + filler + "\nAnswer: "
                if i == 0:
                    pf.write(f"\n===== {args.task} / {cname} (idx 0; prompt tokens={len(tok.encode(full))}) =====\n{full}\n")
                    pf.flush()
                for ks in range(args.ksamp):
                    yield {"task": args.task, "condition": cname, "idx": i, "samp": ks,
                           "answer": str(problems[i]["answer"]), "prompt": full}

    total_jobs = len(conds) * args.n * args.ksamp
    print(f"[{args.task}] {total_jobs} requests over {len(conds)} conditions -> {url}", flush=True)

    def run_job(j):
        rec = {k: j[k] for k in ("task", "condition", "idx", "answer", "samp")}
        try:
            t0 = time.time()
            out = post_generate(url, j["prompt"], temperature=args.temperature,
                                lenient=args.lenient)
            txt = out.get("text", "") or ""
            m = INT_RE.search(txt)
            rec["completion"] = txt[:64]
            rec["pred"] = m.group(0).replace(",", "").lstrip("+") if m else None
            rec["correct"] = (rec["pred"] == rec["answer"])
            rec["prompt_tokens"] = out.get("meta_info", {}).get("prompt_tokens")
            rec["latency_s"] = round(time.time() - t0, 2)
            rec["error"] = None
        except Exception as e:
            rec.update(completion=None, pred=None, correct=False, prompt_tokens=None,
                       latency_s=None, error=str(e)[:200])
        return rec

    out_path = os.path.join(args.out, f"{args.task}_samples.jsonl")
    agg, done, errs = {}, 0, 0
    with open(out_path, "a") as sf, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in gen_jobs()]
        pf.close()
        for fut in as_completed(futs):
            rec = fut.result()
            sf.write(json.dumps(rec) + "\n")
            agg.setdefault(rec["condition"], []).append(rec)
            done += 1
            errs += 1 if rec["error"] else 0
            if done % 300 == 0:
                print(f"  ...{done}/{len(futs)} (errs={errs})", flush=True)

    print(f"\n=== {args.task} (n={args.n}, greedy, seed={args.seed}, errs={errs}) ===", flush=True)
    print(f"{'condition':<32} {'acc':>6} {'k':>4}/{'n':<4} {'95% CI':>15}")
    summary = []
    for cname in conds:
        recs = agg.get(cname, [])
        ok = [r for r in recs if not r["error"]]
        k = sum(r["correct"] for r in ok)
        n = len(ok)
        p = k / n if n else float("nan")
        lo, hi = wilson_ci(k, n)
        print(f"{cname:<32} {p:>6.3f} {k:>4}/{n:<4} [{lo:.3f}, {hi:.3f}]")
        summary.append({"task": args.task, "condition": cname, "acc": p, "k": k, "n": n,
                        "ci95": [round(lo, 4), round(hi, 4)], "errors": len(recs) - n})
    with open(os.path.join(args.out, f"{args.task}_summary.json"), "a") as f:
        f.write(json.dumps({"n": args.n, "seed": args.seed, "rows": summary}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
