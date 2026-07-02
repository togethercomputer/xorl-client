"""Generalized sharded held-out eval against a private SGLang serve.

Splits N held-out games (seed-777 [offset:offset+N]) across the R replicas of a
serve (app pods <app>-{i}.<app>-headless...:30000) as concurrent eval processes and
combines solve counts. Identical floor protocol to eval_holdout_vs_trained.sh.

Usage: shard_eval.py --app <app> --replicas R [--num-games 64] [--offset 0] [--label L]
"""
from __future__ import annotations
import argparse, json, subprocess, sys, os, time, urllib.request

REPO = "/home/apanda/xorl-client-wordle-science-20260614"
PY = "/home/apanda/xorl-internal/.venv/bin/python"
EVAL = f"{REPO}/experiments/wordle/standalone/eval_wordle_sglang.py"
MODEL = "Qwen/Qwen3.6-35B-A3B"


def url(app, i): return f"http://{app}-{i}.{app}-headless.apanda.svc.cluster.local:30000"


def healthy(app, i):
    try:
        with urllib.request.urlopen(url(app, i) + "/v1/models", timeout=5) as r:
            return b"Qwen" in r.read()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--replicas", type=int, required=True)
    ap.add_argument("--num-games", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    label = args.label or args.app
    outroot = f"/shared/apanda/wordle-sft-runs/evals/{label}"
    os.makedirs(outroot, exist_ok=True)

    # wait until >=1 replica healthy (up to 15 min)
    hidx = []
    for _ in range(90):
        hidx = [i for i in range(args.replicas) if healthy(args.app, i)]
        if len(hidx) >= max(1, args.replicas - 1):
            break
        time.sleep(10)
    if not hidx:
        print(f"[{label}] NO healthy replicas for app={args.app}"); sys.exit(2)
    print(f"[{label}] healthy replicas: {hidx}", flush=True)

    n = len(hidx)
    base, rem = args.num_games // n, args.num_games % n
    shards, off = [], args.offset
    for k, ridx in enumerate(hidx):
        cnt = base + (1 if k < rem else 0)
        if cnt == 0:
            continue
        shards.append((ridx, off, cnt)); off += cnt
    print(f"[{label}] plan: {shards}", flush=True)

    procs = []
    for ridx, offset, cnt in shards:
        outdir = f"{outroot}/shard{ridx}_off{offset}"
        os.makedirs(outdir, exist_ok=True)
        cmd = [PY, EVAL, "--base-url", url(args.app, ridx), "--model", MODEL,
               "--num-games", str(cnt), "--seed", "777", "--target-offset", str(offset),
               "--prompt-style", "public_reasoning_constraints_think",
               "--temperature", os.environ.get("EVAL_TEMP", "0.7"), "--max-new-tokens", "4096", "--no-ignore-eos",
               "--invalid-retries", os.environ.get("EVAL_INVALID_RETRIES", "0"), "--batch-size", str(min(cnt, 8)), "--output-dir", outdir]
        lf = open(f"{outdir}/run.log", "w")
        procs.append((ridx, offset, cnt, outdir,
                      subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO,
                                       env={**os.environ, "PYTHONPATH": REPO})))
    print(f"[{label}] launched {len(procs)} eval processes", flush=True)
    for *_, p in procs:
        p.wait()

    solved_total, games_total, per = 0, 0, []
    for ridx, offset, cnt, outdir, p in procs:
        sj = f"{outdir}/summary.json"
        if not os.path.exists(sj):
            per.append((ridx, offset, cnt, "FAILED")); continue
        s = json.load(open(sj))
        em, ng = s["metrics"]["exact_match"], s["num_games"]
        solved = round(em * ng); solved_total += solved; games_total += ng
        per.append((ridx, offset, ng, f"solved={solved} exact={em:.3f} valid={s['metrics']['valid_guess_rate']:.3f}"))
    print(f"\n===== [{label}] held-out (seed-777 [{args.offset}:{args.offset+args.num_games}]) =====")
    for x in per: print("  shard", x)
    rate = solved_total / max(games_total, 1)
    print(f"  TOTAL: solved {solved_total}/{games_total}  => held-out exact = {rate:.4f}")
    json.dump({"label": label, "solved": solved_total, "games": games_total, "exact": rate},
              open(f"{outroot}/RESULT.json", "w"), indent=2)


if __name__ == "__main__":
    main()
