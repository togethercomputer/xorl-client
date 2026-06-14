"""Probe: can base Qwen3.6-35B-A3B ENUMERATE the consistent Wordle candidates
from constraints alone (no candidate list given)?

This is the load-bearing question for the algorithmic-enumeration-think SFT idea.
The scaffold experiment proved the model can FILTER+PICK when handed the list (0.97);
the open question is whether it can GENERATE the list from feedback constraints. If it
recalls consistent words with high VALIDITY (listed words actually satisfy the clues),
SFT on enumeration think is well-motivated. If it confabulates (lists inconsistent
words), SFT would teach confabulation -> pivot to reward-based (GRPO).

Hits the live teacher endpoint over HTTP (inference only; no training, no weight sync).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks import wordle  # noqa: E402

_WORD_RE = re.compile(r"\b([A-Za-z]{5})\b")


def _split_scores(cands):
    n = len(cands)
    scores = {}
    for g in cands:
        b = {}
        for t in cands:
            fb = wordle.compute_feedback(g, t)
            b[fb] = b.get(fb, 0) + 1
        scores[g] = sum(c * c for fb, c in b.items() if fb != "GGGGG") / n
    return scores


def _mid_state(target, opener, max_turn):
    """Play env-optimal hard mode up to a turn whose consistent set is interesting."""
    hist = []
    for turn in range(max_turn):
        cands = wordle.remaining_candidates(hist)
        if turn > 0 and 2 <= len(cands) <= 40:
            return hist, cands
        if turn == 0:
            g = opener
        else:
            sc = _split_scores(cands)  # compute ONCE (was O(n^3) inside the key)
            g = min(cands, key=lambda w: (sc[w], w))
        hist.append((g, wordle.compute_feedback(g, target)))
        if g == target:
            break
    return hist, wordle.remaining_candidates(hist)


def _gen(base_url, tokenizer, messages, *, temperature, max_new_tokens, enable_thinking, timeout=600):
    return _gen_batch(base_url, tokenizer, [messages], temperature=temperature,
                      max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, timeout=timeout)[0]


def _gen_batch(base_url, tokenizer, messages_list, *, temperature, max_new_tokens, enable_thinking, timeout=900):
    """One batched /generate over many prompts -> SGLang decodes them concurrently."""
    batch_ids = [
        list(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=enable_thinking, return_dict=False))
        for m in messages_list
    ]
    r = requests.post(
        base_url.rstrip("/") + "/generate",
        json={"input_ids": batch_ids,
              "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens,
                                  "ignore_eos": False, "stop": []},
              "return_logprob": False},
        timeout=timeout,
    )
    r.raise_for_status()
    res = r.json()
    if isinstance(res, dict):
        res = [res]
    return [str(x.get("text") or "") for x in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://opsd-wordle-q36-teacher-sglang.apanda.svc.cluster.local:30000")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--num-states", type=int, default=24)
    ap.add_argument("--opener", default="slate")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--think", action="store_true", help="Let the model think before listing.")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    targets = rng.sample(list(wordle.WORD_LIST), args.num_states * 3)

    # 1) collect usable mid-game states + their prompts
    states = []
    for target in targets:
        if len(states) >= args.num_states:
            break
        hist, cands = _mid_state(target, args.opener.lower(), wordle.MAX_TURNS)
        if not (2 <= len(cands) <= 40):
            continue
        constraints = wordle.format_public_constraints(hist)
        transcript = "\n".join(f"{i}. {g.upper()} -> {fb}" for i, (g, fb) in enumerate(hist, 1))
        messages = [
            {"role": "system", "content": "You are an expert Wordle solver."},
            {"role": "user", "content": (
                "Wordle feedback so far (G=correct spot, Y=in word wrong spot, X=absent):\n"
                f"{transcript}\n\nDerived constraints:\n{constraints}\n\n"
                "List EVERY common 5-letter English word that is consistent with ALL of these "
                "constraints. Think step by step if needed, then end with a line:\n"
                "ANSWERS: WORD1, WORD2, WORD3, ...\n(uppercase, comma-separated)."
            )},
        ]
        states.append((target, hist, cands, messages))
    print(f"collected {len(states)} states; batch-generating (think={args.think}, mnt={args.max_new_tokens})...", flush=True)

    # 2) ONE batched request -> SGLang decodes all states concurrently
    texts = _gen_batch(args.base_url, tok, [s[3] for s in states], temperature=args.temperature,
                       max_new_tokens=args.max_new_tokens, enable_thinking=args.think)

    # 3) score each
    agg = {"recall": [], "validity": [], "n_true": [], "n_listed": [], "target_listed": [], "best_listed": []}
    records = []
    for i, ((target, hist, cands, _), text) in enumerate(zip(states, texts), 1):
        tail = text.split("</think>")[-1]
        m = re.search(r"ANSWERS\s*:(.*)", tail, re.S | re.I)
        scan = m.group(1) if m else tail
        listed = []
        for w in _WORD_RE.findall(scan):
            wl = w.lower()
            if wl not in listed:
                listed.append(wl)
        true_set = set(cands)
        consistent = [w for w in listed if all(wordle.compute_feedback(g, w) == fb for g, fb in hist)]
        hit = true_set.intersection(listed)
        recall = len(hit) / max(1, len(true_set))
        validity = len(consistent) / max(1, len(listed))
        _bsc = _split_scores(cands)
        best = min(cands, key=lambda w: (_bsc[w], w))
        agg["recall"].append(recall)
        agg["validity"].append(validity)
        agg["n_true"].append(len(true_set))
        agg["n_listed"].append(len(listed))
        agg["target_listed"].append(int(target.lower() in listed))
        agg["best_listed"].append(int(best in listed))
        rec = {"target": target, "history": hist, "n_true": len(true_set), "n_listed": len(listed),
               "recall": round(recall, 3), "validity": round(validity, 3),
               "target_listed": target.lower() in listed, "listed": listed[:60],
               "true": sorted(true_set)[:60], "raw_text": text}
        records.append(rec)
        print(f"[{i:2d}] tgt={target} n_true={len(true_set):2d} listed={len(listed):2d} "
              f"recall={recall:.2f} validity={validity:.2f} tgt_in_list={target.lower() in listed}", flush=True)
    used = len(records)

    def mean(x):
        return sum(x) / max(1, len(x))
    summary = {k: round(mean(v), 3) for k, v in agg.items()}
    summary["states"] = used
    summary["think"] = args.think
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json.dumps({"summary": summary}) + "\n")
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
