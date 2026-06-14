#!/usr/bin/env python3
"""Greedy accuracy on NEVER-TRAINED 4-digit multiplication pairs.

Separates memorization from algorithm internalization: the training pool has
8185 pairs out of ~81M possible; an algorithm generalizes to fresh pairs, a
lookup table cannot. Run against whatever weights the samplers currently hold
(probe-while-hot at the end of a run).

Usage: python fresh_pair_probe.py [--n 256] [--url http://...:30060] [--seed 0]
"""
import argparse
import json
import random
import re
import urllib.request

POOL = "/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json"
MP = ("/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
      "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--url", default="http://er-opd-q36-35b-slots-sglang-0:30060")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool_pairs = set()
    for p in json.load(open(POOL)):
        m = re.search(r"(\d+)\s*\*\s*(\d+)", p[0]["content"])
        pool_pairs.add((int(m.group(1)), int(m.group(2))))

    rng = random.Random(args.seed)
    fresh = []
    while len(fresh) < args.n:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        if (a, b) not in pool_pairs and (b, a) not in pool_pairs:
            fresh.append((a, b))

    ok = scored = 0
    for a, b in fresh:
        payload = {
            "model": "Qwen/Qwen3.6-35B-A3B", "max_tokens": 64, "temperature": 0.0,
            "top_p": 1.0, "top_k": -1, "n": 1, "logprobs": True, "logprob_start_len": 0,
            "messages": [
                {"role": "user", "content": f"/no_think Calculate: {a} * {b}"},
                {"role": "assistant", "content": "Answer: "},
            ],
            "stop": ["\n"], "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(args.url + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        text = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
        nm = re.search(r"([\d][\d,]*)", text)
        if nm:
            scored += 1
            try:
                ok += int(nm.group(1).replace(",", "")) == a * b
            except ValueError:
                pass
    print(f"FRESH-PAIR greedy accuracy: {ok}/{len(fresh)} = {ok/len(fresh):.3f} (scored {scored})")


if __name__ == "__main__":
    main()
