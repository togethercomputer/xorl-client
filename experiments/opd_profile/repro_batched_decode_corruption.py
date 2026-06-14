"""Reproducer / regression gate for SGLang batched-decode corruption.

See docs/notes/sglang_batched_decode_corruption_handoff.md. Pure-HTTP variant:
greedy 4x4-digit multiplication with an "Answer: " assistant prefill, swept over
client concurrency. On a healthy server accuracy is flat in concurrency; on the
regressed worktree it degrades monotonically (0.805 @ conc 8 -> 0.367 @ 128).

Usage:
  python repro_batched_decode_corruption.py --base-url http://localhost:30001 \
      --model Qwen/Qwen3.6-35B-A3B --conc 8 128 [--n 128] [--show-bad 3]

Pass criterion (handoff): acc(conc=128) within +-0.05 of acc(conc=8), cap-hit < 0.05.
"""

import argparse
import asyncio
import json
import random
import re

import httpx


PROMPTS_PATH = "/shared/opd-coord/randnum_4digit_8192_nonempty_cot.json"


def load_cases(n: int, digits: int = 0):
    if digits:
        # Synthetic easier task (e.g. 2x2-digit) for models that floor on 4x4.
        rng = random.Random(1234)
        lo, hi = 10 ** (digits - 1), 10**digits - 1
        return [
            (
                {"role": "user", "content": f"/no_think Calculate: {a} * {b}"},
                a * b,
            )
            for a, b in ((rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n))
        ]
    data = json.load(open(PROMPTS_PATH))[-1024:][:n]
    cases = []
    for msgs in data:
        text = msgs[0]["content"]
        a, b = map(int, re.findall(r"(\d+)\s*\*\s*(\d+)", text)[0])
        cases.append((msgs[0], a * b))
    return cases


async def one(client, sem, base_url, model, user_msg, answer, max_new):
    payload = {
        "model": model,
        "messages": [user_msg, {"role": "assistant", "content": "Answer: "}],
        "continue_final_message": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_new,
        "stop": ["\n"],
        # Match the OPD client request shape — full-sequence logprob return is part
        # of the failing path (see handoff doc).
        "logprobs": True,
        "logprob_start_len": 0,
    }
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
                r.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(2.0)
        choice = r.json()["choices"][0]
        text = choice["message"]["content"] or ""
        cap_hit = choice.get("finish_reason") == "length"
        # Base model emits comma-grouped digits ("17,625,192"); strip separators first.
        m = re.search(r"\d[\d,]*", text)
        correct = bool(m) and m.group().replace(",", "").isdigit() and int(m.group().replace(",", "")) == answer
        # Echo-format-robust metric: the correct product appears anywhere in the
        # completion (commas stripped). Distinguishes "model echoed the equation"
        # from genuinely corrupted output.
        contains = str(answer) in text.replace(",", "")
        return correct, cap_hit, text, contains


async def sweep(args):
    cases = load_cases(args.n, args.digits)
    async with httpx.AsyncClient() as client:
        for conc in args.conc:
            sem = asyncio.Semaphore(conc)
            results = await asyncio.gather(
                *[
                    one(client, sem, args.base_url, args.model, msg, ans, args.max_new_tokens)
                    for msg, ans in cases
                ]
            )
            acc = sum(r[0] for r in results) / len(results)
            cap = sum(r[1] for r in results) / len(results)
            has = sum(r[3] for r in results) / len(results)
            print(f"conc={conc:<4d} acc={acc:.3f} contains={has:.3f} cap_hit={cap:.3f} n={len(results)}")
            if args.show_bad:
                bad = [(c, r) for c, r in zip(cases, results) if not r[0]][: args.show_bad]
                for (msg, ans), (_, _, text, _) in bad:
                    print(f"    BAD: {msg['content']!r} expect={ans} got={text!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--conc", type=int, nargs="+", default=[8, 128])
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--digits", type=int, default=0, help="if >0, synthetic NxN-digit mult instead of the 4x4 file")
    p.add_argument("--show-bad", type=int, default=0)
    args = p.parse_args()
    asyncio.run(sweep(args))
