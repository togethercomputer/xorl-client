"""Generate gold Wordle SFT trajectories via scaffold distillation.

The generator plays full games with the candidate scaffold in the prompt
(`public_reasoning_constraints_candidates`, ~97% solve rate on Qwen3.6) but
records each turn against the *unscaffolded* student prompt
(`public_reasoning_constraints`). SFT on these rows teaches the student to
produce candidate-aware reasoning without being given the list — the
internalization the OPSD eval bottleneck calls for. A stronger external model
(e.g. Kimi-K2.6, once 2x8 GPUs are available) can be slotted in via
--base-url + --gen-prompt-style without changing the data format.

Only solved games are kept. Eval targets (same split logic as the trainer)
are excluded to avoid contamination.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks import wordle  # noqa: E402


_SCAFFOLD_LEAK_RE = re.compile(r"\b(list|listed|provided|given above|shown above)\b", re.IGNORECASE)


def _post_generate(base_url: str, payload: dict[str, Any], *, timeout: float, attempts: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(base_url.rstrip("/") + "/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            return result[0] if isinstance(result, list) else result
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(3.0 * attempt, 15.0))
    assert last is not None
    raise last


def _gen_turn(
    base_url: str,
    tokenizer,
    *,
    target: str,
    history: list[tuple[str, str]],
    gen_prompt_style: str,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
) -> str:
    input_ids = wordle._build_turn_input_ids(
        tokenizer, target=target, history=history, prompt_style=gen_prompt_style
    )
    # Think styles produce multi-line private reasoning, and the model often
    # MENTIONS the output format inside its think text — so neither newline
    # nor </guess> are safe stops there. Rely on natural EOS (the chat turn
    # ends right after the real guess tag).
    stop = [] if gen_prompt_style.endswith("_think") else ["\n"]
    result = _post_generate(
        base_url,
        {
            "input_ids": [list(input_ids)],
            "sampling_params": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "ignore_eos": False,
                "stop": stop,
            },
            "return_logprob": False,
        },
        timeout=timeout,
    )
    text = str(result.get("text") or "")
    if not text:
        output_ids = result.get("output_ids") or []
        if output_ids:
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
    finish = (result.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish, dict) and finish.get("type") == "stop":
        matched_raw = finish.get("matched")
        if isinstance(matched_raw, str) and matched_raw and not text.endswith(matched_raw):
            text = text + matched_raw
    return text


def play_game(
    base_url: str,
    tokenizer,
    *,
    target: str,
    gen_prompt_style: str,
    train_prompt_style: str,
    temperature: float,
    max_new_tokens: int,
    invalid_retries: int,
    timeout: float,
) -> dict[str, Any]:
    history: list[tuple[str, str]] = []
    turns: list[dict[str, Any]] = []
    solved = False
    for _turn in range(wordle.MAX_TURNS):
        text = ""
        parsed: dict[str, Any] = {}
        for _attempt in range(invalid_retries + 1):
            text = _gen_turn(
                base_url,
                tokenizer,
                target=target,
                history=history,
                gen_prompt_style=gen_prompt_style,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
            ).strip()
            # *_think styles: parse the public line after </think>; the stored
            # completion keeps the full think+answer (training supervises it).
            parsed = wordle.parse_turn_response(wordle.strip_think_prefix(text), history)
            guess = str(parsed.get("guess") or "")
            if guess and parsed.get("valid_guess") and parsed.get("public_constraint_valid"):
                break
        guess = str(parsed.get("guess") or "")
        if not guess or not parsed.get("valid_guess") or not parsed.get("public_constraint_valid"):
            break
        feedback = wordle.compute_feedback(guess, target)
        train_messages = wordle._build_turn_messages(
            target=target, history=history, prompt_style=train_prompt_style
        )
        turns.append(
            {
                "history_before": list(history),
                "messages": train_messages,
                "completion": text,
                "guess": guess,
                "feedback": feedback,
                "scaffold_leak": bool(_SCAFFOLD_LEAK_RE.search(str(parsed.get("reasoning") or ""))),
                "think_tokens": len(text.split("</think>")[0].split()) if "</think>" in text else 0,
            }
        )
        history.append((guess, feedback))
        if guess == target:
            solved = True
            break
    return {"target": target, "solved": solved, "turns": turns, "num_turns": len(history)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://opsd-wordle-q36-teacher-sglang.apanda.svc.cluster.local:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Tokenizer for the generating server.")
    parser.add_argument("--gen-prompt-style", default="public_reasoning_constraints_candidates")
    parser.add_argument("--train-prompt-style", default="public_reasoning_constraints")
    parser.add_argument("--num-targets", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--invalid-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--split-seed", type=int, default=9234, help="Trainer split seed; eval targets are excluded.")
    parser.add_argument("--train-pool-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--sample-eval-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Mirror the trainer's split: shuffle WORD_LIST with the trainer seed, the
    # first train_pool_size words are trainable; the next eval slice is held out.
    rng = random.Random(args.split_seed)
    pool = list(wordle.WORD_LIST)
    rng.shuffle(pool)
    train_words = pool[: args.train_pool_size]
    eval_words = set(pool[args.train_pool_size : args.train_pool_size + max(args.eval_size, args.sample_eval_size)])
    targets = [w for w in train_words if w not in eval_words][: args.num_targets]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    solved = 0
    turn_count = 0
    leak_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec, out_path.open("w", encoding="utf-8") as f:

        def run(target: str) -> dict[str, Any]:
            return play_game(
                args.base_url,
                tokenizer,
                target=target,
                gen_prompt_style=args.gen_prompt_style,
                train_prompt_style=args.train_prompt_style,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                invalid_retries=args.invalid_retries,
                timeout=args.timeout,
            )

        for i, game in enumerate(pool_exec.map(run, targets), 1):
            if game["solved"]:
                solved += 1
                for turn in game["turns"]:
                    turn_count += 1
                    leak_count += int(turn["scaffold_leak"])
                    f.write(json.dumps({"target": game["target"], **turn}, sort_keys=True) + "\n")
            if i % 32 == 0:
                print(
                    f"[{i}/{len(targets)}] solved={solved} turns={turn_count} "
                    f"leak_flagged={leak_count} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    summary = {
        "base_url": args.base_url,
        "gen_prompt_style": args.gen_prompt_style,
        "train_prompt_style": args.train_prompt_style,
        "targets": len(targets),
        "solved": solved,
        "solve_rate": solved / max(1, len(targets)),
        "turns_written": turn_count,
        "scaffold_leak_flagged": leak_count,
        "elapsed_sec": time.time() - started,
    }
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
