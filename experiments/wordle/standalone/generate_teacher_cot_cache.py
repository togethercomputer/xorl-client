"""Pre-generate the reason-first teacher CoT cache for OPSD Wordle.

Phase A (harvest): play full games with the *student* policy settings
(think prompt style, sampling temperature) against base-weight samplers and
record every pre-action public state (target, history_before) — solved or
not. These are the states an early OPSD run will visit.

Phase B (cotgen): for each unique state, build the hinted teacher-reasoning
prompt (`response_style="teacher_reasoning"`, candidates hint included) and
generate the private teacher note once. Train-time, the trainer loads this
cache via --teacher-reasoning-cache and only falls back to online generation
on cache misses (policy drift).

Cache row format (JSONL):
    {"key": ..., "target": ..., "history": [[guess, feedback], ...],
     "teacher_reasoning_context": "<teacher_reasoning>...</teacher_reasoning>",
     "raw_text": ...}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone import train_opsd_baseline as opsd  # noqa: E402
from experiments.wordle.standalone.generate_wordle_gold_sft import _gen_turn, _post_generate  # noqa: E402
from experiments.wordle.standalone.tasks import wordle  # noqa: E402


teacher_cot_cache_key = opsd.teacher_cot_cache_key


def _harvest_game(
    base_url: str,
    tokenizer,
    *,
    target: str,
    prompt_style: str,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Play one game with the student policy; return every pre-action state."""
    history: list[tuple[str, str]] = []
    states: list[tuple[str, list[tuple[str, str]]]] = []
    for _turn in range(wordle.MAX_TURNS):
        states.append((target, list(history)))
        text = _gen_turn(
            base_url,
            tokenizer,
            target=target,
            history=history,
            gen_prompt_style=prompt_style,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
        ).strip()
        parsed = wordle.parse_turn_response(wordle.strip_think_prefix(text), history)
        guess = str(parsed.get("guess") or "")
        if not guess or not parsed.get("valid_guess"):
            break
        feedback = wordle.compute_feedback(guess, target)
        history.append((guess, feedback))
        if guess == target:
            break
    return states


def _generate_cot(
    base_url: str,
    tokenizer,
    *,
    target: str,
    history: list[tuple[str, str]],
    prompt_style: str,
    teacher_prompt_style: str,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
) -> tuple[str, str]:
    prompt_ids = opsd._wordle_teacher_reasoning_prompt_ids(
        tokenizer,
        wordle,
        target=target,
        history=history,
        prompt_style=prompt_style,
        teacher_prompt_style=teacher_prompt_style,
    )
    result = _post_generate(
        base_url,
        {
            "input_ids": [list(prompt_ids)],
            "sampling_params": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "ignore_eos": False,
                "stop": [],
            },
            "return_logprob": False,
        },
        timeout=timeout,
    )
    raw_text = str(result.get("text") or "")
    if not raw_text:
        output_ids = result.get("output_ids") or []
        if output_ids:
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # The note prompt renders closed-think; strip any think prefix defensively.
    raw_text = wordle.strip_think_prefix(raw_text)
    return opsd._normalize_teacher_reasoning_context(raw_text), raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-base-url", required=True, help="Comma-separated base-weight sampler URLs.")
    parser.add_argument("--cot-base-url", default="", help="URL for CoT generation; defaults to rollout URLs.")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--prompt-style", default="public_reasoning_constraints_think")
    parser.add_argument("--teacher-prompt-style", default="public_policy_hint")
    parser.add_argument("--num-targets", type=int, default=512)
    parser.add_argument("--passes", type=int, default=1, help="Harvest passes per target (coverage breadth).")
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=8192)
    parser.add_argument("--cot-temperature", type=float, default=0.2)
    parser.add_argument("--cot-max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--split-seed", type=int, default=9234)
    parser.add_argument("--train-pool-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--sample-eval-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rollout_urls = [u.strip() for u in args.rollout_base_url.split(",") if u.strip()]
    cot_urls = [u.strip() for u in (args.cot_base_url or args.rollout_base_url).split(",") if u.strip()]

    # Mirror the trainer's split (same logic as generate_wordle_gold_sft).
    rng = random.Random(args.split_seed)
    pool = list(wordle.WORD_LIST)
    rng.shuffle(pool)
    train_words = pool[: args.train_pool_size]
    eval_words = set(pool[args.train_pool_size : args.train_pool_size + max(args.eval_size, args.sample_eval_size)])
    targets = [w for w in train_words if w not in eval_words][: args.num_targets]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    # Phase A: harvest pre-action states. Turn-1 states are target-independent
    # for the candidates block but the reference action can depend on the
    # target near-tie break, so keep (target, history) keys throughout.
    seen: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    jobs = [(t, p) for p in range(args.passes) for t in targets]
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:

        def harvest(job: tuple[str, int]) -> list[tuple[str, list[tuple[str, str]]]]:
            target, pass_idx = job
            url = rollout_urls[(hash((target, pass_idx)) % len(rollout_urls))]
            return _harvest_game(
                url,
                tokenizer,
                target=target,
                prompt_style=args.prompt_style,
                temperature=args.rollout_temperature,
                max_new_tokens=args.rollout_max_new_tokens,
                timeout=args.timeout,
            )

        for i, states in enumerate(pool_exec.map(harvest, jobs), 1):
            for target, history in states:
                seen.setdefault(teacher_cot_cache_key(target, history), (target, history))
            if i % 32 == 0:
                print(
                    f"[harvest {i}/{len(jobs)}] unique_states={len(seen)} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    print(f"[harvest done] unique_states={len(seen)} elapsed={time.time() - started:.0f}s", flush=True)

    # Phase B: generate one teacher note per unique state.
    written = 0
    items = list(seen.items())
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec, out_path.open("w", encoding="utf-8") as f:

        def cotgen(item: tuple[str, tuple[str, list[tuple[str, str]]]]) -> dict[str, Any]:
            key, (target, history) = item
            url = cot_urls[hash(key) % len(cot_urls)]
            context, raw_text = _generate_cot(
                url,
                tokenizer,
                target=target,
                history=history,
                prompt_style=args.prompt_style,
                teacher_prompt_style=args.teacher_prompt_style,
                temperature=args.cot_temperature,
                max_new_tokens=args.cot_max_new_tokens,
                timeout=args.timeout,
            )
            return {
                "key": key,
                "target": target,
                "history": [[g, fb] for g, fb in history],
                "teacher_reasoning_context": context,
                "raw_text": raw_text,
            }

        for i, row in enumerate(pool_exec.map(cotgen, items), 1):
            f.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            if i % 64 == 0:
                print(f"[cotgen {i}/{len(items)}] elapsed={time.time() - started:.0f}s", flush=True)

    summary = {
        "rollout_base_url": rollout_urls,
        "cot_base_url": cot_urls,
        "prompt_style": args.prompt_style,
        "teacher_prompt_style": args.teacher_prompt_style,
        "targets": len(targets),
        "passes": args.passes,
        "unique_states": len(seen),
        "rows_written": written,
        "elapsed_sec": time.time() - started,
    }
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
