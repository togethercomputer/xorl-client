"""Evaluate an SGLang-served model on multi-turn Wordle.

This is intentionally independent of the training loop: it plays full six-turn
games using the same Wordle parser, legality gate, prompt builder, and feedback
function used by the OPSD baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks import wordle  # noqa: E402


@dataclass
class TurnRecord:
    turn: int
    prompt_tokens: int
    raw_text: str
    guess: str | None
    single_guess_tag: bool
    valid_guess: bool
    feedback: str
    retry_count: int = 0
    rejected_attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GameState:
    index: int
    target: str
    history: list[tuple[str, str]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    solved: bool = False
    stopped_reason: str = ""


def _post_generate(
    base_url: str,
    *,
    input_ids_batch: list[list[int]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str],
    timeout: float,
    attempts: int,
    retry_interval: float,
    lora_path: str = "",
) -> list[dict[str, Any]]:
    payload = {
        "input_ids": [list(ids) for ids in input_ids_batch],
        "sampling_params": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "ignore_eos": bool(ignore_eos),
        },
        "return_logprob": False,
    }
    if stop:
        payload["sampling_params"]["stop"] = list(stop)
    if lora_path:
        payload["lora_path"] = lora_path
    url = base_url.rstrip("/") + "/generate"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json() if response.content else {}
            if isinstance(result, list):
                if len(result) != len(input_ids_batch):
                    raise RuntimeError(f"{url} returned {len(result)} rows for {len(input_ids_batch)} prompts")
                return [dict(item or {}) for item in result]
            if len(input_ids_batch) == 1:
                return [dict(result or {})]
            raise RuntimeError(f"{url} returned a non-list result for a batch: {result!r}")
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(retry_interval)
    raise RuntimeError(f"SGLang generate failed after {attempts} attempts: {last_error}") from last_error


def _generated_text(result: dict[str, Any], tokenizer) -> str:
    text = str(result.get("text") or "")
    finish_reason = (result.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish_reason, dict) and finish_reason.get("type") == "stop":
        matched_raw = finish_reason.get("matched")
        if isinstance(matched_raw, str) and matched_raw and not text.endswith(matched_raw):
            text = text + matched_raw
    if text:
        return text
    output_ids = result.get("output_ids")
    if isinstance(output_ids, list) and output_ids:
        return tokenizer.decode([int(token_id) for token_id in output_ids], skip_special_tokens=False)
    return ""


def _score_game(state: GameState) -> dict[str, Any]:
    turns_used = len(state.turns)
    format_hits = sum(1 for turn in state.turns if turn.single_guess_tag and turn.guess is not None)
    single_tag_hits = sum(1 for turn in state.turns if turn.single_guess_tag)
    valid_hits = sum(1 for turn in state.turns if turn.valid_guess)
    info_scores = [
        wordle._info_score(turn.guess, state.target, turn.feedback)
        for turn in state.turns
        if turn.valid_guess and turn.guess is not None and turn.feedback
    ]
    latest_feedback = ""
    for turn in reversed(state.turns):
        if turn.feedback:
            latest_feedback = turn.feedback
            break
    format_rate = format_hits / max(turns_used, 1)
    single_guess_tag_rate = single_tag_hits / max(turns_used, 1)
    valid_guess_rate = valid_hits / max(turns_used, 1)
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    turn_bonus = ((wordle.MAX_TURNS - turns_used + 1) / wordle.MAX_TURNS) if state.solved else 0.0
    reward = 0.4 * float(state.solved) + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus
    wordle_components = wordle._wordle_reward_components(
        solved=state.solved,
        turns_with_guess=single_tag_hits,
        latest_feedback=latest_feedback,
        format_reward=1.0 if turns_used > 0 and single_tag_hits == turns_used else 0.0,
    )
    return {
        "target": state.target,
        "solved": state.solved,
        "exact_match": float(state.solved),
        "turns_used": turns_used,
        "stopped_reason": state.stopped_reason,
        "format_rate": format_rate,
        "single_guess_tag_rate": single_guess_tag_rate,
        "valid_guess_rate": valid_guess_rate,
        "info_gain": info_gain,
        "reward": reward,
        **wordle_components,
    }


def _invalid_reason(text: str, guess: str | None, history: list[tuple[str, str]]) -> str:
    if not wordle.has_single_guess_tag(text):
        return "not_exactly_one_guess_tag"
    if guess is None:
        return "no_parseable_five_letter_guess"
    normalized = guess.lower()
    if normalized in {prior for prior, _ in history}:
        return "repeated_guess"
    if len(normalized) != 5 or not normalized.isalpha():
        return "not_five_alpha_letters"
    if normalized not in wordle.LEGAL_GUESSES:
        return "not_in_legal_wordle_dictionary"
    return "unknown_invalid"


def _build_retry_input_ids(
    tokenizer,
    *,
    target: str,
    history: list[tuple[str, str]],
    prompt_style: str,
    rejected_attempts: list[dict[str, Any]],
) -> list[int]:
    messages = wordle._build_turn_messages(target=target, history=history, prompt_style=prompt_style)
    rejected_words = [
        str(attempt.get("guess") or "").upper()
        for attempt in rejected_attempts
        if str(attempt.get("guess") or "")
    ]
    last = rejected_attempts[-1]
    messages.append({"role": "assistant", "content": str(last.get("raw_text") or "")})
    retry_note = (
        f"That response was rejected because: {last.get('reason')}. "
        "It did not count as a Wordle guess, and no Wordle feedback is available for it. "
    )
    if rejected_words:
        retry_note += f"Do not use these rejected guesses: {', '.join(rejected_words)}. "
    retry_note += (
        "Try again with exactly one real, common, legal five-letter Wordle word. "
        "Do not repeat any previous accepted guess. "
        "Respond exactly as <reasoning>one short sentence under 15 words</reasoning><guess>[WORD]</guess>"
    )
    messages.append({"role": "user", "content": retry_note})
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    )


def _pick_targets(*, count: int, seed: int, offset: int) -> list[str]:
    rng = random.Random(seed)
    pool = list(wordle.WORD_LIST)
    rng.shuffle(pool)
    if offset + count > len(pool):
        raise ValueError(f"requested offset={offset} count={count}, but Wordle pool has {len(pool)} targets")
    return pool[offset:offset + count]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model, trust_remote_code=True, local_files_only=True)
    targets = _pick_targets(count=args.num_games, seed=args.seed, offset=args.target_offset)
    states = [GameState(index=i, target=target.lower()) for i, target in enumerate(targets)]
    base_urls = [url.strip() for url in args.base_url.split(",") if url.strip()]
    if not base_urls:
        raise ValueError("--base-url must provide at least one URL")

    start = time.time()
    for turn_index in range(wordle.MAX_TURNS):
        active = [state for state in states if not state.solved and not state.stopped_reason]
        if not active:
            break
        prompts = [
            list(
                wordle._build_turn_input_ids(
                    tokenizer,
                    target=state.target,
                    history=state.history,
                    prompt_style=args.prompt_style,
                )
            )
            for state in active
        ]
        print(
            f"turn={turn_index + 1} active={len(active)} "
            f"avg_prompt_tokens={sum(len(ids) for ids in prompts) / max(len(prompts), 1):.1f}",
            flush=True,
        )
        for batch_start in range(0, len(active), args.batch_size):
            batch_states = active[batch_start:batch_start + args.batch_size]
            batch_prompts = prompts[batch_start:batch_start + args.batch_size]
            url = base_urls[(batch_start // args.batch_size) % len(base_urls)]
            results = _post_generate(
                url,
                input_ids_batch=batch_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                ignore_eos=args.ignore_eos,
                stop=args.stop,
                timeout=args.timeout,
                attempts=args.attempts,
                retry_interval=args.retry_interval,
                lora_path=args.lora_path,
            )
            attempts_by_state: dict[int, list[dict[str, Any]]] = {state.index: [] for state in batch_states}
            final_by_state: dict[int, tuple[str, str | None, bool, bool, list[int]]] = {}
            for state, prompt_ids, result in zip(batch_states, batch_prompts, results, strict=True):
                text = _generated_text(result, tokenizer)
                text = wordle.strip_think_prefix(text)
                guess = wordle.extract_guess(text)
                single_guess_tag = wordle.has_single_guess_tag(text)
                valid = wordle.is_valid_guess(guess, state.history)
                final_by_state[state.index] = (text, guess, single_guess_tag, valid, prompt_ids)
                if not valid:
                    attempts_by_state[state.index].append({
                        "raw_text": text,
                        "guess": guess,
                        "reason": _invalid_reason(text, guess, state.history),
                    })

            for retry_index in range(args.invalid_retries):
                retry_states = [
                    state for state in batch_states
                    if final_by_state[state.index][3] is False and not state.solved and not state.stopped_reason
                ]
                if not retry_states:
                    break
                retry_prompts = [
                    _build_retry_input_ids(
                        tokenizer,
                        target=state.target,
                        history=state.history,
                        prompt_style=args.prompt_style,
                        rejected_attempts=attempts_by_state[state.index],
                    )
                    for state in retry_states
                ]
                retry_results = _post_generate(
                    url,
                    input_ids_batch=retry_prompts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    ignore_eos=args.ignore_eos,
                    stop=args.stop,
                    timeout=args.timeout,
                    attempts=args.attempts,
                    retry_interval=args.retry_interval,
                    lora_path=args.lora_path,
                )
                for state, retry_prompt_ids, result in zip(retry_states, retry_prompts, retry_results, strict=True):
                    text = _generated_text(result, tokenizer)
                    text = wordle.strip_think_prefix(text)
                    guess = wordle.extract_guess(text)
                    single_guess_tag = wordle.has_single_guess_tag(text)
                    valid = wordle.is_valid_guess(guess, state.history)
                    final_by_state[state.index] = (text, guess, single_guess_tag, valid, retry_prompt_ids)
                    if not valid:
                        attempts_by_state[state.index].append({
                            "raw_text": text,
                            "guess": guess,
                            "reason": _invalid_reason(text, guess, state.history),
                            "retry_index": retry_index + 1,
                        })

            for state in batch_states:
                text, guess, single_guess_tag, valid, prompt_ids = final_by_state[state.index]
                feedback = ""
                if valid and guess is not None:
                    feedback = wordle.compute_feedback(guess, state.target)
                    state.history.append((guess, feedback))
                    if guess == state.target:
                        state.solved = True
                        state.stopped_reason = "solved"
                    elif len(state.history) >= wordle.MAX_TURNS:
                        state.stopped_reason = "max_turns"
                else:
                    state.stopped_reason = "invalid_or_unparseable"
                state.turns.append(
                    TurnRecord(
                        turn=turn_index + 1,
                        prompt_tokens=len(prompt_ids),
                        raw_text=text,
                        guess=guess,
                        single_guess_tag=single_guess_tag,
                        valid_guess=valid,
                        feedback=feedback,
                        retry_count=max(0, len(attempts_by_state[state.index]) - (0 if valid else 1)),
                        rejected_attempts=attempts_by_state[state.index],
                    )
                )

    for state in states:
        if not state.stopped_reason:
            state.stopped_reason = "max_turns"

    games = []
    for state in states:
        score = _score_game(state)
        games.append({
            **score,
            "index": state.index,
            "history": state.history,
            "turns": [asdict(turn) for turn in state.turns],
        })

    metric_keys = [
        "exact_match",
        "reward",
        "wordle_reward",
        "format_rate",
        "single_guess_tag_rate",
        "valid_guess_rate",
        "info_gain",
        "turns_used",
    ]
    summary = {
        "model": args.model,
        "tokenizer_path": args.tokenizer_path or args.model,
        "base_url": base_urls,
        "num_games": len(games),
        "seed": args.seed,
        "target_offset": args.target_offset,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "stop": args.stop,
        "invalid_retries": args.invalid_retries,
        "word_list_source": wordle.WORD_LIST_SOURCE,
        "legal_guesses_source": wordle.LEGAL_GUESSES_SOURCE,
        "word_list_size": len(wordle.WORD_LIST),
        "legal_guesses_size": len(wordle.LEGAL_GUESSES),
        "elapsed_sec": time.time() - start,
        "metrics": {
            key: sum(float(game[key]) for game in games) / max(len(games), 1)
            for key in metric_keys
        },
        "stopped_reasons": {},
    }
    for game in games:
        reason = str(game["stopped_reason"])
        summary["stopped_reasons"][reason] = int(summary["stopped_reasons"].get(reason, 0)) + 1
    return {"summary": summary, "games": games}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="SGLang base URL, or comma-separated URLs.")
    parser.add_argument("--model", required=True, help="Model id/path for tokenizer metadata.")
    parser.add_argument("--tokenizer-path", default="", help="Tokenizer id/path; defaults to --model.")
    parser.add_argument("--lora-path", default="", help="Served LoRA adapter name to eval (sglang lora_path).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9234)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        default="public_reasoning",
        choices=[
            "default",
            "public_reasoning",
            "public_reasoning_strict",
            "public_reasoning_strict_nocandidates",
            "public_reasoning_constraints",
            "public_reasoning_constraints_candidates",
            "public_reasoning_constraints_think",
            "public_reasoning_constraints_candidates_think",
            "candidate_list",
        ],
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=0,
        help="Same-turn retries after an illegal/unparseable/repeated guess. Retries receive no target feedback.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=["</guess>"],
        help="Stop string passed to SGLang. May be repeated; default stops after the guess tag.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.prompt_style.endswith("_think") and args.stop == ["</guess>"]:
        # The model often mentions the format (incl. </guess>) inside its
        # think text; the default stop truncates mid-think. Rely on EOS.
        args.stop = []
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    with (output_dir / "games.jsonl").open("w", encoding="utf-8") as f:
        for game in result["games"]:
            f.write(json.dumps(game, sort_keys=True) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
