"""Generate ALGORITHMIC enumeration-think gold for Wordle SFT.

Motivation (see OPSD_WORDLE_CANONICAL_RUNBOOK_2026_06_08.md): the whole solve-rate
bottleneck is *internal candidate enumeration*. Given the consistent-word list in the
prompt (scaffold) Qwen3.6-35B-A3B solves ~0.97; without it ~0.22-0.47. The model can
FILTER+PICK when handed candidates but cannot GENERATE the consistent list from feedback
constraints in its head. Distilling an LLM teacher's think is noisy (Kimi t=1.0 rambles,
only-solved coverage) and the reason-first CoT cache teaches post-hoc justification, not
enumeration.

This generator sidesteps the LLM entirely: the env computes the EXACT consistent-candidate
list (`wordle.remaining_candidates`) and the EXACT expected-remaining split score for any
state. We template a clean, terminating, fully-correct private think that:

  1. restates the public constraints (green pattern / present / absent / excluded slots),
  2. ENUMERATES the words still consistent with every clue (the env-computed list),
  3. picks the best public splitter (lowest expected-remaining), then
  4. closes </think> and emits the one-line public action `<reasoning>..</reasoning><guess>..`.

SFT on this teaches the enumerate->filter->pick procedure with PERFECT, FULL-COVERAGE
supervision (every target, every turn, no solved-only bias, no leak — the guess is chosen
by public split score only; the target is never referenced in the think text).

Output schema is a drop-in for `train_opsd_baseline.build_gold_sft_rows`
(objective=sft_gold): one JSON row per turn with `messages` (the *unscaffolded* student
think prompt) + `completion` (think + </think> + public line).

Pure offline: no GPU, no network. Deterministic given the split seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks import wordle  # noqa: E402


# --- exact public split scoring (reimplemented locally to avoid importing the
# heavy trainer module / torch). Mirrors train_opsd_baseline._wordle_policy_scores. ---
_EXACT_MAX_CANDIDATES = 400


def _split_scores(candidates: list[str]) -> dict[str, float]:
    """Expected remaining candidate count after each candidate guess (lower=better)."""
    n = len(candidates)
    scores: dict[str, float] = {}
    if n <= _EXACT_MAX_CANDIDATES:
        for guess in candidates:
            buckets: dict[str, int] = {}
            for tgt in candidates:
                fb = wordle.compute_feedback(guess, tgt)
                buckets[fb] = buckets.get(fb, 0) + 1
            # all-green ends the game -> leaves zero candidates.
            scores[guess] = sum(c * c for fb, c in buckets.items() if fb != "GGGGG") / n
    else:
        letter_counts: dict[str, int] = {}
        for word in candidates:
            for letter in set(word):
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
        for guess in candidates:
            coverage = sum(letter_counts.get(letter, 0) for letter in set(guess))
            scores[guess] = float(n) - coverage / 5.0
    return scores


def _best_public_guess(candidates: list[str]) -> str:
    scores = _split_scores(candidates)
    return min(candidates, key=lambda w: (scores[w], w))


# --- think rendering -------------------------------------------------------------------

def _constraint_lines(history: list[tuple[str, str]]) -> str:
    """Reuse the env's public constraint summary; it is exactly the public state."""
    return wordle.format_public_constraints(history)


def _render_think(
    history: list[tuple[str, str]],
    candidates: list[str],
    best: str,
    *,
    enum_cap: int,
    sample_cap: int,
) -> str:
    """Deterministic, terminating enumeration think. Public-only; no target reference."""
    n = len(candidates)
    constraints = _constraint_lines(history)
    parts: list[str] = []
    parts.append("Let me work from the public feedback only.")
    parts.append(f"Constraints so far:\n{constraints}")

    if n == 1:
        parts.append(
            f"Only one word in the answer list is still consistent with every clue: "
            f"{candidates[0].upper()}. That must be the answer, so I will guess it."
        )
    elif n <= enum_cap:
        listed = ", ".join(w.upper() for w in candidates)
        parts.append(
            f"The words still consistent with all the clues are ({n}): {listed}."
        )
        if n <= 8:
            parts.append(
                f"Checking which of these best narrows the rest, {best.upper()} gives the "
                f"most even split (lowest expected remaining), so I will guess {best.upper()}."
            )
        else:
            parts.append(
                f"Among these {n} candidates, {best.upper()} splits the remaining set most "
                f"evenly (lowest expected remaining), so it is the strongest guess."
            )
    else:
        sample = ", ".join(w.upper() for w in candidates[:sample_cap])
        parts.append(
            f"About {n} words still fit these clues, for example: {sample}. "
            f"That is too many to list, so I pick the consistent word that splits them most "
            f"efficiently: {best.upper()} (lowest expected remaining among the candidates)."
        )
    return "\n\n".join(parts)


def _public_reasoning(candidates: list[str], best: str, history: list[tuple[str, str]]) -> str:
    """One short public sentence (<15 words), no private/oracle language."""
    if not history:
        return "Open with a broad common word to gather maximum information."
    n = len(candidates)
    if n == 1:
        return "Only one word fits every clue, so I guess it."
    if n <= 6:
        return f"Only {n} words fit the clues; {best.upper()} splits them best."
    return f"{n} candidates fit the clues; {best.upper()} narrows them most."


def play_algo_game(
    *,
    target: str,
    opener: str,
    enum_cap: int,
    sample_cap: int,
    use_target_tiebreak: bool,
) -> dict[str, Any]:
    """Play one env-optimal hard-mode game, recording an algorithmic-think row per turn."""
    history: list[tuple[str, str]] = []
    turns: list[dict[str, Any]] = []
    solved = False
    for turn_idx in range(wordle.MAX_TURNS):
        if turn_idx == 0:
            guess = opener
            think = (
                "This is the opening guess with no constraints yet. "
                f"I will open with {opener.upper()}, a common word covering frequent letters "
                "to gather the most information."
            )
            candidates_for_reasoning: list[str] = []
        else:
            candidates = wordle.remaining_candidates(history)
            if not candidates:
                # Should not happen for a legal target; bail defensively.
                break
            guess = _best_public_guess(candidates)
            if use_target_tiebreak and target.lower() in candidates:
                scores = _split_scores(candidates)
                if scores[target.lower()] <= scores[guess] * 1.05 + 1e-9:
                    guess = target.lower()
            think = _render_think(
                history, candidates, guess, enum_cap=enum_cap, sample_cap=sample_cap
            )
            candidates_for_reasoning = candidates

        feedback = wordle.compute_feedback(guess, target)
        reasoning = _public_reasoning(candidates_for_reasoning, guess, history)
        completion = (
            f"{think}</think>"
            f"<reasoning>{reasoning}</reasoning><guess>{guess.upper()}</guess>"
        )
        messages = wordle._build_turn_messages(
            target=target, history=history, prompt_style="public_reasoning_constraints_think"
        )
        turns.append(
            {
                "history_before": list(history),
                "messages": messages,
                "completion": completion,
                "guess": guess,
                "feedback": feedback,
                "scaffold_leak": False,
                "think_tokens": len(think.split()),
            }
        )
        history.append((guess, feedback))
        if guess == target:
            solved = True
            break
    return {"target": target, "solved": solved, "turns": turns, "num_turns": len(history)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opener", default="slate", help="Fixed turn-1 opener.")
    parser.add_argument("--num-targets", type=int, default=512)
    parser.add_argument("--enum-cap", type=int, default=24, help="List the full consistent set up to this size.")
    parser.add_argument("--sample-cap", type=int, default=12, help="When the set is larger, list this many as examples.")
    parser.add_argument(
        "--use-target-tiebreak",
        action="store_true",
        help="Allow the private target to break within-5%% split ties (slightly higher coverage; "
        "never appears in the think text). Default off = purely public selection.",
    )
    parser.add_argument("--keep-unsolved", action="store_true",
                        help="Keep turns from games the env policy did not solve (each turn's think is still correct).")
    # Trainer-split mirror (exclude eval targets) — matches generate_wordle_gold_sft.py.
    parser.add_argument("--split-seed", type=int, default=9234)
    parser.add_argument("--train-pool-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--sample-eval-size", type=int, default=32)
    # Exclude the floor-protocol eval targets (eval_wordle_sglang.py: shuffle(seed)[offset:offset+count]).
    parser.add_argument("--exclude-eval-seed", type=int, default=777)
    parser.add_argument("--exclude-eval-offset", type=int, default=0)
    parser.add_argument("--exclude-eval-count", type=int, default=0,
                        help="Exclude the first N floor-eval targets (set ~128 to cover the seed-777 eval set).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.opener.lower() not in wordle.LEGAL_GUESSES:
        raise SystemExit(f"opener {args.opener!r} is not a legal guess")

    rng = random.Random(args.split_seed)
    pool = list(wordle.WORD_LIST)
    rng.shuffle(pool)
    train_words = pool[: args.train_pool_size]
    eval_words = set(pool[args.train_pool_size : args.train_pool_size + max(args.eval_size, args.sample_eval_size)])
    # Also exclude the held-out floor-eval targets (a DIFFERENT shuffle, seed 777 by
    # default) so generalization is measured on genuinely unseen targets.
    if args.exclude_eval_count > 0:
        erng = random.Random(args.exclude_eval_seed)
        epool = list(wordle.WORD_LIST)
        erng.shuffle(epool)
        eval_words |= set(epool[args.exclude_eval_offset : args.exclude_eval_offset + args.exclude_eval_count])
    targets = [w for w in train_words if w not in eval_words][: args.num_targets]
    print(f"[split] {len(targets)} train targets; excluded {len(eval_words)} eval/holdout words", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    solved = 0
    turn_count = 0
    think_turns = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, target in enumerate(targets, 1):
            game = play_algo_game(
                target=target,
                opener=args.opener.lower(),
                enum_cap=args.enum_cap,
                sample_cap=args.sample_cap,
                use_target_tiebreak=args.use_target_tiebreak,
            )
            if game["solved"]:
                solved += 1
            if not game["solved"] and not args.keep_unsolved:
                continue
            for turn in game["turns"]:
                turn_count += 1
                if len(turn["history_before"]) > 0:
                    think_turns += 1
                f.write(json.dumps({"target": game["target"], **turn}, sort_keys=True) + "\n")
            if i % 64 == 0:
                print(f"[{i}/{len(targets)}] solved={solved} turns_written={turn_count}", flush=True)

    summary = {
        "generator": "algo_think",
        "opener": args.opener.lower(),
        "enum_cap": args.enum_cap,
        "use_target_tiebreak": args.use_target_tiebreak,
        "keep_unsolved": args.keep_unsolved,
        "targets": len(targets),
        "solved": solved,
        "solve_rate": solved / max(1, len(targets)),
        "turns_written": turn_count,
        "enumeration_turns": think_turns,
        "word_list_source": wordle.WORD_LIST_SOURCE,
        "word_list_size": len(wordle.WORD_LIST),
    }
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
