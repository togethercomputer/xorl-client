"""Portable Wordle state, parsing, and deterministic target selection."""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MAX_TURNS = 6
_GUESS_RE = re.compile(r"<guess>\s*\[?\s*([A-Za-z]{5})\s*\]?\s*</guess>", re.I)
_THINK_RE = re.compile(r"<think\b.*?</think>", re.I | re.S)

SYSTEM_PROMPT = """You are playing Wordle. The hidden target is a five-letter English word.
G means correct letter and position, Y means present in another position, and X means absent.
Reply with exactly one five-letter guess as <guess>[WORD]</guess>. Do not repeat a guess."""


def _read_words(path: str | Path) -> tuple[str, ...]:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            if len(word) != 5 or not word.isascii() or not word.isalpha():
                raise ValueError(f"invalid five-letter word {word!r} in {path}")
            values.append(word)
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"word file is empty: {path}")
    return unique


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_feedback(guess: str, target: str) -> str:
    guess, target = guess.lower(), target.lower()
    if len(guess) != 5 or len(target) != 5:
        raise ValueError("guess and target must each have five letters")
    result = ["X"] * 5
    remaining = list(target)
    for index, letter in enumerate(guess):
        if letter == target[index]:
            result[index] = "G"
            remaining[index] = "_"
    for index, letter in enumerate(guess):
        if result[index] == "G":
            continue
        if letter in remaining:
            result[index] = "Y"
            remaining[remaining.index(letter)] = "_"
    return "".join(result)


def extract_guess(text: str) -> str | None:
    matches = _GUESS_RE.findall(text or "")
    return matches[0].lower() if matches else None


def parse_action(text: str) -> tuple[str | None, bool]:
    """Return the first guess and whether the response is exactly one guess tag.

    The environment plays the first completed action. Rollout construction uses
    the same boundary for policy-loss tokens, so malformed multi-action output
    cannot receive reward for a guess that is absent from the training row.
    """

    stripped = (text or "").strip()
    matches = _GUESS_RE.findall(stripped)
    # Format is graded on the post-thinking content: a thinking model's
    # response is `<think>...</think>` followed by the action, and the strict
    # exactly-one-tag contract applies to the action part. (A fullmatch on the
    # raw text silently zeroes format_ok for every thinking turn.)
    action_text = _THINK_RE.sub("", stripped).strip()
    exact = bool(len(matches) == 1 and _GUESS_RE.fullmatch(action_text))
    return (matches[0].lower() if matches else None), exact


@dataclass(frozen=True)
class TargetPools:
    train: tuple[str, ...]
    held_out: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.held_out)
        if overlap:
            raise ValueError(
                f"training and held-out pools overlap: {sorted(overlap)[:3]}"
            )


class WordleTask:
    def __init__(
        self,
        *,
        targets_path: str | Path,
        legal_guesses_path: str | Path,
        train_targets: int,
        eval_targets: int,
        seed: int,
        max_turns: int = MAX_TURNS,
    ) -> None:
        targets = list(_read_words(targets_path))
        self.legal_guesses = frozenset(_read_words(legal_guesses_path))
        if not set(targets).issubset(self.legal_guesses):
            raise ValueError("every target must also be a legal guess")
        if train_targets + eval_targets > len(targets):
            raise ValueError(
                f"requested {train_targets + eval_targets} targets from {len(targets)}"
            )
        random.Random(seed).shuffle(targets)
        self.pools = TargetPools(
            train=tuple(targets[:train_targets]),
            held_out=tuple(targets[train_targets : train_targets + eval_targets]),
        )
        self.max_turns = max_turns
        self.targets_path = str(Path(targets_path).resolve())
        self.legal_guesses_path = str(Path(legal_guesses_path).resolve())

    def select_train_targets(self, *, step: int, count: int, seed: int) -> list[str]:
        """Select cyclic shuffled epochs without replacement within each epoch."""

        if step < 1 or count < 1:
            raise ValueError("step and count must be positive")
        start = (step - 1) * count
        selected: list[str] = []
        while len(selected) < count:
            absolute = start + len(selected)
            epoch, offset = divmod(absolute, len(self.pools.train))
            epoch_words = list(self.pools.train)
            random.Random(seed + epoch).shuffle(epoch_words)
            take = min(count - len(selected), len(epoch_words) - offset)
            selected.extend(epoch_words[offset : offset + take])
        return selected

    def valid_guess(
        self, guess: str | None, history: Sequence[tuple[str, str]]
    ) -> bool:
        return bool(
            guess
            and guess in self.legal_guesses
            and guess not in {prior for prior, _ in history}
        )

    def prompt_messages(
        self, history: Sequence[tuple[str, str]]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Begin. You have six guesses."},
        ]
        for guess, feedback in history:
            messages.append(
                {"role": "assistant", "content": f"<guess>[{guess.upper()}]</guess>"}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{guess.upper()} -> {feedback}. Give the next guess.",
                }
            )
        return messages

    def prompt_tokens(self, tokenizer, history: Sequence[tuple[str, str]]) -> list[int]:
        messages = self.prompt_messages(history)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    # Thinking off by default (24-token budgets); opt in via
                    # env for large-budget runs that want reasoning turns.
                    enable_thinking=os.environ.get("WORDLE_ENABLE_THINKING") == "1",
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return list(tokenizer(text, add_special_tokens=False)["input_ids"])
        text = "\n".join(f"{row['role']}: {row['content']}" for row in messages)
        encoded = tokenizer.encode(text)
        return list(encoded)
