"""Production Wordle state, public prompt contract, and action parsing."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MAX_TURNS = 6
MAX_REASONING_WORDS = 18

SHARED_PUBLIC_SYSTEM_PROMPT = (
    "You are playing Wordle. Follow the public Wordle prompt exactly."
)

_GUESS_RE = re.compile(r"<guess>\s*\[?\s*([A-Za-z]{5})\s*\]?\s*</guess>", re.IGNORECASE)
_GUESS_TAG_RE = re.compile(r"<guess\b[^>]*>.*?</guess>", re.IGNORECASE | re.DOTALL)
_REASONING_TAG_RE = re.compile(
    r"<(?P<tag>think|reasoning)>\s*(.*?)\s*</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_STRICT_REASONING_GUESS_RE = re.compile(
    r"^\s*<(?P<tag>think|reasoning)>\s*(?P<reasoning>.*?)\s*</(?P=tag)>\s*"
    r"<guess>\s*\[?\s*(?P<guess>[A-Za-z]{5})\s*\]?\s*</guess>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_REASONING_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bprivate\b",
        r"\bhint\b",
        r"\boracle\b",
        r"\bhidden (?:target|answer)\b",
        r"\bi know\b",
        r"\btarget (?:is|must be|has)\b",
        r"\banswer (?:is|must be)\b",
    )
)


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
    match = _GUESS_RE.search(text or "")
    return match.group(1).lower() if match is not None else None


def extract_guesses(text: str) -> list[str]:
    return [match.group(1).lower() for match in _GUESS_RE.finditer(text or "")]


def has_single_guess_tag(text: str) -> bool:
    return len(_GUESS_RE.findall(text or "")) == 1


def strip_think_prefix(text: str) -> str:
    if "</think>" in (text or ""):
        return (text or "").split("</think>")[-1].lstrip()
    return text or ""


def extract_action_text(text: str) -> str:
    """Return the first completed public action after a private think prefix."""

    action = strip_think_prefix(text or "")
    tag = _GUESS_TAG_RE.search(action)
    if tag is not None:
        action = action[: tag.end()]
    block_start = action.find("<reasoning>")
    if block_start < 0:
        block_start = action.find("<guess>")
    if block_start > 0:
        action = action[block_start:]
    return action


def parse_action(text: str) -> tuple[str | None, bool]:
    """Compatibility parser: first guess and exact-one-guess-only format."""

    stripped = (text or "").strip()
    matches = _GUESS_RE.findall(stripped)
    exact = bool(len(matches) == 1 and _GUESS_RE.fullmatch(stripped))
    return (matches[0].lower() if matches else None), exact


def format_public_constraints(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "No feedback yet."
    green_pattern = ["_"] * 5
    present_positions: dict[str, set[int]] = {}
    present_letters: set[str] = set()
    grey_letters: set[str] = set()
    for guess, feedback in history:
        for index, (letter, mark) in enumerate(
            zip(guess.lower(), feedback, strict=True)
        ):
            position = index + 1
            if mark == "G":
                green_pattern[index] = letter.upper()
                present_letters.add(letter)
            elif mark == "Y":
                present_letters.add(letter)
                present_positions.setdefault(letter, set()).add(position)
            elif mark == "X":
                grey_letters.add(letter)
                if letter in present_letters:
                    present_positions.setdefault(letter, set()).add(position)
    absent = sorted(grey_letters - present_letters)
    excluded = [
        f"{letter.upper()} not in position(s) "
        + ", ".join(str(value) for value in sorted(present_positions[letter]))
        for letter in sorted(present_positions)
    ]
    return "\n".join(
        (
            f"Green pattern: {' '.join(green_pattern)}",
            "Known present letters: "
            + (
                ", ".join(value.upper() for value in sorted(present_letters))
                or "(none)"
            ),
            "Excluded positions for present letters: "
            + ("; ".join(excluded) or "(none)"),
            "Letters likely absent: "
            + (", ".join(value.upper() for value in absent) or "(none)"),
        )
    )


def build_public_prompt_content(history: Sequence[tuple[str, str]]) -> str:
    if history:
        transcript = "\n".join(
            f"{index}. {guess.upper()} -> {feedback}"
            for index, (guess, feedback) in enumerate(history, 1)
        )
        previous = ", ".join(guess.upper() for guess, _ in history)
        constraints = format_public_constraints(history)
    else:
        transcript = "(none)"
        previous = "(none)"
        constraints = "No constraints yet."
    return (
        "You are playing Wordle. The hidden target is a 5-letter English word. "
        f"You have {MAX_TURNS} attempts.\n\n"
        "Feedback symbols:\n"
        "G = correct letter and correct position\n"
        "Y = letter is in the word but in the wrong position\n"
        "X = letter is not in the word\n\n"
        "Use only the public transcript and feedback in the public response.\n"
        "Do not claim to know a private target.\n\n"
        "State before your next guess:\n"
        f"Previous guesses and feedback:\n{transcript}\n\n"
        f"Public constraint summary:\n{constraints}\n\n"
        f"Already guessed, do not repeat:\n{previous}\n\n"
        "Think briefly in the required public style, then give one Wordle guess.\n\n"
        "Output exactly one line:\n"
        "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
        "Rules:\n"
        "- PUBLIC_REASONING must be one short sentence under 15 words.\n"
        "- PUBLIC_REASONING must not mention private targets, hints, hidden information, or oracle knowledge.\n"
        "- WORD must be exactly five alphabetic letters.\n"
        "- WORD must be a common valid Wordle answer word.\n"
        "- WORD must not repeat a previous guess.\n"
        "- After feedback is available, WORD must fit all public constraints.\n\n"
        "Begin your response now."
    )


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
        legal_words = _read_words(legal_guesses_path)
        self.legal_guesses = frozenset(legal_words)
        if not set(targets).issubset(self.legal_guesses):
            raise ValueError("every target must also be a legal guess")
        if train_targets + eval_targets > len(targets):
            raise ValueError(
                f"requested {train_targets + eval_targets} targets from {len(targets)}"
            )
        random.Random(seed).shuffle(targets)
        self.answer_words = tuple(targets)
        self.pools = TargetPools(
            train=tuple(targets[:train_targets]),
            held_out=tuple(targets[train_targets : train_targets + eval_targets]),
        )
        self.max_turns = max_turns
        self.targets_path = str(Path(targets_path).resolve())
        self.legal_guesses_path = str(Path(legal_guesses_path).resolve())

    def select_train_targets(self, *, step: int, count: int, seed: int) -> list[str]:
        """Select from one continuous cursor through shuffled target epochs."""

        if step < 1 or count < 1:
            raise ValueError("step and count must be positive")
        start = (step - 1) * count
        selected: list[str] = []
        while len(selected) < count:
            absolute = start + len(selected)
            epoch, offset = divmod(absolute, len(self.pools.train))
            epoch_words = list(self.pools.train)
            random.Random(int(seed) * 1_000_003 + epoch).shuffle(epoch_words)
            take = min(count - len(selected), len(epoch_words) - offset)
            selected.extend(epoch_words[offset : offset + take])
        return selected

    def valid_guess(
        self, guess: str | None, history: Sequence[tuple[str, str]]
    ) -> bool:
        return bool(
            guess
            and guess.lower() in self.legal_guesses
            and guess.lower() not in {prior.lower() for prior, _ in history}
        )

    def public_constraints_satisfied(
        self, guess: str | None, history: Sequence[tuple[str, str]]
    ) -> bool:
        return bool(
            guess
            and all(
                compute_feedback(prior_guess, guess) == feedback
                for prior_guess, feedback in history
            )
        )

    def remaining_candidates(self, history: Sequence[tuple[str, str]]) -> list[str]:
        return [
            word
            for word in self.answer_words
            if all(
                compute_feedback(guess, word) == feedback for guess, feedback in history
            )
        ]

    def parse_response(
        self, text: str, history: Sequence[tuple[str, str]]
    ) -> dict[str, object]:
        raw = text or ""
        stripped = raw.strip()
        guess_tags = _GUESS_TAG_RE.findall(raw)
        reasoning_matches = list(_REASONING_TAG_RE.finditer(raw))
        match = _STRICT_REASONING_GUESS_RE.fullmatch(stripped)
        guesses = extract_guesses(raw)
        guess = guesses[-1] if guesses else None
        reasoning = match.group("reasoning").strip() if match is not None else ""
        reasoning_words = re.findall(r"[A-Za-z0-9']+", reasoning)
        target_leak = any(
            pattern.search(reasoning) for pattern in _PRIVATE_REASONING_PATTERNS
        )
        extra_text = bool("\n" in stripped or "\r" in stripped or match is None)
        format_ok = bool(
            match is not None
            and len(guess_tags) == 1
            and len(reasoning_matches) == 1
            and len(reasoning_words) <= MAX_REASONING_WORDS
            and not target_leak
            and not extra_text
        )
        return {
            "guess": guess,
            "reasoning": reasoning,
            "single_guess_tag_ok": len(guesses) == 1,
            "format_ok": format_ok,
            "valid_guess": self.valid_guess(guess, history),
            "public_constraint_valid": self.public_constraints_satisfied(
                guess, history
            ),
            "target_leak": target_leak,
            "extra_text": extra_text,
        }

    def invalid_reason(
        self, text: str, guess: str | None, history: Sequence[tuple[str, str]]
    ) -> str:
        if not has_single_guess_tag(text):
            return "not_exactly_one_guess_tag"
        if guess is None:
            return "no_parseable_five_letter_guess"
        if guess.lower() in {prior.lower() for prior, _ in history}:
            return "repeated_guess"
        if guess.lower() not in self.legal_guesses:
            return "not_in_legal_wordle_dictionary"
        return "unknown_invalid"

    def prompt_messages(
        self, history: Sequence[tuple[str, str]]
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SHARED_PUBLIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_public_prompt_content(history),
            },
        ]

    def prompt_tokens(self, tokenizer, history: Sequence[tuple[str, str]]) -> list[int]:
        """Compatibility helper; adapters normally own prompt rendering."""

        messages = self.prompt_messages(history)
        if hasattr(tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": True,
                "return_dict": False,
            }
            try:
                return list(tokenizer.apply_chat_template(messages, **kwargs))
            except TypeError:
                kwargs.pop("enable_thinking")
                kwargs.pop("return_dict")
                return list(tokenizer.apply_chat_template(messages, **kwargs))
        text = "\n".join(f"{row['role']}: {row['content']}" for row in messages)
        return list(tokenizer.encode(text))
