"""Production Wordle trajectory reward and invalid-action handling."""

from __future__ import annotations

from typing import Sequence


def information_score(feedback: str) -> float:
    greens = str(feedback).count("G")
    yellows = str(feedback).count("Y")
    return min(1.0, 0.5 * (greens / 5.0) + 0.5 * ((greens + yellows) / 5.0))


def shaped_reward(
    *,
    solved: bool,
    format_rate: float,
    valid_guess_rate: float,
    info_gain: float,
    turns_used: int,
    max_turns: int,
    invalid_action: bool,
) -> float:
    turn_bonus = ((max_turns - turns_used + 1) / max_turns) if solved else 0.0
    invalid_penalty = 0.45 if invalid_action else 0.0
    return float(
        0.55 * float(solved)
        + 0.15 * float(format_rate)
        + 0.20 * float(valid_guess_rate)
        + 0.15 * float(info_gain)
        + 0.10 * float(turn_bonus)
        - invalid_penalty
    )


def wordle_reward_components(
    *,
    solved: bool,
    turns_with_guess: int,
    latest_feedback: str,
    format_reward: float,
    valid_guess_rate: float,
    terminal_valid: bool,
    invalid_action: bool,
) -> dict[str, float]:
    correct = float(solved)
    partial = (
        0.0
        if solved
        else 0.2 * latest_feedback.count("G") + 0.1 * latest_feedback.count("Y")
    )
    length_bonus = correct / float(max(turns_with_guess, 1))
    invalid_penalty = 0.8 if invalid_action else 0.0
    terminal_valid_reward = 0.2 if terminal_valid else 0.0
    value = (
        correct
        + 0.5 * length_bonus
        + 0.3 * partial
        + 0.2 * float(format_reward)
        + 0.2 * float(valid_guess_rate)
        + terminal_valid_reward
        - invalid_penalty
    )
    return {
        "wordle_reward": float(value),
        "wordle_correct": correct,
        "wordle_partial": float(partial),
        "wordle_length_bonus": float(length_bonus),
        "wordle_format_reward": float(format_reward),
        "wordle_valid_guess_rate": float(valid_guess_rate),
        "wordle_terminal_valid": float(terminal_valid),
        "wordle_invalid_penalty": float(invalid_penalty),
    }


def score_trajectory(
    turns: Sequence[object],
    *,
    solved: bool,
    max_turns: int,
    mode: str,
    stopped_reason: str = "",
) -> dict[str, float]:
    """Score the same shaped reward used by the pinned production runner."""

    turns_used = len(turns)
    denominator = max(turns_used, 1)
    format_hits = sum(float(bool(getattr(turn, "format_ok", False))) for turn in turns)
    single_tag_hits = sum(
        float(
            bool(getattr(turn, "single_guess_tag", getattr(turn, "format_ok", False)))
        )
        for turn in turns
    )
    valid_hits = sum(float(bool(getattr(turn, "valid_guess", False))) for turn in turns)
    info_values = [
        information_score(str(getattr(turn, "feedback", "")))
        for turn in turns
        if bool(getattr(turn, "valid_guess", False))
        and str(getattr(turn, "feedback", ""))
    ]
    latest_feedback = next(
        (
            str(getattr(turn, "feedback", ""))
            for turn in reversed(turns)
            if str(getattr(turn, "feedback", ""))
        ),
        "",
    )
    format_rate = format_hits / denominator
    single_guess_tag_rate = single_tag_hits / denominator
    valid_guess_rate = valid_hits / denominator
    info_gain = sum(info_values) / max(len(info_values), 1)
    terminal_valid = bool(
        solved
        or (
            (
                stopped_reason == "max_turns"
                or (not stopped_reason and turns_used == max_turns)
            )
            and turns_used > 0
            and valid_hits == turns_used
        )
    )
    # Compatibility callers that score a shorter synthetic valid trajectory do
    # not provide a stop reason. It remains non-invalid while production rollout
    # always supplies an explicit terminal reason.
    if not stopped_reason and turns_used < max_turns and valid_hits == turns_used:
        terminal_valid = True
    invalid_action = bool(
        not terminal_valid
        or any(not bool(getattr(turn, "valid_guess", False)) for turn in turns)
    )
    shaped = shaped_reward(
        solved=solved,
        format_rate=format_rate,
        valid_guess_rate=valid_guess_rate,
        info_gain=info_gain,
        turns_used=turns_used,
        max_turns=max_turns,
        invalid_action=invalid_action,
    )
    components = wordle_reward_components(
        solved=solved,
        turns_with_guess=int(valid_hits),
        latest_feedback=latest_feedback,
        format_reward=single_guess_tag_rate,
        valid_guess_rate=valid_guess_rate,
        terminal_valid=terminal_valid,
        invalid_action=invalid_action,
    )
    reward = (
        float(solved)
        if mode == "exact_match"
        else components["wordle_reward"]
        if mode == "wordle"
        else shaped
    )
    return {
        "reward": float(reward),
        "exact_match": float(solved),
        "format_rate": float(format_rate),
        "single_guess_tag_rate": float(single_guess_tag_rate),
        "valid_guess_rate": float(valid_guess_rate),
        "info_gain": float(info_gain),
        "turn_bonus": (
            float((max_turns - turns_used + 1) / max_turns) if solved else 0.0
        ),
        "turns_used": float(turns_used),
        "invalid_action": float(invalid_action),
        "terminal_valid": float(terminal_valid),
        "strict_format_rate": sum(
            float(
                bool(
                    getattr(turn, "strict_format_ok", getattr(turn, "format_ok", False))
                )
            )
            for turn in turns
        )
        / denominator,
        "public_constraint_rate": sum(
            float(bool(getattr(turn, "public_constraint_valid", True)))
            for turn in turns
        )
        / denominator,
        "target_leak_rate": sum(
            float(bool(getattr(turn, "target_leak", False))) for turn in turns
        )
        / denominator,
        "extra_text_rate": sum(
            float(bool(getattr(turn, "extra_text", False))) for turn in turns
        )
        / denominator,
        **components,
    }
