"""Wordle trajectory reward decomposition."""

from __future__ import annotations

from typing import Sequence


def score_trajectory(
    turns: Sequence[object], *, solved: bool, max_turns: int, mode: str
) -> dict[str, float]:
    """Score exact match, format, validity, information, and shaped reward."""

    count = max(len(turns), 1)
    format_rate = sum(float(bool(getattr(turn, "format_ok"))) for turn in turns) / count
    valid_rate = (
        sum(float(bool(getattr(turn, "valid_guess"))) for turn in turns) / count
    )
    information = 0.0
    for turn in turns:
        feedback = str(getattr(turn, "feedback", ""))
        information += (feedback.count("G") + 0.5 * feedback.count("Y")) / 5.0
    info_gain = information / count
    turns_used = len(turns)
    turn_bonus = ((max_turns - turns_used + 1) / max_turns) if solved else 0.0
    shaped = (
        0.55 * float(solved)
        + 0.15 * format_rate
        + 0.15 * valid_rate
        + 0.10 * info_gain
        + 0.05 * turn_bonus
    )
    reward = float(solved) if mode == "exact_match" else shaped
    return {
        "reward": reward,
        "exact_match": float(solved),
        "format_rate": float(format_rate),
        "valid_guess_rate": float(valid_rate),
        "info_gain": float(info_gain),
        "turn_bonus": float(turn_bonus),
        "turns_used": float(turns_used),
    }
