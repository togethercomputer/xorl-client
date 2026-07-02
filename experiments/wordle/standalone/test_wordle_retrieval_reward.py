"""Tests for wordle.wordle_retrieval_reward (retrieval-targeted reward).

Run: PYTHONPATH=<repo> /home/apanda/xorl-internal/.venv/bin/python -m pytest -q \
     experiments/wordle/standalone/test_wordle_retrieval_reward.py
"""
from __future__ import annotations

from experiments.wordle.standalone.tasks import wordle


def _play(target: str, guesses: list[str]) -> list[tuple[str, str]]:
    """Build a (guess, feedback) sequence by computing real env feedback."""
    return [(g, wordle.compute_feedback(g, target)) for g in guesses]


def _consistent_path(target: str, max_turns: int = 6) -> list[str]:
    """Greedy env-consistent play that does NOT necessarily solve (stop before solve)."""
    history: list[tuple[str, str]] = []
    guesses = ["slate"]
    history.append(("slate", wordle.compute_feedback("slate", target)))
    for _ in range(max_turns - 1):
        cands = [w for w in wordle.remaining_candidates(history) if w != target]
        if not cands:
            break
        g = cands[0]
        guesses.append(g)
        history.append((g, wordle.compute_feedback(g, target)))
    return guesses


def test_solve_dominates_nonsolve():
    target = "abide"
    solved_turns = _play(target, ["slate", target])  # solve on turn 2
    nonsolve_turns = _play(target, _consistent_path(target))  # consistent, never solves
    r_solve = wordle.wordle_retrieval_reward(solved_turns, solved=True)["wordle_retrieval_reward"]
    r_non = wordle.wordle_retrieval_reward(nonsolve_turns, solved=False)["wordle_retrieval_reward"]
    assert r_solve > r_non, (r_solve, r_non)
    # solve should be a clear margin above the best non-solve (audit: >= ~2x)
    assert r_solve >= r_non + 1.5, (r_solve, r_non)


def test_consistent_guesses_score_positive_no_violation():
    target = "abide"
    turns = _play(target, _consistent_path(target))
    out = wordle.wordle_retrieval_reward(turns, solved=False)
    assert out["wr_violate_turns"] == 0.0, out
    assert out["wr_consistent_turns"] >= 1.0, out
    assert out["wr_narrowing"] >= 0.0, out  # narrowing is always non-negative
    assert out["wr_consistency"] > 0.0, out


def test_constraint_violating_guess_penalized():
    target = "abide"
    # opener, then a guess that re-uses an absent letter / contradicts feedback.
    fb0 = wordle.compute_feedback("slate", target)
    # find a word NOT in remaining_candidates after the opener
    history = [("slate", fb0)]
    cands = set(wordle.remaining_candidates(history))
    violating = next(w for w in wordle.WORD_LIST if w not in cands)
    turns = _play(target, ["slate", violating])
    out = wordle.wordle_retrieval_reward(turns, solved=False)
    assert out["wr_violate_turns"] >= 1.0, out
    # a clean consistent non-solve must beat a violating one
    clean = wordle.wordle_retrieval_reward(_play(target, _consistent_path(target)), solved=False)
    assert clean["wordle_retrieval_reward"] > out["wordle_retrieval_reward"], (clean, out)


def test_invalid_penalty_applied():
    target = "abide"
    turns = _play(target, ["slate"])
    with_pen = wordle.wordle_retrieval_reward(turns, solved=False, invalid_action=True)
    without = wordle.wordle_retrieval_reward(turns, solved=False, invalid_action=False)
    assert with_pen["wordle_retrieval_reward"] < without["wordle_retrieval_reward"]
    assert with_pen["wr_invalid_penalty"] == 0.8


def test_fast_solve_beats_slow_solve():
    target = "abide"
    fast = wordle.wordle_retrieval_reward(_play(target, ["slate", target]), solved=True)
    slow = wordle.wordle_retrieval_reward(
        _play(target, ["slate"] + _consistent_path(target)[1:5] + [target]), solved=True)
    assert fast["wr_length_bonus"] >= slow["wr_length_bonus"]
