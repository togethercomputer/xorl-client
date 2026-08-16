from types import SimpleNamespace

from examples.wordle.reward import score_trajectory


def turn(*, format_ok=True, valid_guess=True, feedback="XXXXX"):
    return SimpleNamespace(
        format_ok=format_ok, valid_guess=valid_guess, feedback=feedback
    )


def test_solved_valid_trajectory_scores_all_components():
    score = score_trajectory(
        [turn(feedback="GXXXX"), turn(feedback="GGGGG")],
        solved=True,
        max_turns=6,
        mode="shaped",
    )
    assert score["exact_match"] == 1.0
    assert score["format_rate"] == 1.0
    assert score["valid_guess_rate"] == 1.0
    assert score["reward"] > 0.8


def test_unsolved_malformed_or_illegal_trajectory_loses_dense_credit():
    valid = score_trajectory(
        [turn(feedback="YYXXX")], solved=False, max_turns=6, mode="shaped"
    )
    malformed = score_trajectory(
        [turn(format_ok=False, valid_guess=False)],
        solved=False,
        max_turns=6,
        mode="shaped",
    )
    assert valid["exact_match"] == 0.0
    assert malformed["format_rate"] == 0.0
    assert malformed["valid_guess_rate"] == 0.0
    assert malformed["reward"] < valid["reward"]


def test_exact_match_mode_has_no_proxy_reward():
    score = score_trajectory(
        [turn(feedback="GGXXX")], solved=False, max_turns=6, mode="exact_match"
    )
    assert score["reward"] == 0.0
