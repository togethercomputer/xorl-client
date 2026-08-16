from pathlib import Path

import pytest

from examples.wordle.task import WordleTask, compute_feedback, parse_action


def _files(tmp_path: Path):
    words = ["abide", "above", "actor", "acute", "adieu", "crane"]
    targets = tmp_path / "targets.txt"
    legal = tmp_path / "legal.txt"
    targets.write_text("\n".join(words[:5]) + "\n")
    legal.write_text("\n".join(words) + "\n")
    return targets, legal


def test_feedback_handles_duplicate_letters():
    assert compute_feedback("alley", "apple") == "GYXYX"


def test_parsing_and_legal_repeated_illegal_guesses(tmp_path):
    targets, legal = _files(tmp_path)
    task = WordleTask(
        targets_path=targets,
        legal_guesses_path=legal,
        train_targets=3,
        eval_targets=2,
        seed=1,
    )
    assert parse_action("<guess>[ABIDE]</guess>") == ("abide", True)
    assert parse_action("noise <guess>ABIDE</guess>") == ("abide", False)
    assert parse_action("<guess>ABIDE</guess> chatter <guess>CRANE</guess>") == (
        "abide",
        False,
    )
    assert parse_action("no guess") == (None, False)
    assert task.valid_guess("abide", [])
    assert not task.valid_guess("abide", [("abide", "XXXXX")])
    assert not task.valid_guess("zzzzz", [])


def test_pools_disjoint_and_shuffled_epoch_has_no_replacement(tmp_path):
    targets, legal = _files(tmp_path)
    task = WordleTask(
        targets_path=targets,
        legal_guesses_path=legal,
        train_targets=3,
        eval_targets=2,
        seed=9,
    )
    assert set(task.pools.train).isdisjoint(task.pools.held_out)
    selected = task.select_train_targets(step=1, count=3, seed=7)
    assert len(set(selected)) == 3
    assert selected == task.select_train_targets(step=1, count=3, seed=7)


def test_targets_must_be_legal(tmp_path):
    targets, legal = _files(tmp_path)
    legal.write_text("abide\nabove\nactor\nacute\n")
    with pytest.raises(ValueError, match="target"):
        WordleTask(
            targets_path=targets,
            legal_guesses_path=legal,
            train_targets=3,
            eval_targets=2,
            seed=1,
        )
