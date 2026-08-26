import hashlib
import random
from pathlib import Path

import pytest

from examples.wordle.task import WordleTask, compute_feedback, parse_action

ROOT = Path(__file__).parents[3]
DATA = ROOT / "examples/wordle/data"


def _files(tmp_path: Path):
    words = ["abide", "above", "actor", "acute", "adieu", "crane"]
    train = tmp_path / "train.txt"
    eval_targets = tmp_path / "eval.txt"
    legal = tmp_path / "legal.txt"
    train.write_text("\n".join(words[:3]) + "\n")
    eval_targets.write_text("\n".join(words[3:5]) + "\n")
    legal.write_text("\n".join(words) + "\n")
    return train, eval_targets, legal


def test_feedback_handles_duplicate_letters():
    assert compute_feedback("alley", "apple") == "GYXYX"


def test_parsing_and_legal_repeated_illegal_guesses(tmp_path):
    train, eval_targets, legal = _files(tmp_path)
    task = WordleTask(
        train_targets_path=train,
        eval_targets_path=eval_targets,
        legal_guesses_path=legal,
        train_targets=3,
        eval_targets=2,
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
    train, eval_targets, legal = _files(tmp_path)
    task = WordleTask(
        train_targets_path=train,
        eval_targets_path=eval_targets,
        legal_guesses_path=legal,
        train_targets=3,
        eval_targets=2,
    )
    assert set(task.pools.train).isdisjoint(task.pools.held_out)
    selected = task.select_train_targets(step=1, count=3, seed=7)
    assert len(set(selected)) == 3
    assert selected == task.select_train_targets(step=1, count=3, seed=7)


def test_targets_must_be_legal(tmp_path):
    train, eval_targets, legal = _files(tmp_path)
    legal.write_text("abide\nabove\nactor\nacute\n")
    with pytest.raises(ValueError, match="target"):
        WordleTask(
            train_targets_path=train,
            eval_targets_path=eval_targets,
            legal_guesses_path=legal,
            train_targets=3,
            eval_targets=2,
        )


def test_duplicate_words_fail_closed(tmp_path):
    train, eval_targets, legal = _files(tmp_path)
    train.write_text("abide\nabide\nabove\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate word 'abide'"):
        WordleTask(
            train_targets_path=train,
            eval_targets_path=eval_targets,
            legal_guesses_path=legal,
            train_targets=3,
            eval_targets=2,
        )


def test_exact_dataset_split_and_training_schedule():
    task = WordleTask(
        train_targets_path=DATA / "train_targets.txt",
        eval_targets_path=DATA / "eval_targets.txt",
        legal_guesses_path=DATA / "legal_guesses.txt",
        train_targets=4000,
        eval_targets=170,
    )
    dictionary = tuple(
        (DATA / "legal_guesses.txt").read_text(encoding="utf-8").splitlines()
    )
    train = tuple((DATA / "train_targets.txt").read_text(encoding="utf-8").splitlines())
    held_out = tuple(
        (DATA / "eval_targets.txt").read_text(encoding="utf-8").splitlines()
    )

    assert len(dictionary) == len(set(dictionary)) == 4266
    assert dictionary == tuple(sorted(dictionary))
    assert task.answer_words == dictionary
    assert task.pools.train == train
    assert task.pools.held_out == held_out
    assert set(train).isdisjoint(held_out)
    assert set(train) | set(held_out) <= set(dictionary)
    assert len(set(dictionary) - set(train) - set(held_out)) == 96

    derived_eval = list(dictionary)
    random.Random(777).shuffle(derived_eval)
    derived_eval = derived_eval[:170]
    derived_train = [word for word in dictionary if word not in set(derived_eval)]
    random.Random(9234).shuffle(derived_train)
    assert held_out == tuple(derived_eval)
    assert train == tuple(derived_train[:4000])
    assert task.dataset_hashes["legal_guesses"] == (
        "16a7aadaf0fe977298b0705b8ac796fda47e996cf67f54bd589c90b158e48a9f"
    )
    assert task.dataset_hashes["train_targets"] == (
        "7b667321e1d7aba3e43afa9ddde9023e7e54b4a49e3ab93f7ae15f724d0c8dce"
    )
    assert task.dataset_hashes["eval_targets"] == (
        "2fb7634705b2d6d90b693e19c8d29518b11115f164fe2bfd79cd19667e769877"
    )

    schedule = [
        target
        for step in range(1, 129)
        for target in task.select_train_targets(step=step, count=64, seed=9234)
    ]
    schedule_sha256 = hashlib.sha256(
        "".join(f"{target}\n" for target in schedule).encode("utf-8")
    ).hexdigest()
    assert len(schedule) == 8192
    assert len(set(schedule)) == 4000
    assert schedule_sha256 == (
        "9f9e754aaa996c0a0e3cadc249d5007915f69f3cfafe772a58ddebb80c66ff4f"
    )


def test_explicit_split_count_and_overlap_fail_closed(tmp_path):
    train, eval_targets, legal = _files(tmp_path)
    with pytest.raises(ValueError, match="configured train_targets=4"):
        WordleTask(
            train_targets_path=train,
            eval_targets_path=eval_targets,
            legal_guesses_path=legal,
            train_targets=4,
            eval_targets=2,
        )

    eval_targets.write_text("actor\nacute\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be disjoint"):
        WordleTask(
            train_targets_path=train,
            eval_targets_path=eval_targets,
            legal_guesses_path=legal,
            train_targets=3,
            eval_targets=2,
        )
