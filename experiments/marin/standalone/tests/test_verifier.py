import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from experiments.marin.standalone.tasks.verifier import (
    _call_with_process_timeout,
    answers_equivalent,
    extract_last_boxed,
    extract_reference_final_answer,
    grade_boxed_answer,
    grade_reference_final_answer,
)


def test_extract_last_boxed_handles_nested_latex() -> None:
    assert extract_last_boxed(r"scratch \boxed{1} Answer: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_last_boxed_prefers_answer_anchor() -> None:
    text = r"Earlier \boxed{999}. Answer: therefore \boxed{7}."
    assert extract_last_boxed(text) == "7"


def test_extract_last_boxed_supports_space_form() -> None:
    assert extract_last_boxed(r"Answer: \boxed 42") == "42"


def test_extract_last_boxed_supports_plain_boxed_form() -> None:
    assert extract_last_boxed(r"Answer: boxed{(3, \(\frac{\pi}{2}\))}") == r"(3, \(\frac{\pi}{2}\))"


def test_extract_last_boxed_ignores_boxed_inside_words() -> None:
    assert extract_last_boxed(r"unboxed{999}. Answer: boxed{7}") == "7"


def test_answers_equivalent_uses_math_verify_fraction_equivalence() -> None:
    assert answers_equivalent(r"\frac{1}{2}", "0.5")


def test_answers_equivalent_accepts_coordinate_tuple_without_outer_parens() -> None:
    assert answers_equivalent("-2, -2", "(-2, -2)")


def test_grade_boxed_answer_returns_binary_reward() -> None:
    result = grade_boxed_answer(r"Answer: \boxed{\frac{1}{2}}", "0.5")
    assert result.correct
    assert result.reward == 1.0
    assert result.has_box


def test_grade_boxed_answer_missing_box_is_incorrect() -> None:
    result = grade_boxed_answer("Answer: 2", "2")
    assert not result.correct
    assert result.reward == 0.0
    assert not result.has_box


def test_extract_reference_final_answer_falls_back_to_last_tail_boxed() -> None:
    result = grade_reference_final_answer(r"reasoning without anchor \boxed{(-2, -2)}", "(-2, -2)")

    assert result.correct
    assert result.reward == 1.0
    assert result.has_box


def test_extract_reference_final_answer_rejects_stale_box_outside_tail() -> None:
    text = r"scratch says \boxed{7} " + ("x" * 301)
    result = grade_reference_final_answer(text, "7")

    assert not result.correct
    assert not result.has_box
    assert result.error == "No final Answer: anchor or boxed answer found in completion tail"


def test_extract_reference_final_answer_uses_last_anchored_box() -> None:
    answer, has_box = extract_reference_final_answer(r"Answer: \boxed{1}\nwork\nAnswer: \boxed{\frac{1}{2}}")

    assert answer == r"\frac{1}{2}"
    assert has_box


def test_grade_reference_final_answer_accepts_plain_answer_but_marks_missing_box() -> None:
    result = grade_reference_final_answer("reasoning\nAnswer: 2", "2")

    assert result.correct
    assert result.reward == 1.0
    assert not result.has_box


def test_grade_reference_final_answer_works_inside_pipeline_worker_thread() -> None:
    def grade_in_worker():
        return grade_reference_final_answer(r"reasoning\nAnswer: \boxed{0}", "0")

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(grade_in_worker).result()

    assert result.correct
    assert result.reward == 1.0
    assert result.predicted_answer == "0"
    assert result.parse_ok
    assert result.error is None


def test_grade_reference_final_answer_worker_thread_forwards_math_verify_timeout() -> None:
    def grade_in_worker():
        return grade_reference_final_answer(r"reasoning\nAnswer: \boxed{\sqrt{4}}", "2")

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(grade_in_worker).result()

    assert result.correct
    assert result.reward == 1.0
    assert result.predicted_answer == r"\sqrt{4}"
    assert result.parse_ok
    assert result.error is None


def test_process_timeout_works_inside_pipeline_worker_thread() -> None:
    def run_timed_call():
        with pytest.raises(TimeoutError):
            _call_with_process_timeout(time.sleep, 2.0, process_timeout_s=0.2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run_timed_call).result()
