from __future__ import annotations

import multiprocessing
import queue
import re
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


_ANSWER_ANCHOR_RE = re.compile(r"(?i)(?:^|\b)(?:final\s+answer|answer)\s*[:：]")
_REFERENCE_ANSWER_RE = re.compile(r"(?i)\bAnswer:\s*([^\n<]+)")
_BOXED_COMMAND_RE = re.compile(r"(?<![A-Za-z])(?:\\boxed|boxed)")
_BOXED_SPACE_RE = re.compile(r"(?<![A-Za-z])(?:\\boxed|boxed)\s+([A-Za-z0-9.+\-/]+)")
_FRAC_RE = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d*)?)?%?$")


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    reward: float
    predicted_answer: str | None
    gold_answer: str
    has_box: bool
    parse_ok: bool
    error: str | None = None


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    depth = 0
    for pos in range(open_pos, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
    return None


def _boxed_payloads(text: str) -> list[str]:
    payloads: list[str] = []
    start = 0
    while True:
        match = _BOXED_COMMAND_RE.search(text, start)
        if match is None:
            break
        brace_pos = match.end()
        while brace_pos < len(text) and text[brace_pos] in " \t":
            brace_pos += 1
        if brace_pos < len(text) and text[brace_pos] == "{":
            end_pos = _find_matching_brace(text, brace_pos)
            if end_pos is not None:
                payloads.append(text[brace_pos + 1 : end_pos])
                start = end_pos + 1
                continue
        start = match.end()

    if payloads:
        return payloads

    match = _BOXED_SPACE_RE.search(text)
    if match:
        return [match.group(1)]
    return []


def extract_boxed_answers(text: str, *, prefer_after_answer: bool = True) -> list[str]:
    """Extract boxed payloads, preferring boxes after the final Answer: anchor."""

    if prefer_after_answer:
        anchors = list(_ANSWER_ANCHOR_RE.finditer(text))
        for anchor in reversed(anchors):
            payloads = _boxed_payloads(text[anchor.end() :])
            if payloads:
                return payloads
    return _boxed_payloads(text)


def extract_last_boxed(text: str, *, prefer_after_answer: bool = True) -> str:
    """Return the last boxed payload.

    This mirrors Marin's tinker math environment convention of grading the last
    ``\boxed{...}``, with one reproduction-specific refinement: if an Answer:
    anchor exists, boxes after that anchor are preferred.
    """

    payloads = extract_boxed_answers(text, prefer_after_answer=prefer_after_answer)
    if not payloads:
        raise ValueError("No boxed strings found")
    return payloads[-1]


def extract_reference_final_answer(text: str, *, tail_chars: int = 300) -> tuple[str, bool]:
    """Extract the Marin/SkyRL final answer from the tail of a completion.

    Prefer a final `Answer:` line in the last 300 characters. If the sampled
    completion omits that anchor but still ends with a boxed answer, fall back
    to the last boxed payload in the same tail window.
    """

    tail = text[-tail_chars:] if tail_chars > 0 else text
    matches = list(_REFERENCE_ANSWER_RE.finditer(tail))
    if not matches:
        payloads = _boxed_payloads(tail)
        if payloads:
            return payloads[-1].strip(), True
        raise ValueError("No final Answer: anchor or boxed answer found in completion tail")
    raw_answer = matches[-1].group(1).strip()
    payloads = _boxed_payloads(raw_answer)
    if payloads:
        return payloads[-1].strip(), True
    return raw_answer.strip().strip("$").rstrip(".。").strip(), False


def _math_verify_timeout(timeout_s: int | None) -> int | None:
    return timeout_s if timeout_s is None or timeout_s > 0 else None


def _parse_math(answer: str, *, timeout_s: int | None) -> list[Any]:
    stripped = answer.strip()
    latex_wrapped = stripped if stripped.startswith("$") and stripped.endswith("$") else f"${stripped}$"
    parsing_timeout = _math_verify_timeout(timeout_s)
    parsed = parse(
        latex_wrapped,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        parsing_timeout=parsing_timeout,
    )
    if parsed:
        return parsed
    return parse(
        stripped,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        parsing_timeout=parsing_timeout,
    )


def _normalize_for_string_compare(value: str) -> str:
    normalized = value.strip().strip("$")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(" ", "")
    return normalized


def _strip_outer_group(value: str) -> str:
    pairs = {"(": ")", "[": "]"}
    if len(value) < 2 or value[0] not in pairs or value[-1] != pairs[value[0]]:
        return value
    return value[1:-1]


def _coordinate_tuple_equivalent(predicted_answer: str, gold_answer: str) -> bool:
    predicted = _normalize_for_string_compare(predicted_answer)
    gold = _normalize_for_string_compare(gold_answer)
    if "," not in predicted or "," not in gold:
        return False
    return _strip_outer_group(predicted) == _strip_outer_group(gold)


def _simple_number(value: str) -> Fraction | None:
    normalized = _normalize_for_string_compare(value)
    normalized = normalized.replace(",", "")
    frac_match = _FRAC_RE.match(normalized)
    if frac_match:
        try:
            return Fraction(frac_match.group(1)) / Fraction(frac_match.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    if not _NUMBER_RE.match(normalized):
        return None
    percent = normalized.endswith("%")
    if percent:
        normalized = normalized[:-1]
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            parsed = Fraction(numerator) / Fraction(denominator)
        else:
            parsed = Fraction(normalized)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed / 100 if percent else parsed


def _quick_answers_equivalent(predicted_answer: str, gold_answer: str) -> bool | None:
    predicted = _normalize_for_string_compare(predicted_answer)
    gold = _normalize_for_string_compare(gold_answer)
    if predicted == gold:
        return True
    if _coordinate_tuple_equivalent(predicted_answer, gold_answer):
        return True
    predicted_number = _simple_number(predicted_answer)
    gold_number = _simple_number(gold_answer)
    if predicted_number is not None and gold_number is not None:
        return predicted_number == gold_number
    return None


def _answers_equivalent_math_verify(
    predicted_answer: str,
    gold_answer: str,
    *,
    timeout_s: int | None,
    strict: bool,
) -> bool:
    predicted = _parse_math(predicted_answer, timeout_s=timeout_s)
    gold = _parse_math(gold_answer, timeout_s=timeout_s)
    if not predicted or not gold:
        return _normalize_for_string_compare(predicted_answer) == _normalize_for_string_compare(gold_answer)
    if bool(verify(gold, predicted, strict=strict, timeout_seconds=_math_verify_timeout(timeout_s))):
        return True
    return _coordinate_tuple_equivalent(predicted_answer, gold_answer)


def _process_target(result_queue, func, args: tuple, kwargs: dict) -> None:
    try:
        result_queue.put((True, func(*args, **kwargs)))
    except Exception as exc:  # pragma: no cover - exercised through parent re-raise path
        result_queue.put((False, repr(exc)))


def _call_with_process_timeout(func, *args, process_timeout_s: int | None, **kwargs):
    if process_timeout_s is None or process_timeout_s <= 0:
        return func(*args, **kwargs)
    context_name = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    ctx = multiprocessing.get_context(context_name)
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_process_target, args=(result_queue, func, args, kwargs))
    process.daemon = True
    process.start()
    process.join(process_timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        raise TimeoutError(f"{func.__name__} timed out after {process_timeout_s}s")
    try:
        success, result = result_queue.get(timeout=0.1)
    except queue.Empty as exc:
        raise RuntimeError(f"{func.__name__} exited without returning a result") from exc
    finally:
        result_queue.close()
        result_queue.join_thread()
    if not success:
        raise RuntimeError(str(result))
    return result


def answers_equivalent(predicted_answer: str, gold_answer: str, *, timeout_s: int | None = 5, strict: bool = True) -> bool:
    quick_result = _quick_answers_equivalent(predicted_answer, gold_answer)
    if quick_result is not None:
        return quick_result
    if threading.current_thread() is threading.main_thread():
        return _answers_equivalent_math_verify(
            predicted_answer,
            gold_answer,
            timeout_s=timeout_s,
            strict=strict,
        )
    return bool(
        _call_with_process_timeout(
            _answers_equivalent_math_verify,
            predicted_answer,
            gold_answer,
            process_timeout_s=timeout_s,
            timeout_s=timeout_s,
            strict=strict,
        )
    )


def grade_boxed_answer(
    completion: str,
    gold_answer: str,
    *,
    timeout_s: int | None = 5,
    prefer_after_answer: bool = True,
) -> VerificationResult:
    try:
        predicted = extract_last_boxed(completion, prefer_after_answer=prefer_after_answer)
    except ValueError as exc:
        return VerificationResult(
            correct=False,
            reward=0.0,
            predicted_answer=None,
            gold_answer=gold_answer,
            has_box=False,
            parse_ok=False,
            error=str(exc),
        )

    try:
        correct = answers_equivalent(predicted, gold_answer, timeout_s=timeout_s)
    except Exception as exc:
        return VerificationResult(
            correct=False,
            reward=0.0,
            predicted_answer=predicted,
            gold_answer=gold_answer,
            has_box=True,
            parse_ok=False,
            error=repr(exc),
        )

    return VerificationResult(
        correct=correct,
        reward=1.0 if correct else 0.0,
        predicted_answer=predicted,
        gold_answer=gold_answer,
        has_box=True,
        parse_ok=True,
    )


def grade_reference_final_answer(
    completion: str,
    gold_answer: str,
    *,
    timeout_s: int | None = 5,
    tail_chars: int = 300,
) -> VerificationResult:
    try:
        predicted, has_box = extract_reference_final_answer(completion, tail_chars=tail_chars)
    except ValueError as exc:
        return VerificationResult(
            correct=False,
            reward=0.0,
            predicted_answer=None,
            gold_answer=gold_answer,
            has_box=False,
            parse_ok=False,
            error=str(exc),
        )

    try:
        correct = answers_equivalent(predicted, gold_answer, timeout_s=timeout_s)
    except Exception as exc:
        return VerificationResult(
            correct=False,
            reward=0.0,
            predicted_answer=predicted,
            gold_answer=gold_answer,
            has_box=has_box,
            parse_ok=False,
            error=repr(exc),
        )

    return VerificationResult(
        correct=correct,
        reward=1.0 if correct else 0.0,
        predicted_answer=predicted,
        gold_answer=gold_answer,
        has_box=has_box,
        parse_ok=True,
    )
