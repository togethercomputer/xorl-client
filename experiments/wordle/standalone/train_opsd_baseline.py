"""Native XORL OPSD baseline for Wordle.

This intentionally does not use PEFT.  It drives the XORL training server:

1. sample unhinted on-policy student traces from the current native LoRA,
2. create a native XORL LoRA session,
3. run a hinted teacher over those same student suffix tokens,
4. train the native LoRA adapter with ``loss_fn='opd_loss'``.

For multi-turn tasks such as Wordle, step 1 plays the actual environment:
each row is one sampled student turn, the next prompt is rebuilt from prior
guesses plus feedback, and the KL mask covers only the sampled assistant turn.

``teacher_forced_ce`` is kept as a supervised diagnostic baseline.  The old
``opd_self_kl`` objective is also kept as an explicit diagnostic mode; it is
degenerate because the frozen teacher and cold LoRA student are the same base
model at step 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer
from transformers.utils import cached_file


IGNORE_INDEX = -100
_GUESS_TAG_RE = re.compile(r"<guess>\s*\[?\s*([A-Za-z]{5})\s*\]?\s*</guess>", re.IGNORECASE)
_REASONING_TAG_RE = re.compile(
    r"<(?P<tag>think|reasoning)>\s*(.*?)\s*</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_OPEN_TAG_RE = re.compile(r"<(?:think|reasoning)>\s*", re.IGNORECASE)
_TEACHER_REASONING_TAG_RE = re.compile(
    r"<teacher_reasoning>\s*(.*?)\s*</teacher_reasoning>",
    re.IGNORECASE | re.DOTALL,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks.base import Example, load_task  # noqa: E402


@dataclass
class OpsdRow:
    project: str
    input_ids: list[int]
    labels: list[int]
    teacher_ids: list[int]
    teacher_weights: list[float]
    target_token_count: int
    target: str
    teacher_target_text: str
    teacher_cache_indices: list[int] | None = None
    student_input_ids: list[int] | None = None
    student_labels: list[int] | None = None
    teacher_input_ids: list[int] | None = None
    teacher_labels: list[int] | None = None
    student_text: str = ""


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _raise_on_failed_future(result: dict[str, Any], context: str) -> dict[str, Any]:
    if result.get("type") == "request_failed":
        raise RuntimeError(f"{context} failed: {result.get('error', result)}")
    if result.get("error"):
        raise RuntimeError(f"{context} failed: {result['error']}")
    return result


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def wait_for_future(train_url: str, request_id: str, *, timeout: float, poll_interval: float = 1.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _post_json(
            f"{train_url}/api/v1/retrieve_future",
            {"request_id": request_id},
            timeout=120.0,
        )
        if result.get("type") == "try_again":
            time.sleep(poll_interval)
            continue
        return result
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def call_future(
    train_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    context: str,
    submit_timeout: float = 120.0,
    future_timeout: float = 7200.0,
) -> dict[str, Any]:
    future = _post_json(f"{train_url}{endpoint}", payload, timeout=submit_timeout)
    request_id = future.get("request_id")
    if not request_id:
        raise RuntimeError(f"{context} did not return request_id: {future}")
    return _raise_on_failed_future(
        wait_for_future(train_url, str(request_id), timeout=future_timeout),
        context,
    )


def wait_for_training_service(train_url: str, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{train_url}/health", timeout=5)
            if resp.ok and resp.json().get("engine_running"):
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Training service did not become ready at {train_url} within {timeout}s")


def resolve_model_dir(model: str, *, local_files_only: bool) -> str:
    candidate = Path(model)
    if candidate.is_dir():
        return str(candidate)
    errors: list[str] = []
    for filename in ("model.safetensors.index.json", "model.safetensors"):
        try:
            return str(Path(cached_file(model, filename, local_files_only=local_files_only)).parent)
        except Exception as exc:  # pragma: no cover - depends on local HF cache
            errors.append(f"{filename}: {exc}")
    raise RuntimeError(f"Could not resolve local safetensors directory for {model!r}: {'; '.join(errors)}")


def build_opsd_rows(
    *,
    tokenizer,
    task,
    examples: list[Example],
    trace_args: SimpleNamespace,
    max_length: int,
) -> list[OpsdRow]:
    if not hasattr(task, "build_teacher_forced_example"):
        raise RuntimeError(f"Task {task.__name__!r} does not implement build_teacher_forced_example")

    rows: list[OpsdRow] = []
    for example in examples:
        teacher_example = task.build_teacher_forced_example(tokenizer, example, args=trace_args)
        input_ids = list(teacher_example.prompt_ids)
        target_count = int(teacher_example.metadata["teacher_target_token_count"])
        if target_count <= 0:
            raise RuntimeError(f"{example.project}: teacher target has no tokens")
        if len(input_ids) > max_length:
            raise RuntimeError(f"{example.project}: sequence length {len(input_ids)} exceeds --max-length={max_length}")
        if len(input_ids) <= target_count:
            raise RuntimeError(f"{example.project}: target suffix leaves no prefix token")

        labels = [IGNORE_INDEX] * len(input_ids)
        start = len(input_ids) - target_count
        labels[start:] = input_ids[start:]
        teacher_weights = [1.0 if label != IGNORE_INDEX else 0.0 for label in labels]
        rows.append(
            OpsdRow(
                project=example.project,
                input_ids=input_ids,
                labels=labels,
                teacher_ids=[0] * len(input_ids),
                teacher_weights=teacher_weights,
                target_token_count=target_count,
                target=str(example.metadata.get("target", "")),
                teacher_target_text=str(teacher_example.metadata.get("teacher_target_text", "")),
            )
        )
    return rows


def _split_urls(values: list[str] | None) -> list[str]:
    stop_aliases = {
        "__NEWLINE__": "\n",
        "__NL__": "\n",
        "\\n": "\n",
        "__DOUBLE_NEWLINE__": "\n\n",
        "\\n\\n": "\n\n",
    }
    urls: list[str] = []
    for value in values or []:
        for part in str(value).split():
            if part == "__NONE__":
                # Explicit "no stop strings" (think styles stop on EOS only);
                # an empty env value would fall through to the manifest default.
                continue
            urls.append(stop_aliases.get(part, part))
    return urls


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _asymmetric_prompt_ids(
    tokenizer,
    task,
    example: Example,
    *,
    hinted: bool,
    teacher_prompt_style: str = "answer_hint",
) -> list[int]:
    if not hasattr(task, "SYSTEM_PROMPT"):
        raise RuntimeError("asymmetric_opsd currently requires the Wordle task SYSTEM_PROMPT")
    target = str(example.metadata["target"]).upper()
    messages = [{"role": "system", "content": task.SYSTEM_PROMPT}]
    if hinted and teacher_prompt_style != "no_hint":
        content = (
            "You are a teacher solving one Wordle game for distillation. "
            f"Private hint: the target word is {target}. "
            "Write a compact hidden rationale, then emit complete five-letter guesses in "
            "<guess>[WORD]</guess> tags that solve the game. Do not leave any guess tag empty."
        )
    else:
        content = (
            "You are solving one Wordle game. The target word is hidden from you. "
            "Write a compact hidden rationale, then emit at least two complete five-letter guesses in "
            "<guess>[WORD]</guess> tags. If uncertain, use common Wordle openers; do not leave any guess tag empty."
        )
    messages.append({"role": "user", "content": content})
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    )


def _wordle_public_candidates(task, *, target: str, history: list[tuple[str, str]]) -> list[str]:
    if not hasattr(task, "remaining_candidates"):
        return []
    candidates = list(task.remaining_candidates(history))
    if not hasattr(task, "compute_feedback"):
        return candidates

    for candidate in candidates:
        for guess, feedback in history:
            if task.compute_feedback(guess, candidate) != feedback:
                raise RuntimeError(
                    f"candidate {candidate!r} is inconsistent with public feedback {guess!r}->{feedback!r}"
                )

    if history and target and target.lower() not in candidates:
        raise RuntimeError(f"target {target!r} is missing from public candidates for history={history!r}")

    previous_guesses = {guess.lower() for guess, _ in history}
    repeated = previous_guesses.intersection(candidates)
    if repeated:
        raise RuntimeError(f"previous guesses unexpectedly remain public candidates: {sorted(repeated)}")
    return candidates


def _format_wordle_transcript(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{guess.upper()} -> {feedback}" for guess, feedback in history)


def _format_wordle_candidates(candidates: list[str]) -> str:
    if not candidates:
        return "(none)"
    max_display = 200
    displayed = candidates[:max_display]
    text = ", ".join(word.upper() for word in displayed)
    if len(candidates) > max_display:
        text += f"\n(displaying first {max_display} of {len(candidates)} candidates)"
    return text


def _wordle_reference_action(
    candidates: list[str],
    *,
    target: str,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    if not history:
        return "stare", "Choose a broad opener with common letters."
    target_lower = target.lower()
    if target_lower in candidates:
        reference_guess = target_lower
    elif candidates:
        reference_guess = candidates[0]
    else:
        reference_guess = target_lower

    if len(candidates) == 1:
        reference_reasoning = "Only one candidate fits all feedback."
    elif len(candidates) <= 5:
        reference_reasoning = "Choose a remaining candidate that fits all feedback."
    else:
        reference_reasoning = "Choose a common candidate consistent with the clues."
    return reference_guess, reference_reasoning


# Public-policy reference action: the reference guess must be derivable from
# public state alone, with the private target allowed only to break near-ties.
_POLICY_EXACT_MAX_CANDIDATES = 200
_POLICY_NEAR_TIE_RATIO = 1.05
_POLICY_SCORE_CACHE: dict[tuple[str, ...], dict[str, float]] = {}
_POLICY_SCORE_CACHE_MAX_ENTRIES = 4096


def _wordle_policy_scores(task, candidates: list[str]) -> dict[str, float]:
    """Expected remaining candidate count after each candidate guess (lower is better).

    Exact partition scoring is O(n^2) in the candidate count, so above
    `_POLICY_EXACT_MAX_CANDIDATES` it falls back to a letter-coverage proxy
    on the same lower-is-better scale.
    """
    key = tuple(candidates)
    cached = _POLICY_SCORE_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(candidates)
    scores: dict[str, float] = {}
    if n <= _POLICY_EXACT_MAX_CANDIDATES:
        for guess in candidates:
            buckets: dict[str, int] = {}
            for candidate_target in candidates:
                feedback = task.compute_feedback(guess, candidate_target)
                buckets[feedback] = buckets.get(feedback, 0) + 1
            # The all-green bucket ends the game, so it leaves zero candidates.
            scores[guess] = sum(count * count for feedback, count in buckets.items() if feedback != "GGGGG") / n
    else:
        letter_counts: dict[str, int] = {}
        for word in candidates:
            for letter in set(word):
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
        for guess in candidates:
            coverage = sum(letter_counts.get(letter, 0) for letter in set(guess))
            scores[guess] = float(n) - coverage / 5.0
    if len(_POLICY_SCORE_CACHE) >= _POLICY_SCORE_CACHE_MAX_ENTRIES:
        _POLICY_SCORE_CACHE.clear()
    _POLICY_SCORE_CACHE[key] = scores
    return scores


def _wordle_public_policy_action(
    task,
    candidates: list[str],
    *,
    target: str,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Reference action for `public_policy_hint`: best public candidate split.

    Unlike `_wordle_reference_action`, the target never overrides a strictly
    better public guess; it only breaks near-ties among similar-quality guesses.
    """
    if not history:
        return "stare", "Choose a broad opener with common letters."
    if not candidates:
        return target.lower(), "Pick a common word consistent with all feedback."
    n = len(candidates)
    if n == 1:
        return candidates[0], "Only one word fits every clue."
    scores = _wordle_policy_scores(task, candidates)
    best_guess = min(candidates, key=lambda word: (scores[word], word))
    near_tie_cutoff = scores[best_guess] * _POLICY_NEAR_TIE_RATIO + 1e-9
    target_lower = target.lower()
    reference_guess = best_guess
    if target_lower in scores and scores[target_lower] <= near_tie_cutoff:
        reference_guess = target_lower
    if n <= 6:
        reference_reasoning = f"Only {n} words fit the clues; {reference_guess.upper()} splits them best."
    else:
        reference_reasoning = f"{n} candidates fit the clues; {reference_guess.upper()} narrows them most."
    return reference_guess, reference_reasoning


def _wordle_is_valid_guess(task, guess: str | None, history: list[tuple[str, str]]) -> bool:
    if guess is None or len(guess) != 5 or not guess.isalpha():
        return False
    if hasattr(task, "is_valid_guess"):
        return bool(task.is_valid_guess(guess, history))
    return guess.lower() not in {prior for prior, _ in history}


def _wordle_policy_hint_user_content(
    task,
    *,
    target: str,
    history: list[tuple[str, str]],
    response_style: str = "guess_only",
    teacher_prompt_style: str = "policy_hint",
    student_prompt_style: str = "",
) -> str:
    target_upper = target.upper()
    candidates_only = teacher_prompt_style == "public_candidates_only"
    public_policy = teacher_prompt_style == "public_policy_hint" or candidates_only
    # Prefix alignment: the teacher's shared public block must match what the
    # student saw, including the candidate scaffold when the student has it.
    shared_include_candidates = "candidates" in student_prompt_style
    candidates = _wordle_public_candidates(task, target=target, history=history)
    if public_policy:
        reference_guess, reference_reasoning = _wordle_public_policy_action(
            task, candidates, target=target, history=history
        )
    else:
        reference_guess, reference_reasoning = _wordle_reference_action(candidates, target=target, history=history)
    # Candidate-scaffold-only teacher (public_candidates_only): show the consistent-word
    # list but NOT the computed best guess — the teacher must reason over the candidates
    # to pick one itself (reason-first CoT). This avoids handing the student an answer it
    # cannot derive, and the teacher's own CoT becomes the distillation signal.
    if candidates_only:
        guess_hint_block = "Reason over the candidates above yourself to choose the strongest narrowing guess."
    else:
        guess_hint_block = f"Strongest guess: {reference_guess.upper()}\nWhy it narrows best: {reference_reasoning}"
    if hasattr(task, "build_shared_public_prompt_content"):
        try:
            shared_public_prompt = task.build_shared_public_prompt_content(
                history, include_begin=False, include_candidates=shared_include_candidates
            )
        except TypeError:
            shared_public_prompt = task.build_shared_public_prompt_content(history)
    else:
        transcript = _format_wordle_transcript(history)
        previous = ", ".join(guess.upper() for guess, _ in history) if history else "(none)"
        shared_public_prompt = (
            "You are playing Wordle. The hidden target is a 5-letter English word.\n\n"
            "State before your next guess:\n"
            f"Previous guesses and feedback:\n{transcript}\n\n"
            f"Already guessed, do not repeat:\n{previous}\n\n"
            "Output exactly one line:\n"
            "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
            "Begin your response now."
        )

    if not history:
        if public_policy:
            # PUBLIC framing (2026-06-13): the candidate analysis is derivable from the
            # public transcript, so it is presented as public — NOT "private reference".
            # The prior "Private reference / Reference next guess" framing leaked: the
            # student (trained on its own sampled reasoning) confabulated "Private target:
            # STARE". Public framing removes the leak vector while keeping the teacher's
            # candidate-enumeration edge in the GUESS choice.
            private_reference = (
                "Public candidate analysis (derivable from the transcript above):\n"
                "<candidate_analysis>\n"
                "Strategy: choose the valid word that best narrows the remaining candidates.\n"
                "Remaining candidate answers before this guess: any common opener works on turn 1.\n"
                f"{guess_hint_block}\n"
                "</candidate_analysis>"
            )
        else:
            private_reference = (
                "Private reference information for OPSD teacher only:\n"
                "<private_reference>\n"
                "Target word: omitted on the first turn.\n"
                "Remaining public candidate answers before this guess: omitted on turn 1.\n"
                f"Reference next guess: {reference_guess.upper()}\n"
                f"Reference public reasoning: {reference_reasoning}\n"
                "</private_reference>"
            )
    elif public_policy:
        private_reference = (
            "Public candidate analysis (derivable from the transcript above):\n"
            "<candidate_analysis>\n"
            "Strategy: choose the valid word that best narrows the remaining candidates.\n"
            "Remaining candidate answers consistent with the feedback so far:\n"
            f"count = {len(candidates)}\n"
            f"{_format_wordle_candidates(candidates)}\n"
            f"{guess_hint_block}\n"
            "</candidate_analysis>"
        )
    else:
        private_reference = (
            "Private reference information for OPSD teacher only:\n"
            "<private_reference>\n"
            f"Target word: {target_upper}\n"
            "Remaining public candidate answers before this guess:\n"
            f"count = {len(candidates)}\n"
            f"{_format_wordle_candidates(candidates)}\n"
            f"Reference next guess: {reference_guess.upper()}\n"
            f"Reference public reasoning: {reference_reasoning}\n"
            "</private_reference>"
        )

    if response_style == "public_reasoning":
        response_contract = (
            "Private teacher note:\n"
            f"The reference guess is {reference_guess.upper()}. Prefer the public-facing rationale: "
            f"\"{reference_reasoning}\" Do not reveal the private target or private reference.\n\n"
            "After understanding the private reference, continue in exactly the same public response format:\n"
            "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
            "Begin your response now."
        )
    elif response_style == "guess_only":
        response_contract = (
            "Private teacher note:\n"
            f"The reference guess is {reference_guess.upper()}. Do not reveal the private reference.\n\n"
            "After understanding the private reference, continue in exactly the same public response format:\n"
            "<guess>WORD</guess>\n\n"
            "Begin your response now."
        )
    elif response_style == "public_reasoning_think":
        if public_policy:
            if candidates_only:
                note = (
                    "Candidate-analysis note:\n"
                    "Reason over the candidate analysis above to choose the strongest narrowing guess.\n\n"
                )
            else:
                note = (
                    "Candidate-analysis note:\n"
                    f"The strongest guess (it narrows the candidates most) is {reference_guess.upper()}. "
                    f"Explain it with the public rationale: \"{reference_reasoning}\"\n\n"
                )
            response_contract = (
                note
                + "Continue in exactly the same response format the public prompt requires: think "
                "(enumerate the consistent words and pick the best split) privately first, then output "
                "the public line:\n"
                "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
                "Begin your response now."
            )
        else:
            response_contract = (
                "Private teacher note:\n"
                f"The reference guess is {reference_guess.upper()}. Prefer the public-facing rationale: "
                f"\"{reference_reasoning}\" Do not reveal the private target or private reference.\n\n"
                "After understanding the private reference, continue in exactly the same response format the "
                "public prompt requires: think privately first, then output the public line:\n"
                "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
                "Begin your response now."
            )
    elif response_style == "teacher_reasoning":
        if candidates_only:
            response_contract = (
                "Reason privately over the candidate analysis above to choose the strongest narrowing guess.\n"
                "Reply with exactly one teacher_reasoning tag and no guess tag:\n"
                "<teacher_reasoning>PRIVATE_TEACHER_NOTE</teacher_reasoning>\n"
                "In one or two concise sentences, name the candidate you would guess next and why it best "
                "narrows the remaining candidates. Use only public reasoning (no hidden/oracle target)."
            )
        else:
            response_contract = (
                "Before evaluating the student's next response, write a private teacher rationale.\n"
                "Reply with exactly one teacher_reasoning tag and no guess tag:\n"
                "<teacher_reasoning>PRIVATE_TEACHER_NOTE</teacher_reasoning>\n"
                "The note should be one or two concise sentences. It may use the private reference, but it "
                "must prefer public-facing reasoning and must not invent feedback for the student's current sampled guess."
            )
    else:
        raise ValueError(f"unknown Wordle response style: {response_style!r}")

    return (
        f"{shared_public_prompt}\n\n"
        f"{private_reference}\n\n"
        f"{response_contract}"
    )


def _wordle_teacher_system_prompt() -> str:
    return "You are playing Wordle. Follow the public Wordle prompt exactly."


def _wordle_teacher_score_transition_content(*, response_style: str) -> str:
    if response_style == "public_reasoning":
        return (
            "After understanding the private reference and teacher_reasoning note, continue in exactly "
            "the same public response format:\n"
            "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
            "The public reasoning must be one sentence under 15 words and must not reveal private information.\n"
            "Begin your response now."
        )
    if response_style == "public_reasoning_think":
        return (
            "After understanding the private reference and teacher_reasoning note, continue in exactly "
            "the same response format the public prompt requires: think privately first, then output "
            "the public line:\n"
            "<reasoning>PUBLIC_REASONING</reasoning><guess>WORD</guess>\n\n"
            "The public reasoning must be one sentence under 15 words and must not reveal private information.\n"
            "Begin your response now."
        )
    if response_style == "guess_only":
        return (
            "After understanding the private reference and teacher_reasoning note, continue in exactly "
            "the same public response format:\n"
            "<guess>WORD</guess>\n\n"
            "Begin your response now."
        )
    raise ValueError(f"unknown Wordle response style: {response_style!r}")


def teacher_cot_cache_key(target: str, history: list[tuple[str, str]]) -> str:
    """Stable key for the offline reason-first teacher CoT cache."""
    return json.dumps([target.lower(), [[g.lower(), f] for g, f in history]], sort_keys=True)


_TEACHER_COT_CACHE: dict[str, dict[str, str]] = {}


def _load_teacher_reasoning_cache(path: str) -> dict[str, str]:
    cache = _TEACHER_COT_CACHE.get(path)
    if cache is None:
        cache = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                context = str(row.get("teacher_reasoning_context") or "")
                if context:
                    cache[str(row["key"])] = context
        _TEACHER_COT_CACHE[path] = cache
        print(f"[teacher_cot_cache] loaded {len(cache)} states from {path}", flush=True)
    return cache


def _normalize_teacher_reasoning_context(text: str, *, max_chars: int = 1200) -> str:
    raw = (text or "").strip()
    match = _TEACHER_REASONING_TAG_RE.search(raw)
    if match is not None:
        content = match.group(1).strip()
    else:
        reasoning_match = _REASONING_TAG_RE.search(raw)
        if reasoning_match is not None:
            content = reasoning_match.group(2).strip()
        else:
            content = re.split(r"<guess\b", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    content = re.sub(r"</?teacher_reasoning>", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"</?(?:think|reasoning)>", "", content, flags=re.IGNORECASE).strip()
    if len(content) > max_chars:
        content = content[:max_chars].rstrip()
    if not content:
        content = "Apply the public Wordle constraints and restricted tie-break rule."
    return f"<teacher_reasoning>{content}</teacher_reasoning>"


def _wordle_teacher_reasoning_prompt_ids(
    tokenizer,
    task,
    *,
    target: str,
    history: list[tuple[str, str]],
    prompt_style: str = "default",
    teacher_prompt_style: str = "answer_hint",
) -> list[int]:
    if teacher_prompt_style not in {"policy_hint", "public_policy_hint", "public_candidates_only"}:
        raise RuntimeError("teacher reason-first is currently implemented for policy-hint prompts only")
    messages = [
        {"role": "system", "content": getattr(task, "SHARED_PUBLIC_SYSTEM_PROMPT", _wordle_teacher_system_prompt())},
        {
            "role": "user",
            "content": _wordle_policy_hint_user_content(
                task,
                target=target,
                history=history,
                response_style="teacher_reasoning",
                teacher_prompt_style=teacher_prompt_style,
                student_prompt_style=prompt_style,
            ),
        },
    ]
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    )


def _wordle_turn_prompt_ids(
    tokenizer,
    task,
    *,
    target: str,
    history: list[tuple[str, str]],
    hinted: bool,
    prompt_style: str = "default",
    teacher_prompt_style: str = "answer_hint",
    teacher_reasoning_context: str = "",
) -> list[int]:
    if (not hinted or teacher_prompt_style == "no_hint") and hasattr(task, "_build_turn_input_ids"):
        return list(task._build_turn_input_ids(tokenizer, target=target, history=history, prompt_style=prompt_style))
    if not hasattr(task, "SYSTEM_PROMPT"):
        raise RuntimeError("multi-turn asymmetric_opsd requires task.SYSTEM_PROMPT")

    target_upper = target.upper()
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    system_prompt = task.SYSTEM_PROMPT
    if prompt_style == "public_reasoning" and hasattr(task, "PUBLIC_REASONING_SYSTEM_PROMPT"):
        system_prompt = task.PUBLIC_REASONING_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    if hinted and teacher_prompt_style in {"policy_hint", "public_policy_hint"}:
        messages = [
            {"role": "system", "content": getattr(task, "SHARED_PUBLIC_SYSTEM_PROMPT", _wordle_teacher_system_prompt())}
        ]
        if prompt_style.endswith("_think"):
            response_style = "public_reasoning_think"
        elif prompt_style in {
            "public_reasoning",
            "public_reasoning_strict",
            "public_reasoning_strict_nocandidates",
            "public_reasoning_constraints",
            "public_reasoning_constraints_candidates",
        }:
            response_style = "public_reasoning"
        else:
            response_style = "guess_only"
        if teacher_reasoning_context:
            messages = [
                {"role": "system", "content": getattr(task, "SHARED_PUBLIC_SYSTEM_PROMPT", _wordle_teacher_system_prompt())},
                {
                    "role": "user",
                    "content": _wordle_policy_hint_user_content(
                        task,
                        target=target,
                        history=history,
                        response_style="teacher_reasoning",
                        teacher_prompt_style=teacher_prompt_style,
                        student_prompt_style=prompt_style,
                    ),
                },
                {
                    "role": "assistant",
                    "content": _normalize_teacher_reasoning_context(teacher_reasoning_context),
                },
                {
                    "role": "user",
                    "content": _wordle_teacher_score_transition_content(response_style=response_style),
                },
            ]
        else:
            messages.append({
                "role": "user",
                "content": _wordle_policy_hint_user_content(
                    task,
                    target=target,
                    history=history,
                    response_style=response_style,
                    teacher_prompt_style=teacher_prompt_style,
                    student_prompt_style=prompt_style,
                ),
            })
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                # The teacher scores the student's sampled tokens; under *_think
                # styles those begin inside an open think block, so the teacher
                # context must open one too or the CoT is scored out-of-distribution.
                enable_thinking=prompt_style.endswith("_think"),
                return_dict=False,
            )
        )
    elif hinted:
        first_turn = (
            "Private teacher hint: the target word is "
            f"{target_upper}. Use the visible Wordle transcript exactly, then make your first guess."
        )
    else:
        first_turn = "Make your first guess."
    messages.append({"role": "user", "content": first_turn})

    for turn_idx, (guess, feedback) in enumerate(history):
        guess_upper = guess.upper()
        messages.append({"role": "assistant", "content": f"<guess>[{guess_upper}]</guess>"})
        remaining = max_turns - (turn_idx + 1)
        if hasattr(task, "render_feedback_message"):
            feedback_text = task.render_feedback_message(guess, feedback, remaining)
        else:
            feedback_text = f"Feedback: {feedback} (for guess {guess_upper}). You have {remaining} guess(es) left."
        if guess == target or remaining == 0:
            messages.append({"role": "user", "content": f"{feedback_text}\nGame over."})
            continue
        content = (
            f"{feedback_text}\n"
            "Make your next guess."
        )
        if hinted and teacher_prompt_style == "policy_hint":
            candidates = task.remaining_candidates(history[: turn_idx + 1]) if hasattr(task, "remaining_candidates") else []
            candidate_text = ", ".join(word.upper() for word in candidates) if candidates else "(unavailable)"
            content += (
                f" Public candidates remaining ({len(candidates)}): {candidate_text}. "
                f"Private target for validation only: {target_upper}. "
                "Pick a public-valid guess; use the private target only as a tie-breaker."
            )
        elif hinted:
            content += f" Private teacher hint reminder: the target word is {target_upper}."
        messages.append({"role": "user", "content": content})

    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
    )


def _token_char_offsets(tokenizer, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        pieces.append(piece)
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def _token_indices_overlapping_span(offsets: list[tuple[int, int]], span_start: int, span_end: int) -> list[int]:
    return [
        idx
        for idx, (start, end) in enumerate(offsets)
        if max(start, span_start) < min(end, span_end)
    ]


def _truncate_output_ids_after_first_guess(
    tokenizer,
    output_ids: list[int],
    *,
    think: bool = False,
) -> tuple[list[int], str, bool]:
    if not output_ids:
        return [], "", False
    decoded, offsets = _token_char_offsets(tokenizer, output_ids)
    search_start = 0
    if think:
        # Think text routinely mentions the output format (incl. <guess> tags);
        # only the first guess after the think block closes is the action.
        close = decoded.find("</think>")
        if close < 0:
            return output_ids, decoded, False
        search_start = close + len("</think>")
    match = _GUESS_TAG_RE.search(decoded, search_start)
    if match is None:
        return output_ids, decoded, False

    keep_until = match.end()
    keep_count = 0
    for idx, (start, _end) in enumerate(offsets):
        if start >= keep_until:
            break
        keep_count = idx + 1
    if keep_count <= 0:
        return output_ids, decoded, False
    truncated_ids = output_ids[:keep_count]
    truncated_text = decoded[:keep_until]
    return truncated_ids, truncated_text, keep_count < len(output_ids)


def _wordle_output_weights(
    tokenizer,
    output_ids: list[int],
    *,
    turn_weight: float,
    tag_token_weight: float,
    reasoning_token_weight: float,
    think_token_weight: float = 1.0,
    think: bool = False,
) -> tuple[list[float], int, int]:
    if not output_ids:
        return [], 0, 0
    decoded, offsets = _token_char_offsets(tokenizer, output_ids)
    turn_weight = float(turn_weight)
    tag_token_weight = float(tag_token_weight)
    reasoning_token_weight = float(reasoning_token_weight)
    think_token_weight = float(think_token_weight)
    weights = [turn_weight * tag_token_weight for _ in output_ids]

    # Under *_think styles the sampled output starts inside an open think
    # block; everything before </think> is CoT content (the tag mentions
    # inside it are not structure) and the public line follows the close.
    public_start = 0
    if think:
        close = decoded.find("</think>")
        think_end = close if close >= 0 else len(decoded)
        for idx in _token_indices_overlapping_span(offsets, 0, think_end):
            weights[idx] = turn_weight * think_token_weight
        public_start = close + len("</think>") if close >= 0 else len(decoded)

    reasoning_indices: list[int] = []
    for match in _REASONING_TAG_RE.finditer(decoded, public_start):
        reasoning_indices.extend(_token_indices_overlapping_span(offsets, *match.span(2)))
    if not reasoning_indices:
        open_match = _REASONING_OPEN_TAG_RE.search(decoded, public_start)
        if open_match is not None:
            guess_match = _GUESS_TAG_RE.search(decoded, open_match.end())
            fallback_end = guess_match.start() if guess_match is not None else len(decoded)
            reasoning_indices.extend(_token_indices_overlapping_span(offsets, open_match.end(), fallback_end))
    reasoning_indices = sorted(set(reasoning_indices))
    if reasoning_indices:
        for idx in reasoning_indices:
            weights[idx] = turn_weight * reasoning_token_weight

    match = _GUESS_TAG_RE.search(decoded, public_start)
    if match is None:
        return weights, 0, len(reasoning_indices)

    content_indices = _token_indices_overlapping_span(offsets, *match.span(1))
    if not content_indices:
        return weights, 0, len(reasoning_indices)

    for idx in content_indices:
        weights[idx] = turn_weight
    return weights, len(content_indices), len(reasoning_indices)


def build_multiturn_asymmetric_opsd_rows(
    *,
    tokenizer,
    task,
    examples: list[Example],
    infer_urls: list[str],
    lora_name: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    generation_batch_size: int,
    generation_workers: int | None,
    max_length: int,
    generation_log_path: Path,
    wordle_prompt_style: str,
    wordle_teacher_prompt_style: str,
    wordle_first_turn_weight: float,
    wordle_tag_token_weight: float,
    wordle_reasoning_token_weight: float,
    wordle_think_token_weight: float = 1.0,
    teacher_reasoning_cache: dict[str, str] | None = None,
    teacher_reason_first: bool,
    teacher_reasoning_url: str,
    teacher_reasoning_temperature: float,
    teacher_reasoning_max_new_tokens: int,
    teacher_reasoning_ignore_eos: bool,
    teacher_reasoning_stop: list[str] | None,
    reload_lora: Callable[[], None] | None = None,
) -> list[OpsdRow]:
    if not hasattr(task, "extract_guess") or not hasattr(task, "compute_feedback"):
        raise RuntimeError("multi-turn asymmetric_opsd requires extract_guess and compute_feedback")
    if not infer_urls:
        raise RuntimeError("asymmetric_opsd requires at least one --infer-url")
    if generation_batch_size <= 0:
        raise RuntimeError(f"generation_batch_size must be positive, got {generation_batch_size}")
    if generation_workers is not None and generation_workers <= 0:
        raise RuntimeError(f"generation_workers must be positive when set, got {generation_workers}")
    if teacher_reason_first and not teacher_reasoning_url:
        raise RuntimeError("teacher_reason_first requires a teacher_reasoning_url")

    max_turns = int(getattr(task, "MAX_TURNS", 6))

    def generate_with_fallback(
        primary_infer_url: str,
        *,
        input_ids: list[int],
    ) -> tuple[str, dict[str, Any], list[str]]:
        ordered_urls = [primary_infer_url] + [url for url in infer_urls if url != primary_infer_url]
        errors: list[str] = []
        for candidate_url in ordered_urls:
            try:
                return (
                    candidate_url,
                    generate_with_sglang(
                        candidate_url,
                        input_ids=input_ids,
                        lora_name=lora_name,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        ignore_eos=ignore_eos,
                        stop=stop,
                        reload_lora=reload_lora,
                    ),
                    errors,
                )
            except Exception as exc:
                message = f"{candidate_url}: {type(exc).__name__}: {exc}"
                errors.append(message)
                print(
                    f"WARN: {message}; trying another SGLang sampler for this Wordle turn",
                    flush=True,
                )
        raise RuntimeError(
            f"all {len(ordered_urls)} SGLang samplers failed for Wordle generation; "
            f"last errors: {errors[-3:]}"
        )

    def generate_teacher_reasoning_context(
        *,
        target: str,
        history: list[tuple[str, str]],
    ) -> tuple[str, str, int, int]:
        if not teacher_reason_first:
            return "", "", 0, 0
        if teacher_reasoning_cache:
            cached = teacher_reasoning_cache.get(teacher_cot_cache_key(target, history))
            if cached:
                return cached, cached, 0, 0
        prompt_ids = _wordle_teacher_reasoning_prompt_ids(
            tokenizer,
            task,
            target=target,
            history=history,
            prompt_style=wordle_prompt_style,
            teacher_prompt_style=wordle_teacher_prompt_style,
        )
        result = generate_base_with_sglang(
            teacher_reasoning_url,
            input_ids=prompt_ids,
            temperature=teacher_reasoning_temperature,
            max_new_tokens=teacher_reasoning_max_new_tokens,
            ignore_eos=teacher_reasoning_ignore_eos,
            stop=teacher_reasoning_stop,
        )
        output_ids = _generated_output_ids(result, tokenizer)
        raw_text = _generated_text_from_sglang_result(result, tokenizer)
        return _normalize_teacher_reasoning_context(raw_text), raw_text, len(prompt_ids), len(output_ids)

    def build_one(item: tuple[int, Example]) -> tuple[list[OpsdRow], list[dict[str, Any]]]:
        idx, example = item
        infer_url = infer_urls[idx % len(infer_urls)]
        target = str(example.metadata["target"]).lower()
        history: list[tuple[str, str]] = []
        rows: list[OpsdRow] = []
        logs: list[dict[str, Any]] = []
        for turn in range(max_turns):
            history_before = list(history)
            student_prefix = _wordle_turn_prompt_ids(
                tokenizer,
                task,
                target=target,
                history=history_before,
                hinted=False,
                prompt_style=wordle_prompt_style,
                teacher_prompt_style=wordle_teacher_prompt_style,
            )
            (
                teacher_reasoning_context,
                teacher_reasoning_raw_text,
                teacher_reasoning_prompt_tokens,
                teacher_reasoning_output_tokens,
            ) = generate_teacher_reasoning_context(target=target, history=history_before)
            teacher_prefix = _wordle_turn_prompt_ids(
                tokenizer,
                task,
                target=target,
                history=history_before,
                hinted=True,
                prompt_style=wordle_prompt_style,
                teacher_prompt_style=wordle_teacher_prompt_style,
                teacher_reasoning_context=teacher_reasoning_context,
            )
            infer_url, result, fallback_errors = generate_with_fallback(infer_url, input_ids=student_prefix)
            raw_output_ids = _generated_output_ids(result, tokenizer)
            if not raw_output_ids:
                raise RuntimeError(f"{example.project} turn={turn + 1}: SGLang returned no output tokens: {result}")
            output_ids, text, truncated_after_guess = _truncate_output_ids_after_first_guess(
                tokenizer, raw_output_ids, think=wordle_prompt_style.endswith("_think")
            )
            if not output_ids:
                raise RuntimeError(f"{example.project} turn={turn + 1}: SGLang returned no usable output tokens: {result}")

            student_ids = student_prefix + output_ids
            teacher_ids_input = teacher_prefix + output_ids
            if len(student_ids) > max_length:
                raise RuntimeError(
                    f"{example.project} turn={turn + 1}: student sequence length {len(student_ids)} exceeds "
                    f"{max_length}"
                )
            if len(teacher_ids_input) > max_length:
                raise RuntimeError(
                    f"{example.project} turn={turn + 1}: teacher sequence length {len(teacher_ids_input)} exceeds "
                    f"{max_length}"
                )

            student_labels = [IGNORE_INDEX] * len(student_ids)
            student_labels[len(student_prefix) :] = output_ids
            teacher_labels = [IGNORE_INDEX] * len(teacher_ids_input)
            teacher_labels[len(teacher_prefix) :] = output_ids
            turn_weight = float(wordle_first_turn_weight) if turn == 0 else 1.0
            raw_text = str(result.get("text") or "")
            if not raw_text:
                raw_text = tokenizer.decode(raw_output_ids, skip_special_tokens=False)
            output_weights, content_token_count, reasoning_token_count = _wordle_output_weights(
                tokenizer,
                output_ids,
                turn_weight=turn_weight,
                tag_token_weight=wordle_tag_token_weight,
                reasoning_token_weight=wordle_reasoning_token_weight,
                think_token_weight=wordle_think_token_weight,
                think=wordle_prompt_style.endswith("_think"),
            )
            masked_output_token_count = 0
            for offset, weight in enumerate(output_weights):
                if weight <= 0.0:
                    student_labels[len(student_prefix) + offset] = IGNORE_INDEX
                    teacher_labels[len(teacher_prefix) + offset] = IGNORE_INDEX
                    masked_output_token_count += 1
            weights = [0.0] * len(student_prefix) + output_weights
            trainable_output_token_count = len(output_weights) - masked_output_token_count
            if hasattr(task, "parse_turn_response"):
                # Judge turn validity (which gates multi-turn game continuation)
                # on the think-stripped, first-guess-truncated action line — the
                # SAME extraction the held-out rollout uses. Parsing the raw text
                # spuriously fails under the think contract (the <think> block
                # mentions the output tags and the model emits trailing chatter),
                # so every game previously broke at turn 1 and only the opener
                # — where the teacher carries no candidate hint — was ever trained.
                action_text = task.extract_action_text(raw_text or text or "")
                parsed_turn = task.parse_turn_response(action_text, history)
                guess = str(parsed_turn.get("guess") or "") or None
                format_ok = bool(parsed_turn.get("format_ok", False))
                valid_guess = bool(parsed_turn.get("action_ok", False))
            else:
                parsed_turn = {}
                guess = task.extract_guess(text or "")
                format_ok = bool(task.has_single_guess_tag(text or "")) if hasattr(task, "has_single_guess_tag") else bool(guess)
                valid_guess = format_ok and _wordle_is_valid_guess(task, guess, history)
            feedback = ""
            solved = False
            row_project = f"{example.project}:turn{turn + 1}"
            if trainable_output_token_count > 0:
                rows.append(
                    OpsdRow(
                        project=row_project,
                        input_ids=student_ids,
                        labels=student_labels,
                        teacher_ids=[0] * len(student_ids),
                        teacher_weights=weights,
                        target_token_count=len(output_ids),
                        target=target,
                        teacher_target_text=text,
                        student_input_ids=student_ids,
                        student_labels=student_labels,
                        teacher_input_ids=teacher_ids_input,
                        teacher_labels=teacher_labels,
                        student_text=text,
                    )
                )

            if valid_guess:
                feedback = task.compute_feedback(guess, target)
                history.append((guess, feedback))
                solved = guess == target
            logs.append(
                {
                    "event": "student_multiturn_generation",
                    "time": time.time(),
                    "project": example.project,
                    "row_project": row_project,
                    "target": target,
                    "infer_url": infer_url,
                    "turn": turn + 1,
                    "history_before": history_before,
                    "guess": guess or "",
                    "format_ok": format_ok,
                    "valid_guess": valid_guess,
                    "public_constraint_valid": bool(parsed_turn.get("public_constraint_valid", True)),
                    "target_leak": bool(parsed_turn.get("target_leak", False)),
                    "extra_text": bool(parsed_turn.get("extra_text", False)),
                    "parse_errors": parsed_turn.get("errors", []),
                    "feedback": feedback,
                    "solved": solved,
                    "token_count": len(output_ids),
                    "student_prompt_tokens": len(student_prefix),
                    "teacher_prompt_tokens": len(teacher_prefix),
                    "teacher_reason_first": teacher_reason_first,
                    "teacher_reasoning_context": teacher_reasoning_context,
                    "teacher_reasoning_raw_text": teacher_reasoning_raw_text,
                    "teacher_reasoning_prompt_tokens": teacher_reasoning_prompt_tokens,
                    "teacher_reasoning_output_tokens": teacher_reasoning_output_tokens,
                    "wordle_prompt_style": wordle_prompt_style,
                    "wordle_teacher_prompt_style": wordle_teacher_prompt_style,
                    "wordle_turn_weight": turn_weight,
                    "wordle_tag_token_weight": wordle_tag_token_weight,
                    "wordle_reasoning_token_weight": wordle_reasoning_token_weight,
                    "wordle_think_token_weight": wordle_think_token_weight,
                    "wordle_guess_content_tokens": content_token_count,
                    "wordle_reasoning_content_tokens": reasoning_token_count,
                    "wordle_masked_output_tokens": masked_output_token_count,
                    "wordle_trainable_output_tokens": trainable_output_token_count,
                    "truncated_after_first_guess": truncated_after_guess,
                    "sampler_fallback_errors": fallback_errors,
                    "text": text,
                    "raw_text": raw_text if truncated_after_guess else "",
                }
            )
            if not valid_guess or solved:
                break
        return rows, logs

    max_workers = max(1, min(len(examples), generation_workers or (len(infer_urls) * generation_batch_size)))
    all_rows: list[OpsdRow] = []
    all_logs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rows, logs in executor.map(build_one, enumerate(examples)):
            all_rows.extend(rows)
            all_logs.extend(logs)

    if not all_rows:
        raise RuntimeError("multi-turn asymmetric_opsd produced no training rows")
    for row in all_logs:
        _jsonl(generation_log_path, row)
    return all_rows


def save_weights_for_sampler(train_url: str, model_id: str, name: str, *, future_timeout: float) -> dict[str, Any]:
    return call_future(
        train_url,
        "/api/v1/save_weights_for_sampler",
        {"model_id": model_id, "name": name},
        context=f"save_weights_for_sampler({model_id}, {name})",
        future_timeout=future_timeout,
    )


def submit_save_weights_for_sampler(train_url: str, model_id: str, name: str) -> str:
    future = _post_json(
        f"{train_url}/api/v1/save_weights_for_sampler",
        {"model_id": model_id, "name": name},
        timeout=120.0,
    )
    request_id = future.get("request_id")
    if not request_id:
        raise RuntimeError(f"save_weights_for_sampler({model_id}, {name}) did not return request_id: {future}")
    return str(request_id)


def wait_for_sampler_export(lora_path: Path, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (
            lora_path.is_dir()
            and (lora_path / "adapter_model.safetensors").is_file()
            and (lora_path / "adapter_config.json").is_file()
        ):
            return
        time.sleep(1.0)
    raise TimeoutError(f"Sampler export did not appear at {lora_path} within {timeout:.1f}s")


def _sampler_uri_to_path(model_path: str, server_output_dir: Path) -> Path:
    parsed = urlparse(model_path)
    if parsed.scheme == "xorl":
        pieces = parsed.path.lstrip("/").split("/")
        if len(pieces) >= 2 and pieces[0] == "sampler_weights":
            return server_output_dir / "sampler_weights" / "/".join(pieces[1:])
        raise RuntimeError(f"Unsupported sampler xorl URI: {model_path}")
    if model_path.startswith("sampler_weights/"):
        return server_output_dir / model_path
    return Path(model_path)


def _post_sglang(
    infer_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float = 300.0,
    attempts: int = 1,
    retry_interval: float = 5.0,
) -> dict[str, Any] | list:
    url = f"{infer_url.rstrip('/')}{path}"
    attempts = max(1, int(attempts))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Connection": "close"},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= int(status) < 500 and int(status) not in {408, 409, 425, 429}:
                response_text = getattr(getattr(exc, "response", None), "text", "") or ""
                detail = response_text[:2000]
                raise RuntimeError(f"SGLang POST {url} returned HTTP {status}: {detail}") from exc
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = min(float(retry_interval) * attempt, 30.0)
            print(
                f"WARN: SGLang POST {url} failed attempt {attempt}/{attempts}: {exc}; "
                f"retrying in {sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _is_sglang_missing_lora_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "got lora adapter that has never been loaded" in text
        or "all loaded adapters: dict_keys([])" in text
        or "unknown zorl session" in text
        or "unknown lora" in text
        or "adapter" in text and "never been loaded" in text
    )


def unload_sglang_lora(infer_url: str, lora_name: str) -> None:
    try:
        _post_sglang(
            infer_url,
            "/unload_lora_adapter",
            {"lora_name": lora_name},
            timeout=60.0,
            attempts=_env_int("OPSD_SGLANG_UNLOAD_RETRY_ATTEMPTS", 5),
            retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
        )
    except Exception:
        pass


def load_sglang_lora(infer_url: str, *, lora_name: str, lora_path: str) -> None:
    result = _post_sglang(
        infer_url,
        "/load_lora_adapter",
        {"lora_name": lora_name, "lora_path": lora_path, "pinned": False},
        timeout=_env_float("OPSD_SGLANG_LOAD_TIMEOUT", 300.0),
        attempts=_env_int("OPSD_SGLANG_LOAD_RETRY_ATTEMPTS", 60),
        retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
    )
    if isinstance(result, dict) and result.get("success", True) is False:
        error = result.get("error_message") or result
        if "already loaded" not in str(error).lower():
            raise RuntimeError(f"load_lora_adapter({lora_name}) failed on {infer_url}: {error}")


def generate_with_sglang(
    infer_url: str,
    *,
    input_ids: list[int],
    lora_name: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    reload_lora: Callable[[], None] | None = None,
) -> dict[str, Any]:
    return generate_batch_with_sglang(
        infer_url,
        input_ids_batch=[input_ids],
        lora_name=lora_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
        stop=stop,
        reload_lora=reload_lora,
    )[0]


def generate_batch_with_sglang(
    infer_url: str,
    *,
    input_ids_batch: list[list[int]],
    lora_name: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    reload_lora: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    sampling: dict[str, Any] = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    payload = {
        "input_ids": [list(input_ids) for input_ids in input_ids_batch],
        "sampling_params": sampling,
        "return_logprob": False,
    }
    if lora_name:
        # SGLang accepts a scalar adapter name for batched generation, and SMG
        # requires this form when routing /generate requests. Full-weight mode
        # omits it entirely: the synced base weights ARE the policy.
        payload["lora_path"] = lora_name
    try:
        result = _post_sglang(
            infer_url,
            "/generate",
            payload,
            timeout=900.0,
            attempts=_env_int("OPSD_SGLANG_GENERATE_RETRY_ATTEMPTS", 12),
            retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
        )
    except Exception as exc:
        if reload_lora is None or not _is_sglang_missing_lora_error(exc):
            raise
        print(
            f"WARN: SGLang sampler {infer_url} lost LoRA {lora_name}; reloading current policy and retrying",
            flush=True,
        )
        reload_lora()
        result = _post_sglang(
            infer_url,
            "/generate",
            payload,
            timeout=900.0,
            attempts=_env_int("OPSD_SGLANG_GENERATE_RETRY_ATTEMPTS", 12),
            retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
        )
    if isinstance(result, list):
        if len(result) != len(input_ids_batch):
            raise RuntimeError(f"SGLang returned {len(result)} outputs for {len(input_ids_batch)} prompts")
        return [dict(item or {}) for item in result]
    if len(input_ids_batch) != 1:
        raise RuntimeError(f"SGLang returned a single output for {len(input_ids_batch)} prompts: {result}")
    return [dict(result or {})]


def generate_base_with_sglang(
    infer_url: str,
    *,
    input_ids: list[int],
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
) -> dict[str, Any]:
    sampling: dict[str, Any] = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    payload = {
        "input_ids": [list(input_ids)],
        "sampling_params": sampling,
        "return_logprob": False,
    }
    result = _post_sglang(
        infer_url,
        "/generate",
        payload,
        timeout=900.0,
        attempts=_env_int("OPSD_SGLANG_GENERATE_RETRY_ATTEMPTS", 12),
        retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
    )
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"SGLang returned {len(result)} outputs for one teacher prompt")
        return dict(result[0] or {})
    return dict(result or {})


def _generated_output_ids(result: dict[str, Any], tokenizer) -> list[int]:
    output_ids = result.get("output_ids")
    if isinstance(output_ids, list) and output_ids:
        return [int(x) for x in output_ids]
    text = str(result.get("text") or "")
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _generated_text_from_sglang_result(result: dict[str, Any], tokenizer) -> str:
    text = str(result.get("text") or "")
    if text:
        return text
    output_ids = _generated_output_ids(result, tokenizer)
    return tokenizer.decode(output_ids, skip_special_tokens=True) if output_ids else ""


def register_inference_endpoints(
    train_url: str,
    sampler_urls: list[str],
    *,
    world_size: int,
    timeout: float = 300.0,
    attempts: int = 30,
    retry_interval: float = 20.0,
) -> None:
    """Register sampler shards on the training server for full-weight sync.

    Retries health-check failures: samplers may still be loading weights when
    the trainer reaches this point (SMG readiness does not imply per-shard
    readiness right after a pod cycle).
    """
    for url in sampler_urls:
        parsed = urlparse(url if "//" in url else f"http://{url}")
        payload = {
            "host": parsed.hostname,
            "port": int(parsed.port or 30000),
            "world_size": int(world_size),
        }
        last_message = ""
        for attempt in range(1, attempts + 1):
            resp = requests.post(f"{train_url.rstrip('/')}/add_inference_endpoint", json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if body.get("success", False):
                break
            last_message = str(body.get("message", body))
            if "already" in last_message.lower():
                break
            if attempt >= attempts:
                raise RuntimeError(
                    f"add_inference_endpoint({parsed.hostname}:{parsed.port}) failed after "
                    f"{attempts} attempts: {last_message}"
                )
            print(
                f"WARN: add_inference_endpoint({parsed.hostname}:{parsed.port}) not ready "
                f"(attempt {attempt}/{attempts}): {last_message}; retrying in {retry_interval:.0f}s",
                flush=True,
            )
            time.sleep(retry_interval)
        print(f"[weight_sync] registered inference endpoint {parsed.hostname}:{parsed.port} world_size={world_size}")


def _pod_ip() -> str:
    """Routable address of this pod. Bare-pod hostnames are not resolvable by
    other pods, so the NCCL rendezvous master address must be the pod IP."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]


def sync_weights_to_samplers(
    *,
    train_url: str,
    model_id: str,
    weight_version: str,
    output_dir: Path,
    master_address: str = "",
    buffer_size_mb: int = 1024,
    flush_cache: bool = True,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Full-weight broadcast of the current policy to all registered samplers."""
    started = time.time()
    payload = {
        "model_id": model_id,
        "weight_version": weight_version,
        # The server's auto-detect uses the pod hostname, which samplers
        # cannot resolve; always pass an IP the samplers can reach.
        "master_address": master_address or _pod_ip(),
        "buffer_size_mb": int(buffer_size_mb),
        "pause_mode": "retract",
        "flush_cache": bool(flush_cache),
    }
    resp = requests.post(
        f"{train_url.rstrip('/')}/api/v1/sync_inference_weights", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success", True):
        raise RuntimeError(f"sync_inference_weights({weight_version}) failed: {body}")
    sync_time = time.time() - started
    _jsonl(
        output_dir / "sampler_exports.jsonl",
        {
            "time": time.time(),
            "event": "full_weight_sync",
            "model_id": model_id,
            "weight_version": weight_version,
            "sync_time_s": sync_time,
            "response": {k: v for k, v in body.items() if k in ("success", "message", "results")},
        },
    )
    return body


def export_and_load_sampler(
    *,
    train_url: str,
    model_id: str,
    output_dir: Path,
    server_output_dir: Path,
    infer_urls: list[str],
    lora_name: str,
    save_name: str,
    future_timeout: float,
    gdn_repack: bool = False,
) -> str:
    if not infer_urls:
        raise RuntimeError("export_and_load_sampler requires at least one infer URL")
    save_request_id = submit_save_weights_for_sampler(train_url, model_id, save_name)
    lora_path = server_output_dir / "sampler_weights" / save_name
    wait_for_sampler_export(lora_path, timeout=future_timeout)
    if gdn_repack:
        # SGLang serves GDN LoRA only in the fused in_proj_qkvz/out_proj layout
        # (see repack_gdn_lora.py); repack the per-step export before loading.
        fused_path = lora_path.parent / (lora_path.name + "-fused")
        repack_started = time.time()
        # Cap torch CPU threads and deprioritize: the repack shares the head
        # node with the engine ranks, and an uncapped OMP pool can thrash from
        # ~20s to ~40min under contention.
        repack_env = dict(os.environ)
        repack_env["OMP_NUM_THREADS"] = "16"
        repack_env["MKL_NUM_THREADS"] = "16"
        subprocess.run(
            [
                "nice", "-n", "10",
                sys.executable,
                str(Path(__file__).resolve().parent / "repack_gdn_lora.py"),
                "--input", str(lora_path),
                "--output", str(fused_path),
            ],
            check=True,
            env=repack_env,
        )
        print(f"[gdn-repack] {lora_path.name} -> {fused_path.name} in {time.time() - repack_started:.1f}s", flush=True)
        lora_path = fused_path
    load_started = time.time()

    def load_one(infer_url: str) -> dict[str, Any]:
        endpoint_started = time.time()
        unload_sglang_lora(infer_url, lora_name)
        load_sglang_lora(infer_url, lora_name=lora_name, lora_path=str(lora_path))
        return {"infer_url": infer_url, "load_time_s": time.time() - endpoint_started}

    max_workers = max(1, min(len(infer_urls), _env_int("OPSD_SGLANG_LOAD_WORKERS", len(infer_urls))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        load_results = list(executor.map(load_one, infer_urls))
    if gdn_repack:
        # Fused repacks are ~3x the raw export (uniform 3r rank padding); once the
        # samplers hold this step's weights in memory, earlier fused dirs are dead.
        for stale in sorted(lora_path.parent.glob("*-fused")):
            if stale != lora_path and stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
    _jsonl(
        output_dir / "sampler_exports.jsonl",
        {
            "time": time.time(),
            "model_id": model_id,
            "save_name": save_name,
            "save_request_id": save_request_id,
            "lora_name": lora_name,
            "path": str(lora_path),
            "infer_urls": list(infer_urls),
            "load_results": load_results,
            "load_time_s": time.time() - load_started,
        },
    )
    return str(lora_path)


def build_asymmetric_opsd_rows(
    *,
    tokenizer,
    task,
    examples: list[Example],
    infer_urls: list[str],
    lora_name: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    generation_batch_size: int,
    generation_workers: int | None,
    max_length: int,
    generation_log_path: Path,
    wordle_prompt_style: str,
    wordle_teacher_prompt_style: str,
    wordle_first_turn_weight: float,
    wordle_tag_token_weight: float,
    wordle_reasoning_token_weight: float,
    wordle_think_token_weight: float = 1.0,
    teacher_reasoning_cache: dict[str, str] | None = None,
    teacher_reason_first: bool = False,
    teacher_reasoning_url: str = "",
    teacher_reasoning_temperature: float = 0.2,
    teacher_reasoning_max_new_tokens: int = 96,
    teacher_reasoning_ignore_eos: bool = True,
    teacher_reasoning_stop: list[str] | None = None,
    reload_lora: Callable[[], None] | None = None,
) -> list[OpsdRow]:
    if not infer_urls:
        raise RuntimeError("asymmetric_opsd requires at least one --infer-url")
    if generation_batch_size <= 0:
        raise RuntimeError(f"generation_batch_size must be positive, got {generation_batch_size}")
    if bool(getattr(task, "is_multi_turn", False)):
        return build_multiturn_asymmetric_opsd_rows(
            tokenizer=tokenizer,
            task=task,
            examples=examples,
            infer_urls=infer_urls,
            lora_name=lora_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
            stop=stop,
            generation_batch_size=generation_batch_size,
            generation_workers=generation_workers,
            max_length=max_length,
            generation_log_path=generation_log_path,
            wordle_prompt_style=wordle_prompt_style,
            wordle_teacher_prompt_style=wordle_teacher_prompt_style,
            wordle_first_turn_weight=wordle_first_turn_weight,
            wordle_tag_token_weight=wordle_tag_token_weight,
            wordle_reasoning_token_weight=wordle_reasoning_token_weight,
            wordle_think_token_weight=wordle_think_token_weight,
            teacher_reasoning_cache=teacher_reasoning_cache,
            teacher_reason_first=teacher_reason_first,
            teacher_reasoning_url=teacher_reasoning_url,
            teacher_reasoning_temperature=teacher_reasoning_temperature,
            teacher_reasoning_max_new_tokens=teacher_reasoning_max_new_tokens,
            teacher_reasoning_ignore_eos=teacher_reasoning_ignore_eos,
            teacher_reasoning_stop=teacher_reasoning_stop,
            reload_lora=reload_lora,
        )

    student_prefixes: list[list[int]] = []
    teacher_prefixes: list[list[int]] = []
    outputs_by_index: dict[int, dict[str, Any]] = {}
    indices_by_url: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        infer_url = infer_urls[idx % len(infer_urls)]
        indices_by_url.setdefault(infer_url, []).append(idx)
        student_prefixes.append(_asymmetric_prompt_ids(tokenizer, task, example, hinted=False))
        teacher_prefixes.append(
            _asymmetric_prompt_ids(
                tokenizer,
                task,
                example,
                hinted=True,
                teacher_prompt_style=wordle_teacher_prompt_style,
            )
        )

    def generate_for_url(infer_url: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        local_outputs: dict[int, dict[str, Any]] = {}
        for batch_indices in chunks(indices, generation_batch_size):
            batch_outputs = generate_batch_with_sglang(
                infer_url,
                input_ids_batch=[student_prefixes[idx] for idx in batch_indices],
                lora_name=lora_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
                stop=stop,
            )
            for idx, result in zip(batch_indices, batch_outputs, strict=True):
                local_outputs[idx] = result
        return local_outputs

    if len(indices_by_url) == 1:
        infer_url, indices = next(iter(indices_by_url.items()))
        outputs_by_index.update(generate_for_url(infer_url, indices))
    else:
        with ThreadPoolExecutor(max_workers=len(indices_by_url)) as executor:
            futures = [
                executor.submit(generate_for_url, infer_url, indices)
                for infer_url, indices in indices_by_url.items()
            ]
            for future in futures:
                outputs_by_index.update(future.result())

    rows: list[OpsdRow] = []
    for idx, example in enumerate(examples):
        infer_url = infer_urls[idx % len(infer_urls)]
        student_prefix = student_prefixes[idx]
        teacher_prefix = teacher_prefixes[idx]
        result = outputs_by_index[idx]
        output_ids = _generated_output_ids(result, tokenizer)
        if not output_ids:
            raise RuntimeError(f"{example.project}: SGLang returned no student output tokens: {result}")
        student_ids = student_prefix + output_ids
        teacher_ids_input = teacher_prefix + output_ids
        if len(student_ids) > max_length:
            raise RuntimeError(f"{example.project}: student sequence length {len(student_ids)} exceeds {max_length}")
        if len(teacher_ids_input) > max_length:
            raise RuntimeError(
                f"{example.project}: teacher sequence length {len(teacher_ids_input)} exceeds {max_length}"
            )

        student_labels = [IGNORE_INDEX] * len(student_ids)
        student_labels[len(student_prefix) :] = output_ids
        teacher_labels = [IGNORE_INDEX] * len(teacher_ids_input)
        teacher_labels[len(teacher_prefix) :] = output_ids
        weights = [1.0 if label != IGNORE_INDEX else 0.0 for label in student_labels]
        text = str(result.get("text") or tokenizer.decode(output_ids, skip_special_tokens=False))
        row = OpsdRow(
            project=example.project,
            input_ids=student_ids,
            labels=student_labels,
            teacher_ids=[0] * len(student_ids),
            teacher_weights=weights,
            target_token_count=len(output_ids),
            target=str(example.metadata.get("target", "")),
            teacher_target_text=text,
            student_input_ids=student_ids,
            student_labels=student_labels,
            teacher_input_ids=teacher_ids_input,
            teacher_labels=teacher_labels,
            student_text=text,
        )
        rows.append(row)
        _jsonl(
            generation_log_path,
            {
                "event": "student_generation",
                "time": time.time(),
                "project": row.project,
                "target": row.target,
                "infer_url": infer_url,
                "token_count": len(output_ids),
                "text": text,
            },
        )
    return rows


def _row_ids_and_labels(row: OpsdRow, *, objective: str, cache_view: bool) -> tuple[list[int], list[int]]:
    if objective == "asymmetric_opsd":
        if cache_view:
            if row.teacher_input_ids is None or row.teacher_labels is None:
                raise RuntimeError(f"{row.project}: asymmetric teacher view is missing")
            return row.teacher_input_ids, row.teacher_labels
        if row.student_input_ids is None or row.student_labels is None:
            raise RuntimeError(f"{row.project}: asymmetric student view is missing")
        return row.student_input_ids, row.student_labels
    return row.input_ids, row.labels


def build_gold_sft_rows(
    tokenizer,
    gold_path: str,
    *,
    max_length: int,
    eval_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[list[OpsdRow], list[OpsdRow]]:
    """Rows from a gold trajectory JSONL (messages + completion per turn)."""
    rows: list[OpsdRow] = []
    eos_id = tokenizer.eos_token_id
    with open(gold_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completion_text = str(rec.get("completion", ""))
            prompt_ids = list(
                tokenizer.apply_chat_template(
                    rec["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    # Think-contract gold completions begin inside an open think
                    # block (raw think text ... </think> ... public line); the
                    # prompt rendering must open one or the rows teach mismatched
                    # tags in a context never seen at inference.
                    enable_thinking="</think>" in completion_text,
                    return_dict=False,
                )
            )
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            if eos_id is not None:
                completion_ids = completion_ids + [eos_id]
            input_ids = prompt_ids + completion_ids
            if max_length and len(input_ids) > max_length:
                continue
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
            rows.append(
                OpsdRow(
                    project=f"gold_{i:06d}",
                    input_ids=input_ids,
                    labels=labels,
                    teacher_ids=[],
                    teacher_weights=[],
                    target_token_count=len(completion_ids),
                    target=str(rec.get("target", "")),
                    teacher_target_text=str(rec.get("completion", "")),
                )
            )
    if not rows:
        raise RuntimeError(f"No usable gold rows in {gold_path}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_fraction)) if eval_fraction > 0 else 0
    return rows[n_eval:], rows[:n_eval]


def row_to_datum(
    row: OpsdRow,
    *,
    include_cache: bool,
    objective: str = "opd_self_kl",
    cache_view: bool = False,
) -> dict[str, Any]:
    row_input_ids, row_labels = _row_ids_and_labels(row, objective=objective, cache_view=cache_view)
    if len(row_input_ids) < 2:
        raise RuntimeError(f"{row.project}: sequence must have at least two tokens")

    if objective in {"teacher_forced_ce", "sft_gold"}:
        if include_cache:
            raise RuntimeError(f"{objective} does not use teacher hidden caches")
        return {
            "model_input": {"input_ids": row.input_ids},
            "loss_fn_inputs": {"labels": row.labels},
        }

    if objective not in {"opd_self_kl", "asymmetric_opsd"}:
        raise RuntimeError(f"Unknown objective: {objective}")

    # The server treats target_tokens as already causal-shifted. Keep OPD's
    # per-token teacher fields in that same prediction-position frame.
    input_ids = row_input_ids[:-1]
    target_tokens = row_labels[1:]
    loss_inputs: dict[str, Any] = {
        "target_tokens": target_tokens,
    }
    if not cache_view:
        loss_inputs["teacher_ids"] = row.teacher_ids[1:]
        loss_inputs["teacher_weights"] = row.teacher_weights[1:]
    if include_cache:
        if row.teacher_cache_indices is None:
            raise RuntimeError(f"{row.project}: teacher_cache_indices have not been materialized")
        loss_inputs["teacher_cache_indices"] = row.teacher_cache_indices[1:]
    return {
        "model_input": {"input_ids": input_ids},
        "loss_fn_inputs": loss_inputs,
    }


def apply_cache_indices(rows: list[OpsdRow], cache_indices_by_sample: list[list[int]]) -> None:
    if len(cache_indices_by_sample) != len(rows):
        raise RuntimeError(
            f"teacher_hidden_cache returned {len(cache_indices_by_sample)} samples for {len(rows)} input rows"
        )
    for row, sample_indices in zip(rows, cache_indices_by_sample, strict=True):
        student_labels = row.student_labels if row.student_labels is not None else row.labels
        valid_positions = [i for i, label in enumerate(student_labels) if label != IGNORE_INDEX]
        if len(sample_indices) != len(valid_positions):
            raise RuntimeError(
                f"{row.project}: cache index count {len(sample_indices)} does not match valid token count "
                f"{len(valid_positions)}"
            )
        indices = [0] * len(student_labels)
        for pos, cache_idx in zip(valid_positions, sample_indices, strict=True):
            indices[pos] = int(cache_idx)
        row.teacher_cache_indices = indices


def materialize_teacher_cache(
    *,
    backend: str,
    train_url: str,
    teacher_url: str,
    model_id: str,
    rows: list[OpsdRow],
    cache_path: Path,
    cache_dtype: str,
    objective: str,
    future_timeout: float,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [row_to_datum(row, include_cache=False, objective=objective, cache_view=True) for row in rows]
    if backend == "xorl":
        result = call_future(
            train_url,
            "/api/v1/forward",
            {
                "model_id": model_id,
                "forward_input": {
                    "data": data,
                    "loss_fn": "teacher_hidden_cache",
                    "loss_fn_params": {
                        "teacher_hidden_cache_path": str(cache_path),
                        "teacher_hidden_cache_dtype": cache_dtype,
                    },
                },
            },
            context=f"teacher_hidden_cache({cache_path.name})",
            future_timeout=future_timeout,
        )
        info = result.get("info") or {}
        metadata = info.get("teacher_hidden_cache")
    elif backend == "sglang":
        if not teacher_url:
            raise RuntimeError("--teacher-url is required when --teacher-cache-backend=sglang")
        input_ids = [datum["model_input"]["input_ids"] for datum in data]
        target_tokens = [datum["loss_fn_inputs"]["target_tokens"] for datum in data]
        result = _post_sglang(
            teacher_url,
            "/teacher_hidden_cache",
            {
                "input_ids": input_ids,
                "target_tokens": target_tokens,
                "cache_path": str(cache_path),
                "cache_key": "hidden_states",
                "dtype": cache_dtype,
            },
            timeout=future_timeout,
            attempts=_env_int("OPSD_TEACHER_CACHE_RETRY_ATTEMPTS", 6),
            retry_interval=_env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0),
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"SGLang teacher_hidden_cache returned unexpected payload: {result}")
        if result.get("error"):
            raise RuntimeError(f"SGLang teacher_hidden_cache error: {result['error']}")
        metadata = result
        cache_indices = metadata.get("cache_indices_by_sample") or []
        if len(cache_indices) != len(target_tokens):
            raise RuntimeError(
                f"SGLang teacher returned {len(cache_indices)} cache-index lists for {len(target_tokens)} rows"
            )
        for idx, (target, indices) in enumerate(zip(target_tokens, cache_indices, strict=True)):
            expected = sum(1 for token in target if token != IGNORE_INDEX)
            if len(indices) != expected:
                raise RuntimeError(
                    f"SGLang teacher row {idx} returned {len(indices)} indices; expected {expected} "
                    "positions where target_tokens != IGNORE_INDEX"
                )
    else:
        raise RuntimeError(f"Unknown teacher cache backend: {backend}")

    if not metadata:
        raise RuntimeError(f"teacher_hidden_cache response did not include cache metadata: {result}")
    if metadata.get("path") != str(cache_path):
        raise RuntimeError(f"teacher_hidden_cache wrote unexpected path: {metadata.get('path')} != {cache_path}")
    apply_cache_indices(rows, metadata.get("cache_indices_by_sample") or [])
    return metadata


def create_model(train_url: str, args) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": args.train_model_id,
        "base_model": args.model,
        "lora_config": None
        if getattr(args, "full_weight", False)
        else {
            "rank": args.lora_rank,
            "lora_rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "lora_alpha": args.lora_alpha,
        },
        # Full-weight server mode rejects per-session optimizer overrides; the
        # optimizer comes from the server config yaml in that mode.
        "optimizer_config": None
        if getattr(args, "full_weight", False)
        else {
            "type": args.optimizer,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer_dtype": args.optimizer_dtype,
            "betas": [args.beta1, args.beta2],
            "eps": args.eps,
            **(
                {"optimizer_kwargs": json.loads(args.optimizer_kwargs)}
                if getattr(args, "optimizer_kwargs", "")
                else {}
            ),
        },
        # Full-weight server mode rejects ANY non-None per-session override, incl. zorl_config
        # (endpoints.py: `req.zorl_config is not None`). Gate it like lora_config/optimizer_config.
        "zorl_config": None if getattr(args, "full_weight", False) else {"enabled": False},
    }
    return call_future(
        train_url,
        "/api/v1/create_model",
        payload,
        context=f"create_model({args.train_model_id})",
        future_timeout=args.future_timeout,
    )


def save_weights(train_url: str, model_id: str, name: str, *, future_timeout: float) -> dict[str, Any]:
    return call_future(
        train_url,
        "/api/v1/save_weights",
        {"model_id": model_id, "path": name},
        context=f"save_weights({model_id}, {name})",
        future_timeout=future_timeout,
    )


def opd_loss_params(args, *, teacher_head_dir: str, cache_path: Path) -> dict[str, Any]:
    return {
        "teacher_heads": {"0": teacher_head_dir},
        "teacher_hidden_caches": {"0": str(cache_path)},
        "opd_kl_backend": args.opd_kl_backend,
        "opd_vocab_chunk_size": args.opd_vocab_chunk_size,
        "opd_loss_mode": args.opd_loss_mode,
        "opd_emit_full_vocab_diagnostics": args.emit_full_vocab_diagnostics,
        "opd_profile_timings": args.opd_profile_timings,
        "opd_profile_sync_cuda": args.opd_profile_sync_cuda,
        "return_per_token": False,
    }


def ce_loss_params(args) -> dict[str, Any]:
    return {"return_per_token": False}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_base_name(key: str) -> tuple[str, str]:
    if ":" in key:
        base, suffix = key.rsplit(":", 1)
        return base, suffix
    return key, "mean"


def _valid_tokens(metrics: dict[str, Any]) -> float:
    for key in ("valid_tokens:sum", "valid_tokens", "global_valid_tokens:sum", "global_valid_tokens"):
        value = metrics.get(key)
        if _is_number(value):
            return max(float(value), 0.0)
    return 1.0


def summarize_responses(responses: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    passthrough_sums: dict[str, float] = {}
    loss_sum = 0.0
    loss_sum_tokens = 0.0

    for response in responses:
        metrics = response.get("metrics") or {}
        weight = _valid_tokens(metrics)
        for raw_key, value in metrics.items():
            if not _is_number(value):
                continue
            key, reduction = _metric_base_name(raw_key)
            if key == "loss" and reduction == "sum":
                loss_sum += float(value)
                loss_sum_tokens += weight
            elif key.startswith("opd_profile_") and key.endswith(("_s", "_ms")):
                passthrough_sums[key] = passthrough_sums.get(key, 0.0) + float(value)
            elif reduction == "sum" or key in {"valid_tokens", "global_valid_tokens"}:
                passthrough_sums[key] = passthrough_sums.get(key, 0.0) + float(value)
            elif key == "loss":
                sums[key] = sums.get(key, 0.0) + float(value) * weight
                counts[key] = counts.get(key, 0.0) + weight
            else:
                sums[key] = sums.get(key, 0.0) + float(value) * weight
                counts[key] = counts.get(key, 0.0) + weight

    out = {key: total / max(counts[key], 1.0) for key, total in sums.items()}
    if "loss" not in out and loss_sum_tokens > 0:
        out["loss"] = loss_sum / loss_sum_tokens
    out.update(passthrough_sums)
    out["chunks"] = float(len(responses))
    return out


def format_metric_summary(
    metrics: dict[str, float],
    *,
    include_grad: bool = False,
    step_time_s: float | None = None,
) -> str:
    parts = [f"loss={metrics.get('loss', float('nan')):.6f}"]
    if "ce_loss" in metrics:
        parts.append(f"ce_loss={metrics['ce_loss']:.6f}")
    if "opd_kl" in metrics:
        parts.append(f"opd_kl={metrics['opd_kl']:.6f}")
    parts.append(f"tokens={metrics.get('valid_tokens', 0.0):.0f}")
    if include_grad:
        parts.append(f"grad_norm={metrics.get('grad_norm', float('nan')):.4f}")
    if step_time_s is not None:
        parts.append(f"dt={step_time_s:.1f}s")
    return " ".join(parts)


def format_sample_eval_summary(metrics: dict[str, float]) -> str:
    exact_count = metrics.get("exact_count", 0.0)
    n = max(metrics.get("n", 0.0), 1.0)
    parts = [
        f"exact={exact_count:.0f}/{n:.0f}",
        f"exact_match_rate={metrics.get('exact_match_rate', 0.0):.4f}",
    ]
    for key in ("reward_mean", "format_rate_mean", "info_gain_mean", "turns_used_mean", "error_rate"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    if "sample_eval_time_s" in metrics:
        parts.append(f"dt={metrics['sample_eval_time_s']:.1f}s")
    return " ".join(parts)


def format_teacher_sample_summary(metrics: dict[str, float]) -> str:
    exact_count = metrics.get("exact_count", 0.0)
    n = max(metrics.get("n", 0.0), 1.0)
    parts = [
        f"exact={exact_count:.0f}/{n:.0f}",
        f"exact_match_rate={metrics.get('exact_match_rate', 0.0):.4f}",
    ]
    for key in ("reward_mean", "format_rate_mean", "info_gain_mean", "turns_used_mean", "error_rate"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    if "teacher_sample_time_s" in metrics:
        parts.append(f"dt={metrics['teacher_sample_time_s']:.1f}s")
    return " ".join(parts)


def chunks(rows: list[OpsdRow], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def forward_backward_requests(
    *,
    model_id: str,
    rows: list[OpsdRow],
    loss_fn: str,
    loss_params: dict[str, Any],
    objective: str,
    request_batch_size: int,
    seq_id_start: int = 0,
) -> list[dict[str, Any]]:
    include_cache = objective in {"opd_self_kl", "asymmetric_opsd"}
    requests_out: list[dict[str, Any]] = []
    for local_chunk_idx, batch_rows in enumerate(chunks(rows, request_batch_size)):
        seq_id = int(seq_id_start) + local_chunk_idx
        requests_out.append(
            {
                "model_id": model_id,
                "seq_id": seq_id,
                "forward_backward_input": {
                    "data": [row_to_datum(row, include_cache=include_cache, objective=objective) for row in batch_rows],
                    "loss_fn": loss_fn,
                    "loss_fn_params": loss_params,
                },
            }
        )
    return requests_out


def _loss_params_with_replay_cache(loss_params: dict[str, Any], replay_cache_path: Path) -> dict[str, Any]:
    copied = json.loads(json.dumps(loss_params))
    for key in ("teacher_hidden_caches", "opd_teacher_hidden_caches"):
        value = copied.get(key)
        if isinstance(value, dict):
            copied[key] = {str(teacher_id): str(replay_cache_path) for teacher_id in value}
    for key in ("teacher_hidden_path", "opd_teacher_hidden_path"):
        if key in copied:
            copied[key] = str(replay_cache_path)
    return copied


def dump_forward_backward_replay(
    *,
    replay_path: Path,
    model_id: str,
    rows: list[OpsdRow],
    loss_fn: str,
    loss_params: dict[str, Any],
    objective: str,
    request_batch_size: int,
    seq_id_start: int,
    policy_step: int,
    chunk_idx: int,
    cache_path: Path | None,
) -> int:
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_loss_params = json.loads(json.dumps(loss_params))
    replay_cache_path: Path | None = None
    if cache_path is not None:
        if not cache_path.exists():
            raise RuntimeError(f"Cannot dump f/b replay; missing teacher cache: {cache_path}")
        cache_dir = replay_path.with_name(f"{replay_path.stem}_caches")
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = cache_path.suffix or ".safetensors"
        replay_cache_path = cache_dir / f"policy{policy_step:06d}_chunk{chunk_idx:03d}{suffix}"
        shutil.copy2(cache_path, replay_cache_path)
        replay_loss_params = _loss_params_with_replay_cache(replay_loss_params, replay_cache_path)

    requests_out = forward_backward_requests(
        model_id=model_id,
        rows=rows,
        loss_fn=loss_fn,
        loss_params=replay_loss_params,
        objective=objective,
        request_batch_size=request_batch_size,
        seq_id_start=seq_id_start,
    )
    row_tokens = sum(row.target_token_count for row in rows)
    for request_idx, request in enumerate(requests_out):
        data = request["forward_backward_input"]["data"]
        _jsonl(
            replay_path,
            {
                "event": "forward_backward_request",
                "schema_version": 1,
                "policy_step": int(policy_step),
                "chunk_idx": int(chunk_idx),
                "request_idx": int(request_idx),
                "rows_in_prepared_chunk": len(rows),
                "requests_in_prepared_chunk": len(requests_out),
                "datums": len(data),
                "target_tokens_in_prepared_chunk": row_tokens,
                "teacher_cache_path": str(replay_cache_path) if replay_cache_path is not None else None,
                "request": request,
            },
        )
    return len(requests_out)


def sample_eval_student(
    *,
    tokenizer,
    task,
    examples: list[Example],
    infer_urls: list[str],
    lora_name: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    workers: int,
    step: int,
    log_path: Path,
    log_examples: int,
    wordle_prompt_style: str,
    reload_lora: Callable[[], None] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not examples:
        return {"n": 0.0, "exact_count": 0.0, "exact_match_rate": 0.0}, []
    if not infer_urls:
        raise RuntimeError("sample eval requires at least one infer URL")
    if workers <= 0:
        raise RuntimeError(f"sample eval workers must be positive, got {workers}")

    multi_turn = bool(getattr(task, "is_multi_turn", False))
    rollout_args = SimpleNamespace(
        rollout_temperature=float(temperature),
        rollout_max_new_tokens=int(max_new_tokens),
        wordle_prompt_style=wordle_prompt_style,
    )

    def score_one(item: tuple[int, Example]) -> dict[str, Any]:
        idx, example = item
        infer_url = infer_urls[idx % len(infer_urls)]
        turn_texts: list[str] = []
        try:
            if multi_turn:
                if not hasattr(task, "rollout_completion"):
                    raise RuntimeError(
                        f"Task {getattr(task, '__name__', task)!r} is marked multi-turn without rollout_completion"
                    )

                def generate_turn(input_ids, *, lora_path, temperature, max_new_tokens, stop=None):
                    if stop is None:
                        active_stop: list[str] = []
                    elif isinstance(stop, str):
                        active_stop = _split_urls([stop])
                    else:
                        active_stop = _split_urls(list(stop))
                    if not active_stop:
                        active_stop = list(stop_tokens)
                    result = generate_with_sglang(
                        infer_url,
                        input_ids=list(input_ids),
                        lora_name=str(lora_path),
                        temperature=float(temperature),
                        max_new_tokens=int(max_new_tokens),
                        ignore_eos=bool(ignore_eos),
                        stop=active_stop,
                        reload_lora=reload_lora,
                    )
                    text = str(result.get("text") or "")
                    if not text:
                        output_ids = _generated_output_ids(result, tokenizer)
                        text = tokenizer.decode(output_ids, skip_special_tokens=True) if output_ids else ""
                    turn_texts.append(text)
                    return text

                score = task.rollout_completion(
                    example,
                    generate_turn=generate_turn,
                    lora_path=lora_name,
                    tokenizer=tokenizer,
                    args=rollout_args,
                )
                generated_text = "\n".join(turn_texts)
            else:
                result = generate_with_sglang(
                    infer_url,
                    input_ids=list(example.prompt_ids),
                    lora_name=lora_name,
                    temperature=float(temperature),
                    max_new_tokens=int(max_new_tokens),
                    ignore_eos=bool(ignore_eos),
                    stop=stop_tokens,
                    reload_lora=reload_lora,
                )
                generated_text = str(result.get("text") or "")
                if not generated_text:
                    output_ids = _generated_output_ids(result, tokenizer)
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True) if output_ids else ""
                score = task.score_completion(example, generated_text)
            error = ""
        except Exception as exc:  # Keep eval failures visible without killing long training jobs.
            score = {"reward": 0.0, "exact_match": 0.0, "error": 1.0}
            generated_text = ""
            error = str(exc)

        row = {
            "event": "sample_eval_example",
            "step": step,
            "time": time.time(),
            "index": idx,
            "project": example.project,
            "target": str(example.metadata.get("target", "")),
            "score": score,
            "generated_text": generated_text,
            "turn_texts": turn_texts,
            "error": error,
        }
        if log_examples < 0 or idx < log_examples:
            _jsonl(log_path, row)
        return row

    stop_tokens = list(stop or [])
    with ThreadPoolExecutor(max_workers=min(int(workers), len(examples))) as pool:
        rows = list(pool.map(score_one, enumerate(examples)))

    numeric_keys: set[str] = set()
    for row in rows:
        score = row.get("score") or {}
        for key, value in score.items():
            if _is_number(value):
                numeric_keys.add(str(key))

    n = max(float(len(rows)), 1.0)
    metrics: dict[str, float] = {"n": float(len(rows))}
    for key in sorted(numeric_keys):
        values = [float((row.get("score") or {}).get(key, 0.0)) for row in rows]
        mean = sum(values) / n
        if key == "exact_match":
            metrics["exact_match_rate"] = mean
            metrics["exact_count"] = sum(values)
        elif key == "error":
            metrics["error_rate"] = mean
        else:
            metrics[f"{key}_mean"] = mean
    metrics.setdefault("exact_match_rate", 0.0)
    metrics.setdefault("exact_count", 0.0)
    return metrics, rows


def _extract_reasoning_text(text: str) -> str:
    match = _REASONING_TAG_RE.search(text or "")
    if match is None:
        return ""
    return match.group(2).strip()


def sample_teacher_wordle_diagnostics(
    *,
    tokenizer,
    task,
    examples: list[Example],
    teacher_url: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    workers: int,
    step: int,
    log_path: Path,
    log_examples: int,
    wordle_prompt_style: str,
    wordle_teacher_prompt_style: str,
    log_prompts: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not examples:
        return {"n": 0.0, "exact_count": 0.0, "exact_match_rate": 0.0}, []
    if not teacher_url:
        raise RuntimeError("teacher sample diagnostics require --teacher-url")
    if workers <= 0:
        raise RuntimeError(f"teacher sample workers must be positive, got {workers}")
    if not hasattr(task, "extract_guess") or not hasattr(task, "compute_feedback"):
        raise RuntimeError("teacher sample diagnostics require Wordle extract_guess and compute_feedback")

    max_turns = int(getattr(task, "MAX_TURNS", 6))
    info_score_fn = getattr(task, "_info_score", None)

    def score_rollout(*, solved: bool, turns_used: int, format_hits: int, info_scores: list[float]) -> dict[str, float]:
        format_rate = float(format_hits) / float(max(turns_used, 1))
        info_gain = sum(info_scores) / float(max(len(info_scores), 1))
        turn_bonus = ((max_turns - turns_used + 1) / max_turns) if solved else 0.0
        reward = 0.4 * float(solved) + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus
        return {
            "reward": float(reward),
            "exact_match": float(solved),
            "format_rate": float(format_rate),
            "info_gain": float(info_gain),
            "turns_used": float(turns_used),
        }

    def sample_one(item: tuple[int, Example]) -> dict[str, Any]:
        idx, example = item
        target = str(example.metadata["target"]).lower()
        history: list[tuple[str, str]] = []
        turn_rows: list[dict[str, Any]] = []
        format_hits = 0
        info_scores: list[float] = []
        solved = False
        error = ""
        try:
            for turn in range(max_turns):
                history_before = list(history)
                public_candidates_before = (
                    _wordle_public_candidates(task, target=target, history=history_before)
                    if hasattr(task, "remaining_candidates")
                    else []
                )
                prompt_ids = _wordle_turn_prompt_ids(
                    tokenizer,
                    task,
                    target=target,
                    history=history_before,
                    hinted=True,
                    prompt_style=wordle_prompt_style,
                    teacher_prompt_style=wordle_teacher_prompt_style,
                )
                result = generate_base_with_sglang(
                    teacher_url,
                    input_ids=prompt_ids,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=ignore_eos,
                    stop=stop,
                )
                output_ids = _generated_output_ids(result, tokenizer)
                text = _generated_text_from_sglang_result(result, tokenizer)
                if hasattr(task, "parse_turn_response"):
                    parsed_turn = task.parse_turn_response(text or "", history_before)
                    guess = str(parsed_turn.get("guess") or "") or None
                    format_ok = bool(parsed_turn.get("format_ok", False))
                    valid_guess = bool(parsed_turn.get("action_ok", False))
                else:
                    parsed_turn = {}
                    guess = task.extract_guess(text or "")
                    format_ok = bool(task.has_single_guess_tag(text or "")) if hasattr(task, "has_single_guess_tag") else bool(guess)
                    valid_guess = format_ok and _wordle_is_valid_guess(task, guess, history_before)
                feedback = ""
                info_score = 0.0
                repeated_guess = bool(guess in {prior_guess for prior_guess, _ in history_before}) if guess else False
                guess_in_public_candidates = bool(guess in public_candidates_before) if guess else False
                if format_ok:
                    format_hits += 1
                if valid_guess:
                    assert guess is not None
                    feedback = task.compute_feedback(guess, target)
                    if callable(info_score_fn):
                        info_score = float(info_score_fn(guess, target, feedback))
                    info_scores.append(info_score)
                    history.append((guess, feedback))
                    solved = guess == target
                else:
                    info_scores.append(0.0)

                row = {
                    "event": "teacher_sample_turn",
                    "step": step,
                    "time": time.time(),
                    "index": idx,
                    "project": example.project,
                    "target": target,
                    "turn": turn + 1,
                    "history_before": history_before,
                    "teacher_guess": guess or "",
                    "feedback": feedback,
                    "format_ok": format_ok,
                    "valid_guess": valid_guess,
                    "repeated_guess": repeated_guess,
                    "guess_in_public_candidates": guess_in_public_candidates,
                    "public_constraint_valid": bool(parsed_turn.get("public_constraint_valid", True)),
                    "target_leak": bool(parsed_turn.get("target_leak", False)),
                    "extra_text": bool(parsed_turn.get("extra_text", False)),
                    "parse_errors": parsed_turn.get("errors", []),
                    "public_candidates_before_count": len(public_candidates_before),
                    "target_in_public_candidates": target in public_candidates_before,
                    "solved": solved,
                    "info_score": info_score,
                    "teacher_prompt_tokens": len(prompt_ids),
                    "teacher_output_tokens": len(output_ids),
                    "wordle_prompt_style": wordle_prompt_style,
                    "wordle_teacher_prompt_style": wordle_teacher_prompt_style,
                    "teacher_reasoning": _extract_reasoning_text(text),
                    "teacher_text": text,
                }
                if log_prompts:
                    row["teacher_prompt_text"] = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                turn_rows.append(row)
                if not valid_guess or solved:
                    break
        except Exception as exc:  # Keep diagnostics visible without killing long training jobs.
            error = str(exc)
            turn_rows.append(
                {
                    "event": "teacher_sample_error",
                    "step": step,
                    "time": time.time(),
                    "index": idx,
                    "project": example.project,
                    "target": target,
                    "turn": len(turn_rows) + 1,
                    "history_before": list(history),
                    "teacher_guess": "",
                    "feedback": "",
                    "format_ok": False,
                    "valid_guess": False,
                    "repeated_guess": False,
                    "guess_in_public_candidates": False,
                    "public_candidates_before_count": 0,
                    "target_in_public_candidates": False,
                    "solved": False,
                    "info_score": 0.0,
                    "teacher_prompt_tokens": 0,
                    "teacher_output_tokens": 0,
                    "wordle_prompt_style": wordle_prompt_style,
                    "wordle_teacher_prompt_style": wordle_teacher_prompt_style,
                    "teacher_reasoning": "",
                    "teacher_text": "",
                    "error": error,
                }
            )

        turns_used = max(len(turn_rows), 1)
        score = score_rollout(
            solved=solved,
            turns_used=turns_used,
            format_hits=format_hits,
            info_scores=info_scores,
        )
        if error:
            score["error"] = 1.0
        else:
            score["error"] = 0.0
        for row in turn_rows:
            row["score"] = score
            row.setdefault("error", error)
        return {"score": score, "rows": turn_rows}

    with ThreadPoolExecutor(max_workers=min(int(workers), len(examples))) as pool:
        sampled = list(pool.map(sample_one, enumerate(examples)))

    rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    for item in sampled:
        scores.append(dict(item.get("score") or {}))
        rows.extend(list(item.get("rows") or []))

    if log_examples < 0:
        max_logged_index = len(examples)
    else:
        max_logged_index = int(log_examples)
    for row in rows:
        if log_examples < 0 or int(row.get("index", 0)) < max_logged_index:
            _jsonl(log_path, row)

    n = max(float(len(scores)), 1.0)
    metrics: dict[str, float] = {
        "n": float(len(scores)),
        "turn_rows": float(len(rows)),
    }
    numeric_keys: set[str] = set()
    for score in scores:
        for key, value in score.items():
            if _is_number(value):
                numeric_keys.add(str(key))
    for key in sorted(numeric_keys):
        values = [float(score.get(key, 0.0)) for score in scores]
        mean = sum(values) / n
        if key == "exact_match":
            metrics["exact_match_rate"] = mean
            metrics["exact_count"] = sum(values)
        elif key == "error":
            metrics["error_rate"] = mean
        else:
            metrics[f"{key}_mean"] = mean
    metrics.setdefault("exact_match_rate", 0.0)
    metrics.setdefault("exact_count", 0.0)
    return metrics, rows


def forward_loss(
    *,
    train_url: str,
    model_id: str,
    rows: list[OpsdRow],
    loss_fn: str,
    loss_params: dict[str, Any],
    objective: str,
    request_batch_size: int,
    future_timeout: float,
) -> dict[str, float]:
    responses = []
    include_cache = objective in {"opd_self_kl", "asymmetric_opsd"}
    for chunk_idx, batch_rows in enumerate(chunks(rows, request_batch_size)):
        result = call_future(
            train_url,
            "/api/v1/forward",
            {
                "model_id": model_id,
                "seq_id": chunk_idx,
                "forward_input": {
                    "data": [row_to_datum(row, include_cache=include_cache, objective=objective) for row in batch_rows],
                    "loss_fn": loss_fn,
                    "loss_fn_params": loss_params,
                },
            },
            context=f"forward(eval chunk {chunk_idx})",
            future_timeout=future_timeout,
        )
        responses.append(result)
    return summarize_responses(responses)


def train_one_step(
    *,
    train_url: str,
    model_id: str,
    rows: list[OpsdRow],
    loss_fn: str,
    loss_params: dict[str, Any],
    objective: str,
    request_batch_size: int,
    lr: float,
    gradient_clip: float,
    future_timeout: float,
) -> dict[str, float]:
    responses = forward_backward_responses(
        train_url=train_url,
        model_id=model_id,
        rows=rows,
        loss_fn=loss_fn,
        loss_params=loss_params,
        objective=objective,
        request_batch_size=request_batch_size,
        future_timeout=future_timeout,
        seq_id_start=0,
        context_prefix="train",
    )

    metrics = summarize_responses(responses)
    opt_result = optim_step(
        train_url=train_url,
        model_id=model_id,
        seq_id=len(responses),
        lr=lr,
        gradient_clip=gradient_clip,
        future_timeout=future_timeout,
    )
    opt_metrics = opt_result.get("metrics") or {}
    for key in ("grad_norm", "learning_rate"):
        value = opt_metrics.get(key)
        if _is_number(value):
            metrics[key] = float(value)
    return metrics


def forward_backward_responses(
    *,
    train_url: str,
    model_id: str,
    rows: list[OpsdRow],
    loss_fn: str,
    loss_params: dict[str, Any],
    objective: str,
    request_batch_size: int,
    future_timeout: float,
    seq_id_start: int = 0,
    context_prefix: str = "train",
) -> list[dict[str, Any]]:
    responses = []
    for request in forward_backward_requests(
        model_id=model_id,
        rows=rows,
        loss_fn=loss_fn,
        loss_params=loss_params,
        objective=objective,
        request_batch_size=request_batch_size,
        seq_id_start=seq_id_start,
    ):
        seq_id = int(request["seq_id"])
        result = call_future(
            train_url,
            "/api/v1/forward_backward",
            request,
            context=f"forward_backward({context_prefix} chunk {seq_id})",
            future_timeout=future_timeout,
        )
        responses.append(result)
    return responses


def optim_step(
    *,
    train_url: str,
    model_id: str,
    seq_id: int,
    lr: float,
    gradient_clip: float,
    future_timeout: float,
) -> dict[str, Any]:
    return call_future(
        train_url,
        "/api/v1/optim_step",
        {
            "model_id": model_id,
            "seq_id": int(seq_id),
            "learning_rate": lr,
            "gradient_clip": gradient_clip,
        },
        context="optim_step",
        future_timeout=future_timeout,
    )


def select_train_examples_for_step(
    train_pool: list[Example],
    *,
    train_size: int,
    seed: int,
    step: int,
    resample: bool,
) -> list[Example]:
    if train_size <= 0:
        raise RuntimeError(f"train_size must be positive, got {train_size}")
    if len(train_pool) < train_size:
        raise RuntimeError(f"train_pool has {len(train_pool)} examples, need at least train_size={train_size}")
    if not resample:
        return list(train_pool[:train_size])
    if train_size == len(train_pool):
        indices = list(range(len(train_pool)))
        random.Random(int(seed) * 1_000_003 + int(step)).shuffle(indices)
        return [train_pool[idx] for idx in indices]

    batches_per_epoch = max(1, math.ceil(len(train_pool) / train_size))
    epoch = int(step) // batches_per_epoch
    batch_index = int(step) % batches_per_epoch
    rng = random.Random(int(seed) * 1_000_003 + epoch)
    indices = list(range(len(train_pool)))
    rng.shuffle(indices)
    start = batch_index * train_size
    batch_indices = indices[start : start + train_size]
    if len(batch_indices) < train_size:
        next_indices = list(range(len(train_pool)))
        random.Random(int(seed) * 1_000_003 + epoch + 1).shuffle(next_indices)
        batch_indices.extend(next_indices[: train_size - len(batch_indices)])
    return [train_pool[idx] for idx in batch_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-url", default="http://127.0.0.1:26040")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--task", default="wordle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-model-id", default="opsd-wordle-native")
    parser.add_argument("--teacher-model-id", default="default")
    parser.add_argument("--teacher-cache-backend", default="xorl", choices=["xorl", "sglang"])
    parser.add_argument(
        "--teacher-url",
        default="",
        help="SGLang teacher base URL when --teacher-cache-backend=sglang.",
    )
    parser.add_argument(
        "--teacher-head-model",
        default="",
        help="Model path/name for the teacher LM head used by opd_loss; defaults to --model.",
    )
    parser.add_argument(
        "--teacher-cache-dir",
        default="",
        help="Directory for transient teacher cache files. Use a path shared by trainer and SGLang teacher.",
    )
    parser.add_argument(
        "--objective",
        default="asymmetric_opsd",
        choices=["asymmetric_opsd", "teacher_forced_ce", "opd_self_kl", "sft_gold"],
    )
    parser.add_argument(
        "--infer-url",
        action="append",
        default=[],
        help="SGLang /generate URL(s) for asymmetric_opsd student rollouts; may be repeated or space-separated.",
    )
    parser.add_argument(
        "--sampler-load-url",
        action="append",
        default=[],
        help=(
            "Direct SGLang URL(s) used for LoRA load/unload. Defaults to --infer-url. "
            "Use this when --infer-url points at a router such as SMG."
        ),
    )
    parser.add_argument("--server-output-dir", default="")
    parser.add_argument("--sampler-lora-name", default="")
    parser.add_argument("--sampler-save-prefix", default="")
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-max-new-tokens", type=int, default=96)
    parser.add_argument("--student-generation-batch-size", type=int, default=8)
    parser.add_argument(
        "--student-generation-workers",
        type=int,
        default=0,
        help=(
            "Logical concurrent Wordle examples during student rollout. "
            "0 preserves the legacy len(infer_url) * student_generation_batch_size cap."
        ),
    )
    parser.add_argument("--student-ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-stop", action="append", default=[])
    parser.add_argument(
        "--sample-eval-interval",
        type=int,
        default=16,
        help="Run sampled student task eval every N train steps; 0 disables it.",
    )
    parser.add_argument(
        "--sample-eval-size",
        type=int,
        default=64,
        help="Number of eval examples for sampled task eval; 0 means all eval examples.",
    )
    parser.add_argument(
        "--sample-eval-temperature",
        type=float,
        default=None,
        help="Sampling temperature for task eval; defaults to --student-temperature.",
    )
    parser.add_argument("--sample-eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--sample-eval-ignore-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-eval-stop", action="append", default=[])
    parser.add_argument("--sample-eval-workers", type=int, default=8)
    parser.add_argument(
        "--sample-eval-log-examples",
        type=int,
        default=16,
        help="Number of per-example sampled eval rows to log per eval; negative logs all.",
    )
    parser.add_argument(
        "--teacher-reason-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For asymmetric Wordle OPSD, generate a teacher rationale for each pre-action state and prepend it "
            "to the teacher-side context before scoring the student-sampled tokens."
        ),
    )
    parser.add_argument(
        "--teacher-reasoning-cache",
        default="",
        help=(
            "JSONL produced by generate_teacher_cot_cache.py; reason-first teacher CoTs are read from it "
            "and only cache misses fall back to online generation."
        ),
    )
    parser.add_argument("--teacher-reasoning-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-reasoning-max-new-tokens", type=int, default=96)
    parser.add_argument("--teacher-reasoning-ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-reasoning-stop", action="append", default=[])
    parser.add_argument(
        "--teacher-sample-interval",
        type=int,
        default=0,
        help="Generate diagnostic teacher CoT samples every N train steps; 0 disables it.",
    )
    parser.add_argument(
        "--teacher-sample-size",
        type=int,
        default=8,
        help="Number of eval examples for diagnostic teacher sampling; 0 means all eval examples.",
    )
    parser.add_argument("--teacher-sample-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-sample-max-new-tokens", type=int, default=96)
    parser.add_argument("--teacher-sample-ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-sample-stop", action="append", default=[])
    parser.add_argument("--teacher-sample-workers", type=int, default=8)
    parser.add_argument(
        "--teacher-sample-log-examples",
        type=int,
        default=8,
        help="Number of per-example teacher sample rows to log per eval; negative logs all.",
    )
    parser.add_argument(
        "--teacher-sample-log-prompts",
        action="store_true",
        help="Also store decoded teacher prompts in teacher_samples.jsonl/W&B tables.",
    )
    parser.add_argument("--keep-teacher-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=9234)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--eval-request-batch-size", type=int, default=8)
    parser.add_argument(
        "--opd-pipeline-chunk-size",
        type=int,
        default=0,
        help="If >0, split each asymmetric OPSD train step into this many examples per prepared chunk.",
    )
    parser.add_argument(
        "--opd-pipeline-prefetch-chunks",
        type=int,
        default=1,
        help="Number of future rollout/cache chunks to prepare while the trainer consumes earlier chunks.",
    )
    parser.add_argument(
        "--dump-forward-backward-replay",
        default="",
        help=(
            "Optional JSONL path for static /forward_backward replay records. Relative paths are placed under "
            "--output-dir; teacher cache files are copied beside the artifact."
        ),
    )
    parser.add_argument(
        "--dump-forward-backward-replay-only",
        action="store_true",
        help="After dumping the first train f/b replay chunk, skip live f/b, optimizer, and final save.",
    )
    parser.add_argument("--resample-train-each-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gold-data", default="", help="Gold SFT JSONL (objective=sft_gold).")
    parser.add_argument("--gold-eval-fraction", type=float, default=0.05)
    parser.add_argument("--wordle-teacher-trace-style", default="hinted_cot")
    parser.add_argument(
        "--wordle-prompt-style",
        default="default",
        choices=[
            "default",
            "candidate_list",
            "public_reasoning",
            "public_reasoning_strict",
            "public_reasoning_strict_nocandidates",
            "public_reasoning_constraints",
            "public_reasoning_constraints_candidates",
            "public_reasoning_constraints_think",
            "public_reasoning_constraints_candidates_think",
        ],
    )
    parser.add_argument(
        "--wordle-teacher-prompt-style",
        default="answer_hint",
        choices=["answer_hint", "policy_hint", "public_policy_hint", "public_candidates_only", "no_hint"],
        help="Teacher prompt for asymmetric Wordle OPSD.",
    )
    parser.add_argument(
        "--wordle-first-turn-weight",
        type=float,
        default=1.0,
        help="KL weight for turn-1 sampled tokens; use 0.0/0.1 to avoid distilling direct answer hints.",
    )
    parser.add_argument(
        "--wordle-tag-token-weight",
        type=float,
        default=0.0,
        help=(
            "Per-token KL weight for generated tag/whitespace/extra tokens. Tokens with weight 0 are masked out "
            "of the OPD labels and denominator."
        ),
    )
    parser.add_argument(
        "--wordle-think-token-weight",
        type=float,
        default=1.0,
        help=(
            "Per-token KL weight for think-block content under *_think prompt styles "
            "(everything sampled before </think>)."
        ),
    )
    parser.add_argument(
        "--wordle-reasoning-token-weight",
        type=float,
        default=1.0,
        help=(
            "Per-token KL weight assigned to tokens inside <reasoning>...</reasoning> or <think>...</think>. "
            "Only applies when the student emits public reasoning."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument(
        "--full-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Full-weight training: no LoRA session; policy reaches samplers via sync_inference_weights "
        "(server config must set enable_lora: false and the samplers must run without --enable-lora).",
    )
    parser.add_argument("--sampler-world-size", type=int, default=2, help="GPUs per sampler shard (TP size).")
    parser.add_argument("--weight-sync-buffer-mb", type=int, default=1024)
    parser.add_argument(
        "--weight-sync-flush-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush sampler KV/radix cache after each weight sync (prefixes scored under old weights).",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        default="",
        help='JSON dict of optimizer-specific kwargs, e.g. \'{"momentum": 0.0}\' for low-memory muon.',
    )
    parser.add_argument(
        "--optimizer", default="adamw", choices=["adamw", "anyprecision_adamw", "sgd", "signsgd", "muon"]
    )
    parser.add_argument("--optimizer-dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--opd-kl-backend", default="torch_compile", choices=["streaming", "tilelang", "torch_compile"])
    parser.add_argument("--opd-vocab-chunk-size", type=int, default=32768)
    parser.add_argument("--opd-loss-mode", default="forward_kl_full")
    parser.add_argument("--emit-full-vocab-diagnostics", action="store_true")
    parser.add_argument("--opd-profile-timings", action="store_true")
    parser.add_argument(
        "--opd-profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around OPD and f/b phase timers for real GPU timings.",
    )
    parser.add_argument(
        "--cache-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"]
    )
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--future-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Stop cleanly after this many seconds in the training loop; 0 means run all --steps.",
    )
    parser.add_argument("--eval-interval", type=int, default=16)
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip the cold step-0 eval path; useful for one-shot f/b replay capture jobs.",
    )
    parser.add_argument("--save-interval", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    replay_dump_path = Path(args.dump_forward_backward_replay) if args.dump_forward_backward_replay else None
    if replay_dump_path is not None and not replay_dump_path.is_absolute():
        replay_dump_path = output_dir / replay_dump_path
    if replay_dump_path is not None:
        replay_dump_path.parent.mkdir(parents=True, exist_ok=True)
        replay_dump_path.unlink(missing_ok=True)
    sample_eval_path = output_dir / "sample_eval.jsonl"
    teacher_sample_path = output_dir / "teacher_samples.jsonl"
    infer_urls = _split_urls(args.infer_url)
    sampler_load_urls = _split_urls(args.sampler_load_url)
    _needs_samplers = args.objective in {"asymmetric_opsd", "sft_gold"}
    active_infer_urls = infer_urls if _needs_samplers else []
    active_sampler_load_urls = (sampler_load_urls or active_infer_urls) if _needs_samplers else []
    student_stop = _split_urls(args.student_stop)
    sample_eval_stop = _split_urls(args.sample_eval_stop)
    teacher_reasoning_stop = _split_urls(args.teacher_reasoning_stop)
    teacher_sample_stop = _split_urls(args.teacher_sample_stop)
    if args.sample_eval_temperature is None:
        args.sample_eval_temperature = args.student_temperature
    server_output_dir = Path(args.server_output_dir) if args.server_output_dir else output_dir / "server_output"
    teacher_cache_root = Path(args.teacher_cache_dir) if args.teacher_cache_dir else output_dir / "teacher_cache"
    sampler_lora_name = "" if args.full_weight else (args.sampler_lora_name or f"{args.train_model_id}-{output_dir.name}")
    sampler_save_prefix = args.sampler_save_prefix or f"{output_dir.name}/student"
    args.teacher_url = args.teacher_url.rstrip("/")

    args.train_pool_size = int(args.train_pool_size or args.train_size)
    if args.train_pool_size < args.train_size:
        raise ValueError("--train-pool-size must be >= --train-size")
    if args.request_batch_size <= 0 or args.eval_request_batch_size <= 0:
        raise ValueError("request batch sizes must be positive")
    if args.student_generation_workers < 0:
        raise ValueError("--student-generation-workers must be non-negative")
    if args.opd_pipeline_chunk_size < 0:
        raise ValueError("--opd-pipeline-chunk-size must be non-negative")
    if args.opd_pipeline_prefetch_chunks <= 0:
        raise ValueError("--opd-pipeline-prefetch-chunks must be positive")
    if args.max_runtime_seconds < 0:
        raise ValueError("--max-runtime-seconds must be non-negative")
    if args.dump_forward_backward_replay_only and replay_dump_path is None:
        raise ValueError("--dump-forward-backward-replay-only requires --dump-forward-backward-replay")
    if args.sample_eval_interval > 0 and args.sample_eval_workers <= 0:
        raise ValueError("--sample-eval-workers must be positive when sampled eval is enabled")
    if args.objective == "asymmetric_opsd" and args.teacher_sample_interval > 0 and args.teacher_sample_workers <= 0:
        raise ValueError("--teacher-sample-workers must be positive when teacher sampling is enabled")
    if args.objective == "asymmetric_opsd" and args.teacher_sample_interval > 0 and not args.teacher_url:
        raise ValueError("--teacher-url is required when --teacher-sample-interval is enabled")
    if args.objective == "asymmetric_opsd" and args.teacher_reason_first and not args.teacher_url:
        raise ValueError("--teacher-url is required when --teacher-reason-first is enabled")
    if (
        args.objective == "asymmetric_opsd"
        and args.teacher_reason_first
        and args.wordle_teacher_prompt_style not in {"policy_hint", "public_policy_hint"}
    ):
        raise ValueError(
            "--teacher-reason-first currently requires --wordle-teacher-prompt-style=policy_hint or public_policy_hint"
        )
    if _needs_samplers and not active_infer_urls:
        raise ValueError("asymmetric_opsd requires at least one --infer-url")
    if _needs_samplers and not active_sampler_load_urls:
        raise ValueError("asymmetric_opsd requires at least one sampler load URL")
    if args.teacher_cache_backend == "sglang" and not args.teacher_url:
        raise ValueError("--teacher-url is required with --teacher-cache-backend=sglang")
    if args.teacher_cache_backend == "sglang" and not args.teacher_cache_dir:
        print(
            "WARN: --teacher-cache-backend=sglang without --teacher-cache-dir; "
            f"using {teacher_cache_root}. This path must exist on both trainer and teacher pods.",
            flush=True,
        )
    if args.wordle_first_turn_weight < 0:
        raise ValueError("--wordle-first-turn-weight must be non-negative")
    if args.wordle_tag_token_weight < 0:
        raise ValueError("--wordle-tag-token-weight must be non-negative")
    if args.wordle_reasoning_token_weight < 0:
        raise ValueError("--wordle-reasoning-token-weight must be non-negative")
    if args.wordle_think_token_weight < 0:
        raise ValueError("--wordle-think-token-weight must be non-negative")
    if (
        args.objective in {"opd_self_kl", "asymmetric_opsd"}
        and args.opd_loss_mode == "forward_kl_full"
        and args.opd_kl_backend != "torch_compile"
    ):
        raise ValueError("opd_loss_mode='forward_kl_full' requires --opd-kl-backend=torch_compile")

    print(f"[init] waiting for training server at {args.train_url}")
    wait_for_training_service(args.train_url, timeout=args.startup_timeout)
    print("[init] training server is ready")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only, trust_remote_code=True
    )
    teacher_head_model = args.teacher_head_model or args.model
    if teacher_head_model != args.model:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_head_model,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        if len(teacher_tokenizer) != len(tokenizer):
            raise ValueError(
                "OPSD full-vocab KL requires tokenizer/vocab compatibility: "
                f"student {args.model!r} tokenizer length={len(tokenizer)} but "
                f"teacher_head_model {teacher_head_model!r} tokenizer length={len(teacher_tokenizer)}"
            )
    task = load_task(args.task)
    train_pool, eval_examples = task.build_examples(
        tokenizer,
        train_size=args.train_pool_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    train_rows: list[OpsdRow] = []
    eval_rows: list[OpsdRow] = []
    row_by_project: dict[str, OpsdRow] = {}
    if args.objective == "sft_gold":
        if not args.gold_data:
            raise RuntimeError("objective=sft_gold requires --gold-data")
        train_rows, eval_rows = build_gold_sft_rows(
            tokenizer,
            args.gold_data,
            max_length=args.max_length,
            eval_fraction=args.gold_eval_fraction,
            seed=args.seed,
        )
        print(f"[gold] loaded {len(train_rows)} train rows / {len(eval_rows)} eval rows from {args.gold_data}")
    elif args.objective != "asymmetric_opsd":
        trace_args = SimpleNamespace(wordle_teacher_trace_style=args.wordle_teacher_trace_style)
        train_rows = build_opsd_rows(
            tokenizer=tokenizer,
            task=task,
            examples=train_pool,
            trace_args=trace_args,
            max_length=args.max_length,
        )
        eval_rows = build_opsd_rows(
            tokenizer=tokenizer,
            task=task,
            examples=eval_examples,
            trace_args=trace_args,
            max_length=args.max_length,
        )
        row_by_project = {row.project: row for row in train_rows}

    needs_teacher_cache = args.objective in {"opd_self_kl", "asymmetric_opsd"}
    teacher_head_dir = (
        resolve_model_dir(teacher_head_model, local_files_only=args.local_files_only) if needs_teacher_cache else None
    )
    train_cache_path = (
        teacher_cache_root / "train_hidden.safetensors" if args.objective == "opd_self_kl" else None
    )
    eval_cache_path = (
        teacher_cache_root / "eval_hidden.safetensors" if args.objective == "opd_self_kl" else None
    )

    run_config = {
        **vars(args),
        "infer_urls": infer_urls,
        "sampler_load_urls": sampler_load_urls,
        "active_infer_urls": active_infer_urls,
        "active_sampler_load_urls": active_sampler_load_urls,
        "student_stop": student_stop,
        "sample_eval_stop": sample_eval_stop,
        "teacher_reasoning_stop": teacher_reasoning_stop,
        "teacher_sample_stop": teacher_sample_stop,
        "server_output_dir": str(server_output_dir),
        "sampler_lora_name": sampler_lora_name,
        "sampler_save_prefix": sampler_save_prefix,
        "teacher_head_model_resolved": teacher_head_model,
        "teacher_head_dir": teacher_head_dir,
        "teacher_cache_root": str(teacher_cache_root),
        "dump_forward_backward_replay_path": str(replay_dump_path) if replay_dump_path is not None else "",
        "train_examples": len(train_pool) if args.objective == "asymmetric_opsd" else len(train_rows),
        "eval_examples": len(eval_examples) if args.objective == "asymmetric_opsd" else len(eval_rows),
        "train_cache_path": str(train_cache_path) if train_cache_path is not None else None,
        "eval_cache_path": str(eval_cache_path) if eval_cache_path is not None else None,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    if args.objective == "asymmetric_opsd":
        (output_dir / "train_examples.json").write_text(
            json.dumps([asdict(example) for example in train_pool], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "eval_examples.json").write_text(
            json.dumps([asdict(example) for example in eval_examples], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[init] asymmetric_opsd student generation endpoints="
            f"{len(active_infer_urls)} sampler_load_endpoints={len(active_sampler_load_urls)} "
            f"student_generation_workers={args.student_generation_workers or 'legacy'} "
            f"pipeline_chunk_size={args.opd_pipeline_chunk_size} "
            f"pipeline_prefetch_chunks={args.opd_pipeline_prefetch_chunks} "
            f"teacher_reason_first={args.teacher_reason_first}",
            flush=True,
        )
    else:
        (output_dir / "train_rows.json").write_text(
            json.dumps([asdict(row) for row in train_rows], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "eval_rows.json").write_text(
            json.dumps([asdict(row) for row in eval_rows], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _jsonl(metrics_path, {"event": "init", "time": time.time(), **run_config})

    if args.wandb_project:
        import wandb  # noqa: PLC0415

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or None, config=run_config)
    else:
        wandb_run = None

    print(f"[init] creating native LoRA session {args.train_model_id}")
    create_result = create_model(args.train_url, args)
    _jsonl(metrics_path, {"event": "create_model", "time": time.time(), "result": create_result})

    if args.objective == "opd_self_kl":
        assert teacher_head_dir is not None
        assert train_cache_path is not None
        assert eval_cache_path is not None
        print(f"[cache] materializing train teacher hidden cache: {train_cache_path}")
        train_cache_meta = materialize_teacher_cache(
            backend=args.teacher_cache_backend,
            train_url=args.train_url,
            teacher_url=args.teacher_url,
            model_id=args.teacher_model_id,
            rows=train_rows,
            cache_path=train_cache_path,
            cache_dtype=args.cache_dtype,
            objective=args.objective,
            future_timeout=args.future_timeout,
        )
        print(f"[cache] train tokens={train_cache_meta.get('num_tokens')} hidden={train_cache_meta.get('hidden_size')}")
        _jsonl(metrics_path, {"event": "teacher_cache_train", "time": time.time(), **train_cache_meta})

        print(f"[cache] materializing eval teacher hidden cache: {eval_cache_path}")
        eval_cache_meta = materialize_teacher_cache(
            backend=args.teacher_cache_backend,
            train_url=args.train_url,
            teacher_url=args.teacher_url,
            model_id=args.teacher_model_id,
            rows=eval_rows,
            cache_path=eval_cache_path,
            cache_dtype=args.cache_dtype,
            objective=args.objective,
            future_timeout=args.future_timeout,
        )
        print(f"[cache] eval tokens={eval_cache_meta.get('num_tokens')} hidden={eval_cache_meta.get('hidden_size')}")
        _jsonl(metrics_path, {"event": "teacher_cache_eval", "time": time.time(), **eval_cache_meta})

        loss_fn = "opd_loss"
        train_loss_params = opd_loss_params(args, teacher_head_dir=teacher_head_dir, cache_path=train_cache_path)
        eval_loss_params = opd_loss_params(args, teacher_head_dir=teacher_head_dir, cache_path=eval_cache_path)
    elif args.objective in {"teacher_forced_ce", "sft_gold"}:
        loss_fn = "causallm_loss"
        train_loss_params = ce_loss_params(args)
        eval_loss_params = ce_loss_params(args)
    else:
        assert args.objective == "asymmetric_opsd"
        assert teacher_head_dir is not None
        loss_fn = "opd_loss"
        train_loss_params = {}
        eval_loss_params = {}

    loaded_policy_step: int | None = None
    sampler_reload_lock = threading.RLock()

    def cleanup_teacher_cache(cache_path: Path | None) -> None:
        if cache_path is not None and not args.keep_teacher_cache:
            cache_path.unlink(missing_ok=True)

    endpoints_registered = False

    def ensure_sampler(policy_step: int, *, force: bool = False) -> None:
        nonlocal loaded_policy_step, endpoints_registered
        with sampler_reload_lock:
            if args.objective not in {"asymmetric_opsd", "sft_gold"}:
                return
            if loaded_policy_step == policy_step and not force:
                return
            if args.full_weight:
                if not endpoints_registered:
                    register_inference_endpoints(
                        args.train_url,
                        active_sampler_load_urls,
                        world_size=args.sampler_world_size,
                    )
                    endpoints_registered = True
                weight_version = f"{sampler_save_prefix}/policy-{policy_step:06d}"
                print(f"[sampler step={policy_step}] full-weight sync to samplers: {weight_version}")
                started = time.time()
                sync_weights_to_samplers(
                    train_url=args.train_url,
                    model_id=args.train_model_id,
                    weight_version=weight_version,
                    output_dir=output_dir,
                    buffer_size_mb=args.weight_sync_buffer_mb,
                    flush_cache=args.weight_sync_flush_cache,
                    timeout=args.future_timeout,
                )
                print(
                    f"[sampler step={policy_step}] full-weight sync done in {time.time() - started:.1f}s "
                    f"({len(active_sampler_load_urls)} endpoint(s))"
                )
                loaded_policy_step = policy_step
                return
            save_name = f"{sampler_save_prefix}/policy-{policy_step:06d}"
            action = "reloading" if force else "exporting"
            print(f"[sampler step={policy_step}] {action} native LoRA for SGLang: {save_name}")
            sampler_path = export_and_load_sampler(
                train_url=args.train_url,
                model_id=args.train_model_id,
                output_dir=output_dir,
                server_output_dir=server_output_dir,
                infer_urls=active_sampler_load_urls,
                lora_name=sampler_lora_name,
                save_name=save_name,
                future_timeout=args.future_timeout,
            )
            print(
                f"[sampler step={policy_step}] loaded {sampler_lora_name} from {sampler_path} "
                f"on {len(active_sampler_load_urls)} load endpoint(s); "
                f"generate endpoints={len(active_infer_urls)}"
            )
            loaded_policy_step = policy_step

    def reload_sampler(policy_step: int) -> None:
        with sampler_reload_lock:
            ensure_sampler(policy_step, force=True)

    def selected_sample_eval_examples() -> list[Example]:
        if args.sample_eval_size <= 0 or args.sample_eval_size >= len(eval_examples):
            return list(eval_examples)
        return list(eval_examples[: args.sample_eval_size])

    def selected_teacher_sample_examples() -> list[Example]:
        if args.teacher_sample_size <= 0 or args.teacher_sample_size >= len(eval_examples):
            return list(eval_examples)
        return list(eval_examples[: args.teacher_sample_size])

    def run_sample_eval(policy_step: int) -> None:
        if args.objective not in {"asymmetric_opsd", "sft_gold"} or args.sample_eval_interval <= 0:
            return
        examples = selected_sample_eval_examples()
        if not examples:
            return
        ensure_sampler(policy_step)
        started = time.time()
        metrics, rows = sample_eval_student(
            tokenizer=tokenizer,
            task=task,
            examples=examples,
            infer_urls=active_infer_urls,
            lora_name=sampler_lora_name,
            temperature=float(args.sample_eval_temperature),
            max_new_tokens=args.sample_eval_max_new_tokens,
            ignore_eos=args.sample_eval_ignore_eos,
            stop=sample_eval_stop,
            workers=args.sample_eval_workers,
            step=policy_step,
            log_path=sample_eval_path,
            log_examples=args.sample_eval_log_examples,
            wordle_prompt_style=args.wordle_prompt_style,
            reload_lora=lambda: reload_sampler(policy_step),
        )
        metrics["sample_eval_time_s"] = time.time() - started
        print(f"[sample_eval step={policy_step}] {format_sample_eval_summary(metrics)}")
        _jsonl(metrics_path, {"event": "sample_eval", "step": policy_step, "time": time.time(), **metrics})
        if wandb_run is not None:
            log_payload: dict[str, Any] = {f"sample_eval/{k}": v for k, v in metrics.items()}
            if rows:
                import wandb as wandb_module  # noqa: PLC0415

                if args.sample_eval_log_examples < 0:
                    table_rows = rows
                else:
                    table_rows = [
                        row for row in rows if int(row.get("index", 0)) < int(args.sample_eval_log_examples)
                    ]
                if table_rows:
                    columns = [
                        "step",
                        "index",
                        "project",
                        "target",
                        "reward",
                        "exact_match",
                        "format_rate",
                        "strict_format_rate",
                        "public_constraint_rate",
                        "target_leak_rate",
                        "extra_text_rate",
                        "invalid_action",
                        "generated_text",
                        "turn_texts",
                        "reasoning_snippets",
                        "guess_snippets",
                        "error",
                    ]
                    table = wandb_module.Table(columns=columns)
                    for row in table_rows:
                        score = row.get("score") or {}
                        generated_text = str(row.get("generated_text", ""))
                        reasoning_snippets = [
                            match.group(2).strip()
                            for match in _REASONING_TAG_RE.finditer(generated_text)
                            if match.group(2).strip()
                        ]
                        guess_snippets = [
                            match.group(1).upper()
                            for match in _GUESS_TAG_RE.finditer(generated_text)
                            if match.group(1)
                        ]
                        table.add_data(
                            row.get("step", policy_step),
                            row.get("index", 0),
                            row.get("project", ""),
                            row.get("target", ""),
                            float(score.get("reward", 0.0)) if _is_number(score.get("reward", 0.0)) else 0.0,
                            float(score.get("exact_match", 0.0))
                            if _is_number(score.get("exact_match", 0.0))
                            else 0.0,
                            float(score.get("format_rate", 0.0))
                            if _is_number(score.get("format_rate", 0.0))
                            else 0.0,
                            float(score.get("strict_format_rate", 0.0))
                            if _is_number(score.get("strict_format_rate", 0.0))
                            else 0.0,
                            float(score.get("public_constraint_rate", 0.0))
                            if _is_number(score.get("public_constraint_rate", 0.0))
                            else 0.0,
                            float(score.get("target_leak_rate", 0.0))
                            if _is_number(score.get("target_leak_rate", 0.0))
                            else 0.0,
                            float(score.get("extra_text_rate", 0.0))
                            if _is_number(score.get("extra_text_rate", 0.0))
                            else 0.0,
                            float(score.get("invalid_action", 0.0))
                            if _is_number(score.get("invalid_action", 0.0))
                            else 0.0,
                            generated_text,
                            json.dumps(row.get("turn_texts", []), ensure_ascii=False),
                            json.dumps(reasoning_snippets, ensure_ascii=False),
                            json.dumps(guess_snippets, ensure_ascii=False),
                            row.get("error", ""),
                        )
                    log_payload["sample_eval/examples"] = table

                    # HTML render of the samples for easy per-step eyeballing in wandb.
                    def _esc(s: str) -> str:
                        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

                    def _hl(esc_text: str) -> str:
                        # colorize the contract tags (already HTML-escaped)
                        for tag, col in (("think", "#888"), ("reasoning", "#1a7f37"), ("guess", "#cf222e")):
                            esc_text = esc_text.replace(f"&lt;{tag}&gt;", f"<b style='color:{col}'>&lt;{tag}&gt;</b>")
                            esc_text = esc_text.replace(f"&lt;/{tag}&gt;", f"<b style='color:{col}'>&lt;/{tag}&gt;</b>")
                        return esc_text

                    _blocks = [f"<div style='font-family:monospace;font-size:12px'><h3>sample_eval step {policy_step}</h3>"]
                    for _row in table_rows:
                        _sc = _row.get("score") or {}
                        _gt = _hl(_esc(_row.get("generated_text", "")))
                        _blocks.append(
                            f"<div style='border:1px solid #ccc;margin:6px 0;padding:6px'>"
                            f"<b>target={_esc(_row.get('target',''))}</b> "
                            f"exact={_sc.get('exact_match',0)} format={_sc.get('format_rate',0)} "
                            f"valid={_sc.get('valid_guess_rate',0)} reward={_sc.get('reward',0)}"
                            f"{(' <span style=color:#cf222e>ERR:'+_esc(_row.get('error',''))+'</span>') if _row.get('error') else ''}"
                            f"<pre style='white-space:pre-wrap;margin:4px 0'>{_gt}</pre></div>"
                        )
                    _blocks.append("</div>")
                    try:
                        log_payload["sample_eval/samples_html"] = wandb_module.Html("".join(_blocks))
                    except Exception:
                        pass
            wandb_run.log(log_payload, step=policy_step)

    def run_teacher_sample(policy_step: int) -> None:
        if args.objective != "asymmetric_opsd" or args.teacher_sample_interval <= 0:
            return
        examples = selected_teacher_sample_examples()
        if not examples:
            return
        started = time.time()
        metrics, rows = sample_teacher_wordle_diagnostics(
            tokenizer=tokenizer,
            task=task,
            examples=examples,
            teacher_url=args.teacher_url,
            temperature=args.teacher_sample_temperature,
            max_new_tokens=args.teacher_sample_max_new_tokens,
            ignore_eos=args.teacher_sample_ignore_eos,
            stop=teacher_sample_stop,
            workers=args.teacher_sample_workers,
            step=policy_step,
            log_path=teacher_sample_path,
            log_examples=args.teacher_sample_log_examples,
            wordle_prompt_style=args.wordle_prompt_style,
            wordle_teacher_prompt_style=args.wordle_teacher_prompt_style,
            log_prompts=args.teacher_sample_log_prompts,
        )
        metrics["teacher_sample_time_s"] = time.time() - started
        print(f"[teacher_sample step={policy_step}] {format_teacher_sample_summary(metrics)}")
        _jsonl(metrics_path, {"event": "teacher_sample", "step": policy_step, "time": time.time(), **metrics})
        if wandb_run is None:
            return

        log_payload: dict[str, Any] = {f"teacher_sample/{k}": v for k, v in metrics.items()}
        if rows:
            import wandb as wandb_module  # noqa: PLC0415

            if args.teacher_sample_log_examples < 0:
                table_rows = rows
            else:
                table_rows = [
                    row for row in rows if int(row.get("index", 0)) < int(args.teacher_sample_log_examples)
                ]
            if table_rows:
                columns = [
                    "step",
                    "index",
                    "project",
                    "target",
                    "turn",
                    "history_before",
                    "teacher_guess",
                    "feedback",
                    "valid_guess",
                    "repeated_guess",
                    "guess_in_public_candidates",
                    "public_constraint_valid",
                    "target_leak",
                    "extra_text",
                    "parse_errors",
                    "public_candidates_before_count",
                    "solved",
                    "reward",
                    "exact_match",
                    "teacher_reasoning",
                    "teacher_text",
                    "error",
                    "teacher_prompt_text",
                ]
                table = wandb_module.Table(columns=columns)
                for row in table_rows:
                    score = row.get("score") or {}
                    table.add_data(
                        row.get("step", policy_step),
                        row.get("index", 0),
                        row.get("project", ""),
                        row.get("target", ""),
                        row.get("turn", 0),
                        json.dumps(row.get("history_before", []), sort_keys=True),
                        row.get("teacher_guess", ""),
                        row.get("feedback", ""),
                        row.get("valid_guess", False),
                        row.get("repeated_guess", False),
                        row.get("guess_in_public_candidates", False),
                        row.get("public_constraint_valid", True),
                        row.get("target_leak", False),
                        row.get("extra_text", False),
                        json.dumps(row.get("parse_errors", []), sort_keys=True),
                        row.get("public_candidates_before_count", 0),
                        row.get("solved", False),
                        float(score.get("reward", 0.0)) if _is_number(score.get("reward", 0.0)) else 0.0,
                        float(score.get("exact_match", 0.0)) if _is_number(score.get("exact_match", 0.0)) else 0.0,
                        row.get("teacher_reasoning", ""),
                        row.get("teacher_text", ""),
                        row.get("error", ""),
                        row.get("teacher_prompt_text", ""),
                    )
                log_payload["teacher_sample/examples"] = table
        wandb_run.log(log_payload, step=policy_step)

    def make_asymmetric_rows(
        *,
        split: str,
        examples: list[Example],
        policy_step: int,
    ) -> tuple[list[OpsdRow], dict[str, Any], Path]:
        assert teacher_head_dir is not None
        ensure_sampler(policy_step)
        rollout_started = time.time()
        rows = build_asymmetric_opsd_rows(
            tokenizer=tokenizer,
            task=task,
            examples=examples,
            infer_urls=active_infer_urls,
            lora_name=sampler_lora_name,
            temperature=args.student_temperature,
            max_new_tokens=args.student_max_new_tokens,
            ignore_eos=args.student_ignore_eos,
            stop=student_stop,
            generation_batch_size=args.student_generation_batch_size,
            generation_workers=args.student_generation_workers or None,
            max_length=args.max_length,
            generation_log_path=output_dir / "generations.jsonl",
            wordle_prompt_style=args.wordle_prompt_style,
            wordle_teacher_prompt_style=args.wordle_teacher_prompt_style,
            wordle_first_turn_weight=args.wordle_first_turn_weight,
            wordle_tag_token_weight=args.wordle_tag_token_weight,
            wordle_reasoning_token_weight=args.wordle_reasoning_token_weight,
            wordle_think_token_weight=args.wordle_think_token_weight,
            teacher_reasoning_cache=(
                _load_teacher_reasoning_cache(args.teacher_reasoning_cache) if args.teacher_reasoning_cache else None
            ),
            teacher_reason_first=args.teacher_reason_first,
            teacher_reasoning_url=args.teacher_url,
            teacher_reasoning_temperature=args.teacher_reasoning_temperature,
            teacher_reasoning_max_new_tokens=args.teacher_reasoning_max_new_tokens,
            teacher_reasoning_ignore_eos=args.teacher_reasoning_ignore_eos,
            teacher_reasoning_stop=teacher_reasoning_stop,
            reload_lora=lambda: reload_sampler(policy_step),
        )
        rollout_time_s = time.time() - rollout_started
        rollout_tokens = sum(row.target_token_count for row in rows)
        print(
            f"[rollout {split} policy={policy_step}] rows={len(rows)} tokens={rollout_tokens} "
            f"generate_endpoints={len(active_infer_urls)} "
            f"generation_workers={args.student_generation_workers or (len(active_infer_urls) * args.student_generation_batch_size)} "
            f"dt={rollout_time_s:.1f}s"
        )
        _jsonl(
            metrics_path,
            {
                "event": f"student_rollout_{split}",
                "step": policy_step,
                "time": time.time(),
                "rows": len(rows),
                "tokens": rollout_tokens,
                "generate_endpoints": len(active_infer_urls),
                "generation_workers": args.student_generation_workers
                or (len(active_infer_urls) * args.student_generation_batch_size),
                "rollout_time_s": rollout_time_s,
            },
        )
        cache_path = (
            teacher_cache_root / f"{split}_policy_{policy_step:06d}_{int(time.time() * 1000)}.safetensors"
        )
        print(f"[cache {split} policy={policy_step}] materializing hinted teacher cache: {cache_path}")
        cache_started = time.time()
        cache_meta = materialize_teacher_cache(
            backend=args.teacher_cache_backend,
            train_url=args.train_url,
            teacher_url=args.teacher_url,
            model_id=args.teacher_model_id,
            rows=rows,
            cache_path=cache_path,
            cache_dtype=args.cache_dtype,
            objective=args.objective,
            future_timeout=args.future_timeout,
        )
        cache_time_s = time.time() - cache_started
        print(
            f"[cache {split} policy={policy_step}] "
            f"tokens={cache_meta.get('num_tokens')} hidden={cache_meta.get('hidden_size')} dt={cache_time_s:.1f}s"
        )
        _jsonl(
            metrics_path,
            {
                "event": f"teacher_cache_{split}",
                "step": policy_step,
                "time": time.time(),
                "cache_path": str(cache_path),
                "rows": len(rows),
                "cache_time_s": cache_time_s,
                **cache_meta,
            },
        )
        return rows, opd_loss_params(args, teacher_head_dir=teacher_head_dir, cache_path=cache_path), cache_path

    def prepare_asymmetric_chunk(
        *,
        split: str,
        examples: list[Example],
        policy_step: int,
        chunk_idx: int,
    ) -> dict[str, Any]:
        started = time.time()
        rows, loss_params, cache_path = make_asymmetric_rows(
            split=f"{split}_chunk{chunk_idx:03d}",
            examples=examples,
            policy_step=policy_step,
        )
        return {
            "chunk_idx": chunk_idx,
            "examples": len(examples),
            "rows": rows,
            "loss_params": loss_params,
            "cache_path": cache_path,
            "prepare_time_s": time.time() - started,
        }

    def train_asymmetric_pipeline_step(*, selected_examples: list[Example], policy_step: int) -> dict[str, float]:
        if args.objective != "asymmetric_opsd":
            raise RuntimeError("train_asymmetric_pipeline_step is only valid for asymmetric_opsd")
        if args.opd_pipeline_chunk_size <= 0:
            raise RuntimeError("OPD pipeline requested with non-positive chunk size")
        example_chunks = list(chunks(selected_examples, args.opd_pipeline_chunk_size))
        if not example_chunks:
            raise RuntimeError("OPD pipeline got no train examples")

        ensure_sampler(policy_step)
        prefetch_chunks = max(1, min(args.opd_pipeline_prefetch_chunks, len(example_chunks)))
        print(
            f"[pipeline train policy={policy_step}] chunks={len(example_chunks)} "
            f"chunk_size={args.opd_pipeline_chunk_size} prefetch={prefetch_chunks}",
            flush=True,
        )
        _jsonl(
            metrics_path,
            {
                "event": "opd_pipeline_train_start",
                "step": policy_step + 1,
                "policy_step": policy_step,
                "time": time.time(),
                "chunks": len(example_chunks),
                "chunk_size": args.opd_pipeline_chunk_size,
                "prefetch_chunks": prefetch_chunks,
                "generation_workers": args.student_generation_workers
                or (len(active_infer_urls) * args.student_generation_batch_size),
            },
        )

        executor = ThreadPoolExecutor(max_workers=prefetch_chunks)
        futures: dict[int, Future[dict[str, Any]]] = {}
        responses: list[dict[str, Any]] = []
        live_cache_paths: list[Path] = []
        seq_id = 0
        next_submit_idx = 0
        prepare_wait_s = 0.0
        forward_backward_s = 0.0
        prepared_rows = 0
        prepared_tokens = 0

        def submit_until_prefetched() -> None:
            nonlocal next_submit_idx
            while next_submit_idx < len(example_chunks) and len(futures) < prefetch_chunks:
                idx = next_submit_idx
                futures[idx] = executor.submit(
                    prepare_asymmetric_chunk,
                    split="train",
                    examples=list(example_chunks[idx]),
                    policy_step=policy_step,
                    chunk_idx=idx,
                )
                next_submit_idx += 1

        submit_until_prefetched()
        try:
            for chunk_idx in range(len(example_chunks)):
                wait_started = time.time()
                prepared = futures.pop(chunk_idx).result()
                prepare_wait_s += time.time() - wait_started
                submit_until_prefetched()

                cache_path = prepared["cache_path"]
                live_cache_paths.append(cache_path)
                rows = prepared["rows"]
                prepared_rows += len(rows)
                prepared_tokens += sum(row.target_token_count for row in rows)

                replay_dump_requests = 0
                if replay_dump_path is not None:
                    replay_dump_requests = dump_forward_backward_replay(
                        replay_path=replay_dump_path,
                        model_id=args.train_model_id,
                        rows=rows,
                        loss_fn=loss_fn,
                        loss_params=prepared["loss_params"],
                        objective=args.objective,
                        request_batch_size=args.request_batch_size,
                        seq_id_start=seq_id,
                        policy_step=policy_step,
                        chunk_idx=chunk_idx,
                        cache_path=cache_path,
                    )
                    if args.dump_forward_backward_replay_only:
                        cleanup_teacher_cache(cache_path)
                        live_cache_paths.remove(cache_path)
                        return {
                            "replay_dump_only": 1.0,
                            "replay_dump_requests": float(replay_dump_requests),
                            "opd_pipeline_enabled": 1.0,
                            "opd_pipeline_chunks": float(len(example_chunks)),
                            "opd_pipeline_prefetch_chunks": float(prefetch_chunks),
                            "opd_pipeline_prepare_wait_s": float(prepare_wait_s),
                            "opd_pipeline_forward_backward_s": 0.0,
                            "opd_pipeline_forward_backward_requests": 0.0,
                            "train_examples": float(len(selected_examples)),
                            "train_rows": float(prepared_rows),
                            "train_tokens": float(prepared_tokens),
                        }

                fb_started = time.time()
                chunk_responses = forward_backward_responses(
                    train_url=args.train_url,
                    model_id=args.train_model_id,
                    rows=rows,
                    loss_fn=loss_fn,
                    loss_params=prepared["loss_params"],
                    objective=args.objective,
                    request_batch_size=args.request_batch_size,
                    future_timeout=args.future_timeout,
                    seq_id_start=seq_id,
                    context_prefix=f"train pipeline {chunk_idx}",
                )
                chunk_fb_s = time.time() - fb_started
                forward_backward_s += chunk_fb_s
                seq_id += len(chunk_responses)
                responses.extend(chunk_responses)
                cleanup_teacher_cache(cache_path)
                live_cache_paths.remove(cache_path)

                _jsonl(
                    metrics_path,
                    {
                        "event": "opd_pipeline_train_chunk",
                        "step": policy_step + 1,
                        "policy_step": policy_step,
                        "time": time.time(),
                        "chunk_idx": chunk_idx,
                        "examples": prepared["examples"],
                        "rows": len(rows),
                        "tokens": sum(row.target_token_count for row in rows),
                        "prepare_time_s": prepared["prepare_time_s"],
                        "forward_backward_time_s": chunk_fb_s,
                        "forward_backward_requests": len(chunk_responses),
                        "seq_id_start": seq_id - len(chunk_responses),
                    },
                )

            metrics = summarize_responses(responses)
            opt_result = optim_step(
                train_url=args.train_url,
                model_id=args.train_model_id,
                seq_id=seq_id,
                lr=args.lr,
                gradient_clip=args.gradient_clip,
                future_timeout=args.future_timeout,
            )
            opt_metrics = opt_result.get("metrics") or {}
            for key in ("grad_norm", "learning_rate"):
                value = opt_metrics.get(key)
                if _is_number(value):
                    metrics[key] = float(value)
            metrics["opd_pipeline_enabled"] = 1.0
            metrics["opd_pipeline_chunks"] = float(len(example_chunks))
            metrics["opd_pipeline_prefetch_chunks"] = float(prefetch_chunks)
            metrics["opd_pipeline_prepare_wait_s"] = float(prepare_wait_s)
            metrics["opd_pipeline_forward_backward_s"] = float(forward_backward_s)
            metrics["opd_pipeline_forward_backward_requests"] = float(seq_id)
            metrics["train_examples"] = float(len(selected_examples))
            metrics["train_rows"] = float(prepared_rows)
            metrics["train_tokens"] = float(prepared_tokens)
            return metrics
        finally:
            for future in futures.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for cache_path in list(live_cache_paths):
                cleanup_teacher_cache(cache_path)
                live_cache_paths.remove(cache_path)
            for future in futures.values():
                if future.done() and not future.cancelled():
                    try:
                        prepared = future.result()
                    except Exception:
                        continue
                    cache_path = prepared.get("cache_path")
                    if isinstance(cache_path, Path):
                        cleanup_teacher_cache(cache_path)

    if args.skip_initial_eval:
        print("[eval step=0] skipped")
        _jsonl(metrics_path, {"event": "eval_skipped", "step": 0, "time": time.time()})
    else:
        print(f"[eval step=0] running cold {args.objective} eval")
        eval_cache_for_cleanup: Path | None = None
        if args.objective == "asymmetric_opsd":
            eval_rows, eval_loss_params, eval_cache_for_cleanup = make_asymmetric_rows(
                split="eval",
                examples=eval_examples,
                policy_step=0,
            )
        try:
            eval_metrics = forward_loss(
                train_url=args.train_url,
                model_id=args.train_model_id,
                rows=eval_rows,
                loss_fn=loss_fn,
                loss_params=eval_loss_params,
                objective=args.objective,
                request_batch_size=args.eval_request_batch_size,
                future_timeout=args.future_timeout,
            )
        finally:
            cleanup_teacher_cache(eval_cache_for_cleanup)
        print(f"[eval step=0] {format_metric_summary(eval_metrics)}")
        _jsonl(metrics_path, {"event": "eval", "step": 0, "time": time.time(), **eval_metrics})
        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=0)
        run_sample_eval(0)
        run_teacher_sample(0)

    train_loop_started_at = time.time()
    for step in range(1, args.steps + 1):
        elapsed_s = time.time() - train_loop_started_at
        if args.max_runtime_seconds > 0 and elapsed_s >= args.max_runtime_seconds:
            print(
                f"[time_limit] stopping before step={step}: "
                f"elapsed={elapsed_s:.1f}s limit={args.max_runtime_seconds:.1f}s"
            )
            _jsonl(
                metrics_path,
                {
                    "event": "time_limit",
                    "step": step,
                    "time": time.time(),
                    "elapsed_s": elapsed_s,
                    "max_runtime_seconds": args.max_runtime_seconds,
                },
            )
            break
        selected_examples = select_train_examples_for_step(
            train_pool,
            train_size=args.train_size,
            seed=args.seed,
            step=step - 1,
            resample=args.resample_train_each_step,
        )
        train_cache_for_cleanup: Path | None = None
        started = time.time()
        if args.objective == "asymmetric_opsd" and args.opd_pipeline_chunk_size > 0:
            train_metrics = train_asymmetric_pipeline_step(
                selected_examples=selected_examples,
                policy_step=step - 1,
            )
        else:
            if args.objective == "asymmetric_opsd":
                selected_rows, train_loss_params, train_cache_for_cleanup = make_asymmetric_rows(
                    split="train",
                    examples=selected_examples,
                    policy_step=step - 1,
                )
            elif args.objective == "sft_gold":
                step_rng = random.Random(args.seed * 100003 + step)
                selected_rows = step_rng.sample(train_rows, min(args.train_size, len(train_rows)))
            else:
                selected_rows = [row_by_project[example.project] for example in selected_examples]
            try:
                replay_dump_requests = 0
                if replay_dump_path is not None:
                    replay_dump_requests = dump_forward_backward_replay(
                        replay_path=replay_dump_path,
                        model_id=args.train_model_id,
                        rows=selected_rows,
                        loss_fn=loss_fn,
                        loss_params=train_loss_params,
                        objective=args.objective,
                        request_batch_size=args.request_batch_size,
                        seq_id_start=0,
                        policy_step=step - 1,
                        chunk_idx=0,
                        cache_path=train_cache_for_cleanup,
                    )
                if args.dump_forward_backward_replay_only:
                    train_metrics = {
                        "replay_dump_only": 1.0,
                        "replay_dump_requests": float(replay_dump_requests),
                        "train_tokens": float(sum(row.target_token_count for row in selected_rows)),
                    }
                else:
                    train_metrics = train_one_step(
                        train_url=args.train_url,
                        model_id=args.train_model_id,
                        rows=selected_rows,
                        loss_fn=loss_fn,
                        loss_params=train_loss_params,
                        objective=args.objective,
                        request_batch_size=args.request_batch_size,
                        lr=args.lr,
                        gradient_clip=args.gradient_clip,
                        future_timeout=args.future_timeout,
                    )
            finally:
                cleanup_teacher_cache(train_cache_for_cleanup)
            train_metrics["train_examples"] = float(len(selected_rows))
        train_metrics["step_time_s"] = time.time() - started

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[train step={step}] {format_metric_summary(train_metrics, include_grad=True, step_time_s=train_metrics['step_time_s'])}"
            )
        _jsonl(metrics_path, {"event": "train", "step": step, "time": time.time(), **train_metrics})
        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v for k, v in train_metrics.items()}, step=step)

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_cache_for_cleanup = None
            if args.objective == "asymmetric_opsd":
                eval_rows, eval_loss_params, eval_cache_for_cleanup = make_asymmetric_rows(
                    split="eval",
                    examples=eval_examples,
                    policy_step=step,
                )
            try:
                eval_metrics = forward_loss(
                    train_url=args.train_url,
                    model_id=args.train_model_id,
                    rows=eval_rows,
                    loss_fn=loss_fn,
                    loss_params=eval_loss_params,
                    objective=args.objective,
                    request_batch_size=args.eval_request_batch_size,
                    future_timeout=args.future_timeout,
                )
            finally:
                cleanup_teacher_cache(eval_cache_for_cleanup)
            print(f"[eval step={step}] {format_metric_summary(eval_metrics)}")
            _jsonl(metrics_path, {"event": "eval", "step": step, "time": time.time(), **eval_metrics})
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

        if args.sample_eval_interval > 0 and step % args.sample_eval_interval == 0:
            run_sample_eval(step)

        if args.teacher_sample_interval > 0 and step % args.teacher_sample_interval == 0:
            run_teacher_sample(step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_result = save_weights(
                args.train_url,
                args.train_model_id,
                f"step-{step:06d}",
                future_timeout=args.future_timeout,
            )
            print(f"[save step={step}] {save_result.get('path', save_result)}")
            _jsonl(metrics_path, {"event": "save", "step": step, "time": time.time(), "result": save_result})

    if args.dump_forward_backward_replay_only:
        final_save = {"skipped": True, "reason": "dump_forward_backward_replay_only"}
        print("[done] final checkpoint skipped for replay dump only")
    else:
        final_save = save_weights(args.train_url, args.train_model_id, "final", future_timeout=args.future_timeout)
        print(f"[done] final checkpoint: {final_save.get('path', final_save)}")
    print(f"[done] metrics: {metrics_path}")
    _jsonl(metrics_path, {"event": "done", "step": args.steps, "time": time.time(), "final_save": final_save})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
