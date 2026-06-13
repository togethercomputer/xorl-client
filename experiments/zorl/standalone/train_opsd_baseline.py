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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer
from transformers.utils import cached_file


IGNORE_INDEX = -100

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.zorl.standalone.tasks.base import Example, load_task  # noqa: E402


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
    urls: list[str] = []
    for value in values or []:
        urls.extend(part for part in str(value).split() if part)
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


def _asymmetric_prompt_ids(tokenizer, task, example: Example, *, hinted: bool) -> list[int]:
    if not hasattr(task, "SYSTEM_PROMPT"):
        raise RuntimeError("asymmetric_opsd currently requires the Wordle task SYSTEM_PROMPT")
    target = str(example.metadata["target"]).upper()
    messages = [{"role": "system", "content": task.SYSTEM_PROMPT}]
    if hinted:
        content = (
            "You are a teacher solving one Wordle game for distillation. "
            f"Private hint: the target word is {target}. "
            "Write a compact hidden rationale, then emit complete five-letter guesses in "
            "<guess>WORD</guess> tags that solve the game. Do not leave any guess tag empty."
        )
    else:
        content = (
            "You are solving one Wordle game. The target word is hidden from you. "
            "Write a compact hidden rationale, then emit at least two complete five-letter guesses in "
            "<guess>WORD</guess> tags. If uncertain, use common Wordle openers; do not leave any guess tag empty."
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


def _wordle_turn_prompt_ids(
    tokenizer,
    task,
    *,
    target: str,
    history: list[tuple[str, str]],
    hinted: bool,
) -> list[int]:
    if not hinted and hasattr(task, "_build_turn_input_ids"):
        return list(task._build_turn_input_ids(tokenizer, target=target, history=history))
    if not hasattr(task, "SYSTEM_PROMPT"):
        raise RuntimeError("multi-turn asymmetric_opsd requires task.SYSTEM_PROMPT")

    target_upper = target.upper()
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    messages = [{"role": "system", "content": task.SYSTEM_PROMPT}]
    if hinted:
        first_turn = (
            "Private teacher hint: the target word is "
            f"{target_upper}. Use the visible Wordle transcript exactly, then make your first guess."
        )
    else:
        first_turn = "Make your first guess."
    messages.append({"role": "user", "content": first_turn})

    for turn_idx, (guess, feedback) in enumerate(history):
        guess_upper = guess.upper()
        messages.append({"role": "assistant", "content": f"<guess>{guess_upper}</guess>"})
        remaining = max_turns - (turn_idx + 1)
        if guess == target or remaining == 0:
            messages.append({"role": "user", "content": f"Feedback: {feedback}. Game over."})
            continue
        content = (
            f"Feedback: {feedback} (for guess {guess_upper}). "
            f"You have {remaining} guess(es) left. Make your next guess."
        )
        if hinted:
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
    max_length: int,
    generation_log_path: Path,
) -> list[OpsdRow]:
    if not hasattr(task, "extract_guess") or not hasattr(task, "compute_feedback"):
        raise RuntimeError("multi-turn asymmetric_opsd requires extract_guess and compute_feedback")
    if not infer_urls:
        raise RuntimeError("asymmetric_opsd requires at least one --infer-url")
    if generation_batch_size <= 0:
        raise RuntimeError(f"generation_batch_size must be positive, got {generation_batch_size}")

    max_turns = int(getattr(task, "MAX_TURNS", 6))

    def build_one(item: tuple[int, Example]) -> tuple[list[OpsdRow], list[dict[str, Any]]]:
        idx, example = item
        infer_url = infer_urls[idx % len(infer_urls)]
        target = str(example.metadata["target"]).lower()
        history: list[tuple[str, str]] = []
        rows: list[OpsdRow] = []
        logs: list[dict[str, Any]] = []
        for turn in range(max_turns):
            student_prefix = _wordle_turn_prompt_ids(
                tokenizer,
                task,
                target=target,
                history=history,
                hinted=False,
            )
            teacher_prefix = _wordle_turn_prompt_ids(
                tokenizer,
                task,
                target=target,
                history=history,
                hinted=True,
            )
            result = generate_with_sglang(
                infer_url,
                input_ids=student_prefix,
                lora_name=lora_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
                stop=stop,
            )
            output_ids = _generated_output_ids(result, tokenizer)
            if not output_ids:
                raise RuntimeError(f"{example.project} turn={turn + 1}: SGLang returned no output tokens: {result}")

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
            weights = [1.0 if label != IGNORE_INDEX else 0.0 for label in student_labels]
            text = str(result.get("text") or tokenizer.decode(output_ids, skip_special_tokens=False))
            guess = task.extract_guess(text or "")
            feedback = ""
            solved = False
            row = OpsdRow(
                project=f"{example.project}:turn{turn + 1}",
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
            rows.append(row)

            if guess is not None and len(guess) == 5 and guess.isalpha():
                feedback = task.compute_feedback(guess, target)
                history.append((guess, feedback))
                solved = guess == target
            logs.append(
                {
                    "event": "student_multiturn_generation",
                    "time": time.time(),
                    "project": example.project,
                    "row_project": row.project,
                    "target": target,
                    "infer_url": infer_url,
                    "turn": turn + 1,
                    "history_before": list(history[:-1] if feedback else history),
                    "guess": guess or "",
                    "feedback": feedback,
                    "solved": solved,
                    "token_count": len(output_ids),
                    "student_prompt_tokens": len(student_prefix),
                    "teacher_prompt_tokens": len(teacher_prefix),
                    "text": text,
                }
            )
            if guess is None or len(guess) != 5 or not guess.isalpha() or solved:
                break
        return rows, logs

    max_workers = max(1, min(len(examples), len(infer_urls) * generation_batch_size))
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
                raise
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
        timeout=300.0,
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
) -> dict[str, Any]:
    return generate_batch_with_sglang(
        infer_url,
        input_ids_batch=[input_ids],
        lora_name=lora_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
        stop=stop,
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
) -> list[dict[str, Any]]:
    sampling: dict[str, Any] = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    result = _post_sglang(
        infer_url,
        "/generate",
        {
            "input_ids": [list(input_ids) for input_ids in input_ids_batch],
            "sampling_params": sampling,
            "return_logprob": False,
            "lora_path": [lora_name for _ in input_ids_batch],
        },
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


def _generated_output_ids(result: dict[str, Any], tokenizer) -> list[int]:
    output_ids = result.get("output_ids")
    if isinstance(output_ids, list) and output_ids:
        return [int(x) for x in output_ids]
    text = str(result.get("text") or "")
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


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
) -> str:
    if not infer_urls:
        raise RuntimeError("export_and_load_sampler requires at least one infer URL")
    save_result = save_weights_for_sampler(train_url, model_id, save_name, future_timeout=future_timeout)
    lora_path = _sampler_uri_to_path(str(save_result.get("path") or save_name), server_output_dir)
    if not lora_path.exists():
        raise RuntimeError(f"Sampler export did not appear at {lora_path}")
    load_results = []
    load_started = time.time()
    for infer_url in infer_urls:
        endpoint_started = time.time()
        unload_sglang_lora(infer_url, lora_name)
        load_sglang_lora(infer_url, lora_name=lora_name, lora_path=str(lora_path))
        load_results.append({"infer_url": infer_url, "load_time_s": time.time() - endpoint_started})
    _jsonl(
        output_dir / "sampler_exports.jsonl",
        {
            "time": time.time(),
            "model_id": model_id,
            "save_name": save_name,
            "save_result": save_result,
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
    max_length: int,
    generation_log_path: Path,
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
            max_length=max_length,
            generation_log_path=generation_log_path,
        )

    student_prefixes: list[list[int]] = []
    teacher_prefixes: list[list[int]] = []
    outputs_by_index: dict[int, dict[str, Any]] = {}
    indices_by_url: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        infer_url = infer_urls[idx % len(infer_urls)]
        indices_by_url.setdefault(infer_url, []).append(idx)
        student_prefixes.append(_asymmetric_prompt_ids(tokenizer, task, example, hinted=False))
        teacher_prefixes.append(_asymmetric_prompt_ids(tokenizer, task, example, hinted=True))

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

    if objective == "teacher_forced_ce":
        if include_cache:
            raise RuntimeError("teacher_forced_ce does not use teacher hidden caches")
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
    train_url: str,
    model_id: str,
    rows: list[OpsdRow],
    cache_path: Path,
    cache_dtype: str,
    objective: str,
    future_timeout: float,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result = call_future(
        train_url,
        "/api/v1/forward",
        {
            "model_id": model_id,
            "forward_input": {
                "data": [row_to_datum(row, include_cache=False, objective=objective, cache_view=True) for row in rows],
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
    if not metadata:
        raise RuntimeError(f"teacher_hidden_cache response did not include cache metadata: {result}")
    apply_cache_indices(rows, metadata.get("cache_indices_by_sample") or [])
    return metadata


def create_model(train_url: str, args) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": args.train_model_id,
        "base_model": args.model,
        "lora_config": {
            "rank": args.lora_rank,
            "lora_rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "lora_alpha": args.lora_alpha,
        },
        "optimizer_config": {
            "type": args.optimizer,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer_dtype": args.optimizer_dtype,
            "betas": [args.beta1, args.beta2],
            "eps": args.eps,
        },
        "zorl_config": {"enabled": False},
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


def chunks(rows: list[OpsdRow], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


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
) -> dict[str, float]:
    if not examples:
        return {"n": 0.0, "exact_count": 0.0, "exact_match_rate": 0.0}
    if not infer_urls:
        raise RuntimeError("sample eval requires at least one infer URL")
    if workers <= 0:
        raise RuntimeError(f"sample eval workers must be positive, got {workers}")

    multi_turn = bool(getattr(task, "is_multi_turn", False))
    rollout_args = SimpleNamespace(
        rollout_temperature=float(temperature),
        rollout_max_new_tokens=int(max_new_tokens),
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
    return metrics


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
    responses = []
    include_cache = objective in {"opd_self_kl", "asymmetric_opsd"}
    for chunk_idx, batch_rows in enumerate(chunks(rows, request_batch_size)):
        result = call_future(
            train_url,
            "/api/v1/forward_backward",
            {
                "model_id": model_id,
                "seq_id": chunk_idx,
                "forward_backward_input": {
                    "data": [row_to_datum(row, include_cache=include_cache, objective=objective) for row in batch_rows],
                    "loss_fn": loss_fn,
                    "loss_fn_params": loss_params,
                },
            },
            context=f"forward_backward(train chunk {chunk_idx})",
            future_timeout=future_timeout,
        )
        responses.append(result)

    metrics = summarize_responses(responses)
    opt_result = call_future(
        train_url,
        "/api/v1/optim_step",
        {
            "model_id": model_id,
            "seq_id": len(responses),
            "learning_rate": lr,
            "gradient_clip": gradient_clip,
        },
        context="optim_step",
        future_timeout=future_timeout,
    )
    opt_metrics = opt_result.get("metrics") or {}
    for key in ("grad_norm", "learning_rate"):
        value = opt_metrics.get(key)
        if _is_number(value):
            metrics[key] = float(value)
    return metrics


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
    parser.add_argument(
        "--objective",
        default="asymmetric_opsd",
        choices=["asymmetric_opsd", "teacher_forced_ce", "opd_self_kl"],
    )
    parser.add_argument(
        "--infer-url",
        action="append",
        default=[],
        help="SGLang /generate URL(s) for asymmetric_opsd student rollouts; may be repeated or space-separated.",
    )
    parser.add_argument("--server-output-dir", default="")
    parser.add_argument("--sampler-lora-name", default="")
    parser.add_argument("--sampler-save-prefix", default="")
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-max-new-tokens", type=int, default=96)
    parser.add_argument("--student-generation-batch-size", type=int, default=8)
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
    parser.add_argument("--keep-teacher-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=9234)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--eval-request-batch-size", type=int, default=8)
    parser.add_argument("--resample-train-each-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wordle-teacher-trace-style", default="hinted_cot")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
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
    sample_eval_path = output_dir / "sample_eval.jsonl"
    infer_urls = _split_urls(args.infer_url)
    active_infer_urls = infer_urls if args.objective == "asymmetric_opsd" else []
    student_stop = _split_urls(args.student_stop)
    sample_eval_stop = _split_urls(args.sample_eval_stop)
    if args.sample_eval_temperature is None:
        args.sample_eval_temperature = args.student_temperature
    server_output_dir = Path(args.server_output_dir) if args.server_output_dir else output_dir / "server_output"
    sampler_lora_name = args.sampler_lora_name or f"{args.train_model_id}-{output_dir.name}"
    sampler_save_prefix = args.sampler_save_prefix or f"{output_dir.name}/student"

    args.train_pool_size = int(args.train_pool_size or args.train_size)
    if args.train_pool_size < args.train_size:
        raise ValueError("--train-pool-size must be >= --train-size")
    if args.request_batch_size <= 0 or args.eval_request_batch_size <= 0:
        raise ValueError("request batch sizes must be positive")
    if args.max_runtime_seconds < 0:
        raise ValueError("--max-runtime-seconds must be non-negative")
    if args.sample_eval_interval > 0 and args.sample_eval_workers <= 0:
        raise ValueError("--sample-eval-workers must be positive when sampled eval is enabled")
    if args.objective == "asymmetric_opsd" and not active_infer_urls:
        raise ValueError("asymmetric_opsd requires at least one --infer-url")
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
    if args.objective != "asymmetric_opsd":
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
        resolve_model_dir(args.model, local_files_only=args.local_files_only) if needs_teacher_cache else None
    )
    train_cache_path = (
        output_dir / "teacher_cache" / "train_hidden.safetensors" if args.objective == "opd_self_kl" else None
    )
    eval_cache_path = (
        output_dir / "teacher_cache" / "eval_hidden.safetensors" if args.objective == "opd_self_kl" else None
    )

    run_config = {
        **vars(args),
        "infer_urls": infer_urls,
        "active_infer_urls": active_infer_urls,
        "student_stop": student_stop,
        "sample_eval_stop": sample_eval_stop,
        "server_output_dir": str(server_output_dir),
        "sampler_lora_name": sampler_lora_name,
        "sampler_save_prefix": sampler_save_prefix,
        "teacher_head_dir": teacher_head_dir,
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
        if len(infer_urls) > len(active_infer_urls):
            print(
                f"[init] asymmetric_opsd using one SGLang endpoint for this run: {active_infer_urls[0]} "
                f"(received {len(infer_urls)})"
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
            train_url=args.train_url,
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
            train_url=args.train_url,
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
    elif args.objective == "teacher_forced_ce":
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

    def cleanup_teacher_cache(cache_path: Path | None) -> None:
        if cache_path is not None and not args.keep_teacher_cache:
            cache_path.unlink(missing_ok=True)

    def ensure_sampler(policy_step: int) -> None:
        nonlocal loaded_policy_step
        if args.objective != "asymmetric_opsd":
            return
        if loaded_policy_step == policy_step:
            return
        save_name = f"{sampler_save_prefix}/policy-{policy_step:06d}"
        print(f"[sampler step={policy_step}] exporting native LoRA for SGLang: {save_name}")
        sampler_path = export_and_load_sampler(
            train_url=args.train_url,
            model_id=args.train_model_id,
            output_dir=output_dir,
            server_output_dir=server_output_dir,
            infer_urls=active_infer_urls,
            lora_name=sampler_lora_name,
            save_name=save_name,
            future_timeout=args.future_timeout,
        )
        print(
            f"[sampler step={policy_step}] loaded {sampler_lora_name} from {sampler_path} "
            f"on {len(active_infer_urls)} endpoint(s)"
        )
        loaded_policy_step = policy_step

    def selected_sample_eval_examples() -> list[Example]:
        if args.sample_eval_size <= 0 or args.sample_eval_size >= len(eval_examples):
            return list(eval_examples)
        return list(eval_examples[: args.sample_eval_size])

    def run_sample_eval(policy_step: int) -> None:
        if args.objective != "asymmetric_opsd" or args.sample_eval_interval <= 0:
            return
        examples = selected_sample_eval_examples()
        if not examples:
            return
        ensure_sampler(policy_step)
        started = time.time()
        metrics = sample_eval_student(
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
        )
        metrics["sample_eval_time_s"] = time.time() - started
        print(f"[sample_eval step={policy_step}] {format_sample_eval_summary(metrics)}")
        _jsonl(metrics_path, {"event": "sample_eval", "step": policy_step, "time": time.time(), **metrics})
        if wandb_run is not None:
            wandb_run.log({f"sample_eval/{k}": v for k, v in metrics.items()}, step=policy_step)

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
            max_length=args.max_length,
            generation_log_path=output_dir / "generations.jsonl",
        )
        rollout_time_s = time.time() - rollout_started
        rollout_tokens = sum(row.target_token_count for row in rows)
        print(
            f"[rollout {split} policy={policy_step}] rows={len(rows)} tokens={rollout_tokens} "
            f"samplers={len(active_infer_urls)} dt={rollout_time_s:.1f}s"
        )
        _jsonl(
            metrics_path,
            {
                "event": f"student_rollout_{split}",
                "step": policy_step,
                "time": time.time(),
                "rows": len(rows),
                "tokens": rollout_tokens,
                "samplers": len(active_infer_urls),
                "rollout_time_s": rollout_time_s,
            },
        )
        cache_path = (
            output_dir / "teacher_cache" / f"{split}_policy_{policy_step:06d}_{int(time.time() * 1000)}.safetensors"
        )
        print(f"[cache {split} policy={policy_step}] materializing hinted teacher cache: {cache_path}")
        cache_started = time.time()
        cache_meta = materialize_teacher_cache(
            train_url=args.train_url,
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
        if args.objective == "asymmetric_opsd":
            selected_rows, train_loss_params, train_cache_for_cleanup = make_asymmetric_rows(
                split="train",
                examples=selected_examples,
                policy_step=step - 1,
            )
        else:
            selected_rows = [row_by_project[example.project] for example in selected_examples]
        started = time.time()
        try:
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
        train_metrics["step_time_s"] = time.time() - started
        train_metrics["train_examples"] = float(len(selected_rows))

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

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_result = save_weights(
                args.train_url,
                args.train_model_id,
                f"step-{step:06d}",
                future_timeout=args.future_timeout,
            )
            print(f"[save step={step}] {save_result.get('path', save_result)}")
            _jsonl(metrics_path, {"event": "save", "step": step, "time": time.time(), "result": save_result})

    final_save = save_weights(args.train_url, args.train_model_id, "final", future_timeout=args.future_timeout)
    print(f"[done] final checkpoint: {final_save.get('path', final_save)}")
    print(f"[done] metrics: {metrics_path}")
    _jsonl(metrics_path, {"event": "done", "step": args.steps, "time": time.time(), "final_save": final_save})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
