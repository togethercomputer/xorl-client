from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset

from experiments.marin.standalone.tasks.verifier import extract_last_boxed


DEFAULT_RLVR_MATH_DATASET = "allenai/RLVR-MATH"


@dataclass(frozen=True)
class MathExample:
    prompt: str
    gold_answer: str
    source: str
    example_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str
    config: str | None = None
    local_jsonl: Path | None = None


DATASET_SPECS = {
    "rlvr_math_7500": DatasetSpec(
        name=os.environ.get("MARIN_RLVR_MATH_DATASET", DEFAULT_RLVR_MATH_DATASET),
        split="train",
    ),
    "math500": DatasetSpec(name="HuggingFaceH4/MATH-500", split="test", config="default"),
    "aime24": DatasetSpec(
        name="evalchemy/AIME24",
        split="test",
        local_jsonl=Path(__file__).with_name("data") / "aime24_evalchemy.jsonl",
    ),
    "gsm8k": DatasetSpec(name="openai/gsm8k", split="test", config="main"),
}


def load_examples(
    dataset_key: str = "rlvr_math_7500",
    *,
    split: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
) -> list[MathExample] | Iterator[MathExample]:
    if dataset_key not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset key {dataset_key!r}; known keys: {sorted(DATASET_SPECS)}")

    spec = DATASET_SPECS[dataset_key]
    if spec.local_jsonl is not None:
        examples = iter_examples(_iter_jsonl_rows(spec.local_jsonl), source=dataset_key)
        if streaming:
            return _limit_iter(examples, limit)
        if limit is None:
            return list(examples)
        return [example for _, example in zip(range(limit), examples, strict=False)]

    load_kwargs: dict[str, Any] = {"split": split or spec.split, "streaming": streaming}
    if spec.config is not None:
        load_kwargs["name"] = spec.config
    dataset = load_dataset(spec.name, **load_kwargs)
    examples = iter_examples(dataset, source=dataset_key)
    if streaming:
        return _limit_iter(examples, limit)
    if limit is None:
        return list(examples)
    return [example for _, example in zip(range(limit), examples, strict=False)]


def iter_examples(rows: Iterable[Mapping[str, Any]], *, source: str) -> Iterator[MathExample]:
    for index, row in enumerate(rows):
        try:
            yield row_to_math_example(row, source=source, index=index)
        except ValueError as exc:
            if source == "rlvr_math_7500" and str(exc).startswith("Could not find gold answer field"):
                continue
            raise


def _iter_jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_to_math_example(row: Mapping[str, Any], *, source: str, index: int = 0) -> MathExample:
    prompt = _extract_prompt(row)
    raw_prompt = prompt
    if source == "rlvr_math_7500":
        prompt = _extract_final_question(prompt)
    gold_answer = _extract_gold_answer(row, source=source)
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"messages", "problem", "Problem", "question", "Question", "answer", "Answer"}
    }
    if source == "rlvr_math_7500" and prompt != raw_prompt:
        metadata = dict(metadata)
        metadata["raw_prompt"] = raw_prompt
    return MathExample(
        prompt=prompt,
        gold_answer=gold_answer,
        source=source,
        example_id=str(row.get("id") or row.get("example_id") or f"{source}_{index}"),
        metadata=metadata,
    )


def _extract_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        user_messages = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        if user_messages:
            return user_messages[-1]
    for key in ("prompt", "problem", "Problem", "question", "Question", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Could not find prompt field in row keys={sorted(row.keys())}")


def _extract_final_question(prompt: str) -> str:
    marker = "Question:"
    if marker not in prompt:
        return prompt.strip()
    return f"{marker} {prompt.rsplit(marker, 1)[1].strip()}"


def _extract_gold_answer(row: Mapping[str, Any], *, source: str) -> str:
    for key in ("ground_truth", "expected_answer", "answer", "Answer", "final_answer", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            if source == "gsm8k" and "####" in value:
                return value.rsplit("####", 1)[1].strip()
            return value.strip()
        if value is not None and key in {"expected_answer", "answer", "Answer", "final_answer", "target"}:
            return str(value).strip()
    solution = row.get("solution")
    if isinstance(solution, str) and solution.strip():
        return extract_last_boxed(solution, prefer_after_answer=False).strip()
    raise ValueError(f"Could not find gold answer field in row keys={sorted(row.keys())}")


def _limit_iter(examples: Iterator[MathExample], limit: int | None) -> Iterator[MathExample]:
    if limit is None:
        yield from examples
        return
    for index, example in enumerate(examples):
        if index >= limit:
            break
        yield example
