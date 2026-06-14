"""GSM8K task: solve grade-school math word problems. Verifiable reward by
matching the integer/float answer extracted from the model's completion
against the dataset's ground-truth ``#### N`` final-answer marker.

Loader pulls ``openai/gsm8k`` via Hugging Face Datasets. Cache lives at
``HF_HOME`` (typically /shared/huggingface). The dataset has 7473 train +
1319 test rows; we deterministically slice the requested train/eval sizes.

Prompt format is the simple "Question: ... Answer:" pattern; we let the
model emit reasoning followed by ``#### N``. The reward extractor scans
the completion for the LAST integer or float — robust to chain-of-thought
preamble. Matches researcher's blog setup.

Default train size = 8 (matches the researcher's setup); eval = 128.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .base import Example


is_multi_turn = False


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, then output "
    "your final numeric answer on the last line preceded by '#### '. Show your "
    "reasoning but keep it concise."
)


# Reuse a tiny few-shot exemplar to anchor the answer format. Picked from the
# canonical GSM8K format; the LAST line is what the reward extractor reads.
FEWSHOT = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 each. How much does she make per day?",
        "answer": "She has 16 - 3 - 4 = 9 eggs left to sell.\nShe makes 9 * 2 = 18 dollars per day.\n#### 18",
    },
]


# Final-answer extractor: pull the LAST integer or decimal number from the
# completion. Negative numbers handled. Comma-thousands separators stripped
# before matching. Matches the standard GSM8K eval convention.
_FINAL_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_GROUND_TRUTH_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
ENV_HINTS_PATH = "ZORL_GSM8K_HINTS_PATH"
ENV_REWARD_MODE = "ZORL_GSM8K_REWARD_MODE"


def _extract_ground_truth(answer_text: str) -> float | None:
    """Pull the number after '#### ' from the dataset's gold answer."""
    m = _GROUND_TRUTH_RE.search(answer_text.replace(",", ""))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_predicted(text: str) -> float | None:
    """Pull the model's predicted answer. Prefer the number after '#### ' if
    present (matches dataset convention). Fall back to the last bare number
    in the completion — common when the model omits the marker."""
    text = (text or "").replace(",", "")
    m = _GROUND_TRUTH_RE.search(text)
    if m is not None:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    matches = _FINAL_NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _load_hints(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    hint_path = Path(path)
    if not hint_path.exists():
        raise FileNotFoundError(f"{ENV_HINTS_PATH} points at missing file: {hint_path}")
    if hint_path.suffix == ".jsonl":
        records = [json.loads(line) for line in hint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(hint_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            records = [
                {"id": key, **value} if isinstance(value, dict) else {"id": key, "hint": value}
                for key, value in raw.items()
            ]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(f"{hint_path}: expected JSON object, list, or JSONL records")

    hints: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        hint = record.get("hint") or record.get("teacher_hint") or record.get("rationale")
        key = record.get("project") or record.get("id") or record.get("question")
        if key and hint:
            hints[str(key)] = str(hint).strip()
    return hints


def _build_user_prompt(question: str, *, hint: str | None = None) -> str:
    """Compose the user turn with a single few-shot exemplar. Keeps the
    prompt small so 1.7B-class models don't drown in context."""
    shots = []
    for ex in FEWSHOT:
        shots.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    hint_text = f"\nTeacher hint: {hint.strip()}" if hint else ""
    shots.append(f"Question: {question}{hint_text}\nAnswer:")
    return "\n\n".join(shots)


def build_examples(tokenizer, *, train_size: int = 8, eval_size: int = 128, seed: int = 0):
    """Load GSM8K, take ``train_size`` from train + ``eval_size`` from test.
    Deterministic slicing (no shuffle) so reruns hit the same examples."""
    from datasets import load_dataset  # noqa: PLC0415

    train_split = load_dataset("openai/gsm8k", "main", split="train")
    test_split = load_dataset("openai/gsm8k", "main", split="test")
    hints = _load_hints(os.environ.get(ENV_HINTS_PATH))

    def to_example(row, project_id: str) -> Example:
        question = row["question"].strip()
        gold = _extract_ground_truth(row["answer"])
        if gold is None:
            raise RuntimeError(f"GSM8K row {project_id} has no parseable ground-truth answer")
        hint = hints.get(project_id) or hints.get(question)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, hint=hint)},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        return Example(
            project=project_id,
            prompt_ids=prompt_ids,
            metadata={"question": question, "answer_text": row["answer"], "gold": gold, "hinted": bool(hint)},
        )

    train = [to_example(train_split[i], f"gsm8k_train_{i:04d}") for i in range(min(train_size, len(train_split)))]
    eval_ = [to_example(test_split[i], f"gsm8k_test_{i:04d}") for i in range(min(eval_size, len(test_split)))]
    return train, eval_


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Binary reward: 1.0 iff the predicted number matches the gold within 1e-3.
    Also emits ``has_marker`` (did the model use the '####' format) and
    ``has_number`` (did it produce any number at all) for diagnostic logging."""
    pred = _extract_predicted(generated_text)
    gold = example.metadata["gold"]
    has_marker = float(bool(_GROUND_TRUTH_RE.search((generated_text or "").replace(",", ""))))
    has_number = float(pred is not None)
    if pred is None or gold is None:
        return {
            "reward": 0.0,
            "exact_match": 0.0,
            "has_marker": has_marker,
            "has_number": has_number,
            "numeric_closeness": 0.0,
        }
    correct = float(abs(pred - gold) < 1e-3)
    numeric_closeness = 1.0 / (1.0 + math.log1p(abs(pred - gold)))
    reward_mode = os.environ.get(ENV_REWARD_MODE, "exact").strip().lower()
    if reward_mode == "numeric_shaped":
        reward = max(correct, 0.1 * has_number + 0.9 * numeric_closeness)
    elif reward_mode in {"", "exact"}:
        reward = correct
    else:
        raise ValueError(f"{ENV_REWARD_MODE}={reward_mode!r} must be 'exact' or 'numeric_shaped'")
    return {
        "reward": reward,
        "exact_match": correct,
        "has_marker": has_marker,
        "has_number": has_number,
        "numeric_closeness": numeric_closeness,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running gsm8k sanity checks...")
    ex = Example(project="t", prompt_ids=[], metadata={"question": "q", "answer_text": "junk #### 42", "gold": 42.0})
    # Model emits the gold answer in '#### N' format
    assert score_completion(ex, "Let me think.\n10 + 32 = 42.\n#### 42")["reward"] == 1.0
    # Model emits just the number at the end
    assert score_completion(ex, "Reasoning... and the answer is 42")["reward"] == 1.0
    # Wrong answer
    assert score_completion(ex, "#### 41")["reward"] == 0.0
    # Junk
    r = score_completion(ex, "I don't know")
    assert r["reward"] == 0.0 and r["has_number"] == 0.0
    # Float gold
    ex2 = Example(project="t", prompt_ids=[], metadata={"question": "q", "answer_text": "junk #### 1.5", "gold": 1.5})
    assert score_completion(ex2, "#### 1.5")["reward"] == 1.0
    assert score_completion(ex2, "1.5")["reward"] == 1.0
    # Marker takes priority over later bare number
    assert score_completion(ex, "#### 42\n(don't pick 100)")["reward"] == 1.0
    # Comma thousands separators
    ex3 = Example(project="t", prompt_ids=[], metadata={"question": "q", "answer_text": "junk #### 1200", "gold": 1200.0})
    assert score_completion(ex3, "answer is 1,200")["reward"] == 1.0
    # Negative number
    ex4 = Example(project="t", prompt_ids=[], metadata={"question": "q", "answer_text": "junk #### -5", "gold": -5.0})
    assert score_completion(ex4, "#### -5")["reward"] == 1.0
    print("OK — gsm8k sanity passed")
