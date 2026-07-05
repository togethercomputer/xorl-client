"""CPU tests for the marin_math ZORL task (no cluster, no HF downloads).

Covers: task-registry wiring, prompt rendering (forced-thinking prefill golden),
the held-out split, and the reward semantics — asserted BOTH directly and
against marin's own length_penalty/verifier functions so any drift from the
GRPO reference reward is caught here.

Run from the repo root: PYTHONPATH=. python -m pytest experiments/zorl/standalone/tests
"""

from __future__ import annotations

import math
import os

import pytest

from experiments.marin.standalone.dataset import MathExample
from experiments.marin.standalone.length_penalty import LengthPenaltyConfig, length_penalty_reward
from experiments.marin.standalone.prompts import FORCED_THINK_PREFILL, QUESTION_SUFFIX

from tasks import marin_math
from tasks.base import Example, load_task


class ChatTemplateTokenizer:
    """Mirrors marin's tests/test_prompts.py fixture: 1 char = 1 token."""

    chat_template = "dummy"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<bos><user>{messages[0]['content']}<assistant>"


def _mex(prompt: str, gold: str, idx: int = 0) -> MathExample:
    return MathExample(prompt=prompt, gold_answer=gold, source="rlvr_math_7500", example_id=f"ex{idx}")


def _example(gold: str = "42") -> Example:
    return Example(project="t", prompt_ids=[], metadata={"gold_answer": gold})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        marin_math.ENV_MAX_COMPLETION_TOKENS,
        marin_math.ENV_TARGET_LENGTH,
        marin_math.ENV_LPW,
        marin_math.ENV_MAX_PROMPT_TOKENS,
        marin_math.ENV_DATASET,
        marin_math.ENV_EVAL_DATASET,
        marin_math.ENV_EVAL_SEED,
    ):
        monkeypatch.delenv(key, raising=False)
    # Tests below drive the text-only fallback deliberately.
    monkeypatch.setattr(marin_math, "_TOKENIZER", None)


# ---------------------------------------------------------------------------
# Registry / driver wiring
# ---------------------------------------------------------------------------


def test_load_task_registry() -> None:
    task = load_task("marin_math")
    assert task is marin_math
    assert task.is_multi_turn is False
    assert callable(task.build_examples)
    assert callable(task.score_completion)
    assert callable(task.score_result)  # meta-aware hook consumed by zorl_client


def test_driver_parser_accepts_marin_math() -> None:
    import importlib.util
    from pathlib import Path

    driver_path = Path(__file__).resolve().parents[1] / "run_wordle_zorl_xorl_ps.py"
    spec = importlib.util.spec_from_file_location("zorl_ps_driver_under_test", str(driver_path))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    args = driver.build_parser().parse_args(
        [
            "--task", "marin_math",
            "--model", "/tmp/model",
            "--infer-url", "http://scorer:30000",
            "--xorl-ps-url", "http://ps:26070",
            "--lora-target-modules", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "--perturbation-mode", "fresh_ab",
            "--sync-quantization", "bf16",
            "--dry-run",
        ]
    )
    assert args.task == "marin_math"
    assert args.dry_run is True
    assert args.sync_quantization == "bf16"
    assert args.lora_target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_prompt_rendering_golden() -> None:
    tokenizer = ChatTemplateTokenizer()
    train, eval_ = marin_math.build_examples_from_math_examples(
        [_mex("What is 6*7?", "42")],
        tokenizer=tokenizer,
        train_size=8,
        eval_size=0,
        eval_seed=777,
    )
    assert eval_ == []
    assert len(train) == 1
    expected_text = f"<bos><user>What is 6*7?{QUESTION_SUFFIX}<assistant>{FORCED_THINK_PREFILL}"
    assert train[0].prompt_ids == [ord(c) for c in expected_text]
    assert train[0].metadata["gold_answer"] == "42"
    assert train[0].metadata["prompt_tokens"] == len(expected_text)


def test_prompt_overlength_skipped() -> None:
    tokenizer = ChatTemplateTokenizer()
    train, _ = marin_math.build_examples_from_math_examples(
        [_mex("x" * 5000, "1", idx=0), _mex("short?", "2", idx=1)],
        tokenizer=tokenizer,
        train_size=8,
        eval_size=0,
        eval_seed=777,
        max_prompt_tokens=512,
    )
    assert len(train) == 1
    assert train[0].metadata["gold_answer"] == "2"


def test_holdout_split_is_deterministic_and_disjoint() -> None:
    tokenizer = ChatTemplateTokenizer()
    pool = [_mex(f"q{i}?", str(i), idx=i) for i in range(50)]
    train_a, eval_a = marin_math.build_examples_from_math_examples(
        pool, tokenizer=tokenizer, train_size=100, eval_size=10, eval_seed=777
    )
    train_b, eval_b = marin_math.build_examples_from_math_examples(
        pool, tokenizer=tokenizer, train_size=100, eval_size=10, eval_seed=777
    )
    assert [e.metadata["gold_answer"] for e in eval_a] == [e.metadata["gold_answer"] for e in eval_b]
    assert len(eval_a) == 10 and len(train_a) == 40
    train_golds = {e.metadata["gold_answer"] for e in train_a}
    eval_golds = {e.metadata["gold_answer"] for e in eval_a}
    assert not (train_golds & eval_golds)
    # A different split seed moves the held-out set.
    _, eval_c = marin_math.build_examples_from_math_examples(
        pool, tokenizer=tokenizer, train_size=100, eval_size=10, eval_seed=778
    )
    assert [e.metadata["gold_answer"] for e in eval_c] != [e.metadata["gold_answer"] for e in eval_a]


# ---------------------------------------------------------------------------
# Reward semantics (must equal marin's GRPO scoring)
# ---------------------------------------------------------------------------


# >= min_response_length(16) and <= target_length(768) tokens -> full reward.
_REASONING = "step " * 30


def test_correct_boxed_short_is_full_reward() -> None:
    blob = marin_math.score_completion(_example(), f"{_REASONING}\nAnswer: \\boxed{{42}}")
    assert blob["reward"] == 1.0
    assert blob["exact_match"] == 1.0
    assert blob["has_box"] == 1.0
    assert blob["truncated"] == 0.0


def test_correct_reference_answer_without_box() -> None:
    blob = marin_math.score_completion(_example(), f"{_REASONING}\nAnswer: 42")
    assert blob["reward"] == 1.0
    assert blob["exact_match"] == 1.0
    assert blob["has_box"] == 0.0


def test_suspiciously_short_correct_answer_gets_no_length_bonus() -> None:
    # marin's min_response_length=16 clause: a correct answer in <16 tokens
    # scores 1.0 - lpw = 0.0 (SkyRL reference semantics, kept identical).
    blob = marin_math.score_completion(_example(), "Answer: \\boxed{42}")
    assert blob["reward"] == 0.0
    assert blob["exact_match"] == 1.0


def test_wrong_complete_answer_is_minus_one() -> None:
    blob = marin_math.score_completion(_example(), "Answer: \\boxed{41}")
    assert blob["reward"] == -1.0
    assert blob["exact_match"] == 0.0


def test_unparseable_completion_is_minus_one() -> None:
    blob = marin_math.score_completion(_example(), "I have no idea, sorry!")
    assert blob["reward"] == -1.0
    assert blob["exact_match"] == 0.0
    assert blob["has_box"] == 0.0


def test_equivalent_fraction_verifies() -> None:
    blob = marin_math.score_completion(_example(gold="1/2"), "Answer: \\boxed{0.5}")
    assert blob["exact_match"] == 1.0


def test_truncated_reward_is_minus_two() -> None:
    result = {
        "text": "endless reasoning that never boxes an answer",
        "meta_info": {"completion_tokens": 3584, "finish_reason": {"type": "length"}},
    }
    blob = marin_math.score_result(_example(), result)
    assert blob["reward"] == -2.0
    assert blob["truncated"] == 1.0


def test_truncation_via_token_count_without_finish_reason() -> None:
    result = {"text": "x", "meta_info": {"completion_tokens": 3584}}
    blob = marin_math.score_result(_example(), result)
    assert blob["reward"] == -2.0


def test_correct_long_answer_matches_marin_cosine_ramp() -> None:
    # 2176 tokens: length_frac = (2176-768)/(3584-768) = 0.5 -> reward 0.5.
    result = {
        "text": "long reasoning...\nAnswer: \\boxed{42}",
        "meta_info": {"completion_tokens": 2176, "finish_reason": {"type": "stop"}},
    }
    blob = marin_math.score_result(_example(), result)
    assert blob["reward"] == pytest.approx(0.5)
    reference = length_penalty_reward(
        correct=True,
        completion_tokens=2176,
        has_box=True,
        truncated=False,
        config=LengthPenaltyConfig(max_completion_tokens=3584, lpw=1.0),
    )
    assert blob["reward"] == pytest.approx(reference)


def test_reward_parity_with_marin_reference_across_lengths() -> None:
    config = marin_math.length_config()
    for tokens in (1, 16, 512, 768, 1024, 2000, 3000, 3583):
        result = {
            "text": "Answer: \\boxed{42}",
            "meta_info": {"completion_tokens": tokens, "finish_reason": {"type": "stop"}},
        }
        blob = marin_math.score_result(_example(), result)
        reference = length_penalty_reward(
            correct=True, completion_tokens=tokens, has_box=True, truncated=False, config=config
        )
        assert blob["reward"] == pytest.approx(reference), tokens


def test_reward_range() -> None:
    cases = [
        marin_math.score_completion(_example(), "Answer: \\boxed{42}"),
        marin_math.score_completion(_example(), "Answer: \\boxed{0}"),
        marin_math.score_completion(_example(), "garbage"),
        marin_math.score_result(
            _example(), {"text": "x", "meta_info": {"completion_tokens": 9999, "finish_reason": {"type": "length"}}}
        ),
    ]
    for blob in cases:
        assert -2.0 <= blob["reward"] <= 1.0
        assert math.isfinite(blob["reward"])


def test_score_result_falls_back_without_meta(monkeypatch) -> None:
    blob = marin_math.score_result(_example(), {"text": f"{_REASONING}\nAnswer: \\boxed{{42}}"})
    assert blob["reward"] == 1.0


def test_env_overrides_length_config(monkeypatch) -> None:
    monkeypatch.setenv(marin_math.ENV_MAX_COMPLETION_TOKENS, "100")
    result = {"text": "Answer: \\boxed{42}", "meta_info": {"completion_tokens": 100, "finish_reason": {"type": "stop"}}}
    blob = marin_math.score_result(_example(), result)
    assert blob["reward"] == -2.0  # >= max_completion_tokens counts as truncated
