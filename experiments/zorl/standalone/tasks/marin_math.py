"""marin_math: single-turn ES task over the Marin #6279 RLVR-MATH-7500 math-RL setup.

This is the ZORL/ES counterpart of the PROVEN GRPO repro in
``experiments/marin/standalone`` (dense Qwen3 ~9.7B ``delphi-...-wc386k_lr1e5-sft``
checkpoint; reference trajectory: repro last-10 mean +0.436 vs GRPO reference
+0.247). To keep the ES reward curve directly comparable against that GRPO
reference, this module IMPORTS marin's own pieces rather than re-implementing
them:

  * dataset     experiments.marin.standalone.dataset.load_examples("rlvr_math_7500")
  * prompts     experiments.marin.standalone.prompts.encode_forced_thinking_prefix
                (delphi chat template + forced-thinking ``<|start_think|>\\n`` prefill,
                max_prompt_tokens=512 skip-overlength policy — same as the GRPO run)
  * verifier    experiments.marin.standalone.tasks.verifier.grade_reference_final_answer
                (the Answer:/\\boxed tail grader the GRPO run trained on)
  * shaping     experiments.marin.standalone.length_penalty.shaped_reward with
                LengthPenaltyConfig(max_completion_tokens=3584, lpw=1.0)

Reward semantics are IDENTICAL to train_marin_grpo.py's rollout scoring
(_score-side of _rollout_step): truncated -> -2.0, wrong-complete -> -1.0,
correct -> cosine length ramp in [0, 1] (<=768 tokens -> 1.0). Range [-2, 1].

Fidelity note (the ONE mechanical difference vs the GRPO driver): the GRPO
driver reads completion token counts + stop_reason from the sampler response.
The ZORL harness's preferred hook here is ``score_result(example, result)``
(wired in zorl_client.score_candidates/probe_parent), which reads the same
``meta_info.completion_tokens`` / ``meta_info.finish_reason`` from SGLang —
exact parity. The text-only ``score_completion`` fallback re-encodes the
completion with the build-time tokenizer to estimate token counts (+-1-2
tokens on the smooth cosine ramp; truncation then falls back to
count >= max_completion_tokens).

Train/eval split: the GRPO repro trained on the full RLVR-MATH-7500 train
split (its evals were external suites). For the ES held-out probe we reserve
``eval_size`` examples with a FIXED split seed (ZORL_MARIN_EVAL_SEED, default
777 — mirroring wordle's seed-777 held-out protocol) so the held-out set is
stable across runs and never intersects the train pool. Set
ZORL_MARIN_EVAL_DATASET=math500 to probe on MATH-500 instead (no train
exclusion needed; both are cached under /shared/huggingface).

Env knobs (all optional):
  ZORL_MARIN_MAX_COMPLETION_TOKENS  length-penalty budget (default 3584).
                                    MUST equal --rollout-max-new-tokens or the
                                    truncation cliff diverges from the request.
  ZORL_MARIN_TARGET_LENGTH          cosine ramp target (default 768)
  ZORL_MARIN_LPW                    length-penalty weight (default 1.0 = GRPO run)
  ZORL_MARIN_MAX_PROMPT_TOKENS      overlength-skip cap (default 512 = GRPO run)
  ZORL_MARIN_DATASET                train dataset key (default rlvr_math_7500)
  ZORL_MARIN_EVAL_DATASET           held-out source ('' = reserved train slice,
                                    or math500 / gsm8k / aime24)
  ZORL_MARIN_EVAL_SEED              split seed for the reserved slice (default 777)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .base import Example


is_multi_turn = False

# --------------------------------------------------------------------------
# Import the marin repro modules (they live under experiments/marin/standalone
# in this same repo). The standalone ZORL client puts its own directory on
# sys.path; the repo root may or may not be there depending on the launcher,
# so bootstrap it from our own location.
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
if not (_REPO_ROOT / "experiments" / "marin" / "standalone").is_dir():  # pragma: no cover
    raise ImportError(
        f"marin_math task expects the marin repro at {_REPO_ROOT}/experiments/marin/standalone"
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.marin.standalone.dataset import MathExample, load_examples  # noqa: E402
from experiments.marin.standalone.length_penalty import LengthPenaltyConfig, shaped_reward  # noqa: E402
from experiments.marin.standalone.prompts import encode_forced_thinking_prefix  # noqa: E402
from experiments.marin.standalone.tasks.verifier import grade_reference_final_answer  # noqa: E402


ENV_MAX_COMPLETION_TOKENS = "ZORL_MARIN_MAX_COMPLETION_TOKENS"
ENV_TARGET_LENGTH = "ZORL_MARIN_TARGET_LENGTH"
ENV_LPW = "ZORL_MARIN_LPW"
ENV_MAX_PROMPT_TOKENS = "ZORL_MARIN_MAX_PROMPT_TOKENS"
ENV_DATASET = "ZORL_MARIN_DATASET"
ENV_EVAL_DATASET = "ZORL_MARIN_EVAL_DATASET"
ENV_EVAL_SEED = "ZORL_MARIN_EVAL_SEED"

# Tokenizer captured at build time for the text-only score_completion fallback
# (score_result gets exact counts from SGLang meta_info and doesn't need it).
_TOKENIZER = None


def length_config() -> LengthPenaltyConfig:
    return LengthPenaltyConfig(
        max_completion_tokens=int(os.environ.get(ENV_MAX_COMPLETION_TOKENS, "3584")),
        lpw=float(os.environ.get(ENV_LPW, "1.0")),
        target_length=int(os.environ.get(ENV_TARGET_LENGTH, "768")),
    )


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------


def _to_example(math_example: MathExample, *, tokenizer, max_prompt_tokens: int, split: str, index: int) -> Example | None:
    """Render one MathExample with the GRPO run's forced-thinking prefix.

    Returns None when the rendered prompt exceeds max_prompt_tokens (the GRPO
    run's prompt_overlength_policy=skip)."""
    prefix = encode_forced_thinking_prefix(
        tokenizer,
        math_example.prompt,
        max_prompt_tokens=max_prompt_tokens,
        truncate_to_max=False,
    )
    if prefix.skipped_for_length:
        return None
    return Example(
        project=f"marin_{split}_{index:05d}_{math_example.example_id}",
        prompt_ids=list(prefix.token_ids),
        metadata={
            "gold_answer": math_example.gold_answer,
            "source": math_example.source,
            "example_id": math_example.example_id,
            "prompt": math_example.prompt,
            "prompt_tokens": len(prefix.token_ids),
        },
    )


def build_examples_from_math_examples(
    math_examples: list[MathExample],
    *,
    tokenizer,
    train_size: int,
    eval_size: int,
    eval_examples_override: list[MathExample] | None = None,
    eval_seed: int = 777,
    max_prompt_tokens: int | None = None,
) -> tuple[list[Example], list[Example]]:
    """Pure split/render core (unit-testable without HF datasets).

    When ``eval_examples_override`` is None, ``eval_size`` examples are
    reserved from the train list with a fixed ``eval_seed`` shuffle (held-out
    protocol); otherwise the override list becomes the eval set and the whole
    train list stays trainable.
    """
    import random  # noqa: PLC0415

    cap = int(max_prompt_tokens or os.environ.get(ENV_MAX_PROMPT_TOKENS, "512"))

    if eval_examples_override is not None:
        train_src = list(math_examples)
        eval_src = list(eval_examples_override)[: eval_size or None]
    else:
        indices = list(range(len(math_examples)))
        random.Random(int(eval_seed)).shuffle(indices)
        eval_idx = set(indices[: max(0, int(eval_size))])
        eval_src = [math_examples[i] for i in sorted(eval_idx)]
        train_src = [ex for i, ex in enumerate(math_examples) if i not in eval_idx]

    train: list[Example] = []
    for i, mex in enumerate(train_src):
        if len(train) >= train_size:
            break
        ex = _to_example(mex, tokenizer=tokenizer, max_prompt_tokens=cap, split="train", index=i)
        if ex is not None:
            train.append(ex)

    eval_: list[Example] = []
    for i, mex in enumerate(eval_src):
        ex = _to_example(mex, tokenizer=tokenizer, max_prompt_tokens=cap, split="eval", index=i)
        if ex is not None:
            eval_.append(ex)

    if not train:
        raise RuntimeError("marin_math: no train examples fit the prompt-length cap")
    return train, eval_


def build_examples(tokenizer, *, train_size: int = 32, eval_size: int = 128, seed: int = 0):
    """Load RLVR-MATH-7500 (offline HF cache) and build train/eval Example lists.

    ``seed`` (the run's --seed) intentionally does NOT move the held-out split;
    the split uses ZORL_MARIN_EVAL_SEED (default 777) so every run holds out
    the same examples — the wordle seed-777 protocol.
    """
    global _TOKENIZER
    _TOKENIZER = tokenizer
    _ = seed  # per-step train selection is the driver's job (_select_train_examples_for_step)

    dataset_key = os.environ.get(ENV_DATASET, "rlvr_math_7500")
    eval_dataset_key = os.environ.get(ENV_EVAL_DATASET, "").strip()
    eval_seed = int(os.environ.get(ENV_EVAL_SEED, "777"))

    math_examples = list(load_examples(dataset_key))
    eval_override = None
    if eval_dataset_key:
        eval_override = list(load_examples(eval_dataset_key, limit=max(1, eval_size)))

    train, eval_ = build_examples_from_math_examples(
        math_examples,
        tokenizer=tokenizer,
        train_size=train_size,
        eval_size=eval_size,
        eval_examples_override=eval_override,
        eval_seed=eval_seed,
    )
    print(
        f"[marin_math] dataset={dataset_key} eval={eval_dataset_key or f'reserved slice (seed {eval_seed})'} "
        f"train={len(train)} eval={len(eval_)} max_prompt_tokens={os.environ.get(ENV_MAX_PROMPT_TOKENS, '512')} "
        f"length_config={length_config()}",
        flush=True,
    )
    return train, eval_


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _score(example: Example, text: str, *, completion_tokens: int, stop_reason: str | None) -> dict[str, float]:
    """Exact train_marin_grpo.py rollout scoring (verifier + length shaping)."""
    config = length_config()
    truncated = (stop_reason == "length") or (completion_tokens >= config.max_completion_tokens)
    verification = grade_reference_final_answer(text or "", str(example.metadata["gold_answer"]))
    reward = shaped_reward(
        verifier_reward=verification.reward,
        correct=verification.correct,
        completion_tokens=completion_tokens,
        has_box=verification.has_box,
        truncated=truncated,
        config=config,
    )
    return {
        "reward": float(reward),
        "exact_match": 1.0 if verification.correct else 0.0,
        "verifier_reward": float(verification.reward),
        "has_box": 1.0 if verification.has_box else 0.0,
        "truncated": 1.0 if truncated else 0.0,
        "completion_tokens": float(completion_tokens),
    }


def _finish_reason_type(result: dict) -> str | None:
    """SGLang meta_info.finish_reason is {'type': 'length'|'stop', ...} (or a bare
    string on older builds) — same tolerant read as train_marin_grpo.py."""
    meta = result.get("meta_info") or {}
    finish = meta.get("finish_reason") or result.get("finish_reason") or {}
    if isinstance(finish, dict):
        finish = finish.get("type")
    return str(finish) if finish else None


def score_result(example: Example, result: dict) -> dict[str, float]:
    """Preferred scoring hook: reads exact completion_tokens/finish_reason from
    the SGLang /generate result (GRPO-parity path)."""
    text = result.get("text", "") or ""
    meta = result.get("meta_info") or {}
    completion_tokens = meta.get("completion_tokens")
    if completion_tokens is None:
        return score_completion(example, text)
    return _score(
        example,
        text,
        completion_tokens=int(completion_tokens),
        stop_reason=_finish_reason_type(result),
    )


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Text-only fallback: token count via re-encoding with the build tokenizer."""
    text = generated_text or ""
    if _TOKENIZER is not None:
        completion_tokens = len(_TOKENIZER.encode(text, add_special_tokens=False))
    else:  # pure-text unit tests without a tokenizer: whitespace-token proxy
        completion_tokens = len(text.split())
    return _score(example, text, completion_tokens=completion_tokens, stop_reason=None)


# ---------------------------------------------------------------------------
# Sanity check (no network, no tokenizer)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running marin_math sanity checks...")
    ex = Example(project="t", prompt_ids=[], metadata={"gold_answer": "42"})
    # >= min_response_length(16) tokens and <= target(768): full reward.
    reasoning = "step " * 30
    assert score_completion(ex, f"{reasoning}\nAnswer: \\boxed{{42}}")["reward"] == 1.0
    # Correct via the reference Answer: tail without a box.
    blob = score_completion(ex, f"{reasoning}\nAnswer: 42")
    assert blob["reward"] == 1.0 and blob["exact_match"] == 1.0 and blob["has_box"] == 0.0
    # Suspiciously short correct answer (<16 tokens): 1.0 - lpw = 0.0 (SkyRL clause).
    assert score_completion(ex, "Answer: \\boxed{42}")["reward"] == 0.0
    # Wrong complete answer: -1.
    assert score_completion(ex, f"{reasoning}\nAnswer: \\boxed{{41}}")["reward"] == -1.0
    # No answer at all: graded wrong -> -1 (complete, not truncated).
    assert score_completion(ex, "I cannot solve this, sorry.")["reward"] == -1.0
    # Truncated (>= max_completion_tokens whitespace-proxy tokens): -2.
    long_junk = "word " * int(os.environ.get(ENV_MAX_COMPLETION_TOKENS, "3584"))
    assert score_completion(ex, long_junk)["reward"] == -2.0
    # Equivalent fraction via math_verify.
    ex2 = Example(project="t2", prompt_ids=[], metadata={"gold_answer": "1/2"})
    assert score_completion(ex2, f"{reasoning}\nAnswer: \\boxed{{0.5}}")["exact_match"] == 1.0
    print("OK — marin_math sanity passed")
