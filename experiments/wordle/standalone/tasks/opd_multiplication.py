"""OPD multiplication task: 4-digit × 4-digit integer multiplication, using
the same prompt format as the production OPD pause-distillation runbook
(``experiments/opd_profile/runbooks/opd_complete_runbook.md`` in
xorl-opd-mainline-run).

This is the "ZORL on the OPD task" experiment: same prompts, same prompt
format, verifiable reward (correct product), but updating via ES instead
of OPD's KL-distillation gradient. The teacher is not consulted at all;
the reward is the ground-truth product.

Prompt format (verified against the OPD runbook §1):

    <|im_start|>user
    /no_think Calculate: {A} * {B}<|im_end|>
    <|im_start|>assistant
    <think>
    {' pause' * 100}</think>Answer:_

The model continues with the integer answer. The `</think>Answer: ` lead-in
matches the runbook's confirmed-working format (the 0/20 vs 11/20 base
sweep). Tokenizer's chat template is applied to the user turn, then we
append the assistant prefix with the pause buffer + answer cue raw.

Dataset: ``/shared/opd-coord/randnum_4digit_16384_combined.json`` (16384
prompts, all 4×4-digit pairs). We deterministically slice train/eval.

Reward: extract the last integer in the completion, compare to A*B.
Returns ``{"reward": 1.0|0.0, "exact_match": 1.0|0.0,
"emitted_number": 1.0|0.0}``.

Notes:
* The pause filler is the OPD recipe. Whether ZORL can converge with this
  filler in place is the actual experimental question.
* Default ``--rollout-max-new-tokens`` should be ~32 (an 8-digit product
  is at most 8 tokens plus whitespace; even 24 should suffice). The pause
  buffer is in the *prompt*, not the completion.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from .base import Example


is_multi_turn = False


DEFAULT_DATASET = "/shared/opd-coord/randnum_4digit_16384_combined.json"

# Optional override: set ZORL_OPD_DIGITS to a smaller width (e.g. 2 or 3) to
# generate synthetic prompts in code instead of reading the 4-digit dataset.
# Useful for smoke runs where the base model has zero baseline on 4×4 mult and
# you want a difficulty where ZORL has something to climb on. Defaults to 4
# (production OPD task).
ENV_DIGITS_OVERRIDE = "ZORL_OPD_DIGITS"
ENV_DIGITS_SEED = "ZORL_OPD_DIGITS_SEED"
ENV_REWARD_MODE = "ZORL_OPD_REWARD_MODE"

# Number of pause tokens between <think> and </think>. Matches OPD's training-
# time setting (mt8192 dataset uses 100 pauses). The student is supposed to
# "use" these as a latent thinking buffer.
PAUSE_COUNT = 100

# After-think lead-in. The runbook tested several variants; ``</think>Answer: ``
# (no extra newlines) was confirmed to produce a usable answer prefix on the
# base model (11/20 vs 0/20 with bare ``<think>`` open).
STUDENT_PREFILL_SUFFIX = "</think>Answer: "

_PRODUCT_RE = re.compile(r"(\d{4})\s*\*\s*(\d{4})")
_INTEGER_TOKEN_RE = re.compile(r"-?\d[\d,]*")


def _parse_product(user_text: str) -> tuple[int, int]:
    """Extract the two 4-digit operands from the user prompt. Raises if the
    prompt doesn't match the expected ``Calculate: A * B`` shape."""
    m = _PRODUCT_RE.search(user_text)
    if m is None:
        raise ValueError(f"Could not parse 4-digit product from prompt: {user_text!r}")
    return int(m.group(1)), int(m.group(2))


def _build_prompt_ids(tokenizer, user_text: str) -> list[int]:
    """Render the OPD pause-buffer prompt into input_ids.

    Two-step build:
      1. apply_chat_template on just the user turn with add_generation_prompt
         to get the canonical ``<|im_start|>user…<|im_end|>\n<|im_start|>assistant\n``
         prefix.
      2. Append the OPD-specific suffix: ``<think>\n`` + pause-buffer +
         ``</think>Answer: `` as plain text, tokenized and concatenated.

    We do step 2 by tokenizing the raw suffix string (add_special_tokens=False)
    rather than via chat-template kwargs — keeps the format identical to the
    OPD trainer's on_policy_distillation.py path."""
    msgs = [{"role": "user", "content": user_text}]
    base_ids = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # /no_think directive is in the user content; don't double-add
        return_dict=False,
    )
    pause_buffer = "<think>\n" + (" pause" * PAUSE_COUNT) + STUDENT_PREFILL_SUFFIX
    suffix_ids = tokenizer.encode(pause_buffer, add_special_tokens=False)
    return list(base_ids) + list(suffix_ids)


def _synthetic_pairs(count: int, digits: int, rng: random.Random) -> list[tuple[int, int]]:
    """Generate ``count`` (a, b) operand pairs of exactly ``digits`` digits each.
    Lo = 10^(digits-1) so we never produce a shorter operand (e.g., digits=2
    yields pairs in [10..99] × [10..99])."""
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(count)]


def build_examples(tokenizer, *, train_size: int = 8, eval_size: int = 64, seed: int = 0, dataset_path: str = DEFAULT_DATASET):
    """Build OPD-format multiplication examples.

    Two paths:
      * Default (production): read 4-digit prompts from ``dataset_path``. This
        is the actual OPD training dataset; eval is drawn from a later window
        so train+eval are disjoint.
      * Override (smoke / easier baseline): set the ``ZORL_OPD_DIGITS`` env
        var to a smaller width (e.g. ``2`` for 2-digit ops). Prompts are
        generated synthetically so the model has a workable baseline ZORL can
        climb from. Same OPD pause-filler prompt format either way.
    """
    digits_override = os.environ.get(ENV_DIGITS_OVERRIDE)
    if digits_override and digits_override.strip():
        digits = int(digits_override)
        if digits < 2 or digits > 6:
            raise ValueError(f"{ENV_DIGITS_OVERRIDE}={digits} out of range; expect 2..6")
        rng_seed = int(os.environ.get(ENV_DIGITS_SEED, str(20260602)))
        train_rng = random.Random(rng_seed)
        eval_rng = random.Random(rng_seed + 1_000_003)
        train_pairs = _synthetic_pairs(train_size, digits, train_rng)
        eval_pairs = _synthetic_pairs(eval_size, digits, eval_rng)

        def make(a: int, b: int, pid: str) -> Example:
            user_text = f"/no_think Calculate: {a} * {b}"
            prompt_ids = _build_prompt_ids(tokenizer, user_text)
            return Example(
                project=pid,
                prompt_ids=prompt_ids,
                metadata={"a": a, "b": b, "product": a * b, "user_text": user_text},
            )

        train = [make(a, b, f"opd_train_{i:05d}") for i, (a, b) in enumerate(train_pairs)]
        eval_ = [make(a, b, f"opd_eval_{i:05d}") for i, (a, b) in enumerate(eval_pairs)]
        return train, eval_

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"OPD prompts not found at {path}. The production dataset lives at "
            "/shared/opd-coord/randnum_4digit_16384_combined.json (see the OPD runbook). "
            f"For a smaller-digit smoke, set {ENV_DIGITS_OVERRIDE}=2 (or 3) instead."
        )
    with path.open() as f:
        all_prompts = json.load(f)

    if len(all_prompts) < train_size + eval_size:
        raise ValueError(f"Dataset has {len(all_prompts)} entries, need ≥{train_size + eval_size}")

    def to_example(row, project_id: str) -> Example:
        user_text = row[0]["content"]
        a, b = _parse_product(user_text)
        prompt_ids = _build_prompt_ids(tokenizer, user_text)
        return Example(
            project=project_id,
            prompt_ids=prompt_ids,
            metadata={"a": a, "b": b, "product": a * b, "user_text": user_text},
        )

    train = [to_example(all_prompts[i], f"opd_train_{i:05d}") for i in range(train_size)]
    # Eval drawn from a later window so the two sets are disjoint.
    eval_start = max(train_size, len(all_prompts) - eval_size)
    eval_ = [to_example(all_prompts[eval_start + i], f"opd_eval_{i:05d}") for i in range(eval_size)]
    return train, eval_


def score_completion(example: Example, generated_text: str) -> dict[str, float]:
    """Verifiable reward: 1.0 iff the completion's last integer equals A*B.
    Also surfaces ``emitted_number`` (did the model produce any integer)
    for debugging — useful for catching the OPD failure mode where the
    student emits EOS immediately and produces nothing."""
    if not generated_text:
        return {"reward": 0.0, "exact_match": 0.0, "emitted_number": 0.0}
    # Strip comma thousands separators before scanning; pull the last integer-shaped run.
    cleaned = generated_text.replace(",", "")
    nums = _INTEGER_TOKEN_RE.findall(cleaned)
    if not nums:
        return {"reward": 0.0, "exact_match": 0.0, "emitted_number": 0.0}
    try:
        predicted = int(nums[-1])
    except ValueError:
        return {"reward": 0.0, "exact_match": 0.0, "emitted_number": 0.0}
    gold = int(example.metadata["product"])
    correct = float(predicted == gold)
    if os.environ.get(ENV_REWARD_MODE, "exact").strip().lower() != "numeric_shaped":
        return {"reward": correct, "exact_match": correct, "emitted_number": 1.0}

    abs_error = abs(predicted - gold)
    relative_error = abs_error / max(abs(gold), 1)
    # Smooth dense signal in [0, 1]. Wrong-but-near products get meaningful
    # credit, while orders-of-magnitude misses stay close to zero.
    numeric_closeness = 1.0 / (1.0 + relative_error)
    reward = 1.0 if correct else 0.2 * numeric_closeness
    return {
        "reward": float(reward),
        "exact_match": correct,
        "emitted_number": 1.0,
        "numeric_closeness": float(numeric_closeness),
        "relative_error": float(relative_error),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running opd_multiplication sanity checks...")
    ex = Example(project="t", prompt_ids=[], metadata={"a": 2824, "b": 1409, "product": 2824 * 1409, "user_text": "Calculate: 2824 * 1409"})
    # Correct answer
    assert score_completion(ex, "3979016")["reward"] == 1.0
    # Correct with surrounding text
    assert score_completion(ex, "The answer is 3979016.")["reward"] == 1.0
    # Comma thousands separators
    assert score_completion(ex, "3,979,016")["reward"] == 1.0
    # Wrong answer
    assert score_completion(ex, "1234567")["reward"] == 0.0
    # No number (the OPD failure mode: empty / pure-pause completion)
    r = score_completion(ex, "")
    assert r["reward"] == 0.0 and r["emitted_number"] == 0.0, r
    r = score_completion(ex, "I don't know")
    assert r["reward"] == 0.0 and r["emitted_number"] == 0.0, r
    # Multiple numbers — take the last (typical CoT scratch then final answer)
    assert score_completion(ex, "2824 * 1409 = 3979016")["reward"] == 1.0
    # Wrong final but a correct number appears earlier — should still fail
    assert score_completion(ex, "3979016 is wrong, let me try 7777777")["reward"] == 0.0
    # Parser
    assert _parse_product("/no_think Calculate: 2824 * 1409") == (2824, 1409)
    assert _parse_product("Calculate: 9999 * 0001") == (9999, 1)
    print("OK — opd_multiplication sanity passed")
