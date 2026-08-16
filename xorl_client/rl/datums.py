"""Next-token-aligned policy-loss datum construction."""

from __future__ import annotations

import math
from typing import Any, Sequence

from xorl_client import types

IGNORE_INDEX = -100


def build_policy_loss_inputs(
    tokens: Sequence[int],
    prompt_len: int,
    old_logprobs: Sequence[float],
    advantage: float,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, list[int] | list[float]]:
    """Build shifted targets, old logprobs, and advantages for one turn.

    ``tokens`` contains prompt followed by generated tokens. Loss arrays align
    with ``tokens[:-1]`` and predict ``tokens[1:]``. Prompt predictions are
    masked; every generated token must have one decision-time sampler logprob.
    """

    token_list = [int(token) for token in tokens]
    logprobs = [float(value) for value in old_logprobs]
    if not 0 < prompt_len <= len(token_list):
        raise ValueError(
            f"prompt_len must be in [1, {len(token_list)}], got {prompt_len}"
        )
    generated_count = len(token_list) - prompt_len
    if generated_count <= 0:
        raise ValueError("a policy datum must contain at least one generated token")
    if len(logprobs) != generated_count:
        raise ValueError(
            "generated token/logprob alignment mismatch: "
            f"{generated_count} tokens != {len(logprobs)} logprobs"
        )
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("old_logprobs must all be finite")
    advantage_value = float(advantage)
    if not math.isfinite(advantage_value):
        raise ValueError("advantage must be finite")

    sequence_len = len(token_list) - 1
    targets = [ignore_index] * sequence_len
    aligned_logprobs = [0.0] * sequence_len
    advantages = [0.0] * sequence_len
    generated_start = prompt_len - 1
    for generation_index, position in enumerate(range(generated_start, sequence_len)):
        targets[position] = token_list[position + 1]
        aligned_logprobs[position] = logprobs[generation_index]
        advantages[position] = advantage_value
    return {
        "target_tokens": targets,
        "logprobs": aligned_logprobs,
        "advantages": advantages,
    }


def build_policy_datum(
    *,
    prompt_tokens: Sequence[int],
    output_tokens: Sequence[int],
    old_logprobs: Sequence[float],
    advantage: float,
    routed_experts: Any = None,
    routed_expert_logits: Any = None,
) -> types.Datum:
    """Construct a validated policy-loss datum for one generated turn."""

    prompt = [int(token) for token in prompt_tokens]
    output = [int(token) for token in output_tokens]
    combined = prompt + output
    loss_inputs = build_policy_loss_inputs(
        combined, len(prompt), old_logprobs, advantage
    )
    return types.Datum(
        model_input=types.ModelInput.from_ints(combined[:-1]),
        loss_fn_inputs=loss_inputs,
        routed_experts=routed_experts,
        routed_expert_logits=routed_expert_logits,
    )
