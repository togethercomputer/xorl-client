from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from xorl_client import types


DEFAULT_DRGRPO_LOSS_PARAMS = {
    "ratio_type": "sequence",
    "beta": 0.0,
    "clip_low": 0.2,
    "clip_high": 0.2,
    "kl_type": "k3",
    "num_chunks": 8,
    "compute_per_sample_k3": False,
    "return_per_token": False,
}


@dataclass(frozen=True)
class RolloutRecord:
    prompt_id: Hashable
    prefix_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    reward: float
    completion_text: str = ""
    truncated: bool = False
    ref_completion_logprobs: list[float] | None = None
    routed_experts: Any | None = None


def compute_group_advantages(
    records: Sequence[RolloutRecord], *, eps: float = 1e-8, std_mode: str = "population"
) -> list[float]:
    """Group-relative (GRPO) advantages: per-prompt mean-center, divide by group std.

    std_mode:
        population: population std (divide by N); groups with std <= eps keep advantage 0.
        sample: SkyRL-exact semantics — sample std (N-1, torch.std), always divide by
            (std + 1e-6), singleton groups use mean 0 / std 1 (compute_grpo_outcome_advantage).
    """
    if std_mode not in ("population", "sample"):
        raise ValueError(f"Unknown std_mode {std_mode!r}")
    by_prompt: dict[Hashable, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_prompt[record.prompt_id].append(index)

    advantages = [0.0] * len(records)
    for indices in by_prompt.values():
        rewards = [float(records[index].reward) for index in indices]
        if std_mode == "sample":
            if len(rewards) == 1:
                mean, std = 0.0, 1.0
            else:
                mean = sum(rewards) / len(rewards)
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            for index, reward in zip(indices, rewards, strict=True):
                advantages[index] = (reward - mean) / (std + 1e-6)
            continue
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = sqrt(variance)
        if std <= eps:
            continue
        for index, reward in zip(indices, rewards, strict=True):
            advantages[index] = (reward - mean) / std
    return advantages


def build_drgrpo_datum(record: RolloutRecord, advantage: float, *, include_ref_logprobs: bool = False) -> Any:
    if len(record.completion_tokens) != len(record.completion_logprobs):
        raise ValueError(
            "completion_tokens and completion_logprobs must have the same length: "
            f"{len(record.completion_tokens)} != {len(record.completion_logprobs)}"
        )
    if not record.prefix_tokens:
        raise ValueError("prefix_tokens must be non-empty")
    if not record.completion_tokens:
        raise ValueError("completion_tokens must be non-empty")

    full_tokens = list(record.prefix_tokens) + list(record.completion_tokens)
    model_input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    prefix_target_tokens = max(len(record.prefix_tokens) - 1, 0)
    target_tokens[:prefix_target_tokens] = [-100] * prefix_target_tokens
    advantages = [0.0] * prefix_target_tokens + [float(advantage)] * len(record.completion_tokens)
    old_logprobs = [0.0] * prefix_target_tokens + [float(logprob) for logprob in record.completion_logprobs]

    if len(advantages) != len(target_tokens) or len(old_logprobs) != len(target_tokens):
        raise ValueError(
            "constructed loss fields must align with target_tokens: "
            f"targets={len(target_tokens)} advantages={len(advantages)} logprobs={len(old_logprobs)}"
        )

    loss_fn_inputs: dict[str, list[int] | list[float]] = {
        "target_tokens": target_tokens,
        "logprobs": old_logprobs,
        "advantages": advantages,
    }
    if include_ref_logprobs:
        ref_completion_logprobs = record.ref_completion_logprobs or record.completion_logprobs
        if len(ref_completion_logprobs) != len(record.completion_tokens):
            raise ValueError("ref_completion_logprobs must match completion_tokens length")
        loss_fn_inputs["ref_logprobs"] = [0.0] * prefix_target_tokens + [
            float(logprob) for logprob in ref_completion_logprobs
        ]

    return types.Datum(
        model_input=types.ModelInput.from_ints(model_input_tokens),
        loss_fn_inputs=loss_fn_inputs,
        routed_experts=record.routed_experts,
    )
