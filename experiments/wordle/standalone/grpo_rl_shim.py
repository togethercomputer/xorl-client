"""Local port of the missing ``xorl_client.rl`` GRPO helpers.

The installed ``xorl_client`` (git-pinned) lacks the ``rl`` submodule, so
``train_grpo_wordle.py``'s `from xorl_client.rl.advantages import compute_grpo_advantages`
and `from xorl_client.rl.datums import build_policy_loss_inputs` fail at import.

This re-implements both against the engine's verified contract (read from
src/xorl/ops/loss/importance_sampling_loss.py and
src/xorl/server/runner/model_runner.py):
  * importance_sampling loss consumes per-token `target_tokens` (next-token
    aligned, IGNORE_INDEX=-100 on non-trained positions), `logprobs` (old/sampling
    logprobs), and `advantages`.
  * The Datum's model_input is tokens[:-1]; targets are tokens[1:].

Used as a drop-in fallback when xorl_client.rl is absent. No torch dependency
(returns plain lists; the Datum layer converts to TensorData).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Hashable, Sequence

IGNORE_INDEX = -100


def compute_grpo_advantages(
    rewards: Sequence[float],
    *,
    group_ids: Sequence[Hashable],
    normalize: bool = True,
    std_normalization: bool = True,
    eps: float = 1e-8,
) -> list[float]:
    """GRPO advantages: group-mean baseline, optional per-group std normalization.

    advantage_i = (r_i - mean(group)) [/ (std(group)+eps) if std_normalization].
    With normalize=False the raw reward is returned (no baseline). Order preserved.
    """
    rewards = [float(r) for r in rewards]
    if len(rewards) != len(group_ids):
        raise ValueError(f"rewards/group_ids length mismatch: {len(rewards)} vs {len(group_ids)}")
    by_group: dict[Hashable, list[float]] = defaultdict(list)
    for r, g in zip(rewards, group_ids):
        by_group[g].append(r)
    stats: dict[Hashable, tuple[float, float]] = {}
    for g, rs in by_group.items():
        mean = sum(rs) / len(rs)
        if std_normalization and len(rs) > 1:
            var = sum((x - mean) ** 2 for x in rs) / len(rs)
            std = math.sqrt(var)
        else:
            std = 0.0
        stats[g] = (mean, std)
    out: list[float] = []
    for r, g in zip(rewards, group_ids):
        mean, std = stats[g]
        adv = (r - mean) if normalize else r
        if std_normalization:
            adv = adv / (std + eps)
        out.append(float(adv))
    return out


def build_policy_loss_inputs(
    tokens: Sequence[int],
    prompt_len: int,
    old_logprobs: Sequence[float],
    advantage: float,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, list[Any]]:
    """Per-token loss inputs for the importance_sampling loss.

    Model input is tokens[:-1] (length L-1); position i predicts tokens[i+1].
    A position is TRAINED iff its target (tokens[i+1]) is a generated token
    (i+1 >= prompt_len). Non-trained positions get target=ignore_index,
    advantage=0, logprob=0. old_logprobs are the sampler logprobs of the
    generated tokens (length == L - prompt_len), indexed by (i+1)-prompt_len.
    """
    n = len(tokens)
    seq = max(0, n - 1)
    target_tokens = [ignore_index] * seq
    logprobs = [0.0] * seq
    advantages = [0.0] * seq
    old_logprobs = [float(x) for x in old_logprobs]
    for i in range(seq):
        tgt_pos = i + 1
        if tgt_pos >= prompt_len:
            target_tokens[i] = int(tokens[tgt_pos])
            gen_idx = tgt_pos - prompt_len
            if 0 <= gen_idx < len(old_logprobs):
                logprobs[i] = old_logprobs[gen_idx]
            advantages[i] = float(advantage)
    return {"target_tokens": target_tokens, "logprobs": logprobs, "advantages": advantages}
