"""Task plugin interface for the standalone SGLang-native ZORL client.

A task is a self-contained module that knows how to turn raw inputs (datasets,
puzzle generators, etc.) into prompts the model can answer and how to score
the model's completions. The client (``zorl_client.py``) drives the ZORL
loop; tasks only handle data + reward.

Three pieces every task must implement (duck-typed; no ABC required):

  build_examples(tokenizer, *, train_size, eval_size, seed) -> (train, eval)
      Returns two lists of ``Example`` dicts. Train examples are scored every
      gen and feed into the ES gradient; eval examples are held out and
      scored only by the parent-probe path.

  render_prompt(example, *, tokenizer) -> list[int]
      Returns chat-template-rendered input_ids ready to POST to SGLang's
      /generate. Stored on the Example by ``build_examples`` so we don't
      re-render every gen.

  score_completion(example, generated_text) -> dict[str, float]
      Returns at minimum ``{"reward": float, "exact_match": float}``. Other
      keys (format, partial, etc.) are surfaced in per-step logs unchanged
      so tasks can carry task-specific signal without harness changes.

The standalone client treats ``reward`` as the scalar going into ES updates
and ``exact_match`` as the binary win-rate over the eval set. Both are
needed because some tasks (Wordle, Concision) reward partial progress that
doesn't map to a 0/1 win.

For tasks that need multi-turn rollouts (Wordle), implement an extra method:

  rollout_completion(example, *, generate_turn, lora_path, ...) -> str
      Called by the client when ``task.is_multi_turn`` is truthy. ``generate_turn``
      is a callable the client provides that POSTs one /generate per turn.

Tasks expose ``is_multi_turn = False`` by default; flip to True only if you
need the loop above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """A single training/eval datum.

    project: stable string id used by per-project logs (e.g. ``q_00`` for
        Countdown's first puzzle, ``gsm8k_train_42`` for GSM8K). Lets the
        client emit per-project metrics without depending on task internals.
    prompt_ids: chat-template-rendered tokens for the user-turn prompt. Pre-
        computed at task build time so the hot rollout loop doesn't re-render.
    metadata: task-specific blob. The task's ``score_completion`` reads this
        to compute reward; the client never inspects it.
    """

    project: str
    prompt_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_task(name: str):
    """Import a task module by short name. Keeps the CLI flat (--task gsm8k)
    without exposing the import path."""
    if name == "countdown":
        from . import countdown
        return countdown
    if name == "gsm8k":
        from . import gsm8k
        return gsm8k
    if name == "alphabet_sort":
        from . import alphabet_sort
        return alphabet_sort
    if name == "wordle":
        from . import wordle
        return wordle
    if name == "opd_multiplication":
        from . import opd_multiplication
        return opd_multiplication
    raise ValueError(f"Unknown task: {name!r}. Expected one of: countdown, gsm8k, alphabet_sort, wordle, opd_multiplication")
