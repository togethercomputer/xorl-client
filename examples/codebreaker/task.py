"""Instance pools and chat-prompt rendering for the Codebreaker example.

Thin adapter over the byte-faithful ``codebreaker_env``: pools come from the
env's own ``build_examples`` (disjoint by construction), turn prompts come
from its ``build_mt_*`` builders, and everything game-semantic (parsing,
validity, feedback, consistency) is delegated so the port surface stays
exactly the research program's.
"""

from __future__ import annotations

import os
import random
from typing import Sequence

from . import codebreaker_env as env

Instance = env.Instance
History = list[tuple[tuple[str, ...], tuple[int, int]]]


class CodebreakerTask:
    def __init__(
        self,
        *,
        train_size: int,
        eval_size: int,
        seed: int,
        k: int,
        code_length: int,
        allow_repeats: bool = True,
        max_turns: int = env.MAX_TURNS,
        prompt_style: str = "multiturn_transcript",
    ) -> None:
        if prompt_style not in env.MT_PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {env.MT_PROMPT_STYLES}")
        train, eval_, _ = env.build_examples(
            None,
            train_size=train_size,
            eval_size=eval_size,
            val_size=0,
            seed=seed,
            k=k,
            code_length=code_length,
            allow_repeats=allow_repeats,
        )
        self.train_pool: list[str] = [e.metadata["target"] for e in train]
        self.held_out: list[str] = [e.metadata["target"] for e in eval_]
        self.k = k
        self.code_length = code_length
        self.allow_repeats = allow_repeats
        self.max_turns = max_turns
        self.prompt_style = prompt_style

    def describe(self) -> str:
        return env.describe_difficulty(self.k, self.code_length, allow_repeats=self.allow_repeats)

    def decode(self, encoded: str) -> Instance:
        return env.decode_instance(encoded)

    def select_train_targets(self, *, step: int, count: int, seed: int) -> list[str]:
        """Cyclic shuffled epochs without replacement within each epoch (wordle contract)."""
        if step < 1 or count < 1:
            raise ValueError("step and count must be positive")
        start = (step - 1) * count
        selected: list[str] = []
        while len(selected) < count:
            absolute = start + len(selected)
            epoch, offset = divmod(absolute, len(self.train_pool))
            epoch_items = list(self.train_pool)
            random.Random(seed + epoch).shuffle(epoch_items)
            take = min(count - len(selected), len(epoch_items) - offset)
            selected.extend(epoch_items[offset : offset + take])
        return selected

    # -- turn rendering -----------------------------------------------------

    def prompt_messages(
        self,
        instance: Instance,
        events: Sequence[dict],
    ) -> list[dict[str, str]]:
        """Full chat message list for the next turn.

        ``events`` is the ordered per-turn record list built by the rollout;
        each event carries ``assistant`` (canonical guess-tag render or raw
        fallback) and ``user`` (the pre-rendered feedback / invalid message).
        Prior assistant turns replay as canonical tags — thinking text is not
        re-fed, so prompts stay small while turns can reason at length.
        """
        messages = env.build_mt_initial_messages(instance, max_turns=self.max_turns)
        for event in events:
            messages.append({"role": "assistant", "content": event["assistant"]})
            messages.append({"role": "user", "content": event["user"]})
        return messages

    def prompt_tokens(self, tokenizer, instance: Instance, events: Sequence[dict]) -> list[int]:
        messages = self.prompt_messages(instance, events)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=os.environ.get("CODEBREAKER_ENABLE_THINKING", "1") == "1",
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return list(tokenizer(text, add_special_tokens=False)["input_ids"])
        text = "\n".join(f"{row['role']}: {row['content']}" for row in messages)
        encoded = tokenizer.encode(text)
        return list(encoded if isinstance(encoded, list) else encoded.ids)
