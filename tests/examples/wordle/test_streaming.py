import asyncio
from pathlib import Path

import pytest

from examples.wordle.config import load_config
from examples.wordle.rollout import GroupCoalescer, rollout_complete_groups
from examples.wordle.task import WordleTask
from xorl_client import types


ROOT = Path(__file__).parents[3]


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, len(messages)]


class StragglerSampler:
    def __init__(self):
        self.calls = 0

    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        call = self.calls
        self.calls += 1
        if call == 0:
            await asyncio.sleep(0.05)
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[9], logprobs=[-0.1], text="<guess>[ZZZZZ]</guess>"
                    )
                ],
                meta_info={},
            )
            for _ in prompts
        ]


def test_fast_complete_group_emits_before_straggler_group():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.wordle = config.wordle.model_copy(
        update={
            "train_targets": 4,
            "eval_targets": 2,
            "group_size": 2,
        }
    )
    config.generation = config.generation.model_copy(
        update={"batch_size": 2, "concurrency": 2}
    )
    task = WordleTask(
        targets_path=config.wordle.targets_path,
        legal_guesses_path=config.wordle.legal_guesses_path,
        train_targets=4,
        eval_targets=2,
        seed=config.wordle.target_seed,
    )
    order = []

    async def complete(group):
        order.append(group[0].target)

    asyncio.run(
        rollout_complete_groups(
            task=task,
            tokenizer=Tokenizer(),
            sampler=StragglerSampler(),
            targets=["above", "actor"],
            step=1,
            config=config,
            on_group_complete=complete,
        )
    )
    assert order[0] == "actor"


def test_coalescer_flushes_at_threshold_and_bounds_queue():
    coalescer = GroupCoalescer(min_datums=2, max_groups=10, queue_max_groups=2)
    assert coalescer.add(([object()], {"groups": 1.0})) is None
    datums, metrics = coalescer.add(([object()], {"groups": 1.0}))
    assert len(datums) == 2
    assert metrics["groups"] == 2.0

    bounded = GroupCoalescer(min_datums=100, max_groups=100, queue_max_groups=1)
    bounded.add(([object()], {}))
    with pytest.raises(RuntimeError, match="bound"):
        bounded.add(([object()], {}))
