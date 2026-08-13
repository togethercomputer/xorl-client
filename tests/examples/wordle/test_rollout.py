import asyncio
from pathlib import Path

import pytest

from examples.wordle.config import load_config
from examples.wordle.rollout import build_group_datums, rollout_complete_groups
from examples.wordle.task import WordleTask
from xorl_client import types


ROOT = Path(__file__).parents[3]


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, len(messages)]

    def decode(self, tokens, skip_special_tokens=False):
        return "<guess>[ABIDE]</guess>"


class Sampler:
    def __init__(self):
        self.seeds = []

    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        self.seeds.extend(p.sampling_seed for p in params)
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[10, 11],
                        logprobs=[-0.1, -0.2],
                        text="<guess>[ABIDE]</guess>",
                    )
                ],
                meta_info={"worker": index},
            )
            for index, _ in enumerate(prompts)
        ]


class SixTurnSampler(Sampler):
    words = ["abide", "above", "actor", "acute", "adieu", "crane"]

    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        word = self.words[len(self.seeds) // len(prompts)]
        self.seeds.extend(p.sampling_seed for p in params)
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[10],
                        logprobs=[-0.1],
                        text=f"<guess>[{word.upper()}]</guess>",
                    )
                ],
                meta_info={},
            )
            for _ in prompts
        ]


def _task(config):
    return WordleTask(
        targets_path=config.wordle.targets_path,
        legal_guesses_path=config.wordle.legal_guesses_path,
        train_targets=config.wordle.train_targets,
        eval_targets=config.wordle.eval_targets,
        seed=config.wordle.target_seed,
    )


def test_groups_emit_only_complete_and_every_turn_becomes_a_datum():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.wordle = config.wordle.model_copy(
        update={
            "targets_per_step": 1,
            "group_size": 3,
            "train_targets": 4,
            "eval_targets": 2,
        }
    )
    task = _task(config)
    emitted = []
    sampler = Sampler()

    async def run():
        async def complete(group):
            assert all(row.terminal for row in group)
            emitted.append(group)

        return await rollout_complete_groups(
            task=task,
            tokenizer=Tokenizer(),
            sampler=sampler,
            targets=["above"],
            step=1,
            config=config,
            on_group_complete=complete,
        )

    trajectories = asyncio.run(run())
    assert len(emitted) == 1
    assert len(set(sampler.seeds)) == len(sampler.seeds)
    datums, metrics = build_group_datums(emitted[0], r3_enabled=False)
    assert len(datums) == sum(len(row.turns) for row in trajectories)
    assert metrics["trajectories"] == 3


def test_unsolved_valid_game_reaches_all_six_turns():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.wordle = config.wordle.model_copy(
        update={
            "train_targets": 4,
            "eval_targets": 2,
            "group_size": 2,
        }
    )
    emitted = []
    asyncio.run(
        rollout_complete_groups(
            task=_task(config),
            tokenizer=Tokenizer(),
            sampler=SixTurnSampler(),
            targets=["clown"],
            step=1,
            config=config,
            on_group_complete=lambda group: (emitted.append(group) or asyncio.sleep(0)),
        )
    )
    assert len(emitted[0][0].turns) == 6
    assert not emitted[0][0].solved


def test_r3_datum_seam_fails_closed():
    with pytest.raises(RuntimeError, match="selected-router-weight transport"):
        build_group_datums([], r3_enabled=True)
