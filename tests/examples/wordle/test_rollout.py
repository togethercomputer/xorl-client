import asyncio
from pathlib import Path

import pytest

from examples.wordle.config import load_config
from examples.wordle.rollout import build_group_datums, rollout_complete_groups
from examples.wordle.task import WordleTask
from xorl_client import types


ROOT = Path(__file__).parents[3]


class Tokenizer:
    pieces = {
        10: "<guess>[ABIDE]",
        11: "</guess>",
        20: "<guess>[ABIDE]</guess>",
        21: "<guess>[ABOVE]</guess>",
        22: "<guess>[ACTOR]</guess>",
        23: "<guess>[ACUTE]</guess>",
        24: "<guess>[ADIEU]</guess>",
        25: "<guess>[CRANE]</guess>",
    }

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, len(messages)]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in tokens)


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
        word_index = len(self.seeds) // len(prompts)
        word = self.words[word_index]
        self.seeds.extend(p.sampling_seed for p in params)
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[20 + word_index],
                        logprobs=[-0.1],
                        text=f"<guess>[{word.upper()}]</guess>",
                    )
                ],
                meta_info={},
            )
            for _ in prompts
        ]


class MultiGuessTokenizer(Tokenizer):
    pieces = {
        10: "<guess>[ABIDE]</guess>",
        11: " trailing chatter ",
        12: "<guess>[CRANE]</guess>",
        13: "<|im_end|>",
    }

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in tokens)


class MultiGuessSampler(Sampler):
    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        self.seeds.extend(p.sampling_seed for p in params)
        return [
            types.SampleResponse(
                sequences=[
                    types.SampledSequence(
                        tokens=[10, 11, 12, 13],
                        logprobs=[-0.1, -0.2, -0.3, -0.4],
                        text=(
                            "<guess>[ABIDE]</guess> trailing chatter "
                            "<guess>[CRANE]</guess>"
                        ),
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
            on_group_complete=lambda group: emitted.append(group) or asyncio.sleep(0),
        )
    )
    assert len(emitted[0][0].turns) == 6
    assert not emitted[0][0].solved


def test_multi_guess_reward_and_gradient_use_the_same_first_action():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.wordle = config.wordle.model_copy(
        update={"train_targets": 4, "eval_targets": 2, "group_size": 2}
    )
    emitted = []
    trajectories = asyncio.run(
        rollout_complete_groups(
            task=_task(config),
            tokenizer=MultiGuessTokenizer(),
            sampler=MultiGuessSampler(),
            targets=["abide"],
            step=1,
            config=config,
            on_group_complete=lambda group: emitted.append(group) or asyncio.sleep(0),
        )
    )

    assert len(emitted) == 1
    assert all(
        row.solved and row.history == [("abide", "GGGGG")] for row in trajectories
    )
    for row in trajectories:
        turn = row.turns[0]
        assert turn.guess == "abide"
        assert not turn.format_ok
        assert turn.output_tokens == [10]
        assert turn.old_logprobs == [-0.1]
        assert turn.text == "<guess>[ABIDE]</guess>"
        assert "<guess>[CRANE]</guess>" in turn.raw_text
        assert turn.truncated_after_action

    datums, metrics = build_group_datums(emitted[0], r3_enabled=False)
    assert all(
        datum.loss_fn_inputs["target_tokens"].tolist()[-1] == 10 for datum in datums
    )
    assert metrics["truncated_after_action_turns"] == 2


def test_r3_datum_seam_fails_closed():
    with pytest.raises(RuntimeError, match="selected-router-weight transport"):
        build_group_datums([], r3_enabled=True)
