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


class R3Sampler(Sampler):
    def __init__(self):
        super().__init__()
        self.prefix_starts = []
        self.batch_index = 0

    @staticmethod
    def _descriptor(*, field, path, start_row, source_rows):
        values = {
            "routed_experts": {
                "offset": 0,
                "nbytes": source_rows * 2 * 2 * 4,
                "shape": [source_rows, 2, 2],
                "dtype": "int32",
            },
            "routed_expert_logits": {
                "offset": source_rows * 2 * 2 * 4,
                "nbytes": source_rows * 2 * 2 * 4,
                "shape": [source_rows, 2, 2],
                "dtype": "float32",
            },
        }
        return {
            "schema": "sglang.routed_experts.file.v1",
            "field": field,
            "path": path,
            "error_path": f"{path}.error",
            "start_row": start_row,
            "fields": values,
        }

    async def generate_batch_native_async(self, prompts, params, return_logprobs):
        starts = [param.routed_experts_start_len for param in params]
        self.prefix_starts.append(starts)
        path_prefix = f"/shared/r3-test-{self.batch_index}"
        self.batch_index += 1
        rows = []
        for index, (prompt, start) in enumerate(zip(prompts, starts, strict=True)):
            source_rows = len(prompt) + 2 - 1 - start
            path = f"{path_prefix}-{index}.bin"
            rows.append(
                types.SampleResponse(
                    sequences=[
                        types.SampledSequence(
                            tokens=[10, 11],
                            logprobs=[-0.1, -0.2],
                            text="<guess>[ABIDE]</guess>",
                        )
                    ],
                    meta_info={
                        "routed_experts": self._descriptor(
                            field="routed_experts",
                            path=path,
                            start_row=start,
                            source_rows=source_rows,
                        ),
                        "expert_logits": self._descriptor(
                            field="routed_expert_logits",
                            path=path,
                            start_row=start,
                            source_rows=source_rows,
                        ),
                    },
                )
            )
        return rows


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


def test_r3_reuses_overlapping_prefix_as_append_only_spans():
    config = load_config(ROOT / "examples/wordle/configs/r3.yaml")
    config.wordle = config.wordle.model_copy(
        update={"train_targets": 4, "eval_targets": 2, "group_size": 2}
    )
    emitted = []
    sampler = R3Sampler()
    trajectories = asyncio.run(
        rollout_complete_groups(
            task=_task(config),
            tokenizer=Tokenizer(),
            sampler=sampler,
            targets=["above"],
            step=1,
            config=config,
            on_group_complete=lambda group: emitted.append(group) or asyncio.sleep(0),
        )
    )

    assert sampler.prefix_starts == [[0, 0], [2, 2]]
    assert all(len(row.turns) == 2 for row in trajectories)
    for row in trajectories:
        for field in ("routed_experts", "routed_expert_logits"):
            payload = getattr(row.turns[1], field)
            assert payload["schema"] == "xorl.r3.spans.v1"
            assert payload["rows"] == 4
            assert [span["rows"] for span in payload["spans"]] == [2, 2]

    datums, metrics = build_group_datums(emitted[0], r3_enabled=True)
    assert len(datums) == 4
    assert metrics["r3_payload_present_datums"] == 4
    assert all(datum.routed_experts["rows"] == 4 for datum in datums)
    assert all(datum.routed_expert_logits["rows"] == 4 for datum in datums)
    assert metrics["r3_payload_bytes"] > 0


def test_r3_payload_cap_is_enforced_before_training():
    config = load_config(ROOT / "examples/wordle/configs/r3.yaml")
    config.wordle = config.wordle.model_copy(
        update={"train_targets": 4, "eval_targets": 2, "group_size": 2}
    )
    config.r3 = config.r3.model_copy(update={"max_payload_bytes": 1})
    with pytest.raises(ValueError, match="exceeds max_payload_bytes"):
        asyncio.run(
            rollout_complete_groups(
                task=_task(config),
                tokenizer=Tokenizer(),
                sampler=R3Sampler(),
                targets=["above"],
                step=1,
                config=config,
                on_group_complete=lambda _group: asyncio.sleep(0),
            )
        )
