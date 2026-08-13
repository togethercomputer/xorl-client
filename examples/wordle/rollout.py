"""Native-batch, multi-turn Wordle rollout and datum construction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

from xorl_client import SamplingParams, types
from xorl_client.rl import build_policy_datum, compute_grpo_advantages

from .config import ExperimentConfig
from .reward import score_trajectory
from .task import WordleTask, compute_feedback, parse_action


@dataclass
class TurnRecord:
    turn: int
    prompt_tokens: list[int]
    output_tokens: list[int]
    old_logprobs: list[float]
    text: str
    raw_text: str
    truncated_after_action: bool
    guess: str | None
    format_ok: bool
    valid_guess: bool
    feedback: str


@dataclass
class Trajectory:
    group_id: str
    rollout_id: int
    target: str
    history: list[tuple[str, str]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    terminal: bool = False
    solved: bool = False
    reward: dict[str, float] = field(default_factory=dict)


def _sampling_params(
    config: ExperimentConfig, seeds: Sequence[int]
) -> list[SamplingParams]:
    return [
        SamplingParams(
            max_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            stop=config.generation.stop or None,
            stop_token_ids=config.generation.stop_token_ids or None,
            ignore_eos=config.generation.ignore_eos,
            sampling_seed=seed,
        )
        for seed in seeds
    ]


def _truncate_after_first_action(
    tokenizer, output_tokens: Sequence[int], *, response_text: str
) -> tuple[list[int], str, bool]:
    """Keep generated tokens only through the first completed guess tag."""

    output = [int(token) for token in output_tokens]
    if not output:
        return [], "", False
    if not hasattr(tokenizer, "decode"):
        if parse_action(response_text)[1]:
            return output, response_text, False
        raise TypeError(
            "tokenizer.decode is required to align a malformed Wordle response"
        )
    decoded = tokenizer.decode(output, skip_special_tokens=False)
    guess, _ = parse_action(decoded)
    if guess is None:
        return output, decoded, False

    for keep_count in range(1, len(output) + 1):
        prefix = tokenizer.decode(output[:keep_count], skip_special_tokens=False)
        if parse_action(prefix)[0] is not None:
            return output[:keep_count], prefix, keep_count < len(output)
    raise RuntimeError("decoded Wordle action has no token-aligned completion boundary")


async def rollout_complete_groups(
    *,
    task: WordleTask,
    tokenizer,
    sampler,
    targets: Sequence[str],
    step: int,
    config: ExperimentConfig,
    on_group_complete: Callable[[list[Trajectory]], Awaitable[None]],
) -> list[Trajectory]:
    """Roll out groups and emit each only after every member is terminal."""

    groups = [
        [
            Trajectory(group_id=f"step-{step}:{target}", rollout_id=row, target=target)
            for row in range(config.wordle.group_size)
        ]
        for target in targets
    ]
    trajectories = [trajectory for group in groups for trajectory in group]
    emitted: set[str] = set()
    semaphore = asyncio.Semaphore(config.generation.concurrency)
    seed_counter = 0

    async def generate(chunk: list[tuple[Trajectory, list[int], int]]):
        async with semaphore:
            prompts = [prompt for _, prompt, _ in chunk]
            seeds = [seed for _, _, seed in chunk]
            rows = await sampler.generate_batch_native_async(
                prompts, _sampling_params(config, seeds), return_logprobs=True
            )
            return chunk, rows

    async def emit_ready() -> None:
        for group in groups:
            group_id = group[0].group_id
            if group_id in emitted or not all(row.terminal for row in group):
                continue
            for row in group:
                row.reward = score_trajectory(
                    row.turns,
                    solved=row.solved,
                    max_turns=config.wordle.max_turns,
                    mode=config.wordle.reward,
                )
            await on_group_complete(group)
            emitted.add(group_id)

    for turn in range(1, config.wordle.max_turns + 1):
        active = [row for row in trajectories if not row.terminal]
        if not active:
            break
        work: list[tuple[Trajectory, list[int], int]] = []
        for row in active:
            prompt = task.prompt_tokens(tokenizer, row.history)
            seed = config.generation.sampling_seed + step * 1_000_000 + seed_counter
            seed_counter += 1
            work.append((row, prompt, seed))
        tasks = [
            asyncio.create_task(
                generate(work[start : start + config.generation.batch_size])
            )
            for start in range(0, len(work), config.generation.batch_size)
        ]
        for future in asyncio.as_completed(tasks):
            chunk, responses = await future
            for (trajectory, prompt, _), response in zip(chunk, responses, strict=True):
                raw_output = list(response.tokens)
                raw_logprobs = list(response.logprobs or [])
                if len(raw_output) != len(raw_logprobs):
                    raise ValueError(
                        "generated token/logprob alignment mismatch: "
                        f"{len(raw_output)} tokens != {len(raw_logprobs)} logprobs"
                    )
                raw_text = response.text or tokenizer.decode(
                    raw_output, skip_special_tokens=False
                )
                output, text, truncated = _truncate_after_first_action(
                    tokenizer, raw_output, response_text=raw_text
                )
                logprobs = raw_logprobs[: len(output)]
                guess, format_ok = parse_action(raw_text)
                trained_guess, _ = parse_action(text)
                if guess != trained_guess:
                    raise RuntimeError(
                        "generated text and token IDs disagree on the played Wordle action"
                    )
                valid_guess = task.valid_guess(guess, trajectory.history)
                feedback = ""
                if valid_guess:
                    assert guess is not None
                    feedback = compute_feedback(guess, trajectory.target)
                    trajectory.history.append((guess, feedback))
                    trajectory.solved = guess == trajectory.target
                trajectory.turns.append(
                    TurnRecord(
                        turn=turn,
                        prompt_tokens=prompt,
                        output_tokens=output,
                        old_logprobs=logprobs,
                        text=text,
                        raw_text=raw_text,
                        truncated_after_action=truncated,
                        guess=guess,
                        format_ok=format_ok,
                        valid_guess=valid_guess,
                        feedback=feedback,
                    )
                )
                trajectory.terminal = bool(
                    trajectory.solved
                    or not valid_guess
                    or turn == config.wordle.max_turns
                )
            await emit_ready()
    await emit_ready()
    if len(emitted) != len(groups):
        raise RuntimeError(
            "rollout ended with incomplete advantage-normalization groups"
        )
    return trajectories


def build_group_datums(
    group: Sequence[Trajectory], *, r3_enabled: bool
) -> tuple[list[types.Datum], dict[str, float]]:
    """Build all generated assistant turns for one complete GRPO group."""

    if r3_enabled:
        raise RuntimeError(
            "R3 datum construction is unavailable until the public selected-router-weight transport lands"
        )

    rewards = [row.reward["reward"] for row in group]
    advantages = compute_grpo_advantages(
        rewards, group_ids=[row.group_id for row in group]
    )
    datums: list[types.Datum] = []
    generated_tokens = 0
    truncated_turns = 0
    for trajectory, advantage in zip(group, advantages, strict=True):
        for turn in trajectory.turns:
            if not turn.output_tokens:
                continue
            generated_tokens += len(turn.output_tokens)
            truncated_turns += int(turn.truncated_after_action)
            datums.append(
                build_policy_datum(
                    prompt_tokens=turn.prompt_tokens,
                    output_tokens=turn.output_tokens,
                    old_logprobs=turn.old_logprobs,
                    advantage=advantage,
                )
            )
    metrics = {
        "groups": 1.0,
        "trajectories": float(len(group)),
        "datums": float(len(datums)),
        "generated_tokens": float(generated_tokens),
        "truncated_after_action_turns": float(truncated_turns),
        "reward_sum": float(sum(rewards)),
        "exact_sum": float(sum(row.reward["exact_match"] for row in group)),
        "format_sum": float(sum(row.reward["format_rate"] for row in group)),
        "valid_sum": float(sum(row.reward["valid_guess_rate"] for row in group)),
        "r3_payload_present_datums": 0.0,
        "r3_payload_aligned_datums": 0.0,
        "r3_payload_rows": 0.0,
        "r3_payload_bytes": 0.0,
        "r3_payload_time_ms": 0.0,
    }
    return datums, metrics


class GroupCoalescer:
    """Bound complete groups and avoid chronically tiny trainer submissions."""

    def __init__(self, *, min_datums: int, max_groups: int, queue_max_groups: int):
        self.min_datums = min_datums
        self.max_groups = max_groups
        self.queue_max_groups = queue_max_groups
        self._groups: list[tuple[list[types.Datum], dict[str, float]]] = []

    def add(self, item: tuple[list[types.Datum], dict[str, float]]):
        self._groups.append(item)
        if len(self._groups) > self.queue_max_groups:
            raise RuntimeError(
                "complete-group trainer queue exceeded its configured bound"
            )
        if (
            sum(len(datums) for datums, _ in self._groups) >= self.min_datums
            or len(self._groups) >= self.max_groups
        ):
            return self.flush()
        return None

    def flush(self):
        if not self._groups:
            return None
        datums = [datum for group, _ in self._groups for datum in group]
        metrics: dict[str, float] = {}
        for _, group_metrics in self._groups:
            for key, value in group_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
        self._groups = []
        return datums, metrics
