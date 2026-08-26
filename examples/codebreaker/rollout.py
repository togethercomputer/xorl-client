"""Multi-turn Codebreaker rollouts against a SamplingClient, GRPO + obo rescue.

Mirrors ``examples/wordle/rollout.py``'s streaming architecture (complete-group
emission, batched generation, per-request seeds) adapted to the codebreaker
contract: instance-based play, burn-turn invalid handling (an invalid guess
spends budget without entering history), candidate tracking for info-bits, and
the obo zero-variance rescue — uniform all-failed groups train at -1/sqrt(n),
uniform all-solved at +1/sqrt(n), mixed zero-variance groups are dropped.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from xorl_client import SamplingParams, types
from xorl_client.rl import build_policy_datum

from . import codebreaker_env as env
from .codebreaker_scoring import score_trajectory
from .config import ExperimentConfig
from .task import CodebreakerTask


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    count = 0
    while count < limit and left[count] == right[count]:
        count += 1
    return count


@dataclass
class TurnRecord:
    turn: int
    prompt_tokens: list[int]
    output_tokens: list[int]
    old_logprobs: list[float]
    text: str
    raw_text: str
    truncated_after_action: bool
    guess: tuple[str, ...] | None
    format_ok: bool
    single_guess_tag: bool
    valid_guess: bool
    feedback: tuple[int, int] | None
    info_bits: float
    constraint_consistent: bool | None


@dataclass
class Trajectory:
    group_id: str
    rollout_id: int
    target: str  # encoded instance
    instance: env.Instance = None  # type: ignore[assignment]
    history: list[tuple[tuple[str, ...], tuple[int, int]]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    candidates: list[tuple[str, ...]] | None = None
    candidates_tracked: bool = False
    terminal: bool = False
    solved: bool = False
    stopped_reason: str = ""
    reward: dict[str, float] = field(default_factory=dict)

    @property
    def samples(self) -> list[TurnRecord]:
        return self.turns


def _sampling_params(config: ExperimentConfig, seeds: Sequence[int]) -> list[SamplingParams]:
    return [
        SamplingParams(
            max_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            stop=config.generation.stop or None,
            stop_token_ids=config.generation.stop_token_ids or None,
            ignore_eos=config.generation.ignore_eos,
            no_stop_trim=config.generation.no_stop_trim,
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
    decoded = tokenizer.decode(output, skip_special_tokens=False)
    if env.extract_guess(decoded) is None:
        return output, decoded, False
    for keep_count in range(1, len(output) + 1):
        prefix = tokenizer.decode(output[:keep_count], skip_special_tokens=False)
        if env.extract_guess(prefix) is not None:
            return output[:keep_count], prefix, keep_count < len(output)
    raise RuntimeError("decoded Codebreaker action has no token-aligned completion boundary")


def _assistant_render(guess: tuple[str, ...] | None) -> str:
    if guess is not None:
        return f"<guess>{env.render_guess(guess)}</guess>"
    return "<guess>(no valid guess)</guess>"


async def rollout_complete_groups(
    *,
    task: CodebreakerTask,
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
            Trajectory(
                group_id=f"step-{step}:{target}",
                rollout_id=row,
                target=target,
                instance=task.decode(target),
            )
            for row in range(config.codebreaker.group_size)
        ]
        for target in targets
    ]
    eps = config.codebreaker.feedback_noise_eps
    for group in groups:
        for row in group:
            if eps == 0.0:
                row.candidates = env.enumerate_candidates(row.instance)
                row.candidates_tracked = row.candidates is not None
    trajectories = [trajectory for group in groups for trajectory in group]
    emitted: set[str] = set()
    semaphore = asyncio.Semaphore(config.generation.concurrency)
    seed_counter = 0

    async def generate(chunk: list[tuple[Trajectory, list[int], int]]):
        async with semaphore:
            prompts = [prompt for _, prompt, _ in chunk]
            seeds = [seed for _, _, seed in chunk]
            rows = await sampler.generate_batch_native_async(
                prompts,
                _sampling_params(config, seeds),
                return_logprobs=True,
            )
            return chunk, rows

    async def emit_ready() -> None:
        for group in groups:
            group_id = group[0].group_id
            if group_id in emitted or not all(row.terminal for row in group):
                continue
            for row in group:
                row.reward = score_trajectory(
                    row,
                    code_length=task.code_length,
                    max_turns=task.max_turns,
                )
            await on_group_complete(group)
            emitted.add(group_id)

    noise_rng_seed = config.generation.sampling_seed * 7919 + step

    for turn in range(1, task.max_turns + 1):
        active = [row for row in trajectories if not row.terminal]
        if not active:
            break
        work: list[tuple[Trajectory, list[int], int]] = []
        for row in active:
            prompt = task.prompt_tokens(tokenizer, row.instance, row.events)
            seed = config.generation.sampling_seed + step * 1_000_000 + seed_counter
            seed_counter += 1
            work.append((row, prompt, seed))
        tasks = [
            asyncio.create_task(generate(work[start : start + config.generation.batch_size]))
            for start in range(0, len(work), config.generation.batch_size)
        ]
        for future in asyncio.as_completed(tasks):
            chunk, responses = await future
            for (trajectory, prompt, seed), response in zip(chunk, responses, strict=True):
                raw_output = list(response.tokens)
                raw_logprobs = list(response.logprobs or [])
                if len(raw_output) != len(raw_logprobs):
                    raise ValueError(
                        "generated token/logprob alignment mismatch: "
                        f"{len(raw_output)} tokens != {len(raw_logprobs)} logprobs"
                    )
                raw_text = response.text or tokenizer.decode(raw_output, skip_special_tokens=False)
                output, text, truncated = _truncate_after_first_action(
                    tokenizer, raw_output, response_text=raw_text
                )
                logprobs = raw_logprobs[: len(output)]
                instance = trajectory.instance
                guess = env.extract_guess(raw_text)
                single_tag = env.has_single_guess_tag(raw_text)
                format_ok = single_tag and guess is not None
                well_formed = env.is_well_formed_guess(guess, instance.code_length)
                valid = env.is_valid_guess(guess, instance, trajectory.history)
                consistent = (
                    env.constraints_satisfied(
                        guess, trajectory.history, code_length=instance.code_length
                    )
                    if well_formed
                    else None
                )
                feedback: tuple[int, int] | None = None
                info_bits = -1.0
                remaining = task.max_turns - turn
                if valid:
                    assert guess is not None
                    true_feedback = env.compute_feedback(guess, instance.secret)
                    solved = true_feedback[0] == instance.code_length
                    if eps > 0.0 and not solved:
                        # Seed discipline per the env's caller contract: derive
                        # the RNG from stable identifiers, never share one.
                        import random as _random  # noqa: PLC0415

                        noise_rng = _random.Random(
                            (noise_rng_seed, trajectory.group_id, trajectory.rollout_id, turn)
                        )
                        feedback = env.apply_feedback_noise(
                            true_feedback[0],
                            true_feedback[1],
                            instance.code_length,
                            noise_rng,
                            eps,
                        )
                    else:
                        feedback = true_feedback
                    trajectory.history.append((guess, feedback))
                    if trajectory.candidates_tracked and trajectory.candidates is not None:
                        before = len(trajectory.candidates)
                        trajectory.candidates = env.filter_candidates(
                            trajectory.candidates, guess, feedback
                        )
                        after = max(len(trajectory.candidates), 1)
                        info_bits = math.log2(max(before, 1) / after)
                    if solved:
                        trajectory.solved = True
                        trajectory.terminal = True
                        trajectory.stopped_reason = "solved"
                    user_message = env.build_mt_feedback_content(
                        trajectory.history,
                        instance=instance,
                        prompt_style=task.prompt_style,
                        max_turns=task.max_turns,
                        remaining=remaining,
                    )
                else:
                    reason = env.invalid_reason(guess, raw_text, instance, trajectory.history)
                    user_message = env.build_mt_invalid_feedback_content(
                        reason,
                        trajectory.history,
                        instance=instance,
                        prompt_style=task.prompt_style,
                        remaining=remaining,
                    )
                trajectory.events.append(
                    {"assistant": _assistant_render(guess), "user": user_message}
                )
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
                        single_guess_tag=single_tag,
                        valid_guess=valid,
                        feedback=feedback,
                        info_bits=info_bits,
                        constraint_consistent=consistent,
                    )
                )
                if not trajectory.terminal and turn == task.max_turns:
                    trajectory.terminal = True
                    trajectory.stopped_reason = "max_turns"
            await emit_ready()
    for row in trajectories:
        if not row.terminal:
            row.terminal = True
            row.stopped_reason = "incomplete"
    await emit_ready()
    if len(emitted) != len(groups):
        raise RuntimeError("rollout ended with incomplete advantage-normalization groups")
    return trajectories


def grpo_with_obo_rescue(
    rewards: Sequence[float],
    exact_flags: Sequence[float],
    group_ids: Sequence[Any],
) -> list[float]:
    """Group-normalized advantages with the obo zero-variance rescue.

    Non-uniform groups: (r - mean) / (pstdev + 1e-6). Uniform-reward groups:
    all-failed -> -1/sqrt(n) each, all-solved -> +1/sqrt(n) each, mixed solve
    flags -> all zeros (the drop-equivalent).
    """
    by_group: dict[Any, list[int]] = {}
    for index, gid in enumerate(group_ids):
        by_group.setdefault(gid, []).append(index)
    advantages = [0.0] * len(rewards)
    for indices in by_group.values():
        rs = [rewards[i] for i in indices]
        mean_r = sum(rs) / len(rs)
        std_r = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        if std_r > 0.0:
            for i in indices:
                advantages[i] = (rewards[i] - mean_r) / (std_r + 1e-6)
            continue
        flags = {exact_flags[i] for i in indices}
        if flags == {0.0}:
            value = -1.0 / math.sqrt(len(indices))
        elif flags == {1.0}:
            value = 1.0 / math.sqrt(len(indices))
        else:
            value = 0.0
        for i in indices:
            advantages[i] = value
    return advantages


def build_group_datums(
    group: Sequence[Trajectory], *, reward_key: str
) -> tuple[list[types.Datum], dict[str, float]]:
    """Build all generated assistant turns for one complete group."""

    rewards = [row.reward[reward_key] for row in group]
    exact = [row.reward["exact_match"] for row in group]
    advantages = grpo_with_obo_rescue(rewards, exact, [row.group_id for row in group])
    metrics = {
        "groups": 1.0,
        "trajectories": float(len(group)),
        "reward_sum": float(sum(rewards)),
        "exact_sum": float(sum(exact)),
        "format_sum": float(sum(row.reward["format_rate"] for row in group)),
        "valid_sum": float(sum(row.reward["valid_guess_rate"] for row in group)),
        "consistent_sum": float(
            sum(row.reward["constraint_consistent_rate"] for row in group)
        ),
        "info_bits_sum": float(sum(row.reward["info_bits_total"] for row in group)),
        "solve_le_5_sum": float(sum(row.reward.get("solve_le_5", 0.0) for row in group)),
    }
    if all(a == 0.0 for a in advantages):
        # Mixed zero-variance group (obo's drop-equivalent): metrics only.
        metrics.update({"datums": 0.0, "generated_tokens": 0.0, "zero_variance_groups": 1.0})
        return [], metrics
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
    metrics.update(
        {
            "datums": float(len(datums)),
            "generated_tokens": float(generated_tokens),
            "truncated_after_action_turns": float(truncated_turns),
        }
    )
    return datums, metrics


class GroupCoalescer:
    """Batch complete groups until enough datums accumulate to submit."""

    def __init__(self, *, min_datums: int, max_groups: int, queue_max_groups: int):
        self.min_datums = min_datums
        self.max_groups = max_groups
        self.queue_max_groups = queue_max_groups
        self._datums: list[types.Datum] = []
        self._metrics: dict[str, float] = {}
        self._groups = 0

    def add(self, item: tuple[list[types.Datum], dict[str, float]]):
        datums, metrics = item
        self._datums.extend(datums)
        for key, value in metrics.items():
            self._metrics[key] = self._metrics.get(key, 0.0) + value
        self._groups += 1
        if len(self._datums) >= self.min_datums or self._groups >= self.max_groups:
            return self.flush()
        return None

    def flush(self):
        if not self._groups:
            return None
        batch = (self._datums, self._metrics)
        self._datums, self._metrics, self._groups = [], {}, 0
        return batch
