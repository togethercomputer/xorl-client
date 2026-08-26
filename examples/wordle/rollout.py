"""Shared multi-turn Wordle trajectory and complete-group lifecycle."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Sequence

from xorl_client import SamplingParams, types
from xorl_client.rl import compute_grpo_advantages

from .backends.base import Backend, RenderedPrompt, SampledTurn, SamplingRequest
from .backends.base import generation_budget, rendered_prompt
from .config import ExperimentConfig
from .reward import score_trajectory
from .task import WordleTask, compute_feedback, extract_action_text, extract_guesses
from .training import TrainingSample, xorl_loss_inputs


@dataclass
class TurnRecord:
    turn: int
    sample: SampledTurn
    raw_text: str
    truncated_after_action: bool
    guess: str | None
    single_guess_tag: bool
    format_ok: bool
    strict_format_ok: bool
    valid_guess: bool
    public_constraint_valid: bool
    target_leak: bool
    extra_text: bool
    feedback: str
    solved: bool
    error: str = ""

    # Compatibility accessors for the original public example.
    @property
    def prompt_tokens(self) -> list[int]:
        return self.sample.prompt_tokens

    @property
    def output_tokens(self) -> list[int]:
        return self.sample.output_tokens

    @property
    def trainable_output_tokens(self) -> int:
        value = self.sample.trainable_output_tokens
        return len(self.sample.output_tokens) if value is None else int(value)

    @property
    def old_logprobs(self) -> list[float]:
        return self.sample.logprobs

    @property
    def text(self) -> str:
        return self.sample.text

    @property
    def routed_experts(self) -> Any:
        value = self.sample.backend_metadata
        return value.get("routed_experts") if isinstance(value, dict) else None

    @property
    def routed_expert_logits(self) -> Any:
        value = self.sample.backend_metadata
        return value.get("routed_expert_logits") if isinstance(value, dict) else None


@dataclass
class Trajectory:
    group_id: str
    rollout_id: int
    target: str
    history: list[tuple[str, str]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    terminal: bool = False
    solved: bool = False
    stopped_reason: str = ""
    reward: dict[str, float] = field(default_factory=dict)
    advantage: float = 0.0


def retain_first_completed_action(
    *,
    backend: Backend,
    task: WordleTask,
    request: SamplingRequest,
    sampled: SampledTurn,
) -> tuple[SampledTurn, str, bool]:
    """Co-trim tokens/logprobs through the first completed Wordle action."""

    del task  # The boundary is intentionally task-format specific, not target specific.
    output = [int(token) for token in sampled.output_tokens]
    logprobs = [float(value) for value in sampled.logprobs]
    if len(output) != len(logprobs):
        raise ValueError(
            "generated token/logprob alignment mismatch: "
            f"{len(output)} tokens != {len(logprobs)} logprobs"
        )
    decoded: dict[int, str] = {}

    def decode_prefix(count: int) -> str:
        if count not in decoded:
            decoded[count] = backend.decode_tokens(output[:count])
        return decoded[count]

    def has_completed_action(text: str) -> bool:
        private_think_open = request.prompt.private_think_open
        for match in re.finditer(r"</?think>", text, flags=re.IGNORECASE):
            private_think_open = not match.group(0).startswith("</")
        # The chat template opens the private block in the prompt, outside the
        # generated suffix inspected here. No guess is playable until that
        # inherited block has closed.
        if private_think_open:
            return False
        return bool(extract_guesses(extract_action_text(text)))

    try:
        token_text = decode_prefix(len(output)) if output else ""
    except (AttributeError, TypeError):
        # Compatibility only: the original public tests supplied a tokenizer
        # without decode(). New adapters are required to provide token text.
        token_text = sampled.text or ""
        decoded[len(output)] = token_text
    raw_text = sampled.text or token_text
    action_has_guess = has_completed_action(token_text)
    if not action_has_guess and has_completed_action(raw_text):
        raise RuntimeError(
            "sampler text contains a completed action that cannot be aligned "
            "to its returned token sequence"
        )
    keep = len(output)
    if action_has_guess:
        # Completion is monotone for the required one-think/one-action output,
        # so locate its token boundary in O(log n) tokenizer decodes rather
        # than decoding every growing prefix of a long reasoning response.
        low, high = 1, len(output)
        while low < high:
            count = (low + high) // 2
            try:
                prefix = decode_prefix(count)
            except (AttributeError, TypeError):
                prefix = raw_text if count == len(output) else ""
            if has_completed_action(prefix):
                high = count
            else:
                low = count + 1
        keep = low
        try:
            aligned = has_completed_action(decode_prefix(keep))
        except (AttributeError, TypeError):
            aligned = keep == len(output)
        if not aligned:
            raise RuntimeError(
                "decoded Wordle action has no token-aligned completion boundary"
            )
    if output[:keep]:
        try:
            retained_text = decode_prefix(keep)
        except (AttributeError, TypeError):
            retained_text = raw_text if keep == len(output) else ""
    else:
        retained_text = ""
    metadata = backend.retain_backend_metadata(
        request=request,
        metadata=sampled.backend_metadata,
        original_output_tokens=len(output),
        retained_output_tokens=keep,
    )
    preserve_full_output = bool(
        getattr(backend, "preserve_full_replay_sequence", False)
    )
    submitted_output_tokens = len(output) if preserve_full_output else keep
    return (
        replace(
            sampled,
            output_tokens=output[:submitted_output_tokens],
            logprobs=logprobs[:submitted_output_tokens],
            text=retained_text,
            backend_metadata=metadata,
            trainable_output_tokens=keep,
        ),
        raw_text,
        keep < len(output),
    )


async def rollout_complete_groups(
    *,
    task: WordleTask,
    targets: Sequence[str],
    step: int,
    config: ExperimentConfig,
    on_group_complete: Callable[[list[Trajectory]], Awaitable[None]],
    backend: Backend | None = None,
    tokenizer: Any = None,
    sampler: Any = None,
) -> list[Trajectory]:
    """Roll out groups and emit each only after every member is terminal."""

    if backend is None:
        if tokenizer is None or sampler is None:
            raise TypeError("backend or tokenizer+sampler must be supplied")
        backend = _LegacyXorlSamplerBackend(tokenizer, sampler, config)
    groups = [
        [
            Trajectory(
                group_id=f"step-{step}:{target_index}:{target}",
                rollout_id=row,
                target=target,
            )
            for row in range(config.wordle.group_size)
        ]
        for target_index, target in enumerate(targets)
    ]
    trajectories = [trajectory for group in groups for trajectory in group]
    trajectory_seed_index = {
        (trajectory.group_id, trajectory.rollout_id): index
        for index, trajectory in enumerate(trajectories)
    }
    seed_turn_stride = len(trajectories)
    seed_step_stride = seed_turn_stride * config.wordle.max_turns
    emitted: set[str] = set()
    semaphore = asyncio.Semaphore(config.generation.concurrency)
    legacy_seed_counter = 0

    async def generate(
        chunk: list[tuple[Trajectory, SamplingRequest]],
    ) -> tuple[list[tuple[Trajectory, SamplingRequest]], list[SampledTurn]]:
        async with semaphore:
            rows = await backend.sample_batch([request for _, request in chunk])
        if len(rows) != len(chunk):
            raise RuntimeError(
                "sampler result cardinality mismatch: "
                f"submitted={len(chunk)} returned={len(rows)}"
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
                    stopped_reason=row.stopped_reason,
                )
            await on_group_complete(group)
            emitted.add(group_id)

    for turn in range(1, config.wordle.max_turns + 1):
        active = [row for row in trajectories if not row.terminal]
        if not active:
            break
        work: list[tuple[Trajectory, SamplingRequest]] = []
        for row in active:
            prompt = backend.render_prompt(
                task=task, target=row.target, history=row.history
            )
            if getattr(backend, "legacy_seed", False):
                seed = (
                    int(config.generation.sampling_seed)
                    + int(step) * 1_000_000
                    + legacy_seed_counter
                )
                legacy_seed_counter += 1
            else:
                seed = (
                    int(config.generation.sampling_seed) * 1_000_003
                    + int(step) * seed_step_stride
                    + (int(turn) - 1) * seed_turn_stride
                    + trajectory_seed_index[(row.group_id, row.rollout_id)]
                )
            previous = row.turns[-1].sample if row.turns else None
            work.append(
                (
                    row,
                    SamplingRequest(
                        prompt=prompt,
                        seed=seed,
                        turn=turn,
                        previous_turn=previous,
                    ),
                )
            )
        tasks = [
            asyncio.create_task(
                generate(work[start : start + config.generation.batch_size])
            )
            for start in range(0, len(work), config.generation.batch_size)
        ]
        try:
            for future in asyncio.as_completed(tasks):
                chunk, responses = await future
                for (trajectory, request), response in zip(
                    chunk, responses, strict=True
                ):
                    budget = generation_budget(config, len(request.prompt.tokens))
                    if len(response.output_tokens) > budget:
                        raise RuntimeError(
                            "sampler exceeded the configured response budget: "
                            f"returned={len(response.output_tokens)} budget={budget}"
                        )
                    retained, raw_text, truncated = retain_first_completed_action(
                        backend=backend,
                        task=task,
                        request=request,
                        sampled=response,
                    )
                    action_text = extract_action_text(retained.text or raw_text)
                    parsed = task.parse_response(action_text, trajectory.history)
                    guess = parsed["guess"]
                    guess = str(guess) if guess else None
                    valid_guess = bool(parsed["valid_guess"])
                    feedback = ""
                    solved = False
                    if valid_guess:
                        assert guess is not None
                        feedback = compute_feedback(guess, trajectory.target)
                        trajectory.history.append((guess, feedback))
                        solved = guess == trajectory.target
                        trajectory.solved = solved
                        if solved:
                            trajectory.stopped_reason = "solved"
                        elif turn == config.wordle.max_turns:
                            trajectory.stopped_reason = "max_turns"
                    else:
                        trajectory.stopped_reason = task.invalid_reason(
                            action_text, guess, trajectory.history
                        )
                    trajectory.turns.append(
                        TurnRecord(
                            turn=turn,
                            sample=retained,
                            raw_text=raw_text,
                            truncated_after_action=truncated,
                            guess=guess,
                            single_guess_tag=bool(parsed["single_guess_tag_ok"]),
                            # Production playability is intentionally decoupled from
                            # the stricter public-reasoning format diagnostic.
                            format_ok=bool(
                                parsed["single_guess_tag_ok"]
                                and guess
                                and not (
                                    getattr(backend, "legacy_format", False)
                                    and truncated
                                )
                            ),
                            strict_format_ok=bool(parsed["format_ok"]),
                            valid_guess=valid_guess,
                            public_constraint_valid=bool(
                                parsed["public_constraint_valid"]
                            ),
                            target_leak=bool(parsed["target_leak"]),
                            extra_text=bool(parsed["extra_text"]),
                            feedback=feedback,
                            solved=solved,
                        )
                    )
                    trajectory.terminal = bool(
                        solved or not valid_guess or turn == config.wordle.max_turns
                    )
                    if trajectory.terminal and not trajectory.stopped_reason:
                        trajectory.stopped_reason = "max_turns"
                await emit_ready()
        finally:
            for task_handle in tasks:
                if not task_handle.done():
                    task_handle.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    await emit_ready()
    if len(emitted) != len(groups):
        raise RuntimeError(
            "rollout ended with incomplete advantage-normalization groups"
        )
    return trajectories


class GroupCoalescer:
    """Bound complete groups and avoid chronically tiny trainer submissions."""

    def __init__(self, *, min_datums: int, max_groups: int, queue_max_groups: int):
        self.min_datums = min_datums
        self.max_groups = max_groups
        self.queue_max_groups = queue_max_groups
        self._groups: list[Any] = []

    def add(self, item: Any):
        self._groups.append(item)
        if len(self._groups) > self.queue_max_groups:
            raise RuntimeError(
                "complete-group trainer queue exceeded its configured bound"
            )
        datum_count = sum(
            len(value.samples) if hasattr(value, "samples") else len(value[0])
            for value in self._groups
        )
        if datum_count >= self.min_datums or len(self._groups) >= self.max_groups:
            return self.flush()
        return None

    def flush(self):
        if not self._groups:
            return None
        groups = self._groups
        self._groups = []
        if hasattr(groups[0], "samples"):
            return groups
        datums = [datum for group, _ in groups for datum in group]
        metrics: dict[str, float] = {}
        for _, group_metrics in groups:
            for key, value in group_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
        return datums, metrics


def build_group_datums(
    group: Sequence[Trajectory], *, r3_enabled: bool
) -> tuple[list[types.Datum], dict[str, float]]:
    """Compatibility XoRL conversion retained for existing example callers."""

    rewards = [float(row.reward.get("reward", 0.0)) for row in group]
    advantages = compute_grpo_advantages(
        rewards, group_ids=[row.group_id for row in group]
    )
    datums: list[types.Datum] = []
    truncated = 0
    for trajectory, advantage in zip(group, advantages, strict=True):
        for turn in trajectory.turns:
            sample = TrainingSample(
                group_id=trajectory.group_id,
                rollout_id=trajectory.rollout_id,
                turn=turn.turn,
                prompt_tokens=list(turn.prompt_tokens),
                output_tokens=list(turn.output_tokens),
                sampled_logprobs=list(turn.old_logprobs),
                advantage=float(advantage),
                reward=float(trajectory.reward.get("reward", 0.0)),
                backend_metadata=turn.sample.backend_metadata,
            )
            if not sample.output_tokens:
                continue
            metadata = (
                sample.backend_metadata
                if isinstance(sample.backend_metadata, dict)
                else {}
            )
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(sample.tokens[:-1]),
                    loss_fn_inputs=xorl_loss_inputs(sample),
                    routed_experts=(
                        metadata.get("routed_experts") if r3_enabled else None
                    ),
                    routed_expert_logits=(
                        metadata.get("routed_expert_logits") if r3_enabled else None
                    ),
                )
            )
            truncated += int(turn.truncated_after_action)
    if r3_enabled:
        from .backends.xorl import _routing_payload_bytes, routing_rows

        routing_row_count = sum(
            routing_rows(turn.routed_experts) for row in group for turn in row.turns
        )
        routing_bytes = sum(
            _routing_payload_bytes(turn.routed_experts)
            + _routing_payload_bytes(turn.routed_expert_logits)
            for row in group
            for turn in row.turns
        )
    else:
        routing_row_count = 0
        routing_bytes = 0
    return datums, {
        "groups": 1.0,
        "trajectories": float(len(group)),
        "datums": float(len(datums)),
        "generated_tokens": float(
            sum(len(turn.output_tokens) for row in group for turn in row.turns)
        ),
        "truncated_after_action_turns": float(truncated),
        "reward_sum": float(sum(rewards)),
        "exact_sum": float(sum(row.reward.get("exact_match", 0.0) for row in group)),
        "format_sum": float(sum(row.reward.get("format_rate", 0.0) for row in group)),
        "valid_sum": float(
            sum(row.reward.get("valid_guess_rate", 0.0) for row in group)
        ),
        "r3_payload_present_datums": float(len(datums) if r3_enabled else 0),
        "r3_payload_aligned_datums": float(len(datums) if r3_enabled else 0),
        "r3_payload_rows": float(routing_row_count),
        "r3_payload_bytes": float(routing_bytes),
        "r3_payload_time_ms": 0.0,
    }


class _LegacyXorlSamplerBackend:
    """Small compatibility bridge for the pre-unification rollout tests/API."""

    name = "xorl"
    legacy_format = True
    legacy_seed = True

    def __init__(self, tokenizer: Any, sampler: Any, config: ExperimentConfig):
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.config = config

    def render_prompt(self, *, task, target, history) -> RenderedPrompt:
        return rendered_prompt(
            self.tokenizer,
            task.prompt_tokens(self.tokenizer, history),
            assume_private_think_open=False,
        )

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        if not hasattr(self.tokenizer, "decode"):
            raise AttributeError("legacy tokenizer has no decode method")
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    async def sample_batch(
        self, requests: Sequence[SamplingRequest]
    ) -> list[SampledTurn]:
        replay = self.config.router_replay.enabled
        prefixes = []
        for request in requests:
            previous = request.previous_turn
            previous_input = (
                previous.prompt_tokens + previous.output_tokens
                if previous is not None and replay
                else []
            )
            if previous_input:
                previous_input = previous_input[:-1]
            common = 0
            for left, right in zip(request.prompt.tokens, previous_input):
                if left != right:
                    break
                common += 1
            if replay:
                common = min(common, max(len(request.prompt.tokens) - 1, 0))
            prefixes.append(common)
        params = [
            SamplingParams(
                max_tokens=generation_budget(self.config, len(request.prompt.tokens)),
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                ignore_eos=self.config.generation.ignore_eos,
                no_stop_trim=True,
                sampling_seed=request.seed,
                return_routed_experts=replay,
                return_expert_logits=replay,
                return_routed_experts_file=replay,
                routed_experts_start_len=prefix,
            )
            for request, prefix in zip(requests, prefixes, strict=True)
        ]
        rows = await self.sampler.generate_batch_native_async(
            [request.prompt.tokens for request in requests],
            params,
            return_logprobs=True,
        )
        return [
            SampledTurn(
                prompt_tokens=list(request.prompt.tokens),
                output_tokens=list(row.tokens),
                logprobs=list(row.logprobs or []),
                text=row.text or "",
                backend_metadata=(
                    {
                        "raw": dict(row.meta_info or {}),
                        "prefix_rows": prefix,
                        "previous": (
                            request.previous_turn.backend_metadata
                            if request.previous_turn is not None
                            else None
                        ),
                    }
                    if replay
                    else None
                ),
            )
            for request, row, prefix in zip(requests, rows, prefixes, strict=True)
        ]

    def retain_backend_metadata(
        self,
        *,
        request: SamplingRequest,
        metadata: Any,
        original_output_tokens: int,
        retained_output_tokens: int,
    ) -> Any:
        if not self.config.router_replay.enabled:
            return None
        from .backends.xorl import (
            _routing_payload_bytes,
            _routing_payload_from_descriptor,
        )

        if not isinstance(metadata, dict):
            raise ValueError("Router Replay metadata is missing")
        previous = metadata.get("previous")
        raw = metadata.get("raw")
        prefix = int(metadata.get("prefix_rows", -1))
        rows = len(request.prompt.tokens) + retained_output_tokens - 1
        result = {
            "routed_experts": _routing_payload_from_descriptor(
                raw.get("routed_experts") if isinstance(raw, dict) else None,
                previous=(
                    previous.get("routed_experts")
                    if isinstance(previous, dict)
                    else None
                ),
                prefix_rows=prefix,
                rows=rows,
                field_name="routed_experts",
                dtype="int32",
            ),
            "routed_expert_logits": _routing_payload_from_descriptor(
                raw.get("expert_logits") if isinstance(raw, dict) else None,
                previous=(
                    previous.get("routed_expert_logits")
                    if isinstance(previous, dict)
                    else None
                ),
                prefix_rows=prefix,
                rows=rows,
                field_name="routed_expert_logits",
                dtype="float32",
            ),
        }
        payload_bytes = _routing_payload_bytes(
            result["routed_experts"]
        ) + _routing_payload_bytes(result["routed_expert_logits"])
        payload_limit = self.config.router_replay.max_payload_bytes
        if self.config.r3 is not None:
            payload_limit = self.config.r3.max_payload_bytes
        if payload_bytes > payload_limit:
            raise ValueError(
                "Router Replay payload exceeds max_payload_bytes: "
                f"{payload_bytes} > {payload_limit}"
            )
        return result
