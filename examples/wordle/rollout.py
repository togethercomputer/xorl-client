"""Native-batch, multi-turn Wordle rollout and datum construction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from xorl_client import SamplingParams, types
from xorl_client.rl import build_policy_datum, compute_grpo_advantages

from .config import ExperimentConfig
from .reward import score_trajectory
from .task import WordleTask, compute_feedback, parse_action

_ROUTING_SPANS_SCHEMA = "xorl.r3.spans.v1"
_SGLANG_ROUTING_FILE_SCHEMA = "sglang.routed_experts.file.v1"


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def _routing_rows(payload: Any) -> int:
    if not isinstance(payload, dict) or payload.get("schema") != _ROUTING_SPANS_SCHEMA:
        return 0
    rows = payload.get("rows")
    return int(rows) if isinstance(rows, int) and not isinstance(rows, bool) else 0


def _routing_payload_bytes(payload: Any) -> int:
    if not isinstance(payload, dict) or payload.get("schema") != _ROUTING_SPANS_SCHEMA:
        return 0
    return sum(
        int(span.get("rows", 0)) * int(span.get("row_nbytes", 0))
        for span in payload.get("spans", [])
        if isinstance(span, dict)
    )


def _slice_spans(payload: Any, rows: int) -> list[dict[str, Any]] | None:
    if rows == 0:
        return []
    if rows < 0 or rows > _routing_rows(payload):
        return None
    remaining = rows
    result: list[dict[str, Any]] = []
    for span in payload.get("spans", []):
        if remaining == 0:
            break
        if not isinstance(span, dict) or int(span.get("rows", -1)) < 0:
            return None
        take = min(remaining, int(span["rows"]))
        result.append({**span, "rows": take})
        remaining -= take
    return result if remaining == 0 else None


def _routing_payload_from_descriptor(
    descriptor: Any,
    *,
    previous: Any,
    prefix_rows: int,
    rows: int,
    field_name: str,
    dtype: str,
) -> dict[str, Any]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("schema") != _SGLANG_ROUTING_FILE_SCHEMA
    ):
        raise ValueError("R3 requires an SGLang packed routing descriptor")
    if descriptor.get("field") != field_name:
        raise ValueError(f"R3 descriptor field mismatch for {field_name}")
    fields = descriptor.get("fields")
    field = fields.get(field_name) if isinstance(fields, dict) else None
    shape = field.get("shape") if isinstance(field, dict) else None
    if (
        not isinstance(field, dict)
        or field.get("dtype") != dtype
        or not isinstance(shape, list)
        or len(shape) != 3
        or prefix_rows < 0
        or int(descriptor.get("start_row", -1)) != prefix_rows
        or rows < prefix_rows
    ):
        raise ValueError(f"invalid R3 packed descriptor for {field_name}")
    current_rows = rows - prefix_rows
    source_rows = int(shape[0])
    row_nbytes = int(shape[1]) * int(shape[2]) * 4
    nbytes = int(field.get("nbytes", -1))
    offset = int(field.get("offset", -1))
    if (
        source_rows < current_rows
        or source_rows <= 0
        or int(shape[1]) <= 0
        or int(shape[2]) <= 0
        or nbytes != source_rows * row_nbytes
        or offset < 0
    ):
        raise ValueError(f"invalid R3 packed extent for {field_name}")
    spans = _slice_spans(previous, prefix_rows)
    if spans is None:
        if prefix_rows:
            raise ValueError("R3 prefix is not backed by prior routing spans")
        spans = []
    if current_rows:
        path = descriptor.get("path")
        error_path = descriptor.get("error_path")
        if not isinstance(path, str) or not path or not isinstance(error_path, str) or not error_path:
            raise ValueError("R3 descriptor paths must be non-empty strings")
        spans.append(
            {
                "path": path,
                "error_path": error_path,
                "offset": offset,
                "source_row": 0,
                "rows": current_rows,
                "row_nbytes": row_nbytes,
                "source_shape": [int(value) for value in shape],
                "dtype": dtype,
            }
        )
    return {
        "schema": _ROUTING_SPANS_SCHEMA,
        "rows": rows,
        "shape": [rows, int(shape[1]), int(shape[2])],
        "dtype": dtype,
        "spans": spans,
    }


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
    routed_experts: Any = None
    routed_expert_logits: Any = None


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
    config: ExperimentConfig,
    seeds: Sequence[int],
    prefix_starts: Sequence[int],
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
            no_stop_trim=config.generation.no_stop_trim,
            sampling_seed=seed,
            return_routed_experts=config.r3.enabled,
            return_expert_logits=config.r3.enabled,
            return_routed_experts_file=config.r3.binary_side_channel,
            routed_experts_start_len=prefix_start,
        )
        for seed, prefix_start in zip(seeds, prefix_starts, strict=True)
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

    async def generate(chunk: list[tuple[Trajectory, list[int], int, int]]):
        async with semaphore:
            prompts = [prompt for _, prompt, _, _ in chunk]
            seeds = [seed for _, _, seed, _ in chunk]
            prefix_starts = [prefix for _, _, _, prefix in chunk]
            rows = await sampler.generate_batch_native_async(
                prompts,
                _sampling_params(config, seeds, prefix_starts),
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
        work: list[tuple[Trajectory, list[int], int, int]] = []
        for row in active:
            prompt = task.prompt_tokens(tokenizer, row.history)
            seed = config.generation.sampling_seed + step * 1_000_000 + seed_counter
            seed_counter += 1
            previous = row.turns[-1] if row.turns else None
            previous_model_input = (
                previous.prompt_tokens + previous.output_tokens
                if previous is not None and config.r3.enabled
                else []
            )
            if previous_model_input:
                previous_model_input = previous_model_input[:-1]
            prefix_start = _common_prefix_len(prompt, previous_model_input)
            work.append((row, prompt, seed, prefix_start))
        tasks = [
            asyncio.create_task(
                generate(work[start : start + config.generation.batch_size])
            )
            for start in range(0, len(work), config.generation.batch_size)
        ]
        for future in asyncio.as_completed(tasks):
            chunk, responses = await future
            for (trajectory, prompt, _, prefix_start), response in zip(
                chunk, responses, strict=True
            ):
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
                routed_experts = None
                routed_expert_logits = None
                if config.r3.enabled:
                    previous = trajectory.turns[-1] if trajectory.turns else None
                    rows = len(prompt) + len(output) - 1
                    meta = dict(response.meta_info or {})
                    routed_experts = _routing_payload_from_descriptor(
                        meta.get("routed_experts"),
                        previous=previous.routed_experts if previous else None,
                        prefix_rows=prefix_start,
                        rows=rows,
                        field_name="routed_experts",
                        dtype="int32",
                    )
                    routed_expert_logits = _routing_payload_from_descriptor(
                        meta.get("expert_logits"),
                        previous=previous.routed_expert_logits if previous else None,
                        prefix_rows=prefix_start,
                        rows=rows,
                        field_name="routed_expert_logits",
                        dtype="float32",
                    )
                    if routed_experts["shape"] != routed_expert_logits["shape"]:
                        raise ValueError(
                            "R3 expert IDs and selected weights have different shapes"
                        )
                    payload_bytes = _routing_payload_bytes(
                        routed_experts
                    ) + _routing_payload_bytes(routed_expert_logits)
                    if payload_bytes > config.r3.max_payload_bytes:
                        raise ValueError(
                            "R3 routing payload exceeds max_payload_bytes: "
                            f"{payload_bytes} > {config.r3.max_payload_bytes}"
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
                        valid_guess=valid_guess,
                        feedback=feedback,
                        routed_experts=routed_experts,
                        routed_expert_logits=routed_expert_logits,
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
            datum = build_policy_datum(
                prompt_tokens=turn.prompt_tokens,
                output_tokens=turn.output_tokens,
                old_logprobs=turn.old_logprobs,
                advantage=advantage,
                routed_experts=turn.routed_experts if r3_enabled else None,
                routed_expert_logits=(
                    turn.routed_expert_logits if r3_enabled else None
                ),
            )
            if r3_enabled and (
                _routing_rows(turn.routed_experts)
                != len(turn.prompt_tokens) + len(turn.output_tokens) - 1
                or _routing_rows(turn.routed_expert_logits)
                != len(turn.prompt_tokens) + len(turn.output_tokens) - 1
            ):
                raise ValueError("R3 datum routing rows do not align with model input")
            datums.append(datum)
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
        "r3_payload_present_datums": float(len(datums) if r3_enabled else 0),
        "r3_payload_aligned_datums": float(len(datums) if r3_enabled else 0),
        "r3_payload_rows": float(
            sum(_routing_rows(turn.routed_experts) for row in group for turn in row.turns)
            if r3_enabled
            else 0
        ),
        "r3_payload_bytes": float(
            sum(
                _routing_payload_bytes(turn.routed_experts)
                + _routing_payload_bytes(turn.routed_expert_logits)
                for row in group
                for turn in row.turns
            )
            if r3_enabled
            else 0
        ),
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
