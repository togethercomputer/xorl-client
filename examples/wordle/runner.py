"""Shared logical GRPO lifecycle for XoRL, River, and Tinker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from xorl_client.rl import learning_rate_at_step

from .artifacts import ArtifactStore, WandbSink, terminal_audit
from .backends.base import (
    Backend,
    BackendStep,
    CheckpointResult,
    ForwardBackwardResult,
    OptimizerResult,
    PublishResult,
)
from .config import ExperimentConfig
from .metrics import StructuralAlignmentError, finite_metric_present
from .rollout import rollout_complete_groups
from .task import WordleTask
from .training import GroupTrainingBatch, merge_metrics, retain_training_tokens

_PIPELINE_END = object()


def _group_metrics(totals: dict[str, float]) -> dict[str, float]:
    trajectories = max(int(totals.get("trajectories", 0.0)), 1)
    result = {
        "trajectory_count": float(totals.get("trajectories", 0.0)),
        "datum_count": float(totals.get("datums", 0.0)),
        "group_count": float(totals.get("groups", 0.0)),
        "reward_mean": totals.get("reward_sum", 0.0) / trajectories,
        "advantage_mean": totals.get("advantage_sum", 0.0) / trajectories,
        "advantage_abs_mean": totals.get("advantage_abs_sum", 0.0) / trajectories,
        "skipped_zero_advantage_turns": totals.get("skipped_zero_advantage_turns", 0.0),
        "skipped_empty_turns": totals.get("skipped_empty_turns", 0.0),
        "truncated_after_action_turns": totals.get("truncated_after_action_turns", 0.0),
    }
    for key, value in totals.items():
        if key.endswith("_sum") and key not in {
            "reward_sum",
            "advantage_sum",
            "advantage_abs_sum",
        }:
            result[f"{key[:-4]}_mean"] = value / trajectories
    return result


@dataclass
class _PreparedStep:
    number: int
    backend_step: BackendStep
    targets: list[str]
    trajectories: list[Any]
    totals: dict[str, float]
    started_at: float
    rollout_wall_s: float


@dataclass
class _TrainOutcome:
    prepared: _PreparedStep
    forward: ForwardBackwardResult
    optimizer: OptimizerResult | None
    attempt: int
    has_update: bool
    finite_loss: bool
    finite_gradient: bool | None
    gradient_metrics_reported: bool
    learning_rate: float
    forward_wall_s: float
    optimizer_wall_s: float


@dataclass
class _PipelineRollout:
    number: int
    backend_step: BackendStep
    queue: asyncio.Queue[Any]
    task: asyncio.Task[_PreparedStep]


class ExperimentRunner:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        task: WordleTask,
        backend: Backend,
        store: ArtifactStore,
        session_id: str,
        wandb: WandbSink | None = None,
    ) -> None:
        self.config = config
        self.task = task
        self.backend = backend
        self.store = store
        self.session_id = session_id
        self.wandb = wandb

    def _record_turns(self, step: int, trajectories: list[Any]) -> None:
        for trajectory in trajectories:
            for turn in trajectory.turns:
                self.store.append_turn(
                    {
                        "event": "wordle_turn",
                        "backend": self.backend.name,
                        "step": step,
                        "group_id": trajectory.group_id,
                        "rollout_id": trajectory.rollout_id,
                        "target": trajectory.target,
                        "turn": turn.turn,
                        "guess": turn.guess,
                        "feedback": turn.feedback,
                        "solved": turn.solved,
                        "stopped_reason": trajectory.stopped_reason,
                        "format_ok": turn.format_ok,
                        "strict_format_ok": turn.strict_format_ok,
                        "valid_guess": turn.valid_guess,
                        "public_constraint_valid": turn.public_constraint_valid,
                        "target_leak": turn.target_leak,
                        "extra_text": turn.extra_text,
                        "prompt_tokens": len(turn.prompt_tokens),
                        "retained_response_tokens": len(turn.output_tokens),
                        "truncated_after_action": turn.truncated_after_action,
                        "reward": trajectory.reward,
                        "text": turn.text,
                        "raw_text": (
                            turn.raw_text if turn.truncated_after_action else ""
                        ),
                    }
                )

    def _record_rollout(self, prepared: _PreparedStep) -> None:
        self._record_turns(prepared.number, prepared.trajectories)
        if self.wandb is not None:
            self.wandb.log_samples(
                step=prepared.number, trajectories=prepared.trajectories
            )

    async def _prepare_step(
        self,
        step_number: int,
        *,
        backend_step: BackendStep | None = None,
        batch_queue: asyncio.Queue[Any] | None = None,
    ) -> _PreparedStep:
        started_at = time.monotonic()
        targets = self.task.select_train_targets(
            step=step_number,
            count=self.config.wordle.targets_per_step,
            seed=self.config.wordle.target_seed,
        )
        step = backend_step or await self.backend.begin_step(step_number)
        totals: dict[str, float] = {}

        async def completed(group: list[Any]) -> None:
            batch = retain_training_tokens(group, config=self.config)
            merge_metrics(totals, batch.metrics)
            if batch_queue is None:
                await self.backend.add_group(step, batch)
            else:
                await batch_queue.put(batch)

        rollout_started = time.monotonic()
        trajectories = await rollout_complete_groups(
            task=self.task,
            backend=self.backend,
            targets=targets,
            step=step_number,
            config=self.config,
            on_group_complete=completed,
        )
        return _PreparedStep(
            number=step_number,
            backend_step=step,
            targets=targets,
            trajectories=trajectories,
            totals=totals,
            started_at=started_at,
            rollout_wall_s=time.monotonic() - rollout_started,
        )

    async def _train(self, prepared: _PreparedStep) -> _TrainOutcome:
        step_number = prepared.number
        step = prepared.backend_step
        forward_started = time.monotonic()
        try:
            forward = await self.backend.finish_forward_backward(step)
        except StructuralAlignmentError as exc:
            self.store.write_preoptimizer_gate(
                step_number,
                {
                    "step": step_number,
                    "backend": self.backend.name,
                    "finite_loss": False,
                    "structural_alignment_valid": False,
                    "k3_observational": True,
                    "evaluated_before_optimizer": True,
                    "optimizer_step_requested_at_write": False,
                    "error": str(exc),
                },
            )
            raise
        forward_wall = time.monotonic() - forward_started
        has_update = forward.submitted_datums > 0
        finite_loss = (
            finite_metric_present([forward.metrics], ("loss",)) if has_update else True
        )
        correctness = self.config.correctness
        if correctness is not None and not correctness.require_finite_loss:
            finite_loss = True
        k3 = None
        ratio_error = None
        correctness_gates_passed = True
        if correctness is not None:
            k3_values = [
                abs(float(value))
                for key, value in forward.metrics.items()
                if key.lower().replace(":", "/").rsplit("/", 1)[-1]
                in {"kl_k3_debug_max", "kl_sample_train_k3", "k3", "k3_max"}
            ]
            ratio_values = []
            for key, value in forward.metrics.items():
                leaf = key.lower().replace(":", "/").rsplit("/", 1)[-1]
                parsed = float(value)
                if leaf.endswith("ratio_error"):
                    ratio_values.append(abs(parsed))
                elif leaf.endswith(("ratio_mean", "ratio_min", "ratio_max")):
                    ratio_values.append(abs(parsed - 1.0))
            k3 = max(k3_values) if k3_values else None
            ratio_error = max(ratio_values) if ratio_values else None
            if correctness.max_k3 is not None:
                correctness_gates_passed &= k3 is not None and k3 <= correctness.max_k3
            if correctness.max_ratio_error is not None:
                correctness_gates_passed &= (
                    ratio_error is not None
                    and ratio_error <= correctness.max_ratio_error
                )
        gate = {
            "step": step_number,
            "backend": self.backend.name,
            "finite_loss": finite_loss,
            "structural_alignment_valid": True,
            "k3_observational": True,
            "k3": k3,
            "ratio_error": ratio_error,
            "correctness_gates_passed": correctness_gates_passed,
            "evaluated_before_optimizer": True,
            "optimizer_step_requested_at_write": False,
            **forward.alignment,
        }
        gate_path = self.store.write_preoptimizer_gate(step_number, gate)
        attempt = int(gate_path.stem.rsplit("-", 1)[-1])
        if not finite_loss or not correctness_gates_passed:
            raise RuntimeError(
                f"step {step_number} failed legacy correctness gates; "
                "optimizer not requested"
            )

        learning_rate = learning_rate_at_step(
            base_learning_rate=self.config.trainer.learning_rate,
            step=step_number,
            total_steps=self.config.trainer.steps,
            schedule=self.config.trainer.learning_rate_schedule,
            warmup_steps=self.config.trainer.learning_rate_warmup_steps,
            min_learning_rate=self.config.trainer.min_learning_rate,
        )
        optimizer = None
        optimizer_wall = 0.0
        if has_update:
            optimizer_started = time.monotonic()
            optimizer = await self.backend.optimizer_step(
                step, learning_rate=learning_rate
            )
            optimizer_wall = time.monotonic() - optimizer_started
            gradient_metrics_reported = optimizer.gradient_metrics_reported
            finite_gradient = (
                finite_metric_present([forward.metrics, optimizer.metrics], ("grad",))
                if gradient_metrics_reported
                else None
            )
        else:
            finite_gradient = True
            gradient_metrics_reported = False
        return _TrainOutcome(
            prepared=prepared,
            forward=forward,
            optimizer=optimizer,
            attempt=attempt,
            has_update=has_update,
            finite_loss=finite_loss,
            finite_gradient=finite_gradient,
            gradient_metrics_reported=gradient_metrics_reported,
            learning_rate=learning_rate,
            forward_wall_s=forward_wall,
            optimizer_wall_s=optimizer_wall,
        )

    async def _publish(self, outcome: _TrainOutcome) -> tuple[PublishResult, float]:
        if not outcome.has_update:
            return PublishResult(success=True, detail="no_optimizer_update"), 0.0
        started = time.monotonic()
        publish = await self.backend.publish_policy(outcome.prepared.backend_step)
        elapsed = time.monotonic() - started
        if not publish.success:
            raise RuntimeError(
                f"step {outcome.prepared.number} failed to publish the updated policy: "
                f"{publish.detail}"
            )
        return publish, elapsed

    async def _checkpoint(
        self, outcome: _TrainOutcome
    ) -> tuple[CheckpointResult | None, dict[str, Any] | None]:
        step_number = outcome.prepared.number
        if not (
            step_number % self.config.trainer.checkpoint_every == 0
            or step_number == self.config.trainer.steps
        ):
            return None, None
        result = await self.backend.checkpoint(
            outcome.prepared.backend_step,
            name=(
                f"wordle-{self.session_id}-step-{step_number:08d}"
                f"-attempt-{outcome.attempt:04d}"
            ),
        )
        if not result.optimizer:
            raise RuntimeError("backend checkpoint does not contain optimizer state")
        return result, {
            "step": step_number,
            "path": result.path,
            "model": self.config.model.model,
            "model_id": self.config.model.model_id,
            "session_id": self.session_id,
            "backend": self.backend.name,
            "optimizer": True,
            "attempt": outcome.attempt,
            "metadata": result.metadata,
        }

    def _commit(
        self,
        outcome: _TrainOutcome,
        *,
        publish: PublishResult,
        publish_wall_s: float,
        checkpoint: CheckpointResult | None,
        checkpoint_record: dict[str, Any] | None,
        hybrid: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prepared = outcome.prepared
        optimizer_metrics = outcome.optimizer.metrics if outcome.optimizer else {}
        metrics: dict[str, Any] = {
            "step": prepared.number,
            "backend": self.backend.name,
            "attempt": outcome.attempt,
            "target_cursor": (prepared.number - 1)
            * self.config.wordle.targets_per_step,
            "targets": prepared.targets,
            **_group_metrics(prepared.totals),
            **outcome.forward.alignment,
            "forward_backward_calls": outcome.forward.call_count,
            "forward_backward": outcome.forward.metrics,
            "optimizer": optimizer_metrics,
            "optimizer_complete": outcome.has_update,
            "optimizer_skipped": not outcome.has_update,
            "optimizer_skip_reason": (
                "" if outcome.has_update else "no_retained_datums"
            ),
            "learning_rate": outcome.learning_rate,
            "finite_loss": outcome.finite_loss,
            "finite_gradient": outcome.finite_gradient,
            "gradient_metrics_reported": outcome.gradient_metrics_reported,
            "structural_alignment_valid": True,
            "k3_observational": True,
            "policy_publish": {
                "success": publish.success,
                "detail": publish.detail,
                **publish.metrics,
            },
            "final_sync": publish.success,
            "checkpoint": checkpoint.path if checkpoint is not None else None,
            "rollout_wall_s": prepared.rollout_wall_s,
            "forward_backward_wall_s": outcome.forward_wall_s,
            "optimizer_wall_s": outcome.optimizer_wall_s,
            "policy_publish_wall_s": publish_wall_s,
            "step_wall_s": time.monotonic() - prepared.started_at,
        }
        if hybrid is not None:
            metrics["hybrid"] = hybrid
        # The immutable step JSON is the commit marker. Ledgers may contain a
        # dangling pre-commit row after a crash; resume requires this marker.
        if checkpoint_record is not None:
            self.store.append_checkpoint(checkpoint_record)
        self.store.append_metrics(metrics)
        self.store.write_step(prepared.number, metrics)
        if self.wandb is not None:
            self.wandb.log_step(metrics)
        return metrics

    async def _run_sequential(
        self, *, start_step: int, records: list[dict[str, Any]]
    ) -> None:
        for step_number in range(start_step, self.config.trainer.steps + 1):
            prepared = await self._prepare_step(step_number)
            self._record_rollout(prepared)
            outcome = await self._train(prepared)
            publish, publish_wall = await self._publish(outcome)
            checkpoint, checkpoint_record = await self._checkpoint(outcome)
            records.append(
                self._commit(
                    outcome,
                    publish=publish,
                    publish_wall_s=publish_wall,
                    checkpoint=checkpoint,
                    checkpoint_record=checkpoint_record,
                )
            )

    async def _start_pipeline_rollout(self, step_number: int) -> _PipelineRollout:
        backend_step = await self.backend.begin_step(step_number)
        xorl = self.config.backends.xorl
        assert xorl is not None
        # A paused producer must be able to finish the full target set while the
        # prior step trains. One extra slot is reserved for the terminal marker.
        queue_size = (
            max(
                xorl.streaming.queue_max_groups,
                self.config.wordle.targets_per_step,
            )
            + 1
        )
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_size)

        async def produce() -> _PreparedStep:
            try:
                return await self._prepare_step(
                    step_number,
                    backend_step=backend_step,
                    batch_queue=queue,
                )
            finally:
                queue.put_nowait(_PIPELINE_END)

        return _PipelineRollout(
            number=step_number,
            backend_step=backend_step,
            queue=queue,
            task=asyncio.create_task(produce()),
        )

    async def _cancel_pipeline(self, pipeline: _PipelineRollout) -> None:
        if not pipeline.task.done():
            pipeline.task.cancel()
        await asyncio.gather(pipeline.task, return_exceptions=True)

    async def _drain_pipeline(
        self, pipeline: _PipelineRollout
    ) -> tuple[_PreparedStep | None, Exception | None]:
        try:
            if pipeline.task.done() and not pipeline.task.cancelled():
                failure = pipeline.task.exception()
                if failure is not None:
                    raise failure
            while True:
                item = await pipeline.queue.get()
                if item is _PIPELINE_END:
                    break
                if not isinstance(item, GroupTrainingBatch):
                    raise TypeError(
                        "XoRL pipeline queue received an invalid group batch"
                    )
                if pipeline.task.done() and not pipeline.task.cancelled():
                    failure = pipeline.task.exception()
                    if failure is not None:
                        raise failure
                await self.backend.add_group(pipeline.backend_step, item)
            return await pipeline.task, None
        except Exception as exc:
            if not pipeline.task.done():
                pipeline.task.cancel()
            await asyncio.gather(pipeline.task, return_exceptions=True)
            return None, exc

    async def _run_pipeline(
        self, *, start_step: int, records: list[dict[str, Any]]
    ) -> None:
        current = await self._prepare_step(start_step)
        for step_number in range(start_step, self.config.trainer.steps + 1):
            if current.number != step_number:
                raise RuntimeError("XoRL pipeline lost logical step ordering")
            self._record_rollout(current)
            cycle_started = time.monotonic()
            next_pipeline = (
                await self._start_pipeline_rollout(step_number + 1)
                if step_number < self.config.trainer.steps
                else None
            )
            try:
                outcome = await self._train(current)
                # Save the clean current boundary before any next-step gradients
                # can enter the trainer.
                checkpoint, checkpoint_record = await self._checkpoint(outcome)
            except BaseException:
                if next_pipeline is not None:
                    await self._cancel_pipeline(next_pipeline)
                raise

            next_prepared = None
            next_error: Exception | None = None
            rollout_wait_started = time.monotonic()
            if next_pipeline is not None:
                next_prepared, next_error = await self._drain_pipeline(next_pipeline)
            rollout_wait_wall = time.monotonic() - rollout_wait_started
            publish, publish_wall = await self._publish(outcome)
            hybrid = {
                "enabled": 1.0,
                "trained_step": step_number,
                "generated_step": (
                    next_pipeline.number if next_pipeline is not None else None
                ),
                "generated_policy_step": (
                    step_number - 1 if next_pipeline is not None else None
                ),
                "generated_staleness_updates_when_consumed": (
                    1 if next_pipeline is not None else None
                ),
                "next_rollout_success": (
                    next_error is None if next_pipeline is not None else None
                ),
                "hybrid_train_tail_wall_s": (
                    outcome.forward_wall_s + outcome.optimizer_wall_s
                ),
                "hybrid_post_train_rollout_wait_s": rollout_wait_wall,
                "hybrid_sync_wall_s": publish_wall,
                "hybrid_cycle_wall_s": time.monotonic() - cycle_started,
            }
            records.append(
                self._commit(
                    outcome,
                    publish=publish,
                    publish_wall_s=publish_wall,
                    checkpoint=checkpoint,
                    checkpoint_record=checkpoint_record,
                    hybrid=hybrid,
                )
            )
            if next_error is not None:
                raise next_error
            if next_pipeline is not None:
                assert next_prepared is not None
                current = next_prepared

    async def run(self, *, start_step: int = 1) -> dict[str, Any]:
        records: list[dict[str, Any]] = self.store.load_step_records(
            before_step=start_step
        )
        if start_step <= self.config.trainer.steps:
            xorl = self.config.backends.xorl
            pipeline = bool(
                self.backend.name == "xorl"
                and xorl is not None
                and xorl.streaming.pipeline_rl
            )
            if pipeline:
                await self._run_pipeline(start_step=start_step, records=records)
            else:
                await self._run_sequential(start_step=start_step, records=records)

        audit = terminal_audit(
            records=records,
            expected_steps=self.config.trainer.steps,
            checkpoint_every=self.config.trainer.checkpoint_every,
        )
        audit["backend"] = self.backend.name
        audit["final_checkpoint"] = records[-1].get("checkpoint") if records else None
        self.store.write_terminal_audit(audit)
        if self.wandb is not None:
            self.wandb.finish(audit)
        if not audit["success"]:
            raise RuntimeError(f"terminal audit failed: {audit}")
        return audit
