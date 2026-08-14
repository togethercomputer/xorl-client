"""Strict configuration for the endpoint-driven Wordle application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    model: str
    tokenizer: str | None = None
    mode: Literal["lora", "full"] = "lora"
    model_id: str = "wordle"
    lora_rank: int = Field(default=32, gt=0)


class TrainerConfig(StrictModel):
    loss_fn: Literal["importance_sampling", "cispo"] = "importance_sampling"
    loss_fn_params: dict[str, Any] = Field(default_factory=dict)
    cispo_clip_low_threshold: float = Field(default=0.0, ge=0.0)
    cispo_clip_high_threshold: float = Field(default=4.0, ge=0.0)
    steps: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0)
    learning_rate_schedule: Literal["constant", "cosine"] = "constant"
    learning_rate_warmup_steps: int = Field(default=0, ge=0)
    min_learning_rate: float = Field(default=0.0, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.95, gt=0, lt=1)
    eps: float = Field(default=1e-12, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip_norm: float = Field(default=1.0, ge=0)
    checkpoint_every: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_cispo_bounds(self) -> "TrainerConfig":
        if self.cispo_clip_high_threshold < self.cispo_clip_low_threshold:
            raise ValueError(
                "cispo_clip_high_threshold must be >= cispo_clip_low_threshold"
            )
        if self.learning_rate_warmup_steps > self.steps:
            raise ValueError("learning_rate_warmup_steps cannot exceed steps")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        return self

    def effective_loss_fn_params(self) -> dict[str, Any]:
        params = dict(self.loss_fn_params)
        if self.loss_fn == "cispo":
            params["clip_low_threshold"] = self.cispo_clip_low_threshold
            params["clip_high_threshold"] = self.cispo_clip_high_threshold
        return params


class WordleConfig(StrictModel):
    targets_path: str = "data/targets.txt"
    legal_guesses_path: str = "data/legal_guesses.txt"
    train_targets: int = Field(default=64, gt=0)
    eval_targets: int = Field(default=16, gt=0)
    targets_per_step: int = Field(default=2, gt=0)
    target_selection: Literal["shuffled_epochs_without_replacement"] = (
        "shuffled_epochs_without_replacement"
    )
    group_size: int = Field(default=4, ge=2)
    max_turns: int = Field(default=6, ge=1, le=6)
    reward: Literal["shaped", "exact_match"] = "shaped"
    target_seed: int = 17


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=24, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    ignore_eos: bool = False
    no_stop_trim: bool = True
    stop: list[str] = Field(default_factory=lambda: ["</guess>"])
    stop_token_ids: list[int] = Field(default_factory=list)
    batch_size: int = Field(default=16, gt=0)
    concurrency: int = Field(default=4, gt=0)
    sampling_seed: int = 1000
    timeout: float = Field(default=1800.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EndpointConfig(StrictModel):
    generation_url: str = ""
    sync_urls: list[str] = Field(default_factory=list)
    sync_world_size: int = Field(default=1, gt=0)
    sync_method: Literal["nccl_ep_scatter", "nccl", "rdma_direct", "p2p"] = (
        "nccl_ep_scatter"
    )
    sync_buffer_mb: int = Field(default=1024, gt=0)


class StreamingConfig(StrictModel):
    min_datums_per_submission: int = Field(default=8, gt=0)
    max_groups_per_submission: int = Field(default=8, gt=0)
    queue_max_groups: int = Field(default=32, gt=0)


class CorrectnessConfig(StrictModel):
    require_finite_loss: bool = True
    require_finite_gradient: bool = True
    max_k3: float | None = Field(default=None, ge=0)
    max_ratio_error: float | None = Field(default=None, ge=0)


class ArtifactConfig(StrictModel):
    output_dir: str = "artifacts/wordle/run"
    resume_from: str | None = None
    source_root: str = "."


class R3Config(StrictModel):
    enabled: bool = False
    required: bool = False
    binary_side_channel: bool = True
    max_payload_bytes: int = Field(default=268_435_456, gt=0)

    @model_validator(mode="after")
    def required_implies_enabled(self) -> "R3Config":
        if self.required and not self.enabled:
            raise ValueError("r3.required=true requires r3.enabled=true")
        return self


class ExperimentConfig(StrictModel):
    preset: Literal["importance_sampling", "cispo", "zero_k3", "r3"]
    model: ModelConfig
    trainer: TrainerConfig
    wordle: WordleConfig
    generation: GenerationConfig
    endpoints: EndpointConfig = Field(default_factory=EndpointConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    r3: R3Config = Field(default_factory=R3Config)

    @model_validator(mode="after")
    def preset_contract(self) -> "ExperimentConfig":
        if (self.preset == "cispo") != (self.trainer.loss_fn == "cispo"):
            raise ValueError("the cispo preset and trainer.loss_fn=cispo must agree")
        if self.preset == "zero_k3" and (
            self.correctness.max_k3 != 0.0
            or self.correctness.max_ratio_error != 0.0
        ):
            raise ValueError(
                "zero_k3 preset requires literal max_k3=0 and max_ratio_error=0"
            )
        if (
            self.preset == "zero_k3"
            and self.trainer.loss_fn_params.get("compute_kl_stats") is not True
        ):
            raise ValueError("zero_k3 preset requires compute_kl_stats=true")
        if "</guess>" in self.generation.stop and not self.generation.no_stop_trim:
            raise ValueError(
                "stopping on </guess> requires generation.no_stop_trim=true"
            )
        if self.preset == "r3" and not (self.r3.enabled and self.r3.required):
            raise ValueError("the R3 preset requires r3.enabled=true and r3.required=true")
        if self.preset != "r3" and self.r3.enabled:
            raise ValueError("R3 can only be enabled by the r3 preset")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load strict JSON or YAML and resolve application paths from the config."""

    config_path = Path(path).resolve()
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "YAML configs require PyYAML; install xorl-client[examples]"
            ) from exc
        raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    config = ExperimentConfig.model_validate(raw)
    base = config_path.parent.parent
    updates: dict[str, str] = {}
    for name in ("targets_path", "legal_guesses_path"):
        value = Path(getattr(config.wordle, name))
        if not value.is_absolute():
            updates[name] = str((base / value).resolve())
    if updates:
        config.wordle = config.wordle.model_copy(update=updates)
    return config
