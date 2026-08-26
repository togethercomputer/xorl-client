"""Strict configuration for the unified Wordle GRPO program."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BackendName = Literal["xorl", "river", "tinker"]
LossName = Literal["importance_sampling", "ppo", "cispo"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    model: str
    train_base_model: str | None = None
    tokenizer: str | None = None
    mode: Literal["lora", "full"] = "lora"
    model_id: str = "wordle"
    lora_rank: int = Field(default=16, gt=0)
    lora_seed: int = 9234
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = False

    def resolved_train_base_model(self) -> str:
        return self.train_base_model or self.model


class TrainerConfig(StrictModel):
    loss_fn: LossName = "importance_sampling"
    loss_fn_params: dict[str, Any] = Field(default_factory=dict)
    ppo_clip_low: float = Field(default=0.2, ge=0.0, le=1.0)
    ppo_clip_high: float = Field(default=0.2, ge=0.0)
    ppo_dual_clip: float | None = Field(default=None, gt=1.0)
    cispo_clip_low_threshold: float = Field(default=0.0, ge=0.0)
    cispo_clip_high_threshold: float = Field(default=4.0, ge=0.0)
    steps: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    learning_rate_schedule: Literal["constant", "cosine"] = "cosine"
    learning_rate_warmup_steps: int = Field(default=0, ge=0)
    min_learning_rate: float = Field(default=0.0, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip_norm: float = Field(default=0.0, ge=0)
    checkpoint_every: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def validate_loss_and_schedule(self) -> "TrainerConfig":
        if self.cispo_clip_high_threshold < self.cispo_clip_low_threshold:
            raise ValueError(
                "cispo_clip_high_threshold must be >= cispo_clip_low_threshold"
            )
        if self.learning_rate_warmup_steps > self.steps:
            raise ValueError("learning_rate_warmup_steps cannot exceed steps")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.loss_fn_params.get("return_per_token") is False:
            raise ValueError("return_per_token cannot be disabled for Wordle training")
        return self

    def xorl_loss_name(self) -> str:
        return "policy_loss" if self.loss_fn == "ppo" else self.loss_fn

    def effective_loss_fn_params(
        self, *, backend: BackendName | None = None
    ) -> dict[str, Any]:
        """Return backend wire parameters while enforcing observability fields."""

        params = dict(self.loss_fn_params)
        if backend == "xorl":
            params["return_per_token"] = True
        if self.loss_fn == "ppo":
            if backend == "xorl":
                params.update(
                    eps_clip=self.ppo_clip_low,
                    eps_clip_high=self.ppo_clip_high,
                )
                if self.ppo_dual_clip is not None:
                    params["eps_clip_c"] = self.ppo_dual_clip
            elif backend == "river":
                # River names the asymmetric epsilon distances directly;
                # these are not Tinker's absolute ratio thresholds.
                params.update(
                    clip_low=self.ppo_clip_low,
                    clip_high=self.ppo_clip_high,
                )
            else:
                # Tinker uses absolute ratio thresholds around one.
                params.update(
                    clip_low_threshold=1.0 - self.ppo_clip_low,
                    clip_high_threshold=1.0 + self.ppo_clip_high,
                )
        elif self.loss_fn == "cispo":
            if backend == "river":
                params["eps_max"] = self.cispo_clip_high_threshold
            else:
                params.update(
                    clip_low_threshold=self.cispo_clip_low_threshold,
                    clip_high_threshold=self.cispo_clip_high_threshold,
                )
        # River's importance-sampling schema rejects XoRL diagnostics. Tinker
        # always returns per-token logprobs and has no corresponding switch.
        if backend == "river":
            params.pop("return_per_token", None)
            if self.loss_fn == "importance_sampling":
                params.pop("compute_kl_stats", None)
        if backend == "tinker":
            params.pop("compute_kl_stats", None)
            params.pop("return_per_token", None)
        return params


class WordleConfig(StrictModel):
    train_targets_path: str = "data/train_targets.txt"
    eval_targets_path: str = "data/eval_targets.txt"
    legal_guesses_path: str = "data/legal_guesses.txt"
    train_targets: int = Field(default=4000, gt=0)
    eval_targets: int = Field(default=170, ge=0)
    targets_per_step: int = Field(default=64, gt=0)
    target_selection: Literal["shuffled_epochs_without_replacement"] = (
        "shuffled_epochs_without_replacement"
    )
    group_size: int = Field(default=16, ge=2)
    max_turns: int = Field(default=6, ge=1, le=6)
    reward: Literal["shaped", "exact_match", "wordle"] = "shaped"
    target_seed: int = 9234


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=2048, gt=0)
    max_length: int = Field(default=6144, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    ignore_eos: bool = True
    no_stop_trim: bool = True
    stop: list[str] = Field(default_factory=lambda: ["</guess>"])
    stop_token_ids: list[int] = Field(default_factory=list)
    batch_size: int = Field(default=128, gt=0)
    concurrency: int = Field(default=4, gt=0)
    sampling_seed: int = 9234
    timeout: float = Field(default=3600.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_action_stop(self) -> "GenerationConfig":
        if "</guess>" not in self.stop:
            raise ValueError(
                "Wordle generation must stop at the completed </guess> action"
            )
        if not self.no_stop_trim:
            raise ValueError(
                "stopping on </guess> requires generation.no_stop_trim=true"
            )
        if self.max_new_tokens > self.max_length:
            raise ValueError("max_new_tokens cannot exceed max_length")
        return self


class EndpointConfig(StrictModel):
    """Legacy XoRL-only endpoint shape accepted by the shipped presets."""

    generation_url: str = ""
    sync_urls: list[str] = Field(default_factory=list)
    sync_world_size: int = Field(default=1, gt=0)
    sync_method: Literal["nccl_ep_scatter", "nccl", "rdma_direct", "p2p"] = (
        "nccl_ep_scatter"
    )
    sync_buffer_mb: int = Field(default=1024, gt=0)


class StreamingConfig(StrictModel):
    """Legacy XoRL-only streaming shape accepted by the shipped presets."""

    min_datums_per_submission: int = Field(default=8, gt=0)
    max_groups_per_submission: int = Field(default=8, gt=0)
    queue_max_groups: int = Field(default=32, gt=0)


class CorrectnessConfig(StrictModel):
    """Legacy preset gates retained for source compatibility."""

    require_finite_loss: bool = True
    require_finite_gradient: bool = True
    max_k3: float | None = Field(default=None, ge=0)
    max_ratio_error: float | None = Field(default=None, ge=0)


class ArtifactConfig(StrictModel):
    output_dir: str = "artifacts/wordle/run"
    resume_from: str | None = None
    source_root: str = "."


class WandbConfig(StrictModel):
    project: str = ""
    entity: str = ""
    name: str = ""
    group: str = ""
    tags: list[str] = Field(default_factory=list)
    log_samples: int = Field(default=0, ge=0)


class RouterReplayConfig(StrictModel):
    enabled: bool = False
    max_payload_bytes: int = Field(default=268_435_456, gt=0)


class R3Config(StrictModel):
    """Legacy name and fields for the original XoRL-only presets."""

    enabled: bool = False
    required: bool = False
    binary_side_channel: bool = True
    max_payload_bytes: int = Field(default=268_435_456, gt=0)

    @model_validator(mode="after")
    def required_implies_enabled(self) -> "R3Config":
        if self.required and not self.enabled:
            raise ValueError("r3.required=true requires r3.enabled=true")
        return self


class XorlStreamingConfig(StrictModel):
    pipeline_rl: bool = False
    min_datums_per_submission: int = Field(default=8, gt=0)
    max_groups_per_submission: int = Field(default=8, gt=0)
    queue_max_groups: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def validate_queue_bound(self) -> "XorlStreamingConfig":
        if self.queue_max_groups < self.max_groups_per_submission:
            raise ValueError("queue_max_groups must be >= max_groups_per_submission")
        return self


class XorlBackendConfig(StrictModel):
    trainer_url: str = ""
    generation_url: str = ""
    sync_urls: list[str] = Field(default_factory=list)
    sync_world_size: int = Field(default=1, gt=0)
    sync_method: Literal["nccl_ep_scatter", "nccl", "rdma_direct", "p2p"] = "p2p"
    sync_buffer_mb: int = Field(default=1024, gt=0)
    streaming: XorlStreamingConfig = Field(default_factory=XorlStreamingConfig)


class RiverBackendConfig(StrictModel):
    endpoint: str = "api.river.ai"
    base_model: str | None = None
    project: str = "wordle-grpo-river"
    run_name: str = "wordle-grpo"
    sample_batch_size: int = Field(default=128, gt=0)
    sample_timeout: float = Field(default=3600.0, gt=0)
    operation_timeout: float = Field(default=3600.0, gt=0)
    logprob_temperature: float = Field(default=1.0, gt=0)


class TinkerBackendConfig(StrictModel):
    base_model: str | None = None
    run_name: str = "wordle-grpo"
    sample_inflight: int = Field(default=256, gt=0)
    sample_timeout: float = Field(default=3600.0, gt=0)
    operation_timeout: float = Field(default=3600.0, gt=0)


class BackendConfigs(StrictModel):
    xorl: XorlBackendConfig | None = None
    river: RiverBackendConfig | None = None
    tinker: TinkerBackendConfig | None = None


class ExperimentConfig(StrictModel):
    preset: Literal["importance_sampling", "cispo", "zero_k3", "r3"] | None = None
    model: ModelConfig
    trainer: TrainerConfig
    wordle: WordleConfig
    generation: GenerationConfig
    endpoints: EndpointConfig | None = None
    streaming: StreamingConfig | None = None
    correctness: CorrectnessConfig | None = None
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    r3: R3Config | None = None
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    router_replay: RouterReplayConfig = Field(default_factory=RouterReplayConfig)
    backends: BackendConfigs

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_xorl_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "preset" not in value:
            return value
        upgraded = dict(value)
        if "backends" not in upgraded:
            endpoints = dict(upgraded.get("endpoints") or {})
            streaming = dict(upgraded.get("streaming") or {})
            upgraded["backends"] = {"xorl": {**endpoints, "streaming": streaming}}
        if "router_replay" not in upgraded:
            r3 = dict(upgraded.get("r3") or {})
            upgraded["router_replay"] = {
                "enabled": bool(r3.get("enabled", False)),
                "max_payload_bytes": r3.get("max_payload_bytes", 268_435_456),
            }
        return upgraded

    @model_validator(mode="after")
    def validate_legacy_preset(self) -> "ExperimentConfig":
        if self.preset is None:
            return self
        correctness = self.correctness or CorrectnessConfig()
        r3 = self.r3 or R3Config()
        if (self.preset == "cispo") != (self.trainer.loss_fn == "cispo"):
            raise ValueError("the cispo preset and trainer.loss_fn=cispo must agree")
        if self.preset == "zero_k3" and (
            correctness.max_k3 != 0.0 or correctness.max_ratio_error != 0.0
        ):
            raise ValueError(
                "zero_k3 preset requires literal max_k3=0 and max_ratio_error=0"
            )
        if (
            self.preset == "zero_k3"
            and self.trainer.loss_fn_params.get("compute_kl_stats") is not True
        ):
            raise ValueError("zero_k3 preset requires compute_kl_stats=true")
        if self.preset == "r3" and not (r3.enabled and r3.required):
            raise ValueError(
                "the R3 preset requires r3.enabled=true and r3.required=true"
            )
        if self.preset != "r3" and r3.enabled:
            raise ValueError("R3 can only be enabled by the r3 preset")
        return self

    def validate_backend(self, backend: BackendName) -> None:
        backend_config = getattr(self.backends, backend)
        if backend_config is None:
            raise ValueError(f"configuration has no backends.{backend} section")
        if (
            backend == "xorl"
            and self.router_replay.enabled
            and backend_config.streaming.pipeline_rl
        ):
            raise ValueError(
                "XoRL pipeline mode cannot durably resume Router Replay side-channel files"
            )
        if backend == "tinker" and self.router_replay.enabled:
            raise ValueError("Tinker does not support Router Replay")
        if backend in {"river", "tinker"} and self.model.mode != "lora":
            raise ValueError(
                f"the {backend} adapter currently requires model.mode=lora"
            )
        if (
            backend == "river"
            and self.trainer.loss_fn == "cispo"
            and self.trainer.cispo_clip_low_threshold != 0.0
        ):
            raise ValueError("River CISPO supports only a zero lower ratio bound")
        if (
            backend in {"river", "tinker"}
            and self.trainer.loss_fn == "ppo"
            and self.trainer.ppo_dual_clip is not None
        ):
            raise ValueError(f"{backend} PPO does not support dual clipping")


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
    for name in (
        "train_targets_path",
        "eval_targets_path",
        "legal_guesses_path",
    ):
        value = Path(getattr(config.wordle, name))
        if not value.is_absolute():
            updates[name] = str((base / value).resolve())
    if updates:
        config.wordle = config.wordle.model_copy(update=updates)
    return config
