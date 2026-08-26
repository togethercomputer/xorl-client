from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.wordle.config import ExperimentConfig, load_config
from examples.wordle.train import _validate_capabilities

ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("name", ["importance_sampling", "cispo", "zero_k3", "r3"])
def test_shipped_presets_load_and_resolve_data(name):
    config = load_config(ROOT / f"examples/wordle/configs/{name}.yaml")
    assert Path(config.wordle.train_targets_path).is_file()
    assert Path(config.wordle.eval_targets_path).is_file()
    assert Path(config.wordle.legal_guesses_path).is_file()
    assert Path(config.wordle.train_targets_path).parent.name == "data"
    assert Path(config.wordle.eval_targets_path).parent.name == "data"
    assert config.wordle.train_targets == 4000
    assert config.wordle.eval_targets == 170
    assert config.preset == name
    assert config.model.resolved_train_base_model() == config.model.model
    assert config.generation.no_stop_trim
    if name == "zero_k3":
        assert config.trainer.loss_fn_params["compute_kl_stats"] is True
        assert config.correctness.max_k3 == 0.0
        assert config.correctness.max_ratio_error == 0.0


def test_q36_uses_exact_dataset_files():
    config = load_config(ROOT / "examples/wordle/configs/q36.yaml")
    assert Path(config.wordle.train_targets_path).name == "train_targets.txt"
    assert Path(config.wordle.eval_targets_path).name == "eval_targets.txt"
    assert Path(config.wordle.legal_guesses_path).name == "legal_guesses.txt"
    assert config.wordle.train_targets == 4000
    assert config.wordle.eval_targets == 170


def test_trainer_base_model_can_differ_from_sampler_model():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    config.model = config.model.model_copy(
        update={"train_base_model": "/checkpoints/trainer-base"}
    )

    assert config.model.resolved_train_base_model() == "/checkpoints/trainer-base"
    assert config.model.model == "Qwen/Qwen3-4B-Instruct-2507"


def test_unknown_fields_and_invalid_preset_combinations_rejected():
    config = load_config(ROOT / "examples/wordle/configs/importance_sampling.yaml")
    raw = config.model_dump()
    raw["surprise"] = True
    with pytest.raises(ValidationError, match="surprise"):
        ExperimentConfig.model_validate(raw)
    raw.pop("surprise")
    raw["r3"] = {"enabled": True, "required": True, "max_payload_bytes": 100}
    with pytest.raises(ValidationError, match="R3"):
        ExperimentConfig.model_validate(raw)


def test_demonstrably_missing_endpoint_capability_fails_early():
    config = load_config(ROOT / "examples/wordle/configs/zero_k3.yaml")
    with pytest.raises(ValueError, match="zero_k3"):
        _validate_capabilities(config, {"sampler": {"zero_k3": False}})
    _validate_capabilities(config, {"sampler": {"unavailable": "connection refused"}})


def test_r3_preset_enables_binary_side_channel_without_requiring_zero_k3():
    config = load_config(ROOT / "examples/wordle/configs/r3.yaml")
    assert config.r3.enabled and config.r3.required
    assert config.r3.binary_side_channel
    assert config.trainer.loss_fn_params["compute_kl_stats"] is True
    assert config.correctness.max_k3 == 0.001


def test_cispo_preset_passes_explicit_absolute_ratio_bounds():
    config = load_config(ROOT / "examples/wordle/configs/cispo.yaml")
    assert config.trainer.loss_fn == "cispo"
    assert config.trainer.effective_loss_fn_params() == {
        "compute_kl_stats": True,
        "clip_low_threshold": 0.0,
        "clip_high_threshold": 4.0,
    }


def test_cispo_rejects_inverted_bounds_and_mismatched_preset():
    config = load_config(ROOT / "examples/wordle/configs/cispo.yaml")
    raw = config.model_dump()
    raw["trainer"]["cispo_clip_low_threshold"] = 4.0
    raw["trainer"]["cispo_clip_high_threshold"] = 0.0
    with pytest.raises(ValidationError, match="cispo_clip_high_threshold"):
        ExperimentConfig.model_validate(raw)
    raw = config.model_dump()
    raw["preset"] = "importance_sampling"
    with pytest.raises(ValidationError, match="must agree"):
        ExperimentConfig.model_validate(raw)
