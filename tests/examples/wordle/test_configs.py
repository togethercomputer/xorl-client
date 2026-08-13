from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.wordle.config import ExperimentConfig, load_config
from examples.wordle.train import _validate_capabilities


ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("name", ["importance_sampling", "zero_k3"])
def test_shipped_presets_load_and_resolve_data(name):
    config = load_config(ROOT / f"examples/wordle/configs/{name}.yaml")
    assert Path(config.wordle.targets_path).is_file()
    assert config.preset == name


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


def test_r3_preset_is_a_fail_closed_integration_seam():
    with pytest.raises(ValidationError, match="fail-closed integration seam"):
        load_config(ROOT / "examples/wordle/configs/r3.yaml")
