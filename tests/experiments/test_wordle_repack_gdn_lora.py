import json
import sys

import pytest


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from experiments.wordle.standalone import repack_gdn_lora  # noqa: E402


def test_no_gdn_passthrough_handles_check_and_preserves_native_rank() -> None:
    tensors = {
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(
            1, 4, 8
        ),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
            1, 16, 4
        ),
    }

    new_tensors, old_rank, new_rank = repack_gdn_lora.repack(
        tensors, key_dim=2, value_dim=4
    )

    assert set(new_tensors) == set(tensors)
    for name, tensor in tensors.items():
        assert torch.equal(new_tensors[name], tensor)
    assert old_rank == 4
    assert new_rank == 4
    repack_gdn_lora.check_equivalence(tensors, new_tensors, key_dim=2, value_dim=4)


def test_no_gdn_main_rewrites_shared_outer_slots_without_rank_expansion(
    tmp_path, monkeypatch
) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    tensors = {
        f"base_model.model.model.layers.0.mlp.experts.{slot}.lora_A.weight": torch.ones(
            1, 4, 8
        )
        for slot in ("w1", "w2", "w3")
    }
    tensors.update(
        {
            f"base_model.model.model.layers.0.mlp.experts.{slot}.lora_B.weight": torch.ones(
                1, 16, 4
            )
            for slot in ("w1", "w2", "w3")
        }
    )
    tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = (
        torch.ones(4, 8)
    )
    tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] = (
        torch.ones(8, 4)
    )
    safetensors_torch.save_file(tensors, in_dir / "adapter_model.safetensors")
    (in_dir / "adapter_config.json").write_text(
        json.dumps(
            {"r": 4, "lora_alpha": 8, "target_modules": ["w1", "w2", "w3", "q_proj"]}
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repack_gdn_lora.py",
            "--input",
            str(in_dir),
            "--output",
            str(out_dir),
            "--check",
        ],
    )
    repack_gdn_lora.main()

    cfg = json.loads((out_dir / "adapter_config.json").read_text())
    out_tensors = safetensors_torch.load_file(out_dir / "adapter_model.safetensors")
    assert cfg["r"] == 4
    assert cfg["lora_alpha"] == 8
    assert cfg["target_modules"] == ["down_proj", "gate_proj", "q_proj", "up_proj"]
    assert set(out_tensors) == set(tensors)
