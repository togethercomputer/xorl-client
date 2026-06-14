"""Fabricate a no-op attention-only PEFT LoRA adapter for Qwen3.6-35B-A3B.

Mirrors the layout `xorl.lora.utils` emits with ``lora_export_format=peft`` and
``lora_target_modules=[q_proj,k_proj,v_proj,o_proj]`` so the SGLang adapter
load path (q/k/v stacking onto the fused qkv_proj, GDN layers skipped) can be
validated before a training run. lora_B is zero, so generation is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    hidden = text_config.hidden_size
    head_dim = getattr(text_config, "head_dim", hidden // text_config.num_attention_heads)
    num_heads = text_config.num_attention_heads
    num_kv = text_config.num_key_value_heads
    layer_types = list(text_config.layer_types)
    attn_gate = bool(getattr(text_config, "attn_output_gate", False))
    q_out = num_heads * head_dim * (2 if attn_gate else 1)
    kv_out = num_kv * head_dim
    o_in = num_heads * head_dim

    rank = args.rank
    tensors: dict[str, torch.Tensor] = {}
    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    for i in full_attn_layers:
        prefix = f"base_model.model.model.layers.{i}.self_attn"
        for mod, in_dim, out_dim in (
            ("q_proj", hidden, q_out),
            ("k_proj", hidden, kv_out),
            ("v_proj", hidden, kv_out),
            ("o_proj", o_in, hidden),
        ):
            tensors[f"{prefix}.{mod}.lora_A.weight"] = torch.randn(rank, in_dim, dtype=torch.bfloat16) * 0.01
            tensors[f"{prefix}.{mod}.lora_B.weight"] = torch.zeros(out_dim, rank, dtype=torch.bfloat16)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": args.model,
        "r": rank,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    print(f"wrote {len(tensors)} tensors for {len(full_attn_layers)} full-attention layers -> {out_dir}")
    print(f"dims: hidden={hidden} head_dim={head_dim} q_out={q_out} kv_out={kv_out} attn_gate={attn_gate}")


if __name__ == "__main__":
    main()
