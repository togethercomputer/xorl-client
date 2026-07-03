"""Consolidate a dense-Qwen3 xorl DCP checkpoint to a HuggingFace safetensors repo.

The server saves model-only DCP checkpoints in xorl's fused layout (``qkv_proj``,
``gate_up_proj``). This tool loads the DCP shards into a CPU state dict, applies
the model's checkpoint-handler save transform (split back to HF ``q/k/v_proj`` and
``gate/up_proj``), and writes sharded safetensors plus the config/tokenizer files
copied from the base model — a directory loadable by ``from_pretrained`` / SGLang
and uploadable to the Hub.

Usage:
    python -m experiments.marin_rl_6279.consolidate_dcp_to_hf \
        --ckpt /shared/.../weights/default/<run>-step-000146 \
        --base /shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-wc386k_lr1e5-sft \
        --out  /shared/xorl-marin-rl-6279/hf-export/step-000146
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from xorl.checkpoint.format_utils import dcp_to_torch_state_dict
from xorl.models.transformers.qwen3.checkpoint_handler import Qwen3CheckpointHandler


BASE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="DCP checkpoint dir (contains *.distcp)")
    parser.add_argument("--base", type=Path, required=True, help="Base model dir for config/tokenizer files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    config = json.loads((args.base / "config.json").read_text())
    handler = Qwen3CheckpointHandler(
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"],
    )

    print(f"loading DCP state dict from {args.ckpt} ...")
    state_dict = dcp_to_torch_state_dict(args.ckpt)
    print(f"loaded {len(state_dict)} fused params")

    dtype = getattr(torch, args.dtype)
    hf_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in sorted(state_dict.items()):
        for hf_key, hf_tensor in handler.on_save_weight(key, tensor):
            hf_state_dict[hf_key] = hf_tensor.to(dtype).contiguous()
    del state_dict
    n_params = sum(t.numel() for t in hf_state_dict.values())
    print(f"converted to {len(hf_state_dict)} HF params ({n_params / 1e9:.3f}B elements)")

    args.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import save_torch_state_dict  # noqa: PLC0415

    save_torch_state_dict(hf_state_dict, args.out, max_shard_size=args.max_shard_size)

    for name in BASE_FILES:
        src = args.base / name
        if src.exists():
            shutil.copy2(src, args.out / name)
        else:
            print(f"note: base file {name} absent, skipped")

    print(f"wrote HF checkpoint to {args.out}")
    print(sorted(p.name for p in args.out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
