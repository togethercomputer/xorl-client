"""Build a zero/kaiming-init LoRA adapter for SGLang-native ZORL — no trainer required.

Produces an ``adapter_config.json`` + ``adapter_model.safetensors`` pair on disk
in xorl's ``sglang_shared_outer`` layout, which is what SGLang's
``/load_lora_adapter`` accepts for hybrid_shared MoE LoRA.

Init convention: A is kaiming-uniform (non-zero), B is all-zero. This matches
the standard PEFT init: A*B = 0 at start (so the parent contributes nothing to
the base forward), but ZORL's B-only perturbations produce non-zero candidate
contributions because A is non-zero.

For Qwen3MoE-style MoE LoRA (shared_outer):

  * Attention: ``self_attn.{qkv_proj,o_proj}.lora_{A,B}.weight`` — standard 2D
    PEFT tensors, shapes ``[rank, in]`` and ``[out, rank]``.
  * MoE experts: ``mlp.experts.{w1,w2,w3}.lora_{A,B}.weight`` — 3D tensors with
    the expert dim either 1 (shared) or num_experts (per-expert):
      w1 (gate): A shared ``[1, r, hidden]``, B per-expert ``[E, moe_int, r]``
      w2 (down): A per-expert ``[E, r, moe_int]``, B shared ``[1, hidden, r]``
      w3 (up):   A shared ``[1, r, hidden]``, B per-expert ``[E, moe_int, r]``

Usage:

  python init_zorl_adapter.py \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --output-dir /shared/zorl/init-adapters/qwen3-30b-a3b-r4 \\
      --rank 4 --alpha 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

_PEFT_PREFIX = "base_model.model.model.layers"
_PROJ_TO_SGLANG_W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
# Per-expert vs shared dim in the shared_outer layout:
#   gate/up (w1, w3): A is shared (E=1), B is per-expert (E=num_experts)
#   down   (w2):      A is per-expert,   B is shared
_MOE_A_PER_EXPERT = {"gate_proj": False, "up_proj": False, "down_proj": True}
_MOE_B_PER_EXPERT = {"gate_proj": True, "up_proj": True, "down_proj": False}


def _kaiming_uniform_(tensor: torch.Tensor, a: float = math.sqrt(5)) -> torch.Tensor:
    """In-place kaiming uniform — same defaults as torch.nn.Linear's weight init,
    which is what PEFT uses for lora_A by default."""
    torch.nn.init.kaiming_uniform_(tensor, a=a)
    return tensor


def _attention_dims(cfg) -> dict:
    """Compute attention dimensions from the model config. Caller decides
    whether to use the fused-qkv or split q/k/v dims via ``fused_qkv``.

    ``attn_output_gate`` (Qwen3.5/3.6 hybrid models) fuses a per-head output
    gate into the q section of qkv_proj, doubling it: e.g. Qwen3.6-35B-A3B
    qkv out is 2*16*256 + 2*2*256 = 9216, not 5120. SGLang sizes its LoRA
    buffers accordingly (Qwen3_5MoeForConditionalGeneration.get_hidden_dim),
    and rejects ungated-shaped adapter B tensors.
    """
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
    q_mult = 2 if getattr(cfg, "attn_output_gate", False) else 1
    qkv_out = (n_heads * q_mult + 2 * n_kv) * head_dim
    o_in = n_heads * head_dim
    return {"hidden": hidden, "qkv_out": qkv_out, "o_in": o_in, "head_dim": head_dim, "n_heads": n_heads, "n_kv": n_kv}


def _gdn_dims(cfg) -> dict | None:
    """GatedDeltaNet (linear-attention) projection dims for hybrid models.

    Qwen3.6-35B-A3B: key_dim = 16*128 = 2048, value_dim = 32*128 = 4096 →
      in_proj_qkvz: hidden(2048) → 2*key + 2*value = 12288
      in_proj_ba:   hidden(2048) → 2*num_value_heads = 64
      out_proj:     value_dim(4096) → hidden(2048)
    Returns None when the config has no linear-attention setup.
    """
    n_k = getattr(cfg, "linear_num_key_heads", None)
    n_v = getattr(cfg, "linear_num_value_heads", None)
    d_k = getattr(cfg, "linear_key_head_dim", None)
    d_v = getattr(cfg, "linear_value_head_dim", None)
    if None in (n_k, n_v, d_k, d_v):
        return None
    key_dim = n_k * d_k
    value_dim = n_v * d_v
    return {
        "hidden": cfg.hidden_size,
        "qkvz_out": 2 * key_dim + 2 * value_dim,
        "ba_out": 2 * n_v,
        "out_proj_in": value_dim,
    }


def _full_attention_layer_flags(cfg) -> list[bool]:
    """Per-layer flag: True = full attention, False = linear attention (GDN).

    Prefers the explicit ``layer_types`` list; falls back to
    ``full_attention_interval`` (full attention every Nth layer, i.e. layers
    N-1, 2N-1, ... — Qwen3.6's interval 4 puts them at 3, 7, ..., 39); a
    non-hybrid model (neither field) is all full attention. When both fields
    are present they are cross-checked.
    """
    num_layers = cfg.num_hidden_layers
    layer_types = getattr(cfg, "layer_types", None)
    interval = getattr(cfg, "full_attention_interval", None)
    if layer_types:
        assert len(layer_types) == num_layers, (
            f"layer_types has {len(layer_types)} entries for {num_layers} layers"
        )
        flags = [t == "full_attention" for t in layer_types]
        if interval:
            expected = [(i + 1) % interval == 0 for i in range(num_layers)]
            assert flags == expected, (
                f"layer_types disagrees with full_attention_interval={interval}"
            )
        return flags
    if interval:
        return [(i + 1) % interval == 0 for i in range(num_layers)]
    return [True] * num_layers


def _detect_is_moe(cfg) -> bool:
    """A model is MoE if its config exposes a non-empty experts setup."""
    return getattr(cfg, "num_experts", None) is not None and getattr(cfg, "moe_intermediate_size", None) is not None


def _build_attention_tensors(layer_idx: int, rank: int, dims: dict, dtype: torch.dtype, *, fused_qkv: bool) -> dict:
    """Attention LoRA: standard 2D PEFT tensors. A: [rank, in], B: [out, rank].

    Models split here:
      * fused_qkv=True  → ``qkv_proj`` (Qwen3-30B-A3B-Instruct-2507, etc.).
        Single fused projection; B's out_dim is (n_heads + 2*n_kv) * head_dim.
      * fused_qkv=False → separate q_proj/k_proj/v_proj (Qwen3-0.6B-Base, etc.).
        Each has its own LoRA pair.

    o_proj is always separate (input n_heads*head_dim → output hidden).
    """
    out = {}
    base = f"{_PEFT_PREFIX}.{layer_idx}.self_attn"
    hidden = dims["hidden"]
    head_dim = dims["head_dim"]
    if fused_qkv:
        qkv_A = _kaiming_uniform_(torch.empty(rank, hidden, dtype=dtype))
        qkv_B = torch.zeros(dims["qkv_out"], rank, dtype=dtype)
        out[f"{base}.qkv_proj.lora_A.weight"] = qkv_A
        out[f"{base}.qkv_proj.lora_B.weight"] = qkv_B
    else:
        q_out = dims["n_heads"] * head_dim
        kv_out = dims["n_kv"] * head_dim
        for proj, out_dim in (("q_proj", q_out), ("k_proj", kv_out), ("v_proj", kv_out)):
            out[f"{base}.{proj}.lora_A.weight"] = _kaiming_uniform_(torch.empty(rank, hidden, dtype=dtype))
            out[f"{base}.{proj}.lora_B.weight"] = torch.zeros(out_dim, rank, dtype=dtype)
    # o_proj — output projection: input n_heads*head_dim, output hidden.
    out[f"{base}.o_proj.lora_A.weight"] = _kaiming_uniform_(torch.empty(rank, dims["o_in"], dtype=dtype))
    out[f"{base}.o_proj.lora_B.weight"] = torch.zeros(hidden, rank, dtype=dtype)
    return out


def _build_dense_mlp_tensors(layer_idx: int, rank: int, hidden: int, intermediate: int, dtype: torch.dtype) -> dict:
    """Standard non-MoE FFN LoRA. Three projections (gate/up/down) as 2D
    PEFT-style tensors. Used by dense Qwen3 (0.6B/4B/etc.); MoE models use
    ``_build_moe_tensors`` instead."""
    out = {}
    base = f"{_PEFT_PREFIX}.{layer_idx}.mlp"
    for proj, in_dim, out_dim in (
        ("gate_proj", hidden, intermediate),
        ("up_proj", hidden, intermediate),
        ("down_proj", intermediate, hidden),
    ):
        out[f"{base}.{proj}.lora_A.weight"] = _kaiming_uniform_(torch.empty(rank, in_dim, dtype=dtype))
        out[f"{base}.{proj}.lora_B.weight"] = torch.zeros(out_dim, rank, dtype=dtype)
    return out


def _build_moe_tensors(
    layer_idx: int, rank: int, hidden: int, moe_intermediate: int, num_experts: int, dtype: torch.dtype
) -> dict:
    """Three MoE projections in shared_outer 3D layout. Per the format:
      w1/w3 (gate/up): A shared [1, r, hidden], B per-expert [E, moe_int, r]
      w2 (down):       A per-expert [E, r, moe_int], B shared [1, hidden, r]
    A gets kaiming-uniform fan-in init, B starts at zero."""
    out = {}
    base = f"{_PEFT_PREFIX}.{layer_idx}.mlp.experts"
    for proj, w_slot in _PROJ_TO_SGLANG_W.items():
        if proj == "down_proj":
            # input is moe_int (each expert's intermediate dim), output is hidden.
            a_shape = (num_experts, rank, moe_intermediate)
            b_shape = (1, hidden, rank)
        else:
            # gate or up: input is hidden, output is moe_int per expert.
            a_shape = (1, rank, hidden)
            b_shape = (num_experts, moe_intermediate, rank)
        a_tensor = torch.empty(*a_shape, dtype=dtype)
        # Kaiming over the per-slice [rank, in] portion to keep the same per-slice
        # variance as PEFT — works whether E=1 or E=num_experts.
        for e in range(a_shape[0]):
            _kaiming_uniform_(a_tensor[e])
        b_tensor = torch.zeros(*b_shape, dtype=dtype)
        out[f"{base}.{w_slot}.lora_A.weight"] = a_tensor
        out[f"{base}.{w_slot}.lora_B.weight"] = b_tensor
    return out


def _build_gdn_tensors(layer_idx: int, rank: int, gdn: dict, dtype: torch.dtype) -> dict:
    """GatedDeltaNet-mixer LoRA for hybrid linear-attention layers.

    Standard 2D PEFT tensors named for sglang's wrapped modules
    (``model.layers.N.linear_attn.{in_proj_qkvz,in_proj_ba,out_proj}``):
    A kaiming-uniform [rank, in_dim], B zeros [out_dim, rank]. The fused
    in_proj_qkvz / in_proj_ba are each LoRA-targeted as a single matrix
    (StackedColumnParallelLinearWithLoRA on the serving side).
    """
    out = {}
    base = f"{_PEFT_PREFIX}.{layer_idx}.linear_attn"
    for proj, in_dim, out_dim in (
        ("in_proj_qkvz", gdn["hidden"], gdn["qkvz_out"]),
        ("in_proj_ba", gdn["hidden"], gdn["ba_out"]),
        ("out_proj", gdn["out_proj_in"], gdn["hidden"]),
    ):
        out[f"{base}.{proj}.lora_A.weight"] = _kaiming_uniform_(torch.empty(rank, in_dim, dtype=dtype))
        out[f"{base}.{proj}.lora_B.weight"] = torch.zeros(out_dim, rank, dtype=dtype)
    return out


_ATTN_TARGETS = {"qkv_proj", "o_proj", "q_proj", "k_proj", "v_proj"}
_MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}
_GDN_TARGETS = ["in_proj_qkvz", "in_proj_ba", "out_proj"]


def build_adapter_state_dict(
    cfg,
    *,
    rank: int,
    dtype: torch.dtype,
    fused_qkv: bool,
    is_moe: bool,
    target_modules: list[str] | None = None,
) -> dict:
    """Walk every layer and emit the full state dict. Two branches:

      * MoE (Qwen3MoeForCausalLM family): attention (fused or split q/k/v) +
        MoE expert layers (w1/w2/w3) in sglang_shared_outer 3D format.
      * Dense (Qwen3ForCausalLM family): attention + standard 2D MLP
        (gate_proj, up_proj, down_proj). No expert dim.

    ``target_modules`` (optional) is used to gate which module families get
    emitted. If any of qkv_proj/o_proj/q_proj/k_proj/v_proj appear, attention
    tensors are written; if any of gate_proj/up_proj/down_proj appear, the
    MoE (or dense MLP) tensors are written; if the GDN-mixer targets
    (in_proj_qkvz/in_proj_ba/out_proj) appear, GatedDeltaNet tensors are
    written for the linear-attention layers. When None (default) we emit
    attention + MLP — preserves the prior fully-targeted behavior.

    Hybrid models (layer_types / full_attention_interval in the config):
    attention tensors are emitted ONLY on the full-attention layers
    (Qwen3.6-35B-A3B: 3, 7, ..., 39) and GDN tensors ONLY on the
    linear-attention layers — sglang has no wrapped attention modules on GDN
    layers (and vice versa) and would drop the weights.
    """
    dims = _attention_dims(cfg)
    gdn = _gdn_dims(cfg)
    full_attn_flags = _full_attention_layer_flags(cfg)
    num_layers = cfg.num_hidden_layers
    state_dict = {}
    targets = set(target_modules) if target_modules else None
    emit_attn = targets is None or bool(targets & _ATTN_TARGETS)
    emit_mlp = targets is None or bool(targets & _MLP_TARGETS)
    emit_gdn = targets is not None and bool(targets & set(_GDN_TARGETS))
    if emit_gdn and gdn is None:
        raise ValueError(
            "GDN-mixer targets requested but the model config has no "
            "linear-attention (GatedDeltaNet) dimensions"
        )
    for layer_idx in range(num_layers):
        if emit_attn and full_attn_flags[layer_idx]:
            state_dict.update(_build_attention_tensors(layer_idx, rank, dims, dtype, fused_qkv=fused_qkv))
        if emit_gdn and not full_attn_flags[layer_idx]:
            state_dict.update(_build_gdn_tensors(layer_idx, rank, gdn, dtype))
        if emit_mlp:
            if is_moe:
                state_dict.update(
                    _build_moe_tensors(
                        layer_idx,
                        rank=rank,
                        hidden=cfg.hidden_size,
                        moe_intermediate=cfg.moe_intermediate_size,
                        num_experts=cfg.num_experts,
                        dtype=dtype,
                    )
                )
            else:
                state_dict.update(
                    _build_dense_mlp_tensors(
                        layer_idx,
                        rank=rank,
                        hidden=cfg.hidden_size,
                        intermediate=cfg.intermediate_size,
                        dtype=dtype,
                    )
                )
    return state_dict


def write_adapter_config(
    output_dir: Path,
    *,
    model_name_or_path: str,
    rank: int,
    alpha: int,
    target_modules: list[str],
    is_moe: bool,
) -> Path:
    """Write the PEFT-style adapter_config.json. For MoE models we add xorl's
    two flags (``_sglang_lora_format`` + ``moe_hybrid_shared_lora``) so
    SGLang routes the file through the shared_outer 3D load path. Dense
    models use the standard PEFT path on SGLang."""
    cfg = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model_name_or_path,
        "peft_type": "LORA",
        "inference_mode": True,
        "fan_in_fan_out": False,
    }
    if is_moe:
        cfg["_sglang_lora_format"] = "shared_outer"
        cfg["moe_hybrid_shared_lora"] = True
    path = output_dir / "adapter_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--output-dir", required=True, help="Where to write adapter_model.safetensors + adapter_config.json")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="LoRA target modules. Auto-default if omitted: MoE → [qkv_proj, o_proj, gate_proj, up_proj, down_proj]; dense → [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]",
    )
    parser.add_argument(
        "--gdn-mixer",
        action="store_true",
        help="Also target the GatedDeltaNet projections (in_proj_qkvz, in_proj_ba, out_proj) of hybrid linear-attention layers (Qwen3.5/3.6 hybrid models only)",
    )
    parser.add_argument(
        "--fused-qkv",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether the base model fuses q/k/v into a single qkv_proj. 'auto' picks MoE→fused, dense→split, which matches Qwen3-30B-A3B (fused) vs Qwen3-0.6B-Base (split).",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--zero-a",
        action="store_true",
        help="Use zero-init for A too (DEBUG ONLY — ZORL b_only mode needs non-zero A to produce any candidate signal)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {args.model}...")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Some multimodal-capable configs put the dimensions under text_config.
    text_cfg = getattr(cfg, "text_config", cfg)

    is_moe = _detect_is_moe(text_cfg)
    if args.fused_qkv == "auto":
        fused_qkv = is_moe  # Qwen3MoE fuses qkv; dense Qwen3 keeps them split.
    else:
        fused_qkv = args.fused_qkv == "yes"

    if args.target_modules is None:
        if fused_qkv:
            target_modules = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = list(args.target_modules)
    if args.gdn_mixer:
        target_modules.extend(t for t in _GDN_TARGETS if t not in target_modules)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Detected: is_moe={is_moe} fused_qkv={fused_qkv} target_modules={target_modules}")
    full_attn_flags = _full_attention_layer_flags(text_cfg)
    if not all(full_attn_flags):
        full_layers = [i for i, f in enumerate(full_attn_flags) if f]
        print(
            f"Hybrid model: {len(full_layers)} full-attention layers {full_layers}; "
            f"{len(full_attn_flags) - len(full_layers)} linear-attention (GDN) layers; "
            f"gdn_mixer={args.gdn_mixer} gdn_dims={_gdn_dims(text_cfg)}"
        )
    intermediate = getattr(text_cfg, "intermediate_size", None)
    moe_int = getattr(text_cfg, "moe_intermediate_size", None)
    print(
        f"Model: hidden={text_cfg.hidden_size}, intermediate={intermediate}, moe_int={moe_int}, "
        f"layers={text_cfg.num_hidden_layers}, experts={getattr(text_cfg, 'num_experts', None)}, "
        f"n_heads={text_cfg.num_attention_heads}, n_kv={text_cfg.num_key_value_heads}, "
        f"head_dim={getattr(text_cfg, 'head_dim', None)}"
    )

    state_dict = build_adapter_state_dict(text_cfg, rank=args.rank, dtype=dtype, fused_qkv=fused_qkv, is_moe=is_moe, target_modules=target_modules)
    if args.zero_a:
        print("WARNING: --zero-a is on; ZORL b_only updates will produce no candidate signal.")
        for k, v in list(state_dict.items()):
            if "lora_A" in k:
                state_dict[k] = torch.zeros_like(v)

    weights_path = output_dir / "adapter_model.safetensors"
    save_file(state_dict, str(weights_path))
    print(f"Wrote {len(state_dict)} tensors to {weights_path}")

    cfg_path = write_adapter_config(
        output_dir,
        model_name_or_path=args.model,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=target_modules,
        is_moe=is_moe,
    )
    print(f"Wrote adapter config to {cfg_path}")


if __name__ == "__main__":
    main()
