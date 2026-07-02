"""Fold GDN (linear-attention) LoRA deltas of a PEFT adapter directly into a
copy of the Qwen3.6-35B-A3B base checkpoint (W += scaling * B @ A), producing a
merged checkpoint that serves WITHOUT LoRA. Used as ground truth to validate
the exactness of SGLang's fused-GDN LoRA serving path (repack_gdn_lora.py).

Checkpoint storage layout (model.safetensors.index.json of the base snapshot):
  model.language_model.layers.<L>.linear_attn.in_proj_qkv.weight  [8192, 2048]
      rows [0:key_dim]                       = q   (key_dim = 2048)
      rows [key_dim:2*key_dim]               = k
      rows [2*key_dim:2*key_dim+value_dim]   = v   (value_dim = 4096)
  model.language_model.layers.<L>.linear_attn.in_proj_z.weight    (untouched)
  model.language_model.layers.<L>.linear_attn.out_proj.weight     [2048, 4096]

(the same [q|k|v] row order the trainer uses when splitting in_proj_qkv into
q/k/v_proj — xorl/models/transformers/qwen3_5_shared.py — and that SGLang's
fused in_proj_qkvz assumes: plain concat, z appended last from in_proj_z.)

Adapter names consumed (the trainer's separate-projection PEFT export):
  base_model.model.model.layers.<L>.linear_attn.{q,k,v,o}_proj.lora_{A,B}.weight

Unaffected shard files and aux files are symlinked into the output dir;
affected shards are rewritten (delta computed in fp32, cast back to bf16).

Usage:
  python fold_gdn_lora_into_ckpt.py --adapter <peft_dir> --base <snapshot_dir> \
      --output <merged_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GDN_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj)\.lora_(?P<ab>[AB])\.weight$"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True, help="base checkpoint snapshot dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--key-dim", type=int, default=2048)
    ap.add_argument("--value-dim", type=int, default=4096)
    args = ap.parse_args()

    adapter_dir, base_dir, out_dir = Path(args.adapter), Path(args.base), Path(args.output)
    key_dim, value_dim = args.key_dim, args.value_dim

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"adapter r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}")

    # ---- collect GDN LoRA tensors per (layer, proj) ----
    lora: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as f:
        for name in f.keys():
            m = GDN_RE.match(name)
            if m:
                lora[(int(m.group("layer")), m.group("proj"))][m.group("ab")] = (
                    f.get_tensor(name)
                )
    layers = sorted({k[0] for k in lora})
    print(f"folding GDN LoRA for {len(layers)} layers: {layers}")
    assert lora, "no GDN LoRA tensors found in adapter"

    def delta(layer: int, proj: str) -> torch.Tensor:
        d = lora[(layer, proj)]
        return scaling * (d["B"].float() @ d["A"].float())

    # ---- map checkpoint tensor name -> update fn ----
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    ckpt_re = re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.linear_attn\."
        r"(?P<t>in_proj_qkv|out_proj)\.weight$"
    )
    updates: dict[str, tuple[int, str]] = {}  # ckpt key -> (layer, kind)
    for name in weight_map:
        m = ckpt_re.match(name)
        if m and int(m.group("layer")) in set(layers):
            updates[name] = (int(m.group("layer")), m.group("t"))

    n_qkv = sum(1 for v in updates.values() if v[1] == "in_proj_qkv")
    n_out = sum(1 for v in updates.values() if v[1] == "out_proj")
    assert n_qkv == len(layers) and n_out == len(layers), (n_qkv, n_out, len(layers))

    affected_shards: dict[str, list[str]] = defaultdict(list)
    for name in updates:
        affected_shards[weight_map[name]].append(name)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- symlink every base file; rewrite affected shards ----
    for f in sorted(base_dir.iterdir()):
        dst = out_dir / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if f.name in affected_shards:
            continue  # rewritten below
        dst.symlink_to(f.resolve())

    print(f"rewriting {len(affected_shards)}/{len(set(weight_map.values()))} shards")
    for shard, names in sorted(affected_shards.items()):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(base_dir / shard, "pt") as f:
            metadata = f.metadata()
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
        for name in names:
            layer, kind = updates[name]
            w = tensors[name]
            if kind == "in_proj_qkv":
                assert w.shape == (2 * key_dim + value_dim, w.shape[1]), w.shape
                new = w.float()
                new[0:key_dim] += delta(layer, "q_proj")
                new[key_dim : 2 * key_dim] += delta(layer, "k_proj")
                new[2 * key_dim : 2 * key_dim + value_dim] += delta(layer, "v_proj")
            else:  # out_proj
                new = w.float() + delta(layer, "o_proj")
            tensors[name] = new.to(w.dtype)
        save_file(tensors, out_dir / shard, metadata=metadata)
        print(f"  {shard}: folded {len(names)} tensors")

    print(f"merged checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
