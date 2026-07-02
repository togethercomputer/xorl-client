"""Repack a PEFT LoRA adapter with separate GatedDeltaNet projections into
SGLang's fused-GDN layout so linear-attention LoRA can be served.

Background
----------
The xorl trainer's GDN block (xorl/ops/linear_attention/layers/gated_deltanet.py)
exposes separate q_proj / k_proj / v_proj / o_proj linear-attention projections,
so PEFT exports carry tensors like

    base_model.model.model.layers.<L>.linear_attn.{q,k,v,o}_proj.lora_{A,B}.weight

SGLang's GDN block (sglang/srt/models/qwen3_5.py, Qwen3_5GatedDeltaNet) only has
the fused modules ``linear_attn.in_proj_qkvz`` (MergedColumnParallelLinear with
output_sizes=[key_dim, key_dim, value_dim, value_dim], i.e. PLAIN [q|k|v|z]
row-concat — see create_qkvz_proj and fix_query_key_value_ordering) and the
row-parallel ``linear_attn.out_proj``. The separate names never attach ("not
LoRA-wrapped ... skipped").

Repack (mathematically exact, no SGLang model changes):
  A_qkvz = rowstack(A_q, A_k, A_v)                      [3r, hidden]
  B_qkvz = zeros(2*key_dim + 2*value_dim, 3r)
     B_qkvz[0:key,            0:r  ] = B_q
     B_qkvz[key:2key,         r:2r ] = B_k
     B_qkvz[2key:2key+val,    2r:3r] = B_v
     (z rows stay zero)
  o_proj -> renamed out_proj (shapes already match: value_dim -> hidden),
     zero-padded from rank r to rank 3r.

Then  scaling*B_qkvz@A_qkvz == scaling*(pad_q(B_q A_q)+pad_k(B_k A_k)+pad_v(B_v A_v))
exactly (block structure; the zero blocks kill all cross terms), where the delta
lands on the fused projection output BEFORE the q/k/v/z split + conv — the same
place a full-finetune delta would land.

Rank/scaling: SGLang's LoRA mem pool slices load buffers with the single
adapter-level ``config.r`` (mem_pool.py: ``buffer[buffer_id, :lora_rank*c, :]``
for A / ``buffer[buffer_id, :, :lora_rank]`` for B) and applies one runtime
``scaling = lora_alpha / r`` per adapter — per-module heterogeneous ranks are
NOT supported. So ALL kept tensors are zero-padded to the uniform fused rank
R = 3r, and adapter_config gets r = 3r, lora_alpha = 3r * (old_alpha/old_r) so
the effective scaling is unchanged. Zero-padding is exact: padded A rows /
B cols contribute nothing, and kernels contract over per-adapter ``lora_ranks``.

in_proj_ba (trainer a_proj/b_proj, out=num_v_heads each; fused order [b|a]) is
NOT handled: xorl exports do not include those tensors (target list excludes
them). If they ever appear we fail loudly.

Validated 2026-07-02 (Qwen3.6-35B-A3B, sglang apanda-dev, TP2 pod + TP1 engine):
  * repacked adapters load with zero "not LoRA-wrapped ... skipped" warnings and
    change scored logprobs (previously bitwise-inert);
  * module-level (real base weights + real trained tensors through the real
    StackedColumnParallelLinearWithLoRA / RowParallelLinearWithLoRA + triton
    kernels): |y_lora - (y_base + scaling*B@A@x)| ~ 7e-6 (pure bf16 noise),
    z-slice delta exactly 0; zero-B adapter reproduces base BITWISE;
  * rank padding is exact: a self_attn-only adapter padded 16->48 (alpha 32->96)
    scores BITWISE identical to the original r16 adapter;
  * end-to-end vs a checkpoint with the same deltas folded in (bf16):
    agreement is limited by the FOLD's own bf16 rounding, not by serving.
    NOTE the noise floor: wd2qf policy-000015 GDN deltas (mean |d|~7e-6) are
    SMALLER than the base weights' bf16 half-ulp (~2.5e-5), so folding destroys
    ~80% of the delta and lora-vs-merged mean|dlogprob| is ~the full effect
    size (0.03). With deltas amplified x64 (rounding ~3.6% of delta) the
    lora-vs-merged residual drops to 6.8% of the effect (0.044 vs 0.64),
    scaling exactly as fold-rounding predicts.

Usage:
  python repack_gdn_lora.py --input <peft_adapter_dir> --output <out_dir> \
      [--gdn-only] [--keep-self-attn] [--check]

  --gdn-only        drop every non-GDN tensor (produces a pure fused-GDN adapter)
  --keep-self-attn  with --gdn-only, also keep (padded) self_attn.* tensors
  --check           verify fused-vs-separate delta equivalence on random input
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

GDN_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.linear_attn)\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj|a_proj|b_proj|g_proj)\."
    r"lora_(?P<ab>[AB])\.weight$"
)


def repack(
    tensors: dict[str, torch.Tensor],
    key_dim: int,
    value_dim: int,
    gdn_only: bool = False,
    keep_self_attn: bool = False,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Returns (new_tensors, old_rank, new_rank)."""
    # ---- bucket GDN tensors per layer ----
    gdn: dict[str, dict[str, torch.Tensor]] = {}  # prefix -> {"q_A": ...}
    passthrough: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        m = GDN_RE.match(name)
        if m is None:
            passthrough[name] = t
            continue
        proj = m.group("proj")
        if proj in ("a_proj", "b_proj", "g_proj"):
            raise NotImplementedError(
                f"{name}: in_proj_ba/in_proj_z repack not implemented (xorl "
                "exports are not expected to carry a/b/g_proj LoRA)."
            )
        gdn.setdefault(m.group("prefix"), {})[f"{proj[0]}_{m.group('ab')}"] = t

    if not gdn:
        raise ValueError("no linear_attn.{q,k,v,o}_proj LoRA tensors found")

    # ---- infer ranks ----
    some = next(iter(gdn.values()))
    r = some["q_A"].shape[0]
    R = 3 * r  # fused rank
    hidden = some["q_A"].shape[1]

    out: dict[str, torch.Tensor] = {}

    def pad_A(A: torch.Tensor) -> torch.Tensor:
        assert A.shape[0] == r, f"rank mismatch: {A.shape} vs r={r}"
        return torch.cat([A, torch.zeros(R - r, A.shape[1], dtype=A.dtype)], dim=0)

    def pad_B(B: torch.Tensor) -> torch.Tensor:
        assert B.shape[1] == r, f"rank mismatch: {B.shape} vs r={r}"
        return torch.cat([B, torch.zeros(B.shape[0], R - r, dtype=B.dtype)], dim=1)

    # ---- fused GDN tensors ----
    for prefix, parts in sorted(gdn.items()):
        missing = {k for k in ("q_A", "q_B", "k_A", "k_B", "v_A", "v_B")} - set(parts)
        if missing:
            raise ValueError(f"{prefix}: missing GDN LoRA parts {sorted(missing)}")
        A_q, A_k, A_v = parts["q_A"], parts["k_A"], parts["v_A"]
        B_q, B_k, B_v = parts["q_B"], parts["k_B"], parts["v_B"]
        assert A_q.shape == (r, hidden) and A_k.shape == (r, hidden)
        assert A_v.shape == (r, hidden)
        assert B_q.shape == (key_dim, r), f"{prefix} B_q {B_q.shape}"
        assert B_k.shape == (key_dim, r), f"{prefix} B_k {B_k.shape}"
        assert B_v.shape == (value_dim, r), f"{prefix} B_v {B_v.shape}"

        A_f = torch.cat([A_q, A_k, A_v], dim=0)  # [3r, hidden]
        fused_out = 2 * key_dim + 2 * value_dim
        B_f = torch.zeros(fused_out, R, dtype=B_q.dtype)
        B_f[0:key_dim, 0:r] = B_q
        B_f[key_dim : 2 * key_dim, r : 2 * r] = B_k
        B_f[2 * key_dim : 2 * key_dim + value_dim, 2 * r : 3 * r] = B_v
        # z rows [2*key_dim+value_dim : fused_out] stay zero
        out[f"{prefix}.in_proj_qkvz.lora_A.weight"] = A_f.contiguous()
        out[f"{prefix}.in_proj_qkvz.lora_B.weight"] = B_f.contiguous()

        if "o_A" in parts:
            A_o, B_o = parts["o_A"], parts["o_B"]
            assert A_o.shape == (r, value_dim), f"{prefix} A_o {A_o.shape}"
            assert B_o.shape == (hidden, r), f"{prefix} B_o {B_o.shape}"
            out[f"{prefix}.out_proj.lora_A.weight"] = pad_A(A_o).contiguous()
            out[f"{prefix}.out_proj.lora_B.weight"] = pad_B(B_o).contiguous()

    # ---- non-GDN tensors: pad to uniform rank R ----
    for name, t in passthrough.items():
        if gdn_only and not (keep_self_attn and ".self_attn." in name):
            continue
        if ".lora_A." in name or "lora_embedding_A" in name:
            out[name] = pad_A(t).contiguous()
        elif ".lora_B." in name or "lora_embedding_B" in name:
            out[name] = pad_B(t).contiguous()
        else:
            out[name] = t

    return out, r, R


def check_equivalence(
    tensors_in: dict[str, torch.Tensor],
    tensors_out: dict[str, torch.Tensor],
    key_dim: int,
    value_dim: int,
    n_layers_check: int = 3,
) -> None:
    torch.manual_seed(0)
    prefixes = sorted(
        {GDN_RE.match(n).group("prefix") for n in tensors_in if GDN_RE.match(n)}
    )[:n_layers_check]
    hidden = tensors_in[f"{prefixes[0]}.q_proj.lora_A.weight"].shape[1]
    x = torch.randn(5, hidden, dtype=torch.float64)
    worst = 0.0
    for p in prefixes:
        A_f = tensors_out[f"{p}.in_proj_qkvz.lora_A.weight"].double()
        B_f = tensors_out[f"{p}.in_proj_qkvz.lora_B.weight"].double()
        fused = x @ A_f.T @ B_f.T  # [5, 2k+2v]
        ref = torch.zeros_like(fused)
        for proj, lo, hi in (
            ("q", 0, key_dim),
            ("k", key_dim, 2 * key_dim),
            ("v", 2 * key_dim, 2 * key_dim + value_dim),
        ):
            A = tensors_in[f"{p}.{proj}_proj.lora_A.weight"].double()
            B = tensors_in[f"{p}.{proj}_proj.lora_B.weight"].double()
            ref[:, lo:hi] = x @ A.T @ B.T
        err = (fused - ref).abs().max().item()
        worst = max(worst, err)
        # out_proj pad check
        if f"{p}.out_proj.lora_A.weight" in tensors_out:
            xo = torch.randn(5, value_dim, dtype=torch.float64)
            A_o = tensors_in[f"{p}.o_proj.lora_A.weight"].double()
            B_o = tensors_in[f"{p}.o_proj.lora_B.weight"].double()
            A_p = tensors_out[f"{p}.out_proj.lora_A.weight"].double()
            B_p = tensors_out[f"{p}.out_proj.lora_B.weight"].double()
            err_o = ((xo @ A_p.T @ B_p.T) - (xo @ A_o.T @ B_o.T)).abs().max().item()
            worst = max(worst, err_o)
    print(f"[check] fused-vs-separate max |delta| over {len(prefixes)} layers: {worst:.3e}")
    # The mapping is exact in exact arithmetic; the only residual is fp64 GEMM
    # accumulation-order noise (K=3r fused vs K=r separate contraction).
    assert worst < 1e-12, f"repack equivalence check failed: {worst}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="PEFT adapter dir (separate GDN names)")
    ap.add_argument("--output", required=True, help="output adapter dir (fused GDN names)")
    ap.add_argument("--gdn-only", action="store_true", help="drop all non-GDN tensors")
    ap.add_argument(
        "--keep-self-attn",
        action="store_true",
        help="with --gdn-only, also keep (rank-padded) self_attn tensors",
    )
    ap.add_argument("--check", action="store_true", help="verify repack equivalence")
    ap.add_argument("--key-dim", type=int, default=2048)
    ap.add_argument("--value-dim", type=int, default=4096)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    tensors = load_file(in_dir / "adapter_model.safetensors")
    cfg = json.loads((in_dir / "adapter_config.json").read_text())

    new_tensors, r, R = repack(
        tensors,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        gdn_only=args.gdn_only,
        keep_self_attn=args.keep_self_attn,
    )
    if args.check:
        check_equivalence(tensors, new_tensors, args.key_dim, args.value_dim)

    # ---- adapter_config: uniform rank R, preserve effective scaling ----
    scaling = cfg["lora_alpha"] / cfg["r"]
    cfg["r"] = R
    cfg["lora_alpha"] = scaling * R
    if cfg["lora_alpha"] == int(cfg["lora_alpha"]):
        cfg["lora_alpha"] = int(cfg["lora_alpha"])

    # target_modules: leaf names present in the output tensor set.
    # For SGLang these normalize to themselves for the GDN names
    # (get_normalized_target_modules has no mapping for in_proj_qkvz/out_proj),
    # so the pool wraps model.layers.<L>.linear_attn.{in_proj_qkvz,out_proj}
    # iff these leaf names are in the (server ∪ adapter) target set.
    leaves = sorted({n.split(".lora_")[0].rsplit(".", 1)[-1] for n in new_tensors})
    cfg["target_modules"] = leaves

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_tensors, out_dir / "adapter_model.safetensors")
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for extra in in_dir.iterdir():
        if extra.name not in ("adapter_model.safetensors", "adapter_config.json"):
            if extra.is_file():
                shutil.copy2(extra, out_dir / extra.name)

    n_gdn = sum(1 for n in new_tensors if ".linear_attn." in n)
    print(
        f"wrote {out_dir}: {len(new_tensors)} tensors ({n_gdn} fused-GDN), "
        f"r {r} -> {R}, alpha -> {cfg['lora_alpha']} (scaling {scaling}), "
        f"target_modules={leaves}"
    )


if __name__ == "__main__":
    main()
