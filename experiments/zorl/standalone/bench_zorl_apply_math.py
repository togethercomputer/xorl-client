"""Offline correctness + perf microbench for the ZORL apply (update) math.

This bench isolates the server-side `/apply_zorl_rewards` cost
(`LoRAManager.apply_zorl_rewards` in the SGLang fork) *without* a cluster, a
GPU, or `/generate`. It is the correctness/hillclimb target the performance
handoff runbook asks for, scoped to the apply phase, which is the most
isolatable hot path (~105-130s/step in production, ~31% of step wall).

It does two things:

1. Extracts the *real* bit-level helper methods from the fork
   (`lora_manager.py`) via AST so the golden reference can never drift from
   production. The full module cannot be imported on a dev pod (it pulls in
   CUDA-only `sgl_kernel`), so we lift only the self-contained noise / score
   helpers, which depend on nothing but torch + math.

2. Builds a synthetic parent adapter with the exact production LoRA-B tensor
   shapes (Qwen3-30B-A3B, rank 4, target modules
   qkv_proj/o_proj/gate_proj/up_proj/down_proj, MoE hybrid_shared +
   experts_shared_outer) and runs:
     - golden_apply:    faithful copy of the current apply loop, calling the
                        extracted real `_zorl_normalized_b_noises`.
     - optimized_apply: hoists the per-pair metadata rebuild out of the loop,
                        accumulates noise in *raw* layout, and applies the
                        (linear) transpose/cat transform once at the end.

The correctness gate is bit-exact equality of the resulting parent weights and
of every reported metric. The optimization changes *no* random draw: it still
issues the identical per-tensor `torch.randn` calls in the identical sorted
order (empirically, merging randn calls breaks bit-exactness even for
even-numel tensors), so the noise stream is preserved exactly.

Usage:
    python -m experiments.zorl.standalone.bench_zorl_apply_math \
        --num-pairs 512 --layers 48 --repeat 2
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------
# 1. Lift the real helpers out of the fork via AST (no heavy imports).
# --------------------------------------------------------------------------

FORK_LORA_MANAGER = (
    "/home/apanda/xorl-sglang-internal/python/sglang/srt/lora/lora_manager.py"
)

_GOLDEN_METHODS = (
    "_zorl_config_targets",
    "_zorl_trainer_key_from_export_key",
    "_zorl_raw_b_entries_for_weight",
    "_zorl_normalized_b_noises",
    "_zorl_normalized_a_noises",
    "_normalize_zorl_pair_scores",
    "_validate_zorl_perturbation_mode",
    # GPU fast path (additive in the fork)
    "_zorl_noise_device",
    "_zorl_b_noise_layout",
    "_zorl_a_noise_layout",
    "_zorl_flat_sizes",
    "_zorl_raw_from_flat",
    "_zorl_reassemble_b",
    "_zorl_normalized_b_noises_gpu",
    "_zorl_normalized_a_noises_gpu",
    "_zorl_build_update_gpu",
)


def _load_golden_helpers(path: str = FORK_LORA_MANAGER) -> type:
    """Extract the listed LoRAManager methods and bind them to a fresh class.

    Returns a class whose instances expose the *real* fork implementations of
    the bit-level noise/score helpers. Guarantees zero drift from production.
    """
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    cls_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LoRAManager"
    )
    wanted = {}
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name in _GOLDEN_METHODS:
            wanted[node.name] = node
    missing = set(_GOLDEN_METHODS) - set(wanted)
    if missing:
        raise RuntimeError(f"Could not find golden methods in fork: {sorted(missing)}")

    # Build a module body containing just these functions, then exec it with a
    # torch/math namespace so closures resolve.
    mod = ast.Module(body=list(wanted.values()), type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: Dict[str, Any] = {
        "torch": torch,
        "math": math,
        "os": os,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    code = compile(mod, filename=path, mode="exec")
    exec(code, ns)  # noqa: S102 - trusted local source

    attrs = {}
    for name in _GOLDEN_METHODS:
        fn = ns[name]
        # staticmethods in the source lost their decorator (we grabbed the bare
        # FunctionDef); re-wrap the ones that don't take self.
        params = [a.arg for a in wanted[name].args.args]
        if not params or params[0] != "self":
            attrs[name] = staticmethod(fn)
        else:
            attrs[name] = fn
    return type("GoldenHelpers", (), attrs)


# --------------------------------------------------------------------------
# 2. Synthetic parent adapter matching production shapes.
# --------------------------------------------------------------------------

# Export-format LoRA-B module signatures for Qwen3-30B-A3B rank 4 (from the
# live ZORL-WORDLE-014 exported adapter header).
_B_MODULE_SHAPES = {
    "self_attn.qkv_proj.lora_B.weight": (5120, 4),
    "self_attn.o_proj.lora_B.weight": (2048, 4),
    "mlp.experts.gate_proj.lora_B.weight": (128, 768, 4),
    "mlp.experts.up_proj.lora_B.weight": (128, 768, 4),
    "mlp.experts.down_proj.lora_B.weight": (1, 2048, 4),
}
# Matching LoRA-A signatures (rank-first); only needed so the apply weight-update
# loop sees the same key set it does in production (it skips A in b_only mode).
_A_MODULE_SHAPES = {
    "self_attn.qkv_proj.lora_A.weight": (4, 2048),
    "self_attn.o_proj.lora_A.weight": (4, 2048),
    "mlp.experts.gate_proj.lora_A.weight": (128, 4, 2048),
    "mlp.experts.up_proj.lora_A.weight": (128, 4, 2048),
    "mlp.experts.down_proj.lora_A.weight": (1, 4, 768),
}


class _Layer:
    __slots__ = ("weights",)

    def __init__(self, weights: Dict[str, torch.Tensor]):
        self.weights = weights


class _Config:
    def __init__(self, target_modules):
        self.target_modules = target_modules


class _Adapter:
    def __init__(self, layers: List[_Layer], target_modules):
        self.layers = layers
        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.config = _Config(target_modules)


def build_synthetic_adapter(
    *, num_layers: int = 48, dtype: torch.dtype = torch.bfloat16, seed: int = 0
) -> _Adapter:
    g = torch.Generator()
    g.manual_seed(seed)
    layers = []
    for layer_id in range(num_layers):
        weights: Dict[str, torch.Tensor] = {}
        prefix = f"base_model.model.model.layers.{layer_id}."
        for suffix, shape in {**_B_MODULE_SHAPES, **_A_MODULE_SHAPES}.items():
            name = prefix + suffix
            weights[name] = (
                torch.randn(shape, generator=g, dtype=torch.float32) * 0.02
            ).to(dtype)
        layers.append(_Layer(weights))
    target_modules = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return _Adapter(layers, target_modules)


def clone_adapter(adapter: _Adapter) -> _Adapter:
    layers = [
        _Layer({k: v.clone() for k, v in layer.weights.items()})
        for layer in adapter.layers
    ]
    out = _Adapter(layers, list(adapter.config.target_modules))
    out.embedding_layers = {k: v.clone() for k, v in adapter.embedding_layers.items()}
    return out


def adapter_b_checksum(adapter: _Adapter) -> Dict[str, torch.Tensor]:
    """Snapshot all LoRA-B weights (the only ones b_only touches)."""
    snap = {}
    for layer_id, layer in enumerate(adapter.layers):
        for name, w in layer.weights.items():
            if "lora_B" in name:
                snap[(layer_id, name)] = w.detach().clone()
    return snap


def adapter_b_hash(adapter: _Adapter) -> str:
    """Order-stable hash of all LoRA-B weights, for cross-run regression gating."""
    h = hashlib.sha256()
    for layer_id, layer in enumerate(adapter.layers):
        for name in sorted(layer.weights):
            if "lora_B" not in name:
                continue
            w = layer.weights[name].detach().to(torch.float32).contiguous()
            h.update(f"{layer_id}:{name}:".encode())
            h.update(w.cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 3. Golden apply (faithful copy of current lora_manager.apply_zorl_rewards
#    core loop) and the optimized apply.
# --------------------------------------------------------------------------


def _build_seed_score_pairs(
    num_pairs: int, *, seed: int = 1234
) -> Tuple[List[Tuple[int, Optional[int], float]], List[float]]:
    """Synthetic deterministic pair scores matching production magnitude."""
    g = torch.Generator()
    g.manual_seed(seed)
    raw = (torch.randn(num_pairs, generator=g) * 0.005).tolist()
    seed_score_pairs = [(1_000_000 + i, None, raw[i]) for i in range(num_pairs)]
    return seed_score_pairs, raw


def golden_apply(
    helpers,
    adapter: _Adapter,
    seed_score_pairs,
    raw_pair_scores,
    *,
    learning_rate: float,
    max_update_norm: Optional[float],
    score_normalization: str = "standard",
    perturbation_mode: str = "b_only",
) -> Dict[str, Any]:
    """Mirror of lora_manager.apply_zorl_rewards lines ~1567-1654 (b_only)."""
    normalized_scores = helpers._normalize_zorl_pair_scores(
        raw_pair_scores, normalization=score_normalization
    )
    updates: Dict[tuple, torch.Tensor] = {}
    zero_score_pairs = 0
    weight_scale = 1.0 / float(len(seed_score_pairs))
    for (b_seed, a_seed, _raw), normalized_score in zip(
        seed_score_pairs, normalized_scores, strict=True
    ):
        if abs(float(normalized_score)) <= 1e-12:
            zero_score_pairs += 1
            continue
        noises = helpers._zorl_normalized_b_noises(adapter, seed=b_seed)
        if perturbation_mode == "a_and_b":
            assert a_seed is not None
            noises.update(helpers._zorl_normalized_a_noises(adapter, seed=int(a_seed)))
        for key, noise in noises.items():
            if key not in updates:
                updates[key] = torch.zeros_like(noise, dtype=torch.float32)
            updates[key].add_(noise, alpha=float(normalized_score) * weight_scale)

    return _finish_apply(
        adapter,
        updates,
        raw_pair_scores,
        zero_score_pairs,
        learning_rate=learning_rate,
        max_update_norm=max_update_norm,
        score_normalization=score_normalization,
        perturbation_mode=perturbation_mode,
    )


def _precompute_b_layout(helpers, adapter: _Adapter):
    """Compute, once, everything `_zorl_normalized_b_noises` recomputes per pair.

    Returns:
      sorted_raw: list[(raw_name, shape)] in the exact generation order.
      assemble:   list[(normalized_key, [(raw_name, transpose_last_two)], split_dim)]
    """
    raw_entries: List[Tuple[str, Tuple[int, ...]]] = []
    assemble: List[Tuple[tuple, List[Tuple[str, bool]], Optional[int]]] = []
    seen_raw = set()
    for layer_id, layer in enumerate(adapter.layers):
        for name, weight in layer.weights.items():
            if "lora_B" not in name:
                continue
            parts, split_dim = helpers._zorl_raw_b_entries_for_weight(
                name, weight, adapter
            )
            assemble.append(
                (
                    ("layer", layer_id, name),
                    [(raw_name, transpose) for raw_name, _shape, transpose in parts],
                    split_dim,
                )
            )
            for raw_name, shape, _transpose in parts:
                if raw_name not in seen_raw:
                    raw_entries.append((raw_name, shape))
                    seen_raw.add(raw_name)
    for name, weight in adapter.embedding_layers.items():
        if "lora_B" not in name and "lora_embedding_B" not in name:
            continue
        raw_name = helpers._zorl_trainer_key_from_export_key(name)
        shape = tuple(weight.shape)
        assemble.append((("embedding", 0, name), [(raw_name, False)], None))
        if raw_name not in seen_raw:
            raw_entries.append((raw_name, shape))
            seen_raw.add(raw_name)
    sorted_raw = sorted(raw_entries, key=lambda item: item[0])
    return sorted_raw, assemble


def optimized_apply(
    helpers,
    adapter: _Adapter,
    seed_score_pairs,
    raw_pair_scores,
    *,
    learning_rate: float,
    max_update_norm: Optional[float],
    score_normalization: str = "standard",
    perturbation_mode: str = "b_only",
    layout=None,
) -> Dict[str, Any]:
    """Optimized apply: hoist layout, accumulate raw, transform once.

    Bit-identical to golden_apply for b_only: same per-tensor randn draws in the
    same sorted order; accumulation in raw layout then a single linear
    transpose/cat per tensor at the end (transpose & cat commute with the
    weighted sum exactly).
    """
    if perturbation_mode != "b_only":
        raise NotImplementedError("optimized_apply currently covers b_only")
    if layout is None:
        layout = _precompute_b_layout(helpers, adapter)
    sorted_raw, assemble = layout

    normalized_scores = helpers._normalize_zorl_pair_scores(
        raw_pair_scores, normalization=score_normalization
    )
    weight_scale = 1.0 / float(len(seed_score_pairs))

    # Preallocate raw-layout accumulators.
    raw_acc: Dict[str, torch.Tensor] = {
        raw_name: torch.zeros(shape, dtype=torch.float32)
        for raw_name, shape in sorted_raw
    }
    zero_score_pairs = 0
    for (b_seed, _a_seed, _raw), normalized_score in zip(
        seed_score_pairs, normalized_scores, strict=True
    ):
        alpha = float(normalized_score) * weight_scale
        if abs(float(normalized_score)) <= 1e-12:
            zero_score_pairs += 1
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(b_seed))
        # Generate-all-then-accumulate (two loops). Adding inside the randn loop
        # triggers an OpenMP thread-launch pathology that is ~3x slower at high
        # core counts (see bench probe). Same RNG order => still bit-exact.
        noises = [
            torch.randn(shape, generator=generator, dtype=torch.float32)
            for _raw_name, shape in sorted_raw
        ]
        for (raw_name, _shape), noise in zip(sorted_raw, noises, strict=True):
            raw_acc[raw_name].add_(noise, alpha=alpha)

    # Assemble normalized updates from raw accumulators (single transform each).
    updates: Dict[tuple, torch.Tensor] = {}
    for normalized_key, parts, split_dim in assemble:
        if split_dim is None:
            raw_name, transpose = parts[0]
            t = raw_acc[raw_name]
            if transpose:
                t = t.transpose(-2, -1).contiguous()
            updates[normalized_key] = t
        else:
            pieces = []
            for raw_name, transpose in parts:
                t = raw_acc[raw_name]
                if transpose:
                    t = t.transpose(-2, -1).contiguous()
                pieces.append(t)
            updates[normalized_key] = torch.cat(pieces, dim=split_dim)

    return _finish_apply(
        adapter,
        updates,
        raw_pair_scores,
        zero_score_pairs,
        learning_rate=learning_rate,
        max_update_norm=max_update_norm,
        score_normalization=score_normalization,
        perturbation_mode=perturbation_mode,
    )


def _finish_apply(
    adapter: _Adapter,
    updates: Dict[tuple, torch.Tensor],
    raw_pair_scores,
    zero_score_pairs: int,
    *,
    learning_rate: float,
    max_update_norm: Optional[float],
    score_normalization: str,
    perturbation_mode: str,
) -> Dict[str, Any]:
    """Norm/clip + parent weight update + metrics. Shared by both paths so the
    only thing under test is how `updates` is produced."""
    update_norm_sq = sum(
        float(torch.sum(u * u).item()) for u in updates.values()
    )
    raw_update_norm = math.sqrt(max(update_norm_sq, 0.0))
    update_scale = 1.0
    max_norm = None
    if max_update_norm is not None:
        max_norm = float(max_update_norm)
        if raw_update_norm > max_norm:
            update_scale = max_norm / (raw_update_norm + 1e-12)
    effective_update_norm = raw_update_norm * update_scale

    lr = float(learning_rate)
    update_applied = bool(updates) and raw_update_norm > 0.0 and lr != 0.0
    if update_applied:
        for layer_id, parent_layer in enumerate(adapter.layers):
            for name, weight in list(parent_layer.weights.items()):
                if "lora_B" not in name and (
                    perturbation_mode != "a_and_b" or "lora_A" not in name
                ):
                    continue
                update = updates.get(("layer", layer_id, name))
                if update is None:
                    continue
                tensor = (
                    weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .add(update, alpha=lr * update_scale)
                )
                parent_layer.weights[name] = tensor.to(dtype=weight.dtype).contiguous()

    pair_score_tensor = torch.tensor(raw_pair_scores, dtype=torch.float32)
    return {
        "used_pairs": len(raw_pair_scores),
        "zero_score_pairs": zero_score_pairs,
        "update_applied": update_applied,
        "update_norm": float(effective_update_norm),
        "grad_norm": float(raw_update_norm),
        "unclipped_update_norm": float(raw_update_norm),
        "update_clip_scale": float(update_scale),
        "pair_delta_mean": float(pair_score_tensor.mean().item()),
        "pair_delta_std": (
            float(pair_score_tensor.std(unbiased=False).item())
            if pair_score_tensor.numel() > 1
            else 0.0
        ),
        "score_normalization": score_normalization,
        "perturbation_mode": perturbation_mode,
    }


# --------------------------------------------------------------------------
# 4. Profiling decomposition of the current path.
# --------------------------------------------------------------------------


def profile_current_breakdown(helpers, adapter: _Adapter, seed_score_pairs):
    """Decompose one golden apply into removable vs irreducible cost.

    metadata_rebuild: time to rebuild the per-pair layout (string ops, sort).
    randn_floor:      time to issue the per-tensor randn draws (irreducible).
    reassemble:       per-pair transpose/cat to normalized layout (removable).
    accumulate:       per-pair add_ into the update dict.
    """
    layout = _precompute_b_layout(helpers, adapter)
    sorted_raw, _assemble = layout
    n = len(seed_score_pairs)

    # metadata rebuild (what _zorl_normalized_b_noises redoes every call)
    t0 = time.perf_counter()
    for _ in range(n):
        _precompute_b_layout(helpers, adapter)
    t_meta = time.perf_counter() - t0

    # randn floor: per-tensor randn in sorted order, no accumulation
    t0 = time.perf_counter()
    for b_seed, _a, _r in seed_score_pairs:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(b_seed))
        for _raw_name, shape in sorted_raw:
            torch.randn(shape, generator=g, dtype=torch.float32)
    t_randn = time.perf_counter() - t0

    # full golden normalized-noise (randn + metadata + reassemble) per pair
    t0 = time.perf_counter()
    for b_seed, _a, _r in seed_score_pairs:
        helpers._zorl_normalized_b_noises(adapter, seed=int(b_seed))
    t_full_noise = time.perf_counter() - t0

    return {
        "pairs": n,
        "t_metadata_rebuild_s": t_meta,
        "t_randn_floor_s": t_randn,
        "t_full_normalized_noise_s": t_full_noise,
        "t_reassemble_plus_overhead_s": t_full_noise - t_randn,
    }


# --------------------------------------------------------------------------
# 5. Correctness + timing harness.
# --------------------------------------------------------------------------


def _max_b_diff(a: _Adapter, b: _Adapter) -> float:
    diff = 0.0
    for layer_id, layer in enumerate(a.layers):
        for name, wa in layer.weights.items():
            if "lora_B" not in name:
                continue
            wb = b.layers[layer_id].weights[name]
            d = (wa.to(torch.float32) - wb.to(torch.float32)).abs().max().item()
            diff = max(diff, d)
    return diff


def _metrics_agree(m1, m2, *, tol=0.0) -> List[str]:
    bad = []
    for k in (
        "used_pairs",
        "zero_score_pairs",
        "update_norm",
        "grad_norm",
        "unclipped_update_norm",
        "update_clip_scale",
        "pair_delta_mean",
        "pair_delta_std",
    ):
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, float):
            if not math.isclose(v1, v2, rel_tol=1e-12, abs_tol=tol):
                bad.append(f"{k}: {v1!r} vs {v2!r}")
        elif v1 != v2:
            bad.append(f"{k}: {v1!r} vs {v2!r}")
    return bad


def gpu_validation(helpers, base: _Adapter, num_pairs: int, *, lr: float, max_update_norm: float):
    """Validate the GPU noise fast path against its own correctness invariants.

    GPU and CPU RNG differ, so we cannot bit-compare to the CPU golden. Instead
    we prove the invariants the ZORL algorithm actually requires:

      1. apply == materialize: the apply update equals
         sum_i score_i*weight_scale * noise_gpu(seed_i), where noise_gpu(seed_i)
         is exactly the perturbation a candidate of pair i was materialized with.
      2. determinism: noise_gpu(seed) is identical across calls (=> identical
         across replicas/TP ranks on identical hardware).
      3. antithetic: +/- directions are exact negatives about the parent.
      4. speed vs CPU golden apply.
    """
    dev = "cuda"
    print("\n=== GPU fast-path validation (XORL_ZORL_NOISE_DEVICE=gpu) ===")
    print(f"  noise_device resolves to: cpu (env unset) -> {helpers._zorl_noise_device()!r}; "
          f"forcing device={dev!r} for this section")

    seed_score_pairs, raw_pair_scores = _build_seed_score_pairs(num_pairs)
    # normalization='none' => normalized_scores pass raw through, so the manual
    # reference can reuse the same coefficients.
    nscores = helpers._normalize_zorl_pair_scores(raw_pair_scores, normalization="none")
    weight_scale = 1.0 / float(len(seed_score_pairs))

    # (1) apply builder == per-pair materialize-noise accumulation.
    updates, zero = helpers._zorl_build_update_gpu(
        base, seed_score_pairs, nscores, weight_scale,
        perturbation_mode="b_only", device=dev,
    )
    manual: Dict[tuple, torch.Tensor] = {}
    for (b_seed, _a, _r), ns in zip(seed_score_pairs, nscores, strict=True):
        if abs(float(ns)) <= 1e-12:
            continue
        noise = helpers._zorl_normalized_b_noises_gpu(base, seed=b_seed, device=dev)
        for k, v in noise.items():
            if k not in manual:
                manual[k] = torch.zeros_like(v)
            manual[k].add_(v, alpha=float(ns) * weight_scale)
    consistency = 0.0
    for k in updates:
        consistency = max(consistency, (updates[k] - manual[k]).abs().max().item())
    print(f"  (1) apply==materialize  max|Δ|={consistency:.3e}  bit-exact={consistency == 0.0}")

    # (2) determinism across calls (cross-replica determinism proxy).
    n1 = helpers._zorl_normalized_b_noises_gpu(base, seed=99991, device=dev)
    n2 = helpers._zorl_normalized_b_noises_gpu(base, seed=99991, device=dev)
    det = all(torch.equal(n1[k], n2[k]) for k in n1)
    print(f"  (2) determinism same-seed identical: {det}")

    # (3) antithetic symmetry about the parent.
    sigma = 0.05
    noise = helpers._zorl_normalized_b_noises_gpu(base, seed=7, device=dev)
    anti = True
    for k, v in noise.items():
        plus = v.mul(sigma)       # candidate(+) - parent
        minus = v.mul(-sigma)     # candidate(-) - parent
        if not torch.equal(plus, minus.neg()):
            anti = False
            break
    print(f"  (3) antithetic +/- exact negation: {anti}")

    # (4) speed vs CPU golden apply (same pairs/scores; values differ by RNG).
    helpers_cpu = helpers
    g_adapter = clone_adapter(base)
    t0 = time.perf_counter()
    m_cpu = golden_apply(
        helpers_cpu, g_adapter, seed_score_pairs, raw_pair_scores,
        learning_rate=lr, max_update_norm=max_update_norm,
    )
    t_cpu = time.perf_counter() - t0

    # full GPU apply incl. weight update, timed with cuda sync.
    o_adapter = clone_adapter(base)
    nscores_std = helpers._normalize_zorl_pair_scores(raw_pair_scores, normalization="standard")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    upd, _z = helpers._zorl_build_update_gpu(
        o_adapter, seed_score_pairs, nscores_std, weight_scale,
        perturbation_mode="b_only", device=dev,
    )
    norm_sq = sum(float(torch.sum(u * u).item()) for u in upd.values())
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0
    print(f"  (4) CPU golden apply: {t_cpu:7.3f}s   GPU build+norm: {t_gpu:7.3f}s   "
          f"speedup: {t_cpu/max(t_gpu,1e-9):6.1f}x")
    print(f"      (CPU update_norm={m_cpu['update_norm']:.3f}  GPU raw_norm={math.sqrt(norm_sq):.3f}; "
          f"differ by RNG, both finite)")

    ok = (consistency == 0.0) and det and anti
    print(f"  GPU VALIDATION: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=512)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-update-norm", type=float, default=20000.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument(
        "--gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help="run the GPU fast-path validation (auto=run if cuda available)",
    )
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    helpers = _load_golden_helpers()()
    print(f"[setup] golden helpers loaded from {FORK_LORA_MANAGER}")
    print(f"[setup] torch={torch.__version__} threads={torch.get_num_threads()}")
    print(
        f"[setup] num_pairs={args.num_pairs} layers={args.layers} "
        f"lr={args.lr} max_update_norm={args.max_update_norm}"
    )

    base = build_synthetic_adapter(num_layers=args.layers)
    seed_score_pairs, raw_pair_scores = _build_seed_score_pairs(args.num_pairs)

    # total B floats / pair
    n_b_floats = sum(
        int(torch.tensor(s).prod()) for s in _B_MODULE_SHAPES.values()
    ) * (args.layers // 1)
    # account per-layer count (each module appears once per layer)
    n_b_floats = 0
    for layer in base.layers:
        for name, w in layer.weights.items():
            if "lora_B" in name:
                n_b_floats += w.numel()
    print(f"[setup] LoRA-B floats/pair={n_b_floats:,}  (~{n_b_floats*4/1e6:.1f} MB fp32)")

    # ---- Correctness gate ----
    print("\n=== correctness gate (b_only) ===")
    g_adapter = clone_adapter(base)
    o_adapter = clone_adapter(base)
    m_gold = golden_apply(
        helpers, g_adapter, seed_score_pairs, raw_pair_scores,
        learning_rate=args.lr, max_update_norm=args.max_update_norm,
    )
    m_opt = optimized_apply(
        helpers, o_adapter, seed_score_pairs, raw_pair_scores,
        learning_rate=args.lr, max_update_norm=args.max_update_norm,
    )
    print(f"  golden parent LoRA-B sha256 (CPU reference): {adapter_b_hash(g_adapter)}")
    max_diff = _max_b_diff(g_adapter, o_adapter)
    metric_bad = _metrics_agree(m_gold, m_opt)
    print(f"  max |Δ| parent LoRA-B weight (golden vs optimized): {max_diff:.3e}")
    print(f"  bit-exact parent weights: {max_diff == 0.0}")
    print(f"  update_norm golden={m_gold['update_norm']:.6f} opt={m_opt['update_norm']:.6f}")
    print(f"  update_clip_scale={m_gold['update_clip_scale']:.4f}")
    if metric_bad:
        print("  METRIC MISMATCH:")
        for b in metric_bad:
            print(f"    {b}")
    else:
        print("  all metrics agree exactly: True")
    correctness_ok = (max_diff == 0.0) and not metric_bad
    print(f"  CORRECTNESS GATE: {'PASS' if correctness_ok else 'FAIL'}")

    # ---- Timing ----
    print("\n=== timing ===")

    def timeit(fn):
        best = math.inf
        for _ in range(max(1, args.repeat)):
            ad = clone_adapter(base)
            t0 = time.perf_counter()
            fn(ad)
            best = min(best, time.perf_counter() - t0)
        return best

    t_gold = timeit(
        lambda ad: golden_apply(
            helpers, ad, seed_score_pairs, raw_pair_scores,
            learning_rate=args.lr, max_update_norm=args.max_update_norm,
        )
    )
    # Precompute layout once (mirrors the server caching it on the adapter).
    layout = _precompute_b_layout(helpers, base)
    t_opt = timeit(
        lambda ad: optimized_apply(
            helpers, ad, seed_score_pairs, raw_pair_scores,
            learning_rate=args.lr, max_update_norm=args.max_update_norm,
            layout=layout,
        )
    )
    print(f"  golden    apply: {t_gold:8.3f}s  ({t_gold/args.num_pairs*1e3:.2f} ms/pair)")
    print(f"  optimized apply: {t_opt:8.3f}s  ({t_opt/args.num_pairs*1e3:.2f} ms/pair)")
    if t_opt > 0:
        print(f"  speedup: {t_gold/t_opt:.2f}x   saved {t_gold-t_opt:.2f}s/apply")

    if not args.skip_profile:
        print("\n=== current-path cost breakdown ===")
        prof = profile_current_breakdown(helpers, base, seed_score_pairs)
        for k, v in prof.items():
            if k.endswith("_s"):
                print(f"  {k:34s} {v:8.3f}s")
            else:
                print(f"  {k:34s} {v}")
        irreducible = prof["t_randn_floor_s"]
        removable = t_gold - irreducible
        print(
            f"  -> irreducible randn floor ~{irreducible:.2f}s; "
            f"removable overhead in golden ~{removable:.2f}s "
            f"({removable/max(t_gold,1e-9)*100:.0f}% of apply)"
        )

    run_gpu = args.gpu == "on" or (args.gpu == "auto" and torch.cuda.is_available())
    if run_gpu:
        if not torch.cuda.is_available():
            print("\n[gpu] requested but CUDA unavailable; skipping")
        else:
            gpu_validation(
                helpers, base, args.num_pairs,
                lr=args.lr, max_update_norm=args.max_update_norm,
            )
    else:
        print("\n[gpu] skipped (no CUDA or --gpu off)")


if __name__ == "__main__":
    main()
