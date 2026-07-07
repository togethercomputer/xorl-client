#!/usr/bin/env python3
"""D3 SPIKE (2026-07-07): split-backward STE gradient-equality gate.

The two-pass quantizer OOMs the 35B under FSDP1 because the JOINT graph uses
every layer's params twice (writer pass A + reader pass B), so reduce-scatter
fires only after the pass-A backward and ~n_layers unsharded bf16 layer-grads
pile up (ATTEMPT1_PREREG_20260706.md:206-223). The candidate fix cuts the
graph at the snapped embeddings and runs TWO backwards (pass B, then pass A
seeded with the leaf grad) — mechanism_forward.py `split_backward=True` +
mechanism_sandbox_train.py `--split-backward`.

This gate verifies gradient equality of split vs joint BEFORE any 35B use
(D3_TOKEN_WRITEBACK.md §1 rung 2: "verify gradient equality vs the joint
graph on marin first, max|Δ| gate").

Modes:
  --which cpu    tiny-dense (marin arch) + tiny-hybrid (q35 arch), fp32 CPU,
                 no FSDP. Expect EXACT equality (same op order, same dtype);
                 gate max|Δ| <= 1e-9.
  --which tinyfsdp
                 the same tiny models under REAL FSDP FULL_SHARD (torchrun,
                 world >= 2), fp32 / no MixedPrecision. Exercises exactly the
                 code paths the 35B run needs — one root forward, TWO
                 backwards, per-backward reduce-scatter + sharded-grad
                 accumulation — with fp32 math. GATED: max|Δ| <= 1e-6.
  --which marin  real marin 9.7B RL ckpt under FSDP FULL_SHARD (launch under
                 torchrun, world >= 2 to exercise reduce-scatter), toy batch
                 from the canonical comp-bench band pool, k = 4*n_ops slots.
                 --mp fp32 : no MixedPrecision -> mechanics-exactness gate,
                             max|Δ| <= 1e-6 (only reduce-scatter association
                             differs: RS(gA)+RS(gB) vs RS(gA+gB), fp32).
                             !! needs world >= 8: the JOINT baseline holds
                             ~37 x 1.05 GiB unsharded fp32 grads (the very
                             pathology under test) and OOMs at world 2
                             (measured 2026-07-07 gate pod, 74.6G alloc at
                             backward, tried 902M).
                 --mp bf16 : the production trainer MixedPrecision (param
                             bf16 / reduce fp32). REPORT-ONLY: the joint
                             graph accumulates gA+gB in bf16 (autograd
                             accumulation on the unsharded bf16 grad) while
                             split accumulates the two fp32 reduced shards —
                             split is the MORE precise order; delta is the
                             bf16 accumulation rounding, recorded not gated.
                 A joint-vs-joint rerun measures the run-to-run noise floor
                 (SDPA pinned to MATH for determinism); the gate compares
                 delta against max(threshold, 10x floor).

Everything lands in one JSON (--out) — FILES-ONLY discipline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from mechanism_forward import FeatureConfig, MarinMechModel, Q35BMech

RL_CKPT = "/shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B"
CB_POOL = "/shared/apanda/filler_grpo/comp_bench/data/nops05-07_capped_train64k.jsonl"

FEATS = FeatureConfig(quantize_slots="hard", quantize_metric="logits")


def grads_of(model):
    # snapshots go to CPU: keeping 3 full fp32 grad clones on-GPU OOMed the
    # marin world-2 gates (76.9G alloc at the split pass, 2026-07-07 ~14:35Z)
    out = {}
    for n, p in model.named_parameters():
        if p.grad is not None and p.grad.numel() > 0:
            out[n] = p.grad.detach().float().cpu().clone()
    return out


def zero_grads(model):
    for p in model.parameters():
        p.grad = None


def compare(ga: dict, gb: dict):
    only_a = sorted(set(ga) - set(gb))
    only_b = sorted(set(gb) - set(ga))
    per = {}
    for k in ga:
        if k not in gb:
            continue
        d = (ga[k] - gb[k]).abs()
        scale = ga[k].abs().max().clamp_min(1e-12)
        per[k] = (float(d.max()), float(d.max() / scale))
    max_abs = max(v[0] for v in per.values()) if per else 0.0
    max_rel = max(v[1] for v in per.values()) if per else 0.0
    if only_a or only_b:
        max_abs = float("inf")   # a missing gradient is a hard mismatch
    top = sorted(per.items(), key=lambda kv: -kv[1][0])[:5]
    return {"max_abs": max_abs, "max_rel": max_rel, "n_tensors": len(per),
            "missing_in_b": only_a[:8], "missing_in_a": only_b[:8],
            "n_missing": len(only_a) + len(only_b),
            "top_offenders": [{"name": k, "max_abs": v[0], "rel": v[1]}
                              for k, v in top]}


def backward_schedule(out, split: bool, model=None):
    """ce-only losses (the D3 spike shape). Returns float(ce)."""
    from mechanism_forward import (fsdp_joint_backward_arm_hooks,
                                   fsdp_split_backward_hook_graph,
                                   fsdp_split_backward_rearm,
                                   fsdp_split_backward_save,
                                   fsdp_split_install_jit_hooks,
                                   fsdp_split_jit_arm,
                                   fsdp_split_jit_disarm)
    ce = out["ce"]
    if split:
        if model is not None:
            fsdp_split_install_jit_hooks(model)
        saved = fsdp_split_backward_save(model) if model is not None else []
        if model is not None:
            fsdp_split_jit_arm()
        hooks_b = (fsdp_split_backward_hook_graph(model, saved, [ce])
                   if model is not None else [])
        ce.backward()
        for hd in hooks_b:
            hd.remove()
        if model is not None:
            fsdp_split_jit_disarm()
        fsdp_split_backward_rearm(saved)
        g = out["h1_leaf"].grad
        assert g is not None, "no grad reached the snapped-embedding leaf"
        if model is not None:
            fsdp_split_jit_arm()
        hooks_a = (fsdp_split_backward_hook_graph(model, saved,
                                                  [out["h1_graph"]])
                   if model is not None else [])
        torch.autograd.backward(out["h1_graph"], g)
        for hd in hooks_a:
            hd.remove()
        if model is not None:
            fsdp_split_jit_disarm()
    else:
        # JOINT path: re-arm post-backward hooks for multi-forward units
        # under MixedPrecision (lm_head silent-grad-drop fix, 2026-07-07;
        # no-op for model=None / non-FSDP / fp32)
        hooks_j, refs_j = (fsdp_joint_backward_arm_hooks(model, [ce])
                           if model is not None else ([], []))
        ce.backward()
        for hd in hooks_j:
            hd.remove()
        del refs_j
    return float(ce)


# ---------------------------------------------------------------- CPU tiny --

def run_cpu():
    from mechanism_sandbox_gates import synth_batch, tiny_dense, tiny_hybrid
    report = {}
    for kind in ("dense", "hybrid"):
        torch.manual_seed(99)
        mech = (MarinMechModel(tiny_dense()) if kind == "dense"
                else Q35BMech(tiny_hybrid()))
        batch = synth_batch(B=2, L=24)
        ces, grads = [], []
        for tag, split in (("joint", False), ("joint2", False), ("split", True)):
            zero_grads(mech)
            out = mech(batch, FEATS, {}, split_backward=split)
            ces.append(backward_schedule(out, split))
            grads.append(grads_of(mech))
        floor = compare(grads[0], grads[1])
        delta = compare(grads[0], grads[2])
        report[kind] = {
            "ce_joint": ces[0], "ce_split": ces[2],
            "ce_bitequal": ces[0] == ces[2],
            "joint_vs_joint_floor": floor,
            "joint_vs_split": delta,
            "pass": (ces[0] == ces[2] and delta["max_abs"] <= 1e-9
                     and floor["max_abs"] == 0.0),
        }
        print(f"[split-gate cpu/{kind}] ce_bitequal={report[kind]['ce_bitequal']} "
              f"floor={floor['max_abs']:.3e} delta={delta['max_abs']:.3e} "
              f"pass={report[kind]['pass']}", flush=True)
    report["pass"] = all(report[k]["pass"] for k in ("dense", "hybrid"))
    return report


# ---------------------------------------------------------------- tinyfsdp --

def run_tinyfsdp():
    """Tiny dense + hybrid under real FSDP FULL_SHARD, fp32, no MP —
    the FSDP-mechanics exactness gate (torchrun, world >= 2)."""
    import functools
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        ShardingStrategy)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from mechanism_sandbox_gates import synth_batch, tiny_dense, tiny_hybrid
    from recur_loop import RecurLayer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    def policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return (isinstance(module, (RecurLayer, Qwen3_5MoeDecoderLayer))
                or isinstance(module, nn.Embedding)
                or (isinstance(module, nn.Linear) and module.out_features == 512))

    report = {"world": dist.get_world_size()}
    for kind in ("dense", "hybrid"):
        torch.manual_seed(99)   # same on every rank -> identical tiny weights
        mech = (MarinMechModel(tiny_dense()) if kind == "dense"
                else Q35BMech(tiny_hybrid())).float()
        model = FSDP(mech, auto_wrap_policy=policy,
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=None, device_id=device,
                     sync_module_states=True, use_orig_params=True)
        if kind == "hybrid":
            # mirror the 35B D3 shape: every decoder layer activation-
            # checkpointed (NO_REENTRANT) — covers AC x split-backward
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda m: isinstance(m, Qwen3_5MoeDecoderLayer))
        batch = synth_batch(B=2, L=24, device=device)
        ces, grads = [], []
        for tag, split in (("joint", False), ("joint2", False), ("split", True)):
            zero_grads(model)
            with sdpa_kernel([SDPBackend.MATH]):
                out = model(batch, FEATS, {}, split_backward=split)
                ces.append(backward_schedule(out, split, model=model))
            grads.append(grads_of(model))
        floor = compare(grads[0], grads[1])
        delta = compare(grads[0], grads[2])
        stats = torch.tensor([floor["max_abs"], delta["max_abs"],
                              delta["max_rel"]], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        g_floor, g_abs, g_rel = [float(x) for x in stats]
        ok = ces[0] == ces[2] and g_abs <= max(1e-6, 10 * g_floor)
        report[kind] = {"ce_joint": ces[0], "ce_split": ces[2],
                        "ce_bitequal": ces[0] == ces[2],
                        "joint_vs_joint_floor_max_abs": g_floor,
                        "joint_vs_split": {"max_abs": g_abs, "max_rel": g_rel},
                        "pass": ok}
        if rank == 0:
            print(f"[split-gate tinyfsdp/{kind}] world={report['world']} "
                  f"ce_bitequal={report[kind]['ce_bitequal']} floor={g_floor:.3e} "
                  f"delta={g_abs:.3e} pass={ok}", flush=True)
        del model, mech
        torch.cuda.empty_cache()
    report["pass"] = all(report[k]["pass"] for k in ("dense", "hybrid"))
    dist.barrier()
    dist.destroy_process_group()
    return report if rank == 0 else None


# ---------------------------------------------------------------- marin -----

def adapt(r):
    return {"prompt": r["problem"], "gold": str(r["answer"]), "band": r["n_ops"],
            "k": 4 * r["n_ops"], "n_ops": r["n_ops"], "trace": r.get("trace"),
            "sig": f"cb{r['idx']}", "idx": r["idx"]}


def build_marin_fsdp(rank, device, mp_mode: str):
    import functools
    import torch.nn as nn
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy)
    from transformers import AutoConfig, AutoModelForCausalLM
    from mechanism_sandbox_train import marin_wrap_policy
    if rank == 0:
        hf = AutoModelForCausalLM.from_pretrained(
            RL_CKPT, dtype=torch.float32, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
        model = MarinMechModel(hf)
    else:
        config = AutoConfig.from_pretrained(RL_CKPT)
        with torch.device("meta"):
            hf = AutoModelForCausalLM.from_config(config)
            model = MarinMechModel(hf)
        model = model.to(torch.float32)
    mp = None
    if mp_mode == "bf16":
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                            buffer_dtype=torch.float32)
    elif mp_mode == "bf16kl":
        # the EXACT 35B bf16-reduce rung MP (build_q35b --grad-reduce bf16):
        # reduce in bf16 + keep_low_precision_grads — covers the sharded-grad
        # accumulation path across the two backwards in the rung-2 dtype
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                            buffer_dtype=torch.float32,
                            keep_low_precision_grads=True)
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(marin_wrap_policy,
                                           vocab=model.config.vocab_size),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        param_init_fn=(None if rank == 0 else
                       lambda m: m.to_empty(device=device, recurse=False)),
    )


def run_marin(mp_mode: str, n_items: int):
    import json as _json
    import torch.distributed as dist
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from recur_loop import build_batch, load_tokenizer
    from mechanism_forward import marin_annotate_spans
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)

    tok = load_tokenizer(RL_CKPT)
    rows = []
    with open(CB_POOL) as f:
        for line in f:
            r = _json.loads(line)
            if r["n_ops"] == 5:          # shortest in-band items -> toy batch
                rows.append(adapt(r))
            if len(rows) >= n_items:
                break
    batch = build_batch(tok, rows, "r1", device, with_answer=True)
    marin_annotate_spans(tok, batch, rows, "r1")
    batch["slot_labels"] = torch.full_like(batch["labels"], -100)

    model = build_marin_fsdp(rank, device, mp_mode)
    model.train()
    if rank == 0:
        print(f"[split-gate marin/{mp_mode}] FSDP up, world={dist.get_world_size()} "
              f"L={batch['input_ids'].shape} ks={batch.get('ks')}", flush=True)

    ces, grads = [], []
    for tag, split in (("joint", False), ("joint2", False), ("split", True)):
        zero_grads(model)
        with sdpa_kernel([SDPBackend.MATH]):
            out = model(batch, FEATS, {}, split_backward=split)
            ce = backward_schedule(out, split, model=model)
        ces.append(ce)
        grads.append(grads_of(model))
        if rank == 0:
            print(f"[split-gate marin/{mp_mode}] pass {tag}: ce={ce:.6f} "
                  f"n_grad_tensors={len(grads[-1])}", flush=True)

    floor = compare(grads[0], grads[1])
    delta = compare(grads[0], grads[2])
    # global max across ranks (each rank holds its own shard)
    stats = torch.tensor([floor["max_abs"], floor["max_rel"],
                          delta["max_abs"], delta["max_rel"]], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    g_floor_abs, g_floor_rel, g_delta_abs, g_delta_rel = [float(x) for x in stats]
    thresh = 1e-6
    gated = mp_mode == "fp32"
    finite_ok = all(torch.isfinite(g).all() for g in grads[2].values())
    ok = ((not gated) or g_delta_abs <= max(thresh, 10 * g_floor_abs)) and finite_ok
    report = {
        "mode": mp_mode, "world": dist.get_world_size(),
        "n_items": len(rows), "seq": list(batch["input_ids"].shape),
        "ks": batch.get("ks"),
        "ce_joint": ces[0], "ce_joint2": ces[1], "ce_split": ces[2],
        "ce_bitequal_joint_split": ces[0] == ces[2],
        "joint_vs_joint_floor": {"max_abs": g_floor_abs, "max_rel": g_floor_rel},
        "joint_vs_split": {"max_abs": g_delta_abs, "max_rel": g_delta_rel},
        "rank0_top_offenders": delta["top_offenders"],
        "gated": gated, "threshold": thresh, "pass": ok,
        "split_grads_all_finite": finite_ok,
        "note": ("fp32/no-MP: mechanics-exactness gate" if gated else
                 "bf16 MP report-only: joint accumulates gA+gB in bf16, split "
                 "in fp32 post-reduce — delta records the bf16 rounding term"),
    }
    if rank == 0:
        print(f"[split-gate marin/{mp_mode}] floor={g_floor_abs:.3e} "
              f"delta={g_delta_abs:.3e} (rel {g_delta_rel:.3e}) pass={ok}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return report if rank == 0 else None


# ---------------------------------------------------------------- tinyadj ---

def run_tinyadj():
    """ADJUDICATOR (2026-07-07): the bf16/bf16kl marin gates show a large
    DETERMINISTIC joint-vs-split delta (max_abs 7.125, rel ~8, q/k_proj-
    concentrated). Oracle = an UNWRAPPED bf16 twin (identical seed-99
    weights, same batch, MATH SDPA) — the CPU gate proved joint==split
    exactly without FSDP, so its grads are the truth up to bf16 kernel
    noise. Compare BOTH FSDP schedules against it on the tiny hybrid with
    the 35B config (NO_REENTRANT AC + bf16 MP), FSDP world 1 so orig-param
    grads are full-shaped (no summon needed; summon with_grads +
    use_orig_params + offload_to_cpu is NotImplemented in torch 2.10).
    Single process — launch WITHOUT torchrun or with nproc 1."""
    import functools
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from mechanism_sandbox_gates import synth_batch, tiny_hybrid
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    def policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return (isinstance(module, Qwen3_5MoeDecoderLayer)
                or isinstance(module, nn.Embedding)
                or (isinstance(module, nn.Linear) and module.out_features == 512))

    def strip(n):
        return n.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", "")

    batch = synth_batch(B=2, L=24, device=device)

    # oracle: unwrapped bf16 twin
    torch.manual_seed(99)
    oracle = Q35BMech(tiny_hybrid()).to(device=device, dtype=torch.bfloat16)
    with sdpa_kernel([SDPBackend.MATH]):
        o = oracle(batch, FEATS, {})
        ce_oracle = backward_schedule(o, False)
    ref_g = {n: p.grad.detach().float().cpu().clone()
             for n, p in oracle.named_parameters() if p.grad is not None}
    del oracle
    torch.cuda.empty_cache()

    # FSDP world-1 twin, 35B shape: bf16 MP + NO_REENTRANT AC everywhere
    torch.manual_seed(99)
    mech = Q35BMech(tiny_hybrid()).float()
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32)
    model = FSDP(mech, auto_wrap_policy=policy,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=mp, device_id=device,
                 sync_module_states=True, use_orig_params=True)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, Qwen3_5MoeDecoderLayer))

    def vs_ref(g):
        per = {}
        for k, rg in ref_g.items():
            if k not in g:
                continue
            d = (g[k] - rg).abs()
            scale = rg.abs().max().clamp_min(1e-12)
            per[k] = (float(d.max()), float(d.max() / scale))
        return {"max_abs": max(v[0] for v in per.values()),
                "max_rel": max(v[1] for v in per.values()),
                "n": len(per),
                "missing_vs_oracle": sorted(set(ref_g) - set(g))[:6],
                "extra_vs_oracle": sorted(set(g) - set(ref_g))[:6],
                "worst": sorted([{"name": k, "max_abs": v[0], "rel": v[1]}
                                 for k, v in per.items()],
                                key=lambda x: -x["max_abs"])[:5]}

    res = {}
    ces = {}
    for tag, split in (("joint", False), ("split", True)):
        zero_grads(model)
        with sdpa_kernel([SDPBackend.MATH]):
            out = model(batch, FEATS, {}, split_backward=split)
            ces[tag] = backward_schedule(out, split, model=model)
        g = {strip(n): p.grad.detach().float().cpu().clone()
             for n, p in model.named_parameters() if p.grad is not None}
        res[tag] = vs_ref(g)
        print(f"[split-adj tiny] fsdp-{tag}: ce={ces[tag]:.6f} "
              f"vs_oracle max_abs={res[tag]['max_abs']:.4g} "
              f"max_rel={res[tag]['max_rel']:.4g} (n={res[tag]['n']})", flush=True)

    # verdict v2 (2026-07-07, joint lm_head fix): COMPLETENESS-aware — a
    # tensor missing vs the oracle is a hard mismatch (mirrors compare()'s
    # inf rule; the pre-fix 10x-magnitude rule labeled the 140-vs-141 case
    # "AMBIGUOUS" because both sides sat at the same bf16-noise max_abs).
    # Magnitude bar = the banked bf16-noise level for this tiny recipe
    # (0.0469 measured on both schedules; bar 0.1).
    NOISE_BAR = 0.1
    j, sp = res["joint"]["max_abs"], res["split"]["max_abs"]
    j_ok = not res["joint"]["missing_vs_oracle"] and j <= NOISE_BAR
    s_ok = not res["split"]["missing_vs_oracle"] and sp <= NOISE_BAR
    verdict = ("BOTH-MATCH" if j_ok and s_ok else
               "SPLIT-MATCHES-ORACLE" if s_ok else
               "JOINT-MATCHES-ORACLE" if j_ok else "AMBIGUOUS")
    report = {"config": "tiny-hybrid AC(NO_REENTRANT) + bf16 MP, FSDP world 1",
              "ce": ces, "ce_oracle": ce_oracle,
              "fsdp_joint_vs_oracle": res["joint"],
              "fsdp_split_vs_oracle": res["split"],
              "noise_bar": NOISE_BAR,
              "verdict": verdict,
              "pass": verdict in ("SPLIT-MATCHES-ORACLE", "BOTH-MATCH")}
    print(f"[split-adj tiny] => {verdict}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return report


# ---------------------------------------------------------------- jointmp ---

class _DblForwardToy(torch.nn.Module):
    """Minimal double-forward module for the joint+MP regression gate: the
    head forwards 3x per step — one snap-style call whose only consumer is
    argmax (grad-severed, exactly the quantizer's snap-logits shape) and TWO
    live calls whose losses are summed; the embedding forwards 2x. Under
    FSDP1 + MixedPrecision, torch's first-forward post-backward hook lands on
    the severed call's AccumulateGrad -> pre-fix the head grad is silently
    dropped."""

    def __init__(self, vocab=64, hidden=32):
        super().__init__()
        torch.manual_seed(3)
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids, labels):
        import torch.nn.functional as F
        h = self.emb(ids)
        sim = self.head(h)                    # fwd 1 — argmax-severed
        idx = sim.argmax(-1)
        e = self.emb(idx)                     # emb fwd 2 (STE rows)
        h2 = e + (h - h.detach())
        la = F.cross_entropy(self.head(h2).flatten(0, 1), labels.flatten())      # fwd 2, live
        lb = F.cross_entropy(self.head(h * 0.5).flatten(0, 1), labels.flatten())  # fwd 3, live
        return la + lb


def run_jointmp():
    """REGRESSION GATE (2026-07-07, joint+MP silent-gradient-loss fix): any
    FSDP unit that forwards >= 2x per step must still receive its gradient on
    the JOINT (single-backward) path under MixedPrecision (the D3 oracle
    finding: joint baseline 140/141, `lm_head.weight` missing —
    `d3_spike/gates/split_adj_tiny.json`; mechanism + fix in
    mechanism_forward.fsdp_joint_backward_arm_hooks).

      toy : _DblForwardToy (head 3x = severed + two live summed losses; emb
            2x), FSDP world-1 bf16-MP vs unwrapped bf16 twin. Same op order /
            dtype at world 1 => grads must be PRESENT and EXACT (<= 1e-6;
            measured 0.0).
      q35 : the production quantizer shape — tiny hybrid + quantizer FEATS +
            NO_REENTRANT AC + bf16 MP, FSDP world-1 JOINT backward vs
            unwrapped bf16 twin (the split_adj_tiny recipe's joint side).
            ALL tensors present (141/141) within the banked bf16-noise bar
            (max_abs <= 0.1; banked level 0.0469).

    Joint backwards run through `backward_schedule` (the gates' production
    joint path), so this gate FAILS on pre-fix code and must PASS post-fix.
    Single process — launch WITHOUT torchrun or with nproc 1."""
    import functools
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from mechanism_sandbox_gates import synth_batch, tiny_hybrid
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32)
    report = {}

    # ------------------------------ toy ------------------------------
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, 64, (2, 6), generator=g).to(device)
    labels = torch.randint(0, 64, (2, 6), generator=g).to(device)
    oracle = _DblForwardToy().to(device=device, dtype=torch.bfloat16)
    oracle(ids, labels).backward()
    ref = {n: p.grad.detach().float().cpu().clone()
           for n, p in oracle.named_parameters()}
    del oracle

    def toy_policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return isinstance(module, (nn.Embedding, nn.Linear))

    model = FSDP(_DblForwardToy().float(), auto_wrap_policy=toy_policy,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=mp, device_id=device, use_orig_params=True)
    steps = []
    for step in range(2):        # 2 steps: the re-arm must survive step 1
        zero_grads(model)
        loss = model(ids, labels)
        backward_schedule({"ce": loss}, False, model=model)
        got = {n.replace("_fsdp_wrapped_module.", ""):
               p.grad.detach().float().cpu().clone()
               for n, p in model.named_parameters() if p.grad is not None}
        per = {n: (float((got[n] - ref[n]).abs().max()) if n in got else None)
               for n in ref}
        steps.append({"missing": sorted(n for n, v in per.items() if v is None),
                      "max_abs": max((v for v in per.values() if v is not None),
                                     default=float("inf")),
                      "per_tensor": per})
    toy_ok = all(not s["missing"] and s["max_abs"] <= 1e-6 for s in steps)
    report["toy"] = {"steps": steps, "pass": toy_ok}
    print(f"[jointmp-gate toy] steps={[(s['missing'], s['max_abs']) for s in steps]} "
          f"pass={toy_ok}", flush=True)
    del model
    torch.cuda.empty_cache()

    # ------------------------------ q35 ------------------------------
    def q35_policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return (isinstance(module, Qwen3_5MoeDecoderLayer)
                or isinstance(module, nn.Embedding)
                or (isinstance(module, nn.Linear) and module.out_features == 512))

    def strip(n):
        return n.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", "")

    batch = synth_batch(B=2, L=24, device=device)
    torch.manual_seed(99)
    oracle = Q35BMech(tiny_hybrid()).to(device=device, dtype=torch.bfloat16)
    with sdpa_kernel([SDPBackend.MATH]):
        o = oracle(batch, FEATS, {})
        ce_oracle = backward_schedule(o, False)
    ref_g = {n: p.grad.detach().float().cpu().clone()
             for n, p in oracle.named_parameters() if p.grad is not None}
    del oracle
    torch.cuda.empty_cache()

    torch.manual_seed(99)
    model = FSDP(Q35BMech(tiny_hybrid()).float(), auto_wrap_policy=q35_policy,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=mp, device_id=device,
                 sync_module_states=True, use_orig_params=True)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, Qwen3_5MoeDecoderLayer))
    zero_grads(model)
    with sdpa_kernel([SDPBackend.MATH]):
        out = model(batch, FEATS, {}, split_backward=False)
        ce_joint = backward_schedule(out, False, model=model)
    got = {strip(n): p.grad.detach().float().cpu().clone()
           for n, p in model.named_parameters() if p.grad is not None}
    per = {}
    for k, rg in ref_g.items():
        if k in got:
            per[k] = float((got[k] - rg).abs().max())
    missing = sorted(set(ref_g) - set(got))
    max_abs = max(per.values()) if per else float("inf")
    NOISE_BAR = 0.1   # banked tinyadj bf16-noise level: 0.0469 (both schedules)
    q35_ok = (not missing and len(got) == len(ref_g) and max_abs <= NOISE_BAR
              and "lm.lm_head.weight" in got)
    report["q35"] = {
        "ce_joint": ce_joint, "ce_oracle": ce_oracle,
        "n_oracle": len(ref_g), "n_joint": len(got),
        "missing_vs_oracle": missing[:8],
        "max_abs_vs_oracle": max_abs, "noise_bar": NOISE_BAR,
        "lm_head_present": "lm.lm_head.weight" in got,
        "lm_head_max_abs": per.get("lm.lm_head.weight"),
        "worst": sorted(({"name": k, "max_abs": v} for k, v in per.items()),
                        key=lambda x: -x["max_abs"])[:5],
        "pass": q35_ok}
    print(f"[jointmp-gate q35] n={len(got)}/{len(ref_g)} missing={missing[:4]} "
          f"max_abs={max_abs:.4g} lm_head_present={report['q35']['lm_head_present']} "
          f"pass={q35_ok}", flush=True)
    report["pass"] = toy_ok and q35_ok
    dist.barrier()
    dist.destroy_process_group()
    return report


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cpu", "tinyfsdp", "marin", "tinyadj",
                                        "jointmp"], required=True)
    ap.add_argument("--mp", choices=["fp32", "bf16", "bf16kl"], default="fp32")
    ap.add_argument("--items", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.which == "cpu":
        report = run_cpu()
    elif args.which == "tinyfsdp":
        report = run_tinyfsdp()
    elif args.which == "tinyadj":
        report = run_tinyadj()
    elif args.which == "jointmp":
        report = run_jointmp()
    else:
        report = run_marin(args.mp, args.items)
    if report is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[split-gate] {'PASS' if report.get('pass') else 'REPORT/FAIL'} "
              f"-> {args.out}", flush=True)
        if not report.get("pass"):
            sys.exit(1)


if __name__ == "__main__":
    main()
