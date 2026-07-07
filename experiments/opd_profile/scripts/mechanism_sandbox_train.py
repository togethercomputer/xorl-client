#!/usr/bin/env python3
"""MECHANISM-SANDBOX trainer — standalone HF-side FSDP trainer with full
forward-graph control (2026-07-06). NO xorl engine; single node; torchrun.

Models:
  --model marin : marin 9.7B dense (RL ckpt), MarinMechModel over
                  recur_loop.RecurLayer. Recipe/data/rng for `--arm c0
                  --task sub` are BIT-COMPATIBLE with recur_stage0_train.py
                  (same rng_rank stream, same build_batch, same lr schedule)
                  => gate (i): features-off run overlays the banked
                  recur_stage0 C0 loss curve (step-0 CE must match to <=5e-3;
                  trajectory within the same-seed kernel-noise band).
  --model q35b  : Qwen3.6-35B-A3B hybrid via HF (Qwen3_5MoeForCausalLM,
                  STRICT-audited text-only load), FSDP bf16 compute /
                  fp32 master, activation checkpointing (capture layers
                  excluded), per-step tokens/s + peak-memory logging
                  => gate (ii): 50-step smoke at band 5-7.

Features: FeatureConfig (mechanism_forward.py), all default-off. --features
'{"slot_value_coef":1.0,...}' JSON blob or --features-file.

Eval path is PRODUCTION-ONLY: checkpoints saved here are exported by
mechanism_export_hf.py to sglang-servable HF dirs (incl.
preprocessor_config.json for q35b — the patched-converter lesson); evals run
via comp_bench_eval.py [--pause-slots] / eval_math_suite.py. No eval-time
dependence on sandbox code.

Launch:
  torchrun --standalone --nproc-per-node 8 mechanism_sandbox_train.py \
      --model marin --arm c0 --task sub --steps 500 --out-dir .../parity
  torchrun --standalone --nproc-per-node 8 mechanism_sandbox_train.py \
      --model q35b --band nops05-07 --steps 50 --out-dir .../q35_smoke
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, "/home/apanda/xorl-client")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)

from mechanism_forward import (
    FeatureConfig,
    MarinMechModel,
    Q35BMech,
    Q35_BASE_SNAPSHOT,
    assemble_losses,
    build_batch_q35,
    digit_slot_labels_from_trace,
    load_q35b_text,
    marin_annotate_spans,
    need_from_feats,
    sample_k,
)
from recur_loop import RecurLayer, build_batch, load_tokenizer

RL_CKPT = "/shared/xorl-marin-rl-6279/checkpoints/delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B"
CB_DATA = "/shared/apanda/filler_grpo/comp_bench/data"

ARGS = None


def log(msg):
    if dist.get_rank() == 0:
        print(f"[mech-train] {msg}", flush=True)


# ---------------------------------------------------------------- marin ----

def marin_wrap_policy(module, recurse, nonwrapped_numel, vocab=128256):
    if recurse:
        return True
    return (isinstance(module, RecurLayer)
            or isinstance(module, nn.Embedding)
            or (isinstance(module, nn.Linear) and module.out_features == vocab))


def build_marin(rank, device, need, digit_token_ids=None):
    from transformers import AutoConfig, AutoModelForCausalLM
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
    assert model.n_layers == 37, model.n_layers
    # digit-embedding table for trace-digit-embed hidden targets (arm A):
    # extract on rank0 BEFORE FSDP shards the embedding; broadcast after wrap
    digit_vecs = None
    if digit_token_ids is not None:
        H = model.config.hidden_size
        if rank == 0:
            # DETACH is load-bearing: without it the teacher table carries a
            # grad_fn into the pre-FSDP embedding weight and backward crashes
            # against the flattened shard ("size of tensor a (61562880) must
            # match ... (3840)" — armA P3 crash, 2026-07-06 23:26Z)
            digit_vecs = model.embed_tokens.weight[
                torch.tensor(digit_token_ids)].detach().clone().float().to(device)
        else:
            digit_vecs = torch.zeros(len(digit_token_ids), H, device=device)
    # hooks BEFORE wrapping (they live on inner modules and survive FSDP)
    model.install_hooks(hidden_layers=need["hidden_layers"],
                        kv_layers=need["kv_layers"], attn_layers=need["attn_layers"])
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32)
    model = FSDP(
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
    if digit_vecs is not None:
        dist.broadcast(digit_vecs, src=0)
    return model, digit_vecs


# ---------------------------------------------------------------- q35b ----

def build_q35b(rank, device, feats: FeatureConfig, need):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer)
    if rank == 0:
        free_kb = 0
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                free_kb = int(line.split()[1])
        if free_kb and free_kb / 1e6 < 200:
            raise RuntimeError(
                f"rank0 host RAM too low for fp32 35B load ({free_kb/1e6:.0f}G free; need ~200G)")
        lm, _ = load_q35b_text(dtype=torch.float32, device="cpu")
        log("q35b STRICT load audit PASS (missing=[] unexpected=[])")
    else:
        lm, _ = load_q35b_text(dtype=torch.float32, meta_ok=True)
    model = Q35BMech(lm)
    model.mask_dtype = torch.bfloat16  # == MixedPrecision.param_dtype; masks are
    # built inside forward, so FSDP's root input-cast never reaches them (the
    # U6 "invalid dtype for bias" lesson, 2026-07-06)
    # hooks BEFORE wrapping (they live on inner modules and survive FSDP)
    capture_layers = sorted(set(need["kv_layers"]) | set(need["attn_layers"])
                            | set(need["hidden_layers"]))
    model.install_hooks(hidden_layers=need["hidden_layers"],
                        kv_layers=need["kv_layers"], attn_layers=need["attn_layers"])
    vocab = lm.config.vocab_size

    def policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return (isinstance(module, Qwen3_5MoeDecoderLayer)
                or isinstance(module, nn.Embedding)
                or (isinstance(module, nn.Linear) and module.out_features == vocab))

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=(torch.bfloat16 if ARGS.grad_reduce == "bf16" else torch.float32),
        buffer_dtype=torch.float32,
        keep_low_precision_grads=(ARGS.grad_reduce == "bf16"),
    )
    model = FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        param_init_fn=(None if rank == 0 else
                       lambda m: m.to_empty(device=device, recurse=False)),
        limit_all_gathers=True,
    )
    if ARGS.act_ckpt:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
        skip_ids = set()
        for m in model.modules():
            if isinstance(m, Qwen3_5MoeDecoderLayer):
                li = getattr(getattr(m, "self_attn", None), "layer_idx", None)
                if li is None:
                    li = getattr(getattr(m, "linear_attn", None), "layer_idx", None)
                if li in capture_layers:
                    skip_ids.add(id(m))  # capture layers NOT checkpointed:
                    # checkpointed modules re-run their forward in backward and the
                    # first (no-grad) pass would poison hook captures' grad graph
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, Qwen3_5MoeDecoderLayer)
            and id(m) not in skip_ids)
        log(f"activation checkpointing ON (skipped capture layers: {sorted(capture_layers)})")
    if os.environ.get("D3_BWD_TRACE") == "1" and rank == 0:
        # D3 spike diagnostics: which layer/pass the backward reached (the
        # OOM-point instrument for the morning report). Print-only, rank 0.
        def _mk_trace(idx):
            def hook(_mod, _gout):
                a = torch.cuda.memory_allocated() / 1e9
                print(f"[d3-bwd-trace] layer {idx} backward-pre alloc={a:.1f}G",
                      flush=True)
            return hook
        n_traced = 0
        for m in model.modules():
            if isinstance(m, Qwen3_5MoeDecoderLayer):
                li = getattr(getattr(m, "self_attn", None), "layer_idx", None)
                if li is None:
                    li = getattr(getattr(m, "linear_attn", None), "layer_idx", None)
                m.register_full_backward_pre_hook(_mk_trace(li))
                n_traced += 1
        log(f"D3_BWD_TRACE backward-pre hooks installed on {n_traced} layers (rank0)")
    return model


# ---------------------------------------------------------------- data ----

def load_marin_data():
    pref = "" if ARGS.task == "sub" else "arith_"
    train_rows = [json.loads(l) for l in open(f"{ARGS.data_dir}/{pref}train.jsonl")]
    val_rows = [json.loads(l) for l in open(f"{ARGS.data_dir}/{pref}val.jsonl")]
    bands = sorted({r["band"] for r in train_rows})
    by_band = {d: [r for r in train_rows if r["band"] == d] for d in bands}
    val_by_band = {d: [r for r in val_rows if r["band"] == d] for d in bands}
    return bands, by_band, val_by_band


def load_q35_data():
    path = ARGS.q35_train or f"{CB_DATA}/{ARGS.band}_capped_train.jsonl"
    rows = [json.loads(l) for l in open(path)]
    train = rows[: ARGS.q35_train_n]
    val = rows[ARGS.q35_train_n: ARGS.q35_train_n + 100]
    return train, val


def load_marin_cb_data():
    """Canonical comp-bench pool adapted to marin item schema. Rows [0:N] are
    the SAME items and ORDER as the engine C0 / 35B sandbox C0 protocol; val =
    the next 100 rows (in-band CE telemetry)."""
    path = ARGS.q35_train or f"{CB_DATA}/{ARGS.band}_capped_train64k.jsonl"
    if not os.path.exists(path):
        path = f"{CB_DATA}/{ARGS.band}_capped_train.jsonl"
    rows = [json.loads(l) for l in open(path)]

    def adapt(r):
        return {"prompt": r["problem"], "gold": str(r["answer"]),
                "band": r["n_ops"], "k": r["n_ops"], "n_ops": r["n_ops"],
                "trace": r.get("trace"), "sig": f"cb{r['idx']}", "idx": r["idx"]}

    train = [adapt(r) for r in rows[: ARGS.marin_train_n]]
    val = [adapt(r) for r in rows[ARGS.marin_train_n: ARGS.marin_train_n + 100]]
    return train, val


def parse_coef_schedule(spec: str):
    """'slot_value=1@0,1@25,0@50;slot_hidden=1@0,0@150' -> {name: [(step, v)]}
    Piecewise-linear between knots; constant beyond the last knot."""
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, pts = part.split("=")
        knots = []
        for p in pts.split(","):
            v, s = p.split("@")
            knots.append((int(s), float(v)))
        out[name.strip()] = sorted(knots)
    return out


def coef_at(knots, step: int) -> float:
    if step <= knots[0][0]:
        return knots[0][1]
    for (s0, v0), (s1, v1) in zip(knots, knots[1:]):
        if step <= s1:
            return v0 + (v1 - v0) * (step - s0) / max(1, s1 - s0)
    return knots[-1][1]


def attach_teachers(batch, teacher_dir, feats: FeatureConfig, device):
    """Optional per-item teacher bundles: {teacher_dir}/th_{idx:06d}.pt with
    keys 'hiddens' [m,H] (fp16/bf16 ok), 'k_l{L}'/'v_l{L}' [m,heads,dim],
    'attn_target' [k]."""
    B = batch["input_ids"].shape[0]
    hid, kv, att = [None] * B, [None] * B, [None] * B
    if teacher_dir:
        for b, idx in enumerate(batch.get("idxs", [None] * B)):
            if idx is None:
                continue
            p = os.path.join(teacher_dir, f"th_{idx:06d}.pt")
            if not os.path.exists(p):
                continue
            d = torch.load(p, map_location="cpu")
            if "hiddens" in d and feats.slot_hidden_coef:
                hid[b] = d["hiddens"].to(device=device, dtype=torch.float32)
            if feats.kv_match_coef:
                kv[b] = {L: (d[f"k_l{L}"].to(device=device, dtype=torch.float32),
                             d[f"v_l{L}"].to(device=device, dtype=torch.float32))
                         for L in feats.layer_list(feats.kv_layers)
                         if f"k_l{L}" in d}
            if "attn_target" in d and feats.attn_target_coef:
                att[b] = d["attn_target"].to(device=device, dtype=torch.float32)
    batch["teacher_hiddens"] = hid
    batch["teacher_kv"] = kv
    batch["attn_targets"] = att
    return batch


# -------------------------------------------------------------- saving ----

def disk_ok():
    return shutil.disk_usage("/shared").free / 1e9 >= 250


def save_ckpt(model, path, rank, strip_prefix=""):
    if not disk_ok():
        log(f"[SAVE-SKIP] <250G free on /shared, skipping {path}")
        return
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if rank == 0:
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out, cur, shard_id, weight_map = {}, 0, 1, {}
        base = path[:-len(".safetensors")] if path.endswith(".safetensors") else path
        single = sum(v.numel() * 2 for v in sd.values()) < 12e9
        items = []
        for k, v in sd.items():
            kk = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
            items.append((kk, v.to(torch.bfloat16).contiguous()))
        if single:
            save_file(dict(items), path)
            log(f"saved {path} ({len(items)} tensors)")
        else:
            os.makedirs(base, exist_ok=True)
            for k, v in items:
                out[k] = v
                cur += v.numel() * v.element_size()
                if cur >= 8e9:
                    name = f"model-{shard_id:05d}.safetensors"
                    save_file(out, os.path.join(base, name))
                    weight_map.update({kk: name for kk in out})
                    out, cur, shard_id = {}, 0, shard_id + 1
            if out:
                name = f"model-{shard_id:05d}.safetensors"
                save_file(out, os.path.join(base, name))
                weight_map.update({kk: name for kk in out})
            json.dump({"metadata": {}, "weight_map": weight_map},
                      open(os.path.join(base, "model.safetensors.index.json"), "w"))
            log(f"saved sharded {base} ({len(items)} tensors, {shard_id} shards)")
    dist.barrier()


# ---------------------------------------------------------------- main ----

def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["marin", "q35b"], required=True)
    # marin task/arm (recur-compatible). "mr" = MULTI-READER demand-diversity
    # arm (ms_mr_data.py; marin + --marin-data cb only; additive 2026-07-07)
    ap.add_argument("--arm", choices=["c0", "mech", "mr"], default="c0")
    ap.add_argument("--task", choices=["sub", "arith"], default="sub")
    ap.add_argument("--data-dir", default="/shared/apanda/filler_grpo/recur_stage0/data")
    # marin data source: 'recur' = recur_stage0 jsonls with band-rng sampling
    # (the gate-i parity path, bit-compatible with recur_stage0_train); 'cb' =
    # the canonical comp-bench band pool, SEQUENTIAL slices (identical items +
    # order to the engine/35B C0 protocol) — the attempt-1 arm path
    ap.add_argument("--marin-data", choices=["recur", "cb"], default="recur")
    ap.add_argument("--marin-train-n", type=int, default=12800)
    # q35b data
    ap.add_argument("--band", default="nops05-07")
    ap.add_argument("--q35-train", default=None)
    ap.add_argument("--q35-train-n", type=int, default=12800)
    ap.add_argument("--slot-digits", type=int, default=4,
                    help="digit slots per intermediate for built-in q35 value labels")
    ap.add_argument("--slot-labels", choices=["none", "trace-digits"], default="none")
    ap.add_argument("--slot-hidden-targets", choices=["none", "teacher-dir", "trace-digit-embeds"],
                    default="none",
                    help="trace-digit-embeds: targets = input-embedding rows of the "
                         "digit tokens of the parse-tree intermediates (arm A)")
    # piecewise-linear coefficient schedules, e.g.
    #   --coef-schedule "slot_value=1@0,1@25,0@50;slot_hidden=1@0,1@50,0@150"
    ap.add_argument("--coef-schedule", default="")
    # arm-B warmup: force k = k-fixed-mult * base_k for steps < k-fixed-until,
    # sampling starts after (win-condition k-sampling for the remaining steps)
    ap.add_argument("--k-fixed-until", type=int, default=0)
    ap.add_argument("--k-warmup-mult", type=int, default=4)
    # MR (multi-reader) knobs — consumed only when --arm mr (ms_mr_data.py)
    ap.add_argument("--mr-demands",
                    default="value,signparity,comparison,offset,continuation",
                    help="MR-full = all five; MR-single ablation = 'value'")
    ap.add_argument("--mr-mask-datums", choices=["demand", "all", "none"],
                    default="demand",
                    help="which rows carry the answer->problem block (pair with "
                         "--features '{\"mask_blocks\":\"answer->problem\"}')")
    ap.add_argument("--mr-p-answer", type=float, default=0.5)
    ap.add_argument("--mr-answer-only", action="store_true",
                    help="slotted-C0 rows: every datum an answer datum")
    # features
    ap.add_argument("--features", default="{}", help="FeatureConfig JSON overrides")
    ap.add_argument("--features-file", default=None)
    ap.add_argument("--slot-teacher-dir", default=None)
    # recipe
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--per-rank-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--q35-lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-steps", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-hours", type=float, default=6.0)
    # q35b engineering
    ap.add_argument("--grad-reduce", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--split-backward", action="store_true",
                    help="two-pass quantizer memory fix (D3 spike): cut the "
                         "graph at the snapped embeddings and run backward "
                         "pass-B-then-pass-A so FSDP1 reduce-scatters each "
                         "layer's grad promptly instead of holding ~n_layers "
                         "unsharded bf16 grads across the double-pass sweep. "
                         "Gradient equality vs the joint graph is gated by "
                         "ms_split_grad_gate.py (run it first).")
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--act-ckpt", action="store_true", default=None)
    ap.add_argument("--no-act-ckpt", dest="act_ckpt", action="store_false")
    ap.add_argument("--no-final-ckpt", action="store_true",
                    help="skip the final checkpoint save (smoke runs)")
    ARGS = ap.parse_args()
    if ARGS.act_ckpt is None:
        ARGS.act_ckpt = (ARGS.model == "q35b")

    fdict = json.loads(ARGS.features)
    if ARGS.features_file:
        fdict = {**json.load(open(ARGS.features_file)), **fdict}
    feats = FeatureConfig(**fdict)
    if ARGS.arm == "c0" and ARGS.model == "marin":
        assert feats.is_off(), "marin --arm c0 is the parity arm: features must be OFF"
    if ARGS.arm == "mr":
        assert ARGS.model == "marin" and ARGS.marin_data == "cb", \
            "MR arms are marin + --marin-data cb only (exact masks + trace)"
        # bypass-kill ON = mask_blocks="answer->problem" (+ per-row policy in
        # the builder); the no-mask ablation arm = empty mask_blocks
        assert feats.mask_blocks in ("", "answer->problem"), feats.mask_blocks

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(ARGS.seed + rank)
    t_start = time.time()
    ckpt_steps = {int(x) for x in ARGS.ckpt_steps.split(",") if x.strip()}
    run_name = ARGS.run_name or (
        f"{ARGS.model}_{ARGS.arm}_{ARGS.task if ARGS.model == 'marin' else ARGS.band}")
    os.makedirs(f"{ARGS.out_dir}/ckpts", exist_ok=True)
    os.makedirs(f"{ARGS.out_dir}/logs", exist_ok=True)
    log_f = open(f"{ARGS.out_dir}/logs/train_{run_name}.jsonl", "a") if rank == 0 else None

    def emit(payload):
        if rank == 0:
            log_f.write(json.dumps(payload) + "\n")
            log_f.flush()

    emit({"kind": "config", "run": run_name, "features": feats.to_json(),
          "argv": sys.argv, "world": world})
    log(f"run {run_name} features_off={feats.is_off()} feats={feats.to_json()}")

    need = need_from_feats(feats)

    # ---------------- coefficient schedules + warmup-k policy ----------------
    schedules = parse_coef_schedule(ARGS.coef_schedule)
    import dataclasses as _dc

    def feats_for_step(step: int) -> FeatureConfig:
        f = feats
        upd = {}
        if "slot_value" in schedules:
            upd["slot_value_coef"] = coef_at(schedules["slot_value"], step)
        if "slot_hidden" in schedules:
            upd["slot_hidden_coef"] = coef_at(schedules["slot_hidden"], step)
        if ARGS.k_fixed_until and step < ARGS.k_fixed_until and f.k_mode == "sample":
            upd["k_mode"] = "fixed"
            upd["k_fixed_mult"] = ARGS.k_warmup_mult
        return _dc.replace(f, **upd) if upd else f

    # `need` must cover the schedule MAXIMA (captures stay installed even when
    # a coefficient anneals to 0 — assembly just skips the loss)
    need_feats = feats
    if schedules:
        maxima = {}
        if "slot_value" in schedules:
            maxima["slot_value_coef"] = max(v for _, v in schedules["slot_value"])
        if "slot_hidden" in schedules:
            maxima["slot_hidden_coef"] = max(v for _, v in schedules["slot_hidden"])
        need_feats = _dc.replace(feats, **maxima)

    # ---------------- model + data ----------------
    if ARGS.model == "marin":
        tok = load_tokenizer(RL_CKPT)
        digit_token_ids = None
        if ARGS.slot_hidden_targets == "trace-digit-embeds":
            from mechanism_forward import DIGIT_CHARS
            digit_token_ids = []
            for ch in DIGIT_CHARS:
                ids = tok.encode(ch, add_special_tokens=False)
                assert len(ids) == 1, f"digit char {ch!r} is not a single token: {ids}"
                digit_token_ids.append(ids[0])
        if ARGS.marin_data == "cb":
            cb_train, cb_val = load_marin_cb_data()
            bands, by_band, val_by_band = [], {}, {}
            log(f"marin cb data: {len(cb_train)} train rows band={ARGS.band} (sequential)")
        else:
            bands, by_band, val_by_band = load_marin_data()
        model, digit_vecs = build_marin(rank, device,
                                        need_from_feats(need_feats),
                                        digit_token_ids=digit_token_ids)
        backend = model.module if hasattr(model, "module") else model
        base_lr = ARGS.lr
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(Q35_BASE_SNAPSHOT)
        train_rows, val_rows = load_q35_data()
        model = build_q35b(rank, device, feats, need_from_feats(need_feats))
        backend = model.module if hasattr(model, "module") else model
        base_lr = ARGS.q35_lr
        digit_vecs = None
        log(f"q35b data: {len(train_rows)} train rows band={ARGS.band}")
    need = need_from_feats(need_feats)
    log("FSDP wrapped")

    params = [p for p in model.parameters() if p.requires_grad]
    if ARGS.optim == "adamw":
        opt = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=0.0,
                                fused=(ARGS.model == "q35b"))
    else:
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=base_lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)

    rng_rank = random.Random(ARGS.seed * 100 + rank)  # marin: identical to recur_stage0

    def slot_labels_fn(item, ki):
        if ARGS.slot_labels == "trace-digits":
            return digit_slot_labels_from_trace(tok, item, ki, digits=ARGS.slot_digits)
        return None

    def k_base_fn(item):
        """base k = n_ops (the dose ladder's unit). Digit value-labels fire
        only when the realized k == slot_digits * n_ops (arm-B warmup uses
        --k-warmup-mult == slot_digits to guarantee that)."""
        if ARGS.arm == "c0":
            return 0
        return item.get("n_ops", 0) or item.get("k", item.get("band", 0))

    def make_marin_batch(items, step, f: FeatureConfig):
        if ARGS.arm == "mr":
            # MULTI-READER mixed batch (ms_mr_data.py): demand + answer datums,
            # per-row bypass-kill policy; k/demand sampling is builder-internal
            # (deterministic in seed/step/row) — the sample_k path below is
            # NOT used for MR
            from ms_mr_data import build_mr_batch
            b = build_mr_batch(
                tok, items, device, seed=ARGS.seed, step=step,
                demand_types=tuple(x.strip() for x in ARGS.mr_demands.split(",")
                                   if x.strip()),
                p_answer=ARGS.mr_p_answer, mask_policy=ARGS.mr_mask_datums,
                answer_only=ARGS.mr_answer_only)
            b["items"] = items
            return b
        if ARGS.arm != "c0":
            items = [dict(it) for it in items]
            for r_i, it in enumerate(items):
                base_k = k_base_fn(it) or it.get("k", it.get("band", 0))
                it["k"] = sample_k(base_k, f, ARGS.seed, step, r_i)
                sl_ids = slot_labels_fn(it, it["k"])
                if sl_ids:
                    it["slot_label_ids"] = sl_ids
        b = build_batch(tok, items, "r1" if ARGS.arm == "mech" else "c0", device,
                        with_answer=True)
        marin_annotate_spans(tok, b, items, "r1" if ARGS.arm == "mech" else "c0")
        # optional value labels carried on items as 'slot_label_ids'
        sl = torch.full_like(b["labels"], -100)
        for r_i, it in enumerate(items):
            ids = it.get("slot_label_ids")
            if ids:
                P = b["P"]
                for j, lab in enumerate(ids[: b["ks"][r_i]]):
                    sl[r_i, P + j] = lab
        b["slot_labels"] = sl
        b["idxs"] = [it.get("idx") for it in items]
        b["items"] = items
        return b

    def attach_digit_embed_teachers(batch, f: FeatureConfig):
        """arm A: hidden targets = embedding rows of the trace digit tokens
        (m = digits * n_ops rows per item; the set-matcher handles k < m via
        transpose assignment)."""
        from mechanism_forward import trace_digit_char_indices
        B = batch["input_ids"].shape[0]
        th = [None] * B
        if f.slot_hidden_coef and digit_vecs is not None:
            for b_i, it in enumerate(batch.get("items", [None] * B)):
                if it is None:
                    continue
                ci = trace_digit_char_indices(it, digits=ARGS.slot_digits)
                if ci:
                    th[b_i] = digit_vecs[torch.tensor(ci, device=device)]
        batch["teacher_hiddens"] = th
        return batch

    def forward_losses(batch, f: FeatureConfig):
        # ROOT FSDP forward: all parameter-touching compute happens inside;
        # assembly afterwards uses captured activations only (FSDP-safe).
        out = model(batch, f, need, split_backward=ARGS.split_backward)
        if ARGS.slot_hidden_targets == "trace-digit-embeds":
            attach_digit_embed_teachers(batch, f)
            batch.setdefault("teacher_kv", [None] * batch["input_ids"].shape[0])
            batch.setdefault("attn_targets", [None] * batch["input_ids"].shape[0])
        else:
            attach_teachers(batch, ARGS.slot_teacher_dir, f, device)
        losses = assemble_losses(backend, out, batch, f)
        from mechanism_forward import snap_stats_summary
        ss = snap_stats_summary(out.get("capture", {}))
        if ss:
            losses["_snap_stats"] = ss
        return losses, out

    def split_backward_step(losses, fwd_out, f: FeatureConfig):
        """Two-backward schedule for the graph cut at the snapped embeddings.
        B-side terms (reader-pass graph): ce + attn (rows_probs uses reader
        captures under quantize). A-side terms (writer-pass graph): slot_value
        (passA h_final), slot_hidden (writer_hidden), kv (passA captures).
        Composition == joint backward by the STE identity; equality gated on
        marin by ms_split_grad_gate.py."""
        from mechanism_forward import (fsdp_split_backward_hook_graph,
                                       fsdp_split_backward_rearm,
                                       fsdp_split_backward_save,
                                       fsdp_split_install_jit_hooks,
                                       fsdp_split_jit_arm,
                                       fsdp_split_jit_disarm)
        fsdp_split_install_jit_hooks(model)
        loss_b = losses["ce"]
        if "attn" in losses and f.attn_target_coef:
            loss_b = loss_b + f.attn_target_coef * losses["attn"]
        saved = fsdp_split_backward_save(model)
        fsdp_split_jit_arm()
        hooks_b = fsdp_split_backward_hook_graph(model, saved, [loss_b])
        loss_b.backward()
        for hd in hooks_b:
            hd.remove()
        fsdp_split_jit_disarm()
        fsdp_split_backward_rearm(saved)
        g_leaf = fwd_out["h1_leaf"].grad
        assert g_leaf is not None, \
            "split-backward: no gradient reached the snapped embeddings"
        parts_a = []
        if "slot_value" in losses and f.slot_value_coef:
            parts_a.append(f.slot_value_coef * losses["slot_value"])
        if "slot_hidden" in losses and f.slot_hidden_coef:
            parts_a.append(f.slot_hidden_coef * losses["slot_hidden"])
        if "kv" in losses and f.kv_match_coef:
            parts_a.append(f.kv_match_coef * losses["kv"])
        tot_a = torch.stack(parts_a).sum() if parts_a else None
        fsdp_split_jit_arm()
        hooks_a = fsdp_split_backward_hook_graph(
            model, saved, [fwd_out["h1_graph"], tot_a])
        if tot_a is not None:
            torch.autograd.backward([fwd_out["h1_graph"], tot_a],
                                    [g_leaf, torch.ones_like(tot_a)])
        else:
            torch.autograd.backward(fwd_out["h1_graph"], g_leaf)
        for hd in hooks_a:
            hd.remove()
        fsdp_split_jit_disarm()

    def run_val(step):
        model.eval()
        f = feats_for_step(step)
        rows = {}
        with torch.no_grad():
            if ARGS.model == "marin" and ARGS.marin_data == "cb":
                for c0 in range(0, min(len(cb_val), 64), ARGS.per_rank_bs):
                    items = cb_val[c0: c0 + ARGS.per_rank_bs]
                    b = make_marin_batch(items, step, f)
                    l, _ = forward_losses(b, f)
                    rows.setdefault("val_ce", []).append(float(l["ce"]))
                rows["val_ce"] = round(sum(rows["val_ce"]) / len(rows["val_ce"]), 4)
            elif ARGS.model == "marin":
                for d in bands:
                    b = make_marin_batch(val_by_band[d], step, f)
                    l, _ = forward_losses(b, f)
                    rows[f"d{d}_r1"] = round(float(l["ce"]), 4)
            else:
                for c0 in range(0, min(len(val_rows), 64), ARGS.per_rank_bs):
                    items = val_rows[c0: c0 + ARGS.per_rank_bs]
                    b = build_batch_q35(tok, items, device, feats=f, seed=ARGS.seed,
                                        step=-1, slot_labels_fn=slot_labels_fn,
                                        k_base_fn=k_base_fn)
                    l, _ = forward_losses(b, f)
                    rows.setdefault("val_ce", []).append(float(l["ce"]))
                rows["val_ce"] = round(sum(rows["val_ce"]) / len(rows["val_ce"]), 4)
        model.train()
        emit({"kind": "val", "step": step, "run": run_name, **rows})
        log(f"val s{step}: {rows}")

    model.train()
    if ARGS.val_every > 0:
        run_val(0)
    tokens_cum, sec_cum = 0, 0.0
    for step in range(ARGS.steps):
        t0 = time.time()
        scale = min(1.0, (step + 1) / max(ARGS.warmup, 1))
        for g in opt.param_groups:
            g["lr"] = base_lr * scale
        f_step = feats_for_step(step)
        # ---- batch ----
        if ARGS.model == "marin" and ARGS.marin_data == "cb":
            bs_g = world * ARGS.per_rank_bs
            start = (step * bs_g + rank * ARGS.per_rank_bs) % len(cb_train)
            items = [cb_train[(start + i) % len(cb_train)]
                     for i in range(ARGS.per_rank_bs)]
            batch = make_marin_batch(items, step, f_step)
            d = ARGS.band
        elif ARGS.model == "marin":
            d = rng_rank.choice(bands)
            items = [by_band[d][rng_rank.randrange(len(by_band[d]))]
                     for _ in range(ARGS.per_rank_bs)]
            batch = make_marin_batch(items, step, f_step)
        else:
            bs_g = world * ARGS.per_rank_bs
            start = (step * bs_g + rank * ARGS.per_rank_bs) % len(train_rows)
            items = [train_rows[(start + i) % len(train_rows)]
                     for i in range(ARGS.per_rank_bs)]
            batch = build_batch_q35(tok, items, device, feats=f_step, seed=ARGS.seed,
                                    step=step, slot_labels_fn=slot_labels_fn,
                                    k_base_fn=k_base_fn)
            d = ARGS.band
        losses, fwd_out = forward_losses(batch, f_step)
        loss = losses["total"]
        finite = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1:
            log(f"step {step}: NONFINITE loss, skipping optim step")
            opt.zero_grad(set_to_none=True)
            continue
        if ARGS.split_backward and "h1_leaf" in fwd_out:
            split_backward_step(losses, fwd_out, f_step)
        else:
            # JOINT path (2026-07-07 fix): under MixedPrecision, FSDP1 binds
            # a FRESH AccumulateGrad per forward while its reduce-scatter
            # hook stays on the FIRST forward's node — any unit forwarding
            # >=2x per step (lm_head in the quantizer arm: snap-logits +
            # answer-CE) silently loses its grad if that node never fires.
            # Re-arm on the ACTUAL graph before backward (no-op for
            # fp32/no-MP and single-forward units; gated by
            # ms_split_grad_gate.py --which jointmp). The --split-backward
            # path is unchanged.
            from mechanism_forward import fsdp_joint_backward_arm_hooks
            hooks_j, refs_j = fsdp_joint_backward_arm_hooks(model, [loss])
            loss.backward()
            for hd in hooks_j:
                hd.remove()
            del hooks_j, refs_j
        fwd_out = None  # drop capture/graph references before the optim step
        if ARGS.grad_reduce == "bf16":
            for p in params:  # fused AdamW needs fp32 grads on fp32 master params
                if p.grad is not None and p.grad.dtype != torch.float32:
                    p.grad = p.grad.to(torch.float32)
        model.clip_grad_norm_(ARGS.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # ---- telemetry ----
        step_tokens = int(batch["real"].sum()) * world  # approx: rank-symmetric batches
        sec = time.time() - t0
        tokens_cum += step_tokens
        sec_cum += sec
        ce_mean = losses["ce"].detach().clone()
        dist.all_reduce(ce_mean)
        ce_mean = float(ce_mean) / world
        if rank == 0:
            payload = {"kind": "train", "step": step, "run": run_name,
                       "d_rank0": d, "ce_rank0": round(float(losses["ce"].detach()), 4),
                       "ce_mean": round(ce_mean, 4),
                       "lr": opt.param_groups[0]["lr"], "sec": round(sec, 2),
                       "tok_s": round(step_tokens / sec, 1),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                       "mem_rsvd_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
            for k in ("slot_value", "slot_hidden", "kv", "attn"):
                if k in losses:
                    payload[k] = round(float(losses[k].detach()), 5)
            if "_snap_stats" in losses:
                payload.update(losses["_snap_stats"])
            if f_step is not feats:
                payload["coef_slot_value"] = f_step.slot_value_coef
                payload["coef_slot_hidden"] = f_step.slot_hidden_coef
                payload["k_mode_step"] = f_step.k_mode
            payload["ks_rank0"] = batch.get("ks")
            if ARGS.arm == "mr" and batch.get("meta"):
                payload["mr_kinds_rank0"] = [
                    (m0.get("demand") or "answer") for m0 in batch["meta"]]
                payload["mr_masked_rank0"] = sum(batch.get("masked_rows", []))
            emit(payload)
            if step % 10 == 0 or step == ARGS.steps - 1:
                log(f"step {step} d0={d} ce={ce_mean:.4f} ({sec:.1f}s, "
                    f"{payload['tok_s']} tok/s, {payload['mem_gb']}G)")
        if ARGS.val_every > 0 and (step + 1) % ARGS.val_every == 0:
            run_val(step + 1)
        if (step + 1) in ckpt_steps:
            save_ckpt(model, f"{ARGS.out_dir}/ckpts/{run_name}_s{step+1:04d}.safetensors",
                      rank, strip_prefix="lm." if ARGS.model == "q35b" else "")
        if (time.time() - t_start) / 3600 > ARGS.max_hours:
            log(f"WALL CAP hit at step {step}; saving and exiting")
            break

    if not ARGS.no_final_ckpt:
        save_ckpt(model, f"{ARGS.out_dir}/ckpts/{run_name}_final.safetensors", rank,
                  strip_prefix="lm." if ARGS.model == "q35b" else "")
    emit({"kind": "done", "run": run_name,
          "tok_s_avg": round(tokens_cum / max(sec_cum, 1e-9), 1),
          "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    log(f"DONE {run_name} avg_tok_s={tokens_cum / max(sec_cum, 1e-9):.1f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
