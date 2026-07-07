#!/usr/bin/env python3
"""MECHANISM-SANDBOX unit gates (2026-07-06). Every feature has an
independently runnable gate; CPU tiny-model gates run anywhere; --gpu gates
run on an idle stack via a staged job.

  U1 set-matcher   : Hungarian vs brute force (cost equality); permutation
                     recovery; order-invariance of set_match_loss; Sinkhorn
                     plan sanity.
  U2 quantizer     : hard-snap outputs BIT-equal to embedding rows
                     (torch.equal); STE gradient == identity; top-k mixture
                     differentiable, weights sum to 1.
  U3 mask          : tiny-dense (marin-arch) — blocked attention probs are
                     EXACT zeros at every layer; full-bottleneck
                     (answer->prompt;slots->prompt) makes answer states
                     BIT-invariant to problem-content substitution.
                     tiny-hybrid (q35 arch) — full-attn probs exact zeros AND
                     the substitution DOES change answer states = the GDN
                     leak, demonstrated and measured.
  U4 parity        : tiny-dense MechModel features-off vs plain HF forward
                     (fp32 CPU, allclose + top-1); tiny-hybrid hooks-installed
                     features-off vs plain HF forward BIT-identical; 4D-causal
                     mask == default-mask path at delta 0.0.
  U4b attn-recompute: rows_probs_from_capture vs HF eager attn_weights
                     (tiny-hybrid, attn_implementation='eager').
  U5 loss plumbing : kv/attn/hidden/value losses finite + grads flow to the
                     expected parameter tensors.
  U6 (--gpu)       : real-checkpoint gates — marin MechModel-vs-HF parity
                     (top-1) and 35B strict-load + coherence + hook
                     bit-identity + mask-prob inspection + gdn_leak_probe on
                     real band items (--real marin|q35b).

Usage:
  python mechanism_sandbox_gates.py --which u1,u2,u3,u4,u5      # CPU
  python mechanism_sandbox_gates.py --which u6 --real q35b --gpu
Writes a JSON verdict to --out (default ./mechanism_gates_report.json).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/apanda/xorl-client")

import torch
import torch.nn.functional as F

from mechanism_forward import (
    FeatureConfig,
    MarinMechModel,
    Q35BMech,
    Q35_BASE_SNAPSHOT,
    aligned_loss,
    assemble_losses,
    attn_target_kl,
    build_batch_q35,
    digit_slot_labels_from_trace,
    gdn_leak_probe,
    hungarian_assign,
    kv_prerope_from_capture,
    load_q35b_text,
    need_from_feats,
    rows_probs_from_capture,
    set_match_loss,
    sinkhorn_plan,
    snap_to_vocab,
)

REPORT = {}


def gate(name):
    def deco(fn):
        def wrapper(*a, **k):
            t0 = time.time()
            try:
                res = fn(*a, **k)
                REPORT[name] = {"pass": True, "sec": round(time.time() - t0, 2), **(res or {})}
                print(f"[GATE {name}] PASS {json.dumps(REPORT[name])}", flush=True)
            except Exception as e:
                REPORT[name] = {"pass": False, "error": f"{type(e).__name__}: {e}",
                                "sec": round(time.time() - t0, 2)}
                print(f"[GATE {name}] FAIL {REPORT[name]['error']}", flush=True)
            return REPORT[name]
        return wrapper
    return deco


# ------------------------------------------------------------- tiny models

def tiny_dense():
    """marin-arch tiny model (Qwen3 dense, q_norm/k_norm, no GQA)."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, head_dim=16, intermediate_size=128,
                      vocab_size=512, max_position_embeddings=256, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


def tiny_hybrid():
    """q35-arch tiny model: 8 layers, full attention at [3,7], 6 GDN."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    torch.manual_seed(4321)
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=512, moe_intermediate_size=32, num_experts=8,
        num_experts_per_tok=2, shared_expert_intermediate_size=32,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=2, full_attention_interval=4,
        max_position_embeddings=256, intermediate_size=64, tie_word_embeddings=False)
    m = Qwen3_5MoeForCausalLM(cfg).eval()
    return m


def synth_batch(B=2, L=20, vocab=512, seed=7, device="cpu"):
    """Synthetic batch: prompt [0,10) with problem [2,8); slots [10,14);
    answer region [14,L)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (B, L), generator=g)
    real = torch.ones(B, L, dtype=torch.bool)
    pos = torch.arange(L)[None].expand(B, -1).contiguous()
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 14:L - 1] = ids[:, 15:L]
    spans = [{"problem": (2, 8), "prompt_end": 10, "slots": (10, 14),
              "answer": (14, L)} for _ in range(B)]
    return {"input_ids": ids.to(device), "real": real.to(device), "pos": pos.to(device),
            "labels": labels.to(device), "slot_labels": torch.full((B, L), -100).to(device),
            "spans": spans}


# ------------------------------------------------------------------- U1

@gate("U1_set_matcher")
def u1():
    torch.manual_seed(0)
    # (a) brute force equality on random rectangular instances
    for trial in range(200):
        m = torch.randint(1, 6, (1,)).item()
        k = m + torch.randint(0, 3, (1,)).item()
        cost = torch.rand(m, k)
        assign = hungarian_assign(cost)
        assert len(set(assign)) == m, "assignment not injective"
        hcost = sum(float(cost[i, j]) for i, j in enumerate(assign))
        best = min(sum(float(cost[i, p[i]]) for i in range(m))
                   for p in itertools.permutations(range(k), m))
        assert abs(hcost - best) < 1e-6, (trial, hcost, best)
    # (b) permutation recovery
    H, m = 32, 8
    T = torch.randn(m, H)
    perm = torch.randperm(m)
    S = T[perm] + 0.01 * torch.randn(m, H)
    assign = hungarian_assign(torch.cdist(T, S).pow(2))
    recovered = torch.tensor(assign)
    want = torch.argsort(torch.argsort(perm))  # position of i in perm
    # slot j holds teacher perm[j]; assignment of teacher i must be slot where perm[slot]=i
    inv = torch.empty(m, dtype=torch.long)
    inv[perm] = torch.arange(m)
    assert torch.equal(recovered, inv), (recovered, inv)
    # (c) order invariance of the loss (the scrambled-alignment property)
    S2 = torch.randn(12, H)
    T2 = torch.randn(8, H)
    l1, _ = set_match_loss(S2, T2, "mse", "hungarian")
    l2, _ = set_match_loss(S2, T2[torch.randperm(8)], "mse", "hungarian")
    assert torch.allclose(l1, l2, atol=1e-6), (l1, l2)
    # (d) sinkhorn: doubly-stochastic-ish; sharp instance ~ permutation
    cost = torch.cdist(T, T[perm]).pow(2)
    plan = sinkhorn_plan(cost, tau=0.01, iters=200)
    assert plan.shape == (m, m)
    assert (plan.sum(1) - 1.0 / m).abs().max() < 1e-3
    hard = plan.argmax(1)
    assert torch.equal(hard, inv[torch.arange(m)]) or torch.equal(hard, perm.argsort()), \
        "sinkhorn low-tau plan does not concentrate on the permutation"
    # (e) sinkhorn loss finite + grad flows through cost
    S3 = torch.randn(10, H, requires_grad=True)
    l3, _ = set_match_loss(S3, T2, "mse", "sinkhorn", tau=0.1, iters=50)
    l3.backward()
    assert S3.grad is not None and torch.isfinite(S3.grad).all()
    return {"trials": 200}


# ------------------------------------------------------------------- U2

@gate("U2_quantizer")
def u2():
    torch.manual_seed(0)
    V, H, N = 512, 64, 10
    E = torch.randn(V, H)
    embed = torch.nn.Embedding(V, H)
    embed.weight.data.copy_(E)
    head = torch.nn.Linear(H, V, bias=False)
    h = torch.randn(N, H, requires_grad=True)
    # hard snap, cosine metric: BIT-level row equality
    out, idx = snap_to_vocab(h, "hard", 8, 1.0, "cosine", embed_mod=embed,
                             embed_weight=embed.weight)
    assert torch.equal(out.detach(), E[idx]), "hard snap not bit-equal to embedding rows"
    # STE gradient == identity
    g = torch.randn(N, H)
    out.backward(g)
    assert torch.equal(h.grad, g), "STE gradient is not identity"
    # hard snap, logits metric (the FSDP-safe path)
    h2 = torch.randn(N, H, requires_grad=True)
    out2, idx2 = snap_to_vocab(h2, "hard", 8, 1.0, "logits", lm_head_mod=head,
                               embed_mod=embed)
    assert torch.equal(out2.detach(), E[idx2]), "logits-metric snap not bit-equal"
    # topk soft mixture: differentiable, convex weights
    h3 = torch.randn(N, H, requires_grad=True)
    out3, idx3 = snap_to_vocab(h3, "topk", 4, 0.5, "logits", lm_head_mod=head,
                               embed_mod=embed)
    assert out3.shape == (N, H) and idx3.shape == (N, 4)
    out3.sum().backward()
    assert h3.grad is not None and torch.isfinite(h3.grad).all() and h3.grad.abs().sum() > 0
    # sharded-weight guard fires
    try:
        snap_to_vocab(h.detach(), "hard", 8, 1.0, "cosine", embed_mod=embed,
                      embed_weight=embed.weight[: V // 2])
        raise AssertionError("sharded-weight guard did NOT fire")
    except AssertionError as e:
        if "guard did NOT fire" in str(e):
            raise
    return {}


# ------------------------------------------------------------------- U3

@gate("U3_mask_dense")
def u3_dense():
    hf = tiny_dense()
    mech = MarinMechModel(hf).eval()
    n_l = mech.n_layers
    mech.install_hooks(hidden_layers=[1], kv_layers=[1], attn_layers=list(range(n_l)))
    batch = synth_batch()
    need = {"attn_layers": set(range(n_l)), "kv_layers": {1}, "hidden_layers": {1}}
    # (a) blocked probs exactly zero, every layer, answer rows over problem cols
    feats = FeatureConfig(mask_blocks="answer->problem", attn_target_coef=1.0,
                          attn_layers=",".join(map(str, range(n_l))))
    out = mech(batch, feats, need)
    rows = torch.arange(14, batch["input_ids"].shape[1])
    worst = 0.0
    for l in range(n_l):
        for b in range(2):
            probs = rows_probs_from_capture(out, out["capture"], l, b, rows, "marin")
            worst = max(worst, float(probs[:, :, 2:8].abs().max()))
    assert worst == 0.0, f"blocked attention mass leaked: {worst}"
    # (b) FULL bottleneck: answer bit-invariant to problem-content substitution
    feats2 = FeatureConfig(mask_blocks="answer->prompt;slots->prompt")
    o1 = mech(batch, feats2, {})
    ids2 = batch["input_ids"].clone()
    g = torch.Generator().manual_seed(99)
    ids2[:, 2:8] = torch.randint(0, 512, (2, 6), generator=g)
    b2 = {**batch, "input_ids": ids2}
    o2 = mech(b2, feats2, {})
    d_ans = (o1["h_final"][:, 14:] - o2["h_final"][:, 14:]).abs().max()
    assert float(d_ans) == 0.0, f"dense bottleneck leaked: max|dh|={float(d_ans)}"
    # sanity: WITHOUT blocks the substitution must change the answer states
    o3 = mech(batch, FeatureConfig(), {})
    o4 = mech(b2, FeatureConfig(), {})
    d_open = (o3["h_final"][:, 14:] - o4["h_final"][:, 14:]).abs().max()
    assert float(d_open) > 0, "instrument dead: open-mask substitution changed nothing"
    # (c) strict quantize bottleneck: per-pass masks — writer (pass A) READS the
    # problem; reader (pass B) slots are blocked from it
    feats3 = FeatureConfig(mask_blocks="answer->problem", quantize_slots="hard",
                           quantize_passb_blocks="slots->problem",
                           attn_target_coef=1.0,
                           attn_layers=",".join(map(str, range(n_l))))
    o5 = mech(batch, feats3, need)
    slot_rows = torch.arange(10, 14)
    outA = {**o5, "mask": mech._mask(batch["real"], feats3, batch["spans"], passb=False)}
    pA = rows_probs_from_capture(outA, o5["capture"], 1, 0, slot_rows, "marin", passA=True)
    pB = rows_probs_from_capture(o5, o5["capture"], 1, 0, slot_rows, "marin", passA=False)
    a_reads = float(pA[:, :, 2:8].sum())
    b_reads = float(pB[:, :, 2:8].abs().max())
    assert a_reads > 0, "writer pass cannot read the problem — strict bottleneck dead"
    assert b_reads == 0.0, f"reader-pass slots still read the problem: {b_reads}"
    return {"blocked_max_prob": worst, "bottleneck_max_dh": float(d_ans),
            "open_max_dh": float(d_open), "quantize_passA_problem_mass": a_reads,
            "quantize_passB_problem_mass": b_reads}


@gate("U3_mask_hybrid_gdn_leak")
def u3_hybrid():
    lm = tiny_hybrid()
    mech = Q35BMech(lm).eval()
    fa = mech.full_attn_layers
    mech.install_hooks(hidden_layers=[fa[0]], kv_layers=[fa[0]], attn_layers=fa)
    batch = synth_batch()
    need = {"attn_layers": set(fa), "kv_layers": {fa[0]}, "hidden_layers": {fa[0]}}
    feats = FeatureConfig(mask_blocks="answer->problem", attn_target_coef=1.0,
                          attn_layers=",".join(map(str, fa)))
    out = mech(batch, feats, need)
    rows = torch.arange(14, batch["input_ids"].shape[1])
    worst = 0.0
    for l in fa:
        for b in range(2):
            probs = rows_probs_from_capture(out, out["capture"], l, b, rows, "q35b")
            worst = max(worst, float(probs[:, :, 2:8].abs().max()))
    assert worst == 0.0, f"full-attn blocked mass leaked: {worst}"
    # GDN leak: full bottleneck + problem substitution -> answer states CHANGE
    feats2 = FeatureConfig(mask_blocks="answer->prompt;slots->prompt")
    o1 = mech(batch, feats2, {})
    ids2 = batch["input_ids"].clone()
    g = torch.Generator().manual_seed(99)
    ids2[:, 2:8] = torch.randint(0, 512, (2, 6), generator=g)
    o2 = mech({**batch, "input_ids": ids2}, feats2, {})
    d_ans = float((o1["h_final"][:, 14:] - o2["h_final"][:, 14:]).abs().max())
    assert d_ans > 0, ("tiny-hybrid showed NO GDN leak — instrument suspect "
                       "(30/40 GDN layers must carry blocked content)")
    # bf16 REGRESSION (the real-model U6 "invalid dtype for bias" bug, caught
    # 2026-07-06): masked + quantize-masked forwards must run when the model
    # computes in bf16 (mask dtype must track compute dtype).
    lm16 = lm.to(torch.bfloat16)
    mech16 = Q35BMech(lm16).eval()
    o16 = mech16(batch, feats2, {})
    assert o16["h_final"].dtype == torch.bfloat16
    o16q = mech16(batch, FeatureConfig(mask_blocks="answer->problem",
                                       quantize_slots="hard",
                                       quantize_passb_blocks="slots->problem"), {})
    assert torch.isfinite(o16q["h_final"].float()).all()
    return {"full_attn_blocked_max_prob": worst, "gdn_leak_max_dh": d_ans,
            "bf16_masked_forward_ok": True,
            "note": "leak is EXPECTED on hybrid: mask constrains only full-attn layers"}


# ------------------------------------------------------------------- U4

@gate("U4_parity")
def u4():
    res = {}
    # dense: features-off MechModel vs plain HF forward
    hf = tiny_dense()
    mech = MarinMechModel(hf).eval()
    batch = synth_batch()
    with torch.no_grad():
        out = mech(batch, FeatureConfig(), {})
        logits_mech = mech.lm_head(out["h_final"])
        logits_hf = hf(input_ids=batch["input_ids"]).logits
    d = float((logits_mech - logits_hf).abs().max())
    top1 = float((logits_mech.argmax(-1) == logits_hf.argmax(-1)).float().mean())
    assert top1 == 1.0 and d < 2e-4, (d, top1)
    res["dense_max_dlogit"] = d
    res["dense_top1"] = top1
    # hybrid: hooks installed + features off => BIT-identical to plain HF
    lm = tiny_hybrid()
    mech2 = Q35BMech(lm).eval()
    fa = mech2.full_attn_layers
    mech2.install_hooks(hidden_layers=[fa[0]], kv_layers=[fa[0]], attn_layers=fa)
    with torch.no_grad():
        h_plain = lm.model(input_ids=batch["input_ids"],
                           attention_mask=batch["real"].long(),
                           position_ids=batch["pos"]).last_hidden_state
        o_off = mech2(batch, FeatureConfig(), {})       # spec inactive
        o_cap = mech2(batch, FeatureConfig(),           # captures ACTIVE, features off
                      {"attn_layers": set(fa), "kv_layers": {fa[0]},
                       "hidden_layers": {fa[0]}})
    assert torch.equal(o_off["h_final"], h_plain), "hooks-off forward != plain HF (bit)"
    assert torch.equal(o_cap["h_final"], h_plain), "capture-active forward != plain HF (bit)"
    res["hybrid_bit_identical"] = True
    # 4D explicit causal mask == default path
    L = batch["input_ids"].shape[1]
    m4 = Q35BMech.build_mask4d(batch["real"], [], batch["spans"], torch.float32)
    with torch.no_grad():
        h_4d = lm.model(input_ids=batch["input_ids"], attention_mask=m4,
                        position_ids=batch["pos"]).last_hidden_state
    d4 = float((h_4d - h_plain).abs().max())
    assert d4 == 0.0, f"4D-causal-mask path differs from default: {d4}"
    res["mask4d_delta"] = d4
    return res


@gate("U4b_attn_recompute_vs_eager")
def u4b():
    """rows_probs_from_capture must reproduce HF eager attention weights."""
    lm = tiny_hybrid()
    lm.config._attn_implementation = "eager"
    mech = Q35BMech(lm).eval()
    fa = mech.full_attn_layers
    mech.install_hooks(kv_layers=[], attn_layers=fa)
    eager_probs = {}
    hooks = []
    for l in fa:
        def mk(l):
            def hook(_m, _i, out):
                eager_probs[l] = out[1]      # (attn_output, attn_weights)
            return hook
        hooks.append(lm.model.layers[l].self_attn.register_forward_hook(mk(l)))
    batch = synth_batch()
    with torch.no_grad():
        out = mech(batch, FeatureConfig(), {"attn_layers": set(fa)})
    rows = torch.arange(0, batch["input_ids"].shape[1])
    worst = 0.0
    for l in fa:
        for b in range(2):
            mine = rows_probs_from_capture(out, out["capture"], l, b, rows, "q35b")
            ref = eager_probs[l][b].transpose(0, 1)      # [T, H, T]
            worst = max(worst, float((mine - ref.float()).abs().max()))
    for h in hooks:
        h.remove()
    assert worst < 1e-5, f"attn recompute mismatch vs eager: {worst}"
    return {"max_dprob": worst}


# ------------------------------------------------------------------- U5

@gate("U5_loss_plumbing")
def u5():
    res = {}
    for kind in ("dense", "hybrid"):
        if kind == "dense":
            mech = MarinMechModel(tiny_dense())
            fa = list(range(mech.n_layers))
            kv_l, attn_l, hid_l = 1, 2, 1
        else:
            mech = Q35BMech(tiny_hybrid())
            fa = mech.full_attn_layers
            kv_l, attn_l, hid_l = fa[0], fa[1], 2
        mech.train()
        mech.install_hooks(hidden_layers=[hid_l], kv_layers=[kv_l], attn_layers=[attn_l])
        feats = FeatureConfig(slot_value_coef=1.0, slot_hidden_coef=1.0,
                              slot_hidden_layer=hid_l, set_match="hungarian",
                              kv_match_coef=1.0, kv_layers=str(kv_l),
                              attn_target_coef=1.0, attn_layers=str(attn_l),
                              quantize_slots="hard")
        need = need_from_feats(feats)
        batch = synth_batch()
        batch["slot_labels"][:, 10:14] = torch.randint(0, 512, (2, 4))
        out = mech(batch, feats, need)
        H = out["h_final"].shape[-1]
        # regression (armA P3 crash): teachers CARRY GRAD here on purpose —
        # assemble_losses must detach them (a grad-carrying target would route
        # backward into its producer; on FSDP that is a flat-shard crash)
        batch["teacher_hiddens"] = [torch.randn(3, H, requires_grad=True)
                                    for _ in range(2)]
        hd = (mech.layers[0].inner.self_attn.head_dim if kind == "dense"
              else mech.lm.model.layers[fa[0]].self_attn.head_dim)
        sk, sv = kv_prerope_from_capture(out["capture"], kv_l, 0,
                                         torch.arange(10, 14), hd, passA=True)
        batch["teacher_kv"] = [{kv_l: (sk.detach() + 0.1, sv.detach() + 0.1)}, None]
        batch["attn_targets"] = [None, None]
        losses = assemble_losses(mech, out, batch, feats)
        for k in ("ce", "slot_value", "slot_hidden", "kv", "attn", "total"):
            assert k in losses and torch.isfinite(losses[k]), (kind, k, losses.get(k))
        losses["total"].backward()
        if kind == "dense":
            kproj = mech.layers[kv_l].inner.self_attn.k_proj.weight
            qproj = mech.layers[attn_l].inner.self_attn.q_proj.weight
        else:
            kproj = mech.lm.model.layers[kv_l].self_attn.k_proj.weight
            qproj = mech.lm.model.layers[attn_l].self_attn.q_proj.weight
        assert kproj.grad is not None and kproj.grad.abs().sum() > 0, f"{kind}: no kv grad"
        assert qproj.grad is not None and qproj.grad.abs().sum() > 0, f"{kind}: no attn grad"
        emb = mech.embed_tokens.weight if kind == "dense" else mech.lm.model.embed_tokens.weight
        assert emb.grad is not None, f"{kind}: no embed grad (quantize STE path dead?)"
        for th in batch["teacher_hiddens"]:
            assert th.grad is None, \
                f"{kind}: gradient leaked into a teacher target (armA P3 crash class)"
        res[f"{kind}_losses"] = {k: round(float(v), 5) for k, v in losses.items()}
    return res


# ------------------------------------------------------------------- U6 (GPU)

@gate("U6_real_marin_parity")
def u6_marin(device="cuda:0"):
    from transformers import AutoModelForCausalLM
    from recur_loop import build_batch, load_tokenizer
    from mechanism_forward import marin_annotate_spans
    RL_CKPT = ("/shared/xorl-marin-rl-6279/checkpoints/"
               "delphi-1e22-p33m67-rl-rlvr7500_w1-think-140-10B")
    tok = load_tokenizer(RL_CKPT)
    hf = AutoModelForCausalLM.from_pretrained(RL_CKPT, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to(device).eval()
    mech = MarinMechModel(hf).eval()
    rows = [json.loads(l) for l in open(
        "/shared/apanda/filler_grpo/recur_stage0/data/val.jsonl")][:8]
    batch = build_batch(tok, rows, "c0", device, with_answer=True)
    marin_annotate_spans(tok, batch, rows, "c0")
    with torch.no_grad():
        out = mech(batch, FeatureConfig(), {})
        logits_mech = mech.lm_head(out["h_final"]).float()
        logits_hf = hf(input_ids=batch["input_ids"],
                       attention_mask=batch["real"].long(),
                       position_ids=batch["pos"]).logits.float()
    sel = batch["labels"] != -100
    agree = float((logits_mech[sel].argmax(-1) == logits_hf[sel].argmax(-1)).float().mean())
    dmax = float((logits_mech[sel] - logits_hf[sel]).abs().max())
    assert agree == 1.0, f"top-1 agreement {agree} < 1.0 (max|dlogit|={dmax})"
    return {"top1_answer_cols": agree, "max_dlogit": dmax,
            "note": "bf16 reordering diffs allowed; top-1 must be 100% (recur S0.2 bar)"}


@gate("U6_real_q35b")
def u6_q35b(device="cuda:0", n_items=8):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q35_BASE_SNAPSHOT)
    lm, _ = load_q35b_text(dtype=torch.bfloat16, device=device)   # strict audit inside
    mech = Q35BMech(lm).eval()
    fa = mech.full_attn_layers
    res = {"full_attn_layers": fa}
    # coherence sample (transformers-5.3-loads-this-card guard)
    msgs = [{"role": "user", "content": "Say hello and name three colors."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=device)
    with torch.no_grad():
        gen = lm.generate(ids, max_new_tokens=32, do_sample=False)
    sample = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    res["coherence_sample"] = sample
    assert len(sample.strip()) > 0
    # hook bit-identity on a real batch
    items = [json.loads(l) for l in open(
        "/shared/apanda/filler_grpo/comp_bench/data/nops05-07_capped_val500.jsonl")][:n_items]
    mech.install_hooks(hidden_layers=[fa[1]], kv_layers=[fa[1]], attn_layers=[fa[1]])
    batch = build_batch_q35(tok, items[:2], device, k_base_fn=lambda it: it["n_ops"])
    with torch.no_grad():
        h_plain = lm.model(input_ids=batch["input_ids"],
                           attention_mask=batch["real"].long(),
                           position_ids=batch["pos"]).last_hidden_state
        o_cap = mech(batch, FeatureConfig(), {"attn_layers": {fa[1]},
                                              "kv_layers": {fa[1]},
                                              "hidden_layers": {fa[1]}})
    assert torch.equal(o_cap["h_final"], h_plain), "hooked forward != plain (bit)"
    res["hook_bit_identity"] = True
    # mask-block prob inspection at a real full-attn layer
    feats = FeatureConfig(mask_blocks="answer->problem", attn_target_coef=1.0,
                          attn_layers=str(fa[1]))
    out = mech(batch, feats, {"attn_layers": {fa[1]}})
    sp = batch["spans"][0]
    rows = torch.arange(sp["answer"][0], sp["answer"][1], device=device)
    probs = rows_probs_from_capture(out, out["capture"], fa[1], 0, rows, "q35b")
    leak = float(probs[:, :, sp["problem"][0]:sp["problem"][1]].abs().max())
    assert leak == 0.0, f"blocked full-attn mass: {leak}"
    res["blocked_prob_max"] = leak
    # GDN leak quantification (the hybrid bottleneck caveat, measured)
    feats_b = FeatureConfig(mask_blocks="answer->prompt;slots->prompt")
    res["gdn_leak"] = {k: v for k, v in gdn_leak_probe(
        mech, tok, items, device, feats_b).items() if k != "per_item"}
    return res


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="u1,u2,u3,u4,u5")
    ap.add_argument("--real", choices=["marin", "q35b"], default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="mechanism_gates_report.json")
    args = ap.parse_args()
    which = {w.strip() for w in args.which.split(",")}
    if "u1" in which:
        u1()
    if "u2" in which:
        u2()
    if "u3" in which:
        u3_dense()
        u3_hybrid()
    if "u4" in which:
        u4()
        u4b()
    if "u5" in which:
        u5()
    if "u6" in which:
        assert args.gpu and torch.cuda.is_available(), "U6 needs --gpu on a GPU host"
        if args.real in (None, "marin"):
            u6_marin()
        if args.real in (None, "q35b"):
            u6_q35b()
    ok = all(v.get("pass") for v in REPORT.values())
    REPORT["_all_pass"] = ok
    with open(args.out, "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    print(f"[GATES] {'ALL PASS' if ok else 'FAILURES PRESENT'} -> {args.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
