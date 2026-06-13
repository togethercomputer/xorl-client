"""Measure the true (linearized) cosine between the ES update direction and the
exact SFT gradient for the ZORL 4x4-multiplication setup, and derive the task's
effective dimension.

Measurement only: no k8s, no SGLang, no sglang-repo changes. Loads the HF
Qwen3.6-35B-A3B weights on locally-idle GPUs, computes the exact gradient of
the production SFT loss (mean negative logprob of the product tokens, averaged
over the frozen 256-example train batch, seed=9234) w.r.t. a depth-spanning
subset of the ES-targeted weight matrices, then evaluates the two ES geometries
analytically through the linearization:

    score_i      = <dW_i, g>                      (directional derivative)
    update U_W   = sum_i z(score_i) * dW_i        (z-scored ES update)
    cos          = <U_W, g> / (||U_W|| ||g||)
    D_eff        = N / cos^2

Geometry A (b_only / LOZO):   dW = eps_B . A   (A = parent adapter's Kaiming
                              LoRA-A, frozen; eps_B unit gaussian over B shapes)
Geometry B (fresh_ab/EGGROLL):dW = eps_B . eps_A (both unit gaussian, per-pair
                              fresh A; alpha/r = 1)

Shared-outer expert layout (verified against the adapter tensors):
    w1/gate: A shared [1,r,K=2048],  B per-expert [E=256,I=512,r]  dW_e = B_e A
    w3/up:   same as w1
    w2/down: A per-expert [E,r,I=512], B shared [1,K=2048,r]       dW_e = B A_e
    qkv:     A [r,2048], B [9216,r]   (q gated 8192 + k 512 + v 512)
    o:       A [r,4096], B [2048,r]
HF stores experts.gate_up_proj [E, 2I, K] half-concat (gate first; the forward
chunks dim -2 in two) and experts.down_proj [E, K, I]; attention is unfused
q/k/v/o. Row ordering inside the fused qkv B is irrelevant for every quantity
computed here (iid gaussian rows + a shared A make all row permutations
statistically identical), so g_qkv = cat(g_q, g_k, g_v).

Run (only on fully idle GPUs; the launcher must set CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=2,3 python measure_es_cosine.py --outdir /path/out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MODEL_HUB_DIR = Path("/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots")
ADAPTER_DIR = Path("/shared/zorl/init-adapters/qwen3_6-35b-a3b-r16-eggroll-hybridattn")

LAYERS = [3, 11, 19, 27, 35, 39]  # depth-spanning subset; all are full-attention layers
ALL_LAYERS = list(range(40))
ATTN_LAYERS = list(range(3, 40, 4))  # 3,7,...,39 (10 layers)
NUM_EXPERTS = 256
HIDDEN = 2048
INTER = 512
R = 16
QKV_OUT = 9216  # 8192 (gated q) + 512 (k) + 512 (v)
O_IN = 4096

SIGMA_B_ONLY = 0.012
SIGMA_FRESH_AB_EFF = 1.5e-4 * math.sqrt(R)  # = 6e-4
PROD_PAIR_DELTA_STD = {"b_only": 0.0908, "fresh_ab": 0.0729}
D_B_FULL = 40 * (2 * NUM_EXPERTS * INTER * R + HIDDEN * R) + len(ATTN_LAYERS) * (QKV_OUT * R + HIDDEN * R)
DENSE_FULL = 40 * (2 * NUM_EXPERTS * INTER * HIDDEN + NUM_EXPERTS * HIDDEN * INTER) + len(ATTN_LAYERS) * (
    QKV_OUT * HIDDEN + HIDDEN * O_IN
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snapshot_dir() -> Path:
    snaps = sorted(MODEL_HUB_DIR.iterdir())
    assert snaps, f"no snapshot under {MODEL_HUB_DIR}"
    return snaps[0]


def stable_seed(*parts) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**63 - 1)


# ---------------------------------------------------------------------------
# Data: replicate the production SFT batch exactly via tasks/mult.py
# ---------------------------------------------------------------------------


def build_batches(tok):
    sys.path.insert(0, str(HERE))
    from tasks import mult

    # Canonical frozen train batch: build_examples(train_size=256, eval_size=128, seed=9234).
    train256, _eval = mult.build_examples(tok, train_size=256, eval_size=128, seed=9234)
    # Second disjoint 256-batch for gradient coherence: same seeded shuffle, the
    # 256 problems AFTER the canonical train+eval region (order[384:640]).
    train640, _ = mult.build_examples(tok, train_size=640, eval_size=128, seed=9234)
    assert [e.metadata["prompt_text"] for e in train640[:256]] == [e.metadata["prompt_text"] for e in train256]
    batch2 = train640[384:640]
    k1 = {e.metadata["prompt_text"] for e in train256}
    k2 = {e.metadata["prompt_text"] for e in batch2}
    assert not (k1 & k2), "coherence batch must be disjoint from the canonical train batch"

    def to_tf(examples):
        rows = []
        for e in examples:
            tf = mult.build_teacher_forced_example(tok, e)
            rows.append((list(tf.prompt_ids), int(tf.metadata["teacher_target_token_count"])))
        return rows

    return to_tf(train256), to_tf(batch2)


# ---------------------------------------------------------------------------
# Model + exact gradient of the production SFT loss
# ---------------------------------------------------------------------------

SUBSET_SUFFIXES = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)


def load_model(snap: Path):
    import transformers
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(snap)
    cls = getattr(transformers, cfg.architectures[0])
    n = torch.cuda.device_count()
    assert n >= 2, f"need >=2 idle GPUs, got {n} visible"
    max_memory = {i: "75GiB" for i in range(n)}
    log(f"loading {cfg.architectures[0]} on {n} GPUs ...")
    model = cls.from_pretrained(
        snap,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="sdpa",
    )
    model.eval()
    bad = {k: v for k, v in model.hf_device_map.items() if v in ("cpu", "disk")}
    assert not bad, f"model spilled off GPU: {bad}"
    return model


def find_subset_params(model) -> dict[tuple[int, str], torch.nn.Parameter]:
    """(layer, suffix) -> Parameter for the targeted subset layers (main trunk only)."""
    out = {}
    for name, p in model.named_parameters():
        if name.startswith("mtp.") or ".mtp." in name or "visual" in name:
            continue
        for L in LAYERS:
            tag = f".layers.{L}."
            if tag in name:
                for suf in SUBSET_SUFFIXES:
                    if name.endswith(suf):
                        out[(L, suf)] = p
    want = len(LAYERS) * len(SUBSET_SUFFIXES)
    assert len(out) == want, f"found {len(out)} subset params, expected {want}: {sorted(out)}"
    return out


def compute_gradient(model, tf_rows, subset, mb_size=32, tag=""):
    """Exact grad of L = mean_e mean_t -logprob(target_t) over the 256 examples.

    Returns (loss, {key: fp32 grad tensor on the param's device}).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in subset.values():
        p.requires_grad_(True)
    accum = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in subset.items()}

    order = sorted(range(len(tf_rows)), key=lambda i: len(tf_rows[i][0]))
    pad_id = 0
    n_total = len(tf_rows)
    total_loss = 0.0
    emb_device = model.get_input_embeddings().weight.device
    t0 = time.time()
    for s in range(0, n_total, mb_size):
        idxs = order[s : s + mb_size]
        rows = [tf_rows[i] for i in idxs]
        maxlen = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), maxlen), dtype=torch.long)
        for b, (ids, _n) in enumerate(rows):
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[b, : len(ids)] = 1
        input_ids = input_ids.to(emb_device)
        attn = attn.to(emb_device)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [B, T, V] on the last device
        loss_mb = None
        for b, (ids, n_tgt) in enumerate(rows):
            L = len(ids)
            pos = torch.arange(L - n_tgt, L, device=logits.device)
            lg = logits[b].index_select(0, pos - 1).float()
            tgt = input_ids[b].index_select(0, pos).to(logits.device)
            lp = torch.log_softmax(lg, dim=-1).gather(1, tgt[:, None]).squeeze(1)
            ex_loss = -(lp.mean())
            loss_mb = ex_loss if loss_mb is None else loss_mb + ex_loss
        loss_mb = loss_mb / n_total
        loss_mb.backward()
        total_loss += float(loss_mb.detach().item())
        for k, p in subset.items():
            if p.grad is not None:
                accum[k] += p.grad.float()
                p.grad = None
        del out, logits
        if (s // mb_size) % 2 == 0:
            log(f"  grad{tag} mb {s // mb_size + 1}/{(n_total + mb_size - 1) // mb_size} "
                f"loss so far {total_loss * n_total / (s + len(rows)):.4f} ({time.time() - t0:.0f}s)")
    for p in subset.values():
        p.requires_grad_(False)
    return total_loss, accum


# ---------------------------------------------------------------------------
# Module table: map HF grads + adapter A matrices into the ES coordinates
# ---------------------------------------------------------------------------


def load_adapter_A():
    from safetensors import safe_open

    A = {}
    with safe_open(str(ADAPTER_DIR / "adapter_model.safetensors"), "pt") as f:
        for L in LAYERS:
            pre = f"base_model.model.model.layers.{L}"
            A[(L, "w1")] = f.get_tensor(f"{pre}.mlp.experts.w1.lora_A.weight").float()  # [1,r,2048]
            A[(L, "w3")] = f.get_tensor(f"{pre}.mlp.experts.w3.lora_A.weight").float()  # [1,r,2048]
            A[(L, "w2")] = f.get_tensor(f"{pre}.mlp.experts.w2.lora_A.weight").float()  # [256,r,512]
            A[(L, "qkv")] = f.get_tensor(f"{pre}.self_attn.qkv_proj.lora_A.weight").float()  # [r,2048]
            A[(L, "o")] = f.get_tensor(f"{pre}.self_attn.o_proj.lora_A.weight").float()  # [r,4096]
            # sanity: B shapes match the geometry we assume
            assert tuple(f.get_slice(f"{pre}.mlp.experts.w1.lora_B.weight").get_shape()) == (NUM_EXPERTS, INTER, R)
            assert tuple(f.get_slice(f"{pre}.mlp.experts.w2.lora_B.weight").get_shape()) == (1, HIDDEN, R)
            assert tuple(f.get_slice(f"{pre}.self_attn.qkv_proj.lora_B.weight").get_shape()) == (QKV_OUT, R)
            assert tuple(f.get_slice(f"{pre}.self_attn.o_proj.lora_B.weight").get_shape()) == (HIDDEN, R)
    return A


def build_modules(grads: dict, adapterA: dict) -> list[dict]:
    """Each module dict:
      name, layer, kind, mtype:
        'sharedA': dW_e = eps_B[e] @ A      g [E,O,K], A [r,K], eps_B [E,O,r]
        'perA'   : dW_e = eps_B @ A[e]      g [E,O,I], A [E,r,I], eps_B [O,r]  (B shared)
      g fp32 on GPU; A fp32 moved to g.device.
    """
    mods = []
    for L in LAYERS:
        g_gu = grads[(L, "mlp.experts.gate_up_proj")]  # [256, 1024, 2048], gate = first half
        g_dn = grads[(L, "mlp.experts.down_proj")]  # [256, 2048, 512]
        g_q = grads[(L, "self_attn.q_proj.weight")]  # [8192, 2048]
        g_k = grads[(L, "self_attn.k_proj.weight")]
        g_v = grads[(L, "self_attn.v_proj.weight")]
        g_o = grads[(L, "self_attn.o_proj.weight")]  # [2048, 4096]
        dev = g_gu.device
        mods.append(dict(name=f"L{L}.w1", layer=L, kind="expert", mtype="sharedA",
                         g=g_gu[:, :INTER, :], A=adapterA[(L, "w1")][0].to(dev)))
        mods.append(dict(name=f"L{L}.w3", layer=L, kind="expert", mtype="sharedA",
                         g=g_gu[:, INTER:, :], A=adapterA[(L, "w3")][0].to(dev)))
        mods.append(dict(name=f"L{L}.w2", layer=L, kind="expert", mtype="perA",
                         g=g_dn, A=adapterA[(L, "w2")].to(dev)))
        g_qkv = torch.cat([g_q.to(dev), g_k.to(dev), g_v.to(dev)], dim=0).unsqueeze(0)  # [1,9216,2048]
        mods.append(dict(name=f"L{L}.qkv", layer=L, kind="attn", mtype="sharedA",
                         g=g_qkv, A=adapterA[(L, "qkv")].to(dev)))
        mods.append(dict(name=f"L{L}.o", layer=L, kind="attn", mtype="sharedA",
                         g=g_o.unsqueeze(0).to(dev), A=adapterA[(L, "o")].to(dev)))
    for m in mods:
        g, A = m["g"], m["A"]
        if m["mtype"] == "sharedA":
            E, O, K = g.shape
            assert A.shape == (R, K), (m["name"], A.shape, g.shape)
            m["b_shape"] = (E, O, R)
            m["a_shape"] = (R, K)
        else:
            E, O, I = g.shape
            assert A.shape == (E, R, I), (m["name"], A.shape, g.shape)
            m["b_shape"] = (O, R)
            m["a_shape"] = (E, R, I)
        m["d_b"] = int(np.prod(m["b_shape"]))
        m["d_dense"] = g.numel()
        m["g_sq"] = float((g.double() * g.double()).sum().item())
    return mods


# ---------------------------------------------------------------------------
# ES linearized analysis
# ---------------------------------------------------------------------------


def f64dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum(dtype=torch.float64).item())


def randn(shape, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)


def precompute_GB(mods):
    """Geometry A: G_B per module (projection of g into B-space through A)."""
    for m in mods:
        g, A = m["g"], m["A"]
        if m["mtype"] == "sharedA":
            m["GB"] = torch.einsum("eok,rk->eor", g, A)  # [E,O,r]
        else:
            m["GB"] = torch.einsum("eoi,eri->or", g, A)  # [O,r]
        m["var_A"] = float((m["GB"].double() ** 2).sum().item())  # Var(<eps_B, GB>) = ||GB||^2
        m["var_B"] = R * m["g_sq"]  # E_eps_A Var = r ||g||^2 (holds for both layouts)


def es_draw_geomA(mods, N, draw, seed_base):
    """One population draw for geometry A. Returns dict of results."""
    s = torch.zeros(N, dtype=torch.float64)
    for m in mods:
        GB = m["GB"]
        dev = GB.device
        for i in range(N):
            eps = randn(m["b_shape"], stable_seed(seed_base, "A", draw, i, m["name"]), dev)
            s[i] += f64dot(eps, GB)
    z = (s - s.mean()) / s.std(correction=0)
    num = 0.0  # <U_W, g> == <U_B, G_B>
    uu_w = 0.0  # ||U_W||^2
    uu_b = 0.0  # ||U_B||^2 (B-space norm, for the B-space cosine)
    gg_b = 0.0  # ||G_B||^2
    for m in mods:
        GB, A = m["GB"], m["A"]
        dev = GB.device
        U = torch.zeros(m["b_shape"], dtype=torch.float32, device=dev)
        zs = z.to(dev, torch.float32)
        for i in range(N):
            eps = randn(m["b_shape"], stable_seed(seed_base, "A", draw, i, m["name"]), dev)
            U += zs[i] * eps
        num += f64dot(U, GB)
        uu_b += float((U.double() ** 2).sum().item())
        gg_b += m["var_A"]
        if m["mtype"] == "sharedA":
            M = torch.einsum("eor,eos->rs", U.double(), U.double())  # sum_e U_e^T U_e
            AAt = (A.double() @ A.double().T)
            uu_w += float((M * AAt).sum().item())
        else:
            M = (U.double().T @ U.double())  # [r,r]
            AAt = torch.einsum("eri,esi->rs", A.double(), A.double())  # sum_e A_e A_e^T
            uu_w += float((M * AAt).sum().item())
    gg_w = sum(m["g_sq"] for m in mods)
    cos_w = num / math.sqrt(uu_w * gg_w)
    cos_b = num / math.sqrt(uu_b * gg_b)
    return dict(s=s.numpy(), cos_w=cos_w, cos_b=cos_b, std_s=float(s.std(correction=0)))


def es_draw_geomB(mods, N, draw, seed_base):
    """One population draw for geometry B (fresh eps_A x eps_B outer products)."""
    s = torch.zeros(N, dtype=torch.float64)
    for m in mods:
        g = m["g"]
        dev = g.device
        for i in range(N):
            sa = stable_seed(seed_base, "Ba", draw, i, m["name"])
            sb = stable_seed(seed_base, "Bb", draw, i, m["name"])
            epsA = randn(m["a_shape"], sa, dev)
            epsB = randn(m["b_shape"], sb, dev)
            if m["mtype"] == "sharedA":
                proj = torch.einsum("eok,rk->eor", g, epsA)
                s[i] += f64dot(epsB, proj)
            else:
                proj = torch.einsum("eoi,eri->or", g, epsA)
                s[i] += f64dot(epsB, proj)
    z = (s - s.mean()) / s.std(correction=0)
    num = 0.0
    uu_w = 0.0
    for m in mods:
        g = m["g"]
        dev = g.device
        zs = z.to(dev, torch.float32)
        U = torch.zeros(g.shape, dtype=torch.float32, device=dev)
        for i in range(N):
            sa = stable_seed(seed_base, "Ba", draw, i, m["name"])
            sb = stable_seed(seed_base, "Bb", draw, i, m["name"])
            epsA = randn(m["a_shape"], sa, dev)
            epsB = randn(m["b_shape"], sb, dev)
            if m["mtype"] == "sharedA":
                U += zs[i] * torch.einsum("eor,rk->eok", epsB, epsA)
            else:
                U += zs[i] * torch.einsum("or,eri->eoi", epsB, epsA)
        num += f64dot(U, g)
        uu_w += float((U.double() ** 2).sum().item())
        del U
    gg_w = sum(m["g_sq"] for m in mods)
    cos_w = num / math.sqrt(uu_w * gg_w)
    return dict(s=s.numpy(), cos_w=cos_w, std_s=float(s.std(correction=0)))


# ---------------------------------------------------------------------------
# Depth extrapolation of Var(s) to the full 40-layer module set
# ---------------------------------------------------------------------------


def extrapolate_var(mods, key):
    """Sum per-layer Var contributions, linearly interpolated across depth.
    Experts: interpolated over layers 0..39. Attention: over the 10 attn layers."""
    expert_per_layer = {}
    attn_per_layer = {}
    for m in mods:
        d = expert_per_layer if m["kind"] == "expert" else attn_per_layer
        d[m["layer"]] = d.get(m["layer"], 0.0) + m[key]
    xs = sorted(expert_per_layer)
    ev = np.interp(ALL_LAYERS, xs, [expert_per_layer[x] for x in xs]).sum()
    xa = sorted(attn_per_layer)
    av = np.interp(ATTN_LAYERS, xa, [attn_per_layer[x] for x in xa]).sum()
    subtotal = sum(expert_per_layer.values()) + sum(attn_per_layer.values())
    return float(ev + av), float(subtotal), float(ev), float(av)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "es_cosine_out"))
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--mb-size", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=20260612)
    ap.add_argument("--skip-coherence", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snap = snapshot_dir()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    log("building batches from tasks/mult.py (train_size=256, seed=9234) ...")
    batch1, batch2 = build_batches(tok)
    lens = [len(ids) for ids, _ in batch1]
    tgts = [n for _, n in batch1]
    log(f"batch1: {len(batch1)} examples, seq len min/med/max = {min(lens)}/{sorted(lens)[128]}/{max(lens)}, "
        f"target tokens min/med/max = {min(tgts)}/{sorted(tgts)[128]}/{max(tgts)}")

    model = load_model(snap)
    subset = find_subset_params(model)
    log(f"subset params: {len(subset)} tensors, "
        f"{sum(p.numel() for p in subset.values()) / 1e9:.3f}B dense params")

    log("=== computing exact SFT gradient on the canonical 256 batch (cold parent) ===")
    t0 = time.time()
    loss1, grads1 = compute_gradient(model, batch1, subset, mb_size=args.mb_size, tag="A")
    log(f"COLD LOSS (batch1) = {loss1:.5f}  (mean logprob = {-loss1:.5f}); took {time.time() - t0:.0f}s")

    adapterA = load_adapter_A()
    mods = build_modules(grads1, adapterA)
    precompute_GB(mods)
    gg_w = sum(m["g_sq"] for m in mods)
    d_b_subset = sum(m["d_b"] for m in mods)
    d_dense_subset = sum(m["d_dense"] for m in mods)
    log(f"||g||^2 (subset) = {gg_w:.6e}; D_B_subset = {d_b_subset:.3e}; dense_subset = {d_dense_subset:.3e}")
    per_mod = {m["name"]: dict(g_sq=m["g_sq"], var_A=m["var_A"], var_B=m["var_B"],
                               d_b=m["d_b"], d_dense=m["d_dense"]) for m in mods}

    results = dict(
        loss_cold_batch1=loss1,
        d_b_subset=d_b_subset,
        d_dense_subset=d_dense_subset,
        d_b_full=D_B_FULL,
        dense_full=DENSE_FULL,
        gg_subset=gg_w,
        per_module=per_mod,
        pairs=args.pairs,
        draws=args.draws,
        layers=LAYERS,
    )

    # Analytic Var(s) + extrapolation to the full module set.
    for geom, key, sigma in (("b_only", "var_A", SIGMA_B_ONLY), ("fresh_ab", "var_B", SIGMA_FRESH_AB_EFF)):
        var_full, var_sub, var_exp, var_attn = extrapolate_var(mods, key)
        pred_sub = 2 * sigma * math.sqrt(var_sub)
        pred_full = 2 * sigma * math.sqrt(var_full)
        results[f"{geom}_var_subset"] = var_sub
        results[f"{geom}_var_full_extrap"] = var_full
        results[f"{geom}_pred_pair_delta_std_subset"] = pred_sub
        results[f"{geom}_pred_pair_delta_std_full"] = pred_full
        log(f"[{geom}] analytic std(s): subset {math.sqrt(var_sub):.4f}, full(extrap) {math.sqrt(var_full):.4f} "
            f"(experts {var_exp:.3e} + attn {var_attn:.3e})")
        log(f"[{geom}] predicted pair_delta_std: subset {pred_sub:.4f}, FULL {pred_full:.4f} "
            f"(production: {PROD_PAIR_DELTA_STD[geom]})")

    # ES draws.
    for geom, fn in (("b_only", es_draw_geomA), ("fresh_ab", es_draw_geomB)):
        coss, cosbs, stds = [], [], []
        for d in range(args.draws):
            t0 = time.time()
            r = fn(mods, args.pairs, d, args.seed_base)
            coss.append(r["cos_w"])
            stds.append(r["std_s"])
            if "cos_b" in r:
                cosbs.append(r["cos_b"])
            np.save(outdir / f"s_{geom}_draw{d}.npy", r["s"])
            log(f"[{geom}] draw {d}: cos_W = {r['cos_w']:.6f}" +
                (f", cos_B = {r['cos_b']:.6f}" if "cos_b" in r else "") +
                f", std(s) = {r['std_s']:.4f}  ({time.time() - t0:.0f}s)")
        cm, cs = float(np.mean(coss)), float(np.std(coss))
        results[f"{geom}_cos_draws"] = coss
        results[f"{geom}_cos_mean"] = cm
        results[f"{geom}_cos_std"] = cs
        results[f"{geom}_std_s_empirical"] = float(np.mean(stds))
        d_eff = args.pairs / cm**2
        results[f"{geom}_d_eff_subset"] = d_eff
        if cosbs:
            results[f"{geom}_cos_b_mean"] = float(np.mean(cosbs))
        log(f"[{geom}] cos = {cm:.6f} +/- {cs:.6f}  -> D_eff(subset) = {d_eff:.4e} "
            f"(N={args.pairs}; sqrt(N/D_B_subset) = {math.sqrt(args.pairs / d_b_subset):.6f})")

    # Coherence between gradients on two disjoint 256 batches.
    if not args.skip_coherence:
        log("=== moving grad(batch1) to CPU; computing grad on disjoint batch2 ===")
        g1_cpu = {k: v.cpu() for k, v in grads1.items()}
        for m in mods:
            m["g"] = None
            m["GB"] = None
        del grads1, mods
        torch.cuda.empty_cache()
        loss2, grads2 = compute_gradient(model, batch2, subset, mb_size=args.mb_size, tag="B")
        log(f"COLD LOSS (batch2) = {loss2:.5f}")
        num = uu = vv = 0.0
        for k in g1_cpu:
            a = g1_cpu[k].to(grads2[k].device)
            b = grads2[k]
            num += f64dot(a, b)
            uu += float((a.double() ** 2).sum().item())
            vv += float((b.double() ** 2).sum().item())
        coh = num / math.sqrt(uu * vv)
        results["loss_cold_batch2"] = loss2
        results["grad_coherence_256v256"] = coh
        log(f"BATCH-GRADIENT COHERENCE cos(g_256A, g_256B) = {coh:.5f}")

    # Predicted population sizes (using subset D_eff, and full-space scaling).
    for geom in ("b_only", "fresh_ab"):
        d_eff = results[f"{geom}_d_eff_subset"]
        eta = (d_b_subset if geom == "b_only" else d_b_subset) / d_eff  # search-dim efficiency
        d_eff_full = D_B_FULL / eta
        results[f"{geom}_d_eff_full_scaled"] = d_eff_full
        results[f"{geom}_N_for_cos_0.05"] = 0.05**2 * d_eff_full
        results[f"{geom}_N_for_cos_0.10"] = 0.10**2 * d_eff_full
        log(f"[{geom}] D_eff scaled to full module set = {d_eff_full:.4e}; "
            f"N for cos 0.05 = {0.05**2 * d_eff_full:.3e}, cos 0.10 = {0.10**2 * d_eff_full:.3e}")

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"wrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
