"""R1 gate (CPU-only): does routing G = Σ c_i ΔW_i through xorl Muon reproduce the
live sglang ES fold (_zorl_muon_update) update direction/magnitude?

Three independent checks:
  A. seed->noise stream parity  (iter_zorl_b_noises vs sglang _zorl_normalized_b_noises primitive)
  B. G construction parity      (build_zorl_update_from_rewards vs explicit Σ c_i (1/N) noise_i)
  C. NS-fold parity             (xorl Muon optim_step vs faithful sglang _zorl_muon_update port)

Run:  PYTHONPATH=src <py> scratchpad/r1_validate.py
"""

import math

import torch

from xorl.optim.muon import Muon
from xorl.server.zorl import build_zorl_update_from_rewards, iter_zorl_b_noises


torch.manual_seed(0)
OK = True


def report(name, passed, detail=""):
    global OK
    OK = OK and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# Realistic LoRA-B shapes (out, r=16): attn out_proj, mlp down, a wide one.
SHAPES = {
    "model.layers.0.self_attn.o_proj.lora_B": (4096, 16),
    "model.layers.0.mlp.down_proj.lora_B": (5120, 16),
    "model.layers.1.self_attn.o_proj.lora_B": (4096, 16),
    "model.layers.0.self_attn.o_proj.lora_A": (16, 4096),  # A present but ignored by B-fold
}
lora_params = {name: torch.zeros(shape) for name, shape in SHAPES.items()}


# ---------------------------------------------------------------------------
# A. seed -> noise stream parity
# ---------------------------------------------------------------------------
print("A. seed->noise stream parity (xorl iter_zorl_b_noises vs sglang primitive)")


def sglang_b_noise_primitive(lora_params, seed):
    """Faithful reproduction of sglang _zorl_normalized_b_noises core:
    torch.Generator(cpu).manual_seed(seed); for name in sorted(B-names): randn(shape, fp32).
    (No split_dim / no transpose for plain LoRA-B modules — same as xorl.)"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    names = sorted(n for n in lora_params if "lora_B" in n)
    out = {}
    for n in names:
        out[n] = torch.randn(tuple(lora_params[n].shape), generator=gen, dtype=torch.float32)
    return out


for seed in (12345, 2**31 + 7, 99):
    xorl_noise = dict(iter_zorl_b_noises(lora_params, seed=seed))
    ref_noise = sglang_b_noise_primitive(lora_params, seed)
    same_keys = set(xorl_noise) == set(ref_noise)
    bit_identical = same_keys and all(torch.equal(xorl_noise[k], ref_noise[k]) for k in xorl_noise)
    report(f"seed={seed}: bit-identical B-noise stream", bit_identical,
           f"{len(xorl_noise)} B-tensors, A excluded={'lora_A' not in ''.join(xorl_noise)}")


# ---------------------------------------------------------------------------
# B. G construction parity:  G = Σ_i c_i * (1/N) * noise(seed_i)
# ---------------------------------------------------------------------------
print("\nB. G construction (build_zorl_update_from_rewards vs explicit reward-weighted sum)")

pairs = [(1001, 0.8), (1002, -0.4), (1003, 1.2), (1004, -0.9), (1005, 0.1)]  # (b_seed, normalized_score)
G_xorl, G_norm = build_zorl_update_from_rewards(lora_params, pair_seeds_and_scores=pairs)

# explicit reference: same formula sglang documents (line ~2106): Σ score * (1/N) * noise(seed)
N = len(pairs)
G_ref = {n: torch.zeros(tuple(lora_params[n].shape)) for n in lora_params if "lora_B" in n}
for b_seed, score in pairs:
    for n, noise in iter_zorl_b_noises(lora_params, seed=b_seed):
        G_ref[n].add_(noise, alpha=float(score) * (1.0 / N))

g_match = set(G_xorl) == set(G_ref) and all(torch.allclose(G_xorl[k], G_ref[k], atol=1e-6) for k in G_xorl)
report("G = Σ c_i (1/N) noise_i matches explicit sum", g_match,
       f"{len(G_xorl)} B-modules, ‖G‖={G_norm:.4f}")
# B-only: A tensors must NOT appear in G
report("G contains only lora_B keys", all("lora_B" in k for k in G_xorl))


# ---------------------------------------------------------------------------
# C. NS-fold parity:  xorl Muon optim_step  vs  sglang _zorl_muon_update
# ---------------------------------------------------------------------------
print("\nC. NS-fold parity (xorl Muon vs faithful sglang _zorl_muon_update port)")


def sglang_muon_update_ref(delta, lr, compute_dtype):
    """Faithful port of sglang lora_manager._zorl_muon_update (no restart).
    Returns the update to ADD to base: lr * 0.2*sqrt(max(rows,cols)) * NS(delta)."""
    coeffs = (3.4445, -4.775, 2.0315)
    ns_steps = 5
    eps = 1e-7
    a, b, c = coeffs
    x = delta if delta.ndim == 3 else delta.unsqueeze(0)
    rows, cols = int(x.shape[-2]), int(x.shape[-1])
    o = x.to(compute_dtype)
    transposed = rows > cols
    if transposed:
        o = o.transpose(-2, -1).contiguous()
    norms = o.flatten(start_dim=1).norm(dim=1).clamp(min=eps).reshape(-1, 1, 1)
    o = o / norms
    if o.size(-2) == o.size(-1):
        for _ in range(ns_steps):
            gram = torch.bmm(o, o.transpose(-2, -1))
            gram = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            o = torch.baddbmm(o, gram, o, beta=a)
    else:
        bs = o.size(0)
        R = torch.bmm(o, o.transpose(-2, -1))
        m = R.size(-1)
        identity = torch.eye(m, device=o.device, dtype=o.dtype).unsqueeze(0).expand(bs, -1, -1).contiguous()
        Q = None
        for it in range(ns_steps):
            Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
            if it == 0:
                Q = Z + a * identity
            else:
                Q = torch.baddbmm(Q, Q, Z, beta=a)
            if it < ns_steps - 1:
                RZ = torch.baddbmm(R, R, Z, beta=a)
                R = torch.baddbmm(RZ, Z, RZ, beta=a)
        o = torch.bmm(Q, o)
    if transposed:
        o = o.transpose(-2, -1)
    o = o.to(delta.dtype)
    if delta.ndim == 2:
        o = o.squeeze(0)
    spectral = 0.2 * (float(max(rows, cols)) ** 0.5)
    return o * (float(lr) * spectral)


def xorl_applied_delta(G_module, lr, num_restarts):
    """Run one xorl Muon step with p.grad = -G (the apply_zorl_rewards convention)
    and return the applied delta (p_after - p_before) = +adjusted_lr * NS(G)."""
    p = torch.nn.Parameter(torch.zeros_like(G_module))  # fp32 master param
    p.grad = (-G_module).clone()
    opt = Muon(
        [{"params": [p], "use_muon": True}],
        lr=lr,
        momentum=0.0,                       # live recipe: momentum-off
        nesterov=False,
        weight_decay=0.0,
        adjust_lr_fn="match_rms_adamw",
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,         # CPU
        gram_newton_schulz_num_restarts=num_restarts,
        distributed_mode="full_gradient",   # no-op for non-DTensor CPU param
    )
    before = p.data.clone()
    opt.step()
    return p.data - before


def cosine(a, b):
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-30))


def relerr(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


LR = 2.5e-5  # sglang recipe
test_mods = {
    "o_proj [4096,16]": G_xorl["model.layers.0.self_attn.o_proj.lora_B"],
    "down_proj [5120,16]": G_xorl["model.layers.0.mlp.down_proj.lora_B"],
}
# also a 3D MoE-expert-stack delta [E, out, r]
test_mods["moe_stack [8,2048,16]"] = torch.randn(8, 2048, 16)

for label, Gm in test_mods.items():
    xorl_no_restart = xorl_applied_delta(Gm, LR, num_restarts=0)
    xorl_default = xorl_applied_delta(Gm, LR, num_restarts=1)
    ref_fp32 = sglang_muon_update_ref(Gm, LR, torch.float32)
    ref_bf16 = sglang_muon_update_ref(Gm, LR, torch.bfloat16)

    c0 = cosine(xorl_no_restart, ref_fp32)
    r0 = relerr(xorl_no_restart, ref_fp32)
    report(f"{label}: xorl(no-restart,fp32) == sglang(fp32)", c0 > 0.99999 and r0 < 1e-4,
           f"cos={c0:.7f} relerr={r0:.2e}")

    cd = cosine(xorl_default, ref_fp32)
    rd = relerr(xorl_default, ref_fp32)
    report(f"{label}: xorl(restart=[3]) direction vs sglang(fp32)", cd > 0.999,
           f"cos={cd:.6f} relerr={rd:.2e} (restart is a stability device)")

    cbf = cosine(xorl_no_restart, ref_bf16)
    rbf = relerr(xorl_no_restart, ref_bf16)
    report(f"{label}: xorl(fp32) vs sglang AS-SHIPPED(bf16) — fp32-master gain", True,
           f"cos={cbf:.6f} relerr={rbf:.2e}  (bf16 is the lossy live path)")

    # magnitude: applied per-element RMS should be ~0.2/sqrt(min)*lr-ish; just confirm scales match
    mag_x = float(xorl_no_restart.norm())
    mag_r = float(ref_fp32.norm())
    report(f"{label}: applied-update magnitude matches", abs(mag_x - mag_r) / (mag_r + 1e-30) < 1e-4,
           f"‖xorl‖={mag_x:.4e} ‖sglang‖={mag_r:.4e}")


print("\n" + ("=" * 60))
print("R1 VERDICT:", "PASS — xorl Muon reproduces the ES fold" if OK else "FAIL — see above")
print("=" * 60)
