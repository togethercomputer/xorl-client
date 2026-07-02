"""Analytical FLOP/byte model for the ZORL Muon NS 'apply' step.

Qwen3.6-35B-A3B-FP8, TP=2. Numbers from config.json + lora_manager.py code paths.
No GPU needed; pure arithmetic.
"""

# ---- model dims (config.json text_config) ----
L = 40                      # num_hidden_layers
H = 2048                    # hidden_size
I = 512                     # moe_intermediate_size (per-expert)
E_total = 256               # num_experts
TP = 2
E = E_total // TP           # 128 experts per rank
r = 16                      # lora rank
full_attn_interval = 4
n_attn_layers = L // full_attn_interval   # 10 full-attention layers
head_dim = 256
n_q = 16; n_kv = 2
q_dim = n_q * head_dim          # 4096
kv_dim = n_kv * head_dim        # 512
qkv_out = q_dim + 2 * kv_dim    # 5120 (fused), TP-sliced -> /2
o_in = q_dim                    # o_proj in = 4096

# ---- run params (from live log) ----
used_pairs = 640
momentum_window = 8
# muon_pairs = current + history. update_norm=0.60 implies momentum active.
# assume ~ used_pairs current; history adds up to window*used_pairs but capped by beta^k<eps.
# The log shows used_pairs=640 (1280 antithetic candidates -> 640 pairs).
N_cur = 640
# beta unknown; assume momentum present. Be conservative: count current only for the
# dominant terms, then show momentum multiplier.
N = N_cur

pairs_per_pass = 32         # capped at 32 in gpu_direct
H100_BF16_TFLOPS = 990      # peak dense bf16 (sxm5), realistic ~ 60-70% for these shapes
H100_FP32_TFLOPS = 67       # fp32 (non-tensor-core) ~ tensor-core fp32 ~ 67-495; use cuda-core fp32
H100_HBM_TBs = 3.35         # HBM3 bandwidth TB/s

GFLOP = 1e9
TFLOP = 1e12

def fmt(x):
    return f"{x/TFLOP:8.2f} TFLOP" if x >= TFLOP else f"{x/GFLOP:8.2f} GFLOP"

print("="*78)
print("ZORL Muon NS apply — analytical FLOP/byte model (per replica, TP rank)")
print("="*78)
print(f"L={L} H={H} I={I} E(per rank)={E} r={r} N(pairs)={N} pairs_per_pass={pairs_per_pass}")
print(f"full-attn layers={n_attn_layers}, MoE layers={L}")
print()

# =====================================================================
# TERM 1: per-chunk dense G accumulation (the B@A bmm)
# =====================================================================
# MoE: per layer, per chunk, 3 bmms: bmm(b_gate[E,I,cr], a_gate[E,cr,H]) etc.
# chunk rank cr = r * pairs_per_pass = 16*32 = 512
# gate: [E,I,cr]@[E,cr,H] = E * I*cr*H *2 flops ; up same; down [E,H,cr]@[E,cr,I]
cr = r * pairs_per_pass
n_chunks = (N + pairs_per_pass - 1)//pairs_per_pass
print(f"chunk rank cr=r*pairs_per_pass={cr}, n_chunks per module={n_chunks}")

# per chunk per moe layer:
flop_gate = 2 * E * I * cr * H
flop_up   = 2 * E * I * cr * H
flop_down = 2 * E * H * cr * I
moe_bmm_per_chunk = flop_gate + flop_up + flop_down
moe_bmm_total = moe_bmm_per_chunk * n_chunks * L
print(f"\n[T1a] MoE dense-G bmm: {fmt(moe_bmm_per_chunk)}/chunk/layer "
      f"x {n_chunks} chunks x {L} layers = {fmt(moe_bmm_total)}")

# standard (attn) per chunk: qkv [out,cr]@[cr,H]; o_proj [H,cr]@[cr,o_in]
# qkv fused out (TP-sliced) ~ qkv_out/2 = 2560 ; in = H
qkv_out_shard = qkv_out // TP
flop_qkv = 2 * qkv_out_shard * cr * H
o_out_shard = H            # o_proj out = H (row-parallel: out=H, in sliced)
o_in_shard = o_in // TP
flop_o = 2 * o_out_shard * cr * o_in_shard
std_bmm_per_chunk = flop_qkv + flop_o
std_bmm_total = std_bmm_per_chunk * n_chunks * n_attn_layers
print(f"[T1b] attn dense-G mm: {fmt(std_bmm_per_chunk)}/chunk/layer "
      f"x {n_chunks} x {n_attn_layers} layers = {fmt(std_bmm_total)}")

T1 = moe_bmm_total + std_bmm_total
print(f"[T1] TOTAL dense-G accumulation = {fmt(T1)}  (fp32 cuda-core)")
print(f"     at fp32 {H100_FP32_TFLOPS} TFLOP/s ideal => {T1/(H100_FP32_TFLOPS*TFLOP):6.2f} s")

# =====================================================================
# TERM 2: Newton-Schulz orthogonalization (per module, batched over E)
# =====================================================================
# Gram-NS for rectangular [E, m, n] with m=min dim. 5 steps.
# Cost per step dominated by bmm on Gram (m x m) and Q@o materialize.
# For MoE gate/up: matrix [I,H]=[512,2048] -> m=512, n=2048. transpose so rows<=cols already.
# Gram-NS recurrence per step: ~ a few [E,m,m]@[E,m,m] (m^3) + final Q@o once (E*m*m*n)
ns_steps = 5
def gram_ns_flop(E_, m, n):
    # per step: R@R (m^3*2), baddbmm Z (m^3*2), Q update (m^3*2), R update ~2x(m^3*2)
    # ~ approx 5 mxmxm matmuls per step = 5*2*m^3 ; plus final Q@o = 2*m*m*n
    per_step = 5 * 2 * m**3
    final = 2 * m * m * n
    return E_ * (ns_steps * per_step + final)

# MoE: gate [512,2048], up [512,2048], down: w2 is [K=H=2048, inter=I=512] -> m=512,n=2048 after transpose
ns_gate = gram_ns_flop(E, min(I,H), max(I,H))
ns_up   = gram_ns_flop(E, min(I,H), max(I,H))
ns_down = gram_ns_flop(E, min(H,I), max(H,I))
ns_moe_per_layer = ns_gate + ns_up + ns_down
ns_moe_total = ns_moe_per_layer * L
print(f"\n[T2a] MoE NS (Gram, 5 step, bf16, batched E={E}): "
      f"{fmt(ns_moe_per_layer)}/layer x {L} = {fmt(ns_moe_total)}")

# attn: qkv [2560,2048] m=2048 (square-ish), o [2048,2048] square -> classic quintic
def quintic_flop(m, n):
    # 5 steps, each: gram o@oT (m*m*n*2... for [m,n]: oo^T is m*m*n), gram@gram (m^3*2), gram@o (m*m*n*2)
    # use rows=m<=n after transpose
    per_step = 2*m*m*n + 2*m**3 + 2*m*m*n
    return ns_steps*per_step
ns_qkv = quintic_flop(min(qkv_out_shard,H), max(qkv_out_shard,H))
ns_o   = quintic_flop(H,H)
ns_std_total = (ns_qkv + ns_o) * n_attn_layers
print(f"[T2b] attn NS: {fmt(ns_qkv+ns_o)}/layer x {n_attn_layers} = {fmt(ns_std_total)}")

T2 = ns_moe_total + ns_std_total
print(f"[T2] TOTAL Newton-Schulz = {fmt(T2)}  (bf16 tensor-core)")
print(f"     at bf16 {H100_BF16_TFLOPS} TFLOP/s ideal => {T2/(H100_BF16_TFLOPS*TFLOP):6.3f} s")

# =====================================================================
# TERM 3: GPU noise regeneration (randn) — the module-major tax
# =====================================================================
# _zorl_module_pair_pieces draws the FULL adapter's A+B noise per pair, per module, per chunk!
# full adapter noise numel per seed (A or B) ~ sum over all lora weights.
# Per layer LoRA params (rank r):
#  - MoE gate_up: A shared [2r,H]? B per-expert [E,2I,r]; down A per-expert [E,r,I], B shared [H,r]
#  - attn qkv: A [3r? r, H]?, B [qkv_out, r]; o: A[r, o_in], B[H,r]
# Use the adapter.layers weights numel. Approx from shapes:
def moe_lora_numel():
    # shared-outer: gu_a [1,2r,H], gu_b [E,2I,r], dn_a [E,r,I], dn_b [1,H,r]
    gu_a = 2*r*H
    gu_b = E*2*I*r
    dn_a = E*r*I
    dn_b = H*r
    return gu_a+gu_b+dn_a+dn_b
def attn_lora_numel():
    qkv_a = r*H; qkv_b = qkv_out*r   # pre-TP full
    o_a = r*o_in; o_b = H*r
    return qkv_a+qkv_b+o_a+o_b
# B-noise draw (one seed) regenerates ALL B for the adapter; A-noise draw all A.
# But module_pair_pieces draws BOTH full a_noise and b_noise per pair, keeps one module.
per_seed_numel = L*moe_lora_numel() + n_attn_layers*attn_lora_numel()   # ~ A+B combined order
# It draws a_noise(full) AND b_noise(full) per pair => ~2x per_seed_numel randn floats.
randn_per_pair_per_module_chunk = 2 * per_seed_numel
# called: per module, per chunk, per pair-in-chunk
n_modules = L + 2*n_attn_layers   # 40 MoE + 20 attn = 60
total_pairs_drawn = n_modules * N    # each pair drawn once per module (chunks partition pairs)
total_randn_floats = total_pairs_drawn * (2*per_seed_numel)
print(f"\n[T3] GPU noise REGENERATION (module-major tax):")
print(f"     per-seed full-adapter noise numel (A or B-ish) ~ {per_seed_numel/1e6:.1f} M floats")
print(f"     modules folded = {n_modules} (40 MoE + 20 attn)")
print(f"     each of {N} pairs' FULL noise is regenerated ONCE PER MODULE = {n_modules}x")
print(f"     total randn floats ~ {total_randn_floats/1e9:.1f} G floats (fp32)")
# randn throughput on H100 ~ 20-40 G floats/s (philox, fp32). Use 30 Gf/s.
randn_Gfs = 30
print(f"     at ~{randn_Gfs} Gfloat/s randn => {total_randn_floats/1e9/randn_Gfs:6.2f} s")
# Also the memory traffic: each draw writes per_seed_numel floats, slices, discards.
noise_bytes = total_randn_floats*4
print(f"     noise mem write traffic ~ {noise_bytes/1e12:.2f} TB => "
      f"{noise_bytes/(H100_HBM_TBs*1e12):.2f} s @ HBM")

# =====================================================================
# TERM 4: FP8 stochastic-rounding fold (dequant->add->amax->searchsorted->requant)
# =====================================================================
# Per MoE module finalize: w13 [E,2I,H] + w2 [E,H,I] elements get the full SR pass.
w13_numel = E*2*I*H
w2_numel  = E*H*I
moe_fp8_numel = (w13_numel + w2_numel) * L
# attn fp8: qkv [qkv_out_shard,H], o [H,o_in_shard]
attn_fp8_numel = (qkv_out_shard*H + H*o_in_shard) * n_attn_layers
fp8_numel = moe_fp8_numel + attn_fp8_numel
# SR pass touches each element ~ many times (dequant, add, amax, searchsorted over 127 grid,
# bracket gather, rand, requant). searchsorted is ~log2(127)=7 comparisons. Call it ~15 elementwise
# passes + 1 searchsorted. Bandwidth-bound: ~ 20 * numel * 4 bytes read/write.
fp8_passes = 20
fp8_bytes = fp8_numel * 4 * fp8_passes
print(f"\n[T4] FP8 stochastic-rounding fold:")
print(f"     folded base elements ~ {fp8_numel/1e9:.2f} G (MoE {moe_fp8_numel/1e9:.2f}G + attn)")
print(f"     ~{fp8_passes} elementwise passes (incl searchsorted) => {fp8_bytes/1e12:.2f} TB")
print(f"     @ HBM {H100_HBM_TBs} TB/s => {fp8_bytes/(H100_HBM_TBs*1e12):6.2f} s")

print("\n" + "="*78)
print("SUMMARY (ideal-rate lower bounds; real is 2-5x due to fp32, small bmm, launch)")
print("="*78)
t1s=T1/(H100_FP32_TFLOPS*TFLOP); t2s=T2/(H100_BF16_TFLOPS*TFLOP)
t3s=total_randn_floats/1e9/randn_Gfs; t4s=fp8_bytes/(H100_HBM_TBs*1e12)
print(f"  T1 dense-G bmm (fp32)        : {t1s:6.2f} s")
print(f"  T2 Newton-Schulz (bf16)      : {t2s:6.3f} s   <-- the 'NS itself' is TINY")
print(f"  T3 noise regeneration        : {t3s:6.2f} s")
print(f"  T4 FP8 SR fold               : {t4s:6.2f} s")
print(f"  ideal-sum                    : {t1s+t2s+t3s+t4s:6.2f} s  (vs measured t_apply=173s)")
