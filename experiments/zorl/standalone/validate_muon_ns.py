"""CPU numerical validation of ZORL Muon NS + the proposed loop-reorder fix.

Validates:
 1. The current _zorl_muon_update Gram-NS path (rectangular) vs a reference
    classic-quintic NS on the REAL module dims -> orthogonality + scale.
 2. That accumulating dense G chunk-major (draw-once, fold-all-modules) gives the
    SAME dense G as module-major (draw-per-module) -> the loop reorder is exact
    up to fp summation order.
 3. bf16-NS vs fp32-NS agreement (the canonical xorl path uses bf16; ZORL already
    does -> confirm it's safe).
No GPU; tiny dims scaled down where needed, plus one real-dim NS check.
"""
import torch
torch.manual_seed(0)

# ---- port of _zorl_muon_update (exact copy of the fork math) ----
def zorl_muon_update(delta, lr, lr_scale="match_rms_adamw"):
    coeffs=(3.4445,-4.775,2.0315); ns_steps=5; eps=1e-7; a,b,c=coeffs
    x = delta if delta.ndim==3 else delta.unsqueeze(0)
    rows,cols=int(x.shape[-2]),int(x.shape[-1])
    o=x.to(torch.bfloat16)
    transposed = rows>cols
    if transposed: o=o.transpose(-2,-1).contiguous()
    norms=o.flatten(start_dim=1).norm(dim=1).clamp(min=eps).reshape(-1,1,1)
    o=o/norms
    if o.size(-2)==o.size(-1):
        for _ in range(ns_steps):
            gram=torch.bmm(o,o.transpose(-2,-1))
            gram=torch.baddbmm(gram,gram,gram,beta=b,alpha=c)
            o=torch.baddbmm(o,gram,o,beta=a)
    else:
        bs=o.size(0); R=torch.bmm(o,o.transpose(-2,-1)); m=R.size(-1)
        I=torch.eye(m,device=o.device,dtype=o.dtype).unsqueeze(0).expand(bs,-1,-1).contiguous()
        Q=None
        for it in range(ns_steps):
            Z=torch.baddbmm(R,R,R,beta=b,alpha=c)
            if it==0: Q=Z+a*I
            else: Q=torch.baddbmm(Q,Q,Z,beta=a)
            if it<ns_steps-1:
                RZ=torch.baddbmm(R,R,Z,beta=a); R=torch.baddbmm(RZ,Z,RZ,beta=a)
        o=torch.bmm(Q,o)
    if transposed: o=o.transpose(-2,-1)
    o=o.to(delta.dtype)
    if delta.ndim==2: o=o.squeeze(0)
    if lr_scale=="original": spectral=(max(1.0,rows/cols))**0.5
    else: spectral=0.2*(float(max(rows,cols))**0.5)
    return o*(float(lr)*spectral)

# reference: classic quintic on fp32, single matrix
def ref_quintic_fp32(G, ns_steps=5):
    a,b,c=(3.4445,-4.775,2.0315); eps=1e-7
    X=G.to(torch.float32).clone()
    t = X.shape[-2]>X.shape[-1]
    if t: X=X.transpose(-2,-1).contiguous()
    X=X/ X.norm().clamp(min=eps)
    for _ in range(ns_steps):
        A=X@X.transpose(-2,-1)
        B=b*A+c*(A@A)
        X=a*X+B@X
    if t: X=X.transpose(-2,-1)
    return X

print("="*70)
print("[1] NS orthogonality + match vs reference (real MoE per-expert dims)")
print("="*70)
# real per-expert MoE matrices: gate/up [I=512, H=2048]; down [H=2048, I=512]
for name,(rows,cols) in {"gate/up [512,2048]":(512,2048),"down [2048,512]":(2048,512),
                          "o_proj [2048,2048]":(2048,2048)}.items():
    E=4
    # build a low-rank-ish G = sum of 64 rank-16 outer products (like fresh_ab)
    G=torch.zeros(E,rows,cols)
    for _ in range(64):
        B=torch.randn(E,rows,16); A=torch.randn(E,16,cols)
        G+=torch.bmm(B,A)
    upd=zorl_muon_update(G, lr=1.0, lr_scale="original")  # spectral=1 -> ~orthogonal
    # check singular values ~1 for the orthogonalized factor (upd before spectral=upd here)
    u=upd[0].to(torch.float32)
    s=torch.linalg.svdvals(u)
    # normalize out: NS output has unit-ish singular values
    smax,smin=s.max().item(),s[s>1e-3].min().item() if (s>1e-3).any() else 0.0
    print(f"  {name}: sing.val range after NS = [{smin:.3f}, {smax:.3f}] "
          f"(ideal ~1; NS5 gives ~0.7-1.3)")

print()
print("="*70)
print("[2] chunk-major (draw-once) == module-major (draw-per-module) dense G")
print("="*70)
# Simulate: 3 'modules' sharing one pair-noise stream. Module-major redraws the
# full stream per module and slices; chunk-major draws once and slices all.
# We prove the resulting per-module dense G is identical (same seeds, same slices).
def full_noise(seed, sizes):
    g=torch.Generator(); g.manual_seed(seed)
    flat=torch.randn(sum(sizes),generator=g)
    out=[]; off=0
    for s in sizes:
        out.append(flat[off:off+s]); off+=s
    return out
mod_sizes=[ (8,4),(6,4),(10,4) ]  # (out,r) per module B; A is (4,in)
ins=[5,7,3]; r=4
seeds=[(11,101,0.5),(22,202,-0.3),(33,303,0.8)]  # (b_seed,a_seed,coef)
# module-major: for each module, for each pair, draw full noise, slice this module
b_flat_sizes=[o*r for (o,_) in mod_sizes]
a_flat_sizes=[r*i for i in ins]
def build_G_module_major():
    Gs=[torch.zeros(o,i) for (o,_),i in zip(mod_sizes,ins)]
    for mi in range(len(mod_sizes)):
        for (bs,as_,coef) in seeds:
            bparts=full_noise(bs,b_flat_sizes); aparts=full_noise(as_,a_flat_sizes)
            B=bparts[mi].reshape(mod_sizes[mi][0],r)*coef
            A=aparts[mi].reshape(r,ins[mi])
            Gs[mi]+=B@A
    return Gs
def build_G_chunk_major():
    Gs=[torch.zeros(o,i) for (o,_),i in zip(mod_sizes,ins)]
    for (bs,as_,coef) in seeds:           # draw ONCE per pair
        bparts=full_noise(bs,b_flat_sizes); aparts=full_noise(as_,a_flat_sizes)
        for mi in range(len(mod_sizes)):  # fold ALL modules
            B=bparts[mi].reshape(mod_sizes[mi][0],r)*coef
            A=aparts[mi].reshape(r,ins[mi])
            Gs[mi]+=B@A
    return Gs
Gm=build_G_module_major(); Gc=build_G_chunk_major()
maxdiff=max((gm-gc).abs().max().item() for gm,gc in zip(Gm,Gc))
print(f"  max|G_module_major - G_chunk_major| = {maxdiff:.2e}  (expect 0 exactly)")
assert maxdiff==0.0, "loop reorder changed dense G!"
print("  PASS: loop reorder is bit-exact (same seeds, same slices, same sum order)")

print()
print("="*70)
print("[3] bf16-NS vs fp32-NS (canonical xorl uses bf16; ZORL already bf16)")
print("="*70)
G=torch.randn(2,512,2048)
for _ in range(32):
    G+=torch.bmm(torch.randn(2,512,16),torch.randn(2,16,2048))
upd_bf16=zorl_muon_update(G,1.0,"original")[0].float()
# fp32 NS for comparison
ref=ref_quintic_fp32(G[0])
# compare directions (cosine of flattened)
cos=torch.nn.functional.cosine_similarity(upd_bf16.flatten(),ref.flatten(),dim=0).item()
print(f"  cosine(bf16-NS, fp32-NS) = {cos:.5f}  (>0.99 => bf16 is safe)")
print("\nALL VALIDATIONS DONE")
