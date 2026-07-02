import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/xorl-sglang-zorl/python"))
from sglang.srt.lora.fp8_fold import fp8_fold_, E4M3_MAX
torch.manual_seed(0); BN=BK=128; R=C=256; N=40; DRMS=3e-3
def q(W):
    nbr,nbc=(R+BN-1)//BN,(C+BK-1)//BK; si=torch.zeros(nbr,nbc); Wq=torch.zeros_like(W,dtype=torch.float8_e4m3fn)
    for i in range(nbr):
        for j in range(nbc):
            b=W[i*BN:(i+1)*BN,j*BK:(j+1)*BK]; s=max(float(b.abs().max())/E4M3_MAX,1e-12); si[i,j]=s
            Wq[i*BN:(i+1)*BN,j*BK:(j+1)*BK]=(b/s).clamp(-E4M3_MAX,E4M3_MAX).to(torch.float8_e4m3fn)
    return Wq,si
def eff(Wq,si): return Wq.float()*si.repeat_interleave(BN,0).repeat_interleave(BK,1)[:R,:C]
def re(a,b): return float((a-b).norm()/(b.norm()+1e-12))
W0=torch.randn(R,C)*0.02; deltas=[torch.randn(R,C)*DRMS for _ in range(N)]; truth=W0+torch.stack(deltas).sum(0)
print("Per-step E4M3 ULP at |W|~0.02:  the local grid step near 0.02 (~2^-6) is 2^-6 * 2^-3 = 2^-9 ~ 0.002")
print(f"per-step delta rms {DRMS:.0e} is ~1.5x the ULP -> each step is well-resolved individually")
print()
# 1) eager SR per-step
for mode in ("stochastic","stochastic_lazy","stochastic_ef","stochastic_ef_lazy"):
    Wq,si=q(W0.clone()); g=torch.Generator().manual_seed(1)
    ef=torch.zeros(R,C,dtype=torch.bfloat16) if "ef" in mode else None
    for s in range(N):
        fp8_fold_(Wq,si,deltas[s].to(torch.bfloat16),mode=mode,block_n=BN,block_k=BK,ef_residual=ef,generator=g)
    print(f"  {mode:20s}: rel_err@40 = {re(eff(Wq,si),truth):.4f}")
print()
print("Interpretation: if *_ef modes (error-feedback, NO accumulating SR variance)")
print("track truth far better, the drift is SR variance on moved entries, NOT the")
print("whole-block re-round -> lazy (which only removes whole-block re-round) can't fix it.")
