import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/xorl-sglang-zorl/python"))
from sglang.srt.lora.fp8_fold import fp8_fold_, E4M3_MAX
torch.manual_seed(0); BN=BK=128; R=C=256; N=40
def q(W):
    nbr,nbc=(R+BN-1)//BN,(C+BK-1)//BK; si=torch.zeros(nbr,nbc); Wq=torch.zeros_like(W,dtype=torch.float8_e4m3fn)
    for i in range(nbr):
        for j in range(nbc):
            b=W[i*BN:(i+1)*BN,j*BK:(j+1)*BK]; s=max(float(b.abs().max())/E4M3_MAX,1e-12); si[i,j]=s
            Wq[i*BN:(i+1)*BN,j*BK:(j+1)*BK]=(b/s).clamp(-E4M3_MAX,E4M3_MAX).to(torch.float8_e4m3fn)
    return Wq,si
def eff(Wq,si): return Wq.float()*si.repeat_interleave(BN,0).repeat_interleave(BK,1)[:R,:C]
def re(a,b): return float((a-b).norm()/(b.norm()+1e-12))
W0=torch.randn(R,C)*0.02
for tag,drms,frac in [("dense SGD rms=3e-3",3e-3,1.0),("dense small rms=3e-4",3e-4,1.0),
                      ("dense tiny rms=5e-5 (Muon-ish)",5e-5,1.0),
                      ("SPARSE 5% rms=3e-3",3e-3,0.05),("SPARSE 1% rms=3e-3",3e-3,0.01)]:
    deltas=[]
    for _ in range(N):
        d=torch.randn(R,C)*drms
        if frac<1.0:
            m=(torch.rand(R,C)<frac).float(); d=d*m/ (frac**0.5)  # keep rms ~ same
        deltas.append(d)
    truth=W0+torch.stack(deltas).sum(0)
    line=f"{tag:34s}"
    for mode in ("stochastic","stochastic_lazy"):
        Wq,si=q(W0.clone()); g=torch.Generator().manual_seed(1); tot=0
        for s in range(N):
            tot+=int(fp8_fold_(Wq,si,deltas[s].to(torch.bfloat16),mode=mode,block_n=BN,block_k=BK,generator=g))
        line+=f" | {mode[-4:] if 'lazy' in mode else 'eager'}: re={re(eff(Wq,si),truth):.4f} resc={tot}"
    print(line)
