"""Validate the MoE-seg accumulator path (_zorl_fp8_fold_moe_seg_) + flush."""
import ast, os, sys, textwrap, math
import torch
REPO = os.path.expanduser("~/xorl-sglang-zorl/python"); sys.path.insert(0, REPO)
from sglang.srt.lora.fp8_fold import fp8_fold_, E4M3_MAX
src = open(os.path.join(REPO,"sglang/srt/lora/lora_manager.py")).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="LoRAManager")
WANT={"_zorl_accumulate_in_adapter","_zorl_fp8_accum_buffer","_zorl_register_fp8_accum_scale",
      "_zorl_fp8_fold_moe_seg_","_zorl_fp8_ef_residual","_zorl_fp8_fold_generator",
      "_fp8_block_size","_zorl_fp8_fold_mode","_zorl_fp8_mode_uses_ef","flush_zorl_fp8_accumulators"}
ms=[ast.get_source_segment(src,n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in WANT]
ns={"os":os,"torch":torch,"math":math,"fp8_fold_":fp8_fold_,
    "Optional":__import__("typing").Optional,"Dict":__import__("typing").Dict,"Any":__import__("typing").Any}
exec(compile("class Mgr:\n"+textwrap.indent("\n\n".join(ms),"    "),"<x>","exec"),ns)
Mgr=ns["Mgr"]
for nm in ("_zorl_accumulate_in_adapter","_zorl_fp8_fold_mode","_zorl_fp8_mode_uses_ef","_fp8_block_size"):
    setattr(Mgr,nm,staticmethod(Mgr.__dict__[nm]))
BN=BK=128; E,OUT,K = 4, 256, 256  # 4 experts, w2-like [E, K, inter]
def qfp8_3d(w):
    Ee,rows,cols=w.shape; nbr,nbc=(rows+BN-1)//BN,(cols+BK-1)//BK
    sc=torch.zeros(Ee,nbr,nbc); wq=torch.zeros(Ee,rows,cols,dtype=torch.float8_e4m3fn)
    for e in range(Ee):
        for bi in range(nbr):
            for bj in range(nbc):
                r0,r1=bi*BN,min(rows,(bi+1)*BN); c0,c1=bj*BK,min(cols,(bj+1)*BK)
                blk=w[e,r0:r1,c0:c1]; s=(blk.abs().amax().clamp_min(1e-12)/E4M3_MAX).float(); sc[e,bi,bj]=s
                wq[e,r0:r1,c0:c1]=(blk/s).clamp(-E4M3_MAX,E4M3_MAX).to(torch.float8_e4m3fn)
    return wq,sc
def deq3d(wq,sc):
    Ee,rows,cols=wq.shape
    full=sc.repeat_interleave(BN,1).repeat_interleave(BK,2)[:,:rows,:cols]; return wq.to(torch.float32)*full
def rms(x): return float(x.float().pow(2).mean().sqrt())
class _Mod:  # module with base_layer.quant_method-less default block size
    class base_layer: quant_method=None
torch.manual_seed(3)
os.environ["XORL_ZORL_FP8_FOLD"]="stochastic"; os.environ["XORL_ZORL_ACCUMULATE_IN_ADAPTER"]="1"; os.environ["XORL_ZORL_FP8_FOLD_SEED"]="5"
w=torch.randn(E,OUT,K)*0.02; wq,sc=qfp8_3d(w); base0=wq.view(torch.uint8).clone()
mgr=Mgr(); mgr.device=torch.device("cpu"); mod=_Mod()
g=torch.Generator().manual_seed(11); deltas=[]
for s in range(16):
    d=torch.randn(E,OUT,K,generator=g)*1e-3; deltas.append(d)
    mgr._zorl_fp8_fold_moe_seg_(mod, wq, sc, d, None, "stochastic")
    assert torch.equal(wq.view(torch.uint8),base0)  # pristine until flush
info=mgr.flush_zorl_fp8_accumulators()
changed=not torch.equal(wq.view(torch.uint8),base0)
wq0,sc0=qfp8_3d(w); target=deq3d(wq0,sc0)+sum(deltas)
served=deq3d(wq,sc)
print(f"[MoE seg accum] pristine-until-flush=OK  flush folded_weights={info['folded_weights']}  base changed after flush={changed}")
print(f"   served-vs-target rms={rms(served-target):.3e}")
assert changed and info['folded_weights']==1
print("MoE-seg accumulator + flush: PASSED")
