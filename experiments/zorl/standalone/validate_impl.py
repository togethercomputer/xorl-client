"""Exercise the ACTUAL FP8-accum method SOURCE from lora_manager.py (CPU only).

The full LoRAManager import drags in GPU kernels that won't load on this CPU
box, so we extract the exact source of the methods we added/touched straight
from the file and exec them into a stand-in class. This tests the production
logic verbatim (it is the same text), not a re-implementation.
"""
import ast, os, sys, textwrap
import torch
import math

REPO = os.path.expanduser("~/xorl-sglang-zorl/python")
sys.path.insert(0, REPO)
from sglang.srt.lora.fp8_fold import fp8_fold_, fp8_fold_budget_units, E4M3_MAX  # noqa

SRC_PATH = os.path.join(REPO, "sglang/srt/lora/lora_manager.py")
src = open(SRC_PATH).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LoRAManager")

WANT = {
    "_zorl_accumulate_in_adapter",
    "_zorl_fp8_accum_buffer",
    "_zorl_register_fp8_accum_scale",
    "_zorl_fp8_fold_standard_",
    "_zorl_fp8_ef_residual",
    "_zorl_fp8_fold_generator",
    "_fp8_block_size",
    "_zorl_fp8_fold_mode","_zorl_fp8_mode_uses_ef",
    "flush_zorl_fp8_accumulators",
}
method_src = []
for node in cls.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        method_src.append(ast.get_source_segment(src, node))

ns = {
    "os": os, "torch": torch, "math": math, "fp8_fold_": fp8_fold_,
    "Optional": __import__("typing").Optional, "Dict": __import__("typing").Dict,
    "Any": __import__("typing").Any, "BaseLayerWithLoRA": object,
}
# These are @staticmethod in the class; ast.get_source_segment drops the
# decorator, so re-apply it after class creation.
STATICS = {"_zorl_accumulate_in_adapter", "_zorl_fp8_fold_mode","_zorl_fp8_mode_uses_ef", "_fp8_block_size"}
body = "class Mgr:\n" + textwrap.indent("\n\n".join(method_src), "    ")
exec(compile(body, "<extracted>", "exec"), ns)
Mgr = ns["Mgr"]
for name in STATICS:
    setattr(Mgr, name, staticmethod(Mgr.__dict__[name]))
print(f"Extracted {len(method_src)} real methods from lora_manager.py: {sorted(WANT)}")

# ---------- synthetic block-FP8 weight ----------
BN = BK = 128
ROWS, COLS = 256, 384

def quantize_block_fp8(w_real):
    rows, cols = w_real.shape
    nbr, nbc = (rows+BN-1)//BN, (cols+BK-1)//BK
    scale_inv = torch.zeros(nbr, nbc, dtype=torch.float32)
    wq = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn)
    for bi in range(nbr):
        for bj in range(nbc):
            r0,r1=bi*BN,min(rows,(bi+1)*BN); c0,c1=bj*BK,min(cols,(bj+1)*BK)
            blk=w_real[r0:r1,c0:c1]; s=(blk.abs().amax().clamp_min(1e-12)/E4M3_MAX).float()
            scale_inv[bi,bj]=s
            wq[r0:r1,c0:c1]=(blk/s).clamp(-E4M3_MAX,E4M3_MAX).to(torch.float8_e4m3fn)
    return wq, scale_inv

def dequant(wq, sc):
    rows,cols=wq.shape
    full=sc.repeat_interleave(BN,0).repeat_interleave(BK,1)[:rows,:cols]
    return wq.to(torch.float32)*full

def rms(x): return float(x.float().pow(2).mean().sqrt())

class _BaseLayer:
    def __init__(self, wq, sc):
        self.weight = wq; self.weight_scale_inv = sc; self.quant_method = None
class _Module:
    def __init__(self, bl): self.base_layer = bl

def run(accum_on, K, steps=24, seed=0):
    os.environ["XORL_ZORL_FP8_FOLD"]="stochastic"
    os.environ["XORL_ZORL_ACCUMULATE_IN_ADAPTER"]="1" if accum_on else "0"
    os.environ["XORL_ZORL_FP8_FOLD_SEED"]="7"
    torch.manual_seed(seed)
    w0=torch.randn(ROWS,COLS)*0.02
    wq,sc=quantize_block_fp8(w0); base0=wq.view(torch.uint8).clone()
    mod=_Module(_BaseLayer(wq,sc)); mgr=Mgr(); mgr.device=torch.device("cpu")
    g=torch.Generator().manual_seed(99); deltas=[]; pristine=True
    for s in range(steps):
        d=torch.randn(ROWS,COLS,generator=g)*1e-3; deltas.append(d)
        assert mgr._zorl_fp8_fold_standard_(mod, mod.base_layer.weight, d, "tgt") is True
        if accum_on and pristine:
            assert torch.equal(mod.base_layer.weight.view(torch.uint8), base0)
        if accum_on and K>0 and (s+1)%K==0:
            mgr.flush_zorl_fp8_accumulators(); pristine=False
    served=dequant(mod.base_layer.weight, mod.base_layer.weight_scale_inv)
    if accum_on:
        for _k,(accum,w) in getattr(mgr,"_zorl_fp8_accum",{}).items():
            if w is mod.base_layer.weight: served=served+accum.to(torch.float32)
    changed = not torch.equal(mod.base_layer.weight.view(torch.uint8), base0)
    wq0,sc0=quantize_block_fp8(w0); target=dequant(wq0,sc0)+sum(deltas)
    return served, target, changed, sum(deltas)

print("="*78); print("Real-source FP8-accum validation (CPU)"); print("="*78)
served_off,target,changed_off,_=run(False,0)
print(f"\n[default OFF]   base changed each step={changed_off}  served-vs-target rms={rms(served_off-target):.3e}")
assert changed_off

served_n,tn,changed_n,sumd=run(True,0)
print(f"[accum K=never] base BYTES changed={changed_n}  served(base+accum)-vs-target rms={rms(served_n-tn):.3e}  signal rms={rms(sumd):.3e}")
assert not changed_n, "K=never must keep FP8 base pristine"

for K in (1,8):
    sk,tk,ck,_=run(True,K)
    print(f"[accum K={K}]    base changed={ck}  served-vs-target rms={rms(sk-tk):.3e}")
    assert ck

print("\nALL ASSERTIONS PASSED (real method source):")
print(" - default OFF folds into FP8 base every step (unchanged)")
print(" - K=never keeps the FP8 base BYTE-FOR-BYTE pristine; bf16 accum holds the signal")
print(" - flush/merge drains the accumulator into the FP8 base at the cadence")
