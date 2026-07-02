"""Validate band-major vs module-major loop reorder for the muon dense-G fold.

Models the real structure: multiple layers, each with a MoE module (3 segments,
batched over experts) + (on some layers) attn modules. Each pair's noise is a
seeded full-adapter randn stream. We prove:
  - module-major (draw full noise per module per pair, slice one module)
  - band-major   (draw full noise once per band-chunk, slice all modules)
produce IDENTICAL per-module dense G (bit-exact: same seeds, same slices, same
chunked add order within a module), for arbitrary band_size and chunk_size.
"""
import torch
torch.manual_seed(1)

# small but structurally faithful dims
nL=12; E=4; I=8; H=16; r=4
n_attn_every=4            # attn modules every 4th layer (like full_attention_interval=4)
N=20; chunk_size=7        # pairs, chunk

# full-adapter noise layout per seed: ordered list of (layer, name, shape)
# MoE: gu_b [E,2I,r], dn_b[?]; for simplicity make per-expert B and shared-ish A.
# We just need a deterministic full stream that both methods slice identically.
def layout():
    items=[]  # (layer,name,kind('A'/'B'),shape)
    for lid in range(nL):
        # MoE
        items.append((lid,"moe.gate_up.lora_A","A",(2*r,H)))
        items.append((lid,"moe.gate_up.lora_B","B",(E,2*I,r)))
        items.append((lid,"moe.down.lora_A","A",(E,r,I)))
        items.append((lid,"moe.down.lora_B","B",(H,r)))
        if lid % n_attn_every==0:
            items.append((lid,"qkv.lora_A","A",(r,H)))
            items.append((lid,"qkv.lora_B","B",(3*I,r)))
    return items
LAYOUT=layout()
def draw(seed, kind):
    g=torch.Generator(); g.manual_seed(seed)
    sizes=[int(torch.tensor(sh).prod()) for (_,_,k,sh) in LAYOUT if k==kind]
    flat=torch.randn(sum(sizes),generator=g)
    out={}; off=0
    for (lid,name,k,sh) in LAYOUT:
        if k!=kind: continue
        n=int(torch.tensor(sh).prod())
        out[(lid,name)]=flat[off:off+n].reshape(sh); off+=n
    return out

seeds=[(100+i, 500+i, ((-1)**i)*(0.1+0.01*i)) for i in range(N)]

def cat_a(pieces, lora_rank):
    first=pieces[0]; rank_dim=first.dim()-2; rows=first.shape[rank_dim]
    sf=rows//lora_rank
    if sf==1: return torch.cat(pieces,dim=rank_dim).contiguous()
    seg=[p.unflatten(rank_dim,(sf,lora_rank)) for p in pieces]
    return torch.cat(seg,dim=rank_dim+1).flatten(rank_dim,rank_dim+1).contiguous()

def fold_moe_segs(gu_a,gu_b,dn_a,dn_b,cr):
    # gu_a [2*cr,H] (gate rows then up rows, concat-along-rank), gu_b [E,2I,cr]
    a=gu_a.unsqueeze(0).expand(E,-1,-1)
    a_gate=a[:,0:cr,:]; a_up=a[:,cr:2*cr,:]
    b_gate=gu_b[:,0:I,:cr]; b_up=gu_b[:,I:2*I,:cr]
    g_gate=torch.bmm(b_gate,a_gate); g_up=torch.bmm(b_up,a_up)
    a_dn=dn_a  # [E,cr,I]
    b_dn=dn_b.unsqueeze(0).expand(E,-1,-1)[:,:,:cr]  # [E,H,cr]
    g_dn=torch.bmm(b_dn,a_dn)  # [E,H,I]
    return {"gate":g_gate,"up":g_up,"down":g_dn}

def module_major():
    G={}  # (lid,seg)->tensor
    for lid in range(nL):
        # MoE module: accumulate over chunks
        acc={}
        for s in range(0,N,chunk_size):
            ch=seeds[s:s+chunk_size]
            # stack chunk B (scaled by coef) and A along rank
            gu_a=[];gu_b=[];dn_a=[];dn_b=[]
            for (bs,as_,coef) in ch:
                B=draw(bs,"B"); A=draw(as_,"A")
                gu_b.append(B[(lid,"moe.gate_up.lora_B")]*coef)
                dn_b.append(B[(lid,"moe.down.lora_B")]*coef)
                gu_a.append(A[(lid,"moe.gate_up.lora_A")])
                dn_a.append(A[(lid,"moe.down.lora_A")])
            cr=r*len(ch)
            GU_B=torch.cat(gu_b,dim=-1); DN_B=torch.cat(dn_b,dim=-1)
            GU_A=cat_a(gu_a,r); DN_A=cat_a(dn_a,r)
            segs=fold_moe_segs(GU_A,GU_B,DN_A,DN_B,cr)
            for k,v in segs.items():
                acc[k]=v if k not in acc else acc[k]+v
        for k,v in acc.items(): G[(lid,"moe",k)]=v
        # attn module (if present)
        if lid%n_attn_every==0:
            acc2={}
            for s in range(0,N,chunk_size):
                ch=seeds[s:s+chunk_size]
                qb=[];qa=[]
                for (bs,as_,coef) in ch:
                    B=draw(bs,"B");A=draw(as_,"A")
                    qb.append(B[(lid,"qkv.lora_B")]*coef); qa.append(A[(lid,"qkv.lora_A")])
                QB=torch.cat(qb,dim=-1); QA=torch.cat(qa,dim=0)
                d=QB@QA
                acc2["qkv"]=d if "qkv" not in acc2 else acc2["qkv"]+d
            G[(lid,"qkv","d")]=acc2["qkv"]
    return G

def band_major(band):
    G={}
    for b0 in range(0,nL,band):
        blayers=list(range(b0,min(b0+band,nL)))
        macc={lid:{} for lid in blayers}; aacc={lid:{} for lid in blayers}
        for s in range(0,N,chunk_size):
            ch=seeds[s:s+chunk_size]
            # ONE draw per pair for whole band
            drawn=[(draw(bs,"B"),draw(as_,"A"),coef) for (bs,as_,coef) in ch]
            for lid in blayers:
                gu_a=[];gu_b=[];dn_a=[];dn_b=[];qb=[];qa=[]
                for (B,A,coef) in drawn:
                    gu_b.append(B[(lid,"moe.gate_up.lora_B")]*coef)
                    dn_b.append(B[(lid,"moe.down.lora_B")]*coef)
                    gu_a.append(A[(lid,"moe.gate_up.lora_A")]); dn_a.append(A[(lid,"moe.down.lora_A")])
                    if lid%n_attn_every==0:
                        qb.append(B[(lid,"qkv.lora_B")]*coef); qa.append(A[(lid,"qkv.lora_A")])
                cr=r*len(ch)
                segs=fold_moe_segs(cat_a(gu_a,r),torch.cat(gu_b,-1),cat_a(dn_a,r),torch.cat(dn_b,-1),cr)
                for k,v in segs.items(): macc[lid][k]=v if k not in macc[lid] else macc[lid][k]+v
                if lid%n_attn_every==0:
                    d=torch.cat(qb,-1)@torch.cat(qa,0)
                    aacc[lid]["qkv"]=d if "qkv" not in aacc[lid] else aacc[lid]["qkv"]+d
        for lid in blayers:
            for k,v in macc[lid].items(): G[(lid,"moe",k)]=v
            for k,v in aacc[lid].items(): G[(lid,"qkv","d")]=v
    return G

Gm=module_major()
for band in (1,2,3,4,6,8,nL):  # 4 = the production default; nL = band=-1 (all layers)
    Gb=band_major(band)
    assert set(Gm)==set(Gb), f"key mismatch band={band}"
    md=max((Gm[k]-Gb[k]).abs().max().item() for k in Gm)
    print(f"  band_size={band}: keys={len(Gb)} max|G_mod - G_band| = {md:.2e}")
    assert md==0.0, f"band={band} changed dense G"
print("PASS: band-major dense G is bit-exact vs module-major for all band sizes")
