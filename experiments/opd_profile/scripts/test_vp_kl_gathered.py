#!/usr/bin/env python3
"""Multi-process test for vocab_parallel_reverse_kl_gathered — the FSDP integration
glue (gather activations, shard weights, local-slice grad).

Each rank holds its OWN token slice [n_local,H] + its vocab shard [V/world,H] (the
real FSDP layout: the lm-head shard group also data-shards tokens). We check that
after gather→VP-KL→loss.sum()→backward:
  - this rank's local hidden grad == single-process full-vocab reference grad for
    its token slice (NO cross-rank double-count), and
  - this rank's weight-shard grad == reference grad for its vocab shard.

Run (no GPU needed):
  PYTHONPATH=/home/apanda/xorl-opd-kl-fused/src \
    /home/apanda/xorl-internal/.venv/bin/python test_vp_kl_gathered.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WORLD = 4
NLOCAL = 12           # tokens per rank
H = 96
VLOCAL = 130          # vocab rows per rank
N = WORLD * NLOCAL
V = WORLD * VLOCAL
IGNORE = -100


def _full_inputs():
    torch.manual_seed(7)
    sh = torch.randn(N, H, dtype=torch.float32) * 0.3
    th = torch.randn(N, H, dtype=torch.float32) * 0.3
    sw = torch.randn(V, H, dtype=torch.float32) * 0.05
    tw = torch.randn(V, H, dtype=torch.float32) * 0.05
    labels = torch.randint(0, V, (N,))
    labels[: N // 6] = IGNORE
    return sh, th, sw, tw, labels


def reference(sh, sw, th, tw, labels):
    sh = sh.clone().requires_grad_(True)
    sw = sw.clone().requires_grad_(True)
    s = sh @ sw.t(); t = th @ tw.t()
    slp = F.log_softmax(s, -1); tlp = F.log_softmax(t, -1)
    kl = (slp.exp() * (slp - tlp)).sum(-1) * (labels != IGNORE).float()
    kl.sum().backward()
    return sh.grad.detach(), sw.grad.detach()


def _worker(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from xorl.ops.loss.vocab_parallel_reverse_kl import vocab_parallel_reverse_kl_gathered

    sh, th, sw, tw, labels = _full_inputs()
    ts, te = rank * NLOCAL, (rank + 1) * NLOCAL
    vs, ve = rank * VLOCAL, (rank + 1) * VLOCAL

    local_sh = sh[ts:te].clone().requires_grad_(True)
    local_th = th[ts:te].clone()
    sw_local = sw[vs:ve].clone().requires_grad_(True)
    tw_local = tw[vs:ve].clone()

    kl = vocab_parallel_reverse_kl_gathered(
        local_student_hidden=local_sh, student_weight_local=sw_local,
        local_teacher_hidden=local_th, teacher_weight_local=tw_local,
        labels_full=labels, ignore_index=IGNORE, group=None,
    )
    kl.sum().backward()

    if rank == 0:
        gh_ref, gw_ref = reference(sh, sw, th, tw, labels)
        ret["gh"] = (local_sh.grad - gh_ref[ts:te]).abs().max().item()
        ret["gh_scale"] = gh_ref[ts:te].abs().max().item()
        ret["gw"] = (sw_local.grad - gw_ref[vs:ve]).abs().max().item()
        ret["gw_scale"] = gw_ref[vs:ve].abs().max().item()
        ret["kl0"] = float(kl[0].item())
    dist.barrier()
    dist.destroy_process_group()


def main():
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_worker, args=(WORLD, ret), nprocs=WORLD, join=True)
    gh_rel = ret["gh"] / max(ret["gh_scale"], 1e-30)
    gw_rel = ret["gw"] / max(ret["gw_scale"], 1e-30)
    print(f"gather-activations VP-KL integration (world={WORLD}, N={N}, V={V}, H={H}):")
    print(f"  local hidden grad  max|abs|={ret['gh']:.3e}  rel={gh_rel:.3e}  (slice of full ref)")
    print(f"  weight shard grad  max|abs|={ret['gw']:.3e}  rel={gw_rel:.3e}")
    ok = gh_rel < 1e-4 and gw_rel < 1e-4
    print(f"  => {'PASS — local-slice grad matches ref, no double-count' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
