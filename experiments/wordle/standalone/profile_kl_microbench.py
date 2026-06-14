"""Single-GPU microbenchmark: how expensive is the full-vocab reverse-KL?

Times streaming_reverse_kl (fwd+bwd) on representative per-rank shapes and
compares to the bare lm_head matmul FLOP floor. If KL >> matmul, fb is
KL/memory-bound (not compute) — explaining the ~0.13% MFU.
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get("XORL_SRC_PATH", "/workspace/home/xorl-internal/src"))
import torch  # noqa: E402
from xorl.ops.loss.opd_streaming_kl import streaming_reverse_kl_function  # noqa: E402

dev = "cuda"
dt = torch.bfloat16
H = 2048
V = 248320
print(f"device={torch.cuda.get_device_name()} H={H} V={V} dtype={dt}", flush=True)


def _mk(N):
    sh = torch.randn(N, H, device=dev, dtype=dt, requires_grad=True)
    sw = torch.randn(V, H, device=dev, dtype=dt, requires_grad=True)
    th = torch.randn(N, H, device=dev, dtype=dt)
    tw = torch.randn(V, H, device=dev, dtype=dt)
    lab = torch.randint(0, V, (N,), device=dev)
    return sh, sw, th, tw, lab


def bench_streaming(N, iters=5, vchunk=32768):
    sh, sw, th, tw, lab = _mk(N)

    def run():
        kl = streaming_reverse_kl_function(sh, sw, th, tw, lab, vocab_chunk_size=vchunk)
        kl.sum().backward()
        sh.grad = None
        sw.grad = None

    run()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def bench_matmul(N, iters=5):
    # bare lm_head projection fwd+bwd (CE-style logsumexp) = "useful FLOP" floor
    sh = torch.randn(N, H, device=dev, dtype=dt, requires_grad=True)
    sw = torch.randn(V, H, device=dev, dtype=dt, requires_grad=True)

    def run():
        logits = sh @ sw.t()
        logits.float().logsumexp(-1).sum().backward()
        sh.grad = None
        sw.grad = None

    run()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


for N in [1600, 6400, 12800]:
    s = bench_streaming(N)
    try:
        m = bench_matmul(N)
        mtxt = f"{1000*m:7.1f}ms  KL/matmul={s/m:5.1f}x"
    except RuntimeError as e:
        mtxt = f"matmul OOM/err ({str(e)[:40]})"
    print(f"N={N:6d}: streaming_KL fwd+bwd {1000*s:8.1f}ms ({1000*s/N*1000:6.2f} ms/1k-tok) | bare lm_head {mtxt}", flush=True)
print("DONE", flush=True)
