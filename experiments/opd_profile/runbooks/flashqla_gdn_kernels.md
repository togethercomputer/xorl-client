# Runbook: consuming the FlashQLA GDN backend from another branch

Audience: an agent/engineer on a branch **other** than `feature/flashqla-gdn-kernels`
(e.g. an OPD run on `codex/opd-mainline-run-*`) who wants the faster Gated Delta Rule
(GDN) linear-attention kernels.

**TL;DR:** FlashQLA is an *opt-in, drop-in* replacement for the FLA-Triton GDN chunk
kernel, selected with `XORL_GDN_BACKEND=flashqla`. It is **Hopper (SM90) only** and needs
a specific `tilelang` wheel. It **silently falls back to FLA whenever Ulysses context
parallelism is on** (`ulysses_parallel_size > 1`) — which is the default for the 35B-A3B
OPD config, so read the "Will it actually engage?" section before expecting a speedup.

---

## 0. Does your model even use GDN?

Only models whose linear-attention layers are `GatedDeltaNet`
(`xorl.ops.linear_attention.layers.gated_deltanet`) are affected — the Qwen3.5 / Qwen3.6
"A3B" linear-attention family (e.g. the OPD teacher/student **Qwen3.6-35B-A3B**). Dense
attention models get nothing from this.

GDN is only a *subset* of layers in those models, so **end-to-end step speedup is smaller
than the kernel speedup** (the kernel is fwd 4.5–5.6× / bwd 1.3–2.2× vs FLA on the 35B-A3B
shape, but it's not the whole step).

---

## 1. Get the code onto your branch

The integration is a handful of files. Easiest is to merge the branch (or, once
PR #339 lands on `main`, just rebase your branch on `main`):

```bash
cd ~/xorl-opd-mainline-run            # your worktree
git fetch origin
# Option A — before PR #339 merges: merge the feature branch
git merge origin/feature/flashqla-gdn-kernels
# Option B — after PR #339 merges to main: rebase/merge main as usual
git merge origin/main
```

What it brings in (if you cherry-pick instead of merge, these are the files):

| file | what |
| --- | --- |
| `src/xorl/ops/linear_attention/flashqla/` | vendored FlashQLA TileLang kernels |
| `src/xorl/ops/linear_attention/tilelang_gemm_v1.py` | in-repo `gemm_v1` shim (monkeypatches stock tilelang) |
| `src/xorl/ops/linear_attention/backend.py` | `XORL_GDN_BACKEND` resolver + lazy import |
| `src/xorl/ops/linear_attention/layers/gated_deltanet.py` | routes the chunk kernel to FlashQLA (a *modify* — resolve conflicts against your branch's copy) |
| `pyproject.toml` | the `tilelang` wheel pin + `apache-tvm-ffi` |

The vendored `flashqla/` tree is `T.gemm`-additive: it does **not** touch the `tilelang`
paths GLM-5 / DSv4 use, so merging it is safe for those models.

---

## 2. Install the tilelang dependency

FlashQLA needs **both** the `tl_gemm` builtin (stock tilelang ≥0.1.10) **and**
`prefer_instruction="tma"` (tile-ai/tilelang **PR #2303**, merged upstream but *post-`v0.1.10`*,
so **not** in the PyPI `tilelang==0.1.10` wheel). We therefore pin a prebuilt wheel of stock
upstream `tile-ai/tilelang@a8d93798` (includes #2303; no source fork) hosted on
`togethercomputer/xorl-wheels`.

If you merged `pyproject.toml` (step 1), just sync:

```bash
uv sync
```

If you are NOT taking the pyproject change, add the dep manually:

```bash
uv pip install \
  "tilelang @ https://github.com/togethercomputer/xorl-wheels/releases/download/tilelang_0.1.10_cu131/tilelang-0.1.10%2Bcu131.gita8d93798-cp38-abi3-linux_x86_64.whl" \
  "apache-tvm-ffi>=0.1.10"
```

Notes:
- The wheel is `cp38-abi3` → installs on the repo's Python 3.12. CUDA 13.1 build, runs against
  the cu129 torch / cu131 H100 runtimes.
- `z3-solver` is pulled transitively (tilelang needs `libz3.so.4.15`). If you hit
  `libz3.so.4.15: cannot open shared object file`, confirm `z3-solver` installed; in a
  uv-managed env it resolves automatically (no `LD_LIBRARY_PATH` hack needed).
- Switch back to PyPI `tilelang>=0.1.11` once upstream cuts a release carrying #2303.

---

## 3. Enable it

Set the env var wherever the trainer process is launched (k8s Job env, launch script, etc.):

```bash
export XORL_GDN_BACKEND=flashqla     # default is "fla"
```

In a k8s manifest, add it to the container `env:`. Nothing in the YAML config changes —
selection is purely the env var. No GPU? Wrong arch? Missing wheel? → the lazy import raises
a clear `RuntimeError` the first time a GDN chunk runs (it does **not** silently mis-run).

---

## 4. Will it actually engage? (the important part)

FlashQLA runs only when **all** of these hold (see `gated_deltanet.py:238–244`):

1. SM90 (Hopper) GPU,
2. `XORL_GDN_BACKEND=flashqla`,
3. the GDN layer is in **`chunk`** mode (training forward/backward — true for OPD/SFT;
   decode uses `fused_recurrent`, always FLA),
4. **`cp_context is None`** — i.e. **no Ulysses context parallelism on the GDN layers.**

> ⚠️ **OPD gotcha:** the production Qwen3.6-35B-A3B config sets `ulysses_parallel_size: 8`,
> which makes `cp_context` non-`None`, so condition (4) fails and the layer **falls back to
> FLA** — FlashQLA is bypassed and you get **0 speedup**. You'll see this warning **once**:
>
> ```
> XORL_GDN_BACKEND=flashqla was requested, but an Ulysses cp_context is active;
> FlashQLA does not accept an external CP context, so the FLA Triton GDN kernel is
> used for these layers instead.
> ```

To actually benefit you must either:
- run with **`ulysses_parallel_size: 1`** (you lose cross-GPU sequence parallelism for the
  GDN layers — only viable if the sequence fits without Ulysses), **or**
- compose FlashQLA's own intra-card auto-CP with your distributed layout (a real code change
  in `gated_deltanet.py` — not wired up yet).

Also: q/k are repeated `num_heads → num_v_heads` *before* the kernel (`gated_deltanet.py:226`),
so FlashQLA runs its 32/32 (worst) regime. A GQA-native path (skip the repeat, feed 16/32)
is the remaining headroom but needs a layer change + a GQA parity check.

---

## 5. Verify (don't trust, check)

```bash
# Parity vs FLA on an H100 (fwd cos>0.99, bwd grad cos>0.97). ~3 min incl. JIT compile.
CUDA_VISIBLE_DEVICES=0 pytest tests/ops/test_flashqla_gdn.py -v

# Kernel speedup at your shape (FLA repeated 32/32 vs FlashQLA native):
CUDA_VISIBLE_DEVICES=0 python -m scripts.bench_flashqla_gdn
```

In a real run, confirm engagement by the **absence** of the cp-fallback warning above, and
(if profiling) that the GDN chunk time dropped. If you see the warning, you're on FLA.

---

## 6. One-liner summary for a launch script

```bash
# requires: SM90 GPU, tilelang #2303 wheel installed, ulysses_parallel_size==1 for GDN layers
export XORL_GDN_BACKEND=flashqla
```
