#!/usr/bin/env bash
# Local base-model server for step-0 restatement evals (2026-07-02).
#
# WHY LOCAL: the assigned sampler pod er-opd-q36-megaonly-sglang-0 OOM-crashed at
# 2026-07-02T04:38Z (co-tenant process ate its slot GPU) and the whole megaonly stack
# (pods+sts+svc) was torn down at ~05:07Z, before this session started. Rather than mutate
# the cluster (out of scope), we serve the SAME base snapshot on this dev pod's own
# allocated GPU (physical index 3, UUID GPU-2b9aa365..., hostNetwork).
#
# Faithful to /shared/opd-control/er-opd-q36-megaonly/sglang-0/run.sh except:
#   - tp-size 1 (one GPU instead of the slot pod's 2); mem fraction raised 0.85->0.92
#     (66GB bf16 weights on one 80GB H100; hybrid linear-attention => tiny KV needs)
#   - max-total-tokens 65536, max-running-requests 48, cuda-graph-max-bs 32 (memory)
#   - weight-update / mooncake / expert-return flags dropped (no trainer; forward numerics
#     unaffected)
#   - port 30061 (30060 left free in case the stack is ever relaunched; dev pod is hostNetwork)
set -euo pipefail

SGLANG_REPO="/home/apanda/xorl-sglang-internal"
MODEL_PATH="/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SGLANG_PYTHON_BIN="${SGLANG_REPO}/.venv/bin/python"

export HF_HOME=/shared/huggingface
export TRANSFORMERS_CACHE=/shared/huggingface/hub
export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH:-}"
# k3 recon-parity env (same as the dead sampler: fp32 lm-head/router + native rmsnorm/rope)
export SGLANG_DISABLE_ROPE_COMPILE=1
export SGLANG_RMSNORM_FP32_WEIGHT_MUL=1
export SGLANG_RETURN_ORIGINAL_LOGPROB=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3   # this dev pod's allocated GPU (GPU-2b9aa365-...)
export TRITON_CACHE_DIR="/tmp/triton-cache-restate-eval-20260702"
mkdir -p "${TRITON_CACHE_DIR}"

exec "${SGLANG_PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${MODEL_PATH}" \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --host 127.0.0.1 \
  --port 30061 \
  --tp-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-fp32-lm-head \
  --enable-fp32-router \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-custom-all-reduce \
  --disable-priority-preemption \
  --mem-fraction-static 0.92 \
  --max-running-requests 48 \
  --max-total-tokens 65536 \
  --cuda-graph-max-bs 32 \
  --max-queued-requests 2048 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --sampling-backend flashinfer \
  --skip-server-warmup
