"""Warm-start the ZORL sglang pool from a trained DCP checkpoint (Route B).

One-shot driver: given a full-weight q36 training server that has ALREADY loaded a
GRPO/SFT DCP checkpoint (via the server config's ``load_checkpoint_path``), this
registers the ZORL pool's sglang replicas as inference endpoints and does a single
p2p weight broadcast so the pool's served *base* weights become the fine-tuned policy.
After this, launch ZORL ES against the pool as usual — ES folds its deltas into these
warm-started base weights.

Reuses the production GRPO weight-sync helpers (no new sync/quantize code): the server
auto-detects the pool's block-FP8 quantization config from its HF config.json and runs
the linear-attention FQN remap + MoE expert split + FP8 quant in-flight.

Mechanism + checkpoints documented in:
  experiments/zorl/ZORL_WORDLE_WARMSTART_RUNBOOK.md  (consolidated repo)

Example
-------
    # trainer head pod IP (the NCCL broadcast master the samplers must reach):
    TRAINER_IP=$(kubectl -n apanda get pod <trainer-head> -o jsonpath='{.status.podIP}')

    python warmstart_zorl_pool.py \
        --train-url http://127.0.0.1:26070 \
        --pool-host-template 'http://zorl-ar-sglang-w-{i}.zorl-ar-sglang-w-headless.apanda.svc.cluster.local:30000' \
        --pool-replicas 40 \
        --world-size 2 \
        --model-id default \
        --master-address "$TRAINER_IP" \
        --weight-version warmstart/policy-000000 \
        --output-dir /shared/apanda/wordle-sft-runs/warmstart_zorl
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse the production helpers (same dir).
from train_opsd_baseline import register_inference_endpoints, sync_weights_to_samplers


def build_pool_urls(template: str, replicas: int) -> list[str]:
    """Expand a '{i}'-templated headless-service hostname into per-replica URLs."""
    return [template.format(i=i) for i in range(replicas)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-url",
        required=True,
        help="Full-weight q36 training server URL (must have loaded the warm-start DCP "
        "ckpt via the server config's load_checkpoint_path).",
    )
    ap.add_argument(
        "--pool-host-template",
        required=True,
        help="Per-replica URL template with a '{i}' placeholder, e.g. "
        "'http://zorl-ar-sglang-w-{i}.zorl-ar-sglang-w-headless.apanda.svc.cluster.local:30000'.",
    )
    ap.add_argument("--pool-replicas", type=int, required=True, help="Number of pool replicas (e.g. 40).")
    ap.add_argument("--world-size", type=int, default=2, help="TP size per pool replica (pool is TP2).")
    ap.add_argument("--model-id", default="default", help="Training server model_id holding the warm policy.")
    ap.add_argument(
        "--master-address",
        default="",
        help="NCCL broadcast master = TRAINER head pod IP (samplers must reach it). "
        "If empty, sync_weights_to_samplers uses the driver pod's own IP — only correct "
        "if you run this ON the trainer head pod.",
    )
    ap.add_argument("--weight-version", default="warmstart/policy-000000")
    ap.add_argument("--buffer-size-mb", type=int, default=1024)
    ap.add_argument("--output-dir", required=True, help="Where to write sampler_exports.jsonl audit log.")
    args = ap.parse_args()

    pool_urls = build_pool_urls(args.pool_host_template, args.pool_replicas)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[warmstart] registering {len(pool_urls)} pool endpoints (world_size={args.world_size}) on {args.train_url}")
    register_inference_endpoints(args.train_url, pool_urls, world_size=args.world_size)

    print(f"[warmstart] broadcasting warm policy '{args.weight_version}' (master={args.master_address or 'pod-ip'})")
    body = sync_weights_to_samplers(
        train_url=args.train_url,
        model_id=args.model_id,
        weight_version=args.weight_version,
        output_dir=out,
        master_address=args.master_address,
        buffer_size_mb=args.buffer_size_mb,
        flush_cache=True,
    )
    results = body.get("results", body.get("message", body))
    print(f"[warmstart] DONE. sync response: {results}")
    print(
        "[warmstart] Pool base weights are now the warm-started policy. Next:\n"
        "  1) eval the pool on the floor-170 (probe) to confirm the warm score (Gate A),\n"
        "  2) tear down the trainer (the pool holds the weights; ES folds into the pool),\n"
        "  3) launch ZORL ES with WORDLE_TRAIN_EXCLUDE_SEED=777 COUNT=170 + think prompt."
    )


if __name__ == "__main__":
    main()
