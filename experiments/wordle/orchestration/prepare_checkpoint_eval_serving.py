#!/usr/bin/env python
"""Prepare private serving for floor-evaluating a saved retrieval-SFT checkpoint.

This implements Gate 2 route (b) from GATE_RUNBOOK.md:
  1. start a private TP=2 SGLang endpoint, not the shared opsd-wordle-q36 samplers;
  2. start a short-lived XORL server with load_checkpoint_path=<DCP checkpoint>;
  3. register the private endpoint and sync the checkpoint weights into it;
  4. run eval_holdout_vs_trained.sh against the private endpoint URL.

By default this only writes manifests under /shared/apanda/wordle-sft-runs. Pass
--apply only when Kubernetes auth is available and the checkpoint path is real.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

RESULT_ROOT = Path("/shared/apanda/wordle-sft-runs")
PROMOTED = Path("/shared/apanda/wordle-coord/PROMOTED.json")
MESSAGES = Path("/shared/apanda/wordle-coord/messages.jsonl")
BASE_SGLANG = Path(
    "/home/apanda/xorl-infra-opd-wordle-pr1-20260614/k8s/zorl/"
    "qwen3-6-35b-a3b-opsd-wordle-sglang-pool.yaml"
)


def slugify(text: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug or "ckpt")[:max_len].strip("-")


def host_to_pod_path(path: str) -> str:
    if path.startswith("/home/apanda/"):
        return "/workspace/home/" + path[len("/home/apanda/") :]
    return path


def append_coord(kind: str, text: str) -> None:
    msg = {
        "ts": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "science",
        "to": "throughput",
        "kind": kind,
        "text": text,
    }
    with MESSAGES.open("a") as f:
        f.write(json.dumps(msg) + "\n")


def set_env(env: list[dict[str, Any]], key: str, value: str) -> None:
    for item in env:
        if item.get("name") == key:
            item["value"] = value
            item.pop("valueFrom", None)
            return
    env.append({"name": key, "value": value})


def relabel(obj: dict[str, Any], app: str) -> None:
    metadata = obj.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    labels["app"] = app
    labels["experiment-variant"] = f"{app}-private-eval"


def make_private_sglang(base_docs: list[dict[str, Any]], *, app: str, replicas: int) -> tuple[list[dict[str, Any]], list[str]]:
    service_name = f"{app}-headless"
    docs = []
    for raw in base_docs:
        if not raw:
            continue
        obj = json.loads(json.dumps(raw))
        kind = obj.get("kind")
        relabel(obj, app)
        if kind == "Service":
            obj["metadata"]["name"] = service_name
            obj["spec"]["selector"]["app"] = app
        elif kind == "StatefulSet":
            obj["metadata"]["name"] = app
            obj["spec"]["serviceName"] = service_name
            obj["spec"]["replicas"] = replicas
            obj["spec"]["selector"]["matchLabels"]["app"] = app
            tmpl = obj["spec"]["template"]
            tmpl["metadata"]["labels"]["app"] = app
            tmpl["metadata"]["labels"]["experiment-variant"] = f"{app}-private-eval"
            containers = tmpl["spec"]["containers"]
            sglang = next(c for c in containers if c.get("name") == "sglang")
            env = sglang.setdefault("env", [])
            set_env(env, "SGLANG_MEM_FRACTION_STATIC", "0.85")   # more KV (multi-turn think eval is KV-bound)
            set_env(env, "SGLANG_MAX_TOTAL_TOKENS", "131072")    # ~14.7k-token contexts -> fit ~8 concurrent (was 16384 -> only ~2)
            set_env(env, "SGLANG_MAX_RUNNING_REQUESTS", "32")
            set_env(env, "SGLANG_MAX_QUEUED_REQUESTS", "256")
            command = sglang.get("command")
            if isinstance(command, list) and command:
                script = command[-1]
                if isinstance(script, str) and "--skip-server-warmup" not in script:
                    command[-1] = script.replace(
                        "  --trust-remote-code\n",
                        "  --trust-remote-code \\\n  --skip-server-warmup\n",
                    )
        docs.append(obj)
    endpoints = [
        f"http://{app}-{i}.{service_name}.apanda.svc.cluster.local:30000"
        for i in range(replicas)
    ]
    return docs, endpoints


def infer_gpu_count(config: dict[str, Any]) -> int:
    keys = (
        "data_parallel_shard_size",
        "expert_parallel_size",
        "tensor_parallel_size",
    )
    vals = [int(config.get(k, 1) or 1) for k in keys]
    return max(vals)


def make_sync_job(
    *,
    app: str,
    serve_dir_pod: str,
    config_path_pod: str,
    endpoints: list[str],
    gpu_count: int,
    label: str,
) -> dict[str, Any]:
    endpoint_text = " ".join(endpoints)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": f"{app}-sync-",
            "labels": {
                "app": f"{app}-sync",
                "team": "turbo",
                "workload": "eval",
                "experiment-group": "opsd-wordle",
                "experiment-variant": f"{app}-checkpoint-sync",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 7200,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {
                        "app": f"{app}-sync",
                        "team": "turbo",
                        "workload": "eval",
                        "experiment-group": "opsd-wordle",
                        "experiment-variant": f"{app}-checkpoint-sync",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": "normal",
                    "runtimeClassName": "nvidia",
                    "hostIPC": True,
                    "hostNetwork": False,
                    "dnsPolicy": "ClusterFirst",
                    "nodeSelector": {"node-group": "default", "node-pool": "compute"},  # was nccl (small full pool -> Pending); tolerations cover both
                    "tolerations": [
                        {"key": "node-group", "operator": "Equal", "value": "default", "effect": "NoSchedule"},
                        {"key": "node-group", "operator": "Equal", "value": "nccl", "effect": "NoSchedule"},
                    ],
                    "volumes": [
                        {"name": "shared", "persistentVolumeClaim": {"claimName": "shared-data"}},
                        {"name": "workspace-home", "persistentVolumeClaim": {"claimName": "home-apanda"}},
                        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
                    ],
                    "containers": [
                        {
                            "name": "sync-loader",
                            "image": "nvcr.io/nvidia/pytorch:26.02-py3",
                            "workingDir": "/workspace/home/xorl-client-wordle-science-20260614",
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"add": ["IPC_LOCK"]},
                            },
                            "env": [
                                {"name": "HOME", "value": "/workspace/home"},
                                {"name": "HF_HOME", "value": "/shared/huggingface"},
                                {"name": "HF_HUB_DISABLE_XET", "value": "1"},
                                {"name": "HF_HUB_OFFLINE", "value": "1"},
                                {"name": "TRANSFORMERS_CACHE", "value": "/shared/huggingface/transformers"},
                                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                                {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "XORL_P2P_HANDSHAKE_BASE_PORT", "value": "18000"},
                                {"name": "NCCL_IB_DISABLE", "value": "0"},
                                {"name": "NCCL_NET_GDR_LEVEL", "value": "2"},
                                {"name": "NCCL_SOCKET_IFNAME", "value": "^lo,docker"},
                                {"name": "NCCL_DEBUG", "value": "WARN"},
                                {"name": "P2P_TRAINER_HOSTNAME", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
                            ],
                            "volumeMounts": [
                                {"name": "shared", "mountPath": "/shared"},
                                {"name": "workspace-home", "mountPath": "/workspace/home"},
                                {"name": "dshm", "mountPath": "/dev/shm"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "32",
                                    "memory": "384Gi",
                                    "nvidia.com/gpu": str(gpu_count),
                                },
                                "limits": {
                                    "cpu": "48",
                                    "memory": "512Gi",
                                    "nvidia.com/gpu": str(gpu_count),
                                    "rdma/infiniband": "1",
                                },
                            },
                            "command": [
                                "/bin/bash",
                                "-lc",
                                f"""
set -euo pipefail
RUN_DIR="{serve_dir_pod}"
CONFIG_PATH="{config_path_pod}"
SGLANG_ENDPOINTS="{endpoint_text}"
SAMPLER_WORLD_SIZE="2"
SYNC_WEIGHT_VERSION="{label}"
XORL_PYTHON="/workspace/home/xorl-internal/.venv/bin/python"
API_PORT="26070"
WORKER_PORT="25670"
MASTER_PORT="29670"
mkdir -p "${{RUN_DIR}}"
export PYTHONPATH="/workspace/home/xorl-opsd-wordle-apanda-dev-run-20260607/src:/workspace/home/xorl-client-wordle-science-20260614/src:/workspace/home/xorl-client-wordle-science-20260614${{PYTHONPATH:+:${{PYTHONPATH}}}}"
export PATH="/workspace/home/xorl-internal/.venv/bin:${{PATH}}"
# Match the promoted training manifest: expandable_segments avoids the marginal
# fp8-off 4-GPU OOM during Muon optimizer-state init at engine load.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
exec > >(tee -a "${{RUN_DIR}}/sync_job.log") 2>&1

echo "run_dir=${{RUN_DIR}}"
echo "config_path=${{CONFIG_PATH}}"
echo "sglang_endpoints=${{SGLANG_ENDPOINTS}}"
echo "sync_weight_version=${{SYNC_WEIGHT_VERSION}}"
MASTER_ADDRESS="$(hostname -i | awk '{{print $1}}')"
echo "master_address=${{MASTER_ADDRESS}}"

"${{XORL_PYTHON}}" -u -m xorl.server.launcher \\
  --mode auto \\
  --config "${{CONFIG_PATH}}" \\
  --api-port "${{API_PORT}}" \\
  --worker-address "tcp://0.0.0.0:${{WORKER_PORT}}" \\
  --master-port "${{MASTER_PORT}}" \\
  --operation-timeout 7200 \\
  --log-level INFO \\
  --server.output_dir "${{RUN_DIR}}/server_output" \\
  --server.worker_bind_address "tcp://0.0.0.0:${{WORKER_PORT}}" \\
  --server.idle_session_timeout 3600 \\
  > "${{RUN_DIR}}/xorl_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {{
  if kill -0 "${{SERVER_PID}}" >/dev/null 2>&1; then
    kill "${{SERVER_PID}}" || true
    wait "${{SERVER_PID}}" || true
  fi
}}
trap cleanup EXIT

"${{XORL_PYTHON}}" - "${{RUN_DIR}}" "${{API_PORT}}" "${{SAMPLER_WORLD_SIZE}}" "${{MASTER_ADDRESS}}" "${{SYNC_WEIGHT_VERSION}}" ${{SGLANG_ENDPOINTS}} <<'PY'
import json
import sys
import time
from urllib.parse import urlparse

import requests

run_dir, api_port, world_size, master_address, weight_version, *endpoints = sys.argv[1:]
train_url = f"http://127.0.0.1:{{api_port}}"

def wait_http(name, url, timeout=2400):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                print(f"{{name}} ready via {{url}}", flush=True)
                return
            last = f"HTTP {{r.status_code}} {{r.text[:120]}}"
        except Exception as exc:
            last = f"{{type(exc).__name__}}: {{exc}}"
        print(f"waiting for {{name}}: {{last}}", flush=True)
        time.sleep(10)
    raise SystemExit(f"timed out waiting for {{name}}: {{last}}")

wait_http("xorl server", train_url + "/health")
for endpoint in endpoints:
    wait_http(endpoint, endpoint.rstrip("/") + "/v1/models", timeout=2400)

registered = []
for endpoint in endpoints:
    parsed = urlparse(endpoint)
    payload = {{"host": parsed.hostname, "port": int(parsed.port or 30000), "world_size": int(world_size)}}
    r = requests.post(train_url + "/add_inference_endpoint", json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()
    print("registered", endpoint, body, flush=True)
    if not body.get("success", False) and "already" not in str(body).lower():
        raise SystemExit(f"add_inference_endpoint failed for {{endpoint}}: {{body}}")
    registered.append(body)

sync_payload = {{
    "model_id": "default",
    "weight_version": weight_version,
    "master_address": master_address,
    "buffer_size_mb": 1024,
    "pause_mode": "retract",
    "flush_cache": True,
}}
r = requests.post(train_url + "/api/v1/sync_inference_weights", json=sync_payload, timeout=7200)
r.raise_for_status()
sync = r.json()
print("sync_result", json.dumps(sync, sort_keys=True), flush=True)
if not sync.get("success", True):
    raise SystemExit(f"sync_inference_weights failed: {{sync}}")

out = {{
    "time": time.time(),
    "train_url": train_url,
    "endpoints": endpoints,
    "registered": registered,
    "sync_payload": sync_payload,
    "sync_result": sync,
}}
with open(f"{{run_dir}}/sync_result.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
PY

echo "Checkpoint weights are loaded into: ${{SGLANG_ENDPOINTS}}"
""".strip(),
                            ],
                        }
                    ],
                },
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_path", help="DCP checkpoint directory, e.g. <RUN_DIR>/server_output/weights/default/step-000080")
    ap.add_argument("--label", default="", help="short label, e.g. algosft-step080")
    ap.add_argument("--config-path", default="", help="trainer config used to load the checkpoint; default PROMOTED.json.config_path")
    ap.add_argument("--replicas", type=int, default=1, help="private SGLang TP=2 replicas; 1 is enough for floor eval")
    ap.add_argument("--apply", action="store_true", help="kubectl apply/create the generated manifests")
    args = ap.parse_args()

    checkpoint_host = Path(args.checkpoint_path)
    if not checkpoint_host.exists():
        raise SystemExit(f"checkpoint_path does not exist: {checkpoint_host}")
    if not (checkpoint_host / ".metadata").exists():
        print(f"WARN: {checkpoint_host}/.metadata not found; continuing because DCP layouts can vary", file=sys.stderr)

    promoted = json.loads(PROMOTED.read_text()) if PROMOTED.exists() else {}
    config_path = args.config_path or str(promoted.get("config_path", ""))
    if not config_path:
        raise SystemExit("--config-path is required when PROMOTED.json has no config_path")
    config_host = Path(config_path.replace("/workspace/home/", "/home/apanda/", 1)) if config_path.startswith("/workspace/home/") else Path(config_path)
    if not config_host.exists():
        raise SystemExit(f"config_path does not exist from host: {config_host} (raw {config_path})")

    label = slugify(args.label or checkpoint_host.name, max_len=28)
    app = slugify(f"wordle-sft-eval-{label}", max_len=45)
    serve_dir = RESULT_ROOT / "eval-serving" / label
    serve_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_host.read_text())
    config["load_checkpoint_path"] = host_to_pod_path(str(checkpoint_host))
    # Eval serving only needs model weights. Restoring Muon optimizer state can
    # OOM in the EP4 loader before weights are synced into the private SGLang.
    config["load_checkpoint_optimizer"] = False
    config["output_dir"] = f"/shared/apanda/wordle-sft-runs/eval-serving/{label}/xorl_config_output"
    config["skip_initial_checkpoint"] = True
    # 2026-06-30 (apanda): strip engine_connect_host. The GRPO TRAINING configs
    # (e.g. grpo-ep8x1node-muon-lowlr-isr3k3.yaml) set `engine_connect_host: 127.0.0.1`
    # for multi-rank p2p bring-up. In the EVAL launcher that key forces the orchestrator
    # to connect to tcp://127.0.0.1:5556 while the worker binds tcp://0.0.0.0:25670 →
    # permanent port mismatch ("engine-0 not routable" / ZMQError Host unreachable, never
    # recovers; broke every eval since ~06-26). The eval launcher manages worker addressing
    # itself via --worker_bind_address=25670, so this key must not be present.
    config.pop("engine_connect_host", None)
    patched_config = serve_dir / "checkpoint_load_config.yaml"
    patched_config.write_text(yaml.safe_dump(config, sort_keys=False))
    gpu_count = infer_gpu_count(config)

    base_docs = list(yaml.safe_load_all(BASE_SGLANG.read_text()))
    sglang_docs, endpoints = make_private_sglang(base_docs, app=app, replicas=args.replicas)
    sglang_manifest = serve_dir / "private_sglang.yaml"
    sglang_manifest.write_text(yaml.safe_dump_all(sglang_docs, sort_keys=False))

    sync_job = make_sync_job(
        app=app,
        serve_dir_pod=f"/shared/apanda/wordle-sft-runs/eval-serving/{label}",
        config_path_pod=host_to_pod_path(str(patched_config)),
        endpoints=endpoints,
        gpu_count=gpu_count,
        label=label,
    )
    sync_manifest = serve_dir / "sync_loader_job.yaml"
    sync_manifest.write_text(yaml.safe_dump(sync_job, sort_keys=False))

    meta = {
        "label": label,
        "checkpoint_path_host": str(checkpoint_host),
        "checkpoint_path_pod": host_to_pod_path(str(checkpoint_host)),
        "source_config_path": str(config_host),
        "patched_config": str(patched_config),
        "private_sglang_manifest": str(sglang_manifest),
        "sync_loader_job_manifest": str(sync_manifest),
        "private_eval_base_url": endpoints[0],
        "all_private_endpoints": endpoints,
        "gpu_count_for_sync_loader": gpu_count,
        "eval_command": (
            f"/shared/apanda/wordle-sft-runs/eval_holdout_vs_trained.sh "
            f"{endpoints[0]} {label}"
        ),
    }
    meta_path = serve_dir / "SERVE_META.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(json.dumps(meta, indent=2, sort_keys=True))

    if args.apply:
        try:
            subprocess.run(["kubectl", "apply", "-f", str(sglang_manifest)], check=True)
            subprocess.run(["kubectl", "create", "-f", str(sync_manifest)], check=True)
        except subprocess.CalledProcessError as exc:
            append_coord(
                "blocker",
                f"Private checkpoint eval serving apply failed for {label}: {type(exc).__name__} rc={exc.returncode}. "
                f"Artifacts remain in {serve_dir}.",
            )
            return int(exc.returncode or 1)
        append_coord(
            "status",
            f"Prepared/applied private checkpoint eval serving for {label}. "
            f"Endpoint after sync: {endpoints[0]}; eval command: {meta['eval_command']}. "
            f"Artifacts: {serve_dir}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
