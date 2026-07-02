"""xorl-trainer parameter-server (PS) client helpers for ZORL ES.

This is the client side of the PS-as-xorl-trainer pivot (2026-06-30, R1 PASS):
the parameter server is a `python -m xorl.server.launcher` process (NOT the
sglang-side re-impl). It is the seed authority, exports candidate LoRA adapters,
folds the reward-weighted ES update G=Σ c_i ΔW_i through the SERVER Muon
optimizer (R1-validated bit-for-bit == the sglang fold in fp32), and pushes the
parent to sglang serving replicas via the existing p2p RDMA weight-sync.

These helpers are ADDITIVE — they do not modify the existing sglang-PS path in
zorl_client.py (ps_apply_and_broadcast). Drive them from the PS-mode step loop
when --ps-backend=xorl. See PS_AS_XORL_TRAINER_DEPLOY_GUIDE.md.

Contract (xorl server HTTP API):
  - ZORL ops are ASYNC: POST /api/v1/zorl/{start_generation,apply_rewards,
    abort_generation} returns {"request_id": ...}; poll /api/v1/retrieve_future
    until type != "try_again".
  - Endpoint registration + weight sync are SYNC: POST /add_inference_endpoint,
    POST /sync_inference_weights.
  - accumulated_valid_tokens=0 is hardcoded server-side in apply_zorl_rewards
    (G is NOT token-normalized — correct for ES, R5).
"""

import time
from typing import Any, Dict, List, Optional

import requests


def _post_json(url: str, payload: dict, *, timeout: float = 120.0) -> dict:
    """POST JSON; raise on HTTP >= 400; return parsed JSON ({} for empty 200)."""
    with requests.post(url, json=payload, timeout=timeout) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"POST {url} -> HTTP {r.status_code}: {r.text[:500]}")
        if not r.content:
            return {}
        return r.json()


def wait_for_future(ps_url: str, request_id: str, *, timeout: float, poll_interval: float = 1.0) -> dict:
    """Poll /api/v1/retrieve_future until the async op resolves."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _post_json(f"{ps_url}/api/v1/retrieve_future", {"request_id": request_id}, timeout=120.0)
        if result.get("type") == "try_again":
            time.sleep(poll_interval)
            continue
        if result.get("type") == "error" or result.get("success") is False:
            raise RuntimeError(f"PS future {request_id} failed: {result.get('error_message') or result}")
        return result
    raise TimeoutError(f"PS future {request_id} timed out after {timeout}s")


def _call_future(ps_url: str, endpoint: str, payload: dict, *, context: str, future_timeout: float = 1800.0) -> dict:
    """Submit an async op and block on its future."""
    future = _post_json(f"{ps_url}{endpoint}", payload, timeout=120.0)
    request_id = future.get("request_id")
    if not request_id:
        raise RuntimeError(f"{context} did not return request_id: {future}")
    return wait_for_future(ps_url, str(request_id), timeout=future_timeout)


# ---------------------------------------------------------------------------
# One-time setup: create the ZORL LoRA-parent model with the Muon ES optimizer.
# ---------------------------------------------------------------------------
def create_zorl_model(
    ps_url: str,
    *,
    model_id: str,
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    muon_lr: float,
    b_sigma: float,
    num_perturbation_pairs: int,
    a_refresh_interval: int = 16,
    seed: int = 1234,
    perturbation_mode: str = "b_only",
    muon_momentum: float = 0.0,
    muon_adjust_lr_fn: str = "match_rms_adamw",
    muon_ns_algorithm: str = "gram_newton_schulz",
    muon_gram_ns_num_restarts: int = 0,
    muon_distributed_mode: str = "full_gradient",
    muon_ns_use_quack_kernels: bool = True,
    future_timeout: float = 1800.0,
) -> dict:
    """Create the ZORL LoRA-parent model on the xorl PS with the Muon ES optimizer.

    The optimizer_config.optimizer_kwargs MUST carry the full muon recipe —
    build_optimizer reads every muon_* from optimizer_kwargs and defaults
    muon_distributed_mode to 'shard_local' (the approximation), so full_gradient
    is set explicitly (R1 NS-sharding decision). perturbation_mode='b_only' is the
    parent-perturb probe (R1 Decision 1; the xorl-native, proven form).
    """
    payload = {
        "model_id": model_id,
        "base_model": base_model,
        "lora_config": {
            "rank": int(lora_rank),
            "lora_rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "lora_alpha": int(lora_alpha),
        },
        "optimizer_config": {
            "type": "muon",
            "learning_rate": float(muon_lr),
            "weight_decay": 0.0,
            "optimizer_dtype": "fp32",
            "optimizer_kwargs": {
                "muon_lr": float(muon_lr),
                "muon_momentum": float(muon_momentum),
                "muon_nesterov": False,
                "muon_ns_steps": 5,
                "muon_adjust_lr_fn": muon_adjust_lr_fn,
                "muon_ns_algorithm": muon_ns_algorithm,
                "muon_ns_use_quack_kernels": bool(muon_ns_use_quack_kernels),
                "muon_gram_ns_num_restarts": int(muon_gram_ns_num_restarts),
                "muon_distributed_mode": muon_distributed_mode,
            },
        },
        "zorl_config": {
            "enabled": True,
            "b_sigma": float(b_sigma),
            "num_perturbation_pairs": int(num_perturbation_pairs),
            "a_refresh_interval": int(a_refresh_interval),
            "antithetic_sampling": True,
            "a_init": "gaussian_jl",
            "seed": int(seed),
            "perturbation_mode": perturbation_mode,
        },
    }
    return _call_future(ps_url, "/api/v1/create_model", payload, context="create_model", future_timeout=future_timeout)


# ---------------------------------------------------------------------------
# One-time setup: register the sglang serving replicas as p2p sync receivers.
# ---------------------------------------------------------------------------
def register_inference_endpoint(
    ps_url: str,
    *,
    host: str,
    port: int,
    world_size: int = 1,
    pool: str = "default",
    master_address: Optional[str] = None,
    master_port: int = 0,
    group_name: str = "weight_sync_group",
    buffer_size_mb: int = 1024,
    sync_weights: bool = True,
) -> dict:
    """Register one sglang replica as a weight-sync receiver (POST /add_inference_endpoint).

    The replica must run with --enable-rdma-weight-updates (p2p) so it exposes
    /prepare_weights_update + /complete_weights_update. The PS reshards per
    receiver TP rank from its full unsharded tensor, so PS TP need not match.
    """
    payload: Dict[str, Any] = {
        "host": host,
        "port": int(port),
        "world_size": int(world_size),
        "pool": pool,
        "master_port": int(master_port),
        "group_name": group_name,
        "buffer_size_mb": int(buffer_size_mb),
        "sync_weights": bool(sync_weights),
    }
    if master_address is not None:
        payload["master_address"] = master_address
    return _post_json(f"{ps_url}/add_inference_endpoint", payload, timeout=300.0)


# ---------------------------------------------------------------------------
# Per-step: start generation -> (score on replicas) -> apply rewards.
# ---------------------------------------------------------------------------
def start_generation(
    ps_url: str,
    *,
    model_id: str = "default",
    num_pairs: Optional[int] = None,
    preload_sampling: bool = True,
    future_timeout: float = 1800.0,
) -> dict:
    """Plan one ES generation on the PS and export candidate adapters.

    Returns the generation response: {generation_id, family_id, b_sigma,
    num_pairs, candidates:[{candidate_id, perturbation_index, direction, b_seed,
    path, ...}], ...}. With preload_sampling=True the PS eagerly loads each
    candidate adapter on the registered inference endpoints (PS-authoritative
    adapter export — kills the R4 seed/mixer-parity problem: replicas score
    PS-authored adapters, never re-derive noise).
    """
    payload: Dict[str, Any] = {"model_id": model_id, "preload_sampling": bool(preload_sampling)}
    if num_pairs is not None:
        payload["num_pairs"] = int(num_pairs)
    return _call_future(
        ps_url, "/api/v1/zorl/start_generation", payload,
        context="start_zorl_generation", future_timeout=future_timeout,
    )


def apply_rewards(
    ps_url: str,
    *,
    model_id: str = "default",
    generation_id: str,
    candidate_rewards: List[Dict[str, Any]],
    learning_rate: Optional[float] = None,
    future_timeout: float = 1800.0,
) -> dict:
    """Fold the reward-weighted ES update on the PS (G -> param.grad=-G -> Muon optim_step).

    candidate_rewards: list of {candidate_id, reward_mean, num_rollouts?}.
    Momentum/NS/LR-scale are set by the server muon_* config (R1 recipe:
    momentum 0, match_rms_adamw, gram_ns no-restart, full_gradient). learning_rate
    overrides muon_lr for this step if given.
    """
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "generation_id": generation_id,
        "candidate_rewards": [_candidate_reward(c) for c in candidate_rewards],
    }
    if learning_rate is not None:
        payload["learning_rate"] = float(learning_rate)
    return _call_future(
        ps_url, "/api/v1/zorl/apply_rewards", payload,
        context="apply_zorl_rewards", future_timeout=future_timeout,
    )


def abort_generation(ps_url: str, *, model_id: str = "default", generation_id: str) -> dict:
    """Abort an in-flight generation (cleanup on scoring failure)."""
    return _call_future(
        ps_url, "/api/v1/zorl/abort_generation",
        {"model_id": model_id, "generation_id": generation_id},
        context="abort_zorl_generation",
    )


def _candidate_reward(c: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one candidate reward to the ZORLCandidateReward schema."""
    out: Dict[str, Any] = {"candidate_id": str(c["candidate_id"]), "reward_mean": float(c["reward_mean"])}
    if c.get("num_rollouts") is not None:
        out["num_rollouts"] = int(c["num_rollouts"])
    return out


# ---------------------------------------------------------------------------
# Optional: push the current parent to serving replicas (eval / parent-sync).
# Not on the adapter-export training critical path (candidates carry the parent),
# but needed to serve the trained parent for held-out floor-eval.
# ---------------------------------------------------------------------------
def sync_inference_weights(
    ps_url: str,
    *,
    model_id: str = "default",
    master_address: str,
    master_port: int = 0,
    group_name: str = "weight_sync_group",
    buffer_size_mb: int = 1024,
    flush_cache: bool = True,
    pools: Optional[List[str]] = None,
    future_timeout: float = 1800.0,
) -> dict:
    """Push the current parent (LoRA merged into base, resharded per replica TP) via p2p RDMA."""
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "master_address": master_address,
        "master_port": int(master_port),
        "group_name": group_name,
        "buffer_size_mb": int(buffer_size_mb),
        "flush_cache": bool(flush_cache),
    }
    if pools is not None:
        payload["pools"] = list(pools)
    # /sync_inference_weights is the async future variant; /api/v1 path is the same op.
    return _call_future(
        ps_url, "/sync_inference_weights", payload,
        context="sync_inference_weights", future_timeout=future_timeout,
    )
