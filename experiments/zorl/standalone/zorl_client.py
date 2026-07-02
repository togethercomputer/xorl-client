"""Standalone SGLang-native ZORL client — no xorl trainer required.

Talks only to SGLang's ZORL-native endpoints. All ZORL state (parent LoRA,
perturbation seeds, ES update, snapshots) lives on SGLang. This script just
orchestrates: load an init adapter, run the gen loop, score rollouts, apply
rewards, probe + rollback.

Task-agnostic: the puzzle/reward logic lives under ``tasks/`` (countdown,
gsm8k, alphabet_sort, wordle). Pick one via ``--task NAME``.

Hyperparam guidance (from the May-2026 ES-on-Prime-RL blog the user shared):
  * sigma=0.012 + lr=0.01 work well across GSM8K, Alphabet Sort, Wordle.
  * Population size 64 (=32 num_pairs) is enough; bigger doesn't seem to help.
  * Our previous Countdown runs used sigma=0.1 (8x too high) and that's
    plausibly why we hit the 5/32 plateau.

Usage:

  python zorl_client.py \\
      --task gsm8k \\
      --infer-url http://sglang-0:30060 \\
      --adapter-dir /shared/zorl/init-adapters/qwen3-0.6b-r16 \\
      --model /shared/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/<sha> \\
      --parent-lora-name zorl-native-parent/gsm8k/$(date +%s) \\
      --session-id gsm8k-$(date +%s) \\
      --steps 50 --num-pairs 32 --b-sigma 0.012 --lr 0.01 \\
      --train-size 8 --eval-size 128 \\
      --rollout-temperature 0.6 --rollout-max-new-tokens 2048

  Pre-req: the adapter dir must already exist (see init_zorl_adapter.py).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

# Keep-alive connection pool (DEFAULT ON; set ZORL_HTTP_KEEPALIVE=0 for the legacy
# per-request `Connection: close`, which opens a fresh TCP connection for EVERY
# rollout turn — up to 6 per game x hundreds of games — paying a handshake each
# time and blocking HTTP keep-alive). A shared, thread-safe Session with a large
# pool reuses connections (the faster path, so it is the default).
# pool_maxsize must cover per-host concurrency (~SCORE_MAX_WORKERS_PER_OWNER)
# or connections serialize. Default off so it's a clean A/B and reversible.
_KEEPALIVE_SESSION = None
_KEEPALIVE_LOCK = threading.Lock()


def _keepalive_session() -> requests.Session:
    global _KEEPALIVE_SESSION
    if _KEEPALIVE_SESSION is None:
        with _KEEPALIVE_LOCK:
            if _KEEPALIVE_SESSION is None:
                pool = int(os.environ.get("ZORL_HTTP_POOL_MAXSIZE", "512"))
                s = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=pool, pool_maxsize=pool, max_retries=0
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _KEEPALIVE_SESSION = s
    return _KEEPALIVE_SESSION


# Make the tasks/ package importable regardless of cwd. Adding the script's
# own directory to sys.path lets ``python /path/to/zorl_client.py`` work from
# anywhere (e.g. from the dispatch pod).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks.base import Example, load_task  # noqa: E402  (must come after sys.path tweak)


# ============================================================================
# SGLang HTTP wrappers
# ============================================================================


def _post(url: str, path: str, payload: dict, *, timeout: float = 300.0, headers: dict | None = None) -> dict:
    """POST helper with retry on transient connection failures. Returns parsed
    JSON, or an empty dict for endpoints (like /flush_cache) that respond with
    an empty 200. Raises RuntimeError with the server's error body on HTTP >=400."""
    last_err = None
    # Default ON: reuse a pooled keep-alive session (faster — avoids a fresh TCP
    # handshake per request across the 40-replica fan-out). Set
    # ZORL_HTTP_KEEPALIVE=0 to force per-request Connection: close (legacy).
    keepalive = os.environ.get("ZORL_HTTP_KEEPALIVE", "1") != "0"
    request_headers = {} if keepalive else {"Connection": "close"}
    if headers:
        request_headers.update(headers)
    poster = _keepalive_session().post if keepalive else requests.post
    max_attempts = 10
    retryable_statuses = {429, 503}
    for attempt in range(max_attempts):
        try:
            with poster(f"{url}{path}", json=payload, headers=request_headers, timeout=timeout) as r:
                if r.status_code >= 400:
                    last_err = RuntimeError(f"{path} on {url} → HTTP {r.status_code}: {r.text[:500]}")
                    if r.status_code in retryable_statuses and attempt + 1 < max_attempts:
                        delay = min(30.0, 0.5 * (2 ** attempt)) * (0.75 + random.random() * 0.5)
                        time.sleep(delay)
                        continue
                    raise last_err
                if not r.content:
                    return {}
                try:
                    return r.json()
                except requests.exceptions.JSONDecodeError:
                    # /flush_cache returns plain text "OK".
                    return {"raw": r.text}
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt + 1 < max_attempts:
                delay = min(30.0, 0.5 * (2 ** attempt)) * (0.75 + random.random() * 0.5)
                time.sleep(delay)
    raise RuntimeError(f"{path} on {url} failed after retries: {last_err}")


def _url_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    else:
        raw = list(value)
    urls: list[str] = []
    for item in raw:
        urls.extend(part for part in re.split(r"[\s,]+", str(item).strip()) if part)
    return urls


def _post_all(urls: list[str], path: str, payload: dict, *, timeout: float = 300.0) -> list[dict]:
    if not urls:
        raise RuntimeError(f"{path} needs at least one SGLang URL")
    if len(urls) == 1:
        return [_post(urls[0], path, payload, timeout=timeout)]

    def post_one(url: str) -> tuple[str, dict]:
        return url, _post(url, path, payload, timeout=timeout)

    results: list[dict | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {pool.submit(post_one, url): idx for idx, url in enumerate(urls)}
        for future in as_completed(futures, timeout=timeout + 10):
            idx = futures[future]
            url, result = future.result()
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"{path} failed on {url}: {result.get('error_message') or result}")
            results[idx] = result
    completed = [result for result in results if result is not None]
    if len(completed) != len(urls):
        raise RuntimeError(f"{path} completed on {len(completed)}/{len(urls)} URLs")
    return completed


def load_lora_adapter(url: str, lora_name: str, lora_path: str) -> None:
    _post(url, "/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path}, timeout=180.0)


def unload_lora_adapter(url: str, lora_name: str) -> None:
    try:
        _post(url, "/unload_lora_adapter", {"lora_name": lora_name}, timeout=60.0)
    except RuntimeError:
        pass  # best-effort cleanup


def start_zorl_session(url: str, *, session_id: str, parent_lora_name: str, num_pairs: int, b_sigma: float, seed: int, perturbation_mode: str = "b_only") -> dict:
    return _post(url, "/start_zorl_session", {
        "session_id": session_id,
        "parent_lora_name": parent_lora_name,
        "num_pairs": int(num_pairs),
        "b_sigma": float(b_sigma),
        "seed": int(seed),
        "antithetic_sampling": True,
        "perturbation_mode": perturbation_mode,
    })


def load_lora_adapter_all(urls: list[str], lora_name: str, lora_path: str) -> None:
    # Generous timeout: the control request queues behind any in-flight decode
    # backlog on a busy pool (e.g. right after a prior run), so 180s can time out.
    _post_all(urls, "/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path}, timeout=600.0)


def start_zorl_session_all(urls: list[str], *, session_id: str, parent_lora_name: str, num_pairs: int, b_sigma: float, seed: int, perturbation_mode: str = "b_only") -> dict:
    results = _post_all(urls, "/start_zorl_session", {
        "session_id": session_id,
        "parent_lora_name": parent_lora_name,
        "num_pairs": int(num_pairs),
        "b_sigma": float(b_sigma),
        "seed": int(seed),
        "antithetic_sampling": True,
        "perturbation_mode": perturbation_mode,
    })
    return results[0]


def start_zorl_generation(
    url: str,
    *,
    session_id: str,
    preload_candidates: bool = False,
    num_pairs: int | None = None,
    materialization: dict | None = None,
    owner_url: str | None = None,
) -> dict:
    payload = {"session_id": session_id, "preload_candidates": bool(preload_candidates)}
    if num_pairs is not None:
        payload["num_pairs"] = int(num_pairs)
    if materialization is not None:
        payload["materialization"] = materialization
    if owner_url is not None:
        payload["owner_url"] = owner_url
    return _post(url, "/start_zorl_generation", payload, timeout=300.0)


def _generation_sort_key(candidate: dict) -> tuple[int, int, str]:
    direction_rank = 0 if candidate.get("direction") == "positive" else 1
    return (int(candidate.get("perturbation_index", 0)), direction_rank, str(candidate.get("candidate_id", "")))


def _combine_sharded_generation_results(
    results_by_url: list[tuple[str, dict]],
    *,
    expected_num_pairs: int,
) -> dict:
    if not results_by_url:
        raise RuntimeError("sharded ZORL generation needs at least one shard response")

    first_url, first = results_by_url[0]
    generation_id = str(first.get("generation_id", ""))
    if not generation_id:
        raise RuntimeError(f"ZORL shard response from {first_url} did not include generation_id")

    candidates: list[dict] = []
    seen_candidate_ids: dict[str, str] = {}
    for owner_url, result in results_by_url:
        if str(result.get("generation_id", "")) != generation_id:
            raise RuntimeError(
                "Inconsistent ZORL generation_id across shards: "
                f"{generation_id!r} from {first_url}, {result.get('generation_id')!r} from {owner_url}"
            )
        for candidate in result.get("candidates", []):
            patched = dict(candidate)
            patched["owner_url"] = str(patched.get("owner_url") or owner_url)
            candidate_id = str(patched.get("candidate_id", ""))
            if candidate_id in seen_candidate_ids:
                raise RuntimeError(
                    f"Duplicate ZORL candidate {candidate_id!r} owned by both "
                    f"{seen_candidate_ids[candidate_id]} and {owner_url}"
                )
            seen_candidate_ids[candidate_id] = owner_url
            candidates.append(patched)

    expected_candidates = 2 * int(expected_num_pairs)
    if len(candidates) != expected_candidates:
        raise RuntimeError(
            f"Sharded ZORL generation {generation_id} returned {len(candidates)} candidates; "
            f"expected {expected_candidates}"
        )

    by_pair: dict[int, set[str]] = {}
    pair_owner: dict[int, str] = {}
    for candidate in candidates:
        pair_index = int(candidate["perturbation_index"])
        direction = str(candidate["direction"])
        by_pair.setdefault(pair_index, set()).add(direction)
        owner_url = str(candidate["owner_url"])
        previous_owner = pair_owner.setdefault(pair_index, owner_url)
        if previous_owner != owner_url:
            raise RuntimeError(
                f"ZORL perturbation pair {pair_index} split across owners {previous_owner} and {owner_url}"
            )

    missing_pairs = []
    for pair_index in range(int(expected_num_pairs)):
        if by_pair.get(pair_index) != {"positive", "negative"}:
            missing_pairs.append(pair_index)
    if missing_pairs:
        raise RuntimeError(
            f"Sharded ZORL generation {generation_id} has incomplete perturbation pairs: {missing_pairs[:8]}"
        )

    combined = dict(first)
    combined["candidates"] = sorted(candidates, key=_generation_sort_key)
    combined["global_num_pairs"] = int(first.get("global_num_pairs") or expected_num_pairs)
    combined["global_population"] = int(first.get("global_population") or len(candidates))
    combined["local_num_pairs"] = sum(int(result.get("local_num_pairs", 0)) for _, result in results_by_url)
    combined["num_shards"] = len(results_by_url)
    combined["shard_results"] = [
        {
            "owner_url": owner_url,
            "generation_id": result.get("generation_id"),
            "local_num_pairs": result.get("local_num_pairs"),
            "candidate_count": len(result.get("candidates", [])),
        }
        for owner_url, result in results_by_url
    ]
    return combined


def start_zorl_generation_all(
    urls: list[str],
    *,
    session_id: str,
    preload_candidates: bool = False,
    num_pairs: int | None = None,
    population_sharding: str = "replicated",
    num_shards: int | None = None,
) -> dict:
    payload = {"session_id": session_id, "preload_candidates": bool(preload_candidates)}
    if num_pairs is not None:
        payload["num_pairs"] = int(num_pairs)
    if population_sharding == "pair_shard":
        shard_count = int(num_shards or len(urls))
        if shard_count != len(urls):
            raise RuntimeError(
                f"pair_shard num_shards={shard_count} must match {len(urls)} direct control URLs"
            )
        if num_pairs is None:
            raise RuntimeError("pair_shard generation requires a global num_pairs value")

        results: list[dict | None] = [None] * len(urls)

        def post_one(idx: int, url: str) -> tuple[int, str, dict]:
            shard_payload = dict(payload)
            shard_payload["materialization"] = {
                "mode": "pair_shard",
                "shard_index": idx,
                "num_shards": shard_count,
            }
            shard_payload["owner_url"] = url
            return idx, url, _post(url, "/start_zorl_generation", shard_payload, timeout=300.0)

        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            futures = [pool.submit(post_one, idx, url) for idx, url in enumerate(urls)]
            for future in as_completed(futures, timeout=310.0):
                idx, _url, result = future.result()
                if isinstance(result, dict) and result.get("success") is False:
                    raise RuntimeError(
                        f"/start_zorl_generation failed on {_url}: {result.get('error_message') or result}"
                    )
                results[idx] = result

        results_by_url = [(url, result) for url, result in zip(urls, results, strict=True) if result is not None]
        if len(results_by_url) != len(urls):
            raise RuntimeError(f"/start_zorl_generation completed on {len(results_by_url)}/{len(urls)} shards")
        return _combine_sharded_generation_results(results_by_url, expected_num_pairs=int(num_pairs))

    if population_sharding != "replicated":
        raise RuntimeError(f"unsupported population_sharding={population_sharding!r}")
    results = _post_all(urls, "/start_zorl_generation", payload, timeout=300.0)
    first = results[0]
    first_candidates = [
        (candidate.get("candidate_id"), candidate.get("lora_name"))
        for candidate in first.get("candidates", [])
    ]
    for result in results[1:]:
        candidates = [
            (candidate.get("candidate_id"), candidate.get("lora_name"))
            for candidate in result.get("candidates", [])
        ]
        if candidates != first_candidates:
            raise RuntimeError(
                f"Native SGLang ZORL candidate mismatch for generation {first.get('generation_id')}"
            )
    return first


def apply_zorl_rewards(url: str, *, session_id: str, generation_id: str, candidate_rewards: list[dict], lr: float, max_update_norm: float | None = None, momentum: float = 0.0) -> dict:
    payload = {
        "session_id": session_id,
        "generation_id": generation_id,
        "candidate_rewards": candidate_rewards,
        "learning_rate": float(lr),
        "momentum": float(momentum),
    }
    if max_update_norm is not None:
        payload["max_update_norm"] = float(max_update_norm)
    return _post(url, "/apply_zorl_rewards", payload, timeout=300.0)


def _validate_apply_results_agree(urls: list[str], results: list[dict], *, rel_tol: float = float(os.environ.get("XORL_APPLY_AGREE_RTOL", "1e-5")), abs_tol: float = float(os.environ.get("XORL_APPLY_AGREE_ATOL", "1e-5"))) -> None:
    if not results:
        raise RuntimeError("/apply_zorl_rewards returned no results")
    first = results[0]
    for url, result in zip(urls[1:], results[1:], strict=True):
        for key in ("used_pairs", "dropped_pairs"):
            if int(result.get(key, -1)) != int(first.get(key, -1)):
                raise RuntimeError(
                    f"/apply_zorl_rewards mismatch for {key}: "
                    f"{first.get(key)!r} on {urls[0]} vs {result.get(key)!r} on {url}"
                )
        first_metrics = first.get("metrics", {}) or {}
        metrics = result.get("metrics", {}) or {}
        for key in ("update_norm", "pair_delta_mean", "pair_delta_std"):
            if key not in first_metrics and key not in metrics:
                continue
            if key not in first_metrics or key not in metrics:
                raise RuntimeError(f"/apply_zorl_rewards missing metric {key!r} on one worker")
            first_value = float(first_metrics[key])
            value = float(metrics[key])
            if not math.isclose(value, first_value, rel_tol=rel_tol, abs_tol=abs_tol):
                raise RuntimeError(
                    f"/apply_zorl_rewards metric mismatch for {key}: "
                    f"{first_value:.8g} on {urls[0]} vs {value:.8g} on {url}"
                )


def _sanitize_payload_floats(obj, fallback: float = 0.0):
    """Recursively replace non-finite floats (nan/inf). requests serializes JSON
    with allow_nan=False, so a single nan anywhere in the apply payload raises
    InvalidJSONError and crashes the whole run. This is the universal last line of
    defense — reward_mean is already set to a finite 'worst' value upstream in
    build_out_rewards; this catches leftover nan in aux metric blobs."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else fallback
    if isinstance(obj, dict):
        return {k: _sanitize_payload_floats(v, fallback) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_payload_floats(v, fallback) for v in obj]
    return obj


def apply_zorl_rewards_all(urls: list[str], *, session_id: str, generation_id: str, candidate_rewards: list[dict], lr: float, max_update_norm: float | None = None, momentum: float = 0.0, momentum_window: int = 8) -> dict:
    payload = {
        "session_id": session_id,
        "generation_id": generation_id,
        "candidate_rewards": _sanitize_payload_floats(candidate_rewards),
        "learning_rate": float(lr),
        "momentum": float(momentum),
        "momentum_window": int(momentum_window),
    }
    if max_update_norm is not None:
        payload["max_update_norm"] = float(max_update_norm)
    results = _post_all(urls, "/apply_zorl_rewards", payload, timeout=300.0)
    _validate_apply_results_agree(urls, results)
    for url in urls:
        flush_inference_cache(url)
    return results[0]


def ps_apply_and_broadcast(
    ps_url: str,
    replica_urls: list[str],
    *,
    session_id: str,
    generation_id: str,
    candidate_rewards: list[dict],
    lr: float,
    max_update_norm: float | None = None,
    momentum: float = 0.0,
    momentum_window: int = 8,
) -> dict:
    """fp32-master PARAMETER-SERVER apply + sparse-diff broadcast (PS mode).

    Replaces the per-replica ``apply_zorl_rewards_all`` fold. The PS is the seed
    authority: it ran ``/start_zorl_generation`` for this generation (so it holds
    the same b_seed/a_seed/perturbation_index specs the replicas built their
    population from) and holds the SINGLE fp32 master. This ONE call hands the
    PS the scores; the PS:

      1. folds the reward-weighted Muon update into its fp32 master IN FP32
         (exact sub-ULP accumulation; XORL_ZORL_FP32_MASTER=1 on the PS), and
         refreshes the served bf16 parent from the master;
      2. computes, per folded tensor, the sparse ``(index, value)`` set whose
         bf16 value CHANGED since the last sync, packs it in the in-tree
         ``delta_packed_v1`` format, writes it to the shared FS, and POSTs
         ``/update_weights_from_sparse_delta {delta_path}`` to every replica;
      3. each replica scatters the sparse diff into its served bf16 base
         (TP-shard-aware), landing on the EXACT bytes the PS master holds.

    Because one fold + one identical sparse diff reaches every replica, the
    cross-replica base is bit-identical BY CONSTRUCTION — there is nothing to
    validate (the old ``_validate_apply_results_agree`` is gone), and the
    replicas' KV/prefix caches are flushed by the sync, so no explicit
    per-replica flush loop is needed here.

    The PS endpoint ``/ps_apply_and_broadcast`` is the single server-side seam
    that wraps the existing fold (``apply_zorl_rewards`` with the fp32 master)
    and the existing sparse-delta sync; ``replica_urls`` is forwarded so the PS
    knows whom to push to (the PS may also be pre-registered, in which case the
    list is advisory). Returns the PS apply metrics (used_pairs, update_norm,
    sync stats), shaped like the legacy ``apply_zorl_rewards_all`` result.
    """
    payload = {
        "session_id": session_id,
        "generation_id": generation_id,
        "candidate_rewards": _sanitize_payload_floats(candidate_rewards),
        "learning_rate": float(lr),
        "momentum": float(momentum),
        "momentum_window": int(momentum_window),
        "replica_urls": list(replica_urls),
        # The PS restricts the sync to the folded module names; it knows them
        # (it owns the master), so no name list is needed from the client.
    }
    if max_update_norm is not None:
        payload["max_update_norm"] = float(max_update_norm)
    # The PS fold + N-way sparse broadcast can take longer than a single apply.
    # Live (pre fold-opt, 2026-06-30): the band-staged master fold alone is ~226s
    # + the build + sync, which blew past 600s and made the client RETRY mid-build
    # (idempotent fold, but wasteful re-build). 1800s lets the full fold+build+sync
    # finish in one call; bring it back down once the fold is optimized.
    return _post(ps_url, "/ps_apply_and_broadcast", payload, timeout=1800.0)


def ps_rebroadcast(ps_url: str, replica_urls: list[str], *, session_id: str) -> dict:
    """Re-broadcast the PS's CURRENT fp32-master parent to the replicas WITHOUT
    folding (used after an elitist-rollback restore on the PS). The PS computes
    the sparse bf16 diff of the restored master vs the replicas' last-synced
    state and pushes it, so the replicas serve the rolled-back parent."""
    return _post(
        ps_url,
        "/ps_rebroadcast",
        {"session_id": session_id, "replica_urls": list(replica_urls)},
        timeout=1800.0,
    )


def _is_no_active_generation_error(err: Exception) -> bool:
    """True if a /abort_zorl_generation failure is the benign 'nothing to abort'
    case (the generation was already cleared, e.g. by a server-side complete or a
    prior abort on retry). The SGLang endpoint raises a ValueError surfaced as an
    HTTP 400 whose body contains 'no active ZORL generation' or an 'active ...
    generation mismatch' (the gen we want gone is already gone). Either way the
    post-condition we need (this generation is not active) already holds, so we
    treat it as success and swallow it — abort is idempotent by design."""
    msg = str(err).lower()
    return (
        "no active zorl generation" in msg
        or "no active generation" in msg
        or ("active" in msg and "generation mismatch" in msg)
    )


def abort_zorl_generation_all(urls: list[str], *, session_id: str, generation_id: str) -> None:
    """End (abort) the named ZORL generation on every URL, in lockstep.

    This is the PS-mode replacement for the per-replica generation teardown that
    the legacy ``apply_zorl_rewards`` does implicitly (its server-side handler
    calls ``complete_generation``, clearing the replica's ``active_generation``).
    In PS mode the replicas only receive a sparse weight diff
    (``/update_weights_from_sparse_delta``) from the PS, which does NOT clear
    their active generation — so without this call the NEXT step's
    ``/start_zorl_generation`` fails with HTTP 400 "already has active
    generation". ``/abort_zorl_generation`` clears the active generation WITHOUT
    applying an update (the update already happened on the PS), so the next
    ``begin_generation`` can advance to a fresh generation id.

    Robust to retries / re-entry: if a URL has already cleared this generation
    (server-side complete, or a prior abort that partially succeeded), the
    'no active generation' error is benign and swallowed — the post-condition
    (this generation is not active anywhere) is what matters, and it holds.
    A genuine failure (connection, unexpected server error) is raised so the
    step fails loud rather than silently leaving a replica wedged.
    """
    if not urls:
        return
    payload = {"session_id": session_id, "generation_id": generation_id}

    def abort_one(url: str) -> tuple[str, Exception | None]:
        try:
            _post(url, "/abort_zorl_generation", payload, timeout=120.0)
            return url, None
        except RuntimeError as e:
            if _is_no_active_generation_error(e):
                return url, None
            return url, e

    errors: list[str] = []
    if len(urls) == 1:
        _url, err = abort_one(urls[0])
        if err is not None:
            errors.append(f"{_url}: {err}")
    else:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            futures = [pool.submit(abort_one, url) for url in urls]
            for future in as_completed(futures, timeout=130.0):
                _url, err = future.result()
                if err is not None:
                    errors.append(f"{_url}: {err}")
    if errors:
        raise RuntimeError(
            f"/abort_zorl_generation failed to clear generation {generation_id!r} on "
            f"{len(errors)} URL(s): {errors}"
        )


def snapshot_zorl_parent(url: str, *, session_id: str, snapshot_id: str) -> dict:
    return _post(url, "/snapshot_zorl_parent", {"session_id": session_id, "snapshot_id": snapshot_id})


def restore_zorl_parent(url: str, *, session_id: str, snapshot_id: str) -> dict:
    return _post(url, "/restore_zorl_parent", {"session_id": session_id, "snapshot_id": snapshot_id})


def snapshot_zorl_parent_all(urls: list[str], *, session_id: str, snapshot_id: str) -> dict:
    results = _post_all(urls, "/snapshot_zorl_parent", {"session_id": session_id, "snapshot_id": snapshot_id})
    return results[0]


def restore_zorl_parent_all(urls: list[str], *, session_id: str, snapshot_id: str) -> dict:
    results = _post_all(urls, "/restore_zorl_parent", {"session_id": session_id, "snapshot_id": snapshot_id})
    for url in urls:
        flush_inference_cache(url)
    return results[0]


def merge_zorl_parent_into_base_all(urls: list[str], *, session_id: str) -> dict:
    """EggRoll-style fold: merge the parent LoRA delta into the resident base weights on
    every replica and re-init LoRA-A. Makes the accumulated ES update full-rank (the base
    absorbs each step's low-rank update) instead of rank-capped by the LoRA. Flush KV after
    since the base weights changed. NOTE: incompatible with elitist-rollback (the LoRA
    snapshot no longer captures the update)."""
    results = _post_all(urls, "/merge_zorl_parent_into_base", {"session_id": session_id})
    for url in urls:
        flush_inference_cache(url)
    return results[0]


def export_zorl_parent(
    url: str,
    *,
    session_id: str,
    output_dir: str,
    snapshot_id: str | None = None,
    overwrite: bool = True,
) -> dict:
    payload = {
        "session_id": session_id,
        "output_dir": output_dir,
        "overwrite": bool(overwrite),
    }
    if snapshot_id is not None:
        payload["snapshot_id"] = snapshot_id
    return _post(url, "/export_zorl_parent", payload, timeout=300.0)


def generate_with_lora(
    url: str,
    *,
    input_ids: list[int],
    lora_path: str,
    temperature: float,
    max_new_tokens: int,
    n: int = 1,
    stop: list[str] | None = None,
    headers: dict | None = None,
) -> list[dict]:
    """Single-prompt /generate call. Returns list of dicts (one per sample if
    n>1). For n==1 SGLang returns a single dict; we normalize to list."""
    sampling = {"temperature": float(temperature), "max_new_tokens": int(max_new_tokens), "n": int(n)}
    # Mild repetition penalty breaks the base model's turn-2 <think> repetition loop
    # at temp 0.7 (measured: valid-rate 1/4 -> 3/4 at rep_pen 1.1), recovering both
    # throughput (fewer budget-burning loops) and solve (more valid turns). Env-gated.
    _rp = os.environ.get("ZORL_ROLLOUT_REPETITION_PENALTY")
    if _rp:
        sampling["repetition_penalty"] = float(_rp)
    if stop:
        sampling["stop"] = list(stop)
    payload = {"input_ids": input_ids, "sampling_params": sampling, "return_logprob": False, "lora_path": lora_path}
    data = _post(url, "/generate", payload, timeout=900.0, headers=headers)
    return data if isinstance(data, list) else [data]


def generate_with_lora_logprobs(
    url: str,
    *,
    input_ids: list[int],
    lora_path: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool = False,
    stop: list[str] | None = None,
    headers: dict | None = None,
) -> dict:
    """Single-prompt /generate that also returns per-token output logprobs.

    Used by the OPSD score mode: the candidate (student) samples a CoT and we
    keep its own token logprobs to form ``mean(teacher_lp - student_lp)``.
    """
    sampling = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "n": 1,
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "lora_path": lora_path,
    }
    data = _post(url, "/generate", payload, timeout=900.0, headers=headers)
    return data[0] if isinstance(data, list) else data


def _student_output_ids_and_logprobs(result: dict) -> tuple[list[int], list[float]]:
    """Extract (output_token_ids, output_token_logprobs) from a /generate result.

    SGLang ``meta_info.output_token_logprobs`` entries are ``[logprob, token_id, ...]``.
    """
    meta = result.get("meta_info", {}) or {}
    entries = meta.get("output_token_logprobs") or []
    out_ids: list[int] = []
    logprobs: list[float] = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[1] is None:
            continue
        out_ids.append(int(entry[1]))
        logprobs.append(_logprob_from_entry(entry[0]))
    return out_ids, logprobs


def input_logprobs_with_lora(
    url: str,
    *,
    input_ids: list[int],
    lora_path: str,
    headers: dict | None = None,
    logprob_start_len: int = 0,
) -> dict:
    """Ask SGLang for teacher-forced input token logprobs for a full sequence.

    ``logprob_start_len`` > 0 restricts logprob computation to positions
    [start, len), i.e. the target suffix. This does NOT change the prompt
    prefill (full context is always attended) or the returned target-token
    logprob values; it only avoids materializing the [num_skipped, vocab] logit
    rows for the prompt, cutting score-time logit memory/compute.
    """
    payload = {
        "input_ids": list(input_ids),
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": int(logprob_start_len),
    }
    # lora_path=None scores under the frozen base model (the OPSD teacher).
    if lora_path is not None:
        payload["lora_path"] = lora_path
    return _post(url, "/generate", payload, timeout=900.0, headers=headers)


def input_logprobs_batch_with_lora(
    url: str,
    *,
    input_ids_batch: list[list[int]],
    lora_path: str,
    headers: dict | None = None,
    logprob_start_lens: list[int] | None = None,
) -> list[dict]:
    """Ask SGLang for teacher-forced logprobs for a batch under one LoRA.

    ``logprob_start_lens`` (one per sequence) trims logprob computation to each
    sequence's target suffix; see ``input_logprobs_with_lora``.
    """
    payload = {
        "input_ids": [list(input_ids) for input_ids in input_ids_batch],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": (
            [int(x) for x in logprob_start_lens]
            if logprob_start_lens is not None
            else 0
        ),
    }
    # lora_path=None scores under the frozen base model (the OPSD teacher).
    if lora_path is not None:
        payload["lora_path"] = lora_path
    data = _post(url, "/generate", payload, timeout=900.0, headers=headers)
    if isinstance(data, list):
        return data
    if len(input_ids_batch) == 1 and isinstance(data, dict):
        return [data]
    raise RuntimeError(f"Expected {len(input_ids_batch)} batched logprob results, got {type(data).__name__}")


def generate_batch_with_lora_logprobs(
    url: str,
    *,
    input_ids_batch: list[list[int]],
    lora_path: str,
    temperature: float,
    max_new_tokens: int,
    ignore_eos: bool = False,
    stop: list[str] | None = None,
    headers: dict | None = None,
) -> list[dict]:
    """Batched /generate (one lora_path for the whole batch) with per-token output
    logprobs. Co-batches a candidate's whole example set into one decode batch so
    SGLang runs them ``len(batch)``-wide (the OPSD score path)."""
    sampling = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    payload = {
        "input_ids": [list(x) for x in input_ids_batch],
        "sampling_params": sampling,
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "lora_path": lora_path,
    }
    data = _post(url, "/generate", payload, timeout=900.0, headers=headers)
    if isinstance(data, list):
        return data
    if len(input_ids_batch) == 1 and isinstance(data, dict):
        return [data]
    raise RuntimeError(f"Expected {len(input_ids_batch)} batched generate results, got {type(data).__name__}")


def _opd_kl_reward_from_meta(result: dict) -> float:
    """Pull the server-side full-vocab KL reward out of a /generate result's
    meta_info (set by SGLang's _opd_kl_process_prefill). Returns NaN if absent."""
    mi = (result.get("meta_info") or {})
    r = mi.get("opd_kl_reward")
    if isinstance(r, list):
        return float(r[0]) if r else float("nan")
    if r is not None:
        return float(r)
    return float("nan")


def generate_opd_kl_batch(
    url: str,
    *,
    input_ids_batch: list[list[int]],
    lora_path: str | None,
    opd_kl_specs: list[dict],
    headers: dict | None = None,
    timeout: float = 900.0,
) -> list[float]:
    """Batched /generate carrying per-seq ``opd_kl`` specs (role/key/cot_len/mode).

    SGLang prefills each [prompt + shared CoT] seq, captures the CoT-suffix hidden
    states, and either caches the teacher (role=teacher_set) or computes the
    full-vocab KL vs the cached teacher (role=student), returning the reward in
    ``meta_info['opd_kl_reward']``. ``lora_path=None`` runs the frozen base model
    (the OPSD teacher). Generation itself is a throwaway single token."""
    payload = {
        "input_ids": [list(x) for x in input_ids_batch],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "opd_kl": list(opd_kl_specs),
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    data = _post(url, "/generate", payload, timeout=timeout, headers=headers)
    if isinstance(data, dict):
        data = [data]
    if len(data) != len(input_ids_batch):
        raise RuntimeError(f"opd_kl: expected {len(input_ids_batch)} results, got {len(data)}")
    return [_opd_kl_reward_from_meta(d) for d in data]


def broadcast_opd_kl_teacher(
    urls: list[str],
    *,
    input_ids_batch: list[list[int]],
    opd_kl_specs: list[dict],
    timeout: float = 900.0,
) -> None:
    """Send the teacher_set /generate (base model, hinted prefix + shared CoT) to
    EVERY replica in parallel, so each owner's GPU cache holds the teacher hidden
    for these keys before any candidate (sharded across owners) is scored."""
    if not urls:
        raise RuntimeError("broadcast_opd_kl_teacher needs at least one URL")
    errors: list[str] = []

    def one(url: str):
        try:
            generate_opd_kl_batch(
                url, input_ids_batch=input_ids_batch, lora_path=None,
                opd_kl_specs=opd_kl_specs, timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001 - surface a per-URL failure
            errors.append(f"{url}: {e}")

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        list(pool.map(one, urls))
    if errors:
        raise RuntimeError(f"teacher_set broadcast failed on {len(errors)}/{len(urls)} replicas: {errors[:2]}")


_LOGPROB_NONE_FLOOR = -100.0


def _logprob_from_entry(item) -> float:
    if item is None:
        return _LOGPROB_NONE_FLOOR
    if isinstance(item, (int, float)):
        if not math.isfinite(float(item)):
            return _LOGPROB_NONE_FLOOR
        return float(item)
    if isinstance(item, (list, tuple)) and item:
        value = item[0]
    elif isinstance(item, dict):
        value = item.get("logprob")
    else:
        return _LOGPROB_NONE_FLOOR
    if value is None:
        return _LOGPROB_NONE_FLOOR
    value = float(value)
    return value if math.isfinite(value) else _LOGPROB_NONE_FLOOR


def _teacher_forced_score_from_result(
    result: dict, *, target_token_count: int, use_logprob_reward: bool = False
) -> dict[str, float]:
    input_logprobs = result.get("meta_info", {}).get("input_token_logprobs", [])
    if len(input_logprobs) < target_token_count:
        raise RuntimeError(
            f"Expected at least {target_token_count} input token logprobs, got {len(input_logprobs)}"
        )
    target_entries = input_logprobs[-target_token_count:]
    logprobs = [_logprob_from_entry(item) for item in target_entries]
    mean_logprob = sum(logprobs) / max(len(logprobs), 1)
    prob = math.exp(max(mean_logprob, _LOGPROB_NONE_FLOOR))
    finite_count = sum(1 for value in logprobs if value > _LOGPROB_NONE_FLOOR)
    # --score-mode sft rewards the mean logprob directly (= -SFT loss): linear in
    # the loss and well-behaved under z-scoring, vs teacher_forced's geometric-mean
    # probability exp(mean_logprob). Both are monotonic in the loss, so the ES
    # ranking direction is identical; sft is the principled SFT-loss objective.
    reward = mean_logprob if use_logprob_reward else prob
    return {
        "reward": float(reward),
        "exact_match": 0.0,
        "teacher_forced_logprob": float(mean_logprob),
        "teacher_forced_prob": float(prob),
        "teacher_forced_token_count": float(len(logprobs)),
        "teacher_forced_finite_rate": float(finite_count / max(len(logprobs), 1)),
    }


def flush_inference_cache(url: str) -> None:
    try:
        _post(url, "/flush_cache", {}, timeout=30.0)
    except RuntimeError:
        pass


# ============================================================================
# Scoring loop
# ============================================================================


class _Args:
    """Shallow-copy args wrapper for per-call attribute overrides.
    Lets the probe layer swap rollout-temperature without mutating the
    real argparse namespace."""

    def __init__(self, base, **overrides):
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _make_generate_turn(infer_url, *, headers: dict | None = None):
    """Build the per-turn generate callable a multi-turn task uses. Each
    invocation hits /generate once with a single sample at the given
    sampling params and returns the completion text."""
    def generate_turn(input_ids, *, lora_path, temperature, max_new_tokens, stop=None):
        results = generate_with_lora(
            infer_url,
            input_ids=list(input_ids),
            lora_path=str(lora_path),
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            n=1,
            stop=stop,
            headers=headers,
        )
        if not results:
            return ""
        return results[0].get("text", "") or ""
    return generate_turn


def _candidate_owner_url(candidate: dict) -> str:
    owner_url = candidate.get("owner_url")
    if not owner_url:
        raise RuntimeError(
            f"candidate {candidate.get('candidate_id', '<unknown>')!r} is missing owner_url for owner routing"
        )
    return str(owner_url)


def _candidate_score_route(
    infer_urls: list[str],
    *,
    candidate: dict,
    job_idx: int,
    args,
) -> tuple[str, dict | None]:
    routing = str(getattr(args, "candidate_routing", "smg"))
    if routing == "owner":
        return _candidate_owner_url(candidate), None
    if routing == "owner_via_smg":
        owner_url = _candidate_owner_url(candidate)
        return infer_urls[job_idx % len(infer_urls)], {"X-SMG-Target-Worker": owner_url}
    return infer_urls[job_idx % len(infer_urls)], None


def _candidate_score_owner_key(*, candidate: dict, target_url: str, args) -> str:
    routing = str(getattr(args, "candidate_routing", "smg"))
    if routing in {"owner", "owner_via_smg"}:
        return _candidate_owner_url(candidate)
    return str(target_url)


def _ordered_score_pairs(candidates: list[dict], examples: list[Example], *, args) -> list[tuple[dict, Example]]:
    """Build candidate/example score work in an order that keeps shards fed.

    Candidate-major ordering is pathologically bad for owner-routed sharded
    populations: the first thread-pool wave can target only one or two owners.
    Owner round-robin spreads each wave across owners while preserving
    deterministic candidate order within each owner.
    """
    order = str(getattr(args, "score_job_order", "owner_round_robin"))
    if order == "candidate_major":
        return [(cand, ex) for cand in candidates for ex in examples]
    if order == "example_major" or str(getattr(args, "candidate_routing", "smg")) not in {"owner", "owner_via_smg"}:
        return [(cand, ex) for ex in examples for cand in candidates]
    if order != "owner_round_robin":
        raise RuntimeError(f"unsupported score_job_order={order!r}")

    candidates_by_owner: dict[str, list[dict]] = {}
    for cand in candidates:
        owner_url = cand.get("owner_url")
        if not owner_url:
            raise RuntimeError(
                f"candidate {cand.get('candidate_id', '<unknown>')!r} is missing owner_url for owner scoring"
            )
        candidates_by_owner.setdefault(str(owner_url), []).append(cand)

    owners = list(candidates_by_owner)
    max_candidates_per_owner = max((len(items) for items in candidates_by_owner.values()), default=0)
    pairs: list[tuple[dict, Example]] = []
    for ex in examples:
        for owner_pos in range(max_candidates_per_owner):
            for owner in owners:
                owner_candidates = candidates_by_owner[owner]
                if owner_pos < len(owner_candidates):
                    pairs.append((owner_candidates[owner_pos], ex))
    return pairs


def _chunks(items: list, size: int):
    if size <= 0:
        raise RuntimeError(f"chunk size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _ordered_score_batches(
    candidates: list[dict],
    examples: list[Example],
    *,
    batch_size: int,
    args,
) -> list[tuple[dict, list[Example]]]:
    """Build candidate/example-batch work while preserving owner balancing."""
    if batch_size <= 1:
        return [(cand, [ex]) for cand, ex in _ordered_score_pairs(candidates, examples, args=args)]

    order = str(getattr(args, "score_job_order", "owner_round_robin"))
    example_batches = [list(batch) for batch in _chunks(examples, batch_size)]
    if order == "candidate_major":
        return [(cand, ex_batch) for cand in candidates for ex_batch in example_batches]
    if order == "owner_candidate_contiguous":
        # Candidate-contiguous within each owner, interleaved across owners:
        # each candidate's batches run back-to-back so it loads into the LoRA
        # pool ONCE per step. Essential when candidates/replica > pool slots
        # (e.g. 64 cands vs max-loras-per-batch 33): owner_round_robin cycles
        # every candidate through the pool once per batch round -> eviction
        # churn (measured 6x step-time blowup at 256 pairs).
        by_owner: dict[str, list[dict]] = {}
        for cand in candidates:
            by_owner.setdefault(str(cand.get("owner_url") or ""), []).append(cand)
        owners_l = list(by_owner)
        max_per = max((len(v) for v in by_owner.values()), default=0)
        out: list[tuple[dict, list[Example]]] = []
        for cand_pos in range(max_per):
            for owner in owners_l:
                cands_o = by_owner[owner]
                if cand_pos < len(cands_o):
                    for ex_batch in example_batches:
                        out.append((cands_o[cand_pos], ex_batch))
        return out
    if order == "example_major" or str(getattr(args, "candidate_routing", "smg")) not in {"owner", "owner_via_smg"}:
        return [(cand, ex_batch) for ex_batch in example_batches for cand in candidates]
    if order != "owner_round_robin":
        raise RuntimeError(f"unsupported score_job_order={order!r}")

    candidates_by_owner: dict[str, list[dict]] = {}
    for cand in candidates:
        owner_url = cand.get("owner_url")
        if not owner_url:
            raise RuntimeError(
                f"candidate {cand.get('candidate_id', '<unknown>')!r} is missing owner_url for owner scoring"
            )
        candidates_by_owner.setdefault(str(owner_url), []).append(cand)

    owners = list(candidates_by_owner)
    max_candidates_per_owner = max((len(items) for items in candidates_by_owner.values()), default=0)
    batches: list[tuple[dict, list[Example]]] = []
    for ex_batch in example_batches:
        for owner_pos in range(max_candidates_per_owner):
            for owner in owners:
                owner_candidates = candidates_by_owner[owner]
                if owner_pos < len(owner_candidates):
                    batches.append((owner_candidates[owner_pos], ex_batch))
    return batches


def _teacher_forced_logprob_start_len(prompt_ids, target_token_count, *, margin: int) -> int:
    """Start index for teacher-forced logprob computation: the target suffix
    minus a small margin (the client only consumes the last
    ``target_token_count`` entries, so extra leading entries are harmless and a
    margin keeps it robust to SGLang's start-len off-by-one)."""
    return max(0, len(prompt_ids) - int(target_token_count) - int(margin))


def score_candidates(infer_url, *, candidates, examples: list[Example], task, args, tokenizer=None) -> tuple[list[dict], dict]:
    """For every candidate × example, fire /generate and call task.score_completion.
    Returns (per-candidate reward summary, debug-output map).

    Multi-turn tasks (``task.is_multi_turn = True``) route through
    ``task.rollout_completion(ex, generate_turn=..., lora_path=..., tokenizer=..., args=...)``
    which returns a score blob directly. The harness still parallelizes
    across the (cand, ex) grid; each rollout drives its own per-turn
    generate calls sequentially within the worker.

    Concurrency: thread-pool fans out generate calls across the candidate×example
    grid. SGLang batches them efficiently on its side via the LoRA pool. Watch
    --score-max-workers vs SGLang's --max-running-requests."""
    rewards_per_cand: dict[str, list[float]] = {c["candidate_id"]: [] for c in candidates}
    project_metrics: dict[str, dict[str, dict]] = {c["candidate_id"]: {} for c in candidates}
    outputs: dict[str, dict[str, str]] = {c["candidate_id"]: {} for c in candidates}

    infer_urls = _url_list(infer_url)
    multi_turn = bool(getattr(task, "is_multi_turn", False))
    # sft reuses the teacher-forced answer-logprob scoring machinery (its reward is
    # the mean logprob = -SFT loss; see use_logprob_reward below). The probe path
    # (probe_parent) deliberately does NOT treat sft as teacher_forced, so the
    # parent is probed by greedy exact-match accuracy rather than logprob.
    teacher_forced = args.score_mode in ("teacher_forced", "sft")
    opsd_kl = args.score_mode == "opsd_kl"
    opsd_kl_full = args.score_mode == "opsd_kl_full"
    if teacher_forced and not hasattr(task, "build_teacher_forced_example"):
        raise RuntimeError(f"Task {args.task!r} does not implement build_teacher_forced_example")
    if (opsd_kl or opsd_kl_full) and not hasattr(task, "build_opsd_prompts"):
        raise RuntimeError(f"Task {args.task!r} does not implement build_opsd_prompts")
    opsd_ignore_eos = bool(getattr(args, "opsd_student_ignore_eos", True))
    teacher_forced_batch_size = int(getattr(args, "teacher_forced_batch_size", 1) or 1)
    if teacher_forced_batch_size <= 0:
        raise RuntimeError(f"--teacher-forced-batch-size must be positive, got {teacher_forced_batch_size}")
    tf_logprob_trim = bool(getattr(args, "teacher_forced_logprob_trim", False))
    tf_logprob_trim_margin = int(getattr(args, "teacher_forced_logprob_trim_margin", 8) or 0)
    max_workers_per_owner = int(getattr(args, "score_max_workers_per_owner", 0) or 0)
    owner_semaphores: dict[str, threading.Semaphore] = {}
    if max_workers_per_owner > 0:
        for cand in candidates:
            if str(getattr(args, "candidate_routing", "smg")) in {"owner", "owner_via_smg"}:
                owner_key = _candidate_owner_url(cand)
            else:
                owner_key = str(infer_urls[0])
            owner_semaphores.setdefault(owner_key, threading.Semaphore(max_workers_per_owner))

    def with_owner_slot(candidate: dict, target_url: str, fn):
        if not owner_semaphores:
            return fn()
        owner_key = _candidate_score_owner_key(candidate=candidate, target_url=target_url, args=args)
        semaphore = owner_semaphores.setdefault(owner_key, threading.Semaphore(max_workers_per_owner))
        with semaphore:
            return fn()

    def record_score(cid: str, proj: str, reward_mean, texts: list[str], score_blob: dict | None, err: str | None):
        if err is not None:
            print(f"      WARN: score failed for {cid} {proj}: {err}")
            return
        rewards_per_cand[cid].append(float(reward_mean))
        metric = {"reward": float(reward_mean)}
        # Surface any extra metrics the task computed (format, partial, etc.)
        for k, v in (score_blob or {}).items():
            if k == "reward":
                continue
            metric[k] = float(v) if isinstance(v, (int, float)) else v
        project_metrics[cid][proj] = metric
        if texts:
            outputs[cid][proj] = texts[0]

    def build_out_rewards() -> tuple[list[dict], dict]:
        # First pass: finite per-candidate means; flag candidates with no finite reward.
        means: dict = {}
        degenerate = []
        for cand in candidates:
            cid = cand["candidate_id"]
            rs = [r for r in rewards_per_cand[cid] if r is not None and math.isfinite(r)]
            if rs:
                means[cid] = sum(rs) / len(rs)
            else:
                means[cid] = None
                degenerate.append(cid)
        # A degenerate candidate (all rewards non-finite/empty) is treated as the
        # WORST, never 0.0 — for KL rewards (<=0, 0=best) zeroing would mark it best
        # and pull the ES update toward garbage. Non-finite must also never reach
        # /apply_zorl_rewards (requests uses allow_nan=False -> InvalidJSONError).
        finite_means = [m for m in means.values() if m is not None]
        worst = min(finite_means) if finite_means else 0.0
        out_rewards = []
        for cand in candidates:
            cid = cand["candidate_id"]
            mean = means[cid] if means[cid] is not None else worst
            out_rewards.append({
                "candidate_id": cid,
                "reward_mean": float(mean),
                "num_rollouts": int(args.rollouts_per_puzzle),
                "project_metrics": _sanitize_payload_floats(project_metrics[cid]),
            })
        if degenerate:
            print(f"      WARN: {len(degenerate)}/{len(candidates)} candidates had non-finite/empty "
                  f"rewards (set to worst={worst:.4g} for apply)")
        return out_rewards, outputs

    if teacher_forced and teacher_forced_batch_size > 1:
        tf_by_project = {
            ex.project: task.build_teacher_forced_example(tokenizer, ex, args=args)
            for ex in examples
        }
        batch_jobs = [
            (idx, cand, ex_batch)
            for idx, (cand, ex_batch) in enumerate(
                _ordered_score_batches(candidates, examples, batch_size=teacher_forced_batch_size, args=args)
            )
        ]

        def one_batch(job_idx: int, cand, ex_batch: list[Example]):
            target_url, route_headers = _candidate_score_route(infer_urls, candidate=cand, job_idx=job_idx, args=args)
            tf_examples = [tf_by_project[ex.project] for ex in ex_batch]
            def run_batch():
                start_lens = (
                    [
                        _teacher_forced_logprob_start_len(
                            tf_ex.prompt_ids,
                            tf_ex.metadata["teacher_target_token_count"],
                            margin=tf_logprob_trim_margin,
                        )
                        for tf_ex in tf_examples
                    ]
                    if tf_logprob_trim
                    else None
                )
                results = input_logprobs_batch_with_lora(
                    target_url,
                    input_ids_batch=[tf_ex.prompt_ids for tf_ex in tf_examples],
                    lora_path=str(cand["lora_name"]),
                    headers=route_headers,
                    logprob_start_lens=start_lens,
                )
                if len(results) != len(ex_batch):
                    raise RuntimeError(f"Expected {len(ex_batch)} batched logprob results, got {len(results)}")
                rows = []
                for ex, tf_ex, result in zip(ex_batch, tf_examples, results, strict=True):
                    score_blob = _teacher_forced_score_from_result(
                        result,
                        target_token_count=int(tf_ex.metadata["teacher_target_token_count"]),
                        use_logprob_reward=(args.score_mode == "sft"),
                    )
                    reward_mean = float(score_blob.get("reward", 0.0))
                    rows.append((cand["candidate_id"], ex.project, reward_mean, [], score_blob, None))
                return rows

            try:
                return with_owner_slot(cand, target_url, run_batch)
            except RuntimeError as e:
                return [(cand["candidate_id"], ex.project, None, [], None, str(e)) for ex in ex_batch]

        with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
            for rows in pool.map(lambda j: one_batch(*j), batch_jobs):
                for row in rows:
                    record_score(*row)
        return build_out_rewards()

    if opsd_kl:
        # Batched on-policy distillation. Each candidate generates its WHOLE
        # example set in one /generate (one lora_path, len(examples) seqs) so the
        # decode runs that-many-wide; per_owner candidates run concurrently, so a
        # replica co-batches per_owner * len(examples) sequences (your "8 LoRAs x
        # 8 samples" wave). The frozen base teacher (lora_path=None) with the
        # HINTED prefix is then scored on the same sampled tokens in one batched
        # call. Reward/example = mean(teacher_lp) - mean(student_lp), a single-
        # sample estimate of -KL(student||teacher) on the student's own CoT.
        opsd_by_project = {
            ex.project: task.build_opsd_prompts(tokenizer, ex, args=args) for ex in examples
        }
        batch_jobs = [
            (idx, cand, ex_batch)
            for idx, (cand, ex_batch) in enumerate(
                _ordered_score_batches(candidates, examples, batch_size=len(examples), args=args)
            )
        ]

        def one_opsd_batch(job_idx: int, cand, ex_batch: list[Example]):
            target_url, route_headers = _candidate_score_route(
                infer_urls, candidate=cand, job_idx=job_idx, args=args
            )
            student_prompts = [opsd_by_project[ex.project][0] for ex in ex_batch]
            teacher_prefixes = [opsd_by_project[ex.project][1] for ex in ex_batch]

            def run_batch():
                gens = generate_batch_with_lora_logprobs(
                    target_url,
                    input_ids_batch=student_prompts,
                    lora_path=str(cand["lora_name"]),
                    temperature=args.rollout_temperature,
                    max_new_tokens=args.rollout_max_new_tokens,
                    ignore_eos=opsd_ignore_eos,
                    headers=route_headers,
                )
                if len(gens) != len(ex_batch):
                    raise RuntimeError(f"opsd_kl: expected {len(ex_batch)} gens, got {len(gens)}")
                per_seq = []  # (out_ids, student_lps, text)
                teacher_seqs = []
                for tp, gen in zip(teacher_prefixes, gens, strict=True):
                    out_ids, student_lps = _student_output_ids_and_logprobs(gen)
                    if not out_ids:
                        raise RuntimeError("opsd_kl: student produced no output tokens")
                    per_seq.append((out_ids, student_lps, str(gen.get("text", "") or "")))
                    teacher_seqs.append(list(tp) + out_ids)
                # NOTE: this SGLang build only accepts a scalar logprob_start_len, not
                # a per-seq list (HTTP 422), so we compute full-prompt input logprobs
                # (start_len=0) and slice the last len(out_ids) CoT entries per seq
                # below. The teacher prefix is fully prefilled either way.
                tresults = input_logprobs_batch_with_lora(
                    target_url,
                    input_ids_batch=teacher_seqs,
                    lora_path=None,  # frozen base model = the OPSD teacher
                    headers=route_headers,
                )
                if len(tresults) != len(ex_batch):
                    raise RuntimeError(f"opsd_kl: expected {len(ex_batch)} teacher results, got {len(tresults)}")
                rows = []
                for ex, (out_ids, student_lps, text), tres in zip(ex_batch, per_seq, tresults, strict=True):
                    tf_entries = tres.get("meta_info", {}).get("input_token_logprobs", []) or []
                    teacher_lps = [_logprob_from_entry(item) for item in tf_entries[-len(out_ids):]]
                    mean_student = sum(student_lps) / max(len(student_lps), 1)
                    mean_teacher = sum(teacher_lps) / max(len(teacher_lps), 1)
                    reward = mean_teacher - mean_student  # = -reverse_KL estimate
                    blob = {
                        "reward": float(reward),
                        "opsd_teacher_lp": float(mean_teacher),
                        "opsd_student_lp": float(mean_student),
                        "opsd_reverse_kl": float(mean_student - mean_teacher),
                        "opsd_token_count": float(len(out_ids)),
                    }
                    rows.append((cand["candidate_id"], ex.project, float(reward), [text], blob, None))
                return rows

            try:
                return with_owner_slot(cand, target_url, run_batch)
            except RuntimeError as e:
                return [(cand["candidate_id"], ex.project, None, [], None, str(e)) for ex in ex_batch]

        with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
            for rows in pool.map(lambda j: one_opsd_batch(*j), batch_jobs):
                for row in rows:
                    record_score(*row)
        return build_out_rewards()

    if opsd_kl_full:
        # TRUE full-vocab OPSD via ZORL: the KL is computed server-side on-GPU in
        # SGLang (sglang/srt/zorl/opd_kl.py); only the scalar reward comes back.
        # Scoring is unit-based: a "unit" is one (student_input, teacher_input,
        # cot_len, project) tuple sharing a teacher-cache key.
        #   * single-shot  -> one unit per example (hint-free prefix + parent CoT).
        #   * multi-turn    -> one unit per played TURN (hint-free turn prompt +
        #                      parent's turn reply); the feedback in the turn history
        #                      is what lets the injected behavior GENERALIZE.
        # Then for every unit: broadcast teacher_set ([hinted ctx + tokens], base
        # model) to EVERY real replica (teacher cache is per-replica), and score each
        # candidate (candidate LoRA) -> reward = -mean full-vocab KL(teacher||student),
        # aggregated to a per-project mean. Candidates route DIRECT to their owner
        # (candidate_routing=owner) because the SMG drops the opd_kl field.
        kl_mode = str(getattr(args, "opsd_kl_mode", "forward_kl_full"))
        vocab_chunk = int(getattr(args, "opsd_kl_vocab_chunk_size", 32768) or 0)
        # main() overwrites args.infer_url to a SINGLE control URL; the full
        # direct-replica list is args.control_infer_urls.
        replica_urls = list(getattr(args, "control_infer_urls", None) or _url_list(infer_url))
        parent_lora = str(args.parent_lora_name)
        call_tag = uuid.uuid4().hex[:8]
        multi_turn_opsd = bool(getattr(args, "opsd_multi_turn", False)) and multi_turn

        # units: list of dicts {key, project, student_in, teacher_in, cot_len}
        units: list[dict] = []
        if multi_turn_opsd:
            if not hasattr(task, "rollout_capture"):
                raise RuntimeError(f"Task {args.task!r} lacks rollout_capture for multi-turn opsd_kl_full")

            def _play(ex: Example):
                def parent_turn_ids(input_ids):
                    gen = generate_batch_with_lora_logprobs(
                        replica_urls[0],
                        input_ids_batch=[list(input_ids)],
                        lora_path=parent_lora,
                        temperature=args.rollout_temperature,
                        max_new_tokens=args.rollout_max_new_tokens,
                        ignore_eos=False,
                    )[0]
                    oid, _ = _student_output_ids_and_logprobs(gen)
                    return (gen.get("text", "") or ""), oid

                return ex, task.rollout_capture(
                    ex, generate_turn_ids=parent_turn_ids, tokenizer=tokenizer, args=args
                )

            with ThreadPoolExecutor(max_workers=min(len(examples), args.score_max_workers)) as pool:
                for ex, segs in pool.map(_play, examples):
                    for ti, (sids, tids, oids) in enumerate(segs):
                        if not oids:
                            continue
                        units.append({
                            "key": f"{call_tag}:{ex.project}:{ti}",
                            "project": ex.project,
                            "student_in": list(sids) + list(oids),
                            "teacher_in": list(tids) + list(oids),
                            "cot_len": len(oids),
                        })
        else:
            opsd_by_project = {
                ex.project: task.build_opsd_prompts(tokenizer, ex, args=args) for ex in examples
            }
            student_prefixes = [list(opsd_by_project[ex.project][0]) for ex in examples]
            cot_gens = generate_batch_with_lora_logprobs(
                replica_urls[0],
                input_ids_batch=student_prefixes,
                lora_path=parent_lora,
                temperature=args.rollout_temperature,
                max_new_tokens=args.rollout_max_new_tokens,
                ignore_eos=opsd_ignore_eos,
            )
            for ex, gen in zip(examples, cot_gens, strict=True):
                oid, _ = _student_output_ids_and_logprobs(gen)
                if not oid:
                    continue
                sp, tp = opsd_by_project[ex.project]
                units.append({
                    "key": f"{call_tag}:{ex.project}",
                    "project": ex.project,
                    "student_in": list(sp) + list(oid),
                    "teacher_in": list(tp) + list(oid),
                    "cot_len": len(oid),
                })

        if not units:
            raise RuntimeError("opsd_kl_full: no scorable units (parent produced no tokens)")
        print(f"      opsd_kl_full: {len(units)} units over {len({u['project'] for u in units})} examples "
              f"(multi_turn={multi_turn_opsd}, kl_mode={kl_mode})")

        # Broadcast teacher_set for every unit to every real replica.
        teacher_inputs = [u["teacher_in"] for u in units]
        teacher_specs = [
            {"role": "teacher_set", "key": u["key"], "cot_len": u["cot_len"], "mode": kl_mode, "vocab_chunk_size": vocab_chunk}
            for u in units
        ]
        broadcast_opd_kl_teacher(replica_urls, input_ids_batch=teacher_inputs, opd_kl_specs=teacher_specs)

        student_inputs = [u["student_in"] for u in units]
        student_specs = [
            {"role": "student", "key": u["key"], "cot_len": u["cot_len"], "mode": kl_mode, "vocab_chunk_size": vocab_chunk}
            for u in units
        ]
        unit_projects = [u["project"] for u in units]
        batch_jobs = [(idx, cand) for idx, cand in enumerate(candidates)]

        def one_opsd_full(job_idx: int, cand):
            target_url, route_headers = _candidate_score_route(
                infer_urls, candidate=cand, job_idx=job_idx, args=args
            )

            def run():
                rewards = generate_opd_kl_batch(
                    target_url,
                    input_ids_batch=student_inputs,
                    lora_path=str(cand["lora_name"]),
                    opd_kl_specs=student_specs,
                    headers=route_headers,
                )
                if len(rewards) != len(units):
                    raise RuntimeError(f"opsd_kl_full: expected {len(units)} rewards, got {len(rewards)}")
                # Aggregate per-unit rewards to a per-project mean (turns averaged).
                by_proj: dict[str, list[float]] = {}
                for proj, r in zip(unit_projects, rewards):
                    by_proj.setdefault(proj, []).append(r)
                rows = []
                for proj, rs in by_proj.items():
                    m = sum(rs) / max(len(rs), 1)
                    blob = {"reward": float(m), "opd_kl_reward": float(m), "opsd_units": float(len(rs))}
                    rows.append((cand["candidate_id"], proj, float(m), [], blob, None))
                return rows

            try:
                return with_owner_slot(cand, target_url, run)
            except RuntimeError as e:
                projs = sorted({u["project"] for u in units})
                return [(cand["candidate_id"], p, None, [], None, str(e)) for p in projs]

        with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
            for rows in pool.map(lambda j: one_opsd_full(*j), batch_jobs):
                for row in rows:
                    record_score(*row)
        return build_out_rewards()

    jobs = [(idx, cand, ex) for idx, (cand, ex) in enumerate(_ordered_score_pairs(candidates, examples, args=args))]

    def one_call(job_idx: int, cand, ex: Example):
        target_url, route_headers = _candidate_score_route(infer_urls, candidate=cand, job_idx=job_idx, args=args)
        if teacher_forced:
            tf_ex = task.build_teacher_forced_example(tokenizer, ex, args=args)
            tf_start_len = (
                _teacher_forced_logprob_start_len(
                    tf_ex.prompt_ids,
                    tf_ex.metadata["teacher_target_token_count"],
                    margin=tf_logprob_trim_margin,
                )
                if tf_logprob_trim
                else 0
            )
            try:
                result = with_owner_slot(
                    cand,
                    target_url,
                    lambda: input_logprobs_with_lora(
                        target_url,
                        input_ids=tf_ex.prompt_ids,
                        lora_path=str(cand["lora_name"]),
                        headers=route_headers,
                        logprob_start_len=tf_start_len,
                    ),
                )
                score_blob = _teacher_forced_score_from_result(
                    result,
                    target_token_count=int(tf_ex.metadata["teacher_target_token_count"]),
                    use_logprob_reward=(args.score_mode == "sft"),
                )
            except RuntimeError as e:
                return cand["candidate_id"], ex.project, None, [], None, str(e)
            reward_mean = float(score_blob.get("reward", 0.0))
            return cand["candidate_id"], ex.project, reward_mean, [], score_blob, None

        if multi_turn:
            try:
                score_blob = with_owner_slot(
                    cand,
                    target_url,
                    lambda: task.rollout_completion(
                        ex,
                        generate_turn=_make_generate_turn(target_url, headers=route_headers),
                        lora_path=str(cand["lora_name"]),
                        tokenizer=tokenizer,
                        args=args,
                    ),
                )
            except RuntimeError as e:
                return cand["candidate_id"], ex.project, None, [], None, str(e)
            reward_mean = float(score_blob.get("reward", 0.0))
            return cand["candidate_id"], ex.project, reward_mean, [], score_blob, None

        try:
            results = with_owner_slot(
                cand,
                target_url,
                lambda: generate_with_lora(
                    target_url,
                    input_ids=ex.prompt_ids,
                    lora_path=str(cand["lora_name"]),
                    temperature=args.rollout_temperature,
                    max_new_tokens=args.rollout_max_new_tokens,
                    n=int(args.rollouts_per_puzzle),
                    headers=route_headers,
                ),
            )
        except RuntimeError as e:
            return cand["candidate_id"], ex.project, None, [], None, str(e)
        rewards = []
        texts = []
        score_blob = {}
        for r in results:
            txt = r.get("text", "") or ""
            texts.append(txt)
            s = task.score_completion(ex, txt)
            rewards.append(float(s.get("reward", 0.0)))
            # Capture the last score blob for per-project metrics. If a task wants
            # to aggregate across rollouts it can override; rollouts_per_puzzle=1
            # is the common case.
            score_blob = s
        reward_mean = sum(rewards) / max(len(rewards), 1)
        return cand["candidate_id"], ex.project, reward_mean, texts, score_blob, None

    _total_jobs = len(jobs)
    _done = 0
    _t0 = time.time()
    _next_log = _t0 + 20.0
    with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
        for cid, proj, reward_mean, texts, score_blob, err in pool.map(lambda j: one_call(*j), jobs):
            record_score(cid, proj, reward_mean, texts, score_blob, err)
            _done += 1
            _now = time.time()
            if _now >= _next_log or _done == _total_jobs:
                _elapsed = _now - _t0
                _rate = _done / max(_elapsed, 1e-6)
                _eta = (_total_jobs - _done) / max(_rate, 1e-6)
                print(
                    f"      [scoring] {_done}/{_total_jobs} units "
                    f"({100 * _done / max(_total_jobs, 1):.0f}%) {_rate:.1f} units/s ETA {_eta:.0f}s",
                    flush=True,
                )
                _next_log = _now + 20.0
    return build_out_rewards()


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rank_zscores(values: list[float]) -> list[float]:
    """Return deterministic centered rank scores, scaled to unit-ish variance."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    mean = _mean(ranks)
    std = _std(ranks)
    if std <= 1e-8:
        return [0.0 for _ in values]
    return [(rank - mean) / std for rank in ranks]


def _parent_candidate(parent_lora_name: str) -> dict:
    return {
        "candidate_id": "__parent__",
        "lora_name": parent_lora_name,
    }


def _select_train_examples_for_step(
    train_pool: list[Example],
    *,
    train_size: int,
    seed: int,
    step: int,
    resample: bool,
) -> list[Example]:
    """Select the per-step train batch from a larger deterministic pool."""
    if train_size <= 0:
        raise RuntimeError(f"train_size must be positive, got {train_size}")
    if len(train_pool) < train_size:
        raise RuntimeError(f"train_pool has {len(train_pool)} examples, need at least train_size={train_size}")
    if not resample:
        return list(train_pool[:train_size])
    if train_size == len(train_pool):
        indices = list(range(len(train_pool)))
        random.Random(int(seed) * 1_000_003 + int(step)).shuffle(indices)
        return [train_pool[idx] for idx in indices]

    batches_per_epoch = max(1, math.ceil(len(train_pool) / train_size))
    epoch = int(step) // batches_per_epoch
    batch_index = int(step) % batches_per_epoch
    rng = random.Random(int(seed) * 1_000_003 + epoch)
    indices = list(range(len(train_pool)))
    rng.shuffle(indices)
    start = batch_index * train_size
    batch_indices = indices[start : start + train_size]
    if len(batch_indices) < train_size:
        next_indices = list(range(len(train_pool)))
        random.Random(int(seed) * 1_000_003 + epoch + 1).shuffle(next_indices)
        batch_indices.extend(next_indices[: train_size - len(batch_indices)])
    return [train_pool[idx] for idx in batch_indices]


def _project_reward_map(candidate_reward: dict) -> dict[str, float]:
    projects = candidate_reward.get("project_metrics") or {}
    return {
        str(project): float(metrics.get("reward", 0.0))
        for project, metrics in projects.items()
        if isinstance(metrics, dict)
    }


def transform_rewards_for_update(
    candidate_rewards: list[dict],
    *,
    strategy: str,
    parent_baseline: dict | None = None,
) -> tuple[list[dict], dict]:
    """Transform candidate rewards before the server's antithetic pair update.

    ``raw`` preserves historical behavior: the server computes pair deltas and
    standardizes them. Other strategies send already-shaped scalar rewards and
    mark the payload with ``_zorl_score_normalization=none`` so the server does
    not apply a second z-score pass.
    """
    if strategy == "raw":
        return candidate_rewards, {"update_strategy": "raw", "score_normalization": "standard"}

    out = []
    metadata = {"update_strategy": strategy, "score_normalization": "none"}

    if strategy == "centered":
        values = [float(item["reward_mean"]) for item in candidate_rewards]
        mean = _mean(values)
        metadata["transform_reward_mean"] = mean
        for item, value in zip(candidate_rewards, values, strict=True):
            patched = dict(item)
            patched["reward_mean"] = float(value - mean)
            patched["_zorl_score_normalization"] = "none"
            out.append(patched)
        return out, metadata

    if strategy == "rank":
        values = [float(item["reward_mean"]) for item in candidate_rewards]
        scores = _rank_zscores(values)
        metadata["transform_rank_std"] = _std(scores)
        for item, score in zip(candidate_rewards, scores, strict=True):
            patched = dict(item)
            patched["reward_mean"] = float(score)
            patched["_zorl_score_normalization"] = "none"
            out.append(patched)
        return out, metadata

    if strategy in {"project_baseline", "project_baseline_positive", "project_baseline_standardized"}:
        if not parent_baseline:
            raise RuntimeError(f"update strategy {strategy!r} requires a parent baseline")
        parent_projects = _project_reward_map(parent_baseline)
        project_names = sorted(parent_projects)
        if not project_names:
            raise RuntimeError(f"update strategy {strategy!r} needs parent project metrics")

        parent_values = [parent_projects[name] for name in project_names]
        metadata["parent_train_reward_mean"] = _mean(parent_values)

        if strategy == "project_baseline_standardized":
            project_stats: dict[str, tuple[float, float]] = {}
            for project in project_names:
                vals = [parent_projects[project]]
                for item in candidate_rewards:
                    vals.append(_project_reward_map(item).get(project, parent_projects[project]))
                std = _std(vals)
                project_stats[project] = (_mean(vals), std if std > 1e-8 else 1.0)

        for item in candidate_rewards:
            candidate_projects = _project_reward_map(item)
            shaped_scores = []
            for project in project_names:
                parent_reward = parent_projects[project]
                candidate_reward = candidate_projects.get(project, parent_reward)
                if strategy == "project_baseline":
                    shaped_scores.append(candidate_reward - parent_reward)
                elif strategy == "project_baseline_positive":
                    shaped_scores.append(max(0.0, candidate_reward - parent_reward))
                else:
                    project_mean, project_std = project_stats[project]
                    shaped_scores.append((candidate_reward - project_mean) / project_std)
            patched = dict(item)
            patched["reward_mean"] = float(_mean(shaped_scores))
            patched["_zorl_score_normalization"] = "none"
            out.append(patched)

        transformed_values = [float(item["reward_mean"]) for item in out]
        metadata["transformed_reward_mean"] = _mean(transformed_values)
        metadata["transformed_reward_std"] = _std(transformed_values)
        return out, metadata

    raise RuntimeError(f"unsupported update strategy {strategy!r}")


def probe_parent(infer_url, *, parent_lora_name, examples: list[Example], task, args, tokenizer=None) -> dict:
    """Probe on the eval set. Defaults to greedy (temp=0, n=1), but with
    ``--probe-temperature > 0`` and ``--probe-n > 1`` samples N completions
    per prompt and averages the per-key reward components — this matches
    the candidate-rollout distribution that ES is actually optimizing, so a
    real ES improvement shows up here even when small LoRA updates haven't
    yet flipped the argmax that greedy depends on.

    Returns aggregate metrics: ``{"reward_mean", "exact_match_rate",
    "exact_count", "n"}`` plus per-key means for any extra metrics the task
    emits. Parallelizes across the eval set with the same
    ``--score-max-workers`` budget as the scoring loop."""
    infer_urls = _url_list(infer_url)
    for url in infer_urls:
        flush_inference_cache(url)
    probe_temp = float(getattr(args, "probe_temperature", 0.0))
    probe_n = int(getattr(args, "probe_n", 1))
    multi_turn = bool(getattr(task, "is_multi_turn", False))
    generate_turn = _make_generate_turn(infer_urls[0]) if multi_turn else None
    teacher_forced = args.score_mode == "teacher_forced"
    opsd_kl = args.score_mode in ("opsd_kl", "opsd_kl_full")
    # Multi-turn OPSD must be PROBED multi-turn: a feedback-trained student scores ~0
    # on a single-shot (no-feedback) probe, so route it through the real-game probe
    # (rollout_completion) instead of the single-shot opsd branch below.
    opsd_multi_turn = bool(getattr(args, "opsd_multi_turn", False)) and multi_turn
    if teacher_forced and not hasattr(task, "build_teacher_forced_example"):
        raise RuntimeError(f"Task {args.task!r} does not implement build_teacher_forced_example")
    # Multi-turn tasks own their per-turn sampling params; the only knob the
    # probe layer applies is probe_n (number of independent games to average).
    # For greedy-style probes set --probe-temperature 0 and we mirror that
    # into rollout-temperature for the rollout call by overriding args; for
    # sampling probes use the rollout-temperature the task expects.

    def one_call(item):
        idx, ex = item
        target_url = infer_urls[idx % len(infer_urls)]
        if opsd_kl and not opsd_multi_turn:
            # OPSD eval: SAMPLE the student on the HINT-FREE prompt (no teacher,
            # no answer in context) and check whether its CoT actually emits the
            # target word. This is the real "can the distilled student solve it"
            # signal — the training reward (reverse KL) only measures distribution
            # match, not correctness. exact_match=1 iff target is among its guesses.
            student_prompt, _ = task.build_opsd_prompts(tokenizer, ex, args=args)
            target = str(ex.metadata.get("target", "")).lower()
            try:
                results = generate_with_lora(
                    target_url,
                    input_ids=student_prompt,
                    lora_path=parent_lora_name,
                    temperature=probe_temp,
                    max_new_tokens=args.rollout_max_new_tokens,
                    n=max(probe_n, 1),
                )
            except RuntimeError as e:
                print(f"      WARN: opsd probe failed for {ex.project}: {e}")
                return {"reward": 0.0, "exact_match": 0.0, "opsd_solved": 0.0}
            scored = []
            for r in results:
                guesses = task.extract_guesses(r.get("text", "") or "")
                solved = 1.0 if target and target in guesses else 0.0
                last_correct = 1.0 if guesses and guesses[-1] == target else 0.0
                # DENSE partial credit = best guess's (greens + 0.5*yellows)/5 toward
                # the target. This is the probe "reward" -> drives elitist rollback
                # (the 0/1 solve rate is too sparse early: it stays 0 and the run
                # self-restores to cold). =1.0 exactly when a guess == target.
                closeness = 0.0
                for g in guesses:
                    fb = task.compute_feedback(g, target) if target else ""
                    closeness = max(closeness, (fb.count("G") + 0.5 * fb.count("Y")) / 5.0)
                scored.append({
                    "reward": closeness,    # dense -> elitism signal
                    "exact_match": solved,  # true solve: target among the hint-free guesses -> exact_count
                    "opsd_solved": solved,
                    "opsd_closeness": closeness,
                    "opsd_last_guess_correct": last_correct,
                    "opsd_num_guesses": float(len(guesses)),
                })
            keys = set().union(*(s.keys() for s in scored)) if scored else set()
            return {k: float(sum(s.get(k, 0.0) for s in scored)) / max(len(scored), 1) for k in keys}

        if teacher_forced:
            tf_ex = task.build_teacher_forced_example(tokenizer, ex, args=args)
            try:
                result = input_logprobs_with_lora(
                    target_url,
                    input_ids=tf_ex.prompt_ids,
                    lora_path=parent_lora_name,
                )
                return _teacher_forced_score_from_result(
                    result,
                    target_token_count=int(tf_ex.metadata["teacher_target_token_count"]),
                )
            except RuntimeError as e:
                print(f"      WARN: teacher-forced probe failed for {ex.project}: {e}")
                return {"reward": 0.0, "exact_match": 0.0}

        if multi_turn:
            scored = []
            # Temporarily override rollout temperature with probe-temp for
            # the duration of this call so the task respects our sampling
            # choice. Use a shallow copy of args.
            probe_args = _Args(args, rollout_temperature=probe_temp if probe_temp > 0 else args.rollout_temperature)
            # Per-example replica (target_url) so the 128-eval probe spreads its
            # games across all replicas instead of hammering infer_urls[0] — a
            # single replica can't sustain ~512 multi-turn games and crashes,
            # which on restart wipes its in-process LoRAs + ZORL session.
            local_generate_turn = _make_generate_turn(target_url)
            for _ in range(max(probe_n, 1)):
                try:
                    blob = task.rollout_completion(
                        ex,
                        generate_turn=local_generate_turn,
                        lora_path=parent_lora_name,
                        tokenizer=tokenizer,
                        args=probe_args,
                    )
                except RuntimeError as e:
                    print(f"      WARN: probe failed for {ex.project}: {e}")
                    blob = {"reward": 0.0, "exact_match": 0.0}
                scored.append(blob)
        else:
            try:
                results = generate_with_lora(
                    target_url,
                    input_ids=ex.prompt_ids,
                    lora_path=parent_lora_name,
                    temperature=probe_temp,
                    max_new_tokens=args.rollout_max_new_tokens,
                    n=probe_n,
                )
            except RuntimeError as e:
                print(f"      WARN: probe failed for {ex.project}: {e}")
                return {"reward": 0.0, "exact_match": 0.0}
            scored = [task.score_completion(ex, r.get("text", "")) for r in results]
        # Average per-rollout scores across the N samples. Falls back to a
        # single score when probe_n == 1 (no extra cost).
        keys = set().union(*(s.keys() for s in scored)) if scored else set()
        return {k: float(sum(s.get(k, 0.0) for s in scored)) / max(len(scored), 1) for k in keys}

    with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
        blobs = list(pool.map(one_call, enumerate(examples)))

    n = max(len(blobs), 1)
    # Aggregate any numeric keys task emits.
    keys = set()
    for b in blobs:
        for k, v in b.items():
            if isinstance(v, (int, float)):
                keys.add(k)
    agg = {}
    for k in keys:
        vals = [float(b.get(k, 0.0)) for b in blobs]
        agg[f"{k}_mean"] = sum(vals) / n
    agg["exact_count"] = sum(float(b.get("exact_match", 0.0)) for b in blobs)
    agg["n"] = len(examples)
    return agg


# ============================================================================
# Main loop
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone ZORL harness arg parser.

    Extracted from main() so alternate drivers (e.g. run_wordle_zorl_xorl_ps.py,
    the xorl-trainer PS backend) can reuse the full arg surface that
    score_candidates / probe_parent / the task rollout expect, then add their own
    args. Behavior of main() is unchanged.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["countdown", "gsm8k", "alphabet_sort", "wordle", "opd_multiplication", "mult"], help="Which task to train on")
    parser.add_argument(
        "--infer-url",
        required=True,
        nargs="+",
        help=(
            "Direct SGLang control URL(s), separated by spaces or commas. "
            "Native ZORL state mutations are sent to every URL."
        ),
    )
    parser.add_argument(
        "--reward-infer-url",
        default=None,
        help=(
            "Optional routed generation URL for candidate scoring/probes, e.g. SMG. "
            "Defaults to the direct control URL list."
        ),
    )
    parser.add_argument(
        "--ps-url",
        default=None,
        help=(
            "fp32-master PARAMETER-SERVER control URL (single). When set, ZORL "
            "runs in PS mode: the PS is the seed authority + holds the single "
            "fp32 master, folds in fp32 ONCE per step, and broadcasts a sparse "
            "bf16-diff to the --infer-url replicas (no per-replica fold, no "
            "agreement check). The PS must run with XORL_ZORL_FP32_MASTER=1 and "
            "share the session seed with the replicas (it joins the session + "
            "generation lockstep so it derives the identical perturbation "
            "seeds). Unset = legacy per-replica fold."
        ),
    )
    parser.add_argument(
        "--population-sharding",
        choices=["replicated", "pair_shard"],
        default="replicated",
        help="Candidate population materialization mode across direct control URLs.",
    )
    parser.add_argument(
        "--num-shards",
        default="auto",
        help="Number of population shards. Use 'auto' to match the direct control URL count.",
    )
    parser.add_argument(
        "--pairs-per-shard",
        type=int,
        default=None,
        help="Optional local pair count; when set, global --num-pairs becomes pairs_per_shard * num_shards.",
    )
    parser.add_argument(
        "--candidate-routing",
        choices=["owner", "owner_via_smg", "smg"],
        default=None,
        help=(
            "Candidate scoring route. owner_via_smg sends scoring requests to SMG with "
            "X-SMG-Target-Worker set to the candidate owner."
        ),
    )
    parser.add_argument(
        "--preload-candidates",
        action="store_true",
        help=(
            "Materialize each generation's candidate LoRAs into the SGLang GPU LoRA pool before scoring. "
            "For sharded populations, each owner preloads only its local candidates."
        ),
    )
    parser.add_argument("--adapter-dir", required=True, help="On-disk path to the init adapter (adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--model", required=True, help="HF model path/id for tokenizer loading")
    parser.add_argument("--parent-lora-name", required=True, help="LoRA name to register the parent under (must be unique per run)")
    parser.add_argument("--session-id", required=True, help="ZORL session id (must be unique per run)")
    parser.add_argument("--snapshot-id", default="best", help="Name for the elitist-rollback snapshot")
    # ZORL hyperparams
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Wall-clock training-loop cap (0 = unlimited). Stops cleanly at the next step "
             "boundary once exceeded; pair with a large --steps for overnight runs.",
    )
    parser.add_argument("--num-pairs", type=int, default=32, help="Number of antithetic perturbation pairs (population_size = 2 * num_pairs)")
    parser.add_argument("--b-sigma", type=float, default=0.012, help="LoRA-B noise std (researcher used 0.012; our prior 0.1 was likely 8x too high)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-schedule", choices=["constant", "cosine"], default="constant",
        help="cosine anneals lr from --lr to --lr*--lr-min-frac over --lr-decay-steps "
        "(holds at the floor after). Targets the late-plateau drift: climb fast at high "
        "lr, then shrink the z-scored fixed-norm step so the parent settles at the optimum "
        "instead of random-walking off it.",
    )
    parser.add_argument("--lr-min-frac", type=float, default=0.1,
        help="cosine: lr floor as a fraction of --lr (default 0.1).")
    parser.add_argument("--lr-decay-steps", type=int, default=0,
        help="cosine: steps over which to anneal to the floor (0 = use --steps).")
    parser.add_argument("--lr-hold-steps", type=int, default=0,
        help="cosine: hold lr at --lr for this many steps BEFORE decaying (delayed "
        "decay). 0 = decay from step 0 (legacy). Use to keep full lr through the ES "
        "climb and only shrink the step in the post-peak tail to beat the drift.")
    parser.add_argument(
        "--lr-pop-scale",
        type=float,
        default=float(os.environ.get("LR_POP_SCALE", "0") or "0"),
        help="Opt-in population-size LR scaling (default 0 = OFF, legacy behavior). "
        "When > 0, the effective lr is multiplied by sqrt(num_pairs / --lr-pop-scale-ref), "
        "so doubling --num-pairs raises lr by sqrt(2). Matches HyperscaleES's "
        "lr_scale*sigma^2*sqrt(total_parallel_generations) (LR proportional to sqrt(population)). "
        "Set this flag to 1 to enable with the default reference pop, or env LR_POP_SCALE=1. "
        "Combines multiplicatively with --lr-schedule cosine/hold decay.",
    )
    parser.add_argument(
        "--lr-pop-scale-ref",
        type=float,
        default=float(os.environ.get("LR_POP_SCALE_REF", "8") or "8"),
        help="Baseline population (num_pairs) at which the sqrt-pop LR multiplier is 1.0 "
        "(default 8). Only used when --lr-pop-scale > 0.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Server-side SGD-momentum on the ES update (v = momentum*v + u; step = lr*v). "
        "B-space velocity is only coherent while LoRA-A is fixed, so pair with "
        "--merge-every-steps K where K >~ 2/(1-momentum); the server resets velocity at "
        "every merge/restore.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=8,
        help="fresh_ab truncated-EMA history length H (refold cost ~(1+H)x; window 4 at beta 0.7 keeps 83%% of EMA mass).",
    )
    parser.add_argument("--max-update-norm", type=float, default=3500.0)
    parser.add_argument(
        "--update-strategy",
        choices=[
            "raw",
            "centered",
            "rank",
            "project_baseline",
            "project_baseline_positive",
            "project_baseline_standardized",
        ],
        default="raw",
        help=(
            "Client-side reward transform before /apply_zorl_rewards. raw keeps server standardization; "
            "other strategies send transformed scores with _zorl_score_normalization=none."
        ),
    )
    parser.add_argument(
        "--perturbation-mode",
        choices=["b_only", "a_and_b", "fresh_ab"],
        default="b_only",
        help=(
            "Which LoRA factors SGLang perturbs for ZORL candidates. fresh_ab is the "
            "EGGROLL-style paired outer-product mode: fresh A+B noise per pair, update "
            "folded directly into base weights (requires momentum=0, B=0 parent)."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    # Rollout shaping
    parser.add_argument("--rollouts-per-puzzle", type=int, default=1)
    parser.add_argument(
        "--score-mode",
        choices=["rollout", "teacher_forced", "sft", "opsd_kl", "opsd_kl_full"],
        default="rollout",
        help=(
            "Candidate reward source. sft scores the mean teacher-forced logprob of the task's "
            "ground-truth target (= -SFT loss) via build_teacher_forced_example; candidates are "
            "ranked by SFT loss and the parent is probed by greedy exact-match (the headline). "
            "teacher_forced scores task-provided target traces with input "
            "logprobs. opsd_kl is the legacy top-1 reverse-KL estimate over the student's sampled "
            "tokens. opsd_kl_full is the TRUE full-vocab OPSD KL: the parent samples one shared "
            "hint-free CoT, the hinted teacher is cached on every replica, and each candidate is "
            "rewarded by -mean full-vocab KL (default forward_kl_full = KL(teacher||student), "
            "mode-covering so it injects the hint) computed server-side on-GPU inside SGLang."
        ),
    )
    parser.add_argument(
        "--opsd-kl-mode",
        choices=["forward_kl_full", "reverse_kl_full"],
        default="forward_kl_full",
        help=(
            "opsd_kl_full KL direction. forward_kl_full = KL(teacher||student) is mode-covering and "
            "injects the teacher's answer mass (the point of OPSD with a hinted teacher); "
            "reverse_kl_full = KL(student||teacher) is mode-seeking and will not inject."
        ),
    )
    parser.add_argument(
        "--opsd-kl-vocab-chunk-size",
        type=int,
        default=32768,
        help="opsd_kl_full: vocab chunk size for the streaming on-GPU KL (<=0 = whole vocab).",
    )
    parser.add_argument(
        "--opsd-multi-turn",
        action="store_true",
        help=(
            "opsd_kl_full: distill on the real MULTI-TURN feedback game (parent plays "
            "rollout_capture; one full-vocab KL unit per turn against the hinted teacher) "
            "instead of single-shot. The per-turn feedback is what lets the student generalize."
        ),
    )
    parser.add_argument(
        "--opsd-student-ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="opsd_kl: force the student to generate a full max_new_tokens CoT (ignore EOS).",
    )
    parser.add_argument(
        "--wordle-teacher-trace-style",
        choices=["hinted_cot", "guess_only"],
        default="hinted_cot",
        help="Wordle teacher-forced trace format used when --score-mode=teacher_forced.",
    )
    parser.add_argument(
        "--wordle-prompt-style",
        choices=[
            "default",
            "public_reasoning",
            "enumerate",
            "public_reasoning_constraints",
            "constraints_enumerate",
            # *_think variants open the model's private <think> block (the GRPO
            # floor-protocol that scores 0.469); same board layout, thinking on.
            "public_reasoning_think",
            "public_reasoning_constraints_think",
        ],
        default="default",
        help=(
            "Wordle STUDENT prompt for multi-turn opsd_kl_full. public_reasoning makes the student "
            "emit an explicit public <reasoning> then the guess (paired with --wordle-teacher-prompt-style "
            "policy_hint). public_reasoning_constraints shows only the public board-constraint summary "
            "(no candidate list, no target). constraints_enumerate is the same board-summary-only view but "
            "asks the student to first enumerate the consistent words itself then narrow (counterpart to the "
            "enumerate teacher). Also used by the eval probe so it matches the trained student."
        ),
    )
    parser.add_argument(
        "--wordle-teacher-prompt-style",
        choices=["answer_hint", "policy_hint", "enumerate"],
        default="answer_hint",
        help=(
            "Wordle TEACHER context for multi-turn opsd_kl_full. answer_hint = legacy raw private target "
            "(teaches play, not solving). policy_hint = strong PUBLIC solver over the feedback-derived "
            "candidate set, using the target only to tie-break among public-valid candidates — its edge is "
            "feedback-reasoning (in the student's state), the lever for SOLVING to transfer."
        ),
    )
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=2,
        help=(
            "Wordle multi-turn: how many times to re-prompt after an invalid (unparseable, "
            "not-legal, or repeated) guess before giving up. An invalid action does not count as a "
            "turn or advance the game. Read by tasks.wordle.rollout_capture / rollout_completion."
        ),
    )
    parser.add_argument("--rollout-temperature", type=float, default=0.6)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=64)
    parser.add_argument("--score-max-workers", type=int, default=32)
    parser.add_argument(
        "--score-max-workers-per-owner",
        type=int,
        default=0,
        help=(
            "Optional per-SGL scoring concurrency cap. For owner-routed sharded populations this gates "
            "requests by candidate owner_url; 0 means only --score-max-workers limits concurrency."
        ),
    )
    parser.add_argument(
        "--teacher-forced-batch-size",
        type=int,
        default=1,
        help="Examples per /generate request for teacher-forced scoring; rollout scoring always uses single prompts.",
    )
    parser.add_argument(
        "--teacher-forced-logprob-trim",
        action="store_true",
        help=(
            "Compute teacher-forced logprobs only for the target suffix "
            "(logprob_start_len = prompt_len - target_tokens - margin) instead of "
            "the whole prompt. Same scores, less score-time logit memory/compute."
        ),
    )
    parser.add_argument(
        "--teacher-forced-logprob-trim-margin",
        type=int,
        default=8,
        help="Extra leading logprob positions to keep when --teacher-forced-logprob-trim is set (robustness margin).",
    )
    parser.add_argument(
        "--score-job-order",
        choices=["owner_round_robin", "example_major", "candidate_major", "owner_candidate_contiguous"],
        default="owner_round_robin",
        help=(
            "Submission order for candidate/example score jobs. owner_round_robin keeps sharded owner-routed "
            "SGLang pools hot; candidate_major preserves the original ordering for debugging."
        ),
    )
    # Task sizing (countdown ignores eval-size — train and eval are the same pool)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument(
        "--train-pool-size",
        type=int,
        default=None,
        help="Number of training examples to build before per-step selection; defaults to --train-size.",
    )
    parser.add_argument(
        "--resample-train-each-step",
        action="store_true",
        help="Draw a deterministic per-step train batch from --train-pool-size instead of reusing one fixed split.",
    )
    parser.add_argument("--eval-size", type=int, default=128)
    # Probe + rollback
    parser.add_argument("--probe-interval", type=int, default=5, help="Probe parent every N gens; 0 disables probes")
    parser.add_argument("--probe-temperature", type=float, default=0.0, help="Probe sampling temperature; 0 is greedy. Use rollout-temperature to match the distribution ES is actually optimizing.")
    parser.add_argument("--probe-n", type=int, default=1, help="Probe rollouts per prompt; >1 only makes sense when probe-temperature > 0")
    parser.add_argument("--elitist-rollback", action="store_true", help="If probe reward drops below best, restore parent snapshot")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional shared directory for exported parent LoRA adapters.",
    )
    parser.add_argument(
        "--export-interval",
        type=int,
        default=0,
        help="Export the live parent every N completed steps; 0 disables periodic exports.",
    )
    parser.add_argument(
        "--merge-every-steps",
        type=int,
        default=0,
        help="EggRoll: fold the parent LoRA delta into the resident base weights + re-init "
        "LoRA-A every N steps (0=never). Makes the accumulated ES update full-rank. "
        "Disable --elitist-rollback when using this (the LoRA snapshot won't capture the merge).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    control_infer_urls = _url_list(args.infer_url)
    reward_infer_urls = _url_list(args.reward_infer_url) or control_infer_urls
    if not control_infer_urls:
        raise RuntimeError("--infer-url must provide at least one direct control URL")
    if args.num_shards == "auto":
        args.resolved_num_shards = len(control_infer_urls)
    else:
        args.resolved_num_shards = int(args.num_shards)
    if args.resolved_num_shards <= 0:
        raise RuntimeError(f"--num-shards must be positive, got {args.resolved_num_shards}")
    if args.population_sharding == "pair_shard" and args.resolved_num_shards != len(control_infer_urls):
        raise RuntimeError(
            f"pair_shard requires --num-shards ({args.resolved_num_shards}) to equal "
            f"the direct --infer-url count ({len(control_infer_urls)})"
        )
    if args.pairs_per_shard is not None:
        if args.pairs_per_shard <= 0:
            raise RuntimeError(f"--pairs-per-shard must be positive, got {args.pairs_per_shard}")
        args.num_pairs = int(args.pairs_per_shard) * int(args.resolved_num_shards)
    if args.candidate_routing is None:
        if args.population_sharding != "replicated":
            args.candidate_routing = "owner_via_smg" if reward_infer_urls != control_infer_urls else "owner"
        else:
            args.candidate_routing = "smg"
    if args.population_sharding != "replicated" and args.candidate_routing not in {"owner", "owner_via_smg"}:
        raise RuntimeError("sharded ZORL candidate scoring requires --candidate-routing owner or owner_via_smg")
    if args.train_size <= 0:
        raise RuntimeError(f"--train-size must be positive, got {args.train_size}")
    if args.teacher_forced_batch_size <= 0:
        raise RuntimeError(f"--teacher-forced-batch-size must be positive, got {args.teacher_forced_batch_size}")
    if args.score_max_workers <= 0:
        raise RuntimeError(f"--score-max-workers must be positive, got {args.score_max_workers}")
    if args.score_max_workers_per_owner < 0:
        raise RuntimeError(
            f"--score-max-workers-per-owner must be non-negative, got {args.score_max_workers_per_owner}"
        )
    args.train_pool_size = int(args.train_pool_size or args.train_size)
    if args.train_pool_size < args.train_size:
        raise RuntimeError(
            f"--train-pool-size ({args.train_pool_size}) must be >= --train-size ({args.train_size})"
        )
    if getattr(args, "ps_url", None):
        if args.population_sharding != "replicated":
            raise RuntimeError(
                "--ps-url (fp32-master PS mode) requires --population-sharding "
                "replicated: the PS folds the single master from the full "
                "replicated population, not a pair-shard."
            )
        # Elitist rollback IS supported in PS mode: snapshot/restore now also
        # save/restore the fp32 MASTER on the PS (the base, where the update
        # lives), and a restore re-broadcasts the restored parent to the
        # replicas (handled in the rollback block below).
    args.infer_url = control_infer_urls[0]
    args.control_infer_urls = control_infer_urls
    args.reward_infer_urls = reward_infer_urls
    reward_infer_url = reward_infer_urls[0] if len(reward_infer_urls) == 1 else reward_infer_urls

    print(f"[init] task={args.task} control_infer_urls={control_infer_urls}")
    print(f"[init] reward_infer_urls={reward_infer_urls}")
    print(f"[init] adapter_dir={args.adapter_dir}")
    print(f"[init] parent_lora_name={args.parent_lora_name}")
    print(f"[init] session_id={args.session_id}")
    print(
        f"[init] hyperparams: steps={args.steps} num_pairs={args.num_pairs} sigma={args.b_sigma} "
        f"lr={args.lr} momentum={args.momentum} perturbation_mode={args.perturbation_mode} "
        f"score_mode={args.score_mode} update_strategy={args.update_strategy} "
        f"population_sharding={args.population_sharding} num_shards={args.resolved_num_shards} "
        f"candidate_routing={args.candidate_routing} "
        f"preload_candidates={args.preload_candidates} "
        f"score_max_workers={args.score_max_workers} score_job_order={args.score_job_order} "
        f"score_max_workers_per_owner={args.score_max_workers_per_owner} "
        f"teacher_forced_batch_size={args.teacher_forced_batch_size} "
        f"export_interval={args.export_interval} "
        f"rollout_temp={args.rollout_temperature} "
        f"max_new_tokens={args.rollout_max_new_tokens}"
    )

    def maybe_export(label: str, *, snapshot_id: str | None = None) -> dict | None:
        if not args.export_dir:
            return None
        if getattr(args, "ps_url", None):
            # PS mode: the trained weights live in the served BASE (folded via the
            # PS fp32 master), NOT the LoRA adapter (fresh_ab keeps B≈0). The
            # /export_zorl_parent adapter export would emit a near-zero LoRA, and a
            # snapshot export would fail (the rollback snapshot lives on the PS, not
            # on control_infer_urls[0]). Base-weight export from the PS master is a
            # separate TODO; skip the adapter export here with a loud note rather
            # than write a misleading artifact.
            print(
                f"  export {label}: SKIPPED in PS mode (trained weights are in the "
                f"served base via the fp32 master, not the LoRA adapter; base export "
                f"is a TODO). snapshot_id={snapshot_id}",
                flush=True,
            )
            return None
        output_dir = str(Path(args.export_dir).absolute() / label)
        result = export_zorl_parent(
            control_infer_urls[0],
            session_id=args.session_id,
            output_dir=output_dir,
            snapshot_id=snapshot_id,
            overwrite=True,
        )
        if result.get("success") is False:
            raise RuntimeError(f"/export_zorl_parent failed: {result.get('error_message') or result}")
        print(
            f"  export {label}: output_dir={result.get('output_dir', output_dir)} "
            f"num_tensors={result.get('num_tensors', '?')} num_bytes={result.get('num_bytes', '?')}"
        )
        return result

    # Load task module + tokenizer + data.
    task = load_task(args.task)
    from transformers import AutoTokenizer  # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_pool, eval_examples = task.build_examples(
        tokenizer,
        train_size=args.train_pool_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    print(
        f"[init] train_size={args.train_size} train_pool_size={len(train_pool)} "
        f"resample_train_each_step={args.resample_train_each_step} "
        f"eval_size={len(eval_examples)} multi_turn={getattr(task, 'is_multi_turn', False)}"
    )

    # In PS mode the PS is itself an sglang LoRA-manager instance that joins the
    # SAME session (same seed -> identical perturbation seeds, generation
    # lockstep) and runs the fold. Its /start_zorl_session resolves the parent
    # via the LoRA registry, AND its fold (_apply_zorl_rewards_fresh_ab) reads
    # the parent adapter's rank/A-subspace and reconstructs per-pair A/B from
    # seeds against the parent's shapes — so the PS needs the IDENTICAL parent
    # init adapter loaded, exactly like the replicas. Load it on PS + replicas.
    session_urls = list(control_infer_urls)
    if getattr(args, "ps_url", None) and args.ps_url not in session_urls:
        session_urls = session_urls + [args.ps_url]

    # Register the init adapter as the parent on SGLang (PS + every replica).
    print(
        f"[step 1] /load_lora_adapter parent={args.parent_lora_name} "
        f"on {len(session_urls)} url(s)"
        + (" (incl. PS)" if getattr(args, "ps_url", None) else "")
    )
    load_lora_adapter_all(session_urls, args.parent_lora_name, str(Path(args.adapter_dir).absolute()))

    # Open the ZORL session pointing at that parent (PS + every replica).
    print(
        f"[step 2] /start_zorl_session num_pairs={args.num_pairs} "
        f"b_sigma={args.b_sigma} perturbation_mode={args.perturbation_mode}"
    )
    start_zorl_session_all(
        session_urls,
        session_id=args.session_id,
        parent_lora_name=args.parent_lora_name,
        num_pairs=args.num_pairs,
        b_sigma=args.b_sigma,
        seed=args.seed,
        perturbation_mode=args.perturbation_mode,
    )

    # Initial probe (cold parent) on the eval set.
    print(f"[step 3] cold probe on eval set ({len(eval_examples)} examples)...")
    cold = probe_parent(reward_infer_url, parent_lora_name=args.parent_lora_name, examples=eval_examples, task=task, args=args, tokenizer=tokenizer)
    cold_reward = cold.get("reward_mean", 0.0)
    cold_exact_rate = cold.get("exact_match_mean", 0.0)
    print(f"  cold: reward_mean={cold_reward:.4f} exact_rate={cold_exact_rate:.4f} exact_count={cold['exact_count']:.1f}/{cold['n']}")
    best_reward = cold_reward
    best_blob = cold
    if args.elitist_rollback:
        snapshot_zorl_parent_all(control_infer_urls, session_id=args.session_id, snapshot_id=args.snapshot_id)
        print(f"  snapshot {args.snapshot_id!r} captured at cold")
        maybe_export("best", snapshot_id=args.snapshot_id)

    # Main loop.
    t0 = time.time()
    def _pop_lr_multiplier() -> float:
        # Opt-in sqrt(population) LR scaling (HyperscaleES: LR proportional to
        # sqrt(total_parallel_generations)). OFF by default (--lr-pop-scale<=0),
        # in which case this returns 1.0 and behavior is identical to legacy.
        pop_scale = float(getattr(args, "lr_pop_scale", 0.0) or 0.0)
        if pop_scale <= 0.0:
            return 1.0
        ref = float(getattr(args, "lr_pop_scale_ref", 8.0) or 8.0)
        if ref <= 0.0:
            raise RuntimeError(f"--lr-pop-scale-ref must be positive, got {ref}")
        return math.sqrt(float(args.num_pairs) / ref)

    def _lr_for_step(step: int) -> float:
        mult = _pop_lr_multiplier()
        if args.lr_schedule == "constant":
            return float(args.lr) * mult
        hold = int(getattr(args, "lr_hold_steps", 0) or 0)
        if step < hold:
            return float(args.lr) * mult
        horizon = int(args.lr_decay_steps) if args.lr_decay_steps > 0 else int(args.steps)
        frac = min(1.0, (step - hold) / max(1, horizon))
        floor = float(args.lr) * float(args.lr_min_frac)
        return (floor + 0.5 * (float(args.lr) - floor) * (1.0 + math.cos(math.pi * frac))) * mult

    for step in range(args.steps):
        if args.max_runtime_seconds and (time.time() - t0) >= args.max_runtime_seconds:
            print(
                f"[max-runtime] reached {args.max_runtime_seconds}s after {step} steps; "
                f"stopping cleanly (overnight cap)"
            )
            break
        gen = start_zorl_generation_all(
            control_infer_urls,
            session_id=args.session_id,
            num_pairs=args.num_pairs,
            population_sharding=args.population_sharding,
            num_shards=args.resolved_num_shards,
            preload_candidates=args.preload_candidates,
        )
        gen_id = gen["generation_id"]
        candidates = gen["candidates"]
        if getattr(args, "ps_url", None):
            # Advance the PS's generation in lockstep (replicated, no scoring on
            # the PS): it derives the same generation_id + identical seeds so its
            # fold reconstructs the exact population the replicas scored. Assert
            # the PS agrees on generation_id (cheap structural check).
            ps_gen = start_zorl_generation_all(
                [args.ps_url],
                session_id=args.session_id,
                num_pairs=args.num_pairs,
                population_sharding="replicated",
                num_shards=1,
                preload_candidates=False,
            )
            if str(ps_gen.get("generation_id")) != str(gen_id):
                raise RuntimeError(
                    f"PS generation_id {ps_gen.get('generation_id')!r} != replica "
                    f"{gen_id!r} (PS out of session lockstep)"
                )
        train_examples = _select_train_examples_for_step(
            train_pool,
            train_size=args.train_size,
            seed=args.seed,
            step=step,
            resample=args.resample_train_each_step,
        )
        if args.resample_train_each_step:
            preview = ",".join(ex.project for ex in train_examples[:4])
            print(
                f"  train_batch step={step+1}: size={len(train_examples)} "
                f"pool={len(train_pool)} preview={preview}"
            )

        # Score all candidates × training examples.
        score_t0 = time.time()
        candidate_rewards, outputs = score_candidates(reward_infer_url, candidates=candidates, examples=train_examples, task=task, args=args, tokenizer=tokenizer)
        score_s = time.time() - score_t0
        parent_baseline = None
        if args.update_strategy.startswith("project_baseline"):
            parent_rewards, _parent_outputs = score_candidates(
                reward_infer_url,
                candidates=[_parent_candidate(args.parent_lora_name)],
                examples=train_examples,
                task=task,
                args=_Args(args, candidate_routing="smg"),
                tokenizer=tokenizer,
            )
            parent_baseline = parent_rewards[0] if parent_rewards else None

        reward_means = [c["reward_mean"] for c in candidate_rewards]
        best_cand_reward = max(reward_means) if reward_means else 0.0
        mean_cand_reward = sum(reward_means) / max(len(reward_means), 1)
        rewards_for_update, transform_metrics = transform_rewards_for_update(
            candidate_rewards,
            strategy=args.update_strategy,
            parent_baseline=parent_baseline,
        )

        # Apply the ES update server-side.
        apply_t0 = time.time()
        if getattr(args, "ps_url", None):
            # fp32-master PS mode: ONE fold on the PS (exact fp32 accumulation),
            # then a sparse bf16-diff broadcast to all replicas. Cross-replica
            # base identity is structural (one diff -> identical bytes), so no
            # agreement check and no per-replica fold/flush.
            apply_result = ps_apply_and_broadcast(
                args.ps_url,
                control_infer_urls,
                session_id=args.session_id,
                generation_id=gen_id,
                candidate_rewards=rewards_for_update,
                lr=_lr_for_step(step),
                max_update_norm=(args.max_update_norm if args.max_update_norm > 0 else None),
                momentum=args.momentum,
                momentum_window=args.momentum_window,
            )
            # Surface the sync result so an empty/failed sync is impossible to
            # miss: a successful PS step MUST have a nonzero nnz + written files.
            _ss = (apply_result or {}).get("sync_stats", {}) or {}
            print(
                f"  PS sync: success={apply_result.get('success')} "
                f"stage={apply_result.get('stage')} "
                f"nnz={_ss.get('total_nnz')} density={_ss.get('density')} "
                f"packed_bytes={_ss.get('packed_bytes')} "
                f"masters_present={_ss.get('masters_present')} "
                f"name_resolved={_ss.get('name_resolved')} "
                f"tp_size={_ss.get('tp_size')} "
                f"delta_path={apply_result.get('delta_path')}",
                flush=True,
            )
            if not apply_result.get("success"):
                # FAIL LOUD: the PS now returns success=False with a stage
                # (fold/empty_diff/push/exception) instead of a silent empty sync.
                print(
                    f"  ERROR: PS sync FAILED at stage={apply_result.get('stage')!r}: "
                    f"{apply_result.get('error_message')!r}. "
                    f"failures={apply_result.get('failures')} "
                    f"(see the PS pod log [ZORL-PS-ENDPOINT]/[ZORL-PS] stage trace).",
                    flush=True,
                )
                if apply_result.get("traceback"):
                    print(apply_result["traceback"], flush=True)
            # End this step's generation on every replica AND the PS, in lockstep.
            # The PS fold+broadcast only pushes a sparse weight diff to the replicas
            # (/update_weights_from_sparse_delta), which does NOT clear their
            # active_generation the way the legacy per-replica /apply_zorl_rewards
            # does (its handler calls complete_generation). Without this teardown the
            # replicas stay locked on this generation and the NEXT step's
            # /start_zorl_generation fails with HTTP 400 "already has active
            # generation". /abort_zorl_generation clears the generation WITHOUT
            # applying an update (the update already landed on the PS via the fold),
            # so the next begin_generation advances to a fresh generation id.
            #
            # Only tear down after a SUCCESSFUL apply: on failure we leave state
            # intact (the run fails loud below / on the next start) so the wedged
            # generation is inspectable rather than silently cleared. The abort is
            # idempotent (benign "no active generation" is swallowed in
            # abort_zorl_generation_all), so an apply-timeout retry that re-enters
            # this block is safe.
            if apply_result.get("success"):
                # Replicas first (they are the ones that would otherwise reject the
                # next start), then the PS — defensive in case the PS server-side
                # fold did not complete its own generation. Both must be clear for
                # the next step's lockstep start to advance everywhere.
                abort_zorl_generation_all(
                    control_infer_urls,
                    session_id=args.session_id,
                    generation_id=gen_id,
                )
                if getattr(args, "ps_url", None):
                    abort_zorl_generation_all(
                        [args.ps_url],
                        session_id=args.session_id,
                        generation_id=gen_id,
                    )
        else:
            apply_result = apply_zorl_rewards_all(
                control_infer_urls,
                session_id=args.session_id,
                generation_id=gen_id,
                candidate_rewards=rewards_for_update,
                lr=_lr_for_step(step),
                # <=0 disables norm clipping (required for fresh_ab: the paired
                # outer-product update folds chunk-by-chunk, no global pre-norm).
                max_update_norm=(args.max_update_norm if args.max_update_norm > 0 else None),
                momentum=args.momentum,
                momentum_window=args.momentum_window,
            )
        apply_s = time.time() - apply_t0
        # EggRoll merge-every-step: fold the just-applied parent LoRA delta into the resident
        # base weights + re-init LoRA-A, so the accumulated update is full-rank (not rank-capped).
        # PS mode folds the fp32 master DIRECTLY into the base every step (already full-rank,
        # no LoRA accumulation to merge), so the merge is a no-op there — skip it.
        if (
            not getattr(args, "ps_url", None)
            and getattr(args, "merge_every_steps", 0) > 0
            and (step + 1) % args.merge_every_steps == 0
        ):
            merge_t0 = time.time()
            merge_res = merge_zorl_parent_into_base_all(control_infer_urls, session_id=args.session_id)
            # In FP8-native accumulate-in-adapter mode (server env
            # XORL_ZORL_ACCUMULATE_IN_ADAPTER=1) this merge is the ONE re-quant
            # per K steps: it drains each weight's persistent bf16 ES accumulator
            # into the pristine FP8 base (no parent A re-init). The server
            # metadata flags which path ran; the KV cache flush in
            # merge_zorl_parent_into_base_all covers both.
            merge_md = merge_res.get("metadata", merge_res) or {}
            if merge_md.get("accumulate_in_adapter"):
                print(
                    f"  FP8 accum->base flush @ step {step+1}: "
                    f"success={merge_res.get('success')} "
                    f"folded_weights={merge_md.get('folded_weights')} "
                    f"accum_norm={merge_md.get('accum_norm')} "
                    f"({time.time()-merge_t0:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  EggRoll merge parent->base @ step {step+1}: success={merge_res.get('success')} "
                    f"({time.time()-merge_t0:.1f}s)",
                    flush=True,
                )
        metrics = apply_result.get("metrics", {})
        update_extras = []
        for key, fmt in (
            ("pair_delta_mean", ".4f"),
            ("pair_delta_std", ".4f"),
            ("unclipped_update_norm", ".2f"),
            ("grad_norm", ".2f"),
            ("update_clip_scale", ".4f"),
        ):
            if key in metrics:
                update_extras.append(f"{key}={metrics.get(key, 0):{fmt}}")
        for key in ("zero_score_pairs", "dropped_pairs"):
            value = metrics.get(key, apply_result.get(key))
            if value is not None:
                update_extras.append(f"{key}={int(value)}")
        if "score_normalization" in metrics:
            update_extras.append(f"score_normalization={metrics['score_normalization']}")
        if transform_metrics.get("update_strategy"):
            update_extras.append(f"update_strategy={transform_metrics['update_strategy']}")
        for key, fmt in (
            ("parent_train_reward_mean", ".4f"),
            ("transformed_reward_mean", ".4f"),
            ("transformed_reward_std", ".4f"),
        ):
            if key in transform_metrics:
                update_extras.append(f"{key}={transform_metrics[key]:{fmt}}")
        update_extra_text = " ".join(update_extras)

        # Optional probe + rollback. Probe interval > 1 reduces eval cost but
        # rollback only ever fires on probe gens.
        probe_summary = ""
        if args.probe_interval > 0 and ((step + 1) % args.probe_interval == 0 or step == args.steps - 1):
            probe = probe_parent(reward_infer_url, parent_lora_name=args.parent_lora_name, examples=eval_examples, task=task, args=args, tokenizer=tokenizer)
            probe_reward = probe.get("reward_mean", 0.0)
            probe_exact = probe.get("exact_match_mean", 0.0)
            # Compact extras: any task-emitted metric beyond reward/exact.
            extras = " ".join(f"{k.removesuffix('_mean')}={probe[k]:.3f}" for k in sorted(probe.keys()) if k.endswith("_mean") and k not in ("reward_mean", "exact_match_mean"))
            probe_summary = f", probe_reward={probe_reward:.4f} exact_rate={probe_exact:.4f}"
            if extras:
                probe_summary += f" [{extras}]"
            if args.elitist_rollback:
                # PS mode: snapshot/restore target the PS (it holds the fp32
                # master = the base, where the accumulated update lives). On a
                # restore the PS rolls back its master and re-broadcasts the
                # restored parent to the replicas (a full sparse diff vs their
                # current state). Legacy mode snapshots the per-replica adapter.
                rollback_urls = [args.ps_url] if getattr(args, "ps_url", None) else control_infer_urls
                if probe_reward > best_reward:
                    best_reward = probe_reward
                    best_blob = probe
                    snapshot_zorl_parent_all(rollback_urls, session_id=args.session_id, snapshot_id=args.snapshot_id)
                    maybe_export("best", snapshot_id=args.snapshot_id)
                    probe_summary += "  [best updated → snapshot]"
                else:
                    restore_zorl_parent_all(rollback_urls, session_id=args.session_id, snapshot_id=args.snapshot_id)
                    if getattr(args, "ps_url", None):
                        # Re-broadcast the restored master to the replicas so they
                        # serve the rolled-back parent (the restore moved the PS
                        # master + served bf16; the replicas need the diff).
                        ps_rebroadcast(args.ps_url, control_infer_urls, session_id=args.session_id)
                    probe_summary += f"  [worse than best={best_reward:.4f} → restored]"
            elif probe_reward > best_reward:
                best_reward = probe_reward
                best_blob = probe
                probe_summary += "  [best updated]"

        if args.export_interval > 0 and ((step + 1) % args.export_interval == 0 or step == args.steps - 1):
            maybe_export(f"step-{step+1:06d}")

        print(
            f"  step {step+1}/{args.steps}: "
            f"reward_mean={mean_cand_reward:.4f} best_cand={best_cand_reward:.4f} "
            f"update_norm={metrics.get('update_norm', 0):.2f} "
            f"{update_extra_text + ' ' if update_extra_text else ''}"
            f"used_pairs={apply_result.get('used_pairs', 0)} "
            f"t_score={score_s:.1f}s t_apply={apply_s:.1f}s{probe_summary}"
        )

    total = time.time() - t0
    maybe_export("final")
    print(f"\n[done] {args.steps} steps in {total:.1f}s")
    print(f"[done] best probe: reward_mean={best_reward:.4f} exact_count={best_blob.get('exact_count', 0):.1f}/{best_blob.get('n', 0)}")
    print(f"[done] parent_lora_name={args.parent_lora_name} session_id={args.session_id}")


if __name__ == "__main__":
    main()
