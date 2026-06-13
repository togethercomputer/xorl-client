"""Cluster-facing ZORL microbenchmark against live SGL/SMG services.

This is the benchmark the performance handoff runbook asks for. It runs against
an existing ZORL SGLang pool + SMG router and emits one JSONL record per trial
with enough metrics to hillclimb score and apply throughput. It is the
eval/hillclimb target to use *before* changing the production loop.

  WARNING: score/apply trials drive the SGL pool. Do NOT point this at the pool
  while a production ZORL run is using it -- it will collide with the live
  session and can restart workers. Run it in a dedicated window, against a fresh
  --session-id, on a pool that is otherwise idle.

Three modes (see runbook "Microbenchmark First"):

  gen    isolate /start_zorl_generation + virtual candidate registration.
  score  isolate teacher-forced /generate throughput under candidate LoRAs.
  apply  isolate /apply_zorl_rewards (the path optimized by the GPU noise
         fast path; set XORL_ZORL_NOISE_DEVICE=gpu on the SGL pods to measure
         the speedup, then compare apply_wall_s here).

Examples:
  # apply benchmark (the cheapest to isolate; no /generate needed)
  python -m experiments.zorl.standalone.bench_zorl_sglang apply \
    --control-urls http://sgl-0:30000,http://sgl-1:30000,... \
    --session-id bench-$(whoami) --num-pairs 512 --num-shards 16 \
    --parent-lora /path/to/exports/best --out bench.jsonl

  # generation sweep
  python -m experiments.zorl.standalone.bench_zorl_sglang gen \
    --control-urls ... --session-id bench --pairs-per-shard 1,2,4,8,16,32

  # score throughput (synthetic fixed-length traces)
  python -m experiments.zorl.standalone.bench_zorl_sglang score \
    --control-urls ... --smg-url http://smg:8080 --session-id bench \
    --num-pairs 512 --num-shards 16 --train-size 128 \
    --teacher-forced-batch-size 8 --score-max-workers 64 \
    --score-max-workers-per-owner 4 --route owner_via_smg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from experiments.zorl.standalone import zorl_client as zc


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _now_id() -> str:
    return uuid.uuid4().hex[:8]


def sgl_restart_counts(namespace: str, selector: str = "app=zorl-ar-sglang") -> Dict[str, int]:
    """Best-effort per-pod restartCount via kubectl. {} if kubectl unavailable."""
    try:
        out = subprocess.run(
            [
                "kubectl", "get", "pods", "-n", namespace, "-l", selector,
                "-o",
                "jsonpath={range .items[*]}{.metadata.name} "
                "{.status.containerStatuses[0].restartCount}{\"\\n\"}{end}",
            ],
            capture_output=True, text=True, timeout=30,
        )
        counts: Dict[str, int] = {}
        for line in out.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 2:
                counts[parts[0]] = int(parts[1])
        return counts
    except Exception:
        return {}


def _restart_delta(before: Dict[str, int], after: Dict[str, int]) -> int:
    return sum(max(0, after.get(k, 0) - v) for k, v in before.items())


def write_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def setup_generation(
    control_urls: List[str],
    *,
    session_id: str,
    parent_lora_name: str,
    parent_lora_path: str,
    num_pairs: int,
    num_shards: int,
    b_sigma: float,
    seed: int,
    perturbation_mode: str,
    preload_candidates: bool,
) -> dict:
    """Load parent, start session, start a pair_shard generation. Returns the
    combined generation result (with global candidate list + owner_url)."""
    zc.load_lora_adapter_all(control_urls, parent_lora_name, parent_lora_path)
    zc.start_zorl_session_all(
        control_urls,
        session_id=session_id,
        parent_lora_name=parent_lora_name,
        num_pairs=num_pairs,
        b_sigma=b_sigma,
        seed=seed,
        perturbation_mode=perturbation_mode,
    )
    gen = zc.start_zorl_generation_all(
        control_urls,
        session_id=session_id,
        preload_candidates=preload_candidates,
        num_pairs=num_pairs,
        population_sharding="pair_shard",
        num_shards=num_shards,
    )
    return gen


def teardown(control_urls: List[str], *, session_id: str, generation_id: str, parent_lora_name: str) -> None:
    for url in control_urls:
        try:
            zc._post(
                url, "/abort_zorl_generation",
                {"session_id": session_id, "generation_id": generation_id},
                timeout=120.0,
            )
        except Exception:
            pass
    for url in control_urls:
        try:
            zc.unload_lora_adapter(url, parent_lora_name)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Mode 3: apply (most isolatable; directly measures the GPU fast path)
# --------------------------------------------------------------------------


def run_apply(args, control_urls: List[str]) -> None:
    base = {
        "mode": "apply",
        "model": args.model,
        "replicas": len(control_urls),
        "num_pairs": args.num_pairs,
        "num_shards": args.num_shards,
        "perturbation_mode": args.perturbation_mode,
        "score_normalization": args.score_normalization,
        "learning_rate": args.lr,
        "max_update_norm": args.max_update_norm,
        "noise_device_hint": args.noise_device_hint,
    }
    for trial in range(args.trials):
        session_id = f"{args.session_id}-apply-{_now_id()}"
        restarts_before = sgl_restart_counts(args.namespace, args.sgl_selector)
        gen = setup_generation(
            control_urls,
            session_id=session_id,
            parent_lora_name=f"{session_id}/parent",
            parent_lora_path=args.parent_lora,
            num_pairs=args.num_pairs,
            num_shards=args.num_shards,
            b_sigma=args.b_sigma,
            seed=args.seed,
            perturbation_mode=args.perturbation_mode,
            preload_candidates=False,
        )
        generation_id = gen["generation_id"]
        candidates = gen["candidates"]
        # Deterministic synthetic rewards for every global candidate: a smooth
        # function of perturbation_index/direction so pair deltas are nonzero.
        rewards = []
        for c in candidates:
            pidx = int(c["perturbation_index"])
            sign = 1.0 if str(c["direction"]) == "positive" else -1.0
            reward_mean = 0.5 + 0.01 * sign * ((pidx % 17) - 8) / 8.0
            rewards.append(
                {
                    "candidate_id": c["candidate_id"],
                    "reward_mean": reward_mean,
                    "num_rollouts": 1,
                    "_zorl_score_normalization": args.score_normalization,
                }
            )
        try:
            t0 = time.time()
            if args.apply_fanout == "parallel":
                apply_result = zc.apply_zorl_rewards_all(
                    control_urls,
                    session_id=session_id,
                    generation_id=generation_id,
                    candidate_rewards=rewards,
                    lr=args.lr,
                    max_update_norm=args.max_update_norm,
                )
            else:
                results = []
                for url in control_urls:
                    results.append(
                        zc.apply_zorl_rewards(
                            url,
                            session_id=session_id,
                            generation_id=generation_id,
                            candidate_rewards=rewards,
                            lr=args.lr,
                            max_update_norm=args.max_update_norm,
                        )
                    )
                zc._validate_apply_results_agree(control_urls, results)
                for url in control_urls:
                    zc.flush_inference_cache(url)
                apply_result = results[0]
            apply_wall = time.time() - t0
            metrics = apply_result.get("metrics", {}) or {}
            restarts_after = sgl_restart_counts(args.namespace, args.sgl_selector)
            record = {
                **base,
                "trial": trial,
                "session_id": session_id,
                "apply_fanout": args.apply_fanout,
                "apply_wall_s": round(apply_wall, 3),
                "used_pairs": apply_result.get("used_pairs"),
                "dropped_pairs": apply_result.get("dropped_pairs"),
                "zero_score_pairs": metrics.get("zero_score_pairs"),
                "update_norm": metrics.get("update_norm"),
                "grad_norm": metrics.get("grad_norm"),
                "update_clip_scale": metrics.get("update_clip_scale"),
                "pair_delta_mean": metrics.get("pair_delta_mean"),
                "pair_delta_std": metrics.get("pair_delta_std"),
                "restart_delta": _restart_delta(restarts_before, restarts_after),
                "ok": True,
            }
        except Exception as e:
            record = {**base, "trial": trial, "session_id": session_id, "ok": False, "error": str(e)}
        finally:
            teardown(
                control_urls,
                session_id=session_id,
                generation_id=gen.get("generation_id", ""),
                parent_lora_name=f"{session_id}/parent",
            )
        write_jsonl(args.out, record)


# --------------------------------------------------------------------------
# Mode 1: generation
# --------------------------------------------------------------------------


def run_gen(args, control_urls: List[str]) -> None:
    pairs_list = [int(x) for x in str(args.pairs_per_shard).split(",") if x]
    for preload in args.preload_grid:
        for pps in pairs_list:
            num_pairs = pps * len(control_urls)
            session_id = f"{args.session_id}-gen-{_now_id()}"
            restarts_before = sgl_restart_counts(args.namespace, args.sgl_selector)
            record = {
                "mode": "gen",
                "model": args.model,
                "replicas": len(control_urls),
                "pairs_per_shard": pps,
                "num_pairs": num_pairs,
                "num_shards": len(control_urls),
                "preload_candidates": preload,
                "session_id": session_id,
            }
            gen = None
            try:
                t0 = time.time()
                gen = setup_generation(
                    control_urls,
                    session_id=session_id,
                    parent_lora_name=f"{session_id}/parent",
                    parent_lora_path=args.parent_lora,
                    num_pairs=num_pairs,
                    num_shards=len(control_urls),
                    b_sigma=args.b_sigma,
                    seed=args.seed,
                    perturbation_mode=args.perturbation_mode,
                    preload_candidates=preload,
                )
                gen_wall = time.time() - t0
                restarts_after = sgl_restart_counts(args.namespace, args.sgl_selector)
                record.update(
                    {
                        "gen_wall_s": round(gen_wall, 3),
                        "global_candidate_count": len(gen.get("candidates", [])),
                        "generation_id": gen.get("generation_id"),
                        "restart_delta": _restart_delta(restarts_before, restarts_after),
                        "ok": True,
                    }
                )
            except Exception as e:
                record.update({"ok": False, "error": str(e)})
            finally:
                if gen is not None:
                    teardown(
                        control_urls,
                        session_id=session_id,
                        generation_id=gen.get("generation_id", ""),
                        parent_lora_name=f"{session_id}/parent",
                    )
            write_jsonl(args.out, record)


# --------------------------------------------------------------------------
# Mode 2: score (synthetic fixed-length teacher-forced traces)
# --------------------------------------------------------------------------


def _score_one(url_or_smg: str, *, headers, input_ids_batch, lora_path, owner_sema, logprob_start_len=0) -> tuple[int, float]:
    payload = {
        "input_ids": input_ids_batch,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": logprob_start_len,
        "lora_path": lora_path,
    }
    # Gate concurrency per owning SGL replica (the documented stability cap:
    # >4 in-flight candidate requests per replica restarts the Triton LoRA path).
    with owner_sema:
        t0 = time.time()
        try:
            zc._post(url_or_smg, "/generate", payload, timeout=300.0, headers=headers)
            return 200, time.time() - t0
        except Exception:
            return 500, time.time() - t0


def run_score(args, control_urls: List[str]) -> None:
    session_id = f"{args.session_id}-score-{_now_id()}"
    restarts_before = sgl_restart_counts(args.namespace, args.sgl_selector)
    gen = setup_generation(
        control_urls,
        session_id=session_id,
        parent_lora_name=f"{session_id}/parent",
        parent_lora_path=args.parent_lora,
        num_pairs=args.num_pairs,
        num_shards=args.num_shards,
        b_sigma=args.b_sigma,
        seed=args.seed,
        perturbation_mode=args.perturbation_mode,
        preload_candidates=False,
    )
    generation_id = gen["generation_id"]
    candidates = gen["candidates"]
    # synthetic fixed-length input ids (length ~ production teacher-forced 238)
    seq = list(range(1, args.seq_len + 1))
    batch = [seq for _ in range(args.teacher_forced_batch_size)]
    n_batches_per_candidate = max(1, (args.train_size + args.teacher_forced_batch_size - 1) // args.teacher_forced_batch_size)

    # build (candidate, batch_index) jobs in owner-round-robin order
    jobs = []
    for bi in range(n_batches_per_candidate):
        for c in candidates:
            jobs.append((c, bi))

    owner_sema: Dict[str, threading.Semaphore] = defaultdict(
        lambda: threading.Semaphore(args.score_max_workers_per_owner)
    )
    status_counts = {"200": 0, "5xx": 0}
    latencies: List[float] = []

    def route_for(c) -> tuple[str, dict]:
        owner = str(c["owner_url"])
        if args.route == "owner_via_smg":
            return args.smg_url, {"X-SMG-Target-Worker": owner}
        return owner, {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.score_max_workers) as pool:
        futures = []
        for c, _bi in jobs:
            url, headers = route_for(c)
            futures.append(
                pool.submit(
                    _score_one, url, headers=headers, input_ids_batch=batch,
                    lora_path=c["lora_name"], owner_sema=owner_sema[str(c["owner_url"])],
                    logprob_start_len=args.logprob_start_len,
                )
            )
        for fut in as_completed(futures):
            code, lat = fut.result()
            latencies.append(lat)
            status_counts["200" if code == 200 else "5xx"] += 1
    score_wall = time.time() - t0

    restarts_after = sgl_restart_counts(args.namespace, args.sgl_selector)
    score_sequences = len(jobs) * args.teacher_forced_batch_size
    score_input_tokens = score_sequences * args.seq_len
    record = {
        "mode": "score",
        "model": args.model,
        "replicas": len(control_urls),
        "num_pairs": args.num_pairs,
        "num_shards": args.num_shards,
        "train_size": args.train_size,
        "teacher_forced_batch_size": args.teacher_forced_batch_size,
        "score_max_workers": args.score_max_workers,
        "score_max_workers_per_owner": args.score_max_workers_per_owner,
        "route": args.route,
        "seq_len": args.seq_len,
        "logprob_start_len": args.logprob_start_len,
        "score_wall_s": round(score_wall, 3),
        "score_sequences": score_sequences,
        "score_input_tokens": score_input_tokens,
        "score_tokens_per_s": round(score_input_tokens / max(score_wall, 1e-9), 1),
        "http_200": status_counts["200"],
        "http_5xx": status_counts["5xx"],
        "restart_delta": _restart_delta(restarts_before, restarts_after),
        "session_id": session_id,
        "ok": status_counts["5xx"] == 0 and _restart_delta(restarts_before, restarts_after) == 0,
    }
    teardown(control_urls, session_id=session_id, generation_id=generation_id, parent_lora_name=f"{session_id}/parent")
    write_jsonl(args.out, record)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gen", "score", "apply"])
    ap.add_argument("--control-urls", required=True, help="comma/space list of DIRECT SGL control URLs")
    ap.add_argument("--session-id", required=True, help="fresh session id (must NOT collide with a live run)")
    ap.add_argument("--parent-lora", required=True, help="parent LoRA adapter dir (e.g. exports/best)")
    ap.add_argument("--smg-url", default="", help="SMG router URL (required for --route owner_via_smg)")
    ap.add_argument("--namespace", default="apanda")
    ap.add_argument(
        "--sgl-selector", default="app=zorl-ar-sglang",
        help="kubectl label selector for restart-count tracking (point at the pool under test)",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--out", default="", help="append JSONL records here")

    ap.add_argument("--num-pairs", type=int, default=512)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--b-sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturbation-mode", default="b_only", choices=["b_only", "a_and_b"])

    # apply
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-update-norm", type=float, default=20000.0)
    ap.add_argument("--score-normalization", default="standard", choices=["standard", "none"])
    ap.add_argument("--apply-fanout", default="parallel", choices=["parallel", "sequential"])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument(
        "--noise-device-hint", default="unknown",
        help="record-only: what XORL_ZORL_NOISE_DEVICE the SGL pods are running (cpu/gpu)",
    )

    # gen
    ap.add_argument("--pairs-per-shard", default="1,2,4,8,16,32")
    ap.add_argument("--preload", default="false", help="false,true or both")

    # score
    ap.add_argument("--train-size", type=int, default=128)
    ap.add_argument("--teacher-forced-batch-size", type=int, default=8)
    ap.add_argument("--score-max-workers", type=int, default=64)
    ap.add_argument("--score-max-workers-per-owner", type=int, default=4)
    ap.add_argument("--route", default="owner_via_smg", choices=["owner_via_smg", "direct"])
    ap.add_argument("--seq-len", type=int, default=238)
    ap.add_argument(
        "--logprob-start-len", type=int, default=0,
        help="logprob_start_len for teacher-forced /generate; >0 trims logprob "
        "computation to the target suffix (set to seq_len - target_tokens - margin)",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    control_urls = zc._url_list(args.control_urls)
    if not control_urls:
        raise SystemExit("no --control-urls provided")
    if args.mode == "score" and args.route == "owner_via_smg" and not args.smg_url:
        raise SystemExit("--route owner_via_smg requires --smg-url")
    if args.mode == "gen":
        if str(args.preload).lower() == "both":
            args.preload_grid = [False, True]
        else:
            args.preload_grid = [str(args.preload).lower() == "true"]
        run_gen(args, control_urls)
    elif args.mode == "score":
        run_score(args, control_urls)
    else:
        run_apply(args, control_urls)


if __name__ == "__main__":
    main()
