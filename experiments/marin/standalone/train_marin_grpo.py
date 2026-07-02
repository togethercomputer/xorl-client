from __future__ import annotations

import argparse
import concurrent.futures
import gc
import inspect
import json
import os
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import requests
import xorl_client
import xorl_client.client.chunked_helpers as xorl_chunked_helpers
import xorl_client.client.training_client as xorl_training_client_module
from xorl_client import ServiceClient, TrainingClient, types
from xorl_client.client.chunked_helpers import combine_fwd_bwd_output_results, estimate_datum_bytes

from experiments.marin.standalone.advantages import (
    DEFAULT_DRGRPO_LOSS_PARAMS,
    RolloutRecord,
    build_drgrpo_datum,
    compute_group_advantages,
)
from experiments.marin.standalone.dataset import MathExample, load_examples
from experiments.marin.standalone.length_penalty import LengthPenaltyConfig, shaped_reward
from experiments.marin.standalone.prompts import encode_forced_thinking_prefix
from experiments.marin.standalone.tasks.verifier import grade_reference_final_answer


MODEL_ID = "default"
DEFAULT_XORL_CLIENT_MAX_CHUNK_LEN = xorl_chunked_helpers.MAX_CHUNK_LEN
DEFAULT_XORL_CLIENT_MAX_CHUNK_BYTES = xorl_chunked_helpers.MAX_CHUNK_BYTES_COUNT
DEFAULT_XORL_CLIENT_MIN_TAIL_CHUNK_LEN = 256
TRANSIENT_SAMPLING_STATUS_CODES = {500, 502, 503, 504}


class _TransientSamplingError(RuntimeError):
    pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marin #6279 dense-Qwen3 GRPO driver for xorl server mode.")
    parser.add_argument("--train-url", required=True, help="xorl training server base URL")
    parser.add_argument(
        "--inference-url",
        required=True,
        help="Direct SGLang URL(s) used for xorl endpoint registration and P2P weight sync.",
    )
    parser.add_argument(
        "--sampling-url",
        default=None,
        help="SGLang/SMG URL(s) used for rollout /generate calls. Defaults to --inference-url.",
    )
    parser.add_argument("--model", required=True, help="HF repo or local path for tokenizer/base model")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL metrics and artifacts")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default="rlvr_math_7500", choices=["rlvr_math_7500", "math500", "aime24", "gsm8k"])
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument(
        "--sampling-workers-per-replica",
        type=int,
        default=16,
        help="Maximum concurrent /generate requests per SGLang replica.",
    )
    parser.add_argument(
        "--sampling-max-workers",
        type=int,
        default=None,
        help="Override total concurrent /generate requests, useful when --sampling-url is one SMG router.",
    )
    parser.add_argument(
        "--sampling-request-timeout-s",
        type=float,
        default=float(os.environ.get("MARIN_SAMPLING_REQUEST_TIMEOUT_S", "300")),
        help="Per-attempt HTTP timeout for one SGLang /generate request.",
    )
    parser.add_argument(
        "--sampling-request-max-attempts",
        type=int,
        default=int(os.environ.get("MARIN_SAMPLING_REQUEST_MAX_ATTEMPTS", "2")),
        help="Maximum attempts for one SGLang /generate request before it is counted as failed.",
    )
    parser.add_argument(
        "--sampling-future-idle-timeout-s",
        type=float,
        default=float(os.environ.get("MARIN_SAMPLING_FUTURE_IDLE_TIMEOUT_S", "420")),
        help="Maximum rollout collector wall time with no completed /generate future before failing or dropping tail requests.",
    )
    parser.add_argument(
        "--future-timeout-s",
        "--future-timeout",
        dest="future_timeout_s",
        type=float,
        default=float(os.environ.get("MARIN_FUTURE_TIMEOUT_S", "7200")),
        help="Maximum seconds to wait for one xorl async future before failing the driver.",
    )
    parser.add_argument(
        "--sync-timeout-s",
        type=float,
        default=float(os.environ.get("MARIN_SYNC_TIMEOUT_S", "1800")),
        help="HTTP and future timeout for one trainer-to-sampler weight sync.",
    )
    parser.add_argument(
        "--prefetch-timeout-s",
        type=float,
        default=float(os.environ.get("MARIN_PREFETCH_TIMEOUT_S", "7200")),
        help="Maximum seconds to wait for a pipeline rollout prefetch before failing the driver.",
    )
    parser.add_argument(
        "--max-rollout-failures",
        type=int,
        default=int(os.environ.get("MARIN_MAX_ROLLOUT_FAILURES", "0")),
        help="Maximum failed rollout requests to tolerate in one step. Zero makes any persistent request failure fatal.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-generate-tokens", type=int, default=3584)
    parser.add_argument("--max-model-tokens", type=int, default=int(os.environ.get("MARIN_MAX_MODEL_TOKENS", "4096")))
    parser.add_argument(
        "--context-token-reserve",
        type=int,
        default=int(os.environ.get("MARIN_CONTEXT_TOKEN_RESERVE", "1")),
        help="Reserved context tokens when deriving per-request generation caps for SGLang.",
    )
    parser.add_argument(
        "--prompt-overlength-policy",
        choices=["skip", "trim-left"],
        default=os.environ.get("MARIN_PROMPT_OVERLENGTH_POLICY", "skip"),
        help="How to handle prompts longer than --max-prompt-tokens; trim-left keeps the tail tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--logprob-temperature",
        type=float,
        default=None,
        help=(
            "Temperature used by xorl when recomputing selected-token logprobs for ratios. "
            "Defaults to --temperature for behavior-logprob training. Set to 1.0 only when "
            "SGLang returns raw/original logprobs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-seed-base", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--drgrpo-ratio-type", choices=["token", "sequence"], default="sequence")
    parser.add_argument("--drgrpo-clip-low", type=float, default=0.2)
    parser.add_argument("--drgrpo-clip-high", type=float, default=0.2)
    parser.add_argument(
        "--drgrpo-num-chunks",
        type=int,
        default=int(os.environ.get("MARIN_DRGRPO_NUM_CHUNKS", str(DEFAULT_DRGRPO_LOSS_PARAMS["num_chunks"]))),
        help="Number of chunks used inside the DR-GRPO selected-token CE/logprob computation.",
    )
    parser.add_argument("--sync-master-port", type=int, default=29600)
    parser.add_argument("--sync-buffer-size-mb", type=int, default=1024)
    parser.add_argument("--inference-world-size", type=int, default=1, help="SGLang TP world size at the endpoint")
    parser.add_argument("--skip-initial-sync", action="store_true")
    parser.add_argument("--skip-weight-sync", action="store_true")
    parser.add_argument("--sync-cache-invalidation-mode", choices=["auto", "flush", "none"], default="none")
    parser.add_argument("--sync-pause-mode", choices=["retract", "abort", "in_place"], default="in_place")
    parser.add_argument(
        "--pipeline-rl",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MARIN_PIPELINE_RL"),
        help=(
            "Prefetch the next rollout while the current batch trains. The driver waits for the prefetch to "
            "finish before RDMA weight sync so SGLang does not update weights mid-request."
        ),
    )
    parser.add_argument(
        "--sync-weight-version-prefix",
        default=None,
        help="Prefix for per-policy SGLang weight versions/cache namespaces. Defaults to --run-id.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Save xorl DCP state every N updates.")
    parser.add_argument("--checkpoint-prefix", default=None, help="Checkpoint name prefix. Defaults to --run-id.")
    parser.add_argument("--save-final-checkpoint", action="store_true", help="Save final xorl DCP state.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument(
        "--wandb-rollout-log-interval",
        type=int,
        default=int(os.environ.get("MARIN_WANDB_ROLLOUT_LOG_INTERVAL", "128")),
        help="Log rollout progress to W&B every N completed samples; 0 disables progress logs.",
    )
    parser.add_argument(
        "--rollout-progress-log-interval",
        type=int,
        default=int(os.environ.get("MARIN_ROLLOUT_PROGRESS_LOG_INTERVAL", "128")),
        help="Write local rollout_progress JSONL events every N scored samples; 0 disables local progress logs.",
    )
    parser.add_argument(
        "--sample-text-log-limit",
        type=int,
        default=int(os.environ.get("MARIN_SAMPLE_TEXT_LOG_LIMIT", "8")),
        help="Write decoded rollout_sample_text records for the first N samples per policy step; 0 disables.",
    )
    parser.add_argument(
        "--shuffle-rollout-requests",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MARIN_SHUFFLE_ROLLOUT_REQUESTS", True),
        help="Deterministically shuffle rollout request submission order before sending through SMG/SGLang.",
    )
    parser.add_argument(
        "--rollout-score-workers",
        type=int,
        default=int(os.environ.get("MARIN_ROLLOUT_SCORE_WORKERS", "64")),
        help=(
            "Worker threads used for reward/verifier processing after /generate returns. "
            "Keeping this separate from HTTP workers prevents verifier work from starving SMG."
        ),
    )
    parser.add_argument(
        "--driver-gc",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MARIN_DRIVER_GC", True),
        help="Drop large rollout/train objects and run Python GC at phase boundaries.",
    )
    parser.add_argument(
        "--force-nonzero-advantages-for-smoke",
        action="store_true",
        help="If all rewards tie in a tiny smoke run, force a balanced synthetic advantage vector.",
    )
    parser.add_argument(
        "--xorl-client-max-chunk-len",
        type=int,
        default=int(os.environ.get("XORL_CLIENT_MAX_CHUNK_LEN", str(DEFAULT_XORL_CLIENT_MAX_CHUNK_LEN))),
        help="Max datums per xorl-client forward_backward chunk.",
    )
    parser.add_argument(
        "--xorl-client-max-chunk-bytes",
        type=int,
        default=int(os.environ.get("XORL_CLIENT_MAX_CHUNK_BYTES_COUNT", str(DEFAULT_XORL_CLIENT_MAX_CHUNK_BYTES))),
        help="Max estimated bytes per xorl-client forward_backward chunk.",
    )
    parser.add_argument(
        "--xorl-client-min-tail-chunk-len",
        type=int,
        default=int(os.environ.get("XORL_CLIENT_MIN_TAIL_CHUNK_LEN", str(DEFAULT_XORL_CLIENT_MIN_TAIL_CHUNK_LEN))),
        help=(
            "Merge the final forward_backward chunk into the previous chunk when it has fewer datums than this. "
            "Set to 0 to preserve raw xorl-client chunk boundaries."
        ),
    )
    parser.add_argument(
        "--xorl-client-sequential-chunks",
        action="store_true",
        default=_env_flag("XORL_CLIENT_SEQUENTIAL_CHUNKS"),
        help="Execute xorl-client forward_backward chunks one at a time instead of dispatching them concurrently.",
    )
    parser.add_argument(
        "--direct-train-server-futures",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MARIN_DIRECT_TRAIN_SERVER_FUTURES", True),
        help=(
            "Submit trainer operations directly to /api/v1/* and poll /retrieve_future explicitly. "
            "This keeps request ids visible and avoids opaque xorl-client wrapped-future stalls."
        ),
    )
    parser.add_argument("--remove-endpoint-on-exit", action="store_true")
    return parser.parse_args()


def _url_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Expected URL with host and port, got {url!r}")
    return parsed.hostname, parsed.port


def _parse_url_list(raw_urls: str, *, arg_name: str) -> list[str]:
    urls = [url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip()]
    if not urls:
        raise ValueError(f"{arg_name} must contain at least one URL")
    for url in urls:
        _url_host_port(url)
    return urls


def _parse_inference_urls(raw_urls: str) -> list[str]:
    return _parse_url_list(raw_urls, arg_name="--inference-url")


def _select_step_examples(examples: list[MathExample], *, step: int, count: int, seed: int) -> list[MathExample]:
    if count <= 0:
        raise ValueError("prompts-per-step must be positive")
    if not examples:
        raise ValueError("No examples loaded")
    rng = random.Random(seed + step)
    if count <= len(examples):
        return rng.sample(examples, count)
    return [examples[rng.randrange(len(examples))] for _ in range(count)]


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _current_rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _log_driver_memory(
    metrics_path: Path,
    wandb_run,
    *,
    step: int | None,
    phase: str,
    policy_step: int | None = None,
    force_gc: bool = False,
) -> None:
    gc_collected = gc.collect() if force_gc else None
    payload = {
        "event": "driver_memory",
        "step": step,
        "policy_step": policy_step,
        "phase": phase,
        "rss_mb": _current_rss_mb(),
        "gc_collected": gc_collected,
    }
    _write_jsonl(metrics_path, payload)
    _wandb_log(
        wandb_run,
        {
            f"driver_memory/{phase}/rss_mb": payload["rss_mb"],
            f"driver_memory/{phase}/gc_collected": gc_collected,
        },
        policy_step=policy_step,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jsonable(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _make_loss_params(args: argparse.Namespace) -> dict:
    logprob_temperature = args.logprob_temperature if args.logprob_temperature is not None else args.temperature
    params = dict(DEFAULT_DRGRPO_LOSS_PARAMS)
    params.update(
        {
            "ratio_type": args.drgrpo_ratio_type,
            "clip_low": args.drgrpo_clip_low,
            "clip_high": args.drgrpo_clip_high,
            "beta": 0.0,
            "num_chunks": args.drgrpo_num_chunks,
            "logprob_temperature": logprob_temperature,
        }
    )
    return params


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_head_for_path(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_provenance() -> dict[str, str | None]:
    xorl_client_file = Path(inspect.getfile(xorl_client)).resolve()
    xorl_client_root = xorl_client_file.parents[1]
    return {
        "cwd": os.getcwd(),
        "repo_root": str(_repo_root()),
        "driver_file": str(Path(__file__).resolve()),
        "verifier_file": inspect.getsourcefile(grade_reference_final_answer),
        "git_head": _git_head(),
        "xorl_client_file": str(xorl_client_file),
        "xorl_client_git_head": _git_head_for_path(xorl_client_root),
    }


def _verifier_self_check() -> dict[str, Any]:
    completion = r"reasoning\nAnswer: \boxed{0}"
    gold_answer = "0"
    main_result = grade_reference_final_answer(completion, gold_answer)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        worker_result = executor.submit(grade_reference_final_answer, completion, gold_answer).result()

    payload = {
        "completion": completion,
        "gold_answer": gold_answer,
        "main_thread": _jsonable(main_result),
        "worker_thread": _jsonable(worker_result),
    }
    if not main_result.correct or not worker_result.correct:
        raise RuntimeError(f"Verifier startup self-check failed: {payload}")
    return payload


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_run_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    inference_urls: list[str],
    sampling_urls: list[str],
    num_examples: int,
    prompt_filter: dict[str, int | float | bool],
    loss_params: dict,
    length_config: LengthPenaltyConfig,
    client_chunking: dict[str, int | bool],
) -> dict:
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "git_head": _git_head(),
        "args": vars(args),
        "sync_inference_urls": inference_urls,
        "sampling_urls": sampling_urls,
        "sampling_uses_router": sampling_urls != inference_urls,
        "num_examples": num_examples,
        "prompt_filter": prompt_filter,
        "target_batch": {
            "prompts_per_step": args.prompts_per_step,
            "samples_per_prompt": args.samples_per_prompt,
            "rollouts_per_step": args.prompts_per_step * args.samples_per_prompt,
            "steps": args.steps,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_generate_tokens": args.max_generate_tokens,
            "max_model_tokens": args.max_model_tokens,
            "context_token_reserve": args.context_token_reserve,
            "prompt_overlength_policy": args.prompt_overlength_policy,
            "shuffle_rollout_requests": args.shuffle_rollout_requests,
        },
        "sampling_liveness": {
            "request_timeout_s": args.sampling_request_timeout_s,
            "request_max_attempts": args.sampling_request_max_attempts,
            "future_idle_timeout_s": args.sampling_future_idle_timeout_s,
            "max_rollout_failures": args.max_rollout_failures,
            "workers_per_replica": args.sampling_workers_per_replica,
            "max_workers_override": args.sampling_max_workers,
            "default_worker_limit": len(sampling_urls) * args.sampling_workers_per_replica,
            "rollout_score_workers": args.rollout_score_workers,
            "rollout_progress_log_interval": args.rollout_progress_log_interval,
            "wandb_rollout_log_interval": args.wandb_rollout_log_interval,
        },
        "client_liveness": {
            "future_timeout_s": args.future_timeout_s,
            "sync_timeout_s": args.sync_timeout_s,
            "prefetch_timeout_s": args.prefetch_timeout_s,
            "direct_train_server_futures": args.direct_train_server_futures,
        },
        "loss_params": loss_params,
        "reward_verifier": {
            "name": "reference_final_answer",
            "answer_tail_chars": 300,
            "requires_answer_anchor": False,
            "prefers_answer_anchor": True,
            "fallback": "last_boxed_in_tail",
        },
        "length_penalty": asdict(length_config),
        "xorl_client_chunking": client_chunking,
        "pipeline_rl": {
            "enabled": args.pipeline_rl,
            "policy_lag_steps_when_active": 1,
            "wait_for_prefetch_before_weight_sync": True,
            "cache_invalidation_mode": args.sync_cache_invalidation_mode,
            "sync_pause_mode": args.sync_pause_mode,
        },
        "checkpointing": {
            "checkpoint_interval": args.checkpoint_interval,
            "checkpoint_prefix": args.checkpoint_prefix or run_id,
            "save_final_checkpoint": args.save_final_checkpoint,
        },
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name or run_id,
            "group": args.wandb_group,
            "tags": _split_csv(args.wandb_tags),
            "mode": args.wandb_mode,
            "rollout_log_interval": args.wandb_rollout_log_interval,
        },
    }


def _maybe_init_wandb(args: argparse.Namespace, *, run_id: str, output_dir: Path, run_config: dict):
    if not args.wandb_project:
        return None

    if args.wandb_mode:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)

    import wandb  # noqa: PLC0415

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or run_id,
        group=args.wandb_group or None,
        tags=_split_csv(args.wandb_tags),
        config=run_config,
        dir=str(output_dir),
    )
    wandb_run.define_metric("policy_step")
    wandb_run.define_metric("*", step_metric="policy_step")
    return wandb_run


def _wandb_log(wandb_run, payload: dict[str, Any], *, policy_step: int | None = None) -> None:
    if wandb_run is not None:
        log_payload = {key: value for key, value in payload.items() if value is not None}
        if policy_step is not None:
            log_payload.setdefault("policy_step", policy_step)
        wandb_run.log(log_payload, commit=True)
        summary_payload = {
            f"latest/{key}": value
            for key, value in log_payload.items()
            if isinstance(value, int | float | bool) and not key.startswith("_")
        }
        if summary_payload:
            wandb_run.summary.update(summary_payload)


def _numeric_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, int | float | bool]:
    metrics: dict[str, int | float | bool] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, bool):
            metrics[metric_key] = value
        elif isinstance(value, int | float) and value is not None:
            metrics[metric_key] = value
        elif isinstance(value, dict):
            metrics.update(_numeric_metrics(metric_key, value))
    return metrics


def _configure_xorl_client_chunking(args: argparse.Namespace) -> dict[str, int | bool]:
    max_chunk_len = int(args.xorl_client_max_chunk_len)
    max_chunk_bytes = int(args.xorl_client_max_chunk_bytes)
    min_tail_chunk_len = int(args.xorl_client_min_tail_chunk_len)
    if max_chunk_len <= 0:
        raise ValueError("--xorl-client-max-chunk-len must be positive")
    if max_chunk_bytes <= 0:
        raise ValueError("--xorl-client-max-chunk-bytes must be positive")
    if min_tail_chunk_len < 0:
        raise ValueError("--xorl-client-min-tail-chunk-len must be non-negative")

    xorl_chunked_helpers.MAX_CHUNK_LEN = max_chunk_len
    xorl_chunked_helpers.MAX_CHUNK_BYTES_COUNT = max_chunk_bytes
    # training_client imports these constants by value, so update both modules.
    xorl_training_client_module.MAX_CHUNK_LEN = max_chunk_len
    xorl_training_client_module.MAX_CHUNK_BYTES_COUNT = max_chunk_bytes
    # Recent xorl-client also reads these env vars at call time.
    os.environ["XORL_CLIENT_MAX_CHUNK_LEN"] = str(max_chunk_len)
    os.environ["XORL_CLIENT_MAX_CHUNK_BYTES_COUNT"] = str(max_chunk_bytes)
    os.environ["XORL_CLIENT_MIN_TAIL_CHUNK_LEN"] = str(min_tail_chunk_len)
    return {
        "max_chunk_len": max_chunk_len,
        "max_chunk_bytes": max_chunk_bytes,
        "min_tail_chunk_len": min_tail_chunk_len,
        "default_max_chunk_len": DEFAULT_XORL_CLIENT_MAX_CHUNK_LEN,
        "default_max_chunk_bytes": DEFAULT_XORL_CLIENT_MAX_CHUNK_BYTES,
        "sequential_chunks": bool(args.xorl_client_sequential_chunks),
    }


def _split_forward_backward_chunks(
    datums: Sequence[Any],
    *,
    max_chunk_len: int | None = None,
    max_chunk_bytes: int | None = None,
    min_tail_chunk_len: int | None = None,
) -> list[list[Any]]:
    if max_chunk_len is None:
        max_chunk_len = int(xorl_training_client_module.MAX_CHUNK_LEN)
    if max_chunk_bytes is None:
        max_chunk_bytes = int(xorl_training_client_module.MAX_CHUNK_BYTES_COUNT)
    if min_tail_chunk_len is None:
        min_tail_chunk_len = DEFAULT_XORL_CLIENT_MIN_TAIL_CHUNK_LEN

    chunks: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0

    for datum in datums:
        datum_bytes = int(estimate_datum_bytes(datum))
        if current and (len(current) >= max_chunk_len or current_bytes + datum_bytes > max_chunk_bytes):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(datum)
        current_bytes += datum_bytes

    if current or not chunks:
        chunks.append(current)
    if min_tail_chunk_len > 0 and len(chunks) > 1 and len(chunks[-1]) < min_tail_chunk_len:
        tail = chunks.pop()
        chunks[-1].extend(tail)
    return chunks


def _estimate_forward_backward_chunks(
    datums: Sequence[Any],
    *,
    max_chunk_len: int | None = None,
    max_chunk_bytes: int | None = None,
    min_tail_chunk_len: int | None = None,
) -> dict[str, int | float]:
    if max_chunk_len is None:
        max_chunk_len = int(xorl_training_client_module.MAX_CHUNK_LEN)
    if max_chunk_bytes is None:
        max_chunk_bytes = int(xorl_training_client_module.MAX_CHUNK_BYTES_COUNT)
    if min_tail_chunk_len is None:
        min_tail_chunk_len = DEFAULT_XORL_CLIENT_MIN_TAIL_CHUNK_LEN

    chunks = _split_forward_backward_chunks(
        datums,
        max_chunk_len=max_chunk_len,
        max_chunk_bytes=max_chunk_bytes,
        min_tail_chunk_len=min_tail_chunk_len,
    )
    chunk_counts = [len(chunk) for chunk in chunks]
    chunk_bytes = [sum(int(estimate_datum_bytes(datum)) for datum in chunk) for chunk in chunks]
    datum_count = len(datums)
    datum_bytes_total = sum(chunk_bytes)
    datum_bytes_max = max((int(estimate_datum_bytes(datum)) for datum in datums), default=0)
    return {
        "chunk_count": len(chunks),
        "chunk_limit_datums": max_chunk_len,
        "chunk_limit_bytes": max_chunk_bytes,
        "min_tail_chunk_len": min_tail_chunk_len,
        "datum_count": datum_count,
        "datum_estimated_bytes_total": datum_bytes_total,
        "datum_estimated_bytes_mean": datum_bytes_total / datum_count if datum_count else 0.0,
        "datum_estimated_bytes_max": datum_bytes_max,
        "chunk_datums_min": min(chunk_counts),
        "chunk_datums_max": max(chunk_counts),
        "chunk_datums_mean": statistics.mean(chunk_counts),
        "chunk_estimated_bytes_min": min(chunk_bytes),
        "chunk_estimated_bytes_max": max(chunk_bytes),
        "chunk_estimated_bytes_mean": statistics.mean(chunk_bytes),
    }


def _derived_train_metrics(fb_metrics: dict[str, Any]) -> dict[str, float]:
    valid_tokens = fb_metrics.get("valid_tokens:sum")
    execution_time = fb_metrics.get("execution_time:sum")
    if not isinstance(valid_tokens, int | float) or not isinstance(execution_time, int | float):
        return {}
    if execution_time <= 0:
        return {}
    return {
        "valid_tokens": float(valid_tokens),
        "forward_backward_execution_time_s": float(execution_time),
        "valid_tokens_per_s": float(valid_tokens) / float(execution_time),
    }


def _future_request_id(future: Any) -> str | None:
    for candidate in (future, getattr(future, "_future", None)):
        if candidate is None:
            continue
        request_id = getattr(candidate, "request_id", None)
        if request_id:
            return str(request_id)
        untyped_future = getattr(candidate, "untyped_future", None)
        request_id = getattr(untyped_future, "request_id", None)
        if request_id:
            return str(request_id)
    return None


def _cancel_future(future: Any) -> bool:
    cancel = getattr(future, "cancel", None)
    if not callable(cancel):
        return False
    try:
        return bool(cancel())
    except Exception:
        return False


def _wait_for_future_result(
    future: Any,
    *,
    label: str,
    timeout_s: float,
    metrics_path: Path,
    step: int | None = None,
    policy_step: int | None = None,
    event_prefix: str = "client_future",
) -> Any:
    request_id = _future_request_id(future)
    started_at_s = time.perf_counter()
    base_payload = {
        "label": label,
        "request_id": request_id,
        "step": step,
        "policy_step": policy_step,
        "timeout_s": timeout_s,
    }
    _write_jsonl(metrics_path, {"event": f"{event_prefix}_start", **base_payload})
    try:
        result = future.result(timeout=timeout_s)
    except TimeoutError as exc:
        wall_s = time.perf_counter() - started_at_s
        cancelled = _cancel_future(future)
        _write_jsonl(
            metrics_path,
            {
                "event": f"{event_prefix}_timeout",
                **base_payload,
                "wall_s": wall_s,
                "cancelled": cancelled,
            },
        )
        raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s") from exc
    except Exception as exc:
        wall_s = time.perf_counter() - started_at_s
        _write_jsonl(
            metrics_path,
            {
                "event": f"{event_prefix}_failed",
                **base_payload,
                "wall_s": wall_s,
                "error": repr(exc),
            },
        )
        raise

    _write_jsonl(
        metrics_path,
        {
            "event": f"{event_prefix}_done",
            **base_payload,
            "wall_s": time.perf_counter() - started_at_s,
        },
    )
    return result


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json() if response.content else {}


def _reserve_training_seq_id(training_client: TrainingClient) -> tuple[int, int] | None:
    get_request_id = getattr(training_client, "_get_request_id", None)
    if not callable(get_request_id):
        return None
    request_id = int(get_request_id())
    return request_id, request_id + 1


def _advance_training_turn(training_client: TrainingClient, request_id: int | None) -> None:
    if request_id is None or not hasattr(training_client, "_turn_counter"):
        return
    turn_counter = int(getattr(training_client, "_turn_counter"))
    if turn_counter > request_id:
        return
    if turn_counter < request_id:
        raise RuntimeError(f"xorl-client turn counter out of sync: turn_counter={turn_counter}, request_id={request_id}")
    setattr(training_client, "_turn_counter", request_id + 1)
    waiters = getattr(training_client, "_turn_waiters", {})
    waiter = waiters.get(request_id + 1) if isinstance(waiters, dict) else None
    if waiter is not None:
        loop = training_client.holder.get_loop()
        loop.call_soon_threadsafe(waiter.set)


def _wait_for_server_future(
    *,
    train_url: str,
    server_request_id: str,
    timeout_s: float,
    metrics_path: Path,
    label: str,
    step: int | None,
    policy_step: int | None,
    event_prefix: str,
) -> dict[str, Any]:
    started_at_s = time.perf_counter()
    deadline_s = started_at_s + timeout_s
    polls = 0
    last_queue_state: str | None = None
    while time.perf_counter() < deadline_s:
        polls += 1
        poll_timeout_s = min(60.0, max(1.0, deadline_s - time.perf_counter()))
        try:
            result = _post_json(
                f"{train_url}/api/v1/retrieve_future",
                {"request_id": server_request_id},
                timeout_s=poll_timeout_s,
            )
        except requests.RequestException as exc:
            if time.perf_counter() >= deadline_s:
                break
            _write_jsonl(
                metrics_path,
                {
                    "event": f"{event_prefix}_poll_retry",
                    "label": label,
                    "server_request_id": server_request_id,
                    "step": step,
                    "policy_step": policy_step,
                    "wall_s": time.perf_counter() - started_at_s,
                    "polls": polls,
                    "error": repr(exc),
                },
            )
            time.sleep(1.0)
            continue
        if result.get("type") == "try_again":
            last_queue_state = str(result.get("queue_state") or "")
            time.sleep(1.0)
            continue
        if result.get("type") == "request_failed" or result.get("error"):
            _write_jsonl(
                metrics_path,
                {
                    "event": f"{event_prefix}_failed",
                    "label": label,
                    "server_request_id": server_request_id,
                    "step": step,
                    "policy_step": policy_step,
                    "wall_s": time.perf_counter() - started_at_s,
                    "polls": polls,
                    "error": result.get("error") or result,
                },
            )
            raise RuntimeError(f"{label} failed: {result.get('error', result)}")
        _write_jsonl(
            metrics_path,
            {
                "event": f"{event_prefix}_done",
                "label": label,
                "server_request_id": server_request_id,
                "step": step,
                "policy_step": policy_step,
                "wall_s": time.perf_counter() - started_at_s,
                "polls": polls,
                "last_queue_state": last_queue_state,
            },
        )
        return result

    _write_jsonl(
        metrics_path,
        {
            "event": f"{event_prefix}_timeout",
            "label": label,
            "server_request_id": server_request_id,
            "step": step,
            "policy_step": policy_step,
            "timeout_s": timeout_s,
            "wall_s": time.perf_counter() - started_at_s,
            "polls": polls,
            "last_queue_state": last_queue_state,
        },
    )
    raise TimeoutError(f"{label} future {server_request_id} timed out after {timeout_s:.1f}s")


def _call_train_server_future(
    training_client: TrainingClient,
    *,
    endpoint: str,
    payload: dict[str, Any],
    label: str,
    timeout_s: float,
    metrics_path: Path,
    step: int | None,
    policy_step: int | None,
    event_prefix: str,
) -> dict[str, Any]:
    train_url = str(training_client.holder.base_url).rstrip("/")
    seq_assignment = _reserve_training_seq_id(training_client)
    client_request_id = seq_assignment[0] if seq_assignment is not None else None
    seq_id = seq_assignment[1] if seq_assignment is not None else None
    if seq_id is not None:
        payload = {**payload, "seq_id": seq_id}

    submit_started_at_s = time.perf_counter()
    try:
        future = _post_json(f"{train_url}{endpoint}", payload, timeout_s=min(120.0, max(1.0, timeout_s)))
    finally:
        _advance_training_turn(training_client, client_request_id)

    server_request_id = future.get("request_id")
    if not server_request_id:
        raise RuntimeError(f"{label} did not return request_id: {future}")

    _write_jsonl(
        metrics_path,
        {
            "event": f"{event_prefix}_submit_done",
            "label": label,
            "step": step,
            "policy_step": policy_step,
            "client_request_id": client_request_id,
            "seq_id": seq_id,
            "server_request_id": str(server_request_id),
            "wall_s": time.perf_counter() - submit_started_at_s,
        },
    )
    return _wait_for_server_future(
        train_url=train_url,
        server_request_id=str(server_request_id),
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        label=label,
        step=step,
        policy_step=policy_step,
        event_prefix=event_prefix,
    )


def _convert_datums_for_request(training_client: TrainingClient, datums: Sequence[Any]) -> tuple[list[dict], list[Any]]:
    convert_datums = getattr(training_client, "_convert_datums", None)
    if callable(convert_datums):
        return convert_datums(list(datums))

    datums_dicts = []
    routed_experts = []
    for datum in datums:
        if hasattr(datum, "to_dict"):
            datum_dict = datum.to_dict()
        elif hasattr(datum, "model_dump"):
            datum_dict = datum.model_dump()
        elif isinstance(datum, dict):
            datum_dict = dict(datum)
        else:
            raise TypeError(f"Expected Datum, dict, or Pydantic model, got {type(datum).__name__}")
        routed = datum_dict.pop("routed_experts", None)
        datums_dicts.append(datum_dict)
        if routed is not None:
            routed_experts.append(routed)
    return datums_dicts, routed_experts


def _forward_backward_chunk_direct(
    training_client: TrainingClient,
    chunk: Sequence[Any],
    *,
    loss_fn: str,
    loss_fn_params: dict[str, Any],
    timeout_s: float,
    metrics_path: Path,
    step: int,
    policy_step: int,
    chunk_index: int,
    chunk_count: int,
    chunk_bytes: int,
) -> types.ForwardBackwardOutput:
    datums_dicts, routed_experts = _convert_datums_for_request(training_client, chunk)
    request_data: dict[str, Any] = {
        "model_id": training_client.model_id,
        "forward_backward_input": {
            "data": datums_dicts,
            "loss_fn": loss_fn,
            "loss_fn_params": loss_fn_params,
        },
    }
    if routed_experts and len(routed_experts) == len(datums_dicts):
        request_data["forward_backward_input"]["routed_experts"] = routed_experts

    result = _call_train_server_future(
        training_client,
        endpoint="/api/v1/forward_backward",
        payload=request_data,
        label=f"direct forward_backward step={step} chunk={chunk_index + 1}/{chunk_count} datums={len(chunk)}",
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        step=step,
        policy_step=policy_step,
        event_prefix="train_future_chunk",
    )
    _write_jsonl(
        metrics_path,
        {
            "event": "train_future_chunk_payload",
            "step": step,
            "policy_step": policy_step,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "chunk_datums": len(chunk),
            "chunk_estimated_bytes": chunk_bytes,
        },
    )
    return types.ForwardBackwardOutput.from_dict(result)


def _forward_backward_result(
    training_client: TrainingClient,
    datums: Sequence[Any],
    *,
    loss_fn: str,
    loss_fn_params: dict[str, Any],
    sequential_chunks: bool,
    timeout_s: float,
    metrics_path: Path,
    step: int,
    policy_step: int,
    min_tail_chunk_len: int | None = None,
    direct_server_futures: bool = False,
) -> Any:
    chunk_estimate = _estimate_forward_backward_chunks(datums, min_tail_chunk_len=min_tail_chunk_len)
    chunk_count = int(chunk_estimate["chunk_count"])
    _write_jsonl(
        metrics_path,
        {
            "event": "train_future_xorl_client_forward_backward",
            "step": step,
            "policy_step": policy_step,
            "chunk_count": chunk_count,
            "datum_count": chunk_estimate["datum_count"],
            "datum_estimated_bytes_total": chunk_estimate["datum_estimated_bytes_total"],
            "chunk_estimated_bytes_max": chunk_estimate["chunk_estimated_bytes_max"],
            "requested_sequential_chunks": sequential_chunks,
        },
    )
    if direct_server_futures:
        chunks = (
            _split_forward_backward_chunks(datums, min_tail_chunk_len=min_tail_chunk_len)
            if sequential_chunks
            else [list(datums)]
        )
        started_at_s = time.perf_counter()
        label = (
            f"direct forward_backward step={step} datums={len(datums)} chunks={len(chunks)} "
            f"sequential={sequential_chunks}"
        )
        _write_jsonl(
            metrics_path,
            {
                "event": "train_future_start",
                "label": label,
                "request_id": None,
                "step": step,
                "policy_step": policy_step,
                "timeout_s": timeout_s,
                "chunk_count": len(chunks),
                "direct_server_futures": True,
            },
        )
        results = []
        try:
            for chunk_index, chunk in enumerate(chunks):
                chunk_bytes = sum(int(estimate_datum_bytes(datum)) for datum in chunk)
                results.append(
                    _forward_backward_chunk_direct(
                        training_client,
                        chunk,
                        loss_fn=loss_fn,
                        loss_fn_params=loss_fn_params,
                        timeout_s=timeout_s,
                        metrics_path=metrics_path,
                        step=step,
                        policy_step=policy_step,
                        chunk_index=chunk_index,
                        chunk_count=len(chunks),
                        chunk_bytes=chunk_bytes,
                    )
                )
        except TimeoutError as exc:
            _write_jsonl(
                metrics_path,
                {
                    "event": "train_future_timeout",
                    "label": label,
                    "request_id": None,
                    "step": step,
                    "policy_step": policy_step,
                    "timeout_s": timeout_s,
                    "wall_s": time.perf_counter() - started_at_s,
                    "chunk_count": len(chunks),
                    "completed_chunks": len(results),
                    "direct_server_futures": True,
                },
            )
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s") from exc
        except Exception as exc:
            _write_jsonl(
                metrics_path,
                {
                    "event": "train_future_failed",
                    "label": label,
                    "request_id": None,
                    "step": step,
                    "policy_step": policy_step,
                    "timeout_s": timeout_s,
                    "wall_s": time.perf_counter() - started_at_s,
                    "chunk_count": len(chunks),
                    "completed_chunks": len(results),
                    "direct_server_futures": True,
                    "error": repr(exc),
                },
            )
            raise

        combined = combine_fwd_bwd_output_results(results)
        _write_jsonl(
            metrics_path,
            {
                "event": "train_future_done",
                "label": label,
                "request_id": None,
                "step": step,
                "policy_step": policy_step,
                "timeout_s": timeout_s,
                "wall_s": time.perf_counter() - started_at_s,
                "chunk_count": len(chunks),
                "completed_chunks": len(results),
                "direct_server_futures": True,
            },
        )
        return combined

    if sequential_chunks and chunk_count > 1:
        chunks = _split_forward_backward_chunks(datums, min_tail_chunk_len=min_tail_chunk_len)
        started_at_s = time.perf_counter()
        label = f"xorl_client.forward_backward sequential step={step} datums={len(datums)} chunks={len(chunks)}"
        _write_jsonl(
            metrics_path,
            {
                "event": "train_future_start",
                "label": label,
                "request_id": None,
                "step": step,
                "policy_step": policy_step,
                "timeout_s": timeout_s,
                "chunk_count": len(chunks),
            },
        )
        results = []
        try:
            for chunk_index, chunk in enumerate(chunks):
                chunk_submit_started_at_s = time.perf_counter()
                chunk_bytes = sum(int(estimate_datum_bytes(datum)) for datum in chunk)
                chunk_label = (
                    f"xorl_client.forward_backward step={step} "
                    f"chunk={chunk_index + 1}/{len(chunks)} datums={len(chunk)}"
                )
                future = training_client.forward_backward(chunk, loss_fn=loss_fn, loss_fn_params=loss_fn_params)
                _write_jsonl(
                    metrics_path,
                    {
                        "event": "train_future_chunk_submit_done",
                        "label": chunk_label,
                        "step": step,
                        "policy_step": policy_step,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "chunk_datums": len(chunk),
                        "chunk_estimated_bytes": chunk_bytes,
                        "request_id": _future_request_id(future),
                        "wall_s": time.perf_counter() - chunk_submit_started_at_s,
                    },
                )
                results.append(
                    _wait_for_future_result(
                        future,
                        label=chunk_label,
                        timeout_s=timeout_s,
                        metrics_path=metrics_path,
                        step=step,
                        policy_step=policy_step,
                        event_prefix="train_future_chunk",
                    )
                )
        except TimeoutError as exc:
            _write_jsonl(
                metrics_path,
                {
                    "event": "train_future_timeout",
                    "label": label,
                    "request_id": None,
                    "step": step,
                    "policy_step": policy_step,
                    "timeout_s": timeout_s,
                    "wall_s": time.perf_counter() - started_at_s,
                    "chunk_count": len(chunks),
                    "completed_chunks": len(results),
                },
            )
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s") from exc
        except Exception as exc:
            _write_jsonl(
                metrics_path,
                {
                    "event": "train_future_failed",
                    "label": label,
                    "request_id": None,
                    "step": step,
                    "policy_step": policy_step,
                    "timeout_s": timeout_s,
                    "wall_s": time.perf_counter() - started_at_s,
                    "chunk_count": len(chunks),
                    "completed_chunks": len(results),
                    "error": repr(exc),
                },
            )
            raise

        combined = combine_fwd_bwd_output_results(results)
        _write_jsonl(
            metrics_path,
            {
                "event": "train_future_done",
                "label": label,
                "request_id": None,
                "step": step,
                "policy_step": policy_step,
                "timeout_s": timeout_s,
                "wall_s": time.perf_counter() - started_at_s,
                "chunk_count": len(chunks),
                "completed_chunks": len(results),
            },
        )
        return combined

    submit_started_at_s = time.perf_counter()
    future = training_client.forward_backward(datums, loss_fn=loss_fn, loss_fn_params=loss_fn_params)
    _write_jsonl(
        metrics_path,
        {
            "event": "train_future_submit_done",
            "label": f"xorl_client.forward_backward step={step} datums={len(datums)}",
            "step": step,
            "policy_step": policy_step,
            "request_id": _future_request_id(future),
            "wall_s": time.perf_counter() - submit_started_at_s,
        },
    )
    return _wait_for_future_result(
        future,
        label=f"xorl_client.forward_backward step={step} datums={len(datums)}",
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        step=step,
        policy_step=policy_step,
        event_prefix="train_future",
    )


def _optim_step_result(
    training_client: TrainingClient,
    adam_params: types.AdamParams,
    *,
    timeout_s: float,
    metrics_path: Path,
    step: int,
    policy_step: int,
    direct_server_futures: bool = False,
) -> Any:
    if direct_server_futures:
        started_at_s = time.perf_counter()
        result = _call_train_server_future(
            training_client,
            endpoint="/api/v1/optim_step",
            payload={"model_id": training_client.model_id, "adam_params": adam_params.to_dict()},
            label=f"direct optim_step step={step}",
            timeout_s=timeout_s,
            metrics_path=metrics_path,
            step=step,
            policy_step=policy_step,
            event_prefix="train_future",
        )
        output = types.OptimStepResponse.from_dict(result)
        _write_jsonl(
            metrics_path,
            {
                "event": "train_future_direct_optim_result",
                "step": step,
                "policy_step": policy_step,
                "wall_s": time.perf_counter() - started_at_s,
                "metrics": output.metrics,
            },
        )
        return output
    return _wait_for_future_result(
        training_client.optim_step(adam_params),
        label=f"xorl_client.optim_step step={step}",
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        step=step,
        policy_step=policy_step,
        event_prefix="train_future",
    )


def _save_state_result(
    training_client: TrainingClient,
    name: str,
    *,
    timeout_s: float,
    metrics_path: Path,
    step: int,
    policy_step: int,
    direct_server_futures: bool = False,
) -> Any:
    if direct_server_futures:
        result = _call_train_server_future(
            training_client,
            endpoint="/api/v1/save_weights",
            payload={"model_id": training_client.model_id, "path": name},
            label=f"direct save_state {name}",
            timeout_s=timeout_s,
            metrics_path=metrics_path,
            step=step,
            policy_step=policy_step,
            event_prefix="checkpoint_future",
        )
        return types.SaveWeightsResponse.from_dict(result)
    return _wait_for_future_result(
        training_client.save_state(name),
        label=f"xorl_client.save_state {name}",
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        step=step,
        policy_step=policy_step,
        event_prefix="checkpoint_future",
    )


def _sync_inference_weights(
    training_client: TrainingClient,
    *,
    master_port: int,
    buffer_size_mb: int,
    cache_invalidation_mode: str,
    pause_mode: str,
    weight_version: str | None,
    timeout_s: float,
    metrics_path: Path,
    event: str,
    step: int | None = None,
    policy_step: int | None = None,
):
    master_address = (
        os.environ.get("P2P_TRAINER_HOSTNAME")
        or os.environ.get("XORL_WEIGHT_SYNC_MASTER_ADDRESS")
        or os.environ.get("XORL_P2P_HOSTNAME")
        or os.environ.get("POD_IP")
        or ""
    )
    request_data = {
        "master_address": master_address,
        "master_port": master_port,
        "group_name": "weight_sync_group",
        "buffer_size_mb": buffer_size_mb,
        "flush_cache": cache_invalidation_mode == "flush",
        "cache_invalidation_mode": cache_invalidation_mode,
        "pause_mode": pause_mode,
    }
    if weight_version is not None:
        request_data["weight_version"] = weight_version
    return _wait_for_future_result(
        training_client.holder.post_async("/sync_inference_weights", request_data, timeout=timeout_s),
        label=f"{event} weight_version={weight_version}",
        timeout_s=timeout_s,
        metrics_path=metrics_path,
        step=step,
        policy_step=policy_step,
        event_prefix="sync_future",
    )


def _require_sync_success(sync_result: Any, *, event: str, weight_version: str | None) -> dict[str, Any]:
    payload = _jsonable(sync_result)
    if bool(payload.get("success", False)):
        return payload
    message = payload.get("message") or payload.get("error") or "no error message"
    version_note = f" weight_version={weight_version}" if weight_version is not None else ""
    raise RuntimeError(f"{event} failed{version_note}: {message}")


def _policy_weight_version(args: argparse.Namespace, *, run_id: str, policy_step: int) -> str | None:
    prefix = run_id if args.sync_weight_version_prefix is None else args.sync_weight_version_prefix
    if not prefix:
        return None
    return f"{prefix}/policy-{policy_step:06d}"


def _checkpoint_name(args: argparse.Namespace, *, run_id: str, policy_step: int) -> str:
    return f"{args.checkpoint_prefix or run_id}-step-{policy_step:06d}"


def _save_checkpoint(
    training_client: TrainingClient,
    *,
    args: argparse.Namespace,
    run_id: str,
    policy_step: int,
    metrics_path: Path,
):
    checkpoint_name = _checkpoint_name(args, run_id=run_id, policy_step=policy_step)
    return _save_state_result(
        training_client,
        checkpoint_name,
        timeout_s=args.future_timeout_s,
        metrics_path=metrics_path,
        step=policy_step - 1,
        policy_step=policy_step,
        direct_server_futures=args.direct_train_server_futures,
    )


def _endpoint_registration_accepted(add_result: Any) -> bool:
    if bool(getattr(add_result, "success", False)):
        return True
    message = str(getattr(add_result, "message", "")).lower()
    return "already registered" in message


def _behavior_k3_from_metrics(metrics: dict) -> float | None:
    ratio_mean = metrics.get("is_loss/ratio/mean:mean")
    neg_log_ratio_mean = metrics.get("is_loss/kl_policy/mean:mean")
    if ratio_mean is None or neg_log_ratio_mean is None:
        return None
    return float(ratio_mean) + float(neg_log_ratio_mean) - 1.0


def _force_nonzero_advantages_for_smoke(advantages: list[float]) -> tuple[list[float], bool]:
    if any(abs(advantage) > 1e-12 for advantage in advantages):
        return advantages, False
    if len(advantages) < 2:
        return advantages, False

    forced = [1.0 if index % 2 == 0 else -1.0 for index in range(len(advantages))]
    if len(forced) % 2 == 1:
        forced[-1] = 0.0
    return forced, True


def _nonzero_advantage_pairs(
    records: Sequence[RolloutRecord],
    advantages: Sequence[float],
    *,
    eps: float = 1e-12,
) -> list[tuple[RolloutRecord, float]]:
    if len(records) != len(advantages):
        raise ValueError(f"records/advantages length mismatch: {len(records)} != {len(advantages)}")
    return [
        (record, float(advantage))
        for record, advantage in zip(records, advantages, strict=True)
        if abs(float(advantage)) > eps
    ]


def _client_index_for_sample(example_index: int, sample_index: int, num_clients: int) -> int:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    return (example_index + sample_index) % num_clients


def _sampling_seed_for_sample(args: argparse.Namespace, *, step: int, example_index: int, sample_index: int) -> int:
    return args.sampling_seed_base + step * args.prompts_per_step * args.samples_per_prompt + (
        example_index * args.samples_per_prompt
    ) + sample_index


def _effective_max_generate_tokens(args: argparse.Namespace, *, prompt_tokens: int) -> int:
    context_limit = args.max_model_tokens - prompt_tokens - args.context_token_reserve
    return max(1, min(args.max_generate_tokens, context_limit))


def _filter_examples_by_prompt_length(
    tokenizer,
    examples: Sequence[MathExample],
    args: argparse.Namespace,
) -> tuple[list[MathExample], dict[str, int | float | bool]]:
    if args.prompt_overlength_policy != "skip":
        return list(examples), {
            "enabled": False,
            "input_examples": len(examples),
            "kept_examples": len(examples),
            "skipped_examples": 0,
        }

    kept: list[MathExample] = []
    token_counts: list[int] = []
    for example in examples:
        prefix = encode_forced_thinking_prefix(
            tokenizer,
            example.prompt,
            max_prompt_tokens=args.max_prompt_tokens,
            truncate_to_max=False,
        )
        if prefix.skipped_for_length:
            continue
        kept.append(example)
        token_counts.append(int(prefix.original_token_count or len(prefix.token_ids)))

    if not kept:
        raise RuntimeError(
            "No examples fit the configured max prompt length. "
            "Check dataset prompt extraction or raise --max-prompt-tokens."
        )

    return kept, {
        "enabled": True,
        "input_examples": len(examples),
        "kept_examples": len(kept),
        "skipped_examples": len(examples) - len(kept),
        "max_prompt_tokens": args.max_prompt_tokens,
        "kept_prompt_tokens_min": min(token_counts),
        "kept_prompt_tokens_max": max(token_counts),
        "kept_prompt_tokens_mean": statistics.fmean(token_counts),
    }


def _logprob_value(item: Any) -> float | None:
    if item is None:
        return None
    if isinstance(item, int | float):
        return float(item)
    if isinstance(item, dict):
        for key in ("logprob", "log_prob", "value"):
            if item.get(key) is not None:
                return float(item[key])
        return None
    if isinstance(item, list | tuple) and item:
        return _logprob_value(item[0])
    return None


def _extract_output_logprobs(response_item: dict[str, Any], gen_len: int) -> list[float]:
    if gen_len == 0:
        return []

    meta = response_item.get("meta_info") or {}
    candidates = [
        meta.get("output_token_logprobs"),
        response_item.get("output_token_logprobs"),
        meta.get("token_logprobs"),
        response_item.get("token_logprobs"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        values = [_logprob_value(item) for item in candidate]
        logprobs = [value for value in values if value is not None]
        if len(logprobs) >= gen_len:
            return [float(value) for value in logprobs[-gen_len:]]
    raise RuntimeError(f"SGLang response did not include {gen_len} output logprobs")


def _normalize_generate_response(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if "meta_info" in data or "output_ids" in data or "text" in data:
            return [data]
        numeric_items = [(int(key), value) for key, value in data.items() if isinstance(key, str) and key.isdigit()]
        if numeric_items:
            return [value for _, value in sorted(numeric_items) if isinstance(value, dict)]
    raise RuntimeError(f"Unexpected SGLang /generate response shape: {type(data).__name__}")


def _sample_sglang_generate(
    *,
    base_url: str,
    input_ids: list[int],
    sampling_params: types.SamplingParams,
    cache_extra_key: str | None,
    request_timeout_s: float = 300.0,
    max_attempts: int = 2,
) -> types.SampleResponse:
    sampling_payload = sampling_params.to_dict()
    sampling_payload["n"] = 1
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_payload,
        "return_logprob": True,
        "return_text_in_logprobs": False,
    }
    if cache_extra_key is not None:
        payload["extra_key"] = cache_extra_key

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(f"{base_url}/generate", json=payload, timeout=request_timeout_s)
            if response.status_code >= 400:
                message = f"SGLang /generate failed: status={response.status_code} body={response.text[:1000]!r}"
                if response.status_code in TRANSIENT_SAMPLING_STATUS_CODES:
                    raise _TransientSamplingError(message)
                raise RuntimeError(message)
            items = _normalize_generate_response(response.json())
            sequences = []
            meta_info = None
            for item in items:
                output_ids = [int(token_id) for token_id in item.get("output_ids", [])]
                logprobs = _extract_output_logprobs(item, len(output_ids))
                meta = item.get("meta_info") or {}
                if meta_info is None:
                    meta_info = meta
                finish_reason = meta.get("finish_reason") or item.get("finish_reason") or {}
                finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
                stop_reason = (
                    "length"
                    if finish_type == "length" or len(output_ids) >= sampling_params.max_tokens
                    else "stop"
                )
                sequences.append(
                    types.SampledSequence(
                        tokens=output_ids,
                        logprobs=logprobs,
                        text=str(item.get("text") or ""),
                        stop_reason=stop_reason,
                    )
                )
            return types.SampleResponse(sequences=sequences, meta_info=meta_info)
        except (requests.RequestException, _TransientSamplingError) as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise RuntimeError(f"SGLang sampling failed after {max_attempts} attempts: {exc}") from exc
            time.sleep(min(2.0**attempt, 8.0))
    raise RuntimeError(f"SGLang sampling failed after retries: {last_error}")


def _score_sglang_response(
    *,
    response: types.SampleResponse,
    example: MathExample,
    prefix,
    client_index: int,
    sample_index: int,
    sampling_seed: int,
    request_max_tokens: int,
    length_config: LengthPenaltyConfig,
    step: int,
    cache_extra_key: str | None,
) -> dict[str, Any]:
    score_start_s = time.perf_counter()
    records: list[RolloutRecord] = []
    correctness: list[float] = []
    prompt_correctness: list[tuple[str, bool]] = []
    rollout_events: list[dict[str, Any]] = []
    sample_text_events: list[dict[str, Any]] = []

    for sequence_offset, sequence in enumerate(response.sequences):
        effective_sample_index = sample_index + sequence_offset
        completion_tokens = list(sequence.tokens)
        completion_logprobs = list(sequence.logprobs or [])
        if len(completion_tokens) != len(completion_logprobs):
            raise RuntimeError(
                f"SGLang returned token/logprob length mismatch for {example.example_id}: "
                f"{len(completion_tokens)} != {len(completion_logprobs)}"
            )
        truncated = sequence.stop_reason == "length" or len(completion_tokens) >= request_max_tokens
        completion_text = str(sequence.text or "")
        verification = grade_reference_final_answer(completion_text, example.gold_answer)
        reward = shaped_reward(
            verifier_reward=verification.reward,
            correct=verification.correct,
            completion_tokens=len(completion_tokens),
            has_box=verification.has_box,
            truncated=truncated,
            config=length_config,
        )
        records.append(
            RolloutRecord(
                prompt_id=example.example_id,
                prefix_tokens=prefix.token_ids,
                completion_tokens=completion_tokens,
                completion_logprobs=completion_logprobs,
                reward=reward,
                completion_text="",
                truncated=truncated,
            )
        )
        correctness.append(1.0 if verification.correct else 0.0)
        prompt_correctness.append((str(example.example_id), verification.correct))
        rollout_events.append(
            {
                "event": "rollout",
                "step": step,
                "example_id": example.example_id,
                "sample_index": effective_sample_index,
                "sampling_seed": sampling_seed,
                "request_max_generate_tokens": request_max_tokens,
                "inference_replica_index": client_index,
                "sampling_replica_index": client_index,
                "cache_extra_key": cache_extra_key,
                "verifier_reward": verification.reward,
                "reward": reward,
                "correct": verification.correct,
                "has_box": verification.has_box,
                "truncated": truncated,
                "completion_tokens": len(completion_tokens),
                "mean_neg_logprob": -statistics.fmean(completion_logprobs) if completion_logprobs else None,
            }
        )
        sample_text_events.append(
            {
                "event": "rollout_sample_text",
                "step": step,
                "example_id": example.example_id,
                "sample_index": effective_sample_index,
                "sampling_seed": sampling_seed,
                "prompt": example.prompt,
                "gold_answer": example.gold_answer,
                "predicted_answer": verification.predicted_answer,
                "completion_text": completion_text,
                "verifier_reward": verification.reward,
                "reward": reward,
                "correct": verification.correct,
                "has_box": verification.has_box,
                "parse_ok": verification.parse_ok,
                "verification_error": verification.error,
                "truncated": truncated,
                "completion_tokens": len(completion_tokens),
            }
        )

    return {
        "records": records,
        "correctness": correctness,
        "prompt_correctness": prompt_correctness,
        "rollout_events": rollout_events,
        "sample_text_events": sample_text_events,
        "score_wall_s": time.perf_counter() - score_start_s,
    }


def _rollout_step(
    *,
    sampling_urls: Sequence[str],
    tokenizer,
    examples: list[MathExample],
    step: int,
    args: argparse.Namespace,
    length_config: LengthPenaltyConfig,
    metrics_path: Path,
    cache_extra_key: str | None,
    wandb_run,
) -> tuple[list[RolloutRecord], dict]:
    if not sampling_urls:
        raise ValueError("at least one sampling URL is required")
    records: list[RolloutRecord] = []
    correctness: list[float] = []
    prompt_correctness: dict[str, list[bool]] = {}
    skipped_long = 0
    trimmed_long = 0
    effective_max_generate_tokens: list[int] = []
    selected = _select_step_examples(examples, step=step, count=args.prompts_per_step, seed=args.seed)
    pending = []
    sample_text_logs = 0
    rollout_start_s = time.perf_counter()

    for example_index, example in enumerate(selected):
        prefix = encode_forced_thinking_prefix(
            tokenizer,
            example.prompt,
            max_prompt_tokens=args.max_prompt_tokens,
            truncate_to_max=args.prompt_overlength_policy == "trim-left",
        )
        if prefix.skipped_for_length:
            skipped_long += 1
            continue
        if prefix.truncated_for_length:
            trimmed_long += 1
        input_ids = list(prefix.token_ids)
        for sample_index in range(args.samples_per_prompt):
            client_index = _client_index_for_sample(example_index, sample_index, len(sampling_urls))
            sampling_seed = _sampling_seed_for_sample(
                args, step=step, example_index=example_index, sample_index=sample_index
            )
            request_max_tokens = _effective_max_generate_tokens(args, prompt_tokens=len(prefix.token_ids))
            effective_max_generate_tokens.append(request_max_tokens)
            sampling_params = types.SamplingParams(
                max_tokens=request_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                sampling_seed=sampling_seed,
            )
            pending.append(
                (
                    example,
                    prefix,
                    client_index,
                    sample_index,
                    sampling_seed,
                    input_ids,
                    sampling_params,
                    request_max_tokens,
                )
            )

    request_order_seed = args.seed + args.sampling_seed_base + (step + 1) * 1_000_003
    if args.shuffle_rollout_requests and len(pending) > 1:
        random.Random(request_order_seed).shuffle(pending)

    default_worker_limit = len(sampling_urls) * args.sampling_workers_per_replica
    worker_limit = args.sampling_max_workers if args.sampling_max_workers is not None else default_worker_limit
    max_workers = min(max(len(pending), 1), max(worker_limit, 1))
    records_total = len(pending)
    failed_requests = 0
    dropped_tail_requests = 0
    submitted_requests = 0
    completed_http_requests = 0
    scored_requests = 0
    last_progress_records = 0
    score_workers = min(max(int(args.rollout_score_workers), 1), max(records_total, 1))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    score_executor = concurrent.futures.ThreadPoolExecutor(max_workers=score_workers)
    abort_executor = False
    abort_score_executor = False
    try:
        pending_iter = iter(pending)
        future_contexts: dict[concurrent.futures.Future, tuple[Any, ...]] = {}
        score_future_contexts: dict[concurrent.futures.Future, tuple[Any, ...]] = {}

        def context_payload(context: tuple[Any, ...]) -> dict[str, Any]:
            example, _prefix, client_index, sample_index, sampling_seed, request_max_tokens = context
            return {
                "example_id": example.example_id,
                "client_index": client_index,
                "sample_index": sample_index,
                "sampling_seed": sampling_seed,
                "request_max_generate_tokens": request_max_tokens,
            }

        def write_request_failure(*, context: tuple[Any, ...], error: str, dropped: bool = False) -> None:
            nonlocal failed_requests, dropped_tail_requests
            failed_requests += 1
            if dropped:
                dropped_tail_requests += 1
            _write_jsonl(
                metrics_path,
                {
                    "event": "rollout_request_failed",
                    "step": step,
                    "error": error,
                    "dropped": dropped,
                    "failed_requests": failed_requests,
                    "max_rollout_failures": args.max_rollout_failures,
                    **context_payload(context),
                },
            )

        def submit_next() -> bool:
            nonlocal submitted_requests
            try:
                (
                    example,
                    prefix,
                    client_index,
                    sample_index,
                    sampling_seed,
                    input_ids,
                    sampling_params,
                    request_max_tokens,
                ) = next(pending_iter)
            except StopIteration:
                return False

            future = executor.submit(
                _sample_sglang_generate,
                base_url=sampling_urls[client_index],
                input_ids=input_ids,
                sampling_params=sampling_params,
                cache_extra_key=cache_extra_key,
                request_timeout_s=args.sampling_request_timeout_s,
                max_attempts=args.sampling_request_max_attempts,
            )
            future_contexts[future] = (
                example,
                prefix,
                client_index,
                sample_index,
                sampling_seed,
                request_max_tokens,
            )
            submitted_requests += 1
            return True

        def submit_score(response: types.SampleResponse, context: tuple[Any, ...]) -> None:
            example, prefix, client_index, sample_index, sampling_seed, request_max_tokens = context
            score_future = score_executor.submit(
                _score_sglang_response,
                response=response,
                example=example,
                prefix=prefix,
                client_index=client_index,
                sample_index=sample_index,
                sampling_seed=sampling_seed,
                request_max_tokens=request_max_tokens,
                length_config=length_config,
                step=step,
                cache_extra_key=cache_extra_key,
            )
            score_future_contexts[score_future] = context

        def write_progress_if_needed(*, force: bool = False) -> None:
            nonlocal last_progress_records
            interval = int(args.rollout_progress_log_interval)
            if not records:
                return
            if interval <= 0 and not force:
                return
            if not force and len(records) - last_progress_records < interval:
                return
            last_progress_records = len(records)
            elapsed_s = time.perf_counter() - rollout_start_s
            progress_completion_tokens = sum(len(record.completion_tokens) for record in records)
            progress_prompt_tokens = sum(len(record.prefix_tokens) for record in records)
            payload = {
                "step": step,
                "records_completed": len(records),
                "records_total": records_total,
                "fraction": len(records) / records_total if records_total else 0.0,
                "completion_tokens_total": progress_completion_tokens,
                "prompt_tokens_total": progress_prompt_tokens,
                "elapsed_s": elapsed_s,
                "completion_tokens_per_s": progress_completion_tokens / elapsed_s if elapsed_s > 0 else None,
                "total_tokens_per_s": (progress_completion_tokens + progress_prompt_tokens) / elapsed_s
                if elapsed_s > 0
                else None,
                "mean_completion_tokens": statistics.fmean(len(record.completion_tokens) for record in records),
                "accuracy": statistics.fmean(correctness),
                "truncated_fraction": statistics.fmean(1.0 if record.truncated else 0.0 for record in records),
                "http_inflight_requests": len(future_contexts),
                "score_inflight_requests": len(score_future_contexts),
                "submitted_requests": submitted_requests,
                "completed_http_requests": completed_http_requests,
                "scored_requests": scored_requests,
                "sampling_max_workers": max_workers,
                "rollout_score_workers": score_workers,
            }
            _write_jsonl(metrics_path, {"event": "rollout_progress", **payload})
            if args.wandb_rollout_log_interval > 0:
                _wandb_log(
                    wandb_run,
                    {f"rollout_progress/{key}": value for key, value in payload.items()},
                    policy_step=step,
                )

        def consume_score_future(score_future: concurrent.futures.Future) -> None:
            nonlocal sample_text_logs, scored_requests
            context = score_future_contexts.pop(score_future)
            try:
                score_payload = score_future.result()
            except Exception as exc:
                write_request_failure(context=context, error=f"rollout scoring failed: {exc!r}")
                raise RuntimeError(f"Rollout step {step} scoring failed for {context_payload(context)}") from exc

            records.extend(score_payload["records"])
            correctness.extend(score_payload["correctness"])
            for prompt_id, is_correct in score_payload["prompt_correctness"]:
                prompt_correctness.setdefault(prompt_id, []).append(is_correct)
            for event in score_payload["rollout_events"]:
                event["score_wall_s"] = score_payload["score_wall_s"]
                _write_jsonl(metrics_path, event)
            for event in score_payload["sample_text_events"]:
                if sample_text_logs >= args.sample_text_log_limit:
                    break
                _write_jsonl(metrics_path, event)
                sample_text_logs += 1
            scored_requests += 1
            write_progress_if_needed()

        def drain_ready_score_futures() -> None:
            for score_future in [future for future in score_future_contexts if future.done()]:
                consume_score_future(score_future)

        for _ in range(max_workers):
            if not submit_next():
                break

        while future_contexts:
            drain_ready_score_futures()
            try:
                future = next(
                    concurrent.futures.as_completed(
                        tuple(future_contexts),
                        timeout=args.sampling_future_idle_timeout_s,
                    )
                )
            except concurrent.futures.TimeoutError as exc:
                pending_contexts = list(future_contexts.values())
                abort_executor = True
                for pending_future, context in list(future_contexts.items()):
                    pending_future.cancel()
                    write_request_failure(
                        context=context,
                        error=(
                            "rollout collector idle timeout after "
                            f"{args.sampling_future_idle_timeout_s:.1f}s"
                        ),
                        dropped=True,
                    )
                preview = [context_payload(context) for context in pending_contexts[:8]]
                raise RuntimeError(
                    f"Rollout step {step} stalled: no completed /generate future for "
                    f"{args.sampling_future_idle_timeout_s:.1f}s; "
                    f"completed={len(records)}/{records_total}; pending={len(pending_contexts)}; "
                    f"failed={failed_requests}; max_failures={args.max_rollout_failures}; "
                    f"pending_preview={preview}"
                ) from exc

            example, prefix, client_index, sample_index, sampling_seed, request_max_tokens = future_contexts.pop(future)
            submit_next()
            try:
                response = future.result()
            except Exception as exc:
                write_request_failure(context=(example, prefix, client_index, sample_index, sampling_seed, request_max_tokens), error=repr(exc))
                if failed_requests > args.max_rollout_failures:
                    abort_executor = True
                    raise RuntimeError(
                        f"Rollout step {step} exceeded max failed requests: "
                        f"failed={failed_requests}, max_rollout_failures={args.max_rollout_failures}, "
                        f"last_error={exc!r}"
                    ) from exc
                continue
            completed_http_requests += 1
            submit_score(response, (example, prefix, client_index, sample_index, sampling_seed, request_max_tokens))
            del response

        while score_future_contexts:
            try:
                score_future = next(
                    concurrent.futures.as_completed(
                        tuple(score_future_contexts),
                        timeout=args.sampling_future_idle_timeout_s,
                    )
                )
            except concurrent.futures.TimeoutError as exc:
                abort_score_executor = True
                preview = [context_payload(context) for context in list(score_future_contexts.values())[:8]]
                raise RuntimeError(
                    f"Rollout step {step} scoring stalled for {args.sampling_future_idle_timeout_s:.1f}s; "
                    f"scored={scored_requests}; pending_score={len(score_future_contexts)}; preview={preview}"
                ) from exc
            consume_score_future(score_future)
        write_progress_if_needed(force=True)
    except Exception:
        abort_executor = True
        abort_score_executor = True
        raise
    finally:
        executor.shutdown(wait=not abort_executor, cancel_futures=abort_executor)
        score_executor.shutdown(wait=not abort_score_executor, cancel_futures=abort_score_executor)

    if not records:
        raise RuntimeError(f"Step {step} produced no rollout records; skipped_long={skipped_long}")

    rollout_wall_s = time.perf_counter() - rollout_start_s
    completion_tokens_total = sum(len(record.completion_tokens) for record in records)
    prompt_tokens_total = sum(len(record.prefix_tokens) for record in records)
    pass_at_group = statistics.fmean(1.0 if any(values) else 0.0 for values in prompt_correctness.values())
    summary = {
        "num_records": len(records),
        "skipped_long_prompts": skipped_long,
        "trimmed_long_prompts": trimmed_long,
        "mean_reward": statistics.fmean(record.reward for record in records),
        "pass_at_group": pass_at_group,
        "pass_at_n": args.samples_per_prompt,
        "mean_completion_tokens": statistics.fmean(len(record.completion_tokens) for record in records),
        "completion_tokens_total": completion_tokens_total,
        "prompt_tokens_total": prompt_tokens_total,
        "min_effective_max_generate_tokens": min(effective_max_generate_tokens),
        "max_effective_max_generate_tokens": max(effective_max_generate_tokens),
        "rollout_wall_s": rollout_wall_s,
        "completion_tokens_per_s": completion_tokens_total / rollout_wall_s if rollout_wall_s > 0 else None,
        "total_tokens_per_s": (completion_tokens_total + prompt_tokens_total) / rollout_wall_s
        if rollout_wall_s > 0
        else None,
        "accuracy": statistics.fmean(correctness),
        "truncated_fraction": statistics.fmean(1.0 if record.truncated else 0.0 for record in records),
        "inference_replicas": len(sampling_urls),
        "sampling_replicas": len(sampling_urls),
        "sampling_max_workers": max_workers,
        "sampling_worker_limit": worker_limit,
        "sampling_max_workers_override": args.sampling_max_workers,
        "sampling_workers_per_replica": args.sampling_workers_per_replica,
        "sampling_request_timeout_s": args.sampling_request_timeout_s,
        "sampling_request_max_attempts": args.sampling_request_max_attempts,
        "sampling_future_idle_timeout_s": args.sampling_future_idle_timeout_s,
        "rollout_score_workers": score_workers,
        "submitted_requests": submitted_requests,
        "completed_http_requests": completed_http_requests,
        "scored_requests": scored_requests,
        "max_rollout_failures": args.max_rollout_failures,
        "failed_requests": failed_requests,
        "dropped_tail_requests": dropped_tail_requests,
        "shuffle_rollout_requests": args.shuffle_rollout_requests,
        "request_order_seed": request_order_seed,
        "cache_extra_key": cache_extra_key,
    }
    if args.samples_per_prompt == 16:
        summary["pass_at_16"] = pass_at_group
    return records, summary


@dataclass
class RolloutPrefetch:
    step: int
    weight_version: str | None
    started_at_s: float
    future: concurrent.futures.Future[tuple[list[RolloutRecord], dict]]
    completed_at_s: float | None = None


def _start_rollout_prefetch(
    *,
    executor: concurrent.futures.Executor,
    sampling_urls: Sequence[str],
    tokenizer,
    examples: list[MathExample],
    step: int,
    args: argparse.Namespace,
    length_config: LengthPenaltyConfig,
    metrics_path: Path,
    cache_extra_key: str | None,
    wandb_run,
) -> RolloutPrefetch:
    started_at_s = time.perf_counter()
    prefetch = RolloutPrefetch(
        step=step,
        weight_version=cache_extra_key,
        started_at_s=started_at_s,
        future=executor.submit(
            _rollout_step,
            sampling_urls=sampling_urls,
            tokenizer=tokenizer,
            examples=examples,
            step=step,
            args=args,
            length_config=length_config,
            metrics_path=metrics_path,
            cache_extra_key=cache_extra_key,
            wandb_run=wandb_run,
        ),
    )
    _write_jsonl(
        metrics_path,
        {
            "event": "pipeline_prefetch_start",
            "step": step,
            "weight_version": cache_extra_key,
        },
    )
    _wandb_log(
        wandb_run,
        {
            "pipeline/prefetch_started": True,
            "pipeline/prefetch_step": step,
        },
        policy_step=step,
    )
    return prefetch


def _wait_for_prefetch_before_sync(
    prefetch: RolloutPrefetch,
    *,
    metrics_path: Path,
    wandb_run,
    step: int,
    policy_step: int,
    timeout_s: float,
) -> None:
    wait_start_s = time.perf_counter()
    try:
        prefetch.future.result(timeout=timeout_s)
    except TimeoutError as exc:
        prefetch.future.cancel()
        wait_s = time.perf_counter() - wait_start_s
        _write_jsonl(
            metrics_path,
            {
                "event": "pipeline_prefetch_timeout_before_sync",
                "step": step,
                "prefetch_step": prefetch.step,
                "wait_s": wait_s,
                "timeout_s": timeout_s,
            },
        )
        raise TimeoutError(
            f"Pipeline rollout prefetch for step {prefetch.step} did not finish before sync after {timeout_s:.1f}s"
        ) from exc
    except Exception as exc:
        _write_jsonl(
            metrics_path,
            {
                "event": "pipeline_prefetch_failed_before_sync",
                "step": step,
                "prefetch_step": prefetch.step,
                "error": repr(exc),
            },
        )
        raise

    wait_s = time.perf_counter() - wait_start_s
    prefetch.completed_at_s = time.perf_counter()
    payload = {
        "event": "pipeline_prefetch_ready_before_sync",
        "step": step,
        "policy_step": policy_step,
        "prefetch_step": prefetch.step,
        "wait_s": wait_s,
        "prefetch_wall_s": prefetch.completed_at_s - prefetch.started_at_s,
        "weight_version": prefetch.weight_version,
    }
    _write_jsonl(metrics_path, payload)
    _wandb_log(
        wandb_run,
        {
            "pipeline/prefetch_wait_before_sync_s": wait_s,
            "pipeline/prefetch_wall_s": payload["prefetch_wall_s"],
            "pipeline/prefetch_ready_before_sync": True,
        },
        policy_step=policy_step,
    )


def _consume_rollout_prefetch(
    prefetch: RolloutPrefetch,
    *,
    timeout_s: float,
    metrics_path: Path,
) -> tuple[list[RolloutRecord], dict, dict[str, Any]]:
    wait_start_s = time.perf_counter()
    try:
        records, summary = prefetch.future.result(timeout=timeout_s)
    except TimeoutError as exc:
        prefetch.future.cancel()
        _write_jsonl(
            metrics_path,
            {
                "event": "pipeline_prefetch_timeout_on_consume",
                "prefetch_step": prefetch.step,
                "wait_s": time.perf_counter() - wait_start_s,
                "timeout_s": timeout_s,
            },
        )
        raise TimeoutError(
            f"Pipeline rollout prefetch for step {prefetch.step} did not finish on consume after {timeout_s:.1f}s"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Pipeline rollout prefetch for step {prefetch.step} failed: {exc!r}") from exc

    wait_s = time.perf_counter() - wait_start_s
    completed_at_s = prefetch.completed_at_s or time.perf_counter()
    return (
        records,
        dict(summary),
        {
            "pipeline_prefetched": True,
            "pipeline_prefetch_wait_s": wait_s,
            "pipeline_prefetch_wall_s": completed_at_s - prefetch.started_at_s,
            "pipeline_prefetch_completed_before_consume": prefetch.completed_at_s is not None,
            "pipeline_generated_weight_version": prefetch.weight_version,
        },
    )


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be positive")
    if args.sampling_workers_per_replica <= 0:
        raise ValueError("--sampling-workers-per-replica must be positive")
    if args.sampling_max_workers is not None and args.sampling_max_workers <= 0:
        raise ValueError("--sampling-max-workers must be positive when set")
    if args.sampling_request_timeout_s <= 0:
        raise ValueError("--sampling-request-timeout-s must be positive")
    if args.sampling_request_max_attempts <= 0:
        raise ValueError("--sampling-request-max-attempts must be positive")
    if args.sampling_future_idle_timeout_s <= 0:
        raise ValueError("--sampling-future-idle-timeout-s must be positive")
    if args.rollout_score_workers <= 0:
        raise ValueError("--rollout-score-workers must be positive")
    if args.rollout_progress_log_interval < 0:
        raise ValueError("--rollout-progress-log-interval must be non-negative")
    if args.wandb_rollout_log_interval < 0:
        raise ValueError("--wandb-rollout-log-interval must be non-negative")
    if args.future_timeout_s <= 0:
        raise ValueError("--future-timeout-s must be positive")
    if args.sync_timeout_s <= 0:
        raise ValueError("--sync-timeout-s must be positive")
    if args.prefetch_timeout_s <= 0:
        raise ValueError("--prefetch-timeout-s must be positive")
    if args.max_rollout_failures < 0:
        raise ValueError("--max-rollout-failures must be non-negative")
    if args.wandb_rollout_log_interval < 0:
        raise ValueError("--wandb-rollout-log-interval must be non-negative")
    if args.sample_text_log_limit < 0:
        raise ValueError("--sample-text-log-limit must be non-negative")
    if args.logprob_temperature is not None and args.logprob_temperature <= 0.0:
        raise ValueError("--logprob-temperature must be positive when set")
    if args.drgrpo_num_chunks <= 0:
        raise ValueError("--drgrpo-num-chunks must be positive")
    if args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be positive")
    if args.max_generate_tokens <= 0:
        raise ValueError("--max-generate-tokens must be positive")
    if args.max_model_tokens <= 0:
        raise ValueError("--max-model-tokens must be positive")
    if args.context_token_reserve < 0:
        raise ValueError("--context-token-reserve must be non-negative")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")
    if args.sync_cache_invalidation_mode == "flush" and args.sync_pause_mode == "in_place":
        raise ValueError("--sync-pause-mode in_place requires --sync-cache-invalidation-mode none or auto")
    client_chunking = _configure_xorl_client_chunking(args)

    run_id = args.run_id or time.strftime("marin-rl-6279-%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(args.output_dir) / run_id
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    service_client = ServiceClient(base_url=args.train_url, timeout=args.future_timeout_s)
    training_client = TrainingClient(holder=service_client.holder, model_id=MODEL_ID, base_model=args.model)
    inference_urls = _parse_inference_urls(args.inference_url)
    sampling_urls = _parse_url_list(args.sampling_url or args.inference_url, arg_name="--sampling-url")
    tokenizer = training_client.get_tokenizer()

    examples_raw = load_examples(args.dataset, limit=args.dataset_limit)
    unfiltered_examples = list(examples_raw)
    examples, prompt_filter = _filter_examples_by_prompt_length(tokenizer, unfiltered_examples, args)
    endpoints = [_url_host_port(url) for url in inference_urls]
    wandb_run = None
    saved_checkpoint_steps: set[int] = set()
    last_update_policy_step: int | None = None
    prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
    prefetch: RolloutPrefetch | None = None
    try:
        length_config = LengthPenaltyConfig(max_completion_tokens=args.max_generate_tokens, lpw=1.0)
        loss_params = _make_loss_params(args)
        runtime_provenance = _runtime_provenance()
        verifier_self_check = _verifier_self_check()
        run_config = _build_run_config(
            args=args,
            run_id=run_id,
            output_dir=output_dir,
            inference_urls=inference_urls,
            sampling_urls=sampling_urls,
            num_examples=len(examples),
            prompt_filter=prompt_filter,
            loss_params=loss_params,
            length_config=length_config,
            client_chunking=client_chunking,
        )
        run_config["runtime_provenance"] = runtime_provenance
        run_config["verifier_self_check"] = verifier_self_check
        _write_json(output_dir / "run_config.json", run_config)
        wandb_run = _maybe_init_wandb(args, run_id=run_id, output_dir=output_dir, run_config=run_config)
        _write_jsonl(
            metrics_path,
            {
                "event": "startup",
                "run_id": run_id,
                **run_config,
            },
        )
        _wandb_log(
            wandb_run,
            {
                "run/started": True,
                "run/num_examples": len(examples),
                **_numeric_metrics("run/prompt_filter", prompt_filter),
                "run/steps": args.steps,
                "run/prompts_per_step": args.prompts_per_step,
                "run/samples_per_prompt": args.samples_per_prompt,
                "run/rollouts_per_step": args.prompts_per_step * args.samples_per_prompt,
                "run/max_prompt_tokens": args.max_prompt_tokens,
                "run/max_generate_tokens": args.max_generate_tokens,
                "run/max_model_tokens": args.max_model_tokens,
                "run/context_token_reserve": args.context_token_reserve,
                "run/xorl_client_max_chunk_len": client_chunking["max_chunk_len"],
                "run/xorl_client_max_chunk_bytes": client_chunking["max_chunk_bytes"],
                "run/xorl_client_min_tail_chunk_len": client_chunking["min_tail_chunk_len"],
                "run/xorl_client_sequential_chunks": client_chunking["sequential_chunks"],
                "run/direct_train_server_futures": args.direct_train_server_futures,
                "run/pipeline_rl": args.pipeline_rl,
            },
            policy_step=0,
        )
        _log_driver_memory(metrics_path, wandb_run, step=None, phase="startup", policy_step=0, force_gc=False)
        for endpoint_index, (host, port) in enumerate(endpoints):
            add_result = _wait_for_future_result(
                training_client.add_inference_endpoint(
                    host=host,
                    port=port,
                    world_size=args.inference_world_size,
                    sync_weights=False,
                    master_port=args.sync_master_port,
                    buffer_size_mb=args.sync_buffer_size_mb,
                ),
                label=f"add_inference_endpoint {host}:{port}",
                timeout_s=args.future_timeout_s,
                metrics_path=metrics_path,
                step=None,
                policy_step=0,
                event_prefix="endpoint_future",
            )
            _write_jsonl(
                metrics_path,
                {
                    "event": "add_inference_endpoint",
                    "endpoint_index": endpoint_index,
                    "host": host,
                    "port": port,
                    "accepted": _endpoint_registration_accepted(add_result),
                    "result": _jsonable(add_result),
                },
            )
            if not _endpoint_registration_accepted(add_result):
                raise RuntimeError(f"add_inference_endpoint failed for {host}:{port}: {add_result}")

        current_weight_version = _policy_weight_version(args, run_id=run_id, policy_step=0)
        if not args.skip_initial_sync:
            sync_result = _sync_inference_weights(
                training_client,
                master_port=args.sync_master_port,
                buffer_size_mb=args.sync_buffer_size_mb,
                cache_invalidation_mode=args.sync_cache_invalidation_mode,
                pause_mode=args.sync_pause_mode,
                weight_version=current_weight_version,
                timeout_s=args.sync_timeout_s,
                metrics_path=metrics_path,
                event="initial_weight_sync",
                step=None,
                policy_step=0,
            )
            _write_jsonl(
                metrics_path,
                {
                    "event": "initial_weight_sync",
                    "weight_version": current_weight_version,
                    "result": _jsonable(sync_result),
                },
            )
            sync_payload = _jsonable(sync_result)
            _wandb_log(
                wandb_run,
                {
                    "sync/initial_success": bool(sync_payload.get("success", False)),
                    "sync/initial_transfer_time": sync_payload.get("transfer_time"),
                    "sync/initial_total_bytes": sync_payload.get("total_bytes"),
                    "sync/initial_cache_epoch": sync_payload.get("cache_epoch"),
                    **_numeric_metrics("sync/initial", sync_payload),
                },
                policy_step=0,
            )
            sync_payload = _require_sync_success(
                sync_result,
                event="initial_weight_sync",
                weight_version=current_weight_version,
            )

        if args.pipeline_rl:
            prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        for step in range(args.steps):
            rollout_generated_weight_version = current_weight_version
            if prefetch is not None:
                if prefetch.step != step:
                    raise RuntimeError(f"Pipeline prefetch step mismatch: expected {step}, got {prefetch.step}")
                records, rollout_summary, prefetch_metrics = _consume_rollout_prefetch(
                    prefetch,
                    timeout_s=args.prefetch_timeout_s,
                    metrics_path=metrics_path,
                )
                rollout_summary.update(prefetch_metrics)
                rollout_generated_weight_version = prefetch.weight_version
                prefetch = None
            else:
                records, rollout_summary = _rollout_step(
                    sampling_urls=sampling_urls,
                    tokenizer=tokenizer,
                    examples=examples,
                    step=step,
                    args=args,
                    length_config=length_config,
                    metrics_path=metrics_path,
                    cache_extra_key=current_weight_version,
                    wandb_run=wandb_run,
                )
                rollout_summary.update(
                    {
                        "pipeline_prefetched": False,
                        "pipeline_generated_weight_version": current_weight_version,
                    }
                )

            rollout_summary.update(
                {
                    "policy_weight_version_at_train": current_weight_version,
                    "policy_weight_version_at_sampling": rollout_generated_weight_version,
                    "pipeline_policy_lag": int(
                        args.pipeline_rl and rollout_generated_weight_version != current_weight_version
                    ),
                }
            )
            advantages = compute_group_advantages(records)
            forced_advantages_for_smoke = False
            if args.force_nonzero_advantages_for_smoke:
                advantages, forced_advantages_for_smoke = _force_nonzero_advantages_for_smoke(advantages)
            nonzero_advantages = sum(1 for advantage in advantages if abs(advantage) > 1e-12)
            step_payload = {
                "event": "step_rollout_summary",
                "step": step,
                **rollout_summary,
                "nonzero_advantages": nonzero_advantages,
                "forced_nonzero_advantages_for_smoke": forced_advantages_for_smoke,
            }
            print(json.dumps(step_payload, sort_keys=True), flush=True)
            _write_jsonl(metrics_path, step_payload)
            _wandb_log(
                wandb_run,
                {
                    **_numeric_metrics("rollout", rollout_summary),
                    "rollout/nonzero_advantages": nonzero_advantages,
                    "rollout/forced_nonzero_advantages_for_smoke": forced_advantages_for_smoke,
                },
                policy_step=step + 1,
            )

            if args.pipeline_rl and prefetch_executor is not None and step + 1 < args.steps:
                prefetch = _start_rollout_prefetch(
                    executor=prefetch_executor,
                    sampling_urls=sampling_urls,
                    tokenizer=tokenizer,
                    examples=examples,
                    step=step + 1,
                    args=args,
                    length_config=length_config,
                    metrics_path=metrics_path,
                    cache_extra_key=current_weight_version,
                    wandb_run=wandb_run,
                )

            if nonzero_advantages == 0:
                _write_jsonl(metrics_path, {"event": "skip_update_all_zero_advantages", "step": step})
                del records, advantages, rollout_summary
                if args.driver_gc:
                    _log_driver_memory(
                        metrics_path,
                        wandb_run,
                        step=step,
                        phase="skip_update_cleanup",
                        policy_step=step + 1,
                        force_gc=True,
                    )
                continue

            train_pairs = _nonzero_advantage_pairs(records, advantages)
            records_before_count = len(records)
            train_records_count = len(train_pairs)
            train_completion_tokens = sum(len(record.completion_tokens) for record, _ in train_pairs)
            train_prompt_tokens = sum(len(record.prefix_tokens) for record, _ in train_pairs)
            _write_jsonl(
                metrics_path,
                {
                    "event": "train_batch_filter",
                    "step": step,
                    "records_before": records_before_count,
                    "records_after": train_records_count,
                    "completion_tokens_after": train_completion_tokens,
                    "prompt_tokens_after": train_prompt_tokens,
                },
            )

            datums = [build_drgrpo_datum(record, advantage) for record, advantage in train_pairs]
            forward_backward_chunk_estimate = _estimate_forward_backward_chunks(
                datums,
                min_tail_chunk_len=int(client_chunking["min_tail_chunk_len"]),
            )
            del train_pairs, records, advantages
            if args.driver_gc:
                _log_driver_memory(
                    metrics_path,
                    wandb_run,
                    step=step,
                    phase="after_datum_build_cleanup",
                    policy_step=step + 1,
                    force_gc=True,
                )
            fb_result = _forward_backward_result(
                training_client,
                datums,
                loss_fn="drgrpo",
                loss_fn_params=loss_params,
                sequential_chunks=args.xorl_client_sequential_chunks,
                timeout_s=args.future_timeout_s,
                metrics_path=metrics_path,
                step=step,
                policy_step=step + 1,
                min_tail_chunk_len=int(client_chunking["min_tail_chunk_len"]),
                direct_server_futures=args.direct_train_server_futures,
            )
            derived_train_metrics = _derived_train_metrics(fb_result.metrics)
            optim_result = _optim_step_result(
                training_client,
                types.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    eps=args.adam_eps,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                ),
                timeout_s=args.future_timeout_s,
                metrics_path=metrics_path,
                step=step,
                policy_step=step + 1,
                direct_server_futures=args.direct_train_server_futures,
            )
            update_payload = {
                "event": "train_update",
                "step": step,
                "forward_backward_metrics": fb_result.metrics,
                "forward_backward_chunk_estimate": forward_backward_chunk_estimate,
                "train_batch_filter": {
                    "records_before": records_before_count,
                    "records_after": train_records_count,
                    "completion_tokens_after": train_completion_tokens,
                    "prompt_tokens_after": train_prompt_tokens,
                },
                "derived_train_metrics": derived_train_metrics,
                "behavior_k3": _behavior_k3_from_metrics(fb_result.metrics),
                "optim_metrics": optim_result.metrics,
            }
            print(json.dumps(update_payload, sort_keys=True), flush=True)
            _write_jsonl(metrics_path, update_payload)
            policy_step = step + 1
            last_update_policy_step = policy_step
            _wandb_log(
                wandb_run,
                {
                    "train/behavior_k3": update_payload["behavior_k3"],
                    "train/records_before_filter": records_before_count,
                    "train/records_after_filter": train_records_count,
                    "train/completion_tokens_after_filter": train_completion_tokens,
                    **_numeric_metrics("train/forward_backward", fb_result.metrics),
                    **_numeric_metrics("train/chunks", forward_backward_chunk_estimate),
                    **_numeric_metrics("train/derived", derived_train_metrics),
                    **_numeric_metrics("optim", optim_result.metrics),
                },
                policy_step=policy_step,
            )
            del datums, fb_result, optim_result
            if args.driver_gc:
                _log_driver_memory(
                    metrics_path,
                    wandb_run,
                    step=step,
                    phase="after_train_update_cleanup",
                    policy_step=policy_step,
                    force_gc=True,
                )

            if not args.skip_weight_sync:
                next_weight_version = _policy_weight_version(args, run_id=run_id, policy_step=policy_step)
                if args.pipeline_rl and prefetch is not None:
                    _wait_for_prefetch_before_sync(
                        prefetch,
                        metrics_path=metrics_path,
                        wandb_run=wandb_run,
                        step=step,
                        policy_step=policy_step,
                        timeout_s=args.prefetch_timeout_s,
                    )
                sync_result = _sync_inference_weights(
                    training_client,
                    master_port=args.sync_master_port,
                    buffer_size_mb=args.sync_buffer_size_mb,
                    cache_invalidation_mode=args.sync_cache_invalidation_mode,
                    pause_mode=args.sync_pause_mode,
                    weight_version=next_weight_version,
                    timeout_s=args.sync_timeout_s,
                    metrics_path=metrics_path,
                    event="weight_sync",
                    step=step,
                    policy_step=policy_step,
                )
                current_weight_version = next_weight_version
                _write_jsonl(
                    metrics_path,
                    {
                        "event": "weight_sync",
                        "step": step,
                        "weight_version": next_weight_version,
                        "pipeline_waited_for_prefetch": bool(args.pipeline_rl and prefetch is not None),
                        "result": _jsonable(sync_result),
                    },
                )
                sync_payload = _jsonable(sync_result)
                _wandb_log(wandb_run, _numeric_metrics("sync", sync_payload), policy_step=policy_step)
                sync_payload = _require_sync_success(
                    sync_result,
                    event="weight_sync",
                    weight_version=next_weight_version,
                )

            if args.checkpoint_interval and policy_step % args.checkpoint_interval == 0:
                checkpoint_result = _save_checkpoint(
                    training_client,
                    args=args,
                    run_id=run_id,
                    policy_step=policy_step,
                    metrics_path=metrics_path,
                )
                saved_checkpoint_steps.add(policy_step)
                checkpoint_payload = {
                    "event": "checkpoint",
                    "step": step,
                    "policy_step": policy_step,
                    "name": _checkpoint_name(args, run_id=run_id, policy_step=policy_step),
                    "result": _jsonable(checkpoint_result),
                }
                _write_jsonl(metrics_path, checkpoint_payload)
                _wandb_log(wandb_run, _numeric_metrics("checkpoint", checkpoint_payload), policy_step=policy_step)
                del checkpoint_result, checkpoint_payload

            del update_payload, rollout_summary, forward_backward_chunk_estimate, derived_train_metrics
            if not args.skip_weight_sync:
                del sync_result, sync_payload
            if args.driver_gc:
                _log_driver_memory(
                    metrics_path,
                    wandb_run,
                    step=step,
                    phase="end_step_cleanup",
                    policy_step=policy_step,
                    force_gc=True,
                )

        if (
            args.save_final_checkpoint
            and last_update_policy_step is not None
            and last_update_policy_step not in saved_checkpoint_steps
        ):
            checkpoint_result = _save_checkpoint(
                training_client,
                args=args,
                run_id=run_id,
                policy_step=last_update_policy_step,
                metrics_path=metrics_path,
            )
            checkpoint_payload = {
                "event": "checkpoint",
                "step": last_update_policy_step - 1,
                "policy_step": last_update_policy_step,
                "name": _checkpoint_name(args, run_id=run_id, policy_step=last_update_policy_step),
                "result": _jsonable(checkpoint_result),
            }
            _write_jsonl(metrics_path, checkpoint_payload)
            _wandb_log(wandb_run, _numeric_metrics("checkpoint", checkpoint_payload), policy_step=last_update_policy_step)
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False, cancel_futures=True)
        if wandb_run is not None:
            wandb_run.finish()
        if args.remove_endpoint_on_exit:
            for host, port in endpoints:
                try:
                    remove_result = _wait_for_future_result(
                        training_client.remove_inference_endpoint(host=host, port=port),
                        label=f"remove_inference_endpoint {host}:{port}",
                        timeout_s=min(args.future_timeout_s, 300.0),
                        metrics_path=metrics_path,
                        step=None,
                        policy_step=last_update_policy_step,
                        event_prefix="endpoint_future",
                    )
                    _write_jsonl(
                        metrics_path,
                        {
                            "event": "remove_inference_endpoint",
                            "host": host,
                            "port": port,
                            "result": _jsonable(remove_result),
                        },
                    )
                except Exception as exc:
                    _write_jsonl(
                        metrics_path,
                        {"event": "remove_inference_endpoint_failed", "host": host, "port": port, "error": repr(exc)},
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
