"""Replay rank-local post-slice microbatch tensor dumps through ModelRunner.

This is the middle benchmark between:

* API replay of captured /forward_backward requests, which still includes the
  server request processor / orchestrator / dispatcher path; and
* synthetic local training microbenchmarks, which do not use the real OPSD
  post-collation tensors.

Run under torchrun with the same GPU count/topology as the tensor dump:

    torchrun --nproc_per_node=8 experiments/wordle/standalone/replay_microbatch_tensors.py \
      --config experiments/wordle/configs/qwen3_6_35b_a3b_opsd_wordle_fullft_ep8_warmstart_sft48.yaml \
      --tensor-dir <run_dir>/microbatch_diagnostics \
      --replay-artifact <capture_run>/forward_backward_replay.jsonl \
      --output-dir <run_dir>/bare_tensor_replay
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from xorl.server.runner.model_runner import ModelRunner
from xorl.server.runner.setup import setup_distributed
from xorl.server.server_arguments import ServerArguments
from xorl.utils.compile_cache import configure_rank_local_compile_caches
from xorl.utils.device import get_device_type, synchronize


LOGGER = logging.getLogger(__name__)
H100_BF16_PEAK_FLOPS = 989e12
_FA_METADATA_KEYS = {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}


def _jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return payload


def _load_model_runner_config(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    if "model" in raw and isinstance(raw["model"], dict):
        config = copy.deepcopy(raw)
    else:
        allowed = {field.name for field in fields(ServerArguments)}
        kwargs = {key: value for key, value in raw.items() if key in allowed}
        server_args = ServerArguments(**kwargs)
        config = server_args.to_config_dict()
    config.setdefault("train", {})["output_dir"] = str(output_dir / "server_output")
    return config


def _load_replay_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def _loss_from_replay_artifact(path: Path, request_index: int) -> tuple[str, dict[str, Any]]:
    entries = _load_replay_entries(path)
    fb_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("request"), dict)
        and "forward_backward_input" in entry["request"]
    ]
    if request_index >= len(fb_entries):
        raise IndexError(f"{path} has {len(fb_entries)} forward_backward entries, request_index={request_index}")
    fbi = fb_entries[request_index]["request"]["forward_backward_input"]
    return str(fbi.get("loss_fn", "causallm_loss")), dict(fbi.get("loss_fn_params") or {})


def _rank_tensor_path(tensor_dir: Path, rank: int, request_fragment: str) -> Path:
    if request_fragment:
        pattern = f"microbatch_{request_fragment}_rank{rank:05d}.pt"
    else:
        pattern = f"microbatch_*_rank{rank:05d}.pt"
    matches = sorted(tensor_dir.glob(pattern))
    if len(matches) != 1:
        formatted = ", ".join(str(p) for p in matches[:10])
        raise RuntimeError(f"Expected exactly one tensor dump for rank {rank} with {pattern}; found {len(matches)}: {formatted}")
    return matches[0]


def _as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _all_reduce_sum(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=get_device_type())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_max(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=get_device_type())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _all_reduce_min(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=get_device_type())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def _rough_mfu(tokens: float, wall_s: float, world_size: int, active_params: float, peak_flops_per_gpu: float) -> float:
    if wall_s <= 0 or world_size <= 0 or active_params <= 0:
        return 0.0
    return 6.0 * active_params * tokens / (wall_s * world_size * peak_flops_per_gpu)


def _sequence_dim(tensor: torch.Tensor, seq_len: int) -> int | None:
    if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == seq_len:
        return 1
    if tensor.ndim >= 1 and tensor.shape[0] == seq_len:
        return 0
    return None


def _input_tokens(micro_batches: list[dict[str, Any]]) -> int:
    total = 0
    for batch in micro_batches:
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            total += int(input_ids.numel())
    return total


def _valid_tokens(micro_batches: list[dict[str, Any]]) -> int:
    total = 0
    for batch in micro_batches:
        labels = batch.get("labels", batch.get("target_tokens"))
        if isinstance(labels, torch.Tensor):
            total += int((labels != -100).sum().item())
    return total


def _coalesce_micro_batches(
    micro_batches: list[dict[str, Any]],
    *,
    coalesce_repeat: int,
    target_input_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Create one packed/ragged microbatch per rank by repeating sequence segments.

    The captured tensors already represent post-server, post-rank-slice microbatches.
    Repeating them with position_ids preserved creates a static packed/ragged batch:
    every repeated segment starts at position 0, so flash attention sees independent
    segments while the model still gets more real tokens in one forward/backward.
    """
    base_input_tokens = _input_tokens(micro_batches)
    repeat_factor = max(1, int(coalesce_repeat))
    if target_input_tokens > 0:
        if base_input_tokens <= 0:
            raise ValueError("--coalesce-target-input-tokens requires positive input tokens in the dump")
        repeat_factor = max(repeat_factor, math.ceil(target_input_tokens / base_input_tokens))
    if repeat_factor <= 1 and target_input_tokens <= 0:
        return micro_batches, 1

    pieces: list[dict[str, Any]] = []
    for _ in range(repeat_factor):
        pieces.extend(copy.deepcopy(micro_batches))
    if not pieces:
        raise ValueError("No microbatches to coalesce")

    input_lengths = []
    for piece in pieces:
        input_ids = piece.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Every coalesced microbatch must contain tensor input_ids")
        input_lengths.append(int(input_ids.shape[-1] if input_ids.ndim >= 2 else input_ids.shape[0]))

    out: dict[str, Any] = {}
    keys = set().union(*(piece.keys() for piece in pieces)) - _FA_METADATA_KEYS
    for key in sorted(keys):
        values = [piece.get(key) for piece in pieces]
        if all(isinstance(value, torch.Tensor) for value in values):
            dims = [_sequence_dim(value, seq_len) for value, seq_len in zip(values, input_lengths)]
            if all(dim is not None for dim in dims):
                dim0 = int(dims[0])
                if not all(int(dim) == dim0 for dim in dims):
                    raise ValueError(f"Cannot coalesce key={key}: inconsistent sequence dimensions {dims}")
                out[key] = torch.cat([value.clone() for value in values], dim=dim0)
            else:
                out[key] = values[0].clone()
        elif key == "num_samples":
            out[key] = sum(int(value or 0) for value in values)
        elif key == "batch_id":
            out[key] = 0
        elif key == "request_id":
            out[key] = values[0]
        else:
            out[key] = copy.deepcopy(values[0])
    out["coalesced_repeat_factor"] = repeat_factor
    out["coalesced_source_microbatches"] = len(micro_batches)
    return [out], repeat_factor


def _clear_gradients(runner: ModelRunner, model_id: str) -> None:
    synchronize()
    if runner.optimizer is not None:
        try:
            runner.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            runner.optimizer.zero_grad()
    if runner.model is not None:
        runner.model.zero_grad(set_to_none=True)
    runner._accumulated_valid_tokens[model_id] = 0
    runner._accumulated_active_microbatches[model_id] = 0
    runner._accumulated_active_voter_total[model_id] = 0
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Flat server YAML or nested ModelRunner YAML.")
    parser.add_argument("--tensor-dir", required=True, help="Directory containing microbatch_*_rankNNNNN.pt files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-artifact", default="", help="Captured forward_backward_replay.jsonl for loss_fn_params.")
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--request-fragment", default="", help="Optional microbatch_<fragment>_rankNNNNN.pt selector.")
    parser.add_argument("--loss-fn", default="", help="Override loss function. Defaults to replay artifact or tensor summary.")
    parser.add_argument("--loss-fn-params-json", default="", help="Optional JSON object merged into loss_fn_params.")
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--coalesce-repeat", type=int, default=1)
    parser.add_argument("--coalesce-target-input-tokens", type=int, default=0)
    parser.add_argument("--active-params", type=float, default=3.0e9)
    parser.add_argument("--peak-flops-per-gpu", type=float, default=H100_BF16_PEAK_FLOPS)
    parser.add_argument("--profile-phase-timings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opd-profile-timings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opd-profile-sync-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats must be >= 0")

    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    configure_rank_local_compile_caches()

    rank, world_size, local_rank, _cpu_group = setup_distributed()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank-{rank}][%(levelname)s][%(name)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        force=True,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "bare_tensor_replay_metrics.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)

    try:
        tensor_path = _rank_tensor_path(Path(args.tensor_dir), rank, args.request_fragment)
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        summary = dict(payload.get("summary") or {})
        micro_batches = payload.get("micro_batches")
        if not isinstance(micro_batches, list) or not micro_batches:
            raise ValueError(f"{tensor_path} did not contain a non-empty micro_batches list")
        micro_batches, coalesce_repeat_factor = _coalesce_micro_batches(
            micro_batches,
            coalesce_repeat=args.coalesce_repeat,
            target_input_tokens=args.coalesce_target_input_tokens,
        )

        loss_fn = args.loss_fn or str(summary.get("loss_fn") or "")
        loss_params = dict(payload.get("loss_fn_params") or summary.get("loss_fn_params") or {})
        if args.replay_artifact:
            artifact_loss_fn, artifact_loss_params = _loss_from_replay_artifact(Path(args.replay_artifact), args.request_index)
            loss_fn = args.loss_fn or artifact_loss_fn
            loss_params.update(artifact_loss_params)
        if args.loss_fn_params_json:
            loss_params.update(json.loads(args.loss_fn_params_json))
        if not loss_fn:
            raise ValueError("--loss-fn or --replay-artifact is required because tensor dump did not record loss_fn")

        loss_params["profile_phase_timings"] = bool(args.profile_phase_timings)
        loss_params["opd_profile_timings"] = bool(args.opd_profile_timings)
        loss_params["opd_profile_sync_cuda"] = bool(args.opd_profile_sync_cuda)

        local_input_tokens = float(_input_tokens(micro_batches))
        local_valid_tokens = float(_valid_tokens(micro_batches))
        global_input_tokens = _all_reduce_sum(local_input_tokens)
        dumped_global_valid_tokens = _all_reduce_sum(local_valid_tokens)
        min_local_input_tokens = _all_reduce_min(local_input_tokens)
        max_local_input_tokens = _all_reduce_max(local_input_tokens)
        min_coalesce_repeat_factor = _all_reduce_min(coalesce_repeat_factor)
        max_coalesce_repeat_factor = _all_reduce_max(coalesce_repeat_factor)

        config = _load_model_runner_config(Path(args.config), output_dir)
        if rank == 0:
            _jsonl(
                metrics_path,
                {
                    "event": "init",
                    "time": time.time(),
                    "config": str(args.config),
                    "tensor_dir": str(args.tensor_dir),
                    "replay_artifact": args.replay_artifact,
                    "request_index": args.request_index,
                    "request_id": summary.get("request_id"),
                    "loss_fn": loss_fn,
                    "world_size": world_size,
                    "global_input_tokens": global_input_tokens,
                    "dumped_global_valid_tokens": dumped_global_valid_tokens,
                    "min_local_input_tokens": min_local_input_tokens,
                    "max_local_input_tokens": max_local_input_tokens,
                    "coalesce_repeat": args.coalesce_repeat,
                    "coalesce_target_input_tokens": args.coalesce_target_input_tokens,
                    "min_coalesce_repeat_factor": min_coalesce_repeat_factor,
                    "max_coalesce_repeat_factor": max_coalesce_repeat_factor,
                    "active_params": args.active_params,
                    "peak_flops_per_gpu": args.peak_flops_per_gpu,
                },
            )

        runner = ModelRunner(config=config, rank=rank, world_size=world_size, local_rank=local_rank)

        total_repeats = args.warmup_repeats + args.repeat
        measured_wall_s = 0.0
        measured_valid_tokens = 0.0
        measured_input_tokens = 0.0
        for replay_idx in range(total_repeats):
            warmup = replay_idx < args.warmup_repeats
            repeat_idx = replay_idx if warmup else replay_idx - args.warmup_repeats
            repeat_label = "warmup" if warmup else "measured"

            synchronize()
            started = time.perf_counter()
            result = runner.forward_backward(
                copy.deepcopy(micro_batches),
                loss_fn=loss_fn,
                loss_fn_params=dict(loss_params),
                model_id=args.model_id,
            )
            wall_s = _all_reduce_max(time.perf_counter() - started)
            valid_tokens = _as_number(result.get("global_valid_tokens"), dumped_global_valid_tokens)
            input_tokens = global_input_tokens

            row = {
                "event": "forward_backward",
                "time": time.time(),
                "warmup": warmup,
                "repeat_idx": repeat_idx,
                "wall_s": wall_s,
                "valid_tokens": valid_tokens,
                "input_tokens": input_tokens,
                "valid_tokens_per_s": valid_tokens / wall_s if wall_s > 0 else 0.0,
                "input_tokens_per_s": input_tokens / wall_s if wall_s > 0 else 0.0,
                "rough_valid_mfu": _rough_mfu(
                    valid_tokens, wall_s, world_size, args.active_params, args.peak_flops_per_gpu
                ),
                "rough_input_mfu": _rough_mfu(
                    input_tokens, wall_s, world_size, args.active_params, args.peak_flops_per_gpu
                ),
                "result": {key: value for key, value in result.items() if isinstance(value, (int, float, str, bool))},
            }
            if rank == 0:
                _jsonl(metrics_path, row)
                print(
                    f"[{repeat_label} repeat={repeat_idx}] fb={wall_s:.2f}s "
                    f"valid_tok/s={row['valid_tokens_per_s']:.2f} input_tok/s={row['input_tokens_per_s']:.2f} "
                    f"mfu_valid={100.0 * row['rough_valid_mfu']:.4f}% "
                    f"mfu_input={100.0 * row['rough_input_mfu']:.4f}%",
                    flush=True,
                )

            if not warmup:
                measured_wall_s += wall_s
                measured_valid_tokens += valid_tokens
                measured_input_tokens += input_tokens
            _clear_gradients(runner, args.model_id)

        done = {
            "event": "done",
            "time": time.time(),
            "summary": {
                "forward_backward_wall_s": measured_wall_s,
                "forward_backward_requests": float(args.repeat),
                "valid_tokens": measured_valid_tokens,
                "input_tokens": measured_input_tokens,
                "valid_tokens_per_s": measured_valid_tokens / measured_wall_s if measured_wall_s > 0 else 0.0,
                "input_tokens_per_s": measured_input_tokens / measured_wall_s if measured_wall_s > 0 else 0.0,
                "rough_valid_mfu": _rough_mfu(
                    measured_valid_tokens, measured_wall_s, world_size, args.active_params, args.peak_flops_per_gpu
                ),
                "rough_input_mfu": _rough_mfu(
                    measured_input_tokens, measured_wall_s, world_size, args.active_params, args.peak_flops_per_gpu
                ),
            },
        }
        if rank == 0:
            _jsonl(metrics_path, done)
            summary_out = done["summary"]
            print(
                f"[done] measured_fb={measured_wall_s:.2f}s "
                f"valid_tok/s={summary_out['valid_tokens_per_s']:.2f} "
                f"input_tok/s={summary_out['input_tokens_per_s']:.2f} "
                f"mfu_valid={100.0 * summary_out['rough_valid_mfu']:.4f}% "
                f"mfu_input={100.0 * summary_out['rough_input_mfu']:.4f}%",
                flush=True,
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
