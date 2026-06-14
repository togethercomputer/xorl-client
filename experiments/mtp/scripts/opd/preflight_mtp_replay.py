#!/usr/bin/env python3
"""Preflight SingleShot MTP rollout replay batches before launching OPD training."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.opd.run_opd_pipeline import (  # noqa: E402
    _decode_token_ids,
    _infer_local_dataset_type,
    _iter_prompt_dataset_rows,
    _load_rollout_tokenizer,
    _prompt_from_dataset_row,
    _student_sample_for_opd_batch,
)
from xorl.mtp.singleshot import prepare_singleshot_mtp_opd_batch  # noqa: E402


IGNORE_INDEX = -100
TOKEN_KIND_NAMES = {
    0: "prompt",
    1: "refill",
    2: "mask",
    3: "pad",
}


def _json_arg(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    return json.loads(raw)


def _decode_one(tokenizer: Any | None, token_id: int) -> str | None:
    if tokenizer is None or token_id < 0:
        return None
    return _decode_token_ids(tokenizer, [int(token_id)], max_chars=80)


def _load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.rollout_samples_jsonl:
        samples: list[dict[str, Any]] = []
        with Path(args.rollout_samples_jsonl).open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if len(samples) >= args.num_samples:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt_token_ids")
                prompt_source = "prompt_token_ids"
                if not isinstance(prompt, list):
                    prompt = record.get("prompt_tail_token_ids")
                    prompt_source = "prompt_tail_token_ids"
                continuation = record.get("generated_token_ids_full")
                continuation_source = "generated_token_ids_full"
                if not isinstance(continuation, list):
                    continuation = record.get("generated_token_ids")
                    continuation_source = "generated_token_ids"
                if not isinstance(prompt, list) or not isinstance(continuation, list) or len(prompt) < 1:
                    continue
                samples.append(
                    {
                        "sample_idx": int(record.get("sample_idx", idx)),
                        "dataset_row_idx": int(record.get("dataset_row_idx", record.get("prompt_dataset_row_indices", [idx])[0] if isinstance(record.get("prompt_dataset_row_indices"), list) and record.get("prompt_dataset_row_indices") else idx)),
                        "prompt": [int(x) for x in prompt],
                        "continuation": [int(x) for x in continuation[: args.max_new_tokens]],
                        "metadata": {**record, "_preflight_prompt_source": prompt_source},
                        "continuation_source": f"rollout_samples_jsonl:{continuation_source}",
                        "native_mtp_trace": record.get("native_mtp_debug_trace"),
                    }
                )
        if not samples:
            raise ValueError(f"No usable samples found in {args.rollout_samples_jsonl}")
        return samples

    ds_type = args.prompt_dataset_type or _infer_local_dataset_type(Path(args.prompt_dataset_path))
    assistant_role_token_ids = _json_arg(args.prompt_dataset_assistant_role_token_ids_json, [77091])
    accepted = 0
    samples: list[dict[str, Any]] = []

    for row_idx, token_ids in _iter_prompt_dataset_rows(
        path=args.prompt_dataset_path,
        ds_type=ds_type,
        split=args.prompt_dataset_split,
        column=args.prompt_dataset_column,
    ):
        if token_ids is None:
            continue
        row_tokens = [int(token_id) for token_id in list(token_ids)]
        prompt_result = _prompt_from_dataset_row(
            row_tokens,
            prompt_len=args.prompt_dataset_prompt_len,
            turn_strategy=args.prompt_dataset_turn_strategy,
            im_start_token_id=args.prompt_dataset_im_start_token_id,
            im_end_token_id=args.prompt_dataset_im_end_token_id,
            assistant_role_token_ids=assistant_role_token_ids,
            min_target_tokens=args.prompt_dataset_min_target_tokens,
            assistant_offset_tokens=args.prompt_dataset_assistant_offset_tokens,
        )
        if prompt_result is None:
            continue
        if accepted < args.prompt_dataset_offset:
            accepted += 1
            continue
        prompt, metadata = prompt_result
        if args.prompt_dataset_turn_strategy == "assistant":
            continuation_start = int(metadata["assistant_prompt_end"])
            continuation_end = int(metadata["assistant_content_end"])
        else:
            continuation_start = len(prompt)
            continuation_end = min(len(row_tokens), continuation_start + args.max_new_tokens)
        continuation = row_tokens[continuation_start:continuation_end][: args.max_new_tokens]
        if len(continuation) < args.min_continuation_tokens:
            accepted += 1
            continue
        samples.append(
            {
                "sample_idx": len(samples),
                "dataset_row_idx": row_idx,
                "prompt": prompt,
                "continuation": continuation,
                "metadata": metadata,
            }
        )
        accepted += 1
        if len(samples) >= args.num_samples:
            break

    if not samples:
        raise RuntimeError("No usable prompt/continuation samples found for preflight")
    return samples


def _singleshot_mtp_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _json_arg(args.singleshot_mtp_json, {})
    config.setdefault("k_toks", args.k_toks)
    config.setdefault("mask_token_id", args.mask_token_id)
    config.setdefault("pad_token_id", args.pad_token_id)
    config.setdefault("pad_to_multiple", args.pad_to_multiple)
    config.setdefault("prompt_pad_to_multiple", args.prompt_pad_to_multiple)
    config.setdefault("rollout_replay", True)
    config.setdefault("sample_rollout", True)
    config.setdefault("sampling_mode", "prompt_logprob")
    config.setdefault("flex_block_size", args.flex_block_size)
    config.setdefault("loss_start_key", "mtp_loss_start")
    return config


def _replace_with_student_rollouts(args: argparse.Namespace, samples: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [list(sample["prompt"]) for sample in samples]
    singleshot_mtp = _singleshot_mtp_config(args)
    sampling_params = _json_arg(args.student_sampling_params_json, {"top_k": 1})
    sequences, metrics = _student_sample_for_opd_batch(
        args.student_url,
        prompts,
        args.max_new_tokens,
        singleshot_mtp=singleshot_mtp,
        temperature=args.student_temperature,
        timeout=args.request_timeout,
        sampling_params_extra=sampling_params,
    )
    sample_metadata = metrics.get("_student_sample_metadata", [])
    for sample_idx, (sample, sequence) in enumerate(zip(samples, sequences)):
        prompt_len = len(sample["prompt"])
        sample["continuation"] = list(sequence[prompt_len:])
        sample["continuation_source"] = "student_rollout"
        sample["sampling_metrics"] = metrics
        if sample_idx < len(sample_metadata) and isinstance(sample_metadata[sample_idx], dict):
            trace = sample_metadata[sample_idx].get("mtp_debug_trace")
            if isinstance(trace, list):
                sample["native_mtp_trace"] = trace
    return metrics


def _prepare_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sequence = list(sample["prompt"]) + list(sample["continuation"])
    prompt_len = len(sample["prompt"])
    batch = {
        "input_ids": torch.tensor([sequence[:-1]], dtype=torch.long),
        "target_tokens": torch.tensor([sequence[1:]], dtype=torch.long),
        "teacher_cache_indices": torch.arange(len(sequence) - 1, dtype=torch.long).view(1, -1),
        "mtp_loss_start": torch.tensor([prompt_len - 1], dtype=torch.long),
    }
    native_mtp_trace = sample.get("native_mtp_trace")
    if isinstance(native_mtp_trace, list):
        batch["singleshot_mtp_native_trace"] = [[json.dumps(native_mtp_trace, separators=(",", ":"))]]
    return prepare_singleshot_mtp_opd_batch(
        batch,
        k_toks=args.k_toks,
        mask_token_id=args.mask_token_id,
        pad_token_id=args.pad_token_id,
        pad_to_multiple=args.pad_to_multiple,
        prompt_pad_to_multiple=args.prompt_pad_to_multiple,
        ignore_index=IGNORE_INDEX,
        loss_start_key="mtp_loss_start",
        rollout_replay=True,
        flex_block_size=args.flex_block_size,
        validate_native_mtp_trace=isinstance(native_mtp_trace, list) and len(native_mtp_trace) > 0,
    )


def _slot_table(
    prepared: dict[str, Any],
    *,
    tokenizer: Any | None,
    max_slots: int,
    around_predictions: int,
) -> list[dict[str, Any]]:
    input_ids = prepared["input_ids"][0]
    labels = prepared["labels"][0]
    position_ids = prepared["position_ids"][0]
    source_indices = prepared["_singleshot_mtp_source_indices"][0]
    mask_positions = prepared["_singleshot_mtp_mask_positions"][0]
    prediction_positions = prepared["_singleshot_mtp_prediction_positions"][0]
    prompt_positions = prepared["_singleshot_mtp_prompt_positions"][0]
    refill_positions = prepared["_singleshot_mtp_refill_positions"][0]
    native_commit_positions = prepared.get("_singleshot_mtp_native_commit_positions")
    if native_commit_positions is not None:
        native_commit_positions = native_commit_positions[0]
    native_emit_positions = prepared.get("_singleshot_mtp_native_emit_positions")
    if native_emit_positions is not None:
        native_emit_positions = native_emit_positions[0]
    token_kind = prepared["attention_mask"].token_kind[0]
    block_ids = prepared["attention_mask"].block_ids[0]

    selected: set[int] = set(range(min(max_slots, input_ids.numel())))
    pred_slots = prediction_positions.nonzero(as_tuple=False).reshape(-1).tolist()
    for slot in pred_slots[:8]:
        for pos in range(max(0, int(slot) - around_predictions), min(input_ids.numel(), int(slot) + around_predictions + 1)):
            selected.add(pos)
    for slot in pred_slots[-8:]:
        for pos in range(max(0, int(slot) - around_predictions), min(input_ids.numel(), int(slot) + around_predictions + 1)):
            selected.add(pos)

    table: list[dict[str, Any]] = []
    for slot in sorted(selected):
        token_id = int(input_ids[slot].item())
        label_id = int(labels[slot].item())
        table.append(
            {
                "slot": slot,
                "kind": TOKEN_KIND_NAMES.get(int(token_kind[slot].item()), f"kind{int(token_kind[slot].item())}"),
                "input_id": token_id,
                "input_text": _decode_one(tokenizer, token_id),
                "label_id": None if label_id == IGNORE_INDEX else label_id,
                "label_text": None if label_id == IGNORE_INDEX else _decode_one(tokenizer, label_id),
                "position_id": int(position_ids[slot].item()),
                "source_index": int(source_indices[slot].item()),
                "block_id": int(block_ids[slot].item()),
                "is_prompt": bool(prompt_positions[slot].item()),
                "is_refill": bool(refill_positions[slot].item()),
                "is_mask": bool(mask_positions[slot].item()),
                "is_prediction": bool(prediction_positions[slot].item()),
                "is_native_commit": (
                    bool(native_commit_positions[slot].item()) if native_commit_positions is not None else False
                ),
                "is_native_emit": (
                    bool(native_emit_positions[slot].item()) if native_emit_positions is not None else False
                ),
            }
        )
    return table


def _native_trace_rows_head(native_mtp_steps: list[dict[str, Any]], tokenizer: Any | None, *, max_steps: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in native_mtp_steps[:max_steps]:
        if not isinstance(step, dict):
            continue
        commit_slice = step.get("commit_slice") if isinstance(step.get("commit_slice"), dict) else {}
        emit_window = step.get("emit_window") if isinstance(step.get("emit_window"), dict) else {}
        input_ids = step.get("input_row_token_ids", [])
        pending_ids = step.get("pending_token_ids", step.get("pending_saved_for_next_step", []))
        committed_ids = step.get("committed_appended_to_output_ids", step.get("committed_token_ids", []))
        if not isinstance(input_ids, list):
            input_ids = []
        if not isinstance(pending_ids, list):
            pending_ids = []
        if not isinstance(committed_ids, list):
            committed_ids = []
        rows.append(
            {
                "step_idx": step.get("step_idx"),
                "phase": step.get("phase"),
                "q_len": step.get("decode_q_len_per_req", step.get("decode_q_len_per_req_runtime")),
                "recompute_len": step.get("recompute_len_for_step", step.get("recompute_len")),
                "attempt_k": step.get("attempt_k_for_step", step.get("attempt_k")),
                "commit_start": commit_slice.get("start", step.get("commit_start")),
                "commit_len": commit_slice.get("len", step.get("commit_len")),
                "emit_start": emit_window.get("start", step.get("emit_window_start")),
                "emit_len": emit_window.get("len", step.get("emit_window_len")),
                "input_ids_head": [int(x) for x in input_ids[:24]],
                "input_text_head": _decode_token_ids(tokenizer, [int(x) for x in input_ids[:24]], max_chars=1000),
                "pending_ids_head": [int(x) for x in pending_ids[:24]],
                "pending_text_head": _decode_token_ids(tokenizer, [int(x) for x in pending_ids[:24]], max_chars=1000),
                "committed_ids_head": [int(x) for x in committed_ids[:24]],
                "committed_text_head": _decode_token_ids(tokenizer, [int(x) for x in committed_ids[:24]], max_chars=1000),
            }
        )
    return rows


def _mask_invariants(prepared: dict[str, Any]) -> dict[str, Any]:
    replay_mask = prepared["attention_mask"]
    allowed = replay_mask.allowed()[0]
    token_kind = replay_mask.token_kind[0]
    key_source = replay_mask.key_source_indices[0]
    q_context = replay_mask.query_context_source_indices[0]
    block_ids = replay_mask.block_ids[0]
    prediction_positions = prepared["_singleshot_mtp_prediction_positions"][0]

    pad = token_kind.eq(3)
    real = ~pad
    context_key = token_kind.eq(0) | token_kind.eq(1)
    mask_key = token_kind.eq(2)
    pred_slots = prediction_positions.nonzero(as_tuple=False).reshape(-1)

    failures: list[str] = []
    if allowed[real][:, pad].any():
        failures.append("real query attends to a pad key")
    for q in pad.nonzero(as_tuple=False).reshape(-1).tolist():
        visible = allowed[q].nonzero(as_tuple=False).reshape(-1).tolist()
        if visible != [q]:
            failures.append(f"pad query {q} is not self-only: {visible[:16]}")
            break

    per_prediction: list[dict[str, Any]] = []
    for q_tensor in pred_slots[:32]:
        q = int(q_tensor.item())
        visible = allowed[q]
        future_context = visible & context_key & (key_source > q_context[q])
        other_mask_block = visible & mask_key & block_ids.ge(0) & block_ids.ne(block_ids[q])
        if future_context.any():
            failures.append(f"prediction slot {q} attends to future context sources")
        if other_mask_block.any():
            failures.append(f"prediction slot {q} attends to another MTP mask block")
        visible_slots = visible.nonzero(as_tuple=False).reshape(-1)
        per_prediction.append(
            {
                "slot": q,
                "kind": TOKEN_KIND_NAMES.get(int(token_kind[q].item()), f"kind{int(token_kind[q].item())}"),
                "source_index": int(prepared["_singleshot_mtp_source_indices"][0, q].item()),
                "query_context_source_index": int(q_context[q].item()),
                "visible_slot_count": int(visible_slots.numel()),
                "visible_context_source_min": (
                    int(key_source[visible & context_key].min().item()) if (visible & context_key).any() else None
                ),
                "visible_context_source_max": (
                    int(key_source[visible & context_key].max().item()) if (visible & context_key).any() else None
                ),
                "visible_mask_slots": [int(x) for x in visible_slots[mask_key[visible_slots]].tolist()[:32]],
            }
        )

    return {
        "ok": not failures,
        "failures": failures,
        "allowed_shape": list(allowed.shape),
        "real_to_pad_allowed": bool(allowed[real][:, pad].any()),
        "prediction_count": int(prediction_positions.sum().item()),
        "prediction_checks_head": per_prediction,
    }


def _summarize_sample(sample: dict[str, Any], prepared: dict[str, Any], tokenizer: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    valid_labels = prepared["labels"][0].ne(IGNORE_INDEX)
    native_mtp_trace = sample.get("native_mtp_trace")
    native_mtp_steps = native_mtp_trace if isinstance(native_mtp_trace, list) else []
    native_mtp_q_lens = [
        int(step["decode_q_len_per_req"])
        for step in native_mtp_steps
        if isinstance(step, dict) and step.get("decode_q_len_per_req") is not None
    ]
    native_mtp_phases = [
        str(step["phase"])
        for step in native_mtp_steps
        if isinstance(step, dict) and step.get("phase") is not None
    ]
    return {
        "sample_idx": sample["sample_idx"],
        "dataset_row_idx": sample["dataset_row_idx"],
        "continuation_source": sample.get("continuation_source", "dataset_slice"),
        "sampling_metrics": sample.get("sampling_metrics"),
        "native_mtp_debug_trace_step_count": len(native_mtp_steps),
        "native_mtp_debug_trace_phases": ",".join(native_mtp_phases),
        "native_mtp_debug_trace_q_lens": ",".join(str(q_len) for q_len in native_mtp_q_lens),
        "native_mtp_trace_rows_head": _native_trace_rows_head(native_mtp_steps, tokenizer),
        "prompt_len": len(sample["prompt"]),
        "continuation_len": len(sample["continuation"]),
        "prompt_text_tail": _decode_token_ids(tokenizer, sample["prompt"][-64:], max_chars=3000),
        "continuation_text_head": _decode_token_ids(tokenizer, sample["continuation"][: args.max_new_tokens], max_chars=3000),
        "metadata": sample["metadata"],
        "config": {
            "k_toks": args.k_toks,
            "max_new_tokens": args.max_new_tokens,
            "pad_to_multiple": args.pad_to_multiple,
            "prompt_pad_to_multiple": args.prompt_pad_to_multiple,
            "mask_token_id": args.mask_token_id,
            "pad_token_id": args.pad_token_id,
            "flex_block_size": args.flex_block_size,
        },
        "replay_shape": {
            "input_shape": list(prepared["input_ids"].shape),
            "raw_seq_len": int(prepared["_singleshot_mtp_raw_seq_len"].item()),
            "padded_seq_len": int(prepared["_singleshot_mtp_padded_seq_len"].item()),
            "prompt_pad_count": int(prepared["_singleshot_mtp_prompt_pad_count"].item()),
            "prompt_padded_len": int(prepared["_singleshot_mtp_prompt_padded_len"].item()),
            "generated_token_count": int(prepared["_singleshot_mtp_generated_token_count"].item()),
            "chunk_count": int(prepared["_singleshot_mtp_chunk_count"].item()),
            "valid_label_count": int(valid_labels.sum().item()),
            "native_trace_replay": bool(prepared.get("_singleshot_mtp_native_trace_replay", torch.tensor(False)).item()),
        },
        "selected_label_ids_head": [int(x) for x in prepared["labels"][0, valid_labels][:64].tolist()],
        "selected_label_text_head": _decode_token_ids(tokenizer, [int(x) for x in prepared["labels"][0, valid_labels][:64].tolist()], max_chars=3000),
        "slot_table": _slot_table(
            prepared,
            tokenizer=tokenizer,
            max_slots=args.slot_table_head,
            around_predictions=args.slot_table_around_predictions,
        ),
        "mask_invariants": _mask_invariants(prepared),
    }


def _synthetic_left_pad_example(tokenizer: Any | None, args: argparse.Namespace) -> dict[str, Any]:
    del tokenizer
    prompt = [10, 11, 12, 13, 14]
    continuation = [15, 16, 17, 18]
    sequence = prompt + continuation
    batch = {
        "input_ids": torch.tensor([sequence[:-1]], dtype=torch.long),
        "target_tokens": torch.tensor([sequence[1:]], dtype=torch.long),
        "mtp_loss_start": torch.tensor([len(prompt) - 1], dtype=torch.long),
    }
    prepared = prepare_singleshot_mtp_opd_batch(
        batch,
        k_toks=2,
        mask_token_id=args.mask_token_id,
        pad_token_id=args.pad_token_id,
        pad_to_multiple=None,
        prompt_pad_to_multiple=2,
        ignore_index=IGNORE_INDEX,
        rollout_replay=True,
    )
    return {
        "prompt_ids": prompt,
        "continuation_ids": continuation,
        "input_ids": [int(x) for x in prepared["input_ids"][0].tolist()],
        "labels": [None if int(x) == IGNORE_INDEX else int(x) for x in prepared["labels"][0].tolist()],
        "position_ids": [int(x) for x in prepared["position_ids"][0].tolist()],
        "source_indices": [int(x) for x in prepared["_singleshot_mtp_source_indices"][0].tolist()],
        "prediction_positions": [bool(x) for x in prepared["_singleshot_mtp_prediction_positions"][0].tolist()],
        "slot_table": _slot_table(prepared, tokenizer=None, max_slots=64, around_predictions=0),
        "mask_invariants": _mask_invariants(prepared),
    }


def _flex_forward_backward_smoke(prepared: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "cuda is not available"}

    from xorl.models.layers.attention.backend.flex_attention import flex_attention_forward  # noqa: PLC0415

    device = torch.device("cuda")
    replay_mask = prepared["attention_mask"].to(device)
    block_mask = replay_mask.to_block_mask(device=device)
    batch, seq_len = prepared["input_ids"].shape
    heads = args.flex_smoke_heads
    dim = args.flex_smoke_head_dim
    dtype = torch.bfloat16

    timings_ms: list[float] = []
    grad_norms: list[float] = []
    for _ in range(args.flex_smoke_repeats):
        query = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype, requires_grad=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out, _ = flex_attention_forward(
            SimpleNamespace(),
            query,
            key,
            value,
            block_mask,
            scaling=1.0 / math.sqrt(dim),
        )
        loss = out.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        grad_norms.append(float(query.grad.float().norm().item()))

    return {
        "skipped": False,
        "device": torch.cuda.get_device_name(device),
        "shape": [batch, seq_len, heads, dim],
        "dtype": str(dtype).removeprefix("torch."),
        "repeats": args.flex_smoke_repeats,
        "timings_ms": timings_ms,
        "query_grad_norms": grad_norms,
        "all_grad_norms_finite": all(math.isfinite(x) and x > 0 for x in grad_norms),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-dataset-path", default="")
    parser.add_argument("--rollout-samples-jsonl", default=None)
    parser.add_argument("--prompt-dataset-type", default="parquet")
    parser.add_argument("--prompt-dataset-split", default="train")
    parser.add_argument("--prompt-dataset-column", default="input_ids")
    parser.add_argument("--prompt-dataset-num-prompts", type=int, default=2)
    parser.add_argument("--prompt-dataset-prompt-len", type=int, default=4096)
    parser.add_argument("--prompt-dataset-offset", type=int, default=0)
    parser.add_argument("--prompt-dataset-turn-strategy", default="assistant")
    parser.add_argument("--prompt-dataset-min-target-tokens", type=int, default=33)
    parser.add_argument("--prompt-dataset-assistant-offset-tokens", type=int, default=0)
    parser.add_argument("--prompt-dataset-im-start-token-id", type=int, default=151644)
    parser.add_argument("--prompt-dataset-im-end-token-id", type=int, default=151645)
    parser.add_argument("--prompt-dataset-assistant-role-token-ids-json", default="[77091]")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    parser.add_argument("--min-continuation-tokens", type=int, default=1)
    parser.add_argument("--k-toks", type=int, default=16)
    parser.add_argument("--mask-token-id", type=int, default=151662)
    parser.add_argument("--pad-token-id", type=int, default=151643)
    parser.add_argument("--pad-to-multiple", type=int, default=128)
    parser.add_argument("--prompt-pad-to-multiple", type=int, default=None)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--student-url", default=None)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--student-temperature", type=float, default=0.0)
    parser.add_argument("--student-sampling-params-json", default='{"top_k": 1}')
    parser.add_argument("--singleshot-mtp-json", default=None)
    parser.add_argument("--slot-table-head", type=int, default=24)
    parser.add_argument("--slot-table-around-predictions", type=int, default=2)
    parser.add_argument("--flex-smoke-repeats", type=int, default=2)
    parser.add_argument("--flex-smoke-heads", type=int, default=2)
    parser.add_argument("--flex-smoke-head-dim", type=int, default=32)
    args = parser.parse_args()

    if args.prompt_pad_to_multiple is None:
        args.prompt_pad_to_multiple = args.k_toks
    output_dir = Path(args.output_dir)
    tokenizer = _load_rollout_tokenizer(args.tokenizer_path)
    samples = _load_samples(args)
    sampling_metrics = None
    if args.student_url:
        sampling_metrics = _replace_with_student_rollouts(args, samples)
    prepared_samples = [_prepare_sample(sample, args) for sample in samples]
    sample_reports = [
        _summarize_sample(sample, prepared, tokenizer, args)
        for sample, prepared in zip(samples, prepared_samples)
    ]
    synthetic = _synthetic_left_pad_example(tokenizer, args)
    flex_smoke = _flex_forward_backward_smoke(prepared_samples[0], args)
    ok = all(report["mask_invariants"]["ok"] for report in sample_reports)
    ok = ok and synthetic["mask_invariants"]["ok"]
    ok = ok and (flex_smoke.get("skipped") or flex_smoke.get("all_grad_norms_finite"))

    _write_json(output_dir / "mtp_replay_preflight.json", {
        "ok": bool(ok),
        "sampling_metrics": sampling_metrics,
        "samples": sample_reports,
        "synthetic_left_pad_example": synthetic,
        "flex_forward_backward_smoke": flex_smoke,
    })
    with (output_dir / "mtp_replay_slots.jsonl").open("w", encoding="utf-8") as handle:
        for report in sample_reports:
            for row in report["slot_table"]:
                handle.write(json.dumps({"sample_idx": report["sample_idx"], **row}, sort_keys=True) + "\n")
    print(json.dumps({"ok": bool(ok), "output_dir": str(output_dir), "flex_smoke": flex_smoke}, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
