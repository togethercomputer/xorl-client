#!/usr/bin/env python3
"""Paper-style static SingleShot MTP training smoke.

This is intentionally independent of the online OPD/SGLang rollout harness.
It trains on clean token windows using the SingleShot static batch geometry:

1. Prepare interleaved prefix + MTP-mask regions from next-token-aligned data.
2. Run the student once and take argmax proposals at the MTP prediction slots.
3. Replace only MTP mask-token inputs with the student's first k-1 proposals.
4. Run the frozen teacher on that student-forced batch.
5. Train the student with CE to the teacher argmax labels at the MTP slots.

The goal is a small controlled reproduction gate, not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from tokenizers import AddedToken


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from xorl.models import save_model_weights  # noqa: E402
from xorl.models.auto import build_foundation_model, build_tokenizer  # noqa: E402
from xorl.mtp import (  # noqa: E402
    build_mtp_rope_indices,
    make_mtp_block_mask_partial,
    prepare_singleshot_mtp_batch,
)


IGNORE_INDEX = -100
DEFAULT_MODEL = (
    "/shared/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
)
DEFAULT_DATASET = "/shared/opd-datasets/qwen3_4b_gsm8k_aug_nl_bos_min192_8192.jsonl"


def _read_jsonl_dataset(path: Path, *, column: str, limit_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            token_ids = row.get(column)
            if not isinstance(token_ids, list) or not all(isinstance(x, int) for x in token_ids):
                raise ValueError(f"{path}:{line_idx + 1} missing integer-list column {column!r}")
            row["_dataset_line_idx"] = line_idx
            rows.append(row)
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def _filter_rows(rows: list[dict[str, Any]], *, column: str, min_tokens: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if len(row[column]) >= min_tokens]
    if not filtered:
        raise ValueError(f"No rows have at least {min_tokens} tokens")
    return filtered


def _batch_from_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
    *,
    batch_size: int,
    column: str,
    source_len: int,
    random_window: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    selected_rows = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
    input_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    meta_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        tokens = row[column]
        max_start = len(tokens) - (source_len + 1)
        if max_start < 0:
            raise ValueError(f"row has {len(tokens)} tokens but needs {source_len + 1}")
        start = rng.randint(0, max_start) if random_window and max_start > 0 else 0
        window = tokens[start : start + source_len + 1]
        input_rows.append(torch.tensor(window[:-1], dtype=torch.long))
        target_rows.append(torch.tensor(window[1:], dtype=torch.long))
        meta_rows.append(
            {
                "dataset_line_idx": row.get("_dataset_line_idx"),
                "row_idx": row.get("row_idx"),
                "dataset": row.get("dataset"),
                "window_start": start,
                "window_end": start + source_len + 1,
                "question": row.get("question"),
            }
        )
    return torch.stack(input_rows, dim=0), torch.stack(target_rows, dim=0), meta_rows


def _sample_k_and_offset(
    rng: random.Random,
    *,
    sequence_len: int,
    mask_region_count: int,
    k_min: int,
    k_max: int,
    offset_mode: str,
) -> tuple[int, int, int]:
    k_toks = rng.randint(k_min, k_max)
    prefix_length = sequence_len // mask_region_count - (k_toks - 1)
    if prefix_length <= 0:
        raise ValueError(
            "invalid SingleShot geometry: "
            f"sequence_len={sequence_len} mask_region_count={mask_region_count} k_toks={k_toks}"
        )
    if offset_mode == "zero":
        offset = 0
    elif offset_mode == "random-negative":
        offset = -rng.randint(0, prefix_length - 1)
    elif offset_mode == "random-signed":
        offset = rng.randint(-(prefix_length - 1), prefix_length - 1)
    else:
        raise ValueError(f"Unsupported offset mode {offset_mode!r}")
    return k_toks, offset, prefix_length


def _sample_offset(rng: random.Random, *, prefix_length: int, offset_mode: str) -> int:
    if offset_mode == "zero":
        return 0
    if offset_mode == "random-negative":
        return -rng.randint(0, prefix_length - 1)
    if offset_mode == "random-signed":
        return rng.randint(-(prefix_length - 1), prefix_length - 1)
    raise ValueError(f"Unsupported offset mode {offset_mode!r}")


def _prefix_length_for_k(*, sequence_len: int, mask_region_count: int, k_toks: int) -> int:
    prefix_length = sequence_len // mask_region_count - (k_toks - 1)
    if prefix_length <= 0:
        raise ValueError(
            "invalid SingleShot geometry: "
            f"sequence_len={sequence_len} mask_region_count={mask_region_count} k_toks={k_toks}"
        )
    return prefix_length


def _padded_position_ids(
    *,
    raw_seq_len: int,
    padded_seq_len: int,
    prefix_length: int,
    k_masks: int,
    offset: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    pos = build_mtp_rope_indices(raw_seq_len, prefix_length, k_masks, offset, device=device)
    if padded_seq_len > raw_seq_len:
        pos = torch.cat([pos, torch.arange(raw_seq_len, padded_seq_len, device=device)])
    return pos.unsqueeze(0).expand(batch_size, -1)


def _configure_mtp_token(args: argparse.Namespace, tokenizer: Any | None) -> dict[str, Any]:
    if not args.learned_mtp_token:
        return {
            "learned_mtp_token": False,
            "mask_token_id": int(args.mask_token_id),
            "mtp_token_text": None,
            "tokenizer_len": len(tokenizer) if tokenizer is not None else None,
            "num_added_tokens": 0,
        }
    if tokenizer is None:
        raise RuntimeError("--learned-mtp-token requires tokenizer loading; remove --skip-tokenizer")

    token = AddedToken(
        args.mtp_token_text,
        lstrip=False,
        rstrip=False,
        single_word=False,
        normalized=False,
        special=True,
    )
    try:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [token]},
            replace_extra_special_tokens=False,
        )
    except TypeError:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    token_id = tokenizer.convert_tokens_to_ids(args.mtp_token_text)
    if token_id is None or token_id < 0:
        raise RuntimeError(f"failed to add/resolve MTP token {args.mtp_token_text!r}")
    args.mask_token_id = int(token_id)
    return {
        "learned_mtp_token": True,
        "mask_token_id": int(token_id),
        "mtp_token_text": args.mtp_token_text,
        "tokenizer_len": len(tokenizer),
        "num_added_tokens": int(num_added),
    }


def _set_vocab_size_attrs(model: torch.nn.Module, vocab_size: int) -> None:
    if hasattr(model, "vocab_size"):
        model.vocab_size = vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = vocab_size
    decoder = getattr(model, "model", None)
    if decoder is not None and hasattr(decoder, "vocab_size"):
        decoder.vocab_size = vocab_size


def _init_row_from_embedding_stats(weight: torch.Tensor, token_id: int) -> None:
    if token_id < 0 or token_id >= weight.shape[0]:
        raise ValueError(f"token_id={token_id} is outside weight with vocab={weight.shape[0]}")
    stats = weight.detach().float()
    mean = stats.mean(dim=0)
    std = stats.std(dim=0).clamp_min(1e-6)
    value = mean + torch.randn_like(std) * std
    weight[token_id].copy_(value.to(dtype=weight.dtype))


@torch.no_grad()
def _resize_and_init_student_mtp_token(
    student: torch.nn.Module,
    *,
    token_id: int,
    tokenizer_len: int | None,
) -> dict[str, Any]:
    input_embeddings = student.get_input_embeddings()
    output_embeddings = student.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("learned MTP token requires input and output embeddings")

    old_vocab, hidden_size = input_embeddings.weight.shape
    new_vocab = max(old_vocab, token_id + 1, tokenizer_len or 0)
    tied = output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
    resized = new_vocab > old_vocab

    if resized:
        new_input = torch.nn.Embedding(
            new_vocab,
            hidden_size,
            padding_idx=getattr(input_embeddings, "padding_idx", None),
            device=input_embeddings.weight.device,
            dtype=input_embeddings.weight.dtype,
        )
        new_input.weight[:old_vocab].copy_(input_embeddings.weight)
        for row_id in range(old_vocab, new_vocab):
            _init_row_from_embedding_stats(new_input.weight, row_id)
        student.set_input_embeddings(new_input)

        new_output = torch.nn.Linear(
            hidden_size,
            new_vocab,
            bias=False,
            device=output_embeddings.weight.device,
            dtype=output_embeddings.weight.dtype,
        )
        new_output.weight[: output_embeddings.weight.shape[0]].copy_(output_embeddings.weight)
        for row_id in range(output_embeddings.weight.shape[0], new_vocab):
            _init_row_from_embedding_stats(new_output.weight, row_id)
        student.set_output_embeddings(new_output)
        if tied:
            student.get_output_embeddings()._parameters["weight"] = student.get_input_embeddings()._parameters["weight"]
        _set_vocab_size_attrs(student, new_vocab)

    _init_row_from_embedding_stats(student.get_input_embeddings().weight, token_id)
    if not tied:
        _init_row_from_embedding_stats(student.get_output_embeddings().weight, token_id)

    return {
        "student_vocab_before": int(old_vocab),
        "student_vocab_after": int(student.get_input_embeddings().weight.shape[0]),
        "student_lm_head_vocab_after": int(student.get_output_embeddings().weight.shape[0]),
        "student_embeddings_tied": bool(
            student.get_output_embeddings().weight.data_ptr() == student.get_input_embeddings().weight.data_ptr()
        ),
        "resized_student_embeddings": bool(resized),
        "initialized_mtp_token": True,
    }


@torch.no_grad()
def _student_mtp_token_meta(student: torch.nn.Module, *, token_id: int) -> dict[str, Any]:
    input_embeddings = student.get_input_embeddings()
    output_embeddings = student.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("MTP token metadata requires input and output embeddings")
    if token_id >= input_embeddings.weight.shape[0] or token_id >= output_embeddings.weight.shape[0]:
        raise RuntimeError(
            f"token_id={token_id} is outside student embeddings "
            f"input={input_embeddings.weight.shape[0]} output={output_embeddings.weight.shape[0]}"
        )
    return {
        "student_vocab_before": int(input_embeddings.weight.shape[0]),
        "student_vocab_after": int(input_embeddings.weight.shape[0]),
        "student_lm_head_vocab_after": int(output_embeddings.weight.shape[0]),
        "student_embeddings_tied": bool(output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()),
        "resized_student_embeddings": False,
        "initialized_mtp_token": False,
    }


def _selected_logits(model: torch.nn.Module, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    selected = hidden_states[positions]
    return model.lm_head(selected)


def _entropy_and_conf(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top1_conf = probs.max(dim=-1).values
    return entropy, top1_conf


def _mean_float(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().mean().detach().cpu().item())


def _max_repeat_run(ids: list[int]) -> int:
    best = 0
    cur = 0
    prev = None
    for token_id in ids:
        if token_id == prev:
            cur += 1
        else:
            cur = 1
            prev = token_id
        best = max(best, cur)
    return best


def _decode(tokenizer: Any | None, token_ids: list[int], *, max_tokens: int = 64) -> str:
    token_ids = token_ids[:max_tokens]
    if tokenizer is None:
        return repr(token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _prepare_train_batch(
    rows: list[dict[str, Any]],
    rng: random.Random,
    args: argparse.Namespace,
    *,
    k_toks: int,
    offset: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor, Any, list[dict[str, Any]]]:
    input_ids_cpu, target_ids_cpu, row_meta = _batch_from_rows(
        rows,
        rng,
        batch_size=args.batch_size,
        column=args.dataset_column,
        source_len=args.source_len,
        random_window=args.random_window,
    )
    input_ids = input_ids_cpu.to(device, non_blocking=True)
    target_ids = target_ids_cpu.to(device, non_blocking=True)
    prepared = prepare_singleshot_mtp_batch(
        input_ids,
        target_ids,
        k_toks=k_toks,
        mask_token_id=args.mask_token_id,
        truncation_length=args.sequence_len,
        mask_region_count=args.mask_region_count,
        offset=offset,
        pad_token_id=None,
        pad_to_multiple=args.pad_to_multiple,
        ignore_index=IGNORE_INDEX,
    )
    position_ids = _padded_position_ids(
        raw_seq_len=prepared.raw_seq_len,
        padded_seq_len=prepared.padded_seq_len,
        prefix_length=prepared.prefix_length,
        k_masks=k_toks - 1,
        offset=offset,
        batch_size=prepared.input_ids.shape[0],
        device=device,
    )
    attention_mask_builder = make_mtp_block_mask_partial(
        prepared.prefix_length,
        k_toks - 1,
        prepared.padded_seq_len,
        batch_size=prepared.input_ids.shape[0],
        offset=offset,
        raw_seq_len=prepared.raw_seq_len,
        block_size=args.flex_block_size,
    )
    attention_mask = attention_mask_builder(device=device)
    return prepared, position_ids, attention_mask, row_meta


def _proposal_argmax(logits: torch.Tensor, forbidden_token_ids: list[int]) -> torch.Tensor:
    if not forbidden_token_ids:
        return logits.detach().argmax(dim=-1)
    proposal_logits = logits.detach().float().clone()
    for token_id in forbidden_token_ids:
        if 0 <= token_id < proposal_logits.shape[-1]:
            proposal_logits[:, token_id] = -torch.inf
    return proposal_logits.argmax(dim=-1)


def _proposal_argmax_and_conf(
    logits: torch.Tensor,
    forbidden_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = _proposal_argmax(logits, forbidden_token_ids)
    raw_tokens = logits.detach().argmax(dim=-1)
    log_probs = F.log_softmax(logits.detach().float(), dim=-1)
    probs = log_probs.exp()
    token_conf = probs.gather(dim=-1, index=tokens[:, None]).squeeze(-1)
    return tokens, raw_tokens, token_conf


def _accepted_counts_from_conf(
    token_conf: torch.Tensor,
    *,
    strategy: str,
    threshold: float,
) -> torch.Tensor:
    if strategy == "static":
        return torch.full(
            (token_conf.shape[0],),
            token_conf.shape[1],
            dtype=torch.long,
            device=token_conf.device,
        )
    if strategy != "confidence":
        raise ValueError(f"Unsupported train rollout strategy {strategy!r}")

    accepted: list[int] = []
    for row in token_conf.detach().cpu().tolist():
        count = 0
        for conf in row:
            if float(conf) >= threshold:
                count += 1
            else:
                break
        accepted.append(max(1, count))
    return torch.tensor(accepted, dtype=torch.long, device=token_conf.device)


def _static_forward(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prepared,
    position_ids: torch.Tensor,
    attention_mask: Any,
    *,
    k_toks: int,
    dtype: str,
    device: torch.device,
    forbidden_proposal_token_ids: list[int],
    train_rollout_strategy: str,
    train_confidence_threshold: float,
) -> dict[str, Any]:
    autocast_enabled = device.type == "cuda" and dtype != "float32"
    with torch.autocast(device_type=device.type, dtype=getattr(torch, dtype), enabled=autocast_enabled):
        student_outputs = student(
            input_ids=prepared.input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        student_logits = _selected_logits(student, student_outputs.last_hidden_state, prepared.prediction_positions)
    student_tokens_flat, student_raw_top1_flat, student_token_conf_flat = _proposal_argmax_and_conf(
        student_logits,
        forbidden_proposal_token_ids,
    )
    if student_tokens_flat.numel() % k_toks != 0:
        raise RuntimeError(f"student proposal count {student_tokens_flat.numel()} is not divisible by k={k_toks}")
    student_tokens = student_tokens_flat.view(-1, k_toks)
    student_token_conf = student_token_conf_flat.view(-1, k_toks)
    accepted_counts = _accepted_counts_from_conf(
        student_token_conf,
        strategy=train_rollout_strategy,
        threshold=train_confidence_threshold,
    )
    train_loss_mask = (
        torch.arange(k_toks, device=device).unsqueeze(0) < accepted_counts.unsqueeze(1)
    ).reshape(-1)

    teacher_input_ids = prepared.input_ids.clone()
    if k_toks > 1:
        mask_values = teacher_input_ids[prepared.mask_positions].view(-1, k_toks - 1)
        proposal_prefill = student_tokens[:, : k_toks - 1]
        prefill_mask = torch.arange(k_toks - 1, device=device).unsqueeze(0) < (
            accepted_counts - 1
        ).clamp_min(0).unsqueeze(1)
        mask_values[prefill_mask] = proposal_prefill[prefill_mask]
        teacher_input_ids[prepared.mask_positions] = mask_values.reshape(-1)
    teacher_vocab = teacher.get_input_embeddings().weight.shape[0]
    teacher_max_token = int(teacher_input_ids.max().detach().cpu().item())
    if teacher_max_token >= teacher_vocab:
        raise RuntimeError(
            f"teacher input token {teacher_max_token} exceeds teacher vocab {teacher_vocab}; "
            "check proposal masking for added MTP tokens"
        )

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=getattr(torch, dtype), enabled=autocast_enabled
    ):
        teacher_outputs = teacher(
            input_ids=teacher_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        teacher_logits = _selected_logits(teacher, teacher_outputs.last_hidden_state, prepared.prediction_positions)
        teacher_tokens_flat = teacher_logits.argmax(dim=-1)
        teacher_tokens = teacher_tokens_flat.view(-1, k_toks)

    token_losses = F.cross_entropy(student_logits.float(), teacher_tokens_flat, reduction="none")
    loss = (token_losses * train_loss_mask.float()).sum() / train_loss_mask.float().sum().clamp_min(1.0)
    return {
        "loss": loss,
        "token_losses": token_losses,
        "train_loss_mask": train_loss_mask,
        "accepted_counts": accepted_counts,
        "train_rollout_strategy": train_rollout_strategy,
        "train_confidence_threshold": train_confidence_threshold,
        "student_logits": student_logits,
        "student_tokens_flat": student_tokens_flat,
        "student_raw_top1_flat": student_raw_top1_flat,
        "student_token_conf_flat": student_token_conf_flat,
        "student_tokens": student_tokens,
        "teacher_logits": teacher_logits,
        "teacher_tokens_flat": teacher_tokens_flat,
        "teacher_tokens": teacher_tokens,
    }


def _build_metric_row(
    *,
    args: argparse.Namespace,
    step: int,
    loss: torch.Tensor,
    prepared,
    static_outputs: dict[str, Any],
    k_toks: int,
    offset: int,
    grad_norm: torch.Tensor | float | None,
    learning_rate: float,
    step_time_s: float,
    elapsed_s: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    student_logits = static_outputs["student_logits"]
    teacher_logits = static_outputs["teacher_logits"]
    student_tokens_flat = static_outputs["student_tokens_flat"]
    student_raw_top1_flat = static_outputs["student_raw_top1_flat"]
    teacher_tokens_flat = static_outputs["teacher_tokens_flat"]
    train_loss_mask = static_outputs["train_loss_mask"]
    accepted_counts = static_outputs["accepted_counts"]
    student_token_conf_flat = static_outputs["student_token_conf_flat"]
    with torch.no_grad():
        gt_tokens_flat = prepared.target_ids[prepared.prediction_positions]
        gt_tokens = gt_tokens_flat.view(-1, k_toks)
        teacher_entropy, teacher_top1_conf = _entropy_and_conf(teacher_logits)
        student_entropy, student_top1_conf = _entropy_and_conf(student_logits)
        teacher_entropy_by_pos = teacher_entropy.view(-1, k_toks).mean(dim=0)
        student_entropy_by_pos = student_entropy.view(-1, k_toks).mean(dim=0)
        teacher_conf_by_pos = teacher_top1_conf.view(-1, k_toks).mean(dim=0)
        student_conf_by_pos = student_top1_conf.view(-1, k_toks).mean(dim=0)
        teacher_tokens_by_pos = teacher_tokens_flat.view(-1, k_toks)
        student_tokens_by_pos = student_tokens_flat.view(-1, k_toks)
        accepted_mask = train_loss_mask.bool()
        accepted_counts_cpu = accepted_counts.detach().cpu().tolist()
        region_repeat_runs = [_max_repeat_run(region) for region in static_outputs["student_tokens"].detach().cpu().tolist()]
        region_count = len(region_repeat_runs)
        full_repeat_count = sum(run >= k_toks for run in region_repeat_runs)
        supervised_tokens = int(train_loss_mask.sum().detach().cpu().item())
        attempted_tokens = int(student_logits.shape[0])
        row = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "hard_teacher_ce": float(loss.detach().cpu().item()),
            "train_rollout_strategy": static_outputs["train_rollout_strategy"],
            "train_confidence_threshold": float(static_outputs["train_confidence_threshold"]),
            "k_toks": int(k_toks),
            "offset": int(offset),
            "prefix_length": int(prepared.prefix_length),
            "mask_region_count": int(prepared.mask_region_count),
            "raw_seq_len": int(prepared.raw_seq_len),
            "padded_seq_len": int(prepared.padded_seq_len),
            "valid_tokens": supervised_tokens,
            "attempted_tokens": attempted_tokens,
            "supervised_tokens": supervised_tokens,
            "accepted_token_count": supervised_tokens,
            "accepted_token_rate": float(supervised_tokens / max(attempted_tokens, 1)),
            "accepted_k_mean": float(sum(accepted_counts_cpu) / max(len(accepted_counts_cpu), 1)),
            "accepted_k_min": int(min(accepted_counts_cpu) if accepted_counts_cpu else 0),
            "accepted_k_max": int(max(accepted_counts_cpu) if accepted_counts_cpu else 0),
            "batch_size": int(args.batch_size),
            "mask_token_id": int(args.mask_token_id),
            "learned_mtp_token": bool(args.learned_mtp_token),
            "teacher_entropy": _mean_float(teacher_entropy),
            "student_entropy": _mean_float(student_entropy),
            "teacher_top1_conf": _mean_float(teacher_top1_conf),
            "student_top1_conf": _mean_float(student_top1_conf),
            "student_teacher_top1_agreement": _mean_float(student_tokens_flat == teacher_tokens_flat),
            "student_raw_teacher_top1_agreement": _mean_float(student_raw_top1_flat == teacher_tokens_flat),
            "teacher_gt_top1_agreement": _mean_float(teacher_tokens_flat == gt_tokens_flat),
            "student_gt_top1_agreement": _mean_float(student_tokens_flat == gt_tokens_flat),
            "student_raw_gt_top1_agreement": _mean_float(student_raw_top1_flat == gt_tokens_flat),
            "student_teacher_top1_agreement_accepted": _mean_float(
                (student_tokens_flat == teacher_tokens_flat)[accepted_mask]
            ),
            "student_gt_top1_agreement_accepted": _mean_float((student_tokens_flat == gt_tokens_flat)[accepted_mask]),
            "teacher_gt_top1_agreement_accepted": _mean_float((teacher_tokens_flat == gt_tokens_flat)[accepted_mask]),
            "student_entropy_accepted": _mean_float(student_entropy[accepted_mask]),
            "teacher_entropy_accepted": _mean_float(teacher_entropy[accepted_mask]),
            "student_top1_conf_accepted": _mean_float(student_top1_conf[accepted_mask]),
            "teacher_top1_conf_accepted": _mean_float(teacher_top1_conf[accepted_mask]),
            "student_proposal_conf": _mean_float(student_token_conf_flat),
            "student_proposal_conf_accepted": _mean_float(student_token_conf_flat[accepted_mask]),
            "teacher_rollout_token_agreement": _mean_float(teacher_tokens_flat == student_tokens_flat),
            "student_rollout_token_agreement": _mean_float(student_tokens_flat == student_tokens_flat),
            "student_raw_rollout_token_agreement": _mean_float(student_raw_top1_flat == student_tokens_flat),
            "student_raw_mtp_token_rate": _mean_float(student_raw_top1_flat == args.mask_token_id),
            "student_region_count": int(region_count),
            "student_region_max_repeat_run": int(max(region_repeat_runs) if region_repeat_runs else 0),
            "student_region_mean_repeat_run": float(sum(region_repeat_runs) / max(region_count, 1)),
            "student_regions_with_full_repeat": int(full_repeat_count),
            "student_regions_with_full_repeat_rate": float(full_repeat_count / max(region_count, 1)),
            "grad_norm": (
                float(grad_norm.detach().cpu().item())
                if torch.is_tensor(grad_norm)
                else (float(grad_norm) if grad_norm is not None else None)
            ),
            "learning_rate": float(learning_rate),
            "step_time_s": step_time_s,
            "elapsed_s": elapsed_s,
        }
        for pos in range(k_toks):
            row[f"teacher_entropy_pos{pos + 1}"] = float(teacher_entropy_by_pos[pos].detach().cpu().item())
            row[f"student_entropy_pos{pos + 1}"] = float(student_entropy_by_pos[pos].detach().cpu().item())
            row[f"teacher_top1_conf_pos{pos + 1}"] = float(teacher_conf_by_pos[pos].detach().cpu().item())
            row[f"student_top1_conf_pos{pos + 1}"] = float(student_conf_by_pos[pos].detach().cpu().item())
            row[f"student_teacher_top1_pos{pos + 1}"] = _mean_float(
                student_tokens_by_pos[:, pos] == teacher_tokens_by_pos[:, pos]
            )
            row[f"student_gt_top1_pos{pos + 1}"] = _mean_float(student_tokens_by_pos[:, pos] == gt_tokens[:, pos])
            row[f"teacher_gt_top1_pos{pos + 1}"] = _mean_float(teacher_tokens_by_pos[:, pos] == gt_tokens[:, pos])
            row[f"accepted_pos{pos + 1}_rate"] = _mean_float(accepted_counts > pos)
    return row, gt_tokens


def _write_example(
    fh,
    *,
    step: int,
    tokenizer: Any | None,
    row_meta: list[dict[str, Any]],
    prepared,
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    accepted_counts: torch.Tensor,
    student_token_conf: torch.Tensor,
) -> None:
    prediction_positions = prepared.prediction_positions
    input_ids = prepared.input_ids.detach().cpu()
    student_tokens = student_tokens.detach().cpu()
    teacher_tokens = teacher_tokens.detach().cpu()
    gt_tokens = gt_tokens.detach().cpu()
    accepted_counts = accepted_counts.detach().cpu()
    student_token_conf = student_token_conf.detach().cpu().view_as(student_tokens)
    batch_size = input_ids.shape[0]
    regions_per_row = student_tokens.shape[0] // batch_size
    examples = []
    for row in range(min(batch_size, 2)):
        start = row * regions_per_row
        end = start + min(regions_per_row, 2)
        row_pred_ids = prepared.target_ids[row][prediction_positions[row]].detach().cpu().view(-1, prepared.k_toks)
        examples.append(
            {
                "row": row,
                "meta": row_meta[row],
                "input_head_ids": input_ids[row, : min(input_ids.shape[1], 64)].tolist(),
                "input_head_text": _decode(tokenizer, input_ids[row, : min(input_ids.shape[1], 64)].tolist()),
                "source_indices_head": prepared.source_indices[: min(prepared.source_indices.numel(), 96)]
                .detach()
                .cpu()
                .tolist(),
                "prediction_source_regions_head": row_pred_ids[:4].tolist(),
                "student_regions": student_tokens[start:end].tolist(),
                "teacher_regions": teacher_tokens[start:end].tolist(),
                "ground_truth_regions": gt_tokens[start:end].tolist(),
                "accepted_counts": accepted_counts[start:end].tolist(),
                "student_token_conf": student_token_conf[start:end].tolist(),
                "student_text": [_decode(tokenizer, ids) for ids in student_tokens[start:end].tolist()],
                "teacher_text": [_decode(tokenizer, ids) for ids in teacher_tokens[start:end].tolist()],
                "ground_truth_text": [_decode(tokenizer, ids) for ids in gt_tokens[start:end].tolist()],
            }
        )
    fh.write(
        json.dumps(
            {
                "step": step,
                "k_toks": prepared.k_toks,
                "prefix_length": prepared.prefix_length,
                "mask_region_count": prepared.mask_region_count,
                "offset": prepared.offset,
                "raw_seq_len": prepared.raw_seq_len,
                "padded_seq_len": prepared.padded_seq_len,
                "examples": examples,
            },
            ensure_ascii=True,
        )
        + "\n"
    )
    fh.flush()


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row[key] for row in rows if row.get(key) is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _aggregate_micro_rows(rows: list[dict[str, Any]], *, grad_norm: torch.Tensor | float | None) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty micro-row list")
    row = dict(rows[-1])
    numeric_keys = {
        key
        for item in rows
        for key, value in item.items()
        if isinstance(value, (int, float)) and key not in {"step", "k_toks", "offset", "prefix_length", "raw_seq_len", "padded_seq_len"}
    }
    for key in numeric_keys:
        values = [float(item[key]) for item in rows if isinstance(item.get(key), (int, float))]
        if values:
            row[key] = float(sum(values) / len(values))
    row["loss"] = _mean_metric(rows, "loss")
    row["hard_teacher_ce"] = _mean_metric(rows, "hard_teacher_ce")
    row["valid_tokens"] = int(sum(int(item.get("valid_tokens", 0)) for item in rows))
    row["attempted_tokens"] = int(sum(int(item.get("attempted_tokens", 0)) for item in rows))
    row["supervised_tokens"] = int(sum(int(item.get("supervised_tokens", 0)) for item in rows))
    row["accepted_token_count"] = int(sum(int(item.get("accepted_token_count", 0)) for item in rows))
    row["accepted_token_rate"] = float(row["accepted_token_count"] / max(row["attempted_tokens"], 1))
    row["batch_size"] = int(rows[-1].get("batch_size", 0))
    row["micro_batch_size"] = int(rows[-1].get("batch_size", 0))
    row["grad_accum_steps"] = len(rows)
    row["effective_batch_size"] = int(rows[-1].get("batch_size", 0)) * len(rows)
    row["micro_losses"] = [float(item["loss"]) for item in rows]
    row["micro_offsets"] = [int(item["offset"]) for item in rows]
    row["grad_norm"] = (
        float(grad_norm.detach().cpu().item())
        if torch.is_tensor(grad_norm)
        else (float(grad_norm) if grad_norm is not None else None)
    )
    return row


def _lr_for_step(args: argparse.Namespace, step: int) -> float:
    if args.warmup_steps <= 0:
        return float(args.learning_rate)
    return float(args.learning_rate) * min(1.0, float(step + 1) / float(args.warmup_steps))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _run_eval(
    *,
    args: argparse.Namespace,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    rows: list[dict[str, Any]],
    step: int,
    device: torch.device,
    eval_fh,
    wandb_run: Any | None,
    start_time: float,
    forbidden_proposal_token_ids: list[int],
) -> list[dict[str, Any]]:
    if args.eval_interval <= 0:
        return []
    was_training = student.training
    student.eval()
    summaries: list[dict[str, Any]] = []
    for k_toks in args.eval_k_values:
        prefix_length = _prefix_length_for_k(
            sequence_len=args.sequence_len,
            mask_region_count=args.mask_region_count,
            k_toks=k_toks,
        )
        eval_rows: list[dict[str, Any]] = []
        eval_rng = random.Random(args.eval_seed + k_toks)
        for batch_idx in range(args.eval_batches):
            offset = _sample_offset(eval_rng, prefix_length=prefix_length, offset_mode=args.eval_offset_mode)
            prepared, position_ids, attention_mask, _row_meta = _prepare_train_batch(
                rows,
                eval_rng,
                args,
                k_toks=k_toks,
                offset=offset,
                device=device,
            )
            with torch.no_grad():
                static_outputs = _static_forward(
                    student,
                    teacher,
                    prepared,
                    position_ids,
                    attention_mask,
                    k_toks=k_toks,
                    dtype=args.dtype,
                    device=device,
                    forbidden_proposal_token_ids=forbidden_proposal_token_ids,
                    train_rollout_strategy=args.train_rollout_strategy,
                    train_confidence_threshold=args.train_confidence_threshold,
                )
                row, _gt_tokens = _build_metric_row(
                    args=args,
                    step=step,
                    loss=static_outputs["loss"],
                    prepared=prepared,
                    static_outputs=static_outputs,
                    k_toks=k_toks,
                    offset=offset,
                    grad_norm=None,
                    learning_rate=0.0,
                    step_time_s=0.0,
                    elapsed_s=time.time() - start_time,
                )
                row["eval_batch_idx"] = batch_idx
                eval_rows.append(row)
        summary = {
            "step": step,
            "eval_k_toks": int(k_toks),
            "eval_batches": int(args.eval_batches),
            "loss": _mean_metric(eval_rows, "loss"),
            "train_confidence_threshold": float(args.train_confidence_threshold),
            "attempted_tokens": _mean_metric(eval_rows, "attempted_tokens"),
            "supervised_tokens": _mean_metric(eval_rows, "supervised_tokens"),
            "accepted_token_rate": _mean_metric(eval_rows, "accepted_token_rate"),
            "accepted_k_mean": _mean_metric(eval_rows, "accepted_k_mean"),
            "student_teacher_top1_agreement": _mean_metric(eval_rows, "student_teacher_top1_agreement"),
            "student_raw_teacher_top1_agreement": _mean_metric(eval_rows, "student_raw_teacher_top1_agreement"),
            "teacher_gt_top1_agreement": _mean_metric(eval_rows, "teacher_gt_top1_agreement"),
            "student_gt_top1_agreement": _mean_metric(eval_rows, "student_gt_top1_agreement"),
            "student_teacher_top1_agreement_accepted": _mean_metric(
                eval_rows, "student_teacher_top1_agreement_accepted"
            ),
            "student_gt_top1_agreement_accepted": _mean_metric(eval_rows, "student_gt_top1_agreement_accepted"),
            "teacher_gt_top1_agreement_accepted": _mean_metric(eval_rows, "teacher_gt_top1_agreement_accepted"),
            "teacher_entropy": _mean_metric(eval_rows, "teacher_entropy"),
            "student_entropy": _mean_metric(eval_rows, "student_entropy"),
            "teacher_entropy_accepted": _mean_metric(eval_rows, "teacher_entropy_accepted"),
            "student_entropy_accepted": _mean_metric(eval_rows, "student_entropy_accepted"),
            "teacher_top1_conf_accepted": _mean_metric(eval_rows, "teacher_top1_conf_accepted"),
            "student_top1_conf_accepted": _mean_metric(eval_rows, "student_top1_conf_accepted"),
            "student_proposal_conf": _mean_metric(eval_rows, "student_proposal_conf"),
            "student_proposal_conf_accepted": _mean_metric(eval_rows, "student_proposal_conf_accepted"),
            "student_raw_rollout_token_agreement": _mean_metric(eval_rows, "student_raw_rollout_token_agreement"),
            "student_raw_mtp_token_rate": _mean_metric(eval_rows, "student_raw_mtp_token_rate"),
            "student_region_count": _mean_metric(eval_rows, "student_region_count"),
            "student_region_max_repeat_run": _mean_metric(eval_rows, "student_region_max_repeat_run"),
            "student_region_mean_repeat_run": _mean_metric(eval_rows, "student_region_mean_repeat_run"),
            "student_regions_with_full_repeat": _mean_metric(eval_rows, "student_regions_with_full_repeat"),
            "student_regions_with_full_repeat_rate": _mean_metric(eval_rows, "student_regions_with_full_repeat_rate"),
        }
        position_metric_prefixes = (
            "teacher_entropy_pos",
            "student_entropy_pos",
            "teacher_top1_conf_pos",
            "student_top1_conf_pos",
            "student_teacher_top1_pos",
            "student_gt_top1_pos",
            "teacher_gt_top1_pos",
            "accepted_pos",
        )
        position_metric_keys = sorted(
            {
                key
                for row in eval_rows
                for key in row
                if key.startswith(position_metric_prefixes)
            }
        )
        for key in position_metric_keys:
            summary[key] = _mean_metric(eval_rows, key)
        eval_fh.write(json.dumps(summary, ensure_ascii=True) + "\n")
        eval_fh.flush()
        if wandb_run is not None:
            wandb_run.log(
                {f"eval/k{k_toks}/{key}": value for key, value in summary.items() if key != "step"},
                step=max(step, 0),
            )
        print(json.dumps({"eval": summary}, ensure_ascii=True), flush=True)
        summaries.append(summary)
    if was_training:
        student.train()
    return summaries


def _save_student_checkpoint(
    *,
    checkpoint_dir: Path,
    student: torch.nn.Module,
    tokenizer: Any | None,
    checkpoint_dtype: str,
) -> str:
    model_assets = [student.config]
    if tokenizer is not None:
        model_assets.append(tokenizer)
    save_model_weights(
        checkpoint_dir,
        student.state_dict(),
        save_dtype=checkpoint_dtype,
        safe_serialization=True,
        model_assets=model_assets,
    )
    return str(checkpoint_dir)


def _metric_is_better(value: float, best_value: float | None, *, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"Unsupported best checkpoint mode {mode!r}")


def _maybe_save_best_checkpoint(
    *,
    args: argparse.Namespace,
    student: torch.nn.Module,
    tokenizer: Any | None,
    output_dir: Path,
    eval_summaries: list[dict[str, Any]],
    best_state: dict[str, Any],
    wandb_run: Any | None,
) -> None:
    if not args.save_best_checkpoint or not eval_summaries:
        return
    matching = [row for row in eval_summaries if int(row["eval_k_toks"]) == args.best_checkpoint_k]
    if not matching:
        return
    summary = matching[-1]
    step = int(summary["step"])
    if step < args.best_checkpoint_min_step:
        return
    metric = args.best_checkpoint_metric
    if metric not in summary:
        raise RuntimeError(f"Best-checkpoint metric {metric!r} is not present in eval summary")
    value = float(summary[metric])
    if not _metric_is_better(value, best_state.get("value"), mode=args.best_checkpoint_mode):
        return

    checkpoint_dir = Path(args.best_checkpoint_dir) if args.best_checkpoint_dir else output_dir / "student_checkpoint_best"
    checkpoint_path = _save_student_checkpoint(
        checkpoint_dir=checkpoint_dir,
        student=student,
        tokenizer=tokenizer,
        checkpoint_dtype=args.checkpoint_dtype,
    )
    best_state.clear()
    best_state.update(
        {
            "checkpoint_path": checkpoint_path,
            "step": step,
            "eval_k_toks": int(summary["eval_k_toks"]),
            "metric": metric,
            "mode": args.best_checkpoint_mode,
            "value": value,
            "summary": summary,
        }
    )
    (output_dir / "best_checkpoint_summary.json").write_text(json.dumps(best_state, indent=2, ensure_ascii=True) + "\n")
    if wandb_run is not None:
        wandb_run.log(
            {
                "best_checkpoint/step": step,
                "best_checkpoint/eval_k_toks": int(summary["eval_k_toks"]),
                "best_checkpoint/value": value,
            },
            step=max(step, 0),
        )
    print(json.dumps({"best_checkpoint": best_state}, ensure_ascii=True), flush=True)


def _maybe_save_eval_step_checkpoint(
    *,
    args: argparse.Namespace,
    student: torch.nn.Module,
    tokenizer: Any | None,
    output_dir: Path,
    eval_summaries: list[dict[str, Any]],
    saved_steps: set[int],
    wandb_run: Any | None,
) -> None:
    if not args.save_eval_checkpoint_steps or not eval_summaries:
        return
    step = int(eval_summaries[-1]["step"])
    if step in saved_steps or step not in set(args.save_eval_checkpoint_steps):
        return

    checkpoint_root = Path(args.eval_checkpoint_dir) if args.eval_checkpoint_dir else output_dir
    checkpoint_dir = checkpoint_root / f"{args.eval_checkpoint_prefix}{step}"
    checkpoint_path = _save_student_checkpoint(
        checkpoint_dir=checkpoint_dir,
        student=student,
        tokenizer=tokenizer,
        checkpoint_dtype=args.checkpoint_dtype,
    )
    saved_steps.add(step)
    checkpoint_state = {
        "checkpoint_path": checkpoint_path,
        "step": step,
        "eval_summaries": eval_summaries,
    }
    (checkpoint_dir / "eval_checkpoint_summary.json").write_text(
        json.dumps(checkpoint_state, indent=2, ensure_ascii=True) + "\n"
    )
    with (output_dir / "eval_checkpoint_summaries.jsonl").open("a") as fh:
        fh.write(json.dumps(checkpoint_state, ensure_ascii=True) + "\n")
    if wandb_run is not None:
        wandb_run.log(
            {
                "eval_checkpoint/step": step,
                "eval_checkpoint/saved": 1,
            },
            step=max(step, 0),
        )
    print(json.dumps({"eval_checkpoint": checkpoint_state}, ensure_ascii=True), flush=True)


def _init_wandb(args: argparse.Namespace, output_dir: Path):
    if args.disable_wandb or not args.wandb_project:
        return None
    import wandb  # noqa: PLC0415

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_name or args.run_id,
        "tags": args.wandb_tags,
        "config": config,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    return wandb.init(**init_kwargs)


def _load_model_pair(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    student_model_path = args.student_model_path or args.model_path
    student_weights_path = args.student_weights_path or student_model_path
    teacher_model_path = args.teacher_model_path or args.model_path
    teacher_weights_path = args.teacher_weights_path or teacher_model_path

    print(f"loading student config from {student_model_path}", flush=True)
    print(f"loading student weights from {student_weights_path}", flush=True)
    student = build_foundation_model(
        student_model_path,
        weights_path=student_weights_path,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        init_device=str(device),
    )
    print(f"loading teacher config from {teacher_model_path}", flush=True)
    print(f"loading teacher weights from {teacher_weights_path}", flush=True)
    teacher = build_foundation_model(
        teacher_model_path,
        weights_path=teacher_weights_path,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        init_device=str(device),
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable({"use_reentrant": False})
    return student, teacher


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    examples_path = output_dir / "examples.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset_jsonl)
    rows = _read_jsonl_dataset(dataset_path, column=args.dataset_column, limit_rows=args.limit_rows)
    rows = _filter_rows(rows, column=args.dataset_column, min_tokens=args.source_len + 1)
    print(f"loaded_rows={len(rows)} dataset={dataset_path}", flush=True)

    tokenizer = None
    if not args.skip_tokenizer:
        try:
            tokenizer_path = args.tokenizer_path or args.student_model_path or args.model_path
            tokenizer = build_tokenizer(tokenizer_path)
        except Exception as exc:  # pragma: no cover - tokenizer availability is environment-dependent
            print(f"warning: failed to load tokenizer: {exc}", flush=True)
    mtp_token_meta = _configure_mtp_token(args, tokenizer)
    print(json.dumps({"mtp_token": mtp_token_meta}, ensure_ascii=True), flush=True)

    if args.dry_run:
        dry_rows = []
        for case_idx in range(args.dry_run_cases):
            k_toks, offset, _prefix_length = _sample_k_and_offset(
                rng,
                sequence_len=args.sequence_len,
                mask_region_count=args.mask_region_count,
                k_min=args.k_min,
                k_max=args.k_max,
                offset_mode=args.offset_mode,
            )
            input_ids, target_ids, row_meta = _batch_from_rows(
                rows,
                rng,
                batch_size=args.batch_size,
                column=args.dataset_column,
                source_len=args.source_len,
                random_window=args.random_window,
            )
            prepared = prepare_singleshot_mtp_batch(
                input_ids,
                target_ids,
                k_toks=k_toks,
                mask_token_id=args.mask_token_id,
                truncation_length=args.sequence_len,
                mask_region_count=args.mask_region_count,
                offset=offset,
                pad_token_id=None,
                pad_to_multiple=args.pad_to_multiple,
                ignore_index=IGNORE_INDEX,
            )
            dry_rows.append(
                {
                    "case_idx": case_idx,
                    "k_toks": k_toks,
                    "offset": offset,
                    "prefix_length": prepared.prefix_length,
                    "raw_seq_len": prepared.raw_seq_len,
                    "padded_seq_len": prepared.padded_seq_len,
                    "prediction_tokens": int(prepared.prediction_positions.sum().item()),
                    "mask_tokens": int(prepared.mask_positions.sum().item()),
                    "row_meta": row_meta,
                    "source_indices_head": prepared.source_indices[:96].tolist(),
                }
            )
        summary = {
            "ok": True,
            "mode": "dry_run",
            "dataset_jsonl": str(dataset_path),
            "usable_rows": len(rows),
            "student_model_path": args.student_model_path or args.model_path,
            "student_weights_path": args.student_weights_path or args.student_model_path or args.model_path,
            "teacher_model_path": args.teacher_model_path or args.model_path,
            "teacher_weights_path": args.teacher_weights_path or args.teacher_model_path or args.model_path,
            "tokenizer_path": args.tokenizer_path or args.student_model_path or args.model_path,
            "mtp_token": mtp_token_meta,
            "cases": dry_rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
        return

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    student, teacher = _load_model_pair(args, device)
    mtp_model_meta = {}
    if args.learned_mtp_token:
        if args.initialize_learned_mtp_token:
            mtp_model_meta = _resize_and_init_student_mtp_token(
                student,
                token_id=args.mask_token_id,
                tokenizer_len=mtp_token_meta.get("tokenizer_len"),
            )
        else:
            mtp_model_meta = _student_mtp_token_meta(student, token_id=args.mask_token_id)
        print(json.dumps({"mtp_model": mtp_model_meta}, ensure_ascii=True), flush=True)
    student.train()
    optimizer = torch.optim.AdamW(
        (param for param in student.parameters() if param.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    wandb_run = _init_wandb(args, output_dir)

    metrics_fh = metrics_path.open("a")
    examples_fh = examples_path.open("a")
    eval_metrics_fh = eval_metrics_path.open("a")
    start_time = time.time()
    last_loss_values: list[float] = []
    forbidden_proposal_token_ids = [args.mask_token_id] if args.exclude_mtp_token_from_proposals else []
    best_checkpoint_state: dict[str, Any] = {}
    saved_eval_checkpoint_steps: set[int] = set()

    if args.eval_before_training:
        eval_summaries = _run_eval(
            args=args,
            student=student,
            teacher=teacher,
            rows=rows,
            step=-1,
            device=device,
            eval_fh=eval_metrics_fh,
            wandb_run=wandb_run,
            start_time=start_time,
            forbidden_proposal_token_ids=forbidden_proposal_token_ids,
        )
        _maybe_save_best_checkpoint(
            args=args,
            student=student,
            tokenizer=tokenizer,
            output_dir=output_dir,
            eval_summaries=eval_summaries,
            best_state=best_checkpoint_state,
            wandb_run=wandb_run,
        )
        _maybe_save_eval_step_checkpoint(
            args=args,
            student=student,
            tokenizer=tokenizer,
            output_dir=output_dir,
            eval_summaries=eval_summaries,
            saved_steps=saved_eval_checkpoint_steps,
            wandb_run=wandb_run,
        )

    for step in range(args.steps):
        step_start = time.time()
        step_learning_rate = _lr_for_step(args, step)
        _set_optimizer_lr(optimizer, step_learning_rate)
        optimizer.zero_grad(set_to_none=True)
        k_toks, offset, _prefix_length = _sample_k_and_offset(
            rng,
            sequence_len=args.sequence_len,
            mask_region_count=args.mask_region_count,
            k_min=args.k_min,
            k_max=args.k_max,
            offset_mode=args.offset_mode,
        )
        micro_rows: list[dict[str, Any]] = []
        last_prepared = None
        last_static_outputs = None
        last_gt_tokens = None
        last_row_meta = None
        for micro_step in range(args.grad_accum_steps):
            prepared, position_ids, attention_mask, row_meta = _prepare_train_batch(
                rows,
                rng,
                args,
                k_toks=k_toks,
                offset=offset,
                device=device,
            )
            static_outputs = _static_forward(
                student,
                teacher,
                prepared,
                position_ids,
                attention_mask,
                k_toks=k_toks,
                dtype=args.dtype,
                device=device,
                forbidden_proposal_token_ids=forbidden_proposal_token_ids,
                train_rollout_strategy=args.train_rollout_strategy,
                train_confidence_threshold=args.train_confidence_threshold,
            )
            loss = static_outputs["loss"]
            (loss / args.grad_accum_steps).backward()
            micro_row, gt_tokens = _build_metric_row(
                args=args,
                step=step,
                loss=loss,
                prepared=prepared,
                static_outputs=static_outputs,
                k_toks=k_toks,
                offset=offset,
                grad_norm=None,
                learning_rate=step_learning_rate,
                step_time_s=0.0,
                elapsed_s=time.time() - start_time,
            )
            micro_row["micro_step"] = micro_step
            micro_rows.append(micro_row)
            last_prepared = prepared
            last_static_outputs = static_outputs
            last_gt_tokens = gt_tokens
            last_row_meta = row_meta

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()

        row = _aggregate_micro_rows(micro_rows, grad_norm=grad_norm)
        row["step_time_s"] = time.time() - step_start
        row["elapsed_s"] = time.time() - start_time
        row["learning_rate"] = step_learning_rate
        last_loss_values.append(row["loss"])
        if len(last_loss_values) > max(args.loss_trend_window, 1):
            last_loss_values.pop(0)
        if len(last_loss_values) >= 2:
            row["loss_trend_window_delta"] = last_loss_values[-1] - last_loss_values[0]

        metrics_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        metrics_fh.flush()
        if wandb_run is not None and (step % args.wandb_log_interval == 0):
            wandb_run.log(row, step=step)
        if step % args.example_interval == 0 or step == args.steps - 1:
            if last_prepared is None or last_static_outputs is None or last_gt_tokens is None or last_row_meta is None:
                raise RuntimeError("missing final micro-batch artifacts for example logging")
            _write_example(
                examples_fh,
                step=step,
                tokenizer=tokenizer,
                row_meta=last_row_meta,
                prepared=last_prepared,
                student_tokens=last_static_outputs["student_tokens"],
                teacher_tokens=last_static_outputs["teacher_tokens"],
                gt_tokens=last_gt_tokens,
                accepted_counts=last_static_outputs["accepted_counts"],
                student_token_conf=last_static_outputs["student_token_conf_flat"],
            )
        print(json.dumps(row, ensure_ascii=True), flush=True)
        if args.eval_interval > 0 and ((step + 1) % args.eval_interval == 0 or step == args.steps - 1):
            eval_summaries = _run_eval(
                args=args,
                student=student,
                teacher=teacher,
                rows=rows,
                step=step,
                device=device,
                eval_fh=eval_metrics_fh,
                wandb_run=wandb_run,
                start_time=start_time,
                forbidden_proposal_token_ids=forbidden_proposal_token_ids,
            )
            _maybe_save_best_checkpoint(
                args=args,
                student=student,
                tokenizer=tokenizer,
                output_dir=output_dir,
                eval_summaries=eval_summaries,
                best_state=best_checkpoint_state,
                wandb_run=wandb_run,
            )
            _maybe_save_eval_step_checkpoint(
                args=args,
                student=student,
                tokenizer=tokenizer,
                output_dir=output_dir,
                eval_summaries=eval_summaries,
                saved_steps=saved_eval_checkpoint_steps,
                wandb_run=wandb_run,
            )

    metrics_fh.close()
    examples_fh.close()
    eval_metrics_fh.close()
    checkpoint_path = None
    if args.save_final_checkpoint:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "student_checkpoint"
        checkpoint_path = _save_student_checkpoint(
            checkpoint_dir=checkpoint_dir,
            student=student,
            tokenizer=tokenizer,
            checkpoint_dtype=args.checkpoint_dtype,
        )
    summary = {
        "ok": True,
        "mode": "train",
        "run_id": args.run_id,
        "dataset_jsonl": str(dataset_path),
        "usable_rows": len(rows),
        "steps": args.steps,
        "train_rollout_strategy": args.train_rollout_strategy,
        "train_confidence_threshold": args.train_confidence_threshold,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "student_model_path": args.student_model_path or args.model_path,
        "student_weights_path": args.student_weights_path or args.student_model_path or args.model_path,
        "teacher_model_path": args.teacher_model_path or args.model_path,
        "teacher_weights_path": args.teacher_weights_path or args.teacher_model_path or args.model_path,
        "tokenizer_path": args.tokenizer_path or args.student_model_path or args.model_path,
        "mtp_token": mtp_token_meta,
        "mtp_model": mtp_model_meta,
        "metrics_jsonl": str(metrics_path),
        "examples_jsonl": str(examples_path),
        "eval_metrics_jsonl": str(eval_metrics_path),
        "checkpoint_path": checkpoint_path,
        "best_checkpoint": best_checkpoint_state or None,
        "saved_eval_checkpoint_steps": sorted(saved_eval_checkpoint_steps),
        "final_loss": last_loss_values[-1] if last_loss_values else None,
        "loss_window_delta": (last_loss_values[-1] - last_loss_values[0]) if len(last_loss_values) >= 2 else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    if wandb_run is not None:
        wandb_run.finish()
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"static-mtp-smoke-{int(time.time())}")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--student-model-path", default=None)
    parser.add_argument("--student-weights-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--teacher-weights-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--dataset-jsonl", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-column", default="input_ids")
    parser.add_argument("--output-dir", default="/shared/opd-coord/static-mtp-smoke")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--source-len", type=int, default=192)
    parser.add_argument("--sequence-len", type=int, default=160)
    parser.add_argument("--mask-region-count", type=int, default=5)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--offset-mode", choices=["zero", "random-negative", "random-signed"], default="random-negative")
    parser.add_argument("--mask-token-id", type=int, default=151662)
    parser.add_argument("--learned-mtp-token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initialize-learned-mtp-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mtp-token-text", default="<MTP>")
    parser.add_argument("--exclude-mtp-token-from-proposals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-rollout-strategy", choices=["confidence", "static"], default="confidence")
    parser.add_argument("--train-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--pad-to-multiple", type=int, default=128)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--random-window", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-trend-window", type=int, default=10)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=["flex_attention", "native", "eager", "flash_attention_3", "flash_attention_4"],
        default="flex_attention",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-cases", type=int, default=4)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--example-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-before-training", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-k-values", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument(
        "--eval-offset-mode",
        choices=["zero", "random-negative", "random-signed"],
        default="zero",
    )
    parser.add_argument("--eval-seed", type=int, default=2026060401)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--best-checkpoint-dir", default=None)
    parser.add_argument("--best-checkpoint-k", type=int, default=8)
    parser.add_argument("--best-checkpoint-metric", default="loss")
    parser.add_argument("--best-checkpoint-mode", choices=["min", "max"], default="min")
    parser.add_argument("--best-checkpoint-min-step", type=int, default=0)
    parser.add_argument("--save-eval-checkpoint-steps", nargs="*", type=int, default=[])
    parser.add_argument("--eval-checkpoint-dir", default=None)
    parser.add_argument("--eval-checkpoint-prefix", default="student_checkpoint_step")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--wandb-project", default=os.environ.get("STATIC_MTP_WANDB_PROJECT", "singleshot"))
    parser.add_argument("--wandb-name", default=os.environ.get("STATIC_MTP_WANDB_NAME"))
    parser.add_argument("--wandb-entity", default=os.environ.get("STATIC_MTP_WANDB_ENTITY", "together-research"))
    parser.add_argument("--wandb-tags", nargs="*", default=["static-mtp", "singleshot", "qwen3-4b", "gsm-proxy"])
    parser.add_argument("--wandb-log-interval", type=int, default=1)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR"))
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    if args.sequence_len % args.mask_region_count != 0:
        parser.error("--sequence-len must be divisible by --mask-region-count")
    if args.k_min < 1 or args.k_max < args.k_min:
        parser.error("invalid k range")
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be positive")
    if not 0.0 <= args.train_confidence_threshold <= 1.0:
        parser.error("--train-confidence-threshold must be in [0, 1]")
    max_k = args.sequence_len // args.mask_region_count
    if args.k_max > max_k:
        parser.error(f"--k-max={args.k_max} leaves no prefix tokens; max for this geometry is {max_k}")
    if args.source_len < args.sequence_len:
        parser.error("--source-len must be >= --sequence-len")
    if args.example_interval <= 0:
        parser.error("--example-interval must be positive")
    if args.eval_interval < 0:
        parser.error("--eval-interval must be non-negative")
    if args.eval_before_training and args.eval_interval <= 0:
        parser.error("--eval-before-training requires --eval-interval > 0")
    if args.eval_batches <= 0:
        parser.error("--eval-batches must be positive")
    if not args.eval_k_values:
        parser.error("--eval-k-values must contain at least one k")
    invalid_eval_ks = [k for k in args.eval_k_values if k < 1 or k > max_k]
    if invalid_eval_ks:
        parser.error(f"--eval-k-values outside valid range [1, {max_k}]: {invalid_eval_ks}")
    if args.best_checkpoint_k < 1 or args.best_checkpoint_k > max_k:
        parser.error(f"--best-checkpoint-k must be in [1, {max_k}]")
    if args.save_best_checkpoint and args.eval_interval <= 0:
        parser.error("--save-best-checkpoint requires --eval-interval > 0")
    if args.save_best_checkpoint and args.best_checkpoint_k not in args.eval_k_values:
        parser.error("--best-checkpoint-k must be present in --eval-k-values")
    if args.best_checkpoint_min_step < -1:
        parser.error("--best-checkpoint-min-step must be >= -1")
    if args.save_eval_checkpoint_steps and args.eval_interval <= 0:
        parser.error("--save-eval-checkpoint-steps requires --eval-interval > 0")
    invalid_eval_checkpoint_steps = [step for step in args.save_eval_checkpoint_steps if step < -1]
    if invalid_eval_checkpoint_steps:
        parser.error(f"--save-eval-checkpoint-steps must be >= -1: {invalid_eval_checkpoint_steps}")
    if args.eval_checkpoint_prefix == "":
        parser.error("--eval-checkpoint-prefix must be non-empty")
    if args.learned_mtp_token and args.skip_tokenizer:
        parser.error("--learned-mtp-token requires tokenizer loading; remove --skip-tokenizer")
    if args.wandb_log_interval <= 0:
        parser.error("--wandb-log-interval must be positive")
    if args.pad_to_multiple <= 0:
        args.pad_to_multiple = None
    return args


if __name__ == "__main__":
    run(parse_args())
