#!/usr/bin/env python3
"""Preflight paper-style static SingleShot MTP batches.

This verifier is intentionally CPU-only. It checks xorl's static MTP batch
preparation against the SingleShot LitGPT `truncate_and_mask` tensor semantics
for clean, offline training sequences. It does not use sampled OPD rollouts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xorl.mtp.singleshot import (  # noqa: E402
    dense_interleaved_mtp_attention_mask,
    prepare_singleshot_mtp_batch,
)


IGNORE_INDEX = -100


def _reference_truncate_and_mask(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    k_toks: int,
    mask_id: int | None,
    truncation_length: int,
    mask_region_ct: int,
    offset: int = 0,
    min_mask_id: int | None = None,
    max_mask_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Local reference for `~/singleshot/litgpt/pretrain.py::truncate_and_mask`.

    This covers the clean paper-style path: no tail padding/EOS masking and no
    prelude masking. The arithmetic and offset handling are kept equivalent to
    the SingleShot function so this script can run without importing LitGPT's
    full training dependency stack.
    """

    batch_size, original_seq_len = input_ids.shape
    seq_len = truncation_length
    k_masks = k_toks - 1
    prefix_length = seq_len // mask_region_ct - k_masks
    region_width = prefix_length + k_masks
    if prefix_length <= 0:
        raise ValueError(
            "k_toks leaves no prefix tokens: "
            f"truncation_length={seq_len} mask_region_ct={mask_region_ct} k_toks={k_toks}"
        )
    if abs(offset) >= prefix_length:
        raise ValueError(f"offset magnitude must be less than prefix length {prefix_length}, got {offset}")

    source_block_starts = torch.arange(0, seq_len, prefix_length, device=input_ids.device)
    base_range = torch.arange(region_width, device=input_ids.device)
    block_starts = torch.arange(
        0,
        source_block_starts.numel() * prefix_length,
        prefix_length,
        device=input_ids.device,
    ).unsqueeze(1)
    full_indices = block_starts + base_range

    mask_id_mask = torch.zeros_like(full_indices, dtype=torch.bool)
    pred_pos_mask = torch.zeros_like(full_indices, dtype=torch.bool)
    if k_masks > 0:
        mask_id_mask[:, prefix_length : prefix_length + k_masks] = True
    pred_pos_mask[:, prefix_length - 1 : prefix_length + k_masks] = True

    final_indices = full_indices.flatten()[:seq_len]
    mask_id_mask = mask_id_mask.flatten()[:seq_len]
    pred_pos_mask = pred_pos_mask.flatten()[:seq_len]

    if int(final_indices.max().item()) > original_seq_len - 1:
        raise ValueError("final indices exceed original sequence length")
    consumed = mask_region_ct * prefix_length + k_masks
    if int(final_indices.max().item()) != consumed - 1:
        raise ValueError(
            "final index max does not match SingleShot consumed-token formula: "
            f"max={int(final_indices.max().item())} consumed={consumed}"
        )

    if offset < 0:
        old_final_mask_start = final_indices[-1] + 1 if k_masks == 0 else final_indices[-k_masks]
        final_indices = final_indices.roll(offset)
        final_indices[offset:] = torch.arange(
            old_final_mask_start,
            old_final_mask_start + (-offset),
            device=input_ids.device,
        )
        final_indices = final_indices + offset
        mask_id_mask = mask_id_mask.roll(offset)
        mask_id_mask[offset:] = False
        pred_pos_mask = pred_pos_mask.roll(offset)
        pred_pos_mask[offset:] = False
    elif offset > 0:
        final_indices = final_indices.roll(offset)
        final_indices[:offset] = torch.arange(-offset, 0, device=input_ids.device)
        final_indices = final_indices + offset
        mask_id_mask = mask_id_mask.roll(offset)
        mask_id_mask[:offset] = False
        pred_pos_mask = pred_pos_mask.roll(offset)
        pred_pos_mask[:offset] = False

    prepared_input_ids = torch.index_select(input_ids, dim=1, index=final_indices).clone()
    prepared_target_ids = torch.index_select(target_ids, dim=1, index=final_indices).clone()
    mask_id_mask = mask_id_mask.unsqueeze(0).repeat(batch_size, 1)
    pred_pos_mask = pred_pos_mask.unsqueeze(0).repeat(batch_size, 1)

    if mask_id_mask.any():
        if min_mask_id is None and mask_id is None:
            raise ValueError("mask_id or min_mask_id is required when k_toks > 1")
        mask_start = min_mask_id if min_mask_id is not None else int(mask_id)
        per_position_tokens = (torch.arange(seq_len, device=input_ids.device) - offset) % region_width
        per_position_tokens = per_position_tokens - prefix_length + mask_start
        if min_mask_id is None:
            per_position_tokens = torch.full_like(per_position_tokens, int(mask_id))
        if min_mask_id is not None and max_mask_id is not None:
            inserted_max = int(per_position_tokens.unsqueeze(0).expand(batch_size, -1)[mask_id_mask].max().item())
            if inserted_max > max_mask_id:
                raise ValueError("inserted mask ID exceeds specified max_mask_id")
        prepared_input_ids[mask_id_mask] = per_position_tokens.unsqueeze(0).expand(batch_size, -1)[mask_id_mask]

    prefix_pos_mask = ~pred_pos_mask
    return (
        prepared_input_ids,
        prepared_target_ids,
        mask_id_mask,
        pred_pos_mask,
        prefix_pos_mask,
        final_indices,
        prefix_length,
    )


def _case_tokens(
    rng: random.Random,
    *,
    batch_size: int,
    source_len: int,
    token_min: int,
    token_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for row in range(batch_size):
        base = rng.randint(token_min, token_max - source_len - 2)
        rows.append(torch.arange(base + row * (source_len + 17), base + row * (source_len + 17) + source_len + 1))
    sequence = torch.stack(rows, dim=0).long()
    return sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()


def _offset_for_case(rng: random.Random, *, prefix_length: int, case_idx: int, mode: str) -> int:
    if mode == "zero":
        return 0
    if mode == "cycle":
        return -(case_idx % prefix_length)
    if mode == "random-negative":
        return -rng.randint(0, prefix_length - 1)
    raise ValueError(f"Unsupported offset mode: {mode}")


def _assert_close_bool(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).nonzero(as_tuple=False)
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise AssertionError(f"{name} mismatch at {first}: actual={actual.tolist()} expected={expected.tolist()}")


def _assert_close_long(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).nonzero(as_tuple=False)
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise AssertionError(f"{name} mismatch at {first}: actual={actual.tolist()} expected={expected.tolist()}")


def _prediction_source_rows(source_indices: torch.Tensor, pred_mask: torch.Tensor, *, batch_size: int, k_toks: int) -> list[list[list[int]]]:
    rows: list[list[list[int]]] = []
    for row in range(batch_size):
        selected = source_indices[pred_mask[row]].view(-1, k_toks)
        rows.append([[int(x) for x in region] for region in selected.tolist()])
    return rows


def _mask_invariants(
    *,
    seq_len: int,
    prefix_length: int,
    k_toks: int,
    offset: int,
    prediction_mask: torch.Tensor,
    mask_positions: torch.Tensor,
) -> dict[str, Any]:
    k_masks = k_toks - 1
    allowed = dense_interleaved_mtp_attention_mask(seq_len, prefix_length, k_masks, offset)
    pred_slots = prediction_mask.nonzero(as_tuple=False).reshape(-1)
    mask_slots = mask_positions.nonzero(as_tuple=False).reshape(-1)
    failures: list[str] = []

    if allowed.triu(1).any():
        failures.append("dense mask allows a future key")
    for slot in pred_slots.tolist():
        visible_future_masks = mask_slots[mask_slots > slot]
        if visible_future_masks.numel() > 0 and allowed[slot, visible_future_masks].any():
            failures.append(f"prediction slot {slot} can attend to a future mask slot")
            break
    for slot in mask_slots.tolist():
        if not bool(allowed[slot, slot].item()):
            failures.append(f"mask slot {slot} cannot attend to itself")
            break

    return {
        "ok": not failures,
        "failures": failures,
        "allowed_shape": list(allowed.shape),
        "prediction_slots_head": [int(x) for x in pred_slots[:32].tolist()],
        "mask_slots_head": [int(x) for x in mask_slots[:32].tolist()],
    }


def _run_case(args: argparse.Namespace, rng: random.Random, case_idx: int) -> dict[str, Any]:
    k_toks = rng.randint(args.k_min, args.k_max)
    k_masks = k_toks - 1
    prefix_length = args.sequence_len // args.mask_region_count - k_masks
    offset = _offset_for_case(rng, prefix_length=prefix_length, case_idx=case_idx, mode=args.offset_mode)
    input_ids, target_ids = _case_tokens(
        rng,
        batch_size=args.batch_size,
        source_len=args.source_len,
        token_min=args.token_min,
        token_max=args.token_max,
    )

    ref_input, ref_target, ref_mask, ref_pred, ref_prefix, ref_source, ref_prefix_length = _reference_truncate_and_mask(
        input_ids,
        target_ids,
        k_toks=k_toks,
        mask_id=args.mask_token_id,
        min_mask_id=args.min_mask_token_id,
        max_mask_id=args.max_mask_token_id,
        truncation_length=args.sequence_len,
        mask_region_ct=args.mask_region_count,
        offset=offset,
    )
    prepared = prepare_singleshot_mtp_batch(
        input_ids,
        target_ids,
        k_toks=k_toks,
        mask_token_id=args.mask_token_id,
        min_mask_token_id=args.min_mask_token_id,
        max_mask_token_id=args.max_mask_token_id,
        truncation_length=args.sequence_len,
        mask_region_count=args.mask_region_count,
        offset=offset,
        pad_to_multiple=args.pad_to_multiple,
        ignore_index=IGNORE_INDEX,
    )

    _assert_close_long("input_ids", prepared.input_ids[:, : args.sequence_len], ref_input)
    _assert_close_long("target_ids", prepared.target_ids[:, : args.sequence_len], ref_target)
    _assert_close_bool("mask_positions", prepared.mask_positions[:, : args.sequence_len], ref_mask)
    _assert_close_bool("prediction_positions", prepared.prediction_positions[:, : args.sequence_len], ref_pred)
    _assert_close_bool("prefix_positions", prepared.prefix_positions[:, : args.sequence_len], ref_prefix)
    _assert_close_long("source_indices", prepared.source_indices[: args.sequence_len], ref_source)
    expected_labels = ref_target.masked_fill(~ref_pred, IGNORE_INDEX)
    _assert_close_long("labels", prepared.labels[:, : args.sequence_len], expected_labels)

    expected_pred_tokens = args.batch_size * args.mask_region_count * k_toks
    expected_mask_tokens = args.batch_size * args.mask_region_count * k_masks
    pred_count = int(prepared.prediction_positions.sum().item())
    mask_count = int(prepared.mask_positions.sum().item())
    label_count = int(prepared.labels.ne(IGNORE_INDEX).sum().item())
    if pred_count != expected_pred_tokens:
        raise AssertionError(f"prediction count {pred_count} != expected {expected_pred_tokens}")
    if label_count != expected_pred_tokens:
        raise AssertionError(f"label count {label_count} != expected {expected_pred_tokens}")
    if mask_count != expected_mask_tokens:
        raise AssertionError(f"mask count {mask_count} != expected {expected_mask_tokens}")

    consumed = args.mask_region_count * prefix_length + k_masks
    if ref_prefix_length != prepared.prefix_length:
        raise AssertionError(f"prefix length mismatch: ref={ref_prefix_length} xorl={prepared.prefix_length}")
    real_source_indices = prepared.source_indices[: args.sequence_len]

    invariants = _mask_invariants(
        seq_len=args.sequence_len,
        prefix_length=prefix_length,
        k_toks=k_toks,
        offset=offset,
        prediction_mask=prepared.prediction_positions[0, : args.sequence_len],
        mask_positions=prepared.mask_positions[0, : args.sequence_len],
    )
    if not invariants["ok"]:
        raise AssertionError(f"attention mask invariant failures: {invariants['failures']}")

    case: dict[str, Any] = {
        "case_idx": case_idx,
        "k_toks": k_toks,
        "k_masks": k_masks,
        "prefix_length": prefix_length,
        "offset": offset,
        "sequence_len": args.sequence_len,
        "source_len": args.source_len,
        "mask_region_count": args.mask_region_count,
        "batch_size": args.batch_size,
        "pre_offset_source_tokens_consumed": consumed,
        "source_index_min": int(real_source_indices.min().item()),
        "source_index_max": int(real_source_indices.max().item()),
        "raw_seq_len": prepared.raw_seq_len,
        "padded_seq_len": prepared.padded_seq_len,
        "prediction_count": pred_count,
        "mask_count": mask_count,
        "label_count": label_count,
        "expected_prediction_count": expected_pred_tokens,
        "expected_mask_count": expected_mask_tokens,
        "prediction_source_indices": _prediction_source_rows(
            prepared.source_indices[: args.sequence_len],
            prepared.prediction_positions[:, : args.sequence_len],
            batch_size=args.batch_size,
            k_toks=k_toks,
        )[: args.max_rows_in_case],
        "mask_invariants": invariants,
    }
    return case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-len", type=int, default=160)
    parser.add_argument("--source-len", type=int, default=160)
    parser.add_argument("--mask-region-count", type=int, default=5)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--offset-mode", choices=("random-negative", "cycle", "zero"), default="random-negative")
    parser.add_argument("--mask-token-id", type=int, default=151662)
    parser.add_argument("--min-mask-token-id", type=int, default=None)
    parser.add_argument("--max-mask-token-id", type=int, default=None)
    parser.add_argument("--pad-to-multiple", type=int, default=None)
    parser.add_argument("--token-min", type=int, default=1000)
    parser.add_argument("--token-max", type=int, default=100000)
    parser.add_argument("--max-rows-in-case", type=int, default=1)
    args = parser.parse_args()

    if args.sequence_len % args.mask_region_count != 0:
        raise ValueError("--sequence-len must be divisible by --mask-region-count")
    if args.source_len < args.sequence_len:
        raise ValueError("--source-len must be at least --sequence-len for this clean verifier")
    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError("invalid k range")

    rng = random.Random(args.seed)
    cases: list[dict[str, Any]] = []
    for case_idx in range(args.num_cases):
        cases.append(_run_case(args, rng, case_idx))

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, sort_keys=True) + "\n")

    k_counts = Counter(case["k_toks"] for case in cases)
    consumed_by_k: dict[str, list[int]] = {}
    offsets_by_k: dict[str, list[int]] = {}
    for case in cases:
        key = str(case["k_toks"])
        consumed_by_k.setdefault(key, []).append(int(case["pre_offset_source_tokens_consumed"]))
        offsets_by_k.setdefault(key, []).append(int(case["offset"]))

    summary = {
        "ok": True,
        "num_cases": len(cases),
        "seed": args.seed,
        "settings": {
            "batch_size": args.batch_size,
            "sequence_len": args.sequence_len,
            "source_len": args.source_len,
            "mask_region_count": args.mask_region_count,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "offset_mode": args.offset_mode,
            "mask_token_id": args.mask_token_id,
            "min_mask_token_id": args.min_mask_token_id,
            "max_mask_token_id": args.max_mask_token_id,
            "pad_to_multiple": args.pad_to_multiple,
        },
        "k_counts": {str(key): value for key, value in sorted(k_counts.items())},
        "source_tokens_consumed_by_k": {
            key: sorted(set(values)) for key, values in sorted(consumed_by_k.items(), key=lambda item: int(item[0]))
        },
        "offset_range_by_k": {
            key: [min(values), max(values)] for key, values in sorted(offsets_by_k.items(), key=lambda item: int(item[0]))
        },
        "cases_head": cases[: min(8, len(cases))],
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
