"""AUTO-GENERATED from the live build_rollout_replay_linear_plan (commitlen-fix b0cfc6de).
IDENTICAL except: committed context is trajectory-sorted + deduped (the clean-region change).
Used only by the clean-region proof to forward the CLEAN ordering. Do not edit by hand;
regenerate from the live builder if it changes.
"""
import sys
sys.path.insert(0, "/home/apanda/xorl-mtp-commitlen-fix-20260612/src")
import torch
from typing import Optional
from xorl.mtp.singleshot import (
    RolloutReplayLinearPlan, _ROLL_PROMPT, _ROLL_REFILL, _ROLL_MASK, _ROLL_PAD,
    _STALE_KEY_SOURCE, _build_rollout_replay_stateful_schedule,
)


def build_clean_plan(
    token_kind: torch.Tensor,
    key_source_indices: torch.Tensor,
    query_context_source_indices: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    max_sequences: Optional[int] = None,
    max_outputs: Optional[int] = None,
) -> RolloutReplayLinearPlan:
    """Build dense virtual GDN sequences for a rollout-replay mask.

    This is intentionally eager data preparation. The compiled GDN forward then
    receives only gather/scatter tensors, avoiding Python list construction and
    the varlen ``cu_seqlens`` short-convolution path.
    """

    if token_kind.ndim != 2:
        raise ValueError(f"token_kind must be [batch, seq], got {tuple(token_kind.shape)}")
    expected_shape = token_kind.shape
    for name, tensor in (
        ("key_source_indices", key_source_indices),
        ("query_context_source_indices", query_context_source_indices),
        ("block_ids", block_ids),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")

    device = token_kind.device
    sequences: list[tuple[int, list[int]]] = []
    assignments: list[tuple[int, int, int, int]] = []
    sequence_context_sequence_indices: list[int] = []
    sequence_context_lengths: list[int] = []
    sequence_mask_starts: list[int] = []

    for row in range(token_kind.shape[0]):
        row_kind = token_kind[row]
        is_context_kind = (row_kind == _ROLL_PROMPT) | (row_kind == _ROLL_REFILL)
        context_positions = is_context_kind.nonzero(as_tuple=False).flatten()
        mask_positions = (row_kind == _ROLL_MASK).nonzero(as_tuple=False).flatten()
        context_positions_list = [int(pos) for pos in context_positions.detach().cpu().tolist()]
        mask_positions_list = [int(pos) for pos in mask_positions.detach().cpu().tolist()]
        row_blocks = block_ids[row].detach().cpu().tolist()
        row_key_sources = key_source_indices[row].detach().cpu().tolist()
        row_query_context_sources = query_context_source_indices[row].detach().cpu().tolist()

        # The shared flat sequence carries the CLEAN committed trajectory only.
        # Stale refills (replayed rejected drafts, sentinel key source) are
        # runtime-evicted context: they must not enter the recurrent state that
        # later committed positions read from.
        committed_context_positions = [
            pos for pos in context_positions_list if row_key_sources[pos] < _STALE_KEY_SOURCE
        ]
        stale_positions = [pos for pos in context_positions_list if row_key_sources[pos] >= _STALE_KEY_SOURCE]
        # CLEAN-REGION: committed context = trajectory-sorted (by key_source), deduped (keep first/slot).
        _clean_seen = {}
        for _p in committed_context_positions:
            _clean_seen.setdefault(int(row_key_sources[_p]), _p)
        committed_context_positions = [_clean_seen[_sl] for _sl in sorted(_clean_seen)]
        committed_with_sources = [(pos, int(row_key_sources[pos])) for pos in committed_context_positions]
        context_sequence_idx = -1

        if committed_context_positions:
            sequence_idx = len(sequences)
            context_sequence_idx = sequence_idx
            sequences.append((row, committed_context_positions))
            sequence_context_sequence_indices.append(sequence_idx)
            sequence_context_lengths.append(len(committed_context_positions))
            sequence_mask_starts.append(len(committed_context_positions))
            for local_idx, pos in enumerate(committed_context_positions):
                assignments.append((row, pos, sequence_idx, local_idx))

        # Per-window branches: stale refills and mask slots replay their own
        # decode window verbatim (committed prefix visible BEFORE the window,
        # then the window's tokens in feed order). All window queries share
        # q_context = window_start - 1, so one branch per window suffices.
        branch_query_positions = sorted(stale_positions + mask_positions_list)
        if not branch_query_positions:
            continue

        row_kind_list = row_kind.detach().cpu().tolist()
        window_positions_by_block: dict[int, list[int]] = {}
        for pos, block in enumerate(row_blocks):
            if int(block) >= 0 and int(row_kind_list[pos]) != _ROLL_PAD:
                window_positions_by_block.setdefault(int(block), []).append(pos)

        for block_id in sorted({int(row_blocks[pos]) for pos in branch_query_positions if int(row_blocks[pos]) >= 0}):
            block_queries = [int(pos) for pos in branch_query_positions if int(row_blocks[pos]) == block_id]
            positions_by_context: dict[int, list[int]] = {}
            for query_pos in block_queries:
                positions_by_context.setdefault(int(row_query_context_sources[query_pos]), []).append(query_pos)

            for query_context, query_positions in positions_by_context.items():
                visible_context = tuple(
                    pos for pos, source_idx in committed_with_sources if source_idx <= query_context
                )
                visible_set = set(visible_context)
                last_query_pos = query_positions[-1]
                window_members = [
                    pos
                    for pos in window_positions_by_block.get(block_id, [])
                    if pos <= last_query_pos and pos not in visible_set
                ]
                cols = list(visible_context) + window_members
                if not cols or cols[-1] != last_query_pos:
                    raise ValueError(
                        "SingleShot replay mask did not produce a branch ending at the window query "
                        f"(row={row}, query={last_query_pos})"
                    )
                sequence_idx = len(sequences)
                sequences.append((row, cols))
                sequence_context_sequence_indices.append(context_sequence_idx)
                sequence_context_lengths.append(len(visible_context))
                sequence_mask_starts.append(len(visible_context))
                window_offset = len(visible_context)
                window_position_to_local = {pos: window_offset + idx for idx, pos in enumerate(window_members)}
                for query_pos in query_positions:
                    assignments.append((row, query_pos, sequence_idx, window_position_to_local[query_pos]))

    num_sequences = len(sequences)
    sequence_capacity = num_sequences if max_sequences is None else int(max_sequences)
    if sequence_capacity < num_sequences:
        raise ValueError(
            "SingleShot replay linear-plan sequence capacity is too small: "
            f"capacity={sequence_capacity} required={num_sequences}"
        )
    max_len = token_kind.shape[1] if sequences else 0
    sequence_rows = torch.zeros((sequence_capacity, max_len), dtype=torch.long, device=device)
    sequence_cols = torch.zeros((sequence_capacity, max_len), dtype=torch.long, device=device)
    sequence_valid = torch.zeros((sequence_capacity, max_len), dtype=torch.bool, device=device)
    packed_rows_list: list[int] = []
    packed_cols_list: list[int] = []
    sequence_offsets = [0]
    for sequence_idx, (row, cols) in enumerate(sequences):
        length = len(cols)
        sequence_rows[sequence_idx, :length] = row
        sequence_cols[sequence_idx, :length] = torch.tensor(cols, dtype=torch.long, device=device)
        sequence_valid[sequence_idx, :length] = True
        packed_rows_list.extend([row] * length)
        packed_cols_list.extend(cols)
        sequence_offsets.append(len(packed_rows_list))
    while len(sequence_offsets) <= sequence_capacity:
        sequence_offsets.append(len(packed_rows_list))
    packed_rows = torch.tensor(packed_rows_list, dtype=torch.long, device=device)
    packed_cols = torch.tensor(packed_cols_list, dtype=torch.long, device=device)
    context_sequence_indices = torch.full((sequence_capacity,), -1, dtype=torch.long, device=device)
    context_lengths = torch.zeros(sequence_capacity, dtype=torch.long, device=device)
    mask_starts = torch.zeros(sequence_capacity, dtype=torch.long, device=device)
    if num_sequences:
        context_sequence_indices[:num_sequences] = torch.tensor(
            sequence_context_sequence_indices, dtype=torch.long, device=device
        )
        context_lengths[:num_sequences] = torch.tensor(sequence_context_lengths, dtype=torch.long, device=device)
        mask_starts[:num_sequences] = torch.tensor(sequence_mask_starts, dtype=torch.long, device=device)

    output_count = len(assignments)
    output_capacity = output_count if max_outputs is None else int(max_outputs)
    if output_capacity < output_count:
        raise ValueError(
            "SingleShot replay linear-plan output capacity is too small: "
            f"capacity={output_capacity} required={output_count}"
        )
    output_rows = torch.zeros(output_capacity, dtype=torch.long, device=device)
    output_cols = torch.zeros(output_capacity, dtype=torch.long, device=device)
    output_sequence_indices = torch.zeros(output_capacity, dtype=torch.long, device=device)
    output_token_indices = torch.zeros(output_capacity, dtype=torch.long, device=device)
    output_valid = torch.zeros(output_capacity, dtype=torch.bool, device=device)
    if output_count:
        output_rows[:output_count] = torch.tensor(
            [row for row, _, _, _ in assignments], dtype=torch.long, device=device
        )
        output_cols[:output_count] = torch.tensor(
            [col for _, col, _, _ in assignments], dtype=torch.long, device=device
        )
        output_sequence_indices[:output_count] = torch.tensor(
            [seq for _, _, seq, _ in assignments], dtype=torch.long, device=device
        )
        output_token_indices[:output_count] = torch.tensor(
            [pos for _, _, _, pos in assignments], dtype=torch.long, device=device
        )
        output_valid[:output_count] = True
    output_counts_by_sequence = [0] * sequence_capacity
    for _, _, sequence_idx, _ in assignments:
        output_counts_by_sequence[sequence_idx] += 1
    output_offsets = [0]
    offset = 0
    for count in output_counts_by_sequence:
        offset += count
        output_offsets.append(offset)
    stateful_schedule = _build_rollout_replay_stateful_schedule(
        sequences=sequences,
        context_sequence_indices=sequence_context_sequence_indices,
        context_lengths=sequence_context_lengths,
        mask_starts=sequence_mask_starts,
        assignments=assignments,
        seq_len=int(token_kind.shape[1]),
        device=device,
    )
    return RolloutReplayLinearPlan(
        sequence_rows=sequence_rows,
        sequence_cols=sequence_cols,
        sequence_valid=sequence_valid,
        output_sequence_indices=output_sequence_indices,
        output_token_indices=output_token_indices,
        output_rows=output_rows,
        output_cols=output_cols,
        output_valid=output_valid,
        packed_rows=packed_rows,
        packed_cols=packed_cols,
        sequence_context_sequence_indices=context_sequence_indices,
        sequence_context_lengths=context_lengths,
        sequence_mask_starts=mask_starts,
        sequence_count=num_sequences,
        output_count=output_count,
        sequence_offsets=tuple(sequence_offsets),
        output_offsets=tuple(output_offsets),
        stateful_schedule=stateful_schedule,
    )

