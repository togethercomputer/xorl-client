#!/usr/bin/env python3
"""Build tokenized prompt datasets for SingleShot MTP OPD runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


HISTOGRAM_BINS = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _parse_json_int_list(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"Expected a non-empty JSON list of ints, got {value!r}")
    return [int(item) for item in parsed]


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _histogram(values: list[int], bins: list[int] | None = None) -> dict[str, int]:
    bins = bins or HISTOGRAM_BINS
    counts: dict[str, int] = {}
    if not values:
        return counts
    for lower, upper in zip(bins, bins[1:]):
        counts[f"[{lower},{upper})"] = 0
    counts[f">={bins[-1]}"] = 0
    for value in values:
        placed = False
        for lower, upper in zip(bins, bins[1:]):
            if lower <= value < upper:
                counts[f"[{lower},{upper})"] += 1
                placed = True
                break
        if not placed:
            counts[f">={bins[-1]}"] += 1
    return {key: count for key, count in counts.items() if count}


def _summary_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    ordered = sorted(values)

    def quantile(q: float) -> int:
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[index]

    return {
        "count": len(values),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def _find_assistant_content_spans(
    token_ids: list[int],
    *,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: list[int],
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    """Return assistant ``(turn_start, content_start, content_end)`` spans and marker diagnostics."""

    if not assistant_role_token_ids:
        raise ValueError("assistant_role_token_ids must be non-empty")

    spans: list[tuple[int, int, int]] = []
    stats = {
        "assistant_marker_count": 0,
        "assistant_empty_count": 0,
        "assistant_missing_end_count": 0,
    }
    role_len = len(assistant_role_token_ids)
    i = 0
    while i + 1 + role_len <= len(token_ids):
        if token_ids[i] != im_start_token_id or token_ids[i + 1 : i + 1 + role_len] != assistant_role_token_ids:
            i += 1
            continue

        stats["assistant_marker_count"] += 1
        content_start = i + 1 + role_len
        if content_start < len(token_ids) and token_ids[content_start] == 198:
            content_start += 1
        content_end = content_start
        while content_end < len(token_ids) and token_ids[content_end] != im_end_token_id:
            content_end += 1
        if content_end >= len(token_ids):
            stats["assistant_missing_end_count"] += 1
            i += 1
            continue
        if content_end <= content_start:
            stats["assistant_empty_count"] += 1
        else:
            spans.append((i, content_start, content_end))
        i = max(content_end + 1, i + 1)
    return spans, stats


def _iter_parquet_rows(
    source_path: Path,
    *,
    input_column: str,
    text_column: str,
    doc_id_column: str,
    metadata_columns: list[str],
    include_input: bool,
    include_text: bool,
    max_input_files: int | None,
    max_source_rows: int | None,
) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted(source_path.glob("*.parquet")) if source_path.is_dir() else [source_path]
    if max_input_files is not None:
        files = files[:max_input_files]
    if not files:
        raise ValueError(f"No parquet files found at {source_path}")

    emitted = 0
    for source_file in files:
        parquet_file = pq.ParquetFile(source_file)
        schema_names = set(parquet_file.schema_arrow.names)
        columns = []
        if include_input:
            columns.append(input_column)
        if include_text and text_column in schema_names and text_column not in columns:
            columns.append(text_column)
        if doc_id_column in schema_names:
            columns.append(doc_id_column)
        columns.extend(column for column in metadata_columns if column in schema_names and column not in columns)
        if not columns:
            raise ValueError(f"No requested columns found in {source_file}")
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=columns)
            row_group_offset = sum(parquet_file.metadata.row_group(idx).num_rows for idx in range(row_group_idx))
            for row_group_row_idx, row in enumerate(table.to_pylist()):
                yield {
                    "source_file": str(source_file),
                    "source_file_row": row_group_offset + row_group_row_idx,
                    "input_ids": row.get(input_column),
                    "text": row.get(text_column),
                    "doc_id": row.get(doc_id_column),
                    "metadata": {column: row.get(column) for column in metadata_columns if column in row},
                }
                emitted += 1
                if max_source_rows is not None and emitted >= max_source_rows:
                    return


def _assistant_turn_records_from_row(
    *,
    token_ids: list[int],
    source_row: int,
    source_file: str,
    source_file_row: int,
    doc_id: str,
    metadata: dict[str, Any],
    ctx_len: int,
    target_cap: int,
    min_target_tokens: int,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: list[int],
    assistant_offset_tokens: int,
    drop_prompt_truncated: bool,
    drop_target_truncated: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spans, marker_stats = _find_assistant_content_spans(
        token_ids,
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        assistant_role_token_ids=assistant_role_token_ids,
    )
    records: list[dict[str, Any]] = []
    row_stats = {**marker_stats, "target_too_short_count": 0, "prompt_truncated_drop_count": 0}
    row_stats["target_truncated_drop_count"] = 0

    for turn_idx, (turn_start, content_start, content_end) in enumerate(spans):
        latest_prompt_end = content_end - min_target_tokens
        if latest_prompt_end < content_start:
            row_stats["target_too_short_count"] += 1
            continue
        prompt_end = min(content_start + assistant_offset_tokens, latest_prompt_end)
        prompt_start = max(0, prompt_end - ctx_len)
        target_start = prompt_end
        target_len = content_end - target_start
        if target_len < min_target_tokens:
            row_stats["target_too_short_count"] += 1
            continue
        prompt_truncated = prompt_start > 0
        target_truncated = target_len > target_cap
        if prompt_truncated and drop_prompt_truncated:
            row_stats["prompt_truncated_drop_count"] += 1
            continue
        if target_truncated and drop_target_truncated:
            row_stats["target_truncated_drop_count"] += 1
            continue
        prompt_ids = [int(token_id) for token_id in token_ids[prompt_start:prompt_end]]
        target_ids = [int(token_id) for token_id in token_ids[target_start : target_start + target_cap]]
        if not prompt_ids:
            continue
        records.append(
            {
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "doc_id": doc_id,
                "turn_idx": turn_idx,
                "source_row": source_row,
                "source_file": source_file,
                "source_file_row": source_file_row,
                "context_len": prompt_end,
                "prompt_len": len(prompt_ids),
                "target_len": target_len,
                "target_cap": target_cap,
                "prompt_truncated": prompt_truncated,
                "target_truncated": target_truncated,
                "assistant_turn_start": turn_start,
                "assistant_content_start": content_start,
                "assistant_content_end": content_end,
                "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            }
        )
    return records, row_stats


def _load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _chatml_token_ids_from_tokenizer(tokenizer: Any) -> dict[str, Any]:
    assistant_header_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    im_end_ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(assistant_header_ids) < 3 or assistant_header_ids[-1] != 198:
        raise ValueError(f"Unexpected assistant ChatML header tokenization: {assistant_header_ids}")
    if len(im_end_ids) != 1:
        raise ValueError(f"Unexpected <|im_end|> tokenization: {im_end_ids}")
    return {
        "im_start_token_id": int(assistant_header_ids[0]),
        "assistant_role_token_ids": [int(token_id) for token_id in assistant_header_ids[1:-1]],
        "im_end_token_id": int(im_end_ids[0]),
        "assistant_header_ids": [int(token_id) for token_id in assistant_header_ids],
        "im_end_ids": [int(token_id) for token_id in im_end_ids],
    }


def _verify_qwen_chatml_ids(
    tokenizer_path: str,
    *,
    im_start_token_id: int,
    im_end_token_id: int,
    assistant_role_token_ids: list[int],
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_path)
    actual = _chatml_token_ids_from_tokenizer(tokenizer)
    expected_header = [im_start_token_id, *assistant_role_token_ids, 198]
    result = {
        "tokenizer_path": tokenizer_path,
        **actual,
        "expected_assistant_header_ids": expected_header,
        "expected_im_end_ids": [im_end_token_id],
        "matches": actual["assistant_header_ids"] == expected_header and actual["im_end_ids"] == [im_end_token_id],
    }
    if not result["matches"]:
        raise ValueError(f"Qwen ChatML token id verification failed: {result}")
    return result


def build_assistant_turn_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if args.ctx_len <= 0:
        raise ValueError(f"--ctx-len must be positive, got {args.ctx_len}")
    if args.target_cap <= 0:
        raise ValueError(f"--target-cap must be positive, got {args.target_cap}")
    if not 0.0 <= args.eval_fraction < 1.0:
        raise ValueError(f"--eval-fraction must be in [0, 1), got {args.eval_fraction}")
    write_batch_size = int(getattr(args, "write_batch_size", 2048))
    if write_batch_size <= 0:
        raise ValueError(f"--write-batch-size must be positive, got {write_batch_size}")

    source_path = Path(args.source_tokenized_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assistant_role_token_ids = _parse_json_int_list(args.assistant_role_token_ids_json)
    metadata_columns = _split_csv(args.metadata_columns)

    retokenizer = None
    tokenizer_check = None
    if args.retokenize_tokenizer_path:
        retokenizer = _load_tokenizer(args.retokenize_tokenizer_path)
        derived_ids = _chatml_token_ids_from_tokenizer(retokenizer)
        tokenizer_check = {
            "tokenizer_path": args.retokenize_tokenizer_path,
            **derived_ids,
            "requested_im_start_token_id": args.im_start_token_id,
            "requested_im_end_token_id": args.im_end_token_id,
            "requested_assistant_role_token_ids": assistant_role_token_ids,
            "derive_chatml_token_ids": args.derive_chatml_token_ids,
        }
        if args.derive_chatml_token_ids:
            args.im_start_token_id = int(derived_ids["im_start_token_id"])
            args.im_end_token_id = int(derived_ids["im_end_token_id"])
            assistant_role_token_ids = [int(token_id) for token_id in derived_ids["assistant_role_token_ids"]]

    if args.verify_tokenizer_path:
        tokenizer_check = _verify_qwen_chatml_ids(
            args.verify_tokenizer_path,
            im_start_token_id=args.im_start_token_id,
            im_end_token_id=args.im_end_token_id,
            assistant_role_token_ids=assistant_role_token_ids,
        )

    schema = pa.schema(
        [
            ("prompt_ids", pa.list_(pa.int32())),
            ("target_ids", pa.list_(pa.int32())),
            ("doc_id", pa.string()),
            ("turn_idx", pa.int32()),
            ("source_row", pa.int64()),
            ("source_file", pa.string()),
            ("source_file_row", pa.int32()),
            ("context_len", pa.int32()),
            ("prompt_len", pa.int32()),
            ("target_len", pa.int32()),
            ("target_cap", pa.int32()),
            ("prompt_truncated", pa.bool_()),
            ("target_truncated", pa.bool_()),
            ("assistant_turn_start", pa.int32()),
            ("assistant_content_start", pa.int32()),
            ("assistant_content_end", pa.int32()),
            ("metadata_json", pa.string()),
            ("split", pa.string()),
        ]
    )
    split_paths = {"train": output_dir / "train.parquet", "eval": output_dir / "eval.parquet"}
    for stale_path in [*split_paths.values(), output_dir / "summary.json"]:
        stale_path.unlink(missing_ok=True)
    split_buffers: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    split_writers: dict[str, Any] = {}
    split_rows = {"train": 0, "eval": 0}
    split_docs: dict[str, set[str]] = {"train": set(), "eval": set()}
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    prompt_truncated_count = 0
    target_truncated_count = 0

    def flush_split(split: str) -> None:
        rows = split_buffers[split]
        if not rows:
            return
        if split not in split_writers:
            split_writers[split] = pq.ParquetWriter(split_paths[split], schema)
        table = pa.Table.from_pylist(rows, schema=schema)
        split_writers[split].write_table(table)
        split_buffers[split] = []

    def append_record(split: str, record: dict[str, Any]) -> None:
        nonlocal prompt_truncated_count, target_truncated_count
        record["split"] = split
        split_buffers[split].append(record)
        split_rows[split] += 1
        split_docs[split].add(str(record["doc_id"]))
        prompt_lengths.append(int(record["prompt_len"]))
        target_lengths.append(int(record["target_len"]))
        prompt_truncated_count += int(bool(record["prompt_truncated"]))
        target_truncated_count += int(bool(record["target_truncated"]))
        if len(split_buffers[split]) >= write_batch_size:
            flush_split(split)

    stats = {
        "source_rows": 0,
        "assistant_marker_count": 0,
        "assistant_empty_count": 0,
        "assistant_missing_end_count": 0,
        "target_too_short_count": 0,
        "prompt_truncated_drop_count": 0,
        "target_truncated_drop_count": 0,
        "missing_token_source_count": 0,
    }
    docs: set[str] = set()
    turns_per_doc: dict[str, int] = {}
    source_row = 0
    try:
        for row in _iter_parquet_rows(
            source_path,
            input_column=args.input_column,
            text_column=args.text_column,
            doc_id_column=args.doc_id_column,
            metadata_columns=metadata_columns,
            include_input=retokenizer is None,
            include_text=retokenizer is not None,
            max_input_files=args.max_input_files,
            max_source_rows=args.max_source_rows,
        ):
            if retokenizer is None:
                token_ids = row["input_ids"]
            else:
                text = row["text"]
                token_ids = retokenizer(str(text), add_special_tokens=False)["input_ids"] if text else None
            if token_ids is None:
                stats["missing_token_source_count"] += 1
                source_row += 1
                continue
            doc_id = str(row["doc_id"] or f"{Path(row['source_file']).name}:{row['source_file_row']}")
            docs.add(doc_id)
            row_records, row_stats = _assistant_turn_records_from_row(
                token_ids=[int(token_id) for token_id in token_ids],
                source_row=source_row,
                source_file=row["source_file"],
                source_file_row=int(row["source_file_row"]),
                doc_id=doc_id,
                metadata=row["metadata"],
                ctx_len=args.ctx_len,
                target_cap=args.target_cap,
                min_target_tokens=args.min_target_tokens,
                im_start_token_id=args.im_start_token_id,
                im_end_token_id=args.im_end_token_id,
                assistant_role_token_ids=assistant_role_token_ids,
                assistant_offset_tokens=args.assistant_offset_tokens,
                drop_prompt_truncated=args.drop_prompt_truncated,
                drop_target_truncated=args.drop_target_truncated,
            )
            for key in stats:
                stats[key] += int(row_stats.get(key, 0))
            stats["source_rows"] += 1
            if row_records:
                split = "eval" if _stable_fraction(doc_id) < args.eval_fraction else "train"
                turns_per_doc[doc_id] = turns_per_doc.get(doc_id, 0) + len(row_records)
                for record in row_records:
                    append_record(split, record)
            source_row += 1
        for split in split_buffers:
            flush_split(split)
    finally:
        for writer in split_writers.values():
            writer.close()

    train_path = str(split_paths["train"]) if split_rows["train"] else None
    eval_path = str(split_paths["eval"]) if split_rows["eval"] else None
    total_rows = split_rows["train"] + split_rows["eval"]
    turns_per_doc_values = list(turns_per_doc.values())
    malformed_markers = stats["assistant_empty_count"] + stats["assistant_missing_end_count"]
    marker_count = stats["assistant_marker_count"]
    summary = {
        "source_tokenized_path": str(source_path),
        "output_dir": str(output_dir),
        "train_parquet": train_path,
        "eval_parquet": eval_path,
        "ctx_len": args.ctx_len,
        "target_cap": args.target_cap,
        "min_target_tokens": args.min_target_tokens,
        "input_column": args.input_column,
        "text_column": args.text_column,
        "retokenize_tokenizer_path": args.retokenize_tokenizer_path,
        "doc_id_column": args.doc_id_column,
        "assistant_role_token_ids": assistant_role_token_ids,
        "tokenizer_check": tokenizer_check,
        "write_batch_size": write_batch_size,
        "rows": total_rows,
        "source_rows": stats["source_rows"],
        "documents": len(docs),
        "documents_with_qualifying_turns": len(turns_per_doc),
        "assistant_turns": total_rows,
        "assistant_marker_count": marker_count,
        "malformed_marker_count": malformed_markers,
        "malformed_marker_rate": malformed_markers / marker_count if marker_count else 0.0,
        "empty_assistant_count": stats["assistant_empty_count"],
        "missing_end_count": stats["assistant_missing_end_count"],
        "target_too_short_count": stats["target_too_short_count"],
        "missing_token_source_count": stats["missing_token_source_count"],
        "prompt_truncated_count": prompt_truncated_count,
        "prompt_truncated_rate": prompt_truncated_count / total_rows if total_rows else 0.0,
        "target_truncated_count": target_truncated_count,
        "target_truncation_rate": target_truncated_count / total_rows if total_rows else 0.0,
        "prompt_length_stats": _summary_stats(prompt_lengths),
        "target_length_stats": _summary_stats(target_lengths),
        "assistant_turns_per_document_stats": _summary_stats(turns_per_doc_values),
        "prompt_length_histogram": _histogram(prompt_lengths),
        "target_length_histogram": _histogram(target_lengths),
        "assistant_turns_per_document_histogram": _histogram(turns_per_doc_values),
        "train": {
            "rows": split_rows["train"],
            "documents": len(split_docs["train"]),
            "path": train_path,
        },
        "eval": {
            "rows": split_rows["eval"],
            "documents": len(split_docs["eval"]),
            "path": eval_path,
            "fraction": args.eval_fraction,
        },
        "drop_counts": {
            "prompt_truncated": stats["prompt_truncated_drop_count"],
            "target_truncated": stats["target_truncated_drop_count"],
            "target_too_short": stats["target_too_short_count"],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not total_rows:
        raise SystemExit(f"No assistant-turn rows written; see {summary_path}")
    print(json.dumps(summary, sort_keys=True))
    return summary


def _row_text(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("query") or row.get("question") or row.get("input") or "").strip()
    if not question:
        raise ValueError(f"Could not find question/query/input in row keys: {sorted(row)}")

    if isinstance(row.get("response"), str):
        response = row["response"].strip()
    elif isinstance(row.get("answer"), str) and isinstance(row.get("steps"), list):
        steps = "\n".join(str(step).strip() for step in row["steps"] if str(step).strip())
        answer = row["answer"].strip()
        response = f"{steps}\n#### {answer}".strip() if steps else f"#### {answer}"
    elif isinstance(row.get("answer"), str):
        response = row["answer"].strip()
    else:
        raise ValueError(f"Could not find response/answer in row keys: {sorted(row)}")

    return question, response


def _tokenize_text(tokenizer: Any, text: str, *, prepend_bos: bool) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if prepend_bos:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.pad_token_id
        if bos_id is None:
            raise ValueError("Tokenizer has neither bos_token_id nor pad_token_id; cannot prepend BOS.")
        token_ids = [int(bos_id), *[int(token_id) for token_id in token_ids]]
    return [int(token_id) for token_id in token_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokenized-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-column", default="input_ids")
    parser.add_argument("--text-column", default="chat_template_applied")
    parser.add_argument("--doc-id-column", default="trajectory_id")
    parser.add_argument("--metadata-columns", default="reward")
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--target-cap", type=int, default=256)
    parser.add_argument("--min-target-tokens", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--max-input-files", type=int, default=None)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--assistant-offset-tokens", type=int, default=0)
    parser.add_argument("--im-start-token-id", type=int, default=151644)
    parser.add_argument("--im-end-token-id", type=int, default=151645)
    parser.add_argument("--assistant-role-token-ids-json", default="[77091]")
    parser.add_argument("--drop-prompt-truncated", action="store_true")
    parser.add_argument("--drop-target-truncated", action="store_true")
    parser.add_argument("--retokenize-tokenizer-path", default=None)
    parser.add_argument("--derive-chatml-token-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-tokenizer-path", default=None)
    parser.add_argument("--write-batch-size", type=int, default=2048)

    parser.add_argument("--dataset", default="whynlp/gsm8k-aug-nl")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--prepend-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.source_tokenized_path:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --source-tokenized-path")
        build_assistant_turn_dataset(args)
        return

    if not args.tokenizer_path or not args.output_jsonl:
        raise SystemExit("--tokenizer-path and --output-jsonl are required unless --source-tokenized-path is set")

    from datasets import load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=args.trust_remote_code)
    dataset_kwargs: dict[str, Any] = {"split": args.split}
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, **dataset_kwargs)
    else:
        dataset = load_dataset(args.dataset, **dataset_kwargs)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_short = 0
    skipped_long = 0
    skipped_bad = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, row in enumerate(dataset):
            if written >= args.limit:
                break
            try:
                question, response = _row_text(row)
            except ValueError:
                skipped_bad += 1
                continue
            text = f"{question}{args.separator}{response}"
            token_ids = _tokenize_text(tokenizer, text, prepend_bos=args.prepend_bos)
            if len(token_ids) < args.min_tokens:
                skipped_short += 1
                continue
            if len(token_ids) > args.max_tokens:
                skipped_long += 1
                continue
            handle.write(
                json.dumps(
                    {
                        "input_ids": token_ids,
                        "dataset": args.dataset,
                        "dataset_config": args.dataset_config,
                        "split": args.split,
                        "row_idx": row_idx,
                        "question": question,
                        "response": response,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1

    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "written": written,
                "skipped_short": skipped_short,
                "skipped_long": skipped_long,
                "skipped_bad": skipped_bad,
                "tokenizer_path": args.tokenizer_path,
                "dataset": args.dataset,
                "split": args.split,
                "prepend_bos": args.prepend_bos,
            },
            sort_keys=True,
        )
    )
    if written == 0:
        raise SystemExit("No rows written")


if __name__ == "__main__":
    main()
