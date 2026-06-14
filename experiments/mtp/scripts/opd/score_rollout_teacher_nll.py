#!/usr/bin/env python3
"""Score saved OPD rollout samples under a frozen HF teacher.

This is a debugging tool, not part of the training hot path. It reads
`rollout_samples.jsonl`, reconstructs the prompt plus generated continuation
when the artifact contains enough tokens, and reports teacher NLL plus teacher
argmax labels for the continuation positions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


IGNORE_INDEX = -100


def _decode(tokenizer: Any | None, token_ids: list[int], *, max_chars: int) -> str:
    if tokenizer is None:
        return ""
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text[:max_chars]


def _full_prompt_tokens(record: dict[str, Any]) -> list[int] | None:
    prompt_len = int(record.get("prompt_token_count") or 0)
    prompt_tail = [int(x) for x in record.get("prompt_tail_token_ids") or []]
    if prompt_len > 0 and len(prompt_tail) == prompt_len:
        return prompt_tail

    sequence_len = int(record.get("sequence_token_count") or 0)
    sequence_tail = [int(x) for x in record.get("sequence_tail_token_ids") or []]
    if prompt_len > 0 and sequence_len > 0 and len(sequence_tail) == sequence_len:
        return sequence_tail[:prompt_len]
    return None


def _continuation_tokens(record: dict[str, Any]) -> list[int]:
    tokens = record.get("generated_token_ids_full")
    if tokens is None:
        tokens = record.get("generated_token_ids") or []
    return [int(x) for x in tokens]


def _probe_tokens(tokenizer: Any, text: str, length: int) -> list[int]:
    ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise ValueError(f"Probe text {text!r} produced no tokens")
    return [ids[i % len(ids)] for i in range(length)]


def _most_common_continuation_probe(continuation: list[int]) -> list[int]:
    if not continuation:
        return []
    token_id, _ = Counter(continuation).most_common(1)[0]
    return [int(token_id)] * len(continuation)


def _repeated_run_spans(tokens: list[int], *, min_run_len: int) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start = 0
    while start < len(tokens):
        stop = start + 1
        while stop < len(tokens) and tokens[stop] == tokens[start]:
            stop += 1
        run_len = stop - start
        if run_len >= min_run_len:
            spans.append(
                {
                    "start": start,
                    "end": stop,
                    "length": run_len,
                    "token_id": int(tokens[start]),
                }
            )
        start = stop
    return spans


def _repeated_run_membership(tokens: list[int], *, min_run_len: int) -> tuple[list[dict[str, int]], list[int | None]]:
    spans = _repeated_run_spans(tokens, min_run_len=min_run_len)
    membership: list[int | None] = [None] * len(tokens)
    for run_idx, span in enumerate(spans):
        for pos in range(span["start"], span["end"]):
            membership[pos] = run_idx
    return spans, membership


def _masked_mean(values: list[float], mask: list[bool]) -> float | None:
    selected = [value for value, keep in zip(values, mask, strict=False) if keep]
    return _mean(selected)


def _masked_rate(values: list[bool], mask: list[bool]) -> float | None:
    selected = [value for value, keep in zip(values, mask, strict=False) if keep]
    if not selected:
        return None
    return sum(1 for value in selected if value) / len(selected)


def _annotate_span(tokenizer: Any | None, span: dict[str, int] | None, *, max_chars: int) -> dict[str, Any] | None:
    if span is None:
        return None
    annotated = dict(span)
    annotated["token_text"] = _decode(tokenizer, [span["token_id"]], max_chars=max_chars)
    return annotated


def _token_audit_rows(
    tokenizer: Any | None,
    *,
    record: dict[str, Any],
    continuation: list[int],
    scored: dict[str, Any],
    run_spans: list[dict[str, int]],
    run_membership: list[int | None],
    max_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mtp_k = record.get("mtp_k_toks")
    for pos, token_id in enumerate(continuation):
        run_idx = run_membership[pos]
        run_span = run_spans[run_idx] if run_idx is not None else None
        row: dict[str, Any] = {
            "step": record.get("step"),
            "sample_idx": record.get("sample_idx"),
            "continuation_position": pos,
            "student_token_id": int(token_id),
            "student_token_text": _decode(tokenizer, [int(token_id)], max_chars=max_chars),
            "teacher_argmax_token_id": scored["teacher_argmax_token_ids"][pos],
            "teacher_argmax_text": _decode(
                tokenizer,
                [scored["teacher_argmax_token_ids"][pos]],
                max_chars=max_chars,
            ),
            "teacher_argmax_matches_student": scored["teacher_argmax_matches_continuation"][pos],
            "teacher_nll/student_token": scored["continuation_token_nll"][pos],
            "teacher_prob/student_token": scored["continuation_token_probs"][pos],
            "teacher_prob/argmax": scored["teacher_argmax_probs"][pos],
            "teacher_entropy": scored["teacher_entropy"][pos],
            "in_repeated_run": run_idx is not None,
        }
        if mtp_k:
            mtp_k_int = int(mtp_k)
            row["mtp_k_toks"] = mtp_k_int
            row["mtp_block_index"] = pos // mtp_k_int
            row["mtp_block_offset"] = pos % mtp_k_int
        if run_span is not None:
            row.update(
                {
                    "repeated_run_index": int(run_idx),
                    "repeated_run_start": run_span["start"],
                    "repeated_run_end": run_span["end"],
                    "repeated_run_length": run_span["length"],
                    "repeated_run_token_id": run_span["token_id"],
                    "repeated_run_token_text": _decode(tokenizer, [run_span["token_id"]], max_chars=max_chars),
                }
            )
        rows.append(row)
    return rows


@torch.inference_mode()
def _score_continuation(
    model: Any,
    *,
    prompt: list[int],
    continuation: list[int],
    device: torch.device,
) -> dict[str, Any]:
    if not prompt:
        raise ValueError("Cannot score continuation with an empty prompt")
    if not continuation:
        return {
            "token_count": 0,
            "nll_mean": 0.0,
            "nll_sum": 0.0,
            "teacher_argmax_token_ids": [],
            "teacher_argmax_probs": [],
            "teacher_argmax_matches_continuation": [],
            "teacher_argmax_matches_continuation_rate": 0.0,
            "continuation_token_nll": [],
            "continuation_token_probs": [],
            "teacher_entropy": [],
        }

    sequence = prompt + continuation
    input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor(sequence[1:], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    continuation_start = len(prompt) - 1
    continuation_stop = continuation_start + len(continuation)
    cont_log_probs = log_probs[continuation_start:continuation_stop]
    cont_labels = labels[continuation_start:continuation_stop]
    nll = -cont_log_probs.gather(dim=-1, index=cont_labels[:, None]).squeeze(-1)
    teacher_argmax = cont_log_probs.argmax(dim=-1)
    teacher_argmax_probs = cont_log_probs.gather(dim=-1, index=teacher_argmax[:, None]).squeeze(-1).exp()
    continuation_token_probs = (-nll).exp()
    probs = cont_log_probs.exp()
    teacher_entropy = -(probs * cont_log_probs).sum(dim=-1)
    matches = (teacher_argmax == cont_labels).float()
    return {
        "token_count": int(cont_labels.numel()),
        "nll_mean": float(nll.mean().item()),
        "nll_sum": float(nll.sum().item()),
        "teacher_argmax_token_ids": [int(x) for x in teacher_argmax.detach().cpu().tolist()],
        "teacher_argmax_probs": [float(x) for x in teacher_argmax_probs.detach().cpu().tolist()],
        "teacher_argmax_matches_continuation": [bool(x) for x in matches.bool().detach().cpu().tolist()],
        "teacher_argmax_matches_continuation_rate": float(matches.mean().item()),
        "continuation_token_nll": [float(x) for x in nll.detach().cpu().tolist()],
        "continuation_token_probs": [float(x) for x in continuation_token_probs.detach().cpu().tolist()],
        "teacher_entropy": [float(x) for x in teacher_entropy.detach().cpu().tolist()],
    }


@torch.inference_mode()
def _greedy_teacher_continuation(
    model: Any,
    *,
    prompt: list[int],
    max_new_tokens: int,
    device: torch.device,
    eos_token_id: int | None,
) -> list[int]:
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id,
    )[0].detach().cpu().tolist()
    return [int(x) for x in generated[len(prompt) :]]


def _iter_records(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-samples-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--per-token-output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--include-ar-teacher-rollout", action="store_true")
    parser.add_argument("--min-repeated-run-len", type=int, default=3)
    parser.add_argument("--max-token-text-chars", type=int, default=80)
    parser.add_argument("--text-max-chars", type=int, default=1000)
    args = parser.parse_args()
    if args.min_repeated_run_len <= 1:
        raise ValueError("--min-repeated-run-len must be greater than 1")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    input_path = Path(args.rollout_samples_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_output_path = Path(args.per_token_output_jsonl) if args.per_token_output_jsonl else None
    token_handle = None
    if token_output_path is not None:
        token_output_path.parent.mkdir(parents=True, exist_ok=True)
        token_handle = token_output_path.open("w", encoding="utf-8")
    records = _iter_records(input_path, limit=args.limit)

    summary_values: dict[str, list[float]] = {
        "teacher_nll/student_rollout": [],
        "teacher_argmax_matches_student_rollout": [],
        "teacher_nll/student_rollout_repeated_runs": [],
        "teacher_nll/student_rollout_non_repeated": [],
        "teacher_entropy/student_rollout_repeated_runs": [],
        "teacher_entropy/student_rollout_non_repeated": [],
        "teacher_argmax_matches_student_rollout_repeated_runs": [],
        "teacher_argmax_matches_student_rollout_non_repeated": [],
        "teacher_nll/repeated_most_common": [],
        "teacher_nll/repeated_zero": [],
        "teacher_nll/repeated_newline": [],
        "teacher_nll/repeated_okay": [],
        "teacher_nll/repeated_the": [],
        "teacher_nll/ar_teacher_rollout": [],
    }
    skipped = 0
    repeated_run_record_count = 0
    longest_repeated_run: dict[str, Any] | None = None
    try:
        handle = output_path.open("w", encoding="utf-8")
        close_handle = True
    except Exception:
        if token_handle is not None:
            token_handle.close()
        raise
    try:
        for record in records:
            prompt = _full_prompt_tokens(record)
            continuation = _continuation_tokens(record)
            if prompt is None or not continuation:
                skipped += 1
                continue

            scored = _score_continuation(model, prompt=prompt, continuation=continuation, device=device)
            run_spans, run_membership = _repeated_run_membership(
                continuation,
                min_run_len=args.min_repeated_run_len,
            )
            repeated_mask = [run_idx is not None for run_idx in run_membership]
            non_repeated_mask = [not keep for keep in repeated_mask]
            longest_span = max(run_spans, key=lambda span: span["length"], default=None)
            if longest_span is not None:
                repeated_run_record_count += 1
                annotated_span = _annotate_span(tokenizer, longest_span, max_chars=args.max_token_text_chars)
                if (
                    annotated_span is not None
                    and (
                        longest_repeated_run is None
                        or annotated_span["length"] > int(longest_repeated_run["length"])
                    )
                ):
                    longest_repeated_run = {
                        **annotated_span,
                        "step": record.get("step"),
                        "sample_idx": record.get("sample_idx"),
                    }

            repeated_nll = _masked_mean(scored["continuation_token_nll"], repeated_mask)
            non_repeated_nll = _masked_mean(scored["continuation_token_nll"], non_repeated_mask)
            repeated_entropy = _masked_mean(scored["teacher_entropy"], repeated_mask)
            non_repeated_entropy = _masked_mean(scored["teacher_entropy"], non_repeated_mask)
            repeated_match_rate = _masked_rate(scored["teacher_argmax_matches_continuation"], repeated_mask)
            non_repeated_match_rate = _masked_rate(scored["teacher_argmax_matches_continuation"], non_repeated_mask)

            row: dict[str, Any] = {
                "step": record.get("step"),
                "sample_idx": record.get("sample_idx"),
                "prompt_token_count": len(prompt),
                "continuation_token_count": len(continuation),
                "teacher_nll/student_rollout": scored["nll_mean"],
                "teacher_argmax_matches_student_rollout": scored[
                    "teacher_argmax_matches_continuation_rate"
                ],
                "teacher_nll/student_rollout_repeated_runs": repeated_nll,
                "teacher_nll/student_rollout_non_repeated": non_repeated_nll,
                "teacher_entropy/student_rollout_repeated_runs": repeated_entropy,
                "teacher_entropy/student_rollout_non_repeated": non_repeated_entropy,
                "teacher_argmax_matches_student_rollout_repeated_runs": repeated_match_rate,
                "teacher_argmax_matches_student_rollout_non_repeated": non_repeated_match_rate,
                "repeated_run_count": len(run_spans),
                "repeated_run_token_count": sum(1 for keep in repeated_mask if keep),
                "longest_repeated_run": _annotate_span(
                    tokenizer,
                    longest_span,
                    max_chars=args.max_token_text_chars,
                ),
                "teacher_argmax_token_ids_head": scored["teacher_argmax_token_ids"][:64],
                "teacher_argmax_text_head": _decode(
                    tokenizer,
                    scored["teacher_argmax_token_ids"][:64],
                    max_chars=args.text_max_chars,
                ),
                "student_rollout_text": _decode(tokenizer, continuation, max_chars=args.text_max_chars),
                "prompt_tail_text": _decode(tokenizer, prompt[-128:], max_chars=args.text_max_chars),
            }
            summary_values["teacher_nll/student_rollout"].append(float(scored["nll_mean"]))
            summary_values["teacher_argmax_matches_student_rollout"].append(
                float(scored["teacher_argmax_matches_continuation_rate"])
            )
            if repeated_nll is not None:
                summary_values["teacher_nll/student_rollout_repeated_runs"].append(float(repeated_nll))
            if non_repeated_nll is not None:
                summary_values["teacher_nll/student_rollout_non_repeated"].append(float(non_repeated_nll))
            if repeated_entropy is not None:
                summary_values["teacher_entropy/student_rollout_repeated_runs"].append(float(repeated_entropy))
            if non_repeated_entropy is not None:
                summary_values["teacher_entropy/student_rollout_non_repeated"].append(float(non_repeated_entropy))
            if repeated_match_rate is not None:
                summary_values["teacher_argmax_matches_student_rollout_repeated_runs"].append(
                    float(repeated_match_rate)
                )
            if non_repeated_match_rate is not None:
                summary_values["teacher_argmax_matches_student_rollout_non_repeated"].append(
                    float(non_repeated_match_rate)
                )

            if token_handle is not None:
                for token_row in _token_audit_rows(
                    tokenizer,
                    record=record,
                    continuation=continuation,
                    scored=scored,
                    run_spans=run_spans,
                    run_membership=run_membership,
                    max_chars=args.max_token_text_chars,
                ):
                    token_handle.write(json.dumps(token_row, sort_keys=True) + "\n")

            probes = {
                "repeated_most_common": _most_common_continuation_probe(continuation),
                "repeated_zero": _probe_tokens(tokenizer, "0", len(continuation)),
                "repeated_newline": _probe_tokens(tokenizer, "\n", len(continuation)),
                "repeated_okay": _probe_tokens(tokenizer, "Okay", len(continuation)),
                "repeated_the": _probe_tokens(tokenizer, " the", len(continuation)),
            }
            for name, probe in probes.items():
                probe_score = _score_continuation(model, prompt=prompt, continuation=probe, device=device)
                metric_name = f"teacher_nll/{name}"
                row[metric_name] = probe_score["nll_mean"]
                row[f"{metric_name}_argmax_match"] = probe_score["teacher_argmax_matches_continuation_rate"]
                summary_values[metric_name].append(float(probe_score["nll_mean"]))

            if args.include_ar_teacher_rollout:
                ar_continuation = _greedy_teacher_continuation(
                    model,
                    prompt=prompt,
                    max_new_tokens=len(continuation),
                    device=device,
                    eos_token_id=tokenizer.eos_token_id,
                )
                ar_score = _score_continuation(model, prompt=prompt, continuation=ar_continuation, device=device)
                row["teacher_nll/ar_teacher_rollout"] = ar_score["nll_mean"]
                row["ar_teacher_rollout_text"] = _decode(
                    tokenizer,
                    ar_continuation,
                    max_chars=args.text_max_chars,
                )
                summary_values["teacher_nll/ar_teacher_rollout"].append(float(ar_score["nll_mean"]))

            handle.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        if close_handle:
            handle.close()
        if token_handle is not None:
            token_handle.close()

    summary = {
        "rollout_samples_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "per_token_output_jsonl": str(token_output_path) if token_output_path is not None else None,
        "model_path": model_path,
        "device": str(device),
        "records_read": len(records),
        "records_scored": len(records) - skipped,
        "records_skipped": skipped,
        "records_with_repeated_runs": repeated_run_record_count,
        "min_repeated_run_len": args.min_repeated_run_len,
        "longest_repeated_run": longest_repeated_run,
        **{f"{key}_mean": _mean(values) for key, values in summary_values.items() if values},
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
