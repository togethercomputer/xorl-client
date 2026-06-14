#!/usr/bin/env python3
"""Decode and score saved static SingleShot MTP checkpoints.

This is a checkpoint-native eval harness for the static reproduction path. It
does not use SGLang. For each sampled prompt it runs a saved student checkpoint
with the SingleShot MTP block mask, optionally applies confidence-adaptive
acceptance, then scores the continuation under a frozen AR teacher.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter
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

from xorl.models.auto import build_foundation_model, build_tokenizer  # noqa: E402
from xorl.mtp import build_mtp_rope_indices, make_mtp_block_mask_partial  # noqa: E402


DEFAULT_MODEL = (
    "/shared/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
)
DEFAULT_DATASET = "/shared/opd-datasets/qwen3_4b_gsm8k_aug_nl_bos_min192_8192.jsonl"
DEFAULT_CHECKPOINTS = (
    "k4=/shared/opd-coord/static-mtp-smoke/q34gsm-static-lmtp-k4ckpt-0604/student_checkpoint",
    "k8=/shared/opd-coord/static-mtp-smoke/q34gsm-static-lmtp-k8fromk4-0604/student_checkpoint",
)


def _read_rows(path: Path, *, column: str, min_tokens: int, limit_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = row.get(column)
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise ValueError(f"{path}:{line_idx + 1} missing integer-list column {column!r}")
            if len(tokens) >= min_tokens:
                row["_dataset_line_idx"] = line_idx
                rows.append(row)
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    if not rows:
        raise ValueError(f"No rows with at least {min_tokens} tokens found in {path}")
    return rows


def _sample_prompts(
    rows: list[dict[str, Any]],
    rng: random.Random,
    *,
    column: str,
    prompt_len: int,
    max_new_tokens: int,
    sample_count: int,
    random_window: bool,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    source_len = prompt_len + max_new_tokens
    for sample_idx in range(sample_count):
        row = rows[rng.randrange(len(rows))]
        tokens = [int(token) for token in row[column]]
        max_start = len(tokens) - source_len
        if max_start < 0:
            raise ValueError(f"sampled row has {len(tokens)} tokens but needs {source_len}")
        start = rng.randint(0, max_start) if random_window and max_start > 0 else 0
        window = tokens[start : start + source_len]
        samples.append(
            {
                "sample_idx": sample_idx,
                "dataset_line_idx": row.get("_dataset_line_idx"),
                "row_idx": row.get("row_idx"),
                "window_start": start,
                "window_end": start + source_len,
                "prompt": window[:prompt_len],
                "ground_truth": window[prompt_len:],
            }
        )
    return samples


def _parse_checkpoint_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).parent.name or Path(path).name
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid checkpoint spec {value!r}; expected name=/path")
        specs.append((name, path))
    return specs


def _resolve_mtp_token_id(tokenizer: Any, token_text: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is not None and token_id >= 0:
        return int(token_id)
    token = AddedToken(
        token_text,
        lstrip=False,
        rstrip=False,
        single_word=False,
        normalized=False,
        special=True,
    )
    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [token]},
            replace_extra_special_tokens=False,
        )
    except TypeError:
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is None or token_id < 0:
        raise ValueError(f"Could not resolve MTP token {token_text!r}")
    return int(token_id)


def _round_up_to_multiple(value: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _selected_logits(model: torch.nn.Module, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    selected = hidden_states[positions]
    return model.lm_head(selected)


def _decode(tokenizer: Any | None, token_ids: list[int], *, max_chars: int) -> str:
    if tokenizer is None:
        return ""
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text[:max_chars]


def _max_repeat_run(tokens: list[int]) -> int:
    best = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        best = max(best, current)
    return best


def _repeated_run_spans(tokens: list[int], *, min_run_len: int) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start = 0
    while start < len(tokens):
        stop = start + 1
        while stop < len(tokens) and tokens[stop] == tokens[start]:
            stop += 1
        if stop - start >= min_run_len:
            spans.append({"start": start, "end": stop, "length": stop - start, "token_id": int(tokens[start])})
        start = stop
    return spans


def _probe_tokens(tokenizer: Any, text: str, length: int) -> list[int]:
    ids = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise ValueError(f"Probe text {text!r} produced no tokens")
    return [ids[idx % len(ids)] for idx in range(length)]


def _most_common_probe(tokens: list[int]) -> list[int]:
    if not tokens:
        return []
    token_id, _count = Counter(tokens).most_common(1)[0]
    return [int(token_id)] * len(tokens)


def _proposal_argmax(logits: torch.Tensor, forbidden_token_ids: list[int]) -> torch.Tensor:
    if not forbidden_token_ids:
        return logits.argmax(dim=-1)
    proposal_logits = logits.float().clone()
    for token_id in forbidden_token_ids:
        if 0 <= token_id < proposal_logits.shape[-1]:
            proposal_logits[:, token_id] = -torch.inf
    return proposal_logits.argmax(dim=-1)


@torch.inference_mode()
def _mtp_step(
    model: torch.nn.Module,
    *,
    prefix: list[int],
    k_toks: int,
    mask_token_id: int,
    pad_token_id: int,
    pad_to_multiple: int | None,
    flex_block_size: int,
    dtype: str,
    device: torch.device,
    forbidden_proposal_token_ids: list[int],
) -> dict[str, Any]:
    if not prefix:
        raise ValueError("MTP decode requires a non-empty prefix")
    k_masks = k_toks - 1
    raw_len = len(prefix) + k_masks
    padded_len = _round_up_to_multiple(raw_len, pad_to_multiple)
    input_tokens = prefix + [mask_token_id] * k_masks + [pad_token_id] * (padded_len - raw_len)
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    position_ids = build_mtp_rope_indices(
        raw_len,
        len(prefix),
        k_masks,
        offset=0,
        device=device,
    )
    if padded_len > raw_len:
        position_ids = torch.cat([position_ids, torch.arange(raw_len, padded_len, device=device)])
    position_ids = position_ids.unsqueeze(0)
    pred_positions = torch.zeros((padded_len,), dtype=torch.bool, device=device)
    pred_positions[len(prefix) - 1 : len(prefix) + k_masks] = True

    attention_mask_builder = make_mtp_block_mask_partial(
        len(prefix),
        k_masks,
        padded_len,
        batch_size=1,
        offset=0,
        raw_seq_len=raw_len,
        block_size=flex_block_size,
    )
    attention_mask = attention_mask_builder(device=device)
    autocast_enabled = device.type == "cuda" and dtype != "float32"
    with torch.autocast(device_type=device.type, dtype=getattr(torch, dtype), enabled=autocast_enabled):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = _selected_logits(model, outputs.last_hidden_state[0], pred_positions)
    top1 = _proposal_argmax(logits, forbidden_proposal_token_ids)
    raw_top1 = logits.argmax(dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    top1_conf = probs.gather(dim=-1, index=top1[:, None]).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return {
        "tokens": [int(token) for token in top1.detach().cpu().tolist()],
        "raw_tokens": [int(token) for token in raw_top1.detach().cpu().tolist()],
        "top1_conf": [float(value) for value in top1_conf.detach().cpu().tolist()],
        "entropy": [float(value) for value in entropy.detach().cpu().tolist()],
    }


def _accepted_count(confidences: list[float], *, strategy: str, threshold: float | None) -> int:
    if strategy == "static":
        return len(confidences)
    if strategy != "confidence":
        raise ValueError(f"Unknown strategy {strategy!r}")
    if threshold is None:
        raise ValueError("confidence strategy requires a threshold")
    accepted = 0
    for conf in confidences:
        if conf >= threshold:
            accepted += 1
        else:
            break
    return max(1, accepted)


def _mean_by_attempt_position(steps: list[dict[str, Any]], key: str, k_toks: int) -> list[float | None]:
    means: list[float | None] = []
    for pos in range(k_toks):
        values = [
            float(step[key][pos])
            for step in steps
            if isinstance(step.get(key), list) and len(step[key]) > pos
        ]
        means.append((sum(values) / len(values)) if values else None)
    return means


def _accept_at_least_position_rates(accepted_counts: list[int], k_toks: int) -> list[float | None]:
    if not accepted_counts:
        return [None for _ in range(k_toks)]
    return [
        sum(1 for accepted in accepted_counts if accepted >= pos) / len(accepted_counts)
        for pos in range(1, k_toks + 1)
    ]


@torch.inference_mode()
def _generate_mtp(
    model: torch.nn.Module,
    *,
    prompt: list[int],
    max_new_tokens: int,
    k_toks: int,
    strategy: str,
    threshold: float | None,
    mask_token_id: int,
    pad_token_id: int,
    pad_to_multiple: int | None,
    flex_block_size: int,
    dtype: str,
    device: torch.device,
    eos_token_id: int | None,
    stop_at_eos: bool,
    forbidden_proposal_token_ids: list[int],
) -> dict[str, Any]:
    prefix = list(prompt)
    generated: list[int] = []
    accepted_counts: list[int] = []
    all_confidences: list[float] = []
    all_entropies: list[float] = []
    steps: list[dict[str, Any]] = []
    while len(generated) < max_new_tokens:
        step = _mtp_step(
            model,
            prefix=prefix,
            k_toks=k_toks,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_multiple=pad_to_multiple,
            flex_block_size=flex_block_size,
            dtype=dtype,
            device=device,
            forbidden_proposal_token_ids=forbidden_proposal_token_ids,
        )
        accept = min(
            _accepted_count(step["top1_conf"], strategy=strategy, threshold=threshold),
            max_new_tokens - len(generated),
        )
        tokens = step["tokens"][:accept]
        generated.extend(tokens)
        prefix.extend(tokens)
        accepted_counts.append(accept)
        all_confidences.extend(step["top1_conf"][:accept])
        all_entropies.extend(step["entropy"][:accept])
        steps.append(
            {
                "prefix_len_before": len(prefix) - accept,
                "accepted": int(accept),
                "attempted": int(k_toks),
                "tokens": tokens,
                "top1_conf": step["top1_conf"],
                "entropy": step["entropy"],
            }
        )
        if stop_at_eos and eos_token_id is not None and eos_token_id in tokens:
            first_eos = generated.index(eos_token_id)
            generated = generated[: first_eos + 1]
            break
    return {
        "tokens": generated,
        "steps": steps,
        "accepted_counts": accepted_counts,
        "effective_k_mean": (sum(accepted_counts) / len(accepted_counts)) if accepted_counts else 0.0,
        "accepted_conf_mean": (sum(all_confidences) / len(all_confidences)) if all_confidences else None,
        "accepted_entropy_mean": (sum(all_entropies) / len(all_entropies)) if all_entropies else None,
        "attempted_conf_pos_mean": _mean_by_attempt_position(steps, "top1_conf", k_toks),
        "attempted_entropy_pos_mean": _mean_by_attempt_position(steps, "entropy", k_toks),
        "accept_at_least_pos_rate": _accept_at_least_position_rates(accepted_counts, k_toks),
    }


@torch.inference_mode()
def _score_continuation(
    model: torch.nn.Module,
    *,
    prompt: list[int],
    continuation: list[int],
    device: torch.device,
    dtype: str,
) -> dict[str, Any]:
    if not continuation:
        return {
            "token_count": 0,
            "nll_mean": 0.0,
            "nll_sum": 0.0,
            "teacher_argmax_match_rate": 0.0,
            "teacher_entropy_mean": 0.0,
        }
    sequence = prompt + continuation
    input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(sequence[1:], dtype=torch.long, device=device)
    autocast_enabled = device.type == "cuda" and dtype != "float32"
    with torch.autocast(device_type=device.type, dtype=getattr(torch, dtype), enabled=autocast_enabled):
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = model.lm_head(outputs.last_hidden_state[0]).float()
    log_probs = F.log_softmax(logits, dim=-1)
    start = len(prompt) - 1
    stop = start + len(continuation)
    cont_log_probs = log_probs[start:stop]
    cont_labels = labels[start:stop]
    nll = -cont_log_probs.gather(dim=-1, index=cont_labels[:, None]).squeeze(-1)
    teacher_argmax = cont_log_probs.argmax(dim=-1)
    matches = teacher_argmax == cont_labels
    probs = cont_log_probs.exp()
    entropy = -(probs * cont_log_probs).sum(dim=-1)
    return {
        "token_count": int(cont_labels.numel()),
        "nll_mean": float(nll.mean().item()),
        "nll_sum": float(nll.sum().item()),
        "teacher_argmax_match_rate": float(matches.float().mean().item()),
        "teacher_entropy_mean": float(entropy.mean().item()),
        "teacher_argmax_token_ids_head": [int(x) for x in teacher_argmax[:64].detach().cpu().tolist()],
        "continuation_token_nll_head": [float(x) for x in nll[:64].detach().cpu().tolist()],
    }


@torch.inference_mode()
def _generate_ar_teacher(
    model: torch.nn.Module,
    *,
    prompt: list[int],
    max_new_tokens: int,
    device: torch.device,
    dtype: str,
    eos_token_id: int | None,
    stop_at_eos: bool,
) -> list[int]:
    generated: list[int] = []
    prefix = list(prompt)
    autocast_enabled = device.type == "cuda" and dtype != "float32"
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=getattr(torch, dtype), enabled=autocast_enabled):
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = model.lm_head(outputs.last_hidden_state[0, -1]).float()
        token = int(logits.argmax(dim=-1).detach().cpu().item())
        generated.append(token)
        prefix.append(token)
        if stop_at_eos and eos_token_id is not None and token == eos_token_id:
            break
    return generated


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"sample_idx", "dataset_line_idx", "k_toks"}:
                metrics.setdefault(key, []).append(float(value))
    return {f"{key}_mean": _mean(values) for key, values in sorted(metrics.items()) if values}


def _load_model(
    path: str,
    *,
    weights_path: str,
    args: argparse.Namespace,
    device: torch.device,
    attn_implementation: str,
) -> torch.nn.Module:
    model = build_foundation_model(
        path,
        weights_path=weights_path,
        torch_dtype=args.dtype,
        attn_implementation=attn_implementation,
        init_device=str(device),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _init_wandb(args: argparse.Namespace, output_dir: Path):
    if args.disable_wandb or not args.wandb_project:
        return None
    import wandb  # noqa: PLC0415

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_name or args.run_id,
        "tags": args.wandb_tags,
        "config": {**vars(args), "output_dir": str(output_dir)},
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    return wandb.init(**init_kwargs)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "decode_scores.jsonl"
    summary_path = output_dir / "summary.json"
    examples_path = output_dir / "examples.jsonl"

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = None if args.skip_tokenizer else build_tokenizer(args.tokenizer_path or args.model_path)
    mask_token_id = int(args.mask_token_id)
    if args.mtp_token_text:
        if tokenizer is None:
            raise ValueError("--mtp-token-text requires tokenizer loading")
        mask_token_id = _resolve_mtp_token_id(tokenizer, args.mtp_token_text)
    pad_token_id = int(args.pad_token_id if args.pad_token_id is not None else (getattr(tokenizer, "eos_token_id", 0) or 0))
    eos_token_id = None if tokenizer is None else getattr(tokenizer, "eos_token_id", None)

    rows = _read_rows(
        Path(args.dataset_jsonl),
        column=args.dataset_column,
        min_tokens=args.prompt_len + args.max_new_tokens,
        limit_rows=args.limit_rows,
    )
    samples = _sample_prompts(
        rows,
        rng,
        column=args.dataset_column,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        sample_count=args.sample_count,
        random_window=args.random_window,
    )
    if args.dry_run:
        summary = {
            "ok": True,
            "mode": "dry_run",
            "run_id": args.run_id,
            "sample_count": len(samples),
            "mask_token_id": mask_token_id,
            "samples": [
                {
                    **{key: sample[key] for key in ("sample_idx", "dataset_line_idx", "window_start", "window_end")},
                    "prompt_tail_text": _decode(tokenizer, sample["prompt"][-64:], max_chars=args.text_max_chars),
                    "ground_truth_text": _decode(tokenizer, sample["ground_truth"], max_chars=args.text_max_chars),
                }
                for sample in samples[: min(len(samples), 8)]
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    wandb_run = _init_wandb(args, output_dir)

    teacher = _load_model(
        args.model_path,
        weights_path=args.teacher_weights_path or args.model_path,
        args=args,
        device=device,
        attn_implementation=args.teacher_attn_implementation,
    )
    checkpoint_specs = _parse_checkpoint_specs(args.checkpoint)
    strategies: list[tuple[str, float | None, str]] = [("static", None, "static")]
    for threshold in args.confidence_threshold:
        strategies.append(("confidence", float(threshold), f"conf{threshold:g}"))

    all_rows: list[dict[str, Any]] = []
    t0 = time.time()
    with details_path.open("w", encoding="utf-8") as details_fh, examples_path.open("w", encoding="utf-8") as examples_fh:
        ar_cache: dict[int, list[int]] = {}
        baseline_cache: dict[tuple[int, str], dict[str, Any]] = {}
        for ckpt_name, ckpt_path in checkpoint_specs:
            print(f"loading student {ckpt_name} from {ckpt_path}", flush=True)
            student = _load_model(
                ckpt_path,
                weights_path=ckpt_path,
                args=args,
                device=device,
                attn_implementation=args.student_attn_implementation,
            )
            forbidden = [mask_token_id] if args.exclude_mtp_token_from_proposals else []
            for k_toks in args.k_values:
                for strategy, threshold, strategy_name in strategies:
                    group_rows: list[dict[str, Any]] = []
                    for sample in samples:
                        prompt = sample["prompt"]
                        generated = _generate_mtp(
                            student,
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            k_toks=int(k_toks),
                            strategy=strategy,
                            threshold=threshold,
                            mask_token_id=mask_token_id,
                            pad_token_id=pad_token_id,
                            pad_to_multiple=args.pad_to_multiple,
                            flex_block_size=args.flex_block_size,
                            dtype=args.dtype,
                            device=device,
                            eos_token_id=eos_token_id,
                            stop_at_eos=args.stop_at_eos,
                            forbidden_proposal_token_ids=forbidden,
                        )
                        continuation = generated["tokens"]
                        scored = _score_continuation(
                            teacher,
                            prompt=prompt,
                            continuation=continuation,
                            device=device,
                            dtype=args.dtype,
                        )
                        gt_score = _score_continuation(
                            teacher,
                            prompt=prompt,
                            continuation=sample["ground_truth"][: len(continuation)],
                            device=device,
                            dtype=args.dtype,
                        )
                        if sample["sample_idx"] not in ar_cache:
                            ar_cache[sample["sample_idx"]] = _generate_ar_teacher(
                                teacher,
                                prompt=prompt,
                                max_new_tokens=args.max_new_tokens,
                                device=device,
                                dtype=args.dtype,
                                eos_token_id=eos_token_id,
                                stop_at_eos=args.stop_at_eos,
                            )
                        ar_continuation = ar_cache[sample["sample_idx"]][: len(continuation)]
                        ar_score = _score_continuation(
                            teacher,
                            prompt=prompt,
                            continuation=ar_continuation,
                            device=device,
                            dtype=args.dtype,
                        )
                        probes = {
                            "repeated_most_common": _most_common_probe(continuation),
                            "repeated_zero": _probe_tokens(tokenizer, "0", len(continuation)) if tokenizer else [],
                            "repeated_newline": _probe_tokens(tokenizer, "\n", len(continuation)) if tokenizer else [],
                            "repeated_okay": _probe_tokens(tokenizer, "Okay", len(continuation)) if tokenizer else [],
                            "repeated_the": _probe_tokens(tokenizer, " the", len(continuation)) if tokenizer else [],
                        }
                        probe_scores = {}
                        for name, probe in probes.items():
                            if not probe:
                                continue
                            cache_key = (sample["sample_idx"], name)
                            if cache_key not in baseline_cache:
                                baseline_cache[cache_key] = _score_continuation(
                                    teacher,
                                    prompt=prompt,
                                    continuation=probe,
                                    device=device,
                                    dtype=args.dtype,
                                )
                            probe_scores[name] = baseline_cache[cache_key]

                        repeated_spans = _repeated_run_spans(
                            continuation,
                            min_run_len=args.min_repeated_run_len,
                        )
                        longest = max(repeated_spans, key=lambda span: span["length"], default=None)
                        row = {
                            "checkpoint": ckpt_name,
                            "strategy": strategy_name,
                            "k_toks": int(k_toks),
                            "sample_idx": int(sample["sample_idx"]),
                            "dataset_line_idx": sample.get("dataset_line_idx"),
                            "window_start": sample.get("window_start"),
                            "continuation_token_count": len(continuation),
                            "effective_k_mean": float(generated["effective_k_mean"]),
                            "accepted_conf_mean": generated["accepted_conf_mean"],
                            "accepted_entropy_mean": generated["accepted_entropy_mean"],
                            "teacher_nll/student": scored["nll_mean"],
                            "teacher_argmax_match/student": scored["teacher_argmax_match_rate"],
                            "teacher_entropy/student": scored["teacher_entropy_mean"],
                            "teacher_nll/ground_truth": gt_score["nll_mean"],
                            "teacher_nll/ar_teacher": ar_score["nll_mean"],
                            "student_max_repeat_run": _max_repeat_run(continuation),
                            "student_repeated_run_count": len(repeated_spans),
                            "student_longest_repeated_run_len": int(longest["length"]) if longest else 0,
                        }
                        for pos, value in enumerate(generated["attempted_conf_pos_mean"], start=1):
                            if value is not None:
                                row[f"attempt_conf_pos{pos}"] = float(value)
                        for pos, value in enumerate(generated["attempted_entropy_pos_mean"], start=1):
                            if value is not None:
                                row[f"attempt_entropy_pos{pos}"] = float(value)
                        for pos, value in enumerate(generated["accept_at_least_pos_rate"], start=1):
                            if value is not None:
                                row[f"accept_at_least_pos{pos}_rate"] = float(value)
                        for name, probe_score in probe_scores.items():
                            row[f"teacher_nll/{name}"] = probe_score["nll_mean"]
                        details_fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                        details_fh.flush()
                        group_rows.append(row)
                        all_rows.append(row)

                        if sample["sample_idx"] < args.example_samples_per_group:
                            examples_fh.write(
                                json.dumps(
                                    {
                                        **row,
                                        "prompt_tail_text": _decode(tokenizer, prompt[-96:], max_chars=args.text_max_chars),
                                        "student_text": _decode(tokenizer, continuation, max_chars=args.text_max_chars),
                                        "ground_truth_text": _decode(
                                            tokenizer,
                                            sample["ground_truth"][: len(continuation)],
                                            max_chars=args.text_max_chars,
                                        ),
                                        "ar_teacher_text": _decode(tokenizer, ar_continuation, max_chars=args.text_max_chars),
                                        "accepted_counts": generated["accepted_counts"],
                                        "steps_head": generated["steps"][:8],
                                    },
                                    ensure_ascii=True,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            examples_fh.flush()
                    summary = {
                        "checkpoint": ckpt_name,
                        "strategy": strategy_name,
                        "k_toks": int(k_toks),
                        "sample_count": len(group_rows),
                        **_summarize(group_rows),
                    }
                    if wandb_run is not None:
                        prefix = f"decode/{ckpt_name}/k{k_toks}/{strategy_name}"
                        wandb_run.log({f"{prefix}/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
                    print(json.dumps({"group": summary}, ensure_ascii=True), flush=True)
            del student
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        key = f"{row['checkpoint']}/k{row['k_toks']}/{row['strategy']}"
        grouped.setdefault(key, []).append(row)
    summary = {
        "ok": True,
        "mode": "decode_eval",
        "run_id": args.run_id,
        "model_path": args.model_path,
        "dataset_jsonl": args.dataset_jsonl,
        "sample_count": args.sample_count,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "mask_token_id": mask_token_id,
        "elapsed_s": time.time() - t0,
        "details_jsonl": str(details_path),
        "examples_jsonl": str(examples_path),
        "groups": {key: _summarize(rows) | {"record_count": len(rows)} for key, rows in sorted(grouped.items())},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    if wandb_run is not None:
        final_metrics = {}
        for group_name, group_summary in summary["groups"].items():
            safe_group = group_name.replace("/", "_")
            for key, value in group_summary.items():
                if isinstance(value, (int, float)):
                    final_metrics[f"summary/{safe_group}/{key}"] = value
        wandb_run.log(final_metrics)
        wandb_run.finish()
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"static-mtp-decode-eval-{int(time.time())}")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--teacher-weights-path", default=None)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--dataset-jsonl", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-column", default="input_ids")
    parser.add_argument("--output-dir", default="/shared/opd-coord/static-mtp-decode-eval")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--random-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--confidence-threshold", type=float, nargs="*", default=[0.9, 0.95])
    parser.add_argument("--mtp-token-text", default="<MTP>")
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--pad-to-multiple", type=int, default=128)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--min-repeated-run-len", type=int, default=3)
    parser.add_argument("--example-samples-per-group", type=int, default=2)
    parser.add_argument("--text-max-chars", type=int, default=1000)
    parser.add_argument("--exclude-mtp-token-from-proposals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--student-attn-implementation", default="flex_attention")
    parser.add_argument("--teacher-attn-implementation", default="flash_attention_3")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = list(DEFAULT_CHECKPOINTS)
    return args


if __name__ == "__main__":
    run(parse_args())
