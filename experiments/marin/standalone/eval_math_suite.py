from __future__ import annotations

import argparse
import json
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from xorl_client import SamplingClient, types

from experiments.marin.standalone.dataset import load_examples
from experiments.marin.standalone.prompts import (
    encode_prompt_prefix,
    render_evalchemy_math_prompt,
    render_gsm8k_cot_prompt,
    render_gsm8k_lm_eval_prompt,
    render_gsm8k_prompt,
)
from experiments.marin.standalone.tasks.verifier import grade_boxed_answer


_GSM8K_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Marin #6279 checkpoints with the rollout grader.")
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="math500", choices=["math500", "aime24", "gsm8k", "rlvr_math_7500"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--repeat-seed-base", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-generate-tokens", type=int, default=3584)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--grader", choices=["boxed", "gsm8k-flex", "auto"], default="auto")
    parser.add_argument("--prompt-style", choices=["forced-thinking", "chat"], default="chat")
    parser.add_argument("--boxed-instruction", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--math-prompt-template", choices=["raw", "evalchemy"], default="raw")
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument("--gsm8k-cot-fewshot", type=int, default=0)
    parser.add_argument("--gsm8k-lm-eval-fewshot", type=int, default=0)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def _last_number(value: str) -> str | None:
    matches = _GSM8K_FLEX_RE.findall(value)
    if not matches:
        return None
    match = next((candidate for candidate in matches[-1] if candidate), "")
    normalized = match.replace("$", "").replace(",", "").strip().rstrip(".")
    return normalized.lstrip("+") or None


def _numbers_equal(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def _grade(task: str, grader: str, completion: str, gold_answer: str) -> bool:
    effective = "gsm8k-flex" if grader == "auto" and task == "gsm8k" else grader
    if effective == "gsm8k-flex":
        predicted = _last_number(completion)
        gold = _last_number(gold_answer)
        return predicted is not None and gold is not None and _numbers_equal(predicted, gold)
    return grade_boxed_answer(completion, gold_answer).correct


def _should_add_boxed_instruction(task: str, boxed_instruction: str) -> bool:
    if boxed_instruction == "always":
        return True
    if boxed_instruction == "never":
        return False
    return task != "gsm8k"


def _write_jsonl(path: str | None, payload: dict) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _apply_chat_template_override(tokenizer: object, chat_template_file: str | None) -> None:
    if chat_template_file is None:
        return
    template = Path(chat_template_file).read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"chat template file is empty: {chat_template_file}")
    setattr(tokenizer, "chat_template", template)


def _sampling_seed_for_repeat(seed: int, repeat_index: int, repeat_seed_base: int | None) -> int:
    base = seed if repeat_seed_base is None else repeat_seed_base
    return base + repeat_index


def _load_gsm8k_fewshot_examples(num_fewshot: int) -> list[tuple[str, str]]:
    if num_fewshot <= 0:
        return []
    dataset = load_dataset("openai/gsm8k", name="main", split="train")
    examples = []
    for row in dataset.select(range(num_fewshot)):
        examples.append((row["question"], row["answer"]))
    return examples


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive")
    if args.gsm8k_cot_fewshot > 0 and args.gsm8k_lm_eval_fewshot > 0:
        raise ValueError("--gsm8k-cot-fewshot and --gsm8k-lm-eval-fewshot are mutually exclusive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    _apply_chat_template_override(tokenizer, args.chat_template_file)
    sampling_client = SamplingClient(base_url=args.inference_url, model=args.model)
    examples = list(load_examples(args.task, limit=args.limit))
    if not examples:
        raise RuntimeError(f"No examples loaded for task={args.task}")
    gsm8k_lm_eval_fewshots = _load_gsm8k_fewshot_examples(args.gsm8k_lm_eval_fewshot) if args.task == "gsm8k" else []
    repeat_seed_base = args.seed if args.repeat_seed_base is None else args.repeat_seed_base

    correct_values: list[float] = []
    skipped_long = 0
    submitted = 0
    pending = []
    iterator = iter(
        (repeat_index, index, example)
        for repeat_index in range(args.repeat_count)
        for index, example in enumerate(examples)
    )

    def submit_until_full() -> None:
        nonlocal skipped_long, submitted
        while len(pending) < args.concurrency:
            try:
                repeat_index, index, example = next(iterator)
            except StopIteration:
                return
            submitted += 1
            sampling_seed = _sampling_seed_for_repeat(args.seed, repeat_index, args.repeat_seed_base)
            prompt = example.prompt
            if args.task == "gsm8k":
                if gsm8k_lm_eval_fewshots:
                    prompt = render_gsm8k_lm_eval_prompt(example.prompt, gsm8k_lm_eval_fewshots)
                elif args.gsm8k_cot_fewshot > 0:
                    prompt = render_gsm8k_cot_prompt(example.prompt, num_fewshot=args.gsm8k_cot_fewshot)
                else:
                    prompt = render_gsm8k_prompt(example.prompt)
            elif args.math_prompt_template == "evalchemy":
                prompt = render_evalchemy_math_prompt(example.prompt)
            prefix = encode_prompt_prefix(
                tokenizer,
                prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                prompt_style=args.prompt_style,
                add_boxed_instruction=_should_add_boxed_instruction(args.task, args.boxed_instruction),
            )
            if prefix.skipped_for_length:
                skipped_long += 1
                continue
            sampling_params = types.SamplingParams(
                max_tokens=args.max_generate_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                sampling_seed=sampling_seed,
                stop=args.stop,
            )
            future = sampling_client.sample(
                types.ModelInput.from_ints(prefix.token_ids),
                sampling_params=sampling_params,
                num_samples=1,
                return_logprobs=True,
            )
            pending.append((repeat_index, sampling_seed, index, example, future))

    submit_until_full()
    while pending:
        repeat_index, sampling_seed, index, example, future = pending.pop(0)
        response = future.result()
        completion = response.text
        correct = _grade(args.task, args.grader, completion, example.gold_answer)
        correct_values.append(1.0 if correct else 0.0)
        _write_jsonl(
            args.output_jsonl,
            {
                "index": index,
                "example_id": example.example_id,
                "task": args.task,
                "seed": args.seed,
                "repeat_index": repeat_index,
                "repeat_count": args.repeat_count,
                "sampling_seed": sampling_seed,
                "correct": correct,
                "gold_answer": example.gold_answer,
                "completion": completion,
                "tokens": len(response.tokens),
            },
        )
        if args.progress_interval > 0 and len(correct_values) % args.progress_interval == 0:
            progress = {
                "event": "progress",
                "task": args.task,
                "seed": args.seed,
                "repeat_count": args.repeat_count,
                "repeat_seed_base": repeat_seed_base,
                "submitted": submitted,
                "num_scored": len(correct_values),
                "skipped_long": skipped_long,
                "accuracy_percent_so_far": 100.0 * statistics.fmean(correct_values),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
        submit_until_full()

    if not correct_values:
        raise RuntimeError(f"No scored examples; skipped_long={skipped_long}")
    summary = {
        "task": args.task,
        "seed": args.seed,
        "repeat_count": args.repeat_count,
        "repeat_seed_base": repeat_seed_base,
        "num_scored": len(correct_values),
        "skipped_long": skipped_long,
        "accuracy_percent": 100.0 * statistics.fmean(correct_values),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
