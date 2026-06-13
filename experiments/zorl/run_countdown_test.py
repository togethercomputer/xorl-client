"""
End-to-end Countdown / 24-puzzle verifiable-reward test.

Pivot from letter counting (saturated at 7/8 cold) to a wider-gap task:
"combine N numbers with +, -, *, /, parentheses, each used exactly once,
to reach a target". 24-puzzle variant for tractability. Cold baseline on
Qwen3-30B-A3B-Instruct-2507 expected ~30-50% — broad headroom for ZORL vs
GRPO convergence shape comparison.

Reward: extract the first plausible arithmetic expression from the
generated text; 1.0 if it (a) uses each input number exactly once,
(b) only contains digits, +, -, *, /, parentheses, whitespace, and
(c) evaluates to the target. 0.0 otherwise. Pure binary signal via the
existing rollout-reward plumbing:
  --zorl-reward-mode rollout
  --zorl-rollout-prefix-weight 0
  --zorl-rollout-char-weight 0
  --zorl-rollout-exact-bonus 1.0

Trainer modes (inherited from run_password_test.py):
- `gradient`: conventional forward_backward + optim_step SFT (GRPO-style)
- `zorl`: ZORL ES generations scored by binary verifiable rollout reward
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import re
import sys
import time
import traceback
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer


DEFAULT_TRAIN_MODEL_ID = "countdown-test"

# 24-puzzle variant of Countdown: 4 distinct integers from 1-12, target=24,
# each number used exactly once, ops +, -, *, /, parentheses. Each tuple is
# (numbers, target, sample_solution_string). The sample solution is used as
# the SFT target (`code`) and as the cold-baseline display answer; the
# verifiable reward accepts ANY expression satisfying the constraints, not
# just this canonical one.
#
# We hand-seed the first 8 puzzles for backward compat with existing
# eval scripts and run YAMLs, then auto-fill from an enumerator below so
# the pool can be expanded without risk of shipping unsolvable inputs.
_SEED_COUNTDOWN_PUZZLES = [
    ([3, 4, 6, 9], 24, "(9 - 6 + 3) * 4"),
    ([2, 4, 6, 8], 24, "4 * 8 - 6 - 2"),
    ([1, 6, 7, 9], 24, "(1 + 7) * (9 - 6)"),
    ([3, 5, 7, 11], 24, "(11 - 5) * (7 - 3)"),
    ([2, 3, 8, 10], 24, "2 * 3 + 8 + 10"),
    ([1, 3, 7, 12], 24, "12 * (7 - 1) / 3"),
    ([2, 6, 8, 11], 24, "(11 - 8) * (2 + 6)"),
    ([1, 2, 7, 8], 24, "(7 - 1) * 8 / 2"),
]

# How many puzzles total to expose in COUNTDOWN_PUZZLES. The seed list is
# kept at the head, and the rest are auto-enumerated below.
_COUNTDOWN_POOL_TARGET_SIZE = 32


def _format_expr_str(expr_tree):
    """Render a parenthesised expression tree as a Python-evaluable string."""
    op, left, right = expr_tree
    if isinstance(left, tuple):
        left_s = _format_expr_str(left)
    else:
        left_s = str(left)
    if isinstance(right, tuple):
        right_s = _format_expr_str(right)
    else:
        right_s = str(right)
    return f"({left_s} {op} {right_s})"


def _eval_expr_tree(expr_tree):
    """Evaluate an expression tree built from ints + (op, l, r) tuples. Returns
    None on division by zero (so callers can treat it as no-solution)."""
    if not isinstance(expr_tree, tuple):
        return float(expr_tree)
    op, left, right = expr_tree
    lv = _eval_expr_tree(left)
    rv = _eval_expr_tree(right)
    if lv is None or rv is None:
        return None
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "/":
        if abs(rv) < 1e-12:
            return None
        return lv / rv
    raise ValueError(f"Unknown operator {op!r}")


def _enumerate_countdown_solutions(numbers, target):
    """Yield (expr_tree, solution_string) for every way to combine `numbers`
    via +, -, *, / and parentheses so the result equals `target`. Enumerates
    permutations of the inputs and the 5 binary-tree shapes on 4 leaves."""
    ops = ("+", "-", "*", "/")
    tol = 1e-6
    seen_strings = set()
    for perm in itertools.permutations(numbers):
        a, b, c, d = perm
        for o1 in ops:
            for o2 in ops:
                for o3 in ops:
                    # 5 binary tree shapes on a 4-leaf expression:
                    candidates = [
                        ((o3, (o2, (o1, a, b), c), d)),
                        ((o3, (o1, a, (o2, b, c)), d)),
                        ((o2, (o1, a, b), (o3, c, d))),
                        ((o1, a, (o3, (o2, b, c), d))),
                        ((o1, a, (o2, b, (o3, c, d)))),
                    ]
                    for tree in candidates:
                        val = _eval_expr_tree(tree)
                        if val is None or abs(val - float(target)) > tol:
                            continue
                        s = _format_expr_str(tree)
                        if s in seen_strings:
                            continue
                        seen_strings.add(s)
                        yield tree, s


def _first_countdown_solution(numbers, target):
    for _tree, soln in _enumerate_countdown_solutions(numbers, target):
        return soln
    return None


def _strip_outer_parens(expr_str):
    """Cosmetic: drop a single redundant outer (...) wrapping the whole expr."""
    if expr_str.startswith("(") and expr_str.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr_str):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(expr_str) - 1:
                    return expr_str
        return expr_str[1:-1]
    return expr_str


def _enumerate_extra_countdown_puzzles(seed_puzzles, pool_size, target=24, max_n=12):
    """Pick distinct-int 4-subsets of [1, max_n] that have a valid solution,
    drawn from a deterministically-shuffled candidate order so the resulting
    pool spans easy/medium/hard puzzles instead of stacking the easy
    lexicographic prefix (1,2,3,*) at the front. Distinct integers (no
    repeats) avoids trivial multisets like (1,1,1,8) that the model solves
    instantly and that contribute nothing to the gradient signal."""
    needed = pool_size - len(seed_puzzles)
    if needed <= 0:
        return []
    seed_keys = {tuple(sorted(nums)) for nums, _, _ in seed_puzzles}
    candidates = [
        multiset
        for multiset in itertools.combinations(range(1, max_n + 1), 4)
        if multiset not in seed_keys
    ]
    rng = random.Random(20260520)
    rng.shuffle(candidates)
    extras = []
    for multiset in candidates:
        soln = _first_countdown_solution(list(multiset), target)
        if soln is None:
            continue
        extras.append((list(multiset), target, _strip_outer_parens(soln)))
        if len(extras) >= needed:
            break
    return extras


COUNTDOWN_PUZZLES = list(_SEED_COUNTDOWN_PUZZLES) + _enumerate_extra_countdown_puzzles(
    _SEED_COUNTDOWN_PUZZLES, _COUNTDOWN_POOL_TARGET_SIZE
)

CODES = {f"q_{i:02d}": solution for i, (_nums, _target, solution) in enumerate(COUNTDOWN_PUZZLES)}

# Per-question metadata indexed by the same key as CODES.
QUESTION_META = {
    f"q_{i:02d}": {"numbers": list(numbers), "target": target, "solution": solution}
    for i, (numbers, target, solution) in enumerate(COUNTDOWN_PUZZLES)
}

SYSTEM_PROMPT = (
    "You solve arithmetic puzzles. Combine all of the given numbers using +, -, *, /, "
    "and parentheses so the expression equals the target. Use each number exactly once. "
    "Reply with only the expression — no words, no thinking, no equality sign."
)
# Alternate system prompt used when --zorl-enable-thinking is set. Qwen3's
# native <think> mode handles the reasoning trace; we just need the model to
# emit the final expression on its own line so _extract_expression can pull
# it out after stripping the think block.
SYSTEM_PROMPT_THINKING = (
    "You solve arithmetic puzzles. Combine all of the given numbers using +, -, *, /, "
    "and parentheses so the expression equals the target. Use each number exactly once. "
    "After your reasoning, output ONLY the final arithmetic expression on the last line "
    "— no words, no equality sign, no surrounding markdown."
)


def _resolve_system_prompt(enable_thinking: bool) -> str:
    return SYSTEM_PROMPT_THINKING if enable_thinking else SYSTEM_PROMPT


def _question_text(meta):
    nums_str = ", ".join(str(n) for n in meta["numbers"])
    return (
        f"Combine the numbers {nums_str} using +, -, *, /, and parentheses "
        f"so the expression equals {meta['target']}. Use each number exactly once."
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_training_data(tokenizer, enable_thinking: bool = False):
    data = []
    system_prompt = _resolve_system_prompt(enable_thinking)
    for question_key, code in CODES.items():
        meta = QUESTION_META[question_key]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _question_text(meta)},
            {"role": "assistant", "content": code},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        assert len(labels) == len(full_ids)
        data.append({"model_input": {"input_ids": full_ids}, "loss_fn_inputs": {"labels": labels}})
    return data


def build_reward_data(tokenizer, enable_thinking: bool = False):
    reward_examples = []
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    system_prompt = _resolve_system_prompt(enable_thinking)
    for question_key, code in CODES.items():
        meta = QUESTION_META[question_key]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _question_text(meta)},
            {"role": "assistant", "content": code},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        answer_ids = full_ids[len(prompt_ids) :]
        answer_token_ids = [token_id for token_id in answer_ids if token_id not in special_token_ids]
        reward_examples.append(
            {
                "project": question_key,
                "code": code,
                "target": meta["target"],
                "numbers": list(meta["numbers"]),
                "solution": meta["solution"],
                "prompt_messages": messages[:2],
                "prompt_ids": prompt_ids,
                "full_ids": full_ids,
                "answer_ids": answer_ids,
                "answer_token_ids": answer_token_ids,
            }
        )
    return reward_examples


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _raise_on_failed_future(result, context):
    if result.get("type") == "request_failed":
        raise RuntimeError(f"{context} failed: {result.get('error', result)}")
    if result.get("error"):
        raise RuntimeError(f"{context} failed: {result['error']}")
    return result


def wait_for_future(train_url, request_id, timeout=3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.post(f"{train_url}/api/v1/retrieve_future", json={"request_id": request_id}, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        if result.get("type") == "try_again":
            time.sleep(0.5)
            continue
        return result
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def wait_for_training_service(train_url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{train_url}/health", timeout=5)
            resp.raise_for_status()
            if resp.json().get("engine_running"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def wait_for_inference_service(infer_url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for endpoint in ("/health", "/model_info", "/v1/models"):
            try:
                resp = requests.get(f"{infer_url}{endpoint}", timeout=5)
                resp.raise_for_status()
                return True
            except Exception:
                pass
        time.sleep(3)
    return False


def create_model(train_url, model_name, args):
    payload = {"model_id": args.train_model_id, "base_model": model_name}
    if args.trainer_mode == "zorl":
        payload["lora_config"] = {
            "rank": args.lora_rank,
            "lora_rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "lora_alpha": args.lora_alpha,
        }
        payload["optimizer_config"] = {
            "type": args.zorl_optimizer,
            "learning_rate": args.lr,
            "weight_decay": 0.0,
            "optimizer_dtype": "fp32",
        }
        optimizer_kwargs = {}
        if args.zorl_optimizer == "sgd":
            if args.zorl_sgd_momentum > 0.0:
                optimizer_kwargs["momentum"] = args.zorl_sgd_momentum
            if args.zorl_sgd_nesterov:
                optimizer_kwargs["nesterov"] = True
        if optimizer_kwargs:
            payload["optimizer_config"]["optimizer_kwargs"] = optimizer_kwargs
        payload["zorl_config"] = {
            "enabled": True,
            "b_sigma": args.zorl_b_sigma,
            "num_perturbation_pairs": args.zorl_num_pairs,
            "a_refresh_interval": args.zorl_refresh_interval,
            "antithetic_sampling": True,
            "a_init": "gaussian_jl",
            "seed": args.zorl_seed,
            "perturbation_mode": args.zorl_perturbation_mode,
        }

    resp = requests.post(f"{train_url}/api/v1/create_model", json=payload, timeout=30)
    resp.raise_for_status()
    future = resp.json()
    result = wait_for_future(train_url, future["request_id"])
    return _raise_on_failed_future(result, "create_model")


def add_endpoints(train_url, infer_urls):
    for url in infer_urls:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port
        resp = requests.post(
            f"{train_url}/add_inference_endpoint",
            json={"host": host, "port": port, "worker_port": port},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        endpoint = (result or {}).get("endpoint") or {}
        si = endpoint.get("server_info") or {}
        print(
            f"    Endpoint {host}:{port}: quantization={si.get('quantization')}, "
            f"tp_size={si.get('tp_size')}, model={si.get('model_path')}"
        )


def train_step(train_url, model_id, data, lr):
    fb = requests.post(
        f"{train_url}/api/v1/forward_backward",
        json={"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": "causallm_loss"}},
        timeout=30,
    )
    fb.raise_for_status()
    fb_result = _raise_on_failed_future(wait_for_future(train_url, fb.json()["request_id"]), "forward_backward")
    loss = fb_result.get("metrics", {}).get("loss:mean", "N/A")
    opt = requests.post(
        f"{train_url}/api/v1/optim_step",
        json={"model_id": model_id, "learning_rate": lr, "gradient_clip": 1.0},
        timeout=30,
    )
    opt.raise_for_status()
    opt_result = _raise_on_failed_future(wait_for_future(train_url, opt.json()["request_id"]), "optim_step")
    grad_norm = opt_result.get("metrics", {}).get("grad_norm", "N/A")
    return loss, grad_norm


# ---------------------------------------------------------------------------
# GRPO: group-relative policy optimization
# ---------------------------------------------------------------------------


def _sglang_sample_with_logprobs(infer_url, *, input_ids, num_samples, temperature, max_new_tokens, lora_path=None):
    """Sample N rollouts from SGLang for a single prompt, returning per-rollout
    (sampled_tokens, sampled_logprobs) pairs. Uses SGLang's `n` sampling param
    so all N samples come back in one HTTP request.
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            "n": int(num_samples),
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    resp = requests.post(
        f"{infer_url}/generate",
        json=payload,
        headers={"Connection": "close"},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    # SGLang flattens n>1 results across requests when given a single input.
    # When input_ids is a single list (not a list-of-lists), the response is
    # either a list of N dicts or a single dict; normalize to list.
    if isinstance(data, dict):
        data = [data]
    samples = []
    for entry in data:
        meta = entry.get("meta_info", {}) or {}
        output_ids = list(entry.get("output_ids", []))
        # output_token_logprobs is a list of [logprob, token_id, decoded_text]
        # tuples for the sampled response. We only need the logprobs (in same
        # order as output_ids) for the importance_sampling loss.
        out_lp_entries = meta.get("output_token_logprobs") or []
        out_logprobs = []
        for item in out_lp_entries:
            if item is None:
                out_logprobs.append(_LOGPROB_NONE_FLOOR)
                continue
            value = item[0] if isinstance(item, (list, tuple)) else item
            out_logprobs.append(_LOGPROB_NONE_FLOOR if value is None else float(value))
        samples.append(
            {
                "output_ids": output_ids,
                "output_logprobs": out_logprobs,
                "text": entry.get("text", ""),
            }
        )
    return samples


def _grpo_score_rollout(generated_text, reward_example):
    """Binary reward for GRPO: 1.0 if the rollout's expression evaluates to the
    target with each input number used exactly once, 0.0 otherwise. Reuses the
    existing Countdown evaluator helpers.
    """
    expr = _extract_expression(generated_text)
    if expr is None:
        return 0.0
    if not _expression_uses_numbers_exactly_once(expr, reward_example.get("numbers") or []):
        return 0.0
    val = _safe_eval_expr(expr)
    target = reward_example.get("target")
    if val is None or target is None:
        return 0.0
    return 1.0 if abs(val - float(target)) < 1e-6 else 0.0


def _build_grpo_datum(*, prompt_ids, sampled_tokens, sampled_logprobs, advantage):
    """Pack a single (prompt, rollout) pair into the format the trainer's
    importance_sampling loss expects.

    Following xorl-client/examples/qwen_gsm8k_rl_loop.py:466-488:
      input_tokens = (prompt + response)[:-1]
      target_tokens = [-100]*ob_len + (prompt + response)[ob_len+1:]
      logprobs (old, sampled-time) = [0.0]*ob_len + sampled_logprobs
      advantages = [0.0]*ob_len + [advantage]*response_len
    where ob_len = len(prompt_ids) - 1 (predict from position ob_len onward).
    """
    prompt_ids = list(prompt_ids)
    sampled_tokens = list(sampled_tokens)
    sampled_logprobs = list(sampled_logprobs)
    if len(sampled_tokens) == 0:
        return None
    # Align logprob count to token count (SGLang sometimes returns one fewer).
    if len(sampled_logprobs) < len(sampled_tokens):
        sampled_logprobs = sampled_logprobs + [_LOGPROB_NONE_FLOOR] * (len(sampled_tokens) - len(sampled_logprobs))
    elif len(sampled_logprobs) > len(sampled_tokens):
        sampled_logprobs = sampled_logprobs[: len(sampled_tokens)]

    full = prompt_ids + sampled_tokens
    ob_len = len(prompt_ids) - 1
    input_tokens = full[:-1]
    target_tokens = [-100] * ob_len + full[ob_len + 1 :]
    all_logprobs = [0.0] * ob_len + sampled_logprobs
    response_len = len(input_tokens) - ob_len
    all_advantages = [0.0] * ob_len + [float(advantage)] * response_len
    assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages), (
        f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, "
        f"logprobs={len(all_logprobs)}, advantages={len(all_advantages)}, "
        f"ob_len={ob_len}, prompt_len={len(prompt_ids)}, response_len={len(sampled_tokens)}"
    )
    return {
        "model_input": {"input_ids": [int(t) for t in input_tokens]},
        "loss_fn_inputs": {
            "target_tokens": [int(t) for t in target_tokens],
            "logprobs": [float(x) for x in all_logprobs],
            "advantages": [float(x) for x in all_advantages],
        },
    }


def grpo_train_step(
    train_url,
    model_id,
    *,
    infer_urls,
    reward_data,
    args,
    lr,
):
    """One GRPO step: sample K rollouts per puzzle, compute group-relative
    advantage on binary correctness, and run importance_sampling forward/backward
    via the trainer. Returns aggregate metrics.
    """
    group_size = int(args.grpo_group_size)
    temperature = float(args.grpo_temperature)
    max_new_tokens = int(args.grpo_max_new_tokens)

    datums = []
    rollout_rewards_all = []
    skipped_zero_advantage = 0
    skipped_no_samples = 0
    exact_match_count = 0
    total_rollouts = 0

    # Round-robin across inference replicas to spread load.
    for ex_idx, example in enumerate(reward_data):
        infer_url = infer_urls[ex_idx % len(infer_urls)]
        try:
            samples = _sglang_sample_with_logprobs(
                infer_url,
                input_ids=example["prompt_ids"],
                num_samples=group_size,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(f"      GRPO sample failed for {example['project']}: {type(exc).__name__}: {exc}")
            skipped_no_samples += 1
            continue

        if not samples:
            skipped_no_samples += 1
            continue

        group_rewards = [_grpo_score_rollout(sample.get("text", ""), example) for sample in samples]
        rollout_rewards_all.extend(group_rewards)
        exact_match_count += int(sum(group_rewards))
        total_rollouts += len(group_rewards)

        mean_reward = sum(group_rewards) / max(len(group_rewards), 1)
        advantages = [r - mean_reward for r in group_rewards]
        if args.grpo_skip_zero_advantage and all(abs(a) < 1e-9 for a in advantages):
            skipped_zero_advantage += 1
            continue

        for sample, advantage in zip(samples, advantages):
            datum = _build_grpo_datum(
                prompt_ids=example["prompt_ids"],
                sampled_tokens=sample["output_ids"],
                sampled_logprobs=sample["output_logprobs"],
                advantage=advantage,
            )
            if datum is not None:
                datums.append(datum)

    metrics = {
        "rollouts": total_rollouts,
        "exact_match": exact_match_count,
        "exact_match_rate": (exact_match_count / total_rollouts) if total_rollouts else 0.0,
        "groups_skipped_no_samples": skipped_no_samples,
        "groups_skipped_zero_advantage": skipped_zero_advantage,
        "datums": len(datums),
    }

    if not datums:
        # Nothing to backprop on — every group hit the same reward (or every
        # group skipped). Skip the optim step but keep stats.
        metrics.update({"loss": "skipped_no_datums", "grad_norm": "skipped_no_datums"})
        return metrics

    fb = requests.post(
        f"{train_url}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": datums,
                "loss_fn": "importance_sampling",
            },
        },
        timeout=120,
    )
    fb.raise_for_status()
    fb_result = _raise_on_failed_future(wait_for_future(train_url, fb.json()["request_id"]), "forward_backward")
    fb_metrics = fb_result.get("metrics", {}) or {}
    loss = fb_metrics.get("loss:mean", "N/A")

    opt = requests.post(
        f"{train_url}/api/v1/optim_step",
        json={"model_id": model_id, "learning_rate": float(lr), "gradient_clip": 1.0},
        timeout=60,
    )
    opt.raise_for_status()
    opt_result = _raise_on_failed_future(wait_for_future(train_url, opt.json()["request_id"]), "optim_step")
    grad_norm = (opt_result.get("metrics", {}) or {}).get("grad_norm", "N/A")

    metrics.update({"loss": loss, "grad_norm": grad_norm})
    for k, v in fb_metrics.items():
        if k.startswith("loss") or "kl" in k.lower():
            metrics[f"fb.{k}"] = v
    return metrics


def start_zorl_generation(train_url, model_id, *, preload_sampling=True, candidate_backend="filesystem"):
    resp = requests.post(
        f"{train_url}/api/v1/zorl/start_generation",
        json={
            "model_id": model_id,
            "preload_sampling": preload_sampling,
            "candidate_backend": candidate_backend,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(wait_for_future(train_url, resp.json()["request_id"]), "start_zorl_generation")


def apply_zorl_rewards(train_url, model_id, generation_id, candidate_rewards, lr):
    resp = requests.post(
        f"{train_url}/api/v1/zorl/apply_rewards",
        json={
            "model_id": model_id,
            "generation_id": generation_id,
            "candidate_rewards": candidate_rewards,
            "learning_rate": lr,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(wait_for_future(train_url, resp.json()["request_id"]), "apply_zorl_rewards")


def abort_zorl_generation(train_url, model_id, generation_id):
    resp = requests.post(
        f"{train_url}/api/v1/zorl/abort_generation",
        json={"model_id": model_id, "generation_id": generation_id},
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(wait_for_future(train_url, resp.json()["request_id"]), "abort_zorl_generation")


def _default_zorl_native_session_id(train_model_id):
    safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(train_model_id)).strip("-") or "zorl"
    return f"{safe_model_id}-{time.time_ns()}"


def _raise_for_sglang_zorl_status(resp, *, path, url):
    if resp.status_code < 400:
        return
    body = resp.text.strip()
    if len(body) > 1200:
        body = f"{body[:1200]}..."
    detail = body or getattr(resp, "reason", "")
    raise RuntimeError(f"{path} failed on {url} with HTTP {resp.status_code}: {detail}")


def _post_sglang_zorl_all(infer_urls, path, payload, *, timeout=300):
    urls = infer_urls if isinstance(infer_urls, list) else [infer_urls]
    if len(urls) == 1:
        url = urls[0]
        resp = requests.post(
            f"{url}{path}",
            json=payload,
            headers={"Connection": "close"},
            timeout=timeout,
        )
        _raise_for_sglang_zorl_status(resp, path=path, url=url)
        result = resp.json()
        if not result.get("success", False):
            raise RuntimeError(f"{path} failed on {url}: {result.get('error_message') or result}")
        return [result]

    def post_one(url):
        resp = requests.post(
            f"{url}{path}",
            json=payload,
            headers={"Connection": "close"},
            timeout=timeout,
        )
        _raise_for_sglang_zorl_status(resp, path=path, url=url)
        result = resp.json()
        if not result.get("success", False):
            raise RuntimeError(f"{path} failed on {url}: {result.get('error_message') or result}")
        return result

    results = []
    pool = ThreadPoolExecutor(max_workers=len(urls))
    futures = {pool.submit(post_one, url): url for url in urls}
    try:
        for future in as_completed(futures, timeout=timeout + 10):
            url = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"{path} failed on {url}: {exc}") from exc
        if len(results) != len(urls):
            pending_urls = [url for future, url in futures.items() if not future.done()]
            raise RuntimeError(f"{path} timed out on {pending_urls}")
    except FutureTimeoutError as exc:
        pending_urls = [url for future, url in futures.items() if not future.done()]
        raise RuntimeError(f"{path} timed out on {pending_urls}") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


def start_sglang_zorl_sessions(infer_urls, *, session_id, parent_lora_name, args):
    return _post_sglang_zorl_all(
        infer_urls,
        "/start_zorl_session",
        {
            "session_id": session_id,
            "parent_lora_name": parent_lora_name,
            "b_sigma": float(args.zorl_b_sigma),
            "num_pairs": int(args.zorl_num_pairs),
            "seed": int(args.zorl_seed),
            "antithetic_sampling": True,
            "perturbation_mode": args.zorl_perturbation_mode,
        },
    )


def start_sglang_zorl_generation(infer_urls, *, session_id, preload_candidates=False, num_pairs=None):
    payload = {
        "session_id": session_id,
        "preload_candidates": bool(preload_candidates),
    }
    if num_pairs is not None:
        payload["num_pairs"] = int(num_pairs)
    results = _post_sglang_zorl_all(
        infer_urls,
        "/start_zorl_generation",
        payload,
    )
    first = results[0]
    first_candidate_ids = [candidate["candidate_id"] for candidate in first.get("candidates", [])]
    for result in results[1:]:
        candidate_ids = [candidate["candidate_id"] for candidate in result.get("candidates", [])]
        if candidate_ids != first_candidate_ids:
            raise RuntimeError(f"Native SGLang ZORL candidate mismatch for generation {first.get('generation_id')}")
    return first


def apply_sglang_zorl_rewards(
    infer_urls,
    *,
    session_id,
    generation_id,
    candidate_rewards,
    lr,
    max_update_norm=None,
):
    payload = {
        "session_id": session_id,
        "generation_id": generation_id,
        "candidate_rewards": candidate_rewards,
        "learning_rate": float(lr),
    }
    if max_update_norm is not None:
        payload["max_update_norm"] = float(max_update_norm)
    results = _post_sglang_zorl_all(
        infer_urls,
        "/apply_zorl_rewards",
        payload,
    )
    for url in infer_urls if isinstance(infer_urls, list) else [infer_urls]:
        flush_inference_cache(url)
    return results[0]


def snapshot_sglang_zorl_parent(infer_urls, *, session_id, snapshot_id):
    results = _post_sglang_zorl_all(
        infer_urls,
        "/snapshot_zorl_parent",
        {"session_id": session_id, "snapshot_id": snapshot_id},
        timeout=300,
    )
    return results[0]


def restore_sglang_zorl_parent(infer_urls, *, session_id, snapshot_id):
    results = _post_sglang_zorl_all(
        infer_urls,
        "/restore_zorl_parent",
        {"session_id": session_id, "snapshot_id": snapshot_id},
        timeout=300,
    )
    for url in infer_urls if isinstance(infer_urls, list) else [infer_urls]:
        flush_inference_cache(url)
    return results[0]


def abort_sglang_zorl_generation(infer_urls, *, session_id, generation_id):
    return _post_sglang_zorl_all(
        infer_urls,
        "/abort_zorl_generation",
        {"session_id": session_id, "generation_id": generation_id},
        timeout=60,
    )


def save_weights(train_url, model_id, name):
    resp = requests.post(
        f"{train_url}/api/v1/save_weights",
        json={"model_id": model_id, "path": name},
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(
        wait_for_future(train_url, resp.json()["request_id"]), f"save_weights({model_id}, {name})"
    )


def load_weights(train_url, model_id, path, *, optimizer):
    resp = requests.post(
        f"{train_url}/api/v1/load_weights",
        json={"model_id": model_id, "path": path, "optimizer": optimizer},
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(
        wait_for_future(train_url, resp.json()["request_id"]),
        f"load_weights({model_id}, {path}, optimizer={optimizer})",
    )


def save_weights_for_sampler(train_url, model_id, name):
    resp = requests.post(
        f"{train_url}/api/v1/save_weights_for_sampler",
        json={"model_id": model_id, "name": name},
        timeout=30,
    )
    resp.raise_for_status()
    return _raise_on_failed_future(
        wait_for_future(train_url, resp.json()["request_id"]),
        f"save_weights_for_sampler({model_id}, {name})",
    )


def create_sampling_session(train_url, model_id, model_path):
    resp = requests.post(
        f"{train_url}/api/v1/create_sampling_session",
        json={"model_id": model_id, "model_path": model_path},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def set_sync_quantization(train_url):
    config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    resp = requests.post(f"{train_url}/api/v1/set_sync_quantization", json={"quantization": config}, timeout=10)
    resp.raise_for_status()
    print(f"    Set sync quantization: {resp.json().get('message')}")


def sync_weights(train_url, master_address, weight_version, *, model_id=DEFAULT_TRAIN_MODEL_ID, flush_cache=False):
    t0 = time.time()
    resp = requests.post(
        f"{train_url}/api/v1/sync_inference_weights",
        json={
            "model_id": model_id,
            "master_address": master_address,
            "weight_version": weight_version,
            "flush_cache": flush_cache,
        },
        timeout=600,
    )
    resp.raise_for_status()
    result = resp.json()
    print(
        f"    Sync: success={result.get('success')}, {time.time() - t0:.1f}s, "
        f"params={result.get('num_parameters', 'N/A')}, weight_version={weight_version}, flush_cache={flush_cache}"
    )
    return result


def get_model_info(infer_url):
    resp = requests.get(f"{infer_url}/model_info", timeout=10)
    resp.raise_for_status()
    return resp.json()


def wait_for_inference_weight_version(infer_urls, expected_version, timeout=120):
    deadline = time.time() + timeout
    last_seen = {}
    while time.time() < deadline:
        all_ready = True
        for url in infer_urls:
            try:
                info = get_model_info(url)
                version = info.get("weight_version")
                last_seen[url] = version
                if version != expected_version:
                    all_ready = False
            except Exception:
                all_ready = False
        if all_ready:
            return last_seen
        time.sleep(1)
    raise TimeoutError(f"Inference endpoints did not reach weight_version={expected_version}: {last_seen}")


def flush_inference_cache(infer_url, *, timeout_s=30.0, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                f"{infer_url}/flush_cache",
                params={"timeout": float(timeout_s)},
                headers={"Connection": "close"},
                timeout=max(30.0, float(timeout_s) + 5.0),
            )
            if resp.status_code == 200:
                return
            last_error = RuntimeError(
                f"flush_cache failed on {infer_url} with HTTP {resp.status_code}: {resp.text.strip()}"
            )
        except requests.RequestException as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
    if last_error is not None:
        raise last_error


def query_inference(infer_url, prompt, served_model_name, *, lora_path=None, enable_thinking=False, max_tokens=64):
    body = {
        "model": served_model_name,
        "messages": [
            {"role": "system", "content": _resolve_system_prompt(enable_thinking)},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if lora_path is not None:
        body["lora_path"] = lora_path
    resp = requests.post(
        f"{infer_url}/v1/chat/completions",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    metadata = payload.get("metadata", {})
    return (content.strip() if content else "(empty response)"), metadata


def test_inference(
    infer_urls,
    served_model_name,
    label="",
    expected_weight_version=None,
    lora_path=None,
    *,
    enable_thinking=False,
    max_tokens=64,
):
    print(f"    Inference ({label}):")
    total_correct = 0
    for url in infer_urls:
        port = url.split(":")[-1]
        correct = 0
        for question_key, _solution in CODES.items():
            meta = QUESTION_META[question_key]
            answer, metadata = query_inference(
                url,
                _question_text(meta),
                served_model_name,
                lora_path=lora_path,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
            )
            version = metadata.get("weight_version")
            version_ok = expected_weight_version is None or version == expected_weight_version
            parsed_expr = _extract_expression(answer)
            uses_each_once = parsed_expr is not None and _expression_uses_numbers_exactly_once(
                parsed_expr, meta["numbers"]
            )
            eval_value = _safe_eval_expr(parsed_expr) if uses_each_once else None
            match = eval_value is not None and abs(eval_value - float(meta["target"])) < 1e-6 and version_ok
            correct += match
            version_suffix = f", version={version}" if version is not None else ""
            nums_str = "+".join(str(n) for n in meta["numbers"])
            print(
                f"      [{'OK' if match else 'FAIL'}] :{port} {question_key} ({nums_str}->{meta['target']}): "
                f"'{answer}' (parsed='{parsed_expr}', eval={eval_value}, uses_each_once={uses_each_once}){version_suffix}"
            )
        total_correct += correct
    return total_correct


# ---------------------------------------------------------------------------
# ZORL reward helpers
# ---------------------------------------------------------------------------


_RETRYABLE_INFERENCE_STATUSES = {429, 502, 503, 504}


def _post_inference_with_retry(infer_url, endpoint, payload, *, max_attempts=8, timeout=300):
    url = f"{infer_url}{endpoint}"
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Connection": "close"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise
        else:
            if resp.status_code not in _RETRYABLE_INFERENCE_STATUSES:
                resp.raise_for_status()
                return resp
            if attempt == max_attempts:
                resp.raise_for_status()
        time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to POST {url}")


def generate_with_logprobs(infer_url, input_ids_batch, *, lora_paths):
    payload = {
        "input_ids": input_ids_batch,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
        "lora_path": lora_paths,
    }
    # Connection: close — see comment in generate_rollouts.
    resp = _post_inference_with_retry(infer_url, "/generate", payload)
    payload = resp.json()
    return payload if isinstance(payload, list) else [payload]


def _chat_completion_rollouts(
    infer_url,
    prompt_messages_batch,
    *,
    lora_paths,
    max_new_tokens,
    temperature,
    served_model_name,
    stop=None,
    enable_thinking=False,
):
    if prompt_messages_batch is None:
        raise ValueError("chat_completions rollout scoring requires prompt_messages_batch")
    if len(prompt_messages_batch) != len(lora_paths):
        raise ValueError(
            f"Expected one prompt message list per LoRA path, got {len(prompt_messages_batch)} prompts "
            f"and {len(lora_paths)} LoRA paths"
        )

    results = []
    for prompt_messages, lora_path in zip(prompt_messages_batch, lora_paths, strict=True):
        payload = {
            "model": served_model_name or "default",
            "messages": prompt_messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "n": 1,
            "lora_path": lora_path,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if stop:
            payload["stop"] = list(stop)
        resp = _post_inference_with_retry(infer_url, "/v1/chat/completions", payload)
        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        results.append({"text": message.get("content") or "", "output_ids": []})
    return results


def generate_rollouts(
    infer_url,
    input_ids_batch,
    *,
    lora_paths,
    max_new_tokens,
    temperature=0.0,
    prompt_messages_batch=None,
    served_model_name=None,
    api_format="generate",
    stop=None,
    enable_thinking=False,
):
    if api_format == "chat_completions":
        return _chat_completion_rollouts(
            infer_url,
            prompt_messages_batch,
            lora_paths=lora_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            served_model_name=served_model_name,
            stop=stop,
            enable_thinking=enable_thinking,
        )

    sampling_params = {
        "temperature": float(temperature),
        "max_new_tokens": max_new_tokens,
    }
    if stop:
        sampling_params["stop"] = list(stop)
    payload = {
        "input_ids": input_ids_batch,
        "sampling_params": sampling_params,
        "return_logprob": False,
        "lora_path": lora_paths,
    }
    # Connection: close forces a fresh TCP connection per request. Matches the
    # working /load_lora_adapter pattern; avoids stale keep-alive sockets that
    # appear to dead-block under sustained multi-LoRA + temperature>0 traffic
    # (we observed reproducible silent stalls at gen 5+ with default keep-alive).
    # Bumped timeout from 120s for the same backpressure reason.
    resp = _post_inference_with_retry(infer_url, "/generate", payload)
    payload = resp.json()
    return payload if isinstance(payload, list) else [payload]


# Sentinel for None entries returned by SGLang. The first input token has no
# prior context so SGLang reports its logprob as None; the same can also
# happen for tokens whose probability underflows to 0 (logprob = -inf serialized
# as JSON null). Treat both as "essentially impossible" for ZORL scoring so a
# numerically degenerate candidate gets a very-bad reward instead of crashing.
_LOGPROB_NONE_FLOOR = -100.0


def _logprob_from_entry(item):
    value = item[0]
    if value is None:
        return _LOGPROB_NONE_FLOOR
    return float(value)


def mean_answer_logprob(generate_result, answer_length):
    input_logprobs = generate_result.get("meta_info", {}).get("input_token_logprobs", [])
    if len(input_logprobs) < answer_length:
        raise RuntimeError(
            f"Expected at least {answer_length} input token logprobs, got {len(input_logprobs)}: {generate_result}"
        )
    answer_logprobs = [_logprob_from_entry(item) for item in input_logprobs[-answer_length:]]
    return sum(answer_logprobs) / len(answer_logprobs)


def mean_answer_suffix_logprob(generate_result, answer_length, matched_prefix_tokens):
    input_logprobs = generate_result.get("meta_info", {}).get("input_token_logprobs", [])
    if len(input_logprobs) < answer_length:
        raise RuntimeError(
            f"Expected at least {answer_length} input token logprobs, got {len(input_logprobs)}: {generate_result}"
        )
    prefix_tokens = max(0, min(int(matched_prefix_tokens), int(answer_length)))
    answer_logprobs = [_logprob_from_entry(item) for item in input_logprobs[-answer_length:]]
    suffix_logprobs = answer_logprobs[prefix_tokens:]
    if not suffix_logprobs:
        return 0.0
    return sum(suffix_logprobs) / len(suffix_logprobs)


def first_unresolved_token_logprob(generate_result, answer_length, matched_prefix_tokens):
    input_logprobs = generate_result.get("meta_info", {}).get("input_token_logprobs", [])
    if len(input_logprobs) < answer_length:
        raise RuntimeError(
            f"Expected at least {answer_length} input token logprobs, got {len(input_logprobs)}: {generate_result}"
        )
    prefix_tokens = max(0, min(int(matched_prefix_tokens), int(answer_length)))
    answer_logprobs = [_logprob_from_entry(item) for item in input_logprobs[-answer_length:]]
    if prefix_tokens >= len(answer_logprobs):
        return 0.0
    return answer_logprobs[prefix_tokens]


def _common_prefix_length(lhs, rhs):
    count = 0
    for left, right in zip(lhs, rhs):
        if left != right:
            break
        count += 1
    return count


def _normalize_generated_answer(text):
    return (text or "").strip()


def _char_match_ratio(generated_text, expected_text):
    max_len = max(len(generated_text), len(expected_text), 1)
    matches = sum(
        1
        for generated_char, expected_char in zip(generated_text, expected_text, strict=False)
        if generated_char == expected_char
    )
    return matches / max_len


# Allowed characters in a Countdown answer expression. Anything outside this
# alphabet (letters, function names, ** etc.) gets the candidate rejected.
_EXPR_ALLOWED_RE = re.compile(r"^[\s0-9+\-*/().]+$")
# Greedy first-pass parser: pull a contiguous run of expression characters out
# of the model's text. Models often prefix "Answer: " or wrap with "= 24" — we
# strip those and try again if the first attempt fails to evaluate.
_EXPR_GREEDY_RE = re.compile(r"[0-9+\-*/().\s]+")
# Tokenize numbers within a candidate expression to verify each input number
# is used exactly once.
_NUMBER_TOKEN_RE = re.compile(r"\d+")
# When --zorl-enable-thinking is on, Qwen3 wraps reasoning in <think>...</think>
# tags. Strip the block so _extract_expression sees only the final answer; any
# digits inside the reasoning would otherwise win the "longest expression-shaped
# substring" search.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _safe_eval_expr(expr):
    """Evaluate an arithmetic expression made up of digits, +, -, *, /, parens,
    and whitespace. Returns float on success, None on syntax/semantics error
    (incl. division by zero). Does NOT use Python's eval(); strict whitelist
    plus a hand parser is overkill — instead we whitelist the character set
    and use ``compile + eval`` with empty globals for safety. Parentheses /
    operator precedence handled by Python.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None
    if not _EXPR_ALLOWED_RE.match(expr):
        return None
    try:
        # Empty globals/locals; the regex above already rejects names.
        return float(eval(expr, {"__builtins__": {}}, {}))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError, OverflowError):
        return None


def _extract_expression(text):
    """Try to pull an arithmetic expression out of the model's reply. Strips
    common decoration like ``Answer:``, ``= 24``, surrounding text. Returns
    the first substring whose character set passes the allowed regex; the
    caller is responsible for evaluating it.

    When the model's reply contains a ``<think>...</think>`` block (Qwen3's
    native CoT mode), the block is stripped first so that digits inside the
    reasoning trace can't win the "longest expression-shaped substring"
    search over the actual final answer.
    """
    if not text:
        return None
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Also drop any orphaned closing </think> tag if the open tag was missing or
    # truncated — without this, a half-emitted block can leak digits into the
    # extraction window.
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = cleaned.strip()
    # Strip a trailing equals comparison, e.g. "(9-3)*4 = 24"
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[0].strip()
    # Find the longest run of allowed characters; that's almost always the
    # expression. Pick the longest candidate to avoid trailing punctuation
    # that splits a real expression.
    candidates = _EXPR_GREEDY_RE.findall(cleaned)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        stripped = candidate.strip()
        if any(ch in stripped for ch in "0123456789"):
            return stripped
    return None


def _expression_uses_numbers_exactly_once(expr, expected_numbers):
    """Multiset compare the integer literals in ``expr`` to ``expected_numbers``.
    Returns True iff each expected number appears exactly once and no extras."""
    found = [int(tok) for tok in _NUMBER_TOKEN_RE.findall(expr or "")]
    return sorted(found) == sorted(expected_numbers)


def _rollout_value_score(*, eval_value, target, uses_each_once):
    if not uses_each_once or eval_value is None or target is None:
        return 0.0
    scale = max(abs(float(target)), 1.0)
    relative_error = abs(float(eval_value) - float(target)) / scale
    return max(0.0, 1.0 - min(relative_error, 1.0))


def _rollout_reward_components(generate_result, reward_example):
    generated_text = _normalize_generated_answer(generate_result.get("text", ""))
    expected_text = reward_example["code"]
    expected_output_ids = reward_example["answer_token_ids"]
    generated_output_ids = list(generate_result.get("output_ids", []))

    # Token-prefix metrics retained for compatibility with the reward plumbing
    # and for diagnostic logging — not used in the binary signal below.
    prefix_len = _common_prefix_length(generated_output_ids[: len(expected_output_ids)], expected_output_ids)
    prefix_ratio = prefix_len / max(len(expected_output_ids), 1)
    char_match_ratio = _char_match_ratio(generated_text, expected_text)

    # Verifiable binary reward: extract the candidate expression, validate it
    # uses each input number exactly once, evaluate it, and compare to target.
    target = reward_example.get("target")
    numbers = reward_example.get("numbers") or []
    parsed_expr = _extract_expression(generated_text)
    uses_each_once = parsed_expr is not None and _expression_uses_numbers_exactly_once(parsed_expr, numbers)
    eval_value = _safe_eval_expr(parsed_expr) if uses_each_once else None
    # Tolerance for numerical artifacts (e.g. integer division written as /).
    exact_match = eval_value is not None and target is not None and abs(eval_value - float(target)) < 1e-6
    value_score = _rollout_value_score(eval_value=eval_value, target=target, uses_each_once=uses_each_once)

    return {
        "generated_text": generated_text,
        "parsed_expr": parsed_expr,
        "eval_value": eval_value,
        "uses_each_once": uses_each_once,
        "target": target,
        "numbers": numbers,
        "prefix_len": prefix_len,
        "prefix_ratio": prefix_ratio,
        "exact_match": exact_match,
        "value_score": value_score,
        "char_match_ratio": char_match_ratio,
    }


def _reward_for_mode(
    args,
    *,
    teacher_forced_logprob,
    teacher_forced_suffix_logprob,
    teacher_forced_first_error_logprob,
    rollout_components,
    hard_project_reward_multiplier=1.0,
):
    if args.zorl_reward_mode == "teacher_forced":
        return teacher_forced_logprob

    reward = 0.0
    if args.zorl_reward_mode in {"rollout", "hybrid"}:
        dense_scale = float(hard_project_reward_multiplier or 1.0)
        reward += dense_scale * args.zorl_rollout_prefix_weight * float(rollout_components["prefix_ratio"])
        reward += dense_scale * args.zorl_rollout_char_weight * float(rollout_components["char_match_ratio"])
        reward += dense_scale * getattr(args, "zorl_rollout_valid_weight", 0.0) * float(
            rollout_components.get(
                "uses_each_once_rate",
                1.0 if bool(rollout_components.get("uses_each_once", False)) else 0.0,
            )
        )
        reward += dense_scale * getattr(args, "zorl_rollout_value_weight", 0.0) * float(
            rollout_components.get("value_score", 0.0) or 0.0
        )
        # exact_match is either a 0/1 bool (single-rollout mode) or a fraction in
        # [0, 1] (multi-rollout mode, exact-match RATE across N samples). Multiply
        # the bonus by float(exact_match) so the dense Modal-style reward scales
        # cleanly: 1/N granularity per puzzle → smooth ES gradient.
        reward += args.zorl_rollout_exact_bonus * float(rollout_components["exact_match"])

    if args.zorl_reward_mode == "hybrid":
        reward += args.zorl_teacher_forced_weight * math.exp(teacher_forced_logprob)
        reward += args.zorl_teacher_forced_suffix_weight * math.exp(teacher_forced_suffix_logprob)
        first_error_scale = 1.0
        if args.zorl_teacher_forced_first_error_prefix_power > 0.0:
            first_error_scale = float(rollout_components["prefix_ratio"]) ** float(
                args.zorl_teacher_forced_first_error_prefix_power
            )
        reward += (
            args.zorl_teacher_forced_first_error_weight
            * first_error_scale
            * math.exp(teacher_forced_first_error_logprob)
        )

    return reward


def _default_rollout_max_new_tokens(reward_data, args):
    if args.zorl_rollout_max_new_tokens is not None:
        return args.zorl_rollout_max_new_tokens
    return max(len(example["answer_token_ids"]) + args.zorl_rollout_extra_tokens for example in reward_data)


def score_zorl_candidates(infer_url, candidates, reward_data, args):
    # Accept a single URL or a list; round-robin across replicas.
    infer_urls = infer_url if isinstance(infer_url, list) else [infer_url]
    if args.zorl_inference_api_format == "chat_completions" and args.zorl_reward_mode != "rollout":
        raise ValueError("chat_completions ZORL scoring currently supports --zorl-reward-mode=rollout only")

    reward_totals = {candidate["candidate_id"]: 0.0 for candidate in candidates}
    reward_counts = {candidate["candidate_id"]: 0 for candidate in candidates}
    candidate_metrics = {
        candidate["candidate_id"]: {
            "teacher_forced_logprob_sum": 0.0,
            "teacher_forced_suffix_logprob_sum": 0.0,
            "teacher_forced_first_error_logprob_sum": 0.0,
            "teacher_forced_count": 0,
            "rollout_prefix_sum": 0.0,
            "rollout_char_sum": 0.0,
            "rollout_valid_sum": 0.0,
            "rollout_value_sum": 0.0,
            "rollout_exact_count": 0,
            "rollout_total": 0,
            "rollout_outputs": {},
            "project_metrics": {},
        }
        for candidate in candidates
    }
    teacher_forced_values = []
    teacher_forced_suffix_values = []
    teacher_forced_first_error_values = []
    rollout_prefix_values = []
    rollout_char_values = []
    rollout_valid_values = []
    rollout_value_values = []
    rollout_exact_matches = 0
    rollout_total = 0

    # Submit all inference requests concurrently — TF and rollout calls across
    # examples and candidate batches are independent HTTP requests.
    # Requests are distributed round-robin across inference replicas.
    need_rollouts = args.zorl_reward_mode in {"rollout", "hybrid"}
    rollouts_per_puzzle = max(1, int(getattr(args, "zorl_rollouts_per_puzzle", 1)))
    rollout_temperature = float(getattr(args, "zorl_rollout_temperature", 0.0))
    max_workers = getattr(args, "zorl_score_max_workers", 8)
    url_idx = 0
    need_teacher_forced = args.zorl_reward_mode in {"teacher_forced", "hybrid"}
    score_order = getattr(args, "zorl_score_order", "example_major")
    score_routing = getattr(args, "zorl_score_routing", "round_robin")
    if score_routing == "candidate_chunk_sticky" and score_order != "candidate_major":
        raise ValueError("--zorl-score-routing candidate_chunk_sticky requires --zorl-score-order candidate_major")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:

        def submit_score_batch(ex_idx, example, start, candidate_batch, preferred_url=None):
            nonlocal url_idx
            lora_paths = [candidate["lora_name"] for candidate in candidate_batch]
            tf_future = None
            rollout_future_list = []

            if preferred_url is None:
                target_url = infer_urls[url_idx % len(infer_urls)]
                url_idx += 1
            else:
                target_url = preferred_url
            if need_teacher_forced:
                tf_future = pool.submit(
                    generate_with_logprobs,
                    target_url,
                    input_ids_batch=[example["full_ids"] for _ in candidate_batch],
                    lora_paths=lora_paths,
                )
            if need_rollouts:
                max_new_tokens = (
                    args.zorl_rollout_max_new_tokens
                    if args.zorl_rollout_max_new_tokens is not None
                    else max(len(example["answer_token_ids"]) + args.zorl_rollout_extra_tokens, 1)
                )
                for _rollout_idx in range(rollouts_per_puzzle):
                    if preferred_url is None:
                        # Round-robin URL across rollout submissions too so
                        # parallel samples land on different replicas.
                        ro_target_url = infer_urls[url_idx % len(infer_urls)]
                        url_idx += 1
                    else:
                        ro_target_url = preferred_url
                    rollout_future_list.append(
                        pool.submit(
                            generate_rollouts,
                            ro_target_url,
                            input_ids_batch=[example["prompt_ids"] for _ in candidate_batch],
                            lora_paths=lora_paths,
                            max_new_tokens=max_new_tokens,
                            temperature=rollout_temperature,
                            prompt_messages_batch=[example["prompt_messages"] for _ in candidate_batch],
                            served_model_name=args.served_model_name or args.model,
                            api_format=args.zorl_inference_api_format,
                            stop=args.zorl_rollout_stop,
                        )
                    )
            return ex_idx, example, start, candidate_batch, tf_future, rollout_future_list

        def process_score_batch(ex_idx, example, start, candidate_batch, tf_future, rollout_future_list):
            nonlocal rollout_exact_matches, rollout_total
            if need_teacher_forced:
                tf_results = tf_future.result()
                if len(tf_results) != len(candidate_batch):
                    raise RuntimeError(
                        f"Expected {len(candidate_batch)} batched ZORL teacher-forced results, got {len(tf_results)} "
                        f"for {example['project']}"
                    )
            else:
                tf_results = [None] * len(candidate_batch)

            rollout_results_per_sample = None
            if rollout_future_list:
                # Resolve each of the N rollout-sample futures; check shape.
                rollout_results_per_sample = [f.result() for f in rollout_future_list]
                for sample_idx, sample_results in enumerate(rollout_results_per_sample):
                    if len(sample_results) != len(candidate_batch):
                        raise RuntimeError(
                            f"Expected {len(candidate_batch)} batched ZORL rollout results "
                            f"(sample {sample_idx}), got {len(sample_results)} "
                            f"for {example['project']}"
                        )

            for index, (candidate, tf_result) in enumerate(zip(candidate_batch, tf_results, strict=True)):
                if tf_result is None:
                    teacher_forced_logprob = _LOGPROB_NONE_FLOOR
                    teacher_forced_suffix_logprob = _LOGPROB_NONE_FLOOR
                    teacher_forced_first_error_logprob = _LOGPROB_NONE_FLOOR
                else:
                    teacher_forced_logprob = mean_answer_logprob(tf_result, len(example["answer_ids"]))
                rollout_components = {
                    "generated_text": "",
                    "prefix_len": 0,
                    "prefix_ratio": 0.0,
                    "exact_match": False,
                    "char_match_ratio": 0.0,
                }
                if rollout_results_per_sample is not None:
                    # Average rollout components across the N samples for this
                    # (puzzle, candidate) pair. exact_match becomes a fraction
                    # in [0, 1] with 1/N granularity — the dense per-perturbation
                    # signal that lets ES find consistent gradient directions.
                    per_sample_components = [
                        _rollout_reward_components(sample_results[index], example)
                        for sample_results in rollout_results_per_sample
                    ]
                    n_samples = len(per_sample_components)
                    prefix_ratio_avg = sum(float(c["prefix_ratio"]) for c in per_sample_components) / n_samples
                    char_match_ratio_avg = sum(float(c["char_match_ratio"]) for c in per_sample_components) / n_samples
                    valid_rate_avg = (
                        sum(1.0 if bool(c["uses_each_once"]) else 0.0 for c in per_sample_components) / n_samples
                    )
                    value_score_avg = sum(float(c["value_score"]) for c in per_sample_components) / n_samples
                    exact_match_rate = sum(int(c["exact_match"]) for c in per_sample_components) / n_samples
                    # Use the first sample's prefix_len / generated_text for
                    # downstream TF-suffix and first-error logprob computations
                    # (these care about a representative rollout, not an average).
                    rollout_components = dict(per_sample_components[0])
                    rollout_components["prefix_ratio"] = prefix_ratio_avg
                    rollout_components["char_match_ratio"] = char_match_ratio_avg
                    rollout_components["uses_each_once_rate"] = valid_rate_avg
                    rollout_components["value_score"] = value_score_avg
                    # exact_match is consumed as a bool in the binary reward
                    # bonus path (`exact_bonus * float(exact_match)`); make it
                    # the fractional rate so the reward is dense.
                    rollout_components["exact_match"] = exact_match_rate
                    rollout_components["exact_match_rate"] = exact_match_rate
                    rollout_components["num_rollout_samples"] = n_samples
                    rollout_prefix_values.append(prefix_ratio_avg)
                    rollout_char_values.append(char_match_ratio_avg)
                    rollout_valid_values.append(valid_rate_avg)
                    rollout_value_values.append(value_score_avg)
                    rollout_exact_matches += int(round(exact_match_rate * n_samples))
                    rollout_total += n_samples

                if tf_result is not None:
                    teacher_forced_suffix_logprob = mean_answer_suffix_logprob(
                        tf_result,
                        len(example["answer_ids"]),
                        int(rollout_components["prefix_len"]),
                    )
                    teacher_forced_first_error_logprob = first_unresolved_token_logprob(
                        tf_result,
                        len(example["answer_ids"]),
                        int(rollout_components["prefix_len"]),
                    )
                hard_project_names = getattr(args, "_zorl_hard_project_reward_names", None) or set()
                hard_project_multiplier = (
                    float(getattr(args, "_zorl_hard_project_reward_multiplier", 1.0) or 1.0)
                    if example["project"] in hard_project_names
                    else 1.0
                )
                reward = _reward_for_mode(
                    args,
                    teacher_forced_logprob=teacher_forced_logprob,
                    teacher_forced_suffix_logprob=teacher_forced_suffix_logprob,
                    teacher_forced_first_error_logprob=teacher_forced_first_error_logprob,
                    rollout_components=rollout_components,
                    hard_project_reward_multiplier=hard_project_multiplier,
                )
                reward *= float(args.zorl_project_weights.get(example["project"], 1.0))
                candidate_id = candidate["candidate_id"]
                reward_totals[candidate_id] += reward
                reward_counts[candidate_id] += 1
                candidate_metric = candidate_metrics[candidate_id]
                teacher_forced_prob = math.exp(teacher_forced_logprob)
                candidate_metric["teacher_forced_logprob_sum"] += teacher_forced_logprob
                candidate_metric["teacher_forced_suffix_logprob_sum"] += teacher_forced_suffix_logprob
                candidate_metric["teacher_forced_first_error_logprob_sum"] += teacher_forced_first_error_logprob
                candidate_metric["teacher_forced_count"] += 1
                rollout_exact_rate = float(rollout_components["exact_match"])
                candidate_metric["project_metrics"][example["project"]] = {
                    "reward": float(reward),
                    "teacher_forced_logprob": float(teacher_forced_logprob),
                    "teacher_forced_prob": float(teacher_forced_prob),
                    "teacher_forced_suffix_logprob": float(teacher_forced_suffix_logprob),
                    "teacher_forced_suffix_prob": float(math.exp(teacher_forced_suffix_logprob)),
                    "teacher_forced_first_error_logprob": float(teacher_forced_first_error_logprob),
                    "teacher_forced_first_error_prob": float(math.exp(teacher_forced_first_error_logprob)),
                    "rollout_prefix_ratio": float(rollout_components["prefix_ratio"]),
                    "rollout_char_match_ratio": float(rollout_components["char_match_ratio"]),
                    "rollout_valid_rate": float(
                        rollout_components.get(
                            "uses_each_once_rate",
                            1.0 if bool(rollout_components.get("uses_each_once", False)) else 0.0,
                        )
                    ),
                    "rollout_value_score": float(rollout_components.get("value_score", 0.0) or 0.0),
                    "rollout_exact_match": rollout_exact_rate > 0.0,
                    "rollout_exact_rate": rollout_exact_rate,
                    "generated_text": rollout_components["generated_text"],
                }
                if rollout_results_per_sample is not None:
                    candidate_metric["rollout_prefix_sum"] += float(rollout_components["prefix_ratio"])
                    candidate_metric["rollout_char_sum"] += float(rollout_components["char_match_ratio"])
                    candidate_metric["rollout_valid_sum"] += float(
                        rollout_components.get(
                            "uses_each_once_rate",
                            1.0 if bool(rollout_components.get("uses_each_once", False)) else 0.0,
                        )
                    )
                    candidate_metric["rollout_value_sum"] += float(rollout_components.get("value_score", 0.0) or 0.0)
                    # In multi-rollout mode this accumulates the fractional
                    # exact-match RATE per puzzle (∈ [0, 1]); summed across
                    # puzzles, total ∈ [0, num_puzzles]. Int casts in
                    # downstream display logic floor it for "X/N" formatting.
                    candidate_metric["rollout_exact_count"] += float(rollout_components["exact_match"])
                    candidate_metric["rollout_total"] += 1
                    candidate_metric["rollout_outputs"][example["project"]] = rollout_components["generated_text"]
                if tf_result is not None:
                    teacher_forced_values.append(teacher_forced_logprob)
                    teacher_forced_suffix_values.append(teacher_forced_suffix_logprob)
                    teacher_forced_first_error_values.append(teacher_forced_first_error_logprob)

        if score_order == "candidate_major":
            candidate_starts = list(range(0, len(candidates), args.zorl_score_batch_size))
            if score_routing == "candidate_chunk_sticky" and len(infer_urls) > 1:
                for wave_start in range(0, len(candidate_starts), len(infer_urls)):
                    futures = []
                    wave_starts = candidate_starts[wave_start : wave_start + len(infer_urls)]
                    # Interleave by example so the first executor slots spread
                    # across endpoints instead of overfilling one SGLang pod.
                    for ex_idx, example in enumerate(reward_data):
                        for url_offset, start in enumerate(wave_starts):
                            candidate_batch = candidates[start : start + args.zorl_score_batch_size]
                            target_url = infer_urls[url_offset % len(infer_urls)]
                            futures.append(
                                submit_score_batch(
                                    ex_idx,
                                    example,
                                    start,
                                    candidate_batch,
                                    preferred_url=target_url,
                                )
                            )
                    for future_args in futures:
                        process_score_batch(*future_args)
            else:
                for start in candidate_starts:
                    candidate_batch = candidates[start : start + args.zorl_score_batch_size]
                    futures = [
                        submit_score_batch(ex_idx, example, start, candidate_batch)
                        for ex_idx, example in enumerate(reward_data)
                    ]
                    for future_args in futures:
                        process_score_batch(*future_args)
        elif score_order == "example_major":
            futures = {}
            for ex_idx, example in enumerate(reward_data):
                for start in range(0, len(candidates), args.zorl_score_batch_size):
                    candidate_batch = candidates[start : start + args.zorl_score_batch_size]
                    futures[(ex_idx, start)] = submit_score_batch(ex_idx, example, start, candidate_batch)

            # Process results in original order (futures resolve in background).
            for ex_idx, _example in enumerate(reward_data):
                for start in range(0, len(candidates), args.zorl_score_batch_size):
                    process_score_batch(*futures[(ex_idx, start)])
        else:
            raise ValueError(f"Unsupported ZORL score order: {score_order}")

    candidate_rewards = []
    reward_values = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        reward_count = reward_counts[candidate_id]
        if reward_count <= 0:
            raise RuntimeError(f"No ZORL rewards were computed for candidate {candidate_id}")
        reward_mean = reward_totals[candidate_id] / reward_count
        candidate_metric = candidate_metrics[candidate_id]
        teacher_forced_count = int(candidate_metric["teacher_forced_count"])
        teacher_forced_logprob_mean = (
            candidate_metric["teacher_forced_logprob_sum"] / teacher_forced_count if teacher_forced_count > 0 else None
        )
        teacher_forced_suffix_logprob_mean = (
            candidate_metric["teacher_forced_suffix_logprob_sum"] / teacher_forced_count
            if teacher_forced_count > 0
            else None
        )
        teacher_forced_first_error_logprob_mean = (
            candidate_metric["teacher_forced_first_error_logprob_sum"] / teacher_forced_count
            if teacher_forced_count > 0
            else None
        )
        teacher_forced_prob_mean = (
            math.exp(teacher_forced_logprob_mean) if teacher_forced_logprob_mean is not None else None
        )
        teacher_forced_suffix_prob_mean = (
            math.exp(teacher_forced_suffix_logprob_mean) if teacher_forced_suffix_logprob_mean is not None else None
        )
        teacher_forced_first_error_prob_mean = (
            math.exp(teacher_forced_first_error_logprob_mean)
            if teacher_forced_first_error_logprob_mean is not None
            else None
        )
        rollout_candidate_total = int(candidate_metric["rollout_total"])
        rollout_prefix_mean = (
            candidate_metric["rollout_prefix_sum"] / rollout_candidate_total if rollout_candidate_total > 0 else None
        )
        rollout_char_match_mean = (
            candidate_metric["rollout_char_sum"] / rollout_candidate_total if rollout_candidate_total > 0 else None
        )
        rollout_valid_mean = (
            candidate_metric["rollout_valid_sum"] / rollout_candidate_total if rollout_candidate_total > 0 else None
        )
        rollout_value_mean = (
            candidate_metric["rollout_value_sum"] / rollout_candidate_total if rollout_candidate_total > 0 else None
        )
        rollout_exact_count = float(candidate_metric["rollout_exact_count"])
        rollout_exact_rate = rollout_exact_count / rollout_candidate_total if rollout_candidate_total > 0 else None
        candidate_rewards.append(
            {
                "candidate_id": candidate_id,
                "reward_mean": reward_mean,
                "num_rollouts": reward_count,
                "teacher_forced_logprob_mean": teacher_forced_logprob_mean,
                "teacher_forced_prob_mean": teacher_forced_prob_mean,
                "teacher_forced_suffix_logprob_mean": teacher_forced_suffix_logprob_mean,
                "teacher_forced_suffix_prob_mean": teacher_forced_suffix_prob_mean,
                "teacher_forced_first_error_logprob_mean": teacher_forced_first_error_logprob_mean,
                "teacher_forced_first_error_prob_mean": teacher_forced_first_error_prob_mean,
                "rollout_prefix_mean": rollout_prefix_mean,
                "rollout_char_match_mean": rollout_char_match_mean,
                "rollout_valid_mean": rollout_valid_mean,
                "rollout_value_mean": rollout_value_mean,
                "rollout_exact_count": rollout_exact_count,
                "rollout_exact_rate": rollout_exact_rate,
                "rollout_outputs": candidate_metric["rollout_outputs"],
                "project_metrics": candidate_metric["project_metrics"],
            }
        )
        reward_values.append(reward_mean)

    reward_stats = {
        "reward_mean": sum(reward_values) / len(reward_values),
        "reward_min": min(reward_values),
        "reward_max": max(reward_values),
        "score_order": score_order,
        "score_routing": score_routing,
    }
    if teacher_forced_values:
        reward_stats["teacher_forced_logprob_mean"] = sum(teacher_forced_values) / len(teacher_forced_values)
        reward_stats["teacher_forced_prob_mean"] = sum(math.exp(value) for value in teacher_forced_values) / len(
            teacher_forced_values
        )
    if teacher_forced_suffix_values:
        reward_stats["teacher_forced_suffix_logprob_mean"] = sum(teacher_forced_suffix_values) / len(
            teacher_forced_suffix_values
        )
        reward_stats["teacher_forced_suffix_prob_mean"] = sum(
            math.exp(value) for value in teacher_forced_suffix_values
        ) / len(teacher_forced_suffix_values)
    if teacher_forced_first_error_values:
        reward_stats["teacher_forced_first_error_logprob_mean"] = sum(teacher_forced_first_error_values) / len(
            teacher_forced_first_error_values
        )
        reward_stats["teacher_forced_first_error_prob_mean"] = sum(
            math.exp(value) for value in teacher_forced_first_error_values
        ) / len(teacher_forced_first_error_values)
    if rollout_total > 0:
        reward_stats["rollout_prefix_mean"] = sum(rollout_prefix_values) / rollout_total
        reward_stats["rollout_char_match_mean"] = sum(rollout_char_values) / rollout_total
        reward_stats["rollout_valid_mean"] = sum(rollout_valid_values) / rollout_total
        reward_stats["rollout_value_mean"] = sum(rollout_value_values) / rollout_total
        reward_stats["rollout_exact_rate"] = rollout_exact_matches / rollout_total

    return candidate_rewards, reward_stats


def _candidate_sort_key(candidate_reward):
    return (
        float(candidate_reward.get("rollout_exact_count", 0.0) or 0.0),
        float(candidate_reward.get("reward_mean", float("-inf"))),
        float(candidate_reward.get("teacher_forced_first_error_prob_mean", 0.0) or 0.0),
        float(candidate_reward.get("teacher_forced_suffix_prob_mean", 0.0) or 0.0),
        float(candidate_reward.get("rollout_value_mean", 0.0) or 0.0),
        float(candidate_reward.get("rollout_valid_mean", 0.0) or 0.0),
        float(candidate_reward.get("rollout_char_match_mean", 0.0) or 0.0),
        float(candidate_reward.get("rollout_prefix_mean", 0.0) or 0.0),
        float(candidate_reward.get("teacher_forced_prob_mean", 0.0) or 0.0),
    )


def _project_metric_exact_rate(project_metric):
    if "rollout_exact_rate" in project_metric:
        return float(project_metric.get("rollout_exact_rate", 0.0) or 0.0)
    return 1.0 if bool(project_metric.get("rollout_exact_match", False)) else 0.0


def _project_metric_sort_key(project_metric):
    return (
        _project_metric_exact_rate(project_metric),
        float(project_metric.get("reward", float("-inf"))),
        float(project_metric.get("teacher_forced_first_error_prob", 0.0) or 0.0),
        float(project_metric.get("teacher_forced_suffix_prob", 0.0) or 0.0),
        float(project_metric.get("rollout_value_score", 0.0) or 0.0),
        float(project_metric.get("rollout_valid_rate", 0.0) or 0.0),
        float(project_metric.get("rollout_char_match_ratio", 0.0) or 0.0),
        float(project_metric.get("rollout_prefix_ratio", 0.0) or 0.0),
        float(project_metric.get("teacher_forced_prob", 0.0) or 0.0),
    )


def _best_candidate_for_project(candidate_rewards, project_name):
    candidates_with_project = [
        candidate_reward
        for candidate_reward in candidate_rewards
        if project_name in candidate_reward.get("project_metrics", {})
    ]
    if not candidates_with_project:
        return None
    return max(
        candidates_with_project,
        key=lambda candidate_reward: _project_metric_sort_key(candidate_reward["project_metrics"][project_name]),
    )


def _collect_pair_rows(generation, candidate_rewards):
    candidate_specs = {candidate["candidate_id"]: candidate for candidate in generation["candidates"]}
    reward_by_candidate = {candidate_reward["candidate_id"]: candidate_reward for candidate_reward in candidate_rewards}

    pair_specs = {}
    for candidate in generation["candidates"]:
        pair_specs.setdefault(int(candidate["perturbation_index"]), {})[candidate["direction"]] = candidate

    pair_rows = []
    for pair_index in sorted(pair_specs):
        directions = pair_specs[pair_index]
        positive = directions.get("positive")
        negative = directions.get("negative")
        if positive is None or negative is None:
            raise RuntimeError(f"Missing antithetic ZORL pair entries for perturbation_index={pair_index}")

        positive_reward = reward_by_candidate.get(positive["candidate_id"])
        negative_reward = reward_by_candidate.get(negative["candidate_id"])
        if positive_reward is None or negative_reward is None:
            raise RuntimeError(f"Missing candidate rewards for perturbation_index={pair_index}")

        positive_mean = float(positive_reward["reward_mean"])
        negative_mean = float(negative_reward["reward_mean"])
        positive_key = _candidate_sort_key(positive_reward)
        negative_key = _candidate_sort_key(negative_reward)
        winner = positive if positive_key >= negative_key else negative
        winner_reward = positive_reward if positive_key >= negative_key else negative_reward

        pair_rows.append(
            {
                "pair_index": pair_index,
                "positive_candidate_id": positive["candidate_id"],
                "negative_candidate_id": negative["candidate_id"],
                "positive_reward_mean": positive_mean,
                "negative_reward_mean": negative_mean,
                "delta": positive_mean - negative_mean,
                "abs_delta": abs(positive_mean - negative_mean),
                "winner_candidate_id": winner["candidate_id"],
                "winner_direction": winner["direction"],
                "winner_reward_mean": float(winner_reward["reward_mean"]),
                "winner_exact_count": float(winner_reward.get("rollout_exact_count", 0.0) or 0.0),
                "winner_num_rollouts": int(winner_reward["num_rollouts"]),
                "positive_lora_name": candidate_specs[positive["candidate_id"]]["lora_name"],
                "negative_lora_name": candidate_specs[negative["candidate_id"]]["lora_name"],
            }
        )

    return pair_rows


def _encode_pair_winners(selected_rows, *, score_fn):
    payload = []
    for row in selected_rows:
        score = float(score_fn(row))
        if score < 0:
            raise ValueError(f"Encoded winner score must be non-negative, got {score}")
        payload.append(
            {
                "candidate_id": row["winner_candidate_id"],
                "reward_mean": score,
                "num_rollouts": row["winner_num_rollouts"],
            }
        )
    return payload


def _set_zorl_score_normalization(candidate_rewards, normalization):
    return [
        {
            **candidate_reward,
            "_zorl_score_normalization": normalization,
        }
        for candidate_reward in candidate_rewards
    ]


def _parent_project_metric(parent_baseline, project_name):
    if parent_baseline is None:
        return None
    project_metrics = parent_baseline.get("project_metrics", {})
    return project_metrics.get(project_name)


def _filter_parent_advantage_reward_data(reward_data, parent_baseline, *, solved_exact_rate):
    if parent_baseline is None:
        return list(reward_data), {}
    project_metrics = parent_baseline.get("project_metrics", {})
    if not project_metrics:
        return list(reward_data), {}

    active_reward_data = []
    skipped_projects = []
    active_projects = []
    threshold = float(solved_exact_rate)
    for example in reward_data:
        project_name = example["project"]
        project_metric = project_metrics.get(project_name, {})
        exact_rate = _project_metric_exact_rate(project_metric)
        if exact_rate >= threshold:
            skipped_projects.append(project_name)
        else:
            active_projects.append(project_name)
            active_reward_data.append(example)

    metadata = {
        "parent_advantage_active_projects": len(active_projects),
        "parent_advantage_skipped_projects": len(skipped_projects),
        "parent_advantage_total_projects": len(reward_data),
        "parent_advantage_active_project_names": ",".join(active_projects),
        "parent_advantage_skipped_project_names": ",".join(skipped_projects),
    }
    if not active_reward_data:
        metadata["parent_advantage_all_projects_solved"] = True
        return list(reward_data), metadata
    return active_reward_data, metadata


def _select_zorl_update_payload(generation, candidate_rewards, args, lr, parent_baseline=None):
    metadata = {
        "apply_lr": float(lr),
        "update_strategy": args.zorl_update_strategy,
        "score_normalization": "standard",
    }
    if args.zorl_update_strategy == "es":
        metadata["selected_pairs"] = int(generation["num_pairs"])
        return candidate_rewards, float(lr) * args.zorl_update_lr_scale, metadata

    if args.zorl_update_strategy == "best_candidate":
        best_candidate = max(candidate_rewards, key=_candidate_sort_key)
        metadata["selected_pairs"] = 1
        metadata["selected_candidate_id"] = str(best_candidate["candidate_id"])
        metadata["selected_candidate_reward"] = float(best_candidate["reward_mean"])
        metadata["selected_candidate_exact_count"] = float(best_candidate.get("rollout_exact_count", 0.0) or 0.0)
        metadata["score_normalization"] = "none"
        apply_lr = args.zorl_b_sigma * args.zorl_best_candidate_step_scale
        metadata["apply_lr"] = float(apply_lr)
        update_score = 1.0 if float(best_candidate["reward_mean"]) > 0.0 else 0.0
        return (
            _set_zorl_score_normalization(
                [
                    {
                        "candidate_id": str(best_candidate["candidate_id"]),
                        "reward_mean": update_score,
                        "num_rollouts": int(best_candidate.get("num_rollouts", 1) or 1),
                    }
                ],
                "none",
            ),
            apply_lr,
            metadata,
        )

    if args.zorl_update_strategy in {
        "project_winner_reward",
        "project_winner_delta",
        "project_parent_advantage",
        "project_baseline_es",
    }:
        project_names = sorted(
            {
                project_name
                for candidate_reward in candidate_rewards
                for project_name in candidate_reward.get("project_metrics", {})
            }
        )
        if not project_names:
            raise ValueError(f"{args.zorl_update_strategy} requires per-project candidate metrics")

        candidate_specs = {candidate["candidate_id"]: candidate for candidate in generation["candidates"]}
        candidate_scores = {}
        project_rows = []
        for project_name in project_names:
            project_candidates = []
            for candidate_reward in candidate_rewards:
                project_metric = candidate_reward.get("project_metrics", {}).get(project_name)
                if project_metric is None:
                    continue
                project_candidates.append((candidate_reward, project_metric))
            if not project_candidates:
                raise ValueError(f"Missing project metrics for project {project_name!r}")

            winner_reward, winner_metric = max(project_candidates, key=lambda item: _project_metric_sort_key(item[1]))
            candidate_id = str(winner_reward["candidate_id"])
            score = float(winner_metric["reward"])
            parent_metric = _parent_project_metric(parent_baseline, project_name)
            parent_score = 0.0 if parent_metric is None else float(parent_metric.get("reward", 0.0) or 0.0)
            advantage = score - parent_score
            update_score = advantage if args.zorl_update_strategy == "project_parent_advantage" else score
            candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + score
            project_rows.append(
                {
                    "project": project_name,
                    "candidate_id": candidate_id,
                    "reward": score,
                    "parent_reward": parent_score,
                    "advantage": advantage,
                    "update_score": update_score,
                    "rollout_prefix_ratio": float(winner_metric.get("rollout_prefix_ratio", 0.0) or 0.0),
                    "rollout_exact_match": bool(winner_metric.get("rollout_exact_match", False)),
                    "rollout_exact_rate": _project_metric_exact_rate(winner_metric),
                }
            )

        if args.zorl_update_strategy == "project_baseline_es":
            candidate_scores = {str(candidate_reward["candidate_id"]): 0.0 for candidate_reward in candidate_rewards}
            candidate_project_counts = {
                str(candidate_reward["candidate_id"]): 0 for candidate_reward in candidate_rewards
            }
            project_std_values = []
            project_advantages = []
            project_parent_rewards = []
            project_max_rewards = []
            eps = 1e-6
            for project_name in project_names:
                project_candidates = []
                for candidate_reward in candidate_rewards:
                    project_metric = candidate_reward.get("project_metrics", {}).get(project_name)
                    if project_metric is None:
                        continue
                    project_candidates.append((candidate_reward, project_metric))
                if not project_candidates:
                    continue

                parent_metric = _parent_project_metric(parent_baseline, project_name)
                parent_score = 0.0 if parent_metric is None else float(parent_metric.get("reward", 0.0) or 0.0)
                candidate_values = [
                    float(project_metric.get("reward", 0.0) or 0.0)
                    for _candidate_reward, project_metric in project_candidates
                ]
                # Match the useful baseline-subtraction structure from
                # HyperscaleES: include two zero-noise parent anchors in the
                # per-project scale estimate, then update with all +/- pairs.
                scale_values = candidate_values + [parent_score, parent_score]
                mean_value = sum(scale_values) / len(scale_values)
                variance = sum((value - mean_value) ** 2 for value in scale_values) / len(scale_values)
                std_value = max(variance**0.5, eps)
                project_std_values.append(std_value)
                project_parent_rewards.append(parent_score)
                project_max_rewards.append(max(candidate_values))
                project_advantages.append(max(0.0, max(candidate_values) - parent_score))

                for candidate_reward, project_metric in project_candidates:
                    candidate_id = str(candidate_reward["candidate_id"])
                    score = (float(project_metric.get("reward", 0.0) or 0.0) - parent_score) / std_value
                    candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + score
                    candidate_project_counts[candidate_id] = candidate_project_counts.get(candidate_id, 0) + 1

            rewards_for_update = []
            for candidate_reward in candidate_rewards:
                candidate_id = str(candidate_reward["candidate_id"])
                project_count = int(candidate_project_counts.get(candidate_id, 0))
                if project_count <= 0:
                    continue
                rewards_for_update.append(
                    {
                        "candidate_id": candidate_id,
                        "reward_mean": float(candidate_scores[candidate_id]) / float(project_count),
                        "num_rollouts": 1,
                    }
                )
            if not rewards_for_update:
                best_candidate = max(candidate_rewards, key=_candidate_sort_key)
                rewards_for_update = [
                    {
                        "candidate_id": str(best_candidate["candidate_id"]),
                        "reward_mean": 0.0,
                        "num_rollouts": int(best_candidate.get("num_rollouts", 1) or 1),
                    }
                ]
                metadata["zero_signal_update"] = True
            metadata["selected_pairs"] = int(generation["num_pairs"])
            metadata["selected_projects"] = len(project_std_values)
            if project_std_values:
                metadata["selected_project_score_std_mean"] = sum(project_std_values) / len(project_std_values)
                metadata["selected_project_score_std_min"] = min(project_std_values)
                metadata["selected_project_score_std_max"] = max(project_std_values)
                metadata["selected_project_parent_reward_mean"] = sum(project_parent_rewards) / len(
                    project_parent_rewards
                )
                metadata["selected_project_reward_mean"] = sum(project_max_rewards) / len(project_max_rewards)
                metadata["selected_project_advantage_mean"] = sum(project_advantages) / len(project_advantages)
                metadata["selected_project_improved"] = sum(1 for value in project_advantages if value > 0.0)
            if rewards_for_update:
                metadata["selected_candidate_score_abs_mean"] = sum(
                    abs(float(item["reward_mean"])) for item in rewards_for_update
                ) / len(rewards_for_update)
                metadata["selected_candidate_score_abs_max"] = max(
                    abs(float(item["reward_mean"])) for item in rewards_for_update
                )
            apply_lr = args.zorl_b_sigma * args.zorl_candidate_delta_step_scale
            metadata["score_normalization"] = "none"
            metadata["apply_lr"] = apply_lr
            return _set_zorl_score_normalization(rewards_for_update, "none"), apply_lr, metadata

        if args.zorl_update_strategy == "project_parent_advantage":
            candidate_scores = {}
            for row in project_rows:
                if float(row["advantage"]) <= 0.0:
                    continue
                candidate_id = row["candidate_id"]
                candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + float(row["advantage"])

            if candidate_scores:
                selected_pair_count = len(
                    {int(candidate_specs[candidate_id]["perturbation_index"]) for candidate_id in candidate_scores}
                )
                score_sum = sum(float(score) for score in candidate_scores.values())
                rewards_for_update = [
                    {
                        "candidate_id": candidate_id,
                        "reward_mean": float(selected_pair_count) * float(score) / score_sum,
                        "num_rollouts": 1,
                    }
                    for candidate_id, score in sorted(candidate_scores.items())
                ]
                metadata["selected_project_advantage_sum"] = score_sum
                metadata["selected_project_advantage_mean"] = score_sum / max(len(project_rows), 1)
                metadata["selected_project_improved"] = sum(1 for row in project_rows if float(row["advantage"]) > 0.0)
                metadata["selected_candidate_weight_max"] = max(
                    float(score) / score_sum for score in candidate_scores.values()
                )
            else:
                best_candidate = max(candidate_rewards, key=_candidate_sort_key)
                rewards_for_update = [
                    {
                        "candidate_id": str(best_candidate["candidate_id"]),
                        "reward_mean": 0.0,
                        "num_rollouts": int(best_candidate.get("num_rollouts", 1) or 1),
                    }
                ]
                metadata["selected_project_advantage_sum"] = 0.0
                metadata["selected_project_advantage_mean"] = 0.0
                metadata["selected_project_improved"] = 0
                metadata["selected_candidate_weight_max"] = 0.0
                metadata["zero_signal_update"] = True
            apply_lr = args.zorl_b_sigma * args.zorl_candidate_delta_step_scale
        elif args.zorl_update_strategy == "project_winner_delta":
            positive_scores = {
                candidate_id: score for candidate_id, score in candidate_scores.items() if float(score) > 0.0
            }
            if positive_scores:
                candidate_scores = positive_scores
                selected_pair_count = len(
                    {int(candidate_specs[candidate_id]["perturbation_index"]) for candidate_id in candidate_scores}
                )
                score_sum = sum(float(score) for score in candidate_scores.values())
                rewards_for_update = [
                    {
                        "candidate_id": candidate_id,
                        "reward_mean": float(selected_pair_count) * float(score) / score_sum,
                        "num_rollouts": 1,
                    }
                    for candidate_id, score in sorted(candidate_scores.items())
                ]
                metadata["selected_project_reward_sum"] = score_sum
                metadata["selected_candidate_weight_max"] = max(
                    float(score) / score_sum for score in candidate_scores.values()
                )
            else:
                best_candidate = max(candidate_rewards, key=_candidate_sort_key)
                rewards_for_update = [
                    {
                        "candidate_id": str(best_candidate["candidate_id"]),
                        "reward_mean": 0.0,
                        "num_rollouts": int(best_candidate.get("num_rollouts", 1) or 1),
                    }
                ]
                metadata["selected_project_reward_sum"] = 0.0
                metadata["selected_candidate_weight_max"] = 0.0
                metadata["zero_signal_update"] = True
            apply_lr = args.zorl_b_sigma * args.zorl_candidate_delta_step_scale
        else:
            rewards_for_update = [
                {
                    "candidate_id": candidate_id,
                    "reward_mean": float(score),
                    "num_rollouts": 1,
                }
                for candidate_id, score in sorted(candidate_scores.items())
            ]
            apply_lr = float(lr) * args.zorl_update_lr_scale
        metadata["selected_pairs"] = len(rewards_for_update)
        metadata["selected_projects"] = len(project_rows)
        metadata["selected_project_reward_mean"] = sum(row["reward"] for row in project_rows) / len(project_rows)
        if args.zorl_update_strategy == "project_parent_advantage":
            metadata["selected_project_parent_reward_mean"] = sum(row["parent_reward"] for row in project_rows) / len(
                project_rows
            )
        metadata["selected_project_prefix_mean"] = sum(row["rollout_prefix_ratio"] for row in project_rows) / len(
            project_rows
        )
        metadata["selected_project_exact_mean"] = sum(row["rollout_exact_rate"] for row in project_rows) / len(
            project_rows
        )
        metadata["score_normalization"] = "none"
        metadata["apply_lr"] = apply_lr
        return _set_zorl_score_normalization(rewards_for_update, "none"), apply_lr, metadata

    pair_rows = _collect_pair_rows(generation, candidate_rewards)
    selected_rows = pair_rows

    if args.zorl_update_strategy in {"elite_pair_delta", "elite_winner_reward"}:
        elite_pairs = min(args.zorl_elite_pairs, len(pair_rows))
        sort_key = "abs_delta" if args.zorl_update_strategy == "elite_pair_delta" else "winner_reward_mean"
        selected_rows = sorted(pair_rows, key=lambda row: float(row[sort_key]), reverse=True)[:elite_pairs]
        metadata["elite_sort_key"] = sort_key

    if args.zorl_update_strategy == "winner_sign":
        rewards_for_update = _encode_pair_winners(selected_rows, score_fn=lambda _row: 1.0)
    elif args.zorl_update_strategy == "elite_pair_delta":
        rewards_for_update = _encode_pair_winners(selected_rows, score_fn=lambda row: row["abs_delta"])
    elif args.zorl_update_strategy == "elite_winner_reward":
        rewards_for_update = _encode_pair_winners(selected_rows, score_fn=lambda row: row["winner_reward_mean"])
    else:
        raise ValueError(f"Unsupported ZORL update strategy: {args.zorl_update_strategy}")

    metadata["selected_pairs"] = len(selected_rows)
    metadata["selected_pair_abs_delta_mean"] = sum(float(row["abs_delta"]) for row in selected_rows) / len(
        selected_rows
    )
    metadata["selected_pair_winner_reward_mean"] = sum(float(row["winner_reward_mean"]) for row in selected_rows) / len(
        selected_rows
    )
    metadata["selected_pair_winner_exact_mean"] = sum(float(row["winner_exact_count"]) for row in selected_rows) / len(
        selected_rows
    )
    metadata["score_normalization"] = "none"
    apply_lr = float(lr) * args.zorl_update_lr_scale
    metadata["apply_lr"] = apply_lr
    return _set_zorl_score_normalization(rewards_for_update, "none"), apply_lr, metadata


def _resolve_zorl_max_update_norm(args, update_metadata):
    base_norm = getattr(args, "zorl_max_update_norm", None)
    if base_norm is None:
        return None, {}

    base_norm = float(base_norm)
    if not getattr(args, "zorl_adaptive_update_norm", False):
        return base_norm, {}

    signal = update_metadata.get("selected_project_advantage_mean")
    metadata = {"adaptive_update_norm_base": base_norm}
    if signal is None:
        metadata["adaptive_update_norm_skipped"] = True
        metadata["adaptive_update_norm_reason"] = "missing_selected_project_advantage_mean"
        return base_norm, metadata

    reference = float(getattr(args, "zorl_adaptive_update_norm_reference", 1.0))
    min_scale = float(getattr(args, "zorl_adaptive_update_norm_min_scale", 1.0))
    max_scale = float(getattr(args, "zorl_adaptive_update_norm_max_scale", 1.0))
    signal = max(0.0, float(signal))
    raw_scale = signal / reference
    scale = min(max(raw_scale, min_scale), max_scale)
    effective_norm = base_norm * scale
    metadata.update(
        {
            "adaptive_update_norm_signal": signal,
            "adaptive_update_norm_reference": reference,
            "adaptive_update_norm_raw_scale": raw_scale,
            "adaptive_update_norm_scale": scale,
            "adaptive_update_norm_effective": effective_norm,
        }
    )
    return effective_norm, metadata


def _resolve_zorl_generation_num_pairs(args):
    base_pairs = int(args.zorl_num_pairs)
    multiplier = float(getattr(args, "zorl_adaptive_num_pairs_multiplier", 1.0))
    threshold = int(getattr(args, "zorl_adaptive_num_pairs_active_threshold", 0))
    max_pairs = int(getattr(args, "zorl_adaptive_num_pairs_max", 0))
    metadata = {
        "base_num_pairs": base_pairs,
        "adaptive_num_pairs": base_pairs,
        "adaptive_num_pairs_multiplier": multiplier,
        "adaptive_num_pairs_active_threshold": threshold,
        "adaptive_num_pairs_max": max_pairs,
    }
    if multiplier <= 1.0:
        metadata["adaptive_num_pairs_reason"] = "disabled"
        return base_pairs, metadata

    previous_active_projects = getattr(args, "_zorl_previous_active_projects", None)
    if previous_active_projects is None:
        metadata["adaptive_num_pairs_reason"] = "no_previous_active_projects"
        return base_pairs, metadata

    previous_active_projects = int(previous_active_projects)
    metadata["adaptive_previous_active_projects"] = previous_active_projects
    if previous_active_projects > threshold:
        metadata["adaptive_num_pairs_reason"] = "active_projects_above_threshold"
        return base_pairs, metadata

    target_pairs = int(math.ceil(base_pairs * multiplier))
    if max_pairs > 0:
        target_pairs = min(target_pairs, max_pairs)
    target_pairs = max(base_pairs, target_pairs)
    metadata["adaptive_num_pairs"] = target_pairs
    metadata["adaptive_num_pairs_reason"] = "active_projects_at_or_below_threshold"
    return target_pairs, metadata


def _resolve_zorl_hard_project_score_args(args, parent_baseline_metadata):
    base_rollouts = max(1, int(getattr(args, "zorl_rollouts_per_puzzle", 1)))
    multiplier = float(getattr(args, "zorl_hard_project_rollout_multiplier", 1.0))
    threshold = int(getattr(args, "zorl_hard_project_rollout_active_threshold", 0))
    max_rollouts = int(getattr(args, "zorl_hard_project_rollout_max", 0))
    metadata = {
        "hard_project_base_rollouts_per_puzzle": base_rollouts,
        "hard_project_rollouts_per_puzzle": base_rollouts,
        "hard_project_rollout_multiplier": multiplier,
        "hard_project_rollout_active_threshold": threshold,
        "hard_project_rollout_max": max_rollouts,
    }
    if multiplier <= 1.0:
        metadata["hard_project_rollout_reason"] = "disabled"
        return args, metadata

    active_projects = parent_baseline_metadata.get("parent_advantage_active_projects")
    if active_projects is None:
        metadata["hard_project_rollout_reason"] = "no_active_project_signal"
        return args, metadata

    active_projects = int(active_projects)
    metadata["hard_project_rollout_active_projects"] = active_projects
    total_projects = parent_baseline_metadata.get("parent_advantage_total_projects")
    if total_projects is not None:
        metadata["hard_project_rollout_total_projects"] = int(total_projects)
    if active_projects > threshold:
        metadata["hard_project_rollout_reason"] = "active_projects_above_threshold"
        return args, metadata

    target_rollouts = int(math.ceil(base_rollouts * multiplier))
    if max_rollouts > 0:
        target_rollouts = min(target_rollouts, max_rollouts)
    target_rollouts = max(base_rollouts, target_rollouts)
    score_args = argparse.Namespace(**vars(args))
    score_args.zorl_rollouts_per_puzzle = target_rollouts
    metadata["hard_project_rollouts_per_puzzle"] = target_rollouts
    metadata["hard_project_rollout_effective_multiplier"] = target_rollouts / float(base_rollouts)
    metadata["hard_project_rollout_reason"] = "active_projects_at_or_below_threshold"
    return score_args, metadata


def _select_rotated_puzzle_subset(reward_data, args, *, generation_index):
    """Sample --zorl-active-puzzle-count puzzles from `reward_data` using a
    deterministic per-generation seed. Returns (subset, metadata). When the
    flag is 0 or >= len(reward_data), returns reward_data unchanged.

    Rotation gives ZORL gradient variance across generations even when the
    parent already solves the same N puzzles every gen — instead of staying
    locked on a fixed solved/unsolved partition, each gen sees a different
    mix, so the search keeps probing directions that help unseen puzzles.
    """
    active_count = int(getattr(args, "zorl_active_puzzle_count", 0) or 0)
    metadata = {
        "puzzle_pool_size": len(reward_data),
        "puzzle_active_count": len(reward_data),
        "puzzle_rotation_seed": int(getattr(args, "zorl_puzzle_rotation_seed", 0) or 0),
    }
    if active_count <= 0 or active_count >= len(reward_data):
        metadata["puzzle_rotation_reason"] = "disabled"
        return list(reward_data), metadata

    base_seed = int(getattr(args, "zorl_puzzle_rotation_seed", 0) or 0)
    # Mix base_seed with generation_index into a single int so the rng accepts
    # it directly (random.Random doesn't accept tuple seeds).
    seed = (base_seed * 1_000_003 + int(generation_index)) & 0x7FFFFFFFFFFFFFFF
    rng = random.Random(seed)
    indices = list(range(len(reward_data)))
    rng.shuffle(indices)
    selected = sorted(indices[:active_count])
    subset = [reward_data[i] for i in selected]
    metadata["puzzle_active_count"] = len(subset)
    metadata["puzzle_active_project_names"] = ",".join(example["project"] for example in subset)
    metadata["puzzle_rotation_reason"] = "rotated"
    return subset, metadata


def _resolve_zorl_hard_project_reward_set(args, parent_baseline_metadata):
    """Decide which projects get a boosted dense-reward weight this generation.

    Mirrors the gating shape of `_resolve_zorl_hard_project_score_args` so a
    single set of active-project thresholds can drive both knobs. Returns
    (hard_project_set, multiplier, metadata). The reward multiplier only
    applies to the dense components (prefix/char/valid/value) — the exact-match
    bonus is left untouched so a clean win is worth the same absolute reward
    as before, and the multiplier just sharpens the climbable slope below it.
    """
    multiplier = float(getattr(args, "zorl_hard_project_reward_multiplier", 1.0))
    threshold = int(getattr(args, "zorl_hard_project_reward_active_threshold", 0))
    metadata = {
        "hard_project_reward_multiplier": multiplier,
        "hard_project_reward_active_threshold": threshold,
    }
    if multiplier <= 1.0:
        metadata["hard_project_reward_reason"] = "disabled"
        return set(), 1.0, metadata

    active_names_csv = parent_baseline_metadata.get("parent_advantage_active_project_names")
    if not active_names_csv:
        metadata["hard_project_reward_reason"] = "no_active_project_signal"
        return set(), 1.0, metadata

    active_names = [name for name in active_names_csv.split(",") if name]
    active_count = len(active_names)
    metadata["hard_project_reward_active_projects"] = active_count
    if active_count == 0:
        metadata["hard_project_reward_reason"] = "no_active_projects"
        return set(), 1.0, metadata
    if active_count > threshold:
        metadata["hard_project_reward_reason"] = "active_projects_above_threshold"
        return set(), 1.0, metadata

    hard_set = set(active_names)
    metadata["hard_project_reward_set_size"] = len(hard_set)
    metadata["hard_project_reward_set_names"] = ",".join(sorted(hard_set))
    metadata["hard_project_reward_reason"] = "active_projects_at_or_below_threshold"
    return hard_set, multiplier, metadata


def probe_parent_adapter(train_url, model_id, infer_url, reward_data, args, *, generation_index):
    score_args = args
    parent_probe_rollouts = getattr(args, "zorl_parent_probe_rollouts", None)
    parent_probe_temperature = getattr(args, "zorl_parent_probe_temperature", None)
    if parent_probe_rollouts is not None or parent_probe_temperature is not None:
        score_args = argparse.Namespace(**vars(args))
        if parent_probe_rollouts is not None:
            score_args.zorl_rollouts_per_puzzle = max(1, int(parent_probe_rollouts))
        if parent_probe_temperature is not None:
            score_args.zorl_rollout_temperature = float(parent_probe_temperature)

    if args.zorl_candidate_backend == "sglang_native":
        sampler_result = {"path": "__sglang_native__"}
        lora_name = args.zorl_native_parent_lora_name
        for url in args.infer_url:
            flush_inference_cache(url)
    else:
        probe_name = f"zorl-parent-probes/{model_id}/g{generation_index:06d}-{time.time_ns()}"
        sampler_result = save_weights_for_sampler(train_url, model_id, probe_name)
        sampling_session = create_sampling_session(train_url, model_id, sampler_result["path"])
        lora_name = sampling_session["lora_name"]

    parent_candidate = {
        "candidate_id": f"parent-g{generation_index:06d}",
        "lora_name": lora_name,
    }
    candidate_rewards, reward_stats = score_zorl_candidates(infer_url, [parent_candidate], reward_data, score_args)

    # Reuse rollout results from scoring when available to avoid a redundant inference call.
    parent_project_metrics = candidate_rewards[0].get("project_metrics", {})
    has_scoring_rollouts = args.zorl_reward_mode in {"rollout", "hybrid"} and any(
        parent_project_metrics.get(example["project"], {}).get("generated_text") for example in reward_data
    )

    outputs = []
    exact_count = 0
    prefix_total = 0.0
    if has_scoring_rollouts:
        for example in reward_data:
            pm = parent_project_metrics.get(example["project"], {})
            generated = pm.get("generated_text", "")
            pr = float(pm.get("rollout_prefix_ratio", 0.0))
            cr = float(pm.get("rollout_char_match_ratio", 0.0))
            em = bool(pm.get("rollout_exact_match", False))
            exact_count += int(em)
            prefix_total += pr
            outputs.append(
                {
                    "project": example["project"],
                    "expected": example["code"],
                    "generated": generated,
                    "prefix_ratio": pr,
                    "char_match_ratio": cr,
                    "exact_match": em,
                }
            )
    else:
        fallback_url = infer_url[0] if isinstance(infer_url, list) else infer_url
        rollout_results = generate_rollouts(
            fallback_url,
            input_ids_batch=[example["prompt_ids"] for example in reward_data],
            lora_paths=[lora_name for _ in reward_data],
            max_new_tokens=_default_rollout_max_new_tokens(reward_data, args),
            prompt_messages_batch=[example["prompt_messages"] for example in reward_data],
            served_model_name=args.served_model_name or args.model,
            api_format=score_args.zorl_inference_api_format,
            enable_thinking=bool(getattr(args, "zorl_enable_thinking", False)),
        )
        for example, rollout_result in zip(reward_data, rollout_results, strict=True):
            rollout_components = _rollout_reward_components(rollout_result, example)
            exact_count += int(rollout_components["exact_match"])
            prefix_total += float(rollout_components["prefix_ratio"])
            outputs.append(
                {
                    "project": example["project"],
                    "expected": example["code"],
                    "generated": rollout_components["generated_text"],
                    "prefix_ratio": float(rollout_components["prefix_ratio"]),
                    "char_match_ratio": float(rollout_components["char_match_ratio"]),
                    "exact_match": bool(rollout_components["exact_match"]),
                }
            )

    scored_rollout_exact_rate = reward_stats.get("rollout_exact_rate")
    reward_stats["rollout_prefix_mean"] = prefix_total / len(outputs)
    reward_stats["rollout_exact_rate"] = exact_count / len(outputs)
    if scored_rollout_exact_rate is not None and parent_probe_rollouts is not None:
        exact_count = float(scored_rollout_exact_rate) * len(outputs)
        reward_stats["rollout_exact_rate"] = float(scored_rollout_exact_rate)

    return {
        "generation_index": generation_index,
        "sampler_path": sampler_result["path"],
        "lora_name": lora_name,
        "reward_mean": float(candidate_rewards[0]["reward_mean"]),
        "reward_stats": reward_stats,
        "exact_count": exact_count,
        "outputs": outputs,
    }


def score_parent_baseline_for_update(train_url, model_id, infer_url, reward_data, args, *, generation_index):
    score_args = args
    baseline_rollouts = getattr(args, "zorl_parent_baseline_rollouts", None)
    if baseline_rollouts is not None:
        score_args = argparse.Namespace(**vars(args))
        score_args.zorl_rollouts_per_puzzle = max(1, int(baseline_rollouts))

    if args.zorl_candidate_backend == "sglang_native":
        sampler_path = "__sglang_native__"
        lora_name = args.zorl_native_parent_lora_name
        for url in args.infer_url:
            flush_inference_cache(url)
    else:
        baseline_name = f"zorl-parent-baseline/{model_id}/g{generation_index:06d}-{time.time_ns()}"
        sampler_result = save_weights_for_sampler(train_url, model_id, baseline_name)
        sampler_path = sampler_result["path"]
        sampling_session = create_sampling_session(train_url, model_id, sampler_path)
        lora_name = sampling_session["lora_name"]

    parent_candidate = {
        "candidate_id": f"parent-baseline-g{generation_index:06d}",
        "lora_name": lora_name,
    }
    candidate_rewards, reward_stats = score_zorl_candidates(
        infer_url,
        [parent_candidate],
        reward_data,
        score_args,
    )
    parent_reward = candidate_rewards[0]
    return {
        "generation_index": generation_index,
        "sampler_path": sampler_path,
        "lora_name": lora_name,
        "reward": parent_reward,
        "reward_mean": float(parent_reward["reward_mean"]),
        "reward_stats": reward_stats,
    }


def _parent_probe_key(parent_probe):
    reward_stats = parent_probe.get("reward_stats") or {}
    return (
        float(parent_probe["exact_count"]),
        float(parent_probe["reward_mean"]),
        float(reward_stats.get("rollout_prefix_mean", 0.0) or 0.0),
        float(reward_stats.get("rollout_char_match_mean", 0.0) or 0.0),
    )


def _rescore_top_zorl_candidates(infer_url, candidates, candidate_rewards, reward_data, args):
    top_k = int(getattr(args, "zorl_rescore_top_k", 0) or 0)
    rescore_rollouts = getattr(args, "zorl_rescore_rollouts", None)
    rescore_strategies = {
        "best_candidate",
        "project_winner_reward",
        "project_winner_delta",
        "project_parent_advantage",
    }
    if args.zorl_update_strategy not in rescore_strategies or top_k <= 0 or rescore_rollouts is None:
        return candidate_rewards, {}

    base_rollouts = max(1, int(getattr(args, "zorl_rollouts_per_puzzle", 1)))
    rescore_rollouts = max(1, int(rescore_rollouts))
    if rescore_rollouts <= base_rollouts:
        return candidate_rewards, {}

    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    top_rewards = []
    top_candidate_ids = []
    initial_project_rows = []
    if args.zorl_update_strategy == "best_candidate":
        top_rewards = sorted(candidate_rewards, key=_candidate_sort_key, reverse=True)[:top_k]
        top_candidate_ids = [str(row["candidate_id"]) for row in top_rewards]
    else:
        project_names = sorted(
            {
                project_name
                for candidate_reward in candidate_rewards
                for project_name in candidate_reward.get("project_metrics", {})
            }
        )
        if not project_names:
            return candidate_rewards, {}
        seen_candidate_ids = set()
        for project_name in project_names:
            project_candidates = [
                (candidate_reward, candidate_reward["project_metrics"][project_name])
                for candidate_reward in candidate_rewards
                if project_name in candidate_reward.get("project_metrics", {})
            ]
            ranked_project_candidates = sorted(
                project_candidates,
                key=lambda item: _project_metric_sort_key(item[1]),
                reverse=True,
            )
            if not ranked_project_candidates:
                continue
            winner_reward, winner_metric = ranked_project_candidates[0]
            initial_project_rows.append(
                {
                    "project": project_name,
                    "candidate_id": str(winner_reward["candidate_id"]),
                    "reward": float(winner_metric.get("reward", 0.0) or 0.0),
                    "rollout_exact_rate": _project_metric_exact_rate(winner_metric),
                }
            )
            for candidate_reward, _project_metric in ranked_project_candidates[:top_k]:
                candidate_id = str(candidate_reward["candidate_id"])
                if candidate_id in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(candidate_id)
                top_candidate_ids.append(candidate_id)

    top_candidates = [
        candidate_by_id[candidate_id] for candidate_id in top_candidate_ids if candidate_id in candidate_by_id
    ]
    if not top_candidates:
        return candidate_rewards, {}

    initial_best = (
        max(top_rewards, key=_candidate_sort_key) if top_rewards else max(candidate_rewards, key=_candidate_sort_key)
    )
    score_args = argparse.Namespace(**vars(args))
    score_args.zorl_rollouts_per_puzzle = rescore_rollouts
    rescore_start_t = time.time()
    rescored_rewards, rescored_stats = score_zorl_candidates(
        infer_url,
        top_candidates,
        reward_data,
        score_args,
    )
    rescore_s = time.time() - rescore_start_t
    rescored_best = max(rescored_rewards, key=_candidate_sort_key)
    rescore_rollout_count = 0
    if args.zorl_reward_mode in {"rollout", "hybrid"}:
        rescore_rollout_count = len(top_candidates) * len(reward_data) * rescore_rollouts

    metadata = {
        "time_rescore_s": float(rescore_s),
        "rescore_top_k": int(top_k),
        "rescore_candidate_count": int(len(top_candidates)),
        "rescore_rollouts_per_puzzle": int(rescore_rollouts),
        "rescore_score_rollouts": int(rescore_rollout_count),
        "rescore_initial_best_candidate_id": str(initial_best["candidate_id"]),
        "rescore_initial_best_exact_count": float(initial_best.get("rollout_exact_count", 0.0) or 0.0),
        "rescore_initial_best_reward": float(initial_best["reward_mean"]),
        "rescore_best_candidate_id": str(rescored_best["candidate_id"]),
        "rescore_best_exact_count": float(rescored_best.get("rollout_exact_count", 0.0) or 0.0),
        "rescore_best_reward": float(rescored_best["reward_mean"]),
    }
    if rescore_s > 0 and rescore_rollout_count > 0:
        metadata["rescore_rollouts_per_s"] = rescore_rollout_count / rescore_s
    if args.zorl_update_strategy in {"project_winner_reward", "project_winner_delta"}:
        project_names = sorted(
            {
                project_name
                for candidate_reward in rescored_rewards
                for project_name in candidate_reward.get("project_metrics", {})
            }
        )
        rescored_project_rows = []
        for project_name in project_names:
            project_candidates = [
                (candidate_reward, candidate_reward["project_metrics"][project_name])
                for candidate_reward in rescored_rewards
                if project_name in candidate_reward.get("project_metrics", {})
            ]
            if not project_candidates:
                continue
            winner_reward, winner_metric = max(project_candidates, key=lambda item: _project_metric_sort_key(item[1]))
            rescored_project_rows.append(
                {
                    "project": project_name,
                    "candidate_id": str(winner_reward["candidate_id"]),
                    "reward": float(winner_metric.get("reward", 0.0) or 0.0),
                    "rollout_exact_rate": _project_metric_exact_rate(winner_metric),
                }
            )
        if initial_project_rows:
            metadata["rescore_initial_project_exact_mean"] = sum(
                row["rollout_exact_rate"] for row in initial_project_rows
            ) / len(initial_project_rows)
            metadata["rescore_initial_project_reward_mean"] = sum(row["reward"] for row in initial_project_rows) / len(
                initial_project_rows
            )
        if rescored_project_rows:
            metadata["rescore_project_count"] = int(len(rescored_project_rows))
            metadata["rescore_project_exact_mean"] = sum(
                row["rollout_exact_rate"] for row in rescored_project_rows
            ) / len(rescored_project_rows)
            metadata["rescore_project_reward_mean"] = sum(row["reward"] for row in rescored_project_rows) / len(
                rescored_project_rows
            )
    for key, value in rescored_stats.items():
        metadata[f"rescore_{key}"] = value
    return rescored_rewards, metadata


def run_zorl_generation(train_url, model_id, infer_url, reward_data, args, lr, *, generation_index=0):
    generation_start_t = time.time()
    generation_pair_metadata = {}
    if args.zorl_candidate_backend == "sglang_native":
        generation_num_pairs, generation_pair_metadata = _resolve_zorl_generation_num_pairs(args)
        generation = start_sglang_zorl_generation(
            args.infer_url,
            session_id=args.zorl_native_session_id,
            preload_candidates=args.zorl_preload_candidates,
            num_pairs=generation_num_pairs,
        )
    else:
        generation = start_zorl_generation(
            train_url,
            model_id,
            preload_sampling=True,
            candidate_backend=args.zorl_candidate_backend,
        )
    start_generation_s = time.time() - generation_start_t
    generation_id = generation["generation_id"]

    try:
        if not generation.get("sampling_ready"):
            raise RuntimeError(
                f"ZORL generation {generation_id} did not preload sampling adapters. "
                "Check that inference endpoints are registered and LoRA is enabled on SGLang."
            )
        parent_baseline = None
        parent_baseline_metadata = {}
        rotated_reward_data, puzzle_rotation_metadata = _select_rotated_puzzle_subset(
            reward_data, args, generation_index=generation_index
        )
        score_reward_data = rotated_reward_data
        strategy_needs_baseline = args.zorl_update_strategy in {
            "project_parent_advantage",
            "project_baseline_es",
        }
        # Decoupled from --zorl-parent-advantage-filter-solved: we compute the
        # baseline (and the active-project metadata) whenever the strategy needs
        # it OR whenever the hard-reward shaping needs the active-project list.
        # Only the actual "drop solved projects from the gradient" step is
        # gated on --zorl-parent-advantage-filter-solved.
        hard_reward_needs_active_set = (
            float(getattr(args, "zorl_hard_project_reward_multiplier", 1.0) or 1.0) > 1.0
        )
        compute_baseline_early = strategy_needs_baseline and (
            args.zorl_parent_advantage_filter_solved or hard_reward_needs_active_set
        )
        if compute_baseline_early:
            parent_baseline_start_t = time.time()
            parent_baseline = score_parent_baseline_for_update(
                train_url,
                model_id,
                infer_url,
                rotated_reward_data,
                args,
                generation_index=generation_index,
            )
            parent_baseline_stats = parent_baseline["reward_stats"]
            parent_baseline_metadata.update(
                {
                    "time_parent_baseline_s": time.time() - parent_baseline_start_t,
                    "parent_baseline_reward_mean": parent_baseline["reward_mean"],
                    "parent_baseline_exact_rate": parent_baseline_stats.get("rollout_exact_rate", 0.0),
                    "parent_baseline_prefix_mean": parent_baseline_stats.get("rollout_prefix_mean", 0.0),
                }
            )
            filtered_reward_data, filter_metadata = _filter_parent_advantage_reward_data(
                rotated_reward_data,
                parent_baseline["reward"],
                solved_exact_rate=args.zorl_parent_advantage_solved_exact_rate,
            )
            parent_baseline_metadata.update(filter_metadata)
            if args.zorl_parent_advantage_filter_solved:
                score_reward_data = filtered_reward_data
            # else: leave score_reward_data = rotated_reward_data so solved
            # projects keep contributing reinforcement signal, while the
            # filter_metadata (active_project_names, etc.) still drives the
            # hard-reward shaping decision below.

        score_args, hard_project_rollout_metadata = _resolve_zorl_hard_project_score_args(
            args, parent_baseline_metadata
        )
        hard_reward_names, hard_reward_multiplier, hard_reward_metadata = (
            _resolve_zorl_hard_project_reward_set(args, parent_baseline_metadata)
        )
        score_args = argparse.Namespace(**vars(score_args))
        score_args._zorl_hard_project_reward_names = hard_reward_names
        score_args._zorl_hard_project_reward_multiplier = hard_reward_multiplier
        score_start_t = time.time()
        candidate_rewards, reward_stats = score_zorl_candidates(
            infer_url,
            generation["candidates"],
            score_reward_data,
            score_args,
        )
        reward_stats.update(generation_pair_metadata)
        reward_stats.update(hard_project_rollout_metadata)
        reward_stats.update(hard_reward_metadata)
        reward_stats.update(puzzle_rotation_metadata)
        reward_stats.update(parent_baseline_metadata)
        score_s = time.time() - score_start_t
        candidate_rewards, rescore_metadata = _rescore_top_zorl_candidates(
            infer_url,
            generation["candidates"],
            candidate_rewards,
            score_reward_data,
            args,
        )
        reward_stats.update(rescore_metadata)
        if args.zorl_update_strategy in {"project_parent_advantage", "project_baseline_es"} and parent_baseline is None:
            parent_baseline_start_t = time.time()
            parent_baseline = score_parent_baseline_for_update(
                train_url,
                model_id,
                infer_url,
                rotated_reward_data,
                args,
                generation_index=generation_index,
            )
            parent_baseline_stats = parent_baseline["reward_stats"]
            reward_stats.update(
                {
                    "time_parent_baseline_s": time.time() - parent_baseline_start_t,
                    "parent_baseline_reward_mean": parent_baseline["reward_mean"],
                    "parent_baseline_exact_rate": parent_baseline_stats.get("rollout_exact_rate", 0.0),
                    "parent_baseline_prefix_mean": parent_baseline_stats.get("rollout_prefix_mean", 0.0),
                }
            )
        rewards_for_update, apply_lr, update_metadata = _select_zorl_update_payload(
            generation,
            candidate_rewards,
            args,
            lr,
            parent_baseline=parent_baseline["reward"] if parent_baseline is not None else None,
        )
        reward_stats.update(update_metadata)
        effective_max_update_norm, update_norm_metadata = _resolve_zorl_max_update_norm(args, update_metadata)
        reward_stats.update(update_norm_metadata)
        apply_start_t = time.time()
        if args.zorl_candidate_backend == "sglang_native":
            apply_result = apply_sglang_zorl_rewards(
                args.infer_url,
                session_id=args.zorl_native_session_id,
                generation_id=generation_id,
                candidate_rewards=rewards_for_update,
                lr=apply_lr,
                max_update_norm=effective_max_update_norm,
            )
        else:
            apply_result = apply_zorl_rewards(train_url, model_id, generation_id, rewards_for_update, apply_lr)
        apply_s = time.time() - apply_start_t

        candidate_count = len(generation["candidates"])
        reward_example_count = len(score_reward_data)
        candidate_examples = candidate_count * reward_example_count
        rollouts_per_puzzle = max(1, int(getattr(score_args, "zorl_rollouts_per_puzzle", 1)))
        rollout_count = 0
        if args.zorl_reward_mode in {"rollout", "hybrid"}:
            rollout_count = candidate_examples * rollouts_per_puzzle
        reward_stats.update(
            {
                "time_start_generation_s": start_generation_s,
                "time_score_s": score_s,
                "time_apply_s": apply_s,
                "time_generation_step_s": time.time() - generation_start_t,
                "candidate_count": candidate_count,
                "score_candidate_examples": candidate_examples,
                "score_rollouts": rollout_count,
            }
        )
        candidate_creation_metadata = generation.get("candidate_creation_metadata") or {}
        if candidate_creation_metadata:
            reward_stats["candidate_preload_time_s"] = float(candidate_creation_metadata.get("preload_time_s", 0.0))
            reward_stats["candidate_preloaded_count"] = int(
                candidate_creation_metadata.get("preloaded_candidate_count", 0)
            )
            reward_stats["candidate_virtual_count"] = int(candidate_creation_metadata.get("virtual_candidate_count", 0))
            reward_stats["candidate_materialized_count"] = int(
                candidate_creation_metadata.get("materialized_candidate_count", 0)
            )
            reward_stats["candidate_gpu_pool_loaded_loras"] = int(
                candidate_creation_metadata.get("gpu_pool_loaded_loras", 0)
            )
            reward_stats["candidate_gpu_pool_capacity"] = int(candidate_creation_metadata.get("gpu_pool_capacity", 0))
        if score_s > 0:
            reward_stats["score_candidate_examples_per_s"] = candidate_examples / score_s
            if rollout_count > 0:
                reward_stats["score_rollouts_per_s"] = rollout_count / score_s
        if "parent_advantage_active_projects" in reward_stats:
            args._zorl_previous_active_projects = int(reward_stats["parent_advantage_active_projects"])
        return generation, candidate_rewards, reward_stats, apply_result
    except Exception:
        try:
            if args.zorl_candidate_backend == "sglang_native":
                abort_sglang_zorl_generation(
                    args.infer_url,
                    session_id=args.zorl_native_session_id,
                    generation_id=generation_id,
                )
            else:
                abort_zorl_generation(train_url, model_id, generation_id)
        except Exception as abort_error:
            print(f"    WARN: failed to abort ZORL generation {generation_id}: {abort_error}")
        raise


def _save_checkpoint_if_requested(train_url, model_id, checkpoint_name, *, label):
    checkpoint = save_weights(train_url, model_id, checkpoint_name)
    print(f"      Saved {label} checkpoint: {checkpoint['path']}")
    return checkpoint


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_lr(step, num_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / num_steps))


def get_lr(step, args):
    """Return learning rate for the given global step (0-indexed)."""
    if args.lr_schedule == "constant":
        return args.lr
    if args.lr_schedule == "cosine":
        lr_min = args.lr * args.lr_min_ratio
        return cosine_lr(step, args.steps, lr_max=args.lr, lr_min=lr_min)
    if args.lr_schedule == "warmup_cosine":
        if step < args.warmup_steps:
            return args.lr
        cosine_step = step - args.warmup_steps
        cosine_total = args.steps - args.warmup_steps
        lr_min = args.lr * args.lr_min_ratio
        return cosine_lr(cosine_step, cosine_total, lr_max=args.lr, lr_min=lr_min)
    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def run_test(args, training_data, reward_data):
    infer_urls = args.infer_url
    reward_infer_urls = args.reward_infer_url or infer_urls
    train_model_id = args.train_model_id
    served_model_name = args.served_model_name or args.model
    total_codes = len(CODES) * len(infer_urls)
    best_parent_probe = None

    print("  Checking services...")
    if not wait_for_training_service(args.train_url, timeout=args.train_startup_timeout):
        print("  FAILED: Training server not running")
        return None
    for url in infer_urls:
        if not wait_for_inference_service(url, timeout=args.infer_startup_timeout):
            print(f"  FAILED: {url} not running")
            return None
    for url in reward_infer_urls:
        if url not in infer_urls and not wait_for_inference_service(url, timeout=args.infer_startup_timeout):
            print(f"  FAILED: reward endpoint {url} not running")
            return None
    print("    All services ready.")

    create_result = create_model(args.train_url, args.model, args)
    print(f"    Model created: model_id={create_result.get('model_id', train_model_id)}")
    add_endpoints(args.train_url, infer_urls)
    if args.init_weights_path:
        load_result = load_weights(
            args.train_url,
            train_model_id,
            args.init_weights_path,
            optimizer=args.init_weights_optimizer,
        )
        print(f"    Loaded initial checkpoint: path={load_result['path']}, optimizer={args.init_weights_optimizer}")
    if args.run_baseline:
        test_inference(
            infer_urls,
            served_model_name,
            "baseline",
            enable_thinking=bool(getattr(args, "zorl_enable_thinking", False)),
            max_tokens=int(getattr(args, "zorl_rollout_max_new_tokens", None) or 64),
        )
    else:
        print("    Skipping baseline inference to avoid seeding stale prompt cache before sync.")

    if args.trainer_mode == "gradient":
        print(f"\n    Gradient training ({args.steps} steps, lr={args.lr}, schedule={args.lr_schedule})...")
        t0 = time.time()
        for step in range(args.steps):
            step_lr = get_lr(step, args)
            loss, grad_norm = train_step(args.train_url, train_model_id, training_data, step_lr)
            step_num = step + 1
            if step_num == 1 or step_num == args.steps or step_num % args.log_interval == 0:
                print(f"      Step {step_num}/{args.steps}: loss={loss}, grad_norm={grad_norm}, lr={step_lr:.2e}")
        print(f"    Training done in {time.time() - t0:.1f}s")
    elif args.trainer_mode == "grpo":
        print(
            f"\n    GRPO training ({args.steps} steps, lr={args.lr}, group_size={args.grpo_group_size}, "
            f"temperature={args.grpo_temperature}, max_new_tokens={args.grpo_max_new_tokens})..."
        )
        print(f"    Reward endpoints: {infer_urls}")
        t0 = time.time()
        for step in range(args.steps):
            step_lr = get_lr(step, args)
            try:
                metrics = grpo_train_step(
                    args.train_url,
                    train_model_id,
                    infer_urls=infer_urls,
                    reward_data=reward_data,
                    args=args,
                    lr=step_lr,
                )
            except Exception as exc:
                print(f"      GRPO step {step + 1}/{args.steps} FAILED: {type(exc).__name__}: {exc}")
                raise
            step_num = step + 1
            if step_num == 1 or step_num == args.steps or step_num % args.log_interval == 0:
                exact_rate = metrics.get("exact_match_rate", 0.0)
                exact_count = metrics.get("exact_match", 0)
                rollouts = metrics.get("rollouts", 0)
                skipped_zero = metrics.get("groups_skipped_zero_advantage", 0)
                datums = metrics.get("datums", 0)
                loss = metrics.get("loss", "N/A")
                grad_norm = metrics.get("grad_norm", "N/A")
                print(
                    f"      Step {step_num}/{args.steps}: "
                    f"loss={loss}, grad_norm={grad_norm}, lr={step_lr:.2e}, "
                    f"exact={exact_count}/{rollouts} ({exact_rate:.4f}), "
                    f"datums={datums}, skipped_zero_adv={skipped_zero}"
                )
        print(f"    GRPO training done in {time.time() - t0:.1f}s")
    else:
        reward_infer_url = reward_infer_urls
        print(f"\n    ZORL training ({args.steps} generations, lr={args.lr}, schedule={args.lr_schedule})...")
        print(
            f"    Reward endpoints: {reward_infer_url}, api_format={args.zorl_inference_api_format}, "
            f"candidate_backend={args.zorl_candidate_backend}, num_pairs={args.zorl_num_pairs}, "
            f"adaptive_pairs={args.zorl_adaptive_num_pairs_multiplier}x@"
            f"{args.zorl_adaptive_num_pairs_active_threshold}, "
            f"hard_rollouts={args.zorl_hard_project_rollout_multiplier}x@"
            f"{args.zorl_hard_project_rollout_active_threshold}, "
            f"score_batch_size={args.zorl_score_batch_size}, score_order={args.zorl_score_order}, "
            f"score_routing={args.zorl_score_routing}"
        )
        if args.zorl_candidate_backend == "sglang_native":
            if int(args.zorl_refresh_interval) != 0:
                raise ValueError("sglang_native currently requires --zorl-refresh-interval 0")
            if args.zorl_elitist_rollback and not args.zorl_parent_probe_interval:
                raise ValueError("--zorl-elitist-rollback requires --zorl-parent-probe-interval > 0")
            args.zorl_native_session_id = args.zorl_native_session_id or _default_zorl_native_session_id(train_model_id)
            if args.zorl_native_parent_lora_name is None:
                parent_name = f"zorl-native-parent/{train_model_id}/{time.time_ns()}"
                parent_save = save_weights_for_sampler(args.train_url, train_model_id, parent_name)
                parent_session = create_sampling_session(args.train_url, train_model_id, parent_save["path"])
                args.zorl_native_parent_lora_name = parent_session["lora_name"]
                print(f"    Native SGLang ZORL parent loaded as: {args.zorl_native_parent_lora_name}")
            start_sglang_zorl_sessions(
                infer_urls,
                session_id=args.zorl_native_session_id,
                parent_lora_name=args.zorl_native_parent_lora_name,
                args=args,
            )
            print(f"    Native SGLang ZORL session: {args.zorl_native_session_id}")
            if args.zorl_elitist_rollback:
                snapshot_sglang_zorl_parent(
                    infer_urls,
                    session_id=args.zorl_native_session_id,
                    snapshot_id=args.zorl_elitist_snapshot_id,
                )
                print(f"    Native SGLang ZORL elitist snapshot initialized: {args.zorl_elitist_snapshot_id}")
        t0 = time.time()
        if args.zorl_probe_initial_parent:
            parent_probe = probe_parent_adapter(
                args.train_url,
                train_model_id,
                reward_infer_url,
                reward_data,
                args,
                generation_index=0,
            )
            parent_stats = parent_probe["reward_stats"]
            print(
                f"      Parent probe 0: exact={parent_probe['exact_count']}/{len(CODES)}, "
                f"reward_mean={parent_probe['reward_mean']:.4f}, "
                f"tf_prob_mean={parent_stats.get('teacher_forced_prob_mean', 0.0):.4f}, "
                f"rollout_prefix_mean={parent_stats.get('rollout_prefix_mean', 0.0):.4f}, "
                f"rollout_exact_rate={parent_stats.get('rollout_exact_rate', 0.0):.4f}"
            )
            best_parent_probe = parent_probe
            if args.zorl_elitist_rollback and args.zorl_candidate_backend == "sglang_native":
                snapshot_sglang_zorl_parent(
                    infer_urls,
                    session_id=args.zorl_native_session_id,
                    snapshot_id=args.zorl_elitist_snapshot_id,
                )
                print(f"      Parent snapshot updated at generation 0: {args.zorl_elitist_snapshot_id}")
            if args.save_best_parent_name and args.zorl_candidate_backend == "sglang_native":
                print(
                    "      WARN: --save-best-parent-name is not supported for native SGLang ZORL yet; "
                    "parent lives in SGLang."
                )
            elif args.save_best_parent_name:
                _save_checkpoint_if_requested(
                    args.train_url,
                    train_model_id,
                    f"{args.save_best_parent_name}-g000000",
                    label="best parent",
                )
        for step in range(args.steps):
            step_lr = get_lr(step, args)
            generation, candidate_rewards, reward_stats, apply_result = run_zorl_generation(
                args.train_url,
                train_model_id,
                reward_infer_url,
                reward_data,
                args,
                step_lr,
                generation_index=step,
            )
            metrics = apply_result.get("metrics", {})
            step_num = step + 1
            if step_num == 1 or step_num == args.steps or step_num % args.log_interval == 0:
                best_candidate = max(candidate_rewards, key=_candidate_sort_key)
                reward_components = []
                if "teacher_forced_prob_mean" in reward_stats:
                    reward_components.append(f"tf_prob_mean={reward_stats['teacher_forced_prob_mean']:.4f}")
                if "teacher_forced_first_error_prob_mean" in reward_stats:
                    reward_components.append(
                        f"tf_firsterr_prob_mean={reward_stats['teacher_forced_first_error_prob_mean']:.4f}"
                    )
                if "teacher_forced_suffix_prob_mean" in reward_stats:
                    reward_components.append(
                        f"tf_suffix_prob_mean={reward_stats['teacher_forced_suffix_prob_mean']:.4f}"
                    )
                if "rollout_prefix_mean" in reward_stats:
                    reward_components.append(f"rollout_prefix_mean={reward_stats['rollout_prefix_mean']:.4f}")
                if "rollout_char_match_mean" in reward_stats:
                    reward_components.append(f"rollout_char_mean={reward_stats['rollout_char_match_mean']:.4f}")
                if "rollout_valid_mean" in reward_stats:
                    reward_components.append(f"rollout_valid_mean={reward_stats['rollout_valid_mean']:.4f}")
                if "rollout_value_mean" in reward_stats:
                    reward_components.append(f"rollout_value_mean={reward_stats['rollout_value_mean']:.4f}")
                if "rollout_exact_rate" in reward_stats:
                    reward_components.append(f"rollout_exact_rate={reward_stats['rollout_exact_rate']:.4f}")
                if "time_start_generation_s" in reward_stats:
                    reward_components.append(f"t_start={reward_stats['time_start_generation_s']:.1f}s")
                if "candidate_preload_time_s" in reward_stats:
                    reward_components.append(
                        f"preload={reward_stats['candidate_preload_time_s']:.1f}s/"
                        f"{int(reward_stats.get('candidate_preloaded_count', 0))}"
                    )
                if "candidate_virtual_count" in reward_stats:
                    reward_components.append(f"virtual={int(reward_stats['candidate_virtual_count'])}")
                if "candidate_materialized_count" in reward_stats:
                    reward_components.append(f"materialized={int(reward_stats['candidate_materialized_count'])}")
                if "candidate_gpu_pool_loaded_loras" in reward_stats:
                    reward_components.append(
                        f"gpu_pool={int(reward_stats['candidate_gpu_pool_loaded_loras'])}/"
                        f"{int(reward_stats.get('candidate_gpu_pool_capacity', 0))}"
                    )
                if "adaptive_num_pairs" in reward_stats:
                    reward_components.append(f"pairs={int(reward_stats['adaptive_num_pairs'])}")
                if "adaptive_previous_active_projects" in reward_stats:
                    reward_components.append(f"prev_active={int(reward_stats['adaptive_previous_active_projects'])}")
                if "hard_project_rollouts_per_puzzle" in reward_stats:
                    reward_components.append(f"hard_rollouts={int(reward_stats['hard_project_rollouts_per_puzzle'])}")
                if "time_score_s" in reward_stats:
                    reward_components.append(f"t_score={reward_stats['time_score_s']:.1f}s")
                if "time_rescore_s" in reward_stats:
                    reward_components.append(f"t_rescore={reward_stats['time_rescore_s']:.1f}s")
                if "time_parent_baseline_s" in reward_stats:
                    reward_components.append(f"t_parent_baseline={reward_stats['time_parent_baseline_s']:.1f}s")
                if "parent_baseline_exact_rate" in reward_stats:
                    reward_components.append(
                        f"parent_baseline_exact={reward_stats['parent_baseline_exact_rate'] * len(CODES):.2f}/{len(CODES)}"
                    )
                if "parent_baseline_reward_mean" in reward_stats:
                    reward_components.append(
                        f"parent_baseline_reward={reward_stats['parent_baseline_reward_mean']:.4f}"
                    )
                if "parent_advantage_active_projects" in reward_stats:
                    reward_components.append(
                        f"active_projects={int(reward_stats['parent_advantage_active_projects'])}/"
                        f"{int(reward_stats.get('parent_advantage_total_projects', len(CODES)))}"
                    )
                if "parent_advantage_skipped_projects" in reward_stats:
                    reward_components.append(
                        f"skipped_projects={int(reward_stats['parent_advantage_skipped_projects'])}"
                    )
                if "rescore_top_k" in reward_stats:
                    reward_components.append(f"rescore_top_k={int(reward_stats['rescore_top_k'])}")
                if "rescore_candidate_count" in reward_stats:
                    reward_components.append(f"rescore_candidates={int(reward_stats['rescore_candidate_count'])}")
                if "rescore_rollouts_per_puzzle" in reward_stats:
                    reward_components.append(f"rescore_rollouts={int(reward_stats['rescore_rollouts_per_puzzle'])}")
                if "rescore_best_exact_count" in reward_stats:
                    reward_components.append(
                        f"rescore_best_exact={reward_stats['rescore_best_exact_count']:.2f}/{len(CODES)}"
                    )
                if "rescore_best_reward" in reward_stats:
                    reward_components.append(f"rescore_best_reward={reward_stats['rescore_best_reward']:.4f}")
                if "rescore_project_exact_mean" in reward_stats:
                    reward_components.append(
                        f"rescore_project_exact_mean={reward_stats['rescore_project_exact_mean']:.2f}"
                    )
                if "rescore_project_reward_mean" in reward_stats:
                    reward_components.append(
                        f"rescore_project_reward={reward_stats['rescore_project_reward_mean']:.4f}"
                    )
                if "score_order" in reward_stats:
                    reward_components.append(f"score_order={reward_stats['score_order']}")
                if "score_routing" in reward_stats:
                    reward_components.append(f"score_routing={reward_stats['score_routing']}")
                if "time_apply_s" in reward_stats:
                    reward_components.append(f"t_apply={reward_stats['time_apply_s']:.1f}s")
                if "time_generation_step_s" in reward_stats:
                    reward_components.append(f"t_step={reward_stats['time_generation_step_s']:.1f}s")
                if "score_rollouts_per_s" in reward_stats:
                    reward_components.append(f"rollouts_per_s={reward_stats['score_rollouts_per_s']:.2f}")
                if "score_candidate_examples_per_s" in reward_stats:
                    reward_components.append(
                        f"candidate_examples_per_s={reward_stats['score_candidate_examples_per_s']:.2f}"
                    )
                if "score_rollouts" in reward_stats:
                    reward_components.append(f"score_rollouts={int(reward_stats['score_rollouts'])}")
                if "selected_pairs" in reward_stats:
                    reward_components.append(f"selected_pairs={int(reward_stats['selected_pairs'])}")
                if "score_normalization" in reward_stats:
                    reward_components.append(f"score_norm={reward_stats['score_normalization']}")
                if "adaptive_update_norm_scale" in reward_stats:
                    reward_components.append(
                        f"adaptive_update_norm_scale={reward_stats['adaptive_update_norm_scale']:.4f}"
                    )
                if "adaptive_update_norm_signal" in reward_stats:
                    reward_components.append(
                        f"adaptive_update_norm_signal={reward_stats['adaptive_update_norm_signal']:.4f}"
                    )
                if "adaptive_update_norm_effective" in reward_stats:
                    reward_components.append(
                        f"adaptive_update_norm_effective={reward_stats['adaptive_update_norm_effective']:.2f}"
                    )
                if reward_stats.get("adaptive_update_norm_skipped"):
                    reward_components.append("adaptive_update_norm_skipped=1")
                if metrics.get("max_update_norm") is not None:
                    reward_components.append(f"max_update_norm={float(metrics['max_update_norm']):.2f}")
                if metrics.get("update_clip_scale") is not None:
                    reward_components.append(f"update_clip_scale={float(metrics['update_clip_scale']):.4f}")
                if metrics.get("unclipped_update_norm") is not None:
                    reward_components.append(f"unclipped_update_norm={float(metrics['unclipped_update_norm']):.2f}")
                if "selected_pair_abs_delta_mean" in reward_stats:
                    reward_components.append(f"selected_abs_delta={reward_stats['selected_pair_abs_delta_mean']:.4f}")
                if "selected_pair_winner_reward_mean" in reward_stats:
                    reward_components.append(
                        f"selected_winner_reward={reward_stats['selected_pair_winner_reward_mean']:.4f}"
                    )
                if "selected_pair_winner_exact_mean" in reward_stats:
                    reward_components.append(
                        f"selected_winner_exact_mean={reward_stats['selected_pair_winner_exact_mean']:.2f}"
                    )
                if "selected_project_reward_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_reward={reward_stats['selected_project_reward_mean']:.4f}"
                    )
                if "selected_project_prefix_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_prefix={reward_stats['selected_project_prefix_mean']:.4f}"
                    )
                if "selected_project_exact_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_exact_mean={reward_stats['selected_project_exact_mean']:.2f}"
                    )
                if "selected_project_reward_sum" in reward_stats:
                    reward_components.append(
                        f"selected_project_reward_sum={reward_stats['selected_project_reward_sum']:.4f}"
                    )
                if "selected_project_parent_reward_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_parent_reward={reward_stats['selected_project_parent_reward_mean']:.4f}"
                    )
                if "selected_project_score_std_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_score_std={reward_stats['selected_project_score_std_mean']:.4f}"
                    )
                if "selected_project_advantage_sum" in reward_stats:
                    reward_components.append(
                        f"selected_project_advantage_sum={reward_stats['selected_project_advantage_sum']:.4f}"
                    )
                if "selected_project_advantage_mean" in reward_stats:
                    reward_components.append(
                        f"selected_project_advantage_mean={reward_stats['selected_project_advantage_mean']:.4f}"
                    )
                if "selected_project_improved" in reward_stats:
                    reward_components.append(
                        f"selected_project_improved={int(reward_stats['selected_project_improved'])}"
                    )
                if "selected_candidate_weight_max" in reward_stats:
                    reward_components.append(f"selected_weight_max={reward_stats['selected_candidate_weight_max']:.4f}")
                if "selected_candidate_score_abs_mean" in reward_stats:
                    reward_components.append(
                        f"selected_score_abs_mean={reward_stats['selected_candidate_score_abs_mean']:.4f}"
                    )
                if "selected_candidate_score_abs_max" in reward_stats:
                    reward_components.append(
                        f"selected_score_abs_max={reward_stats['selected_candidate_score_abs_max']:.4f}"
                    )
                if reward_stats.get("zero_signal_update"):
                    reward_components.append("zero_signal_update=1")
                if "selected_candidate_reward" in reward_stats:
                    reward_components.append(f"selected_reward={reward_stats['selected_candidate_reward']:.4f}")
                if "selected_candidate_exact_count" in reward_stats:
                    reward_components.append(
                        f"selected_exact={float(reward_stats['selected_candidate_exact_count']):.2f}/{len(CODES)}"
                    )
                reward_components.append(
                    f"best_candidate_exact={float(best_candidate.get('rollout_exact_count', 0.0) or 0.0):.2f}/{len(CODES)}"
                )
                if best_candidate.get("rollout_prefix_mean") is not None:
                    reward_components.append(f"best_candidate_prefix={best_candidate['rollout_prefix_mean']:.4f}")
                reward_components.append(f"best_candidate_reward={best_candidate['reward_mean']:.4f}")
                reward_component_suffix = ", ".join(reward_components)
                if reward_component_suffix:
                    reward_component_suffix = ", " + reward_component_suffix
                apply_lr = float(reward_stats.get("apply_lr", step_lr))
                print(
                    f"      Generation {step_num}/{args.steps}: "
                    f"reward_mean={reward_stats['reward_mean']:.4f}, "
                    f"reward_min={reward_stats['reward_min']:.4f}, "
                    f"reward_max={reward_stats['reward_max']:.4f}, "
                    f"pair_delta_mean={metrics.get('pair_delta_mean', 'N/A')}, "
                    f"update_norm={metrics.get('update_norm', 'N/A')}, "
                    f"grad_norm={metrics.get('grad_norm', 'N/A')}, "
                    f"lr={step_lr:.2e}, apply_lr={apply_lr:.2e}, "
                    f"family_refreshed={generation.get('family_refreshed')}"
                    f"{reward_component_suffix}"
                )
                best_candidate_outputs = best_candidate.get("rollout_outputs") or {}
                if best_candidate_outputs:
                    print(
                        f"        Best candidate {best_candidate['candidate_id']}: "
                        f"exact={float(best_candidate.get('rollout_exact_count', 0.0) or 0.0):.2f}/{len(CODES)}, "
                        f"reward_mean={float(best_candidate['reward_mean']):.4f}"
                    )
                    for project in sorted(best_candidate_outputs):
                        print(f"          {project}: '{best_candidate_outputs[project]}'")
                if args.zorl_log_project_bests:
                    for project_name in sorted(CODES):
                        project_best = _best_candidate_for_project(candidate_rewards, project_name)
                        if project_best is None:
                            continue
                        project_metric = project_best["project_metrics"][project_name]
                        print(
                            f"        Project best {project_name}: candidate={project_best['candidate_id']}, "
                            f"reward={float(project_metric['reward']):.4f}, "
                            f"exact={float(project_metric.get('rollout_exact_rate', 0.0) or 0.0):.2f}, "
                            f"firsterr={float(project_metric.get('teacher_forced_first_error_prob', 0.0) or 0.0):.4f}, "
                            f"suffix={float(project_metric.get('teacher_forced_suffix_prob', 0.0) or 0.0):.4f}, "
                            f"valid={float(project_metric.get('rollout_valid_rate', 0.0) or 0.0):.2f}, "
                            f"value={float(project_metric.get('rollout_value_score', 0.0) or 0.0):.2f}, "
                            f"prefix={float(project_metric['rollout_prefix_ratio']):.4f}, "
                            f"char={float(project_metric['rollout_char_match_ratio']):.4f}"
                        )
                        print(f"          output: '{project_metric['generated_text']}'")

            if args.zorl_parent_probe_interval > 0 and (
                step_num == 1 or step_num == args.steps or step_num % args.zorl_parent_probe_interval == 0
            ):
                parent_probe = probe_parent_adapter(
                    args.train_url,
                    train_model_id,
                    reward_infer_url,
                    reward_data,
                    args,
                    generation_index=step_num,
                )
                parent_stats = parent_probe["reward_stats"]
                print(
                    f"      Parent probe {step_num}: exact={parent_probe['exact_count']}/{len(CODES)}, "
                    f"reward_mean={parent_probe['reward_mean']:.4f}, "
                    f"tf_prob_mean={parent_stats.get('teacher_forced_prob_mean', 0.0):.4f}, "
                    f"rollout_prefix_mean={parent_stats.get('rollout_prefix_mean', 0.0):.4f}, "
                    f"rollout_exact_rate={parent_stats.get('rollout_exact_rate', 0.0):.4f}"
                )
                parent_key = _parent_probe_key(parent_probe)
                best_key = (
                    (-1, float("-inf"), float("-inf"), float("-inf"))
                    if best_parent_probe is None
                    else _parent_probe_key(best_parent_probe)
                )
                if parent_key > best_key:
                    best_parent_probe = parent_probe
                    print(
                        f"      Parent best updated at generation {step_num}: "
                        f"exact={parent_probe['exact_count']}/{len(CODES)}, "
                        f"reward_mean={parent_probe['reward_mean']:.4f}"
                    )
                    if args.zorl_elitist_rollback and args.zorl_candidate_backend == "sglang_native":
                        snapshot_sglang_zorl_parent(
                            infer_urls,
                            session_id=args.zorl_native_session_id,
                            snapshot_id=args.zorl_elitist_snapshot_id,
                        )
                        print(
                            f"      Parent snapshot updated at generation {step_num}: {args.zorl_elitist_snapshot_id}"
                        )
                    if args.save_best_parent_name and args.zorl_candidate_backend != "sglang_native":
                        _save_checkpoint_if_requested(
                            args.train_url,
                            train_model_id,
                            f"{args.save_best_parent_name}-g{step_num:06d}",
                            label="best parent",
                        )
                    for output in parent_probe["outputs"]:
                        status = "OK" if output["exact_match"] else "FAIL"
                        print(
                            f"        [{status}] {output['project']}: '{output['generated']}' "
                            f"(prefix_ratio={output['prefix_ratio']:.4f})"
                        )
                elif (
                    args.zorl_elitist_rollback
                    and args.zorl_candidate_backend == "sglang_native"
                    and best_parent_probe is not None
                    and parent_key < best_key
                ):
                    restore_sglang_zorl_parent(
                        infer_urls,
                        session_id=args.zorl_native_session_id,
                        snapshot_id=args.zorl_elitist_snapshot_id,
                    )
                    print(
                        f"      Parent restored to best snapshot {args.zorl_elitist_snapshot_id}: "
                        f"exact={best_parent_probe['exact_count']}/{len(CODES)}, "
                        f"reward_mean={best_parent_probe['reward_mean']:.4f}"
                    )
                if args.zorl_stop_on_parent_exact and parent_probe["exact_count"] == len(CODES):
                    print(f"      Parent reached exact recall at generation {step_num}; stopping early before sync.")
                    break

        print(f"    Training done in {time.time() - t0:.1f}s")
        if best_parent_probe is not None:
            print(
                f"    Best parent probe: generation={best_parent_probe['generation_index']}, "
                f"exact={best_parent_probe['exact_count']}/{len(CODES)}, "
                f"reward_mean={best_parent_probe['reward_mean']:.4f}"
            )
        if args.save_final_name and args.zorl_candidate_backend == "sglang_native":
            print("    WARN: --save-final-name is not supported for native SGLang ZORL yet; parent lives in SGLang.")
        elif args.save_final_name:
            _save_checkpoint_if_requested(
                args.train_url,
                train_model_id,
                args.save_final_name,
                label="final parent",
            )

    # Post-training greedy eval via the sampler-LoRA path. Robust alternative
    # to the NCCL-based sync_inference_weights flow (which has been flaky):
    # save the trained LoRA to disk, register it on SGLang via
    # create_sampling_session (= /load_lora_adapter), then test_inference at
    # temperature=0 with that LoRA name as `lora_path`. Works for any
    # trainer-mode (gradient/zorl/grpo) and produces a clean apples-to-apples
    # number that can be compared to the saved ZORL parent probes.
    if args.eval_via_sampler_lora:
        eval_lora_name = args.save_final_name or f"countdown-eval-{args.trainer_mode}-{int(time.time())}"
        print(f"\n    Post-train eval via sampler LoRA: {eval_lora_name}")
        try:
            if args.zorl_candidate_backend == "sglang_native":
                registered_name = args.zorl_native_parent_lora_name
                print(f"      Using native SGLang parent LoRA: {registered_name}")
            else:
                save_result = save_weights_for_sampler(args.train_url, args.train_model_id, eval_lora_name)
                sampler_path = save_result.get("path") or save_result.get("xorl_uri") or eval_lora_name
                print(f"      Saved to: {sampler_path}")
                session = create_sampling_session(args.train_url, args.train_model_id, sampler_path)
                registered_name = session.get("lora_name") or eval_lora_name
                print(f"      Registered on SGLang as: {registered_name}")
            for url in infer_urls:
                flush_inference_cache(url)
            correct = test_inference(
                infer_urls,
                served_model_name,
                label=f"sampler-LoRA greedy ({eval_lora_name})",
                lora_path=registered_name,
                enable_thinking=bool(getattr(args, "zorl_enable_thinking", False)),
                max_tokens=int(getattr(args, "zorl_rollout_max_new_tokens", None) or 64),
            )
            print(f"    Sampler-LoRA score: {correct}/{total_codes}")
            return int(correct)
        except Exception as exc:
            print(f"    Post-train sampler-LoRA eval FAILED: {type(exc).__name__}: {exc}")
            # Fall through to the NCCL sync path, or the skip path.

    if args.skip_final_sync:
        if best_parent_probe is not None:
            print(
                f"    Skipping final sync/eval; using best parent probe score "
                f"{best_parent_probe['exact_count']}/{len(CODES)}"
            )
            return int(best_parent_probe["exact_count"])
        print("    Skipping final sync/eval.")
        return None

    if args.sync_quant == "fp8":
        set_sync_quantization(args.train_url)
    sync_version = f"password-sync-{int(time.time())}"
    sync_result = sync_weights(
        args.train_url,
        args.master_address,
        sync_version,
        model_id=args.train_model_id,
        flush_cache=True,
    )
    if not sync_result.get("success"):
        print(f"    SYNC FAILED: {sync_result}")
        return None
    versions = wait_for_inference_weight_version(
        infer_urls,
        sync_version,
        timeout=args.sync_wait_timeout,
    )
    for url, version in versions.items():
        print(f"    Endpoint ready: {url} weight_version={version}")
        flush_inference_cache(url)
        print(f"    Cache flushed: {url}")

    correct = test_inference(
        infer_urls,
        served_model_name,
        f"after sync ({sync_version})",
        expected_weight_version=sync_version,
        enable_thinking=bool(getattr(args, "zorl_enable_thinking", False)),
        max_tokens=int(getattr(args, "zorl_rollout_max_new_tokens", None) or 64),
    )
    print(f"    Score: {correct}/{total_codes}")
    return correct


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Password memorization e2e test")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name (e.g. Qwen/Qwen3-8B)")
    parser.add_argument(
        "--trainer-mode",
        type=str,
        default="gradient",
        choices=["gradient", "zorl", "grpo"],
        help=(
            "Training mode: 'gradient' = conventional teacher-forced SFT; "
            "'zorl' = zeroth-order RL via ES on LoRA; "
            "'grpo' = group-relative policy optimization (sample K rollouts, "
            "binary reward, advantage = reward - group mean, importance-sampled "
            "policy gradient via the trainer's importance_sampling loss)."
        ),
    )
    parser.add_argument(
        "--grpo-group-size",
        type=int,
        default=8,
        help="GRPO: number of rollouts per prompt per step. Group-relative advantage uses these K samples.",
    )
    parser.add_argument(
        "--grpo-temperature",
        type=float,
        default=0.6,
        help="GRPO: sampling temperature for rollouts. Must be >0 for stochastic policy gradient.",
    )
    parser.add_argument(
        "--grpo-max-new-tokens",
        type=int,
        default=48,
        help="GRPO: max decode length per rollout.",
    )
    parser.add_argument(
        "--grpo-skip-zero-advantage",
        action="store_true",
        default=True,
        help=(
            "GRPO: if all K rollouts in a group get the same reward (so all "
            "advantages are zero), skip that group's update. Standard practice — "
            "zero-advantage groups contribute zero gradient anyway and just add noise."
        ),
    )
    parser.add_argument("--steps", type=int, default=64, help="Training steps or ZORL generations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine", "warmup_cosine"],
        help="LR schedule: constant, cosine, or warmup_cosine",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.01,
        help="lr_min = lr * lr_min_ratio for cosine schedules",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps (constant LR) before cosine decay (for warmup_cosine)",
    )
    parser.add_argument(
        "--sync-quant",
        type=str,
        default="fp8",
        choices=["fp8", "none"],
        help="Sync quantization: fp8 (block e4m3) or none (bf16)",
    )
    parser.add_argument("--train-url", type=str, default="http://localhost:6000", help="Training server URL")
    parser.add_argument(
        "--infer-url",
        type=str,
        nargs="+",
        default=["http://localhost:30000"],
        help="Direct inference endpoint URL(s), space-separated. These are registered with the trainer for adapter loads.",
    )
    parser.add_argument(
        "--reward-infer-url",
        type=str,
        nargs="+",
        default=None,
        help="Optional reward sampling URL(s). Use a Dispatch URL here while keeping --infer-url as direct SGLang URLs.",
    )
    parser.add_argument("--master-address", type=str, default="localhost", help="Master address for NCCL weight sync")
    parser.add_argument(
        "--train-model-id",
        type=str,
        default=DEFAULT_TRAIN_MODEL_ID,
        help="Model/session identifier used for xorl training APIs",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name to send to SGLang chat-completions requests; defaults to --model",
    )
    parser.add_argument("--log-interval", type=int, default=16, help="Print metrics every N steps")
    parser.add_argument(
        "--train-startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for the training server to report engine_running=true",
    )
    parser.add_argument(
        "--infer-startup-timeout",
        type=int,
        default=120,
        help="Seconds to wait for each inference endpoint to respond before starting training",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run a pre-training inference check before weight sync",
    )
    parser.add_argument(
        "--sync-wait-timeout",
        type=int,
        default=120,
        help="Seconds to wait for inference endpoints to report the new weight_version",
    )
    parser.add_argument(
        "--skip-final-sync",
        action="store_true",
        help="Skip the final sync-and-chat-eval stage and report the best parent probe score instead",
    )
    parser.add_argument(
        "--eval-via-sampler-lora",
        action="store_true",
        default=False,
        help=(
            "Run a final greedy temp=0 eval by saving the trained LoRA via "
            "save_weights_for_sampler and registering it on SGLang via "
            "create_sampling_session. Bypasses the flaky NCCL-based "
            "sync_inference_weights flow. Recommended for grpo and gradient modes."
        ),
    )
    parser.add_argument(
        "--init-weights-path",
        type=str,
        default=None,
        help="Optional checkpoint URI/path to load into the newly created training session before training",
    )
    parser.add_argument(
        "--init-weights-optimizer",
        action="store_true",
        help="Also restore optimizer state when --init-weights-path is provided",
    )
    parser.add_argument(
        "--save-best-parent-name",
        type=str,
        default=None,
        help="Optional checkpoint name prefix used to save every improved parent probe",
    )
    parser.add_argument(
        "--save-final-name",
        type=str,
        default=None,
        help="Optional checkpoint name used to save the final parent state after training",
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank override used for ZORL sessions")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha override used for ZORL sessions")
    parser.add_argument("--zorl-b-sigma", type=float, default=0.01, help="ZORL LoRA-B perturbation scale")
    parser.add_argument("--zorl-num-pairs", type=int, default=8, help="ZORL perturbation pairs per generation")
    parser.add_argument(
        "--zorl-perturbation-mode",
        type=str,
        default="b_only",
        choices=["b_only", "a_and_b"],
        help="Native SGLang ZORL perturbation mode. b_only preserves the current fixed-A search.",
    )
    parser.add_argument(
        "--zorl-refresh-interval",
        type=int,
        default=16,
        help="Generations between LoRA-A family refreshes",
    )
    parser.add_argument("--zorl-seed", type=int, default=1234, help="Seed for ZORL family/noise planning")
    parser.add_argument(
        "--zorl-score-batch-size",
        type=int,
        default=16,
        help="How many candidate LoRAs to score together in one batched /generate request",
    )
    parser.add_argument(
        "--zorl-score-max-workers",
        type=int,
        default=8,
        help="Max concurrent HTTP threads for scoring inference requests (default: 8)",
    )
    parser.add_argument(
        "--zorl-score-order",
        type=str,
        default="example_major",
        choices=["example_major", "candidate_major"],
        help=(
            "Order used to schedule ZORL scoring requests. candidate_major keeps one candidate "
            "LoRA window hot across all puzzles, which is better for lazy/small GPU LoRA pools."
        ),
    )
    parser.add_argument(
        "--zorl-score-routing",
        type=str,
        default="round_robin",
        choices=["round_robin", "candidate_chunk_sticky"],
        help=(
            "Routing policy for scoring requests. candidate_chunk_sticky assigns each candidate "
            "window to one direct SGLang endpoint for a wave, avoiding duplicate lazy LoRA loads."
        ),
    )
    parser.add_argument(
        "--zorl-preload-candidates",
        action="store_true",
        help="Preload native SGLang ZORL candidate LoRAs into the GPU LoRA pool before scoring",
    )
    parser.add_argument(
        "--zorl-optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "signsgd", "adamw"],
        help="Optimizer used for applying ZORL parent updates",
    )
    parser.add_argument(
        "--zorl-sgd-momentum",
        type=float,
        default=0.0,
        help="Momentum used when --zorl-optimizer=sgd",
    )
    parser.add_argument(
        "--zorl-sgd-nesterov",
        action="store_true",
        help="Enable Nesterov momentum when --zorl-optimizer=sgd",
    )
    parser.add_argument(
        "--zorl-reward-mode",
        type=str,
        default="hybrid",
        choices=["teacher_forced", "rollout", "hybrid"],
        help="Reward mode for candidate scoring: teacher_forced, rollout, or hybrid",
    )
    parser.add_argument(
        "--zorl-inference-api-format",
        type=str,
        default="generate",
        choices=["generate", "chat_completions"],
        help="Inference API used for ZORL rollout scoring. Use chat_completions when sampling through Dispatch.",
    )
    parser.add_argument(
        "--zorl-candidate-backend",
        type=str,
        default="filesystem",
        choices=["filesystem", "sglang", "sglang_native"],
        help="Where to materialize ZORL candidate adapters before scoring.",
    )
    parser.add_argument(
        "--zorl-adaptive-num-pairs-multiplier",
        type=float,
        default=1.0,
        help=(
            "For sglang_native, multiply candidate pairs after the previous parent baseline "
            "has at most --zorl-adaptive-num-pairs-active-threshold unsolved projects."
        ),
    )
    parser.add_argument(
        "--zorl-adaptive-num-pairs-active-threshold",
        type=int,
        default=4,
        help="Active-project threshold that enables --zorl-adaptive-num-pairs-multiplier.",
    )
    parser.add_argument(
        "--zorl-adaptive-num-pairs-max",
        type=int,
        default=0,
        help="Optional cap for adaptive candidate pairs; 0 means uncapped.",
    )
    parser.add_argument(
        "--zorl-native-session-id",
        type=str,
        default=None,
        help="Native SGLang ZORL session id. Defaults to a unique id based on --train-model-id.",
    )
    parser.add_argument(
        "--zorl-native-parent-lora-name",
        type=str,
        default=None,
        help="Already-loaded parent LoRA name for --zorl-candidate-backend=sglang_native.",
    )
    parser.add_argument(
        "--zorl-teacher-forced-weight",
        type=float,
        default=0.25,
        help="Weight on exp(mean answer logprob) when --zorl-reward-mode=hybrid",
    )
    parser.add_argument(
        "--zorl-teacher-forced-suffix-weight",
        type=float,
        default=0.0,
        help="Weight on exp(mean unresolved-suffix logprob) when --zorl-reward-mode=hybrid",
    )
    parser.add_argument(
        "--zorl-teacher-forced-first-error-weight",
        type=float,
        default=0.0,
        help="Weight on exp(logprob of the first unresolved target token) when --zorl-reward-mode=hybrid",
    )
    parser.add_argument(
        "--zorl-teacher-forced-first-error-prefix-power",
        type=float,
        default=0.0,
        help="Optional power used to gate the first-error reward by rollout prefix_ratio**power",
    )
    parser.add_argument(
        "--zorl-rollout-prefix-weight",
        type=float,
        default=1.0,
        help="Weight on prompt-only rollout token-prefix match reward",
    )
    parser.add_argument(
        "--zorl-rollout-char-weight",
        type=float,
        default=0.0,
        help="Weight on prompt-only rollout character-position match reward",
    )
    parser.add_argument(
        "--zorl-rollout-valid-weight",
        type=float,
        default=0.0,
        help="Weight on the rate of rollouts that parse and use each input number exactly once",
    )
    parser.add_argument(
        "--zorl-rollout-value-weight",
        type=float,
        default=0.0,
        help="Weight on target-value proximity for valid rollout expressions",
    )
    parser.add_argument(
        "--zorl-rollout-exact-bonus",
        type=float,
        default=2.0,
        help="Bonus added when the prompt-only rollout begins with the exact target code",
    )
    parser.add_argument(
        "--zorl-rollout-extra-tokens",
        type=int,
        default=4,
        help="Extra decode budget beyond the target answer length for rollout-based rewards",
    )
    parser.add_argument(
        "--zorl-rollout-max-new-tokens",
        type=int,
        default=None,
        help="Optional explicit rollout decode length override for rollout-based rewards",
    )
    parser.add_argument(
        "--zorl-rollouts-per-puzzle",
        type=int,
        default=1,
        help=(
            "Number of independent rollouts to sample per (perturbation, puzzle) pair. "
            "When >1, the binary reward per-perturbation becomes the mean exact-match rate "
            "across these N samples (∈ [0, 1] with 1/N granularity), giving a denser "
            "ES gradient signal at a fixed perturbation count. Pairs with a non-zero "
            "rollout temperature; with temperature=0 the N rollouts are deterministic and "
            "reduce to a single sample. Modal-style: N=8 + temperature=0.6."
        ),
    )
    parser.add_argument(
        "--zorl-hard-project-rollout-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply rollout samples for candidate scoring after parent-baseline filtering "
            "leaves at most --zorl-hard-project-rollout-active-threshold active projects."
        ),
    )
    parser.add_argument(
        "--zorl-hard-project-rollout-active-threshold",
        type=int,
        default=4,
        help="Active-project threshold that enables --zorl-hard-project-rollout-multiplier.",
    )
    parser.add_argument(
        "--zorl-hard-project-rollout-max",
        type=int,
        default=0,
        help="Optional cap for hard-project rollout samples per puzzle; 0 means uncapped.",
    )
    parser.add_argument(
        "--zorl-hard-project-reward-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplicative boost on the rollout_{prefix,char,valid,value} reward weights "
            "for projects in the parent-baseline 'active' (unsolved) set when the active-project "
            "count is at most --zorl-hard-project-reward-active-threshold. Leaves the "
            "rollout_exact_bonus unchanged so the absolute signal scale of a clean win is unchanged; "
            "the boost only sharpens the dense pre-solve signal that ES needs to climb on "
            "projects with no exact rollouts yet. >1.0 enables the boost; <=1.0 disables it."
        ),
    )
    parser.add_argument(
        "--zorl-hard-project-reward-active-threshold",
        type=int,
        default=4,
        help="Active-project threshold that enables --zorl-hard-project-reward-multiplier.",
    )
    parser.add_argument(
        "--zorl-active-puzzle-count",
        type=int,
        default=0,
        help=(
            "If >0 and < total puzzle pool size, each ZORL generation samples this many "
            "puzzles from the pool (deterministically seeded by generation index + "
            "--zorl-puzzle-rotation-seed). 0 disables rotation and uses the full pool. "
            "Rotation injects variance into the gradient and prevents the 'stuck on a fixed "
            "hard subset' plateau when the parent solves the same N puzzles every gen. "
            "Parent probes always use the full pool regardless of this setting."
        ),
    )
    parser.add_argument(
        "--zorl-puzzle-rotation-seed",
        type=int,
        default=0,
        help="Base seed for per-generation puzzle rotation (mixed with the generation index).",
    )
    parser.add_argument(
        "--zorl-rollout-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for ZORL rollouts. >0 gives stochastic samples needed when --zorl-rollouts-per-puzzle > 1.",
    )
    parser.add_argument(
        "--zorl-enable-thinking",
        action="store_true",
        help=(
            "Enable Qwen3's native <think>...</think> reasoning mode. The chat template is "
            "applied with enable_thinking=True, the model is free to write a reasoning trace "
            "before emitting its final expression, and the reward extractor strips the think "
            "block before parsing. Pair with a larger --zorl-rollout-max-new-tokens (e.g. 512)."
        ),
    )
    parser.add_argument(
        "--zorl-rollout-stop",
        action="append",
        default=[],
        help=(
            "Optional stop string for rollout generation; may be repeated. "
            "Useful for Countdown to avoid decoding commentary after the first expression."
        ),
    )
    parser.add_argument(
        "--zorl-update-strategy",
        type=str,
        default="es",
        choices=[
            "es",
            "best_candidate",
            "winner_sign",
            "elite_pair_delta",
            "elite_winner_reward",
            "project_winner_reward",
            "project_winner_delta",
            "project_parent_advantage",
            "project_baseline_es",
        ],
        help="How to turn candidate rewards into a parent update",
    )
    parser.add_argument(
        "--zorl-update-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier on scheduled LR for non-best-candidate ZORL update strategies",
    )
    parser.add_argument(
        "--zorl-best-candidate-step-scale",
        type=float,
        default=1.0,
        help="Multiplier on sigma when --zorl-update-strategy=best_candidate",
    )
    parser.add_argument(
        "--zorl-rescore-top-k",
        type=int,
        default=0,
        help=(
            "Rescore noisy winners with --zorl-rescore-rollouts before selecting the parent update. "
            "For best_candidate this means global top K; for project_winner_* this means top K per project. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--zorl-rescore-rollouts",
        type=int,
        default=None,
        help="Rollouts per puzzle for the top-K rescore pass.",
    )
    parser.add_argument(
        "--zorl-candidate-delta-step-scale",
        type=float,
        default=1.0,
        help="Multiplier on sigma for candidate-delta update strategies",
    )
    parser.add_argument(
        "--zorl-max-update-norm",
        type=float,
        default=None,
        help=(
            "Optional native SGLang ZORL trust-region cap on the unscaled LoRA-B update norm "
            "before applying learning_rate."
        ),
    )
    parser.add_argument(
        "--zorl-adaptive-update-norm",
        action="store_true",
        help=(
            "Scale --zorl-max-update-norm by selected project-parent advantage. "
            "This keeps late low-signal project_parent_advantage updates smaller without changing scoring."
        ),
    )
    parser.add_argument(
        "--zorl-adaptive-update-norm-reference",
        type=float,
        default=0.25,
        help=(
            "Advantage mean that maps to the full --zorl-max-update-norm when --zorl-adaptive-update-norm is enabled."
        ),
    )
    parser.add_argument(
        "--zorl-adaptive-update-norm-min-scale",
        type=float,
        default=0.15,
        help="Minimum scale applied to --zorl-max-update-norm when adaptive update norm is enabled.",
    )
    parser.add_argument(
        "--zorl-adaptive-update-norm-max-scale",
        type=float,
        default=1.0,
        help="Maximum scale applied to --zorl-max-update-norm when adaptive update norm is enabled.",
    )
    parser.add_argument(
        "--zorl-elite-pairs",
        type=int,
        default=2,
        help="Number of perturbation pairs to keep for elite ZORL update strategies",
    )
    parser.add_argument(
        "--zorl-parent-probe-interval",
        type=int,
        default=0,
        help="Probe the current parent adapter every N generations via sampler export (0 disables)",
    )
    parser.add_argument(
        "--zorl-parent-probe-rollouts",
        type=int,
        default=None,
        help=(
            "Optional rollout samples per puzzle used only for parent probes. Defaults to --zorl-rollouts-per-puzzle."
        ),
    )
    parser.add_argument(
        "--zorl-parent-probe-temperature",
        type=float,
        default=None,
        help=(
            "Optional sampling temperature override used only for parent probes. Defaults to "
            "--zorl-rollout-temperature. Pair with --zorl-parent-probe-rollouts 1 + temperature 0 "
            "to make the probe metric match the final greedy eval — without this, probes can "
            "overstate capability via stochastic luck."
        ),
    )
    parser.add_argument(
        "--zorl-parent-baseline-rollouts",
        type=int,
        default=None,
        help=(
            "Optional rollout samples per puzzle used for project_parent_advantage's current-parent "
            "baseline. Defaults to --zorl-rollouts-per-puzzle."
        ),
    )
    parser.add_argument(
        "--zorl-parent-advantage-filter-solved",
        action="store_true",
        help=(
            "For project_parent_advantage, score the current parent before candidates and only score "
            "candidate rollouts on projects whose parent exact rate is below the solved threshold."
        ),
    )
    parser.add_argument(
        "--zorl-parent-advantage-solved-exact-rate",
        type=float,
        default=1.0,
        help="Exact-rate threshold used by --zorl-parent-advantage-filter-solved.",
    )
    parser.add_argument(
        "--zorl-elitist-rollback",
        action="store_true",
        help=("For native SGLang ZORL, snapshot the best probed parent and restore it when a later probe regresses."),
    )
    parser.add_argument(
        "--zorl-elitist-snapshot-id",
        type=str,
        default="best",
        help="Snapshot id used by --zorl-elitist-rollback",
    )
    parser.add_argument(
        "--zorl-project-weight",
        action="append",
        default=[],
        help="Optional per-project reward weight override in project=value form; may be repeated",
    )
    parser.add_argument(
        "--zorl-stop-on-parent-exact",
        action="store_true",
        help="Stop ZORL training early when a parent probe reaches exact recall on all password prompts",
    )
    parser.add_argument(
        "--zorl-log-project-bests",
        action="store_true",
        help="Log the single best candidate per project at each reporting interval",
    )
    parser.add_argument(
        "--zorl-probe-initial-parent",
        action="store_true",
        help="Probe the loaded parent once before the first ZORL generation",
    )
    args = parser.parse_args()
    if args.zorl_elite_pairs <= 0:
        raise ValueError("--zorl-elite-pairs must be positive")
    if args.zorl_update_lr_scale <= 0:
        raise ValueError("--zorl-update-lr-scale must be positive")
    if args.zorl_rescore_top_k < 0:
        raise ValueError("--zorl-rescore-top-k must be >= 0")
    if args.zorl_rescore_top_k > 0 and args.zorl_rescore_rollouts is None:
        raise ValueError("--zorl-rescore-top-k requires --zorl-rescore-rollouts")
    if args.zorl_rescore_rollouts is not None and args.zorl_rescore_rollouts <= 0:
        raise ValueError("--zorl-rescore-rollouts must be positive")
    if args.zorl_adaptive_num_pairs_multiplier < 1.0:
        raise ValueError("--zorl-adaptive-num-pairs-multiplier must be >= 1")
    if args.zorl_adaptive_num_pairs_active_threshold < 0:
        raise ValueError("--zorl-adaptive-num-pairs-active-threshold must be >= 0")
    if args.zorl_adaptive_num_pairs_max < 0:
        raise ValueError("--zorl-adaptive-num-pairs-max must be >= 0")
    if args.zorl_adaptive_num_pairs_max and args.zorl_adaptive_num_pairs_max < args.zorl_num_pairs:
        raise ValueError("--zorl-adaptive-num-pairs-max must be 0 or >= --zorl-num-pairs")
    if args.zorl_hard_project_rollout_multiplier < 1.0:
        raise ValueError("--zorl-hard-project-rollout-multiplier must be >= 1")
    if args.zorl_hard_project_rollout_active_threshold < 0:
        raise ValueError("--zorl-hard-project-rollout-active-threshold must be >= 0")
    if args.zorl_hard_project_rollout_max < 0:
        raise ValueError("--zorl-hard-project-rollout-max must be >= 0")
    if args.zorl_hard_project_rollout_max and args.zorl_hard_project_rollout_max < args.zorl_rollouts_per_puzzle:
        raise ValueError("--zorl-hard-project-rollout-max must be 0 or >= --zorl-rollouts-per-puzzle")
    if args.zorl_hard_project_reward_multiplier < 1.0:
        raise ValueError("--zorl-hard-project-reward-multiplier must be >= 1")
    if args.zorl_hard_project_reward_active_threshold < 0:
        raise ValueError("--zorl-hard-project-reward-active-threshold must be >= 0")
    if args.zorl_active_puzzle_count < 0:
        raise ValueError("--zorl-active-puzzle-count must be >= 0")
    if args.zorl_max_update_norm is not None and args.zorl_max_update_norm <= 0:
        raise ValueError("--zorl-max-update-norm must be positive")
    if args.zorl_adaptive_update_norm and args.zorl_max_update_norm is None:
        raise ValueError("--zorl-adaptive-update-norm requires --zorl-max-update-norm")
    if args.zorl_adaptive_update_norm_reference <= 0:
        raise ValueError("--zorl-adaptive-update-norm-reference must be positive")
    if args.zorl_adaptive_update_norm_min_scale <= 0:
        raise ValueError("--zorl-adaptive-update-norm-min-scale must be positive")
    if args.zorl_adaptive_update_norm_max_scale <= 0:
        raise ValueError("--zorl-adaptive-update-norm-max-scale must be positive")
    if args.zorl_adaptive_update_norm_min_scale > args.zorl_adaptive_update_norm_max_scale:
        raise ValueError("--zorl-adaptive-update-norm-min-scale must be <= max scale")
    if args.zorl_parent_probe_interval < 0:
        raise ValueError("--zorl-parent-probe-interval must be >= 0")
    if args.zorl_parent_probe_rollouts is not None and args.zorl_parent_probe_rollouts <= 0:
        raise ValueError("--zorl-parent-probe-rollouts must be positive")
    if args.zorl_parent_baseline_rollouts is not None and args.zorl_parent_baseline_rollouts <= 0:
        raise ValueError("--zorl-parent-baseline-rollouts must be positive")
    if not (0.0 <= args.zorl_parent_advantage_solved_exact_rate <= 1.0):
        raise ValueError("--zorl-parent-advantage-solved-exact-rate must be in [0, 1]")
    if not (0.0 <= args.zorl_sgd_momentum < 1.0):
        raise ValueError("--zorl-sgd-momentum must be in [0, 1)")
    if args.zorl_sgd_nesterov and args.zorl_sgd_momentum <= 0.0:
        raise ValueError("--zorl-sgd-nesterov requires --zorl-sgd-momentum > 0")
    if args.init_weights_optimizer and not args.init_weights_path:
        raise ValueError("--init-weights-optimizer requires --init-weights-path")
    project_weights = {}
    for item in args.zorl_project_weight:
        if "=" not in item:
            raise ValueError(f"--zorl-project-weight must be project=value, got {item!r}")
        project_name, raw_weight = item.split("=", 1)
        project_name = project_name.strip()
        if project_name not in CODES:
            raise ValueError(f"Unknown project {project_name!r} in --zorl-project-weight")
        project_weights[project_name] = float(raw_weight)
    args.zorl_project_weights = project_weights

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enable_thinking = bool(getattr(args, "zorl_enable_thinking", False))
    training_data = build_training_data(tokenizer, enable_thinking=enable_thinking)
    reward_data = build_reward_data(tokenizer, enable_thinking=enable_thinking)
    print(f"Model: {args.model}")
    print(f"Trainer mode: {args.trainer_mode}")
    print(f"Training model_id: {args.train_model_id}")
    print(f"Served model name: {args.served_model_name or args.model}")
    print(f"Training data: {len(training_data)} examples")
    print(f"Schedule: {args.lr_schedule}, steps={args.steps}, lr={args.lr}")
    if args.trainer_mode == "zorl":
        print(
            f"ZORL config: rank={args.lora_rank}, alpha={args.lora_alpha}, sigma={args.zorl_b_sigma}, "
            f"num_pairs={args.zorl_num_pairs}, refresh_interval={args.zorl_refresh_interval}, seed={args.zorl_seed}, "
            f"perturbation_mode={args.zorl_perturbation_mode}, reward_mode={args.zorl_reward_mode}, optimizer={args.zorl_optimizer}, "
            f"sgd_momentum={args.zorl_sgd_momentum}, update_strategy={args.zorl_update_strategy}, "
            f"update_lr_scale={args.zorl_update_lr_scale}, "
            f"adaptive_num_pairs_multiplier={args.zorl_adaptive_num_pairs_multiplier}, "
            f"adaptive_num_pairs_threshold={args.zorl_adaptive_num_pairs_active_threshold}, "
            f"adaptive_num_pairs_max={args.zorl_adaptive_num_pairs_max}, "
            f"hard_project_rollout_multiplier={args.zorl_hard_project_rollout_multiplier}, "
            f"hard_project_rollout_threshold={args.zorl_hard_project_rollout_active_threshold}, "
            f"hard_project_rollout_max={args.zorl_hard_project_rollout_max}, "
            f"hard_project_reward_multiplier={args.zorl_hard_project_reward_multiplier}, "
            f"hard_project_reward_threshold={args.zorl_hard_project_reward_active_threshold}, "
            f"active_puzzle_count={args.zorl_active_puzzle_count}, "
            f"puzzle_rotation_seed={args.zorl_puzzle_rotation_seed}, "
            f"puzzle_pool_size={len(COUNTDOWN_PUZZLES)}, "
            f"rescore_top_k={args.zorl_rescore_top_k}, "
            f"rescore_rollouts={args.zorl_rescore_rollouts}, "
            f"candidate_delta_step_scale={args.zorl_candidate_delta_step_scale}, "
            f"max_update_norm={args.zorl_max_update_norm}, "
            f"adaptive_update_norm={args.zorl_adaptive_update_norm}, "
            f"adaptive_update_norm_reference={args.zorl_adaptive_update_norm_reference}, "
            f"adaptive_update_norm_min_scale={args.zorl_adaptive_update_norm_min_scale}, "
            f"adaptive_update_norm_max_scale={args.zorl_adaptive_update_norm_max_scale}, "
            f"rollout_char_weight={args.zorl_rollout_char_weight}, "
            f"rollout_valid_weight={args.zorl_rollout_valid_weight}, "
            f"rollout_value_weight={args.zorl_rollout_value_weight}, "
            f"rollout_stop={args.zorl_rollout_stop}, "
            f"tf_suffix_weight={args.zorl_teacher_forced_suffix_weight}, "
            f"tf_first_error_weight={args.zorl_teacher_forced_first_error_weight}, "
            f"tf_first_error_prefix_power={args.zorl_teacher_forced_first_error_prefix_power}, "
            f"elite_pairs={args.zorl_elite_pairs}, "
            f"parent_probe_interval={args.zorl_parent_probe_interval}, "
            f"parent_probe_rollouts={args.zorl_parent_probe_rollouts}, "
            f"parent_probe_temperature={args.zorl_parent_probe_temperature}, "
            f"parent_baseline_rollouts={args.zorl_parent_baseline_rollouts}, "
            f"parent_advantage_filter_solved={args.zorl_parent_advantage_filter_solved}, "
            f"parent_advantage_solved_exact_rate={args.zorl_parent_advantage_solved_exact_rate}, "
            f"elitist_rollback={args.zorl_elitist_rollback}"
        )
        if args.zorl_project_weights:
            print(f"ZORL project weights: {args.zorl_project_weights}")

    total_codes = len(CODES) * len(args.infer_url)
    try:
        correct = run_test(args, training_data, reward_data)
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        correct = None

    print(f"\n{'=' * 60}")
    if correct is not None:
        print(f"  Result: {correct}/{total_codes} [{'PASS' if correct >= total_codes - 1 else 'FAIL'}]")
    else:
        print("  Result: ERROR")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
