"""Offline teacher-prompt probe for the OPSD Wordle env.

Evaluates whether a teacher prompt style yields a *fair and informative*
teacher distribution without launching a trainer job. Reuses the exact prompt
builders and token-weighting code from ``train_opsd_baseline.py`` so probe
results transfer to training.

Tests, per the OPSD design review:

1. Preference test (style fairness): under the teacher context, a public-facing
   rationale must outscore a target-leaking rationale for the same guess.
2. Action signal: under the teacher context, the reference guess must outscore
   an alternative public-valid guess (same rationale), and the teacher-student
   gap on guess tokens is the usable OPSD advantage.
3. First-turn invariance: with an empty transcript, the teacher must not prefer
   ``<guess>TARGET</guess>`` over a strong opener (oracle-leak detector).
4. On-policy per-class gap: sample student responses and report the
   teacher-minus-student logprob gap split by token class
   (tag / reasoning / guess), i.e. where the KL signal would actually flow.

Scoring uses SGLang ``return_logprob`` over a provided continuation; the
teacher and student contexts are scored on the same base-model server, which
is exact for a cold (step-0) student.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.wordle.standalone.tasks.base import load_task  # noqa: E402
from experiments.wordle.standalone.train_opsd_baseline import (  # noqa: E402
    _truncate_output_ids_after_first_guess,
    _wordle_output_weights,
    _wordle_public_candidates,
    _wordle_public_policy_action,
    _wordle_turn_prompt_ids,
)


GENERIC_REASONING = "Choose a common candidate consistent with the clues."
OPENER_REASONING = "Choose a broad opener with common letters."
LEAK_REASONING_TEMPLATE = "The private target is {word}."


def _post_generate(url: str, payload: dict[str, Any], *, timeout: float = 300.0, attempts: int = 4) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url.rstrip("/") + "/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= int(status) < 500 and int(status) not in {408, 429}:
                text = getattr(getattr(exc, "response", None), "text", "") or ""
                raise RuntimeError(f"SGLang /generate HTTP {status}: {text[:500]}") from exc
            last_exc = exc
            time.sleep(min(2.0 * attempt, 10.0))
    assert last_exc is not None
    raise last_exc


def score_continuation(url: str, ctx_ids: list[int], resp_ids: list[int]) -> list[float]:
    """Per-token logprobs of ``resp_ids`` given ``ctx_ids`` via prompt logprobs."""
    payload = {
        "input_ids": list(ctx_ids) + list(resp_ids),
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": max(0, len(ctx_ids) - 1),
    }
    try:
        result = _post_generate(url, payload)
    except RuntimeError:
        payload["sampling_params"]["max_new_tokens"] = 1
        payload["sampling_params"]["ignore_eos"] = True
        result = _post_generate(url, payload)
    if isinstance(result, list):
        result = result[0]
    meta = result.get("meta_info", {})
    entries = meta.get("input_token_logprobs") or []
    if len(entries) < len(resp_ids):
        raise RuntimeError(
            f"input_token_logprobs too short: got {len(entries)}, need {len(resp_ids)} "
            f"(logprob_start_len={payload['logprob_start_len']})"
        )
    tail = entries[-len(resp_ids):]
    logprobs: list[float] = []
    for offset, entry in enumerate(tail):
        logprob, token_id = float(entry[0]), int(entry[1])
        if token_id != int(resp_ids[offset]):
            raise RuntimeError(
                f"scored token mismatch at offset {offset}: got id {token_id}, expected {resp_ids[offset]}"
            )
        logprobs.append(logprob)
    return logprobs


def sample_continuations(
    url: str,
    ctx_ids: list[int],
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[int]]:
    payload = {
        "input_ids": [list(ctx_ids) for _ in range(n)],
        "sampling_params": {
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            "ignore_eos": True,
            "stop": ["\n"],
        },
        "return_logprob": False,
    }
    result = _post_generate(url, payload)
    results = result if isinstance(result, list) else [result]
    outputs: list[list[int]] = []
    for item in results:
        ids = item.get("output_ids") or item.get("meta_info", {}).get("output_ids") or []
        outputs.append([int(t) for t in ids])
    return outputs


def classify_tokens(tokenizer, resp_ids: list[int]) -> list[str]:
    """Token classes via the training weight code: tag / reasoning / guess."""
    weights, _, _ = _wordle_output_weights(
        tokenizer,
        resp_ids,
        turn_weight=1.0,
        tag_token_weight=0.0,
        reasoning_token_weight=0.5,
    )
    classes = []
    for w in weights:
        if w == 1.0:
            classes.append("guess")
        elif w == 0.5:
            classes.append("reasoning")
        else:
            classes.append("tag")
    return classes


def render_response(reasoning: str, guess: str) -> str:
    return f"<reasoning>{reasoning}</reasoning><guess>{guess.upper()}</guess>"


def _class_sum(logprobs: list[float], classes: list[str], cls: str) -> float | None:
    values = [lp for lp, c in zip(logprobs, classes) if c == cls]
    if not values:
        return None
    return float(sum(values))


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.fmean(values)) if values else None


def build_probe_states(task, *, num_states: int, seed: int, max_depth: int = 4) -> list[dict[str, Any]]:
    """Mix of reference-policy and weak-player histories at varying depths."""
    rng = random.Random(seed)
    words = list(task.WORD_LIST)
    rng.shuffle(words)
    states: list[dict[str, Any]] = []
    idx = 0
    while len(states) < num_states and idx < len(words):
        target = words[idx]
        idx += 1
        depth = len(states) % max_depth  # 0..max_depth-1 completed turns
        weak_player = len(states) % 2 == 1
        history: list[tuple[str, str]] = []
        solved = False
        for _turn in range(depth):
            candidates = _wordle_public_candidates(task, target=target, history=history)
            if weak_player and history and candidates:
                guess = rng.choice(candidates)
                reasoning = GENERIC_REASONING
            else:
                guess, reasoning = _wordle_public_policy_action(task, candidates, target=target, history=history)
            feedback = task.compute_feedback(guess, target)
            history.append((guess, feedback))
            if guess == target:
                solved = True
                break
        if solved:
            continue
        states.append({"target": target, "history": history, "weak_player": weak_player})
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-url", default="http://opsd-wordle-30b-teacher-sglang.apanda.svc.cluster.local:30000")
    parser.add_argument("--student-url", default="", help="defaults to --teacher-url (exact for a cold student)")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--task", default="wordle")
    parser.add_argument("--prompt-style", default="public_reasoning_constraints")
    parser.add_argument("--teacher-prompt-style", default="public_policy_hint")
    parser.add_argument("--num-states", type=int, default=24)
    parser.add_argument("--first-turn-targets", type=int, default=16)
    parser.add_argument("--samples-per-state", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-max-new-tokens", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    student_url = args.student_url or args.teacher_url
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    task = load_task(args.task)

    def teacher_ctx(target: str, history: list[tuple[str, str]]) -> list[int]:
        return _wordle_turn_prompt_ids(
            tokenizer,
            task,
            target=target,
            history=history,
            hinted=True,
            prompt_style=args.prompt_style,
            teacher_prompt_style=args.teacher_prompt_style,
        )

    def student_ctx(target: str, history: list[tuple[str, str]]) -> list[int]:
        return _wordle_turn_prompt_ids(
            tokenizer,
            task,
            target=target,
            history=history,
            hinted=False,
            prompt_style=args.prompt_style,
            teacher_prompt_style=args.teacher_prompt_style,
        )

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))

    # ----------------------------------------------------------------- states
    states = build_probe_states(task, num_states=args.num_states, seed=args.seed)
    mid_states = [s for s in states if s["history"]]

    # ------------------------------------------------- 1+2: preference/action
    def probe_state(state: dict[str, Any]) -> dict[str, Any] | None:
        target, history = state["target"], state["history"]
        candidates = _wordle_public_candidates(task, target=target, history=history)
        ref_guess, ref_reasoning = _wordle_public_policy_action(task, candidates, target=target, history=history)
        previous = {g for g, _ in history}
        alt_guess = next((c for c in candidates if c != ref_guess and c not in previous), None)
        t_ctx = teacher_ctx(target, history)
        s_ctx = student_ctx(target, history)

        variants = {
            "ref_public": render_response(ref_reasoning, ref_guess),
            "ref_generic": render_response(GENERIC_REASONING, ref_guess),
            "leak": render_response(LEAK_REASONING_TEMPLATE.format(word=ref_guess.upper()), ref_guess),
        }
        if alt_guess is not None:
            variants["alt_generic"] = render_response(GENERIC_REASONING, alt_guess)
        if history:
            variants["repeat_generic"] = render_response(GENERIC_REASONING, history[-1][0])

        row: dict[str, Any] = {
            "target": target,
            "history": history,
            "weak_player": state["weak_player"],
            "candidate_count": len(candidates),
            "reference_guess": ref_guess,
        }
        for name, text in variants.items():
            resp_ids = encode(text)
            classes = classify_tokens(tokenizer, resp_ids)
            t_lp = score_continuation(args.teacher_url, t_ctx, resp_ids)
            s_lp = score_continuation(student_url, s_ctx, resp_ids)
            row[name] = {
                "teacher_total": float(sum(t_lp)),
                "student_total": float(sum(s_lp)),
                "teacher_guess": _class_sum(t_lp, classes, "guess"),
                "student_guess": _class_sum(s_lp, classes, "guess"),
                "teacher_reasoning": _class_sum(t_lp, classes, "reasoning"),
                "student_reasoning": _class_sum(s_lp, classes, "reasoning"),
            }
        return row

    pref_rows = [r for r in pool.map(probe_state, mid_states) if r is not None]

    # ------------------------------------------------- 3: first-turn invariance
    rng = random.Random(args.seed + 1)
    ft_targets = rng.sample(list(task.WORD_LIST), min(args.first_turn_targets, len(task.WORD_LIST)))

    def probe_first_turn(target: str) -> dict[str, Any]:
        t_ctx = teacher_ctx(target, [])
        opener_ids = encode(render_response(OPENER_REASONING, "STARE"))
        target_ids = encode(render_response(OPENER_REASONING, target))
        opener_classes = classify_tokens(tokenizer, opener_ids)
        target_classes = classify_tokens(tokenizer, target_ids)
        t_open = score_continuation(args.teacher_url, t_ctx, opener_ids)
        t_target = score_continuation(args.teacher_url, t_ctx, target_ids)
        return {
            "target": target,
            "teacher_opener_guess": _class_sum(t_open, opener_classes, "guess"),
            "teacher_target_guess": _class_sum(t_target, target_classes, "guess"),
        }

    ft_rows = list(pool.map(probe_first_turn, ft_targets))

    # ------------------------------------------------- 4: on-policy class gaps
    def probe_samples(state: dict[str, Any]) -> list[dict[str, Any]]:
        target, history = state["target"], state["history"]
        t_ctx = teacher_ctx(target, history)
        s_ctx = student_ctx(target, history)
        sampled = sample_continuations(
            student_url,
            s_ctx,
            n=args.samples_per_state,
            temperature=args.sample_temperature,
            max_new_tokens=args.sample_max_new_tokens,
        )
        out = []
        for resp_ids in sampled:
            resp_ids, text, _ = _truncate_output_ids_after_first_guess(tokenizer, resp_ids)
            if not resp_ids:
                continue
            classes = classify_tokens(tokenizer, resp_ids)
            t_lp = score_continuation(args.teacher_url, t_ctx, resp_ids)
            s_lp = score_continuation(student_url, s_ctx, resp_ids)
            gaps = {"tag": [], "reasoning": [], "guess": []}
            for tlp, slp, cls in zip(t_lp, s_lp, classes):
                gaps[cls].append(tlp - slp)
            out.append(
                {
                    "target": target,
                    "text": text,
                    "gap_tag_mean": _mean(gaps["tag"]),
                    "gap_reasoning_mean": _mean(gaps["reasoning"]),
                    "gap_guess_mean": _mean(gaps["guess"]),
                    "gap_guess_sum": float(sum(gaps["guess"])) if gaps["guess"] else None,
                }
            )
        return out

    sample_rows = [row for rows in pool.map(probe_samples, states) for row in rows]

    # ----------------------------------------------------------------- summary
    def agg(rows: list[dict[str, Any]], a: str, b: str, field: str) -> float | None:
        diffs = []
        for r in rows:
            if a in r and b in r and r[a].get(field) is not None and r[b].get(field) is not None:
                diffs.append(r[a][field] - r[b][field])
        return _mean(diffs)

    summary = {
        "teacher_prompt_style": args.teacher_prompt_style,
        "prompt_style": args.prompt_style,
        "model": args.model,
        "num_mid_states": len(pref_rows),
        "num_first_turn_targets": len(ft_rows),
        "num_student_samples": len(sample_rows),
        # style fairness: public rationale vs leaking rationale (same guess), reasoning tokens
        "teacher_public_minus_leak_reasoning": agg(pref_rows, "ref_public", "leak", "teacher_reasoning"),
        "student_public_minus_leak_reasoning": agg(pref_rows, "ref_public", "leak", "student_reasoning"),
        # action signal: reference guess vs alternative candidate (same rationale), guess tokens
        "teacher_ref_minus_alt_guess": agg(pref_rows, "ref_generic", "alt_generic", "teacher_guess"),
        "student_ref_minus_alt_guess": agg(pref_rows, "ref_generic", "alt_generic", "student_guess"),
        # validity signal: reference guess vs repeating the previous guess
        "teacher_ref_minus_repeat_guess": agg(pref_rows, "ref_generic", "repeat_generic", "teacher_guess"),
        "student_ref_minus_repeat_guess": agg(pref_rows, "ref_generic", "repeat_generic", "student_guess"),
        # first-turn oracle leak: positive means the teacher prefers the hidden target over STARE
        "first_turn_target_minus_opener": _mean(
            [
                r["teacher_target_guess"] - r["teacher_opener_guess"]
                for r in ft_rows
                if r["teacher_target_guess"] is not None and r["teacher_opener_guess"] is not None
            ]
        ),
        # on-policy KL flow by token class (teacher - student logprob per token)
        "onpolicy_gap_tag_mean": _mean([r["gap_tag_mean"] for r in sample_rows]),
        "onpolicy_gap_reasoning_mean": _mean([r["gap_reasoning_mean"] for r in sample_rows]),
        "onpolicy_gap_guess_mean": _mean([r["gap_guess_mean"] for r in sample_rows]),
    }

    report = {"summary": summary, "pref_rows": pref_rows, "first_turn_rows": ft_rows, "sample_rows": sample_rows}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
