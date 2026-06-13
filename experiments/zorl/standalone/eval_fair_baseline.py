#!/usr/bin/env python3
"""Base-model fair-env Wordle baseline (Deliverable 1).

Plays full multi-turn Wordle games on the BASE Qwen3-30B-A3B-Instruct-2507
(no LoRA, no ZORL session) using THIS worktree's ported ``tasks.wordle`` fair
env. The student prompt is ``public_reasoning_constraints`` (a public board-
state summary: green pattern / present / excluded / likely-absent — NO
candidate list, NO target; ``X`` = absent; bracketed ``<guess>[WORD]</guess>``)
with ``invalid_retries=2`` (illegal / repeated / unparseable guesses are
re-prompted without burning a turn, gated by the vendored 693-word legal list).

This is the number both ES arms (the GRPO-via-ZORL direct-reward arm and the
OPSD-KL arm) must beat. It is also the gradient GRPO baseline's starting point.

Scoring reuses ``wordle.rollout_completion`` so the eval plays games exactly
the way the ZORL client scores candidates. ``generate_turn`` POSTs to a
replica's ``/generate`` with ``lora_path=None`` (base model), spread across the
16 idle pool replicas by game index. Per-game we additionally instrument the
generate callable to measure first-try validity and invalid-retries used (which
``rollout_completion``'s return blob does not surface).

Usage (run from inside a pool replica so the headless DNS resolves):

  kubectl exec zorl-ar-sglang-0 -n apanda -- \\
    /workspace/home/xorl-sglang-internal/.venv/bin/python \\
    /workspace/home/xorl-apanda-dev-zorl-consolidated/experiments/zorl/standalone/eval_fair_baseline.py 128
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer


# Make ``tasks/`` importable regardless of cwd (mirrors zorl_client.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks import wordle  # noqa: E402
from tasks.base import Example  # noqa: E402


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT_STYLE = "public_reasoning_constraints"
INVALID_RETRIES = 2
ROLLOUT_TEMPERATURE = 0.2
ROLLOUT_MAX_NEW_TOKENS = 96
SEED = 9234
DEFAULT_N = 128
MAX_WORKERS = 16

REPLICAS = [
    f"http://zorl-ar-sglang-{i}.zorl-ar-sglang-headless.apanda.svc.cluster.local:30000"
    for i in range(16)
]


def _generate(url: str, *, input_ids: list[int], temperature: float, max_new_tokens: int) -> str:
    """POST one /generate to a base-model replica (no LoRA). Retries transient
    connection/5xx failures a few times, then surfaces the error."""
    sampling = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "n": 1,
    }
    payload = {"input_ids": list(input_ids), "sampling_params": sampling, "return_logprob": False}
    last_err = None
    for attempt in range(6):
        try:
            with requests.post(
                f"{url}/generate", json=payload, headers={"Connection": "close"}, timeout=300.0
            ) as response:
                if response.status_code >= 400:
                    last_err = RuntimeError(f"/generate {url} HTTP {response.status_code}: {response.text[:300]}")
                    if response.status_code in {429, 503} and attempt + 1 < 6:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise last_err
                data = response.json()
                if isinstance(data, list):
                    data = data[0] if data else {}
                return str((data or {}).get("text", "") or "")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            if attempt + 1 < 6:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"/generate on {url} failed after retries: {last_err}")


def _play_one_game(url: str, tokenizer, example: Example, ns: SimpleNamespace) -> dict[str, float]:
    """Play one full game through ``wordle.rollout_completion`` and also count
    first-try validity and invalid-retries used.

    ``rollout_completion`` re-prompts internally on an invalid guess. We wrap
    the generate callable so we can tell, per game turn, whether the FIRST
    generation for that turn produced a guess that passed ``is_valid_guess``
    (against the current history), and how many extra (retry) generations were
    spent. ``rollout_completion`` rebuilds the system+user prompt afresh at the
    start of each real turn, so a turn boundary is detectable by the presence of
    the system prompt token prefix; instead of fragile prefix matching we count
    turns/retries directly off the validity outcome of each generation, in the
    same order ``rollout_completion`` issues them."""
    history: list[tuple[str, str]] = []
    # Per-turn accounting, reconstructed from the validity gate.
    counters = {"first_try_valid": 0, "turns_first_attempted": 0, "retries_used": 0}
    pending = {"is_first_attempt_of_turn": True}

    def tracking_generate(input_ids, *, lora_path, temperature, max_new_tokens, stop=None):
        text = _generate(
            url, input_ids=input_ids, temperature=temperature, max_new_tokens=max_new_tokens
        )
        guess = wordle.extract_guess(text or "")
        valid = wordle.is_valid_guess(guess, history)
        if pending["is_first_attempt_of_turn"]:
            counters["turns_first_attempted"] += 1
            if valid:
                counters["first_try_valid"] += 1
            # If this first attempt is valid, the turn advances and the next
            # generate call starts a new turn; otherwise the next call is a retry.
            pending["is_first_attempt_of_turn"] = bool(valid)
        else:
            counters["retries_used"] += 1
            # A valid retry advances the turn; the next call is a fresh turn.
            pending["is_first_attempt_of_turn"] = bool(valid)
        if valid:
            assert guess is not None
            # Mirror the history the task will build, so subsequent validity
            # checks (repeat detection) and the next turn's board summary match.
            history.append((guess, wordle.compute_feedback(guess, ns.target)))
        return text

    blob = wordle.rollout_completion(
        example,
        generate_turn=tracking_generate,
        lora_path=None,
        tokenizer=tokenizer,
        args=ns,
    )
    turns_first = max(counters["turns_first_attempted"], 1)
    return {
        "exact_match": float(blob["exact_match"]),
        "format_rate": float(blob["format_rate"]),
        "info_gain": float(blob["info_gain"]),
        "turns_used": float(blob["turns_used"]),
        "reward": float(blob["reward"]),
        "valid_guess_rate_first_try": counters["first_try_valid"] / turns_first,
        "invalid_retries_used": float(counters["retries_used"]),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    print(
        f"[init] base-model fair-env Wordle baseline: model={MODEL} prompt_style={PROMPT_STYLE} "
        f"invalid_retries={INVALID_RETRIES} temp={ROLLOUT_TEMPERATURE} max_new_tokens={ROLLOUT_MAX_NEW_TOKENS} "
        f"N={n} seed={SEED} replicas={len(REPLICAS)}",
        flush=True,
    )
    print(
        f"[init] legal-word source={wordle.LEGAL_GUESSES_SOURCE} "
        f"legal_size={len(wordle.LEGAL_GUESSES)} word_list_size={len(wordle.WORD_LIST)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)

    # N distinct eval targets. build_examples splits train/eval disjointly; we
    # use train_size=0 and eval_size=n so all N come from the eval slice.
    _train, eval_examples = wordle.build_examples(tokenizer, train_size=0, eval_size=n, seed=SEED)
    if len(eval_examples) != n:
        raise RuntimeError(f"expected {n} eval examples, got {len(eval_examples)}")

    started = time.time()
    results: list[dict[str, float]] = []

    def run_game(index_example):
        index, example = index_example
        url = REPLICAS[index % len(REPLICAS)]
        ns = SimpleNamespace(
            wordle_prompt_style=PROMPT_STYLE,
            invalid_retries=INVALID_RETRIES,
            rollout_temperature=ROLLOUT_TEMPERATURE,
            rollout_max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
            target=str(example.metadata["target"]).lower(),
        )
        return _play_one_game(url, tokenizer, example, ns)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n)) as pool:
        futures = {pool.submit(run_game, (i, ex)): i for i, ex in enumerate(eval_examples)}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 16 == 0 or done == n:
                print(f"[progress] {done}/{n} games played ({time.time() - started:.1f}s)", flush=True)

    count = max(len(results), 1)
    metrics = {
        "n": len(results),
        "model": MODEL,
        "prompt_style": PROMPT_STYLE,
        "invalid_retries": INVALID_RETRIES,
        "rollout_temperature": ROLLOUT_TEMPERATURE,
        "rollout_max_new_tokens": ROLLOUT_MAX_NEW_TOKENS,
        "seed": SEED,
        "legal_word_source": wordle.LEGAL_GUESSES_SOURCE,
        "legal_word_size": len(wordle.LEGAL_GUESSES),
        "solve_rate": sum(r["exact_match"] for r in results) / count,
        "format_rate": sum(r["format_rate"] for r in results) / count,
        "info_gain": sum(r["info_gain"] for r in results) / count,
        "valid_guess_rate_first_try": sum(r["valid_guess_rate_first_try"] for r in results) / count,
        "avg_invalid_retries_used": sum(r["invalid_retries_used"] for r in results) / count,
        "avg_turns": sum(r["turns_used"] for r in results) / count,
        "reward_mean": sum(r["reward"] for r in results) / count,
        "wall_seconds": time.time() - started,
    }
    print("[result] " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
