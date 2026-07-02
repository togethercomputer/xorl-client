#!/usr/bin/env python3
"""Honest held-out Wordle eval for a River-trained checkpoint (retries=0).

Reconstructs the SAME 170 held-out targets the training run reserved via
WORDLE_TRAIN_EXCLUDE_SEED/COUNT (random.Random(seed).shuffle(WORD_LIST)[:count]
— matches tasks/wordle.build_examples and the SGLang floor-eval _pick_targets),
loads a River checkpoint, plays each target multi-turn with ONE sample per turn
(no retry crutch), and reports the solve rate. This is the river-native analogue
of the SGLang honest eval; compare against GRPO-WQ36-LORA16-IS-control (~0.65
held-out) and the untrained base (~0.00).

Usage:
  eval_wordle_river.py --checkpoint river://<run>/weights/final [--base-model ...]
  eval_wordle_river.py --base-model Qwen/Qwen3.6-35B-A3B-FP8   # untrained base
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import river_client as river  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from experiments.wordle.standalone.tasks.base import load_task  # noqa: E402
from experiments.wordle.standalone.train_grpo_wordle_river import (  # noqa: E402
    Trajectory, TurnSample, _score_trajectory, _wordle_is_valid_guess, render_turn_prompt,
)


def heldout_targets(task, seed: int, count: int) -> list[str]:
    """The reserved held-out words: the same shuffle+prefix build_examples excludes."""
    reserve = list(task.WORD_LIST)
    random.Random(seed).shuffle(reserve)
    return [w.lower() for w in reserve[:count]]


def play(model, tokenizer, task, targets, *, prompt_style, temperature, top_p,
         max_new_tokens, max_length, chunk, seed, timeout):
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    trajs = [Trajectory(project=f"eval_{i}", target=t, group_id=f"eval_{i}", rollout_id=0,
                        prompt_style=prompt_style) for i, t in enumerate(targets)]
    for turn in range(1, max_turns + 1):
        active = [t for t in trajs if not t.solved and not t.stopped_reason]
        if not active:
            break
        rendered = [render_turn_prompt(tokenizer, task, target=t.target, history=t.history,
                                       prompt_style=t.prompt_style) for t in active]
        prompt_strs = [r[0] for r in rendered]
        prompt_ids_list = [r[1] for r in rendered]
        pendings = []
        for start in range(0, len(active), chunk):
            sub = prompt_strs[start:start + chunk]
            pendings.append((start, model.submit_sample(
                sub, num_samples=1, max_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=-1, seed=(seed + turn * 101 + start) if seed is not None else None,
                return_prompt_logprobs=False, return_expert_routing=False, timeout=timeout)))
        samples = [None] * len(active)
        for start, pend in pendings:
            for j, g in enumerate(pend.result()):
                samples[start + j] = g[0] if g else None
        for traj, prompt_ids, samp in zip(active, prompt_ids_list, samples):
            text = (samp.text or "") if samp is not None else ""
            action = task.extract_action_text(text) if hasattr(task, "extract_action_text") else text
            guesses = task.extract_guesses(action) if hasattr(task, "extract_guesses") else []
            guess = guesses[-1] if guesses else (task.extract_guess(action) if hasattr(task, "extract_guess") else None)
            valid = bool(guess is not None and _wordle_is_valid_guess(task, guess, traj.history))
            single = bool(task.has_single_guess_tag(action)) if hasattr(task, "has_single_guess_tag") else bool(guess)
            fb = ""
            solved = False
            if valid:
                fb = task.compute_feedback(guess, traj.target)
                traj.history.append((guess, fb))
                solved = guess == traj.target
                traj.solved = solved
                if solved:
                    traj.stopped_reason = "solved"
                elif len(traj.history) >= max_turns:
                    traj.stopped_reason = "max_turns"
            else:
                traj.stopped_reason = "invalid"
            traj.samples.append(TurnSample(
                project=traj.project, target=traj.target, rollout_id=0, turn=turn,
                prompt_ids=list(prompt_ids), output_ids=list(samp.tokens) if samp else [],
                old_logprobs=[], text=text, guess=guess, single_guess_tag=single,
                format_ok=bool(single and guess is not None), valid_guess=valid, feedback=fb, solved=solved))
    for t in trajs:
        if not t.stopped_reason:
            t.stopped_reason = "max_turns"
        t.score = _score_trajectory(task, t)
    return trajs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=os.environ.get("RIVER_ENDPOINT", "api.river.ai"))
    p.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--checkpoint", default="", help="river://.../weights/<name>; empty = untrained base")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--exclude-seed", type=int, default=777)
    p.add_argument("--exclude-count", type=int, default=170)
    p.add_argument("--wordle-prompt-style", default="public_reasoning_constraints_think")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=1, help="independent plays per target (majority not used; solve if ANY solves)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--sample-chunk", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--output", default="")
    args = p.parse_args()

    api_key = os.environ["RIVER_API_KEY"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    task = load_task("wordle")
    targets = heldout_targets(task, args.exclude_seed, args.exclude_count)
    print(f"[eval] {len(targets)} held-out targets (seed={args.exclude_seed}); "
          f"checkpoint={args.checkpoint or '(untrained base)'} temp={args.temperature} "
          f"num_samples={args.num_samples}", flush=True)

    client = river.Client(api_key=api_key, endpoint=args.endpoint)
    with client.session(project="wordle-grpo-river", run="eval") as session:
        model = session.create_model(
            base_model=args.base_model,
            lora=river.LoraConfig(rank=args.lora_rank),
            checkpoint=(args.checkpoint or None),
            timeout=args.timeout,
        )
        t0 = time.time()
        solved_flags = {t: False for t in targets}
        all_scores = []
        for rep in range(args.num_samples):
            trajs = play(model, tokenizer, task, targets, prompt_style=args.wordle_prompt_style,
                         temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                         max_length=args.max_length, chunk=args.sample_chunk,
                         seed=args.seed + rep * 100_003, timeout=args.timeout)
            for t in trajs:
                if t.solved:
                    solved_flags[t.target] = True
            all_scores.extend(trajs)
            solve_rate = sum(solved_flags.values()) / len(targets)
            print(f"[eval rep={rep+1}/{args.num_samples}] cumulative solve_rate={solve_rate:.4f} "
                  f"({sum(solved_flags.values())}/{len(targets)}) dt={time.time()-t0:.0f}s", flush=True)

    n = len(all_scores)
    def mean(k): return sum(s.score.get(k, 0.0) for s in all_scores) / max(n, 1)
    summary = {
        "checkpoint": args.checkpoint or "base",
        "n_targets": len(targets),
        "num_samples": args.num_samples,
        "solve_rate_any": sum(solved_flags.values()) / len(targets),
        "per_play_exact_match": mean("exact_match"),
        "per_play_format_rate": mean("format_rate"),
        "per_play_valid_guess_rate": mean("valid_guess_rate"),
        "per_play_reward": mean("reward"),
        "per_play_wordle_retrieval_reward": mean("wordle_retrieval_reward"),
        "temperature": args.temperature,
    }
    print("[eval] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
