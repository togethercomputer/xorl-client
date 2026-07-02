#!/usr/bin/env python3
"""Wordle GRPO training loop against the managed River API (river_client).

Port of ``train_grpo_wordle.py`` from the in-house xorl-client + SGLang samplers +
SMG + explicit-weight-sync k8s stack onto River's managed training API. The whole
samplers/SMG/export/broadcast machinery collapses because ``model.sample()`` reads
the model's live in-memory training weights — weight sync is implicit.

What is reused verbatim from ``tasks/wordle.py`` (pure, no xorl/SGLang coupling):
  env + reward (compute_feedback, remaining_candidates, wordle_retrieval_reward),
  guess parsing (extract_guesses / extract_guess / has_single_guess_tag /
  extract_action_text), prompt builder (_build_turn_messages), build_examples.

What is copied here (was xorl-coupled by file, not by logic): the GRPO math
(compute_grpo_advantages, build_policy_loss_inputs), _score_trajectory /
_invalid_reason, the cosine LR schedule, and the Trajectory/TurnSample dataclasses.

What is reimplemented against River: rollout (model.sample, multi-turn, live
weights), datum build (input_ids/target_tokens/old_logprobs/advantages + R3
routing keys), and the fwd/bwd + Adam optim step.

River specifics validated by the pre-flight probe:
  * base model  = Qwen/Qwen3.6-35B-A3B-FP8 (get_capabilities); LoRA rank 16 OK.
  * loss_fn="importance_sampling" accepted; returns loss/kl/mean_ratio/grad_norm.
  * sample() returns per-token logprobs aligned to tokens; prompt_token_ids is
    NOT returned, but the server tokenizes our rendered chat STRING identically
    to the local tokenizer (verified: routing num_tokens == prompt+gen-1), so we
    tokenize prompts client-side for prompt_len / input_ids.
  * expert routing is returned as an NFS reference (Sample.routing_datum_keys()
    -> {"expert_routing_nfs_path": ...}); replay via force_routing_replay.

Optimizer note: River is Adam-only (no muon). Per the run owner, use lr = 5e-4
(= 10x the muon 5e-5, scaled up for LoRA), cosine, min_lr = 5e-5, warmup 8.
LoraConfig has no client-side alpha (server-fixed) — a divergence from alpha 32.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import river_client as river  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from experiments.wordle.standalone.tasks.base import Example, load_task  # noqa: E402


# --------------------------------------------------------------------------- #
# GRPO math (copied from train_grpo_wordle.py inlined fallbacks, verbatim)     #
# --------------------------------------------------------------------------- #
def compute_grpo_advantages(rewards, *, group_ids, normalize=True,
                            std_normalization=True, eps=1e-8):
    """Per-group mean/std-normalized advantages (GRPO)."""
    rewards = [float(r) for r in rewards]
    if len(rewards) != len(group_ids):
        raise ValueError(f"rewards/group_ids mismatch: {len(rewards)} vs {len(group_ids)}")
    from collections import defaultdict
    by_group = defaultdict(list)
    for r, g in zip(rewards, group_ids):
        by_group[g].append(r)
    stats = {}
    for g, rs in by_group.items():
        mean = sum(rs) / len(rs)
        if std_normalization and len(rs) > 1:
            std = math.sqrt(sum((x - mean) ** 2 for x in rs) / len(rs))
        else:
            std = 0.0
        stats[g] = (mean, std)
    out = []
    for r, g in zip(rewards, group_ids):
        mean, std = stats[g]
        adv = (r - mean) if normalize else r
        if std_normalization:
            adv = adv / (std + eps)
        out.append(float(adv))
    return out


def build_policy_loss_inputs(tokens, prompt_len, old_logprobs, advantage, *, ignore_index=-100):
    """Next-token-shifted per-token {target_tokens, logprobs, advantages}, masked
    to generation positions only (ignore_index on prompt). Aligned to tokens[:-1]."""
    n = len(tokens)
    seq = max(0, n - 1)
    target_tokens = [ignore_index] * seq
    logprobs = [0.0] * seq
    advantages = [0.0] * seq
    old_logprobs = [float(x) for x in old_logprobs]
    for i in range(seq):
        tgt_pos = i + 1
        if tgt_pos >= prompt_len:
            target_tokens[i] = int(tokens[tgt_pos])
            gen_idx = tgt_pos - prompt_len
            if 0 <= gen_idx < len(old_logprobs):
                logprobs[i] = old_logprobs[gen_idx]
            advantages[i] = float(advantage)
    return {"target_tokens": target_tokens, "logprobs": logprobs, "advantages": advantages}


def get_lr(args, step: int) -> float:
    """Client-side LR schedule (1-indexed step). Matches train_grpo_wordle.get_lr."""
    if args.lr_schedule != "cosine":
        return args.lr
    w = max(0, args.lr_warmup_steps)
    if w > 0 and step <= w:
        return args.lr * step / w
    denom = max(1, args.steps - w)
    progress = min(1.0, max(0.0, (step - w) / denom))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Small local helpers                                                         #
# --------------------------------------------------------------------------- #
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _jsonl(path: Path, obj: dict) -> None:
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _retry(fn, what, *, attempts=4, base_delay=15.0):
    """Retry a River API call on transient connection/auth blips with backoff.
    A persistently-invalid key (e.g. credits exhausted) exhausts retries and raises."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except river.RiverError as e:
            last = e
            print(f"[retry] {what} attempt {i + 1}/{attempts} failed: {e}", flush=True)
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise last


def _wordle_is_valid_guess(task, guess, history) -> bool:
    if guess is None or len(guess) != 5 or not guess.isalpha():
        return False
    if hasattr(task, "is_valid_guess"):
        return bool(task.is_valid_guess(guess, history))
    return guess.lower() not in {prior for prior, _ in history}


def _invalid_reason(task, text, guess, history) -> str:
    if hasattr(task, "has_single_guess_tag") and not task.has_single_guess_tag(text):
        return "not_exactly_one_guess_tag"
    if guess is None:
        return "no_parseable_five_letter_guess"
    normalized = guess.lower()
    if normalized in {prior for prior, _ in history}:
        return "repeated_guess"
    if len(normalized) != 5 or not normalized.isalpha():
        return "not_five_alpha_letters"
    legal = getattr(task, "LEGAL_GUESSES", None)
    if legal is not None and normalized not in legal:
        return "not_in_legal_wordle_dictionary"
    return "unknown_invalid"


# --------------------------------------------------------------------------- #
# Dataclasses (river-flavored: expert_routing is a river ExpertRouting object) #
# --------------------------------------------------------------------------- #
@dataclass
class TurnSample:
    project: str
    target: str
    rollout_id: int
    turn: int
    prompt_ids: list[int]
    output_ids: list[int]
    old_logprobs: list[float]
    text: str
    guess: str | None
    single_guess_tag: bool
    format_ok: bool
    valid_guess: bool
    feedback: str
    solved: bool
    truncated: bool = False
    error: str = ""
    expert_routing: Any | None = None  # river ExpertRouting (carries nfs_path)

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


@dataclass
class Trajectory:
    project: str
    target: str
    group_id: str
    rollout_id: int
    samples: list[TurnSample] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    solved: bool = False
    stopped_reason: str = ""
    score: dict[str, float] = field(default_factory=dict)
    advantage: float = 0.0
    prompt_style: str = ""


def _score_trajectory(task, trajectory: Trajectory) -> dict[str, float]:
    """Verbatim from train_grpo_wordle.py:416 (uses only task methods)."""
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    turns_used = len(trajectory.samples)
    format_hits = sum(1 for s in trajectory.samples if s.format_ok)
    single_tag_hits = sum(1 for s in trajectory.samples if s.single_guess_tag)
    valid_hits = sum(1 for s in trajectory.samples if s.valid_guess)
    info_scores = [
        float(task._info_score(s.guess, trajectory.target, s.feedback))
        for s in trajectory.samples
        if s.valid_guess and s.guess is not None and s.feedback
    ]
    latest_feedback = ""
    for s in reversed(trajectory.samples):
        if s.feedback:
            latest_feedback = s.feedback
            break
    format_rate = format_hits / max(turns_used, 1)
    single_guess_tag_rate = single_tag_hits / max(turns_used, 1)
    valid_guess_rate = valid_hits / max(turns_used, 1)
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    terminal_valid = bool(
        trajectory.solved
        or (trajectory.stopped_reason == "max_turns" and turns_used > 0 and valid_hits == turns_used)
    )
    invalid_action = bool(not terminal_valid or any(not s.valid_guess for s in trajectory.samples))
    if hasattr(task, "_wordle_shaped_reward"):
        reward = task._wordle_shaped_reward(
            solved=trajectory.solved, format_rate=format_rate, valid_guess_rate=valid_guess_rate,
            info_gain=info_gain, turns_used=turns_used, max_turns=max_turns, invalid_action=invalid_action,
        )
    else:
        turn_bonus = ((max_turns - turns_used + 1) / max_turns) if trajectory.solved else 0.0
        reward = 0.4 * float(trajectory.solved) + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus
    wordle_components = {}
    if hasattr(task, "_wordle_reward_components"):
        wordle_components = task._wordle_reward_components(
            solved=trajectory.solved, turns_with_guess=valid_hits, latest_feedback=latest_feedback,
            format_reward=single_guess_tag_rate, valid_guess_rate=valid_guess_rate,
            terminal_valid=terminal_valid, invalid_action=invalid_action,
        )
    retrieval_components: dict[str, float] = {}
    if hasattr(task, "wordle_retrieval_reward"):
        played = [(str(s.guess), str(s.feedback)) for s in trajectory.samples if s.guess and s.feedback]
        retrieval_components = task.wordle_retrieval_reward(
            played, solved=trajectory.solved, invalid_action=invalid_action,
            valid_guess_rate=valid_guess_rate, format_rate=single_guess_tag_rate,
        )
    return {
        "reward": float(reward),
        "exact_match": float(trajectory.solved),
        "format_rate": float(format_rate),
        "single_guess_tag_rate": float(single_guess_tag_rate),
        "valid_guess_rate": float(valid_guess_rate),
        "info_gain": float(info_gain),
        "turns_used": float(turns_used),
        "invalid_action": float(invalid_action),
        "terminal_valid": float(terminal_valid),
        **{k: float(v) for k, v in wordle_components.items()},
        **{k: float(v) for k, v in retrieval_components.items()},
    }


# --------------------------------------------------------------------------- #
# Prompt rendering (STRING for River + local token ids for prompt_len)        #
# --------------------------------------------------------------------------- #
def render_turn_prompt(tokenizer, task, *, target, history, prompt_style):
    """Return (prompt_string, prompt_ids). The server tokenizes the string
    identically to prompt_ids (verified), so prompt_len = len(prompt_ids)."""
    msgs = task._build_turn_messages(target=target, history=history, prompt_style=prompt_style)
    enable_thinking = prompt_style.endswith("_think")
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False
    )
    return text, list(ids)


def routing_keys(er, routing_mode: str) -> dict:
    """R3 per-datum routing keys for force_routing_replay, built directly from a
    river ExpertRouting object (mirrors Sample.routing_datum_keys). Prefers inline
    bytes, falls back to the shipped NFS-reference key. Empty when unavailable/off."""
    if er is None or not routing_mode:
        return {}
    if getattr(er, "topk_ids", b"") and getattr(er, "topk_weights", b""):
        return {"expert_routing_topk_ids": bytes(er.topk_ids),
                "expert_routing_topk_weights": bytes(er.topk_weights)}
    if getattr(er, "nfs_path", ""):
        return {"expert_routing_nfs_path": er.nfs_path.encode("utf-8")}
    return {}


# --------------------------------------------------------------------------- #
# Rollout (multi-turn, batched) against model.sample()                         #
# --------------------------------------------------------------------------- #
def rollout(model, tokenizer, task, examples, *, group_size, args, step, log_path):
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    prompt_style = args.wordle_prompt_style
    trajectories: list[Trajectory] = []
    for ex in examples:
        target = str(ex.metadata["target"]).lower()
        for rid in range(group_size):
            trajectories.append(Trajectory(
                project=ex.project, target=target, group_id=ex.project,
                rollout_id=rid, prompt_style=prompt_style,
            ))

    t0 = time.time()
    for turn in range(1, max_turns + 1):
        active = [t for t in trajectories if not t.solved and not t.stopped_reason]
        if not active:
            break
        rendered = [render_turn_prompt(tokenizer, task, target=t.target, history=t.history,
                                       prompt_style=t.prompt_style) for t in active]
        prompt_strs = [r[0] for r in rendered]
        prompt_ids_list = [r[1] for r in rendered]

        # Dispatch all chunks of this turn CONCURRENTLY (submit -> collect) so
        # River runs them in parallel across its fleet rather than one-at-a-time.
        # Per-rollout seed is offset by the chunk start so seeds are globally
        # unique across the turn (River further varies seed+prompt_idx per prompt).
        chunk = max(1, args.sample_chunk)
        base_seed = args.seed * 1_000_003 + step * 10_007 + turn * 101

        def _sample_turn():
            pend = []
            for start in range(0, len(active), chunk):
                sub = prompt_strs[start:start + chunk]
                seed = (base_seed + start) if args.per_rollout_seed else None
                pend.append((start, model.submit_sample(
                    sub, num_samples=1, max_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=-1,
                    seed=seed, return_prompt_logprobs=False,
                    return_expert_routing=args.return_expert_routing, timeout=args.sample_timeout)))
            out: list[Any] = [None] * len(active)
            for start, p in pend:
                for j, g in enumerate(p.result()):
                    out[start + j] = g[0] if g else None
            return out

        samples = _retry(_sample_turn, f"sample step={step} turn={turn}")

        for traj, prompt_ids, samp in zip(active, prompt_ids_list, samples):
            if samp is None:
                traj.stopped_reason = "sampling_error"
                traj.samples.append(TurnSample(
                    project=traj.project, target=traj.target, rollout_id=traj.rollout_id, turn=turn,
                    prompt_ids=list(prompt_ids), output_ids=[], old_logprobs=[], text="",
                    guess=None, single_guess_tag=False, format_ok=False, valid_guess=False,
                    feedback="", solved=False, error="empty_sample",
                ))
                continue
            output_ids = list(samp.tokens)
            old_logprobs = list(samp.logprobs)
            text = samp.text or ""
            error = ""
            truncated = False
            # logprob/token alignment guard (River returns per-gen-token logprobs).
            if len(old_logprobs) < len(output_ids):
                error = f"logprob_count_mismatch: out={len(output_ids)} lp={len(old_logprobs)}"
                output_ids, old_logprobs = [], []
            else:
                old_logprobs = old_logprobs[:len(output_ids)]
            # max_length clip
            budget = max(0, args.max_length - len(prompt_ids))
            if output_ids and len(output_ids) > budget:
                output_ids = output_ids[:budget]
                old_logprobs = old_logprobs[:budget]
                text = tokenizer.decode(output_ids, skip_special_tokens=False)
                truncated = True

            action_text = task.extract_action_text(text) if hasattr(task, "extract_action_text") else text
            guesses = task.extract_guesses(action_text) if hasattr(task, "extract_guesses") else []
            guess = guesses[-1] if guesses else (task.extract_guess(action_text) if hasattr(task, "extract_guess") else None)
            single_guess_tag = bool(task.has_single_guess_tag(action_text)) if hasattr(task, "has_single_guess_tag") else bool(guess)
            format_ok = bool(single_guess_tag and guess is not None)
            valid_guess = bool(guess is not None and _wordle_is_valid_guess(task, guess, traj.history))
            feedback = ""
            solved = False
            if valid_guess:
                feedback = task.compute_feedback(guess, traj.target)
                traj.history.append((guess, feedback))
                solved = guess == traj.target
                traj.solved = solved
                if solved:
                    traj.stopped_reason = "solved"
                elif len(traj.history) >= max_turns:
                    traj.stopped_reason = "max_turns"
            else:
                traj.stopped_reason = _invalid_reason(task, text, guess, traj.history)

            traj.samples.append(TurnSample(
                project=traj.project, target=traj.target, rollout_id=traj.rollout_id, turn=turn,
                prompt_ids=list(prompt_ids), output_ids=list(output_ids), old_logprobs=list(old_logprobs),
                text=text, guess=guess, single_guess_tag=single_guess_tag, format_ok=format_ok,
                valid_guess=valid_guess, feedback=feedback, solved=solved, truncated=truncated,
                error=error, expert_routing=getattr(samp, "expert_routing", None),
            ))

    for traj in trajectories:
        if not traj.stopped_reason:
            traj.stopped_reason = "max_turns"
        traj.score = _score_trajectory(task, traj)
        for s in traj.samples:
            _jsonl(log_path, {
                "event": "grpo_student_turn", "step": step, "time": time.time(),
                "project": s.project, "target": s.target, "rollout_id": s.rollout_id, "turn": s.turn,
                "guess": s.guess or "", "single_guess_tag": s.single_guess_tag, "format_ok": s.format_ok,
                "valid_guess": s.valid_guess, "feedback": s.feedback, "solved": s.solved,
                "stopped_reason": traj.stopped_reason, "reward": traj.score.get("reward", 0.0),
                "wordle_retrieval_reward": traj.score.get("wordle_retrieval_reward", 0.0),
                "output_tokens": len(s.output_ids), "prompt_tokens": len(s.prompt_ids),
                "truncated": s.truncated, "error": s.error,
            })
    return trajectories, time.time() - t0


# --------------------------------------------------------------------------- #
# Datum build (xorl Datum.loss_fn_inputs -> River data dict keys)              #
# --------------------------------------------------------------------------- #
def build_river_data(trajectories, *, reward_key, normalize, std_normalization,
                     normalize_advantage_by_turns, skip_zero_advantage, routing_mode):
    if not trajectories:
        return [], {"trajectory_count": 0.0}
    rewards = [float(t.score.get(reward_key, t.score.get("reward", 0.0))) for t in trajectories]
    group_ids = [t.group_id for t in trajectories]
    advantages = compute_grpo_advantages(rewards, group_ids=group_ids, normalize=normalize,
                                          std_normalization=std_normalization)
    data: list[dict] = []
    skipped_zero = skipped_empty = 0
    routed_present = 0
    for traj, adv in zip(trajectories, advantages, strict=True):
        traj.advantage = float(adv)
        if normalize_advantage_by_turns:
            tc = max(1, sum(1 for s in traj.samples if s.output_ids and s.old_logprobs))
            sample_adv = float(adv) / float(tc)
        else:
            sample_adv = float(adv)
        if skip_zero_advantage and abs(sample_adv) <= 1e-12:
            skipped_zero += len(traj.samples)
            continue
        for s in traj.samples:
            if not s.output_ids or not s.old_logprobs:
                skipped_empty += 1
                continue
            token_list = s.tokens
            if len(token_list) <= s.prompt_len:
                skipped_empty += 1
                continue
            li = build_policy_loss_inputs(token_list, s.prompt_len, s.old_logprobs, sample_adv)
            datum = {
                "input_ids": token_list[:-1],
                "target_tokens": li["target_tokens"],
                "old_logprobs": li["logprobs"],
                "advantages": li["advantages"],
            }
            # R3: attach routing keys only for non-truncated turns (truncation would
            # desync the server-side [seqlen-1,L,K] routing buffer from input_ids).
            if not s.truncated:
                rk = routing_keys(s.expert_routing, routing_mode)
                if rk:
                    datum.update(rk)
                    routed_present += 1
            data.append(datum)

    n = max(float(len(trajectories)), 1.0)
    exact = [float(t.score.get("exact_match", 0.0)) for t in trajectories]
    metrics = {
        "trajectory_count": float(len(trajectories)),
        "datum_count": float(len(data)),
        "reward_mean": sum(rewards) / n,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "advantage_mean": sum(float(a) for a in advantages) / n,
        "advantage_abs_mean": sum(abs(float(a)) for a in advantages) / n,
        "exact_match_rate": sum(exact) / n,
        "exact_count": sum(exact),
        "skipped_zero_advantage_turns": float(skipped_zero),
        "skipped_empty_turns": float(skipped_empty),
        "r3_routed_datum_count": float(routed_present),
        "r3_routed_present_rate": float(routed_present) / max(float(len(data)), 1.0),
    }
    numeric_keys = {k for t in trajectories for k, v in t.score.items() if _is_number(v)}
    for k in sorted(numeric_keys):
        metrics[f"{k}_mean"] = sum(float(t.score.get(k, 0.0)) for t in trajectories) / n
    return data, metrics


# --------------------------------------------------------------------------- #
# Client-side k3 (Schulman KL estimator: sampler vs trainer logprobs)          #
# --------------------------------------------------------------------------- #
def compute_k3(data, fb_logprobs) -> dict[str, float]:
    """k3 = exp(delta) - delta - 1, delta = trainer_lp - sampler_lp, over gen tokens.
    Aligned via the target_tokens != -100 mask (same masking as advantages)."""
    if not fb_logprobs:
        return {}
    deltas: list[float] = []
    aligned = mismatched = 0
    for datum, tr in zip(data, fb_logprobs):
        tgt = np.asarray(datum["target_tokens"])
        samp = np.asarray(datum["old_logprobs"], dtype=np.float64)
        train = np.asarray(tr, dtype=np.float64)
        if train.shape[0] != tgt.shape[0]:
            mismatched += 1
            continue
        mask = tgt != -100
        d = train[mask] - samp[mask]
        if d.size:
            deltas.append(d)
            aligned += 1
    if not deltas:
        return {"k3_aligned_datums": float(aligned), "k3_mismatched_datums": float(mismatched)}
    delta = np.concatenate(deltas)
    k3 = np.exp(delta) - delta - 1.0
    return {
        "k3_mean": float(k3.mean()),
        "k3_p90": float(np.percentile(k3, 90)),
        "k3_max": float(k3.max()),
        "ratio_mean": float(np.exp(delta).mean()),
        "abs_delta_mean": float(np.abs(delta).mean()),
        "k3_aligned_datums": float(aligned),
        "k3_mismatched_datums": float(mismatched),
        "k3_gen_tokens": float(delta.size),
    }


def _metrics(obj) -> dict[str, float]:
    return {str(k): float(v) for k, v in (getattr(obj, "metrics", {}) or {}).items() if _is_number(v)}


# --------------------------------------------------------------------------- #
# Example selection per step                                                   #
# --------------------------------------------------------------------------- #
def select_examples(train_pool, train_size, seed, step):
    rng = random.Random(seed * 7_919 + step)
    if len(train_pool) <= train_size:
        return list(train_pool)
    return rng.sample(train_pool, train_size)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wordle GRPO on the River managed API")
    p.add_argument("--endpoint", default=os.environ.get("RIVER_ENDPOINT", "api.river.ai"))
    p.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--tokenizer", default="", help="override tokenizer id/path (default: base-model)")
    p.add_argument("--task", default="wordle")
    p.add_argument("--run-name", default="GRPO-WQ36-LORA16-IS-river")
    p.add_argument("--output-dir", default="")
    # LoRA (alpha is server-fixed; not settable client-side)
    p.add_argument("--lora-rank", type=int, default=16)
    # loss / RL
    p.add_argument("--loss-fn", default="importance_sampling")
    p.add_argument("--logprob-temperature", type=float, default=0.7)
    p.add_argument("--compute-kl-stats", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reward-key", default="wordle_retrieval_reward")
    p.add_argument("--wordle-prompt-style", default="public_reasoning_constraints_think")
    p.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--std-normalization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-advantage-by-turns", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-zero-advantage", action=argparse.BooleanOptionalAction, default=True)
    # optimizer (Adam only on River)
    p.add_argument("--lr", type=float, default=5e-4)          # 10x muon 5e-5 (LoRA)
    p.add_argument("--min-lr", type=float, default=5e-5)      # 10x muon 5e-6
    p.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    p.add_argument("--lr-warmup-steps", type=int, default=8)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 disables (River grad_clip_norm=None)")
    # rollout / batch
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--train-size", type=int, default=32)
    p.add_argument("--train-pool-size", type=int, default=4096)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--per-rollout-seed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-chunk", type=int, default=128, help="prompts per model.sample() call")
    p.add_argument("--sample-timeout", type=float, default=3600.0)
    p.add_argument("--op-timeout", type=float, default=3600.0)
    # R3 routing
    p.add_argument("--return-expert-routing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--routing-mode", default="ids+weights", choices=["", "ids+weights", "ids"])
    p.add_argument("--compute-expert-flip-metric", action=argparse.BooleanOptionalAction, default=False)
    # misc
    p.add_argument("--seed", type=int, default=9234)
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--wandb-log-samples", type=int, default=8)
    p.add_argument("--train-exclude-seed", type=int, default=777)
    p.add_argument("--train-exclude-count", type=int, default=170)
    p.add_argument("--max-runtime-seconds", type=float, default=0.0)
    return p


def main() -> int:
    args = build_argparser().parse_args()

    api_key = os.environ.get("RIVER_API_KEY")
    if not api_key:
        print("FATAL: RIVER_API_KEY not set", flush=True)
        return 2

    # held-out disjointness: same env contract the eval + build_examples use.
    if args.train_exclude_count > 0:
        os.environ["WORDLE_TRAIN_EXCLUDE_SEED"] = str(args.train_exclude_seed)
        os.environ["WORDLE_TRAIN_EXCLUDE_COUNT"] = str(args.train_exclude_count)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (out_dir / "grpo_turns.jsonl") if out_dir else None

    tok_id = args.tokenizer or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    task = load_task(args.task)
    train_pool, eval_examples = task.build_examples(
        tokenizer, train_size=args.train_pool_size, eval_size=args.eval_size, seed=args.seed
    )
    print(f"[setup] base_model={args.base_model} train_pool={len(train_pool)} eval={len(eval_examples)} "
          f"lr={args.lr} loss_fn={args.loss_fn} routing_mode={args.routing_mode!r}", flush=True)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["optimizer"] = "adam"
    wandb_run = None
    if args.wandb_project:
        import wandb  # noqa: PLC0415
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or args.run_name,
                               config=run_config)

    client = river.Client(api_key=api_key, endpoint=args.endpoint)
    caps = client.get_capabilities()
    if args.base_model not in caps:
        print(f"WARNING: base_model {args.base_model!r} not in capabilities {caps}", flush=True)

    grad_clip = None if args.grad_clip <= 0 else float(args.grad_clip)
    loss_config = {"logprob_temperature": float(args.logprob_temperature),
                   "compute_kl_stats": 1.0 if args.compute_kl_stats else 0.0}

    total_prompt_tokens = total_gen_tokens = total_train_tokens = 0
    t_start = time.time()

    with client.session(project="wordle-grpo-river", run=args.run_name) as session:
        model = session.create_model(
            base_model=args.base_model,
            lora=river.LoraConfig(rank=args.lora_rank, train_attn=True, train_mlp=True,
                                  train_unembed=False, seed=args.seed),
            timeout=args.op_timeout,
        )
        print(f"[setup] model created: {getattr(model, '_model_id', '?')}", flush=True)

        for step in range(1, args.steps + 1):
            if args.max_runtime_seconds and (time.time() - t_start) > args.max_runtime_seconds:
                print(f"[stop] hit max_runtime_seconds={args.max_runtime_seconds}", flush=True)
                break
            examples = select_examples(train_pool, args.train_size, args.seed, step)
            trajectories, rollout_s = rollout(model, tokenizer, task, examples,
                                              group_size=args.group_size, args=args, step=step, log_path=log_path)
            data, gmetrics = build_river_data(
                trajectories, reward_key=args.reward_key, normalize=args.normalize_advantages,
                std_normalization=args.std_normalization,
                normalize_advantage_by_turns=args.normalize_advantage_by_turns,
                skip_zero_advantage=args.skip_zero_advantage, routing_mode=args.routing_mode,
            )
            step_prompt_tok = sum(len(s.prompt_ids) for t in trajectories for s in t.samples)
            step_gen_tok = sum(len(s.output_ids) for t in trajectories for s in t.samples)
            step_train_tok = sum(len(d["input_ids"]) for d in data)
            total_prompt_tokens += step_prompt_tok
            total_gen_tokens += step_gen_tok
            total_train_tokens += step_train_tok

            train_metrics: dict[str, float] = dict(gmetrics)
            if not data:
                train_metrics["skipped_train_step"] = 1.0
                print(f"[train step={step}] no datums (all-zero advantage?) — skipping fwd/bwd", flush=True)
            else:
                fb_started = time.time()
                fb = _retry(lambda: model.forward_backward(
                    data, loss_fn=args.loss_fn, return_logprobs=True, zero_out=True,
                    force_routing_replay=args.routing_mode,
                    compute_expert_flip_metric=args.compute_expert_flip_metric,
                    timeout=args.op_timeout, **loss_config,
                ), f"forward_backward step={step}")
                train_metrics.update({f"loss/{k}": v for k, v in _metrics(fb).items()})
                train_metrics["forward_backward_time_s"] = time.time() - fb_started
                # client-side k3 (sampler vs trainer logprobs)
                train_metrics.update({f"k3/{k}": v for k, v in compute_k3(data, fb.logprobs).items()})

                opt_started = time.time()
                lr = get_lr(args, step)
                opt = _retry(lambda: model.optim_step(lr=lr, beta1=args.beta1, beta2=args.beta2,
                                       eps=args.eps, weight_decay=args.weight_decay,
                                       grad_clip_norm=grad_clip, timeout=args.op_timeout),
                             f"optim_step step={step}")
                train_metrics.update({f"optim/{k}": v for k, v in _metrics(opt).items()})
                train_metrics["optim_time_s"] = time.time() - opt_started
                train_metrics["lr"] = lr

            train_metrics["rollout_time_s"] = rollout_s
            train_metrics["step_time_s"] = (time.time() - fb_started) + rollout_s if data else rollout_s
            train_metrics["cum_prompt_tokens"] = float(total_prompt_tokens)
            train_metrics["cum_gen_tokens"] = float(total_gen_tokens)
            train_metrics["cum_train_tokens"] = float(total_train_tokens)

            print(
                f"[train step={step}] reward={train_metrics.get('reward_mean', float('nan')):.4f} "
                f"exact={train_metrics.get('exact_count', 0.0):.0f}/{train_metrics.get('trajectory_count', 0.0):.0f} "
                f"datums={train_metrics.get('datum_count', 0.0):.0f} "
                f"adv_abs={train_metrics.get('advantage_abs_mean', 0.0):.4f} "
                f"k3={train_metrics.get('k3/k3_mean', float('nan')):.5f} "
                f"loss={train_metrics.get('loss/loss', float('nan')):.5f} "
                f"lr={train_metrics.get('lr', 0.0):.2e} "
                f"routed={train_metrics.get('r3_routed_present_rate', 0.0):.2f} "
                f"dt={train_metrics['step_time_s']:.1f}s "
                f"cum_gen_tok={total_gen_tokens/1e6:.2f}M",
                flush=True,
            )
            if log_path:
                _jsonl(log_path, {"event": "grpo_train_step", "step": step, "time": time.time(),
                                  **{k: v for k, v in train_metrics.items() if _is_number(v)}})
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in train_metrics.items() if _is_number(v)}, step=step)
                if args.wandb_log_samples and trajectories:
                    import wandb as _wb  # noqa: PLC0415
                    rows = [[t.target, "|".join(f"{g}->{f}" for g, f in t.history),
                             float(t.solved), round(t.score.get(args.reward_key, 0.0), 3), round(t.advantage, 3),
                             (t.samples[-1].text[:600] if t.samples else "")]
                            for t in trajectories[:args.wandb_log_samples]]
                    tbl = _wb.Table(columns=["target", "history", "solved", "reward", "advantage", "last_text"],
                                    data=rows)
                    wandb_run.log({"train/samples": tbl}, step=step)

            if args.save_interval and step % args.save_interval == 0:
                try:
                    model.save_weights(f"step-{step:06d}", mode="training", timeout=args.op_timeout)
                    print(f"[save step={step}] checkpoint saved", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[save step={step}] FAILED: {e!r}", flush=True)

        try:
            ckpt = model.save_weights("final", mode="training", timeout=args.op_timeout)
            print(f"[done] final checkpoint: {ckpt}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[done] final save FAILED: {e!r}", flush=True)

    print(f"[done] steps complete. cum tokens: prompt={total_prompt_tokens/1e6:.2f}M "
          f"gen={total_gen_tokens/1e6:.2f}M train={total_train_tokens/1e6:.2f}M "
          f"wall={(time.time()-t_start)/60:.1f}min", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
