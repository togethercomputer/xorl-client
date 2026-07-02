"""Native XORL GRPO baseline for the latest Wordle environment.

This is a gradient-based RL baseline, not OPSD:

1. export the current native XORL LoRA to the SGLang sampler pool,
2. sample grouped multi-turn Wordle trajectories from the student policy,
3. score each trajectory with the environment reward,
4. compute GRPO advantages across rollouts of the same target,
5. train the native LoRA with XORL's ``importance_sampling`` loss.

The training rows are the assistant turns the student actually generated.
Advantages are computed at the trajectory level and then applied to that
trajectory's generated turns so long games do not bias the group baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_module
import random
import sys
import threading
import time
from collections.abc import Callable
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
from transformers import AutoConfig, AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for candidate in (
    os.environ.get("XORL_CLIENT_ROOT", ""),
    "/workspace/home/xorl-client",
    "/home/apanda/xorl-client",
):
    if candidate:
        path = Path(candidate)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

import xorl_client  # noqa: E402
try:
    from xorl_client.rl.advantages import compute_grpo_advantages  # noqa: E402
    from xorl_client.rl.datums import build_policy_loss_inputs  # noqa: E402
except ImportError:
    # The installed xorl_client lacks the `rl` submodule. Inline the port here
    # (the standalone grpo_rl_shim module isn't reliably importable in-pod), built
    # against the engine's importance_sampling contract: per-token target_tokens
    # (next-token aligned, IGNORE_INDEX=-100 on prompt), logprobs, advantages.
    import math as _math  # noqa: E402
    from collections import defaultdict as _defaultdict  # noqa: E402

    def compute_grpo_advantages(rewards, *, group_ids, normalize=True,
                                std_normalization=True, eps=1e-8):
        rewards = [float(r) for r in rewards]
        if len(rewards) != len(group_ids):
            raise ValueError(f"rewards/group_ids mismatch: {len(rewards)} vs {len(group_ids)}")
        by_group = _defaultdict(list)
        for r, g in zip(rewards, group_ids):
            by_group[g].append(r)
        stats = {}
        for g, rs in by_group.items():
            mean = sum(rs) / len(rs)
            if std_normalization and len(rs) > 1:
                std = _math.sqrt(sum((x - mean) ** 2 for x in rs) / len(rs))
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

    def build_policy_loss_inputs(tokens, prompt_len, old_logprobs, advantage,
                                 *, ignore_index=-100):
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

from experiments.wordle.standalone.tasks.base import Example, load_task  # noqa: E402
from experiments.wordle.standalone.train_opsd_baseline import (  # noqa: E402
    _env_float,
    _env_int,
    _generated_output_ids,
    _is_number,
    _is_sglang_missing_lora_error,
    _jsonl,
    _post_sglang,
    _split_urls,
    _truncate_output_ids_after_first_guess,
    _wordle_is_valid_guess,
    _wordle_turn_prompt_ids,
    create_model,
    export_and_load_sampler,
    format_sample_eval_summary,
    register_inference_endpoints,
    sample_eval_student,
    save_weights,
    select_train_examples_for_step,
    sync_weights_to_samplers,
    wait_for_training_service,
)


# Routed-expert-logits replay (R3) layout. The sglang captured host buffer is
# [num_tokens, num_hidden_layers, num_experts_per_tok], so the flat decoded array
# reshapes to [T, L, K] with K=num_experts_per_tok, L=num_hidden_layers. We resolve
# L/K once from AutoConfig at startup (see _resolve_routing_shape) and stash them here
# so the rollout extraction code can reshape the base64 buffers without re-reading config.
_ROUTING_SHAPE: dict[str, int | None] = {"L": None, "K": 8}


def _resolve_routing_shape(model: str, *, local_files_only: bool = False) -> dict[str, int | None]:
    """Read num_experts_per_tok (K) and num_hidden_layers (L) from the model config.

    Tries top-level config, then a nested text config (cfg.get_text_config() / cfg.text_config)
    for multimodal wrappers. Defaults K=8 (the MoE-per-tok default for these Qwen3 MoE models)
    and leaves L=None when it can't be found (the decoder then declines to reshape rather than
    guess). Never raises — any problem returns the conservative defaults.
    """
    K: int | None = 8
    L: int | None = None
    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
        candidates = [cfg]
        getter = getattr(cfg, "get_text_config", None)
        if callable(getter):
            try:
                candidates.append(getter())
            except Exception:
                pass
        nested = getattr(cfg, "text_config", None)
        if nested is not None:
            candidates.append(nested)
        for cand in candidates:
            if cand is None:
                continue
            k = getattr(cand, "num_experts_per_tok", None)
            if k is not None:
                try:
                    K = int(k)
                except Exception:
                    pass
            ell = getattr(cand, "num_hidden_layers", None)
            if ell is not None:
                try:
                    L = int(ell)
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: _resolve_routing_shape({model}) failed ({type(exc).__name__}: {exc}); using K={K}, L={L}", flush=True)
    return {"L": L, "K": K}


def _decode_routing(b64, dtype, L: int | None, K: int | None):
    """meta_info base64 buffer -> numpy array shaped [T, L, K], or None on any problem.

    The sglang sampler returns expert routing as a base64-encoded flat buffer (int32 for
    routed_experts, float32 for expert_logits). We only reshape when BOTH L and K are known
    and the decoded size divides evenly into (-1, L, K); otherwise we return None rather than
    guessing a layout. Any decode/reshape failure also returns None (R3 just skips replay).
    """
    if not b64 or not isinstance(b64, str):
        return None
    if not L or not K:
        return None
    try:
        arr = np.frombuffer(pybase64.b64decode(b64.encode("utf-8")), dtype=dtype)
        if arr.size == 0 or arr.size % (L * K) != 0:
            return None
        return arr.reshape(-1, L, K)
    except Exception:  # noqa: BLE001
        return None


def _routing_payload_from_array(arr: np.ndarray | None, rows: int) -> dict[str, Any] | None:
    """Compact R3 payload for xorl's server-side routing decoder."""
    if arr is None or rows <= 0 or arr.shape[0] < rows:
        return None
    clipped = np.ascontiguousarray(arr[:rows])
    return {
        "data": pybase64.b64encode(clipped.tobytes()).decode("ascii"),
        "shape": [int(dim) for dim in clipped.shape],
    }


def _routing_payload_rows(payload: Any) -> int:
    if isinstance(payload, dict):
        shape = payload.get("shape")
        if isinstance(shape, list) and shape:
            try:
                return int(shape[0])
            except Exception:  # noqa: BLE001
                return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


@dataclass
class TurnSample:
    project: str
    target: str
    rollout_id: int
    turn: int
    history_before: list[tuple[str, str]]
    prompt_ids: list[int]
    output_ids: list[int]
    old_logprobs: list[float]
    text: str
    raw_text: str
    guess: str | None
    single_guess_tag: bool
    format_ok: bool
    valid_guess: bool
    feedback: str
    solved: bool
    truncated_after_first_guess: bool
    error: str = ""
    # MoE train/inference parity: SGLang's per-output-token expert routing, fed to the
    # trainer forward so xorl's recomputed logprobs match the sampler's (ratio->1, low k3).
    routed_experts: Any | None = None
    # R3: the routed-expert GATE LOGITS the sampler used (base64 float32 -> [T, L, K]),
    # replayed alongside routed_experts so the trainer forward reproduces the gate weights.
    routed_expert_logits: Any | None = None

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
    # POPE curriculum: the prompt style used for THIS trajectory's turns (may be the
    # faded candidate-scaffold style on a fraction of rollouts; defaults to the run style).
    prompt_style: str = ""


def _sglang_output_logprobs(result: dict[str, Any]) -> list[float]:
    meta = result.get("meta_info") or {}
    raw_logprobs = meta.get("output_token_logprobs")
    if not isinstance(raw_logprobs, list):
        return []
    out: list[float] = []
    for item in raw_logprobs:
        if isinstance(item, (list, tuple)) and item:
            value = item[0]
        else:
            value = item
        out.append(float(value) if value is not None else 0.0)
    return out


def _generated_text_with_stop(result: dict[str, Any], tokenizer) -> str:
    text = str(result.get("text") or "")
    finish_reason = (result.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish_reason, dict) and finish_reason.get("type") == "stop":
        matched = str(finish_reason.get("matched") or "")
        if matched and not text.endswith(matched):
            text = text + matched
    if text:
        return text
    output_ids = _generated_output_ids(result, tokenizer)
    return tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""


# Per-rollout sampling seeds (see --per-rollout-seed). Under deterministic/batch-invariant
# samplers a single shared sampling_seed makes a group's rollouts IDENTICAL (zero within-group
# reward variance -> GRPO adv=0 -> NaN). A unique seed per generated row restores diversity while
# keeping the deterministic (low-k3) forward: each row draws differently from the temp distribution.
_SAMPLING_SEED_COUNTER = None


def _init_sampling_seed_counter(base_seed: int) -> None:
    global _SAMPLING_SEED_COUNTER
    _SAMPLING_SEED_COUNTER = itertools.count(base_seed * 1_000_003 + 1)


def _generate_batch_with_logprobs(
    infer_url: str,
    *,
    input_ids_batch: list[list[int]],
    lora_name: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    return_routed_experts: bool,
    return_expert_logits: bool = False,
    per_rollout_seed: bool = False,
    reload_lora: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    sampling: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }
    if stop:
        sampling["stop"] = list(stop)
    # Per-rollout distinct seeds (for deterministic/batch-invariant samplers): a shared
    # sampling_params makes a group's rollouts identical -> adv=0 -> NaN. A unique seed per row
    # lets each draw differently from the temperature distribution (diverse, still reproducible).
    if per_rollout_seed and _SAMPLING_SEED_COUNTER is not None:
        sampling_params_payload: Any = [
            {**sampling, "sampling_seed": next(_SAMPLING_SEED_COUNTER)} for _ in input_ids_batch
        ]
    else:
        sampling_params_payload = sampling
    payload: dict[str, Any] = {
        "input_ids": [list(input_ids) for input_ids in input_ids_batch],
        "sampling_params": sampling_params_payload,
        "return_logprob": True,
    }
    if lora_name:
        # Full-weight policy reaches the sampler via weight-sync; no adapter to select.
        payload["lora_path"] = lora_name
    if return_routed_experts:
        payload["return_routed_experts"] = True
    if return_expert_logits:
        payload["return_expert_logits"] = True
    attempts = _env_int("GRPO_SGLANG_GENERATE_RETRY_ATTEMPTS", _env_int("OPSD_SGLANG_GENERATE_RETRY_ATTEMPTS", 12))
    retry_interval = _env_float("GRPO_SGLANG_RETRY_INTERVAL", _env_float("OPSD_SGLANG_RETRY_INTERVAL", 5.0))
    try:
        result = _post_sglang(
            infer_url,
            "/generate",
            payload,
            timeout=900.0,
            attempts=attempts,
            retry_interval=retry_interval,
        )
    except Exception as exc:
        if reload_lora is None or not _is_sglang_missing_lora_error(exc):
            raise
        print(
            f"WARN: SGLang sampler {infer_url} lost LoRA {lora_name}; reloading current policy and retrying",
            flush=True,
        )
        reload_lora()
        result = _post_sglang(
            infer_url,
            "/generate",
            payload,
            timeout=900.0,
            attempts=attempts,
            retry_interval=retry_interval,
        )
    if isinstance(result, list):
        if len(result) != len(input_ids_batch):
            raise RuntimeError(f"SGLang returned {len(result)} outputs for {len(input_ids_batch)} prompts")
        return [dict(item or {}) for item in result]
    if len(input_ids_batch) != 1:
        raise RuntimeError(f"SGLang returned a single output for {len(input_ids_batch)} prompts: {result}")
    return [dict(result or {})]


def _invalid_reason(task, text: str, guess: str | None, history: list[tuple[str, str]]) -> str:
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


def _score_trajectory(task, trajectory: Trajectory) -> dict[str, float]:
    max_turns = int(getattr(task, "MAX_TURNS", 6))
    turns_used = len(trajectory.samples)
    format_hits = sum(1 for sample in trajectory.samples if sample.format_ok)
    single_tag_hits = sum(1 for sample in trajectory.samples if sample.single_guess_tag)
    valid_hits = sum(1 for sample in trajectory.samples if sample.valid_guess)
    info_scores = [
        float(task._info_score(sample.guess, trajectory.target, sample.feedback))
        for sample in trajectory.samples
        if sample.valid_guess and sample.guess is not None and sample.feedback
    ]
    latest_feedback = ""
    for sample in reversed(trajectory.samples):
        if sample.feedback:
            latest_feedback = sample.feedback
            break

    format_rate = format_hits / max(turns_used, 1)
    single_guess_tag_rate = single_tag_hits / max(turns_used, 1)
    valid_guess_rate = valid_hits / max(turns_used, 1)
    info_gain = sum(info_scores) / max(len(info_scores), 1)
    terminal_valid = bool(
        trajectory.solved
        or (
            trajectory.stopped_reason == "max_turns"
            and turns_used > 0
            and valid_hits == turns_used
        )
    )
    invalid_action = bool(
        not terminal_valid
        or any(not sample.valid_guess for sample in trajectory.samples)
    )
    if hasattr(task, "_wordle_shaped_reward"):
        reward = task._wordle_shaped_reward(
            solved=trajectory.solved,
            format_rate=format_rate,
            valid_guess_rate=valid_guess_rate,
            info_gain=info_gain,
            turns_used=turns_used,
            max_turns=max_turns,
            invalid_action=invalid_action,
        )
    else:
        turn_bonus = ((max_turns - turns_used + 1) / max_turns) if trajectory.solved else 0.0
        reward = 0.4 * float(trajectory.solved) + 0.3 * format_rate + 0.2 * info_gain + 0.1 * turn_bonus

    wordle_components = {}
    if hasattr(task, "_wordle_reward_components"):
        wordle_components = task._wordle_reward_components(
            solved=trajectory.solved,
            turns_with_guess=valid_hits,
            latest_feedback=latest_feedback,
            # Graded (was binary "all turns single-tagged or 0") so the model has a climbable
            # format gradient instead of a cliff; pairs with lenient extraction so it can play+learn.
            format_reward=single_guess_tag_rate,
            valid_guess_rate=valid_guess_rate,
            terminal_valid=terminal_valid,
            invalid_action=invalid_action,
        )

    # Retrieval-targeted reward (env audit): dominant solve term + per-turn
    # consistency (guess in remaining_candidates) + candidate-set narrowing.
    # Directly penalizes the constraint-violating / invalid guesses that are the
    # dominant SFT failure mode. Select with --reward-key wordle_retrieval_reward.
    retrieval_components: dict[str, float] = {}
    if hasattr(task, "wordle_retrieval_reward"):
        played = [
            (str(sample.guess), str(sample.feedback))
            for sample in trajectory.samples
            if sample.guess and sample.feedback
        ]
        retrieval_components = task.wordle_retrieval_reward(
            played,
            solved=trajectory.solved,
            invalid_action=invalid_action,
            valid_guess_rate=valid_guess_rate,   # -> partial-credit invalid penalty (per-turn rate)
            format_rate=single_guess_tag_rate,   # -> dense graded format reward
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
        **{key: float(value) for key, value in wordle_components.items()},
        **{key: float(value) for key, value in retrieval_components.items()},
    }


def rollout_grpo_trajectories(
    *,
    tokenizer,
    task,
    examples: list[Example],
    group_size: int,
    infer_urls: list[str],
    lora_name: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    ignore_eos: bool,
    stop: list[str] | None,
    generation_batch_size: int,
    max_length: int,
    wordle_prompt_style: str,
    return_routed_experts: bool,
    return_expert_logits: bool = False,
    per_rollout_seed: bool = False,
    step: int,
    log_path: Path,
    reload_lora: Callable[[], None] | None = None,
    scaffold_style: str = "",
    scaffold_fade_steps: int = 0,
    generation_workers: int = 0,
) -> list[Trajectory]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not infer_urls:
        raise ValueError("rollout_grpo_trajectories requires at least one infer URL")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")

    # POPE curriculum: a fraction of this step's rollouts get the candidate-scaffold prompt
    # (faded linearly to 0 by `scaffold_fade_steps`). The reward is UNCHANGED — it verifies the
    # actual guess regardless of whether the prompt showed candidates — so the scaffold is a faded
    # RL *input* aid, not a distillation target. fade_steps<=0 (default) => never scaffold (the
    # original behaviour: every trajectory uses `wordle_prompt_style`).
    scaffold_fraction = 0.0
    if scaffold_style and scaffold_fade_steps > 0:
        scaffold_fraction = max(0.0, 1.0 - float(step - 1) / float(scaffold_fade_steps))
    style_rng = random.Random(1_000_003 * step + 17)

    max_turns = int(getattr(task, "MAX_TURNS", 6))
    trajectories = [
        Trajectory(
            project=example.project,
            target=str(example.metadata["target"]).lower(),
            group_id=example.project,
            rollout_id=rollout_idx,
            prompt_style=(
                scaffold_style
                if (scaffold_fraction > 0.0 and style_rng.random() < scaffold_fraction)
                else wordle_prompt_style
            ),
        )
        for example in examples
        for rollout_idx in range(group_size)
    ]
    n_scaffold = sum(1 for t in trajectories if t.prompt_style == scaffold_style and scaffold_style)
    if scaffold_style and scaffold_fade_steps > 0:
        print(
            f"[POPE] step {step}: scaffold_fraction={scaffold_fraction:.3f} "
            f"-> {n_scaffold}/{len(trajectories)} rollouts use '{scaffold_style}'",
            flush=True,
        )

    batch_counter = 0
    for turn in range(1, max_turns + 1):
        active = [traj for traj in trajectories if not traj.solved and not traj.stopped_reason]
        if not active:
            break

        prompts = [
            list(
                _wordle_turn_prompt_ids(
                    tokenizer,
                    task,
                    target=traj.target,
                    history=traj.history,
                    hinted=False,
                    prompt_style=traj.prompt_style,
                    teacher_prompt_style="no_hint",
                )
            )
            for traj in active
        ]
        # Parallel sampling: dispatch this turn's batches CONCURRENTLY across the samplers
        # (round-robin over infer_urls; a single SMG URL load-balances the concurrent requests).
        # The blocking _generate_batch_with_logprobs calls run in a thread pool so every sampler
        # stays busy instead of one batch at a time; per-traj result processing below stays serial
        # (each batch only mutates its own trajectories). Mirrors the OPSD baseline generation_workers.
        batch_specs = []
        for start in range(0, len(active), generation_batch_size):
            infer_url = infer_urls[batch_counter % len(infer_urls)]
            batch_counter += 1
            batch_specs.append(
                (
                    active[start : start + generation_batch_size],
                    prompts[start : start + generation_batch_size],
                    infer_url,
                )
            )

        def _gen_batch(spec):
            _bt, _bp, _url = spec
            try:
                return spec, _generate_batch_with_logprobs(
                    _url,
                    input_ids_batch=_bp,
                    lora_name=lora_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=ignore_eos,
                    stop=stop,
                    return_routed_experts=return_routed_experts,
                    return_expert_logits=return_expert_logits,
                    per_rollout_seed=per_rollout_seed,
                    reload_lora=reload_lora,
                ), None
            except Exception as exc:  # noqa: BLE001
                return spec, None, exc

        _nw = max(1, min(len(batch_specs), generation_workers or (len(infer_urls) * generation_batch_size)))
        if _nw <= 1 or len(batch_specs) <= 1:
            gen_results = [_gen_batch(s) for s in batch_specs]
        else:
            with ThreadPoolExecutor(max_workers=_nw) as _ex:
                gen_results = list(_ex.map(_gen_batch, batch_specs))

        for (batch_trajectories, batch_prompts, infer_url), results, exc in gen_results:
            if exc is not None:
                for traj, prompt_ids in zip(batch_trajectories, batch_prompts, strict=True):
                    traj.stopped_reason = "sampling_error"
                    sample = TurnSample(
                        project=traj.project,
                        target=traj.target,
                        rollout_id=traj.rollout_id,
                        turn=turn,
                        history_before=list(traj.history),
                        prompt_ids=list(prompt_ids),
                        output_ids=[],
                        old_logprobs=[],
                        text="",
                        raw_text="",
                        guess=None,
                        single_guess_tag=False,
                        format_ok=False,
                        valid_guess=False,
                        feedback="",
                        solved=False,
                        truncated_after_first_guess=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    traj.samples.append(sample)
                continue

            for traj, prompt_ids, result in zip(batch_trajectories, batch_prompts, results, strict=True):
                raw_output_ids = _generated_output_ids(result, tokenizer)
                raw_logprobs = _sglang_output_logprobs(result)
                _meta = result.get("meta_info") or {}
                raw_routed_experts = _meta.get("routed_experts")
                raw_expert_logits = _meta.get("expert_logits")
                # sglang returns these as base64 buffers (int32 experts, float32 logits),
                # decoded+reshaped to [T, L, K]; None on any decode/shape problem (R3 skips replay).
                experts_arr = _decode_routing(raw_routed_experts, np.int32, _ROUTING_SHAPE["L"], _ROUTING_SHAPE["K"])
                logits_arr = _decode_routing(raw_expert_logits, np.float32, _ROUTING_SHAPE["L"], _ROUTING_SHAPE["K"])
                raw_text = _generated_text_with_stop(result, tokenizer)
                output_ids, text, truncated_after_first_guess = _truncate_output_ids_after_first_guess(
                    tokenizer,
                    raw_output_ids,
                    think=traj.prompt_style.endswith("_think"),
                )
                if not text:
                    text = raw_text
                if len(output_ids) > max(0, max_length - len(prompt_ids)):
                    output_ids = output_ids[: max(0, max_length - len(prompt_ids))]
                    text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    truncated_after_first_guess = True
                if len(raw_logprobs) < len(output_ids):
                    error = f"logprob_count_mismatch: output_ids={len(output_ids)} logprobs={len(raw_logprobs)}"
                    output_ids = []
                    old_logprobs: list[float] = []
                    routed_experts = None
                    routed_expert_logits = None
                else:
                    error = ""
                    old_logprobs = raw_logprobs[: len(output_ids)]
                    # SGLang captures routing for the actual forward positions used during
                    # generation: prompt prefill + decode prefixes. For a kept sequence
                    # prompt_ids + output_ids, that is len(prompt_ids) + len(output_ids) - 1
                    # rows. The xorl-client datum is already next-token shifted and sends
                    # model_input=token_list[:-1], so R3 must align to that shifted length.
                    # Expanding to the full token length inserts one extra row per datum; in
                    # packed batches the replay handler concatenates per-datum routing before
                    # trimming the micro-batch, shifting every later datum.
                    _model_input_len = max(0, len(prompt_ids) + len(output_ids) - 1)
                    routed_experts = _routing_payload_from_array(experts_arr, _model_input_len)
                    routed_expert_logits = _routing_payload_from_array(logits_arr, _model_input_len)

                # Judge the action on the think-stripped, first-guess-truncated action
                # line (think output mentions the tags + emits trailing chatter); without
                # this every think-contract game dies at turn 1. Mirrors the OPSD rollout.
                action_text = task.extract_action_text(raw_text or text or "") if hasattr(task, "extract_action_text") else (text or "")
                # Lenient extraction: take the LAST 5-letter <guess> (the real answer follows any
                # CoT, echoed template, or candidate list). PLAYABILITY is decoupled from format —
                # a legal guess lets the game continue; format is shaped by the graded format reward,
                # not by killing the turn. (Previously valid_guess required exactly-one-tag, so the
                # thinking model's template echo killed ~91% of turns and it never got to play/learn.)
                _guesses = task.extract_guesses(action_text) if hasattr(task, "extract_guesses") else []
                guess = _guesses[-1] if _guesses else task.extract_guess(action_text)
                single_guess_tag = bool(task.has_single_guess_tag(action_text)) if hasattr(task, "has_single_guess_tag") else bool(guess)
                format_ok = bool(single_guess_tag and guess is not None)
                valid_guess = bool(guess is not None and _wordle_is_valid_guess(task, guess, traj.history))
                feedback = ""
                solved = False
                if valid_guess:
                    assert guess is not None
                    feedback = task.compute_feedback(guess, traj.target)
                    traj.history.append((guess, feedback))
                    solved = guess == traj.target
                    traj.solved = solved
                    if solved:
                        traj.stopped_reason = "solved"
                    elif len(traj.history) >= max_turns:
                        traj.stopped_reason = "max_turns"
                else:
                    traj.stopped_reason = _invalid_reason(task, text or raw_text, guess, traj.history)

                sample = TurnSample(
                    project=traj.project,
                    target=traj.target,
                    rollout_id=traj.rollout_id,
                    turn=turn,
                    history_before=list(traj.history[:-1] if valid_guess else traj.history),
                    prompt_ids=list(prompt_ids),
                    output_ids=list(output_ids),
                    old_logprobs=list(old_logprobs),
                    text=text,
                    raw_text=raw_text,
                    guess=guess,
                    single_guess_tag=single_guess_tag,
                    format_ok=format_ok,
                    valid_guess=valid_guess,
                    feedback=feedback,
                    solved=solved,
                    truncated_after_first_guess=truncated_after_first_guess,
                    error=error,
                    routed_experts=routed_experts,
                    routed_expert_logits=routed_expert_logits,
                )
                traj.samples.append(sample)

    for traj in trajectories:
        if not traj.stopped_reason:
            traj.stopped_reason = "max_turns"
        traj.score = _score_trajectory(task, traj)
        for sample in traj.samples:
            _jsonl(
                log_path,
                {
                    "event": "grpo_student_turn",
                    "step": step,
                    "time": time.time(),
                    "project": sample.project,
                    "target": sample.target,
                    "rollout_id": sample.rollout_id,
                    "turn": sample.turn,
                    "history_before": sample.history_before,
                    "guess": sample.guess or "",
                    "single_guess_tag": sample.single_guess_tag,
                    "format_ok": sample.format_ok,
                    "valid_guess": sample.valid_guess,
                    "feedback": sample.feedback,
                    "solved": sample.solved,
                    "stopped_reason": traj.stopped_reason,
                    "reward": traj.score.get("reward", 0.0),
                    "wordle_reward": traj.score.get("wordle_reward", traj.score.get("reward", 0.0)),
                    "output_tokens": len(sample.output_ids),
                    "prompt_tokens": len(sample.prompt_ids),
                    "truncated_after_first_guess": sample.truncated_after_first_guess,
                    "error": sample.error,
                    "text": sample.text,
                    "raw_text": sample.raw_text if sample.truncated_after_first_guess else "",
                },
            )
    return trajectories


def build_grpo_actor_datums(
    trajectories: list[Trajectory],
    *,
    reward_key: str,
    normalize: bool,
    std_normalization: bool,
    normalize_advantage_by_turns: bool,
    skip_zero_advantage: bool,
) -> tuple[list[xorl_client.types.Datum], dict[str, float]]:
    if not trajectories:
        return [], {"trajectory_count": 0.0}

    rewards = [float(traj.score.get(reward_key, traj.score.get("reward", 0.0))) for traj in trajectories]
    group_ids = [traj.group_id for traj in trajectories]
    advantages = compute_grpo_advantages(
        rewards,
        group_ids=group_ids,
        normalize=normalize,
        std_normalization=std_normalization,
    )
    datums: list[xorl_client.types.Datum] = []
    skipped_zero = 0
    skipped_empty = 0
    routed_datum_count = 0
    routed_aligned_count = 0
    routed_rows_sum = 0
    for traj, advantage in zip(trajectories, advantages, strict=True):
        traj.advantage = float(advantage)
        if normalize_advantage_by_turns:
            turn_count = max(1, sum(1 for sample in traj.samples if sample.output_ids and sample.old_logprobs))
            sample_advantage = float(advantage) / float(turn_count)
        else:
            sample_advantage = float(advantage)
        if skip_zero_advantage and abs(sample_advantage) <= 1e-12:
            skipped_zero += len(traj.samples)
            continue
        for sample in traj.samples:
            if not sample.output_ids or not sample.old_logprobs:
                skipped_empty += 1
                continue
            token_list = sample.tokens
            if len(token_list) <= sample.prompt_len:
                skipped_empty += 1
                continue
            model_input_tokens = token_list[:-1]
            routed_rows = _routing_payload_rows(sample.routed_experts)
            if sample.routed_experts is not None:
                routed_datum_count += 1
                routed_rows_sum += routed_rows
                if routed_rows == len(model_input_tokens):
                    routed_aligned_count += 1
            loss_inputs = build_policy_loss_inputs(
                token_list,
                sample.prompt_len,
                sample.old_logprobs,
                sample_advantage,
            )
            datums.append(
                xorl_client.types.Datum(
                    model_input=xorl_client.types.ModelInput.from_ints(model_input_tokens),
                    loss_fn_inputs=loss_inputs,
                    routed_experts=sample.routed_experts,
                    routed_expert_logits=sample.routed_expert_logits,
                )
            )

    n = max(float(len(trajectories)), 1.0)
    exact_values = [float(traj.score.get("exact_match", 0.0)) for traj in trajectories]
    metrics = {
        "trajectory_count": float(len(trajectories)),
        "datum_count": float(len(datums)),
        "reward_mean": sum(rewards) / n,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "advantage_mean": sum(float(a) for a in advantages) / n,
        "advantage_abs_mean": sum(abs(float(a)) for a in advantages) / n,
        "exact_match_rate": sum(exact_values) / n,
        "exact_count": sum(exact_values),
        "skipped_zero_advantage_turns": float(skipped_zero),
        "skipped_empty_turns": float(skipped_empty),
        "r3_routed_datum_count": float(routed_datum_count),
        "r3_routed_aligned_count": float(routed_aligned_count),
        "r3_routing_missing_count": float(max(len(datums) - routed_datum_count, 0)),
        "r3_routed_rows_mean": float(routed_rows_sum) / max(float(routed_datum_count), 1.0),
        "r3_routed_present_rate": float(routed_datum_count) / max(float(len(datums)), 1.0),
        "r3_routed_aligned_rate": float(routed_aligned_count) / max(float(len(datums)), 1.0),
    }
    numeric_score_keys = {
        key
        for traj in trajectories
        for key, value in traj.score.items()
        if _is_number(value)
    }
    for key in sorted(numeric_score_keys):
        values = [float(traj.score.get(key, 0.0)) for traj in trajectories]
        metrics[f"{key}_mean"] = sum(values) / n
    return datums, metrics


def _metrics_from_output(output) -> dict[str, float]:
    metrics = {}
    for key, value in (getattr(output, "metrics", {}) or {}).items():
        if _is_number(value):
            metrics[str(key)] = float(value)
    return metrics


def _metrics_from_optim(output) -> dict[str, float]:
    metrics = {}
    for key, value in (getattr(output, "metrics", {}) or {}).items():
        if _is_number(value):
            metrics[str(key)] = float(value)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-url", default="http://127.0.0.1:26040")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--task", default="wordle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-model-id", default="grpo-wordle-native")
    parser.add_argument("--infer-url", action="append", default=[])
    parser.add_argument("--sampler-load-url", action="append", default=[])
    parser.add_argument("--server-output-dir", default="")
    parser.add_argument("--sampler-lora-name", default="")
    parser.add_argument("--sampler-save-prefix", default="")
    parser.add_argument("--seed", type=int, default=9234)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--train-pool-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument(
        "--resume-from-step",
        type=int,
        default=0,
        help="Resume a crashed run: start the step loop at this step+1 (the server must load the "
        "matching DCP checkpoint via load_checkpoint_path). LR schedule + per-step data are step-keyed, "
        "so resuming the counter is an exact resume. 0 (default) = fresh start, no behavior change.",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--resample-train-each-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-key", default="wordle_reward")
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--std-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-advantage-by-turns", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-zero-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-top-p", type=float, default=0.95)
    parser.add_argument("--student-max-new-tokens", type=int, default=96)
    parser.add_argument("--student-generation-batch-size", type=int, default=8)
    parser.add_argument(
        "--student-generation-workers",
        type=int,
        default=0,
        help="Concurrent in-flight generation requests across the samplers per turn (0 = "
        "len(infer_urls)*generation_batch_size). Set > number-of-batches to fully overlap all samplers.",
    )
    parser.add_argument("--student-ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-stop", action="append", default=[])
    parser.add_argument("--return-routed-experts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return-expert-logits", action=argparse.BooleanOptionalAction, default=False)
    # Distinct sampling_seed per rollout: required with deterministic/batch-invariant samplers so a
    # group's rollouts diverge (else adv=0 -> NaN). Harmless/off for stochastic flashinfer samplers.
    parser.add_argument("--per-rollout-seed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--wordle-prompt-style",
        default="public_reasoning_constraints",
        choices=[
            "default",
            "candidate_list",
            "public_reasoning",
            "public_reasoning_strict",
            "public_reasoning_strict_nocandidates",
            "public_reasoning_constraints",
            "public_reasoning_constraints_candidates",
            "public_reasoning_constraints_think",
            "public_reasoning_constraints_candidates_think",
        ],
    )
    # POPE curriculum: fade a candidate-scaffold prompt out of the rollouts over the first N steps.
    # The scaffold is an RL INPUT aid (reward is unchanged); default off (fade-steps=0).
    parser.add_argument(
        "--wordle-scaffold-style",
        default="public_reasoning_constraints_candidates_think",
        help="POPE: prompt style used for the scaffolded fraction of rollouts (shows the candidate list).",
    )
    parser.add_argument(
        "--wordle-scaffold-fade-steps",
        type=int,
        default=0,
        help="POPE: linearly fade the scaffolded-rollout fraction from 1.0 at step 1 to 0.0 at this "
        "step. 0 (default) disables POPE entirely (every rollout uses --wordle-prompt-style).",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument(
        "--full-weight",
        action="store_true",
        help="Full-weight training (no LoRA): policy reaches samplers via sync_inference_weights "
        "(register endpoints + full-weight NCCL/p2p broadcast). Matches the q36 full-weight sampler pool.",
    )
    parser.add_argument("--sampler-world-size", type=int, default=2, help="GPUs per sampler shard (TP size).")
    parser.add_argument("--weight-sync-buffer-mb", type=int, default=1024)
    parser.add_argument(
        "--weight-sync-flush-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pipeline-rl",
        action="store_true",
        help="Pipeline RL: a background thread generates the next step's rollouts on the current "
        "sampler weights while the trainer runs forward/backward+optim on this step, overlapping the "
        "two phases (~halves wall-clock). Consumed batches are one optim step stale (off-policy by one "
        "step); requires --full-weight and is intended to be paired with --loss-fn policy_loss.",
    )
    parser.add_argument(
        "--loss-fn",
        default="importance_sampling",
        choices=["importance_sampling", "policy_loss"],
        help="Trainer loss: 'importance_sampling' (unclipped IS policy gradient; on-policy default) or "
        "'policy_loss' (PPO-clipped policy gradient; bounds the importance ratio, the robust choice for "
        "pipelined/off-policy data).",
    )
    parser.add_argument("--eps-clip", type=float, default=0.2, help="policy_loss: lower PPO clip ratio.")
    parser.add_argument("--eps-clip-high", type=float, default=0.2, help="policy_loss: upper PPO clip ratio.")
    parser.add_argument(
        "--eps-clip-c",
        type=float,
        default=None,
        help="policy_loss: optional dual-clip ratio for negative advantages (must be >1.0 if set).",
    )
    parser.add_argument(
        "--optimizer", default="adamw", choices=["adamw", "anyprecision_adamw", "sgd", "signsgd", "muon"]
    )
    parser.add_argument("--optimizer-dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--lr-schedule", default="constant", choices=["constant", "cosine"],
        help="LR schedule applied CLIENT-SIDE per optim_step (sets all param groups incl. muon). "
        "Server-side cosine/warmup in the config yaml crashes engine init, so schedule here instead.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="cosine: linear warmup 0->lr over N steps.")
    parser.add_argument("--min-lr", type=float, default=0.0, help="cosine: decay floor.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--compute-kl-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-ref-logprobs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--logprob-temperature",
        type=float,
        default=None,
        help=(
            "Temperature used by the trainer when recomputing selected-token logprobs for RL ratios. "
            "Defaults to --student-temperature when positive; pass 1.0 for raw model logprobs."
        ),
    )
    parser.add_argument("--dump-kl-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--diagnostic-microbatch-dump-dir",
        default="",
        help="Optional trainer-side directory for packed microbatch/R3 replay dumps.",
    )
    parser.add_argument(
        "--diagnostic-microbatch-dump-tensors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When dumping microbatch diagnostics, also save torch tensors and R3 payload slices.",
    )
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--future-timeout", type=float, default=7200.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--sample-eval-interval", type=int, default=8)
    parser.add_argument("--sample-eval-size", type=int, default=32)
    parser.add_argument("--sample-eval-temperature", type=float, default=None)
    parser.add_argument("--sample-eval-max-new-tokens", type=int, default=96)
    parser.add_argument("--sample-eval-ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-eval-stop", action="append", default=[])
    parser.add_argument("--sample-eval-workers", type=int, default=32)
    parser.add_argument("--sample-eval-log-examples", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-log-samples", type=int, default=16)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # R3: routed_expert_logits replay only makes sense alongside routed_experts replay
    # (the engine pairs the gate logits with the selected experts), so auto-enable the
    # latter when only --return-expert-logits was passed.
    if args.return_expert_logits and not args.return_routed_experts:
        args.return_routed_experts = True
        print(
            "[init] --return-expert-logits set without --return-routed-experts; "
            "auto-enabling --return-routed-experts (R3 replay needs both)",
            flush=True,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    generations_path = output_dir / "generations.jsonl"
    sample_eval_path = output_dir / "sample_eval.jsonl"

    infer_urls = _split_urls(args.infer_url)
    sampler_load_urls = _split_urls(args.sampler_load_url) or infer_urls
    student_stop = _split_urls(args.student_stop)
    sample_eval_stop = _split_urls(args.sample_eval_stop)
    if args.sample_eval_temperature is None:
        args.sample_eval_temperature = args.student_temperature
    if args.train_pool_size < args.train_size:
        raise ValueError("--train-pool-size must be >= --train-size")
    if not infer_urls:
        raise ValueError("--infer-url is required")
    if not sampler_load_urls:
        raise ValueError("--sampler-load-url or --infer-url is required")
    if args.max_runtime_seconds < 0:
        raise ValueError("--max-runtime-seconds must be non-negative")
    if args.logprob_temperature is not None and args.logprob_temperature <= 0.0:
        raise ValueError("--logprob-temperature must be positive when set")
    trainer_logprob_temperature = (
        float(args.logprob_temperature)
        if args.logprob_temperature is not None
        else (float(args.student_temperature) if args.student_temperature > 0.0 else 1.0)
    )

    server_output_dir = Path(args.server_output_dir) if args.server_output_dir else output_dir / "server_output"
    # Full-weight: no LoRA adapter name — the synced base weights ARE the policy, so
    # generation must omit lora_path (gated on truthiness of sampler_lora_name below).
    sampler_lora_name = "" if args.full_weight else (args.sampler_lora_name or f"{args.train_model_id}-{output_dir.name}")
    sampler_save_prefix = args.sampler_save_prefix or f"{output_dir.name}/student"

    print(f"[init] waiting for training server at {args.train_url}", flush=True)
    wait_for_training_service(args.train_url, timeout=args.startup_timeout)
    print("[init] training server is ready", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    # R3: resolve the routed-expert routing layout once (L=num_hidden_layers,
    # K=num_experts_per_tok) so the rollout extraction can reshape the base64
    # routing buffers to [T, L, K]. Stashed module-level for _decode_routing.
    _ROUTING_SHAPE.update(_resolve_routing_shape(args.model, local_files_only=args.local_files_only))
    print(
        f"[init] routed-expert routing shape: L(num_hidden_layers)={_ROUTING_SHAPE['L']} "
        f"K(num_experts_per_tok)={_ROUTING_SHAPE['K']}",
        flush=True,
    )
    task = load_task(args.task)
    train_pool, eval_examples = task.build_examples(
        tokenizer,
        train_size=args.train_pool_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    service_client = xorl_client.ServiceClient(base_url=args.train_url, timeout=args.future_timeout)
    training_client = xorl_client.TrainingClient(
        holder=service_client.holder,
        model_id=args.train_model_id,
        base_model=args.model,
    )

    run_config = {
        **vars(args),
        "infer_urls": infer_urls,
        "sampler_load_urls": sampler_load_urls,
        "student_stop": student_stop,
        "sample_eval_stop": sample_eval_stop,
        "server_output_dir": str(server_output_dir),
        "sampler_lora_name": sampler_lora_name,
        "sampler_save_prefix": sampler_save_prefix,
        "trainer_logprob_temperature": trainer_logprob_temperature,
        "train_examples": len(train_pool),
        "eval_examples": len(eval_examples),
        "word_list_source": getattr(task, "WORD_LIST_SOURCE", "unknown"),
        "legal_guesses_source": getattr(task, "LEGAL_GUESSES_SOURCE", "unknown"),
        "word_list_size": len(getattr(task, "WORD_LIST", [])),
        "legal_guesses_size": len(getattr(task, "LEGAL_GUESSES", [])),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "train_examples.json").write_text(
        json.dumps([asdict(example) for example in train_pool], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "eval_examples.json").write_text(
        json.dumps([asdict(example) for example in eval_examples], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _jsonl(metrics_path, {"event": "init", "time": time.time(), **run_config})

    if args.wandb_project:
        import wandb  # noqa: PLC0415

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or None, config=run_config)
    else:
        wandb = None
        wandb_run = None

    print(f"[init] creating native LoRA session {args.train_model_id}", flush=True)
    create_result = create_model(args.train_url, args)
    _jsonl(metrics_path, {"event": "create_model", "time": time.time(), "result": create_result})

    loaded_policy_step: int | None = None
    endpoints_registered = False

    def ensure_sampler(policy_step: int, *, force: bool = False) -> None:
        nonlocal loaded_policy_step, endpoints_registered
        if loaded_policy_step == policy_step and not force:
            return
        if args.full_weight:
            if not endpoints_registered:
                register_inference_endpoints(
                    args.train_url,
                    sampler_load_urls,
                    world_size=args.sampler_world_size,
                )
                endpoints_registered = True
            weight_version = f"{sampler_save_prefix}/policy-{policy_step:06d}"
            print(f"[sampler step={policy_step}] full-weight sync to samplers: {weight_version}", flush=True)
            started = time.time()
            sync_weights_to_samplers(
                train_url=args.train_url,
                model_id=args.train_model_id,
                weight_version=weight_version,
                output_dir=output_dir,
                buffer_size_mb=args.weight_sync_buffer_mb,
                flush_cache=args.weight_sync_flush_cache,
                timeout=args.future_timeout,
            )
            print(
                f"[sampler step={policy_step}] full-weight sync done in {time.time() - started:.1f}s "
                f"({len(sampler_load_urls)} endpoint(s))",
                flush=True,
            )
            loaded_policy_step = policy_step
            return
        save_name = f"{sampler_save_prefix}/policy-{policy_step:06d}"
        print(f"[sampler step={policy_step}] exporting native LoRA for SGLang: {save_name}", flush=True)
        sampler_path = export_and_load_sampler(
            train_url=args.train_url,
            model_id=args.train_model_id,
            output_dir=output_dir,
            server_output_dir=server_output_dir,
            infer_urls=sampler_load_urls,
            lora_name=sampler_lora_name,
            save_name=save_name,
            future_timeout=args.future_timeout,
        )
        print(
            f"[sampler step={policy_step}] loaded {sampler_lora_name} from {sampler_path} "
            f"on {len(sampler_load_urls)} load endpoint(s); generate endpoints={len(infer_urls)}",
            flush=True,
        )
        loaded_policy_step = policy_step

    def selected_sample_eval_examples() -> list[Example]:
        if args.sample_eval_size <= 0 or args.sample_eval_size >= len(eval_examples):
            return list(eval_examples)
        return list(eval_examples[: args.sample_eval_size])

    def run_sample_eval(policy_step: int) -> dict[str, float]:
        if args.sample_eval_interval <= 0:
            return {}
        examples = selected_sample_eval_examples()
        if not examples:
            return {}
        ensure_sampler(policy_step)
        started = time.time()
        metrics = sample_eval_student(
            tokenizer=tokenizer,
            task=task,
            examples=examples,
            infer_urls=infer_urls,
            lora_name=sampler_lora_name,
            temperature=float(args.sample_eval_temperature),
            max_new_tokens=args.sample_eval_max_new_tokens,
            ignore_eos=args.sample_eval_ignore_eos,
            stop=sample_eval_stop,
            workers=args.sample_eval_workers,
            step=policy_step,
            log_path=sample_eval_path,
            log_examples=args.sample_eval_log_examples,
            wordle_prompt_style=args.wordle_prompt_style,
            reload_lora=lambda: ensure_sampler(policy_step, force=True),
        )
        metrics["sample_eval_time_s"] = time.time() - started
        print(f"[sample_eval step={policy_step}] {format_sample_eval_summary(metrics)}", flush=True)
        _jsonl(metrics_path, {"event": "sample_eval", "step": policy_step, "time": time.time(), **metrics})
        if wandb_run is not None:
            wandb_run.log({f"sample_eval/{key}": value for key, value in metrics.items()}, step=policy_step)
        return metrics

    def log_wandb_samples(step: int, trajectories: list[Trajectory]) -> None:
        if wandb_run is None or args.wandb_log_samples == 0:
            return
        import wandb as wandb_module  # noqa: PLC0415

        rows: list[tuple[Trajectory, TurnSample]] = []
        for traj in trajectories:
            for sample in traj.samples:
                rows.append((traj, sample))
        if args.wandb_log_samples > 0:
            rows = rows[: args.wandb_log_samples]
        if not rows:
            return
        table = wandb_module.Table(
            columns=[
                "step",
                "project",
                "target",
                "rollout_id",
                "turn",
                "history_before",
                "guess",
                "feedback",
                "valid_guess",
                "stopped_reason",
                "reward",
                "wordle_reward",
                "advantage",
                "text",
                "error",
            ]
        )
        for traj, sample in rows:
            table.add_data(
                step,
                sample.project,
                sample.target,
                sample.rollout_id,
                sample.turn,
                json.dumps(sample.history_before),
                sample.guess or "",
                sample.feedback,
                sample.valid_guess,
                traj.stopped_reason,
                float(traj.score.get("reward", 0.0)),
                float(traj.score.get("wordle_reward", traj.score.get("reward", 0.0))),
                float(traj.advantage),
                sample.text,
                sample.error,
            )
        wandb_run.log({"train/samples": table}, step=step)

    run_sample_eval(0)

    train_loop_started_at = time.time()

    def get_lr(step: int) -> float:
        """Client-side LR schedule (1-indexed step). Passed to optim_step => sets ALL param
        groups incl. muon (model_runner overwrites every group's lr with this value)."""
        if args.lr_schedule != "cosine":
            return args.lr
        import math  # noqa: PLC0415
        w = max(0, args.lr_warmup_steps)
        if w > 0 and step <= w:
            return args.lr * step / w
        denom = max(1, args.steps - w)
        progress = min(1.0, max(0.0, (step - w) / denom))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _hit_time_limit(step: int) -> bool:
        elapsed_s = time.time() - train_loop_started_at
        if args.max_runtime_seconds > 0 and elapsed_s >= args.max_runtime_seconds:
            print(
                f"[time_limit] stopping before step={step}: elapsed={elapsed_s:.1f}s "
                f"limit={args.max_runtime_seconds:.1f}s",
                flush=True,
            )
            _jsonl(
                metrics_path,
                {
                    "event": "time_limit",
                    "step": step,
                    "time": time.time(),
                    "elapsed_s": elapsed_s,
                    "max_runtime_seconds": args.max_runtime_seconds,
                },
            )
            return True
        return False

    if args.per_rollout_seed:
        _init_sampling_seed_counter(int(args.seed))

    def do_rollout(step: int, reload_cb: Callable[[], None]) -> tuple[list[Trajectory], float]:
        selected_examples = select_train_examples_for_step(
            train_pool,
            train_size=args.train_size,
            seed=args.seed,
            step=step - 1,
            resample=args.resample_train_each_step,
        )
        rollout_started = time.time()
        trajectories = rollout_grpo_trajectories(
            tokenizer=tokenizer,
            task=task,
            examples=selected_examples,
            group_size=args.group_size,
            infer_urls=infer_urls,
            lora_name=sampler_lora_name,
            temperature=args.student_temperature,
            top_p=args.student_top_p,
            max_new_tokens=args.student_max_new_tokens,
            ignore_eos=args.student_ignore_eos,
            stop=student_stop,
            generation_batch_size=args.student_generation_batch_size,
            max_length=args.max_length,
            wordle_prompt_style=args.wordle_prompt_style,
            return_routed_experts=args.return_routed_experts,
            return_expert_logits=args.return_expert_logits,
            per_rollout_seed=args.per_rollout_seed,
            step=step,
            log_path=generations_path,
            reload_lora=reload_cb,
            scaffold_style=args.wordle_scaffold_style,
            scaffold_fade_steps=args.wordle_scaffold_fade_steps,
            generation_workers=args.student_generation_workers,
        )
        return trajectories, time.time() - rollout_started

    def do_train(step: int, trajectories: list[Trajectory], rollout_time_s: float) -> None:
        train_started = time.time()
        datums, grpo_metrics = build_grpo_actor_datums(
            trajectories,
            reward_key=args.reward_key,
            normalize=args.normalize_advantages,
            std_normalization=args.std_normalization,
            normalize_advantage_by_turns=args.normalize_advantage_by_turns,
            skip_zero_advantage=args.skip_zero_advantage,
        )
        log_wandb_samples(step, trajectories)

        train_metrics: dict[str, float] = {
            **grpo_metrics,
            "rollout_time_s": float(rollout_time_s),
            "train_target_count": float(args.train_size),
            "group_size": float(args.group_size),
        }
        if datums:
            loss_params = {
                "compute_kl_stats": bool(args.compute_kl_stats),
                "compute_ref_logprobs": bool(args.compute_ref_logprobs),
                "dump_kl_diagnostics": bool(args.dump_kl_diagnostics),
                "logprob_temperature": trainer_logprob_temperature,
            }
            if args.diagnostic_microbatch_dump_dir:
                loss_params["diagnostic_microbatch_dump_dir"] = str(args.diagnostic_microbatch_dump_dir)
                loss_params["diagnostic_microbatch_dump_tensors"] = bool(args.diagnostic_microbatch_dump_tensors)
            if args.loss_fn == "policy_loss":
                loss_params["eps_clip"] = float(args.eps_clip)
                loss_params["eps_clip_high"] = float(args.eps_clip_high)
                if args.eps_clip_c is not None:
                    loss_params["eps_clip_c"] = float(args.eps_clip_c)
            fwd_started = time.time()
            fwd_bwd = training_client.forward_backward(
                datums,
                loss_fn=args.loss_fn,
                loss_fn_params=loss_params,
            ).result()
            train_metrics.update({f"loss/{key}": value for key, value in _metrics_from_output(fwd_bwd).items()})
            train_metrics["forward_backward_time_s"] = time.time() - fwd_started

            opt_started = time.time()
            opt_result = training_client.optim_step(
                xorl_client.types.AdamParams(
                    learning_rate=get_lr(step),
                    beta1=args.beta1,
                    beta2=args.beta2,
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.gradient_clip,
                )
            ).result()
            train_metrics.update({f"optim/{key}": value for key, value in _metrics_from_optim(opt_result).items()})
            train_metrics["optim_time_s"] = time.time() - opt_started
        else:
            train_metrics["skipped_train_step"] = 1.0

        train_metrics["train_time_s"] = time.time() - train_started
        train_metrics["step_time_s"] = float(rollout_time_s) + train_metrics["train_time_s"]
        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[train step={step}] reward={train_metrics.get('reward_mean', float('nan')):.4f} "
                f"exact={train_metrics.get('exact_count', 0.0):.0f}/"
                f"{max(train_metrics.get('trajectory_count', 0.0), 1.0):.0f} "
                f"datums={train_metrics.get('datum_count', 0.0):.0f} "
                f"adv_abs={train_metrics.get('advantage_abs_mean', 0.0):.4f} "
                f"loss={train_metrics.get('loss/loss', train_metrics.get('loss/loss:mean', float('nan'))):.6f} "
                f"dt={train_metrics['step_time_s']:.1f}s",
                flush=True,
            )
        _jsonl(metrics_path, {"event": "train", "step": step, "time": time.time(), **train_metrics})
        if wandb_run is not None:
            wandb_run.log({f"train/{key}": value for key, value in train_metrics.items()}, step=step)

        if args.sample_eval_interval > 0 and step % args.sample_eval_interval == 0:
            run_sample_eval(step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_result = save_weights(
                args.train_url,
                args.train_model_id,
                f"step-{step:06d}",
                future_timeout=args.future_timeout,
            )
            print(f"[save step={step}] {save_result.get('path', save_result)}", flush=True)
            _jsonl(metrics_path, {"event": "save", "step": step, "time": time.time(), "result": save_result})

    resume_from = max(0, int(getattr(args, "resume_from_step", 0)))
    if resume_from:
        print(
            f"[resume] starting at step {resume_from + 1} (server should have loaded the step-{resume_from:06d} "
            f"checkpoint via load_checkpoint_path; LR/data are step-keyed so this is exact)",
            flush=True,
        )

    if args.pipeline_rl:
        # Pipeline RL: a background thread generates step N+1's rollouts on the current
        # sampler weights while the trainer runs forward/backward+optim on step N, overlapping
        # the sampler-busy and trainer-busy phases. Consumed batches are one optim step stale
        # (off-policy by one step), which is why this path is meant to be paired with
        # --loss-fn policy_loss (PPO clipping bounds the importance ratio under that staleness).
        # Mirrors the working example in xorl-client examples/qwen_gsm8k_rl_loop.py.
        if not args.full_weight:
            raise ValueError("--pipeline-rl requires --full-weight (samplers receive the policy via full-weight sync)")
        ensure_sampler(resume_from)  # prime samplers with the (possibly resumed) policy for the worker's first rollout
        gen_queue: queue_module.Queue = queue_module.Queue(maxsize=1)
        gen_stop = threading.Event()

        def _generation_worker() -> None:
            try:
                for wstep in range(resume_from + 1, args.steps + 1):
                    while gen_queue.full() and not gen_stop.is_set():
                        time.sleep(0.1)
                    if gen_stop.is_set():
                        break
                    # full-weight has no LoRA to reload; the worker never re-syncs (main owns sync ordering).
                    trajectories, rollout_time_s = do_rollout(wstep, reload_cb=lambda: None)
                    if gen_stop.is_set():
                        break
                    gen_queue.put((wstep, trajectories, rollout_time_s))
            except Exception as exc:  # noqa: BLE001
                import traceback as _traceback

                print(
                    f"[pipeline] generation worker failed: {type(exc).__name__}: {exc}\n{_traceback.format_exc()}",
                    flush=True,
                )
            finally:
                try:
                    gen_queue.put_nowait(None)
                except queue_module.Full:
                    pass

        worker = threading.Thread(target=_generation_worker, name="grpo-gen-worker", daemon=True)
        worker.start()
        print(
            f"[pipeline] started generation worker (one-step-stale overlap); loss_fn={args.loss_fn} "
            f"eps_clip={args.eps_clip}/{args.eps_clip_high}",
            flush=True,
        )
        try:
            for step in range(resume_from + 1, args.steps + 1):
                if _hit_time_limit(step):
                    break
                item = gen_queue.get()
                if item is None:
                    print(
                        f"[pipeline] generation worker ended before producing step {step}; "
                        "stopping training loop",
                        flush=True,
                    )
                    break
                wstep, trajectories, rollout_time_s = item
                if wstep != step:
                    print(f"[pipeline] WARN: queued step {wstep} != expected {step}", flush=True)
                do_train(step, trajectories, rollout_time_s)
                # publish the freshly-updated policy so the worker's NEXT rollout is ~1 step stale
                if step < args.steps:
                    ensure_sampler(step)
        finally:
            gen_stop.set()
    else:
        for step in range(resume_from + 1, args.steps + 1):
            if _hit_time_limit(step):
                break
            ensure_sampler(step - 1)
            trajectories, rollout_time_s = do_rollout(
                step, reload_cb=lambda: ensure_sampler(step - 1, force=True)
            )
            do_train(step, trajectories, rollout_time_s)

    final_save = save_weights(args.train_url, args.train_model_id, "final", future_timeout=args.future_timeout)
    print(f"[done] final checkpoint: {final_save.get('path', final_save)}", flush=True)
    print(f"[done] metrics: {metrics_path}", flush=True)
    _jsonl(metrics_path, {"event": "done", "step": args.steps, "time": time.time(), "final_save": final_save})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
