"""Client-side on-policy distillation loop for XoRL.

This example keeps OPD orchestration in xorl-client instead of relying on a
repo-local shell/Python driver. It expects:

1. A XoRL student trainer API at ``base_url``.
2. One or more student sampler endpoints in ``inference_base_urls``.
3. A XoRL teacher prefill API at ``teacher_base_url``.

The loop is intentionally small: sample trajectories from the current student,
prefill those trajectories through the teacher into a shared hidden-state cache,
train the student with ``opd_loss``, optionally optimizer-step, and optionally
sync the trainer weights back to the registered sampler endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import numbers
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import requests

import xorl_client as tomi
from xorl_client.client.training_client import TrainingClient


logging.basicConfig(
    level=os.environ.get("OPD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xorl-client-opd")


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0 or chunk_size >= len(items):
        return [list(items)]
    return [
        list(items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    ]


def get_inference_urls(inference_base_urls: str, inference_port: int) -> list[str]:
    """Parse comma-separated inference URLs or research-common node suffixes."""
    urls: list[str] = []
    for part in _split_csv(inference_base_urls):
        if part.startswith("http://") or part.startswith("https://"):
            urls.append(part.rstrip("/"))
        else:
            urls.append(f"http://research-common-{part}:{inference_port}")
    if not urls:
        raise ValueError(
            "inference_base_urls must contain at least one URL or host suffix"
        )
    return urls


def _wait_for_future(base_url: str, request_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.post(
            f"{base_url}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") == "try_again":
            time.sleep(0.5)
            continue
        if payload.get("type") == "request_failed" or payload.get("error"):
            raise RuntimeError(f"Future {request_id} failed: {payload}")
        return payload
    raise TimeoutError(f"Future {request_id} timed out after {timeout}s")


def _wait_for_xorl(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200 and response.json().get("engine_running"):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"XoRL server at {base_url} not healthy within {timeout}s")


def _ensure_xorl_session(
    base_url: str,
    *,
    model_id: str,
    base_model: str,
    timeout: float,
) -> None:
    """Register model_id for direct /api/v1 calls on a XORL server."""
    response = requests.post(
        f"{base_url}/api/v1/create_session",
        json={"session_id": model_id, "base_model": base_model},
        timeout=min(timeout, 60.0),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Failed to register XORL session {model_id!r} at {base_url}: "
            f"{response.status_code} {response.text}"
        ) from exc
    logger.info(
        "Registered XORL session: base_url=%s model_id=%s response=%s",
        base_url,
        model_id,
        response.text,
    )


def _wait_for_sglang(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server at {base_url} not healthy within {timeout}s")


def _opd_causal_pair(sequence: list[int]) -> tuple[list[int], list[int]]:
    if len(sequence) < 2:
        raise ValueError(
            f"OPD trajectory must contain at least two tokens, got {len(sequence)}"
        )
    return list(sequence[:-1]), list(sequence[1:])


def _teacher_hidden_cache_data(
    sequences: list[list[int]],
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | None = None,
    prompt_token_lens: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build the teacher forward request payload.

    Two independent ways to give the teacher extra context beyond the student:

    1. ``teacher_prefix_tokens`` — prepended to the very start of every teacher
       input_ids. Their target positions are set to ``-100`` (IGNORE_INDEX) so
       they don't appear in the returned cache. This is a "system-instruction
       prefix" injection — the teacher reads the prefix BEFORE the user prompt.

    2. ``teacher_filler_tokens`` + ``prompt_token_lens`` — inserted AT the
       user→assistant boundary of each sample. This matches the filler-eval
       setup (pause/lorem/etc filler tokens added between the user prompt and
       the assistant answer), giving the teacher a "thinking budget" of fixed
       content right before it commits the answer.

    The xorl trainer's ``_split_hidden_cache_rows`` (model_runner.py:818)
    filters hidden states by ``target_tokens != IGNORE_INDEX``, so the
    resulting ``cache_indices_by_sample`` aligns 1-to-1 with the student-side
    OPD input regardless of how much extra context the teacher saw.
    """
    prefix = list(teacher_prefix_tokens) if teacher_prefix_tokens else []
    filler = list(teacher_filler_tokens) if teacher_filler_tokens else []
    use_filler_insert = bool(filler)
    if use_filler_insert and not prompt_token_lens:
        raise ValueError(
            "teacher_filler_tokens requires prompt_token_lens for per-sample boundary insertion"
        )
    if use_filler_insert and len(prompt_token_lens) != len(sequences):
        raise ValueError(
            f"prompt_token_lens has {len(prompt_token_lens)} entries for {len(sequences)} sequences"
        )

    data: list[dict[str, Any]] = []
    for idx, sequence in enumerate(sequences):
        input_ids, target_tokens = _opd_causal_pair(sequence)

        if use_filler_insert:
            # Insert `filler` at boundary position prompt_token_lens[idx].
            # The full teacher_seq = sequence[:p] + filler + sequence[p:], so:
            #   teacher_input  = teacher_seq[:-1]
            #   teacher_target = teacher_seq[1:]
            # The K positions [p-1, p-1+K) of teacher_target are predicting
            # filler tokens — mask them with -100 so the cache aligns 1-to-1
            # with the student's N-1 predictions.
            p = int(prompt_token_lens[idx])
            K = len(filler)
            full_teacher_seq = list(sequence[:p]) + filler + list(sequence[p:])
            input_ids = full_teacher_seq[:-1]
            target_tokens = list(full_teacher_seq[1:])
            # Mask the K filler-predicting positions
            for j in range(max(0, p - 1), max(0, p - 1) + K):
                if 0 <= j < len(target_tokens):
                    target_tokens[j] = -100

        if prefix:
            input_ids = prefix + input_ids
            target_tokens = [-100] * len(prefix) + target_tokens

        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {"target_tokens": target_tokens},
            }
        )
    return data


def _opd_loss_data(
    sequences: list[list[int]], cache_indices: list[list[int]]
) -> list[dict[str, Any]]:
    if len(cache_indices) != len(sequences):
        raise RuntimeError(
            f"Got {len(cache_indices)} cache-index lists for {len(sequences)} sequences"
        )

    data: list[dict[str, Any]] = []
    for sequence, indices in zip(sequences, cache_indices):
        input_ids, target_tokens = _opd_causal_pair(sequence)
        if len(indices) != len(input_ids):
            raise RuntimeError(
                f"Teacher cache index length {len(indices)} does not match OPD input length {len(input_ids)}"
            )
        data.append(
            {
                "model_input": {"input_ids": input_ids},
                "loss_fn_inputs": {
                    "target_tokens": target_tokens,
                    "teacher_ids": [0] * len(target_tokens),
                    "teacher_weights": [1.0] * len(target_tokens),
                    "teacher_cache_indices": indices,
                },
            }
        )
    return data


def _teacher_cache_from_xorl(
    teacher_url: str,
    sequences: list[list[int]],
    cache_path: Path,
    model_id: str,
    timeout: float,
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | None = None,
    prompt_token_lens: list[int] | None = None,
) -> dict[str, Any]:
    """Ask the XORL teacher server to write a hidden-state cache on shared storage."""
    response = requests.post(
        f"{teacher_url}/api/v1/forward",
        json={
            "model_id": model_id,
            "forward_input": {
                "data": _teacher_hidden_cache_data(
                    sequences,
                    teacher_prefix_tokens,
                    teacher_filler_tokens,
                    prompt_token_lens,
                ),
                "loss_fn": "teacher_hidden_cache",
                "loss_fn_params": {
                    "teacher_hidden_cache_path": str(cache_path),
                    "teacher_hidden_cache_dtype": "bfloat16",
                },
            },
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Teacher cache request failed: {response.status_code} {response.text}"
        ) from exc
    future = _wait_for_future(
        teacher_url, response.json()["request_id"], timeout=timeout
    )
    cache = future.get("info", {}).get("teacher_hidden_cache")
    if not cache:
        raise RuntimeError(
            f"XORL teacher did not return teacher_hidden_cache metadata: {future}"
        )

    cache_indices = cache.get("cache_indices_by_sample") or []
    if len(cache_indices) != len(sequences):
        raise RuntimeError(
            f"XORL teacher returned {len(cache_indices)} cache-index lists for {len(sequences)}"
        )
    for sequence, indices in zip(sequences, cache_indices):
        input_ids, _ = _opd_causal_pair(sequence)
        if len(indices) != len(input_ids):
            raise RuntimeError(
                f"XORL teacher returned {len(indices)} indices for input length {len(input_ids)}"
            )
    if cache.get("path") != str(cache_path):
        raise RuntimeError(
            f"XORL teacher wrote unexpected cache path: {cache.get('path')} != {cache_path}"
        )
    return {
        "cache_indices_by_sample": cache_indices,
        "metrics": future.get("metrics", {}),
        "info": future.get("info", {}),
    }


@dataclass
class PreparedOpdBatch:
    sequences: list[list[int]]
    data: list[dict[str, Any]]
    cache_path: Path
    metrics: dict[str, Any]


@chz.chz
class Config:
    base_url: str = "http://127.0.0.1:6000"
    teacher_base_url: str = "http://127.0.0.1:30002"
    inference_base_urls: str = "http://127.0.0.1:30001"
    inference_port: int = 30001
    inference_api_format: str = ""
    model_name: str = "default"
    model_id: str = "default"
    teacher_model_id: str = "default"
    teacher_head: str = ""
    chat_tokenizer_path: str = ""
    output_dir: str = "/tmp/xorl-client-opd"
    profile_output: str = ""

    num_steps: int = 2
    profile_warmup_steps: int = 1
    num_prompts: int = 2
    opd_microbatch_size: int = 0
    opd_prepare_batch_size: int = 0
    opd_prepare_concurrency: int = 1
    prompt_len: int = 32
    prompts_json: str = ""
    prompts_json_path: str = ""
    max_new_tokens: int = 8
    temperature: float = 1.0

    request_timeout: float = 900.0
    endpoint_timeout: float = 7200.0
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    skip_optim_step: bool = False

    sync_weights: bool = True
    sync_method: str = "p2p"
    weight_sync_master_address: str | None = None
    weight_sync_timeout: float = 1800.0

    opd_kl_backend: str = "streaming"
    opd_vocab_chunk_size: int | None = None
    opd_sharded_head_device_cache: bool = True
    profile_sync_cuda: bool = False

    # Teacher-prefix prepend for context-distillation recipes (e.g., teacher
    # sees an instruction system prompt the student does not). If non-empty,
    # `_teacher_hidden_cache_data` prepends these tokens to the teacher input
    # and marks their target_tokens as IGNORE_INDEX (-100) so the returned
    # `cache_indices_by_sample` aligns 1-to-1 with the student tokens.
    teacher_system_prefix: str = ""
    teacher_system_prefix_path: str = ""

    # Teacher filler-token INSERTION at the user→assistant boundary. The student
    # generates trajectories directly from the user prompt; the teacher sees
    # `teacher_filler_count` repetitions of `teacher_filler_text` inserted right
    # at the position where the assistant turn begins (= prompt_token_len). This
    # gives the teacher a "thinking budget" of fixed-content filler tokens that
    # the student does not get, matching the filler-eval methodology that showed
    # `pause`/`lorem`/`ellipsis` lift Qwen3.6-35B accuracy +4-6pp at 100 fillers.
    teacher_filler_text: str = ""       # e.g. " pause" (single token in Qwen tokenizer)
    teacher_filler_count: int = 0       # number of filler tokens to insert

    # Wandb logging. Disabled by default; set wandb_enabled=true to opt in.
    # WANDB_API_KEY must be set, or set wandb_mode="offline" to log locally only.
    wandb_enabled: bool = False
    wandb_project: str = "xorl-prefill-time-compute"
    wandb_entity: str = "together-research"
    wandb_run_name: str = ""             # auto-generated from output_dir if empty
    wandb_mode: str = "online"           # online | offline | disabled
    wandb_group: str = ""                # optional run grouping


def _maybe_init_wandb(config: Config):
    """Initialize wandb if `wandb_enabled` is set. Returns the run handle or None.

    Imports wandb lazily so the dependency is only required when actually used.
    Logs `OPD step N profile` rows via `wandb.log()` keyed by step number.
    """
    if not config.wandb_enabled:
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        logger.warning("wandb_enabled=true but `import wandb` failed; install wandb or set wandb_enabled=false")
        return None
    if not os.environ.get("WANDB_API_KEY") and config.wandb_mode == "online":
        logger.warning("WANDB_API_KEY not set; falling back to offline mode")
        os.environ.setdefault("WANDB_MODE", "offline")
    elif config.wandb_mode != "online":
        os.environ.setdefault("WANDB_MODE", config.wandb_mode)
    run_name = config.wandb_run_name or f"opd-{Path(config.output_dir).name}"
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=run_name,
        group=config.wandb_group or None,
        config={
            "model_name": config.model_name,
            "teacher_head": config.teacher_head,
            "num_steps": config.num_steps,
            "num_prompts": config.num_prompts,
            "opd_microbatch_size": config.opd_microbatch_size,
            "opd_prepare_batch_size": config.opd_prepare_batch_size,
            "opd_prepare_concurrency": config.opd_prepare_concurrency,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "learning_rate": config.learning_rate,
            "sync_method": config.sync_method,
            "sync_weights": config.sync_weights,
            "teacher_filler_text": config.teacher_filler_text,
            "teacher_filler_count": config.teacher_filler_count,
            "teacher_system_prefix": (config.teacher_system_prefix or "")[:200],
        },
        reinit=True,
    )
    logger.info("wandb run: %s (project=%s entity=%s)", run.url if run else "n/a", config.wandb_project, config.wandb_entity)
    return run


def _default_prompts(num_prompts: int, prompt_len: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for idx in range(num_prompts):
        base = 1000 + idx * 4096
        prompts.append([base + offset for offset in range(prompt_len)])
    return prompts


def _default_chat_prompts(num_prompts: int) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "role": "user",
                "content": f"Write a concise answer to synthetic OPD prompt {idx}.",
            }
        ]
        for idx in range(num_prompts)
    ]


def _uses_chat_completions(config: Config) -> bool:
    return config.inference_api_format.strip().lower().replace("-", "_") in {
        "chat",
        "openai",
        "openai_chat",
        "chat_completion",
        "chat_completions",
    }


def _load_prompts(config: Config) -> list[Any]:
    if config.prompts_json_path:
        # Path-based loading avoids ARG_MAX explosion for large prompt sets
        # (passing a 100k+-prompt JSON directly via `prompts_json=...` blows
        # past Linux's ~128 KB MAX_ARG_STRLEN).
        prompts = json.loads(Path(config.prompts_json_path).read_text())
    elif config.prompts_json:
        prompts = json.loads(config.prompts_json)
    elif _uses_chat_completions(config):
        prompts = _default_chat_prompts(config.num_prompts)
    else:
        prompts = _default_prompts(config.num_prompts, config.prompt_len)
    if not isinstance(prompts, list):
        raise ValueError("prompts_json must encode a list of prompts")

    normalized: list[Any] = []
    for prompt in prompts:
        if isinstance(prompt, str):
            normalized.append(prompt)
        elif isinstance(prompt, list) and all(
            isinstance(token, int) for token in prompt
        ):
            normalized.append([int(token) for token in prompt])
        elif (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(message, dict) for message in prompt)
        ):
            normalized.append(prompt)
        else:
            raise ValueError(
                "prompts_json entries must be token-id lists, strings, or chat-message lists"
            )
    return normalized


def _sample_prompt(prompt: Any) -> Any:
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return tomi.ModelInput.from_ints(prompt)
    return prompt


def _tokenize_teacher_prefix(
    config: Config, chat_tokenizer: Any | None
) -> list[int]:
    """Tokenize the teacher system-prompt prefix.

    Returns an empty list when no prefix is configured. Uses a paired
    [system, dummy-user] render and strips the user portion so the resulting
    tokens are exactly the system header that should be prepended to the
    student's input_ids (which already start with their own user turn).
    """
    text = config.teacher_system_prefix
    if not text and config.teacher_system_prefix_path:
        text = Path(config.teacher_system_prefix_path).read_text(encoding="utf-8")
    if not text:
        return []
    if chat_tokenizer is None:
        raise ValueError(
            "teacher_system_prefix requires a chat_tokenizer (set chat_tokenizer_path)"
        )
    # Many chat templates (Qwen, Llama) error if there's no user turn. Render
    # [system, dummy-user] then split on the dummy-user marker to recover the
    # pure system header.
    dummy_marker = "__XORL_OPD_DUMMY_USER__"
    rendered_full = chat_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": text},
            {"role": "user", "content": dummy_marker},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if dummy_marker not in rendered_full:
        raise RuntimeError(
            f"Tokenizer did not echo the dummy-user marker; rendered={rendered_full!r}"
        )
    # Everything before the dummy user turn is the system header. We also need
    # to drop the dummy "<|im_start|>user\n" preamble that the template emits
    # right before the marker. Easiest: locate the marker and walk backward to
    # the previous "<|im_start|>" (which opens the dummy user turn).
    idx = rendered_full.index(dummy_marker)
    # Find the start of the dummy user's opening token. Common open markers:
    # "<|im_start|>user", "<|start_header_id|>user", "<s>[INST]", etc.
    # Strategy: scan backwards from idx for the LAST occurrence of "<|" or "<s>"
    # marker open.
    cut = idx
    for marker in ("<|im_start|>user", "<|start_header_id|>user", "[INST]"):
        pos = rendered_full.rfind(marker, 0, idx)
        if pos != -1 and pos > cut - len(marker) - 8:
            cut = pos
            break
        if pos != -1:
            cut = min(cut, pos)
    system_prefix_text = rendered_full[:cut]
    encoded = chat_tokenizer.encode(system_prefix_text, add_special_tokens=False)
    return [int(token) for token in encoded]


def _load_chat_tokenizer(config: Config) -> Any | None:
    if not _uses_chat_completions(config):
        return None
    tokenizer_path = config.chat_tokenizer_path or config.model_name
    if not tokenizer_path or tokenizer_path == "default":
        raise ValueError(
            "chat_tokenizer_path must be set when inference_api_format=chat_completions"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _as_flat_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
    elif hasattr(value, "input_ids"):
        value = value.input_ids
        if hasattr(value, "tolist"):
            value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if isinstance(value, (list, tuple)) and all(
        isinstance(token, numbers.Integral) for token in value
    ):
        return [int(token) for token in value]
    raise ValueError("Tokenizer did not return a flat list of chat prompt token IDs")


def _encode_chat_prompt(prompt: Any, tokenizer: Any) -> list[int]:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif (
        isinstance(prompt, list)
        and prompt
        and all(isinstance(message, dict) for message in prompt)
    ):
        messages = prompt
    else:
        raise ValueError("Chat OPD prompt must be a string or chat-message list")

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return _as_flat_token_ids(token_ids)


def _sampled_sequence_tokens(
    prompt: Any, sampled: tomi.SampledSequence, chat_tokenizer: Any | None = None
) -> list[int]:
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return list(prompt) + list(sampled.tokens)
    if sampled.prompt_tokens:
        return list(sampled.prompt_tokens) + list(sampled.tokens)
    if chat_tokenizer is not None:
        return _encode_chat_prompt(prompt, chat_tokenizer) + list(sampled.tokens)
    if sampled.prompt_tokens is None:
        raise RuntimeError(
            "Chat-completions OPD sampling requires backend input_token_ids in each choice "
            "or a configured chat_tokenizer_path"
        )
    raise RuntimeError("Chat-completions backend returned empty input_token_ids")


async def _sample_student_batch(
    sampling_clients: list[tomi.SamplingClient],
    prompts: list[Any],
    max_new_tokens: int,
    temperature: float,
    return_logprobs: bool = False,
    chat_tokenizer: Any | None = None,
) -> tuple[list[list[int]], list[int]]:
    params = tomi.SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
    futures = []
    for idx, prompt in enumerate(prompts):
        client = sampling_clients[idx % len(sampling_clients)]
        futures.append(
            client.sample(
                prompt=_sample_prompt(prompt),
                sampling_params=params,
                num_samples=1,
                return_logprobs=return_logprobs,
            )
        )

    responses = await asyncio.gather(*futures)
    sequences: list[list[int]] = []
    prompt_token_lens: list[int] = []
    for prompt, response in zip(prompts, responses):
        if not response.sequences:
            raise RuntimeError("Student sampler returned no sequences")
        sampled = response.sequences[0]
        sequences.append(_sampled_sequence_tokens(prompt, sampled, chat_tokenizer))
        if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
            prompt_token_lens.append(len(prompt))
        elif sampled.prompt_tokens:
            prompt_token_lens.append(len(sampled.prompt_tokens))
        elif chat_tokenizer is not None:
            prompt_token_lens.append(len(_encode_chat_prompt(prompt, chat_tokenizer)))
        else:
            prompt_token_lens.append(0)
    return sequences, prompt_token_lens


async def _prepare_opd_batch(
    config: Config,
    sampling_clients: list[tomi.SamplingClient],
    teacher_url: str,
    prompts: list[Any],
    output_dir: Path,
    step: int,
    chat_tokenizer: Any | None = None,
    microbatch_idx: int = 0,
    teacher_prefix_tokens: list[int] | None = None,
    teacher_filler_tokens: list[int] | None = None,
) -> PreparedOpdBatch:
    prepare_t0 = time.perf_counter()
    sample_t0 = time.perf_counter()
    sequences, prompt_token_lens = await _sample_student_batch(
        sampling_clients,
        prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        return_logprobs=_uses_chat_completions(config),
        chat_tokenizer=chat_tokenizer,
    )
    sample_s = _elapsed(sample_t0)

    cache_path = (
        output_dir / f"teacher_hidden_step{step}_mb{microbatch_idx}.safetensors"
    )
    teacher_t0 = time.perf_counter()
    teacher_cache = await asyncio.to_thread(
        _teacher_cache_from_xorl,
        teacher_url,
        sequences,
        cache_path,
        config.teacher_model_id,
        config.request_timeout,
        teacher_prefix_tokens,
        teacher_filler_tokens,
        prompt_token_lens,
    )
    teacher_s = _elapsed(teacher_t0)
    teacher_tokens = sum(len(sequence) - 1 for sequence in sequences)
    sample_output_tokens = sum(
        max(0, len(sequence) - prompt_len)
        for sequence, prompt_len in zip(sequences, prompt_token_lens)
    )

    data = _opd_loss_data(sequences, teacher_cache["cache_indices_by_sample"])
    metrics = {
        "student_sampling_s": sample_s,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s
        if sample_s > 0
        else 0.0,
        "teacher_prefill_s": teacher_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_s
        if teacher_s > 0
        else 0.0,
        "teacher_prefill_forward_compute_s": teacher_cache["metrics"].get(
            "teacher_prefill_forward_compute_s", 0.0
        ),
        "teacher_hidden_cache_write_s": teacher_cache["metrics"].get(
            "teacher_hidden_cache_write_s", 0.0
        ),
        "prepare_s": _elapsed(prepare_t0),
    }
    return PreparedOpdBatch(
        sequences=sequences, data=data, cache_path=cache_path, metrics=metrics
    )


def _aggregate_prepared_metrics(
    prepared_batches: list[PreparedOpdBatch],
) -> dict[str, Any]:
    sample_s = sum(
        float(batch.metrics.get("student_sampling_s", 0.0))
        for batch in prepared_batches
    )
    sample_output_tokens = sum(
        int(batch.metrics.get("student_sampling_output_tokens", 0))
        for batch in prepared_batches
    )
    teacher_s = sum(
        float(batch.metrics.get("teacher_prefill_s", 0.0)) for batch in prepared_batches
    )
    teacher_tokens = sum(
        int(batch.metrics.get("teacher_prefill_tokens", 0))
        for batch in prepared_batches
    )
    prepare_s = sum(
        float(batch.metrics.get("prepare_s", 0.0)) for batch in prepared_batches
    )

    return {
        "num_prepare_batches": len(prepared_batches),
        "student_sampling_s": sample_s,
        "student_sampling_output_tokens": sample_output_tokens,
        "student_sampling_output_tok_per_s": sample_output_tokens / sample_s
        if sample_s > 0
        else 0.0,
        "teacher_prefill_s": teacher_s,
        "teacher_prefill_tokens": teacher_tokens,
        "teacher_prefill_tok_per_s": teacher_tokens / teacher_s
        if teacher_s > 0
        else 0.0,
        "teacher_prefill_forward_compute_s": sum(
            float(batch.metrics.get("teacher_prefill_forward_compute_s", 0.0))
            for batch in prepared_batches
        ),
        "teacher_hidden_cache_write_s": sum(
            float(batch.metrics.get("teacher_hidden_cache_write_s", 0.0))
            for batch in prepared_batches
        ),
        "prepare_s": prepare_s,
    }


def _loss_mean(output: tomi.ForwardBackwardOutput) -> float:
    losses = [item.loss for item in output.loss_fn_outputs if item.loss is not None]
    return _mean([float(loss) for loss in losses])


def _loss_mean_many(outputs: list[tomi.ForwardBackwardOutput]) -> float:
    losses = [
        float(item.loss)
        for output in outputs
        for item in output.loss_fn_outputs
        if item.loss is not None
    ]
    return _mean(losses)


def _valid_tokens(output: tomi.ForwardBackwardOutput) -> int | float:
    return output.metrics.get(
        "is_valid_tokens:sum",
        output.metrics.get("valid_tokens:sum", output.metrics.get("valid_tokens", 0)),
    )


def _weight_sync_master_address(config: Config) -> str | None:
    return (
        config.weight_sync_master_address
        or os.environ.get("XORL_WEIGHT_SYNC_MASTER_ADDRESS")
        or None
    )


async def main(config: Config) -> None:
    if not config.teacher_head:
        raise ValueError(
            "teacher_head must point to the teacher prediction head or teacher model path"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = (
        Path(config.profile_output)
        if config.profile_output
        else output_dir / "opd_profile.jsonl"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("", encoding="utf-8")

    wandb_run = _maybe_init_wandb(config)

    prompts = _load_prompts(config)
    if not prompts:
        raise ValueError("OPD requires at least one prompt")
    chat_tokenizer = _load_chat_tokenizer(config)
    teacher_prefix_tokens = _tokenize_teacher_prefix(config, chat_tokenizer)
    if teacher_prefix_tokens:
        logger.info(
            "Teacher system prefix: %d tokens (%r...)",
            len(teacher_prefix_tokens),
            (config.teacher_system_prefix or "<file>")[:80],
        )
    teacher_filler_tokens: list[int] = []
    if config.teacher_filler_count > 0 and config.teacher_filler_text:
        if chat_tokenizer is None:
            raise ValueError(
                "teacher_filler_text requires a chat_tokenizer (set chat_tokenizer_path)"
            )
        # Tokenize the filler text repeated `teacher_filler_count` times
        filler_string = config.teacher_filler_text * config.teacher_filler_count
        teacher_filler_tokens = [
            int(t) for t in chat_tokenizer.encode(filler_string, add_special_tokens=False)
        ]
        logger.info(
            "Teacher filler insert: %d tokens from %r × %d",
            len(teacher_filler_tokens),
            config.teacher_filler_text,
            config.teacher_filler_count,
        )
    student_urls = get_inference_urls(config.inference_base_urls, config.inference_port)

    logger.info("Waiting for trainer: %s", config.base_url)
    _wait_for_xorl(config.base_url, timeout=config.endpoint_timeout)
    logger.info("Waiting for teacher: %s", config.teacher_base_url)
    _wait_for_xorl(config.teacher_base_url, timeout=config.endpoint_timeout)
    for url in student_urls:
        logger.info("Waiting for student sampler: %s", url)
        _wait_for_sglang(url, timeout=config.endpoint_timeout)
    _ensure_xorl_session(
        config.base_url,
        model_id=config.model_id,
        base_model=config.model_name,
        timeout=config.request_timeout,
    )
    _ensure_xorl_session(
        config.teacher_base_url,
        model_id=config.teacher_model_id,
        base_model=config.teacher_head,
        timeout=config.request_timeout,
    )

    service_client = tomi.ServiceClient(
        base_url=config.base_url, timeout=config.request_timeout
    )
    training_client = TrainingClient(
        holder=service_client.holder,
        model_id=config.model_id,
        base_model=config.model_name,
    )
    sampling_clients = [
        tomi.SamplingClient(
            base_url=url,
            model=config.model_name,
            timeout=config.request_timeout,
            api_format=config.inference_api_format or None,
        )
        for url in student_urls
    ]
    train_microbatch_size = config.opd_microbatch_size
    prepare_batch_size = config.opd_prepare_batch_size or train_microbatch_size
    prepare_concurrency = max(1, config.opd_prepare_concurrency)
    prompt_batches = _chunked(prompts, prepare_batch_size)

    rows: list[dict[str, Any]] = []
    for step in range(config.num_steps):
        logger.info("=== OPD step %s ===", step)
        step_t0 = time.perf_counter()

        prepare_window_t0 = time.perf_counter()
        prepared_batches: list[PreparedOpdBatch] = []
        fb_futures: list[Any] = []
        first_fb_submit_t0: float | None = None
        train_microbatch_count = 0

        next_prepare_idx = 0
        completed_prepare_count = 0
        inflight_prepare: dict[asyncio.Task[PreparedOpdBatch], int] = {}

        def schedule_prepare_batches() -> None:
            nonlocal next_prepare_idx
            while (
                next_prepare_idx < len(prompt_batches)
                and len(inflight_prepare) < prepare_concurrency
            ):
                task = asyncio.create_task(
                    _prepare_opd_batch(
                        config,
                        sampling_clients,
                        config.teacher_base_url,
                        prompt_batches[next_prepare_idx],
                        output_dir,
                        step,
                        chat_tokenizer,
                        next_prepare_idx,
                        teacher_prefix_tokens,
                        teacher_filler_tokens,
                    )
                )
                inflight_prepare[task] = next_prepare_idx
                next_prepare_idx += 1

        schedule_prepare_batches()
        while inflight_prepare:
            done, _ = await asyncio.wait(
                inflight_prepare.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                inflight_prepare.pop(task)
                prepared = task.result()
                completed_prepare_count += 1
                schedule_prepare_batches()

                prepared_batches.append(prepared)

                loss_params: dict[str, Any] = {
                    "teacher_heads": {"0": config.teacher_head},
                    "teacher_hidden_caches": {"0": str(prepared.cache_path)},
                    "opd_sort_by_teacher": True,
                    "opd_kl_backend": config.opd_kl_backend,
                    "opd_vocab_chunk_size": config.opd_vocab_chunk_size,
                    "opd_sharded_head_device_cache": config.opd_sharded_head_device_cache,
                    "opd_profile_timings": True,
                    "opd_profile_sync_cuda": config.profile_sync_cuda,
                    "num_chunks": 8,
                }
                train_batches = _chunked(prepared.data, train_microbatch_size)
                for train_idx, train_batch in enumerate(train_batches):
                    train_loss_params = dict(loss_params)
                    is_last_train_batch = (
                        completed_prepare_count == len(prompt_batches)
                        and train_idx == len(train_batches) - 1
                    )
                    if config.skip_optim_step and is_last_train_batch:
                        train_loss_params["profile_clear_gradients_after_backward"] = True
                    if first_fb_submit_t0 is None:
                        first_fb_submit_t0 = time.perf_counter()
                    fb_futures.append(
                        training_client.forward_backward(
                            train_batch,
                            loss_fn="opd_loss",
                            loss_fn_params=train_loss_params,
                        )
                    )
                    train_microbatch_count += 1
        prepare_window_s = _elapsed(prepare_window_t0)

        optim_future = None
        optim_t0 = None
        if not config.skip_optim_step:
            optim_t0 = time.perf_counter()
            optim_future = training_client.optim_step(
                tomi.AdamParams(
                    learning_rate=config.learning_rate,
                    grad_clip_norm=config.grad_clip_norm,
                )
            )

        fb_t0 = first_fb_submit_t0 or time.perf_counter()
        fb_results = await asyncio.gather(*fb_futures)
        fb_s = _elapsed(fb_t0)

        optim_s = 0.0
        optim_queued_s = 0.0
        if optim_future is not None:
            optim_wait_t0 = time.perf_counter()
            await optim_future
            optim_s = _elapsed(optim_wait_t0)
            optim_queued_s = _elapsed(optim_t0 or time.perf_counter())

        sync_s = 0.0
        sync_result = None
        sync_failure: str | None = None
        if config.sync_weights and not config.skip_optim_step:
            sync_t0 = time.perf_counter()
            sync_result = await training_client.sync_weights_to_inference(
                sync_method=config.sync_method,
                master_address=_weight_sync_master_address(config),
                timeout=config.weight_sync_timeout,
            )
            sync_s = _elapsed(sync_t0)
            if not sync_result.success:
                sync_failure = sync_result.message or "sync_weights_to_inference failed"

        valid_tokens = sum(_valid_tokens(result) for result in fb_results)
        prepared_metrics = _aggregate_prepared_metrics(prepared_batches)
        row: dict[str, Any] = {
            "step": step,
            "profile_warmup": step < config.profile_warmup_steps,
            "step_total_s": _elapsed(step_t0),
            "opd_microbatch_size": config.opd_microbatch_size,
            "opd_prepare_batch_size": prepare_batch_size,
            "opd_prepare_concurrency": prepare_concurrency,
            "opd_train_microbatch_size": train_microbatch_size,
            "num_microbatches": train_microbatch_count,
            "prepare_window_s": prepare_window_s,
            "forward_backward_s": fb_s,
            "optim_step_s": optim_s,
            "optim_step_queued_s": optim_queued_s,
            "sync_inference_weights_s": sync_s,
            "loss": _loss_mean_many(fb_results),
            "valid_tokens": valid_tokens,
            **prepared_metrics,
        }
        if row["step_total_s"] > 0:
            row["valid_tokens_per_step_s"] = valid_tokens / row["step_total_s"]
        if prepare_window_s > 0:
            row["student_sampling_output_tok_per_prepare_window_s"] = (
                row["student_sampling_output_tokens"] / prepare_window_s
            )
            row["teacher_prefill_tok_per_prepare_window_s"] = (
                row["teacher_prefill_tokens"] / prepare_window_s
            )
        if sync_result is not None:
            row.update(
                {
                    "sync_success": sync_result.success,
                    "sync_message": sync_result.message,
                    "sync_transfer_time_s": sync_result.transfer_time,
                    "sync_total_bytes": sync_result.total_bytes,
                    "sync_num_buckets": sync_result.num_buckets,
                }
            )
        rows.append(row)
        with profile_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        logger.info("OPD step %s profile: %s", step, json.dumps(row, sort_keys=True))
        if wandb_run is not None:
            # Log every numeric field from the profile row to wandb, keyed by step.
            wandb_metrics = {
                k: v for k, v in row.items()
                if isinstance(v, numbers.Number) and not isinstance(v, bool)
            }
            wandb_metrics["sync_success_int"] = int(bool(row.get("sync_success", True)))
            wandb_run.log(wandb_metrics, step=step)
        if sync_failure is not None:
            raise RuntimeError(f"OPD weight sync failed at step {step}: {sync_failure}")

    steady = [row for row in rows if not row["profile_warmup"]]
    if steady:
        steady_summary = {
            "steady/total_s": _mean([row["step_total_s"] for row in steady]),
            "steady/sample_s": _mean([row["student_sampling_s"] for row in steady]),
            "steady/teacher_s": _mean([row["teacher_prefill_s"] for row in steady]),
            "steady/fwd_bwd_s": _mean([row["forward_backward_s"] for row in steady]),
            "steady/sync_s": _mean([row["sync_inference_weights_s"] for row in steady]),
        }
        logger.info(
            "Steady OPD mean: total=%(steady/total_s).3fs sample=%(steady/sample_s).3fs "
            "teacher=%(steady/teacher_s).3fs fwd_bwd=%(steady/fwd_bwd_s).3fs sync=%(steady/sync_s).3fs",
            steady_summary,
        )
        if wandb_run is not None:
            wandb_run.summary.update(steady_summary)
    logger.info("Wrote OPD profile rows to %s", profile_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":

    def _main(config: Config) -> None:
        asyncio.run(main(config))

    chz.nested_entrypoint(_main)
