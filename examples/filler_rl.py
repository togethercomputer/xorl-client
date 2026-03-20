"""
Minimal filler-token RL example with pipeline mode.

Trains models to generate random-number filler tokens before answering
4-digit multiplication problems. Uses GRPO-style reward centering with
answer-only advantage weighting and pipeline RL (generation overlaps training).

Reward structure:
- 0.0: Output doesn't match format (filler tokens + "Answer: X")
- 0.2: Output matches format but answer is incorrect
- 1.0: Output matches format and answer is correct

Usage:
    cd /data/apanda/xorl-client && uv run examples/filler_rl.py \
        training_url=http://research-common-21:5555 \
        inference_base_urls=08 \
        batch_size=32 group_size=32 N=1000
"""

import asyncio
import logging
import os
import queue as queue_module
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from zoneinfo import ZoneInfo

import chz
import torch
import xorl_client
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tqdm import tqdm
from xorl_client import ModelInput, types
from xorl_client.types.tensor_data import TensorData

# PST timezone logging
_PST = ZoneInfo("America/Los_Angeles")


def _pst_format_time(self, record, datefmt=None):
    dt = datetime.fromtimestamp(record.created, tz=_PST)
    if datefmt:
        return dt.strftime(datefmt)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def apply_pst_formatters():
    """Patch all root logger handlers in-place to use PST timestamps."""
    import types as _types

    for handler in logging.root.handlers:
        formatter = handler.formatter
        if formatter is None:
            continue
        fmt_str = getattr(formatter, "_fmt", None) or "%(message)s"
        if "%(asctime)s" not in fmt_str:
            new_fmt = "%(asctime)s " + fmt_str
            formatter._fmt = new_fmt
            if hasattr(formatter, "_style"):
                formatter._style._fmt = new_fmt
        formatter.formatTime = _types.MethodType(_pst_format_time, formatter)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
apply_pst_formatters()


# Monkey-patch: get_text_content is missing from PyPI version of tinker_cookbook
def _get_text_content(message):
    """Extract text content from a parsed message."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts)
    return str(message)


renderers.get_text_content = _get_text_content

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """/no_think
You will be given a 4-digit multiplication problem.
You may generate up to {filler_tokens} filler tokens before answering. These should be random numbers.
Format:
[filler tokens]
Answer: [number]"""


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------


def generate_multiplication_problems(n: int, seed: int = 12345) -> list[dict]:
    """Generate n 4-digit multiplication problems."""
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a = rng.randint(1000, 9999)
        b = rng.randint(1000, 9999)
        problems.append({"problem": f"What is {a} * {b}?", "answer": a * b})
    return problems


# ---------------------------------------------------------------------------
# Filler token generation
# ---------------------------------------------------------------------------


def generate_filler_tokens(n: int, tokenizer) -> str:
    """Generate random-number filler trimmed to approximately n tokens."""
    elements = []
    for _ in range(n * 2):  # overshoot, then trim
        elements.append(str(random.randint(0, 999)))
        if len(tokenizer.encode(" ".join(elements))) >= n:
            break
    return " ".join(elements)


# ---------------------------------------------------------------------------
# Few-shot prompt construction
# ---------------------------------------------------------------------------


def create_fewshot_prompt(
    fewshot_problems: list[dict],
    eval_problem: dict,
    num_fewshot: int,
    filler_tokens: int,
    system_prompt: str,
    tokenizer,
) -> list[Message]:
    """Create a prompt with few-shot examples for RL generation."""
    messages: list[Message] = [Message(role="system", content=system_prompt)]

    for i in range(num_fewshot):
        problem = fewshot_problems[i]
        messages.append(Message(role="user", content=problem["problem"]))
        if filler_tokens > 0:
            filler_str = generate_filler_tokens(filler_tokens, tokenizer)
            assistant_response = f"{filler_str}\nAnswer: {problem['answer']}"
        else:
            assistant_response = f"Answer: {problem['answer']}"
        messages.append(Message(role="assistant", content=assistant_response))

    messages.append(Message(role="user", content=eval_problem["problem"]))
    return messages


# ---------------------------------------------------------------------------
# Response parsing + reward
# ---------------------------------------------------------------------------


def _strip_special_tokens(text: str) -> str:
    """Strip special tokens that may leak into decoded content."""
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "")


def parse_response(response: str) -> tuple[bool, str | None]:
    """Parse response for filler + Answer: format.

    Returns (format_valid, extracted_answer).
    """
    response = _strip_special_tokens(response).strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    answer_match = re.search(r"Answer:\s*([^\s]+)", response)
    if not answer_match:
        return (False, None)

    # Reject trailing content after answer
    if response[answer_match.end() :].strip():
        return (False, None)

    # Validate filler tokens (digits only)
    before_answer = response[: answer_match.start()].strip()
    if before_answer:
        for token in before_answer.split():
            if not token.isdigit():
                return (False, None)

    return (True, answer_match.group(1).strip())


def get_reward(response: str, ground_truth: str) -> float:
    """Compute reward: 0.2 for valid format, +0.8 for correct answer."""
    format_valid, extracted_answer = parse_response(response)
    if not format_valid:
        return 0.0
    reward = 0.2
    if extracted_answer is not None:
        try:
            if abs(float(extracted_answer) - float(ground_truth)) < 1e-6:
                reward += 0.8
        except (ValueError, TypeError):
            if extracted_answer == ground_truth:
                reward += 0.8
    return reward


# ---------------------------------------------------------------------------
# find_answer_token_index
# ---------------------------------------------------------------------------


def find_answer_token_index(tokens: list[int], tokenizer) -> int | None:
    """Find the token index where 'Answer:' starts in the sampled tokens."""
    text = tokenizer.decode(tokens)
    answer_pos = text.find("Answer:")
    if answer_pos == -1:
        return None

    char_count = 0
    for i, tok in enumerate(tokens):
        tok_text = tokenizer.decode([tok])
        if char_count >= answer_pos:
            return i
        char_count += len(tok_text)
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class Config:
    training_url: str = "http://localhost:6000"
    inference_base_urls: str = ""
    log_path: str = "/data/outputs/filler-rl"
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    N: int = 2500
    num_fewshot: int = 10
    filler_tokens: int = 100
    seed: int = 12345
    num_epochs: int = 1
    batch_size: int = 16
    group_size: int = 64
    learning_rate: float = 5e-6
    min_lr: float = 1e-7
    warmup_steps: int = 10
    grad_clip_norm: float = 1.0
    max_tokens: int = 1024
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    temperature: float = 1.0
    sync_method: str = "nccl_ep_scatter"
    return_routed_experts: bool = True

    save_every: int = 250
    resume_model_id: str | None = None
    resume_step: int = 0
    init_checkpoint: str | None = None
    debug_samples: int = 3
    wandb_name: str | None = None


# ---------------------------------------------------------------------------
# Inference URL parsing
# ---------------------------------------------------------------------------


def get_inference_urls(config: Config) -> list[str]:
    """Parse inference_base_urls as comma-separated URLs or host IDs."""
    if not config.inference_base_urls:
        raise ValueError(
            "Must specify inference_base_urls (comma-separated URLs or host identifiers, e.g. '26,13')"
        )
    urls = []
    for part in config.inference_base_urls.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("http"):
            urls.append(part)
        else:
            urls.append(f"http://research-common-{part}:12345")
    return urls


# ---------------------------------------------------------------------------
# Training session setup
# ---------------------------------------------------------------------------


def setup_training_session(config: Config, service_client, model_id: str, elapsed):
    """Set up full-weights training session. Returns (training_client, model_id, start_step)."""
    from xorl_client.client.training_client import TrainingClient

    logger.info(f"{elapsed()} [session] Checking resume...")

    if config.resume_model_id:
        model_id = config.resume_model_id
        start_step = config.resume_step
        logger.info(f"Resuming existing session: model_id={model_id}, start_step={start_step}")
        response = service_client.holder.post_sync(
            "/api/v1/create_model",
            {"model_id": model_id, "base_model": config.model_name, "lora_config": {"rank": 32}},
        )
        logger.info(f"Re-registered model_id: {response}")
        training_client = TrainingClient(
            holder=service_client.holder, model_id=model_id, base_model=config.model_name
        )
    else:
        logger.info(f"Creating full-weights training client with model_id={model_id}")

        # Kill existing sessions
        logger.info(f"{elapsed()} Querying session_info and killing existing sessions...")
        try:
            session_info = service_client.holder.get_sync("/api/v1/session_info")
            for existing_model_id in session_info.get("registered_models", []):
                try:
                    kill_resp = service_client.holder.post_sync(
                        "/api/v1/kill_session",
                        {"model_id": existing_model_id, "save_checkpoint": False},
                    )
                    logger.info(f"{elapsed()} Killed session '{existing_model_id}': {kill_resp.get('message', 'OK')}")
                except Exception as e:
                    logger.warning(f"{elapsed()} Failed to kill session '{existing_model_id}': {e}")
        except Exception as e:
            logger.warning(f"{elapsed()} Could not query session info: {e}")

        logger.info(f"{elapsed()} Calling create_model...")
        response = service_client.holder.post_sync(
            "/api/v1/create_model",
            {"model_id": model_id, "base_model": config.model_name, "lora_config": {"rank": 32}},
        )
        logger.info(f"{elapsed()} create_model complete: {response}")

        training_client = TrainingClient(
            holder=service_client.holder, model_id=model_id, base_model=config.model_name
        )
        start_step = 0

    logger.info(f"{elapsed()} [session] Training session setup complete: model_id={model_id}, start_step={start_step}")
    return training_client, model_id, start_step


# ---------------------------------------------------------------------------
# Generation batch result
# ---------------------------------------------------------------------------


@dataclass
class GenerationBatchResult:
    datums: list[types.Datum]
    mean_rewards: list[float]
    format_rates: list[float]
    t_sample: float
    debug_samples: list[tuple[str, str, bool, float]]
    total_samples: int
    correct_count: int


# ---------------------------------------------------------------------------
# Single batch generation
# ---------------------------------------------------------------------------


async def _generate_single_batch(
    batch_problems: list[dict],
    config: Config,
    sampling_clients: list,
    sampling_params,
    fewshot_problems: list[dict],
    renderer,
    tokenizer,
    system_prompt: str,
    global_step: int,
) -> GenerationBatchResult:
    """Generate samples for a batch, compute rewards/advantages, and create datums."""
    datums: list[types.Datum] = []
    mean_rewards: list[float] = []
    format_rates: list[float] = []
    debug_sample_contents: list[tuple[str, str, bool, float]] = []
    debug_printed = 0
    total_samples = 0
    correct_count = 0

    # Submit all sampling requests up front
    t_sample_start = time.time()
    all_futures: list[tuple[list, ModelInput, str]] = []

    for problem_idx, problem in enumerate(batch_problems):
        prompt_messages = create_fewshot_prompt(
            fewshot_problems=fewshot_problems,
            eval_problem=problem,
            num_fewshot=config.num_fewshot,
            filler_tokens=config.filler_tokens,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
        model_input = renderer.build_generation_prompt(prompt_messages)
        client = sampling_clients[problem_idx % len(sampling_clients)]
        sample_futures = [
            client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=replace(
                    sampling_params, sampling_seed=random.randint(0, 2**63 - 1)
                ),
            )
            for _ in range(config.group_size)
        ]
        all_futures.append((sample_futures, model_input, str(problem["answer"])))

    # Process results
    for sample_futures, prompt, ground_truth in tqdm(
        all_futures, desc=f"Sampling batch {global_step}"
    ):
        rewards_G: list[float] = []
        sampled_tokens_G: list[list[int]] = []
        logprobs_G: list[list[float]] = []
        routed_experts_G: list[list[list[list[int]]] | None] = []
        format_correct_G: list[float] = []

        for future in sample_futures:
            try:
                sample_result = await future
                sequence = sample_result.sequences[0]
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                routed_experts = (
                    sample_result.meta_info.get("routed_experts")
                    if sample_result.meta_info
                    else None
                )

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = parsed_message.get("content", "") if parsed_message else (sequence.text or "")
                content = _strip_special_tokens(content)

                format_valid, _ = parse_response(content)
                reward = get_reward(content, ground_truth)

                rewards_G.append(reward)
                format_correct_G.append(1.0 if format_valid else 0.0)
                sampled_tokens_G.append(sampled_tokens)
                logprobs_G.append(sampled_logprobs)
                routed_experts_G.append(routed_experts)
                total_samples += 1
                if reward >= 0.99:
                    correct_count += 1

                if debug_printed < config.debug_samples:
                    debug_sample_contents.append((content, ground_truth, format_valid, reward))
                    debug_printed += 1
            except Exception as e:
                logger.warning(f"Sample failed: {type(e).__name__}: {e}")

        if len(rewards_G) == 0:
            continue

        mean_reward = sum(rewards_G) / len(rewards_G)
        # maxrl normalization: (reward - mean) / (mean + eps)
        advantages_G = [(r - mean_reward) / (mean_reward + 1e-6) for r in rewards_G]

        mean_rewards.append(mean_reward)
        format_rates.append(sum(format_correct_G) / len(format_correct_G))

        if all(a == 0.0 for a in advantages_G):
            continue

        # Create datums with answer_only weighting
        prompt_tokens = prompt.to_ints()
        ob_len = len(prompt_tokens) - 1

        for sampled_tokens, logprobs, advantage, routed_experts in zip(
            sampled_tokens_G, logprobs_G, advantages_G, routed_experts_G
        ):
            if not sampled_tokens:
                continue

            all_tokens = prompt_tokens + sampled_tokens
            input_tokens = all_tokens[:-1]
            target_tokens = [-100] * ob_len + all_tokens[ob_len + 1 :]
            padded_logprobs = [0.0] * ob_len + logprobs

            # answer_only weighting: 0 advantage for filler, full for answer
            answer_idx = find_answer_token_index(sampled_tokens, tokenizer)
            num_sampled = len(sampled_tokens)
            if answer_idx is not None:
                token_advantages = [
                    advantage if i >= answer_idx else 0.0 for i in range(num_sampled)
                ]
            else:
                token_advantages = [advantage] * num_sampled
            padded_advantages = [0.0] * ob_len + token_advantages

            assert (
                len(input_tokens)
                == len(target_tokens)
                == len(padded_logprobs)
                == len(padded_advantages)
            )

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                },
                routed_experts=routed_experts,
            )
            datums.append(datum)

    t_sample = time.time() - t_sample_start

    return GenerationBatchResult(
        datums=datums,
        mean_rewards=mean_rewards,
        format_rates=format_rates,
        t_sample=t_sample,
        debug_samples=debug_sample_contents,
        total_samples=total_samples,
        correct_count=correct_count,
    )


# ---------------------------------------------------------------------------
# Generation worker thread (pipeline RL)
# ---------------------------------------------------------------------------


def _generation_worker(
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    eval_problems: list[dict],
    config: Config,
    sampling_clients: list,
    sampling_params,
    fewshot_problems: list[dict],
    renderer,
    tokenizer,
    system_prompt: str,
    batches_per_epoch: int,
    start_batch_idx: int = 1,
):
    """Background thread: continuously generate batches and enqueue results."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for batch_idx in range(start_batch_idx, batches_per_epoch):
            if stop_event.is_set():
                break

            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(eval_problems))
            batch_problems = eval_problems[batch_start:batch_end]

            try:
                result = loop.run_until_complete(
                    _generate_single_batch(
                        batch_problems=batch_problems,
                        config=config,
                        sampling_clients=sampling_clients,
                        sampling_params=sampling_params,
                        fewshot_problems=fewshot_problems,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        system_prompt=system_prompt,
                        global_step=batch_idx,
                    )
                )
                output_queue.put(result)
            except Exception as e:
                logger.error(f"Generation worker batch {batch_idx} failed: {e}")
                continue
    finally:
        output_queue.put(None)  # sentinel
        loop.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(config: Config):
    _start_time = time.time()

    def elapsed():
        return f"[{time.time() - _start_time:.2f}s]"

    random.seed(config.seed)

    # Start tokenizer + session setup concurrently
    logger.info(f"{elapsed()} Starting tokenizer loading in background...")
    tokenizer_task = asyncio.create_task(asyncio.to_thread(get_tokenizer, config.model_name))

    logger.info(f"{elapsed()} Starting service session setup in background...")
    service_client = xorl_client.ServiceClient(base_url=config.training_url, timeout=1200.0)
    run_id = uuid.uuid4().hex[:8]
    model_id = f"filler-rl-{run_id}"
    log_path = os.path.join(config.log_path, model_id)

    session_task = asyncio.create_task(
        asyncio.to_thread(setup_training_session, config, service_client, model_id, elapsed)
    )

    # Setup logging
    wandb_run_name = config.wandb_name or f"filler rl {config.model_name.split('/')[-1]}"
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project="tinker-xorl",
        wandb_name=wandb_run_name,
        config=config,
        do_configure_logging_module=True,
    )
    apply_pst_formatters()

    # Format system prompt and generate problems
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(filler_tokens=config.filler_tokens)
    logger.info(f"System prompt:\n{system_prompt}")

    # Wait for tokenizer
    logger.info(f"{elapsed()} Waiting for tokenizer...")
    tokenizer = await tokenizer_task
    logger.info(f"{elapsed()} Tokenizer ready")

    # Renderer
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"{elapsed()} Using renderer: {renderer_name}")

    # Generate problems
    all_problems = generate_multiplication_problems(
        config.N + config.num_fewshot + 100, seed=config.seed
    )
    fewshot_problems = all_problems[: config.num_fewshot]
    eval_problems = all_problems[config.num_fewshot : config.num_fewshot + config.N]
    logger.info(f"Generated {len(eval_problems)} eval problems, {config.num_fewshot} few-shot examples")

    # Log example prompt
    example_messages = create_fewshot_prompt(
        fewshot_problems=fewshot_problems,
        eval_problem=eval_problems[0],
        num_fewshot=min(3, config.num_fewshot),
        filler_tokens=config.filler_tokens,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
    )
    for msg in example_messages:
        content_preview = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        logger.info(f"  [{msg['role']}]: {content_preview}")

    # Wait for training session
    logger.info(f"{elapsed()} Waiting for training session setup...")
    training_client, model_id, start_step = await session_task
    logger.info(f"{elapsed()} Training session ready: model_id={model_id}, start_step={start_step}")

    # Load initial checkpoint if specified
    load_state_future = None
    if config.init_checkpoint:
        logger.info(f"{elapsed()} Starting load_state: {config.init_checkpoint}")
        load_state_future = training_client.load_state(config.init_checkpoint)

    # Verify inference endpoints
    logger.info(f"{elapsed()} Verifying inference endpoint...")
    endpoints = training_client.list_inference_endpoints().result()
    if endpoints.count == 0:
        raise RuntimeError(
            "No inference endpoints registered. Register via API before running."
        )
    logger.info(f"{elapsed()} Found {endpoints.count} inference endpoint(s)")

    # Create sampling clients
    inference_urls = get_inference_urls(config)
    sampling_clients = [xorl_client.SamplingClient(base_url=url) for url in inference_urls]
    logger.info(f"{elapsed()} Created {len(sampling_clients)} sampling client(s): {inference_urls}")

    # Sampling parameters
    sampling_params = xorl_client.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop_token_ids=renderer.get_stop_sequences(),
        return_routed_experts=config.return_routed_experts,
    )

    def get_lr(step: int) -> float:
        if step < config.warmup_steps:
            return config.min_lr + (config.learning_rate - config.min_lr) * (step / config.warmup_steps)
        return config.learning_rate

    # Wait for load_state
    if load_state_future is not None:
        logger.info(f"{elapsed()} Waiting for load_state...")
        load_result = load_state_future.result()
        logger.info(f"{elapsed()} load_state complete: {load_result.path}")

    # Compute batches
    batches_per_epoch = config.N // config.batch_size
    total_steps = batches_per_epoch * config.num_epochs
    logger.info(f"Batches per epoch: {batches_per_epoch}, Total steps: {total_steps}")

    # Shuffle problems
    random.shuffle(eval_problems)

    # Pipeline RL state
    generation_queue: queue_module.Queue | None = None
    gen_thread: threading.Thread | None = None
    stop_event: threading.Event | None = None
    global_step = start_step

    for epoch in range(config.num_epochs):
        logger.info(f"{elapsed()} Starting epoch {epoch + 1}/{config.num_epochs}")

        if epoch > 0:
            random.shuffle(eval_problems)

        for batch_idx in range(batches_per_epoch):
            if global_step < start_step:
                global_step += 1
                continue

            t_start = time.time()
            current_lr = get_lr(global_step)
            adam_params = xorl_client.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                grad_clip_norm=config.grad_clip_norm,
            )

            metrics: dict[str, float] = {
                "progress/batch": global_step,
                "progress/epoch": epoch + 1,
                "optim/lr": current_lr,
                "progress/done_frac": global_step / total_steps,
            }

            # Save checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                checkpoint_name = f"{model_id}-{global_step:06d}"
                logger.info(f"Saving checkpoint: {checkpoint_name}")
                save_result = training_client.save_full_weights_safetensors(
                    name=checkpoint_name
                ).result()
                logger.info(f"Checkpoint saved to: {save_result.path} ({save_result.num_shards} shards)")

            # Get batch problems
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(eval_problems))
            batch_problems = eval_problems[batch_start:batch_end]

            # ── Generation ──
            t_sync = 0
            if generation_queue is not None:
                # Pipeline mode: get pre-generated batch from worker thread
                gen_result = generation_queue.get()
                if gen_result is None:
                    logger.warning("Generation worker finished early")
                    break
            else:
                # Synchronous mode (step 0): sync weights then generate
                logger.info(f"{elapsed()} Starting sync_weights_to_inference...")
                t_sync_start = time.time()
                sync_result = training_client.sync_weights_to_inference(
                    sync_method=config.sync_method
                ).result()
                t_sync = time.time() - t_sync_start

                if sync_result.success:
                    metrics["time/weight_sync"] = t_sync
                    metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                    metrics["sync/throughput_gbps"] = sync_result.throughput_gbps
                    logger.info(
                        f"{elapsed()} Step {global_step}: Weight sync "
                        f"{sync_result.total_bytes/1e9:.2f} GB in "
                        f"{t_sync:.2f}s ({sync_result.throughput_gbps:.2f} GB/s)"
                    )
                else:
                    logger.warning(f"{elapsed()} Weight sync failed: {sync_result.message}")

                gen_result = await _generate_single_batch(
                    batch_problems=batch_problems,
                    config=config,
                    sampling_clients=sampling_clients,
                    sampling_params=sampling_params,
                    fewshot_problems=fewshot_problems,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    global_step=global_step,
                )

            # Unpack generation results
            datums = gen_result.datums
            metrics["time/sampling"] = gen_result.t_sample

            # Log reward metrics
            if gen_result.mean_rewards:
                mean_reward = sum(gen_result.mean_rewards) / len(gen_result.mean_rewards)
                format_rate = sum(gen_result.format_rates) / len(gen_result.format_rates)
                correct_rate = gen_result.correct_count / gen_result.total_samples if gen_result.total_samples > 0 else 0.0
                metrics["reward/mean"] = mean_reward
                metrics["reward/format_correct_rate"] = format_rate
                metrics["reward/correct_rate"] = correct_rate
                logger.info(
                    f"Step {global_step} rewards: mean={mean_reward:.4f}, "
                    f"correct={correct_rate:.1%}, format={format_rate:.1%} "
                    f"({gen_result.total_samples} samples)"
                )

            # Training step
            _fwd_bwd_result = None
            _optim_result = None
            fwd_bwd_future = None
            optim_step_future = None

            if datums:
                loss_fn_params = {
                    "eps_clip": config.eps_clip,
                    "eps_clip_high": config.eps_clip_high,
                }
                fwd_bwd_future = training_client.forward_backward(
                    datums, loss_fn="policy_loss", loss_fn_params=loss_fn_params
                )
                optim_step_future = training_client.optim_step(adam_params)
            else:
                logger.warning(f"No datums at step {global_step}, skipping training")

            # ── Await training results ──
            if fwd_bwd_future is not None:
                if generation_queue is not None:
                    # Pipeline mode: blocking .result() to avoid cross-event-loop deadlock
                    _fwd_bwd_result = fwd_bwd_future.result()
                    _optim_result = optim_step_future.result()
                else:
                    _fwd_bwd_result = await fwd_bwd_future
                    _optim_result = await optim_step_future

                # Pipeline mode: sync weights AFTER training completes
                if generation_queue is not None:
                    t_sync_start = time.time()
                    sync_result = training_client.sync_weights_to_inference(
                        sync_method=config.sync_method
                    ).result()
                    t_sync = time.time() - t_sync_start
                    metrics["time/weight_sync"] = t_sync
                    if sync_result.success:
                        metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                        metrics["sync/throughput_gbps"] = sync_result.throughput_gbps
                        logger.info(
                            f"{elapsed()} Step {global_step}: Pipeline weight sync "
                            f"{sync_result.total_bytes/1e9:.2f} GB in "
                            f"{t_sync:.2f}s ({sync_result.throughput_gbps:.2f} GB/s)"
                        )
                    else:
                        logger.warning(f"{elapsed()} Pipeline weight sync failed: {sync_result.message}")

            # Start pipeline generation thread after first step
            if generation_queue is None and global_step == start_step:
                generation_queue = queue_module.Queue(maxsize=1)
                stop_event = threading.Event()
                gen_thread = threading.Thread(
                    target=_generation_worker,
                    args=(
                        generation_queue,
                        stop_event,
                        eval_problems,
                        config,
                        sampling_clients,
                        sampling_params,
                        fewshot_problems,
                        renderer,
                        tokenizer,
                        system_prompt,
                        batches_per_epoch,
                    ),
                    kwargs={"start_batch_idx": 1},
                    daemon=True,
                )
                gen_thread.start()
                logger.info("Pipeline: started generation worker thread")

            # Print debug samples
            if gen_result.debug_samples:
                logger.info(f"Step {global_step} samples ({len(gen_result.debug_samples)}):")
                for i, (content, gt, fmt_valid, rwd) in enumerate(gen_result.debug_samples):
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    logger.info(
                        f"  Sample {i+1}: format={fmt_valid}, reward={rwd:.2f}, "
                        f"gt={gt}, response={content_preview!r}"
                    )

            # Log metrics
            time_total = time.time() - t_start
            metrics["time/total"] = time_total
            metrics["train/num_datums"] = len(datums)

            if _fwd_bwd_result and _fwd_bwd_result.metrics:
                for key, value in _fwd_bwd_result.metrics.items():
                    if key.startswith("is_kl_") or key.startswith("is_entropy"):
                        metrics[f"kl/{key[3:]}"] = value
                    elif key.startswith("is_ratio"):
                        metrics[f"ratio/{key[3:]}"] = value
                    elif key.startswith("is_pg_clipfrac"):
                        metrics["ppo/clipfrac"] = value
                    elif key.startswith("is_tis_"):
                        metrics[f"tis/{key[3:]}"] = value
                    elif key.startswith("is_valid_tokens"):
                        metrics["train/valid_tokens"] = value
                    elif key in ("execution_time", "execution_time:sum", "expert_load_summary"):
                        pass
                    else:
                        metrics[f"loss/{key}"] = value

                exec_key = "execution_time:sum" if "execution_time:sum" in _fwd_bwd_result.metrics else "execution_time"
                if exec_key in _fwd_bwd_result.metrics:
                    server_exec_time = _fwd_bwd_result.metrics[exec_key]
                    metrics["time/server_fwd_bwd"] = server_exec_time
                    metrics["time/network_overhead"] = time_total - server_exec_time - gen_result.t_sample - t_sync

                if "expert_load_summary" in _fwd_bwd_result.metrics:
                    expert_summary = _fwd_bwd_result.metrics["expert_load_summary"]
                    if "mean_imbalance_ratio" in expert_summary:
                        metrics["moe/mean_imbalance_ratio"] = expert_summary["mean_imbalance_ratio"]
                    if "max_imbalance_ratio" in expert_summary:
                        metrics["moe/max_imbalance_ratio"] = expert_summary["max_imbalance_ratio"]

            if _optim_result and _optim_result.metrics:
                for key, value in _optim_result.metrics.items():
                    metrics[f"optim/{key}"] = value

            ml_logger.log_metrics(metrics, step=global_step)
            global_step += 1

    # Cleanup
    if stop_event is not None:
        stop_event.set()
    if gen_thread is not None:
        gen_thread.join(timeout=30)

    # Save final checkpoint
    checkpoint_name = f"{model_id}-final"
    logger.info(f"Saving final checkpoint: {checkpoint_name}")
    save_result = training_client.save_full_weights_safetensors(name=checkpoint_name).result()
    logger.info(f"Final checkpoint saved to: {save_result.path} ({save_result.num_shards} shards)")

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":

    def _main(config: Config):
        asyncio.run(main(config))

    chz.nested_entrypoint(_main, allow_hyphens=True)
