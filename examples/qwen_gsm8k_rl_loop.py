import asyncio
import json
import logging
import os
import queue as queue_module
import random
import re
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from zoneinfo import ZoneInfo

import chz
import datasets
import xorl_client
import torch
from xorl_client import types
from xorl_client.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tqdm import tqdm

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

SYSTEM_PROMPT = """/no_think
Answer the provided question directly without providing any additional reasoning or output. Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format.
<answer>
...
</answer>
"""

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # Server URL
    training_url: str = "http://localhost:6000"
    # Paths
    log_path: str = "/data/outputs"
    ml_log_path: str = "/data/outputs"
    # Model config
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    model_id: str | None = (
        None  # Unique model_id for parallel training (auto-generated if None)
    )
    # Training config
    batch_size: int = 8
    group_size: int = 4
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 50
    val_every: int = 50
    max_tokens: int = 2048
    num_epochs: int = 5
    max_steps: int | None = None

    # LR warmup
    min_lr: float = 1e-7
    warmup_steps: int = 25

    # Gradient clipping
    grad_clip_norm: float = 1.0

    # Loss function (PPO-style clipping)
    loss_fn: str = "importance_sampling"  # or "policy_loss"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.2

    # KL divergence monitoring
    compute_kl_stats: bool = True
    compute_ref_logprobs: bool = False
    dump_kl_diagnostics: bool = False

    # Sampling
    temperature: float = 1.0

    # Multi-server inference
    inference_base_urls: str = ""  # Comma-separated URLs or host IDs (e.g., "26,13")

    # Debugging
    debug_samples: int = 3
    dump_samples: bool = False
    synthetic_advantage: bool = False
    seed: int = 12345

    # Wandb
    wandb_name: str | None = None

    # Full-weights RL
    use_full_weights: bool = False
    inference_host: str = ""
    inference_port: int = 12345
    inference_world_size: int = 8
    master_port: int = 29741
    sync_method: str = "nccl_ep_scatter"
    return_routed_experts: bool = True
    pipeline_rl: bool = False

    # Resume / init
    resume_model_id: str | None = None
    resume_step: int = 0
    init_checkpoint: str | None = None
    save_step_0: bool = False


def extract_xml(text, opening_tag, closing_tag):
    content = text.split(opening_tag.strip())[-1]
    content = content.split(closing_tag.strip())[0]
    return content.strip()


async def get_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0

    format_pattern = r"\s*.*?\s*</answer>$"

    if re.match(format_pattern, response):
        format_reward += 0.2
        parsed_answer = extract_xml(response, "<answer>", "</answer>")
        if parsed_answer == answer:
            correctness_reward += 0.8

    return {
        "score": format_reward + correctness_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
    }


def get_inference_urls(config: Config) -> list[str]:
    """Parse inference URLs from config.

    Supports multiple inference servers for parallel sampling.
    Parses inference_base_urls as comma-separated URLs or host IDs.
    """
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


def setup_training_session(config, service_client, model_id, elapsed):
    """Set up training session. Returns (training_client, model_id, start_step).

    Safe to run in a background thread.
    """
    from xorl_client.client.training_client import TrainingClient

    logger.info(f"{elapsed()} [session] Checking resume...")

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if config.resume_model_id:
        # Resume from existing session
        model_id = config.resume_model_id
        start_step = config.resume_step
        logger.info(
            f"Resuming existing session: model_id={model_id}, start_step={start_step}"
        )

        response = service_client.holder.post_sync(
            "/api/v1/create_model",
            {
                "model_id": model_id,
                "base_model": config.model_name,
                "lora_config": {"rank": config.lora_rank},
            },
        )
        logger.info(f"Re-registered model_id: {response}")

        training_client = TrainingClient(
            holder=service_client.holder,
            model_id=model_id,
            base_model=config.model_name,
        )
    elif resume_info:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                resume_info["state_path"]
            )
        )
        start_step = resume_info["batch"]
        logger.info(f"Resuming from step {start_step}")
    elif config.use_full_weights:
        # Full-weights mode: create TrainingClient directly (no LoRA)
        logger.info(f"Creating full-weights training client with model_id={model_id}")

        # Kill existing sessions
        logger.info(
            f"{elapsed()} Querying session_info and killing existing sessions..."
        )
        try:
            session_info = service_client.holder.get_sync("/api/v1/session_info")
            registered_models = session_info.get("registered_models", [])
            if registered_models:
                logger.info(
                    f"{elapsed()} Found {len(registered_models)} existing session(s): {registered_models}"
                )
                for existing_model_id in registered_models:
                    try:
                        kill_resp = service_client.holder.post_sync(
                            "/api/v1/kill_session",
                            {
                                "model_id": existing_model_id,
                                "save_checkpoint": False,
                            },
                        )
                        logger.info(
                            f"{elapsed()} Killed session '{existing_model_id}': {kill_resp.get('message', 'OK')}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"{elapsed()} Failed to kill session '{existing_model_id}': {e}"
                        )
            else:
                logger.info(f"{elapsed()} No existing sessions to kill")
        except Exception as e:
            logger.warning(f"{elapsed()} Could not query session info: {e}")
        logger.info(f"{elapsed()} Session cleanup complete")

        logger.info(f"{elapsed()} Calling create_model...")
        response = service_client.holder.post_sync(
            "/api/v1/create_model",
            {
                "model_id": model_id,
                "base_model": config.model_name,
                "lora_config": {"rank": config.lora_rank},
            },
        )
        logger.info(f"{elapsed()} create_model complete: {response}")

        training_client = TrainingClient(
            holder=service_client.holder,
            model_id=model_id,
            base_model=config.model_name,
        )
        start_step = 0
    else:
        # LoRA mode
        logger.info(f"Creating training client with model_id={model_id}")
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name,
            rank=config.lora_rank,
            model_id=model_id,
        )
        start_step = 0

    logger.info(
        f"{elapsed()} [session] Training session setup complete: model_id={model_id}, start_step={start_step}"
    )
    return training_client, model_id, start_step


@dataclass
class GenerationBatchResult:
    datums: list[types.Datum]
    reward_metrics: dict[str, list[float]]
    sampling_time: float
    debug_samples: list[tuple[str, str, bool, float]]
    skipped_no_samples: int
    skipped_zero_advantage: int
    zero_adv_samples: list[dict]


async def _generate_training_batch(
    *,
    questions: list[str],
    answers: list[str],
    config: Config,
    sampling_clients: list,
    sampling_params,
    convo_prefix: list[dict[str, str]],
    renderer,
    step: int,
) -> GenerationBatchResult:
    training_datums: list[types.Datum] = []
    batch_metrics: defaultdict[str, list[float]] = defaultdict(list)
    debug_samples: list[tuple[str, str, bool, float]] = []
    problems_skipped_no_samples = 0
    problems_skipped_zero_advantage = 0
    zero_adv_samples: list[dict] = []

    batch_requests = []

    t_sample_start = time.time()
    for problem_idx, (question, answer) in enumerate(zip(questions, answers)):
        convo = [*convo_prefix, {"role": "user", "content": question}]
        model_input = renderer.build_generation_prompt(convo, prefill="<answer>\n")
        client = sampling_clients[problem_idx % len(sampling_clients)]
        sample_futures = [
            client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=replace(
                    sampling_params,
                    sampling_seed=random.randint(0, 2**63 - 1),
                ),
            )
            for _ in range(config.group_size)
        ]
        batch_requests.append((sample_futures, model_input, answer))

    for problem_idx, (sample_futures, model_input, answer) in enumerate(
        tqdm(batch_requests, total=len(batch_requests), desc=f"Sampling batch {step}")
    ):
        reward_tasks = []
        group_rewards: list[float] = []
        group_tokens: list[list[int]] = []
        group_logprobs: list[list[float]] = []
        group_ob_lens: list[int] = []
        group_routed_experts: list[list[list[list[int]]] | None] = []
        group_contents: list[dict[str, str]] = []
        prompt_tokens = model_input.to_ints()

        for future in sample_futures:
            try:
                sample_result = await future
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)
                routed_experts = (
                    sample_result.meta_info.get("routed_experts")
                    if sample_result.meta_info
                    else None
                )
                group_routed_experts.append(routed_experts)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                response_text = renderers.get_text_content(parsed_message)
                reward_tasks.append(
                    get_reward(prompt_tokens + sampled_tokens, response_text, answer)
                )
                group_contents.append({"content": response_text, "answer": answer})
            except Exception as e:
                logger.warning(f"Training sample failed: {type(e).__name__}: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        rewards = await asyncio.gather(*reward_tasks)
        for sample_idx, reward in enumerate(rewards):
            group_rewards.append(reward["score"])
            for metric, value in reward.items():
                batch_metrics[metric].append(value)

            if len(debug_samples) < config.debug_samples:
                debug_samples.append(
                    (
                        group_contents[sample_idx]["content"],
                        answer,
                        reward["format_reward"] > 0,
                        reward["score"],
                    )
                )

        if len(group_rewards) == 0:
            problems_skipped_no_samples += 1
            logger.warning("All samples failed for this group, skipping")
            continue

        mean_reward = sum(group_rewards) / len(group_rewards)
        if config.synthetic_advantage:
            advantages = [1.0 for _ in group_rewards]
        else:
            advantages = [r - mean_reward for r in group_rewards]

        if not config.synthetic_advantage and all(adv == 0.0 for adv in advantages):
            problems_skipped_zero_advantage += 1
            if config.dump_samples:
                zero_adv_samples.append(
                    {
                        "problem_idx": problem_idx,
                        "ground_truth": answer,
                        "mean_reward": mean_reward,
                        "num_samples": len(group_rewards),
                        "samples": group_contents,
                    }
                )
            continue

        for tokens, logprob, advantage, ob_len, routed_experts in zip(
            group_tokens,
            group_logprobs,
            advantages,
            group_ob_lens,
            group_routed_experts,
        ):
            input_tokens = [int(t) for t in tokens[:-1]]
            target_tokens = [-100] * ob_len + tokens[ob_len + 1 :]
            all_logprobs = [0.0] * ob_len + logprob
            all_advantages = [0.0] * ob_len + [advantage] * (
                len(input_tokens) - ob_len
            )

            assert (
                len(input_tokens)
                == len(target_tokens)
                == len(all_logprobs)
                == len(all_advantages)
            )

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                },
                routed_experts=routed_experts,
            )
            training_datums.append(datum)

    return GenerationBatchResult(
        datums=training_datums,
        reward_metrics=dict(batch_metrics),
        sampling_time=time.time() - t_sample_start,
        debug_samples=debug_samples,
        skipped_no_samples=problems_skipped_no_samples,
        skipped_zero_advantage=problems_skipped_zero_advantage,
        zero_adv_samples=zero_adv_samples,
    )


def _generation_worker(
    output_queue: queue_module.Queue,
    stop_event: threading.Event,
    train_dataset: datasets.Dataset,
    config: Config,
    sampling_clients: list,
    sampling_params,
    convo_prefix: list[dict[str, str]],
    renderer,
    start_batch_idx: int,
    end_batch_idx: int,
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for batch_idx in range(start_batch_idx, end_batch_idx):
            while output_queue.full() and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break

            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
            batch_rows = train_dataset.select(range(batch_start, batch_end))

            result = loop.run_until_complete(
                _generate_training_batch(
                    questions=list(batch_rows["question"]),
                    answers=list(batch_rows["answer"]),
                    config=config,
                    sampling_clients=sampling_clients,
                    sampling_params=sampling_params,
                    convo_prefix=convo_prefix,
                    renderer=renderer,
                    step=batch_idx,
                )
            )
            if stop_event.is_set():
                break

            output_queue.put(result)
    except Exception as e:
        logger.error(f"Generation worker failed: {type(e).__name__}: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        try:
            output_queue.put_nowait(None)
        except queue_module.Full:
            pass
        loop.close()


async def main(config: Config):
    _start_time = time.time()

    def elapsed():
        return f"[{time.time() - _start_time:.2f}s]"

    # Set random seed
    random.seed(config.seed)

    # Start tokenizer loading in background
    logger.info(f"{elapsed()} Starting tokenizer loading in background...")
    tokenizer_task = asyncio.create_task(
        asyncio.to_thread(get_tokenizer, config.model_name)
    )

    # Start service session setup in background
    logger.info(f"{elapsed()} Starting service session setup in background...")
    service_client = xorl_client.ServiceClient(base_url=config.training_url, timeout=1200.0)

    run_id = uuid.uuid4().hex[:8]
    model_id = config.model_id or f"rl-{run_id}"
    log_path = config.log_path

    session_task = asyncio.create_task(
        asyncio.to_thread(
            setup_training_session, config, service_client, model_id, elapsed
        )
    )

    # Setup logging (runs concurrently with background tasks)
    logger.info(f"{elapsed()} Starting ml_log.setup_logging...")
    wandb_run_name = config.wandb_name or (f'gsm8k {config.model_name.split("/")[-1]}')
    ml_logger = ml_log.setup_logging(
        log_dir=config.ml_log_path,
        wandb_project="tinker-xorl",
        wandb_name=wandb_run_name,
        config=config,
        do_configure_logging_module=True,
    )
    logger.info(f"{elapsed()} ml_log.setup_logging complete")
    apply_pst_formatters()

    # Wait for tokenizer
    logger.info(f"{elapsed()} Waiting for tokenizer (started in background)...")
    tokenizer = await tokenizer_task
    logger.info(f"{elapsed()} Tokenizer ready")

    # Get renderer
    model_name_for_renderer = config.model_name
    if model_name_for_renderer.startswith("unsloth/Llama"):
        model_name_for_renderer = model_name_for_renderer.replace(
            "unsloth/", "meta-llama/"
        )
    renderer_name = model_info.get_recommended_renderer_name(model_name_for_renderer)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"{elapsed()} Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info(f"{elapsed()} Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    N_train = 7473
    N_val = 512

    train_dataset = datasets.Dataset.from_list(
        [
            {
                "question": example["question"],
                "answer": example["answer"].split("####")[-1].strip(),
            }
            for example in dataset["train"].select(range(N_train))
        ]
    )
    train_dataset = datasets.concatenate_datasets(
        [train_dataset] * config.num_epochs
    ).shuffle()

    val_dataset = datasets.Dataset.from_list(
        [
            {
                "question": example["question"],
                "answer": example["answer"].split("####")[-1].strip(),
            }
            for example in dataset["test"].select(range(N_val))
        ]
    )

    convo_prefix = [{"role": "system", "content": SYSTEM_PROMPT}]

    n_train_batches = len(train_dataset) // config.batch_size

    # Wait for training session setup
    logger.info(
        f"{elapsed()} Waiting for training session setup (started in background)..."
    )
    training_client, model_id, start_step = await session_task
    logger.info(
        f"{elapsed()} Training session ready: model_id={model_id}, start_step={start_step}"
    )

    # Load initial weights from checkpoint if specified
    load_state_future = None
    if config.init_checkpoint:
        logger.info(
            f"{elapsed()} Starting load_state (non-blocking): {config.init_checkpoint}"
        )
        load_state_future = training_client.load_state(config.init_checkpoint)
        logger.info(f"{elapsed()} load_state request submitted (running in background)")

    # Full-weights mode: Setup inference endpoint verification
    if config.use_full_weights:
        logger.info(f"{elapsed()} Verifying inference endpoint is registered...")
        endpoints = training_client.list_inference_endpoints().result()
        if endpoints.count == 0:
            raise RuntimeError(
                "No inference endpoints registered. Please register via API before running:\n"
                f"  curl -X POST {config.training_url}/add_inference_endpoint -H 'Content-Type: application/json' \\\n"
                f'    -d \'{{"host": "{config.inference_host}", "port": {config.inference_port}, "world_size": {config.inference_world_size}}}\'\n'
                f"  curl -X POST {config.training_url}/connect_inference_endpoint -H 'Content-Type: application/json' \\\n"
                f"    -d '{{\"master_port\": {config.master_port}}}'"
            )
        logger.info(
            f"{elapsed()} Found {endpoints.count} registered inference endpoint(s)"
        )

        # Create sampling clients for all inference servers (reused across batches)
        inference_urls = get_inference_urls(config)
        sampling_clients = [xorl_client.SamplingClient(base_url=url) for url in inference_urls]
        logger.info(
            f"{elapsed()} Created {len(sampling_clients)} sampling client(s): {inference_urls}"
        )
    else:
        sampling_clients = None

    # Sampling parameters
    sampling_params = xorl_client.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
        return_routed_experts=config.return_routed_experts,
    )

    def get_lr(step: int) -> float:
        """Compute learning rate with linear warmup."""
        if step < config.warmup_steps:
            return config.min_lr + (config.learning_rate - config.min_lr) * (
                step / config.warmup_steps
            )
        return config.learning_rate

    # Wait for load_state if it was started earlier
    if load_state_future is not None:
        logger.info(f"{elapsed()} Waiting for load_state to complete...")
        load_result = load_state_future.result()
        logger.info(
            f"{elapsed()} load_state complete: loaded weights from {load_result.path}"
        )

    if config.pipeline_rl and not config.use_full_weights:
        raise ValueError("pipeline_rl requires use_full_weights=True")

    if config.max_steps is not None:
        n_train_batches = min(n_train_batches, start_step + config.max_steps)

    def sync_inference_weights(step: int, label: str):
        logger.info(f"{elapsed()} Starting sync_weights_to_inference...")
        t_sync_start = time.time()
        sync_result = training_client.sync_weights_to_inference(
            sync_method=config.sync_method
        ).result()
        t_sync = time.time() - t_sync_start

        if sync_result.success:
            logger.info(
                f"{elapsed()} Step {step}: {label} {sync_result.total_bytes/1e9:.2f} GB in "
                f"{t_sync:.2f}s ({sync_result.throughput_gbps:.2f} GB/s)"
            )
        else:
            logger.warning(
                f"{elapsed()} Step {step}: {label} failed: {sync_result.message}"
            )

        return t_sync, sync_result

    logger.info(f"{elapsed()} Training for {n_train_batches - start_step} batches")

    generation_queue: queue_module.Queue | None = None
    generation_stop_event: threading.Event | None = None
    generation_thread: threading.Thread | None = None

    # Main training loop
    for batch_idx in range(start_step, n_train_batches):
        t_start = time.time()
        step = batch_idx

        # Compute learning rate with warmup
        current_lr = get_lr(step)
        adam_params = xorl_client.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            grad_clip_norm=config.grad_clip_norm,
        )

        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": current_lr,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            if config.use_full_weights:
                checkpoint_name = f"{model_id}-{step:06d}"
                logger.info(f"Saving full-weights checkpoint: {checkpoint_name}")
                save_result = training_client.save_full_weights_safetensors(
                    name=checkpoint_name
                ).result()
                logger.info(
                    f"Checkpoint saved to: {save_result.path} ({save_result.num_shards} shards)"
                )
            else:
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{model_id}-{step:06d}",
                    log_path=log_path,
                    kind="state",
                    loop_state={"batch": batch_idx},
                )

        # Save step 0 checkpoint if requested
        if step == 0 and config.save_step_0:
            if config.use_full_weights:
                checkpoint_name = f"{model_id}-step0"
                logger.info(f"Saving checkpoint after first step: {checkpoint_name}")
                save_result = training_client.save_full_weights_safetensors(
                    name=checkpoint_name
                ).result()
                logger.info(
                    f"Step 0 checkpoint saved to: {save_result.path} ({save_result.num_shards} shards)"
                )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))
        batch_questions = list(batch_rows["question"])
        batch_answers = list(batch_rows["answer"])

        t_sync = 0.0
        prepared_sampling = False
        if generation_queue is None:
            if config.use_full_weights:
                sync_time, sync_result = sync_inference_weights(step, "Weight sync")
                t_sync += sync_time
                metrics["time/weight_sync"] = t_sync
                if sync_result.success:
                    metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                    metrics["sync/throughput_gbps"] = sync_result.throughput_gbps
            else:
                save_response = await training_client.save_weights_for_sampler(
                    name=f"{model_id}-{step:06d}"
                )
                sampling_path = save_response.model_path or save_response.path
                inference_urls = get_inference_urls(config)
                sampling_clients = [
                    service_client.create_sampling_client(
                        base_url=url,
                        model_path=sampling_path,
                    )
                    for url in inference_urls
                ]
            prepared_sampling = True

        # Validation (in batches of 32)
        if step % config.val_every == 0 and step > 0:
            val_metrics = defaultdict(list)
            VAL_BATCH_SIZE = 32

            val_items = [
                (example["question"], example["answer"]) for example in val_dataset
            ]

            for vb_start in range(0, len(val_items), VAL_BATCH_SIZE):
                vb_end = min(vb_start + VAL_BATCH_SIZE, len(val_items))
                batch_items = val_items[vb_start:vb_end]

                futures_and_data = []
                for i, (question, answer) in enumerate(batch_items):
                    convo = [*convo_prefix, {"role": "user", "content": question}]
                    model_input = renderer.build_generation_prompt(
                        convo, prefill="<answer>\n"
                    )
                    client_idx = (vb_start + i) % len(sampling_clients)
                    future = sampling_clients[client_idx].sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    futures_and_data.append((future, model_input, answer))

                reward_tasks = []
                for future, model_input, answer in futures_and_data:
                    try:
                        sample_result = await future
                        sampled_tokens = sample_result.sequences[0].tokens
                        parsed_message, _ = renderer.parse_response(sampled_tokens)
                        response_text = renderers.get_text_content(parsed_message)
                        prompt_tokens = model_input.to_ints()
                        reward_tasks.append(
                            get_reward(
                                prompt_tokens + sampled_tokens, response_text, answer
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Validation sample failed: {type(e).__name__}: {e}"
                        )

                rewards = await asyncio.gather(*reward_tasks)
                for reward in rewards:
                    for metric, value in reward.items():
                        val_metrics[metric].append(value)

            for metric, values in val_metrics.items():
                metrics[f"val/{metric}/mean"] = sum(values) / len(values)

        gen_result = None
        used_pipeline_batch = False
        if generation_queue is not None:
            queued_result = generation_queue.get()
            if queued_result is None:
                logger.warning(
                    "Generation worker finished early; falling back to synchronous sampling"
                )
                if generation_stop_event is not None:
                    generation_stop_event.set()
                generation_queue = None
                generation_stop_event = None
                generation_thread = None
            else:
                gen_result = queued_result
                used_pipeline_batch = True

        if gen_result is None:
            if not prepared_sampling:
                if config.use_full_weights:
                    sync_time, sync_result = sync_inference_weights(
                        step, "Weight sync"
                    )
                    t_sync += sync_time
                    metrics["time/weight_sync"] = t_sync
                    if sync_result.success:
                        metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                        metrics["sync/throughput_gbps"] = (
                            sync_result.throughput_gbps
                        )
                else:
                    save_response = await training_client.save_weights_for_sampler(
                        name=f"{model_id}-{step:06d}"
                    )
                    sampling_path = save_response.model_path or save_response.path
                    inference_urls = get_inference_urls(config)
                    sampling_clients = [
                        service_client.create_sampling_client(
                            base_url=url,
                            model_path=sampling_path,
                        )
                        for url in inference_urls
                    ]

            gen_result = await _generate_training_batch(
                questions=batch_questions,
                answers=batch_answers,
                config=config,
                sampling_clients=sampling_clients,
                sampling_params=sampling_params,
                convo_prefix=convo_prefix,
                renderer=renderer,
                step=step,
            )

        training_datums = gen_result.datums
        batch_metrics = gen_result.reward_metrics
        debug_sample_contents = gen_result.debug_samples
        problems_skipped_no_samples = gen_result.skipped_no_samples
        problems_skipped_zero_advantage = gen_result.skipped_zero_advantage
        zero_adv_samples = gen_result.zero_adv_samples
        metrics["time/sampling"] = gen_result.sampling_time

        # Training step - skip if no valid datums
        _fwd_bwd_result = None
        _optim_result = None
        fwd_bwd_future = None
        optim_step_future = None
        if len(training_datums) == 0:
            logger.warning(
                f"Batch {batch_idx}: No valid training datums, skipping training step. "
                f"Problems skipped: {problems_skipped_no_samples} (no samples), "
                f"{problems_skipped_zero_advantage} (zero advantage)"
            )
        else:
            loss_fn_params = {
                "compute_kl_stats": config.compute_kl_stats,
                "compute_ref_logprobs": config.compute_ref_logprobs,
                "dump_kl_diagnostics": config.dump_kl_diagnostics,
            }
            if config.loss_fn == "policy_loss":
                loss_fn_params["eps_clip"] = config.eps_clip
                loss_fn_params["eps_clip_high"] = config.eps_clip_high

            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn=config.loss_fn,
                loss_fn_params=loss_fn_params,
            )
            optim_step_future = training_client.optim_step(adam_params)
            if used_pipeline_batch:
                _fwd_bwd_result = fwd_bwd_future.result()
                _optim_result = optim_step_future.result()
            else:
                _fwd_bwd_result = await fwd_bwd_future
                _optim_result = await optim_step_future

        if config.pipeline_rl and config.use_full_weights and batch_idx + 1 < n_train_batches:
            if generation_queue is None:
                if len(training_datums) > 0:
                    sync_time, sync_result = sync_inference_weights(
                        step, "Pipeline weight sync"
                    )
                    t_sync += sync_time
                    metrics["time/weight_sync"] = t_sync
                    if sync_result.success:
                        metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                        metrics["sync/throughput_gbps"] = (
                            sync_result.throughput_gbps
                        )

                generation_queue = queue_module.Queue(maxsize=1)
                generation_stop_event = threading.Event()
                generation_thread = threading.Thread(
                    target=_generation_worker,
                    args=(
                        generation_queue,
                        generation_stop_event,
                        train_dataset,
                        config,
                        sampling_clients,
                        sampling_params,
                        convo_prefix,
                        renderer,
                        batch_idx + 1,
                        n_train_batches,
                    ),
                    daemon=True,
                )
                generation_thread.start()
                logger.info("Pipeline: started generation worker thread")
            elif used_pipeline_batch and len(training_datums) > 0:
                sync_time, sync_result = sync_inference_weights(
                    step, "Pipeline weight sync"
                )
                t_sync += sync_time
                metrics["time/weight_sync"] = t_sync
                if sync_result.success:
                    metrics["sync/bytes_gb"] = sync_result.total_bytes / 1e9
                    metrics["sync/throughput_gbps"] = sync_result.throughput_gbps

        # Log skip reason metrics
        metrics["skip/no_samples"] = problems_skipped_no_samples
        metrics["skip/zero_advantage"] = problems_skipped_zero_advantage

        # Print debug samples
        if debug_sample_contents:
            logger.info(
                f"Step {step} sample responses ({len(debug_sample_contents)} samples):"
            )
            for i, (content, gt, fmt_valid, rwd) in enumerate(debug_sample_contents):
                content_preview = (
                    content[:200] + "..." if len(content) > 200 else content
                )
                logger.info(
                    f"  Sample {i+1}: format_valid={fmt_valid}, reward={rwd:.2f}, "
                    f"ground_truth={gt}, response={content_preview!r}"
                )

        if config.dump_samples and zero_adv_samples:
            samples_dir = os.path.join(log_path, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            path = os.path.join(samples_dir, f"step_{step:06d}.jsonl")
            with open(path, "w") as f:
                for entry in zero_adv_samples:
                    f.write(json.dumps(entry) + "\n")
            logger.info(
                f"Dumped {len(zero_adv_samples)} zero-advantage problems to {path}"
            )

        # Log metrics
        time_total = time.time() - t_start
        metrics["time/total"] = time_total
        metrics["train/num_datums"] = len(training_datums)
        for metric, values in batch_metrics.items():
            if values:
                metrics[f"reward/{metric}/mean"] = sum(values) / len(values)

        # Log loss metrics from forward_backward (organized by category)
        if _fwd_bwd_result and _fwd_bwd_result.metrics:
            for key, value in _fwd_bwd_result.metrics.items():
                if key.startswith("is_kl_") or key.startswith("is_entropy"):
                    clean_key = key[3:]  # Remove "is_" prefix
                    metrics[f"kl/{clean_key}"] = value
                elif key.startswith("is_ratio"):
                    clean_key = key[3:]
                    metrics[f"ratio/{clean_key}"] = value
                elif key.startswith("is_pg_clipfrac"):
                    metrics["ppo/clipfrac"] = value
                elif key.startswith("is_tis_"):
                    clean_key = key[3:]
                    metrics[f"tis/{clean_key}"] = value
                elif key.startswith("is_valid_tokens"):
                    metrics["train/valid_tokens"] = value
                elif key == "execution_time":
                    pass  # Handle separately below
                elif key == "expert_load_summary":
                    pass  # Handle separately below
                else:
                    metrics[f"loss/{key}"] = value
            # Log server-side execution time breakdown
            if "execution_time" in _fwd_bwd_result.metrics:
                server_exec_time = _fwd_bwd_result.metrics["execution_time"]
                metrics["time/server_fwd_bwd"] = server_exec_time
                metrics["time/network_overhead"] = (
                    time_total - server_exec_time - gen_result.sampling_time - t_sync
                )
            # Log expert load imbalance metrics
            if "expert_load_summary" in _fwd_bwd_result.metrics:
                expert_summary = _fwd_bwd_result.metrics["expert_load_summary"]
                if "mean_imbalance_ratio" in expert_summary:
                    metrics["moe/mean_imbalance_ratio"] = expert_summary[
                        "mean_imbalance_ratio"
                    ]
                if "max_imbalance_ratio" in expert_summary:
                    metrics["moe/max_imbalance_ratio"] = expert_summary[
                        "max_imbalance_ratio"
                    ]

        # Log optim metrics (grad_norm, lr, etc.)
        if _optim_result and _optim_result.metrics:
            for key, value in _optim_result.metrics.items():
                metrics[f"optim/{key}"] = value

        ml_logger.log_metrics(metrics, step=batch_idx)

    if generation_stop_event is not None:
        generation_stop_event.set()
    if generation_thread is not None:
        generation_thread.join(timeout=5)

    # Save final checkpoint
    if config.use_full_weights:
        checkpoint_name = f"{model_id}-final"
        logger.info(f"Saving final full-weights checkpoint: {checkpoint_name}")
        save_result = training_client.save_full_weights_safetensors(
            name=checkpoint_name
        ).result()
        logger.info(
            f"Final checkpoint saved to: {save_result.path} ({save_result.num_shards} shards)"
        )
    else:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{model_id}-final",
            log_path=log_path,
            kind="both",
            loop_state={"batch": n_train_batches},
        )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":

    def _main(config: Config):
        asyncio.run(main(config))

    chz.nested_entrypoint(_main, allow_hyphens=True)
