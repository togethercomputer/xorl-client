import asyncio
import logging
import os
import re
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import Future

import chz
import datasets
import xorl_client
import torch
from xorl_client import ModelInput, types
from xorl_client.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log


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

SYSTEM_PROMPT = '''/no_think
Answer the provided question directly without providing any additional reasoning or output. Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format.
<answer>
...
</answer>
'''

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # API routing
    base_url: str = "http://localhost:6000"
    training_model: str = "sbharti/Qwen/Qwen3-32B-c7249570"
    inference_model: str = "sbharti/Qwen/Qwen3-32B-9ff2716a"
    api_key: str | None = None  # API key (or set XORL_API_KEY env var)
    log_path: str = "/scratch/outputs/checkpoints/qwen3-32b-gsm8k"
    ml_log_path: str = "./log_outputs/ml-logs/qwen3-32-gsm8k"
    model_name: str = "Qwen/Qwen3-32B"
    batch_size: int = 8
    group_size: int = 4
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 50
    val_every: int = 50
    max_tokens: int = 2048
    num_epochs: int = 5
    max_steps: int | None = None


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
        parsed_answer = extract_xml(response, '<answer>', '</answer>')
        if parsed_answer == answer:
            correctness_reward += 0.8

    return {
        "score": format_reward + correctness_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
    }


async def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.ml_log_path,
        wandb_project='tinker-xorl',
        wandb_name=f'gsm8k {config.model_name.split("/")[-1]}',
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    model_name_for_renderer = config.model_name
    if model_name_for_renderer.startswith("unsloth/Llama"):
        model_name_for_renderer = model_name_for_renderer.replace("unsloth/", "meta-llama/")
    renderer_name = model_info.get_recommended_renderer_name(model_name_for_renderer)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    N_train = 7473
    N_val = 512

    train_dataset = datasets.Dataset.from_list([
        {
            'question': example['question'],
            'answer': example['answer'].split('####')[-1].strip(),
        }
        for example in dataset["train"].select(range(N_train))
    ])
    train_dataset = datasets.concatenate_datasets([train_dataset] * config.num_epochs).shuffle()

    val_dataset = datasets.Dataset.from_list([
        {
            'question': example['question'],
            'answer': example['answer'].split('####')[-1].strip(),
        }
        for example in dataset["test"].select(range(N_val))
    ])

    convo_prefix = [{"role": "system", "content": SYSTEM_PROMPT}]

    n_train_batches = len(train_dataset) // config.batch_size
    n_val_batches = len(val_dataset) // config.batch_size

    # Setup training client
    api_key = config.api_key or os.environ.get("XORL_API_KEY")
    service_client = xorl_client.ServiceClient(base_url=config.base_url, model=config.training_model, api_key=api_key)
    sampling_params = xorl_client.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Generate unique run_id to avoid "adapter already loaded" errors across runs
    run_id = uuid.uuid4().hex[:8]

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}, run_id={run_id}")
    else:
        model_id = f"rl-{run_id}"
        logger.info(f"Creating training client with model_id={model_id}, run_id={run_id}")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank, model_id=model_id
        )
        start_batch = 0

    if config.max_steps is not None:
        n_train_batches = min(n_train_batches, start_batch + config.max_steps)

    logger.info(f"Training for {n_train_batches - start_batch} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Save weights for sampling
        sampling_path = (await training_client.save_weights_for_sampler(name=f"{run_id}-{step:06d}")).path
        sampling_client = service_client.create_sampling_client(
            base_url=config.base_url,
            model=config.inference_model,
            model_path=sampling_path,
            api_key=api_key,
        )

        # Validation (in batches of 32)
        if step % config.val_every == 0 and step > 0:
            val_metrics = defaultdict(list)
            VAL_BATCH_SIZE = 32

            # Collect all validation items
            val_items = [
                (example['question'], example['answer'])
                for example in val_dataset
            ]

            # Process in batches of 32
            for batch_start in range(0, len(val_items), VAL_BATCH_SIZE):
                batch_end = min(batch_start + VAL_BATCH_SIZE, len(val_items))
                batch_items = val_items[batch_start:batch_end]

                # Send all requests in this batch
                futures_and_data = []
                for question, answer in batch_items:
                    convo = [*convo_prefix, {"role": "user", "content": question}]
                    model_input = renderer.build_generation_prompt(convo, prefill='<answer>\n')
                    future = sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    futures_and_data.append((future, model_input, answer))

                # Await all in this batch
                reward_tasks = []
                for future, model_input, answer in futures_and_data:
                    try:
                        sample_result = await future
                        sampled_tokens = sample_result.sequences[0].tokens
                        parsed_message, _ = renderer.parse_response(sampled_tokens)
                        response_text = renderers.get_text_content(parsed_message)
                        prompt_tokens = model_input.to_ints()
                        reward_tasks.append(get_reward(prompt_tokens + sampled_tokens, response_text, answer))
                    except Exception as e:
                        logger.warning(f"Validation sample failed: {type(e).__name__}: {e}")

                rewards = await asyncio.gather(*reward_tasks)
                for reward in rewards:
                    for metric, value in reward.items():
                        val_metrics[metric].append(value)

            for metric, values in val_metrics.items():
                metrics[f"val/{metric}/mean"] = sum(values) / len(values)

        # Training
        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_inputs: list[ModelInput] = []
        batch_metrics = defaultdict(list)

        for question, answer in zip(batch_rows["question"], batch_rows["answer"]):
            convo = [*convo_prefix, {"role": "user", "content": question}]
            model_input = renderer.build_generation_prompt(convo, prefill='<answer>\n')
            prompt_tokens = model_input.to_ints()

            sample_futures = [
                sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
                for _ in range(config.group_size)
            ]
            batch_futures.append(sample_futures)
            batch_inputs.append(model_input)

        for sample_futures, model_input, answer in zip(batch_futures, batch_inputs, batch_rows["answer"]):
            reward_tasks = []
            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
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

                    parsed_message, _ = renderer.parse_response(sampled_tokens)
                    response_text = renderers.get_text_content(parsed_message)
                    reward_tasks.append(get_reward(prompt_tokens + sampled_tokens, response_text, answer))
                except Exception as e:
                    logger.warning(f"Training sample failed: {type(e).__name__}: {e}")
                    logger.warning(f"Full traceback:\n{traceback.format_exc()}")

            rewards = await asyncio.gather(*reward_tasks)
            for reward in rewards:
                group_rewards.append(reward["score"])
                for metric, value in reward.items():
                    batch_metrics[metric].append(value)

            if len(group_rewards) == 0:
                logger.warning("All samples failed for this group, skipping")
                continue

            advantages = [r - (sum(group_rewards) / len(group_rewards)) for r in group_rewards]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            if all(adv == 0.0 for adv in advantages):
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = [int(t) for t in tokens[:-1]]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages)

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step - skip if no valid datums (all groups had zero advantage or failed)
        if len(training_datums) == 0:
            logger.warning(f"Batch {batch_idx}: No valid training datums, skipping training step")
            metrics["time/total"] = time.time() - t_start
            for metric, values in batch_metrics.items():
                if values:
                    metrics[f"reward/{metric}/mean"] = sum(values) / len(values)
            ml_logger.log_metrics(metrics, step=batch_idx)
            continue

        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn="importance_sampling")
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = await fwd_bwd_future
        _optim_result = await optim_step_future

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        for metric, values in batch_metrics.items():
            metrics[f"reward/{metric}/mean"] = sum(values) / len(values)

        if _fwd_bwd_result.metrics:
            for key, value in _fwd_bwd_result.metrics.items():
                metrics[f"loss/{key}"] = value

        ml_logger.log_metrics(metrics, step=batch_idx)

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    def _main(config: Config):
        asyncio.run(main(config))
    chz.nested_entrypoint(_main)
