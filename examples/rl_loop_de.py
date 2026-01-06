"""
RL training loop for provider AI dedicated endpoints.

Uses GRPO-style reward centering with API routing.
"""

import logging
import os
import time
import uuid
from concurrent.futures import Future

import datasets
import torch

import xorl_client
from xorl_client import types
from tqdm import tqdm
from xorl_client.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
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

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# Hardcoded config
BASE_URL = "http://localhost:6000"
TRAINING_MODEL = "sbharti/Qwen/Qwen3-32B-c7249570"
INFERENCE_MODEL = "sbharti/Qwen/Qwen3-32B-9ff2716a"
MODEL_NAME = "Qwen/Qwen3-32B"
LOG_PATH = "outputs/xorl_client-rl-de"
BATCH_SIZE = 8
GROUP_SIZE = 4
LEARNING_RATE = 4e-5
LORA_RANK = 32
SAVE_EVERY = 50
MAX_TOKENS = 32000


def get_reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def main():
    api_key = os.environ.get("XORL_API_KEY")
    if not api_key:
        raise ValueError("XORL_API_KEY environment variable must be set")

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=LOG_PATH,
        wandb_project=None,
        wandb_name=None,
        config={
            "training_model": TRAINING_MODEL,
            "inference_model": INFERENCE_MODEL,
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
        },
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = "qwen3_disable_thinking"
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    convo_prefix = [
        {
            "role": "user",
            "content": "How many r's are in strawberry?" + question_suffix,
        },
        {
            "role": "assistant",
            "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
        },
    ]

    n_train_batches = len(train_dataset) // BATCH_SIZE

    # Setup training client with API routing
    service_client = xorl_client.ServiceClient(base_url=BASE_URL, model=TRAINING_MODEL, api_key=api_key)

    resume_info = checkpoint_utils.get_last_checkpoint(LOG_PATH)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_NAME, rank=LORA_RANK
        )
        start_batch = 0

    sampling_params = xorl_client.types.SamplingParams(
        max_tokens=MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Generate unique run_id to avoid "adapter already loaded" errors across runs
    run_id = uuid.uuid4().hex[:8]

    logger.info(f"Training for {n_train_batches} batches, run_id={run_id}")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": LEARNING_RATE,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if SAVE_EVERY > 0 and batch_idx % SAVE_EVERY == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=LOG_PATH,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Save weights and create sampling client with API routing
        sampling_path = training_client.save_weights_for_sampler(name=f"{run_id}-{batch_idx:06d}").result().path
        sampling_client = service_client.create_sampling_client(
            base_url=BASE_URL,
            model=INFERENCE_MODEL,
            model_path=sampling_path,
            api_key=api_key,
        )

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[list[int]] = []

        for question in batch_rows["question"]:
            convo = [
                *convo_prefix,
                {"role": "user", "content": question + question_suffix},
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=GROUP_SIZE,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(prompt_tokens)

        for future, prompt_tokens, answer in tqdm(
            zip(futures_P, prompts_P, batch_rows["answer"]),
            total=len(futures_P),
            desc=f"Sampling batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            ob_lens_G: list[int] = []

            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                tokens_G_T.append(all_tokens)
                ob_lens_G.append(len(prompt_tokens) - 1)
                logprobs_G_T.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)
                reward = get_reward(content, answer)
                rewards_G.append(reward)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [reward - mean_reward for reward in rewards_G]
            rewards_P.append(mean_reward)

            if all(advantage == 0.0 for advantage in advantages_G):
                continue

            for tokens, logprobs, advantage, ob_len in zip(
                tokens_G_T, logprobs_G_T, advantages_G, ob_lens_G
            ):
                input_tokens = [int(t) for t in tokens[:-1]]
                target_tokens = tokens[1:]
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

                assert len(input_tokens) == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # Training step (skip if no datums - all advantages were 0)
        if len(datums_D) == 0:
            logger.warning(f"Batch {batch_idx}: No training datums (all advantages were 0), skipping training step")
        else:
            fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
            optim_step_future = training_client.optim_step(adam_params)
            _fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_step_future.result()

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        ml_logger.log_metrics(metrics, step=batch_idx)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=LOG_PATH,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    main()
