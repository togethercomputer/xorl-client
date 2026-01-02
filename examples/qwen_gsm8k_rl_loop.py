import asyncio
import logging
import re
import time
import uuid
from collections import defaultdict
from functools import partial
from concurrent.futures import Future

import chz
import datasets
import numpy
import xorl_client
import torch
from xorl_client import ModelInput, SamplingClient, types
from xorl_client.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

DOUBLE_COT_SYSTEM_PROMPT = '''/no_think
Answer the provided questions after providing a few sentences of reasoning for each question. Your answer must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format:
<reasoning1>
...
</reasoning1>

<answer1>
...
</answer1>

<reasoning2>
...
</reasoning2>

<answer2>
...
</answer2>
'''

ENCODED_SYSTEM_PROMPT = '''/no_think
You will be given two questions and must answer both questions. You can provide a few sentences of reasoning but only pertaining to the first question. You are not allowed to discuss the second question in your reasoning. 

Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
<reasoning>
...
</reasoning>

<answer1>
...
</answer1>

<answer2>
...
</answer2>
'''

BLIND_SINGLE_SYSTEM_PROMPT = '''/no_think
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
    base_url: str | None = "http://localhost:5555"
    inference_base_url: str = "http://localhost:30000"
    log_path: str = "/scratch/outputs/checkpoints/qwen3-32b-gsm8k"
    model_name: str = "Qwen/Qwen3-32B"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 50
    val_every: int = 5
    max_tokens: int = 2048
    num_epochs: int = 5
    reward: str = "length_penalty"
    n: int = 5
    max_steps: int | None = None  # Limit number of training steps (None = train on full dataset)

def extract_xml(text, opening_tag, closing_tag):
    content = text.split(opening_tag.strip())[-1]
    content = content.split(closing_tag.strip())[0]
    return content.strip()

def find_last_subsequence(lst, sub):
    n, m = len(lst), len(sub)
    if m == 0 or m > n:
        return -1

    # iterate backwards so the first match we find is the last occurrence
    for i in range(n - m, -1, -1):
        if lst[i:i + m] == sub:
            return i
    return -1

def get_base_reward(response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0
    
    format_pattern = fr"<reasoning1>\s*.*?\s*</reasoning1>\s*<answer1>\s*.*?\s*</answer1>\s*<reasoning2>\s*.*?\s*</reasoning2>\s*<answer2>\s*.*?\s*</answer2>$"

    reasoning_trace_1 = str()
    reasoning_trace_2 = str()

    if re.search(format_pattern, response):
        format_reward += 0.2

        parsed_answer = extract_xml(response, '<answer2>', '</answer2>')
        reasoning_trace_1 = extract_xml(response, '<reasoning1>', '</reasoning1>')
        reasoning_trace_2 = extract_xml(response, '<reasoning2>', '</reasoning2>')

        if parsed_answer == answer:
            correctness_reward += 0.8
    return format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2

async def get_length_penalty_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    length_reward = 0

    if correctness_reward > 0:
        length_ratio = len(reasoning_trace_1) / (len(reasoning_trace_1) + len(reasoning_trace_2))
        length_reward += length_ratio * 0.5

    return {
        "score": format_reward + correctness_reward + length_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "length_reward": length_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }

async def get_length_penalty_q2_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    length_reward = 0

    if correctness_reward > 0:
        length_reward += 1 / (1 + 0.01 * len(reasoning_trace_2))

    return {
        "score": format_reward + correctness_reward + length_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "length_reward": length_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }

async def get_encoded_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0

    format_pattern = fr"<reasoning>\s*.*?\s*</reasoning>\s*<answer1>\s*.*?\s*</answer1>\s*<answer2>\s*.*?\s*</answer2>$"

    if re.search(format_pattern, response):
        format_reward += 0.2

        parsed_answer = extract_xml(response, '<answer2>', '</answer2>')

        if parsed_answer == answer:
            correctness_reward += 0.8

    return {
        "score": format_reward + correctness_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
    }

async def get_blind_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0

    format_pattern = fr"\s*.*?\s*</answer2>$"

    if re.match(format_pattern, response):
        format_reward += 0.2

        parsed_answer = extract_xml(response, '<answer2>', '</answer2>')

        if parsed_answer == answer:
            correctness_reward += 0.8

    return {
        "score": format_reward + correctness_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
    }

async def get_blind_single_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0

    format_pattern = fr"\s*.*?\s*</answer>$"

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

async def get_logprob_reward(base_sampling_client: SamplingClient, tokenizer, tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    logprob_reward = 0

    if correctness_reward > 0:
        base_logprobs = await base_sampling_client.compute_logprobs_async(ModelInput.from_ints(tokens))

        # we know the answer is correct so just look for it
        tokenized_answer = tokenizer.encode(answer)
        answer_start = find_last_subsequence(tokens, tokenized_answer)
        answer_logprobs = base_logprobs[answer_start:answer_start + len(tokenized_answer)]
        mean_logprob = sum(answer_logprobs) / max(len(answer_logprobs), 1)
        logprob_reward -= 0.5 * numpy.exp(mean_logprob)

    return {
        "score": format_reward + correctness_reward + logprob_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "logprob_reward": logprob_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }

async def get_attn_reward(sampling_client: SamplingClient, tokenizer, tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    attn_reward = 0

    if correctness_reward > 0:
        orig_logprobs = await sampling_client.compute_logprobs_async(ModelInput.from_ints(tokens))

        # we know formatted
        # find whats between reasoning1 and /reasoning1
        # cut that out (replace with tokenize(...))

        reasoning1_start_pos = max(find_last_subsequence(tokens, tokenizer.encode(pattern)) + len(tokenizer.encode(pattern)) for pattern in ['<reasoning1', '\n<reasoning1', '.<reasoning1']) + 1
        reasoning1_end_pos = max(find_last_subsequence(tokens, tokenizer.encode(pattern)) for pattern in ['</reasoning1', '\n</reasoning1', '.</reasoning1'])
        spliced_tokens = tokens[:reasoning1_start_pos] + tokenizer.encode('...\n') + tokens[reasoning1_end_pos:]
        masked_logprobs = await sampling_client.compute_logprobs_async(ModelInput.from_ints(spliced_tokens))

        # we know the answer is correct so just look for it
        tokenized_answer = tokenizer.encode(answer)

        orig_answer_start = find_last_subsequence(tokens, tokenized_answer)
        orig_answer_logprobs = orig_logprobs[orig_answer_start:orig_answer_start + len(tokenized_answer)]
        orig_mean_logprob = sum(orig_answer_logprobs) / max(len(orig_answer_logprobs), 1)

        masked_answer_start = find_last_subsequence(spliced_tokens, tokenized_answer)
        masked_answer_logprobs = masked_logprobs[masked_answer_start:masked_answer_start + len(tokenized_answer)]
        masked_mean_logprob = sum(masked_answer_logprobs) / max(len(masked_answer_logprobs), 1)

        # positive if the model did better seeing the q1 trace
        logprob_diff = orig_mean_logprob - masked_mean_logprob

        alpha = 2
        logprob_diff = min(alpha, logprob_diff)
        attn_reward += (numpy.exp(logprob_diff) - 1) / (alpha - 1)


    return {
        "score": format_reward + correctness_reward + attn_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "attn_reward": attn_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }


async def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project='tinker-xorl',
        wandb_name=f'n={config.n} ' if config.n > 0 else 'gsm8k ' + config.reward + ' ' + config.model_name.split('/')[-1],
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    # Map model name to canonical form for renderer lookup
    # e.g., "unsloth/Llama-3.2-1B-Instruct" -> "meta-llama/Llama-3.2-1B-Instruct"
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
    train_dataset = datasets.Dataset.from_list(
    [
        {
            'question': example2['question'] if config.reward == 'blind_single' else f'''Question 1:\n{example1['question']}\n\nQuestion 2:\n{example2['question']}''',
            'answer': example1['answer'].split('####')[-1].strip() + '\n' + example2['answer'].split('####')[-1].strip(),
        }
        for example1, example2 in zip(
            dataset["train"].select([i%N_train for i in range(0, N_train*2, 2)]), 
            dataset["train"].select([i%N_train for i in range(1, N_train*2, 2)])
        )
    ])
    train_dataset = datasets.concatenate_datasets([train_dataset] * config.num_epochs).shuffle()
    val_dataset = datasets.Dataset.from_list(
    [
        {
            'question': example2['question'] if config.reward == 'blind_single' else f'''Question 1:\n{example1['question']}\n\nQuestion 2:\n{example2['question']}''',
            'answer': example1['answer'].split('####')[-1].strip() + '\n' + example2['answer'].split('####')[-1].strip(),
        }
        for example1, example2 in zip(
            dataset["test"].select([i%N_val for i in range(0, N_val*2, 2)]), 
            dataset["test"].select([i%N_val for i in range(1, N_val*2, 2)])
        )
    ])
    if config.n > 0:
        import random
        random.seed(12345)
        N_train = 5000
        N_val = 500
        n = config.n

        train_samples = []
        val_samples = []

        for _ in range(N_train * config.num_epochs):
            a1, b1 = random.randint(10 ** (n-1), 10 ** n - 1), random.randint(10 ** (n-1), 10 ** n - 1)
            a2, b2 = random.randint(10 ** (n-1), 10 ** n - 1), random.randint(10 ** (n-1), 10 ** n - 1)
            train_samples.append({
                # 'question': f'What is {a2} times {b2}?' if config.reward == 'blind_single' else f'''Question 1:\nWhat is {a1} times {b1}?\n\nQuestion 2:\nWhat is {a2} times {b2}?''',
                'question': f'x={a2} and y={b2}. What is their product?' if config.reward == 'blind_single' else f'''Question 1:\nWhat is {a1} times {b1}?\n\nQuestion 2:\nWhat is {a2} times {b2}?''',
                'answer': f'{a1*b1}\n{a2*b2}',
            })

        for _ in range(N_val):
            a1, b1 = random.randint(10 ** (n-1), 10 ** n - 1), random.randint(10 ** (n-1), 10 ** n - 1)
            a2, b2 = random.randint(10 ** (n-1), 10 ** n - 1), random.randint(10 ** (n-1), 10 ** n - 1)
            val_samples.append({
                # 'question': f'What is {a2} times {b2}?' if config.reward == 'blind_single' else f'''Question 1:\nWhat is {a1} times {b1}?\n\nQuestion 2:\nWhat is {a2} times {b2}?''',
                'question': f'x={a2} and y={b2}. What is their product?' if config.reward == 'blind_single' else f'''Question 1:\nWhat is {a1} times {b1}?\n\nQuestion 2:\nWhat is {a2} times {b2}?''',
                'answer': f'{a1*b1}\n{a2*b2}',
            })

        train_dataset = datasets.Dataset.from_list(train_samples)
        val_dataset = datasets.Dataset.from_list(val_samples)

    convo_prefix = [
        {
            "role": "system",
            "content": ENCODED_SYSTEM_PROMPT if config.reward == 'encoded' else BLIND_SINGLE_SYSTEM_PROMPT if config.reward == 'blind_single' else DOUBLE_COT_SYSTEM_PROMPT,
        }
    ]

    n_train_batches = len(train_dataset) // config.batch_size
    n_val_batches = len(val_dataset) // config.batch_size

    # Setup training client
    service_client = xorl_client.ServiceClient(base_url=config.base_url)
    sampling_params = xorl_client.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
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

    # Limit number of batches if max_steps is set
    if config.max_steps is not None:
        n_train_batches = min(n_train_batches, start_batch + config.max_steps)

    logger.info(f"Training for {n_train_batches - start_batch} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
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

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Include run_id in name to avoid collisions across runs
        sampling_path = (await training_client.save_weights_for_sampler(name=f"{run_id}-{step:06d}")).path
        sampling_client = service_client.create_sampling_client(
            base_url=config.inference_base_url,
            model_path=sampling_path
        )
        # Set up sampling parameters

        if config.reward == "length_penalty":
            get_reward = get_length_penalty_reward
        elif config.reward == "length_penalty_q2":
            get_reward = get_length_penalty_q2_reward
        elif config.reward == "logprob":
            # Note: This creates a separate sampling client for the base model
            # You may need to set up a separate inference server for the base model
            base_sampling_client = service_client.create_sampling_client(
                base_model=config.model_name,
                inference_base_url=config.inference_base_url
            )
            get_reward = partial(get_logprob_reward, base_sampling_client, tokenizer)
        elif config.reward == "attn":
            get_reward = partial(get_attn_reward, sampling_client, tokenizer)
        elif config.reward == "encoded":
            get_reward = get_encoded_reward
        elif config.reward == "blind":
            get_reward = get_blind_reward
        elif config.reward == "blind_single":
            get_reward = get_blind_single_reward
        else:
            raise NotImplementedError(config.reward)

        if step % config.val_every == 0:
            val_metrics = defaultdict(list)

            reward_tasks = []
            for val_batch_idx in range(n_val_batches):
                val_batch_start = val_batch_idx * config.batch_size
                val_batch_end = min((val_batch_idx + 1) * config.batch_size, len(val_dataset))
                val_batch_rows = val_dataset.select(range(val_batch_start, val_batch_end))

                val_batch_futures: list[list[Future[types.SampleResponse]]] = []
                val_batch_inputs: list[ModelInput] = []
                for question, answers in zip(val_batch_rows["question"], val_batch_rows["answer"]):
                    convo = [
                        *convo_prefix,
                        {"role": "user", "content": question},
                    ]
                    answer1, answer = answers.split('\n')
                    prefill = f'<reasoning1>\n...\n</reasoning1>\n\n<answer1>\n{answer1}\n</answer1>\n\n<reasoning2>\n...\n</reasoning2>\n\n<answer2>\n' if config.reward == "blind" else '<answer>\n' if config.reward == "blind_single" else None
                    model_input = renderer.build_generation_prompt(convo, prefill=prefill)
                    prompt_tokens = model_input.to_ints()

                    # Generate response
                    sample_futures: list[Future[types.SampleResponse]] = []
                    sample_futures.append(
                        sampling_client.sample(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )

                    val_batch_futures.append(sample_futures)
                    val_batch_inputs.append(model_input)

                for sample_futures, model_input, answers in zip(
                    val_batch_futures, val_batch_inputs, val_batch_rows["answer"]
                ):
                    answer1, answer = answers.split('\n')
                    prompt_tokens = model_input.to_ints()
                    for future in sample_futures:
                        try:
                            sample_result = await future
                            sampled_tokens = sample_result.sequences[0].tokens
                            parsed_message, _ = renderer.parse_response(sampled_tokens)
                            response_text = renderers.get_text_content(parsed_message)

                            # Removed excessive logging
                            # if val_batch_idx == 0:
                            #     logger.info(tokenizer.decode(prompt_tokens) + response_text)
                            reward_tasks.append(get_reward(prompt_tokens + sampled_tokens, response_text, answer))
                        except Exception as e:
                            # Skip samples that fail - just drop them from the batch
                            logger.warning(f"Validation sample failed, dropping from batch: {type(e).__name__}")
                            continue


            rewards = await asyncio.gather(*reward_tasks)
            for reward in rewards:
                for metric, value in reward.items():
                    val_metrics[metric].append(value)

            for metric, values in val_metrics.items():
                metrics[f"val/{metric}/mean"] = sum(values) / len(values)

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_inputs: list[ModelInput] = []
        batch_metrics = defaultdict(list)
        for question, answers in zip(batch_rows["question"], batch_rows["answer"]):
            convo = [
                *convo_prefix,
                {"role": "user", "content": question},
            ]
            answer1, answer = answers.split('\n')
            prefill = f'<reasoning1>\n...\n</reasoning1>\n\n<answer1>\n{answer1}\n</answer1>\n\n<reasoning2>\n...\n</reasoning2>\n\n<answer2>\n' if config.reward == "blind" else '<answer>\n' if config.reward == "blind_single" else None
            model_input = renderer.build_generation_prompt(convo, prefill=prefill)
            prompt_tokens = model_input.to_ints()

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_inputs.append(model_input)

        for sample_futures, model_input, answers in zip(
            batch_futures, batch_inputs, batch_rows["answer"]
        ):
            answer1, answer = answers.split('\n')
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
                    # Skip samples that fail - just drop them from the batch
                    logger.warning(f"Training sample failed, dropping from batch: {type(e).__name__}")
                    continue

            rewards = await asyncio.gather(*reward_tasks)
            for reward in rewards:
                group_rewards.append(reward["score"])
                for metric, value in reward.items():
                    batch_metrics[metric].append(value)

            # Skip this group if all samples failed (empty group)
            if len(group_rewards) == 0:
                logger.warning(f"All samples failed for this group, skipping")
                continue

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = await fwd_bwd_future
        _optim_result = await optim_step_future

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        for metric, values in batch_metrics.items():
            metrics[f"reward/{metric}/mean"] = sum(values) / len(values)

        # Add KL divergence and importance sampling metrics from loss function
        if _fwd_bwd_result.metrics:
            for key, value in _fwd_bwd_result.metrics.items():
                metrics[f"loss/{key}"] = value

        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    #await checkpoint_utils.save_checkpoint_async(
    #    training_client=training_client,
    #    name="final",
    #    log_path=config.log_path,
    #    kind="both",
    #    loop_state={"batch": n_train_batches},
    #)
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    def _main(config: Config):
        asyncio.run(main(config))
    chz.nested_entrypoint(_main)
