#!/usr/bin/env python3
"""
XoRL Training Example Script

This script demonstrates how to use the xorl_client client library to train a model
on a XoRL training server. It trains a simple English to Pig Latin translation task.

Usage:
    python simple_sft.py --server-url http://localhost:5001 --base-model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import numpy as np

import xorl_client
from xorl_client import types


def process_example(example: dict, tokenizer) -> types.Datum:
    """
    Process a single training example into a Datum for the loss function.

    Args:
        example: Dictionary with 'input' and 'output' keys
        tokenizer: The tokenizer from the training client

    Returns:
        A Datum object with model_input and loss_fn_inputs
    """
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]  # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def visualize_example(datum: types.Datum, tokenizer) -> None:
    """Print a visualization of a training example showing input, target, and weight."""
    print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
    print("-" * 50)
    for inp, tgt, wgt in zip(
        datum.model_input.to_ints(),
        datum.loss_fn_inputs['target_tokens'].tolist(),
        datum.loss_fn_inputs['weights'].tolist()
    ):
        print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")


def main():
    parser = argparse.ArgumentParser(description="XoRL Training Example")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:6000",
        help="URL of the XoRL training server"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=6,
        help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the first training example"
    )
    args = parser.parse_args()

    # Create training client
    print(f"Connecting to server: {args.server_url}")
    print(f"Base model: {args.base_model}")
    training_client = xorl_client.ServiceClient(
        base_url=args.server_url
    ).create_lora_training_client(base_model=args.base_model)

    # Create some training examples (English to Pig Latin)
    examples = [
        {"input": "banana split", "output": "anana-bay plit-say"},
        {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
        {"input": "donut shop", "output": "onut-day op-shay"},
        {"input": "pickle jar", "output": "ickle-pay ar-jay"},
        {"input": "space exploration", "output": "ace-spay exploration-way"},
        {"input": "rubber duck", "output": "ubber-ray uck-day"},
        {"input": "coding wizard", "output": "oding-cay izard-way"},
    ]

    # Get the tokenizer from the training client
    tokenizer = training_client.get_tokenizer()

    # Process all examples
    processed_examples = [process_example(ex, tokenizer) for ex in examples]
    print(f"\nProcessed {len(processed_examples)} training examples")

    # Optionally visualize the first example
    if args.visualize:
        print("\nFirst example visualization:")
        visualize_example(processed_examples[0], tokenizer)
        print()

    # Training loop
    print(f"\nStarting training loop ({args.num_steps} steps)")
    print("-" * 60)

    for step in range(args.num_steps):
        # Run forward-backward pass
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")

        # Run optimizer step
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=args.learning_rate))

        # Wait for the results
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        print("Forward-backward result:")
        print(fwdbwd_result)

        print("Optimizer result:")
        print(optim_result)

        # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
        # average log loss per token.
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
        loss_per_token = -np.dot(logprobs, weights) / weights.sum()

        # Get optimizer metrics
        grad_norm = optim_result.metrics.get('grad_norm', 0.0)

        print(f"Step {step + 1}: Loss per token = {loss_per_token:.4f}, Grad Norm = {grad_norm:.4f}")

    print("-" * 60)
    print("Training completed!")


if __name__ == "__main__":
    main()
