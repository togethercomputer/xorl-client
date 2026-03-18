# XoRL

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
import xorl_client
from xorl_client import types

# Connect to server
service_client = xorl_client.ServiceClient(base_url="http://localhost:5000")

# Create training client
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507"
)

# Get tokenizer
tokenizer = training_client.get_tokenizer()

# Prepare training data
prompt = "English: hello world\nPig Latin:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
prompt_weights = [0] * len(prompt_tokens)

completion_tokens = tokenizer.encode(" ello-hay orld-way\n\n", add_special_tokens=False)
completion_weights = [1] * len(completion_tokens)

tokens = prompt_tokens + completion_tokens
weights = prompt_weights + completion_weights

input_tokens = tokens[:-1]
target_tokens = tokens[1:]
weights = weights[1:]

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
)

# Training step
fwdbwd_result = training_client.forward_backward([datum], "cross_entropy").result()
optim_result = training_client.optim_step(
    types.AdamParams(learning_rate=1e-4)
).result()

print(f"Grad norm: {optim_result.metrics.get('grad_norm', 0.0):.4f}")

# Save checkpoint
training_client.save_state("checkpoint_001").result()
```

## Checkpoint Management

```python
# Create REST client for checkpoint operations
rest_client = service_client.create_rest_client()

# List all checkpoints
checkpoints = rest_client.list_checkpoints().result()
for cp in checkpoints.checkpoints:
    print(f"{cp.checkpoint_id}: {cp.path}")

# Delete a checkpoint (by ID or xorl:// path)
rest_client.delete_checkpoint("weights/checkpoint_001").result()
rest_client.delete_checkpoint("xorl://default/weights/checkpoint_001").result()
```

## License

```
Copyright 2025 XoRL Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
