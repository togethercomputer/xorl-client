# XoRL Python Client

Lightweight Python client for remote training on GPU clusters. Run training scripts from your laptop without PyTorch or CUDA.

## Why Use This?

- **Lightweight**: ~10 MB install (no PyTorch/CUDA)
- **Fast**: Installs in seconds
- **Laptop-friendly**: No GPU required
- **Same API**: Compatible with Tinker's API

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Set Up SSH Tunnel

```bash
ssh -L 5555:localhost:5555 your-gpu-server
```

### 2. Train & Sample

```python
import xorl_client

# Connect to server
client = xorl_client.ServiceClient(base_url="http://localhost:5555")

# Create training client
training = client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-3B-Instruct",
    rank=32
)

# Training step
datum = xorl_client.types.Datum(
    model_input=xorl_client.types.ModelInput.from_str("Prompt"),
    loss_fn_inputs={"target": "Expected output"}
)
fwd_result = training.forward_backward(data=[datum], loss_fn="cross_entropy").result()
optim_result = training.optim_step(
    adam_params=xorl_client.types.AdamParams(learning_rate=1e-5)
).result()

print(f"Loss: {fwd_result.loss_fn_outputs[0]['loss']}")

# Sample from model
sampler = training.save_weights_and_get_sampling_client(name="checkpoint")
response = sampler.sample(
    prompt="The capital of France is",
    sampling_params=xorl_client.types.SamplingParams(max_tokens=20)
)
print(f"Generated: {response.text}")
```

## Core API

### ServiceClient
```python
client = xorl_client.ServiceClient(base_url="http://localhost:5555")
training = client.create_lora_training_client(base_model="model", rank=32)
sampler = client.create_sampling_client(model_path="xorl://model/step-100")
```

### TrainingClient
```python
# Training operations
forward_backward(data, loss_fn)              # Forward + backward pass
optim_step(adam_params, gradient_clip)       # Optimizer step
save_weights_and_get_sampling_client(name)   # Save + get sampler
health_check()                               # Check server
```

### SamplingClient
```python
# Inference
sample(prompt, sampling_params, return_logprobs)
```

All methods return `APIFuture` objects. Call `.result()` to wait for completion.

## Advanced Features

### Serverless Inference

Use XoRL's serverless LoRA inference without managing infrastructure:

```python
from xorl_client.client.serverless_sampling_client import ServerlessSamplingClient

# After uploading weights with training_client.update_serverless_weights()
client = ServerlessSamplingClient(
    provider_api_key="your-api-key",
    model_id="your-org/adapter-name",
)

response = client.sample(
    prompt="Hello, how are you?",
    sampling_params=xorl_client.SamplingParams(max_tokens=50),
)
print(response.text)
```

### Dedicated Endpoints

Use XoRL's dedicated endpoints for production inference:

```python
from xorl_client.dedicated_endpoint import DedicatedEndpointSamplingClient

client = DedicatedEndpointSamplingClient(
    provider_api_key="your-api-key",
    endpoint_id="your-endpoint-id",
)

# After training_client.update_dedicated_endpoint()
client.set_model(result.provider_model_id)
result = client.sample("Hello!", sampling_params=xorl_client.SamplingParams(max_tokens=50))
```

## Troubleshooting

**Connection refused?** Check SSH tunnels:
```bash
lsof -i :5555 -sTCP:LISTEN
```

**Import error?** Install the package:
```bash
pip install -e .
```

## Optional Dependencies

Install extra features as needed:

```bash
# For serverless inference
pip install xorl-client[serverless]

# For dedicated endpoints
pip install xorl-client[dedicated-endpoint]

# For development
pip install xorl-client[dev]
```

## License

Apache 2.0
