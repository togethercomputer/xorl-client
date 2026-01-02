"""
TrainingClient - Tinker-compatible training client with futures.

This is the new version that uses ClientHolder and returns APIFutures.

Request Ordering:
================
Each request gets a sequential `seq_id` that the server uses to enforce
execution order via SeqIdAwareFIFOPolicy. The client dispatches requests
immediately (non-blocking) and the server handles ordering.

This prevents race conditions like optim_step executing before forward_backward.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union

from xorl_client import types
from xorl_client.client.api_future import APIFuture, wrap_future
from xorl_client.client.client_holder import ClientHolder

logger = logging.getLogger(__name__)


class TrainingClient:
    """Client for training ML models with forward/backward passes and optimization.

    The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
    You typically get one by calling `service_client.create_lora_training_client()`.

    Key methods:
    - forward_backward() - compute gradients for training (returns APIFuture)
    - optim_step() - update model parameters with Adam optimizer (returns APIFuture)
    - save_weights_for_sampler() - save weights for inference (returns APIFuture)
    - save_weights_and_get_sampling_client() - atomic save + create sampling client
    - save_state() - save full checkpoint including optimizer state
    - get_tokenizer() - get the tokenizer for the base model

    Args:
        holder: ClientHolder managing HTTP connections and async operations
        model_id: Unique identifier for the model to train
        base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")

    Example:
        >>> service_client = ServiceClient(base_url="http://localhost:5555")
        >>> training_client = service_client.create_lora_training_client(
        ...     base_model="Qwen/Qwen2.5-3B-Instruct",
        ...     rank=32
        ... )
        >>>
        >>> # Get tokenizer for the base model
        >>> tokenizer = training_client.get_tokenizer()
        >>>
        >>> # Pipeline operations (submit both immediately)
        >>> fwd_bwd_future = training_client.forward_backward(datums, "importance_sampling")
        >>> optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-5))
        >>>
        >>> # Wait for both to complete
        >>> fwd_bwd_result = fwd_bwd_future.result()
        >>> optim_result = optim_future.result()
    """

    def __init__(self, holder: ClientHolder, model_id: str, base_model: str):
        """Initialize TrainingClient.

        Args:
            holder: ClientHolder for connection management
            model_id: Model ID
            base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        """
        self.holder = holder
        self.model_id = model_id
        self.base_model = base_model
        self._tokenizer = None  # Lazy-loaded tokenizer

        # Request ordering mechanism (similar to Tinker)
        # _take_turn ensures HTTP requests are dispatched in seq_id order
        # This prevents race conditions where optim_step arrives before forward_backward
        self._request_id_lock = threading.Lock()
        self._request_id_counter = 0
        self._turn_counter = 0
        self._turn_waiters: Dict[int, asyncio.Event] = {}

        logger.info(f"TrainingClient initialized: model_id={model_id}, base_model={base_model}")

    def _get_request_id(self) -> int:
        """Get the next request ID (thread-safe).

        Returns:
            Sequential request ID starting from 0
        """
        with self._request_id_lock:
            request_id = self._request_id_counter
            self._request_id_counter += 1
            return request_id

    @asynccontextmanager
    async def _take_turn(self, request_id: int):
        """Wait for turn to dispatch HTTP request in seq_id order.

        This ensures that even if multiple requests are made concurrently,
        they are dispatched to the server in the correct order.

        Args:
            request_id: The request ID to wait for

        Example:
            async with self._take_turn(request_id):
                # HTTP request is dispatched here
                future = self.holder.post_async("/api/v1/forward_backward", data)
        """
        assert self._turn_counter <= request_id, f"Same request id cannot be taken twice: turn_counter={self._turn_counter}, request_id={request_id}"

        # Wait if it's not our turn yet
        if self._turn_counter < request_id:
            try:
                event = asyncio.Event()
                self._turn_waiters[request_id] = event
                await event.wait()
            finally:
                del self._turn_waiters[request_id]

        assert self._turn_counter == request_id

        try:
            yield
        finally:
            # Release turn and wake up next waiter
            self._turn_counter += 1
            if self._turn_counter in self._turn_waiters:
                self._turn_waiters[self._turn_counter].set()

    def _convert_tinker_datum(self, datum_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tinker.Datum format to xorl_client format.

        tinker format:
            model_input: {"chunks": [{"tokens": [...], "type": "encoded_text"}]}
            loss_fn_inputs: {"weights": {"data": [...], "dtype": "...", "shape": [...]}}

        xorl_client format:
            model_input: {"input_ids": [...]}
            loss_fn_inputs: {"weights": {"data": [...], "dtype": "...", "shape": [...]}}
        """
        result = {}

        # Convert model_input
        model_input = datum_dict.get("model_input", {})
        if "chunks" in model_input:
            # tinker format - extract tokens from chunks
            tokens = []
            for chunk in model_input["chunks"]:
                if "tokens" in chunk:
                    tokens.extend(chunk["tokens"])
            result["model_input"] = {"input_ids": tokens}
        elif "input_ids" in model_input:
            # Already in xorl_client format
            result["model_input"] = model_input
        else:
            result["model_input"] = model_input

        # Copy loss_fn_inputs as-is (format is compatible)
        if "loss_fn_inputs" in datum_dict:
            result["loss_fn_inputs"] = datum_dict["loss_fn_inputs"]

        return result

    def get_tokenizer(self):
        """Get the tokenizer for the base model.

        The tokenizer is lazy-loaded and cached for subsequent calls.
        Uses HuggingFace's AutoTokenizer to load the tokenizer for the base model.

        Returns:
            PreTrainedTokenizer: The tokenizer for the base model

        Example:
            >>> training_client = service_client.create_lora_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct"
            ... )
            >>> tokenizer = training_client.get_tokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.base_model

            # Avoid gating of Llama 3 models
            if model_name.startswith("meta-llama/Llama-3"):
                model_name = "thinkingmachineslabinc/meta-llama-3-tokenizer"

            kwargs = {}
            if model_name == "moonshotai/Kimi-K2-Thinking":
                kwargs["trust_remote_code"] = True
                kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

            self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
            logger.info(f"Tokenizer loaded for base_model={self.base_model}")
        return self._tokenizer

    def forward_backward(
        self,
        data: Union[List[types.Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Execute forward and backward pass.

        Args:
            data: List of training examples (Datum objects or dicts)
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", etc.)

        Returns:
            APIFuture[ForwardBackwardOutput] that can be awaited

        Example:
            >>> fwd_bwd_future = training_client.forward_backward(datums, "importance_sampling")
            >>> result = fwd_bwd_future.result()  # Block and wait
            >>> print(result.loss_fn_outputs[0]["loss"])
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        # Convert Datum objects to dicts if needed
        # Support both xorl_client.types.Datum and tinker.Datum (Pydantic model)
        datums_dicts = []
        for datum in data:
            if isinstance(datum, types.Datum):
                datums_dicts.append(datum.to_dict())
            elif hasattr(datum, 'to_dict'):
                # Support objects with to_dict() method
                datums_dicts.append(datum.to_dict())
            elif hasattr(datum, 'model_dump'):
                # Support Pydantic v2 models (like tinker.Datum)
                # Convert tinker format to xorl_client format
                datum_dict = datum.model_dump()
                datums_dicts.append(self._convert_tinker_datum(datum_dict))
            elif isinstance(datum, dict):
                datums_dicts.append(datum)
            else:
                raise TypeError(
                    f"Expected Datum, dict, or Pydantic model, got {type(datum).__name__}"
                )

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,  # seq_id starts from 1 (like Tinker)
            "forward_backward_input": {
                "data": datums_dicts,
                "loss_fn": loss_fn,
            }
        }

        # Use _take_turn to ensure sequential HTTP dispatch (exactly like Tinker)
        async def _forward_backward_async():
            # Define HTTP sender (like Tinker's _send_request pattern)
            async def _send_request():
                # Use async HTTP directly (like Tinker) - no asyncio.to_thread!
                return await self.holder.post(
                    "/api/v1/forward_backward",
                    request_data
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Turn released here - now parse and return (like Tinker)
            raw_outputs = result.get("loss_fn_outputs", [])
            metrics = result.get("metrics", {})
            logger.info(
                f"Forward-backward completed: num_outputs={len(raw_outputs)}, "
                f"loss_mean={metrics.get('loss:mean', 'N/A')}"
            )

            # Convert loss_fn_outputs dict values to TensorData objects
            converted_outputs = []
            for output in raw_outputs:
                converted_output = {}
                for key, value in output.items():
                    if isinstance(value, dict) and "data" in value:
                        # This looks like a serialized TensorData, convert it
                        converted_output[key] = types.TensorData.from_dict(value)
                    else:
                        # Pass through other values (like scalar 'loss')
                        converted_output[key] = value
                converted_outputs.append(converted_output)

            return types.ForwardBackwardOutput(
                loss_fn_outputs=converted_outputs,
                metrics=metrics,
            )

        # Schedule to event loop and return directly (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_forward_backward_async()))

    def optim_step(
        self,
        adam_params: Union[types.AdamParams, Dict[str, float]],
    ) -> APIFuture[types.OptimStepResponse]:
        """Perform optimizer step.

        Args:
            adam_params: Adam parameters (AdamParams object or dict with learning_rate, etc.)

        Returns:
            APIFuture[OptimStepResponse] that can be awaited

        Example:
            >>> optim_future = training_client.optim_step(
            ...     types.AdamParams(learning_rate=1e-5)
            ... )
            >>> result = optim_future.result()
            >>> print(result.metrics["grad_norm"])
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        # Convert AdamParams to dict if needed
        if isinstance(adam_params, types.AdamParams):
            adam_params_dict = adam_params.to_dict()
        else:
            adam_params_dict = adam_params

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,  # seq_id starts from 1 (like Tinker)
            "adam_params": adam_params_dict,
        }

        # Use _take_turn to ensure sequential HTTP dispatch (exactly like Tinker)
        async def _optim_step_async():
            # Define HTTP sender (like Tinker's _send_request pattern)
            async def _send_request():
                # Use async HTTP directly (like Tinker) - no asyncio.to_thread!
                return await self.holder.post(
                    "/api/v1/optim_step",
                    request_data
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Turn released here - now parse and return (like Tinker)
            metrics = result.get("metrics", {})
            logger.info(
                f"Optimizer step completed: "
                f"grad_norm={metrics.get('grad_norm', 'N/A')}"
            )
            return types.OptimStepResponse(
                metrics=metrics,
            )

        # Schedule to event loop and return directly (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_optim_step_async()))

    def save_weights_for_sampler(
        self,
        name: str,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Save weights specifically for sampling/inference.

        This is called frequently (e.g., every batch in RL training) to make
        the latest policy weights available for sampling. It's separate from
        save_state() which includes optimizer state for checkpointing.

        The returned model_path can be used to create a SamplingClient.

        Args:
            name: Name for this checkpoint (e.g., "step-000100")

        Returns:
            APIFuture[SaveWeightsForSamplerResponse] with model_path

        Example:
            >>> save_future = training_client.save_weights_for_sampler("step-100")
            >>> response = save_future.result()
            >>> print(response.path)  # e.g., "xorl://model-123/step-100"
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
        }

        # Use _take_turn to ensure sequential HTTP dispatch (like Tinker)
        async def _save_weights_for_sampler_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/save_weights_for_sampler",
                    request_data
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Parse and return (like Tinker)
            model_path = result.get("model_path")
            if not model_path:
                raise RuntimeError("No model_path returned from save_weights_for_sampler")
            logger.info(f"Weights saved for sampler: {model_path}")
            return types.SaveWeightsForSamplerResponse(path=model_path)

        # Schedule to event loop and return directly (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async()))

    async def save_weights_for_sampler_async(
        self,
        name: str,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Async version of save_weights_for_sampler.

        Save weights specifically for sampling/inference.
        This is the async wrapper that allows using save_weights_for_sampler in async contexts.

        Args:
            name: Name for this checkpoint (e.g., "step-000100")

        Returns:
            APIFuture[SaveWeightsForSamplerResponse] with model_path

        Example:
            >>> # Option 1: Await the future directly
            >>> save_future = await training_client.save_weights_for_sampler_async("step-100")
            >>> result = await save_future
            >>> print(f"Weights saved to: {result.path}")
            >>>
            >>> # Option 2: Use result_async
            >>> save_future = await training_client.save_weights_for_sampler_async("step-100")
            >>> result = await save_future.result_async()
        """
        return self.save_weights_for_sampler(name)

    def save_weights_and_get_sampling_client(
        self,
        name: Optional[str] = None,
        inference_base_url: str = "http://localhost:30000",
    ) -> "SamplingClient":
        """Atomic operation: save weights and create sampling client.

        This is the recommended way to get a sampling client with the latest
        policy weights in RL training.

        Args:
            name: Optional name for checkpoint (default: auto-generated)
            inference_base_url: Base URL for inference server (default: http://localhost:30000)

        Returns:
            SamplingClient ready to use

        Example:
            >>> # Every batch in RL loop
            >>> sampling_client = training_client.save_weights_and_get_sampling_client()
            >>> response = sampling_client.sample(prompt, sampling_params)
        """
        from xorl_client.client.sampling_client import SamplingClient

        # Generate name if not provided
        if name is None:
            import time
            name = f"step-{int(time.time())}"

        # Save weights
        save_result = self.save_weights_for_sampler(name).result()
        model_path = save_result.path

        # Call training server to create sampling session (loads LoRA on inference workers)
        logger.info(f"Creating sampling session: model_path={model_path}")
        try:
            response = self.holder.post_sync(
                "/api/v1/create_sampling_session",
                {"model_path": model_path},
                timeout=60.0,  # LoRA loading can take some time
            )

            if not response.get("success", False):
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Failed to create sampling session: {error_msg}")

            lora_name = response.get("lora_name", "")
            logger.info(f"Sampling session created: lora_name={lora_name}, model_path={model_path}")

        except Exception as e:
            logger.error(f"Failed to create sampling session for {model_path}: {e}")
            raise

        # Create sampling client with new API
        return SamplingClient(
            base_url=inference_base_url,
            model_path=model_path,
            api_key=self.holder.api_key,
            timeout=self.holder.timeout,
        )

    async def save_weights_and_get_sampling_client_async(
        self,
        name: Optional[str] = None,
        inference_base_url: str = "http://localhost:30000",
    ) -> "SamplingClient":
        """Async version of save_weights_and_get_sampling_client.

        Atomic operation: save weights and create sampling client.
        This is the async wrapper for use in async contexts.

        Args:
            name: Optional name for checkpoint (default: auto-generated)
            inference_base_url: Base URL for inference server (default: http://localhost:30000)

        Returns:
            SamplingClient ready to use

        Example:
            >>> # Every batch in RL loop (async context)
            >>> sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            >>> response = await sampling_client.sample_async(prompt, sampling_params)
        """
        from xorl_client.client.sampling_client import SamplingClient

        # Generate name if not provided
        if name is None:
            import time
            name = f"step-{int(time.time())}"

        # Save weights - await the async version and then get the result
        save_future = await self.save_weights_for_sampler_async(name)
        save_result = await save_future.result_async()
        model_path = save_result.path

        # Call training server to create sampling session (loads LoRA on inference workers)
        logger.info(f"Creating sampling session: model_path={model_path}")
        try:
            response = self.holder.post_sync(
                "/api/v1/create_sampling_session",
                {"model_path": model_path},
                timeout=60.0,  # LoRA loading can take some time
            )

            if not response.get("success", False):
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Failed to create sampling session: {error_msg}")

            lora_name = response.get("lora_name", "")
            logger.info(f"Sampling session created: lora_name={lora_name}, model_path={model_path}")

        except Exception as e:
            logger.error(f"Failed to create sampling session for {model_path}: {e}")
            raise

        # Create sampling client with new API
        return SamplingClient(
            base_url=inference_base_url,
            model_path=model_path,
            api_key=self.holder.api_key,
            timeout=self.holder.timeout,
        )

    def save_state(
        self,
        name: Optional[str] = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save model weights (and optimizer state) to persistent storage.

        This is used for checkpointing/resume, not for frequent sampling updates.
        For RL training, use save_weights_for_sampler() instead.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001"). Auto-generated if not specified.

        Returns:
            APIFuture[SaveWeightsResponse] with xorl:// URI

        Example:
            >>> save_future = training_client.save_state("checkpoint-001")
            >>> response = save_future.result()
            >>> print(response.path)  # xorl://default/weights/checkpoint-001
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "path": name,
        }

        # Use _take_turn to ensure sequential HTTP dispatch (like Tinker)
        async def _save_state_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/save_weights",
                    request_data
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Parse and return (like Tinker)
            path = result.get("path")
            if not path:
                raise RuntimeError("No path returned from save_weights")
            logger.info(f"Checkpoint saved: {path}")
            return types.SaveWeightsResponse(path=path)

        # Schedule to event loop and return directly (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_save_state_async()))

    async def save_state_async(
        self,
        name: Optional[str] = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Async version of save_state.

        Save model weights (and optimizer state) to persistent storage.
        This is the async wrapper that allows using save_state in async contexts.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001"). Auto-generated if not specified.

        Returns:
            APIFuture[SaveWeightsResponse] with xorl:// URI

        Example:
            >>> # Option 1: Await the future directly
            >>> save_future = await training_client.save_state_async("checkpoint-001")
            >>> result = await save_future
            >>> print(f"Saved to: {result.path}")
            >>>
            >>> # Option 2: Use result_async
            >>> save_future = await training_client.save_state_async()
            >>> result = await save_future.result_async()
        """
        return self.save_state(name)

    def _load_weights_impl(
        self,
        path: str,
        optimizer: bool,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Internal implementation for loading weights.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")
            optimizer: Whether to load optimizer state

        Returns:
            APIFuture[LoadWeightsResponse]
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "path": path,
            "optimizer": optimizer,
        }

        # Use _take_turn to ensure sequential HTTP dispatch (like Tinker)
        async def _load_weights_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/load_weights",
                    request_data
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Parse and return (like Tinker)
            loaded_path = result.get("path")
            if not loaded_path:
                raise RuntimeError(f"Failed to load checkpoint: {path}")
            logger.info(f"Checkpoint loaded: {loaded_path} (optimizer={optimizer})")
            return types.LoadWeightsResponse(path=loaded_path)

        # Schedule to event loop and return directly (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_load_weights_async()))

    def load_state(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights from a saved checkpoint.

        This loads only the model weights, not optimizer state (e.g., Adam momentum).
        To also restore optimizer state, use load_state_with_optimizer.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse] with the loaded path

        Example:
            >>> # Load checkpoint to continue training (weights only, optimizer resets)
            >>> load_future = training_client.load_state("xorl://default/weights/checkpoint-001")
            >>> result = load_future.result()
            >>> print(result.path)
        """
        return self._load_weights_impl(path, optimizer=False)

    async def load_state_async(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state.

        Load model weights from a saved checkpoint (without optimizer state).

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse]

        Example:
            >>> load_future = await training_client.load_state_async("xorl://default/weights/checkpoint-001")
            >>> result = await load_future
            >>> print(f"Loaded: {result.path}")
        """
        return self.load_state(path)

    def load_state_with_optimizer(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint.

        This restores both model weights and optimizer state (e.g., Adam momentum),
        allowing training to resume exactly where it left off.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse] with the loaded path

        Example:
            >>> # Resume training with optimizer state
            >>> load_future = training_client.load_state_with_optimizer(
            ...     "xorl://default/weights/checkpoint-001"
            ... )
            >>> result = load_future.result()
            >>> print(result.path)
        """
        return self._load_weights_impl(path, optimizer=True)

    async def load_state_with_optimizer_async(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state_with_optimizer.

        Load model weights and optimizer state from a checkpoint.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse]

        Example:
            >>> load_future = await training_client.load_state_with_optimizer_async(
            ...     "xorl://default/weights/checkpoint-001"
            ... )
            >>> result = await load_future
            >>> print(f"Loaded with optimizer: {result.path}")
        """
        return self.load_state_with_optimizer(path)

    def health_check(self) -> Dict[str, Any]:
        """Check API server health.

        Returns:
            Health check response
        """
        return self.holder.get_sync("/health", timeout=10)

    def update_dedicated_endpoint(
        self,
        name: str,
        endpoint_id: str,
        provider_api_key: str,
        hf_token: str,
        hf_org: str = "xorl-ai",
        provider_model_id: Optional[str] = None,
    ) -> APIFuture[types.UpdateDedicatedEndpointResponse]:
        """
        Save checkpoint and update dedicated endpoint with new LoRA weights.

        This triggers the server to:
        1. Save LoRA checkpoint locally
        2. Upload to HuggingFace (private repo)
        3. Upload via models.upload API
        4. Wait for model to be ready
        5. Probe endpoint to verify adapter is loaded

        All heavy lifting happens on the server. The client just waits for completion.

        Note: This operation can take 30-60 seconds due to HF propagation and provider upload.

        Args:
            name: Checkpoint name (e.g., "step-000001")
            endpoint_id: dedicated endpoint ID
            provider_api_key: provider API key
            hf_token: HuggingFace token for uploading
            hf_org: HuggingFace organization (default: "xorl-ai")
            provider_model_id: Optional provider model ID (auto-generated if not provided)

        Returns:
            APIFuture[UpdateDedicatedEndpointResponse] with provider_model_id for sampling

        Example:
            >>> # Update dedicated endpoint
            >>> update_future = training_client.update_dedicated_endpoint(
            ...     name="step-000001",
            ...     endpoint_id="your-endpoint-id",
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     hf_token=os.environ["HF_TOKEN"],
            ...     hf_org="your-hf-org",
            ... )
            >>> result = update_future.result()
            >>>
            >>> # Use the model ID for sampling
            >>> sampling_client.set_model(result.provider_model_id)
            >>> response = sampling_client.sample(prompt, sampling_params)
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
            "endpoint_id": endpoint_id,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "hf_org": hf_org,
            "provider_model_id": provider_model_id,
        }

        # Use _take_turn to ensure sequential HTTP dispatch
        async def _update_dedicated_endpoint_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/update_dedicated_endpoint",
                    request_data,
                    timeout=300,  # 5 minute timeout
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Parse and return
            success = result.get("success", False)
            provider_model_id = result.get("provider_model_id", "")
            if success:
                logger.info(
                    f"Dedicated endpoint updated: {provider_model_id} "
                    f"(HF: {result.get('upload_time', 0):.1f}s, "
                    f"Provider: {result.get('provider_upload_time', 0):.1f}s, "
                    f"Probe: {result.get('probe_time', 0):.1f}s)"
                )
            else:
                logger.error(f"Failed to update dedicated endpoint: {result.get('error')}")
            return types.UpdateDedicatedEndpointResponse(
                success=success,
                checkpoint_path=result.get("checkpoint_path", ""),
                hf_path=result.get("hf_path", ""),
                provider_model_id=provider_model_id,
                upload_time=result.get("upload_time", 0.0),
                provider_upload_time=result.get("provider_upload_time", 0.0),
                probe_time=result.get("probe_time", 0.0),
                error=result.get("error"),
            )

        # Schedule to event loop and return directly
        return wrap_future(self.holder.run_coroutine_threadsafe(_update_dedicated_endpoint_async()))

    def update_serverless_weights(
        self,
        name: str,
        provider_api_key: str,
        hf_token: str,
        provider_base_model: str,
        hf_repo_prefix: str = "xorl_client-adapter",
        private_repo: bool = True,
        hf_org: str = "xorl-ai",
        wait_for_completion: bool = True,
    ) -> APIFuture[types.UpdateServerlessWeightsResponse]:
        """Save weights and register with XoRL for serverless inference.

        This is the serverless alternative to save_weights_for_sampler().
        The adapter is uploaded to HuggingFace Hub and registered for serverless inference.

        Args:
            name: Checkpoint name (e.g., "step-001")
            provider_api_key: API key
            hf_token: HuggingFace token with write access
            provider_base_model: base model name
            hf_repo_prefix: Prefix for HuggingFace repository names
            private_repo: Whether to create private HuggingFace repository
            hf_org: HuggingFace organization to upload to (default: "xorl-ai")
            wait_for_completion: Whether to wait for upload to complete

        Returns:
            APIFuture[UpdateServerlessWeightsResponse] with provider_model_id for inference

        Example:
            >>> result = training_client.update_serverless_weights(
            ...     name="step-001",
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     hf_token=os.environ["HF_TOKEN"],
            ...     provider_base_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
            ... ).result()
            >>>
            >>> # Create sampling client with the returned model ID
            >>> from xorl_client.client.serverless_sampling_client import ServerlessSamplingClient
            >>> sampling_client = ServerlessSamplingClient(
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     model_id=result.provider_model_id,
            ... )
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "provider_base_model": provider_base_model,
            "hf_repo_prefix": hf_repo_prefix,
            "private_repo": private_repo,
            "hf_org": hf_org,
            "wait_for_completion": wait_for_completion,
        }

        # Use _take_turn to ensure sequential HTTP dispatch
        async def _update_serverless_weights_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/update_serverless_weights",
                    request_data,
                    timeout=600.0,  # 10 minute timeout
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await _send_request()

            # Parse and return
            provider_model_id = result.get("provider_model_id", "")
            hf_repo_url = result.get("hf_repo_url", "")
            checkpoint_path = result.get("checkpoint_path", "")
            status = result.get("status", "Complete")

            logger.info(
                f"Serverless weights updated: provider_model_id={provider_model_id}, "
                f"hf_repo={hf_repo_url}, status={status}"
            )
            return types.UpdateServerlessWeightsResponse(
                provider_model_id=provider_model_id,
                hf_repo_url=hf_repo_url,
                checkpoint_path=checkpoint_path,
                status=status,
            )

        # Schedule to event loop and return directly
        return wrap_future(self.holder.run_coroutine_threadsafe(_update_serverless_weights_async()))

    def list_checkpoints(self) -> APIFuture[types.CheckpointsListResponse]:
        """List all available checkpoints.

        Returns both training checkpoints (weights/) and sampler checkpoints (sampler_weights/).
        Checkpoints are sorted by creation time (newest first).

        Returns:
            APIFuture[CheckpointsListResponse] with list of checkpoints

        Example:
            >>> response = training_client.list_checkpoints().result()
            >>> for ckpt in response.checkpoints:
            ...     print(f"{ckpt.checkpoint_id}: {ckpt.checkpoint_type}")
        """
        request_data = {
            "model_id": self.model_id,
        }

        future = self.holder.post_async("/api/v1/list_checkpoints", request_data)

        def parse_response(result: Dict[str, Any]) -> types.CheckpointsListResponse:
            checkpoints = [
                types.Checkpoint.from_dict(c)
                for c in result.get("checkpoints", [])
            ]
            logger.info(f"Listed {len(checkpoints)} checkpoints")
            return types.CheckpointsListResponse(checkpoints=checkpoints)

        from concurrent.futures import Future
        parsed_future: Future[types.CheckpointsListResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def list_checkpoints_async(self) -> APIFuture[types.CheckpointsListResponse]:
        """Async version of list_checkpoints.

        Returns:
            APIFuture[CheckpointsListResponse] with list of checkpoints
        """
        return self.list_checkpoints()

    def delete_checkpoint(
        self,
        checkpoint_id: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete (e.g., 'weights/000' or 'sampler_weights/step-100')

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status

        Example:
            >>> response = training_client.delete_checkpoint("weights/000").result()
            >>> print(f"Deleted: {response.deleted_path}")
        """
        request_data = {
            "model_id": self.model_id,
            "checkpoint_id": checkpoint_id,
        }

        future = self.holder.post_async("/api/v1/delete_checkpoint", request_data)

        def parse_response(result: Dict[str, Any]) -> types.DeleteCheckpointResponse:
            success = result.get("success", False)
            deleted_path = result.get("deleted_path")
            error = result.get("error")
            if success:
                logger.info(f"Deleted checkpoint: {deleted_path}")
            else:
                logger.error(f"Failed to delete checkpoint: {error}")
            return types.DeleteCheckpointResponse(
                success=success,
                deleted_path=deleted_path,
                error=error,
            )

        from concurrent.futures import Future
        parsed_future: Future[types.DeleteCheckpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def delete_checkpoint_async(
        self,
        checkpoint_id: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Async version of delete_checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status
        """
        return self.delete_checkpoint(checkpoint_id)

    def delete_checkpoint_from_xorl_path(
        self,
        xorl_path: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Delete a checkpoint using its xorl:// path.

        This is a convenience method that parses the xorl:// path and calls delete_checkpoint.

        Args:
            xorl_path: Full xorl:// path (e.g., 'xorl://default/weights/000')

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status

        Example:
            >>> response = training_client.delete_checkpoint_from_xorl_path(
            ...     "xorl://default/weights/000"
            ... ).result()
            >>> print(f"Deleted: {response.deleted_path}")
        """
        parsed = types.ParsedCheckpointXoRLPath.from_xorl_path(xorl_path)
        return self.delete_checkpoint(parsed.checkpoint_id)

    async def delete_checkpoint_from_xorl_path_async(
        self,
        xorl_path: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Async version of delete_checkpoint_from_xorl_path.

        Args:
            xorl_path: Full xorl:// path

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status
        """
        return self.delete_checkpoint_from_xorl_path(xorl_path)