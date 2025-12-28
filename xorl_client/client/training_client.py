"""
TrainingClient - Tinker-compatible training client with futures.

This is the new version that uses ClientHolder and returns APIFutures.
"""

from __future__ import annotations

import logging
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

    Args:
        holder: ClientHolder managing HTTP connections and async operations
        model_id: Unique identifier for the model to train

    Example:
        >>> service_client = ServiceClient(base_url="http://localhost:5555")
        >>> training_client = service_client.create_lora_training_client(
        ...     base_model="Qwen/Qwen2.5-3B-Instruct",
        ...     rank=32
        ... )
        >>>
        >>> # Pipeline operations (submit both immediately)
        >>> fwd_bwd_future = training_client.forward_backward(datums, "importance_sampling")
        >>> optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-5))
        >>>
        >>> # Wait for both to complete
        >>> fwd_bwd_result = fwd_bwd_future.result()
        >>> optim_result = optim_future.result()
    """

    def __init__(self, holder: ClientHolder, model_id: str):
        """Initialize TrainingClient.

        Args:
            holder: ClientHolder for connection management
            model_id: Model ID
        """
        self.holder = holder
        self.model_id = model_id
        logger.info(f"TrainingClient initialized: model_id={model_id}")

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
        # Convert Datum objects to dicts if needed
        datums_dicts = []
        for datum in data:
            if isinstance(datum, types.Datum):
                datums_dicts.append(datum.to_dict())
            else:
                datums_dicts.append(datum)

        request_data = {
            "model_id": self.model_id,
            "forward_backward_input": {
                "data": datums_dicts,
                "loss_fn": loss_fn,
            }
        }

        # Submit async request
        future = self.holder.post_async("/api/v1/forward_backward", request_data)

        # Wrap in APIFuture and parse response
        def parse_response(result: Dict[str, Any]) -> types.ForwardBackwardOutput:
            logger.info(f"Forward-backward completed: loss={result.get('loss_fn_outputs', [{}])[0].get('loss', 'N/A')}")
            return types.ForwardBackwardOutput(
                loss_fn_outputs=result.get("loss_fn_outputs", []),
                metrics=result.get("metrics", {}),
            )

        # Chain parsing onto the future
        from concurrent.futures import Future
        parsed_future: Future[types.ForwardBackwardOutput] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def optim_step(
        self,
        adam_params: Union[types.AdamParams, Dict[str, float]],
        gradient_clip: float = 1.0,
    ) -> APIFuture[types.OptimStepResponse]:
        """Perform optimizer step.

        Args:
            adam_params: Adam parameters (AdamParams object or dict with learning_rate, etc.)
            gradient_clip: Gradient clipping threshold

        Returns:
            APIFuture[OptimStepResponse] that can be awaited

        Example:
            >>> optim_future = training_client.optim_step(
            ...     types.AdamParams(learning_rate=1e-5)
            ... )
            >>> result = optim_future.result()
            >>> print(result.metrics["grad_norm"])
        """
        # Convert AdamParams to dict if needed
        if isinstance(adam_params, types.AdamParams):
            adam_params_dict = adam_params.to_dict()
        else:
            adam_params_dict = adam_params

        request_data = {
            "model_id": self.model_id,
            "adam_params": adam_params_dict,
            "gradient_clip": gradient_clip,
        }

        # Submit async request
        future = self.holder.post_async("/api/v1/optim_step", request_data)

        # Wrap and parse
        def parse_response(result: Dict[str, Any]) -> types.OptimStepResponse:
            metrics = result.get("metrics", {})
            logger.info(
                f"Optimizer step completed: "
                f"step={metrics.get('step', 'N/A')}, "
                f"grad_norm={metrics.get('grad_norm', 'N/A')}"
            )
            return types.OptimStepResponse(
                metrics=metrics,
                step=metrics.get("step", 0),
            )

        from concurrent.futures import Future
        parsed_future: Future[types.OptimStepResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

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
        request_data = {
            "model_id": self.model_id,
            "name": name,
        }

        future = self.holder.post_async("/api/v1/save_weights_for_sampler", request_data)

        def parse_response(result: Dict[str, Any]) -> types.SaveWeightsForSamplerResponse:
            model_path = result.get("model_path")
            if not model_path:
                raise RuntimeError("No model_path returned from save_weights_for_sampler")
            logger.info(f"Weights saved for sampler: {model_path}")
            return types.SaveWeightsForSamplerResponse(path=model_path)

        from concurrent.futures import Future
        parsed_future: Future[types.SaveWeightsForSamplerResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def save_weights_and_get_sampling_client(
        self,
        name: Optional[str] = None,
    ) -> "SamplingClient":
        """Atomic operation: save weights and create sampling client.

        This is the recommended way to get a sampling client with the latest
        policy weights in RL training.

        Args:
            name: Optional name for checkpoint (default: auto-generated)

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

        # Create sampling client
        return SamplingClient(holder=self.holder, model_path=model_path)

    def save_state(
        self,
        checkpoint_path: Optional[str] = None,
        save_optimizer: bool = True,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save full training state (checkpoint) including optimizer state.

        This is used for checkpointing/resume, not for frequent sampling updates.
        For RL training, use save_weights_for_sampler() instead.

        Args:
            checkpoint_path: Path for checkpoint (None = auto-generate)
            save_optimizer: Whether to save optimizer state

        Returns:
            APIFuture[SaveWeightsResponse] with checkpoint path

        Example:
            >>> save_future = training_client.save_state()
            >>> response = save_future.result()
            >>> print(response.path)
        """
        request_data = {
            "model_id": self.model_id,
            "path": checkpoint_path,
            "save_optimizer": save_optimizer,
        }

        future = self.holder.post_async("/api/v1/save_state", request_data)

        def parse_response(result: Dict[str, Any]) -> types.SaveWeightsResponse:
            path = result.get("path")
            if not path:
                raise RuntimeError("No path returned from save_state")
            logger.info(f"Checkpoint saved: {path}")
            return types.SaveWeightsResponse(path=path)

        from concurrent.futures import Future
        parsed_future: Future[types.SaveWeightsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def load_state(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state

        Returns:
            APIFuture[LoadWeightsResponse]

        Example:
            >>> load_future = training_client.load_state("/path/to/checkpoint")
            >>> response = load_future.result()
            >>> print(response.success)
        """
        request_data = {
            "model_id": self.model_id,
            "path": checkpoint_path,
            "load_optimizer": load_optimizer,
        }

        future = self.holder.post_async("/api/v1/load_state", request_data)

        def parse_response(result: Dict[str, Any]) -> types.LoadWeightsResponse:
            success = result.get("success", False)
            if success:
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
            else:
                logger.error(f"Failed to load checkpoint: {checkpoint_path}")
            return types.LoadWeightsResponse(success=success)

        from concurrent.futures import Future
        parsed_future: Future[types.LoadWeightsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def health_check(self) -> Dict[str, Any]:
        """Check API server health.

        Returns:
            Health check response
        """
        return self.holder.get("/health", timeout=10)

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
        request_data = {
            "model_id": self.model_id,
            "name": name,
            "endpoint_id": endpoint_id,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "hf_org": hf_org,
            "provider_model_id": provider_model_id,
        }

        # Use longer timeout for this operation (can take 30-60+ seconds)
        future = self.holder.post_async(
            "/api/v1/update_dedicated_endpoint",
            request_data,
            timeout=300,  # 5 minute timeout
        )

        def parse_response(result: Dict[str, Any]) -> types.UpdateDedicatedEndpointResponse:
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

        from concurrent.futures import Future
        parsed_future: Future[types.UpdateDedicatedEndpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

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
        request_data = {
            "model_id": self.model_id,
            "name": name,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "provider_base_model": provider_base_model,
            "hf_repo_prefix": hf_repo_prefix,
            "private_repo": private_repo,
            "hf_org": hf_org,
            "wait_for_completion": wait_for_completion,
        }

        # Use longer timeout for serverless weights (HF + provider upload can take time)
        future = self.holder.post_async(
            "/api/v1/update_serverless_weights",
            request_data,
            timeout=600.0,  # 10 minute timeout
        )

        def parse_response(result: Dict[str, Any]) -> types.UpdateServerlessWeightsResponse:
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

        from concurrent.futures import Future
        parsed_future: Future[types.UpdateServerlessWeightsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)
