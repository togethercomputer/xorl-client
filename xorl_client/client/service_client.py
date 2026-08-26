"""
ServiceClient - Main entry point for XoRL API.

Provides factory methods to create TrainingClient and SamplingClient instances.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict, Optional

from xorl_client import types
from xorl_client.client.api_future_impl import _APIFuture
from xorl_client.client.client_holder import ClientHolder
from xorl_client.exceptions import InternalServerError, BadRequestError

if TYPE_CHECKING:
    from xorl_client.client.rest_client import RestClient
    from xorl_client.client.sampling_client import SamplingClient
    from xorl_client.client.training_client import TrainingClient

logger = logging.getLogger(__name__)


def _extract_model_id_from_xorl_uri(checkpoint_path: str) -> Optional[str]:
    """Extract model_id from a xorl:// URI.

    Args:
        checkpoint_path: XoRL URI (e.g., "xorl://rl-abc123/weights/000010")

    Returns:
        The model_id (e.g., "rl-abc123") or None if not a valid xorl:// URI
    """
    if not checkpoint_path.startswith("xorl://"):
        return None

    # Remove the "xorl://" prefix
    path_without_prefix = checkpoint_path[7:]  # len("xorl://") == 7

    # Split by "/" and get the first component (model_id)
    parts = path_without_prefix.split("/")
    if parts and parts[0]:
        return parts[0]

    return None


class ServiceClient:
    """The ServiceClient is the main entry point for the XoRL API.

    It provides methods to:
    - Create TrainingClient instances for model training workflows
    - Create SamplingClient instances for text generation and inference
    - Resume training from checkpoints

    Args:
        base_url: Base URL for the XoRL training API (default: XORL_BASE_URL env var or "http://localhost:5555")
        api_key: API key for authentication (default: XORL_API_KEY env var)
        timeout: Default timeout for HTTP requests in seconds
        **kwargs: Additional arguments for future compatibility

    Example:
        >>> service_client = ServiceClient(base_url="http://localhost:5555")
        >>> training_client = service_client.create_lora_training_client(
        ...     base_model="Qwen/Qwen2.5-3B-Instruct",
        ...     rank=32
        ... )
        >>> sampling_client = service_client.create_sampling_client(
        ...     base_url="http://localhost:30000",
        ...     model_path="xorl://model-123/step-100"
        ... )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 1800.0,
        **kwargs: Any,
    ):
        """Initialize ServiceClient.

        Args:
            base_url: Base URL for the training API (default: XORL_BASE_URL env var or "http://localhost:5555")
            model: Model identifier for API routing (optional)
            api_key: API key for authentication (default: XORL_API_KEY env var)
            timeout: Default timeout for requests
            **kwargs: Additional arguments
        """
        if base_url is None:
            base_url = os.environ.get("XORL_BASE_URL", "http://localhost:5555")
        if api_key is None:
            api_key = os.environ.get("XORL_API_KEY")

        self.holder = ClientHolder(
            base_url=base_url, model=model, api_key=api_key, timeout=timeout, **kwargs
        )
        logger.info(f"ServiceClient initialized: base_url={base_url}, model={model}")

    def _create_lora_training_client_submit(
        self,
        base_model: str,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> Future["TrainingClient"]:
        """Helper function that submits the create_lora_training_client request (two-phase pattern)."""
        import time
        from xorl_client.client.training_client import TrainingClient

        # Warn about LoRA parameters that can't be adjusted from user side
        if (
            rank != 32
            or alpha is not None
            or dropout != 0.0
            or target_modules is not None
        ):
            logger.warning(
                "Note: LoRA parameters (rank, alpha, dropout, target_modules) are currently "
                "configured on the server side and cannot be adjusted from the client. "
                "The values provided here will be sent to the server but may be ignored "
                "if the server is already configured with different LoRA settings."
            )

        # Use provided model_id or default to "default"
        if model_id is None:
            model_id = "default"

        # Create LoRA config
        lora_config = types.LoraConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

        # Send create model request to server
        logger.info(
            f"Creating LoRA training client: model_id={model_id}, base_model={base_model}, rank={rank}"
        )
        request_start_time = time.time()

        try:
            response = self.holder.post_sync(
                "/api/v1/create_model",
                {
                    "model_id": model_id,
                    "base_model": base_model,
                    "lora_config": lora_config.to_dict(),
                },
            )

            # Handle both response formats:
            # - Old format: {model_id, status} - immediate response
            # - New format: {request_id, model_id} - two-phase pattern
            if "request_id" in response:
                # Two-phase pattern: poll for result
                untyped_future = types.UntypedAPIFuture.from_dict(response)
                api_future = _APIFuture(
                    model_cls=types.CreateModelResponse,
                    holder=self.holder,
                    untyped_future=untyped_future,
                    request_start_time=request_start_time,
                    request_type="CreateModel",
                )
                create_result = api_future.result()
                logger.info(f"Model created: model_id={create_result.model_id}")
            else:
                # Old format: immediate response with {model_id, status}
                if "model_id" in response and "status" in response:
                    logger.info(
                        f"Model created: model_id={response['model_id']}, status={response['status']}"
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected response format from create_model: {response}"
                    )

        except RuntimeError as e:
            logger.error(f"Failed to create model: {e}")
            raise

        # Create and return TrainingClient
        training_client = TrainingClient(
            holder=self.holder, model_id=model_id, base_model=base_model
        )

        # Return a completed future
        future: Future[TrainingClient] = Future()
        future.set_result(training_client)
        return future

    def create_training_client(
        self,
        base_model: str,
        model_id: Optional[str] = None,
    ) -> "TrainingClient":
        """Create a training client for full-weights training (non-LoRA).

        This method creates a TrainingClient for training without LoRA adapters.
        Use this when the server is configured with enable_lora=False.

        The method:
        1. Registers the model_id with the training server
        2. Returns a TrainingClient ready for training

        Args:
            base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
            model_id: Optional model ID (default: "default"). Use different IDs
                     for multiple concurrent training sessions.

        Returns:
            TrainingClient instance

        Example:
            >>> # For full-weights training (server has enable_lora=False)
            >>> training_client = service_client.create_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct"
            ... )
            >>> # With custom model_id for concurrent runs
            >>> training_client = service_client.create_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     model_id="training-run-001"
            ... )
        """
        return self._create_training_client_submit(base_model, model_id).result()

    def _create_training_client_submit(
        self,
        base_model: str,
        model_id: Optional[str] = None,
    ) -> Future["TrainingClient"]:
        """Helper function that submits the create_training_client request."""
        import time
        from xorl_client.client.training_client import TrainingClient

        # Use provided model_id or default to "default"
        if model_id is None:
            model_id = "default"

        # Send create model request to server
        # For full-weights mode, we still call create_model but without meaningful lora_config
        logger.info(
            f"Creating training client: model_id={model_id}, base_model={base_model}"
        )
        request_start_time = time.time()

        try:
            response = self.holder.post_sync(
                "/api/v1/create_model",
                {
                    "model_id": model_id,
                    "base_model": base_model,
                },
            )

            # Handle both response formats:
            # - Old format: {model_id, status} - immediate response
            # - New format: {request_id, model_id} - two-phase pattern
            if "request_id" in response:
                # Two-phase pattern: poll for result
                untyped_future = types.UntypedAPIFuture.from_dict(response)
                api_future = _APIFuture(
                    model_cls=types.CreateModelResponse,
                    holder=self.holder,
                    untyped_future=untyped_future,
                    request_start_time=request_start_time,
                    request_type="CreateModel",
                )
                create_result = api_future.result()
                logger.info(f"Model created: model_id={create_result.model_id}")
            else:
                # Old format: immediate response with {model_id, status}
                if "model_id" in response and "status" in response:
                    logger.info(
                        f"Model created: model_id={response['model_id']}, status={response['status']}"
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected response format from create_model: {response}"
                    )

        except RuntimeError as e:
            logger.error(f"Failed to create model: {e}")
            raise

        # Create and return TrainingClient
        training_client = TrainingClient(
            holder=self.holder, model_id=model_id, base_model=base_model
        )

        # Return a completed future
        future: Future[TrainingClient] = Future()
        future.set_result(training_client)
        return future

    async def create_training_client_async(
        self,
        base_model: str,
        model_id: Optional[str] = None,
    ) -> "TrainingClient":
        """Async version of create_training_client for full-weights training."""
        future = self._create_training_client_submit(base_model, model_id)
        return await asyncio.wrap_future(future)

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> "TrainingClient":
        """Create a LoRA training client.

        This method:
        1. Sends a request to create a new model on the training server
        2. Initializes LoRA training setup
        3. Returns a TrainingClient ready for training

        Args:
            base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha parameter (default: None, uses rank)
            dropout: LoRA dropout (default: 0.0)
            target_modules: Which modules to apply LoRA to (default: None, uses model default)
            model_id: Optional model ID (default: "default"). Use different IDs for multiple tests
                     to avoid "adapter already loaded" errors without restarting inference worker

        Returns:
            TrainingClient instance

        Example:
            >>> training_client = service_client.create_lora_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     rank=32
            ... )
            >>> # For testing with unique model per run:
            >>> training_client = service_client.create_lora_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     model_id="test-run-001"
            ... )
        """
        return self._create_lora_training_client_submit(
            base_model, rank, alpha, dropout, target_modules, model_id
        ).result()

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> "TrainingClient":
        """Async version of create_lora_training_client."""
        future = self._create_lora_training_client_submit(
            base_model, rank, alpha, dropout, target_modules, model_id
        )
        return await asyncio.wrap_future(future)

    def create_sampling_client(
        self,
        base_url: str,
        model_path: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 1800.0,
        api_format: Optional[str] = None,
    ) -> "SamplingClient":
        """Create a sampling client for inference.

        This method:
        1. Calls /api/v1/create_sampling_session on the training server to load
           the LoRA adapter on all inference workers
        2. Creates and returns a SamplingClient that connects to the inference engine

        The SamplingClient connects directly to the inference engine using
        base_url and api_key.

        Args:
            base_url: Base URL for the inference engine (required)
            model_path: Path to saved model weights (required, e.g., "xorl://default/sampler_weights/step-100"
                       or "sampler_weights/step-100")
            model: Model identifier for API routing (optional)
            api_key: API key for authentication (default: XORL_INFERENCE_API_KEY env var)
            timeout: Request timeout in seconds (default: 120.0)
            api_format: Inference API format passed to SamplingClient. Use
                "chat_completions" when sampling through Dispatch.

        Returns:
            SamplingClient instance

        Raises:
            RuntimeError: If the model_path doesn't exist or LoRA loading fails

        Example:
            >>> sampling_client = service_client.create_sampling_client(
            ...     base_url="http://localhost:30000",
            ...     model_path="xorl://default/sampler_weights/step-100"
            ... )
        """
        from xorl_client.client.sampling_client import SamplingClient

        logger.info(f"Creating sampling session: model_path={model_path}")

        # Call training server to create sampling session (loads LoRA on inference workers)
        try:
            response = self.holder.post_sync(
                "/api/v1/create_sampling_session",
                {"model_path": model_path},
                timeout=1800.0,  # LoRA loading can take some time
            )

            if not response.get("success", False):
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Failed to create sampling session: {error_msg}")

            lora_name = response.get("lora_name", "")
            logger.info(
                f"Sampling session created: lora_name={lora_name}, model_path={model_path}"
            )

        except (InternalServerError, BadRequestError) as e:
            # Check if this is a "LoRA already loaded" error - not fatal, just warn
            error_msg = str(e)
            if "already loaded" in error_msg.lower():
                logger.warning(
                    f"LoRA adapter is already loaded on inference worker, continuing: {error_msg}"
                )
            else:
                logger.error(f"Failed to create sampling session for {model_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to create sampling session for {model_path}: {e}")
            raise

        # Create SamplingClient that connects to inference engine
        logger.info(
            f"Creating sampling client: base_url={base_url}, model={model}, model_path={model_path}"
        )
        return SamplingClient(
            base_url=base_url,
            model_path=model_path,
            model=model,
            api_key=api_key,
            timeout=timeout,
            api_format=api_format,
        )

    def create_rest_client(
        self,
        model_id: str = "default",
    ) -> "RestClient":
        """Create a REST client for checkpoint management and other REST operations.

        The RestClient provides access to various REST endpoints for querying
        checkpoint information and managing checkpoints.

        Args:
            model_id: Model identifier (default: "default")

        Returns:
            RestClient instance

        Example:
            >>> rest_client = service_client.create_rest_client()
            >>> checkpoints = rest_client.list_checkpoints().result()
            >>> for checkpoint in checkpoints.checkpoints:
            ...     print(f"{checkpoint.checkpoint_id}: {checkpoint.checkpoint_type}")
        """
        from xorl_client.client.rest_client import RestClient

        logger.info(f"Creating REST client for model_id: {model_id}")
        return RestClient(holder=self.holder, model_id=model_id)

    def create_training_client_from_state(
        self,
        checkpoint_path: str,
        base_model: Optional[str] = None,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ) -> "TrainingClient":
        """Create a training client from a saved checkpoint (weights only).

        This loads only the model weights, not optimizer state. To also restore
        optimizer state (e.g., Adam momentum), use create_training_client_from_state_with_optimizer.

        Args:
            checkpoint_path: XoRL URI to the checkpoint (e.g., "xorl://default/weights/checkpoint-001")
            base_model: Base model name. If None, will try to query from server.
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha parameter (default: None, uses rank)
            dropout: LoRA dropout (default: 0.0)
            target_modules: Which modules to apply LoRA to

        Returns:
            TrainingClient with loaded weights (optimizer state reset)

        Example:
            >>> training_client = service_client.create_training_client_from_state(
            ...     checkpoint_path="xorl://default/weights/checkpoint-001",
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     rank=32
            ... )
        """
        # Try to get checkpoint info from server if base_model not provided
        if base_model is None:
            try:
                checkpoint_info = self._get_checkpoint_info(checkpoint_path)
                base_model = checkpoint_info.get("base_model")
                rank = checkpoint_info.get("lora_rank", rank)
            except Exception as e:
                logger.warning(f"Could not get checkpoint info: {e}")
                raise ValueError(
                    "base_model is required. Could not retrieve checkpoint metadata from server. "
                    "Please provide base_model explicitly."
                )

        if base_model is None:
            raise ValueError(
                "base_model is required to create training client from state"
            )

        # Extract model_id from checkpoint path to preserve the original model_id
        model_id = _extract_model_id_from_xorl_uri(checkpoint_path)
        if model_id:
            logger.info(f"Using model_id from checkpoint path: {model_id}")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            model_id=model_id,
        )

        # Load weights (without optimizer state)
        training_client.load_state(checkpoint_path).result()
        logger.info(f"Loaded checkpoint from {checkpoint_path} (weights only)")

        return training_client

    def create_training_client_from_state_with_optimizer(
        self,
        checkpoint_path: str,
        base_model: Optional[str] = None,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ) -> "TrainingClient":
        """Create a training client from a saved checkpoint with optimizer state.

        This loads both model weights and optimizer state (e.g., Adam momentum),
        allowing training to resume exactly where it left off.

        Args:
            checkpoint_path: XoRL URI to the checkpoint (e.g., "xorl://default/weights/checkpoint-001")
            base_model: Base model name. If None, will try to query from server.
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha parameter (default: None, uses rank)
            dropout: LoRA dropout (default: 0.0)
            target_modules: Which modules to apply LoRA to

        Returns:
            TrainingClient with loaded weights and optimizer state

        Example:
            >>> training_client = service_client.create_training_client_from_state_with_optimizer(
            ...     checkpoint_path="xorl://default/weights/checkpoint-001",
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     rank=32
            ... )
        """
        # Try to get checkpoint info from server if base_model not provided
        if base_model is None:
            try:
                checkpoint_info = self._get_checkpoint_info(checkpoint_path)
                base_model = checkpoint_info.get("base_model")
                rank = checkpoint_info.get("lora_rank", rank)
            except Exception as e:
                logger.warning(f"Could not get checkpoint info: {e}")
                raise ValueError(
                    "base_model is required. Could not retrieve checkpoint metadata from server. "
                    "Please provide base_model explicitly."
                )

        if base_model is None:
            raise ValueError(
                "base_model is required to create training client from state"
            )

        # Extract model_id from checkpoint path to preserve the original model_id
        model_id = _extract_model_id_from_xorl_uri(checkpoint_path)
        if model_id:
            logger.info(f"Using model_id from checkpoint path: {model_id}")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            model_id=model_id,
        )

        # Load weights WITH optimizer state
        training_client.load_state_with_optimizer(checkpoint_path).result()
        logger.info(f"Loaded checkpoint from {checkpoint_path} (with optimizer state)")

        return training_client

    async def create_training_client_from_state_async(
        self,
        checkpoint_path: str,
        base_model: Optional[str] = None,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ) -> "TrainingClient":
        """Async version of create_training_client_from_state.

        Create a training client from a saved checkpoint (weights only, no optimizer state).

        Args:
            checkpoint_path: XoRL URI to the checkpoint (e.g., "xorl://default/weights/checkpoint-001")
            base_model: Base model name. If None, will try to query from server.
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha parameter (default: None, uses rank)
            dropout: LoRA dropout (default: 0.0)
            target_modules: Which modules to apply LoRA to

        Returns:
            TrainingClient with loaded weights (optimizer state reset)
        """
        # Try to get checkpoint info from server if base_model not provided
        if base_model is None:
            try:
                checkpoint_info = self._get_checkpoint_info(checkpoint_path)
                base_model = checkpoint_info.get("base_model")
                rank = checkpoint_info.get("lora_rank", rank)
            except Exception as e:
                logger.warning(f"Could not get checkpoint info: {e}")
                raise ValueError(
                    "base_model is required. Could not retrieve checkpoint metadata from server. "
                    "Please provide base_model explicitly."
                )

        if base_model is None:
            raise ValueError(
                "base_model is required to create training client from state"
            )

        # Extract model_id from checkpoint path to preserve the original model_id
        model_id = _extract_model_id_from_xorl_uri(checkpoint_path)
        if model_id:
            logger.info(f"Using model_id from checkpoint path: {model_id}")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            model_id=model_id,
        )

        # Load weights (without optimizer state)
        load_future = training_client.load_state(checkpoint_path)
        await load_future.result_async()
        logger.info(f"Loaded checkpoint from {checkpoint_path} (weights only)")

        return training_client

    async def create_training_client_from_state_with_optimizer_async(
        self,
        checkpoint_path: str,
        base_model: Optional[str] = None,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
    ) -> "TrainingClient":
        """Async version of create_training_client_from_state_with_optimizer.

        Create a training client from a saved checkpoint with optimizer state.

        Args:
            checkpoint_path: XoRL URI to the checkpoint (e.g., "xorl://default/weights/checkpoint-001")
            base_model: Base model name. If None, will try to query from server.
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha parameter (default: None, uses rank)
            dropout: LoRA dropout (default: 0.0)
            target_modules: Which modules to apply LoRA to

        Returns:
            TrainingClient with loaded weights and optimizer state
        """
        # Try to get checkpoint info from server if base_model not provided
        if base_model is None:
            try:
                checkpoint_info = self._get_checkpoint_info(checkpoint_path)
                base_model = checkpoint_info.get("base_model")
                rank = checkpoint_info.get("lora_rank", rank)
            except Exception as e:
                logger.warning(f"Could not get checkpoint info: {e}")
                raise ValueError(
                    "base_model is required. Could not retrieve checkpoint metadata from server. "
                    "Please provide base_model explicitly."
                )

        if base_model is None:
            raise ValueError(
                "base_model is required to create training client from state"
            )

        # Extract model_id from checkpoint path to preserve the original model_id
        model_id = _extract_model_id_from_xorl_uri(checkpoint_path)
        if model_id:
            logger.info(f"Using model_id from checkpoint path: {model_id}")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            model_id=model_id,
        )

        # Load weights WITH optimizer state
        load_future = training_client.load_state_with_optimizer(checkpoint_path)
        await load_future.result_async()
        logger.info(f"Loaded checkpoint from {checkpoint_path} (with optimizer state)")

        return training_client

    def _get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get metadata about a checkpoint from the server.

        This queries the /api/v1/weights_info endpoint (same as tinker's RestClient).

        Args:
            checkpoint_path: XoRL URI to the checkpoint

        Returns:
            Dictionary with checkpoint metadata (base_model, lora_rank, etc.)
        """
        try:
            return self.holder.post_sync(
                "/api/v1/weights_info",
                {"xorl_path": checkpoint_path},
                timeout=120,
            )
        except Exception as e:
            logger.warning(f"Failed to get checkpoint info for {checkpoint_path}: {e}")
            raise

    def get_server_capabilities(self) -> types.ServerCapabilities:
        """Get the capabilities of the connected training server.

        Returns:
            ServerCapabilities object containing information about what the server supports.

        Example:
            >>> service_client = ServiceClient(base_url="http://localhost:5555")
            >>> capabilities = service_client.get_server_capabilities()
            >>> print(capabilities.supported_models)
        """
        raise NotImplementedError("get_server_capabilities is not yet implemented.")

    def shutdown(self):
        """Shutdown the service client and cleanup resources."""
        self.holder.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
