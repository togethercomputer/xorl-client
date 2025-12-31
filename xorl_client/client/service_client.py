"""
ServiceClient - Main entry point for XoRL API.

Provides factory methods to create TrainingClient and SamplingClient instances.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from xorl_client import types
from xorl_client.client.client_holder import ClientHolder

logger = logging.getLogger(__name__)


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
        ...     model_path="xorl://model-123/step-100"
        ... )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        """Initialize ServiceClient.

        Args:
            base_url: Base URL for the training API (default: XORL_BASE_URL env var or "http://localhost:5555")
            api_key: API key for authentication (default: XORL_API_KEY env var)
            timeout: Default timeout for requests
            **kwargs: Additional arguments
        """
        if base_url is None:
            base_url = os.environ.get("XORL_BASE_URL", "http://localhost:5555")
        if api_key is None:
            api_key = os.environ.get("XORL_API_KEY")

        self.holder = ClientHolder(base_url=base_url, api_key=api_key, timeout=timeout, **kwargs)
        logger.info(f"ServiceClient initialized: base_url={base_url}")

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
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

        Returns:
            TrainingClient instance

        Example:
            >>> training_client = service_client.create_lora_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct",
            ...     rank=32
            ... )
        """
        from xorl_client.client.training_client import TrainingClient

        # Use default model ID - xorl server always uses "default" for single-model setup
        model_id = "default"

        # Create LoRA config
        lora_config = types.LoraConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

        # Send create model request to server
        logger.info(f"Creating LoRA training client: model_id={model_id}, base_model={base_model}, rank={rank}")

        try:
            response = self.holder.post(
                "/api/v1/create_model",
                {
                    "model_id": model_id,
                    "base_model": base_model,
                    "lora_config": lora_config.to_dict(),
                },
            )
            logger.info(f"Model created: {response}")
        except RuntimeError as e:
            logger.error(f"Failed to create model: {e}")
            raise

        # Create and return TrainingClient
        return TrainingClient(holder=self.holder, model_id=model_id, base_model=base_model)

    def create_sampling_client(
        self,
        model_path: str,
    ) -> "SamplingClient":
        """Create a sampling client for a specific model.

        The model_path is an opaque identifier (e.g., "xorl://model-123/step-100")
        that the service uses to route requests to the correct inference workers.

        Args:
            model_path: Model path identifier

        Returns:
            SamplingClient instance

        Example:
            >>> sampling_client = service_client.create_sampling_client(
            ...     model_path="xorl://model-123/step-100"
            ... )
        """
        from xorl_client.client.sampling_client import SamplingClient

        logger.info(f"Creating sampling client for model_path: {model_path}")
        return SamplingClient(holder=self.holder, model_path=model_path)

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
            raise ValueError("base_model is required to create training client from state")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
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
            raise ValueError("base_model is required to create training client from state")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
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
            raise ValueError("base_model is required to create training client from state")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
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
            raise ValueError("base_model is required to create training client from state")

        # Create training client with the checkpoint's config
        training_client = self.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
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
            return self.holder.post(
                "/api/v1/weights_info",
                {"xorl_path": checkpoint_path},
                timeout=30,
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
