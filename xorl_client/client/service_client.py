"""
ServiceClient - Main entry point for XoRL API.

Provides factory methods to create TrainingClient and SamplingClient instances.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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
        base_url: Base URL for the XoRL training API (default: "http://localhost:5555")
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
        base_url: str = "http://localhost:5555",
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        """Initialize ServiceClient.

        Args:
            base_url: Base URL for the training API
            timeout: Default timeout for requests
            **kwargs: Additional arguments
        """
        self.holder = ClientHolder(base_url=base_url, timeout=timeout, **kwargs)
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

        # Generate model ID
        model_id = self.holder.get_model_id()

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
        return TrainingClient(holder=self.holder, model_id=model_id)

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

    def create_training_client_from_state(
        self,
        checkpoint_path: str,
    ) -> "TrainingClient":
        """Create a training client from a saved checkpoint.

        This is a convenience method that:
        1. Loads checkpoint metadata to determine base_model, rank, etc.
        2. Creates a new TrainingClient
        3. Loads the checkpoint weights

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            TrainingClient with loaded weights

        Example:
            >>> training_client = service_client.create_training_client_from_state(
            ...     checkpoint_path="/tmp/checkpoints/step-100"
            ... )
        """
        from xorl_client.client.training_client import TrainingClient

        # TODO: Query checkpoint metadata to get base_model, rank
        # For now, we'll need to pass these as arguments or read from checkpoint
        logger.warning(
            "create_training_client_from_state is not fully implemented yet. "
            "Use create_lora_training_client() then call load_state() manually."
        )

        # Placeholder implementation
        raise NotImplementedError(
            "create_training_client_from_state is not yet implemented. "
            "Create a training client and call load_state() manually."
        )

    def shutdown(self):
        """Shutdown the service client and cleanup resources."""
        self.holder.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
