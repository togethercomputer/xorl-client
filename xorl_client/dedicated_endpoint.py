"""
Dedicated Endpoint Sampling Client.

A client for sampling from XoRL's dedicated endpoints.
Supports both using an existing endpoint or creating one on-the-fly.

Usage:
    from xorl_client import DedicatedEndpointSamplingClient
    from xorl_client.types import SamplingParams

    # Option 1: Create a new endpoint automatically
    client = DedicatedEndpointSamplingClient(
        provider_api_key=os.environ["PROVIDER_API_KEY"],
        model="Qwen/Qwen3-4B",  # Creates endpoint for this model
    )

    # Option 2: Use an existing endpoint
    client = DedicatedEndpointSamplingClient(
        provider_api_key=os.environ["PROVIDER_API_KEY"],
        endpoint_id="endpoint-xxxxx",  # Use existing endpoint
    )

    # Sample (after training_client.update_dedicated_endpoint() sets the model)
    result = client.sample(
        prompt="Hello, how are you?",
        sampling_params=SamplingParams(max_tokens=100, temperature=0.7),
        return_logprobs=True,
    )

    # Clean up (only deletes if we created the endpoint)
    client.close()
"""

import atexit
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "DedicatedEndpointSamplingClient requires the 'openai' package. "
        "Install with: pip install xorl-client[dedicated-endpoint]"
    ) from e

# Together SDK is optional - only needed for creating/deleting endpoints
try:
    from together import Together
    TOGETHER_SDK_AVAILABLE = True
except ImportError:
    Together = None
    TOGETHER_SDK_AVAILABLE = False

from xorl_client.types import SamplingParams


logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result from sampling."""
    text: str
    tokens: List[int]
    logprobs: List[float] = field(default_factory=list)
    finish_reason: str = "stop"


class DedicatedEndpointSamplingClient:
    """
    Sampling client for provider AI's dedicated endpoints.

    This client can either:
    1. Use an existing dedicated endpoint (provide endpoint_id)
    2. Create a new dedicated endpoint on-the-fly (provide model)

    When creating an endpoint, it will be automatically deleted when close() is called
    or when the process exits (unless keep_endpoint=True).
    """

    def __init__(
        self,
        provider_api_key: str,
        model: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        hardware: str = "1x_nvidia_a100_80gb_sxm",
        keep_endpoint: bool = False,
        base_url: str = "https://api.together.xyz/v1",
    ):
        """
        Initialize the dedicated endpoint sampling client.

        Args:
            provider_api_key: provider API key for authentication
            model: Base model name (e.g., "Qwen/Qwen3-4B"). If provided without
                   endpoint_id, a new dedicated endpoint will be created.
            endpoint_id: Existing endpoint ID to use. If provided, no new endpoint
                        is created.
            hardware: Hardware configuration for new endpoints
                     (default: "1x_nvidia_a100_80gb_sxm")
            keep_endpoint: If True, don't delete the endpoint on close()
                          (only relevant when creating new endpoints)
            base_url: provider API base URL for inference

        Raises:
            ValueError: If neither model nor endpoint_id is provided
            ImportError: If model is provided but 'together' SDK is not installed
        """
        if model is None and endpoint_id is None:
            raise ValueError("Must provide either 'model' or 'endpoint_id'")

        self.provider_api_key = provider_api_key
        self.base_url = base_url
        self.model = model
        self.hardware = hardware
        self.keep_endpoint = keep_endpoint
        self._created_endpoint = False  # Track if we created the endpoint
        self._closed = False

        # OpenAI client for inference
        self.openai_client = OpenAI(
            api_key=provider_api_key,
            base_url=base_url,
        )

        # Together client for endpoint management (only if creating endpoint)
        self._together_client: Optional["Together"] = None

        if endpoint_id is not None:
            # Use existing endpoint
            self.endpoint_id = endpoint_id
            logger.info(f"Using existing endpoint: {endpoint_id}")
        else:
            # Create new endpoint
            self._create_endpoint()

        # Model ID for sampling (set via set_model after update_dedicated_endpoint)
        self.current_model_id: Optional[str] = None

        # Register cleanup on exit
        atexit.register(self._cleanup)

        logger.info(f"DedicatedEndpointSamplingClient initialized (endpoint={self.endpoint_id})")

    def _create_endpoint(self):
        """Create a new dedicated endpoint for the model."""
        if not TOGETHER_SDK_AVAILABLE:
            raise ImportError(
                "Creating dedicated endpoints requires the 'together' package. "
                "Install with: pip install together"
            )

        self._together_client = Together(api_key=self.provider_api_key)

        logger.info(f"Creating dedicated endpoint for model: {self.model}")
        logger.info(f"  Hardware: {self.hardware}")

        endpoint = self._together_client.endpoints.create(
            display_name=f"xorl_client-{int(time.time())}",
            model=self.model,
            hardware=self.hardware,
            min_replicas=1,
            max_replicas=1,
        )

        self.endpoint_id = endpoint.id
        self._created_endpoint = True
        logger.info(f"Created endpoint: {self.endpoint_id}")

        # Wait for endpoint to be ready
        self._wait_for_endpoint()

    def _wait_for_endpoint(self, timeout: int = 600):
        """Wait for the endpoint to be in STARTED state."""
        if self._together_client is None:
            return

        logger.info("Waiting for endpoint to start...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self._together_client.endpoints.get(self.endpoint_id)
            state = status.state

            if state == "STARTED":
                logger.info(f"Endpoint {self.endpoint_id} is ready!")
                return
            elif state in ("FAILED", "ERROR"):
                raise RuntimeError(f"Endpoint failed to start: {state}")

            elapsed = int(time.time() - start_time)
            logger.info(f"  Endpoint state: {state} (elapsed: {elapsed}s)")
            time.sleep(10)

        raise TimeoutError(f"Endpoint {self.endpoint_id} did not start within {timeout}s")

    def set_model(self, model_id: str):
        """
        Set the model ID to use for sampling.

        This should be the provider_model_id returned by
        training_client.update_dedicated_endpoint().

        Args:
            model_id: provider model ID (e.g., "your-org/lora-adapter-12345")
        """
        self.current_model_id = model_id
        logger.info(f"Model set to: {model_id}")

    def sample(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        return_logprobs: bool = False,
    ) -> SampleResult:
        """
        Generate text using the dedicated endpoint.

        Args:
            prompt: The prompt to generate from
            sampling_params: Sampling parameters (max_tokens, temperature, etc.)
            return_logprobs: Whether to return log probabilities

        Returns:
            SampleResult with generated text, tokens, and optionally logprobs

        Raises:
            ValueError: If no model ID is set
            RuntimeError: If generation fails
        """
        if self.current_model_id is None:
            raise ValueError(
                "No model ID set. Call set_model() with the provider_model_id "
                "returned by training_client.update_dedicated_endpoint()"
            )

        if sampling_params is None:
            sampling_params = SamplingParams()

        try:
            response = self.openai_client.completions.create(
                model=self.current_model_id,
                prompt=prompt,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                logprobs=1 if return_logprobs else None,
                echo=False,
            )

            choice = response.choices[0]
            text = choice.text

            # Extract tokens and logprobs if available
            tokens: List[int] = []
            logprobs_list: List[float] = []

            if choice.logprobs is not None and choice.logprobs.tokens is not None:
                if choice.logprobs.token_logprobs is not None:
                    logprobs_list = [
                        lp if lp is not None else 0.0
                        for lp in choice.logprobs.token_logprobs
                    ]

            return SampleResult(
                text=text,
                tokens=tokens,
                logprobs=logprobs_list,
                finish_reason=choice.finish_reason or "stop",
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate from {self.current_model_id}: {e}") from e

    def sample_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        return_logprobs: bool = False,
    ) -> List[SampleResult]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts to generate from
            sampling_params: Sampling parameters
            return_logprobs: Whether to return log probabilities

        Returns:
            List of SampleResult, one per prompt
        """
        results = []
        for prompt in prompts:
            result = self.sample(prompt, sampling_params, return_logprobs)
            results.append(result)
        return results

    def health_check(self) -> bool:
        """
        Check if the endpoint is healthy by attempting a small generation.

        Returns:
            bool: True if endpoint is responding
        """
        if self.current_model_id is None:
            logger.warning("No model ID set, cannot health check")
            return False

        try:
            self.openai_client.completions.create(
                model=self.current_model_id,
                prompt="hi",
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def _cleanup(self):
        """Clean up resources (delete endpoint if we created it)."""
        if self._closed:
            return

        if self._created_endpoint and not self.keep_endpoint:
            self._delete_endpoint()

        self._closed = True

    def _delete_endpoint(self):
        """Delete the dedicated endpoint."""
        if self._together_client is None:
            return

        try:
            self._together_client.endpoints.delete(self.endpoint_id)
            logger.info(f"Deleted endpoint: {self.endpoint_id}")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint {self.endpoint_id}: {e}")

    def close(self):
        """
        Close the client and clean up resources.

        If we created the endpoint and keep_endpoint is False, the endpoint
        will be deleted.
        """
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()
        return False

    @property
    def endpoint_ready(self) -> bool:
        """Check if the endpoint is ready for use."""
        return self.endpoint_id is not None
