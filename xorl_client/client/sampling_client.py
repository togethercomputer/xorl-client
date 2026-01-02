"""
SamplingClient - Client for sampling from inference engine.

This version uses asyncio and httpx for efficient async HTTP requests.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Union

import httpx

from xorl_client import types
from xorl_client.client.api_future import AsyncAPIFuture, wrap_coroutine

logger = logging.getLogger(__name__)


class SamplingClient:
    """Client for sampling from inference engine.

    Uses asyncio and httpx for efficient concurrent requests.
    Returns AsyncAPIFuture that supports both sync (.result()) and async (await) usage.

    Example:
        >>> sampling_client = SamplingClient(
        ...     base_url="http://localhost:30000",
        ...     model_path="xorl://model-123/step-100"
        ... )
        >>> # Sync usage
        >>> future = sampling_client.sample(
        ...     prompt="The capital of France is",
        ...     sampling_params=types.SamplingParams(max_tokens=20, temperature=0.8)
        ... )
        >>> response = future.result()
        >>> print(response.text)

        >>> # Async usage
        >>> response = await sampling_client.sample(prompt, params)

        >>> # Multiple concurrent requests
        >>> futures = [sampling_client.sample(prompt, params) for _ in range(4)]
        >>> responses = [f.result() for f in futures]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:6000",
        model_path: str = "",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Initialize SamplingClient.

        Args:
            base_url: Base URL for the API endpoint (e.g., "http://localhost:6000")
            model_path: Path to saved model weights (e.g., "xorl://model-123/step-100")
            model: Model identifier for API routing (e.g., "sbharti/Qwen/Qwen3-32B-af4738d6")
            api_key: API key for authentication (default: XORL_INFERENCE_API_KEY env var)
            timeout: Request timeout in seconds (default: 120.0)
        """
        if api_key is None:
            api_key = os.environ.get("XORL_INFERENCE_API_KEY")

        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.model_path = model_path
        self.timeout = timeout

        # Headers for authentication
        self._headers = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        # Extract the adapter name from model_path for SGLang
        # xorl://model_id/sampler_weights/checkpoint_name -> checkpoint_name
        self._lora_name = self._extract_lora_name(model_path)

        logger.info(f"SamplingClient initialized: base_url={self.base_url}, model={self._model}, model_path={self.model_path}, lora_name={self._lora_name}")

    @staticmethod
    def _extract_lora_name(model_path: str) -> Optional[str]:
        """Extract the LoRA adapter name from a xorl:// model path.

        SGLang's /generate endpoint expects just the adapter name, not the full xorl:// path.
        This matches how xorl's create_sampling_session registers adapters with SGLang.

        Args:
            model_path: Full path like "xorl://default/sampler_weights/test-001"

        Returns:
            Just the adapter name, e.g., "test-001"
        """
        if not model_path:
            return None

        if model_path.startswith("xorl://"):
            # Format: xorl://model_id/sampler_weights/checkpoint_name
            parts = model_path[7:].split("/")
            if len(parts) >= 3:
                # checkpoint_name is everything after model_id/sampler_weights/
                return "/".join(parts[2:])
            return None
        elif model_path.startswith("sampler_weights/"):
            # Format: sampler_weights/checkpoint_name
            return model_path[len("sampler_weights/"):]
        else:
            # Assume it's already just the adapter name
            return model_path

    def _create_client(self) -> httpx.AsyncClient:
        """Create a new async HTTP client.

        Note: We create a fresh client for each request instead of caching because
        httpx.AsyncClient is tied to the event loop it was created in. When using
        AsyncAPIFuture.result() from a Jupyter notebook or other async environment,
        the coroutine may run in a different thread with a different event loop.
        """
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=httpx.Timeout(self.timeout),
        )

    def sample(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
    ) -> AsyncAPIFuture[types.SampleResponse]:
        """Sample from the model.

        Returns an AsyncAPIFuture that can be awaited or called with .result().

        Args:
            prompt: Text prompt (ModelInput or string)
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate (default: 1)
            return_logprobs: Whether to return logprobs

        Returns:
            AsyncAPIFuture[SampleResponse] - use .result() or await to get response

        Example:
            >>> # Sync usage
            >>> future = sampling_client.sample(prompt, params)
            >>> response = future.result()

            >>> # Async usage
            >>> response = await sampling_client.sample(prompt, params)

            >>> # Multiple concurrent requests
            >>> futures = [sampling_client.sample(prompt, params) for _ in range(4)]
            >>> responses = [f.result() for f in futures]
        """
        # Create the coroutine and wrap it
        coro = self._sample_async(prompt, sampling_params, num_samples, return_logprobs)
        return wrap_coroutine(coro)

    async def _sample_async(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams],
        num_samples: int,
        return_logprobs: bool,
    ) -> types.SampleResponse:
        """Internal async implementation of sample.

        Note: SGLang has a bug where n > 1 crashes the server when using LoRA adapters.
        As a workaround, when num_samples > 1, we make multiple concurrent requests
        instead of using the native parallel sampling feature.
        """
        # Convert prompt to the format expected by the inference engine (SGLang)
        # SGLang expects either "text" (string) or "input_ids" (list of token IDs)
        if isinstance(prompt, str):
            # For string prompts, send as text directly
            prompt_data = {"text": prompt}
        else:
            # For ModelInput, convert to token IDs
            # ModelInput uses chunks structure, but SGLang expects flat token IDs
            try:
                input_ids = prompt.to_ints()
                prompt_data = {"input_ids": input_ids}
            except ValueError as e:
                # to_ints() fails if there are non-text chunks (e.g., images)
                raise ValueError(
                    f"Cannot convert ModelInput to token IDs for inference: {e}. "
                    "The inference server may not support multimodal inputs."
                ) from e

        # Convert sampling params to dict
        if sampling_params is None:
            sampling_params = types.SamplingParams()
        sampling_params_dict = sampling_params.to_dict()

        logger.debug(f"Sampling from {self.base_url}, num_samples={num_samples}")

        # Workaround for SGLang bug: n > 1 crashes the server when using LoRA adapters.
        # Additionally, concurrent requests to LoRA endpoints can also crash the server.
        # As a workaround, we make multiple sequential requests instead.
        #if num_samples > 1 and self._lora_name:
        if False:
            logger.debug(f"Using sequential requests workaround for num_samples={num_samples} with LoRA")
            sequences: List[types.SampledSequence] = []
            for i in range(num_samples):
                seq = await self._single_sample_request(prompt_data, sampling_params_dict, return_logprobs)
                sequences.append(seq)
            return types.SampleResponse(sequences=sequences)
        else:
            # Single sample or no LoRA - can use native n parameter
            sampling_params_dict["n"] = num_samples
            sequences = await self._batch_sample_request(prompt_data, sampling_params_dict, return_logprobs, num_samples)
            return types.SampleResponse(sequences=sequences)

    async def _single_sample_request(
        self,
        prompt_data: dict,
        sampling_params_dict: dict,
        return_logprobs: bool,
    ) -> types.SampledSequence:
        """Make a single sample request to the inference server (n=1)."""
        # Make a copy and explicitly set n=1 for single sample requests
        params = {**sampling_params_dict, "n": 1}
        sequences = await self._batch_sample_request(prompt_data, params, return_logprobs, num_samples=1)
        return sequences[0]

    def _validate_generate_payload(self, payload: dict) -> None:
        """Validate the /generate request payload before sending to server.

        Args:
            payload: The request payload dict

        Raises:
            ValueError: If the payload is invalid
        """
        # Must have either text or input_ids, but not both
        has_text = "text" in payload and payload["text"] is not None
        has_input_ids = "input_ids" in payload and payload["input_ids"] is not None

        if not has_text and not has_input_ids:
            raise ValueError("Request must have either 'text' or 'input_ids'")

        if has_text and has_input_ids:
            raise ValueError("Request cannot have both 'text' and 'input_ids'")

        # Validate text format
        if has_text:
            text = payload["text"]
            if not isinstance(text, str):
                raise ValueError(f"'text' must be a string, got {type(text).__name__}")
            if not text:
                raise ValueError("'text' cannot be empty")

        # Validate input_ids format
        if has_input_ids:
            input_ids = payload["input_ids"]
            if not isinstance(input_ids, list):
                raise ValueError(f"'input_ids' must be a list, got {type(input_ids).__name__}")
            if not input_ids:
                raise ValueError("'input_ids' cannot be empty")
            if not all(isinstance(x, int) for x in input_ids):
                raise ValueError("'input_ids' must be a list of integers")

        # Validate sampling_params
        if "sampling_params" in payload:
            params = payload["sampling_params"]
            if not isinstance(params, dict):
                raise ValueError(f"'sampling_params' must be a dict, got {type(params).__name__}")

            # Validate specific sampling params
            if "max_new_tokens" in params:
                max_tokens = params["max_new_tokens"]
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    raise ValueError(f"'max_new_tokens' must be a positive integer, got {max_tokens}")

            if "temperature" in params:
                temp = params["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0:
                    raise ValueError(f"'temperature' must be a non-negative number, got {temp}")

            if "n" in params:
                n = params["n"]
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"'n' must be a positive integer, got {n}")

            if "stop" in params:
                stop = params["stop"]
                if not isinstance(stop, list):
                    raise ValueError(f"'stop' must be a list of strings, got {type(stop).__name__}")
                if not all(isinstance(s, str) for s in stop):
                    raise ValueError("'stop' must be a list of strings")

        # Validate lora_path
        if "lora_path" in payload and payload["lora_path"] is not None:
            lora_path = payload["lora_path"]
            if not isinstance(lora_path, str):
                raise ValueError(f"'lora_path' must be a string, got {type(lora_path).__name__}")

    async def _batch_sample_request(
        self,
        prompt_data: dict,
        sampling_params_dict: dict,
        return_logprobs: bool,
        num_samples: int,
    ) -> List[types.SampledSequence]:
        """Make a sample request to the inference server, handling both single and batch responses.

        SGLang returns:
        - A single dict when n=1
        - A list of dicts when n>1
        """
        # Build request payload
        # SGLang expects: input_ids/text, sampling_params (dict), return_logprob, lora_path
        payload = {
            **prompt_data,
            "sampling_params": sampling_params_dict,
            "return_logprob": return_logprobs,
        }

        # Add model for API routing (if set)
        if self._model:
            payload["model"] = self._model

        # Add LoRA adapter name for routing (SGLang expects the registered adapter name, not the full path)
        if self._lora_name:
            payload["lora_path"] = self._lora_name

        # Validate payload before sending
        self._validate_generate_payload(payload)

        logger.debug(f"Request payload: {payload}")

        # Make async request
        try:
            async with self._create_client() as client:
                response = await client.post("/generate", json=payload)
                response.raise_for_status()
                data = response.json()

            # SGLang returns a list when n>1, single dict when n=1
            if isinstance(data, list):
                response_list = data
            else:
                response_list = [data]

            sequences: List[types.SampledSequence] = []
            for item in response_list:
                seq = self._parse_sample_response(item, return_logprobs)
                sequences.append(seq)

            return sequences

        except httpx.HTTPStatusError as e:
            # Check if this is a transient error (ReadError, ConnectError, TimeoutError)
            error_text = e.response.text
            is_transient = any(err in error_text for err in ["ReadError", "ConnectError", "TimeoutError"])

            if is_transient:
                logger.warning(f"Sampling failed with transient error from {self.base_url}: HTTP {e.response.status_code}")
            else:
                logger.error(f"Sampling failed from {self.base_url}: HTTP {e.response.status_code} - {error_text}")

            raise RuntimeError(f"Sampling failed: HTTP {e.response.status_code} - {error_text}") from e
        except httpx.RequestError as e:
            # Server disconnected - likely crashed due to LoRA bug
            error_msg = str(e)
            if "disconnected" in error_msg.lower():
                logger.error(f"SGLang server crashed during request. This may be due to a known LoRA bug. Error: {e}")
                raise RuntimeError(
                    f"Sampling failed: SGLang server disconnected (likely crashed). "
                    f"If using LoRA, try reducing num_samples or restarting the inference server. "
                    f"Original error: {e}"
                ) from e
            logger.error(f"Sampling failed from {self.base_url}: {e}")
            raise RuntimeError(f"Sampling failed: {e}") from e

    def _parse_sample_response(self, data: dict, return_logprobs: bool) -> types.SampledSequence:
        """Parse a single sample response - just pass through the fields directly."""
        raw_logprobs = data["meta_info"].get("output_token_logprobs")
        # output_token_logprobs is a list of [logprob, token_id, ???] tuples
        # Extract just the logprob (first element) from each tuple
        output_logprobs = None
        if raw_logprobs is not None:
            output_logprobs = [item[0] if item[0] is not None else 0.0 for item in raw_logprobs]
        return types.SampledSequence(
            tokens=data.get("output_ids", []),
            logprobs=output_logprobs,
            text=data.get("text", ""),
        )

    async def sample_async(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
    ) -> types.SampleResponse:
        """Async version of sample that directly returns the response.

        Use this when you're already in an async context and want the result directly.

        Args:
            prompt: Text prompt (ModelInput or string)
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate (default: 1)
            return_logprobs: Whether to return logprobs

        Returns:
            SampleResponse with generated text, tokens, and logprobs

        Example:
            >>> response = await sampling_client.sample_async(prompt, params)
            >>> print(response.text)
        """
        return await self._sample_async(prompt, sampling_params, num_samples, return_logprobs)

    def close(self):
        """Close the client (no-op since we create fresh clients per request)."""
        pass

    async def close_async(self):
        """Close the client (no-op since we create fresh clients per request)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
