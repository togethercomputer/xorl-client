"""
SamplingClient - Client for sampling from inference engine.

This version uses asyncio and httpx for efficient async HTTP requests.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import httpx

from xorl_client import types
from xorl_client.client.api_future import AsyncAPIFuture, wrap_coroutine

logger = logging.getLogger(__name__)


@dataclass
class BatchSampleResult:
    """Result of a batch sampling operation with straggler handling.

    Attributes:
        completed: List of (prompt_index, response) for successful samples
        failed: List of (prompt_index, exception) for failed samples
        cancelled: List of prompt indices that were cancelled (stragglers)
    """

    completed: List[Tuple[int, types.SampleResponse]]
    failed: List[Tuple[int, Exception]]
    cancelled: List[int]


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
        timeout: float = 1800.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize SamplingClient.

        Args:
            base_url: Base URL for the API endpoint (e.g., "http://localhost:6000")
            model_path: Path to saved model weights (e.g., "xorl://model-123/step-100")
            model: Model identifier for API routing (e.g., "sbharti/Qwen/Qwen3-32B-af4738d6")
            api_key: API key for authentication (default: XORL_INFERENCE_API_KEY env var)
            timeout: Request timeout in seconds (default: 120.0)
            max_retries: Maximum number of retries for transient errors (default: 3)
            retry_delay: Initial delay between retries in seconds, doubles each retry (default: 1.0)
        """
        if api_key is None:
            api_key = os.environ.get("XORL_INFERENCE_API_KEY")

        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.model_path = model_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (0-indexed)

        Returns:
            Delay in seconds before the next retry
        """
        # Exponential backoff: delay * 2^attempt
        base_delay = self.retry_delay * (2 ** attempt)
        # Add jitter (±25%) to prevent thundering herd
        jitter = base_delay * 0.25 * (2 * random.random() - 1)
        return base_delay + jitter

    def sample(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
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
        coro = self._sample_async(prompt, sampling_params, num_samples, return_logprobs, lora_path)
        return wrap_coroutine(coro)

    async def _sample_async(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams],
        num_samples: int,
        return_logprobs: bool,
        lora_path: Optional[str] = None,
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
        return_routed_experts = sampling_params.return_routed_experts

        logger.debug(f"Sampling from {self.base_url}, num_samples={num_samples}")

        # Workaround for SGLang bug: n > 1 crashes the server when using LoRA adapters.
        # Additionally, concurrent requests to LoRA endpoints can also crash the server.
        # As a workaround, we make multiple sequential requests instead.
        #if num_samples > 1 and self._lora_name:
        if False:
            logger.debug(f"Using sequential requests workaround for num_samples={num_samples} with LoRA")
            sequences: List[types.SampledSequence] = []
            for i in range(num_samples):
                response = await self._single_sample_request(prompt_data, sampling_params_dict, return_logprobs, return_routed_experts, lora_path)
                sequences.extend(response.sequences)
            return types.SampleResponse(sequences=sequences, meta_info=response.meta_info if response else None)
        else:
            # Single sample or no LoRA - can use native n parameter
            sampling_params_dict["n"] = num_samples
            return await self._batch_sample_request(prompt_data, sampling_params_dict, return_logprobs, num_samples, return_routed_experts, lora_path)

    async def _single_sample_request(
        self,
        prompt_data: dict,
        sampling_params_dict: dict,
        return_logprobs: bool,
        return_routed_experts: bool = False,
        lora_path: Optional[str] = None,
    ) -> types.SampleResponse:
        """Make a single sample request to the inference server (n=1)."""
        # Make a copy and explicitly set n=1 for single sample requests
        params = {**sampling_params_dict, "n": 1}
        return await self._batch_sample_request(prompt_data, params, return_logprobs, num_samples=1, return_routed_experts=return_routed_experts, lora_path=lora_path)

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
        return_routed_experts: bool = False,
        lora_path: Optional[str] = None,
    ) -> types.SampleResponse:
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
        # Use the passed lora_path if provided, otherwise fall back to self._lora_name
        effective_lora = lora_path if lora_path is not None else self._lora_name
        if effective_lora:
            payload["lora_path"] = effective_lora

        # Add return_routed_experts for R3 (Rollout Routing Replay)
        # SGLang expects this at the top level, not inside sampling_params
        if return_routed_experts:
            payload["return_routed_experts"] = True

        # Validate payload before sending
        self._validate_generate_payload(payload)

        logger.debug(f"Request payload: {payload}")

        # Make async request with retry logic for transient errors
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._create_client() as client:
                    response = await client.post("/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()

                # SGLang returns different formats:
                # - Single response (n=1): dict with 'text', 'output_ids', 'meta_info' keys
                # - Batched response (n>1): dict with string keys '0', '1', etc., each containing a response dict
                # - Or sometimes a list of response dicts
                if isinstance(data, list):
                    response_list = data
                elif isinstance(data, dict):
                    # Check if this is a batched response (keys are numeric strings) or single response
                    if "meta_info" in data or "output_ids" in data or "text" in data:
                        # Single response format
                        response_list = [data]
                    else:
                        # Batched response format - dict with '0', '1', etc. keys
                        response_list = [data[str(i)] for i in range(len(data))]
                else:
                    response_list = [data]

                sequences: List[types.SampledSequence] = []
                meta_info = None
                for item in response_list:
                    seq = self._parse_sample_response(item, return_logprobs)
                    sequences.append(seq)
                    # Extract meta_info from first response (for R3 routed_experts)
                    if meta_info is None and "meta_info" in item:
                        meta_info = item["meta_info"]

                return types.SampleResponse(sequences=sequences, meta_info=meta_info)

            except httpx.HTTPStatusError as e:
                # Check if this is a transient error
                error_text = e.response.text
                status_code = e.response.status_code
                # 503 Service Unavailable is transient, as are errors mentioning connection issues
                is_transient = status_code == 503 or any(
                    err in error_text for err in ["ReadError", "ConnectError", "TimeoutError"]
                )

                if is_transient and attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Sampling failed with transient error from {self.base_url}: HTTP {e.response.status_code}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                logger.error(f"Sampling failed from {self.base_url}: HTTP {e.response.status_code} - {error_text}")
                raise RuntimeError(f"Sampling failed: HTTP {e.response.status_code} - {error_text}") from e

            except httpx.TimeoutException as e:
                # Request timed out - retry if we have attempts left
                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Sampling request timed out after {self.timeout}s from {self.base_url}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                logger.error(f"Sampling request timed out after {self.timeout}s from {self.base_url}: {e}")
                raise RuntimeError(
                    f"Sampling failed: Request timed out after {self.timeout} seconds. "
                    f"Try increasing the timeout parameter when creating SamplingClient "
                    f"(e.g., timeout=1800.0 for 30 minutes)."
                ) from e

            except httpx.RequestError as e:
                # Server disconnected or other connection error - retry if we have attempts left
                error_msg = str(e)
                is_disconnect = "disconnected" in error_msg.lower()

                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Sampling request failed (connection error) from {self.base_url}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                if is_disconnect:
                    logger.error(f"SGLang server disconnected after {self.max_retries + 1} attempts. Error: {e}")
                    raise RuntimeError(
                        f"Sampling failed: Server disconnected after {self.max_retries + 1} attempts. "
                        f"The server may be overloaded or restarting. Original error: {e}"
                    ) from e

                logger.error(f"Sampling failed from {self.base_url}: {e}")
                raise RuntimeError(f"Sampling failed: {e}") from e

        # Should not reach here, but just in case
        raise RuntimeError(f"Sampling failed after {self.max_retries + 1} attempts: {last_error}")

    def _parse_sample_response(self, data: dict, return_logprobs: bool) -> types.SampledSequence:
        """Parse one SGLang response without weakening behavior-policy evidence."""
        output_ids = data.get("output_ids")
        if not isinstance(output_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in output_ids
        ):
            raise ValueError("sample response output_ids must be a list of integers")

        meta_info = data.get("meta_info", {})
        if not isinstance(meta_info, dict):
            raise ValueError("sample response meta_info must be an object")
        raw_logprobs = meta_info.get("output_token_logprobs")

        output_logprobs = None
        if return_logprobs:
            if not isinstance(raw_logprobs, list):
                raise ValueError(
                    "sample response is missing output_token_logprobs"
                )
            if len(raw_logprobs) != len(output_ids):
                raise ValueError(
                    "sample response token/logprob cardinality mismatch: "
                    f"{len(output_ids)} tokens != {len(raw_logprobs)} logprobs"
                )
            output_logprobs = []
            for index, (token_id, item) in enumerate(
                zip(output_ids, raw_logprobs, strict=True)
            ):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError(
                        f"output_token_logprobs[{index}] must contain logprob and token ID"
                    )
                value, recorded_token_id = item[0], item[1]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"output_token_logprobs[{index}] has a non-numeric logprob"
                    )
                logprob = float(value)
                if not math.isfinite(logprob):
                    raise ValueError(
                        f"output_token_logprobs[{index}] has a non-finite logprob"
                    )
                if (
                    isinstance(recorded_token_id, bool)
                    or not isinstance(recorded_token_id, int)
                    or recorded_token_id != token_id
                ):
                    raise ValueError(
                        f"output_token_logprobs[{index}] token ID does not match output_ids"
                    )
                output_logprobs.append(logprob)

        # Extract stop reason from SGLang's meta_info
        finish_reason = meta_info.get("finish_reason", {})
        stop_reason: types.StopReason = "length" if finish_reason == "length" else "stop"

        return types.SampledSequence(
            tokens=output_ids,
            logprobs=output_logprobs,
            text=data.get("text", ""),
            stop_reason=stop_reason,
        )

    async def sample_async(
        self,
        prompt: Union[types.ModelInput, str],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
    ) -> types.SampleResponse:
        """Async version of sample that directly returns the response.

        Use this when you're already in an async context and want the result directly.

        Args:
            prompt: Text prompt (ModelInput or string)
            sampling_params: Sampling parameters
            num_samples: Number of samples to generate (default: 1)
            return_logprobs: Whether to return logprobs
            lora_path: Optional runtime override for the LoRA adapter path

        Returns:
            SampleResponse with generated text, tokens, and logprobs

        Example:
            >>> response = await sampling_client.sample_async(prompt, params)
            >>> print(response.text)
        """
        return await self._sample_async(prompt, sampling_params, num_samples, return_logprobs, lora_path)

    async def generate_batch_native_async(
        self,
        input_ids_batch: List[List[int]],
        sampling_params: Union[types.SamplingParams, List[types.SamplingParams]],
        *,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
    ) -> List[types.SampleResponse]:
        """Generate one response per token-ID prompt in one SGLang request.

        A list of sampling parameters enables a distinct ``sampling_seed`` for
        each row. Cardinality and decision-time token/logprob alignment are
        strict because either mismatch would corrupt grouped policy-loss data.
        """

        if not input_ids_batch or not all(
            isinstance(row, list)
            and row
            and all(isinstance(token, int) for token in row)
            for row in input_ids_batch
        ):
            raise ValueError("input_ids_batch must contain non-empty integer rows")
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(input_ids_batch):
                raise ValueError(
                    "sampling parameter cardinality mismatch: "
                    f"{len(sampling_params)} != {len(input_ids_batch)}"
                )
            params_payload: dict | list[dict] = [
                params.to_dict() for params in sampling_params
            ]
            routed_flags = {
                params.return_routed_experts for params in sampling_params
            }
            if len(routed_flags) != 1:
                raise ValueError(
                    "return_routed_experts must be shared across a native batch"
                )
            return_routed_experts = next(iter(routed_flags))
        elif isinstance(sampling_params, types.SamplingParams):
            params_payload = sampling_params.to_dict()
            return_routed_experts = sampling_params.return_routed_experts
        else:
            raise TypeError("sampling_params must be SamplingParams or a list of them")

        payload: dict[str, Any] = {
            "input_ids": input_ids_batch,
            "sampling_params": params_payload,
            "return_logprob": return_logprobs,
        }
        if self._model:
            payload["model"] = self._model
        effective_lora = lora_path if lora_path is not None else self._lora_name
        if effective_lora:
            payload["lora_path"] = effective_lora
        if return_routed_experts:
            payload["return_routed_experts"] = True

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._create_client() as client:
                    response = await client.post("/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()
                if isinstance(data, list):
                    response_list = data
                elif isinstance(data, dict) and any(
                    key in data for key in ("meta_info", "output_ids", "text")
                ):
                    response_list = [data]
                elif isinstance(data, dict) and all(
                    str(index) in data for index in range(len(data))
                ):
                    response_list = [data[str(index)] for index in range(len(data))]
                else:
                    raise ValueError("native batch response must be a list of rows")
                if len(response_list) != len(input_ids_batch):
                    raise ValueError(
                        "native batch response cardinality mismatch: "
                        f"expected {len(input_ids_batch)}, got {len(response_list)}"
                    )
                rows: List[types.SampleResponse] = []
                for item in response_list:
                    if not isinstance(item, dict):
                        raise ValueError("native batch response rows must be objects")
                    sequence = self._parse_sample_response(item, return_logprobs)
                    if return_logprobs and (
                        sequence.logprobs is None
                        or len(sequence.logprobs) != len(sequence.tokens)
                    ):
                        raise ValueError(
                            "native batch output token/logprob alignment mismatch: "
                            f"{len(sequence.tokens)} tokens vs "
                            f"{0 if sequence.logprobs is None else len(sequence.logprobs)} logprobs"
                        )
                    rows.append(
                        types.SampleResponse(
                            sequences=[sequence],
                            meta_info=dict(item.get("meta_info") or {}),
                        )
                    )
                return rows
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (502, 503, 504):
                    raise RuntimeError(
                        f"Sampling failed: HTTP {exc.response.status_code} - {exc.response.text}"
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_error = exc
            if attempt >= self.max_retries:
                break
            await asyncio.sleep(self._get_retry_delay(attempt))
        raise RuntimeError(
            f"Native batch generation failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def generate_batch_native(
        self,
        input_ids_batch: List[List[int]],
        sampling_params: Union[types.SamplingParams, List[types.SamplingParams]],
        *,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
    ) -> AsyncAPIFuture[List[types.SampleResponse]]:
        """Future wrapper for :meth:`generate_batch_native_async`."""

        return wrap_coroutine(
            self.generate_batch_native_async(
                input_ids_batch,
                sampling_params,
                return_logprobs=return_logprobs,
                lora_path=lora_path,
            )
        )

    async def sample_batch_async(
        self,
        prompts: List[Union[types.ModelInput, str]],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        timeout: Optional[float] = None,
    ) -> BatchSampleResult:
        """Sample from multiple prompts concurrently with straggler handling.

        Submits all sampling requests concurrently. If timeout is specified,
        returns whatever has completed when timeout is reached and cancels
        the rest (stragglers).

        Args:
            prompts: List of prompts to sample from
            sampling_params: Sampling parameters (applied to all prompts)
            num_samples: Number of samples per prompt
            return_logprobs: Whether to return logprobs
            timeout: Max time to wait in seconds (None = wait for all)

        Returns:
            BatchSampleResult with completed, failed, and cancelled indices

        Example:
            >>> result = await client.sample_batch_async(
            ...     prompts=["Hello", "World", "Test"],
            ...     timeout=30.0,
            ... )
            >>> for idx, response in result.completed:
            ...     print(f"Prompt {idx}: {response.sequences[0].text}")
            >>> print(f"Cancelled {len(result.cancelled)} stragglers")
        """
        if not prompts:
            return BatchSampleResult(completed=[], failed=[], cancelled=[])

        # Create tasks with index tracking
        tasks = []
        task_to_idx = {}
        for idx, prompt in enumerate(prompts):
            task = asyncio.create_task(
                self._sample_async(prompt, sampling_params, num_samples, return_logprobs)
            )
            tasks.append(task)
            task_to_idx[task] = idx

        # Wait for all tasks with optional timeout
        done, pending = await asyncio.wait(tasks, timeout=timeout)

        # Cancel stragglers
        cancelled_indices = []
        for task in pending:
            task.cancel()
            cancelled_indices.append(task_to_idx[task])

        # Wait for cancellations to complete (avoid warnings)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Collect results with indices
        completed = []
        failed = []
        for task in done:
            idx = task_to_idx[task]
            try:
                result = task.result()
                completed.append((idx, result))
            except Exception as e:
                failed.append((idx, e))

        return BatchSampleResult(
            completed=completed,
            failed=failed,
            cancelled=cancelled_indices
        )

    def sample_batch(
        self,
        prompts: List[Union[types.ModelInput, str]],
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        timeout: Optional[float] = None,
    ) -> AsyncAPIFuture[BatchSampleResult]:
        """Sample from multiple prompts concurrently with straggler handling.

        Returns an AsyncAPIFuture - use .result() for sync or await for async.
        See sample_batch_async() for full documentation.

        Example:
            >>> result = client.sample_batch(prompts, timeout=30.0).result()
            >>> print(f"Got {len(result.completed)}, cancelled {len(result.cancelled)}")
        """
        coro = self.sample_batch_async(
            prompts=prompts,
            sampling_params=sampling_params,
            num_samples=num_samples,
            return_logprobs=return_logprobs,
            timeout=timeout,
        )
        return wrap_coroutine(coro)

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
