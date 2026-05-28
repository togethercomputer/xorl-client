"""
SamplingClient - Client for sampling from inference engine.

This version uses asyncio and httpx for efficient async HTTP requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Tuple, Union

import httpx

from xorl_client import types
from xorl_client.client.api_future import AsyncAPIFuture, wrap_coroutine

logger = logging.getLogger(__name__)

PromptInput = Union[types.ModelInput, str, List[dict], List[object]]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r", name, value)
        return default
    if parsed < minimum:
        logger.warning("Ignoring env %s=%r below minimum %s", name, value, minimum)
        return default
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid float env %s=%r", name, value)
        return default


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
        api_format: Optional[str] = None,
        http_client_mode: Optional[str] = None,
        http_max_connections: Optional[int] = None,
        http_max_keepalive_connections: Optional[int] = None,
        http_keepalive_expiry: Optional[float] = None,
        http2: Optional[bool] = None,
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
            api_format: Inference API format: "generate" for native SGLang /generate,
                or "chat_completions" for OpenAI-compatible /v1/chat/completions.
                Defaults to XORL_INFERENCE_API_FORMAT or "generate".
            http_client_mode: "request" creates a fresh HTTP client per request
                (legacy behavior); "pooled" reuses one AsyncClient per event loop.
                Defaults to XORL_SAMPLING_HTTP_CLIENT_MODE or "request".
            http_max_connections: Maximum concurrent connections per HTTP client.
                Defaults to XORL_SAMPLING_HTTP_MAX_CONNECTIONS or 1024.
            http_max_keepalive_connections: Maximum idle keep-alive connections per
                HTTP client. Set 0 for Dispatch runs that prefer no keep-alive reuse.
                Defaults to XORL_SAMPLING_HTTP_MAX_KEEPALIVE_CONNECTIONS or 256.
            http_keepalive_expiry: Keep-alive expiry in seconds. Defaults to
                XORL_SAMPLING_HTTP_KEEPALIVE_EXPIRY or 5.0.
            http2: Whether to enable HTTP/2 in httpx. Defaults to
                XORL_SAMPLING_HTTP2 or False.
        """
        if api_key is None:
            api_key = os.environ.get("XORL_INFERENCE_API_KEY")
        if api_format is None:
            api_format = os.environ.get("XORL_INFERENCE_API_FORMAT", "generate")

        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.model_path = model_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_format = self._normalize_api_format(api_format)
        mode = http_client_mode or os.environ.get(
            "XORL_SAMPLING_HTTP_CLIENT_MODE", "request"
        )
        self.http_client_mode = mode.strip().lower()
        if self.http_client_mode not in {"request", "pooled"}:
            raise ValueError(
                "http_client_mode must be 'request' or 'pooled', "
                f"got {http_client_mode!r}"
            )
        self.http_max_connections = (
            http_max_connections
            if http_max_connections is not None
            else _env_int("XORL_SAMPLING_HTTP_MAX_CONNECTIONS", 1024, minimum=1)
        )
        self.http_max_keepalive_connections = (
            http_max_keepalive_connections
            if http_max_keepalive_connections is not None
            else _env_int(
                "XORL_SAMPLING_HTTP_MAX_KEEPALIVE_CONNECTIONS", 256, minimum=0
            )
        )
        self.http_keepalive_expiry = (
            http_keepalive_expiry
            if http_keepalive_expiry is not None
            else _env_float("XORL_SAMPLING_HTTP_KEEPALIVE_EXPIRY", 5.0)
        )
        self.http2 = (
            http2 if http2 is not None else _env_bool("XORL_SAMPLING_HTTP2", False)
        )
        self._pooled_clients: dict[int, httpx.AsyncClient] = {}

        # Headers for authentication
        self._headers = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        # Extract the adapter name from model_path for SGLang
        # xorl://model_id/sampler_weights/checkpoint_name -> checkpoint_name
        self._lora_name = self._extract_lora_name(model_path)

        logger.info(
            f"SamplingClient initialized: base_url={self.base_url}, model={self._model}, "
            f"model_path={self.model_path}, lora_name={self._lora_name}, api_format={self.api_format}, "
            f"http_client_mode={self.http_client_mode}, http_max_connections={self.http_max_connections}, "
            f"http_max_keepalive_connections={self.http_max_keepalive_connections}"
        )

    @staticmethod
    def _normalize_api_format(api_format: str) -> str:
        """Normalize user-facing API format names."""
        normalized = api_format.strip().lower().replace("-", "_")
        aliases = {
            "chat": "chat_completions",
            "openai": "chat_completions",
            "openai_chat": "chat_completions",
            "chat_completion": "chat_completions",
            "chat_completions": "chat_completions",
            "generate": "generate",
            "sglang": "generate",
            "native": "generate",
        }
        if normalized not in aliases:
            valid = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unsupported api_format {api_format!r}; expected one of: {valid}"
            )
        return aliases[normalized]

    @staticmethod
    def _looks_like_messages(prompt: Any) -> bool:
        if not isinstance(prompt, list) or not prompt:
            return False
        return all(
            isinstance(message, dict)
            or (hasattr(message, "role") and hasattr(message, "content"))
            for message in prompt
        )

    @staticmethod
    def _normalize_messages(prompt: list[Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("chat messages must include string role and content")
            messages.append({"role": role, "content": content})
        return messages

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
            return model_path[len("sampler_weights/") :]
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
            limits=httpx.Limits(
                max_connections=self.http_max_connections,
                max_keepalive_connections=self.http_max_keepalive_connections,
                keepalive_expiry=self.http_keepalive_expiry,
            ),
            http2=self.http2,
        )

    def _pooled_client(self) -> httpx.AsyncClient:
        """Return the AsyncClient for the current event loop.

        httpx AsyncClient instances are tied to the event loop that uses them.
        Pipeline RL can sample from both the main loop and a generation thread, so
        pooled mode keeps one client per loop instead of sharing a single client
        across threads.
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        client = self._pooled_clients.get(loop_id)
        if client is None or client.is_closed:
            client = self._create_client()
            self._pooled_clients[loop_id] = client
        return client

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.http_client_mode == "pooled":
            yield self._pooled_client()
            return

        async with self._create_client() as client:
            yield client

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (0-indexed)

        Returns:
            Delay in seconds before the next retry
        """
        # Exponential backoff: delay * 2^attempt
        base_delay = self.retry_delay * (2**attempt)
        # Add jitter (±25%) to prevent thundering herd
        jitter = base_delay * 0.25 * (2 * random.random() - 1)
        return base_delay + jitter

    def sample(
        self,
        prompt: PromptInput,
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> AsyncAPIFuture[types.SampleResponse]:
        """Sample from the model.

        Returns an AsyncAPIFuture that can be awaited or called with .result().

        Args:
            prompt: Text prompt, ModelInput, or chat messages
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
        coro = self._sample_async(
            prompt,
            sampling_params,
            num_samples,
            return_logprobs,
            lora_path,
            logprob_start_len=logprob_start_len,
        )
        return wrap_coroutine(coro)

    async def _sample_async(
        self,
        prompt: PromptInput,
        sampling_params: Optional[types.SamplingParams],
        num_samples: int,
        return_logprobs: bool,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> types.SampleResponse:
        """Internal async implementation of sample.

        Note: SGLang has a bug where n > 1 crashes the server when using LoRA adapters.
        As a workaround, when num_samples > 1, we make multiple concurrent requests
        instead of using the native parallel sampling feature.
        """
        # Convert prompt to the format expected by the inference engine.
        # Native SGLang /generate accepts token IDs. OpenAI chat-completions
        # requires messages, so callers that already have structured chat turns
        # should pass those instead of a rendered ModelInput.
        if isinstance(prompt, str):
            prompt_data = {"text": prompt}
        elif self._looks_like_messages(prompt):
            prompt_data = {"messages": self._normalize_messages(prompt)}
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
        return_expert_logits = sampling_params.return_expert_logits

        logger.debug(f"Sampling from {self.base_url}, num_samples={num_samples}")

        if self.api_format == "chat_completions":
            return await self._chat_completions_sample_request(
                prompt_data,
                sampling_params,
                return_logprobs,
                num_samples,
                return_routed_experts,
                return_expert_logits,
                lora_path,
                logprob_start_len=logprob_start_len,
            )

        # Workaround for SGLang bug: n > 1 crashes the server when using LoRA adapters.
        # Additionally, concurrent requests to LoRA endpoints can also crash the server.
        # As a workaround, we make multiple sequential requests instead.
        # if num_samples > 1 and self._lora_name:
        if False:
            logger.debug(
                f"Using sequential requests workaround for num_samples={num_samples} with LoRA"
            )
            sequences: List[types.SampledSequence] = []
            for i in range(num_samples):
                response = await self._single_sample_request(
                    prompt_data,
                    sampling_params_dict,
                    return_logprobs,
                    return_routed_experts,
                    return_expert_logits,
                    lora_path,
                    logprob_start_len=logprob_start_len,
                )
                sequences.extend(response.sequences)
            return types.SampleResponse(
                sequences=sequences, meta_info=response.meta_info if response else None
            )
        else:
            # Single sample or no LoRA - can use native n parameter
            sampling_params_dict["n"] = num_samples
            return await self._batch_sample_request(
                prompt_data,
                sampling_params_dict,
                return_logprobs,
                num_samples,
                return_routed_experts,
                return_expert_logits,
                lora_path,
                logprob_start_len=logprob_start_len,
            )

    async def _single_sample_request(
        self,
        prompt_data: dict,
        sampling_params_dict: dict,
        return_logprobs: bool,
        return_routed_experts: bool = False,
        return_expert_logits: bool = False,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> types.SampleResponse:
        """Make a single sample request to the inference server (n=1)."""
        # Make a copy and explicitly set n=1 for single sample requests
        params = {**sampling_params_dict, "n": 1}
        return await self._batch_sample_request(
            prompt_data,
            params,
            return_logprobs,
            num_samples=1,
            return_routed_experts=return_routed_experts,
            return_expert_logits=return_expert_logits,
            lora_path=lora_path,
            logprob_start_len=logprob_start_len,
        )

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
        input_ids_is_batch = False
        if has_input_ids:
            input_ids = payload["input_ids"]
            if not isinstance(input_ids, list):
                raise ValueError(
                    f"'input_ids' must be a list, got {type(input_ids).__name__}"
                )
            if not input_ids:
                raise ValueError("'input_ids' cannot be empty")
            if all(isinstance(x, int) for x in input_ids):
                input_ids_is_batch = False
            elif all(isinstance(x, list) for x in input_ids):
                input_ids_is_batch = True
                if not all(
                    seq and all(isinstance(tok, int) for tok in seq)
                    for seq in input_ids
                ):
                    raise ValueError(
                        "'input_ids' batch must be a non-empty list of non-empty integer lists"
                    )
            else:
                raise ValueError(
                    "'input_ids' must be either a list of integers or a list of integer lists"
                )

        # Validate top-level logprob_start_len (used for prompt scoring)
        if "logprob_start_len" in payload and payload["logprob_start_len"] is not None:
            logprob_start_len = payload["logprob_start_len"]
            if isinstance(logprob_start_len, int):
                if logprob_start_len < 0:
                    raise ValueError("'logprob_start_len' must be >= 0")
            elif isinstance(logprob_start_len, list):
                if not logprob_start_len:
                    raise ValueError("'logprob_start_len' list cannot be empty")
                if not all(isinstance(v, int) and v >= 0 for v in logprob_start_len):
                    raise ValueError(
                        "'logprob_start_len' list must contain non-negative integers"
                    )
                if has_input_ids and not input_ids_is_batch:
                    raise ValueError(
                        "'logprob_start_len' list requires batched 'input_ids'"
                    )
                if has_input_ids and input_ids_is_batch:
                    input_ids_batch = payload["input_ids"]
                    if len(logprob_start_len) != len(input_ids_batch):
                        raise ValueError(
                            f"'logprob_start_len' length ({len(logprob_start_len)}) must match "
                            f"batched 'input_ids' length ({len(input_ids_batch)})"
                        )
            else:
                raise ValueError(
                    f"'logprob_start_len' must be an int or list of ints, got {type(logprob_start_len).__name__}"
                )

        # Validate sampling_params
        if "sampling_params" in payload:
            params = payload["sampling_params"]
            if not isinstance(params, dict):
                raise ValueError(
                    f"'sampling_params' must be a dict, got {type(params).__name__}"
                )

            # Validate specific sampling params
            if "max_new_tokens" in params:
                max_tokens = params["max_new_tokens"]
                if not isinstance(max_tokens, int) or max_tokens < 0:
                    raise ValueError(
                        f"'max_new_tokens' must be a non-negative integer, got {max_tokens}"
                    )

            if "temperature" in params:
                temp = params["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0:
                    raise ValueError(
                        f"'temperature' must be a non-negative number, got {temp}"
                    )

            if "n" in params:
                n = params["n"]
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"'n' must be a positive integer, got {n}")

            if "stop" in params:
                stop = params["stop"]
                if not isinstance(stop, list):
                    raise ValueError(
                        f"'stop' must be a list of strings, got {type(stop).__name__}"
                    )
                if not all(isinstance(s, str) for s in stop):
                    raise ValueError("'stop' must be a list of strings")

        # Validate lora_path
        if "lora_path" in payload and payload["lora_path"] is not None:
            lora_path = payload["lora_path"]
            if not isinstance(lora_path, str):
                raise ValueError(
                    f"'lora_path' must be a string, got {type(lora_path).__name__}"
                )

    def _build_chat_completions_payload(
        self,
        prompt_data: dict,
        sampling_params: types.SamplingParams,
        return_logprobs: bool,
        num_samples: int,
        return_routed_experts: bool = False,
        return_expert_logits: bool = False,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> dict:
        """Build an OpenAI-compatible chat-completions payload for SGLang/Dispatch."""
        if num_samples <= 0:
            raise ValueError(
                f"'num_samples' must be a positive integer, got {num_samples}"
            )
        if sampling_params.max_tokens <= 0:
            raise ValueError(
                "chat_completions api_format requires SamplingParams.max_tokens > 0; "
                "use api_format='generate' for prompt scoring requests."
            )
        if isinstance(logprob_start_len, list):
            raise ValueError(
                "chat_completions api_format only supports an int logprob_start_len for a single prompt"
            )

        params = sampling_params.to_dict()
        payload: dict[str, Any] = {
            "model": self._model or "default",
            "max_tokens": params["max_new_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "n": num_samples,
            "logprobs": return_logprobs,
        }

        if "messages" in prompt_data:
            payload["messages"] = prompt_data["messages"]
        elif "input_ids" in prompt_data:
            raise ValueError(
                "chat_completions api_format requires a string prompt or a list of chat messages; "
                "ModelInput/token-id prompts are not accepted by SGLang's /v1/chat/completions endpoint."
            )
        else:
            payload["messages"] = [{"role": "user", "content": prompt_data["text"]}]

        if "stop" in params:
            payload["stop"] = params["stop"]
        if "stop_token_ids" in params:
            payload["stop_token_ids"] = params["stop_token_ids"]
        if "custom_params" in params:
            payload["custom_params"] = params["custom_params"]

        seed = (
            sampling_params.sampling_seed
            if sampling_params.sampling_seed is not None
            else sampling_params.seed
        )
        if seed is not None:
            payload["seed"] = seed

        if logprob_start_len is not None:
            payload["logprob_start_len"] = logprob_start_len

        effective_lora = lora_path if lora_path is not None else self._lora_name
        if effective_lora:
            payload["lora_path"] = effective_lora

        if return_routed_experts:
            payload["return_routed_experts"] = True
        if return_expert_logits:
            payload["return_expert_logits"] = True

        if sampling_params.chat_continue_final_message:
            payload["continue_final_message"] = True

        self._validate_chat_completions_payload(payload)
        return payload

    def _validate_chat_completions_payload(self, payload: dict) -> None:
        """Validate the chat-completions request payload before sending."""
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("'model' must be a non-empty string")

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")

        if not messages:
            raise ValueError(
                "chat_completions payload must include at least one message"
            )
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("'messages' entries must be dictionaries")
            if not isinstance(message.get("role"), str) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError(
                    "'messages' entries must include string role and content"
                )

        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"'max_tokens' must be a positive integer, got {max_tokens}"
            )

        n = payload.get("n")
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"'n' must be a positive integer, got {n}")

        if "logprob_start_len" in payload and payload["logprob_start_len"] is not None:
            logprob_start_len = payload["logprob_start_len"]
            if not isinstance(logprob_start_len, int) or logprob_start_len < 0:
                raise ValueError("'logprob_start_len' must be a non-negative integer")

        if "stop" in payload:
            stop = payload["stop"]
            if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
                raise ValueError("'stop' must be a list of strings")

        if "stop_token_ids" in payload:
            stop_token_ids = payload["stop_token_ids"]
            if not isinstance(stop_token_ids, list) or not all(
                isinstance(tok, int) for tok in stop_token_ids
            ):
                raise ValueError("'stop_token_ids' must be a list of integers")

        if "lora_path" in payload and payload["lora_path"] is not None:
            lora_path = payload["lora_path"]
            if not isinstance(lora_path, str):
                raise ValueError(
                    f"'lora_path' must be a string, got {type(lora_path).__name__}"
                )

    @staticmethod
    def _decode_sglext_value(value: Any) -> Any:
        """Decode SGLang extension values that are serialized as JSON strings."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @classmethod
    def _extract_chat_meta_info(cls, data: dict) -> Optional[dict[str, Any]]:
        """Extract response-level metadata/extensions from chat-completions output."""
        meta_info: dict[str, Any] = {}
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            meta_info.update(metadata)

        sglext = data.get("sglext")
        if isinstance(sglext, dict):
            for key, value in sglext.items():
                meta_info[key] = cls._decode_sglext_value(value)

        return meta_info or None

    def _parse_chat_completion_choice(
        self, choice: dict, return_logprobs: bool
    ) -> types.SampledSequence:
        """Parse one OpenAI-compatible chat-completions choice."""
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            text = str(text)

        tokens: List[int] = []
        output_logprobs: Optional[List[float]] = None
        raw_output_ids = choice.get("output_ids")
        if isinstance(raw_output_ids, list) and all(
            isinstance(tok, int) for tok in raw_output_ids
        ):
            tokens = list(raw_output_ids)
        if return_logprobs:
            output_logprobs = []
            logprobs = choice.get("logprobs") or {}
            content_logprobs = logprobs.get("content") or []
            logprob_tokens: List[int] = []
            for item in content_logprobs:
                if not isinstance(item, dict):
                    continue
                token_id = item.get("token_id")
                if token_id is None:
                    continue
                logprob_tokens.append(int(token_id))
                value = item.get("logprob")
                output_logprobs.append(float(value) if value is not None else 0.0)
            if logprob_tokens:
                tokens = logprob_tokens

        raw_prompt_tokens = choice.get("input_token_ids")
        prompt_tokens = None
        if isinstance(raw_prompt_tokens, list) and all(
            isinstance(tok, int) for tok in raw_prompt_tokens
        ):
            prompt_tokens = list(raw_prompt_tokens)

        finish_reason = choice.get("finish_reason")
        stop_reason: types.StopReason = (
            "length" if finish_reason == "length" else "stop"
        )
        return types.SampledSequence(
            tokens=tokens,
            logprobs=output_logprobs,
            text=text,
            stop_reason=stop_reason,
            prompt_tokens=prompt_tokens,
        )

    async def _chat_completions_sample_request(
        self,
        prompt_data: dict,
        sampling_params: types.SamplingParams,
        return_logprobs: bool,
        num_samples: int,
        return_routed_experts: bool = False,
        return_expert_logits: bool = False,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> types.SampleResponse:
        """Make a sample request through OpenAI-compatible /v1/chat/completions."""
        payload = self._build_chat_completions_payload(
            prompt_data,
            sampling_params,
            return_logprobs,
            num_samples,
            return_routed_experts=return_routed_experts,
            return_expert_logits=return_expert_logits,
            lora_path=lora_path,
            logprob_start_len=logprob_start_len,
        )

        logger.debug(f"Chat completions request payload: {payload}")

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._request_client() as client:
                    response = await client.post("/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()

                choices = data.get("choices", [])
                if not isinstance(choices, list):
                    raise RuntimeError(
                        "Chat completions response missing list field 'choices'"
                    )

                sequences = [
                    self._parse_chat_completion_choice(choice, return_logprobs)
                    for choice in choices
                    if isinstance(choice, dict)
                ]
                return types.SampleResponse(
                    sequences=sequences,
                    meta_info=self._extract_chat_meta_info(data),
                )

            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                status_code = e.response.status_code
                # Treat 502/503/504 as transient — proxy/gateway errors from
                # dispatch when an upstream sglang backend is briefly unreachable
                # (e.g. during sglang's post-sync pause window). Original logic
                # only retried 503, so a single 502 crashed the trainer and lost
                # all per-step progress. Repro: dispatch returning HTTP 502
                # "All backends failed" mid-step during Q3.6-35B-A3B OPD Run B.
                is_transient = status_code in (502, 503, 504) or any(
                    err in error_text
                    for err in ["ReadError", "ConnectError", "TimeoutError"]
                )

                if is_transient and attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Sampling failed with transient error from {self.base_url}: HTTP {status_code}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                logger.error(
                    f"Sampling failed from {self.base_url}: HTTP {status_code} - {error_text}"
                )
                raise RuntimeError(
                    f"Sampling failed: HTTP {status_code} - {error_text}"
                ) from e

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Sampling request timed out after {self.timeout}s from {self.base_url}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                logger.error(
                    f"Sampling request timed out after {self.timeout}s from {self.base_url}: {e}"
                )
                raise RuntimeError(
                    f"Sampling failed: Request timed out after {self.timeout} seconds. "
                    f"Try increasing the timeout parameter when creating SamplingClient "
                    f"(e.g., timeout=1800.0 for 30 minutes)."
                ) from e

            except httpx.RequestError as e:
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
                    logger.error(
                        f"Inference server disconnected after {self.max_retries + 1} attempts. Error: {e}"
                    )
                    raise RuntimeError(
                        f"Sampling failed: Server disconnected after {self.max_retries + 1} attempts. "
                        f"The server may be overloaded or restarting. Original error: {e}"
                    ) from e

                logger.error(f"Sampling failed from {self.base_url}: {e}")
                raise RuntimeError(f"Sampling failed: {e}") from e

        raise RuntimeError(
            f"Sampling failed after {self.max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _normalize_generate_response(data: Any) -> List[dict]:
        """Normalize /generate response into a list of response dicts."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "meta_info" in data or "output_ids" in data or "text" in data:
                return [data]

            numeric_items: List[Tuple[int, dict]] = []
            for key, value in data.items():
                if isinstance(key, str) and key.isdigit() and isinstance(value, dict):
                    numeric_items.append((int(key), value))
            if numeric_items:
                numeric_items.sort(key=lambda x: x[0])
                return [value for _, value in numeric_items]

        return [data]

    @staticmethod
    def _extract_input_logprobs_from_item(item: dict) -> List[Optional[float]]:
        """Extract prompt/input token logprobs from a /generate item."""
        meta_info = item.get("meta_info", {})
        raw_logprobs = meta_info.get("input_token_logprobs")
        if raw_logprobs is None:
            return []

        parsed: List[Optional[float]] = []
        for entry in raw_logprobs:
            value: Optional[float]
            if isinstance(entry, (list, tuple)) and entry:
                value = entry[0]
            elif isinstance(entry, dict):
                value = entry.get("logprob", entry.get("value"))
            else:
                value = None
            parsed.append(float(value) if value is not None else None)
        return parsed

    async def _batch_sample_request(
        self,
        prompt_data: dict,
        sampling_params_dict: dict,
        return_logprobs: bool,
        num_samples: int,
        return_routed_experts: bool = False,
        return_expert_logits: bool = False,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
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
        if logprob_start_len is not None:
            payload["logprob_start_len"] = logprob_start_len

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

        # Add return_expert_logits for R3 expert logits replay.
        if return_expert_logits:
            payload["return_expert_logits"] = True

        # Validate payload before sending
        self._validate_generate_payload(payload)

        logger.debug(f"Request payload: {payload}")

        # Make async request with retry logic for transient errors
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._request_client() as client:
                    response = await client.post("/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()

                response_list = self._normalize_generate_response(data)

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
                # Treat 502/503/504 as transient — proxy/gateway errors from
                # dispatch when an upstream sglang backend is briefly unreachable
                # (e.g. during sglang's post-sync pause window). Original logic
                # only retried 503, so a single 502 crashed the trainer and lost
                # all per-step progress. Repro: dispatch returning HTTP 502
                # "All backends failed" mid-step during Q3.6-35B-A3B OPD Run B.
                is_transient = status_code in (502, 503, 504) or any(
                    err in error_text
                    for err in ["ReadError", "ConnectError", "TimeoutError"]
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

                logger.error(
                    f"Sampling failed from {self.base_url}: HTTP {e.response.status_code} - {error_text}"
                )
                raise RuntimeError(
                    f"Sampling failed: HTTP {e.response.status_code} - {error_text}"
                ) from e

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

                logger.error(
                    f"Sampling request timed out after {self.timeout}s from {self.base_url}: {e}"
                )
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
                    logger.error(
                        f"SGLang server disconnected after {self.max_retries + 1} attempts. Error: {e}"
                    )
                    raise RuntimeError(
                        f"Sampling failed: Server disconnected after {self.max_retries + 1} attempts. "
                        f"The server may be overloaded or restarting. Original error: {e}"
                    ) from e

                logger.error(f"Sampling failed from {self.base_url}: {e}")
                raise RuntimeError(f"Sampling failed: {e}") from e

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Sampling failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_sample_response(
        self, data: dict, return_logprobs: bool
    ) -> types.SampledSequence:
        """Parse a single sample response - just pass through the fields directly."""
        meta_info = data.get("meta_info", {})
        raw_logprobs = meta_info.get("output_token_logprobs")

        # output_token_logprobs is a list of [logprob, token_id, ???] tuples
        # Extract just the logprob (first element) from each tuple
        output_logprobs = None
        if raw_logprobs is not None:
            output_logprobs = [
                item[0] if item[0] is not None else 0.0 for item in raw_logprobs
            ]

        # Extract stop reason from SGLang's meta_info
        # SGLang may return finish_reason as a string ("length") or
        # a dict ({"type": "length"} / {"type": "stop", "matched": ...}).
        finish_reason = meta_info.get("finish_reason", {})
        if isinstance(finish_reason, dict):
            is_length = finish_reason.get("type") == "length"
        else:
            is_length = finish_reason == "length"
        stop_reason: types.StopReason = "length" if is_length else "stop"

        return types.SampledSequence(
            tokens=data.get("output_ids", []),
            logprobs=output_logprobs,
            text=data.get("text", ""),
            stop_reason=stop_reason,
        )

    async def sample_async(
        self,
        prompt: PromptInput,
        sampling_params: Optional[types.SamplingParams] = None,
        num_samples: int = 1,
        return_logprobs: bool = True,
        lora_path: Optional[str] = None,
        logprob_start_len: Optional[Union[int, List[int]]] = None,
    ) -> types.SampleResponse:
        """Async version of sample that directly returns the response.

        Use this when you're already in an async context and want the result directly.

        Args:
            prompt: Text prompt, ModelInput, or chat messages
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
        return await self._sample_async(
            prompt,
            sampling_params,
            num_samples,
            return_logprobs,
            lora_path,
            logprob_start_len=logprob_start_len,
        )

    def score_prompt_logprobs_batch(
        self,
        input_ids_batch: List[List[int]],
        logprob_start_lens: Optional[Union[int, List[int]]] = None,
        lora_path: Optional[str] = None,
    ) -> AsyncAPIFuture[List[List[Optional[float]]]]:
        """Compute prompt/input token logprobs for batched tokenized prompts."""
        coro = self.score_prompt_logprobs_batch_async(
            input_ids_batch=input_ids_batch,
            logprob_start_lens=logprob_start_lens,
            lora_path=lora_path,
        )
        return wrap_coroutine(coro)

    async def score_prompt_logprobs_batch_async(
        self,
        input_ids_batch: List[List[int]],
        logprob_start_lens: Optional[Union[int, List[int]]] = None,
        lora_path: Optional[str] = None,
    ) -> List[List[Optional[float]]]:
        """Async implementation for deterministic batched prompt scoring."""
        if not input_ids_batch:
            return []
        if not all(
            isinstance(seq, list) and seq and all(isinstance(tok, int) for tok in seq)
            for seq in input_ids_batch
        ):
            raise ValueError(
                "'input_ids_batch' must be a non-empty list of non-empty integer lists"
            )

        if logprob_start_lens is None:
            starts = [0] * len(input_ids_batch)
        elif isinstance(logprob_start_lens, int):
            if logprob_start_lens < 0:
                raise ValueError("'logprob_start_lens' must be >= 0")
            starts = [logprob_start_lens] * len(input_ids_batch)
        elif isinstance(logprob_start_lens, list):
            if len(logprob_start_lens) != len(input_ids_batch):
                raise ValueError(
                    f"'logprob_start_lens' length ({len(logprob_start_lens)}) must match "
                    f"batch length ({len(input_ids_batch)})"
                )
            if not all(isinstance(v, int) and v >= 0 for v in logprob_start_lens):
                raise ValueError(
                    "'logprob_start_lens' entries must be non-negative integers"
                )
            starts = logprob_start_lens
        else:
            raise ValueError("'logprob_start_lens' must be None, int, or list[int]")

        payload: dict[str, Any] = {
            "input_ids": input_ids_batch,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
            },
            "return_logprob": True,
            "logprob_start_len": starts,
        }

        if self._model:
            payload["model"] = self._model

        effective_lora = lora_path if lora_path is not None else self._lora_name
        if effective_lora:
            payload["lora_path"] = effective_lora

        self._validate_generate_payload(payload)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._request_client() as client:
                    response = await client.post("/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()

                response_list = self._normalize_generate_response(data)
                if len(response_list) != len(input_ids_batch):
                    raise RuntimeError(
                        "Prompt logprob batch response size mismatch: "
                        f"requested {len(input_ids_batch)}, got {len(response_list)}"
                    )

                prompt_logprobs_batch: List[List[Optional[float]]] = []
                for item in response_list:
                    prompt_logprobs = self._extract_input_logprobs_from_item(item)
                    if not prompt_logprobs:
                        raise RuntimeError(
                            "Inference response missing 'meta_info.input_token_logprobs'. "
                            "Prompt logprob scoring requires SGLang support for return_logprob."
                        )
                    prompt_logprobs_batch.append(prompt_logprobs)
                return prompt_logprobs_batch

            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                status_code = e.response.status_code
                # Treat 502/503/504 as transient — proxy/gateway errors from
                # dispatch when an upstream sglang backend is briefly unreachable
                # (e.g. during sglang's post-sync pause window). Original logic
                # only retried 503, so a single 502 crashed the trainer and lost
                # all per-step progress. Repro: dispatch returning HTTP 502
                # "All backends failed" mid-step during Q3.6-35B-A3B OPD Run B.
                is_transient = status_code in (502, 503, 504) or any(
                    err in error_text
                    for err in ["ReadError", "ConnectError", "TimeoutError"]
                )

                if is_transient and attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Prompt scoring transient HTTP error from {self.base_url}: HTTP {status_code}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue

                raise RuntimeError(
                    f"Prompt scoring failed: HTTP {status_code} - {error_text}"
                ) from e

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Prompt scoring timeout after {self.timeout}s from {self.base_url}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RuntimeError(
                    f"Prompt scoring timed out after {self.timeout} seconds."
                ) from e

            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Prompt scoring request error from {self.base_url}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RuntimeError(f"Prompt scoring failed: {e}") from e

        raise RuntimeError(
            f"Prompt scoring failed after {self.max_retries + 1} attempts: {last_error}"
        )

    async def sample_batch_async(
        self,
        prompts: List[PromptInput],
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
                self._sample_async(
                    prompt, sampling_params, num_samples, return_logprobs
                )
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
            completed=completed, failed=failed, cancelled=cancelled_indices
        )

    def sample_batch(
        self,
        prompts: List[PromptInput],
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

    async def pause_generation_async(self, mode: str = "in_place") -> dict:
        """Pause all in-flight generation on this SGLang server."""
        async with self._request_client() as client:
            response = await client.post(
                "/pause_generation",
                json={"mode": mode},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def continue_generation_async(self) -> dict:
        """Resume paused generation on this SGLang server."""
        async with self._request_client() as client:
            response = await client.post(
                "/continue_generation",
                json={},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    def close(self):
        """Close any pooled clients.

        In request mode there are no retained clients. In pooled mode, prefer
        `await close_async()` when already inside async code so the close is
        awaited on the current loop.
        """
        if not self._pooled_clients:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close_async())
            return
        loop.create_task(self.close_async())

    async def close_async(self):
        """Close any pooled clients owned by this SamplingClient."""
        clients = list(self._pooled_clients.values())
        self._pooled_clients.clear()
        for client in clients:
            if not client.is_closed:
                await client.aclose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()
