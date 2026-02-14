"""
ClientHolder - Manages API connections and session state.

This is the internal class that handles HTTP connections, session management,
model IDs, and async execution. ServiceClient and clients use this internally.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Coroutine

import httpx

from xorl_client.exceptions import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError,
)
from xorl_client.resources import (
    TrainingResource,
    ModelsResource,
    WeightsResource,
    FuturesResource,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class _ClientHolderEventLoopSingleton:
    """Global singleton event loop for all ClientHolder instances (like Tinker)."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started: bool = False
        self._lifecycle_lock: threading.Lock = threading.Lock()

    def _ensure_started(self):
        if self._started:
            return

        with self._lifecycle_lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
            self._thread.start()
            self._started = True

    def _background_thread_func(self):
        assert self._loop is not None, "Loop must not be None"
        logger.info("Global event loop thread started")
        self._loop.run_forever()
        logger.info("Global event loop thread stopped")

    def get_loop(self) -> asyncio.AbstractEventLoop:
        self._ensure_started()
        assert self._loop is not None, "Loop must not be None"
        return self._loop


_event_loop_singleton = _ClientHolderEventLoopSingleton()


class ClientHolder:
    """Manages API connections and session state for XoRL clients.

    This class:
    - Manages HTTP session and base URL
    - Tracks session ID and model IDs
    - Provides async execution via thread pool

    Args:
        base_url: Base URL for the training API (e.g., "http://localhost:5555")
        api_key: API key for authentication (optional)
        timeout: Default timeout for HTTP requests in seconds
        **kwargs: Additional arguments (for future compatibility)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:6000",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 1800.0,
        **kwargs: Any,
    ):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.timeout = timeout

        # Session management
        self._session_id = str(uuid.uuid4())
        self._model_counter = 0
        self._model_counter_lock = threading.Lock()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="xorl_client-client")

        # Use global singleton event loop (like Tinker)
        # This allows _take_turn to work correctly across ALL client instances
        self._loop: asyncio.AbstractEventLoop = _event_loop_singleton.get_loop()

        # HTTP client (async like Tinker)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_client_headers = headers

        # Lazy-initialized resources (like Tinker's client.models, client.training, etc.)
        self._training: Optional[TrainingResource] = None
        self._models: Optional[ModelsResource] = None
        self._weights: Optional[WeightsResource] = None
        self._futures: Optional[FuturesResource] = None

        logger.info(f"ClientHolder initialized: session_id={self._session_id}, base_url={self.base_url}")

    @property
    def training(self) -> TrainingResource:
        """Low-level training resource for forward, forward_backward, optim_step."""
        if self._training is None:
            self._training = TrainingResource(self)
        return self._training

    @property
    def models(self) -> ModelsResource:
        """Low-level models resource for create, unload, get_info."""
        if self._models is None:
            self._models = ModelsResource(self)
        return self._models

    @property
    def weights(self) -> WeightsResource:
        """Low-level weights resource for save, load, checkpoint management."""
        if self._weights is None:
            self._weights = WeightsResource(self)
        return self._weights

    @property
    def futures(self) -> FuturesResource:
        """Low-level futures resource for retrieve_future."""
        if self._futures is None:
            self._futures = FuturesResource(self)
        return self._futures

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    def get_model_id(self) -> str:
        """Generate a new model ID.

        Returns:
            New model ID (e.g., "model-123")
        """
        with self._model_counter_lock:
            model_seq = self._model_counter
            self._model_counter += 1
        model_id = f"model-{model_seq}"
        logger.info(f"Generated model_id: {model_id}")
        return model_id

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (must be called from event loop thread)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers=self._http_client_headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._http_client

    async def post(self, endpoint: str, data: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send async POST request (like Tinker).

        Args:
            endpoint: API endpoint (e.g., "/api/v1/forward_backward")
            data: JSON data to send
            timeout: Optional timeout override

        Returns:
            Response JSON

        Raises:
            APIConnectionError: If unable to connect to the server
            APITimeoutError: If the request times out
            BadRequestError: If the server returns HTTP 400
            AuthenticationError: If the server returns HTTP 401
            NotFoundError: If the server returns HTTP 404
            InternalServerError: If the server returns HTTP 500+
            APIStatusError: For other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        timeout_val = timeout if timeout is not None else self.timeout
        client = self._get_http_client()

        # Add model to request payload for API routing (if set)
        if self._model:
            data = {**data, "model": self._model}

        try:
            response = await client.post(url, json=data, timeout=timeout_val)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectTimeout as e:
            logger.error(f"POST {url} connection timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.ReadTimeout as e:
            logger.error(f"POST {url} read timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.ConnectError as e:
            logger.error(f"POST {url} connection failed: {e}")
            raise APIConnectionError(url, e) from e
        except httpx.HTTPStatusError as e:
            # Extract error detail if available
            error_detail = None
            response_body = None
            status_code = e.response.status_code
            try:
                response_body = e.response.text
                error_json = e.response.json()
                if "detail" in error_json:
                    error_detail = error_json["detail"]
            except Exception:
                pass

            # Use warning level for "already loaded" errors (not fatal)
            error_msg = error_detail or response_body or ""
            if "already loaded" in str(error_msg).lower():
                logger.warning(f"POST {url} returned HTTP {status_code}: {error_msg}")
            else:
                logger.error(f"POST {url} failed with HTTP {status_code}: {error_msg}")

            # Raise appropriate exception based on status code
            if status_code == 400:
                raise BadRequestError(url, error_detail or "Bad request", response_body) from e
            elif status_code == 401:
                raise AuthenticationError(url, error_detail) from e
            elif status_code == 404:
                raise NotFoundError(url, error_detail) from e
            elif status_code >= 500:
                raise InternalServerError(url, status_code, error_detail) from e
            else:
                raise APIStatusError(
                    error_detail or f"HTTP {status_code} error",
                    url,
                    status_code,
                    response_body,
                ) from e
        except httpx.TimeoutException as e:
            logger.error(f"POST {url} timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.RequestError as e:
            # Catch-all for other request exceptions
            logger.error(f"POST {url} failed: {e}")
            raise APIConnectionError(url, e) from e

    def post_sync(self, endpoint: str, data: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Synchronous wrapper for post (schedules async post on event loop).

        Used for legacy code that needs sync interface.
        """
        future = self.run_coroutine_threadsafe(self.post(endpoint, data, timeout))
        return future.result()

    def post_async(
        self, endpoint: str, data: Dict[str, Any], timeout: Optional[float] = None
    ) -> Future[Dict[str, Any]]:
        """Legacy async wrapper (schedules async post on event loop).

        Returns concurrent.futures.Future for backward compatibility.
        """
        return self.run_coroutine_threadsafe(self.post(endpoint, data, timeout))

    async def get(self, endpoint: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send async GET request (like Tinker).

        Args:
            endpoint: API endpoint
            timeout: Optional timeout override

        Returns:
            Response JSON

        Raises:
            APIConnectionError: If unable to connect to the server
            APITimeoutError: If the request times out
            APIStatusError: For HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        timeout_val = timeout if timeout is not None else self.timeout
        client = self._get_http_client()

        try:
            response = await client.get(url, timeout=timeout_val)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectTimeout as e:
            logger.error(f"GET {url} connection timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.ReadTimeout as e:
            logger.error(f"GET {url} read timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.ConnectError as e:
            logger.error(f"GET {url} connection failed: {e}")
            raise APIConnectionError(url, e) from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = None
            try:
                error_json = e.response.json()
                if "detail" in error_json:
                    error_detail = error_json["detail"]
            except Exception:
                pass
            logger.error(f"GET {url} failed with HTTP {status_code}")
            raise APIStatusError(
                error_detail or f"HTTP {status_code} error",
                url,
                status_code,
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"GET {url} timed out")
            raise APITimeoutError(url, timeout_val, e) from e
        except httpx.RequestError as e:
            logger.error(f"GET {url} failed: {e}")
            raise APIConnectionError(url, e) from e

    def get_sync(self, endpoint: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Synchronous wrapper for get (schedules async get on event loop).

        Used for legacy code that needs sync interface.
        """
        future = self.run_coroutine_threadsafe(self.get(endpoint, timeout))
        return future.result()

    def get_async(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Future[Dict[str, Any]]:
        """Legacy async wrapper (schedules async get on event loop).

        Returns concurrent.futures.Future for backward compatibility.
        """
        # Build URL with query params
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            full_endpoint = f"{endpoint}?{query_string}"
        else:
            full_endpoint = endpoint

        return self.run_coroutine_threadsafe(self.get(full_endpoint, timeout))

    @staticmethod
    def _is_retryable_status_code(status_code: int) -> bool:
        """Check if an HTTP status code indicates a retryable error.

        Retryable status codes:
        - 408: Request Timeout
        - 429: Too Many Requests (rate limited)
        - 5xx: Server errors

        Non-retryable:
        - 409: Conflict (resource already exists)
        - 4xx: Client errors (except 408, 429)

        Args:
            status_code: HTTP status code

        Returns:
            True if the error is retryable
        """
        return status_code in (408, 429) or (500 <= status_code < 600)

    @staticmethod
    def _is_retryable_exception(exception: Exception) -> bool:
        """Check if an exception indicates a retryable error.

        Retryable exceptions:
        - asyncio.TimeoutError
        - APIConnectionError (includes APITimeoutError)
        - httpx.TimeoutException
        - APIStatusError with retryable status code (408, 429, 5xx)

        Args:
            exception: The exception to check

        Returns:
            True if the error is retryable
        """
        RETRYABLE_EXCEPTIONS = (
            asyncio.TimeoutError,
            APIConnectionError,  # Includes APITimeoutError
            httpx.TimeoutException,
        )
        if isinstance(exception, RETRYABLE_EXCEPTIONS):
            return True
        if isinstance(exception, APIStatusError):
            return ClientHolder._is_retryable_status_code(exception.status_code)
        return False

    async def execute_with_retries(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute an async function with automatic retries on transient errors.

        This method implements exponential backoff retry logic for transient errors
        like connection issues, timeouts, and server errors. It matches Tinker's
        retry behavior for consistency.

        Retry behavior:
        - Max total wait time: 5 minutes
        - Exponential backoff: 2^attempt seconds (capped at 30 seconds)
        - Retries on: timeouts, connection errors, HTTP 408/429/5xx
        - Does NOT retry: HTTP 400, 401, 404, 409 (client errors)

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            The result of func(*args, **kwargs)

        Raises:
            Exception: Re-raises the last exception if all retries fail or
                      if the exception is not retryable

        Example:
            >>> async def make_request():
            ...     return await self.post("/api/v1/forward", data)
            >>> result = await holder.execute_with_retries(make_request)
        """
        MAX_WAIT_TIME = 60 * 5  # 5 minutes
        start_time = time.time()
        attempt_count = 0

        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                is_retryable = self._is_retryable_exception(e)
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Log the error
                func_name = getattr(func, "__qualname__", getattr(func, "__name__", type(func).__name__))
                status_code = getattr(e, "status_code", None)

                if is_retryable and elapsed_time < MAX_WAIT_TIME:
                    # Apply exponential backoff
                    time_to_wait = min(2 ** attempt_count, 30)
                    attempt_count += 1
                    # Don't wait too long if we're almost at the max wait time
                    time_to_wait = min(time_to_wait, start_time + MAX_WAIT_TIME - current_time)

                    logger.warning(
                        f"Retryable error in {func_name} (attempt {attempt_count}, "
                        f"status_code={status_code}, elapsed={elapsed_time:.1f}s): {e}. "
                        f"Retrying in {time_to_wait:.1f}s..."
                    )
                    await asyncio.sleep(time_to_wait)
                    continue

                # Not retryable or max wait time exceeded
                if is_retryable:
                    logger.error(
                        f"Max retry time exceeded for {func_name} after {elapsed_time:.1f}s "
                        f"and {attempt_count} attempts: {e}"
                    )
                else:
                    logger.error(
                        f"Non-retryable error in {func_name} (status_code={status_code}): {e}"
                    )

                raise e

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the global singleton event loop.

        Returns:
            The shared asyncio event loop running in background thread
        """
        return self._loop

    def run_coroutine_threadsafe(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule a coroutine to run in the shared event loop.

        This is thread-safe and can be called from any thread.
        Returns a concurrent.futures.Future that will contain the result.

        Args:
            coro: Coroutine to run in the event loop

        Returns:
            Future that will be completed when the coroutine finishes
        """
        return asyncio.run_coroutine_threadsafe(coro, self.get_loop())

    def shutdown(self):
        """Shutdown the client holder and cleanup resources."""
        logger.info("Shutting down ClientHolder")

        # Close HTTP client
        if self._http_client is not None:
            # Schedule close on event loop
            future = asyncio.run_coroutine_threadsafe(self._http_client.aclose(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                pass

        # Note: We don't stop the global event loop (it's shared across all holders)
        self._executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
