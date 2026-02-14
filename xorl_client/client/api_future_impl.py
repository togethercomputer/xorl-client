"""
Internal APIFuture implementation with polling logic.

This module implements the two-phase request pattern:
1. Initial request returns UntypedAPIFuture with request_id
2. _APIFuture polls /api/v1/retrieve_future until result is ready

Based on Tinker's api_future_impl.py implementation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Type, TypeVar, cast

from xorl_client import types
from xorl_client.exceptions import APIConnectionError, RequestFailedError, RetryableException

if TYPE_CHECKING:
    from xorl_client.client.client_holder import ClientHolder

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Sentinel object to indicate that the result hasn't been computed yet
_UNCOMPUTED = object()


class QueueState(Enum):
    """State of the request in the server's processing queue."""

    ACTIVE = "active"
    PAUSED_RATE_LIMIT = "paused_rate_limit"
    PAUSED_CAPACITY = "paused_capacity"
    UNKNOWN = "unknown"


class QueueStateObserver(ABC):
    """Observer interface for queue state changes.

    Implement this to receive notifications when the server's queue state changes,
    e.g., when requests are paused due to capacity limits.
    """

    @abstractmethod
    def on_queue_state_change(self, queue_state: QueueState, queue_state_reason: str | None) -> None:
        """Called when queue state changes.

        Args:
            queue_state: New queue state
            queue_state_reason: Optional reason for the state change
        """
        raise NotImplementedError


class _APIFuture(Generic[T]):
    """Internal APIFuture implementation with automatic polling.

    This class wraps an UntypedAPIFuture and automatically polls the
    /api/v1/retrieve_future endpoint until the result is ready.

    The polling logic handles:
    - TryAgainResponse: Continue polling
    - RequestFailedResponse: Raise RequestFailedError
    - Success response: Parse and return result
    - Connection errors: Retry with exponential backoff
    - HTTP 408/5xx: Retry
    - HTTP 410: Raise RetryableException (promise expired)

    Args:
        model_cls: The expected result type class
        holder: ClientHolder for making HTTP requests
        untyped_future: The UntypedAPIFuture from Phase 1
        request_start_time: Timestamp when the request was initiated
        request_type: Type of request (for telemetry/logging)
        queue_state_observer: Optional observer for queue state changes

    Example:
        >>> untyped = await client.post("/api/v1/forward_backward", data)
        >>> future = _APIFuture(
        ...     ForwardBackwardOutput,
        ...     holder,
        ...     UntypedAPIFuture.from_dict(untyped),
        ...     time.time(),
        ...     "ForwardBackward",
        ... )
        >>> result = future.result()  # Blocks until complete
    """

    def __init__(
        self,
        model_cls: Type[T],
        holder: "ClientHolder",
        untyped_future: types.UntypedAPIFuture,
        request_start_time: float,
        request_type: str,
        queue_state_observer: QueueStateObserver | None = None,
    ):
        self.model_cls = model_cls
        self.holder = holder
        self.untyped_future = untyped_future
        self.request_type = request_type
        self._cached_result: Any = _UNCOMPUTED

        # Timing for telemetry
        self.request_start_time = request_start_time
        self.request_future_start_time = time.time()
        self.request_queue_roundtrip_time = self.request_future_start_time - request_start_time

        # Queue state observer
        self._queue_state_observer = queue_state_observer

        # Start polling in background
        self._future = self.holder.run_coroutine_threadsafe(self._poll_for_result())

    @property
    def request_id(self) -> str:
        """Get the request ID from the untyped future."""
        return self.untyped_future.request_id

    async def _poll_for_result(self, timeout: float | None = None) -> T:
        """Poll /api/v1/retrieve_future until result is ready.

        This implements the main polling loop with retry logic.

        Args:
            timeout: Maximum time to wait for result (None = indefinite)

        Returns:
            The parsed result of type T

        Raises:
            TimeoutError: If timeout exceeded
            RequestFailedError: If request failed on server
            RetryableException: If promise expired (HTTP 410)
        """
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        start_time = time.time()
        iteration = -1
        connection_error_retries = 0

        while True:
            iteration += 1

            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                logger.error(
                    f"Timeout waiting for {self.request_id} after {timeout}s "
                    f"(iterations={iteration})"
                )
                raise TimeoutError(
                    f"Timeout of {timeout} seconds reached while waiting for "
                    f"result of request_id={self.request_id}"
                )

            # Build request
            retrieve_request = types.FutureRetrieveRequest(request_id=self.request_id)

            try:
                # Poll the retrieve_future endpoint
                # Use 60s client timeout > 45s server long-poll timeout to avoid race condition
                response = await self.holder.post(
                    "/api/v1/retrieve_future",
                    retrieve_request.to_dict(),
                    timeout=600,  # Per-request timeout
                )
                connection_error_retries = 0  # Reset on successful connection

            except Exception as e:
                # Handle connection errors with exponential backoff.
                # post() wraps all transport-level errors (ConnectError, ReadError,
                # RemoteProtocolError, TimeoutError, etc.) as APIConnectionError,
                # so an isinstance check catches them all.
                is_connection_error = isinstance(e, APIConnectionError)

                if is_connection_error:
                    wait_time = min(2**connection_error_retries, 30)
                    logger.warning(
                        f"Connection error polling {self.request_id}, "
                        f"retry {connection_error_retries} after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    connection_error_retries += 1
                    continue

                # Check for HTTP status errors
                if hasattr(e, "status_code"):
                    status_code = getattr(e, "status_code")

                    # HTTP 408: Request still processing
                    if status_code == 408:
                        # Try to extract queue state from response
                        self._handle_try_again_status(e)
                        continue

                    # HTTP 410: Promise expired
                    if status_code == 410:
                        raise RetryableException(
                            f"Promise expired/broken for request {self.request_id}"
                        ) from e

                    # HTTP 5xx: Server error, retry
                    if 500 <= status_code < 600:
                        logger.warning(
                            f"Server error {status_code} polling {self.request_id}, retrying"
                        )
                        continue

                # Re-raise other errors
                raise

            # Parse response
            result = self._parse_response(response)
            if result is not None:
                return result

            # If _parse_response returns None, it means TryAgainResponse - continue polling
            # Wait 1 second between polls to avoid overwhelming the server
            await asyncio.sleep(1.0)

    def _handle_try_again_status(self, error: Exception) -> None:
        """Handle HTTP 408 response and extract queue state."""
        if self._queue_state_observer is None:
            return

        try:
            # Try to get response body
            response_data = None
            if hasattr(error, "response"):
                response = getattr(error, "response")
                if hasattr(response, "json"):
                    response_data = response.json()

            if response_data:
                queue_state_str = response_data.get("queue_state")
                queue_state_reason = response_data.get("queue_state_reason")

                if queue_state_str:
                    queue_state = {
                        "active": QueueState.ACTIVE,
                        "paused_rate_limit": QueueState.PAUSED_RATE_LIMIT,
                        "paused_capacity": QueueState.PAUSED_CAPACITY,
                    }.get(queue_state_str, QueueState.UNKNOWN)

                    self._queue_state_observer.on_queue_state_change(
                        queue_state, queue_state_reason
                    )
        except Exception:
            # Don't fail on observer errors
            pass

    def _parse_response(self, response: dict) -> T | None:
        """Parse the response from retrieve_future.

        Args:
            response: Response dictionary from the server

        Returns:
            Parsed result of type T, or None if TryAgainResponse

        Raises:
            RequestFailedError: If response is RequestFailedResponse
        """
        # Check for TryAgainResponse
        if response.get("type") == "try_again":
            # Extract queue state for logging
            queue_state = response.get("queue_state", "unknown")
            if queue_state != "active":
                logger.debug(f"Request {self.request_id} pending: queue_state={queue_state}")

            # Notify observer
            if self._queue_state_observer:
                queue_state_enum = {
                    "active": QueueState.ACTIVE,
                    "paused_rate_limit": QueueState.PAUSED_RATE_LIMIT,
                    "paused_capacity": QueueState.PAUSED_CAPACITY,
                }.get(queue_state, QueueState.UNKNOWN)

                self._queue_state_observer.on_queue_state_change(queue_state_enum, None)

            return None  # Signal to continue polling

        # Check for RequestFailedResponse
        if "error" in response:
            error_message = response["error"]
            error_category = response.get("category", "unknown")

            logger.error(
                f"Request {self.request_id} failed: {error_message} "
                f"(category={error_category})"
            )

            raise RequestFailedError(
                message=f"Request failed: {error_message}",
                request_id=self.request_id,
                category=error_category,
            )

        # Success - parse the result
        try:
            # Check if model_cls has from_dict method
            if hasattr(self.model_cls, "from_dict"):
                self._cached_result = self.model_cls.from_dict(response)
            else:
                # For types without from_dict, return raw response
                self._cached_result = response

            logger.debug(f"Request {self.request_id} completed successfully")
            return cast(T, self._cached_result)

        except Exception as e:
            logger.error(
                f"Error parsing result for {self.request_id}: {e}\n"
                f"Response: {response}"
            )
            raise ValueError(
                f"Error parsing result for request_id={self.request_id}: {e}"
            ) from e

    def result(self, timeout: float | None = None) -> T:
        """Get the result synchronously (blocking).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The result value of type T

        Raises:
            TimeoutError: If timeout exceeded
            RequestFailedError: If request failed
        """
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        """Get the result asynchronously.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The result value of type T

        Raises:
            TimeoutError: If timeout exceeded
            RequestFailedError: If request failed
        """
        if timeout is not None:
            return await asyncio.wait_for(asyncio.wrap_future(self._future), timeout)
        return await asyncio.wrap_future(self._future)

    def done(self) -> bool:
        """Check if the result is ready."""
        return self._future.done()

    def __await__(self):
        """Make _APIFuture directly awaitable."""
        return self.result_async().__await__()


class _CombinedAPIFuture(Generic[T]):
    """Combines multiple APIFutures into a single result.

    Used when a request is chunked into multiple sub-requests. Each chunk
    gets its own _APIFuture, and _CombinedAPIFuture waits for all of them
    and combines the results using a transform function.

    Args:
        futures: List of _APIFuture instances to combine
        transform: Function to combine individual results into final result
        holder: ClientHolder (for running async code)

    Example:
        >>> # Create futures for each chunk
        >>> futures = [_APIFuture(...) for chunk in chunks]
        >>>
        >>> # Combine results
        >>> combined = _CombinedAPIFuture(
        ...     futures,
        ...     combine_fwd_bwd_output_results,
        ...     holder,
        ... )
        >>> result = combined.result()  # Single combined result
    """

    def __init__(
        self,
        futures: List[_APIFuture[T]],
        transform: Callable[[List[T]], T],
        holder: "ClientHolder",
    ):
        self.futures = futures
        self.transform = transform
        self.holder = holder

    def result(self, timeout: float | None = None) -> T:
        """Get the combined result synchronously.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Combined result
        """
        return self.holder.run_coroutine_threadsafe(
            self.result_async(timeout)
        ).result()

    async def result_async(self, timeout: float | None = None) -> T:
        """Get the combined result asynchronously.

        Waits for all futures to complete in parallel, then combines
        the results using the transform function.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Combined result
        """
        # Wait for all futures in parallel
        results = await asyncio.gather(
            *[future.result_async(timeout) for future in self.futures]
        )
        # Combine results
        return self.transform(results)

    def done(self) -> bool:
        """Check if all futures are complete."""
        return all(future.done() for future in self.futures)

    def __await__(self):
        """Make _CombinedAPIFuture directly awaitable."""
        return self.result_async().__await__()
