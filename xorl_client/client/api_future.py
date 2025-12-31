"""
APIFuture - Future wrapper for async operations.

Provides a consistent interface for both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Generator, Generic, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")


class APIFuture(Generic[T]):
    """Wrapper around concurrent.futures.Future that supports both sync and async.

    This allows users to:
    - Call .result() for synchronous blocking wait
    - Call .result_async() for async await
    - Pipeline multiple operations before waiting

    Example:
        >>> # Submit multiple operations
        >>> fwd_bwd_future = training_client.forward_backward(data, "cross_entropy")
        >>> optim_future = training_client.optim_step(adam_params)
        >>>
        >>> # Wait for both (they run in parallel)
        >>> fwd_bwd_result = fwd_bwd_future.result()
        >>> optim_result = optim_future.result()
    """

    def __init__(self, future: Future[T]):
        """Initialize APIFuture.

        Args:
            future: The underlying concurrent.futures.Future
        """
        self._future = future

    def result(self, timeout: Optional[float] = None) -> T:
        """Get result synchronously (blocking).

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The result value

        Raises:
            TimeoutError: If timeout is reached
            Exception: Any exception raised during execution
        """
        try:
            return self._future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"APIFuture.result() failed: {e}")
            raise

    async def result_async(self, timeout: Optional[float] = None) -> T:
        """Get result asynchronously (non-blocking in async context).

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The result value

        Raises:
            TimeoutError: If timeout is reached
            Exception: Any exception raised during execution
        """
        try:
            # Wrap the Future in an asyncio Future
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._future.result, timeout)
        except Exception as e:
            logger.error(f"APIFuture.result_async() failed: {e}")
            raise

    def done(self) -> bool:
        """Check if the future is done.

        Returns:
            True if the future is done (completed or failed)
        """
        return self._future.done()

    def cancel(self) -> bool:
        """Attempt to cancel the operation.

        Returns:
            True if the operation was successfully cancelled
        """
        return self._future.cancel()

    def cancelled(self) -> bool:
        """Check if the future was cancelled.

        Returns:
            True if the future was cancelled
        """
        return self._future.cancelled()

    def add_done_callback(self, fn):
        """Add a callback to be run when the future completes.

        Args:
            fn: Callback function that takes the future as argument
        """
        self._future.add_done_callback(fn)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """Get the exception raised by the operation, if any.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The exception, or None if no exception was raised
        """
        return self._future.exception(timeout=timeout)

    def __await__(self) -> Generator[None, None, T]:
        """Make APIFuture directly awaitable.

        This allows using `await future` syntax instead of `await future.result_async()`.

        Returns:
            The result value

        Example:
            >>> save_future = await training_client.save_state_async("checkpoint-001")
            >>> result = await save_future  # Uses __await__
            >>> print(f"Saved to: {result.path}")
        """
        return self.result_async().__await__()


class ImmediateAPIFuture(APIFuture[T]):
    """APIFuture that already has a result (no async operation needed).

    Used for operations that complete immediately.
    """

    def __init__(self, result: T):
        """Initialize with an immediate result.

        Args:
            result: The result value
        """
        # Create a Future that's already done
        future: Future[T] = Future()
        future.set_result(result)
        super().__init__(future)


def wrap_future(future: Future[T]) -> APIFuture[T]:
    """Wrap a concurrent.futures.Future in an APIFuture.

    Args:
        future: The future to wrap

    Returns:
        APIFuture wrapping the input future
    """
    return APIFuture(future)


def immediate_future(result: T) -> APIFuture[T]:
    """Create an APIFuture with an immediate result.

    Args:
        result: The result value

    Returns:
        APIFuture that's already complete
    """
    return ImmediateAPIFuture(result)
