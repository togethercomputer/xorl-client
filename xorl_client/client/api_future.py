"""
APIFuture - Future wrapper for async operations.

Provides a consistent interface for both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import Future
from typing import Coroutine, Generator, Generic, TypeVar, Optional, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _BackgroundLoopSingleton:
    """Singleton that manages a background event loop for async operations.

    This allows async coroutines to be executed from any context (sync or async)
    by using a dedicated event loop running in a background thread.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()

    def _ensure_started(self):
        """Ensure the background loop is started."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._started = True

    def _run_loop(self):
        """Run the event loop forever in the background thread."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the background event loop, starting it if necessary."""
        self._ensure_started()
        assert self._loop is not None
        return self._loop

    def run_coroutine_threadsafe(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule a coroutine to run on the background loop.

        Args:
            coro: The coroutine to run

        Returns:
            A concurrent.futures.Future that will contain the result
        """
        loop = self.get_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def stop(self):
        """Stop the background loop (for cleanup)."""
        if self._loop is not None and self._started:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5.0)


# Global singleton for the background event loop
_background_loop = _BackgroundLoopSingleton()

# Register cleanup on exit
atexit.register(_background_loop.stop)


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


class AsyncAPIFuture(Generic[T]):
    """APIFuture backed by an asyncio coroutine scheduled on a background loop.

    This provides the same interface as APIFuture but uses asyncio internally
    for better efficiency with I/O-bound operations like HTTP requests.

    The coroutine is scheduled on a background event loop (managed by
    _BackgroundLoopSingleton), which allows .result() to work from any context
    including Jupyter notebooks with running event loops.

    Supports:
    - .result() for synchronous blocking wait (works from any context)
    - .result_async() or await for async usage
    - Works in Jupyter notebooks and other async environments

    Example:
        >>> # Async usage
        >>> future = sampling_client.sample(prompt, params)
        >>> response = await future

        >>> # Sync usage (works even in Jupyter)
        >>> future = sampling_client.sample(prompt, params)
        >>> response = future.result()
    """

    def __init__(self, coro: Coroutine[Any, Any, T]):
        """Initialize AsyncAPIFuture.

        Args:
            coro: The coroutine to execute
        """
        # Schedule the coroutine on the background loop immediately
        self._future: Future[T] = _background_loop.run_coroutine_threadsafe(coro)

    def result(self, timeout: Optional[float] = None) -> T:
        """Get result synchronously (blocking).

        This blocks until the result is available. Works from any context
        including Jupyter notebooks with running event loops.

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
            logger.error(f"AsyncAPIFuture.result() failed: {e}")
            raise

    async def result_async(self, timeout: Optional[float] = None) -> T:
        """Get result asynchronously.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The result value

        Raises:
            TimeoutError: If timeout is reached
            Exception: Any exception raised during execution
        """
        try:
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    return await asyncio.wrap_future(self._future)
            else:
                return await asyncio.wrap_future(self._future)
        except asyncio.TimeoutError as e:
            logger.error(f"AsyncAPIFuture.result_async() timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s") from e
        except Exception as e:
            # Check if this is a transient error (ReadError, ConnectError, TimeoutError)
            error_str = str(e)
            is_transient = any(err in error_str for err in ["ReadError", "ConnectError", "TimeoutError"])

            if is_transient:
                logger.warning(f"AsyncAPIFuture.result_async() failed with transient error")
            else:
                logger.error(f"AsyncAPIFuture.result_async() failed: {e}")

            raise

    def done(self) -> bool:
        """Check if the future is done."""
        return self._future.done()

    def __await__(self) -> Generator[None, None, T]:
        """Make AsyncAPIFuture directly awaitable."""
        return self.result_async().__await__()


def wrap_future(future: Future[T]) -> APIFuture[T]:
    """Wrap a concurrent.futures.Future in an APIFuture.

    Args:
        future: The future to wrap

    Returns:
        APIFuture wrapping the input future
    """
    return APIFuture(future)


def wrap_coroutine(coro) -> AsyncAPIFuture:
    """Wrap an asyncio coroutine in an AsyncAPIFuture.

    Args:
        coro: The coroutine to wrap

    Returns:
        AsyncAPIFuture wrapping the coroutine
    """
    return AsyncAPIFuture(coro)


def immediate_future(result: T) -> APIFuture[T]:
    """Create an APIFuture with an immediate result.

    Args:
        result: The result value

    Returns:
        APIFuture that's already complete
    """
    return ImmediateAPIFuture(result)
