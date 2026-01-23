"""Tests for ClientHolder.execute_with_retries method."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from xorl_client.client.client_holder import ClientHolder
from xorl_client.exceptions import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError,
)


class TestIsRetryableStatusCode:
    """Tests for _is_retryable_status_code static method."""

    def test_408_is_retryable(self):
        """HTTP 408 Request Timeout should be retryable."""
        assert ClientHolder._is_retryable_status_code(408) is True

    def test_429_is_retryable(self):
        """HTTP 429 Too Many Requests should be retryable."""
        assert ClientHolder._is_retryable_status_code(429) is True

    def test_500_is_retryable(self):
        """HTTP 500 Internal Server Error should be retryable."""
        assert ClientHolder._is_retryable_status_code(500) is True

    def test_502_is_retryable(self):
        """HTTP 502 Bad Gateway should be retryable."""
        assert ClientHolder._is_retryable_status_code(502) is True

    def test_503_is_retryable(self):
        """HTTP 503 Service Unavailable should be retryable."""
        assert ClientHolder._is_retryable_status_code(503) is True

    def test_504_is_retryable(self):
        """HTTP 504 Gateway Timeout should be retryable."""
        assert ClientHolder._is_retryable_status_code(504) is True

    def test_400_is_not_retryable(self):
        """HTTP 400 Bad Request should NOT be retryable."""
        assert ClientHolder._is_retryable_status_code(400) is False

    def test_401_is_not_retryable(self):
        """HTTP 401 Unauthorized should NOT be retryable."""
        assert ClientHolder._is_retryable_status_code(401) is False

    def test_403_is_not_retryable(self):
        """HTTP 403 Forbidden should NOT be retryable."""
        assert ClientHolder._is_retryable_status_code(403) is False

    def test_404_is_not_retryable(self):
        """HTTP 404 Not Found should NOT be retryable."""
        assert ClientHolder._is_retryable_status_code(404) is False

    def test_409_is_not_retryable(self):
        """HTTP 409 Conflict should NOT be retryable (resource already exists)."""
        assert ClientHolder._is_retryable_status_code(409) is False

    def test_422_is_not_retryable(self):
        """HTTP 422 Unprocessable Entity should NOT be retryable."""
        assert ClientHolder._is_retryable_status_code(422) is False


class TestIsRetryableException:
    """Tests for _is_retryable_exception static method."""

    def test_asyncio_timeout_is_retryable(self):
        """asyncio.TimeoutError should be retryable."""
        err = asyncio.TimeoutError()
        assert ClientHolder._is_retryable_exception(err) is True

    def test_api_connection_error_is_retryable(self):
        """APIConnectionError should be retryable."""
        err = APIConnectionError("http://localhost:6000/api")
        assert ClientHolder._is_retryable_exception(err) is True

    def test_api_timeout_error_is_retryable(self):
        """APITimeoutError should be retryable (inherits from APIConnectionError)."""
        err = APITimeoutError("http://localhost:6000/api", 60.0)
        assert ClientHolder._is_retryable_exception(err) is True

    def test_internal_server_error_is_retryable(self):
        """InternalServerError (5xx) should be retryable."""
        err = InternalServerError("http://localhost:6000/api", 500, "Server error")
        assert ClientHolder._is_retryable_exception(err) is True

    def test_api_status_error_429_is_retryable(self):
        """APIStatusError with 429 status should be retryable."""
        err = APIStatusError("Rate limited", "http://localhost:6000/api", 429)
        assert ClientHolder._is_retryable_exception(err) is True

    def test_api_status_error_408_is_retryable(self):
        """APIStatusError with 408 status should be retryable."""
        err = APIStatusError("Request timeout", "http://localhost:6000/api", 408)
        assert ClientHolder._is_retryable_exception(err) is True

    def test_bad_request_error_is_not_retryable(self):
        """BadRequestError (400) should NOT be retryable."""
        err = BadRequestError("http://localhost:6000/api", "Invalid request")
        assert ClientHolder._is_retryable_exception(err) is False

    def test_authentication_error_is_not_retryable(self):
        """AuthenticationError (401) should NOT be retryable."""
        err = AuthenticationError("http://localhost:6000/api")
        assert ClientHolder._is_retryable_exception(err) is False

    def test_not_found_error_is_not_retryable(self):
        """NotFoundError (404) should NOT be retryable."""
        err = NotFoundError("http://localhost:6000/api")
        assert ClientHolder._is_retryable_exception(err) is False

    def test_api_status_error_409_is_not_retryable(self):
        """APIStatusError with 409 status should NOT be retryable."""
        err = APIStatusError("Conflict", "http://localhost:6000/api", 409)
        assert ClientHolder._is_retryable_exception(err) is False

    def test_value_error_is_not_retryable(self):
        """ValueError should NOT be retryable."""
        err = ValueError("Invalid value")
        assert ClientHolder._is_retryable_exception(err) is False

    def test_runtime_error_is_not_retryable(self):
        """RuntimeError should NOT be retryable."""
        err = RuntimeError("Something went wrong")
        assert ClientHolder._is_retryable_exception(err) is False


class TestExecuteWithRetries:
    """Tests for execute_with_retries method."""

    @pytest.fixture
    def holder(self):
        """Create a ClientHolder for testing."""
        # Patch the event loop singleton to avoid starting background thread
        with patch('xorl_client.client.client_holder._event_loop_singleton') as mock_singleton:
            mock_loop = MagicMock()
            mock_singleton.get_loop.return_value = mock_loop
            holder = ClientHolder(base_url="http://localhost:6000")
            yield holder

    @pytest.mark.asyncio
    async def test_successful_execution_no_retries(self, holder):
        """Test that successful execution returns without retries."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return {"result": "success"}

        result = await holder.execute_with_retries(success_func)

        assert result == {"result": "success"}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, holder):
        """Test that connection errors trigger retry."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError("http://localhost:6000/api")
            return {"result": "success"}

        result = await holder.execute_with_retries(fail_then_succeed)

        assert result == {"result": "success"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self, holder):
        """Test that timeout errors trigger retry."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APITimeoutError("http://localhost:6000/api", 60.0)
            return {"result": "success"}

        result = await holder.execute_with_retries(fail_then_succeed)

        assert result == {"result": "success"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_500_error(self, holder):
        """Test that 500 errors trigger retry."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise InternalServerError("http://localhost:6000/api", 500, "Server error")
            return {"result": "success"}

        result = await holder.execute_with_retries(fail_then_succeed)

        assert result == {"result": "success"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_429_error(self, holder):
        """Test that 429 rate limit errors trigger retry."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APIStatusError("Rate limited", "http://localhost:6000/api", 429)
            return {"result": "success"}

        result = await holder.execute_with_retries(fail_then_succeed)

        assert result == {"result": "success"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400_error(self, holder):
        """Test that 400 errors do NOT trigger retry."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise BadRequestError("http://localhost:6000/api", "Invalid request")

        with pytest.raises(BadRequestError):
            await holder.execute_with_retries(always_fail)

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_no_retry_on_401_error(self, holder):
        """Test that 401 errors do NOT trigger retry."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("http://localhost:6000/api")

        with pytest.raises(AuthenticationError):
            await holder.execute_with_retries(always_fail)

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_no_retry_on_404_error(self, holder):
        """Test that 404 errors do NOT trigger retry."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise NotFoundError("http://localhost:6000/api")

        with pytest.raises(NotFoundError):
            await holder.execute_with_retries(always_fail)

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_no_retry_on_409_conflict(self, holder):
        """Test that 409 Conflict errors do NOT trigger retry."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise APIStatusError("Resource already exists", "http://localhost:6000/api", 409)

        with pytest.raises(APIStatusError) as exc_info:
            await holder.execute_with_retries(always_fail)

        assert exc_info.value.status_code == 409
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, holder):
        """Test that retries use exponential backoff."""
        call_times = []

        async def record_and_fail():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise APIConnectionError("http://localhost:6000/api")
            return {"result": "success"}

        result = await holder.execute_with_retries(record_and_fail)

        assert result == {"result": "success"}
        assert len(call_times) == 4

        # Check delays are approximately exponential (1s, 2s, 4s)
        # Allow some tolerance for timing
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]

        assert 0.5 <= delay1 <= 2.0, f"First delay was {delay1}s, expected ~1s"
        assert 1.0 <= delay2 <= 3.0, f"Second delay was {delay2}s, expected ~2s"
        assert 2.0 <= delay3 <= 6.0, f"Third delay was {delay3}s, expected ~4s"

    @pytest.mark.asyncio
    async def test_max_wait_time_exceeded(self, holder):
        """Test that retries stop after max wait time."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise APIConnectionError("http://localhost:6000/api")

        # Patch time to simulate time passing quickly
        start_time = time.time()

        def mock_time():
            # Simulate time passing: each call adds 100 seconds
            return start_time + (call_count * 100)

        with patch('xorl_client.client.client_holder.time.time', mock_time):
            with pytest.raises(APIConnectionError):
                await holder.execute_with_retries(always_fail)

        # Should have stopped after a few retries due to max wait time (5 min = 300s)
        assert call_count <= 5  # Should stop within ~3-4 calls at 100s each

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self, holder):
        """Test that args and kwargs are passed to the function."""
        received_args = None
        received_kwargs = None

        async def capture_args(*args, **kwargs):
            nonlocal received_args, received_kwargs
            received_args = args
            received_kwargs = kwargs
            return {"result": "success"}

        result = await holder.execute_with_retries(
            capture_args, "arg1", "arg2", key1="value1", key2="value2"
        )

        assert result == {"result": "success"}
        assert received_args == ("arg1", "arg2")
        assert received_kwargs == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_retry_on_asyncio_timeout(self, holder):
        """Test that asyncio.TimeoutError triggers retry."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return {"result": "success"}

        result = await holder.execute_with_retries(fail_then_succeed)

        assert result == {"result": "success"}
        assert call_count == 2
