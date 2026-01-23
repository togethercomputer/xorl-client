"""
Integration tests for the two-phase request pattern.

These tests demonstrate the complete flow:
1. Client submits request via async endpoint
2. Server returns UntypedAPIFuture immediately
3. Client polls retrieve_future until result is ready

Note: These are unit tests that mock the HTTP layer.
Full integration tests require a running server.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xorl_client.types import (
    UntypedAPIFuture,
    TryAgainResponse,
    RequestFailedResponse,
    ForwardBackwardOutput,
    LossFnOutput,
)
from xorl_client.client.api_future_impl import (
    _APIFuture,
    _CombinedAPIFuture,
    QueueState,
    QueueStateObserver,
)
from xorl_client.client.chunked_helpers import (
    combine_fwd_bwd_output_results,
    MAX_CHUNK_LEN,
    MAX_CHUNK_BYTES_COUNT,
)
from xorl_client.exceptions import RequestFailedError


class TestAPIFuture:
    """Tests for _APIFuture class."""

    @pytest.mark.asyncio
    async def test_poll_until_complete(self):
        """Test that _APIFuture polls until result is ready."""
        # Mock holder
        holder = MagicMock()
        holder.run_coroutine_threadsafe = lambda coro: asyncio.ensure_future(coro)

        # Track poll count
        poll_count = 0

        async def mock_post(url, data, timeout=None):
            nonlocal poll_count
            poll_count += 1

            if poll_count < 3:
                # Return TryAgainResponse for first 2 polls
                return {
                    "type": "try_again",
                    "request_id": "future_abc",
                    "queue_state": "active"
                }
            else:
                # Return actual result on 3rd poll
                return {
                    "loss_fn_output_type": "cross_entropy",
                    "loss_fn_outputs": [{"loss": 0.5}],
                    "metrics": {"loss:mean": 0.5}
                }

        holder.post = AsyncMock(side_effect=mock_post)

        untyped_future = UntypedAPIFuture(request_id="future_abc")

        future = _APIFuture(
            model_cls=ForwardBackwardOutput,
            holder=holder,
            untyped_future=untyped_future,
            request_start_time=0.0,
            request_type="ForwardBackward",
        )

        # Get result
        result = await future.result_async()

        assert isinstance(result, ForwardBackwardOutput)
        assert result.metrics["loss:mean"] == 0.5
        assert poll_count == 3

    @pytest.mark.asyncio
    async def test_poll_handles_error(self):
        """Test that _APIFuture raises on RequestFailedResponse."""
        holder = MagicMock()
        holder.run_coroutine_threadsafe = lambda coro: asyncio.ensure_future(coro)

        async def mock_post(url, data, timeout=None):
            return {
                "error": "Invalid input",
                "category": "user"
            }

        holder.post = AsyncMock(side_effect=mock_post)

        untyped_future = UntypedAPIFuture(request_id="future_fail")

        future = _APIFuture(
            model_cls=ForwardBackwardOutput,
            holder=holder,
            untyped_future=untyped_future,
            request_start_time=0.0,
            request_type="ForwardBackward",
        )

        with pytest.raises(RequestFailedError) as exc_info:
            await future.result_async()

        assert "Invalid input" in str(exc_info.value)


class TestCombinedAPIFuture:
    """Tests for _CombinedAPIFuture class."""

    @pytest.mark.asyncio
    async def test_combines_results(self):
        """Test that _CombinedAPIFuture combines multiple results."""
        holder = MagicMock()
        holder.run_coroutine_threadsafe = lambda coro: asyncio.ensure_future(coro)

        # Create mock futures that return immediately
        results = [
            ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[LossFnOutput(loss=0.3)],
                metrics={"loss:mean": 0.3}
            ),
            ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[LossFnOutput(loss=0.7)],
                metrics={"loss:mean": 0.7}
            ),
        ]

        # Create mock _APIFuture objects
        mock_futures = []
        for result in results:
            mock_future = MagicMock()
            mock_future.result_async = AsyncMock(return_value=result)
            mock_futures.append(mock_future)

        combined = _CombinedAPIFuture(
            futures=mock_futures,
            transform=combine_fwd_bwd_output_results,
            holder=holder,
        )

        result = await combined.result_async()

        assert isinstance(result, ForwardBackwardOutput)
        # Two outputs combined
        assert len(result.loss_fn_outputs) == 2


class TestQueueStateObserver:
    """Tests for QueueStateObserver."""

    def test_observer_is_abstract(self):
        """Test that QueueStateObserver is abstract."""
        with pytest.raises(TypeError):
            QueueStateObserver()

    def test_concrete_observer(self):
        """Test concrete observer implementation."""
        class MyObserver(QueueStateObserver):
            def __init__(self):
                self.states = []

            def on_queue_state_change(self, queue_state, queue_state_reason):
                self.states.append((queue_state, queue_state_reason))

        observer = MyObserver()
        observer.on_queue_state_change(QueueState.PAUSED_CAPACITY, "GPU full")

        assert len(observer.states) == 1
        assert observer.states[0] == (QueueState.PAUSED_CAPACITY, "GPU full")


class TestChunkedHelpers:
    """Tests for chunking helper functions."""

    def test_combine_empty_results(self):
        """Test combining empty results."""
        result = combine_fwd_bwd_output_results([])

        assert result.loss_fn_output_type == ""
        assert result.metrics == {}
        assert result.loss_fn_outputs == []

    def test_combine_single_result(self):
        """Test combining single result."""
        single = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={"loss:mean": 0.5}
        )

        result = combine_fwd_bwd_output_results([single])

        assert result.loss_fn_output_type == "cross_entropy"
        assert len(result.loss_fn_outputs) == 1
        assert result.metrics["loss:mean"] == 0.5

    def test_combine_multiple_results(self):
        """Test combining multiple results with metrics reduction."""
        results = [
            ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[LossFnOutput(loss=0.3), LossFnOutput(loss=0.4)],  # 2 outputs, weight=2
                metrics={"loss:mean": 0.35, "loss:sum": 0.7}
            ),
            ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[LossFnOutput(loss=0.8)],  # 1 output, weight=1
                metrics={"loss:mean": 0.8, "loss:sum": 0.8}
            ),
        ]

        combined = combine_fwd_bwd_output_results(results)

        # 3 total outputs
        assert len(combined.loss_fn_outputs) == 3

        # Mean is weighted: (0.35 * 2 + 0.8 * 1) / 3 = 0.5
        assert combined.metrics["loss:mean"] == pytest.approx(0.5, rel=0.01)

        # Sum is just summed: 0.7 + 0.8 = 1.5
        assert combined.metrics["loss:sum"] == pytest.approx(1.5, rel=0.01)

    def test_chunking_constants(self):
        """Test chunking constants are defined."""
        assert MAX_CHUNK_LEN == 1024
        assert MAX_CHUNK_BYTES_COUNT == 5_000_000


class TestTwoPhaseFlow:
    """End-to-end tests for two-phase flow simulation."""

    @pytest.mark.asyncio
    async def test_complete_flow_simulation(self):
        """Simulate complete two-phase flow."""
        # Simulated server state
        server_requests = {}
        request_counter = [0]

        async def submit_request(request_data):
            """Simulate Phase 1: Submit request."""
            request_id = f"future_{request_counter[0]:06d}"
            request_counter[0] += 1

            # Queue request for "processing"
            server_requests[request_id] = {
                "status": "processing",
                "data": request_data,
                "result": None,
            }

            # Simulate async processing
            async def process():
                await asyncio.sleep(0.1)
                server_requests[request_id]["status"] = "completed"
                server_requests[request_id]["result"] = {
                    "loss_fn_output_type": "cross_entropy",
                    "loss_fn_outputs": [{"loss": 0.5}],
                    "metrics": {"loss:mean": 0.5}
                }

            asyncio.create_task(process())

            return {"request_id": request_id, "model_id": "test-model"}

        async def retrieve_result(request_id):
            """Simulate Phase 2: Retrieve result."""
            if request_id not in server_requests:
                return {"error": "Not found", "category": "user"}

            req = server_requests[request_id]
            if req["status"] == "processing":
                return {
                    "type": "try_again",
                    "request_id": request_id,
                    "queue_state": "active"
                }
            elif req["status"] == "completed":
                return req["result"]
            else:
                return {"error": "Unknown status", "category": "server"}

        # Phase 1: Submit
        submit_response = await submit_request({"test": "data"})
        assert "request_id" in submit_response
        request_id = submit_response["request_id"]

        # Phase 2: Poll until complete
        max_polls = 10
        for i in range(max_polls):
            result = await retrieve_result(request_id)

            if result.get("type") == "try_again":
                await asyncio.sleep(0.05)
                continue

            if "error" in result:
                pytest.fail(f"Request failed: {result['error']}")

            # Success!
            assert result["loss_fn_output_type"] == "cross_entropy"
            assert result["metrics"]["loss:mean"] == 0.5
            break
        else:
            pytest.fail("Request never completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
