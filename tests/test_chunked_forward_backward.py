"""Tests for chunked forward/backward functionality.

Tests the transparent chunking of large batches in TrainingClient.forward_backward()
and TrainingClient.forward(), as well as the estimate_datum_bytes fix.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import Future

from xorl_client import types
from xorl_client.client.training_client import TrainingClient
from xorl_client.client.client_holder import ClientHolder
from xorl_client.client.chunked_helpers import (
    MAX_CHUNK_LEN,
    MAX_CHUNK_BYTES_COUNT,
    estimate_datum_bytes,
    combine_fwd_bwd_output_results,
    _infer_reduction_type,
)
from xorl_client.types import ForwardBackwardOutput, LossFnOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_datum(num_tokens=10):
    """Create a simple Datum with the given number of tokens."""
    tokens = list(range(num_tokens))
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens),
        loss_fn_inputs={
            "target_tokens": tokens[1:] + [0],
            "weights": [1.0] * num_tokens,
        },
    )


def _make_datum_dict(num_tokens=10):
    """Create a datum as a plain dict."""
    tokens = list(range(num_tokens))
    return {
        "model_input": {"input_ids": tokens},
        "loss_fn_inputs": {
            "target_tokens": {
                "data": tokens[1:] + [0],
                "dtype": "int64",
                "shape": [num_tokens],
            },
            "weights": {
                "data": [1.0] * num_tokens,
                "dtype": "float32",
                "shape": [num_tokens],
            },
        },
    }


def _make_training_client():
    """Create a TrainingClient with a mock holder."""
    mock_holder = Mock(spec=ClientHolder)
    return TrainingClient(
        holder=mock_holder,
        model_id="default",
        base_model="test-model",
    )


# ---------------------------------------------------------------------------
# estimate_datum_bytes tests
# ---------------------------------------------------------------------------


class TestEstimateDatumBytes:
    """Tests for the fixed estimate_datum_bytes function."""

    def test_datum_with_tokens_attribute(self):
        """Verify estimate works with EncodedTextChunk.tokens (the bug fix)."""
        datum = _make_datum(num_tokens=100)
        estimated = estimate_datum_bytes(datum)
        # 100 tokens * 4 bytes + 100 target_tokens * 4 + 100 weights * 4 = 1200
        assert estimated >= 100 * 4  # At least the token bytes
        assert estimated > 100  # Must be larger than the minimum

    def test_datum_with_few_tokens(self):
        """Very small datum should still return at least 100 bytes."""
        datum = _make_datum(num_tokens=1)
        estimated = estimate_datum_bytes(datum)
        assert estimated >= 100  # Minimum estimate

    def test_dict_datum(self):
        """Verify estimate works with dict datums."""
        datum_dict = _make_datum_dict(num_tokens=200)
        estimated = estimate_datum_bytes(datum_dict)
        # 200 input_ids * 4 = 800 bytes from model_input alone
        assert estimated >= 200 * 4

    def test_empty_dict_datum(self):
        """Empty dict datum should return minimum."""
        estimated = estimate_datum_bytes({})
        assert estimated == 100

    def test_large_datum(self):
        """Large datum with many tokens."""
        datum = _make_datum(num_tokens=4096)
        estimated = estimate_datum_bytes(datum)
        # 4096 tokens * 4 * 3 (tokens + targets + weights) = ~49152
        assert estimated > 4096 * 4


# ---------------------------------------------------------------------------
# _chunked_datums tests
# ---------------------------------------------------------------------------


class TestChunkedDatums:
    """Tests for the _chunked_datums method on TrainingClient."""

    def test_single_datum(self):
        """Single datum should produce one chunk."""
        client = _make_training_client()
        data = [_make_datum()]
        chunks = client._chunked_datums(data)
        assert len(chunks) == 1
        request_id, chunk_data = chunks[0]
        assert len(chunk_data) == 1

    def test_empty_list(self):
        """Empty list should produce one chunk (empty)."""
        client = _make_training_client()
        chunks = client._chunked_datums([])
        assert len(chunks) == 1
        _, chunk_data = chunks[0]
        assert len(chunk_data) == 0

    def test_under_limit(self):
        """Data under the limit should produce one chunk."""
        client = _make_training_client()
        data = [_make_datum() for _ in range(100)]
        chunks = client._chunked_datums(data)
        assert len(chunks) == 1
        _, chunk_data = chunks[0]
        assert len(chunk_data) == 100

    def test_count_limit_splits(self):
        """Two full datum chunks plus a remainder should split three ways."""
        client = _make_training_client()
        total = 2 * MAX_CHUNK_LEN + 452
        data = [_make_datum(num_tokens=5) for _ in range(total)]
        chunks = client._chunked_datums(data)

        assert len(chunks) == 3
        assert len(chunks[0][1]) == MAX_CHUNK_LEN
        assert len(chunks[1][1]) == MAX_CHUNK_LEN
        assert len(chunks[2][1]) == 452

    def test_exact_limit(self):
        """Exactly MAX_CHUNK_LEN datums should produce one chunk."""
        client = _make_training_client()
        data = [_make_datum(num_tokens=5) for _ in range(MAX_CHUNK_LEN)]
        chunks = client._chunked_datums(data)
        assert len(chunks) == 1
        assert len(chunks[0][1]) == MAX_CHUNK_LEN

    def test_one_over_limit(self):
        """MAX_CHUNK_LEN + 1 should produce two chunks."""
        client = _make_training_client()
        data = [_make_datum(num_tokens=5) for _ in range(MAX_CHUNK_LEN + 1)]
        chunks = client._chunked_datums(data)
        assert len(chunks) == 2
        assert len(chunks[0][1]) == MAX_CHUNK_LEN
        assert len(chunks[1][1]) == 1

    def test_byte_limit_splits(self, monkeypatch):
        """Large datums should split based on byte limit."""
        # Pin a small byte cap: the shipped default is 512 MiB, which these 200
        # ~49KB datums (~9.8MB) would never split.
        monkeypatch.setenv("XORL_CLIENT_MAX_CHUNK_BYTES_COUNT", "5000000")
        client = _make_training_client()
        # Each datum: ~4096 tokens * 4 * 3 fields ~= 49152 bytes
        # 5MB / 49152 ~= ~100 datums per chunk
        data = [_make_datum(num_tokens=4096) for _ in range(200)]
        chunks = client._chunked_datums(data)
        # Should have more than 1 chunk due to byte limit
        assert len(chunks) > 1
        # Total datums across all chunks should equal input
        total = sum(len(c) for _, c in chunks)
        assert total == 200

    def test_request_id_sequencing(self):
        """Request IDs should be sequential across chunks."""
        client = _make_training_client()
        data = [_make_datum(num_tokens=5) for _ in range(2 * MAX_CHUNK_LEN + 1)]
        chunks = client._chunked_datums(data)

        request_ids = [rid for rid, _ in chunks]
        assert len(request_ids) == 3
        # IDs should be contiguous
        assert request_ids[1] == request_ids[0] + 1
        assert request_ids[2] == request_ids[1] + 1

    def test_request_id_counter_advances(self):
        """After chunking, the request_id counter should have advanced by len(chunks)."""
        client = _make_training_client()
        initial_counter = client._request_id_counter
        data = [_make_datum(num_tokens=5) for _ in range(2500)]
        chunks = client._chunked_datums(data)
        assert client._request_id_counter == initial_counter + len(chunks)


# ---------------------------------------------------------------------------
# _convert_datums / _convert_datums_simple tests
# ---------------------------------------------------------------------------


class TestConvertDatums:
    """Tests for datum conversion methods."""

    def test_convert_datums_with_datum_objects(self):
        """Converting Datum objects should produce dicts."""
        client = _make_training_client()
        data = [_make_datum(num_tokens=5) for _ in range(3)]
        dicts, routed = client._convert_datums(data)
        assert len(dicts) == 3
        assert len(routed) == 0
        for d in dicts:
            assert "model_input" in d
            assert "loss_fn_inputs" in d

    def test_convert_datums_with_routed_experts(self):
        """Should extract routed_experts from Datum objects."""
        client = _make_training_client()
        data = [
            types.Datum(
                model_input=types.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={"weights": [1.0, 1.0, 1.0]},
                routed_experts=[[[0, 1]], [[1, 2]], [[0, 2]]],
            )
        ]
        dicts, routed = client._convert_datums(data)
        assert len(dicts) == 1
        assert len(routed) == 1
        assert "routed_experts" not in dicts[0]  # Removed from dict

    def test_convert_datums_with_dicts(self):
        """Converting dict datums should pass through."""
        client = _make_training_client()
        data = [_make_datum_dict(num_tokens=5)]
        dicts, routed = client._convert_datums(data)
        assert len(dicts) == 1
        assert len(routed) == 0

    def test_convert_datums_simple(self):
        """Simple conversion should not extract routed_experts."""
        client = _make_training_client()
        data = [_make_datum(num_tokens=5) for _ in range(3)]
        dicts = client._convert_datums_simple(data)
        assert len(dicts) == 3

    def test_convert_datums_type_error(self):
        """Invalid datum type should raise TypeError."""
        client = _make_training_client()
        with pytest.raises(TypeError, match="Expected Datum"):
            client._convert_datums([42])

    def test_convert_datums_simple_type_error(self):
        """Invalid datum type should raise TypeError in simple conversion."""
        client = _make_training_client()
        with pytest.raises(TypeError, match="Expected Datum"):
            client._convert_datums_simple([42])


# ---------------------------------------------------------------------------
# combine_fwd_bwd_output_results tests
# ---------------------------------------------------------------------------


class TestCombineFwdBwdOutputResults:
    """Tests for combining chunked results."""

    def test_combine_empty(self):
        """Combining empty results should return empty output."""
        result = combine_fwd_bwd_output_results([])
        assert result.loss_fn_outputs == []
        assert result.metrics == {}

    def test_combine_single(self):
        """Combining single result should return it unchanged."""
        output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={"loss:mean": 0.5},
        )
        result = combine_fwd_bwd_output_results([output])
        assert len(result.loss_fn_outputs) == 1
        assert result.loss_fn_outputs[0].loss == 0.5
        assert result.loss_fn_output_type == "cross_entropy"

    def test_combine_multiple_loss_outputs(self):
        """Loss outputs should be concatenated across chunks."""
        output1 = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5), LossFnOutput(loss=0.6)],
            metrics={"loss:mean": 0.55},
        )
        output2 = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.3)],
            metrics={"loss:mean": 0.3},
        )
        result = combine_fwd_bwd_output_results([output1, output2])
        assert len(result.loss_fn_outputs) == 3
        assert result.loss_fn_outputs[0].loss == 0.5
        assert result.loss_fn_outputs[1].loss == 0.6
        assert result.loss_fn_outputs[2].loss == 0.3


# ---------------------------------------------------------------------------
# Integration: chunked forward_backward with mock HTTP layer
# ---------------------------------------------------------------------------


class TestChunkedForwardBackwardIntegration:
    """Integration test for chunked forward_backward with mocked HTTP."""

    def test_chunked_forward_backward_two_chunks(self):
        """Test that 2048 datums produces 2 POST requests with correct seq_ids."""
        # Set up mock holder with event loop
        loop = asyncio.new_event_loop()
        mock_holder = Mock(spec=ClientHolder)

        # Track all POST calls
        post_calls = []
        request_counter = [0]

        async def mock_post(endpoint, data, timeout=None):
            post_calls.append((endpoint, data))
            request_counter[0] += 1
            return {"request_id": f"future_{request_counter[0]}"}

        async def mock_execute_with_retries(fn):
            return await fn()

        mock_holder.post = AsyncMock(side_effect=mock_post)
        mock_holder.execute_with_retries = AsyncMock(
            side_effect=mock_execute_with_retries
        )
        mock_holder.run_coroutine_threadsafe = (
            lambda coro: asyncio.run_coroutine_threadsafe(coro, loop)
        )

        # Mock the _APIFuture polling to return immediately
        mock_fwd_bwd_output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={"loss:mean": 0.5, "total_tokens:sum": 1024},
        )

        client = TrainingClient(
            holder=mock_holder,
            model_id="default",
            base_model="test-model",
        )

        data = [_make_datum(num_tokens=5) for _ in range(2 * MAX_CHUNK_LEN)]

        import threading

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        try:
            with patch("xorl_client.client.training_client._APIFuture") as MockAPIFuture:
                # Make _APIFuture return immediately with mock result
                mock_internal_future = Mock()
                mock_internal_future.result_async = AsyncMock(
                    return_value=mock_fwd_bwd_output
                )
                MockAPIFuture.return_value = mock_internal_future

                future = client.forward_backward(data, "cross_entropy")
                result = future.result(timeout=10)

            # Verify 2 POST requests were made
            assert len(post_calls) == 2

            # Verify endpoints
            assert post_calls[0][0] == "/api/v1/forward_backward"
            assert post_calls[1][0] == "/api/v1/forward_backward"

            # Verify seq_ids are sequential (1-based)
            seq_id_1 = post_calls[0][1]["seq_id"]
            seq_id_2 = post_calls[1][1]["seq_id"]
            assert seq_id_2 == seq_id_1 + 1

            # Verify chunk sizes
            assert len(post_calls[0][1]["forward_backward_input"]["data"]) == MAX_CHUNK_LEN
            assert len(post_calls[1][1]["forward_backward_input"]["data"]) == MAX_CHUNK_LEN

        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_single_chunk_no_combined_future(self):
        """Test that small batches don't use _CombinedAPIFuture overhead."""
        loop = asyncio.new_event_loop()
        mock_holder = Mock(spec=ClientHolder)

        post_calls = []

        async def mock_post(endpoint, data, timeout=None):
            post_calls.append((endpoint, data))
            return {"request_id": "future_1"}

        async def mock_execute_with_retries(fn):
            return await fn()

        mock_holder.post = AsyncMock(side_effect=mock_post)
        mock_holder.execute_with_retries = AsyncMock(
            side_effect=mock_execute_with_retries
        )
        mock_holder.run_coroutine_threadsafe = (
            lambda coro: asyncio.run_coroutine_threadsafe(coro, loop)
        )

        mock_fwd_bwd_output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={},
        )

        client = TrainingClient(
            holder=mock_holder,
            model_id="default",
            base_model="test-model",
        )

        data = [_make_datum(num_tokens=5) for _ in range(100)]

        import threading

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        try:
            with patch("xorl_client.client.training_client._APIFuture") as MockAPIFuture:
                # _APIFuture constructor returns an object that is awaited.
                # We need an object whose __await__ returns an iterator yielding the result.
                class _AwaitableMock:
                    def __await__(self):
                        async def _coro():
                            return mock_fwd_bwd_output

                        return _coro().__await__()

                MockAPIFuture.return_value = _AwaitableMock()

                future = client.forward_backward(data, "cross_entropy")
                result = future.result(timeout=10)

            # Should be exactly 1 POST
            assert len(post_calls) == 1
            assert len(post_calls[0][1]["forward_backward_input"]["data"]) == 100
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_chunked_forward_two_chunks(self):
        """Test that forward() also chunks large batches."""
        loop = asyncio.new_event_loop()
        mock_holder = Mock(spec=ClientHolder)

        post_calls = []
        request_counter = [0]

        async def mock_post(endpoint, data, timeout=None):
            post_calls.append((endpoint, data))
            request_counter[0] += 1
            return {"request_id": f"future_{request_counter[0]}"}

        async def mock_execute_with_retries(fn):
            return await fn()

        mock_holder.post = AsyncMock(side_effect=mock_post)
        mock_holder.execute_with_retries = AsyncMock(
            side_effect=mock_execute_with_retries
        )
        mock_holder.run_coroutine_threadsafe = (
            lambda coro: asyncio.run_coroutine_threadsafe(coro, loop)
        )

        mock_fwd_output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.3)],
            metrics={"loss:mean": 0.3},
        )

        client = TrainingClient(
            holder=mock_holder,
            model_id="default",
            base_model="test-model",
        )

        data = [_make_datum(num_tokens=5) for _ in range(2 * MAX_CHUNK_LEN)]

        import threading

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        try:
            with patch("xorl_client.client.training_client._APIFuture") as MockAPIFuture:
                mock_internal_future = Mock()
                mock_internal_future.result_async = AsyncMock(
                    return_value=mock_fwd_output
                )
                MockAPIFuture.return_value = mock_internal_future

                future = client.forward(data, "cross_entropy")
                result = future.result(timeout=10)

            # Verify 2 POST requests to /api/v1/forward
            assert len(post_calls) == 2
            assert post_calls[0][0] == "/api/v1/forward"
            assert post_calls[1][0] == "/api/v1/forward"

            # Verify chunk sizes
            assert len(post_calls[0][1]["forward_input"]["data"]) == MAX_CHUNK_LEN
            assert len(post_calls[1][1]["forward_input"]["data"]) == MAX_CHUNK_LEN

        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_optim_step_after_chunked_forward_backward(self):
        """Verify optim_step gets a seq_id after all chunk seq_ids."""
        client = _make_training_client()

        data = [_make_datum(num_tokens=5) for _ in range(2 * MAX_CHUNK_LEN + 1)]
        chunks = client._chunked_datums(data)
        assert len(chunks) == 3

        # The next request_id should be after all chunks
        next_id = client._get_request_id()
        last_chunk_id = chunks[-1][0]
        assert next_id == last_chunk_id + 1


# ---------------------------------------------------------------------------
# _infer_reduction_type tests
# ---------------------------------------------------------------------------


class TestInferReductionType:
    """Tests for the _infer_reduction_type helper."""

    def test_min_suffix(self):
        assert _infer_reduction_type("is_ratio_min") == "min"

    def test_max_suffix(self):
        assert _infer_reduction_type("is_ratio_max") == "max"

    def test_valid_tokens_sum(self):
        assert _infer_reduction_type("valid_tokens") == "sum"

    def test_is_valid_tokens_sum(self):
        assert _infer_reduction_type("is_valid_tokens") == "sum"

    def test_execution_time_sum(self):
        assert _infer_reduction_type("execution_time") == "sum"

    def test_default_mean(self):
        assert _infer_reduction_type("is_kl_sample_train_k3") == "mean"
        assert _infer_reduction_type("is_entropy_sample") == "mean"


# ---------------------------------------------------------------------------
# IS metrics preservation in multi-chunk combine
# ---------------------------------------------------------------------------


class TestCombineISMetrics:
    """Tests that IS metrics (without `:` suffix) survive multi-chunk combining."""

    def _make_chunk_output(self, n_outputs, metrics):
        return ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5) for _ in range(n_outputs)],
            metrics=metrics,
        )

    def test_is_metrics_preserved(self):
        """All IS metrics without `:` suffix should be preserved after combine."""
        metrics_chunk1 = {
            "loss:mean": 0.5,
            "is_kl_sample_train_k3": 0.10,
            "is_entropy_sample": 1.2,
            "is_ratio_mean": 0.95,
            "is_ratio_min": 0.80,
            "is_ratio_max": 1.10,
            "is_valid_tokens": 500,
            "is_pg_clipfrac": 0.02,
            "valid_tokens": 1000,
            "execution_time": 2.0,
        }
        metrics_chunk2 = {
            "loss:mean": 0.6,
            "is_kl_sample_train_k3": 0.20,
            "is_entropy_sample": 1.4,
            "is_ratio_mean": 1.05,
            "is_ratio_min": 0.70,
            "is_ratio_max": 1.30,
            "is_valid_tokens": 600,
            "is_pg_clipfrac": 0.04,
            "valid_tokens": 1200,
            "execution_time": 3.0,
        }

        chunk1 = self._make_chunk_output(4, metrics_chunk1)
        chunk2 = self._make_chunk_output(6, metrics_chunk2)
        result = combine_fwd_bwd_output_results([chunk1, chunk2])

        # Every key from the input should appear in the output
        for key in metrics_chunk1:
            assert key in result.metrics, f"Metric {key} was dropped"

    def test_is_metrics_weighted_mean(self):
        """IS metrics without suffix should use weighted mean (by datum count)."""
        chunk1 = self._make_chunk_output(
            4, {"is_kl_sample_train_k3": 1.0, "is_entropy_sample": 2.0}
        )
        chunk2 = self._make_chunk_output(
            6, {"is_kl_sample_train_k3": 2.0, "is_entropy_sample": 4.0}
        )
        result = combine_fwd_bwd_output_results([chunk1, chunk2])

        # Weighted mean: (4*1.0 + 6*2.0) / (4+6) = 16/10 = 1.6
        assert abs(result.metrics["is_kl_sample_train_k3"] - 1.6) < 1e-9
        # Weighted mean: (4*2.0 + 6*4.0) / 10 = 32/10 = 3.2
        assert abs(result.metrics["is_entropy_sample"] - 3.2) < 1e-9

    def test_is_ratio_min_uses_min(self):
        """is_ratio_min should use min reduction."""
        chunk1 = self._make_chunk_output(4, {"is_ratio_min": 0.8})
        chunk2 = self._make_chunk_output(6, {"is_ratio_min": 0.7})
        result = combine_fwd_bwd_output_results([chunk1, chunk2])
        assert result.metrics["is_ratio_min"] == 0.7

    def test_is_ratio_max_uses_max(self):
        """is_ratio_max should use max reduction."""
        chunk1 = self._make_chunk_output(4, {"is_ratio_max": 1.1})
        chunk2 = self._make_chunk_output(6, {"is_ratio_max": 1.3})
        result = combine_fwd_bwd_output_results([chunk1, chunk2])
        assert result.metrics["is_ratio_max"] == 1.3

    def test_valid_tokens_uses_sum(self):
        """valid_tokens and is_valid_tokens should use sum reduction."""
        chunk1 = self._make_chunk_output(
            4, {"valid_tokens": 1000, "is_valid_tokens": 500}
        )
        chunk2 = self._make_chunk_output(
            6, {"valid_tokens": 1200, "is_valid_tokens": 600}
        )
        result = combine_fwd_bwd_output_results([chunk1, chunk2])
        assert result.metrics["valid_tokens"] == 2200
        assert result.metrics["is_valid_tokens"] == 1100

    def test_mixed_suffixed_and_unsuffixed(self):
        """Both `:` suffixed and plain keys should coexist correctly."""
        chunk1 = self._make_chunk_output(
            4, {"loss:mean": 0.5, "is_kl_sample_train_k3": 0.1, "total_tokens:sum": 100}
        )
        chunk2 = self._make_chunk_output(
            6, {"loss:mean": 0.6, "is_kl_sample_train_k3": 0.2, "total_tokens:sum": 200}
        )
        result = combine_fwd_bwd_output_results([chunk1, chunk2])

        # Suffixed keys still work
        assert "loss:mean" in result.metrics
        assert "total_tokens:sum" in result.metrics
        assert result.metrics["total_tokens:sum"] == 300

        # Unsuffixed keys are preserved
        assert "is_kl_sample_train_k3" in result.metrics
