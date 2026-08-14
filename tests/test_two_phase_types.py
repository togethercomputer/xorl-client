"""
Tests for two-phase request pattern types in xorl_client.
"""

import pytest
from xorl_client.types import (
    RequestID,
    TensorData,
    UntypedAPIFuture,
    TryAgainResponse,
    RequestErrorCategory,
    RequestFailedResponse,
    FutureRetrieveRequest,
    parse_future_retrieve_response,
    ForwardBackwardOutput,
    LossFnOutput,
)


class TestRequestID:
    """Tests for RequestID type alias."""

    def test_request_id_is_string(self):
        """RequestID should be a string type alias."""
        request_id: RequestID = "future_abc123"
        assert isinstance(request_id, str)
        assert request_id == "future_abc123"


class TestUntypedAPIFuture:
    """Tests for UntypedAPIFuture class."""

    def test_basic_creation(self):
        """Test basic UntypedAPIFuture creation."""
        future = UntypedAPIFuture(request_id="future_abc123")
        assert future.request_id == "future_abc123"
        assert future.model_id is None

    def test_with_model_id(self):
        """Test UntypedAPIFuture with model_id."""
        future = UntypedAPIFuture(
            request_id="future_def456",
            model_id="model-123"
        )
        assert future.request_id == "future_def456"
        assert future.model_id == "model-123"

    def test_to_dict(self):
        """Test to_dict method."""
        future = UntypedAPIFuture(
            request_id="future_abc",
            model_id="model-1"
        )
        d = future.to_dict()
        assert d["request_id"] == "future_abc"
        assert d["model_id"] == "model-1"

    def test_to_dict_without_model_id(self):
        """Test to_dict excludes model_id when None."""
        future = UntypedAPIFuture(request_id="future_abc")
        d = future.to_dict()
        assert d["request_id"] == "future_abc"
        assert "model_id" not in d

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {"request_id": "future_xyz", "model_id": "model-5"}
        future = UntypedAPIFuture.from_dict(data)
        assert future.request_id == "future_xyz"
        assert future.model_id == "model-5"

    def test_from_dict_without_model_id(self):
        """Test from_dict with missing model_id."""
        data = {"request_id": "future_xyz"}
        future = UntypedAPIFuture.from_dict(data)
        assert future.request_id == "future_xyz"
        assert future.model_id is None


class TestTryAgainResponse:
    """Tests for TryAgainResponse class."""

    def test_basic_creation(self):
        """Test basic TryAgainResponse creation."""
        response = TryAgainResponse(
            request_id="future_abc",
            queue_state="active"
        )
        assert response.type == "try_again"
        assert response.request_id == "future_abc"
        assert response.queue_state == "active"

    def test_paused_capacity(self):
        """Test TryAgainResponse with paused_capacity state."""
        response = TryAgainResponse(
            request_id="future_xyz",
            queue_state="paused_capacity"
        )
        assert response.queue_state == "paused_capacity"

    def test_paused_rate_limit(self):
        """Test TryAgainResponse with paused_rate_limit state."""
        response = TryAgainResponse(
            request_id="future_xyz",
            queue_state="paused_rate_limit"
        )
        assert response.queue_state == "paused_rate_limit"

    def test_to_dict(self):
        """Test to_dict method."""
        response = TryAgainResponse(
            request_id="future_123",
            queue_state="active"
        )
        d = response.to_dict()
        assert d["type"] == "try_again"
        assert d["request_id"] == "future_123"
        assert d["queue_state"] == "active"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "type": "try_again",
            "request_id": "future_456",
            "queue_state": "paused_capacity"
        }
        response = TryAgainResponse.from_dict(data)
        assert response.type == "try_again"
        assert response.request_id == "future_456"
        assert response.queue_state == "paused_capacity"


class TestRequestErrorCategory:
    """Tests for RequestErrorCategory enum."""

    def test_unknown(self):
        """Test Unknown category."""
        assert RequestErrorCategory.Unknown == "unknown"

    def test_server(self):
        """Test Server category."""
        assert RequestErrorCategory.Server == "server"

    def test_user(self):
        """Test User category."""
        assert RequestErrorCategory.User == "user"


class TestRequestFailedResponse:
    """Tests for RequestFailedResponse class."""

    def test_basic_creation(self):
        """Test basic RequestFailedResponse creation."""
        response = RequestFailedResponse(
            error="Something went wrong",
            category=RequestErrorCategory.Server
        )
        assert response.error == "Something went wrong"
        assert response.category == RequestErrorCategory.Server

    def test_user_error(self):
        """Test RequestFailedResponse with user error."""
        response = RequestFailedResponse(
            error="Invalid input",
            category=RequestErrorCategory.User
        )
        assert response.category == RequestErrorCategory.User

    def test_to_dict(self):
        """Test to_dict method."""
        response = RequestFailedResponse(
            error="Test error",
            category=RequestErrorCategory.Unknown
        )
        d = response.to_dict()
        assert d["error"] == "Test error"
        assert d["category"] == "unknown"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {"error": "Server error", "category": "server"}
        response = RequestFailedResponse.from_dict(data)
        assert response.error == "Server error"
        assert response.category == RequestErrorCategory.Server

    def test_from_dict_unknown_category(self):
        """Test from_dict with unknown category value."""
        data = {"error": "Some error", "category": "invalid_category"}
        response = RequestFailedResponse.from_dict(data)
        assert response.category == RequestErrorCategory.Unknown


class TestFutureRetrieveRequest:
    """Tests for FutureRetrieveRequest class."""

    def test_basic_creation(self):
        """Test basic FutureRetrieveRequest creation."""
        request = FutureRetrieveRequest(request_id="future_abc")
        assert request.request_id == "future_abc"

    def test_to_dict(self):
        """Test to_dict method."""
        request = FutureRetrieveRequest(request_id="future_xyz")
        d = request.to_dict()
        assert d == {"request_id": "future_xyz"}

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {"request_id": "future_123"}
        request = FutureRetrieveRequest.from_dict(data)
        assert request.request_id == "future_123"


class TestParseFutureRetrieveResponse:
    """Tests for parse_future_retrieve_response function."""

    def test_parse_try_again(self):
        """Test parsing TryAgainResponse."""
        data = {
            "type": "try_again",
            "request_id": "future_abc",
            "queue_state": "active"
        }
        result = parse_future_retrieve_response(data)
        assert isinstance(result, TryAgainResponse)
        assert result.request_id == "future_abc"

    def test_parse_error(self):
        """Test parsing RequestFailedResponse."""
        data = {
            "error": "Something failed",
            "category": "server"
        }
        result = parse_future_retrieve_response(data)
        assert isinstance(result, RequestFailedResponse)
        assert result.error == "Something failed"

    def test_parse_forward_backward_output(self):
        """Test parsing ForwardBackwardOutput."""
        data = {
            "loss_fn_output_type": "cross_entropy",
            "loss_fn_outputs": [{"loss": 0.5}],
            "metrics": {"loss:mean": 0.5}
        }
        result = parse_future_retrieve_response(data)
        assert isinstance(result, ForwardBackwardOutput)
        assert result.loss_fn_output_type == "cross_entropy"
        assert result.metrics["loss:mean"] == 0.5


class TestForwardBackwardOutput:
    """Tests for ForwardBackwardOutput class."""

    def test_basic_creation(self):
        """Test basic ForwardBackwardOutput creation."""
        output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={"loss:mean": 0.5}
        )
        assert output.loss_fn_output_type == "cross_entropy"
        assert len(output.loss_fn_outputs) == 1
        assert output.loss_fn_outputs[0].loss == 0.5
        assert output.metrics["loss:mean"] == 0.5

    def test_to_dict(self):
        """Test to_dict method."""
        output = ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[LossFnOutput(loss=0.5)],
            metrics={"loss:mean": 0.5}
        )
        d = output.to_dict()
        assert d["loss_fn_output_type"] == "cross_entropy"
        assert d["metrics"]["loss:mean"] == 0.5

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "loss_fn_output_type": "cross_entropy",
            "loss_fn_outputs": [{"loss": 0.3}],
            "metrics": {"loss:mean": 0.3}
        }
        output = ForwardBackwardOutput.from_dict(data)
        assert output.loss_fn_output_type == "cross_entropy"
        assert output.loss_fn_outputs[0].loss == 0.3
        assert output.metrics["loss:mean"] == 0.3


class TestLossFnOutput:
    """Tests for LossFnOutput class."""

    def test_basic_creation(self):
        """Test basic LossFnOutput creation."""
        output = LossFnOutput(loss=0.5)
        assert output.loss == 0.5
        assert output.logprobs is None
        assert output.elementwise_loss is None

    def test_scalar_tensor_loss_and_optional_k3(self):
        output = LossFnOutput.from_dict(
            {
                "loss": {"data": [0.25], "dtype": "float32", "shape": []},
                "k3": 0.0,
            }
        )
        assert output.loss == 0.25
        assert output.k3 == 0.0
        assert output.to_dict()["k3"] == 0.0

    def test_with_all_fields(self):
        """Test LossFnOutput with all fields."""
        output = LossFnOutput(
            loss=0.5,
            logprobs={"data": [-1.0, -2.0]},
            elementwise_loss={"data": [0.3, 0.7]}
        )
        assert output.loss == 0.5
        assert output.logprobs is not None
        assert output.elementwise_loss is not None

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        output = LossFnOutput(loss=0.5)
        d = output.to_dict()
        assert "loss" in d
        assert "logprobs" not in d
        assert "elementwise_loss" not in d

    def test_from_dict(self):
        """Test from_dict class method (logprobs converted to TensorData)."""
        data = {"loss": 0.7, "logprobs": {"data": [-1.5]}}
        output = LossFnOutput.from_dict(data)
        assert output.loss == 0.7
        assert isinstance(output.logprobs, TensorData)
        assert output.logprobs.data == [-1.5]
        assert output.elementwise_loss is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
