"""Tests for xorl_client exceptions module."""

import pytest

from xorl_client.exceptions import (
    XorlClientError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError,
)


class TestXorlClientError:
    """Tests for base XorlClientError class."""

    def test_xorl_client_error_is_exception(self):
        """Test that XorlClientError is an Exception."""
        err = XorlClientError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


class TestAPIError:
    """Tests for APIError class."""

    def test_api_error_with_url(self):
        """Test APIError with URL."""
        err = APIError("Something went wrong", url="http://localhost:6000/api/v1/test")
        assert "Something went wrong" in str(err)
        assert "http://localhost:6000/api/v1/test" in str(err)

    def test_api_error_without_url(self):
        """Test APIError without URL."""
        err = APIError("Something went wrong")
        assert str(err) == "Something went wrong"


class TestAPIConnectionError:
    """Tests for APIConnectionError class."""

    def test_connection_error_message(self):
        """Test that APIConnectionError has helpful message."""
        err = APIConnectionError("http://localhost:6000/api/v1/create_model")

        error_str = str(err)
        assert "localhost:6000" in error_str
        assert "XoRL server is running" in error_str
        assert "server address is correct" in error_str

    def test_connection_error_with_original_exception(self):
        """Test APIConnectionError preserves original exception."""
        original = ConnectionRefusedError("Connection refused")
        err = APIConnectionError("http://localhost:6000/api", original)

        assert "ConnectionRefusedError" in str(err)
        assert err.__cause__ is original


class TestAPITimeoutError:
    """Tests for APITimeoutError class."""

    def test_timeout_error_message(self):
        """Test that APITimeoutError has helpful message."""
        err = APITimeoutError("http://localhost:6000/api/v1/forward_backward", 300.0)

        error_str = str(err)
        assert "timed out" in error_str
        assert "300" in error_str
        assert "timeout" in error_str.lower()

    def test_timeout_error_is_connection_error(self):
        """Test that APITimeoutError is subclass of APIConnectionError."""
        err = APITimeoutError("http://localhost:6000/api", 60.0)
        # Note: APITimeoutError inherits from APIConnectionError but overrides __init__
        assert isinstance(err, APIError)


class TestAPIStatusError:
    """Tests for APIStatusError class."""

    def test_status_error_properties(self):
        """Test APIStatusError has status_code property."""
        err = APIStatusError("Not found", "http://localhost:6000/api", 404, "response body")

        assert err.status_code == 404
        assert err.response_body == "response body"
        assert "Not found" in str(err)


class TestBadRequestError:
    """Tests for BadRequestError class."""

    def test_bad_request_error(self):
        """Test BadRequestError."""
        err = BadRequestError(
            "http://localhost:6000/api",
            "Invalid model_id format",
            '{"detail": "Invalid model_id format"}'
        )

        assert err.status_code == 400
        assert "Invalid model_id format" in str(err)


class TestAuthenticationError:
    """Tests for AuthenticationError class."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        err = AuthenticationError("http://localhost:6000/api")

        assert err.status_code == 401
        assert "Authentication failed" in str(err)
        assert "API key" in str(err)


class TestNotFoundError:
    """Tests for NotFoundError class."""

    def test_not_found_error(self):
        """Test NotFoundError."""
        err = NotFoundError("http://localhost:6000/api/v1/model/abc123", "Model not found")

        assert err.status_code == 404
        assert "not found" in str(err).lower()


class TestInternalServerError:
    """Tests for InternalServerError class."""

    def test_internal_server_error(self):
        """Test InternalServerError."""
        err = InternalServerError("http://localhost:6000/api", 500, "Database connection failed")

        assert err.status_code == 500
        assert "500" in str(err)
        assert "Database connection failed" in str(err)

    def test_internal_server_error_502(self):
        """Test InternalServerError with 502 status."""
        err = InternalServerError("http://localhost:6000/api", 502, "Bad gateway")

        assert err.status_code == 502
        assert "502" in str(err)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_errors_inherit_from_xorl_client_error(self):
        """Test that all errors inherit from XorlClientError."""
        errors = [
            APIError("test"),
            APIConnectionError("http://localhost:6000"),
            APITimeoutError("http://localhost:6000", 60.0),
            APIStatusError("test", "http://localhost:6000", 400),
            BadRequestError("http://localhost:6000", "test"),
            AuthenticationError("http://localhost:6000"),
            NotFoundError("http://localhost:6000"),
            InternalServerError("http://localhost:6000", 500),
        ]

        for err in errors:
            assert isinstance(err, XorlClientError), f"{type(err).__name__} should inherit from XorlClientError"

    def test_status_errors_inherit_from_api_status_error(self):
        """Test that HTTP status errors inherit from APIStatusError."""
        errors = [
            BadRequestError("http://localhost:6000", "test"),
            AuthenticationError("http://localhost:6000"),
            NotFoundError("http://localhost:6000"),
            InternalServerError("http://localhost:6000", 500),
        ]

        for err in errors:
            assert isinstance(err, APIStatusError), f"{type(err).__name__} should inherit from APIStatusError"
