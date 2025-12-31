"""
Exception classes for the XoRL API client.

These exceptions provide clear, user-friendly error messages for common error conditions.
"""

from __future__ import annotations

from typing import Optional


__all__ = [
    "XorlClientError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "BadRequestError",
    "AuthenticationError",
    "NotFoundError",
    "InternalServerError",
]


class XorlClientError(Exception):
    """Base exception for all XoRL-related errors."""

    pass


class APIError(XorlClientError):
    """Base class for all API-related errors."""

    message: str
    url: Optional[str]

    def __init__(self, message: str, url: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.url = url

    def __str__(self) -> str:
        if self.url:
            return f"{self.message} (url: {self.url})"
        return self.message


class APIConnectionError(APIError):
    """Raised when unable to connect to the XoRL API server.

    Common causes:
    - The server is not running
    - Wrong host or port
    - Network connectivity issues
    - Firewall blocking the connection
    """

    def __init__(self, url: str, original_error: Optional[Exception] = None) -> None:
        # Extract host:port from URL for clearer messaging
        host_port = url.replace("http://", "").replace("https://", "").split("/")[0]

        message = (
            f"Failed to connect to XoRL server at {host_port}.\n"
            f"\n"
            f"Please check that:\n"
            f"  1. The XoRL server is running\n"
            f"  2. The server address is correct (base_url=\"{url.rsplit('/', 1)[0] if '/' in url else url}\")\n"
            f"  3. There are no firewall or network issues blocking the connection"
        )

        if original_error:
            # Add the original error type for debugging
            error_type = type(original_error).__name__
            message += f"\n\nOriginal error: {error_type}"

        super().__init__(message, url)
        self.__cause__ = original_error


class APITimeoutError(APIConnectionError):
    """Raised when an API request times out."""

    def __init__(self, url: str, timeout: float, original_error: Optional[Exception] = None) -> None:
        message = (
            f"Request to XoRL server timed out after {timeout} seconds.\n"
            f"\n"
            f"The server at {url} did not respond in time. This could mean:\n"
            f"  1. The server is overloaded or processing a large request\n"
            f"  2. Network latency is high\n"
            f"  3. The timeout value is too low (current: {timeout}s)\n"
            f"\n"
            f"Try increasing the timeout: ServiceClient(base_url=..., timeout=600)"
        )
        # Call APIError.__init__ directly to avoid APIConnectionError's message
        APIError.__init__(self, message, url)
        self.__cause__ = original_error


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    status_code: int
    response_body: Optional[str]

    def __init__(
        self,
        message: str,
        url: str,
        status_code: int,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message, url)
        self.status_code = status_code
        self.response_body = response_body


class BadRequestError(APIStatusError):
    """HTTP 400: The request was invalid or malformed."""

    def __init__(self, url: str, detail: str, response_body: Optional[str] = None) -> None:
        message = f"Bad request: {detail}"
        super().__init__(message, url, 400, response_body)


class AuthenticationError(APIStatusError):
    """HTTP 401: Authentication credentials are missing or invalid."""

    def __init__(self, url: str, detail: Optional[str] = None) -> None:
        message = "Authentication failed. Please check your API key."
        if detail:
            message += f" Detail: {detail}"
        super().__init__(message, url, 401)


class NotFoundError(APIStatusError):
    """HTTP 404: The requested resource was not found."""

    def __init__(self, url: str, detail: Optional[str] = None) -> None:
        message = f"Resource not found: {url}"
        if detail:
            message += f" - {detail}"
        super().__init__(message, url, 404)


class InternalServerError(APIStatusError):
    """HTTP 500+: An error occurred on the server."""

    def __init__(self, url: str, status_code: int, detail: Optional[str] = None) -> None:
        message = f"Server error (HTTP {status_code})"
        if detail:
            message += f": {detail}"
        super().__init__(message, url, status_code)
