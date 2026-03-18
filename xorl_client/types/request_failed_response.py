"""
RequestFailedResponse - Server response indicating request failed.

When polling /api/v1/retrieve_future, if the request failed, the server
returns a RequestFailedResponse with error details and category.
"""

from __future__ import annotations

from .strict_base import StrictBase
from .request_error_category import RequestErrorCategory


class RequestFailedResponse(StrictBase):
    """Response indicating the request failed.

    Contains the error message and category to help the client decide
    whether to retry or report the error to the user.

    Attributes:
        error: Human-readable error message.
        category: Category of error (Unknown, Server, or User).

    Example:
        >>> response = await client.post("/api/v1/retrieve_future",
        ...     {"request_id": "future_abc123"})
        >>> if "error" in response:
        ...     failed = RequestFailedResponse.from_dict(response)
        ...     if failed.category == RequestErrorCategory.User:
        ...         raise ValueError(f"Invalid request: {failed.error}")
        ...     else:
        ...         raise RuntimeError(f"Server error: {failed.error}")
    """

    error: str
    category: RequestErrorCategory

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "error": self.error,
            "category": self.category.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RequestFailedResponse":
        """Create from dictionary."""
        category = data.get("category", "unknown")
        if isinstance(category, str):
            try:
                category = RequestErrorCategory(category)
            except ValueError:
                category = RequestErrorCategory.Unknown
        return cls(
            error=data["error"],
            category=category,
        )
