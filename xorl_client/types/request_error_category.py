"""
RequestErrorCategory - Categorization of request errors.

Used in RequestFailedResponse to indicate whether an error is a server error
(retryable) or a user error (not retryable).
"""

from __future__ import annotations

from enum import StrEnum, auto


class RequestErrorCategory(StrEnum):
    """Category of error that caused a request to fail.

    Attributes:
        Unknown: Error category could not be determined.
        Server: Server-side error (may be retryable).
        User: User/client error (not retryable, e.g., invalid input).

    Example:
        >>> if error.category == RequestErrorCategory.User:
        ...     # Don't retry, fix the request
        ...     raise ValueError(error.error)
        >>> elif error.category == RequestErrorCategory.Server:
        ...     # May retry
        ...     pass
    """

    Unknown = "unknown"
    Server = "server"
    User = "user"
