"""
FutureRetrieveRequest - Request to retrieve async operation result.

Used in Phase 2 of the two-phase request pattern to poll for results.
"""

from __future__ import annotations

from .strict_base import StrictBase
from .request_id import RequestID


class FutureRetrieveRequest(StrictBase):
    """Request to retrieve the result of an async operation.

    Sent to /api/v1/retrieve_future endpoint with the request_id obtained
    from the initial request (UntypedAPIFuture).

    Attributes:
        request_id: The ID of the request to retrieve results for.

    Example:
        >>> # After getting UntypedAPIFuture from forward_backward
        >>> retrieve_request = FutureRetrieveRequest(
        ...     request_id=untyped_future.request_id
        ... )
        >>> response = await client.post(
        ...     "/api/v1/retrieve_future",
        ...     retrieve_request.to_dict()
        ... )
    """

    request_id: RequestID

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"request_id": self.request_id}

    @classmethod
    def from_dict(cls, data: dict) -> "FutureRetrieveRequest":
        """Create from dictionary."""
        return cls(request_id=data["request_id"])
