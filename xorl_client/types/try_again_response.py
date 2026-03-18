"""
TryAgainResponse - Server response indicating request is still processing.

When polling /api/v1/retrieve_future, if the request is not yet complete,
the server returns a TryAgainResponse. The client should continue polling.
"""

from __future__ import annotations

from typing import Literal, Optional

from .strict_base import StrictBase


class TryAgainResponse(StrictBase):
    """Response indicating the request is still being processed.

    The client should continue polling /api/v1/retrieve_future until
    a different response type is received.

    Attributes:
        type: Always "try_again" to identify this response type.
        request_id: The request ID being polled.
        queue_state: Current state of the request in the server's queue.
            - "active": Request is actively being processed
            - "paused_capacity": Waiting for compute capacity
            - "paused_rate_limit": Rate limited, waiting
        queue_state_reason: Optional human-readable reason for the queue state.

    Example:
        >>> response = await client.post("/api/v1/retrieve_future",
        ...     {"request_id": "future_abc123"})
        >>> if response.get("type") == "try_again":
        ...     try_again = TryAgainResponse.from_dict(response)
        ...     if try_again.queue_state == "paused_capacity":
        ...         logger.warning("Server at capacity, waiting...")
        ...     # Continue polling
    """

    type: Literal["try_again"] = "try_again"
    request_id: str
    queue_state: Literal["active", "paused_capacity", "paused_rate_limit"]
    queue_state_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "type": self.type,
            "request_id": self.request_id,
            "queue_state": self.queue_state,
        }
        if self.queue_state_reason is not None:
            result["queue_state_reason"] = self.queue_state_reason
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TryAgainResponse":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "try_again"),
            request_id=data["request_id"],
            queue_state=data["queue_state"],
            queue_state_reason=data.get("queue_state_reason"),
        )
