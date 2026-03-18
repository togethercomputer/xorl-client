"""
UntypedAPIFuture - Server response for Phase 1 of two-phase request pattern.

When a client submits a request (e.g., forward_backward), the server immediately
returns an UntypedAPIFuture containing a unique request_id. The client then polls
the /api/v1/retrieve_future endpoint with this request_id to get the actual result.
"""

from __future__ import annotations

from typing import Optional

from .strict_base import StrictBase
from .request_id import RequestID
from .model_id import ModelID


class UntypedAPIFuture(StrictBase):
    """Server response containing a request_id for async result retrieval.

    This is returned by endpoints like /api/v1/forward_backward in Phase 1
    of the two-phase request pattern.

    Attributes:
        request_id: Unique identifier for this async request. Used to poll
            for results via /api/v1/retrieve_future.
        model_id: Optional model identifier associated with this request.

    Example:
        >>> # Phase 1: Submit request
        >>> response = await client.post("/api/v1/forward_backward", data)
        >>> untyped_future = UntypedAPIFuture.from_dict(response)
        >>> print(untyped_future.request_id)  # "future_a1b2c3d4"
        >>>
        >>> # Phase 2: Poll for result
        >>> result = await client.post("/api/v1/retrieve_future",
        ...     {"request_id": untyped_future.request_id})
    """

    request_id: RequestID
    model_id: Optional[ModelID] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"request_id": self.request_id}
        if self.model_id is not None:
            result["model_id"] = self.model_id
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UntypedAPIFuture":
        """Create from dictionary.

        Raises:
            KeyError: If 'request_id' is missing. This typically means the server
                is running old code that doesn't support the two-phase pattern.
        """
        if "request_id" not in data:
            # Provide helpful error message for debugging
            raise KeyError(
                f"'request_id' not found in server response. "
                f"Got keys: {list(data.keys())}. "
                f"This may indicate the server is running old code that doesn't "
                f"support the two-phase request pattern. Please restart the server."
            )
        return cls(
            request_id=data["request_id"],
            model_id=data.get("model_id"),
        )
