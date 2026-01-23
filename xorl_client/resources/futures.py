"""
Futures resource - low-level API for retrieving async results.

Maps to:
- POST /api/v1/retrieve_future
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AsyncResource

__all__ = ["FuturesResource"]


class FuturesResource(AsyncResource):
    """Low-level resource for retrieving async operation results.

    This is used in the two-phase request pattern:
    1. Phase 1: Submit request to endpoint -> Get UntypedAPIFuture with request_id
    2. Phase 2: Poll retrieve_future with request_id -> Get actual result

    Usage:
        futures = FuturesResource(holder)
        result = await futures.retrieve(request_id, model_id)
        # Returns TryAgainResponse, actual result, or RequestFailedResponse
    """

    async def retrieve(
        self,
        request_id: str,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve the result of an async operation.

        Args:
            request_id: The request ID from UntypedAPIFuture
            model_id: Optional model ID for routing
            timeout: Optional timeout override (for long polling)

        Returns:
            One of:
            - TryAgainResponse: Result not ready, continue polling
            - Actual result dict: Operation completed successfully
            - RequestFailedResponse: Operation failed
        """
        request_data = {
            "request_id": request_id,
        }
        if model_id:
            request_data["model_id"] = model_id

        return await self._post("/api/v1/retrieve_future", request_data, timeout)
