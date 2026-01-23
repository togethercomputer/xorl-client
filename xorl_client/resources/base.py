"""
Base class for API resources.

Provides common functionality for low-level HTTP resource wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from xorl_client.client.client_holder import ClientHolder


class AsyncResource:
    """Base class for async API resources.

    Resources are thin HTTP wrappers that directly map to server endpoints.
    They handle HTTP requests and response parsing but don't add business logic.

    Args:
        holder: The ClientHolder that manages HTTP connections
    """

    def __init__(self, holder: ClientHolder):
        self._holder = holder

    async def _post(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        """Send a POST request to the API.

        Args:
            endpoint: API endpoint path (e.g., "/api/v1/forward")
            body: Request body as dictionary
            timeout: Optional timeout override

        Returns:
            Response JSON as dictionary
        """
        return await self._holder.post(endpoint, body, timeout)

    async def _get(
        self,
        endpoint: str,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        """Send a GET request to the API.

        Args:
            endpoint: API endpoint path (e.g., "/api/v1/health")
            timeout: Optional timeout override

        Returns:
            Response JSON as dictionary
        """
        return await self._holder.get(endpoint, timeout)
