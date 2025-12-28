"""
ClientHolder - Manages API connections and session state.

This is the internal class that handles HTTP connections, session management,
model IDs, and async execution. ServiceClient and clients use this internally.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ClientHolder:
    """Manages API connections and session state for XoRL clients.

    This class:
    - Manages HTTP session and base URL
    - Tracks session ID and model IDs
    - Provides async execution via thread pool
    - Handles worker routing for sampling

    Args:
        base_url: Base URL for the training API (e.g., "http://localhost:5555")
        timeout: Default timeout for HTTP requests in seconds
        **kwargs: Additional arguments (for future compatibility)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5555",
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Session management
        self._session_id = str(uuid.uuid4())
        self._model_counter = 0
        self._model_counter_lock = threading.Lock()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="xorl_client-client")

        # Worker registry: maps model_path -> worker URLs
        self._worker_registry: Dict[str, list[str]] = {}
        self._worker_registry_lock = threading.Lock()

        # HTTP session
        self._http_session = requests.Session()

        logger.info(f"ClientHolder initialized: session_id={self._session_id}, base_url={self.base_url}")

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    def get_model_id(self) -> str:
        """Generate a new model ID.

        Returns:
            New model ID (e.g., "model-123")
        """
        with self._model_counter_lock:
            model_seq = self._model_counter
            self._model_counter += 1
        model_id = f"model-{model_seq}"
        logger.info(f"Generated model_id: {model_id}")
        return model_id

    def register_workers_for_model(self, model_path: str, worker_urls: list[str]):
        """Register which workers serve a particular model.

        Args:
            model_path: Model path (e.g., "xorl://model-123/step-100")
            worker_urls: List of worker URLs that serve this model
        """
        with self._worker_registry_lock:
            self._worker_registry[model_path] = worker_urls
            logger.info(f"Registered workers for {model_path}: {worker_urls}")

    def get_workers_for_model(self, model_path: str) -> list[str]:
        """Get worker URLs for a model path.

        First checks local cache, then queries the training server.

        Args:
            model_path: Model path

        Returns:
            List of worker URLs

        Raises:
            ValueError: If model_path is not registered
        """
        with self._worker_registry_lock:
            # Check local cache first
            if model_path in self._worker_registry:
                return self._worker_registry[model_path]

        # Query server for worker URLs
        try:
            response = self.get(f"/api/v1/get_workers?model_path={model_path}", timeout=10)
            worker_urls = response.get("worker_urls", [])
            if worker_urls:
                with self._worker_registry_lock:
                    self._worker_registry[model_path] = worker_urls
                return worker_urls
        except Exception as e:
            logger.warning(f"Failed to fetch workers from server for {model_path}: {e}")

        raise ValueError(f"Model path not found in registry: {model_path}")

    def post(self, endpoint: str, data: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send synchronous POST request.

        Args:
            endpoint: API endpoint (e.g., "/api/v1/forward_backward")
            data: JSON data to send
            timeout: Optional timeout override

        Returns:
            Response JSON

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        timeout_val = timeout if timeout is not None else self.timeout

        try:
            response = self._http_session.post(url, json=data, timeout=timeout_val)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Extract error detail if available
            error_detail = str(e)
            try:
                if e.response is not None:
                    error_json = e.response.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
            except Exception:
                pass
            logger.error(f"POST {url} failed: {error_detail}")
            raise RuntimeError(error_detail) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"POST {url} failed: {e}")
            raise RuntimeError(str(e)) from e

    def post_async(
        self, endpoint: str, data: Dict[str, Any], timeout: Optional[float] = None
    ) -> Future[Dict[str, Any]]:
        """Send asynchronous POST request.

        Args:
            endpoint: API endpoint
            data: JSON data to send
            timeout: Optional timeout override

        Returns:
            Future that will contain the response JSON
        """

        def _post():
            return self.post(endpoint, data, timeout)

        return self._executor.submit(_post)

    def get(self, endpoint: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send synchronous GET request.

        Args:
            endpoint: API endpoint
            timeout: Optional timeout override

        Returns:
            Response JSON

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        timeout_val = timeout if timeout is not None else self.timeout

        try:
            response = self._http_session.get(url, timeout=timeout_val)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET {url} failed: {e}")
            raise RuntimeError(str(e)) from e

    def shutdown(self):
        """Shutdown the client holder and cleanup resources."""
        logger.info("Shutting down ClientHolder")
        self._executor.shutdown(wait=True)
        self._http_session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
