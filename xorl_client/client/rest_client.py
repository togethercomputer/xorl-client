"""RestClient for XoRL API REST operations.

This module provides a RestClient class that mirrors tinker's RestClient API
for checkpoint management and other REST operations.

The RestClient provides access to various REST endpoints for querying
checkpoint information and managing checkpoints. You typically get one
by calling `service_client.create_rest_client()`.

Example:
    >>> service_client = xorl_client.ServiceClient(base_url="http://localhost:5000")
    >>> rest_client = service_client.create_rest_client()
    >>> checkpoints = rest_client.list_checkpoints().result()
    >>> for checkpoint in checkpoints.checkpoints:
    ...     print(f"{checkpoint.checkpoint_id}: {checkpoint.checkpoint_type}")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from xorl_client import types
from xorl_client.client.api_future import APIFuture, wrap_future
from xorl_client.client.client_holder import ClientHolder

logger = logging.getLogger(__name__)


class RestClient:
    """Client for REST API operations like listing and deleting checkpoints.

    The RestClient provides access to various REST endpoints for querying
    checkpoint information and managing checkpoints. You typically get one
    by calling `service_client.create_rest_client()`.

    Key methods:
    - list_checkpoints() - list available model checkpoints (both training and sampler)
    - delete_checkpoint() - delete an existing checkpoint (accepts checkpoint_id or xorl:// path)
    - get_weights_info() - get checkpoint metadata
    - list_training_runs() - list training runs with pagination

    Note: Unlike tinker's RestClient, xorl_client's RestClient operates on a single
    model (model_id="default") since xorl_client is designed for single-model training.

    Args:
        holder: ClientHolder managing HTTP connections and async operations
        model_id: Model identifier (default: "default")

    Example:
        >>> rest_client = service_client.create_rest_client()
        >>> checkpoints = rest_client.list_checkpoints().result()
        >>> print(f"Found {len(checkpoints.checkpoints)} checkpoints")
        >>> for checkpoint in checkpoints.checkpoints:
        ...     print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
    """

    def __init__(self, holder: ClientHolder, model_id: str = "default"):
        """Initialize RestClient.

        Args:
            holder: ClientHolder for connection management
            model_id: Model identifier (default: "default")
        """
        self.holder = holder
        self.model_id = model_id

    def list_checkpoints(
        self,
        model_id: Optional[str] = None,
    ) -> APIFuture[types.CheckpointsListResponse]:
        """List available checkpoints (both training and sampler).

        Args:
            model_id: The model ID to list checkpoints for (default: self.model_id)

        Returns:
            APIFuture[CheckpointsListResponse] with available checkpoints

        Example:
            >>> future = rest_client.list_checkpoints()
            >>> response = future.result()
            >>> for checkpoint in response.checkpoints:
            ...     if checkpoint.checkpoint_type == "training":
            ...         print(f"Training checkpoint: {checkpoint.checkpoint_id}")
            ...     elif checkpoint.checkpoint_type == "sampler":
            ...         print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
        """
        if model_id is None:
            model_id = self.model_id

        request_data = {
            "model_id": model_id,
        }

        future = self.holder.post_async("/api/v1/list_checkpoints", request_data)

        def parse_response(result: Dict[str, Any]) -> types.CheckpointsListResponse:
            checkpoints = [
                types.Checkpoint.from_dict(c)
                for c in result.get("checkpoints", [])
            ]
            logger.info(f"Listed {len(checkpoints)} checkpoints")
            return types.CheckpointsListResponse(checkpoints=checkpoints)

        from concurrent.futures import Future
        parsed_future: Future[types.CheckpointsListResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def list_checkpoints_async(
        self,
        model_id: Optional[str] = None,
    ) -> types.CheckpointsListResponse:
        """Async version of list_checkpoints.

        Args:
            model_id: The model ID to list checkpoints for (default: self.model_id)

        Returns:
            CheckpointsListResponse with available checkpoints
        """
        future = self.list_checkpoints(model_id)
        return await future.result_async()

    def delete_checkpoint(
        self,
        checkpoint_id_or_path: str,
        model_id: Optional[str] = None,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Delete a checkpoint.

        Args:
            checkpoint_id_or_path: Checkpoint ID (e.g., 'weights/000060') or
                xorl:// path (e.g., 'xorl://default/weights/000060')
            model_id: The model ID (default: self.model_id). Ignored if xorl:// path is provided.

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status

        Example:
            >>> # Using checkpoint_id
            >>> future = rest_client.delete_checkpoint("weights/000060")
            >>> response = future.result()
            >>> print(f"Deleted: {response.deleted_path}")
            >>>
            >>> # Using xorl:// path
            >>> future = rest_client.delete_checkpoint("xorl://default/weights/000060")
            >>> response = future.result()
        """
        # Check if input is a xorl:// path
        if checkpoint_id_or_path.startswith("xorl://"):
            parsed = types.ParsedCheckpointXoRLPath.from_xorl_path(checkpoint_id_or_path)
            checkpoint_id = parsed.checkpoint_id
            model_id = parsed.model_id
        else:
            checkpoint_id = checkpoint_id_or_path
            if model_id is None:
                model_id = self.model_id

        request_data = {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
        }

        future = self.holder.post_async("/api/v1/delete_checkpoint", request_data)

        def parse_response(result: Dict[str, Any]) -> types.DeleteCheckpointResponse:
            success = result.get("success", False)
            deleted_path = result.get("deleted_path")
            error = result.get("error")
            if success:
                logger.info(f"Deleted checkpoint: {deleted_path}")
            else:
                logger.error(f"Failed to delete checkpoint: {error}")
            return types.DeleteCheckpointResponse(
                success=success,
                deleted_path=deleted_path,
                error=error,
            )

        from concurrent.futures import Future
        parsed_future: Future[types.DeleteCheckpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def delete_checkpoint_async(
        self,
        checkpoint_id_or_path: str,
        model_id: Optional[str] = None,
    ) -> types.DeleteCheckpointResponse:
        """Async version of delete_checkpoint.

        Args:
            checkpoint_id_or_path: Checkpoint ID (e.g., 'weights/000060') or
                xorl:// path (e.g., 'xorl://default/weights/000060')
            model_id: The model ID (default: self.model_id). Ignored if xorl:// path is provided.

        Returns:
            DeleteCheckpointResponse with success status
        """
        future = self.delete_checkpoint(checkpoint_id_or_path, model_id)
        return await future.result_async()

    def get_weights_info(
        self,
        xorl_path: str,
    ) -> APIFuture[types.WeightsInfoResponse]:
        """Get checkpoint information from a xorl:// path.

        Args:
            xorl_path: The xorl:// path to the checkpoint

        Returns:
            APIFuture[WeightsInfoResponse] containing checkpoint metadata

        Example:
            >>> future = rest_client.get_weights_info("xorl://default/weights/000060")
            >>> response = future.result()
            >>> print(f"Base Model: {response.base_model}, LoRA Rank: {response.lora_rank}")
        """
        request_data = {
            "xorl_path": xorl_path,
        }

        future = self.holder.post_async("/api/v1/weights_info", request_data)

        def parse_response(result: Dict[str, Any]) -> types.WeightsInfoResponse:
            return types.WeightsInfoResponse(
                base_model=result.get("base_model", ""),
                is_lora=result.get("is_lora", True),
                lora_rank=result.get("lora_rank"),
            )

        from concurrent.futures import Future
        parsed_future: Future[types.WeightsInfoResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def get_weights_info_async(
        self,
        xorl_path: str,
    ) -> types.WeightsInfoResponse:
        """Async version of get_weights_info.

        Args:
            xorl_path: The xorl:// path to the checkpoint

        Returns:
            WeightsInfoResponse containing checkpoint metadata
        """
        future = self.get_weights_info(xorl_path)
        return await future.result_async()

    # Alias for tinker API compatibility
    def get_weights_info_by_xorl_path(
        self,
        xorl_path: str,
    ) -> APIFuture[types.WeightsInfoResponse]:
        """Alias for get_weights_info for tinker API compatibility.

        Args:
            xorl_path: The xorl:// path to the checkpoint

        Returns:
            APIFuture[WeightsInfoResponse] containing checkpoint metadata
        """
        return self.get_weights_info(xorl_path)

    async def get_weights_info_by_xorl_path_async(
        self,
        xorl_path: str,
    ) -> types.WeightsInfoResponse:
        """Async version of get_weights_info_by_xorl_path.

        Args:
            xorl_path: The xorl:// path to the checkpoint

        Returns:
            WeightsInfoResponse containing checkpoint metadata
        """
        return await self.get_weights_info_async(xorl_path)

    def list_training_runs(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> APIFuture[types.TrainingRunsResponse]:
        """List training runs with pagination support.

        Note: In xorl_client, there is typically only a single training run with
        model_id="default". This method provides API compatibility with tinker.

        Args:
            limit: Maximum number of training runs to return (default 20)
            offset: Offset for pagination (default 0)

        Returns:
            APIFuture[TrainingRunsResponse] with training runs and cursor info

        Example:
            >>> future = rest_client.list_training_runs(limit=50)
            >>> response = future.result()
            >>> print(f"Found {len(response.training_runs)} training runs")
            >>> print(f"Total: {response.cursor.total_count}")
            >>> for run in response.training_runs:
            ...     print(f"  {run.training_run_id}: {run.base_model}")
        """
        params = {
            "limit": limit,
            "offset": offset,
        }

        future = self.holder.get_async("/api/v1/training_runs", params=params)

        def parse_response(result: Dict[str, Any]) -> types.TrainingRunsResponse:
            return types.TrainingRunsResponse.from_dict(result)

        from concurrent.futures import Future
        parsed_future: Future[types.TrainingRunsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def list_training_runs_async(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> types.TrainingRunsResponse:
        """Async version of list_training_runs.

        Args:
            limit: Maximum number of training runs to return (default 20)
            offset: Offset for pagination (default 0)

        Returns:
            TrainingRunsResponse with training runs and cursor info
        """
        future = self.list_training_runs(limit, offset)
        return await future.result_async()
