"""
Weights resource - low-level API for weight management.

Maps to:
- POST /api/v1/save_weights
- POST /api/v1/load_weights
- POST /api/v1/save_weights_for_sampler
- GET /api/v1/list_checkpoints
- DELETE /api/v1/delete_checkpoint
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AsyncResource

__all__ = ["WeightsResource"]


class WeightsResource(AsyncResource):
    """Low-level resource for weight management operations.

    Provides direct HTTP access to weight endpoints. Returns UntypedAPIFuture
    for two-phase pattern endpoints.

    Usage:
        weights = WeightsResource(holder)
        future_dict = await weights.save(model_id, seq_id, path)
        # Returns raw dict, caller must parse UntypedAPIFuture
    """

    async def save(
        self,
        model_id: str,
        seq_id: int,
        path: str,
        save_optimizer: bool = True,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Save model weights to storage.

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            path: Path/name for the checkpoint
            save_optimizer: Whether to save optimizer state
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "path": path,
            "save_optimizer": save_optimizer,
        }

        return await self._post("/api/v1/save_weights", request_data, timeout)

    async def load(
        self,
        model_id: str,
        seq_id: int,
        path: str,
        load_optimizer: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load model weights from storage.

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            path: Path to the checkpoint
            load_optimizer: Whether to load optimizer state
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "path": path,
            "load_optimizer": load_optimizer,
        }

        return await self._post("/api/v1/load_weights", request_data, timeout)

    async def save_for_sampler(
        self,
        model_id: str,
        seq_id: int,
        name: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Save weights for sampling/inference.

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            name: Name for the sampler weights
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "name": name,
        }

        return await self._post("/api/v1/save_weights_for_sampler", request_data, timeout)

    async def list_checkpoints(
        self,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """List available checkpoints for a model.

        Args:
            model_id: Model identifier
            timeout: Optional timeout override

        Returns:
            CheckpointsListResponse dict with checkpoints list
        """
        request_data = {
            "model_id": model_id,
        }

        return await self._post("/api/v1/list_checkpoints", request_data, timeout)

    async def delete_checkpoint(
        self,
        model_id: str,
        checkpoint_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Delete a checkpoint.

        Args:
            model_id: Model identifier
            checkpoint_id: Checkpoint identifier to delete
            timeout: Optional timeout override

        Returns:
            DeleteCheckpointResponse dict
        """
        request_data = {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
        }

        return await self._post("/api/v1/delete_checkpoint", request_data, timeout)

    async def get_weights_info(
        self,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get information about model weights.

        Args:
            model_id: Model identifier
            timeout: Optional timeout override

        Returns:
            WeightsInfoResponse dict with weights metadata
        """
        request_data = {
            "model_id": model_id,
        }

        return await self._post("/api/v1/weights_info", request_data, timeout)
