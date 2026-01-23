"""
Training resource - low-level API for training operations.

Maps to:
- POST /api/v1/forward
- POST /api/v1/forward_backward
- POST /api/v1/optim_step
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from xorl_client import types
from .base import AsyncResource

__all__ = ["TrainingResource"]


class TrainingResource(AsyncResource):
    """Low-level resource for training operations.

    Provides direct HTTP access to training endpoints. Returns UntypedAPIFuture
    for two-phase pattern endpoints.

    Usage:
        training = TrainingResource(holder)
        future_dict = await training.forward(model_id, seq_id, data, loss_fn)
        # Returns raw dict, caller must parse UntypedAPIFuture
    """

    async def forward(
        self,
        model_id: str,
        seq_id: int,
        data: List[Dict[str, Any]],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute forward pass (no gradients).

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            data: List of datum dictionaries
            loss_fn: Loss function name
            loss_fn_config: Optional loss function configuration
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "forward_input": {
                "data": data,
                "loss_fn": loss_fn,
            }
        }
        if loss_fn_config:
            request_data["forward_input"]["loss_fn_config"] = loss_fn_config

        return await self._post("/api/v1/forward", request_data, timeout)

    async def forward_backward(
        self,
        model_id: str,
        seq_id: int,
        data: List[Dict[str, Any]],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute forward and backward pass (computes gradients).

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            data: List of datum dictionaries
            loss_fn: Loss function name
            loss_fn_config: Optional loss function configuration
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "forward_backward_input": {
                "data": data,
                "loss_fn": loss_fn,
            }
        }
        if loss_fn_config:
            request_data["forward_backward_input"]["loss_fn_config"] = loss_fn_config

        return await self._post("/api/v1/forward_backward", request_data, timeout)

    async def optim_step(
        self,
        model_id: str,
        seq_id: int,
        adam_params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute optimizer step to update model parameters.

        Args:
            model_id: Model identifier
            seq_id: Sequence ID for ordering
            adam_params: Adam optimizer parameters dict
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "seq_id": seq_id,
            "adam_params": adam_params,
        }

        return await self._post("/api/v1/optim_step", request_data, timeout)
