"""
Models resource - low-level API for model management.

Maps to:
- POST /api/v1/create_model
- POST /api/v1/unload_model
- POST /api/v1/get_info (if needed)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AsyncResource

__all__ = ["ModelsResource"]


class ModelsResource(AsyncResource):
    """Low-level resource for model management operations.

    Provides direct HTTP access to model endpoints. Returns UntypedAPIFuture
    for two-phase pattern endpoints.

    Usage:
        models = ModelsResource(holder)
        future_dict = await models.create(base_model, lora_config)
        # Returns raw dict, caller must parse UntypedAPIFuture
    """

    async def create(
        self,
        base_model: str,
        lora_config: Dict[str, Any],
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new model with LoRA configuration.

        Args:
            base_model: Base model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            lora_config: LoRA configuration dictionary
            model_id: Optional model ID (generated if not provided)
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "base_model": base_model,
            "lora_config": lora_config,
        }
        if model_id:
            request_data["model_id"] = model_id

        return await self._post("/api/v1/create_model", request_data, timeout)

    async def unload(
        self,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Unload a model and end the training session.

        This releases all resources associated with the model:
        - Server-side state (registered model IDs, session tracking)
        - Worker resources (training adapter, GPU memory)

        Args:
            model_id: Model identifier to unload
            timeout: Optional timeout override

        Returns:
            UntypedAPIFuture dict with request_id and model_id
        """
        request_data = {
            "model_id": model_id,
            "type": "unload_model",
        }

        return await self._post("/api/v1/unload_model", request_data, timeout)

    async def get_info(
        self,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model_id: Model identifier
            timeout: Optional timeout override

        Returns:
            GetInfoResponse dict with model metadata
        """
        request_data = {
            "model_id": model_id,
        }

        return await self._post("/api/v1/get_info", request_data, timeout)
