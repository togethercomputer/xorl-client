"""
UnloadModelResponse - Response from unload_model operation.
"""

from __future__ import annotations

from typing import Literal, Optional

from .strict_base import StrictBase
from .model_id import ModelID

__all__ = ["UnloadModelResponse"]


class UnloadModelResponse(StrictBase):
    """Response from unloading a model.

    Attributes:
        model_id: The ID of the unloaded model.
        type: Optional type identifier.
    """

    model_id: ModelID
    type: Optional[Literal["unload_model"]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"model_id": self.model_id}
        if self.type is not None:
            result["type"] = self.type
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UnloadModelResponse":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            type=data.get("type"),
        )
