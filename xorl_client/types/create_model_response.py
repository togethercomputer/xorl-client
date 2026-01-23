"""
CreateModelResponse - Response from create_model operation.
"""

from __future__ import annotations

from typing import Literal, Optional

from .strict_base import StrictBase
from .model_id import ModelID

__all__ = ["CreateModelResponse"]


class CreateModelResponse(StrictBase):
    """Response from creating a model.

    Attributes:
        model_id: The ID of the created model.
        type: Always "create_model" to identify this response type.
    """

    model_id: ModelID
    type: Literal["create_model"] = "create_model"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CreateModelResponse":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            type=data.get("type", "create_model"),
        )
