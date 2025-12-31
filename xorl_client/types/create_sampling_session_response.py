"""CreateSamplingSessionResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["CreateSamplingSessionResponse"]


@dataclass
class CreateSamplingSessionResponse:
    """Response from create_sampling_session operation.

    Returned when a sampling session is created, indicating the LoRA adapter
    has been loaded on the inference workers.
    """

    success: bool
    """Whether the session was created successfully"""

    model_path: str
    """The model path that was loaded"""

    lora_name: str
    """The name of the LoRA adapter that was loaded"""

    message: Optional[str] = None
    """Optional status message"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CreateSamplingSessionResponse":
        """Create from dictionary (for deserialization)."""
        return cls(
            success=d.get("success", False),
            model_path=d.get("model_path", ""),
            lora_name=d.get("lora_name", ""),
            message=d.get("message"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "model_path": self.model_path,
            "lora_name": self.lora_name,
            "message": self.message,
        }
