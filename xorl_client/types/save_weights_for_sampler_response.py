"""SaveWeightsForSamplerResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["SaveWeightsForSamplerResponse"]


@dataclass
class SaveWeightsForSamplerResponse:
    """Response from save_weights_for_sampler operation."""

    path: str  # Full filesystem path like "outputs/.../sampler_weights/step-100"
    model_path: str | None = None  # Identifier for sampling session like "sampler_weights/step-100"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SaveWeightsForSamplerResponse":
        """Create from dictionary."""
        return cls(path=data["path"], model_path=data.get("model_path"))
