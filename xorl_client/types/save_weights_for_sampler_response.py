"""SaveWeightsForSamplerResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SaveWeightsForSamplerResponse"]


@dataclass
class SaveWeightsForSamplerResponse:
    """Response from save_weights_for_sampler operation."""

    path: str  # Model path like "xorl://model-123/step-100"

    @classmethod
    def from_dict(cls, data: dict) -> "SaveWeightsForSamplerResponse":
        """Create from dictionary."""
        return cls(path=data["path"])
