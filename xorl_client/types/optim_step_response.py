"""OptimStepResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["OptimStepResponse"]


@dataclass
class OptimStepResponse:
    """Response from optimizer step."""

    metrics: Dict[str, float]

    @classmethod
    def from_dict(cls, data: dict) -> "OptimStepResponse":
        """Create from dictionary."""
        return cls(metrics=data.get("metrics", {}))
