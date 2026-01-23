"""LoadWeightsResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LoadWeightsResponse"]


@dataclass
class LoadWeightsResponse:
    """Response from load_weights operation."""

    path: str  # XoRL URI that was loaded

    @classmethod
    def from_dict(cls, data: dict) -> "LoadWeightsResponse":
        """Create from dictionary."""
        return cls(path=data["path"])
