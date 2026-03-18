"""SaveWeightsResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SaveWeightsResponse"]


@dataclass
class SaveWeightsResponse:
    """Response from save_weights operation.

    Returns a xorl:// URI pointing to the saved checkpoint.
    """

    path: str  # XoRL URI (e.g., xorl://default/weights/checkpoint-001)

    @classmethod
    def from_dict(cls, data: dict) -> "SaveWeightsResponse":
        """Create from dictionary."""
        return cls(path=data["path"])
