"""ServerCapabilities type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ["ServerCapabilities"]


@dataclass
class ServerCapabilities:
    """Capabilities of the XoRL training server.

    Contains information about what the server supports, including
    supported models, loss functions, and other features.
    """

    supported_models: List[str]
    supported_loss_functions: List[str]
    max_batch_size: int
    max_sequence_length: int
    version: str
    features: Dict[str, bool]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerCapabilities":
        """Create from dictionary (for deserialization)."""
        return cls(
            supported_models=d.get("supported_models", []),
            supported_loss_functions=d.get("supported_loss_functions", []),
            max_batch_size=d.get("max_batch_size", 0),
            max_sequence_length=d.get("max_sequence_length", 0),
            version=d.get("version", "unknown"),
            features=d.get("features", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "supported_models": self.supported_models,
            "supported_loss_functions": self.supported_loss_functions,
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length,
            "version": self.version,
            "features": self.features,
        }
