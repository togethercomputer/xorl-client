"""Checkpoint types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "Checkpoint",
    "CheckpointsListResponse",
    "DeleteCheckpointResponse",
    "ParsedCheckpointXoRLPath",
    "CheckpointType",
]

# Type aliases for backward compatibility
CheckpointType = str  # "training" or "sampler"


@dataclass
class Checkpoint:
    """A checkpoint saved during training.

    Matches tinker's Checkpoint API for compatibility.
    """

    checkpoint_id: str
    """The checkpoint ID (e.g., 'weights/000' or 'sampler_weights/step-100')"""

    checkpoint_type: CheckpointType
    """The type of checkpoint ('training' or 'sampler')"""

    time: str
    """ISO format timestamp when the checkpoint was created"""

    path: str
    """The xorl:// path to the checkpoint"""

    size_bytes: Optional[int] = None
    """The size of the checkpoint in bytes"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary (for deserialization)."""
        return cls(
            checkpoint_id=d["checkpoint_id"],
            checkpoint_type=d["checkpoint_type"],
            time=d["time"],
            path=d["path"],
            size_bytes=d.get("size_bytes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type,
            "time": self.time,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@dataclass
class CheckpointsListResponse:
    """Response from list_checkpoints operation.

    Matches tinker's CheckpointsListResponse API for compatibility.
    """

    checkpoints: List[Checkpoint]
    """List of available checkpoints"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointsListResponse":
        """Create from dictionary (for deserialization)."""
        checkpoints = [Checkpoint.from_dict(c) for c in d.get("checkpoints", [])]
        return cls(checkpoints=checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }


@dataclass
class DeleteCheckpointResponse:
    """Response from delete_checkpoint operation."""

    success: bool
    """Whether the deletion was successful"""

    deleted_path: Optional[str] = None
    """The xorl:// path that was deleted"""

    error: Optional[str] = None
    """Error message if deletion failed"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeleteCheckpointResponse":
        """Create from dictionary (for deserialization)."""
        return cls(
            success=d["success"],
            deleted_path=d.get("deleted_path"),
            error=d.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "deleted_path": self.deleted_path,
            "error": self.error,
        }


@dataclass
class ParsedCheckpointXoRLPath:
    """Parsed xorl:// checkpoint path.

    Matches tinker's ParsedCheckpointTinkerPath for compatibility.
    """

    xorl_path: str
    """The original xorl:// path"""

    model_id: str
    """The model ID extracted from the path"""

    checkpoint_type: CheckpointType
    """The type of checkpoint ('training' or 'sampler')"""

    checkpoint_id: str
    """The checkpoint ID (e.g., 'weights/000')"""

    @classmethod
    def from_xorl_path(cls, xorl_path: str) -> "ParsedCheckpointXoRLPath":
        """Parse a xorl:// path into its components.

        Args:
            xorl_path: The xorl:// path (e.g., 'xorl://default/weights/000')

        Returns:
            ParsedCheckpointXoRLPath with extracted components

        Raises:
            ValueError: If the path is invalid
        """
        if not xorl_path.startswith("xorl://"):
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        # Remove "xorl://" prefix
        parts = xorl_path[7:].split("/")

        if len(parts) < 3:
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        model_id = parts[0]
        checkpoint_type_dir = parts[1]

        if checkpoint_type_dir not in ["weights", "sampler_weights"]:
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        checkpoint_type = "training" if checkpoint_type_dir == "weights" else "sampler"
        checkpoint_id = "/".join([checkpoint_type_dir] + parts[2:])

        return cls(
            xorl_path=xorl_path,
            model_id=model_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )
