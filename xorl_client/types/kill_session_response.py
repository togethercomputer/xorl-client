"""
KillSessionResponse - Response from kill_session operation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .strict_base import StrictBase

__all__ = ["KillSessionResponse"]


class KillSessionResponse(StrictBase):
    """Response from killing a full-weights training session.

    In full-weights training mode (enable_lora=False), the server operates in
    single-tenant mode. This response indicates the result of killing the
    active session to allow starting a new one.

    For LoRA mode, this is a no-op since multi-tenancy is supported.

    Attributes:
        success: Whether the session was killed successfully.
        message: Description of the result.
        checkpoint_path: Path to the saved checkpoint (if save_checkpoint was True
                        and the save succeeded). None otherwise.
    """

    success: bool
    message: str
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "success": self.success,
            "message": self.message,
        }
        if self.checkpoint_path is not None:
            result["checkpoint_path"] = self.checkpoint_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KillSessionResponse":
        """Create from dictionary."""
        return cls(
            success=data["success"],
            message=data["message"],
            checkpoint_path=data.get("checkpoint_path"),
        )
