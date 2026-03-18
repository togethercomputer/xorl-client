"""WeightsInfoResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["WeightsInfoResponse"]


@dataclass
class WeightsInfoResponse:
    """Response from get_weights_info operation.

    Contains metadata about a checkpoint needed to resume training.
    """

    base_model: str
    """The base model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')"""

    is_lora: bool = True
    """Whether this is a LoRA checkpoint"""

    lora_rank: Optional[int] = None
    """The LoRA rank if this is a LoRA checkpoint"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeightsInfoResponse":
        """Create from dictionary (for deserialization)."""
        return cls(
            base_model=d.get("base_model", ""),
            is_lora=d.get("is_lora", True),
            lora_rank=d.get("lora_rank"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "base_model": self.base_model,
            "is_lora": self.is_lora,
            "lora_rank": self.lora_rank,
        }
