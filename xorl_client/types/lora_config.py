"""LoraConfig type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["LoraConfig"]


@dataclass
class LoraConfig:
    """LoRA configuration for training."""

    rank: int = 32
    alpha: Optional[int] = None
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"rank": self.rank, "dropout": self.dropout}
        if self.alpha is not None:
            result["alpha"] = self.alpha
        if self.target_modules is not None:
            result["target_modules"] = self.target_modules
        return result
