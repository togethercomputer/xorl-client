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
    lora_seed: Optional[int] = None
    train_mlp: Optional[bool] = None
    train_attn: Optional[bool] = None
    train_unembed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"rank": self.rank, "dropout": self.dropout}
        if self.alpha is not None:
            result["alpha"] = self.alpha
        if self.target_modules is not None:
            result["target_modules"] = self.target_modules
        if self.lora_seed is not None:
            result["lora_seed"] = self.lora_seed
        if self.train_mlp is not None:
            result["train_mlp"] = self.train_mlp
        if self.train_attn is not None:
            result["train_attn"] = self.train_attn
        if self.train_unembed is not None:
            result["train_unembed"] = self.train_unembed
        return result
