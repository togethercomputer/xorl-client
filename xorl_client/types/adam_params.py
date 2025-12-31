"""AdamParams type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["AdamParams"]


@dataclass
class AdamParams:
    """Adam optimizer parameters.

    Defaults match Tinker's AdamParams for compatibility.
    """

    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""

    weight_decay: float = 0.0
    """Weight decay for the optimizer. Uses decoupled weight decay."""

    grad_clip_norm: float = 0.0
    """Gradient clip norm for the optimizer. 0.0 means no clipping."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }
