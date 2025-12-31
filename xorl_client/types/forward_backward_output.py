"""ForwardBackwardOutput type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ["ForwardBackwardOutput"]


@dataclass
class ForwardBackwardOutput:
    """Output from forward-backward pass."""

    loss_fn_outputs: List[Dict[str, Any]]
    metrics: Dict[str, float]
