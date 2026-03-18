"""TensorDtype type definition."""

from __future__ import annotations

__all__ = ["TensorDtype"]

# TensorDtype matches tinker's definition
TensorDtype = str  # Literal["int64", "float32"] in tinker, but we use str for flexibility
