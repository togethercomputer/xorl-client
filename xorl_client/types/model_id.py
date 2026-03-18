"""ModelID type alias."""

from __future__ import annotations

__all__ = ["ModelID", "LossFnType"]

# Type aliases for backward compatibility
ModelID = str
LossFnType = str  # "cross_entropy", "importance_sampling", "ppo", etc.
