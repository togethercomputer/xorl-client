"""LoadWeightsResponse type definition."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LoadWeightsResponse"]


@dataclass
class LoadWeightsResponse:
    """Response from load_weights operation."""

    path: str  # XoRL URI that was loaded
