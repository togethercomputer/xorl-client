"""Cursor type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["Cursor"]


@dataclass
class Cursor:
    """Pagination cursor information.

    Matches tinker's Cursor for API compatibility.
    """

    offset: int
    """The offset used for pagination"""

    limit: int
    """The maximum number of items requested"""

    total_count: int
    """The total number of items available"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cursor":
        """Create from dictionary (for deserialization)."""
        return cls(
            offset=d["offset"],
            limit=d["limit"],
            total_count=d["total_count"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "offset": self.offset,
            "limit": self.limit,
            "total_count": self.total_count,
        }
