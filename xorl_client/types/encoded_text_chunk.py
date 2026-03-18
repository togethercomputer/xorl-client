"""EncodedTextChunk type definition."""

from __future__ import annotations

from typing import Sequence

from typing_extensions import Literal

from .strict_base import StrictBase

__all__ = ["EncodedTextChunk"]


class EncodedTextChunk(StrictBase):
    """A chunk of encoded text tokens.

    Matches tinker's EncodedTextChunk.
    """

    tokens: Sequence[int]
    """Array of token IDs"""

    type: Literal["encoded_text"] = "encoded_text"

    @property
    def length(self) -> int:
        return len(self.tokens)
