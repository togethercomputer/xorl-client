"""ModelInput type definition."""

from __future__ import annotations

from typing import List

from .strict_base import StrictBase
from .model_input_chunk import ModelInputChunk
from .encoded_text_chunk import EncodedTextChunk

__all__ = ["ModelInput"]


class ModelInput(StrictBase):
    """Model input data as a sequence of chunks.

    Matches tinker's ModelInput API. Supports multimodal inputs with
    text tokens and images.

    Example:
        >>> # Create from token IDs
        >>> model_input = ModelInput.from_ints([1, 2, 3, 4, 5])
        >>> model_input.to_ints()
        [1, 2, 3, 4, 5]

        >>> # Create empty and append
        >>> model_input = ModelInput.empty()
        >>> model_input = model_input.append(EncodedTextChunk(tokens=[1, 2, 3]))
    """

    chunks: List[ModelInputChunk]
    """Sequence of input chunks"""

    @classmethod
    def from_ints(cls, tokens: List[int]) -> "ModelInput":
        """Create a ModelInput from a list of ints (tokens).

        Args:
            tokens: List of token IDs

        Returns:
            ModelInput instance with a single EncodedTextChunk
        """
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    def to_ints(self) -> List[int]:
        """Convert the ModelInput to a list of ints (tokens).

        Returns:
            Flattened list of all token IDs from all EncodedTextChunks

        Raises:
            ValueError: If there are any non-token chunks (e.g., images)
        """
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(
                f"to_ints only supported for ModelInput with EncodedTextChunks, "
                f"got {[type(chunk).__name__ for chunk in self.chunks]}"
            )
        return [token for chunk in self.chunks for token in chunk.tokens]

    @property
    def length(self) -> int:
        """Return the total context length used by this ModelInput."""
        return sum(chunk.length for chunk in self.chunks)

    @classmethod
    def empty(cls) -> "ModelInput":
        """Create an empty ModelInput."""
        return cls(chunks=[])

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        """Add a new chunk, return a new ModelInput."""
        return ModelInput(chunks=list(self.chunks) + [chunk])

    def append_int(self, token: int) -> "ModelInput":
        """Add a new token, return a new ModelInput."""
        return self.append(EncodedTextChunk(tokens=[token]))
