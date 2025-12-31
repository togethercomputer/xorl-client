"""ModelInputChunk type alias."""

from __future__ import annotations

from typing import Union

import pydantic
from typing_extensions import Annotated, TypeAlias

from .encoded_text_chunk import EncodedTextChunk
from .image_asset_pointer_chunk import ImageAssetPointerChunk
from .image_chunk import ImageChunk

__all__ = ["ModelInputChunk"]

# Type alias for model input chunks (matching tinker's ModelInputChunk)
ModelInputChunk: TypeAlias = Annotated[
    Union[EncodedTextChunk, ImageAssetPointerChunk, ImageChunk],
    pydantic.Field(discriminator="type"),
]
