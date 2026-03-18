"""Pydantic base classes for xorl_client types."""

from __future__ import annotations

import pydantic
from pydantic import ConfigDict

__all__ = ["StrictBase"]


class StrictBase(pydantic.BaseModel):
    """Don't allow extra fields, so user errors are caught earlier.

    Use this for request types. Matches tinker's StrictBase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)
