"""Dedicated endpoint types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["UpdateDedicatedEndpointResponse", "DedicatedEndpointConfig"]


@dataclass
class UpdateDedicatedEndpointResponse:
    """Response from update_dedicated_endpoint operation."""

    success: bool
    checkpoint_path: str
    hf_path: str
    provider_model_id: str  # Use this for sampling from the dedicated endpoint
    upload_time: float  # Time to upload to HuggingFace
    provider_upload_time: float  # Time for API upload
    probe_time: float  # Time to probe endpoint
    error: Optional[str] = None


@dataclass
class DedicatedEndpointConfig:
    """Configuration for Dedicated Endpoint (passed to training server)."""

    endpoint_id: str
    provider_api_key: str
    hf_token: str
    hf_org: str = "xorl-ai"
