"""Serverless weights types."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["UpdateServerlessWeightsResponse"]


@dataclass
class UpdateServerlessWeightsResponse:
    """Response from update_serverless_weights operation."""

    provider_model_id: str  # model name for inference
    hf_repo_url: str  # HuggingFace repository URL
    checkpoint_path: str  # Local checkpoint path
    status: str  # Upload status ('Complete', 'submitted', etc.)
