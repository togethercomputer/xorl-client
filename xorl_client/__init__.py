"""
XoRL Python Client - Lightweight client for remote training.

This package provides a minimal client SDK for communicating with XoRL
training servers. It does not include server-side dependencies like PyTorch.
"""

__version__ = "0.1.0"

# Import types module
from . import types

# Import clients
from .client.service_client import ServiceClient
from .client.api_future import APIFuture
from .client.training_client import TrainingClient
from .client.sampling_client import SamplingClient
from .client.rest_client import RestClient

# Import exceptions
from .exceptions import (
    XorlClientError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError,
)

# DedicatedEndpointSamplingClient requires openai package
# Import explicitly: from xorl_client.dedicated_endpoint import DedicatedEndpointSamplingClient
# This will raise ImportError with clear message if openai is not installed

# Import commonly used types for convenience
from .types import (
    # Base classes
    StrictBase,
    # Model input types (matching tinker)
    EncodedTextChunk,
    ImageChunk,
    ImageAssetPointerChunk,
    ModelInputChunk,
    ModelInput,
    # Training types
    Datum,
    TensorData,
    AdamParams,
    SamplingParams,
    SampledSequence,
    SampleResponse,
    ForwardBackwardOutput,
    OptimStepResponse,
    SaveWeightsResponse,
    SaveWeightsForSamplerResponse,
    LoadWeightsResponse,
    LoraConfig,
    UpdateDedicatedEndpointResponse,
    DedicatedEndpointConfig,
)

__all__ = [
    # Core clients
    "ServiceClient",
    "TrainingClient",
    "SamplingClient",
    "RestClient",
    "APIFuture",
    # Types module
    "types",
    # Exceptions
    "XorlClientError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "BadRequestError",
    "AuthenticationError",
    "NotFoundError",
    "InternalServerError",
    # Base classes
    "StrictBase",
    # Model input types (matching tinker)
    "EncodedTextChunk",
    "ImageChunk",
    "ImageAssetPointerChunk",
    "ModelInputChunk",
    "ModelInput",
    # Training types
    "Datum",
    "TensorData",
    "AdamParams",
    "SamplingParams",
    "SampledSequence",
    "SampleResponse",
    "ForwardBackwardOutput",
    "OptimStepResponse",
    "SaveWeightsResponse",
    "SaveWeightsForSamplerResponse",
    "LoadWeightsResponse",
    "LoraConfig",
    "UpdateDedicatedEndpointResponse",
    "DedicatedEndpointConfig",
    # Version
    "__version__",
]
