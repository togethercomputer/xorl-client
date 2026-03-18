"""
Low-level API resources for xorl_client.

This module provides thin HTTP wrappers that directly map to the xorl server endpoints.
These resources handle serialization/deserialization but don't add business logic.

The high-level clients (TrainingClient, ServiceClient) use these resources internally
and add additional features like:
- Request chunking
- Turn-based ordering
- Two-phase polling
- Retry handling
- Result combining

Usage:
    # Resources are typically accessed via the holder's resources property
    # or used internally by high-level clients

    from xorl_client.resources import TrainingResource, WeightsResource

    # Low-level access (advanced usage)
    training = TrainingResource(holder)
    result = await training.forward(request_data)
"""

from .base import AsyncResource
from .training import TrainingResource
from .models import ModelsResource
from .weights import WeightsResource
from .futures import FuturesResource

__all__ = [
    "AsyncResource",
    "TrainingResource",
    "ModelsResource",
    "WeightsResource",
    "FuturesResource",
]
