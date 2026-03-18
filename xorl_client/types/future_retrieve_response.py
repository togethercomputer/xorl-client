"""
FutureRetrieveResponse - Union type for all possible retrieve_future responses.

The /api/v1/retrieve_future endpoint can return different response types
depending on the state of the request:
- TryAgainResponse: Request still processing
- RequestFailedResponse: Request failed
- Various result types: Request completed successfully
"""

from __future__ import annotations

from typing import Union

from .try_again_response import TryAgainResponse
from .request_failed_response import RequestFailedResponse
from .forward_backward_output import ForwardBackwardOutput
from .optim_step_response import OptimStepResponse
from .save_weights_response import SaveWeightsResponse
from .load_weights_response import LoadWeightsResponse
from .save_weights_for_sampler_response import SaveWeightsForSamplerResponse
from .create_model_response import CreateModelResponse
from .unload_model_response import UnloadModelResponse

# Union of all possible responses from /api/v1/retrieve_future
#
# When polling for results, the response will be one of:
# - TryAgainResponse: Request still processing, continue polling
# - RequestFailedResponse: Request failed with error
# - ForwardBackwardOutput: forward_backward completed
# - OptimStepResponse: optim_step completed
# - SaveWeightsResponse: save_weights completed
# - LoadWeightsResponse: load_weights completed
# - SaveWeightsForSamplerResponse: save_weights_for_sampler completed
# - CreateModelResponse: create_model completed
# - UnloadModelResponse: unload_model completed
FutureRetrieveResponse = Union[
    TryAgainResponse,
    ForwardBackwardOutput,
    OptimStepResponse,
    SaveWeightsResponse,
    LoadWeightsResponse,
    SaveWeightsForSamplerResponse,
    CreateModelResponse,
    UnloadModelResponse,
    RequestFailedResponse,
]


def parse_future_retrieve_response(data: dict) -> FutureRetrieveResponse:
    """Parse a dictionary response into the appropriate type.

    Args:
        data: Dictionary response from /api/v1/retrieve_future

    Returns:
        One of the FutureRetrieveResponse union types

    Example:
        >>> response = await client.post("/api/v1/retrieve_future", request)
        >>> result = parse_future_retrieve_response(response)
        >>> if isinstance(result, TryAgainResponse):
        ...     # Continue polling
        ...     pass
        >>> elif isinstance(result, RequestFailedResponse):
        ...     raise RuntimeError(result.error)
        >>> elif isinstance(result, ForwardBackwardOutput):
        ...     # Process result
        ...     print(result.metrics)
    """
    # Check for try_again response
    if data.get("type") == "try_again":
        return TryAgainResponse.from_dict(data)

    # Check for error response
    if "error" in data:
        return RequestFailedResponse.from_dict(data)

    # Check for create_model response
    if data.get("type") == "create_model":
        return CreateModelResponse.from_dict(data)

    # Check for unload_model response
    if data.get("type") == "unload_model":
        return UnloadModelResponse.from_dict(data)

    # Check for specific result types based on their unique fields
    if "loss_fn_outputs" in data:
        return ForwardBackwardOutput.from_dict(data)

    if "metrics" in data and "grad_norm" in data.get("metrics", {}):
        return OptimStepResponse.from_dict(data)

    # For save/load responses, check for 'path' field
    if "path" in data:
        # Distinguish between different save/load types
        # SaveWeightsForSamplerResponse typically has model_path too
        if "model_path" in data:
            # This looks like SaveWeightsForSamplerResponse but that type only has 'path'
            # Need to check the context - for now, treat as SaveWeightsResponse
            pass
        return SaveWeightsResponse.from_dict(data)

    # Check for model_id without type (could be UnloadModelResponse without type field)
    if "model_id" in data and len(data) == 1:
        return UnloadModelResponse.from_dict(data)

    # Default fallback - try to parse as ForwardBackwardOutput
    # This handles cases where the response structure is recognized
    try:
        return ForwardBackwardOutput.from_dict(data)
    except Exception:
        pass

    # If we can't determine the type, return as RequestFailedResponse
    return RequestFailedResponse(
        error=f"Unknown response type: {data}",
        category="unknown",
    )
