"""
Types module for XoRL API.

This module provides type definitions that match Tinker's API for compatibility.
Users can construct training data using these types.

Like tinker, each type is in its own file and can be imported directly:
    from xorl_client.types.tensor_data import TensorData
    from xorl_client.types.model_input import ModelInput

Or import from the package:
    from xorl_client.types import TensorData, ModelInput
    from xorl_client import types
    types.TensorData
"""

from __future__ import annotations

# Base classes
from .strict_base import StrictBase as StrictBase

# Tensor types
from .tensor_dtype import TensorDtype as TensorDtype
from .tensor_data import TensorData as TensorData

# Model input chunk types
from .encoded_text_chunk import EncodedTextChunk as EncodedTextChunk
from .image_chunk import ImageChunk as ImageChunk
from .image_asset_pointer_chunk import ImageAssetPointerChunk as ImageAssetPointerChunk
from .model_input_chunk import ModelInputChunk as ModelInputChunk

# Model input
from .model_input import ModelInput as ModelInput

# Training data
from .datum import Datum as Datum

# Optimizer params
from .adam_params import AdamParams as AdamParams

# Sampling types
from .sampling_params import SamplingParams as SamplingParams
from .sampled_sequence import SampledSequence as SampledSequence
from .sampled_sequence import StopReason as StopReason
from .sample_response import SampleResponse as SampleResponse

# Training response types
from .forward_backward_output import ForwardBackwardOutput as ForwardBackwardOutput
from .forward_backward_output import LossFnOutput as LossFnOutput
from .optim_step_response import OptimStepResponse as OptimStepResponse
from .save_weights_response import SaveWeightsResponse as SaveWeightsResponse
from .save_weights_for_sampler_response import (
    SaveWeightsForSamplerResponse as SaveWeightsForSamplerResponse,
)
from .create_sampling_session_response import (
    CreateSamplingSessionResponse as CreateSamplingSessionResponse,
)
from .load_weights_response import LoadWeightsResponse as LoadWeightsResponse
from .weights_info_response import WeightsInfoResponse as WeightsInfoResponse

# LoRA config
from .lora_config import LoraConfig as LoraConfig

# Dedicated endpoint types
from .dedicated_endpoint import (
    UpdateDedicatedEndpointResponse as UpdateDedicatedEndpointResponse,
)
from .dedicated_endpoint import DedicatedEndpointConfig as DedicatedEndpointConfig

# Serverless weights
from .serverless_weights import (
    UpdateServerlessWeightsResponse as UpdateServerlessWeightsResponse,
)

# Server capabilities
from .server_capabilities import ServerCapabilities as ServerCapabilities

# Checkpoint types
from .checkpoint import Checkpoint as Checkpoint
from .checkpoint import CheckpointsListResponse as CheckpointsListResponse
from .checkpoint import DeleteCheckpointResponse as DeleteCheckpointResponse
from .checkpoint import ParsedCheckpointXoRLPath as ParsedCheckpointXoRLPath
from .checkpoint import CheckpointType as CheckpointType

# Cursor
from .cursor import Cursor as Cursor

# Training run types
from .training_run import TrainingRun as TrainingRun
from .training_run import TrainingRunsResponse as TrainingRunsResponse

# Type aliases
from .model_id import ModelID as ModelID
from .model_id import LossFnType as LossFnType
from .request_id import RequestID as RequestID

# Two-phase request pattern types
from .untyped_api_future import UntypedAPIFuture as UntypedAPIFuture
from .try_again_response import TryAgainResponse as TryAgainResponse
from .request_error_category import RequestErrorCategory as RequestErrorCategory
from .request_failed_response import RequestFailedResponse as RequestFailedResponse
from .future_retrieve_request import FutureRetrieveRequest as FutureRetrieveRequest
from .future_retrieve_response import FutureRetrieveResponse as FutureRetrieveResponse
from .future_retrieve_response import parse_future_retrieve_response as parse_future_retrieve_response
from .create_model_response import CreateModelResponse as CreateModelResponse
from .unload_model_response import UnloadModelResponse as UnloadModelResponse
