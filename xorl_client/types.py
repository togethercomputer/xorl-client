"""
Types module for XoRL API.

This module provides type definitions that match Tinker's API for compatibility.
Users can construct training data using these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class TensorData:
    """Wrapper for tensor data that can be serialized and sent over the network.

    Supports conversion from PyTorch tensors, NumPy arrays, and Python lists.
    """

    data: List[float]
    dtype: str = "float32"
    shape: Optional[List[int]] = None

    @classmethod
    def from_torch(cls, tensor: Any) -> TensorData:
        """Create TensorData from a PyTorch tensor.

        Args:
            tensor: PyTorch tensor

        Returns:
            TensorData instance
        """
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        # Convert to CPU and numpy
        tensor_np = tensor.detach().cpu().numpy()
        return cls.from_numpy(tensor_np)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> TensorData:
        """Create TensorData from a NumPy array.

        Args:
            array: NumPy array

        Returns:
            TensorData instance
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

        # Flatten and convert to list
        data = array.flatten().tolist()
        shape = list(array.shape)
        dtype = str(array.dtype)

        return cls(data=data, dtype=dtype, shape=shape)

    @classmethod
    def from_list(cls, data: List[float], dtype: str = "float32") -> TensorData:
        """Create TensorData from a Python list.

        Args:
            data: List of numbers
            dtype: Data type string

        Returns:
            TensorData instance
        """
        return cls(data=data, dtype=dtype, shape=[len(data)])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data": self.data,
            "dtype": self.dtype,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TensorData:
        """Create from dictionary (for deserialization)."""
        return cls(
            data=d["data"],
            dtype=d.get("dtype", "float32"),
            shape=d.get("shape"),
        )


@dataclass
class ModelInput:
    """Model input data - typically token IDs.

    Can be created from:
    - List of integers (token IDs)
    - String (will be tokenized by the model)
    """

    tokens: Optional[List[int]] = None
    text: Optional[str] = None

    @classmethod
    def from_ints(cls, tokens: List[int]) -> ModelInput:
        """Create ModelInput from list of token IDs.

        Args:
            tokens: List of token IDs

        Returns:
            ModelInput instance
        """
        return cls(tokens=tokens, text=None)

    @classmethod
    def from_str(cls, text: str) -> ModelInput:
        """Create ModelInput from text string.

        The text will be tokenized by the model.

        Args:
            text: Input text

        Returns:
            ModelInput instance
        """
        return cls(tokens=None, text=text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {}
        if self.tokens is not None:
            result["input_ids"] = self.tokens
        if self.text is not None:
            result["text"] = self.text
        return result

    @property
    def length(self) -> int:
        """Get length of input."""
        if self.tokens is not None:
            return len(self.tokens)
        if self.text is not None:
            return len(self.text)  # Rough estimate
        return 0


@dataclass
class Datum:
    """A single training example with model input and loss function inputs.

    Example:
        >>> datum = Datum(
        ...     model_input=ModelInput.from_ints(tokens=[1, 2, 3, 4]),
        ...     loss_fn_inputs={
        ...         "target_tokens": TensorData.from_list([2, 3, 4, 5]),
        ...         "weights": TensorData.from_list([1.0, 1.0, 1.0, 1.0]),
        ...     }
        ... )
    """

    model_input: ModelInput
    loss_fn_inputs: Dict[str, Union[TensorData, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        loss_fn_inputs_dict = {}
        for key, value in self.loss_fn_inputs.items():
            if isinstance(value, TensorData):
                loss_fn_inputs_dict[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to TensorData
                loss_fn_inputs_dict[key] = TensorData.from_numpy(value).to_dict()
            else:
                # Pass through lists and other values unchanged
                # The server expects plain lists for InputType = Union[List[int], List[float], List[str]]
                loss_fn_inputs_dict[key] = value

        return {
            "model_input": self.model_input.to_dict(),
            "loss_fn_inputs": loss_fn_inputs_dict,
        }


@dataclass
class AdamParams:
    """Adam optimizer parameters."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.stop is not None:
            result["stop"] = self.stop
        return result


@dataclass
class SampledSequence:
    """A single sampled sequence from the model."""

    tokens: List[int]
    logprobs: Optional[List[float]] = None
    text: Optional[str] = None


@dataclass
class SampleResponse:
    """Response from sampling operation."""

    sequences: List[SampledSequence]

    @property
    def text(self) -> str:
        """Get text from first sequence."""
        if self.sequences and self.sequences[0].text:
            return self.sequences[0].text
        return ""

    @property
    def tokens(self) -> List[int]:
        """Get tokens from first sequence."""
        if self.sequences:
            return self.sequences[0].tokens
        return []

    @property
    def logprobs(self) -> Optional[List[float]]:
        """Get logprobs from first sequence."""
        if self.sequences:
            return self.sequences[0].logprobs
        return None


@dataclass
class ForwardBackwardOutput:
    """Output from forward-backward pass."""

    loss_fn_outputs: List[Dict[str, Any]]
    metrics: Dict[str, float]


@dataclass
class OptimStepResponse:
    """Response from optimizer step."""

    metrics: Dict[str, float]
    step: int


@dataclass
class SaveWeightsResponse:
    """Response from save_state operation."""

    path: str


@dataclass
class SaveWeightsForSamplerResponse:
    """Response from save_weights_for_sampler operation."""

    path: str  # Model path like "xorl://model-123/step-100"


@dataclass
class LoadWeightsResponse:
    """Response from load_state operation."""

    success: bool


@dataclass
class LoraConfig:
    """LoRA configuration for training."""

    rank: int = 32
    alpha: Optional[int] = None
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"rank": self.rank, "dropout": self.dropout}
        if self.alpha is not None:
            result["alpha"] = self.alpha
        if self.target_modules is not None:
            result["target_modules"] = self.target_modules
        return result


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


@dataclass
class UpdateServerlessWeightsResponse:
    """Response from update_serverless_weights operation."""

    provider_model_id: str  # model name for inference
    hf_repo_url: str  # HuggingFace repository URL
    checkpoint_path: str  # Local checkpoint path
    status: str  # Upload status ('Complete', 'submitted', etc.)


# Type aliases for backward compatibility
ModelID = str
LossFnType = str  # "cross_entropy", "importance_sampling", "ppo", etc.
