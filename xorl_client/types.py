"""
Types module for XoRL API.

This module provides type definitions that match Tinker's API for compatibility.
Users can construct training data using these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


# TensorDtype matches tinker's definition
TensorDtype = str  # Literal["int64", "float32"] in tinker, but we use str for flexibility


@dataclass
class TensorData:
    """Wrapper for tensor data that can be serialized and sent over the network.

    Supports conversion from PyTorch tensors, NumPy arrays, and Python lists.
    Matches tinker's TensorData API.
    """

    data: List[Union[int, float]]
    dtype: TensorDtype = "float32"
    shape: Optional[List[int]] = None

    def __post_init__(self):
        """Validate shape matches data length."""
        if self.shape is not None:
            expected_size = 1
            for dim in self.shape:
                expected_size *= dim
            if expected_size != len(self.data):
                raise ValueError(
                    f"TensorData shape {self.shape} (size {expected_size}) "
                    f"doesn't match data length {len(self.data)}"
                )

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
        # Convert numpy dtype to TensorDtype (matches tinker's logic)
        if array.dtype.kind == "f":
            dtype = "float32"
        elif array.dtype.kind == "i":
            dtype = "int64"
        else:
            dtype = str(array.dtype)

        return cls(data=data, dtype=dtype, shape=shape)

    @classmethod
    def from_list(cls, data: List[Union[int, float]], dtype: TensorDtype = "float32") -> TensorData:
        """Create TensorData from a Python list.

        Args:
            data: List of numbers
            dtype: Data type string ("int64" or "float32")

        Returns:
            TensorData instance
        """
        return cls(data=data, dtype=dtype, shape=[len(data)])

    def to_numpy(self) -> np.ndarray:
        """Convert TensorData to numpy array.

        Returns:
            NumPy array with correct dtype and shape
        """
        if self.dtype == "float32":
            numpy_dtype = np.float32
        elif self.dtype == "int64":
            numpy_dtype = np.int64
        else:
            numpy_dtype = np.float32  # Default fallback

        arr = np.array(self.data, dtype=numpy_dtype)
        if self.shape is not None:
            arr = arr.reshape(self.shape)
        return arr

    def to_torch(self) -> Any:
        """Convert TensorData to torch tensor.

        Returns:
            PyTorch tensor with correct dtype and shape
        """
        import torch

        if self.dtype == "float32":
            torch_dtype = torch.float32
        elif self.dtype == "int64":
            torch_dtype = torch.int64
        else:
            torch_dtype = torch.float32  # Default fallback

        tensor = torch.tensor(self.data, dtype=torch_dtype)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data": self.data,
            "dtype": self.dtype,
            "shape": self.shape,
        }

    def __repr__(self) -> str:
        """Return a string representation matching tinker's TensorData format."""
        return f"TensorData(data={self.data}, dtype={self.dtype!r}, shape={self.shape})"

    def tolist(self) -> List:
        """Return the data as a Python list, respecting shape.

        This method provides compatibility with numpy/torch tensor APIs.
        For 1D data, returns flat list. For multi-dimensional data,
        returns nested lists matching the shape.

        Returns:
            The data as a (possibly nested) list
        """
        return self.to_numpy().tolist()

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

    def to_ints(self) -> List[int]:
        """Get the token IDs as a list of integers.

        Returns:
            List of token IDs

        Raises:
            ValueError: If tokens are not set (only text is provided)
        """
        if self.tokens is None:
            raise ValueError("Cannot convert to ints: tokens are not set (only text is provided)")
        return self.tokens

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

    def __repr__(self) -> str:
        """Return a string representation matching tinker's ModelInput format."""
        parts = []
        if self.tokens is not None:
            parts.append(f"tokens={self.tokens}")
        if self.text is not None:
            parts.append(f"text={self.text!r}")
        return f"ModelInput({', '.join(parts)})"


# Mapping from loss_fn_inputs key names to their expected dtypes (matches tinker)
_KEY_TO_DTYPE = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}


class Datum:
    """A single training example with model input and loss function inputs.

    Lists in loss_fn_inputs are automatically converted to TensorData objects,
    matching tinker's behavior. This allows using .tolist() on the values.

    Example:
        >>> datum = Datum(
        ...     model_input=ModelInput.from_ints(tokens=[1, 2, 3, 4]),
        ...     loss_fn_inputs={
        ...         "target_tokens": [2, 3, 4, 5],  # Auto-converted to TensorData
        ...         "weights": [1.0, 1.0, 1.0, 1.0],  # Auto-converted to TensorData
        ...     }
        ... )
        >>> datum.loss_fn_inputs["weights"].tolist()  # Works!
        [1.0, 1.0, 1.0, 1.0]
    """

    def __init__(
        self,
        model_input: ModelInput,
        loss_fn_inputs: Dict[str, Union[TensorData, List, Any]],
    ):
        """Initialize Datum with automatic conversion of lists to TensorData.

        Args:
            model_input: The input tokens for the model
            loss_fn_inputs: Dictionary of loss function inputs. Lists and numpy
                arrays are automatically converted to TensorData.
        """
        self.model_input = model_input
        self.loss_fn_inputs = self._convert_loss_fn_inputs(loss_fn_inputs)

    @staticmethod
    def _convert_loss_fn_inputs(
        loss_fn_inputs: Dict[str, Any]
    ) -> Dict[str, Union[TensorData, Any]]:
        """Convert lists and numpy arrays to TensorData."""
        converted = {}
        for key, value in loss_fn_inputs.items():
            if isinstance(value, TensorData):
                converted[key] = value
            elif isinstance(value, np.ndarray):
                converted[key] = TensorData.from_numpy(value)
            elif isinstance(value, list):
                # Infer dtype from key name, default to float32
                dtype = _KEY_TO_DTYPE.get(key, "float32")
                converted[key] = TensorData(data=value, dtype=dtype, shape=[len(value)])
            else:
                converted[key] = value
        return converted

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
                # Pass through other values unchanged
                loss_fn_inputs_dict[key] = value

        return {
            "model_input": self.model_input.to_dict(),
            "loss_fn_inputs": loss_fn_inputs_dict,
        }

    def __repr__(self) -> str:
        """Return a string representation matching tinker's Datum format."""
        return f"Datum(loss_fn_inputs={self.loss_fn_inputs!r}, model_input={self.model_input!r})"


@dataclass
class AdamParams:
    """Adam optimizer parameters.

    Defaults match Tinker's AdamParams for compatibility.
    """

    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""

    weight_decay: float = 0.0
    """Weight decay for the optimizer. Uses decoupled weight decay."""

    grad_clip_norm: float = 0.0
    """Gradient clip norm for the optimizer. 0.0 means no clipping."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
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


@dataclass
class SaveWeightsResponse:
    """Response from save_weights operation.

    Returns a xorl:// URI pointing to the saved checkpoint.
    """

    path: str  # XoRL URI (e.g., xorl://default/weights/checkpoint-001)


@dataclass
class SaveWeightsForSamplerResponse:
    """Response from save_weights_for_sampler operation."""

    path: str  # Model path like "xorl://model-123/step-100"


@dataclass
class LoadWeightsResponse:
    """Response from load_weights operation."""

    path: str  # XoRL URI that was loaded


@dataclass
class WeightsInfoResponse:
    """Response from get_weights_info operation.

    Contains metadata about a checkpoint needed to resume training.
    """

    base_model: str
    """The base model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')"""

    is_lora: bool = True
    """Whether this is a LoRA checkpoint"""

    lora_rank: Optional[int] = None
    """The LoRA rank if this is a LoRA checkpoint"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeightsInfoResponse":
        """Create from dictionary (for deserialization)."""
        return cls(
            base_model=d.get("base_model", ""),
            is_lora=d.get("is_lora", True),
            lora_rank=d.get("lora_rank"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "base_model": self.base_model,
            "is_lora": self.is_lora,
            "lora_rank": self.lora_rank,
        }


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


@dataclass
class ServerCapabilities:
    """Capabilities of the XoRL training server.

    Contains information about what the server supports, including
    supported models, loss functions, and other features.
    """

    supported_models: List[str]
    supported_loss_functions: List[str]
    max_batch_size: int
    max_sequence_length: int
    version: str
    features: Dict[str, bool]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerCapabilities":
        """Create from dictionary (for deserialization)."""
        return cls(
            supported_models=d.get("supported_models", []),
            supported_loss_functions=d.get("supported_loss_functions", []),
            max_batch_size=d.get("max_batch_size", 0),
            max_sequence_length=d.get("max_sequence_length", 0),
            version=d.get("version", "unknown"),
            features=d.get("features", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "supported_models": self.supported_models,
            "supported_loss_functions": self.supported_loss_functions,
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length,
            "version": self.version,
            "features": self.features,
        }


# Type aliases for backward compatibility
ModelID = str
LossFnType = str  # "cross_entropy", "importance_sampling", "ppo", etc.
CheckpointType = str  # "training" or "sampler"


@dataclass
class Checkpoint:
    """A checkpoint saved during training.

    Matches tinker's Checkpoint API for compatibility.
    """

    checkpoint_id: str
    """The checkpoint ID (e.g., 'weights/000' or 'sampler_weights/step-100')"""

    checkpoint_type: CheckpointType
    """The type of checkpoint ('training' or 'sampler')"""

    time: str
    """ISO format timestamp when the checkpoint was created"""

    path: str
    """The xorl:// path to the checkpoint"""

    size_bytes: Optional[int] = None
    """The size of the checkpoint in bytes"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary (for deserialization)."""
        return cls(
            checkpoint_id=d["checkpoint_id"],
            checkpoint_type=d["checkpoint_type"],
            time=d["time"],
            path=d["path"],
            size_bytes=d.get("size_bytes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type,
            "time": self.time,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@dataclass
class CheckpointsListResponse:
    """Response from list_checkpoints operation.

    Matches tinker's CheckpointsListResponse API for compatibility.
    """

    checkpoints: List[Checkpoint]
    """List of available checkpoints"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointsListResponse":
        """Create from dictionary (for deserialization)."""
        checkpoints = [Checkpoint.from_dict(c) for c in d.get("checkpoints", [])]
        return cls(checkpoints=checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }


@dataclass
class DeleteCheckpointResponse:
    """Response from delete_checkpoint operation."""

    success: bool
    """Whether the deletion was successful"""

    deleted_path: Optional[str] = None
    """The xorl:// path that was deleted"""

    error: Optional[str] = None
    """Error message if deletion failed"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeleteCheckpointResponse":
        """Create from dictionary (for deserialization)."""
        return cls(
            success=d["success"],
            deleted_path=d.get("deleted_path"),
            error=d.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "deleted_path": self.deleted_path,
            "error": self.error,
        }


@dataclass
class ParsedCheckpointXoRLPath:
    """Parsed xorl:// checkpoint path.

    Matches tinker's ParsedCheckpointTinkerPath for compatibility.
    """

    xorl_path: str
    """The original xorl:// path"""

    model_id: str
    """The model ID extracted from the path"""

    checkpoint_type: CheckpointType
    """The type of checkpoint ('training' or 'sampler')"""

    checkpoint_id: str
    """The checkpoint ID (e.g., 'weights/000')"""

    @classmethod
    def from_xorl_path(cls, xorl_path: str) -> "ParsedCheckpointXoRLPath":
        """Parse a xorl:// path into its components.

        Args:
            xorl_path: The xorl:// path (e.g., 'xorl://default/weights/000')

        Returns:
            ParsedCheckpointXoRLPath with extracted components

        Raises:
            ValueError: If the path is invalid
        """
        if not xorl_path.startswith("xorl://"):
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        # Remove "xorl://" prefix
        parts = xorl_path[7:].split("/")

        if len(parts) < 3:
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        model_id = parts[0]
        checkpoint_type_dir = parts[1]

        if checkpoint_type_dir not in ["weights", "sampler_weights"]:
            raise ValueError(f"Invalid xorl_client path: {xorl_path}")

        checkpoint_type = "training" if checkpoint_type_dir == "weights" else "sampler"
        checkpoint_id = "/".join([checkpoint_type_dir] + parts[2:])

        return cls(
            xorl_path=xorl_path,
            model_id=model_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )


@dataclass
class Cursor:
    """Pagination cursor information.

    Matches tinker's Cursor for API compatibility.
    """

    offset: int
    """The offset used for pagination"""

    limit: int
    """The maximum number of items requested"""

    total_count: int
    """The total number of items available"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cursor":
        """Create from dictionary (for deserialization)."""
        return cls(
            offset=d["offset"],
            limit=d["limit"],
            total_count=d["total_count"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "offset": self.offset,
            "limit": self.limit,
            "total_count": self.total_count,
        }


@dataclass
class TrainingRun:
    """Information about a training run.

    Matches tinker's TrainingRun for API compatibility.
    Note: In xorl_client, there is typically only a single training run with model_id="default".
    """

    training_run_id: str
    """The unique identifier for the training run"""

    base_model: str
    """The base model name this model is derived from"""

    model_owner: str
    """The owner/creator of this model"""

    is_lora: bool
    """Whether this model uses LoRA (Low-Rank Adaptation)"""

    last_request_time: str
    """ISO timestamp of the last request made to this model"""

    corrupted: bool = False
    """Whether the model is in a corrupted state"""

    lora_rank: Optional[int] = None
    """The LoRA rank if this is a LoRA model, null otherwise"""

    last_checkpoint: Optional[Checkpoint] = None
    """The most recent training checkpoint, if available"""

    last_sampler_checkpoint: Optional[Checkpoint] = None
    """The most recent sampler checkpoint, if available"""

    user_metadata: Optional[Dict[str, str]] = None
    """Optional metadata about this training run, set by the end-user"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingRun":
        """Create from dictionary (for deserialization)."""
        last_checkpoint = None
        if d.get("last_checkpoint"):
            last_checkpoint = Checkpoint.from_dict(d["last_checkpoint"])

        last_sampler_checkpoint = None
        if d.get("last_sampler_checkpoint"):
            last_sampler_checkpoint = Checkpoint.from_dict(d["last_sampler_checkpoint"])

        return cls(
            training_run_id=d["training_run_id"],
            base_model=d["base_model"],
            model_owner=d["model_owner"],
            is_lora=d["is_lora"],
            last_request_time=d["last_request_time"],
            corrupted=d.get("corrupted", False),
            lora_rank=d.get("lora_rank"),
            last_checkpoint=last_checkpoint,
            last_sampler_checkpoint=last_sampler_checkpoint,
            user_metadata=d.get("user_metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "training_run_id": self.training_run_id,
            "base_model": self.base_model,
            "model_owner": self.model_owner,
            "is_lora": self.is_lora,
            "corrupted": self.corrupted,
            "lora_rank": self.lora_rank,
            "last_request_time": self.last_request_time,
            "user_metadata": self.user_metadata,
        }
        if self.last_checkpoint:
            result["last_checkpoint"] = self.last_checkpoint.to_dict()
        if self.last_sampler_checkpoint:
            result["last_sampler_checkpoint"] = self.last_sampler_checkpoint.to_dict()
        return result


@dataclass
class TrainingRunsResponse:
    """Response from list_training_runs operation.

    Matches tinker's TrainingRunsResponse for API compatibility.
    """

    training_runs: List[TrainingRun]
    """List of training runs"""

    cursor: Cursor
    """Pagination cursor information"""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingRunsResponse":
        """Create from dictionary (for deserialization)."""
        training_runs = [TrainingRun.from_dict(tr) for tr in d.get("training_runs", [])]
        cursor = Cursor.from_dict(d["cursor"])
        return cls(training_runs=training_runs, cursor=cursor)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "training_runs": [tr.to_dict() for tr in self.training_runs],
            "cursor": self.cursor.to_dict(),
        }
