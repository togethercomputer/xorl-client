"""ForwardBackwardOutput type definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .tensor_data import TensorData

__all__ = ["ForwardBackwardOutput", "LossFnOutput"]


def _to_tensor_data(
    value: Optional[Union[Dict[str, Any], TensorData]],
) -> Optional[TensorData]:
    """Convert a dict or TensorData to TensorData, or return None."""
    if value is None:
        return None
    if isinstance(value, TensorData):
        return value
    if isinstance(value, dict):
        return TensorData.from_dict(value)
    return None


def _to_scalar_loss(value: Any) -> Optional[float]:
    """Convert scalar loss wire formats to a Python float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, TensorData):
        data = value.data
    elif isinstance(value, dict):
        if {"data", "dtype"}.issubset(value):
            data = TensorData.from_dict(value).data
        elif "loss" in value:
            return _to_scalar_loss(value["loss"])
        else:
            raise TypeError(f"Unsupported loss value format: {value!r}")
    else:
        raise TypeError(f"Unsupported loss value type: {type(value)!r}")

    if len(data) != 1:
        raise ValueError(f"Expected scalar loss TensorData, got {len(data)} values")
    return float(data[0])


@dataclass
class LossFnOutput:
    """Single loss function output.

    For standard loss functions, only 'loss' is populated.
    For cross_entropy with return_per_token=True, 'logprobs' and
    'elementwise_loss' contain per-token TensorData.

    This class supports dict-like access for compatibility with tinker's
    LossFnOutput type alias (Dict[str, TensorData]).

    Example:
        >>> output = LossFnOutput(loss=0.5, logprobs=TensorData(data=[1, 2, 3], dtype="float32"))
        >>> output["loss"]  # dict-like access
        0.5
        >>> output["logprobs"].to_torch()  # returns torch tensor
        >>> output.loss  # attribute access
        0.5
        >>> "loss" in output  # containment check
        True
    """

    loss: Optional[float] = None
    logprobs: Optional[TensorData] = None
    elementwise_loss: Optional[TensorData] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.loss is not None:
            result["loss"] = self.loss
        if self.logprobs is not None:
            result["logprobs"] = (
                self.logprobs.to_dict()
                if isinstance(self.logprobs, TensorData)
                else self.logprobs
            )
        if self.elementwise_loss is not None:
            result["elementwise_loss"] = (
                self.elementwise_loss.to_dict()
                if isinstance(self.elementwise_loss, TensorData)
                else self.elementwise_loss
            )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossFnOutput":
        """Create from dictionary, converting logprobs/elementwise_loss to TensorData."""
        return cls(
            loss=_to_scalar_loss(data.get("loss")),
            logprobs=_to_tensor_data(data.get("logprobs")),
            elementwise_loss=_to_tensor_data(data.get("elementwise_loss")),
        )

    # Dict-like access methods for tinker compatibility
    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access: output['loss']."""
        if key == "loss":
            return self.loss
        elif key == "logprobs":
            return self.logprobs
        elif key == "elementwise_loss":
            return self.elementwise_loss
        else:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Enable 'in' operator: 'loss' in output."""
        if key == "loss":
            return self.loss is not None
        elif key == "logprobs":
            return self.logprobs is not None
        elif key == "elementwise_loss":
            return self.elementwise_loss is not None
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method with default value."""
        try:
            value = self[key]
            return value if value is not None else default
        except KeyError:
            return default

    def keys(self) -> List[str]:
        """Return keys of non-None values."""
        result = []
        if self.loss is not None:
            result.append("loss")
        if self.logprobs is not None:
            result.append("logprobs")
        if self.elementwise_loss is not None:
            result.append("elementwise_loss")
        return result

    def values(self) -> List[Any]:
        """Return non-None values."""
        result = []
        if self.loss is not None:
            result.append(self.loss)
        if self.logprobs is not None:
            result.append(self.logprobs)
        if self.elementwise_loss is not None:
            result.append(self.elementwise_loss)
        return result

    def items(self) -> List[tuple]:
        """Return (key, value) pairs for non-None values."""
        result = []
        if self.loss is not None:
            result.append(("loss", self.loss))
        if self.logprobs is not None:
            result.append(("logprobs", self.logprobs))
        if self.elementwise_loss is not None:
            result.append(("elementwise_loss", self.elementwise_loss))
        return result


@dataclass
class ForwardBackwardOutput:
    """Output from forward-backward pass.

    Attributes:
        loss_fn_output_type: Type of loss function used (e.g., "cross_entropy")
        loss_fn_outputs: List of loss outputs for each datum in the batch
        metrics: Training metrics (e.g., loss, grad_norm)
        info: Additional information (optional)
    """

    loss_fn_outputs: List[LossFnOutput] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    loss_fn_output_type: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss_fn_output_type": self.loss_fn_output_type,
            "loss_fn_outputs": [
                o.to_dict() if hasattr(o, "to_dict") else o
                for o in self.loss_fn_outputs
            ],
            "metrics": self.metrics,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForwardBackwardOutput":
        """Create from dictionary."""
        loss_fn_outputs_raw = data.get("loss_fn_outputs", [])
        loss_fn_outputs = []
        for output in loss_fn_outputs_raw:
            if isinstance(output, dict):
                loss_fn_outputs.append(LossFnOutput.from_dict(output))
            elif isinstance(output, LossFnOutput):
                loss_fn_outputs.append(output)
            else:
                # Handle legacy format (raw dict stored as-is)
                loss_fn_outputs.append(
                    LossFnOutput(
                        loss=output.get("loss") if isinstance(output, dict) else None
                    )
                )

        return cls(
            loss_fn_output_type=data.get("loss_fn_output_type", ""),
            loss_fn_outputs=loss_fn_outputs,
            metrics=data.get("metrics", {}),
            info=data.get("info", {}),
        )
