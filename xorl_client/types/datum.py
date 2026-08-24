"""Datum type definition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .model_input import ModelInput
from .tensor_data import TensorData

__all__ = ["Datum"]

# Mapping from loss_fn_inputs key names to their expected dtypes (matches tinker)
_KEY_TO_DTYPE = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
    # Value-model (critic) training fields: per-token value targets and
    # pre-update value predictions (PPO-style value clipping).
    "returns": "float32",
    "old_values": "float32",
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

    For MOE models with R3 (Rollout Routing Replay), you can pass routed_experts
    to replay the same expert routing decisions from inference during training:

        >>> datum = Datum(
        ...     model_input=ModelInput.from_ints(tokens=[1, 2, 3, 4]),
        ...     loss_fn_inputs={...},
        ...     routed_experts=[[0, 1], [1, 2], [0, 2]],  # Per-token expert indices
        ... )
    """

    def __init__(
        self,
        model_input: ModelInput,
        loss_fn_inputs: Dict[str, Union[TensorData, List, Any]],
        routed_experts: Optional[List[List[List[int]]]] = None,
    ):
        """Initialize Datum with automatic conversion of lists to TensorData.

        Args:
            model_input: The input tokens for the model
            loss_fn_inputs: Dictionary of loss function inputs. Lists and numpy
                arrays are automatically converted to TensorData.
            routed_experts: Optional MOE routing data for R3 (Rollout Routing Replay).
                Shape: [num_tokens, num_layers, topk]. When provided, the model
                will replay these routing decisions instead of computing new ones.
        """
        self.model_input = model_input
        self.loss_fn_inputs = self._convert_loss_fn_inputs(loss_fn_inputs)
        self.routed_experts = routed_experts

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
        """Convert to dictionary for JSON serialization.

        Converts model_input to the flat input_ids format expected by the server:
            {"input_ids": [1, 2, 3, ...]}

        This matches the server's Datum schema which expects:
            model_input: Dict[str, InputType]  # e.g., {"input_ids": [...]}
        """
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

        # Convert model_input to flat input_ids format expected by server
        # Server expects: {"input_ids": [1, 2, 3, ...]}
        # Not the pydantic format: {"chunks": [{"tokens": [...], "type": "..."}]}
        result = {
            "model_input": {"input_ids": self.model_input.to_ints()},
            "loss_fn_inputs": loss_fn_inputs_dict,
        }

        # Include routed_experts for R3 (Rollout Routing Replay) if provided
        if self.routed_experts is not None:
            result["routed_experts"] = self.routed_experts

        return result

    def __repr__(self) -> str:
        """Return a string representation matching tinker's Datum format."""
        if self.routed_experts is not None:
            return (
                f"Datum(loss_fn_inputs={self.loss_fn_inputs!r}, "
                f"model_input={self.model_input!r}, "
                f"routed_experts=[{len(self.routed_experts)} tokens])"
            )
        return f"Datum(loss_fn_inputs={self.loss_fn_inputs!r}, model_input={self.model_input!r})"
