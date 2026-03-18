"""TensorData type definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .tensor_dtype import TensorDtype

__all__ = ["TensorData"]


@dataclass
class TensorData:
    """Wrapper for tensor data that can be serialized and sent over the network.

    Supports conversion from PyTorch tensors, NumPy arrays, and Python lists.
    Matches tinker's TensorData API.
    """

    data: List[Union[int, float]]
    dtype: TensorDtype = "float32"
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
