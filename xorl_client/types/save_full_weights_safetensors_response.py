"""Response type for save_full_weights_safetensors operation."""

from dataclasses import dataclass


@dataclass
class SaveFullWeightsSafetensorsResponse:
    """Response from save_full_weights_safetensors operation.

    Returns information about the saved safetensors checkpoint.
    """

    path: str  # Filesystem path to saved safetensors directory
    dtype: str  # Dtype used for saving (e.g., "bfloat16")
    num_shards: int  # Number of safetensor shards created

    @classmethod
    def from_dict(cls, data: dict) -> "SaveFullWeightsSafetensorsResponse":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            dtype=data["dtype"],
            num_shards=data["num_shards"],
        )
