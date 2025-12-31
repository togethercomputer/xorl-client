"""TrainingRun types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .checkpoint import Checkpoint
from .cursor import Cursor

__all__ = ["TrainingRun", "TrainingRunsResponse"]


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
