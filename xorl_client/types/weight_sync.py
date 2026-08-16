"""Weight sync types for NCCL-based weight transfer to inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

__all__ = [
    "InferenceEndpoint",
    "InferenceEndpointServerInfo",
    "AddInferenceEndpointResponse",
    "ListInferenceEndpointsResponse",
    "RemoveInferenceEndpointResponse",
    "EndpointSyncResult",
    "SyncWeightsResponse",
    "ConnectedEndpoint",
    "ConnectEndpointResponse",
    "DisconnectResponse",
]


@dataclass
class InferenceEndpointServerInfo:
    """Server info from inference endpoint's /server_info endpoint."""

    model_path: Optional[str] = None
    served_model_name: Optional[str] = None
    tp_size: Optional[int] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceEndpointServerInfo":
        return cls(
            model_path=data.get("model_path"),
            served_model_name=data.get("served_model_name"),
            tp_size=data.get("tp_size"),
            enable_lora=data.get("enable_lora"),
            max_lora_rank=data.get("max_lora_rank"),
            version=data.get("version"),
        )


@dataclass
class InferenceEndpoint:
    """Information about a registered inference endpoint."""

    host: str
    port: int
    worker_port: int
    world_size: int
    healthy: bool = True
    server_info: Optional[InferenceEndpointServerInfo] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceEndpoint":
        server_info = None
        if data.get("server_info"):
            server_info = InferenceEndpointServerInfo.from_dict(data["server_info"])
        return cls(
            host=data["host"],
            port=data["port"],
            worker_port=data.get("worker_port", data["port"]),
            world_size=data.get("world_size", 1),
            healthy=data.get("healthy", True),
            server_info=server_info,
        )


@dataclass
class AddInferenceEndpointResponse:
    """Response from add_inference_endpoint operation."""

    success: bool
    message: str
    endpoint: Optional[InferenceEndpoint] = None
    weights_synced: bool = False
    sync_message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AddInferenceEndpointResponse":
        endpoint = None
        if data.get("endpoint"):
            endpoint = InferenceEndpoint.from_dict(data["endpoint"])
        return cls(
            success=data["success"],
            message=data["message"],
            endpoint=endpoint,
            weights_synced=data.get("weights_synced", False),
            sync_message=data.get("sync_message"),
        )


@dataclass
class ListInferenceEndpointsResponse:
    """Response from list_inference_endpoints operation."""

    endpoints: List[InferenceEndpoint] = field(default_factory=list)
    count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListInferenceEndpointsResponse":
        endpoints = [
            InferenceEndpoint.from_dict(ep)
            for ep in data.get("endpoints", [])
        ]
        return cls(
            endpoints=endpoints,
            count=data.get("count", len(endpoints)),
        )


@dataclass
class RemoveInferenceEndpointResponse:
    """Response from remove_inference_endpoint operation."""

    success: bool
    message: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoveInferenceEndpointResponse":
        return cls(
            success=data["success"],
            message=data["message"],
        )


@dataclass
class EndpointSyncResult:
    """Result of syncing weights to a single endpoint."""

    host: str
    port: int
    success: bool
    message: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointSyncResult":
        return cls(
            host=data["host"],
            port=data["port"],
            success=data["success"],
            message=data.get("message", ""),
        )


@dataclass
class SyncWeightsResponse:
    """Response from sync_inference_weights operation.

    Contains throughput and transfer statistics for monitoring.
    """

    success: bool
    message: str
    transfer_time: float = 0.0  # seconds
    total_bytes: int = 0
    num_parameters: int = 0
    num_buckets: int = 0
    endpoints_synced: List[EndpointSyncResult] = field(default_factory=list)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    p2p_rank_summaries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def throughput_gbps(self) -> float:
        """Compute throughput in GB/s."""
        if self.transfer_time > 0:
            return (self.total_bytes / 1e9) / self.transfer_time
        return 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncWeightsResponse":
        endpoints_synced = [
            EndpointSyncResult.from_dict(ep)
            for ep in data.get("endpoints_synced", [])
        ]
        return cls(
            success=data["success"],
            message=data["message"],
            transfer_time=data.get("transfer_time", 0.0),
            total_bytes=data.get("total_bytes", 0),
            num_parameters=data.get("num_parameters", 0),
            num_buckets=data.get("num_buckets", 0),
            endpoints_synced=endpoints_synced,
            timing_breakdown=data.get("timing_breakdown", {}),
            p2p_rank_summaries=data.get("p2p_rank_summaries", []),
        )


@dataclass
class ConnectedEndpoint:
    """Information about a successfully connected inference endpoint."""

    host: str
    port: int
    world_size: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectedEndpoint":
        return cls(
            host=data["host"],
            port=data["port"],
            world_size=data.get("world_size", 1),
        )


@dataclass
class ConnectEndpointResponse:
    """Response from connect_inference_endpoint operation.

    This is returned when establishing a persistent NCCL connection
    to inference endpoints for RL training loops.
    """

    success: bool
    message: str
    connected_endpoints: List[ConnectedEndpoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectEndpointResponse":
        connected_endpoints = [
            ConnectedEndpoint.from_dict(ep)
            for ep in data.get("connected_endpoints", [])
        ]
        return cls(
            success=data["success"],
            message=data["message"],
            connected_endpoints=connected_endpoints,
        )


@dataclass
class DisconnectResponse:
    """Response from disconnect_inference_endpoint operation.

    This is returned when disconnecting from inference endpoints
    and cleaning up NCCL process groups.
    """

    success: bool
    message: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisconnectResponse":
        return cls(
            success=data["success"],
            message=data["message"],
        )
