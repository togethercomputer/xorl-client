"""
TrainingClient - Tinker-compatible training client with futures.

This is the new version that uses ClientHolder and returns APIFutures.

Request Ordering:
================
Each request gets a sequential `seq_id` that the server uses to enforce
execution order via SeqIdAwareFIFOPolicy. The client dispatches requests
immediately (non-blocking) and the server handles ordering.

This prevents race conditions like optim_step executing before forward_backward.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union

from xorl_client import types
from xorl_client.client.api_future import APIFuture, wrap_future
from xorl_client.client.api_future_impl import _APIFuture, _CombinedAPIFuture
from xorl_client.client.chunked_helpers import (
    MAX_CHUNK_LEN,
    MAX_CHUNK_BYTES_COUNT,
    estimate_datum_bytes,
    combine_fwd_bwd_output_results,
)
from xorl_client.client.client_holder import ClientHolder
from xorl_client.exceptions import InternalServerError, BadRequestError

logger = logging.getLogger(__name__)

_R3_ROUTING_FIELDS = ("routed_experts", "routed_expert_logits")


def _sync_quantization_from_env() -> Optional[Dict[str, Any]]:
    raw = os.environ.get("XORL_WEIGHT_SYNC_QUANTIZATION") or os.environ.get(
        "XORL_SYNC_QUANTIZATION"
    )
    if not raw:
        return None
    config = json.loads(raw)
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("XORL_WEIGHT_SYNC_QUANTIZATION must be a JSON object or null")
    return config


if TYPE_CHECKING:
    from xorl_client.client.sampling_client import SamplingClient

def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %s", name, raw_value, default
        )
        return default
    return value


class TrainingClient:
    """Client for training ML models with forward/backward passes and optimization.

    The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
    You typically get one by calling `service_client.create_lora_training_client()`.

    Key methods:
    - forward_backward() - compute gradients for training (returns APIFuture)
    - optim_step() - update model parameters with Adam optimizer (returns APIFuture)
    - save_weights_for_sampler() - save weights for inference (returns APIFuture)
    - save_weights_and_get_sampling_client() - atomic save + create sampling client
    - save_state() - save full checkpoint including optimizer state
    - get_tokenizer() - get the tokenizer for the base model

    Args:
        holder: ClientHolder managing HTTP connections and async operations
        model_id: Unique identifier for the model to train
        base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")

    Example:
        >>> service_client = ServiceClient(base_url="http://localhost:5555")
        >>> training_client = service_client.create_lora_training_client(
        ...     base_model="Qwen/Qwen2.5-3B-Instruct",
        ...     rank=32
        ... )
        >>>
        >>> # Get tokenizer for the base model
        >>> tokenizer = training_client.get_tokenizer()
        >>>
        >>> # Pipeline operations (submit both immediately)
        >>> fwd_bwd_future = training_client.forward_backward(datums, "importance_sampling")
        >>> optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-5))
        >>>
        >>> # Wait for both to complete
        >>> fwd_bwd_result = fwd_bwd_future.result()
        >>> optim_result = optim_future.result()
    """

    def __init__(self, holder: ClientHolder, model_id: str, base_model: str):
        """Initialize TrainingClient.

        Args:
            holder: ClientHolder for connection management
            model_id: Model ID
            base_model: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        """
        self.holder = holder
        self.model_id = model_id
        self.base_model = base_model
        self._tokenizer = None  # Lazy-loaded tokenizer

        # Request ordering mechanism (similar to Tinker)
        # _take_turn ensures HTTP requests are dispatched in seq_id order
        # This prevents race conditions where optim_step arrives before forward_backward
        self._request_id_lock = threading.Lock()
        self._request_id_counter = 0
        self._turn_counter = 0
        self._turn_waiters: Dict[int, asyncio.Event] = {}

        logger.info(
            f"TrainingClient initialized: model_id={model_id}, base_model={base_model}"
        )

    def _get_request_id(self) -> int:
        """Get the next request ID (thread-safe).

        Returns:
            Sequential request ID starting from 0
        """
        with self._request_id_lock:
            request_id = self._request_id_counter
            self._request_id_counter += 1
            return request_id

    @asynccontextmanager
    async def _take_turn(self, request_id: int):
        """Wait for turn to dispatch HTTP request in seq_id order.

        This ensures that even if multiple requests are made concurrently,
        they are dispatched to the server in the correct order.

        Args:
            request_id: The request ID to wait for

        Example:
            async with self._take_turn(request_id):
                # HTTP request is dispatched here
                future = self.holder.post_async("/api/v1/forward_backward", data)
        """
        assert (
            self._turn_counter <= request_id
        ), f"Same request id cannot be taken twice: turn_counter={self._turn_counter}, request_id={request_id}"

        # Wait if it's not our turn yet
        if self._turn_counter < request_id:
            try:
                event = asyncio.Event()
                self._turn_waiters[request_id] = event
                await event.wait()
            finally:
                del self._turn_waiters[request_id]

        assert self._turn_counter == request_id

        try:
            yield
        finally:
            # Release turn and wake up next waiter
            self._turn_counter += 1
            if self._turn_counter in self._turn_waiters:
                self._turn_waiters[self._turn_counter].set()

    def _convert_tinker_datum(self, datum_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tinker.Datum format to xorl_client format.

        tinker format:
            model_input: {"chunks": [{"tokens": [...], "type": "encoded_text"}]}
            loss_fn_inputs: {"weights": {"data": [...], "dtype": "...", "shape": [...]}}

        xorl_client format:
            model_input: {"input_ids": [...]}
            loss_fn_inputs: {"weights": {"data": [...], "dtype": "...", "shape": [...]}}
        """
        result = {}

        # Convert model_input
        model_input = datum_dict.get("model_input", {})
        if "chunks" in model_input:
            # tinker format - extract tokens from chunks
            tokens = []
            for chunk in model_input["chunks"]:
                if "tokens" in chunk:
                    tokens.extend(chunk["tokens"])
            result["model_input"] = {"input_ids": tokens}
        elif "input_ids" in model_input:
            # Already in xorl_client format
            result["model_input"] = model_input
        else:
            result["model_input"] = model_input

        # Copy loss_fn_inputs as-is (format is compatible)
        if "loss_fn_inputs" in datum_dict:
            result["loss_fn_inputs"] = datum_dict["loss_fn_inputs"]

        return result

    def _chunked_datums(
        self, data: Union[List[types.Datum], List[Dict[str, Any]]]
    ) -> List[tuple]:
        """Split data into chunks respecting count and byte limits.

        Each chunk gets its own request_id allocated from _get_request_id().

        Args:
            data: List of datums (Datum objects or dicts)

        Returns:
            List of (request_id, chunk) tuples where chunk is a list of datums
        """
        self._validate_r3_routing(data)

        if not data:
            # Even empty data gets one chunk so a request is made
            return [(self._get_request_id(), [])]

        chunks = []
        current_chunk = []
        current_bytes = 0
        max_chunk_len = _positive_int_env("XORL_CLIENT_MAX_CHUNK_LEN", MAX_CHUNK_LEN)
        max_chunk_bytes = _positive_int_env(
            "XORL_CLIENT_MAX_CHUNK_BYTES_COUNT",
            _positive_int_env("XORL_CLIENT_MAX_CHUNK_BYTES", MAX_CHUNK_BYTES_COUNT),
        )

        for datum in data:
            datum_bytes = estimate_datum_bytes(datum)

            # Start new chunk if adding this datum would exceed limits
            if current_chunk and (
                len(current_chunk) >= max_chunk_len
                or current_bytes + datum_bytes > max_chunk_bytes
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0

            current_chunk.append(datum)
            current_bytes += datum_bytes

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Allocate request_ids for all chunks
        return [(self._get_request_id(), chunk) for chunk in chunks]

    @staticmethod
    def _routing_field(datum: Any, field: str) -> Any:
        if isinstance(datum, dict):
            return datum.get(field)
        value = getattr(datum, field, None)
        if value is not None:
            return value
        if hasattr(datum, "model_dump"):
            dumped = datum.model_dump()
            if isinstance(dumped, dict):
                return dumped.get(field)
        if hasattr(datum, "to_dict"):
            dumped = datum.to_dict()
            if isinstance(dumped, dict):
                return dumped.get(field)
        return None

    @classmethod
    def _validate_r3_routing(cls, data: List[Any]) -> None:
        """Require complete expert indices and, when supplied, complete logits."""
        if not data:
            return
        experts_present = [
            cls._routing_field(datum, "routed_experts") is not None for datum in data
        ]
        logits_present = [
            cls._routing_field(datum, "routed_expert_logits") is not None
            for datum in data
        ]
        if not any(experts_present) and not any(logits_present):
            return
        if any(logits_present) and not all(logits_present):
            raise ValueError(
                "R3 routed_expert_logits must be present on every datum or absent from the request"
            )
        if not all(experts_present):
            raise ValueError(
                "R3 routed_experts must be present on every datum when routing replay is requested"
            )

    def _convert_datums(self, data: List) -> tuple:
        """Convert a list of datums to dicts, extracting routed_experts.

        Args:
            data: List of Datum objects or dicts

        Returns:
            Tuple of datum dicts, routed expert IDs, and selected router weights.
        """
        datums_dicts = []
        all_routed_experts = []
        all_routed_expert_logits = []

        for datum in data:
            if isinstance(datum, types.Datum):
                datum_dict = datum.to_dict()
                datums_dicts.append(datum_dict)
            elif hasattr(datum, "to_dict"):
                datum_dict = datum.to_dict()
                datums_dicts.append(datum_dict)
            elif hasattr(datum, "model_dump"):
                datum_dict = datum.model_dump()
                datums_dicts.append(self._convert_tinker_datum(datum_dict))
            elif isinstance(datum, dict):
                datum_dict = dict(datum)
                datums_dicts.append(datum_dict)
            else:
                raise TypeError(
                    f"Expected Datum, dict, or Pydantic model, got {type(datum).__name__}"
                )

            routed = datum_dict.pop("routed_experts", None)
            routed_logits = datum_dict.pop("routed_expert_logits", None)
            if routed is not None:
                all_routed_experts.append(routed)
            if routed_logits is not None:
                all_routed_expert_logits.append(routed_logits)

        return datums_dicts, all_routed_experts, all_routed_expert_logits

    def _convert_datums_simple(self, data: List) -> List[Dict[str, Any]]:
        """Convert a list of datums to dicts (no routed_experts extraction).

        Args:
            data: List of Datum objects or dicts

        Returns:
            List of datum dicts
        """
        datums_dicts = []

        for datum in data:
            if isinstance(datum, types.Datum):
                datums_dicts.append(datum.to_dict())
            elif hasattr(datum, "to_dict"):
                datums_dicts.append(datum.to_dict())
            elif hasattr(datum, "model_dump"):
                datum_dict = datum.model_dump()
                datums_dicts.append(self._convert_tinker_datum(datum_dict))
            elif isinstance(datum, dict):
                datums_dicts.append(datum)
            else:
                raise TypeError(
                    f"Expected Datum, dict, or Pydantic model, got {type(datum).__name__}"
                )

        return datums_dicts

    def get_tokenizer(self):
        """Get the tokenizer for the base model.

        The tokenizer is lazy-loaded and cached for subsequent calls.
        Uses HuggingFace's AutoTokenizer to load the tokenizer for the base model.

        Returns:
            PreTrainedTokenizer: The tokenizer for the base model

        Example:
            >>> training_client = service_client.create_lora_training_client(
            ...     base_model="Qwen/Qwen2.5-3B-Instruct"
            ... )
            >>> tokenizer = training_client.get_tokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.base_model

            # Avoid gating of Llama 3 models
            if model_name.startswith("meta-llama/Llama-3"):
                model_name = "thinkingmachineslabinc/meta-llama-3-tokenizer"

            kwargs = {}
            if model_name == "moonshotai/Kimi-K2-Thinking":
                kwargs["trust_remote_code"] = True
                kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, **kwargs
            )
            logger.info(f"Tokenizer loaded for base_model={self.base_model}")
        return self._tokenizer

    def forward_backward(
        self,
        data: Union[List[types.Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
        loss_fn_params: Optional[Dict[str, Any]] = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Execute forward and backward pass (two-phase pattern).

        Large batches are transparently chunked into multiple HTTP requests
        (max 1024 datums or 5MB per chunk). Results are combined into a single
        ForwardBackwardOutput.

        Args:
            data: List of training examples (Datum objects or dicts)
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", "miles_policy_loss", etc.)
            loss_fn_params: Optional parameters for the loss function (e.g., {"eps_clip": 0.2} for PPO clipping)

        Returns:
            APIFuture[ForwardBackwardOutput] that can be awaited

        Example:
            >>> fwd_bwd_future = training_client.forward_backward(datums, "importance_sampling")
            >>> result = fwd_bwd_future.result()  # Block and wait
            >>> print(result.loss_fn_outputs[0]["loss"])
        """
        import time

        # Split data into chunks, each with its own request_id
        chunked = self._chunked_datums(data)

        # Single chunk: use simple path (no _CombinedAPIFuture overhead)
        if len(chunked) == 1:
            request_id, chunk_data = chunked[0]
            datums_dicts, all_routed_experts, all_routed_expert_logits = (
                self._convert_datums(chunk_data)
            )

            request_data = {
                "model_id": self.model_id,
                "seq_id": request_id + 1,
                "forward_backward_input": {
                    "data": datums_dicts,
                    "loss_fn": loss_fn,
                },
            }
            if loss_fn_params:
                request_data["forward_backward_input"][
                    "loss_fn_params"
                ] = loss_fn_params
            if all_routed_experts and len(all_routed_experts) == len(datums_dicts):
                request_data["forward_backward_input"][
                    "routed_experts"
                ] = all_routed_experts
            if all_routed_expert_logits and len(all_routed_expert_logits) == len(
                datums_dicts
            ):
                request_data["forward_backward_input"][
                    "routed_expert_logits"
                ] = all_routed_expert_logits

            async def _forward_backward_async():
                start_time = time.time()
                async with self._take_turn(request_id):

                    async def _send_request():
                        return await self.holder.post(
                            "/api/v1/forward_backward", request_data
                        )

                    result = await self.holder.execute_with_retries(_send_request)
                untyped_future = types.UntypedAPIFuture.from_dict(result)
                return await _APIFuture(
                    model_cls=types.ForwardBackwardOutput,
                    holder=self.holder,
                    untyped_future=untyped_future,
                    request_start_time=start_time,
                    request_type="ForwardBackward",
                )

            return wrap_future(
                self.holder.run_coroutine_threadsafe(_forward_backward_async())
            )

        # Multiple chunks: dispatch each sequentially, poll in parallel, combine
        logger.info(
            f"Chunking forward_backward into {len(chunked)} chunks "
            f"({sum(len(c) for _, c in chunked)} total datums)"
        )

        # Pre-convert all chunks and build request data
        chunk_requests = []
        for request_id, chunk_data in chunked:
            datums_dicts, all_routed_experts, all_routed_expert_logits = (
                self._convert_datums(chunk_data)
            )
            rd = {
                "model_id": self.model_id,
                "seq_id": request_id + 1,
                "forward_backward_input": {
                    "data": datums_dicts,
                    "loss_fn": loss_fn,
                },
            }
            if loss_fn_params:
                rd["forward_backward_input"]["loss_fn_params"] = loss_fn_params
            if all_routed_experts and len(all_routed_experts) == len(datums_dicts):
                rd["forward_backward_input"]["routed_experts"] = all_routed_experts
            if all_routed_expert_logits and len(all_routed_expert_logits) == len(
                datums_dicts
            ):
                rd["forward_backward_input"][
                    "routed_expert_logits"
                ] = all_routed_expert_logits
            chunk_requests.append((request_id, rd))

        async def _chunked_forward_backward_async():
            futures = []
            # Dispatch chunks sequentially (ordering via _take_turn)
            for req_id, rd in chunk_requests:
                start_time = time.time()
                async with self._take_turn(req_id):
                    # Use default arg to capture rd by value
                    async def _send_request(request_data=rd):
                        return await self.holder.post(
                            "/api/v1/forward_backward", request_data
                        )

                    result = await self.holder.execute_with_retries(_send_request)
                untyped_future = types.UntypedAPIFuture.from_dict(result)
                # _APIFuture starts polling immediately on construction
                futures.append(
                    _APIFuture(
                        model_cls=types.ForwardBackwardOutput,
                        holder=self.holder,
                        untyped_future=untyped_future,
                        request_start_time=start_time,
                        request_type="ForwardBackward",
                    )
                )

            # Wait for all polls in parallel and combine results
            results = await asyncio.gather(*[f.result_async() for f in futures])
            return combine_fwd_bwd_output_results(results)

        return wrap_future(
            self.holder.run_coroutine_threadsafe(_chunked_forward_backward_async())
        )

    def forward(
        self,
        data: Union[List[types.Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
        loss_fn_params: Optional[Dict[str, Any]] = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Execute forward pass only (no backward/gradient computation, two-phase pattern).

        This is useful for validation/evaluation during training where you want
        to compute loss metrics without updating gradients. The server uses
        torch.no_grad() for efficiency.

        Large batches are transparently chunked into multiple HTTP requests
        (max 1024 datums or 5MB per chunk). Results are combined into a single
        ForwardBackwardOutput.

        Args:
            data: List of training examples (Datum objects or dicts)
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", etc.)
            loss_fn_params: Optional parameters for the loss function (e.g., {"compute_kl_stats": True})

        Returns:
            APIFuture[ForwardBackwardOutput] that can be awaited

        Example:
            >>> # Validation pass
            >>> fwd_future = training_client.forward(val_datums, "cross_entropy")
            >>> result = fwd_future.result()
            >>> print(f"Validation loss: {result.metrics['loss:mean']}")
        """
        import time

        # Split data into chunks, each with its own request_id
        chunked = self._chunked_datums(data)

        # Single chunk: use simple path
        if len(chunked) == 1:
            request_id, chunk_data = chunked[0]
            datums_dicts, all_routed_experts, all_routed_expert_logits = (
                self._convert_datums(chunk_data)
            )

            request_data = {
                "model_id": self.model_id,
                "seq_id": request_id + 1,
                "forward_input": {
                    "data": datums_dicts,
                    "loss_fn": loss_fn,
                },
            }
            if loss_fn_params:
                request_data["forward_input"]["loss_fn_params"] = loss_fn_params
            if all_routed_experts and len(all_routed_experts) == len(datums_dicts):
                request_data["forward_input"]["routed_experts"] = all_routed_experts
            if all_routed_expert_logits and len(all_routed_expert_logits) == len(
                datums_dicts
            ):
                request_data["forward_input"][
                    "routed_expert_logits"
                ] = all_routed_expert_logits

            async def _forward_async():
                start_time = time.time()
                async with self._take_turn(request_id):

                    async def _send_request():
                        return await self.holder.post("/api/v1/forward", request_data)

                    result = await self.holder.execute_with_retries(_send_request)
                untyped_future = types.UntypedAPIFuture.from_dict(result)
                return await _APIFuture(
                    model_cls=types.ForwardBackwardOutput,
                    holder=self.holder,
                    untyped_future=untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                )

            return wrap_future(self.holder.run_coroutine_threadsafe(_forward_async()))

        # Multiple chunks: dispatch each sequentially, poll in parallel, combine
        logger.info(
            f"Chunking forward into {len(chunked)} chunks "
            f"({sum(len(c) for _, c in chunked)} total datums)"
        )

        chunk_requests = []
        for request_id, chunk_data in chunked:
            datums_dicts, all_routed_experts, all_routed_expert_logits = (
                self._convert_datums(chunk_data)
            )
            rd = {
                "model_id": self.model_id,
                "seq_id": request_id + 1,
                "forward_input": {
                    "data": datums_dicts,
                    "loss_fn": loss_fn,
                },
            }
            if loss_fn_params:
                rd["forward_input"]["loss_fn_params"] = loss_fn_params
            if all_routed_experts and len(all_routed_experts) == len(datums_dicts):
                rd["forward_input"]["routed_experts"] = all_routed_experts
            if all_routed_expert_logits and len(all_routed_expert_logits) == len(
                datums_dicts
            ):
                rd["forward_input"][
                    "routed_expert_logits"
                ] = all_routed_expert_logits
            chunk_requests.append((request_id, rd))

        async def _chunked_forward_async():
            futures = []
            for req_id, rd in chunk_requests:
                start_time = time.time()
                async with self._take_turn(req_id):

                    async def _send_request(request_data=rd):
                        return await self.holder.post("/api/v1/forward", request_data)

                    result = await self.holder.execute_with_retries(_send_request)
                untyped_future = types.UntypedAPIFuture.from_dict(result)
                futures.append(
                    _APIFuture(
                        model_cls=types.ForwardBackwardOutput,
                        holder=self.holder,
                        untyped_future=untyped_future,
                        request_start_time=start_time,
                        request_type="Forward",
                    )
                )

            results = await asyncio.gather(*[f.result_async() for f in futures])
            return combine_fwd_bwd_output_results(results)

        return wrap_future(
            self.holder.run_coroutine_threadsafe(_chunked_forward_async())
        )

    async def forward_async(
        self,
        data: Union[List[types.Datum], List[Dict[str, Any]]],
        loss_fn: str = "cross_entropy",
        loss_fn_params: Optional[Dict[str, Any]] = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward.

        Execute forward pass only (no backward/gradient computation).
        This is the async wrapper that allows using forward in async contexts.

        Args:
            data: List of training examples (Datum objects or dicts)
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", etc.)
            loss_fn_params: Optional parameters for the loss function

        Returns:
            APIFuture[ForwardBackwardOutput] that can be awaited

        Example:
            >>> # Option 1: Await the future directly
            >>> fwd_future = await training_client.forward_async(val_datums, "cross_entropy")
            >>> result = await fwd_future
            >>> print(f"Validation loss: {result.metrics['loss:mean']}")
            >>>
            >>> # Option 2: Use result_async
            >>> fwd_future = await training_client.forward_async(val_datums, "cross_entropy")
            >>> result = await fwd_future.result_async()
        """
        return self.forward(data, loss_fn, loss_fn_params=loss_fn_params)

    def optim_step(
        self,
        adam_params: Union[types.AdamParams, Dict[str, float]],
    ) -> APIFuture[types.OptimStepResponse]:
        """Perform optimizer step (two-phase pattern).

        Args:
            adam_params: Adam parameters (AdamParams object or dict with learning_rate, etc.)

        Returns:
            APIFuture[OptimStepResponse] that can be awaited

        Example:
            >>> optim_future = training_client.optim_step(
            ...     types.AdamParams(learning_rate=1e-5)
            ... )
            >>> result = optim_future.result()
            >>> print(result.metrics["grad_norm"])
        """
        import time

        # Get request ID for ordering
        request_id = self._get_request_id()

        # Convert AdamParams to dict if needed
        if isinstance(adam_params, types.AdamParams):
            adam_params_dict = adam_params.to_dict()
        else:
            adam_params_dict = adam_params

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,  # seq_id starts from 1 (like Tinker)
            "adam_params": adam_params_dict,
        }

        # Async function that handles both phases
        async def _optim_step_async():
            start_time = time.time()

            # Phase 1: Submit request inside _take_turn to ensure ordering
            async with self._take_turn(request_id):

                async def _send_request():
                    return await self.holder.post("/api/v1/optim_step", request_data)

                result = await self.holder.execute_with_retries(_send_request)

            # Two-phase pattern: parse UntypedAPIFuture and poll for result
            untyped_future = types.UntypedAPIFuture.from_dict(result)

            # Phase 2: Create _APIFuture and await for the result
            return await _APIFuture(
                model_cls=types.OptimStepResponse,
                holder=self.holder,
                untyped_future=untyped_future,
                request_start_time=start_time,
                request_type="OptimStep",
            )

        # Schedule to event loop and return
        return wrap_future(self.holder.run_coroutine_threadsafe(_optim_step_async()))

    def save_weights_for_sampler(
        self,
        name: str,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Save weights specifically for sampling/inference (two-phase pattern).

        This is called frequently (e.g., every batch in RL training) to make
        the latest policy weights available for sampling. It's separate from
        save_state() which includes optimizer state for checkpointing.

        The returned model_path can be used to create a SamplingClient.

        Args:
            name: Name for this checkpoint (e.g., "step-000100")

        Returns:
            APIFuture[SaveWeightsForSamplerResponse] with model_path

        Example:
            >>> save_future = training_client.save_weights_for_sampler("step-100")
            >>> response = save_future.result()
            >>> print(response.path)  # e.g., "xorl://model-123/step-100"
        """
        import time

        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
        }

        # Async function that handles both phases
        async def _save_weights_for_sampler_async():
            start_time = time.time()

            # Phase 1: Submit request inside _take_turn to ensure ordering
            async with self._take_turn(request_id):

                async def _send_request():
                    return await self.holder.post(
                        "/api/v1/save_weights_for_sampler", request_data
                    )

                result = await self.holder.execute_with_retries(_send_request)

            # Two-phase pattern: poll for result
            untyped_future = types.UntypedAPIFuture.from_dict(result)
            return await _APIFuture(
                model_cls=types.SaveWeightsForSamplerResponse,
                holder=self.holder,
                untyped_future=untyped_future,
                request_start_time=start_time,
                request_type="SaveWeightsForSampler",
            )

        # Schedule to event loop and return
        return wrap_future(
            self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())
        )

    async def save_weights_for_sampler_async(
        self,
        name: str,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Async version of save_weights_for_sampler.

        Save weights specifically for sampling/inference.
        This is the async wrapper that allows using save_weights_for_sampler in async contexts.

        Args:
            name: Name for this checkpoint (e.g., "step-000100")

        Returns:
            APIFuture[SaveWeightsForSamplerResponse] with model_path

        Example:
            >>> # Option 1: Await the future directly
            >>> save_future = await training_client.save_weights_for_sampler_async("step-100")
            >>> result = await save_future
            >>> print(f"Weights saved to: {result.path}")
            >>>
            >>> # Option 2: Use result_async
            >>> save_future = await training_client.save_weights_for_sampler_async("step-100")
            >>> result = await save_future.result_async()
        """
        return self.save_weights_for_sampler(name)

    def save_weights_and_get_sampling_client(
        self,
        name: Optional[str] = None,
        inference_base_url: str = "http://localhost:30000",
        api_format: Optional[str] = None,
    ) -> "SamplingClient":
        """Atomic operation: save weights and create sampling client.

        This is the recommended way to get a sampling client with the latest
        policy weights in RL training.

        Args:
            name: Optional name for checkpoint (default: auto-generated)
            inference_base_url: Base URL for inference server (default: http://localhost:30000)
            api_format: Inference API format passed to SamplingClient. Use
                "chat_completions" when sampling through Dispatch.

        Returns:
            SamplingClient ready to use

        Example:
            >>> # Every batch in RL loop
            >>> sampling_client = training_client.save_weights_and_get_sampling_client()
            >>> response = sampling_client.sample(prompt, sampling_params)
        """
        from xorl_client.client.sampling_client import SamplingClient

        # Generate name if not provided
        if name is None:
            import time

            name = f"step-{int(time.time())}"

        # Save weights
        save_result = self.save_weights_for_sampler(name).result()
        model_path = save_result.path

        # Call training server to create sampling session (loads LoRA on inference workers)
        logger.info(f"Creating sampling session: model_path={model_path}")
        try:
            response = self.holder.post_sync(
                "/api/v1/create_sampling_session",
                {"model_path": model_path},
                timeout=300.0,  # LoRA loading can take some time
            )

            if not response.get("success", False):
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Failed to create sampling session: {error_msg}")

            lora_name = response.get("lora_name", "")
            logger.info(
                f"Sampling session created: lora_name={lora_name}, model_path={model_path}"
            )

        except (InternalServerError, BadRequestError) as e:
            # Check if this is a "LoRA already loaded" error - not fatal, just warn
            error_msg = str(e)
            if "already loaded" in error_msg.lower():
                logger.warning(
                    f"LoRA adapter is already loaded on inference worker, continuing: {error_msg}"
                )
            else:
                logger.error(f"Failed to create sampling session for {model_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to create sampling session for {model_path}: {e}")
            raise

        # Create sampling client with new API
        return SamplingClient(
            base_url=inference_base_url,
            model_path=model_path,
            model=self.holder._model,
            api_key=self.holder.api_key,
            timeout=self.holder.timeout,
            api_format=api_format,
        )

    async def save_weights_and_get_sampling_client_async(
        self,
        name: Optional[str] = None,
        inference_base_url: str = "http://localhost:30000",
        api_format: Optional[str] = None,
    ) -> "SamplingClient":
        """Async version of save_weights_and_get_sampling_client.

        Atomic operation: save weights and create sampling client.
        This is the async wrapper for use in async contexts.

        Args:
            name: Optional name for checkpoint (default: auto-generated)
            inference_base_url: Base URL for inference server (default: http://localhost:30000)
            api_format: Inference API format passed to SamplingClient. Use
                "chat_completions" when sampling through Dispatch.

        Returns:
            SamplingClient ready to use

        Example:
            >>> # Every batch in RL loop (async context)
            >>> sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            >>> response = await sampling_client.sample_async(prompt, sampling_params)
        """
        from xorl_client.client.sampling_client import SamplingClient

        # Generate name if not provided
        if name is None:
            import time

            name = f"step-{int(time.time())}"

        # Save weights - await the async version and then get the result
        save_future = await self.save_weights_for_sampler_async(name)
        save_result = await save_future.result_async()
        model_path = save_result.path

        # Call training server to create sampling session (loads LoRA on inference workers)
        logger.info(f"Creating sampling session: model_path={model_path}")
        try:
            response = self.holder.post_sync(
                "/api/v1/create_sampling_session",
                {"model_path": model_path},
                timeout=300.0,  # LoRA loading can take some time
            )

            if not response.get("success", False):
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Failed to create sampling session: {error_msg}")

            lora_name = response.get("lora_name", "")
            logger.info(
                f"Sampling session created: lora_name={lora_name}, model_path={model_path}"
            )

        except (InternalServerError, BadRequestError) as e:
            # Check if this is a "LoRA already loaded" error - not fatal, just warn
            error_msg = str(e)
            if "already loaded" in error_msg.lower():
                logger.warning(
                    f"LoRA adapter is already loaded on inference worker, continuing: {error_msg}"
                )
            else:
                logger.error(f"Failed to create sampling session for {model_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to create sampling session for {model_path}: {e}")
            raise

        # Create sampling client with new API
        return SamplingClient(
            base_url=inference_base_url,
            model_path=model_path,
            model=self.holder._model,
            api_key=self.holder.api_key,
            timeout=self.holder.timeout,
            api_format=api_format,
        )

    def save_state(
        self,
        name: Optional[str] = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save model weights (and optimizer state) to persistent storage.

        This is used for checkpointing/resume, not for frequent sampling updates.
        For RL training, use save_weights_for_sampler() instead.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001"). Auto-generated if not specified.

        Returns:
            APIFuture[SaveWeightsResponse] with xorl:// URI

        Example:
            >>> save_future = training_client.save_state("checkpoint-001")
            >>> response = save_future.result()
            >>> print(response.path)  # xorl://default/weights/checkpoint-001
        """
        import time

        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "path": name,
        }

        # Async function that handles both phases (like Tinker)
        async def _save_state_async():
            start_time = time.time()

            # Phase 1: Submit request inside _take_turn to ensure ordering
            async with self._take_turn(request_id):

                async def _send_request():
                    return await self.holder.post("/api/v1/save_weights", request_data)

                result = await self.holder.execute_with_retries(_send_request)

            # Parse UntypedAPIFuture response
            untyped_future = types.UntypedAPIFuture.from_dict(result)

            # Phase 2: Create _APIFuture and await for the result
            return await _APIFuture(
                model_cls=types.SaveWeightsResponse,
                holder=self.holder,
                untyped_future=untyped_future,
                request_start_time=start_time,
                request_type="SaveWeights",
            )

        # Schedule to event loop and return (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_save_state_async()))

    async def save_state_async(
        self,
        name: Optional[str] = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Async version of save_state.

        Save model weights (and optimizer state) to persistent storage.
        This is the async wrapper that allows using save_state in async contexts.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001"). Auto-generated if not specified.

        Returns:
            APIFuture[SaveWeightsResponse] with xorl:// URI

        Example:
            >>> # Option 1: Await the future directly
            >>> save_future = await training_client.save_state_async("checkpoint-001")
            >>> result = await save_future
            >>> print(f"Saved to: {result.path}")
            >>>
            >>> # Option 2: Use result_async
            >>> save_future = await training_client.save_state_async()
            >>> result = await save_future.result_async()
        """
        return self.save_state(name)

    def save_full_weights_safetensors(
        self,
        name: str,
        dtype: str = "bfloat16",
        base_model_path: Optional[str] = None,
    ) -> APIFuture[types.SaveFullWeightsSafetensorsResponse]:
        """Save full model weights as safetensors for direct SGLang/HuggingFace loading.

        This saves the complete model weights (not just LoRA) in safetensors format
        with config files, allowing direct loading by SGLang or other inference engines.

        For full-weights RL training, this is the recommended way to save checkpoints
        since there are no LoRA adapters to save separately.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001"). Will be saved to
                  output_dir/safetensors/{name}/
            dtype: Target dtype for weights ("bfloat16", "float16", "float32")
            base_model_path: Path to base model for config files. If None, uses
                           server's configured model_path.

        Returns:
            APIFuture[SaveFullWeightsSafetensorsResponse] with:
                - path: Filesystem path to saved safetensors directory
                - dtype: Dtype used for saving
                - num_shards: Number of safetensor shards created

        Example:
            >>> # Save checkpoint during training
            >>> save_future = training_client.save_full_weights_safetensors("step-1000")
            >>> result = save_future.result()
            >>> print(f"Saved to: {result.path} ({result.num_shards} shards)")
        """
        request_data = {
            "model_id": self.model_id,
            "name": name,
            "dtype": dtype,
        }
        if base_model_path is not None:
            request_data["base_model_path"] = base_model_path

        # This is a direct response endpoint (not two-phase pattern)
        async def _save_full_weights_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/save_full_weights_safetensors", request_data
                )

            result = await self.holder.execute_with_retries(_send_request)
            return types.SaveFullWeightsSafetensorsResponse.from_dict(result)

        return wrap_future(
            self.holder.run_coroutine_threadsafe(_save_full_weights_async())
        )

    async def save_full_weights_safetensors_async(
        self,
        name: str,
        dtype: str = "bfloat16",
        base_model_path: Optional[str] = None,
    ) -> APIFuture[types.SaveFullWeightsSafetensorsResponse]:
        """Async version of save_full_weights_safetensors.

        Save full model weights as safetensors for direct SGLang/HuggingFace loading.

        Args:
            name: Checkpoint name (e.g., "checkpoint-001")
            dtype: Target dtype for weights ("bfloat16", "float16", "float32")
            base_model_path: Path to base model for config files

        Returns:
            APIFuture[SaveFullWeightsSafetensorsResponse]

        Example:
            >>> save_future = await training_client.save_full_weights_safetensors_async("step-1000")
            >>> result = await save_future
            >>> print(f"Saved to: {result.path}")
        """
        return self.save_full_weights_safetensors(name, dtype, base_model_path)

    def _load_weights_impl(
        self,
        path: str,
        optimizer: bool,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Internal implementation for loading weights (two-phase pattern).

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")
            optimizer: Whether to load optimizer state

        Returns:
            APIFuture[LoadWeightsResponse]
        """
        import time

        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "path": path,
            "optimizer": optimizer,
        }

        # Async function that handles both phases (like Tinker)
        async def _load_weights_async():
            start_time = time.time()

            # Phase 1: Submit request inside _take_turn to ensure ordering
            async with self._take_turn(request_id):

                async def _send_request():
                    return await self.holder.post("/api/v1/load_weights", request_data)

                result = await self.holder.execute_with_retries(_send_request)

            # Parse UntypedAPIFuture response
            untyped_future = types.UntypedAPIFuture.from_dict(result)

            # Phase 2: Create _APIFuture and await for the result
            return await _APIFuture(
                model_cls=types.LoadWeightsResponse,
                holder=self.holder,
                untyped_future=untyped_future,
                request_start_time=start_time,
                request_type="LoadWeights",
            )

        # Schedule to event loop and return (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_load_weights_async()))

    def load_state(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights from a saved checkpoint.

        This loads only the model weights, not optimizer state (e.g., Adam momentum).
        To also restore optimizer state, use load_state_with_optimizer.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse] with the loaded path

        Example:
            >>> # Load checkpoint to continue training (weights only, optimizer resets)
            >>> load_future = training_client.load_state("xorl://default/weights/checkpoint-001")
            >>> result = load_future.result()
            >>> print(result.path)
        """
        return self._load_weights_impl(path, optimizer=False)

    async def load_state_async(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state.

        Load model weights from a saved checkpoint (without optimizer state).

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse]

        Example:
            >>> load_future = await training_client.load_state_async("xorl://default/weights/checkpoint-001")
            >>> result = await load_future
            >>> print(f"Loaded: {result.path}")
        """
        return self.load_state(path)

    def load_state_with_optimizer(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint.

        This restores both model weights and optimizer state (e.g., Adam momentum),
        allowing training to resume exactly where it left off.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse] with the loaded path

        Example:
            >>> # Resume training with optimizer state
            >>> load_future = training_client.load_state_with_optimizer(
            ...     "xorl://default/weights/checkpoint-001"
            ... )
            >>> result = load_future.result()
            >>> print(result.path)
        """
        return self._load_weights_impl(path, optimizer=True)

    async def load_state_with_optimizer_async(
        self,
        path: str,
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state_with_optimizer.

        Load model weights and optimizer state from a checkpoint.

        Args:
            path: XoRL URI to load from (e.g., "xorl://default/weights/checkpoint-001")

        Returns:
            APIFuture[LoadWeightsResponse]

        Example:
            >>> load_future = await training_client.load_state_with_optimizer_async(
            ...     "xorl://default/weights/checkpoint-001"
            ... )
            >>> result = await load_future
            >>> print(f"Loaded with optimizer: {result.path}")
        """
        return self.load_state_with_optimizer(path)

    def health_check(self) -> Dict[str, Any]:
        """Check API server health.

        Returns:
            Health check response
        """
        return self.holder.get_sync("/health", timeout=60)

    def update_dedicated_endpoint(
        self,
        name: str,
        endpoint_id: str,
        provider_api_key: str,
        hf_token: str,
        hf_org: str = "xorl-ai",
        provider_model_id: Optional[str] = None,
    ) -> APIFuture[types.UpdateDedicatedEndpointResponse]:
        """
        Save checkpoint and update dedicated endpoint with new LoRA weights.

        This triggers the server to:
        1. Save LoRA checkpoint locally
        2. Upload to HuggingFace (private repo)
        3. Upload via models.upload API
        4. Wait for model to be ready
        5. Probe endpoint to verify adapter is loaded

        All heavy lifting happens on the server. The client just waits for completion.

        Note: This operation can take 30-60 seconds due to HF propagation and provider upload.

        Args:
            name: Checkpoint name (e.g., "step-000001")
            endpoint_id: dedicated endpoint ID
            provider_api_key: provider API key
            hf_token: HuggingFace token for uploading
            hf_org: HuggingFace organization (default: "xorl-ai")
            provider_model_id: Optional provider model ID (auto-generated if not provided)

        Returns:
            APIFuture[UpdateDedicatedEndpointResponse] with provider_model_id for sampling

        Example:
            >>> # Update dedicated endpoint
            >>> update_future = training_client.update_dedicated_endpoint(
            ...     name="step-000001",
            ...     endpoint_id="your-endpoint-id",
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     hf_token=os.environ["HF_TOKEN"],
            ...     hf_org="your-hf-org",
            ... )
            >>> result = update_future.result()
            >>>
            >>> # Use the model ID for sampling
            >>> sampling_client.set_model(result.provider_model_id)
            >>> response = sampling_client.sample(prompt, sampling_params)
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
            "endpoint_id": endpoint_id,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "hf_org": hf_org,
            "provider_model_id": provider_model_id,
        }

        # Use _take_turn to ensure sequential HTTP dispatch
        async def _update_dedicated_endpoint_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/update_dedicated_endpoint",
                    request_data,
                    timeout=900,  # 15 minute timeout
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await self.holder.execute_with_retries(_send_request)

            # Parse and return
            success = result.get("success", False)
            provider_model_id = result.get("provider_model_id", "")
            if success:
                logger.info(
                    f"Dedicated endpoint updated: {provider_model_id} "
                    f"(HF: {result.get('upload_time', 0):.1f}s, "
                    f"Provider: {result.get('provider_upload_time', 0):.1f}s, "
                    f"Probe: {result.get('probe_time', 0):.1f}s)"
                )
            else:
                logger.error(
                    f"Failed to update dedicated endpoint: {result.get('error')}"
                )
            return types.UpdateDedicatedEndpointResponse(
                success=success,
                checkpoint_path=result.get("checkpoint_path", ""),
                hf_path=result.get("hf_path", ""),
                provider_model_id=provider_model_id,
                upload_time=result.get("upload_time", 0.0),
                provider_upload_time=result.get("provider_upload_time", 0.0),
                probe_time=result.get("probe_time", 0.0),
                error=result.get("error"),
            )

        # Schedule to event loop and return directly
        return wrap_future(
            self.holder.run_coroutine_threadsafe(_update_dedicated_endpoint_async())
        )

    def update_serverless_weights(
        self,
        name: str,
        provider_api_key: str,
        hf_token: str,
        provider_base_model: str,
        hf_repo_prefix: str = "xorl_client-adapter",
        private_repo: bool = True,
        hf_org: str = "xorl-ai",
        wait_for_completion: bool = True,
    ) -> APIFuture[types.UpdateServerlessWeightsResponse]:
        """Save weights and register with XoRL for serverless inference.

        This is the serverless alternative to save_weights_for_sampler().
        The adapter is uploaded to HuggingFace Hub and registered for serverless inference.

        Args:
            name: Checkpoint name (e.g., "step-001")
            provider_api_key: API key
            hf_token: HuggingFace token with write access
            provider_base_model: base model name
            hf_repo_prefix: Prefix for HuggingFace repository names
            private_repo: Whether to create private HuggingFace repository
            hf_org: HuggingFace organization to upload to (default: "xorl-ai")
            wait_for_completion: Whether to wait for upload to complete

        Returns:
            APIFuture[UpdateServerlessWeightsResponse] with provider_model_id for inference

        Example:
            >>> result = training_client.update_serverless_weights(
            ...     name="step-001",
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     hf_token=os.environ["HF_TOKEN"],
            ...     provider_base_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
            ... ).result()
            >>>
            >>> # Create sampling client with the returned model ID
            >>> from xorl_client.client.serverless_sampling_client import ServerlessSamplingClient
            >>> sampling_client = ServerlessSamplingClient(
            ...     provider_api_key=os.environ["PROVIDER_API_KEY"],
            ...     model_id=result.provider_model_id,
            ... )
        """
        # Get request ID for ordering
        request_id = self._get_request_id()

        request_data = {
            "model_id": self.model_id,
            "seq_id": request_id + 1,
            "name": name,
            "provider_api_key": provider_api_key,
            "hf_token": hf_token,
            "provider_base_model": provider_base_model,
            "hf_repo_prefix": hf_repo_prefix,
            "private_repo": private_repo,
            "hf_org": hf_org,
            "wait_for_completion": wait_for_completion,
        }

        # Use _take_turn to ensure sequential HTTP dispatch
        async def _update_serverless_weights_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/update_serverless_weights",
                    request_data,
                    timeout=1800.0,  # 30 minute timeout
                )

            # Execute inside _take_turn to ensure ordering
            async with self._take_turn(request_id):
                result = await self.holder.execute_with_retries(_send_request)

            # Parse and return
            provider_model_id = result.get("provider_model_id", "")
            hf_repo_url = result.get("hf_repo_url", "")
            checkpoint_path = result.get("checkpoint_path", "")
            status = result.get("status", "Complete")

            logger.info(
                f"Serverless weights updated: provider_model_id={provider_model_id}, "
                f"hf_repo={hf_repo_url}, status={status}"
            )
            return types.UpdateServerlessWeightsResponse(
                provider_model_id=provider_model_id,
                hf_repo_url=hf_repo_url,
                checkpoint_path=checkpoint_path,
                status=status,
            )

        # Schedule to event loop and return directly
        return wrap_future(
            self.holder.run_coroutine_threadsafe(_update_serverless_weights_async())
        )

    def list_checkpoints(self) -> APIFuture[types.CheckpointsListResponse]:
        """List all available checkpoints.

        Returns both training checkpoints (weights/) and sampler checkpoints (sampler_weights/).
        Checkpoints are sorted by creation time (newest first).

        Returns:
            APIFuture[CheckpointsListResponse] with list of checkpoints

        Example:
            >>> response = training_client.list_checkpoints().result()
            >>> for ckpt in response.checkpoints:
            ...     print(f"{ckpt.checkpoint_id}: {ckpt.checkpoint_type}")
        """
        request_data = {
            "model_id": self.model_id,
        }

        future = self.holder.post_async("/api/v1/list_checkpoints", request_data)

        def parse_response(result: Dict[str, Any]) -> types.CheckpointsListResponse:
            checkpoints = [
                types.Checkpoint.from_dict(c) for c in result.get("checkpoints", [])
            ]
            logger.info(f"Listed {len(checkpoints)} checkpoints")
            return types.CheckpointsListResponse(checkpoints=checkpoints)

        from concurrent.futures import Future

        parsed_future: Future[types.CheckpointsListResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def list_checkpoints_async(self) -> APIFuture[types.CheckpointsListResponse]:
        """Async version of list_checkpoints.

        Returns:
            APIFuture[CheckpointsListResponse] with list of checkpoints
        """
        return self.list_checkpoints()

    def delete_checkpoint(
        self,
        checkpoint_id: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete (e.g., 'weights/000' or 'sampler_weights/step-100')

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status

        Example:
            >>> response = training_client.delete_checkpoint("weights/000").result()
            >>> print(f"Deleted: {response.deleted_path}")
        """
        request_data = {
            "model_id": self.model_id,
            "checkpoint_id": checkpoint_id,
        }

        future = self.holder.post_async("/api/v1/delete_checkpoint", request_data)

        def parse_response(result: Dict[str, Any]) -> types.DeleteCheckpointResponse:
            success = result.get("success", False)
            deleted_path = result.get("deleted_path")
            error = result.get("error")
            if success:
                logger.info(f"Deleted checkpoint: {deleted_path}")
            else:
                logger.error(f"Failed to delete checkpoint: {error}")
            return types.DeleteCheckpointResponse(
                success=success,
                deleted_path=deleted_path,
                error=error,
            )

        from concurrent.futures import Future

        parsed_future: Future[types.DeleteCheckpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    async def delete_checkpoint_async(
        self,
        checkpoint_id: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Async version of delete_checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status
        """
        return self.delete_checkpoint(checkpoint_id)

    def delete_checkpoint_from_xorl_path(
        self,
        xorl_path: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Delete a checkpoint using its xorl:// path.

        This is a convenience method that parses the xorl:// path and calls delete_checkpoint.

        Args:
            xorl_path: Full xorl:// path (e.g., 'xorl://default/weights/000')

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status

        Example:
            >>> response = training_client.delete_checkpoint_from_xorl_path(
            ...     "xorl://default/weights/000"
            ... ).result()
            >>> print(f"Deleted: {response.deleted_path}")
        """
        parsed = types.ParsedCheckpointXoRLPath.from_xorl_path(xorl_path)
        return self.delete_checkpoint(parsed.checkpoint_id)

    async def delete_checkpoint_from_xorl_path_async(
        self,
        xorl_path: str,
    ) -> APIFuture[types.DeleteCheckpointResponse]:
        """Async version of delete_checkpoint_from_xorl_path.

        Args:
            xorl_path: Full xorl:// path

        Returns:
            APIFuture[DeleteCheckpointResponse] with success status
        """
        return self.delete_checkpoint_from_xorl_path(xorl_path)

    # =========================================================================
    # Weight Sync Methods (NCCL-based weight transfer to inference endpoints)
    # =========================================================================

    def add_inference_endpoint(
        self,
        host: str,
        port: int,
        world_size: int = 1,
        sync_weights: bool = False,
        master_address: Optional[str] = None,
        master_port: int = 29600,
        group_name: str = "weight_sync_group",
        buffer_size_mb: int = 1024,
        pool: str = "default",
    ) -> APIFuture[types.AddInferenceEndpointResponse]:
        """Register an SGLang inference endpoint for NCCL weight sync.

        This endpoint:
        1. Checks if the endpoint is healthy via /health
        2. Fetches server info via /server_info to get model and LoRA configuration
        3. Validates that LoRA is enabled on the inference endpoint
        4. If sync_weights is True, syncs weights to the new endpoint

        Args:
            host: Hostname or IP address of the inference endpoint
            port: Port number of the SGLang server
            world_size: Number of TP workers at this endpoint (default: 1)
            sync_weights: Whether to auto-sync weights after adding (default: False)
            master_address: Training server hostname for NCCL rendezvous (auto-detected if None)
            master_port: Port for NCCL rendezvous (default: 29600)
            group_name: NCCL process group name (default: weight_sync_group)
            buffer_size_mb: Transfer bucket size in MB (default: 1024)
            pool: Endpoint pool tag (default: "default"). Syncs can be restricted
                to pools via sync_weights_to_inference(pools=[...]) — e.g. register
                dedicated eval endpoints with pool="eval" so the per-step training
                sync leaves them on frozen weights.

        Returns:
            APIFuture[AddInferenceEndpointResponse] with endpoint info and sync status

        Example:
            >>> # Register SGLang endpoint on node 13
            >>> result = training_client.add_inference_endpoint(
            ...     host="research-common-13",
            ...     port=30000,
            ...     world_size=8,
            ...     sync_weights=True,  # Auto-sync weights
            ... ).result()
            >>> print(f"Added: {result.endpoint.host}:{result.endpoint.port}")
            >>> if result.weights_synced:
            ...     print(f"Weights synced: {result.sync_message}")
        """
        request_data = {
            "host": host,
            "port": port,
            "world_size": world_size,
            "sync_weights": sync_weights,
            "master_address": master_address or "",
            "master_port": master_port,
            "group_name": group_name,
            "buffer_size_mb": buffer_size_mb,
            "pool": pool,
        }

        future = self.holder.post_async("/add_inference_endpoint", request_data)

        def parse_response(
            result: Dict[str, Any],
        ) -> types.AddInferenceEndpointResponse:
            response = types.AddInferenceEndpointResponse.from_dict(result)
            if response.success:
                logger.info(
                    f"Added inference endpoint: {host}:{port} (world_size={world_size})"
                )
                if response.weights_synced:
                    logger.info(f"Weights synced: {response.sync_message}")
            else:
                logger.error(f"Failed to add inference endpoint: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.AddInferenceEndpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def sync_inference_weights(
        self,
        master_address: Optional[str] = None,
        master_port: int = 29600,
        group_name: str = "weight_sync_group",
        buffer_size_mb: int = 1024,
        quantization: Optional[Dict[str, Any]] = None,
    ) -> APIFuture[types.SyncWeightsResponse]:
        """Sync weights to all registered inference endpoints via NCCL.

        This triggers the training server to broadcast current model weights
        to all registered inference endpoints using NCCL for high-performance transfer.

        The process:
        1. Training server extracts model weights (state_dict)
        2. Initializes NCCL process group with inference endpoints
        3. Transfers weights in buckets via NCCL broadcast (to avoid OOM)
        4. Cleans up process groups

        Args:
            master_address: Training server hostname for NCCL rendezvous (auto-detected if None)
            master_port: Port for NCCL rendezvous (default: 29600)
            group_name: NCCL process group name (default: weight_sync_group)
            buffer_size_mb: Transfer bucket size in MB (default: 1024)

        Returns:
            APIFuture[SyncWeightsResponse] with transfer stats (time, bytes, throughput)

        Example:
            >>> # Sync weights after training
            >>> result = training_client.sync_inference_weights(
            ...     master_port=29600,
            ...     buffer_size_mb=1024,
            ... ).result()
            >>> print(f"Transfer: {result.total_bytes/1e9:.2f} GB in {result.transfer_time:.2f}s")
            >>> print(f"Throughput: {result.throughput_gbps:.2f} GB/s")
        """
        request_data = {
            "master_address": master_address or "",
            "master_port": master_port,
            "group_name": group_name,
            "buffer_size_mb": buffer_size_mb,
        }
        quantization = (
            quantization if quantization is not None else _sync_quantization_from_env()
        )
        if quantization is not None:
            request_data["quantization"] = quantization

        # Use extended timeout for weight sync (can take minutes for large models)
        future = self.holder.post_async(
            "/sync_inference_weights",
            request_data,
            timeout=1800.0,  # 30 minute timeout
        )

        def parse_response(result: Dict[str, Any]) -> types.SyncWeightsResponse:
            response = types.SyncWeightsResponse.from_dict(result)
            if response.success:
                logger.info(
                    f"Weight sync complete: {response.total_bytes/1e9:.2f} GB in "
                    f"{response.transfer_time:.2f}s ({response.throughput_gbps:.2f} GB/s)"
                )
            else:
                logger.error(f"Weight sync failed: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.SyncWeightsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def list_inference_endpoints(
        self,
    ) -> APIFuture[types.ListInferenceEndpointsResponse]:
        """List all registered inference endpoints.

        Performs health checks on all endpoints and removes unhealthy ones.

        Returns:
            APIFuture[ListInferenceEndpointsResponse] with list of healthy endpoints

        Example:
            >>> response = training_client.list_inference_endpoints().result()
            >>> for ep in response.endpoints:
            ...     print(f"{ep.host}:{ep.port} (world_size={ep.world_size})")
        """
        future = self.holder.get_async("/list_inference_endpoints")

        def parse_response(
            result: Dict[str, Any],
        ) -> types.ListInferenceEndpointsResponse:
            response = types.ListInferenceEndpointsResponse.from_dict(result)
            logger.info(f"Listed {response.count} inference endpoint(s)")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.ListInferenceEndpointsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def remove_inference_endpoint(
        self,
        host: str,
        port: int,
    ) -> APIFuture[types.RemoveInferenceEndpointResponse]:
        """Remove an inference endpoint from the registry.

        Args:
            host: Hostname or IP address of the inference endpoint
            port: Port number of the inference endpoint

        Returns:
            APIFuture[RemoveInferenceEndpointResponse] with success status

        Example:
            >>> response = training_client.remove_inference_endpoint(
            ...     host="research-common-13",
            ...     port=30000,
            ... ).result()
            >>> print(f"Removed: {response.message}")
        """
        request_data = {
            "host": host,
            "port": port,
        }

        future = self.holder.post_async("/remove_inference_endpoint", request_data)

        def parse_response(
            result: Dict[str, Any],
        ) -> types.RemoveInferenceEndpointResponse:
            response = types.RemoveInferenceEndpointResponse.from_dict(result)
            if response.success:
                logger.info(f"Removed inference endpoint: {host}:{port}")
            else:
                logger.warning(f"Failed to remove endpoint: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.RemoveInferenceEndpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def connect_inference_endpoint(
        self,
        master_address: Optional[str] = None,
        master_port: int = 29600,
        group_name: str = "weight_sync_group",
        buffer_size_mb: int = 256,
    ) -> APIFuture[types.ConnectEndpointResponse]:
        """Establish a persistent NCCL connection to inference endpoints.

        Call this once at the start of an RL training loop to avoid
        connection overhead during frequent weight syncs. The connection
        persists until disconnect_inference_endpoint is called.

        Prerequisites:
        - At least one inference endpoint must be registered via add_inference_endpoint

        Args:
            master_address: Training server hostname for NCCL rendezvous (auto-detected if None)
            master_port: Port for NCCL rendezvous (default: 29600)
            group_name: NCCL process group name (default: weight_sync_group)
            buffer_size_mb: Transfer buffer size in MB (default: 256)

        Returns:
            APIFuture[ConnectEndpointResponse] with connection status

        Example:
            >>> # Establish persistent connection at RL loop start
            >>> result = training_client.connect_inference_endpoint(
            ...     master_port=29741,
            ... ).result()
            >>> print(f"Connected to {len(result.connected_endpoints)} endpoint(s)")
            >>>
            >>> # Now sync_inference_weights will be faster
            >>> for step in range(num_steps):
            ...     training_client.sync_inference_weights().result()
            ...     # ... training loop ...
            >>>
            >>> # Disconnect at the end
            >>> training_client.disconnect_inference_endpoint().result()
        """
        request_data = {
            "master_address": master_address or "",
            "master_port": master_port,
            "group_name": group_name,
            "buffer_size_mb": buffer_size_mb,
        }

        # Use extended timeout for connection establishment
        future = self.holder.post_async(
            "/connect_inference_endpoint",
            request_data,
            timeout=300.0,  # 5 minute timeout
        )

        def parse_response(result: Dict[str, Any]) -> types.ConnectEndpointResponse:
            response = types.ConnectEndpointResponse.from_dict(result)
            if response.success:
                logger.info(
                    f"Connected to {len(response.connected_endpoints)} inference endpoint(s)"
                )
            else:
                logger.error(f"Connection failed: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.ConnectEndpointResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def disconnect_inference_endpoint(self) -> APIFuture[types.DisconnectResponse]:
        """Disconnect from inference endpoints and clean up NCCL process groups.

        Call this at the end of an RL training loop or when changing
        inference endpoints. This cleans up the persistent connection
        established by connect_inference_endpoint.

        Returns:
            APIFuture[DisconnectResponse] with disconnection status

        Example:
            >>> # At the end of RL training loop
            >>> result = training_client.disconnect_inference_endpoint().result()
            >>> print(f"Disconnected: {result.message}")
        """
        request_data = {}

        future = self.holder.post_async(
            "/disconnect_inference_endpoint",
            request_data,
            timeout=60.0,  # 1 minute timeout
        )

        def parse_response(result: Dict[str, Any]) -> types.DisconnectResponse:
            response = types.DisconnectResponse.from_dict(result)
            if response.success:
                logger.info(f"Disconnected from inference endpoints")
            else:
                logger.warning(f"Disconnect failed: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.DisconnectResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    def sync_weights_to_inference(
        self,
        sync_method: str = "nccl_ep_scatter",
        master_address: Optional[str] = None,
        timeout: float = 1800.0,
        quantization: Optional[Dict[str, Any]] = None,
        pools: Optional[List[str]] = None,
        group_name: Optional[str] = None,
    ) -> APIFuture[types.SyncWeightsResponse]:
        """Sync current model weights to connected inference endpoint.

        This is a convenience method for RL training that calls
        sync_inference_weights with the specified sync method.
        Assumes connect_inference_endpoint was called first.

        Args:
            sync_method: Transfer method - "nccl_ep_scatter" (default, multi-rank parallel),
                        "nccl" (single-rank), or "rdma_direct" (RDMA push)
            master_address: Optional trainer address for rendezvous. If omitted,
                XORL_WEIGHT_SYNC_MASTER_ADDRESS is used when set.
            timeout: HTTP request timeout in seconds.
            pools: Restrict the sync to endpoints registered with a pool tag in
                this list (None = all endpoints, backward compatible). E.g.
                pools=["default"] for the per-step training-sampler sync,
                pools=["eval"] to refresh a dedicated eval pool at eval steps.
            group_name: Optional NCCL/P2P process-group name override. Use a
                distinct name per pool (e.g. "weight_sync_group_eval") so pool
                syncs don't collide on the cached transfer group.
            quantization: Optional transport quantization configuration.

        Returns:
            APIFuture[SyncWeightsResponse] with transfer stats

        Example:
            >>> # In RL training loop
            >>> training_client.sync_weights_to_inference().result()
            >>> # Sample from inference server with updated weights
            >>> responses = sampling_client.sample(prompts, params)
        """
        request_data = {
            "sync_method": sync_method,
            "timeout_s": timeout,
        }
        if pools is not None:
            request_data["pools"] = pools
        if group_name is not None:
            request_data["group_name"] = group_name
        master_address = master_address or os.environ.get(
            "XORL_WEIGHT_SYNC_MASTER_ADDRESS"
        )
        if master_address:
            request_data["master_address"] = master_address
        quantization = (
            quantization if quantization is not None else _sync_quantization_from_env()
        )
        if quantization is not None:
            request_data["quantization"] = quantization

        # Use extended timeout for weight sync
        future = self.holder.post_async(
            "/sync_inference_weights",
            request_data,
            timeout=timeout,
        )

        def parse_response(result: Dict[str, Any]) -> types.SyncWeightsResponse:
            response = types.SyncWeightsResponse.from_dict(result)
            if response.success:
                logger.info(
                    f"Weight sync complete: {response.total_bytes/1e9:.2f} GB in "
                    f"{response.transfer_time:.2f}s ({response.throughput_gbps:.2f} GB/s)"
                )
            else:
                logger.error(f"Weight sync failed: {response.message}")
            return response

        from concurrent.futures import Future

        parsed_future: Future[types.SyncWeightsResponse] = Future()

        def callback(f: Future):
            try:
                result = f.result()
                parsed_future.set_result(parse_response(result))
            except Exception as e:
                parsed_future.set_exception(e)

        future.add_done_callback(callback)
        return wrap_future(parsed_future)

    # =========================================================================
    # Model Lifecycle Methods
    # =========================================================================

    def unload(self) -> APIFuture[types.UnloadModelResponse]:
        """Unload the model and release all resources (two-phase pattern).

        This ends the training session and releases:
        - Server-side state (registered model IDs, session tracking)
        - Worker resources (training adapter, GPU memory)

        After calling unload(), this TrainingClient should not be used anymore.

        Returns:
            APIFuture[UnloadModelResponse] with success status

        Example:
            >>> # End training session
            >>> unload_future = training_client.unload()
            >>> result = unload_future.result()
            >>> print(f"Model unloaded: {result.success}")
        """
        import time

        # Get request ID for ordering
        request_id = self._get_request_id()

        # Async function that handles both phases (like Tinker)
        async def _unload_async():
            start_time = time.time()

            # Phase 1: Submit request inside _take_turn to ensure ordering
            async with self._take_turn(request_id):

                async def _send_request():
                    return await self.holder.models.unload(self.model_id)

                result = await self.holder.execute_with_retries(_send_request)

            # Parse UntypedAPIFuture response
            untyped_future = types.UntypedAPIFuture.from_dict(result)

            # Phase 2: Create _APIFuture and await for the result
            return await _APIFuture(
                model_cls=types.UnloadModelResponse,
                holder=self.holder,
                untyped_future=untyped_future,
                request_start_time=start_time,
                request_type="UnloadModel",
            )

        # Schedule to event loop and return (like Tinker)
        return wrap_future(self.holder.run_coroutine_threadsafe(_unload_async()))

    async def unload_async(self) -> APIFuture[types.UnloadModelResponse]:
        """Async version of unload.

        Unload the model and release all resources.

        Returns:
            APIFuture[UnloadModelResponse] with success status

        Example:
            >>> unload_future = await training_client.unload_async()
            >>> result = await unload_future
            >>> print(f"Model unloaded: {result.success}")
        """
        return self.unload()

    def kill_session(
        self,
        save_checkpoint: bool = True,
    ) -> APIFuture[types.KillSessionResponse]:
        """Kill the active full-weights training session.

        In full-weights training mode (enable_lora=False), the server operates in
        single-tenant mode. This method kills the active session, resets optimizer
        state and step counters, and allows starting a new session.

        For LoRA mode, this is a no-op since multi-tenancy is supported.

        Args:
            save_checkpoint: Whether to save checkpoint before killing (default: True)

        Returns:
            APIFuture[KillSessionResponse] with status and optional checkpoint path

        Example:
            >>> # Kill current session to start fresh
            >>> response = training_client.kill_session(save_checkpoint=True).result()
            >>> if response.success:
            ...     print(f"Session killed: {response.message}")
            ...     if response.checkpoint_path:
            ...         print(f"Checkpoint saved to: {response.checkpoint_path}")
        """
        request_data = {
            "model_id": self.model_id,
            "save_checkpoint": save_checkpoint,
        }

        async def _kill_session_async():
            async def _send_request():
                return await self.holder.post(
                    "/api/v1/kill_session",
                    request_data,
                    timeout=120.0,
                )

            result = await self.holder.execute_with_retries(_send_request)
            return types.KillSessionResponse.from_dict(result)

        return wrap_future(self.holder.run_coroutine_threadsafe(_kill_session_async()))

    async def kill_session_async(
        self,
        save_checkpoint: bool = True,
    ) -> APIFuture[types.KillSessionResponse]:
        """Async version of kill_session.

        Kill the active full-weights training session.

        Args:
            save_checkpoint: Whether to save checkpoint before killing (default: True)

        Returns:
            APIFuture[KillSessionResponse] with status and optional checkpoint path

        Example:
            >>> response = await training_client.kill_session_async(save_checkpoint=True)
            >>> result = await response
            >>> print(f"Session killed: {result.message}")
        """
        return self.kill_session(save_checkpoint=save_checkpoint)
