"""
Helpers for combining chunked forward-backward results.

When forward_backward requests are split into chunks (due to size limits),
this module provides functions to combine the results back into a single
ForwardBackwardOutput.

Based on Tinker's chunked_fwdbwd_helpers.py implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set, cast

import numpy as np

from xorl_client.types import ForwardBackwardOutput, LossFnOutput

logger = logging.getLogger(__name__)

Metrics = Dict[str, float]

# Chunking constants
MAX_CHUNK_LEN = 1024  # Maximum number of data items per chunk
MAX_CHUNK_BYTES_COUNT = 5_000_000  # Maximum bytes per chunk (5MB)


def combine_fwd_bwd_output_results(
    results: Sequence[ForwardBackwardOutput],
) -> ForwardBackwardOutput:
    """Combine multiple ForwardBackwardOutput results into one.

    This is used when a large forward_backward request is split into chunks.
    Each chunk produces its own ForwardBackwardOutput, and this function
    combines them back into a single result.

    Args:
        results: Sequence of ForwardBackwardOutput from each chunk

    Returns:
        Combined ForwardBackwardOutput with:
        - loss_fn_outputs: Concatenated from all chunks
        - metrics: Aggregated using reduction rules (mean, sum, min, max, etc.)
        - loss_fn_output_type: Taken from first result

    Example:
        >>> # Results from 3 chunks
        >>> chunk_results = [
        ...     ForwardBackwardOutput(loss_fn_outputs=[...], metrics={...}),
        ...     ForwardBackwardOutput(loss_fn_outputs=[...], metrics={...}),
        ...     ForwardBackwardOutput(loss_fn_outputs=[...], metrics={...}),
        ... ]
        >>> combined = combine_fwd_bwd_output_results(chunk_results)
    """
    if not results:
        return ForwardBackwardOutput(
            loss_fn_output_type="",
            metrics={},
            loss_fn_outputs=[],
        )

    combined_metrics = _metrics_reduction(results)
    combined_outputs = _combine_loss_fn_outputs(results)

    return ForwardBackwardOutput(
        loss_fn_output_type=results[0].loss_fn_output_type,
        metrics=combined_metrics,
        loss_fn_outputs=combined_outputs,
    )


def _combine_loss_fn_outputs(
    results: Sequence[ForwardBackwardOutput],
) -> List[LossFnOutput]:
    """Concatenate loss_fn_outputs from all results.

    Args:
        results: Sequence of ForwardBackwardOutput

    Returns:
        Flattened list of all LossFnOutput
    """
    return [output for result in results for output in result.loss_fn_outputs]


def _order_insensitive_hash(xs: Sequence[Set[Any]] | Sequence[float]) -> int:
    """Combine hash values in an order-insensitive way.

    Args:
        xs: Either a sequence of sets (original data) or a sequence of
            already-computed hash values

    Returns:
        Combined hash value
    """
    # If we have sets, flatten and hash them (original behavior)
    if xs and isinstance(xs[0], set):
        return hash(tuple(sorted([y for x in xs for y in cast(Set[Any], x)])))

    # If we have already-computed hash values, combine them deterministically
    # Sort them to ensure order-insensitive combination
    return hash(tuple(sorted(int(cast(float, x)) for x in xs)))


def _mean(xs: Sequence[float | int], weights: Sequence[float] | None = None) -> float:
    """Calculate weighted or unweighted mean."""
    if weights is None:
        return float(np.mean(xs))
    return float(np.average(xs, weights=weights))


def _sum(xs: Sequence[float | int]) -> float:
    """Sum all values."""
    return float(np.sum(xs))


def _min(xs: Sequence[float | int]) -> float:
    """Find minimum value."""
    return float(np.min(xs))


def _max(xs: Sequence[float | int]) -> float:
    """Find maximum value."""
    return float(np.max(xs))


def _slack(xs: Sequence[float | int], weights: Sequence[float] | None = None) -> float:
    """Calculate slack (max - mean)."""
    if weights is None:
        return float(np.max(xs) - np.mean(xs))
    return float(np.max(xs) - np.average(xs, weights=weights))


def _unique(xs: Sequence[float | int]) -> Sequence[float | int]:
    """Identity function for unique metrics.

    A unique metric can't actually fold. But in order to work around the fact
    that it's a str:float dict, we just insert unique keys with some suffix
    for each unique value. It's a hack, that's for sure.

    This is a dummy identity function just for documentation and consistency.
    """
    return xs


# Map of reduction type names to reduction functions
REDUCE_MAP = {
    "mean": _mean,
    "sum": _sum,
    "min": _min,
    "max": _max,
    "slack": _slack,
    "hash_unordered": _order_insensitive_hash,
    "unique": _unique,
}


def _metrics_reduction(results: Sequence[ForwardBackwardOutput]) -> Metrics:
    """Reduce metrics from all chunks using reduction rules.

    Every metric must indicate a reduction_type in its name, for example "mfu:mean".
    Metrics are weighted by the number of loss_fn_outputs (data points) each chunk processed.

    Supported reduction types:
    - mean: Weighted average
    - sum: Sum of all values
    - min: Minimum value
    - max: Maximum value
    - slack: max - mean
    - hash_unordered: Order-insensitive hash combination
    - unique: Keep first value, create suffixed entries for others

    Args:
        results: Sequence of ForwardBackwardOutput from each chunk

    Returns:
        Aggregated metrics dictionary
    """
    if not results:
        return {}

    keys = results[0].metrics.keys()

    # Weights are the number of data points each chunk processed
    weights = [len(m.loss_fn_outputs) for m in results]

    res: Metrics = {}
    for key in keys:
        # Split metric name from reduction type (e.g., "mfu:mean")
        if ":" not in key:
            logger.debug(f"Metric {key} missing reduction type, skipping")
            continue

        name, reduction = key.rsplit(":", 1)

        if reduction not in REDUCE_MAP:
            # Can happen when a new reduction type is added
            logger.debug(
                f"Invalid {reduction=} for metric {name=}. "
                f"Expecting one of {list(REDUCE_MAP.keys())}"
            )
            continue

        # Check that all results have this metric
        if not all(key in m.metrics for m in results):
            continue

        reduce_fn = REDUCE_MAP[reduction]
        values = [m.metrics[key] for m in results]

        if reduction in ["mean", "slack"]:
            # These reductions are weighted
            res[key] = reduce_fn(values, weights)
        elif reduction == "unique":
            # For unique, keep first and add suffixed entries for others
            res[key] = values[0]
            res.update({f"{key}_{i + 1}": v for i, v in enumerate(values[1:])})
        else:
            # sum, min, max, hash_unordered
            res[key] = reduce_fn(values)

    return res


def estimate_datum_bytes(datum) -> int:
    """Estimate the serialized size of a Datum in bytes.

    Used for chunking to stay under MAX_CHUNK_BYTES_COUNT.

    Args:
        datum: A Datum instance

    Returns:
        Estimated size in bytes
    """
    # Simple estimate based on input_ids length
    # Each token ID is roughly 4 bytes when serialized
    total_bytes = 0

    # Check model_input for input_ids
    if hasattr(datum, "model_input"):
        model_input = datum.model_input
        if hasattr(model_input, "chunks"):
            for chunk in model_input.chunks:
                if hasattr(chunk, "input_ids"):
                    total_bytes += len(chunk.input_ids) * 4

    # Check loss_fn_inputs
    if hasattr(datum, "loss_fn_inputs"):
        loss_fn_inputs = datum.loss_fn_inputs
        if isinstance(loss_fn_inputs, dict):
            for key, value in loss_fn_inputs.items():
                if hasattr(value, "__len__"):
                    total_bytes += len(value) * 4

    # Minimum estimate
    return max(total_bytes, 100)
