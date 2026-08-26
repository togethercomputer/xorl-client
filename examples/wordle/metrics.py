"""Backend-neutral token alignment and K3 aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LogprobPair:
    sampled: Sequence[float]
    trainer: Sequence[float]


class StructuralAlignmentError(RuntimeError):
    """A trainer result cannot be paired losslessly with submitted datums."""


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def numeric_metrics(values: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in dict(values or {}).items():
        parsed = number(value)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def tensor_values(value: Any) -> list[float]:
    if value is None:
        return []
    for name in ("to_numpy", "numpy"):
        fn = getattr(value, name, None)
        if callable(fn):
            return np.asarray(fn(), dtype=np.float64).reshape(-1).tolist()
    if hasattr(value, "data"):
        value = value.data
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _length_metrics(prefix: str, lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {
            f"{prefix}_tokens": 0.0,
            f"{prefix}_length_mean": 0.0,
            f"{prefix}_length_min": 0.0,
            f"{prefix}_length_max": 0.0,
        }
    values = [int(value) for value in lengths]
    return {
        f"{prefix}_tokens": float(sum(values)),
        f"{prefix}_length_mean": float(sum(values)) / len(values),
        f"{prefix}_length_min": float(min(values)),
        f"{prefix}_length_max": float(max(values)),
    }


def compute_k3_metrics(
    pairs: Iterable[LogprobPair],
    *,
    submitted_datums: int,
    returned_rows: int,
    missing_datums: int,
    mismatched_datums: int,
    trainer_returned_tokens: int,
    prompt_lengths: Sequence[int],
    response_lengths: Sequence[int],
) -> dict[str, float]:
    """Aggregate K3 tokenwise across every backend request in one step."""

    deltas: list[np.ndarray] = []
    aligned = 0
    for pair in pairs:
        sampled = np.asarray(pair.sampled, dtype=np.float64).reshape(-1)
        trainer = np.asarray(pair.trainer, dtype=np.float64).reshape(-1)
        if sampled.shape != trainer.shape:
            raise ValueError(
                "internal aligned logprob pair has unequal lengths: "
                f"{sampled.size} != {trainer.size}"
            )
        if not np.isfinite(sampled).all() or not np.isfinite(trainer).all():
            raise ValueError("sampled and trainer logprobs must be finite")
        aligned += 1
        if sampled.size:
            deltas.append(trainer - sampled)

    sampled_tokens = sum(int(value) for value in response_lengths)
    valid_tokens = sum(int(value.size) for value in deltas)
    metrics = {
        "submitted_datums": float(submitted_datums),
        "returned_trainer_rows": float(returned_rows),
        "aligned_datums": float(aligned),
        "missing_datums": float(missing_datums),
        "mismatched_datums": float(mismatched_datums),
        "sampled_response_tokens": float(sampled_tokens),
        "trainer_returned_tokens": float(trainer_returned_tokens),
        "k3_valid_tokens": float(valid_tokens),
        "k3_coverage": float(valid_tokens) / max(float(sampled_tokens), 1.0),
        **_length_metrics("prompt", prompt_lengths),
        **_length_metrics("retained_response", response_lengths),
    }
    if not deltas:
        return metrics

    delta = np.concatenate(deltas)
    ratio = np.exp(delta)
    k3 = ratio - delta - 1.0
    metrics.update(
        {
            "k3_mean": float(k3.mean()),
            "k3_max": float(k3.max()),
            "k3_p90": float(np.percentile(k3, 90)),
            "ratio_mean": float(ratio.mean()),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "logratio_mean": float(delta.mean()),
            "abs_logratio_mean": float(np.abs(delta).mean()),
            "abs_logratio_max": float(np.abs(delta).max()),
        }
    )
    return metrics


def require_complete_alignment(metrics: dict[str, float]) -> None:
    """Reject structural row identity/length failures before optimization."""

    submitted = int(metrics.get("submitted_datums", 0.0))
    returned = int(metrics.get("returned_trainer_rows", 0.0))
    aligned = int(metrics.get("aligned_datums", 0.0))
    missing = int(metrics.get("missing_datums", 0.0))
    mismatched = int(metrics.get("mismatched_datums", 0.0))
    if returned != submitted or aligned != submitted or missing or mismatched:
        raise StructuralAlignmentError(
            "trainer result alignment is structurally invalid: "
            f"submitted={submitted} returned={returned} aligned={aligned} "
            f"missing={missing} mismatched={mismatched}"
        )


def finite_metric_present(
    records: Iterable[dict[str, float]], names: Sequence[str]
) -> bool:
    values = [
        float(value)
        for record in records
        for key, value in record.items()
        if any(name in key.lower() for name in names)
    ]
    return bool(values) and all(math.isfinite(value) for value in values)
