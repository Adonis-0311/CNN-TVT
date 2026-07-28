"""Dataset isolation and component-identity checks with vehicular-neutral keys."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class DataQAError(ValueError):
    """Raised when an evidence dataset violates a frozen QA invariant."""


def assert_unique_sample_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise DataQAError("every row must have a non-empty sample_id")
        if sample_id in seen:
            raise DataQAError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def assert_group_disjoint(
    rows: Iterable[Mapping[str, Any]],
    *,
    split_field: str = "split",
    group_fields: Sequence[str] = ("source_id",),
) -> None:
    if not group_fields:
        raise DataQAError("at least one group field is required")
    ownership: dict[tuple[Any, ...], str] = {}
    for row in rows:
        split = row.get(split_field)
        if not isinstance(split, str) or not split:
            raise DataQAError(f"missing {split_field}")
        group = tuple(row.get(field) for field in group_fields)
        if any(value is None or value == "" for value in group):
            raise DataQAError(f"missing group identity field in {group_fields}")
        previous = ownership.setdefault(group, split)
        if previous != split:
            raise DataQAError(
                f"group {group!r} crosses splits {previous!r} and {split!r}"
            )


def assert_exact_cell_counts(
    rows: Iterable[Mapping[str, Any]],
    *,
    factors: Sequence[str],
    expected: Mapping[tuple[Any, ...], int],
) -> None:
    observed: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = tuple(row.get(field) for field in factors)
        observed[key] = observed.get(key, 0) + 1
    if observed != dict(expected):
        missing = sorted(set(expected) - set(observed), key=repr)
        extra = sorted(set(observed) - set(expected), key=repr)
        wrong = {
            key: (expected[key], observed[key])
            for key in expected.keys() & observed.keys()
            if expected[key] != observed[key]
        }
        raise DataQAError(
            f"factorial cell mismatch; missing={missing}, extra={extra}, wrong={wrong}"
        )


def payload_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}".encode("ascii")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


@dataclass(frozen=True)
class ComponentQA:
    relative_reconstruction_error: float
    measured_snr_db: float
    measured_sir_db: float


def verify_component_identity(
    observed: np.ndarray,
    desired: np.ndarray,
    interference: np.ndarray,
    noise: np.ndarray,
    *,
    relative_tolerance: float = 1e-6,
) -> ComponentQA:
    arrays = [np.asarray(item) for item in (observed, desired, interference, noise)]
    if len({item.shape for item in arrays}) != 1:
        raise DataQAError("component arrays must have identical shapes")
    if any(not np.all(np.isfinite(item)) for item in arrays):
        raise DataQAError("component arrays must be finite")
    reconstructed = arrays[1] + arrays[2] + arrays[3]
    denominator = max(float(np.linalg.norm(arrays[0])), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(arrays[0] - reconstructed) / denominator)
    if relative_error > relative_tolerance:
        raise DataQAError(
            f"component identity failed: {relative_error:.3e} > {relative_tolerance:.3e}"
        )

    def power(value: np.ndarray) -> float:
        return float(np.mean(np.abs(value) ** 2))

    desired_power = power(arrays[1])
    noise_power = power(arrays[3])
    interference_power = power(arrays[2])
    tiny = np.finfo(float).tiny
    snr_db = 10.0 * np.log10(max(desired_power, tiny) / max(noise_power, tiny))
    sir_db = 10.0 * np.log10(
        max(desired_power, tiny) / max(interference_power, tiny)
    )
    return ComponentQA(relative_error, float(snr_db), float(sir_db))
