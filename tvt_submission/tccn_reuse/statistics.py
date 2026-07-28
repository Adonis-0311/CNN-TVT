"""Preregistered statistical-protocol checks for paired vehicular AMC evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class StatisticsProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StatisticalProtocol:
    evidence_tier: str
    seeds: tuple[int, ...]
    bootstrap_cluster: str = "source_id"
    calibration_split: str = "calibration"
    test_split: str = "test"
    family_declared_before_test: bool = True

    def validate(self) -> None:
        minimum = {"screening": 3, "headline": 5}.get(self.evidence_tier)
        if minimum is None:
            raise StatisticsProtocolError("evidence_tier must be screening or headline")
        if len(self.seeds) < minimum or len(self.seeds) != len(set(self.seeds)):
            raise StatisticsProtocolError(
                f"{self.evidence_tier} evidence requires at least {minimum} unique seeds"
            )
        if not self.bootstrap_cluster:
            raise StatisticsProtocolError("bootstrap cluster must be explicit")
        if self.calibration_split == self.test_split:
            raise StatisticsProtocolError("calibration and test must be isolated")
        if not self.family_declared_before_test:
            raise StatisticsProtocolError("hypothesis family was not preregistered")


def validate_paired_records(
    sample_ids_a: Sequence[str],
    sample_ids_b: Sequence[str],
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    clusters_a: Sequence[str],
    clusters_b: Sequence[str],
) -> None:
    arrays = [
        np.asarray(value)
        for value in (
            sample_ids_a,
            sample_ids_b,
            labels_a,
            labels_b,
            clusters_a,
            clusters_b,
        )
    ]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise StatisticsProtocolError("paired records must be non-empty and equal length")
    names = ("sample order", "labels", "clusters")
    for name, left, right in zip(names, arrays[::2], arrays[1::2]):
        if not np.array_equal(left, right):
            raise StatisticsProtocolError(f"paired {name} differ")


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise StatisticsProtocolError("p_values must be a non-empty vector")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise StatisticsProtocolError("p_values must be finite values in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (values.size - np.arange(values.size)) * values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
