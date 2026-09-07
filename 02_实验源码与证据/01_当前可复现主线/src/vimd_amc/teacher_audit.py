"""Deterministic per-cell audits for simulator-ledger mask teachers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np
import torch
from torch import nn


FORMAL_TVT_EVIDENCE_DESIGNATION = "headline_formal_tvt_evidence"
FORMAL_TVT_V2_EVIDENCE_DESIGNATION = "headline_formal_tvt_evidence_v2"
FORMAL_TVT_EVIDENCE_DESIGNATIONS = frozenset(
    {
        FORMAL_TVT_EVIDENCE_DESIGNATION,
        FORMAL_TVT_V2_EVIDENCE_DESIGNATION,
    }
)
OCCUPANCY_GAIN_MECHANISM_PROTOCOL: dict[str, Any] = {
    "hypothesis_id": "H_mech_occupancy_gain",
    "unit": "jammer_family",
    "required_families": [
        "tone",
        "multitone",
        "chirp",
        "sweep",
        "partial_band",
        "comb",
    ],
    "occupancy_definition": (
        "per-source fraction of periodic-Hann complex-STFT cells in the single "
        "inference view (view1) whose jammer power is at least 0.01 times that "
        "source view's maximum jammer-cell power (-20 dB support); family value "
        "is the arithmetic mean over sources"
    ),
    "gain_definition": (
        "for each jammer family, compute paired A5-minus-A0 macro-F1 on that "
        "family's inference-view source subset within each algorithm seed, "
        "then take the arithmetic mean across algorithm seeds"
    ),
    "outcome": "paired A5 minus A0 macro-F1 percentage-point gain by family",
    "direction": (
        "gain is non-increasing as spectral occupancy increases; primary "
        "rank alternative is Spearman rho < 0"
    ),
    "tie_policy": "average ranks for both occupancy and gain",
    "primary_test": (
        "one-sided negative Spearman exact permutation over all 6! gain-label "
        "assignments"
    ),
    "alpha": 0.05,
    "supportive_order_check": (
        "zero strict-occupancy inversions under 1e-12 gain tolerance"
    ),
    "selection_or_tuning_use": False,
}


@dataclass
class _ExactCellAccumulator:
    view_count: int = 0
    chunks: list[np.ndarray] = field(default_factory=list)

    def add(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float32).reshape(-1)
        if flattened.size == 0:
            return
        if not np.isfinite(flattened).all():
            raise ValueError("teacher target contains a non-finite cell")
        if flattened.min() < -1e-6 or flattened.max() > 1.0 + 1e-6:
            raise ValueError("teacher target contains a cell outside [0, 1]")
        self.view_count += 1
        self.chunks.append(flattened)

    def summarize(self) -> dict[str, float | int | str]:
        if not self.chunks:
            raise ValueError("cannot summarize an empty teacher-cell group")
        values = np.concatenate(self.chunks)
        return {
            "source_view_count": int(self.view_count),
            "cell_count": int(values.size),
            "mean": float(np.mean(values, dtype=np.float64)),
            "p90": float(np.quantile(values, 0.90, method="linear")),
            "nonzero_cell_fraction": float(np.mean(values > 0.0)),
            "nonzero_definition": "M_s > 0 exactly in float32 teacher output",
            "quantile_method": "numpy linear exact pooled-cell quantile",
        }


@dataclass
class _ScalarAccumulator:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("non-finite scalar in teacher audit")
        self.values.append(float(value))

    def summarize(self) -> dict[str, float | int]:
        if not self.values:
            raise ValueError("cannot summarize an empty scalar group")
        return {
            "source_view_count": len(self.values),
            "mean": float(np.mean(self.values, dtype=np.float64)),
            "p90": float(
                np.quantile(
                    np.asarray(self.values, dtype=np.float64),
                    0.90,
                    method="linear",
                )
            ),
        }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman_average_rank(x: np.ndarray, y: np.ndarray) -> float:
    ranked_x = _average_ranks(x)
    ranked_y = _average_ranks(y)
    if np.ptp(ranked_x) == 0.0 or np.ptp(ranked_y) == 0.0:
        raise ValueError("Spearman statistic is undefined for a constant vector")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def occupancy_gain_mechanism_test(
    family_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the frozen family-level occupancy/gain mechanism hypothesis."""

    required = set(OCCUPANCY_GAIN_MECHANISM_PROTOCOL["required_families"])
    rows_by_family: dict[str, Mapping[str, Any]] = {}
    for row in family_rows:
        family = str(row.get("family", ""))
        if not family or family in rows_by_family:
            raise ValueError("family rows require unique nonempty family names")
        rows_by_family[family] = row
    if set(rows_by_family) != required:
        raise ValueError(
            "family rows must contain exactly the frozen mechanism families"
        )
    families = sorted(required)
    occupancy = np.asarray(
        [float(rows_by_family[name]["occupancy"]) for name in families],
        dtype=np.float64,
    )
    gain = np.asarray(
        [
            (
                float(rows_by_family[name]["gain_pp"])
                if "gain_pp" in rows_by_family[name]
                else 100.0
                * (
                    float(rows_by_family[name]["a5_macro_f1"])
                    - float(rows_by_family[name]["a0_macro_f1"])
                )
            )
            for name in families
        ],
        dtype=np.float64,
    )
    if not np.isfinite(occupancy).all() or not np.isfinite(gain).all():
        raise ValueError("occupancy and gain must be finite")
    if np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise ValueError("occupancy must lie in [0, 1]")
    observed = _spearman_average_rank(occupancy, gain)
    permutation_statistics = np.asarray(
        [
            _spearman_average_rank(occupancy, gain[list(permutation)])
            for permutation in permutations(range(len(gain)))
        ],
        dtype=np.float64,
    )
    p_value = float(np.mean(permutation_statistics <= observed + 1e-12))
    ordered = sorted(
        zip(occupancy.tolist(), families, gain.tolist()),
        key=lambda row: (row[0], row[1]),
    )
    inversions = [
        {
            "lower_occupancy_family": left[1],
            "higher_occupancy_family": right[1],
            "lower_occupancy": left[0],
            "higher_occupancy": right[0],
            "lower_occupancy_gain_pp": left[2],
            "higher_occupancy_gain_pp": right[2],
        }
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
        if right[0] > left[0] and right[2] > left[2] + 1e-12
    ]
    alpha = float(OCCUPANCY_GAIN_MECHANISM_PROTOCOL["alpha"])
    return {
        "status": "evaluated",
        "protocol": OCCUPANCY_GAIN_MECHANISM_PROTOCOL,
        "family_rows": [
            {
                "family": family,
                "occupancy": float(occupancy[index]),
                "gain_pp": float(gain[index]),
            }
            for index, family in enumerate(families)
        ],
        "spearman_rho": observed,
        "one_sided_negative_exact_permutation_p": p_value,
        "permutation_count": int(len(permutation_statistics)),
        "negative_rank_test_passed": bool(observed < 0.0 and p_value <= alpha),
        "nonincreasing_order_holds": not inversions,
        "strict_occupancy_inversion_count": len(inversions),
        "strict_occupancy_inversions": inversions,
        "confirmatory_mechanism_passed": bool(
            observed < 0.0 and p_value <= alpha and not inversions
        ),
    }


def _records_for_dataset(
    dataset: object,
    manifest: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    records = manifest.get("records")
    split = str(getattr(dataset, "split", ""))
    if isinstance(records, Mapping):
        records = records.get(split)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(
            "dataset manifest does not expose ordered per-source view records"
        )
    if len(records) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "manifest record count does not match the selected dataset split"
        )
    return records


def _jammer_family(view_record: Mapping[str, Any]) -> str:
    direct = view_record.get("jammer_name")
    if isinstance(direct, str) and direct:
        return direct
    components = view_record.get("jammer_components")
    if (
        isinstance(components, Sequence)
        and not isinstance(components, (str, bytes))
        and len(components) == 1
        and isinstance(components[0], str)
    ):
        return components[0]
    raise ValueError("view record has no unambiguous jammer-family label")


def _teacher_spec(teacher: nn.Module) -> dict[str, object]:
    spec = getattr(teacher, "spec", None)
    if spec is not None and callable(getattr(spec, "to_dict", None)):
        return dict(spec.to_dict())
    return {"mode": teacher.__class__.__name__}


@torch.no_grad()
def hard_split_teacher_cell_statistics(
    teachers: Mapping[str, nn.Module],
    dataset: object,
    *,
    batch_size: int = 32,
    maximum_sources: int | None = None,
    operating_points_db: Sequence[tuple[float, float]] = ((-10.0, -15.0),),
) -> dict[str, Any]:
    """Pool target-route teacher cells by jammer family on one hard split.

    The routine is deterministic: sources and both views are traversed in
    manifest order, there is no sampling or model state, and P90 is computed
    from every selected float32 target cell.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not teachers:
        raise ValueError("at least one teacher is required")
    split = str(getattr(dataset, "split", ""))
    if split != "hard_interference" and split != "hard":
        raise ValueError(
            "teacher-cell audit requires the hard_interference/hard split"
        )
    manifest_method = getattr(dataset, "manifest", None)
    if not callable(manifest_method):
        raise ValueError("dataset must expose a manifest() method")
    manifest = manifest_method()
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be a mapping")
    records = _records_for_dataset(dataset, manifest)
    source_count = len(dataset)  # type: ignore[arg-type]
    if maximum_sources is not None:
        if maximum_sources <= 0:
            raise ValueError("maximum_sources must be positive when provided")
        source_count = min(source_count, int(maximum_sources))

    accumulators: dict[
        str,
        dict[str, _ExactCellAccumulator],
    ] = {
        teacher_name: defaultdict(_ExactCellAccumulator)
        for teacher_name in teachers
    }
    selected_operating_point_views: dict[str, int] = {
        f"snr={snr:g},sir={sir:g}": 0
        for snr, sir in operating_points_db
    }
    family_view_counts: dict[str, int] = defaultdict(int)
    occupancy_accumulators: dict[str, _ScalarAccumulator] = defaultdict(
        _ScalarAccumulator
    )

    for start in range(0, source_count, batch_size):
        stop = min(source_count, start + batch_size)
        component_batches: dict[str, list[torch.Tensor]] = {
            "clean": [],
            "jammer": [],
            "unexplained": [],
        }
        batch_metadata: list[tuple[str, float, float]] = []
        for index in range(start, stop):
            item = dataset[index]  # type: ignore[index]
            record = records[index]
            views = record.get("views")
            if (
                not isinstance(views, Sequence)
                or isinstance(views, (str, bytes))
                or len(views) != 2
            ):
                raise ValueError("every manifest source record must have two views")
            for view_index, view_name in enumerate(("view1", "view2")):
                view = item[view_name]
                view_record = views[view_index]
                if not isinstance(view_record, Mapping):
                    raise ValueError("manifest view record must be a mapping")
                family = _jammer_family(view_record)
                family_view_counts[family] += 1
                snr_db = float(view["snr_db"])
                sir_db = float(view["sir_db"])
                batch_metadata.append((family, snr_db, sir_db))
                for component in component_batches:
                    component_batches[component].append(view[component].float())
        stacked = {
            name: torch.stack(values, dim=0)
            for name, values in component_batches.items()
        }
        first_teacher = next(iter(teachers.values()))
        front_end = getattr(first_teacher, "front_end", None)
        if not isinstance(front_end, nn.Module):
            raise ValueError("teacher must expose its physical STFT front_end")
        jammer_power = front_end(stacked["jammer"]).abs().square()
        jammer_peak = jammer_power.amax(dim=(1, 2), keepdim=True)
        occupied = jammer_power >= 0.01 * jammer_peak
        occupancy = occupied.float().mean(dim=(1, 2)).cpu().numpy()
        for row, (family, _snr_db, _sir_db) in enumerate(batch_metadata):
            occupancy_accumulators[family].add(float(occupancy[row]))
        for teacher_name, teacher in teachers.items():
            teacher.eval()
            masks = teacher(
                stacked["clean"],
                stacked["jammer"],
                stacked["unexplained"],
            )
            if masks.ndim != 4 or masks.shape[1] != 3:
                raise ValueError(
                    f"{teacher_name} did not return [view, 3, frequency, time]"
                )
            target = masks[:, 0].detach().cpu().numpy()
            for row, (family, snr_db, sir_db) in enumerate(batch_metadata):
                values = target[row]
                accumulators[teacher_name]["all_hard"].add(values)
                accumulators[teacher_name][f"family::{family}"].add(values)
                for point_snr, point_sir in operating_points_db:
                    if (
                        abs(snr_db - point_snr) <= 1e-6
                        and abs(sir_db - point_sir) <= 1e-6
                    ):
                        point = f"snr={point_snr:g},sir={point_sir:g}"
                        accumulators[teacher_name][f"point::{point}"].add(values)
                        accumulators[teacher_name][
                            f"point_family::{point}::{family}"
                        ].add(values)
                        if teacher_name == next(iter(teachers)):
                            selected_operating_point_views[point] += 1

    designation = str(
        manifest.get("configuration", {}).get(
            "evidence_designation",
            manifest.get("evidence_designation", "unspecified"),
        )
    )
    formal_tvt_evidence_eligible = (
        designation in FORMAL_TVT_EVIDENCE_DESIGNATIONS
    )
    result_teachers: dict[str, Any] = {}
    for teacher_name, groups in accumulators.items():
        family_records = {
            key.removeprefix("family::"): accumulator.summarize()
            for key, accumulator in sorted(groups.items())
            if key.startswith("family::")
        }
        points: dict[str, Any] = {}
        for point in selected_operating_point_views:
            overall_key = f"point::{point}"
            if overall_key not in groups:
                points[point] = {
                    "status": "unavailable",
                    "reason": "no view at the exact requested SNR/SIR point",
                    "by_jammer_family": {},
                }
                continue
            prefix = f"point_family::{point}::"
            points[point] = {
                "status": "available",
                "overall": groups[overall_key].summarize(),
                "by_jammer_family": {
                    key.removeprefix(prefix): accumulator.summarize()
                    for key, accumulator in sorted(groups.items())
                    if key.startswith(prefix)
                },
            }
        result_teachers[teacher_name] = {
            "teacher_class": teachers[teacher_name].__class__.__name__,
            "teacher_spec": _teacher_spec(teachers[teacher_name]),
            "target_route_formula": (
                "[q_s-q_j]_+"
                if _teacher_spec(teachers[teacher_name]).get("mode")
                == "closed_form_margin"
                else (
                    "q_s"
                    if _teacher_spec(teachers[teacher_name]).get("mode")
                    == "proportional_power"
                    else (
                        "(q_s+q_j)[p_s/e_s - p_j/e_j]_+/"
                        "(p_s/e_s+p_j/e_j)"
                    )
                )
            ),
            "overall": groups["all_hard"].summarize(),
            "by_jammer_family": family_records,
            "operating_points_db": points,
        }
    return {
        "schema_version": 1,
        "artifact": "hard_split_teacher_target_cell_statistics",
        "split": split,
        "source_count": int(source_count),
        "source_view_count": int(2 * source_count),
        "family_source_view_counts": dict(sorted(family_view_counts.items())),
        "cache_digest": manifest.get("cache_digest"),
        "cache_evidence_designation": designation,
        "formal_tvt_evidence_eligible": formal_tvt_evidence_eligible,
        "claim_scope": (
            "formal-cache descriptive teacher evidence"
            if formal_tvt_evidence_eligible
            else "development-only diagnostic; not formal TVT result evidence"
        ),
        "stft": {
            "window": "periodic Hann",
            "center": False,
            "normalized": True,
            "onesided": False,
        },
        "operating_point_view_counts": selected_operating_point_views,
        "jammer_spectral_occupancy": {
            "support_threshold_relative_to_view_peak": 0.01,
            "support_threshold_db": -20.0,
            "by_jammer_family": {
                family: accumulator.summarize()
                for family, accumulator in sorted(
                    occupancy_accumulators.items()
                )
            },
        },
        "occupancy_gain_mechanism": {
            "status": "pending",
            "reason": (
                "formal paired A5/A0 macro-F1 gains by jammer family have not "
                "been supplied to occupancy_gain_mechanism_test"
            ),
            "protocol": OCCUPANCY_GAIN_MECHANISM_PROTOCOL,
        },
        "teachers": result_teachers,
    }


__all__ = [
    "FORMAL_TVT_EVIDENCE_DESIGNATION",
    "FORMAL_TVT_V2_EVIDENCE_DESIGNATION",
    "FORMAL_TVT_EVIDENCE_DESIGNATIONS",
    "OCCUPANCY_GAIN_MECHANISM_PROTOCOL",
    "hard_split_teacher_cell_statistics",
    "occupancy_gain_mechanism_test",
]
