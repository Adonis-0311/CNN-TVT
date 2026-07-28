"""Append condition slices and physical-teacher route probes to a control run.

The original HOC ridge solution is deterministically reconstructed from the
same immutable cache because the v1 control artifact did not persist row-level
probabilities.  Its aggregate metrics must reproduce the original artifact
before any slice is emitted.  No iterative retraining, model selection, or
hyperparameter search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_learnability_controls as base  # noqa: E402
from vimd_amc.data.controls import CleanOracleInputDataset  # noqa: E402
from vimd_amc.metrics import PredictionBundle, classification_metrics  # noqa: E402
from vimd_amc.models.classical import (  # noqa: E402
    ClassicalHOCyclostationaryFeatures,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.oracle_probes import PhysicalTeacherRouteProbe  # noqa: E402
from vimd_amc.standards import CachedPairedAMCDataset  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_existing_checksums(run_root: Path) -> dict[str, Any]:
    path = run_root / "checksums.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in payload["files"].items():
        actual = _sha256_file(run_root / relative)
        if actual != expected:
            raise RuntimeError(
                f"existing artifact checksum mismatch for {relative}: "
                f"{actual} != {expected}"
            )
    return payload


def _metadata_rows(
    dataset: CachedPairedAMCDataset,
    *,
    batch_size: int,
) -> pd.DataFrame:
    records = dataset.manifest()["records"][dataset.split]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        for view_index in (0, 1):
            for index in indices:
                record = records[index]
                view = record["views"][view_index]
                rows.append(
                    {
                        "index": index,
                        "view": view_index + 1,
                        "source_id": int(record["source_sequence_id"]),
                        "label": int(dataset[index]["label"]),
                        "jammer_family": str(view["jammer_name"]),
                        "sir_db": float(view["measured_sir_db"]),
                        "overlap_profile": str(
                            view["overlap_profile_requested"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _slice_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    sir_db: np.ndarray,
    selected: np.ndarray,
    *,
    classes: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "worst_recall": np.nan,
            "supported_class_count": 0,
            "class_coverage": 0.0,
            "minimum_class_count": 0,
            "sample_count": 0,
            "unique_source_count": 0,
        }
    bundle = PredictionBundle(
        probabilities=probabilities[indices],
        labels=labels[indices],
        source_ids=source_ids[indices],
        snr_db=np.zeros(len(indices), dtype=np.float64),
        sir_db=sir_db[indices],
    )
    metrics = classification_metrics(bundle, classes)
    counts = np.bincount(labels[indices], minlength=classes)
    supported_counts = counts[counts > 0]
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "worst_recall": metrics["worst_recall"],
        "supported_class_count": metrics["supported_class_count"],
        "class_coverage": metrics["class_coverage"],
        "minimum_class_count": int(supported_counts.min()),
        "sample_count": len(indices),
        "unique_source_count": int(np.unique(source_ids[indices]).size),
    }


def _condition_slice_rows(
    *,
    split: str,
    metadata: pd.DataFrame,
    mixture_probabilities: np.ndarray,
    clean_probabilities: np.ndarray,
    classes: int,
) -> list[dict[str, Any]]:
    labels = metadata["label"].to_numpy(dtype=np.int64)
    source_ids = metadata["source_id"].to_numpy(dtype=np.int64)
    sir_db = metadata["sir_db"].to_numpy(dtype=np.float64)
    definitions: list[tuple[str, str, np.ndarray]] = []
    for value in sorted(metadata["jammer_family"].unique()):
        definitions.append(
            (
                "jammer_family",
                str(value),
                metadata["jammer_family"].to_numpy() == value,
            )
        )
    sir_definitions = (
        ("sir_le_-5_db", sir_db <= -5.0),
        ("sir_gt_-5_le_0_db", (sir_db > -5.0) & (sir_db <= 0.0)),
        ("sir_gt_0_le_5_db", (sir_db > 0.0) & (sir_db <= 5.0)),
        ("sir_gt_5_db", sir_db > 5.0),
    )
    definitions.extend(
        ("sir_bin", name, selected)
        for name, selected in sir_definitions
    )
    for value in sorted(metadata["overlap_profile"].unique()):
        definitions.append(
            (
                "overlap_profile",
                str(value),
                metadata["overlap_profile"].to_numpy() == value,
            )
        )

    rows: list[dict[str, Any]] = []
    for dimension, value, selected in definitions:
        mixture = _slice_metrics(
            mixture_probabilities,
            labels,
            source_ids,
            sir_db,
            selected,
            classes=classes,
        )
        clean = _slice_metrics(
            clean_probabilities,
            labels,
            source_ids,
            sir_db,
            selected,
            classes=classes,
        )
        sample_count = int(mixture["sample_count"])
        rows.append(
            {
                "split": split,
                "slice_dimension": dimension,
                "slice_value": value,
                "is_cochannel_focus": (
                    dimension == "jammer_family" and value == "cochannel"
                ),
                "sample_count": sample_count,
                "unique_source_count": mixture["unique_source_count"],
                "minimum_class_count": mixture["minimum_class_count"],
                "supported_class_count": mixture["supported_class_count"],
                "class_coverage": mixture["class_coverage"],
                "mixture_accuracy": mixture["accuracy"],
                "mixture_macro_f1": mixture["macro_f1"],
                "mixture_worst_recall": mixture["worst_recall"],
                "clean_oracle_accuracy": clean["accuracy"],
                "clean_oracle_macro_f1": clean["macro_f1"],
                "clean_oracle_worst_recall": clean["worst_recall"],
                "clean_minus_mixture_accuracy": (
                    clean["accuracy"] - mixture["accuracy"]
                    if sample_count
                    else np.nan
                ),
                "clean_minus_mixture_macro_f1": (
                    clean["macro_f1"] - mixture["macro_f1"]
                    if sample_count
                    else np.nan
                ),
                "small_slice_warning": (
                    sample_count < 100
                    or int(mixture["minimum_class_count"]) < 5
                ),
                "paper_conclusion_allowed": False,
                "evidence_designation": base.EVIDENCE_DESIGNATION,
            }
        )
    return rows


def _route_feature_matrix(
    dataset: Dataset,
    probe: PhysicalTeacherRouteProbe,
    extractor: ClassicalHOCyclostationaryFeatures,
    *,
    route_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probe.eval()
    extractor.eval()
    with torch.no_grad():
        for batch in loader:
            def paired(field: str) -> torch.Tensor:
                return torch.cat(
                    (batch["view1"][field], batch["view2"][field]),
                    dim=0,
                ).to(device)

            mixture = paired("x")
            output = probe(
                mixture,
                paired("clean"),
                paired("jammer"),
                paired("unexplained"),
            )
            matrices.append(
                extractor(output[route_name]).cpu().numpy().astype(np.float64)
            )
            labels.append(
                torch.cat((batch["label"], batch["label"]))
                .numpy()
                .astype(np.int64)
            )
    return np.concatenate(matrices), np.concatenate(labels)


def _reference_metric(
    original: dict[str, Any],
    model_name: str,
    split: str,
    key: str,
) -> float:
    for result in original["results"]:
        if result["model"] == model_name:
            return float(result["metrics"][split][key])
    raise KeyError((model_name, split, key))


def append_audit(args: argparse.Namespace) -> Path:
    run_root = args.run_root.resolve()
    existing_checksums = _verify_existing_checksums(run_root)
    targets = (
        run_root / "condition_slices.csv",
        run_root / "route_oracle_metrics.csv",
        run_root / "hoc_ridge_predictions.npz",
        run_root / "control_audit.json",
    )
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing companion artifacts: "
            + ", ".join(collisions)
        )
    original = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if original.get("evidence_designation") != base.EVIDENCE_DESIGNATION:
        raise ValueError("run is not a diagnostic learnability-control artifact")

    protocol = original["protocol"]
    batch_size = int(protocol["batch_size"])
    ridge = float(protocol["ridge"])
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    cache_root = Path(original["cache"]["path"])
    classes = len(original["cache"]["modulations"])
    datasets = {
        split: CachedPairedAMCDataset(
            cache_root,
            split,
            verify_checksums=True,
        )
        for split in base.REQUIRED_SPLITS
    }
    try:
        oracle_datasets = {
            split: CleanOracleInputDataset(dataset)
            for split, dataset in datasets.items()
        }
        extractor = ClassicalHOCyclostationaryFeatures().to(device)
        mixture_features = {
            split: base._feature_matrix(
                dataset,
                extractor,
                batch_size=batch_size,
                device=device,
            )
            for split, dataset in datasets.items()
        }
        clean_features = {
            split: base._feature_matrix(
                dataset,
                extractor,
                batch_size=batch_size,
                device=device,
            )
            for split, dataset in oracle_datasets.items()
        }
        mixture_ridge = base._fit_ridge(
            *mixture_features["train"],
            classes=classes,
            ridge=ridge,
        )
        clean_ridge = base._fit_ridge(
            *clean_features["train"],
            classes=classes,
            ridge=ridge,
        )

        probabilities: dict[str, dict[str, np.ndarray]] = {
            "mixture": {},
            "clean_oracle": {},
        }
        reproduction_errors: list[float] = []
        slice_rows: list[dict[str, Any]] = []
        prediction_archive: dict[str, np.ndarray] = {}
        for split in ("validation", "heldout_channel"):
            probabilities["mixture"][split] = base._ridge_probabilities(
                mixture_ridge,
                mixture_features[split][0],
            )
            probabilities["clean_oracle"][split] = base._ridge_probabilities(
                clean_ridge,
                clean_features[split][0],
            )
            metadata = _metadata_rows(
                datasets[split],
                batch_size=batch_size,
            )
            labels, source_ids, _, sir_db = base._prediction_context(
                datasets[split],
                batch_size=batch_size,
            )
            if not np.array_equal(
                labels,
                metadata["label"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError("manifest/data-loader label ordering mismatch")
            if not np.array_equal(
                source_ids,
                metadata["source_id"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError("manifest/data-loader source ordering mismatch")
            for condition, model_name in (
                ("mixture", "hoc_cyclostationary_ridge_mixture"),
                (
                    "clean_oracle",
                    "hoc_cyclostationary_ridge_clean_oracle",
                ),
            ):
                metrics = base._metrics_from_probabilities(
                    probabilities[condition][split],
                    (
                        datasets[split]
                        if condition == "mixture"
                        else oracle_datasets[split]
                    ),
                    classes=classes,
                    batch_size=batch_size,
                )
                for key in ("accuracy", "macro_f1"):
                    reproduction_errors.append(
                        abs(
                            float(metrics[key])
                            - _reference_metric(
                                original,
                                model_name,
                                split,
                                key,
                            )
                        )
                    )
            slice_rows.extend(
                _condition_slice_rows(
                    split=split,
                    metadata=metadata,
                    mixture_probabilities=probabilities["mixture"][split],
                    clean_probabilities=probabilities["clean_oracle"][split],
                    classes=classes,
                )
            )
            prefix = split
            prediction_archive[f"{prefix}_labels"] = labels
            prediction_archive[f"{prefix}_source_ids"] = source_ids
            prediction_archive[f"{prefix}_sir_db"] = sir_db
            prediction_archive[f"{prefix}_jammer_family"] = metadata[
                "jammer_family"
            ].to_numpy(dtype=str)
            prediction_archive[f"{prefix}_overlap_profile"] = metadata[
                "overlap_profile"
            ].to_numpy(dtype=str)
            prediction_archive[f"{prefix}_mixture_probabilities"] = (
                probabilities["mixture"][split]
            )
            prediction_archive[f"{prefix}_clean_oracle_probabilities"] = (
                probabilities["clean_oracle"][split]
            )
        maximum_reproduction_error = max(reproduction_errors, default=0.0)
        if maximum_reproduction_error > 1e-12:
            raise RuntimeError(
                "reconstructed HOC control does not reproduce original metrics: "
                f"max error={maximum_reproduction_error}"
            )

        a0_result = next(
            result
            for result in original["results"]
            if result["model"] == "a0_backbone_clean_oracle"
        )
        model_config = ModelConfig(**a0_result["training"]["model_config"])
        probe = PhysicalTeacherRouteProbe(model_config).to(device)
        first = datasets["train"][0]["view1"]
        first_mixture = first["x"].unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed, coverage = probe.inverse_for_feature_probe(
                probe.front_end(first_mixture),
                length=first_mixture.shape[-1],
            )
        complex_original = torch.complex(
            first_mixture[:, 0],
            first_mixture[:, 1],
        )
        complex_reconstructed = torch.complex(
            reconstructed[:, 0],
            reconstructed[:, 1],
        )
        covered_error = (
            complex_reconstructed[:, coverage]
            - complex_original[:, coverage]
        ).abs()
        inverse_audit = {
            "method": (
                "explicit weighted overlap-add inverse of normalized, "
                "center-false periodic-Hann STFT"
            ),
            "n_fft": model_config.n_fft,
            "hop_length": model_config.hop_length,
            "sample_length": int(first_mixture.shape[-1]),
            "covered_sample_count": int(coverage.sum()),
            "uncovered_sample_indices": (
                torch.nonzero(~coverage, as_tuple=False).flatten().cpu().tolist()
            ),
            "coverage_fraction": float(coverage.float().mean()),
            "covered_round_trip_max_abs_error": float(covered_error.max()),
            "covered_round_trip_mean_abs_error": float(covered_error.mean()),
            "uncovered_boundary_policy": "deterministic_zero_fill",
        }

        route_rows: list[dict[str, Any]] = []
        for split in ("validation", "heldout_channel"):
            for condition, model_name in (
                ("mixture", "hoc_cyclostationary_ridge_mixture"),
                (
                    "clean_oracle",
                    "hoc_cyclostationary_ridge_clean_oracle",
                ),
            ):
                metrics = base._metrics_from_probabilities(
                    probabilities[condition][split],
                    (
                        datasets[split]
                        if condition == "mixture"
                        else oracle_datasets[split]
                    ),
                    classes=classes,
                    batch_size=batch_size,
                )
                route_rows.append(
                    {
                        "model": model_name,
                        "route": condition,
                        "split": split,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "worst_recall": metrics["worst_recall"],
                        "sample_count": metrics["sample_count"],
                        "oracle_component_access": condition == "clean_oracle",
                        "deployment_eligible": condition == "mixture",
                        "waveform_reconstruction_claimed": False,
                        "paper_conclusion_allowed": False,
                        "evidence_designation": base.EVIDENCE_DESIGNATION,
                    }
                )

        route_metrics_payload: dict[str, Any] = {}
        for route_name in probe.route_names:
            route_features = {
                split: _route_feature_matrix(
                    dataset,
                    probe,
                    extractor,
                    route_name=route_name,
                    batch_size=batch_size,
                    device=device,
                )
                for split, dataset in datasets.items()
            }
            route_ridge = base._fit_ridge(
                *route_features["train"],
                classes=classes,
                ridge=ridge,
            )
            route_metrics_payload[route_name] = {}
            for split in ("validation", "heldout_channel"):
                route_probabilities = base._ridge_probabilities(
                    route_ridge,
                    route_features[split][0],
                )
                metrics = base._metrics_from_probabilities(
                    route_probabilities,
                    datasets[split],
                    classes=classes,
                    batch_size=batch_size,
                )
                route_metrics_payload[route_name][split] = metrics
                route_rows.append(
                    {
                        "model": f"physical_teacher_{route_name}_hoc_ridge",
                        "route": route_name,
                        "split": split,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "worst_recall": metrics["worst_recall"],
                        "sample_count": metrics["sample_count"],
                        "oracle_component_access": True,
                        "deployment_eligible": False,
                        "waveform_reconstruction_claimed": False,
                        "paper_conclusion_allowed": False,
                        "evidence_designation": base.EVIDENCE_DESIGNATION,
                    }
                )
                prediction_archive[
                    f"{split}_{route_name}_probabilities"
                ] = route_probabilities

        condition_path = run_root / "condition_slices.csv"
        route_path = run_root / "route_oracle_metrics.csv"
        predictions_path = run_root / "hoc_ridge_predictions.npz"
        audit_path = run_root / "control_audit.json"
        pd.DataFrame(slice_rows).to_csv(condition_path, index=False)
        pd.DataFrame(route_rows).to_csv(route_path, index=False)
        np.savez_compressed(predictions_path, **prediction_archive)
        audit = {
            "schema_version": "learnability-control-companion-v1",
            "status": "complete",
            "evidence_designation": base.EVIDENCE_DESIGNATION,
            "headline_evidence_eligible": False,
            "paper_conclusion_allowed": False,
            "original_run": str(run_root / "run.json"),
            "original_hoc_solution_reconstruction": {
                "reason": (
                    "v1 did not persist row probabilities; recomputed the "
                    "declared deterministic closed-form ridge solution"
                ),
                "iterative_training_performed": False,
                "hyperparameter_search_performed": False,
                "ridge": ridge,
                "maximum_aggregate_metric_reproduction_error": (
                    maximum_reproduction_error
                ),
                "reproduction_tolerance": 1e-12,
                "passed": True,
            },
            "condition_slice_policy": {
                "dimensions": [
                    "jammer_family",
                    "SIR: <=-5, (-5,0], (0,5], >5 dB",
                    "requested overlap profile",
                ],
                "cochannel_reported_separately": True,
                "small_slice_rule": (
                    "view rows <100 or minimum represented-class count <5"
                ),
                "inference_performed": False,
                "warning": (
                    "descriptive small-sample diagnostic slices; no paper "
                    "conclusion or multiplicity-adjusted inference"
                ),
            },
            "physical_teacher_route_probe": {
                **probe.control_metadata(),
                "stft_inverse_audit": inverse_audit,
                "ridge": ridge,
                "metrics": route_metrics_payload,
            },
            "warnings": [
                "All physical-teacher routes require ground-truth components.",
                "Inverse STFT outputs are feature adapters, not reconstructed waveforms.",
                "The periodic-Hann center-false lattice leaves sample zero uncovered.",
                "All route and condition results are single-seed diagnostics.",
            ],
            "source_files": {
                Path(__file__).relative_to(ROOT).as_posix(): _sha256_file(
                    Path(__file__)
                ),
                "src/vimd_amc/models/oracle_probes.py": _sha256_file(
                    ROOT
                    / "src"
                    / "vimd_amc"
                    / "models"
                    / "oracle_probes.py"
                ),
            },
        }
        audit_path.write_text(
            json.dumps(
                audit,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        updated_checksums = {
            **existing_checksums,
            "scope": (
                "generated diagnostic artifacts excluding this checksum file"
            ),
            "files": {
                **existing_checksums["files"],
                "condition_slices.csv": _sha256_file(condition_path),
                "route_oracle_metrics.csv": _sha256_file(route_path),
                "hoc_ridge_predictions.npz": _sha256_file(predictions_path),
                "control_audit.json": _sha256_file(audit_path),
            },
        }
        (run_root / "checksums.json").write_text(
            json.dumps(
                updated_checksums,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
    return run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(append_audit(parse_args()))
