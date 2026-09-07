"""Derive and validate the fail-closed TVT v2 scientific release gate."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_formal_tvt_v2 import (  # noqa: E402
    runner,
    validate_learning_prerequisite,
)
from tvt_submission.formal_v2_contract import (  # noqa: E402
    DEFAULT_FREEZE,
    EXPECTED_OOD_AXES,
    EXPECTED_STRESS_AXES,
    load_contract,
)
from vimd_amc.metrics import (  # noqa: E402
    PredictionBundle,
    ablation_family_paired_bootstrap,
    headline_paired_bootstrap,
    ood_axis_calibrated_bootstrap,
)
from vimd_amc.teacher_audit import (  # noqa: E402
    OCCUPANCY_GAIN_MECHANISM_PROTOCOL,
    occupancy_gain_mechanism_test,
)
from vimd_amc.reproducibility import (  # noqa: E402
    TVT_V2_SOURCE_PROFILE,
    execution_environment_reasons,
    resolve_frozen_device,
    source_tree_audit_reasons,
)


GATE_SCHEMA = "vimd_amc.tvt.scientific_release_gate.v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bundle(
    run_root: Path,
    model: str,
    seed: int,
    regime: str,
    expected_cache_digest: str,
) -> PredictionBundle:
    path = (
        run_root
        / "models"
        / f"{model}_seed{seed}"
        / f"predictions_{regime}.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        cache_digest = str(np.asarray(archive["cache_digest"]).item())
        split = str(np.asarray(archive["split"]).item())
        if cache_digest != expected_cache_digest or split != regime:
            raise ValueError(
                f"prediction artifact binding mismatch: {path}"
            )
        bundle = PredictionBundle(
            probabilities=np.asarray(archive["probabilities"]),
            labels=np.asarray(archive["labels"]),
            source_ids=np.asarray(archive["source_ids"]),
            snr_db=np.asarray(archive["snr_db"]),
            sir_db=np.asarray(archive["sir_db"]),
            target_profile_index=np.asarray(
                archive["target_profile_index"]
            ),
        )
    bundle.validate()
    return bundle


def _fit_grid_reasons(
    run: dict[str, Any],
    run_root: Path,
    contract: dict[str, Any],
) -> list[str]:
    experiment = contract["experiment"]
    expected = {
        (model, int(seed))
        for model in experiment["models"]
        for seed in experiment["seeds"]
    }
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    reasons: list[str] = []
    results = run.get("results")
    if not isinstance(results, list):
        return ["result_grid_missing_or_malformed"]
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            reasons.append(f"result_{index}_not_object")
            continue
        try:
            key = (str(result["model"]), int(result["seed"]))
        except (KeyError, TypeError, ValueError):
            reasons.append(f"result_{index}_model_or_seed_malformed")
            continue
        if key in observed:
            reasons.append(f"duplicate_fit:{key[0]}/seed{key[1]}")
        observed[key] = result
    missing = sorted(expected.difference(observed))
    unexpected = sorted(set(observed).difference(expected))
    if missing:
        reasons.append(f"missing_fit_count:{len(missing)}")
    if unexpected:
        reasons.append(f"unexpected_fit_count:{len(unexpected)}")
    expected_regimes = set(
        contract["cache"]["expected_split_source_counts"]
    ).difference({"train"})
    for key in sorted(expected.intersection(observed)):
        result = observed[key]
        label = f"{key[0]}/seed{key[1]}"
        training = result.get("training")
        training = training if isinstance(training, dict) else {}
        checkpoint = training.get("checkpoint_selection")
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        eligible_checkpoint_count = checkpoint.get(
            "eligible_checkpoint_count", 0
        )
        try:
            eligible_checkpoint_count = int(eligible_checkpoint_count)
        except (TypeError, ValueError):
            eligible_checkpoint_count = 0
        if not (
            checkpoint.get("status")
            == "eligible_validation_checkpoint_selected"
            and checkpoint.get("selected_checkpoint_eligible") is True
            and checkpoint.get("fallback_used") is False
            and eligible_checkpoint_count >= 1
        ):
            reasons.append(f"checkpoint_ineligible:{label}")
        relative_checkpoint = result.get("checkpoint")
        if (
            not isinstance(relative_checkpoint, str)
            or not relative_checkpoint
            or not (run_root / relative_checkpoint).is_file()
        ):
            reasons.append(f"checkpoint_file_missing:{label}")
        regimes = result.get("regimes")
        if not isinstance(regimes, dict) or not expected_regimes.issubset(
            regimes
        ):
            reasons.append(f"evaluation_regimes_incomplete:{label}")
    return reasons


def _prediction_grid_reasons(
    *,
    run: dict[str, Any],
    run_root: Path,
    cache_root: Path,
    cache_digest: str,
    contract: dict[str, Any],
) -> list[str]:
    """Validate every model/seed/regime NPZ against immutable cache arrays."""

    experiment = contract["experiment"]
    split_counts = contract["cache"]["expected_split_source_counts"]
    regimes = tuple(split for split in split_counts if split != "train")
    observed = {
        (str(result.get("model")), int(result.get("seed"))): result
        for result in run.get("results", ())
        if isinstance(result, dict)
        and not isinstance(result.get("seed"), bool)
        and isinstance(result.get("seed"), int)
    }
    cache_arrays: dict[str, dict[str, np.ndarray]] = {}
    reasons: list[str] = []
    for regime in regimes:
        try:
            arrays = {
                name: np.load(
                    cache_root / regime / f"{name}.npy",
                    mmap_mode="r",
                )
                for name in (
                    "source_id",
                    "label",
                    "snr_db",
                    "sir_db",
                    "target_profile_index",
                )
            }
        except (OSError, ValueError) as error:
            reasons.append(f"cache_binding_arrays_invalid:{regime}:{error}")
            continue
        expected_count = int(split_counts[regime])
        if any(len(array) != expected_count for array in arrays.values()):
            reasons.append(f"cache_binding_array_count_drift:{regime}")
            continue
        cache_arrays[regime] = arrays
    for model in experiment["models"]:
        for seed in experiment["seeds"]:
            label = f"{model}/seed{seed}"
            result = observed.get((str(model), int(seed)))
            if result is None:
                continue
            result_path = (
                run_root
                / "models"
                / f"{model}_seed{seed}"
                / "result.json"
            )
            try:
                if _read_json(result_path) != result:
                    reasons.append(f"result_json_disagrees:{label}")
            except (OSError, ValueError, json.JSONDecodeError):
                reasons.append(f"result_json_missing_or_malformed:{label}")
            checkpoint = result.get("checkpoint")
            checkpoint_path: Path | None = None
            if isinstance(checkpoint, str) and checkpoint:
                candidate_path = (run_root / checkpoint).resolve()
                try:
                    candidate_path.relative_to(run_root.resolve())
                except ValueError:
                    reasons.append(f"checkpoint_outside_run_root:{label}")
                else:
                    checkpoint_path = candidate_path
            if (
                checkpoint_path is None
                or not checkpoint_path.is_file()
                or checkpoint_path.stat().st_size <= 0
            ):
                reasons.append(f"checkpoint_empty_or_missing:{label}")
            for regime in regimes:
                if regime not in cache_arrays:
                    continue
                artifact_label = f"{label}/{regime}"
                try:
                    bundle = _load_bundle(
                        run_root,
                        str(model),
                        int(seed),
                        regime,
                        cache_digest,
                    )
                except (OSError, KeyError, ValueError) as error:
                    reasons.append(
                        f"prediction_artifact_invalid:{artifact_label}:{error}"
                    )
                    continue
                arrays = cache_arrays[regime]
                expected_count = int(split_counts[regime])
                expected_profile = np.asarray(
                    arrays["target_profile_index"]
                )
                expected_snr = np.asarray(arrays["snr_db"])
                expected_sir = np.asarray(arrays["sir_db"])
                if expected_profile.ndim > 1:
                    expected_profile = expected_profile[:, 0]
                if expected_snr.ndim > 1:
                    expected_snr = expected_snr[:, 0]
                if expected_sir.ndim > 1:
                    expected_sir = expected_sir[:, 0]
                if (
                    len(bundle.labels) != expected_count
                    or not np.array_equal(
                        bundle.source_ids,
                        np.asarray(arrays["source_id"]),
                    )
                    or not np.array_equal(
                        bundle.labels,
                        np.asarray(arrays["label"]),
                    )
                    or bundle.target_profile_index is None
                    or not np.array_equal(
                        bundle.target_profile_index,
                        expected_profile,
                    )
                    or not np.array_equal(
                        bundle.snr_db,
                        expected_snr,
                        equal_nan=True,
                    )
                    or not np.array_equal(
                        bundle.sir_db,
                        expected_sir,
                        equal_nan=True,
                    )
                ):
                    reasons.append(
                        f"prediction_cache_binding_mismatch:{artifact_label}"
                    )
    return reasons


def _csv_bool(value: str, label: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} is not a canonical CSV boolean")


def _csv_seed_ids(value: str, label: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"{label} is not a serialized seed list") from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise ValueError(f"{label} is not a unique nonempty string list")
    return parsed


def _same_number(left: Any, right: Any) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(
            left_value,
            right_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )


def _frozen_precision_classification(
    *,
    point: float,
    simultaneous_low: float,
    simultaneous_high: float,
    sesoi: float,
) -> str:
    """Apply the preregistered primary-effect interpretation precedence."""

    values = (point, simultaneous_low, simultaneous_high, sesoi)
    if not all(math.isfinite(value) for value in values) or sesoi <= 0.0:
        raise ValueError("primary precision classification inputs are invalid")
    if simultaneous_low > simultaneous_high:
        raise ValueError("primary simultaneous interval is inverted")
    if simultaneous_high <= 0.0:
        return "no_positive_primary_gain_supported"
    if simultaneous_low > 0.0 and point >= sesoi:
        return (
            "statistically_positive_and_meets_preregistered_"
            "materiality_threshold"
        )
    if simultaneous_low > 0.0 and point < sesoi:
        return (
            "statistically_positive_but_below_preregistered_"
            "materiality_threshold"
        )
    if simultaneous_low <= 0.0 and simultaneous_high >= sesoi:
        return "inconclusive_at_prespecified_precision"
    return "materiality_threshold_not_supported"


def _confirmatory_family_test(
    *,
    run_root: Path,
    cache_digest: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the formal family from NPZ bundles and audit its CSV exactly."""

    experiment = contract["experiment"]
    specification = experiment["confirmatory_family"]
    seeds = [int(value) for value in experiment["seeds"]]
    contrasts = {
        str(record["contrast_id"]): (
            str(record["reference"]),
            str(record["candidate"]),
        )
        for record in specification["contrasts"]
    }
    required_models = {
        model
        for reference, candidate in contrasts.values()
        for model in (reference, candidate)
    }
    bundles = {
        model: {
            seed: _load_bundle(
                run_root,
                model,
                seed,
                str(specification["regime"]),
                cache_digest,
            )
            for seed in seeds
        }
        for model in sorted(required_models)
    }
    bootstrap_seed = runner.analysis_seed(
        int(experiment["statistics"]["bootstrap_seed"]),
        "ablation_family",
        str(specification["family_id"]),
        str(specification["regime"]),
    )
    derived = ablation_family_paired_bootstrap(
        bundles,
        contrasts,
        draws=int(experiment["statistics"]["bootstrap_draws"]),
        seed=bootstrap_seed,
        confidence_level=float(specification["confidence_level"]),
    )
    if derived["multiplicity_method"] != specification[
        "multiplicity_method"
    ]:
        raise ValueError("confirmatory multiplicity method drifted")
    critical_value = float(derived["simultaneous_critical_value"])
    if not math.isfinite(critical_value) or critical_value <= 0.0:
        raise ValueError(
            "confirmatory simultaneous critical value is not positive"
        )

    by_id = {
        str(record["contrast_id"]): record
        for record in specification["contrasts"]
    }
    threshold = float(
        specification["simultaneous_ci95_low_strictly_greater_than"]
    )
    gate_by_id = {
        contrast_id: (
            float(values["macro_f1_simultaneous_ci95_low"]) > threshold
        )
        for contrast_id, values in derived["contrasts"].items()
    }
    family_passed = all(gate_by_id.values())
    expected_rows: dict[str, dict[str, Any]] = {}
    for contrast_id, statistics in derived["contrasts"].items():
        record = by_id[contrast_id]
        expected_rows[contrast_id] = {
            "family_id": str(specification["family_id"]),
            "contrast_id": contrast_id,
            "reference": str(record["reference"]),
            "candidate": str(record["candidate"]),
            "intervention": str(record["intervention"]),
            "interpretation_scope": str(record["interpretation_scope"]),
            "regime": str(specification["regime"]),
            "metric": str(specification["metric"]),
            "direction": str(specification["direction"]),
            "cache_digest": cache_digest,
            **statistics,
            **{
                key: derived[key]
                for key in (
                    "bootstrap_draws",
                    "bootstrap_seed",
                    "algorithm_seed_count",
                    "algorithm_seed_ids",
                    "test_source_cluster_count",
                    "bootstrap_stratified_by_class",
                    "bootstrap_hierarchy",
                    "confidence_level",
                    "family_size",
                    "multiplicity_method",
                    "simultaneous_critical_value",
                    "quantile_method",
                    "source_alignment_verified",
                )
            },
            "gate_threshold": threshold,
            "gate_passed": gate_by_id[contrast_id],
            "family_gate_passed": family_passed,
        }

    artifact = run_root / "ablation_paired_statistics.csv"
    with artifact.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != runner.ABLATION_PAIRED_COLUMNS:
            raise ValueError(
                "confirmatory CSV columns differ from the runner contract"
            )
        rows = list(reader)
    expected_ids = list(contrasts)
    if [row.get("contrast_id") for row in rows] != expected_ids:
        raise ValueError("confirmatory CSV membership or ordering drifted")

    integer_fields = {
        "bootstrap_draws",
        "bootstrap_seed",
        "algorithm_seed_count",
        "test_source_cluster_count",
        "family_size",
    }
    boolean_fields = {
        "bootstrap_stratified_by_class",
        "source_alignment_verified",
        "gate_passed",
        "family_gate_passed",
    }
    numeric_fields = {
        "reference_macro_f1_mean",
        "candidate_macro_f1_mean",
        "macro_f1_difference",
        "macro_f1_marginal_ci95_low",
        "macro_f1_marginal_ci95_high",
        "macro_f1_simultaneous_ci95_low",
        "macro_f1_simultaneous_ci95_high",
        "confidence_level",
        "simultaneous_critical_value",
        "gate_threshold",
    }
    for row in rows:
        contrast_id = str(row["contrast_id"])
        expected = expected_rows[contrast_id]
        for field in runner.ABLATION_PAIRED_COLUMNS:
            observed_value = row[field]
            expected_value = expected[field]
            label = f"confirmatory CSV {contrast_id}/{field}"
            if field in integer_fields:
                try:
                    matches = int(observed_value) == int(expected_value)
                except (TypeError, ValueError):
                    matches = False
            elif field in boolean_fields:
                matches = _csv_bool(observed_value, label) is bool(
                    expected_value
                )
            elif field == "algorithm_seed_ids":
                matches = _csv_seed_ids(observed_value, label) == list(
                    expected_value
                )
            elif field in numeric_fields:
                matches = _same_number(observed_value, expected_value)
            else:
                matches = observed_value == str(expected_value)
            if not matches:
                raise ValueError(
                    f"{label} differs from deterministic NPZ re-derivation"
                )
    precision_design = contract["prospective_statistical_design"]
    sesoi_record = precision_design[
        "smallest_effect_size_of_interest"
    ]
    primary_contrast = str(sesoi_record["contrast_id"])
    sesoi = float(sesoi_record["absolute_macro_f1_gain"])
    material_classification = (
        "statistically_positive_and_meets_preregistered_"
        "materiality_threshold"
    )
    reported_contrasts: dict[str, dict[str, Any]] = {}
    for contrast_id in expected_ids:
        row = expected_rows[contrast_id]
        point = float(row["macro_f1_difference"])
        marginal_low = float(row["macro_f1_marginal_ci95_low"])
        marginal_high = float(row["macro_f1_marginal_ci95_high"])
        simultaneous_low = float(
            row["macro_f1_simultaneous_ci95_low"]
        )
        simultaneous_high = float(
            row["macro_f1_simultaneous_ci95_high"]
        )
        interval_width = simultaneous_high - simultaneous_low
        if not math.isfinite(interval_width) or interval_width < 0.0:
            raise ValueError(
                f"confirmatory interval width is invalid: {contrast_id}"
            )
        reported: dict[str, Any] = {
            "reference": row["reference"],
            "candidate": row["candidate"],
            "gain": point,
            "marginal_ci95_low": marginal_low,
            "marginal_ci95_high": marginal_high,
            "simultaneous_ci95_low": simultaneous_low,
            "simultaneous_ci95_high": simultaneous_high,
            "simultaneous_interval_width": interval_width,
            "precision_reporting_complete": True,
            "passed": bool(gate_by_id[contrast_id]),
        }
        if contrast_id == primary_contrast:
            classification = _frozen_precision_classification(
                point=point,
                simultaneous_low=simultaneous_low,
                simultaneous_high=simultaneous_high,
                sesoi=sesoi,
            )
            reported.update(
                {
                    "sesoi_absolute_macro_f1_gain": sesoi,
                    "simultaneous_interval_width_divided_by_sesoi": (
                        interval_width / sesoi
                    ),
                    "frozen_precision_classification": classification,
                    "material_primary_gain_wording_licensed": (
                        classification == material_classification
                    ),
                }
            )
        reported_contrasts[contrast_id] = reported
    if primary_contrast not in reported_contrasts:
        raise ValueError(
            "prospective SESOI contrast is absent from confirmatory family"
        )
    return {
        "family_id": str(specification["family_id"]),
        "contrast_ids": expected_ids,
        "passed": family_passed,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": _sha256(artifact),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": int(derived["bootstrap_draws"]),
        "simultaneous_critical_value": critical_value,
        "prospective_precision_reporting": {
            "complete": all(
                record["precision_reporting_complete"]
                for record in reported_contrasts.values()
            ),
            "primary_contrast_id": primary_contrast,
            "sesoi_absolute_macro_f1_gain": sesoi,
            "wide_or_inconclusive_interval_requires_design_extension": (
                precision_design["precision_reporting"][
                    "wide_or_inconclusive_interval_requires_design_extension"
                ]
            ),
            "wide_or_inconclusive_interval_must_be_reported": (
                precision_design["precision_reporting"][
                    "wide_or_inconclusive_interval_must_be_reported"
                ]
            ),
        },
        "contrasts": reported_contrasts,
    }


def _macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> float:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )
    represented = support > 0
    return float(f1[represented].mean())


def _spectral_support_occupancy(iq: np.ndarray) -> float:
    values = np.asarray(iq[0], dtype=np.float64) + 1j * np.asarray(
        iq[1],
        dtype=np.float64,
    )
    n_fft, hop = 64, 16
    window = np.hanning(n_fft + 1)[:-1]
    powers: list[np.ndarray] = []
    for start in range(0, len(values) - n_fft + 1, hop):
        spectrum = np.fft.fft(
            values[start : start + n_fft] * window
        )
        powers.append(np.abs(spectrum) ** 2)
    flat = np.concatenate(powers)
    maximum = float(flat.max())
    if maximum <= 0.0:
        return 0.0
    return float(np.mean(flat >= 0.01 * maximum))


def _view1_jammer_families_and_occupancy(
    records: list[dict[str, Any]],
    jammer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-level jammer metadata aligned with view1 inference."""

    if jammer.ndim < 2 or jammer.shape[0] != len(records):
        raise ValueError(
            "hard-interference jammer array is not source-aligned"
        )
    if jammer.shape[1] < 1:
        raise ValueError(
            "hard-interference jammer array does not contain view1"
        )
    family_names: list[str] = []
    for record in records:
        views = record.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(
                "hard-interference record does not contain view1 metadata"
            )
        view1 = views[0]
        if not isinstance(view1, dict):
            raise ValueError(
                "hard-interference view1 metadata is malformed"
            )
        family = view1.get("jammer_name")
        if not isinstance(family, str) or not family:
            raise ValueError(
                "hard-interference view1 lacks a jammer family"
            )
        family_names.append(family)
    occupancy_by_source = np.asarray(
        [
            _spectral_support_occupancy(jammer[index, 0])
            for index in range(len(jammer))
        ],
        dtype=np.float64,
    )
    return np.asarray(family_names, dtype=object), occupancy_by_source


def _jammer_occupancy_directional_test(
    *,
    run_root: Path,
    cache_root: Path,
    manifest: dict[str, Any],
    cache_digest: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    specification = contract["experiment"]["scientific_release_gates"][
        "jammer_occupancy_directional_test"
    ]
    regime = specification["regime"]
    family_pool = list(specification["family_pool"])
    seeds = contract["experiment"]["seeds"]
    baseline = specification["baseline_model"]
    candidate = specification["candidate_model"]
    bundles_a0 = {
        seed: _load_bundle(
            run_root,
            baseline,
            seed,
            regime,
            cache_digest,
        )
        for seed in seeds
    }
    bundles_a5 = {
        seed: _load_bundle(
            run_root,
            candidate,
            seed,
            regime,
            cache_digest,
        )
        for seed in seeds
    }
    source_ids = np.load(
        cache_root / regime / "source_id.npy",
        mmap_mode="r",
    )
    jammer = np.load(
        cache_root / regime / "jammer.npy",
        mmap_mode="r",
    )
    records = manifest.get("records", {}).get(regime)
    if not isinstance(records, list) or len(records) != len(source_ids):
        raise ValueError("hard-interference cache records are malformed")
    first_bundle = bundles_a0[seeds[0]]
    if not np.array_equal(first_bundle.source_ids, source_ids):
        raise ValueError(
            "hard-interference predictions are not in exact cache-source order"
        )
    family_names, occupancy_by_source = (
        _view1_jammer_families_and_occupancy(
            records,
            jammer,
        )
    )
    family_indices = {
        family: np.flatnonzero(family_names == family)
        for family in family_pool
    }
    if any(len(indices) == 0 for indices in family_indices.values()):
        raise ValueError(
            "hard-interference split lacks a preregistered jammer family"
        )
    labels = np.asarray(first_bundle.labels, dtype=np.int64)
    classes = int(first_bundle.probabilities.shape[1])
    a0_predictions = {
        seed: bundles_a0[seed].predictions for seed in seeds
    }
    a5_predictions = {
        seed: bundles_a5[seed].predictions for seed in seeds
    }

    def family_gain(
        family: str,
        selected_seeds: np.ndarray,
        selected_sources: np.ndarray,
    ) -> float:
        gains = []
        selected_labels = labels[selected_sources]
        for seed_index in selected_seeds:
            algorithm_seed = seeds[int(seed_index)]
            gains.append(
                _macro_f1(
                    selected_labels,
                    a5_predictions[algorithm_seed][selected_sources],
                    classes,
                )
                - _macro_f1(
                    selected_labels,
                    a0_predictions[algorithm_seed][selected_sources],
                    classes,
                )
            )
        return float(np.mean(gains))

    all_seed_indices = np.arange(len(seeds))
    family_occupancy = np.asarray(
        [
            occupancy_by_source[family_indices[family]].mean()
            for family in family_pool
        ],
        dtype=np.float64,
    )
    family_gains = np.asarray(
        [
            family_gain(
                family,
                all_seed_indices,
                family_indices[family],
            )
            for family in family_pool
        ],
        dtype=np.float64,
    )
    frozen_test = occupancy_gain_mechanism_test(
        [
            {
                "family": family,
                "occupancy": float(family_occupancy[index]),
                "gain_pp": float(100.0 * family_gains[index]),
            }
            for index, family in enumerate(family_pool)
        ]
    )
    observed_rho = float(frozen_test["spearman_rho"])
    one_sided_p = float(
        frozen_test["one_sided_negative_exact_permutation_p"]
    )
    draws = int(contract["experiment"]["statistics"]["bootstrap_draws"])
    bootstrap_seed = (
        int(contract["experiment"]["statistics"]["bootstrap_seed"])
        + 20_000
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_rho = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_seed_indices = rng.integers(
            0,
            len(seeds),
            size=len(seeds),
        )
        sampled_occupancy = []
        sampled_gains = []
        for family in family_pool:
            indices = family_indices[family]
            sampled = indices[
                rng.integers(0, len(indices), size=len(indices))
            ]
            sampled_occupancy.append(
                float(occupancy_by_source[sampled].mean())
            )
            sampled_gains.append(
                family_gain(
                    family,
                    selected_seed_indices,
                    sampled,
                )
            )
        statistic = float(
            spearmanr(sampled_occupancy, sampled_gains).statistic
        )
        bootstrap_rho[draw] = statistic
    finite_bootstrap = bootstrap_rho[np.isfinite(bootstrap_rho)]
    if len(finite_bootstrap) < 0.95 * draws:
        raise ValueError(
            "jammer-occupancy bootstrap is excessively degenerate"
        )
    ci_low, ci_high = np.quantile(
        finite_bootstrap,
        [0.025, 0.975],
    )
    direction_supported = bool(
        frozen_test["confirmatory_mechanism_passed"]
        and ci_high < 0.0
    )
    return {
        "regime": regime,
        "families": [
            {
                "family": family,
                "source_count": int(len(family_indices[family])),
                "mean_minus20db_support_occupancy": float(
                    family_occupancy[index]
                ),
                "a5_minus_a0_macro_f1_gain": float(
                    family_gains[index]
                ),
            }
            for index, family in enumerate(family_pool)
        ],
        "family_count": len(family_pool),
        "minimum_family_count": int(
            specification["minimum_family_count"]
        ),
        "spearman_rho": observed_rho,
        "spearman_bootstrap_ci95_low": float(ci_low),
        "spearman_bootstrap_ci95_high": float(ci_high),
        "exact_one_sided_permutation_p_value": one_sided_p,
        "exact_permutation_count": int(
            frozen_test["permutation_count"]
        ),
        "nonincreasing_order_holds": bool(
            frozen_test["nonincreasing_order_holds"]
        ),
        "strict_occupancy_inversion_count": int(
            frozen_test["strict_occupancy_inversion_count"]
        ),
        "strict_occupancy_inversions": frozen_test[
            "strict_occupancy_inversions"
        ],
        "predicted_direction": "negative",
        "direction_supported": direction_supported,
        "positive_mechanism_claim_eligible": direction_supported,
        "directional_success_required_for_submission_release": False,
        "bootstrap_draws": draws,
        "bootstrap_seed": bootstrap_seed,
        "occupancy_definition": specification["occupancy_definition"],
        "gain_definition": specification["gain_definition"],
        "frozen_protocol": OCCUPANCY_GAIN_MECHANISM_PROTOCOL,
    }


def _clean_retention_tests(
    *,
    run_root: Path,
    cache_digest: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Derive both preregistered clean-retention noninferiority gates."""

    experiment = contract["experiment"]
    specification = experiment["scientific_release_gates"][
        "clean_retention"
    ]
    regime = str(specification["regime"])
    baseline = str(specification["baseline_model"])
    candidate = str(specification["candidate_model"])
    seeds = [int(value) for value in experiment["seeds"]]
    references = {
        seed: _load_bundle(
            run_root,
            baseline,
            seed,
            regime,
            cache_digest,
        )
        for seed in seeds
    }
    candidates = {
        seed: _load_bundle(
            run_root,
            candidate,
            seed,
            regime,
            cache_digest,
        )
        for seed in seeds
    }
    first = references[seeds[0]]
    if first.target_profile_index is None:
        raise ValueError(
            "clean-retention predictions lack target_profile_index"
        )
    profiles = np.asarray(first.target_profile_index, dtype=np.int64)
    draws = int(experiment["statistics"]["bootstrap_draws"])
    seed_base = int(experiment["statistics"]["bootstrap_seed"])
    point_floor = float(specification["point_gain_floor"])
    interval_floor = float(specification["ci95_low_floor"])
    records: dict[str, dict[str, Any]] = {}
    for stratum_index, (name, profile_indices) in enumerate(
        specification["profile_strata"].items()
    ):
        selected = np.isin(
            profiles,
            np.asarray(profile_indices, dtype=np.int64),
        )
        if not np.any(selected):
            raise ValueError(
                f"clean-retention stratum has no sources: {name}"
            )
        references_subset = {
            seed: references[seed].subset(selected) for seed in seeds
        }
        candidates_subset = {
            seed: candidates[seed].subset(selected) for seed in seeds
        }
        bootstrap_seed = seed_base + 30_000 + stratum_index
        estimate = headline_paired_bootstrap(
            references_subset,
            candidates_subset,
            draws=draws,
            seed=bootstrap_seed,
        )
        point = float(estimate["macro_f1_difference"])
        ci_low = float(estimate["macro_f1_ci95_low"])
        passed = point >= point_floor and ci_low >= interval_floor
        records[name] = {
            "regime": regime,
            "profile_indices": [int(value) for value in profile_indices],
            "source_count": int(np.count_nonzero(selected)),
            "baseline_model": baseline,
            "candidate_model": candidate,
            "metric": "macro_f1",
            "candidate_minus_baseline": point,
            "ci95_low": ci_low,
            "ci95_high": float(estimate["macro_f1_ci95_high"]),
            "point_gain_floor": point_floor,
            "ci95_low_floor": interval_floor,
            "algorithm_seed_count": int(
                estimate["algorithm_seed_count"]
            ),
            "test_source_cluster_count": int(
                estimate["test_source_cluster_count"]
            ),
            "bootstrap_draws": draws,
            "bootstrap_seed": bootstrap_seed,
            "passed": bool(passed),
        }
    return {
        "regime": regime,
        "baseline_model": baseline,
        "candidate_model": candidate,
        "strata": records,
        "all_strata_complete": set(records)
        == set(specification["profile_strata"]),
        "all_strata_passed": bool(
            records
            and all(record["passed"] for record in records.values())
        ),
        "all_strata_must_pass": True,
    }


def _execution_integrity_reasons(
    *,
    run: dict[str, Any],
    learning_evidence: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate device, runtime, and full TVT-v2 source provenance."""

    expected_device = resolve_frozen_device(
        contract["experiment"]["device"]
    )
    bound_source_tree = learning_evidence.get("source_tree_binding")
    source_reasons = source_tree_audit_reasons(
        run.get("source_tree_execution_audit"),
        ROOT,
        Path(runner.__file__),
        expected_record=bound_source_tree,
        profile=TVT_V2_SOURCE_PROFILE,
    )
    environment_reasons = execution_environment_reasons(
        run.get("environment"),
        expected_device=expected_device,
        expected_source_tree=bound_source_tree,
    )
    learning_device_matches = (
        learning_evidence.get("execution_device") == expected_device
    )
    reasons = [*source_reasons, *environment_reasons]
    if not learning_device_matches:
        reasons.append("learning_curve_execution_device_drift")
    summary = {
        "device_contract": {
            "frozen": expected_device,
            "formal_run": (
                run.get("environment", {}).get("device")
                if isinstance(run.get("environment"), dict)
                else None
            ),
            "learning_curve": learning_evidence.get("execution_device"),
            "matches": not environment_reasons and learning_device_matches,
        },
        "source_tree_profile": TVT_V2_SOURCE_PROFILE,
        "source_tree_aggregate_digest": (
            bound_source_tree.get("aggregate_digest")
            if isinstance(bound_source_tree, dict)
            else None
        ),
        "source_tree_file_count": (
            bound_source_tree.get("file_count")
            if isinstance(bound_source_tree, dict)
            else None
        ),
        "source_tree_matches_learning_and_current": not source_reasons,
        "formal_runtime": run.get("environment"),
    }
    return reasons, summary


def derive_gate(
    *,
    run_json: Path,
    learning_evidence: Path,
    freeze: Path = DEFAULT_FREEZE,
    write: bool = False,
) -> dict[str, Any]:
    contract = load_contract(freeze)
    validated_learning = validate_learning_prerequisite(
        learning_evidence,
        contract,
    )
    run_json = run_json.resolve()
    run_root = run_json.parent
    run = _read_json(run_json)
    experiment = contract["experiment"]
    expected_cache = contract["cache"]
    reasons: list[str] = []
    integrity_reasons, execution_integrity = _execution_integrity_reasons(
        run=run,
        learning_evidence=validated_learning,
        contract=contract,
    )
    reasons.extend(integrity_reasons)
    if run.get("execution_status") != "complete":
        reasons.append("execution_status_not_complete")
    if run.get("run_id") != experiment["run_id"]:
        reasons.append("run_id_drift")
    if run.get("models") != experiment["models"]:
        reasons.append("model_grid_drift")
    if run.get("seeds") != experiment["seeds"]:
        reasons.append("algorithm_seed_grid_drift")
    reasons.extend(_fit_grid_reasons(run, run_root, contract))
    eligibility = run.get("evidence_eligibility")
    eligibility = eligibility if isinstance(eligibility, dict) else {}
    if not (
        eligibility.get("eligible") is True
        and eligibility.get("formal_paper_evidence_eligible") is True
        and eligibility.get("headline_eligible") is True
        and isinstance(eligibility.get("gates"), dict)
        and eligibility["gates"]
        and all(
            isinstance(gate, dict) and gate.get("passed") is True
            for gate in eligibility["gates"].values()
        )
    ):
        reasons.append("runner_evidence_eligibility_not_passed")
    runner_release = run.get("submission_release")
    runner_release = (
        runner_release if isinstance(runner_release, dict) else {}
    )
    if runner_release.get("macro_generation_permitted") is not True:
        reasons.append("runner_submission_release_not_eligible")
    comparison = run.get("comparison_protocol")
    comparison = comparison if isinstance(comparison, dict) else {}
    if (
        comparison.get("bootstrap_seed_base")
        != experiment["statistics"]["bootstrap_seed"]
    ):
        reasons.append("bootstrap_seed_drift")
    if (
        comparison.get("bootstrap_seed_base")
        == expected_cache["master_seed"]
    ):
        reasons.append("bootstrap_seed_equals_master_seed")
    cache_root = Path(str(run.get("cache_root", ""))).expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        manifest = {}
        reasons.append("cache_manifest_missing_or_malformed")
    run_cache_digest = str(run.get("cache_digest", ""))
    manifest_cache_digest = str(manifest.get("cache_digest", ""))
    if (
        len(run_cache_digest) != 64
        or run_cache_digest != manifest_cache_digest
    ):
        reasons.append("run_cache_digest_not_exactly_bound_to_current_cache")
    configuration = manifest.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    try:
        observed_split_counts = {
            str(split): int(size)
            for split, size in configuration.get("split_sizes", ())
        }
    except (TypeError, ValueError):
        observed_split_counts = {}
    if observed_split_counts != expected_cache[
        "expected_split_source_counts"
    ]:
        reasons.append("cache_exact_split_counts_drift")
    manifest_source_ids = manifest.get("source_ids")
    if (
        not isinstance(manifest_source_ids, dict)
        or set(manifest_source_ids) != set(observed_split_counts)
        or any(
            not isinstance(manifest_source_ids.get(split), list)
            or len(manifest_source_ids[split]) != count
            for split, count in observed_split_counts.items()
        )
    ):
        reasons.append("cache_source_id_counts_incomplete")
    if (
        configuration.get("master_seed") != expected_cache["master_seed"]
        or configuration.get("evidence_designation")
        != expected_cache["expected_designation"]
    ):
        reasons.append("cache_configuration_drift")
    if len(run_cache_digest) == 64 and cache_root.is_dir():
        reasons.extend(
            _prediction_grid_reasons(
                run=run,
                run_root=run_root,
                cache_root=cache_root,
                cache_digest=run_cache_digest,
                contract=contract,
            )
        )
    else:
        reasons.append("prediction_grid_not_derivable")

    family = experiment["confirmatory_family"]
    expected_ids = [
        record["contrast_id"] for record in family["contrasts"]
    ]
    confirmatory_family: dict[str, Any] = {
        "family_id": family["family_id"],
        "contrast_ids": expected_ids,
        "passed": False,
    }
    try:
        confirmatory_family = _confirmatory_family_test(
            run_root=run_root,
            cache_digest=run_cache_digest,
            contract=contract,
        )
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as error:
        reasons.append(f"confirmatory_family_validation_failed:{error}")
        confirmatory_family["validation_error"] = str(error)
    confirmatory_passed = confirmatory_family.get("passed") is True
    if not confirmatory_passed:
        reasons.append("confirmatory_family_gate_failed")

    axis_records: dict[str, dict[str, Any]] = {}
    calibration = experiment["scientific_release_gates"][
        "ood_axis_calibration"
    ]
    if not reasons or reasons == ["confirmatory_family_gate_failed"]:
        baseline = calibration["baseline_model"]
        candidate = calibration["candidate_model"]
        seeds = experiment["seeds"]
        try:
            id_by_seed = {
                seed: _load_bundle(
                    run_root,
                    baseline,
                    seed,
                    calibration["id_regime"],
                    run_cache_digest,
                )
                for seed in seeds
            }
            hard_by_seed = {
                seed: _load_bundle(
                    run_root,
                    baseline,
                    seed,
                    "hard_interference",
                    run_cache_digest,
                )
                for seed in seeds
            }
            for axis_index, axis in enumerate(
                (*EXPECTED_OOD_AXES, *EXPECTED_STRESS_AXES)
            ):
                axis_seed = (
                    int(experiment["statistics"]["bootstrap_seed"])
                    + 10_000
                    + axis_index
                )
                record = ood_axis_calibrated_bootstrap(
                    (
                        id_by_seed
                        if axis in EXPECTED_OOD_AXES
                        else hard_by_seed
                    ),
                    {
                        seed: _load_bundle(
                            run_root,
                            baseline,
                            seed,
                            axis,
                            run_cache_digest,
                        )
                        for seed in seeds
                    },
                    {
                        seed: _load_bundle(
                            run_root,
                            candidate,
                            seed,
                            axis,
                            run_cache_digest,
                        )
                        for seed in seeds
                    },
                    draws=int(experiment["statistics"]["bootstrap_draws"]),
                    seed=axis_seed,
                    recovery_fraction=float(
                        calibration["recovery_fraction"]
                    ),
                    opportunity_floor=float(
                        calibration["opportunity_floor"]
                    ),
                    noninferiority_floor=-0.01,
                )
                record["nominal_reference_regime"] = (
                    calibration["id_regime"]
                    if axis in EXPECTED_OOD_AXES
                    else experiment["scientific_release_gates"][
                        "receiver_robustness"
                    ]["nominal_reference_regime"]
                )
                axis_records[axis] = record
        except (OSError, KeyError, ValueError) as error:
            reasons.append(f"axis_calibration_failed:{error}")
    required_axes = (*EXPECTED_OOD_AXES, *EXPECTED_STRESS_AXES)
    axes_complete = set(axis_records) == set(required_axes)
    axes_passed = bool(
        axes_complete
        and all(
            axis_records[axis]["axis_gate_passed"]
            for axis in required_axes
        )
    )
    if not axes_complete:
        reasons.append("ood_or_receiver_axis_artifacts_incomplete")
    elif not axes_passed:
        reasons.append("ood_or_receiver_axis_gate_failed")
    clean_retention: dict[str, Any] | None = None
    if len(run_cache_digest) == 64:
        try:
            clean_retention = _clean_retention_tests(
                run_root=run_root,
                cache_digest=run_cache_digest,
                contract=contract,
            )
        except (OSError, KeyError, ValueError) as error:
            reasons.append(f"clean_retention_test_failed:{error}")
    else:
        reasons.append("clean_retention_test_not_derivable")
    clean_complete = bool(
        clean_retention
        and clean_retention["all_strata_complete"] is True
    )
    clean_passed = bool(
        clean_retention
        and clean_retention["all_strata_passed"] is True
    )
    if not clean_complete:
        reasons.append("clean_retention_artifacts_incomplete")
    elif not clean_passed:
        reasons.append("clean_retention_gate_failed")
    occupancy_trend: dict[str, Any] | None = None
    if manifest and len(run_cache_digest) == 64:
        try:
            occupancy_trend = _jammer_occupancy_directional_test(
                run_root=run_root,
                cache_root=cache_root,
                manifest=manifest,
                cache_digest=run_cache_digest,
                contract=contract,
            )
        except (OSError, KeyError, ValueError) as error:
            reasons.append(
                f"jammer_occupancy_directional_test_failed:{error}"
            )
    else:
        reasons.append("jammer_occupancy_directional_test_not_derivable")
    scientific_evidence_passed = not reasons
    gate = {
        "schema_version": GATE_SCHEMA,
        "passed": scientific_evidence_passed,
        "scientific_evidence_passed": scientific_evidence_passed,
        "submission_unlocked": False,
        "paper_release_unlocked": False,
        "paper_promotion": {
            "status": "not_run",
            "required": True,
            "reason": (
                "v2 scientific evidence must be transformed into the "
                "canonical paper macro manifest and bound release lock "
                "before submission can be unlocked"
            ),
        },
        "run_id": run.get("run_id"),
        "run_json": str(run_json),
        "run_json_sha256": _sha256(run_json),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": (
            _sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "cache_digest": run_cache_digest,
        "learning_curve_evidence": str(
            learning_evidence.resolve()
        ),
        "learning_curve_evidence_sha256": _sha256(
            learning_evidence.resolve()
        ),
        "confirmatory_family": confirmatory_family,
        "axis_calibration": axis_records,
        "all_ood_and_receiver_axes_complete": axes_complete,
        "all_ood_and_receiver_axes_passed": axes_passed,
        "clean_retention": clean_retention,
        "all_clean_retention_strata_complete": clean_complete,
        "all_clean_retention_strata_passed": clean_passed,
        "jammer_occupancy_directional_test": occupancy_trend,
        "jammer_occupancy_directional_result_required_to_pass": False,
        "bootstrap_seed_base": comparison.get("bootstrap_seed_base"),
        "master_seed": configuration.get("master_seed"),
        "execution_integrity": {
            **execution_integrity,
            "release_validator_runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "reasons": reasons,
        "manually_typed_performance_values_used": False,
    }
    output = run_root / "v2_scientific_release_gate.json"
    if write:
        output.write_text(
            json.dumps(
                gate,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument(
        "--learning-curve-evidence",
        type=Path,
        required=True,
    )
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        gate = derive_gate(
            run_json=arguments.run_json,
            learning_evidence=arguments.learning_curve_evidence,
            freeze=arguments.freeze,
            write=arguments.write,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "submission_unlocked": False,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
