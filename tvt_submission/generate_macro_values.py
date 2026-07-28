"""Derive the TVT result-macro manifest from one eligible formal run.

No performance value is accepted from the command line.  The generator first
passes the source run through ``validate_release.validate_source_run``, then
cross-checks the runner-native JSON, CSV, and prediction NPZ artifacts used by
the paper claims.  Any missing, duplicate, nonfinite, inconsistent, or
ambiguous input closes the release.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tvt_submission import validate_release  # noqa: E402


METHOD_MODEL = "a5_vimd_full"
PRIMARY_REFERENCE_MODEL = "cssl_amc_supervised_adaptation"
HARD_REGIME = "hard_interference"
CLEAN_PROFILE_REGIMES = {
    "clean_retention_seen_acd": (0, 2, 3),
    "clean_retention_held_be": (1, 4),
}
HEADLINE_MODELS = {
    "AZero": "a0_backbone",
    "MCLDNN": "mcldnn_reimplementation",
    "IQFormer": "iqformer_inspired",
    "CSSL": "cssl_amc_supervised_adaptation",
    "AFive": METHOD_MODEL,
}
HEADLINE_METRIC_SUFFIXES = {
    "accuracy": "Accuracy",
    "macro_f1": "MacroFOne",
    "worst_recall": "WorstRecall",
    "nll": "NLL",
    "ece": "ECE",
}
OOD_REGIMES = {
    "Hard": HARD_REGIME,
    "UnseenJammer": "unseen_jammer",
    "UnseenSpeed": "unseen_speed",
    "HeldoutChannel": "heldout_channel",
    "CombinedOOD": "combined_ood",
    "CleanACD": "clean_retention_seen_acd",
    "CleanBE": "clean_retention_held_be",
}
BASELINE_MODELS = (
    "a0_backbone",
    "mcldnn_reimplementation",
    "iqformer_inspired",
    "cssl_amc_supervised_adaptation",
)
ABLATION_CONTROL_MODELS = ("a1_single_mask", "a6_dual_full")
FORMAL_HOLM_CANDIDATES = (
    "a0_backbone",
    "a1_single_mask",
    "a5_vimd_full",
    "mcldnn_reimplementation",
    "iqformer_inspired",
)
OOD_GATE_REGIMES = ("unseen_jammer", "unseen_speed", "heldout_channel")
SCIENTIFIC_RELEASE_THRESHOLDS = {
    "hard_macro_f1_min_gain_pp_each_baseline": 5.0,
    "hard_ablation_controls": list(ABLATION_CONTROL_MODELS),
    "hard_ablation_strictly_positive": True,
    "ood_macro_f1_min_gain_pp": 3.0,
    "ood_required_pass_count": 2,
    "ood_regimes": list(OOD_GATE_REGIMES),
    "clean_macro_f1_min_point_gain_pp": -1.0,
    "clean_macro_f1_min_ci95_low_pp": -2.0,
    "mechanism_required_finite_fields": [
        "mask_js",
        "overlap_uncertainty_route_weighted_correlation",
        "target_energy_transfer_ratio_mean",
        "target_energy_transfer_ratio_amplification_share",
        "jammer_leakage",
        "oracle_vs_predicted_overlap_spearman",
        "overlap_permutation_p_value",
        "counterfactual_tf_sir_gain_db",
    ],
    "mechanism_nonnegative_fields": [
        "overlap_uncertainty_route_weighted_correlation",
        "oracle_vs_predicted_overlap_spearman",
    ],
    "oracle_spectral_ratio_field": "counterfactual_tf_sir_gain_db",
    "oracle_spectral_ratio_strictly_positive": True,
}
MECHANISM_MACRO_FIELDS = {
    "MechanismMaskJS": "mask_js",
    "MechanismThirdRouteWeightedCorrelation": (
        "overlap_uncertainty_route_weighted_correlation"
    ),
    "MechanismTargetTransferRatio": "target_energy_transfer_ratio_mean",
    "MechanismTargetAmplificationShare": (
        "target_energy_transfer_ratio_amplification_share"
    ),
    "MechanismJammerLeakage": "jammer_leakage",
    "MechanismThirdRouteSpearman": "oracle_vs_predicted_overlap_spearman",
    "MechanismThirdRoutePermutationP": "overlap_permutation_p_value",
    "OracleSpectralRatioGain": "counterfactual_tf_sir_gain_db",
}
HEADLINE_PROVENANCE_MACROS = tuple(
    f"HeadlineHard{model}{metric}"
    for model in HEADLINE_MODELS
    for metric in HEADLINE_METRIC_SUFFIXES.values()
)
REGIME_PROVENANCE_MACROS = tuple(
    f"Regime{regime}{field}"
    for regime in OOD_REGIMES
    for field in ("Reference", "AFive", "Gain", "CILow", "CIHigh")
)
PROVENANCE_MACROS = (
    "PrimaryReference",
    *HEADLINE_PROVENANCE_MACROS,
    *REGIME_PROVENANCE_MACROS,
    *MECHANISM_MACRO_FIELDS,
    "VIMDParameters",
    "VIMDLatencyPFifty",
    "VIMDLatencyPNinetyFive",
    "VIMDLatencyDevice",
)
MODEL_LABELS = {
    "a0_backbone": "direct spectral backbone",
    "mcldnn_reimplementation": "MCLDNN reimplementation",
    "iqformer_inspired": "IQFormer-inspired local baseline",
    "cssl_amc_supervised_adaptation": (
        "CSSL-AMC official-architecture supervised adaptation"
    ),
}
METRIC_NAMES = ("accuracy", "macro_f1", "worst_recall", "nll", "ece")
PREDICTION_KEYS = frozenset(
    {
        "probabilities",
        "labels",
        "source_ids",
        "snr_db",
        "sir_db",
        "target_profile_index",
        "cache_digest",
        "split",
    }
)
HEADLINE_NUMERIC_COLUMNS = (
    "difference",
    "ci95_low",
    "ci95_high",
    "accuracy_difference",
    "accuracy_ci95_low",
    "accuracy_ci95_high",
    "macro_f1_difference",
    "macro_f1_ci95_low",
    "macro_f1_ci95_high",
    "accuracy_test_source_only_ci95_low",
    "accuracy_test_source_only_ci95_high",
    "macro_f1_test_source_only_ci95_low",
    "macro_f1_test_source_only_ci95_high",
    "accuracy_algorithm_seed_only_ci95_low",
    "accuracy_algorithm_seed_only_ci95_high",
    "macro_f1_algorithm_seed_only_ci95_low",
    "macro_f1_algorithm_seed_only_ci95_high",
)
FLOAT_ABS_TOLERANCE = 1e-12
FLOAT_REL_TOLERANCE = 1e-9


class MacroGenerationError(RuntimeError):
    """A macro derivation invariant failed."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MacroGenerationError(f"{label} is boolean, not numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MacroGenerationError(f"{label} is not numeric") from error
    if not math.isfinite(numeric):
        raise MacroGenerationError(f"{label} is nonfinite")
    return numeric


def _integer(value: Any, label: str) -> int:
    numeric = _finite_number(value, label)
    if not numeric.is_integer():
        raise MacroGenerationError(f"{label} is not an integer")
    return int(numeric)


def _same_number(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _load_csv(
    path: Path,
    *,
    required_columns: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if not header or any(not name.strip() for name in header):
                raise MacroGenerationError(f"{path.name} has a blank header")
            if len(header) != len(set(header)):
                raise MacroGenerationError(
                    f"{path.name} has duplicate columns"
                )
            missing = sorted(required_columns.difference(header))
            if missing:
                raise MacroGenerationError(
                    f"{path.name} is missing columns: {','.join(missing)}"
                )
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise MacroGenerationError(
                        f"{path.name}:{line_number} has {len(values)} cells; "
                        f"expected {len(header)}"
                    )
                rows.append(dict(zip(header, values)))
    except (OSError, UnicodeError, csv.Error) as error:
        raise MacroGenerationError(
            f"could not read strict CSV {path}: {error}"
        ) from error
    if not rows:
        raise MacroGenerationError(f"{path.name} contains no data rows")
    return header, rows


def _result_index(
    run_record: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    results = run_record.get("results")
    if not isinstance(results, list):
        raise MacroGenerationError("run results are not a list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for position, result in enumerate(results):
        if not isinstance(result, dict):
            raise MacroGenerationError(
                f"run result index {position} is not an object"
            )
        model = result.get("model")
        if not isinstance(model, str) or not model:
            raise MacroGenerationError(
                f"run result index {position} has invalid model"
            )
        seed = _integer(result.get("seed"), f"{model} result seed")
        key = (model, seed)
        if key in indexed:
            raise MacroGenerationError(
                f"duplicate run result for {model}/seed{seed}"
            )
        indexed[key] = result
    return indexed


def _audit_metrics_csv(
    run_json: Path,
    run_record: dict[str, Any],
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, float]]:
    path = run_json.parent / "metrics.csv"
    required = {
        "model",
        "seed",
        "regime",
        "cache_digest",
        *METRIC_NAMES,
    }
    _, rows = _load_csv(path, required_columns=required)
    models = run_record["models"]
    seeds = run_record["seeds"]
    regimes = [name for name in run_record["splits"] if name != "train"]
    expected = {
        (str(model), int(seed), str(regime))
        for model in models
        for seed in seeds
        for regime in regimes
    }
    indexed: dict[tuple[str, int, str], dict[str, float]] = {}
    for line_index, row in enumerate(rows, start=2):
        model = row["model"]
        seed = _integer(row["seed"], f"metrics.csv:{line_index} seed")
        regime = row["regime"]
        key = (model, seed, regime)
        if key in indexed:
            raise MacroGenerationError(
                f"metrics.csv has duplicate row {key}"
            )
        if row["cache_digest"] != run_record["cache_digest"]:
            raise MacroGenerationError(
                f"metrics.csv row {key} has wrong cache digest"
            )
        values = {
            metric: _finite_number(
                row[metric],
                f"metrics.csv row {key} column {metric}",
            )
            for metric in METRIC_NAMES
        }
        for bounded_metric in ("accuracy", "macro_f1", "worst_recall", "ece"):
            if not 0.0 <= values[bounded_metric] <= 1.0:
                raise MacroGenerationError(
                    f"metrics.csv row {key} column {bounded_metric} falls "
                    "outside [0,1]"
                )
        if values["nll"] < 0.0:
            raise MacroGenerationError(
                f"metrics.csv row {key} column nll is negative"
            )
        result = results.get((model, seed))
        regime_record = (
            result.get("regimes", {}).get(regime)
            if isinstance(result, dict)
            else None
        )
        if not isinstance(regime_record, dict):
            raise MacroGenerationError(
                f"run.json has no metric record for {key}"
            )
        for metric, csv_value in values.items():
            run_value = _finite_number(
                regime_record.get(metric),
                f"run.json result {key} metric {metric}",
            )
            if not _same_number(csv_value, run_value):
                raise MacroGenerationError(
                    f"metrics.csv disagrees with run.json for "
                    f"{key}/{metric}"
                )
        indexed[key] = values
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))
        unexpected = sorted(set(indexed).difference(expected))
        raise MacroGenerationError(
            "metrics.csv model/seed/regime matrix mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return indexed


def _macro_f1_and_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: int,
) -> tuple[float, float]:
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
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
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    if not supported.any():
        raise MacroGenerationError("prediction bundle has no supported class")
    return (
        float(f1[supported].mean()),
        float((predictions == labels).mean()),
    )


def _scalar_text(array: np.ndarray, label: str) -> str:
    if array.size != 1:
        raise MacroGenerationError(f"{label} is not scalar")
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MacroGenerationError(
                f"{label} is not UTF-8"
            ) from error
    return str(value)


def _load_prediction_bundle(
    path: Path,
    *,
    cache_digest: str,
    regime: str,
    classes: int,
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != PREDICTION_KEYS:
                missing = sorted(PREDICTION_KEYS.difference(archive.files))
                unexpected = sorted(set(archive.files).difference(PREDICTION_KEYS))
                raise MacroGenerationError(
                    f"{path.name} NPZ schema mismatch; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except MacroGenerationError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise MacroGenerationError(
            f"could not load prediction NPZ {path}: {error}"
        ) from error

    probabilities = arrays["probabilities"]
    labels = arrays["labels"]
    source_ids = arrays["source_ids"]
    snr_db = arrays["snr_db"]
    sir_db = arrays["sir_db"]
    target_profile_index = arrays["target_profile_index"]
    if probabilities.ndim != 2 or probabilities.shape[1] != classes:
        raise MacroGenerationError(
            f"{path.name} probabilities have the wrong shape"
        )
    sample_count = probabilities.shape[0]
    for name, array in (
        ("labels", labels),
        ("source_ids", source_ids),
        ("snr_db", snr_db),
        ("sir_db", sir_db),
        ("target_profile_index", target_profile_index),
    ):
        if array.ndim != 1 or len(array) != sample_count:
            raise MacroGenerationError(
                f"{path.name} {name} does not align with probabilities"
            )
    if sample_count <= 0:
        raise MacroGenerationError(f"{path.name} contains no predictions")
    if not np.issubdtype(labels.dtype, np.integer):
        raise MacroGenerationError(f"{path.name} labels are not integers")
    if not np.issubdtype(target_profile_index.dtype, np.integer):
        raise MacroGenerationError(
            f"{path.name} target_profile_index is not integer-valued"
        )
    integer_labels = labels.astype(np.int64, copy=False)
    integer_profiles = target_profile_index.astype(np.int64, copy=False)
    if integer_labels.min() < 0 or integer_labels.max() >= classes:
        raise MacroGenerationError(
            f"{path.name} labels fall outside the class taxonomy"
        )
    if integer_profiles.min() < 0 or integer_profiles.max() > 4:
        raise MacroGenerationError(
            f"{path.name} target_profile_index falls outside TDL-A--E"
        )
    if len(np.unique(source_ids)) != sample_count:
        raise MacroGenerationError(
            f"{path.name} source_ids are not unique source clusters"
        )
    if not np.isfinite(probabilities).all():
        raise MacroGenerationError(
            f"{path.name} probabilities contain nonfinite values"
        )
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise MacroGenerationError(
            f"{path.name} probabilities fall outside [0,1]"
        )
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise MacroGenerationError(
            f"{path.name} probability rows do not sum to one"
        )
    if not np.isfinite(snr_db).all() or not np.isfinite(sir_db).all():
        raise MacroGenerationError(
            f"{path.name} required jammer-active SNR/SIR metadata is nonfinite"
        )
    if _scalar_text(arrays["cache_digest"], f"{path.name} cache_digest") != (
        cache_digest
    ):
        raise MacroGenerationError(f"{path.name} has the wrong cache digest")
    if _scalar_text(arrays["split"], f"{path.name} split") != regime:
        raise MacroGenerationError(f"{path.name} has the wrong split token")
    macro_f1, accuracy = _macro_f1_and_accuracy(
        probabilities,
        integer_labels,
        classes,
    )
    return {
        "labels": integer_labels,
        "source_ids": source_ids,
        "probabilities": probabilities,
        "target_profile_index": integer_profiles,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def _audit_required_predictions(
    run_json: Path,
    run_record: dict[str, Any],
    metrics: dict[tuple[str, int, str], dict[str, float]],
    reference_model: str,
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[str, int],
]:
    seeds = [int(seed) for seed in run_record["seeds"]]
    classes = _integer(run_record.get("num_classes"), "run num_classes")
    models_by_regime = {
        HARD_REGIME: (
            set(BASELINE_MODELS)
            | set(ABLATION_CONTROL_MODELS)
            | {METHOD_MODEL}
        ),
        "unseen_jammer": {reference_model, METHOD_MODEL},
        "unseen_speed": {reference_model, METHOD_MODEL},
        "heldout_channel": {reference_model, METHOD_MODEL},
        "combined_ood": {reference_model, METHOD_MODEL},
        "clean_retention": {reference_model, METHOD_MODEL},
    }
    bundles: dict[tuple[str, int, str], dict[str, Any]] = {}
    sample_counts: dict[str, int] = {}
    for regime, models in models_by_regime.items():
        canonical_labels: np.ndarray | None = None
        canonical_sources: np.ndarray | None = None
        canonical_profiles: np.ndarray | None = None
        for model in sorted(models):
            for seed in seeds:
                path = (
                    run_json.parent
                    / "models"
                    / f"{model}_seed{seed}"
                    / f"predictions_{regime}.npz"
                )
                bundle = _load_prediction_bundle(
                    path,
                    cache_digest=str(run_record["cache_digest"]),
                    regime=regime,
                    classes=classes,
                )
                key = (model, seed, regime)
                metric_row = metrics[key]
                for name in ("macro_f1", "accuracy"):
                    if not _same_number(bundle[name], metric_row[name]):
                        raise MacroGenerationError(
                            f"{path.name} predictions disagree with "
                            f"metrics.csv for {key}/{name}"
                        )
                if canonical_labels is None:
                    canonical_labels = bundle["labels"]
                    canonical_sources = bundle["source_ids"]
                    canonical_profiles = bundle["target_profile_index"]
                elif not (
                    np.array_equal(bundle["labels"], canonical_labels)
                    and np.array_equal(bundle["source_ids"], canonical_sources)
                    and np.array_equal(
                        bundle["target_profile_index"],
                        canonical_profiles,
                    )
                ):
                    raise MacroGenerationError(
                        f"prediction bundles are not source-aligned in {regime}"
                    )
                bundles[key] = bundle
        if canonical_labels is None:
            raise MacroGenerationError(
                f"no required prediction bundle was audited for {regime}"
            )
        sample_counts[regime] = len(canonical_labels)
    return bundles, sample_counts


def _csv_boolean(value: str, label: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise MacroGenerationError(f"{label} is not a strict boolean")


def _seed_ids(value: str, label: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise MacroGenerationError(f"{label} is not a list literal") from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, (str, int)) for item in parsed)
    ):
        raise MacroGenerationError(f"{label} is not a nonempty seed list")
    return [str(item) for item in parsed]


def _analysis_seed(base_seed: int, *tokens: Any) -> int:
    payload = "|".join([str(int(base_seed)), *(str(token) for token in tokens)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _audit_headline_csv(
    run_json: Path,
    run_record: dict[str, Any],
    metrics: dict[tuple[str, int, str], dict[str, float]],
    bundles: dict[tuple[str, int, str], dict[str, Any]],
    sample_counts: dict[str, int],
    reference_model: str,
) -> dict[tuple[str, str], dict[str, float]]:
    path = run_json.parent / "headline_paired_statistics.csv"
    required_columns = {
        "reference",
        "reference_selection",
        "reference_strength_claimed",
        "candidate",
        "regime",
        "parent_regime",
        "target_profile_indices",
        "cache_digest",
        *HEADLINE_NUMERIC_COLUMNS,
        "bootstrap_draws",
        "bootstrap_seed",
        "algorithm_seed_count",
        "algorithm_seed_ids",
        "test_source_cluster_count",
        "bootstrap_stratified_by_class",
        "bootstrap_hierarchy",
        "mcnemar_exact_test_performed",
        "mcnemar_reason",
    }
    _, rows = _load_csv(path, required_columns=required_columns)
    heldout_regimes = {
        str(regime)
        for regime in run_record["splits"]
        if str(regime) not in {"train", "validation"}
    }
    candidates = {
        str(model)
        for model in run_record["models"]
        if str(model) != reference_model
    }
    expected = {
        (candidate, regime)
        for candidate in candidates
        for regime in heldout_regimes
    }
    expected.update(
        (METHOD_MODEL, regime) for regime in CLEAN_PROFILE_REGIMES
    )
    seeds = [int(seed) for seed in run_record["seeds"]]
    expected_seed_ids = sorted(str(seed) for seed in seeds)
    comparison = run_record.get("comparison_protocol")
    if not isinstance(comparison, dict):
        raise MacroGenerationError("run comparison_protocol is malformed")
    indexed: dict[tuple[str, str], dict[str, float]] = {}
    for line_number, row in enumerate(rows, start=2):
        if row["reference"] != reference_model:
            raise MacroGenerationError(
                f"headline CSV line {line_number} has wrong reference"
            )
        if row["reference_selection"] != comparison.get(
            "reference_selection"
        ):
            raise MacroGenerationError(
                f"headline CSV line {line_number} has wrong reference selection"
            )
        if _csv_boolean(
            row["reference_strength_claimed"],
            f"headline CSV line {line_number} reference_strength_claimed",
        ):
            raise MacroGenerationError(
                "runner headline statistics improperly claim reference strength"
            )
        candidate = row["candidate"]
        regime = row["regime"]
        key = (candidate, regime)
        if key in indexed:
            raise MacroGenerationError(
                f"headline CSV has duplicate row {key}"
            )
        if row["cache_digest"] != run_record["cache_digest"]:
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong cache digest"
            )
        if regime in CLEAN_PROFILE_REGIMES:
            if (
                candidate != METHOD_MODEL
                or row["parent_regime"] != "clean_retention"
            ):
                raise MacroGenerationError(
                    f"headline CSV row {key} has wrong clean-profile role"
                )
            try:
                profile_indices = tuple(
                    int(value)
                    for value in ast.literal_eval(
                        row["target_profile_indices"]
                    )
                )
            except (SyntaxError, ValueError, TypeError) as error:
                raise MacroGenerationError(
                    f"headline CSV row {key} has malformed profile indices"
                ) from error
            if profile_indices != CLEAN_PROFILE_REGIMES[regime]:
                raise MacroGenerationError(
                    f"headline CSV row {key} has wrong profile indices"
                )
        elif (
            row["parent_regime"] != regime
            or row["target_profile_indices"] != "all"
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong parent/profile scope"
            )
        values = {
            column: _finite_number(
                row[column],
                f"headline CSV row {key} column {column}",
            )
            for column in HEADLINE_NUMERIC_COLUMNS
        }
        for prefix in (
            "accuracy",
            "macro_f1",
            "accuracy_test_source_only",
            "macro_f1_test_source_only",
            "accuracy_algorithm_seed_only",
            "macro_f1_algorithm_seed_only",
        ):
            low = values[f"{prefix}_ci95_low"]
            high = values[f"{prefix}_ci95_high"]
            if low > high:
                raise MacroGenerationError(
                    f"headline CSV row {key} has reversed {prefix} interval"
                )
        if not (
            _same_number(values["difference"], values["accuracy_difference"])
            and _same_number(values["ci95_low"], values["accuracy_ci95_low"])
            and _same_number(values["ci95_high"], values["accuracy_ci95_high"])
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} accuracy aliases disagree"
            )
        if _integer(
            row["algorithm_seed_count"],
            f"headline CSV row {key} algorithm_seed_count",
        ) != len(seeds):
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong seed count"
            )
        if sorted(
            _seed_ids(
                row["algorithm_seed_ids"],
                f"headline CSV row {key} algorithm_seed_ids",
            )
        ) != expected_seed_ids:
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong seed ids"
            )
        if _integer(
            row["bootstrap_draws"],
            f"headline CSV row {key} bootstrap_draws",
        ) != _integer(
            comparison.get("bootstrap_draws"),
            "run comparison bootstrap_draws",
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong bootstrap draw count"
            )
        expected_bootstrap_seed = _analysis_seed(
            _integer(
                comparison.get("bootstrap_seed_base"),
                "run comparison bootstrap_seed_base",
            ),
            "hierarchical",
            reference_model,
            candidate,
            regime,
        )
        if _integer(
            row["bootstrap_seed"],
            f"headline CSV row {key} bootstrap_seed",
        ) != expected_bootstrap_seed:
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong deterministic bootstrap seed"
            )
        if not _csv_boolean(
            row["bootstrap_stratified_by_class"],
            f"headline CSV row {key} bootstrap_stratified_by_class",
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} is not class-stratified"
            )
        if row["bootstrap_hierarchy"] != (
            "algorithm_seed_and_class_stratified_test_source_cluster"
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong bootstrap hierarchy"
            )
        if _csv_boolean(
            row["mcnemar_exact_test_performed"],
            f"headline CSV row {key} mcnemar_exact_test_performed",
        ):
            raise MacroGenerationError(
                f"headline CSV row {key} improperly pools McNemar tests"
            )
        indexed[key] = values
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))
        unexpected = sorted(set(indexed).difference(expected))
        raise MacroGenerationError(
            "headline_paired_statistics.csv row matrix mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for regime in (
        HARD_REGIME,
        "unseen_jammer",
        "unseen_speed",
        "heldout_channel",
        "combined_ood",
        "clean_retention",
    ):
        key = (METHOD_MODEL, regime)
        row = indexed[key]
        per_seed_differences = [
            bundles[(METHOD_MODEL, seed, regime)]["macro_f1"]
            - bundles[(reference_model, seed, regime)]["macro_f1"]
            for seed in seeds
        ]
        recomputed = float(np.mean(per_seed_differences))
        metric_difference = float(
            np.mean(
                [
                    metrics[(METHOD_MODEL, seed, regime)]["macro_f1"]
                    - metrics[(reference_model, seed, regime)]["macro_f1"]
                    for seed in seeds
                ]
            )
        )
        if not (
            _same_number(row["macro_f1_difference"], recomputed)
            and _same_number(row["macro_f1_difference"], metric_difference)
        ):
            raise MacroGenerationError(
                f"headline macro-F1 point estimate disagrees with "
                f"NPZ/metrics artifacts for {regime}"
            )
        headline_row = next(
            item
            for item in rows
            if item["candidate"] == METHOD_MODEL
            and item["regime"] == regime
        )
        if _integer(
            headline_row["test_source_cluster_count"],
            f"headline CSV row {key} test_source_cluster_count",
        ) != sample_counts[regime]:
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong source-cluster count"
            )
    clean_bundle = bundles[
        (reference_model, seeds[0], "clean_retention")
    ]
    canonical_profiles = clean_bundle["target_profile_index"]
    for regime, profile_indices in CLEAN_PROFILE_REGIMES.items():
        key = (METHOD_MODEL, regime)
        selected = np.isin(
            canonical_profiles,
            np.asarray(profile_indices, dtype=np.int64),
        )
        if not selected.any():
            raise MacroGenerationError(
                f"clean-retention profile stratum {regime} is empty"
            )
        per_seed_differences: list[float] = []
        for seed in seeds:
            reference_bundle = bundles[
                (reference_model, seed, "clean_retention")
            ]
            method_bundle = bundles[
                (METHOD_MODEL, seed, "clean_retention")
            ]
            current_selected = np.isin(
                reference_bundle["target_profile_index"],
                np.asarray(profile_indices, dtype=np.int64),
            )
            if not np.array_equal(current_selected, selected):
                raise MacroGenerationError(
                    "clean-retention profile stratum differs across seeds"
                )
            reference_f1, _ = _macro_f1_and_accuracy(
                reference_bundle["probabilities"][selected],
                reference_bundle["labels"][selected],
                reference_bundle["probabilities"].shape[1],
            )
            method_f1, _ = _macro_f1_and_accuracy(
                method_bundle["probabilities"][selected],
                method_bundle["labels"][selected],
                method_bundle["probabilities"].shape[1],
            )
            per_seed_differences.append(method_f1 - reference_f1)
        if not _same_number(
            indexed[key]["macro_f1_difference"],
            float(np.mean(per_seed_differences)),
        ):
            raise MacroGenerationError(
                f"headline clean-profile macro-F1 point estimate disagrees "
                f"with prediction artifacts for {regime}"
            )
        headline_row = next(
            item
            for item in rows
            if item["candidate"] == METHOD_MODEL
            and item["regime"] == regime
        )
        if _integer(
            headline_row["test_source_cluster_count"],
            f"headline CSV row {key} test_source_cluster_count",
        ) != int(selected.sum()):
            raise MacroGenerationError(
                f"headline CSV row {key} has wrong source-cluster count"
            )
    return indexed


def _method_summary(
    run_record: dict[str, Any],
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, float | int | str]:
    seeds = [int(seed) for seed in run_record["seeds"]]
    mechanism_fields = (
        "mask_js",
        "overlap_uncertainty_route_weighted_correlation",
        "target_energy_transfer_ratio_mean",
        "target_energy_transfer_ratio_amplification_share",
        "jammer_leakage",
        "oracle_vs_predicted_overlap_spearman",
        "overlap_permutation_p_value",
        "counterfactual_tf_sir_gain_db",
    )
    mechanism_values: dict[str, list[float]] = {
        field: [] for field in mechanism_fields
    }
    parameters: list[int] = []
    latencies_p50: list[float] = []
    latencies_p95: list[float] = []
    latency_devices: set[str] = set()
    for seed in seeds:
        result = results.get((METHOD_MODEL, seed))
        if not isinstance(result, dict):
            raise MacroGenerationError(
                f"run is missing {METHOD_MODEL}/seed{seed}"
            )
        mechanism = result.get("mechanism")
        if not isinstance(mechanism, dict):
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} has no mechanism record"
            )
        if mechanism.get("schema_version") != 2:
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} mechanism schema is not v2"
            )
        split = mechanism.get("split")
        if split not in (None, "heldout_channel"):
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} mechanism uses wrong split"
            )
        for field in mechanism_fields:
            mechanism_values[field].append(
                _finite_number(
                    mechanism.get(field),
                    f"{METHOD_MODEL}/seed{seed} mechanism {field}",
                )
            )
        for field in (
            "mask_js",
            "target_energy_transfer_ratio_mean",
            "target_energy_transfer_ratio_amplification_share",
            "jammer_leakage",
            "overlap_permutation_p_value",
        ):
            if mechanism_values[field][-1] < 0:
                raise MacroGenerationError(
                    f"{METHOD_MODEL}/seed{seed} mechanism {field} is negative"
                )
        for field in (
            "target_energy_transfer_ratio_amplification_share",
            "overlap_permutation_p_value",
        ):
            if mechanism_values[field][-1] > 1:
                raise MacroGenerationError(
                    f"{METHOD_MODEL}/seed{seed} mechanism {field} exceeds one"
                )
        for field in (
            "overlap_uncertainty_route_weighted_correlation",
            "oracle_vs_predicted_overlap_spearman",
        ):
            if abs(mechanism_values[field][-1]) > 1:
                raise MacroGenerationError(
                    f"{METHOD_MODEL}/seed{seed} mechanism {field} "
                    "falls outside [-1,1]"
                )
        complexity = result.get("complexity")
        if not isinstance(complexity, dict):
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} has no complexity record"
            )
        parameter_value = _finite_number(
            complexity.get("parameters"),
            f"{METHOD_MODEL}/seed{seed} parameters",
        )
        if parameter_value <= 0 or not parameter_value.is_integer():
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} parameter count is invalid"
            )
        parameters.append(int(parameter_value))
        latency_p50 = _finite_number(
            complexity.get("latency_ms_p50"),
            f"{METHOD_MODEL}/seed{seed} latency_ms_p50",
        )
        latency_p95 = _finite_number(
            complexity.get("latency_ms_p95"),
            f"{METHOD_MODEL}/seed{seed} latency_ms_p95",
        )
        if latency_p50 <= 0 or latency_p95 < latency_p50:
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} latency quantiles are invalid"
            )
        latencies_p50.append(latency_p50)
        latencies_p95.append(latency_p95)
        device = complexity.get("latency_device")
        if (
            not isinstance(device, str)
            or not device.strip()
            or device != device.strip()
            or any(character in device for character in "\\{}%\r\n")
        ):
            raise MacroGenerationError(
                f"{METHOD_MODEL}/seed{seed} latency device is invalid"
            )
        latency_devices.add(device)
    if len(set(parameters)) != 1:
        raise MacroGenerationError(
            f"{METHOD_MODEL} parameter count differs across seeds"
        )
    if len(latency_devices) != 1:
        raise MacroGenerationError(
            f"{METHOD_MODEL} latency device differs across seeds"
        )
    for field in SCIENTIFIC_RELEASE_THRESHOLDS[
        "mechanism_nonnegative_fields"
    ]:
        if any(value < 0.0 for value in mechanism_values[field]):
            raise MacroGenerationError(
                "scientific release gate failed: "
                f"{field} must be nonnegative for every seed"
            )
    oracle_field = str(
        SCIENTIFIC_RELEASE_THRESHOLDS["oracle_spectral_ratio_field"]
    )
    if any(value <= 0.0 for value in mechanism_values[oracle_field]):
        raise MacroGenerationError(
            "scientific release gate failed: OracleSpectralRatioGain must be "
            "strictly positive for every seed"
        )
    return {
        **{
            field: float(np.mean(values))
            for field, values in mechanism_values.items()
        },
        "parameters": parameters[0],
        "latency_ms_p50": float(np.median(latencies_p50)),
        "latency_ms_p95": float(np.median(latencies_p95)),
        "latency_device": next(iter(latency_devices)),
    }


def _mean_metric(
    metrics: dict[tuple[str, int, str], dict[str, float]],
    *,
    seeds: list[int],
    model: str,
    regime: str,
    metric: str,
) -> float:
    return float(
        np.mean([metrics[(model, seed, regime)][metric] for seed in seeds])
    )


def _clean_profile_macro_f1_means(
    bundles: dict[tuple[str, int, str], dict[str, Any]],
    *,
    seeds: list[int],
    profile_indices: tuple[int, ...],
    reference_model: str,
) -> tuple[float, float]:
    reference_values: list[float] = []
    method_values: list[float] = []
    for seed in seeds:
        reference_bundle = bundles[
            (reference_model, seed, "clean_retention")
        ]
        method_bundle = bundles[(METHOD_MODEL, seed, "clean_retention")]
        selected = np.isin(
            reference_bundle["target_profile_index"],
            np.asarray(profile_indices, dtype=np.int64),
        )
        if not selected.any():
            raise MacroGenerationError(
                "clean-retention public-table stratum is empty"
            )
        for bundle, destination in (
            (reference_bundle, reference_values),
            (method_bundle, method_values),
        ):
            macro_f1, _ = _macro_f1_and_accuracy(
                bundle["probabilities"][selected],
                bundle["labels"][selected],
                bundle["probabilities"].shape[1],
            )
            destination.append(macro_f1)
    return float(np.mean(reference_values)), float(np.mean(method_values))


def _validate_comparison_protocol(run_record: dict[str, Any]) -> str:
    comparison = run_record.get("comparison_protocol")
    if not isinstance(comparison, dict):
        raise MacroGenerationError("run comparison_protocol is malformed")
    reference_model = comparison.get("reference_model")
    if reference_model != PRIMARY_REFERENCE_MODEL:
        raise MacroGenerationError(
            "run paired reference drifted from the preregistered "
            f"{PRIMARY_REFERENCE_MODEL}"
        )
    exact_fields = {
        "reference_selection": "explicit_cli",
        "reference_strength_claimed": False,
        "primary_reference_predeclared": True,
        "method_model": METHOD_MODEL,
        "required_nonoracle_baselines": list(BASELINE_MODELS),
        "clean_retention_profile_strata": {
            name: list(indices)
            for name, indices in CLEAN_PROFILE_REGIMES.items()
        },
        "scientific_release_thresholds": SCIENTIFIC_RELEASE_THRESHOLDS,
        "holm_candidate_family": list(FORMAL_HOLM_CANDIDATES),
    }
    drift = {
        field: {
            "actual": comparison.get(field),
            "expected": expected,
        }
        for field, expected in exact_fields.items()
        if comparison.get(field) != expected
    }
    if drift:
        raise MacroGenerationError(
            "run comparison_protocol drifted from the scientific release "
            f"preregistration: {drift}"
        )
    if PRIMARY_REFERENCE_MODEL in FORMAL_HOLM_CANDIDATES:
        raise MacroGenerationError(
            "internal release contract error: CSSL primary reference entered "
            "the Holm candidate family"
        )
    return str(reference_model)


def _scientific_release_gate(
    *,
    run_record: dict[str, Any],
    metrics: dict[tuple[str, int, str], dict[str, float]],
    bundles: dict[tuple[str, int, str], dict[str, Any]],
    headline: dict[tuple[str, str], dict[str, float]],
    method_summary: dict[str, float | int | str],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in run_record["seeds"]]
    hard_method = _mean_metric(
        metrics,
        seeds=seeds,
        model=METHOD_MODEL,
        regime=HARD_REGIME,
        metric="macro_f1",
    )
    hard_gains_pp: dict[str, float] = {}
    for baseline in BASELINE_MODELS:
        metric_gain = hard_method - _mean_metric(
            metrics,
            seeds=seeds,
            model=baseline,
            regime=HARD_REGIME,
            metric="macro_f1",
        )
        bundle_gain = float(
            np.mean(
                [
                    bundles[(METHOD_MODEL, seed, HARD_REGIME)]["macro_f1"]
                    - bundles[(baseline, seed, HARD_REGIME)]["macro_f1"]
                    for seed in seeds
                ]
            )
        )
        if not _same_number(metric_gain, bundle_gain):
            raise MacroGenerationError(
                "hard-regime gate disagrees between metrics.csv and "
                f"prediction NPZ for {baseline}"
            )
        hard_gains_pp[baseline] = metric_gain * 100.0
    minimum_hard_gain = float(
        SCIENTIFIC_RELEASE_THRESHOLDS[
            "hard_macro_f1_min_gain_pp_each_baseline"
        ]
    )
    failed_hard = {
        model: gain
        for model, gain in hard_gains_pp.items()
        if gain + FLOAT_ABS_TOLERANCE < minimum_hard_gain
    }
    if failed_hard:
        raise MacroGenerationError(
            "scientific release gate failed: A5 hard macro-F1 gain must be "
            f">= {minimum_hard_gain:.2f} pp versus every non-oracle baseline; "
            f"failed={failed_hard}"
        )

    hard_ablation_gains_pp: dict[str, float] = {}
    for control in ABLATION_CONTROL_MODELS:
        gain = float(
            np.mean(
                [
                    bundles[(METHOD_MODEL, seed, HARD_REGIME)]["macro_f1"]
                    - bundles[(control, seed, HARD_REGIME)]["macro_f1"]
                    for seed in seeds
                ]
            )
        )
        hard_ablation_gains_pp[control] = gain * 100.0
    failed_ablation = {
        model: gain
        for model, gain in hard_ablation_gains_pp.items()
        if gain <= 0.0
    }
    if failed_ablation:
        raise MacroGenerationError(
            "scientific release gate failed: A5 hard macro-F1 must be "
            f"strictly greater than A1 and A6; failed={failed_ablation}"
        )

    ood_gains_pp = {
        regime: headline[(METHOD_MODEL, regime)]["macro_f1_difference"] * 100.0
        for regime in OOD_GATE_REGIMES
    }
    ood_threshold = float(
        SCIENTIFIC_RELEASE_THRESHOLDS["ood_macro_f1_min_gain_pp"]
    )
    ood_pass_count = sum(
        gain + FLOAT_ABS_TOLERANCE >= ood_threshold
        for gain in ood_gains_pp.values()
    )
    required_ood_pass_count = int(
        SCIENTIFIC_RELEASE_THRESHOLDS["ood_required_pass_count"]
    )
    if ood_pass_count < required_ood_pass_count:
        raise MacroGenerationError(
            "scientific release gate failed: at least "
            f"{required_ood_pass_count}/{len(OOD_GATE_REGIMES)} OOD regimes "
            f"must gain >= {ood_threshold:.2f} pp versus CSSL; "
            f"actual={ood_gains_pp}"
        )

    clean_gates: dict[str, dict[str, float]] = {}
    point_threshold = float(
        SCIENTIFIC_RELEASE_THRESHOLDS[
            "clean_macro_f1_min_point_gain_pp"
        ]
    )
    ci_threshold = float(
        SCIENTIFIC_RELEASE_THRESHOLDS[
            "clean_macro_f1_min_ci95_low_pp"
        ]
    )
    for regime in CLEAN_PROFILE_REGIMES:
        row = headline[(METHOD_MODEL, regime)]
        point = row["macro_f1_difference"] * 100.0
        ci_low = row["macro_f1_ci95_low"] * 100.0
        clean_gates[regime] = {"gain_pp": point, "ci95_low_pp": ci_low}
        if (
            point + FLOAT_ABS_TOLERANCE < point_threshold
            or ci_low + FLOAT_ABS_TOLERANCE < ci_threshold
        ):
            raise MacroGenerationError(
                "scientific release gate failed: clean-retention "
                f"{regime} requires point >= {point_threshold:.2f} pp and "
                f"CI lower >= {ci_threshold:.2f} pp; "
                f"actual point={point:.6g}, CI lower={ci_low:.6g}"
            )

    mechanism_means = {
        field: float(method_summary[field])
        for field in SCIENTIFIC_RELEASE_THRESHOLDS[
            "mechanism_required_finite_fields"
        ]
    }
    return {
        "passed": True,
        "hard_gain_pp_each_nonoracle_baseline": hard_gains_pp,
        "hard_ablation_gain_pp": hard_ablation_gains_pp,
        "ood_gain_pp": ood_gains_pp,
        "ood_pass_count": ood_pass_count,
        "clean_noninferiority": clean_gates,
        "mechanism_means": mechanism_means,
    }


def _format_pp(value: float) -> str:
    return f"{value * 100.0:+.2f}"


def _derive_manifest(
    run_json: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = run_json.expanduser().resolve()
    try:
        run_record, _, _ = validate_release.validate_source_run(resolved)
    except validate_release.ReleaseValidationError as error:
        raise MacroGenerationError(
            f"source run failed release validation: {error}"
        ) from error
    models = run_record.get("models")
    seed_values = run_record.get("seeds")
    splits = run_record.get("splits")
    if (
        not isinstance(models, list)
        or not isinstance(seed_values, list)
        or not isinstance(splits, dict)
    ):
        raise MacroGenerationError("run model/seed/split contract is malformed")
    seeds = [int(seed) for seed in seed_values]
    required_models = (
        set(BASELINE_MODELS)
        | set(ABLATION_CONTROL_MODELS)
        | {METHOD_MODEL}
    )
    missing_models = sorted(required_models.difference(models))
    if missing_models:
        raise MacroGenerationError(
            "run is missing release models: " + ",".join(missing_models)
        )
    required_regimes = {
        HARD_REGIME,
        "unseen_jammer",
        "unseen_speed",
        "heldout_channel",
        "combined_ood",
        "clean_retention",
    }
    missing_regimes = sorted(required_regimes.difference(splits))
    if missing_regimes:
        raise MacroGenerationError(
            "run is missing release regimes: " + ",".join(missing_regimes)
        )
    reference_model = _validate_comparison_protocol(run_record)

    results = _result_index(run_record)
    metrics = _audit_metrics_csv(resolved, run_record, results)
    bundles, sample_counts = _audit_required_predictions(
        resolved,
        run_record,
        metrics,
        reference_model,
    )
    headline = _audit_headline_csv(
        resolved,
        run_record,
        metrics,
        bundles,
        sample_counts,
        reference_model,
    )
    method_summary = _method_summary(run_record, results)
    scientific_gate = _scientific_release_gate(
        run_record=run_record,
        metrics=metrics,
        bundles=bundles,
        headline=headline,
        method_summary=method_summary,
    )

    macros: dict[str, dict[str, str]] = {
        "PrimaryReference": {
            "value": MODEL_LABELS[reference_model],
            "source_artifact": "run.json",
            "derivation": (
                "Exact comparison_protocol.reference_model label; CSSL-AMC "
                "is the prospectively frozen paired reference and no "
                "post-hoc baseline ranking is inferred"
            ),
        }
    }
    for model_token, model in HEADLINE_MODELS.items():
        for metric, suffix in HEADLINE_METRIC_SUFFIXES.items():
            mean_value = _mean_metric(
                metrics,
                seeds=seeds,
                model=model,
                regime=HARD_REGIME,
                metric=metric,
            )
            value = (
                f"{mean_value * 100.0:.2f}"
                if metric in {"accuracy", "macro_f1", "worst_recall"}
                else f"{mean_value:.4f}"
            )
            macros[f"HeadlineHard{model_token}{suffix}"] = {
                "value": value,
                "source_artifact": "metrics.csv",
                "derivation": (
                    f"Arithmetic mean across formal seeds of {model}/{metric} "
                    "on hard_interference; accuracy, macro-F1, and worst-class "
                    "recall are scaled to percent while NLL and ECE retain "
                    "their recorded units"
                ),
            }

    for regime_token, regime in OOD_REGIMES.items():
        if regime in CLEAN_PROFILE_REGIMES:
            reference_mean, method_mean = _clean_profile_macro_f1_means(
                bundles,
                seeds=seeds,
                profile_indices=CLEAN_PROFILE_REGIMES[regime],
                reference_model=reference_model,
            )
            point_source = "headline_paired_statistics.csv"
            point_derivation = (
                "Arithmetic mean across formal seeds of profile-stratified "
                "clean_retention macro-F1 recomputed from source-aligned "
                "prediction NPZ bundles"
            )
        else:
            reference_mean = _mean_metric(
                metrics,
                seeds=seeds,
                model=reference_model,
                regime=regime,
                metric="macro_f1",
            )
            method_mean = _mean_metric(
                metrics,
                seeds=seeds,
                model=METHOD_MODEL,
                regime=regime,
                metric="macro_f1",
            )
            point_source = "metrics.csv"
            point_derivation = (
                "Arithmetic mean across formal seeds of macro-F1 from the "
                "audited model/seed/regime metric matrix"
            )
        row = headline[(METHOD_MODEL, regime)]
        if not _same_number(
            method_mean - reference_mean,
            row["macro_f1_difference"],
        ):
            raise MacroGenerationError(
                f"OOD table point estimates disagree with the hierarchical "
                f"paired row for {regime}"
            )
        prefix = f"Regime{regime_token}"
        for name, value, role in (
            (f"{prefix}Reference", reference_mean, "primary CSSL reference"),
            (f"{prefix}AFive", method_mean, "A5 VIMD method"),
        ):
            macros[name] = {
                "value": f"{value * 100.0:.2f}",
                "source_artifact": point_source,
                "derivation": f"{point_derivation}; value is the {role} mean",
            }
        for suffix, field in (
            ("Gain", "macro_f1_difference"),
            ("CILow", "macro_f1_ci95_low"),
            ("CIHigh", "macro_f1_ci95_high"),
        ):
            macros[f"{prefix}{suffix}"] = {
                "value": _format_pp(row[field]),
                "source_artifact": "headline_paired_statistics.csv",
                "derivation": (
                    f"100 times {field} for A5 versus the predeclared CSSL "
                    f"reference on {regime}; the hierarchical resampling unit "
                    "is algorithm seed and class-stratified source cluster"
                ),
            }

    for macro_name, field in MECHANISM_MACRO_FIELDS.items():
        exported_value = float(method_summary[field])
        if macro_name == "MechanismTargetAmplificationShare":
            exported_value *= 100.0
        macros[macro_name] = {
            "value": f"{exported_value:.6f}",
            "source_artifact": "run.json",
            "derivation": (
                f"Arithmetic mean across formal A5 seeds of mechanism.{field} "
                "from the heldout_channel diagnostic; the oracle spectral "
                "ratio is not a waveform SIR or SDR claim; amplification "
                "share is scaled to percent"
            ),
        }
    macros.update(
        {
            "VIMDParameters": {
                "value": str(int(method_summary["parameters"])),
                "source_artifact": "run.json",
                "derivation": (
                    "Exact positive A5 complexity.parameters count, required "
                    "to be identical across all formal seeds"
                ),
            },
            "VIMDLatencyPFifty": {
                "value": f"{float(method_summary['latency_ms_p50']):.3f}",
                "source_artifact": "run.json",
                "derivation": (
                    "Median across formal seeds of A5 latency_ms_p50 on the "
                    "single frozen latency_device"
                ),
            },
            "VIMDLatencyPNinetyFive": {
                "value": f"{float(method_summary['latency_ms_p95']):.3f}",
                "source_artifact": "run.json",
                "derivation": (
                    "Median across formal seeds of A5 latency_ms_p95 on the "
                    "single frozen latency_device"
                ),
            },
            "VIMDLatencyDevice": {
                "value": str(method_summary["latency_device"]),
                "source_artifact": "run.json",
                "derivation": (
                    "Exact A5 complexity.latency_device token, required to be "
                    "identical across every formal seed"
                ),
            },
        }
    )
    if set(macros) != set(PROVENANCE_MACROS):
        raise MacroGenerationError(
            "derived macro set disagrees with the generator provenance contract"
        )
    release_contract = getattr(validate_release, "PROVENANCE_MACROS", None)
    if release_contract is not None and set(macros) != set(release_contract):
        raise MacroGenerationError(
            "derived macro set disagrees with the release provenance contract"
        )
    manifest = {
        "schema_version": validate_release.MACRO_MANIFEST_SCHEMA,
        "run_id": run_record["run_id"],
        "cache_digest": run_record["cache_digest"],
        "run_json_sha256": validate_release.sha256_file(resolved),
        "scientific_release_gate": scientific_gate,
        "macros": macros,
    }
    return manifest, run_record


def generate_macro_manifest(run_json: Path) -> dict[str, Any]:
    """Return a deterministic manifest without writing any file."""

    manifest, _ = _derive_manifest(run_json)
    return manifest


def write_macro_manifest(
    *,
    run_json: Path,
    output: Path,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Validate and atomically write a deterministic macro manifest."""

    resolved_run = run_json.expanduser().resolve()
    destination = output.expanduser().resolve()
    try:
        destination.relative_to(resolved_run.parent)
    except ValueError:
        pass
    else:
        raise MacroGenerationError(
            "macro manifest output must be outside the immutable run directory"
        )
    if destination.exists() and not replace_existing:
        raise MacroGenerationError(
            f"output already exists: {destination}; pass --replace-existing"
        )
    manifest, run_record = _derive_manifest(resolved_run)
    encoded = validate_release.canonical_macro_manifest_text(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            validate_release.validate_macro_manifest(
                temporary,
                run_json=resolved_run,
                run_record=run_record,
            )
        except validate_release.ReleaseValidationError as error:
            raise MacroGenerationError(
                f"generated manifest failed release validation: {error}"
            ) from error
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "ok": True,
        "action": "macro_manifest_written",
        "submission_unlocked": False,
        "run_id": run_record["run_id"],
        "macro_count": len(manifest["macros"]),
        "output": str(destination),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="intentionally replace an existing manifest after full validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        result = write_macro_manifest(
            run_json=arguments.run_json,
            output=arguments.output,
            replace_existing=arguments.replace_existing,
        )
    except MacroGenerationError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "submission_unlocked": False,
                    "error": str(error),
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
