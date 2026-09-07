"""Canonical TVT-v2 scientific-evidence to paper-release bridge.

The v2 scientific gate deliberately leaves ``submission_unlocked`` false.
This module is the only supported next stage.  It deterministically rebuilds
every manuscript macro from the exact prediction/result artifacts, binds the
macro manifest to the freeze, learning curve, cache, run, and scientific gate,
and can then atomically publish ``results_auto.tex`` plus a portable release
lock.

No performance value is accepted on the command line.  A supplied macro
manifest must be byte-for-byte equal to deterministic re-derivation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.run_formal_tvt_v2 import (  # noqa: E402
    bound_scientific_gate_payload,
    runner,
)
from tvt_submission.formal_v2_contract import (  # noqa: E402
    DEFAULT_FREEZE,
    EXPECTED_OOD_AXES,
    EXPECTED_RUN_ID,
    EXPECTED_STRESS_AXES,
    load_contract,
    loaded_freeze_identity,
)
from tvt_submission.validate_v2_release import (  # noqa: E402
    GATE_SCHEMA,
    _load_bundle,
    derive_gate,
)
from vimd_amc.metrics import (  # noqa: E402
    PredictionBundle,
    ablation_family_paired_bootstrap,
    classification_metrics,
    headline_paired_bootstrap,
)


MACRO_MANIFEST_SCHEMA = "vimd_amc.tvt.paper_macro_values.v4"
RELEASE_LOCK_SCHEMA = "vimd_amc.tvt.release_lock.v3"
FORMAL_DESIGNATION = "headline_formal_tvt_evidence_v2"
RELEASE_SENTINEL = "EligibleLockedResults"
RELEASE_SENTINEL_VALUE = "eligible_locked_formal_v2_run"
METHOD_MODEL = "a5_vimd_full"
A0_MODEL = "a0_backbone"
PRIMARY_REFERENCE_MODEL = "cssl_amc_supervised_adaptation"
HARD_REGIME = "hard_interference"

HEADLINE_MODELS = {
    "AZero": A0_MODEL,
    "MCLDNN": "mcldnn_reimplementation",
    "IQFormer": "iqformer_inspired",
    "CSSL": PRIMARY_REFERENCE_MODEL,
    "AFive": METHOD_MODEL,
}
HEADLINE_MODEL_TOKENS = tuple(HEADLINE_MODELS)
HEADLINE_METRICS = {
    "Accuracy": "accuracy",
    "MacroFOne": "macro_f1",
    "WorstRecall": "worst_recall",
    "NLL": "nll",
    "ECE": "ece",
}
HEADLINE_METRIC_TOKENS = tuple(HEADLINE_METRICS)
HEADLINE_PROVENANCE_MACROS = tuple(
    f"HeadlineHard{model}{metric}"
    for model in HEADLINE_MODEL_TOKENS
    for metric in HEADLINE_METRIC_TOKENS
)

REGIME_TOKENS = (
    "Hard",
    "UnseenJammer",
    "UnseenSpeed",
    "HeldoutChannel",
    "CombinedOOD",
    "CleanACD",
    "CleanBE",
    "ADCTen",
    "ADCTwelve",
    "EmitterSync",
)
REGIME_SPECS = {
    "Hard": ("hard_interference", A0_MODEL, "AZero"),
    "UnseenJammer": ("unseen_jammer", A0_MODEL, "AZero"),
    "UnseenSpeed": ("unseen_speed", A0_MODEL, "AZero"),
    "HeldoutChannel": ("heldout_channel", A0_MODEL, "AZero"),
    "CombinedOOD": ("combined_ood", A0_MODEL, "AZero"),
    "CleanACD": ("clean_retention", PRIMARY_REFERENCE_MODEL, "CSSL"),
    "CleanBE": ("clean_retention", PRIMARY_REFERENCE_MODEL, "CSSL"),
    "ADCTen": ("adc_10bit_agc", A0_MODEL, "AZero"),
    "ADCTwelve": ("adc_12bit_agc", A0_MODEL, "AZero"),
    "EmitterSync": ("per_emitter_sync", A0_MODEL, "AZero"),
}
REGIME_PROVENANCE_MACROS = tuple(
    name
    for token in REGIME_TOKENS
    for name in (
        f"Regime{token}{REGIME_SPECS[token][2]}",
        f"Regime{token}AFive",
        f"Regime{token}Gain",
        f"Regime{token}CILow",
        f"Regime{token}CIHigh",
    )
)

ABLATION_MEAN_MODELS = {
    "HeadlineHardAOneMacroFOne": "a1_single_mask",
    "HeadlineHardATwoMacroFOne": "a2_tri_no_teacher",
    "HeadlineHardAThreePrimeMacroFOne": (
        "a3p_tri_proportional_teacher"
    ),
    "HeadlineHardAThreeMacroFOne": "a3_tri_teacher",
    "HeadlineHardAFourMacroFOne": "a4_tri_teacher_mtl",
    "HeadlineHardASixMacroFOne": "a6_dual_full",
    "HeadlineHardASevenMacroFOne": "a7_vimd_no_residual",
}
ABLATION_MEAN_PROVENANCE_MACROS = tuple(ABLATION_MEAN_MODELS)
ABLATION_CONTRASTS = {
    "full_vs_backbone": (
        "AblationFullVsBackbone",
        A0_MODEL,
        METHOD_MODEL,
        True,
    ),
    "margin_vs_proportional_teacher": (
        "AblationMarginVsProportional",
        "a3p_tri_proportional_teacher",
        "a3_tri_teacher",
        True,
    ),
    "tri_vs_dual_route": (
        "AblationFullVsDual",
        "a6_dual_full",
        METHOD_MODEL,
        True,
    ),
    "teacher_presence": (
        "AblationTeacher",
        "a2_tri_no_teacher",
        "a3_tri_teacher",
        False,
    ),
    "multitask_bundle": (
        "AblationMultitask",
        "a3_tri_teacher",
        "a4_tri_teacher_mtl",
        False,
    ),
    "cross_condition_contrast": (
        "AblationExactSourceContrast",
        "a4_tri_teacher_mtl",
        METHOD_MODEL,
        False,
    ),
    "full_vs_single_mask": (
        "AblationFullVsSingle",
        "a1_single_mask",
        METHOD_MODEL,
        False,
    ),
    "bounded_bypass": (
        "AblationBypass",
        "a7_vimd_no_residual",
        METHOD_MODEL,
        False,
    ),
}
ABLATION_CONTRAST_PREFIXES = tuple(
    record[0] for record in ABLATION_CONTRASTS.values()
)
ABLATION_CONTRAST_PROVENANCE_MACROS = tuple(
    f"{prefix}{suffix}"
    for prefix in ABLATION_CONTRAST_PREFIXES
    for suffix in ("Gain", "CILow", "CIHigh")
)

MECHANISM_FIELDS = {
    "MechanismMaskJS": "mask_js",
    "MechanismThirdRouteWeightedCorrelation": (
        "overlap_uncertainty_route_weighted_correlation"
    ),
    "MechanismTargetTransferRatio": (
        "target_energy_transfer_ratio_mean"
    ),
    "MechanismTargetAmplificationShare": (
        "target_energy_transfer_ratio_amplification_share"
    ),
    "MechanismJammerLeakage": "jammer_leakage",
    "MechanismThirdRouteSpearman": (
        "oracle_vs_predicted_overlap_spearman"
    ),
    "MechanismThirdRoutePermutationP": (
        "overlap_permutation_p_value"
    ),
    "OracleSpectralRatioGain": "counterfactual_tf_sir_gain_db",
}
MECHANISM_PROVENANCE_MACROS = (
    *tuple(MECHANISM_FIELDS),
    "MechanismOccupancyGainSpearman",
    "MechanismOccupancyGainPermutationP",
)
PROVENANCE_MACROS = (
    "PrimaryReference",
    *HEADLINE_PROVENANCE_MACROS,
    *REGIME_PROVENANCE_MACROS,
    *ABLATION_MEAN_PROVENANCE_MACROS,
    *ABLATION_CONTRAST_PROVENANCE_MACROS,
    *MECHANISM_PROVENANCE_MACROS,
    "VIMDParameters",
    "VIMDLatencyPFifty",
    "VIMDLatencyPNinetyFive",
    "VIMDLatencyDevice",
)
RESULT_MACROS = PROVENANCE_MACROS
NON_SENTINEL_RESULT_MACROS = ("ResultSource", *PROVENANCE_MACROS)
ALL_MACROS = (RELEASE_SENTINEL, *NON_SENTINEL_RESULT_MACROS)

MODEL_LABELS = {
    PRIMARY_REFERENCE_MODEL: (
        "CSSL-AMC official-architecture supervised adaptation"
    ),
}
RESULTS_AUTO_COMMENTS = (
    "% AUTO-GENERATED by tvt_submission/validate_v2_paper_release.py.",
    "% Do not edit reported values by hand; release_lock.json binds source.",
)
PLACEHOLDER_TERMS = (
    "pending",
    "generated",
    "placeholder",
    "tbd",
    "todo",
    "unavailable",
    "n/a",
    "nan",
    "no eligible locked run",
    "internal review",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_RELATIVE_SOURCE = re.compile(
    r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)
MACRO_LINE = re.compile(
    r"^\s*\\newcommand\{\\(?P<name>[A-Za-z]+)\}"
    r"\{(?P<value>.*)\}\s*$"
)
ATOMIC_DECIMAL = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
DANGEROUS_TEX = re.compile(
    r"[\\{}%#&\r\n]"
)
RELEASE_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "submission_unlocked",
        "paper_release_unlocked",
        "scientific_evidence_passed",
        "generated_utc",
        "run_id",
        "cache_digest",
        "formal_cache_designation",
        "release_sentinel_name",
        "release_sentinel_value",
        "run_json_sha256",
        "scientific_gate_sha256",
        "learning_curve_evidence_sha256",
        "freeze_sha256",
        "cache_manifest_sha256",
        "macro_value_manifest_sha256",
        "results_auto_sha256",
        "source_inventory_sha256",
        "source_inventory",
        "macro_provenance",
    }
)


class ReleaseValidationError(RuntimeError):
    """A v2 paper-release invariant failed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def load_strict_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseValidationError(
            f"could not read strict JSON {resolved}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ReleaseValidationError(
            f"JSON root must be an object: {resolved}"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def canonical_macro_manifest_text(manifest: dict[str, Any]) -> str:
    return _canonical_json(manifest)


def canonical_release_lock_text(lock: dict[str, Any]) -> str:
    return _canonical_json(lock)


def _is_placeholder(value: str) -> bool:
    normalized = " ".join(value.strip().casefold().split())
    if not normalized or normalized in {
        "-",
        "--",
        "---",
        "–",
        "—",
        r"\textemdash",
    }:
        return True
    return any(term in normalized for term in PLACEHOLDER_TERMS)


def _validate_macro_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ReleaseValidationError(
            f"macro {name} must be a trimmed string"
        )
    if _is_placeholder(value):
        raise ReleaseValidationError(f"macro {name} is a placeholder")
    if DANGEROUS_TEX.search(value):
        raise ReleaseValidationError(
            f"macro {name} contains prohibited TeX/control characters"
        )
    if len(value) > 160:
        raise ReleaseValidationError(f"macro {name} is unreasonably long")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReleaseValidationError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_run_source(run_root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or SAFE_RUN_RELATIVE_SOURCE.fullmatch(relative) is None
        or any(part in {".", ".."} for part in relative.split("/"))
    ):
        raise ReleaseValidationError(
            f"invalid run-relative source path: {relative!r}"
        )
    resolved = (run_root / Path(relative)).resolve()
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as error:
        raise ReleaseValidationError(
            f"source escapes the formal run: {relative}"
        ) from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ReleaseValidationError(
            f"macro source is missing or empty: {relative}"
        )
    return resolved


def _bundle_relative(model: str, seed: int, regime: str) -> str:
    return (
        f"models/{model}_seed{seed}/predictions_{regime}.npz"
    )


def _result_relative(model: str, seed: int) -> str:
    return f"models/{model}_seed{seed}/result.json"


def _format_pp(value: float) -> str:
    if not math.isfinite(value):
        raise ReleaseValidationError("cannot format a nonfinite estimate")
    return f"{100.0 * value:+.2f}"


def _format_percent(value: float) -> str:
    if not math.isfinite(value):
        raise ReleaseValidationError("cannot format a nonfinite estimate")
    return f"{100.0 * value:.2f}"


def _mean(values: list[float], label: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ReleaseValidationError(f"{label} is empty or nonfinite")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _derive_macro_records(
    *,
    run_json: Path,
    run: dict[str, Any],
    gate: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    run_root = run_json.parent.resolve()
    experiment = contract["experiment"]
    seeds = [int(value) for value in experiment["seeds"]]
    cache_digest = str(run["cache_digest"])
    classes = int(run.get("num_classes", 0))
    if classes <= 0:
        raise ReleaseValidationError("run num_classes is invalid")
    bundle_cache: dict[tuple[str, int, str], PredictionBundle] = {}
    digest_cache: dict[str, str] = {}

    def bundle(model: str, seed: int, regime: str) -> PredictionBundle:
        key = (model, seed, regime)
        if key not in bundle_cache:
            bundle_cache[key] = _load_bundle(
                run_root,
                model,
                seed,
                regime,
                cache_digest,
            )
            if bundle_cache[key].probabilities.shape[1] != classes:
                raise ReleaseValidationError(
                    f"class taxonomy drift for {model}/seed{seed}/{regime}"
                )
        return bundle_cache[key]

    def sources(
        models: tuple[str, ...] | list[str],
        regime: str,
    ) -> list[str]:
        return [
            _bundle_relative(model, seed, regime)
            for model in models
            for seed in seeds
        ]

    def source_records(paths: list[str]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for relative in sorted(set(paths)):
            path = _safe_run_source(run_root, relative)
            if relative not in digest_cache:
                digest_cache[relative] = sha256_file(path)
            records.append(
                {
                    "path": relative,
                    "sha256": digest_cache[relative],
                }
            )
        if not records:
            raise ReleaseValidationError("macro provenance is empty")
        return records

    def record(
        value: str,
        source_paths: list[str],
        derivation: str,
    ) -> dict[str, Any]:
        if len(derivation) < 24 or "\n" in derivation:
            raise ReleaseValidationError(
                "macro derivation is missing or non-canonical"
            )
        return {
            "value": value,
            "sources": source_records(source_paths),
            "derivation": derivation,
        }

    def metric_mean(model: str, regime: str, metric: str) -> float:
        return _mean(
            [
                float(
                    classification_metrics(
                        bundle(model, seed, regime),
                        classes,
                    )[metric]
                )
                for seed in seeds
            ],
            f"{model}/{regime}/{metric}",
        )

    def subset_macro_f1_mean(
        model: str,
        regime: str,
        selected_profiles: list[int],
    ) -> float:
        values: list[float] = []
        for seed in seeds:
            current = bundle(model, seed, regime)
            if current.target_profile_index is None:
                raise ReleaseValidationError(
                    f"{regime} lacks target_profile_index"
                )
            selected = np.isin(
                current.target_profile_index,
                np.asarray(selected_profiles, dtype=np.int64),
            )
            if not selected.any():
                raise ReleaseValidationError(
                    f"{regime} profile stratum is empty"
                )
            values.append(
                float(
                    classification_metrics(
                        current.subset(selected),
                        classes,
                    )["macro_f1"]
                )
            )
        return _mean(values, f"{model}/{regime}/profile_macro_f1")

    macros: dict[str, dict[str, Any]] = {
        "PrimaryReference": record(
            MODEL_LABELS[PRIMARY_REFERENCE_MODEL],
            ["run.json"],
            "Exact preregistered comparison reference from the v2 run; no "
            "post-hoc baseline winner is selected",
        )
    }

    for token, model in HEADLINE_MODELS.items():
        hard_sources = sources([model], HARD_REGIME)
        for suffix, metric in HEADLINE_METRICS.items():
            mean_value = metric_mean(model, HARD_REGIME, metric)
            value = (
                _format_percent(mean_value)
                if metric in {"accuracy", "macro_f1", "worst_recall"}
                else f"{mean_value:.4f}"
            )
            macros[f"HeadlineHard{token}{suffix}"] = record(
                value,
                hard_sources,
                "Arithmetic mean across the ten frozen algorithm seeds, "
                f"recomputed from {model} hard-interference prediction "
                f"probabilities for {metric}",
            )

    for name, model in ABLATION_MEAN_MODELS.items():
        macros[name] = record(
            _format_percent(metric_mean(model, HARD_REGIME, "macro_f1")),
            sources([model], HARD_REGIME),
            "Arithmetic mean across the ten frozen seeds of hard-interference "
            f"macro-F1 recomputed from {model} prediction bundles",
        )

    confirmatory = gate.get("confirmatory_family")
    if (
        not isinstance(confirmatory, dict)
        or confirmatory.get("passed") is not True
        or not isinstance(confirmatory.get("contrasts"), dict)
    ):
        raise ReleaseValidationError(
            "scientific gate lacks a passing rederived confirmatory family"
        )
    confirmatory_values = confirmatory["contrasts"]
    confirmatory_specs = {
        item["contrast_id"]: item
        for item in experiment["confirmatory_family"]["contrasts"]
    }
    if set(confirmatory_values) != set(confirmatory_specs):
        raise ReleaseValidationError(
            "confirmatory gate membership differs from the freeze"
        )

    exploratory_specs = {
        str(item["contrast_id"]): (
            str(item["reference"]),
            str(item["candidate"]),
        )
        for item in experiment["exploratory_contrasts"]
    }
    expected_exploratory = {
        contrast_id: (reference, candidate)
        for contrast_id, (
            _,
            reference,
            candidate,
            is_confirmatory,
        ) in ABLATION_CONTRASTS.items()
        if not is_confirmatory
    }
    if exploratory_specs != expected_exploratory:
        raise ReleaseValidationError(
            "exploratory contrast contract differs from the macro adapter"
        )
    exploratory_models = sorted(
        {
            model
            for pair in exploratory_specs.values()
            for model in pair
        }
    )
    exploratory_bundles = {
        model: {
            seed: bundle(model, seed, HARD_REGIME)
            for seed in seeds
        }
        for model in exploratory_models
    }
    exploratory = ablation_family_paired_bootstrap(
        exploratory_bundles,
        exploratory_specs,
        draws=int(experiment["statistics"]["bootstrap_draws"]),
        seed=runner.analysis_seed(
            int(experiment["statistics"]["bootstrap_seed"]),
            "paper_exploratory_ablation_family",
            HARD_REGIME,
        ),
        confidence_level=0.95,
    )

    for contrast_id, (
        prefix,
        reference,
        candidate,
        is_confirmatory,
    ) in ABLATION_CONTRASTS.items():
        if is_confirmatory:
            values = confirmatory_values[contrast_id]
            point = float(values["gain"])
            low = float(values["simultaneous_ci95_low"])
            high = float(values["simultaneous_ci95_high"])
            interval_label = (
                "confirmatory family-wise simultaneous interval"
            )
            contrast_sources = [
                *sources([reference, candidate], HARD_REGIME),
                "ablation_paired_statistics.csv",
                "v2_scientific_release_gate.json",
            ]
        else:
            values = exploratory["contrasts"][contrast_id]
            point = float(values["macro_f1_difference"])
            low = float(values["macro_f1_marginal_ci95_low"])
            high = float(values["macro_f1_marginal_ci95_high"])
            interval_label = (
                "exploratory marginal hierarchical paired-bootstrap interval"
            )
            contrast_sources = sources(
                [reference, candidate],
                HARD_REGIME,
            )
        for suffix, value in (
            ("Gain", point),
            ("CILow", low),
            ("CIHigh", high),
        ):
            macros[f"{prefix}{suffix}"] = record(
                _format_pp(value),
                contrast_sources,
                f"A5-style candidate-minus-reference hard-interference "
                f"macro-F1 contrast {contrast_id}; {interval_label}; "
                "percentage-point scaling is applied exactly once",
            )

    combined_seed = runner.analysis_seed(
        int(experiment["statistics"]["bootstrap_seed"]),
        "paper_exploratory_regime",
        "combined_ood",
    )
    combined_statistics = headline_paired_bootstrap(
        {
            seed: bundle(A0_MODEL, seed, "combined_ood")
            for seed in seeds
        },
        {
            seed: bundle(METHOD_MODEL, seed, "combined_ood")
            for seed in seeds
        },
        draws=int(experiment["statistics"]["bootstrap_draws"]),
        seed=combined_seed,
    )

    clean_spec = experiment["scientific_release_gates"]["clean_retention"]
    clean_strata = clean_spec["profile_strata"]
    gate_clean = gate.get("clean_retention")
    if (
        not isinstance(gate_clean, dict)
        or gate_clean.get("all_strata_passed") is not True
        or set(gate_clean.get("strata", {})) != set(clean_strata)
    ):
        raise ReleaseValidationError(
            "scientific gate lacks both passing clean-retention strata"
        )

    confirmatory_full = confirmatory_values["full_vs_backbone"]
    axis_records = gate.get("axis_calibration")
    if (
        not isinstance(axis_records, dict)
        or set(axis_records)
        != set((*EXPECTED_OOD_AXES, *EXPECTED_STRESS_AXES))
    ):
        raise ReleaseValidationError(
            "scientific gate lacks the exact OOD/receiver axis family"
        )

    for token, (regime, reference, reference_suffix) in REGIME_SPECS.items():
        prefix = f"Regime{token}"
        if token == "CleanACD":
            stratum_name = "clean_retention_seen_acd"
        elif token == "CleanBE":
            stratum_name = "clean_retention_held_be"
        else:
            stratum_name = None

        if stratum_name is not None:
            profiles = list(clean_strata[stratum_name])
            reference_mean = subset_macro_f1_mean(
                reference,
                regime,
                profiles,
            )
            method_mean = subset_macro_f1_mean(
                METHOD_MODEL,
                regime,
                profiles,
            )
            statistics = gate_clean["strata"][stratum_name]
            point = float(statistics["candidate_minus_baseline"])
            low = float(statistics["ci95_low"])
            high = float(statistics["ci95_high"])
            regime_sources = [
                *sources([reference, METHOD_MODEL], regime),
                "v2_scientific_release_gate.json",
            ]
            inference = (
                "preregistered clean-retention profile-stratified "
                "hierarchical paired bootstrap"
            )
        else:
            reference_mean = metric_mean(reference, regime, "macro_f1")
            method_mean = metric_mean(METHOD_MODEL, regime, "macro_f1")
            if token == "Hard":
                point = float(confirmatory_full["gain"])
                low = float(confirmatory_full["simultaneous_ci95_low"])
                high = float(confirmatory_full["simultaneous_ci95_high"])
                regime_sources = [
                    *sources([reference, METHOD_MODEL], regime),
                    "ablation_paired_statistics.csv",
                    "v2_scientific_release_gate.json",
                ]
                inference = (
                    "confirmatory family-wise simultaneous interval"
                )
            elif token == "CombinedOOD":
                point = float(
                    combined_statistics["macro_f1_difference"]
                )
                low = float(
                    combined_statistics["macro_f1_ci95_low"]
                )
                high = float(
                    combined_statistics["macro_f1_ci95_high"]
                )
                regime_sources = sources(
                    [reference, METHOD_MODEL],
                    regime,
                )
                inference = (
                    "exploratory hierarchical paired-bootstrap interval"
                )
            else:
                statistics = axis_records[regime]
                if statistics.get("axis_gate_passed") is not True:
                    raise ReleaseValidationError(
                        f"axis gate did not pass: {regime}"
                    )
                point = float(
                    statistics[
                        "candidate_minus_a0_ood_macro_f1_gain"
                    ]
                )
                low = float(
                    statistics[
                        "candidate_minus_a0_ood_macro_f1_gain_ci95_low"
                    ]
                )
                high = float(
                    statistics[
                        "candidate_minus_a0_ood_macro_f1_gain_ci95_high"
                    ]
                )
                nominal = (
                    "id_test"
                    if regime in EXPECTED_OOD_AXES
                    else HARD_REGIME
                )
                regime_sources = [
                    *sources([A0_MODEL], nominal),
                    *sources([reference, METHOD_MODEL], regime),
                    "v2_scientific_release_gate.json",
                ]
                inference = (
                    "predeclared opportunity-calibrated hierarchical "
                    "paired-bootstrap axis interval"
                )
        if not math.isclose(
            method_mean - reference_mean,
            point,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ReleaseValidationError(
                f"regime point estimate disagrees with absolute means: {regime}"
            )
        macros[f"{prefix}{reference_suffix}"] = record(
            _format_percent(reference_mean),
            regime_sources,
            f"Arithmetic mean across the ten frozen seeds of {reference} "
            f"macro-F1 on {regime}, recomputed from prediction bundles",
        )
        macros[f"{prefix}AFive"] = record(
            _format_percent(method_mean),
            regime_sources,
            f"Arithmetic mean across the ten frozen seeds of {METHOD_MODEL} "
            f"macro-F1 on {regime}, recomputed from prediction bundles",
        )
        for suffix, value in (
            ("Gain", point),
            ("CILow", low),
            ("CIHigh", high),
        ):
            macros[f"{prefix}{suffix}"] = record(
                _format_pp(value),
                regime_sources,
                f"A5-minus-reference macro-F1 on {regime}; {inference}; "
                "percentage-point scaling is applied exactly once",
            )

    results = run.get("results")
    if not isinstance(results, list):
        raise ReleaseValidationError("run results are malformed")
    indexed = {
        (str(item.get("model")), int(item.get("seed"))): item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("seed"), int)
        and not isinstance(item.get("seed"), bool)
    }
    result_sources = [
        _result_relative(METHOD_MODEL, seed) for seed in seeds
    ]
    mechanism_values = {field: [] for field in MECHANISM_FIELDS.values()}
    parameters: list[int] = []
    latency_p50: list[float] = []
    latency_p95: list[float] = []
    devices: set[str] = set()
    for seed in seeds:
        result = indexed.get((METHOD_MODEL, seed))
        if not isinstance(result, dict):
            raise ReleaseValidationError(
                f"run lacks {METHOD_MODEL}/seed{seed}"
            )
        mechanism = result.get("mechanism")
        complexity = result.get("complexity")
        if not isinstance(mechanism, dict) or not isinstance(
            complexity,
            dict,
        ):
            raise ReleaseValidationError(
                f"A5 result lacks mechanism/complexity for seed {seed}"
            )
        for field in mechanism_values:
            try:
                value = float(mechanism[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ReleaseValidationError(
                    f"A5 mechanism {field} is missing for seed {seed}"
                ) from error
            if not math.isfinite(value):
                raise ReleaseValidationError(
                    f"A5 mechanism {field} is nonfinite for seed {seed}"
                )
            mechanism_values[field].append(value)
        parameter_value = float(complexity.get("parameters", 0.0))
        p50 = float(complexity.get("latency_ms_p50", 0.0))
        p95 = float(complexity.get("latency_ms_p95", 0.0))
        device = complexity.get("latency_device")
        if (
            not parameter_value.is_integer()
            or parameter_value <= 0
            or not math.isfinite(p50)
            or not math.isfinite(p95)
            or p50 <= 0
            or p95 < p50
            or not isinstance(device, str)
            or not device.strip()
            or DANGEROUS_TEX.search(device)
        ):
            raise ReleaseValidationError(
                f"A5 complexity is invalid for seed {seed}"
            )
        parameters.append(int(parameter_value))
        latency_p50.append(p50)
        latency_p95.append(p95)
        devices.add(device)
    if len(set(parameters)) != 1 or len(devices) != 1:
        raise ReleaseValidationError(
            "A5 parameter count or latency device differs across seeds"
        )

    for name, field in MECHANISM_FIELDS.items():
        value = _mean(mechanism_values[field], f"mechanism/{field}")
        if name == "MechanismTargetAmplificationShare":
            value *= 100.0
        macros[name] = record(
            f"{value:.6f}",
            result_sources,
            "Arithmetic mean across the ten frozen A5 result records of "
            f"mechanism.{field}; amplification share alone is scaled to "
            "percent and the oracle spectral ratio is not waveform SDR",
        )
    occupancy = gate.get("jammer_occupancy_directional_test")
    if not isinstance(occupancy, dict):
        raise ReleaseValidationError(
            "scientific gate lacks the occupancy/gain diagnostic"
        )
    occupancy_sources = [
        *sources([A0_MODEL, METHOD_MODEL], HARD_REGIME),
        "v2_scientific_release_gate.json",
    ]
    for name, field in (
        ("MechanismOccupancyGainSpearman", "spearman_rho"),
        (
            "MechanismOccupancyGainPermutationP",
            "exact_one_sided_permutation_p_value",
        ),
    ):
        value = float(occupancy[field])
        if not math.isfinite(value):
            raise ReleaseValidationError(
                f"occupancy/gain {field} is nonfinite"
            )
        macros[name] = record(
            f"{value:.6f}",
            occupancy_sources,
            "Frozen six-jammer-family periodic-Hann minus-20-dB support "
            f"occupancy/gain directional diagnostic field {field}",
        )
    macros.update(
        {
            "VIMDParameters": record(
                str(parameters[0]),
                result_sources,
                "Exact A5 parameter count, required to be identical across "
                "all ten formal fits",
            ),
            "VIMDLatencyPFifty": record(
                f"{float(np.median(latency_p50)):.3f}",
                result_sources,
                "Median across formal A5 fits of the recorded batch-one "
                "latency p50 on one frozen device",
            ),
            "VIMDLatencyPNinetyFive": record(
                f"{float(np.median(latency_p95)):.3f}",
                result_sources,
                "Median across formal A5 fits of the recorded batch-one "
                "latency p95 on one frozen device",
            ),
            "VIMDLatencyDevice": record(
                next(iter(devices)),
                result_sources,
                "Exact latency-device label, required to be identical across "
                "all ten formal A5 fits",
            ),
        }
    )

    if len(PROVENANCE_MACROS) != len(set(PROVENANCE_MACROS)):
        raise ReleaseValidationError(
            "v2 macro allowlist contains duplicate names"
        )
    if set(macros) != set(PROVENANCE_MACROS):
        raise ReleaseValidationError(
            "derived v2 macro set mismatch; missing="
            + ",".join(sorted(set(PROVENANCE_MACROS).difference(macros)))
            + "; unexpected="
            + ",".join(sorted(set(macros).difference(PROVENANCE_MACROS)))
        )
    for name in PROVENANCE_MACROS:
        _validate_macro_value(name, macros[name]["value"])
    return macros


def _canonical_gate_text(gate: dict[str, Any]) -> str:
    return _canonical_json(gate)


def generate_macro_manifest(
    *,
    run_json: Path,
    learning_evidence: Path,
    freeze: Path = DEFAULT_FREEZE,
) -> dict[str, Any]:
    resolved_run = run_json.expanduser().resolve()
    resolved_learning = learning_evidence.expanduser().resolve()
    resolved_freeze = freeze.expanduser().resolve()
    contract = load_contract(resolved_freeze)
    loaded_freeze_path, freeze_digest = loaded_freeze_identity(contract)
    if loaded_freeze_path != resolved_freeze:
        raise ReleaseValidationError(
            "loaded v2 contract path differs from the requested freeze"
        )
    learning = load_strict_json(resolved_learning)
    expected_scientific_gate = derive_gate(
        run_json=resolved_run,
        learning_evidence=resolved_learning,
        freeze=resolved_freeze,
        write=False,
    )
    if not (
        expected_scientific_gate.get("schema_version") == GATE_SCHEMA
        and expected_scientific_gate.get("passed") is True
        and expected_scientific_gate.get("scientific_evidence_passed")
        is True
        and expected_scientific_gate.get("submission_unlocked") is False
        and expected_scientific_gate.get("paper_release_unlocked") is False
        and expected_scientific_gate.get("reasons") == []
    ):
        raise ReleaseValidationError(
            "v2 scientific evidence gate is not eligible for paper promotion"
        )
    try:
        expected_gate = bound_scientific_gate_payload(
            gate=expected_scientific_gate,
            freeze_path=resolved_freeze,
            freeze_digest=freeze_digest,
            learning_evidence=learning,
        )
    except ValueError as error:
        raise ReleaseValidationError(
            f"could not bind the scientific gate to the freeze: {error}"
        ) from error
    gate_path = resolved_run.parent / "v2_scientific_release_gate.json"
    observed_gate = load_strict_json(gate_path)
    if observed_gate != expected_gate:
        raise ReleaseValidationError(
            "stored scientific gate differs from deterministic re-derivation"
        )
    try:
        observed_gate_text = gate_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseValidationError(
            f"could not read scientific gate bytes: {error}"
        ) from error
    if observed_gate_text != _canonical_gate_text(expected_gate):
        raise ReleaseValidationError(
            "scientific gate is not in canonical serialization"
        )
    run = load_strict_json(resolved_run)
    cache_manifest = Path(str(expected_gate["cache_manifest"])).resolve()
    if not cache_manifest.is_file():
        raise ReleaseValidationError("bound cache manifest is missing")
    bindings = {
        "run_json_sha256": sha256_file(resolved_run),
        "scientific_gate_sha256": sha256_file(gate_path),
        "learning_curve_evidence_sha256": sha256_file(resolved_learning),
        "freeze_sha256": sha256_file(resolved_freeze),
        "cache_manifest_sha256": sha256_file(cache_manifest),
    }
    if (
        bindings["run_json_sha256"]
        != expected_gate["run_json_sha256"]
        or bindings["learning_curve_evidence_sha256"]
        != expected_gate["learning_curve_evidence_sha256"]
        or bindings["cache_manifest_sha256"]
        != expected_gate["cache_manifest_sha256"]
        or bindings["freeze_sha256"]
        != expected_gate["freeze_sha256"]
        or expected_gate["learning_curve_freeze_sha256"]
        != bindings["freeze_sha256"]
        or expected_gate["freeze"]
        != str(resolved_freeze)
        or expected_gate["freeze_identity_verified"] is not True
    ):
        raise ReleaseValidationError(
            "scientific gate top-level source bindings drifted"
        )
    macros = _derive_macro_records(
        run_json=resolved_run,
        run=run,
        gate=expected_gate,
        contract=contract,
    )
    return {
        "schema_version": MACRO_MANIFEST_SCHEMA,
        "run_id": run["run_id"],
        "cache_digest": run["cache_digest"],
        "formal_cache_designation": FORMAL_DESIGNATION,
        "bindings": bindings,
        "scientific_gate_summary": {
            "schema_version": expected_gate["schema_version"],
            "scientific_evidence_passed": True,
            "submission_unlocked": False,
            "paper_release_unlocked": False,
        },
        "macros": macros,
        "manually_typed_performance_values_used": False,
    }


def validate_macro_manifest(
    path: Path,
    *,
    run_json: Path,
    learning_evidence: Path,
    freeze: Path = DEFAULT_FREEZE,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    supplied = load_strict_json(resolved)
    expected = generate_macro_manifest(
        run_json=run_json,
        learning_evidence=learning_evidence,
        freeze=freeze,
    )
    if supplied != expected:
        raise ReleaseValidationError(
            "macro manifest differs from deterministic artifact "
            "re-derivation; manually supplied values are prohibited"
        )
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseValidationError(
            f"could not read macro manifest bytes: {error}"
        ) from error
    if text != canonical_macro_manifest_text(expected):
        raise ReleaseValidationError(
            "macro manifest bytes are not the canonical serialization"
        )
    return expected


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
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
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_macro_manifest(
    *,
    run_json: Path,
    learning_evidence: Path,
    output: Path,
    freeze: Path = DEFAULT_FREEZE,
    replace_existing: bool = False,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists() and not replace_existing:
        raise ReleaseValidationError(
            "macro manifest already exists; pass --replace-existing-manifest "
            "for an intentional deterministic rebuild"
        )
    manifest = generate_macro_manifest(
        run_json=run_json,
        learning_evidence=learning_evidence,
        freeze=freeze,
    )
    _atomic_write_text(
        destination,
        canonical_macro_manifest_text(manifest),
    )
    validate_macro_manifest(
        destination,
        run_json=run_json,
        learning_evidence=learning_evidence,
        freeze=freeze,
    )
    return {
        "ok": True,
        "action": "v2_macro_manifest_written",
        "submission_unlocked": False,
        "macro_count": len(PROVENANCE_MACROS),
        "output": str(destination),
        "sha256": sha256_file(destination),
    }


def _escape_metadata(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("&", r"\&"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(source, replacement)
    return text


def render_results_auto(
    run_record: dict[str, Any],
    values: dict[str, str],
) -> str:
    if set(values) != set(PROVENANCE_MACROS):
        raise ReleaseValidationError(
            "render input does not contain the exact v2 macro set"
        )
    lines = [
        *RESULTS_AUTO_COMMENTS,
        (
            rf"\newcommand{{\{RELEASE_SENTINEL}}}"
            rf"{{{RELEASE_SENTINEL_VALUE}}}"
        ),
        (
            rf"\newcommand{{\ResultSource}}"
            rf"{{{result_source_value(run_record)}}}"
        ),
    ]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{_validate_macro_value(name, values[name])}}}"
        for name in PROVENANCE_MACROS
    )
    return "\n".join(lines) + "\n"


def result_source_value(run_record: dict[str, Any]) -> str:
    """Return the exact portable ResultSource identity value."""

    run_id = _escape_metadata(run_record["run_id"])
    cache_digest = _escape_metadata(
        str(run_record["cache_digest"])[:16]
    )
    return (
        rf"locked formal v2 run \texttt{{{run_id}}}, "
        rf"cache \texttt{{{cache_digest}}}"
    )


def parse_results_auto(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ReleaseValidationError(
            f"paper result macro file is missing or empty: {resolved}"
        )
    parsed: dict[str, str] = {}
    observed_comments: list[str] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            if stripped not in RESULTS_AUTO_COMMENTS:
                raise ReleaseValidationError(
                    f"unapproved macro-file comment at line {line_number}"
                )
            observed_comments.append(stripped)
            continue
        match = MACRO_LINE.fullmatch(line)
        if match is None:
            raise ReleaseValidationError(
                f"unknown macro-file content at line {line_number}"
            )
        name = match.group("name")
        if name not in ALL_MACROS or name in parsed:
            raise ReleaseValidationError(
                f"unexpected or duplicate macro at line {line_number}: {name}"
            )
        parsed[name] = match.group("value")
    if tuple(observed_comments) != RESULTS_AUTO_COMMENTS:
        raise ReleaseValidationError(
            "macro-file controlled comments are missing or reordered"
        )
    if set(parsed) != set(ALL_MACROS):
        raise ReleaseValidationError(
            "macro-file set mismatch; missing="
            + ",".join(sorted(set(ALL_MACROS).difference(parsed)))
            + "; unexpected="
            + ",".join(sorted(set(parsed).difference(ALL_MACROS)))
        )
    if parsed[RELEASE_SENTINEL] != RELEASE_SENTINEL_VALUE:
        raise ReleaseValidationError("v2 release sentinel is invalid")
    for name in PROVENANCE_MACROS:
        _validate_macro_value(name, parsed[name])
    return parsed


def _source_inventory(
    manifest: dict[str, Any],
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    macros = manifest.get("macros")
    if not isinstance(macros, dict) or set(macros) != set(
        PROVENANCE_MACROS
    ):
        raise ReleaseValidationError("manifest macro set is malformed")
    for name, record in macros.items():
        if not isinstance(record, dict) or set(record) != {
            "value",
            "sources",
            "derivation",
        }:
            raise ReleaseValidationError(
                f"manifest macro record is malformed: {name}"
            )
        _validate_macro_value(name, record["value"])
        if not isinstance(record["sources"], list) or not record["sources"]:
            raise ReleaseValidationError(
                f"manifest macro sources are empty: {name}"
            )
        for source in record["sources"]:
            if not isinstance(source, dict) or set(source) != {
                "path",
                "sha256",
            }:
                raise ReleaseValidationError(
                    f"manifest macro source is malformed: {name}"
                )
            path = source["path"]
            digest = _require_sha256(
                source["sha256"],
                f"manifest source {name}/{path}",
            )
            if path in inventory and inventory[path] != digest:
                raise ReleaseValidationError(
                    f"source digest disagreement across macros: {path}"
                )
            inventory[path] = digest
    return dict(sorted(inventory.items()))


def _inventory_digest(inventory: dict[str, str]) -> str:
    return hashlib.sha256(
        _canonical_json(inventory).encode("utf-8")
    ).hexdigest()


def validate_release_lock_structure(
    lock: dict[str, Any],
    *,
    results_auto_sha256: str | None = None,
    expected_run_id: str = EXPECTED_RUN_ID,
) -> None:
    if set(lock) != RELEASE_LOCK_KEYS:
        raise ReleaseValidationError(
            "v2 release-lock keys mismatch; missing="
            + ",".join(sorted(RELEASE_LOCK_KEYS.difference(lock)))
            + "; unexpected="
            + ",".join(sorted(set(lock).difference(RELEASE_LOCK_KEYS)))
        )
    exact = {
        "schema_version": RELEASE_LOCK_SCHEMA,
        "submission_unlocked": True,
        "paper_release_unlocked": True,
        "scientific_evidence_passed": True,
        "formal_cache_designation": FORMAL_DESIGNATION,
        "release_sentinel_name": RELEASE_SENTINEL,
        "release_sentinel_value": RELEASE_SENTINEL_VALUE,
    }
    drift = [
        key for key, expected in exact.items() if lock.get(key) != expected
    ]
    if drift:
        raise ReleaseValidationError(
            "v2 release-lock state mismatch: " + ",".join(drift)
        )
    for key in (
        "cache_digest",
        "run_json_sha256",
        "scientific_gate_sha256",
        "learning_curve_evidence_sha256",
        "freeze_sha256",
        "cache_manifest_sha256",
        "macro_value_manifest_sha256",
        "results_auto_sha256",
        "source_inventory_sha256",
    ):
        _require_sha256(lock.get(key), f"release lock {key}")
    if results_auto_sha256 is not None and (
        lock["results_auto_sha256"]
        != _require_sha256(
            results_auto_sha256,
            "expected results_auto_sha256",
        )
    ):
        raise ReleaseValidationError(
            "release lock is not bound to results_auto.tex"
        )
    if lock.get("run_id") != expected_run_id:
        raise ReleaseValidationError(
            "release-lock run_id differs from the frozen run"
        )
    try:
        generated = datetime.fromisoformat(str(lock["generated_utc"]))
    except ValueError as error:
        raise ReleaseValidationError(
            "release-lock generated_utc is not ISO-8601"
        ) from error
    if generated.tzinfo is None:
        raise ReleaseValidationError(
            "release-lock generated_utc lacks a timezone"
        )
    inventory = lock.get("source_inventory")
    if (
        not isinstance(inventory, dict)
        or not inventory
        or any(
            not isinstance(path, str)
            or not path
            or "\\" in path
            or Path(path).is_absolute()
            or SAFE_RUN_RELATIVE_SOURCE.fullmatch(path) is None
            or any(part in {".", ".."} for part in path.split("/"))
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            for path, digest in inventory.items()
        )
    ):
        raise ReleaseValidationError(
            "release-lock source inventory is malformed"
        )
    if list(inventory) != sorted(inventory):
        raise ReleaseValidationError(
            "release-lock source inventory is not canonical"
        )
    if lock["source_inventory_sha256"] != _inventory_digest(inventory):
        raise ReleaseValidationError(
            "release-lock source inventory digest mismatch"
        )
    provenance = lock.get("macro_provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(
        PROVENANCE_MACROS
    ):
        raise ReleaseValidationError(
            "release-lock macro provenance is incomplete"
        )
    for name, record in provenance.items():
        if not isinstance(record, dict) or set(record) != {
            "sources",
            "derivation",
        }:
            raise ReleaseValidationError(
                f"release-lock provenance is malformed: {name}"
            )
        sources = record.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) for source in sources)
            or any(source not in inventory for source in sources)
            or sources != sorted(set(sources))
            or not isinstance(record["derivation"], str)
            or len(record["derivation"]) < 24
        ):
            raise ReleaseValidationError(
                f"release-lock provenance is incomplete: {name}"
            )


def _paper_paths(paper_root: Path) -> tuple[Path, Path]:
    root = paper_root.expanduser().resolve()
    if not (root / "main.tex").is_file():
        raise ReleaseValidationError(
            f"paper root does not contain main.tex: {root}"
        )
    return root / "results_auto.tex", root / "release_lock.json"


def write_release(
    *,
    run_json: Path,
    learning_evidence: Path,
    macro_values: Path,
    paper_root: Path,
    freeze: Path = DEFAULT_FREEZE,
    replace_existing_release: bool = False,
) -> dict[str, Any]:
    resolved_run = run_json.expanduser().resolve()
    resolved_learning = learning_evidence.expanduser().resolve()
    resolved_freeze = freeze.expanduser().resolve()
    manifest = validate_macro_manifest(
        macro_values,
        run_json=resolved_run,
        learning_evidence=resolved_learning,
        freeze=resolved_freeze,
    )
    run = load_strict_json(resolved_run)
    values = {
        name: str(manifest["macros"][name]["value"])
        for name in PROVENANCE_MACROS
    }
    rendered = render_results_auto(run, values)
    rendered_digest = hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    results_path, lock_path = _paper_paths(paper_root)
    if lock_path.exists() and not replace_existing_release:
        raise ReleaseValidationError(
            "release_lock.json already exists; pass "
            "--replace-existing-release for an intentional replacement"
        )
    inventory = _source_inventory(manifest)
    bindings = manifest["bindings"]
    lock = {
        "schema_version": RELEASE_LOCK_SCHEMA,
        "submission_unlocked": True,
        "paper_release_unlocked": True,
        "scientific_evidence_passed": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "cache_digest": manifest["cache_digest"],
        "formal_cache_designation": FORMAL_DESIGNATION,
        "release_sentinel_name": RELEASE_SENTINEL,
        "release_sentinel_value": RELEASE_SENTINEL_VALUE,
        **bindings,
        "macro_value_manifest_sha256": hashlib.sha256(
            canonical_macro_manifest_text(manifest).encode("utf-8")
        ).hexdigest(),
        "results_auto_sha256": rendered_digest,
        "source_inventory_sha256": _inventory_digest(inventory),
        "source_inventory": inventory,
        "macro_provenance": {
            name: {
                "sources": [
                    source["path"]
                    for source in manifest["macros"][name]["sources"]
                ],
                "derivation": manifest["macros"][name]["derivation"],
            }
            for name in PROVENANCE_MACROS
        },
    }
    validate_release_lock_structure(
        lock,
        results_auto_sha256=rendered_digest,
        expected_run_id=str(manifest["run_id"]),
    )
    # Publish the lock first: any interrupted second replacement leaves a
    # digest mismatch and therefore remains fail-closed.
    _atomic_write_text(lock_path, canonical_release_lock_text(lock))
    _atomic_write_text(results_path, rendered)
    return validate_existing_release(
        run_json=resolved_run,
        learning_evidence=resolved_learning,
        macro_values=macro_values,
        paper_root=paper_root,
        freeze=resolved_freeze,
    )


def validate_existing_release(
    *,
    run_json: Path,
    learning_evidence: Path,
    macro_values: Path,
    paper_root: Path,
    freeze: Path = DEFAULT_FREEZE,
) -> dict[str, Any]:
    resolved_run = run_json.expanduser().resolve()
    resolved_learning = learning_evidence.expanduser().resolve()
    resolved_freeze = freeze.expanduser().resolve()
    manifest = validate_macro_manifest(
        macro_values,
        run_json=resolved_run,
        learning_evidence=resolved_learning,
        freeze=resolved_freeze,
    )
    run = load_strict_json(resolved_run)
    expected_values = {
        name: str(manifest["macros"][name]["value"])
        for name in PROVENANCE_MACROS
    }
    expected_results = render_results_auto(run, expected_values)
    results_path, lock_path = _paper_paths(paper_root)
    parsed = parse_results_auto(results_path)
    try:
        observed_results = results_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseValidationError(
            f"could not read released macro bytes: {error}"
        ) from error
    if observed_results != expected_results or any(
        parsed[name] != expected_values[name]
        for name in PROVENANCE_MACROS
    ):
        raise ReleaseValidationError(
            "released macros differ from deterministic artifact re-derivation"
        )
    lock = load_strict_json(lock_path)
    try:
        observed_lock_text = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseValidationError(
            f"could not read release-lock bytes: {error}"
        ) from error
    if observed_lock_text != canonical_release_lock_text(lock):
        raise ReleaseValidationError(
            "release_lock.json is not in canonical serialization"
        )
    validate_release_lock_structure(
        lock,
        results_auto_sha256=sha256_file(results_path),
        expected_run_id=str(manifest["run_id"]),
    )
    expected_bindings = manifest["bindings"]
    exact = {
        "run_id": manifest["run_id"],
        "cache_digest": manifest["cache_digest"],
        **expected_bindings,
        "macro_value_manifest_sha256": hashlib.sha256(
            canonical_macro_manifest_text(manifest).encode("utf-8")
        ).hexdigest(),
        "results_auto_sha256": sha256_file(results_path),
    }
    mismatches = [
        key for key, value in exact.items() if lock.get(key) != value
    ]
    if mismatches:
        raise ReleaseValidationError(
            "release-lock source binding mismatch: "
            + ",".join(sorted(mismatches))
        )
    inventory = _source_inventory(manifest)
    if lock["source_inventory"] != inventory:
        raise ReleaseValidationError(
            "release-lock source inventory differs from the macro manifest"
        )
    run_root = resolved_run.parent
    for relative, expected_digest in inventory.items():
        if sha256_file(_safe_run_source(run_root, relative)) != (
            expected_digest
        ):
            raise ReleaseValidationError(
                f"macro source changed after release: {relative}"
            )
    expected_provenance = {
        name: {
            "sources": [
                source["path"]
                for source in manifest["macros"][name]["sources"]
            ],
            "derivation": manifest["macros"][name]["derivation"],
        }
        for name in PROVENANCE_MACROS
    }
    if lock["macro_provenance"] != expected_provenance:
        raise ReleaseValidationError(
            "release-lock macro provenance differs from deterministic "
            "re-derivation"
        )
    return {
        "ok": True,
        "action": "v2_existing_release_validated",
        "submission_unlocked": True,
        "paper_release_unlocked": True,
        "scientific_evidence_passed": True,
        "run_id": manifest["run_id"],
        "macro_count": len(PROVENANCE_MACROS),
        "results_auto": str(results_path),
        "release_lock": str(lock_path),
        "results_auto_sha256": sha256_file(results_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument(
        "--learning-curve-evidence",
        type=Path,
        required=True,
    )
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--macro-values",
        type=Path,
        default=PROJECT_ROOT
        / "tvt_submission"
        / "formal_macro_values_v2.json",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=PROJECT_ROOT / "paper",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--write-macro-manifest", action="store_true")
    actions.add_argument(
        "--validate-macro-manifest",
        action="store_true",
    )
    actions.add_argument("--write-release", action="store_true")
    actions.add_argument("--validate-release", action="store_true")
    parser.add_argument(
        "--replace-existing-manifest",
        action="store_true",
    )
    parser.add_argument(
        "--replace-existing-release",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.write_macro_manifest:
            result = write_macro_manifest(
                run_json=arguments.run_json,
                learning_evidence=arguments.learning_curve_evidence,
                output=arguments.macro_values,
                freeze=arguments.freeze,
                replace_existing=arguments.replace_existing_manifest,
            )
        elif arguments.validate_macro_manifest:
            manifest = validate_macro_manifest(
                arguments.macro_values,
                run_json=arguments.run_json,
                learning_evidence=arguments.learning_curve_evidence,
                freeze=arguments.freeze,
            )
            result = {
                "ok": True,
                "action": "v2_macro_manifest_validated",
                "write_performed": False,
                "submission_unlocked": False,
                "scientific_evidence_passed": True,
                "macro_count": len(manifest["macros"]),
                "run_id": manifest["run_id"],
                "manifest_sha256": sha256_file(
                    arguments.macro_values
                ),
            }
        elif arguments.write_release:
            result = write_release(
                run_json=arguments.run_json,
                learning_evidence=arguments.learning_curve_evidence,
                macro_values=arguments.macro_values,
                paper_root=arguments.paper_root,
                freeze=arguments.freeze,
                replace_existing_release=(
                    arguments.replace_existing_release
                ),
            )
        elif arguments.validate_release:
            result = validate_existing_release(
                run_json=arguments.run_json,
                learning_evidence=arguments.learning_curve_evidence,
                macro_values=arguments.macro_values,
                paper_root=arguments.paper_root,
                freeze=arguments.freeze,
            )
        else:
            manifest = generate_macro_manifest(
                run_json=arguments.run_json,
                learning_evidence=arguments.learning_curve_evidence,
                freeze=arguments.freeze,
            )
            result = {
                "ok": True,
                "action": "v2_paper_release_preflight",
                "write_performed": False,
                "submission_unlocked": False,
                "scientific_evidence_passed": True,
                "macro_count": len(manifest["macros"]),
                "run_id": manifest["run_id"],
                "manifest_sha256": hashlib.sha256(
                    canonical_macro_manifest_text(manifest).encode("utf-8")
                ).hexdigest(),
            }
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "submission_unlocked": False,
                    "paper_release_unlocked": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
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
