"""Independently audit an immutable schema-2 factor-isolated cache.

This command does not train or evaluate a model.  It verifies the serialized
cache against the locked screening protocol and writes a strict-JSON audit
next to (not inside) the immutable cache directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.data.split import manifest_digest  # noqa: E402
from vimd_amc.data.synthesis import SynthesisConfig  # noqa: E402
from vimd_amc.reproducibility import source_tree_record  # noqa: E402
from vimd_amc.standards import (  # noqa: E402
    CachedPairedAMCDataset,
    FACTOR_ISOLATED_SPLITS,
    factor_isolated_split_policies,
    validate_cached_components,
)


EXPECTED_SPLIT_SIZES = {
    "train": 1_000,
    "validation": 200,
    "id_test": 500,
    "hard_interference": 500,
    "unseen_jammer": 500,
    "unseen_speed": 500,
    "heldout_channel": 500,
    "combined_ood": 500,
    "clean_retention": 500,
}
EXPECTED_MODULATIONS = (
    "BPSK",
    "PI2BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "256QAM",
    "GMSK",
    "CPFSK",
    "4FSK",
)
EXPECTED_DESIGNATION = "screening_not_formal_tvt_evidence"
EXPECTED_SAMPLE_LENGTH = 1024
EXPECTED_GUARD_SAMPLES = 96
EXPECTED_MASTER_SEED = 20260727
COMPONENT_TOLERANCE = 2e-6
POWER_TOLERANCE_DB = 2e-3
QUALITY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a screening schema-2 factor-isolated cache."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prebuild-record", type=Path, required=True)
    return parser.parse_args()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _float_set(values: Iterable[float]) -> set[float]:
    return {round(float(value), 9) for value in values}


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks[name] = {"passed": bool(passed), **_canonical(details)}


def _file_audit(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    shape_or_dtype_mismatches: list[dict[str, Any]] = []
    nonfinite_files: list[str] = []
    declared_paths: set[str] = set()
    declared_bytes = 0
    verified = 0
    for split, arrays in manifest["files"].items():
        for array_name, metadata in arrays.items():
            relative = str(metadata["path"]).replace("\\", "/")
            declared_paths.add(relative)
            path = root / relative
            if not path.is_file():
                mismatches.append(
                    {"path": relative, "reason": "declared_file_missing"}
                )
                continue
            declared_bytes += path.stat().st_size
            actual_hash = _sha256_file(path)
            if actual_hash != metadata["sha256"]:
                mismatches.append(
                    {
                        "path": relative,
                        "reason": "sha256_mismatch",
                        "expected": metadata["sha256"],
                        "actual": actual_hash,
                    }
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(v) for v in metadata["shape"])
            expected_dtype = np.dtype(metadata["dtype"])
            if array.shape != expected_shape or array.dtype != expected_dtype:
                shape_or_dtype_mismatches.append(
                    {
                        "path": relative,
                        "expected_shape": list(expected_shape),
                        "actual_shape": list(array.shape),
                        "expected_dtype": str(expected_dtype),
                        "actual_dtype": str(array.dtype),
                    }
                )
            if (
                np.issubdtype(array.dtype, np.floating)
                and not bool(np.all(np.isfinite(array)))
            ):
                nonfinite_files.append(relative)
            verified += 1
            del array
    actual_npy_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.npy")
        if path.is_file()
    }
    undeclared = sorted(actual_npy_paths.difference(declared_paths))
    declared_missing = sorted(declared_paths.difference(actual_npy_paths))
    all_files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "passed": not (
            mismatches
            or shape_or_dtype_mismatches
            or nonfinite_files
            or undeclared
            or declared_missing
        ),
        "declared_array_file_count": len(declared_paths),
        "verified_array_file_count": verified,
        "total_cache_file_count_including_manifest": len(all_files),
        "declared_array_bytes": declared_bytes,
        "total_cache_bytes_including_manifest": sum(
            path.stat().st_size for path in all_files
        ),
        "checksum_mismatches": mismatches,
        "shape_or_dtype_mismatches": shape_or_dtype_mismatches,
        "nonfinite_files": sorted(nonfinite_files),
        "undeclared_npy_files": undeclared,
        "declared_missing_npy_files": declared_missing,
    }


def _source_audit(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    owners: dict[int, str] = {}
    collisions: list[dict[str, Any]] = []
    duplicate_within_split: dict[str, int] = {}
    array_manifest_mismatches: dict[str, int] = {}
    record_manifest_mismatches: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for split in FACTOR_ISOLATED_SPLITS:
        declared = [int(value) for value in manifest["source_ids"][split]]
        duplicate_within_split[split] = len(declared) - len(set(declared))
        split_counts[split] = len(declared)
        for source_id in declared:
            prior = owners.get(source_id)
            if prior is not None and prior != split:
                collisions.append(
                    {
                        "source_id": source_id,
                        "first_split": prior,
                        "second_split": split,
                    }
                )
            owners[source_id] = split
        source_array = np.load(
            root / split / "source_id.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        array_manifest_mismatches[split] = sum(
            int(left) != int(right)
            for left, right in zip(source_array.tolist(), declared, strict=True)
        )
        record_ids = [
            int(record["source_sequence_id"])
            for record in manifest["records"][split]
        ]
        record_manifest_mismatches[split] = sum(
            left != right
            for left, right in zip(record_ids, declared, strict=True)
        )
    passed = (
        not collisions
        and all(value == 0 for value in duplicate_within_split.values())
        and all(value == 0 for value in array_manifest_mismatches.values())
        and all(value == 0 for value in record_manifest_mismatches.values())
        and split_counts == EXPECTED_SPLIT_SIZES
    )
    return {
        "passed": passed,
        "globally_unique_source_count": len(owners),
        "expected_source_count": sum(EXPECTED_SPLIT_SIZES.values()),
        "split_counts": split_counts,
        "cross_split_collisions": collisions[:20],
        "duplicate_count_within_split": duplicate_within_split,
        "source_array_manifest_mismatch_count": array_manifest_mismatches,
        "record_manifest_mismatch_count": record_manifest_mismatches,
    }


def _record_audit(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_split: dict[str, Any] = {}
    factor_failures: list[str] = []
    guard_failures: list[str] = []
    pairing_failures: list[str] = []
    excluded_seen: set[str] = set()
    expected_policies = {
        policy.split: asdict(policy)
        for policy in factor_isolated_split_policies(EXPECTED_SPLIT_SIZES)
    }
    global_min_guard_margin = EXPECTED_GUARD_SAMPLES
    global_max_required_guard = 0
    for split in FACTOR_ISOLATED_SPLITS:
        policy = expected_policies[split]
        records = manifest["records"][split]
        class_counts = Counter(record["modulation"] for record in records)
        actual_jammers: set[str] = set()
        actual_target_profiles: set[str] = set()
        actual_jammer_profiles: set[str] = set()
        actual_speeds: set[float] = set()
        actual_snr: set[float] = set()
        actual_sir: set[float] = set()
        clean_views = 0
        active_views = 0
        sir_invalid_views = 0
        condition_seed_equal_pairs = 0
        for record_index, record in enumerate(records):
            views = record["views"]
            if len(views) != 2:
                pairing_failures.append(f"{split}:{record_index}:view_count")
                continue
            if int(views[0]["condition_seed"]) == int(
                views[1]["condition_seed"]
            ):
                condition_seed_equal_pairs += 1
            for view_index, view in enumerate(views):
                prefix = f"{split}:{record_index}:{view_index}"
                if (
                    int(view["source_sequence_id"])
                    != int(record["source_sequence_id"])
                    or view["modulation"] != record["modulation"]
                ):
                    pairing_failures.append(f"{prefix}:source_or_label")
                jammer = str(view["jammer_name"])
                actual_jammers.add(jammer)
                excluded_seen.update(
                    {jammer}.intersection({"cochannel", "mixed"})
                )
                actual_target_profiles.add(str(view["tdl_target_profile"]))
                actual_jammer_profiles.add(str(view["tdl_jammer_profile"]))
                actual_speeds.add(round(float(view["vehicle_speed_kmh"]), 9))
                actual_snr.add(round(float(view["snr_db"]), 9))
                sir_valid = bool(view["sir_valid"])
                if sir_valid:
                    active_views += 1
                    actual_sir.add(round(float(view["sir_db"]), 9))
                    if jammer == "none" or not bool(
                        view["interference_present"]
                    ):
                        factor_failures.append(f"{prefix}:active_semantics")
                else:
                    clean_views += 1
                    sir_invalid_views += 1
                    if jammer != "none" or bool(view["interference_present"]):
                        factor_failures.append(f"{prefix}:clean_semantics")
                guard = int(view["tdl_guard_samples"])
                required = int(view["tdl_required_guard_samples"])
                margin = int(view["tdl_guard_margin_samples"])
                recomputed_required = max(
                    int(view["tdl_target_channel_filter_delay_samples"])
                    + int(view["tdl_target_maximum_channel_delay_samples"]),
                    int(view["tdl_jammer_channel_filter_delay_samples"])
                    + int(view["tdl_jammer_maximum_channel_delay_samples"]),
                )
                global_min_guard_margin = min(
                    global_min_guard_margin, margin
                )
                global_max_required_guard = max(
                    global_max_required_guard, required
                )
                if not (
                    guard == EXPECTED_GUARD_SAMPLES
                    and required == recomputed_required
                    and margin == guard - required
                    and margin >= 0
                    and int(view["tdl_crop_start_sample"]) == guard
                    and int(view["tdl_crop_stop_sample"])
                    == guard + EXPECTED_SAMPLE_LENGTH
                ):
                    guard_failures.append(prefix)
        allowed_jammers = set(policy["jammer_choices"])
        expected_actual_jammers = set(allowed_jammers)
        if float(policy["clean_fraction"]) > 0:
            expected_actual_jammers.add("none")
        expected_clean = round(
            2 * int(policy["size"]) * float(policy["clean_fraction"])
        )
        factor_checks = {
            "modulations_complete": set(class_counts)
            == set(EXPECTED_MODULATIONS),
            "classes_exactly_balanced": len(set(class_counts.values())) == 1,
            "jammers_complete": actual_jammers
            == expected_actual_jammers,
            "target_profiles_complete": actual_target_profiles
            == set(policy["profiles"]),
            "jammer_profiles_complete": actual_jammer_profiles
            == set(policy["profiles"]),
            "speeds_complete": actual_speeds
            == _float_set(policy["speeds_kmh"]),
            "snr_grid_complete": actual_snr
            == _float_set(policy["snr_db_values"]),
            "sir_grid_complete": (
                not actual_sir
                if policy["sir_db_values"] is None
                else actual_sir == _float_set(policy["sir_db_values"])
            ),
            "clean_view_count_exact": clean_views == expected_clean,
            "active_view_count_exact": active_views
            == 2 * int(policy["size"]) - expected_clean,
            "paired_condition_seeds_distinct": condition_seed_equal_pairs == 0,
        }
        if split == "hard_interference":
            factor_checks["hard_sir_no_greater_than_zero"] = (
                bool(actual_sir) and max(actual_sir) <= 0.0
            )
        if not all(factor_checks.values()):
            factor_failures.append(split)
        per_split[split] = {
            "source_count": len(records),
            "view_count": 2 * len(records),
            "class_counts": dict(sorted(class_counts.items())),
            "minimum_per_class_count": min(class_counts.values()),
            "actual_jammers": sorted(actual_jammers),
            "actual_target_profiles": sorted(actual_target_profiles),
            "actual_jammer_profiles": sorted(actual_jammer_profiles),
            "actual_speeds_kmh": sorted(actual_speeds),
            "actual_snr_db": sorted(actual_snr),
            "actual_sir_db": sorted(actual_sir),
            "clean_view_count": clean_views,
            "active_view_count": active_views,
            "sir_invalid_view_count": sir_invalid_views,
            "factor_checks": factor_checks,
        }
    seen_jammers = set(expected_policies["train"]["jammer_choices"])
    held_jammers = set(expected_policies["unseen_jammer"]["jammer_choices"])
    seen_speeds = set(expected_policies["train"]["speeds_kmh"])
    held_speeds = set(expected_policies["unseen_speed"]["speeds_kmh"])
    seen_profiles = set(expected_policies["train"]["profiles"])
    held_profiles = set(expected_policies["heldout_channel"]["profiles"])
    isolation = {
        "seen_held_jammers_disjoint": not bool(
            seen_jammers.intersection(held_jammers)
        ),
        "seen_held_speeds_disjoint": not bool(
            seen_speeds.intersection(held_speeds)
        ),
        "seen_held_profiles_disjoint": not bool(
            seen_profiles.intersection(held_profiles)
        ),
        "excluded_jammers_absent": not excluded_seen,
        "factor_failures": factor_failures[:50],
        "pairing_failures": pairing_failures[:50],
    }
    isolation["passed"] = (
        all(
            isolation[key]
            for key in (
                "seen_held_jammers_disjoint",
                "seen_held_speeds_disjoint",
                "seen_held_profiles_disjoint",
                "excluded_jammers_absent",
            )
        )
        and not factor_failures
        and not pairing_failures
    )
    guard = {
        "passed": not guard_failures and global_min_guard_margin >= 0,
        "configured_guard_samples": EXPECTED_GUARD_SAMPLES,
        "maximum_required_guard_samples": global_max_required_guard,
        "minimum_guard_margin_samples": global_min_guard_margin,
        "failure_examples": guard_failures[:50],
    }
    return per_split, isolation, guard


def main() -> None:
    args = parse_args()
    root = args.cache_root.resolve()
    output = args.output.resolve()
    prebuild_path = args.prebuild_record.resolve()
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    manifest_path = root / "manifest.json"
    manifest = _strict_json(manifest_path)
    prebuild = _strict_json(prebuild_path)
    checks: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []

    digest_payload = {
        key: value
        for key, value in manifest.items()
        if key != "cache_digest"
    }
    recomputed_cache_digest = manifest_digest([digest_payload])
    exact_splits = set(manifest.get("files", {})) == set(
        FACTOR_ISOLATED_SPLITS
    )
    _check(
        checks,
        "manifest_schema_and_digest",
        manifest.get("schema_version") == 2
        and exact_splits
        and recomputed_cache_digest == manifest.get("cache_digest"),
        schema_version=manifest.get("schema_version"),
        exact_nine_splits=exact_splits,
        declared_cache_digest=manifest.get("cache_digest"),
        recomputed_cache_digest=recomputed_cache_digest,
    )

    configuration_bytes = json.dumps(
        manifest["configuration"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    configuration_sha256 = sha256(configuration_bytes).hexdigest()
    expected_policy = {
        policy.split: _canonical(asdict(policy))
        for policy in factor_isolated_split_policies(EXPECTED_SPLIT_SIZES)
    }
    policy_exact = (
        manifest.get("preregistered_split_policy") == expected_policy
        and manifest["configuration"].get("split_policies")
        == [expected_policy[split] for split in FACTOR_ISOLATED_SPLITS]
    )
    config = manifest["configuration"]
    exclusions = manifest.get("protocol_exclusions", {})
    policy_jammers = {
        jammer
        for policy in expected_policy.values()
        for jammer in policy["jammer_choices"]
    }
    expected_taxonomy = list(SynthesisConfig().jammer_types)
    _check(
        checks,
        "locked_configuration_and_policy",
        policy_exact
        and config.get("evidence_designation") == EXPECTED_DESIGNATION
        and int(config.get("sample_length", -1)) == EXPECTED_SAMPLE_LENGTH
        and int(config.get("guard_samples", -1)) == EXPECTED_GUARD_SAMPLES
        and int(config.get("master_seed", -1)) == EXPECTED_MASTER_SEED
        and config.get("modulations") == list(EXPECTED_MODULATIONS)
        and configuration_sha256 == prebuild.get("configuration_sha256")
        and set(exclusions) == {"cochannel", "mixed"}
        and not {"cochannel", "mixed"}.intersection(policy_jammers)
        and manifest.get("jammer_taxonomy") == expected_taxonomy,
        policy_exact=policy_exact,
        evidence_designation=config.get("evidence_designation"),
        sample_length=config.get("sample_length"),
        guard_samples=config.get("guard_samples"),
        master_seed=config.get("master_seed"),
        configuration_sha256=configuration_sha256,
        prebuild_configuration_sha256=prebuild.get(
            "configuration_sha256"
        ),
        protocol_exclusions=exclusions,
        jammer_taxonomy=manifest.get("jammer_taxonomy"),
    )

    critical_hashes = {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in prebuild["code_sha256"]
    }
    critical_code_unchanged = critical_hashes == prebuild["code_sha256"]
    current_source_tree = source_tree_record(
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "experiments" / "run_standard_experiment.py",
    )
    source_tree_unchanged = (
        current_source_tree["aggregate_digest"]
        == prebuild.get("source_tree_sha256")
    )
    if not source_tree_unchanged:
        warnings.append(
            {
                "severity": "medium",
                "finding": "aggregate_source_tree_changed_after_prebuild_record",
                "impact": (
                    "The broader experiment source tree changed concurrently. "
                    "All three cache-construction-critical files retained their "
                    "prebuild hashes, so this does not invalidate serialized "
                    "cache generation, but the drift must remain disclosed."
                ),
            }
        )
    _check(
        checks,
        "build_provenance",
        critical_code_unchanged,
        build_critical_code_unchanged=critical_code_unchanged,
        prebuild_code_sha256=prebuild["code_sha256"],
        audit_code_sha256=critical_hashes,
        source_tree_unchanged=source_tree_unchanged,
        prebuild_source_tree_sha256=prebuild.get("source_tree_sha256"),
        audit_source_tree_sha256=current_source_tree["aggregate_digest"],
    )

    file_audit = _file_audit(root, manifest)
    checks["files_checksums_shapes_finiteness"] = file_audit
    source_audit = _source_audit(root, manifest)
    checks["source_integrity"] = source_audit
    per_split, factor_audit, guard_audit = _record_audit(manifest)
    checks["factor_isolation_and_coverage"] = factor_audit
    checks["guard_and_crop"] = guard_audit

    manifest_coverage_match = True
    for split, split_audit in per_split.items():
        actual = manifest["factor_coverage"][split]["actual"]
        comparisons = (
            set(actual["modulations"]) == set(split_audit["class_counts"]),
            set(actual["jammer_choices"])
            == set(split_audit["actual_jammers"]),
            set(actual["target_profiles"])
            == set(split_audit["actual_target_profiles"]),
            set(actual["jammer_profiles"])
            == set(split_audit["actual_jammer_profiles"]),
            _float_set(actual["speeds_kmh"])
            == set(split_audit["actual_speeds_kmh"]),
            _float_set(actual["snr_db_values"])
            == set(split_audit["actual_snr_db"]),
            _float_set(actual["sir_db_values"])
            == set(split_audit["actual_sir_db"]),
        )
        manifest_coverage_match = manifest_coverage_match and all(comparisons)
    _check(
        checks,
        "manifest_factor_coverage_crosscheck",
        manifest_coverage_match
        and all(
            bool(coverage["all_actual_values_within_policy"])
            for coverage in manifest["factor_coverage"].values()
        ),
        independent_coverage_matches_manifest=manifest_coverage_match,
        all_manifest_factor_policy_flags_true=all(
            bool(coverage["all_actual_values_within_policy"])
            for coverage in manifest["factor_coverage"].values()
        ),
    )

    component_validation: dict[str, Any] = {}
    component_failures: dict[str, str] = {}
    for split in FACTOR_ISOLATED_SPLITS:
        try:
            with CachedPairedAMCDataset(
                root, split, verify_checksums=False
            ) as dataset:
                component_validation[split] = validate_cached_components(
                    dataset,
                    component_tolerance=COMPONENT_TOLERANCE,
                    power_tolerance_db=POWER_TOLERANCE_DB,
                )
        except Exception as exc:  # audit must serialize failures
            component_failures[split] = f"{type(exc).__name__}: {exc}"
    if component_validation:
        worst = {
            "max_component_error": max(
                value["max_component_error"]
                for value in component_validation.values()
            ),
            "max_snr_error_db": max(
                value["max_snr_error_db"]
                for value in component_validation.values()
            ),
            "max_sir_error_db": max(
                value["max_sir_error_db"]
                for value in component_validation.values()
            ),
            "max_quality_normalization_error": max(
                value["max_quality_normalization_error"]
                for value in component_validation.values()
            ),
            "max_clean_jammer_power": max(
                value["max_clean_jammer_power"]
                for value in component_validation.values()
            ),
        }
    else:
        worst = {}
    _check(
        checks,
        "component_snr_sir_quality_recomputation",
        not component_failures
        and set(component_validation) == set(FACTOR_ISOLATED_SPLITS)
        and worst.get("max_component_error", float("inf"))
        <= COMPONENT_TOLERANCE
        and worst.get("max_snr_error_db", float("inf"))
        <= POWER_TOLERANCE_DB
        and worst.get("max_sir_error_db", float("inf"))
        <= POWER_TOLERANCE_DB
        and worst.get("max_quality_normalization_error", float("inf"))
        <= QUALITY_TOLERANCE
        and worst.get("max_clean_jammer_power", float("inf")) <= 1e-12,
        tolerances={
            "component": COMPONENT_TOLERANCE,
            "power_db": POWER_TOLERANCE_DB,
            "quality": QUALITY_TOLERANCE,
            "clean_jammer_power": 1e-12,
        },
        worst=worst,
        per_split=component_validation,
        failures=component_failures,
    )

    normalization = manifest.get("quality_normalization", {})
    max_speed_kmh = max(float(value) for value in config["speeds_kmh"])
    expected_doppler_scale = (
        max_speed_kmh
        / 3.6
        * float(config["carrier_frequency_hz"])
        / 299_792_458.0
    )
    normalization_explicit = (
        normalization.get("snr_db", {}).get("scale") == 20.0
        and normalization.get("snr_db", {}).get("unit") == "dB"
        and normalization.get("sir_db", {}).get("scale") == 20.0
        and normalization.get("sir_db", {}).get("unit") == "dB"
        and normalization.get("doppler_hz", {}).get("unit") == "Hz"
        and abs(
            float(normalization.get("doppler_hz", {}).get("scale", -1.0))
            - expected_doppler_scale
        )
        <= 1e-9
    )
    _check(
        checks,
        "quality_normalization_contract",
        normalization_explicit,
        normalization=normalization,
        expected_doppler_scale_hz=expected_doppler_scale,
    )

    class_complete = all(
        set(split_audit["class_counts"]) == set(EXPECTED_MODULATIONS)
        and split_audit["minimum_per_class_count"]
        >= EXPECTED_SPLIT_SIZES[split] // len(EXPECTED_MODULATIONS)
        for split, split_audit in per_split.items()
    )
    _check(
        checks,
        "complete_balanced_modulation_support",
        class_complete,
        expected_modulations=list(EXPECTED_MODULATIONS),
        minimum_per_class_by_split={
            split: values["minimum_per_class_count"]
            for split, values in per_split.items()
        },
    )

    mandatory_pass = all(
        bool(result["passed"]) for result in checks.values()
    )
    audit = {
        "audit_schema_version": 1,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(root),
        "cache_digest": manifest["cache_digest"],
        "intended_grain": (
            "one source sequence per row, two independently impaired views "
            "per source, globally source-disjoint across nine splits"
        ),
        "intended_use": "stable model screening and pipeline triage",
        "summary": {
            "status": "pass" if mandatory_pass else "fail",
            "stable_screening_usable": mandatory_pass,
            "formal_tvt_or_headline_eligible": False,
            "evidence_designation": config["evidence_designation"],
            "source_count": sum(EXPECTED_SPLIT_SIZES.values()),
            "paired_view_count": 2 * sum(EXPECTED_SPLIT_SIZES.values()),
            "modulation_class_count": len(EXPECTED_MODULATIONS),
            "failed_checks": [
                name
                for name, result in checks.items()
                if not bool(result["passed"])
            ],
        },
        "checks": checks,
        "per_split_profile": per_split,
        "warnings": warnings,
        "claim_boundary": (
            "This administrative screening cache may be used for stable model "
            "triage. It must not be relabeled or cited as formal/headline TVT "
            "evidence; no model training or result claim is part of this audit."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    if not mandatory_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
