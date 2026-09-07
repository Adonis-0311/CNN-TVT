"""Small source-provenance helpers for artifact-producing runners."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


SOURCE_TREE_SCHEMA = "vimd_amc.source_tree.v2"
BASE_SOURCE_PROFILE = "python_entrypoint"
TVT_V2_SOURCE_PROFILE = "tvt_v2_execution"

# This list is intentionally explicit.  Python package sources are discovered
# below, while these repository-owned control-plane files would otherwise sit
# outside ``src/vimd_amc`` and escape the execution fingerprint.
TVT_V2_EXECUTION_CONTROL_FILES = (
    "experiments/run_standard_experiment.py",
    "experiments/run_learning_curve_v2.py",
    "experiments/run_formal_tvt_v2.py",
    "standards/build_factor_cache.py",
    "standards/matlab/vimd_apply_nrtdl_batch.m",
    "tvt_submission/configs/formal_tvt_freeze_v2.json",
    "tvt_submission/configs/formal_tvt_freeze_v3_8gb.json",
    "tvt_submission/configs/formal_tvt_freeze_v4_8gb_dual.json",
    "tvt_submission/formal_v2_contract.py",
    "tvt_submission/run_v2_after_gpu_free.ps1",
    "tvt_submission/run_v3_8gb_after_gpu_free.ps1",
    "tvt_submission/validate_formal_freeze_v2.py",
    "tvt_submission/validate_v2_release.py",
    "tvt_submission/validate_v2_paper_release.py",
    "pyproject.toml",
)
REQUIRED_RUNTIME_VERSION_FIELDS = (
    "python",
    "torch",
    "numpy",
    "pandas",
    "scipy",
)


def resolve_frozen_device(
    expected: Any,
    requested: Any = None,
) -> str:
    """Resolve a device override without permitting contract drift."""

    if not isinstance(expected, str) or not expected:
        raise ValueError("frozen execution device is missing or malformed")
    selected = expected if requested is None else requested
    if not isinstance(selected, str) or selected != expected:
        raise ValueError(
            "device contract mismatch: "
            f"frozen={expected!r}, requested={selected!r}"
        )
    return expected


def source_tree_record(
    project_root: Path,
    entrypoint: Path,
    *,
    profile: str = BASE_SOURCE_PROFILE,
) -> dict[str, Any]:
    """Hash all repository-owned source controlling one execution.

    The workspace is not required to be a Git repository, so artifacts bind
    themselves directly to file content.  Paths are relative and sorted to
    make the aggregate digest stable across checkout locations.  The TVT-v2
    profile additionally requires and fingerprints every repository-level
    wrapper, freeze, validator, cache entrypoint, and MATLAB backend that
    controls the formal execution path.
    """

    root = project_root.resolve()
    resolved_entrypoint = entrypoint.resolve()
    try:
        resolved_entrypoint.relative_to(root)
    except ValueError as error:
        raise ValueError("entrypoint must be inside project_root") from error
    if profile not in (BASE_SOURCE_PROFILE, TVT_V2_SOURCE_PROFILE):
        raise ValueError(f"unknown source-tree profile: {profile!r}")
    candidates = list((root / "src" / "vimd_amc").rglob("*.py"))
    candidates.extend((resolved_entrypoint, root / "pyproject.toml"))
    required_control_files: tuple[str, ...] = ()
    if profile == TVT_V2_SOURCE_PROFILE:
        required_control_files = TVT_V2_EXECUTION_CONTROL_FILES
        missing = [
            relative
            for relative in required_control_files
            if not (root / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "TVT-v2 execution source binding is incomplete; missing: "
                + ",".join(missing)
            )
        candidates.extend(root / relative for relative in required_control_files)
    unique = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in unique:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": SOURCE_TREE_SCHEMA,
        "profile": profile,
        "algorithm": "sha256",
        "aggregate_digest": aggregate.hexdigest(),
        "file_count": len(records),
        "required_control_files": list(required_control_files),
        "files": records,
    }


def source_tree_record_matches(
    record: Any,
    project_root: Path,
    entrypoint: Path,
    *,
    profile: str = TVT_V2_SOURCE_PROFILE,
) -> bool:
    """Return whether a serialized record exactly matches current sources."""

    if not isinstance(record, dict):
        return False
    current = source_tree_record(
        project_root,
        entrypoint,
        profile=profile,
    )
    return (
        record.get("schema_version") == SOURCE_TREE_SCHEMA
        and record.get("profile") == profile
        and record.get("algorithm") == "sha256"
        and record.get("aggregate_digest") == current["aggregate_digest"]
        and record.get("file_count") == current["file_count"]
        and record.get("required_control_files")
        == current["required_control_files"]
        and record.get("files") == current["files"]
    )


def source_tree_audit_reasons(
    audit: Any,
    project_root: Path,
    entrypoint: Path,
    *,
    expected_record: Any = None,
    profile: str = TVT_V2_SOURCE_PROFILE,
) -> list[str]:
    """Explain why a start/end source audit is not release-grade."""

    if not isinstance(audit, dict):
        return ["source_tree_execution_audit_missing"]
    reasons: list[str] = []
    if audit.get("unchanged") is not True:
        reasons.append("source_tree_execution_audit_not_unchanged")
    start = audit.get("start")
    end = audit.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        reasons.append("source_tree_start_or_end_record_missing")
        return reasons
    if start != end:
        reasons.append("source_tree_start_end_records_differ")
    if expected_record is not None and start != expected_record:
        reasons.append("source_tree_record_differs_from_bound_evidence")
    if not source_tree_record_matches(
        start,
        project_root,
        entrypoint,
        profile=profile,
    ):
        reasons.append("source_tree_record_differs_from_current_sources")
    return reasons


def execution_environment_reasons(
    environment: Any,
    *,
    expected_device: str,
    expected_source_tree: Any = None,
) -> list[str]:
    """Explain runtime/device drift from a frozen execution contract."""

    if not isinstance(environment, dict):
        return ["execution_environment_missing"]
    reasons: list[str] = []
    if environment.get("device") != expected_device:
        reasons.append("execution_device_drift")
    expected_type = expected_device.split(":", 1)[0]
    if environment.get("device_type") != expected_type:
        reasons.append("execution_device_type_drift")
    for field in REQUIRED_RUNTIME_VERSION_FIELDS:
        value = environment.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"runtime_version_missing:{field}")
    if expected_type == "cuda":
        if environment.get("cuda_available") is not True:
            reasons.append("cuda_not_available_during_execution")
        runtime = environment.get("cuda_runtime")
        if not isinstance(runtime, str) or not runtime.strip():
            reasons.append("cuda_runtime_missing")
        gpu = environment.get("gpu")
        if not isinstance(gpu, str) or not gpu.strip():
            reasons.append("gpu_identity_missing")
    if (
        expected_source_tree is not None
        and environment.get("source_tree") != expected_source_tree
    ):
        reasons.append("environment_source_tree_binding_drift")
    return reasons


__all__ = [
    "BASE_SOURCE_PROFILE",
    "REQUIRED_RUNTIME_VERSION_FIELDS",
    "SOURCE_TREE_SCHEMA",
    "TVT_V2_EXECUTION_CONTROL_FILES",
    "TVT_V2_SOURCE_PROFILE",
    "execution_environment_reasons",
    "resolve_frozen_device",
    "source_tree_audit_reasons",
    "source_tree_record",
    "source_tree_record_matches",
]
