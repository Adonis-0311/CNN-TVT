"""Fail-closed release assessment for TVT evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .manifest import ManifestError, validate_run_manifest


@dataclass(frozen=True)
class GateResult:
    eligible: bool
    failures: tuple[str, ...]

    def require(self) -> None:
        if not self.eligible:
            raise ManifestError("release denied: " + "; ".join(self.failures))


def assess_release(
    manifest_path: str | Path,
    *,
    expected_config_sha256: str,
    expected_seeds: Iterable[int],
    required_artifacts: Iterable[str],
    require_clean_worktree: bool = True,
) -> GateResult:
    failures: list[str] = []
    try:
        payload = validate_run_manifest(
            manifest_path, required_artifacts=required_artifacts
        )
    except ManifestError as exc:
        return GateResult(False, (str(exc),))

    if payload.get("frozen_config_sha256") != expected_config_sha256:
        failures.append("frozen configuration digest does not match release plan")
    if payload.get("evidence_tier") not in {"screening", "headline"}:
        failures.append("evidence tier is not declared")
    completed = payload.get("seeds_completed")
    if not isinstance(completed, list) or set(completed) != set(expected_seeds):
        failures.append("completed seeds do not exactly match the frozen seed set")

    git = payload.get("git")
    if not isinstance(git, Mapping) or not git.get("commit"):
        failures.append("source commit is missing")
    elif require_clean_worktree and git.get("dirty") is not False:
        failures.append("source worktree was not clean")

    qa = payload.get("qa")
    required_qa = {
        "sample_id_unique",
        "split_group_disjoint",
        "component_identity",
        "duplicate_scan",
    }
    if not isinstance(qa, Mapping):
        failures.append("data QA record is missing")
    else:
        failed_qa = sorted(key for key in required_qa if qa.get(key) is not True)
        if failed_qa:
            failures.append(f"required data QA checks did not pass: {failed_qa}")

    statistics = payload.get("statistics")
    required_statistics = {
        "paired_by_source",
        "family_preregistered",
        "calibration_isolated",
    }
    if not isinstance(statistics, Mapping):
        failures.append("statistical governance record is missing")
    else:
        failed_stats = sorted(
            key for key in required_statistics if statistics.get(key) is not True
        )
        if failed_stats:
            failures.append(
                f"required statistical controls did not pass: {failed_stats}"
            )
    return GateResult(not failures, tuple(failures))
