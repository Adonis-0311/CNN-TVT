"""Small repository audit used before an evidence bundle is considered releasable."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditIssue:
    path: str
    message: str


def audit_repository(
    root: str | Path,
    *,
    semantic_firewall_paths: Iterable[str | Path] = (),
) -> list[AuditIssue]:
    repository = Path(root).resolve()
    issues: list[AuditIssue] = []
    for path in repository.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(AuditIssue(str(path.relative_to(repository)), f"invalid JSON: {exc}"))
    for path in repository.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error) as exc:
            issues.append(AuditIssue(str(path.relative_to(repository)), f"invalid CSV: {exc}"))
            continue
        if rows and "claim_id" in rows[0]:
            ids = [row.get("claim_id", "").strip() for row in rows]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                issues.append(
                    AuditIssue(str(path.relative_to(repository)), "claim_id is empty or duplicated")
                )

    forbidden = ("orbit_pass", "16apsk", "32apsk", "satellite_channel")
    for supplied in semantic_firewall_paths:
        path = (repository / supplied).resolve()
        try:
            path.relative_to(repository)
        except ValueError:
            issues.append(AuditIssue(str(supplied), "semantic-firewall path escapes root"))
            continue
        text = path.read_text(encoding="utf-8").lower()
        found = [token for token in forbidden if token in text]
        if found:
            issues.append(
                AuditIssue(str(path.relative_to(repository)), f"foreign scenario tokens: {found}")
            )
    return issues
