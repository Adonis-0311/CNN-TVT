"""Claim-to-evidence ledger validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

_STATUSES = {"planned", "blocked", "in_progress", "supported", "rejected"}


class ClaimLedgerError(ValueError):
    pass


def validate_claim_rows(rows: list[Mapping[str, str]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        claim_id = (row.get("claim_id") or "").strip()
        status = (row.get("status") or "").strip()
        evidence = (row.get("evidence_artifact") or "").strip()
        if not claim_id or claim_id in seen:
            raise ClaimLedgerError(f"row {index}: claim_id is missing or duplicated")
        seen.add(claim_id)
        if status not in _STATUSES:
            raise ClaimLedgerError(f"row {index}: invalid status {status!r}")
        if status == "supported" and not evidence:
            raise ClaimLedgerError(
                f"row {index}: supported claim requires evidence_artifact"
            )


def load_and_validate_claim_ledger(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_claim_rows(rows)
    return rows
