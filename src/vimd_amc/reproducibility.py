"""Small source-provenance helpers for artifact-producing runners."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def source_tree_record(
    project_root: Path,
    entrypoint: Path,
) -> dict[str, Any]:
    """Hash the executable Python source associated with one run.

    The workspace is not required to be a Git repository, so artifacts bind
    themselves directly to file content.  Paths are relative and sorted to
    make the aggregate digest stable across checkout locations.
    """

    root = project_root.resolve()
    candidates = list((root / "src" / "vimd_amc").rglob("*.py"))
    candidates.extend((entrypoint.resolve(), root / "pyproject.toml"))
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
        "algorithm": "sha256",
        "aggregate_digest": aggregate.hexdigest(),
        "files": records,
    }


__all__ = ["source_tree_record"]
