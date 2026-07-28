"""Create-new publication and checksum-closed run-manifest validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from .freeze import canonical_json_bytes

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = "vimd_amc.tvt.run_manifest.v1"


class ManifestError(ValueError):
    """Raised when a run cannot supply immutable, closed evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON and refuse to overwrite any existing target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.publish.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"publication lock already exists: {lock}") from exc
    temporary: Path | None = None
    try:
        os.close(descriptor)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite immutable file: {target}")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable file: {target}"
            ) from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def validate_run_manifest(
    path: str | Path,
    *,
    required_artifacts: Iterable[str] = (),
) -> Mapping[str, Any]:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load run manifest: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest root must be an object")
    if payload.get("schema") != _SCHEMA:
        raise ManifestError(f"schema must equal {_SCHEMA!r}")
    if payload.get("execution_status") != "completed":
        raise ManifestError("only completed runs can be evidence")
    frozen_digest = payload.get("frozen_config_sha256")
    if not isinstance(frozen_digest, str) or _SHA256.fullmatch(frozen_digest) is None:
        raise ManifestError("frozen_config_sha256 must be a lowercase SHA-256")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ManifestError("artifacts must be an object")
    missing = sorted(set(required_artifacts) - set(artifacts))
    if missing:
        raise ManifestError(f"required artifacts missing: {missing}")

    root = manifest_path.parent
    for name, record in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            raise ManifestError("artifact entries must be named objects")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ManifestError(f"artifact {name!r} path must be relative")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ManifestError(f"artifact {name!r} has invalid sha256")
        candidate = (root / relative).resolve()
        if not _inside(root, candidate):
            raise ManifestError(f"artifact {name!r} escapes the attempt directory")
        if not candidate.is_file():
            raise ManifestError(f"artifact {name!r} does not exist: {relative}")
        actual = _sha256_file(candidate)
        if actual != expected:
            raise ManifestError(f"artifact {name!r} checksum mismatch")
    return payload
