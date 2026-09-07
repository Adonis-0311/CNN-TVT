"""Immutable experiment-specification loading for vehicular AMC evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = "vimd_amc.tvt.freeze.v1"


class FreezeError(ValueError):
    """Raised when an experiment specification is not safely frozen."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenConfig:
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment_id"])


def _validate_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FreezeError(f"{field} must be a lowercase SHA-256 digest")


def validate_freeze_spec(payload: Mapping[str, Any], *, allow_template: bool = False) -> None:
    if payload.get("schema") != _SCHEMA:
        raise FreezeError(f"schema must equal {_SCHEMA!r}")
    allowed_status = {"frozen_before_results"}
    if allow_template:
        allowed_status.add("template_not_frozen")
    if payload.get("status") not in allowed_status:
        raise FreezeError("configuration is not frozen before results")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise FreezeError("experiment_id must be non-empty")

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ):
        raise FreezeError("seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)):
        raise FreezeError("seeds must be unique")

    roles = payload.get("split_roles")
    required_roles = {"train", "validation", "calibration", "test"}
    if not isinstance(roles, Mapping) or set(roles) != required_roles:
        raise FreezeError(f"split_roles must define exactly {sorted(required_roles)}")
    split_names: list[str] = []
    for role, names in roles.items():
        if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise FreezeError(f"split_roles.{role} must be a non-empty string list")
        split_names.extend(names)
    if len(split_names) != len(set(split_names)):
        raise FreezeError("a split cannot serve more than one access role")

    manifests = payload.get("data_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise FreezeError("data_manifests must be a non-empty list")
    for index, item in enumerate(manifests):
        if not isinstance(item, Mapping):
            raise FreezeError(f"data_manifests[{index}] must be an object")
        path = item.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise FreezeError(f"data_manifests[{index}].path must be relative")
        _validate_sha256(item.get("sha256"), f"data_manifests[{index}].sha256")


def load_frozen_config(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    allow_template: bool = False,
) -> FrozenConfig:
    config_path = Path(path).resolve()
    file_digest = sha256_file(config_path)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, "expected_sha256")
        if file_digest != expected_sha256:
            raise FreezeError(
                f"frozen configuration digest mismatch: {file_digest} != {expected_sha256}"
            )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot load frozen configuration: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FreezeError("frozen configuration root must be an object")
    validate_freeze_spec(payload, allow_template=allow_template)
    return FrozenConfig(config_path, file_digest, payload)
