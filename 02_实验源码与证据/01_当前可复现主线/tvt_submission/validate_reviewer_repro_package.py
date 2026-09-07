"""Validate the hashes and membership of a reviewer reproduction ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


PREFIX = "tvt_reviewer_reproduction_v1"
MANIFEST = f"{PREFIX}/PACKAGE_MANIFEST.json"


def validate(path: Path) -> dict[str, object]:
    errors: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            errors.append("duplicate ZIP members")
        for name in names:
            token = PurePosixPath(name)
            if token.is_absolute() or ".." in token.parts:
                errors.append(f"unsafe member: {name}")
        if MANIFEST not in names:
            return {"ok": False, "errors": errors + ["manifest missing"]}
        manifest = json.loads(archive.read(MANIFEST))
        expected = {f"{PREFIX}/{row['path']}" for row in manifest["files"]}
        actual = set(names) - {MANIFEST}
        for missing in sorted(expected - actual):
            errors.append(f"missing member: {missing}")
        for unexpected in sorted(actual - expected):
            errors.append(f"unexpected member: {unexpected}")
        for row in manifest["files"]:
            name = f"{PREFIX}/{row['path']}"
            if name not in actual:
                continue
            data = archive.read(name)
            if len(data) != row["bytes"]:
                errors.append(f"size mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                errors.append(f"hash mismatch: {name}")
    return {
        "ok": not errors,
        "archive": str(path.resolve()),
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validated_files": len(expected),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = validate(args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
