"""Build a deterministic, explicitly non-upload-ready TVT handoff archive.

The archive contains source, tests, audit documents, the IEEE template class,
and the current internal-review paper.  It deliberately excludes caches,
training artifacts, diagnostics, user source documents, and any formal result
that has not passed the release chain.  Dry-run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "deliverables"
    / "TVT_VIMD_Net_pre_submission_handoff_20260728.zip"
)
SCHEMA = "vimd_amc.tvt.pre_submission_handoff.v1"
FIXED_ZIP_TIME = (2026, 7, 28, 0, 0, 0)
# This is an explicit schema-snapshot marker, not a filesystem timestamp.
# Change it only for an intentional package-schema snapshot so identical
# content remains byte-reproducible across clones and checkouts.
SOURCE_SNAPSHOT_UTC = "2026-07-28T00:00:00+00:00"
MAX_FILE_BYTES = 20 * 1024 * 1024

ROOT_FILES = {
    "HANDOFF.md",
    "README.md",
    "pyproject.toml",
}
TREE_SUFFIXES = {
    "src": {".py"},
    "experiments": {".py"},
    "tests": {".py"},
    "docs": {".md"},
    "standards": {".py", ".m", ".md"},
}
PAPER_FILES = {
    "IEEEtran.cls",
    "main.tex",
    "references.bib",
    "results_auto.tex",
    "authors_verified.example.tex",
    "EVIDENCE_LEDGER.md",
    "SUBMISSION_READINESS.md",
    "TEMPLATE_PROVENANCE.md",
}
TVT_SUFFIXES = {
    ".py",
    ".ps1",
    ".json",
    ".md",
    ".csv",
    ".txt",
}
BLOCKED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "cache_factor_screening_1024_v1",
    "cache_factor_headline_1024_v1",
    "tmp",
    "logs",
}
BLOCKED_NAMES = {
    "release_lock.json",
    "formal_macro_values.json",
}


class PackageError(RuntimeError):
    """The handoff archive could not be constructed safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(path: Path) -> str:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    token = PurePosixPath(relative.as_posix())
    if token.is_absolute() or ".." in token.parts:
        raise PackageError(f"unsafe package path: {path}")
    return token.as_posix()


def _eligible(path: Path, suffixes: set[str]) -> bool:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    if any(part in BLOCKED_PARTS for part in relative.parts):
        return False
    if path.name in BLOCKED_NAMES:
        return False
    return path.is_file() and path.suffix.casefold() in suffixes


def _collect() -> list[Path]:
    selected: set[Path] = set()
    for name in ROOT_FILES:
        selected.add(PROJECT_ROOT / name)
    for directory, suffixes in TREE_SUFFIXES.items():
        root = PROJECT_ROOT / directory
        selected.update(
            path for path in root.rglob("*") if _eligible(path, suffixes)
        )
    paper_root = PROJECT_ROOT / "paper"
    selected.update(paper_root / name for name in PAPER_FILES)
    figures = paper_root / "figures"
    if figures.is_dir():
        selected.update(
            path
            for path in figures.rglob("*")
            if _eligible(path, {".pdf", ".png", ".jpg", ".jpeg", ".eps"})
        )
    internal_pdf = paper_root / "build" / "main.pdf"
    if internal_pdf.is_file():
        selected.add(internal_pdf)
    tvt_root = PROJECT_ROOT / "tvt_submission"
    selected.update(
        path
        for path in tvt_root.rglob("*")
        if _eligible(path, TVT_SUFFIXES)
    )

    missing = sorted(str(path) for path in selected if not path.is_file())
    if missing:
        raise PackageError("required package files are missing: " + ", ".join(missing))
    files = sorted(selected, key=_safe_relative)
    relative = [_safe_relative(path) for path in files]
    if len(relative) != len(set(relative)):
        raise PackageError("duplicate package-relative paths")
    for path in files:
        size = path.stat().st_size
        if size <= 0:
            raise PackageError(f"empty package file: {_safe_relative(path)}")
        if size > MAX_FILE_BYTES:
            raise PackageError(
                f"package file exceeds {MAX_FILE_BYTES} bytes: "
                f"{_safe_relative(path)}"
            )
    return files


def _role(relative: str) -> str:
    if relative.startswith("paper/"):
        return "internal_paper"
    if relative.startswith("tests/"):
        return "test"
    if relative.startswith("docs/"):
        return "audit"
    if relative.startswith("tvt_submission/sources/"):
        return "provenance_or_license"
    if relative.startswith("tvt_submission/"):
        return "release_or_handoff"
    return "source"


def _external_sources() -> list[dict[str, object]]:
    candidates = (
        (
            PROJECT_ROOT.parent
            / "DFI257727-基于神经网络的干扰环境下信号调制识别方法及系统(定稿).docx",
            "patent_source",
        ),
        (
            PROJECT_ROOT.parent / "TVT_Flagship_VIMD_Net_AMC_Full_Design_Idea.md",
            "design_idea",
        ),
        (
            Path(
                r"C:\Users\Administrator\Downloads"
                r"\IEEE-Transactions-LaTeX2e-templates-and-instructions.zip"
            ),
            "ieee_template_archive",
        ),
    )
    records: list[dict[str, object]] = []
    for path, role in candidates:
        if not path.is_file():
            records.append(
                {
                    "name": path.name,
                    "role": role,
                    "present_at_packaging": False,
                }
            )
            continue
        data = path.read_bytes()
        records.append(
            {
                "name": path.name,
                "role": role,
                "present_at_packaging": True,
                "bytes": len(data),
                "sha256": _sha256(data),
                "included": False,
            }
        )
    return records


def build_manifest(files: list[Path]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in files:
        data = path.read_bytes()
        relative = _safe_relative(path)
        records.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": _sha256(data),
                "role": _role(relative),
            }
        )
    return {
        "schema_version": SCHEMA,
        "source_snapshot_utc": SOURCE_SNAPSHOT_UTC,
        "package_role": "pre_submission_handoff",
        "upload_ready": False,
        "reason_not_upload_ready": [
            "formal headline cache is absent",
            "five-seed formal run is absent",
            "eligible result release lock is absent",
            "verified human authorship file is absent",
            "human TVT AI-policy, citation, patent-timing, and disclosure review is pending",
        ],
        "long_running_work_started_by_packager": False,
        "files": records,
        "external_sources": _external_sources(),
        "excluded_project_trees": [
            "artifacts/",
            "diagnostics/",
            "logs/",
            "tmp/",
            "standards/cache_factor_screening_1024_v1/",
            "standards/cache_factor_headline_1024_v1/",
            "../tccn_satellite_amc/",
        ],
    }


def write_archive(
    output: Path,
    files: list[Path],
    manifest: dict[str, object],
    *,
    replace: bool,
) -> dict[str, object]:
    target = output.expanduser().resolve()
    if target.exists() and not replace:
        raise PackageError(f"output exists; pass --replace to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    manifest_data = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in files:
                info = zipfile.ZipInfo(
                    filename=f"TVT_VIMD_Net/{_safe_relative(path)}",
                    date_time=FIXED_ZIP_TIME,
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
            info = zipfile.ZipInfo(
                filename="TVT_VIMD_Net/PRE_SUBMISSION_MANIFEST.json",
                date_time=FIXED_ZIP_TIME,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, manifest_data)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return {
        "archive": str(target),
        "archive_bytes": target.stat().st_size,
        "archive_sha256": digest,
        "sha256_sidecar": str(sidecar),
        "file_count": len(files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    try:
        files = _collect()
        manifest = build_manifest(files)
        result: dict[str, object] = {
            "ok": True,
            "dry_run": not arguments.write,
            "output": str(arguments.output.expanduser().resolve()),
            "file_count": len(files),
            "total_uncompressed_bytes": sum(
                int(record["bytes"]) for record in manifest["files"]
            ),
            "upload_ready": False,
        }
        if arguments.write:
            result.update(
                write_archive(
                    arguments.output,
                    files,
                    manifest,
                    replace=arguments.replace,
                )
            )
    except (OSError, ValueError, PackageError) as error:
        print(
            json.dumps(
                {"ok": False, "error": str(error)},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 1
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
    sys.exit(main())
