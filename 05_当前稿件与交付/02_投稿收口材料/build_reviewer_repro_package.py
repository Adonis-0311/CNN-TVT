"""Build a deterministic lightweight reviewer reproduction archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vimd_amc.tvt.reviewer_reproduction.v1"
ZIP_PREFIX = "tvt_reviewer_reproduction_v1"
FIXED_TIME = (2026, 8, 17, 0, 0, 0)
MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_OUTPUT = ROOT / "output" / "repro" / f"{ZIP_PREFIX}.zip"

EXACT_FILES = {
    "README.md",
    "pyproject.toml",
    "TVT_MULTI_AGENT_WORK_BREAKDOWN.md",
    "docs/TVT_REPOSITIONING_AND_CLAIM_LADDER.md",
    "docs/RECENT_COMPARATOR_AUDIT.md",
    "docs/RECENT_INTERFERENCE_BASELINE_AUDIT.md",
    "docs/RECENT_LITERATURE_POSITIONING_2026.md",
    "docs/FINAL_TVT_REVIEWER_ATTACK_AUDIT.md",
    "analysis_zero_compute/RESULTS.md",
    "analysis_zero_compute/tier2_gpu/PREREGISTRATION_TIER2.md",
    "paper/main.tex",
    "paper/references.bib",
    "paper/IEEEtran.cls",
    "paper/figures/physical_teacher_example.pdf",
    "paper/authors_verified.example.tex",
    "paper/SUBMISSION_READINESS.md",
    "paper/EVIDENCE_LEDGER.md",
    "paper/AUTHOR_SUBMISSION_SIGNOFF.md",
    "paper_data_layer/build_paper_numbers.py",
    "paper_data_layer/validate_paper_numbers.py",
    "paper_data_layer/outputs/paper_numbers.json",
    "paper_data_layer/outputs/paper_macros.tex",
    "tvt_submission/REVIEWER_REPRODUCTION_PACKAGE.md",
    "tvt_submission/build_reviewer_repro_package.py",
    "tvt_submission/validate_reviewer_repro_package.py",
    "tvt_submission/validate_honest_paper_release.py",
    "tvt_submission/honest_paper_release.json",
    "tvt_submission/configs/formal_tvt_freeze_v4_8gb_dual.json",
    "tvt_submission/configs/formal_tvt_recovery_v4r_110plus10.json",
    "output/pdf/tvt_operating_envelope_integration.pdf",
}

TREE_RULES = {
    "analysis_zero_compute": {".py"},
    "analysis_zero_compute/outputs/csv": {".csv", ".json"},
    "analysis_zero_compute/tier2_gpu": {".py", ".md", ".ps1"},
    "paper_figures": {".py", ".csv", ".json", ".pdf", ".png", ".md"},
    "tvt_submission/sources": {".json", ".txt", ".md"},
}

SUMMARY_ARTIFACTS = (
    "artifacts/tvt_v4r_headline_composite",
    "artifacts/tier2_condition_coverage_v1",
    "artifacts/tier2_iq_sidecar_v1",
    "artifacts/tier2_capacity_M_v2",
    "artifacts/tier2_capacity_L_v1",
    "artifacts/tier2_presence_gated_v1",
)
SUMMARY_NAMES = {
    "metrics.csv",
    "seed_aggregates.csv",
    "headline_paired_statistics.csv",
    "paired_statistics.csv",
    "tier2_execution_context.json",
}


class PackageError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    token = PurePosixPath(path.resolve().relative_to(ROOT.resolve()).as_posix())
    if token.is_absolute() or ".." in token.parts:
        raise PackageError(f"unsafe path: {path}")
    return token.as_posix()


def collect() -> list[Path]:
    files = {ROOT / item for item in EXACT_FILES}
    for tree, suffixes in TREE_RULES.items():
        base = ROOT / tree
        if not base.is_dir():
            continue
        files.update(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in suffixes
            and "__pycache__" not in path.parts
        )
    for artifact in SUMMARY_ARTIFACTS:
        base = ROOT / artifact
        if not base.is_dir():
            continue
        files.update(path for path in base.iterdir() if path.name in SUMMARY_NAMES)
    missing = sorted(relative(path) for path in files if not path.is_file())
    if missing:
        raise PackageError("missing required files: " + ", ".join(missing))
    selected = sorted(files, key=relative)
    for path in selected:
        size = path.stat().st_size
        if size <= 0 or size > MAX_FILE_BYTES:
            raise PackageError(f"invalid package file size: {relative(path)} ({size})")
    return selected


def heavy_ledgers() -> list[dict[str, object]]:
    records = []
    for artifact in SUMMARY_ARTIFACTS:
        run = ROOT / artifact / "run.json"
        record: dict[str, object] = {
            "artifact": artifact,
            "included": False,
            "reason": "heavyweight run ledger; summary tables are packaged",
            "present": run.is_file(),
        }
        if run.is_file():
            data = run.read_bytes()
            record.update(bytes=len(data), sha256=sha256_bytes(data))
        records.append(record)
    return records


def manifest(files: list[Path]) -> dict[str, object]:
    rows = []
    for path in files:
        data = path.read_bytes()
        rows.append(
            {
                "path": relative(path),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    return {
        "schema_version": SCHEMA,
        "package_role": "lightweight_reviewer_claim_audit",
        "simulation_only": True,
        "full_training_self_contained": False,
        "tier2_evidence_class": "exploratory_outside_frozen_confirmatory_family",
        "files": rows,
        "excluded_heavy_run_ledgers": heavy_ledgers(),
        "excluded_classes": [
            "cache arrays and root per-window manifest",
            "model checkpoints",
            "per-window prediction bundles",
            "logs and interrupted runs",
        ],
    }


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build(output: Path, *, replace: bool) -> dict[str, object]:
    files = collect()
    payload = manifest(files)
    target = output.resolve()
    if target.exists() and not replace:
        raise PackageError(f"output exists; use --replace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    manifest_data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.writestr(zip_info(f"{ZIP_PREFIX}/{relative(path)}"), path.read_bytes())
            archive.writestr(zip_info(f"{ZIP_PREFIX}/PACKAGE_MANIFEST.json"), manifest_data)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_bytes(target.read_bytes())
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return {
        "ok": True,
        "archive": str(target),
        "archive_bytes": target.stat().st_size,
        "archive_sha256": digest,
        "file_count": len(files),
        "manifest_member": f"{ZIP_PREFIX}/PACKAGE_MANIFEST.json",
        "sha256_sidecar": str(sidecar),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.output, replace=args.replace)
    except (OSError, ValueError, PackageError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
