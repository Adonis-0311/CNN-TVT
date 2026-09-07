"""Validate the honest TVT integration manuscript without unlocking submission.

This validator is intentionally separate from ``validate_paper_build.py``.
The legacy gate represents a positive-release path in which every scientific
gate passes and ``results_auto.tex`` is release locked.  The present manuscript
has a different, fail-closed contract: it must disclose the failed gates,
preserve sealed/exploratory evidence classes, and remain submission-locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_MAIN = REPO / "paper" / "main.tex"
DEFAULT_MACROS = REPO / "paper_data_layer" / "outputs" / "paper_macros.tex"
DEFAULT_NUMBERS = REPO / "paper_data_layer" / "outputs" / "paper_numbers.json"
DEFAULT_LOG = REPO / "paper" / "build_integrated" / "main.log"
DEFAULT_PDF = REPO / "output" / "pdf" / "tvt_operating_envelope_integration.pdf"
DEFAULT_REPORT = REPO / "tvt_submission" / "honest_paper_release.json"

REQUIRED_FAILED_GATES = {
    "confirmatory_family_gate_failed",
    "clean_retention_gate_failed",
}
REQUIRED_DISCLOSURES = (
    r"family gate (?:did not pass|does not pass)",
    r"clean-retention gate (?:did not pass|does not pass)",
    r"post hoc|post-hoc",
    r"exploratory\s+and\s+outside\s+the\s+frozen\s+confirmatory\s+family",
    r"simulation.only|scope is simulation only",
)
FORBIDDEN = (
    r"\bmeasured\b",
    r"\bon-?board\b",
    r"\bfield trial\b",
    r"\boperational evidence\b",
    r"\bin-?orbit\b",
)
PLACEHOLDERS = (r"\bTODO\b", r"\bTBD\b", r"No eligible locked run", r"undefined reference")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def artifact(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def check(checks: list[dict], issues: list[str], check_id: str, passed: bool,
          actual, required) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "actual": actual, "required": required})
    if not passed:
        issues.append(f"{check_id}: expected {required!r}, got {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--macros", type=Path, default=DEFAULT_MACROS)
    parser.add_argument("--numbers", type=Path, default=DEFAULT_NUMBERS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    paths = {
        "main_tex": args.main.resolve(),
        "paper_macros": args.macros.resolve(),
        "paper_numbers": args.numbers.resolve(),
        "latex_log": args.log.resolve(),
        "pdf": args.pdf.resolve(),
    }
    checks: list[dict] = []
    issues: list[str] = []
    artifacts: dict[str, dict] = {}
    for name, path in paths.items():
        valid = path.is_file() and path.stat().st_size > 0
        check(checks, issues, f"artifact_{name}", valid,
              "stable_nonempty_file" if valid else "missing", "stable_nonempty_file")
        if valid:
            artifacts[name] = artifact(path)

    if issues:
        record = {
            "schema": "vimd_amc.tvt.honest_paper_release.v1",
            "honest_manuscript_validated": False,
            "submission_unlocked": False,
            "scientific_evidence_passed": False,
            "failed_gates": sorted(REQUIRED_FAILED_GATES),
            "checks": checks,
            "issues": issues,
            "artifacts": artifacts,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False))
        return 1

    manuscript = paths["main_tex"].read_text(encoding="utf-8")
    macros = paths["paper_macros"].read_text(encoding="utf-8")
    log = paths["latex_log"].read_text(encoding="utf-8", errors="replace")
    numbers = json.loads(paths["paper_numbers"].read_text(encoding="utf-8"))

    gate = numbers.get("gate_status", {})
    failed = set(gate.get("failed_gates", []))
    check(checks, issues, "submission_remains_locked",
          gate.get("submission_unlocked") is False, gate.get("submission_unlocked"), False)
    check(checks, issues, "scientific_gate_not_rewritten",
          gate.get("scientific_evidence_passed") is False,
          gate.get("scientific_evidence_passed"), False)
    check(checks, issues, "failed_gate_set", failed == REQUIRED_FAILED_GATES,
          sorted(failed), sorted(REQUIRED_FAILED_GATES))

    for index, pattern in enumerate(REQUIRED_DISCLOSURES, start=1):
        found = bool(re.search(pattern, manuscript, flags=re.IGNORECASE | re.DOTALL))
        check(checks, issues, f"required_disclosure_{index:02d}", found,
              pattern if found else "missing", pattern)
    for index, pattern in enumerate(FORBIDDEN, start=1):
        found = bool(re.search(pattern, manuscript, flags=re.IGNORECASE))
        check(checks, issues, f"forbidden_term_{index:02d}", not found,
              pattern if found else "absent", "absent")

    include_ok = "\\input{../paper_data_layer/outputs/paper_macros.tex}" in manuscript
    check(checks, issues, "artifact_macro_input", include_ok, include_ok, True)
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    evidence_keys = set(numbers.get("keys", {}))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", manuscript)) & evidence_keys
    missing_macros = sorted(used - defined)
    check(checks, issues, "used_evidence_macros_defined", not missing_macros,
          missing_macros, [])

    stale_sources = []
    for relative, expected in numbers.get("source_sha256", {}).items():
        source = REPO / Path(relative)
        if not source.is_file() or digest(source) != expected:
            stale_sources.append(relative)
    check(checks, issues, "source_digest_closure", not stale_sources, stale_sources, [])

    log_patterns = {
        "fatal_errors": r"LaTeX Error|Undefined control sequence|Emergency stop|Fatal error",
        "undefined_citations": r"Citation .* undefined",
        "undefined_references": r"undefined references|Reference .* undefined",
        "overfull_boxes": r"Overfull \\[hv]box",
        "rerun_required": r"Rerun to get cross-references right",
    }
    for name, pattern in log_patterns.items():
        count = len(re.findall(pattern, log, flags=re.IGNORECASE))
        check(checks, issues, f"log_no_{name}", count == 0, count, 0)

    output = re.findall(r"Output written on .*?\((\d+) pages?,\s*(\d+) bytes\)", log, flags=re.DOTALL)
    check(checks, issues, "single_log_output", len(output) == 1, output,
          "one page/byte record")
    if len(output) == 1:
        pages, logged_bytes = map(int, output[0])
        check(checks, issues, "positive_page_count", pages > 0, pages, "> 0")
        actual_bytes = paths["pdf"].stat().st_size
        check(checks, issues, "log_pdf_byte_match", logged_bytes == actual_bytes,
              {"log": logged_bytes, "pdf": actual_bytes}, "exact equality")

    signature = paths["pdf"].read_bytes()[:8]
    check(checks, issues, "pdf_signature", signature.startswith(b"%PDF-"),
          signature.decode("ascii", errors="replace"), "%PDF-")
    newest_source = max(paths["main_tex"].stat().st_mtime, paths["paper_macros"].stat().st_mtime)
    fresh = min(paths["latex_log"].stat().st_mtime, paths["pdf"].stat().st_mtime) >= newest_source
    check(checks, issues, "build_freshness", fresh, fresh, True)

    extracted = ""
    try:
        completed = subprocess.run(
            ["pdftotext", str(paths["pdf"]), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        extracted = completed.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append(f"pdf_text_extraction: {exc}")
        checks.append({"id": "pdf_text_extraction", "passed": False,
                       "actual": str(exc), "required": "successful pdftotext"})
    else:
        check(checks, issues, "pdf_text_extraction", len(extracted) > 1000,
              len(extracted), "> 1000 characters")
        placeholders = [p for p in PLACEHOLDERS if re.search(p, extracted, flags=re.IGNORECASE)]
        check(checks, issues, "pdf_no_placeholders", not placeholders, placeholders, [])
        for index, pattern in enumerate(REQUIRED_DISCLOSURES[:4], start=1):
            found = bool(re.search(pattern, extracted, flags=re.IGNORECASE | re.DOTALL))
            check(checks, issues, f"pdf_disclosure_{index:02d}", found,
                  pattern if found else "missing", pattern)

    ok = not issues
    record = {
        "schema": "vimd_amc.tvt.honest_paper_release.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript_mode": "honest_scientific_integration",
        "honest_manuscript_validated": ok,
        "submission_unlocked": False,
        "scientific_evidence_passed": False,
        "failed_gates": sorted(REQUIRED_FAILED_GATES),
        "evidence_key_count": len(evidence_keys),
        "checks": checks,
        "issues": issues,
        "artifacts": artifacts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
