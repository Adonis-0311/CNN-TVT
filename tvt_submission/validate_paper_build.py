"""Read-only, fail-closed validator for a built TVT manuscript.

The validator does not invoke LaTeX and never writes a file.  It checks that
the final log and PDF are mutually consistent, that the build is newer than
its source inputs, and that internal-review versus release state is explicit.
Every CLI outcome is emitted as one JSON object on stdout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

try:
    from tvt_submission import validate_release as release_contract
except ModuleNotFoundError:  # Direct ``python tvt_submission/...py`` execution.
    import validate_release as release_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "vimd_amc.tvt.paper_build_gate.v2"
RELEASE_LOCK_SCHEMA = release_contract.RELEASE_LOCK_SCHEMA
FORMAL_CACHE_DESIGNATION = "headline_formal_tvt_evidence"
DEFAULT_MAX_RELEASE_PAGES = 14
DEFAULT_MTIME_TOLERANCE_SECONDS = 2.0
DEFAULT_BUILD_ARTIFACT_SKEW_SECONDS = 60.0
DEFAULT_PDF_TOOL_TIMEOUT_SECONDS = 30.0

RELEASE_SENTINEL = release_contract.RELEASE_SENTINEL
RELEASE_SENTINEL_VALUE = release_contract.RELEASE_SENTINEL_VALUE
TEXT_RESULT_MACROS = (
    "ResultSource",
    "PrimaryReference",
    "VIMDLatencyDevice",
)
HEADLINE_MODELS = ("AZero", "MCLDNN", "IQFormer", "CSSL", "AFive")
HEADLINE_METRICS = (
    "Accuracy",
    "MacroFOne",
    "WorstRecall",
    "NLL",
    "ECE",
)
HEADLINE_NUMERIC_MACROS = tuple(
    f"HeadlineHard{model}{metric}"
    for model in HEADLINE_MODELS
    for metric in HEADLINE_METRICS
)
REGIME_TOKENS = (
    "Hard",
    "UnseenJammer",
    "UnseenSpeed",
    "HeldoutChannel",
    "CombinedOOD",
    "CleanACD",
    "CleanBE",
)
REGIME_FIELDS = ("Reference", "AFive", "Gain", "CILow", "CIHigh")
REGIME_NUMERIC_MACROS = tuple(
    f"Regime{regime}{field}"
    for regime in REGIME_TOKENS
    for field in REGIME_FIELDS
)
MECHANISM_NUMERIC_MACROS = (
    "MechanismMaskJS",
    "MechanismThirdRouteWeightedCorrelation",
    "MechanismTargetTransferRatio",
    "MechanismTargetAmplificationShare",
    "MechanismJammerLeakage",
    "MechanismThirdRouteSpearman",
    "MechanismThirdRoutePermutationP",
    "OracleSpectralRatioGain",
    "VIMDParameters",
    "VIMDLatencyP50",
    "VIMDLatencyP95",
)
NUMERIC_RESULT_MACROS = (
    *HEADLINE_NUMERIC_MACROS,
    *REGIME_NUMERIC_MACROS,
    *MECHANISM_NUMERIC_MACROS,
)
RESULT_MACROS = (
    *TEXT_RESULT_MACROS,
    *NUMERIC_RESULT_MACROS,
)
PROVENANCE_MACROS = (
    "PrimaryReference",
    "VIMDLatencyDevice",
    *NUMERIC_RESULT_MACROS,
)
LEGACY_RESULT_MACROS = (
    "StrongestBaseline",
    "HardMacroFOneGain",
    "HardMacroFOneCI",
    "HeldoutJammerGain",
    "HeldoutChannelGain",
    "FeatureSIRGain",
    "VIMDLatency",
)
PLACEHOLDER_TERMS = release_contract.PLACEHOLDER_TERMS
SHA256_TEXT = re.compile(r"^[0-9a-f]{64}$")
PRIMARY_REFERENCE_VALUE = (
    "CSSL-AMC official-architecture supervised adaptation"
)
CANONICAL_FINITE_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)

REVIEW_DIRECTIVE = re.compile(
    r"^\s*\\internalreview(?P<state>true|false)\s*$",
    flags=re.IGNORECASE,
)
RESULT_MACRO_LINE = re.compile(
    r"^\s*\\newcommand\{\\(?P<name>[A-Za-z][A-Za-z0-9]*)\}"
    r"\{(?P<value>.*)\}\s*$"
)
OUTPUT_RECORD = re.compile(
    r"Output\s+written\s+on\s+.+?"
    r"\(\s*(?P<pages>\d+)\s+pag\s*es?\s*,\s*"
    r"(?P<bytes>[\d,]+)\s+bytes\s*\)\.",
    flags=re.IGNORECASE | re.DOTALL,
)
FATAL_PATTERNS = (
    re.compile(r"^\s*!\s+"),
    re.compile(r"\bEmergency\s+stop\b", flags=re.IGNORECASE),
    re.compile(r"\bFatal\s+error\b", flags=re.IGNORECASE),
    re.compile(r"\bNo\s+pages\s+of\s+output\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:LaTeX|pdfTeX|Package\s+\S+|Class\s+\S+)\s+Error\b",
        flags=re.IGNORECASE,
    ),
)
CITATION_UNDEFINED = re.compile(
    r"(?:\b(?:citation|citations)\b.*\bundefined\b|"
    r"\bundefined\b.*\b(?:citation|citations)\b)",
    flags=re.IGNORECASE,
)
REFERENCE_UNDEFINED = re.compile(
    r"(?:\b(?:reference|references)\b.*\bundefined\b|"
    r"\bundefined\b.*\b(?:reference|references)\b)",
    flags=re.IGNORECASE,
)
OVERFULL_BOX = re.compile(
    r"\bOverfull\s+\\[hv]box\b",
    flags=re.IGNORECASE,
)
RERUN_REQUIRED = re.compile(
    r"(?:Rerun\s+to\s+get\s+cross-references\s+right|"
    r"Rerun\s+to\s+get\s+outlines\s+right|"
    r"Label\(s\)\s+may\s+have\s+changed|"
    r"Please\s+\(re\)run\s+(?:Biber|BibTeX)|"
    r"Please\s+rerun\s+LaTeX)",
    flags=re.IGNORECASE,
)
TEX_CONDITIONAL_TOKEN = re.compile(
    r"\\(?P<command>newif|if[A-Za-z@]+|else|fi)(?![A-Za-z@])"
)
PUBLIC_BRANCH_PLACEHOLDER = re.compile(
    r"(?<![A-Za-z])(?:pending|generated)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
PDF_PLACEHOLDER_LINE = re.compile(
    r"""^\s*["'`“”‘’]*(?:pending|generated)
        ["'`“”‘’.,;:!?]*\s*$""",
    flags=re.IGNORECASE | re.VERBOSE,
)
PDFINFO_PAGES = re.compile(r"^Pages:\s*(?P<pages>\d+)\s*$", re.MULTILINE)
PDFINFO_FILE_SIZE = re.compile(
    r"^File\s+size:\s*(?P<bytes>\d+)\s+bytes\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class PaperBuildValidationError(RuntimeError):
    """An artifact could not be read as a stable, bounded input."""


class JSONArgumentParser(argparse.ArgumentParser):
    """Argument parser whose failures obey the JSON-only stdout contract."""

    def error(self, message: str) -> None:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "ok": False,
                    "mode": None,
                    "issues": [f"argument_error: {message}"],
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        raise SystemExit(2)


@dataclass(frozen=True)
class StableArtifact:
    """A file snapshot that did not change while it was read."""

    path: Path
    data: bytes
    size: int
    mtime_ns: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size,
            "mtime_utc": datetime.fromtimestamp(
                self.mtime_ns / 1_000_000_000,
                tz=timezone.utc,
            ).isoformat(),
            "sha256": self.sha256,
        }


def _strip_tex_comment(line: str) -> str:
    """Remove a TeX comment while respecting escaped percent characters."""

    for index, character in enumerate(line):
        if character != "%":
            continue
        preceding_backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            preceding_backslashes += 1
            cursor -= 1
        if preceding_backslashes % 2 == 0:
            return line[:index]
    return line


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _stable_read(path: Path, label: str) -> StableArtifact:
    resolved = path.expanduser().resolve()
    try:
        before = resolved.stat()
    except OSError as error:
        raise PaperBuildValidationError(
            f"{label} is missing or unreadable: {resolved}: {error}"
        ) from error
    if not resolved.is_file():
        raise PaperBuildValidationError(
            f"{label} is not a regular file: {resolved}"
        )
    if before.st_size <= 0:
        raise PaperBuildValidationError(f"{label} is empty: {resolved}")
    try:
        data = resolved.read_bytes()
        after = resolved.stat()
    except OSError as error:
        raise PaperBuildValidationError(
            f"{label} could not be read: {resolved}: {error}"
        ) from error
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != after.st_size
    ):
        raise PaperBuildValidationError(
            f"{label} changed while it was being audited: {resolved}"
        )
    return StableArtifact(
        path=resolved,
        data=data,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
    )


def _decode_utf8(artifact: StableArtifact, label: str) -> str:
    try:
        return artifact.data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PaperBuildValidationError(
            f"{label} is not valid UTF-8: {artifact.path}: {error}"
        ) from error


def _load_strict_json(artifact: StableArtifact, label: str) -> dict[str, Any]:
    text = _decode_utf8(artifact, label)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise PaperBuildValidationError(
            f"{label} is not strict JSON: {artifact.path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PaperBuildValidationError(
            f"{label} JSON root is not an object: {artifact.path}"
        )
    return payload


def _is_placeholder(value: str) -> bool:
    normalized = " ".join(value.strip().casefold().split())
    if not normalized:
        return True
    if normalized in {"-", "--", "---", "–", "—", r"\textemdash"}:
        return True
    return any(term in normalized for term in PLACEHOLDER_TERMS)


def _active_tex_lines(text: str) -> list[tuple[int, str]]:
    active: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = _strip_tex_comment(line).strip()
        if stripped:
            active.append((line_number, stripped))
    return active


def _parse_review_state(text: str) -> tuple[bool | None, list[str]]:
    declarations: list[tuple[int, bool]] = []
    for line_number, line in _active_tex_lines(text):
        match = REVIEW_DIRECTIVE.fullmatch(line)
        if match is not None:
            declarations.append(
                (line_number, match.group("state").casefold() == "true")
            )
    if len(declarations) != 1:
        return None, [
            "main.tex must contain exactly one active, line-level "
            r"\internalreviewtrue or \internalreviewfalse directive; "
            f"found {len(declarations)}"
        ]
    return declarations[0][1], []


def _parse_results_macros(
    text: str,
) -> tuple[dict[str, str], list[str], list[str]]:
    macros: dict[str, str] = {}
    unexpected_lines: list[str] = []
    errors: list[str] = []
    allowed = set(RESULT_MACROS) | {RELEASE_SENTINEL}
    for line_number, line in _active_tex_lines(text):
        match = RESULT_MACRO_LINE.fullmatch(line)
        if match is None:
            unexpected_lines.append(f"line {line_number}: {line}")
            continue
        name = match.group("name")
        if name not in allowed:
            errors.append(f"results_auto.tex defines unexpected macro {name}")
            continue
        if name in macros:
            errors.append(f"results_auto.tex defines duplicate macro {name}")
            continue
        macros[name] = match.group("value").strip()
    missing = sorted(set(RESULT_MACROS).difference(macros))
    if missing:
        errors.append(
            "results_auto.tex is missing required macros: " + ",".join(missing)
        )
    if unexpected_lines:
        preview = "; ".join(unexpected_lines[:5])
        suffix = (
            ""
            if len(unexpected_lines) <= 5
            else f"; ... {len(unexpected_lines) - 5} more"
        )
        errors.append(
            "results_auto.tex contains executable content outside its "
            f"allowlisted one-line macros: {preview}{suffix}"
        )
    placeholders = sorted(
        name
        for name in RESULT_MACROS
        if name in macros and _is_placeholder(macros[name])
    )
    return macros, placeholders, errors


def _main_result_contract_errors(text: str) -> list[str]:
    """Require every macro exported by the canonical release contract."""

    active = "\n".join(line for _, line in _active_tex_lines(text))
    errors: list[str] = []
    missing_references = [
        name
        for name in RESULT_MACROS
        if re.search(rf"\\{re.escape(name)}(?![A-Za-z])", active) is None
    ]
    if missing_references:
        errors.append(
            "main.tex does not reference required public-table macros: "
            + ",".join(missing_references)
        )
    return errors


def _sha256_value(value: Any) -> bool:
    return isinstance(value, str) and SHA256_TEXT.fullmatch(value) is not None


def _release_lock_state_errors(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_keys = set(
        getattr(release_contract, "RELEASE_LOCK_KEYS", frozenset())
    )
    if expected_keys and set(lock) != expected_keys:
        errors.append(
            "release-lock key mismatch; missing="
            + ",".join(sorted(expected_keys.difference(lock)))
            + "; unexpected="
            + ",".join(sorted(set(lock).difference(expected_keys)))
        )
    if lock.get("schema_version") != RELEASE_LOCK_SCHEMA:
        errors.append("schema_version mismatch")
    if lock.get("submission_unlocked") is not True:
        errors.append("submission_unlocked is not true")
    if lock.get("release_sentinel_name") != RELEASE_SENTINEL:
        errors.append("release_sentinel_name mismatch")
    if lock.get("release_sentinel_value") != RELEASE_SENTINEL_VALUE:
        errors.append("release_sentinel_value mismatch")
    generated_utc = lock.get("generated_utc")
    try:
        timestamp = datetime.fromisoformat(str(generated_utc).replace("Z", "+00:00"))
    except ValueError:
        errors.append("generated_utc is not an ISO-8601 timestamp")
    else:
        if timestamp.tzinfo is None:
            errors.append("generated_utc has no timezone")
    return errors


def _release_lock_identity_errors(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if lock.get("formal_cache_designation") != FORMAL_CACHE_DESIGNATION:
        errors.append("formal_cache_designation mismatch")
    run_id = lock.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or _is_placeholder(run_id)
    ):
        errors.append("run_id is absent or placeholder")
    if not _sha256_value(lock.get("cache_digest")):
        errors.append("cache_digest is not a lowercase SHA-256 digest")
    for field in (
        "run_json_sha256",
        "results_auto_sha256",
        "macro_value_manifest_sha256",
        "source_gate_sha256",
    ):
        if not _sha256_value(lock.get(field)):
            errors.append(f"{field} is not a lowercase SHA-256 digest")
    artifact_audit = lock.get("artifact_audit")
    if (
        not isinstance(artifact_audit, dict)
        or artifact_audit.get("passed") is not True
    ):
        errors.append("artifact_audit is absent, malformed, or not passed")
    return errors


def _release_lock_provenance_errors(lock: dict[str, Any]) -> list[str]:
    provenance = lock.get("macro_provenance")
    if not isinstance(provenance, dict):
        return ["macro_provenance is not an object"]
    expected = set(PROVENANCE_MACROS)
    actual = set(provenance)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        return [
            "macro_provenance key mismatch; missing="
            + ",".join(missing)
            + "; unexpected="
            + ",".join(unexpected)
        ]
    errors: list[str] = []
    for name in PROVENANCE_MACROS:
        record = provenance[name]
        if not isinstance(record, dict):
            errors.append(f"{name}: provenance record is not an object")
            continue
        if set(record) != {
            "source_artifact",
            "source_sha256",
            "derivation",
        }:
            errors.append(f"{name}: provenance fields are not exact")
            continue
        source = record.get("source_artifact")
        if (
            not isinstance(source, str)
            or not source.strip()
            or Path(source).is_absolute()
            or ".." in Path(source).parts
        ):
            errors.append(
                f"{name}: source_artifact is not a safe run-relative path"
            )
        if not _sha256_value(record.get("source_sha256")):
            errors.append(f"{name}: source_sha256 is invalid")
        derivation = record.get("derivation")
        if (
            not isinstance(derivation, str)
            or len(derivation.strip()) < 12
            or _is_placeholder(derivation)
            or "\n" in derivation
            or "\r" in derivation
        ):
            errors.append(f"{name}: derivation is absent or non-auditable")
    return errors


def _finding(line_number: int, line: str) -> dict[str, Any]:
    return {"line_number": line_number, "text": line.strip()}


def _parse_log(text: str) -> dict[str, Any]:
    findings: dict[str, Any] = {
        "fatal_errors": [],
        "undefined_citations": [],
        "undefined_references": [],
        "overfull_boxes": [],
        "rerun_required": [],
        "output_records": [],
    }
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in FATAL_PATTERNS):
            findings["fatal_errors"].append(_finding(line_number, line))
        if CITATION_UNDEFINED.search(line):
            findings["undefined_citations"].append(
                _finding(line_number, line)
            )
        if REFERENCE_UNDEFINED.search(line):
            findings["undefined_references"].append(
                _finding(line_number, line)
            )
        if OVERFULL_BOX.search(line):
            findings["overfull_boxes"].append(_finding(line_number, line))
        if RERUN_REQUIRED.search(line):
            findings["rerun_required"].append(_finding(line_number, line))

    normalized = re.sub(r"\s+", " ", text)
    for match in OUTPUT_RECORD.finditer(normalized):
        findings["output_records"].append(
            {
                "page_count": int(match.group("pages")),
                "pdf_bytes": int(match.group("bytes").replace(",", "")),
            }
        )
    return findings


def _expand_internalreview_source(
    text: str,
    *,
    internal_review: bool,
) -> tuple[str, str, list[str]]:
    r"""Expand only ``\ifinternalreview`` while preserving other conditionals.

    The second returned string contains text selected from an
    ``\ifinternalreview`` branch.  In release mode this isolates public branch
    text so that internal-only placeholder prose is not mistaken for public
    content.
    """

    active_text = "\n".join(_strip_tex_comment(line) for line in text.splitlines())
    expanded: list[str] = []
    selected_target_text: list[str] = []
    stack: list[dict[str, Any]] = []
    cursor = 0
    declaration_if_start: int | None = None
    errors: list[str] = []

    def selection_enabled() -> bool:
        selected_else = not internal_review
        return all(
            not frame["target"] or frame["in_else"] == selected_else
            for frame in stack
        )

    def in_selected_target() -> bool:
        return any(frame["target"] for frame in stack) and selection_enabled()

    def append_segment(segment: str) -> None:
        if selection_enabled():
            expanded.append(segment)
            if in_selected_target():
                selected_target_text.append(segment)

    for match in TEX_CONDITIONAL_TOKEN.finditer(active_text):
        append_segment(active_text[cursor : match.start()])
        command = match.group("command")
        token = match.group(0)
        if command == "newif":
            append_segment(token)
            declaration_if_start = match.end()
        elif command.startswith("if"):
            is_declaration = (
                declaration_if_start is not None
                and match.start() == declaration_if_start
            )
            declaration_if_start = None
            if is_declaration:
                append_segment(token)
            else:
                target = command.casefold() == "ifinternalreview"
                if not target:
                    append_segment(token)
                stack.append(
                    {
                        "target": target,
                        "in_else": False,
                        "command": command,
                    }
                )
        elif command == "else":
            declaration_if_start = None
            if not stack:
                errors.append(r"unmatched \else while expanding main.tex")
                append_segment(token)
            elif stack[-1]["target"]:
                stack[-1]["in_else"] = True
            else:
                append_segment(token)
        else:
            declaration_if_start = None
            if not stack:
                errors.append(r"unmatched \fi while expanding main.tex")
                append_segment(token)
            else:
                frame = stack[-1]
                if not frame["target"]:
                    append_segment(token)
                stack.pop()
        cursor = match.end()
    append_segment(active_text[cursor:])
    if stack:
        errors.append(
            "unterminated TeX conditional(s) while expanding main.tex: "
            + ",".join(str(frame["command"]) for frame in stack)
        )
    return "".join(expanded), "".join(selected_target_text), errors


def _placeholder_findings(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line) is None:
            continue
        findings.append(
            {
                "line_number": line_number,
                "text": line.strip()[:500],
            }
        )
        if len(findings) >= 50:
            break
    return findings


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _parse_fls(
    text: str,
    *,
    paper_root: Path,
) -> tuple[list[Path], list[Path], Path | None, list[str]]:
    pwd_values: list[Path] = []
    raw_inputs: list[str] = []
    raw_outputs: list[str] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("PWD "):
            value = line[4:].strip()
            if not value:
                errors.append(f"main.fls line {line_number} has an empty PWD")
            else:
                pwd_values.append(Path(value).expanduser().resolve())
        elif line.startswith("INPUT "):
            value = line[6:].strip()
            if not value:
                errors.append(f"main.fls line {line_number} has an empty INPUT")
            else:
                raw_inputs.append(value)
        elif line.startswith("OUTPUT "):
            value = line[7:].strip()
            if not value:
                errors.append(f"main.fls line {line_number} has an empty OUTPUT")
            else:
                raw_outputs.append(value)
    distinct_pwd = {_path_key(path): path for path in pwd_values}
    if len(distinct_pwd) != 1:
        errors.append(
            "main.fls must contain exactly one distinct PWD; "
            f"found {len(distinct_pwd)}"
        )
        base = paper_root
    else:
        base = next(iter(distinct_pwd.values()))
    if base.resolve() != paper_root.resolve():
        errors.append(
            "main.fls PWD is not the audited paper root: "
            f"{base.resolve()} != {paper_root.resolve()}"
        )

    def resolve_record(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()

    inputs = {
        _path_key(path): path
        for path in (resolve_record(value) for value in raw_inputs)
    }
    outputs = {
        _path_key(path): path
        for path in (resolve_record(value) for value in raw_outputs)
    }
    if not inputs:
        errors.append("main.fls contains no INPUT records")
    if not outputs:
        errors.append("main.fls contains no OUTPUT records")
    return list(inputs.values()), list(outputs.values()), base, errors


def _declared_source_dependencies(
    expanded_main: str,
    *,
    paper_root: Path,
) -> tuple[list[Path], list[str]]:
    dependencies: dict[str, Path] = {}
    errors: list[str] = []

    def add_relative(raw_value: str, default_suffix: str | None = None) -> None:
        value = raw_value.strip()
        if not value or "\\" in value:
            errors.append(
                "main.tex dependency path is empty or macro-derived: "
                f"{raw_value!r}"
            )
            return
        candidate = Path(value)
        if default_suffix is not None and not candidate.suffix:
            candidate = candidate.with_suffix(default_suffix)
        if not candidate.is_absolute():
            candidate = paper_root / candidate
        resolved = candidate.expanduser().resolve()
        dependencies[_path_key(resolved)] = resolved

    for match in re.finditer(
        r"\\(?:input|include)\s*(?:\{(?P<braced>[^{}]+)\}|"
        r"(?P<plain>[^\s%]+))",
        expanded_main,
    ):
        add_relative(
            match.group("braced") or match.group("plain"),
            default_suffix=".tex",
        )
    for match in re.finditer(r"\\bibliography\s*\{(?P<names>[^{}]+)\}", expanded_main):
        for value in match.group("names").split(","):
            add_relative(value, default_suffix=".bib")
    for match in re.finditer(
        r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{(?P<name>[^{}]+)\}",
        expanded_main,
    ):
        value = match.group("name").strip()
        candidate = Path(value)
        if candidate.suffix:
            add_relative(value)
            continue
        alternatives = [
            (paper_root / candidate).with_suffix(suffix)
            for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".eps")
        ]
        existing = [path for path in alternatives if path.is_file()]
        if len(existing) == 1:
            add_relative(str(existing[0]))
        elif len(existing) > 1:
            errors.append(
                "main.tex graphic dependency is ambiguous without a suffix: "
                f"{value!r}"
            )
        else:
            add_relative(value, default_suffix=".pdf")
    for match in re.finditer(
        r"\\documentclass(?:\s*\[[^\]]*\])?\s*\{(?P<name>[^{}]+)\}",
        expanded_main,
    ):
        candidate = (paper_root / match.group("name").strip()).with_suffix(".cls")
        if candidate.is_file():
            add_relative(str(candidate))
    return list(dependencies.values()), errors


def _resolve_read_only_tool(
    explicit_path: Path | None,
    *,
    command_name: str,
) -> Path | None:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        return candidate if candidate.is_file() else None
    names = (
        (f"{command_name}.exe", command_name)
        if os.name == "nt"
        else (command_name,)
    )
    for name in names:
        located = shutil.which(name)
        if located:
            candidate = Path(located).resolve()
            if candidate.is_file():
                return candidate
    return None


def _run_read_only_tool(
    executable: Path,
    arguments: list[str],
    *,
    timeout_seconds: float,
) -> tuple[str, str, str | None]:
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "", "", f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[:1000]
        return (
            completed.stdout,
            completed.stderr,
            f"exit {completed.returncode}: {stderr}",
        )
    return completed.stdout, completed.stderr, None


def _inspect_pdf(
    artifact: StableArtifact,
    *,
    pdfinfo_path: Path | None,
    pdftotext_path: Path | None,
    require_text_audit: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    inspection: dict[str, Any] = {
        "parser": None,
        "page_count": None,
        "reported_file_size": None,
        "text_extractor": None,
        "text_placeholder_findings": [],
    }
    errors: list[str] = []
    pdfinfo = _resolve_read_only_tool(
        pdfinfo_path,
        command_name="pdfinfo",
    )
    if pdfinfo is None:
        errors.append(
            "no usable pdfinfo parser was found; PDF structure/page count "
            "cannot be trusted"
        )
    else:
        inspection["parser"] = str(pdfinfo)
        stdout, _, error = _run_read_only_tool(
            pdfinfo,
            [str(artifact.path)],
            timeout_seconds=timeout_seconds,
        )
        if error is not None:
            errors.append(f"pdfinfo failed: {error}")
        else:
            page_matches = list(PDFINFO_PAGES.finditer(stdout))
            size_matches = list(PDFINFO_FILE_SIZE.finditer(stdout))
            if len(page_matches) != 1:
                errors.append(
                    "pdfinfo output does not contain exactly one Pages record"
                )
            else:
                pages = int(page_matches[0].group("pages"))
                if pages <= 0:
                    errors.append("pdfinfo reported a nonpositive page count")
                else:
                    inspection["page_count"] = pages
            if len(size_matches) != 1:
                errors.append(
                    "pdfinfo output does not contain exactly one File size record"
                )
            else:
                reported_size = int(size_matches[0].group("bytes"))
                inspection["reported_file_size"] = reported_size
                if reported_size != artifact.size:
                    errors.append(
                        "pdfinfo file size differs from the audited PDF bytes"
                    )

    if require_text_audit:
        pdftotext = _resolve_read_only_tool(
            pdftotext_path,
            command_name="pdftotext",
        )
        if pdftotext is None:
            errors.append(
                "no usable pdftotext extractor was found; public PDF "
                "placeholder text cannot be audited"
            )
        else:
            inspection["text_extractor"] = str(pdftotext)
            stdout, _, error = _run_read_only_tool(
                pdftotext,
                ["-enc", "UTF-8", str(artifact.path), "-"],
                timeout_seconds=timeout_seconds,
            )
            if error is not None:
                errors.append(f"pdftotext failed: {error}")
            elif not stdout.strip():
                errors.append("pdftotext returned no extractable manuscript text")
            else:
                inspection["text_placeholder_findings"] = _placeholder_findings(
                    stdout,
                    PDF_PLACEHOLDER_LINE,
                )

    try:
        after = _stable_read(artifact.path, "built PDF after parser audit")
    except PaperBuildValidationError as error:
        errors.append(str(error))
    else:
        if (
            after.sha256 != artifact.sha256
            or after.size != artifact.size
            or after.mtime_ns != artifact.mtime_ns
        ):
            errors.append("built PDF changed during external parser inspection")
    return inspection, errors


def _resolve_under_paper(
    paper_root: Path,
    supplied: Path | None,
    default: str,
) -> Path:
    value = Path(default) if supplied is None else supplied
    if value.is_absolute():
        return value.resolve()
    return (paper_root / value).resolve()


def _record_check(
    report: dict[str, Any],
    check_id: str,
    passed: bool,
    *,
    actual: Any,
    required: Any,
    failure: str,
) -> None:
    report["checks"].append(
        {
            "id": check_id,
            "passed": bool(passed),
            "actual": actual,
            "required": required,
        }
    )
    if not passed:
        report["issues"].append(f"{check_id}: {failure}")


def audit_paper_build(
    *,
    paper_root: Path,
    mode: str,
    log_path: Path | None = None,
    pdf_path: Path | None = None,
    fls_path: Path | None = None,
    tex_path: Path | None = None,
    results_path: Path | None = None,
    release_lock_path: Path | None = None,
    pdfinfo_path: Path | None = None,
    pdftotext_path: Path | None = None,
    max_release_pages: int = DEFAULT_MAX_RELEASE_PAGES,
    mtime_tolerance_seconds: float = DEFAULT_MTIME_TOLERANCE_SECONDS,
    max_build_artifact_skew_seconds: float = (
        DEFAULT_BUILD_ARTIFACT_SKEW_SECONDS
    ),
    pdf_tool_timeout_seconds: float = DEFAULT_PDF_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Audit one already-built paper without compiling or writing."""

    normalized_mode = mode.strip().casefold()
    root = paper_root.expanduser().resolve()
    paths = {
        "main_tex": _resolve_under_paper(root, tex_path, "main.tex"),
        "results_auto": _resolve_under_paper(
            root, results_path, "results_auto.tex"
        ),
        "latex_log": _resolve_under_paper(root, log_path, "build/main.log"),
        "pdf": _resolve_under_paper(root, pdf_path, "build/main.pdf"),
        "fls": _resolve_under_paper(root, fls_path, "build/main.fls"),
        "release_lock": _resolve_under_paper(
            root, release_lock_path, "release_lock.json"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "ok": False,
        "mode": normalized_mode,
        "paper_root": str(root),
        "read_only": True,
        "compiled_by_validator": False,
        "page_count": None,
        "log_page_count": None,
        "max_release_pages": (
            max_release_pages if normalized_mode == "release" else None
        ),
        "internal_review": None,
        "placeholder_macros": [],
        "numeric_result_values": {},
        "result_contract_errors": [],
        "release_lock_provenance_errors": [],
        "public_source_placeholders": [],
        "release_sentinel_defined": False,
        "pdf_inspection": {
            "parser": None,
            "page_count": None,
            "reported_file_size": None,
            "text_extractor": None,
            "text_placeholder_findings": [],
        },
        "fls": {
            "pwd": None,
            "project_inputs": [],
            "outputs": [],
        },
        "source_dependencies": [],
        "checks": [],
        "issues": [],
        "artifacts": {},
        "log_findings": {
            "fatal_errors": [],
            "undefined_citations": [],
            "undefined_references": [],
            "overfull_boxes": [],
            "rerun_required": [],
            "output_records": [],
        },
    }

    if normalized_mode not in {"internal", "release"}:
        report["issues"].append(
            "mode: expected 'internal' or 'release'"
        )
        return report
    if max_release_pages <= 0:
        report["issues"].append(
            "max_release_pages: expected a positive integer"
        )
        return report
    if mtime_tolerance_seconds < 0:
        report["issues"].append(
            "mtime_tolerance_seconds: expected a nonnegative value"
        )
        return report
    if max_build_artifact_skew_seconds < 0:
        report["issues"].append(
            "max_build_artifact_skew_seconds: expected a nonnegative value"
        )
        return report
    if pdf_tool_timeout_seconds <= 0:
        report["issues"].append(
            "pdf_tool_timeout_seconds: expected a positive value"
        )
        return report

    expected_results_path = (root / "results_auto.tex").resolve()
    _record_check(
        report,
        "results_path_binding",
        paths["results_auto"] == expected_results_path,
        actual=str(paths["results_auto"]),
        required=str(expected_results_path),
        failure=(
            "the audited result macros are not the paper_root/results_auto.tex "
            "that main.tex is required to include"
        ),
    )

    required_labels = {
        "main_tex": "main TeX source",
        "results_auto": "paper result macro file",
        "latex_log": "LaTeX log",
        "pdf": "built PDF",
        "fls": "LaTeX recorder file",
    }
    if normalized_mode == "release":
        required_labels["release_lock"] = "release lock"

    artifacts: dict[str, StableArtifact] = {}
    for key, label in required_labels.items():
        try:
            artifact = _stable_read(paths[key], label)
        except PaperBuildValidationError as error:
            _record_check(
                report,
                f"artifact_{key}",
                False,
                actual="missing_or_unreadable",
                required="stable nonempty regular file",
                failure=str(error),
            )
        else:
            artifacts[key] = artifact
            report["artifacts"][key] = artifact.metadata()
            _record_check(
                report,
                f"artifact_{key}",
                True,
                actual="stable_nonempty_file",
                required="stable nonempty regular file",
                failure="",
            )

    decoded: dict[str, str] = {}
    for key, label in (
        ("main_tex", "main TeX source"),
        ("results_auto", "paper result macro file"),
        ("latex_log", "LaTeX log"),
        ("fls", "LaTeX recorder file"),
    ):
        artifact = artifacts.get(key)
        if artifact is None:
            continue
        try:
            decoded[key] = _decode_utf8(artifact, label)
        except PaperBuildValidationError as error:
            report["issues"].append(f"utf8_{key}: {error}")

    expanded_main: str | None = None
    declared_dependencies: list[Path] = []
    main_text = decoded.get("main_tex")
    if main_text is not None:
        review_state, review_errors = _parse_review_state(main_text)
        report["internal_review"] = review_state
        required_state = normalized_mode == "internal"
        _record_check(
            report,
            "review_mode",
            review_state is required_state and not review_errors,
            actual=review_state,
            required=required_state,
            failure=(
                "; ".join(review_errors)
                if review_errors
                else (
                    f"{normalized_mode} mode requires "
                    rf"\internalreview"
                    f"{'true' if required_state else 'false'}"
                )
            ),
        )
        active_main = "\n".join(line for _, line in _active_tex_lines(main_text))
        input_present = (
            re.search(
                r"\\input\s*\{\s*results_auto(?:\.tex)?\s*\}",
                active_main,
            )
            is not None
        )
        _record_check(
            report,
            "results_auto_included",
            input_present,
            actual=input_present,
            required=True,
            failure="main.tex does not include results_auto.tex",
        )
        result_contract_errors = _main_result_contract_errors(main_text)
        report["result_contract_errors"] = result_contract_errors
        _record_check(
            report,
            "public_table_macro_wiring",
            not result_contract_errors,
            actual=result_contract_errors,
            required=(
                "every v2 result macro referenced; no retired macros or "
                "literal pending/generated result text"
            ),
            failure="; ".join(result_contract_errors),
        )
        expanded_main, selected_branch_text, expansion_errors = (
            _expand_internalreview_source(
                main_text,
                internal_review=normalized_mode == "internal",
            )
        )
        _record_check(
            report,
            "internalreview_conditional_structure",
            not expansion_errors,
            actual=expansion_errors,
            required=[],
            failure="; ".join(expansion_errors),
        )
        if normalized_mode == "release":
            public_placeholders = _placeholder_findings(
                selected_branch_text,
                PUBLIC_BRANCH_PLACEHOLDER,
            )
            report["public_source_placeholders"] = public_placeholders
            _record_check(
                report,
                "release_public_source_nonplaceholder",
                not public_placeholders,
                actual=public_placeholders,
                required=[],
                failure=(
                    "main.tex public \\ifinternalreview branches still contain "
                    "pending/generated placeholder content"
                ),
            )
        declared_dependencies, dependency_errors = (
            _declared_source_dependencies(
                expanded_main,
                paper_root=root,
            )
        )
        _record_check(
            report,
            "declared_source_dependencies",
            not dependency_errors,
            actual={
                "paths": [str(path) for path in declared_dependencies],
                "errors": dependency_errors,
            },
            required="all literal local dependencies resolve unambiguously",
            failure="; ".join(dependency_errors),
        )

    fls_inputs: list[Path] = []
    fls_outputs: list[Path] = []
    fls_text = decoded.get("fls")
    if fls_text is not None:
        fls_inputs, fls_outputs, fls_pwd, fls_errors = _parse_fls(
            fls_text,
            paper_root=root,
        )
        report["fls"] = {
            "pwd": str(fls_pwd) if fls_pwd is not None else None,
            "project_inputs": [],
            "outputs": [str(path) for path in fls_outputs],
        }
        _record_check(
            report,
            "fls_structure",
            not fls_errors,
            actual=fls_errors,
            required="one paper-root PWD plus nonempty INPUT/OUTPUT records",
            failure="; ".join(fls_errors),
        )
        fls_input_keys = {_path_key(path) for path in fls_inputs}
        fls_output_keys = {_path_key(path) for path in fls_outputs}
        for check_id, selected_path, keys, role in (
            ("fls_main_input", paths["main_tex"], fls_input_keys, "INPUT"),
            (
                "fls_results_input",
                paths["results_auto"],
                fls_input_keys,
                "INPUT",
            ),
            ("fls_log_output", paths["latex_log"], fls_output_keys, "OUTPUT"),
            ("fls_pdf_output", paths["pdf"], fls_output_keys, "OUTPUT"),
        ):
            present = _path_key(selected_path) in keys
            _record_check(
                report,
                check_id,
                present,
                actual=str(selected_path),
                required=f"exact path present as main.fls {role}",
                failure=(
                    f"main.fls does not bind the audited {selected_path.name} "
                    f"as an exact {role}"
                ),
            )
        missing_declared = [
            str(path)
            for path in declared_dependencies
            if path.suffix.casefold() != ".bib"
            and _path_key(path) not in fls_input_keys
        ]
        _record_check(
            report,
            "fls_declared_input_binding",
            not missing_declared,
            actual=missing_declared,
            required=[],
            failure=(
                "main.fls omits source dependencies selected by main.tex: "
                + ",".join(missing_declared)
            ),
        )

    project_root = root.parent.resolve()
    build_root = paths["latex_log"].parent.resolve()
    project_fls_inputs = [
        path
        for path in fls_inputs
        if _is_under(path, project_root) and not _is_under(path, build_root)
    ]
    report["fls"]["project_inputs"] = [
        str(path) for path in project_fls_inputs
    ]
    dependency_paths = {
        _path_key(path): path
        for path in (*project_fls_inputs, *declared_dependencies)
    }
    dependency_artifacts: dict[str, StableArtifact] = {}
    for index, dependency_path in enumerate(
        sorted(dependency_paths.values(), key=lambda item: str(item).casefold()),
        start=1,
    ):
        key = _path_key(dependency_path)
        if key == _path_key(paths["main_tex"]):
            artifact = artifacts.get("main_tex")
        elif key == _path_key(paths["results_auto"]):
            artifact = artifacts.get("results_auto")
        else:
            artifact = None
        if artifact is None:
            try:
                artifact = _stable_read(
                    dependency_path,
                    f"project source dependency {index}",
                )
            except PaperBuildValidationError as error:
                _record_check(
                    report,
                    f"source_dependency_{index:03d}",
                    False,
                    actual=str(dependency_path),
                    required="stable nonempty regular file",
                    failure=str(error),
                )
                continue
            _record_check(
                report,
                f"source_dependency_{index:03d}",
                True,
                actual=str(dependency_path),
                required="stable nonempty regular file",
                failure="",
            )
        dependency_artifacts[key] = artifact
        if key not in {
            _path_key(paths["main_tex"]),
            _path_key(paths["results_auto"]),
        }:
            report["source_dependencies"].append(artifact.metadata())

    macros: dict[str, str] = {}
    results_text = decoded.get("results_auto")
    if results_text is not None:
        macros, placeholders, macro_errors = _parse_results_macros(results_text)
        report["placeholder_macros"] = placeholders
        sentinel_value = macros.get(RELEASE_SENTINEL)
        report["release_sentinel_defined"] = sentinel_value is not None
        _record_check(
            report,
            "results_macro_structure",
            not macro_errors,
            actual=sorted(macros),
            required=list(RESULT_MACROS),
            failure="; ".join(macro_errors),
        )
        if normalized_mode == "release":
            _record_check(
                report,
                "release_results_nonplaceholder",
                not placeholders,
                actual=placeholders,
                required=[],
                failure=(
                    "release results contain placeholder macros: "
                    + ",".join(placeholders)
                ),
            )
            sentinel_valid = (
                sentinel_value is not None
                and sentinel_value.strip() == RELEASE_SENTINEL_VALUE
            )
            _record_check(
                report,
                "release_sentinel",
                sentinel_valid,
                actual=sentinel_value,
                required=(
                    rf"\newcommand{{\{RELEASE_SENTINEL}}}"
                    rf"{{{RELEASE_SENTINEL_VALUE}}}"
                ),
                failure=(
                    "release results do not define the main.tex evidence-lock "
                    f"sentinel {RELEASE_SENTINEL}=true"
                ),
            )
    log_text = decoded.get("latex_log")
    if log_text is not None:
        findings = _parse_log(log_text)
        report["log_findings"] = findings
        for key, check_id in (
            ("fatal_errors", "log_no_fatal_errors"),
            ("undefined_citations", "log_no_undefined_citations"),
            ("undefined_references", "log_no_undefined_references"),
            ("overfull_boxes", "log_no_overfull_boxes"),
            ("rerun_required", "log_no_rerun_required"),
        ):
            _record_check(
                report,
                check_id,
                not findings[key],
                actual=len(findings[key]),
                required=0,
                failure=f"LaTeX log contains {len(findings[key])} {key}",
            )
        output_records = findings["output_records"]
        output_record_valid = (
            len(output_records) == 1
            and output_records[0]["page_count"] > 0
            and output_records[0]["pdf_bytes"] > 0
        )
        _record_check(
            report,
            "log_output_record",
            output_record_valid,
            actual=output_records,
            required="exactly one positive Output written on ... record",
            failure=(
                "LaTeX log is missing, ambiguous, or has an invalid final "
                "PDF output record"
            ),
        )
        if output_record_valid:
            output = output_records[0]
            report["log_page_count"] = output["page_count"]
            pdf_artifact = artifacts.get("pdf")
            if pdf_artifact is not None:
                _record_check(
                    report,
                    "log_pdf_byte_match",
                    output["pdf_bytes"] == pdf_artifact.size,
                    actual={
                        "log_pdf_bytes": output["pdf_bytes"],
                        "actual_pdf_bytes": pdf_artifact.size,
                    },
                    required="exact equality",
                    failure=(
                        "LaTeX log PDF byte count does not match the audited "
                        "PDF"
                    ),
                )
    pdf_artifact = artifacts.get("pdf")
    if pdf_artifact is not None:
        signature_ok = pdf_artifact.data.startswith(b"%PDF-")
        _record_check(
            report,
            "pdf_signature",
            signature_ok,
            actual=pdf_artifact.data[:8].decode("ascii", errors="replace"),
            required="%PDF-",
            failure="built PDF does not start with a PDF signature",
        )
        pdf_inspection, pdf_errors = _inspect_pdf(
            pdf_artifact,
            pdfinfo_path=pdfinfo_path,
            pdftotext_path=pdftotext_path,
            require_text_audit=normalized_mode == "release",
            timeout_seconds=pdf_tool_timeout_seconds,
        )
        report["pdf_inspection"] = pdf_inspection
        report["page_count"] = pdf_inspection["page_count"]
        _record_check(
            report,
            "pdf_parser",
            not pdf_errors,
            actual={
                "parser": pdf_inspection["parser"],
                "reported_file_size": pdf_inspection["reported_file_size"],
                "errors": pdf_errors,
            },
            required="successful structural PDF parse and exact file-size audit",
            failure="; ".join(pdf_errors),
        )
        logged_pages = report["log_page_count"]
        parsed_pages = pdf_inspection["page_count"]
        page_counts_match = (
            isinstance(logged_pages, int)
            and isinstance(parsed_pages, int)
            and logged_pages == parsed_pages
        )
        _record_check(
            report,
            "log_pdf_page_match",
            page_counts_match,
            actual={
                "log_pages": logged_pages,
                "parsed_pdf_pages": parsed_pages,
            },
            required="exact equality",
            failure=(
                "LaTeX log page count does not match the independently parsed "
                "PDF page count"
            ),
        )
        if normalized_mode == "release":
            release_page_ok = (
                isinstance(parsed_pages, int)
                and parsed_pages <= max_release_pages
            )
            _record_check(
                report,
                "release_page_limit",
                release_page_ok,
                actual=parsed_pages,
                required=f"<= {max_release_pages}",
                failure=(
                    f"release PDF has {parsed_pages!r} independently parsed "
                    f"pages, above or unverifiable against the "
                    f"{max_release_pages}-page limit"
                ),
            )
            text_placeholders = pdf_inspection[
                "text_placeholder_findings"
            ]
            _record_check(
                report,
                "release_pdf_text_nonplaceholder",
                not text_placeholders and not pdf_errors,
                actual=text_placeholders,
                required=[],
                failure=(
                    "public PDF text contains standalone pending/generated "
                    "placeholder cells or could not be extracted reliably"
                ),
            )

    if normalized_mode == "release" and "release_lock" in artifacts:
        try:
            lock = _load_strict_json(
                artifacts["release_lock"],
                "release lock",
            )
        except PaperBuildValidationError as error:
            report["issues"].append(f"release_lock_json: {error}")
        else:
            results_artifact = artifacts.get("results_auto")
            contract_error: str | None = None
            try:
                release_contract.validate_release_lock_structure(
                    lock,
                    results_auto_sha256=(
                        results_artifact.sha256
                        if results_artifact is not None
                        else None
                    ),
                )
            except release_contract.ReleaseValidationError as error:
                contract_error = str(error)
            _record_check(
                report,
                "release_lock_state",
                contract_error is None,
                actual={
                    "schema_version": lock.get("schema_version"),
                    "submission_unlocked": lock.get("submission_unlocked"),
                    "release_sentinel_name": lock.get(
                        "release_sentinel_name"
                    ),
                    "release_sentinel_value": lock.get(
                        "release_sentinel_value"
                    ),
                    "keys": sorted(lock),
                    "contract_error": contract_error,
                },
                required={
                    "schema_version": RELEASE_LOCK_SCHEMA,
                    "submission_unlocked": True,
                    "release_sentinel_name": RELEASE_SENTINEL,
                    "release_sentinel_value": RELEASE_SENTINEL_VALUE,
                    "keys": sorted(release_contract.RELEASE_LOCK_KEYS),
                    "full_portable_v2_contract": True,
                },
                failure=(
                    "release lock fails the canonical portable v2 contract: "
                    f"{contract_error}"
                ),
            )
            if results_artifact is not None:
                digest_match = (
                    lock.get("results_auto_sha256")
                    == results_artifact.sha256
                )
                _record_check(
                    report,
                    "release_lock_results_digest",
                    digest_match,
                    actual=lock.get("results_auto_sha256"),
                    required=results_artifact.sha256,
                    failure=(
                        "release lock is not bound to the audited "
                        "results_auto.tex bytes"
                    ),
                )

    log_artifact = artifacts.get("latex_log")
    fls_artifact = artifacts.get("fls")
    if (
        log_artifact is not None
        and pdf_artifact is not None
        and fls_artifact is not None
    ):
        build_times = (
            log_artifact.mtime_ns,
            pdf_artifact.mtime_ns,
            fls_artifact.mtime_ns,
        )
        skew_seconds = (max(build_times) - min(build_times)) / 1_000_000_000
        _record_check(
            report,
            "build_artifact_timestamp_skew",
            skew_seconds <= max_build_artifact_skew_seconds,
            actual=skew_seconds,
            required=f"<= {max_build_artifact_skew_seconds} seconds",
            failure=(
                "LaTeX log, PDF, and recorder timestamps are too far apart to "
                "be treated as one final build pass"
            ),
        )
        build_floor_ns = min(build_times)
        tolerance_ns = int(mtime_tolerance_seconds * 1_000_000_000)
        source_keys = ["main_tex", "results_auto"]
        if normalized_mode == "release":
            source_keys.append("release_lock")
        for key in source_keys:
            source = artifacts.get(key)
            if source is None:
                continue
            delta_seconds = (
                source.mtime_ns - build_floor_ns
            ) / 1_000_000_000
            _record_check(
                report,
                f"freshness_{key}",
                source.mtime_ns <= build_floor_ns + tolerance_ns,
                actual={
                    "source_minus_earliest_build_output_seconds": (
                        delta_seconds
                    ),
                    "tolerance_seconds": mtime_tolerance_seconds,
                },
                required="source not newer than final log/PDF",
                failure=(
                    f"{key} is newer than the final build artifacts; rebuild "
                    "the paper before validating it"
                ),
            )
        primary_dependency_keys = {
            _path_key(paths["main_tex"]),
            _path_key(paths["results_auto"]),
        }
        for index, (key, source) in enumerate(
            sorted(
                dependency_artifacts.items(),
                key=lambda item: str(item[1].path).casefold(),
            ),
            start=1,
        ):
            if key in primary_dependency_keys:
                continue
            delta_seconds = (
                source.mtime_ns - build_floor_ns
            ) / 1_000_000_000
            _record_check(
                report,
                f"freshness_dependency_{index:03d}",
                source.mtime_ns <= build_floor_ns + tolerance_ns,
                actual={
                    "path": str(source.path),
                    "source_minus_earliest_build_output_seconds": (
                        delta_seconds
                    ),
                    "tolerance_seconds": mtime_tolerance_seconds,
                },
                required="dependency not newer than final log/PDF/FLS",
                failure=(
                    f"source dependency {source.path} is newer than the final "
                    "build artifacts; rebuild the paper before validating it"
                ),
            )

    report["ok"] = not report["issues"]
    report["release_eligible"] = (
        report["ok"] and normalized_mode == "release"
    )
    report["internal_build_validated"] = (
        report["ok"] and normalized_mode == "internal"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=PROJECT_ROOT / "paper",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("internal", "release"),
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--fls", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--release-lock", type=Path)
    parser.add_argument("--pdfinfo", type=Path)
    parser.add_argument("--pdftotext", type=Path)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_RELEASE_PAGES,
    )
    parser.add_argument(
        "--mtime-tolerance-seconds",
        type=float,
        default=DEFAULT_MTIME_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--max-build-artifact-skew-seconds",
        type=float,
        default=DEFAULT_BUILD_ARTIFACT_SKEW_SECONDS,
    )
    parser.add_argument(
        "--pdf-tool-timeout-seconds",
        type=float,
        default=DEFAULT_PDF_TOOL_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        report = audit_paper_build(
            paper_root=arguments.paper_root,
            mode=arguments.mode,
            log_path=arguments.log,
            pdf_path=arguments.pdf,
            fls_path=arguments.fls,
            tex_path=arguments.tex,
            results_path=arguments.results,
            release_lock_path=arguments.release_lock,
            pdfinfo_path=arguments.pdfinfo,
            pdftotext_path=arguments.pdftotext,
            max_release_pages=arguments.max_pages,
            mtime_tolerance_seconds=arguments.mtime_tolerance_seconds,
            max_build_artifact_skew_seconds=(
                arguments.max_build_artifact_skew_seconds
            ),
            pdf_tool_timeout_seconds=arguments.pdf_tool_timeout_seconds,
        )
    except Exception as error:  # pragma: no cover - final fail-closed boundary
        report = {
            "schema_version": REPORT_SCHEMA,
            "ok": False,
            "mode": arguments.mode,
            "issues": [
                f"unexpected_validator_error: {type(error).__name__}: {error}"
            ],
            "read_only": True,
            "compiled_by_validator": False,
        }
        exit_code = 3
    else:
        exit_code = 0 if report["ok"] else 2
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
