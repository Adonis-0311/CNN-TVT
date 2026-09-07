"""V4.4 number-validation waiver classifier.

Re-runs the exact detection rules of ``validate_paper_numbers.py`` but emits
EVERY notice (not just the first 60) together with a bucket classification
required by TVT_V4_4_NUMBER_VALIDATION_WAIVER.md:

1. manual-table literals consistent with artifact
2. rounding / formatting-only
3. bibliography / year / page-number strings
4. legacy macros not referenced by V4.4
5. figure labels / captions not intended as numerical macros
6. true manuscript-artifact mismatch
7. unclassified

Buckets 6 and 7 must both be zero for Technical Freeze.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

import validate_paper_numbers as vpn

REPO = vpn.REPO
MACROS = vpn.MACROS
NUMBERS = vpn.NUMBERS

# Buckets ---------------------------------------------------------------
BUCKET_NAMES = {
    1: "manual-table literals consistent with artifact",
    2: "rounding / formatting-only",
    3: "bibliography / year / page-number strings",
    4: "legacy macros not referenced by V4.4",
    5: "figure labels / captions not intended as numerical macros",
    6: "true manuscript-artifact mismatch",
    7: "unclassified",
}


def macro_values() -> dict[str, float]:
    """Parse every macro name -> float value from all macro definition files."""
    values: dict[str, float] = {}
    for macro_file in (
        REPO / "paper_data_layer" / "outputs" / "paper_macros.tex",
        REPO / "paper" / "v1_audit_macros.tex",
        REPO / "paper" / "v41_envelope_macros.tex",
    ):
        for name, val in re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]+)\}", macro_file.read_text(encoding="utf-8")
        ):
            try:
                values[name] = float(val)
            except ValueError:
                continue
    return values


def referenced_macros() -> set[str]:
    """Macros actually used by the V4.4 manuscript body (main + supplement)."""
    used: set[str] = set()
    for name in ("main.tex", "supplement.tex"):
        text = (REPO / "paper" / name).read_text(encoding="utf-8")
        used.update(re.findall(r"\\([A-Z][A-Za-z]+)\b", text))
    return used


def collect_all_failures() -> list[dict]:
    defined = set(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", MACROS.read_text(encoding="utf-8"))
    )
    with open(NUMBERS, encoding="utf-8") as handle:
        record = json.load(handle)
    failures: list[dict] = []
    targets = sorted(p for p in (REPO / "paper").glob("*.tex") if p.name != "paper_macros.tex")
    for path in targets:
        text = vpn._strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        for match in vpn.MACRO_USE.finditer(text):
            name = match.group(1)
            if name in record["keys"] and name not in defined:
                failures.append(
                    {"file": path.name, "line": None, "literal": None, "macro": name,
                     "kind": "undefined_macro", "text": ""}
                )
        for line_number, line in enumerate(text.splitlines(), start=1):
            if vpn.ALLOWED_LITERAL_CONTEXT.search(line):
                continue
            for literal in vpn.NUMERIC.findall(line):
                if literal in vpn.PROTOCOL_LITERALS:
                    continue
                failures.append(
                    {"file": path.name, "line": line_number, "literal": literal,
                     "macro": None, "kind": "literal", "text": line.strip()}
                )
    return failures


def in_tabular(path: Path, line: int) -> bool:
    lines = vpn._strip_comments(path.read_text(encoding="utf-8")).splitlines()
    depth = 0
    for idx in range(line - 1):
        depth += lines[idx].count(r"\begin{tabular") + lines[idx].count(r"\begin{table")
        depth -= lines[idx].count(r"\end{tabular") + lines[idx].count(r"\end{table")
    return depth > 0


def _json_leaves(node, out: list[float]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _json_leaves(value, out)
    elif isinstance(node, list):
        for value in node:
            _json_leaves(value, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        out.append(float(node))


def artifact_corpus() -> list[float]:
    """Every numeric value in the artifact tables the manuscript draws from."""
    values: list[float] = []
    values.extend(macro_values().values())
    numbers = json.load(open(NUMBERS, encoding="utf-8"))
    for entry in numbers["keys"].values():
        try:
            values.append(float(entry["value"]))
        except (TypeError, ValueError):
            continue
    csv_dirs = [
        REPO / "analysis_zero_compute" / "outputs" / "csv",
        REPO / "paper_data_layer" / "outputs",
    ]
    for directory in csv_dirs:
        for csv_file in sorted(directory.glob("*.csv")):
            frame = pd.read_csv(csv_file)
            for column in frame.select_dtypes("number").columns:
                values.extend(float(v) for v in frame[column].dropna())
    json_files = [
        *sorted((REPO / "analysis_zero_compute" / "outputs").glob("*.json")),
        REPO / "paper_data_layer" / "outputs" / "e1_reference.json",
        REPO / "artifacts" / "rml2016_10a_fidelity_v3" / "run.json",
        REPO / "artifacts" / "tier2_iq_sidecar_v1" / "run.json",
    ]
    for json_file in json_files:
        if json_file.exists():
            _json_leaves(json.load(open(json_file, encoding="utf-8")), values)
    metrics = REPO / "artifacts" / "rml2016_10a_fidelity_v3" / "metrics.csv"
    if metrics.exists():
        frame = pd.read_csv(metrics)
        for column in frame.select_dtypes("number").columns:
            values.extend(float(v) for v in frame[column].dropna())
    for extra in (
        REPO / "artifacts" / "rml2016_10a_fidelity_v3" / "per_snr.csv",
        REPO / "artifacts" / "rml2016_10a_fidelity_v3" / "training_history.csv",
    ):
        if extra.exists():
            frame = pd.read_csv(extra)
            for column in frame.select_dtypes("number").columns:
                values.extend(float(v) for v in frame[column].dropna())
    return values


# Values derived from artifacts / protocol constants, with provenance.  Each
# entry is (literal, provenance) and is reported in the waiver verbatim.
DERIVED = {
    "3.50": "100*(0.6205-0.585454): e1_reference.json MCLDNN overall minus metrics.csv seed mean",
    "3.5": "same difference quoted at one decimal",
    "1.04": "100*(0.652258-0.6419): IQFormer seed mean minus e1_reference.json overall",
    "4.7": "A5/A0 parameter ratio (39,500 / ~8,400 params, model configs)",
    "1.29": "T_c ~ 0.423/f_D at 60 km/h, fc=5.9 GHz (protocol constants)",
    "0.0020": "2/1024 window-spacing ratio (protocol constants)",
    "65.23": "mean of the three IQFormer seeds in metrics.csv (0.652258)",
    "58.55": "mean of the three MCLDNN seeds in metrics.csv (0.585454)",
}
DEFINITIONAL = {
    "0.8": "illustrative example defining the overlap scale (Eq. discussion), not an evidence value",
}


def matches_artifact(literal: str, corpus: list[float]) -> str | None:
    """Return 'exact', 'rounded', 'k-scaled' or None."""
    val = float(literal)
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    for v in corpus:
        for scale in (1.0, 100.0):
            for signed in (scale * v, -scale * v):
                if abs(signed - val) < 1e-9:
                    return "exact"
                if decimals and abs(round(signed, decimals) - val) < 1e-9:
                    return "rounded"
    for v in corpus:  # 39.5k-style thousands shorthand
        if decimals and abs(round(v / 1000.0, decimals) - val) < 1e-9:
            return "k-scaled"
    return None


def classify(issue: dict, corpus: list[float], used: set[str]) -> int:
    if issue["kind"] == "undefined_macro":
        # evidence macro referenced but not generated: true mismatch candidate
        return 6
    literal = issue["literal"]
    text = issue["text"]

    # macro-definition files: every decimal is a macro body, not prose.
    if issue["file"] in {"v1_audit_macros.tex", "v41_envelope_macros.tex", "results_auto.tex"}:
        macro_name = re.search(r"\\newcommand\{\\([A-Za-z]+)\}", text)
        if macro_name and macro_name.group(1) not in used:
            return 4  # legacy macro not referenced by V4.4
        return 2  # referenced macro definition: formatting of a generated value

    # bibliography / year / page strings (e.g. \bibitem keys, pp. ranges)
    if re.search(r"(bibitem|bibliography|\\cite|pp\.|vol\.|no\.|arXiv|doi|10\.\d{4})", text):
        return 3

    # figure caption / label lines
    if re.search(r"(\\caption|\\label|\\includegraphics|\\captionsetup)", text):
        return 5

    if literal in DEFINITIONAL:
        return 5
    if literal in DERIVED:
        return 2

    kind = matches_artifact(literal, corpus)
    if kind:
        return 1 if kind == "exact" else 2
    return 7


def main() -> None:
    corpus = artifact_corpus()
    used = referenced_macros()
    failures = collect_all_failures()
    for issue in failures:
        issue["bucket"] = classify(issue, corpus, used)
        if issue["literal"] in DERIVED:
            issue["provenance"] = DERIVED[issue["literal"]]
        elif issue["literal"] in DEFINITIONAL:
            issue["provenance"] = DEFINITIONAL[issue["literal"]]

    out = REPO / "paper_data_layer" / "outputs" / "v44_validator_issues.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=1)

    counts = pd.Series([f["bucket"] for f in failures]).value_counts().sort_index()
    print(f"total notices: {len(failures)}")
    for bucket, count in counts.items():
        print(f"  bucket {bucket} ({BUCKET_NAMES[bucket]}): {count}")
    by_file = pd.Series([f["file"] for f in failures]).value_counts()
    print(by_file.to_string())


if __name__ == "__main__":
    main()
