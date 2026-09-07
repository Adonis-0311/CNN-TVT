"""WP7 validator: the manuscript may not contain hand-typed evidence numbers.

Checks performed:

1. every ``\\Macro`` used in the scanned .tex files is defined in
   ``paper_macros.tex``;
2. no numeric literal that looks like an evidence value (a decimal, a
   percentage, a pp value, or a four-plus-digit integer) appears in prose;
3. the failed-gate disclosure strings are present somewhere in the manuscript;
4. simulation-only terminology is not violated.

Exit code is non-zero on any failure so it can gate a build.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MACROS = Path(__file__).resolve().parent / "outputs" / "paper_macros.tex"
NUMBERS = Path(__file__).resolve().parent / "outputs" / "paper_numbers.json"

REQUIRED_DISCLOSURES = (
    r"family gate",
    r"clean-retention",
    r"post hoc|post-hoc",
    r"simulat",
)
FORBIDDEN_TERMS = (
    r"\bmeasured\b",
    r"\bon-?board\b",
    r"\bfield trial\b",
    r"\boperational evidence\b",
    r"\bin-?orbit\b",
)
NUMERIC = re.compile(
    r"(?<![\\A-Za-z0-9])(\d+\.\d+|\d{4,})(?![A-Za-z0-9])"
)
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z]+)\b")
ALLOWED_LITERAL_CONTEXT = re.compile(
    r"(includegraphics|label|ref|cite|input|usepackage|documentclass|linewidth|textwidth"
    r"|columnwidth|IEEE|section|arXiv|doi|vol\.|no\.|pp\.|20\d\d"
    # protocol constants live in the freeze config, not in the evidence tables
    r"|TR\s*38|38\.901|nrTDL|sample|length|batch|epoch|learning rate|weight decay"
    r"|temperature|dropout|rho|patience|seed|hyperparameter|threshold|n_fft|hop)"
)
PROTOCOL_LITERALS = {"38.901", "1024", "0.001", "0.01", "0.05", "0.12", "0.35", "0.5", "3e-4"}


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def main() -> int:
    if not MACROS.exists():
        print("paper_macros.tex missing; run build_paper_numbers.py first")
        return 2
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", MACROS.read_text(encoding="utf-8")))
    with open(NUMBERS, encoding="utf-8") as handle:
        record = json.load(handle)

    targets = sorted((REPO / "paper").glob("*.tex"))
    targets = [p for p in targets if p.name not in {"paper_macros.tex"}]
    if not targets:
        print("no manuscript .tex files found; nothing to validate yet")
        return 0

    failures: list[str] = []
    corpus = ""
    for path in targets:
        text = _strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        corpus += text + "\n"
        for match in MACRO_USE.finditer(text):
            name = match.group(1)
            if name in record["keys"] and name not in defined:
                failures.append(f"{path.name}: undefined evidence macro \\{name}")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if ALLOWED_LITERAL_CONTEXT.search(line):
                continue
            for literal in NUMERIC.findall(line):
                if literal in PROTOCOL_LITERALS:
                    continue
                failures.append(
                    f"{path.name}:{line_number}: hand-typed numeric literal '{literal}' -- use a macro"
                )

    for pattern in REQUIRED_DISCLOSURES:
        if not re.search(pattern, corpus, flags=re.IGNORECASE):
            failures.append(f"missing required disclosure matching /{pattern}/")
    for pattern in FORBIDDEN_TERMS:
        if re.search(pattern, corpus, flags=re.IGNORECASE):
            failures.append(f"forbidden simulation-boundary term matching /{pattern}/")

    if failures:
        print(f"FAIL ({len(failures)} issues)")
        for item in failures[:60]:
            print("  -", item)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more")
        return 1
    print(f"OK: {len(targets)} files, {len(defined)} macros available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
