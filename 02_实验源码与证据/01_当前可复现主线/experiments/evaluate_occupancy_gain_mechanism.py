"""Evaluate the frozen jammer-occupancy versus A5-minus-A0 gain hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.teacher_audit import occupancy_gain_mechanism_test  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared one-sided exact Spearman mechanism test. "
            "Input JSON must contain family_rows with family, occupancy, and "
            "either gain_pp or a5_macro_f1/a0_macro_f1."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    rows = payload.get("family_rows")
    if not isinstance(rows, list):
        raise ValueError("input JSON must contain a family_rows list")
    result = occupancy_gain_mechanism_test(rows)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
