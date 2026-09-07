"""Collect Tier-2 clean-probe runs (v1 and v2) into one auditable table."""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zc_core as z  # noqa: E402

ROOTS = (
    (z.REPO / "artifacts" / "tier2_clean_probe_v1", "v1"),
    (z.REPO / "artifacts" / "tier2_clean_probe_v2", "v2"),
)


def run() -> None:
    rows: list[dict] = []
    for root, version in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            row = {
                "probe_version": version,
                "file": path.name,
                "model": record.get("model"),
                "algorithm_seed": record.get("algorithm_seed"),
                "design": record.get("design", "counterfactual_x_minus_jammer"),
                "frozen_head_macro_f1": record.get("frozen_head_macro_f1"),
                "conclusion": record.get("conclusion"),
            }
            if version == "v1":
                high = record.get("probe_f1_qpsk_16qam_64qam", [])
                row.update(
                    {
                        "probe_macro_f1_linear": record.get("probe_macro_f1"),
                        "probe_high_order_linear": sum(high) / len(high) if high else None,
                        "probe_macro_f1_mlp": None,
                        "probe_high_order_mlp": None,
                        "validity": "confounded: jammer removed on stored post-AGC scale without re-normalisation",
                    }
                )
            else:
                probes = record.get("probes", {})
                row.update(
                    {
                        "probe_macro_f1_linear": probes.get("linear", {}).get("macro_f1"),
                        "probe_high_order_linear": probes.get("linear", {}).get("f1_high_order_mean"),
                        "probe_macro_f1_mlp": probes.get("mlp", {}).get("macro_f1"),
                        "probe_high_order_mlp": probes.get("mlp", {}).get("f1_high_order_mean"),
                        "validity": "valid",
                    }
                )
            rows.append(row)
    if not rows:
        raise SystemExit("no probe runs found")
    z.write_csv("tier2_clean_probe_runs.csv", rows)

    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
