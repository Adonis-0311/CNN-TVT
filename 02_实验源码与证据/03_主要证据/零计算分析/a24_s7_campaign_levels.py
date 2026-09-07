"""A24 -- S7 campaign-level sanity: hard-interference macro-F1 of the
retrained zeroed-I/Q sidecar against A5 and the full sidecar.

Zero-compute read of existing prediction NPZ files (sealed campaign
composite, tier2 sidecar run, S7 retrain run). Confirms that the zeroed
variant trained normally on the campaign (it lands at the A5 level while
the full sidecar sits above it), so the severe-held null result is not a
training failure.

Usage:
    python a24_s7_campaign_levels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis_zero_compute"))

import zc_core as z  # noqa: E402

OUT = REPO / "analysis_zero_compute" / "outputs"
CSV = OUT / "csv"
SEEDS = (17, 29, 43, 71, 101)
SPLIT = "hard_interference"

SOURCES = {
    "iqzero": REPO / "artifacts" / "tier2_iq_sidecar_iqzero_v1" / "models"
    / "tier2_h2_f2_iq_sidecar_iqzero_seed{s}" / f"predictions_{SPLIT}.npz",
    "sidecar_full": REPO / "artifacts" / "tier2_iq_sidecar_v1" / "models"
    / "tier2_h2_f2_iq_sidecar_seed{s}" / f"predictions_{SPLIT}.npz",
    "A5": REPO / "artifacts" / "tvt_v4r_headline_composite" / "models"
    / "a5_vimd_full_seed{s}" / f"predictions_{SPLIT}.npz",
}


def main() -> None:
    rows = []
    levels: dict[str, list[float]] = {}
    for tag, template in SOURCES.items():
        f1s = []
        for seed in SEEDS:
            data = np.load(str(template).format(s=seed))
            f1s.append(z.macro_f1(data["labels"], np.argmax(data["probabilities"], -1)))
        levels[tag] = f1s
        rows.append(
            {
                "model": tag,
                "split": SPLIT,
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_seed_std": float(np.std(f1s, ddof=1)),
                "per_seed": str([round(v, 4) for v in f1s]),
            }
        )
    contrasts = {
        "iqzero_vs_A5": float(np.mean(levels["iqzero"]) - np.mean(levels["A5"])),
        "sidecar_full_vs_A5": float(np.mean(levels["sidecar_full"]) - np.mean(levels["A5"])),
        "iqzero_vs_sidecar_full": float(np.mean(levels["iqzero"]) - np.mean(levels["sidecar_full"])),
    }
    summary = {
        "schema": "vimd_amc.a24_s7_campaign_levels.v1",
        "evidence_class": "exploratory_retrained_ablation",
        "split": SPLIT,
        "seeds": list(SEEDS),
        "levels_pct": {tag: float(100 * np.mean(v)) for tag, v in levels.items()},
        "contrasts_pp": {k: float(100 * v) for k, v in contrasts.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    with (CSV / "a24_s7_campaign_levels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = _csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "a24_s7_campaign_levels.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
