"""A16: clean-sentinel fitting-cell exclusion sensitivity (V4.4).

The historical 32-cell envelope fit retains exactly one aggregate that is
dominated by jammer-free sentinel windows: the ``id_test`` / SIR = 0 dB /
lowest-overlap-quartile cell (o ~ 1e-11, 1053 windows, SIR taken from the
cache sentinel).  The primary held envelope already excludes jammer-free
clean retention because a jammer-free window has no physical finite SIR; a
reviewer may therefore ask why the fitting domain keeps a sentinel-dominated
aggregate.  This sensitivity answers that question directly:

* drop exactly that one fitting aggregate (32 -> 31 fitting cells);
* refit the two-variable overlap--SIR envelope on the remaining 31 cells;
* re-score the unchanged 80 finite-SIR interfered held cells;
* change no model prediction and retrain nothing.

Both the 32-cell primary fit and the 31-cell sensitivity are recomputed here
from the same frozen cell table so the comparison is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis_zero_compute" / "outputs" / "csv"
FIT_SPLITS = ("id_test", "hard_interference")
HELD_SPLITS = ("unseen_jammer", "unseen_speed", "heldout_channel", "combined_ood")
REFERENCES = {
    "gain_vs_IQFormer": "IQFormer",
    "gain_vs_MCLDNN": "MCLDNN",
}


def sentinel_mask(cells: pd.DataFrame) -> np.ndarray:
    """The id_test / SIR=0 / lowest-overlap-quartile clean-sentinel aggregate."""
    sel = (cells.split == "id_test") & (cells.sir_db == 0.0)
    lowest = sel & (cells.occupancy_mean <= cells.loc[sel, "occupancy_mean"].min())
    return lowest.to_numpy()


def fit_coefficients(cells: pd.DataFrame, fit_idx: np.ndarray, response: str) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(cells)), cells.occupancy_mean.to_numpy(), cells.sir_db.to_numpy()]
    )
    coefficient, *_ = np.linalg.lstsq(
        design[fit_idx], cells[response].to_numpy()[fit_idx], rcond=None
    )
    return coefficient


def score_held(actual: np.ndarray, predicted: np.ndarray, constant: float) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    const_rmse = float(np.sqrt(np.mean((actual - constant) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((actual - actual.mean()) ** 2)))
    return {
        "held_rmse_pp": 100 * rmse,
        "fit_domain_constant_rmse_pp": 100 * const_rmse,
        "r_squared_skill": 1.0 - rmse**2 / const_rmse**2,
        "oracle_constant_rmse_pp": 100 * oracle_rmse,
    }


def break_even(coefficient: np.ndarray, sir_db: float) -> float:
    intercept, overlap_slope, sir_slope = coefficient
    return float(-(intercept + sir_slope * sir_db) / overlap_slope)


def run() -> None:
    cells = pd.read_csv(CSV / "a14_envelope_cells.csv")
    fit_all = cells.split.isin(FIT_SPLITS).to_numpy()
    held = cells.split.isin(HELD_SPLITS).to_numpy()
    sentinel = sentinel_mask(cells)
    if int(fit_all.sum()) != 32 or int(sentinel.sum()) != 1:
        raise RuntimeError("expected 32 fitting cells and exactly one sentinel cell")
    fit_31 = fit_all & ~sentinel
    sent = cells.loc[sentinel].iloc[0]

    rows: list[dict] = []
    report: dict = {
        "schema": "vimd_amc.a16_clean_sentinel_sensitivity.v1",
        "evidence_class": "exploratory_zero_compute_reanalysis_of_sealed_predictions",
        "source_cells": "analysis_zero_compute/outputs/csv/a14_envelope_cells.csv",
        "excluded_cell": {
            "split": sent["split"],
            "sir_db": float(sent["sir_db"]),
            "occupancy_mean": float(sent["occupancy_mean"]),
            "row_count": int(sent["row_count"]),
            "description": "id_test / SIR=0 / lowest-overlap quartile aggregate "
            "dominated by jammer-free sentinel windows",
        },
        "fit_cells_primary": 32,
        "fit_cells_sensitivity": 31,
        "held_cells": int(held.sum()),
        "references": {},
    }
    for response, short in REFERENCES.items():
        actual = cells[response].to_numpy()
        for label, fit_idx in (("32", fit_all), ("31", fit_31)):
            coefficient = fit_coefficients(cells, fit_idx, response)
            predicted = (
                coefficient[0]
                + coefficient[1] * cells.occupancy_mean.to_numpy()
                + coefficient[2] * cells.sir_db.to_numpy()
            )
            constant = float(actual[fit_idx].mean())
            metrics = score_held(actual[held], predicted[held], constant)
            row = {
                "reference": short,
                "fit_cells": int(fit_idx.sum()),
                "intercept_pp": 100 * float(coefficient[0]),
                "overlap_coefficient_pp": 100 * float(coefficient[1]),
                "sir_coefficient_pp_per_db": 100 * float(coefficient[2]),
                "break_even_overlap_sir_minus15": break_even(coefficient, -15.0),
                "break_even_overlap_sir_minus10": break_even(coefficient, -10.0),
                **metrics,
            }
            rows.append(row)
            report["references"].setdefault(short, {})[f"fit_{label}"] = row

    frame = pd.DataFrame(rows)
    frame.to_csv(CSV / "a16_clean_sentinel_sensitivity.csv", index=False)
    (ROOT / "analysis_zero_compute" / "outputs" / "a16_clean_sentinel_sensitivity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
