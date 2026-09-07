"""V4.1 envelope audit with an interfered-only 80-cell held set.

This prospective analysis never writes a frozen artifact.  It consumes the
existing artifact-derived cell table from A14 and makes the V4.1 reporting
choice explicit: the primary held evaluation excludes the single jammer-free
clean-retention cell whose SIR is a cache sentinel rather than a physical SIR.

The historical A14 fit is retained exactly as recorded: 32 overlap-quartile
cells from ``id_test`` and ``hard_interference``.  It is not relabelled as an
8-SNR by 4-SIR grid, because that would contradict the source cell table.
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
    "iqformer_inspired": "IQFormer",
    "mcldnn_reimplementation": "MCLDNN",
}


def fit_predict(frame: pd.DataFrame, response: str, with_snr: bool) -> tuple[np.ndarray, np.ndarray]:
    fit = frame.split.isin(FIT_SPLITS).to_numpy()
    cols = [np.ones(len(frame)), frame.occupancy_mean.to_numpy(), frame.sir_db.to_numpy()]
    if with_snr:
        cols.append(frame.snr_db_mean.to_numpy())
    design = np.column_stack(cols)
    coefficient, *_ = np.linalg.lstsq(design[fit], frame[response].to_numpy()[fit], rcond=None)
    return design @ coefficient, coefficient


def score(actual: np.ndarray, prediction: np.ndarray, constant: float) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    const_rmse = float(np.sqrt(np.mean((actual - constant) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((actual - actual.mean()) ** 2)))
    skill = float(1.0 - (rmse**2) / (const_rmse**2)) if const_rmse else float("nan")
    return {
        "rmse_pp": 100 * rmse,
        "constant_rmse_pp": 100 * const_rmse,
        "r_squared_skill": skill,
        "oracle_constant_rmse_pp": 100 * oracle_rmse,
        "sign_agreement": float((np.sign(actual) == np.sign(prediction)).mean()),
        "negative_gain_share": float((actual < 0).mean()),
    }


def run() -> None:
    cells = pd.read_csv(CSV / "a14_envelope_cells.csv")
    fit_mask = cells.split.isin(FIT_SPLITS)
    held_mask = cells.split.isin(HELD_SPLITS)
    if int(fit_mask.sum()) != 32:
        raise RuntimeError(f"expected the historical 32-cell fit, found {fit_mask.sum()}")
    if int(held_mask.sum()) != 80:
        raise RuntimeError(f"expected four interfered held regimes x 20 cells, found {held_mask.sum()}")

    report: dict[str, object] = {
        "schema": "vimd_amc.envelope_v41_interfered_holdout.v1",
        "evidence_class": "exploratory_zero_compute_reanalysis_of_sealed_predictions",
        "source_cells": "analysis_zero_compute/outputs/csv/a14_envelope_cells.csv",
        "fit_definition": {
            "cell_count": 32,
            "splits": list(FIT_SPLITS),
            "historical_definition": "split x SIR x overlap quartile; retained without relabelling",
        },
        "primary_held_definition": {
            "cell_count": 80,
            "splits": list(HELD_SPLITS),
            "excluded": "clean_retention: jammer-free cell has no physical finite SIR",
        },
        "references": {},
    }
    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for model, short in REFERENCES.items():
        response = f"gain_vs_{short}"
        actual = cells[response].to_numpy()
        fit_mean = float(actual[fit_mask].mean())
        primary_pred, coefficient = fit_predict(cells, response, with_snr=False)
        snr_pred, snr_coefficient = fit_predict(cells, response, with_snr=True)
        held_actual = actual[held_mask]
        primary = score(held_actual, primary_pred[held_mask], fit_mean)
        snr = score(held_actual, snr_pred[held_mask], fit_mean)
        reference_report: dict[str, object] = {
            "fit_mean_gain_pp": 100 * fit_mean,
            "two_variable_coefficients": {
                "intercept_pp": 100 * float(coefficient[0]),
                "overlap_pp_per_unit": 100 * float(coefficient[1]),
                "sir_pp_per_db": 100 * float(coefficient[2]),
            },
            "three_variable_coefficients": {
                "intercept_pp": 100 * float(snr_coefficient[0]),
                "overlap_pp_per_unit": 100 * float(snr_coefficient[1]),
                "sir_pp_per_db": 100 * float(snr_coefficient[2]),
                "snr_pp_per_db": 100 * float(snr_coefficient[3]),
            },
            "interfered_80_cell_primary": primary,
            "three_variable_sensitivity": snr,
            "by_regime": {},
        }
        rows.append({"reference": short, "regime": "pooled_interfered_held", "cells": 80, **primary})
        sensitivity_rows.append({"reference": short, "specification": "overlap_plus_sir", **primary})
        sensitivity_rows.append({"reference": short, "specification": "overlap_plus_sir_plus_snr", **snr})
        for split in HELD_SPLITS:
            mask = cells.split.eq(split).to_numpy()
            split_score = score(actual[mask], primary_pred[mask], fit_mean)
            reference_report["by_regime"][split] = {"cells": int(mask.sum()), **split_score}
            rows.append({"reference": short, "regime": split, "cells": int(mask.sum()), **split_score})
        report["references"][short] = reference_report

    pd.DataFrame(rows).to_csv(CSV / "a14_v41_envelope_interfered_holdout.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(CSV / "a14_v41_envelope_sensitivity.csv", index=False)
    (CSV / "a14_v41_envelope_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    run()
