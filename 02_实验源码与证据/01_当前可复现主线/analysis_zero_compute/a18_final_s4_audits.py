"""FINAL-S4 zero-compute audits (a18).

Reads only sealed artifacts and the frozen A14 cell table; runs no training
and no new simulation.  Produces:

1. ``a18_leave15_out.csv``          -- leave-one-SIR-stratum-out extrapolation
   test: refit the two-variable envelope on the 28 fitting cells excluding the
   four hard_interference SIR=-15 dB cells, then predict those four cells,
   against a 28-cell constant and a 28-cell SIR-only baseline.
2. ``a18_leave15_out_cells.csv``    -- per-cell observed vs predicted gains
   for the four held-out -15 dB cells (full envelope).
3. ``a18_overlap_increment_bootstrap.csv`` -- block-paired bootstrap of the
   incremental value of overlap beyond SIR: paired squared-error difference
   d_i = e^2(SIR-only) - e^2(overlap+SIR) over the 80 held cells, resampling
   the 20 held-regime x SIR blocks with replacement (B = 10000).
4. ``a18_selector_confusion.csv``   -- TP/FP/FN/TN of the overlap-SIR sign
   rule on the 80 held cells, computed directly from artifacts.
5. ``a18_leave15out_selector.csv``  -- same-calibre selector audit restricted
   to the four -15 dB cells using the 28-cell sign rule.

Exploratory throughout; changes no sealed gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import zc_core as z

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis_zero_compute" / "outputs" / "csv"
FIT_SPLITS = ("id_test", "hard_interference")
HELD_SPLITS = ("unseen_jammer", "unseen_speed", "heldout_channel", "combined_ood")
REFERENCES = {"gain_vs_IQFormer": "IQFormer", "gain_vs_MCLDNN": "MCLDNN"}
MACS_M = {"A5": 41.8, "IQFormer": 355.6}  # sealed complexity table values
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260820
MIN_ROWS = 150


def _fit(design_fit: np.ndarray, response_fit: np.ndarray) -> np.ndarray:
    coefficient, *_ = np.linalg.lstsq(design_fit, response_fit, rcond=None)
    return coefficient


def _full_design(cells: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(cells))] + [cells[c].to_numpy() for c in covariates]
    )


def leave15out(cells: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    """Refit on 28 fitting cells; predict the four excluded -15 dB cells."""
    fit_all = cells.split.isin(FIT_SPLITS).to_numpy()
    excluded = (
        (cells.split == "hard_interference") & (cells.sir_db == -15.0)
    ).to_numpy()
    fit28 = fit_all & ~excluded
    assert int(fit28.sum()) == 28 and int(excluded.sum()) == 4
    rows: list[dict] = []
    cell_rows: list[dict] = []
    report: dict = {"fit_cells": 28, "test_cells": 4, "references": {}}
    test = cells[excluded].sort_values("occupancy_mean").reset_index(drop=True)
    for response, short in REFERENCES.items():
        actual = cells[response].to_numpy()
        predictors: dict[str, np.ndarray] = {}
        constant = float(actual[fit28].mean())
        predictors["constant"] = np.full(len(cells), constant)
        for name, covariates in (
            ("sir_only", ["sir_db"]),
            ("overlap_sir", ["occupancy_mean", "sir_db"]),
        ):
            design = _full_design(cells, covariates)
            coefficient = _fit(design[fit28], actual[fit28])
            predictors[name] = design @ coefficient
            if name == "overlap_sir":
                report["references"][short] = {
                    "beta0_intercept_fraction": float(coefficient[0]),
                    "beta1_overlap_fraction_per_unit": float(coefficient[1]),
                    "beta2_sir_fraction_per_db": float(coefficient[2]),
                }
        for name, predicted in predictors.items():
            error = actual[excluded] - predicted[excluded]
            rows.append(
                {
                    "reference": short,
                    "predictor": name,
                    "fit_cells": 28,
                    "test_cells": 4,
                    "rmse_pp": float(100 * np.sqrt(np.mean(error**2))),
                    "mae_pp": float(100 * np.mean(np.abs(error))),
                    "sign_accuracy": float(
                        np.mean(np.sign(predicted[excluded]) == np.sign(actual[excluded]))
                    ),
                }
            )
        observed = actual[excluded]
        predicted_full = predictors["overlap_sir"][excluded]
        for quartile in range(len(test)):
            cell_rows.append(
                {
                    "reference": short,
                    "quartile_rank": quartile,
                    "occupancy_mean": float(test.occupancy_mean.iloc[quartile]),
                    "observed_gain_pp": float(100 * observed[quartile]),
                    "predicted_gain_pp": float(100 * predicted_full[quartile]),
                }
            )
    return rows, cell_rows, report


def paired_bootstrap(cells: pd.DataFrame) -> tuple[list[dict], dict]:
    """Block-paired bootstrap of the overlap increment beyond SIR."""
    fit = cells.split.isin(FIT_SPLITS).to_numpy()
    held = cells.split.isin(HELD_SPLITS).to_numpy()
    held_frame = cells[held].reset_index(drop=True)
    block_ids, block_index = pd.factorize(
        held_frame.split + ":" + held_frame.sir_db.astype(str)
    )
    n_blocks = int(block_ids.max()) + 1
    assert n_blocks == 20, f"expected 20 regime x SIR blocks, found {n_blocks}"
    blocks = [np.flatnonzero(block_ids == b) for b in range(n_blocks)]
    assert all(len(b) == 4 for b in blocks)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    report: dict = {"blocks": n_blocks, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}
    for response, short in REFERENCES.items():
        actual = cells[response].to_numpy()
        design_sir = _full_design(cells, ["sir_db"])
        design_full = _full_design(cells, ["occupancy_mean", "sir_db"])
        pred_sir = design_sir @ _fit(design_sir[fit], actual[fit])
        pred_full = design_full @ _fit(design_full[fit], actual[fit])
        held_actual = actual[held]
        d = (held_actual - pred_sir[held]) ** 2 - (held_actual - pred_full[held]) ** 2
        point_dmse = float(d.mean())
        draws = np.empty(BOOTSTRAP_DRAWS)
        for i in range(BOOTSTRAP_DRAWS):
            pick = rng.integers(0, n_blocks, n_blocks)
            idx = np.concatenate([blocks[b] for b in pick])
            draws[i] = d[idx].mean()
        low, median, high = np.quantile(draws, [0.025, 0.5, 0.975])
        rmse_sir = float(100 * np.sqrt(np.mean((held_actual - pred_sir[held]) ** 2)))
        rmse_full = float(100 * np.sqrt(np.mean((held_actual - pred_full[held]) ** 2)))
        rows.append(
            {
                "reference": short,
                "point_delta_mse_pp2": 1e4 * point_dmse,
                "mean_delta_mse_pp2": float(1e4 * draws.mean()),
                "median_delta_mse_pp2": float(1e4 * median),
                "ci_low_pp2": float(1e4 * low),
                "ci_high_pp2": float(1e4 * high),
                "prob_delta_mse_positive": float((draws > 0).mean()),
                "rmse_sir_only_pp": rmse_sir,
                "rmse_overlap_sir_pp": rmse_full,
                "rmse_reduction_pp": rmse_sir - rmse_full,
            }
        )
    return rows, report


def selector_confusion(cells: pd.DataFrame) -> tuple[list[dict], dict]:
    """TP/FP/FN/TN of the overlap-SIR sign rule on the 80 held cells."""
    fit = cells.split.isin(FIT_SPLITS).to_numpy()
    held = cells.split.isin(HELD_SPLITS).to_numpy()
    actual = cells["gain_vs_IQFormer"].to_numpy()
    design = _full_design(cells, ["occupancy_mean", "sir_db"])
    coefficient = _fit(design[fit], actual[fit])
    predicted = design @ coefficient
    truth = actual[held] > 0
    chosen = predicted[held] > 0
    tp = int((chosen & truth).sum())
    fp = int((chosen & ~truth).sum())
    fn = int((~chosen & truth).sum())
    tn = int((~chosen & ~truth).sum())
    assert tp + fp + fn + tn == 80
    row = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "positive_cells": int(truth.sum()),
        "selected_cells": int(chosen.sum()),
    }
    report = {
        "note": "sign rule fitted on the 32 fitting cells (gain vs IQFormer-"
        "inspired); truth uses the ten-seed held-cell gains; the oracle "
        "baseline is hindsight",
    }
    return [row], report


def leave15out_selector(cells: pd.DataFrame) -> tuple[list[dict], dict]:
    """Same-calibre selector audit restricted to the four -15 dB cells.

    Uses ten-seed cell-level absolute macro-F1 reconstructed from the sealed
    predictions (the same cell definition as A14), and a sign rule fitted on
    the 28 fitting cells that exclude those four cells.
    """
    fit28 = (
        cells.split.isin(FIT_SPLITS)
        & ~((cells.split == "hard_interference") & (cells.sir_db == -15.0))
    ).to_numpy()
    actual = cells["gain_vs_IQFormer"].to_numpy()
    design = _full_design(cells, ["occupancy_mean", "sir_db"])
    coefficient = _fit(design[fit28], actual[fit28])

    split = "hard_interference"
    meta = z.load_cache_metadata(split)
    overlap = meta["overlap"]
    proposed = z.load_preds(z.PROPOSED, split)
    iq = z.load_preds("iqformer_inspired", split)
    quartiles = np.unique(np.quantile(overlap, [0, 0.25, 0.5, 0.75, 1.0]))
    bins = [(quartiles[i], quartiles[i + 1]) for i in range(len(quartiles) - 1)]
    frozen = cells[
        (cells.split == split) & (cells.sir_db == -15.0)
    ].sort_values("occupancy_mean")
    rows: list[dict] = []
    for low, high in bins:
        selector = overlap >= low
        selector &= overlap <= high if high == quartiles[-1] else overlap < high
        mask = selector & (proposed.sir_db == -15.0)
        labels = proposed.labels[mask]
        if mask.sum() < MIN_ROWS or len(np.unique(labels)) < 8:
            continue
        f1_a5 = float(
            np.mean(
                [z.macro_f1(labels, proposed.pred[s][mask]) for s in range(len(proposed.seeds))]
            )
        )
        f1_iq = float(
            np.mean(
                [z.macro_f1(labels, iq.pred[s][mask]) for s in range(len(iq.seeds))]
            )
        )
        occupancy_mean = float(overlap[mask].mean())
        rows.append(
            {
                "occupancy_mean": occupancy_mean,
                "row_count": int(mask.sum()),
                "f1_a5": f1_a5,
                "f1_iqformer": f1_iq,
                "gain_vs_IQFormer": f1_a5 - f1_iq,
                "predicted_gain_fraction": float(
                    coefficient[0]
                    + coefficient[1] * occupancy_mean
                    + coefficient[2] * (-15.0)
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("occupancy_mean").reset_index(drop=True)
    if len(frame) != 4:
        raise RuntimeError(f"expected 4 -15 dB cells, reconstructed {len(frame)}")
    # verify the reconstruction against the frozen cell table
    check = frame.set_index("occupancy_mean")
    for _, row in frozen.iterrows():
        match = check.index[np.isclose(check.index, row.occupancy_mean, atol=1e-6)]
        if len(match) != 1:
            raise RuntimeError("reconstructed -15 cells do not match the frozen table")
        if not np.isclose(check.loc[match[0], "row_count"], row.row_count):
            raise RuntimeError("window counts differ from the frozen -15 cells")
        if not np.isclose(
            check.loc[match[0], "gain_vs_IQFormer"], row.gain_vs_IQFormer, atol=5e-4
        ):
            raise RuntimeError("gains differ from the frozen -15 cells")

    gain = frame.gain_vs_IQFormer.to_numpy()
    choose_a5 = frame.predicted_gain_fraction.to_numpy() > 0
    realized = float(np.where(choose_a5, frame.f1_a5, frame.f1_iqformer).mean())
    oracle = float(np.where(gain > 0, frame.f1_a5, frame.f1_iqformer).mean())
    fraction = float(choose_a5.mean())
    macs = fraction * MACS_M["A5"] + (1 - fraction) * MACS_M["IQFormer"]
    summary = {
        "cells": 4,
        "hindsight_a5_favoring_cells": int((gain > 0).sum()),
        "selected_a5_cells": int(choose_a5.sum()),
        "sign_accuracy": float(np.mean((frame.predicted_gain_fraction > 0) == (gain > 0))),
        "realized_cell_mean_macro_f1": realized,
        "oracle_cell_mean_macro_f1": oracle,
        "regret_vs_oracle_pp": 100 * (oracle - realized),
        "expected_macs_million": macs,
        "mac_saving_vs_always_iqformer": 1.0 - macs / MACS_M["IQFormer"],
        "note": "ten-seed cell-level aggregates at SIR=-15 dB; the 28-cell "
        "sign rule never sees these cells; same MAC cost model as the "
        "80-cell selector audit",
    }
    return frame.assign(choose_a5=choose_a5).to_dict("records"), summary


def run() -> None:
    cells = pd.read_csv(CSV / "a14_envelope_cells.csv")
    assert len(cells) == 113

    l15_rows, l15_cells, l15_report = leave15out(cells)
    z.write_csv("a18_leave15_out.csv", l15_rows)
    z.write_csv("a18_leave15_out_cells.csv", l15_cells)

    boot_rows, boot_report = paired_bootstrap(cells)
    z.write_csv("a18_overlap_increment_bootstrap.csv", boot_rows)

    conf_rows, conf_report = selector_confusion(cells)
    z.write_csv("a18_selector_confusion.csv", conf_rows)

    sel_rows, sel_report = leave15out_selector(cells)
    z.write_csv("a18_leave15out_selector.csv", sel_rows)

    import json

    report = {
        "schema": "vimd_amc.a18_final_s4_audits.v1",
        "evidence_class": "exploratory_zero_compute_reanalysis_of_sealed_predictions",
        "sources": {
            "cells": "analysis_zero_compute/outputs/csv/a14_envelope_cells.csv",
            "predictions": "artifacts/tvt_v4r_headline_composite/models/*/predictions_*.npz",
        },
        "leave15out": l15_report,
        "bootstrap": boot_report,
        "selector_confusion": conf_report,
        "leave15out_selector": sel_report,
    }
    with open(
        ROOT / "analysis_zero_compute" / "outputs" / "a18_final_s4_audits.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for rows, name in (
        (l15_rows, "leave15_out"),
        (boot_rows, "overlap_increment_bootstrap"),
        (conf_rows, "selector_confusion"),
    ):
        print(f"== {name}")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("== leave15out_selector")
    print(json.dumps(sel_report, indent=2))


if __name__ == "__main__":
    run()
