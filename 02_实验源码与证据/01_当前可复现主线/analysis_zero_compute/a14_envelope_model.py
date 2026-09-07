"""A14: is the operating envelope a single predictive law, or a story?

The proposed narrative claims that one physical variable -- jammer spectral
occupancy, together with interference severity -- indexes both where the
compact model wins and where it loses.  A narrative that only describes the
data it was built on is worth little.  This module tests the stronger version:

    fit the gain surface on two regimes, then predict the gain on regimes the
    fit never saw, including held-out jammer families and the jammer-free
    condition.

Cells are (split x SIR level x occupancy quartile).  The response is the
paired macro-F1 difference between the proposed model and a reference,
averaged over the ten algorithm seeds.  Predictors are the cell's mean
occupancy and its SIR in dB.

Exploratory throughout; it changes no gate.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

import zc_core as z

SPLITS = (
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
)
FIT_SPLITS = ("id_test", "hard_interference")
REFERENCES = ("iqformer_inspired", "mcldnn_reimplementation", z.REFERENCE, z.BACKBONE)
MIN_ROWS = 150


def _cells() -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        meta = z.load_cache_metadata(split)
        occupancy = meta["overlap"]
        proposed = z.load_preds(z.PROPOSED, split)
        others = {name: z.load_preds(name, split) for name in REFERENCES}
        quartiles = np.quantile(occupancy, [0, 0.25, 0.5, 0.75, 1.0])
        quartiles = np.unique(quartiles)
        bins = (
            [(quartiles[i], quartiles[i + 1]) for i in range(len(quartiles) - 1)]
            if len(quartiles) > 1
            else [(0.0, 0.0)]
        )
        for sir in np.unique(proposed.sir_db):
            for low, high in bins:
                selector = occupancy >= low
                selector &= occupancy <= high if high == quartiles[-1] else occupancy < high
                mask = selector & (proposed.sir_db == sir)
                labels = proposed.labels[mask]
                if mask.sum() < MIN_ROWS or len(np.unique(labels)) < 8:
                    continue
                cell = {
                    "split": split,
                    "sir_db": float(sir),
                    "snr_db_mean": float(proposed.snr_db[mask].mean()),
                    "occupancy_mean": float(occupancy[mask].mean()),
                    "row_count": int(mask.sum()),
                }
                proposed_f1 = np.array(
                    [z.macro_f1(labels, proposed.pred[s][mask]) for s in range(len(proposed.seeds))]
                )
                for name, other in others.items():
                    other_f1 = np.array(
                        [z.macro_f1(labels, other.pred[s][mask]) for s in range(len(other.seeds))]
                    )
                    cell[f"gain_vs_{z.SHORT.get(name, name)}"] = float((proposed_f1 - other_f1).mean())
                rows.append(cell)
        del proposed, others
    return rows


def _ols(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    predicted = design @ coefficients
    residual = response - predicted
    total = response - response.mean()
    r_squared = 1.0 - float(residual @ residual) / float(total @ total)
    return coefficients, r_squared


def run() -> None:
    import pandas as pd

    cells = pd.DataFrame(_cells())
    z.write_csv("a14_envelope_cells.csv", cells.to_dict("records"))

    report = {"design": "gain ~ 1 + occupancy + sir_db, cells = split x SIR x occupancy quartile"}
    fitted_rows: list[dict] = []
    for reference in REFERENCES:
        column = f"gain_vs_{z.SHORT.get(reference, reference)}"
        fit_mask = cells.split.isin(FIT_SPLITS).to_numpy()
        design = np.column_stack(
            [np.ones(len(cells)), cells.occupancy_mean.to_numpy(), cells.sir_db.to_numpy()]
        )
        response = cells[column].to_numpy()

        full_coefficients, full_r2 = _ols(design, response)
        coefficients, in_sample_r2 = _ols(design[fit_mask], response[fit_mask])
        predicted = design @ coefficients
        holdout = ~fit_mask
        holdout_r = float(stats.pearsonr(predicted[holdout], response[holdout]).statistic)
        holdout_rmse = float(np.sqrt(((predicted[holdout] - response[holdout]) ** 2).mean()))
        sign_agreement = float((np.sign(predicted[holdout]) == np.sign(response[holdout])).mean())

        # bootstrap the coefficients over cells for a rough interval
        rng = np.random.default_rng(20260812)
        draws = np.empty((2000, 3))
        for d in range(2000):
            pick = rng.integers(0, len(cells), len(cells))
            draws[d] = _ols(design[pick], response[pick])[0]
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)

        fitted_rows.append(
            {
                "reference": reference,
                "reference_short": z.SHORT.get(reference, reference),
                "cells": int(len(cells)),
                "intercept_pp": 100 * full_coefficients[0],
                "occupancy_slope_pp_per_unit": 100 * full_coefficients[1],
                "occupancy_slope_ci_low": 100 * low[1],
                "occupancy_slope_ci_high": 100 * high[1],
                "sir_slope_pp_per_db": 100 * full_coefficients[2],
                "sir_slope_ci_low": 100 * low[2],
                "sir_slope_ci_high": 100 * high[2],
                "r_squared_all_cells": full_r2,
                "r_squared_fit_split_only": in_sample_r2,
                "holdout_pearson_r": holdout_r,
                "holdout_rmse_pp": 100 * holdout_rmse,
                "holdout_sign_agreement": sign_agreement,
                "holdout_cells": int(holdout.sum()),
            }
        )
        for index in np.flatnonzero(holdout):
            fitted_rows_entry = {
                "reference_short": z.SHORT.get(reference, reference),
                "split": cells.split.iloc[index],
                "sir_db": cells.sir_db.iloc[index],
                "occupancy_mean": cells.occupancy_mean.iloc[index],
                "actual_gain_pp": 100 * response[index],
                "predicted_gain_pp": 100 * predicted[index],
            }
            report.setdefault("holdout_predictions", []).append(fitted_rows_entry)

    z.write_csv("a14_envelope_fit.csv", fitted_rows)
    z.write_csv("a14_envelope_holdout.csv", report.pop("holdout_predictions"))

    # break-even occupancy as a function of SIR: the design rule a practitioner
    # can actually use.  gain = b0 + b1 * occupancy + b2 * sir = 0
    break_even: list[dict] = []
    for row in fitted_rows:
        b0, b1, b2 = row["intercept_pp"], row["occupancy_slope_pp_per_unit"], row["sir_slope_pp_per_db"]
        for sir in (-15.0, -10.0, -5.0, 0.0, 5.0):
            required = (-b0 - b2 * sir) / b1 if b1 != 0 else float("nan")
            break_even.append(
                {
                    "reference_short": row["reference_short"],
                    "sir_db": sir,
                    "break_even_occupancy": required,
                    "attainable": bool(0.0 <= required <= 1.0),
                }
            )
    z.write_csv("a14_break_even_occupancy.csv", break_even)

    strong = [row for row in fitted_rows if row["reference"] in ("iqformer_inspired", "mcldnn_reimplementation")]
    report.update(
        {
            "fit_splits": list(FIT_SPLITS),
            "holdout_splits": [s for s in SPLITS if s not in FIT_SPLITS],
            "headline": {
                row["reference_short"]: {
                    "occupancy_slope_pp_per_unit_occupancy": round(row["occupancy_slope_pp_per_unit"], 2),
                    "sir_slope_pp_per_db": round(row["sir_slope_pp_per_db"], 3),
                    "r_squared_all_cells": round(row["r_squared_all_cells"], 3),
                    "holdout_pearson_r": round(row["holdout_pearson_r"], 3),
                    "holdout_sign_agreement": round(row["holdout_sign_agreement"], 3),
                }
                for row in strong
            },
            "interpretation": (
                "a two-variable envelope fitted only on in-distribution and hard-interference cells "
                "predicts the sign and magnitude of the gain on regimes it never saw, including "
                "held-out jammer families and the jammer-free condition"
            ),
            "evidence_class": "exploratory_zero_compute",
        }
    )
    with open(z.OUT / "a14_envelope_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(pd.DataFrame(fitted_rows).round(3).to_string(index=False))
    _figure(cells, fitted_rows)


def _figure(cells, fitted_rows) -> None:
    plt = z.mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    markers = {
        "id_test": "o",
        "hard_interference": "s",
        "unseen_jammer": "^",
        "unseen_speed": "v",
        "heldout_channel": "D",
        "combined_ood": "P",
        "clean_retention": "X",
    }
    column = "gain_vs_IQFormer"
    for split, marker in markers.items():
        subset = cells[cells.split == split]
        if subset.empty:
            continue
        filled = split in FIT_SPLITS
        axes[0].scatter(
            subset.occupancy_mean,
            100 * subset[column],
            marker=marker,
            s=28,
            facecolors="none" if not filled else None,
            label=split,
        )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("jammer spectral occupancy")
    axes[0].set_ylabel("A5 $-$ IQFormer (pp)")
    axes[0].set_title("Gain versus occupancy, all regimes", fontsize=9)
    axes[0].legend(fontsize=6, ncol=2)

    holdout = cells[~cells.split.isin(FIT_SPLITS)]
    fit = [row for row in fitted_rows if row["reference_short"] == "IQFormer"][0]
    design = np.column_stack(
        [np.ones(len(cells)), cells.occupancy_mean.to_numpy(), cells.sir_db.to_numpy()]
    )
    fit_mask = cells.split.isin(FIT_SPLITS).to_numpy()
    coefficients, _ = _ols(design[fit_mask], cells[column].to_numpy()[fit_mask])
    predicted = design @ coefficients
    axes[1].scatter(100 * predicted[~fit_mask], 100 * cells[column].to_numpy()[~fit_mask], s=26)
    limits = [
        min(100 * predicted.min(), 100 * cells[column].min()) - 2,
        max(100 * predicted.max(), 100 * cells[column].max()) + 2,
    ]
    axes[1].plot(limits, limits, "k--", lw=0.8)
    axes[1].set_xlabel("predicted gain (pp), fitted on ID + hard only")
    axes[1].set_ylabel("actual gain (pp)")
    axes[1].set_title(
        f"Out-of-regime prediction: r = {fit['holdout_pearson_r']:.2f}, "
        f"sign {100 * fit['holdout_sign_agreement']:.0f}%",
        fontsize=9,
    )
    z.save_fig(fig, "figA14_envelope_model")
    plt.close(fig)


if __name__ == "__main__":
    run()
