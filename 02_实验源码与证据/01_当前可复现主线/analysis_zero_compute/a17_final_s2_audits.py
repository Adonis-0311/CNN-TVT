"""A17: FINAL-S2 zero-compute audits (covariate ladder, selector, convergence).

Three reviewer-facing audits recomputed entirely from frozen artifacts; no
model is retrained and no prediction changes:

1. Covariate ladder M0--M4 on the same 32 fitting cells and the same 80
   finite-SIR interfered held cells as the primary envelope.  M1 isolates
   SIR-only skill and M2 isolates overlap-only skill, so the incremental
   value of each covariate beyond the other is directly reported.
2. Model-selection selector evaluation on the 80 held cells, comparing
   always-A5 / always-IQFormer / SIR-only sign selector / overlap+SIR sign
   selector / hindsight oracle, with realized macro-F1, regret, A5
   selection fraction, and expected MAC cost.
3. Comparator convergence audit reading the sealed training histories
   (selected epoch, completed epochs, early-stopping status) for the two
   I/Q references and CSSL.

Exploratory throughout; nothing here changes a sealed conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import zc_core as z

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis_zero_compute" / "outputs" / "csv"
OUT = ROOT / "analysis_zero_compute" / "outputs"
FIT_SPLITS = ("id_test", "hard_interference")
HELD_SPLITS = ("unseen_jammer", "unseen_speed", "heldout_channel", "combined_ood")
REFERENCES = {"gain_vs_IQFormer": "IQFormer", "gain_vs_MCLDNN": "MCLDNN"}
MACS_M = {"A5": 41.8, "IQFormer": 355.6}  # sealed complexity table values
EPOCH_BUDGET = 30
MIN_ROWS = 150

LADDER = {
    "M0_constant": [],
    "M1_sir_only": ["sir_db"],
    "M2_overlap_only": ["occupancy_mean"],
    "M3_overlap_sir": ["occupancy_mean", "sir_db"],
    "M4_overlap_sir_snr": ["occupancy_mean", "sir_db", "snr_db_mean"],
}


def covariate_ladder(cells: pd.DataFrame) -> tuple[list[dict], dict]:
    fit = cells.split.isin(FIT_SPLITS).to_numpy()
    held = cells.split.isin(HELD_SPLITS).to_numpy()
    assert int(fit.sum()) == 32 and int(held.sum()) == 80
    rows: list[dict] = []
    report: dict = {"fit_cells": 32, "held_cells": 80, "references": {}}
    for response, short in REFERENCES.items():
        actual = cells[response].to_numpy()
        constant = float(actual[fit].mean())
        const_rmse = float(np.sqrt(np.mean((actual[held] - constant) ** 2)))
        reference_rows = []
        for model, covariates in LADDER.items():
            design = np.column_stack(
                [np.ones(len(cells))] + [cells[c].to_numpy() for c in covariates]
            )
            coefficient, *_ = np.linalg.lstsq(design[fit], actual[fit], rcond=None)
            predicted = design @ coefficient
            rmse = float(np.sqrt(np.mean((actual[held] - predicted[held]) ** 2)))
            skill = 1.0 - rmse**2 / const_rmse**2
            row = {
                "reference": short,
                "model": model,
                "covariates": "+".join(covariates) if covariates else "intercept",
                "held_rmse_pp": 100 * rmse,
                "r_squared_skill": skill,
                "fit_domain_constant_rmse_pp": 100 * const_rmse,
            }
            rows.append(row)
            reference_rows.append(row)
        by = {r["model"]: r["held_rmse_pp"] for r in reference_rows}
        report["references"][short] = {
            "rows": reference_rows,
            "incremental_rmse_reduction_pp": {
                "overlap_given_sir": by["M1_sir_only"] - by["M3_overlap_sir"],
                "sir_given_overlap": by["M2_overlap_only"] - by["M3_overlap_sir"],
                "snr_given_overlap_sir": by["M3_overlap_sir"] - by["M4_overlap_sir_snr"],
            },
        }
    return rows, report


def _held_cells_absolute() -> tuple[pd.DataFrame, dict]:
    """Reconstruct per-cell absolute macro-F1 for A5 and IQFormer on the 80
    held cells, following the A14 cell definition exactly."""
    a5 = {split: z.load_preds(z.PROPOSED, split) for split in HELD_SPLITS}
    iq = {split: z.load_preds("iqformer_inspired", split) for split in HELD_SPLITS}
    rows: list[dict] = []
    for split in HELD_SPLITS:
        meta = z.load_cache_metadata(split)
        overlap = meta["overlap"]
        proposed = a5[split]
        quartiles = np.unique(np.quantile(overlap, [0, 0.25, 0.5, 0.75, 1.0]))
        bins = [(quartiles[i], quartiles[i + 1]) for i in range(len(quartiles) - 1)]
        for sir in np.unique(proposed.sir_db):
            for quartile, (low, high) in enumerate(bins):
                selector = overlap >= low
                selector &= overlap <= high if high == quartiles[-1] else overlap < high
                mask = selector & (proposed.sir_db == sir)
                labels = proposed.labels[mask]
                if mask.sum() < MIN_ROWS or len(np.unique(labels)) < 8:
                    continue
                rows.append(
                    {
                        "split": split,
                        "sir_db": float(sir),
                        "quartile": quartile,
                        "snr_db_mean": float(proposed.snr_db[mask].mean()),
                        "occupancy_mean": float(overlap[mask].mean()),
                        "row_count": int(mask.sum()),
                        "f1_a5": float(
                            np.mean(
                                [z.macro_f1(labels, proposed.pred[s][mask]) for s in range(len(proposed.seeds))]
                            )
                        ),
                        "f1_iqformer": float(
                            np.mean(
                                [z.macro_f1(labels, iq[split].pred[s][mask]) for s in range(len(iq[split].seeds))]
                            )
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    # verify against the frozen cell table
    frozen = pd.read_csv(CSV / "a14_envelope_cells.csv")
    frozen_held = frozen[frozen.split.isin(HELD_SPLITS)].copy()
    if len(frame) != 80 or len(frozen_held) != 80:
        raise RuntimeError("expected 80 reconstructed and 80 frozen held cells")
    key = lambda f: f.sort_values(["split", "sir_db", "occupancy_mean"]).reset_index(drop=True)
    rec, ref = key(frame), key(frozen_held)
    if not np.array_equal(rec.split.to_numpy(), ref.split.to_numpy()):
        raise RuntimeError("cell split order differs from the frozen cell table")
    if not np.allclose(rec.sir_db, ref.sir_db, atol=1e-9):
        raise RuntimeError("cell SIR levels differ from the frozen cell table")
    if not np.allclose(rec.occupancy_mean, ref.occupancy_mean, atol=1e-6):
        raise RuntimeError("cell occupancy means differ from the frozen cell table")
    if not np.array_equal(rec.row_count.to_numpy(), ref.row_count.to_numpy()):
        raise RuntimeError("cell window counts differ from the frozen cell table")
    return frame, {"held_cells": int(len(frame))}


def selector_evaluation(cells: pd.DataFrame) -> tuple[list[dict], dict]:
    gain = (cells.f1_a5 - cells.f1_iqformer).to_numpy()
    sir = cells.sir_db.to_numpy()
    overlap = cells.occupancy_mean.to_numpy()
    ones = np.ones(len(cells))

    # sign selectors are driven by envelope fits on the FITTING cells only
    frozen = pd.read_csv(CSV / "a14_envelope_cells.csv")
    fit_mask = frozen.split.isin(FIT_SPLITS).to_numpy()

    def fit_coeff(covariates: list[str]) -> np.ndarray:
        design = np.column_stack([np.ones(len(frozen))] + [frozen[c].to_numpy() for c in covariates])
        coefficient, *_ = np.linalg.lstsq(
            design[fit_mask], frozen["gain_vs_IQFormer"].to_numpy()[fit_mask], rcond=None
        )
        return coefficient, design

    coefficient_1, design_1 = fit_coeff(["sir_db"])
    coefficient_3, design_3 = fit_coeff(["sir_db", "occupancy_mean"])
    # transfer the fitted decision surface onto held cells by covariate value
    held_pred_sir = coefficient_1[0] + coefficient_1[1] * sir
    held_pred_full = coefficient_3[0] + coefficient_3[1] * sir + coefficient_3[2] * overlap
    del design_1, design_3

    selectors = {
        "always_a5": np.ones(len(cells), dtype=bool),
        "always_iqformer": np.zeros(len(cells), dtype=bool),
        "sir_only_selector": held_pred_sir > 0,
        "overlap_sir_selector": held_pred_full > 0,
        "oracle": gain > 0,
    }
    rows: list[dict] = []
    for name, choose_a5 in selectors.items():
        realized = np.where(choose_a5, cells.f1_a5.to_numpy(), cells.f1_iqformer.to_numpy())
        score = float(realized.mean())
        oracle = float(np.where(gain > 0, cells.f1_a5, cells.f1_iqformer).mean())
        fraction = float(choose_a5.mean())
        macs = fraction * MACS_M["A5"] + (1 - fraction) * MACS_M["IQFormer"]
        rows.append(
            {
                "selector": name,
                "held_cell_mean_macro_f1": score,
                "regret_vs_oracle_pp": 100 * (oracle - score),
                "a5_selection_fraction": fraction,
                "expected_macs_million": macs,
                "mac_saving_vs_always_iqformer": 1.0 - macs / MACS_M["IQFormer"],
            }
        )
    report = {
        "held_cells": int(len(cells)),
        "macs_million": MACS_M,
        "oracle_cell_mean_macro_f1": float(
            np.where(gain > 0, cells.f1_a5, cells.f1_iqformer).mean()
        ),
        "note": "sign selectors use envelope coefficients fitted only on the 32 "
        "fitting cells; the oracle is hindsight and not achievable",
    }
    return rows, report


def convergence_audit() -> tuple[list[dict], dict]:
    models = (
        "mcldnn_reimplementation",
        "iqformer_inspired",
        "cssl_amc_supervised_adaptation",
        "a5_vimd_full",
    )
    rows: list[dict] = []
    summary: dict = {}
    for model in models:
        selected, completed = [], []
        for seed in z.SEEDS:
            training = z.result_json(model, seed)["training"]
            selected.append(int(training["selected_epoch"]))
            completed.append(int(training["epochs_completed"]))
        selected_arr = np.array(selected)
        completed_arr = np.array(completed)
        early_stopped = int((completed_arr < EPOCH_BUDGET).sum())
        at_boundary = int((completed_arr == EPOCH_BUDGET).sum())
        rows.append(
            {
                "model": z.SHORT.get(model, model),
                "median_selected_epoch": float(np.median(selected_arr)),
                "selected_epoch_min": int(selected_arr.min()),
                "selected_epoch_max": int(selected_arr.max()),
                "seeds_at_epoch_budget": at_boundary,
                "seeds_selected_inside_budget": int((selected_arr < EPOCH_BUDGET).sum()),
                "early_stopped_seeds": early_stopped,
                "median_epochs_completed": float(np.median(completed_arr)),
            }
        )
        summary[z.SHORT.get(model, model)] = {
            "selected_epochs": selected,
            "epochs_completed": completed,
        }
    return rows, {"epoch_budget": EPOCH_BUDGET, "seeds": list(z.SEEDS), "per_model": summary}


def run() -> None:
    cells = pd.read_csv(CSV / "a14_envelope_cells.csv")

    ladder_rows, ladder_report = covariate_ladder(cells)
    z.write_csv("a17_covariate_ladder.csv", ladder_rows)

    held_abs, held_meta = _held_cells_absolute()
    z.write_csv("a17_held_cell_absolute_f1.csv", held_abs.to_dict("records"))
    selector_rows, selector_report = selector_evaluation(held_abs)
    z.write_csv("a17_selector_evaluation.csv", selector_rows)

    convergence_rows, convergence_report = convergence_audit()
    z.write_csv("a17_convergence_audit.csv", convergence_rows)

    report = {
        "schema": "vimd_amc.a17_final_s2_audits.v1",
        "evidence_class": "exploratory_zero_compute_reanalysis_of_sealed_predictions",
        "sources": {
            "cells": "analysis_zero_compute/outputs/csv/a14_envelope_cells.csv",
            "predictions": "artifacts/tvt_v4r_headline_composite/models",
            "overlap": "standards/cache_factor_headline_1024_v2/<split>/overlap.npy",
            "training_histories": "artifacts/tvt_v4r_headline_composite/models/*/result.json",
        },
        "covariate_ladder": ladder_report,
        "held_cell_reconstruction": held_meta,
        "selector_evaluation": selector_report,
        "convergence_audit": convergence_report,
    }
    (OUT / "a17_final_s2_audits.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("== covariate ladder ==")
    print(pd.DataFrame(ladder_rows).round(4).to_string(index=False))
    print("\n== selector evaluation ==")
    print(pd.DataFrame(selector_rows).round(4).to_string(index=False))
    print("\n== convergence audit ==")
    print(pd.DataFrame(convergence_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    run()
