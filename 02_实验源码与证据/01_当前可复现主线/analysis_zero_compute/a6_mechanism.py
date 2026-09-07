"""A6: mechanism mediation and a re-analysis of the occupancy hypothesis.

The preregistered occupancy gate asked for a *non-increasing* relation between
jammer spectral occupancy and the A5-A0 gain and failed (sealed Spearman
rho = +0.714, one-sided p = 0.949, ``direction_supported = false``).  This
module does not attempt to rescue that hypothesis.  It quantifies the
relation that the sealed evidence actually shows, and separately asks which
recorded internal mechanism quantities co-vary with the realised gain across
the 50 mechanism-instrumented fits.

Everything here is exploratory and must be reported as such: it is not part
of the frozen confirmatory family.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

import zc_core as z

MECHANISM_MODELS = (
    "a3_tri_teacher",
    "a3p_tri_proportional_teacher",
    "a4_tri_teacher_mtl",
    "a5_vimd_full",
    "a7_vimd_no_residual",
)
REGIME = "hard_interference"
DRAWS = 2000


def _seed_gains(regime: str = REGIME) -> dict[tuple[str, int], float]:
    import pandas as pd

    metrics = pd.read_csv(z.COMPOSITE / "metrics.csv")
    subset = metrics[metrics.regime == regime]
    table = subset.pivot_table(index="seed", columns="model", values="macro_f1")
    gains = {}
    for model in table.columns:
        for seed in table.index:
            gains[(model, int(seed))] = float(table.loc[seed, model] - table.loc[seed, z.BACKBONE])
    return gains


def mechanism_association() -> list[dict]:
    gains = _seed_gains()
    records: list[dict] = []
    values: dict[str, list[float]] = {}
    targets: list[float] = []
    keys: list[tuple[str, int]] = []
    for model in MECHANISM_MODELS:
        for seed in z.SEEDS:
            mechanism = z.result_json(model, seed).get("mechanism") or {}
            numeric = {
                key: float(value)
                for key, value in mechanism.items()
                if isinstance(value, (int, float)) and value is not None and np.isfinite(float(value))
            }
            if not numeric:
                continue
            keys.append((model, seed))
            targets.append(gains[(model, seed)])
            for key, value in numeric.items():
                values.setdefault(key, []).append(value)

    target_array = np.asarray(targets)
    for key, series in values.items():
        if len(series) != len(target_array):
            continue
        array = np.asarray(series)
        if np.allclose(array, array[0]):
            continue
        rho, p_value = stats.spearmanr(array, target_array)
        records.append(
            {
                "mechanism_metric": key,
                "fit_count": len(array),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)),
            }
        )
    records.sort(key=lambda row: -abs(row["spearman_rho"]))
    p_values = np.array([row["p_value"] for row in records])
    order = np.argsort(p_values)
    ranked = np.empty(len(p_values))
    ranked[order] = (p_values[order] * len(p_values) / (np.arange(len(p_values)) + 1)).cumsum() / (
        np.arange(len(p_values)) + 1
    ).clip(min=1)
    adjusted = np.minimum.accumulate((p_values[order] * len(p_values) / (np.arange(len(p_values)) + 1))[::-1])[::-1]
    for index, position in enumerate(order):
        records[position]["benjamini_hochberg_q"] = float(min(adjusted[index], 1.0))
    return records


def occupancy_reanalysis() -> tuple[list[dict], list[dict]]:
    meta = z.load_cache_metadata(REGIME)
    occupancy = meta["overlap"]
    families, order = z.jammer_family_labels(REGIME)
    proposed = z.load_preds(z.PROPOSED, REGIME)
    backbone = z.load_preds(z.BACKBONE, REGIME)
    reference = z.load_preds(z.REFERENCE, REGIME)
    labels = proposed.labels

    bin_rows: list[dict] = []
    edges = np.quantile(occupancy, np.linspace(0, 1, 6))
    edges = np.unique(edges)
    for index in range(len(edges) - 1):
        upper = occupancy <= edges[index + 1] if index == len(edges) - 2 else occupancy < edges[index + 1]
        mask = (occupancy >= edges[index]) & upper
        if mask.sum() < 200 or len(np.unique(labels[mask])) < 8:
            continue
        row = {
            "bin": f"[{edges[index]:.3f},{edges[index + 1]:.3f}]",
            "occupancy_mean": float(occupancy[mask].mean()),
            "row_count": int(mask.sum()),
            "mean_sir_db": float(proposed.sir_db[mask].mean()),
        }
        for name, other in (("vs_a0", backbone), ("vs_cssl", reference)):
            stats_row = z.hierarchical_paired_diff(
                labels[mask], other.pred[:, mask], proposed.pred[:, mask], draws=DRAWS
            )
            row[f"gain_{name}"] = stats_row["macro_f1_difference"]
            row[f"gain_{name}_ci_low"] = stats_row["macro_f1_ci95_low"]
            row[f"gain_{name}_ci_high"] = stats_row["macro_f1_ci95_high"]
        bin_rows.append(row)

    family_rows: list[dict] = []
    for name in order:
        mask = families == name
        if mask.sum() < 200 or len(np.unique(labels[mask])) < 8:
            continue
        stats_row = z.hierarchical_paired_diff(
            labels[mask], backbone.pred[:, mask], proposed.pred[:, mask], draws=DRAWS
        )
        family_rows.append(
            {
                "jammer_family": name,
                "row_count": int(mask.sum()),
                "occupancy_mean": float(occupancy[mask].mean()),
                "gain_vs_a0": stats_row["macro_f1_difference"],
                "gain_vs_a0_ci_low": stats_row["macro_f1_ci95_low"],
                "gain_vs_a0_ci_high": stats_row["macro_f1_ci95_high"],
            }
        )

    summary = {}
    if len(bin_rows) >= 3:
        occupancies = np.array([row["occupancy_mean"] for row in bin_rows])
        gains = np.array([row["gain_vs_a0"] for row in bin_rows])
        gains_cssl = np.array([row["gain_vs_cssl"] for row in bin_rows])
        sirs = np.array([row["mean_sir_db"] for row in bin_rows])
        rho_a0 = stats.spearmanr(occupancies, gains)
        rho_cssl = stats.spearmanr(occupancies, gains_cssl)
        summary = {
            "bin_spearman_rho_vs_a0": float(rho_a0.statistic),
            "bin_spearman_p_two_sided_vs_a0": float(rho_a0.pvalue),
            "bin_spearman_rho_vs_cssl": float(rho_cssl.statistic),
            "bin_spearman_p_two_sided_vs_cssl": float(rho_cssl.pvalue),
            "bin_occupancy_sir_spearman": float(stats.spearmanr(occupancies, sirs).statistic),
        }
    if len(family_rows) >= 3:
        occupancies = np.array([row["occupancy_mean"] for row in family_rows])
        gains = np.array([row["gain_vs_a0"] for row in family_rows])
        rho = stats.spearmanr(occupancies, gains)
        summary.update(
            {
                "family_spearman_rho_vs_a0": float(rho.statistic),
                "family_spearman_p_two_sided_vs_a0": float(rho.pvalue),
                "family_count": len(family_rows),
            }
        )
    for row in bin_rows:
        row.update({"analysis": "occupancy_quintile"})
    for row in family_rows:
        row.update({"analysis": "jammer_family"})
    z.ensure_dirs()
    with open(z.CSV.parent / "a6_occupancy_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "preregistered_gate": {
                    "predicted_direction": "negative",
                    "sealed_spearman_rho": 0.7142857,
                    "sealed_one_sided_p": 0.9486111,
                    "direction_supported": False,
                    "positive_mechanism_claim_eligible": False,
                    "source": "artifacts/tvt_v4r_headline_composite/v2_scientific_release_gate.json",
                },
                "exploratory_reanalysis": summary,
                "interpretation": (
                    "The sealed gate is not rescued.  The exploratory re-analysis reproduces a "
                    "positive association between jammer spectral occupancy and the realised gain, "
                    "i.e. the opposite of the preregistered direction.  It must be reported as a "
                    "falsified prediction plus an exploratory observation, never as a confirmed "
                    "mechanism."
                ),
            },
            handle,
            indent=2,
        )
    return bin_rows, family_rows


def run() -> None:
    association = mechanism_association()
    z.write_csv("a6_mechanism_association.csv", association)
    bin_rows, family_rows = occupancy_reanalysis()
    z.write_csv("a6_occupancy_bins.csv", bin_rows)
    z.write_csv("a6_occupancy_families.csv", family_rows)
    _figure(association, bin_rows, family_rows)


def _figure(association, bin_rows, family_rows) -> None:
    import pandas as pd

    plt = z.mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))

    frame = pd.DataFrame(association).head(12).iloc[::-1]
    axes[0].barh(frame.mechanism_metric, frame.spearman_rho, color="#2c7fb8")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].tick_params(axis="y", labelsize=6)
    axes[0].set_xlabel("Spearman $\\rho$ with hard-interference gain")
    axes[0].set_title("Mechanism quantities vs realised gain (50 fits)", fontsize=9)

    bins = pd.DataFrame(bin_rows)
    if not bins.empty:
        axes[1].errorbar(
            bins.occupancy_mean,
            100 * bins.gain_vs_a0,
            yerr=[
                100 * (bins.gain_vs_a0 - bins.gain_vs_a0_ci_low),
                100 * (bins.gain_vs_a0_ci_high - bins.gain_vs_a0),
            ],
            marker="o",
            capsize=3,
            label="A5 $-$ A0",
        )
        axes[1].errorbar(
            bins.occupancy_mean,
            100 * bins.gain_vs_cssl,
            yerr=[
                100 * (bins.gain_vs_cssl - bins.gain_vs_cssl_ci_low),
                100 * (bins.gain_vs_cssl_ci_high - bins.gain_vs_cssl),
            ],
            marker="s",
            capsize=3,
            label="A5 $-$ CSSL",
        )
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("jammer spectral occupancy")
        axes[1].set_ylabel("$\\Delta$ macro-F1 (pp)")
        axes[1].set_title("Gain increases with occupancy\n(opposite of the preregistered direction)", fontsize=8)
        axes[1].legend(fontsize=7)

    families = pd.DataFrame(family_rows)
    if not families.empty:
        axes[2].scatter(families.occupancy_mean, 100 * families.gain_vs_a0, s=28)
        for _, row in families.iterrows():
            axes[2].annotate(
                row.jammer_family,
                (row.occupancy_mean, 100 * row.gain_vs_a0),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axes[2].set_xlabel("family mean occupancy")
        axes[2].set_ylabel("A5 $-$ A0 (pp)")
        axes[2].set_title("Per-family view", fontsize=9)

    z.save_fig(fig, "figA6_mechanism_and_occupancy")
    plt.close(fig)


if __name__ == "__main__":
    run()
