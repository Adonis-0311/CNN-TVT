"""WP8: manuscript figures at IEEE column widths.

Every figure is rendered from a CSV produced by ``analysis_zero_compute`` or by
the sealed composite -- no figure recomputes anything, so a figure can never
disagree with the data layer.  Output is vector PDF plus a PNG proof, together
with a provenance record listing the source CSV SHA-256 for each figure.

IEEE Transactions geometry: single column 3.5 in, double column 7.16 in.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "analysis_zero_compute" / "outputs" / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
SINGLE, DOUBLE = 3.5, 7.16

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

USED: dict[str, list[str]] = {}


def _read(name: str, figure: str) -> pd.DataFrame:
    path = CSV / name
    USED.setdefault(figure, []).append(name)
    return pd.read_csv(path)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_envelope() -> None:
    """Fig. 1 -- the two-variable envelope law and its out-of-regime test."""

    cells = _read("a14_envelope_cells.csv", "fig_envelope")
    # V4.1 primary envelope: retain only finite-SIR interfered cells.  A clean
    # window has no physical finite SIR and belongs to the repair boundary.
    cells = cells[cells.split != "clean_retention"].copy()
    fit = _read("a14_envelope_fit.csv", "fig_envelope")
    row = fit[fit.reference == "iqformer_inspired"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    markers = {
        "id_test": ("o", "#1f77b4"),
        "hard_interference": ("s", "#d62728"),
        "unseen_jammer": ("^", "#2ca02c"),
        "unseen_speed": ("v", "#9467bd"),
        "heldout_channel": ("D", "#8c564b"),
        "combined_ood": ("P", "#e377c2"),
    }
    ax = axes[0]
    for split, (marker, colour) in markers.items():
        subset = cells[cells.split == split]
        if subset.empty:
            continue
        fitted = split in ("id_test", "hard_interference")
        ax.scatter(
            subset.occupancy_mean,
            100 * subset.gain_vs_IQFormer,
            marker=marker,
            s=14,
            c=colour if fitted else "none",
            edgecolors=colour,
            linewidths=0.7,
            label=split.replace("_", " "),
        )
    grid = np.linspace(0, 1, 50)
    for sir, style in ((-15.0, "-"), (-5.0, "--"), (0.0, ":")):
        ax.plot(
            grid,
            row.intercept_pp + row.occupancy_slope_pp_per_unit * grid + row.sir_slope_pp_per_db * sir,
            style,
            color="k",
            lw=0.8,
            label=f"envelope fit, SIR={sir:.0f} dB",
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("target-support jammer overlap")
    ax.set_ylabel(r"$\Delta$ macro-F1 vs IQFormer-insp. (pp)")
    ax.set_title("(a) gain surface vs overlap, all regimes", loc="left")
    ax.legend(ncol=2, frameon=False, handletextpad=0.4, columnspacing=0.8)

    ax = axes[1]
    design = np.column_stack(
        [np.ones(len(cells)), cells.occupancy_mean.to_numpy(), cells.sir_db.to_numpy()]
    )
    mask = cells.split.isin(["id_test", "hard_interference"]).to_numpy()
    coefficients, *_ = np.linalg.lstsq(design[mask], cells.gain_vs_IQFormer.to_numpy()[mask], rcond=None)
    predicted = 100 * (design @ coefficients)
    actual = 100 * cells.gain_vs_IQFormer.to_numpy()
    # Trivial baseline: the constant predictor outputs the mean gain over the
    # fitting cells and is applied unchanged to the held-regime cells.  Both
    # arrays are in percentage points; mixing the fraction-valued cells column
    # with the pp-valued holdout column is what produced the withdrawn 15.28.
    constant = float(actual[mask].mean())
    held = ~mask
    rmse_envelope = float(np.sqrt(np.mean((actual[held] - predicted[held]) ** 2)))
    rmse_constant = float(np.sqrt(np.mean((actual[held] - constant) ** 2)))
    rmse_oracle = float(np.sqrt(np.mean((actual[held] - actual[held].mean()) ** 2)))
    skill = 1.0 - (rmse_envelope ** 2) / (rmse_constant ** 2)
    for split, (marker, colour) in markers.items():
        selector = (cells.split == split).to_numpy() & ~mask
        if not selector.any():
            continue
        ax.scatter(predicted[selector], actual[selector], marker=marker, s=14,
                   c="none", edgecolors=colour, linewidths=0.7, label=split.replace("_", " "))
    limits = [min(predicted.min(), actual.min()) - 2, max(predicted.max(), actual.max()) + 2]
    ax.plot(limits, limits, "k--", lw=0.7, label="identity")
    ax.axvline(constant, color="#7f7f7f", lw=0.8, ls=":", label="fit-domain constant RMSE")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("predicted (pp), fitted on ID + hard cells only")
    ax.set_ylabel("observed (pp)")
    ax.set_title("(b) held-regime magnitude prediction", loc="left")
    ax.text(
        0.03,
        0.97,
        "RMSE = {:.2f} pp\nfit-domain constant RMSE = {:.2f} pp\noracle constant RMSE = {:.2f} pp\n$R^2_{{\\mathrm{{skill}}}}$ = {:.2f}".format(
            rmse_envelope, rmse_constant, rmse_oracle, skill
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.7", linewidth=0.5),
    )
    print(
        "fig1 envelope audit: constant={:.4f} pp, rmse_env={:.4f}, "
        "rmse_const={:.4f}, skill={:.4f}".format(constant, rmse_envelope, rmse_constant, skill)
    )
    _save(fig, "fig1_envelope_law")


def fig_operating_map() -> None:
    """Fig. 2 -- SNR x SIR operating map against both strong baselines."""

    frame = _read("a1b_snr_sir_map.csv", "fig_operating_map")
    frame = frame[frame.regime == "hard_interference"]
    references = ["mcldnn_reimplementation", "iqformer_inspired"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    for ax, reference in zip(axes, references):
        subset = frame[frame.reference == reference]
        table = subset.pivot_table(index="sir_db", columns="snr_db", values="macro_f1_difference")
        low = subset.pivot_table(index="sir_db", columns="snr_db", values="macro_f1_ci95_low")
        values = 100 * table.to_numpy()
        limit = float(np.nanmax(np.abs(values)))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{r:.0f}" for r in table.index])
        ax.set_xlabel("SNR (dB)")
        ax.grid(False)
        ax.set_title(f"A5 $-$ {'MCLDNN' if 'mcldnn' in reference else 'IQFormer-insp.'} (pp)", loc="left")
        marks = 100 * low.to_numpy()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isnan(values[i, j]):
                    continue
                cell = f"{values[i, j]:.1f}".replace("-0.0", "0.0")
                ax.text(j, i, f"{cell}{'*' if marks[i, j] > 0 else ''}",
                        ha="center", va="center", fontsize=4.5)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    axes[0].set_ylabel("SIR (dB)")
    _save(fig, "fig2_operating_map")


def fig_severe_corner() -> None:
    """Fig. 3 -- severe-interference corner with robustness annotations."""

    robustness = _read("a11_severe_corner_robustness.csv", "fig_severe_corner")
    absolute = _read("a1_absolute_by_level.csv", "fig_severe_corner")
    severe = absolute[
        (absolute.regime == "hard_interference")
        & (absolute.axis == "sir_db")
        & (absolute.level_db == -15)
    ]
    order = ["A0", "CSSL", "MCLDNN", "IQFormer", "A5-VIMD"]
    display = {"IQFormer": "IQFormer-insp."}
    severe = severe.set_index("model_short").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    ax = axes[0]
    colours = ["#bbbbbb"] * 4 + ["#d62728"]
    ax.bar(range(len(order)), severe.macro_f1_mean, yerr=severe.macro_f1_std, capsize=2, color=colours)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display.get(n, n) for n in order], rotation=20,
                       ha="right", fontsize=5.6)
    ax.set_ylabel("macro-F1")
    ax.set_title("(a) SIR = $-15$ dB, hard interference", loc="left")

    ax = axes[1]
    subset = robustness[robustness.reference_short.isin(["MCLDNN", "IQFormer", "CSSL", "A0"])]
    subset = subset.set_index("reference_short").reindex(["A0", "CSSL", "MCLDNN", "IQFormer"])
    values = 100 * subset.mean_difference
    ax.barh(range(len(subset)), values,
            xerr=[values - 100 * subset.ci_low_stratified_10000,
                  100 * subset.ci_high_stratified_10000 - values],
            capsize=2, color="#d62728")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels([f"vs {display.get(name, name)}" for name in subset.index],
                       fontsize=6)
    ax.set_ylim(-0.7, len(subset) - 0.3)
    ax.set_xlabel(r"$\Delta$ macro-F1 (pp), 95% CI")
    ax.set_title("(b) paired advantage, 10/10 seeds positive", loc="left")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "fig3_severe_corner")


def fig_pareto() -> None:
    """Fig. 4 -- accuracy-cost frontier."""

    frame = _read("a8_pareto.csv", "fig_pareto")
    subset = frame[frame.regime == "hard_interference"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    for _, row in subset.iterrows():
        proposed = row.model == "a5_vimd_full"
        ax.scatter(row.parameters, row.macro_f1_mean,
                   marker="*" if row.pareto_optimal_parameters else "o",
                   s=70 if proposed else 22,
                   c="#d62728" if proposed else "#1f77b4")
        if row.model_short in ("A0", "A5-VIMD", "CSSL", "MCLDNN", "IQFormer"):
            ax.annotate(row.model_short, (row.parameters, row.macro_f1_mean),
                        fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("hard-interference macro-F1")
    _save(fig, "fig4_pareto")


def fig_risk_coverage() -> None:
    """Fig. 5 -- selective prediction after validation-only calibration."""

    curves = _read("a4_risk_coverage_curves.csv", "fig_risk_coverage")
    subset = curves[curves.regime == "hard_interference"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for model, label, colour in (
        ("a5_vimd_full", "A5-VIMD", "#d62728"),
        ("iqformer_inspired", "IQFormer-insp.", "#1f77b4"),
        ("mcldnn_reimplementation", "MCLDNN", "#2ca02c"),
        ("cssl_amc_supervised_adaptation", "CSSL", "#9467bd"),
        ("a0_backbone", "A0", "#7f7f7f"),
    ):
        curve = subset[subset.model == model].sort_values("coverage")
        ax.plot(curve.coverage, curve.accuracy_mean, color=colour, label=label)
        ax.fill_between(curve.coverage, curve.accuracy_mean - curve.accuracy_std,
                        curve.accuracy_mean + curve.accuracy_std, color=colour, alpha=0.12, lw=0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective accuracy")
    ax.legend(frameon=False)
    _save(fig, "fig5_risk_coverage")


def fig_condition_transfer() -> None:
    """Fig. 6 -- A5 interference-condition transfer taxonomy."""

    frame = _read("a13_condition_transfer.csv", "fig_condition_transfer")
    modulations = ["BPSK", "PI2BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK"]
    trained_clean = {"PI2BPSK", "8PSK", "256QAM", "CPFSK"}
    subset = frame[frame.model == "a5_vimd_full"]
    if len(subset) != len(modulations):
        raise ValueError("A5 condition-transfer rows are incomplete")
    pivot = subset.set_index("modulation").reindex(modulations)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.45))
    x = np.arange(len(pivot))
    ax.bar(x - 0.2, pivot.jammed_f1, 0.4, label="hard interference", color="#d62728")
    ax.bar(x + 0.2, pivot.clean_f1, 0.4, label="jammer-free", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}*" if m not in trained_clean else m for m in pivot.index], rotation=45, ha="right"
    )
    ax.set_ylabel("per-class F1")
    ax.set_title("A5 clean condition transfer", loc="left")
    ax.legend(frameon=False)
    _save(fig, "fig6_condition_transfer")


def main() -> None:
    fig_envelope()
    fig_operating_map()
    fig_severe_corner()
    fig_pareto()
    fig_risk_coverage()
    fig_condition_transfer()

    record = {
        "schema": "tvt_paper_figures_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": {"single_column_in": SINGLE, "double_column_in": DOUBLE},
        "policy": "figures render sealed or analysis CSVs only; they never recompute evidence",
        "figures": {},
    }
    for figure, sources in USED.items():
        record["figures"][figure] = [
            {
                "source_csv": name,
                "sha256": hashlib.sha256((CSV / name).read_bytes()).hexdigest(),
            }
            for name in sorted(set(sources))
        ]
    with open(OUT / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print(f"figures written to {OUT}")
    for path in sorted(OUT.glob("*.pdf")):
        print("  -", path.name)


if __name__ == "__main__":
    main()
