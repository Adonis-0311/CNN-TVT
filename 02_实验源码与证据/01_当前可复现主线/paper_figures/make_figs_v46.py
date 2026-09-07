"""IEEE TVT V4.6 redraw of manuscript Figs. 2--5.

Fig. 1 (the simulation-component teacher) is intentionally untouched.  Every
quantitative mark below comes from the frozen analysis CSV/JSON layer.  The
script changes presentation only: typography, palette, panel hierarchy, and
vector export.  It does not train a model, refit the frozen envelope, or alter
any experiment artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "analysis_zero_compute" / "outputs" / "csv"
OUT = Path(__file__).resolve().parent / "outputs_v46"

SINGLE_IN = 3.5
DOUBLE_IN = 7.16
MM_PER_IN = 25.4

# Okabe-Ito-derived, print-safe semantic palette.
A5_BLUE = "#0072B2"
A5_LIGHT = "#8EC7E6"
IQ_GREEN = "#009E73"
MC_ORANGE = "#D55E00"
SIDECAR_PURPLE = "#8A5FBF"
NEUTRAL = "#5F6368"
NEUTRAL_LIGHT = "#C8CDD2"
INK = "#1F2328"
GRID = "#D9DDE1"
WHITE = "#FFFFFF"

GAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "a5_gain", [MC_ORANGE, "#F7F7F7", A5_BLUE], N=256
)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.8,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "figure.dpi": 180,
    }
)

USED: dict[str, list[Path]] = {}
EXCLUSIONS: dict[str, dict[str, object]] = {}


def read_csv(name: str, figure: str) -> pd.DataFrame:
    path = CSV / name
    USED.setdefault(figure, []).append(path)
    return pd.read_csv(path)


def read_json(name: str, figure: str) -> dict:
    path = CSV / name
    USED.setdefault(figure, []).append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )


def style_numeric_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8, pad=2.5)
    if y_grid:
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def fig2_rank_instability() -> None:
    """Campaign-internal A5 rank map against the two strong I/Q references."""

    figure = "fig2_rank_instability"
    frame = read_csv("a1b_snr_sir_map.csv", figure)
    frame = frame[frame["regime"] == "hard_interference"].copy()
    references = [
        ("mcldnn_reimplementation", "A5 - MCLDNN"),
        ("iqformer_inspired", "A5 - IQFormer-inspired"),
    ]

    arrays: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    limit = 0.0
    for reference, _ in references:
        subset = frame[frame["reference"] == reference]
        values = subset.pivot(index="sir_db", columns="snr_db", values="macro_f1_difference") * 100
        lows = subset.pivot(index="sir_db", columns="snr_db", values="macro_f1_ci95_low") * 100
        arrays.append((values, lows))
        limit = max(limit, float(np.nanmax(np.abs(values.to_numpy()))))
    limit = float(np.ceil(limit / 2.0) * 2.0)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.42), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.18, top=0.88, wspace=0.12)

    image = None
    for panel, (ax, (_, title), (values, lows)) in enumerate(zip(axes, references, arrays)):
        image = ax.imshow(
            values.to_numpy(),
            cmap=GAIN_CMAP,
            norm=norm,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(values.columns)))
        ax.set_xticklabels([f"{x:.0f}" for x in values.columns])
        ax.set_yticks(np.arange(len(values.index)))
        ax.set_yticklabels([f"{y:.0f}" for y in values.index])
        ax.set_xlabel("SNR (dB)")
        # Keep the panel marker inside the panel boundary and center the
        # comparison title above the heatmap.  This avoids the detached labels
        # that appeared at the outer figure margins in the manuscript render.
        ax.text(
            0.0,
            1.035,
            "(a)" if panel == 0 else "(b)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.035,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.8,
            clip_on=False,
        )
        ax.tick_params(direction="out", length=2.6, width=0.8, pad=2.0)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = float(values.iloc[i, j])
                rgba = GAIN_CMAP(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = WHITE if luminance < 0.46 else INK
                mark = "*" if float(lows.iloc[i, j]) > 0 else ""
                ax.text(
                    j,
                    i,
                    f"{value:.1f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_color,
                )

    axes[0].set_ylabel("SIR (dB)")
    assert image is not None
    cbar_ax = fig.add_axes([0.945, 0.18, 0.016, 0.70])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("A5 gain (percentage points)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    save_figure(fig, figure)


def fig3_campaign_diagnostic_envelope() -> None:
    """Frozen campaign diagnostic; coefficients are read, never refitted."""

    figure = "fig3_campaign_diagnostic_envelope"
    cells_all = read_csv("a14_envelope_cells.csv", figure)
    summary = read_json("a14_v41_envelope_summary.json", figure)
    skill_table = read_csv("a14_v41_envelope_interfered_holdout.csv", figure)

    cells = cells_all[cells_all["split"] != "clean_retention"].copy()
    EXCLUSIONS[figure] = {
        "before": int(len(cells_all)),
        "after": int(len(cells)),
        "rule": "split != clean_retention",
        "reason": "A jammer-free window has no physical finite SIR; this is the locked V4.1 envelope definition.",
    }
    fit_mask = cells["split"].isin(["id_test", "hard_interference"])
    held_mask = ~fit_mask
    if int(fit_mask.sum()) != 32 or int(held_mask.sum()) != 80:
        raise ValueError("Envelope cell contract drifted from 32 fit / 80 held cells")

    iq = summary["references"]["IQFormer"]
    coef = iq["two_variable_coefficients"]
    intercept = float(coef["intercept_pp"])
    overlap_slope = float(coef["overlap_pp_per_unit"])
    sir_slope = float(coef["sir_pp_per_db"])
    predicted = intercept + overlap_slope * cells["occupancy_mean"].to_numpy() + sir_slope * cells["sir_db"].to_numpy()
    observed = 100.0 * cells["gain_vs_IQFormer"].to_numpy()
    fit_mean = float(iq["fit_mean_gain_pp"])

    pooled = skill_table[
        (skill_table["reference"] == "IQFormer")
        & (skill_table["regime"] == "pooled_interfered_held")
    ].iloc[0]

    groups = {
        "id_test": ("o", A5_BLUE, True, "In distribution"),
        "hard_interference": ("s", A5_BLUE, True, "Hard interference"),
        "unseen_speed": ("v", MC_ORANGE, False, "Unseen speed"),
        "heldout_channel": ("D", MC_ORANGE, False, "Held-out channel"),
        "unseen_jammer": ("^", IQ_GREEN, False, "Unseen jammer"),
        "combined_ood": ("P", IQ_GREEN, False, "Combined OOD"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.92))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.90, wspace=0.24)

    ax = axes[0]
    grid = np.linspace(0.0, 1.0, 100)
    line_styles = [(-15.0, "-"), (-5.0, "--"), (0.0, ":")]
    for sir, linestyle in line_styles:
        line = intercept + overlap_slope * grid + sir_slope * sir
        ax.plot(grid, line, color=NEUTRAL, linestyle=linestyle, linewidth=1.15, zorder=1)
        x_label = 0.62
        y_label = intercept + overlap_slope * x_label + sir_slope * sir
        ax.text(
            x_label,
            y_label + 0.8,
            f"SIR {sir:.0f} dB",
            color=NEUTRAL,
            fontsize=7.2,
            ha="center",
            va="bottom",
        )

    for split, (marker, color, filled, _) in groups.items():
        subset = cells[cells["split"] == split]
        ax.scatter(
            subset["occupancy_mean"],
            100.0 * subset["gain_vs_IQFormer"],
            marker=marker,
            s=25,
            facecolors=color if filled else "none",
            edgecolors=color,
            linewidths=0.9,
            zorder=3,
        )
    ax.axhline(0.0, color=INK, linewidth=0.75, zorder=0)
    ax.set_xlim(-0.035, 1.035)
    ax.set_xlabel("Target-support jammer overlap, $o$")
    ax.set_ylabel("A5 - IQFormer-inspired macro-F1 (pp)")
    ax.set_title("Frozen-campaign diagnostic surface", loc="left", pad=4)
    add_panel_label(ax, "(a)")
    style_numeric_axis(ax, y_grid=True)

    ax = axes[1]
    held_pred = predicted[held_mask.to_numpy()]
    held_obs = observed[held_mask.to_numpy()]
    low = min(float(held_pred.min()), float(held_obs.min())) - 1.5
    high = max(float(held_pred.max()), float(held_obs.max())) + 1.5
    ax.plot([low, high], [low, high], color=INK, linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(fit_mean, color=NEUTRAL, linestyle=":", linewidth=1.0, zorder=1)
    for split, (marker, color, _, _) in groups.items():
        select = (cells["split"] == split).to_numpy() & held_mask.to_numpy()
        if select.any():
            ax.scatter(
                predicted[select],
                observed[select],
                marker=marker,
                s=25,
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                zorder=3,
            )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Predicted gain (pp)")
    ax.set_ylabel("Observed gain (pp)")
    ax.set_title("Held-regime diagnostic", loc="left", pad=4)
    add_panel_label(ax, "(b)")
    style_numeric_axis(ax, y_grid=True)
    ax.text(
        0.04,
        0.96,
        f"Envelope RMSE  {pooled.rmse_pp:.2f} pp\n"
        f"Fit-domain constant  {pooled.constant_rmse_pp:.2f} pp\n"
        f"R² skill  {pooled.r_squared_skill:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.28,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": WHITE, "edgecolor": GRID, "linewidth": 0.6},
    )

    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor=color if filled else "none",
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=5.0,
            label=label,
        )
        for marker, color, filled, label in groups.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.53, 0.06),
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0.0,
        fontsize=7.0,
    )
    save_figure(fig, figure)


def fig4_complexity_and_representation() -> None:
    """Aligned performance and parameter view on the matched five-seed subset."""

    figure = "fig4_complexity_and_representation"
    frame = read_csv("a15_v41_pareto_matched.csv", figure)
    if set(frame["seed_count"]) != {5}:
        raise ValueError("Fig. 4 requires the matched five-seed subset")

    key_styles = {
        "a0_backbone": (A5_LIGHT, "o", True, 28),
        "a5_vimd_full": (A5_BLUE, "o", True, 42),
        "tier2_h2_f2_iq_sidecar": (SIDECAR_PURPLE, "D", False, 46),
        "iqformer_inspired": (IQ_GREEN, "s", True, 34),
        "mcldnn_reimplementation": (MC_ORANGE, "^", True, 34),
        "cssl_amc_supervised_adaptation": (NEUTRAL, "v", True, 32),
    }

    order = [
        "a0_backbone",
        "a1_single_mask",
        "a2_tri_no_teacher",
        "a3_tri_teacher",
        "a3p_tri_proportional_teacher",
        "a4_tri_teacher_mtl",
        "a5_vimd_full",
        "a6_dual_full",
        "a7_vimd_no_residual",
        "tier2_h2_f2_iq_sidecar",
        "mcldnn_reimplementation",
        "iqformer_inspired",
        "cssl_amc_supervised_adaptation",
    ]
    display = {
        "a0_backbone": "A0",
        "a1_single_mask": "A1",
        "a2_tri_no_teacher": "A2",
        "a3_tri_teacher": "A3",
        "a3p_tri_proportional_teacher": "A3-P",
        "a4_tri_teacher_mtl": "A4",
        "a5_vimd_full": "A5",
        "a6_dual_full": "A6",
        "a7_vimd_no_residual": "A7",
        "tier2_h2_f2_iq_sidecar": "Received-I/Q sidecar",
        "mcldnn_reimplementation": "MCLDNN",
        "iqformer_inspired": "IQFormer-inspired",
        "cssl_amc_supervised_adaptation": "CSSL",
    }
    if set(frame["model"]) != set(order):
        raise ValueError("Fig. 3 model set changed; update the declared row order")
    rows = frame.set_index("model").loc[order].reset_index()

    fig = plt.figure(figsize=(3.5, 3.28))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.05], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_params = fig.add_subplot(grid[0, 1], sharey=ax)
    fig.subplots_adjust(left=0.34, right=0.985, bottom=0.13, top=0.93)
    y = np.arange(len(rows))

    for idx in range(len(rows)):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color="#F5F7F8", zorder=0)
            ax_params.axhspan(idx - 0.5, idx + 0.5, color="#F5F7F8", zorder=0)
    ax.axhline(8.5, color=GRID, linewidth=0.8, zorder=1)
    ax_params.axhline(8.5, color=GRID, linewidth=0.8, zorder=1)

    for idx, row in enumerate(rows.itertuples(index=False)):
        if row.model in key_styles:
            color, marker, filled, size = key_styles[row.model]
        else:
            color, marker, filled, size = A5_LIGHT, "o", True, 24
        ax.errorbar(
            row.macro_f1_mean,
            idx,
            xerr=row.macro_f1_std,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.8,
            zorder=3,
        )
        ax.scatter(
            row.macro_f1_mean,
            idx,
            marker=marker,
            s=size,
            facecolors=color if filled else WHITE,
            edgecolors=color,
            linewidths=1.0,
            zorder=4,
        )

        params = float(row.parameters)
        param_label = f"{params / 1_000_000:.2g}M" if params >= 1_000_000 else f"{params / 1_000:.1f}k"
        ax_params.text(0.5, idx, param_label, ha="center", va="center", fontsize=7.0)

    ax.set_yticks(y)
    ax.set_yticklabels([display[name] for name in order], fontsize=7.0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.38, 0.522)
    ax.set_xticks([0.40, 0.44, 0.48, 0.52])
    ax.set_xlabel("Hard-interference macro-F1")
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(direction="out", length=2.5, width=0.8, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax_params.set_xlim(0, 1)
    ax_params.set_xticks([])
    ax_params.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax_params.spines.values():
        spine.set_visible(False)
    ax_params.set_title("Parameters", fontsize=7.4, pad=5)
    save_figure(fig, figure)


def fig5_clean_condition_transfer() -> None:
    """Family-level clean condition-transfer taxonomy."""

    figure = "fig5_clean_condition_transfer"
    frame = read_csv("a13_condition_transfer.csv", figure)
    modulations = [
        "BPSK",
        "PI2BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "64QAM",
        "256QAM",
        "GMSK",
        "CPFSK",
        "4FSK",
    ]
    display = {"PI2BPSK": r"$\pi$/2-BPSK"}
    trained_clean = {"PI2BPSK", "8PSK", "256QAM", "CPFSK"}
    families = [
        ("compact_spectral_A0_A7", "(a) Compact spectral family"),
        ("iq_domain_high_capacity", "(b) I/Q-domain baselines"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.72), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.31, top=0.84, wspace=0.11)
    x = np.arange(len(modulations))
    width = 0.38

    for panel, (ax, (family, title)) in enumerate(zip(axes, families)):
        subset = frame[frame["model_family"] == family]
        aggregate = subset.groupby("modulation", as_index=True)[["jammed_f1", "clean_f1"]].mean()
        aggregate = aggregate.reindex(modulations)
        if aggregate.isna().any().any():
            raise ValueError(f"Missing modulation rows for {family}")
        ax.bar(
            x - width / 2,
            aggregate["jammed_f1"],
            width,
            color=MC_ORANGE,
            edgecolor=MC_ORANGE,
            linewidth=0.5,
            hatch="///",
            label="Hard interference",
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            aggregate["clean_f1"],
            width,
            color=A5_BLUE,
            edgecolor=A5_BLUE,
            linewidth=0.5,
            label="Jammer-free",
            zorder=2,
        )
        labels = [
            display.get(name, name) + ("*" if name not in trained_clean else "")
            for name in modulations
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_title(title, loc="left", pad=4, fontsize=7.2, fontweight="bold")
        ax.set_ylim(0.0, 0.98)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        style_numeric_axis(ax, y_grid=True)

    axes[0].set_ylabel("Per-class F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.99),
        ncol=2,
        fontsize=7.3,
        handlelength=1.6,
        columnspacing=1.2,
    )
    save_figure(fig, figure)


def write_provenance() -> None:
    record = {
        "schema": "tvt_paper_figures_v46",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "python/matplotlib",
        "geometry": {
            "single_column_in": SINGLE_IN,
            "single_column_mm": SINGLE_IN * MM_PER_IN,
            "double_column_in": DOUBLE_IN,
            "double_column_mm": DOUBLE_IN * MM_PER_IN,
        },
        "policy": "Presentation-only redraw from frozen analysis outputs; no training, new data, or envelope refit.",
        "figures": {},
        "exclusions": EXCLUSIONS,
    }
    for figure, paths in USED.items():
        record["figures"][figure] = [
            {
                "source": str(path.relative_to(REPO)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(set(paths))
        ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance_v46.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    fig2_rank_instability()
    fig3_campaign_diagnostic_envelope()
    fig4_complexity_and_representation()
    fig5_clean_condition_transfer()
    write_provenance()
    for path in sorted(OUT.glob("*.pdf")):
        print(path.name)


if __name__ == "__main__":
    main()
