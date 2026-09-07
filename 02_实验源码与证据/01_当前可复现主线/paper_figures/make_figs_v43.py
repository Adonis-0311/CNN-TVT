"""V4.3 redraw of Fig. 3 (overlap-SIR envelope) and Fig. 4 (complexity frontier).

Both figures are rendered from the same frozen CSV files the manuscript cites,
and Fig. 3 recomputes its own annotated statistics from the arrays it plots so
the panel cannot drift from the text.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.lines import Line2D

CSV = Path("csv")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SINGLE, DOUBLE = 3.5, 7.16

# Validated 3-slot categorical palette (all-pairs, light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#22221f", "#6b6a63", "#d8d7d2"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.0, "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Split -> (marker, colour, filled, group label)
GROUPS = {
    "id_test":           ("o", BLUE,   True,  "fit"),
    "hard_interference": ("s", BLUE,   True,  "fit"),
    "unseen_speed":      ("v", ORANGE, False, "held: channel-side shift"),
    "heldout_channel":   ("D", ORANGE, False, "held: channel-side shift"),
    "unseen_jammer":     ("^", AQUA,   False, "held: interference-side shift"),
    "combined_ood":      ("P", AQUA,   False, "held: interference-side shift"),
}
NICE = {"id_test": "in distribution", "hard_interference": "hard interference",
        "unseen_speed": "unseen speed", "heldout_channel": "held-out channel",
        "unseen_jammer": "unseen jammer", "combined_ood": "combined OOD"}


def draw(ax, x, y, split, size=17):
    m, c, filled, _ = GROUPS[split]
    ax.scatter(x, y, marker=m, s=size,
               facecolors=c if filled else "none", edgecolors=c,
               linewidths=0.9, zorder=3,
               path_effects=None)


def fig_envelope():
    cells = pd.read_csv(CSV / "a14_envelope_cells.csv")
    cells = cells[cells.split != "clean_retention"].copy()
    fit = pd.read_csv(CSV / "a14_envelope_fit.csv")
    row = fit[fit.reference == "iqformer_inspired"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.85))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.34, wspace=0.24)

    # ---------------- panel (a): gain surface vs overlap ----------------
    ax = axes[0]
    grid = np.linspace(0, 1, 60)
    for sir, style in ((-15.0, "-"), (-5.0, "--"), (0.0, ":")):
        ax.plot(grid,
                row.intercept_pp + row.occupancy_slope_pp_per_unit * grid
                + row.sir_slope_pp_per_db * sir,
                style, color=MUTED, lw=0.9, zorder=2)
        xl = 0.55
        yl = row.intercept_pp + row.occupancy_slope_pp_per_unit * xl + row.sir_slope_pp_per_db * sir
        ax.annotate(f"SIR $-{abs(sir):.0f}$ dB" if sir < 0 else "SIR 0 dB",
                    (xl, yl), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.8, color=MUTED, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))
    for split in GROUPS:
        s = cells[cells.split == split]
        if not s.empty:
            draw(ax, s.occupancy_mean, 100 * s.gain_vs_IQFormer, split)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel("target-support jammer overlap $o$")
    ax.set_ylabel(r"$\Delta$ macro-F1, A5 $-$ IQFormer-insp. (pp)")
    ax.set_title("(a) gain surface over the fitted plane", loc="left", color=INK)

    # ---------------- panel (b): held-regime prediction ----------------
    ax = axes[1]
    design = np.column_stack([np.ones(len(cells)), cells.occupancy_mean.to_numpy(),
                              cells.sir_db.to_numpy()])
    mask = cells.split.isin(["id_test", "hard_interference"]).to_numpy()
    coef, *_ = np.linalg.lstsq(design[mask], cells.gain_vs_IQFormer.to_numpy()[mask], rcond=None)
    pred = 100 * (design @ coef)
    act = 100 * cells.gain_vs_IQFormer.to_numpy()
    const = float(act[mask].mean()); held = ~mask
    rmse = float(np.sqrt(np.mean((act[held] - pred[held]) ** 2)))
    rmse_c = float(np.sqrt(np.mean((act[held] - const) ** 2)))
    rmse_o = float(np.sqrt(np.mean((act[held] - act[held].mean()) ** 2)))
    skill = 1.0 - rmse ** 2 / rmse_c ** 2

    lim = [min(pred.min(), act.min()) - 2.5, max(pred.max(), act.max()) + 2.5]
    ax.plot(lim, lim, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.annotate("identity", (lim[1], lim[1]), xytext=(-3, -8), textcoords="offset points",
                ha="right", va="top", fontsize=5.8, color=MUTED, rotation=45, zorder=4)
    ax.axvline(const, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.annotate("fit-domain constant", (const, lim[1]), xytext=(3, -3),
                textcoords="offset points", ha="left", va="top",
                fontsize=5.8, color=MUTED, zorder=4,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))
    for split in GROUPS:
        sel = (cells.split == split).to_numpy() & held
        if sel.any():
            draw(ax, pred[sel], act[sel], split)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted gain (pp), envelope fitted on the 32 fit cells")
    ax.set_ylabel("observed gain (pp)")
    ax.set_title("(b) transport to the 80 held-regime cells", loc="left", color=INK)
    ax.text(0.035, 0.965,
            "RMSE {:.2f} pp\nfit-domain constant {:.2f} pp\noracle constant {:.2f} pp\n"
            r"$R^2_{{\mathrm{{skill}}}}$ {:.2f}".format(rmse, rmse_c, rmse_o, skill),
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID, linewidth=0.5), zorder=5)

    # ---------------- shared legend below both panels ----------------
    handles = [Line2D([], [], marker=GROUPS[s][0], linestyle="none",
                      markerfacecolor=GROUPS[s][1] if GROUPS[s][2] else "none",
                      markeredgecolor=GROUPS[s][1], markeredgewidth=0.9,
                      markersize=4.0, label=NICE[s]) for s in GROUPS]
    leg = fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                     bbox_to_anchor=(0.5, 0.135), handletextpad=0.35,
                     columnspacing=1.1, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.5, 0.055,
             "filled = cells used to fit the envelope   ·   open = held-regime cells   ·   "
             "blue = fit,  orange = channel-side shift,  green = interference-side shift",
             ha="center", va="bottom", fontsize=5.9, color=MUTED)

    fig.savefig(OUT / "fig1_envelope_law.pdf")
    fig.savefig(OUT / "fig1_envelope_law.png", dpi=320)
    plt.close(fig)
    print(f"fig3 audit: const={const:.4f} rmse={rmse:.4f} const_rmse={rmse_c:.4f} "
          f"oracle={rmse_o:.4f} skill={skill:.4f}")


def fig_pareto():
    d = pd.read_csv(CSV / "a15_v41_pareto_matched.csv")
    family = ["a1_single_mask", "a2_tri_no_teacher", "a3_tri_teacher",
              "a3p_tri_proportional_teacher", "a4_tri_teacher_mtl",
              "a6_dual_full", "a7_vimd_no_residual"]
    labelled = {"a0_backbone": ("A0", 8, 0, "left"),
                "a5_vimd_full": ("A5 / VIMD-Net", 10, 0, "left"),
                "tier2_h2_f2_iq_sidecar": ("I/Q sidecar", -9, 0, "right"),
                "mcldnn_reimplementation": ("MCLDNN", 9, 0, "left"),
                "iqformer_inspired": ("IQFormer-insp.", 0, 21, "center"),
                "cssl_amc_supervised_adaptation": ("CSSL", -8, 0, "right")}

    fig, ax = plt.subplots(figsize=(SINGLE, 2.45))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.965, bottom=0.185)

    def pt(r, colour, filled, size, z, ring=False):
        ax.errorbar(r.parameters, r.macro_f1_mean, yerr=r.macro_f1_std,
                    fmt="none", ecolor=colour, elinewidth=0.7, capsize=1.5,
                    capthick=0.7, alpha=0.8, zorder=z)
        ax.scatter(r.parameters, r.macro_f1_mean, s=size,
                   marker="o" if filled else "s",
                   facecolors=colour if filled else "none", edgecolors=colour,
                   linewidths=1.0, zorder=z + 1)
        if ring:
            ax.scatter(r.parameters, r.macro_f1_mean, s=size + 42, marker="o",
                       facecolors="none", edgecolors="white", linewidths=1.6,
                       zorder=z + 0.5)

    for r in d.itertuples(index=False):
        if r.model in family:
            pt(r, BLUE, True, 8, 3)
    for r in d.itertuples(index=False):
        if r.model == "tier2_h2_f2_iq_sidecar":
            pt(r, ORANGE, False, 32, 6)
        elif r.model == "a5_vimd_full":
            pt(r, BLUE, True, 30, 6, ring=True)
        elif r.model == "a0_backbone":
            pt(r, BLUE, True, 26, 5)
        elif r.model not in family:
            pt(r, MUTED, True, 24, 4)

    for r in d.itertuples(index=False):
        if r.model in labelled:
            txt, dx, dy, ha = labelled[r.model]
            ax.annotate(txt, (r.parameters, r.macro_f1_mean), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=6.1, color=INK, zorder=8, linespacing=1.15)
    fam = d[d.model.isin(family)]
    ax.annotate("A1–A7", (fam.parameters.min(), fam.macro_f1_mean.max()),
                xytext=(-4, 7), textcoords="offset points", ha="right", va="bottom",
                fontsize=5.9, color=MUTED, zorder=8)

    ax.set_xscale("log")
    ax.set_xlim(5.0e3, 3.4e7)
    ax.set_ylim(0.376, 0.534)
    ax.set_xlabel("parameters")
    ax.set_ylabel("hard-interference macro-F1")

    fig.savefig(OUT / "fig4_pareto.pdf")
    fig.savefig(OUT / "fig4_pareto.png", dpi=320)
    plt.close(fig)
    print("fig4 points:", len(d))


if __name__ == "__main__":
    fig_envelope()
    fig_pareto()
