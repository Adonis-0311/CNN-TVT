"""V4.4 redraw of Fig. 5 (clean condition-transfer taxonomy).

Fixes the V4.3 figure/text conflict: only the six modulation classes whose
jammer-free condition was never seen during training carry a star (BPSK,
QPSK, 16QAM, 64QAM, GMSK, 4FSK); the four covered classes (PI/2-BPSK, 8PSK,
256QAM, CPFSK) are unstarred.  The figure keeps the V4.3 two-panel layout --
panel (a) the compact spectral family aggregate, panel (b) the I/Q-domain
high-capacity baselines -- and renames PI2BPSK to the manuscript's
pi/2-BPSK spelling on the axis.

Rendered from the same frozen CSV the manuscript cites; nothing is
recomputed beyond family averaging already present in the source table.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "analysis_zero_compute" / "outputs" / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
SINGLE = 3.5

MODULATIONS = ["BPSK", "PI2BPSK", "QPSK", "8PSK", "16QAM",
               "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK"]
DISPLAY = {"PI2BPSK": r"$\pi/2$-BPSK"}
TRAINED_CLEAN = {"PI2BPSK", "8PSK", "256QAM", "CPFSK"}
FAMILIES = {
    "compact_spectral_A0_A7": "(a) compact spectral family (A0--A7)",
    "iq_domain_high_capacity": "(b) I/Q-domain baselines",
}

RED, BLUE = "#d62728", "#1f77b4"

plt.rcParams.update({
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 5.9,
    "legend.fontsize": 5.8, "xtick.labelsize": 5.9, "ytick.labelsize": 6.0,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def star_label(name: str) -> str:
    label = DISPLAY.get(name, name)
    return label if name in TRAINED_CLEAN else label + "*"


def fig_condition_transfer() -> None:
    frame = pd.read_csv(CSV / "a13_condition_transfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 2.6), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.99, wspace=0.12, top=0.90, bottom=0.28)
    for ax, (family, title) in zip(axes, FAMILIES.items()):
        subset = frame[frame.model_family == family]
        pivot = subset.pivot_table(index="modulation",
                                   values=["jammed_f1", "clean_f1"], aggfunc="mean")
        pivot = pivot.reindex(MODULATIONS)
        x = np.arange(len(pivot))
        ax.bar(x - 0.2, pivot.jammed_f1, 0.4, label="hard interference", color=RED)
        ax.bar(x + 0.2, pivot.clean_f1, 0.4, label="jammer-free", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels([star_label(m) for m in pivot.index],
                           rotation=45, ha="right")
        ax.set_title(title, loc="left", pad=4)
        if family == "compact_spectral_A0_A7":
            ax.set_ylabel("per-class F1")
            ax.legend(frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig6_condition_transfer.pdf")
    fig.savefig(OUT / "fig6_condition_transfer.png", dpi=300)
    plt.close(fig)
    starred = [m for m in MODULATIONS if m not in TRAINED_CLEAN]
    print("fig5 stars:", ", ".join(starred))


if __name__ == "__main__":
    fig_condition_transfer()
