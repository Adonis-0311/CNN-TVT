"""V1: rebuild Fig 5 (pareto) to include Tier-2 sidecar and capacity tiers.

Sealed points are read from analysis_zero_compute/outputs/csv/a8_pareto.csv
(hard_interference). The Tier-2 sidecar (46,794 params, 48.43 macro-F1) and
capacity tiers M (99,596 / 45.54) and L (233,244 / 46.78) are added from the
V1 artifact audit values and drawn as open markers labeled exploratory.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "analysis_zero_compute" / "outputs" / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
SINGLE = 3.5

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

frame = pd.read_csv(CSV / "a8_pareto.csv")
subset = frame[frame.regime == "hard_interference"]

fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
for _, row in subset.iterrows():
    proposed = row.model == "a5_vimd_full"
    ax.scatter(row.parameters, row.macro_f1_mean,
               marker="*" if row.pareto_optimal_parameters else "o",
               s=70 if proposed else 22,
               c="#d62728" if proposed else "#1f77b4")
    label = {"IQFormer": "IQFormer-insp."}.get(row.model_short, row.model_short)
    if row.model_short in ("A0", "A5-VIMD", "CSSL", "MCLDNN", "IQFormer"):
        ax.annotate(label, (row.parameters, row.macro_f1_mean),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")

# Tier-2 exploratory points (open markers): sidecar uses ten seeds; the
# capacity tiers are evaluated on the five-seed matched subset (Table VIII).
tier2 = [
    ("sidecar", 46794, 0.4843, "s"),
    ("S-tier", 39500, 0.4360, "o"),
    ("M-tier", 99596, 0.4554, "^"),
    ("L-tier", 233244, 0.4678, "v"),
]
for name, params, f1, marker in tier2:
    ax.scatter(params, f1, marker=marker, s=46, facecolors="none",
               edgecolors="#7f7f7f", linewidths=1.1)
    ax.annotate(name, (params, f1), fontsize=5.5,
                xytext=(4, -8), textcoords="offset points")
ax.text(
    0.03, 0.03,
    "capacity tiers: five-seed matched subset",
    transform=ax.transAxes, fontsize=5.5, ha="left", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="0.7", linewidth=0.5),
)

ax.set_xscale("log")
ax.set_xlabel("parameters")
ax.set_ylabel("hard-interference macro-F1")
ax.set_title("sealed comparison set (filled) and prospective Tier-2 controls (open)",
             loc="left", fontsize=6.5)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig4_pareto.pdf")
fig.savefig(OUT / "fig4_pareto.png", dpi=300)
plt.close(fig)
print("rebuilt fig4_pareto.pdf / fig4_pareto.png")
