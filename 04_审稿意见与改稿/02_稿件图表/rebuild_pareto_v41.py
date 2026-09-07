"""Render V4.1 Fig. 4 from a five-seed matched, artifact-derived CSV."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
frame = pd.read_csv(REPO / "analysis_zero_compute" / "outputs" / "csv" / "a15_v41_pareto_matched.csv")
out = Path(__file__).resolve().parent / "outputs"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 200, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .02})
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for row in frame.itertuples(index=False):
    sidecar = row.evidence_class.startswith("exploratory")
    proposed = row.model == "a5_vimd_full"
    ax.scatter(row.parameters, row.macro_f1_mean, marker="s" if sidecar else ("*" if proposed else "o"), s=48 if (sidecar or proposed) else 22, facecolors="none" if sidecar else None, edgecolors="#7f7f7f" if sidecar else None, c="#d62728" if proposed else "#1f77b4")
    if row.model_short in {"A0", "A5-VIMD", "CSSL", "MCLDNN", "IQFormer", "I/Q sidecar"}:
        label = "IQFormer-insp." if row.model_short == "IQFormer" else row.model_short
        ax.annotate(label, (row.parameters, row.macro_f1_mean), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("hard-interference macro-F1")
ax.set_title("five-seed matched comparison (sidecar open)", loc="left", fontsize=6.5)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig4_pareto.pdf"); fig.savefig(out / "fig4_pareto.png", dpi=300)
plt.close(fig)
