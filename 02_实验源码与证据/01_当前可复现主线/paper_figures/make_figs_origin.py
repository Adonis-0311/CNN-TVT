"""Origin-style unified redraw of the manuscript figures.

Purpose
-------
Presentation only.  Every quantitative mark is read from the frozen
analysis CSV/JSON layer or from the sealed cache record; nothing here refits
the envelope, retrains a model, or edits an experiment artifact.  The script
exists so that all main-manuscript figures share one visual system.

House style (OriginLab scientific convention)
---------------------------------------------
* Arial throughout, one type scale for every panel.
* Full black box frame on all four sides.
* Ticks point inward, mirrored on the top and right axes, with minor ticks.
* No background grid.
* Framed legends with a thin black rule.
* Solid marker fills carrying a darker edge.
* One validated three-hue categorical palette plus neutral black/gray.

Outputs are written to ``outputs_origin/`` and never overwrite the existing
figure sets.
"""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "analysis_zero_compute" / "outputs" / "csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs_origin"
DEFAULT_TEACHER = REPO / "teacher_record_v43.npz"

SINGLE_IN = 3.5
DOUBLE_IN = 7.16

# --------------------------------------------------------------------------
# Palette.  Three categorical slots, validated all-pairs on a light surface
# (worst CVD dE 11.0, worst normal-vision dE 18.7, all >= 3:1 contrast), plus
# neutral ink and gray for reference geometry.
# --------------------------------------------------------------------------
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
INK = "#000000"
GRAY = "#5A5A5A"
GRAY_LIGHT = "#B4B4B4"
WHITE = "#FFFFFF"

# Diverging map for the signed gain heat map: vermilion (reference favoured)
# through white to blue (A5 favoured).
GAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "origin_gain", [VERMILION, "#FBFBFB", BLUE], N=256
)
# Single-hue sequential ramps for the spectrogram panels.
POWER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "origin_power", ["#FFFFFF", "#CFE3F0", "#7FB2D4", "#2E7FB2", "#0B4A73", "#05263B"], N=256
)
ALLOC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "origin_alloc", ["#FFFFFF", "#F6DCC7", "#EDA871", "#D55E00", "#8F3F00", "#4A2100"], N=256
)

BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.6,
    "ytick.major.size": 3.6,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": INK,
    "legend.fancybox": False,
    "legend.borderpad": 0.35,
    "legend.handletextpad": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.4,
    "lines.markeredgewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 180,
}
matplotlib.rcParams.update(BASE_RC)

USED: dict[str, list[str]] = {}


# --------------------------------------------------------------------------
# Style helpers
# --------------------------------------------------------------------------
def origin_frame(ax: plt.Axes, *, minor_x: bool = True, minor_y: bool = True) -> None:
    """Apply the Origin box frame: four black spines, inward mirrored ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(INK)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   color=INK, pad=2.6)
    ax.tick_params(which="major", length=3.6, width=0.9)
    ax.tick_params(which="minor", length=2.0, width=0.7)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, *, x: float = 0.028, y: float = 0.972,
                outside: bool = False) -> None:
    """Bold panel key, Origin convention.

    ``outside`` places the key just above the frame, which is the right choice
    for filled image panels where an inset box would cover a data cell.
    """
    if outside:
        ax.text(0.0, 1.012, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.6, fontweight="bold", color=INK, clip_on=False)
        return
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.6, fontweight="bold", color=INK, zorder=10,
            bbox={"boxstyle": "square,pad=0.16", "facecolor": WHITE,
                  "edgecolor": "none", "alpha": 0.82})


def origin_legend(target, handles, labels=None, **kwargs):
    """Framed legend with a thin black rule."""
    legend = target.legend(handles=handles, labels=labels, **kwargs) if labels \
        else target.legend(handles=handles, **kwargs)
    frame = legend.get_frame()
    frame.set_linewidth(0.7)
    frame.set_edgecolor(INK)
    frame.set_facecolor(WHITE)
    return legend


def save_figure(fig: plt.Figure, stem: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.svg")
    fig.savefig(out / f"{stem}.png", dpi=600)
    fig.savefig(out / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {stem}.{{pdf,svg,png,tiff}}")


def read_csv(csv_dir: Path, name: str, figure: str) -> pd.DataFrame:
    USED.setdefault(figure, []).append(name)
    return pd.read_csv(csv_dir / name)


def read_json(csv_dir: Path, name: str, figure: str) -> dict:
    USED.setdefault(figure, []).append(name)
    return json.loads((csv_dir / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fig. 1 -- simulation-component teacher on the 64x61 front-end lattice
# --------------------------------------------------------------------------
MODULATIONS = ("BPSK", r"$\pi$/2-BPSK", "QPSK", "8PSK", "16QAM",
               "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK")
PROFILES = ("TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E")
JAMMERS = ("tone", "multitone", "chirp", "sweep", "pulse",
           "partial band", "comb", "cochannel", "OFDM-like")
NFFT, HOP = 64, 16


def _stft(iq: np.ndarray) -> np.ndarray:
    """Two-sided Hann STFT matching the model front end (n_fft 64, hop 16)."""
    x = iq[0] + 1j * iq[1]
    window = np.hanning(NFFT + 1)[:-1]  # periodic Hann
    frames = (len(x) - NFFT) // HOP + 1
    out = np.empty((NFFT, frames), dtype=complex)
    for t in range(frames):
        out[:, t] = np.fft.fft(x[t * HOP:t * HOP + NFFT] * window) / np.sqrt(NFFT)
    return out


def _power_db(spectrum: np.ndarray) -> np.ndarray:
    power = np.abs(spectrum) ** 2
    reference = max(float(power.max()), np.finfo(np.float32).tiny)
    return np.fft.fftshift(10.0 * np.log10(np.maximum(power / reference, 1e-8)), axes=0)


def fig1_teacher(teacher_npz: Path, out: Path) -> None:
    figure = "fig1_component_teacher"
    USED.setdefault(figure, []).append(teacher_npz.name)
    data = np.load(teacher_npz)
    meta = json.loads(teacher_npz.with_suffix(".json").read_text(encoding="utf-8"))

    spectra = {name: _stft(data[name]) for name in ("x", "clean", "jammer", "unexplained")}
    ps, pj, pu = (np.abs(spectra[n]) ** 2 for n in ("clean", "jammer", "unexplained"))
    total = ps + pj + pu
    eps = max(1e-8 * float(total.mean()), np.finfo(np.float32).tiny)
    den = np.maximum(total, eps)
    qs, qj, qu = ps / den, pj / den, pu / den
    empty = total <= eps
    masks = np.stack([np.maximum(qs - qj, 0.0), np.maximum(qj - qs, 0.0),
                      qu + 2.0 * np.minimum(qs, qj)], axis=0)
    masks[0][empty] = 0.0
    masks[1][empty] = 0.0
    masks[2][empty] = 1.0
    masks = masks / np.maximum(masks.sum(axis=0, keepdims=True), 1e-8)
    masks = np.fft.fftshift(masks, axes=1)

    bins, frames = spectra["x"].shape
    if (bins, frames) != (64, 61):
        raise ValueError(f"lattice drifted: {bins}x{frames}")

    fig = plt.figure(figsize=(DOUBLE_IN, 3.05))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.02, 0.062],
                            wspace=0.20, hspace=0.62)
    extent = (0, frames - 1, -0.5, 0.5)
    ticks = [0, 20, 40, 60]

    top_titles = (("x", "Received mixture"), ("clean", "Tracked target"),
                  ("jammer", "Tracked jammer"), ("unexplained", "Noise + artifact"))
    image = None
    for col, (name, title) in enumerate(top_titles):
        ax = fig.add_subplot(grid[0, col])
        image = ax.imshow(_power_db(spectra[name]), origin="lower", aspect="auto",
                          cmap=POWER_CMAP, vmin=-70.0, vmax=0.0, extent=extent)
        ax.set_title(title, pad=3.5, fontsize=7.8)
        ax.set_xticks(ticks)
        ax.set_yticks([-0.5, 0.0, 0.5])
        origin_frame(ax)
        panel_label(ax, "(%s)" % "abcd"[col])
        if col == 0:
            ax.set_ylabel("Norm. frequency")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("STFT frame")
    cax = fig.add_subplot(grid[0, 4])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Relative power (dB)", fontsize=7.4)
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=7.0, direction="in", length=2.6, width=0.8)
    bar.outline.set_linewidth(0.9)
    bar.outline.set_edgecolor(INK)

    route_titles = (r"Target dominant, $M_s^\star$",
                    r"Jammer dominant, $M_j^\star$",
                    r"Ambiguous, $M_o^\star$")
    mask_image = None
    for col, title in enumerate(route_titles):
        ax = fig.add_subplot(grid[1, col])
        mask_image = ax.imshow(masks[col], origin="lower", aspect="auto",
                               cmap=ALLOC_CMAP, vmin=0.0, vmax=1.0, extent=extent)
        ax.set_title(title, pad=3.5, fontsize=7.8)
        ax.set_xticks(ticks)
        ax.set_yticks([-0.5, 0.0, 0.5])
        origin_frame(ax)
        panel_label(ax, "(%s)" % "efg"[col])
        if col == 0:
            ax.set_ylabel("Norm. frequency")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("STFT frame")
    cax2 = fig.add_subplot(grid[1, 4])
    bar2 = fig.colorbar(mask_image, cax=cax2)
    bar2.set_label("Teacher allocation", fontsize=7.4)
    bar2.ax.minorticks_off()
    bar2.ax.tick_params(labelsize=7.0, direction="in", length=2.6, width=0.8)
    bar2.outline.set_linewidth(0.9)
    bar2.outline.set_edgecolor(INK)

    active = [JAMMERS[k] for k in np.flatnonzero(np.asarray(meta["jam_labels"]) > 0.5)]
    note = fig.add_subplot(grid[1, 3])
    note.axis("off")
    note.text(0.0, 0.98, "\n".join((
        "Sealed cache record",
        f"split: {meta['split'].replace('_', ' ')}",
        f"source {meta['index']}, view {meta['view']}",
        f"modulation: {MODULATIONS[meta['label']]}",
        f"jammer: {'+'.join(active) if active else 'none'}",
        f"SNR / SIR: {meta['snr_db']:.0f} / {meta['sir_db']:.0f} dB",
        f"TDL: {PROFILES[meta['target_profile_index']]} / "
        f"{PROFILES[meta['jammer_profile_index']]}",
        f"lattice: {bins} x {frames}",
        f"(NFFT {NFFT}, hop {HOP})",
        "",
        "Training-only supervision;",
        "no learned prediction shown.",
    )), ha="left", va="top", fontsize=6.6, linespacing=1.40, color=INK,
        transform=note.transAxes)

    save_figure(fig, figure, out)


# --------------------------------------------------------------------------
# Fig. 2 -- campaign-internal rank instability
# --------------------------------------------------------------------------
def fig2_rank_instability(csv_dir: Path, out: Path) -> None:
    figure = "fig2_rank_instability"
    frame = read_csv(csv_dir, "a1b_snr_sir_map.csv", figure)
    frame = frame[frame["regime"] == "hard_interference"].copy()
    references = [("mcldnn_reimplementation", "A5 $-$ MCLDNN"),
                  ("iqformer_inspired", "A5 $-$ IQFormer-inspired")]

    arrays, limit = [], 0.0
    for reference, _ in references:
        subset = frame[frame["reference"] == reference]
        values = subset.pivot(index="sir_db", columns="snr_db",
                              values="macro_f1_difference") * 100
        lows = subset.pivot(index="sir_db", columns="snr_db",
                            values="macro_f1_ci95_low") * 100
        arrays.append((values, lows))
        limit = max(limit, float(np.nanmax(np.abs(values.to_numpy()))))
    limit = float(np.ceil(limit / 2.0) * 2.0)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_IN, 2.46), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.915, bottom=0.185, top=0.855, wspace=0.085)

    image = None
    for panel, (ax, (_, title), (values, lows)) in enumerate(zip(axes, references, arrays)):
        image = ax.imshow(values.to_numpy(), cmap=GAIN_CMAP, norm=norm,
                          aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(np.arange(len(values.columns)))
        ax.set_xticklabels([f"{x:.0f}" for x in values.columns])
        ax.set_yticks(np.arange(len(values.index)))
        ax.set_yticklabels([f"{y:.0f}" for y in values.index])
        ax.set_xlabel("SNR (dB)")
        ax.set_title(title, pad=5.5, fontsize=8.0)
        origin_frame(ax, minor_x=False, minor_y=False)
        ax.tick_params(which="both", top=False, right=False)
        panel_label(ax, "(a)" if panel == 0 else "(b)", outside=True)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = float(values.iloc[i, j])
                rgba = GAIN_CMAP(norm(value))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                mark = "*" if float(lows.iloc[i, j]) > 0 else ""
                ax.text(j, i, f"{value:.1f}{mark}", ha="center", va="center",
                        fontsize=6.6, color=WHITE if lum < 0.5 else INK)

    axes[0].set_ylabel("SIR (dB)")
    cbar_ax = fig.add_axes([0.932, 0.185, 0.015, 0.67])
    bar = fig.colorbar(image, cax=cbar_ax)
    bar.set_label("A5 gain (percentage points)", fontsize=7.6)
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=7.0, direction="in", length=2.6, width=0.8)
    bar.outline.set_linewidth(0.9)
    bar.outline.set_edgecolor(INK)
    save_figure(fig, figure, out)


# --------------------------------------------------------------------------
# Fig. 3 -- campaign-diagnostic overlap-SIR surface
# --------------------------------------------------------------------------
GROUPS = {
    "id_test":           ("o", BLUE, True, "In distribution"),
    "hard_interference": ("s", BLUE, True, "Hard interference"),
    "unseen_speed":      ("v", VERMILION, False, "Unseen speed"),
    "heldout_channel":   ("D", VERMILION, False, "Held-out channel"),
    "unseen_jammer":     ("^", GREEN, False, "Unseen jammer"),
    "combined_ood":      ("P", GREEN, False, "Combined OOD"),
}


def fig3_envelope(csv_dir: Path, out: Path) -> None:
    figure = "fig3_campaign_diagnostic_envelope"
    cells_all = read_csv(csv_dir, "a14_envelope_cells.csv", figure)
    summary = read_json(csv_dir, "a14_v41_envelope_summary.json", figure)
    skill = read_csv(csv_dir, "a14_v41_envelope_interfered_holdout.csv", figure)

    cells = cells_all[cells_all["split"] != "clean_retention"].copy()
    fit_mask = cells["split"].isin(["id_test", "hard_interference"])
    held_mask = ~fit_mask
    if int(fit_mask.sum()) != 32 or int(held_mask.sum()) != 80:
        raise ValueError("envelope cell contract drifted from 32 fit / 80 held")

    iq = summary["references"]["IQFormer"]
    coef = iq["two_variable_coefficients"]
    intercept = float(coef["intercept_pp"])
    slope_o = float(coef["overlap_pp_per_unit"])
    slope_sir = float(coef["sir_pp_per_db"])
    predicted = intercept + slope_o * cells["occupancy_mean"].to_numpy() \
        + slope_sir * cells["sir_db"].to_numpy()
    observed = 100.0 * cells["gain_vs_IQFormer"].to_numpy()
    fit_mean = float(iq["fit_mean_gain_pp"])
    pooled = skill[(skill["reference"] == "IQFormer")
                   & (skill["regime"] == "pooled_interfered_held")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_IN, 3.06))
    fig.subplots_adjust(left=0.094, right=0.988, bottom=0.255, top=0.925, wspace=0.215)

    # ---- panel (a) ----
    ax = axes[0]
    grid = np.linspace(0.0, 1.0, 100)
    for sir, style in ((-15.0, "-"), (-5.0, "--"), (0.0, ":")):
        ax.plot(grid, intercept + slope_o * grid + slope_sir * sir,
                color=INK, linestyle=style, linewidth=1.0, zorder=2)
        xl = 0.545
        yl = intercept + slope_o * xl + slope_sir * sir
        ax.annotate(("SIR $-$%.0f dB" % abs(sir)) if sir < 0 else "SIR 0 dB",
                    (xl, yl), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.8, color=INK, zorder=6,
                    bbox={"boxstyle": "square,pad=0.14", "facecolor": WHITE,
                          "edgecolor": "none", "alpha": 0.88})
    for split, (marker, color, filled, _) in GROUPS.items():
        subset = cells[cells["split"] == split]
        ax.scatter(subset["occupancy_mean"], 100.0 * subset["gain_vs_IQFormer"],
                   marker=marker, s=22,
                   facecolors=color if filled else "none", edgecolors=color,
                   linewidths=0.85, zorder=4)
    ax.axhline(0.0, color=GRAY, linewidth=0.7, zorder=1)
    ax.set_xlim(-0.035, 1.035)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Target-support jammer overlap, $o$")
    ax.set_ylabel("A5 $-$ IQFormer-insp. macro-F1 (pp)")
    origin_frame(ax)
    panel_label(ax, "(a)", outside=True)

    # ---- panel (b) ----
    ax = axes[1]
    held = held_mask.to_numpy()
    low = min(float(predicted[held].min()), float(observed[held].min())) - 1.8
    high = max(float(predicted[held].max()), float(observed[held].max())) + 1.8
    ax.plot([low, high], [low, high], color=INK, linestyle="--", linewidth=1.0, zorder=2)
    anchor = low + 0.86 * (high - low)
    ax.annotate("identity", (anchor, anchor), xytext=(-3, 5), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.8, color=INK, rotation=45,
                rotation_mode="anchor", zorder=6,
                bbox={"boxstyle": "square,pad=0.12", "facecolor": WHITE,
                      "edgecolor": "none", "alpha": 0.85})
    ax.axvline(fit_mean, color=GRAY, linestyle=":", linewidth=1.0, zorder=1)
    ax.annotate("fit-domain\nconstant", (fit_mean, high), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.8, color=GRAY, zorder=6, linespacing=1.25,
                bbox={"boxstyle": "square,pad=0.12", "facecolor": WHITE,
                      "edgecolor": "none", "alpha": 0.85})
    for split, (marker, color, _, _) in GROUPS.items():
        select = (cells["split"] == split).to_numpy() & held
        if select.any():
            ax.scatter(predicted[select], observed[select], marker=marker, s=22,
                       facecolors="none", edgecolors=color, linewidths=0.85, zorder=4)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Predicted gain (pp)")
    ax.set_ylabel("Observed gain (pp)")
    origin_frame(ax)
    panel_label(ax, "(b)", outside=True)
    ax.text(0.975, 0.045,
            f"Envelope RMSE  {pooled.rmse_pp:.2f} pp\n"
            f"Fit-domain constant  {pooled.constant_rmse_pp:.2f} pp\n"
            f"$R^2$ skill  {pooled.r_squared_skill:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            linespacing=1.32, zorder=7,
            bbox={"boxstyle": "square,pad=0.3", "facecolor": WHITE,
                  "edgecolor": INK, "linewidth": 0.7})

    handles = [Line2D([], [], marker=m, linestyle="none",
                      markerfacecolor=c if f else "none", markeredgecolor=c,
                      markeredgewidth=0.85, markersize=4.6, label=lab)
               for m, c, f, lab in GROUPS.values()]
    origin_legend(fig, handles, loc="lower center", ncol=6,
                  bbox_to_anchor=(0.541, 0.012), columnspacing=0.85,
                  handletextpad=0.3, borderaxespad=0.0, fontsize=7.0)
    save_figure(fig, figure, out)


# --------------------------------------------------------------------------
# Fig. 4 -- clean condition-transfer taxonomy
# --------------------------------------------------------------------------
def fig4_condition_transfer(csv_dir: Path, out: Path) -> None:
    figure = "fig4_clean_condition_transfer"
    frame = read_csv(csv_dir, "a13_condition_transfer.csv", figure)
    modulations = ["BPSK", "PI2BPSK", "QPSK", "8PSK", "16QAM",
                   "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK"]
    display = {"PI2BPSK": r"$\pi$/2-BPSK"}
    trained_clean = {"PI2BPSK", "8PSK", "256QAM", "CPFSK"}
    families = [("compact_spectral_A0_A7", "Compact spectral family"),
                ("iq_domain_high_capacity", "I/Q-domain baselines")]

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_IN, 2.92), sharey=True)
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.30, top=0.845, wspace=0.09)
    x = np.arange(len(modulations))
    width = 0.38

    for panel, (ax, (family, title)) in enumerate(zip(axes, families)):
        subset = frame[frame["model_family"] == family]
        aggregate = subset.groupby("modulation", as_index=True)[
            ["jammed_f1", "clean_f1"]].mean().reindex(modulations)
        if aggregate.isna().any().any():
            raise ValueError(f"missing modulation rows for {family}")
        ax.bar(x - width / 2, aggregate["jammed_f1"], width, color=VERMILION,
               edgecolor=INK, linewidth=0.45, label="Hard interference", zorder=3)
        ax.bar(x + width / 2, aggregate["clean_f1"], width, color=BLUE,
               edgecolor=INK, linewidth=0.45, label="Jammer-free", zorder=3)
        labels = [display.get(n, n) + ("*" if n not in trained_clean else "")
                  for n in modulations]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor",
                           fontsize=6.6)
        ax.set_title(title, pad=3.0, fontsize=7.2, loc="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        origin_frame(ax, minor_x=False)
        ax.tick_params(axis="x", which="both", top=False)
        panel_label(ax, "(a)" if panel == 0 else "(b)", outside=True)

    axes[0].set_ylabel("Per-class F1")
    handles, labels = axes[0].get_legend_handles_labels()
    origin_legend(fig, handles, labels, loc="upper center",
                  bbox_to_anchor=(0.565, 1.0), ncol=2, fontsize=6.9,
                  handlelength=1.3, columnspacing=0.85, borderpad=0.3)
    save_figure(fig, figure, out)


# --------------------------------------------------------------------------
# Fig. 5 -- complexity and representation (supplementary, same house style)
# --------------------------------------------------------------------------
def fig5_complexity(csv_dir: Path, out: Path) -> None:
    figure = "fig5_complexity_and_representation"
    frame = read_csv(csv_dir, "a15_v41_pareto_matched.csv", figure)
    if set(frame["seed_count"]) != {5}:
        raise ValueError("this figure requires the matched five-seed subset")

    order = ["a0_backbone", "a1_single_mask", "a2_tri_no_teacher", "a3_tri_teacher",
             "a3p_tri_proportional_teacher", "a4_tri_teacher_mtl", "a5_vimd_full",
             "a6_dual_full", "a7_vimd_no_residual", "tier2_h2_f2_iq_sidecar",
             "mcldnn_reimplementation", "iqformer_inspired",
             "cssl_amc_supervised_adaptation"]
    display = {"a0_backbone": "A0", "a1_single_mask": "A1", "a2_tri_no_teacher": "A2",
               "a3_tri_teacher": "A3", "a3p_tri_proportional_teacher": "A3-P",
               "a4_tri_teacher_mtl": "A4", "a5_vimd_full": "A5", "a6_dual_full": "A6",
               "a7_vimd_no_residual": "A7",
               "tier2_h2_f2_iq_sidecar": "Received-I/Q sidecar",
               "mcldnn_reimplementation": "MCLDNN",
               "iqformer_inspired": "IQFormer-inspired",
               "cssl_amc_supervised_adaptation": "CSSL"}
    styles = {"tier2_h2_f2_iq_sidecar": (VERMILION, "D", False),
              "mcldnn_reimplementation": (GRAY, "^", True),
              "iqformer_inspired": (GRAY, "s", True),
              "cssl_amc_supervised_adaptation": (GRAY, "v", True)}
    if set(frame["model"]) != set(order):
        raise ValueError("model set changed; update the declared row order")
    rows = frame.set_index("model").loc[order].reset_index()

    fig = plt.figure(figsize=(SINGLE_IN, 3.22))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.0], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    ax_params = fig.add_subplot(grid[0, 1], sharey=ax)
    fig.subplots_adjust(left=0.335, right=0.985, bottom=0.115, top=0.945)

    for idx, row in enumerate(rows.itertuples(index=False)):
        color, marker, filled = styles.get(row.model, (BLUE, "o", True))
        size = 40 if row.model in ("a5_vimd_full", "tier2_h2_f2_iq_sidecar") else 26
        ax.errorbar(row.macro_f1_mean, idx, xerr=row.macro_f1_std, fmt="none",
                    ecolor=color, elinewidth=0.85, capsize=1.7, capthick=0.8, zorder=3)
        ax.scatter(row.macro_f1_mean, idx, marker=marker, s=size,
                   facecolors=color if filled else WHITE, edgecolors=color,
                   linewidths=0.95, zorder=4)
        params = float(row.parameters)
        label = f"{params / 1e6:.2g}M" if params >= 1e6 else f"{params / 1e3:.1f}k"
        ax_params.text(0.5, idx, label, ha="center", va="center", fontsize=6.8)

    ax.axhline(8.5, color=GRAY_LIGHT, linewidth=0.7, zorder=1)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([display[n] for n in order], fontsize=6.9)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.38, 0.522)
    ax.set_xticks([0.40, 0.44, 0.48, 0.52])
    ax.set_xlabel("Hard-interference macro-F1")
    origin_frame(ax, minor_y=False)
    ax.tick_params(axis="y", which="both", length=0, right=False)

    ax_params.set_xlim(0, 1)
    ax_params.set_xticks([])
    ax_params.tick_params(axis="both", which="both", length=0,
                          labelleft=False, labelbottom=False)
    for spine in ax_params.spines.values():
        spine.set_visible(False)
    ax_params.set_title("Parameters", fontsize=7.2, pad=4)
    save_figure(fig, figure, out)


# --------------------------------------------------------------------------
def write_provenance(csv_dir: Path, out: Path) -> None:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"

    record = {
        "schema": "tvt_paper_figures_origin_style.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Presentation-only restyle: unified OriginLab-convention house style.",
        "guarantee": "No model is trained and no frozen artifact is refitted or edited.",
        "house_style": {
            "font": "Arial",
            "frame": "four-sided black box, 0.9 pt",
            "ticks": "inward, mirrored on top and right, minor ticks enabled",
            "grid": "none",
            "legend": "framed, 0.7 pt black rule",
            "palette": {"blue": BLUE, "vermilion": VERMILION, "green": GREEN,
                        "ink": INK, "gray": GRAY},
            "palette_validation": ("three categorical slots; all-pairs light surface; "
                                   "worst CVD dE 11.0, worst normal-vision dE 18.7, "
                                   "all slots >= 3:1 contrast"),
        },
        "sources": {fig: sorted(set(names)) for fig, names in sorted(USED.items())},
        "source_sha256": {name: digest(csv_dir / name)
                          for names in USED.values() for name in set(names)
                          if (csv_dir / name).exists()},
        "outputs": sorted(p.name for p in out.glob("*") if p.is_file()),
    }
    (out / "provenance_origin.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote provenance_origin.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-figure", action="store_true",
                        help="also restyle Fig. 1; off by default, the existing "
                             "teacher figure is kept unchanged")
    args = parser.parse_args()

    print(f"Origin-style figures -> {args.out}")
    # Fig. 1 (the simulation-component teacher) is intentionally NOT regenerated:
    # the existing paper/figures/physical_teacher_example.pdf is kept as is.
    if args.teacher_figure:
        if args.teacher.exists():
            fig1_teacher(args.teacher, args.out)
        else:
            print(f"  skipping Fig. 1: {args.teacher} not found")
    fig2_rank_instability(args.csv, args.out)
    fig3_envelope(args.csv, args.out)
    fig4_condition_transfer(args.csv, args.out)
    fig5_complexity(args.csv, args.out)
    write_provenance(args.csv, args.out)


if __name__ == "__main__":
    main()
