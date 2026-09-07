"""A1: SNR- and SIR-resolved absolute performance and paired contrasts.

Zero new simulation: every number comes from the sealed prediction bundles and
their per-row ``snr_db`` / ``sir_db`` metadata.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

REGIMES = (
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
)
CURVE_MODELS = (
    z.BACKBONE,
    z.PROPOSED,
    z.REFERENCE,
    "mcldnn_reimplementation",
    "iqformer_inspired",
)
CONTRAST_REFERENCES = (z.BACKBONE, z.REFERENCE, "mcldnn_reimplementation", "iqformer_inspired")
DRAWS = 1000


def _cells(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.unique(values)]


def run(regimes: tuple[str, ...] = REGIMES) -> None:
    absolute_rows: list[dict] = []
    paired_rows: list[dict] = []

    for regime in regimes:
        loaded = {model: z.load_preds(model, regime) for model in CURVE_MODELS}
        base = loaded[z.PROPOSED]
        axes = {"snr_db": base.snr_db, "sir_db": base.sir_db}

        for axis_name, axis_values in axes.items():
            levels = _cells(axis_values)
            if len(levels) < 2:
                continue
            for level in levels:
                mask = axis_values == level
                labels = base.labels[mask]
                if len(np.unique(labels)) < 2:
                    continue

                for model, preds in loaded.items():
                    f1 = np.array(
                        [z.macro_f1(labels, preds.pred[s][mask]) for s in range(len(preds.seeds))]
                    )
                    acc = np.array(
                        [float((preds.pred[s][mask] == labels).mean()) for s in range(len(preds.seeds))]
                    )
                    absolute_rows.append(
                        {
                            "regime": regime,
                            "axis": axis_name,
                            "level_db": level,
                            "model": model,
                            "model_short": z.SHORT.get(model, model),
                            "row_count": int(mask.sum()),
                            "macro_f1_mean": f1.mean(),
                            "macro_f1_std": f1.std(ddof=1),
                            "accuracy_mean": acc.mean(),
                            "accuracy_std": acc.std(ddof=1),
                        }
                    )

                for reference in CONTRAST_REFERENCES:
                    stats = z.hierarchical_paired_diff(
                        labels,
                        loaded[reference].pred[:, mask],
                        base.pred[:, mask],
                        draws=DRAWS,
                    )
                    paired_rows.append(
                        {
                            "regime": regime,
                            "axis": axis_name,
                            "level_db": level,
                            "candidate": z.PROPOSED,
                            "reference": reference,
                            "reference_short": z.SHORT.get(reference, reference),
                            "row_count": int(mask.sum()),
                            **stats,
                        }
                    )
        del loaded

    tag = "all" if tuple(regimes) == REGIMES else "_".join(regimes)
    z.write_shard("a1_absolute_by_level", tag, absolute_rows)
    z.write_shard("a1_paired_by_level", tag, paired_rows)


def finalize() -> None:
    absolute = z.merge_shards("a1_absolute_by_level")
    paired = z.merge_shards("a1_paired_by_level")
    _figures(absolute.to_dict("records"), paired.to_dict("records"))


def _figures(absolute_rows: list[dict], paired_rows: list[dict]) -> None:
    import pandas as pd

    plt = z.mpl()
    absolute = pd.DataFrame(absolute_rows)
    paired = pd.DataFrame(paired_rows)

    panels = ["id_test", "hard_interference", "unseen_jammer", "clean_retention"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1), sharey=True)
    for ax, regime in zip(axes, panels):
        subset = absolute[(absolute.regime == regime) & (absolute.axis == "snr_db")]
        for model in CURVE_MODELS:
            curve = subset[subset.model == model].sort_values("level_db")
            if curve.empty:
                continue
            ax.errorbar(
                curve.level_db,
                curve.macro_f1_mean,
                yerr=curve.macro_f1_std,
                marker="o",
                ms=3,
                lw=1.3,
                capsize=2,
                label=z.SHORT.get(model, model),
            )
        ax.set_title(regime.replace("_", " "), fontsize=9)
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Macro-F1 versus SNR (10 algorithm seeds, mean $\\pm$ s.d.)", fontsize=10)
    z.save_fig(fig, "figA1a_macro_f1_vs_snr")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1), sharey=True)
    for ax, reference in zip(axes, CONTRAST_REFERENCES):
        subset = paired[(paired.reference == reference) & (paired.axis == "snr_db")]
        for regime in ("id_test", "hard_interference", "unseen_jammer", "clean_retention"):
            curve = subset[subset.regime == regime].sort_values("level_db")
            if curve.empty:
                continue
            ax.plot(curve.level_db, 100 * curve.macro_f1_difference, marker="o", ms=3, lw=1.3, label=regime)
            ax.fill_between(
                curve.level_db,
                100 * curve.macro_f1_ci95_low,
                100 * curve.macro_f1_ci95_high,
                alpha=0.15,
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"A5 $-$ {z.SHORT.get(reference, reference)}", fontsize=9)
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("$\\Delta$ macro-F1 (pp)")
    axes[0].legend(fontsize=7)
    fig.suptitle("SNR-resolved paired contrasts (hierarchical bootstrap, 95% CI)", fontsize=10)
    z.save_fig(fig, "figA1b_paired_diff_vs_snr")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, regime in zip(axes, ("hard_interference", "unseen_jammer", "id_test")):
        subset = absolute[(absolute.regime == regime) & (absolute.axis == "sir_db")]
        for model in CURVE_MODELS:
            curve = subset[subset.model == model].sort_values("level_db")
            if curve.empty:
                continue
            ax.errorbar(
                curve.level_db,
                curve.macro_f1_mean,
                yerr=curve.macro_f1_std,
                marker="s",
                ms=3,
                lw=1.3,
                capsize=2,
                label=z.SHORT.get(model, model),
            )
        ax.set_title(regime.replace("_", " "), fontsize=9)
        ax.set_xlabel("SIR (dB)")
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Macro-F1 versus SIR", fontsize=10)
    z.save_fig(fig, "figA1c_macro_f1_vs_sir")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args == ["--finalize"]:
        finalize()
    else:
        run(tuple(args) if args else REGIMES)
