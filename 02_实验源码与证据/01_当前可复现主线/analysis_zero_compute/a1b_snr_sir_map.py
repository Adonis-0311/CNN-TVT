"""A1b: joint SNR x SIR operating map for the proposed model against baselines.

Localises the region where the compact physics-guided model is ahead of the
high-capacity literature baselines.  Cells with fewer than ``MIN_ROWS`` rows or
fewer than 8 represented classes are reported as NaN.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

REFERENCES = ("mcldnn_reimplementation", "iqformer_inspired", z.REFERENCE, z.BACKBONE)
DRAWS = 800
MIN_ROWS = 100


def run(regimes: tuple[str, ...] = ("hard_interference",)) -> None:
    rows: list[dict] = []
    for regime in regimes:
        loaded = {model: z.load_preds(model, regime) for model in (z.PROPOSED, *REFERENCES)}
        base = loaded[z.PROPOSED]
        for snr in np.unique(base.snr_db):
            for sir in np.unique(base.sir_db):
                mask = (base.snr_db == snr) & (base.sir_db == sir)
                labels = base.labels[mask]
                if mask.sum() < MIN_ROWS or len(np.unique(labels)) < 8:
                    continue
                f1_proposed = np.mean(
                    [z.macro_f1(labels, base.pred[s][mask]) for s in range(len(base.seeds))]
                )
                for reference in REFERENCES:
                    stats = z.hierarchical_paired_diff(
                        labels, loaded[reference].pred[:, mask], base.pred[:, mask], draws=DRAWS
                    )
                    rows.append(
                        {
                            "regime": regime,
                            "snr_db": float(snr),
                            "sir_db": float(sir),
                            "row_count": int(mask.sum()),
                            "reference": reference,
                            "reference_short": z.SHORT.get(reference, reference),
                            "a5_macro_f1": float(f1_proposed),
                            **stats,
                        }
                    )
        del loaded
    z.write_shard("a1b_snr_sir_map", "_".join(regimes), rows)


def finalize() -> None:
    frame = z.merge_shards("a1b_snr_sir_map")
    plt = z.mpl()
    references = ["mcldnn_reimplementation", "iqformer_inspired", z.REFERENCE, z.BACKBONE]
    fig, axes = plt.subplots(1, len(references), figsize=(3.3 * len(references), 3.0))
    for ax, reference in zip(np.atleast_1d(axes), references):
        subset = frame[(frame.reference == reference) & (frame.regime == "hard_interference")]
        table = subset.pivot_table(index="sir_db", columns="snr_db", values="macro_f1_difference")
        significant = subset.pivot_table(index="sir_db", columns="snr_db", values="macro_f1_ci95_low")
        values = 100 * table.to_numpy()
        limit = float(np.nanmax(np.abs(values))) if values.size else 1.0
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in table.columns], fontsize=7)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{r:.0f}" for r in table.index], fontsize=7)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"A5 $-$ {z.SHORT.get(reference, reference)} (pp)", fontsize=9)
        ax.grid(False)
        low = 100 * significant.to_numpy()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isnan(values[i, j]):
                    continue
                marker = "*" if low[i, j] > 0 else ""
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.0f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="k",
                )
        fig.colorbar(image, ax=ax, fraction=0.046)
    np.atleast_1d(axes)[0].set_ylabel("SIR (dB)")
    fig.suptitle(
        "Hard-interference operating map: * marks a strictly positive 95% CI for the compact model",
        fontsize=9,
    )
    z.save_fig(fig, "figA1d_snr_sir_operating_map")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--finalize"]:
        finalize()
    else:
        run(tuple(sys.argv[1:]) or ("hard_interference",))
