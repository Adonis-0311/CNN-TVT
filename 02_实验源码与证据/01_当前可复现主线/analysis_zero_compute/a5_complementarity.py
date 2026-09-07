"""A5: error complementarity and probability-level fusion.

Question answered: does the compact physics-guided model carry decision
information that the high-capacity baselines do not already have?  Two
controls keep the answer honest:

* the fused pair is compared against its *stronger* member, not its weaker one;
* a same-architecture two-seed ensemble of the stronger member is reported as
  the "ensembling alone" control, so a fusion gain cannot be attributed to
  averaging two networks per se.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

PARTNERS = ("iqformer_inspired", "mcldnn_reimplementation", z.REFERENCE, z.BACKBONE)
REGIMES = ("hard_interference", "id_test", "unseen_jammer", "combined_ood", "clean_retention")
DRAWS = 2000


def _load(model: str, regime: str) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.stack([z.load_probabilities(model, seed, regime) for seed in z.SEEDS])
    with np.load(z.fit_dir(model, z.SEEDS[0]) / f"predictions_{regime}.npz") as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
    return probabilities, labels


def run() -> None:
    overlap_rows: list[dict] = []
    fusion_rows: list[dict] = []

    for regime in REGIMES:
        proposed, labels = _load(z.PROPOSED, regime)
        proposed_pred = proposed.argmax(axis=2)
        proposed_f1 = np.array([z.macro_f1(labels, proposed_pred[s]) for s in range(len(z.SEEDS))])

        for partner in PARTNERS:
            other, _ = _load(partner, regime)
            other_pred = other.argmax(axis=2)
            other_f1 = np.array([z.macro_f1(labels, other_pred[s]) for s in range(len(z.SEEDS))])

            proposed_correct = proposed_pred == labels[None, :]
            other_correct = other_pred == labels[None, :]
            both = (proposed_correct & other_correct).mean(axis=1)
            only_proposed = (proposed_correct & ~other_correct).mean(axis=1)
            only_other = (~proposed_correct & other_correct).mean(axis=1)
            neither = (~proposed_correct & ~other_correct).mean(axis=1)
            disagreement = (proposed_pred != other_pred).mean(axis=1)
            phi = np.array(
                [
                    float(np.corrcoef(proposed_correct[s].astype(float), other_correct[s].astype(float))[0, 1])
                    for s in range(len(z.SEEDS))
                ]
            )
            overlap_rows.append(
                {
                    "regime": regime,
                    "partner": partner,
                    "partner_short": z.SHORT.get(partner, partner),
                    "a5_accuracy": float(proposed_correct.mean()),
                    "partner_accuracy": float(other_correct.mean()),
                    "both_correct": float(both.mean()),
                    "only_a5_correct": float(only_proposed.mean()),
                    "only_partner_correct": float(only_other.mean()),
                    "neither_correct": float(neither.mean()),
                    "oracle_accuracy": float((both + only_proposed + only_other).mean()),
                    "prediction_disagreement": float(disagreement.mean()),
                    "error_correlation_phi": float(phi.mean()),
                }
            )

            fused_pred = ((proposed + other) / 2.0).argmax(axis=2)
            fused_f1 = np.array([z.macro_f1(labels, fused_pred[s]) for s in range(len(z.SEEDS))])
            stronger = z.PROPOSED if proposed_f1.mean() >= other_f1.mean() else partner
            stronger_pred = proposed_pred if stronger == z.PROPOSED else other_pred
            stats = z.hierarchical_paired_diff(labels, stronger_pred, fused_pred, draws=DRAWS)

            rolled = np.roll(other if stronger == partner else proposed, 1, axis=0)
            control_pred = (((other if stronger == partner else proposed) + rolled) / 2.0).argmax(axis=2)
            control_f1 = np.array([z.macro_f1(labels, control_pred[s]) for s in range(len(z.SEEDS))])
            control_stats = z.hierarchical_paired_diff(labels, stronger_pred, control_pred, draws=DRAWS)

            fusion_rows.append(
                {
                    "regime": regime,
                    "partner": partner,
                    "partner_short": z.SHORT.get(partner, partner),
                    "a5_macro_f1": float(proposed_f1.mean()),
                    "partner_macro_f1": float(other_f1.mean()),
                    "stronger_member": stronger,
                    "fused_macro_f1": float(fused_f1.mean()),
                    "control_two_seed_macro_f1": float(control_f1.mean()),
                    "fusion_gain_vs_stronger": stats["macro_f1_difference"],
                    "fusion_ci95_low": stats["macro_f1_ci95_low"],
                    "fusion_ci95_high": stats["macro_f1_ci95_high"],
                    "control_gain_vs_stronger": control_stats["macro_f1_difference"],
                    "control_ci95_low": control_stats["macro_f1_ci95_low"],
                    "control_ci95_high": control_stats["macro_f1_ci95_high"],
                    "cross_model_excess_over_control": stats["macro_f1_difference"]
                    - control_stats["macro_f1_difference"],
                }
            )
            del other, other_pred
        del proposed, proposed_pred

    z.write_csv("a5_error_overlap.csv", overlap_rows)
    z.write_csv("a5_fusion.csv", fusion_rows)
    _figure(overlap_rows, fusion_rows)


def _figure(overlap_rows: list[dict], fusion_rows: list[dict]) -> None:
    import pandas as pd

    plt = z.mpl()
    overlap = pd.DataFrame(overlap_rows)
    fusion = pd.DataFrame(fusion_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))

    ax = axes[0]
    subset = overlap[overlap.regime == "hard_interference"]
    x = np.arange(len(subset))
    ax.bar(x, 100 * subset.both_correct, label="both correct")
    ax.bar(x, 100 * subset.only_a5_correct, bottom=100 * subset.both_correct, label="only A5 correct")
    ax.bar(
        x,
        100 * subset.only_partner_correct,
        bottom=100 * (subset.both_correct + subset.only_a5_correct),
        label="only partner correct",
    )
    ax.bar(
        x,
        100 * subset.neither_correct,
        bottom=100 * (subset.both_correct + subset.only_a5_correct + subset.only_partner_correct),
        label="neither",
        color="#bbbbbb",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset.partner_short, rotation=20, fontsize=7)
    ax.set_ylabel("% of hard-interference windows")
    ax.set_title("Decision overlap with A5 (hard interference)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for partner in PARTNERS:
        subset = fusion[fusion.partner == partner]
        ax.plot(
            subset.regime,
            100 * subset.fusion_gain_vs_stronger,
            marker="o",
            ms=4,
            label=f"A5+{z.SHORT.get(partner, partner)}",
        )
        ax.plot(
            subset.regime,
            100 * subset.control_gain_vs_stronger,
            marker="x",
            ls="--",
            lw=0.9,
            alpha=0.6,
            label=f"2-seed control ({z.SHORT.get(partner, partner)})",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$\\Delta$ macro-F1 vs stronger member (pp)")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.set_title("Fusion gain and ensembling-only control", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    z.save_fig(fig, "figA5_complementarity_and_fusion")
    plt.close(fig)


if __name__ == "__main__":
    run()
