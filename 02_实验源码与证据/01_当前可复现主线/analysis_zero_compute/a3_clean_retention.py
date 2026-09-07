"""A3: anatomy of the clean-retention deficit.

The sealed release gate reports A5 - CSSL = -9.693 pp macro-F1 on the
clean-retention split.  This module localises that deficit by modulation
class, SNR, channel profile and confusion structure, using only the sealed
prediction bundles.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

SPLIT = "clean_retention"
MODELS = (z.BACKBONE, z.PROPOSED, z.REFERENCE, "mcldnn_reimplementation", "iqformer_inspired")
DRAWS = 2000


def _confusion(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    code = labels.astype(np.int64) * z.CLASSES + pred.astype(np.int64)
    return np.bincount(code, minlength=z.CLASSES**2).reshape(z.CLASSES, z.CLASSES)


def run() -> None:
    loaded = {model: z.load_preds(model, SPLIT) for model in MODELS}
    base = loaded[z.PROPOSED]
    labels = base.labels
    n_seeds = len(base.seeds)

    per_class_rows: list[dict] = []
    for model, preds in loaded.items():
        f1 = np.stack([z.per_class_f1(labels, preds.pred[s]) for s in range(n_seeds)])
        recall = np.stack(
            [
                np.divide(
                    np.diag(_confusion(labels, preds.pred[s])).astype(float),
                    _confusion(labels, preds.pred[s]).sum(axis=1).astype(float),
                    out=np.zeros(z.CLASSES),
                    where=_confusion(labels, preds.pred[s]).sum(axis=1) > 0,
                )
                for s in range(n_seeds)
            ]
        )
        for c in range(z.CLASSES):
            per_class_rows.append(
                {
                    "split": SPLIT,
                    "model": model,
                    "model_short": z.SHORT.get(model, model),
                    "class_index": c,
                    "modulation": z.MODULATIONS[c],
                    "support": int((labels == c).sum()),
                    "f1_mean": float(f1[:, c].mean()),
                    "f1_std": float(f1[:, c].std(ddof=1)),
                    "recall_mean": float(recall[:, c].mean()),
                    "recall_std": float(recall[:, c].std(ddof=1)),
                }
            )
    z.write_csv("a3_clean_per_class.csv", per_class_rows)

    # per-class paired deficit against the CSSL reference and the backbone
    deficit_rows: list[dict] = []
    for reference in (z.REFERENCE, z.BACKBONE, "iqformer_inspired"):
        ref = loaded[reference]
        f1_ref = np.stack([z.per_class_f1(labels, ref.pred[s]) for s in range(n_seeds)])
        f1_cand = np.stack([z.per_class_f1(labels, base.pred[s]) for s in range(n_seeds)])
        delta = f1_cand - f1_ref
        for c in range(z.CLASSES):
            mean, low, high = z.seed_level_ci(delta[:, c])
            deficit_rows.append(
                {
                    "split": SPLIT,
                    "reference": reference,
                    "reference_short": z.SHORT.get(reference, reference),
                    "class_index": c,
                    "modulation": z.MODULATIONS[c],
                    "delta_f1_mean": mean,
                    "delta_f1_seed_ci_low": low,
                    "delta_f1_seed_ci_high": high,
                    "share_of_macro_deficit": float(delta[:, c].mean() / delta.mean(axis=1).mean() / z.CLASSES)
                    if delta.mean() != 0
                    else float("nan"),
                }
            )
    z.write_csv("a3_clean_per_class_deficit.csv", deficit_rows)

    # channel-profile strata (the preregistered seen A/C/D vs held B/E split)
    profile_rows: list[dict] = []
    seen = np.isin(base.profile, [0, 2, 3])
    strata = {
        "all": np.ones(len(labels), dtype=bool),
        "seen_profiles_ACD": seen,
        "held_profiles_BE": ~seen,
        **{f"profile_{int(p)}": base.profile == p for p in np.unique(base.profile)},
    }
    for name, mask in strata.items():
        if mask.sum() < 100:
            continue
        for reference in (z.REFERENCE, z.BACKBONE, "iqformer_inspired"):
            stats = z.hierarchical_paired_diff(
                labels[mask], loaded[reference].pred[:, mask], base.pred[:, mask], draws=DRAWS
            )
            profile_rows.append(
                {
                    "split": SPLIT,
                    "stratum": name,
                    "reference": reference,
                    "reference_short": z.SHORT.get(reference, reference),
                    "row_count": int(mask.sum()),
                    **stats,
                }
            )
    z.write_csv("a3_clean_by_profile.csv", profile_rows)

    # confusion structure, pooled over seeds
    confusion_rows: list[dict] = []
    matrices = {}
    for model in (z.PROPOSED, z.REFERENCE):
        matrix = sum(_confusion(labels, loaded[model].pred[s]) for s in range(n_seeds)).astype(float)
        matrices[model] = matrix / n_seeds
    difference = matrices[z.PROPOSED] - matrices[z.REFERENCE]
    for i in range(z.CLASSES):
        for j in range(z.CLASSES):
            confusion_rows.append(
                {
                    "split": SPLIT,
                    "true_class": z.MODULATIONS[i],
                    "predicted_class": z.MODULATIONS[j],
                    "a5_count_mean": matrices[z.PROPOSED][i, j],
                    "cssl_count_mean": matrices[z.REFERENCE][i, j],
                    "difference": difference[i, j],
                }
            )
    z.write_csv("a3_clean_confusion.csv", confusion_rows)

    # predicted-class distribution: is the deficit a collapse onto few classes?
    collapse_rows: list[dict] = []
    for model, preds in loaded.items():
        shares = np.stack(
            [np.bincount(preds.pred[s].astype(int), minlength=z.CLASSES) / len(labels) for s in range(n_seeds)]
        )
        entropy = -(shares * np.log(np.clip(shares, 1e-12, None))).sum(axis=1)
        collapse_rows.append(
            {
                "split": SPLIT,
                "model": model,
                "model_short": z.SHORT.get(model, model),
                "predicted_share_max_mean": float(shares.max(axis=1).mean()),
                "predicted_share_entropy_mean": float(entropy.mean()),
                "predicted_share_entropy_max": float(np.log(z.CLASSES)),
                "dominant_predicted_class": z.MODULATIONS[int(shares.mean(axis=0).argmax())],
            }
        )
    z.write_csv("a3_clean_prediction_collapse.csv", collapse_rows)

    _figure(per_class_rows, deficit_rows, difference)


def _figure(per_class_rows, deficit_rows, confusion_difference) -> None:
    import pandas as pd

    plt = z.mpl()
    per_class = pd.DataFrame(per_class_rows)
    deficit = pd.DataFrame(deficit_rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))

    ax = axes[0]
    width = 0.27
    for k, model in enumerate((z.BACKBONE, z.PROPOSED, z.REFERENCE)):
        subset = per_class[per_class.model == model].sort_values("class_index")
        ax.bar(
            np.arange(z.CLASSES) + (k - 1) * width,
            subset.f1_mean,
            width,
            yerr=subset.f1_std,
            capsize=2,
            label=z.SHORT.get(model, model),
        )
    ax.set_xticks(range(z.CLASSES))
    ax.set_xticklabels(z.MODULATIONS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("per-class F1")
    ax.set_title("Clean retention: per-class F1", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    subset = deficit[deficit.reference == z.REFERENCE].sort_values("class_index")
    ax.bar(
        np.arange(z.CLASSES),
        100 * subset.delta_f1_mean,
        yerr=[
            100 * (subset.delta_f1_mean - subset.delta_f1_seed_ci_low),
            100 * (subset.delta_f1_seed_ci_high - subset.delta_f1_mean),
        ],
        capsize=2,
        color=["#c0392b" if v < 0 else "#27ae60" for v in subset.delta_f1_mean],
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(z.CLASSES))
    ax.set_xticklabels(z.MODULATIONS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("$\\Delta$ F1 vs CSSL (pp)")
    ax.set_title("Where the clean deficit lives", fontsize=9)

    ax = axes[2]
    limit = float(np.abs(confusion_difference).max())
    image = ax.imshow(confusion_difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(z.CLASSES))
    ax.set_xticklabels(z.MODULATIONS, rotation=90, fontsize=6)
    ax.set_yticks(range(z.CLASSES))
    ax.set_yticklabels(z.MODULATIONS, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("A5 $-$ CSSL confusion counts", fontsize=9)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046)

    z.save_fig(fig, "figA3_clean_retention_anatomy")
    plt.close(fig)


if __name__ == "__main__":
    run()
