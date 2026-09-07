"""A7: sample-efficiency reading of the sealed learning-curve evidence.

Reuses ``artifacts/tvt_learning_curve_v4_8gb_dual/learning_curve_evidence.json``
(10k / 30k / 100k source sequences, 5 algorithm seeds, A0 vs A5).  The formal
scale was fixed at 100k before the headline run, so this is a re-reading of
preregistered evidence, not a post-hoc scale selection.
"""

from __future__ import annotations

import json

import numpy as np

import zc_core as z

EVIDENCE = z.REPO / "artifacts" / "tvt_learning_curve_v4_8gb_dual" / "learning_curve_evidence.json"
METRICS = ("hard_interference_macro_f1", "id_test_macro_f1", "selected_validation_macro_f1")


def run() -> None:
    with open(EVIDENCE, encoding="utf-8") as handle:
        evidence = json.load(handle)
    curves = evidence["curves"]

    rows: list[dict] = []
    paired: list[dict] = []
    for scale in sorted(curves, key=int):
        per_model: dict[str, dict[str, list[float]]] = {}
        for key, record in curves[scale].items():
            model, seed = key.split("/")
            store = per_model.setdefault(model, {metric: [] for metric in METRICS})
            for metric in METRICS:
                store[metric].append(float(record[metric]))
        for model, store in per_model.items():
            for metric in METRICS:
                values = np.asarray(store[metric])
                mean, low, high = z.seed_level_ci(values)
                rows.append(
                    {
                        "train_sources": int(scale),
                        "model": model,
                        "model_short": z.SHORT.get(model, model),
                        "metric": metric,
                        "seed_count": len(values),
                        "mean": mean,
                        "std": float(values.std(ddof=1)),
                        "seed_ci_low": low,
                        "seed_ci_high": high,
                    }
                )
        if {"a0_backbone", "a5_vimd_full"} <= set(per_model):
            for metric in METRICS:
                delta = np.asarray(per_model["a5_vimd_full"][metric]) - np.asarray(
                    per_model["a0_backbone"][metric]
                )
                mean, low, high = z.seed_level_ci(delta)
                paired.append(
                    {
                        "train_sources": int(scale),
                        "metric": metric,
                        "delta_mean": mean,
                        "delta_seed_ci_low": low,
                        "delta_seed_ci_high": high,
                        "relative_gain_percent": 100.0
                        * mean
                        / float(np.mean(per_model["a0_backbone"][metric])),
                    }
                )

    z.write_csv("a7_learning_curve_levels.csv", rows)
    z.write_csv("a7_learning_curve_gains.csv", paired)

    # data-equivalence reading: how much backbone training data would be needed
    # to match A5 at a smaller scale (log-linear interpolation, exploratory).
    import pandas as pd

    frame = pd.DataFrame(rows)
    equivalence: list[dict] = []
    for metric in METRICS:
        backbone = frame[(frame.model == "a0_backbone") & (frame.metric == metric)].sort_values(
            "train_sources"
        )
        proposed = frame[(frame.model == "a5_vimd_full") & (frame.metric == metric)].sort_values(
            "train_sources"
        )
        for _, row in proposed.iterrows():
            target = row["mean"]
            xs = np.log10(backbone.train_sources.to_numpy(dtype=float))
            ys = backbone["mean"].to_numpy()
            if target <= ys.max():
                needed = float(10 ** np.interp(target, ys, xs))
                status = "interpolated"
            else:
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                needed = float(10 ** (xs[-1] + (target - ys[-1]) / slope)) if slope > 0 else float("nan")
                status = "extrapolated_beyond_100k"
            equivalence.append(
                {
                    "metric": metric,
                    "a5_train_sources": int(row["train_sources"]),
                    "a5_value": float(target),
                    "backbone_sources_for_same_value": needed,
                    "data_multiplier": needed / float(row["train_sources"]) if needed == needed else float("nan"),
                    "status": status,
                }
            )
    z.write_csv("a7_data_equivalence.csv", equivalence)
    _figure(frame, pd.DataFrame(paired))


def _figure(frame, paired) -> None:
    plt = z.mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for model in ("a0_backbone", "a5_vimd_full"):
        subset = frame[(frame.model == model) & (frame.metric == "hard_interference_macro_f1")].sort_values(
            "train_sources"
        )
        axes[0].errorbar(
            subset.train_sources,
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            capsize=3,
            label=z.SHORT.get(model, model),
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training source sequences")
    axes[0].set_ylabel("hard-interference macro-F1")
    axes[0].set_title("Learning curve (5 seeds, mean $\\pm$ s.d.)", fontsize=9)
    axes[0].legend(fontsize=7)

    subset = paired[paired.metric == "hard_interference_macro_f1"].sort_values("train_sources")
    axes[1].errorbar(
        subset.train_sources,
        100 * subset.delta_mean,
        yerr=[
            100 * (subset.delta_mean - subset.delta_seed_ci_low),
            100 * (subset.delta_seed_ci_high - subset.delta_mean),
        ],
        marker="s",
        capsize=3,
        color="#c0392b",
    )
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training source sequences")
    axes[1].set_ylabel("A5 $-$ A0 (pp)")
    axes[1].set_title("Gain versus training scale", fontsize=9)
    z.save_fig(fig, "figA7_learning_curve")
    plt.close(fig)


if __name__ == "__main__":
    run()
