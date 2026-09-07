"""A2: gain decomposition by jammer family, channel profile, speed and occupancy.

All strata come from the frozen cache metadata (view 0, verified row-aligned
with the sealed prediction bundles).  No regeneration, no retraining.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

REGIMES = ("hard_interference", "unseen_jammer", "combined_ood", "id_test", "unseen_speed")
MODELS = (z.BACKBONE, z.PROPOSED, z.REFERENCE, "mcldnn_reimplementation", "iqformer_inspired")
REFERENCES = (z.BACKBONE, z.REFERENCE, "iqformer_inspired")
DRAWS = 1500
MIN_ROWS = 150


def _strata(regime: str, base: z.Preds) -> list[tuple[str, str, np.ndarray]]:
    meta = z.load_cache_metadata(regime)
    out: list[tuple[str, str, np.ndarray]] = []

    families, order = z.jammer_family_labels(regime)
    if len(families):
        for name in order:
            mask = families == name
            if mask.sum() >= MIN_ROWS:
                out.append(("jammer_family", name, mask))

    for profile in np.unique(base.profile):
        mask = base.profile == profile
        if mask.sum() >= MIN_ROWS:
            out.append(("target_profile_index", str(int(profile)), mask))

    speed = meta.get("speed_kmh")
    if speed is not None:
        levels = np.unique(speed)
        if 1 < len(levels) <= 12:
            for level in levels:
                mask = speed == level
                if mask.sum() >= MIN_ROWS:
                    out.append(("speed_kmh", f"{float(level):.0f}", mask))
        elif len(levels) > 12:
            edges = np.quantile(speed, [0, 0.25, 0.5, 0.75, 1.0])
            for i in range(4):
                mask = (speed >= edges[i]) & (speed <= edges[i + 1] if i == 3 else speed < edges[i + 1])
                if mask.sum() >= MIN_ROWS:
                    out.append(("speed_kmh_quartile", f"Q{i + 1} [{edges[i]:.0f},{edges[i + 1]:.0f}]", mask))

    overlap = meta.get("overlap")
    if overlap is not None and len(np.unique(np.quantile(overlap, [0, 0.25, 0.5, 0.75, 1.0]))) == 5:
        edges = np.quantile(overlap, [0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            upper = overlap <= edges[i + 1] if i == 3 else overlap < edges[i + 1]
            mask = (overlap >= edges[i]) & upper
            if mask.sum() >= MIN_ROWS:
                out.append(("occupancy_quartile", f"Q{i + 1} [{edges[i]:.3f},{edges[i + 1]:.3f}]", mask))

    return out


def run(regimes: tuple[str, ...] = REGIMES) -> None:
    absolute_rows: list[dict] = []
    paired_rows: list[dict] = []

    for regime in regimes:
        loaded = {model: z.load_preds(model, regime) for model in MODELS}
        base = loaded[z.PROPOSED]
        for kind, value, mask in _strata(regime, base):
            labels = base.labels[mask]
            if len(np.unique(labels)) < 2:
                continue
            for model, preds in loaded.items():
                f1 = np.array([z.macro_f1(labels, preds.pred[s][mask]) for s in range(len(preds.seeds))])
                absolute_rows.append(
                    {
                        "regime": regime,
                        "stratum_type": kind,
                        "stratum": value,
                        "model": model,
                        "model_short": z.SHORT.get(model, model),
                        "row_count": int(mask.sum()),
                        "macro_f1_mean": f1.mean(),
                        "macro_f1_std": f1.std(ddof=1),
                    }
                )
            for reference in REFERENCES:
                stats = z.hierarchical_paired_diff(
                    labels, loaded[reference].pred[:, mask], base.pred[:, mask], draws=DRAWS
                )
                paired_rows.append(
                    {
                        "regime": regime,
                        "stratum_type": kind,
                        "stratum": value,
                        "candidate": z.PROPOSED,
                        "reference": reference,
                        "reference_short": z.SHORT.get(reference, reference),
                        "row_count": int(mask.sum()),
                        **stats,
                    }
                )
        del loaded

    tag = "all" if tuple(regimes) == REGIMES else "_".join(regimes)
    z.write_shard("a2_strata_absolute", tag, absolute_rows)
    z.write_shard("a2_strata_paired", tag, paired_rows)


def finalize() -> None:
    z.merge_shards("a2_strata_absolute")
    paired = z.merge_shards("a2_strata_paired")
    _figure(paired.to_dict("records"))


def _figure(paired_rows: list[dict]) -> None:
    import pandas as pd

    plt = z.mpl()
    paired = pd.DataFrame(paired_rows)
    families = paired[
        (paired.stratum_type == "jammer_family")
        & (paired.regime.isin(["hard_interference", "unseen_jammer"]))
    ]
    if families.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, regime in zip(axes, ("hard_interference", "unseen_jammer")):
        subset = families[families.regime == regime]
        strata = sorted(subset.stratum.unique())
        width = 0.26
        for k, reference in enumerate(REFERENCES):
            rows = subset[subset.reference == reference].set_index("stratum").reindex(strata)
            x = np.arange(len(strata)) + (k - 1) * width
            values = 100 * rows.macro_f1_difference.to_numpy()
            lower = values - 100 * rows.macro_f1_ci95_low.to_numpy()
            upper = 100 * rows.macro_f1_ci95_high.to_numpy() - values
            ax.bar(x, values, width, yerr=[lower, upper], capsize=2, label=f"vs {z.SHORT.get(reference, reference)}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(strata)))
        ax.set_xticklabels(strata, rotation=30, ha="right", fontsize=7)
        ax.set_title(regime.replace("_", " "), fontsize=9)
    axes[0].set_ylabel("$\\Delta$ macro-F1 (pp)")
    axes[0].legend(fontsize=7)
    fig.suptitle("A5 gain decomposed by jammer family (95% CI)", fontsize=10)
    z.save_fig(fig, "figA2_gain_by_jammer_family")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--finalize"]:
        finalize()
    else:
        run(tuple(sys.argv[1:]) or REGIMES)
