"""A13: interference-condition transfer as a benchmark property.

A9 audits jammer-free coverage across both views actually consumed by the
training loop.  This module turns that observation into a
quantitative taxonomy:

* which modulation groups transfer from the jammed training condition to the
  jammer-free evaluation condition;
* which model families succeed at that transfer;
* the size of the transfer gap for each combination.

The result is a property of the benchmark plus a discriminative finding about
representation families, not a defect report about one model.
"""

from __future__ import annotations

import json

import numpy as np

import zc_core as z

ENVELOPE_DISTINCTIVE = ("BPSK", "PI2BPSK", "GMSK", "CPFSK", "4FSK")
CONSTELLATION_ORDER = ("QPSK", "8PSK", "16QAM", "64QAM", "256QAM")
FAMILIES = {
    "compact_spectral_A0_A7": [m for m in z.MODELS if m.startswith(("a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"))],
    "iq_domain_high_capacity": ["cssl_amc_supervised_adaptation", "mcldnn_reimplementation", "iqformer_inspired"],
}


def run() -> None:
    import pandas as pd

    coverage = pd.read_csv(z.CSV / "a9_condition_coverage.csv")
    train_coverage = coverage[coverage.split == "train"]
    trained_clean = tuple(train_coverage[train_coverage.jammer_free > 0].modulation)
    missing_clean = tuple(train_coverage[train_coverage.jammer_free == 0].modulation)
    rows: list[dict] = []
    for model in z.MODELS:
        jammed = z.load_preds(model, "hard_interference")
        clean = z.load_preds(model, "clean_retention")
        jammed_f1 = np.stack([z.per_class_f1(jammed.labels, jammed.pred[s]) for s in range(10)]).mean(0)
        clean_f1 = np.stack([z.per_class_f1(clean.labels, clean.pred[s]) for s in range(10)]).mean(0)
        family = next((k for k, v in FAMILIES.items() if model in v), "other")
        for c, name in enumerate(z.MODULATIONS):
            rows.append(
                {
                    "model": model,
                    "model_short": z.SHORT.get(model, model),
                    "model_family": family,
                    "modulation": name,
                    "training_condition_coverage": "jammed_and_clean"
                    if name in trained_clean
                    else "jammed_only",
                    "discriminability_type": "constellation_order"
                    if name in CONSTELLATION_ORDER
                    else "envelope_distinctive",
                    "jammed_f1": float(jammed_f1[c]),
                    "clean_f1": float(clean_f1[c]),
                    "transfer_gap": float(clean_f1[c] - jammed_f1[c]),
                    "transfer_failed": bool(clean_f1[c] < 0.05 and jammed_f1[c] > 0.10),
                }
            )
        del jammed, clean
    z.write_csv("a13_condition_transfer.csv", rows)

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["model_family", "training_condition_coverage", "discriminability_type"])
        .agg(
            jammed_f1=("jammed_f1", "mean"),
            clean_f1=("clean_f1", "mean"),
            transfer_gap=("transfer_gap", "mean"),
            transfer_failed_share=("transfer_failed", "mean"),
            cells=("transfer_failed", "size"),
        )
        .reset_index()
    )
    z.write_csv("a13_condition_transfer_summary.csv", grouped.to_dict("records"))

    failures = frame[frame.transfer_failed]
    summary = {
        "benchmark_property": (
            "the frozen factor-isolated cache presents jammer-free training views for "
            + ", ".join(trained_clean)
            + "; for the remaining "
            + str(len(missing_clean))
            + " modulations the clean-retention split is an "
            "out-of-distribution interference-condition transfer by construction"
        ),
        "train_classes_with_jammer_free_views": list(trained_clean),
        "train_classes_without_jammer_free_views": list(missing_clean),
        "transfer_failure_definition": "clean-retention F1 < 0.05 while hard-interference F1 > 0.10",
        "failures_by_family": failures.groupby("model_family").modulation.apply(
            lambda values: sorted(set(values))
        ).to_dict(),
        "failure_cell_count": {
            family: int((frame[frame.model_family == family].transfer_failed).sum())
            for family in frame.model_family.unique()
        },
        "finding": (
            "transfer failure is concentrated in the compact spectral family and "
            "constellation-order classes; coverage is reported separately because the "
            "failure taxonomy does not reduce to condition coverage alone"
        ),
        "consequence_for_reporting": (
            "the clean-retention contrast measures condition transfer, not clean-signal accuracy; "
            "it should be reported as a characterised operating boundary of compact spectral "
            "representations under this benchmark, with the preregistered gate outcome stated as "
            "failed"
        ),
    }
    with open(z.OUT / "a13_condition_transfer_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(grouped.round(3).to_string(index=False))
    print()
    print(json.dumps(summary["failures_by_family"], indent=1, ensure_ascii=False))
    _figure(frame, trained_clean)


def _figure(frame, trained_clean: tuple[str, ...]) -> None:
    plt = z.mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        subset = frame[frame.model_family == family]
        pivot = subset.pivot_table(index="modulation", values=["jammed_f1", "clean_f1"], aggfunc="mean")
        pivot = pivot.reindex(z.MODULATIONS)
        x = np.arange(len(pivot))
        ax.bar(x - 0.2, pivot.jammed_f1, 0.4, label="hard interference")
        ax.bar(x + 0.2, pivot.clean_f1, 0.4, label="clean retention")
        for index, name in enumerate(pivot.index):
            if name not in trained_clean:
                ax.text(index, -0.06, "*", ha="center", fontsize=9, color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(family.replace("_", " "), fontsize=9)
    axes[0].set_ylabel("per-class F1")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Interference-condition transfer; * marks modulations never presented jammer-free in training",
        fontsize=9,
    )
    z.save_fig(fig, "figA13_condition_transfer")
    plt.close(fig)


if __name__ == "__main__":
    run()
