"""A3b: is the clean-condition collapse caused by VIMD, or inherited?

Scans all twelve formal models on the clean-retention split, per modulation
class.  This decides whether the -9.69 pp clean deficit is an artefact of the
tri-route decomposition (then it should appear in A5 but not in A0) or a
property of the shared spectral backbone family (then it appears in A0 too).
"""

from __future__ import annotations

import numpy as np

import zc_core as z

SPLIT = "clean_retention"
LINEAR_HIGH_ORDER = ("QPSK", "16QAM", "64QAM")


def run() -> None:
    rows: list[dict] = []
    for model in z.MODELS:
        preds = z.load_preds(model, SPLIT)
        f1 = np.stack([z.per_class_f1(preds.labels, preds.pred[s]) for s in range(len(preds.seeds))])
        recall = np.zeros_like(f1)
        for s in range(len(preds.seeds)):
            for c in range(z.CLASSES):
                mask = preds.labels == c
                recall[s, c] = float((preds.pred[s][mask] == c).mean()) if mask.any() else np.nan
        for c in range(z.CLASSES):
            rows.append(
                {
                    "split": SPLIT,
                    "model": model,
                    "model_short": z.SHORT.get(model, model),
                    "modulation": z.MODULATIONS[c],
                    "support": int((preds.labels == c).sum()),
                    "f1_mean": float(f1[:, c].mean()),
                    "f1_std": float(f1[:, c].std(ddof=1)),
                    "recall_mean": float(recall[:, c].mean()),
                    "collapsed_class": bool(recall[:, c].mean() < 0.02),
                }
            )
    z.write_csv("a3b_clean_family_scan.csv", rows)

    import pandas as pd

    frame = pd.DataFrame(rows)
    summary = []
    for model in z.MODELS:
        subset = frame[frame.model == model]
        high = subset[subset.modulation.isin(LINEAR_HIGH_ORDER)]
        summary.append(
            {
                "model": model,
                "model_short": z.SHORT.get(model, model),
                "collapsed_class_count": int(subset.collapsed_class.sum()),
                "collapsed_classes": ",".join(subset[subset.collapsed_class].modulation),
                "mean_f1_qpsk_16qam_64qam": float(high.f1_mean.mean()),
                "macro_f1_mean": float(subset.f1_mean.mean()),
            }
        )
    z.write_csv("a3b_clean_collapse_summary.csv", summary)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    run()
