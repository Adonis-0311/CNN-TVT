"""A9: which (modulation, interference condition) combinations does training cover?

Motivated by the A3b result.  If the clean-condition collapse is confined to
classes that training never presents without a jammer, then the
clean-retention gate is measuring an out-of-distribution condition transfer,
not a defect introduced by the proposed mechanism.

Pure metadata accounting over the frozen cache; no model is loaded.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

SPLITS = ("train", "validation", "id_test", "clean_retention", "hard_interference", "unseen_jammer")


def run() -> None:
    rows: list[dict] = []
    for split in SPLITS:
        root = z.CACHE / split
        if not root.exists():
            continue
        labels = np.load(root / "label.npy")
        jam = np.load(root / "jam_labels.npy", mmap_mode="r")
        active = np.asarray(jam) > 0.5
        clean = active.sum(axis=2) == 0
        for c in range(z.CLASSES):
            source_mask = labels == c
            view_mask = np.repeat(source_mask[:, None], 2, axis=1)
            rows.append(
                {
                    "split": split,
                    "class_index": c,
                    "modulation": z.MODULATIONS[c],
                    "total": int(view_mask.sum()),
                    "jammer_free": int((view_mask & clean).sum()),
                    "jammed": int((view_mask & ~clean).sum()),
                    "jammer_free_fraction": float((view_mask & clean).sum() / view_mask.sum())
                    if view_mask.any()
                    else 0.0,
                    "accounting_unit": "training_view",
                }
            )
    z.write_csv("a9_condition_coverage.csv", rows)

    import pandas as pd

    frame = pd.DataFrame(rows)
    train = frame[frame.split == "train"]
    uncovered = train[train.jammer_free == 0].modulation.tolist()
    summary = {
        "train_classes_with_zero_jammer_free_windows": uncovered,
        "train_classes_with_jammer_free_windows": train[train.jammer_free > 0].modulation.tolist(),
        "clean_retention_classes": int((frame[frame.split == "clean_retention"].total > 0).sum()),
        "interpretation": (
            "clean_retention evaluates all 10 modulations without interference, while training "
            "presents only the listed classes in that condition; for the remaining classes the "
            "jammer-free condition is out of distribution by construction of the frozen cache."
        ),
    }
    import json

    with open(z.OUT / "a9_condition_coverage_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    pivot = frame.pivot_table(index="modulation", columns="split", values="jammer_free", fill_value=0)
    print(pivot.reindex(z.MODULATIONS).to_string())
    print("\ntrain classes never seen jammer-free:", uncovered)


if __name__ == "__main__":
    run()
