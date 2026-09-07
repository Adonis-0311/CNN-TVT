"""A10: audit of the cache mixing law and of jammer-free counterfactual windows.

``src/vimd_amc/data/synthesis.py`` builds every stored window as

    mixture = AGC(clean + jammer + noise + receiver_artifact),
    AGC(z)  = z / sqrt(mean |z|^2)

with an optional ADC quantization stage on the receiver-artifact splits.  This
module verifies that law numerically against the frozen cache and then checks
the construction used to produce jammer-free training views for modulation
classes without natural coverage:

    x_counterfactual = AGC(clean + noise + receiver_artifact)

The stored components are already expressed on the same post-AGC scale and
sum to ``x``. This is a re-normalized re-mix of already-sealed components, not
a new channel realisation:
no MATLAB, no nrTDLChannel call, no new source sequence.  Whether it may be
used for *training* is a preregistration question (see PREREGISTRATION_TIER2);
this module only establishes that the reconstruction is exact.

It also documents why the first Tier-2 probe run is confounded: it fed
``x - jammer`` without re-normalizing the jammer-free result.
"""

from __future__ import annotations

import json

import numpy as np

import zc_core as z

SPLITS = ("train", "validation", "clean_retention", "hard_interference", "adc_10bit_agc")
SAMPLE = 256


def _agc(values: np.ndarray) -> np.ndarray:
    power = (values**2).sum(axis=(1, 2), keepdims=True) / values.shape[2]
    return values / np.sqrt(power)


def run() -> None:
    rows: list[dict] = []
    for split in SPLITS:
        root = z.CACHE / split
        if not root.exists():
            continue
        index = np.arange(min(SAMPLE, len(np.load(root / "label.npy"))))
        load = lambda name: np.asarray(np.load(root / f"{name}.npy", mmap_mode="r")[index, 0])  # noqa: E731
        x = load("x")
        components = load("clean") + load("jammer") + load("noise") + load("receiver_artifact")
        reconstruction = _agc(components)
        scale = np.abs(x).max(axis=(1, 2))
        relative = np.abs(reconstruction - x).max(axis=(1, 2)) / scale

        naive = x - load("jammer")
        counterfactual = _agc(load("clean") + load("noise") + load("receiver_artifact"))
        rows.append(
            {
                "split": split,
                "windows_checked": int(len(index)),
                "mixing_law_max_relative_error": float(relative.max()),
                "mixing_law_median_relative_error": float(np.median(relative)),
                "mixing_law_verified": bool(relative.max() < 1e-5),
                "stored_x_mean_square": float((x**2).mean()),
                "counterfactual_mean_square_min": float((counterfactual**2).mean(axis=(1, 2)).min()),
                "counterfactual_mean_square_max": float((counterfactual**2).mean(axis=(1, 2)).max()),
                "naive_x_minus_jammer_mean_square_min": float((naive**2).mean(axis=(1, 2)).min()),
                "naive_x_minus_jammer_mean_square_max": float((naive**2).mean(axis=(1, 2)).max()),
            }
        )
    z.write_csv("a10_counterfactual_check.csv", rows)

    summary = {
        "mixing_law": "x = (clean + jammer + noise + receiver_artifact) / sqrt(mean|.|^2)",
        "source": "src/vimd_amc/data/synthesis.py lines 712-714",
        "verified_splits": [row["split"] for row in rows if row["mixing_law_verified"]],
        "valid_jammer_free_counterfactual": "AGC(clean + noise + receiver_artifact)",
        "invalid_construction_used_in_first_probe_run": "x - jammer",
        "why_invalid": (
            "the stored components share x's post-AGC scale, but no post-removal renormalisation "
            "is applied; the resulting windows have mean square between about 0.06 and 0.50 while "
            "every window the networks were trained on has mean square exactly 0.5"
        ),
        "consequence": (
            "the representation_limited verdict of artifacts/tier2_clean_probe_v1 is confounded by "
            "input-scale and composition shift and must not be used as the H2-D decision"
        ),
        "evidence_class": "exploratory_zero_compute_audit",
    }
    with open(z.OUT / "a10_counterfactual_check_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
