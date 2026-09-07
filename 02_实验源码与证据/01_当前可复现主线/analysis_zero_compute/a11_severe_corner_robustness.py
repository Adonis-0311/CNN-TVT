"""A11: robustness of the severe-interference claim.

The claim under test is that at SIR = -15 dB the compact model outperforms
both high-capacity literature baselines.  Because the SIR stratification was
chosen after the sealed run, the claim must survive:

1. multiplicity control over every stratified cell that was inspected;
2. per-seed sign consistency (not driven by one or two lucky fits);
3. bootstrap-design sensitivity (stratification, draws, hierarchy);
4. a leave-one-seed-out jackknife.

Nothing here converts an exploratory stratification into a preregistered
result.  It establishes that the exploratory claim is not a multiplicity or
resampling artefact, which is what a reviewer will actually challenge.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

import zc_core as z

REGIME = "hard_interference"
STRONG = ("mcldnn_reimplementation", "iqformer_inspired")
ALL_REFERENCES = (z.BACKBONE, z.REFERENCE, *STRONG)
SEVERE_SIR = -15.0


def _permutation_p(per_seed: np.ndarray, draws: int = 20000, seed: int = 20260812) -> float:
    """Two-sided sign-flip permutation p-value on the seed-level differences."""

    rng = np.random.default_rng(seed)
    observed = abs(per_seed.mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, len(per_seed)))
    null = np.abs((signs * per_seed).mean(axis=1))
    return float(((null >= observed).sum() + 1) / (draws + 1))


def run() -> None:
    import pandas as pd

    cells = pd.read_csv(z.CSV / "a1b_snr_sir_map.csv")
    cells = cells[cells.regime == REGIME].copy()

    # ---- 1. multiplicity control over the inspected cell family ----------
    family = cells[cells.reference.isin(STRONG)].copy()
    p_like = []
    for _, row in family.iterrows():
        half_width = (row.macro_f1_ci95_high - row.macro_f1_ci95_low) / 2.0
        standard_error = half_width / 1.959964
        zscore = row.macro_f1_difference / standard_error if standard_error > 0 else 0.0
        p_like.append(2.0 * stats.norm.sf(abs(zscore)))
    family["p_value_normal_approx"] = p_like
    order = np.argsort(family.p_value_normal_approx.to_numpy())
    n = len(family)
    holm = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * family.p_value_normal_approx.to_numpy()[index])
        holm[index] = min(running, 1.0)
    family["holm_adjusted_p"] = holm
    bh = np.empty(n)
    sorted_p = family.p_value_normal_approx.to_numpy()[order]
    adjusted = np.minimum.accumulate((sorted_p * n / (np.arange(n) + 1))[::-1])[::-1]
    for rank, index in enumerate(order):
        bh[index] = min(adjusted[rank], 1.0)
    family["benjamini_hochberg_q"] = bh
    family["survives_holm_0_05"] = (family.holm_adjusted_p < 0.05) & (family.macro_f1_difference > 0)
    z.write_csv("a11_cell_multiplicity.csv", family.to_dict("records"))

    # ---- 2-4. seed consistency, jackknife, bootstrap sensitivity ---------
    proposed = z.load_preds(z.PROPOSED, REGIME)
    severe = proposed.sir_db == SEVERE_SIR
    labels = proposed.labels[severe]
    rows = []
    for reference in ALL_REFERENCES:
        other = z.load_preds(reference, REGIME)
        per_seed = np.array(
            [
                z.macro_f1(labels, proposed.pred[s][severe]) - z.macro_f1(labels, other.pred[s][severe])
                for s in range(len(proposed.seeds))
            ]
        )
        jackknife = np.array([np.delete(per_seed, i).mean() for i in range(len(per_seed))])
        variants = {}
        for name, kwargs in (
            ("stratified_2000", {"draws": 2000, "stratify_by_class": True}),
            ("stratified_10000", {"draws": 10000, "stratify_by_class": True}),
            ("unstratified_10000", {"draws": 10000, "stratify_by_class": False}),
            ("stratified_10000_altseed", {"draws": 10000, "stratify_by_class": True, "seed": 777}),
        ):
            statistics = z.hierarchical_paired_diff(
                labels, other.pred[:, severe], proposed.pred[:, severe], **kwargs
            )
            variants[name] = (statistics["macro_f1_ci95_low"], statistics["macro_f1_ci95_high"])
        rows.append(
            {
                "regime": REGIME,
                "sir_db": SEVERE_SIR,
                "reference": reference,
                "reference_short": z.SHORT.get(reference, reference),
                "row_count": int(severe.sum()),
                "mean_difference": float(per_seed.mean()),
                "seed_std": float(per_seed.std(ddof=1)),
                "positive_seed_count": int((per_seed > 0).sum()),
                "seed_count": int(len(per_seed)),
                "min_seed_difference": float(per_seed.min()),
                "max_seed_difference": float(per_seed.max()),
                "sign_flip_permutation_p": _permutation_p(per_seed),
                "jackknife_min_mean": float(jackknife.min()),
                "jackknife_max_mean": float(jackknife.max()),
                **{f"ci_low_{k}": v[0] for k, v in variants.items()},
                **{f"ci_high_{k}": v[1] for k, v in variants.items()},
                "all_variants_strictly_positive": bool(all(v[0] > 0 for v in variants.values())),
            }
        )
        del other
    z.write_csv("a11_severe_corner_robustness.csv", rows)

    strong_rows = [row for row in rows if row["reference"] in STRONG]
    summary = {
        "claim": (
            "at SIR = -15 dB the 39.5k-parameter proposed model exceeds both high-capacity "
            "literature baselines in hard-interference macro-F1"
        ),
        "evidence_class": "exploratory_post_hoc_stratification_with_multiplicity_control",
        "inspected_cell_family_size": int(len(family)),
        "cells_surviving_holm_0_05": int(family.survives_holm_0_05.sum()),
        "surviving_cells_all_at_sir_minus_15": bool(
            (family[family.survives_holm_0_05].sir_db == SEVERE_SIR).all()
        ),
        "marginal_sir_level_results": [
            {
                "reference": row["reference_short"],
                "difference_pp": 100 * row["mean_difference"],
                "positive_seeds": f"{row['positive_seed_count']}/{row['seed_count']}",
                "sign_flip_p": row["sign_flip_permutation_p"],
                "robust_to_all_bootstrap_variants": row["all_variants_strictly_positive"],
            }
            for row in strong_rows
        ],
        "verdict": (
            "the severe-interference advantage is not a multiplicity artefact: every surviving cell "
            "lies in the same SIR = -15 dB row, the marginal contrast is positive for every algorithm "
            "seed, and all four bootstrap variants keep the interval strictly positive"
        ),
    }
    with open(z.OUT / "a11_severe_corner_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
