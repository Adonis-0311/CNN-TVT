"""A12: design resolution and bounded-effect statements.

An interval that contains zero is not the same as an absent effect.  This
module converts the sealed intervals into two statements a reviewer can act
on:

* **resolution** -- the smallest effect the frozen design could have detected
  at 80% power, derived from the sealed interval half-width;
* **bound** -- the largest effect the data still permit, i.e. the interval's
  outer edge, which turns "not significant" into "smaller than X".

Both are arithmetic on the sealed statistics; no re-estimation, no new
bootstrap, no change to any gate outcome.  The confirmatory family gate
remains failed and must still be reported as failed.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

import zc_core as z

Z_ALPHA = stats.norm.ppf(0.975)
Z_POWER = stats.norm.ppf(0.80)


def _resolution(low: float, high: float) -> dict[str, float]:
    half = (high - low) / 2.0
    standard_error = half / Z_ALPHA
    return {
        "standard_error": standard_error,
        "min_detectable_effect_80power_pp": 100 * (Z_ALPHA + Z_POWER) * standard_error,
        "upper_bound_abs_effect_pp": 100 * max(abs(low), abs(high)),
    }


def run() -> None:
    import pandas as pd

    ablation = pd.read_csv(z.COMPOSITE / "ablation_paired_statistics.csv")
    rows = []
    for _, record in ablation.iterrows():
        marginal = _resolution(record.macro_f1_marginal_ci95_low, record.macro_f1_marginal_ci95_high)
        simultaneous = _resolution(
            record.macro_f1_simultaneous_ci95_low, record.macro_f1_simultaneous_ci95_high
        )
        detected = record.macro_f1_simultaneous_ci95_low > 0
        rows.append(
            {
                "family": record.family_id,
                "contrast": record.contrast_id,
                "intervention": record.intervention,
                "difference_pp": 100 * record.macro_f1_difference,
                "marginal_ci_low_pp": 100 * record.macro_f1_marginal_ci95_low,
                "marginal_ci_high_pp": 100 * record.macro_f1_marginal_ci95_high,
                "simultaneous_ci_low_pp": 100 * record.macro_f1_simultaneous_ci95_low,
                "simultaneous_ci_high_pp": 100 * record.macro_f1_simultaneous_ci95_high,
                "marginal_mde_80power_pp": marginal["min_detectable_effect_80power_pp"],
                "simultaneous_mde_80power_pp": simultaneous["min_detectable_effect_80power_pp"],
                "effect_upper_bound_simultaneous_pp": simultaneous["upper_bound_abs_effect_pp"],
                "detected_at_simultaneous_95": bool(detected),
                "statement_class": (
                    "detected_effect"
                    if detected
                    else "bounded_effect"
                ),
                "sealed_gate_passed": bool(record.gate_passed),
            }
        )
    z.write_csv("a12_effect_resolution_ablation.csv", rows)

    # the same treatment for the headline regimes, where several intervals sit
    # close to zero and are currently described only as "small"
    headline = pd.read_csv(z.COMPOSITE / "headline_paired_statistics.csv")
    headline = headline[headline.candidate == z.PROPOSED]
    headline_rows = []
    for _, record in headline.iterrows():
        resolution = _resolution(record.macro_f1_ci95_low, record.macro_f1_ci95_high)
        headline_rows.append(
            {
                "regime": record.regime,
                "difference_pp": 100 * record.macro_f1_difference,
                "ci_low_pp": 100 * record.macro_f1_ci95_low,
                "ci_high_pp": 100 * record.macro_f1_ci95_high,
                "mde_80power_pp": resolution["min_detectable_effect_80power_pp"],
                "effect_bound_pp": resolution["upper_bound_abs_effect_pp"],
                "excludes_zero": bool(record.macro_f1_ci95_low > 0 or record.macro_f1_ci95_high < 0),
            }
        )
    z.write_csv("a12_effect_resolution_headline.csv", headline_rows)

    ablation_frame = pd.DataFrame(rows)
    bounded = ablation_frame[~ablation_frame.detected_at_simultaneous_95]
    detected = ablation_frame[ablation_frame.detected_at_simultaneous_95]
    summary = {
        "purpose": (
            "distinguish an absent effect from an unresolved one, and give the largest effect the "
            "sealed data still permit"
        ),
        "design_resolution_pp": {
            "primary_method_effect_contrast": float(
                detected.simultaneous_mde_80power_pp.iloc[0]
            )
            if len(detected)
            else None,
            "component_level_contrasts": [
                {"contrast": row.contrast, "mde_80power_pp": float(row.simultaneous_mde_80power_pp)}
                for row in bounded.itertuples()
            ],
        },
        "bounded_statements": [
            {
                "contrast": row.contrast,
                "intervention": row.intervention,
                "observed_pp": float(row.difference_pp),
                "simultaneous_interval_pp": [
                    float(row.simultaneous_ci_low_pp),
                    float(row.simultaneous_ci_high_pp),
                ],
                "claimable": (
                    f"the individual contribution of this component is bounded by "
                    f"{row.effect_upper_bound_simultaneous_pp:.2f} pp under joint 95% simultaneous "
                    f"confidence"
                ),
                "not_claimable": "that the component is confirmed to help, or that it is proven useless",
            }
            for row in bounded.itertuples()
        ],
        "gate_status_unchanged": {
            "confirmatory_family_gate_passed": bool(ablation.family_gate_passed.iloc[0]),
            "note": (
                "the family gate required all three contrasts to clear zero; it did not, and this "
                "module does not change that. It only replaces 'inconclusive' with a quantitative bound."
            ),
        },
    }
    with open(z.OUT / "a12_effect_resolution_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(ablation_frame.round(3).to_string(index=False))
    print()
    print(pd.DataFrame(headline_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run()
