# V1 audit: envelope constant-baseline and skill score

**Date:** 2026-08-18
**Status:** the previously reported values are withdrawn and replaced.
**Scope:** Section V-C, the abstract, Fig. 4, and `v1_audit_macros.tex`.

## Finding

The manuscript reported a constant-baseline RMSE of 15.28 pp (IQFormer-inspired)
and 13.29 pp (MCLDNN), giving `R^2_skill` of 0.84 and 0.63. **These are wrong.**
They are the product of a unit mismatch, not a definitional disagreement.

`analysis_zero_compute/outputs/csv/a14_envelope_cells.csv` stores the
`gain_vs_*` columns as **fractions** (raw macro-F1 differences). Every
percentage-point quantity in `a14_envelope_fit.csv` and
`a14_envelope_holdout.csv` is multiplied by 100 at write time
(`a14_envelope_model.py`, lines that emit `holdout_rmse_pp`,
`actual_gain_pp`, `predicted_gain_pp`).

The withdrawn constant baseline was computed by subtracting the
*fraction-valued* fitting-cell mean (-0.0827) from the *pp-valued* held-regime
actuals. That is arithmetically equivalent to scoring against a constant of
approximately 0 pp, which inflates the baseline RMSE and therefore the skill
score.

## Corrected values

Baseline definition (unchanged in intent): the constant predictor outputs the
mean gain over the 32 fitting cells (`id_test` + `hard_interference`) and is
evaluated unchanged on the 81 held-regime cells.

| reference | fit-cell mean | holdout mean | envelope RMSE | constant RMSE | `R^2_skill` |
|---|---:|---:|---:|---:|---:|
| IQFormer-inspired | -8.27 pp | -14.12 pp | 6.20 pp | **8.40 pp** | **0.456** |
| MCLDNN            | -4.63 pp | -11.48 pp | 8.05 pp | **9.63 pp** | **0.301** |
| CSSL              | +3.94 pp |  -2.23 pp | 6.75 pp |   9.97 pp   |   0.542   |
| A0                | +4.64 pp |  +3.56 pp | 1.76 pp |   2.07 pp   |   0.276   |

Cell counts: 32 fitting, 81 held-regime, 113 total. The envelope RMSE values
are unchanged and reproduce `a14_envelope_fit.csv::holdout_rmse_pp` exactly.

This matches the independent artifact review (8.40 / 9.63 / 0.456 / 0.301).

## Additional disclosure now in the manuscript

An oracle constant equal to the mean of the held-regime cells themselves would
attain 6.03 pp (IQFormer-inspired) and 6.77 pp (MCLDNN), i.e. `R^2_skill` of
-0.05 and -0.41 against that reference. The envelope's skill therefore lies in
transporting the *level* of the gain across regimes, not in resolving variation
within them. This is stated in Section V-C rather than left for a reviewer to
derive, by symmetry with the existing majority-sign baseline disclosure.

## Guard against recurrence

`paper_figures/build_paper_figures.py::fig_envelope` now computes the constant
baseline, the envelope RMSE, and the skill score from the same arrays it plots,
in percentage points throughout, and prints them at build time:

```
fig1 envelope audit: constant=-8.2670 pp, rmse_env=6.1960, rmse_const=8.4035, skill=0.4564
```

The figure annotation is therefore generated from the data rather than
transcribed, so Fig. 4 can no longer disagree with Section V-C.
