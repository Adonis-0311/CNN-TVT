# TVT v2 M4–M7 execution and evidence contract

The historical v1 freeze remains a read-only audit record. It is not edited,
relabelled, or promoted. All future confirmatory execution uses
`tvt_submission/configs/formal_tvt_freeze_v2.json`.

## Prospective sample-size and stopping justification

The v2 freeze byte-binds
`docs/TVT_V2_PROSPECTIVE_STATISTICAL_JUSTIFICATION.md`. The formal loader and
preflight fail if that document is absent or its SHA-256 drifts.

The design makes no unsupported 80%/90% power claim because no eligible v2
pilot estimate of algorithm-seed dispersion or paired source-cluster
dependence existed before execution. It uses a prospective precision and
decision-rule justification instead:

- the hard-interference family is exactly three contrasts with a joint
  family-wise simultaneous 95% interval;
- all 12 models use the same 10 frozen algorithm seeds, and every
  confirmatory test regime has 5,000 source clusters;
- the A5-minus-A0 primary comparison has a one-percentage-point macro-F1
  SESOI for “practically material” wording, while statistical success still
  requires its simultaneous lower bound to exceed zero;
- the post-run evidence must report interval widths and a frozen
  SESOI-relative precision classification; and
- test outcomes cannot trigger added seeds, sources, epochs, models, or a
  second confirmatory attempt.

Fit-level stopping remains validation-only under the fixed 30-epoch/patience-8
configuration. At study level the exact 120-fit grid must complete without
interim efficacy/futility looks. Technical failures may only be retried with
the same model, seed, cache, and configuration and must remain auditable.

## Training scale and learning curve

The formal v2 training cache contains 100,000 source sequences. Before the
confirmatory run, A0 and A5 must be trained on the fixed 10k/30k/100k grid with
five predeclared seeds. The three caches share the same master seed and exact
validation source IDs. Every epoch records training and validation macro-F1
and loss. The formal scale remains 100k regardless of the curve, so the curve
cannot be used for post-hoc scale or hyperparameter selection.

## Confirmatory and exploratory inference

The confirmatory hard-interference family has exactly three jointly resampled
contrasts and ten algorithm seeds:

1. A5 minus A0;
2. margin teacher minus A3′ proportional teacher; and
3. A5 minus A6.

The simultaneous interval is the existing non-studentized maximum absolute
centered-deviation hierarchical paired bootstrap interval. Every other
ablation contrast is explicitly exploratory and cannot support a family-wise
positive claim. The cache master seed is `20260727`; the bootstrap base seed
is `20260803`.

## OOD opportunity calibration

For each unseen-jammer, unseen-speed, and held-out-channel axis, the gate
jointly reports A0 ID-to-axis degradation, the paired A5-minus-A0 gain, their
95% intervals, and `gain - 0.25 * max(A0 degradation, 0)`.

An axis is consequential only when the A0 degradation lower bound exceeds
0.5 pp. It then requires a positive recovery-margin lower bound. Otherwise it
licenses no improvement claim and uses the predeclared -1 pp noninferiority
floor. This replaces the unscaled fixed 3 pp rule.

## Receiver robustness

Three source-disjoint, test-only hard-interference splits are materialized:
10-bit ADC after window-RMS AGC, 12-bit ADC after the same AGC, and independent
target/jammer CFO in ±0.006 plus integer timing offsets in ±3 samples before
mixing. Quantization residual is retained in `receiver_artifact`, so the
component sum remains auditable. These axes use the same opportunity-calibrated
branch logic, but their nominal reference is the unmodified
`hard_interference` split, not `id_test`. The stress and nominal policies share
the same SNR/SIR, jammer, speed, and TDL support, avoiding a severity
distribution confound.

## Clean retention

The clean-control gate is derived rather than copied from the generic runner.
It compares A5 with the preregistered CSSL-AMC reference on the source-paired
`clean_retention` predictions, separately for seen-profile A/C/D and
held-profile B/E. Each stratum uses the same hierarchical algorithm-seed and
class-stratified source-cluster bootstrap as the headline comparison. Both the
point estimate must be at least -1 pp and the 95% lower confidence bound at
least -2 pp; both strata must pass.

## Directional mechanism test

The six-family hard-split occupancy/gain test reuses the frozen
`OCCUPANCY_GAIN_MECHANISM_PROTOCOL`. For each source, only the single inference
view (`view1`) is used: periodic-Hann complex-STFT cells at or above -20 dB of
that source view's jammer-cell maximum define source-level occupancy, and
family occupancy is the arithmetic mean across sources in that family. Within
each family source subset, paired A5-minus-A0 macro-F1 is calculated for each
algorithm seed and then averaged arithmetically across seeds to define family
gain. The predicted direction is non-increasing gain with occupancy, tested by
family-level negative Spearman, exact 6! one-sided permutation, a hierarchical
bootstrap interval, and the frozen inversion check. The result must be
reported. Failure blocks a positive mechanism claim but does not block an
honest null-result submission.

## Fail-closed execution

`tvt_submission/run_v2_after_gpu_free.ps1` is the future default queue. It
validates the freeze, builds immutable caches, completes and hashes the
learning-curve evidence, launches the exact formal grid, and derives
`v2_scientific_release_gate.json`.

The gate remains closed if a run/cache digest differs, the learning curve is
absent, a fit/checkpoint/prediction is missing, confirmatory family membership
drifts, a clean-retention stratum fails, or a required OOD/receiver CI branch
fails. Historical screening
artifacts and manually typed performance values cannot enter this path.

`passed` in `v2_scientific_release_gate.json` means that the scientific
evidence contract passed. It never sets `submission_unlocked`: a separate,
hash-bound v2-to-paper macro promotion and public-build validation must still
complete. This distinction prevents a completed model grid from being mistaken
for a submission-ready manuscript.
