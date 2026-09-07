# TVT v2 prospective statistical justification

## Status and scope

This document was frozen before any formal v2 cache, learning-curve, or
confirmatory artifact existed. It justifies the fixed design and the wording
licensed by that design. It is not a retrospective power calculation and does
not claim a conventional 80% or 90% power level.

No eligible v2 pilot estimate of algorithm-seed variance, source-cluster
correlation, or paired discordance was available at freeze time. An analytical
power number would therefore require invented nuisance parameters or an
independence model that does not match macro-F1 under paired multi-seed
evaluation. The prospective design instead freezes:

- the estimand and confirmatory family;
- a smallest effect size of interest for the primary method comparison;
- the algorithm seeds and source-cluster counts;
- simultaneous interval construction and multiplicity control;
- fit-level and study-level stopping rules; and
- a precision/sensitivity report that must be derived after execution without
  changing the design.

The machine-readable counterpart is
`prospective_statistical_design` in
`tvt_submission/configs/formal_tvt_freeze_v2.json`. The freeze records this
file's SHA-256, and the formal loader refuses a missing or byte-drifted copy.

## Estimand and confirmatory family

The outcome is absolute macro-F1 on the source-disjoint
`hard_interference` split. Each contrast is candidate minus reference:

1. `full_vs_backbone`: A5 minus A0;
2. `margin_vs_proportional_teacher`: A3 minus A3-prime; and
3. `tri_vs_dual_route`: A5 minus A6.

The family contains exactly these three contrasts. Its 95% intervals use the
predeclared joint maximum-absolute-centered-deviation hierarchical paired
bootstrap. Algorithm seed is the outer resampling level. Within a selected
seed, class-stratified source IDs are resampled as clusters, preserving paired
model predictions and all views belonging to a source. Validation examples
are excluded from inference.

Statistical success for a contrast requires its family-wise simultaneous 95%
lower confidence bound to be strictly greater than zero. All three contrasts
must satisfy that rule for the confirmatory-family gate to pass. Per-seed
McNemar results and all non-family contrasts are supplemental or exploratory;
they cannot replace the simultaneous family result.

## Smallest effect size of interest

For the primary complete-method comparison `full_vs_backbone`, the
prospectively chosen smallest effect size of interest (SESOI) is an absolute
macro-F1 gain of `0.01`, i.e. one percentage point. This is a decision and
wording threshold, not an effect estimated from pilot results and not a claim
that one percentage point has universal operational value. It is aligned with
the already frozen one-percentage-point scale used for the nonconsequential
OOD noninferiority tolerance and clean-retention point tolerance.

A “practically material primary gain” statement is licensed only when:

- the `full_vs_backbone` simultaneous 95% lower bound is strictly above zero;
  and
- its point estimate is at least `0.01`.

If the simultaneous interval excludes zero but the point estimate is below
`0.01`, the result may be described as statistically resolved but below the
preregistered materiality threshold. It must not be described as a materially
important primary gain. The two structural confirmatory contrasts retain the
directional simultaneous-interval rule; their magnitudes must be reported,
but this document does not invent a common operational SESOI for a teacher
functional-form contrast or a composite route contrast.

## Fixed sample and seed configuration

The confirmatory grid has 12 frozen models and the same 10 algorithm seeds for
every model: `17, 29, 43, 71, 101, 131, 173, 211, 257, 307`. This yields 120
required model-seed fits. Missing fits cannot be silently dropped, and failed
seeds cannot be replaced by new seed values.

Each confirmatory evaluation regime has 5,000 source clusters. The
hard-interference contrast family therefore compares all models on the same
5,000 source IDs for each of the 10 algorithm seeds. The fixed training and
validation counts are 100,000 and 2,000 source sequences, respectively. The
learning-curve prerequisite separately fixes 10k, 30k, and 100k training
sources for A0 and A5 over five predeclared seeds; it cannot change the 100k
formal scale.

The justification for this configuration is precision-oriented:

- ten matched algorithm seeds expose between-training variation instead of
  treating one optimization path as deterministic;
- 5,000 source clusters per test regime provide class-stratified,
  source-disjoint support while keeping paired comparisons at the physical
  source unit rather than falsely counting correlated views as independent;
- pairing candidate and reference predictions on identical sources removes
  avoidable between-source variation from each contrast; and
- 10,000 joint bootstrap draws are fixed for the simultaneous family
  calculation, with a bootstrap seed distinct from the cache master seed.

These design features improve precision but do not prove a numerical power
level. Between-seed dispersion can dominate source-level uncertainty, and it
was not known prospectively.

## Precision and sensitivity reporting

After the one fixed execution, every confirmatory contrast must report the
point estimate, marginal 95% interval, simultaneous 95% interval, and
simultaneous interval width. For `full_vs_backbone`, the evidence record must
also report the interval width relative to the `0.01` SESOI and classify the
result using these frozen cases:

| Case | Required interpretation |
|---|---|
| simultaneous lower bound `> 0` and point gain `>= 0.01` | statistically positive and meets the preregistered materiality threshold |
| simultaneous lower bound `> 0` and point gain `< 0.01` | statistically positive but below the materiality threshold |
| interval contains both `0` and `0.01` | inconclusive at the prespecified precision |
| simultaneous upper bound `< 0.01` | effects at or above the materiality threshold are not supported |
| simultaneous upper bound `<= 0` | no positive primary gain is supported |

This achieved-precision classification is descriptive evidence from the
frozen run, not permission to change its sample size. No seed, source, model,
contrast, bootstrap draw count, or decision threshold may be added after
viewing formal outcomes.

## Stopping and failure rules

There are two distinct stopping levels:

1. **Fit level.** Each fit uses at most 30 epochs and the frozen
   validation-only checkpoint/early-stopping configuration: patience 8,
   minimum full-stage duration 3 epochs, and fixed mask/contrastive ramps.
   Test metrics never select a checkpoint or stop a fit.
2. **Study level.** The study completes the exact 120-fit confirmatory grid.
   There is no interim examination of test contrasts, no efficacy or futility
   stop, and no optional extension of seeds, sources, epochs, or models.

A process interruption, corrupt write, out-of-memory termination, or other
technical failure may be retried only with the same model, seed, cache,
configuration, and clean output destination. The failure and retry must remain
in the execution audit. A technical retry is not a new replicate. If the
frozen design is found infeasible before formal result inspection, it must be
replaced by a newly versioned prospective freeze; v2 cannot be amended and
then represented as the original confirmatory study.

## Limitations

- The design has no model-based prospective power percentage because the
  required paired, hierarchical nuisance parameters were unavailable.
- Ten algorithm seeds support an explicit between-training resampling level
  but do not represent every possible optimizer or hardware realization.
- The 5,000 clusters quantify uncertainty within the frozen simulated factor
  distributions. They do not establish field, SDR, blockage, trajectory, or
  complete V2X generalization.
- Macro-F1 is nonlinear, and paired views share a source; treating views or
  class cells as independent binomial observations would overstate precision.
- The learning curve can reveal scale behavior but cannot trigger post-hoc
  sample-size re-estimation, hyperparameter selection, or a second
  confirmatory attempt.

Accordingly, a wide or non-significant interval is an admissible scientific
outcome. It must be reported as uncertainty or lack of support, not repaired
by an outcome-dependent design change.
