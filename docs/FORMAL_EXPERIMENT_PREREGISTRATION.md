# Formal TVT experiment preregistration

Status: prospective design; no formal cache or formal test result has been
opened.

Configuration source:
`tvt_submission/configs/formal_tvt_freeze_v1.json`.
The runner output base is `artifacts`, and the exact frozen run directory is
`artifacts/tvt_headline_1024_5seed_v1`; an existing directory is never merged
or overwritten by the hand-off script.

## 1. Scientific question and claim boundary

The formal question is whether a simulation-component-supervised
representation-allocation model improves single-view AMC under structured
interference and held-out TDL-profile factors relative to a predeclared local
reference. The primary reference is the prospectively fixed
`cssl_amc_supervised_adaptation`; it is a CSSL-AMC official-architecture
supervised adaptation, not a complete published-protocol reproduction or an
official result. The evidence is limited to 3GPP TR 38.901 TDL-profile
simulations parameterized by vehicular Doppler. It does not establish a
complete V2X trajectory, hardware transfer, field performance, waveform
reconstruction, source separation, or real-time operation.

Unanchored in-taxonomy cochannel collisions and compositional multi-jammer
records are outside the single-label headline task. They cannot be silently
added or removed after results are observed.

## 2. Locked data roles

The formal cache contains globally source-disjoint roles:

| Role | Sources | Use |
|---|---:|---|
| train | 10,000 | parameter fitting |
| validation | 2,000 | checkpoint selection and frozen model choice only |
| ID test | 5,000 | in-distribution test |
| hard interference | 5,000 | primary hard-region endpoint |
| unseen jammer | 5,000 | isolated family holdout |
| unseen speed | 5,000 | isolated mobility holdout |
| held-out channel | 5,000 | isolated TDL-profile holdout |
| combined OOD | 5,000 | joint stress test |
| clean retention | 5,000 | no-jammer guard, stratified by seen/held profile |

Every role is class-balanced across ten modulation classes. Each source has
two independently impaired views. Both views are available during training;
headline prediction uses view 1 only. Source pairing is used for statistical
alignment, not two-observation inference.

## 3. Primary endpoint and reference

The primary endpoint is hard-interference macro-F1 for `a5_vimd_full` versus
the prospectively fixed `cssl_amc_supervised_adaptation`, averaged through a
hierarchical paired bootstrap over the exact algorithm seeds
`17,29,43,71,101` and the same source clusters.

`iqformer_inspired` remains a required non-oracle comparator and Holm-family
candidate, but it is not the primary reference. It remains labeled “inspired”
because the local implementation is not an unchanged execution of the
official code. No comparator is asserted to be the strongest published or
strongest structured-interference-specific method.

The hard-region promotion gate requires at least +5.0 percentage points in
macro-F1 for A5 versus **each** frozen non-oracle baseline: A0, MCLDNN,
IQFormer-inspired, and the CSSL supervised adaptation. A positive point
estimate smaller than this threshold is reported as below the promotion target
even if a confidence interval excludes zero.

## 4. Secondary endpoints

Secondary performance endpoints are accuracy, worst supported-class recall,
NLL, and 15-bin ECE. Generalization endpoints are macro-F1 on unseen jammer,
unseen speed, held-out TDL profile, and combined OOD. The desired
generalization effect is at least +3.0 percentage points on at least two
independently isolated held factors.

Clean-retention non-inferiority is predeclared as:

- point-estimate degradation no worse than -1.0 percentage point; and
- paired 95% lower confidence bound no lower than -2.0 percentage points.

Clean retention is reported separately for seen TDL-A/C/D and held TDL-B/E
profiles before any aggregate.

Mechanism outcomes are supportive and cannot replace the performance
endpoint: mask JS, route-wise weighted correlation/MAE, occupancy,
target-energy transfer ratio, amplification share, jammer leakage, and the
oracle-conditioned spectral component ratio. That diagnostic is not waveform
SIR, SDR, or source-separation evidence.

## 5. Sample-size sensitivity

The 5,000-source test role supplies 500 independent source clusters per class.
This choice supports stable per-class recall and a class-stratified paired
bootstrap; it is not presented as a universal power guarantee for macro-F1.

For orientation only, let the paired accuracy difference per source be
`D in {-1,0,1}` and let `q` be the probability that the two classifiers are
discordant. The standard error is bounded by `sqrt(q/n)`. With `n=5,000`:

| Discordance q | SE upper bound | Approx. power for +3 pp | Approx. power for +5 pp |
|---:|---:|---:|---:|
| 0.10 | 0.45 pp | > 0.999 | > 0.999 |
| 0.25 | 0.71 pp | 0.989 | > 0.999 |
| 0.50 | 1.00 pp | 0.851 | 0.999 |
| 1.00 | 1.41 pp | 0.564 | 0.942 |

These normal-approximation values use two-sided alpha 0.05 and are a
sensitivity analysis, not a post-hoc claim about actual discordance.
Macro-F1 and seed variation are handled by the predeclared bootstrap rather
than this approximation. Source-only, seed-only, and hierarchical intervals
are all reported so that five seeds cannot be mistaken for five independent
data points.

## 6. Model, seed, and checkpoint policy

The frozen model family is A0--A7, the MCLDNN reimplementation, the
IQFormer-inspired comparator, and the CSSL-AMC official-architecture
supervised adaptation. Algorithm seeds are exactly 17, 29, 43, 71, and 101,
in that order. The resulting campaign is 11 models by five seeds, or 55 fits.
The source cache seed is separate and fixed at 20260727.

All models receive identical source identities and a unified optimizer/budget
protocol. This is labeled a unified-budget comparison, not
architecture-optimized fairness. Any architecture-specific sensitivity must
be predeclared and reported separately.

The CSSL-AMC architecture is locked to the authors' Apache-2.0 source at commit
`2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f` and therefore requires the frozen
1,024-sample raw-IQ input. It is randomly initialized, receives no external
weights, and uses the common paired-view supervised cross-entropy objective.
Its required label is “CSSL-AMC official-architecture supervised adaptation,”
not a reproduction of the published 200-epoch contrastive-pretraining plus
200-epoch fine-tuning method. It is a recent auditable AMC comparator, not a
structured-interference-specific method. A native two-stage sensitivity, if
later desired, requires a new prospective protocol before formal test access.

Checkpoint selection begins only after all active loss ramps have completed
and the full objective has remained active for the minimum full-stage period.
A formal fit is ineligible if no checkpoint has
`checkpoint_selection_eligible=1` or if it uses a final-state fallback. The
selected epoch, view, loss, label-smoothing convention, and tie rule are
serialized.

## 7. A0--A7 ablation family

Six direct hard-interference macro-F1 contrasts are predeclared as one family:

| Contrast ID | Reference | Candidate | Admissible interpretation |
|---|---|---|---|
| `teacher` | A2 | A3 | incremental fixed physical teacher |
| `multitask` | A3 | A4 | jammer/quality/orthogonality bundled intervention |
| `exact_source_contrast` | A4 | A5 | incremental exact-source cross-condition contrast |
| `full_vs_single` | A1 | A5 | full method versus single-mask composite control |
| `full_vs_dual` | A6 | A5 | composite full versus dual-route full-objective control; no route-count-only attribution |
| `bypass` | A7 | A5 | bounded additive bypass forward-path intervention |

All directions are candidate minus reference. These contrasts reuse the
already frozen A0--A7 fits and do not add training jobs; the campaign remains
55 fits.

The six effects share one class-stratified hierarchical paired bootstrap over
the exact ordered five algorithm seeds and the same aligned test-source
clusters. The family-wise 95% intervals use the non-studentized
`joint_max_absolute_centered_deviation_hierarchical_paired_bootstrap`, not
bootstrap-t. Every simultaneous lower bound must be strictly greater than
zero, and the common simultaneous critical value must be finite and strictly
positive. Marginal intervals are reported but do not decide the family gate.
Direct intervals cannot be constructed by subtracting two separate
candidate-versus-CSSL intervals.

The runner-native `ablation_paired_statistics.csv` has a frozen 33-column
schema and records a lowercase 64-hex cache digest, five seed IDs, class
stratification, verified source alignment, bootstrap hierarchy, and family
critical value. Deterministic rebuilding from the source-aligned prediction
NPZ bundles additionally verifies source IDs, labels, SNR, SIR, and profile
indices. The compact `run.json` summary binds the family and artifact rather
than duplicating those row-level fields.

## 8. Multiplicity and reporting

The hierarchical paired bootstrap is the headline inference. Exact McNemar
tests are per-seed supplemental diagnostics only; all are retained. Holm
families are declared before training and are corrected within each reported
regime/seed/candidate family. Validation is excluded from inferential tables.
No pooled multi-seed McNemar test is permitted.

Jammer auxiliary metrics are computed only for labels with positive training
support. Held-out or excluded family logits are not interpreted as trained
family recognition. A generic interference-presence task, if later added,
must be separately preregistered.

## 9. Stopping, failure, and amendment rules

There is no performance-based early termination of the formal family. A run
stops only for a recorded software/hardware failure or after every frozen
model/seed job completes. Failed jobs are rerun only with the same immutable
configuration and a disclosed operational reason.

Test metrics cannot change the architecture, loss, reference, effect
threshold, split, seed family, or reported regime. A necessary amendment
before test access creates a new version and invalidates the prior formal
designation.

If the selected method misses the primary gate, the result is reported
internally and the manuscript remains locked, is narrowed to a validated
methodological contribution, or is not submitted. Difficult classes,
low-SIR cells, or held-out factors are not deleted to repair the conclusion.

VIMD-v4 DSBN is a separate untrained screening candidate. It cannot enter the
formal family unless it first passes its prospective screening gate and a new
formal-freeze version is created before any formal test result is opened.

## 10. Current evidence and authorship boundary

As of 2026-07-28, neither the formal cache nor the 55-fit formal run has been
executed, so this preregistration contains no formal result and does not unlock
paper macros. It cannot guarantee 90% or any other acceptance probability.
Human authors must substantively author and verify the final manuscript,
validate all primary sources and claims, provide authorship/funding/conflict
metadata, make any disclosure required by the policy in force on the upload
date, and must not upload this engineering draft unchanged.
