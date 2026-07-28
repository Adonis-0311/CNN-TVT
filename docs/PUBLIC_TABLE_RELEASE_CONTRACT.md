# Public table release contract

Status: implemented on the paper/build side; internal-review placeholder only  
Contract date: 2026-07-28  
Build-gate schema: `vimd_amc.tvt.paper_build_gate.v2`  
Release-lock schema: `vimd_amc.tvt.release_lock.v2`

## Purpose

The four public result tables in `paper/main.tex` contain no hand-entered
performance cell and no literal `pending` or `generated` substitute. Every
cell is backed by an atomic command defined in `paper/results_auto.tex`.
The checked-in file defines the complete interface as em-dash placeholders
and deliberately omits `EligibleLockedResults`, so internal review remains
compilable while a public build remains locked.

An eligible release must replace all placeholders with values derived from
one formal run. `tvt_submission/validate_paper_build.py` accepts that release
only when the exact macro file is bound by a portable schema-v2 release lock
and every reportable value has one provenance record.

## Fixed sentinels and text values

- `EligibleLockedResults` must equal `eligible_locked_formal_run`.
- `ResultSource` names the locked formal run and cache.
- `PrimaryReference` must equal
  `CSSL-AMC official-architecture supervised adaptation`.
- `VIMDLatencyDevice` records the one frozen measurement device.

`PrimaryReference` is a predeclared paired-comparison anchor. It is not a
claim that CSSL-AMC is the strongest published or structured-interference
method.

## Atomic numeric interface

Every numeric value is a unit-free canonical finite decimal. Units and
brackets are fixed in `main.tex`, never embedded in a value.

### Hard-interference headline table

For each model token

```text
AZero, MCLDNN, IQFormer, CSSL, AFive
```

and metric token

```text
Accuracy, MacroFOne, WorstRecall, NLL, ECE
```

the required command is

```text
HeadlineHard<Model><Metric>
```

This is exactly 25 commands. Accuracy, macro-F1, and worst-class recall are
percentages in `[0,100]`; NLL is nonnegative; ECE is a raw fraction in
`[0,1]`.

### A0--A7 hard-interference ablation table

The A0 and A5 macro-F1 means reuse
`HeadlineHardAZeroMacroFOne` and `HeadlineHardAFiveMacroFOne`.  The other six
five-seed means are:

```text
HeadlineHardAOneMacroFOne
HeadlineHardATwoMacroFOne
HeadlineHardAThreeMacroFOne
HeadlineHardAFourMacroFOne
HeadlineHardASixMacroFOne
HeadlineHardASevenMacroFOne
```

The six predeclared candidate-minus-reference contrasts use the prefixes

```text
AblationTeacher
AblationMultitask
AblationExactSourceContrast
AblationFullVsSingle
AblationFullVsDual
AblationBypass
```

and each prefix requires `Gain`, `CILow`, and `CIHigh`.  These 18 interval
commands plus the six missing A0--A7 means add exactly 24 numeric commands.
Gain and interval values are percentage points. The runner-native CSV retains
macro-F1 means, differences, and bounds on raw proportion scale; the macro
generator multiplies them by 100 exactly once. `CILow` and `CIHigh` are
simultaneous bootstrap bounds targeting family-wise coverage from one joint,
non-studentized, max-absolute-centered-deviation hierarchical paired
bootstrap. They are not ordinary marginal intervals, bootstrap-t/max-t
bounds, or an exact finite-sample family-wise guarantee. The common draw
resamples algorithm seeds and class-stratified test-source clusters once and
applies that same resample to every model and contrast.

The six directions are A3--A2 (teacher), A4--A3 (the bundled
jammer/quality/orthogonality intervention), A5--A4 (exact-source contrast),
A5--A1 (full versus single-mask composite), A5--A6 (tri-route versus
dual-route composite), and A5--A7 (bounded-bypass intervention).  The A5--A6
row does not support a route-count-only causal claim.

The paper-build gate parses the public `table`/`table*` environment that
contains `\label{tab:ablations}`. Each of the 24 new commands above must be
consumed exactly once inside that environment. A command mentioned only in
public prose, an internal-only branch, a different table, or a decoy
post-table line does not satisfy the Table III contract.

### Factor-isolated OOD/clean table

For each regime token

```text
Hard, UnseenJammer, UnseenSpeed, HeldoutChannel, CombinedOOD, CleanACD, CleanBE
```

the five commands are

```text
Regime<Regime>Reference
Regime<Regime>AFive
Regime<Regime>Gain
Regime<Regime>CILow
Regime<Regime>CIHigh
```

This is exactly 35 commands. Reference and A5 values are macro-F1
percentages. Gain and confidence limits are percentage points. The gate
requires `CILow <= Gain <= CIHigh`.

### Mechanism and complexity table

The required commands are:

```text
MechanismMaskJS
MechanismThirdRouteWeightedCorrelation
MechanismTargetTransferRatio
MechanismTargetAmplificationShare
MechanismJammerLeakage
MechanismThirdRouteSpearman
MechanismThirdRoutePermutationP
OracleSpectralRatioGain
VIMDParameters
VIMDLatencyPFifty
VIMDLatencyPNinetyFive
```

The mechanism sources are, respectively, five-seed audited values of
`mask_js`, `overlap_uncertainty_route_weighted_correlation`,
`target_energy_transfer_ratio_mean`,
`target_energy_transfer_ratio_amplification_share`, `jammer_leakage`,
`oracle_vs_predicted_overlap_spearman`, `overlap_permutation_p_value`, and
`counterfactual_tf_sir_gain_db`. The amplification share is exported as a
percentage. `OracleSpectralRatioGain` is an oracle-conditioned spectral
component-ratio diagnostic in dB; it is not waveform SIR gain, SDR,
source separation, or a real-world SIR estimate.

Latency is batch-one P50/P95 on `VIMDLatencyDevice`. The operator must isolate
the recorded device before the eligible measurement. The table makes no
CPU-versus-GPU or cross-device comparison.

## Release-lock binding

Release mode requires the exact schema-v2 key set and verifies:

- `submission_unlocked=true` and the exact sentinel name/value;
- formal designation `headline_formal_tvt_evidence`;
- non-placeholder run ID and lowercase SHA-256 cache digest;
- SHA-256 bindings for `run.json`, `results_auto.tex`, the macro-value
  manifest, and the source gate;
- `artifact_audit.passed=true`;
- exact SHA-256 equality between the audited `results_auto.tex` bytes and
  `release_lock.json`;
- `macro_provenance` containing exactly `PrimaryReference`,
  `VIMDLatencyDevice`, and all 95 numeric commands (97 provenance-bound
  commands total);
- for every provenance record, one safe run-relative source artifact, its
  SHA-256, and a non-placeholder derivation.

`ResultSource` is itself determined by the locked run identity. All other
visible result values are individually represented in `macro_provenance`.

## Explicit rejection cases

The public gate rejects:

- a missing, extra, duplicate, blank, dash, nonnumeric, nonfinite, or
  unit-bearing numeric command;
- the retired `StrongestBaseline`, `HardMacroFOneGain`,
  `HardMacroFOneCI`, `HeldoutJammerGain`, `HeldoutChannelGain`,
  `FeatureSIRGain`, or `VIMDLatency` interfaces;
- a literal `pending` or `generated` result cell in `main.tex` or the
  extracted public PDF;
- a confidence interval with reversed bounds or a point estimate outside it;
- an incomplete headline model/metric grid or missing OOD/clean stratum;
- an incomplete A0--A7 mean set, a missing/reordered ablation contrast, or an
  ablation simultaneous lower bound that is at or below zero either in raw
  percentage points or after the frozen two-decimal public rendering (so a
  positive raw bound rendered as `+0.00` remains release-ineligible);
- any of the 24 new A0--A7 commands absent from, duplicated within, or moved
  outside the `table`/`table*` containing `\label{tab:ablations}`, even when a
  decoy public-prose mention keeps the command present elsewhere;
- an old lock schema, a changed macro file, or incomplete per-command
  provenance.

The complete interface is 97 provenance-bound commands, 98 non-sentinel
result commands after adding `ResultSource`, and 99 released commands after
adding `EligibleLockedResults`.

No training or performance-value fabrication is part of this contract.
