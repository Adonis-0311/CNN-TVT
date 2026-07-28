# Public table release contract

Status: implemented on the paper/build side; internal-review placeholder only  
Contract date: 2026-07-28  
Build-gate schema: `vimd_amc.tvt.paper_build_gate.v2`  
Release-lock schema: `vimd_amc.tvt.release_lock.v2`

## Purpose

The three public result tables in `paper/main.tex` contain no hand-entered
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
VIMDLatencyP50
VIMDLatencyP95
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
  `VIMDLatencyDevice`, and all 71 numeric commands;
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
- an old lock schema, a changed macro file, or incomplete per-command
  provenance.

No training or performance-value fabrication is part of this contract.
