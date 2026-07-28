# Oracle and Classical Learnability Controls

## Scope

This control path is deliberately separate from the paper's deployable model
runner. Its sole purpose is to diagnose whether the cached target modulation
is learnable, and how much mixture corruption changes a transparent
fixed-feature classifier. Every generated artifact is hard-labeled
`diagnostic_upper_control_only`.

No result from this path is eligible as:

- a deployable baseline;
- headline evidence for VIMD-Net;
- evidence that VIMD-Net recovers a clean waveform;
- source-separation performance; or
- a multi-seed/statistically powered paper claim.

## Clean received-target oracle

`CleanOracleInputDataset` wraps a paired dataset without writing to it. For
both views it maps `view["clean"]` to the classifier input `view["x"]`.
Everything else remains unchanged: label, source ID, view order, channel
profile, speed, SNR/SIR annotations, jammer/noise/receiver components, and
condition seeds.

The cached `clean` component is the target after its simulated TDL channel but
before jammer, receiver artifact, and additive noise are added. It is
unavailable to a deployed receiver. The wrapper therefore declares:

```text
oracle_clean_input = true
deployment_eligible = false
evidence_designation = diagnostic_upper_control_only
```

The oracle answers an upper-control question: *can the classifier recover the
modulation taxonomy from the target component under the chosen finite data and
training budget?* It does not estimate an achievable receiver.

## Transparent classical-feature control

`ClassicalHOCyclostationaryFeatures` accepts `[B, 2, N]` real I/Q tensors and
computes a fixed, named feature vector. It has zero trainable parameters and
uses the following families:

- magnitudes of normalized complex moments of orders 2, 3, 4, 5, 6, and 8;
- normalized fourth- and sixth-order cumulant controls;
- normalized amplitude moments;
- circular statistics of adjacent-sample phase differences;
- normalized complex autocorrelation at lags 1, 2, 4, 8, and 16;
- finite-frame cyclic-autocorrelation magnitudes at declared lags and cycle
  counts; and
- normalized spectral entropy, flatness, concentration, centroid, spread, and
  edge-band energies.

The signal is centered and unit-power normalized before statistics are
computed. The features are therefore invariant to a positive global amplitude
scale and a global carrier phase rotation (within numerical tolerance). All
statistics are evaluated in float64/complex128 and guarded against zero-power
input.

This is a **local classical-feature control**, not a reproduction of a
specific HOC/cyclostationary paper. The reusable neural control adds only a
fixed-budget linear or 64-unit MLP head. The lightweight diagnostic runner
instead uses a closed-form ridge head on train-standardized features, with a
fixed regularization value and an unregularized intercept.

## Diagnostic runner

Run from the project root:

```powershell
python experiments/run_learnability_controls.py `
  --cache-root standards/cache_screening_v1 `
  --output artifacts/learnability_controls `
  --run-id screening_v1_seed20260727 `
  --seed 20260727 `
  --epochs 12 `
  --device cpu
```

The runner refuses to overwrite an existing run and verifies every cache file
checksum by default. It fits:

1. mixture-input HOC/cyclostationary features plus ridge;
2. clean-oracle HOC/cyclostationary features plus ridge; and
3. a short clean-oracle A0 backbone control selected on the validation split.

It evaluates validation and held-out TDL-channel splits, retains source IDs for
the two view rows, and writes:

- `run.json`: complete protocol, provenance, metrics, clean-minus-mixture
  diagnostic deltas, cache digest, and source hashes;
- `metrics.csv`: compact control-by-split metrics; and
- `checksums.json`: SHA-256 hashes of the generated JSON and CSV.

For a completed v1 run, the companion audit adds row-preserving condition
slices and two fixed-physical-teacher route probes:

```powershell
python experiments/append_learnability_control_audit.py `
  --run-root artifacts/learnability_controls/screening_v1_seed20260727_v2 `
  --device cpu
```

Because the first runner version did not save row probabilities, the companion
recomputes the same deterministic closed-form HOC ridge solution with the
recorded cache, regularization, and feature schema. It aborts unless aggregate
accuracy and macro-F1 reproduce the original artifact to `1e-12`. This is not
iterative retraining and performs no hyperparameter selection. It then persists
the reconstructed probabilities for future audits.

`condition_slices.csv` reports mixture and clean-oracle controls on identical
rows by jammer family, four disjoint SIR bins, and requested overlap profile.
Cochannel interference has an explicit focus row. These are descriptive
small-sample slices without confidence intervals or multiplicity-adjusted
inference, so they cannot support paper conclusions.

`route_oracle_metrics.csv` additionally asks whether the fixed physical teacher
retains class information in `M_s` and `M_s + 0.5M_o`. The teacher requires
ground-truth components. A carefully audited weighted overlap-add inverse maps
the masked STFT to I/Q solely for the same HOC feature probe. With the current
`center=False` periodic-Hann lattice, sample zero has no window coverage and is
deterministically zero-filled; coverage and round-trip error are recorded in
`control_audit.json`. These outputs are not reconstructed waveforms.

The mixture HOC control consumes deployment-available input, but its artifact
still remains diagnostic because it is single-seed, fixed-budget screening.
The two clean controls are always non-deployable.

## Interpretation gate

A large clean-minus-mixture gap is evidence that corruption obscures
class-discriminative structure for this feature/control budget. A small gap
does not prove the mixture is easy, because both controls may be underfit. A
weak clean A0 score is a red flag for dataset size, taxonomy, model budget, or
training sufficiency; it must be resolved before treating main-model scores as
algorithmic evidence.

Only formal, pre-registered, multi-seed experiments on factor-isolated caches
may enter the paper's performance tables.

## Executed screening diagnostic (2026-07-27)

The audited companion run is
`artifacts/learnability_controls/screening_v1_seed20260727_v2`, bound to cache
digest
`e219930800a24844146087b6dfa7b2fa2daf1c61aaaf2ab5b0158c4c79a80b9a`.
All six generated-file checksums pass.

On held-out TDL profiles, fixed HOC/cyclostationary features plus ridge reached
12.70% accuracy / 11.67% macro-F1 from the mixture and 71.29% / 71.03% from
the unavailable clean component. The 58.59 percentage-point accuracy gap is a
learnability diagnostic, not a performance comparison. The 12-epoch
clean-oracle A0 control reached only 17.77% / 13.85%; its training accuracy was
still rising, so this short neural run is underfit and must not be interpreted
as an oracle ceiling.

The physical-teacher route probe reached 30.66% / 27.29% for `M_s` and 24.02%
/ 22.21% for `M_s + 0.5M_o` on the same held-out split. Thus the target-
dominant allocation retains more class information than the raw mixture for
this fixed feature head, while naively returning half of the overlap mass
reduces that gain. This observation motivates learned overlap routing but does
not validate a deployable mask.

The held-out cochannel slice contains 90 view rows (minimum six per represented
class): mixture accuracy is 12.22%, clean-oracle accuracy is 73.33%, and the
gap is 61.11 points. It is flagged small-sample. For the better-supported
`SIR <= -5 dB` slice (181 rows, minimum 13 per class), the corresponding values
are 6.63%, 69.61%, and 62.98 points. These descriptive slices localize the
identifiability bottleneck; they do not carry inferential or paper-claim status.
