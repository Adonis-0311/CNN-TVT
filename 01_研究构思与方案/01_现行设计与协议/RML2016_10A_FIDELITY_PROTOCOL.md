# E1: RML2016.10a Comparator-Fidelity Check

## Purpose and boundary

E1 is a post-review external comparator-fidelity check.  It evaluates only the
local PyTorch `MCLDNNReimplementation` and `IQFormerInspiredClassifier` on the
public RadioML 2016.10a benchmark.  It does not evaluate VIMD-Net, the I/Q
sidecar, the overlap--SIR envelope, or any claim of external superiority.  It
is outside the sealed V4R confirmatory family and cannot revise a sealed gate.

## Dataset and source integrity

Input: `data/RML2016.10a_dict.pkl`, a 220-key pickle containing 11 modulation
classes, 20 SNR levels (-20 to 18 dB in 2-dB steps), 1,000 windows per
class--SNR stratum, and raw `float32 [2, 128]` I/Q windows.  The runner records
the file SHA-256, ordered class list, SNR grid, split indices, source code
hashes, package versions, and GPU environment in its immutable run directory.

## Public-protocol fidelity

**Reference-source discovery (2026-08-19).**  The only machine-readable
published MCLDNN and IQFormer results on RML2016.10a are the per-SNR accuracy
table `testA.xlsx` in the official IQFormer repository
(`WestdoorSad/IQFormer`, benchmark table for AMC-NET / FEA-T / MCLDNN /
PET-CGDNN / IQFormer).  It is a machine-readable xlsx produced under the
per-stratum 80/20 then 75/25 `train_test_split`, each `random_state=233`
(= 600/200/200 per stratum; 11 classes; full -20..18 dB grid).  The official
MCLDNN repository (wzjialang/MCLDNN) uses a different `np.random.seed(2016)`
set-based partition and publishes performance only as a figure, which the E1
rule below forbids reading; its released Keras weights could not be faithfully
re-loaded for evaluation (Keras 2.2.4/TF1 CuDNNLSTM checkpoint, all tested
gate/bias conversions give chance-level output).

**E1 realignment decision.**  Because the only machine-readable published
reference table covers *both* comparators under the `random_state=233`
protocol, E1 evaluates *both* local comparators under that same split.  This
makes ``our implementation vs. published/reference`` a same-protocol
comparison for both rows of Table II.  MCLDNN no longer uses the original
repo's 2016 partition (which has no machine-readable reference); the change is
recorded in the runner docstring and the run directory.  Neither branch adds
AGC, data augmentation, resampling, or an input-length change; both retain the
public 11-class set and full SNR grid.

The local MCLDNN is a PyTorch reimplementation of a Keras reference and the
local IQFormer is explicitly architecture-inspired rather than binary-identical
to the public release.  E1 therefore tests whether these local comparators
recover the published-protocol performance range, not exact checkpoint
equivalence.

## Execution and decision rule

The runner first executes one complete algorithm seed per model (`2016` for
MCLDNN and `233` for IQFormer) with each model's public optimizer family and
checkpoint criterion, then, if the first seed is within the fidelity bands
below, a three-seed follow-up (`17`, `29`, `43` for both models; the split is
fixed at `random_state=233`).  It reports all-SNR and per-SNR
accuracy/macro-F1.  The machine-readable reference trace is
`paper_data_layer/outputs/e1_reference.json` (parsed from `testA.xlsx`:
MCLDNN 62.05%, IQFormer 64.19% all-SNR overall, mean over the 20 SNR bins).
The interpretation thresholds are predeclared: <2 pp difference is close
fidelity; 2--4 pp is the same performance range/trend; >4--5 pp requires
preprocessing, split, and topology investigation before the result can appear
as a comparator-fidelity anchor.

No numerical "published/reference" value is inferred from a figure or entered
by hand.  If a verified numeric external trace is unavailable, E1 may document
the exact protocol and local performance but must be reported as an incomplete
fidelity comparison, not as a passed check.
