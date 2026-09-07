# Baseline and Ablation Audit

This note fixes the model names, provenance boundaries, and objective ladder
used by both experiment runners.  It is a protocol record, not a performance
claim.

## MCLDNN

- Local name: `mcldnn_reimplementation`
- Required paper label: **MCLDNN literature-faithful PyTorch
  reimplementation**
- Paper: https://doi.org/10.1109/LWC.2020.2999453
- Authors' code: https://github.com/wzjialang/MCLDNN
- Audited revision: `f1093eea5a04ba6f7fc5297171ffbae5c9853f93`

The local model retains the public three-stream topology:

1. a 50-channel joint-I/Q `2 x 8` convolution with same padding;
2. separate 50-channel causal length-8 I and Q convolutions;
3. a 50-channel `1 x 8` same-padded fusion convolution;
4. a 100-channel `2 x 5` valid fusion convolution;
5. two 128-unit LSTM layers;
6. two 128-unit SELU layers with dropout 0.5; and
7. a class-logit layer.

For 11 classes the local topology has 406,199 parameters, matching the
canonical public PyTorch port bundled in the audited IQFormer repository.
The authors' released Keras checkpoint is not loaded and is not
binary-compatible.  Therefore, neither "official weights" nor "official
result" is an admissible label.

## IQFormer

- Local name: `iqformer_inspired`
- Required paper label: **IQFormer-inspired local baseline**
- Paper: https://doi.org/10.1109/TCCN.2024.3485118
- Official code: https://github.com/WestdoorSad/IQFormer
- Audited revision: `7ee6ac949551b24d45f218762cab919e0cb6b4f9`

The local implementation checks the public RML2016 configuration item by item:

- stage depths `[2, 3, 2]`;
- widths `[64, 64, 64]`;
- MLP ratio 4;
- one additive-attention block at the end of each stage;
- a two-layer, 32-unit-per-direction bidirectional LSTM;
- fusion/LSTM dropout 0.2;
- drop-path rate 0;
- outer layer scaling disabled; and
- 355,049 parameters for 11 classes.

The internal Blackman STFT reproduces the real component produced by the
public SciPy call (`nperseg=31`, `noverlap=30`, `nfft=128`, first 32 frequency
bins) to numerical tolerance.  It is calculated in the model rather than in a
dataset worker.  The local implementation also uses batch-safe dimension
handling and has not passed an official-checkpoint equivalence test.  These
differences prohibit an exact-reproduction label.

## CSSL-AMC

- Local name: `cssl_amc_supervised_adaptation`
- Required paper label: **CSSL-AMC official-architecture supervised
  adaptation**
- Paper: https://doi.org/10.1109/TWC.2025.3532438
- Official code: https://github.com/dumingyang20/CSSL-AMC-Pytorch
- Audited revision: `2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f`
- Source/hash lock: `tvt_submission/sources/cssl_amc_2025.lock.json`

The 1,024-sample local implementation retains the official three-convolution
noise estimator, raw-I/Q-plus-estimated-noise concatenation, `[2, 2]`
one-dimensional residual stages, `128 x 512 -> 128` readout, and
`128 -> 64 -> classes` classifier. The local ten-class topology has 8,631,948
parameters. Structural tests bind the layer shapes, fixed sample length,
registry name, source commit, and license hash.

The formal runner does not load the released pretraining state and does not
execute the official momentum-encoder contrastive-pretraining stage followed
by fine-tuning. It instead applies the common paired-view supervised
classification objective and frozen TVT optimizer budget. This makes CSSL-AMC
a recent auditable AMC architecture comparator, but not a reproduction of the
complete published method, an official CSSL result, or a
structured-interference-specific comparator.

## A0--A7 controlled ladder

| ID | Runner name | Architecture or objective change |
|---|---|---|
| A0 | `a0_backbone` | Shared spectral classifier, no masks |
| A1 | `a1_single_mask` | One learned target mask, CE only |
| A2 | `a2_tri_no_teacher` | Tri-mask architecture, CE only, residual retained |
| A3 | `a3_tri_teacher` | A2 plus fixed physical mask supervision |
| A4 | `a4_tri_teacher_mtl` | A3 plus jammer/quality tasks and branch orthogonality |
| A5 | `a5_vimd_full` | A4 plus XCC; full tri-mask model with residual |
| A6 | `a6_dual_full` | Full objective and residual with two masks/two task branches |
| A7 | `a7_vimd_no_residual` | A5 with the applied residual removed |

A2--A5 and A7 instantiate the same `VIMDNet`; their parameter counts must be
identical for a given `ModelConfig`.  A6 uses `DualMaskVIMDNet` with both
modulation and jammer branches.  Its fixed dual teacher retains the
modulation-dominant tri route and collapses the jammer-dominant plus
overlap/unexplained routes into one non-target route.  The loss family and
every enabled component are stored explicitly in `run.json` and each model's
`result.json`.  No Python model-type check is permitted to turn a disabled
component back on.

A6 is therefore a composite two-route control, not a pure route-count
intervention: it changes both route count/capacity and the policy for assigning
overlap mass.  Any causal interpretation requires a separately pre-registered
overlap-allocation sensitivity analysis.

A5 and A7 share weights and parameterization; A7 returns an applied residual
of exactly zero while retaining the otherwise identical residual head so that
the comparison isolates the forward-path intervention.

The fixed teacher is always evaluated in float32 with automatic mixed
precision disabled.  Only the normalized target is converted to the student
mask dtype.

## Comparison boundary

Both runners hold data realizations, source splits, optimizer, schedule,
checkpoint rule, and reported seed fixed across models.  They report
parameters, convolution/linear/recurrent MACs, STFT operation estimates, and
measured latency separately.

The models are not parameter-matched and the current shared schedule is not an
architecture-specific hyperparameter sweep.  A final paper claim needs both:

1. the fixed-protocol comparison, which controls training choices; and
2. a preregistered tuning-sensitivity analysis for each literature baseline.

Smoke-cache accuracy and significance outputs are pipeline checks only.
