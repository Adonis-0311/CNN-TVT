# VIMD-v4 Shared-Weight DSBN Diagnostic Preregistration

## Status and scope

This document freezes the sole permitted post-v3 candidate before any
VIMD-v4 training:

`VIMDIQFormerRouteDSBNNet`

No VIMD-v4 result exists at preregistration time. The candidate is an internal
falsification diagnostic outside the immutable paper A0--A7 suite. It does not
replace A5, cannot enter screening automatically, and is never manuscript
evidence under this one-seed protocol.

The v3 negative result motivates one narrow hypothesis only: sequentially
encoding raw mixtures and student routes through shared BatchNorm statistics
may mix distinct activation domains. V4 changes that one property and nothing
else in the IQFormer encoder.

## Frozen architecture

The received mixture and student modulation route pass through the same
IQFormer-inspired encoder object. V4 instantiates exactly one copy of every:

- convolution and depthwise convolution;
- LSTM parameter;
- additive-attention query, key, projection, final projection, and global
  attention weight;
- feed-forward linear/pointwise weight;
- layer-scale parameter;
- classifier weight.

Each of the encoder's 16 BatchNorm locations instead has two named domains:
`raw` and `route`. Only BatchNorm affine parameters, running means, running
variances, and batch counters are domain-specific. Both domains start from
identical initialization.

The audited accounting is:

| Item | Count |
|---|---:|
| Single-domain IQFormer encoder and classifier | 354,984 trainable parameters |
| BatchNorm affine parameters per domain | 1,510 |
| V4 DSBN encoder and classifier | 356,494 |
| V4 addition over the v3 shared encoder | 1,510 |
| Extra stored route-domain running-state values | 1,526 |
| Shared encoder calls per forward pass | 2 |

Thus V4 adds 0.43% trainable capacity to the 354,984-parameter shared encoder;
it does not create a second feature extractor. Like v3, it still pays for two
encoder evaluations at inference.

The spectral tri-mask, inverse overlap-add, bounded gate `[0.10, 0.90]`, raw
safety path, training teacher, losses, and optimization schedule are unchanged
from v3. The only candidate-side architectural delta is domain-specific
BatchNorm.

## Deployment contract

The public forward signature is:

```text
forward(received_iq_mixture)
```

Teacher masks, clean components, jammer components, SNR, SIR, channel labels,
and condition metadata are forbidden at inference. The checkpoint must contain
no teacher keys. A fresh V4 instance must accept the checkpoint with
`strict=True` and reproduce fixed-input logits with maximum absolute error no
greater than `1e-7`.

Each deployment forward must call the one shared IQFormer stem exactly twice:
once under the `raw` BN domain and once under `route`. All domain selectors
must be restored after each call, so an encoder call without an explicit
domain fails closed.

## Frozen data and budget

The diagnostic entry point is
`diagnostics/run_iqformer_route_v4_diagnostic.py`. Importing it does not train,
and command-line execution requires the explicit
`--execute-preregistered-diagnostic` acknowledgement.

The script refuses a dataset whose digest or sample length differs from:

| Field | Frozen value |
|---|---|
| Cache | `standards/cache_factor_screening_1024_v1` |
| Cache SHA-style digest | `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3` |
| I/Q sample length | 1024 |
| Algorithm seed | 17 |
| Train subset | 32 sources/class, 320 total |
| Validation subset | 10 sources/class, 100 total |
| Held-out-channel subset | 20 sources/class, 200 total |
| Epochs | 12 |
| Batch size | 32 |
| Learning rate | `3e-4` |
| Weight decay | `1e-2` |
| Mask start | epoch 2 |
| Contrastive start | epoch 5 |

Subset indices are class balanced, deterministically selected, and recorded by
SHA-256. V3 and V4 are initialized under the same seed and trained/evaluated on
the identical subsets. The implementation audit must show exact equality of
all initial non-BatchNorm tensors between the two candidates.

This is a one-seed candidate falsification, not a 3-seed screening run or a
5-seed headline experiment.

## All-required success gate

V4 passes only if every condition below is true:

1. validation accuracy is not below the paired v3 run;
2. held-out-channel accuracy is at least 2 percentage points above v3;
3. held-out-channel macro-F1 is not below v3;
4. the existing deployment route non-collapse audit passes;
5. strict reload passes and fixed-input maximum logit error is at most `1e-7`;
6. inference accepts only the received mixture and the checkpoint has no
   teacher keys;
7. the shared encoder is called exactly twice per forward;
8. all non-BatchNorm tensors are single-copy and initially identical to v3;
9. V4 adds exactly 1,510 trainable parameters over v3;
10. the source tree remains unchanged for the full execution.

No averaging, compensating condition, near-pass interpretation, or metric
substitution is allowed. In particular, macro-F1 movement cannot compensate
for a held-out accuracy gain below 2 points.

The inherited route non-collapse audit requires all of:

- mean route gate in `[0.15, 0.85]`;
- gate standard deviation at least `0.005`;
- mean relative raw/route embedding difference at least `0.02`;
- route-weight standard deviation at least `0.02`;
- mean route/raw waveform relative difference at least `0.05`;
- minimum reliable overlap-add coverage at least 95%.

## Disposition rule

If any gate item fails, the DSBN hypothesis is closed:

- do not tune its thresholds, seed, subset, epoch budget, or loss schedule;
- do not add another normalization variant;
- do not promote it to screening;
- do not report its diagnostic metrics in the manuscript;
- do not modify A0--A7 or reinterpret A5.

If all items pass, the result only authorizes a separate, results-before-frozen
3-seed screening proposal. It does not itself support a paper claim.

## Pre-run structural verification

Only non-training tests are authorized before execution:

```text
python -m unittest tests.test_iqformer_route_v4 -v
```

These tests must cover single-copy shared weights, independent BN state,
gradient flow, two encoder calls, nontrivial route behavior, exact parameter
accounting, strict serialization/reload, received-mixture-only inference, the
1024-cache lock, and the untouched A0--A7 registry.
