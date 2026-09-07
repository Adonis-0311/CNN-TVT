# VIMD-v3 Shared-IQFormer Route Diagnostic

## Locked scope

`VIMDIQFormerRouteNet` is a diagnostic candidate outside A0--A7. It was
motivated by directional IQFormer-inspired behavior in a source-mutated run;
those prior numbers are invalid and are not imported into this comparison.

The candidate and `IQFormerRawOnlyControl` use the same
`IQFormerInspiredClassifier` encoder and classification head. The candidate
has exactly one encoder instance, called once on the received mixture and once
on the student modulation route. It does not duplicate encoder weights.

## Route and inverse audit

The student tri-mask lives on the project's full complex STFT lattice. Its
modulation route is transformed back to I/Q by explicit differentiable
overlap-add:

- inverse FFT uses the orthonormal convention paired with
  `ComplexSTFT(normalized=True)`;
- synthesis multiplies the same periodic Hann window;
- normalization uses the exact overlap sum of squared windows;
- samples without reliable overlap support are zero-filled and reported by a
  coverage mask and count;
- output length always matches the received frame.

The raw received path bypasses this route. A condition gate in `[0.10, 0.90]`
fuses raw and routed embeddings, guaranteeing a non-zero raw safety
coefficient. Validation, held-out evaluation, strict checkpoint reload, and
deployment inference call only `forward(received_iq)`.

## Pre-registered diagnostic gate

One seed on bounded class-balanced subsets of `cache_screening_v1` passes only
if all conditions hold:

1. validation accuracy is not below the same-encoder raw-only control;
2. held-out-channel accuracy improves by at least 2 percentage points;
3. held-out-channel macro-F1 improves by at least 2 percentage points;
4. the deployment-only route audit is non-collapsed.

The route audit requires a non-boundary, source-varying gate; non-trivial
raw/route embedding and waveform differences; varying route weights; and at
least 95% reliable overlap-add coverage. Failure closes this candidate without
promotion to screening or the manuscript.

All outputs are labeled `diagnostic_non_evidence_do_not_cite`.

## Result and strict gate decision

The source tree remained unchanged for the complete run, the saved candidate
contained no teacher keys, and strict reload reproduced logits with zero
error. The immutable result is
`artifacts/diagnostic_iqformer_route_v3/iqformer_route_v3_20260728_003538.json`
with SHA-256
`9983696b1c4a4e5716834500a8b0b95695fa725535671f8cc83010aafd62d02f`.

| Model | Validation acc. / F1 | Held-out acc. / F1 |
|---|---:|---:|
| shared-IQFormer raw-only control | 20.00% / 15.01% | 17.50% / 11.13% |
| VIMD-v3 shared-IQFormer route | 23.00% / 16.78% | 17.50% / 13.44% |

The candidate gained 3.00 percentage points in validation accuracy and 2.31
points in held-out macro-F1, but held-out accuracy changed by exactly 0.00
points. Because the locked rule required all four conditions, the
held-out-accuracy condition failed and the outcome is
`vimd_v3_tiny_success_threshold_not_met`.

This is not a near-pass or a positive result. Identical held-out accuracy means
the route did not increase the number of correctly classified held-out
sources. The macro-F1 movement only indicates a more balanced distribution of
the same total number of correct decisions under this one-seed diagnostic.
That trade cannot support robustness or performance claims.

## Failure attribution

The deployment-only route audit passed, which rules out the simplest
implementation failures:

- gate mean/std were 0.6414/0.00631 within the bounded fusion interval;
- raw/route embedding cosine was 0.3567 and relative embedding difference was
  0.9243;
- route-weight standard deviation was 0.1111;
- route/raw waveform relative difference was 0.6552;
- overlap-add reliable coverage was 98.05%, with five explicitly zero-filled
  edge samples.

Thus the failure is not explained by an identity route, constant mask, gate
boundary collapse, missing route gradient, length drift, or teacher leakage.
The route changes the representation substantially, but the change does not
add held-out correct decisions. The five OLA edge samples remain a documented
limitation, although the raw safety path and 98% coverage make them
insufficient to claim a causal explanation.

The raw control has 354,984 parameters. The candidate has 371,263, adding
16,279 parameters (+4.59%): 2,376 spectral-context, 6,880 environment,
6,990 tri-mask, and 33 gate parameters. The IQFormer encoder and classifier
remain one shared 354,984-parameter instance. Parameter sharing avoids a
double-capacity comparison, but inference still evaluates that encoder twice,
so computational cost is materially higher and was not justified by an
accuracy gain.

## Disposition

VIMD-v3 is rejected at the diagnostic gate:

- do not add it to the screening or headline suites;
- do not report its metrics in the manuscript;
- do not describe macro-F1 movement as an improvement;
- retain its code and artifact only as an auditable negative result;
- do not modify A5 or the locked A0--A7 registry.

## At most one next candidate: domain-specific normalization

No further training is authorized by this document. One low-cost,
independently falsifiable candidate may be considered later:
`VIMDIQFormerRouteNormNet`.

Hypothesis: calling the same BatchNorm-based IQFormer encoder sequentially on
raw and routed inputs mixes two distinct activation distributions in shared
running statistics, limiting held-out accuracy even though convolution,
LSTM, attention, and classifier weights are useful for both paths.

The candidate would share every non-normalization weight with VIMD-v3 but keep
separate raw/route BatchNorm affine parameters and running statistics. The
audited encoder has 16 BatchNorm layers with 1,510 affine parameters, so this
change adds exactly 1,510 trainable parameters (0.43% of the raw encoder) and
no second convolution/LSTM/attention stack.

Before any run, lock the same cache digest, seed 17, 320/100/200 source
subsets, 12-epoch budget, optimizer, teacher loss, and checkpoint policy used
above. It passes only if all conditions hold:

1. validation accuracy is at least the current VIMD-v3 value;
2. held-out accuracy is at least 2 percentage points above current VIMD-v3;
3. held-out macro-F1 is not below current VIMD-v3;
4. the existing route non-collapse and teacher-free strict-reload audits pass;
5. source fingerprint remains unchanged and the implementation audit confirms
   exactly one shared set of all non-BatchNorm weights.

Failure of any condition permanently closes the domain-normalization
hypothesis. This proposal is not implemented and no training has been started.
