# Auxiliary and Mechanism Evidence Contract

This note fixes the meanings, support rules, and admissible interpretations of
the auxiliary-task and routing-mechanism metrics. It is a protocol record, not
a performance claim.

## FP32 mask divergence

`jensen_shannon_mask_loss` performs normalization, probability flooring,
logarithms, and reduction in float32 with automatic mixed precision explicitly
disabled. The returned value is a differentiable float32 scalar; casting a
float16 or bfloat16 student mask to float32 does not detach it from autograd.

The implementation uses natural logarithms, so a valid per-cell
Jensen--Shannon divergence is bounded by `ln(2)`. Exact zeros are allowed.
An all-zero route vector is converted to a uniform distribution after
flooring, rather than producing `log(0)` or `0/0`. Shape, floating-point type,
device agreement, finite loss, gradient propagation, extreme probabilities,
and nested CPU autocast are covered by tests.

This guard prevents AMP underflow from corrupting the mask objective. It does
not make a nonfinite upstream model output acceptable; such an upstream fault
must still fail evidence QA.

## Auxiliary-task evaluation

`vimd_amc.evaluation.auxiliary_task_metrics` evaluates one named dataset view
and emits a JSON-native dictionary carrying:

- the split, seed, view, sample count, and fixed jammer threshold;
- jammer multi-label micro/macro F1;
- jammer micro/macro AUROC;
- per-jammer F1 and AUROC with positive and negative support;
- SNR, SIR, and Doppler MAE in physical units; and
- a validity audit of every `quality_mask` entry.

Strict `json.dumps(result, allow_nan=False)` succeeds. An unavailable metric
uses `{"status": "unavailable", "value": null, "reason": ...}` instead of a
NaN placeholder.

### Jammer support rules

The decision rule is `sigmoid(logit) >= threshold`; the default threshold is
0.5 and must be fixed before test evaluation.

Micro F1 is unavailable when the evaluated split contains no positive jammer
label. Macro F1 averages only classes with positive ground-truth support and
reports the number of included and total classes. This avoids silently scoring
an absent jammer class as a perfect or zero class.

AUROC is computed by the rank/Mann--Whitney identity with average ranks for
ties. A per-class AUROC is unavailable unless that class has both positive and
negative examples. Macro AUROC averages only classes meeting that condition.
Micro AUROC applies the same rule to all flattened label decisions.

The output also audits nonfinite and nonbinary multi-hot labels. Invalid labels
make jammer metrics unavailable; they are never thresholded into apparently
valid evidence.

### Physical quality scales

The quality head predicts normalized SNR, SIR, and Doppler in that order.
Physical MAE requires the normalization multiplier for each component. The
evaluator accepts either:

1. an explicit `quality_denormalization` argument; or
2. a dataset `quality_normalization` configuration attribute; or
3. a `quality_normalization` mapping in the dataset manifest, either at the
   top level or inside `configuration`/`synthesis_configuration`.

The schema is:

```python
{
    "snr_db": {"scale": 20.0, "offset": 0.0, "unit": "dB"},
    "sir_db": {"scale": 20.0, "offset": 0.0, "unit": "dB"},
    "doppler_hz": {
        "scale": MAX_DOPPLER_HZ,
        "offset": 0.0,
        "unit": "Hz",
    },
}
```

A positive numeric value may replace each `{scale, unit}` object when the
default unit and zero offset are appropriate. Explicit arguments override
manifest entries component by component.

No scale is inferred from observed target values or from an undocumented
implementation constant. Consequently, legacy caches without this manifest
schema must pass explicit scales. Otherwise physical MAE is marked
unavailable, even though normalized predictions exist. This is intentional:
guessing a multiplier would create a plausible-looking but unauditable dB/Hz
claim.

When a view exposes `snr_db`, `sir_db`, or `doppler_hz` directly, that raw
physical value is the MAE target; only the prediction is denormalized. This
avoids hiding target clipping. The output also reports the discrepancy between
the denormalized quality target and the independent physical target. When no
independent physical value exists, both prediction and target use the same
affine scale/offset. If a future target uses a nonlinear transform, this
interface must be extended rather than approximated with an affine mapping.

### `quality_mask` validity

The mask audit records:

- nonfinite entries;
- entries outside `[0, 1]`;
- nonbinary entries;
- enabled targets that are nonfinite;
- enabled predictions that are nonfinite; and
- enabled support for each physical component.

Only finite pairs enabled by a valid binary mask contribute to MAE. In
particular, no-jammer samples can disable SIR without contaminating SIR error.

## Routing mechanism evidence

`mechanism_metrics` retains the historical flat keys needed by old artifact
readers, but its version-2 names remove two scientific ambiguities.

### Target-energy transfer ratio

For target component spectrum `S` and learned modulation-path multiplier `W_m`,
the diagnostic is

```text
R_target = mean(|W_m S|^2) / mean(|S|^2).
```

Because `W_m = M_s + lambda M_o + rho`, `R_target` may exceed one. Therefore
the admissible name is **target-energy transfer ratio**, not “signal
retention.” The evaluator reports its mean, maximum, number of samples above
one, and amplification share. The same statistics are reported for clean-only
samples when such samples exist.

The old numeric aliases `signal_retention` and
`clean_only_signal_retention` remain in the dictionary. The accompanying
`deprecated_metric_aliases` mapping marks them as deprecated. A value above
one is evidence of forward-path gain, not source recovery, denoising, or
physically conserved energy.

### Two constituents of the overlap teacher

The fixed overlap/uncertainty teacher is the sum of:

- `unexplained_fraction`: receiver artifact plus noise power share; and
- `signal_jammer_ambiguity`: target--jammer equal-dominance mass.

The evaluator no longer reports only their sum. For each constituent it
records:

- oracle energy-weighted occupancy;
- direct correlation between the learned overlap route and the constituent;
- direct energy-weighted MAE;
- an oracle-conditioned post-hoc attribution occupancy;
- attribution correlation; and
- attribution MAE.

The direct comparison answers whether the one learned overlap route follows a
particular physical source of uncertainty. Its MAE is not a mask-supervision
loss because the learned route targets the sum of both constituents.

The oracle-conditioned attribution partitions the learned overlap route in
proportion to the two oracle constituents. It is useful for error accounting,
but it is not an independently predicted mask and must never be presented as
one. This limitation is embedded in every constituent record.

Historical `oracle_unexplained_share` and
`oracle_signal_jammer_ambiguity_share` keys remain as deprecated aliases for
the newly named oracle occupancies.

### SNR/SIR strata

Optional SNR and SIR interior edges can be supplied through
`snr_strata_edges_db` and `sir_strata_edges_db`. Boundaries must be finite and
strictly increasing. The resulting half-open strata cover
`[-inf, first)`, intermediate intervals, and `[last, inf)`.

No default or post-hoc quantile boundary is selected. If edges are absent, the
stratified result is explicitly unavailable. This keeps the paper protocol
responsible for preregistering interpretable bins. Infinite SIR values from
no-jammer samples are excluded and counted rather than assigned to a finite
bin.

Each populated stratum reports transfer/amplification statistics, predicted
overlap occupancy, both oracle-constituent occupancies, occupancy-level
Spearman correlations when supported, and constituent-specific direct MAE.

## Evidence boundary

These metrics describe behavior on the supplied deterministic samples. They
do not establish causal separation, waveform reconstruction, calibrated
confidence, standards-level V2X compliance, or generalization beyond the
locked split. Headline use still requires the full seed protocol, held-out
conditions, multiplicity control, confidence intervals, and immutable run
artifacts defined elsewhere in the project.
