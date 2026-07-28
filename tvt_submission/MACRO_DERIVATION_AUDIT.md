# Automatic result-macro derivation audit

`generate_macro_values.py` is the supported producer of the macro-value
manifest. It accepts only `--run-json` and `--output`; it has no argument for a
model name, regime, metric, confidence interval, or performance value.

The generator does not build a cache, load a checkpoint, train a model, or
write inside the formal run directory.

## Runner artifact audit

The final standard runner provides four relevant evidence layers:

| Artifact | Relevant content | Generator check |
|---|---|---|
| `run.json` and per-fit `result.json` | eligibility, exact model/seed matrix, per-regime metrics, A5 mechanism and complexity records | `validate_source_run` first; result files must equal the embedded records |
| `metrics.csv` | one scalar-metric row per model, seed, and evaluation regime | exact row matrix, unique headers/keys, finite cells, cache digest, equality with `run.json` |
| `headline_paired_statistics.csv` | hierarchical seed/source paired differences and intervals | exact candidate/regime matrix, finite cells, seed IDs/count, bootstrap contract, cache digest, and point-estimate equality with metrics and NPZ predictions |
| `models/*/predictions_*.npz` | probabilities, labels, source IDs, SNR/SIR, split, cache digest | exact array schema, shapes, bounds, finite required metadata, probability normalization, source alignment, and recomputed accuracy/macro-F1 |

Required CSV keys are never inferred by column position. Duplicate columns,
duplicate model/seed/regime rows, missing rows, blank cells, `NaN`, infinity,
or disagreement between evidence layers abort generation before the output
path is created.

## Macro rules

| Macro | Deterministic rule | Manifest source |
|---|---|---|
| `StrongestBaseline` | unique maximum mean hard-interference macro-F1 over A0, the MCLDNN reimplementation, the IQFormer-inspired local baseline, and the CSSL-AMC official-architecture supervised adaptation; the winner must also be the predeclared paired reference | `metrics.csv` |
| `HardMacroFOneGain` | A5-versus-reference hierarchical macro-F1 difference on `hard_interference`, multiplied by 100 | `headline_paired_statistics.csv` |
| `HardMacroFOneCI` | corresponding hierarchical 95% macro-F1 interval, multiplied by 100 | `headline_paired_statistics.csv` |
| `HeldoutJammerGain` | A5-versus-reference hierarchical macro-F1 difference on `unseen_jammer`, multiplied by 100 | `headline_paired_statistics.csv` |
| `HeldoutChannelGain` | A5-versus-reference hierarchical macro-F1 difference on `heldout_channel`, multiplied by 100 | `headline_paired_statistics.csv` |
| `FeatureSIRGain` | arithmetic mean across formal seeds of A5 `mechanism.counterfactual_tf_sir_gain_db` on the held-out-channel probe | `run.json` |
| `VIMDParameters` | exact positive integer A5 parameter count, required to be identical across seeds | `run.json` |
| `VIMDLatency` | median across seed-level A5 `latency_ms_p50` values, with one identical recorded device required | `run.json` |

The strongest-baseline rule is intentionally strict. A tie, a missing
comparator, or a winner different from the predeclared reference fails. The
latter case cannot reuse the existing interval because the runner emits
hierarchical pairs only against its predeclared reference.

The CSSL-AMC row is a 2025 official-architecture supervised adaptation under
the common budget. Including it in this local selection does not turn it into
a complete CSSL reproduction, official result, or
structured-interference-specific method, and `StrongestBaseline` must not be
expanded into “strongest published method.”

## Removed legacy SDR macro

The prior placeholder contract listed `SDRGain`, but:

1. the runner emits no source-to-distortion-ratio estimate;
2. prediction NPZ files contain class probabilities, not a reconstructed
   waveform from which SDR could be computed;
3. `paper/main.tex` never references `SDRGain`; and
4. the manuscript explicitly states that no SDR claim is made.

Consequently, populating it from `counterfactual_tf_sir_gain_db` would confuse
TF feature SIR with waveform SDR. The minimal fail-closed patch removes this
unused macro from `validate_release.RESULT_MACROS` and the placeholder file.
Because the exact macro set changed, the manifest schema is explicitly bumped
from `vimd_amc.tvt.macro_values.v1` to
`vimd_amc.tvt.macro_values.v2`; old nine-record manifests fail validation.
If an SDR claim is later desired, it requires a prospectively defined SDR
endpoint, runner-native waveform/metric artifacts, and a new formal run. It
must not be backfilled into the current run.

## Commands

Generate a deterministic manifest outside the immutable run directory:

```powershell
& "D:\Python\python.exe" .\tvt_submission\generate_macro_values.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --output .\tvt_submission\formal_macro_values.json
```

Then run the release preflight:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_release.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --paper-root .\paper `
  --macro-values .\tvt_submission\formal_macro_values.json
```

Both commands exit nonzero on any rejected invariant. Generation leaves
`submission_unlocked=false`; only the separately requested release write may
create `paper/results_auto.tex` and `paper/release_lock.json`. The writer emits
`\EligibleLockedResults` with the fixed value `eligible_locked_formal_run`;
the parser and schema-v2 release lock both verify it before a public build can
be revalidated.
