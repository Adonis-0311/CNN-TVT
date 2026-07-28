# Formal result-macro release protocol

This protocol is fail-closed. A completed experiment is not, by itself, an
unlocked submission. The standard runner must first serialize all formal
evidence gates as passed, and `validate_release.py` must then bind every paper
macro to the exact `run.json` and a named run artifact.

## Strict evidence boundary

The release validator accepts only the exact cache designation
`headline_formal_tvt_evidence` under policy `vimd-evidence-gate-v2`.
Screening caches, other headline aliases, diagnostic runs, incomplete
model/seed matrices, source-mutated runs, fallback checkpoints, and missing or
nonfinite result cells are rejected.

The frozen formal cache and training campaign are defined by
`configs/formal_tvt_freeze_v1.json` and printed by:

```powershell
& "D:\Python\python.exe" .\tvt_submission\run_local.ps1
```

This is a dry preflight unless `-Execute` is supplied. Do not use the
screening cache as a substitute, and do not change the frozen seed, model,
checkpoint, split, or multiplicity settings after inspecting formal results.

## Macro-value manifest

Before writing paper macros, run the automatic generator:

```powershell
& "D:\Python\python.exe" .\tvt_submission\generate_macro_values.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --output .\tvt_submission\formal_macro_values.json
```

Do not type or copy performance values into this file. The generator accepts
no performance-value arguments and writes a strict JSON object with this
schema:

```json
{
  "schema_version": "vimd_amc.tvt.macro_values.v2",
  "run_id": "<exact run_id>",
  "cache_digest": "<exact cache_digest>",
  "run_json_sha256": "<SHA-256 of the final run.json>",
  "macros": {
    "StrongestBaseline": {
      "value": "<TeX-safe non-placeholder value>",
      "source_artifact": "metrics.csv",
      "derivation": "<auditable selection and calculation rule>"
    }
  }
}
```

The `macros` object must contain exactly these eight records:

- `StrongestBaseline`
- `HardMacroFOneGain`
- `HardMacroFOneCI`
- `HeldoutJammerGain`
- `HeldoutChannelGain`
- `FeatureSIRGain`
- `VIMDParameters`
- `VIMDLatency`

Every record has exactly `value`, `source_artifact`, and `derivation`.
`source_artifact` is relative to the formal run directory and must exist.
The manifest may not contain blank, dash, `pending`, `generated`, NaN,
unavailable, or other placeholder values. The generator cross-checks
`run.json`, both relevant CSV layers, and the source-aligned prediction NPZ
bundles before it writes. A human author must still verify each numerical
derivation. In particular, `StrongestBaseline` is populated only when the
predeclared local comparator family---A0, MCLDNN, IQFormer-inspired, and the
CSSL-AMC official-architecture supervised adaptation---has one unique
hard-interference macro-F1 winner and that winner is the paired reference.
This is a local audited selection, not a strongest-published-method or
structured-interference-specific claim.

The former unused `SDRGain` placeholder was removed because neither the
runner nor the prediction bundles contain an SDR endpoint and the manuscript
explicitly makes no SDR claim. The counterfactual TF-SIR mechanism diagnostic
is not waveform SDR. See `MACRO_DERIVATION_AUDIT.md`.

## Preflight, write, and revalidation

Validate the run and macro manifest without writing:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_release.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --paper-root .\paper `
  --macro-values .\tvt_submission\formal_macro_values.json
```

Write `paper/results_auto.tex` and `paper/release_lock.json` only after the
preflight passes:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_release.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --paper-root .\paper `
  --macro-values .\tvt_submission\formal_macro_values.json `
  --write
```

An eligible write defines
`\newcommand{\EligibleLockedResults}{eligible_locked_formal_run}` in
`results_auto.tex`. The internal placeholder intentionally omits this command.
The schema-v2 release lock records the sentinel name/value and the complete
macro-file hash; parsing and existing-release validation reject a missing,
changed, or hand-forged sentinel.

Revalidate an existing release before compilation or packaging:

```powershell
& "D:\Python\python.exe" .\tvt_submission\validate_release.py `
  --run-json .\artifacts\tvt_headline_1024_5seed_v1\run.json `
  --paper-root .\paper
```

Any failure exits nonzero. A failed write attempt does not replace the
placeholder macro file and does not create an unlock file. Replacing an
existing release additionally requires the explicit
`--replace-existing-release` flag.
