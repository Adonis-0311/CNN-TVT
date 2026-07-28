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
& .\tvt_submission\run_local.ps1 -Python "D:\Python\python.exe"
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
no performance-value arguments and writes a strict JSON object. The following
is an abridged structural example; the exact keys and counts stated below
govern:

```json
{
  "schema_version": "vimd_amc.tvt.macro_values.v3",
  "run_id": "<exact run_id>",
  "cache_digest": "<exact cache_digest>",
  "run_json_sha256": "<SHA-256 of the final run.json>",
  "scientific_release_gate": {
    "passed": true,
    "hard_gain_pp_each_nonoracle_baseline": {
      "a0_backbone": 5.0,
      "mcldnn_reimplementation": 5.0,
      "iqformer_inspired": 5.0,
      "cssl_amc_supervised_adaptation": 5.0
    },
    "hard_ablation_family": {
      "passed": true,
      "family_id": "hard_macro_f1_ablation_family_v1",
      "regime": "hard_interference",
      "metric": "macro_f1",
      "direction": "candidate_minus_reference",
      "confidence_level": 0.95,
      "multiplicity_method": "joint_max_absolute_centered_deviation_hierarchical_paired_bootstrap",
      "simultaneous_ci95_low_strictly_greater_than_pp": 0.0,
      "contrasts": {
        "teacher": {
          "reference": "a2_tri_no_teacher",
          "candidate": "a3_tri_teacher",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        },
        "multitask": {
          "reference": "a3_tri_teacher",
          "candidate": "a4_tri_teacher_mtl",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        },
        "exact_source_contrast": {
          "reference": "a4_tri_teacher_mtl",
          "candidate": "a5_vimd_full",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        },
        "full_vs_single": {
          "reference": "a1_single_mask",
          "candidate": "a5_vimd_full",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        },
        "full_vs_dual": {
          "reference": "a6_dual_full",
          "candidate": "a5_vimd_full",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        },
        "bypass": {
          "reference": "a7_vimd_no_residual",
          "candidate": "a5_vimd_full",
          "gain_pp": 0.02,
          "marginal_ci95_low_pp": 0.01,
          "marginal_ci95_high_pp": 0.03,
          "simultaneous_ci95_low_pp": 0.01,
          "simultaneous_ci95_high_pp": 0.04,
          "passed": true
        }
      }
    },
    "ood_gain_pp": {
      "unseen_jammer": 3.0,
      "unseen_speed": 3.0,
      "heldout_channel": 0.0
    },
    "ood_pass_count": 2,
    "clean_noninferiority": {
      "clean_retention_seen_acd": {
        "gain_pp": -1.0,
        "ci95_low_pp": -2.0
      },
      "clean_retention_held_be": {
        "gain_pp": -1.0,
        "ci95_low_pp": -2.0
      }
    },
    "mechanism_means": {
      "mask_js": 0.0,
      "overlap_uncertainty_route_weighted_correlation": 0.0,
      "target_energy_transfer_ratio_mean": 0.0,
      "target_energy_transfer_ratio_amplification_share": 0.0,
      "jammer_leakage": 0.0,
      "oracle_vs_predicted_overlap_spearman": 0.0,
      "overlap_permutation_p_value": 0.0,
      "counterfactual_tf_sir_gain_db": 0.01
    }
  },
  "macros": {
    "PrimaryReference": {
      "value": "<TeX-safe non-placeholder value>",
      "source_artifact": "run.json",
      "derivation": "<auditable predeclared-reference rule>"
    }
  }
}
```

The numeric values above only illustrate a threshold-edge shape. They are not
formal results and may
never be copied into a real manifest: the generator must derive the exact
values from the named artifacts, and the release validator reconstructs them
before accepting the file.

The `macros` object must contain exactly 97 provenance records:

- one `PrimaryReference`, fixed to the CSSL-AMC official-architecture
  supervised adaptation;
- 25 `HeadlineHard{Model}{Metric}` records for five models by five metrics;
- 35 `Regime{Regime}{Field}` records for seven regimes by five fields;
- six absolute hard-region macro-F1 records for A1, A2, A3, A4, A6, and A7
  (A0 and A5 are already in the headline group);
- 18 records giving `Gain`, `CILow`, and `CIHigh` for each of the six direct
  ablation contrasts;
- eight mechanism records; and
- `VIMDParameters`, `VIMDLatencyPFifty`,
  `VIMDLatencyPNinetyFive`, and `VIMDLatencyDevice`.

The percentile tokens are spelled out because the controlled TeX parser accepts
letter-only command names; `P50` and `P95` are not macro-name tokens in this
contract. Ordinary p50/p95 latency statistic labels remain valid in the runner
artifacts.

Every record has exactly `value`, `source_artifact`, and `derivation`.
`source_artifact` is relative to the formal run directory and must exist.
The manifest may not contain blank, dash, `pending`, `generated`, NaN,
unavailable, or other placeholder values. The generator cross-checks
`run.json`, both relevant CSV layers, and the source-aligned prediction NPZ
bundles before it writes. A human author must still verify each numerical
derivation. There is no baseline-winner record and no post-hoc winner
selection. IQFormer-inspired remains a required non-oracle comparator, while
the CSSL supervised adaptation remains the prospectively fixed primary
reference; neither role supports a strongest-published-method or
structured-interference-specific claim.

The top-level `scientific_release_gate` is mandatory. It is reconstructed from
the formal artifacts and must pass the preregistered hard-baseline,
hard-ablation, two-of-three OOD, clean-stratum noninferiority, and mechanism
checks. Canonical mismatch between the supplied manifest and the reconstructed
manifest aborts release.

The hard-ablation gate is one six-contrast family, not two unadjusted point
comparisons. It fixes A3-A2 (teacher), A4-A3
(jammer/quality/orthogonality bundle), A5-A4 (exact-source cross-condition
contrast), A5-A1 (full versus single-mask), A5-A6 (full versus dual-route
composite control), and A5-A7 (bounded additive bypass). Every simultaneous
95% lower bound must be strictly greater than zero. A4-A3 cannot be decomposed
into single-loss causal claims, and A5-A6 cannot be labeled a route-count-only
effect.

All six contrasts share the exact ordered seeds `17,29,43,71,101` and one
class-stratified hierarchical paired bootstrap over aligned test-source
clusters. The family-wise method is the non-studentized
`joint_max_absolute_centered_deviation_hierarchical_paired_bootstrap`; its
common critical value must be finite and strictly positive. The
`run.json` family summary binds the family and artifact, while the generator
and release validator deterministically rebuild the 33-column
`ablation_paired_statistics.csv` from source-aligned NPZ bundles to verify the
cache digest, seeds, source/SNR/SIR/profile alignment, stratification, and
critical value. No new fits are required; the formal campaign remains 55 fits.

The old v2 eight-record contract and its winner, feature-SIR, compact-gain, and
single-latency names are rejected. Its former SDR placeholder also remains
removed because neither the runner nor the prediction bundles contain an SDR
endpoint. The oracle-conditioned spectral component ratio diagnostic is not
waveform SIR or SDR. See `MACRO_DERIVATION_AUDIT.md`.

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
`results_auto.tex`. The manifest contributes 97 provenance records; the writer
adds `ResultSource`, producing 98 non-sentinel commands, and then adds the
sentinel for 99 commands in the released file. The internal placeholder
intentionally omits the sentinel. The macro manifest is schema v3, while the
release lock remains schema v2 and records the sentinel name/value and the
complete macro-file hash; parsing and existing-release validation reject a
missing, changed, or hand-forged sentinel.

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
