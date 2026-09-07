# TVT V4.4 Number-Validation Waiver

**Date**: 2026-08-20
**Scope**: `paper/main.tex`, `paper/supplement.tex`, `paper/v1_audit_macros.tex`,
`paper/v41_envelope_macros.tex`, `paper/results_auto.tex` (the exact scan set of
`paper_data_layer/validate_paper_numbers.py`).
**Tooling**: `paper_data_layer/waiver_classify_v44.py` re-runs the validator's
detection rules verbatim, emits every notice (the validator itself prints only
the first 60), and assigns each notice to one of the seven waiver buckets.
**Manifest**: `paper_data_layer/outputs/v44_validator_issues.json` (436 entries,
each with file, line, literal, bucket, and provenance where applicable).

## Summary

```text
Total validator notices: 436
Expected/manual/formatting/legacy: 436
True mismatches found: 0
True mismatches corrected: 0
Remaining true mismatches: 0
Unclassified: 0
```

The previously reported count of 406 referred to the V4.3 manuscript state.
V4.4 editing (sentinel-sensitivity table, two-decimal E1 literals, caption and
wording changes) changed the notice set to 436; every notice is classified
below. The validator's total is *not* required to be zero for Technical Freeze;
the requirement is **0 unexplained numerical mismatches**, which is satisfied.

## Buckets

| # | Bucket | Count | Meaning |
|---|---|---:|---|
| 1 | manual-table literals consistent with artifact | 107 | table/prose literals equal (at scale 1 or 100, sign as printed) to a value in the frozen artifact corpus |
| 2 | rounding / formatting-only | 319 | literals that are rounded, percent-scaled, k-shorthand, or derived renderings of artifact values or protocol constants (incl. all macro-definition bodies) |
| 3 | bibliography / year / page-number strings | 0 | none detected in the scanned .tex set (bibliography lives in `main.bbl`, outside the scan) |
| 4 | legacy macros not referenced by V4.4 | 9 | audit macros defined in `v1_audit_macros.tex` but no longer used by the V4.4 body |
| 5 | figure labels / captions / definitional numbers not intended as numerical macros | 1 | the illustrative overlap example `$o=0.8$` (main.tex line 600) |
| 6 | true manuscript-artifact mismatch | **0** | — |
| 7 | unclassified | **0** | — |

## Bucket 4 legacy macros (retained for provenance, unused by V4.4)

`\EnvelopeMaxCellOverlap` (0.84; its only prose use, the "exceeds the largest
per-cell overlap" clause, was withdrawn in V4.4 with the break-even
correction), `\SidecarParamsAdd` (7294), `\EnvelopeNoCleanRmseIQ/MC`,
`\EnvelopeNoCleanSkillIQ/MC`, `\CapacityMCLDNNFiveSeedAbs`,
`\CapacityTenSeedGapRef`, `\CapacityFiveSeedGapRef`. All remain correct against
their source artifacts; they are superseded disclosures, not mismatches.

## Derived / definitional literals (bucket 2/5, provenance verbatim)

| Literal | Provenance |
|---|---|
| 3.50 / 3.5 | 100×(0.6205 − 0.585454): `e1_reference.json` MCLDNN overall minus `metrics.csv` three-seed mean |
| 1.04 | 100×(0.652258 − 0.6419): IQFormer three-seed mean minus `e1_reference.json` overall |
| 58.55 / 65.23 | three-seed means of `metrics.csv` (0.585454 / 0.652258), two decimals |
| 4.7 | A5/A0 parameter ratio (~39,500 / ~8,400 params, model configs) |
| 1.29 | coherence time T_c ≈ 0.423/f_D at 60 km/h, f_c = 5.9 GHz (protocol constants) |
| 0.0020 | 2/1024 window-spacing ratio (protocol constants) |
| 0.8 | definitional illustration of the overlap scale, not an evidence value |

## Artifact corpus used for consistency matching

`paper_data_layer/outputs/{paper_numbers.json, paper_macros.tex, e1_reference.json}`;
`paper/v1_audit_macros.tex`; `paper/v41_envelope_macros.tex`;
`analysis_zero_compute/outputs/csv/*.csv` (envelope cells/fit/holdout, a16
sentinel sensitivity, tier-2 contrasts, pareto, capacity ladders);
`analysis_zero_compute/outputs/*.json`;
`artifacts/rml2016_10a_fidelity_v3/{metrics.csv, per_snr.csv, training_history.csv, run.json}`;
`artifacts/tier2_iq_sidecar_v1/run.json`.
Matching rule: literal equals an artifact value at scale 1 or 100 (sign as
printed), or equals it after rounding to the literal's decimal count, or
k-shorthand (v/1000). Every bucket-1/2 notice therefore traces to a frozen
artifact value; the full per-notice manifest is the JSON file above.

## Freeze statement

With buckets 6 and 7 both empty, the V4.4 manuscript contains no unexplained
numerical mismatch against its artifact base. This waiver, the classifier, and
the manifest are part of the submission provenance and must be re-run after any
post-freeze edit; any new notice landing in bucket 6 or 7 blocks release.
