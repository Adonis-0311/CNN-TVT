# Lightweight Reviewer Reproduction Package

This package is the small, review-facing audit layer for the TVT manuscript.
It is intentionally not a replacement for the frozen cache, checkpoints, or
per-window prediction bundles.

## What can be checked offline

1. Inspect the frozen experiment configurations and source/provenance locks.
2. Trace every manuscript number through `paper_data_layer/outputs/paper_numbers.json`.
3. Re-run the paper-number validator and the honest-release validator from a
   repository checkout that contains the referenced artifacts.
4. Inspect all zero-compute analysis scripts and their derived CSV/JSON
   outputs.
5. Inspect Tier-2 preregistration, launchers, comparisons, and compact summary
   tables without downloading model checkpoints.
6. Verify every packaged byte with `PACKAGE_MANIFEST.json` and
   `validate_reviewer_repro_package.py`.

## Deliberately excluded

- the approximately 2 GB cache manifest and all cached arrays;
- model checkpoints;
- per-window prediction NPZ bundles;
- interrupted/superseded runs and logs; and
- author identities or disclosure answers that have not been supplied and
  signed by a human author.

The manifest records SHA-256 hashes for the heavyweight run ledgers that remain
in the controlled evidence store. Recomputing the original predictions still
requires that store and the software named by the frozen configuration. The
small package supports claim tracing and derived-table auditing, not a claim
that the full training campaign is self-contained in a few megabytes.

## Validation

Extract the ZIP, then from the directory containing the ZIP run the packaged
validator:

```text
python tvt_reviewer_reproduction_v1/tvt_submission/validate_reviewer_repro_package.py tvt_reviewer_reproduction_v1.zip
```

Expected result: JSON with `ok: true`, no missing/unexpected members, and no
hash mismatches.

## Evidence boundary

All reported experiments are controlled simulation. The package contains no
field, SDR, over-the-air, vehicular deployment, or operational measurements.
Tier-2 results are exploratory and outside the frozen confirmatory family.
The honest-release record intentionally keeps submission unlocked status false
while the clean-retention and confirmatory-family scientific gates remain
failed.
