# Vehicular AMC evidence governance

This directory contains scenario-neutral governance mechanisms adapted for the
VIMD-Net TVT work. It does not contain imported datasets, trained weights,
reported numbers, or source-scenario assumptions.

The intended order is:

1. Fill the freeze template, replace every placeholder, and set
   `status=frozen_before_results` before running.
2. Record the exact configuration file SHA-256 in the run launcher.
3. Keep train, validation, calibration, and test access roles separate.
4. Run the data-QA checks and save their evidence.
5. Publish each attempt once with a checksum-closed manifest.
6. Pass `assess_release`; any absent, malformed, dirty, or mismatched evidence
   denies release.

These helpers do not launch training and do not assert any experimental result.
