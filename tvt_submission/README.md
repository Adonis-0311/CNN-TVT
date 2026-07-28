# TVT convergence package

This directory is the controlled hand-off for the IEEE Transactions on
Vehicular Technology manuscript. It is deliberately fail-closed: the current
paper remains an internal evidence-lock manuscript, and no screening,
diagnostic, oracle, or source-mutated number is promoted as a paper result.

## Implemented, pending final regression

- The method, A0--A7 ablation registry, source-disjoint nine-split protocol,
  statistical plan, and simulation-only claim boundary are implemented.
- The 1024-sample screening cache passed its independent integrity audit:
  `standards/cache_factor_screening_1024_v1.audit.json`.
- The 2025 CSSL-AMC official encoder/classifier architecture is pinned by
  commit, source hashes, and Apache-2.0 license and registered as the bounded
  `cssl_amc_supervised_adaptation` formal comparator. It is not a complete CSSL
  reproduction or a structured-interference-specific method.
- The latest internal compile-check produces an 8-page IEEE two-column PDF with
  placeholder results. It is not a release build; the older 7-page
  `paper/build/main.pdf` is stale relative to the current sources.
- Formal cache generation and the five-seed experiment are specified below
  but intentionally not launched by Codex.

## What remains machine-executed

The paper-evidence path uses `run_local.ps1` for the formal cache/experiment
hand-off. The one-shot VIMD-v4 DSBN path in `run_candidate_local.ps1` is
optional diagnostic work and is not required for the current paper. Without
`-Execute`, both scripts perform only preflight checks and print the exact
long-running commands. Do not change model, seed, split, checkpoint, or
multiplicity settings after inspecting an outcome.

The formal result is admissible only if the final run records
`evidence_eligibility.eligible=true`, its source-tree start/end fingerprints
match, every required artifact exists, and the release validator accepts it.
`generate_macro_values.py` is the fail-closed bridge from those runner-native
artifacts to the release manifest; it accepts no manually supplied result
values. Its exact derivations and JSON/CSV/NPZ consistency checks are recorded
in `MACRO_DERIVATION_AUDIT.md`. A successful write also generates the
`EligibleLockedResults` sentinel required by the public LaTeX branch and binds
that sentinel to `paper/release_lock.json`; the internal placeholder
deliberately cannot unlock the paper. The current contract is macro-manifest
v3 with 73 provenance records, 74 non-sentinel TeX commands, and 75 commands
only after the release sentinel is added. CSSL supervised adaptation is the
predeclared primary reference.

IQFormer-inspired is a required comparator, not the primary reference.

## Submission boundary

This package does not guarantee an editorial outcome or a 90% acceptance
probability. Until eligible formal results exist, the deliverable is a
scientifically honest **pre-submission convergence package**, not an
upload-ready paper. Human authors must still provide authorship, affiliations,
funding/conflicts, patent-publication review, primary-source verification, and
the final submission decision.

## Directory map

- `configs/formal_tvt_freeze_v1.json`: immutable intended experiment.
- `run_candidate_local.ps1`: non-executing-by-default, one-shot VIMD-v4
  diagnostic hand-off.
- `run_local.ps1`: non-executing-by-default local hand-off.
- `generate_macro_values.py`: deterministic formal-run macro derivation.
- `MACRO_DERIVATION_AUDIT.md`: macro rules and artifact consistency audit.
- `PRE_SUBMISSION_CHECKLIST.md`: human and machine release gates.
- `COVER_LETTER_DRAFT.md`: claim-bounded cover-letter draft.
- `tccn_reuse/`: provenance-recorded generic assets reused from the separate
  TCCN workstream; no satellite result or scenario claim is imported.
