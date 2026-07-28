# VIMD-Net Evidence Ledger

This ledger separates implemented mechanisms from measured claims.  A green
software check is not automatically a publishable scientific result.

## Claim vocabulary

- **Implemented**: present in code and covered by a targeted test.
- **Execution-validated**: gradients and optimization behave on a bounded
  diagnostic.
- **Development evidence**: measured on the heuristic proxy generator.
- **Standards-aligned evidence**: measured on a frozen MATLAB
  `nrTDLChannel` cache with source-disjoint manifests.
- **Submission evidence**: repeated, statistically analysed
  standards-aligned and/or session-held-out SDR results.

## Current ledger

| ID | Scientific claim | Required artifact | Current status | Manuscript permission |
|---|---|---|---|---|
| C1 | Target/jammer/unexplained teacher masks are nonnegative, sum to one, and are invariant to common amplitude scaling | unit tests for conservation, symmetry, noise-only, and scaling | Implemented | Method only |
| C2 | Student masks share the exact complex-STFT lattice with the physical teacher | shape/alignment tests and source inspection | Implemented | Method only |
| C3 | Real mask application preserves the phase of retained coefficients | analytical identity plus numerical and overlap-add tests | Implemented and tested | Method only |
| C4 | All VIMD loss terms reach the intended parameter groups | fixed-batch gradient audit | Execution-validated (`artifacts/diagnostics/diagnostics_20260727_230445.json`) | Internal diagnostics only |
| C5 | The complete objective can overfit a fixed batch | bounded overfit gate | Execution-validated: accuracy 0.25 to 1.00; total loss reduced 87.3% | Internal diagnostics only |
| C6 | Tri-mask improves hard-interference AMC over direct and single-mask classifiers | five frozen seeds, hard test, hierarchical paired 95% CI, supplemental per-seed exact McNemar, Holm correction | Not established | Prohibited |
| C7 | Physical teacher improves mask agreement and feature SIR | A2/A3 paired ablation plus mask/permutation metrics | Not established; diagnostic teacher masks track overlap, but student routing has not improved AMC | Prohibited |
| C8 | Multi-task supervision improves branch semantics without negative transfer | A3/A4 paired ablation and jammer/quality metrics | Not established | Prohibited |
| C9 | Exact-source cross-condition learning improves held-out mobility/channel robustness | A4/A5 paired ablation on independently held-out speed and TDL profile | Not established | Prohibited |
| C10 | VIMD generalizes to unseen jammer families | frozen unseen-jammer test with no generator-family leakage | Not established | Prohibited |
| C11 | VIMD is evaluated under vehicular-Doppler TDL-profile simulations | TDL-A/C/D train, TDL-B/E held-out, mobility grid, immutable formal cache | MATLAB TDL backend and a fully audited 1024-sample screening cache are implemented; formal cache/results remain pending | Method/protocol only |
| C12 | VIMD closes a simulation-to-hardware gap | session/device-held-out SDR experiment | Not available | Prohibited |
| C13 | VIMD meets a real-time deployment target | isolated CPU/GPU batch-1 P50/P95, peak memory, fixed power state | Not established; shared GPU is currently contaminated | Prohibited |
| C14 | A recent auditable AMC architecture comparator is included | pinned CSSL-AMC official source/hash/license lock, exact local topology tests, synchronized formal registry | Implemented as an official-architecture supervised adaptation; no formal result exists | Comparison protocol only |

## Diagnostic findings that cannot be promoted

The following results constrain the research direction but are explicitly
ineligible for manuscript performance tables:

- On the source-disjoint TDL screening cache, the fixed 61-dimensional
  HOC/cyclostationary control reaches 71.29% accuracy and 71.03% macro-F1 on
  the tracked clean target, versus 12.70% and 11.67% on the received mixture.
- The fixed-teacher target-only route reaches 30.66% accuracy and 27.29%
  macro-F1; adding half of the overlap route lowers them to 24.02% and 22.21%.
  This is an oracle-conditioned representation probe, not deployable
  separation.
- A phase-aware temporal candidate improves tracked-clean representation but
  fails on received mixtures.  A fully annealed teacher-routing curriculum
  and a fixed-descriptor late-fusion candidate also fail their prospective
  mixture gates.
- The legacy 10-model, three-seed run completed, but six executable files
  changed during execution.  Its directory contains
  `INVALIDATED_FOR_EVIDENCE_SOURCE_MUTATION.md`; none of its metrics may be
  promoted.
- The VIMD-v3 shared-IQFormer route met its validation and route-noncollapse
  diagnostics but missed its required held-out accuracy gain; its outcome is
  permanently closed as `vimd_v3_tiny_success_threshold_not_met`.
- VIMD-v4 DSBN has code and structural tests only.  No VIMD-v4 performance
  result exists, and running it requires an explicit diagnostic switch.

## Screening-cache integrity record

The independent audit
`standards/cache_factor_screening_1024_v1.audit.json` establishes only
pipeline stability:

- cache digest
  `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3`;
- schema 2 with exactly nine factor-isolated splits;
- 4,700 globally unique source sequences and 9,400 paired views;
- 189 declared array files with checksums, shapes, dtypes, and finiteness
  verified;
- maximum component identity error `4.48e-7`, maximum realized-SIR error
  `5.61e-6` dB, and maximum realized-SNR error `4.44e-8` dB;
- all cache-construction-critical source files unchanged between preregistered
  build metadata and audit.

The broader source tree changed during cache construction and remains
disclosed.  The cache designation is screening-only, so this integrity record
does not support an AMC performance claim.

## Non-negotiable TVT gates

1. **Data credibility**: exact realized SNR/SIR, source-sequence-disjoint
   train/validation/test splits, immutable manifests and checksums.
2. **Strong comparison**: classical feature baseline; direct neural baselines;
   MCLDNN; a transparent IQFormer-inspired implementation; the CSSL-AMC
   official-architecture supervised adaptation; single-, dual-, and tri-mask
   variants; clean and oracle controls. A verified recent
   structured-interference-specific implementation and architecture-specific
   tuning sensitivity remain open.
3. **Mechanism evidence**: route-wise weighted correlation and MAE,
   target-energy transfer ratio and amplification, jammer leakage,
   counterfactual TF-SIR gain, separate unexplained/target--jammer ambiguity
   tests, overlap permutation test, and mask-collapse checks.
4. **Generalization**: jammer family, speed, and TDL profile held out
   independently and jointly.
5. **Statistics**: at least three seeds for the screening stage and five for
   the locked headline comparison; source-cluster bootstrap, exact paired
   McNemar, and family-wise Holm adjustment.
6. **Engineering realism**: isolated latency measurements and
   session-held-out SDR.  Without SDR evidence the paper must narrow its
   engineering claims.

## Result promotion rule

No number is copied manually into the paper.  A paper macro may be generated
only from a run directory that contains:

- `run.json`;
- schema-2 manifest with the exact nine locked splits, factor policies,
  quality-normalization metadata, jammer taxonomy, and protocol exclusions;
- frozen train/validation/test manifests and digests;
- per-model checkpoints;
- source-aligned probability bundles;
- `metrics.csv`;
- `paired_statistics.csv`;
- seed aggregates when more than one seed is claimed;
- the exact code/config/environment identifier.
- a machine-readable `evidence_eligibility.eligible=true` decision with no
  unresolved gate reason.
- identical executable source-tree fingerprints at run start and end.
- an eligible checkpoint for every model/seed fit, with no final-state
  fallback.
- the exact expected model-by-seed matrix and finite required regime metrics.
- an automatically generated macro-value manifest with per-macro artifact
  provenance, followed by a successful fail-closed release lock.
- a release-writer-generated `EligibleLockedResults` sentinel whose fixed value
  and containing-file hash are bound by the schema-v2 release lock.

Smoke, screening, superseded, and proxy runs are never eligible for headline
tables, even when their execution status is `complete`.
