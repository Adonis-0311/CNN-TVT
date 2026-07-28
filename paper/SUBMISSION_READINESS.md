# IEEE TVT Submission Readiness

Status date: 2026-07-28

This file is a release gate, not an estimate of acceptance probability.
Editorial decisions cannot be guaranteed.  The target is to remove avoidable
scientific, reproducibility, scope, and presentation weaknesses before the
human authors decide whether to submit.

## Journal constraints verified from the current TVT pages

- Use the standard IEEE two-column, 10-point transaction format.
- A regular initial submission may contain at most 14 printed pages including
  references and author biographies.
- Revised and final regular papers may contain at most 16 pages.
- Mandatory overlength charges are US$220 per printed page above 10 pages for
  a regular paper. Effective 2026-07-01, a corresponding author who is a VTS
  member receives two additional free pages or a US$440 overlength discount;
  this is a fee benefit, not a scientific or submission gate.
- The work must fit vehicular technology scope; the manuscript therefore has
  to make mobility, time-varying channel, interference, and receiver utility
  part of the system model and evidence, not only the motivation.
- The current TVT instructions state that AI tools may not be used in place
  of an author to generate article content. They permit AI-assisted
  modification of text written by the paper authors (for example, grammar
  improvement) only with disclosure in the acknowledgments. Consequently,
  this repository is an internal research/engineering draft, not text that
  may be uploaded unchanged: the human authors must substantively author and
  verify the final manuscript, make the required acknowledgment disclosure,
  and recheck the policy in force on the upload date.

Official pages:

- https://vtsociety.org/publication/ieee-transactions-vehicular-technology
- https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions
- https://vtsociety.org/publication/ieee-transactions-vehicular-technology/guidelines-authors/instructions/page-charges
- https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/

## Release gates

| Gate | Requirement | Current state |
|---|---|---|
| G0 Scope | Vehicular system model, mobility/channel factors, receiver use case | Drafted |
| G1 Data integrity | Source-disjoint splits, exact realized SNR/SIR, immutable manifests/checksums | Passed for the immutable 1024-sample screening cache: nine splits, 4,700 globally unique sources, 9,400 paired views, 189 declared arrays, and cache digest `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3`. Its designation is screening-only; the formal headline cache remains pending |
| G2 Algorithm effectiveness | Full model exceeds every frozen non-oracle baseline by at least 5 percentage points in hard-region macro-F1 | Not established; all existing learned-model outcomes are diagnostic or invalidated and are excluded from the manuscript |
| G3 Mechanism/ablation | Required mechanism fields finite, required third-route correlations nonnegative, oracle-conditioned spectral component ratio positive, and all six predeclared direct A0--A7 contrasts have family-wise simultaneous 95% lower bounds strictly above zero at full precision and after two-decimal public rendering | Not established; the formal cache/run and six-row ablation artifact do not exist, so no ablation result may be promoted |
| G4 Generalization | At least two of unseen jammer, unseen speed, and held-out channel gain at least +3 pp macro-F1 versus the predeclared CSSL reference; clean A/C/D and held B/E satisfy the frozen noninferiority floors | Not measured |
| G5 Engineering | Reproducible parameter/operator-count and frozen-device latency evidence, with its exclusions stated exactly | Not measured; no SDR or deployment gate is claimed in this simulation-only manuscript |
| G6 Reproducibility | Five locked seeds, paired accuracy/macro-F1 CI, seed/source uncertainty, preregistered McNemar/Holm family, one joint six-contrast ablation family, frozen configuration | Formal configuration, checkpoint-eligibility serialization, no-fallback rule, jammer-support masking, source fingerprints, and fail-closed macro release are implemented; the final non-training regression passed 199 main and 10 standards tests. The formal run remains pending |
| G7 Manuscript | No unsupported novelty, artifact-generated figures/tables only, primary-source citation audit, 14 pages or fewer | Canonical `paper/build/main.pdf` is a fresh 8-page internal evidence-lock build (420,895 bytes; SHA-256 `e99d21895ef31fc8c4291777ae8f34d53842571ca649ee986e483409385dc636`), with internal gate 36/36 and visual QA 8/8. No release PDF exists. Upload is prohibited until formal release and human citation/authorship review |

## Automatic rejection of evidence

A run is ineligible for the manuscript if any of these is true:

- `proxy_evidence_only=true`;
- cache designation is smoke, screening, diagnostic, invalid, or superseded;
- incomplete class support;
- an expected factor-isolated split or pre-registered comparison is absent;
- train/validation/test source identities overlap;
- the test set influenced hyperparameter choice;
- no immutable cache digest or file checksums;
- fewer seeds than claimed;
- component validation or checksum verification was skipped;
- no explicit machine-readable `evidence_eligibility.eligible=true` decision;
- the executable source-tree fingerprint changes between run start and end;
- metrics were copied or edited manually;
- a baseline uses a different source set, training budget, or selection rule
  without a clearly separated sensitivity analysis;
- latency was measured under shared GPU contention;
- a failed or superseded run is selected;
- the model or dataset label overstates official-code reproduction, 3GPP
  compliance, or SDR evidence.

## Current scientific stop/go state

- The screening cache integrity audit passed, but its machine-readable
  designation is `screening_not_formal_tvt_evidence`; no screening metric may
  be promoted.
- Temporal, fully annealed teacher-routing, fixed-descriptor late-fusion, and
  VIMD-v3 shared-IQFormer candidates failed their promotion gates.
- VIMD-v4 DSBN is implemented and structurally verified but remains an
  unexecuted one-shot diagnostic; it cannot enter the formal family without
  passing its prospective gate and creating a new freeze before formal test
  access.
- The completed 30-job legacy screening run changed six source files during
  execution and is formally invalidated.  Its only permissible use is
  architectural triage.
- Cochannel collisions between two in-taxonomy emitters are excluded from the
  single-label headline protocol unless a physical target anchor, a
  dominant-emitter rule, or an ambiguity-aware label is pre-registered.
- The 2025 CSSL-AMC official architecture is now a pinned, executable
  1,024-sample supervised adaptation in the formal comparator family. It is a
  recent auditable AMC comparator only: no complete CSSL reproduction,
  official result, or structured-interference-specific claim is permitted.
- The A0--A7 family fixes A3-A2, A4-A3, A5-A4, A5-A1, A5-A6, and A5-A7 in
  candidate-minus-reference direction. They share the exact ordered five seeds
  and class-stratified source clusters under the non-studentized
  `joint_max_absolute_centered_deviation_hierarchical_paired_bootstrap`.
  Every simultaneous 95% lower bound must be strictly positive; A4-A3 remains
  a bundled intervention and A5-A6 a composite control. The family reuses the
  frozen jobs, so the total remains 55 fits.
- A verified recent structured-interference-specific comparator and
  architecture-specific baseline tuning sensitivity remain open.
  `PrimaryReference` is prospectively fixed to the CSSL-AMC supervised
  adaptation; IQFormer-inspired remains a required non-oracle comparator.
  No post-hoc baseline-winner macro is part of the v3 contract, and no
  strongest-published or strongest interference-specific claim is allowed.

## Human-author actions before submission

1. Supply verified author names, affiliations, funding, conflicts, and
   acknowledgments.
2. Review every mathematical statement, source, result, and conclusion.
3. Arrange patent-publication timing review with a qualified patent
   professional before public disclosure.
4. Preserve the present simulation-only title and claims unless lawful,
   session-held-out SDR evidence is added under a separate protocol.
5. Replace the internal evidence-lock build with automatically generated result
   macros and figures.
6. Run `tvt_submission/validate_release.py` and require an intact
   `paper/release_lock.json` before changing `\internalreviewtrue`.
7. Prepare any IEEE-required acknowledgment disclosure for generated content;
   identify the system, affected sections, and level of use as applicable.
8. Recheck the live TVT instructions, page-charge policy, and AI policy
   immediately before upload.
