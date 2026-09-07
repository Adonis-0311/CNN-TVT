# TVT V4.1 execution audit

Date: 2026-08-19.  Manuscript baseline: `output/pdf/tvt_operating_envelope_integration_V4.pdf`.
This document records prospective V4.1 work only; sealed artifacts are read-only.

## Completed manuscript integration

- `paper/main.tex` was compressed from the 13-page V4 baseline to ten pages.
  The retained central claims are rank exchange, the overlap--SIR envelope,
  bounded post-hoc severe-corner evidence, and clean-boundary repair.
- The teacher visualization (Fig. 1) is **retained** per the author's explicit
  preference (V4.2-style compression: full-width layout, three-line caption,
  pointer text instead of a repeated explanation).  The severe-corner figure,
  risk--coverage figure, coherence-time table, information/notation tables,
  capacity table, and break-even table (reduced to two prose sentences) were
  removed.  The five remaining figures are the teacher illustration, the full
  operating map, the envelope, the matched-subset Pareto view, and the A5
  clean transfer taxonomy; the five tables are the comparator summary, the
  post-review RML2016.10a fidelity check, the sealed family, the held-regime
  envelope skill, and the Tier-2 repair table.
- The Tier-2 table retains only intervention, parameter count, clean effect,
  and hard-interference effect.  It remains exploratory and outside the frozen
  confirmatory family.
- Evaluation-cell granularity is explicit in Section IV-A: each
  hard-interference SNR--SIR cell contains 131--181 source-disjoint test
  windows (mean 156; all ten classes, 5--26 per class), read from the frozen
  prediction cache.

## Envelope integrity correction

The requested notation “8 SNR x 4 SIR fitting cells” conflicts with immutable
analysis records.  The actual historical 32 fitting cells are `id_test` and
`hard_interference`, further partitioned by SIR and overlap quartile.  They are
retained without relabelling.  The V4.1 primary holdout contains the 80 finite-
SIR interfered cells in unseen-jammer, unseen-speed, held-out-channel, and
combined-OOD regimes.  Clean retention is intentionally excluded because it has
no jammer and therefore no physical finite SIR.

The resulting IQFormer-inspired/MCLDNN primary scores are respectively
6.23/8.10 pp envelope RMSE, 8.37/9.63 pp fit-domain-constant RMSE, and
0.45/0.29 $R^2_{skill}$.  These are exploratory reanalyses of sealed
predictions, not a replacement for any frozen family result.

## E1 external comparator-fidelity check

`experiments/run_rml2016_fidelity.py` records the exact RadioML2016.10a data
hash, split indices, code hashes, per-epoch history, per-SNR metrics, and
checkpoints.  The local MCLDNN remains a PyTorch reimplementation; the local
IQFormer remains architecture-inspired.  E1 is solely a comparator-fidelity
check: it does not test VIMD-Net, sidecar repair, or external validity of the
operating envelope.

**Protocol realignment (2026-08-19).**  The machine-readable published
reference for BOTH comparators is the official IQFormer benchmark result table
`testA.xlsx` (MCLDNN 62.05%, IQFormer 64.19% all-SNR overall), produced under
the per-stratum 80/20 then 75/25 `train_test_split(random_state=233)`
(= 600/200/200) protocol.  The original MCLDNN repository's 2016-set-based
partition has no machine-readable reference (performance is a figure only; its
Keras 2.2.4/TF1 weights could not be faithfully re-loaded).  E1 therefore
evaluates both local comparators under the 233 protocol so that Table II is a
same-protocol ``our implementation vs. published/reference`` for both rows.
See `docs/RML2016_10A_FIDELITY_PROTOCOL.md`.

**First-seed results (v2 run, 233 protocol, inspected 2026-08-19):** all-SNR
test accuracy is 58.19% (MCLDNN, seed 2016) and 65.81% (IQFormer-inspired,
seed 233), versus published 62.05%/64.19% (deltas -3.86 pp / +1.62 pp).  Both
fall inside the predeclared fidelity bands (close <2 pp for IQFormer-inspired;
same range 2--4 pp for MCLDNN), so the three-seed follow-up was authorized.

**Three-seed results (v3 run, seeds 17/29/43, 233 protocol, completed
2026-08-19):** all-SNR test accuracy is 58.55% [58.07, 59.39] for MCLDNN and
65.23% [64.79, 65.79] for IQFormer-inspired, versus published 62.05%/64.19%
(deltas -3.50 pp / +1.04 pp).  Both rows of Table II are now a same-protocol
``our implementation vs. published/reference'' (published values from the
official IQFormer benchmark result table testA.xlsx).  Per-SNR deviations up
to about 8 pp in individual bins (low-SNR bins favor the local
implementations, mid-SNR bins the reference) are documented as consistent
with cross-implementation training differences.  **E1 is complete and is
reported as a passed comparator-fidelity check within the predeclared bands.**
The earlier `rml2016_10a_fidelity_v1/` MCLDNN row is superseded: its
2016-set-based split differed from the official set-based partition in 24% of
validation/test indices and has no machine-readable reference.

## Release gate

The V4.1 PDF is `output/pdf/tvt_operating_envelope_integration_V4_1.pdf`
(10 pages, rebuilt 2026-08-19 after E1 backfill and Fig. 1 restoration; the
pre-E1 build is preserved as `tvt_operating_envelope_integration_V4_1_preE1.pdf`).
It is not a final submission PDF until author-only submission fields
(author/affiliation/funding/AI disclosure) are completed by the human authors.
