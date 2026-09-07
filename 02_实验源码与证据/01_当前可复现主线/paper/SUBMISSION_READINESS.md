# TVT Honest-Manuscript Integration Readiness

Status date: 2026-08-18
Scope: simulation-only scientific manuscript; this record does not unlock journal submission.

## Decision

The technical integration manuscript is built and audited. The paper now uses
the occupancy--SIR operating envelope as its organizing claim, preserves the
sealed confirmatory result and failed gates, and incorporates the prospective
Tier-2 probe/repair and capacity ladder as exploratory evidence outside the
frozen confirmatory family.

`honest_manuscript_validated=true` means that the manuscript is internally
consistent, artifact-traceable, freshly compiled, and explicit about its
scientific boundary. It does **not** mean that all scientific gates passed or
that submission is authorized. The governing state remains:

- `scientific_evidence_passed=false`;
- `submission_unlocked=false`;
- failed gates: `confirmatory_family_gate_failed` and
  `clean_retention_gate_failed`.

## Completed integration

| Area | State | Evidence |
|---|---|---|
| Main narrative | **Complete** | `paper/main.tex` is rewritten around the operating envelope, full-region rank exchange, severe corner, transfer boundary, repair chain, and capacity ladder. |
| Data layer | **Complete** | `paper_data_layer/outputs/paper_numbers.json` contains artifact-traceable keys with source file, row selector, evidence class, precision, and source SHA-256. |
| Tier-2 separation | **Complete** | Probe, coverage, sidecar, capacity, and H2-G negative-control values are labeled exploratory and outside the frozen confirmatory family. |
| Figures | **Complete** | Six artifact-derived analysis figures plus the teacher-definition figure are included; no figure recomputes evidence. |
| LaTeX build | **Complete** | Seven-page IEEE two-column PDF; no fatal error, undefined citation/reference, rerun warning, or overfull box. |
| Visual QA | **Complete** | All pages rendered to PNG and inspected; equations, figures, tables, margins, and references are legible and unclipped. |
| Honest-manuscript audit | **Complete** | `tvt_submission/honest_paper_release.json` reports `honest_manuscript_validated=true` while retaining `submission_unlocked=false`. |

## Current scientific position

- The sealed A5--A0 whole-method contrast is positive with a simultaneous
  confidence interval above zero.
- The teacher-form and route-count component contrasts do not individually
  clear zero; therefore the confirmatory family gate did not pass.
- The compact spectral model has a reproducible severe-interference operating
  region but trails stronger I/Q baselines in weak, clean, and low-occupancy
  regions; the paper reports the complete map rather than uniform superiority.
- The clean-retention gate did not pass. Frozen probes lean toward a
  representation limitation; adding per-class clean exposure alone does not
  repair it, whereas the received-I/Q sidecar does. The preregistered H2-G
  gate also does not repair it (clean −0.30 pp, 95% CI [−0.66, +0.08]); its
  gate saturates near one on both clean and jammed inputs, so it is reported
  as a negative control rather than a functional clean fallback.
- Widening the spectral model improves average fitting but does not eliminate
  the boundary to IQFormer-inspired, so the remaining gap is not explained by
  parameter count alone.

## Remaining before an actual submission decision

WP10 and WP12 are now technically complete: the dated recent-literature audit
is in `docs/RECENT_LITERATURE_POSITIONING_2026.md`, and the deterministic
lightweight reviewer archive validates against its embedded hash manifest.
WP13's technical checklist is prepared in `paper/AUTHOR_SUBMISSION_SIGNOFF.md`.

The remaining blockers require human responsibility rather than more automatic
integration: verified author identities/affiliations, funding, conflicts,
acknowledgments, truthful AI-use disclosure, patent/publication-timing review,
data/code availability wording, and the corresponding author's signed upload
authorization. H2-G has been integrated as the preregistered negative
control. The reviewer package has been rebuilt and independently validated
(171 files; its final SHA-256 is supplied by the adjacent archive sidecar).

## Canonical artifacts

- manuscript source: `paper/main.tex`;
- generated values: `paper_data_layer/outputs/paper_numbers.json` and
  `paper_data_layer/outputs/paper_macros.tex`;
- compiled proof: `output/pdf/tvt_operating_envelope_integration.pdf`;
- audit implementation/report:
  `tvt_submission/validate_honest_paper_release.py` and
  `tvt_submission/honest_paper_release.json`.
- reviewer reproduction archive:
  `output/repro/tvt_reviewer_reproduction_v1.zip` and its `.sha256` sidecar;
- human release gate: `paper/AUTHOR_SUBMISSION_SIGNOFF.md`.
