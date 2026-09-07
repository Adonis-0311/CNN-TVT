# TVT V4.11 Final Revision Operation Log (2026-08-21)

## Scope

- Baseline: V4.10 main manuscript (10 pages) and V4.10 Supporting Material (15 pages).
- Source instruction: `TVT_V4_11_投稿前最终改稿指令.md`.
- Revision type: final evidence-boundary and wording closure only.
- No experiment, result, main-story, figure, or table was added or removed.

## Main manuscript changes

- Deleted the complete A5--A0 capacity-attribution footnote, including the extrapolated share and lower-bound language.
- Changed the Abstract generalization from `changes` to `can change`.
- Replaced the Abstract retraining-gap statement with an inference-only evidence boundary.
- Replaced the I/Q-family easiness claim with evaluation-specific performance wording and retained the confounding boundary.
- Added `where reported` to the Table III confidence-interval caption statement.
- Replaced active retrained-ablation language in the Discussion and Conclusion with the inference-only quantitative-attribution boundary.
- Prevented the checked `source-disjoint` and `route-gating` compounds from splitting across lines.

## Supporting Material changes

- Added the required space in `classifier (fused embedding)`.
- Replaced the intrinsic-easiness wording with evaluation-specific I/Q-domain performance wording.
- Prevented the checked `source-disjoint` compound from splitting across lines.
- Retained the retrained no-I/Q variant only as a Supplement-level future diagnostic suggestion.

## Final QA

- Main manuscript: 10 pages, 4 figures, 3 tables.
- Supporting Material: 15 pages.
- Compilation: MiKTeX `latexmk -pdf`; no LaTeX errors, undefined citations/references, overfull boxes, or oversized floats.
- Final search gate: zero hits in the main manuscript for the speculative capacity attribution, intrinsic/inherent easiness wording, active retrained-ablation wording, causal-overclaim terms, and checked broken compounds.
- Visual QA: every page rendered and inspected; no clipping, overlap, black squares, missing graphics, malformed tables, or broken mathematical signs were found.

## Frozen V4.11 artifacts (SHA-256)

- `output/pdf/tvt_rank_instability_representation_repair_V4_11.pdf`
  `111696052F6A02FC30AD652B73C00EA882359E333E7442582AE46F8D78DA056D`
- `output/pdf/tvt_supporting_material_V4_11.pdf`
  `DE526215324125246DB9E9EEEE1EFE56DA9AA6F5FFD2767A8733798DC01CBC86`

These two V4.11 PDFs are the frozen submission artifacts for this revision round. The V4.10 PDFs were preserved byte-for-byte.
