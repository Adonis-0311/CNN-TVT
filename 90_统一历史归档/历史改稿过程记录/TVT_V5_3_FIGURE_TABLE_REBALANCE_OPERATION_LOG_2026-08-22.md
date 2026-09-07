# TVT V5.3 Figure/Table Rebalance Operation Log (2026-08-22)

Base: V5.2 (`tvt_rank_instability_representation_repair_V5_2.pdf`,
`tvt_supporting_material_V5_2.pdf`). Trigger: user review judged the former
main-text Fig. 3 (performance--complexity Pareto panel) weak and required the
main text to carry its strength without relying on the supplement. V5.3 is a
figure/table rebalance only: no sealed result, aggregation convention, or
claim ladder changes.

## 1. Rationale (assessment outcome)

- Former Fig. 3 (`fig4_complexity_and_representation.pdf`) rated C: the
  parameter/MAC budget point is already carried by Table~`tab:baselines` and
  the A7-vs-A5 sentence; the figure added no load-bearing evidence.
- Supplement Fig. 1 (campaign-diagnostic envelope,
  `fig3_campaign_diagnostic_envelope.pdf`) rated A for the main text: it is
  the only visual evidence for the V-C envelope diagnostic that Sections V-C
  and V-F argue from. Promoted to main-text Fig. 3.
- Three self-containment gaps closed with inline numbers (G1--G3 below) so the
  main text states the quantitative core without pointing to the supplement.

## 2. Manuscript edits (paper/main.tex)

- Version header V5.2 -> V5.3.
- Deleted the `fig:pareto` figure environment (former Fig. 3); the A7-vs-A5
  sentence in Section V-A now cites `Supplement Table~IV` (capabs) instead.
- Inserted the envelope `figure*` (`fig:envelope`) after the V-C synthesis
  paragraph; caption is macro-driven (`\EnvelopeFitCells`,
  `\EnvelopeHoldoutCells`, `\EnvelopeIqformerHoldoutRmse`,
  `\EnvelopeConstantRmseIQ`, `\EnvelopeSkillRTwoIQ`) and states the
  regime-heterogeneous skill and the non-transport boundary.
- G1 (regime skill): V-C synthesis paragraph now reports per-regime held
  $R^2_{\mathrm{skill}}$ for both references via eight new audit macros
  (`\RegimeSkillIq{Speed,Channel,Jammer,Ood}`,
  `\RegimeSkillMc{Speed,Channel,Jammer,Ood}`) with a `Supplement Table~II`
  pointer.
- G2 (capacity intervals): V-F capacity paragraph now reports the
  $[\cdot,\cdot]$ intervals for the L-vs-IQFormer hard and clean contrasts
  (existing generated macros) plus the new severe-stratum disclosure: L gains
  only `\CapacityLSevereVsSDiff` pp ($[\CapacityLSevereVsSLow,
  \CapacityLSevereVsSHigh]$) over the sealed tier at SIR = -15 dB.
- G3 (transfer pooled gain): V-F transfer paragraph now reports the
  window-pooled A5 gain (`\SevereTransferGainIqformer`,
  `\SevereTransferGainMcldnn`) and the seed-wise convention values
  (`\SevereSeedMeanGapIqformer`, `\SevereSeedMeanGapMcldnn`), all pre-existing
  generated macros.

## 3. Supplement edits (paper/supplement.tex)

- Version header V5.2 -> V5.3.
- Deleted the envelope figure (promoted to the manuscript).
- Merged former Tables XII (`tab:leave15`) and XIII (`tab:bootpair`) into one
  two-panel table (panel (a) leave-15-out, panel (b) block-paired bootstrap);
  both labels retained, prose cross-references updated.
- Merged former Tables XVI/XVII: lesion-ratio rows appended as panel (c) of
  `tab:controls`; `tab:lesionratios` label retained at the merged table end;
  caption extended to mention the `\SevereRepairDiff`-pp repair effect.
- Post-merge table numbering is I--XVII. The manuscript's hard-coded
  supplement pointers (Table II regime skill, Table IV capabs, Table VIII
  convergence) all precede the first merge point and remain correct; all
  other supplement cross-references use `\ref`.

## 4. New audit macros (paper/v1_audit_macros.tex)

Eleven new macros appended (names contain no digits, per project convention),
each verified against artifact CSVs:

| Macro | Value | Source row |
|---|---|---|
| `\RegimeSkillIqSpeed` | 0.75 | `a14_v41_envelope_interfered_holdout.csv`, IQFormer unseen_speed |
| `\RegimeSkillIqChannel` | 0.59 | same, IQFormer heldout_channel |
| `\RegimeSkillIqJammer` | 0.20 | same, IQFormer unseen_jammer |
| `\RegimeSkillIqOod` | 0.34 | same, IQFormer combined_ood |
| `\RegimeSkillMcSpeed` | 0.72 | same, MCLDNN unseen_speed |
| `\RegimeSkillMcChannel` | 0.64 | same, MCLDNN heldout_channel |
| `\RegimeSkillMcJammer` | -0.02 | same, MCLDNN unseen_jammer |
| `\RegimeSkillMcOod` | 0.01 | same, MCLDNN combined_ood |
| `\CapacityLSevereVsSDiff` | 4.96 | `tier2_capacity_L_v1_vs_S.csv`, hard_interference_sir_minus15 diff |
| `\CapacityLSevereVsSLow` | 3.48 | same row, CI low |
| `\CapacityLSevereVsSHigh` | 6.48 | same row, CI high |

Regime-skill values match Supplement Table II; the severe capacity contrast
matches Supplement Table V. These are audit-verified constants, not generated
`paper_data_layer` macros, and must be re-derived by the build pipeline
before release (same standing caveat as the rest of `v1_audit_macros.tex`).

## 5. Verification

- Number validator (`python paper_data_layer\validate_paper_numbers.py`):
  518 issues vs the V5.2 baseline of 508 (+10). Diff traced line-by-line:
  all ten new issues are the new macro definitions inside
  `v1_audit_macros.tex` itself (`0.01` is exempt via PROTOCOL_LITERALS);
  no new hand-typed evidence literal entered main.tex or supplement.tex
  prose or tables. The validator remains in its documented waiver state
  (`docs/TVT_V4_4_NUMBER_VALIDATION_WAIVER.md`).
- Compilation (latexmk full chain incl. bibtex, from `paper/`):
  main.tex -> `tvt_rank_instability_representation_repair_V5_3.pdf`,
  10 pages, zero undefined references, zero undefined citations, zero
  undefined control sequences; supplement.tex ->
  `tvt_supporting_material_V5_3.pdf`, 15 pages, same checks clean.
- Supplement contains no hard-coded table/figure numbers, so the renumbering
  introduced by the merges is fully absorbed by `\ref`.

## 6. Page-budget compression (main text, 11 -> 10 pages)

The promoted envelope `figure*` initially pushed the manuscript to 11 pages
(overflow: trailing references). Restored to the 10-page target by prose
compression and float tightening only; no figure, table, evidence number, or
claim was removed:

- Figure widths tightened: teacher figure `\textwidth` -> `0.85\textwidth`;
  rank-instability figure -> `0.95\textwidth`; envelope figure ->
  `0.75\textwidth` (two-panel layout stays legible at this width).
- Envelope caption condensed (panel definitions, RMSE/skill disclosure, and
  non-transport statement all retained).
- Introduction paragraph 2 condensed (same claims, tighter wording); intro
  result-summary paragraph condensed (all conclusions retained).
- Related Work condensed: distribution-shift phenomenon sentence merged into
  one citation group (all three citations kept); vehicular-receiver and
  structured-interference sentences tightened.
- Implementation Details: optimization paragraph and seeds sentence
  condensed (all hyperparameters and selection-rule disclosures kept);
  E1 protocol sentence tightened (protocol content unchanged).
- Minor sentence-level tightening in V-B, V-C, V-E (facts unchanged).

## 7. Freeze

- `output/pdf/tvt_rank_instability_representation_repair_V5_3.pdf`
  SHA-256 `C93380CE9CB7BD6DE43847F1E5B4CDBD0ED3D665652E988489E6B037683839B7`
  (10 pages).
- `output/pdf/tvt_supporting_material_V5_3.pdf`
  SHA-256 `DE86D25513877C6AA81A57D720CBC6AA0B9986C9D08F06C6CA2BCFEA9D2B206B`
  (15 pages).
- Number validator re-run after all compression edits: 518 issues, unchanged
  from the post-rebalance state (no new hand-typed literals introduced).

## 8. Not changed

- No sealed numbers, aggregation conventions, decision rules, or claim
  content were altered (Section 6 compression changes wording only, with all
  facts, disclosures, and citations retained). G4 (gap-recovery ratio inline)
  was assessed and left supplement-only as optional detail. The former Fig. 3
  content remains available in `paper_figures/outputs_v46/` for any future
  reuse.
