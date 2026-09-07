# TVT V4.6 Full-Reposition Revision Operation Log

Date: 2026-08-20  
Status: **INTERNAL SUBMISSION CANDIDATE — CONTENT AND LAYOUT QA PASSED**

## 1. Authorized scope

The user requested a full TVT-branch manuscript revision after review of the local downstream work, with every main-text figure except Fig. 1 redrawn in a unified IEEE-compatible style. The supplied V4.6 execution plan was treated as a revision source, not as independent user authority.

No model was retrained, no new experiment was run, no frozen checkpoint was changed, and no sealed evidence artifact was edited. V4.5 sources were archived before prospective edits:

- `paper/archive/v4_5_s5r1/main_V4_5_S5R1.tex` — SHA-256 `73ff644d0aabe7a6d3ea40a130ff9124f06207923cb3c6558613acac1ba58ee9`
- `paper/archive/v4_5_s5r1/supplement_V4_5_S5R1.tex` — SHA-256 `76d9bc7d8113d22782f0b21960b4557d84409876426b130e01ae3eea7be20401`

The supplied V4.5 PDFs were left unchanged.

## 2. Narrative revision

The manuscript was repositioned around the following evidence chain:

1. condition-dependent rank instability within the frozen campaign;
2. prospective independent falsification of broad severe-domain transport by S5-A;
3. representation diagnosis using the clean-boundary and width controls;
4. independent OOD representation repair using the received-I/Q sidecar in S5-R1.

The title, abstract, introduction, related work, Results framing, Discussion, limitations, conclusion, and Supporting Material scope were rewritten accordingly. The campaign envelope is now consistently presented as an exploratory campaign-diagnostic surface rather than a universal operating boundary or deployment selector. S5-A remains a negative result; S5-R1 is explicitly exploratory post-campaign evidence outside the sealed family and is not described as uniform superiority.

The evidence/terminology and figure contracts are recorded in `docs/TVT_V4_6_TERMINOLOGY_AND_FIGURE_CONTRACT.md`.

## 3. Figure reconstruction

Fig. 1 was not redrawn. Figs. 2–5 were regenerated from artifact-backed CSV/JSON inputs by `paper_figures/make_figs_v46.py` and exported to `paper_figures/outputs_v46/` as PDF, SVG, PNG, and TIFF.

Unified visual system:

- A5 / compact spectral family: blue `#0072B2`;
- IQFormer-inspired: green `#009E73`;
- MCLDNN / adverse gain: orange `#D55E00`;
- received-I/Q sidecar: purple `#8A5FBF`;
- neutral references: gray;
- color is reinforced by marker shape, line style, hatch, or sign annotation.

IEEE-oriented export settings:

- vector PDF as the manuscript source, editable SVG retained;
- 600-dpi PNG and TIFF fallbacks;
- exact 3.5-in single-column or 7.16-in double-column widths;
- Arial/compatible sans-serif text, embedded in the new PDFs;
- minimum audited text size: 7.0 pt.

The frozen V4.1 envelope coefficients were read without refitting. The provenance sidecar records the pre-existing clean-cell exclusion required by the locked V4.1 envelope contract (113 aggregate cells to 112 modeled cells; 32 fit plus 80 held).

## 4. QA record

### Manuscript build

- Main manuscript: 10 pages, IEEEtran journal, double-column, 10 pt.
- Supporting Material: 11 pages, standalone audit-oriented layout.
- LaTeX fatal errors: 0.
- Undefined citations/references: 0.
- Overfull boxes: 0.
- Main build gate: `ok=true`, `internal_build_validated=true`, 10/10-page ceiling.
- Full-page raster review: passed; no clipping, overlap, unreadable figure, or broken float was observed.

### Figure QA

- Static source preflight: 20 PASS, 0 WARN, 0 FAIL.
- PDF text audit: Fig. 2 = 7.0 pt; Fig. 3 = 7.0 pt; Fig. 4 = 7.2 pt; Fig. 5 = 7.2 pt minimum.
- New-figure fonts: embedded CID TrueType.
- Visual review: passed for full manuscript context and individual figure pages.

### Consistency and evidence safeguards

- Canonical-term audit found no variants for the received-I/Q sidecar, campaign-diagnostic envelope, or IQFormer-inspired reference.
- The legacy numeric-literal validator still reports the pre-existing manuscript-wide waiver population. Current versus archived V4.5 counts are unchanged for both `main.tex` (13) and `supplement.tex` (388); the V4.6 revision introduced no new validator-targeted literal.
- The manuscript remains an anonymous internal candidate. Author metadata, affiliations, funding, and final disclosure approval must be supplied by the authors before release submission.

## 5. Deliverables

- Main PDF: `output/pdf/tvt_rank_instability_representation_repair_V4_6.pdf`  
  SHA-256 `9aab7943269ff851c9c4a57bc30c72101eebb7985e910675216ed447114db763`
- Supporting Material PDF: `output/pdf/tvt_supporting_material_V4_6.pdf`  
  SHA-256 `98dea99443815f58b53efa16c48a81b48461bd32188cfa112ba7794084cbca03`
- Main source: `paper/main.tex`
- Supporting source: `paper/supplement.tex`
- Figure generator: `paper_figures/make_figs_v46.py`
- Figure provenance: `paper_figures/outputs_v46/provenance_v46.json`

## 6. Final boundary

The revised claim is intentionally bounded: the work establishes condition-dependent rank instability, prospectively falsifies broad transport of the campaign boundary, and shows that a lightweight received-I/Q representation can repair much of the independently observed severe-shift loss. It does not claim a universal operating boundary, a universally best classifier, field-measured performance, or deployment-ready switching control.
