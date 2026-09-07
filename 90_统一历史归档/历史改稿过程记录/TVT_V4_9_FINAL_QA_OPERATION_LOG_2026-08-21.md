# TVT V4.9 Final-QA Operation Log

Date: 2026-08-21 (Asia/Shanghai)

## Scope and governing instructions

This pass implements the user's final-revision request against the frozen V4.8 PDFs, using the following two local documents as revision specifications:

- `C:/Users/Administrator/Downloads/TVT_V4_9_Final_QA_最终改稿执行方案.md`
- `C:/Users/Administrator/Downloads/TVT_V4_9_正文图表精简调整指令.md`

Where the documents overlap, the later main-text figure/table instruction governs the narrow presentation conflict: no new figure was introduced, Section V-F was compressed, and one core main-text table was added. No new training, simulation, checkpoint selection, held set, model, or claim family was created.

Frozen V4.8 baselines:

- Main PDF SHA-256: `E969FCC898563298D76B775FA997C6CE24785979D392B2AFCD03C95BD2FC0A66`
- Supporting PDF SHA-256: `EFA4A6A89EE1D37146D4B6479F6934A774F16056BF7A1D65258F5DB6576A90F3`

## Evidence and numeric corrections

- Verified the severe-held cache manifest and source IDs: R1 has 6,400 unique sources, R2 has 6,400 unique sources, their intersection is empty, and the total is **12,800** (not 12,820).
- Re-read `a21_control_matched5_levels.csv`: matched-five A5 is 20.72%, M is 21.77%, L is 20.89%, and sidecar is 26.11%.
- Kept the full-ten-seed severe-held levels separate: A5 20.72%, IQFormer-inspired 23.29%, MCLDNN 33.75%, sidecar 25.09%, I/Q-zero 12.80%, I/Q-shuffle 16.25%, and no-side-head 7.09%.
- Added derived provenance macros for the 12,800-source total and the +2.58/+13.03-pp reference leads.
- Read post-campaign `result.json` histories rather than inferring convergence values: M selected-epoch median/range 29/27--30 (5/5 completed epoch 30), L 27/24--30 (5/5), and sidecar 30/29--30 (10/10); no post-campaign run early-stopped.

## Manuscript changes

- Preserved the title exactly.
- Replaced severe-held `preregistered` wording with prospectively frozen/frozen-analysis-plan wording and explicitly bounded the evidence to local plans, hashes, and execution records rather than external timestamping.
- Corrected the first-page Evidence Status so the original sidecar is prospective intervention evidence and the third attribution-control stage is explicit.
- Replaced causal or over-strong language (`reject simpler explanations`, `representation failure`, isolated I/Q interpretation) with bounded joint representation-and-fusion language.
- Standardized the capacity statement to `tested spectral width scaling alone`.
- Added direct distribution-shift references: Taori et al. (NeurIPS 2020), Miller et al. (ICML 2021), and Koh et al. (ICML 2021).
- Added main-text Table III with a full-ten-seed severe-held panel and a matched-five capacity-control panel; the surrounding prose now carries only the shortest sufficient evidence chain.
- Split Supplement Table XV into strictly matched-five and full-ten-seed panels with explicit seed/diagnostic notes.
- Extended Supplement Table VIII with a post-campaign training-history panel.
- Preserved Fig. 1 and all four main figures; the envelope figure remains in the Supplement only. No figure was redrawn or added in V4.9 because the later instruction explicitly requested main-text figure/table simplification rather than new graphics.

## Build and QA

- Rebuilt the paper-number layer: 286 macros (67 sealed, 219 exploratory), 35 source artifacts.
- Main: 10 pages, four figures, three tables.
- Supplement: 13 pages, one envelope figure, sixteen tables.
- Final LaTeX logs: zero undefined references, zero overfull boxes, zero LaTeX warnings, and zero package warnings.
- Visually inspected the first page, revised Section V-F/Table III, conclusion/references, Supplement convergence table, Table XV, provenance, and final artifact-index pages after Poppler rendering.
- `paper_data_layer/validate_paper_numbers.py` still reports 508 legacy hand-typed literals across the V4.8 source; this pre-existing broad audit debt is not introduced by the V4.9 core values, which are macro-backed. It remains recorded rather than masked or converted into new computation.

## Frozen outputs

- `output/pdf/tvt_rank_instability_representation_repair_V4_9.pdf`
  - SHA-256: `3F64A23D49467C78710C4C8275FAA72B3F475CA860B8E77D8E8F8DA5F034293C`
- `output/pdf/tvt_supporting_material_V4_9.pdf`
  - SHA-256: `A05686DFDA3F95C35BDD7863CC941696F5B2CE7BC24947906C1185F0DDD89D29`

Source snapshot:

- `paper/archive/v4_9/main_V4_9.tex`
- `paper/archive/v4_9/supplement_V4_9.tex`
- `paper/archive/v4_9/references_V4_9.bib`
- `paper/archive/v4_9/paper_macros_V4_9.tex`
