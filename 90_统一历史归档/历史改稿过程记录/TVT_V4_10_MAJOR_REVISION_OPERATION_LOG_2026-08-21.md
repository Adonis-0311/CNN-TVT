# TVT V4.10 Major Revision Operation Log (2026-08-21)

## Source instruction
Self-check document `TVT_V4_9_评分与AI痕迹自检.md` applied to the V4.9
manuscript and supporting material; scope = first batch (zero-compute
revisions only). The decisive second batch (retrained no-I/Q sidecar
ablation, 5 seeds) is NOT part of V4.10 and remains an open follow-up.

## Evidence base added (zero compute)
- `analysis_zero_compute/outputs/a22_severe_held_decomposition.json` and
  `analysis_zero_compute/outputs/csv/a22_*.csv` (three CSVs): per-jammer-family,
  per-regime severe-held decomposition and inference-lesion loss ratios.
- Pooled family-level macro-F1 (%): pulse A5 29.03 / IQFormer 37.45 /
  MCLDNN 55.34 / sidecar 38.22; OFDM-like 4.60 / 4.56 / 4.92 / 5.74.
- Regime-level (%): R1 21.13 / 23.82 / 33.98 / 25.34; R2 20.31 / 22.77 /
  33.52 / 24.83; pooled 20.72 / 23.29 / 33.75 / 25.09.
- Lesion ratios (repair = 4.37 pp): iq_zero 12.28 pp = 2.81x, iq_shuffle
  8.84 pp = 2.02x, no_side_head 17.99 pp = 4.12x; all three fall below A5.

## Data layer
- `paper_data_layer/build_paper_numbers.py`: added CapacityLCleanVsIqformer
  contrast, capacity-footnote derived block (slope = gain / log2(ratio),
  implied share = doublings x slope), and a22 family/regime/ratio macros.
  Regenerated `paper_macros.tex` / `paper_numbers.json` (322 keys: 69
  sealed, 253 exploratory; 37 sources).
- New macros: CapacityLCleanVsIqformerDiff (-18.14), CapacityRatioMidToShort
  (2.52), CapacityRatioLargeToShort (5.90), CapacityRatioAFiveToAZero (4.67,
  sealed), CapacitySlopeMidPerDoubling (1.46), CapacitySlopeLargePerDoubling
  (1.24), CapacityAFiveDoublingsVsAZero (2.22, sealed),
  CapacityImpliedShareLow/High (2.8/3.2), CapacityImpliedShareLowPct/HighPct
  (60/71), SevereHeldFamily{Pulse,Ofdm}{AFive,Iqformer,Mcldnn,Sidecar},
  SevereHeldRegime{One,Two}{...}, SevereControl{IqZero,IqShuffle,
  NoSideHead}RepairRatio, SevereControlNoSideHeadLoss.

## Manuscript (main.tex) changes by self-check item
- §2.1: removed all five "joint representation-and-fusion" claims
  (abstract, contribution 4, claim ladder, controls, conclusion; grep = 0).
  Controls B/C now reported as inference-time lesion sensitivity, with the
  2.02--4.12x loss/repair range and attribution deferred to a retrained
  no-I/Q ablation. Control A retained.
- §2.2: V-F adds SIR fixed at -15 dB (same as campaign corner cells),
  per-family paragraph (OFDM-like floor vs learnable pulse family,
  confounding of OOD-ness with family difficulty), and the three-factor
  compound-shift limitation sentence.
- §2.3: restored -2.84 / -18.1 pp (V-F capacity sentence), in-campaign
  sidecar +11.77 / +4.37 pp sentence, Table III split into Model/intervention
  | Macro-F1 | Reference | Contrast with C1 row fully disclosed.
- §2.4: IV-C footnote with capacity extrapolation arithmetic (2.52x and 5.90x
  ratios, per-doubling slopes, 2.22 doublings, implied 2.8--3.2 pp =
  60--71% share, concavity => lower bound).
- §3.4 style surgery: abstract 230 words, longest sentence 33 words, zero
  em-dash appositives, "below MCLDNN / above IQFormer" in one sentence;
  Evidence-status box 37 words; Limitations merged to six bullets without
  count words and containing "The control chain constrains the explanation
  without establishing a mechanism."; rather than = 7; best interpreted as
  = 0; deliberately = 0; ", not " = 3; prospectively frozen = 1;
  reference [26] title article fixed in references.bib.

## Supplement (supplement.tex) changes
- §I: single protocol sentence ("All post-campaign analyses ... forward-only
  on frozen checkpoints; no model is retrained or re-selected"); six
  scattered "no model is retrained" sentences deleted.
- §XIII: new table tab:s5decomp (family/regime/pooled x 4 models) plus
  MCLDNN reversal explanation (pulse family 55.34% intrinsically easier for
  I/Q models).
- §XV: Control C softened; new table tab:lesionratios with loss/repair
  ratios and the adopted limitation paragraph; decision rule (c) restated
  as not well posed for a jointly trained path; retrained permanently-zeroed
  sidecar named as the decisive follow-up.
- §XVI: SHA-256 digests certify content integrity, not temporal ordering.
- Artifact index: a22 entries added. "deliberately" removed (count = 0).

## Build and QA
- `paper/main.tex` -> main.pdf: 10 pages, 549806 bytes, log clean
  (no undefined control sequence / citation / reference, no LaTeX Error).
- `paper/supplement.tex` -> supplement.pdf: 15 pages, 311171 bytes, log clean.
- Command: `latexmk -pdf -g -interaction=nonstopmode`.

## Frozen V4.10 artifacts (SHA-256)
- `output/pdf/tvt_rank_instability_representation_repair_V4_10.pdf`
  D1A4F7B586EBDFA835CC0DDA675CF30500CD0C72784BA78D8C342C93D03C3608
- `output/pdf/tvt_supporting_material_V4_10.pdf`
  B732FCDD01990D16D0A3DC11F1D5EC91B5857D1A2DCE41EA77AD8E88CF6BD62E

## Open follow-up (not in V4.10)
Second batch per self-check §2.1: retrain the sidecar variant with the I/Q
input permanently zeroed on the frozen cache (same architecture, same
parameter budget, 5 seeds). This is the only experiment that can support or
refute the I/Q-information attribution; the self-check labels it optional
but decisive (acceptance estimate 65--71% -> 70--76%).
