# TVT V5 Major Revision Operation Log (2026-08-21)

Base: V4.11 (sealed by user: `output\pdf\tvt_rank_instability_representation_repair_V4_11.pdf`,
`tvt_supporting_material_V4_11.pdf`). V5 incorporates the preregistered
retrained zeroed-I/Q ablation (S7) and upgrades the attribution claim from
inference-time dependence to attribution supported.

## 1. Preregistration (frozen before any retraining)

- Plan: `docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md`, SHA-256 first 16 hex
  `d4c9ecccb652231a`.
- Decision rule: attribution supported if the 95% CI upper bound of the
  zeroed-versus-A5 contrast (pooled seed-mean macro-F1 on the severe-held
  set) is below +1.0 pp; attribution refuted if the point estimate is at
  least +3.0 pp; otherwise unresolved.

## 2. Execution

- Variant: `IQZeroLightweightIQSidecarVIMD` — identical architecture and
  parameter count (46,794) to the full sidecar; the received-I/Q branch
  receives zeros at every training and inference step. Launcher:
  `analysis_zero_compute/tier2_gpu/run_tier2_iqzero_experiment.py`.
- Training: 5 matched seeds (17, 29, 43, 71, 101), 30 epochs each, same
  cache/recipe as the full sidecar; run id `tier2_iq_sidecar_iqzero_v1`;
  all seeds completed (selected epochs 27–30, best val loss ≈ 1.10).
- Severe-held inference: `analysis_zero_compute/a23_s7_iqzero_severe_held.py`,
  predictions under `artifacts/s7_iqzero_severe_held_v1/`.
- Campaign sanity: `analysis_zero_compute/a24_s7_campaign_levels.py`.

## 3. Results (verdict: attribution_supported)

Severe-held pooled seed-mean macro-F1 (%):

| Condition | Level | vs A5 (pp, 95% CI) |
|---|---|---|
| A5 | 20.72 | — |
| Sidecar retrained, I/Q zeroed | 20.18 | −0.54 [−1.34, +0.26] |
| Full sidecar (matched seeds) | 26.11 | +5.39 [+3.54, +7.66] |

CI upper bound +0.26 < +1.0 → preregistered rule returns
attribution_supported. Training-failure explanation excluded: on the
campaign hard-interference split the zeroed variant lands at the A5 level
(43.38% vs 43.59%, −0.21 pp) while the full sidecar gains (48.35%).

## 4. Data layer

- `paper_data_layer/build_paper_numbers.py` extended with the S7 block
  (a23 + a24 sources, all evidence_class = exploratory); builder re-run:
  337 keys (69 sealed, 268 exploratory), 39 sources; 15 new `Retrained*`
  macros (TeX macro names avoid digits, so `S7` prefix not used).

## 5. Manuscript edits (paper/main.tex)

- Version header V5; abstract control sentence upgraded to retrained
  ablation; contribution 4 and claim boundary updated; Table III gains
  panel (c) (two retrained rows, column header omitted, five matched seeds
  stated in caption); controls paragraph reports the retrained contrast and
  the campaign hard-split sanity; Discussion "Representation Repair"
  subsection updated (removed stale "remains open" wording); conclusion
  updated.
- Length control: compressed Discussion (Cross-Regime, Representation
  Repair, Vehicular implications), Limitations, and conclusion to restore
  the 10-page limit after the added table rows.

## 6. Supplement edits (paper/supplement.tex)

- Version header V5; new section "Preregistered Retrained Zeroed-I/Q
  Ablation" (label sec:retrained) with Table `tab:retrained`;
  recorded-outcome paragraph in the controls section updated;
  Prospective-Freeze Provenance updated to four frozen plans (S7 row with
  digest `d4c9ecccb652231a`, freeze order amended); artifact index entries
  for the launcher, checkpoints, a23/a24 outputs, and the S7 plan.

## 7. QA

- Compile: `latexmk -pdf -g` both documents; main 10 pages (549039 bytes),
  supplement 15 pages (315760 bytes); no undefined macros/references/citations.
- Word-frequency gates: abstract ≈ 227 words (≤ 230); longest abstract
  sentence 35 words (≤ 35); ", not " = 3 (≤ 3); "rather than" = 8 (≤ 10);
  "deliberately" / "joint representation" / "best interpreted as" = 0.
- No stale "remains open" / "inference-only controls" phrasing remains.

## 8. Freeze

- `output\pdf\tvt_rank_instability_representation_repair_V5.pdf`
  SHA-256: `BE51237B07AB2A881B4145C30989AAF83C034259BB37153818A97E363AB7BD27`
- `output\pdf\tvt_supporting_material_V5.pdf`
  SHA-256: `918E2415B449EBEAA598488332628841B4730E7A4D07FE1F9192BA8BE5225CFD`
