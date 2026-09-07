# TVT V4.8 Major Revision Operation Log (2026-08-20)

Status: **FROZEN**. This log records the execution of
`TVT_V4_8_融合终审意见_大修执行总方案.md` (superset of the V4.7 plan) on top
of the frozen V4.6 full reposition, producing the V4.8 manuscript and
supporting material.

## 1. Frozen artifacts

| Artifact | Path | Pages | SHA-256 |
|---|---|---:|---|
| Manuscript V4.8 | `output/pdf/tvt_rank_instability_representation_repair_V4_8.pdf` | 10 | `e969fcc898563298d76b775fa997c6ce24785979d392b2afcd03c95bd2fc0a66` |
| Supporting material V4.8 | `output/pdf/tvt_supporting_material_V4_8.pdf` | 13 | `efa4a6a89ee1d37146d4b6479f6934a774F16056bf7a1d65258f5db6576a90f3` |

Build: `latexmk -pdf -interaction=nonstopmode` in `paper/`, no `!` errors;
main 10 pages / 536,256 bytes, supplement 13 pages / 294,029 bytes.
Title retained per Batch 5 (control verdict is mixed, so no polarity
change): *Rank Instability and Representation Repair for Automatic
Modulation Classification under Structured Interference*.

## 2. New computation executed this round (Batch 2 only)

Zero training, zero new simulation; frozen checkpoints + frozen
`standards/cache_s5_severe_held_v1/` forward inference only
(cache digest `e655abee…`, 6,400 sources per regime per manifest
`view_count//2`; SIR fixed at $-15$ dB; R1 = held jammer families
(pulse/ofdm_like) + TDL-A/C/D + 180 km/h; R2 = held channels (TDL-B/E) +
held jammer families + 250 km/h; SNR grid $-10$ to $+18$ dB).

Decision rule frozen pre-run in
`docs/S6_SEVERE_HELD_INFERENCE_CONTROLS_PREREG.md`. **Verdict:
mixed_representation_and_fusion** — conditions (a) and (b) hold, (c) fails.

Pooled seed-mean macro-F1 (%): A5 20.72, M 21.77, L 20.89, sidecar 25.09,
iq_zero 12.80, iq_shuffle 16.25, no_side_head 7.09. Matched-five levels:
A5 20.718 vs sidecar 26.109. Full−zero 12.28 pp; full−shuffle 8.84 pp;
side-head disabled retains −13.62 pp deficit. Paired contrasts vs A5
(pooled, 95%): M 1.05 [−1.47, 3.16], L 0.17 [−2.47, 2.51], iq_zero −7.91,
iq_shuffle −4.47, no_side_head −13.62.

Locked manuscript language: "The repair is best interpreted as a joint
representation-and-fusion intervention rather than an isolated
I/Q-information effect." Artifacts:
`analysis_zero_compute/outputs/a21_s6_controls_summary.json` plus the three
CSVs (`a21_control_levels.csv`, `a21_control_paired_contrasts.csv`,
`a21_control_matched5_levels.csv`).

## 3. Manuscript and supplement changes

- **Batch 1 (P0)**: severe-held construction disclosed in V-F (fixed SIR,
  two regime factor sets, source-disjoint, checkpoints unchanged);
  "frozen campaign-fitted coefficients applied unchanged" everywhere
  (refitted eliminated for the severe-held comparison); macro-F1
  conventions unified — headline is seed-wise paired (A5 trails IQFormer
  2.57 pp, MCLDNN 13.03 pp), window-pooled (−3.12 / −12.45 pp) reported as
  matching-conclusion cross-check and never mixed within one contrast;
  leave-−15 wording corrected (MCLDNN full lowers RMSE but sign 2/4 vs
  SIR-only 4/4; IQFormer full no better than SIR-only); true preregistration
  chronology written (transfer freeze → boundary exposed → repair freeze →
  controls freeze), internal stage codes removed from the formal text;
  operating-point cooling (4.37-pp relative repair at an extreme point where
  all models remain low in absolute macro-F1; relative robustness repair,
  not deployable accuracy); checkpoint-selection caveat and comparator
  convergence added to Limitations.
- **Batch 3**: V-C compressed to a campaign diagnostic (Fig. 3 moved to
  supplement as `fig:envelope_supp`; break-even and selector detail reduced
  to descriptive references / supplement); V-F expanded as the key section.
- **Batch 4**: Intro learned-receiver positioning; novelty stated as the
  evidence chain; contributions rewritten; VI-C renamed Cross-Regime
  Transport Boundary; VI-E renamed Vehicular and Learned-Receiver Design
  Implications; Conclusion lands on ranking stability and the low-cost
  information path with the mixed attribution.
- **Supplement**: audit-only positioning in header/abstract/scope;
  new sections `Severe-Held Frozen-Inference Attribution Controls`
  (`tab:controls`) and `Preregistration Provenance` (`tab:preregp`);
  transfer section rewritten with construction, convention paragraph, and
  frozen table note; leave-−15 corrected; standalone compilation via 18
  `\providecommand` placeholders before `\input{v41_envelope_macros.tex}`.

## 4. Macro / data layer

`paper_data_layer/build_paper_numbers.py` regenerated
`paper_data_layer/outputs/paper_macros.tex`: **283 keys** (67 sealed, 216
exploratory/prospective), all with SHA-256 provenance. New keys include
`SevereControlDecisionPattern`, `SevereControlLevel*`,
`SevereControlIq{Zero,Shuffle}Loss`, `SevereControlDiff*{,Low,High}`,
`SevereControlMatchedLevel*`, `SevereSeedMeanGap*`,
`SevereRepairResidualGap*`, `SevereHeldSourcesPerRegime`.
`validate_paper_numbers.py` on main.tex returns to the historical baseline
(13 pre-existing violations identical to V4.5/V4.6; no new violations from
this round — two hand-written numbers introduced mid-edit were macro-ized
before freeze).

## 5. Preregistration provenance (as stated in `tab:preregp`)

| Stage | Document | SHA-256 |
|---|---|---|
| Transfer | `docs/S5_A_SEVERE_HELD_PREREG.md` | `2759558b50a046eaa60cfaf0997744a80ad84b28b8ddbe3b713621bcfbc27405` |
| Repair | `docs/S5_R1_PREREG.md` | `058cc76dd693b7152e61dac44690327bb22fe5b08ee20939611d90e5f82d6fa3` |
| Controls | `docs/S6_SEVERE_HELD_INFERENCE_CONTROLS_PREREG.md` | `be0dc26ccff5b69f69c27a1a52fc63d0353f94c468f3c1db0f9fea16bc452c80` |

No commit-level timestamp service was used (repository has no git history);
the supplement states this explicitly and grounds the freeze ordering in
these documents plus the operation logs, per V4.8 §P0-5.

## 6. QA search checklist (V4.8 §27) — results

Searches across `paper/main.tex` and `paper/supplement.tex`:

- `refitted/refit`: supplement only — 31-cell clean-sentinel sensitivity
  refit and campaign-internal leave-one-SIR-stratum-out refit (both
  legitimate campaign-internal analyses), plus the P0-2 phrase "without
  refitting to the independent severe-held data". No severe-held refit
  claim anywhere. PASS.
- `S5/R1` internal codes: 0 matches in main.tex; supplement retains only
  artifact paths (`cache_s5_severe_held_v1/`, prereg filenames) as required
  provenance. PASS.
- `falsif / preserves the severe-region / strong repair / restores /
  capacity alone`: 0 matches. PASS.
- `switching rule`: 3 matches, all negative/denial contexts. PASS.
- `pooled / seed-wise / seed mean`: every occurrence labels its convention;
  conventions never mixed within a single contrast. PASS.
- `break-even`: main text keeps qualitative descriptive references without
  numerical promotion; numbers live in supplement. PASS.
- `Supporting Material`: main is self-contained (header states no claim
  depends on the supplement); supplement carries the audit-only statement.
  PASS.
- Unicode minus/en-dash in numeric text: 0 matches. PASS.
- Legacy 81-cell macros: 0 matches. PASS.

## 7. Freeze criteria (V4.8 §29) — item-by-item

1. Severe-held construction self-sufficient in main text: **PASS** (V-F).
2. Frozen vs refitted zero contradiction: **PASS**.
3. Macro-F1 convention zero mixing: **PASS** (seed-wise paired headline;
   pooled labeled cross-check; convention sentence in Statistical
   Procedures).
4. Chronology truthful: **PASS** (three sequential freezes, no implied
   single upfront preregistration).
5. Provenance auditable: **PASS** (SHA-256 table; no overstated timestamp
   claim).
6. Fig. 3 not core main-text content: **PASS** (moved to supplement).
7. Operating point honest: **PASS** (cooling language in Abstract, V-F,
   Conclusion).
8. Three controls completed and reported per frozen rule: **PASS**
   (mixed verdict stated, condition (c) failure disclosed).
9. Attribution factual: **PASS** (joint representation-and-fusion language;
   no I/Q-information-only claim).
10. Main at 10 pages: **PASS**.
11. Supplement carries no main claim: **PASS** (audit-only positioning).

## 8. Known accepted residuals

- 13 historical validator violations in main.tex (lines ~248/466/468/471/
  481/533/629/769 etc.), unchanged from V4.5/V4.6 baseline and not touched
  by this round.
- Supplement hand-written descriptive numbers from earlier versions
  (e.g. 12,820 / 152,000 windows) kept as-is to limit change surface; none
  contradicts the new macros.

End of log. V4.8 frozen 2026-08-20.
