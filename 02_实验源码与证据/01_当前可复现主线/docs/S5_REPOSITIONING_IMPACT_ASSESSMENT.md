# S5 Repositioning Impact Assessment (2026-08-20)

Purpose: before any manuscript edit, quantify what a repositioning to
"Rank Instability + Representation Repair" would touch, what stays sealed,
and what the page/risk profile is. This document changes nothing.

Evidence inputs:
- S5-A: independent −15 dB severe-held verdict **fail** (A5 pooled gain
  −3.12 pp vs IQFormer, −12.45 pp vs MCLDNN; frozen envelope sign skill at
  chance; `docs/S5_A_OPERATION_LOG.md`).
- S5-R1: sidecar frozen repair verdict **strong_repair** (pooled +4.37 pp
  over A5, CI [+2.96, +5.90], 10/10 seeds, 8/8 SNR strata; overtakes
  IQFormer on the severe-held set but still trails MCLDNN by ~8.7 pp;
  `docs/S5_R1_OPERATION_LOG.md`).

## 1. What is now true vs what the manuscript currently says

Conflicts that a repositioning must resolve (all currently hedged, none
asserted as sealed):

| Manuscript location | Current claim | New fact | Action needed |
|---|---|---|---|
| Abstract L65-67 | severe corner: A5 gains SevereGainMcldnn / SevereGainIqformer | in-sample only; fails on independent severe-held | reframe as campaign-internal, add falsification sentence |
| V-B L546-556 | severe rank reversal "reinforce[d]" by post-hoc checks | does not transfer across regimes | keep numbers, add transfer failure |
| V-C envelope | "regime-dependent transport" | frozen fit fails on independent −15 set (sign acc ≤ 0.58) | one sentence: independent severe falsifies −15 transport |
| V-F L842-849 | sidecar "remains ahead of IQFormer at −15" + "preserves the structured-interference advantage" | the second clause is now false for A5; sidecar itself still leads IQFormer on independent severe-held (+1.79 pp) but trails MCLDNN | replace with S5-R1 numbers |
| Limitations item 7 L956-959 | epoch cap "unlikely explanation for the severe-region A5–MCLDNN advantage" | advantage itself does not exist out of sample | reword |
| Limitations item 8 L960-966 | leave-15-out preserves reversal vs MCLDNN | independent −15 held falsifies it | update with S5-A verdict |
| Conclusion L982-984 | "reproducible compact-model advantage in one corner" | not reproducible across independent regimes | reword |

Sealed facts unaffected: A5–A0 whole-configuration positive contrast,
component contrasts bounded, clean-retention gate fail, capacity ladder,
Tier-2 clean/hard sidecar numbers, all comparator-fidelity evidence.

## 2. Two options

### Option A — full reposition (instruction §17/§18)
New title: "Rank Instability and Representation Repair for Automatic
Modulation Classification under Structured Interference". Narrative: sealed
A5–A0 contrast → campaign-internal severe rank reversal → prospective
independent severe-held falsification of the stable-boundary reading →
sidecar strongly repairs the independent severe degradation → width-only
capacity cannot explain the boundary.
- Touches: title, abstract, keywords, intro contributions list, V-B/V-C
  framing, V-F, Discussion VI-A/VI-B/VI-E, Limitations, Conclusion.
- Estimated main-text churn: ~30-40 line edits across 9 locations + 2-4 new
  macros; net length controllable within 10 pages (existing hedges already
  consume space that the new story reuses).
- Risk: largest edit surface on a frozen-quality text; needs a full
  re-audit pass (search list, macro sync, page check) before re-freeze.

### Option B — bounded disclosure (recommended)
Keep title and structure. The manuscript already says the envelope is
"regime-dependent ... rather than uniform", the severe corner is "in-sample
stratification", and break-even thresholds are "conditional extrapolations".
S5-A/S5-R1 convert those hedges into measured outcomes:
- add one subsection-sized paragraph to V-F (or VI-D): independent
  severe-held falsification (S5-A numbers) + strong sidecar repair (S5-R1
  numbers), explicitly "prospective post-hoc robustness validation";
- fix the 7 conflict sites above with 1-2 sentences each;
- full tables → new supp sections.
- Estimated main-text churn: ~15-20 line edits + ~6 new macros; lower
  page-risk; keeps the sealed story intact while removing every claim the
  new evidence contradicts.

Both options keep the paper honest; Option A changes the story's center of
gravity, Option B keeps the envelope as a bounded diagnostic (which the
Discussion already frames it as) and elevates the repair chain.

## 3. Supplement additions (either option)

New sections after current §XII (Artifact Index moves last):
- **XIII/§ new — Independent Severe-Held Validation (S5-A)**: protocol table
  (two regimes, 6400 sources each, factor support), 64-cell per-SNR gain
  table, envelope-vs-constant/SIR-only RMSE comparison, selector audit,
  fail verdict wording, prereg + cache digest + disjointness provenance.
- **XIV/§ new — Severe-Held Representation Repair (S5-R1)**: checkpoint
  audit summary (10 seeds, SHA-256 pointer), model-level table (4 models ×
  2 regimes + pooled), paired contrasts with CIs, per-SNR table, per-class
  delta table, descriptive gap-recovery ratios, strong_repair wording.
- Artifact Index additions: `standards/cache_s5_severe_held_v1` (digest),
  `artifacts/s5_severe_held_v1`, `artifacts/s5_r1_sidecar_severe_held_v1`,
  a19/a20 scripts, prereg + operation logs.
- Supp page budget: currently 10 pages; +2 sections ≈ +1.5-2 pages → ~12
  pages. Supp has no 10-page constraint (confirm with venue rules before
  re-freeze; TVT allows extended supplementary).

## 4. Page-risk assessment (main, 10-page cap)

- Option B: net ~+3-6 lines main text after trimming now-invalid hedging
  sentences; compiles to 10 pages with high confidence (prior compressions
  have headroom documented in the build notes).
- Option A: neutral-to-positive length but requires re-verifying every
  figure caption and macro reference that mentions "operating envelope" as
  the central object; moderate risk of a 10.1-page overflow needing another
  compression pass.
- Either option: `results_auto.tex` gains ~6-8 macros; the paper-number
  validator needs the corresponding derivations (existing pattern:
  `paper_data_layer/build_paper_numbers.py` + `validate_paper_numbers.py`).

## 5. Sealed-evidence compatibility checklist

- No sealed artifact rewritten: S5-A/S5-R1 use new cache + new prediction
  directories; sealed composite untouched (verified by the operation logs).
- Evidence classes: both marked prospective/post-hoc robustness validation;
  neither enters the sealed confirmatory family; Eq.(8) envelope numbers
  stay as historical in-campaign fit.
- The existing "leave-−15-dB-out" supp section remains valid as an
  in-campaign extrapolation test; S5-A supersedes its interpretation and
  the cross-reference must say so.
- Historical identities (V4.4 FINAL-S4 freeze) unchanged; any new freeze is
  a new version label (e.g. V4.5 S5R1), never a rewrite of V4.4.

## 6. Open risks and questions

1. **Venue supplement length**: confirm TVT supplementary has no page cap
   before adding ~2 sections.
2. **MCLDNN gap remains**: even after repair, sidecar trails MCLDNN by
   ~8.7 pp on the severe-held set; the repositioned story must state this
   plainly (already required by §14 Level C wording).
3. **S5-C still open**: the training-budget sensitivity (instruction S5-C)
   would strengthen the "not a budget artifact" defense of S5-R1; optional
   follow-up, does not block Option B.
4. **Receiver-observable overlap proxy (S5-B)**: becomes more relevant if
   the envelope stays in the story (Option B) than if it is demoted
   (Option A); optional.
5. Macro naming: new macros must avoid digits per the TeX naming rule.

## 7. Recommendation

**Option B (bounded disclosure) first.** It resolves every conflict the new
evidence creates, preserves the 10-page envelope with minimal churn, and
keeps the door open to Option A's title change later if the operator wants
the stronger narrative pivot. If the operator prefers Option A, the same
supp sections and macros serve both; only the main-text framing differs.
