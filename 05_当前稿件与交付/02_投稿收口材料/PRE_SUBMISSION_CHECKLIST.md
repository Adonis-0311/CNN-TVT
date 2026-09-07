# TVT v2 pre-submission checklist

Status date: 2026-07-29

This checklist applies only to the v2 freeze and its exact local run. The v1
five-seed/55-fit checklist is historical and cannot unlock the v2 manuscript.
No machine-result item below is checked at delivery time because the formal
campaign has not run.

This file is a later submission checklist, not the authority for initial-draft
scientific closure. The latter is recorded in
`docs/TVT_EVIDENCE_CLOSURE_DELIVERY_2026-07-29.md`. Submission-only items below
do not reopen the simulation-only evidence chain.

## Formal machine evidence

- [ ] The only supported fresh compute entry point,
  `run_v2_after_gpu_free.ps1`, exited zero.
- [ ] The 10k, 30k, and 100k cache manifests have their exact expected
  designations, immutable hashes, globally disjoint source identities, valid
  checksums/components, and identical frozen validation identities.
- [ ] Learning-curve evidence contains A0/A5 at all three scales and all five
  fixed seeds, with every required per-epoch train/validation metric.
- [ ] Learning-curve evidence did not select a scale or change
  hyperparameters; the formal scale remains 100k.
- [ ] All 12 models × 10 fixed seeds complete exactly once.
- [ ] Every fit has an eligible validation-selected checkpoint, no fallback,
  and exact result/checkpoint/history agreement.
- [ ] Start/end executable-source fingerprints and freeze hashes agree.
- [ ] Every model/seed/regime prediction NPZ exists and matches cache digest,
  split, source IDs, labels, profile indices, SNR, and SIR.
- [ ] The simultaneous confirmatory family is exactly A5−A0, A3−A3′, and
  A5−A6 and rederives exactly from prediction bundles.
- [ ] All exploratory contrasts remain labeled exploratory.
- [ ] All calibrated OOD, ADC10, ADC12, synchronization, and seen/held clean
  branches are complete and honestly reported.
- [ ] The six-family occupancy/gain diagnostic is complete; positive
  mechanism wording is used only if its preregistered directional conditions
  pass.
- [ ] `v2_scientific_release_gate.json` reports a passing, exact hash-bound
  scientific gate.

## Paper promotion and build

- [ ] `validate_v2_paper_release.py --write-macro-manifest` generated
  `tvt_submission/formal_macro_values_v2.json` from the exact passing run.
- [ ] `validate_v2_paper_release.py --validate-macro-manifest` rederived and
  validated that exact existing manifest without writing.
- [ ] No paper value was typed, copied, rounded, or replaced by hand.
- [ ] The macro manifest covers every required v2 headline, regime, clean,
  stress, A3′, confirmatory/exploratory, mechanism, complexity, and latency
  macro with source hashes.
- [ ] Human authors inspected every gate outcome and approved only supported
  claims.
- [ ] An author explicitly ran `validate_v2_paper_release.py --write-release`;
  the training queue did not perform this action.
- [ ] `validate_v2_paper_release.py --validate-release` passed immediately
  before the public manuscript build.
- [ ] `paper/results_auto.tex` and `paper/release_lock.json` revalidate
  atomically against the exact run, learning evidence, macro manifest, freeze,
  scientific gate, and source tree.
- [ ] The public LaTeX branch contains no controlled placeholder or internal
  banner.
- [ ] `validate_paper_build.py --mode release` passes with no fatal error,
  undefined citation/reference, overfull box, rerun requirement, stale
  dependency, authorship placeholder, or release-lock mismatch.
- [ ] Human authors visually inspected every final PDF page and the exact
  source archive.

## Scientific claim audit

- [ ] The CSSL comparator is described only as an official-architecture
  supervised adaptation, not a full reproduction or official result.
- [ ] IQFormer-inspired and MCLDNN retain their bounded provenance labels.
- [ ] A5−A0, A3−A3′, and A5−A6 interpretations match the frozen composite
  interventions; no exploratory result is presented as confirmatory.
- [ ] OOD and receiver claims follow their predeclared consequential versus
  noninferiority branches.
- [ ] Clean retention remains stratified into seen A/C/D and held B/E
  profiles.
- [ ] Occupancy/gain wording follows the actual directional result, including
  an honest null or contrary result if observed.
- [ ] No screening, diagnostic, oracle, source-mutated, historical-v1, or
  manually entered number appears as formal evidence.
- [ ] No claim exceeds the simulation-only TDL scope or implies SDR/field,
  real-time, waveform-separation, complete V2X geometry, or hardware transfer
  evidence that was not collected.

## Human and external gates

- [ ] Names, affiliations, corresponding author, ORCID, funding,
  acknowledgments, and conflicts are supplied.
- [ ] Patent/publication timing is reviewed by a qualified person.
- [ ] Every equation, table, figure, numerical statement, and limitation is
  checked by a human author.
- [ ] Citations, DOI/year/volume/pages, 3GPP versions, source commits,
  licenses, and comparator labels are verified against primary sources.
- [ ] Current TVT template, page limit, fees, disclosure, data/code, and
  submission rules are checked on the upload date.
- [ ] Any required disclosure of AI-assisted language or code work is
  prepared by the human authors.
- [ ] If the submission expands beyond the current scope, RadioML evidence is
  provenance-verified and prospectively reported. The current draft makes no
  public-benchmark transfer claim.
- [ ] If the submission adds an MFENet numerical row or superiority claim, it
  uses an authorized hash-pinned implementation/contact trail. The current
  draft treats MFENet only as related work.
- [ ] Human authors substantively authored and approved the final PDF/source
  archive and made the final upload decision.
