# Internal Reviewer #2 Red-Team Audit

**Recommendation:** Major revision; do not submit the current evidence-lock
manuscript.

**Audit date:** 2026-07-28  
**Review role:** independent methodological and reproducibility red team  
**Scope:** `paper/main.tex`, `paper/references.bib`,
`paper/EVIDENCE_LEDGER.md`, `paper/SUBMISSION_READINESS.md`, the schema-2
factor policy/manifest, A0--A7 registry, model/loss/training/evaluation code,
standard runner, cochannel identifiability record, baseline audit, and the
current seven-page rendered PDF.

This review changes no manuscript, code, manifest, or experiment. It does not
recommend promoting any diagnostic result or number into the paper.

## Overall assessment

The manuscript has an unusually good evidence-lock discipline. The complex
STFT, FiLM conditioning, scalar temperature/overlap/residual gates, tri-mask
simplex, fixed teacher equations, exact-source contrastive denominator, loss
weights, and A0--A7 registry are mostly faithful to the implementation. The
paper also correctly disclaims time-domain source reconstruction, coherent
energy conservation, V2X system compliance, and unreferenced-field teacher
use.

The submission is nevertheless not ready. The principal reason is not missing
polish but several protocol contradictions:

1. the SNR/SIR protocol printed in the paper is not the protocol in the
   schema-2 cache policy;
2. the abstract says paired inference although headline prediction uses only
   `view1`;
3. the headline protocol excludes unanchored cochannel collisions while the
   motivation and the only teacher figure visually center a cochannel record;
4. the teacher measures component-power dominance, not modulation evidence;
5. the standard runner has no verified recent structured-interference
   comparator and no baseline-specific tuning sensitivity;
6. the paper's “validation-locked comparator” wording does not describe the
   runner's explicit predeclared reference;
7. the evidence gate does not prevent a formal run whose epoch budget ends
   before any checkpoint becomes selection-eligible; and
8. the vehicular claim rests on generic TDL primitives plus Doppler, with no
   trajectory, system mapping, or SDR transfer evidence.

These are credible Reviewer #2 rejection grounds even if later accuracy is
high.

## Major findings

### R2-M01 — Printed SNR/SIR ranges contradict the executable nine-split policy

**Evidence**

- `paper/main.tex:512-514` states SNR
  `{-12,-8,-4,0,4,8,12,16,20}` dB, SIR
  `{-15,-10,-5,0,5,10,15,20}` dB, and says training samples draw
  continuously within the range.
- `src/vimd_amc/standards/cache.py:74-76` fixes SNR to
  `{-10,-6,-2,2,6,10,14,18}` dB, ordinary active SIR to
  `{-10,-5,0,5,10}` dB, and hard SIR to `{-15,-10,-5,0}` dB.
- `docs/FACTOR_ISOLATED_CACHE_PROTOCOL.md`, “Locked factor policy,” describes
  the same discrete, split-specific grids.
- `standards/cache_factor_micro_v4/manifest.json` serializes those grids in
  `preregistered_split_policy`; it does not serialize continuous training
  sampling.

**Why major**

This makes the experiment irreproducible from the manuscript and can change
both task difficulty and the interpretation of the hard region.

**Required exact fix**

Replace the global-grid paragraph with the values read from the final
headline manifest, separately stating the ordinary active grid, hard-SIR
grid, and clean invalid-SIR semantics. Delete “continuously draw” unless a new
frozen cache actually implements continuous training draws. Add the final
cache digest as the single source of truth.

### R2-M02 — “Paired inference” is false for the implemented headline path

**Evidence**

- `paper/main.tex:64-67` describes the evaluation as using “paired inference.”
- `paper/main.tex:241-252` correctly defines paired views for training.
- `src/vimd_amc/evaluation.py:285-311` constructs the prediction bundle from
  `batch["view1"]` only.
- `src/vimd_amc/training.py:249-299` uses both views during training and XCC,
  which is not inference fusion.
- `paper/main.tex:238-239` says inference requires only one `x`, consistent
  with the code and inconsistent with “paired inference.”

**Why major**

Two-observation inference and single-window inference are different receiver
contracts. A paired-view claim can be interpreted as test-time ensembling and
an unfair advantage.

**Required exact fix**

For the current implementation, replace “paired inference” with
“single-view inference with source-paired statistical comparisons,” and state
that the second view exists only for training and data-identity audits.
Alternatively, implement and preregister a deployment-realistic two-window
fusion protocol for every model; do not change this after viewing test
results.

### R2-M03 — Cochannel exclusion is mathematically sound but narratively
contradicted by the abstract and Fig. 2

**Evidence**

- `paper/main.tex:55-66` opens with cochannel collisions as the motivating
  impairment and does not immediately state that they are excluded from the
  evaluation.
- `paper/main.tex:504-511` correctly excludes unanchored in-taxonomy
  cochannel collisions and states the source-exchange lower bound.
- `docs/COCHANNEL_IDENTIFIABILITY_BOUNDARY.md` and the isolated executable
  counterexample support that exclusion without entering headline evidence.
- The rendered Fig. 2 on PDF page 4 explicitly labels its jammer
  “cochannel,” although the caption calls it an auditable held-out-channel
  cache record and the current headline protocol excludes this case.
- `standards/cache_factor_micro_v4/manifest.json` records `cochannel` as
  excluded from primary modulation classification.

**Why major**

The only physical-teacher illustration invites the reader to believe that a
target route is meaningful in precisely the source-exchange case the paper
declares unidentifiable. Simulation bookkeeping can name a target; a
mixture-only receiver cannot recover that hidden identity.

**Required exact fix**

Use an admitted structured-jammer record for the method figure. In the first
abstract sentence, distinguish general motivation from evaluated scope:
unanchored in-taxonomy collisions motivate the boundary but are excluded from
headline single-label results. Keep the impossibility result as a task
definition, not as evidence of algorithm capability.

Suggested scope sentence:

> Unanchored cochannel emitters drawn from the target taxonomy are excluded
> from the single-label evaluation because a pre-designated target is not
> identifiable under source exchange; the evaluated interferers are the
> structured families declared in the frozen manifest.

### R2-M04 — The teacher does not identify “modulation-dominant” evidence

**Evidence**

- `paper/main.tex:59`, `151-157`, `345-346`, and `374-375` repeatedly call
  the first route modulation-dominant.
- `src/vimd_amc/models/spectral.py:169-236` constructs the route only from
  tracked target, jammer, and unexplained STFT powers. It has no modulation
  cue, symbol likelihood, class gradient, or class-conditional statistic.
- Equal target/jammer power is routed to overlap even if one component has
  much stronger class-discriminative structure.
- `paper/main.tex:782-785` correctly admits that this is a
  simulation-component allocation target rather than a conserved mixture
  decomposition.

**Why major**

“Target-power dominant” is an auditable physical statement.
“Modulation-dominant” is a semantic claim that the teacher cannot establish.
The distinction is central to the novelty claim.

**Required exact fix**

Rename the teacher routes throughout as `target-power-dominant`,
`jammer-power-dominant`, and `unexplained/power-ambiguous`. Reserve
“modulation branch” for the task-trained downstream encoder. State explicitly
that teacher agreement alone cannot prove preservation of discriminative
modulation cues; that requires A2/A3 downstream and mechanism evidence.

### R2-M05 — The cache-audit table is not bound to the current ledger sentinel

**Evidence**

- `paper/main.tex:561-580` prints a manually typed integration-sentinel table.
- Its component, SNR, and SIR error values differ from the micro-v4 values
  recorded in `docs/FACTOR_ISOLATED_CACHE_PROTOCOL.md`, “Executed micro
  sentinel.”
- The table gives no cache version or digest.
- `paper/EVIDENCE_LEDGER.md:C11` identifies micro-v4 as the current schema-2
  sentinel.

**Why major**

A reader cannot determine which immutable artifact supports the displayed
numbers. A manually typed audit table also violates the paper's own rule that
tables derive from machine-readable artifacts.

**Required exact fix**

Generate this table from one named manifest and print its version/digest in
the caption or reproducibility statement. Do not combine extrema from
different sentinels. If the table remains a plumbing illustration, label it
pipeline-only and keep it outside performance claims.

### R2-M06 — No eligible scientific result exists, and administrative sample
floors are not a power analysis

**Evidence**

- `paper/main.tex:683-690` intentionally withholds results.
- `paper/EVIDENCE_LEDGER.md:C6-C13` marks every effectiveness,
  generalization, SDR, and latency claim as unestablished or prohibited.
- `paper/SUBMISSION_READINESS.md:G2-G6` has no established algorithm,
  mechanism, held-factor, engineering, or multi-seed gate.
- `experiments/run_standard_experiment.py:122-149` labels sample thresholds as
  administrative floors, not prospective power justification.

**Why major**

The method/protocol manuscript is not a TVT contribution without eligible
effectiveness and generalization evidence. Passing a software gate cannot
substitute for power, effect stability, or engineering relevance.

**Required exact fix**

Before unlocking the paper, freeze a headline cache-size justification,
minimum meaningful paired effect, model/seed family, and stopping rule. Run
the untouched source tree. Populate tables only from an artifact whose final
machine-readable decision is `evidence_eligibility.eligible=true`. Do not
include smoke, screening, oracle-control, failed-candidate, or source-mutated
numbers.

### R2-M07 — The formal comparator set is not strong enough for a TVT
interference claim

**Evidence**

- `paper/main.tex:624-630` promises a classical control, MCLDNN,
  IQFormer-inspired, RepCCNet where verified, and MFENet or another recent
  interference-tolerant model.
- `experiments/run_standard_experiment.py:102-120` requires only A0--A7,
  MCLDNN, and IQFormer-inspired for headline eligibility.
- `docs/RECENT_INTERFERENCE_BASELINE_AUDIT.md` concludes that MFENet and
  RepCCNet are not presently reproducible from admitted primary evidence.
- `docs/BASELINE_AND_ABLATION_AUDIT.md` records that the common schedule has
  no architecture-specific tuning sensitivity.
- The direct classical feature control is outside the standard headline
  registry and is diagnostic-only.

**Why major**

MCLDNN is old and IQFormer is not an exact reproduction. Neither is a verified
recent structured-interference comparator. “Strongest verified recent” cannot
be filled under the current admission state.

**Required exact fix**

Obtain and verify one recent interference-tolerant implementation, or narrow
the claim to the exact local comparator set and remove “strongest” language.
For every literature model, report unified-protocol training plus a
predeclared architecture-specific tuning sensitivity. Keep official-paper
numbers separate. Either admit the transparent mixture-input classical
classifier to the formal runner under identical sources, or describe it only
as a separate non-headline control.

### R2-M08 — “Comparator locked on validation” does not describe the runner

**Evidence**

- `paper/main.tex:718-719` says the primary comparator is locked on validation
  before test evaluation.
- `experiments/run_standard_experiment.py:447-462` receives an explicit
  `--reference-model` and predeclared `--holm-candidates`.
- `experiments/run_standard_experiment.py:1231-1242` requires both for
  headline eligibility.
- `experiments/run_standard_experiment.py:1616-1629` explicitly records the
  reference as a paired-comparison anchor and does not claim it is strongest.
- There is no implemented validation-ranking procedure that chooses the
  reference.

**Why major**

Predeclared reference selection and validation-based comparator selection are
different protocols. The latter can create selection bias unless the
candidate family and rule are fixed.

**Required exact fix**

If the CLI reference is fixed before training, call it the
“predeclared primary reference.” If validation truly selects it, add a
machine-readable candidate family, metric, tie rule, and selection artifact,
and ensure test probabilities remain unopened until selection is frozen.

### R2-M09 — Formal eligibility does not enforce a valid checkpoint-selection
window

**Evidence**

- `src/vimd_amc/training.py:191-197` computes the first selection-eligible
  epoch after both ramps plus a minimum full-objective stage.
- `src/vimd_amc/training.py:329-345` falls back to the final state when no
  eligible checkpoint exists.
- `experiments/run_standard_experiment.py:1316-1331` validates positivity but
  does not require `epochs > selection_start_epoch`.
- `experiments/run_standard_experiment.py:1011-1285` does not make checkpoint
  eligibility/history a headline evidence gate.
- Validation selection uses single-view, un-smoothed modulation CE only
  (`src/vimd_amc/training.py:130-146`), a detail absent from the paper.

**Why major**

A formally designated run can theoretically pass administrative evidence
gates while never selecting an eligible validation checkpoint. The paper's
claim that early checkpoints cannot bypass the objective would then be false.

**Required exact fix**

Make formal eligibility require at least one checkpoint with
`checkpoint_selection_eligible=1`, persist the selected epoch and criterion,
and reject fallback checkpoints for headline use. State the exact validation
criterion: view used, loss, label smoothing, tie tolerance, patience, and
whether auxiliary losses influence selection.

### R2-M10 — Jammer auxiliary semantics are ill-defined for excluded and
held-out families

**Evidence**

- `standards/cache_factor_micro_v4/manifest.json` retains a nine-column jammer
  taxonomy including `cochannel`, while `protocol_exclusions` forbids
  cochannel from every primary split.
- The held jammer families are absent from training by design.
- `src/vimd_amc/losses.py:207-220` applies unmasked BCE to every jammer output
  column, so excluded and held-family columns receive only negative training
  targets.
- `paper/main.tex:670-673` says jammer F1/AUROC is evaluated directly but does
  not define supported-label filtering or interpret held-family outputs.

**Why major**

A column with no positive training support is not an ordinary supervised
class. Macro metrics can either reward trivial absence or present an
untrained held-family output as an open-set recognizer.

**Required exact fix**

Predeclare the auxiliary task separately from AMC. For seen-family evaluation,
compute metrics only over supported training labels and report support. For
the unseen-jammer split, use an explicit generic interference-presence or
unknown-family task if desired; do not interpret held-type logits as trained
family recognition. Remove or loss-mask excluded taxonomy columns in the
formal objective, and serialize the training-support mask.

### R2-M11 — McNemar/Holm wording obscures the actual multiplicity family

**Evidence**

- `paper/main.tex:643-647` says McNemar is restricted to “one fitted-model
  seed.”
- `experiments/run_standard_experiment.py:1851-1970` performs a separate exact
  McNemar test for every seed, regime, and predeclared candidate, then applies
  Holm only across candidates within each `(regime, seed)` group.
- No correction spans seeds or held-out regimes.
- `experiments/run_standard_experiment.py:2012-2091` separately uses the
  hierarchical seed/source bootstrap for headline differences.

**Why major**

If any one-seed McNemar result is used to support a general claim, repeatedly
testing seeds and regimes inflates the false-positive opportunity beyond the
stated Holm family.

**Required exact fix**

Make the hierarchical paired bootstrap the headline inference. Describe
McNemar as per-seed supplemental diagnostics, report all such tests, and
avoid an omnibus significance claim from them. If McNemar is inferential,
predeclare one seed or expand the family correction across every reported
seed/regime/candidate test. State the family explicitly in the table note.

### R2-M12 — The vehicular/TVT connection is currently parameteric, not a
validated vehicular system study

**Evidence**

- The title and conclusion claim vehicular modulation classification
  (`paper/main.tex:42`, `788-797`).
- The evidence uses generic TR 38.901 TDL profiles with speed-derived maximum
  Doppler (`paper/main.tex:485-498`).
- `paper/main.tex:495-498` admits that no geometry, blockage, or
  nonstationary trajectory is instantiated.
- `paper/SUBMISSION_READINESS.md:G0/G5` has no trajectory, receiver-utility,
  or engineering validation.
- The target taxonomy is a generic AMC taxonomy rather than a mapped NR-V2X
  or IEEE 802.11p waveform/procedure.

**Why major**

For TVT, merely setting a 5.9 GHz carrier and Doppler can look like generic
AMC relabeled as vehicular. The receiver role, mobility dynamics, and system
benefit remain untested.

**Required exact fix**

Either add a frozen vehicular scenario mapping with continuous trajectories,
time-correlated channels, speeds, packet/window semantics, and receiver
utility, or narrow the title and claims to AMC under TDL-profile simulations
parameterized by vehicular Doppler. Explain why the selected modulation
taxonomy and 1 MHz observation are relevant to the stated vehicular receiver.

### R2-M13 — No SDR/field evidence supports transfer, latency, or deployment

**Evidence**

- `paper/main.tex:491-493`, `765-766`, and `777-786` correctly make SDR
  conditional.
- `paper/EVIDENCE_LEDGER.md:C12-C13` prohibits hardware-transfer and real-time
  claims.
- `paper/SUBMISSION_READINESS.md:G5` is not measured.

**Why major**

The title and framing imply receiver relevance, while training depends on
simulation-only component bookkeeping. Without a hardware layer, reviewers
can reject the engineering significance even if simulation is strong.

**Required exact fix**

Prefer session-, device-, day-, and interference-source-held-out SDR
evaluation with no clean component at inference. If unavailable, remove every
engineering-transfer and real-time implication, keep latency descriptive, and
state that the contribution is simulation-only representation learning.

## Moderate findings

### R2-O01 — The “nine-split” table visually contains eight rows

`paper/main.tex:533-553` merges train and validation although the manifest
contains nine distinct splits and roles. Split them into separate rows and
call the table “protocol splits,” not “evaluation splits,” because training
and checkpoint selection are not evaluation domains.

### R2-O02 — The system model permits multiple simultaneous jammers, but the
headline protocol excludes mixtures

Equation (1) uses `Q` interferers (`paper/main.tex:205-220`), while
`protocol_exclusions.mixed` and
`docs/FACTOR_ISOLATED_CACHE_PROTOCOL.md` exclude compositional mixtures. State
that the general model permits `Q>=1` but the locked evidence uses one jammer
family per active view. Do not imply demonstrated multi-jammer robustness.

### R2-O03 — “Residual protection” is a hypothesis, not an implemented fact

The residual gate is bounded and always positive when enabled
(`ModelConfig.rho_min=0.05`, `rho_max=0.35`; `models/spectral.py:151-163`),
but code does not prove that it protects class evidence. Fig. 1's caption and
the contributions should call it a “bounded additive bypass/residual gate”
until A5/A7 demonstrates protection. Preserve the disclosure that branch
weights are non-conservative.

### R2-O04 — A6 is not a pure route-count ablation

The paper correctly labels A6 a composite control
(`paper/main.tex:616-617`), and
`docs/BASELINE_AND_ABLATION_AUDIT.md` explains that it changes route count,
capacity, and overlap assignment. However, the rejection rule at
`paper/main.tex:759-761` can be read as attributing any A5/A6 difference to
the tri-mask. State that A6 supports a composite comparison only; a causal
route-count claim requires a capacity/allocation-matched sensitivity.

### R2-O05 — The overlap route combines two physically different phenomena

The implementation sums unexplained power and target--jammer ambiguity into
one teacher route (`models/spectral.py:207-216`). The paper promises separate
probes, which is good, but no student head can uniquely decompose those
constituents from its single `M_o`. Phrase constituent correlations as
diagnostic associations, not attribution or identification. Correct for the
multiple mechanism tests or identify them as exploratory.

### R2-O06 — “Counterfactual TF-SIR gain” needs a strict definition boundary

Equation (24) is consistent with applying the learned spectral weight to
tracked target and jammer components. It is not output waveform SIR, a
receiver SIR estimate, or evidence of successful separation. Retain
“oracle-conditioned spectral component probe” in every table/figure caption,
and avoid using it as a standalone physical-performance claim.

### R2-O07 — Exact reproducibility parameters are missing from the manuscript

The code fixes a periodic Hann window, `center=False`, `normalized=True`,
full spectrum, scalar `lambda/rho/tau`, model widths, impairment ranges, AGC,
and normalized-quality clipping. The paper gives the functional design but
not the final `n_fft`, hop, window, widths, range values, batch/optimizer
details, or quality scales. Add a configuration table or immutable supplement
generated from the eligible run, not hand-copied prose.

### R2-O08 — Clean retention mixes seen and held channel profiles

`paper/main.tex:550` and the manifest put A/B/C/D/E in `clean_retention`.
Therefore an aggregate clean score combines interference removal with channel
generalization. Report clean retention stratified into seen A/C/D and held
B/E profiles, or explain why the aggregate is the predeclared estimand.

### R2-O09 — “Independent spectral encoders” is correct, but the diagram can
suggest stronger semantic separation than the loss supports

The two branch encoders are separately parameterized
(`models/vimd.py:45-46`), and orthogonality is applied only to active-jammer
examples (`losses.py:261-273`). State that branch names are task assignments,
not identified sources. Avoid “separates” in the abstract; use
“allocates/partitions the mixture representation.”

### R2-O10 — The final macro and table release path needs a fail-closed check

`paper/results_auto.tex` contains placeholder dashes. Merely changing
`\internalreviewfalse` would compile a visually complete but invalid paper.
The release process should fail compilation unless the result source is an
eligible run and every required macro/table cell is non-placeholder.

### R2-O11 — Citation metadata needs a final primary-source/version audit

All 24 citation keys used by `main.tex` exist in `references.bib`; there are no
undefined references. However, recent 2026 entries and the 3GPP reports need
human verification against publisher/3GPP records at submission. Add the
exact TR 38.901 release/version used by MATLAB, not only year 2024. Do not cite
an inaccessible abstract as sufficient baseline implementation evidence.

### R2-O12 — The manuscript should distinguish fixed-protocol fairness from
optimized-model fairness

Identical epochs, optimizer family, label smoothing, and checkpoint rule
control one source of variation (`paper/main.tex:584-589`), but they may
disadvantage MCLDNN/IQFormer. State explicitly that this is a unified-budget
comparison and add a preregistered tuning-sensitivity analysis before making
model-superiority claims.

## Minor and presentation findings

### R2-N01 — Fig. 2 is too dense and its provenance text is too small

On rendered PDF page 4, repeated vertical axis labels crowd adjacent panels,
and record metadata is small. A final figure should use shared axes, larger
panel labels, and a short caption; provenance belongs in the supplement or
artifact identifier.

### R2-N02 — Table III is legible only at close zoom

The nine-column ablation table on PDF page 6 uses very small text and long
purpose phrases. Shorten the purpose column or split architecture and
objective controls into two tables.

### R2-N03 — The PDF has no fatal layout error, but typography is not final

The current compile is seven pages with no overfull box, undefined citation,
or broken reference. The log contains multiple underfull boxes, notably in the
baseline paragraph and conclusion. Reflow these after scientific content is
frozen.

### R2-N04 — The last page has substantial unused space

PDF page 7 ends high on the page after balanced references. This is not a
submission violation, but the final result figures, configuration table, and
limitations should use the available space before compressing critical
tables.

### R2-N05 — Three bibliography entries are unused

`tvt_scope`, `tvt_instructions`, and `fu2022decentralized` are present in
`references.bib` but not cited. Remove them from the final source or cite them
only where scientifically relevant; journal instructions normally belong in
the submission checklist, not the scholarly bibliography.

### R2-N06 — “Unlabelled mixtures” and `mixed` need one definition

`paper/main.tex:504-505` says “unlabelled mixtures,” while the manifest calls
the excluded generator option `mixed`, meaning a composition of jammer
primitives. Use one exact term so readers do not confuse unlabeled data,
multi-jammer composition, and cochannel modulation collisions.

## Formula-to-code concordance that passed review

The following items are materially consistent and should be preserved:

- Complex STFT input and real/imag/log-magnitude context:
  `paper/main.tex:256-270` versus `models/spectral.py:16-58`.
- FiLM scaling and scalar temperature:
  `paper/main.tex:325-345` versus `models/spectral.py:119-149`.
- Scalar overlap gate and bounded residual applied to the complex spectrum:
  `paper/main.tex:348-356` versus `models/vimd.py:68-84`.
- Ideal teacher simplex plus low-power numerical guard:
  `paper/main.tex:359-401` versus `models/spectral.py:169-236`.
- Exact-source, class-aware, bidirectional contrast:
  `paper/main.tex:435-455` versus `losses.py:96-131`.
- Loss weights and staged mask/XCC ramps:
  `paper/main.tex:457-476`, `losses.py:10-20`, and
  `training.py:224-248`.
- A0--A7 mapping:
  `paper/main.tex:591-622`, `src/vimd_amc/ablation.py`, and model factories in
  `run_standard_experiment.py:327-405`.
- Residual removal in A7 preserves the residual head but applies zero in the
  forward path, so A5/A7 is parameter-count controlled.
- The paper correctly disclaims end-to-end complex neural layers, waveform
  reconstruction, strict source separation, conserved mixture-energy
  decomposition, and real-world SIR estimation.

## Minimum revision sequence

1. Resolve R2-M01 through R2-M05 before building any final evidence cache;
   otherwise the paper and executable protocol describe different tasks.
2. Freeze the comparator, sample-size, tuning, checkpoint, and multiplicity
   protocol addressing R2-M06 through R2-M11.
3. Decide the TVT scope: add trajectory/SDR evidence or narrow the vehicular
   and engineering claims as required by R2-M12/R2-M13.
4. Run the formal source-stable evidence pipeline once; do not tune from test
   outcomes.
5. Generate every result table, cache-audit value, and figure provenance from
   the eligible artifact.
6. Re-render all pages and repeat citation, claim, and layout review before
   disabling the internal evidence lock.

## Reviewer #2 decision boundary

I would reconsider after the protocol contradictions are removed and the
paper contains eligible, multi-seed, source-disjoint evidence against an
auditable recent comparator. High performance alone would not resolve the
current identifiability, scope, checkpoint, and reporting issues.
