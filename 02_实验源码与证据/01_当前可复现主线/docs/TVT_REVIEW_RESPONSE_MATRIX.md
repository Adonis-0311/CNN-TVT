# TVT review-response evidence matrix

Status date: 2026-07-29
Authority: `docs/TVT_EVIDENCE_CLOSURE_DELIVERY_2026-07-29.md`

This matrix separates prose/code facts from measurements and external actions.
It controls the evidence boundary of the simulation-only initial manuscript;
it does not cover authorship, journal policy, upload, or any other submission
operation.

Status vocabulary:

- **Statically closed**: source, manuscript, mathematics, or primary-reference
  evidence is present and can be checked without a new scientific run.
- **Pending formal run**: the design or reporting contract is specified, but
  no eligible result exists. Screening, smoke, proxy, invalidated, or one-seed
  diagnostic numbers cannot close the item.
- **Outside current claim scope**: the manuscript makes the narrower statement
  shown in the response column, so the omitted experiment is neither evidence
  for that statement nor a blocker for the current initial draft. Adding the
  broader claim requires a new prospective protocol.
- **Human/external action required**: closure requires author contact, policy
  confirmation, human authorship/citation review, or another action that the
  repository cannot truthfully claim has occurred. Such work is listed here
  only when it would expand the current claim scope; submission administration
  is tracked separately.

| Review item | Manuscript/repository response | Evidence anchor | Status and release condition |
|---|---|---|---|
| M1 teacher form and low-SNR audit | The teacher is now written directly as \(M_s^\star=[q_s-q_j]_+\), \(M_j^\star=[q_j-q_s]_+\), \(M_o^\star=q_u+2\min(q_s,q_j)\). The simplex/scale observation is a paragraph, not a proposition. A deterministic all-cell audit reports mean/P90/nonzero fraction by jammer family and at the exact \(-10\)-dB SNR/\(-15\)-dB SIR cell. On the 500-source/1,000-view screening hard split, that cell has 41 views and target-route mean 0.068658, P90 0.258110, and nonzero fraction 0.565611. | `paper/main.tex`; `experiments/audit_hard_split_teacher.py`; `docs/TEACHER_FUNCTION_EVIDENCE.md`; `tests/test_teacher_form_and_audit.py` | Algebra, code, and the explicitly development-only screening diagnosis are **statically closed**. The screening artifact is bound to its digest and has `formal_tvt_evidence_eligible=false`; formal-cache distributions, A3/A3′ performance, and family-stratified gains remain **pending formal run**. |
| M2 phase claim | Abstract/title-level “phase-preserving” contribution language is removed. The manuscript states only that a real mask retains mixture phase for nonzero coefficients and estimates no phase correction. PSM and cIRM are cited. | Erdogan et al., ICASSP 2015, DOI `10.1109/ICASSP.2015.7178061`; Williamson et al., TASLP 2016, DOI `10.1109/TASLP.2015.2512042` | **Statically closed**. |
| M2 masking lineage | IBM/IRM training targets and deep clustering are acknowledged; the claimed distinction is the AMC task and a third route containing \(q_u+2\min(q_s,q_j)\), not invention of masking. | Narayanan--Wang 2013; Wang--Narayanan--Wang 2014; Hershey et al. 2016 | **Statically closed**. |
| M3 teacher-function ablation | Required row is A3′, runner `a3p_tri_proportional_teacher` (alias `a3_ratio_teacher`), with target \((q_s,q_j,q_u)\). A3′ holds architecture, objective weight, data, and optimizer fixed relative to A3. Confirmatory comparison is margin A3 minus proportional A3′. | `src/vimd_amc/ablation.py`; standard and proxy runners; v2 freeze; teacher tests; manuscript table | Implementation and registry integration are **statically closed**. A3′ performance and the simultaneous interval are **pending formal run**; no direction is asserted. |
| M4 training scale | Future formal execution is fixed at 100,000 training sources. Before it, A0/A5 must complete a 10k/30k/100k, five-seed learning curve with per-epoch training/validation macro-F1 and loss. The 100k scale is retained regardless of the curve, so the curve cannot select scale or hyperparameters post hoc. The prospective statistical justification states the target effect, uncertainty unit, multiplicity treatment, simulation assumptions, and interpretation limits before any formal result is opened. | `docs/TVT_V2_PROSPECTIVE_STATISTICAL_JUSTIFICATION.md`; `tvt_submission/configs/formal_tvt_freeze_v2.json`; `experiments/run_learning_curve_v2.py`; `src/vimd_amc/training.py`; v2 contract tests | Scale/statistical rationale, protocol, and executable fail-closed prerequisite are **statically closed**. Curves, precision, and saturation evidence are **pending formal run**. |
| M5 seed count and confirmatory family | The prospective grid uses 12 models × 10 fixed seeds = 120 fits. The confirmatory family is exactly A5−A0, A3−A3′, and A5−A6 under one joint hierarchical paired simultaneous interval; all other direct contrasts are exploratory. | v2 freeze/validator; `experiments/run_formal_tvt_v2.py`; `tvt_submission/validate_v2_release.py` | Design, model/seed grid checks, checkpoint/no-fallback checks, and negative fail-closed tests are **statically closed**. All estimates and family-wise intervals are **pending formal run**. |
| M6 OOD opportunity calibration | Each unseen-jammer, unseen-speed, and held-out-channel axis reports A0 ID-to-axis degradation with CI. If its lower bound exceeds 0.5 pp, A5 must recover 25% of the positive degradation with a positive calibrated-margin lower bound; otherwise A5 must meet a −1 pp noninferiority lower bound. Every axis must pass its predeclared branch; no fixed 3-pp rule remains. | `src/vimd_amc/metrics.py::ood_axis_calibrated_bootstrap`; v2 freeze; release validator; synthetic contract tests | Formula, resampling hierarchy, axis branches, and tests are **statically closed**. Axis classifications and outcomes are **pending formal run**. |
| M7 ADC/AGC and independent emitters | Three source-disjoint, hard-interference test-only splits add post-AGC 10-bit ADC, post-AGC 12-bit ADC, and independent target/jammer CFO/timing before mixing. Clipping fraction and exact quantization residual are retained; component identity remains auditable. Receiver axes use nominal hard-interference—not ID—as the A0 reference, preventing SIR-distribution confounding. | `src/vimd_amc/data/synthesis.py`; v2 cache policies; Table III; v2 contract and component-identity tests | Generator, manifests, split policy, nominal-reference contract, and audit tests are **statically closed**. Robustness estimates and CIs are **pending formal run**. |
| Clean-retention release gate | A5 is compared with the preregistered CSSL-AMC reference on source-paired clean predictions, separately for seen-profile A/C/D and held-profile B/E. Each stratum uses hierarchical algorithm-seed/class-stratified-source bootstrap; point gain must be at least -1 pp and the 95% lower bound at least -2 pp. Both strata must pass. | v2 freeze strict loader; `tvt_submission/validate_v2_release.py::_clean_retention_tests`; synthetic gate test | Freeze semantics, derivation, and fail-closed thresholds are **statically closed**. Estimates and pass/fail outcomes are **pending formal run**. |
| Scientific gate versus paper backfill | `v2_scientific_release_gate.json` distinguishes scientific evidence from manuscript promotion. Even when every scientific gate passes, it records `submission_unlocked=false`; the separate hash-bound v2 adapter must rederive every macro before the initial manuscript can be populated. | `tvt_submission/validate_v2_release.py`; `tvt_submission/validate_v2_paper_release.py`; v2 protocol; local queue; v2 paper-release tests | Scientific/paper separation and the canonical v2-to-paper adapter are **statically closed and implemented**. After the formal run, macro derivation, locked backfill, and internal rebuild are deterministic pending actions; no placeholder or manually typed value may bridge them. |
| M8 MFENet | MFENet remains cited as related work, not an executed row. A bounded audit found no author-controlled public code, checkpoint, or complete executable specification; the manuscript does not convert that negative search into a universal claim and makes no superiority claim over MFENet. | `docs/RECENT_INTERFERENCE_BASELINE_AUDIT.md`; IEEE DOI `10.1109/TVT.2025.3555769`; `paper/main.tex` | The current simulation-only claim boundary is **statically closed**. MFENet contact/reproduction is **outside current claim scope** and becomes required only if a future draft adds a numerical MFENet row or superiority claim. |
| M8 IQFormer provenance | The local row remains “IQFormer-inspired.” The official MIT-licensed repository and audited commit are cited, and absence of official-checkpoint/native-protocol equivalence is explicit. | Official repository commit `7ee6ac949551b24d45f218762cab919e0cb6b4f9`; `docs/BASELINE_AND_ABLATION_AUDIT.md` | Local provenance and bounded label are **statically closed**. Native RadioML reproduction is **outside current claim scope**; the formal run measures only the frozen local adaptation under the common simulation protocol. |
| M8 CSSL provenance | The row remains “CSSL-AMC official-architecture supervised adaptation.” Random initialization and replacement of the published two-stage protocol by common-budget supervised training are explicit. | Official Apache-2.0 commit `2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f`; local source lock | Local provenance and bounded label are **statically closed**. Native two-stage reproduction is **outside current claim scope**; the formal run measures only the frozen supervised adaptation. |
| M8 public-data anchor | The current paper is explicitly limited to the frozen source-disjoint TDL-profile simulation protocol and makes no public-dataset transfer or native-protocol reproduction claim. A future public-data supplement would require authorized RADIOML acquisition, immutable provenance, native IQFormer/CSSL checks, and a separately frozen synthetic-interference overlay; the DeepSig historical/known-errata warning must be retained. | DeepSig official dataset page; O’Shea et al., JSTSP 2018, DOI `10.1109/JSTSP.2018.2797022`; `paper/main.tex` scope paragraph | **Outside current claim scope and non-blocking for the initial draft.** No RadioML data exist locally and no RadioML number is claimed. The 2026-07-28 failed download attempt remains provenance only, not a missing input to the v2 queue. |
| Minor 1 waveform parameters | Table III now states 4 sps / 250 ksym/s, fixed RRC roll-off 0.35 and 8-symbol span, randomized integer timing phase, the exact Gaussian MSK kernel (nominal \(BT\approx0.24\)), CPFSK \(h=0.5\), and 4FSK offsets. Fixed versus randomized quantities are separated. | `src/vimd_amc/data/synthesis.py`; `paper/main.tex` | **Statically closed** for the current generator. Human authors must decide whether these fixed settings are sufficiently diverse for the final study. |
| Minor 2 STFT window | Table III now states periodic Hann, FFT/hop 64/16, `center=false`, unitary normalization, full two-sided spectrum, and 61 frames for \(L=1024\). | `src/vimd_amc/models/spectral.py`; `paper/main.tex` | **Statically closed**. |
| Minor 3 Fig. 2 mismatch | Caption now identifies the plot as a micro-smoke proxy visualization with \(L=64\), STFT 32/8, and five frames, distinct from the 61-frame formal lattice. | `paper/make_teacher_figure.py`; micro-cache manifest; figure caption | **Statically closed** as disclosure. Replacing it with a formal-cache rendering remains optional until the formal cache exists. |
| Minor 4 scalar gates | The manuscript gives the auditability rationale for per-window \(\lambda,\rho,\tau_c\) and states that scalar \(\lambda\) cannot adapt within a sweep/pulse. | `paper/main.tex` | Rationale and limitation are **statically closed**. Per-cell \(\lambda[f,t]\) sensitivity is **outside current claim scope** and would require a new freeze. |
| Minor 5 \(\Delta_{\rm spec}\) bound | From \(\rho\le W_m\le1+\rho\), the manuscript gives \(\Delta_{\rm spec}\le20\log_{10}((1+\rho)/\rho)\le26.4\) dB for \(\rho_{\min}=0.05\). | `paper/main.tex` | **Statically closed**. |
| Minor 6 JS rationale | The manuscript states symmetry, zero-support finiteness, and the natural-log \(\log2\) bound, while disclosing possible weak gradients far from the teacher. | Lin, IEEE TIT 1991, DOI `10.1109/18.61115`; `docs/AUXILIARY_AND_MECHANISM_EVIDENCE.md` | **Statically closed** for rationale; loss/teacher-agreement curves are **pending formal run**. |
| Minor 7 cochannel bound | The \(1/2\) error bound is limited to the balanced exchangeable class with identical observations; it is not asserted for unequal power/channel statistics. A dominant-emitter split with at least 6 dB received-power margin is identified as separate and unexecuted. | `docs/COCHANNEL_IDENTIFIABILITY_BOUNDARY.md`; `paper/main.tex` | Boundary and exclusion are **statically closed**. A dominant-emitter experiment is **outside current claim scope** and would require a new freeze. |
| Minor 8 \(L_\perp\) attribution | A4--A3 remains explicitly a jammer/quality/orthogonality bundle and cannot attribute an effect to \(L_\perp\). | A0--A7 table/prose; baseline audit | Bundle disclosure is **statically closed**. A4a/A4b is **outside current claim scope**; it is required only before making a single-factor \(L_\perp\) attribution. |
| Minor 9 contrastive confound | The manuscript states that every trainable comparator receives the same paired-view batches; A5--A4 changes the XCC loss rather than access to a second view. | comparator protocol and runner audit | Batch-contract disclosure is **statically closed**; marginal gain is **pending formal run**. |
| Minor 10 seed reuse | The v2 preregistration retains generator master seed `20260727` but sets bootstrap seed `20260803`; the strict loader rejects equality, and Table III matches it. No scientific result is asserted because the formal run has not started. | Table III; v2 freeze/validator; `tests/test_formal_v2_protocol.py` | The prospective seed contract and enforcement are **statically closed**; artifact verification is **pending formal run**. |
| Minor 11 references | The weak Sensors cross-SNR anchor is removed from the manuscript/bibliography in favor of the IEEE TWC contrastive learner. 2026 metadata were checked: DenoMAE2.0 vol. 74, pp. 929--943; CGPCDA vol. 12, pp. 6929--6941; the malicious-interference TVT paper remains Early Access with no fabricated volume/issue. | Crossref DOI records; IEEE/author institutional records; `paper/references.bib` | Metadata is **statically closed as of 2026-07-28**. Human authors must recheck at submission because Early Access metadata can change. |
| Minor 12 scope rhetoric and initial-draft structure | Repeated disclaimers are consolidated under one positive Scope paragraph; introduction/conclusion state the supported simulation claim directly. The current file remains an internal evidence-lock build because results are unavailable. The artifact-derived result macros and evidence-conditioned conclusion are already wired for deterministic backfill. | `paper/main.tex`; paper build gate; v2 paper-release validator | Method/protocol narrative and backfill structure are **statically closed**. Populated results and the corresponding evidence-conditioned conclusion are **pending formal run and deterministic backfill**. Submission-only editing, metadata, and policy review are outside this matrix. |

## MFENet author-contact log template

This template is intentionally unsent and is not a current initial-draft
requirement. Use it only if the claim scope is prospectively expanded to an
MFENet numerical comparison. Filling it does not itself establish a
reproduction.

```text
Contact status: NOT SENT
Corresponding author/address source:
Date/time sent (UTC):
Sender (human author):
Subject: Reproduction materials for MFENet (TVT 2025, DOI 10.1109/TVT.2025.3555769)

Requested:
1. exact source revision/archive and license;
2. RML2016.10a/10b preprocessing, split seeds, and class/SNR protocol;
3. complete layer/configuration and parameter/FLOP script;
4. optimizer, scheduler, regularization, seeds, and checkpoint rule;
5. official weights plus one reference input/logit batch; and
6. exact interference generation and SIR definition.

Reply date:
Materials received and immutable hashes:
License/admission decision:
Native-result reproduction command/artifact:
Adaptation differences:
```

## Primary external anchors

- PSM: https://www.merl.com/publications/TR2015-031
- cIRM: https://pmc.ncbi.nlm.nih.gov/articles/PMC4826046/
- Deep clustering: https://www.merl.com/publications/TR2016-003
- IQFormer official code:
  https://github.com/WestdoorSad/IQFormer/commit/7ee6ac949551b24d45f218762cab919e0cb6b4f9
- CSSL-AMC official code:
  https://github.com/dumingyang20/CSSL-AMC-Pytorch/commit/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f
- DeepSig historical datasets and errata notice:
  https://www.deepsig.ai/datasets/
- RADIOML generation code: https://github.com/radioML/dataset
