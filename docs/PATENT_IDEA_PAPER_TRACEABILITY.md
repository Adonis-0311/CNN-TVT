# Patent--Idea--TVT manuscript traceability

Status date: 2026-07-28  
Purpose: prove design lineage while separating implemented method statements
from unmeasured scientific claims.

## Source lock

| Source | Role | SHA-256 |
|---|---|---|
| `DFI257727-基于神经网络的干扰环境下信号调制识别方法及系统(定稿).docx` | Root patent specification | `135403A345037FA8B71E8ED4AA094858EB37602ACC64A3F83BE8A672BE05141D` |
| `TVT_Flagship_VIMD_Net_AMC_Full_Design_Idea.md` | Scientific design directive | `E1DACF2B55C310E3D34AD4ECA27B77D8C570604B769AA35A989C86CD08DB5194` |

Both files are read-only inputs. They are not copied into the paper source
archive. The patent was structurally extracted from OOXML for this audit;
paragraph indices below refer to that extraction and are not legal claim
constructions.

## Technical lineage

| Patent root | Idea upgrade | Current paper/implementation | Evidence state |
|---|---|---|---|
| Claim 1 and S110--S115: construct an interference-adapted feature by associating interference and signal characteristics | Replace a discrete interference-template database with a continuous environment encoder and condition embedding | Pooled environment encoder conditions the exact complex-STFT tri-mask; jammer and quality heads provide training-only structural supervision | Implemented and tested; effectiveness remains unmeasured formally |
| Claims 3--4 and S120--S125: decompose features, estimate interference contribution, retain signal-labelled components, and remove interference-labelled components | Replace hard binary labelling/deletion with target-power-dominant, jammer-power-dominant, and unexplained-or-power-ambiguous soft allocations; add overlap gates and a bounded raw bypass | `VIMDNet` applies three simplex masks on the physical STFT lattice. The target-task route is `(M_s + lambda M_o + rho) Z`; the jammer route is `(M_j + (1-lambda) M_o) Z` | Conservation, phase, gradient, and bypass controls are tested; no waveform-separation claim is permitted |
| Patent interference-contribution template and threshold | Use clean simulation bookkeeping to construct a training-only latent/physical teacher rather than an inference-time rule | Deterministic component-power teacher uses independently tracked target, jammer, and unexplained post-channel components on the identical STFT lattice | Teacher algebra is tested; teacher-to-student benefit is not established |
| Claims 5--6 and S130--S135: map cleaned features into a modulation space and match a modulation representation | Replace nearest-prototype/rule-forced classification with a discriminative learned modulation head | Modulation branch predicts the observed class directly from one received mixture; A0--A7 isolate mask, teacher, MTL, contrast, and bypass effects | Implemented; formal five-seed comparison pending |
| Claims 7--8 and S140--S145: use interference type and a rule library to revise the preliminary modulation result | Keep recognition and link recommendation separate; use jammer information only as an auxiliary learning signal | Jammer logits never override the modulation label. Training-support masks prevent held-out jammer columns from being interpreted as trained recognition | Implemented and release-gated; deliberate scope change from the patent |
| Claim 9 and S150--S155: generate a signal report and transmit it to an analysis terminal | Defer reporting, rule fusion, and downstream recommendation until after scientific validation | Paper outputs only modulation probabilities and diagnostic auxiliary quantities; no report-delivery system is claimed | Outside the present paper |
| Claim 10: processor/storage system executing the method | Provide a reproducible local software pipeline before engineering conversion | Python/MATLAB cache builder, runner, tests, evidence gates, and dry-run PowerShell hand-off are present | Software structure tested; deployment target and SDR transfer remain open |

## Idea directives retained

1. **Three-way overlap-aware allocation.** The paper retains the core
   target/jammer/uncertain simplex rather than reducing the method to a single
   attention mask.
2. **Training-only physical supervision.** Component bookkeeping is forbidden
   at inference, and checkpoints are audited to contain no teacher.
3. **Dual task semantics.** Modulation and jammer/quality objectives constrain
   different routes; unsupported jammer labels are masked.
4. **Exact-source cross-condition learning.** Two independently impaired views
   share one immutable source identity. Headline inference still consumes one
   view.
5. **Vehicular factor isolation.** Speed/Doppler, TDL profile, jammer family,
   and their joint shift are separated into source-disjoint roles.
6. **Falsifiable mechanism evidence.** A0--A7, route occupancy, teacher
   agreement, transfer/leakage ratios, counterfactual spectral ratio, and
   clean-retention guards are preregistered.
7. **No hand-entered paper results.** Formal artifacts, statistical outputs,
   macro derivation, and the paper release lock are machine-bound.

## Scientifically necessary amendments to the Idea

These are controlled corrections, not silent drift:

- The title and claims are narrowed to 3GPP TR 38.901 TDL-profile simulations
  parameterized by vehicular Doppler. A complete geometry-driven V2X
  trajectory and SDR transfer are not claimed because those evidence layers do
  not exist.
- Unanchored in-taxonomy modulated cochannel collisions are excluded from the
  single-label headline task. Source exchange leaves the observation
  distribution unchanged while changing the requested label, yielding a
  50-percent Bayes-error lower bound in the balanced symmetric case.
- Multi-jammer compositions are excluded until a separate frozen
  compositional protocol exists.
- The teacher is a deterministic component-power allocation on the physical
  STFT, not an EMA feature encoder. This makes its meaning auditable and avoids
  calling learned attention a physical source estimate.
- Cross-condition positives are the two views of the exact same source.
  Other same-class sources are omitted from the denominator rather than treated
  as interchangeable positives or false negatives.
- “Signal” and “interference” route names are replaced by
  target-power-dominant, jammer-power-dominant, and
  unexplained-or-power-ambiguous. The method allocates representations; it
  neither identifies physical emitters nor reconstructs separated waveforms.
- The formal primary effect is hard-region macro-F1 under a hierarchical
  seed/source paired bootstrap. Per-seed McNemar tests are supplemental, not
  pooled across seeds.

## Evidence chain and stop/go state

| Layer | Authoritative artifact | Current decision |
|---|---|---|
| Method structure | `src/vimd_amc/`, A0--A7 registry, structural tests | Pass for implementation claims |
| Data pipeline | `standards/cache_factor_screening_1024_v1.audit.json`, digest `241b3aec...49e3` | Pass for screening stability only |
| Identifiability boundary | `docs/COCHANNEL_IDENTIFIABILITY_BOUNDARY.md` and counterexample tests | Pass; primary protocol excludes unanchored cochannel |
| Post-v3 candidate | `docs/VIMD_V4_DSBN_PREREGISTRATION.md` | Unexecuted diagnostic; no performance claim |
| Formal experiment | `tvt_submission/configs/formal_tvt_freeze_v1.json` | Preregistered; cache and five-seed run absent |
| Paper-number release | `tvt_submission/generate_macro_values.py` and `validate_release.py` | Fail-closed; `paper/release_lock.json` absent |
| Manuscript | `paper/main.tex`, `paper/results_auto.tex` | Internal evidence-lock build; quantitative cells pending |

The lineage is therefore established, but scientific effectiveness is not.
The patent and Idea justify the hypothesis and experimental design; they do
not substitute for the missing formal result bundle.
