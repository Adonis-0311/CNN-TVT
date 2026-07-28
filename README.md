# VIMD-AMC Reproducible Research Workspace

This directory is the executable research companion for the proposed VIMD-Net
paper.  It is intentionally separate from the source patent and the existing
engineering reports.

The implementation is organized around six evidence gates:

1. deterministic, source-sequence-disjoint paired data;
2. verified signal/noise/interference power control;
3. a physically aligned simulation-component target/jammer/unexplained mask
   teacher;
4. fair direct-classifier, mask, ablation, and literature-baseline comparisons;
5. held-out jammer, speed, and channel tests with single-view inference and
   source-paired statistical comparisons;
6. quantitative mechanism checks for mask agreement, target-energy transfer
   and amplification, interference leakage, and counterfactual time-frequency
   SIR gain.

The default Python generator is a controlled V2X-motivated heuristic proxy.  It
is useful for unit tests and development, but it is **not** standards-grade
vehicular evidence.  The `standards/` path uses MATLAB `nrTDLChannel` to build
offline, auditable 3GPP-aligned TDL data.  Neither source establishes complete
V2X geometry or hardware transfer; an engineering-transfer claim additionally
requires session-held-out SDR evidence.

## Quick validation

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s standards/tests -v
python experiments/run_diagnostics.py
python experiments/run_experiment.py --profile smoke --models backbone,single_mask,vimd --seeds 17 --device cpu
```

The complete controlled ablation ladder and both strong local baselines can be
smoke-tested with:

```powershell
python experiments/run_experiment.py --profile smoke --models a0_backbone,a1_single_mask,a2_tri_no_teacher,a3_tri_teacher,a4_tri_teacher_mtl,a5_vimd_full,a6_dual_full,a7_vimd_no_residual,mcldnn_reimplementation,iqformer_inspired --seeds 17 --device cpu
```

The 2025 CSSL-AMC comparator is intentionally absent from this 256-sample
smoke command. Its audited official architecture has a fixed 1,024-sample
flattening contract and is registered as
`cssl_amc_supervised_adaptation` only in compatible 1,024-sample runs,
including the frozen formal suite.

## Development experiment

```powershell
python experiments/run_experiment.py --profile dev --models backbone,single_mask,vimd --seeds 17,29,43
```

Outputs are written under `artifacts/` as machine-readable JSON, CSV, NPZ, and
model checkpoints.  Figures and paper tables must be generated from those
artifacts; manually edited result tables are not accepted.

## Implemented scientific mechanism

The current code deliberately follows the *auditable implementation*, which is
more precise than the older conceptual wording in the original idea document:

- a deterministic full-spectrum complex STFT with no centering;
- phase-preserving real/imaginary/log-magnitude spectral features;
- an environment encoder that conditions an overlap-aware tri-mask;
- target-power-dominant, jammer-power-dominant, and
  unexplained/power-ambiguous masks on the exact physical STFT lattice;
- a fixed, training-only component-power teacher constructed from independently
  tracked target, jammer, and unexplained components;
- modulation and jammer spectral branches, jammer multi-label supervision, and
  SNR/SIR/Doppler-quality regression;
- exact-source paired cross-condition InfoNCE, with other same-class sources
  ignored rather than mislabeled as positives or negatives.

The current implementation is **not** an EMA latent teacher, a time-domain
waveform separator, or a reconstructive denoiser.  The real masks allocate
complex spectral coefficients to task-specific representations; the bounded
residual path intentionally makes the two branch weights non-conservative.

## Evidence policy

Smoke profiles and the fixed-batch overfit diagnostic prove that the pipeline,
gradients, objectives, and artifact writer execute.  They are not paper
performance evidence.  A result can enter the manuscript only when its
manifest, configuration digest, source-disjoint split, model checkpoint,
predictions, statistical comparison, and environment record are present.
Execution completion is distinct from claim eligibility: smoke and screening
artifacts remain ineligible even when their job status is `complete`.

The audited `standards/cache_screening_v1` cache contains 512/128/256 sources,
all ten declared modulation classes, six active jammer families, four speeds,
and TDL-A/C/D-to-TDL-B/E profile holdout.  Its immutable digest is
`e219930800a24844146087b6dfa7b2fa2daf1c61aaaf2ab5b0158c4c79a80b9a`;
its designation is `screening_not_formal_tvt_evidence`.

## Baseline and ablation claim boundaries

The registry exposes A0--A7 as the paper's fixed evidence ladder.  A2--A5 and
A7 share the same tri-mask parameterization; A6 is a composite dual-mask,
dual-task-branch control that also merges overlap mass into the non-target
teacher route.

| ID | Routes | Teacher | MTL bundle | XCC | Residual |
|---|---:|---:|---:|---:|---:|
| A0 | 0 | no | no | no | no |
| A1 | 1 | no | no | no | no |
| A2 | 3 | no | no | no | yes |
| A3 | 3 | fixed component-power | no | no | yes |
| A4 | 3 | fixed component-power | yes | no | yes |
| A5 | 3 | fixed component-power | yes | yes | yes |
| A6 | 2 | collapsed component-power | yes | yes | yes |
| A7 | 3 | fixed component-power | yes | yes | no |

The MTL bundle comprises jammer identification, quality regression, and branch
orthogonality.  XCC is paired same-source cross-condition contrastive learning.
For A6, the dual teacher retains the modulation-dominant route and collapses
the tri-teacher's jammer-dominant plus overlap/unexplained mass into the second
route.  A5 versus A7 isolates only whether the bounded residual is applied;
their parameterization remains identical.
The fixed-protocol runner shares data realizations, model seeds, optimizer,
schedule, and checkpoint rule.  Architectures are **not parameter-matched**;
parameter counts, convolution/linear/recurrent MACs, STFT operation estimates,
and latency are reported per model.  Final literature comparisons still need
an architecture-specific tuning sensitivity check.

`mcldnn_reimplementation` is a literature-faithful PyTorch port audited
against the [authors' public MCLDNN code](https://github.com/wzjialang/MCLDNN)
at commit `f1093eea5a04ba6f7fc5297171ffbae5c9853f93` and the
[IEEE WCL paper](https://doi.org/10.1109/LWC.2020.2999453).  It is not
checkpoint-compatible with the Keras implementation and must be labeled a
reimplementation.

`iqformer_inspired` follows the public RML2016 stage widths/depths and IQ/STFT
fusion pattern from the
[official IQFormer repository](https://github.com/WestdoorSad/IQFormer) at
commit `7ee6ac949551b24d45f218762cab919e0cb6b4f9` and the
[IEEE TCCN paper](https://doi.org/10.1109/TCCN.2024.3485118).  Its internal
PyTorch STFT and dependency-free blocks prevent an exact-reproduction claim;
all outputs must retain the `IQFormer-inspired` label.

`cssl_amc_supervised_adaptation` retains the encoder/classifier graph from the
authors' 2025 Apache-2.0 code at commit
`2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f`. It is randomly initialized and
trained under the common supervised budget; it does not load official weights
or execute the published momentum-encoder pretraining and fine-tuning
protocol. Its required label is therefore **CSSL-AMC official-architecture
supervised adaptation**, not a complete CSSL reproduction, official result, or
structured-interference-specific method.

The item-by-item architecture checks and admissible claim language are fixed
in [`docs/BASELINE_AND_ABLATION_AUDIT.md`](docs/BASELINE_AND_ABLATION_AUDIT.md).
