# Recent Interference-Baseline Reproducibility Audit

**Audit date:** 2026-07-27  
**Scope:** MFENet first, RepCCNet second  
**Decision for MFENet/RepCCNet:** do not add either model to the executable
registry yet. A 2026-07-28 addendum below separately admits a bounded CSSL-AMC
architecture adaptation; it does not change either decision.

This record separates discoverable bibliographic facts from evidence that is
strong enough to implement and label a literature baseline. It is a
reproducibility gate, not a performance claim.

## Admission rule

A model may be registered under the published method name only if an
author-controlled implementation or a complete primary-source specification
resolves, at minimum:

- every layer, width, kernel, padding, activation, normalization, temporal
  readout, and train/test graph difference;
- preprocessing and normalization;
- all loss, optimizer, schedule, regularization, initialization, data-split,
  checkpoint-selection, and random-seed details; and
- a testable parameter/MAC target or an official checkpoint.

An abstract, a block-name list, or a diagram with unresolved execution details
does not meet this rule. No architecture was reconstructed from a secondary
summary.

## Source ledger

| Item | Primary or author-controlled source | What it establishes |
|---|---|---|
| MFENet metadata | [Crossref record](https://api.crossref.org/works/10.1109/TVT.2025.3555769) | DOI, title, venue, volume, issue, pages, publication date, author affiliations |
| MFENet publisher page | [IEEE Xplore document 11037461](https://ieeexplore.ieee.org/document/11037461/) | Canonical publisher landing page; full text was not accessible in this environment |
| MFENet public-code search | [GitHub exact-title search](https://github.com/search?q=%22Toward+Interference-Tolerant+Automatic+Modulation+Recognition%22&type=repositories), [GitHub DOI search](https://github.com/search?q=%223555769%22&type=repositories) | No matching public repository was located on the audit date; this is a bounded negative result, not proof that private or renamed code does not exist |
| RepCCNet metadata | [Crossref record](https://api.crossref.org/works/10.1109/TVT.2024.3361928) | DOI, title, venue, volume, issue, pages, publication date, authors and affiliations |
| RepCCNet publisher page | [IEEE Xplore document 10420493](https://ieeexplore.ieee.org/document/10420493/) | Canonical publisher landing page; full text was not accessible in this environment |
| RepCCNet author identity/code surface | [OUC laboratory people page](https://ouc.ai/zhenghaiyong/research/people.html), [Ning Tang's linked GitHub account](https://github.com/tangning411) | The OUC page links Ning Tang to the GitHub account; its seven owner repositories contain no RepCCNet implementation |
| RepCCNet related primary disclosure | [CN117354106B full patent record](https://patents.google.com/patent/CN117354106B/zh), [CNIPA-issued patent PDF](https://patentimages.storage.googleapis.com/3f/a4/52/df5ad46fad8c74/CN117354106B.pdf) | Same five-person author/inventor team, model diagram, layer sizes, reparameterization rule, training protocol, reported deployment size |
| RepCCNet faculty disclosure | [OUC Yaohui Lyu profile](https://it.ouc.edu.cn/lyh/main.htm) | Lists the RepCCNet communication-signal modulation-recognition patent |
| RepCCNet public-code search | [GitHub exact-title search](https://github.com/search?q=%22Reparameterization+Causal+Convolutional+Network%22&type=repositories), [GitHub method-name search](https://github.com/search?q=RepCCNet&type=repositories) | No matching public repository was located on the audit date |

Crossref and GitHub API queries were also run for the exact titles, DOI suffixes,
method names, and author names. IEEE's automated HTML/PDF endpoints returned an
HTTP 418 verification page. Discovery indexes were used only to look for an
original copy; they were not accepted as architecture evidence.

## MFENet

### Verified

- **Title:** *Toward Interference-Tolerant Automatic Modulation Recognition via
  Multi-Stage Feature Extraction Network*
- **DOI:** `10.1109/TVT.2025.3555769`
- **Venue:** IEEE Transactions on Vehicular Technology, vol. 74, no. 8,
  pp. 12629-12640, Aug. 2025.
- The Crossref record points only to the IEEE publication and exposes no
  author-repository or supplementary-code relation.

### Not reproducible from admitted evidence

No author-controlled repository, checkpoint, archived supplement, or publicly
accessible author manuscript was found. Consequently, none of the following
can be fixed without guessing:

- the complete MSP/MFE/classifier execution graph;
- FFT convention, retained bins, complex-to-real representation and scaling;
- filter counts, kernel sizes, strides, widths and stage repetitions;
- the exact sandglass and Res2-attention definitions;
- input length, dataset normalization and augmentation;
- optimizer, schedule, regularization, seeds and checkpoint selection; and
- parameter/MAC target and released weights.

**MFENet decision:** no local implementation. The paper may remain a related-work
citation, but it must not appear as an executed baseline until the missing
primary evidence is obtained.

## RepCCNet

### Verified topology from the same-team patent

CN117354106B has the same five people as the IEEE paper, in a different order,
and discloses the following:

1. raw I/Q input;
2. a two-dimensional RepCasual front end that treats I/Q as the spatial
   dimension and uses a `(2, K)` valid kernel across that dimension;
3. three parallel causal-convolution branches during training, with branch
   weights summed into one kernel for testing;
4. input dropout on each training branch, removed after reparameterization;
5. an initial `K=17`, `C_in=2`, `C_out=128` operation;
6. squeeze/excitation attention using global average pooling and `1x1`
   projections `128 -> 32 -> 128`, followed by a sigmoid gate;
7. a `K=1`, `128 -> 32` bottleneck convolution, batch normalization and ReLU;
8. two `K=9`, `32 -> 32` one-dimensional RepCasual blocks with ReLU; and
9. a final fully connected classifier.

The patent also fixes a RadioML 2016.10a training example:

- PyTorch 1.12.1;
- random 60%/20%/20% train/validation/test split;
- SNR range `-20` to `18` dB;
- 200 epochs, batch size 400;
- AdamW, weight decay `1e-4`;
- initial learning rate `0.01`, halved after eight epochs without validation
  loss improvement;
- cross-entropy, end-to-end training from scratch; and
- NVIDIA GTX 3080.

It reports 35.9K parameters. The figure's fused deployment graph reconciles to
35,883 trainable parameters for 11 classes if convolution biases are retained:

```text
K17  2->128                 4,480
SE   128->32->128           8,352
K1   128->32                4,128
BN   32                         64
2 x (K9 32->32)            18,496
FC   32->11                   363
                            ------
                            35,883
```

This arithmetic is a consistency check, not proof of every runtime choice.

### Remaining reproduction blockers

The primary sources inspected do not specify:

- dropout probability and whether it is shared or independently sampled across
  the three branches;
- the temporal readout before the `32 -> classes` head (last causal position,
  mean, max, or another reduction);
- bias use and any normalization inside RepCasual branches;
- initialization and exact train-to-deploy fusion procedure;
- input normalization and sample-level preprocessing;
- data-split seed and whether the random split is stratified by modulation/SNR
  or grouped by original source;
- validation checkpoint rule, AdamW beta/epsilon values, scheduler threshold,
  minimum learning rate and early stopping;
- the complete RadioML 2018.01a protocol used by the IEEE article; and
- official source, checkpoint, license, or reference logits.

The patent evidence concerns RadioML noise/channel impairments and SNR sweeps.
It does not establish robustness to the structured tone, chirp, pulse,
partial-band, cochannel, or multitone interferers in the VIMD protocol.

**RepCCNet decision:** no executable model under the `RepCCNet` name. A local
implementation made now would require at least two material architecture
choices and several training guesses. Calling it official or
literature-faithful would therefore be misleading.

## 2026-07-28 addendum: CSSL-AMC architecture adaptation

The 2025 paper *A Contrastive Learner for Automatic Modulation
Classification* has author-controlled Apache-2.0 code. The immutable local
source lock is `tvt_submission/sources/cssl_amc_2025.lock.json`, pinned to
commit `2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f` with byte hashes for
`models/cssl.py`, `run.py`, `contrastive_learning.py`, `finetune.py`, and
`dataset.py`.

The local `cssl_amc_supervised_adaptation` retains the official 1,024-sample
encoder/classifier topology: a three-convolution noise estimator, concatenated
raw I/Q and estimated noise, `[2, 2]` residual stages, a fixed
`128 x 512 -> 128` readout, and a `128 -> 64 -> classes` head. Tests bind the
layer shapes, 8,631,948-parameter ten-class count, input contract, source
commit, license hash, and registration in the formal model family.

Material protocol changes remain explicit: the ten-class TVT taxonomy replaces
the native 24-class head, no external checkpoint is loaded, and the common
paired-view supervised objective replaces the published momentum-encoder
pretraining plus fine-tuning. The only admissible label is **CSSL-AMC
official-architecture supervised adaptation**. It is a recent auditable AMC
comparator, not a complete CSSL reproduction, not an official CSSL result, and
not a structured-interference-specific method.

## Adequacy of the current executable baselines

The current executable set is useful for controlled comparison:

- `mcldnn_reimplementation` is a literature-faithful PyTorch port audited
  against the authors' public Keras code at commit
  `f1093eea5a04ba6f7fc5297171ffbae5c9853f93`;
- `iqformer_inspired` is explicitly labeled as an inspired local baseline and
  audited against public code at commit
  `7ee6ac949551b24d45f218762cab919e0cb6b4f9`; and
- `cssl_amc_supervised_adaptation` is the bounded recent comparator described
  above and is restricted to compatible 1,024-sample runs.

For ten output classes and 256-sample inputs, the local complexity audit gives:

| Local model | Parameters | Conv/linear/recurrent MACs | Extra STFT real-op estimate |
|---|---:|---:|---:|
| MCLDNN reimplementation | 406,070 | 98,536,064 | 0 |
| IQFormer-inspired | 354,984 | 88,894,080 | 589,312 |

MCLDNN and IQFormer span recurrent and IQ/time-frequency Transformer-like
inductive biases and are sufficient to reject obviously weak candidates during
bounded screening. CSSL-AMC adds a recent, auditable low-SNR/noise-robust AMC
architecture to the 1,024-sample formal family. The set is **not sufficient for
a strongest-published-method or structured-interference-specific conclusion**
because:

1. MCLDNN is a 2020 architecture;
2. IQFormer is not an exact official reproduction;
3. CSSL-AMC is a supervised architecture adaptation rather than the complete
   published two-stage method;
4. none of the executed rows is a verified recent
   structured-interference-specific method;
5. MFENet and RepCCNet are cited but not executed; and
6. the current fixed schedule has not yet been supplemented by
   architecture-specific tuning sensitivity.

The formal `StrongestBaseline` audit must include CSSL-AMC alongside A0,
MCLDNN, and IQFormer and remain locked until eligible results exist. The
manuscript must not upgrade that local selection to “strongest published” or
“strongest structured-interference” language.

## Minimum author-contact and reproduction plan

### MFENet request

Ask the corresponding author for:

- the exact code revision and license;
- the RML2016.10a/10b preprocessing scripts and split seeds;
- complete model configuration and parameter/FLOP script;
- optimizer, scheduler, regularization and checkpoint-selection configuration;
- official weights and one reference batch with logits; and
- clarification of the interference model beyond SNR/noise.

### RepCCNet request

Ask the corresponding author for:

- the PyTorch 1.12.1 source or an immutable archive;
- dropout probability and sampling semantics;
- the temporal readout feeding the fully connected layer;
- RepCasual bias/normalization/initialization details;
- the exact split seed and checkpoint-selection rule;
- the deploy-time fusion script plus a 35.9K checkpoint; and
- the RadioML 2018.01a configuration.

### Admission sequence after receipt

1. archive the source with a commit/hash and record its license;
2. reproduce the native RadioML result before adapting the data loader;
3. verify train/deploy output equivalence for RepCCNet if applicable;
4. assert parameter count, tensor shapes, deterministic reference logits and
   MAC accounting in unit tests;
5. keep a native-protocol result separate from the unified VIMD protocol; and
6. only then add the model to both experiment registries under an exact
   provenance label.

## Local verification performed

No model or runner source was changed by this audit. The existing baseline and
ablation suite was run after inspection:

```text
python -m unittest discover -s tests -p test_baselines_and_ablations.py -v
15 tests passed
```

The passing checks include MCLDNN/IQFormer batch shapes and parameter targets,
the IQFormer STFT numerical check, recurrent/frontend operation accounting,
and the canonical A0-A7 mapping.
