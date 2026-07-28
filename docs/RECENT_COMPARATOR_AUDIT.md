# Recent AMC Comparator Audit and Preregistration

**Audit date:** 2026-07-28  
**Reviewer issue:** M07, recent low-SNR/structured-interference comparator  
**Decision:** admit one executable recent low-SNR comparator as an
official-architecture **supervised adaptation**; do not claim a complete
published-method reproduction

This audit is a claim-control record. It identifies what was verified from
publisher or author-controlled sources, what was implemented locally, what was
deliberately omitted, and how the comparator is kept inside the common formal
budget. It contains no performance result.

## Resolution

`cssl_amc_supervised_adaptation` is now preregistered in the formal
model-by-seed family. It is based on the official architecture released for:

> M. Du, J. Pan, and D. Bi, “A Contrastive Learner for Automatic Modulation
> Classification,” *IEEE Transactions on Wireless Communications*, vol. 24,
> no. 4, pp. 3575--3589, 2025, doi: 10.1109/TWC.2025.3532438.

The authors describe their repository as the official implementation and
license it under Apache-2.0. The local source lock fixes commit
`2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f`.

This closes the absence of an executable 2024--2026 **low-SNR** architecture
comparator. It does not close the narrower gap of an executable, officially
released comparator designed for the same tone/chirp/pulse/partial-band/
cochannel structured-interference taxonomy.

## Admission criteria

A recent comparator could enter the formal family only if all of the following
were resolved before any formal result was opened:

- a publisher record and author-controlled source could be linked;
- an immutable source revision and a redistribution license could be recorded;
- the executable graph and input contract were sufficiently explicit to avoid
  inventing material layers;
- the method could consume received IQ alone at inference, without clean
  signals, oracle components, instantaneous SNR, or external test metadata;
- differences from the published training method could be named precisely;
- both runner registries, the frozen model list, and the expected
  model-by-seed release gate could be synchronized; and
- a local unit test could assert tensor shape, parameter count, provenance, and
  freeze synchronization without training.

## Candidate disposition

| Candidate | Year | Direct relevance | Primary-source/code outcome | Formal disposition |
|---|---:|---|---|---|
| CSSL-AMC | 2025 | Noise corruption and low-SNR robustness | Official author repository, immutable commit, Apache-2.0, complete executable topology | **Admitted as supervised architecture adaptation** |
| MFENet | 2025 | Direct interference-tolerant/LSNR AMR | Publisher record found; bounded search found no author-controlled code, checkpoint, or complete accessible specification | Related work only; not executed |
| RepCCNet | 2024 | Recent AMC architecture | Same-team patent resolves much of the topology, but dropout, temporal readout, branch details, and official source remain unresolved | Excluded from executable registry |
| CAIC-Net | 2026 | Cross-SNR robustness | Publisher full text resolves the method; dynamic cross-attention consumes instantaneous SNR at inference, and no verified official code was found | Excluded because it changes the mixture-only information contract |
| DenoMAE2.0 | 2026 | Low-SNR denoising representation | Publisher record and an author announcement were found, but the advertised code destination could not be resolved and audited | Excluded pending verifiable source/license |

The negative code findings are bounded to the audit date and the exact-title,
DOI, method-name, and author surfaces inspected. They are not claims that
private, renamed, or later-released code does not exist.

## Primary-source ledger

| Evidence | Immutable or canonical source | Use in this audit |
|---|---|---|
| CSSL-AMC publisher record | [IEEE Xplore 10857965](https://ieeexplore.ieee.org/document/10857965/) | Canonical bibliographic record |
| CSSL-AMC official code | [Author repository](https://github.com/dumingyang20/CSSL-AMC-Pytorch) | Official status, data outline, two-stage procedure, paper metadata, license |
| Audited revision | [Commit `2fbc5b3e...`](https://github.com/dumingyang20/CSSL-AMC-Pytorch/commit/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f) | Immutable implementation boundary |
| Encoder/classifier | [`models/cssl.py`](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/models/cssl.py) | Network topology and forward semantics |
| Native pretraining | [`run.py`](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/run.py), [`contrastive_learning.py`](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/contrastive_learning.py) | Momentum encoder, contrastive objective, native schedule |
| Native fine-tuning | [`finetune.py`](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/finetune.py) | Downstream classifier and native fine-tuning |
| Input contract | [`dataset.py`](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/dataset.py) | Raw-IQ shape and 1024-sample frame |
| License | [Apache-2.0 at the audited revision](https://raw.githubusercontent.com/dumingyang20/CSSL-AMC-Pytorch/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f/LICENSE) | Permitted local adaptation and notice obligation |
| MFENet | [IEEE Xplore 11037461](https://ieeexplore.ieee.org/document/11037461/) | Canonical direct low-SNR/interference-tolerant related work |
| RepCCNet | [IEEE Xplore 10420493](https://ieeexplore.ieee.org/document/10420493/), [CN117354106B](https://patents.google.com/patent/CN117354106B/zh) | Canonical paper and same-team topology disclosure |
| CAIC-Net | [Publisher full text](https://www.mdpi.com/1424-8220/26/3/756), [archived full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12899915/) | Dynamic SNR-conditioned cross-attention and training method |
| DenoMAE2.0 | [DOI record](https://doi.org/10.1109/TCOMM.2025.3626031) | Canonical bibliographic record |

Discovery/search pages were used only to locate original sources. No
architecture was reconstructed from a search snippet or secondary summary.
The machine-readable immutable ledger is
`tvt_submission/sources/cssl_amc_2025.lock.json`; the local license copy is
`tvt_submission/sources/licenses/Apache-2.0.txt`.

## Audited CSSL-AMC topology

For an input tensor `[batch, 2, 1024]`, the local class
`CSSLAMCSupervisedAdaptation` retains the following official architecture:

1. a three-convolution noise-level estimator `2 -> 32 -> 32 -> 2`, with
   kernel size 3, unit stride, same padding, and ReLU after each convolution;
2. concatenation of raw IQ and estimated noise into four channels;
3. a one-dimensional residual encoder with stage widths 64 and 128, two
   residual blocks per stage, and stride 2 in the first block of stage 2;
4. fixed flattening of `128 x 512` activations into a 128-dimensional
   representation, followed by BatchNorm and ReLU;
5. the official two-linear-layer classifier `128 -> 64 -> classes`; and
6. the input BatchNorm parameters that are defined but unused by the official
   forward path, retained for parameter/state-dictionary topology.

For the frozen ten-class taxonomy, the implementation has exactly 8,631,948
trainable parameters. The model deliberately rejects non-1024-sample frames
instead of silently changing the published fixed readout.

## Material adaptation boundary

The published CSSL method is a two-stage learner: momentum-encoder contrastive
pretraining followed by downstream fine-tuning. The official scripts use a
substantially larger native procedure than the common formal budget. The local
formal comparator changes:

- the 24-class RadioML head to the frozen ten-class taxonomy;
- random initialization in place of any official or external checkpoint;
- the complete contrastive-pretraining-plus-fine-tuning procedure to the same
  paired-view supervised cross-entropy protocol used by other classification
  baselines;
- the native optimizer/schedule to the frozen common optimizer, 30-epoch
  budget, checkpoint rule, and seeds; and
- the return type to the local evaluator's `logits`/`embedding` dictionary.

Therefore the only admissible paper label is **“CSSL-AMC supervised
adaptation.”** It is not an official CSSL reproduction, an official CSSL
result, or evidence about the complete two-stage method.

## Fairness and interpretation

The formal comparison is controlled for data and optimization budget:

- identical source identities, globally source-disjoint splits, and received
  IQ inputs;
- the same five algorithm seeds `17, 29, 43, 71, 101`;
- the same 30-epoch ceiling, Adam-family configuration, label smoothing, and
  fail-closed checkpoint policy;
- no external weights, clean-signal input, component target, SNR metadata, or
  test-time side information; and
- the same single-view prediction contract and evaluation roles.

This is **unified-budget fairness**, not architecture-optimized fairness.
CSSL-AMC's native two-stage procedure totals 200 pretraining epochs plus 200
fine-tuning epochs in the audited scripts, so the common-budget adaptation may
understate the published method's attainable performance. Conversely, its
8.63-million-parameter model is much larger than the local compact baselines.
Parameters, MACs, and measured latency must therefore accompany accuracy/F1;
the study does not claim parameter matching.

The comparator is eligible for the same paired bootstrap and predeclared Holm
family as the other formal candidates. It must remain “pending” in internal
review until every frozen seed has completed and the release gate validates
the run. No quoted CSSL paper number may be inserted into the unified-retraining
table.

## Formal integration

The following surfaces now use the same registry name
`cssl_amc_supervised_adaptation`:

- `src/vimd_amc/models/baselines.py`;
- `experiments/run_experiment.py`;
- `experiments/run_standard_experiment.py`;
- the `headline` preregistered suite;
- `tvt_submission/configs/formal_tvt_freeze_v1.json`;
- the formal Holm candidate family; and
- `tests/test_recent_comparator.py`.

This synchronization means the existing expected model-by-seed release gate
will fail if the new comparator or any of its five seeds is absent. No
performance-based decision was used to admit the model, and no formal cache or
formal result was opened during the audit.

## Bibliographic correction

The MFENet record has seven authors: Xiaochuan Sun, Changcheng Wu, Yiqing Li,
Yingqi Li, Tianyu Huang, Jike Yu, and Haijun Zhang. The local BibTeX record was
corrected to include both Yiqing Li and Yingqi Li. CSSL-AMC already had the
publisher-supplied author, volume, issue, page, year, and DOI fields.

## Verification performed

No formal training, formal cache generation, checkpoint import, GPU job, or
large data download was performed. The existing standard-runner suite includes
one isolated one-epoch CPU smoke fit on a tiny synthetic temporary cache; it
cannot be promoted and produces no formal result. The dedicated comparator
test verifies:

- exact parameter count, fixed readout, block counts, and output shapes;
- strict `[batch, 2, 1024]` input validation;
- claim-safe provenance and Apache-2.0 metadata;
- both experiment registries and the paired-view CE objective;
- synchronization of the formal freeze, Holm family, and headline suite; and
- presence of the immutable source lock and local license copy.

The completed lightweight checks were:

```text
test_recent_comparator.py          6/6 passed
test_baselines_and_ablations.py   15/15 passed
test_standard_runner.py            5/5 passed
JSON freeze/source-lock parse      passed
```

The formal result status remains `not_executed`.

## Residual evidence gaps

1. No recent, directly structured-interference-specific model with adequately
   licensed official executable source passed this audit.
2. CSSL-AMC provides a recent low-SNR/noise-corruption comparator, not a model
   designed for the exact structured jammer taxonomy.
3. A native two-stage CSSL sensitivity would require a separate prospective
   protocol and materially larger compute budget; it cannot be added after
   inspecting formal test results.
4. No performance, superiority, “state of the art,” or strongest-baseline
   statement is permitted until the complete formal run and release gate pass.
