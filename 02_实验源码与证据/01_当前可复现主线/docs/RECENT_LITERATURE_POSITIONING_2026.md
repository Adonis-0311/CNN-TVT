# 2024--2026 AMC Literature Positioning Update

**Audit date:** 2026-08-17  
**Search window:** 2024-01-01 through 2026-08-17  
**Purpose:** current related-work positioning and executable-comparator claim control

This update is a literature and reproducibility audit, not a performance
comparison. Publisher/DOI records and author-controlled repositories are the
admitted sources. Search indexes were used only to locate those primary
surfaces.

## Current positioning

Recent AMC work covers four neighboring but non-identical problems:

| Line of work | Representative primary records | Relation to this study |
|---|---|---|
| Compact or efficient AMC | [RepCCNet, IEEE](https://ieeexplore.ieee.org/document/10420493/) | Motivates compactness; it does not test this study's frozen structured-jammer operating plane. |
| Robust representation and adaptation | [SigDA](https://doi.org/10.1109/TWC.2024.3399067), [MCLHN](https://doi.org/10.1109/TWC.2024.3412234), [CSSL-AMC](https://ieeexplore.ieee.org/document/10857965/), [DenoMAE2.0](https://ieeexplore.ieee.org/document/11218877/), [CGPCDA](https://ieeexplore.ieee.org/document/11455267/) | Establishes that domain adaptation, contrastive objectives, and denoising representations are active alternatives. They do not make masking or contrastive learning alone a novelty claim here. |
| Interference or overlap-aware recognition | [Overlapped signals](https://doi.org/10.1109/OJCOMS.2024.3416750), [MFENet](https://ieeexplore.ieee.org/document/11037461/), [malicious-interference MIMO tensor method](https://ieeexplore.ieee.org/document/11454674/) | Closest motivation, but the signal models, information contracts, and evaluation axes differ. These papers are contextual references rather than quoted-number rows in the unified retraining table. |
| Transformer/multimodal AMC | [IQFormer](https://doi.org/10.1109/TCCN.2024.3485118) | Motivates the Transformer-like comparator. The local row is explicitly IQFormer-inspired, not an exact published reproduction. |

The defensible novelty is therefore not “first robust AMC,” “first
interference-tolerant AMC,” or universal state of the art. It is the audited
**operating-envelope result**: under one frozen source-disjoint structured-
interference simulation protocol, a compact spectral decomposition model
changes rank as occupancy and SIR change; the study maps that boundary,
reports the failed global gates, and prospectively diagnoses and repairs its
clean-retention representation gap.

## 2026 metadata refresh

The following records were resolved from the DOI registry and their IEEE
publisher identifiers on 2026-08-17:

| Paper | Verified publication metadata | IEEE document |
|---|---|---|
| DenoMAE2.0 | *IEEE Transactions on Communications*, vol. 74, pp. 929--943, 2026; DOI `10.1109/TCOMM.2025.3626031` | `11218877` |
| Confidence-Guided Prototypical Contrastive Domain Adaptation | *IEEE Transactions on Cognitive Communications and Networking*, vol. 12, pp. 6929--6941, 2026; DOI `10.1109/TCCN.2026.3677189` | `11455267` |
| Modulation Recognition via Tensor Analysis for MIMO Systems With Malicious Interference | *IEEE Transactions on Vehicular Technology*, vol. 75, no. 8, pp. 18609--18613, Aug. 2026; DOI `10.1109/TVT.2026.3674535` | `11454674` |

The last record was previously stored as Early Access (`pp. 1--5`). The local
BibTeX is updated to its assigned volume, issue, and pages.

## Executable-code disposition

- CSSL-AMC remains the only 2024--2026 architecture in the frozen comparator
  family with an audited author repository, immutable source lock, and
  redistribution license. The executable row remains a **supervised
  architecture adaptation**, not the complete published two-stage method.
- A public DenoMAE2.0 repository is now visible under the first author's
  account: [atik666/denoMAE2](https://github.com/atik666/denoMAE2). The page
  exposes pretraining/fine-tuning scripts and links the paper, but on the audit
  date it displayed no license and no release. This supersedes the older
  “destination unresolved” finding but still fails the frozen admission rule;
  no source is copied and no executable baseline is registered.
- MFENet and RepCCNet retain their earlier non-admission decisions. No
  published topology is reconstructed from missing details.
- CGPCDA and the malicious-interference MIMO method are cited for positioning
  only. They have not passed the repository/license/input-contract audit needed
  for inclusion in the unified retraining family.

## Manuscript claim controls

The manuscript may say that it evaluates recent and classical **local
comparators under one unified budget**, and that the severe-SIR result is
positive against both MCLDNN and IQFormer-inspired in this simulation. It must
not say:

- strongest published or state-of-the-art AMC method;
- complete reproduction of IQFormer or CSSL-AMC;
- measured, field, SDR, over-the-air, vehicular deployment, or operational
  validation; or
- direct superiority to MFENet, DenoMAE2.0, CGPCDA, or the malicious-
  interference tensor method, because those methods were not retrained in the
  frozen protocol.

## Primary-source ledger

- IEEE publisher records: [RepCCNet](https://ieeexplore.ieee.org/document/10420493/), [CSSL-AMC](https://ieeexplore.ieee.org/document/10857965/), [MFENet](https://ieeexplore.ieee.org/document/11037461/), [DenoMAE2.0](https://ieeexplore.ieee.org/document/11218877/), [CGPCDA](https://ieeexplore.ieee.org/document/11455267/), and [malicious-interference MIMO](https://ieeexplore.ieee.org/document/11454674/).
- Author-controlled DenoMAE2.0 repository: [atik666/denoMAE2](https://github.com/atik666/denoMAE2).
- Immutable CSSL source boundary: [commit `2fbc5b3e...`](https://github.com/dumingyang20/CSSL-AMC-Pytorch/commit/2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f).

Negative or incomplete code findings are bounded to this audit date and the
exact-title/DOI/author surfaces inspected. They are not claims about private,
renamed, or later-released code.
