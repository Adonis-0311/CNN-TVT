# Evidence Gate and Statistical Inference Protocol

Version: `vimd-evidence-gate-v1`

This protocol prevents a successful program run from being mistaken for
publication-grade evidence. A completed execution is necessary but not
sufficient. Passing the administrative gates below also does not establish
statistical power, scientific validity, novelty, or publication readiness.

## 1. Two independent statuses

`run.json` records both:

- `execution_status`: `running`, `complete`, or `failed`;
- `evidence_eligibility`: a structured gate report with booleans, actual and
  required values, and explicit failure reasons.

The legacy `status` field is retained for existing tooling and mirrors
`execution_status`. Smoke and diagnostic runs are allowed to complete normally,
but their evidence eligibility remains false.

## 2. Cache designation is mandatory

The runner reads `manifest.json` at
`configuration.evidence_designation`. The closed designations are:

- screening: `screening` or
  `screening_not_formal_tvt_evidence`;
- formal headline: `headline`, `headline_formal_tvt_evidence`, or
  `formal_headline_tvt_evidence`.

A missing or unrecognized designation cannot be promoted. A screening cache
can become screening-eligible, but `formal_paper_evidence_eligible` remains
false by construction.

## 3. Locked promotion gates

Every designated tier requires:

1. execution completed successfully;
2. immutable-file checksum verification enabled and passed;
3. component validation run on every required split, including reconstruction,
   SNR, SIR, and active-jammer-power checks;
4. complete modulation-class support in train, validation, and held-out data;
5. the minimum split and per-class sample floors;
6. the minimum number of independent algorithm seeds;
7. the complete preregistered model suite.

The administrative floors are:

| Tier | Seeds | Train | Validation | Held-out | Per-class train/validation/held-out |
|---|---:|---:|---:|---:|---:|
| Screening | 3 | 200 | 50 | 200 | 20 / 5 / 20 |
| Headline | 5 | 1,000 | 200 | 500 | 100 / 20 / 50 |

The screening suite is:

`a0_backbone`, `a5_vimd_full`, `mcldnn_reimplementation`,
`iqformer_inspired`.

The headline suite is:

`a0_backbone`, `a1_single_mask`, `a2_tri_no_teacher`,
`a3_tri_teacher`, `a4_tri_teacher_mtl`, `a5_vimd_full`,
`a6_dual_full`, `a7_vimd_no_residual`,
`mcldnn_reimplementation`, `iqformer_inspired`.

Headline promotion additionally requires an explicit `--reference-model` and a
nonempty, predeclared `--holm-candidates` family. Model provenance labels remain
binding: an “inspired” implementation must not be presented as an exact
reproduction of the cited architecture.

These floors are conservative workflow controls, not a prospective power
calculation. The final experimental design still requires a power/sensitivity
argument appropriate to the expected effect size and source-cluster structure.

## 4. Strict paired inference

All paired comparisons require identical probability shapes, labels, source
IDs, and source ordering. A source cluster may not cross modulation classes.
The single-seed bootstrap:

- resamples source clusters, keeping all rows from a source together;
- resamples within modulation class by default, so every draw retains class
  support;
- reports paired differences and 95% percentile intervals for both accuracy
  and macro-F1;
- retains `difference`, `ci95_low`, and `ci95_high` as backward-compatible
  aliases for the accuracy result;
- records the bootstrap seed, number of draws, cluster count, and resampling
  unit.

## 5. Two-layer headline uncertainty

`headline_paired_bootstrap` treats algorithm initialization and held-out
sources as two different uncertainty layers. Each hierarchical draw resamples:

1. fitted-model seeds with replacement; and
2. class-stratified held-out source clusters with replacement.

The same resampled sources are evaluated for every selected algorithm seed.
The output contains the hierarchical interval plus source-only and
algorithm-seed-only intervals for accuracy and macro-F1. The reported point
estimate is the mean paired difference across algorithm seeds.

Predictions from multiple fitted models reuse the same test sources, so they
are not pooled into a McNemar contingency table. The multi-seed output
explicitly records that no pooled exact McNemar test was performed.

## 6. McNemar and Holm boundaries

McNemar's exact test is run only when all of the following hold:

- one fitted-model seed is being compared;
- the split is held out, not validation;
- the candidate was declared in `--holm-candidates` before training.

Validation comparisons receive descriptive paired bootstrap intervals only.
They never enter a Holm family.

Holm correction is applied separately for each held-out regime and algorithm
seed, using only the declared candidate family against the declared reference.
Every included row records a complete `holm_family_id`, family size, reference,
regime, seed, and candidate list. Undeclared exploratory candidates are not
silently added to the family.

If `--reference-model` is omitted, the runner may choose a backbone as a
descriptive comparison anchor so diagnostic workflows remain convenient. The
record explicitly says `automatic_descriptive_anchor_not_strongest`; no
strongest-baseline claim is made, and formal headline eligibility fails.

## 7. Audit artifacts

Each run writes:

- `run.json`: execution status, gate decision, protocol, environment, models,
  and results;
- `metrics.csv`: per-model, per-seed metrics;
- `seed_aggregates.csv`: ordinary across-seed summaries when multiple seeds
  exist;
- `paired_statistics.csv`: single-seed paired intervals and only eligible
  McNemar/Holm results;
- `headline_paired_statistics.csv`: hierarchical multi-seed/source paired
  intervals for held-out regimes only;
- prediction archives keyed by model, seed, cache digest, and split.

The runner exposes `evaluation_split_names(contract)` as the discovery hook for
additional manifest-declared factor-isolated held-out regimes. Train remains a
fitting split and validation remains tuning-only; every other declared
evaluation split must receive its own inference and Holm family.
