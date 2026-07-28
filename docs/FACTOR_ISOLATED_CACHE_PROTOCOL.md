# Factor-Isolated MATLAB-TDL Cache Protocol

## Purpose and claim boundary

This protocol makes the split plan in the manuscript executable. It separates
jammer-family, terminal-speed, and TDL-profile shifts instead of reporting only
a combined out-of-distribution score. The channel primitive is MATLAB
`nrTDLChannel` configured with 3GPP TR 38.901 TDL profiles. This is controlled,
standards-aligned synthetic evidence; it is not a complete V2X geometry,
trajectory, blockage, hardware, or system-compliance claim.

The immutable policy lives in
`src/vimd_amc/standards/cache.py::factor_isolated_split_policies`. Split sizes
are parameters, but factor support is shared across the `micro`, `screening`,
and `headline` CLI presets. A preset size and an evidence label are
administrative controls, not a prospective power analysis.

## Locked factor policy

| Split | Role | Jammer families | Speeds (km/h) | TDL profiles | SIR |
|---|---|---|---|---|---|
| `train` | model fitting | seen + 20% clean | 0/60/120/150 | A/C/D | -10/-5/0/5/10 dB when active |
| `validation` | checkpoint selection only | seen + 20% clean | 0/60/120/150 | A/C/D | -10/-5/0/5/10 dB when active |
| `id_test` | source-disjoint ID test | seen + 20% clean | 0/60/120/150 | A/C/D | -10/-5/0/5/10 dB when active |
| `hard_interference` | severity stress test | seen | 0/60/120/150 | A/C/D | -15/-10/-5/0 dB |
| `unseen_jammer` | isolated jammer-family shift | held | 0/60/120/150 | A/C/D | -10/-5/0/5/10 dB |
| `unseen_speed` | isolated mobility shift | seen | 180/250 | A/C/D | -10/-5/0/5/10 dB |
| `heldout_channel` | isolated channel-profile shift | seen | 0/60/120/150 | B/E | -10/-5/0/5/10 dB |
| `combined_ood` | joint stress test | held | 180/250 | B/E | -10/-5/0/5/10 dB |
| `clean_retention` | no-interference control | none | 0/60/120/150 | A/B/C/D/E | invalid |

The seen jammer set is
`{tone, multitone, chirp, sweep, partial_band, comb}`. The held jammer set is
`{pulse, ofdm_like}`. These sets are validated as disjoint. Tone and multitone
remain on the same side of the split because they share a sparse-line spectral
structure; using one as the held-out family for the other would overstate the
novelty of the shift. The seen and held speed sets and the A/C/D versus B/E
profile sets are also validated as disjoint. Every split has a unique
`source_key`, and source IDs are checked for global pairwise disjointness.

### Identifiability exclusions

`cochannel` is excluded from the primary modulation-classification protocol.
With only a single-antenna mixture of two in-taxonomy modulated emitters, a
preassigned “target” label is not identifiable under exchange of the two
sources, particularly near equal power. A defensible cochannel experiment
therefore needs a physical emitter anchor, a collision label outside the
modulation taxonomy, a pre-registered dominant-emitter rule, or a separate
ambiguity-stress analysis. It must not silently enter the primary accuracy
average.

`mixed` is also excluded from the locked nine-split protocol. The generator
can compose two jammer primitives, but a frozen combination distribution and
an independently identified mixture label have not yet been pre-registered.
It belongs in a separate compositional-generalization experiment.

Both exclusions and their required resolutions are serialized in
`protocol_exclusions`. The ordered generator taxonomy still retains these
output columns for schema compatibility; auxiliary metrics must report
unsupported classes as unsupported rather than treating their absence as
success.

All active-jammer splits use the same explicit SNR grid:
`{-10, -6, -2, 2, 6, 10, 14, 18}` dB. The clean split uses that SNR grid but
has no defined SIR.

Clean examples in train, validation, and ID test are not left to random
chance. The builder computes an exact 20% quota over paired-view slots and
places those slots deterministically across each split. For very small
sentinels, it preserves at least one clean and one active view, so the actual
fraction can differ from 20%; requested and realized counts/fractions are both
recorded. Hard, unseen-jammer, unseen-speed, held-out-channel, and combined-OOD
splits remain entirely active-interference tests.

## Clean-record semantics

The cache never encodes a clean record as an arbitrarily large SIR:

- the stored jammer waveform and jammer multi-hot labels are exactly zero;
- `quality_mask[..., 1]` is zero, making the finite `sir_db` storage sentinel
  semantically invalid;
- per-view metadata records `sir_db=null`, `sir_valid=false`, and
  `interference_present=false`;
- component validation skips the SIR identity only when that validity bit is
  false and separately verifies that jammer power is no greater than
  `1e-12`.

For active-jammer records, `quality_mask[..., 1]` is one and the validator
recomputes realized SIR from the stored target and jammer IQ.

## Manifest schema and audit trail

Factor caches use manifest schema version 2. In addition to the existing
configuration, file SHA-256 values, source IDs, record metadata, and
deterministic cache digest, the manifest includes:

- `preregistered_split_policy`: requested role, profiles, jammer families,
  speeds, SNR/SIR grids, size, source key, and held factors for every split;
- `split_roles`: machine-readable train/selection/test semantics;
- `factor_coverage`: actual modulation, jammer, speed, profile, SNR, SIR, and
  SIR-validity coverage, plus an in-policy check for every factor;
- `component_audit`: per-split component residual, SNR/SIR error, active and
  clean view counts, jammer-power bounds, and guard margin;
- `quality_normalization`: explicit physical scales and units for SNR, SIR,
  and Doppler. SNR/SIR use a 20 dB scale; Doppler uses the maximum configured
  speed converted by
  `speed_kmh/3.6 * carrier_frequency_hz / 299792458`;
- `jammer_taxonomy`: the exact ordered names corresponding to every column of
  the jammer multi-hot target;
- `protocol_exclusions`: machine-readable exclusions for `cochannel` and
  `mixed`, including the scientific reason and the condition required to
  admit either into a future protocol;
- `files`: array shape, dtype, path, and SHA-256 for every split.

`CachedPairedAMCDataset` discovers arbitrary splits from the manifest and
checks all declared source-ID groups for leakage. Schema-1 caches
`cache_smoke` and `cache_screening_v1` remain readable without rewriting.
Schema-2 caches additionally store `doppler_hz` per view and audit agreement
between physical targets and normalized quality labels.

## Building caches

Print and validate the policy without calling MATLAB:

```powershell
python standards/build_factor_cache.py `
  --output standards/cache_factor_preview `
  --preset micro `
  --print-policy-only
```

Build a one-source-per-split integration sentinel:

```powershell
python standards/build_factor_cache.py `
  --output standards/cache_factor_micro_v4 `
  --preset micro `
  --sample-length 64 `
  --guard-samples 48
```

Build sizes can be changed without editing factor support:

```powershell
python standards/build_factor_cache.py `
  --output standards/cache_factor_screening_v1 `
  --preset screening `
  --split-size train=2000 `
  --split-size validation=400
```

The builder refuses an existing destination. Never relabel or mutate a
previous cache; choose a new versioned path.

The `headline` preset is intentionally large, but it is not automatically
adequate. Before creating it, freeze the sample-size justification, model
suite, reference model, multiplicity family, seeds, stopping rule, and hardware
measurement protocol. Passing software checks does not authorize a paper
claim.

## Executed micro sentinel

`standards/cache_factor_micro_v4` was built with MATLAB R2025a using one source
and two independently impaired views per split. It is pipeline evidence only.

- schema: 2
- cache digest:
  `d2283702be8f608422613db199308386257653b71088f13b4043a096948150e7`
- files / total bytes: 190 / 280,005
- globally disjoint source IDs: 9
- hard-interference maximum requested/realized SIR grid value: 0 dB
- train/validation/ID actual clean fraction: 1 of 2 views each; the inflated
  50% is an explicitly recorded micro-sentinel consequence, not the formal
  target
- clean SIR-invalid views: 2 of 2
- worst post-read component residual across splits:
  `1.5819330221106526e-7`
- worst post-read realized SNR absolute error:
  `9.576586279536059e-8` dB
- worst post-read active-record SIR absolute error:
  `9.687971136429496e-8` dB
- worst physical-quality normalization error: `5.960464477539063e-8`
- maximum clean jammer power: 0

These values validate serialization, component identities, clean validity
semantics, factor constraints, and MATLAB/Python plumbing. With one BPSK source
per split, the sentinel has neither class support nor sample size for model
comparison.

`cache_factor_micro_v1`, `cache_factor_micro_v2`, and
`cache_factor_micro_v3` are retained immutably as development sentinels and
are superseded by v4. Version 1 predates stratified training-clean views;
version 2 predates the explicit jammer-label taxonomy; version 3 used the
scientifically weaker tone-versus-multitone separation and admitted the
single-antenna cochannel ambiguity into the primary protocol. None is the
current protocol reference.

## Runner and promotion gates

`experiments/run_standard_experiment.py` loads every manifest-declared split,
uses only `train` for fitting and `validation` for checkpoint selection, and
evaluates all remaining regimes. It writes a split reference and
source-aligned prediction bundle for each regime.

After checkpoint selection is complete, models that actually expose
`jam_logits` and/or `quality` heads receive a separate auxiliary evaluation on
every evaluation split. Physical SNR/SIR/Doppler MAE uses the manifest scales
and the stored physical targets; clean SIR is reported as unavailable with
zero support, never as zero error. Models without auxiliary heads receive an
explicit `status=unavailable` record. These metrics are marked
`used_for_checkpoint_or_model_selection=false`.

Headline promotion is rejected unless all of the following hold:

- schema version is at least 2 and the split set is exactly the nine locked
  regimes;
- pre-registered policies and split roles are complete;
- every actual factor value lies within its policy;
- the clean split has only `none` jammer records and all SIR validity bits are
  false;
- physical quality normalization is explicit;
- the ordered jammer-label taxonomy is explicit and matches the generator;
- the locked `cochannel` and `mixed` exclusions are present, and neither
  appears in a primary split;
- checksums and component validation cover every split;
- every split meets the sample/per-class administrative floors;
- the executable source-tree fingerprint is identical at run start and end.

If source files change during execution, the machine-readable reason is
`source_tree_mutated_during_execution`. Schema-1 caches may support diagnostic
or screening work but can never be promoted to headline evidence.

## Required tests

```powershell
python -m unittest standards.tests.test_factor_isolated_cache -v
python -m unittest discover -s standards/tests -v
python -m unittest discover -s tests -v
```

The integration test uses a temporary schema-2 cache and is skipped only when
MATLAB is unavailable. The unit tests still validate all isolation constraints
and deterministic source construction without MATLAB.
