# MATLAB 3GPP TDL backend

This directory contains an offline batch bridge from Python to MATLAB R2025a
5G Toolbox `nrTDLChannel`.  It provides a standards-aligned alternative to the
repository's lightweight heuristic proxy channel.

Implemented profiles are the 3GPP TR 38.901 tapped-delay-line profiles TDL-A,
TDL-B, TDL-C, TDL-D, and TDL-E.  Every waveform uses an independent SISO
channel object with a recorded seed.  The maximum Doppler shift is computed as

`speed_mps * carrier_frequency_hz / 299792458`.

The default sample rate is 1 MHz.  MATLAB returns the same number of samples as
the input; startup filter transients remain part of the waveform and the
channel/filter delays are recorded in metadata so downstream crop policy can
be audited.

## Smoke validation

From the repository root:

```powershell
python standards/smoke_nrtdl_backend.py
```

The test exercises:

- TDL-A, TDL-B, TDL-C, TDL-D, and TDL-E in one MATLAB process;
- exact repeatability for identical profile/speed/seed inputs;
- distinct realizations for different seeds;
- zero versus positive Doppler;
- finite, nonzero, same-length outputs;
- returned MATLAB, 5G Toolbox, path, filter, and power metadata.

The smoke script uses an automatically removed temporary directory for its
transfer MAT files and writes a compact `standards/smoke_result.json`.  To
inspect the raw interchange files during backend development, pass an explicit
empty `work_directory` to `apply_nrtdl_batch`; the wrapper only auto-removes
files when that argument is omitted.

The integration test can be run separately:

```powershell
python -m unittest standards.tests.test_nrtdl_backend -v
```

## Python API

```python
from vimd_amc.standards import NRTDLConfiguration, apply_nrtdl_batch

result = apply_nrtdl_batch(
    waveforms,  # complex NumPy array [batch, samples]
    [
        NRTDLConfiguration(
            profile="TDL-C",
            delay_spread_s=300e-9,
            speed_mps=33.33,
            carrier_frequency_hz=5.9e9,
            seed=1001,
        )
        for _ in range(waveforms.shape[0])
    ],
    sample_rate_hz=1e6,
)
```

This is a cache-generation backend, not an online data-loader transform:
MATLAB startup cost should be amortized using batches.  A final paper dataset
must additionally document scenario-to-profile/delay-spread mapping, crop
policy, cross-window trajectory continuity, jammer placement relative to the
channel, and statistical validation against the intended vehicular scenario.
For a strict held-out-profile experiment, choose disjoint training and test
sets explicitly (for example, train on TDL-A/C/D and reserve TDL-B/E); backend
support alone does not make a split held out.

## Offline paired cache

`cache_builder.py` builds a read-only `.npy` cache whose large arrays are
memory-mapped by `CachedPairedAMCDataset`.  Its three source groups are
disjoint, train/validation use only TDL-A/C/D, and `heldout_channel` uses only
TDL-B/E.  Each paired view records independent target/jammer channel seeds,
returned path/filter delays, fixed guard/crop indices, realized component
power, file checksums, and a deterministic manifest digest.

```powershell
python standards/cache_builder.py --output standards/cache_smoke_rebuild
python -m unittest standards.tests.test_tdl_cache -v
```

The smoke defaults above are unchanged.  Larger audited screening caches can
provide explicit comma-separated grids.  Negative SNR/SIR grids should use
the `--option=value` spelling so `argparse` does not interpret the leading
minus sign as another option:

```powershell
python standards/cache_builder.py `
  --output standards/cache_screening_example `
  --modulations BPSK,PI2BPSK,QPSK,8PSK,16QAM,64QAM,256QAM,GMSK,CPFSK,4FSK `
  --jammers tone,chirp,pulse,partial_band,cochannel,multitone `
  --train-profiles TDL-A,TDL-C,TDL-D `
  --heldout-profiles TDL-B,TDL-E `
  --speeds-kmh 0,60,120,150 `
  --delay-spreads-ns 30,100,300 `
  --snr-db-values=-10,-6,-2,2,6,10,14,18 `
  --sir-db-values=-15,-10,-5,0,5,10 `
  --evidence-designation screening_not_formal_tvt_evidence
```

Every effective list, split size, waveform setting, seed, continuous fallback
range, and evidence designation is written to `manifest.json`.  A screening
designation is an explicit warning: such a cache is for model triage and
pipeline validation, not a source of formal TVT result claims.

The builder refuses to overwrite an existing destination.  Target and jammer
are independently channelized before
`SignalSynthesizer.finalize_received_components` applies shared receiver
impairments, exact SNR/SIR scaling, noise, and AGC.  The cache validator
recomputes `x = clean + jammer + unexplained`, SNR, and SIR from the stored IQ
arrays.

## Factor-isolated cache

`build_factor_cache.py` implements the manuscript's nine source-disjoint
splits: train, validation, ID test, hard interference, unseen jammer, unseen
speed, held-out TDL profile, combined OOD, and clean retention.  The split
policy is stored verbatim in schema-2 manifests together with actual factor
coverage and per-split component audits.

```powershell
python standards/build_factor_cache.py `
  --output standards/cache_factor_preview `
  --preset micro `
  --print-policy-only

python -m unittest standards.tests.test_factor_isolated_cache -v
```

The clean split stores an exactly zero jammer and uses the quality validity
mask to mark SIR as undefined.  Seen/held jammer, speed, and TDL-profile pools
are validated as disjoint before MATLAB is invoked. Train, validation, and ID
test use a deterministic 20% clean-view quota; the requested and realized
counts are recorded rather than left to random sampling. Schema-2 caches also
store physical Doppler targets and their explicit normalization scale. Full
policy, commands, sentinel audit, runner promotion gates, and evidence
restrictions are in
`docs/FACTOR_ISOLATED_CACHE_PROTOCOL.md`.

### Current 1024-sample screening cache

`cache_factor_screening_1024_v1` is the current ambiguity-safe schema-2 cache
for stable model screening only. It was built with the locked nine-split
policy, MATLAB R2025a / 5G Toolbox 25.1, 1,024 samples, a 96-sample guard, and
master seed `20260727`.

- designation: `screening_not_formal_tvt_evidence`;
- split sizes: 1,000 train, 200 validation, and 500 in each of the seven test
  regimes;
- support: 4,700 globally source-disjoint rows, 9,400 views, and all ten
  modulation classes exactly balanced within every split;
- manifest digest:
  `241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3`;
- storage: 190 files and 517,687,920 bytes;
- independent audit: all 189 declared array checksums, shapes, dtypes, and
  finiteness checks passed; worst component/SNR/SIR/quality-normalization
  errors were `4.471745123805678e-7`, `4.433454936503267e-8` dB,
  `5.601630050477979e-6` dB, and `5.960464477539063e-8`;
- guard audit: maximum required guard 21 samples and minimum realized margin
  75 samples;
- clean jammer power: exactly zero.

The durable provenance and independent audit are
`cache_factor_screening_1024_v1.prebuild.json` and
`cache_factor_screening_1024_v1.audit.json`; the audit JSON SHA-256 is
`1b934e9be4cb8576f24e7bad0aaff7150653cae350490c631e4991a3bdc49f28`.
The reusable checker is `audit_factor_cache.py`.

The audit reports this cache as usable for stable screening. It also records
that the broader experiment source tree changed concurrently after the
prebuild snapshot, while all three cache-construction-critical files retained
their frozen hashes. This cache must never be relabeled or cited as formal or
headline TVT evidence.
