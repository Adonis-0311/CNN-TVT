# V1.3 record: the envelope covariate is an overlap, not an occupancy

**Date:** 2026-08-18
**Scope:** naming and provenance only. No number, model, split, or claim changed.

## What the covariate actually is

Source of truth: `src/vimd_amc/data/synthesis.py::spectral_overlap_metrics`,
field `jammer_to_signal_overlap`.

1. The tracked target and tracked jammer components are transformed separately
   with a 128-point Hann-windowed STFT, hop 32 (`_stft_power`, defaults
   `fft_size=128, hop=32`). For a 512-sample window this is a 128x13 lattice
   at 7.8 kHz per bin.
2. Time-frequency cells are sorted by descending **target** power, and the
   target's 99%-energy support `S` is the smallest set of cells whose
   cumulative target power reaches 99% of the target total.
3. The covariate is the fraction of the **jammer's total** STFT power that
   falls inside `S`:

   `o = sum_{(f,t) in S} P_j(f,t) / sum_{(f,t)} P_j(f,t)`

So `o = 0.8` means 80% of the jammer's energy lands inside the target's
dominant support. It does **not** mean the jammer fills 80% of the plane.
The manuscript previously called this "jammer spectral occupancy", which
invites exactly the wrong reading. It is now the **target-support jammer
overlap** throughout.

## Why the analysis lattice differs from the model lattice

| | FFT | hop | window | bins x frames | resolution |
|---|---:|---:|---|---|---|
| VIMD front end (`ComplexSTFT`) | 64 | 16 | periodic Hann | 64 x 29 | 15.6 kHz/bin |
| Overlap analysis (`_stft_power`) | 128 | 32 | symmetric Hann | 128 x 13 | 7.8 kHz/bin |

These are different by design and the difference is not post-hoc:

- the overlap is computed **at cache-build time**, before any model is trained;
- it is stored as `overlap.npy` per split and **SHA-256 checksummed in the
  sealed cache manifest**, so it is covered by `cache_digest`;
- every model, seed, split, and comparison in the paper reads the same stored
  values;
- it is carried in the batch dictionary but is **never consumed** by
  `training.py` or `losses.py`, so it is not an input feature of VIMD-Net or of
  any comparator, and it cannot have been tuned against model predictions.

The analysis lattice is a measurement instrument for the physical audit and is
given finer frequency resolution for that purpose; the front-end lattice is
what the model sees.

## Two internal inconsistencies fixed in this pass

1. **Pearson r scope.** The text attributed `r = -0.03` to the fitting cells.
   Recomputed: `r = -0.0026` over the 32 fitting cells and `-0.0254` over all
   113 cells. The manuscript now reports both, with the fitting-cell value
   first, since the collinearity claim is about the fit.
2. **Break-even attainability.** The IQFormer-inspired break-even at
   SIR = -10 dB is 0.87, which exceeds the largest per-cell mean overlap
   observed in the campaign (0.84). The manuscript now says so, rather than
   presenting it as a verified crossing point.

## Numbers re-verified against artifacts in this pass

| quantity | manuscript | artifact |
|---|---:|---:|
| overlap slope vs IQFormer-insp. | 11.39 pp/unit | 11.387 |
| SIR slope vs IQFormer-insp. | -0.666 pp/dB | -0.6663 |
| envelope holdout RMSE (IQ / MC) | 6.20 / 8.05 pp | 6.196 / 8.051 |
| fit-domain constant RMSE (IQ / MC) | 8.40 / 9.63 pp | 8.4035 / 9.6326 |
| skill score (IQ / MC) | 0.46 / 0.30 | 0.4564 / 0.3014 |
| break-even (IQ: -15, -10 dB) | 0.58, 0.87 | 0.5800, 0.8726 |
| break-even (MC: -15, -10 dB) | 0.48, 0.66 | 0.4823, 0.6631 |
| cells (fit / holdout / total) | 32 / 81 / 113 | 32 / 81 / 113 |
| max Doppler at 150 / 250 km/h | 820 / 1367 Hz | 820.0 / 1366.7 |
| coherence time at 250 km/h | ~309 us | 309.5 us |

Fig. 4 recomputes the RMSE, the fit-domain constant, and the skill score from
the same arrays it plots, and prints them at build time, so the figure cannot
drift from Section V-C.
