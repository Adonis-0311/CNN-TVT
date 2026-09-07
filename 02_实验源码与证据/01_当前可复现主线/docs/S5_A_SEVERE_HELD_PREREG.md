# S5-A Pre-registration: Independent Severe-Held Validation at SIR = −15 dB

**Status:** FROZEN — no rule below may be changed after data generation starts.
**Evidence class:** prospective/post-hoc robustness validation. It does NOT
retroactively alter the sealed confirmatory family or the historical identity
of any sealed artifact.
**Date frozen:** 2026-08-20
**Instruction source:** `TVT_S5_冲击92分与70收稿率_深化推进路线.md` §3 (S5-A2 enhanced version).

## 1. Scientific question

Does the A5 rank exchange at SIR = −15 dB survive on a severe regime that
never participated in the Eq.(8) envelope fitting (32 fitting cells from
`id_test` + `hard_interference`)?

## 2. Two held regimes (S5-A2)

| Factor | Regime 1 (R1) | Regime 2 (R2) |
|---|---|---|
| SIR | −15 dB fixed | −15 dB fixed |
| Jammer families | pulse, ofdm_like (held-out) | pulse, ofdm_like (held-out) |
| TDL profiles | TDL-A/C/D (seen) | TDL-B/E (held-out channel) |
| Speed | 180 km/h (held-out) | 250 km/h (held-out) |
| SNR grid | −10, −6, −2, 2, 6, 10, 14, 18 dB | same |
| Modulation classes | all 10 taxonomy classes | same |

R1 combines held jammer family + held speed. R2 additionally holds the TDL
profile (combined-OOD factor support). Both factor supports were held out of
training in the sealed campaign, but no sealed split evaluates SIR = −15 dB
under them; the SIR = −15 stratum itself is the new severe condition.

## 3. Sample counts and source identity

- 6400 sources per regime (≥ hard_interference's 5000), 2 views per source;
  evaluation uses view 0 only (5000-window analogue: 6400 evaluated windows
  per regime), matching the sealed per-source evaluation convention.
- Builder: `vimd_amc.standards.cache.build_tdl_paired_cache` with the sealed
  headline domain parameters: `sample_length=1024`, `guard_samples=96`,
  `master_seed=20260727`, delay spreads (30, 100, 300) ns, carrier 5.9 GHz,
  sample rate 1 MHz, 10 modulation classes.
- Source keys: `factor_isolated_s5::s5_severe_r1` and
  `factor_isolated_s5::s5_severe_r2`. Source IDs derive from
  `stable_seed("factor_cache_source", master_seed, source_key, index)`,
  so disjointness from every sealed split is guaranteed by construction and
  will be asserted numerically against the sealed source-ID manifests.
- Overlap is computed post-generation as the audit variable
  (`jammer_to_signal_overlap`) and is NEVER used to select or reject samples
  (jammer pre-channel power retries are the only generation-time filter, as
  in the sealed builder).
- Cache destination: `standards/cache_s5_severe_held_v1` (new directory;
  the sealed cache is read-only).

## 4. Inference

- Models: `a5_vimd_full`, `mcldnn_reimplementation`, `iqformer_inspired`.
- All 10 sealed algorithm seeds (17, 29, 43, 71, 101, 131, 173, 211, 257, 307);
  sealed `model.pt` checkpoints loaded unchanged; NO retraining.
- Output: `artifacts/s5_severe_held_v1/models/<model>_seed<seed>/
  predictions_<regime>.npz` with the sealed field schema
  (probabilities, labels, source_ids, snr_db, sir_db, target_profile_index).

## 5. Cell construction (locked)

- Cells = SNR stratum × overlap quartile within each regime:
  8 × 4 = 32 cells per regime, 64 cells total.
- Quartiles are the empirical overlap quartiles of the regime itself
  (np.quantile at 0/.25/.5/.75/1, unique'd; last bin inclusive).
- Minimum windows per cell: 150 (cells below the floor are reported but
  excluded from envelope RMSE, matching a14's MIN_ROWS).

## 6. Envelope evaluation (fixed fit, no refitting)

Coefficients are the frozen 32-cell primary fit, re-derived deterministically
from `analysis_zero_compute/outputs/csv/a14_envelope_cells.csv` restricted to
`split ∈ {id_test, hard_interference}`, design `[1, occupancy_mean, sir_db]`,
OLS, per reference (IQFormer-inspired, MCLDNN):

- **full:** β·[1, o, SIR] (the paper's primary two-variable envelope);
- **SIR-only:** OLS of `[1, SIR]` on the same 32 fitting cells;
- **constant:** the mean gain over the same 32 fitting cells.

All three predictors are evaluated unchanged on the 64 new cells.
Metrics per reference: RMSE, MAE, sign accuracy (pp units), plus the
constant/SIR-only baseline RMSEs for comparison.

## 7. Model-level metrics

Per regime and pooled: per-model macro-F1 (ten-seed mean, zc_core estimator),
A5−IQFormer and A5−MCLDNN differences with paired seed bootstrap CIs.

## 8. Selector audit (locked)

Sign rule on the frozen envelope: select A5 where predicted gain > 0.
Against the hindsight-optimal cell indicator (observed ten-seed mean gain > 0):
TP/FP/FN/TN, recall, precision, realized pooled macro-F1 of the selector
policy vs oracle vs always-IQFormer, oracle regret, expected MACs and MAC
saving (MACs from the sealed composite `result.json` complexity records).

## 9. Pass/fail interpretation (locked, per instruction §3.7)

For EACH reference (IQFormer-inspired and MCLDNN), on the pooled 64 cells:

- **Strong pass** (both references): sign accuracy ≥ 75% AND envelope RMSE <
  SIR-only RMSE AND envelope RMSE < constant RMSE AND pooled severe A5 gain
  against that reference > 0 AND selector recall ≥ 75%.
  → wording: "independent severe-condition transport supports the
  rank-exchange boundary."
- **Partial pass** (MCLDNN satisfies all; IQFormer direction roughly right
  but RMSE not better than SIR-only): → "reference-dependent severe
  transport" with the per-reference breakdown stated.
- **Fail** (IQFormer shows systematic sign errors, e.g. sign accuracy < 50%):
  → the severe A5 advantage remains an observed post-hoc phenomenon, not a
  cross-regime predictable boundary. Reported honestly; no re-tuning of the
  held set to repair the result.

Per-regime results are reported separately as well; the pooled verdict uses
both regimes with equal cell weight.

## 10. Prohibited actions (locked)

- Redefining the held set, quartiles, or metrics after seeing results.
- Refitting the envelope on or with the new cells.
- Using overlap or any oracle quantity to select samples.
- Presenting S5-A results as sealed confirmatory evidence.

## 11. Analysis seed and provenance

- Analysis/bootstrap seed: 20260820 (matches a18).
- Scripts: `analysis_zero_compute/a19_s5_severe_held.py` (build + inference +
  analysis), outputs `a19_*` CSVs/JSON in `analysis_zero_compute/outputs`.
- Operation log: `docs/S5_A_OPERATION_LOG.md` after completion.
