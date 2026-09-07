# S5-A Severe Held Set — Operation Log (2026-08-20)

> **Status: ARCHIVED — decision 2026-08-20: archive-only; no manuscript modification; S5-B/C/D suspended pending further instructions.**

Evidence class: **prospective / post-hoc robustness validation** (does not alter any sealed gate or sealed number).

## 1. Preregistration

- File: `docs/S5_A_SEVERE_HELD_PREREG.md` (status: FROZEN, written **before** any data generation).
- No rule in the preregistration was modified after generation began.

## 2. Execution chain

| Step | Command | Log | Outcome |
|---|---|---|---|
| Preflight | `python tmp/a19_preflight.py` | (console) | `config.validate()` OK; 12 820 sources, 7 MATLAB chunks × 2 invocations, transfer within mat-v7 limit |
| Generate | `python analysis_zero_compute/a19_s5_severe_held.py generate --matlab-batch-size 4096` | `logs/a19_generate_v1.log` | cache built; source-disjointness assertion passed |
| Predict | `python analysis_zero_compute/a19_s5_severe_held.py predict --device cuda` | `logs/a19_predict_v1.log` | 3 models × 10 seeds × 2 regimes = 60 frozen-checkpoint inference runs, all completed |
| Analyze | `python analysis_zero_compute/a19_s5_severe_held.py analyze` | `logs/a19_analyze_v1.log` | verdict computed per preregistration §9 |

## 3. Provenance

- **Cache root**: `standards/cache_s5_severe_held_v1/`
- **Cache digest**: `e655abeebfccb8dbd7d1e9eac2642d2d128cdd82c67ff03e6ff5e9d5358c13e2`
- **Evidence designation** (manifest): `s5_prospective_post_hoc_robustness_validation_v1`
- **Source disjointness vs sealed cache** (`cache_factor_headline_1024_v2`): 12 820 new source IDs vs 152 000 sealed source IDs — intersection **empty** (asserted programmatically).
- **Splits**: `s5_severe_r1` 6 400 sources (TDL-A/C/D, pulse/ofdm_like jammers, 180 km/h, SIR=−15), `s5_severe_r2` 6 400 sources (TDL-B/E, 250 km/h, SIR=−15); train/validation are 10-sample placeholders never used.
- **Frozen checkpoints**: `artifacts/tvt_v4r_headline_composite/models/{model}_seed{s}/model.pt`, seeds (17,29,43,71,101,131,173,211,257,307), no retraining.
- **Frozen envelope fit**: 32 fitting cells from `a14_envelope_cells.csv` (id_test + hard_interference), not refit.
- **Predictions**: `artifacts/s5_severe_held_v1/models/*_seed*/predictions_s5_severe_r{1,2}.npz` (sealed npz schema).
- **Analysis outputs**: `analysis_zero_compute/outputs/a19_s5_severe_held.json`, `csv/a19_severe_held_cells.csv` (64 usable cells), `csv/a19_severe_held_envelope.csv`, `csv/a19_severe_held_model_levels.csv`.
- **Analysis seed**: 20260820 (preregistered).
- **Environment**: RTX 5060 Ti 8 GB (CUDA), MATLAB R2025a, Python 3.12 (torch/numpy/pandas).

## 4. Result vs preregistered expectations

All 64/64 cells usable (row_count ≥ 150, ≥ 8 classes). Pooled cell-mean results:

| Reference | Pooled gain (pp) | Sign accuracy | RMSE full (pp) | RMSE SIR-only (pp) | RMSE constant (pp) | Strong pass? |
|---|---:|---:|---:|---:|---:|---|
| IQFormer | −3.12 | 0.578 | 9.75 | 9.05 | 8.11 | no |
| MCLDNN | −12.45 | 0.406 | 22.56 | 22.29 | 15.41 | no |

Selector audit (64 cells, sign rule from frozen 32-cell fit): recall 0.581, precision 0.563, regret vs oracle 2.76 pp — all below the preregistered strong-pass bar (recall ≥ 0.75, sign accuracy ≥ 0.75).

SNR-stratified mean gains (fraction):

| SNR (dB) | −10 | −6 | −2 | 2 | 6 | 10 | 14 | 18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vs IQFormer | +0.023 | +0.029 | +0.000 | −0.042 | −0.067 | −0.074 | −0.064 | −0.056 |
| vs MCLDNN | +0.015 | −0.023 | −0.106 | −0.150 | −0.180 | −0.196 | −0.177 | −0.179 |

A5 retains a small advantage only in the two lowest-SNR strata against IQFormer; the advantage does **not** survive pooling and does not exist at all against MCLDNN beyond −10 dB.

## 5. Pass/fail reading (preregistration §9 wording)

**Overall verdict: `fail`.**

- The A5 advantage observed in the sealed hard_interference split does **not** transfer to the independent −15 dB severe held set: pooled gains are negative against both references, in both regimes (r1: −3.26 / −12.39 pp; r2: −2.98 / −12.51 pp).
- The frozen overlap–SIR envelope adds no predictive value on the severe set: the constant predictor beats it in RMSE for IQFormer, and sign accuracy is near/below chance for both references.
- The selector recall (0.58) is far below the preregistered 0.75 bar.

## 6. Consequence per the S5 roadmap

Instruction §7 Go/No-Go and §17 apply: the severe advantage has disappeared on an independent held set, so the current severe result is a simulator-regime-specific phenomenon. Per §17: do **not** loop additional held-set variants until the result improves. Recommended path: (1) demote the Operating Envelope to diagnostic status, (2) reposition the manuscript (rank instability / failure diagnosis / sidecar repair), (3) reassess target venue. Awaiting operator decision; S5-B/S5-C/S5-D are **not** started.

## 7. What was NOT done (integrity notes)

- No retraining, no envelope refit, no held-set redefinition, no proxy-feature search, no sealed-artifact modification.
- No result-dependent rule change; the verdict logic in `a19_s5_severe_held.py` was fixed before analysis ran.
