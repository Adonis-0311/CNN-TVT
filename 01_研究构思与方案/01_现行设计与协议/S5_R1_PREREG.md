# S5-R1 Pre-registration: Independent Severe-Held Representation-Repair Validation

**Status:** FROZEN — no rule below may be changed after sidecar inference starts.
**Evidence class:** prospective/post-hoc robustness validation. Does NOT
retroactively alter the sealed confirmatory family, the S5-A archive, or any
sealed artifact.
**Date frozen:** 2026-08-20
**Instruction source:** `TVT_S5_R1_独立SevereHeld表示修复验证方案.md`.

## 1. Scientific question

Does the previously trained received-I/Q sidecar (A5 + lightweight raw-I/Q
temporal branch, inference input = received mixture only) repair the A5
collapse that S5-A exposed on the independent −15 dB severe-held regimes?

Primary endpoint (fixed): Δ_repair = F1_sidecar − F1_A5 (macro-F1), reported
for R1, R2, and pooled equal-weight. This is NOT sidecar-vs-IQFormer.

## 2. Data (locked, reused from S5-A)

- Cache: `standards/cache_s5_severe_held_v1`, digest
  `e655abeebfccb8dbd7d1e9eac2642d2d128cdd82c67ff03e6ff5e9d5358c13e2`.
  No regeneration, no third severe-held set.
- R1 = `s5_severe_r1` (6400 sources, TDL-A/C/D, pulse/ofdm_like, 180 km/h,
  SIR −15); R2 = `s5_severe_r2` (6400 sources, TDL-B/E, 250 km/h, SIR −15).
- Evaluation uses view 0 only (sealed per-source convention), identical to
  S5-A. No sample removal, no overlap-based selection.

## 3. Models and checkpoints (locked)

- **Sidecar**: `tier2_h2_f2_iq_sidecar`, 10 seeds
  (17, 29, 43, 71, 101, 131, 173, 211, 257, 307) from
  `artifacts/tier2_iq_sidecar_v1/models/`. All 10 exist → **Rule §4-A:
  ten-seed primary analysis** (no retraining).
- **A5 / IQFormer-inspired / MCLDNN**: reuse the S5-A frozen-checkpoint
  predictions at `artifacts/s5_severe_held_v1/models/` (same seeds, same
  cache, same view-0 windows). No re-inference of these three models.
- Architecture identity: `LightweightIQSidecarVIMD` in
  `analysis_zero_compute/tier2_gpu/run_tier2_experiment.py`; total params
  46,794 = 39,500 (A5) + 7,294 (sidecar increment), per Supporting Table IX;
  verified by checkpoint audit before inference (SHA-256 + state_dict count).
- No severe-held data ever entered sidecar training or tuning; no
  fine-tuning now.

## 4. Inference protocol

- Output: `artifacts/s5_r1_sidecar_severe_held_v1/models/
  tier2_h2_f2_iq_sidecar_seed<seed>/predictions_<regime>.npz`, sealed S5-A
  field schema (probabilities, labels, source_ids, snr_db, sir_db,
  target_profile_index, cache_digest, split).
- Deterministic: `torch.no_grad()`, eval mode, batch 512, device cuda.

## 5. Metrics (locked)

- Macro-F1: `zc_core.macro_f1` on argmax predictions, per seed; reporting
  uses the ten-seed mean.
- **Primary contrast** Δ_repair per scope ∈ {R1, R2, pooled}: pooled is the
  equal-weight mean of the R1 and R2 per-seed differences.
- **Paired statistics** (matched seeds): seed-level paired bootstrap over the
  10 matched per-seed differences, B = 10,000 draws, seed 20260820,
  percentile 95% CI; plus per-seed signs. As a secondary window-level audit,
  `zc_core.hierarchical_paired_diff` per regime.
- **Secondary contrasts**: sidecar − IQFormer, sidecar − MCLDNN (same
  protocol). Not double-counted as independent evidence.
- **Per-SNR**: 8 strata (−10…18 dB) × {R1, R2, pooled}, seed-mean macro-F1
  per model, sidecar−A5 column.
- **Per-class**: per-class F1 from the confusion matrix, seed-mean,
  sidecar − A5 delta, pooled and per regime; focus classes QPSK, 16QAM,
  64QAM, BPSK, GMSK, 4FSK reported explicitly.
- **Descriptive gap-recovery ratio** (only when the reference beats A5):
  R_b = (F1_sidecar − F1_A5) / |F1_b − F1_A5|, pooled. Descriptive only —
  never a causal effect size.

## 6. Complexity reading (locked)

A5 = 41.8M MACs, sidecar = 43.1M MACs, IQFormer = 355.6M, MCLDNN = 398.2M.
If repair holds, headline form: "+X pp severe-held macro-F1 for ~3%
additional MACs over A5". No new efficiency metric.

## 7. Success criteria (locked, per instruction §13)

Let pooled Δ = pooled sidecar−A5 (pp), CI = its 95% bootstrap CI, and
R_IQ = pooled gap recovery vs IQFormer.

- **Strong repair**: pooled Δ > +3 pp AND CI entirely > 0 AND R1 > 0 AND
  R2 > 0 AND sidecar > A5 in ≥ 6 of 8 SNR strata (pooled) AND R_IQ ≥ 0.5
  AND no severe-held fine-tuning (true by construction).
- **Moderate repair**: pooled Δ > +1.5 pp AND CI > 0 AND R1, R2 > 0 AND
  0.25 ≤ R_IQ < 0.5.
- **Weak / unresolved**: pooled Δ > 0 but CI crosses 0, or R1/R2 disagree.
- **Fail**: pooled Δ ≤ 0, or R1/R2 strongly inconsistent with weak pooled,
  or CI entirely ≤ 0.

Baseline level (instruction §14, descriptive): Level A (repair but below
strong baselines), B (statistically near IQFormer), C (leads ≥ 1 strong
baseline) — no uniform-superiority wording at any level.

## 8. Prohibited actions (locked)

- Fine-tuning the sidecar on severe-held data; modifying R1/R2; picking a
  single regime for the headline; dropping unfavorable SNR strata;
  overlap-based sample selection; creating a third held set; framing the
  result as retroactive proof that the original A5 design was right;
  presenting post-hoc validation as sealed evidence.
- The overlap–SIR envelope is NOT an S5-R1 metric; any condition map is
  descriptive post-hoc visualization only.

## 9. Manuscript policy

No manuscript change until the verdict is produced and a decision is made
(instruction §16). S5-R1 is a decision experiment.

## 10. Provenance and outputs

- Script: `analysis_zero_compute/a20_s5_r1_sidecar_severe_held.py`
  (predict + analyze).
- Outputs: `a20_s5_r1_model_levels.csv`, `a20_s5_r1_paired_contrasts.csv`,
  `a20_s5_r1_per_snr.csv`, `a20_s5_r1_per_class.csv`,
  `a20_s5_r1_gap_recovery.csv`, `a20_s5_r1_summary.json` in
  `analysis_zero_compute/outputs` (+ `csv/`).
- Documents: this preregistration, `S5_R1_CHECKPOINT_AUDIT.md`,
  `S5_R1_OPERATION_LOG.md`.
- Analysis/bootstrap seed: 20260820.
