# Retrained Permanently-Zeroed-I/Q Sidecar — Preregistration

**Status:** FROZEN — no rule below may be changed after training starts.
**Evidence class:** exploratory retrained ablation (post-campaign; below the
sealed family; cannot alter any sealed, transfer, or repair verdict already
recorded).
**Date frozen:** 2026-08-21
**Instruction source:** `TVT_V4_9_评分与AI痕迹自检.md` §2.1 second batch —
"a sidecar variant retrained on the frozen cache with the I/Q input
permanently zeroed is the decisive follow-up" (adopted in V4.10 supp §XV).

## 1. Scientific question

The pooled severe-held repair (sidecar − A5, +4.37 pp over ten seeds;
matched-five-seed subset reported alongside) admits an attribution
ambiguity: the inference-time lesion B1 (I/Q input zeroed at inference of
the jointly trained sidecar) removes 12.28 pp, 2.81× the repair, but a
lesion of a jointly trained network is off its training distribution and
measures sensitivity, not information contribution. Retraining the same
architecture with the I/Q input permanently zeroed is the only design that
keeps capacity, fusion structure, optimizer, and data identical while
removing the I/Q information channel from training onward.

## 2. Variant definition (locked)

- Model: `tier2_h2_f2_iq_sidecar_iqzero` — `LightweightIQSidecarVIMD`
  architecture unchanged (total 46,794 parameters); in `forward`, the I/Q
  branch receives `torch.zeros_like(values)` at training AND inference.
  The spectral path, teacher, objectives, loss weights, and fusion/classifier
  head are identical to `tier2_h2_f2_iq_sidecar`.
- Training data: sealed cache `standards/cache_factor_headline_1024_v2`
  (digest `f2003d4bfb0895ed8c883c6432b82345999be64df3d9834d2bd9451dc0697d80`),
  train/validation splits only, wrapped by the same deterministic 10%
  counterfactual clean re-mix (selection seed 20260812) as the original
  sidecar run.
- Optimizer/schedule: identical to `artifacts/tier2_iq_sidecar_v1` —
  30 epochs max, batch 16, lr 3e-4, weight decay 0.01, patience 8, AMP,
  gradient clip 5.0, same mask/contrastive ramps.
- Seeds: the matched five (17, 29, 43, 71, 101) — the same subset used by
  Control A capacity tiers.
- No hyperparameter change, no architecture change, no data change relative
  to the original sidecar arm. Run id: `tier2_iq_sidecar_iqzero_v1`.

## 3. Evaluation (locked)

- Held data: `standards/cache_s5_severe_held_v1` (regimes `s5_severe_r1`,
  `s5_severe_r2`; SIR −15 dB; view 0 only; sealed S5-A npz schema). The
  severe-held cache never enters training, tuning, or early stopping.
- Frozen-checkpoint inference only after training completes; no fine-tuning
  on held data.
- Primary contrast: Δ_zero = F1_iqzero − F1_A5 (pooled equal-weight over R1
  and R2; matched five seeds; seed-paired bootstrap, B = 10,000, seed
  20260821). Secondary: F1_iqzero vs the matched-five-seed full sidecar
  level, and per-regime values.

## 4. Decision rule (locked, applied automatically)

Let Δ_zero be the pooled iqzero-versus-A5 contrast and Δ_full the
matched-five-seed pooled full-sidecar-versus-A5 contrast on the same regimes:

1. **Attribution supported** if the 95% seed-paired CI upper bound of
   Δ_zero < +1.0 pp (the retrained structure without I/Q information does
   not reproduce the repair).
2. **Attribution refuted** if the Δ_zero point estimate ≥ +3.0 pp (the
   repair reproduces without the I/Q information channel; it is then a
   capacity/structure/joint-training effect).
3. **Unresolved** otherwise (report both numbers and leave the attribution
   open).

## 5. Reporting constraints (locked)

- Whichever branch fires, the wording states the contrast, CI, and seed
  count; no claim extends beyond the −15 dB severe-held regimes.
- This experiment does not reopen, modify, or reweight any sealed artifact,
  the S5-A archive, the S5-R1 repair verdict, or the S6 control verdicts.
- Results enter the manuscript as exploratory retrained-ablation evidence
  with its own artifact path and digest.
