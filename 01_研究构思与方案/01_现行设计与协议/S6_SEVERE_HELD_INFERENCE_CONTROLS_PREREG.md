# Severe-Held Frozen-Inference Attribution Controls — Preregistration

Status: **FROZEN before any control inference** (2026-08-20).
Scope: three forward-only attribution controls on the already locked
independent severe-held cache `standards/cache_s5_severe_held_v1/`.
Evidence class: **post-campaign diagnostic control** (below the sealed
family; cannot change any sealed conclusion and cannot change the
preregistered transfer/repair verdicts already recorded).

## 1. Motivation

The independent severe-held repair contrast (sidecar minus A5, pooled
+4.37 pp, [2.96, 5.90]) admits competing explanations: (i) additional
received-I/Q information, (ii) additional capacity, (iii) the added
fusion/classifier path. These controls constrain those explanations at
inference only.

## 2. Locked inputs (nothing is regenerated)

- Data: `standards/cache_s5_severe_held_v1/` (regimes `s5_severe_r1`,
  `s5_severe_r2`; SIR −15 dB; 8-SNR grid; source-disjoint; cache digest
  recorded in the manifest).
- Checkpoints: capacity tiers M (`artifacts/tier2_capacity_M_v2/models/`)
  and L (`artifacts/tier2_capacity_L_v1/models/`), five matched seeds
  (17, 29, 43, 71, 101); sidecar
  (`artifacts/tier2_iq_sidecar_v1/models/`), ten seeds. All frozen, no
  retraining, no tuning, no checkpoint reselection.
- Reused predictions: A5 / IQFormer-inspired / MCLDNN from
  `artifacts/s5_severe_held_v1/`, sidecar from
  `artifacts/s5_r1_sidecar_severe_held_v1/` (control conditions are
  compared against the existing full sidecar predictions; the full
  sidecar pass is not re-run).

## 3. Controls

### Control A — tested spectral width scaling on the same held set

Forward the M and L checkpoints on both regimes. Matched-five-seed
levels for A5 and the sidecar are restricted to the same five seeds for
this comparison. Primary quantity: pooled seed-mean macro-F1 per tier
and the paired contrasts sidecar − M and sidecar − L (per-seed paired
differences, pooled as the equally weighted regime mean, seed bootstrap
10,000 draws, seed 20260820).

### Control B — received-I/Q information dependence (inference interventions)

Applied to the frozen sidecar checkpoints (ten seeds, both regimes):

- **B1 iq_zero**: the I/Q branch input is replaced by zeros of the same
  shape; every other weight and the spectral path are unchanged.
- **B2 iq_shuffle**: a fixed random temporal permutation of the 1024
  samples, identical for both channels and drawn once per window from
  `np.random.default_rng(20260821)` indexed by window position; the
  marginal sample distribution is preserved while modulation temporal
  structure is destroyed.
- B3 (magnitude-only) is declared optional and **not executed**; the
  two-channel convolutional branch has no canonical magnitude-only
  input, and B1/B2 already bound the information question.

These are inference interventions, i.e. an information-dependence
diagnostic, not an architecture ablation and not causal proof.

### Control C — added classifier path

**C1 no_side_head**: the sidecar's final logits are the base spectral
path logits only (the residual side-head contribution
`classifier(fused_embedding)` is removed). The sidecar base was trained
jointly with the branch, so this isolates how much of the repair flows
through the added head versus the co-trained base.

## 4. Metric conventions

- Per-seed macro-F1 per regime; pooled = equally weighted mean of the
  two regimes, then seed mean; paired contrasts use per-seed paired
  differences with percentile bootstrap over seeds (10,000 draws,
  seed 20260820).
- Control A additionally reports matched-five-seed versions of A5 and
  sidecar so that every within-table comparison shares the seed set.
- All controls report R1, R2, and pooled scopes.

## 5. Frozen decision patterns (no post-hoc redefinition)

Let `S` = full sidecar, `L` = widest capacity tier, `Z` = B1,
`H` = B2, `N` = C1, and measure all quantities as pooled seed-mean
macro-F1 or the paired repair contrast versus A5:

1. **Strong representation-supportive** if all hold:
   (a) `S > L` on pooled severe-held macro-F1 with positive paired
   contrast sidecar − L in point estimate;
   (b) `S > Z` and `S > H` with `S − Z ≥ 1.0 pp` and `S − H ≥ 1.0 pp`;
   (c) `N` retains at least half of the full repair contrast
   (`N − A5 ≥ 0.5 × (S − A5)`).
2. **Mixed (joint representation-and-fusion)** if (a) and (b) hold but
   (c) fails.
3. **Capacity-dominated** if (a) fails because `L ≥ S` in point
   estimate: representation attribution is downgraded; the contribution
   is restated as a low-cost augmented receiver path.
4. **Path-dominated** if (b) fails because `Z ≥ S − 1.0 pp`: the repair
   is no longer described as information-preserving; it is restated as a
   lightweight sidecar repair.

Patterns 1–4 are evaluated in this order; the first whose conditions
hold is reported. Thresholds (1.0 pp, half) are fixed here before any
control inference.

## 6. Interpretation language (locked)

- Pattern 1 → "the severe-held controls support a representation-
  information interpretation: width scaling does not reproduce the
  repair, destroying received-I/Q information removes a substantial
  fraction of it, and the improvement is not explained solely by the
  auxiliary logit path."
- Pattern 2 → "the repair is best interpreted as a joint
  representation-and-fusion intervention rather than an isolated
  I/Q-information effect."
- Pattern 3 → capacity-language only; no representation-information
  claim for the repair.
- Pattern 4 → "lightweight sidecar repair" without information-
  preservation language.

## 7. Prohibitions

No retraining; no new simulation; no new held set; no refitting of the
envelope; no checkpoint reselection on held data; no changing the
sealed family or the two earlier severe-held verdicts; no selective
regime/SNR reporting — R1, R2, pooled, and per-SNR summaries are all
written to CSV regardless of direction.

## 8. Outputs

- Predictions: `artifacts/s6_severe_held_controls_v1/` (per control,
  per seed, per regime npz with cache digest).
- Analysis: `analysis_zero_compute/a21_s6_severe_held_controls.py`,
  writing `a21_*` CSVs and `a21_s6_controls_summary.json` under
  `analysis_zero_compute/outputs/`.
- Evidence class recorded in every output: `post_campaign_diagnostic_control`.
