# S5-R1 Operation Log — Independent Severe-Held Representation-Repair Validation (2026-08-20)

Evidence class: **prospective / post-hoc robustness validation** (decision
experiment). No sealed artifact, sealed number, or manuscript text was
modified. Manuscript policy per instruction §16: **no paper change yet** —
this log exists so the operator can decide whether to reposition to
"Rank Instability + Representation Repair".

## 1. Documents and rules

| Document | Status |
|---|---|
| `docs/S5_R1_PREREG.md` | FROZEN 2026-08-20, before any inference; no rule changed afterwards |
| `docs/S5_R1_CHECKPOINT_AUDIT.md` | completed before inference; 10/10 sidecar checkpoints usable |
| `docs/S5_A_SEVERE_HELD_PREREG.md` / `docs/S5_A_OPERATION_LOG.md` | S5-A archive (verdict `fail`), reused unchanged |

Seed rule: §4-A applied (all 10 seeds exist) → ten-seed primary analysis,
no retraining; rule C (retrain) was never triggered.

## 2. Execution chain

| Step | Command | Log | Outcome |
|---|---|---|---|
| Checkpoint audit | `python tmp/a20_checkpoint_audit.py` | `logs/a20_checkpoint_audit.log` | 10/10 exist, params 46,794 each, strict load OK, SHA-256 recorded in audit doc |
| Sidecar inference | `python analysis_zero_compute/a20_s5_r1_sidecar_severe_held.py predict --device cuda` | `logs/a20_predict_v1.log` | 10 seeds × 2 regimes = 20 runs completed |
| Analysis | `python analysis_zero_compute/a20_s5_r1_sidecar_severe_held.py analyze` | `logs/a20_analyze_v1.log` | verdict computed automatically per preregistration §7 |

## 3. Provenance

- **Data**: S5-A cache `standards/cache_s5_severe_held_v1` reused byte-for-byte
  (digest `e655abeebfccb8dbd7d1e9eac2642d2d128cdd82c67ff03e6ff5e9d5358c13e2`);
  no third severe-held set created; view 0 evaluation; no sample selection.
- **Sidecar checkpoints**: `artifacts/tier2_iq_sidecar_v1/models/
  tier2_h2_f2_iq_sidecar_seed{s}/model.pt`, seeds (17,…,307), frozen.
- **A5 / IQFormer / MCLDNN predictions**: reused from S5-A
  (`artifacts/s5_severe_held_v1/models/`), same seeds, same windows —
  matched-seed pairing exact.
- **New predictions**: `artifacts/s5_r1_sidecar_severe_held_v1/models/
  tier2_h2_f2_iq_sidecar_seed{s}/predictions_s5_severe_r{1,2}.npz`
  (sealed S5-A npz schema).
- **Analysis outputs**: `analysis_zero_compute/outputs/` →
  `a20_s5_r1_summary.json`, `csv/a20_s5_r1_{model_levels,paired_contrasts,
  per_snr,per_class,gap_recovery}.csv`.
- **Bootstrap**: seed-level paired bootstrap, B = 10,000, seed 20260820;
  secondary window-level `zc_core.hierarchical_paired_diff` per regime.
- **Environment**: RTX 5060 Ti 8 GB (CUDA), Python 3.12.

## 4. Results

Pooled severe-held macro-F1 (ten-seed mean): sidecar **0.2509**, A5 0.2072,
IQFormer 0.2329, MCLDNN 0.3375.

**Primary contrast (sidecar − A5):**

| Scope | Δ (pp) |
|---|---:|
| R1 | +4.22 |
| R2 | +4.52 |
| pooled (equal weight) | **+4.37** |
| 95% seed-paired CI | [+2.96, +5.90] |
| positive seeds | 10/10 |
| SNR strata with sidecar > A5 (pooled) | **8/8** |

Per-SNR pooled Δ_repair (pp): −10: +0.9; −6: +2.4; −2: +3.2; 2: +4.7;
6: +6.2; 10: +5.6; 14: +6.1; 18: +5.1 — the repair exists across all
SNR strata and is largest mid/high SNR, exactly where S5-A showed A5
collapsing relative to the I/Q references.

**Descriptive gap recovery** (pooled; defined only because the reference
beats A5): vs IQFormer R = **1.70** (sidecar overtakes IQFormer on the
severe-held set); vs MCLDNN R = 0.34 (recovers ~one third of the MCLDNN
gap; MCLDNN still leads). Baseline level per instruction §14: **Level C**
(leads one strong I/Q baseline; no uniform-superiority wording).

Per-class pooled sidecar−A5 delta (pp): BPSK **+11.4**, QPSK **+7.4**,
8PSK +7.1, GMSK **+4.8**, CPFSK +5.9, 4FSK **+3.5**, 16QAM +2.6,
64QAM −0.5, 256QAM +0.1, PI2BPSK +1.4. Repair concentrates in the
phase/constant-envelope families the representation diagnosis flagged;
64QAM is the only focus class with a (negligible) negative delta.

Complexity: sidecar = 43.1M MACs ≈ **+3% over A5** (41.8M). Headline form:
"+4.37 pp independent severe-held macro-F1 for ~3% additional MACs".

## 5. Verdict (preregistration §7, applied automatically)

All six strong-repair criteria satisfied:

1. pooled Δ = +4.37 pp > +3 pp ✓
2. CI [+2.96, +5.90] entirely > 0 ✓
3. R1 and R2 both positive ✓
4. 8/8 ≥ 6 SNR strata ✓
5. R_IQ = 1.70 ≥ 0.5 ✓
6. no severe-held fine-tuning (by construction) ✓

**Verdict: `strong_repair`.**
Wording (locked): *"Independent severe-held validation supports a
representation-repair effect that generalizes beyond the original campaign."*

## 6. Interpretation boundaries (locked)

- The result **supports** the representation-information-gap reading over
  pure capacity insufficiency (Q3); support ≠ causal proof.
- The overlap–SIR envelope remains falsified on this set (S5-A); S5-R1 does
  not resurrect it. Any condition map is descriptive only.
- Sidecar does NOT beat MCLDNN on the severe-held set; no uniform
  superiority claim is permitted.

## 7. What was NOT done (integrity notes)

- No fine-tuning, no retraining, no R1/R2 modification, no third held set,
  no sample selection by overlap, no SNR-stratum dropping, no manuscript
  change, no sealed-artifact change.
- Verdict logic was frozen in `docs/S5_R1_PREREG.md` §7 before inference
  and executed by the script without manual override.

## 8. Decision now open to the operator (instruction §16/§17)

Strong repair obtained → instruction §17 recommends repositioning the
manuscript to **"Rank Instability and Representation Repair under Structured
Interference"**, with logic chain: (1) sealed A5–A0 positive contrast →
(2) original-campaign severe rank reversal → (3) S5-A prospective falsification
of the stable-boundary reading → (4) S5-R1 strong independent repair →
(5) width-only capacity control unable to explain the boundary. Awaiting
operator decision before any manuscript edit.
