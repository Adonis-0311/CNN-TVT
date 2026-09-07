# S5-R1 Checkpoint Provenance Audit (2026-08-20)

Scope: verify the frozen received-I/Q sidecar checkpoints before any S5-R1
inference. No training, no fine-tuning, no data generation.
Audit script: `tmp/a20_checkpoint_audit.py`; raw log:
`logs/a20_checkpoint_audit.log`.

## 1. Model definition identity

- Class: `LightweightIQSidecarVIMD`
  (`analysis_zero_compute/tier2_gpu/run_tier2_experiment.py`), wrapping the
  sealed A5 `VIMDNet` with a lightweight raw-I/Q temporal branch.
- Instantiated with the frozen scientific invariants from
  `tvt_submission/configs/formal_tvt_recovery_v4r_110plus10.json`
  (same ModelConfig as S5-A inference).
- Live parameter count: **total 46,794 = base 39,500 + sidecar increment
  7,294** — matches Supporting Table IX exactly (layer-by-layer sum:
  iq_branch 1,404 + iq_projector 1,800 + fusion 3,600 + classifier 490).
- Inference consumes the received two-channel mixture only
  (`forward(values)` with `values` = [batch, 2, 1024] I/Q); no component
  bookkeeping at inference. Side-head logits are residual-added to the base
  spectral logits.

## 2. Checkpoint table

All checkpoints: `artifacts/tier2_iq_sidecar_v1/models/tier2_h2_f2_iq_sidecar_seed<seed>/model.pt`.
Every checkpoint loads with `load_state_dict(strict=True)` and contains
46,794 trainable parameters plus 64 buffer elements (the deterministic STFT
window `base.front_end.window`, not a learned parameter).

| Seed | Exists | SHA-256 (model.pt) | Params | Strict load | Usable |
|---|---|---|---:|---|---|
| 17 | yes | `baf0fb93ff38adc756b712bec7d5d22f10772b03b3af4149586088301ef563bb` | 46,794 | yes | yes |
| 29 | yes | `2b4739d709e2da1f23ba7b059ffb3890c93dcc8eb6bc8758674c898cc403c9cb` | 46,794 | yes | yes |
| 43 | yes | `19418f5c7d7354630aabead3f900ff6b8e7f36af8f3405d4e2ec54a3d67a7a0c` | 46,794 | yes | yes |
| 71 | yes | `6ff49eaa4057ffad6948ccc295ac49e56d4a3ef08ff7946b7567be3a5b712c11` | 46,794 | yes | yes |
| 101 | yes | `97e93fbf2d5b91c09c180801efa17927199e76add865888bc95ff148999b07b3` | 46,794 | yes | yes |
| 131 | yes | `13fdf74ee498309f4bf8b9d7db545dc5ebf532b98de37b70c7d52bc0039d2de2` | 46,794 | yes | yes |
| 173 | yes | `bd85c90d9a9af273894330c2f44606473d91fce9a8c031047940f099ac9a80e1` | 46,794 | yes | yes |
| 211 | yes | `96d8a9c4771929bec38ec7a21655f82b39a500d2547e446911df07187c2c7cb5` | 46,794 | yes | yes |
| 257 | yes | `8dd59ae0198232412c5c051320705d198217af0947995eb74f03a49abc04af5b` | 46,794 | yes | yes |
| 307 | yes | `089d96decfd4b3a31d4d17bf3b0f5f1a9228c0b290458ad5c43429c16ba5bb7f` | 46,794 | yes | yes |

**Usable: 10/10** → instruction seed rule **§4-A** applies: ten-seed primary
analysis, no retraining (rule C is not triggered).

## 3. Held-set isolation

- The severe-held cache `standards/cache_s5_severe_held_v1` was generated on
  2026-08-20, strictly after the sidecar campaign
  (`artifacts/tier2_iq_sidecar_v1`) completed; its source IDs are disjoint
  from every sealed split (asserted in the S5-A operation log).
- Sidecar training used only the sealed cache train/validation splits
  (with the deterministic 10% counterfactual clean re-mix wrapper); no
  severe-held windows participated in training, tuning, or early stopping.
- S5-R1 performs frozen inference only; no fine-tuning will occur.

## 4. Note on the earlier "46,858" count

A first audit pass counted every state-dict tensor and reported 46,858.
The extra 64 elements are the non-learnable STFT window buffer
(`base.front_end.window`); counting named parameters only yields 46,794 for
all ten checkpoints. This is a counting-convention correction, not a
checkpoint change.
