"""A22 -- zero-compute decomposition audit of the severe-held set.

Post-review diagnostic (TVT V4.9 review, sections 2.1/2.2) computed entirely
from frozen predictions and the frozen cache manifest; no inference, no
retraining, no new simulation. It reports:

* seed-mean pooled macro-F1 per jammer family (pulse / ofdm_like) for the
  four frozen models, pooled and per regime;
* the per-regime model levels restated next to the campaign hard-split
  SIR = -15 dB levels for the level-drift reading;
* the inference-lesion loss/repair ratios of the S6 controls, with the
  consistent definition ratio = (full - condition) / (full - A5).

Outputs:
    analysis_zero_compute/outputs/a22_severe_held_decomposition.json
    analysis_zero_compute/outputs/csv/a22_per_jammer_levels.csv
    analysis_zero_compute/outputs/csv/a22_regime_drift.csv
    analysis_zero_compute/outputs/csv/a22_lesion_ratios.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import zc_core as z  # noqa: E402

ROOT = REPO
CACHE_ROOT = ROOT / "standards" / "cache_s5_severe_held_v1"
S5A_ARTIFACTS = ROOT / "artifacts" / "s5_severe_held_v1"
SIDECAR_ARTIFACTS = ROOT / "artifacts" / "s5_r1_sidecar_severe_held_v1"
A21_SUMMARY = ROOT / "analysis_zero_compute" / "outputs" / "a21_s6_controls_summary.json"
OUT = ROOT / "analysis_zero_compute" / "outputs"
CSV = OUT / "csv"

REGIMES = ("s5_severe_r1", "s5_severe_r2")
SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
SIDECAR = "tier2_h2_f2_iq_sidecar"
MODELS = ("a5_vimd_full", "iqformer_inspired", "mcldnn_reimplementation", SIDECAR)
SHORT = {
    "a5_vimd_full": "A5",
    "iqformer_inspired": "IQFormer",
    "mcldnn_reimplementation": "MCLDNN",
    SIDECAR: "sidecar",
}
# Campaign hard-split SIR = -15 dB five-seed macro-F1 (supplement Table IV).
CAMPAIGN_SIR_MINUS15 = {"A5": 0.353, "IQFormer": 0.262, "MCLDNN": 0.255}


def _pred_path(model: str, seed: int, regime: str) -> Path:
    root = SIDECAR_ARTIFACTS if model == SIDECAR else S5A_ARTIFACTS
    return root / "models" / f"{model}_seed{seed}" / f"predictions_{regime}.npz"


def _jammer_family(manifest: dict, regime: str) -> np.ndarray:
    """Per-source jammer family of view 0, ordered as the cache rows.

    The predictions evaluate view 0 only (cache arrays are indexed
    ``[:, 0]`` upstream), and both views of a source share the same
    jammer family; we still assert that view 0 carries interference.
    """
    families = []
    for record in manifest["records"][regime]:
        view = record["views"][0]
        assert view.get("interference_present"), "view 0 must be an active-jammer view"
        families.append(view["jammer_name"])
    return np.asarray(families)


def main() -> None:
    manifest = json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))

    family: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    preds: dict[str, dict[str, np.ndarray]] = {regime: {} for regime in REGIMES}

    for regime in REGIMES:
        family[regime] = _jammer_family(manifest, regime)
        assert len(family[regime]) == 6400
        assert set(family[regime].tolist()) == {"pulse", "ofdm_like"}
        for model in MODELS:
            per_seed = []
            for seed in SEEDS:
                data = np.load(_pred_path(model, seed, regime))
                if labels.get(regime) is None:
                    labels[regime] = data["labels"].astype(np.int64)
                    cache_label = np.load(CACHE_ROOT / regime / "label.npy")
                    cache_label = cache_label[:, 0] if cache_label.ndim >= 2 else cache_label
                    assert np.array_equal(labels[regime], cache_label.astype(np.int64)), \
                        "prediction/cache row order mismatch"
                    cache_source = np.load(CACHE_ROOT / regime / "source_id.npy")
                    cache_source = cache_source[:, 0] if cache_source.ndim >= 2 else cache_source
                    manifest_ids = np.asarray(
                        [r["source_sequence_id"] for r in manifest["records"][regime]],
                        dtype=np.int64,
                    )
                    assert np.array_equal(data["source_ids"].astype(np.int64), cache_source)
                    assert np.array_equal(cache_source, manifest_ids), \
                        "manifest/cache source order mismatch"
                per_seed.append(np.argmax(data["probabilities"], axis=-1))
            preds[regime][model] = np.stack(per_seed)

    # -- per-jammer-family seed-mean pooled macro-F1 -----------------------
    fam_rows = []
    fam_levels: dict[str, dict[str, float]] = {}
    for fam in ("pulse", "ofdm_like"):
        fam_levels[fam] = {}
        for model in MODELS:
            per_seed_pooled = []
            for s in range(len(SEEDS)):
                per_regime = []
                for regime in REGIMES:
                    mask = family[regime] == fam
                    per_regime.append(
                        z.macro_f1(labels[regime][mask], preds[regime][model][s][mask])
                    )
                per_seed_pooled.append(float(np.mean(per_regime)))
            fam_levels[fam][SHORT[model]] = float(np.mean(per_seed_pooled))
            fam_rows.append({
                "scope": "pooled", "jammer_family": fam, "model": SHORT[model],
                "macro_f1": float(np.mean(per_seed_pooled)),
                "macro_f1_seed_std": float(np.std(per_seed_pooled, ddof=1)),
            })
        # per-regime levels for the same family
        for regime in REGIMES:
            for model in MODELS:
                mask = family[regime] == fam
                values = [
                    z.macro_f1(labels[regime][mask], preds[regime][model][s][mask])
                    for s in range(len(SEEDS))
                ]
                fam_rows.append({
                    "scope": regime, "jammer_family": fam, "model": SHORT[model],
                    "macro_f1": float(np.mean(values)),
                    "macro_f1_seed_std": float(np.std(values, ddof=1)),
                })

    # -- regime drift table --------------------------------------------------
    drift_rows = []
    severe_levels: dict[str, dict[str, float]] = {"r1": {}, "r2": {}, "pooled": {}}
    for model in MODELS:
        per_seed_regime = {
            regime: np.array([
                z.macro_f1(labels[regime], preds[regime][model][s])
                for s in range(len(SEEDS))
            ])
            for regime in REGIMES
        }
        severe_levels["r1"][SHORT[model]] = float(per_seed_regime[REGIMES[0]].mean())
        severe_levels["r2"][SHORT[model]] = float(per_seed_regime[REGIMES[1]].mean())
        pooled = 0.5 * (per_seed_regime[REGIMES[0]] + per_seed_regime[REGIMES[1]])
        severe_levels["pooled"][SHORT[model]] = float(pooled.mean())
        for scope, value in (("s5_severe_r1", severe_levels["r1"][SHORT[model]]),
                             ("s5_severe_r2", severe_levels["r2"][SHORT[model]]),
                             ("pooled", severe_levels["pooled"][SHORT[model]])):
            campaign = CAMPAIGN_SIR_MINUS15.get(SHORT[model])
            drift_rows.append({
                "scope": scope, "model": SHORT[model],
                "severe_held_macro_f1": value,
                "campaign_hard_sir_minus15_macro_f1": campaign,
                "drift_pp": None if campaign is None else 100 * (value - campaign),
            })

    # -- lesion loss/repair ratios ------------------------------------------
    a21 = json.loads(A21_SUMMARY.read_text(encoding="utf-8"))
    levels = a21["pooled_seed_mean_macro_f1"]
    full = levels["S"]
    a5 = levels["A5"]
    repair_pp = 100 * (full - a5)
    ratio_rows = []
    for condition, key in (("iq_zero", "sidecar_iq_zero"), ("iq_shuffle", "sidecar_iq_shuffle"),
                           ("no_side_head", "sidecar_no_side_head")):
        value = levels[key]
        loss_pp = 100 * (full - value)
        ratio_rows.append({
            "condition": condition,
            "level_pct": 100 * value,
            "loss_vs_full_sidecar_pp": loss_pp,
            "vs_a5_pp": 100 * (value - a5),
            "loss_over_repair_ratio": loss_pp / repair_pp,
            "below_a5": bool(value < a5),
        })

    summary = {
        "schema": "vimd_amc.a22_severe_held_decomposition.v1",
        "evidence_class": "post_review_diagnostic_zero_compute",
        "note": ("Per-jammer and per-regime decomposition of the frozen severe-held "
                 "predictions plus the S6 lesion loss/repair ratios. No inference, "
                 "no retraining."),
        "repair_effect_pp": repair_pp,
        "pooled_family_levels_pct": {
            fam: {m: 100 * v for m, v in fam_levels[fam].items()}
            for fam in ("pulse", "ofdm_like")
        },
        "severe_held_levels_pct": {
            scope: {m: 100 * v for m, v in levels_scope.items()}
            for scope, levels_scope in severe_levels.items()
        },
    }

    CSV.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fam_rows).to_csv(CSV / "a22_per_jammer_levels.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(CSV / "a22_regime_drift.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(CSV / "a22_lesion_ratios.csv", index=False)
    with open(OUT / "a22_severe_held_decomposition.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
