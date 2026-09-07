"""A20 -- S5-R1: independent severe-held representation-repair validation.

Frozen-checkpoint inference of the received-I/Q sidecar on the two S5-A
severe-held regimes, followed by the preregistered analysis:

* primary contrast   F1_sidecar - F1_A5          (R1, R2, pooled)
* secondary          F1_sidecar - F1_IQFormer / - F1_MCLDNN
* per-SNR and per-class decomposition
* descriptive gap-recovery ratios
* automatic verdict: strong / moderate / weak_unresolved / fail

Preregistration: docs/S5_R1_PREREG.md (FROZEN 2026-08-20).
Evidence class: prospective/post-hoc robustness validation.

Usage:
    python a20_s5_r1_sidecar_severe_held.py predict [--device cuda]
    python a20_s5_r1_sidecar_severe_held.py analyze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import zc_core as z  # noqa: E402

ROOT = REPO
CACHE_ROOT = ROOT / "standards" / "cache_s5_severe_held_v1"
S5A_ARTIFACTS = ROOT / "artifacts" / "s5_severe_held_v1"
SIDECAR_CKPT = ROOT / "artifacts" / "tier2_iq_sidecar_v1" / "models"
OUT_ROOT = ROOT / "artifacts" / "s5_r1_sidecar_severe_held_v1"
FROZEN_CONFIG = ROOT / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"

REGIMES = ("s5_severe_r1", "s5_severe_r2")
SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
SIDECAR = "tier2_h2_f2_iq_sidecar"
MODELS = (SIDECAR, "a5_vimd_full", "iqformer_inspired", "mcldnn_reimplementation")
SHORT = {
    SIDECAR: "sidecar",
    "a5_vimd_full": "A5",
    "iqformer_inspired": "IQFormer",
    "mcldnn_reimplementation": "MCLDNN",
}
CLASSES = 10
CLASS_NAMES = z.MODULATIONS
FOCUS_CLASSES = ("BPSK", "QPSK", "16QAM", "64QAM", "GMSK", "4FSK")
MACS_M = {"A5": 41.8, "sidecar": 43.1, "IQFormer": 355.6, "MCLDNN": 398.2}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260820
SNR_GRID = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0)


# --------------------------------------------------------------------------
# inference (sidecar only; the other three models reuse S5-A predictions)
# --------------------------------------------------------------------------
def _load_sidecar(seed: int, device: str):
    from analysis_zero_compute.tier2_gpu.run_tier2_experiment import (
        LightweightIQSidecarVIMD,
    )
    from vimd_amc.models.common import ModelConfig

    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["scientific_invariants"]
    config = ModelConfig(
        feature_channels=max(32, int(frozen["spectral_channels"])),
        environment_dim=int(frozen["environment_dim"]),
        embedding_dim=int(frozen["embedding_dim"]),
        spectral_channels=int(frozen["spectral_channels"]),
        n_fft=int(frozen["n_fft"]),
        hop_length=int(frozen["hop_length"]),
        dropout=float(frozen["dropout"]),
    )
    model = LightweightIQSidecarVIMD(CLASSES, 9, config)
    state = torch.load(
        SIDECAR_CKPT / f"{SIDECAR}_seed{seed}" / "model.pt", map_location="cpu"
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def _regime_metadata(regime: str) -> dict[str, np.ndarray]:
    root = CACHE_ROOT / regime
    out = {}
    for name in ("label", "snr_db", "sir_db", "overlap", "source_id"):
        array = np.load(root / f"{name}.npy")
        out[name] = array[:, 0] if array.ndim >= 2 else array
    return out


@torch.no_grad()
def _probabilities(model, windows: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    rows = []
    for start in range(0, len(windows), batch):
        chunk = torch.from_numpy(
            np.asarray(windows[start : start + batch], dtype=np.float32)
        ).to(device)
        logits = model(chunk)["logits"].float()
        rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def predict(device: str = "cuda") -> None:
    cache_manifest = json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for regime in REGIMES:
        windows = np.asarray(np.load(CACHE_ROOT / regime / "x.npy", mmap_mode="r")[:, 0])
        meta = _regime_metadata(regime)
        for seed in SEEDS:
            out_dir = OUT_ROOT / "models" / f"{SIDECAR}_seed{seed}"
            out_path = out_dir / f"predictions_{regime}.npz"
            if out_path.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            model = _load_sidecar(seed, device)
            probabilities = _probabilities(model, windows, device)
            del model
            np.savez_compressed(
                out_path,
                probabilities=probabilities,
                labels=meta["label"].astype(np.int64),
                source_ids=meta["source_id"].astype(np.int64),
                snr_db=meta["snr_db"].astype(np.float32),
                sir_db=meta["sir_db"].astype(np.float32),
                target_profile_index=np.zeros(len(windows), dtype=np.int64),
                cache_digest=np.asarray(cache_manifest["cache_digest"]),
                split=np.asarray(regime),
            )
            print(f"predicted {SIDECAR} seed{seed} on {regime}")


# --------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------
def _pred_path(model: str, seed: int, regime: str) -> Path:
    root = OUT_ROOT if model == SIDECAR else S5A_ARTIFACTS
    return root / "models" / f"{model}_seed{seed}" / f"predictions_{regime}.npz"


def _load_predictions(regime: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return {model: argmax [S, N]}, labels [N], snr [N] (all shared)."""
    preds: dict[str, np.ndarray] = {}
    labels = snr = None
    for model in MODELS:
        per_seed = []
        for seed in SEEDS:
            data = np.load(_pred_path(model, seed, regime))
            probabilities = data["probabilities"]
            per_seed.append(np.argmax(probabilities, axis=-1))
            if labels is None:
                labels = data["labels"].astype(np.int64)
                snr = data["snr_db"].astype(np.float64)
                assert np.array_equal(labels, data["labels"]), "label mismatch"
        preds[model] = np.stack(per_seed)
    return preds, labels, snr


def _per_class_f1(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    matrix = np.bincount(
        labels * CLASSES + pred.astype(np.int64), minlength=CLASSES * CLASSES
    ).reshape(CLASSES, CLASSES).astype(np.float64)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = recall + precision
    f1 = np.divide(2 * recall * precision, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1


def _seed_bootstrap_ci(diffs: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = diffs[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_pp": float(100 * diffs.mean()),
        "ci95_low_pp": float(100 * lo),
        "ci95_high_pp": float(100 * hi),
        "per_seed_signs": [int(np.sign(d)) for d in diffs],
        "positive_seeds": int((diffs > 0).sum()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _paired_rows(scope: str, diffs_by_seed: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for contrast, diffs in diffs_by_seed.items():
        stats = _seed_bootstrap_ci(diffs)
        rows.append(
            {
                "scope": scope,
                "contrast": contrast,
                "mean_diff_pp": stats["mean_pp"],
                "ci95_low_pp": stats["ci95_low_pp"],
                "ci95_high_pp": stats["ci95_high_pp"],
                "positive_seeds": stats["positive_seeds"],
                "per_seed_signs": str(stats["per_seed_signs"]),
            }
        )
    return rows


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyze() -> None:
    data = {regime: _load_predictions(regime) for regime in REGIMES}

    # -- per-seed macro-F1 per model per regime ----------------------------
    f1 = {
        regime: {
            model: np.array(
                [z.macro_f1(data[regime][1], data[regime][0][model][s]) for s in range(len(SEEDS))]
            )
            for model in MODELS
        }
        for regime in REGIMES
    }

    model_rows: list[dict] = []
    for regime in REGIMES:
        for model in MODELS:
            model_rows.append(
                {
                    "scope": regime,
                    "model": SHORT[model],
                    "macro_f1": float(f1[regime][model].mean()),
                    "macro_f1_seed_std": float(f1[regime][model].std(ddof=1)),
                }
            )
    for model in MODELS:
        pooled = 0.5 * (f1[REGIMES[0]][model] + f1[REGIMES[1]][model])
        model_rows.append(
            {
                "scope": "pooled",
                "model": SHORT[model],
                "macro_f1": float(pooled.mean()),
                "macro_f1_seed_std": float(pooled.std(ddof=1)),
            }
        )

    # -- paired contrasts (primary + secondary) ----------------------------
    paired_rows: list[dict] = []
    contrasts = ("sidecar_vs_A5", "sidecar_vs_IQFormer", "sidecar_vs_MCLDNN")
    refs = {"sidecar_vs_A5": "a5_vimd_full",
            "sidecar_vs_IQFormer": "iqformer_inspired",
            "sidecar_vs_MCLDNN": "mcldnn_reimplementation"}
    per_seed_contrasts: dict[str, dict[str, np.ndarray]] = {c: {} for c in contrasts}
    for regime in REGIMES:
        diffs = {
            contrast: f1[regime][SIDECAR] - f1[regime][refs[contrast]]
            for contrast in contrasts
        }
        for contrast in contrasts:
            per_seed_contrasts[contrast][regime] = diffs[contrast]
        paired_rows += _paired_rows(regime, diffs)
        # window-level hierarchical audit (secondary, per regime)
        for contrast, ref in refs.items():
            audit = z.hierarchical_paired_diff(
                data[regime][1],
                data[regime][0][ref],
                data[regime][0][SIDECAR],
            )
            paired_rows.append(
                {
                    "scope": f"{regime}_window_level_audit",
                    "contrast": contrast,
                    "mean_diff_pp": 100 * audit["macro_f1_difference"],
                    "ci95_low_pp": 100 * audit["macro_f1_ci95_low"],
                    "ci95_high_pp": 100 * audit["macro_f1_ci95_high"],
                    "positive_seeds": int((diffs[contrast] > 0).sum()),
                    "per_seed_signs": "-",
                }
            )
    pooled_diffs = {
        contrast: 0.5
        * (per_seed_contrasts[contrast][REGIMES[0]] + per_seed_contrasts[contrast][REGIMES[1]])
        for contrast in contrasts
    }
    paired_rows += _paired_rows("pooled", pooled_diffs)

    # -- per-SNR table -------------------------------------------------------
    snr_rows: list[dict] = []
    snr_strata_positive = 0
    for snr in SNR_GRID:
        row: dict = {"snr_db": snr}
        for scope in (*REGIMES, "pooled"):
            if scope == "pooled":
                masks = [data[r][2] == snr for r in REGIMES]
                values = []
                for model in MODELS:
                    per_regime = [
                        np.mean(
                            [
                                z.macro_f1(data[r][1][m], data[r][0][model][s][m])
                                for s in range(len(SEEDS))
                            ]
                        )
                        for r, m in zip(REGIMES, masks)
                    ]
                    values.append(float(np.mean(per_regime)))
            else:
                mask = data[scope][2] == snr
                labels_m = data[scope][1][mask]
                values = [
                    float(
                        np.mean(
                            [
                                z.macro_f1(labels_m, data[scope][0][model][s][mask])
                                for s in range(len(SEEDS))
                            ]
                        )
                    )
                    for model in MODELS
                ]
            prefix = {"s5_severe_r1": "r1_", "s5_severe_r2": "r2_", "pooled": "pooled_"}[scope]
            row[prefix + "A5"] = values[1]
            row[prefix + "sidecar"] = values[0]
            row[prefix + "IQFormer"] = values[2]
            row[prefix + "MCLDNN"] = values[3]
            row[prefix + "sidecar_minus_A5"] = values[0] - values[1]
        if row["pooled_sidecar_minus_A5"] > 0:
            snr_strata_positive += 1
        snr_rows.append(row)

    # -- per-class table (pooled + per regime) -----------------------------
    class_rows: list[dict] = []
    for scope in (*REGIMES, "pooled"):
        per_class_delta: dict[int, list[float]] = {}
        per_class_sidecar: dict[int, list[float]] = {}
        per_class_a5: dict[int, list[float]] = {}
        for regime in REGIMES if scope == "pooled" else (scope,):
            preds, labels, _ = data[regime]
            for s in range(len(SEEDS)):
                f1_sc = _per_class_f1(labels, preds[SIDECAR][s])
                f1_a5 = _per_class_f1(labels, preds["a5_vimd_full"][s])
                for c in range(CLASSES):
                    per_class_delta.setdefault(c, []).append(f1_sc[c] - f1_a5[c])
                    per_class_sidecar.setdefault(c, []).append(f1_sc[c])
                    per_class_a5.setdefault(c, []).append(f1_a5[c])
        for c in range(CLASSES):
            name = CLASS_NAMES[c] if CLASS_NAMES else f"class_{c}"
            class_rows.append(
                {
                    "scope": scope,
                    "class": name,
                    "focus": name in FOCUS_CLASSES,
                    "sidecar_f1": float(np.mean(per_class_sidecar[c])),
                    "a5_f1": float(np.mean(per_class_a5[c])),
                    "delta_pp": float(100 * np.mean(per_class_delta[c])),
                }
            )

    # -- descriptive gap recovery -------------------------------------------
    pooled_levels = {
        model: float(
            0.5 * (f1[REGIMES[0]][model] + f1[REGIMES[1]][model]).mean()
        )
        for model in MODELS
    }
    recovery_rows: list[dict] = []
    recovery: dict[str, float | None] = {}
    for contrast, ref in (("sidecar_vs_IQFormer", "iqformer_inspired"),
                          ("sidecar_vs_MCLDNN", "mcldnn_reimplementation")):
        delta_repair = pooled_levels[SIDECAR] - pooled_levels["a5_vimd_full"]
        gap = pooled_levels[ref] - pooled_levels["a5_vimd_full"]
        if gap > 0:
            ratio = delta_repair / abs(gap)
        else:
            ratio = None
        recovery[contrast] = ratio
        recovery_rows.append(
            {
                "contrast": contrast,
                "a5_macro_f1": pooled_levels["a5_vimd_full"],
                "sidecar_macro_f1": pooled_levels[SIDECAR],
                "reference_macro_f1": pooled_levels[ref],
                "delta_repair_pp": 100 * delta_repair,
                "reference_gap_pp": 100 * gap,
                "recovery_ratio": ratio,
                "note": "descriptive normalized gap recovery; defined only when "
                "the reference beats A5; not a causal effect size",
            }
        )

    # -- automatic verdict (preregistration section 7) ----------------------
    pooled_primary = pooled_diffs["sidecar_vs_A5"]
    primary_stats = _seed_bootstrap_ci(pooled_primary)
    delta_pp = primary_stats["mean_pp"]
    ci_lo, ci_hi = primary_stats["ci95_low_pp"], primary_stats["ci95_high_pp"]
    r1_positive = bool((f1[REGIMES[0]][SIDECAR] - f1[REGIMES[0]]["a5_vimd_full"]).mean() > 0)
    r2_positive = bool((f1[REGIMES[1]][SIDECAR] - f1[REGIMES[1]]["a5_vimd_full"]).mean() > 0)
    r_iq = recovery["sidecar_vs_IQFormer"]

    if delta_pp > 3.0 and ci_lo > 0 and r1_positive and r2_positive \
            and snr_strata_positive >= 6 and r_iq is not None and r_iq >= 0.5:
        verdict = "strong_repair"
        wording = ("Independent severe-held validation supports a representation-repair "
                   "effect that generalizes beyond the original campaign.")
    elif delta_pp > 1.5 and ci_lo > 0 and r1_positive and r2_positive \
            and r_iq is not None and 0.25 <= r_iq < 0.5:
        verdict = "moderate_repair"
        wording = "The sidecar partially repairs the independent severe-held degradation."
    elif delta_pp > 0:
        verdict = "weak_unresolved"
        wording = "Positive point estimate, but independent severe-held repair is unresolved."
    else:
        verdict = "fail"
        wording = "The original sidecar repair does not transfer to the independent severe-held regimes."

    # baseline level (instruction section 14)
    iq_diff = pooled_levels[SIDECAR] - pooled_levels["iqformer_inspired"]
    mc_diff = pooled_levels[SIDECAR] - pooled_levels["mcldnn_reimplementation"]
    if verdict in ("strong_repair", "moderate_repair"):
        if mc_diff > 0 or iq_diff > 0:
            level = "C"
        elif abs(iq_diff) < 1.0:
            level = "B"
        else:
            level = "A"
    else:
        level = "n/a"

    summary = {
        "schema": "vimd_amc.a20_s5_r1_sidecar_severe_held.v1",
        "evidence_class": "prospective_post_hoc_robustness_validation",
        "preregistration": "docs/S5_R1_PREREG.md",
        "checkpoint_audit": "docs/S5_R1_CHECKPOINT_AUDIT.md",
        "cache_digest": json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))[
            "cache_digest"
        ],
        "seeds": list(SEEDS),
        "seed_rule": "A (all ten seeds present; ten-seed primary analysis)",
        "pooled_levels_macro_f1": {SHORT[m]: v for m, v in pooled_levels.items()},
        "primary": {
            "contrast": "sidecar_minus_A5",
            "r1_pp": float(
                100 * (f1[REGIMES[0]][SIDECAR] - f1[REGIMES[0]]["a5_vimd_full"]).mean()
            ),
            "r2_pp": float(
                100 * (f1[REGIMES[1]][SIDECAR] - f1[REGIMES[1]]["a5_vimd_full"]).mean()
            ),
            "pooled_pp": delta_pp,
            "ci95_pp": [ci_lo, ci_hi],
            "positive_seeds": primary_stats["positive_seeds"],
            "snr_strata_sidecar_better_pooled": snr_strata_positive,
        },
        "gap_recovery": {
            "vs_IQFormer": r_iq,
            "vs_MCLDNN": recovery["sidecar_vs_MCLDNN"],
        },
        "baseline_level": level,
        "macs_million": MACS_M,
        "verdict": verdict,
        "verdict_wording": wording,
    }

    out = ROOT / "analysis_zero_compute" / "outputs"
    csv = out / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(model_rows).to_csv(csv / "a20_s5_r1_model_levels.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(csv / "a20_s5_r1_paired_contrasts.csv", index=False)
    pd.DataFrame(snr_rows).to_csv(csv / "a20_s5_r1_per_snr.csv", index=False)
    pd.DataFrame(class_rows).to_csv(csv / "a20_s5_r1_per_class.csv", index=False)
    pd.DataFrame(recovery_rows).to_csv(csv / "a20_s5_r1_gap_recovery.csv", index=False)
    with open(out / "a20_s5_r1_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "analyze"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args.device)
    else:
        analyze()


if __name__ == "__main__":
    main()
