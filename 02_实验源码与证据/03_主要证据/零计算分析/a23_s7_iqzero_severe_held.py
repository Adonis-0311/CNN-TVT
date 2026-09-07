"""A23 -- S7: severe-held evaluation of the retrained zeroed-I/Q sidecar.

Frozen-checkpoint inference of ``tier2_h2_f2_iq_sidecar_iqzero`` (retrained
with the I/Q branch input permanently zeroed; preregistration
``docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md``) on the locked severe-held
cache, followed by the preregistered paired analysis:

* primary contrast   F1_iqzero - F1_A5                 (R1, R2, pooled)
* secondary          F1_iqzero - F1_sidecar(full)
* reference          matched-five-seed full-sidecar-versus-A5 (Delta_full)
* automatic decision rule (preregistration Section 4)

Matched five seeds only (17, 29, 43, 71, 101). Reused predictions: A5 from
``artifacts/s5_severe_held_v1`` and the full sidecar from
``artifacts/s5_r1_sidecar_severe_held_v1`` (identical windows and labels).

Usage:
    python a23_s7_iqzero_severe_held.py predict [--device cuda]
    python a23_s7_iqzero_severe_held.py analyze
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
sys.path.insert(0, str(REPO / "analysis_zero_compute"))
sys.path.insert(0, str(REPO / "analysis_zero_compute" / "tier2_gpu"))

import zc_core as z  # noqa: E402

CACHE_ROOT = REPO / "standards" / "cache_s5_severe_held_v1"
S5A_ARTIFACTS = REPO / "artifacts" / "s5_severe_held_v1"
SIDECAR_ARTIFACTS = REPO / "artifacts" / "s5_r1_sidecar_severe_held_v1"
IQZERO_CKPT = REPO / "artifacts" / "tier2_iq_sidecar_iqzero_v1" / "models"
OUT_ROOT = REPO / "artifacts" / "s7_iqzero_severe_held_v1"
OUT = REPO / "analysis_zero_compute" / "outputs"
CSV = OUT / "csv"
FROZEN_CONFIG = REPO / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"

REGIMES = ("s5_severe_r1", "s5_severe_r2")
SEEDS = (17, 29, 43, 71, 101)
IQZERO = "tier2_h2_f2_iq_sidecar_iqzero"
FULL_SIDECAR = "tier2_h2_f2_iq_sidecar"
A5 = "a5_vimd_full"
CLASSES = 10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260821


def _base_config() -> dict:
    return json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["scientific_invariants"]


def _model_config():
    from vimd_amc.models.common import ModelConfig

    frozen = _base_config()
    return ModelConfig(
        feature_channels=max(32, int(frozen["spectral_channels"])),
        environment_dim=int(frozen["environment_dim"]),
        embedding_dim=int(frozen["embedding_dim"]),
        spectral_channels=int(frozen["spectral_channels"]),
        n_fft=int(frozen["n_fft"]),
        hop_length=int(frozen["hop_length"]),
        dropout=float(frozen["dropout"]),
    )


def _load_iqzero(seed: int, device: str):
    from run_tier2_iqzero_experiment import IQZeroLightweightIQSidecarVIMD

    model = IQZeroLightweightIQSidecarVIMD(CLASSES, 9, _model_config())
    state = torch.load(
        IQZERO_CKPT / f"{IQZERO}_seed{seed}" / "model.pt",
        map_location="cpu",
        weights_only=True,
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
            np.asarray(windows[start:start + batch], dtype=np.float32)
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
            out_dir = OUT_ROOT / "models" / f"{IQZERO}_seed{seed}"
            out_path = out_dir / f"predictions_{regime}.npz"
            if out_path.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            model = _load_iqzero(seed, device)
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
            print(f"predicted {IQZERO} seed{seed} on {regime}")


# --------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------
def _pred_path(model: str, seed: int, regime: str) -> Path:
    if model == IQZERO:
        root = OUT_ROOT
    elif model == FULL_SIDECAR:
        root = SIDECAR_ARTIFACTS
    else:
        root = S5A_ARTIFACTS
    return root / "models" / f"{model}_seed{seed}" / f"predictions_{regime}.npz"


def _load_predictions(regime: str) -> tuple[dict, np.ndarray, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    labels = snr = None
    for model in (IQZERO, A5, FULL_SIDECAR):
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
        "positive_seeds": int((diffs > 0).sum()),
        "per_seed_signs": [int(np.sign(d)) for d in diffs],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def analyze() -> None:
    data = {regime: _load_predictions(regime) for regime in REGIMES}

    f1 = {
        regime: {
            model: np.array(
                [z.macro_f1(data[regime][1], data[regime][0][model][s]) for s in range(len(SEEDS))]
            )
            for model in (IQZERO, A5, FULL_SIDECAR)
        }
        for regime in REGIMES
    }

    model_rows: list[dict] = []
    short = {IQZERO: "iqzero", A5: "A5", FULL_SIDECAR: "sidecar_full"}
    for regime in REGIMES:
        for model in (IQZERO, A5, FULL_SIDECAR):
            model_rows.append(
                {
                    "scope": regime,
                    "model": short[model],
                    "macro_f1": float(f1[regime][model].mean()),
                    "macro_f1_seed_std": float(f1[regime][model].std(ddof=1)),
                }
            )
    pooled = {
        model: 0.5 * (f1[REGIMES[0]][model] + f1[REGIMES[1]][model])
        for model in (IQZERO, A5, FULL_SIDECAR)
    }
    for model in (IQZERO, A5, FULL_SIDECAR):
        model_rows.append(
            {
                "scope": "pooled",
                "model": short[model],
                "macro_f1": float(pooled[model].mean()),
                "macro_f1_seed_std": float(pooled[model].std(ddof=1)),
            }
        )

    # -- paired contrasts ----------------------------------------------------
    contrasts = {
        "iqzero_vs_A5": (IQZERO, A5),
        "iqzero_vs_sidecar_full": (IQZERO, FULL_SIDECAR),
        "sidecar_full_vs_A5": (FULL_SIDECAR, A5),
    }
    paired_rows: list[dict] = []
    per_seed: dict[str, dict[str, np.ndarray]] = {c: {} for c in contrasts}
    for regime in REGIMES:
        diffs = {c: f1[regime][a] - f1[regime][b] for c, (a, b) in contrasts.items()}
        for c in contrasts:
            per_seed[c][regime] = diffs[c]
        for c, d in diffs.items():
            stats = _seed_bootstrap_ci(d)
            paired_rows.append(
                {
                    "scope": regime,
                    "contrast": c,
                    "mean_diff_pp": stats["mean_pp"],
                    "ci95_low_pp": stats["ci95_low_pp"],
                    "ci95_high_pp": stats["ci95_high_pp"],
                    "positive_seeds": stats["positive_seeds"],
                    "per_seed_signs": str(stats["per_seed_signs"]),
                }
            )
    pooled_diffs = {
        c: 0.5 * (per_seed[c][REGIMES[0]] + per_seed[c][REGIMES[1]]) for c in contrasts
    }
    pooled_stats = {c: _seed_bootstrap_ci(d) for c, d in pooled_diffs.items()}
    for c, stats in pooled_stats.items():
        paired_rows.append(
            {
                "scope": "pooled",
                "contrast": c,
                "mean_diff_pp": stats["mean_pp"],
                "ci95_low_pp": stats["ci95_low_pp"],
                "ci95_high_pp": stats["ci95_high_pp"],
                "positive_seeds": stats["positive_seeds"],
                "per_seed_signs": str(stats["per_seed_signs"]),
            }
        )

    # -- preregistration decision rule (Section 4) ----------------------------
    delta_zero = pooled_stats["iqzero_vs_A5"]
    delta_full = pooled_stats["sidecar_full_vs_A5"]
    if delta_zero["ci95_high_pp"] < 1.0:
        verdict = "attribution_supported"
    elif delta_zero["mean_pp"] >= 3.0:
        verdict = "attribution_refuted"
    else:
        verdict = "unresolved"

    summary = {
        "schema": "vimd_amc.a23_s7_iqzero_severe_held.v1",
        "evidence_class": "exploratory_retrained_ablation",
        "preregistration": "docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md",
        "seeds": list(SEEDS),
        "regimes": list(REGIMES),
        "cache_digest": json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))[
            "cache_digest"
        ],
        "pooled_macro_f1": {short[m]: float(pooled[m].mean()) for m in pooled},
        "pooled_contrasts_pp": {
            c: {
                "mean": s["mean_pp"],
                "ci95_low": s["ci95_low_pp"],
                "ci95_high": s["ci95_high_pp"],
                "positive_seeds": s["positive_seeds"],
            }
            for c, s in pooled_stats.items()
        },
        "decision_rule": {
            "delta_zero_point_pp": delta_zero["mean_pp"],
            "delta_zero_ci95_high_pp": delta_zero["ci95_high_pp"],
            "delta_full_point_pp": delta_full["mean_pp"],
            "verdict": verdict,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = _csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(CSV / "a23_s7_model_levels.csv", model_rows)
    _write_csv(CSV / "a23_s7_paired_contrasts.csv", paired_rows)
    (OUT / "a23_s7_iqzero_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


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
