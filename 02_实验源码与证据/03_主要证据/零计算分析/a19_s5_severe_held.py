"""S5-A independent severe-held validation at SIR = -15 dB (a19).

Prospective/post-hoc robustness validation per ``docs/S5_A_SEVERE_HELD_PREREG.md``
(frozen 2026-08-20).  Two source-disjoint severe regimes that never
participated in the Eq.(8) envelope fitting are generated with the sealed
standards builder, evaluated with the sealed frozen checkpoints (no
retraining), and scored against the frozen 32-cell primary envelope fit.

Subcommands:
    generate   build the two-regime cache into standards/cache_s5_severe_held_v1
    predict    run the 30 sealed checkpoints over the new windows
    analyze    cells + fixed-fit envelope prediction + selector audit

Evidence class: prospective post-hoc robustness validation.  Nothing here
writes into the sealed composite or the sealed cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CACHE_ROOT = ROOT / "standards" / "cache_s5_severe_held_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "s5_severe_held_v1"
SEALED_CACHE = ROOT / "standards" / "cache_factor_headline_1024_v2"
COMPOSITE = ROOT / "artifacts" / "tvt_v4r_headline_composite"
FROZEN_CONFIG = ROOT / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"
A14_CELLS = ROOT / "analysis_zero_compute" / "outputs" / "csv" / "a14_envelope_cells.csv"
OUT = ROOT / "analysis_zero_compute" / "outputs"
CSV = OUT / "csv"

REGIMES = ("s5_severe_r1", "s5_severe_r2")
MODELS = ("a5_vimd_full", "mcldnn_reimplementation", "iqformer_inspired")
SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
SNR_DB_VALUES = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0)
SIR_DB = -15.0
SOURCES_PER_REGIME = 6400
MIN_ROWS = 150
MIN_CLASSES = 8
MACS_M = {"a5_vimd_full": 41.8, "iqformer_inspired": 355.6}  # sealed complexity table
MODULATIONS = (
    "BPSK", "PI2BPSK", "QPSK", "8PSK", "16QAM",
    "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK",
)
CLASSES = 10


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _policies():
    from vimd_amc.standards.cache import FactorSplitPolicy

    common = dict(
        snr_db_values=SNR_DB_VALUES,
        sir_db_values=(SIR_DB,),
        clean_fraction=0.0,
    )
    return (
        # administrative placeholders required by the builder validator;
        # never used for fitting or evaluation in S5-A
        FactorSplitPolicy(
            split="train",
            role="administrative_placeholder_not_used_by_s5",
            size=10,
            source_key="factor_isolated_s5::s5_placeholder_train",
            profiles=("TDL-A", "TDL-C", "TDL-D"),
            jammer_choices=("pulse", "ofdm_like"),
            speeds_kmh=(180.0,),
            snr_db_values=SNR_DB_VALUES,
            sir_db_values=(SIR_DB,),
            clean_fraction=0.2,
        ),
        FactorSplitPolicy(
            split="validation",
            role="administrative_placeholder_not_used_by_s5",
            size=10,
            source_key="factor_isolated_s5::s5_placeholder_validation",
            profiles=("TDL-A", "TDL-C", "TDL-D"),
            jammer_choices=("pulse", "ofdm_like"),
            speeds_kmh=(180.0,),
            snr_db_values=SNR_DB_VALUES,
            sir_db_values=(SIR_DB,),
            clean_fraction=0.2,
        ),
        FactorSplitPolicy(
            split="s5_severe_r1",
            role="independent_severe_held_validation_regime1_unseen_jammer_held_speed",
            size=SOURCES_PER_REGIME,
            source_key="factor_isolated_s5::s5_severe_r1",
            profiles=("TDL-A", "TDL-C", "TDL-D"),
            jammer_choices=("pulse", "ofdm_like"),
            speeds_kmh=(180.0,),
            held_factors=("jammer_family", "speed"),
            isolation_factors=("sir_severity", "jammer_family", "speed"),
            **common,
        ),
        FactorSplitPolicy(
            split="s5_severe_r2",
            role="independent_severe_held_validation_regime2_combined_ood",
            size=SOURCES_PER_REGIME,
            source_key="factor_isolated_s5::s5_severe_r2",
            profiles=("TDL-B", "TDL-E"),
            jammer_choices=("pulse", "ofdm_like"),
            speeds_kmh=(250.0,),
            held_factors=("jammer_family", "speed", "tdl_profile"),
            isolation_factors=("sir_severity", "jammer_family", "speed", "tdl_profile"),
            **common,
        ),
    )


def generate(matlab_batch_size: int = 4096) -> None:
    from vimd_amc.standards.cache import TDLCacheBuildConfig, build_tdl_paired_cache

    policies = _policies()
    config = TDLCacheBuildConfig(
        split_sizes=tuple((policy.split, policy.size) for policy in policies),
        sample_length=1024,
        guard_samples=96,
        sample_rate_hz=1_000_000.0,
        carrier_frequency_hz=5_900_000_000.0,
        master_seed=20260727,
        modulations=MODULATIONS,
        jammer_choices=("pulse", "ofdm_like"),
        train_profiles=("TDL-A", "TDL-C", "TDL-D"),
        heldout_profiles=("TDL-B", "TDL-E"),
        delay_spreads_s=(30e-9, 100e-9, 300e-9),
        speeds_kmh=(0.0, 60.0, 120.0, 150.0, 180.0, 250.0),
        snr_db_values=SNR_DB_VALUES,
        sir_db_values=(SIR_DB,),
        evidence_designation="s5_prospective_post_hoc_robustness_validation_v1",
        split_policies=policies,
    )
    result = build_tdl_paired_cache(
        CACHE_ROOT,
        config=config,
        matlab_batch_size=matlab_batch_size,
        matlab_timeout_s=3600.0,
    )
    _assert_source_disjointness(result.manifest)
    print(f"cache built at {CACHE_ROOT}")
    print(f"cache digest: {result.manifest['cache_digest']}")


def _assert_source_disjointness(manifest: dict) -> None:
    sealed = json.loads((SEALED_CACHE / "manifest.json").read_text(encoding="utf-8"))
    sealed_ids: set[int] = set()
    for ids in sealed["source_ids"].values():
        sealed_ids.update(int(value) for value in ids)
    new_ids: set[int] = set()
    for split in (*REGIMES, "train", "validation"):
        new_ids.update(int(value) for value in manifest["source_ids"][split])
    overlap = sealed_ids.intersection(new_ids)
    if overlap:
        raise AssertionError(f"source leakage vs sealed cache: {sorted(overlap)[:5]}")
    print(f"source disjointness OK: {len(new_ids)} new vs {len(sealed_ids)} sealed")


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def _load_model(model_name: str, seed: int, device: str):
    from experiments.run_standard_experiment import available_model_factories
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
    built = available_model_factories()[model_name](CLASSES, 9, config)
    state = torch.load(
        COMPOSITE / "models" / f"{model_name}_seed{seed}" / "model.pt",
        map_location="cpu",
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    built.model.load_state_dict(state)
    built.model.eval()
    return built.model.to(device)


def _regime_metadata(regime: str) -> dict[str, np.ndarray]:
    root = CACHE_ROOT / regime
    out = {}
    for name in ("label", "snr_db", "sir_db", "overlap", "source_id", "speed_kmh"):
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
        for model_name in MODELS:
            for seed in SEEDS:
                out_dir = ARTIFACT_ROOT / "models" / f"{model_name}_seed{seed}"
                out_path = out_dir / f"predictions_{regime}.npz"
                if out_path.exists():
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                model = _load_model(model_name, seed, device)
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
                print(f"predicted {model_name} seed{seed} on {regime}")


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    matrix = np.bincount(
        labels.astype(np.int64) * CLASSES + pred.astype(np.int64),
        minlength=CLASSES * CLASSES,
    ).reshape(CLASSES, CLASSES).astype(np.float64)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2 * recall * precision,
        recall + precision,
        out=np.zeros_like(tp),
        where=(recall + precision) > 0,
    )
    return float(f1[support > 0].mean())


def _frozen_fits() -> dict[str, dict]:
    """Re-derive the frozen 32-cell primary fits (no refitting on new data)."""
    import pandas as pd

    cells = pd.read_csv(A14_CELLS)
    fit_mask = cells.split.isin(("id_test", "hard_interference")).to_numpy()
    fit = cells[fit_mask]
    assert len(fit) == 32, f"expected 32 fitting cells, found {len(fit)}"
    fits: dict[str, dict] = {}
    for column, short in (("gain_vs_IQFormer", "IQFormer"), ("gain_vs_MCLDNN", "MCLDNN")):
        response = fit[column].to_numpy()
        full_design = np.column_stack(
            [np.ones(len(fit)), fit.occupancy_mean.to_numpy(), fit.sir_db.to_numpy()]
        )
        full_coef, *_ = np.linalg.lstsq(full_design, response, rcond=None)
        sir_design = np.column_stack([np.ones(len(fit)), fit.sir_db.to_numpy()])
        sir_coef, *_ = np.linalg.lstsq(sir_design, response, rcond=None)
        fits[short] = {
            "column": column,
            "full": full_coef,
            "sir_only": sir_coef,
            "constant": float(response.mean()),
        }
    return fits


def _cells_for_regime(regime: str) -> "tuple[list[dict], dict]":
    meta = _regime_metadata(regime)
    overlap = meta["overlap"].astype(np.float64)
    preds = {}
    for model_name in MODELS:
        stack = []
        for seed in SEEDS:
            with np.load(
                ARTIFACT_ROOT / "models" / f"{model_name}_seed{seed}"
                / f"predictions_{regime}.npz"
            ) as data:
                stack.append(np.asarray(data["probabilities"]).argmax(axis=1))
        preds[model_name] = np.stack(stack)  # [S, N]
    labels = meta["label"].astype(np.int64)
    snr = meta["snr_db"].astype(np.float64)
    quartiles = np.unique(np.quantile(overlap, [0.0, 0.25, 0.5, 0.75, 1.0]))
    bins = [(quartiles[i], quartiles[i + 1]) for i in range(len(quartiles) - 1)]
    rows: list[dict] = []
    for snr_value in np.unique(snr):
        for low, high in bins:
            selector = overlap >= low
            selector &= overlap <= high if high == quartiles[-1] else overlap < high
            mask = selector & (snr == snr_value)
            count = int(mask.sum())
            row = {
                "regime": regime,
                "snr_db": float(snr_value),
                "occupancy_mean": float(overlap[mask].mean()) if count else float("nan"),
                "row_count": count,
                "usable": bool(count >= MIN_ROWS and len(np.unique(labels[mask])) >= MIN_CLASSES),
            }
            if count:
                cell_labels = labels[mask]
                for model_name in MODELS:
                    f1 = np.array(
                        [_macro_f1(cell_labels, preds[model_name][s][mask]) for s in range(len(SEEDS))]
                    )
                    row[f"f1_{model_name}_mean"] = float(f1.mean())
                    row[f"f1_{model_name}_seeds"] = f1.tolist()
                row["gain_vs_IQFormer"] = row["f1_a5_vimd_full_mean"] - row["f1_iqformer_inspired_mean"]
                row["gain_vs_MCLDNN"] = row["f1_a5_vimd_full_mean"] - row["f1_mcldnn_reimplementation_mean"]
            rows.append(row)
    return rows, preds


def analyze() -> None:
    import pandas as pd

    fits = _frozen_fits()
    all_rows: list[dict] = []
    regime_preds: dict[str, dict] = {}
    for regime in REGIMES:
        rows, preds = _cells_for_regime(regime)
        all_rows.extend(rows)
        regime_preds[regime] = preds
    cells = pd.DataFrame(all_rows)
    usable = cells[cells.usable].copy()
    assert len(usable) >= 56, f"too few usable cells: {len(usable)}"

    # envelope evaluation with the frozen 32-cell fits (no refitting)
    env_rows: list[dict] = []
    env_report: dict = {"fit_cells": 32, "references": {}}
    for short, fit in fits.items():
        column = fit["column"]
        actual = usable[column].to_numpy()
        predicted = {
            "constant": np.full(len(usable), fit["constant"]),
            # SIR is constant at -15 dB across the severe set, so the frozen
            # SIR-only fit degenerates to its -15 dB fitted value on every cell
            "sir_only": np.full(
                len(usable), fit["sir_only"][0] + fit["sir_only"][1] * SIR_DB
            ),
            "overlap_sir": (
                fit["full"][0]
                + fit["full"][1] * usable.occupancy_mean.to_numpy()
                + fit["full"][2] * SIR_DB
            ),
        }
        for name, prediction in predicted.items():
            error = actual - prediction
            env_rows.append(
                {
                    "reference": short,
                    "predictor": name,
                    "cells": int(len(usable)),
                    "rmse_pp": float(100 * np.sqrt(np.mean(error**2))),
                    "mae_pp": float(100 * np.mean(np.abs(error))),
                    "sign_accuracy": float(
                        np.mean(np.sign(prediction) == np.sign(actual))
                    ),
                }
            )
        env_report["references"][short] = {
            "beta0": float(fit["full"][0]),
            "beta1_overlap": float(fit["full"][1]),
            "beta2_sir": float(fit["full"][2]),
        }

    # model-level pooled gains with paired-seed bootstrap CI
    level_rows: list[dict] = []
    rng = np.random.default_rng(20260820)
    for scope, frame in (("pooled", usable), *[(r, usable[usable.regime == r]) for r in REGIMES]):
        for column, short in (("gain_vs_IQFormer", "IQFormer"), ("gain_vs_MCLDNN", "MCLDNN")):
            level_rows.append(
                {
                    "scope": scope,
                    "reference": short,
                    "mean_gain_pp": float(100 * frame[column].mean()),
                    "cells": int(len(frame)),
                }
            )

    # selector audit on the pooled usable cells (IQFormer fallback policy)
    gain = usable["gain_vs_IQFormer"].to_numpy()
    pred_gain = (
        fits["IQFormer"]["full"][0]
        + fits["IQFormer"]["full"][1] * usable.occupancy_mean.to_numpy()
        + fits["IQFormer"]["full"][2] * SIR_DB
    )
    choose = pred_gain > 0
    positive = gain > 0
    f1_a5 = usable["f1_a5_vimd_full_mean"].to_numpy()
    f1_iq = usable["f1_iqformer_inspired_mean"].to_numpy()
    realized = float(np.where(choose, f1_a5, f1_iq).mean())
    oracle = float(np.where(positive, f1_a5, f1_iq).mean())
    fraction = float(choose.mean())
    macs = fraction * MACS_M["a5_vimd_full"] + (1 - fraction) * MACS_M["iqformer_inspired"]
    selector_report = {
        "cells": int(len(usable)),
        "tp": int((choose & positive).sum()),
        "fp": int((choose & ~positive).sum()),
        "fn": int((~choose & positive).sum()),
        "tn": int((~choose & ~positive).sum()),
        "recall": float((choose & positive).sum() / max(1, positive.sum())),
        "precision": float((choose & positive).sum() / max(1, choose.sum())),
        "realized_cell_mean_macro_f1": realized,
        "oracle_cell_mean_macro_f1": oracle,
        "regret_vs_oracle_pp": 100 * (oracle - realized),
        "expected_macs_million": macs,
        "mac_saving_vs_always_iqformer": 1.0 - macs / MACS_M["iqformer_inspired"],
    }

    # locked pass/fail verdict per reference
    verdicts: dict[str, dict] = {}
    for short in ("IQFormer", "MCLDNN"):
        env = {row["predictor"]: row for row in env_rows if row["reference"] == short}
        pooled = [r for r in level_rows if r["scope"] == "pooled" and r["reference"] == short][0]
        strong = (
            env["overlap_sir"]["sign_accuracy"] >= 0.75
            and env["overlap_sir"]["rmse_pp"] < env["sir_only"]["rmse_pp"]
            and env["overlap_sir"]["rmse_pp"] < env["constant"]["rmse_pp"]
            and pooled["mean_gain_pp"] > 0
            and selector_report["recall"] >= 0.75
        )
        direction_ok = env["overlap_sir"]["sign_accuracy"] >= 0.5
        verdicts[short] = {
            "strong_pass": bool(strong),
            "direction_ok": bool(direction_ok),
            "sign_accuracy": env["overlap_sir"]["sign_accuracy"],
            "rmse_full": env["overlap_sir"]["rmse_pp"],
            "rmse_sir_only": env["sir_only"]["rmse_pp"],
            "rmse_constant": env["constant"]["rmse_pp"],
            "pooled_gain_pp": pooled["mean_gain_pp"],
        }
    if all(v["strong_pass"] for v in verdicts.values()):
        overall = "strong_pass"
    elif verdicts["MCLDNN"]["strong_pass"] and verdicts["IQFormer"]["direction_ok"]:
        overall = "partial_pass"
    else:
        overall = "fail"

    cells_out = cells.drop(
        columns=[c for c in cells.columns if c.endswith("_seeds")]
    )
    CSV.mkdir(parents=True, exist_ok=True)
    cells_out.to_csv(CSV / "a19_severe_held_cells.csv", index=False)
    pd.DataFrame(env_rows).to_csv(CSV / "a19_severe_held_envelope.csv", index=False)
    pd.DataFrame(level_rows).to_csv(CSV / "a19_severe_held_model_levels.csv", index=False)
    report = {
        "schema": "vimd_amc.a19_s5_severe_held.v1",
        "evidence_class": "prospective_post_hoc_robustness_validation",
        "preregistration": "docs/S5_A_SEVERE_HELD_PREREG.md",
        "cache_root": str(CACHE_ROOT),
        "usable_cells": int(len(usable)),
        "total_cells": int(len(cells)),
        "envelope": env_report,
        "selector": selector_report,
        "verdicts": verdicts,
        "overall_verdict": overall,
    }
    with open(OUT / "a19_s5_severe_held.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "predict", "analyze", "all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matlab-batch-size", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.stage in ("generate", "all"):
        generate(matlab_batch_size=arguments.matlab_batch_size)
    if arguments.stage in ("predict", "all"):
        predict(device=arguments.device)
    if arguments.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
