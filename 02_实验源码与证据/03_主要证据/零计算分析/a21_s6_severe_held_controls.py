"""A21 -- S6: severe-held frozen-inference attribution controls.

Three forward-only controls on the locked independent severe-held cache
(``standards/cache_s5_severe_held_v1``), per the frozen preregistration
``docs/S6_SEVERE_HELD_INFERENCE_CONTROLS_PREREG.md``:

* Control A   capacity tiers M / L forwarded on the same held regimes
* Control B   sidecar I/Q-branch information interventions (zero / shuffle)
* Control C   sidecar side-head logits removed

No training, no new simulation, no checkpoint reselection.

Usage:
    python a21_s6_severe_held_controls.py predict [--device cuda]
    python a21_s6_severe_held_controls.py analyze
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

import zc_core as z  # noqa: E402

CACHE_ROOT = REPO / "standards" / "cache_s5_severe_held_v1"
S5A_ARTIFACTS = REPO / "artifacts" / "s5_severe_held_v1"
SIDECAR_CKPT = REPO / "artifacts" / "tier2_iq_sidecar_v1" / "models"
CAPACITY_CKPT = {
    "M": REPO / "artifacts" / "tier2_capacity_M_v2" / "models",
    "L": REPO / "artifacts" / "tier2_capacity_L_v1" / "models",
}
OUT_ROOT = REPO / "artifacts" / "s6_severe_held_controls_v1"
FROZEN_CONFIG = REPO / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"

REGIMES = ("s5_severe_r1", "s5_severe_r2")
SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
MATCHED_FIVE = (17, 29, 43, 71, 101)
SIDECAR = "tier2_h2_f2_iq_sidecar"
CLASSES = 10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260820
SHUFFLE_SEED = 20260821
TIERS = {
    # spectral_channels, embedding_dim, environment_dim
    "M": (40, 80, 48),
    "L": (64, 128, 64),
}
CONDITIONS = ("capacity_M", "capacity_L", "sidecar_iq_zero",
              "sidecar_iq_shuffle", "sidecar_no_side_head")


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def _base_config() -> dict:
    return json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["scientific_invariants"]


def _model_config(spectral_channels: int, embedding_dim: int, environment_dim: int):
    from vimd_amc.models.common import ModelConfig

    frozen = _base_config()
    return ModelConfig(
        feature_channels=max(32, spectral_channels),
        environment_dim=environment_dim,
        embedding_dim=embedding_dim,
        spectral_channels=spectral_channels,
        n_fft=int(frozen["n_fft"]),
        hop_length=int(frozen["hop_length"]),
        dropout=float(frozen["dropout"]),
    )


def _load_tier(tier: str, seed: int, device: str):
    from experiments.run_standard_experiment import available_model_factories

    spectral, embedding, environment = TIERS[tier]
    config = _model_config(spectral, embedding, environment)
    built = available_model_factories()["a5_vimd_full"](CLASSES, 9, config)
    state = torch.load(
        CAPACITY_CKPT[tier] / f"a5_vimd_full_seed{seed}" / "model.pt",
        map_location="cpu",
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    built.model.load_state_dict(state, strict=True)
    built.model.eval()
    return built.model.to(device)


def _load_sidecar(seed: int, device: str):
    from analysis_zero_compute.tier2_gpu.run_tier2_experiment import (
        LightweightIQSidecarVIMD,
    )

    frozen = _base_config()
    config = _model_config(
        int(frozen["spectral_channels"]),
        int(frozen["embedding_dim"]),
        int(frozen["environment_dim"]),
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


# --------------------------------------------------------------------------
# forward variants
# --------------------------------------------------------------------------
@torch.no_grad()
def _plain_probabilities(model, windows: np.ndarray, device: str, batch: int = 512):
    rows = []
    for start in range(0, len(windows), batch):
        chunk = torch.from_numpy(
            np.asarray(windows[start:start + batch], dtype=np.float32)
        ).to(device)
        logits = model(chunk)["logits"].float()
        rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def _sidecar_logits(model, chunk: torch.Tensor, mode: str, perm=None) -> torch.Tensor:
    if mode == "sidecar_no_side_head":
        return model.base(chunk)["logits"].float()
    output = model.base(chunk)
    iq_input = chunk
    if mode == "sidecar_iq_zero":
        iq_input = torch.zeros_like(chunk)
    elif mode == "sidecar_iq_shuffle":
        index = torch.from_numpy(perm).to(chunk.device)
        iq_input = torch.gather(chunk, 2, index.unsqueeze(1).expand(-1, chunk.shape[1], -1))
    sequence = model.iq_branch(iq_input)
    pooled = torch.cat(
        (
            sequence.mean(-1),
            sequence.var(-1, unbiased=False).add(1e-6).sqrt(),
            sequence.amax(-1),
        ),
        dim=1,
    )
    iq_embedding = model.iq_projector(pooled)
    embedding = model.fusion(torch.cat((output["embedding"], iq_embedding), dim=1))
    return (output["logits"] + model.classifier(embedding)).float()


def _batch_permutation(start: int, count: int, length: int) -> np.ndarray:
    rng = np.random.default_rng([SHUFFLE_SEED, start])
    return np.argsort(rng.random((count, length)), axis=1).astype(np.int64)


@torch.no_grad()
def _sidecar_probabilities(model, windows: np.ndarray, mode: str, device: str,
                           batch: int = 512) -> np.ndarray:
    rows = []
    for start in range(0, len(windows), batch):
        chunk = torch.from_numpy(
            np.asarray(windows[start:start + batch], dtype=np.float32)
        ).to(device)
        perm = None
        if mode == "sidecar_iq_shuffle":
            perm = _batch_permutation(start, chunk.shape[0], chunk.shape[-1])
        logits = _sidecar_logits(model, chunk, mode, perm)
        rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
def _regime_metadata(regime: str) -> dict[str, np.ndarray]:
    root = CACHE_ROOT / regime
    out = {}
    for name in ("label", "snr_db", "sir_db", "source_id"):
        array = np.load(root / f"{name}.npy")
        out[name] = array[:, 0] if array.ndim >= 2 else array
    return out


def _save_predictions(out_path: Path, probabilities: np.ndarray, meta: dict,
                      regime: str, cache_digest: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        probabilities=probabilities,
        labels=meta["label"].astype(np.int64),
        source_ids=meta["source_id"].astype(np.int64),
        snr_db=meta["snr_db"].astype(np.float32),
        sir_db=meta["sir_db"].astype(np.float32),
        target_profile_index=np.zeros(len(meta["label"]), dtype=np.int64),
        cache_digest=np.asarray(cache_digest),
        split=np.asarray(regime),
    )


def predict(device: str = "cuda") -> None:
    cache_manifest = json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cache_digest = cache_manifest["cache_digest"]
    for regime in REGIMES:
        windows = np.asarray(np.load(CACHE_ROOT / regime / "x.npy", mmap_mode="r")[:, 0])
        meta = _regime_metadata(regime)
        for tier in ("M", "L"):
            for seed in MATCHED_FIVE:
                out_path = (OUT_ROOT / "models" / f"capacity_{tier}_seed{seed}"
                            / f"predictions_{regime}.npz")
                if out_path.exists():
                    continue
                model = _load_tier(tier, seed, device)
                probabilities = _plain_probabilities(model, windows, device)
                del model
                _save_predictions(out_path, probabilities, meta, regime, cache_digest)
                print(f"predicted capacity_{tier} seed{seed} on {regime}")
        for seed in SEEDS:
            model = _load_sidecar(seed, device)
            for mode in ("sidecar_iq_zero", "sidecar_iq_shuffle",
                         "sidecar_no_side_head"):
                out_path = (OUT_ROOT / "models" / f"{mode}_seed{seed}"
                            / f"predictions_{regime}.npz")
                if out_path.exists():
                    continue
                probabilities = _sidecar_probabilities(model, windows, mode, device)
                _save_predictions(out_path, probabilities, meta, regime, cache_digest)
                print(f"predicted {mode} seed{seed} on {regime}")
            del model


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _condition_path(condition: str, seed: int, regime: str) -> Path:
    if condition == "A5":
        return S5A_ARTIFACTS / "models" / f"a5_vimd_full_seed{seed}" / f"predictions_{regime}.npz"
    if condition in ("M", "L"):
        return OUT_ROOT / "models" / f"capacity_{condition}_seed{seed}" / f"predictions_{regime}.npz"
    if condition == "S":
        return REPO / "artifacts" / "s5_r1_sidecar_severe_held_v1" / "models" \
            / f"{SIDECAR}_seed{seed}" / f"predictions_{regime}.npz"
    return OUT_ROOT / "models" / f"{condition}_seed{seed}" / f"predictions_{regime}.npz"


def _condition_seeds(condition: str) -> tuple[int, ...]:
    return MATCHED_FIVE if condition in ("M", "L") else SEEDS


def _argmax_stack(condition: str, regime: str) -> tuple[np.ndarray, np.ndarray]:
    seeds = _condition_seeds(condition)
    per_seed = []
    labels = None
    for seed in seeds:
        with np.load(_condition_path(condition, seed, regime)) as data:
            per_seed.append(np.argmax(data["probabilities"], axis=-1))
            if labels is None:
                labels = data["labels"].astype(np.int64)
                snr = data["snr_db"].astype(np.float64)
            else:
                assert np.array_equal(labels, data["labels"]), "label mismatch"
    return np.stack(per_seed), labels, snr


def _seed_bootstrap_ci(diffs: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = diffs[rng.integers(0, n, n)].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_pp": float(100 * diffs.mean()),
        "ci95_low_pp": float(100 * low),
        "ci95_high_pp": float(100 * high),
        "positive_seeds": int((diffs > 0).sum()),
        "n_seeds": int(n),
    }


def analyze() -> None:
    import pandas as pd

    conditions = ("A5", "M", "L", "S", "sidecar_iq_zero",
                  "sidecar_iq_shuffle", "sidecar_no_side_head")
    preds = {regime: {c: _argmax_stack(c, regime) for c in conditions}
             for regime in REGIMES}

    # per-seed macro-F1 per condition per regime
    f1: dict[str, dict[str, np.ndarray]] = {}
    for regime in REGIMES:
        f1[regime] = {}
        for condition in conditions:
            stack, labels, _ = preds[regime][condition]
            f1[regime][condition] = np.array(
                [z.macro_f1(labels, stack[s]) for s in range(stack.shape[0])]
            )

    level_rows: list[dict] = []
    levels: dict[str, dict[str, float]] = {c: {} for c in conditions}
    for condition in conditions:
        for regime in REGIMES:
            value = float(f1[regime][condition].mean())
            levels[condition][regime] = value
            level_rows.append({
                "scope": regime,
                "condition": condition,
                "n_seeds": len(_condition_seeds(condition)),
                "macro_f1": value,
                "macro_f1_seed_std": float(f1[regime][condition].std(ddof=1)),
            })
        pooled = 0.5 * (f1[REGIMES[0]][condition] + f1[REGIMES[1]][condition])
        levels[condition]["pooled"] = float(pooled.mean())
        level_rows.append({
            "scope": "pooled",
            "condition": condition,
            "n_seeds": len(_condition_seeds(condition)),
            "macro_f1": float(pooled.mean()),
            "macro_f1_seed_std": float(pooled.std(ddof=1)),
        })

    # paired contrasts versus A5 on the matched seed set of the condition
    paired_rows: list[dict] = []
    contrasts: dict[str, dict[str, np.ndarray]] = {}
    for condition in ("M", "L", "S", "sidecar_iq_zero",
                      "sidecar_iq_shuffle", "sidecar_no_side_head"):
        n_seeds = len(_condition_seeds(condition))
        per_regime = {}
        for regime in REGIMES:
            a5 = f1[regime]["A5"][:n_seeds]
            per_regime[regime] = f1[regime][condition] - a5
        pooled = 0.5 * (per_regime[REGIMES[0]] + per_regime[REGIMES[1]])
        contrasts[condition] = pooled
        scope_diffs = [(r, per_regime[r]) for r in REGIMES] + [("pooled", pooled)]
        for scope, diffs in scope_diffs:
            stats = _seed_bootstrap_ci(diffs)
            paired_rows.append({
                "scope": scope,
                "contrast": f"{condition}_minus_A5",
                "n_seeds": n_seeds,
                **stats,
            })

    # matched-five-seed reference levels for Control A comparisons
    matched_five_rows: list[dict] = []
    for condition in ("A5", "S"):
        per_regime = {r: f1[r][condition][:len(MATCHED_FIVE)] for r in REGIMES}
        pooled = 0.5 * (per_regime[REGIMES[0]] + per_regime[REGIMES[1]])
        scope_values = [(r, per_regime[r]) for r in REGIMES] + [("pooled", pooled)]
        for scope, values in scope_values:
            matched_five_rows.append({
                "scope": scope,
                "condition": f"{condition}_matched5",
                "n_seeds": len(MATCHED_FIVE),
                "macro_f1": float(values.mean()),
                "macro_f1_seed_std": float(values.std(ddof=1)),
            })

    # ---------------- decision patterns (preregistration section 5) --------
    s_pool = levels["S"]["pooled"]
    l_pool = levels["L"]["pooled"]
    z_pool = levels["sidecar_iq_zero"]["pooled"]
    h_pool = levels["sidecar_iq_shuffle"]["pooled"]
    n_pool = levels["sidecar_no_side_head"]["pooled"]
    a5_pool = levels["A5"]["pooled"]
    full_repair = 100 * (s_pool - a5_pool)
    head_retained = 100 * (n_pool - a5_pool)
    cond_a = bool(s_pool > l_pool)
    cond_b = (100 * (s_pool - z_pool) >= 1.0) and (100 * (s_pool - h_pool) >= 1.0)
    cond_c = head_retained >= 0.5 * full_repair
    if not cond_a:
        pattern = "capacity_dominated"
    elif not cond_b:
        pattern = "path_dominated"
    elif cond_c:
        pattern = "strong_representation_supportive"
    else:
        pattern = "mixed_representation_and_fusion"

    summary = {
        "schema": "vimd_amc.a21_s6_severe_held_controls.v1",
        "evidence_class": "post_campaign_diagnostic_control",
        "preregistration": "docs/S6_SEVERE_HELD_INFERENCE_CONTROLS_PREREG.md",
        "cache_digest": json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))[
            "cache_digest"
        ],
        "seeds": list(SEEDS),
        "matched_five_seeds": list(MATCHED_FIVE),
        "pooled_seed_mean_macro_f1": {c: levels[c]["pooled"] for c in conditions},
        "regime_seed_mean_macro_f1": {
            c: {r: levels[c][r] for r in REGIMES} for c in conditions
        },
        "control_a": {
            "matched_five_levels": {
                "A5": float(np.mean([levels["A5"][r] for r in REGIMES])),
                "sidecar": float(np.mean([levels["S"][r] for r in REGIMES])),
            },
            "note": "matched-five rows restrict A5/sidecar to seeds 17-101; "
                    "levels above are full ten-seed for A5/S",
        },
        "control_b": {
            "full_minus_zero_pp": 100 * (s_pool - z_pool),
            "full_minus_shuffle_pp": 100 * (s_pool - h_pool),
        },
        "control_c": {
            "full_repair_pp": full_repair,
            "no_head_retained_pp": head_retained,
            "retained_fraction_of_repair": float(head_retained / full_repair)
            if full_repair else None,
        },
        "decision_pattern": pattern,
        "pattern_conditions": {"a": cond_a, "b": cond_b, "c": cond_c},
    }

    out = REPO / "analysis_zero_compute" / "outputs"
    csv = out / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(level_rows).to_csv(csv / "a21_control_levels.csv", index=False)
    pd.DataFrame(matched_five_rows).to_csv(csv / "a21_control_matched5_levels.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(csv / "a21_control_paired_contrasts.csv", index=False)
    with open(out / "a21_s6_controls_summary.json", "w", encoding="utf-8") as handle:
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
