"""Zero-compute analysis core for the TVT V4R composite evidence.

Read-only with respect to every frozen artifact.  Nothing in this package
writes into ``artifacts/tvt_v4r_headline_composite`` or the standards cache;
all products land in ``analysis_zero_compute/outputs``.

Estimators intentionally mirror ``src/vimd_amc/metrics.py``:

* macro-F1 is averaged over classes with nonzero ground-truth support;
* source clusters are resampled with replacement within modulation class;
* the hierarchical paired bootstrap resamples algorithm seeds and source
  clusters independently, applying the same resampled sources to every
  selected seed.

The reimplementation here is vectorised (bincount over confusion codes) so
that per-stratum intervals remain affordable; ``validate_against_composite``
in ``run_all.py`` cross-checks it against the sealed headline statistics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
COMPOSITE = REPO / "artifacts" / "tvt_v4r_headline_composite"
CACHE = REPO / "standards" / "cache_factor_headline_1024_v2"
OUT = Path(__file__).resolve().parent / "outputs"
CSV = OUT / "csv"
FIG = OUT / "figures"

CLASSES = 10
MODULATIONS = (
    "BPSK",
    "PI2BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "256QAM",
    "GMSK",
    "CPFSK",
    "4FSK",
)
JAMMER_TAXONOMY = (
    "tone",
    "multitone",
    "chirp",
    "sweep",
    "pulse",
    "partial_band",
    "comb",
    "cochannel",
    "ofdm_like",
)
SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
MODELS = (
    "a0_backbone",
    "a1_single_mask",
    "a2_tri_no_teacher",
    "a3_tri_teacher",
    "a3p_tri_proportional_teacher",
    "a4_tri_teacher_mtl",
    "a5_vimd_full",
    "a6_dual_full",
    "a7_vimd_no_residual",
    "cssl_amc_supervised_adaptation",
    "mcldnn_reimplementation",
    "iqformer_inspired",
)
SPLITS = (
    "validation",
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
    "adc_10bit_agc",
    "adc_12bit_agc",
    "per_emitter_sync",
)
PROPOSED = "a5_vimd_full"
BACKBONE = "a0_backbone"
REFERENCE = "cssl_amc_supervised_adaptation"
STRONG = ("mcldnn_reimplementation", "iqformer_inspired")
SHORT = {
    "a0_backbone": "A0",
    "a5_vimd_full": "A5-VIMD",
    "a3_tri_teacher": "A3",
    "a3p_tri_proportional_teacher": "A3p",
    "a6_dual_full": "A6",
    "a7_vimd_no_residual": "A7",
    "cssl_amc_supervised_adaptation": "CSSL",
    "mcldnn_reimplementation": "MCLDNN",
    "iqformer_inspired": "IQFormer",
}

BOOTSTRAP_SEED = 20260812
DEFAULT_DRAWS = int(os.environ.get("ZC_DRAWS", "2000"))


def ensure_dirs() -> None:
    CSV.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@dataclass
class Preds:
    """Predictions for one (model, split) across all algorithm seeds."""

    model: str
    split: str
    labels: np.ndarray  # [N]
    source_ids: np.ndarray  # [N]
    snr_db: np.ndarray  # [N]
    sir_db: np.ndarray  # [N]
    profile: np.ndarray  # [N]
    pred: np.ndarray  # [S, N] int8 argmax per seed
    conf: np.ndarray  # [S, N] float32 max probability per seed
    seeds: tuple[int, ...]


def fit_dir(model: str, seed: int) -> Path:
    return COMPOSITE / "models" / f"{model}_seed{seed}"


def load_probabilities(model: str, seed: int, split: str) -> np.ndarray:
    with np.load(fit_dir(model, seed) / f"predictions_{split}.npz") as data:
        return np.asarray(data["probabilities"], dtype=np.float32)


def load_preds(model: str, split: str, seeds: tuple[int, ...] = SEEDS) -> Preds:
    pred = np.empty((len(seeds), 0), dtype=np.int8)
    conf = np.empty((len(seeds), 0), dtype=np.float32)
    labels = source_ids = snr = sir = profile = None
    for index, seed in enumerate(seeds):
        with np.load(fit_dir(model, seed) / f"predictions_{split}.npz") as data:
            probabilities = np.asarray(data["probabilities"], dtype=np.float32)
            if labels is None:
                labels = np.asarray(data["labels"], dtype=np.int64)
                source_ids = np.asarray(data["source_ids"], dtype=np.int64)
                snr = np.asarray(data["snr_db"], dtype=np.float32)
                sir = np.asarray(data["sir_db"], dtype=np.float32)
                profile = np.asarray(data["target_profile_index"], dtype=np.int64)
                pred = np.empty((len(seeds), len(labels)), dtype=np.int8)
                conf = np.empty((len(seeds), len(labels)), dtype=np.float32)
            elif not np.array_equal(np.asarray(data["source_ids"], dtype=np.int64), source_ids):
                raise ValueError(f"source order differs for {model}/{seed}/{split}")
            pred[index] = probabilities.argmax(axis=1).astype(np.int8)
            conf[index] = probabilities.max(axis=1)
    return Preds(model, split, labels, source_ids, snr, sir, profile, pred, conf, seeds)


def load_cache_metadata(split: str) -> dict[str, np.ndarray]:
    """Per-sample generative metadata, view 0 (verified row-aligned)."""

    root = CACHE / split
    out: dict[str, np.ndarray] = {}
    for name in (
        "source_id",
        "label",
        "snr_db",
        "sir_db",
        "overlap",
        "jam_labels",
        "jammer_profile_index",
        "target_profile_index",
        "speed_kmh",
        "doppler_hz",
    ):
        path = root / f"{name}.npy"
        if not path.exists():
            continue
        array = np.load(path)
        out[name] = array[:, 0] if (array.ndim >= 2 and name != "jam_labels") else array
    if "jam_labels" in out and out["jam_labels"].ndim == 3:
        out["jam_labels"] = out["jam_labels"][:, 0, :]
    return out


def jammer_family_labels(split: str) -> tuple[np.ndarray, list[str]]:
    """Single dominant family name per row; 'mixed' when several are active."""

    meta = load_cache_metadata(split)
    multihot = meta.get("jam_labels")
    if multihot is None:
        return np.array([], dtype=object), []
    active = multihot > 0.5
    counts = active.sum(axis=1)
    names = np.full(len(active), "none", dtype=object)
    single = counts == 1
    names[single] = [JAMMER_TAXONOMY[i] for i in active[single].argmax(axis=1)]
    names[counts > 1] = "mixed"
    order = [name for name in (*JAMMER_TAXONOMY, "mixed", "none") if (names == name).any()]
    return names, order


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def macro_f1(labels: np.ndarray, pred: np.ndarray, classes: int = CLASSES) -> float:
    code = labels.astype(np.int64) * classes + pred.astype(np.int64)
    matrix = np.bincount(code, minlength=classes * classes).reshape(classes, classes)
    return _macro_f1_from_matrix(matrix.astype(np.float64))


def _macro_f1_from_matrix(matrix: np.ndarray) -> float:
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    return float(f1[supported].mean()) if supported.any() else float("nan")


def per_class_f1(labels: np.ndarray, pred: np.ndarray, classes: int = CLASSES) -> np.ndarray:
    code = labels.astype(np.int64) * classes + pred.astype(np.int64)
    matrix = np.bincount(code, minlength=classes * classes).reshape(classes, classes).astype(float)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    return np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )


def _weighted_macro_f1(codes: np.ndarray, weights: np.ndarray, classes: int) -> float:
    matrix = np.bincount(codes, weights=weights, minlength=classes * classes)
    return _macro_f1_from_matrix(matrix.reshape(classes, classes))


def _class_strata(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == c) for c in np.unique(labels)]


def hierarchical_paired_diff(
    labels: np.ndarray,
    ref_pred: np.ndarray,
    cand_pred: np.ndarray,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    classes: int = CLASSES,
    stratify_by_class: bool = True,
) -> dict[str, float]:
    """Paired macro-F1/accuracy difference with a hierarchical bootstrap.

    ``ref_pred`` / ``cand_pred`` have shape [S, N] (algorithm seeds x rows).
    One row equals one test source cluster in this evidence base.
    """

    labels = labels.astype(np.int64)
    n_seeds, n_rows = ref_pred.shape
    ref_codes = labels[None, :] * classes + ref_pred.astype(np.int64)
    cand_codes = labels[None, :] * classes + cand_pred.astype(np.int64)
    ones = np.ones(n_rows, dtype=np.float64)

    per_seed_f1 = np.empty(n_seeds)
    per_seed_acc = np.empty(n_seeds)
    for s in range(n_seeds):
        per_seed_f1[s] = _weighted_macro_f1(cand_codes[s], ones, classes) - _weighted_macro_f1(
            ref_codes[s], ones, classes
        )
        per_seed_acc[s] = float((cand_pred[s] == labels).mean() - (ref_pred[s] == labels).mean())

    strata = _class_strata(labels) if stratify_by_class else [np.arange(n_rows)]
    rng = np.random.default_rng(seed)
    f1_draws = np.empty(draws)
    acc_draws = np.empty(draws)
    ref_correct = (ref_pred == labels[None, :]).astype(np.float64)
    cand_correct = (cand_pred == labels[None, :]).astype(np.float64)
    for d in range(draws):
        picked = np.concatenate(
            [group[rng.integers(0, len(group), len(group))] for group in strata]
        )
        mult = np.bincount(picked, minlength=n_rows).astype(np.float64)
        seed_pick = rng.integers(0, n_seeds, n_seeds)
        f1_values = np.empty(n_seeds)
        acc_values = np.empty(n_seeds)
        total = mult.sum()
        for j, s in enumerate(seed_pick):
            f1_values[j] = _weighted_macro_f1(cand_codes[s], mult, classes) - _weighted_macro_f1(
                ref_codes[s], mult, classes
            )
            acc_values[j] = (cand_correct[s] @ mult - ref_correct[s] @ mult) / total
        f1_draws[d] = f1_values.mean()
        acc_draws[d] = acc_values.mean()

    lo_f1, hi_f1 = np.quantile(f1_draws, [0.025, 0.975])
    lo_acc, hi_acc = np.quantile(acc_draws, [0.025, 0.975])
    return {
        "macro_f1_difference": float(per_seed_f1.mean()),
        "macro_f1_ci95_low": float(lo_f1),
        "macro_f1_ci95_high": float(hi_f1),
        "macro_f1_seed_std": float(per_seed_f1.std(ddof=1)),
        "accuracy_difference": float(per_seed_acc.mean()),
        "accuracy_ci95_low": float(lo_acc),
        "accuracy_ci95_high": float(hi_acc),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "row_count": int(n_rows),
        "algorithm_seed_count": int(n_seeds),
    }


def seed_level_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t-interval across algorithm seeds (descriptive)."""

    from scipy import stats

    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def tost_noninferiority(values: np.ndarray, margin: float) -> dict[str, float]:
    """One-sided seed-level non-inferiority test against ``-margin``."""

    from scipy import stats

    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    t_stat = (mean + margin) / se if se > 0 else np.inf
    p = float(stats.t.sf(t_stat, len(values) - 1))
    return {
        "mean": mean,
        "margin": float(margin),
        "t_statistic": float(t_stat),
        "p_value_noninferiority": p,
        "non_inferior_at_0_05": bool(p < 0.05),
    }


def write_csv(name: str, rows: list[dict]) -> Path:
    import pandas as pd

    ensure_dirs()
    path = CSV / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_shard(stem: str, tag: str, rows: list[dict]) -> Path:
    """Write a chunk of an analysis; chunks are merged by ``merge_shards``."""

    import pandas as pd

    parts = CSV / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    path = parts / f"{stem}__{tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def merge_shards(stem: str):
    import pandas as pd

    parts = sorted((CSV / "parts").glob(f"{stem}__*.csv"))
    if not parts:
        raise FileNotFoundError(f"no shards for {stem}")
    frame = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    ensure_dirs()
    frame.to_csv(CSV / f"{stem}.csv", index=False)
    return frame


def save_fig(fig, name: str) -> list[Path]:
    ensure_dirs()
    paths = []
    for suffix in ("pdf", "png"):
        path = FIG / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    return paths


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
            "savefig.transparent": False,
        }
    )
    return plt


def result_json(model: str, seed: int) -> dict:
    with open(fit_dir(model, seed) / "result.json", encoding="utf-8") as handle:
        return json.load(handle)
