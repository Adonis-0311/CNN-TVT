"""Tier-2 diagnostic H2-D (revision 2): is constellation-order information
present in the frozen representation on jammer-free windows?

Why a revision.  The first run (``artifacts/tier2_clean_probe_v1``) trained the
probe on ``x - jammer``.  The cache mixing law, verified in
``analysis_zero_compute/a10_counterfactual_check.py``, is an additive identity
on the stored post-AGC component scale:

    x = (clean + jammer + noise + receiver_artifact) / sqrt(mean |.|^2)

Removing the stored jammer without re-normalising yields windows whose mean square ranges from
about 0.046 to 0.5, whereas every training window has mean square exactly 0.5.
The resulting ``representation_limited`` verdict is confounded by input-scale
and composition shift and must not be used.

Two probe designs are provided; run both, they answer slightly different
questions and neither requires new simulation.

``--design counterfactual``
    Training windows are re-mixed from sealed components as
    ``AGC(clean + noise + receiver_artifact)`` -- the exact generative pipeline
    minus the jammer term.  All ten modulations are available, balanced.

``--design cv``
    No synthesis at all: fit the probe on real ``clean_retention`` windows with
    source-disjoint 5-fold cross-validation and evaluate on held-out folds.
    Because this fits on test-split sources it is a representation diagnostic
    only and may never be quoted as a performance number.

Both designs report a linear probe and a small MLP probe; the MLP is the more
honest upper bound on decodable information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
COMPOSITE = REPO / "artifacts" / "tvt_v4r_headline_composite"
CACHE = REPO / "standards" / "cache_factor_headline_1024_v2"
FROZEN_CONFIG = REPO / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"
HIGH_ORDER = (2, 4, 5)  # QPSK, 16QAM, 64QAM
DECISION_THRESHOLD = 0.25


def _agc(values: np.ndarray) -> np.ndarray:
    power = (values**2).sum(axis=(1, 2), keepdims=True) / values.shape[2]
    return values / np.sqrt(power)


def _load_model(model_name: str, seed: int, num_classes: int = 10, num_jammers: int = 9):
    import sys

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from experiments.run_standard_experiment import available_model_factories
    from vimd_amc.models.common import ModelConfig

    with open(FROZEN_CONFIG, encoding="utf-8") as handle:
        frozen = json.load(handle)["scientific_invariants"]
    config = ModelConfig(
        feature_channels=max(32, int(frozen["spectral_channels"])),
        environment_dim=int(frozen["environment_dim"]),
        embedding_dim=int(frozen["embedding_dim"]),
        spectral_channels=int(frozen["spectral_channels"]),
        n_fft=int(frozen["n_fft"]),
        hop_length=int(frozen["hop_length"]),
        dropout=float(frozen["dropout"]),
    )
    built = available_model_factories()[model_name](num_classes, num_jammers, config)
    state = torch.load(COMPOSITE / "models" / f"{model_name}_seed{seed}" / "model.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    built.model.load_state_dict(state)
    built.model.eval()
    return built.model


@torch.no_grad()
def _embed(model, inputs, device: str, batch: int = 256):
    embeddings, logits = [], []
    for start in range(0, len(inputs), batch):
        chunk = torch.from_numpy(np.array(inputs[start : start + batch], dtype=np.float32, copy=True)).to(device)
        output = model(chunk)
        if "embedding" not in output:
            raise SystemExit("model does not expose an 'embedding' output; probe not applicable")
        embeddings.append(output["embedding"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def _counterfactual_train(limit_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    root = CACHE / "train"
    labels = np.load(root / "label.npy")
    selected = np.concatenate(
        [np.flatnonzero(labels == c)[:limit_per_class] for c in range(10)]
    )
    selected.sort()
    clean = np.asarray(np.load(root / "clean.npy", mmap_mode="r")[selected, 0])
    noise = np.asarray(np.load(root / "noise.npy", mmap_mode="r")[selected, 0])
    artifact = np.asarray(np.load(root / "receiver_artifact.npy", mmap_mode="r")[selected, 0])
    return _agc(clean + noise + artifact), labels[selected]


def _fit_probes(train_x, train_y, test_x, test_y, device: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    train_scaled, test_scaled = scaler.transform(train_x), scaler.transform(test_x)

    results = {}
    for name, estimator in (
        ("linear", LogisticRegression(max_iter=3000, C=1.0)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, early_stopping=True, random_state=0)),
    ):
        estimator.fit(train_scaled, train_y)
        prediction = estimator.predict(test_scaled)
        per_class = f1_score(test_y, prediction, average=None, labels=list(range(10)), zero_division=0)
        results[name] = {
            "macro_f1": float(f1_score(test_y, prediction, average="macro", zero_division=0)),
            "f1_per_class": per_class.tolist(),
            "f1_high_order_mean": float(np.mean([per_class[i] for i in HIGH_ORDER])),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="a5_vimd_full")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--design", choices=("counterfactual", "cv"), default="counterfactual")
    parser.add_argument("--per-class", type=int, default=1200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "tier2_clean_probe_v2")
    arguments = parser.parse_args()

    from sklearn.metrics import f1_score

    model = _load_model(arguments.model, arguments.seed).to(arguments.device)

    test_x = np.asarray(np.load(CACHE / "clean_retention" / "x.npy", mmap_mode="r")[:, 0])
    test_y = np.load(CACHE / "clean_retention" / "label.npy")
    test_sources = np.load(CACHE / "clean_retention" / "source_id.npy")
    test_embedding, test_logits = _embed(model, test_x, arguments.device)
    head_per_class = f1_score(
        test_y, test_logits.argmax(axis=1), average=None, labels=list(range(10)), zero_division=0
    )

    if arguments.design == "counterfactual":
        train_x, train_y = _counterfactual_train(arguments.per_class)
        train_embedding, _ = _embed(model, train_x, arguments.device)
        probes = _fit_probes(train_embedding, train_y, test_embedding, test_y, arguments.device)
        train_description = {
            "train_source": "AGC(clean + noise + receiver_artifact) re-mixed from sealed components",
            "train_windows": int(len(train_y)),
            "mixing_law_verified_by": "analysis_zero_compute/a10_counterfactual_check.py",
        }
    else:
        rng = np.random.default_rng(20260812)
        order = rng.permutation(len(np.unique(test_sources)))
        fold_of_source = dict(zip(np.unique(test_sources), order % arguments.folds))
        folds = np.array([fold_of_source[s] for s in test_sources])
        accumulated = {"linear": [], "mlp": []}
        for fold in range(arguments.folds):
            hold = folds == fold
            fold_result = _fit_probes(
                test_embedding[~hold], test_y[~hold], test_embedding[hold], test_y[hold], arguments.device
            )
            for key in accumulated:
                accumulated[key].append(fold_result[key])
        probes = {
            key: {
                "macro_f1": float(np.mean([r["macro_f1"] for r in values])),
                "f1_per_class": np.mean([r["f1_per_class"] for r in values], axis=0).tolist(),
                "f1_high_order_mean": float(np.mean([r["f1_high_order_mean"] for r in values])),
            }
            for key, values in accumulated.items()
        }
        train_description = {
            "train_source": "real clean_retention windows, source-disjoint cross-validation",
            "folds": arguments.folds,
            "usage_restriction": "representation diagnostic only; never quote as a performance number",
        }

    record = {
        "schema": "tier2_clean_probe_v2",
        "evidence_class": "exploratory_diagnostic_outside_frozen_family",
        "supersedes": "tier2_clean_probe_v1 (confounded: x - jammer without post-removal renormalisation)",
        "model": arguments.model,
        "algorithm_seed": arguments.seed,
        "design": arguments.design,
        **train_description,
        "frozen_head_macro_f1": float(f1_score(test_y, test_logits.argmax(axis=1), average="macro", zero_division=0)),
        "frozen_head_f1_high_order_mean": float(np.mean([head_per_class[i] for i in HIGH_ORDER])),
        "frozen_head_f1_per_class": head_per_class.tolist(),
        "probes": probes,
        "decision_rule": (
            f"MLP probe mean F1 over QPSK/16QAM/64QAM > {DECISION_THRESHOLD} => information present, "
            "head/training-level fix; otherwise => representation-level deficiency"
        ),
        "conclusion": (
            "information_present_head_limited"
            if probes["mlp"]["f1_high_order_mean"] > DECISION_THRESHOLD
            else "representation_limited"
        ),
    }
    arguments.out.mkdir(parents=True, exist_ok=True)
    name = f"probe_{arguments.design}_{arguments.model}_seed{arguments.seed}.json"
    with open(arguments.out / name, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: record[k] for k in ("design", "model", "algorithm_seed", "probes",
                                             "frozen_head_f1_high_order_mean", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
