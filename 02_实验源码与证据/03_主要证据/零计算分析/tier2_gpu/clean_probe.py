"""Tier-2 diagnostic H2-D: is the constellation-order information present but
unused, or absent from the frozen representation?

Background (zero-compute result A3b): on the clean-retention split every one of
the nine A0--A7 models collapses on exactly QPSK / 16QAM / 64QAM (mean recall
< 0.01), while CSSL, MCLDNN and IQFormer do not.  The collapse is therefore a
property of the shared spectral backbone family, not of the VIMD tri-route
decomposition.

This probe decides between two explanations without training any new network:

* **Representation is sufficient, head is not** -- a linear probe on the frozen
  modulation embedding recovers the three classes on unjammed windows.  The fix
  is then cheap (clean-conditional head / rebalanced objective).
* **Representation is insufficient** -- the probe also fails.  The fix must
  change the front end (amplitude/phase statistics or an IQ side branch), which
  is an architecture-level revision.

Reads sealed checkpoints read-only.  Writes nothing outside ``--out``.
Requires torch + CUDA on the local machine; not executable in the analysis
sandbox.
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


def _load_model(model_name: str, seed: int, num_classes: int, num_jammers: int):
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
    factories = available_model_factories()
    built = factories[model_name](num_classes, num_jammers, config)
    state = torch.load(COMPOSITE / "models" / f"{model_name}_seed{seed}" / "model.pt", map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    built.model.load_state_dict(state)
    built.model.eval()
    return built.model


@torch.no_grad()
def _embed(model, inputs: np.ndarray, device: str, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    embeddings, logits = [], []
    for start in range(0, len(inputs), batch):
        chunk = torch.from_numpy(np.array(inputs[start : start + batch], dtype=np.float32, copy=True)).to(device)
        output = model(chunk)
        embeddings.append(output["embedding"].float().cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def _counterfactual_clean_train(limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove the sealed jammer component and sample every class evenly.

    The cached train split's naturally clean views cover only four modulation
    classes, so they cannot identify the preregistered ten-class probe.  The
    factor-isolated cache provides an audited additive jammer component for
    every frozen view; subtracting it creates a no-new-simulation clean
    counterfactual while retaining the frozen channel, noise, and receiver
    realization.
    """

    labels = np.load(CACHE / "train" / "label.npy", mmap_mode="r")
    mixtures = np.load(CACHE / "train" / "x.npy", mmap_mode="r")
    jammers = np.load(CACHE / "train" / "jammer.npy", mmap_mode="r")
    per_class = limit // 10
    remainder = limit % 10
    windows, window_labels = [], []
    for class_index in range(10):
        source_indices = np.flatnonzero(np.asarray(labels) == class_index)
        flat_indices = np.column_stack(
            (np.repeat(source_indices, 2), np.tile(np.arange(2), len(source_indices)))
        )
        take = per_class + (1 if class_index < remainder else 0)
        selected = flat_indices[:take]
        windows.append(
            np.asarray(mixtures[selected[:, 0], selected[:, 1]], dtype=np.float32)
            - np.asarray(jammers[selected[:, 0], selected[:, 1]], dtype=np.float32)
        )
        window_labels.append(np.full(take, class_index, dtype=np.int64))
    return np.concatenate(windows), np.concatenate(window_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="a5_vimd_full")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "tier2_clean_probe_v1")
    arguments = parser.parse_args()

    model = _load_model(arguments.model, arguments.seed, 10, 9).to(arguments.device)

    train_x, train_y = _counterfactual_clean_train(arguments.train_limit)
    test_x = np.load(CACHE / "clean_retention" / "x.npy", mmap_mode="r")[:, 0]
    test_y = np.load(CACHE / "clean_retention" / "label.npy")

    train_embedding, _ = _embed(model, train_x, arguments.device, arguments.batch)
    test_embedding, test_logits = _embed(model, test_x, arguments.device, arguments.batch)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(train_embedding, train_y)
    probe_pred = probe.predict(test_embedding)
    head_pred = test_logits.argmax(axis=1)

    per_class_probe = f1_score(test_y, probe_pred, average=None, labels=list(range(10)), zero_division=0)
    per_class_head = f1_score(test_y, head_pred, average=None, labels=list(range(10)), zero_division=0)

    record = {
        "schema": "tier2_clean_probe_v1",
        "evidence_class": "exploratory_diagnostic_outside_frozen_family",
        "model": arguments.model,
        "algorithm_seed": arguments.seed,
        "clean_train_windows": int(len(train_y)),
        "clean_train_source": "counterfactual_clean_from_sealed_components:x_minus_jammer",
        "clean_train_class_counts": np.bincount(train_y, minlength=10).tolist(),
        "probe_macro_f1": float(f1_score(test_y, probe_pred, average="macro", zero_division=0)),
        "frozen_head_macro_f1": float(f1_score(test_y, head_pred, average="macro", zero_division=0)),
        "probe_f1_per_class": per_class_probe.tolist(),
        "head_f1_per_class": per_class_head.tolist(),
        "probe_f1_qpsk_16qam_64qam": [float(per_class_probe[i]) for i in HIGH_ORDER],
        "head_f1_qpsk_16qam_64qam": [float(per_class_head[i]) for i in HIGH_ORDER],
        "decision_rule": (
            "probe mean F1 over QPSK/16QAM/64QAM > 0.25 => information present, head-level fix; "
            "otherwise => representation-level deficiency, front-end revision required"
        ),
    }
    record["conclusion"] = (
        "information_present_head_limited"
        if float(np.mean(record["probe_f1_qpsk_16qam_64qam"])) > 0.25
        else "representation_limited"
    )

    arguments.out.mkdir(parents=True, exist_ok=True)
    with open(arguments.out / f"clean_probe_{arguments.model}_seed{arguments.seed}.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: record[k] for k in ("probe_macro_f1", "frozen_head_macro_f1",
                                             "probe_f1_qpsk_16qam_64qam", "head_f1_qpsk_16qam_64qam",
                                             "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
