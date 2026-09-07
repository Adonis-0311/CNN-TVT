"""Run non-headline learnability controls on an immutable TDL cache.

The emitted numbers answer two diagnostic questions only:

1. How separable are received mixtures under a transparent fixed-feature
   ridge control?
2. How separable are the unavailable clean received target components under
   the same features and under the A0 neural backbone?

Clean-input results are oracle upper controls.  They are not deployable
baselines, source-separation scores, or evidence for a paper performance claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.data.controls import CleanOracleInputDataset  # noqa: E402
from vimd_amc.metrics import PredictionBundle, classification_metrics  # noqa: E402
from vimd_amc.models.baselines import BackboneClassifier  # noqa: E402
from vimd_amc.models.classical import (  # noqa: E402
    ClassicalHOCyclostationaryClassifier,
    ClassicalHOCyclostationaryFeatures,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.standards import CachedPairedAMCDataset  # noqa: E402


EVIDENCE_DESIGNATION = "diagnostic_upper_control_only"
REQUIRED_SPLITS = ("train", "validation", "heldout_channel")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_dependency_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "vimd_amc" / "data" / "__init__.py",
        ROOT / "src" / "vimd_amc" / "data" / "controls.py",
        ROOT / "src" / "vimd_amc" / "models" / "__init__.py",
        ROOT / "src" / "vimd_amc" / "models" / "classical.py",
        ROOT / "src" / "vimd_amc" / "models" / "baselines.py",
        ROOT / "src" / "vimd_amc" / "models" / "spectral.py",
        ROOT / "src" / "vimd_amc" / "metrics.py",
        ROOT / "src" / "vimd_amc" / "standards" / "cache.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }


def _paired_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat(
        (batch["view1"]["x"], batch["view2"]["x"]),
        dim=0,
    ).to(device)
    labels = torch.cat((batch["label"], batch["label"]), dim=0).to(
        device=device,
        dtype=torch.long,
    )
    return values, labels


def _feature_matrix(
    dataset: Dataset,
    extractor: ClassicalHOCyclostationaryFeatures,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    extractor.eval()
    with torch.no_grad():
        for batch in loader:
            values, target = _paired_batch(batch, device)
            matrices.append(extractor(values).cpu().numpy().astype(np.float64))
            labels.append(target.cpu().numpy().astype(np.int64))
    return np.concatenate(matrices), np.concatenate(labels)


def _fit_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    classes: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.concatenate(
        (standardized, np.ones((len(standardized), 1), dtype=np.float64)),
        axis=1,
    )
    targets = np.eye(classes, dtype=np.float64)[labels]
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    right = design.T @ targets
    try:
        weights = np.linalg.solve(gram, right)
        solver = "solve"
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ right
        solver = "pinv_fallback"
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "solver": np.array(solver),
    }


def _ridge_probabilities(
    model: dict[str, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.concatenate(
        (standardized, np.ones((len(standardized), 1), dtype=np.float64)),
        axis=1,
    )
    logits = design @ model["weights"]
    logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _prediction_context(
    dataset: Dataset,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    labels: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    snr_db: list[np.ndarray] = []
    sir_db: list[np.ndarray] = []
    for batch in loader:
        labels.append(
            torch.cat((batch["label"], batch["label"])).numpy().astype(np.int64)
        )
        source_ids.append(
            torch.cat((batch["source_id"], batch["source_id"]))
            .numpy()
            .astype(np.int64)
        )
        snr_db.append(
            torch.cat((batch["view1"]["snr_db"], batch["view2"]["snr_db"]))
            .numpy()
            .astype(np.float64)
        )
        sir_db.append(
            torch.cat((batch["view1"]["sir_db"], batch["view2"]["sir_db"]))
            .numpy()
            .astype(np.float64)
        )
    return (
        np.concatenate(labels),
        np.concatenate(source_ids),
        np.concatenate(snr_db),
        np.concatenate(sir_db),
    )


def _metrics_from_probabilities(
    probabilities: np.ndarray,
    dataset: Dataset,
    *,
    classes: int,
    batch_size: int,
) -> dict[str, Any]:
    labels, source_ids, snr_db, sir_db = _prediction_context(
        dataset,
        batch_size=batch_size,
    )
    bundle = PredictionBundle(
        probabilities=probabilities,
        labels=labels,
        source_ids=source_ids,
        snr_db=snr_db,
        sir_db=sir_db,
    )
    return classification_metrics(bundle, classes)


def _train_clean_a0(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    classes: int,
    sample_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[BackboneClassifier, dict[str, Any]]:
    n_fft = 64 if sample_length >= 64 else 32
    config = ModelConfig(
        n_fft=n_fft,
        hop_length=max(4, n_fft // 4),
        spectral_channels=24,
        embedding_dim=48,
        environment_dim=16,
        dropout=0.0,
    )
    model = BackboneClassifier(classes, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch in train_loader:
            values, labels = _paired_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
        model.eval()
        validation_correct = 0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                values, labels = _paired_batch(batch, device)
                logits = model(values)["logits"]
                validation_correct += int((logits.argmax(dim=1) == labels).sum())
                validation_count += len(labels)
        validation_accuracy = validation_correct / max(validation_count, 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(count, 1),
                "train_accuracy": correct / max(count, 1),
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("A0 training failed to create a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "epochs": epochs,
        "best_validation_accuracy": best_accuracy,
        "selected_epoch": int(
            max(history, key=lambda row: row["validation_accuracy"])["epoch"]
        ),
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "history": history,
        "model_config": asdict(config),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _neural_probabilities(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values, _ = _paired_batch(batch, device)
            logits = model(values)["logits"]
            probabilities.append(
                torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            )
    return np.concatenate(probabilities)


def _metric_record(
    *,
    model: str,
    input_condition: str,
    split: str,
    metrics: dict[str, Any],
    oracle_clean_input: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "input_condition": input_condition,
        "split": split,
        "oracle_clean_input": oracle_clean_input,
        "deployment_eligible": False if oracle_clean_input else True,
        "evidence_designation": EVIDENCE_DESIGNATION,
        **metrics,
    }


def run(args: argparse.Namespace) -> Path:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cache_root = args.cache_root.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_splits = set(REQUIRED_SPLITS).difference(manifest["files"])
    if missing_splits:
        raise ValueError(f"cache is missing splits: {sorted(missing_splits)}")
    configuration = manifest["configuration"]
    modulations = tuple(configuration["modulations"])
    sample_length = int(configuration["sample_length"])
    classes = len(modulations)
    if classes <= 1:
        raise ValueError("cache must contain at least two modulation classes")

    run_root = args.output.resolve() / args.run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_root}")
    run_root.mkdir(parents=True)

    source_hashes_start = _source_dependency_hashes()
    datasets = {
        split: CachedPairedAMCDataset(
            cache_root,
            split,
            verify_checksums=not args.skip_cache_checksums,
        )
        for split in REQUIRED_SPLITS
    }
    try:
        oracle_datasets = {
            split: CleanOracleInputDataset(dataset)
            for split, dataset in datasets.items()
        }
        extractor = ClassicalHOCyclostationaryFeatures().to(device)
        mixture_features = {
            split: _feature_matrix(
                dataset,
                extractor,
                batch_size=args.batch_size,
                device=device,
            )
            for split, dataset in datasets.items()
        }
        oracle_features = {
            split: _feature_matrix(
                dataset,
                extractor,
                batch_size=args.batch_size,
                device=device,
            )
            for split, dataset in oracle_datasets.items()
        }
        mixture_ridge = _fit_ridge(
            *mixture_features["train"],
            classes=classes,
            ridge=args.ridge,
        )
        oracle_ridge = _fit_ridge(
            *oracle_features["train"],
            classes=classes,
            ridge=args.ridge,
        )

        metric_records: list[dict[str, Any]] = []
        detailed_results: list[dict[str, Any]] = []
        for control_name, feature_sets, ridge_model, source_datasets, oracle in (
            (
                "hoc_cyclostationary_ridge_mixture",
                mixture_features,
                mixture_ridge,
                datasets,
                False,
            ),
            (
                "hoc_cyclostationary_ridge_clean_oracle",
                oracle_features,
                oracle_ridge,
                oracle_datasets,
                True,
            ),
        ):
            split_metrics: dict[str, Any] = {}
            for split in ("validation", "heldout_channel"):
                probabilities = _ridge_probabilities(
                    ridge_model,
                    feature_sets[split][0],
                )
                metrics = _metrics_from_probabilities(
                    probabilities,
                    source_datasets[split],
                    classes=classes,
                    batch_size=args.batch_size,
                )
                split_metrics[split] = metrics
                metric_records.append(
                    _metric_record(
                        model=control_name,
                        input_condition="clean_oracle" if oracle else "mixture",
                        split=split,
                        metrics=metrics,
                        oracle_clean_input=oracle,
                    )
                )
            detailed_results.append(
                {
                    "model": control_name,
                    "input_condition": "clean_oracle" if oracle else "mixture",
                    "oracle_clean_input": oracle,
                    "deployment_eligible": False if oracle else True,
                    "head": {
                        "type": "closed_form_ridge_on_standardized_features",
                        "ridge": args.ridge,
                        "solver": str(ridge_model["solver"]),
                        "intercept_regularized": False,
                    },
                    "feature_schema": list(extractor.feature_names),
                    "feature_count": extractor.output_dim,
                    "metrics": split_metrics,
                }
            )

        clean_a0, clean_a0_training = _train_clean_a0(
            oracle_datasets["train"],
            oracle_datasets["validation"],
            classes=classes,
            sample_length=sample_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        clean_a0_metrics: dict[str, Any] = {}
        for split in ("validation", "heldout_channel"):
            probabilities = _neural_probabilities(
                clean_a0,
                oracle_datasets[split],
                batch_size=args.batch_size,
                device=device,
            )
            metrics = _metrics_from_probabilities(
                probabilities,
                oracle_datasets[split],
                classes=classes,
                batch_size=args.batch_size,
            )
            clean_a0_metrics[split] = metrics
            metric_records.append(
                _metric_record(
                    model="a0_backbone_clean_oracle",
                    input_condition="clean_oracle",
                    split=split,
                    metrics=metrics,
                    oracle_clean_input=True,
                )
            )
        detailed_results.append(
            {
                "model": "a0_backbone_clean_oracle",
                "input_condition": "clean_oracle",
                "oracle_clean_input": True,
                "deployment_eligible": False,
                "training": clean_a0_training,
                "metrics": clean_a0_metrics,
            }
        )

        by_key = {
            (row["model"], row["split"]): row
            for row in metric_records
        }
        deltas = {
            split: {
                "hoc_clean_minus_mixture_accuracy": (
                    by_key[
                        ("hoc_cyclostationary_ridge_clean_oracle", split)
                    ]["accuracy"]
                    - by_key[
                        ("hoc_cyclostationary_ridge_mixture", split)
                    ]["accuracy"]
                ),
                "hoc_clean_minus_mixture_macro_f1": (
                    by_key[
                        ("hoc_cyclostationary_ridge_clean_oracle", split)
                    ]["macro_f1"]
                    - by_key[
                        ("hoc_cyclostationary_ridge_mixture", split)
                    ]["macro_f1"]
                ),
            }
            for split in ("validation", "heldout_channel")
        }
        source_hashes_end = _source_dependency_hashes()
        payload = {
            "schema_version": "learnability-controls-v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_designation": EVIDENCE_DESIGNATION,
            "headline_evidence_eligible": False,
            "paper_performance_claim_allowed": False,
            "warnings": [
                "Clean-input controls use unavailable ground-truth components.",
                "These single-seed diagnostics are not deployable baselines.",
                "No statistical power or paper-performance claim is supported.",
                "Two views of each source are evaluated as rows; source IDs remain recorded.",
            ],
            "cache": {
                "path": str(cache_root),
                "cache_digest": manifest["cache_digest"],
                "cache_evidence_designation": configuration.get(
                    "evidence_designation"
                ),
                "checksums_verified": not args.skip_cache_checksums,
                "modulations": list(modulations),
                "sample_length": sample_length,
                "split_sizes": {
                    split: len(dataset)
                    for split, dataset in datasets.items()
                },
            },
            "protocol": {
                "algorithm_seed": args.seed,
                "device": str(device),
                "threads": args.threads,
                "batch_size": args.batch_size,
                "ridge": args.ridge,
                "a0_epochs": args.epochs,
                "a0_learning_rate": args.learning_rate,
                "evaluation_unit": "view_row_with_source_id_retained",
                "validation_use": "A0 checkpoint selection only",
            },
            "classical_control_provenance": (
                ClassicalHOCyclostationaryClassifier.provenance
            ),
            "clean_oracle_provenance": oracle_datasets[
                "train"
            ].control_metadata(),
            "results": detailed_results,
            "clean_vs_mixture_diagnostic_deltas": deltas,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "source_dependency_hashes": {
                "start": source_hashes_start,
                "end": source_hashes_end,
                "unchanged": source_hashes_start == source_hashes_end,
            },
        }
        if not payload["source_dependency_hashes"]["unchanged"]:
            raise RuntimeError("source dependencies changed during control run")

        metrics_path = run_root / "metrics.csv"
        run_path = run_root / "run.json"
        pd.DataFrame(metric_records).to_csv(metrics_path, index=False)
        run_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        checksums = {
            "algorithm": "sha256",
            "scope": "generated diagnostic artifacts excluding this checksum file",
            "cache_manifest_digest": manifest["cache_digest"],
            "files": {
                "metrics.csv": _sha256_file(metrics_path),
                "run.json": _sha256_file(run_path),
            },
        }
        (run_root / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
    return run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "standards" / "cache_screening_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "learnability_controls",
    )
    parser.add_argument(
        "--run-id",
        default=(
            "learnability_controls_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cache-checksums", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(output)
