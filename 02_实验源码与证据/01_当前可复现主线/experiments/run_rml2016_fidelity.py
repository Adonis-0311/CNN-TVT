"""Run the bounded E1 RadioML2016.10a comparator-fidelity check.

This is deliberately independent of the VIMD-Net experiment family.  It
records enough provenance to make a local PyTorch comparator result auditable,
but does not create an external-validation claim for the VIMD method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimd_amc.models.baselines import (  # noqa: E402
    IQFormerInspiredClassifier,
    MCLDNNReimplementation,
)

IQFORMER_CLASSES = [
    "8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK",
    "AM-DSB", "AM-SSB", "WBFM",
]
MCLDNN_CLASSES = sorted(IQFORMER_CLASSES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logits_of(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    result = model(features)
    return result["logits"] if isinstance(result, dict) else result


def load_data(path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    snrs = sorted({int(key[1]) for key in data})
    expected = {(mod, snr) for mod in classes for snr in snrs}
    if set(data) != expected:
        raise ValueError("Unexpected RML2016.10a key set; refusing to silently subset data")
    samples, labels, sample_snrs = [], [], []
    for label, modulation in enumerate(classes):
        for snr in snrs:
            values = np.asarray(data[(modulation, snr)], dtype=np.float32)
            if values.shape != (1000, 2, 128):
                raise ValueError(f"{modulation}/{snr}: expected [1000,2,128], got {values.shape}")
            samples.append(values)
            labels.append(np.full(1000, label, dtype=np.int64))
            sample_snrs.append(np.full(1000, snr, dtype=np.int16))
    return np.concatenate(samples), np.concatenate(labels), np.concatenate(sample_snrs), snrs


def mcldnn_split(labels: np.ndarray, snrs: np.ndarray, classes: list[str], snr_grid: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E1-realigned MCLDNN split: the random_state=233 per-stratum 80/20 then
    75/25 protocol, identical to the IQFormer benchmark pipeline.

    E1 history: the original MCLDNN repository (wzjialang/MCLDNN, dataset2016.py)
    uses an np.random.seed(2016) set-based per-stratum 600/200/200 partition, and
    the official MCLDNN performance is published only as a figure, not as a
    machine-readable trace.  The only machine-readable published MCLDNN result
    on RML2016.10a is the IQFormer benchmark result table (testA.xlsx in the
    official IQFormer repository), which was produced under the random_state=233
    per-stratum 80/20 -> 75/25 split for *both* MCLDNN and IQFormer.  E1
    therefore evaluates both local comparators under that same split so that
    ``our implementation vs. published/reference`` is a same-protocol
    comparison for both rows of Table II.
    """
    return iqformer_split(labels, snrs, classes, snr_grid)


def iqformer_split(labels: np.ndarray, snrs: np.ndarray, classes: list[str], snr_grid: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match IQFormer's two sklearn splits, each random_state=233."""
    train, validation, test = [], [], []
    for label in range(len(classes)):
        for snr in snr_grid:
            indices = np.flatnonzero((labels == label) & (snrs == snr))
            target = labels[indices]
            train_indices, test_indices = train_test_split(
                indices, test_size=0.2, random_state=233, stratify=target
            )
            train_indices, validation_indices = train_test_split(
                train_indices, test_size=0.25, random_state=233,
                stratify=labels[train_indices],
            )
            train.append(train_indices); validation.append(validation_indices); test.append(test_indices)
    return np.concatenate(train), np.concatenate(validation), np.concatenate(test)


def split_for(model_name: str, labels: np.ndarray, snrs: np.ndarray, classes: list[str], snr_grid: list[int]):
    if model_name == "mcldnn":
        return mcldnn_split(labels, snrs, classes, snr_grid)
    return iqformer_split(labels, snrs, classes, snr_grid)


def loader(samples: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(samples[indices]), torch.from_numpy(labels[indices]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval(); criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = total = correct = 0
    targets, predictions = [], []
    for features, labels in data_loader:
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        scores = logits_of(model, features)
        total_loss += float(criterion(scores, labels).item())
        predicted = scores.argmax(dim=1)
        correct += int((predicted == labels).sum().item()); total += int(labels.numel())
        targets.append(labels.cpu().numpy()); predictions.append(predicted.cpu().numpy())
    y_true, y_pred = np.concatenate(targets), np.concatenate(predictions)
    return total_loss / total, correct / total, float(f1_score(y_true, y_pred, average="macro", zero_division=0)), y_true, y_pred


def train_one(model_name: str, seed: int, samples: np.ndarray, labels: np.ndarray, snrs: np.ndarray, classes: list[str], snr_grid: list[int], epochs: int, device: torch.device, output: Path) -> dict:
    seed_everything(seed)
    train_indices, validation_indices, test_indices = split_for(model_name, labels, snrs, classes, snr_grid)
    if not (len(train_indices) == 132000 and len(validation_indices) == len(test_indices) == 44000):
        raise AssertionError("E1 split no longer is the declared 600/200/200 per stratum")
    batch_size = 400 if model_name == "mcldnn" else 256
    model = MCLDNNReimplementation(len(classes)) if model_name == "mcldnn" else IQFormerInspiredClassifier(len(classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if model_name == "mcldnn" else torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8 if model_name == "mcldnn" else 0.5,
        patience=5 if model_name == "mcldnn" else 3, min_lr=1e-7 if model_name == "mcldnn" else 5e-5,
    )
    patience = 60 if model_name == "mcldnn" else 10
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(samples, labels, train_indices, batch_size, True)
    validation_loader = loader(samples, labels, validation_indices, batch_size, False)
    test_loader = loader(samples, labels, test_indices, batch_size, False)
    best_value = float("inf") if model_name == "mcldnn" else -float("inf")
    best_state, wait, history = None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = total = 0
        for features, target in train_loader:
            features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits_of(model, features), target)
            loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * len(target); total += len(target)
        validation_loss, validation_accuracy, validation_f1, _, _ = evaluate(model, validation_loader, device)
        scheduler.step(validation_loss)
        choice = validation_loss if model_name == "mcldnn" else validation_accuracy
        improved = choice < best_value if model_name == "mcldnn" else choice > best_value
        history.append({"model": model_name, "seed": seed, "epoch": epoch, "train_loss": total_loss / total, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_f1, "learning_rate": optimizer.param_groups[0]["lr"], "best": int(improved)})
        print(f"{model_name} seed={seed} epoch={epoch}/{epochs} val_acc={validation_accuracy:.4f} val_loss={validation_loss:.4f}", flush=True)
        if improved:
            best_value, wait = choice, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"{model_name} seed={seed}: early stop at epoch {epoch}", flush=True); break
    model.load_state_dict(best_state)
    test_loss, test_accuracy, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    checkpoint = output / "checkpoints" / f"{model_name}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_name, "seed": seed, "classes": classes, "state_dict": best_state}, checkpoint)
    per_snr = []
    test_snr = snrs[test_indices]
    for snr in snr_grid:
        mask = test_snr == snr
        per_snr.append({"model": model_name, "seed": seed, "snr_db": snr, "n": int(mask.sum()), "accuracy": float((y_true[mask] == y_pred[mask]).mean()), "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))})
    return {"metric": {"model": model_name, "seed": seed, "epochs_completed": len(history), "test_loss": test_loss, "test_accuracy": test_accuracy, "test_macro_f1": test_f1, "checkpoint": str(checkpoint.relative_to(output)), "selection": "min_validation_loss" if model_name == "mcldnn" else "max_validation_accuracy", "fidelity_status": "reference_available; comparison reported in Table II"}, "history": history, "per_snr": per_snr, "split_indices": {"train": train_indices.tolist(), "validation": validation_indices.tolist(), "test": test_indices.tolist()}}


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "RML2016.10a_dict.pkl")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "rml2016_10a_fidelity_v1")
    parser.add_argument("--models", nargs="+", choices=["mcldnn", "iqformer"], default=["mcldnn", "iqformer"])
    parser.add_argument("--mcldnn-epochs", type=int, default=200)
    parser.add_argument("--iqformer-epochs", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Algorithm seeds applied to every selected model; if exactly one value per selected model, applied per model. Defaults to the public protocol seeds (2016/233).")
    args = parser.parse_args()
    if args.run_dir.exists():
        if any(args.run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing E1 evidence: {args.run_dir}")
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    if not args.data.exists(): raise FileNotFoundError(args.data)
    if args.seeds is None:
        seeds_by_model = {"mcldnn": [2016], "iqformer": [233]}
    elif len(args.seeds) == len(args.models):
        seeds_by_model = {model_name: [seed] for model_name, seed in zip(args.models, args.seeds)}
    else:
        seeds_by_model = {model_name: list(args.seeds) for model_name in args.models}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics, all_history, all_per_snr, all_splits = [], [], [], {}
    for position, model_name in enumerate(args.models):
        classes = MCLDNN_CLASSES if model_name == "mcldnn" else IQFORMER_CLASSES
        samples, labels, snrs, snr_grid = load_data(args.data, classes)
        for seed in seeds_by_model[model_name]:
            result = train_one(model_name, seed, samples, labels, snrs, classes, snr_grid, args.mcldnn_epochs if model_name == "mcldnn" else args.iqformer_epochs, device, args.run_dir)
            all_metrics.append(result["metric"]); all_history.extend(result["history"]); all_per_snr.extend(result["per_snr"]); all_splits[f"{model_name}_seed{seed}"] = result["split_indices"]
    write_rows(args.run_dir / "metrics.csv", all_metrics); write_rows(args.run_dir / "training_history.csv", all_history); write_rows(args.run_dir / "per_snr.csv", all_per_snr)
    provenance = {"purpose": "E1 external comparator-fidelity check only; not VIMD external validation", "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": {"path": str(args.data.resolve()), "sha256": sha256(args.data), "shape": "220 strata x 1000 x [2,128]", "classes_mcldnn": MCLDNN_CLASSES, "classes_iqformer": IQFORMER_CLASSES, "snr_db": list(range(-20, 20, 2))}, "protocol": {"mcldnn": {"split": "per-stratum 80/20 then 75/25 train_test_split, each random_state=233 (= 600/200/200), aligned to the machine-readable IQFormer benchmark reference table", "optimizer": "Adam(1e-3)", "checkpoint": "minimum validation loss", "max_epochs": args.mcldnn_epochs}, "iqformer": {"split": "two train_test_split calls, each random_state=233, per-stratum 600/200/200", "optimizer": "AdamW(1e-3)", "checkpoint": "maximum validation accuracy", "max_epochs": args.iqformer_epochs}}, "reference": {"source": "official IQFormer repository benchmark result table (testA.xlsx): per-SNR accuracy on RML2016.10a for AMC-NET/FEA-T/MCLDNN/PET-CGDNN/IQFormer", "url": "https://github.com/WestdoorSad/IQFormer/blob/main/testA.xlsx", "mcldnn_overall_all_snr": 0.6205, "iqformer_overall_all_snr": 0.6419, "status": "machine-readable reference verified (xlsx parsed with openpyxl; overall = mean over the 20 SNR bins of per-SNR accuracy)"}, "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}, "source_hashes": {"runner": sha256(Path(__file__)), "baselines": sha256(ROOT / "src" / "vimd_amc" / "models" / "baselines.py")}}
    (args.run_dir / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (args.run_dir / "split_indices.json").write_text(json.dumps(all_splits), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
