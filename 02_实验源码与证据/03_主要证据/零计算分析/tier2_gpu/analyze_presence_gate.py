"""Summarize the H2-G gate on clean and jammed evaluation windows.

This is a descriptive, post-training audit required by the Tier-2
preregistration.  It never changes a cache, checkpoint, prediction bundle, or
sealed artifact.  The model is evaluated on the same view-1 mixture input used
by the standard runner and only compact per-seed quantiles are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_array(cache_root: Path, manifest_root: Path, split: str) -> np.ndarray:
    manifest_path = manifest_root / f"{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specification = manifest["files"]["x"]
    array_path = cache_root / specification["path"]
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    expected = tuple(specification["shape"])
    if array.shape != expected:
        raise RuntimeError(f"{split}: {array.shape} != manifest {expected}")
    return array


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run.resolve()
    cache_root = args.cache_root.resolve()
    run_json_path = run_root / "run.json"
    execution_context_path = run_root / "tier2_execution_context.json"
    run = json.loads(run_json_path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise RuntimeError("presence-gate run must be complete before audit")
    if not execution_context_path.is_file():
        raise RuntimeError("Tier-2 execution context is missing")
    context = json.loads(execution_context_path.read_text(encoding="utf-8"))
    if context.get("tier2_arm") != "presence_gate":
        raise RuntimeError("run is not a presence-gate arm")
    if args.out_csv.exists() or args.out_json.exists():
        raise RuntimeError("refusing to overwrite existing descriptive gate audit")

    from analysis_zero_compute.tier2_gpu.presence_gated_vimd import PresenceGatedVIMDNet
    from vimd_amc.models.common import ModelConfig

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = ModelConfig(**run["model_configuration"])
    splits = ("clean_retention", "hard_interference")
    arrays = {
        split: load_split_array(cache_root, run_root / "manifests", split)
        for split in splits
    }
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for seed in run["seeds"]:
            checkpoint = (
                run_root
                / "models"
                / f"diagnostic_vimd_v5_presence_gated_seed{int(seed)}"
                / "model.pt"
            )
            model = PresenceGatedVIMDNet(10, 9, config).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            for split, values in arrays.items():
                gate_batches = []
                for start in range(0, len(values), args.batch_size):
                    # Standard evaluation consumes view1 only.  Copying a
                    # batch makes the read-only memory map safe for torch.
                    batch = torch.from_numpy(
                        np.array(values[start : start + args.batch_size, 0], copy=True)
                    ).to(device)
                    gate_batches.append(model(batch)["presence_gate"].float().cpu().numpy())
                summary = summarize(np.concatenate(gate_batches))
                rows.append({"seed": int(seed), "split": split, **summary})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    headers = ["seed", "split", "count", "mean", "std", "q05", "q50", "q95", "minimum", "maximum"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for split in splits:
        seed_means = np.asarray([row["mean"] for row in rows if row["split"] == split])
        grouped[split] = {
            "seed_count": int(seed_means.size),
            "mean_of_seed_means": float(seed_means.mean()),
            "min_seed_mean": float(seed_means.min()),
            "max_seed_mean": float(seed_means.max()),
        }
    payload = {
        "schema": "vimd_amc.tier2.presence_gate_distribution.v1",
        "evidence_class": "exploratory_tier2_outside_frozen_confirmatory_family",
        "input_view": "view1 received-IQ mixture",
        "clean_split": "clean_retention",
        "jammed_split": "hard_interference",
        "run_id": run["run_id"],
        "run_json_sha256": sha256_file(run_json_path),
        "execution_context_sha256": sha256_file(execution_context_path),
        "cache_digest": run.get("cache_digest"),
        "model_configuration": run["model_configuration"],
        "per_seed_summary_csv": str(args.out_csv.resolve()),
        "aggregate": grouped,
    }
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
