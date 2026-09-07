"""Reproduce the manuscript A5 MAC count and profile the frozen capacity controls.

This audit instantiates the existing A5, Width M, Width L, and received-I/Q
sidecar architectures at the manuscript input length. It uses the same forward
hook counter as the sealed complexity table. No checkpoint, training data, or
evaluation sample is required, and no model is trained or evaluated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

CONFIG_PATH = (
    REPO / "tvt_submission" / "configs" / "formal_tvt_recovery_v4r_110plus10.json"
)
OUTPUT_PATH = REPO / "analysis_zero_compute" / "outputs" / "v55_capacity_compute_audit.json"
SAMPLE_LENGTH = 1024
NUM_CLASSES = 10
NUM_JAMMERS = 9

# spectral_channels, embedding_dim, environment_dim
TIERS = {
    "A5": (24, 48, 32),
    "Width M": (40, 80, 48),
    "Width L": (64, 128, 64),
}

EXPECTED = {
    "A5": {"parameters": 39_500, "macs": 41_798_528},
    "I/Q sidecar": {"parameters": 46_794, "macs": 43_119_008},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_config(spectral_channels: int, embedding_dim: int, environment_dim: int):
    from vimd_amc.models.common import ModelConfig

    frozen = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["scientific_invariants"]
    return ModelConfig(
        feature_channels=max(32, spectral_channels),
        environment_dim=environment_dim,
        embedding_dim=embedding_dim,
        spectral_channels=spectral_channels,
        n_fft=int(frozen["n_fft"]),
        hop_length=int(frozen["hop_length"]),
        dropout=float(frozen["dropout"]),
    )


def _models() -> dict[str, torch.nn.Module]:
    from analysis_zero_compute.tier2_gpu.run_tier2_experiment import (
        LightweightIQSidecarVIMD,
    )
    from experiments.run_standard_experiment import available_model_factories

    factory = available_model_factories()["a5_vimd_full"]
    models = {}
    for name, dimensions in TIERS.items():
        models[name] = factory(
            NUM_CLASSES,
            NUM_JAMMERS,
            _model_config(*dimensions),
        ).model
    models["I/Q sidecar"] = LightweightIQSidecarVIMD(
        NUM_CLASSES,
        NUM_JAMMERS,
        _model_config(*TIERS["A5"]),
    )
    return models


def run(device_name: str) -> dict:
    from vimd_amc.evaluation import complexity_metrics

    device = torch.device(device_name)
    rows = []
    for name, model in _models().items():
        metrics = complexity_metrics(
            model,
            sample_length=SAMPLE_LENGTH,
            device=device,
            latency_runs=1,
        )
        macs = int(metrics["conv_linear_recurrent_macs_excluding_stft"])
        parameters = int(metrics["parameters"])
        rows.append(
            {
                "model": name,
                "parameters": parameters,
                "conv_linear_recurrent_macs_excluding_stft": macs,
                "macs_million": macs / 1_000_000,
                "stft_estimated_real_operations": int(
                    metrics["stft_estimated_real_operations"]
                ),
                "dimensions": list(TIERS.get(name, TIERS["A5"])),
            }
        )

    indexed = {row["model"]: row for row in rows}
    for name, expected in EXPECTED.items():
        observed = indexed[name]
        if observed["parameters"] != expected["parameters"]:
            raise RuntimeError(
                f"{name} parameter gate failed: {observed['parameters']} != "
                f"{expected['parameters']}"
            )
        if observed["conv_linear_recurrent_macs_excluding_stft"] != expected["macs"]:
            raise RuntimeError(
                f"{name} MAC gate failed: "
                f"{observed['conv_linear_recurrent_macs_excluding_stft']} != "
                f"{expected['macs']}"
            )

    dependencies = [
        Path(__file__),
        CONFIG_PATH,
        REPO / "src" / "vimd_amc" / "evaluation.py",
        REPO / "analysis_zero_compute" / "a21_s6_severe_held_controls.py",
        REPO
        / "analysis_zero_compute"
        / "tier2_gpu"
        / "run_tier2_experiment.py",
    ]
    payload = {
        "purpose": "TVT V5.5 matched-capacity compute audit",
        "counter": "vimd_amc.evaluation.complexity_metrics forward-hook MAC counter",
        "sample_length": SAMPLE_LENGTH,
        "device": str(device),
        "torch_version": torch.__version__,
        "a5_gate_passed": True,
        "sidecar_crosscheck_passed": True,
        "models": rows,
        "dependency_sha256": {
            str(path.relative_to(REPO)): _sha256(path) for path in dependencies
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()
    payload = run(args.device)
    for row in payload["models"]:
        print(
            f"{row['model']}: params={row['parameters']:,}; "
            f"MACs={row['macs_million']:.6f} M"
        )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
