"""Pre-training gradient and fixed-batch learnability gates for VIMD-Net."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.data.dataset import PairedAMCDataset, Regime  # noqa: E402
from vimd_amc.data.synthesis import SignalSynthesizer, SynthesisConfig  # noqa: E402
from vimd_amc.losses import VIMDLossWeights, vimd_two_view_loss  # noqa: E402
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import PhysicalTriMaskTeacher, VIMDNet  # noqa: E402
from vimd_amc.training import seed_everything  # noqa: E402


def _move(view: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in view.items()}


def _losses(
    model: VIMDNet,
    teacher: PhysicalTriMaskTeacher,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    first = _move(batch["view1"], device)
    second = _move(batch["view2"], device)
    labels = batch["label"].to(device)
    first_output = model(first["x"])
    second_output = model(second["x"])
    losses = vimd_two_view_loss(
        first_output=first_output,
        second_output=second_output,
        first_batch=first,
        second_batch=second,
        labels=labels,
        first_teacher_mask=teacher(
            first["clean"],
            first["jammer"],
            first["unexplained"],
        ),
        second_teacher_mask=teacher(
            second["clean"],
            second["jammer"],
            second["unexplained"],
        ),
        weights=VIMDLossWeights(),
        enable_mask=True,
        enable_contrastive=True,
    )
    return losses, first_output, second_output


def _accuracy(
    first_output: dict[str, torch.Tensor],
    second_output: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> float:
    correct = (
        first_output["logits"].argmax(dim=1).eq(labels).float().mean()
        + second_output["logits"].argmax(dim=1).eq(labels).float().mean()
    ) / 2.0
    return float(correct.detach())


def _gradient_vector(
    model: VIMDNet,
    *,
    shared_only: bool,
) -> torch.Tensor:
    pieces = []
    for name, parameter in model.named_parameters():
        if shared_only and not (
            name.startswith("context_encoder") or name.startswith("tri_mask")
        ):
            continue
        pieces.append(
            (
                parameter.grad.detach().flatten().cpu()
                if parameter.grad is not None
                else torch.zeros(parameter.numel())
            )
        )
    return torch.cat(pieces) if pieces else torch.zeros(1)


def gradient_audit(
    model: VIMDNet,
    teacher: PhysicalTriMaskTeacher,
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    weights = VIMDLossWeights()
    scale = {
        "modulation": 1.0,
        "jammer": weights.jammer,
        "quality": weights.quality,
        "mask": weights.mask,
        "contrastive": weights.contrastive,
        "orthogonality": weights.orthogonality,
    }
    vectors: dict[str, torch.Tensor] = {}
    records: dict[str, object] = {}
    for loss_name in scale:
        model.zero_grad(set_to_none=True)
        losses, _, _ = _losses(model, teacher, batch, device)
        weighted = scale[loss_name] * losses[loss_name]
        weighted.backward()
        vector = _gradient_vector(model, shared_only=True)
        vectors[loss_name] = vector
        records[loss_name] = {
            "raw_loss": float(losses[loss_name].detach()),
            "weighted_loss": float(weighted.detach()),
            "shared_gradient_norm": float(vector.norm()),
        }
    ce_norm = max(float(vectors["modulation"].norm()), 1e-12)
    for loss_name, vector in vectors.items():
        records[loss_name]["shared_gradient_to_ce_ratio"] = float(vector.norm()) / ce_norm
    cosine: dict[str, float] = {}
    ce_vector = vectors["modulation"]
    for loss_name, vector in vectors.items():
        denominator = float(ce_vector.norm() * vector.norm())
        cosine[loss_name] = (
            float(torch.dot(ce_vector, vector) / denominator)
            if denominator > 0
            else float("nan")
        )
    return {"components": records, "cosine_with_modulation": cosine}


def overfit_gate(
    model: VIMDNet,
    teacher: PhysicalTriMaskTeacher,
    batch: dict[str, object],
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    labels = batch["label"].to(device)
    initial: dict[str, float] | None = None
    final: dict[str, float] | None = None
    for step in range(steps + 1):
        model.train()
        losses, first_output, second_output = _losses(model, teacher, batch, device)
        snapshot = {
            **{name: float(value.detach()) for name, value in losses.items()},
            "accuracy": _accuracy(first_output, second_output, labels),
        }
        if step == 0:
            initial = snapshot
        if step == steps:
            final = snapshot
            break
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    assert initial is not None and final is not None
    reductions = {
        name: 1.0 - final[name] / max(initial[name], 1e-12)
        for name in ("total", "mask", "jammer", "quality")
    }
    passed = (
        final["accuracy"] >= 0.95
        and reductions["total"] >= 0.50
        and reductions["mask"] >= 0.25
        and reductions["jammer"] >= 0.30
        and reductions["quality"] >= 0.30
        and all(np.isfinite(list(initial.values())))
        and all(np.isfinite(list(final.values())))
    )
    return {
        "steps": steps,
        "initial": initial,
        "final": final,
        "fractional_reductions": reductions,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "diagnostics")
    arguments = parser.parse_args()
    # Small fixed-batch convolution workloads are substantially faster and
    # more repeatable with one CPU worker than with a large OpenMP thread pool.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(20260727)
    config = ModelConfig(
        spectral_channels=16,
        environment_dim=16,
        embedding_dim=32,
        n_fft=32,
        hop_length=8,
        dropout=0.0,
    )
    dataset = PairedAMCDataset(
        synthesizer=SignalSynthesizer(SynthesisConfig(sample_length=128)),
        split="hard",
        size=8,
        regime=Regime.hard_interference(),
        master_seed=20260727,
        modulations=("BPSK", "QPSK", "8PSK", "16QAM"),
        cache_in_memory=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))

    seed_everything(20260727)
    audit_model = VIMDNet(4, 9, config).to(device)
    teacher = PhysicalTriMaskTeacher(config).to(device)
    audit = gradient_audit(audit_model, teacher, batch, device)

    seed_everything(20260727)
    overfit_model = VIMDNet(4, 9, config).to(device)
    gate = overfit_gate(overfit_model, teacher, batch, device, arguments.steps)
    payload = {
        "seed": 20260727,
        "device": str(device),
        "gradient_audit": audit,
        "overfit_gate": gate,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(path.resolve())
    if not gate["passed"]:
        raise SystemExit("fixed-batch overfit gate failed")


if __name__ == "__main__":
    main()
