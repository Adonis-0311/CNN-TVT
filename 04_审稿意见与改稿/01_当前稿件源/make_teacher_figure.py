"""Render an auditable physical-teacher example from an immutable TDL cache.

This figure visualizes a deterministic cache record and the fixed teacher
defined by the implementation.  It is a method illustration, not learned-model
performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.spectral import ComplexSTFT, PhysicalTriMaskTeacher  # noqa: E402
from vimd_amc.standards import CachedPairedAMCDataset  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "standards" / "cache_factor_micro_v4",
    )
    parser.add_argument("--split", default="hard_interference")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--view", type=int, choices=(1, 2), default=1)
    parser.add_argument("--n-fft", type=int, default=32)
    parser.add_argument("--hop-length", type=int, default=8)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "paper" / "figures" / "physical_teacher_example",
    )
    return parser.parse_args()


def shifted(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(values, axes=0)


def power_db(values: torch.Tensor) -> np.ndarray:
    power = values.abs().square().squeeze(0).cpu().numpy()
    reference = max(float(power.max()), np.finfo(np.float32).tiny)
    return shifted(10.0 * np.log10(np.maximum(power / reference, 1e-8)))


def main() -> None:
    arguments = parse_arguments()
    manifest = json.loads(
        (arguments.cache_root / "manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["records"][arguments.split][arguments.index]
    metadata = record["views"][arguments.view - 1]
    with CachedPairedAMCDataset(
        arguments.cache_root,
        arguments.split,
        verify_checksums=True,
    ) as dataset:
        view = dataset[arguments.index][f"view{arguments.view}"]
        tensors = {
            name: view[name].unsqueeze(0).float()
            for name in ("x", "clean", "jammer", "unexplained")
        }

    config = ModelConfig(
        n_fft=arguments.n_fft,
        hop_length=arguments.hop_length,
    )
    transform = ComplexSTFT(config.n_fft, config.hop_length)
    teacher = PhysicalTriMaskTeacher(config)
    with torch.no_grad():
        spectra = {name: transform(value) for name, value in tensors.items()}
        decomposition = teacher.decompose(
            tensors["clean"],
            tensors["jammer"],
            tensors["unexplained"],
        )
    masks = decomposition["masks"].squeeze(0).cpu().numpy()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(7.16, 3.18),
        constrained_layout=True,
    )
    component_names = (
        ("x", "Received mixture"),
        ("clean", "Tracked target"),
        ("jammer", "Tracked jammer"),
        ("unexplained", "Noise + receiver artifact"),
    )
    image = None
    for axis, (name, title) in zip(axes[0], component_names):
        image = axis.imshow(
            power_db(spectra[name]),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-80.0,
            vmax=0.0,
            extent=(0, spectra[name].shape[-1] - 1, -0.5, 0.5),
        )
        axis.set_title(title)
        axis.set_xlabel("STFT frame")
        axis.set_ylabel("Normalized frequency")
    assert image is not None
    figure.colorbar(
        image,
        ax=axes[0].tolist(),
        label="Relative power (dB)",
        shrink=0.82,
        pad=0.01,
    )

    route_titles = (
        "Target-power\n" + r"dominant $M_s^\star$",
        "Jammer-power\n" + r"dominant $M_j^\star$",
        "Unexplained-or-\n" + r"power-ambiguous $M_o^\star$",
    )
    mask_image = None
    for route, (axis, title) in enumerate(zip(axes[1, :3], route_titles)):
        mask_image = axis.imshow(
            shifted(masks[route]),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=(0, masks.shape[-1] - 1, -0.5, 0.5),
        )
        axis.set_title(title, fontsize=7.5)
        axis.set_xlabel("STFT frame")
        axis.set_ylabel("Normalized frequency")
    assert mask_image is not None
    figure.colorbar(
        mask_image,
        ax=axes[1, :3].tolist(),
        label="Teacher allocation",
        shrink=0.82,
        pad=0.01,
    )

    summary = axes[1, 3]
    summary.axis("off")
    summary.text(
        0.02,
        0.98,
        "\n".join(
            (
                "Immutable cache record",
                f"split: {arguments.split}",
                f"source index: {arguments.index}, view: {arguments.view}",
                f"modulation: {record['modulation']}",
                f"jammer: {metadata['jammer_name']}",
                f"SNR/SIR: {metadata['snr_db']:.2f}/{metadata['sir_db']:.2f} dB",
                (
                    "target/jammer TDL: "
                    f"{metadata['tdl_target_profile']}/"
                    f"{metadata['tdl_jammer_profile']}"
                ),
                "",
                "Admitted structured jammer.",
                "Teacher only; no learned",
                "prediction or performance",
                "evidence is shown.",
            )
        ),
        ha="left",
        va="top",
        linespacing=1.25,
    )

    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        arguments.output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    figure.savefig(
        arguments.output_prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(arguments.output_prefix.with_suffix(".pdf").resolve())


if __name__ == "__main__":
    main()
