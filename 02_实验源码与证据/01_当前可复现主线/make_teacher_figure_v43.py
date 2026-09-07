"""V4.3: regenerate Fig. 1 on the manuscript's own 64x61 front-end lattice.

The previous figure was rendered from ``cache_factor_micro_v4`` (64-sample
windows) with ``--n-fft 32 --hop-length 8``, giving a five-frame lattice that
contradicts the 64x61 lattice stated in the Implementation Details.  This
script reads the headline 1024-sample cache arrays directly (bypassing the
1.9 GB manifest) and renders the same panels at n_fft=64, hop=16.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.spectral import ComplexSTFT, PhysicalTriMaskTeacher  # noqa: E402

MODULATIONS = (
    "BPSK", "PI2BPSK", "QPSK", "8PSK", "16QAM",
    "64QAM", "256QAM", "GMSK", "CPFSK", "4FSK",
)
PROFILES = ("TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E")
JAMMERS = (
    "tone", "multitone", "chirp", "sweep", "pulse",
    "partial_band", "comb", "cochannel", "ofdm_like",
)


def shifted(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(values, axes=0)


def power_db(values: torch.Tensor) -> np.ndarray:
    power = values.abs().square().squeeze(0).cpu().numpy()
    reference = max(float(power.max()), np.finfo(np.float32).tiny)
    return shifted(10.0 * np.log10(np.maximum(power / reference, 1e-8)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "standards" / "cache_factor_headline_1024_v2")
    parser.add_argument("--split", default="hard_interference")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--view", type=int, choices=(1, 2), default=1)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "paper" / "figures" / "physical_teacher_example")
    arguments = parser.parse_args()

    root = arguments.cache_root / arguments.split
    v = arguments.view - 1
    i = arguments.index

    def load(name: str) -> np.ndarray:
        return np.array(np.load(root / f"{name}.npy", mmap_mode="r")[i, v])

    tensors = {
        name: torch.from_numpy(load(name)).unsqueeze(0).float()
        for name in ("x", "clean", "jammer", "unexplained")
    }
    label = int(np.load(root / "label.npy", mmap_mode="r")[i])
    snr = float(np.load(root / "snr_db.npy", mmap_mode="r")[i, v])
    sir = float(np.load(root / "sir_db.npy", mmap_mode="r")[i, v])
    tgt = int(np.load(root / "target_profile_index.npy", mmap_mode="r")[i, v])
    jam = int(np.load(root / "jammer_profile_index.npy", mmap_mode="r")[i, v])
    jam_labels = np.array(np.load(root / "jam_labels.npy", mmap_mode="r")[i, v])
    active = [JAMMERS[k] for k in np.flatnonzero(jam_labels > 0.5) if k < len(JAMMERS)]
    jammer_name = "+".join(active) if active else "none"

    config = ModelConfig(n_fft=arguments.n_fft, hop_length=arguments.hop_length)
    transform = ComplexSTFT(config.n_fft, config.hop_length)
    teacher = PhysicalTriMaskTeacher(config)
    with torch.no_grad():
        spectra = {name: transform(value) for name, value in tensors.items()}
        decomposition = teacher.decompose(
            tensors["clean"], tensors["jammer"], tensors["unexplained"]
        )
    masks = decomposition["masks"].squeeze(0).cpu().numpy()
    frames = spectra["x"].shape[-1]
    bins = spectra["x"].shape[-2]
    print(f"lattice: {bins} bins x {frames} frames")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 3.18), constrained_layout=True)
    component_names = (
        ("x", "Received mixture"),
        ("clean", "Tracked target"),
        ("jammer", "Tracked jammer"),
        ("unexplained", "Noise + receiver artifact"),
    )
    ticks = [0, (frames - 1) // 2, frames - 1]
    image = None
    for axis, (name, title) in zip(axes[0], component_names):
        image = axis.imshow(
            power_db(spectra[name]), origin="lower", aspect="auto", cmap="magma",
            vmin=-80.0, vmax=0.0, extent=(0, frames - 1, -0.5, 0.5),
        )
        axis.set_title(title)
        axis.set_xlabel("STFT frame")
        axis.set_ylabel("Normalized frequency")
        axis.set_xticks(ticks)
    assert image is not None
    figure.colorbar(image, ax=axes[0].tolist(), label="Relative power (dB)",
                    shrink=0.82, pad=0.01)

    route_titles = (
        "Target-power\n" + r"dominant $M_s^\star$",
        "Jammer-power\n" + r"dominant $M_j^\star$",
        "Unexplained-or-\n" + r"power-ambiguous $M_o^\star$",
    )
    mask_image = None
    for route, (axis, title) in enumerate(zip(axes[1, :3], route_titles)):
        mask_image = axis.imshow(
            shifted(masks[route]), origin="lower", aspect="auto", cmap="viridis",
            vmin=0.0, vmax=1.0, extent=(0, masks.shape[-1] - 1, -0.5, 0.5),
        )
        axis.set_title(title, fontsize=7.5)
        axis.set_xlabel("STFT frame")
        axis.set_ylabel("Normalized frequency")
        axis.set_xticks(ticks)
    assert mask_image is not None
    figure.colorbar(mask_image, ax=axes[1, :3].tolist(), label="Teacher allocation",
                    shrink=0.82, pad=0.01)

    summary = axes[1, 3]
    summary.axis("off")
    summary.text(
        0.02, 0.98,
        "\n".join((
            "Immutable cache record",
            f"split: {arguments.split}",
            f"source index: {i}, view: {arguments.view}",
            f"modulation: {MODULATIONS[label]}",
            f"jammer: {jammer_name}",
            f"SNR/SIR: {snr:.2f}/{sir:.2f} dB",
            f"target/jammer TDL: {PROFILES[tgt]}/{PROFILES[jam]}",
            f"lattice: {bins}x{frames} "
            f"(NFFT {arguments.n_fft}, hop {arguments.hop_length})",
            "",
            "Admitted structured jammer.",
            "Teacher only; no learned",
            "prediction or performance",
            "evidence is shown.",
        )),
        ha="left", va="top", linespacing=1.25,
    )

    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(arguments.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "lattice_bins": int(bins), "lattice_frames": int(frames),
        "n_fft": arguments.n_fft, "hop_length": arguments.hop_length,
        "modulation": MODULATIONS[label], "jammer": jammer_name,
        "snr_db": snr, "sir_db": sir,
        "target_profile": PROFILES[tgt], "jammer_profile": PROFILES[jam],
    }, indent=2))


if __name__ == "__main__":
    main()
