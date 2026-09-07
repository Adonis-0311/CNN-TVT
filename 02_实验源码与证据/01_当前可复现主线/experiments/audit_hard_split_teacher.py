"""Emit deterministic hard-split per-cell teacher statistics by jammer family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import (  # noqa: E402
    LocalWhitenedTriMaskTeacher,
    PhysicalTriMaskTeacher,
    ProportionalTriMaskTeacher,
)
from vimd_amc.standards import CachedPairedAMCDataset  # noqa: E402
from vimd_amc.teacher_audit import (  # noqa: E402
    hard_split_teacher_cell_statistics,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool M_s teacher cells on a cached hard_interference split and "
            "report mean, P90, and nonzero-cell fraction by jammer family."
        )
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="hard_interference")
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-sources", type=int, default=None)
    parser.add_argument(
        "--teachers",
        default="closed_form_margin",
        help=(
            "Comma-separated subset of closed_form_margin, "
            "proportional_power, local_whitened_margin."
        ),
    )
    parser.add_argument("--local-frequency-bins", type=int, default=9)
    parser.add_argument("--local-time-frames", type=int, default=5)
    parser.add_argument("--local-whitening-exponent", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics" / "hard_split_teacher_cell_stats.json",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def build_teachers(
    names: list[str],
    config: ModelConfig,
    arguments: argparse.Namespace,
) -> dict[str, torch.nn.Module]:
    available = {
        "closed_form_margin": lambda: PhysicalTriMaskTeacher(config),
        "proportional_power": lambda: ProportionalTriMaskTeacher(config),
        "local_whitened_margin": lambda: LocalWhitenedTriMaskTeacher(
            config,
            frequency_bins=arguments.local_frequency_bins,
            time_frames=arguments.local_time_frames,
            whitening_exponent=arguments.local_whitening_exponent,
        ),
    }
    unknown = set(names).difference(available)
    if unknown:
        raise ValueError(
            f"unknown teachers {sorted(unknown)}; available={sorted(available)}"
        )
    return {name: available[name]() for name in names}


def main() -> None:
    arguments = parse_arguments()
    torch.set_num_threads(arguments.cpu_threads)
    teacher_names = [
        name.strip() for name in arguments.teachers.split(",") if name.strip()
    ]
    if not teacher_names or len(set(teacher_names)) != len(teacher_names):
        raise ValueError("teachers must be a nonempty list without duplicates")
    config = ModelConfig(
        n_fft=arguments.n_fft,
        hop_length=arguments.hop_length,
    )
    teachers = build_teachers(teacher_names, config, arguments)
    with CachedPairedAMCDataset(
        arguments.cache_root,
        arguments.split,
        verify_checksums=False,
    ) as dataset:
        result = hard_split_teacher_cell_statistics(
            teachers,
            dataset,
            batch_size=arguments.batch_size,
            maximum_sources=arguments.maximum_sources,
        )
    result["audit_configuration"] = {
        "cache_root": str(arguments.cache_root.resolve()),
        "n_fft": arguments.n_fft,
        "hop_length": arguments.hop_length,
        "batch_size": arguments.batch_size,
        "maximum_sources": arguments.maximum_sources,
        "teacher_order": teacher_names,
        "cpu_threads": arguments.cpu_threads,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
