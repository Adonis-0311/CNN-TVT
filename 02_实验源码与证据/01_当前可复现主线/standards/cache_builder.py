"""Build the small audited MATLAB-TDL cache used for integration validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.standards import (  # noqa: E402
    CachedPairedAMCDataset,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    validate_cached_components,
)


def _csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return values


def _csv_floats(raw: str) -> tuple[float, ...]:
    tokens = _csv_strings(raw)
    try:
        return tuple(float(value) for value in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of numbers"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "standards" / "cache_smoke",
    )
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--guard-samples", type=int, default=48)
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--validation-size", type=int, default=1)
    parser.add_argument("--heldout-size", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260727)
    parser.add_argument(
        "--modulations",
        type=_csv_strings,
        help="Comma-separated modulation list (for example BPSK,QPSK,16QAM).",
    )
    parser.add_argument(
        "--jammers",
        type=_csv_strings,
        help="Comma-separated active jammer list; 'none' is not permitted.",
    )
    parser.add_argument(
        "--train-profiles",
        type=_csv_strings,
        help="Comma-separated training/validation TDL profiles.",
    )
    parser.add_argument(
        "--heldout-profiles",
        type=_csv_strings,
        help="Comma-separated held-out TDL profiles.",
    )
    parser.add_argument(
        "--speeds-kmh",
        type=_csv_floats,
        help="Comma-separated nonnegative terminal speeds in km/h.",
    )
    parser.add_argument(
        "--delay-spreads-ns",
        type=_csv_floats,
        help="Comma-separated positive RMS delay spreads in nanoseconds.",
    )
    parser.add_argument(
        "--snr-db-values",
        "--snr-db",
        dest="snr_db_values",
        type=_csv_floats,
        help=(
            "Comma-separated discrete SNR values in dB. When omitted, the "
            "unchanged default smoke range is sampled continuously."
        ),
    )
    parser.add_argument(
        "--sir-db-values",
        "--sir-db",
        dest="sir_db_values",
        type=_csv_floats,
        help=(
            "Comma-separated discrete SIR values in dB. When omitted, the "
            "unchanged default smoke range is sampled continuously."
        ),
    )
    parser.add_argument(
        "--evidence-designation",
        default="integration_smoke_only",
        help="Audit label written verbatim to the immutable manifest.",
    )
    parser.add_argument(
        "--matlab-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for each of the two batched MATLAB channel calls.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TDLCacheBuildConfig:
    """Translate CLI values into one validated, fully manifested config."""
    defaults = TDLCacheBuildConfig()
    delay_spreads_s = (
        tuple(value * 1e-9 for value in args.delay_spreads_ns)
        if args.delay_spreads_ns is not None
        else defaults.delay_spreads_s
    )
    config = TDLCacheBuildConfig(
        split_sizes=(
            ("train", args.train_size),
            ("validation", args.validation_size),
            ("heldout_channel", args.heldout_size),
        ),
        sample_length=args.sample_length,
        guard_samples=args.guard_samples,
        master_seed=args.master_seed,
        modulations=args.modulations or defaults.modulations,
        jammer_choices=args.jammers or defaults.jammer_choices,
        train_profiles=args.train_profiles or defaults.train_profiles,
        heldout_profiles=args.heldout_profiles or defaults.heldout_profiles,
        delay_spreads_s=delay_spreads_s,
        speeds_kmh=args.speeds_kmh or defaults.speeds_kmh,
        snr_db_values=args.snr_db_values,
        sir_db_values=args.sir_db_values,
        evidence_designation=args.evidence_designation,
    )
    config.validate()
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.matlab_timeout_s <= 0:
        raise ValueError("--matlab-timeout-s must be positive")
    result = build_tdl_paired_cache(
        args.output,
        config=config,
        matlab_timeout_s=args.matlab_timeout_s,
    )
    validation = {}
    for split, _ in config.split_sizes:
        with CachedPairedAMCDataset(
            result.root, split, verify_checksums=True
        ) as dataset:
            validation[split] = validate_cached_components(dataset)
    summary = {
        "cache_root": str(result.root),
        "cache_digest": result.manifest["cache_digest"],
        "configuration": result.manifest["configuration"],
        "evidence_designation": result.manifest["configuration"][
            "evidence_designation"
        ],
        "source_ids": result.manifest["source_ids"],
        "profile_policy": result.manifest["profile_policy"],
        "validation": validation,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
