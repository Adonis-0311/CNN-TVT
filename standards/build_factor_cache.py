"""Build an immutable nine-split factor-isolated MATLAB-TDL cache.

The factor policy is locked in ``factor_isolated_split_policies``.  Presets
change only split sizes and evidence designation; they do not change which
jammer, speed, or channel factors are seen or held out.
"""

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
    FACTOR_ISOLATED_SPLITS,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    factor_isolated_split_policies,
    validate_cached_components,
)


_FULL_MODULATIONS = (
    "BPSK",
    "PI2BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "256QAM",
    "GMSK",
    "CPFSK",
    "4FSK",
)
_PRESET_SIZES: dict[str, dict[str, int]] = {
    "micro": {
        "train": 1,
        "validation": 1,
        "id_test": 1,
        "hard_interference": 1,
        "unseen_jammer": 1,
        "unseen_speed": 1,
        "heldout_channel": 1,
        "combined_ood": 1,
        "clean_retention": 1,
    },
    "screening": {
        "train": 1_000,
        "validation": 200,
        "id_test": 500,
        "hard_interference": 500,
        "unseen_jammer": 500,
        "unseen_speed": 500,
        "heldout_channel": 500,
        "combined_ood": 500,
        "clean_retention": 500,
    },
    "headline": {
        "train": 10_000,
        "validation": 2_000,
        "id_test": 5_000,
        "hard_interference": 5_000,
        "unseen_jammer": 5_000,
        "unseen_speed": 5_000,
        "heldout_channel": 5_000,
        "combined_ood": 5_000,
        "clean_retention": 5_000,
    },
}
_PRESET_DESIGNATIONS = {
    "micro": "factor_protocol_micro_smoke_only",
    "screening": "screening_not_formal_tvt_evidence",
    "headline": "headline_formal_tvt_evidence",
}


def _split_override(raw: str) -> tuple[str, int]:
    try:
        split, size_text = raw.split("=", maxsplit=1)
        size = int(size_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected SPLIT=POSITIVE_INTEGER"
        ) from exc
    split = split.strip()
    if split not in FACTOR_ISOLATED_SPLITS:
        raise argparse.ArgumentTypeError(
            f"unknown split {split!r}; expected one of {FACTOR_ISOLATED_SPLITS}"
        )
    if size <= 0:
        raise argparse.ArgumentTypeError("split size must be positive")
    return split, size


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-disjoint factor-isolated nrTDL cache. The output "
            "directory must not already exist."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=tuple(_PRESET_SIZES),
        default="micro",
        help=(
            "micro is a pipeline sentinel; screening is model triage; headline "
            "is an administrative size preset, not a power justification."
        ),
    )
    parser.add_argument(
        "--split-size",
        type=_split_override,
        action="append",
        default=[],
        metavar="SPLIT=N",
        help="Override one preset split size; may be repeated.",
    )
    parser.add_argument("--sample-length", type=int, default=256)
    parser.add_argument("--guard-samples", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=20260727)
    parser.add_argument("--matlab-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--evidence-designation",
        default=None,
        help=(
            "Override the preset audit label. A label never makes a cache "
            "scientifically sufficient by itself."
        ),
    )
    parser.add_argument(
        "--print-policy-only",
        action="store_true",
        help="Validate and print the complete policy without invoking MATLAB.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TDLCacheBuildConfig:
    sizes = dict(_PRESET_SIZES[args.preset])
    sizes.update(dict(args.split_size))
    policies = factor_isolated_split_policies(sizes)
    active_jammers = tuple(
        dict.fromkeys(
            jammer
            for policy in policies
            for jammer in policy.jammer_choices
            if jammer != "none"
        )
    )
    speeds = tuple(
        sorted({speed for policy in policies for speed in policy.speeds_kmh})
    )
    config = TDLCacheBuildConfig(
        split_sizes=tuple((policy.split, policy.size) for policy in policies),
        sample_length=args.sample_length,
        guard_samples=args.guard_samples,
        master_seed=args.master_seed,
        modulations=_FULL_MODULATIONS,
        jammer_choices=active_jammers,
        train_profiles=("TDL-A", "TDL-C", "TDL-D"),
        heldout_profiles=("TDL-B", "TDL-E"),
        delay_spreads_s=(30e-9, 100e-9, 300e-9),
        speeds_kmh=speeds,
        snr_db_values=(-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0),
        sir_db_values=(-15.0, -10.0, -5.0, 0.0, 5.0, 10.0),
        evidence_designation=(
            args.evidence_designation
            or _PRESET_DESIGNATIONS[args.preset]
        ),
        split_policies=policies,
    )
    config.validate()
    return config


def main() -> None:
    args = parse_args()
    if args.matlab_timeout_s <= 0:
        raise ValueError("--matlab-timeout-s must be positive")
    config = config_from_args(args)
    policy_summary = {
        "preset": args.preset,
        "output": str(args.output.resolve()),
        "configuration": config,
        "warning": (
            "Preset sample counts are administrative. Formal use still "
            "requires a prospective power justification and all evidence gates."
        ),
    }
    if args.print_policy_only:
        from dataclasses import asdict

        print(
            json.dumps(
                {
                    **policy_summary,
                    "configuration": asdict(config),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    result = build_tdl_paired_cache(
        args.output,
        config=config,
        matlab_timeout_s=args.matlab_timeout_s,
    )
    validation: dict[str, object] = {}
    for policy in config.split_policies or ():
        with CachedPairedAMCDataset(
            result.root, policy.split, verify_checksums=True
        ) as dataset:
            validation[policy.split] = validate_cached_components(dataset)
    print(
        json.dumps(
            {
                "cache_root": str(result.root),
                "cache_digest": result.manifest["cache_digest"],
                "schema_version": result.manifest["schema_version"],
                "evidence_designation": config.evidence_designation,
                "split_roles": result.manifest["split_roles"],
                "jammer_taxonomy": result.manifest["jammer_taxonomy"],
                "quality_normalization": result.manifest[
                    "quality_normalization"
                ],
                "protocol_exclusions": result.manifest[
                    "protocol_exclusions"
                ],
                "factor_coverage": result.manifest["factor_coverage"],
                "component_validation": validation,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
