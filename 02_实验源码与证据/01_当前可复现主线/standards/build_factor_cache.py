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
    DEFAULT_MATLAB_BATCH_SIZE,
    FACTOR_ISOLATED_SPLITS,
    TVT_V2_FACTOR_SPLITS,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    cache_build_execution_preflight,
    factor_isolated_split_policies,
    factor_isolated_split_policies_v2,
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
    "learning_10k_v2": {
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
    "learning_30k_v2": {
        "train": 30_000,
        "validation": 2_000,
        "id_test": 5_000,
        "hard_interference": 5_000,
        "unseen_jammer": 5_000,
        "unseen_speed": 5_000,
        "heldout_channel": 5_000,
        "combined_ood": 5_000,
        "clean_retention": 5_000,
    },
    "headline_v2": {
        "train": 100_000,
        "validation": 2_000,
        "id_test": 5_000,
        "hard_interference": 5_000,
        "unseen_jammer": 5_000,
        "unseen_speed": 5_000,
        "heldout_channel": 5_000,
        "combined_ood": 5_000,
        "clean_retention": 5_000,
        "adc_10bit_agc": 5_000,
        "adc_12bit_agc": 5_000,
        "per_emitter_sync": 5_000,
    },
}
_PRESET_DESIGNATIONS = {
    "micro": "factor_protocol_micro_smoke_only",
    "screening": "screening_not_formal_tvt_evidence",
    "headline": "headline_formal_tvt_evidence",
    "learning_10k_v2": "learning_curve_v2_not_formal_evidence",
    "learning_30k_v2": "learning_curve_v2_not_formal_evidence",
    "headline_v2": "headline_formal_tvt_evidence_v2",
}
_RECEIVER_STRESS_SPLITS = tuple(
    split
    for split in TVT_V2_FACTOR_SPLITS
    if split not in FACTOR_ISOLATED_SPLITS
)


def _split_override(raw: str) -> tuple[str, int]:
    try:
        split, size_text = raw.split("=", maxsplit=1)
        size = int(size_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected SPLIT=POSITIVE_INTEGER"
        ) from exc
    split = split.strip()
    if split not in TVT_V2_FACTOR_SPLITS:
        raise argparse.ArgumentTypeError(
            f"unknown split {split!r}; expected one of {TVT_V2_FACTOR_SPLITS}"
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
            "preserves the historical v1 contract; learning_*_v2 and "
            "headline_v2 implement the prospective TVT revision protocol."
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
        "--matlab-batch-size",
        type=int,
        default=DEFAULT_MATLAB_BATCH_SIZE,
        help=(
            "Maximum contiguous view count per MATLAB transfer. This is an "
            "execution-only bound and does not enter the cache manifest digest."
        ),
    )
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
    args = parser.parse_args(argv)
    receiver_stress_overrides = sorted(
        {
            split
            for split, _size in args.split_size
            if split in _RECEIVER_STRESS_SPLITS
        }
    )
    if (
        args.preset != "headline_v2"
        and receiver_stress_overrides
    ):
        parser.error(
            "receiver-stress --split-size override(s) "
            f"{', '.join(receiver_stress_overrides)} require "
            "--preset headline_v2"
        )
    return args


def config_from_args(args: argparse.Namespace) -> TDLCacheBuildConfig:
    sizes = dict(_PRESET_SIZES[args.preset])
    sizes.update(dict(args.split_size))
    policies = (
        factor_isolated_split_policies_v2(sizes)
        if args.preset == "headline_v2"
        else factor_isolated_split_policies(sizes)
    )
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
    execution_preflight = cache_build_execution_preflight(
        config,
        matlab_batch_size=args.matlab_batch_size,
    )
    policy_summary = {
        "preset": args.preset,
        "output": str(args.output.resolve()),
        "configuration": config,
        "execution_preflight": execution_preflight,
        "warning": (
            "Preset sample counts are administrative. Formal use additionally "
            "requires the separately frozen, hash-verified prospective "
            "statistical justification and all evidence gates."
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
        matlab_batch_size=args.matlab_batch_size,
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
                "execution_preflight": execution_preflight,
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
