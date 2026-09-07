"""Validate the prospective TVT freeze without executing any experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tvt_submission.formal_v2_contract import (
    DEFAULT_FREEZE,
    EXPECTED_DEFAULT_FREEZE_SHA256,
    FormalV2ContractError,
    load_contract,
    loaded_freeze_identity,
    loaded_justification_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--expected-freeze-sha256",
        help=(
            "optional exact byte identity for a non-default freeze; the "
            "repository default is always checked against its pinned SHA-256"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        contract = load_contract(arguments.freeze)
        freeze_path, freeze_digest = loaded_freeze_identity(contract)
        justification_path, justification_digest = (
            loaded_justification_identity(contract)
        )
        expected_digest = arguments.expected_freeze_sha256
        if (
            expected_digest is not None
            and freeze_digest != expected_digest.strip().lower()
        ):
            raise FormalV2ContractError(
                "formal v2 freeze SHA-256 does not match "
                "--expected-freeze-sha256"
            )
    except FormalV2ContractError as error:
        print(
            json.dumps(
                {"ok": False, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": contract["schema_version"],
                "status": contract["status"],
                "freeze": str(freeze_path),
                "freeze_sha256": freeze_digest,
                "default_freeze_sha256": EXPECTED_DEFAULT_FREEZE_SHA256,
                "freeze_sha256_bound": True,
                "prospective_statistical_design_verified": True,
                "prospective_justification": str(justification_path),
                "prospective_justification_sha256": justification_digest,
                "model_based_power_percentage_claimed": False,
                "primary_sesoi_absolute_macro_f1": contract[
                    "prospective_statistical_design"
                ]["smallest_effect_size_of_interest"][
                    "absolute_macro_f1_gain"
                ],
                "training_sources": contract["cache"][
                    "expected_split_source_counts"
                ]["train"],
                "confirmatory_seed_count": len(
                    contract["experiment"]["seeds"]
                ),
                "confirmatory_contrast_count": len(
                    contract["experiment"]["confirmatory_family"]["contrasts"]
                ),
                "hard_interference_source_clusters": contract[
                    "cache"
                ]["expected_split_source_counts"]["hard_interference"],
                "study_level_stopping_rule": contract[
                    "prospective_statistical_design"
                ]["stopping_rule"]["study_level"],
                "bootstrap_seed_separated": (
                    contract["cache"]["master_seed"]
                    != contract["experiment"]["statistics"]["bootstrap_seed"]
                ),
                "execution_started": False,
                "cuda_execution": contract.get("execution_amendment", {}).get(
                    "cuda_execution",
                    {
                        "minimum_free_gpu_mib": None,
                        "per_device_batch_size": contract["experiment"]["training"][
                            "batch_size"
                        ],
                        "use_amp": contract["experiment"]["training"]["use_amp"],
                    },
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
