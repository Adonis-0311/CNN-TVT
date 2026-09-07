"""Provenance and self-verification for the zero-compute analysis products.

Records the SHA-256 of every consumed sealed input and every produced file,
and re-checks three sealed headline numbers against this pipeline's own
estimator so that a reviewer can see the reimplementation is faithful.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np

import zc_core as z

SEALED_INPUTS = (
    z.COMPOSITE / "run.json",
    z.COMPOSITE / "composite_seal.json",
    z.COMPOSITE / "recovery_inventory.json",
    z.COMPOSITE / "v2_scientific_release_gate.json",
    z.COMPOSITE / "metrics.csv",
    z.COMPOSITE / "seed_aggregates.csv",
    z.COMPOSITE / "headline_paired_statistics.csv",
    z.COMPOSITE / "ablation_paired_statistics.csv",
    z.CACHE / "manifest.json",
    z.REPO / "artifacts" / "tvt_learning_curve_v4_8gb_dual" / "learning_curve_evidence.json",
)

CHECKS = (
    ("hard_interference", 0.049107, -0.096933),
)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> list[dict]:
    import pandas as pd

    sealed = pd.read_csv(z.COMPOSITE / "headline_paired_statistics.csv")
    rows = []
    for regime in ("hard_interference", "clean_retention", "id_test", "unseen_jammer"):
        expected = sealed[(sealed.candidate == z.PROPOSED) & (sealed.regime == regime)]
        if expected.empty:
            continue
        expected_value = float(expected.macro_f1_difference.iloc[0])
        candidate = z.load_preds(z.PROPOSED, regime)
        reference = z.load_preds(z.REFERENCE, regime)
        recomputed = float(
            np.mean(
                [
                    z.macro_f1(candidate.labels, candidate.pred[s])
                    - z.macro_f1(candidate.labels, reference.pred[s])
                    for s in range(len(candidate.seeds))
                ]
            )
        )
        rows.append(
            {
                "check": f"A5 - CSSL macro-F1 on {regime}",
                "sealed_value": expected_value,
                "recomputed_value": recomputed,
                "absolute_difference": abs(expected_value - recomputed),
                "passed": bool(abs(expected_value - recomputed) < 1e-9),
            }
        )
    return rows


def main() -> None:
    z.ensure_dirs()
    products = sorted(
        [path for path in z.OUT.rglob("*") if path.is_file() and path.name != "provenance.json"]
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=z.REPO, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = ""

    record = {
        "schema": "tvt_zero_compute_analysis_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_analysis_time": commit,
        "evidence_class": "exploratory_zero_compute_reanalysis_of_sealed_predictions",
        "statement": (
            "No new simulation, no retraining, no modification of any sealed artifact. "
            "Every number is derived from the sealed prediction bundles of the 120-fit "
            "V4R composite, its frozen cache metadata, and the sealed learning-curve evidence."
        ),
        "scientific_gate_status_unchanged": {
            "scientific_evidence_passed": False,
            "submission_unlocked": False,
            "failed_gates": ["confirmatory_family_gate_failed", "clean_retention_gate_failed"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sealed_inputs": [
            {"path": str(path.relative_to(z.REPO)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in SEALED_INPUTS
            if path.exists()
        ],
        "analysis_scripts": [
            {"path": path.name, "sha256": sha256(path)}
            for path in sorted(z.OUT.parent.glob("*.py"))
        ],
        "tier2_scripts": [
            {"path": f"tier2_gpu/{path.name}", "sha256": sha256(path)}
            for path in sorted((z.OUT.parent / "tier2_gpu").glob("*"))
            if path.is_file()
        ],
        "products": [
            {"path": str(path.relative_to(z.OUT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in products
        ],
        "estimator_verification": verify(),
    }
    with open(z.OUT / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    for row in record["estimator_verification"]:
        print(f"{row['check']:45s} sealed={row['sealed_value']:+.6f} "
              f"recomputed={row['recomputed_value']:+.6f} passed={row['passed']}")
    print(f"products: {len(products)}  sealed inputs: {len(record['sealed_inputs'])}")


if __name__ == "__main__":
    main()
