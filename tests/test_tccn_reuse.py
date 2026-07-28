"""Contract tests for the read-only governance-mechanism adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tvt_submission.tccn_reuse.claim_ledger import (  # noqa: E402
    ClaimLedgerError,
    validate_claim_rows,
)
from tvt_submission.tccn_reuse.data_qa import (  # noqa: E402
    DataQAError,
    assert_group_disjoint,
    verify_component_identity,
)
from tvt_submission.tccn_reuse.freeze import (  # noqa: E402
    FreezeError,
    load_frozen_config,
    sha256_file,
)
from tvt_submission.tccn_reuse.manifest import (  # noqa: E402
    ManifestError,
    atomic_write_json_new,
    validate_run_manifest,
)
from tvt_submission.tccn_reuse.publication_gate import assess_release  # noqa: E402
from tvt_submission.tccn_reuse.repository_audit import audit_repository  # noqa: E402
from tvt_submission.tccn_reuse.statistics import (  # noqa: E402
    StatisticalProtocol,
    StatisticsProtocolError,
    validate_paired_records,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_payload() -> dict[str, object]:
    return {
        "schema": "vimd_amc.tvt.freeze.v1",
        "status": "frozen_before_results",
        "experiment_id": "synthetic-contract-test",
        "seeds": [17, 29, 43],
        "split_roles": {
            "train": ["train"],
            "validation": ["validation"],
            "calibration": ["calibration"],
            "test": ["test"],
        },
        "data_manifests": [{"path": "data/manifest.json", "sha256": "a" * 64}],
    }


class FreezeTests(unittest.TestCase):
    def test_frozen_file_is_bound_to_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps(_freeze_payload()), encoding="utf-8")
            digest = sha256_file(path)
            loaded = load_frozen_config(path, expected_sha256=digest)
            self.assertEqual(loaded.experiment_id, "synthetic-contract-test")
            path.write_text(json.dumps({**_freeze_payload(), "seeds": [1]}), encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "digest mismatch"):
                load_frozen_config(path, expected_sha256=digest)

    def test_template_is_not_silently_treated_as_frozen(self) -> None:
        template = (
            REPOSITORY_ROOT
            / "tvt_submission"
            / "tccn_reuse"
            / "vehicular_experiment_freeze_template.json"
        )
        with self.assertRaisesRegex(FreezeError, "not frozen"):
            load_frozen_config(template)
        loaded = load_frozen_config(template, allow_template=True)
        self.assertEqual(loaded.payload["status"], "template_not_frozen")


class ManifestTests(unittest.TestCase):
    def test_create_new_publication_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_write_json_new(path, {"first": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                atomic_write_json_new(path, {"first": False})
            self.assertEqual(path.read_bytes(), before)

    def test_manifest_rejects_path_escape_and_checksum_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            attempt.mkdir()
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            manifest = {
                "schema": "vimd_amc.tvt.run_manifest.v1",
                "execution_status": "completed",
                "frozen_config_sha256": "a" * 64,
                "artifacts": {
                    "predictions": {"path": "../outside.bin", "sha256": _sha(outside)}
                },
            }
            path = attempt / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "escapes"):
                validate_run_manifest(path)

            artifact = attempt / "predictions.bin"
            artifact.write_bytes(b"version-one")
            manifest["artifacts"]["predictions"] = {
                "path": "predictions.bin",
                "sha256": _sha(artifact),
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ManifestError, "checksum mismatch"):
                validate_run_manifest(path)


class DataQATests(unittest.TestCase):
    def test_source_group_cannot_cross_splits(self) -> None:
        rows = [
            {"sample_id": "a", "source_id": "source-1", "split": "train"},
            {"sample_id": "b", "source_id": "source-1", "split": "test"},
        ]
        with self.assertRaisesRegex(DataQAError, "crosses splits"):
            assert_group_disjoint(rows)

    def test_component_identity_recomputes_snr_and_sir(self) -> None:
        desired = np.ones(32, dtype=np.complex64)
        interference = 0.5j * np.ones(32, dtype=np.complex64)
        noise = 0.1 * np.ones(32, dtype=np.complex64)
        result = verify_component_identity(
            desired + interference + noise,
            desired,
            interference,
            noise,
        )
        self.assertAlmostEqual(result.measured_snr_db, 20.0, places=4)
        self.assertAlmostEqual(result.measured_sir_db, 10.0 * np.log10(4.0), places=4)
        with self.assertRaisesRegex(DataQAError, "identity failed"):
            verify_component_identity(desired, desired, interference, noise)


class GovernanceTests(unittest.TestCase):
    def test_statistics_require_seed_depth_isolation_and_exact_pairing(self) -> None:
        StatisticalProtocol("screening", (17, 29, 43)).validate()
        with self.assertRaisesRegex(StatisticsProtocolError, "at least 5"):
            StatisticalProtocol("headline", (17, 29, 43)).validate()
        with self.assertRaisesRegex(StatisticsProtocolError, "sample order"):
            validate_paired_records(
                ["a", "b"],
                ["b", "a"],
                [0, 1],
                [0, 1],
                ["s1", "s2"],
                ["s1", "s2"],
            )

    def test_supported_claim_requires_named_evidence(self) -> None:
        with self.assertRaisesRegex(ClaimLedgerError, "requires evidence"):
            validate_claim_rows(
                [{"claim_id": "C-1", "status": "supported", "evidence_artifact": ""}]
            )

    def test_release_gate_passes_only_complete_closed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.npz"
            predictions.write_bytes(b"synthetic-test-predictions")
            config_digest = "b" * 64
            manifest = {
                "schema": "vimd_amc.tvt.run_manifest.v1",
                "execution_status": "completed",
                "evidence_tier": "screening",
                "frozen_config_sha256": config_digest,
                "seeds_completed": [17, 29, 43],
                "git": {"commit": "0123456789abcdef", "dirty": False},
                "qa": {
                    "sample_id_unique": True,
                    "split_group_disjoint": True,
                    "component_identity": True,
                    "duplicate_scan": True,
                },
                "statistics": {
                    "paired_by_source": True,
                    "family_preregistered": True,
                    "calibration_isolated": True,
                },
                "artifacts": {
                    "predictions": {
                        "path": predictions.name,
                        "sha256": _sha(predictions),
                    }
                },
            }
            path = root / "manifest.json"
            atomic_write_json_new(path, manifest)
            result = assess_release(
                path,
                expected_config_sha256=config_digest,
                expected_seeds=[17, 29, 43],
                required_artifacts=["predictions"],
            )
            self.assertTrue(result.eligible, result.failures)

            manifest["qa"]["duplicate_scan"] = False
            second = root / "failed_manifest.json"
            atomic_write_json_new(second, manifest)
            denied = assess_release(
                second,
                expected_config_sha256=config_digest,
                expected_seeds=[17, 29, 43],
                required_artifacts=["predictions"],
            )
            self.assertFalse(denied.eligible)

    def test_adapted_code_and_config_pass_semantic_firewall(self) -> None:
        root = REPOSITORY_ROOT / "tvt_submission" / "tccn_reuse"
        checked = [
            "freeze.py",
            "manifest.py",
            "data_qa.py",
            "statistics.py",
            "publication_gate.py",
            "vehicular_experiment_freeze_template.json",
        ]
        self.assertEqual(
            audit_repository(root, semantic_firewall_paths=checked),
            [],
        )


if __name__ == "__main__":
    unittest.main()
