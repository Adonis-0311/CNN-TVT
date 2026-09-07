"""Tests for the isolated cochannel source-exchange counterexample."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vimd_amc.diagnostics.cochannel_identifiability import (
    CounterexampleConfig,
    build_source_swap_counterexample,
    run_counterexample_suite,
    validate_counterexample,
)


RUNNER = ROOT / "diagnostics" / "run_cochannel_identifiability.py"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class SourceExchangeCounterexampleTest(unittest.TestCase):
    def test_swapped_roles_share_observation_but_require_different_labels(self) -> None:
        result = build_source_swap_counterexample(
            CounterexampleConfig(power_gap_db=0.5)
        )
        audit = result["exchange_audit"]
        assignments = result["latent_assignments"]
        self.assertTrue(audit["observations_equal"])
        self.assertTrue(audit["bitwise_checksum_equal"])
        self.assertEqual(audit["maximum_absolute_error"], 0.0)
        self.assertTrue(audit["required_labels_differ"])
        self.assertTrue(audit["inside_predeclared_ambiguous_band"])
        self.assertEqual(
            assignments[0]["observation_sha256"],
            assignments[1]["observation_sha256"],
        )
        self.assertNotEqual(
            assignments[0]["required_target_label"],
            assignments[1]["required_target_label"],
        )
        self.assertEqual(
            result["identifiability_result"][
                "minimum_average_error_probability_mixture_only"
            ],
            0.5,
        )
        self.assertFalse(result["headline_evidence_eligible"])
        self.assertFalse(result["paper_performance_claim_allowed"])

    def test_suite_is_deterministic_and_inside_ambiguous_band(self) -> None:
        first = run_counterexample_suite()
        second = run_counterexample_suite()
        self.assertEqual(first, second)
        self.assertTrue(first["all_checks_passed"])
        self.assertEqual(len(first["examples"]), 3)

    def test_validator_fails_closed_on_label_collision(self) -> None:
        result = build_source_swap_counterexample(CounterexampleConfig())
        result["latent_assignments"][1]["required_target_label"] = result[
            "latent_assignments"
        ][0]["required_target_label"]
        with self.assertRaisesRegex(ValueError, "different target labels"):
            validate_counterexample(result)

    def test_runner_persists_checksummed_non_evidence_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vimd_ident_test_") as temporary:
            output = Path(temporary) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--output",
                    str(output),
                    "--sample-length",
                    "128",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertTrue(Path(completed.stdout.strip()).samefile(output))
            result_path = output / "counterexample.json"
            checksums_path = output / "checksums.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["all_checks_passed"])
            self.assertFalse(payload["headline_evidence_eligible"])
            self.assertEqual(
                checksums["files"]["counterexample.json"],
                _sha256(result_path),
            )
            self.assertEqual(
                payload["protocol_options"]["dominant_emitter"][
                    "minimum_power_gap_db"
                ],
                3.0,
            )


if __name__ == "__main__":
    unittest.main()
