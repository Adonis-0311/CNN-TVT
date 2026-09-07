"""Fail-closed tests for the formal TVT-v2 execution evidence chain."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_formal_tvt_v2 import main as formal_main
from experiments.run_learning_curve_v2 import main as learning_main
from tvt_submission.formal_v2_contract import load_contract
from tvt_submission.validate_v2_release import (
    _execution_integrity_reasons,
    _frozen_precision_classification,
)
from vimd_amc.reproducibility import (
    TVT_V2_EXECUTION_CONTROL_FILES,
    TVT_V2_SOURCE_PROFILE,
    execution_environment_reasons,
    resolve_frozen_device,
    source_tree_audit_reasons,
    source_tree_record,
    source_tree_record_matches,
)


STANDARD_ENTRYPOINT = ROOT / "experiments" / "run_standard_experiment.py"


def _bound_source_tree() -> dict[str, object]:
    return source_tree_record(
        ROOT,
        STANDARD_ENTRYPOINT,
        profile=TVT_V2_SOURCE_PROFILE,
    )


def _runtime_environment(
    source_tree: dict[str, object],
) -> dict[str, object]:
    return {
        "python": "3.12-test",
        "torch": "2-test",
        "numpy": "2-test",
        "pandas": "2-test",
        "scipy": "1-test",
        "device": "cuda",
        "device_type": "cuda",
        "device_index": None,
        "cuda_available": True,
        "cuda_runtime": "12-test",
        "gpu": "fixture-gpu",
        "source_tree": source_tree,
    }


class V2ExecutionIntegrityTest(unittest.TestCase):
    def test_frozen_device_accepts_only_exact_contract_value(self) -> None:
        self.assertEqual(resolve_frozen_device("cuda"), "cuda")
        self.assertEqual(resolve_frozen_device("cuda", "cuda"), "cuda")
        for override in ("cpu", "auto", "cuda:0", "CUDA", ""):
            with self.subTest(override=override):
                with self.assertRaisesRegex(
                    ValueError,
                    "device contract mismatch",
                ):
                    resolve_frozen_device("cuda", override)

    def test_both_v2_runners_reject_cpu_before_preflight_success(
        self,
    ) -> None:
        for entrypoint in (learning_main, formal_main):
            with self.subTest(entrypoint=entrypoint.__module__):
                output = StringIO()
                with redirect_stdout(output):
                    with self.assertRaisesRegex(
                        ValueError,
                        "device contract mismatch",
                    ):
                        entrypoint(
                            ["--preflight-only", "--device", "cpu"]
                        )
                self.assertEqual(output.getvalue(), "")

    def test_preflights_disclose_matching_device_and_source_contract(
        self,
    ) -> None:
        payloads: list[dict[str, object]] = []
        with patch.object(
            Path,
            "write_text",
            side_effect=AssertionError("preflight attempted a write"),
        ):
            for entrypoint in (learning_main, formal_main):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        entrypoint(["--preflight-only"]),
                        0,
                    )
                payloads.append(json.loads(output.getvalue()))
        for payload in payloads:
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["execution_started"])
            self.assertEqual(
                payload["device_contract"],
                {
                    "frozen": "cuda",
                    "selected": "cuda",
                    "matches": True,
                },
            )
            binding = payload["source_tree_binding"]
            self.assertEqual(binding["profile"], TVT_V2_SOURCE_PROFILE)
            self.assertEqual(binding["file_count"], len(binding["files"]))
            for relative in TVT_V2_EXECUTION_CONTROL_FILES:
                self.assertIn(relative, binding["files"])
        self.assertEqual(
            payloads[0]["source_tree_binding"],
            payloads[1]["source_tree_binding"],
        )

    def test_v2_source_profile_is_complete_and_tamper_evident(self) -> None:
        record = _bound_source_tree()
        self.assertTrue(
            source_tree_record_matches(
                record,
                ROOT,
                STANDARD_ENTRYPOINT,
                profile=TVT_V2_SOURCE_PROFILE,
            )
        )
        tampered = deepcopy(record)
        tampered["aggregate_digest"] = "0" * 64
        self.assertFalse(
            source_tree_record_matches(
                tampered,
                ROOT,
                STANDARD_ENTRYPOINT,
                profile=TVT_V2_SOURCE_PROFILE,
            )
        )
        audit = {
            "unchanged": True,
            "reason": "unchanged",
            "start": record,
            "end": record,
        }
        self.assertEqual(
            source_tree_audit_reasons(
                audit,
                ROOT,
                STANDARD_ENTRYPOINT,
                expected_record=record,
            ),
            [],
        )
        drifted_audit = deepcopy(audit)
        drifted_audit["end"] = tampered
        reasons = source_tree_audit_reasons(
            drifted_audit,
            ROOT,
            STANDARD_ENTRYPOINT,
            expected_record=record,
        )
        self.assertIn("source_tree_start_end_records_differ", reasons)

    def test_v2_source_profile_fails_closed_when_controls_are_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_source_missing_"
        ) as temporary:
            root = Path(temporary)
            entrypoint = root / "experiments" / "run_standard_experiment.py"
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(
                FileNotFoundError,
                "source binding is incomplete",
            ):
                source_tree_record(
                    root,
                    entrypoint,
                    profile=TVT_V2_SOURCE_PROFILE,
                )

    def test_runtime_and_release_integrity_reject_device_or_source_drift(
        self,
    ) -> None:
        contract = load_contract()
        source_tree = _bound_source_tree()
        environment = _runtime_environment(source_tree)
        audit = {
            "unchanged": True,
            "reason": "unchanged",
            "start": source_tree,
            "end": source_tree,
        }
        run = {
            "environment": environment,
            "source_tree_execution_audit": audit,
        }
        learning = {
            "execution_device": "cuda",
            "source_tree_binding": source_tree,
        }
        reasons, summary = _execution_integrity_reasons(
            run=run,
            learning_evidence=learning,
            contract=contract,
        )
        self.assertEqual(reasons, [])
        self.assertTrue(summary["device_contract"]["matches"])
        self.assertTrue(
            summary["source_tree_matches_learning_and_current"]
        )

        cpu_environment = {**environment, "device": "cpu"}
        reasons = execution_environment_reasons(
            cpu_environment,
            expected_device="cuda",
            expected_source_tree=source_tree,
        )
        self.assertIn("execution_device_drift", reasons)
        drifted_run = {**run, "environment": cpu_environment}
        reasons, summary = _execution_integrity_reasons(
            run=drifted_run,
            learning_evidence=learning,
            contract=contract,
        )
        self.assertIn("execution_device_drift", reasons)
        self.assertFalse(summary["device_contract"]["matches"])

        tampered_learning = deepcopy(learning)
        tampered_learning["source_tree_binding"]["aggregate_digest"] = (
            "f" * 64
        )
        reasons, summary = _execution_integrity_reasons(
            run=run,
            learning_evidence=tampered_learning,
            contract=contract,
        )
        self.assertIn(
            "source_tree_record_differs_from_bound_evidence",
            reasons,
        )
        self.assertFalse(
            summary["source_tree_matches_learning_and_current"]
        )

    def test_primary_precision_classification_uses_frozen_precedence(
        self,
    ) -> None:
        cases = (
            (
                0.02,
                -0.02,
                0.0,
                "no_positive_primary_gain_supported",
            ),
            (
                0.01,
                0.001,
                0.02,
                (
                    "statistically_positive_and_meets_preregistered_"
                    "materiality_threshold"
                ),
            ),
            (
                0.009,
                0.001,
                0.012,
                (
                    "statistically_positive_but_below_preregistered_"
                    "materiality_threshold"
                ),
            ),
            (
                0.006,
                -0.003,
                0.012,
                "inconclusive_at_prespecified_precision",
            ),
            (
                0.003,
                -0.002,
                0.009,
                "materiality_threshold_not_supported",
            ),
        )
        for point, low, high, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    _frozen_precision_classification(
                        point=point,
                        simultaneous_low=low,
                        simultaneous_high=high,
                        sesoi=0.01,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
