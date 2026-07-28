"""Focused tests for the paper-side public-table release contract."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest

from tvt_submission import validate_paper_build as gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_numeric_macros() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in gate.HEADLINE_NUMERIC_MACROS:
        if name.endswith("NLL"):
            values[name] = "0.50"
        elif name.endswith("ECE"):
            values[name] = "0.10"
        else:
            values[name] = "50.00"
    for regime in gate.REGIME_TOKENS:
        values[f"Regime{regime}Reference"] = "50.00"
        values[f"Regime{regime}AFive"] = "55.00"
        values[f"Regime{regime}Gain"] = "+5.00"
        values[f"Regime{regime}CILow"] = "+2.00"
        values[f"Regime{regime}CIHigh"] = "+8.00"
    values.update(
        {
            "MechanismMaskJS": "0.10",
            "MechanismThirdRouteWeightedCorrelation": "0.50",
            "MechanismTargetTransferRatio": "1.20",
            "MechanismTargetAmplificationShare": "20.00",
            "MechanismJammerLeakage": "0.30",
            "MechanismThirdRouteSpearman": "0.40",
            "MechanismThirdRoutePermutationP": "0.02",
            "OracleSpectralRatioGain": "+1.50",
            "VIMDParameters": "12345",
            "VIMDLatencyP50": "1.00",
            "VIMDLatencyP95": "2.00",
        }
    )
    return values


def _valid_lock() -> dict:
    digest = "a" * 64
    provenance = {
        name: {
            "source_artifact": f"evidence/{name}.json",
            "source_sha256": digest,
            "derivation": f"Audited deterministic derivation for {name}.",
        }
        for name in gate.PROVENANCE_MACROS
    }
    return {
        "schema_version": gate.RELEASE_LOCK_SCHEMA,
        "submission_unlocked": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "formal-fixture",
        "cache_digest": digest,
        "formal_cache_designation": gate.FORMAL_CACHE_DESIGNATION,
        "release_sentinel_name": gate.RELEASE_SENTINEL,
        "release_sentinel_value": gate.RELEASE_SENTINEL_VALUE,
        "run_json_sha256": digest,
        "results_auto_sha256": digest,
        "macro_value_manifest_sha256": digest,
        "source_gate_sha256": digest,
        "artifact_audit": {"passed": True},
        "macro_provenance": provenance,
    }


class PublicTableContractTest(unittest.TestCase):
    def test_checked_in_placeholder_has_exact_interface_and_wiring(self) -> None:
        result_text = (PROJECT_ROOT / "paper" / "results_auto.tex").read_text(
            encoding="utf-8"
        )
        macros, placeholders, errors = gate._parse_results_macros(result_text)
        self.assertEqual(errors, [])
        self.assertEqual(set(macros), set(gate.RESULT_MACROS))
        self.assertEqual(set(placeholders), set(gate.RESULT_MACROS))
        self.assertNotIn(gate.RELEASE_SENTINEL, macros)

        main_text = (PROJECT_ROOT / "paper" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertEqual(gate._main_result_contract_errors(main_text), [])
        self.assertIn("Automatic\nModulation Classification (AMC)", main_text)
        self.assertNotIn(r"\FeatureSIRGain", main_text)
        self.assertNotIn(r"\StrongestBaseline", main_text)

    def test_complete_numeric_contract_accepts_finite_atomic_values(self) -> None:
        values, errors = gate._parse_release_numbers(
            _valid_numeric_macros()
        )
        self.assertEqual(errors, [])
        self.assertEqual(set(values), set(gate.NUMERIC_RESULT_MACROS))
        self.assertEqual(len(values), 71)

    def test_numeric_contract_rejects_units_nonfinite_and_bad_ordering(
        self,
    ) -> None:
        macros = _valid_numeric_macros()
        macros["HeadlineHardAZeroAccuracy"] = "nan"
        macros["RegimeUnseenSpeedCILow"] = "+9.00"
        macros["RegimeUnseenSpeedCIHigh"] = "+8.00"
        macros["VIMDParameters"] = "12,345"
        macros["VIMDLatencyP50"] = "3.00 ms"
        _, errors = gate._parse_release_numbers(macros)
        joined = "\n".join(errors)
        self.assertIn("HeadlineHardAZeroAccuracy", joined)
        self.assertIn("RegimeUnseenSpeed: CI lower", joined)
        self.assertIn("VIMDParameters", joined)
        self.assertIn("VIMDLatencyP50", joined)

    def test_release_lock_v2_binds_every_reportable_value(self) -> None:
        lock = _valid_lock()
        self.assertEqual(gate._release_lock_state_errors(lock), [])
        self.assertEqual(gate._release_lock_identity_errors(lock), [])
        self.assertEqual(gate._release_lock_provenance_errors(lock), [])

        lock["macro_provenance"].pop("RegimeCombinedOODGain")
        errors = gate._release_lock_provenance_errors(lock)
        self.assertTrue(errors)
        self.assertIn("RegimeCombinedOODGain", errors[0])

    def test_legacy_or_literal_table_contract_fails_closed(self) -> None:
        result_text = (PROJECT_ROOT / "paper" / "results_auto.tex").read_text(
            encoding="utf-8"
        )
        result_text += r"\newcommand{\StrongestBaseline}{legacy}" + "\n"
        _, _, errors = gate._parse_results_macros(result_text)
        self.assertTrue(any("StrongestBaseline" in item for item in errors))

        main_text = (PROJECT_ROOT / "paper" / "main.tex").read_text(
            encoding="utf-8"
        )
        tampered = main_text.replace(
            r"\HeadlineHardAZeroAccuracy",
            "generated",
            1,
        )
        wiring_errors = gate._main_result_contract_errors(tampered)
        joined = "\n".join(wiring_errors)
        self.assertIn("HeadlineHardAZeroAccuracy", joined)
        self.assertIn("literal pending/generated", joined)


if __name__ == "__main__":
    unittest.main()
