"""Static and portable-contract tests for the TVT-v2 paper release bridge."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from tvt_submission import validate_paper_build
from tvt_submission import validate_v2_paper_release as release


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MACRO_DEFINITION = re.compile(
    r"\\newcommand\{\\(?P<name>[A-Za-z]+)\}"
)


class V2PaperReleaseContractTest(unittest.TestCase):
    def test_macro_manifest_accepts_exact_runner_bound_gate(self) -> None:
        freeze_digest = release.sha256_file(release.DEFAULT_FREEZE)
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_bound_gate_"
        ) as temporary:
            root = Path(temporary)
            run_root = root / "formal_run"
            run_root.mkdir()
            run_path = run_root / "run.json"
            learning_path = root / "learning.json"
            cache_manifest = root / "cache_manifest.json"
            run = {
                "run_id": "tvt_headline_1024_10seed_v2",
                "cache_digest": "a" * 64,
            }
            run_path.write_text(
                release._canonical_json(run),
                encoding="utf-8",
            )
            learning = {"freeze_sha256": freeze_digest}
            learning_path.write_text(
                release._canonical_json(learning),
                encoding="utf-8",
            )
            cache_manifest.write_text(
                release._canonical_json({"cache_digest": "a" * 64}),
                encoding="utf-8",
            )
            scientific_gate = {
                "schema_version": release.GATE_SCHEMA,
                "passed": True,
                "scientific_evidence_passed": True,
                "submission_unlocked": False,
                "paper_release_unlocked": False,
                "reasons": [],
                "run_json_sha256": release.sha256_file(run_path),
                "learning_curve_evidence_sha256": release.sha256_file(
                    learning_path
                ),
                "cache_manifest": str(cache_manifest.resolve()),
                "cache_manifest_sha256": release.sha256_file(
                    cache_manifest
                ),
            }
            bound_gate = release.bound_scientific_gate_payload(
                gate=scientific_gate,
                freeze_path=release.DEFAULT_FREEZE.resolve(),
                freeze_digest=freeze_digest,
                learning_evidence=learning,
            )
            gate_path = run_root / "v2_scientific_release_gate.json"
            gate_path.write_text(
                release._canonical_gate_text(bound_gate),
                encoding="utf-8",
            )
            gate_digest = release.sha256_file(gate_path)
            macro_records = {
                name: {
                    "value": (
                        "CSSL-AMC supervised adaptation"
                        if name == "PrimaryReference"
                        else "1.00"
                    ),
                    "sources": [
                        {
                            "path": "run.json",
                            "sha256": release.sha256_file(run_path),
                        }
                    ],
                    "derivation": (
                        "Deterministic bound-gate regression fixture "
                        f"for macro {name}"
                    ),
                }
                for name in release.PROVENANCE_MACROS
            }
            with patch.object(
                release,
                "derive_gate",
                return_value=scientific_gate,
            ), patch.object(
                release,
                "_derive_macro_records",
                return_value=macro_records,
            ):
                manifest = release.generate_macro_manifest(
                    run_json=run_path,
                    learning_evidence=learning_path,
                )
        self.assertEqual(
            manifest["bindings"]["freeze_sha256"],
            freeze_digest,
        )
        self.assertEqual(
            manifest["bindings"]["scientific_gate_sha256"],
            gate_digest,
        )

    def test_placeholder_and_manuscript_cover_exact_v2_macro_contract(
        self,
    ) -> None:
        placeholder = (PAPER / "results_auto.tex").read_text(
            encoding="utf-8"
        )
        main = (PAPER / "main.tex").read_text(encoding="utf-8")
        defined = [
            match.group("name")
            for match in MACRO_DEFINITION.finditer(placeholder)
        ]
        self.assertEqual(len(release.PROVENANCE_MACROS), 121)
        self.assertEqual(
            set(defined),
            set(release.NON_SENTINEL_RESULT_MACROS),
        )
        self.assertEqual(len(defined), len(set(defined)))
        for name in release.NON_SENTINEL_RESULT_MACROS:
            self.assertIn(rf"\{name}", main)
        self.assertNotIn(r"\RegimeHardReference", main)
        self.assertNotIn(r"\RegimeCleanACDReference", main)
        self.assertIn(r"\RegimeHardAZero", main)
        self.assertIn(r"\RegimeCleanACDCSSL", main)
        self.assertIn("over the shared A0 backbone", main)

    def test_render_and_parse_round_trip_exact_allowlist(self) -> None:
        values = {name: "1.00" for name in release.PROVENANCE_MACROS}
        values["PrimaryReference"] = (
            "CSSL-AMC official-architecture supervised adaptation"
        )
        values["VIMDLatencyDevice"] = "NVIDIA GeForce RTX 5060 Ti"
        rendered = release.render_results_auto(
            {
                "run_id": "tvt_headline_1024_10seed_v2",
                "cache_digest": "a" * 64,
            },
            values,
        )
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_results_contract_"
        ) as temporary:
            path = Path(temporary) / "results_auto.tex"
            path.write_text(rendered, encoding="utf-8")
            parsed = release.parse_results_auto(path)
        self.assertEqual(set(parsed), set(release.ALL_MACROS))
        self.assertEqual(
            parsed[release.RELEASE_SENTINEL],
            release.RELEASE_SENTINEL_VALUE,
        )
        self.assertEqual(
            parsed["PrimaryReference"],
            values["PrimaryReference"],
        )

    def test_portable_release_lock_rejects_inventory_tampering(
        self,
    ) -> None:
        inventory = {"run.json": "a" * 64}
        payload = {
            "schema_version": release.RELEASE_LOCK_SCHEMA,
            "submission_unlocked": True,
            "paper_release_unlocked": True,
            "scientific_evidence_passed": True,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": "tvt_headline_1024_10seed_v2",
            "cache_digest": "b" * 64,
            "formal_cache_designation": release.FORMAL_DESIGNATION,
            "release_sentinel_name": release.RELEASE_SENTINEL,
            "release_sentinel_value": release.RELEASE_SENTINEL_VALUE,
            "run_json_sha256": "c" * 64,
            "scientific_gate_sha256": "d" * 64,
            "learning_curve_evidence_sha256": "e" * 64,
            "freeze_sha256": "f" * 64,
            "cache_manifest_sha256": "1" * 64,
            "macro_value_manifest_sha256": "2" * 64,
            "results_auto_sha256": "3" * 64,
            "source_inventory_sha256": hashlib.sha256(
                release._canonical_json(inventory).encode("utf-8")
            ).hexdigest(),
            "source_inventory": inventory,
            "macro_provenance": {
                name: {
                    "sources": ["run.json"],
                    "derivation": (
                        f"Deterministic fixture derivation for macro {name}"
                    ),
                }
                for name in release.PROVENANCE_MACROS
            },
        }
        release.validate_release_lock_structure(
            payload,
            results_auto_sha256="3" * 64,
        )
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "run_id",
        ):
            release.validate_release_lock_structure(
                {**payload, "run_id": "smoke_or_decoy"}
            )
        payload["source_inventory"]["run.json"] = "9" * 64
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "inventory digest",
        ):
            release.validate_release_lock_structure(payload)

    def test_portable_release_lock_rejects_noncanonical_source_paths(
        self,
    ) -> None:
        inventory = {"run.json": "a" * 64}
        payload = {
            "schema_version": release.RELEASE_LOCK_SCHEMA,
            "submission_unlocked": True,
            "paper_release_unlocked": True,
            "scientific_evidence_passed": True,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": "tvt_headline_1024_10seed_v2",
            "cache_digest": "b" * 64,
            "formal_cache_designation": release.FORMAL_DESIGNATION,
            "release_sentinel_name": release.RELEASE_SENTINEL,
            "release_sentinel_value": release.RELEASE_SENTINEL_VALUE,
            "run_json_sha256": "c" * 64,
            "scientific_gate_sha256": "d" * 64,
            "learning_curve_evidence_sha256": "e" * 64,
            "freeze_sha256": "f" * 64,
            "cache_manifest_sha256": "1" * 64,
            "macro_value_manifest_sha256": "2" * 64,
            "results_auto_sha256": "3" * 64,
            "source_inventory_sha256": release._inventory_digest(
                inventory
            ),
            "source_inventory": inventory,
            "macro_provenance": {
                name: {
                    "sources": ["run.json"],
                    "derivation": (
                        f"Deterministic fixture derivation for macro {name}"
                    ),
                }
                for name in release.PROVENANCE_MACROS
            },
        }
        release.validate_release_lock_structure(payload)
        for unsafe in (
            "../run.json",
            "models\\run.json",
            "C:/formal/run.json",
            "models/./run.json",
        ):
            with self.subTest(path=unsafe):
                tampered = {
                    **payload,
                    "source_inventory": {unsafe: "a" * 64},
                    "source_inventory_sha256": release._inventory_digest(
                        {unsafe: "a" * 64}
                    ),
                    "macro_provenance": {
                        name: {
                            "sources": [unsafe],
                            "derivation": record["derivation"],
                        }
                        for name, record in payload[
                            "macro_provenance"
                        ].items()
                    },
                }
                with self.assertRaisesRegex(
                    release.ReleaseValidationError,
                    "source inventory",
                ):
                    release.validate_release_lock_structure(tampered)
        unsorted = {
            **payload,
            "source_inventory": {
                "a.json": "4" * 64,
                "run.json": "a" * 64,
            },
        }
        unsorted["source_inventory_sha256"] = release._inventory_digest(
            unsorted["source_inventory"]
        )
        unsorted["macro_provenance"] = {
            name: {
                "sources": ["run.json", "a.json"],
                "derivation": record["derivation"],
            }
            for name, record in payload["macro_provenance"].items()
        }
        with self.assertRaisesRegex(
            release.ReleaseValidationError,
            "provenance is incomplete",
        ):
            release.validate_release_lock_structure(unsorted)

    def test_paper_build_gate_auto_selects_v2_without_changing_v1_default(
        self,
    ) -> None:
        main = (PAPER / "main.tex").read_text(encoding="utf-8")
        try:
            validate_paper_build._select_release_contract(
                (PAPER / "results_auto.tex").read_text(encoding="utf-8")
            )
            self.assertIs(
                validate_paper_build.release_contract,
                release,
            )
            self.assertEqual(
                set(validate_paper_build.PROVENANCE_MACROS),
                set(release.PROVENANCE_MACROS),
            )
            self.assertEqual(
                validate_paper_build._main_result_contract_errors(main),
                [],
            )
            identity = {
                "run_id": release.EXPECTED_RUN_ID,
                "cache_digest": "a" * 64,
            }
            result_source = release.result_source_value(identity)
            self.assertEqual(
                validate_paper_build._release_result_source_identity_errors(
                    identity,
                    {"ResultSource": result_source},
                ),
                [],
            )
            self.assertTrue(
                validate_paper_build._release_result_source_identity_errors(
                    {**identity, "cache_digest": "b" * 64},
                    {"ResultSource": result_source},
                )
            )
        finally:
            validate_paper_build._configure_release_contract(
                validate_paper_build.LEGACY_RELEASE_CONTRACT
            )
        self.assertIs(
            validate_paper_build.release_contract,
            validate_paper_build.LEGACY_RELEASE_CONTRACT,
        )


if __name__ == "__main__":
    unittest.main()
