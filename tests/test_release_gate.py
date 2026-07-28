"""Fail-closed tests for the TVT paper-result release boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "tvt_submission" / "validate_release.py"


def _load_validator():
    specification = importlib.util.spec_from_file_location(
        "vimd_release_validator_test_module",
        VALIDATOR_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load release validator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ReleaseGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = _load_validator()
        cls.runner = cls.validator._load_runner_module()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        cache_root = root / "formal_cache"
        self._write_json(
            cache_root / "manifest.json",
            {
                "cache_digest": "a" * 64,
                "configuration": {
                    "evidence_designation": (
                        self.validator.FORMAL_DESIGNATION
                    )
                },
            },
        )
        run_root = root / "formal_run"
        model_root = run_root / "models" / "model_seed1"
        model_root.mkdir(parents=True)
        (model_root / "model.pt").write_bytes(b"checkpoint")
        with zipfile.ZipFile(
            model_root / "predictions_validation.npz",
            mode="w",
        ) as archive:
            archive.writestr("dummy.npy", b"prediction")
        for name in (
            "metrics.csv",
            "paired_statistics.csv",
            "headline_paired_statistics.csv",
        ):
            (run_root / name).write_text("value\n1\n", encoding="utf-8")
        for name in ("cache_reference", "train", "validation"):
            self._write_json(
                run_root / "manifests" / f"{name}.json",
                {"status": "present"},
            )
        criterion = {
            "view": "view1",
            "loss": "modulation_cross_entropy",
            "label_smoothing": 0.0,
            "direction": "minimize",
            "strict_improvement_tolerance": 1e-5,
            "patience_eligible_epochs": 1,
            "auxiliary_losses_included": False,
            "validation_loader_shuffle": False,
        }
        result = {
            "model": "model",
            "seed": 1,
            "checkpoint": "models/model_seed1/model.pt",
            "teacher_checkpoint": None,
            "training": {
                "history": [
                    {
                        "epoch": 1.0,
                        "checkpoint_selection_eligible": 1.0,
                        "validation_loss": 0.25,
                    }
                ],
                "checkpoint_selection": {
                    "status": "eligible_validation_checkpoint_selected",
                    "selected_checkpoint_eligible": True,
                    "fallback_used": False,
                    "selected_epoch": 1,
                    "selected_validation_loss": 0.25,
                    "eligible_checkpoint_count": 1,
                    "criterion": criterion,
                },
            },
            "regimes": {
                "validation": {
                    "accuracy": 0.8,
                    "macro_f1": 0.79,
                    "worst_recall": 0.70,
                    "nll": 0.5,
                    "ece": 0.03,
                }
            },
        }
        self._write_json(model_root / "result.json", result)
        run_record = {
            "run_id": "formal_test_run",
            "runner": "experiments/run_standard_experiment.py",
            "status": "complete",
            "execution_status": "complete",
            "cache_root": str(cache_root.resolve()),
            "cache_digest": "a" * 64,
            "checksums_verified": True,
            "models": ["model"],
            "seeds": [1],
            "splits": {"train": 1, "validation": 1},
            "training_configuration": {
                "epochs": 1,
                "mask_start_epoch": 0,
                "contrastive_start_epoch": 0,
                "mask_ramp_epochs": 1,
                "contrastive_ramp_epochs": 1,
                "minimum_full_stage_epochs": 1,
                "patience": 1,
            },
            "results": [result],
            "evidence_eligibility": {
                "policy_version": "vimd-evidence-gate-v2",
                "cache_designation": self.validator.FORMAL_DESIGNATION,
                "eligible": True,
                "formal_paper_evidence_eligible": True,
                "headline_eligible": True,
                "gates": {
                    name: {"passed": True}
                    for name in (
                        self.runner.FORMAL_RELEASE_REQUIRED_GATES
                    )
                },
            },
            "source_tree_execution_audit": {"unchanged": True},
            "statistical_outputs": {
                "single_seed_pairs": "paired_statistics.csv",
                "multi_seed_headline_pairs": (
                    "headline_paired_statistics.csv"
                ),
                "holm_families": [],
            },
        }
        run_record["submission_release"] = (
            self.runner.submission_release_source_gate(run_record)
        )
        run_json = run_root / "run.json"
        self._write_json(run_json, run_record)

        paper_root = root / "paper"
        paper_root.mkdir()
        (paper_root / "main.tex").write_text(
            "\\documentclass{article}\n",
            encoding="utf-8",
        )
        (paper_root / "results_auto.tex").write_text(
            "\\newcommand{\\ResultSource}{No eligible locked run}\n",
            encoding="utf-8",
        )
        macro_manifest = root / "macro_values.json"
        self._write_macro_manifest(macro_manifest, run_json, run_record)
        return run_json, paper_root, macro_manifest

    def _write_macro_manifest(
        self,
        path: Path,
        run_json: Path,
        run_record: dict,
        *,
        placeholder_macro: str | None = None,
    ) -> None:
        records = {}
        for index, name in enumerate(self.validator.PROVENANCE_MACROS):
            value = (
                "--"
                if name == placeholder_macro
                else "CSSL-AMC supervised adaptation"
                if name == "PrimaryReference"
                else f"{index + 1}.0"
            )
            records[name] = {
                "value": value,
                "source_artifact": "metrics.csv",
                "derivation": (
                    f"Deterministic fixture derivation for macro {name}"
                ),
            }
        manifest = {
            "schema_version": self.validator.MACRO_MANIFEST_SCHEMA,
            "run_id": run_record["run_id"],
            "cache_digest": run_record["cache_digest"],
            "run_json_sha256": self.validator.sha256_file(run_json),
            "scientific_release_gate": {
                "passed": True,
                "hard_gain_pp_each_nonoracle_baseline": {
                    "a0_backbone": 6.0,
                    "mcldnn_reimplementation": 6.0,
                    "iqformer_inspired": 6.0,
                    "cssl_amc_supervised_adaptation": 6.0,
                },
                "hard_ablation_gain_pp": {
                    "a1_single_mask": 1.0,
                    "a6_dual_full": 1.0,
                },
                "ood_gain_pp": {
                    "unseen_jammer": 4.0,
                    "unseen_speed": 4.0,
                    "heldout_channel": 0.0,
                },
                "ood_pass_count": 2,
                "clean_noninferiority": {
                    "clean_retention_seen_acd": {
                        "gain_pp": 0.0,
                        "ci95_low_pp": -1.0,
                    },
                    "clean_retention_held_be": {
                        "gain_pp": 0.0,
                        "ci95_low_pp": -1.0,
                    },
                },
                "mechanism_means": {
                    "mask_js": 0.1,
                    "overlap_uncertainty_route_weighted_correlation": 0.2,
                    "target_energy_transfer_ratio_mean": 1.1,
                    "target_energy_transfer_ratio_amplification_share": 0.3,
                    "jammer_leakage": 0.1,
                    "oracle_vs_predicted_overlap_spearman": 0.2,
                    "overlap_permutation_p_value": 0.01,
                    "counterfactual_tf_sir_gain_db": 1.0,
                },
            },
            "macros": records,
        }
        path.write_text(
            self.validator.canonical_macro_manifest_text(manifest),
            encoding="utf-8",
        )

    def _expected_manifest(self, path: Path) -> dict:
        return self.validator.load_strict_json(path)

    def test_macro_contract_matches_generator_exactly(self) -> None:
        from tvt_submission import generate_macro_values as generator

        self.assertEqual(
            tuple(self.validator.PROVENANCE_MACROS),
            tuple(generator.PROVENANCE_MACROS),
        )
        self.assertEqual(
            self.validator.RESULT_MACROS,
            self.validator.PROVENANCE_MACROS,
        )
        self.assertEqual(len(self.validator.PROVENANCE_MACROS), 73)
        self.assertEqual(len(set(self.validator.PROVENANCE_MACROS)), 73)
        self.assertEqual(len(self.validator.NON_SENTINEL_RESULT_MACROS), 74)
        self.assertEqual(len(self.validator.ALL_MACROS), 75)
        self.assertEqual(
            self.validator.ALL_MACROS,
            (
                self.validator.RELEASE_SENTINEL,
                "ResultSource",
                *self.validator.PROVENANCE_MACROS,
            ),
        )
        self.assertTrue(
            all(
                name.startswith("Regime")
                for name in self.validator.REGIME_PROVENANCE_MACROS
            )
        )
        self.assertFalse(
            any(
                name.startswith("OOD")
                for name in self.validator.PROVENANCE_MACROS
            )
        )

    def test_placeholder_detection_uses_token_boundaries(self) -> None:
        self.assertFalse(
            self.validator._is_placeholder(
                "Arithmetic mean of mcldnn_reimplementation/accuracy"
            )
        )
        for placeholder in ("n/a", "NaN", "pending", "generated"):
            with self.subTest(placeholder=placeholder):
                self.assertTrue(
                    self.validator._is_placeholder(placeholder)
                )

    def test_eligible_release_writes_and_revalidates_lock(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_success_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            expected = self._expected_manifest(macro_manifest)
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=expected,
            ):
                released = self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
                self.assertTrue(released["submission_unlocked"])
                macros = self.validator.parse_results_auto(
                    paper_root / "results_auto.tex"
                )
                self.assertEqual(set(macros), set(self.validator.ALL_MACROS))
                self.assertEqual(
                    macros[self.validator.RELEASE_SENTINEL],
                    self.validator.RELEASE_SENTINEL_VALUE,
                )
                self.assertNotEqual(macros["RegimeHardGain"], "--")
                lock = self.validator.load_strict_json(
                    paper_root / "release_lock.json"
                )
                self.assertTrue(lock["submission_unlocked"])
                self.assertEqual(
                    lock["release_sentinel_name"],
                    self.validator.RELEASE_SENTINEL,
                )
                self.assertEqual(
                    lock["release_sentinel_value"],
                    self.validator.RELEASE_SENTINEL_VALUE,
                )
                self.assertEqual(
                    lock["run_json_sha256"],
                    self.validator.sha256_file(run_json),
                )
                validated = self.validator.validate_existing_release(
                    run_json=run_json,
                    paper_root=paper_root,
                )
                self.assertTrue(validated["submission_unlocked"])

    def test_placeholder_manifest_cannot_write_or_unlock(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_placeholder_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            run_record = self.validator.load_strict_json(run_json)
            self._write_macro_manifest(
                macro_manifest,
                run_json,
                run_record,
                placeholder_macro="RegimeHardGain",
            )
            before = (paper_root / "results_auto.tex").read_bytes()
            with self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "placeholder",
            ):
                self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
            self.assertEqual(
                (paper_root / "results_auto.tex").read_bytes(),
                before,
            )
            self.assertFalse((paper_root / "release_lock.json").exists())

    def test_ineligible_or_fallback_run_cannot_generate_macros(self) -> None:
        for mutation in ("ineligible", "fallback"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"vimd_release_{mutation}_"
            ) as temporary:
                run_json, paper_root, macro_manifest = self._fixture(
                    Path(temporary)
                )
                run_record = self.validator.load_strict_json(run_json)
                if mutation == "ineligible":
                    run_record["evidence_eligibility"]["eligible"] = False
                else:
                    selection = run_record["results"][0]["training"][
                        "checkpoint_selection"
                    ]
                    selection["selected_checkpoint_eligible"] = False
                    selection["fallback_used"] = True
                    self._write_json(
                        run_json.parent
                        / "models"
                        / "model_seed1"
                        / "result.json",
                        run_record["results"][0],
                    )
                self._write_json(run_json, run_record)
                self._write_macro_manifest(
                    macro_manifest,
                    run_json,
                    run_record,
                )
                before = (paper_root / "results_auto.tex").read_bytes()
                with self.assertRaises(
                    self.validator.ReleaseValidationError
                ):
                    self.validator.write_release(
                        run_json=run_json,
                        paper_root=paper_root,
                        macro_values=macro_manifest,
                    )
                self.assertEqual(
                    (paper_root / "results_auto.tex").read_bytes(),
                    before,
                )
                self.assertFalse(
                    (paper_root / "release_lock.json").exists()
                )

    def test_existing_placeholder_without_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_existing_"
        ) as temporary:
            run_json, paper_root, _ = self._fixture(Path(temporary))
            with self.assertRaises(self.validator.ReleaseValidationError):
                self.validator.validate_existing_release(
                    run_json=run_json,
                    paper_root=paper_root,
                )

    def test_internal_placeholder_cannot_forge_release_with_sentinel(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_forged_sentinel_"
        ) as temporary:
            run_json, paper_root, _ = self._fixture(Path(temporary))
            results = paper_root / "results_auto.tex"
            results.write_text(
                (
                    "\\newcommand{\\EligibleLockedResults}"
                    "{eligible_locked_formal_run}\n"
                    "\\newcommand{\\ResultSource}{No eligible locked run}\n"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(self.validator.ReleaseValidationError):
                self.validator.validate_existing_release(
                    run_json=run_json,
                    paper_root=paper_root,
                )

    def test_wrong_release_sentinel_is_rejected_by_parser(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_wrong_sentinel_"
        ) as temporary:
            _, paper_root, _ = self._fixture(Path(temporary))
            values = {
                name: "value"
                for name in self.validator.RESULT_MACROS
            }
            rendered = self.validator.render_results_auto(
                {"run_id": "fixture", "cache_digest": "a" * 64},
                values,
            ).replace(
                self.validator.RELEASE_SENTINEL_VALUE,
                "forged_internal_file",
                1,
            )
            results = paper_root / "results_auto.tex"
            results.write_text(rendered, encoding="utf-8")
            with self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "eligible-release sentinel",
            ):
                self.validator.parse_results_auto(results)

    def test_manual_macro_value_is_rejected_before_release_write(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_manual_value_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            expected = self._expected_manifest(macro_manifest)
            tampered = json.loads(json.dumps(expected))
            tampered["macros"]["RegimeHardGain"]["value"] = "+999.99"
            macro_manifest.write_text(
                self.validator.canonical_macro_manifest_text(tampered),
                encoding="utf-8",
            )
            before = (paper_root / "results_auto.tex").read_bytes()
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=expected,
            ), self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "deterministic artifact re-derivation",
            ):
                self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
            self.assertEqual(
                (paper_root / "results_auto.tex").read_bytes(),
                before,
            )
            self.assertFalse((paper_root / "release_lock.json").exists())

    def test_scientific_gate_tamper_is_rejected_before_release_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_gate_tamper_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            expected = self._expected_manifest(macro_manifest)
            tampered = json.loads(json.dumps(expected))
            tampered["scientific_release_gate"]["mechanism_means"][
                "mask_js"
            ] = 999.0
            macro_manifest.write_text(
                self.validator.canonical_macro_manifest_text(tampered),
                encoding="utf-8",
            )
            before = (paper_root / "results_auto.tex").read_bytes()
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=expected,
            ), self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "deterministic artifact re-derivation",
            ):
                self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
            self.assertEqual(
                (paper_root / "results_auto.tex").read_bytes(),
                before,
            )
            self.assertFalse((paper_root / "release_lock.json").exists())

    def test_existing_release_rejects_tamper_even_if_lock_hash_is_updated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_existing_tamper_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            expected = self._expected_manifest(macro_manifest)
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=expected,
            ):
                self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
                results_path = paper_root / "results_auto.tex"
                original = results_path.read_text(encoding="utf-8")
                original_value = expected["macros"]["RegimeHardGain"][
                    "value"
                ]
                tampered = original.replace(
                    (
                        r"\newcommand{\RegimeHardGain}{"
                        f"{original_value}"
                        "}"
                    ),
                    r"\newcommand{\RegimeHardGain}{+1234.56}",
                    1,
                )
                self.assertNotEqual(tampered, original)
                results_path.write_text(tampered, encoding="utf-8")
                lock_path = paper_root / "release_lock.json"
                lock = self.validator.load_strict_json(lock_path)
                lock["results_auto_sha256"] = self.validator.sha256_file(
                    results_path
                )
                lock_path.write_text(
                    json.dumps(
                        lock,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    self.validator.ReleaseValidationError,
                    "deterministic artifact re-derivation",
                ):
                    self.validator.validate_existing_release(
                        run_json=run_json,
                        paper_root=paper_root,
                    )

    def test_existing_release_lock_binds_scientific_gate_exactly(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_existing_gate_tamper_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            expected = self._expected_manifest(macro_manifest)
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=expected,
            ):
                self.validator.write_release(
                    run_json=run_json,
                    paper_root=paper_root,
                    macro_values=macro_manifest,
                )
                tampered = json.loads(json.dumps(expected))
                tampered["scientific_release_gate"]["ood_pass_count"] = 0
                tampered_digest = hashlib.sha256(
                    self.validator.canonical_macro_manifest_text(
                        tampered
                    ).encode("utf-8")
                ).hexdigest()
                lock_path = paper_root / "release_lock.json"
                lock = self.validator.load_strict_json(lock_path)
                lock["macro_value_manifest_sha256"] = tampered_digest
                lock_path.write_text(
                    json.dumps(
                        lock,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    self.validator.ReleaseValidationError,
                    "manifest digest disagrees",
                ):
                    self.validator.validate_existing_release(
                        run_json=run_json,
                        paper_root=paper_root,
                    )

    def test_results_parser_rejects_every_non_contract_line(self) -> None:
        values = {
            name: (
                "CSSL-AMC supervised adaptation"
                if name == "PrimaryReference"
                else "1.0"
            )
            for name in self.validator.PROVENANCE_MACROS
        }
        rendered = self.validator.render_results_auto(
            {"run_id": "fixture", "cache_digest": "a" * 64},
            values,
        )
        injections = (
            r"\input{untrusted-extra.tex}",
            r"\newcommand{\UnexpectedMacro}{1}",
            "% user-supplied unbound comment",
        )
        for injection in injections:
            with self.subTest(injection=injection), tempfile.TemporaryDirectory(
                prefix="vimd_release_parser_injection_"
            ) as temporary:
                results = Path(temporary) / "results_auto.tex"
                results.write_text(
                    rendered + injection + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(self.validator.ReleaseValidationError):
                    self.validator.parse_results_auto(results)

    def test_parser_accepts_spelled_latency_macros_and_rejects_digits(
        self,
    ) -> None:
        values = {
            name: (
                "CSSL-AMC supervised adaptation"
                if name == "PrimaryReference"
                else "1.0"
            )
            for name in self.validator.PROVENANCE_MACROS
        }
        rendered = self.validator.render_results_auto(
            {"run_id": "fixture", "cache_digest": "a" * 64},
            values,
        )
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_parser_latency_"
        ) as temporary:
            results = Path(temporary) / "results_auto.tex"
            results.write_text(rendered, encoding="utf-8")
            parsed = self.validator.parse_results_auto(results)
            self.assertEqual(parsed["VIMDLatencyPFifty"], "1.0")
            self.assertEqual(parsed["VIMDLatencyPNinetyFive"], "1.0")

            for valid_name, invalid_name in (
                ("VIMDLatencyPFifty", "VIMDLatencyP50"),
                ("VIMDLatencyPNinetyFive", "VIMDLatencyP95"),
            ):
                with self.subTest(invalid_name=invalid_name):
                    results.write_text(
                        rendered.replace(valid_name, invalid_name, 1),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        self.validator.ReleaseValidationError,
                        "executable or unknown",
                    ):
                        self.validator.parse_results_auto(results)


if __name__ == "__main__":
    unittest.main()
