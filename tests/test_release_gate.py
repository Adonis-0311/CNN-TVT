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
        run_root.mkdir(parents=True)
        seeds = list(self.runner.FORMAL_ABLATION_ALGORITHM_SEEDS)
        for name in (
            "metrics.csv",
            "paired_statistics.csv",
            "headline_paired_statistics.csv",
            "seed_aggregates.csv",
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
        results = []
        for seed in seeds:
            model_root = run_root / "models" / f"model_seed{seed}"
            model_root.mkdir(parents=True)
            (model_root / "model.pt").write_bytes(b"checkpoint")
            with zipfile.ZipFile(
                model_root / "predictions_validation.npz",
                mode="w",
            ) as archive:
                archive.writestr("dummy.npy", b"prediction")
            result = {
                "model": "model",
                "seed": seed,
                "checkpoint": f"models/model_seed{seed}/model.pt",
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
            results.append(result)
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
            "seeds": seeds,
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
            "results": results,
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

    def _artifact_derived_fixture(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, unittest.TestCase]:
        from tests.test_macro_generator import MacroGeneratorTest
        from tvt_submission import generate_macro_values as generator

        generator_fixture = MacroGeneratorTest()
        generator_fixture.runner = self.runner
        run_json, macro_manifest = generator_fixture._fixture(root)
        generator.write_macro_manifest(
            run_json=run_json,
            output=macro_manifest,
        )
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
        return run_json, paper_root, macro_manifest, generator_fixture

    def _write_macro_manifest(
        self,
        path: Path,
        run_json: Path,
        run_record: dict,
        *,
        placeholder_macro: str | None = None,
    ) -> None:
        records = {}
        contrast_values = {
            f"{prefix}Gain": "+1.50"
            for prefix in self.validator.ABLATION_CONTRAST_PREFIXES
        }
        contrast_values.update(
            {
                f"{prefix}CILow": "+0.50"
                for prefix in self.validator.ABLATION_CONTRAST_PREFIXES
            }
        )
        contrast_values.update(
            {
                f"{prefix}CIHigh": "+2.50"
                for prefix in self.validator.ABLATION_CONTRAST_PREFIXES
            }
        )
        for index, name in enumerate(self.validator.PROVENANCE_MACROS):
            value = (
                "--"
                if name == placeholder_macro
                else contrast_values[name]
                if name in contrast_values
                else "50.00"
                if name in self.validator.ABLATION_MEAN_PROVENANCE_MACROS
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
                "hard_ablation_family": {
                    "passed": True,
                    "family_id": "hard_macro_f1_ablation_family_v1",
                    "regime": "hard_interference",
                    "metric": "macro_f1",
                    "direction": "candidate_minus_reference",
                    "confidence_level": 0.95,
                    "multiplicity_method": (
                        "joint_max_absolute_centered_deviation_"
                        "hierarchical_paired_bootstrap"
                    ),
                    "simultaneous_ci95_low_strictly_greater_than_pp": 0.0,
                    "contrasts": {
                        contrast_id: {
                            "reference": reference,
                            "candidate": candidate,
                            "gain_pp": 1.5,
                            "marginal_ci95_low_pp": 0.75,
                            "marginal_ci95_high_pp": 2.25,
                            "simultaneous_ci95_low_pp": 0.5,
                            "simultaneous_ci95_high_pp": 2.5,
                            "passed": True,
                        }
                        for contrast_id, reference, candidate in (
                            self.validator.HARD_ABLATION_CONTRASTS
                        )
                    },
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
        self.assertEqual(len(self.validator.PROVENANCE_MACROS), 97)
        self.assertEqual(len(set(self.validator.PROVENANCE_MACROS)), 97)
        self.assertEqual(len(self.validator.NON_SENTINEL_RESULT_MACROS), 98)
        self.assertEqual(len(self.validator.ALL_MACROS), 99)
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
        self.assertEqual(
            len(self.validator.ABLATION_MEAN_PROVENANCE_MACROS),
            6,
        )
        self.assertEqual(
            len(self.validator.ABLATION_CONTRAST_PROVENANCE_MACROS),
            18,
        )
        self.assertTrue(
            all(
                name.isalpha()
                for name in (
                    *self.validator.ABLATION_MEAN_PROVENANCE_MACROS,
                    *self.validator.ABLATION_CONTRAST_PROVENANCE_MACROS,
                )
            )
        )

    def test_source_run_rejects_noncanonical_cache_digest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_cache_digest_"
        ) as temporary:
            run_json, _, _ = self._fixture(Path(temporary))
            run_record = self.validator.load_strict_json(run_json)
            cache_manifest_path = (
                Path(run_record["cache_root"]) / "manifest.json"
            )
            cache_manifest = self.validator.load_strict_json(
                cache_manifest_path
            )
            run_record["cache_digest"] = "A" * 64
            cache_manifest["cache_digest"] = "A" * 64
            self._write_json(cache_manifest_path, cache_manifest)
            self._write_json(run_json, run_record)

            with self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "run.json cache_digest is not a lowercase SHA-256",
            ):
                self.validator.validate_source_run(run_json)

    def test_release_rederives_hardened_ablation_evidence(
        self,
    ) -> None:
        from tvt_submission import generate_macro_values as generator
        import numpy as np

        cases = (
            "zero_critical",
            "nonfinite_critical",
            "wrong_confidence",
            "unstratified",
            "wrong_seed_ids",
            "snr_pair_drift",
            "sir_pair_drift",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                prefix=f"vimd_release_ablation_{case}_"
            ) as temporary:
                root = Path(temporary)
                (
                    run_json,
                    paper_root,
                    macro_manifest,
                    generator_fixture,
                ) = self._artifact_derived_fixture(root)
                run_root = run_json.parent
                ablation_csv = (
                    run_root / "ablation_paired_statistics.csv"
                )
                if case in {
                    "zero_critical",
                    "nonfinite_critical",
                    "wrong_confidence",
                    "unstratified",
                    "wrong_seed_ids",
                }:
                    column, value = {
                        "zero_critical": (
                            "simultaneous_critical_value",
                            "0",
                        ),
                        "nonfinite_critical": (
                            "simultaneous_critical_value",
                            "nan",
                        ),
                        "wrong_confidence": (
                            "confidence_level",
                            "0.90",
                        ),
                        "unstratified": (
                            "bootstrap_stratified_by_class",
                            "false",
                        ),
                        "wrong_seed_ids": (
                            "algorithm_seed_ids",
                            "['17', '29', '43', '101', '71']",
                        ),
                    }[case]
                    generator_fixture._rewrite_ablation_value(
                        ablation_csv,
                        contrast_id="teacher",
                        column=column,
                        value=value,
                    )
                else:
                    candidate = next(
                        record["candidate"]
                        for record in generator.ABLATION_CONTRASTS
                        if record["contrast_id"] == "teacher"
                    )
                    bundle_path = (
                        run_root
                        / "models"
                        / f"{candidate}_seed17"
                        / "predictions_hard_interference.npz"
                    )
                    with np.load(bundle_path, allow_pickle=False) as archive:
                        arrays = {
                            name: archive[name]
                            for name in archive.files
                        }
                    field = "snr_db" if case == "snr_pair_drift" else "sir_db"
                    arrays[field] = arrays[field].copy()
                    arrays[field][0] += 0.25
                    np.savez(bundle_path, **arrays)

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

    def test_release_rebuild_rejects_coherently_reordered_formal_seeds(
        self,
    ) -> None:
        import csv

        with tempfile.TemporaryDirectory(
            prefix="vimd_release_coherent_seed_reorder_"
        ) as temporary:
            root = Path(temporary)
            (
                run_json,
                _,
                _,
                generator_fixture,
            ) = self._artifact_derived_fixture(root)
            run_record = self.validator.load_strict_json(run_json)
            reordered = [29, 17, 43, 71, 101]
            run_record["seeds"] = reordered
            generator_fixture._write_json(run_json, run_record)

            ablation_csv = (
                run_json.parent / "ablation_paired_statistics.csv"
            )
            with ablation_csv.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                reader = csv.DictReader(stream)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            for row in rows:
                row["algorithm_seed_ids"] = repr(
                    [str(seed) for seed in reordered]
                )
            generator_fixture._write_csv(
                ablation_csv,
                fieldnames,
                rows,
            )

            with self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "ordered formal ablation algorithm seeds",
            ):
                self.validator.rebuild_expected_macro_manifest(run_json)

    def test_ablation_family_gate_is_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_ablation_family_"
        ) as temporary:
            _, _, macro_manifest = self._fixture(Path(temporary))
            base_gate = self._expected_manifest(macro_manifest)[
                "scientific_release_gate"
            ]

            def extra_family_key(gate: dict) -> None:
                gate["hard_ablation_family"]["unexpected"] = True

            def missing_contrast(gate: dict) -> None:
                del gate["hard_ablation_family"]["contrasts"]["teacher"]

            def reverse_direction(gate: dict) -> None:
                record = gate["hard_ablation_family"]["contrasts"]["teacher"]
                record["reference"], record["candidate"] = (
                    record["candidate"],
                    record["reference"],
                )

            def drift_method(gate: dict) -> None:
                gate["hard_ablation_family"]["multiplicity_method"] = (
                    "marginal_percentile"
                )

            def drift_threshold(gate: dict) -> None:
                gate["hard_ablation_family"][
                    "simultaneous_ci95_low_strictly_greater_than_pp"
                ] = -0.01

            def nonpositive_simultaneous_low(gate: dict) -> None:
                gate["hard_ablation_family"]["contrasts"]["teacher"][
                    "simultaneous_ci95_low_pp"
                ] = 0.0

            def point_outside_interval(gate: dict) -> None:
                gate["hard_ablation_family"]["contrasts"]["teacher"][
                    "marginal_ci95_high_pp"
                ] = 1.0

            mutations = (
                extra_family_key,
                missing_contrast,
                reverse_direction,
                drift_method,
                drift_threshold,
                nonpositive_simultaneous_low,
                point_outside_interval,
            )
            for mutate in mutations:
                with self.subTest(mutation=mutate.__name__):
                    gate = json.loads(json.dumps(base_gate))
                    mutate(gate)
                    with self.assertRaises(
                        self.validator.ReleaseValidationError
                    ):
                        self.validator._validate_scientific_release_gate(gate)

    def test_ablation_macro_numbers_are_bound_to_family_gate(self) -> None:
        mutations = {
            "gate_disagreement": (
                "AblationTeacherGain",
                "+999.99",
            ),
            "point_outside_ci": (
                "AblationTeacherCILow",
                "+2.00",
            ),
            "non_atomic": (
                "AblationTeacherGain",
                "1e0",
            ),
            "mean_out_of_range": (
                "HeadlineHardAOneMacroFOne",
                "101.00",
            ),
        }
        for label, (macro_name, value) in mutations.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory(
                prefix=f"vimd_release_ablation_macro_{label}_"
            ) as temporary:
                run_json, _, macro_manifest = self._fixture(Path(temporary))
                manifest = self._expected_manifest(macro_manifest)
                manifest["macros"][macro_name]["value"] = value
                macro_manifest.write_text(
                    self.validator.canonical_macro_manifest_text(manifest),
                    encoding="utf-8",
                )
                with self.assertRaises(
                    self.validator.ReleaseValidationError
                ):
                    self.validator.validate_macro_manifest(
                        macro_manifest,
                        run_json=run_json,
                        run_record=self.validator.load_strict_json(run_json),
                    )

    def test_near_zero_ablation_is_ineligible_at_public_precision(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_release_ablation_public_precision_"
        ) as temporary:
            run_json, paper_root, macro_manifest = self._fixture(
                Path(temporary)
            )
            manifest = self._expected_manifest(macro_manifest)
            family = manifest["scientific_release_gate"][
                "hard_ablation_family"
            ]
            teacher = family["contrasts"]["teacher"]
            teacher["simultaneous_ci95_low_pp"] = 0.004
            teacher["passed"] = True
            manifest["macros"]["AblationTeacherCILow"]["value"] = "+0.00"
            macro_manifest.write_text(
                self.validator.canonical_macro_manifest_text(manifest),
                encoding="utf-8",
            )

            self.assertGreater(
                teacher["simultaneous_ci95_low_pp"],
                family[
                    "simultaneous_ci95_low_strictly_greater_than_pp"
                ],
            )
            self.assertEqual(
                self.validator._public_pp_number(
                    teacher["simultaneous_ci95_low_pp"]
                ),
                0.0,
            )
            self.assertEqual(
                manifest["run_json_sha256"],
                self.validator.sha256_file(run_json),
            )
            before = (paper_root / "results_auto.tex").read_bytes()
            with mock.patch.object(
                self.validator,
                "rebuild_expected_macro_manifest",
                return_value=manifest,
            ), self.assertRaisesRegex(
                self.validator.ReleaseValidationError,
                "two-decimal public rendering",
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
                        / f"model_seed{run_record['seeds'][0]}"
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
