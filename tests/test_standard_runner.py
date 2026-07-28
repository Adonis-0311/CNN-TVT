"""Contract tests for the offline nrTDL experiment runner."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "experiments" / "run_standard_experiment.py"
CACHE_ROOT = REPOSITORY_ROOT / "standards" / "cache_smoke"
FACTOR_CACHE_ROOT = (
    REPOSITORY_ROOT / "standards" / "cache_factor_micro_v4"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "vimd_standard_runner_test_module",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load standard runner module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class StandardRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def _formal_training_evidence(
        self,
        models: list[str],
        seeds: list[int],
    ):
        training = self.runner.TrainingConfig(
            epochs=1,
            mask_start_epoch=0,
            contrastive_start_epoch=0,
            mask_ramp_epochs=1,
            contrastive_ramp_epochs=1,
            minimum_full_stage_epochs=1,
            patience=1,
        )
        criterion = self.runner.checkpoint_selection_protocol(training)[
            "criterion"
        ]
        support_mask = [
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
        ]
        support = {
            "support_mask": support_mask,
            "supported_training_labels": [
                "tone",
                "multitone",
                "chirp",
                "sweep",
                "partial_band",
                "comb",
            ],
            "held_out_labels": ["pulse", "ofdm_like"],
            "excluded_taxonomy_labels": ["cochannel"],
            "unsupported_columns_receive_training_loss_or_gradient": False,
            "unsupported_logit_columns_receive_direct_bce_gradient": False,
            "held_or_excluded_logits_are_trained_family_recognizers": False,
            "formal_contract_valid": True,
        }
        results = []
        for model in models:
            for seed in seeds:
                uses_jammer = model == "a5_vimd_full"
                results.append(
                    {
                        "model": model,
                        "seed": seed,
                        "objective": {
                            "use_jammer_auxiliary": uses_jammer,
                        },
                        "training": {
                            "history": [
                                {
                                    "epoch": 1.0,
                                    "checkpoint_selection_eligible": 1.0,
                                    "validation_loss": 0.25,
                                }
                            ],
                            "checkpoint_selection": {
                                "status": (
                                    "eligible_validation_checkpoint_selected"
                                ),
                                "selected_checkpoint_eligible": True,
                                "fallback_used": False,
                                "selected_epoch": 1,
                                "selected_validation_loss": 0.25,
                                "eligible_checkpoint_count": 1,
                                "criterion": criterion,
                            },
                            "jammer_auxiliary_training": {
                                "enabled": uses_jammer,
                                "support_mask": support_mask,
                                "unsupported_columns_receive_loss_or_gradient": (
                                    False if uses_jammer else None
                                ),
                                "unsupported_logit_columns_receive_direct_bce_gradient": (
                                    False if uses_jammer else None
                                ),
                            },
                        },
                    }
                )
        return training, results, support

    def test_formal_ablation_rows_use_one_joint_six_contrast_family(
        self,
    ) -> None:
        from vimd_amc.metrics import PredictionBundle

        labels = np.repeat(np.arange(3, dtype=np.int64), 20)
        sources = np.arange(2_000, 2_060, dtype=np.int64)

        def bundle(errors_per_class: int) -> PredictionBundle:
            predictions = labels.copy()
            for class_index in range(3):
                selected = np.flatnonzero(labels == class_index)
                predictions[selected[:errors_per_class]] = (
                    class_index + 1
                ) % 3
            probabilities = np.full((len(labels), 3), 0.05)
            probabilities[np.arange(len(labels)), predictions] = 0.90
            return PredictionBundle(
                probabilities=probabilities,
                labels=labels,
                source_ids=sources,
                snr_db=np.full(len(labels), 5.0),
                sir_db=np.full(len(labels), -5.0),
            )

        errors = {
            "a1_single_mask": 15,
            "a2_tri_no_teacher": 18,
            "a3_tri_teacher": 12,
            "a4_tri_teacher_mtl": 6,
            "a5_vimd_full": 0,
            "a6_dual_full": 15,
            "a7_vimd_no_residual": 15,
        }
        seeds = [17, 29, 43, 71, 101]
        prediction_bundles = {
            (model, seed, self.runner.FORMAL_ABLATION_REGIME): bundle(count)
            for model, count in errors.items()
            for seed in seeds
        }
        rows, summary = self.runner.build_ablation_paired_rows(
            prediction_bundles=prediction_bundles,
            seeds=seeds,
            cache_digest="b" * 64,
            bootstrap_draws=200,
            bootstrap_seed_base=20260727,
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row["contrast_id"] for row in rows],
            [
                "teacher",
                "multitask",
                "exact_source_contrast",
                "full_vs_single",
                "full_vs_dual",
                "bypass",
            ],
        )
        self.assertEqual(
            set(rows[0]),
            set(self.runner.ABLATION_PAIRED_COLUMNS),
        )
        self.assertEqual(
            {row["bootstrap_seed"] for row in rows},
            {rows[0]["bootstrap_seed"]},
        )
        self.assertEqual(
            {row["simultaneous_critical_value"] for row in rows},
            {rows[0]["simultaneous_critical_value"]},
        )
        self.assertTrue(summary["family_gate_passed"])
        self.assertTrue(all(row["gate_passed"] for row in rows))
        self.assertEqual(
            rows[0]["algorithm_seed_ids"],
            ["17", "29", "43", "71", "101"],
        )

    def test_formal_ablation_rows_reject_seed_or_cache_identity_drift(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "exact ordered algorithm seeds"):
            self.runner.build_ablation_paired_rows(
                prediction_bundles={},
                seeds=[17, 29, 43, 71],
                cache_digest="b" * 64,
                bootstrap_draws=20,
                bootstrap_seed_base=20260727,
            )
        with self.assertRaisesRegex(ValueError, "lowercase 64-hex"):
            self.runner.build_ablation_paired_rows(
                prediction_bundles={},
                seeds=[17, 29, 43, 71, 101],
                cache_digest="B" * 64,
                bootstrap_draws=20,
                bootstrap_seed_base=20260727,
            )

    def test_formal_ablation_rows_reject_degenerate_family(self) -> None:
        from vimd_amc.metrics import PredictionBundle

        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
            ]
        )
        models = {
            model
            for contrast in self.runner.FORMAL_ABLATION_CONTRASTS
            for model in (contrast["reference"], contrast["candidate"])
        }
        seeds = list(self.runner.FORMAL_ABLATION_ALGORITHM_SEEDS)
        prediction_bundles = {
            (
                model,
                seed,
                self.runner.FORMAL_ABLATION_REGIME,
            ): PredictionBundle(
                probabilities=probabilities,
                labels=labels,
                source_ids=np.arange(len(labels)),
                snr_db=np.zeros(len(labels)),
                sir_db=np.full(len(labels), -5.0),
            )
            for model in models
            for seed in seeds
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "simultaneous_critical_value must be finite and strictly positive",
        ):
            self.runner.build_ablation_paired_rows(
                prediction_bundles=prediction_bundles,
                seeds=seeds,
                cache_digest="b" * 64,
                bootstrap_draws=20,
                bootstrap_seed_base=20260727,
            )

    def test_cache_contract_and_required_models_are_compatible(self) -> None:
        contract = self.runner.inspect_cache_contract(CACHE_ROOT)
        self.assertEqual(contract.sample_length, 128)
        self.assertEqual(contract.modulations, ("BPSK", "QPSK", "8PSK", "16QAM"))
        self.assertEqual(contract.num_jammers, 9)
        self.assertEqual(
            set(contract.split_sizes),
            {"train", "validation", "heldout_channel"},
        )

        datasets = self.runner.load_cache_datasets(
            contract,
            verify_checksums=True,
        )
        try:
            config = self.runner.make_model_config(
                sample_length=contract.sample_length,
                n_fft=32,
                hop_length=8,
                spectral_channels=16,
                embedding_dim=32,
                environment_dim=16,
                dropout=0.0,
            )
            factories = self.runner.available_model_factories()
            self.assertTrue(
                {"backbone", "single_mask", "vimd"}.issubset(factories)
            )
            batch = datasets["train"][0]["view1"]["x"].unsqueeze(0)
            for model_name in ("backbone", "single_mask", "vimd"):
                built = factories[model_name](
                    len(contract.modulations),
                    contract.num_jammers,
                    config,
                )
                built.model.eval()
                with torch.no_grad():
                    output = built.model(batch)
                self.assertEqual(
                    tuple(output["logits"].shape),
                    (1, len(contract.modulations)),
                )
            vimd = factories["vimd"](
                len(contract.modulations),
                contract.num_jammers,
                config,
            )
            self.assertEqual(vimd.objective.name, "full_vimd")
            self.assertTrue(vimd.objective.use_mask_supervision)
            self.assertTrue(vimd.objective.use_cross_condition_contrastive)
        finally:
            for dataset in datasets.values():
                dataset.close()

    def test_one_epoch_cpu_smoke_emits_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_standard_runner_"
        ) as temporary:
            command = [
                sys.executable,
                str(RUNNER_PATH),
                "--cache-root",
                str(CACHE_ROOT),
                "--models",
                "backbone,vimd",
                "--reference-model",
                "backbone",
                "--holm-candidates",
                "vimd",
                "--seeds",
                "5",
                "--device",
                "cpu",
                "--output",
                temporary,
                "--run-id",
                "test_run",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--mask-start-epoch",
                "0",
                "--contrastive-start-epoch",
                "0",
                "--mask-ramp-epochs",
                "1",
                "--contrastive-ramp-epochs",
                "1",
                "--minimum-full-stage-epochs",
                "1",
                "--patience",
                "1",
                "--n-fft",
                "32",
                "--hop-length",
                "8",
                "--spectral-channels",
                "8",
                "--embedding-dim",
                "16",
                "--environment-dim",
                "8",
                "--latency-runs",
                "1",
                "--bootstrap-draws",
                "20",
                "--verify-checksums",
                "--validate-components",
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            run_root = Path(completed.stdout.strip())
            self.assertTrue((run_root / "run.json").is_file())
            self.assertTrue((run_root / "metrics.csv").is_file())
            self.assertTrue((run_root / "paired_statistics.csv").is_file())
            self.assertTrue(
                (run_root / "headline_paired_statistics.csv").is_file()
            )
            self.assertTrue(
                (run_root / "models" / "backbone_seed5" / "model.pt").is_file()
            )
            self.assertTrue(
                (
                    run_root
                    / "models"
                    / "vimd_seed5"
                    / "predictions_heldout_channel.npz"
                ).is_file()
            )
            payload = self.runner.json.loads(
                (run_root / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["execution_status"], "complete")
            self.assertTrue(payload["checksums_verified"])
            self.assertFalse(payload["v2x_system_level_compliance_claimed"])
            self.assertFalse(payload["evidence_eligibility"]["eligible"])
            self.assertFalse(
                payload["evidence_eligibility"][
                    "eligible_for_designated_tier"
                ]
            )
            self.assertTrue(
                payload["source_tree_execution_audit"]["unchanged"]
            )
            self.assertEqual(
                payload["comparison_protocol"]["reference_selection"],
                "explicit_cli",
            )
            self.assertFalse(
                payload["comparison_protocol"]["reference_strength_claimed"]
            )
            self.assertEqual(
                payload["dataset_support_audit"]["evidence_readiness"],
                "pipeline_smoke_only_incomplete_class_support",
            )
            vimd = next(
                result
                for result in payload["results"]
                if result["model"] == "vimd"
            )
            backbone = next(
                result
                for result in payload["results"]
                if result["model"] == "backbone"
            )
            self.assertEqual(
                backbone["auxiliary_metrics"]["status"],
                "unavailable",
            )
            self.assertFalse(
                backbone["auxiliary_metrics"][
                    "used_for_checkpoint_or_model_selection"
                ]
            )
            self.assertEqual(
                vimd["auxiliary_metrics"]["status"],
                "available",
            )
            self.assertEqual(
                set(vimd["auxiliary_metrics"]["available_heads"]),
                {"jammer_multilabel", "quality"},
            )
            self.assertEqual(
                set(vimd["auxiliary_metrics"]["regimes"]),
                {"validation", "heldout_channel"},
            )
            self.assertFalse(
                vimd["auxiliary_metrics"][
                    "used_for_checkpoint_or_model_selection"
                ]
            )
            self.assertEqual(vimd["objective"]["name"], "full_vimd")
            self.assertEqual(
                vimd["training"]["history"][0]["mask_enabled"],
                1.0,
            )
            checkpoint_selection = vimd["training"][
                "checkpoint_selection"
            ]
            self.assertTrue(
                checkpoint_selection["selected_checkpoint_eligible"]
            )
            self.assertFalse(checkpoint_selection["fallback_used"])
            self.assertEqual(checkpoint_selection["selected_epoch"], 1)
            self.assertEqual(
                checkpoint_selection["criterion"]["view"],
                "view1",
            )
            self.assertFalse(
                payload["submission_release"]["macro_generation_permitted"]
            )
            paired = self.runner.pd.read_csv(
                run_root / "paired_statistics.csv"
            )
            validation = paired[paired["regime"] == "validation"].iloc[0]
            heldout = paired[
                paired["regime"] == "heldout_channel"
            ].iloc[0]
            self.assertFalse(bool(validation["holm_included"]))
            self.assertTrue(bool(heldout["holm_included"]))
            self.assertTrue(self.runner.pd.isna(validation["exact_p_value"]))
            self.assertFalse(self.runner.pd.isna(heldout["exact_p_value"]))

    def test_schema1_cache_cannot_be_promoted_to_headline(self) -> None:
        original = self.runner.inspect_cache_contract(CACHE_ROOT)
        manifest = json.loads(json.dumps(original.manifest))
        manifest["configuration"][
            "evidence_designation"
        ] = "headline_formal_tvt_evidence"
        contract = replace(original, manifest=manifest)
        support = {
            "all_splits_class_complete": True,
            "splits": {
                "train": {
                    "sample_count": 1_000,
                    "minimum_class_count": 100,
                },
                "validation": {
                    "sample_count": 200,
                    "minimum_class_count": 20,
                },
                "heldout_channel": {
                    "sample_count": 500,
                    "minimum_class_count": 50,
                },
            },
        }
        component_validation = {
            split: {
                "max_component_error": 1e-7,
                "max_snr_error_db": 1e-6,
                "max_sir_error_db": 1e-6,
                "min_active_jammer_power": 0.01,
            }
            for split in self.runner.REQUIRED_SPLITS
        }
        models = list(
            self.runner.PREREGISTERED_MODEL_SUITES["headline"]
        )
        eligible = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=True,
            component_validation=component_validation,
            models=models,
            seeds=[11, 17, 23, 29, 31],
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
        )
        self.assertFalse(eligible["eligible"])
        self.assertFalse(
            eligible["gates"]["factor_isolated_protocol"]["passed"]
        )

        no_checksum = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=False,
            component_validation=component_validation,
            models=models,
            seeds=[11, 17, 23, 29, 31],
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
        )
        self.assertFalse(no_checksum["eligible"])
        self.assertFalse(
            no_checksum["gates"]["checksum_verification"]["passed"]
        )

        too_few_seeds = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=True,
            component_validation=component_validation,
            models=models,
            seeds=[11, 17, 23],
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
        )
        self.assertFalse(too_few_seeds["eligible"])
        self.assertEqual(
            too_few_seeds["gates"]["minimum_algorithm_seeds"]["required"],
            5,
        )

    def test_factor_cache_is_dynamic_and_can_satisfy_administrative_gates(
        self,
    ) -> None:
        original = self.runner.inspect_cache_contract(FACTOR_CACHE_ROOT)
        self.assertEqual(
            tuple(original.split_sizes),
            self.runner.FACTOR_ISOLATED_SPLITS,
        )
        self.assertEqual(
            self.runner.evaluation_split_names(original),
            tuple(
                split
                for split in self.runner.FACTOR_ISOLATED_SPLITS
                if split != "train"
            ),
        )
        datasets = self.runner.load_cache_datasets(
            original,
            verify_checksums=True,
        )
        try:
            self.assertEqual(set(datasets), set(original.split_sizes))
            support_audit = self.runner.dataset_support_audit(
                datasets,
                original,
            )
            self.assertEqual(
                set(support_audit["splits"]),
                set(self.runner.FACTOR_ISOLATED_SPLITS),
            )
        finally:
            for dataset in datasets.values():
                dataset.close()

        manifest = json.loads(json.dumps(original.manifest))
        manifest["configuration"][
            "evidence_designation"
        ] = "headline_formal_tvt_evidence"
        contract = replace(original, manifest=manifest)
        support = {
            "all_splits_class_complete": True,
            "splits": {
                split: {
                    "sample_count": (
                        1_000
                        if split == "train"
                        else 200
                        if split == "validation"
                        else 500
                    ),
                    "minimum_class_count": (
                        100
                        if split == "train"
                        else 20
                        if split == "validation"
                        else 50
                    ),
                }
                for split in self.runner.FACTOR_ISOLATED_SPLITS
            },
        }
        component_validation = {
            split: {
                "max_component_error": 1e-7,
                "max_snr_error_db": 1e-6,
                "max_sir_error_db": 1e-6,
                "min_active_jammer_power": (
                    0.0 if split == "clean_retention" else 0.01
                ),
            }
            for split in self.runner.FACTOR_ISOLATED_SPLITS
        }
        models = list(
            self.runner.PREREGISTERED_MODEL_SUITES["headline"]
        )
        seeds = [11, 17, 23, 29, 31]
        training, training_results, jammer_support = (
            self._formal_training_evidence(models, seeds)
        )
        eligible = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=True,
            component_validation=component_validation,
            models=models,
            seeds=seeds,
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
            training_configuration=training,
            training_results=training_results,
            jammer_training_support=jammer_support,
        )
        self.assertTrue(eligible["eligible"])
        self.assertEqual(eligible["reasons"], [])

        mutated = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=True,
            component_validation=component_validation,
            models=models,
            seeds=seeds,
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
            source_tree_unchanged=False,
            training_configuration=training,
            training_results=training_results,
            jammer_training_support=jammer_support,
        )
        self.assertFalse(mutated["eligible"])
        self.assertIn(
            "source_tree_mutated_during_execution",
            mutated["reasons"],
        )

        fallback_results = json.loads(json.dumps(training_results))
        fallback_results[0]["training"]["checkpoint_selection"].update(
            {
                "selected_checkpoint_eligible": False,
                "fallback_used": True,
            }
        )
        fallback = self.runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=support,
            checksums_verified=True,
            component_validation=component_validation,
            models=models,
            seeds=seeds,
            execution_status="complete",
            explicit_reference_model=(
                self.runner.FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            holm_candidates=list(self.runner.FORMAL_HOLM_CANDIDATES),
            training_configuration=training,
            training_results=fallback_results,
            jammer_training_support=jammer_support,
        )
        self.assertFalse(fallback["eligible"])
        self.assertFalse(
            fallback["gates"]["eligible_selected_checkpoints"]["passed"]
        )

    def test_factor_runner_emits_physical_auxiliary_metrics(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_factor_runner_"
        ) as temporary:
            command = [
                sys.executable,
                str(RUNNER_PATH),
                "--cache-root",
                str(FACTOR_CACHE_ROOT),
                "--models",
                "vimd",
                "--seeds",
                "7",
                "--device",
                "cpu",
                "--output",
                temporary,
                "--run-id",
                "factor_run",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--mask-start-epoch",
                "0",
                "--contrastive-start-epoch",
                "0",
                "--mask-ramp-epochs",
                "1",
                "--contrastive-ramp-epochs",
                "1",
                "--minimum-full-stage-epochs",
                "1",
                "--patience",
                "1",
                "--n-fft",
                "32",
                "--hop-length",
                "8",
                "--spectral-channels",
                "8",
                "--embedding-dim",
                "16",
                "--environment-dim",
                "8",
                "--latency-runs",
                "1",
                "--bootstrap-draws",
                "20",
                "--verify-checksums",
                "--validate-components",
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            run_root = Path(completed.stdout.strip())

            def reject_nonstandard_constant(value: str) -> None:
                raise ValueError(f"non-standard JSON constant: {value}")

            with (run_root / "run.json").open(encoding="utf-8") as stream:
                payload = json.load(
                    stream,
                    parse_constant=reject_nonstandard_constant,
                )
            result = payload["results"][0]
            support = payload["jammer_auxiliary_training_support"]
            self.assertEqual(
                support["support_mask"],
                [True, False, False, False, False, False, False, False, False],
            )
            self.assertFalse(
                support["class_records"]["pulse"]["training_supported"]
            )
            self.assertFalse(
                support["class_records"]["cochannel"]["training_supported"]
            )
            self.assertFalse(
                support["class_records"]["ofdm_like"]["training_supported"]
            )
            self.assertEqual(
                result["training"]["jammer_auxiliary_training"][
                    "support_mask"
                ],
                support["support_mask"],
            )
            checkpoint_selection = result["training"][
                "checkpoint_selection"
            ]
            self.assertTrue(
                checkpoint_selection["selected_checkpoint_eligible"]
            )
            self.assertFalse(checkpoint_selection["fallback_used"])
            self.assertEqual(checkpoint_selection["selected_epoch"], 1)
            auxiliary = result["auxiliary_metrics"]
            self.assertEqual(auxiliary["status"], "available")
            self.assertFalse(
                auxiliary["used_for_checkpoint_or_model_selection"]
            )
            self.assertEqual(
                set(auxiliary["regimes"]),
                set(self.runner.evaluation_split_names(
                    self.runner.inspect_cache_contract(FACTOR_CACHE_ROOT)
                )),
            )
            doppler = auxiliary["regimes"]["unseen_speed"]["quality"][
                "physical_mae"
            ]["doppler_hz"]
            self.assertEqual(doppler["status"], "available")
            self.assertEqual(doppler["unit"], "Hz")
            self.assertEqual(
                doppler["physical_target_source"],
                "view.doppler_hz",
            )
            clean_sir = auxiliary["regimes"]["clean_retention"][
                "quality"
            ]["physical_mae"]["sir_db"]
            self.assertEqual(clean_sir["status"], "unavailable")
            self.assertEqual(clean_sir["support"], 0)
            self.assertIsNone(clean_sir["value"])
            self.assertTrue(
                payload["source_tree_execution_audit"]["unchanged"]
            )
            self.assertFalse(
                payload["evidence_eligibility"]["eligible"]
            )
            held_pulse = auxiliary["regimes"]["unseen_jammer"][
                "jammer_multilabel"
            ]["per_class"]["pulse"]
            self.assertFalse(held_pulse["training_supported"])
            self.assertEqual(held_pulse["f1"]["status"], "unavailable")
            self.assertFalse(
                payload["submission_release"]["macro_generation_permitted"]
            )


if __name__ == "__main__":
    unittest.main()
