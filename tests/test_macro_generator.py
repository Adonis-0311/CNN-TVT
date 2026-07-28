"""Offline contract tests for automatic TVT macro derivation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tvt_submission import generate_macro_values as generator


class MacroGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = generator.validate_release._load_runner_module()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_csv(
        self,
        path: Path,
        fieldnames: list[str],
        rows: list[dict],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _classification(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        classes: int,
    ) -> tuple[float, float, float]:
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (labels, predictions), 1)
        true_positive = np.diag(matrix).astype(np.float64)
        support = matrix.sum(axis=1).astype(np.float64)
        predicted = matrix.sum(axis=0).astype(np.float64)
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive),
            where=support > 0,
        )
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(recall),
            where=(precision + recall) > 0,
        )
        supported = support > 0
        return (
            float((predictions == labels).mean()),
            float(f1[supported].mean()),
            float(recall[supported].min()),
        )

    def _fixture(
        self,
        root: Path,
        *,
        tied_strongest: bool = False,
    ) -> tuple[Path, Path]:
        cache_digest = "a" * 64
        cache_root = root / "formal_cache"
        self._write_json(
            cache_root / "manifest.json",
            {
                "cache_digest": cache_digest,
                "configuration": {
                    "evidence_designation": (
                        generator.validate_release.FORMAL_DESIGNATION
                    )
                },
            },
        )
        run_root = root / "formal_run"
        models = [
            "a0_backbone",
            "mcldnn_reimplementation",
            "iqformer_inspired",
            "cssl_amc_supervised_adaptation",
            "a5_vimd_full",
        ]
        seeds = [17, 29]
        regimes = [
            "hard_interference",
            "unseen_jammer",
            "heldout_channel",
        ]
        classes = 3
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        source_ids = np.arange(100, 106, dtype=np.int64)
        predictions = {
            "a0_backbone": np.asarray([0, 1, 1, 0, 2, 0]),
            "mcldnn_reimplementation": np.asarray([0, 0, 1, 0, 2, 0]),
            "iqformer_inspired": np.asarray([0, 0, 1, 1, 2, 0]),
            "cssl_amc_supervised_adaptation": np.asarray(
                [0, 1, 1, 1, 2, 0]
            ),
            "a5_vimd_full": labels.copy(),
        }
        if tied_strongest:
            predictions["mcldnn_reimplementation"] = predictions[
                "iqformer_inspired"
            ].copy()

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
        results: list[dict] = []
        metric_rows: list[dict] = []
        metric_lookup: dict[tuple[str, int, str], dict[str, float]] = {}
        for model in models:
            for seed in seeds:
                model_root = run_root / "models" / f"{model}_seed{seed}"
                model_root.mkdir(parents=True, exist_ok=True)
                (model_root / "model.pt").write_bytes(b"checkpoint")
                regime_records: dict[str, dict[str, float]] = {}
                for regime in regimes:
                    predicted = predictions[model]
                    probabilities = np.full(
                        (len(labels), classes),
                        0.1,
                        dtype=np.float64,
                    )
                    probabilities[
                        np.arange(len(labels)),
                        predicted,
                    ] = 0.8
                    np.savez_compressed(
                        model_root / f"predictions_{regime}.npz",
                        probabilities=probabilities,
                        labels=labels,
                        source_ids=source_ids,
                        snr_db=np.zeros(len(labels), dtype=np.float64),
                        sir_db=np.zeros(len(labels), dtype=np.float64),
                        cache_digest=np.asarray(cache_digest),
                        split=np.asarray(regime),
                    )
                    accuracy, macro_f1, worst_recall = self._classification(
                        labels,
                        predicted,
                        classes,
                    )
                    metrics = {
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "worst_recall": worst_recall,
                        "nll": 0.25,
                        "ece": 0.02,
                    }
                    regime_records[regime] = metrics
                    metric_lookup[(model, seed, regime)] = metrics
                    metric_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "regime": regime,
                            "cache_digest": cache_digest,
                            "standards_evidence_label": "fixture",
                            **metrics,
                        }
                    )
                result = {
                    "model": model,
                    "seed": seed,
                    "checkpoint": (
                        f"models/{model}_seed{seed}/model.pt"
                    ),
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
                    },
                    "complexity": {
                        "parameters": 123456.0,
                        "latency_ms_p50": (
                            1.2 if seed == seeds[0] else 1.4
                        ),
                        "latency_device": "cuda",
                    },
                    "regimes": regime_records,
                }
                if model == generator.METHOD_MODEL:
                    result["mechanism"] = {
                        "schema_version": 2,
                        "split": "heldout_channel",
                        "seed": seed,
                        "counterfactual_tf_sir_gain_db": (
                            2.0 if seed == seeds[0] else 4.0
                        ),
                    }
                self._write_json(model_root / "result.json", result)
                results.append(result)

        metric_fields = [
            "model",
            "seed",
            "regime",
            "cache_digest",
            "standards_evidence_label",
            "accuracy",
            "macro_f1",
            "worst_recall",
            "nll",
            "ece",
        ]
        self._write_csv(
            run_root / "metrics.csv",
            metric_fields,
            metric_rows,
        )
        (run_root / "seed_aggregates.csv").write_text(
            "value\n1\n",
            encoding="utf-8",
        )
        (run_root / "paired_statistics.csv").write_text(
            "value\n1\n",
            encoding="utf-8",
        )

        headline_fields = [
            "reference",
            "reference_selection",
            "reference_strength_claimed",
            "candidate",
            "regime",
            "cache_digest",
            "difference",
            "ci95_low",
            "ci95_high",
            "accuracy_difference",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "macro_f1_difference",
            "macro_f1_ci95_low",
            "macro_f1_ci95_high",
            "accuracy_test_source_only_ci95_low",
            "accuracy_test_source_only_ci95_high",
            "macro_f1_test_source_only_ci95_low",
            "macro_f1_test_source_only_ci95_high",
            "accuracy_algorithm_seed_only_ci95_low",
            "accuracy_algorithm_seed_only_ci95_high",
            "macro_f1_algorithm_seed_only_ci95_low",
            "macro_f1_algorithm_seed_only_ci95_high",
            "bootstrap_draws",
            "bootstrap_seed",
            "algorithm_seed_count",
            "algorithm_seed_ids",
            "test_source_cluster_count",
            "bootstrap_stratified_by_class",
            "bootstrap_hierarchy",
            "mcnemar_exact_test_performed",
            "mcnemar_reason",
        ]
        headline_rows: list[dict] = []
        reference = "iqformer_inspired"
        for candidate in models:
            if candidate == reference:
                continue
            for regime in regimes:
                accuracy_difference = float(
                    np.mean(
                        [
                            metric_lookup[(candidate, seed, regime)][
                                "accuracy"
                            ]
                            - metric_lookup[(reference, seed, regime)][
                                "accuracy"
                            ]
                            for seed in seeds
                        ]
                    )
                )
                macro_difference = float(
                    np.mean(
                        [
                            metric_lookup[(candidate, seed, regime)][
                                "macro_f1"
                            ]
                            - metric_lookup[(reference, seed, regime)][
                                "macro_f1"
                            ]
                            for seed in seeds
                        ]
                    )
                )
                headline_rows.append(
                    {
                        "reference": reference,
                        "reference_selection": "explicit_cli",
                        "reference_strength_claimed": False,
                        "candidate": candidate,
                        "regime": regime,
                        "cache_digest": cache_digest,
                        "difference": accuracy_difference,
                        "ci95_low": accuracy_difference - 0.05,
                        "ci95_high": accuracy_difference + 0.05,
                        "accuracy_difference": accuracy_difference,
                        "accuracy_ci95_low": accuracy_difference - 0.05,
                        "accuracy_ci95_high": accuracy_difference + 0.05,
                        "macro_f1_difference": macro_difference,
                        "macro_f1_ci95_low": macro_difference - 0.05,
                        "macro_f1_ci95_high": macro_difference + 0.05,
                        "accuracy_test_source_only_ci95_low": (
                            accuracy_difference - 0.04
                        ),
                        "accuracy_test_source_only_ci95_high": (
                            accuracy_difference + 0.04
                        ),
                        "macro_f1_test_source_only_ci95_low": (
                            macro_difference - 0.04
                        ),
                        "macro_f1_test_source_only_ci95_high": (
                            macro_difference + 0.04
                        ),
                        "accuracy_algorithm_seed_only_ci95_low": (
                            accuracy_difference - 0.03
                        ),
                        "accuracy_algorithm_seed_only_ci95_high": (
                            accuracy_difference + 0.03
                        ),
                        "macro_f1_algorithm_seed_only_ci95_low": (
                            macro_difference - 0.03
                        ),
                        "macro_f1_algorithm_seed_only_ci95_high": (
                            macro_difference + 0.03
                        ),
                        "bootstrap_draws": 100,
                        "bootstrap_seed": 7,
                        "algorithm_seed_count": len(seeds),
                        "algorithm_seed_ids": repr(
                            [str(seed) for seed in seeds]
                        ),
                        "test_source_cluster_count": len(source_ids),
                        "bootstrap_stratified_by_class": True,
                        "bootstrap_hierarchy": (
                            "algorithm_seed_and_class_stratified_"
                            "test_source_cluster"
                        ),
                        "mcnemar_exact_test_performed": False,
                        "mcnemar_reason": (
                            "repeated sources across fitted-model seeds"
                        ),
                    }
                )
        self._write_csv(
            run_root / "headline_paired_statistics.csv",
            headline_fields,
            headline_rows,
        )

        for name in ("cache_reference", "train", *regimes):
            self._write_json(
                run_root / "manifests" / f"{name}.json",
                {"status": "present"},
            )
        run_record = {
            "run_id": "formal_macro_fixture",
            "runner": "experiments/run_standard_experiment.py",
            "status": "complete",
            "execution_status": "complete",
            "cache_root": str(cache_root.resolve()),
            "cache_digest": cache_digest,
            "checksums_verified": True,
            "models": models,
            "seeds": seeds,
            "splits": {
                "train": 6,
                **{regime: 6 for regime in regimes},
            },
            "num_classes": classes,
            "training_configuration": {
                "epochs": 1,
                "mask_start_epoch": 0,
                "contrastive_start_epoch": 0,
                "mask_ramp_epochs": 1,
                "contrastive_ramp_epochs": 1,
                "minimum_full_stage_epochs": 1,
                "patience": 1,
            },
            "comparison_protocol": {
                "reference_model": reference,
                "reference_selection": "explicit_cli",
                "reference_strength_claimed": False,
                "bootstrap_draws": 100,
            },
            "results": results,
            "evidence_eligibility": {
                "policy_version": "vimd-evidence-gate-v2",
                "cache_designation": (
                    generator.validate_release.FORMAL_DESIGNATION
                ),
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
        self.assertTrue(
            run_record["submission_release"]["macro_generation_permitted"]
        )
        run_json = run_root / "run.json"
        self._write_json(run_json, run_record)
        return run_json, root / "macro_values.json"

    def _rewrite_csv_without_column(self, path: Path, column: str) -> None:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        index = rows[0].index(column)
        for row in rows:
            row.pop(index)
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def _rewrite_headline_value(
        self,
        path: Path,
        *,
        candidate: str,
        regime: str,
        column: str,
        value: str,
    ) -> None:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            if (
                row["candidate"] == candidate
                and row["regime"] == regime
            ):
                row[column] = value
                break
        else:
            self.fail("target headline row is absent")
        self._write_csv(path, fieldnames, rows)

    def test_generates_release_compatible_manifest_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_success_"
        ) as temporary:
            run_json, output = self._fixture(Path(temporary))
            result = generator.write_macro_manifest(
                run_json=run_json,
                output=output,
            )
            self.assertEqual(
                result["macro_count"],
                len(generator.validate_release.RESULT_MACROS),
            )
            manifest = generator.validate_release.load_strict_json(output)
            self.assertEqual(
                set(manifest["macros"]),
                set(generator.validate_release.RESULT_MACROS),
            )
            self.assertEqual(
                manifest["macros"]["StrongestBaseline"]["value"],
                "IQFormer-inspired local baseline",
            )
            self.assertIn(
                "cssl_amc_supervised_adaptation",
                manifest["macros"]["StrongestBaseline"]["derivation"],
            )
            self.assertEqual(
                manifest["macros"]["FeatureSIRGain"]["value"],
                "+3.00",
            )
            self.assertEqual(
                manifest["macros"]["VIMDParameters"]["value"],
                "123,456",
            )
            self.assertEqual(
                manifest["macros"]["VIMDLatency"]["value"],
                "1.30 ms",
            )
            run_record = generator.validate_release.load_strict_json(
                run_json
            )
            values, provenance = (
                generator.validate_release.validate_macro_manifest(
                    output,
                    run_json=run_json,
                    run_record=run_record,
                )
            )
            self.assertEqual(
                set(values),
                set(generator.validate_release.RESULT_MACROS),
            )
            self.assertEqual(set(provenance), set(values))

    def test_missing_headline_column_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_missing_column_"
        ) as temporary:
            run_json, output = self._fixture(Path(temporary))
            self._rewrite_csv_without_column(
                run_json.parent / "headline_paired_statistics.csv",
                "macro_f1_ci95_low",
            )
            with self.assertRaisesRegex(
                generator.MacroGenerationError,
                "missing columns",
            ):
                generator.write_macro_manifest(
                    run_json=run_json,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_nonfinite_headline_cell_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_nan_"
        ) as temporary:
            run_json, output = self._fixture(Path(temporary))
            self._rewrite_headline_value(
                run_json.parent / "headline_paired_statistics.csv",
                candidate=generator.METHOD_MODEL,
                regime=generator.HARD_REGIME,
                column="macro_f1_difference",
                value="NaN",
            )
            with self.assertRaisesRegex(
                generator.MacroGenerationError,
                "nonfinite",
            ):
                generator.write_macro_manifest(
                    run_json=run_json,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_ambiguous_strongest_baseline_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_tie_"
        ) as temporary:
            run_json, output = self._fixture(
                Path(temporary),
                tied_strongest=True,
            )
            with self.assertRaisesRegex(
                generator.MacroGenerationError,
                "strongest baseline is ambiguous",
            ):
                generator.write_macro_manifest(
                    run_json=run_json,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_failed_source_run_fails_before_derivation(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_failed_run_"
        ) as temporary:
            run_json, output = self._fixture(Path(temporary))
            run_record = generator.validate_release.load_strict_json(
                run_json
            )
            run_record["execution_status"] = "failed"
            self._write_json(run_json, run_record)
            with self.assertRaisesRegex(
                generator.MacroGenerationError,
                "source run failed release validation",
            ):
                generator.write_macro_manifest(
                    run_json=run_json,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_missing_prediction_array_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_macro_npz_schema_"
        ) as temporary:
            run_json, output = self._fixture(Path(temporary))
            path = (
                run_json.parent
                / "models"
                / "a5_vimd_full_seed17"
                / "predictions_hard_interference.npz"
            )
            with np.load(path, allow_pickle=False) as archive:
                payload = {
                    name: archive[name]
                    for name in archive.files
                    if name != "labels"
                }
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(
                generator.MacroGenerationError,
                "NPZ schema mismatch",
            ):
                generator.write_macro_manifest(
                    run_json=run_json,
                    output=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
