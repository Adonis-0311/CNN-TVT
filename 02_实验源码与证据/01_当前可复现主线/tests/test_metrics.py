from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.metrics import (  # noqa: E402
    PredictionBundle,
    ablation_family_paired_bootstrap,
    classification_metrics,
    headline_paired_bootstrap,
    holm_adjust,
    mcnemar_exact_test,
    paired_accuracy_macro_f1_bootstrap,
    paired_bundle_statistics,
)


def _bundle(probabilities: np.ndarray, labels: np.ndarray) -> PredictionBundle:
    count = len(labels)
    return PredictionBundle(
        probabilities=probabilities,
        labels=labels,
        source_ids=np.arange(count, dtype=np.int64),
        snr_db=np.zeros(count),
        sir_db=np.zeros(count),
    )


def _bundle_from_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    source_ids: np.ndarray | None = None,
    classes: int = 2,
) -> PredictionBundle:
    probabilities = np.full((len(predictions), classes), 0.1 / (classes - 1))
    probabilities[np.arange(len(predictions)), predictions] = 0.9
    return PredictionBundle(
        probabilities=probabilities,
        labels=labels,
        source_ids=(
            np.arange(len(labels), dtype=np.int64)
            if source_ids is None
            else source_ids
        ),
        snr_db=np.zeros(len(labels)),
        sir_db=np.zeros(len(labels)),
    )


class MetricsTests(unittest.TestCase):
    def test_unsupported_classes_do_not_force_worst_recall_to_zero(self) -> None:
        bundle = _bundle(
            np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]),
            np.asarray([0, 1]),
        )
        metrics = classification_metrics(bundle, classes=3)
        self.assertEqual(metrics["worst_recall"], 1.0)
        self.assertAlmostEqual(metrics["class_coverage"], 2.0 / 3.0)

    def test_invalid_probability_rows_are_rejected(self) -> None:
        bundle = _bundle(np.asarray([[0.8, 0.8]]), np.asarray([0]))
        with self.assertRaises(ValueError):
            classification_metrics(bundle, classes=2)

    def test_paired_statistics_require_identical_source_order(self) -> None:
        reference = _bundle(
            np.asarray([[0.8, 0.2], [0.8, 0.2]]),
            np.asarray([0, 1]),
        )
        candidate = _bundle(
            np.asarray([[0.9, 0.1], [0.1, 0.9]]),
            np.asarray([0, 1]),
        )
        candidate.source_ids = candidate.source_ids[::-1]
        with self.assertRaises(ValueError):
            paired_bundle_statistics(reference, candidate, draws=20)

    def test_prediction_bundle_profile_subset_preserves_alignment(self) -> None:
        bundle = _bundle_from_predictions(
            np.asarray([0, 1, 0, 1]),
            np.asarray([0, 1, 0, 1]),
            source_ids=np.asarray([10, 20, 30, 40]),
        )
        bundle.target_profile_index = np.asarray([0, 1, 2, 4])
        subset = bundle.subset(np.asarray([True, False, True, False]))
        subset.validate()
        np.testing.assert_array_equal(subset.source_ids, [10, 30])
        np.testing.assert_array_equal(subset.target_profile_index, [0, 2])

    def test_paired_statistics_reject_profile_metadata_drift(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        reference = _bundle_from_predictions(labels, labels)
        candidate = _bundle_from_predictions(labels, labels)
        reference.target_profile_index = np.asarray([0, 1, 2, 4])
        candidate.target_profile_index = np.asarray([0, 1, 3, 4])
        with self.assertRaisesRegex(ValueError, "target-profile"):
            paired_accuracy_macro_f1_bootstrap(
                reference,
                candidate,
                draws=10,
            )

    def test_mcnemar_reports_exact_p_value(self) -> None:
        reference = np.asarray([True, True, False, False, False])
        candidate = np.asarray([True, False, True, True, True])
        result = mcnemar_exact_test(reference, candidate)
        self.assertEqual(result["reference_only_correct"], 1)
        self.assertEqual(result["candidate_only_correct"], 3)
        self.assertGreaterEqual(result["exact_p_value"], 0.0)
        self.assertLessEqual(result["exact_p_value"], 1.0)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        raw = np.asarray([0.04, 0.001, 0.02])
        adjusted = holm_adjust(raw)
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))
        self.assertTrue(np.all(adjusted >= raw))

    def test_paired_bootstrap_reports_accuracy_and_macro_f1(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        sources = np.asarray([10, 11, 12, 20, 21, 22])
        reference = _bundle_from_predictions(
            np.asarray([0, 1, 1, 1, 0, 0]),
            labels,
            source_ids=sources,
        )
        candidate = _bundle_from_predictions(
            np.asarray([0, 0, 1, 1, 1, 0]),
            labels,
            source_ids=sources,
        )
        result = paired_accuracy_macro_f1_bootstrap(
            reference,
            candidate,
            draws=100,
            seed=1234,
        )
        repeated = paired_accuracy_macro_f1_bootstrap(
            reference,
            candidate,
            draws=100,
            seed=1234,
        )
        self.assertAlmostEqual(result["accuracy_difference"], 1.0 / 3.0)
        self.assertEqual(result["difference"], result["accuracy_difference"])
        self.assertEqual(result["ci95_low"], result["accuracy_ci95_low"])
        self.assertIn("macro_f1_difference", result)
        self.assertEqual(result["bootstrap_cluster_count"], 6)
        self.assertTrue(result["bootstrap_stratified_by_class"])
        self.assertEqual(result, repeated)

    def test_source_cluster_cannot_cross_modulation_classes(self) -> None:
        labels = np.asarray([0, 1])
        sources = np.asarray([7, 7])
        reference = _bundle_from_predictions(
            np.asarray([0, 0]),
            labels,
            source_ids=sources,
        )
        candidate = _bundle_from_predictions(
            np.asarray([0, 1]),
            labels,
            source_ids=sources,
        )
        with self.assertRaisesRegex(ValueError, "exactly one modulation"):
            paired_accuracy_macro_f1_bootstrap(
                reference,
                candidate,
                draws=10,
            )

    def test_headline_bootstrap_exposes_two_uncertainty_layers(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        sources = np.asarray([100, 101, 200, 201])
        references = {
            17: _bundle_from_predictions(
                np.asarray([0, 1, 1, 0]),
                labels,
                source_ids=sources,
            ),
            23: _bundle_from_predictions(
                np.asarray([0, 1, 0, 1]),
                labels,
                source_ids=sources,
            ),
        }
        candidates = {
            17: _bundle_from_predictions(
                np.asarray([0, 0, 1, 0]),
                labels,
                source_ids=sources,
            ),
            23: _bundle_from_predictions(
                np.asarray([0, 0, 0, 1]),
                labels,
                source_ids=sources,
            ),
        }
        result = headline_paired_bootstrap(
            references,
            candidates,
            draws=80,
            seed=99,
        )
        self.assertEqual(result["algorithm_seed_count"], 2)
        self.assertEqual(result["test_source_cluster_count"], 4)
        self.assertIn("algorithm_seed", result["bootstrap_hierarchy"])
        self.assertIn("test_source", result["bootstrap_hierarchy"])
        self.assertIn("accuracy_test_source_only_ci95_low", result)
        self.assertIn("accuracy_algorithm_seed_only_ci95_low", result)
        self.assertFalse(result["mcnemar_exact_test_performed"])

    def test_headline_bootstrap_rejects_test_source_drift_across_seeds(self) -> None:
        labels = np.asarray([0, 1])
        references = {
            1: _bundle_from_predictions(
                np.asarray([0, 1]),
                labels,
                source_ids=np.asarray([10, 20]),
            ),
            2: _bundle_from_predictions(
                np.asarray([0, 1]),
                labels,
                source_ids=np.asarray([10, 21]),
            ),
        }
        candidates = {
            seed: _bundle_from_predictions(
                np.asarray([0, 1]),
                labels,
                source_ids=bundle.source_ids.copy(),
            )
            for seed, bundle in references.items()
        }
        with self.assertRaisesRegex(ValueError, "differ across seeds"):
            headline_paired_bootstrap(
                references,
                candidates,
                draws=10,
            )

    def test_ablation_family_joint_bootstrap_is_deterministic_and_simultaneous(
        self,
    ) -> None:
        labels = np.repeat(np.arange(3, dtype=np.int64), 20)
        sources = np.arange(1_000, 1_060, dtype=np.int64)

        def with_errors(errors_per_class: int) -> np.ndarray:
            predictions = labels.copy()
            for class_index in range(3):
                selected = np.flatnonzero(labels == class_index)
                predictions[selected[:errors_per_class]] = (
                    class_index + 1
                ) % 3
            return predictions

        model_errors = {
            "a2": 15,
            "a3": 10,
            "a4": 5,
        }
        algorithm_seeds = (17, 29, 43, 71, 101)
        bundles = {
            model: {
                seed: _bundle_from_predictions(
                    with_errors(errors),
                    labels,
                    source_ids=sources,
                    classes=3,
                )
                for seed in algorithm_seeds
            }
            for model, errors in model_errors.items()
        }
        contrasts = {
            "teacher": ("a2", "a3"),
            "multitask": ("a3", "a4"),
        }
        result = ablation_family_paired_bootstrap(
            bundles,
            contrasts,
            draws=300,
            seed=919,
        )
        repeated = ablation_family_paired_bootstrap(
            bundles,
            contrasts,
            draws=300,
            seed=919,
        )
        self.assertEqual(result, repeated)
        self.assertEqual(
            result["multiplicity_method"],
            (
                "joint_max_absolute_centered_deviation_"
                "hierarchical_paired_bootstrap"
            ),
        )
        self.assertEqual(result["family_size"], 2)
        self.assertEqual(
            result["algorithm_seed_ids"],
            ["17", "29", "43", "71", "101"],
        )
        half_widths = []
        for record in result["contrasts"].values():
            point = record["macro_f1_difference"]
            low = record["macro_f1_simultaneous_ci95_low"]
            high = record["macro_f1_simultaneous_ci95_high"]
            self.assertLessEqual(low, point)
            self.assertGreaterEqual(high, point)
            self.assertGreater(low, 0.0)
            half_widths.append(high - point)
        self.assertAlmostEqual(half_widths[0], half_widths[1])
        self.assertAlmostEqual(
            half_widths[0],
            result["simultaneous_critical_value"],
        )

    def test_ablation_family_rejects_cross_model_source_drift(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        reference = _bundle_from_predictions(
            labels,
            labels,
            source_ids=np.asarray([10, 11, 20, 21]),
        )
        candidate = _bundle_from_predictions(
            labels,
            labels,
            source_ids=np.asarray([10, 12, 20, 21]),
        )
        with self.assertRaisesRegex(ValueError, "source IDs"):
            ablation_family_paired_bootstrap(
                {
                    "reference": {17: reference},
                    "candidate": {17: candidate},
                },
                {"contrast": ("reference", "candidate")},
                draws=10,
            )

    def test_ablation_family_rejects_snr_or_sir_pairing_drift(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        reference = _bundle_from_predictions(labels, labels)
        candidate = _bundle_from_predictions(labels, labels)
        candidate.snr_db[1] = 4.0
        with self.assertRaisesRegex(ValueError, "snr_db"):
            ablation_family_paired_bootstrap(
                {
                    "reference": {17: reference},
                    "candidate": {17: candidate},
                },
                {"contrast": ("reference", "candidate")},
                draws=10,
            )

        candidate.snr_db = reference.snr_db.copy()
        candidate.sir_db[2] = -5.0
        with self.assertRaisesRegex(ValueError, "sir_db"):
            ablation_family_paired_bootstrap(
                {
                    "reference": {17: reference},
                    "candidate": {17: candidate},
                },
                {"contrast": ("reference", "candidate")},
                draws=10,
            )

    def test_ablation_family_rejects_nonfinite_hard_regime_covariates(
        self,
    ) -> None:
        labels = np.asarray([0, 0, 1, 1])
        reference = _bundle_from_predictions(labels, labels)
        candidate = _bundle_from_predictions(
            np.asarray([0, 1, 1, 1]),
            labels,
        )
        reference.sir_db[0] = np.inf
        candidate.sir_db[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite sir_db"):
            ablation_family_paired_bootstrap(
                {
                    "reference": {17: reference},
                    "candidate": {17: candidate},
                },
                {"contrast": ("reference", "candidate")},
                draws=10,
            )

    def test_ablation_family_requires_fixed_ninety_five_percent_stratification(
        self,
    ) -> None:
        labels = np.asarray([0, 0, 1, 1])
        reference = _bundle_from_predictions(labels, labels)
        candidate = _bundle_from_predictions(
            np.asarray([0, 1, 1, 1]),
            labels,
        )
        family = {
            "reference": {17: reference},
            "candidate": {17: candidate},
        }
        contrasts = {"contrast": ("reference", "candidate")}
        with self.assertRaisesRegex(ValueError, "confidence_level=0.95"):
            ablation_family_paired_bootstrap(
                family,
                contrasts,
                draws=10,
                confidence_level=0.90,
            )
        with self.assertRaisesRegex(ValueError, "stratify_by_class=True"):
            ablation_family_paired_bootstrap(
                family,
                contrasts,
                draws=10,
                stratify_by_class=False,
            )

    def test_ablation_family_exposes_degenerate_zero_width_interval(
        self,
    ) -> None:
        labels = np.asarray([0, 0, 1, 1])
        reference = _bundle_from_predictions(labels, labels)
        candidate = _bundle_from_predictions(labels, labels)
        result = ablation_family_paired_bootstrap(
            {
                "reference": {17: reference},
                "candidate": {17: candidate},
            },
            {"contrast": ("reference", "candidate")},
            draws=10,
        )
        self.assertEqual(result["simultaneous_critical_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
