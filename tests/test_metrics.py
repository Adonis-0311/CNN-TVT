from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.metrics import (  # noqa: E402
    PredictionBundle,
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


if __name__ == "__main__":
    unittest.main()
