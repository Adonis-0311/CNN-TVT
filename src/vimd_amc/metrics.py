"""Classification, calibration, and paired statistical utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass
class PredictionBundle:
    probabilities: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray
    snr_db: np.ndarray
    sir_db: np.ndarray
    target_profile_index: np.ndarray | None = None

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    def validate(self) -> None:
        count = len(self.labels)
        if self.probabilities.ndim != 2 or len(self.probabilities) != count:
            raise ValueError("probabilities must have shape [N, classes]")
        if self.probabilities.shape[1] <= 0:
            raise ValueError("probabilities must contain at least one class")
        if np.asarray(self.labels).ndim != 1:
            raise ValueError("labels must be one-dimensional")
        for name in ("source_ids", "snr_db", "sir_db"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != count:
                raise ValueError(f"{name} must have N entries")
        if self.target_profile_index is not None:
            target_profiles = np.asarray(self.target_profile_index)
            if target_profiles.ndim != 1 or len(target_profiles) != count:
                raise ValueError("target_profile_index must have N entries")
            if not np.issubdtype(target_profiles.dtype, np.integer):
                raise ValueError("target_profile_index must be integer-valued")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("probabilities contain non-finite values")
        if (self.probabilities < 0.0).any():
            raise ValueError("probabilities contain negative values")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("probability rows must sum to one")
        if not np.issubdtype(np.asarray(self.labels).dtype, np.integer):
            raise ValueError("labels must be integer-valued")

    def subset(self, selected: np.ndarray) -> "PredictionBundle":
        """Return a nonempty, row-aligned subset for preregistered strata."""

        self.validate()
        mask = np.asarray(selected)
        if mask.ndim != 1 or len(mask) != len(self.labels):
            raise ValueError("prediction subset selector must have N entries")
        if mask.dtype != np.bool_:
            raise ValueError("prediction subset selector must be boolean")
        if not mask.any():
            raise ValueError("prediction subset selector is empty")
        return PredictionBundle(
            probabilities=np.asarray(self.probabilities)[mask],
            labels=np.asarray(self.labels)[mask],
            source_ids=np.asarray(self.source_ids)[mask],
            snr_db=np.asarray(self.snr_db)[mask],
            sir_db=np.asarray(self.sir_db)[mask],
            target_profile_index=(
                np.asarray(self.target_profile_index)[mask]
                if self.target_profile_index is not None
                else None
            ),
        )


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    return matrix


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            score += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(score)


def classification_metrics(bundle: PredictionBundle, classes: int) -> dict[str, Any]:
    bundle.validate()
    probabilities = np.clip(bundle.probabilities, 1e-9, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    labels = bundle.labels.astype(int)
    if len(labels) == 0:
        raise ValueError("classification metrics require at least one sample")
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError("labels fall outside the declared class range")
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, classes)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
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
    nll = -np.log(probabilities[np.arange(len(labels)), labels]).mean()
    supported = support > 0
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1[supported].mean()),
        "worst_recall": float(recall[supported].min()),
        "class_coverage": float(supported.mean()),
        "supported_class_count": int(supported.sum()),
        "nll": float(nll),
        "ece": expected_calibration_error(probabilities, labels),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(labels)),
    }


def paired_bootstrap_difference(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    draws: int = 10_000,
    seed: int = 20260727,
) -> dict[str, float]:
    if len(reference_correct) != len(candidate_correct):
        raise ValueError("paired arrays must have equal length")
    if draws <= 0:
        raise ValueError("draws must be positive")
    if len(reference_correct) == 0:
        raise ValueError("paired arrays must be non-empty")
    rng = np.random.default_rng(seed)
    delta = candidate_correct.astype(np.float64) - reference_correct.astype(np.float64)
    observed = float(delta.mean())
    if cluster_ids is None:
        cluster_ids = np.arange(len(delta))
    cluster_ids = np.asarray(cluster_ids)
    if len(cluster_ids) != len(delta):
        raise ValueError("cluster_ids must have the same length as paired arrays")
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_members = [np.flatnonzero(inverse == index) for index in range(len(unique_clusters))]
    samples = np.empty(draws, dtype=np.float64)
    singleton_clusters = len(unique_clusters) == len(delta)
    for draw in range(draws):
        sampled_clusters = rng.integers(0, len(unique_clusters), len(unique_clusters))
        if singleton_clusters:
            samples[draw] = delta[sampled_clusters].mean()
        else:
            indices = np.concatenate([cluster_members[index] for index in sampled_clusters])
            samples[draw] = delta[indices].mean()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"difference": observed, "ci95_low": float(lower), "ci95_high": float(upper)}


def _macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> float:
    """Macro-F1 over classes represented in the resampled ground truth."""

    matrix = confusion_matrix(labels, predictions, classes)
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
    if not supported.any():
        raise ValueError("macro-F1 requires at least one represented class")
    return float(f1[supported].mean())


def _validate_paired_bundles(
    reference: PredictionBundle,
    candidate: PredictionBundle,
) -> int:
    """Validate exact row pairing and return the common class count."""

    reference.validate()
    candidate.validate()
    if reference.probabilities.shape != candidate.probabilities.shape:
        raise ValueError("paired bundles have different probability shapes")
    if not np.array_equal(reference.source_ids, candidate.source_ids):
        raise ValueError("paired bundles have different source IDs or ordering")
    if not np.array_equal(reference.labels, candidate.labels):
        raise ValueError("paired bundles have different labels")
    if (reference.target_profile_index is None) != (
        candidate.target_profile_index is None
    ):
        raise ValueError(
            "paired bundles disagree on target-profile metadata presence"
        )
    if (
        reference.target_profile_index is not None
        and not np.array_equal(
            reference.target_profile_index,
            candidate.target_profile_index,
        )
    ):
        raise ValueError(
            "paired bundles have different target-profile indices"
        )
    classes = int(reference.probabilities.shape[1])
    labels = reference.labels.astype(np.int64)
    if len(labels) == 0:
        raise ValueError("paired bundles must be non-empty")
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError("paired labels fall outside the probability taxonomy")
    return classes


def _cluster_strata(
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    stratify_by_class: bool,
) -> tuple[list[list[np.ndarray]], int]:
    """Create source-cluster groups, rejecting clusters that cross classes."""

    labels = np.asarray(labels, dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.ndim != 1 or len(cluster_ids) != len(labels):
        raise ValueError("cluster_ids must be one-dimensional with one entry per row")
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    if len(unique_clusters) == 0:
        raise ValueError("at least one source cluster is required")

    grouped: dict[int, list[np.ndarray]] = {}
    for cluster_index in range(len(unique_clusters)):
        members = np.flatnonzero(inverse == cluster_index)
        cluster_labels = np.unique(labels[members])
        if len(cluster_labels) != 1:
            raise ValueError(
                "each source cluster must belong to exactly one modulation class"
            )
        stratum = int(cluster_labels[0]) if stratify_by_class else 0
        grouped.setdefault(stratum, []).append(members)
    return [grouped[key] for key in sorted(grouped)], len(unique_clusters)


def _sample_cluster_indices(
    rng: np.random.Generator,
    strata: list[list[np.ndarray]],
) -> np.ndarray:
    sampled_members: list[np.ndarray] = []
    for clusters in strata:
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled_members.extend(clusters[index] for index in selected)
    return np.concatenate(sampled_members)


def _paired_metric_difference(
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    classes: int,
) -> tuple[float, float]:
    accuracy = float(
        np.mean(candidate_predictions == labels)
        - np.mean(reference_predictions == labels)
    )
    macro_f1 = (
        _macro_f1(labels, candidate_predictions, classes)
        - _macro_f1(labels, reference_predictions, classes)
    )
    return accuracy, float(macro_f1)


def _confidence_interval(samples: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def paired_accuracy_macro_f1_bootstrap(
    reference: PredictionBundle,
    candidate: PredictionBundle,
    *,
    cluster_ids: np.ndarray | None = None,
    draws: int = 10_000,
    seed: int = 20260727,
    stratify_by_class: bool = True,
) -> dict[str, Any]:
    """Paired source-cluster bootstrap for accuracy and macro-F1 differences.

    Rows must be in exactly the same source order.  The default resamples
    source clusters within modulation class, preserving class support in every
    draw.  All rows for a sampled source are kept together.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    classes = _validate_paired_bundles(reference, candidate)
    labels = reference.labels.astype(np.int64)
    resolved_clusters = (
        np.asarray(reference.source_ids)
        if cluster_ids is None
        else np.asarray(cluster_ids)
    )
    strata, cluster_count = _cluster_strata(
        labels,
        resolved_clusters,
        stratify_by_class=stratify_by_class,
    )
    reference_predictions = reference.predictions
    candidate_predictions = candidate.predictions
    observed_accuracy, observed_macro_f1 = _paired_metric_difference(
        labels,
        reference_predictions,
        candidate_predictions,
        classes,
    )
    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(draws, dtype=np.float64)
    macro_f1_samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = _sample_cluster_indices(rng, strata)
        accuracy_samples[draw], macro_f1_samples[draw] = (
            _paired_metric_difference(
                labels[indices],
                reference_predictions[indices],
                candidate_predictions[indices],
                classes,
            )
        )
    accuracy_low, accuracy_high = _confidence_interval(accuracy_samples)
    macro_f1_low, macro_f1_high = _confidence_interval(macro_f1_samples)
    return {
        # Backward-compatible aliases: these have always denoted accuracy.
        "difference": observed_accuracy,
        "ci95_low": accuracy_low,
        "ci95_high": accuracy_high,
        "accuracy_difference": observed_accuracy,
        "accuracy_ci95_low": accuracy_low,
        "accuracy_ci95_high": accuracy_high,
        "macro_f1_difference": observed_macro_f1,
        "macro_f1_ci95_low": macro_f1_low,
        "macro_f1_ci95_high": macro_f1_high,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_cluster_count": int(cluster_count),
        "bootstrap_stratified_by_class": bool(stratify_by_class),
        "bootstrap_unit": "source_cluster",
    }


def mcnemar_exact_test(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
) -> dict[str, float | int]:
    if len(reference_correct) != len(candidate_correct):
        raise ValueError("paired arrays must have equal length")
    reference_only = int((reference_correct & ~candidate_correct).sum())
    candidate_only = int((~reference_correct & candidate_correct).sum())
    discordant = reference_only + candidate_only
    p_value = (
        float(binomtest(min(reference_only, candidate_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "reference_only_correct": reference_only,
        "candidate_only_correct": candidate_only,
        "discordant_pairs": discordant,
        "exact_p_value": p_value,
    }


def paired_bundle_statistics(
    reference: PredictionBundle,
    candidate: PredictionBundle,
    *,
    cluster_ids: np.ndarray | None = None,
    draws: int = 10_000,
    seed: int = 20260727,
) -> dict[str, Any]:
    """Single-seed paired inference after strict source-order validation.

    McNemar's exact test is intentionally confined to this single fitted-model
    comparison.  Use :func:`headline_paired_bootstrap` across algorithm seeds;
    pooled multi-seed rows are not independent Bernoulli pairs.
    """

    _validate_paired_bundles(reference, candidate)
    reference_correct = reference.predictions == reference.labels
    candidate_correct = candidate.predictions == candidate.labels
    if cluster_ids is None:
        cluster_ids = reference.source_ids
    return {
        **paired_accuracy_macro_f1_bootstrap(
            reference,
            candidate,
            cluster_ids=cluster_ids,
            draws=draws,
            seed=seed,
        ),
        **mcnemar_exact_test(reference_correct, candidate_correct),
        "inference_scope": "single_algorithm_seed",
        "mcnemar_scope": "single_seed_strictly_paired_test_sources",
    }


def headline_paired_bootstrap(
    references_by_seed: Mapping[Any, PredictionBundle],
    candidates_by_seed: Mapping[Any, PredictionBundle],
    *,
    draws: int = 10_000,
    seed: int = 20260727,
    stratify_by_class: bool = True,
) -> dict[str, Any]:
    """Hierarchical paired bootstrap across algorithm seeds and test sources.

    Each draw independently resamples (i) fitted-model seeds and (ii) held-out
    source clusters within class.  The same resampled sources are applied to
    every selected seed.  Source-only and algorithm-seed-only intervals are
    also returned so the two uncertainty layers remain visible.  No pooled
    McNemar test is produced because repeated predictions of the same source
    across fitted-model seeds are not independent pairs.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    reference_keys = set(references_by_seed)
    candidate_keys = set(candidates_by_seed)
    if not reference_keys or reference_keys != candidate_keys:
        raise ValueError(
            "reference and candidate mappings must have the same nonempty seed keys"
        )
    seed_keys = sorted(reference_keys, key=lambda value: str(value))
    first_reference = references_by_seed[seed_keys[0]]
    first_candidate = candidates_by_seed[seed_keys[0]]
    classes = _validate_paired_bundles(first_reference, first_candidate)
    labels = first_reference.labels.astype(np.int64)
    source_ids = np.asarray(first_reference.source_ids)
    target_profile_index = (
        np.asarray(first_reference.target_profile_index)
        if first_reference.target_profile_index is not None
        else None
    )

    reference_predictions: list[np.ndarray] = []
    candidate_predictions: list[np.ndarray] = []
    for algorithm_seed in seed_keys:
        reference = references_by_seed[algorithm_seed]
        candidate = candidates_by_seed[algorithm_seed]
        current_classes = _validate_paired_bundles(reference, candidate)
        if current_classes != classes:
            raise ValueError("class taxonomy differs across algorithm seeds")
        if not np.array_equal(reference.source_ids, source_ids):
            raise ValueError("test source IDs or ordering differ across seeds")
        if not np.array_equal(reference.labels, labels):
            raise ValueError("test labels differ across algorithm seeds")
        current_profiles = (
            np.asarray(reference.target_profile_index)
            if reference.target_profile_index is not None
            else None
        )
        if (current_profiles is None) != (target_profile_index is None) or (
            current_profiles is not None
            and not np.array_equal(current_profiles, target_profile_index)
        ):
            raise ValueError(
                "target-profile indices differ across algorithm seeds"
            )
        reference_predictions.append(reference.predictions)
        candidate_predictions.append(candidate.predictions)

    strata, cluster_count = _cluster_strata(
        labels,
        source_ids,
        stratify_by_class=stratify_by_class,
    )
    seed_count = len(seed_keys)
    per_seed_accuracy = np.empty(seed_count, dtype=np.float64)
    per_seed_macro_f1 = np.empty(seed_count, dtype=np.float64)
    all_indices = np.arange(len(labels))
    for seed_index in range(seed_count):
        per_seed_accuracy[seed_index], per_seed_macro_f1[seed_index] = (
            _paired_metric_difference(
                labels,
                reference_predictions[seed_index],
                candidate_predictions[seed_index],
                classes,
            )
        )
    observed_accuracy = float(per_seed_accuracy.mean())
    observed_macro_f1 = float(per_seed_macro_f1.mean())

    seed_sequence = np.random.SeedSequence(seed)
    hierarchical_rng, source_rng, algorithm_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(3)
    )
    hierarchical_accuracy = np.empty(draws, dtype=np.float64)
    hierarchical_macro_f1 = np.empty(draws, dtype=np.float64)
    source_only_accuracy = np.empty(draws, dtype=np.float64)
    source_only_macro_f1 = np.empty(draws, dtype=np.float64)
    algorithm_only_accuracy = np.empty(draws, dtype=np.float64)
    algorithm_only_macro_f1 = np.empty(draws, dtype=np.float64)

    def mean_difference(
        selected_seeds: np.ndarray,
        selected_sources: np.ndarray,
    ) -> tuple[float, float]:
        accuracy_values = []
        macro_f1_values = []
        selected_labels = labels[selected_sources]
        for seed_index in selected_seeds:
            accuracy, macro_f1 = _paired_metric_difference(
                selected_labels,
                reference_predictions[int(seed_index)][selected_sources],
                candidate_predictions[int(seed_index)][selected_sources],
                classes,
            )
            accuracy_values.append(accuracy)
            macro_f1_values.append(macro_f1)
        return float(np.mean(accuracy_values)), float(np.mean(macro_f1_values))

    all_seed_indices = np.arange(seed_count)
    for draw in range(draws):
        hierarchical_seed_indices = hierarchical_rng.integers(
            0,
            seed_count,
            size=seed_count,
        )
        hierarchical_source_indices = _sample_cluster_indices(
            hierarchical_rng,
            strata,
        )
        (
            hierarchical_accuracy[draw],
            hierarchical_macro_f1[draw],
        ) = mean_difference(
            hierarchical_seed_indices,
            hierarchical_source_indices,
        )

        source_indices = _sample_cluster_indices(source_rng, strata)
        source_only_accuracy[draw], source_only_macro_f1[draw] = mean_difference(
            all_seed_indices,
            source_indices,
        )

        algorithm_seed_indices = algorithm_rng.integers(
            0,
            seed_count,
            size=seed_count,
        )
        (
            algorithm_only_accuracy[draw],
            algorithm_only_macro_f1[draw],
        ) = mean_difference(algorithm_seed_indices, all_indices)

    accuracy_low, accuracy_high = _confidence_interval(hierarchical_accuracy)
    macro_f1_low, macro_f1_high = _confidence_interval(hierarchical_macro_f1)
    source_accuracy_low, source_accuracy_high = _confidence_interval(
        source_only_accuracy
    )
    source_macro_low, source_macro_high = _confidence_interval(
        source_only_macro_f1
    )
    algorithm_accuracy_low, algorithm_accuracy_high = _confidence_interval(
        algorithm_only_accuracy
    )
    algorithm_macro_low, algorithm_macro_high = _confidence_interval(
        algorithm_only_macro_f1
    )
    return {
        # Accuracy aliases preserve the single-seed table vocabulary.
        "difference": observed_accuracy,
        "ci95_low": accuracy_low,
        "ci95_high": accuracy_high,
        "accuracy_difference": observed_accuracy,
        "accuracy_ci95_low": accuracy_low,
        "accuracy_ci95_high": accuracy_high,
        "macro_f1_difference": observed_macro_f1,
        "macro_f1_ci95_low": macro_f1_low,
        "macro_f1_ci95_high": macro_f1_high,
        "accuracy_test_source_only_ci95_low": source_accuracy_low,
        "accuracy_test_source_only_ci95_high": source_accuracy_high,
        "macro_f1_test_source_only_ci95_low": source_macro_low,
        "macro_f1_test_source_only_ci95_high": source_macro_high,
        "accuracy_algorithm_seed_only_ci95_low": algorithm_accuracy_low,
        "accuracy_algorithm_seed_only_ci95_high": algorithm_accuracy_high,
        "macro_f1_algorithm_seed_only_ci95_low": algorithm_macro_low,
        "macro_f1_algorithm_seed_only_ci95_high": algorithm_macro_high,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "algorithm_seed_count": int(seed_count),
        "algorithm_seed_ids": [str(value) for value in seed_keys],
        "test_source_cluster_count": int(cluster_count),
        "bootstrap_stratified_by_class": bool(stratify_by_class),
        "bootstrap_hierarchy": (
            "algorithm_seed_and_class_stratified_test_source_cluster"
        ),
        "mcnemar_exact_test_performed": False,
        "mcnemar_reason": (
            "pooled multi-seed predictions reuse test sources and are not "
            "independent Bernoulli pairs"
        ),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise error correction in original hypothesis order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
