"""Evaluation, mechanism probes, complexity, and latency measurements."""

from __future__ import annotations

from contextlib import contextmanager
import io
from numbers import Real
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .losses import jensen_shannon_mask_loss
from .metrics import PredictionBundle, classification_metrics
from .models.spectral import ComplexSTFT
from .models.vimd import PhysicalTriMaskTeacher, VIMDNet


_QUALITY_COMPONENTS = ("snr_db", "sir_db", "doppler_hz")
_QUALITY_UNITS = {
    "snr_db": "dB",
    "sir_db": "dB",
    "doppler_hz": "Hz",
}


def _move_view(view: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in view.items()}


def _weighted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    weights = weights / total
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_centered = first - np.sum(weights * first)
    second_centered = second - np.sum(weights * second)
    covariance = np.sum(weights * first_centered * second_centered)
    denominator = np.sqrt(
        np.sum(weights * first_centered**2)
        * np.sum(weights * second_centered**2)
    )
    return float(covariance / denominator) if denominator > 0 else float("nan")


def _finite_mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else None


def _available_metric(value: float, support: int, **metadata: Any) -> dict[str, Any]:
    return {
        "status": "available",
        "value": float(value),
        "support": int(support),
        **metadata,
    }


def _unavailable_metric(reason: str, support: int = 0, **metadata: Any) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "support": int(support),
        "reason": reason,
        **metadata,
    }


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    true_positive = int(np.sum(labels & predictions))
    false_positive = int(np.sum(~labels & predictions))
    false_negative = int(np.sum(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "value": (
            float(2 * true_positive / denominator)
            if denominator > 0
            else float("nan")
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "denominator": denominator,
    }


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with tie handling and explicit support checks."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[labels].sum())
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def _parse_quality_scale_mapping(
    values: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for component in _QUALITY_COMPONENTS:
        if component not in values:
            continue
        raw = values[component]
        if isinstance(raw, Mapping):
            scale = raw.get("scale")
            offset = raw.get("offset", 0.0)
            unit = str(raw.get("unit", _QUALITY_UNITS[component]))
        else:
            scale = raw
            offset = 0.0
            unit = _QUALITY_UNITS[component]
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise ValueError(
                f"quality scale for {component} must be a positive number"
            )
        numeric = float(scale)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(
                f"quality scale for {component} must be positive and finite"
            )
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise ValueError(
                f"quality offset for {component} must be a finite number"
            )
        numeric_offset = float(offset)
        if not np.isfinite(numeric_offset):
            raise ValueError(
                f"quality offset for {component} must be finite"
            )
        parsed[component] = {
            "scale": numeric,
            "offset": numeric_offset,
            "unit": unit,
            "source": source,
        }
    return parsed


def _quality_scales_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read only an explicit quality-normalization schema from a manifest."""

    candidates: list[tuple[str, Any]] = [
        ("manifest.quality_normalization", manifest.get("quality_normalization")),
    ]
    for parent_name in ("configuration", "synthesis_configuration"):
        parent = manifest.get(parent_name)
        if isinstance(parent, Mapping):
            candidates.append(
                (
                    f"manifest.{parent_name}.quality_normalization",
                    parent.get("quality_normalization"),
                )
            )
    parsed: dict[str, dict[str, Any]] = {}
    for source, candidate in candidates:
        if isinstance(candidate, Mapping):
            parsed.update(
                _parse_quality_scale_mapping(candidate, source=source)
            )
    return parsed


def _resolve_quality_scales(
    dataset,
    explicit: Mapping[str, Any] | None,
    dataset_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve physical scales without inferring undocumented constants."""

    explicit_scales = (
        _parse_quality_scale_mapping(
            explicit,
            source="explicit_argument",
        )
        if explicit is not None
        else {}
    )
    dataset_scales: dict[str, dict[str, Any]] = {}
    dataset_configuration = getattr(dataset, "quality_normalization", None)
    if isinstance(dataset_configuration, Mapping):
        dataset_scales = _parse_quality_scale_mapping(
            dataset_configuration,
            source="dataset.quality_normalization",
        )
    resolved: dict[str, dict[str, Any]] = {
        **dataset_scales,
    }
    manifest_error: str | None = None
    manifest = dataset_manifest
    if (
        manifest is None
        and len(set(dataset_scales) | set(explicit_scales))
        < len(_QUALITY_COMPONENTS)
    ):
        manifest_method = getattr(dataset, "manifest", None)
        if callable(manifest_method):
            try:
                candidate = manifest_method()
                if isinstance(candidate, Mapping):
                    manifest = candidate
                else:
                    manifest_error = "dataset.manifest() did not return a mapping"
            except Exception as error:  # pragma: no cover - defensive metadata boundary
                manifest_error = (
                    "dataset.manifest() failed: "
                    f"{type(error).__name__}: {error}"
                )
    if manifest is not None:
        resolved.update(_quality_scales_from_manifest(manifest))
    resolved.update(explicit_scales)
    components: dict[str, dict[str, Any]] = {}
    for component in _QUALITY_COMPONENTS:
        if component in resolved:
            components[component] = {
                "status": "available",
                **resolved[component],
            }
        else:
            components[component] = {
                "status": "unavailable",
                "scale": None,
                "offset": None,
                "unit": _QUALITY_UNITS[component],
                "source": None,
                "reason": (
                    manifest_error
                    or "no explicit quality_normalization scale in manifest "
                    "or function arguments"
                ),
            }
    available = sum(
        item["status"] == "available" for item in components.values()
    )
    return {
        "status": (
            "available"
            if available == len(_QUALITY_COMPONENTS)
            else "partial"
            if available
            else "unavailable"
        ),
        "components": components,
    }


def _infer_split_name(dataset) -> str:
    split = getattr(dataset, "split", None)
    if split is not None:
        return str(split)
    regime = getattr(dataset, "regime", None)
    name = getattr(regime, "name", None)
    return str(name) if name is not None else "unspecified"


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[PredictionBundle, dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probabilities, labels, sources, snr_values, sir_values = [], [], [], [], []
    target_profile_values: list[np.ndarray] = []
    target_profile_present: bool | None = None
    for batch in loader:
        view = _move_view(batch["view1"], device)
        output = model(view["x"])
        probabilities.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
        sources.append(batch["source_id"].numpy())
        snr_values.append(batch["view1"]["snr_db"].numpy())
        sir_values.append(batch["view1"]["sir_db"].numpy())
        current_profile_present = "target_profile_index" in batch["view1"]
        if target_profile_present is None:
            target_profile_present = current_profile_present
        elif target_profile_present != current_profile_present:
            raise RuntimeError(
                "target_profile_index presence changed between batches"
            )
        if current_profile_present:
            target_profile_values.append(
                batch["view1"]["target_profile_index"].numpy()
            )
    bundle = PredictionBundle(
        probabilities=np.concatenate(probabilities),
        labels=np.concatenate(labels),
        source_ids=np.concatenate(sources),
        snr_db=np.concatenate(snr_values),
        sir_db=np.concatenate(sir_values),
        target_profile_index=(
            np.concatenate(target_profile_values)
            if target_profile_present
            else None
        ),
    )
    return bundle, classification_metrics(bundle, len(dataset.modulations))


@torch.no_grad()
def auxiliary_task_metrics(
    model: nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str | None = None,
    view_name: str = "view1",
    jammer_threshold: float = 0.5,
    jammer_names: Sequence[str] | None = None,
    jammer_training_support_mask: Sequence[bool] | None = None,
    jammer_training_support_source: str | None = None,
    quality_denormalization: Mapping[str, Any] | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate jammer and physical-quality auxiliary heads.

    The result contains only JSON-native values and records ``split`` and
    ``seed`` explicitly.  Quality MAE is emitted in physical units only when
    each normalization scale is stated either by ``quality_denormalization``
    or by an explicit ``quality_normalization`` mapping in the dataset
    manifest.  Undocumented constants are never inferred from observed
    targets.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 < jammer_threshold < 1.0:
        raise ValueError("jammer_threshold must lie strictly between zero and one")
    if view_name not in {"view1", "view2"}:
        raise ValueError("view_name must be 'view1' or 'view2'")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    jammer_scores: list[np.ndarray] = []
    jammer_targets: list[np.ndarray] = []
    quality_predictions: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []
    quality_masks: list[np.ndarray] = []
    physical_truth_batches: dict[str, list[np.ndarray]] = {
        component: [] for component in _QUALITY_COMPONENTS
    }
    physical_truth_presence: dict[str, bool | None] = {
        component: None for component in _QUALITY_COMPONENTS
    }
    physical_truth_view_keys = {
        "snr_db": "snr_db",
        "sir_db": "sir_db",
        "doppler_hz": "doppler_hz",
    }
    jammer_head_state: bool | None = None
    quality_head_state: bool | None = None
    processed = 0
    for batch in loader:
        if view_name not in batch:
            raise KeyError(f"dataset batch has no {view_name}")
        view = _move_view(batch[view_name], device)
        output = model(view["x"])
        has_jammer = "jam_logits" in output
        has_quality = "quality" in output
        if jammer_head_state is None:
            jammer_head_state = has_jammer
            quality_head_state = has_quality
        elif jammer_head_state != has_jammer or quality_head_state != has_quality:
            raise RuntimeError("model output keys changed between evaluation batches")
        target_jammer = view["jam_labels"].detach().float().cpu().numpy()
        target_quality = view["quality"].detach().float().cpu().numpy()
        target_quality_mask = (
            view["quality_mask"].detach().float().cpu().numpy()
        )
        if target_jammer.ndim != 2:
            raise ValueError("jam_labels must have shape [batch, jammer_class]")
        if target_quality.ndim != 2 or target_quality.shape[1] != 3:
            raise ValueError("quality targets must have shape [batch, 3]")
        if target_quality_mask.shape != target_quality.shape:
            raise ValueError("quality_mask must have the same shape as quality")
        jammer_targets.append(target_jammer)
        quality_targets.append(target_quality)
        quality_masks.append(target_quality_mask)
        for component, view_key in physical_truth_view_keys.items():
            present = view_key in view
            prior = physical_truth_presence[component]
            if prior is None:
                physical_truth_presence[component] = present
            elif prior != present:
                raise RuntimeError(
                    f"physical target key {view_key} changed presence "
                    "between evaluation batches"
                )
            if present:
                physical = (
                    view[view_key]
                    .detach()
                    .float()
                    .reshape(-1)
                    .cpu()
                    .numpy()
                )
                if len(physical) != len(view["x"]):
                    raise ValueError(
                        f"{view_key} must provide one physical target per sample"
                    )
                physical_truth_batches[component].append(physical)
        if has_jammer:
            logits = output["jam_logits"]
            if tuple(logits.shape) != tuple(view["jam_labels"].shape):
                raise ValueError(
                    "jammer head/target shape mismatch: "
                    f"{tuple(logits.shape)} != {tuple(view['jam_labels'].shape)}"
                )
            jammer_scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        if has_quality:
            prediction = output["quality"]
            if tuple(prediction.shape) != tuple(view["quality"].shape):
                raise ValueError(
                    "quality head/target shape mismatch: "
                    f"{tuple(prediction.shape)} != {tuple(view['quality'].shape)}"
                )
            quality_predictions.append(prediction.float().cpu().numpy())
        processed += len(view["x"])

    scales = _resolve_quality_scales(
        dataset,
        quality_denormalization,
        dataset_manifest,
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "split": str(split) if split is not None else _infer_split_name(dataset),
        "seed": int(seed),
        "view": view_name,
        "sample_count": int(processed),
        "jammer_decision_threshold": float(jammer_threshold),
        "quality_denormalization": scales,
    }
    if processed == 0:
        reason = "dataset contains no samples"
        result["jammer_multilabel"] = {
            "status": "unavailable",
            "reason": reason,
        }
        result["quality"] = {
            "status": "unavailable",
            "reason": reason,
        }
        return result

    target_jammer_array = np.concatenate(jammer_targets, axis=0)
    target_quality_array = np.concatenate(quality_targets, axis=0)
    quality_mask_array = np.concatenate(quality_masks, axis=0)
    class_count = target_jammer_array.shape[1]
    if jammer_names is None:
        names = tuple(f"class_{index}" for index in range(class_count))
    else:
        names = tuple(str(name) for name in jammer_names)
        if len(names) != class_count:
            raise ValueError(
                "jammer_names length does not match jammer head width"
            )
        if len(set(names)) != len(names):
            raise ValueError("jammer_names must be unique")
    if jammer_training_support_mask is None:
        training_support = np.ones(class_count, dtype=bool)
        training_support_source = (
            jammer_training_support_source
            or "not_provided_legacy_all_columns"
        )
    else:
        training_support = np.asarray(
            jammer_training_support_mask,
            dtype=bool,
        ).reshape(-1)
        if len(training_support) != class_count:
            raise ValueError(
                "jammer_training_support_mask length does not match "
                "jammer head width"
            )
        training_support_source = (
            jammer_training_support_source
            or "explicit_training_support_mask"
        )
    result["jammer_training_support"] = {
        "mask": [bool(value) for value in training_support.tolist()],
        "source": training_support_source,
        "supported_names": [
            name
            for name, supported in zip(names, training_support, strict=True)
            if supported
        ],
        "unsupported_names": [
            name
            for name, supported in zip(names, training_support, strict=True)
            if not supported
        ],
        "held_or_excluded_logits_interpreted_as_trained_family_recognition": (
            False
        ),
        "training_gradient_semantics": (
            "metrics are restricted to supported columns; unsupported "
            "jammer logits had zero direct BCE derivative"
        ),
    }

    label_finite = np.isfinite(target_jammer_array)
    label_binary = np.isclose(target_jammer_array, 0.0, atol=1e-6) | np.isclose(
        target_jammer_array, 1.0, atol=1e-6
    )
    jammer_label_validity = {
        "status": (
            "valid"
            if bool(np.all(label_finite & label_binary))
            else "invalid"
        ),
        "nonfinite_entry_count": int(np.sum(~label_finite)),
        "nonbinary_entry_count": int(np.sum(label_finite & ~label_binary)),
        "entry_count": int(target_jammer_array.size),
    }
    if not jammer_head_state:
        jammer_result: dict[str, Any] = {
            "status": "unavailable",
            "reason": "model output has no jam_logits auxiliary head",
            "label_validity": jammer_label_validity,
        }
    elif jammer_label_validity["status"] != "valid":
        jammer_result = {
            "status": "unavailable",
            "reason": "jammer labels are not finite binary indicators",
            "label_validity": jammer_label_validity,
        }
    else:
        score_array = np.concatenate(jammer_scores, axis=0).astype(
            np.float64, copy=False
        )
        if not np.isfinite(score_array).all():
            jammer_result = {
                "status": "unavailable",
                "reason": "jammer probabilities contain nonfinite values",
                "label_validity": jammer_label_validity,
            }
        else:
            labels_bool = target_jammer_array > 0.5
            predictions_bool = score_array >= jammer_threshold
            per_class: dict[str, dict[str, Any]] = {}
            supported_f1_values: list[float] = []
            supported_auc_values: list[float] = []
            for class_index, name in enumerate(names):
                class_labels = labels_bool[:, class_index]
                class_predictions = predictions_bool[:, class_index]
                positives = int(class_labels.sum())
                negatives = int((~class_labels).sum())
                f1 = _binary_f1(class_labels, class_predictions)
                if not training_support[class_index]:
                    f1_record = _unavailable_metric(
                        "taxonomy column has no positive training support; "
                        "trained family-recognition interpretation is prohibited",
                        support=len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                    )
                elif positives == 0:
                    f1_record = _unavailable_metric(
                        "class has no positive ground-truth support",
                        support=len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                    )
                else:
                    f1_record = _available_metric(
                        float(f1["value"]),
                        len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                        true_positive=int(f1["true_positive"]),
                        false_positive=int(f1["false_positive"]),
                        false_negative=int(f1["false_negative"]),
                    )
                    supported_f1_values.append(float(f1["value"]))
                auc = _binary_auroc(class_labels, score_array[:, class_index])
                if not training_support[class_index]:
                    auc_record = _unavailable_metric(
                        "taxonomy column has no positive training support; "
                        "trained family-recognition interpretation is prohibited",
                        support=len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                    )
                elif positives == 0 or negatives == 0:
                    auc_record = _unavailable_metric(
                        "AUROC requires both positive and negative support",
                        support=len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                    )
                else:
                    auc_record = _available_metric(
                        auc,
                        len(class_labels),
                        positive_support=positives,
                        negative_support=negatives,
                    )
                    supported_auc_values.append(auc)
                per_class[name] = {
                    "training_supported": bool(
                        training_support[class_index]
                    ),
                    "f1": f1_record,
                    "auroc": auc_record,
                }

            supported_class_count = int(training_support.sum())
            if supported_class_count:
                supervised_labels = labels_bool[:, training_support]
                supervised_predictions = predictions_bool[:, training_support]
                supervised_scores = score_array[:, training_support]
                flattened_labels = supervised_labels.ravel()
                flattened_predictions = supervised_predictions.ravel()
                micro_f1 = _binary_f1(
                    flattened_labels,
                    flattened_predictions,
                )
                micro_positive_support = int(flattened_labels.sum())
                micro_negative_support = int((~flattened_labels).sum())
            else:
                flattened_labels = np.zeros(0, dtype=bool)
                flattened_predictions = np.zeros(0, dtype=bool)
                supervised_scores = np.zeros(0, dtype=np.float64)
                micro_f1 = {
                    "value": 0.0,
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                }
                micro_positive_support = 0
                micro_negative_support = 0
            if supported_class_count == 0:
                micro_f1_record = _unavailable_metric(
                    "no jammer taxonomy column has positive training support",
                    support=0,
                    positive_support=0,
                )
            elif micro_positive_support == 0:
                micro_f1_record = _unavailable_metric(
                    "training-supported columns have no positive "
                    "ground-truth support in this split",
                    support=flattened_labels.size,
                    positive_support=0,
                )
            else:
                micro_f1_record = _available_metric(
                    float(micro_f1["value"]),
                    flattened_labels.size,
                    positive_support=micro_positive_support,
                    true_positive=int(micro_f1["true_positive"]),
                    false_positive=int(micro_f1["false_positive"]),
                    false_negative=int(micro_f1["false_negative"]),
                )
            micro_auc = (
                _binary_auroc(
                    flattened_labels,
                    supervised_scores.ravel(),
                )
                if supported_class_count
                else 0.0
            )
            if (
                supported_class_count == 0
                or micro_positive_support == 0
                or micro_negative_support == 0
            ):
                micro_auc_record = _unavailable_metric(
                    "AUROC over training-supported columns requires both "
                    "positive and negative support",
                    support=flattened_labels.size,
                    positive_support=micro_positive_support,
                    negative_support=micro_negative_support,
                )
            else:
                micro_auc_record = _available_metric(
                    micro_auc,
                    flattened_labels.size,
                    positive_support=micro_positive_support,
                    negative_support=micro_negative_support,
                )
            jammer_result = {
                "status": "available",
                "label_validity": jammer_label_validity,
                "metric_scope": (
                    "training-supported taxonomy columns only"
                ),
                "training_supported_class_count": supported_class_count,
                "total_class_count": class_count,
                "micro_f1": micro_f1_record,
                "macro_f1": (
                    _available_metric(
                        float(np.mean(supported_f1_values)),
                        len(supported_f1_values),
                        averaging=(
                            "training-supported classes with positive "
                            "evaluation support"
                        ),
                        total_class_count=class_count,
                        training_supported_class_count=(
                            supported_class_count
                        ),
                    )
                    if supported_f1_values
                    else _unavailable_metric(
                        "no training-supported jammer class has positive "
                        "ground-truth support",
                        total_class_count=class_count,
                        training_supported_class_count=(
                            supported_class_count
                        ),
                    )
                ),
                "micro_auroc": micro_auc_record,
                "macro_auroc": (
                    _available_metric(
                        float(np.mean(supported_auc_values)),
                        len(supported_auc_values),
                        averaging=(
                            "training-supported classes with positive and "
                            "negative evaluation support"
                        ),
                        total_class_count=class_count,
                        training_supported_class_count=(
                            supported_class_count
                        ),
                    )
                    if supported_auc_values
                    else _unavailable_metric(
                        "no training-supported jammer class has both positive "
                        "and negative evaluation support",
                        total_class_count=class_count,
                        training_supported_class_count=(
                            supported_class_count
                        ),
                    )
                ),
                "per_class": per_class,
            }
    result["jammer_multilabel"] = jammer_result

    mask_finite = np.isfinite(quality_mask_array)
    mask_in_range = (quality_mask_array >= 0.0) & (quality_mask_array <= 1.0)
    mask_binary = np.isclose(quality_mask_array, 0.0, atol=1e-6) | np.isclose(
        quality_mask_array, 1.0, atol=1e-6
    )
    enabled = mask_finite & mask_binary & (quality_mask_array > 0.5)
    target_finite = np.isfinite(target_quality_array)
    prediction_array = (
        np.concatenate(quality_predictions, axis=0).astype(
            np.float64, copy=False
        )
        if quality_head_state
        else None
    )
    prediction_finite = (
        np.isfinite(prediction_array)
        if prediction_array is not None
        else np.zeros_like(target_finite, dtype=bool)
    )
    physical_truth = {
        component: (
            np.concatenate(physical_truth_batches[component], axis=0).astype(
                np.float64, copy=False
            )
            if physical_truth_batches[component]
            else None
        )
        for component in _QUALITY_COMPONENTS
    }
    component_valid_counts = {
        component: int(enabled[:, index].sum())
        for index, component in enumerate(_QUALITY_COMPONENTS)
    }
    mask_validity = {
        "status": (
            "valid"
            if bool(
                np.all(mask_finite & mask_in_range & mask_binary)
                and not np.any(enabled & ~target_finite)
            )
            else "invalid"
        ),
        "entry_count": int(quality_mask_array.size),
        "nonfinite_entry_count": int(np.sum(~mask_finite)),
        "out_of_range_entry_count": int(
            np.sum(mask_finite & ~mask_in_range)
        ),
        "nonbinary_entry_count": int(
            np.sum(mask_finite & mask_in_range & ~mask_binary)
        ),
        "enabled_target_nonfinite_count": int(
            np.sum(enabled & ~target_finite)
        ),
        "enabled_prediction_nonfinite_count": (
            int(np.sum(enabled & ~prediction_finite))
            if prediction_array is not None
            else None
        ),
        "enabled_count_by_component": component_valid_counts,
    }
    physical_mae: dict[str, dict[str, Any]] = {}
    normalization_consistency: dict[str, dict[str, Any]] = {}
    for component_index, component in enumerate(_QUALITY_COMPONENTS):
        scale_record = scales["components"][component]
        raw_truth = physical_truth[component]
        valid = (
            enabled[:, component_index]
            & target_finite[:, component_index]
            & prediction_finite[:, component_index]
        )
        if raw_truth is not None:
            valid = valid & np.isfinite(raw_truth)
        support = int(valid.sum())
        if prediction_array is None:
            physical_mae[component] = _unavailable_metric(
                "model output has no quality auxiliary head",
                support=support,
                unit=scale_record["unit"],
            )
        elif scale_record["status"] != "available":
            physical_mae[component] = _unavailable_metric(
                "physical denormalization scale is unavailable",
                support=support,
                unit=scale_record["unit"],
            )
        elif support == 0:
            physical_mae[component] = _unavailable_metric(
                "no finite quality target/prediction pair is enabled by quality_mask",
                support=0,
                unit=scale_record["unit"],
                scale=float(scale_record["scale"]),
                offset=float(scale_record["offset"]),
                scale_source=str(scale_record["source"]),
            )
        else:
            predicted_physical = (
                prediction_array[valid, component_index]
                * scale_record["scale"]
                + scale_record["offset"]
            )
            if raw_truth is not None:
                target_physical = raw_truth[valid]
                target_source = f"view.{physical_truth_view_keys[component]}"
            else:
                target_physical = (
                    target_quality_array[valid, component_index]
                    * scale_record["scale"]
                    + scale_record["offset"]
                )
                target_source = (
                    "normalized_quality_target_denormalized_with_same_scale"
                )
            physical_mae[component] = _available_metric(
                float(np.abs(predicted_physical - target_physical).mean()),
                support,
                unit=scale_record["unit"],
                scale=float(scale_record["scale"]),
                offset=float(scale_record["offset"]),
                scale_source=str(scale_record["source"]),
                physical_target_source=target_source,
            )
        if scale_record["status"] != "available":
            normalization_consistency[component] = _unavailable_metric(
                "physical denormalization scale is unavailable",
                unit=scale_record["unit"],
            )
        elif raw_truth is None:
            normalization_consistency[component] = _unavailable_metric(
                "dataset view does not expose an independent physical target",
                unit=scale_record["unit"],
            )
        else:
            consistency_valid = (
                enabled[:, component_index]
                & target_finite[:, component_index]
                & np.isfinite(raw_truth)
            )
            consistency_support = int(consistency_valid.sum())
            if consistency_support == 0:
                normalization_consistency[component] = _unavailable_metric(
                    "no finite enabled physical target pair",
                    unit=scale_record["unit"],
                )
            else:
                denormalized_target = (
                    target_quality_array[
                        consistency_valid, component_index
                    ]
                    * scale_record["scale"]
                    + scale_record["offset"]
                )
                discrepancy = np.abs(
                    denormalized_target - raw_truth[consistency_valid]
                )
                normalization_consistency[component] = {
                    "status": "available",
                    "support": consistency_support,
                    "mean_absolute_discrepancy": float(discrepancy.mean()),
                    "max_absolute_discrepancy": float(discrepancy.max()),
                    "unit": scale_record["unit"],
                    "physical_target_source": (
                        f"view.{physical_truth_view_keys[component]}"
                    ),
                }
    all_physical_mae_available = all(
        record["status"] == "available"
        for record in physical_mae.values()
    )
    quality_complete = bool(
        quality_head_state
        and mask_validity["status"] == "valid"
        and mask_validity["enabled_prediction_nonfinite_count"] == 0
        and all_physical_mae_available
    )
    result["quality"] = {
        "status": (
            "available"
            if quality_complete
            else "partial"
            if quality_head_state
            else "unavailable"
        ),
        "head_present": bool(quality_head_state),
        "quality_mask_validity": mask_validity,
        "physical_mae": physical_mae,
        "normalization_consistency": normalization_consistency,
    }
    return result


def _validated_strata_edges(
    edges: Sequence[float] | None,
    *,
    name: str,
) -> np.ndarray | None:
    if edges is None:
        return None
    array = np.asarray(tuple(edges), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} strata edges must be a nonempty 1-D sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} strata edges must be finite")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} strata edges must be strictly increasing")
    return array


def _stratum_label(lower: float, upper: float) -> str:
    lower_text = "-inf" if np.isneginf(lower) else f"{lower:g}"
    upper_text = "inf" if np.isposinf(upper) else f"{upper:g}"
    return f"[{lower_text},{upper_text})"


def _mechanism_strata(
    values: Sequence[float],
    edges: np.ndarray | None,
    *,
    target_transfer: Sequence[float],
    predicted_overlap: Sequence[float],
    oracle_unexplained: Sequence[float],
    oracle_ambiguity: Sequence[float],
    unexplained_direct_mae: Sequence[float],
    ambiguity_direct_mae: Sequence[float],
) -> dict[str, Any]:
    if edges is None:
        return {
            "status": "unavailable",
            "reason": "strata edges were not supplied by the evaluation protocol",
        }
    values_array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values_array)
    arrays = {
        "target_transfer": np.asarray(target_transfer, dtype=np.float64),
        "predicted_overlap": np.asarray(predicted_overlap, dtype=np.float64),
        "oracle_unexplained": np.asarray(oracle_unexplained, dtype=np.float64),
        "oracle_ambiguity": np.asarray(oracle_ambiguity, dtype=np.float64),
        "unexplained_direct_mae": np.asarray(
            unexplained_direct_mae, dtype=np.float64
        ),
        "ambiguity_direct_mae": np.asarray(
            ambiguity_direct_mae, dtype=np.float64
        ),
    }
    if any(len(array) != len(values_array) for array in arrays.values()):
        raise RuntimeError("mechanism stratum arrays are not sample-aligned")
    boundaries = np.concatenate(([-np.inf], edges, [np.inf]))
    records: dict[str, Any] = {}
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = finite & (values_array >= lower) & (values_array < upper)
        count = int(selected.sum())
        label = _stratum_label(float(lower), float(upper))
        if count == 0:
            records[label] = {
                "status": "unavailable",
                "sample_count": 0,
                "reason": "no finite samples in stratum",
            }
            continue
        transfer = arrays["target_transfer"][selected]
        predicted = arrays["predicted_overlap"][selected]
        unexplained = arrays["oracle_unexplained"][selected]
        ambiguity = arrays["oracle_ambiguity"][selected]
        if (
            count > 2
            and np.std(predicted) > 0
            and np.std(unexplained) > 0
        ):
            unexplained_spearman: float | None = float(
                spearmanr(predicted, unexplained).statistic
            )
        else:
            unexplained_spearman = None
        if (
            count > 2
            and np.std(predicted) > 0
            and np.std(ambiguity) > 0
        ):
            ambiguity_spearman: float | None = float(
                spearmanr(predicted, ambiguity).statistic
            )
        else:
            ambiguity_spearman = None
        records[label] = {
            "status": "available",
            "sample_count": count,
            "target_energy_transfer_ratio_mean": float(transfer.mean()),
            "target_energy_transfer_ratio_amplification_share": float(
                np.mean(transfer > 1.0)
            ),
            "target_energy_transfer_ratio_max": float(transfer.max()),
            "predicted_overlap_route_share": float(predicted.mean()),
            "unexplained_fraction_oracle_occupancy": float(
                unexplained.mean()
            ),
            "signal_jammer_ambiguity_oracle_occupancy": float(
                ambiguity.mean()
            ),
            "overlap_vs_unexplained_occupancy_spearman": (
                unexplained_spearman
            ),
            "overlap_vs_signal_jammer_ambiguity_occupancy_spearman": (
                ambiguity_spearman
            ),
            "overlap_vs_unexplained_fraction_weighted_mae": float(
                arrays["unexplained_direct_mae"][selected].mean()
            ),
            "overlap_vs_signal_jammer_ambiguity_weighted_mae": float(
                arrays["ambiguity_direct_mae"][selected].mean()
            ),
        }
    return {
        "status": "available",
        "interior_edges_db": [float(edge) for edge in edges],
        "finite_sample_count": int(finite.sum()),
        "excluded_nonfinite_sample_count": int((~finite).sum()),
        "strata": records,
    }


@torch.no_grad()
def mechanism_metrics(
    model: VIMDNet,
    teacher: PhysicalTriMaskTeacher,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    maximum_samples: int = 1_024,
    split: str | None = None,
    seed: int | None = None,
    snr_strata_edges_db: Sequence[float] | None = None,
    sir_strata_edges_db: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Probe the learned routing mechanism against component-level evidence.

    The modulation path multiplier can exceed one because of the learned
    residual ``rho``.  It is therefore reported as a target-energy transfer
    ratio, never as an implicitly bounded "retention" fraction.  Optional
    SNR/SIR strata must be supplied by the evaluation protocol; this function
    does not choose post-hoc boundaries from the evaluated samples.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    snr_edges = _validated_strata_edges(
        snr_strata_edges_db,
        name="SNR",
    )
    sir_edges = _validated_strata_edges(
        sir_strata_edges_db,
        name="SIR",
    )
    sample_limit = min(int(maximum_samples), len(dataset))
    if sample_limit <= 0:
        raise ValueError("mechanism evaluation dataset contains no samples")
    loader = DataLoader(
        Subset(dataset, range(sample_limit)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    teacher.eval()
    js_values, js_weights = [], []
    route_correlations: list[list[float]] = [[], [], []]
    route_weighted_error = np.zeros(3, dtype=np.float64)
    route_weight = 0.0
    predicted_route_occupancies: list[list[float]] = [[], [], []]
    oracle_route_occupancies: list[list[float]] = [[], [], []]
    component_evidence: dict[str, dict[str, Any]] = {
        "unexplained_fraction": {
            "direct_correlations": [],
            "attribution_correlations": [],
            "direct_weighted_error": 0.0,
            "attribution_weighted_error": 0.0,
            "oracle_occupancies": [],
            "attributed_predicted_occupancies": [],
            "sample_direct_mae": [],
        },
        "signal_jammer_ambiguity": {
            "direct_correlations": [],
            "attribution_correlations": [],
            "direct_weighted_error": 0.0,
            "attribution_weighted_error": 0.0,
            "oracle_occupancies": [],
            "attributed_predicted_occupancies": [],
            "sample_direct_mae": [],
        },
    }
    target_energy_transfer_ratio: list[float] = []
    jammer_leakage: list[float] = []
    feature_sir_gain: list[float] = []
    clean_only_target_energy_transfer_ratio: list[float] = []
    oracle_overlap_share: list[float] = []
    overlap_mask_share: list[float] = []
    metadata_overlap: list[float] = []
    sample_snr_db: list[float] = []
    sample_sir_db: list[float] = []
    lambda_values, rho_values, temperature_values = [], [], []
    processed = 0
    for batch in loader:
        view = _move_view(batch["view1"], device)
        output = model(view["x"])
        oracle = teacher.decompose(view["clean"], view["jammer"], view["unexplained"])
        target = oracle["masks"]
        if not torch.isfinite(output["masks"]).all():
            raise ValueError("predicted mechanism masks contain nonfinite values")
        if not torch.isfinite(target).all():
            raise ValueError("physical teacher masks contain nonfinite values")
        js_value = float(
            jensen_shannon_mask_loss(target, output["masks"])
        )
        if not np.isfinite(js_value):
            raise ValueError("mask Jensen--Shannon divergence is nonfinite")
        js_values.append(js_value)
        js_weights.append(len(view["x"]))

        predicted_numpy = output["masks"].detach().cpu().numpy()
        target_numpy = target.detach().cpu().numpy()
        power_numpy = oracle["component_power"].detach().cpu().numpy()
        unexplained_numpy = (
            oracle["unexplained_fraction"].detach().cpu().numpy()
        )
        ambiguity_numpy = (
            oracle["signal_jammer_ambiguity"].detach().cpu().numpy()
        )
        for sample_index in range(len(view["x"])):
            weights = power_numpy[sample_index].ravel()
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(
                    "mechanism evidence requires positive finite component "
                    f"energy for sample {processed + sample_index}"
                )
            route_weight += weight_sum
            for route_index in range(3):
                predicted_route = predicted_numpy[sample_index, route_index].ravel()
                target_route = target_numpy[sample_index, route_index].ravel()
                correlation = _weighted_correlation(
                    predicted_route,
                    target_route,
                    weights,
                )
                if np.isfinite(correlation):
                    route_correlations[route_index].append(correlation)
                route_weighted_error[route_index] += float(
                    np.sum(weights * np.abs(predicted_route - target_route))
                )
                predicted_route_occupancies[route_index].append(
                    float(np.sum(weights * predicted_route) / weight_sum)
                )
                oracle_route_occupancies[route_index].append(
                    float(np.sum(weights * target_route) / weight_sum)
                )
            predicted_overlap = predicted_numpy[
                sample_index, 2
            ].ravel()
            oracle_overlap = target_numpy[sample_index, 2].ravel()
            oracle_overlap_share.append(
                float(np.sum(weights * oracle_overlap) / weight_sum)
            )
            overlap_mask_share.append(
                float(np.sum(weights * predicted_overlap) / weight_sum)
            )
            constituent_arrays = {
                "unexplained_fraction": unexplained_numpy[
                    sample_index
                ].ravel(),
                "signal_jammer_ambiguity": ambiguity_numpy[
                    sample_index
                ].ravel(),
            }
            overlap_sum = (
                constituent_arrays["unexplained_fraction"]
                + constituent_arrays["signal_jammer_ambiguity"]
            )
            for component_name, component in constituent_arrays.items():
                evidence = component_evidence[component_name]
                direct_correlation = _weighted_correlation(
                    predicted_overlap,
                    component,
                    weights,
                )
                if np.isfinite(direct_correlation):
                    evidence["direct_correlations"].append(
                        direct_correlation
                    )
                direct_error = float(
                    np.sum(weights * np.abs(predicted_overlap - component))
                )
                evidence["direct_weighted_error"] += direct_error
                evidence["sample_direct_mae"].append(
                    direct_error / weight_sum
                )
                oracle_occupancy = float(
                    np.sum(weights * component) / weight_sum
                )
                evidence["oracle_occupancies"].append(oracle_occupancy)
                partition = np.divide(
                    component,
                    overlap_sum,
                    out=np.zeros_like(component),
                    where=overlap_sum > 1e-12,
                )
                attributed_prediction = predicted_overlap * partition
                attribution_correlation = _weighted_correlation(
                    attributed_prediction,
                    component,
                    weights,
                )
                if np.isfinite(attribution_correlation):
                    evidence["attribution_correlations"].append(
                        attribution_correlation
                    )
                evidence["attribution_weighted_error"] += float(
                    np.sum(
                        weights
                        * np.abs(attributed_prediction - component)
                    )
                )
                evidence["attributed_predicted_occupancies"].append(
                    float(
                        np.sum(weights * attributed_prediction)
                        / weight_sum
                    )
                )

        clean_spectrum = model.encode(view["clean"])
        jammer_spectrum = model.encode(view["jammer"])
        weight = output["modulation_weight"]
        clean_before = clean_spectrum.abs().square().mean(dim=(1, 2)).clamp_min(1e-9)
        jammer_before = jammer_spectrum.abs().square().mean(dim=(1, 2)).clamp_min(1e-9)
        transfer_ratio = (
            (weight * clean_spectrum).abs().square().mean(dim=(1, 2))
            / clean_before
        )
        beta = (weight * jammer_spectrum).abs().square().mean(dim=(1, 2)) / jammer_before
        active = view["quality_mask"][:, 1].gt(0.5)
        target_energy_transfer_ratio.extend(
            transfer_ratio.cpu().tolist()
        )
        clean_only_target_energy_transfer_ratio.extend(
            transfer_ratio[~active].cpu().tolist()
        )
        jammer_leakage.extend(beta[active].cpu().tolist())
        feature_sir_gain.extend(
            (
                10.0
                * torch.log10(
                    transfer_ratio[active].clamp_min(1e-9)
                    / beta[active].clamp_min(1e-9)
                )
            )
            .cpu()
            .tolist()
        )
        metadata_overlap.extend(batch["view1"]["overlap"].numpy().tolist())
        sample_snr_db.extend(batch["view1"]["snr_db"].numpy().tolist())
        sample_sir_db.extend(batch["view1"]["sir_db"].numpy().tolist())
        lambda_values.extend(output["lambda_overlap"].flatten().cpu().tolist())
        rho_values.extend(output["rho"].flatten().cpu().tolist())
        temperature_values.extend(output["temperature"].flatten().cpu().tolist())
        processed += len(view["x"])
    overlap_truth_array = np.asarray(oracle_overlap_share)
    overlap_share_array = np.asarray(overlap_mask_share)
    if (
        len(overlap_truth_array) > 2
        and np.std(overlap_truth_array) > 0
        and np.std(overlap_share_array) > 0
    ):
        overlap_spearman = float(
            spearmanr(overlap_truth_array, overlap_share_array, nan_policy="omit").statistic
        )
        rng = np.random.default_rng(20260727)
        null_statistics = np.asarray(
            [
                spearmanr(
                    rng.permutation(overlap_truth_array),
                    overlap_share_array,
                    nan_policy="omit",
                ).statistic
                for _ in range(1_000)
            ],
            dtype=np.float64,
        )
        permutation_p_value = float(
            (1 + np.sum(np.abs(null_statistics) >= abs(overlap_spearman)))
            / (len(null_statistics) + 1)
        )
        permutation_null95 = float(np.quantile(np.abs(null_statistics), 0.95))
    else:
        overlap_spearman = float("nan")
        permutation_p_value = float("nan")
        permutation_null95 = float("nan")
    route_names = ("signal", "jammer", "overlap_uncertainty")
    transfer_array = np.asarray(
        target_energy_transfer_ratio, dtype=np.float64
    )
    if not np.isfinite(transfer_array).all():
        raise ValueError("target-energy transfer ratios contain nonfinite values")
    clean_transfer_array = np.asarray(
        clean_only_target_energy_transfer_ratio, dtype=np.float64
    )
    component_results: dict[str, dict[str, Any]] = {}
    for component_name, evidence in component_evidence.items():
        component_results[component_name] = {
            "oracle_occupancy": _finite_mean(
                evidence["oracle_occupancies"]
            ),
            "predicted_oracle_conditioned_attribution_occupancy": (
                _finite_mean(
                    evidence["attributed_predicted_occupancies"]
                )
            ),
            "predicted_overlap_direct_weighted_correlation": _finite_mean(
                evidence["direct_correlations"]
            ),
            "predicted_overlap_direct_weighted_mae": float(
                evidence["direct_weighted_error"]
                / max(route_weight, 1e-12)
            ),
            "oracle_conditioned_attribution_weighted_correlation": (
                _finite_mean(evidence["attribution_correlations"])
            ),
            "oracle_conditioned_attribution_weighted_mae": float(
                evidence["attribution_weighted_error"]
                / max(route_weight, 1e-12)
            ),
            "interpretation": (
                "direct statistics compare the one learned overlap route "
                "with this teacher constituent; oracle-conditioned "
                "attribution partitions that route only for post-hoc "
                "diagnosis and is not an independently predicted mask"
            ),
        }
    result: dict[str, Any] = {
        "schema_version": 2,
        "split": str(split) if split is not None else _infer_split_name(dataset),
        "seed": int(seed) if seed is not None else None,
        "sample_count": int(processed),
        "active_jammer_sample_count": int(len(jammer_leakage)),
        "mask_js": float(np.average(js_values, weights=js_weights)),
        **{
            f"{route_name}_route_weighted_correlation": (
                float(np.mean(route_correlations[index]))
                if route_correlations[index]
                else float("nan")
            )
            for index, route_name in enumerate(route_names)
        },
        **{
            f"{route_name}_route_weighted_mae": float(
                route_weighted_error[index] / max(route_weight, 1e-12)
            )
            for index, route_name in enumerate(route_names)
        },
        **{
            f"predicted_{route_name}_route_share": float(
                np.mean(predicted_route_occupancies[index])
            )
            for index, route_name in enumerate(route_names)
        },
        **{
            f"oracle_{route_name}_route_share": float(
                np.mean(oracle_route_occupancies[index])
            )
            for index, route_name in enumerate(route_names)
        },
        "overlap_teacher_constituents": component_results,
        "unexplained_fraction_oracle_occupancy": component_results[
            "unexplained_fraction"
        ]["oracle_occupancy"],
        "unexplained_fraction_overlap_route_weighted_correlation": (
            component_results["unexplained_fraction"][
                "predicted_overlap_direct_weighted_correlation"
            ]
        ),
        "unexplained_fraction_overlap_route_weighted_mae": (
            component_results["unexplained_fraction"][
                "predicted_overlap_direct_weighted_mae"
            ]
        ),
        "signal_jammer_ambiguity_oracle_occupancy": component_results[
            "signal_jammer_ambiguity"
        ]["oracle_occupancy"],
        "signal_jammer_ambiguity_overlap_route_weighted_correlation": (
            component_results["signal_jammer_ambiguity"][
                "predicted_overlap_direct_weighted_correlation"
            ]
        ),
        "signal_jammer_ambiguity_overlap_route_weighted_mae": (
            component_results["signal_jammer_ambiguity"][
                "predicted_overlap_direct_weighted_mae"
            ]
        ),
        "target_energy_transfer_ratio_mean": float(transfer_array.mean()),
        "target_energy_transfer_ratio_amplification_share": float(
            np.mean(transfer_array > 1.0)
        ),
        "target_energy_transfer_ratio_amplification_count": int(
            np.sum(transfer_array > 1.0)
        ),
        "target_energy_transfer_ratio_max": float(transfer_array.max()),
        "clean_only_target_energy_transfer_ratio_mean": (
            float(clean_transfer_array.mean())
            if clean_transfer_array.size
            else float("nan")
        ),
        "clean_only_target_energy_transfer_ratio_amplification_share": (
            float(np.mean(clean_transfer_array > 1.0))
            if clean_transfer_array.size
            else float("nan")
        ),
        "clean_only_target_energy_transfer_ratio_max": (
            float(clean_transfer_array.max())
            if clean_transfer_array.size
            else float("nan")
        ),
        "jammer_leakage": (
            float(np.mean(jammer_leakage))
            if jammer_leakage
            else float("nan")
        ),
        "counterfactual_tf_sir_gain_db": (
            float(np.mean(feature_sir_gain))
            if feature_sir_gain
            else float("nan")
        ),
        "oracle_vs_predicted_overlap_spearman": overlap_spearman,
        "overlap_permutation_p_value": permutation_p_value,
        "overlap_permutation_null_abs95": permutation_null95,
        "metadata_support_overlap_mean": float(np.mean(metadata_overlap)),
        "lambda_overlap": float(np.mean(lambda_values)),
        "residual_rho": float(np.mean(rho_values)),
        "mask_temperature": float(np.mean(temperature_values)),
        "stratified_mechanism": {
            "snr_db": _mechanism_strata(
                sample_snr_db,
                snr_edges,
                target_transfer=target_energy_transfer_ratio,
                predicted_overlap=overlap_mask_share,
                oracle_unexplained=component_evidence[
                    "unexplained_fraction"
                ]["oracle_occupancies"],
                oracle_ambiguity=component_evidence[
                    "signal_jammer_ambiguity"
                ]["oracle_occupancies"],
                unexplained_direct_mae=component_evidence[
                    "unexplained_fraction"
                ]["sample_direct_mae"],
                ambiguity_direct_mae=component_evidence[
                    "signal_jammer_ambiguity"
                ]["sample_direct_mae"],
            ),
            "sir_db": _mechanism_strata(
                sample_sir_db,
                sir_edges,
                target_transfer=target_energy_transfer_ratio,
                predicted_overlap=overlap_mask_share,
                oracle_unexplained=component_evidence[
                    "unexplained_fraction"
                ]["oracle_occupancies"],
                oracle_ambiguity=component_evidence[
                    "signal_jammer_ambiguity"
                ]["oracle_occupancies"],
                unexplained_direct_mae=component_evidence[
                    "unexplained_fraction"
                ]["sample_direct_mae"],
                ambiguity_direct_mae=component_evidence[
                    "signal_jammer_ambiguity"
                ]["sample_direct_mae"],
            ),
        },
    }
    # Numeric aliases are retained so historical artifact readers keep
    # working.  The mapping makes their deprecated interpretation explicit.
    result.update(
        {
            "signal_retention": result[
                "target_energy_transfer_ratio_mean"
            ],
            "clean_only_signal_retention": result[
                "clean_only_target_energy_transfer_ratio_mean"
            ],
            "oracle_unexplained_share": result[
                "unexplained_fraction_oracle_occupancy"
            ],
            "oracle_signal_jammer_ambiguity_share": result[
                "signal_jammer_ambiguity_oracle_occupancy"
            ],
            "deprecated_metric_aliases": {
                "signal_retention": (
                    "target_energy_transfer_ratio_mean; old name incorrectly "
                    "suggested a value bounded by one"
                ),
                "clean_only_signal_retention": (
                    "clean_only_target_energy_transfer_ratio_mean; old name "
                    "incorrectly suggested a value bounded by one"
                ),
                "oracle_unexplained_share": (
                    "unexplained_fraction_oracle_occupancy"
                ),
                "oracle_signal_jammer_ambiguity_share": (
                    "signal_jammer_ambiguity_oracle_occupancy"
                ),
            },
        }
    )
    return result


@contextmanager
def _operation_counter(model: nn.Module):
    totals = {
        "convolution_linear_macs": 0,
        "recurrent_macs": 0,
    }
    handles = []

    def convolution_hook(module, inputs, output):
        batch = output.shape[0]
        output_elements = output.numel() // batch
        kernel = int(np.prod(module.kernel_size))
        operations_per_output = module.in_channels * kernel // module.groups
        totals["convolution_linear_macs"] += int(
            batch * output_elements * operations_per_output
        )

    def linear_hook(module, inputs, output):
        output_positions = output.numel() // module.out_features
        totals["convolution_linear_macs"] += int(
            output_positions * module.in_features * module.out_features
        )

    def lstm_hook(module, inputs, output):
        values = inputs[0]
        if module.batch_first:
            batch, steps = values.shape[:2]
        else:
            steps, batch = values.shape[:2]
        directions = 2 if module.bidirectional else 1
        layer_input = module.input_size
        total = 0
        for _ in range(module.num_layers):
            # Four gates each consume the current input and recurrent state.
            total += (
                batch
                * steps
                * directions
                * 4
                * module.hidden_size
                * (layer_input + module.hidden_size)
            )
            layer_input = module.hidden_size * directions
        totals["recurrent_macs"] += int(total)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.LSTM):
            handles.append(module.register_forward_hook(lstm_hook))
    try:
        yield totals
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def complexity_metrics(
    model: nn.Module,
    *,
    sample_length: int,
    device: torch.device,
    latency_runs: int = 60,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    dummy = torch.zeros(1, 2, sample_length, device=device)
    with _operation_counter(model) as counter:
        model(dummy)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    for _ in range(10):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations = []
    for _ in range(latency_runs):
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1_000.0)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    stft_modules = [module for module in model.modules() if isinstance(module, ComplexSTFT)]
    stft_real_operations = 0.0
    for module in stft_modules:
        frame_count = 1 + max(0, sample_length - module.n_fft) // module.hop_length
        # Standard order estimate for one complex FFT plus complex windowing.
        stft_real_operations += frame_count * (
            5.0 * module.n_fft * np.log2(module.n_fft) + 2.0 * module.n_fft
        )
    reported_frontend = getattr(model, "estimated_frontend_real_operations", None)
    if callable(reported_frontend):
        stft_real_operations += float(reported_frontend(sample_length))
    convolution_linear_macs = float(counter["convolution_linear_macs"])
    recurrent_macs = float(counter["recurrent_macs"])
    return {
        "parameters": float(parameters),
        "conv_linear_macs_excluding_stft": convolution_linear_macs,
        "recurrent_macs_excluding_stft": recurrent_macs,
        "conv_linear_recurrent_macs_excluding_stft": (
            convolution_linear_macs + recurrent_macs
        ),
        "stft_estimated_real_operations": float(stft_real_operations),
        "state_dict_megabytes": float(len(buffer.getvalue()) / (1024**2)),
        "latency_ms_p50": float(np.percentile(durations, 50)),
        "latency_ms_p95": float(np.percentile(durations, 95)),
        "latency_device": str(device),
    }
