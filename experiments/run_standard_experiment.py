"""Train and evaluate AMC models from an immutable MATLAB-nrTDL cache.

This runner deliberately keeps the standards claim narrow.  The cache uses
3GPP TR 38.901 TDL profile primitives through MATLAB ``nrTDLChannel``; it is
controlled synthetic evidence, not a system-level vehicular channel claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.data.split import assert_disjoint_source_ids  # noqa: E402
from vimd_amc.data.synthesis import SynthesisConfig  # noqa: E402
from vimd_amc.ablation import (  # noqa: E402
    PAPER_ABLATION_PROTOCOLS,
    paper_ablation_protocol,
)
from vimd_amc.evaluation import (  # noqa: E402
    auxiliary_task_metrics,
    complexity_metrics,
    mechanism_metrics,
    predict,
)
from vimd_amc.losses import VIMDLossWeights  # noqa: E402
from vimd_amc.metrics import (  # noqa: E402
    headline_paired_bootstrap,
    holm_adjust,
    paired_accuracy_macro_f1_bootstrap,
    paired_bundle_statistics,
)
from vimd_amc.models import baselines as baseline_models  # noqa: E402
from vimd_amc.models.baselines import (  # noqa: E402
    BackboneClassifier,
    SingleMaskClassifier,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import (  # noqa: E402
    DualMaskVIMDNet,
    PhysicalDualMaskTeacher,
    PhysicalTriMaskTeacher,
    VIMDNet,
)
from vimd_amc.models.temporal_vimd import VIMDTemporalNet  # noqa: E402
from vimd_amc.models.iqformer_route import (  # noqa: E402
    IQFormerRawOnlyControl,
    VIMDIQFormerRouteNet,
)
from vimd_amc.reproducibility import source_tree_record  # noqa: E402
from vimd_amc.standards import (  # noqa: E402
    CachedPairedAMCDataset,
    FACTOR_ISOLATED_SPLITS,
    factor_isolated_split_policies,
    validate_cached_components,
)
from vimd_amc.training import (  # noqa: E402
    TrainingConfig,
    TrainingObjective,
    checkpoint_selection_protocol,
    seed_everything,
    train_model,
)


REQUIRED_SPLITS = ("train", "validation", "heldout_channel")
STANDARDS_EVIDENCE_LABEL = (
    "3GPP TR 38.901 TDL-profile channel primitive via MATLAB nrTDLChannel; "
    "controlled synthetic AMC evidence"
)
EVIDENCE_GATE_POLICY_VERSION = "vimd-evidence-gate-v2"
FORMAL_RELEASE_DESIGNATION = "headline_formal_tvt_evidence"
FORMAL_METHOD_MODEL = "a5_vimd_full"
FORMAL_PRIMARY_REFERENCE_MODEL = "cssl_amc_supervised_adaptation"
FORMAL_REQUIRED_NONORACLE_BASELINES = (
    "a0_backbone",
    "mcldnn_reimplementation",
    "iqformer_inspired",
    "cssl_amc_supervised_adaptation",
)
FORMAL_HOLM_CANDIDATES = (
    "a0_backbone",
    "a1_single_mask",
    "a5_vimd_full",
    "mcldnn_reimplementation",
    "iqformer_inspired",
)
CLEAN_RETENTION_PROFILE_STRATA = {
    "clean_retention_seen_acd": (0, 2, 3),
    "clean_retention_held_be": (1, 4),
}
SCIENTIFIC_RELEASE_THRESHOLDS = {
    "hard_macro_f1_min_gain_pp_each_baseline": 5.0,
    "hard_ablation_controls": (
        "a1_single_mask",
        "a6_dual_full",
    ),
    "hard_ablation_strictly_positive": True,
    "ood_macro_f1_min_gain_pp": 3.0,
    "ood_required_pass_count": 2,
    "ood_regimes": (
        "unseen_jammer",
        "unseen_speed",
        "heldout_channel",
    ),
    "clean_macro_f1_min_point_gain_pp": -1.0,
    "clean_macro_f1_min_ci95_low_pp": -2.0,
    "mechanism_required_finite_fields": (
        "mask_js",
        "overlap_uncertainty_route_weighted_correlation",
        "target_energy_transfer_ratio_mean",
        "target_energy_transfer_ratio_amplification_share",
        "jammer_leakage",
        "oracle_vs_predicted_overlap_spearman",
        "overlap_permutation_p_value",
        "counterfactual_tf_sir_gain_db",
    ),
    "mechanism_nonnegative_fields": (
        "overlap_uncertainty_route_weighted_correlation",
        "oracle_vs_predicted_overlap_spearman",
    ),
    "oracle_spectral_ratio_field": "counterfactual_tf_sir_gain_db",
    "oracle_spectral_ratio_strictly_positive": True,
}
FORMAL_RELEASE_REQUIRED_GATES = frozenset(
    {
        "execution_complete",
        "recognized_cache_designation",
        "checksum_verification",
        "component_validation",
        "source_tree_stability",
        "class_taxonomy_complete",
        "minimum_split_samples",
        "minimum_per_class_samples",
        "minimum_algorithm_seeds",
        "preregistered_model_suite",
        "factor_isolated_protocol",
        "factor_split_sample_support",
        "factor_split_per_class_support",
        "explicit_reference_model",
        "predeclared_holm_candidate_family",
        "configured_checkpoint_selection_window",
        "eligible_selected_checkpoints",
        "jammer_auxiliary_training_support",
    }
)
SCREENING_DESIGNATIONS = frozenset(
    {
        "screening",
        "screening_not_formal_tvt_evidence",
    }
)
HEADLINE_DESIGNATIONS = frozenset(
    {
        "headline",
        "headline_formal_tvt_evidence",
        "formal_headline_tvt_evidence",
    }
)
PREREGISTERED_MODEL_SUITES: dict[str, tuple[str, ...]] = {
    "screening": (
        "a0_backbone",
        "a5_vimd_full",
        "mcldnn_reimplementation",
        "iqformer_inspired",
    ),
    "headline": (
        "a0_backbone",
        "a1_single_mask",
        "a2_tri_no_teacher",
        "a3_tri_teacher",
        "a4_tri_teacher_mtl",
        "a5_vimd_full",
        "a6_dual_full",
        "a7_vimd_no_residual",
        "mcldnn_reimplementation",
        "iqformer_inspired",
        "cssl_amc_supervised_adaptation",
    ),
}
EVIDENCE_MINIMUMS: dict[str, dict[str, Any]] = {
    # Administrative floors, not a substitute for prospective power analysis.
    "screening": {
        "seeds": 3,
        "split_samples": {
            "train": 200,
            "validation": 50,
            "heldout_channel": 200,
        },
        "per_class_samples": {
            "train": 20,
            "validation": 5,
            "heldout_channel": 20,
        },
    },
    "headline": {
        "seeds": 5,
        "split_samples": {
            "train": 1_000,
            "validation": 200,
            "heldout_channel": 500,
        },
        "per_class_samples": {
            "train": 100,
            "validation": 20,
            "heldout_channel": 50,
        },
    },
}


@dataclass(frozen=True)
class CacheContract:
    cache_root: Path
    cache_digest: str
    sample_length: int
    modulations: tuple[str, ...]
    num_jammers: int
    jammer_names: tuple[str, ...]
    split_sizes: dict[str, int]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BuiltStandardModel:
    model: nn.Module
    teacher: nn.Module | None
    objective: TrainingObjective
    loss_weights: VIMDLossWeights


ModelFactory = Callable[[int, int, ModelConfig], BuiltStandardModel]


def _baseline_factory(
    model_class: type[nn.Module],
    *,
    uses_model_config: bool,
) -> ModelFactory:
    def build(
        classes: int,
        _jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        if uses_model_config:
            model = model_class(classes, config)
        else:
            model = model_class(classes)
        return BuiltStandardModel(
            model=model,
            teacher=None,
            objective=TrainingObjective(
                name="paired_view_modulation_ce",
                loss_family="classification",
            ),
            loss_weights=VIMDLossWeights(),
        )

    return build


def available_model_factories() -> dict[str, ModelFactory]:
    """Return required models plus optional baselines present in this checkout."""

    def classification_model(
        model: nn.Module,
    ) -> BuiltStandardModel:
        return BuiltStandardModel(
            model=model,
            teacher=None,
            objective=TrainingObjective(
                name="paired_view_modulation_ce",
                loss_family="classification",
            ),
            loss_weights=VIMDLossWeights(),
        )

    def full_vimd(
        classes: int,
        jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        return BuiltStandardModel(
            model=VIMDNet(classes, jammers, config),
            teacher=PhysicalTriMaskTeacher(config),
            objective=TrainingObjective(
                name="full_vimd",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_cross_condition_contrastive=True,
                use_orthogonality=True,
            ),
            loss_weights=VIMDLossWeights(),
        )

    def diagnostic_temporal_vimd(
        classes: int,
        jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        return BuiltStandardModel(
            model=VIMDTemporalNet(classes, jammers, config),
            teacher=PhysicalTriMaskTeacher(config),
            objective=TrainingObjective(
                name="diagnostic_vimd_v2_temporal_full_objective",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_cross_condition_contrastive=True,
                use_orthogonality=True,
            ),
            loss_weights=VIMDLossWeights(),
        )

    def diagnostic_iqformer_route(
        classes: int,
        jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        weights = replace(
            VIMDLossWeights(),
            jammer=0.0,
            quality=0.0,
            orthogonality=0.0,
        )
        return BuiltStandardModel(
            model=VIMDIQFormerRouteNet(classes, jammers, config),
            teacher=PhysicalTriMaskTeacher(config),
            objective=TrainingObjective(
                name="diagnostic_vimd_v3_shared_iqformer_route",
                loss_family="vimd",
                use_mask_supervision=True,
                use_cross_condition_contrastive=True,
            ),
            loss_weights=weights,
        )

    def full_dual_vimd(
        classes: int,
        jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        return BuiltStandardModel(
            model=DualMaskVIMDNet(classes, jammers, config),
            teacher=PhysicalDualMaskTeacher(config),
            objective=TrainingObjective(
                name="full_vimd",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_cross_condition_contrastive=True,
                use_orthogonality=True,
            ),
            loss_weights=VIMDLossWeights(),
        )

    def full_vimd_without_residual(
        classes: int,
        jammers: int,
        config: ModelConfig,
    ) -> BuiltStandardModel:
        return BuiltStandardModel(
            model=VIMDNet(
                classes,
                jammers,
                config,
                use_residual=False,
            ),
            teacher=PhysicalTriMaskTeacher(config),
            objective=TrainingObjective(
                name="full_vimd",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_cross_condition_contrastive=True,
                use_orthogonality=True,
            ),
            loss_weights=VIMDLossWeights(),
        )

    def tri_ablation(
        *,
        objective: TrainingObjective,
        loss_weights: VIMDLossWeights,
        needs_teacher: bool,
    ) -> ModelFactory:
        def build(
            classes: int,
            jammers: int,
            config: ModelConfig,
        ) -> BuiltStandardModel:
            return BuiltStandardModel(
                model=VIMDNet(classes, jammers, config),
                teacher=(
                    PhysicalTriMaskTeacher(config)
                    if needs_teacher
                    else None
                ),
                objective=objective,
                loss_weights=loss_weights,
            )

        return build

    default_weights = VIMDLossWeights()
    no_auxiliary = replace(
        default_weights,
        jammer=0.0,
        quality=0.0,
        contrastive=0.0,
        orthogonality=0.0,
    )
    factories: dict[str, ModelFactory] = {
        "backbone": lambda classes, _jammers, config: classification_model(
            BackboneClassifier(classes, config)
        ),
        "single_mask": lambda classes, _jammers, config: classification_model(
            SingleMaskClassifier(classes, config)
        ),
        "vimd": full_vimd,
        "dual_mask": full_dual_vimd,
        "a0_backbone": lambda classes, _jammers, config: classification_model(
            BackboneClassifier(classes, config)
        ),
        "a1_single_mask": lambda classes, _jammers, config: classification_model(
            SingleMaskClassifier(classes, config)
        ),
        "a2_tri_no_teacher": tri_ablation(
            objective=TrainingObjective(
                name="tri_mask_no_teacher_no_mtl_no_xcc_with_residual",
                loss_family="vimd",
            ),
            loss_weights=replace(no_auxiliary, mask=0.0),
            needs_teacher=False,
        ),
        "a3_tri_teacher": tri_ablation(
            objective=TrainingObjective(
                name="tri_mask_fixed_teacher",
                loss_family="vimd",
                use_mask_supervision=True,
            ),
            loss_weights=no_auxiliary,
            needs_teacher=True,
        ),
        "a4_tri_teacher_mtl": tri_ablation(
            objective=TrainingObjective(
                name="tri_mask_teacher_mtl_no_xcc",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_orthogonality=True,
            ),
            loss_weights=replace(default_weights, contrastive=0.0),
            needs_teacher=True,
        ),
        "a5_vimd_full": full_vimd,
        "a6_dual_full": full_dual_vimd,
        "a7_vimd_no_residual": full_vimd_without_residual,
        # Explicitly outside the immutable manuscript A0--A7 registry.
        "diagnostic_vimd_v2_temporal": diagnostic_temporal_vimd,
        "diagnostic_iqformer_raw_only": lambda classes, _jammers, _config: (
            classification_model(IQFormerRawOnlyControl(classes))
        ),
        "diagnostic_vimd_v3_iqformer_route": diagnostic_iqformer_route,
    }
    optional = (
        ("mcldnn_reimplementation", "MCLDNNReimplementation", False),
        ("iqformer_inspired", "IQFormerInspiredClassifier", False),
        (
            "cssl_amc_supervised_adaptation",
            "CSSLAMCSupervisedAdaptation",
            False,
        ),
    )
    for public_name, attribute, uses_config in optional:
        model_class = getattr(baseline_models, attribute, None)
        if isinstance(model_class, type) and issubclass(model_class, nn.Module):
            factories[public_name] = _baseline_factory(
                model_class,
                uses_model_config=uses_config,
            )
    return factories


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-disjoint AMC experiments from an nrTDL cache."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "standards" / "cache_smoke",
    )
    parser.add_argument(
        "--models",
        default="backbone,single_mask,vimd",
        help=(
            "Comma-separated model names. Required models are backbone, "
            "single_mask, and vimd. The canonical a0_backbone through "
            "a7_vimd_no_residual ladder and locally available strong "
            "baselines are also accepted."
        ),
    )
    parser.add_argument("--seeds", default="17")
    parser.add_argument(
        "--reference-model",
        default=None,
        help=(
            "Explicit paired-comparison reference. If omitted, the runner may "
            "select a descriptive backbone anchor but never labels it strongest."
        ),
    )
    parser.add_argument(
        "--holm-candidates",
        default="",
        help=(
            "Comma-separated, predeclared candidate family for Holm correction. "
            "Families are corrected separately within each held-out regime and "
            "algorithm seed; validation is always excluded."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute cached component sums and measured SNR/SIR before training.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-start-epoch", type=int, default=2)
    parser.add_argument("--contrastive-start-epoch", type=int, default=5)
    parser.add_argument("--mask-ramp-epochs", type=int, default=3)
    parser.add_argument("--contrastive-ramp-epochs", type=int, default=3)
    parser.add_argument("--minimum-full-stage-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--spectral-channels", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--environment-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def evaluation_split_names(contract: CacheContract) -> tuple[str, ...]:
    """Return validation and every manifest-declared held-out regime."""

    return tuple(
        split for split in contract.split_sizes if split != "train"
    )


def manifest_split_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Resolve declared split order while validating manifest consistency."""

    file_splits = set(manifest.get("files", {}))
    source_splits = set(manifest.get("source_ids", {}))
    if file_splits != source_splits:
        raise ValueError(
            "cache files/source_ids declare different split sets: "
            f"files={sorted(file_splits)}, source_ids={sorted(source_splits)}"
        )
    configured = manifest.get("configuration", {}).get("split_sizes")
    if configured is None:
        ordered = tuple(str(split) for split in manifest["files"])
    else:
        try:
            ordered = tuple(str(item[0]) for item in configured)
            configured_sizes = {
                str(item[0]): int(item[1]) for item in configured
            }
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "configuration.split_sizes must contain [split, size] pairs"
            ) from error
        if len(ordered) != len(set(ordered)):
            raise ValueError("configuration.split_sizes contains duplicates")
        if set(ordered) != file_splits:
            raise ValueError(
                "configuration.split_sizes disagrees with files/source_ids"
            )
        if any(size <= 0 for size in configured_sizes.values()):
            raise ValueError("configuration.split_sizes must be positive")
    if not set(REQUIRED_SPLITS).issubset(ordered):
        missing = sorted(set(REQUIRED_SPLITS).difference(ordered))
        raise ValueError(f"required cache splits are absent: {missing}")
    return ordered


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _require_array_shape(
    manifest: dict[str, Any],
    split: str,
    name: str,
    expected: tuple[int | None, ...],
) -> list[int]:
    try:
        shape = manifest["files"][split][name]["shape"]
    except KeyError as error:
        raise ValueError(
            f"cache manifest is missing {split}/{name}.npy"
        ) from error
    if len(shape) != len(expected):
        raise ValueError(
            f"{split}/{name}.npy has rank {len(shape)}, expected {len(expected)}"
        )
    for actual, wanted in zip(shape, expected):
        if wanted is not None and int(actual) != wanted:
            raise ValueError(
                f"{split}/{name}.npy has shape {shape}, expected {expected}"
            )
    return [int(value) for value in shape]


def inspect_cache_contract(cache_root: Path) -> CacheContract:
    root = cache_root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("cache_digest", "configuration", "files", "source_ids"):
        if key not in manifest:
            raise ValueError(f"cache manifest is missing {key!r}")
    configuration = manifest["configuration"]
    modulations = tuple(str(value) for value in configuration["modulations"])
    if not modulations or len(set(modulations)) != len(modulations):
        raise ValueError("cache modulation taxonomy is empty or duplicated")
    sample_length = int(configuration["sample_length"])
    if sample_length < 8:
        raise ValueError("cache sample length is shorter than the model minimum")

    split_names = manifest_split_names(manifest)
    split_sizes: dict[str, int] = {}
    jammer_dimensions: set[int] = set()
    configured_sizes = {
        str(item[0]): int(item[1])
        for item in configuration.get("split_sizes", ())
    }
    for split in split_names:
        x_shape = _require_array_shape(
            manifest,
            split,
            "x",
            (None, 2, 2, sample_length),
        )
        size = x_shape[0]
        _require_array_shape(manifest, split, "clean", tuple(x_shape))
        _require_array_shape(manifest, split, "jammer", tuple(x_shape))
        _require_array_shape(manifest, split, "unexplained", tuple(x_shape))
        label_shape = _require_array_shape(
            manifest,
            split,
            "label",
            (size,),
        )
        source_shape = _require_array_shape(
            manifest,
            split,
            "source_id",
            (size,),
        )
        if label_shape != source_shape:
            raise ValueError(f"{split} label/source shapes disagree")
        jammer_shape = _require_array_shape(
            manifest,
            split,
            "jam_labels",
            (size, 2, None),
        )
        jammer_dimensions.add(jammer_shape[-1])
        for name, trailing in (
            ("quality", 3),
            ("quality_mask", 3),
        ):
            _require_array_shape(
                manifest,
                split,
                name,
                (size, 2, trailing),
            )
        scalar_view_arrays = ["snr_db", "sir_db", "overlap", "speed_kmh"]
        if int(manifest.get("schema_version", 1)) >= 2:
            scalar_view_arrays.append("doppler_hz")
        for name in scalar_view_arrays:
            _require_array_shape(manifest, split, name, (size, 2))
        if len(manifest["source_ids"][split]) != size:
            raise ValueError(
                f"{split} source-id manifest count does not match array size"
            )
        if configured_sizes and configured_sizes[split] != size:
            raise ValueError(
                f"{split} configured size disagrees with array size"
            )
        split_sizes[split] = size
    if len(jammer_dimensions) != 1:
        raise ValueError("jammer-label dimensions differ across cache splits")
    num_jammers = jammer_dimensions.pop()
    if num_jammers <= 0:
        raise ValueError("cache jammer-label dimension must be positive")
    declared_jammers = manifest.get("jammer_taxonomy")
    if declared_jammers is None:
        jammer_names = tuple(
            f"class_{index}" for index in range(num_jammers)
        )
    else:
        jammer_names = tuple(str(value) for value in declared_jammers)
        if (
            len(jammer_names) != num_jammers
            or len(set(jammer_names)) != len(jammer_names)
        ):
            raise ValueError(
                "jammer_taxonomy must uniquely name every jammer-label column"
            )
    assert_disjoint_source_ids(
        *(manifest["source_ids"][split] for split in split_names)
    )
    return CacheContract(
        cache_root=root,
        cache_digest=str(manifest["cache_digest"]),
        sample_length=sample_length,
        modulations=modulations,
        num_jammers=num_jammers,
        jammer_names=jammer_names,
        split_sizes=split_sizes,
        manifest=manifest,
    )


def load_cache_datasets(
    contract: CacheContract,
    *,
    verify_checksums: bool,
) -> dict[str, CachedPairedAMCDataset]:
    datasets: dict[str, CachedPairedAMCDataset] = {}
    try:
        for split in contract.split_sizes:
            dataset = CachedPairedAMCDataset(
                contract.cache_root,
                split,
                verify_checksums=verify_checksums,
            )
            # The generic evaluation code intentionally reads this taxonomy
            # from its dataset.  The immutable cache stores it in the root
            # manifest, so expose the same read-only tuple on each split.
            dataset.modulations = contract.modulations
            if len(dataset) != contract.split_sizes[split]:
                raise ValueError(f"{split} dataset length disagrees with manifest")
            labels = np.asarray(dataset._arrays["label"])
            if labels.size != len(dataset):
                raise ValueError(f"{split} label count disagrees with dataset length")
            if labels.min() < 0 or labels.max() >= len(contract.modulations):
                raise ValueError(
                    f"{split} contains labels outside the declared taxonomy"
                )
            first = dataset[0]
            for view_name in ("view1", "view2"):
                view = first[view_name]
                if tuple(view["x"].shape) != (2, contract.sample_length):
                    raise ValueError(
                        f"{split}/{view_name} sample shape is incompatible"
                    )
                if tuple(view["jam_labels"].shape) != (
                    contract.num_jammers,
                ):
                    raise ValueError(
                        f"{split}/{view_name} jammer dimension is incompatible"
                    )
                if tuple(view["quality"].shape) != (3,):
                    raise ValueError(
                        f"{split}/{view_name} quality dimension is incompatible"
                    )
            datasets[split] = dataset
    except Exception:
        for dataset in datasets.values():
            dataset.close()
        raise
    assert_disjoint_source_ids(
        *(datasets[split].source_ids() for split in contract.split_sizes)
    )
    return datasets


def jammer_training_support_contract(
    contract: CacheContract,
    train_dataset: CachedPairedAMCDataset,
) -> dict[str, Any]:
    """Freeze which jammer columns are genuinely supervised in training."""

    labels = np.asarray(
        train_dataset._arrays["jam_labels"],
        dtype=np.float64,
    )
    if labels.shape != (
        len(train_dataset),
        2,
        contract.num_jammers,
    ):
        raise ValueError("train jammer-label array violates cache contract")
    if not np.all(np.isfinite(labels)):
        raise ValueError("train jammer-label array contains nonfinite values")
    if not np.all(np.isclose(labels, 0.0) | np.isclose(labels, 1.0)):
        raise ValueError("train jammer-label array is not binary")
    positive_counts = labels.sum(axis=(0, 1)).astype(np.int64)
    observed_positive = {
        name
        for name, count in zip(
            contract.jammer_names,
            positive_counts,
            strict=True,
        )
        if int(count) > 0
    }
    split_policy = contract.manifest.get("preregistered_split_policy", {})
    train_policy = split_policy.get("train", {})
    unseen_policy = split_policy.get("unseen_jammer", {})
    expected_seen = {
        str(name)
        for name in train_policy.get("jammer_choices", ())
        if str(name) != "none"
    }
    held_out = {
        str(name)
        for name in unseen_policy.get("jammer_choices", ())
        if str(name) != "none"
    }.difference(expected_seen)
    exclusions = contract.manifest.get("protocol_exclusions", {})
    excluded_in_taxonomy = set(contract.jammer_names).intersection(exclusions)
    supervised = observed_positive.difference(
        held_out | excluded_in_taxonomy
    )
    support_mask = [
        name in supervised for name in contract.jammer_names
    ]
    class_records = {
        name: {
            "index": index,
            "positive_training_view_count": int(positive_counts[index]),
            "training_supported": bool(support_mask[index]),
            "role": (
                "excluded_from_primary_auxiliary_training"
                if name in excluded_in_taxonomy
                else "held_out_unknown_family_not_trained"
                if name in held_out
                else "seen_supervised_family"
                if support_mask[index]
                else "zero_positive_training_support"
            ),
        }
        for index, name in enumerate(contract.jammer_names)
    }
    formal_contract_valid = bool(
        int(contract.manifest.get("schema_version", 1)) >= 2
        and expected_seen
        and observed_positive == expected_seen
        and not observed_positive.intersection(
            held_out | excluded_in_taxonomy
        )
        and any(support_mask)
    )
    return {
        "schema_version": 1,
        "training_split": "train",
        "taxonomy": list(contract.jammer_names),
        "positive_training_view_counts": [
            int(value) for value in positive_counts.tolist()
        ],
        "support_mask": support_mask,
        "supported_training_labels": [
            name
            for name, supported in zip(
                contract.jammer_names,
                support_mask,
                strict=True,
            )
            if supported
        ],
        "held_out_labels": sorted(held_out),
        "excluded_taxonomy_labels": sorted(excluded_in_taxonomy),
        "excluded_protocol_conditions_not_in_taxonomy": sorted(
            set(exclusions).difference(contract.jammer_names)
        ),
        "missing_expected_seen_positive_support": sorted(
            expected_seen.difference(observed_positive)
        ),
        "unexpected_positive_training_labels": sorted(
            observed_positive.difference(expected_seen)
        ),
        "class_records": class_records,
        "support_mask_source": (
            "positive labels observed in immutable train split, constrained "
            "by preregistered seen/held/excluded semantics"
        ),
        "loss_reduction": (
            "binary cross entropy mean over batch and supported columns only"
        ),
        "unsupported_columns_receive_training_loss_or_gradient": False,
        "unsupported_logit_columns_receive_direct_bce_gradient": False,
        "shared_backbone_gradient_scope": (
            "shared features are updated through supported jammer columns; "
            "d(loss)/d(unsupported jammer logit) is exactly zero"
        ),
        "held_or_excluded_logits_are_trained_family_recognizers": False,
        "formal_contract_valid": formal_contract_valid,
    }


def checkpoint_selection_audit(
    results: list[dict[str, Any]] | None,
    *,
    models: list[str],
    seeds: list[int],
    expected_criterion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require one non-fallback, selection-eligible checkpoint per fit."""

    expected = {(model, int(seed)) for model in models for seed in seeds}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[list[Any]] = []
    malformed_results: list[str] = []
    raw_results = results if isinstance(results, list) else []
    for index, result in enumerate(raw_results):
        if not isinstance(result, dict):
            malformed_results.append(f"index{index}:not_an_object")
            continue
        try:
            key = (str(result.get("model")), int(result.get("seed", -1)))
        except (TypeError, ValueError, OverflowError):
            malformed_results.append(f"index{index}:invalid_model_or_seed")
            continue
        if key in observed:
            duplicates.append([key[0], key[1]])
        observed[key] = result
    per_fit: dict[str, Any] = {}
    invalid: list[str] = []
    for model, seed in sorted(expected):
        key_text = f"{model}/seed{seed}"
        result = observed.get((model, seed))
        if result is None:
            per_fit[key_text] = {
                "passed": False,
                "reason": "training result missing",
            }
            invalid.append(key_text)
            continue
        training = result.get("training", {})
        training = training if isinstance(training, dict) else {}
        selection = training.get("checkpoint_selection", {})
        selection = selection if isinstance(selection, dict) else {}
        history = training.get("history", [])
        history = history if isinstance(history, list) else []
        selected_epoch = selection.get("selected_epoch")
        criterion = selection.get("criterion", {})
        criterion = criterion if isinstance(criterion, dict) else {}
        selected_rows: list[dict[str, Any]] = []
        eligible_history_count = 0
        conversion_valid = True
        try:
            if isinstance(selected_epoch, bool) or not isinstance(
                selected_epoch, int
            ):
                conversion_valid = False
            else:
                for row in history:
                    if not isinstance(row, dict):
                        conversion_valid = False
                        continue
                    if (
                        float(
                            row.get(
                                "checkpoint_selection_eligible", 0.0
                            )
                        )
                        > 0.5
                    ):
                        eligible_history_count += 1
                    if int(float(row.get("epoch", -1))) == selected_epoch:
                        selected_rows.append(row)
            eligible_count = int(
                selection.get("eligible_checkpoint_count", 0)
            )
            label_smoothing = float(
                criterion.get("label_smoothing", -1.0)
            )
            tolerance = float(
                criterion.get("strict_improvement_tolerance", -1.0)
            )
            patience = int(
                criterion.get("patience_eligible_epochs", 0)
            )
            selected_row_eligible = (
                float(
                    selected_rows[0].get(
                        "checkpoint_selection_eligible", 0.0
                    )
                )
                if len(selected_rows) == 1
                else 0.0
            )
            selected_validation_loss = float(
                selection.get("selected_validation_loss")
            )
            selected_row_validation_loss = (
                float(selected_rows[0].get("validation_loss"))
                if len(selected_rows) == 1
                else float("nan")
            )
        except (TypeError, ValueError, OverflowError):
            conversion_valid = False
            eligible_count = 0
            label_smoothing = -1.0
            tolerance = -1.0
            patience = 0
            selected_row_eligible = 0.0
            selected_validation_loss = float("nan")
            selected_row_validation_loss = float("nan")
        criterion_complete = bool(
            criterion.get("view") == "view1"
            and criterion.get("loss") == "modulation_cross_entropy"
            and label_smoothing == 0.0
            and criterion.get("direction") == "minimize"
            and tolerance == 1e-5
            and criterion.get("auxiliary_losses_included") is False
            and criterion.get("validation_loader_shuffle") is False
            and patience > 0
            and (
                expected_criterion is None
                or criterion == expected_criterion
            )
        )
        passed = bool(
            conversion_valid
            and selection.get("status")
            == "eligible_validation_checkpoint_selected"
            and selection.get("selected_checkpoint_eligible") is True
            and selection.get("fallback_used") is False
            and eligible_count >= 1
            and eligible_count == eligible_history_count
            and len(selected_rows) == 1
            and selected_row_eligible > 0.5
            and math.isfinite(selected_validation_loss)
            and math.isfinite(selected_row_validation_loss)
            and math.isclose(
                selected_validation_loss,
                selected_row_validation_loss,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and criterion_complete
        )
        per_fit[key_text] = {
            "passed": passed,
            "selected_epoch": selected_epoch,
            "eligible_checkpoint_count": selection.get(
                "eligible_checkpoint_count"
            ),
            "fallback_used": selection.get("fallback_used"),
            "criterion": criterion,
            "criterion_matches_configuration": (
                expected_criterion is None
                or criterion == expected_criterion
            ),
            "selected_validation_loss": selection.get(
                "selected_validation_loss"
            ),
        }
        if not passed:
            invalid.append(key_text)
    unexpected = sorted(
        f"{model}/seed{seed}"
        for model, seed in set(observed).difference(expected)
    )
    return {
        "passed": (
            not invalid
            and not duplicates
            and not unexpected
            and not malformed_results
            and set(observed) == expected
        ),
        "expected_fit_count": len(expected),
        "observed_fit_count": len(observed),
        "invalid_or_missing_fits": invalid,
        "duplicate_fits": duplicates,
        "malformed_results": malformed_results,
        "unexpected_fits": unexpected,
        "expected_criterion": expected_criterion,
        "per_fit": per_fit,
    }


def jammer_support_result_audit(
    support: dict[str, Any] | None,
    results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Verify auxiliary fits used exactly the serialized support mask."""

    if not isinstance(support, dict):
        return {
            "passed": False,
            "reason": "jammer training-support contract missing",
        }
    mask = support.get("support_mask")
    mask_valid = bool(
        isinstance(mask, list)
        and mask
        and all(isinstance(value, bool) for value in mask)
        and any(mask)
        and support.get("formal_contract_valid") is True
        and support.get(
            "unsupported_columns_receive_training_loss_or_gradient"
        )
        is False
        and support.get(
            "unsupported_logit_columns_receive_direct_bce_gradient"
        )
        is False
        and support.get(
            "held_or_excluded_logits_are_trained_family_recognizers"
        )
        is False
    )
    auxiliary_fits = 0
    invalid_fits: list[str] = []
    if not isinstance(results, list):
        return {
            "passed": False,
            "reason": "training results missing or malformed",
            "support_contract_valid": mask_valid,
        }
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            invalid_fits.append(f"index{index}:result_not_object")
            continue
        objective = result.get("objective", {})
        objective = objective if isinstance(objective, dict) else {}
        if not bool(objective.get("use_jammer_auxiliary")):
            continue
        auxiliary_fits += 1
        training = result.get("training", {})
        training = training if isinstance(training, dict) else {}
        training_support = training.get(
            "jammer_auxiliary_training", {}
        )
        training_support = (
            training_support
            if isinstance(training_support, dict)
            else {}
        )
        if not (
            training_support.get("enabled") is True
            and training_support.get("support_mask") == mask
            and training_support.get(
                "unsupported_columns_receive_loss_or_gradient"
            )
            is False
            and training_support.get(
                "unsupported_logit_columns_receive_direct_bce_gradient"
            )
            is False
        ):
            invalid_fits.append(
                f"{result.get('model')}/seed{result.get('seed')}"
            )
    return {
        "passed": mask_valid and auxiliary_fits > 0 and not invalid_fits,
        "support_contract_valid": mask_valid,
        "auxiliary_fit_count": auxiliary_fits,
        "invalid_auxiliary_fits": invalid_fits,
        "support_mask": mask,
        "supported_training_labels": support.get(
            "supported_training_labels", []
        ),
        "held_out_labels": support.get("held_out_labels", []),
        "excluded_taxonomy_labels": support.get(
            "excluded_taxonomy_labels", []
        ),
    }


def configured_checkpoint_window_audit(
    configuration: TrainingConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate that the configured epoch budget reaches selection."""

    if isinstance(configuration, TrainingConfig):
        config = configuration
    elif isinstance(configuration, dict):
        try:
            config = TrainingConfig(
                **{
                    key: configuration[key]
                    for key in asdict(TrainingConfig())
                    if key in configuration
                }
            )
        except (TypeError, ValueError) as error:
            return {
                "passed": False,
                "reason": f"invalid training configuration: {error}",
            }
    else:
        return {
            "passed": False,
            "reason": "training configuration missing",
        }
    protocol = checkpoint_selection_protocol(config)
    return {
        "passed": int(protocol["configured_eligible_epoch_count"]) >= 1,
        "epochs": int(config.epochs),
        **protocol,
    }


def make_model_config(
    *,
    sample_length: int,
    n_fft: int | None,
    hop_length: int | None,
    spectral_channels: int,
    embedding_dim: int,
    environment_dim: int,
    dropout: float,
) -> ModelConfig:
    resolved_n_fft = int(n_fft or min(64, 2 ** int(math.log2(sample_length))))
    resolved_hop = int(hop_length or max(1, resolved_n_fft // 4))
    if resolved_n_fft < 8 or resolved_n_fft > sample_length:
        raise ValueError(
            f"n_fft={resolved_n_fft} must be in [8, sample_length={sample_length}]"
        )
    if resolved_hop <= 0 or resolved_hop > resolved_n_fft:
        raise ValueError("hop_length must be in [1, n_fft]")
    if spectral_channels <= 0 or embedding_dim <= 0 or environment_dim <= 0:
        raise ValueError("model dimensions must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    return ModelConfig(
        feature_channels=max(32, spectral_channels),
        environment_dim=environment_dim,
        embedding_dim=embedding_dim,
        spectral_channels=spectral_channels,
        n_fft=resolved_n_fft,
        hop_length=resolved_hop,
        dropout=dropout,
    )


def dataset_support_audit(
    datasets: dict[str, CachedPairedAMCDataset],
    contract: CacheContract,
) -> dict[str, Any]:
    split_audits: dict[str, Any] = {}
    every_split_complete = True
    for split in contract.split_sizes:
        labels = np.asarray(datasets[split]._arrays["label"], dtype=np.int64)
        counts = np.bincount(
            labels,
            minlength=len(contract.modulations),
        )
        complete = bool(np.all(counts > 0))
        every_split_complete = every_split_complete and complete
        split_audits[split] = {
            "sample_count": int(len(labels)),
            "minimum_class_count": int(counts.min()),
            "class_counts": {
                modulation: int(counts[index])
                for index, modulation in enumerate(contract.modulations)
            },
            "observed_class_count": int(np.count_nonzero(counts)),
            "declared_class_count": len(contract.modulations),
            "class_complete": complete,
        }
    return {
        "splits": split_audits,
        "all_splits_class_complete": every_split_complete,
        "evidence_readiness": (
            "controlled_standard_channel_experiment"
            if every_split_complete
            else "pipeline_smoke_only_incomplete_class_support"
        ),
        "interpretation": (
            "Incomplete class support makes classification and paired "
            "statistics diagnostic only."
            if not every_split_complete
            else (
                "Class support is complete; statistical adequacy still "
                "depends on sample size, seeds, and the preregistered protocol."
            )
        ),
    }


def resolve_evidence_tier(designation: Any) -> str:
    """Map the cache's explicit designation to a closed evidence tier."""

    normalized = str(designation or "").strip().lower()
    if normalized in SCREENING_DESIGNATIONS:
        return "screening"
    if normalized in HEADLINE_DESIGNATIONS:
        return "headline"
    return "unrecognized_or_missing"


def _component_validation_passed(
    component_validation: dict[str, dict[str, float]] | None,
    expected_splits: tuple[str, ...] = REQUIRED_SPLITS,
) -> bool:
    if component_validation is None or set(component_validation) != set(
        expected_splits
    ):
        return False
    required_metrics = {
        "max_component_error",
        "max_snr_error_db",
        "max_sir_error_db",
        "min_active_jammer_power",
    }
    for metrics in component_validation.values():
        if not required_metrics.issubset(metrics):
            return False
        if not all(np.isfinite(float(metrics[name])) for name in required_metrics):
            return False
    return True


def factor_protocol_gate(contract: CacheContract) -> dict[str, Any]:
    """Audit the exact nine-split protocol required for headline evidence."""

    manifest = contract.manifest
    expected = set(FACTOR_ISOLATED_SPLITS)
    actual = set(contract.split_sizes)
    policy = manifest.get("preregistered_split_policy")
    coverage = manifest.get("factor_coverage")
    roles = manifest.get("split_roles")
    normalization = manifest.get("quality_normalization")
    jammer_taxonomy = manifest.get("jammer_taxonomy")
    exclusions = manifest.get("protocol_exclusions")
    clean_coverage = (
        coverage.get("clean_retention", {})
        if isinstance(coverage, dict)
        else {}
    )
    clean_actual = clean_coverage.get("actual", {})

    def normalization_entry_valid(name: str, unit: str) -> bool:
        try:
            entry = normalization[name]
            scale = float(entry["scale"])
            return (
                isinstance(entry, dict)
                and np.isfinite(scale)
                and scale > 0
                and entry.get("unit") == unit
            )
        except (KeyError, TypeError, ValueError):
            return False

    normalization_valid = (
        isinstance(normalization, dict)
        and set(normalization) == {"snr_db", "sir_db", "doppler_hz"}
        and normalization_entry_valid("snr_db", "dB")
        and normalization_entry_valid("sir_db", "dB")
        and normalization_entry_valid("doppler_hz", "Hz")
    )
    expected_policy = {
        record.split: {
            **json_safe(asdict(record)),
            "size": contract.split_sizes.get(record.split),
        }
        for record in factor_isolated_split_policies(
            contract.split_sizes
            if set(contract.split_sizes) == expected
            else None
        )
    }
    locked_policy_match = (
        isinstance(policy, dict)
        and set(policy) == expected
        and all(
            policy[split] == expected_policy[split]
            for split in FACTOR_ISOLATED_SPLITS
        )
    )
    coverage_valid = (
        isinstance(coverage, dict)
        and set(coverage) == expected
        and all(
            record.get("all_actual_values_within_policy") is True
            for record in coverage.values()
        )
    )
    clean_view_count = int(clean_coverage.get("view_count", -1))
    clean_invalid_count = int(
        clean_coverage.get("sir_invalid_view_count", -2)
    )
    clean_valid = (
        clean_view_count > 0
        and clean_invalid_count == clean_view_count
        and clean_coverage.get("sir_valid_view_count") == 0
        and clean_actual.get("jammer_choices") == ["none"]
        and clean_coverage.get("actual_clean_fraction") == 1.0
    )
    checks = {
        "schema_version_at_least_2": int(
            manifest.get("schema_version", 0)
        )
        >= 2,
        "exact_split_set": actual == expected,
        "preregistered_policy_complete": (
            isinstance(policy, dict) and set(policy) == expected
        ),
        "locked_policy_exact_match": locked_policy_match,
        "split_roles_complete": (
            isinstance(roles, dict) and set(roles) == expected
        ),
        "factor_coverage_in_policy": coverage_valid,
        "clean_sir_validity": clean_valid,
        "quality_normalization_explicit": normalization_valid,
        "jammer_taxonomy_explicit": (
            isinstance(jammer_taxonomy, list)
            and tuple(str(value) for value in jammer_taxonomy)
            == SynthesisConfig().jammer_types
            and contract.jammer_names == SynthesisConfig().jammer_types
        ),
        "ambiguous_conditions_excluded": (
            isinstance(exclusions, dict)
            and set(exclusions) == {"cochannel", "mixed"}
            and exclusions["cochannel"].get("status")
            == "excluded_from_primary_modulation_classification"
            and exclusions["mixed"].get("status")
            == "excluded_from_locked_factor_protocol"
            and isinstance(policy, dict)
            and all(
                "cochannel" not in record.get("jammer_choices", [])
                and "mixed" not in record.get("jammer_choices", [])
                for record in policy.values()
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_splits": list(contract.split_sizes),
        "required_splits": list(FACTOR_ISOLATED_SPLITS),
        "clean_view_count": clean_view_count,
        "clean_sir_invalid_view_count": clean_invalid_count,
    }


def assess_evidence_eligibility(
    *,
    contract: CacheContract,
    support_audit: dict[str, Any],
    checksums_verified: bool,
    component_validation: dict[str, dict[str, float]] | None,
    models: list[str],
    seeds: list[int],
    execution_status: str,
    explicit_reference_model: str | None,
    holm_candidates: list[str],
    source_tree_unchanged: bool = True,
    training_configuration: TrainingConfig | dict[str, Any] | None = None,
    training_results: list[dict[str, Any]] | None = None,
    jammer_training_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply non-negotiable screening/headline evidence promotion gates."""

    designation = contract.manifest["configuration"].get(
        "evidence_designation"
    )
    tier = resolve_evidence_tier(designation)
    policy_tier = tier if tier in EVIDENCE_MINIMUMS else "screening"
    minimums = EVIDENCE_MINIMUMS[policy_tier]
    required_suite = PREREGISTERED_MODEL_SUITES[policy_tier]

    split_sample_checks = {
        split: (
            int(support_audit["splits"][split]["sample_count"])
            >= int(minimums["split_samples"][split])
        )
        for split in REQUIRED_SPLITS
    }
    per_class_checks = {
        split: (
            int(support_audit["splits"][split]["minimum_class_count"])
            >= int(minimums["per_class_samples"][split])
        )
        for split in REQUIRED_SPLITS
    }
    missing_models = sorted(set(required_suite).difference(models))
    validation_splits = tuple(contract.split_sizes)
    component_passed = _component_validation_passed(
        component_validation,
        validation_splits,
    )
    reference_predeclared = (
        explicit_reference_model is not None
        and explicit_reference_model in models
    )
    primary_reference_predeclared = (
        reference_predeclared
        and explicit_reference_model == FORMAL_PRIMARY_REFERENCE_MODEL
    )
    holm_family_predeclared = (
        primary_reference_predeclared
        and tuple(holm_candidates) == FORMAL_HOLM_CANDIDATES
        and all(candidate in models for candidate in holm_candidates)
    )

    gates: dict[str, dict[str, Any]] = {
        "execution_complete": {
            "passed": execution_status == "complete",
            "actual": execution_status,
            "required": "complete",
        },
        "recognized_cache_designation": {
            "passed": tier != "unrecognized_or_missing",
            "actual": designation,
            "required": sorted(
                SCREENING_DESIGNATIONS | HEADLINE_DESIGNATIONS
            ),
        },
        "checksum_verification": {
            "passed": bool(checksums_verified),
            "actual": (
                "passed" if checksums_verified else "skipped_by_cli"
            ),
            "required": "passed",
        },
        "component_validation": {
            "passed": component_passed,
            "actual": (
                "passed"
                if component_passed
                else "not_run_or_incomplete"
            ),
            "required": list(validation_splits),
        },
        "source_tree_stability": {
            "passed": bool(source_tree_unchanged),
            "actual": (
                "unchanged"
                if source_tree_unchanged
                else "source_tree_mutated_during_execution"
            ),
            "required": "unchanged",
        },
        "class_taxonomy_complete": {
            "passed": bool(support_audit["all_splits_class_complete"]),
            "actual": bool(support_audit["all_splits_class_complete"]),
            "required": True,
        },
        "minimum_split_samples": {
            "passed": all(split_sample_checks.values()),
            "actual": {
                split: int(
                    support_audit["splits"][split]["sample_count"]
                )
                for split in REQUIRED_SPLITS
            },
            "required": minimums["split_samples"],
            "per_split_passed": split_sample_checks,
        },
        "minimum_per_class_samples": {
            "passed": all(per_class_checks.values()),
            "actual": {
                split: int(
                    support_audit["splits"][split]["minimum_class_count"]
                )
                for split in REQUIRED_SPLITS
            },
            "required": minimums["per_class_samples"],
            "per_split_passed": per_class_checks,
        },
        "minimum_algorithm_seeds": {
            "passed": len(seeds) >= int(minimums["seeds"]),
            "actual": len(seeds),
            "required": int(minimums["seeds"]),
        },
        "preregistered_model_suite": {
            "passed": not missing_models,
            "actual": models,
            "required": list(required_suite),
            "missing": missing_models,
        },
    }
    if policy_tier == "headline":
        factor_audit = factor_protocol_gate(contract)
        checkpoint_window_audit = configured_checkpoint_window_audit(
            training_configuration
        )
        checkpoint_audit = checkpoint_selection_audit(
            training_results,
            models=models,
            seeds=seeds,
            expected_criterion=checkpoint_window_audit.get("criterion"),
        )
        jammer_support_audit = jammer_support_result_audit(
            jammer_training_support,
            training_results,
        )
        heldout_sample_floor = int(
            minimums["split_samples"]["heldout_channel"]
        )
        heldout_class_floor = int(
            minimums["per_class_samples"]["heldout_channel"]
        )
        factor_sample_checks = {
            split: (
                split in support_audit["splits"]
                and int(
                    support_audit["splits"][split]["sample_count"]
                )
                >= (
                    int(minimums["split_samples"][split])
                    if split in minimums["split_samples"]
                    else heldout_sample_floor
                )
            )
            for split in FACTOR_ISOLATED_SPLITS
        }
        factor_class_checks = {
            split: (
                split in support_audit["splits"]
                and int(
                    support_audit["splits"][split]["minimum_class_count"]
                )
                >= (
                    int(minimums["per_class_samples"][split])
                    if split in minimums["per_class_samples"]
                    else heldout_class_floor
                )
            )
            for split in FACTOR_ISOLATED_SPLITS
        }
        gates["factor_isolated_protocol"] = {
            "passed": factor_audit["passed"],
            "actual": factor_audit,
            "required": (
                "schema>=2 with exact nine-split preregistration, in-policy "
                "coverage, clean SIR invalidity, and physical normalization"
            ),
        }
        gates["factor_split_sample_support"] = {
            "passed": all(factor_sample_checks.values()),
            "actual": {
                split: (
                    int(support_audit["splits"][split]["sample_count"])
                    if split in support_audit["splits"]
                    else None
                )
                for split in FACTOR_ISOLATED_SPLITS
            },
            "required": {
                split: (
                    int(minimums["split_samples"][split])
                    if split in minimums["split_samples"]
                    else heldout_sample_floor
                )
                for split in FACTOR_ISOLATED_SPLITS
            },
            "per_split_passed": factor_sample_checks,
        }
        gates["factor_split_per_class_support"] = {
            "passed": all(factor_class_checks.values()),
            "actual": {
                split: (
                    int(
                        support_audit["splits"][split][
                            "minimum_class_count"
                        ]
                    )
                    if split in support_audit["splits"]
                    else None
                )
                for split in FACTOR_ISOLATED_SPLITS
            },
            "required": {
                split: (
                    int(minimums["per_class_samples"][split])
                    if split in minimums["per_class_samples"]
                    else heldout_class_floor
                )
                for split in FACTOR_ISOLATED_SPLITS
            },
            "per_split_passed": factor_class_checks,
        }
        gates["explicit_reference_model"] = {
            "passed": primary_reference_predeclared,
            "actual": explicit_reference_model,
            "required": FORMAL_PRIMARY_REFERENCE_MODEL,
        }
        gates["predeclared_holm_candidate_family"] = {
            "passed": holm_family_predeclared,
            "actual": holm_candidates,
            "required": (
                "exact preregistered candidate family selected before training: "
                + ",".join(FORMAL_HOLM_CANDIDATES)
            ),
        }
        gates["configured_checkpoint_selection_window"] = {
            "passed": checkpoint_window_audit["passed"],
            "actual": checkpoint_window_audit,
            "required": (
                "epoch budget includes at least one epoch at or after the "
                "locked full-objective checkpoint-selection start"
            ),
        }
        gates["eligible_selected_checkpoints"] = {
            "passed": checkpoint_audit["passed"],
            "actual": checkpoint_audit,
            "required": (
                "every model/seed fit selects a validation checkpoint from "
                "the eligible window; final-state fallback is prohibited"
            ),
        }
        gates["jammer_auxiliary_training_support"] = {
            "passed": jammer_support_audit["passed"],
            "actual": jammer_support_audit,
            "required": (
                "positive-support mask serialized and applied to every jammer "
                "auxiliary fit; held/excluded columns receive no BCE gradient"
            ),
        }

    designated_tier_passed = (
        tier in EVIDENCE_MINIMUMS
        and all(bool(gate["passed"]) for gate in gates.values())
    )
    screening_eligible = bool(designated_tier_passed)
    headline_eligible = bool(
        designated_tier_passed and tier == "headline"
    )
    reasons = [
        f"{name}: required={gate['required']!r}, actual={gate['actual']!r}"
        for name, gate in gates.items()
        if not bool(gate["passed"])
    ]
    if not source_tree_unchanged:
        reasons.insert(0, "source_tree_mutated_during_execution")
    if tier == "screening":
        reasons.append(
            "cache designation permits screening only and explicitly excludes "
            "formal TVT headline evidence"
        )
    elif tier == "unrecognized_or_missing":
        reasons.append(
            "manifest configuration.evidence_designation is missing or not "
            "recognized by the locked policy"
        )
    return {
        "policy_version": EVIDENCE_GATE_POLICY_VERSION,
        "cache_designation": designation,
        "requested_tier": tier,
        "eligible": headline_eligible,
        "formal_paper_evidence_eligible": headline_eligible,
        "headline_eligible": headline_eligible,
        "screening_eligible": screening_eligible,
        "eligible_for_designated_tier": designated_tier_passed,
        "reasons": reasons,
        "gates": gates,
        "administrative_floor_warning": (
            "Passing these floors does not establish statistical power, "
            "scientific validity, or publication readiness."
        ),
    }


def submission_release_source_gate(
    run_record: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed before any submission macro generator reads a run."""

    reasons: list[str] = []
    eligibility = run_record.get("evidence_eligibility", {})
    eligibility = eligibility if isinstance(eligibility, dict) else {}
    if eligibility.get("policy_version") != EVIDENCE_GATE_POLICY_VERSION:
        reasons.append("evidence_gate_policy_version_is_not_current")
    if run_record.get("execution_status") != "complete":
        reasons.append("execution_status_is_not_complete")
    if eligibility.get("eligible") is not True:
        reasons.append("evidence_eligibility_eligible_is_not_true")
    if eligibility.get("formal_paper_evidence_eligible") is not True:
        reasons.append("formal_paper_evidence_eligible_is_not_true")
    if eligibility.get("headline_eligible") is not True:
        reasons.append("headline_eligible_is_not_true")
    serialized_gates = eligibility.get("gates")
    if not isinstance(serialized_gates, dict) or not serialized_gates:
        reasons.append("evidence_gates_missing_or_malformed")
        serialized_gates = {}
    missing_required_gates = sorted(
        FORMAL_RELEASE_REQUIRED_GATES.difference(serialized_gates)
    )
    if missing_required_gates:
        reasons.append(
            "required_formal_evidence_gates_missing:"
            + ",".join(missing_required_gates)
        )
    failed_gates = sorted(
        str(name)
        for name, gate in serialized_gates.items()
        if not isinstance(gate, dict) or gate.get("passed") is not True
    )
    if failed_gates:
        reasons.append("failed_evidence_gates:" + ",".join(failed_gates))
    source_audit = run_record.get("source_tree_execution_audit", {})
    source_audit = source_audit if isinstance(source_audit, dict) else {}
    if source_audit.get("unchanged") is not True:
        reasons.append("source_tree_execution_audit_not_unchanged")
    designation = eligibility.get("cache_designation")
    if designation != FORMAL_RELEASE_DESIGNATION:
        reasons.append("cache_designation_is_not_formal_headline")
    models = run_record.get("models")
    seeds = run_record.get("seeds")
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(model, str) and model for model in models)
        or len(set(models)) != len(models)
    ):
        reasons.append("models_missing_malformed_or_duplicated")
        models = []
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool)
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        reasons.append("seeds_missing_malformed_or_duplicated")
        seeds = []
    results = run_record.get("results")
    expected_result_count = len(models) * len(seeds)
    if (
        not isinstance(results, list)
        or not results
        or len(results) != expected_result_count
    ):
        reasons.append("result_set_missing_or_incomplete")
        results = results if isinstance(results, list) else []
    configured_window = configured_checkpoint_window_audit(
        run_record.get("training_configuration")
    )
    if configured_window.get("passed") is not True:
        reasons.append("configured_checkpoint_selection_window_invalid")
    checkpoint_audit = checkpoint_selection_audit(
        results,
        models=models,
        seeds=seeds,
        expected_criterion=configured_window.get("criterion"),
    )
    if checkpoint_audit.get("passed") is not True:
        reasons.append("checkpoint_selection_audit_failed")
    required_metrics = (
        "accuracy",
        "macro_f1",
        "worst_recall",
        "nll",
        "ece",
    )
    placeholder_results: list[str] = []
    invalid_checkpoints: list[str] = []
    configured_splits = run_record.get("splits")
    expected_regimes = (
        {str(split) for split in configured_splits if str(split) != "train"}
        if isinstance(configured_splits, dict)
        else set()
    )
    if not expected_regimes:
        reasons.append("evaluation_regime_contract_missing")
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            placeholder_results.append(f"index{index}:result_not_object")
            continue
        key = f"{result.get('model')}/seed{result.get('seed')}"
        training = result.get("training", {})
        training = training if isinstance(training, dict) else {}
        selection = training.get("checkpoint_selection", {})
        selection = selection if isinstance(selection, dict) else {}
        if not (
            selection.get("selected_checkpoint_eligible") is True
            and selection.get("fallback_used") is False
            and isinstance(selection.get("selected_epoch"), int)
        ):
            invalid_checkpoints.append(key)
        regimes = result.get("regimes")
        if not isinstance(regimes, dict) or not regimes:
            placeholder_results.append(f"{key}:regimes")
            continue
        if set(regimes) != expected_regimes:
            placeholder_results.append(f"{key}:regime_set")
        for split, metrics in regimes.items():
            for metric in required_metrics:
                value = (
                    metrics.get(metric)
                    if isinstance(metrics, dict)
                    else None
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    placeholder_results.append(
                        f"{key}:{split}:{metric}"
                    )
    if invalid_checkpoints:
        reasons.append(
            "invalid_or_fallback_checkpoints:"
            + ",".join(sorted(invalid_checkpoints))
        )
    if placeholder_results:
        reasons.append(
            "required_result_cells_missing_or_placeholder:"
            + ",".join(sorted(placeholder_results))
        )
    statistical_outputs = run_record.get("statistical_outputs", {})
    statistical_outputs = (
        statistical_outputs
        if isinstance(statistical_outputs, dict)
        else {}
    )
    for name in (
        "single_seed_pairs",
        "multi_seed_headline_pairs",
        "holm_families",
    ):
        if name not in statistical_outputs:
            reasons.append(f"statistical_output_missing:{name}")
    permitted = not reasons
    return {
        "schema_version": 1,
        "macro_generation_permitted": permitted,
        "submission_unlocked": False,
        "reasons": reasons,
        "required_next_step": (
            "run the fail-closed submission macro generator with a complete "
            "non-placeholder macro-value manifest"
            if permitted
            else "resolve every listed source-run failure before macro generation"
        ),
    }


def resolve_reference_model(
    models: list[str],
    requested: str | None,
) -> tuple[str | None, str]:
    """Resolve a paired-comparison anchor without making a strength claim."""

    if requested is not None:
        normalized = requested.strip()
        if not normalized:
            raise ValueError("--reference-model cannot be blank")
        if normalized not in models:
            raise ValueError("--reference-model must be among --models")
        return normalized, "explicit_cli"
    if "a0_backbone" in models:
        return "a0_backbone", "automatic_descriptive_anchor_not_strongest"
    if "backbone" in models:
        return "backbone", "automatic_descriptive_anchor_not_strongest"
    return None, "none"


def analysis_seed(base_seed: int, *tokens: Any) -> int:
    """Derive a stable, recorded bootstrap seed without Python hash state."""

    payload = "|".join([str(int(base_seed)), *(str(token) for token in tokens)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def validate_training_config(config: TrainingConfig) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("optimizer parameters are invalid")
    for name in ("mask_start_epoch", "contrastive_start_epoch"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "mask_ramp_epochs",
        "contrastive_ramp_epochs",
        "minimum_full_stage_epochs",
        "patience",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")


def environment_record(
    device: torch.device,
    *,
    contract: CacheContract,
    checksums_verified: bool,
    source_tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cache_digest": contract.cache_digest,
        "checksum_verification": (
            "passed" if checksums_verified else "skipped_by_cli"
        ),
        "standards_evidence_label": STANDARDS_EVIDENCE_LABEL,
        "v2x_system_level_compliance_claimed": False,
        "source_tree": (
            source_tree
            if source_tree is not None
            else source_tree_record(ROOT, Path(__file__))
        ),
    }


def source_tree_execution_audit(
    start: dict[str, Any],
) -> dict[str, Any]:
    """Compare executable-source fingerprints at run start and completion."""

    end = source_tree_record(ROOT, Path(__file__))
    unchanged = (
        start.get("aggregate_digest") == end.get("aggregate_digest")
        and start.get("files") == end.get("files")
    )
    return {
        "unchanged": unchanged,
        "reason": (
            "unchanged"
            if unchanged
            else "source_tree_mutated_during_execution"
        ),
        "start": start,
        "end": end,
    }


def split_manifest_reference(
    contract: CacheContract,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": int(contract.manifest.get("schema_version", 1)),
        "source_cache_manifest": str(contract.cache_root / "manifest.json"),
        "cache_digest": contract.cache_digest,
        "split": split,
        "size": contract.split_sizes[split],
        "source_ids": contract.manifest["source_ids"][split],
        "modulations": contract.modulations,
        "jammer_taxonomy": contract.jammer_names,
        "standards_evidence_label": STANDARDS_EVIDENCE_LABEL,
        "profile_policy": contract.manifest.get("profile_policy"),
        "split_role": contract.manifest.get("split_roles", {}).get(split),
        "preregistered_split_policy": contract.manifest.get(
            "preregistered_split_policy", {}
        ).get(split),
        "factor_coverage": contract.manifest.get(
            "factor_coverage", {}
        ).get(split),
        "component_audit": contract.manifest.get(
            "component_audit", {}
        ).get(split),
        "quality_normalization": contract.manifest.get(
            "quality_normalization"
        ),
        "protocol_exclusions": contract.manifest.get(
            "protocol_exclusions"
        ),
        "files": contract.manifest["files"][split],
    }


def model_provenance(model: nn.Module) -> dict[str, Any]:
    provenance = getattr(model, "provenance", None)
    if isinstance(provenance, dict):
        return provenance
    return {
        "display_name": model.__class__.__name__,
        "claim_level": "local implementation",
    }


def _aggregate_seed_metrics(
    metrics_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = ("accuracy", "macro_f1", "worst_recall", "nll", "ece")
    for (model_name, regime_name), group in metrics_frame.groupby(
        ["model", "regime"]
    ):
        row: dict[str, Any] = {
            "model": model_name,
            "regime": regime_name,
            "seed_count": int(group["seed"].nunique()),
        }
        for metric_name in metric_names:
            values = group[metric_name].astype(float)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    arguments = parse_arguments()
    if arguments.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if arguments.latency_runs <= 0 or arguments.bootstrap_draws <= 0:
        raise ValueError("latency runs and bootstrap draws must be positive")
    if arguments.bootstrap_seed < 0:
        raise ValueError("--bootstrap-seed must be nonnegative")
    torch.set_num_threads(arguments.cpu_threads)
    torch.set_num_interop_threads(1)
    source_tree_start = source_tree_record(ROOT, Path(__file__))

    contract = inspect_cache_contract(arguments.cache_root)
    model_config = make_model_config(
        sample_length=contract.sample_length,
        n_fft=arguments.n_fft,
        hop_length=arguments.hop_length,
        spectral_channels=arguments.spectral_channels,
        embedding_dim=arguments.embedding_dim,
        environment_dim=arguments.environment_dim,
        dropout=arguments.dropout,
    )
    training_config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        mask_start_epoch=arguments.mask_start_epoch,
        contrastive_start_epoch=arguments.contrastive_start_epoch,
        mask_ramp_epochs=arguments.mask_ramp_epochs,
        contrastive_ramp_epochs=arguments.contrastive_ramp_epochs,
        minimum_full_stage_epochs=arguments.minimum_full_stage_epochs,
        patience=arguments.patience,
        use_amp=arguments.use_amp,
    )
    validate_training_config(training_config)
    factories = available_model_factories()
    models = [
        name.strip() for name in arguments.models.split(",") if name.strip()
    ]
    if not models:
        raise ValueError("at least one model must be selected")
    unknown = set(models).difference(factories)
    if unknown:
        raise ValueError(
            f"unknown models {sorted(unknown)}; available={sorted(factories)}"
        )
    if len(models) != len(set(models)):
        raise ValueError("model list contains duplicates")
    seeds = [int(seed) for seed in arguments.seeds.split(",") if seed.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed list must be nonempty and contain no duplicates")
    explicit_reference_model = (
        arguments.reference_model.strip()
        if arguments.reference_model is not None
        else None
    )
    reference_model, reference_selection = resolve_reference_model(
        models,
        explicit_reference_model,
    )
    holm_candidates = [
        name.strip()
        for name in arguments.holm_candidates.split(",")
        if name.strip()
    ]
    if len(holm_candidates) != len(set(holm_candidates)):
        raise ValueError("--holm-candidates contains duplicates")
    unknown_holm_candidates = set(holm_candidates).difference(models)
    if unknown_holm_candidates:
        raise ValueError(
            "--holm-candidates must be selected models; "
            f"unknown={sorted(unknown_holm_candidates)}"
        )
    if reference_model is not None and reference_model in holm_candidates:
        raise ValueError("--holm-candidates cannot contain the reference model")
    if (
        contract.manifest["configuration"].get("evidence_designation")
        == FORMAL_RELEASE_DESIGNATION
    ):
        if reference_model != FORMAL_PRIMARY_REFERENCE_MODEL:
            raise ValueError(
                "formal headline execution requires predeclared primary "
                f"reference {FORMAL_PRIMARY_REFERENCE_MODEL}"
            )
        if tuple(holm_candidates) != FORMAL_HOLM_CANDIDATES:
            raise ValueError(
                "formal headline execution requires the exact preregistered "
                "Holm family: " + ",".join(FORMAL_HOLM_CANDIDATES)
            )
    device = resolve_device(arguments.device)

    datasets = load_cache_datasets(
        contract,
        verify_checksums=arguments.verify_checksums,
    )
    component_validation: dict[str, dict[str, float]] | None = None
    if arguments.validate_components:
        component_validation = {
            split: validate_cached_components(datasets[split])
            for split in contract.split_sizes
        }
    support_audit = dataset_support_audit(datasets, contract)
    jammer_training_support = jammer_training_support_contract(
        contract,
        datasets["train"],
    )
    initial_eligibility = assess_evidence_eligibility(
        contract=contract,
        support_audit=support_audit,
        checksums_verified=bool(arguments.verify_checksums),
        component_validation=component_validation,
        models=models,
        seeds=seeds,
        execution_status="running",
        explicit_reference_model=explicit_reference_model,
        holm_candidates=holm_candidates,
        training_configuration=training_config,
        training_results=[],
        jammer_training_support=jammer_training_support,
    )

    run_id = arguments.run_id or (
        "standard_tdl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_directory = arguments.output.resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    for split in contract.split_sizes:
        write_json(
            run_directory / "manifests" / f"{split}.json",
            split_manifest_reference(contract, split),
        )
    write_json(
        run_directory / "manifests" / "cache_reference.json",
        {
            "cache_root": contract.cache_root,
            "cache_manifest": contract.cache_root / "manifest.json",
            "cache_digest": contract.cache_digest,
            "checksum_verification": (
                "passed"
                if arguments.verify_checksums
                else "skipped_by_cli"
            ),
            "component_validation": component_validation,
            "dataset_support_audit": support_audit,
            "jammer_auxiliary_training_support": jammer_training_support,
            "evidence_designation": contract.manifest["configuration"].get(
                "evidence_designation"
            ),
            "evidence_gate_policy_version": EVIDENCE_GATE_POLICY_VERSION,
            "standards_evidence_label": STANDARDS_EVIDENCE_LABEL,
            "v2x_system_level_compliance_claimed": False,
        },
    )

    run_record: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "execution_status": "running",
        "runner": "experiments/run_standard_experiment.py",
        "cache_root": contract.cache_root,
        "cache_digest": contract.cache_digest,
        "checksums_verified": bool(arguments.verify_checksums),
        "component_validation": component_validation,
        "dataset_support_audit": support_audit,
        "evidence_eligibility": initial_eligibility,
        "standards_evidence_label": STANDARDS_EVIDENCE_LABEL,
        "v2x_system_level_compliance_claimed": False,
        "split_roles": contract.manifest.get(
            "split_roles",
            {
                "train": "model fitting",
                "validation": (
                    "checkpoint selection and tuning; not an unbiased test"
                ),
                "heldout_channel": (
                    "source-disjoint evaluation with TDL-B/TDL-E profiles "
                    "held out from TDL-A/TDL-C/TDL-D training and validation"
                ),
            },
        ),
        "comparison_protocol": {
            "cache_realizations_shared_across_models": True,
            "optimizer_and_schedule_shared_across_models": True,
            "checkpoint_rule_shared_across_models": True,
            "model_initialization_seed_shared_by_reported_seed": True,
            "parameter_matched": False,
            "complexity_reported_per_model": True,
            "architecture_specific_hyperparameter_sweep_performed": False,
            "reference_model": reference_model,
            "reference_selection": reference_selection,
            "reference_strength_claimed": False,
            "reference_interpretation": (
                "paired-comparison anchor only; not asserted to be strongest"
            ),
            "primary_reference_predeclared": (
                explicit_reference_model == FORMAL_PRIMARY_REFERENCE_MODEL
            ),
            "method_model": FORMAL_METHOD_MODEL,
            "required_nonoracle_baselines": list(
                FORMAL_REQUIRED_NONORACLE_BASELINES
            ),
            "clean_retention_profile_strata": {
                name: list(indices)
                for name, indices in CLEAN_RETENTION_PROFILE_STRATA.items()
            },
            "scientific_release_thresholds": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in SCIENTIFIC_RELEASE_THRESHOLDS.items()
            },
            "holm_candidate_family": holm_candidates,
            "holm_family_declaration": (
                "explicit_cli_before_training"
                if holm_candidates
                else "none_no_multiplicity_claim"
            ),
            "holm_scope": (
                "separate family per held-out regime and algorithm seed"
            ),
            "validation_in_holm_family": False,
            "multi_seed_inference": (
                "hierarchical paired bootstrap over algorithm seeds and "
                "class-stratified held-out source clusters; no pooled McNemar"
            ),
            "bootstrap_draws": int(arguments.bootstrap_draws),
            "bootstrap_seed_base": int(arguments.bootstrap_seed),
        },
        "paper_ablation_protocols": PAPER_ABLATION_PROTOCOLS,
        "splits": contract.split_sizes,
        "modulations": contract.modulations,
        "num_classes": len(contract.modulations),
        "num_jammers": contract.num_jammers,
        "jammer_taxonomy": contract.jammer_names,
        "jammer_auxiliary_training_support": jammer_training_support,
        "protocol_exclusions": contract.manifest.get(
            "protocol_exclusions"
        ),
        "sample_length": contract.sample_length,
        "model_configuration": asdict(model_config),
        "training_configuration": asdict(training_config),
        "models": models,
        "seeds": seeds,
        "environment": environment_record(
            device,
            contract=contract,
            checksums_verified=arguments.verify_checksums,
            source_tree=source_tree_start,
        ),
        "results": [],
        "submission_release": {
            "schema_version": 1,
            "macro_generation_permitted": False,
            "submission_unlocked": False,
            "reasons": ["execution_not_complete"],
        },
    }
    write_json(run_directory / "run.json", run_record)

    flat_rows: list[dict[str, Any]] = []
    prediction_bundles: dict[tuple[str, int, str], Any] = {}
    evaluation_splits = {
        split: datasets[split] for split in evaluation_split_names(contract)
    }
    try:
        for model_name in models:
            for seed in seeds:
                seed_everything(seed)
                built = factories[model_name](
                    len(contract.modulations),
                    contract.num_jammers,
                    model_config,
                )
                model = built.model
                teacher = built.teacher
                started = time.perf_counter()
                training_result = train_model(
                    model=model,
                    teacher=teacher,
                    train_dataset=datasets["train"],
                    validation_dataset=datasets["validation"],
                    device=device,
                    seed=seed,
                    config=training_config,
                    objective=built.objective,
                    loss_weights=built.loss_weights,
                    jammer_support_mask=torch.tensor(
                        jammer_training_support["support_mask"],
                        dtype=torch.float32,
                    ),
                )
                training_seconds = time.perf_counter() - started

                model_directory = (
                    run_directory / "models" / f"{model_name}_seed{seed}"
                )
                model_directory.mkdir(parents=True, exist_ok=False)
                checkpoint_path = model_directory / "model.pt"
                torch.save(model.state_dict(), checkpoint_path)
                teacher_path: Path | None = None
                if teacher is not None:
                    teacher_path = model_directory / "teacher.pt"
                    torch.save(teacher.state_dict(), teacher_path)

                result: dict[str, Any] = {
                    "model": model_name,
                    "model_class": model.__class__.__name__,
                    "model_provenance": model_provenance(model),
                    "ablation_protocol": paper_ablation_protocol(model_name),
                    "objective": asdict(built.objective),
                    "loss_weights": asdict(built.loss_weights),
                    "jammer_auxiliary_training_support": (
                        jammer_training_support
                        if built.objective.use_jammer_auxiliary
                        else {
                            "status": "not_applicable",
                            "reason": (
                                "objective has no jammer auxiliary loss"
                            ),
                        }
                    ),
                    "seed": seed,
                    "training_seconds": training_seconds,
                    "training": training_result,
                    "checkpoint": str(checkpoint_path.relative_to(run_directory)),
                    "teacher_checkpoint": (
                        str(teacher_path.relative_to(run_directory))
                        if teacher_path is not None
                        else None
                    ),
                    "complexity": complexity_metrics(
                        model,
                        sample_length=contract.sample_length,
                        device=device,
                        latency_runs=arguments.latency_runs,
                    ),
                    "regimes": {},
                }
                for split, dataset in evaluation_splits.items():
                    bundle, metrics = predict(
                        model,
                        dataset,
                        device=device,
                        batch_size=training_config.batch_size,
                    )
                    result["regimes"][split] = metrics
                    prediction_bundles[(model_name, seed, split)] = bundle
                    prediction_path = (
                        model_directory / f"predictions_{split}.npz"
                    )
                    if bundle.target_profile_index is None:
                        raise RuntimeError(
                            "standard-cache predictions require "
                            "target_profile_index metadata"
                        )
                    np.savez_compressed(
                        prediction_path,
                        probabilities=bundle.probabilities,
                        labels=bundle.labels,
                        source_ids=bundle.source_ids,
                        snr_db=bundle.snr_db,
                        sir_db=bundle.sir_db,
                        target_profile_index=bundle.target_profile_index,
                        cache_digest=np.asarray(contract.cache_digest),
                        split=np.asarray(split),
                    )
                    flat_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "regime": split,
                            "cache_digest": contract.cache_digest,
                            "standards_evidence_label": (
                                STANDARDS_EVIDENCE_LABEL
                            ),
                            **{
                                key: value
                                for key, value in metrics.items()
                                if isinstance(value, (int, float))
                            },
                        }
                    )
                probe_x = (
                    datasets["validation"][0]["view1"]["x"]
                    .unsqueeze(0)
                    .to(device)
                )
                model.eval()
                with torch.no_grad():
                    probe_output = model(probe_x)
                available_auxiliary_heads = [
                    name
                    for name, output_key in (
                        ("jammer_multilabel", "jam_logits"),
                        ("quality", "quality"),
                    )
                    if output_key in probe_output
                ]
                if available_auxiliary_heads:
                    result["auxiliary_metrics"] = {
                        "status": "available",
                        "used_for_checkpoint_or_model_selection": False,
                        "available_heads": available_auxiliary_heads,
                        "regimes": {
                            split: auxiliary_task_metrics(
                                model,
                                dataset,
                                device=device,
                                batch_size=training_config.batch_size,
                                seed=seed,
                                split=split,
                                jammer_names=contract.jammer_names,
                                jammer_training_support_mask=(
                                    jammer_training_support["support_mask"]
                                ),
                                jammer_training_support_source=(
                                    jammer_training_support[
                                        "support_mask_source"
                                    ]
                                ),
                                quality_denormalization=(
                                    contract.manifest.get(
                                        "quality_normalization"
                                    )
                                ),
                                dataset_manifest=contract.manifest,
                            )
                            for split, dataset in evaluation_splits.items()
                        },
                    }
                else:
                    result["auxiliary_metrics"] = {
                        "status": "unavailable",
                        "reason": (
                            "model output has neither jam_logits nor quality "
                            "auxiliary head"
                        ),
                        "used_for_checkpoint_or_model_selection": False,
                        "available_heads": [],
                        "regimes": {},
                    }
                if (
                    built.objective.loss_family == "vimd"
                    and teacher is not None
                    and bool(getattr(model, "supports_tri_mechanism", False))
                ):
                    result["mechanism"] = mechanism_metrics(
                        model,
                        teacher,
                        datasets["heldout_channel"],
                        device=device,
                        batch_size=training_config.batch_size,
                        maximum_samples=len(datasets["heldout_channel"]),
                    )
                write_json(model_directory / "result.json", result)
                run_record["results"].append(result)
                write_json(run_directory / "run.json", run_record)

        metrics_frame = pd.DataFrame(flat_rows)
        metrics_frame.to_csv(run_directory / "metrics.csv", index=False)
        if len(seeds) > 1 and not metrics_frame.empty:
            _aggregate_seed_metrics(metrics_frame).to_csv(
                run_directory / "seed_aggregates.csv",
                index=False,
            )

        paired_rows: list[dict[str, Any]] = []
        if reference_model is not None:
            for candidate_name in models:
                if candidate_name == reference_model:
                    continue
                for seed in seeds:
                    for split in evaluation_splits:
                        reference = prediction_bundles[
                            (reference_model, seed, split)
                        ]
                        candidate = prediction_bundles[
                            (candidate_name, seed, split)
                        ]
                        bootstrap_seed = analysis_seed(
                            arguments.bootstrap_seed,
                            "single_seed",
                            reference_model,
                            candidate_name,
                            seed,
                            split,
                        )
                        preregistered_mcnemar = (
                            split != "validation"
                            and candidate_name in holm_candidates
                        )
                        if preregistered_mcnemar:
                            statistics = paired_bundle_statistics(
                                reference,
                                candidate,
                                draws=arguments.bootstrap_draws,
                                seed=bootstrap_seed,
                            )
                        else:
                            statistics = paired_accuracy_macro_f1_bootstrap(
                                reference,
                                candidate,
                                draws=arguments.bootstrap_draws,
                                seed=bootstrap_seed,
                            )
                            statistics.update(
                                {
                                    "reference_only_correct": None,
                                    "candidate_only_correct": None,
                                    "discordant_pairs": None,
                                    "exact_p_value": None,
                                    "inference_scope": "descriptive_paired_ci",
                                    "mcnemar_scope": "not_performed",
                                }
                            )
                        paired_rows.append(
                            {
                                "reference": reference_model,
                                "reference_selection": reference_selection,
                                "reference_strength_claimed": False,
                                "candidate": candidate_name,
                                "seed": seed,
                                "regime": split,
                                "comparison_role": (
                                    "validation_descriptive"
                                    if split == "validation"
                                    else "heldout_single_seed"
                                ),
                                "mcnemar_preregistered": preregistered_mcnemar,
                                "cache_digest": contract.cache_digest,
                                **statistics,
                                "holm_included": False,
                                "holm_family_id": None,
                                "holm_family_size": None,
                                "holm_adjusted_p_value": None,
                                "holm_exclusion_reason": (
                                    "validation_is_excluded_from_all_holm_families"
                                    if split == "validation"
                                    else (
                                        "candidate_not_in_predeclared_holm_family"
                                        if candidate_name not in holm_candidates
                                        else None
                                    )
                                ),
                            }
                        )
        holm_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in paired_rows:
            if (
                row["regime"] != "validation"
                and row["candidate"] in holm_candidates
            ):
                holm_groups.setdefault(
                    (str(row["regime"]), int(row["seed"])),
                    [],
                ).append(row)
        for (regime, algorithm_seed), family_rows in holm_groups.items():
            family_order = [
                candidate
                for candidate in holm_candidates
                if any(row["candidate"] == candidate for row in family_rows)
            ]
            ordered_rows = [
                next(
                    row
                    for row in family_rows
                    if row["candidate"] == candidate
                )
                for candidate in family_order
            ]
            adjusted = holm_adjust(
                np.asarray(
                    [row["exact_p_value"] for row in ordered_rows],
                    dtype=np.float64,
                )
            )
            family_token = ",".join(family_order)
            family_id = (
                f"regime={regime}|seed={algorithm_seed}|"
                f"reference={reference_model}|candidates={family_token}"
            )
            for row, adjusted_value in zip(ordered_rows, adjusted):
                row["holm_included"] = True
                row["holm_family_id"] = family_id
                row["holm_family_size"] = len(ordered_rows)
                row["holm_adjusted_p_value"] = float(adjusted_value)
                row["holm_exclusion_reason"] = None
        paired_columns = [
            "reference",
            "reference_selection",
            "reference_strength_claimed",
            "candidate",
            "seed",
            "regime",
            "comparison_role",
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
            "bootstrap_draws",
            "bootstrap_seed",
            "bootstrap_cluster_count",
            "bootstrap_stratified_by_class",
            "bootstrap_unit",
            "reference_only_correct",
            "candidate_only_correct",
            "discordant_pairs",
            "exact_p_value",
            "inference_scope",
            "mcnemar_scope",
            "mcnemar_preregistered",
            "holm_included",
            "holm_family_id",
            "holm_family_size",
            "holm_adjusted_p_value",
            "holm_exclusion_reason",
        ]
        pd.DataFrame(paired_rows, columns=paired_columns).to_csv(
            run_directory / "paired_statistics.csv",
            index=False,
        )

        headline_rows: list[dict[str, Any]] = []
        if reference_model is not None and len(seeds) > 1:
            for candidate_name in models:
                if candidate_name == reference_model:
                    continue
                for split in evaluation_splits:
                    if split == "validation":
                        continue
                    references_by_seed = {
                        seed: prediction_bundles[
                            (reference_model, seed, split)
                        ]
                        for seed in seeds
                    }
                    candidates_by_seed = {
                        seed: prediction_bundles[
                            (candidate_name, seed, split)
                        ]
                        for seed in seeds
                    }
                    bootstrap_seed = analysis_seed(
                        arguments.bootstrap_seed,
                        "hierarchical",
                        reference_model,
                        candidate_name,
                        split,
                    )
                    headline_rows.append(
                        {
                            "reference": reference_model,
                            "reference_selection": reference_selection,
                            "reference_strength_claimed": False,
                            "candidate": candidate_name,
                            "regime": split,
                            "parent_regime": split,
                            "target_profile_indices": "all",
                            "cache_digest": contract.cache_digest,
                            **headline_paired_bootstrap(
                                references_by_seed,
                                candidates_by_seed,
                                draws=arguments.bootstrap_draws,
                                seed=bootstrap_seed,
                            ),
                        }
                    )
            if (
                FORMAL_METHOD_MODEL in models
                and reference_model != FORMAL_METHOD_MODEL
                and "clean_retention" in evaluation_splits
            ):
                for stratum_name, profile_indices in (
                    CLEAN_RETENTION_PROFILE_STRATA.items()
                ):
                    references_by_seed = {}
                    candidates_by_seed = {}
                    for seed in seeds:
                        reference_bundle = prediction_bundles[
                            (reference_model, seed, "clean_retention")
                        ]
                        candidate_bundle = prediction_bundles[
                            (FORMAL_METHOD_MODEL, seed, "clean_retention")
                        ]
                        if reference_bundle.target_profile_index is None:
                            raise RuntimeError(
                                "clean-retention profile statistics require "
                                "target_profile_index"
                            )
                        selected = np.isin(
                            reference_bundle.target_profile_index,
                            np.asarray(profile_indices, dtype=np.int64),
                        )
                        references_by_seed[seed] = reference_bundle.subset(
                            selected
                        )
                        candidates_by_seed[seed] = candidate_bundle.subset(
                            selected
                        )
                    bootstrap_seed = analysis_seed(
                        arguments.bootstrap_seed,
                        "hierarchical",
                        reference_model,
                        FORMAL_METHOD_MODEL,
                        stratum_name,
                    )
                    headline_rows.append(
                        {
                            "reference": reference_model,
                            "reference_selection": reference_selection,
                            "reference_strength_claimed": False,
                            "candidate": FORMAL_METHOD_MODEL,
                            "regime": stratum_name,
                            "parent_regime": "clean_retention",
                            "target_profile_indices": repr(
                                list(profile_indices)
                            ),
                            "cache_digest": contract.cache_digest,
                            **headline_paired_bootstrap(
                                references_by_seed,
                                candidates_by_seed,
                                draws=arguments.bootstrap_draws,
                                seed=bootstrap_seed,
                            ),
                        }
                    )
        headline_columns = [
            "reference",
            "reference_selection",
            "reference_strength_claimed",
            "candidate",
            "regime",
            "parent_regime",
            "target_profile_indices",
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
        pd.DataFrame(headline_rows, columns=headline_columns).to_csv(
            run_directory / "headline_paired_statistics.csv",
            index=False,
        )
        run_record["statistical_outputs"] = {
            "single_seed_pairs": "paired_statistics.csv",
            "multi_seed_headline_pairs": "headline_paired_statistics.csv",
            "clean_retention_profile_strata": {
                name: {
                    "parent_regime": "clean_retention",
                    "target_profile_indices": list(indices),
                    "reference": reference_model,
                    "candidate": FORMAL_METHOD_MODEL,
                }
                for name, indices in CLEAN_RETENTION_PROFILE_STRATA.items()
            },
            "holm_families": [
                {
                    "family_id": family_rows[0]["holm_family_id"],
                    "regime": regime,
                    "algorithm_seed": algorithm_seed,
                    "reference": reference_model,
                    "candidates": [
                        row["candidate"] for row in family_rows
                    ],
                    "validation_included": False,
                }
                for (regime, algorithm_seed), family_rows in holm_groups.items()
            ],
            "pooled_multi_seed_mcnemar_performed": False,
        }
        source_audit = source_tree_execution_audit(source_tree_start)
        run_record["source_tree_execution_audit"] = source_audit
        run_record["status"] = "complete"
        run_record["execution_status"] = "complete"
        run_record["evidence_eligibility"] = assess_evidence_eligibility(
            contract=contract,
            support_audit=support_audit,
            checksums_verified=bool(arguments.verify_checksums),
            component_validation=component_validation,
            models=models,
            seeds=seeds,
            execution_status="complete",
            explicit_reference_model=explicit_reference_model,
            holm_candidates=holm_candidates,
            source_tree_unchanged=bool(source_audit["unchanged"]),
            training_configuration=training_config,
            training_results=run_record["results"],
            jammer_training_support=jammer_training_support,
        )
        run_record["submission_release"] = (
            submission_release_source_gate(run_record)
        )
        run_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_directory / "run.json", run_record)
    except Exception as error:
        source_audit = source_tree_execution_audit(source_tree_start)
        run_record["source_tree_execution_audit"] = source_audit
        run_record["status"] = "failed"
        run_record["execution_status"] = "failed"
        run_record["evidence_eligibility"] = assess_evidence_eligibility(
            contract=contract,
            support_audit=support_audit,
            checksums_verified=bool(arguments.verify_checksums),
            component_validation=component_validation,
            models=models,
            seeds=seeds,
            execution_status="failed",
            explicit_reference_model=explicit_reference_model,
            holm_candidates=holm_candidates,
            source_tree_unchanged=bool(source_audit["unchanged"]),
            training_configuration=training_config,
            training_results=run_record["results"],
            jammer_training_support=jammer_training_support,
        )
        run_record["submission_release"] = (
            submission_release_source_gate(run_record)
        )
        run_record["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_directory / "run.json", run_record)
        raise
    finally:
        for dataset in datasets.values():
            dataset.close()

    print(run_directory)


if __name__ == "__main__":
    main()
