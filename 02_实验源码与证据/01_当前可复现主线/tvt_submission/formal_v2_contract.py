"""Strict loader for the prospective TVT formal-freeze v2 contract."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = (
    ROOT / "tvt_submission" / "configs" / "formal_tvt_freeze_v2.json"
)
V3_8GB_FREEZE = (
    ROOT / "tvt_submission" / "configs" / "formal_tvt_freeze_v3_8gb.json"
)
V4_8GB_DUAL_FREEZE = (
    ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_freeze_v4_8gb_dual.json"
)
EXPECTED_SCHEMA = "vimd_amc.tvt.formal_freeze.v2"
EXPECTED_DEFAULT_FREEZE_SHA256 = (
    "926e8eebdfadaab7876686f4762ec9bf2ffbb564a3e013b46932598f0a25b423"
)
EXPECTED_V3_8GB_FREEZE_SHA256 = (
    "8ff752ead790663f38e25e7c72a00d44d3dde2291a14a8ac2b580a4fd38dafc6"
)
EXPECTED_V4_8GB_DUAL_FREEZE_SHA256 = (
    "3b7076e3f6cf09f7d1b5e1d45d33989f8eb34510345349bb1683f6c6f52ec1aa"
)
V3_8GB_SCHEMA = "vimd_amc.tvt.formal_freeze.v3_8gb"
V4_8GB_DUAL_SCHEMA = "vimd_amc.tvt.formal_freeze.v4_8gb_dual"
EXPECTED_V3_8GB_CUDA_EXECUTION = {
    "minimum_free_gpu_mib": 6000,
    "per_device_batch_size": 32,
    "use_amp": True,
    "allocator_configuration": (
        "expandable_segments:True,max_split_size_mb:128,"
        "garbage_collection_threshold:0.8"
    ),
    "cuda_fit_concurrency": 1,
    "oom_policy": "fail_closed_no_adaptive_batch_resize_or_model_change",
}
EXPECTED_V4_8GB_DUAL_CUDA_EXECUTION = {
    "minimum_free_gpu_mib": 6000,
    "per_device_batch_size": 16,
    "use_amp": True,
    "allocator_configuration": (
        "expandable_segments:True,max_split_size_mb:128,"
        "garbage_collection_threshold:0.8"
    ),
    "cuda_fit_concurrency": 2,
    "oom_policy": "fail_closed_no_adaptive_batch_resize_or_model_change",
}
EXPECTED_JUSTIFICATION_PATH = (
    "docs/TVT_V2_PROSPECTIVE_STATISTICAL_JUSTIFICATION.md"
)
EXPECTED_JUSTIFICATION_SHA256 = (
    "055b330c20ef7a908ef191b75bb7a39302c1c51be29a1d3b517639002cae26a7"
)
EXPECTED_CACHE_OUTPUT = "standards/cache_factor_headline_1024_v2"
EXPECTED_CACHE_PRESET = "headline_v2"
EXPECTED_CACHE_DESIGNATION = "headline_formal_tvt_evidence_v2"
EXPECTED_CACHE_MASTER_SEED = 20260727
EXPECTED_SPLIT_SOURCE_COUNTS = (
    ("train", 100_000),
    ("validation", 2_000),
    ("id_test", 5_000),
    ("hard_interference", 5_000),
    ("unseen_jammer", 5_000),
    ("unseen_speed", 5_000),
    ("heldout_channel", 5_000),
    ("combined_ood", 5_000),
    ("clean_retention", 5_000),
    ("adc_10bit_agc", 5_000),
    ("adc_12bit_agc", 5_000),
    ("per_emitter_sync", 5_000),
)
EXPECTED_LEARNING_SOURCE_COUNTS = (10_000, 30_000, 100_000)
EXPECTED_LEARNING_CACHE_PRESETS = (
    "learning_10k_v2",
    "learning_30k_v2",
    "headline_v2",
)
EXPECTED_LEARNING_CACHE_OUTPUTS = (
    "standards/cache_factor_learning_10k_1024_v2",
    "standards/cache_factor_learning_30k_1024_v2",
    EXPECTED_CACHE_OUTPUT,
)
EXPECTED_LEARNING_MODELS = ("a0_backbone", "a5_vimd_full")
EXPECTED_LEARNING_SEEDS = (17, 29, 43, 71, 101)
EXPECTED_EPOCH_METRICS = (
    "training_macro_f1",
    "validation_macro_f1",
    "training_total_loss",
    "validation_loss",
)
EXPECTED_SCALE_METRICS = (
    "selected_validation_macro_f1",
    "id_test_macro_f1",
    "hard_interference_macro_f1",
)
EXPECTED_MODELS = (
    "a0_backbone",
    "a1_single_mask",
    "a2_tri_no_teacher",
    "a3p_tri_proportional_teacher",
    "a3_tri_teacher",
    "a4_tri_teacher_mtl",
    "a5_vimd_full",
    "a6_dual_full",
    "a7_vimd_no_residual",
    "mcldnn_reimplementation",
    "iqformer_inspired",
    "cssl_amc_supervised_adaptation",
)
EXPECTED_SEEDS = (17, 29, 43, 71, 101, 131, 173, 211, 257, 307)
EXPECTED_RUNNER = "experiments/run_formal_tvt_v2.py"
EXPECTED_RUN_ID = "tvt_headline_1024_10seed_v2"
EXPECTED_RUN_OUTPUT = "artifacts"
EXPECTED_RUN_DIRECTORY = f"{EXPECTED_RUN_OUTPUT}/{EXPECTED_RUN_ID}"
EXPECTED_REFERENCE_MODEL = "cssl_amc_supervised_adaptation"
EXPECTED_HOLM_CANDIDATES = (
    "a0_backbone",
    "a1_single_mask",
    "a5_vimd_full",
    "mcldnn_reimplementation",
    "iqformer_inspired",
)
EXPECTED_TRAINING_CONFIGURATION = (
    ("epochs", 30),
    ("batch_size", 64),
    ("learning_rate", 0.0003),
    ("weight_decay", 0.01),
    ("mask_start_epoch", 2),
    ("contrastive_start_epoch", 5),
    ("mask_ramp_epochs", 3),
    ("contrastive_ramp_epochs", 3),
    ("minimum_full_stage_epochs", 3),
    ("patience", 8),
    ("use_amp", True),
)
EXPECTED_MODEL_CONFIGURATION = (
    ("n_fft", 64),
    ("hop_length", 16),
    ("spectral_channels", 24),
    ("embedding_dim", 48),
    ("environment_dim", 32),
    ("dropout", 0.05),
)
EXPECTED_PROSPECTIVE_STATISTICAL_DESIGN = {
    "frozen_date": "2026-07-29",
    "status": "frozen_before_any_formal_v2_artifact",
    "eligible_formal_v2_artifacts_observed_before_freeze": False,
    "justification_path": EXPECTED_JUSTIFICATION_PATH,
    "justification_sha256": EXPECTED_JUSTIFICATION_SHA256,
    "design_basis": (
        "precision_and_decision_rule_justification_without_model_based_"
        "power_percentage"
    ),
    "power_claim": (
        "none_no_eligible_pilot_variance_or_paired_dependence_estimates"
    ),
    "estimand": {
        "regime": "hard_interference",
        "metric": "absolute_macro_f1",
        "direction": "candidate_minus_reference",
        "resampling_unit": (
            "algorithm_seed_then_class_stratified_source_id_cluster_"
            "preserving_paired_predictions_and_views"
        ),
        "validation_excluded": True,
    },
    "smallest_effect_size_of_interest": {
        "contrast_id": "full_vs_backbone",
        "absolute_macro_f1_gain": 0.01,
        "purpose": (
            "minimum_point_gain_for_practically_material_primary_gain_wording"
        ),
        "statistical_success_still_requires_simultaneous_ci95_low_gt": 0.0,
        "below_threshold_wording": (
            "statistically_resolved_but_below_preregistered_materiality_"
            "threshold"
        ),
    },
    "fixed_design": {
        "confirmatory_model_count": 12,
        "confirmatory_algorithm_seed_count": 10,
        "required_model_seed_fit_count": 120,
        "hard_interference_source_cluster_count": 5_000,
        "confirmatory_family_size": 3,
        "bootstrap_draws": 10_000,
        "learning_curve_fit_count": 30,
        "posthoc_seed_or_source_extension_permitted": False,
    },
    "precision_reporting": {
        "required_per_contrast": [
            "point_estimate",
            "marginal_ci95",
            "simultaneous_ci95",
            "simultaneous_interval_width",
        ],
        "primary_required": [
            "simultaneous_interval_width_divided_by_sesoi",
            "frozen_precision_classification",
        ],
        "wide_or_inconclusive_interval_requires_design_extension": False,
        "wide_or_inconclusive_interval_must_be_reported": True,
    },
    "stopping_rule": {
        "fit_level": (
            "maximum_30_epochs_with_frozen_validation_only_early_stopping"
        ),
        "study_level": (
            "complete_exact_120_fit_grid_without_interim_test_looks_or_"
            "efficacy_futility_stopping"
        ),
        "test_metrics_used_for_checkpoint_or_stopping": False,
        "technical_retry": (
            "same_model_seed_cache_configuration_and_clean_output_with_"
            "failure_audit"
        ),
        "replacement_seed_permitted": False,
        "outcome_dependent_amendment_permitted": False,
    },
    "amendment_rule": (
        "if_infeasible_before_formal_result_inspection_issue_a_new_versioned_"
        "prospective_freeze_and_do_not_relabel_it_as_v2"
    ),
}
EXPECTED_PROMOTION_REQUIREMENTS = (
    ("prospective_statistical_design_verified", True),
    ("learning_curve_complete_before_confirmatory", True),
    ("run_eligible_true", True),
    ("source_tree_unchanged", True),
    ("all_required_models_and_seeds", True),
    ("at_least_one_checkpoint_selection_eligible_epoch_per_fit", True),
    ("no_fallback_checkpoint", True),
    ("confirmatory_family_artifact_derived", True),
    ("ood_axis_calibration_artifact_derived", True),
    ("receiver_robustness_artifact_derived", True),
    ("no_placeholder_result_macro", True),
    ("human_primary_source_audit", True),
)
EXPECTED_CONFIRMATORY = (
    (
        "full_vs_backbone",
        "a0_backbone",
        "a5_vimd_full",
    ),
    (
        "margin_vs_proportional_teacher",
        "a3p_tri_proportional_teacher",
        "a3_tri_teacher",
    ),
    (
        "tri_vs_dual_route",
        "a6_dual_full",
        "a5_vimd_full",
    ),
)
EXPECTED_OOD_AXES = (
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
)
EXPECTED_STRESS_AXES = (
    "adc_10bit_agc",
    "adc_12bit_agc",
    "per_emitter_sync",
)
EXPECTED_OCCUPANCY_DEFINITION = (
    "per-source fraction of periodic-Hann complex-STFT cells in the single "
    "inference view (view1) whose jammer power is at least 0.01 times that "
    "source view's maximum jammer-cell power (-20 dB support); family value "
    "is the arithmetic mean over sources"
)
EXPECTED_GAIN_DEFINITION = (
    "for each jammer family, compute paired A5-minus-A0 macro-F1 on that "
    "family's inference-view source subset within each algorithm seed, then "
    "take the arithmetic mean across algorithm seeds"
)


class FormalV2ContractError(ValueError):
    """Raised when the prospective freeze is incomplete or internally unsafe."""


class LoadedFormalV2Contract(dict[str, Any]):
    """Validated contract carrying the identity of the bytes that defined it."""

    freeze_path: Path
    freeze_sha256: str
    justification_path: Path
    justification_sha256: str

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        freeze_path: Path,
        freeze_sha256: str,
        justification_path: Path,
        justification_sha256: str,
    ) -> None:
        super().__init__(payload)
        self.freeze_path = freeze_path
        self.freeze_sha256 = freeze_sha256
        self.justification_path = justification_path
        self.justification_sha256 = justification_sha256


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalV2ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalV2ContractError(f"{label} must be an object")
    return value


def _positive_unique_ints(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in value
        )
    ):
        raise FormalV2ContractError(
            f"{label} must be a nonempty list of nonnegative integers"
        )
    values = tuple(int(item) for item in value)
    if len(values) != len(set(values)):
        raise FormalV2ContractError(f"{label} contains duplicates")
    return values


def _exact_sequence(
    value: Any,
    expected: tuple[Any, ...],
    label: str,
) -> tuple[Any, ...]:
    if (
        not isinstance(value, (list, tuple))
        or tuple(value) != expected
    ):
        raise FormalV2ContractError(
            f"{label} drifted from the immutable formal v2 freeze"
        )
    return tuple(value)


def _exact_mapping_items(
    value: Any,
    expected: tuple[tuple[str, Any], ...],
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    if tuple(mapping.items()) != expected:
        raise FormalV2ContractError(
            f"{label} drifted from the immutable formal v2 freeze"
        )
    return mapping


def freeze_sha256(path: Path) -> str:
    """Return the SHA-256 identity of a freeze file's exact bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise FormalV2ContractError(
            f"could not read formal v2 freeze: {error}"
        ) from error
    return digest.hexdigest()


def loaded_freeze_identity(
    contract: Mapping[str, Any],
) -> tuple[Path, str]:
    """Return the byte identity carried by :func:`load_contract`."""

    path = getattr(contract, "freeze_path", None)
    digest = getattr(contract, "freeze_sha256", None)
    if (
        not isinstance(path, Path)
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise FormalV2ContractError(
            "formal v2 contract is not bound to loaded freeze bytes"
        )
    return path, digest


def validate_justification_document(
    contract: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[Path, str]:
    """Validate the frozen prospective-justification document bytes."""

    design = _mapping(
        contract.get("prospective_statistical_design"),
        "prospective_statistical_design",
    )
    relative_path = design.get("justification_path")
    expected_digest = design.get("justification_sha256")
    if (
        relative_path != EXPECTED_JUSTIFICATION_PATH
        or expected_digest != EXPECTED_JUSTIFICATION_SHA256
    ):
        raise FormalV2ContractError(
            "prospective statistical justification identity drifted"
        )
    path = (root / EXPECTED_JUSTIFICATION_PATH).resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FormalV2ContractError(
            "prospective statistical justification is missing or unreadable: "
            f"{error}"
        ) from error
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        raise FormalV2ContractError(
            "prospective statistical justification SHA-256 drifted: "
            f"{observed_digest} != {expected_digest}"
        )
    return path, observed_digest


def loaded_justification_identity(
    contract: Mapping[str, Any],
) -> tuple[Path, str]:
    """Return the verified justification identity carried by the loader."""

    path = getattr(contract, "justification_path", None)
    digest = getattr(contract, "justification_sha256", None)
    if (
        not isinstance(path, Path)
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise FormalV2ContractError(
            "formal v2 contract is not bound to verified justification bytes"
        )
    return path, digest


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != EXPECTED_SCHEMA:
        raise FormalV2ContractError("formal v2 schema mismatch")
    if contract.get("status") != "preregistered_not_executed":
        raise FormalV2ContractError(
            "formal v2 must remain explicitly not executed until artifacts exist"
        )
    if contract.get("historical_v1_results_reinterpreted") is not False:
        raise FormalV2ContractError(
            "historical v1 results cannot be reinterpreted as v2 evidence"
        )
    if not (
        contract.get("created_date") == "2026-07-28"
        and contract.get("supersedes_for_future_execution")
        == "tvt_submission/configs/formal_tvt_freeze_v1.json"
        and contract.get("journal_target")
        == "IEEE Transactions on Vehicular Technology"
        and contract.get("claim_scope")
        == (
            "simulation-only AMC under 3GPP TR 38.901 TDL-profile "
            "channels parameterized by vehicular Doppler"
        )
    ):
        raise FormalV2ContractError(
            "top-level formal v2 protocol identity drifted"
        )
    prospective = _mapping(
        contract.get("prospective_statistical_design"),
        "prospective_statistical_design",
    )
    if prospective != EXPECTED_PROSPECTIVE_STATISTICAL_DESIGN:
        raise FormalV2ContractError(
            "prospective statistical design, SESOI, precision, or stopping "
            "contract drifted"
        )
    cache = _mapping(contract.get("cache"), "cache")
    if not (
        cache.get("output") == EXPECTED_CACHE_OUTPUT
        and cache.get("preset") == EXPECTED_CACHE_PRESET
        and cache.get("sample_length") == 1024
        and cache.get("guard_samples") == 96
        and cache.get("master_seed") == EXPECTED_CACHE_MASTER_SEED
        and cache.get("matlab_timeout_s") == 3600
        and cache.get("expected_designation")
        == EXPECTED_CACHE_DESIGNATION
    ):
        raise FormalV2ContractError(
            "cache identity or construction constants drifted"
        )
    _exact_mapping_items(
        cache.get("expected_split_source_counts"),
        EXPECTED_SPLIT_SOURCE_COUNTS,
        "cache.expected_split_source_counts",
    )
    receiver = _mapping(
        cache.get("receiver_robustness_contract"),
        "cache.receiver_robustness_contract",
    )
    if not (
        receiver.get("nominal_training") is True
        and receiver.get("stress_splits_are_test_only") is True
        and _mapping(receiver.get("agc"), "cache.agc")
        == {
            "mode": "window_rms_before_adc",
            "applied_before_quantization": True,
        }
    ):
        raise FormalV2ContractError(
            "receiver robustness training/AGC constants drifted"
        )
    adc_10bit = _mapping(
        receiver.get("adc_10bit_agc"),
        "adc_10bit_agc",
    )
    adc_12bit = _mapping(
        receiver.get("adc_12bit_agc"),
        "adc_12bit_agc",
    )
    if not (
        adc_10bit
        == {
            "adc_bits": 10,
            "adc_full_scale": 3.0,
            "isolated_factor": "adc_precision",
        }
        and adc_12bit
        == {
            "adc_bits": 12,
            "adc_full_scale": 3.0,
            "isolated_factor": "adc_precision",
        }
    ):
        raise FormalV2ContractError(
            "receiver robustness must include exact 10-bit and 12-bit ADC axes"
        )
    sync = _mapping(receiver.get("per_emitter_sync"), "per_emitter_sync")
    if sync != {
        "target_and_jammer_offsets_independent": True,
        "application_stage": (
            "pre_mixing_after_independent_TDL_channelization"
        ),
        "per_emitter_cfo_norm_max": 0.006,
        "per_emitter_timing_offset_max_samples": 3,
        "component_identity_retained": True,
        "isolated_factor": "emitter_synchronization",
    }:
        raise FormalV2ContractError(
            "per-emitter CFO/timing contract is incomplete"
        )
    if set(receiver) != {
        "nominal_training",
        "stress_splits_are_test_only",
        "agc",
        "adc_10bit_agc",
        "adc_12bit_agc",
        "per_emitter_sync",
    }:
        raise FormalV2ContractError(
            "receiver robustness contract contains key drift"
        )

    learning = _mapping(contract.get("learning_curve"), "learning_curve")
    if learning.get("role") != (
        "preregistered_scale_adequacy_evidence_not_model_selection"
    ):
        raise FormalV2ContractError("learning-curve role drifted")
    _exact_sequence(
        learning.get("training_source_counts"),
        EXPECTED_LEARNING_SOURCE_COUNTS,
        "learning_curve.training_source_counts",
    )
    _exact_sequence(
        learning.get("cache_presets"),
        EXPECTED_LEARNING_CACHE_PRESETS,
        "learning_curve.cache_presets",
    )
    _exact_sequence(
        learning.get("cache_outputs"),
        EXPECTED_LEARNING_CACHE_OUTPUTS,
        "learning_curve.cache_outputs",
    )
    _exact_sequence(
        learning.get("models"),
        EXPECTED_LEARNING_MODELS,
        "learning_curve.models",
    )
    _exact_sequence(
        learning.get("algorithm_seeds"),
        EXPECTED_LEARNING_SEEDS,
        "learning_curve.algorithm_seeds",
    )
    _exact_sequence(
        learning.get("required_epoch_metrics"),
        EXPECTED_EPOCH_METRICS,
        "learning_curve.required_epoch_metrics",
    )
    _exact_sequence(
        learning.get("required_scale_metrics"),
        EXPECTED_SCALE_METRICS,
        "learning_curve.required_scale_metrics",
    )
    if (
        learning.get("must_complete_before_confirmatory_run") is not True
        or learning.get("posthoc_scale_or_hyperparameter_selection_permitted")
        is not False
        or int(learning.get("fixed_formal_scale_regardless_of_curve", 0))
        != 100_000
    ):
        raise FormalV2ContractError(
            "learning-curve ordering or no-selection rule is unsafe"
        )

    experiment = _mapping(contract.get("experiment"), "experiment")
    models = tuple(str(value) for value in experiment.get("models", ()))
    if len(models) != len(set(models)) or not models:
        raise FormalV2ContractError(
            "experiment.models must be nonempty and unique"
        )
    seeds = _positive_unique_ints(experiment.get("seeds"), "experiment.seeds")
    if len(seeds) < 10:
        raise FormalV2ContractError(
            "confirmatory execution requires at least 10 algorithm seeds"
        )
    if models != EXPECTED_MODELS:
        raise FormalV2ContractError(
            "experiment.models drifted from the immutable formal v2 freeze"
        )
    if seeds != EXPECTED_SEEDS:
        raise FormalV2ContractError(
            "experiment.seeds drifted from the immutable formal v2 freeze"
        )
    if not (
        experiment.get("runner") == EXPECTED_RUNNER
        and experiment.get("run_id") == EXPECTED_RUN_ID
        and experiment.get("output") == EXPECTED_RUN_OUTPUT
        and experiment.get("expected_run_directory")
        == EXPECTED_RUN_DIRECTORY
        and experiment.get("reference_model") == EXPECTED_REFERENCE_MODEL
        and experiment.get("reference_selection")
        == "predeclared before training; not described as strongest"
        and tuple(experiment.get("holm_candidates", ()))
        == EXPECTED_HOLM_CANDIDATES
        and experiment.get("device") == "cuda"
        and experiment.get("verify_checksums") is True
        and experiment.get("validate_components") is True
    ):
        raise FormalV2ContractError(
            "formal runner, run_id, output, reference, or execution flags drifted"
        )
    _exact_mapping_items(
        experiment.get("training"),
        EXPECTED_TRAINING_CONFIGURATION,
        "experiment.training",
    )
    _exact_mapping_items(
        experiment.get("model"),
        EXPECTED_MODEL_CONFIGURATION,
        "experiment.model",
    )
    family = _mapping(
        experiment.get("confirmatory_family"),
        "experiment.confirmatory_family",
    )
    if int(family.get("minimum_algorithm_seeds", 0)) < 10:
        raise FormalV2ContractError(
            "confirmatory family seed floor must be at least 10"
        )
    if not (
        family.get("family_id")
        == "hard_macro_f1_confirmatory_family_v2"
        and family.get("regime") == "hard_interference"
        and family.get("metric") == "macro_f1"
        and family.get("direction") == "candidate_minus_reference"
        and family.get("confidence_level") == 0.95
        and family.get("multiplicity_method")
        == (
            "joint_max_absolute_centered_deviation_hierarchical_paired_"
            "bootstrap"
        )
        and family.get("simultaneous_ci95_low_strictly_greater_than")
        == 0.0
        and family.get("minimum_algorithm_seeds") == 10
    ):
        raise FormalV2ContractError(
            "confirmatory-family inference constants drifted"
        )
    contrasts = family.get("contrasts")
    if not isinstance(contrasts, list):
        raise FormalV2ContractError(
            "confirmatory_family.contrasts must be a list"
        )
    observed = tuple(
        (
            str(record.get("contrast_id")),
            str(record.get("reference")),
            str(record.get("candidate")),
        )
        for record in contrasts
        if isinstance(record, Mapping)
    )
    if observed != EXPECTED_CONFIRMATORY:
        raise FormalV2ContractError(
            "confirmatory family must be exactly A5-A0, margin-vs-ratio, "
            "and A5-A6"
        )
    if contrasts != [
        {
            "contrast_id": "full_vs_backbone",
            "reference": "a0_backbone",
            "candidate": "a5_vimd_full",
            "intervention": "complete_method_vs_shared_backbone",
            "interpretation_scope": "composite_primary_method_effect",
        },
        {
            "contrast_id": "margin_vs_proportional_teacher",
            "reference": "a3p_tri_proportional_teacher",
            "candidate": "a3_tri_teacher",
            "intervention": "margin_teacher_vs_naive_power_ratio_teacher",
            "interpretation_scope": "teacher_function_form",
        },
        {
            "contrast_id": "tri_vs_dual_route",
            "reference": "a6_dual_full",
            "candidate": "a5_vimd_full",
            "intervention": "tri_route_vs_dual_route_full_objective",
            "interpretation_scope": (
                "composite_control_no_single_block_attribution"
            ),
        },
    ]:
        raise FormalV2ContractError(
            "confirmatory contrast semantics drifted"
        )
    required_family_models = {
        model for _, reference, candidate in observed for model in (reference, candidate)
    }
    if not required_family_models.issubset(models):
        raise FormalV2ContractError(
            "confirmatory models are missing from experiment.models"
        )
    exploratory = experiment.get("exploratory_contrasts")
    if not isinstance(exploratory, list) or not exploratory:
        raise FormalV2ContractError(
            "non-confirmatory contrasts must be declared exploratory"
        )
    if exploratory != [
        {
            "contrast_id": "teacher_presence",
            "reference": "a2_tri_no_teacher",
            "candidate": "a3_tri_teacher",
            "role": "exploratory_no_familywise_positive_claim",
        },
        {
            "contrast_id": "multitask_bundle",
            "reference": "a3_tri_teacher",
            "candidate": "a4_tri_teacher_mtl",
            "role": "exploratory_no_familywise_positive_claim",
        },
        {
            "contrast_id": "cross_condition_contrast",
            "reference": "a4_tri_teacher_mtl",
            "candidate": "a5_vimd_full",
            "role": "exploratory_no_familywise_positive_claim",
        },
        {
            "contrast_id": "full_vs_single_mask",
            "reference": "a1_single_mask",
            "candidate": "a5_vimd_full",
            "role": "exploratory_no_familywise_positive_claim",
        },
        {
            "contrast_id": "bounded_bypass",
            "reference": "a7_vimd_no_residual",
            "candidate": "a5_vimd_full",
            "role": "exploratory_no_familywise_positive_claim",
        },
    ]:
        raise FormalV2ContractError(
            "exploratory contrast semantics drifted"
        )
    confirmatory_ids = {record[0] for record in observed}
    exploratory_ids: set[str] = set()
    for record in exploratory:
        if not isinstance(record, Mapping):
            raise FormalV2ContractError(
                "exploratory contrast records must be objects"
            )
        contrast_id = str(record.get("contrast_id", ""))
        if (
            not contrast_id
            or contrast_id in confirmatory_ids
            or record.get("role")
            != "exploratory_no_familywise_positive_claim"
        ):
            raise FormalV2ContractError(
                "exploratory contrasts must be disjoint and explicitly non-confirmatory"
            )
        exploratory_ids.add(contrast_id)
    if len(exploratory_ids) != len(exploratory):
        raise FormalV2ContractError(
            "exploratory contrast identifiers contain duplicates"
        )

    gates = _mapping(
        experiment.get("scientific_release_gates"),
        "experiment.scientific_release_gates",
    )
    if not (
        gates.get("method_model") == "a5_vimd_full"
        and gates.get("primary_reference_model")
        == EXPECTED_REFERENCE_MODEL
        and tuple(gates.get("required_nonoracle_baselines", ()))
        == (
            "a0_backbone",
            "mcldnn_reimplementation",
            "iqformer_inspired",
            EXPECTED_REFERENCE_MODEL,
        )
        and gates.get("confirmatory_family_all_simultaneous_ci95_low_above")
        == 0.0
    ):
        raise FormalV2ContractError(
            "scientific-release model family or threshold drifted"
        )
    ood = _mapping(gates.get("ood_axis_calibration"), "ood_axis_calibration")
    if tuple(ood.get("axes", ())) != EXPECTED_OOD_AXES:
        raise FormalV2ContractError("OOD calibration axes drifted")
    if ood != {
        "id_regime": "id_test",
        "baseline_model": "a0_backbone",
        "candidate_model": "a5_vimd_full",
        "axes": list(EXPECTED_OOD_AXES),
        "opportunity_metric": (
            "a0_id_macro_f1_minus_a0_axis_macro_f1"
        ),
        "opportunity_floor": 0.005,
        "consequential_axis_rule": "opportunity_ci95_low_gt_floor",
        "recovery_fraction": 0.25,
        "consequential_axis_gate": (
            "ci95_low_of_gain_minus_recovery_fraction_times_positive_"
            "a0_degradation_gt_0"
        ),
        "nonconsequential_axis_gate": (
            "a5_minus_a0_ci95_low_ge_minus_0.01"
        ),
        "all_axes_must_pass_their_predeclared_branch": True,
        "fixed_three_pp_threshold_used": False,
    }:
        raise FormalV2ContractError(
            "OOD gates are not calibrated to A0 degradation and its CI"
        )
    robustness = _mapping(
        gates.get("receiver_robustness"),
        "receiver_robustness",
    )
    if robustness != {
        "axes": list(EXPECTED_STRESS_AXES),
        "nominal_reference_regime": "hard_interference",
        "calibration_rule": (
            "A0 hard-interference minus A0 stress-axis degradation with "
            "the same consequential/noninferiority CI branches; no id_test "
            "comparison"
        ),
        "all_axes_must_be_reported": True,
        "claim_positive_only_if_axis_gate_passed": True,
    }:
        raise FormalV2ContractError("receiver robustness axes drifted")
    clean = _mapping(
        gates.get("clean_retention"),
        "clean_retention",
    )
    _mapping(
        clean.get("profile_strata"),
        "clean_retention.profile_strata",
    )
    if clean != {
        "regime": "clean_retention",
        "baseline_model": EXPECTED_REFERENCE_MODEL,
        "candidate_model": "a5_vimd_full",
        "profile_strata": {
            "clean_retention_seen_acd": [0, 2, 3],
            "clean_retention_held_be": [1, 4],
        },
        "metric": "macro_f1",
        "point_gain_floor": -0.01,
        "ci95_low_floor": -0.02,
        "all_strata_must_pass": True,
    }:
        raise FormalV2ContractError(
            "clean-retention stratification or noninferiority gates drifted"
        )
    trend = _mapping(
        gates.get("jammer_occupancy_directional_test"),
        "jammer_occupancy_directional_test",
    )
    if trend != {
        "regime": "hard_interference",
        "baseline_model": "a0_backbone",
        "candidate_model": "a5_vimd_full",
        "family_pool": [
            "tone",
            "multitone",
            "chirp",
            "sweep",
            "partial_band",
            "comb",
        ],
        "minimum_family_count": 6,
        "occupancy_definition": EXPECTED_OCCUPANCY_DEFINITION,
        "gain_definition": EXPECTED_GAIN_DEFINITION,
        "test_statistic": (
            "Spearman correlation across jammer-family occupancy and gain"
        ),
        "predicted_direction": "negative",
        "permutation_test": (
            "exact one-sided enumeration of family-gain assignments"
        ),
        "bootstrap_ci": (
            "hierarchical algorithm-seed and within-family source bootstrap"
        ),
        "must_be_reported": True,
        "directional_success_required_for_submission_release": False,
        (
            "positive_mechanism_claim_permitted_only_if_rho_lt_0_and_"
            "one_sided_p_lte_0.05_and_no_strict_occupancy_inversions_and_"
            "ci95_high_lt_0"
        ): True,
    }:
        raise FormalV2ContractError(
            "jammer-occupancy directional mechanism contract is incomplete"
        )
    if set(gates) != {
        "method_model",
        "primary_reference_model",
        "required_nonoracle_baselines",
        "confirmatory_family_all_simultaneous_ci95_low_above",
        "ood_axis_calibration",
        "receiver_robustness",
        "jammer_occupancy_directional_test",
        "clean_retention",
    }:
        raise FormalV2ContractError(
            "scientific-release gate family contains key drift"
        )
    statistics = _mapping(
        experiment.get("statistics"),
        "experiment.statistics",
    )
    master_seed = int(cache.get("master_seed", -1))
    bootstrap_seed = int(statistics.get("bootstrap_seed", -1))
    if master_seed < 0 or bootstrap_seed < 0 or master_seed == bootstrap_seed:
        raise FormalV2ContractError(
            "master seed and bootstrap seed must be distinct nonnegative integers"
        )
    if statistics.get("master_seed_must_differ_from_bootstrap_seed") is not True:
        raise FormalV2ContractError(
            "seed-separation requirement must be explicit"
        )
    if not (
        statistics.get("bootstrap_draws") == 10_000
        and statistics.get("bootstrap_seed") == 20260803
        and statistics.get("headline_inference")
        == (
            "hierarchical bootstrap over algorithm seed and class-stratified "
            "source cluster"
        )
        and statistics.get("ood_calibration_inference")
        == (
            "joint algorithm-seed bootstrap with independent ID/OOD source "
            "resampling and paired A5/A0 OOD resampling"
        )
        and statistics.get("mcnemar_role")
        == "per-seed supplemental diagnostic"
        and statistics.get("validation_excluded") is True
        and set(statistics)
        == {
            "bootstrap_draws",
            "bootstrap_seed",
            "master_seed_must_differ_from_bootstrap_seed",
            "headline_inference",
            "ood_calibration_inference",
            "mcnemar_role",
            "validation_excluded",
        }
    ):
        raise FormalV2ContractError(
            "formal statistical inference constants drifted"
        )
    fixed_design = _mapping(
        prospective.get("fixed_design"),
        "prospective_statistical_design.fixed_design",
    )
    sesoi = _mapping(
        prospective.get("smallest_effect_size_of_interest"),
        "prospective_statistical_design.smallest_effect_size_of_interest",
    )
    if not (
        fixed_design.get("confirmatory_model_count") == len(models)
        and fixed_design.get("confirmatory_algorithm_seed_count")
        == len(seeds)
        and fixed_design.get("required_model_seed_fit_count")
        == len(models) * len(seeds)
        and fixed_design.get("hard_interference_source_cluster_count")
        == cache["expected_split_source_counts"]["hard_interference"]
        and fixed_design.get("confirmatory_family_size") == len(contrasts)
        and fixed_design.get("bootstrap_draws")
        == statistics["bootstrap_draws"]
        and fixed_design.get("learning_curve_fit_count")
        == (
            len(learning["training_source_counts"])
            * len(learning["models"])
            * len(learning["algorithm_seeds"])
        )
        and sesoi.get("contrast_id") in confirmatory_ids
        and (
            sesoi.get(
                "statistical_success_still_requires_simultaneous_ci95_low_gt"
            )
            == family["simultaneous_ci95_low_strictly_greater_than"]
            == gates[
                "confirmatory_family_all_simultaneous_ci95_low_above"
            ]
        )
    ):
        raise FormalV2ContractError(
            "prospective statistical design is not coherent with the frozen "
            "model, seed, source, family, bootstrap, or SESOI contract"
        )
    _exact_mapping_items(
        contract.get("promotion_requirements"),
        EXPECTED_PROMOTION_REQUIREMENTS,
        "promotion_requirements",
    )
    _exact_sequence(
        contract.get("prohibited_imports"),
        (
            "screening metrics",
            "diagnostic metrics",
            "oracle-control metrics",
            "source-mutated metrics",
            "historical v1 metrics reinterpreted as v2",
            "manually typed performance values",
        ),
        "prohibited_imports",
    )
    return dict(contract)


def _validate_v3_8gb_amendment(amendment: Mapping[str, Any]) -> None:
    """Validate the only permitted pre-result 8 GiB execution amendment."""

    required_keys = {
        "schema_version",
        "status",
        "created_date",
        "supersedes_for_future_execution",
        "base_freeze",
        "base_freeze_sha256",
        "eligible_formal_v2_artifacts_observed_before_amendment",
        "amendment_reason",
        "amendment_scope",
        "cuda_execution",
        "effective_experiment",
    }
    if set(amendment) != required_keys:
        raise FormalV2ContractError(
            "v3 8 GiB amendment keys are incomplete or contain drift"
        )
    if not (
        amendment.get("schema_version") == V3_8GB_SCHEMA
        and amendment.get("status") == "preregistered_not_executed"
        and amendment.get("created_date") == "2026-08-02"
        and amendment.get("supersedes_for_future_execution")
        == "tvt_submission/configs/formal_tvt_freeze_v2.json"
        and amendment.get("base_freeze")
        == "tvt_submission/configs/formal_tvt_freeze_v2.json"
        and amendment.get("base_freeze_sha256")
        == EXPECTED_DEFAULT_FREEZE_SHA256
        and amendment.get("eligible_formal_v2_artifacts_observed_before_amendment")
        is False
        and isinstance(amendment.get("amendment_reason"), str)
        and isinstance(amendment.get("amendment_scope"), str)
    ):
        raise FormalV2ContractError(
            "v3 8 GiB amendment provenance or pre-result status drifted"
        )
    if _mapping(amendment.get("cuda_execution"), "cuda_execution") != (
        EXPECTED_V3_8GB_CUDA_EXECUTION
    ):
        raise FormalV2ContractError(
            "v3 8 GiB CUDA memory contract drifted"
        )
    if _mapping(
        amendment.get("effective_experiment"), "effective_experiment"
    ) != {
        "run_id": "tvt_headline_1024_10seed_v3_8gb",
        "expected_run_directory": (
            "artifacts/tvt_headline_1024_10seed_v3_8gb"
        ),
        "learning_curve_protocol_tag": "v3_8gb",
    }:
        raise FormalV2ContractError(
            "v3 8 GiB effective run identity drifted"
        )


def _effective_v3_8gb_contract(
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the auditable resource-only amendment over validated v2."""

    base = load_contract(DEFAULT_FREEZE)
    effective = json.loads(json.dumps(dict(base)))
    execution = _mapping(amendment["cuda_execution"], "cuda_execution")
    run = _mapping(
        amendment["effective_experiment"], "effective_experiment"
    )
    effective["schema_version"] = V3_8GB_SCHEMA
    effective["created_date"] = amendment["created_date"]
    effective["supersedes_for_future_execution"] = amendment[
        "supersedes_for_future_execution"
    ]
    effective["execution_amendment"] = dict(amendment)
    experiment = _mapping(effective["experiment"], "experiment")
    training = _mapping(experiment["training"], "experiment.training")
    training["batch_size"] = int(execution["per_device_batch_size"])
    training["use_amp"] = bool(execution["use_amp"])
    experiment["run_id"] = str(run["run_id"])
    experiment["expected_run_directory"] = str(
        run["expected_run_directory"]
    )
    learning = _mapping(effective["learning_curve"], "learning_curve")
    learning["protocol_tag"] = str(run["learning_curve_protocol_tag"])
    return effective


def _validate_v4_8gb_dual_amendment(amendment: Mapping[str, Any]) -> None:
    """Validate the only permitted prospective 8 GiB dual-fit amendment."""

    required_keys = {
        "schema_version",
        "status",
        "created_date",
        "supersedes_for_future_execution",
        "base_freeze",
        "base_freeze_sha256",
        "eligible_formal_v2_artifacts_observed_before_amendment",
        "amendment_reason",
        "amendment_scope",
        "cuda_execution",
        "effective_experiment",
    }
    if set(amendment) != required_keys:
        raise FormalV2ContractError(
            "v4 8 GiB dual amendment keys are incomplete or contain drift"
        )
    if not (
        amendment.get("schema_version") == V4_8GB_DUAL_SCHEMA
        and amendment.get("status") == "preregistered_not_executed"
        and amendment.get("created_date") == "2026-08-02"
        and amendment.get("supersedes_for_future_execution")
        == "tvt_submission/configs/formal_tvt_freeze_v3_8gb.json"
        and amendment.get("base_freeze")
        == "tvt_submission/configs/formal_tvt_freeze_v2.json"
        and amendment.get("base_freeze_sha256")
        == EXPECTED_DEFAULT_FREEZE_SHA256
        and amendment.get("eligible_formal_v2_artifacts_observed_before_amendment")
        is False
        and isinstance(amendment.get("amendment_reason"), str)
        and isinstance(amendment.get("amendment_scope"), str)
    ):
        raise FormalV2ContractError(
            "v4 8 GiB dual amendment provenance or pre-result status drifted"
        )
    if _mapping(amendment.get("cuda_execution"), "cuda_execution") != (
        EXPECTED_V4_8GB_DUAL_CUDA_EXECUTION
    ):
        raise FormalV2ContractError(
            "v4 8 GiB dual CUDA execution contract drifted"
        )
    if _mapping(
        amendment.get("effective_experiment"), "effective_experiment"
    ) != {
        "run_id": "tvt_headline_1024_10seed_v4_8gb_dual",
        "expected_run_directory": (
            "artifacts/tvt_headline_1024_10seed_v4_8gb_dual"
        ),
        "learning_curve_protocol_tag": "v4_8gb_dual",
    }:
        raise FormalV2ContractError(
            "v4 8 GiB dual effective run identity drifted"
        )


def _effective_v4_8gb_dual_contract(
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the auditable V4 dual-fit profile over validated v2."""

    base = load_contract(DEFAULT_FREEZE)
    effective = json.loads(json.dumps(dict(base)))
    execution = _mapping(amendment["cuda_execution"], "cuda_execution")
    run = _mapping(
        amendment["effective_experiment"], "effective_experiment"
    )
    effective["schema_version"] = V4_8GB_DUAL_SCHEMA
    effective["created_date"] = amendment["created_date"]
    effective["supersedes_for_future_execution"] = amendment[
        "supersedes_for_future_execution"
    ]
    effective["execution_amendment"] = dict(amendment)
    experiment = _mapping(effective["experiment"], "experiment")
    training = _mapping(experiment["training"], "experiment.training")
    training["batch_size"] = int(execution["per_device_batch_size"])
    training["use_amp"] = bool(execution["use_amp"])
    experiment["run_id"] = str(run["run_id"])
    experiment["expected_run_directory"] = str(
        run["expected_run_directory"]
    )
    learning = _mapping(effective["learning_curve"], "learning_curve")
    learning["protocol_tag"] = str(run["learning_curve_protocol_tag"])
    return effective


def load_contract(path: Path = DEFAULT_FREEZE) -> LoadedFormalV2Contract:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FormalV2ContractError(
            f"could not read formal v2 freeze: {error}"
        ) from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FormalV2ContractError(
                    f"non-finite JSON constant is prohibited: {value}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FormalV2ContractError(
            f"could not read formal v2 freeze: {error}"
        ) from error
    amendment = _mapping(payload, "formal freeze")
    resolved = path.resolve()
    if amendment.get("schema_version") == V3_8GB_SCHEMA:
        if resolved != V3_8GB_FREEZE.resolve():
            raise FormalV2ContractError(
                "v3 8 GiB amendment must be loaded from its canonical path"
            )
        if digest != EXPECTED_V3_8GB_FREEZE_SHA256:
            raise FormalV2ContractError(
                "v3 8 GiB amendment SHA-256 drifted: "
                f"{digest} != {EXPECTED_V3_8GB_FREEZE_SHA256}"
            )
        _validate_v3_8gb_amendment(amendment)
        effective = _effective_v3_8gb_contract(amendment)
        justification_path, justification_digest = (
            validate_justification_document(effective)
        )
        return LoadedFormalV2Contract(
            effective,
            freeze_path=resolved,
            freeze_sha256=digest,
            justification_path=justification_path,
            justification_sha256=justification_digest,
        )
    if amendment.get("schema_version") == V4_8GB_DUAL_SCHEMA:
        if resolved != V4_8GB_DUAL_FREEZE.resolve():
            raise FormalV2ContractError(
                "v4 8 GiB dual amendment must be loaded from its canonical path"
            )
        if digest != EXPECTED_V4_8GB_DUAL_FREEZE_SHA256:
            raise FormalV2ContractError(
                "v4 8 GiB dual amendment SHA-256 drifted: "
                f"{digest} != {EXPECTED_V4_8GB_DUAL_FREEZE_SHA256}"
            )
        _validate_v4_8gb_dual_amendment(amendment)
        effective = _effective_v4_8gb_dual_contract(amendment)
        justification_path, justification_digest = (
            validate_justification_document(effective)
        )
        return LoadedFormalV2Contract(
            effective,
            freeze_path=resolved,
            freeze_sha256=digest,
            justification_path=justification_path,
            justification_sha256=justification_digest,
        )
    validated = validate_contract(amendment)
    justification_path, justification_digest = (
        validate_justification_document(validated)
    )
    if (
        resolved == DEFAULT_FREEZE.resolve()
        and digest != EXPECTED_DEFAULT_FREEZE_SHA256
    ):
        raise FormalV2ContractError(
            "default formal v2 freeze SHA-256 drifted: "
            f"{digest} != {EXPECTED_DEFAULT_FREEZE_SHA256}"
        )
    return LoadedFormalV2Contract(
        validated,
        freeze_path=resolved,
        freeze_sha256=digest,
        justification_path=justification_path,
        justification_sha256=justification_digest,
    )
